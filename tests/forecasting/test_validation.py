"""Test common-window walk-forward forecast evaluation."""

from datetime import date, timedelta

import pytest

from cycle_forecast.data import CycleDataset, CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.forecasting import (
    ForecastBatch,
    WalkForwardContext,
    evaluate_walk_forward,
    forecast_with_expanding_mean,
    forecast_with_previous_cycle,
    forecast_with_rolling_mean,
    generate_walk_forward_forecasts,
)


def _dataset(*, cycle_lengths: tuple[int, ...]) -> CycleDataset:
    """Build a synthetic dataset with specified completed-cycle lengths.

    Parameters
    ----------
    cycle_lengths
        Positive cycle lengths to encode as period starts.

    Returns
    -------
    CycleDataset
        Synthetic completed-cycle dataset.
    """
    starts = [date(2024, 1, 1)]
    for cycle_length in cycle_lengths:
        starts.append(starts[-1] + timedelta(days=cycle_length))
    return build_cycle_dataset(
        records=tuple(
            CycleHistoryRecord(
                cycle_start_date=cycle_start,
                period_length_days=5,
            )
            for cycle_start in starts
        )
    )


def test_walk_forward_evaluation_uses_identical_cycle_cutoffs() -> None:
    """Compare every forecaster only where all can make predictions."""
    dataset = _dataset(cycle_lengths=(28, 30, 32, 34, 36))
    previous = forecast_with_previous_cycle(dataset=dataset)
    rolling_mean = forecast_with_rolling_mean(dataset=dataset, window_size=3)

    result = evaluate_walk_forward(
        dataset=dataset,
        forecast_batches=(previous, rolling_mean),
    )

    expected_starts = (
        dataset.rows[3].cycle_start_date,
        dataset.rows[4].cycle_start_date,
    )
    assert result.dataset_fingerprint == dataset.fingerprint
    assert result.cycle_start_dates == expected_starts
    assert tuple(evaluation.forecaster_name for evaluation in result.evaluations) == (
        "previous-cycle",
        "rolling-mean-3",
    )
    assert all(
        tuple(error.cycle_start_date for error in evaluation.errors) == expected_starts
        for evaluation in result.evaluations
    )
    assert result.evaluations[0].metrics.forecast_count == 2
    assert result.evaluations[0].metrics.mean_absolute_error_days == 2.0
    assert result.evaluations[1].metrics.forecast_count == 2
    assert result.evaluations[1].metrics.mean_absolute_error_days == 4.0


def test_walk_forward_generation_exposes_only_completed_history() -> None:
    """Keep each current target outside the predictor's supplied context."""
    dataset = _dataset(cycle_lengths=(28, 30, 32, 34))
    observed_contexts: list[WalkForwardContext] = []

    def predict(context: WalkForwardContext) -> float:
        """Record the cutoff-safe context and predict its latest target."""
        observed_contexts.append(context)
        return float(context.history[-1].cycle_length_days)

    forecasts = generate_walk_forward_forecasts(
        dataset=dataset,
        forecaster_name="context-test",
        forecaster_version="context-test-v1",
        minimum_history=1,
        predictor=predict,
    )

    assert len(forecasts.forecasts) == 3
    assert tuple(len(context.history) for context in observed_contexts) == (1, 2, 3)
    assert all(
        context.history[-1].next_cycle_start_date == context.cycle_start_date
        for context in observed_contexts
    )
    assert tuple(
        forecast.predicted_cycle_length_days for forecast in forecasts.forecasts
    ) == (28.0, 30.0, 32.0)


def test_walk_forward_generation_requires_history() -> None:
    """Reject a generator configuration that could expose an empty context."""
    dataset = _dataset(cycle_lengths=(28, 30))

    def predict(context: WalkForwardContext) -> float:
        """Return an arbitrary valid prediction for the invalid setup."""
        return float(context.history[-1].cycle_length_days)

    with pytest.raises(ValueError, match="minimum_history must be positive"):
        generate_walk_forward_forecasts(
            dataset=dataset,
            forecaster_name="invalid",
            forecaster_version="invalid-v1",
            minimum_history=0,
            predictor=predict,
        )


def test_walk_forward_evaluation_preserves_forecaster_order() -> None:
    """Return evaluations in the explicit caller-supplied comparison order."""
    dataset = _dataset(cycle_lengths=(28, 30, 32))
    expanding = forecast_with_expanding_mean(dataset=dataset)
    previous = forecast_with_previous_cycle(dataset=dataset)

    result = evaluate_walk_forward(
        dataset=dataset,
        forecast_batches=(expanding, previous),
    )

    assert tuple(evaluation.forecaster_name for evaluation in result.evaluations) == (
        "expanding-mean",
        "previous-cycle",
    )


def test_walk_forward_evaluation_allows_empty_common_window() -> None:
    """Represent insufficient shared history without inventing zero metrics."""
    dataset = _dataset(cycle_lengths=(28, 30))
    previous = forecast_with_previous_cycle(dataset=dataset)
    rolling_mean = forecast_with_rolling_mean(dataset=dataset, window_size=2)

    result = evaluate_walk_forward(
        dataset=dataset,
        forecast_batches=(previous, rolling_mean),
    )

    assert not result.cycle_start_dates
    assert all(not evaluation.errors for evaluation in result.evaluations)
    assert all(
        evaluation.metrics.mean_absolute_error_days is None
        for evaluation in result.evaluations
    )


def test_walk_forward_evaluation_requires_a_forecaster() -> None:
    """Reject a comparison with no forecasting methods."""
    dataset = _dataset(cycle_lengths=(28, 30))

    with pytest.raises(ValueError, match="at least one forecast batch"):
        evaluate_walk_forward(dataset=dataset, forecast_batches=())


def test_walk_forward_evaluation_rejects_duplicate_forecaster_identity() -> None:
    """Reject ambiguous metric attribution for duplicate configurations."""
    dataset = _dataset(cycle_lengths=(28, 30, 32))
    previous = forecast_with_previous_cycle(dataset=dataset)

    with pytest.raises(ValueError, match="identities must be unique"):
        evaluate_walk_forward(
            dataset=dataset,
            forecast_batches=(previous, previous),
        )


def test_walk_forward_evaluation_validates_before_intersection() -> None:
    """Reject invalid provenance even when another batch has no forecasts."""
    dataset = _dataset(cycle_lengths=(28, 30))
    empty = forecast_with_rolling_mean(dataset=dataset, window_size=2)
    invalid = ForecastBatch(
        forecaster_name="invalid",
        forecaster_version="invalid-v1",
        dataset_fingerprint="sha256:not-the-dataset",
        forecasts=(),
    )

    with pytest.raises(ValueError, match="fingerprint does not match"):
        evaluate_walk_forward(
            dataset=dataset,
            forecast_batches=(empty, invalid),
        )
