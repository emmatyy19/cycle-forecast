"""Test alignment and metrics for cycle-length forecasts."""

from datetime import date, timedelta
from math import sqrt

import pytest

from cycle_forecast.data import CycleDataset, CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.forecasting import (
    CycleLengthForecast,
    ForecastBatch,
    evaluate_forecasts,
    forecast_with_rolling_mean,
    round_cycle_length_days,
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


def _forecast(
    *, cycle_start_date: date, predicted_cycle_length_days: float
) -> CycleLengthForecast:
    """Construct a consistent synthetic cycle-length forecast.

    Parameters
    ----------
    cycle_start_date
        Prediction cutoff date.
    predicted_cycle_length_days
        Raw point prediction.

    Returns
    -------
    CycleLengthForecast
        Forecast with a matching operational date.
    """
    operational_days = round_cycle_length_days(value=predicted_cycle_length_days)
    return CycleLengthForecast(
        cycle_start_date=cycle_start_date,
        predicted_cycle_length_days=predicted_cycle_length_days,
        operational_cycle_length_days=operational_days,
        predicted_next_cycle_start_date=(
            cycle_start_date + timedelta(days=operational_days)
        ),
    )


def _batch(
    *, dataset: CycleDataset, forecasts: tuple[CycleLengthForecast, ...]
) -> ForecastBatch:
    """Wrap synthetic forecasts with matching provenance.

    Parameters
    ----------
    dataset
        Dataset supplying the fingerprint.
    forecasts
        Synthetic forecasts to wrap.

    Returns
    -------
    ForecastBatch
        Versioned synthetic forecast batch.
    """
    return ForecastBatch(
        forecaster_name="test-forecaster",
        forecaster_version="test-forecaster-v1",
        dataset_fingerprint=dataset.fingerprint,
        forecasts=forecasts,
    )


def test_evaluate_forecasts_calculates_day_metrics() -> None:
    """Calculate signed, absolute, squared, and tolerance metrics."""
    dataset = _dataset(cycle_lengths=(30, 30, 30, 30))
    forecasts = tuple(
        _forecast(
            cycle_start_date=row.cycle_start_date,
            predicted_cycle_length_days=prediction,
        )
        for row, prediction in zip(
            dataset.rows,
            (29.0, 32.0, 27.0, 36.0),
            strict=True,
        )
    )

    evaluation = evaluate_forecasts(
        dataset=dataset,
        forecast_batch=_batch(dataset=dataset, forecasts=forecasts),
    )

    assert evaluation.forecaster_name == "test-forecaster"
    assert evaluation.forecaster_version == "test-forecaster-v1"
    assert evaluation.dataset_fingerprint == dataset.fingerprint
    assert tuple(error.error_days for error in evaluation.errors) == (
        -1.0,
        2.0,
        -3.0,
        6.0,
    )
    assert tuple(error.absolute_error_days for error in evaluation.errors) == (
        1.0,
        2.0,
        3.0,
        6.0,
    )
    assert evaluation.metrics.forecast_count == 4
    assert evaluation.metrics.mean_error_days == 1.0
    assert evaluation.metrics.mean_absolute_error_days == 3.0
    assert evaluation.metrics.median_absolute_error_days == 2.5
    assert evaluation.metrics.root_mean_squared_error_days == pytest.approx(sqrt(12.5))
    assert evaluation.metrics.within_1_day == 0.25
    assert evaluation.metrics.within_2_days == 0.5
    assert evaluation.metrics.within_3_days == 0.75
    assert evaluation.metrics.within_5_days == 0.75


def test_evaluate_forecasts_uses_raw_prediction() -> None:
    """Evaluate fractional raw output instead of its operational rounding."""
    dataset = _dataset(cycle_lengths=(28, 29, 32))
    forecasts = forecast_with_rolling_mean(dataset=dataset, window_size=2)

    evaluation = evaluate_forecasts(dataset=dataset, forecast_batch=forecasts)

    assert len(evaluation.errors) == 1
    assert evaluation.errors[0].predicted_cycle_length_days == 28.5
    assert forecasts.forecasts[0].operational_cycle_length_days == 29
    assert evaluation.errors[0].error_days == -3.5


def test_evaluate_forecasts_returns_undefined_metrics_when_empty() -> None:
    """Represent a valid insufficient-history evaluation without failure."""
    dataset = _dataset(cycle_lengths=(28,))
    evaluation = evaluate_forecasts(
        dataset=dataset,
        forecast_batch=_batch(dataset=dataset, forecasts=()),
    )

    assert not evaluation.errors
    assert evaluation.metrics.forecast_count == 0
    assert evaluation.metrics.mean_error_days is None
    assert evaluation.metrics.mean_absolute_error_days is None
    assert evaluation.metrics.median_absolute_error_days is None
    assert evaluation.metrics.root_mean_squared_error_days is None
    assert evaluation.metrics.within_1_day is None
    assert evaluation.metrics.within_2_days is None
    assert evaluation.metrics.within_3_days is None
    assert evaluation.metrics.within_5_days is None


def test_evaluate_forecasts_rejects_mismatched_fingerprint() -> None:
    """Reject forecasts that cannot belong to the supplied actuals."""
    dataset = _dataset(cycle_lengths=(28, 30))
    mismatched_batch = ForecastBatch(
        forecaster_name="test-forecaster",
        forecaster_version="test-forecaster-v1",
        dataset_fingerprint="sha256:not-the-dataset",
        forecasts=(),
    )

    with pytest.raises(ValueError, match="fingerprint does not match"):
        evaluate_forecasts(dataset=dataset, forecast_batch=mismatched_batch)


def test_evaluate_forecasts_rejects_unknown_cycle_start() -> None:
    """Reject a forecast that cannot be aligned to an actual target."""
    dataset = _dataset(cycle_lengths=(28, 30))
    unknown_start = date(2030, 1, 1)
    forecasts = (
        _forecast(cycle_start_date=unknown_start, predicted_cycle_length_days=30),
    )

    with pytest.raises(ValueError, match="has no actual dataset row"):
        evaluate_forecasts(
            dataset=dataset,
            forecast_batch=_batch(dataset=dataset, forecasts=forecasts),
        )


def test_evaluate_forecasts_rejects_duplicate_or_nonchronological_starts() -> None:
    """Reject ambiguous or reordered forecast-to-actual alignment."""
    dataset = _dataset(cycle_lengths=(28, 30))
    first_start = dataset.rows[0].cycle_start_date
    duplicate_forecast = _forecast(
        cycle_start_date=first_start,
        predicted_cycle_length_days=28,
    )

    with pytest.raises(ValueError, match="strictly chronological and unique"):
        evaluate_forecasts(
            dataset=dataset,
            forecast_batch=_batch(
                dataset=dataset,
                forecasts=(duplicate_forecast, duplicate_forecast),
            ),
        )


@pytest.mark.parametrize("prediction", [0.0, -1.0, float("nan"), float("inf")])
def test_evaluate_forecasts_rejects_invalid_raw_predictions(
    prediction: float,
) -> None:
    """Reject raw predictions that cannot be evaluated meaningfully.

    Parameters
    ----------
    prediction
        Invalid raw cycle-length forecast.
    """
    dataset = _dataset(cycle_lengths=(28,))
    start = dataset.rows[0].cycle_start_date
    invalid_forecast = CycleLengthForecast(
        cycle_start_date=start,
        predicted_cycle_length_days=prediction,
        operational_cycle_length_days=1,
        predicted_next_cycle_start_date=start + timedelta(days=1),
    )

    with pytest.raises(ValueError, match="positive and finite"):
        evaluate_forecasts(
            dataset=dataset,
            forecast_batch=_batch(
                dataset=dataset,
                forecasts=(invalid_forecast,),
            ),
        )
