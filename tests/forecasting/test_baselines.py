"""Test leakage-safe cycle-history forecasting baselines."""

from datetime import date, timedelta

import pytest

from cycle_forecast.data import CycleDataset, CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.forecasting import (
    EXPANDING_MEAN_BASELINE_NAME,
    EXPANDING_MEAN_BASELINE_VERSION,
    PREVIOUS_CYCLE_BASELINE_NAME,
    PREVIOUS_CYCLE_BASELINE_VERSION,
    ROLLING_MEAN_BASELINE_VERSION,
    ROLLING_MEDIAN_BASELINE_VERSION,
    forecast_with_expanding_mean,
    forecast_with_previous_cycle,
    forecast_with_rolling_mean,
    forecast_with_rolling_median,
    round_cycle_length_days,
)


def _dataset(*, cycle_lengths: tuple[int, ...]) -> CycleDataset:
    """Build a synthetic dataset with the requested cycle lengths.

    Parameters
    ----------
    cycle_lengths
        Positive cycle lengths to encode as consecutive period starts.

    Returns
    -------
    CycleDataset
        Dataset containing one completed row per requested length.
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


def test_previous_cycle_baseline_returns_versioned_forecasts() -> None:
    """Forecast each eligible row from its immediately preceding target."""
    dataset = _dataset(cycle_lengths=(28, 30, 30))

    result = forecast_with_previous_cycle(dataset=dataset)

    assert result.forecaster_name == PREVIOUS_CYCLE_BASELINE_NAME
    assert result.forecaster_version == PREVIOUS_CYCLE_BASELINE_VERSION
    assert result.dataset_fingerprint == dataset.fingerprint
    assert tuple(
        (
            forecast.cycle_start_date,
            forecast.predicted_cycle_length_days,
            forecast.operational_cycle_length_days,
            forecast.predicted_next_cycle_start_date,
        )
        for forecast in result.forecasts
    ) == (
        (date(2024, 1, 29), 28.0, 28, date(2024, 2, 26)),
        (date(2024, 2, 28), 30.0, 30, date(2024, 3, 29)),
    )


def test_rolling_mean_uses_only_complete_fixed_windows() -> None:
    """Average exactly the configured number of preceding targets."""
    dataset = _dataset(cycle_lengths=(28, 29, 35, 31))

    result = forecast_with_rolling_mean(dataset=dataset, window_size=2)

    assert result.forecaster_name == "rolling-mean-2"
    assert result.forecaster_version == ROLLING_MEAN_BASELINE_VERSION
    assert tuple(
        (
            forecast.predicted_cycle_length_days,
            forecast.operational_cycle_length_days,
        )
        for forecast in result.forecasts
    ) == ((28.5, 29), (32.0, 32))
    assert result.forecasts[0].predicted_next_cycle_start_date == (
        result.forecasts[0].cycle_start_date + timedelta(days=29)
    )


def test_operational_cycle_length_rounds_half_up() -> None:
    """Preserve the explicit half-up policy at fractional boundaries."""
    assert round_cycle_length_days(value=28.49) == 28
    assert round_cycle_length_days(value=28.5) == 29


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_operational_cycle_length_rejects_invalid_values(value: float) -> None:
    """Reject raw predictions that cannot represent a cycle length.

    Parameters
    ----------
    value
        Invalid raw point prediction.
    """
    with pytest.raises(
        ValueError,
        match="predicted cycle length must be positive and finite",
    ):
        round_cycle_length_days(value=value)


def test_rolling_median_is_robust_to_an_extreme_history_value() -> None:
    """Use the middle window value instead of its arithmetic mean."""
    dataset = _dataset(cycle_lengths=(20, 20, 50, 30))

    result = forecast_with_rolling_median(dataset=dataset, window_size=3)

    assert result.forecaster_name == "rolling-median-3"
    assert result.forecaster_version == ROLLING_MEDIAN_BASELINE_VERSION
    assert len(result.forecasts) == 1
    assert result.forecasts[0].predicted_cycle_length_days == 20.0
    assert result.forecasts[0].operational_cycle_length_days == 20


def test_expanding_mean_uses_all_history_before_each_cutoff() -> None:
    """Grow the mean history without including the current target."""
    dataset = _dataset(cycle_lengths=(20, 30, 40, 50))

    result = forecast_with_expanding_mean(dataset=dataset)

    assert result.forecaster_name == EXPANDING_MEAN_BASELINE_NAME
    assert result.forecaster_version == EXPANDING_MEAN_BASELINE_VERSION
    assert tuple(
        forecast.predicted_cycle_length_days for forecast in result.forecasts
    ) == (20.0, 25.0, 30.0)


def test_baselines_require_complete_prior_history() -> None:
    """Return no forecasts until each configured history is available."""
    one_completed_cycle = _dataset(cycle_lengths=(28,))
    three_completed_cycles = _dataset(cycle_lengths=(28, 29, 30))

    assert not forecast_with_previous_cycle(dataset=one_completed_cycle).forecasts
    assert not forecast_with_expanding_mean(dataset=one_completed_cycle).forecasts
    assert not forecast_with_rolling_mean(
        dataset=three_completed_cycles,
        window_size=3,
    ).forecasts
    assert not forecast_with_rolling_median(
        dataset=three_completed_cycles,
        window_size=3,
    ).forecasts


def test_baselines_do_not_use_current_cycle_target() -> None:
    """Keep latest forecasts unchanged when only their targets change."""
    shorter_current_cycle = _dataset(cycle_lengths=(28, 30, 25))
    longer_current_cycle = _dataset(cycle_lengths=(28, 30, 40))

    forecast_pairs = (
        (
            forecast_with_previous_cycle(dataset=shorter_current_cycle),
            forecast_with_previous_cycle(dataset=longer_current_cycle),
        ),
        (
            forecast_with_expanding_mean(dataset=shorter_current_cycle),
            forecast_with_expanding_mean(dataset=longer_current_cycle),
        ),
        (
            forecast_with_rolling_mean(dataset=shorter_current_cycle, window_size=2),
            forecast_with_rolling_mean(dataset=longer_current_cycle, window_size=2),
        ),
        (
            forecast_with_rolling_median(dataset=shorter_current_cycle, window_size=2),
            forecast_with_rolling_median(dataset=longer_current_cycle, window_size=2),
        ),
    )

    for shorter_batch, longer_batch in forecast_pairs:
        assert shorter_batch.forecasts[-1] == longer_batch.forecasts[-1]


def test_rolling_baselines_reject_nonpositive_windows() -> None:
    """Reject rolling configurations that cannot contain history."""
    dataset = _dataset(cycle_lengths=(28, 30))

    with pytest.raises(ValueError, match="window_size must be positive"):
        forecast_with_rolling_mean(dataset=dataset, window_size=0)
    with pytest.raises(ValueError, match="window_size must be positive"):
        forecast_with_rolling_median(dataset=dataset, window_size=0)
