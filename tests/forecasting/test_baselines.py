"""Test leakage-safe cycle-history forecasting baselines."""

from datetime import date

from cycle_forecast.data import CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.forecasting import (
    PREVIOUS_CYCLE_BASELINE_NAME,
    PREVIOUS_CYCLE_BASELINE_VERSION,
    forecast_with_previous_cycle,
)


def _record(*, year: int, month: int, day: int) -> CycleHistoryRecord:
    """Create a synthetic cycle-history record.

    Parameters
    ----------
    year
        Cycle-start year.
    month
        Cycle-start month.
    day
        Cycle-start day.

    Returns
    -------
    CycleHistoryRecord
        Synthetic record with a fixed, valid period length.
    """
    return CycleHistoryRecord(
        cycle_start_date=date(year, month, day),
        period_length_days=5,
    )


def test_previous_cycle_baseline_returns_versioned_forecasts() -> None:
    """Forecast each eligible row from its immediately preceding target."""
    dataset = build_cycle_dataset(
        records=(
            _record(year=2024, month=1, day=1),
            _record(year=2024, month=1, day=29),
            _record(year=2024, month=2, day=28),
            _record(year=2024, month=3, day=29),
        )
    )

    result = forecast_with_previous_cycle(dataset=dataset)

    assert result.forecaster_name == PREVIOUS_CYCLE_BASELINE_NAME
    assert result.forecaster_version == PREVIOUS_CYCLE_BASELINE_VERSION
    assert result.dataset_fingerprint == dataset.fingerprint
    assert tuple(
        (
            forecast.cycle_start_date,
            forecast.predicted_cycle_length_days,
            forecast.predicted_next_cycle_start_date,
        )
        for forecast in result.forecasts
    ) == (
        (date(2024, 1, 29), 28, date(2024, 2, 26)),
        (date(2024, 2, 28), 30, date(2024, 3, 29)),
    )


def test_previous_cycle_baseline_requires_one_prior_completed_cycle() -> None:
    """Return no forecasts when fewer than two completed rows exist."""
    no_completed_cycles = build_cycle_dataset(
        records=(_record(year=2024, month=1, day=1),)
    )
    one_completed_cycle = build_cycle_dataset(
        records=(
            _record(year=2024, month=1, day=1),
            _record(year=2024, month=1, day=29),
        )
    )

    assert not forecast_with_previous_cycle(dataset=no_completed_cycles).forecasts
    assert not forecast_with_previous_cycle(dataset=one_completed_cycle).forecasts


def test_previous_cycle_baseline_does_not_use_current_cycle_target() -> None:
    """Keep a forecast unchanged when only its eventual target changes."""
    shared_records = (
        _record(year=2024, month=1, day=1),
        _record(year=2024, month=1, day=29),
    )
    shorter_current_cycle = build_cycle_dataset(
        records=(*shared_records, _record(year=2024, month=2, day=26))
    )
    longer_current_cycle = build_cycle_dataset(
        records=(*shared_records, _record(year=2024, month=3, day=2))
    )

    shorter_forecast = forecast_with_previous_cycle(
        dataset=shorter_current_cycle
    ).forecasts[0]
    longer_forecast = forecast_with_previous_cycle(
        dataset=longer_current_cycle
    ).forecasts[0]

    assert shorter_forecast == longer_forecast
    assert shorter_forecast.predicted_cycle_length_days == 28
