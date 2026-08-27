"""Test probabilistic daily history and wearable-informed baselines."""

from datetime import UTC, date, datetime, timedelta

from cycle_forecast.data.wearable_alignment import AlignedDailyObservation
from cycle_forecast.features.wearable import WEARABLE_FEATURE_NAMES, WearableFeatureRow
from cycle_forecast.forecasting.wearable_baselines import (
    forecast_with_empirical_cycle_hazard,
    forecast_with_wearable_neighbors,
)


def _row(*, index: int, outcome: int | None, signal: float) -> WearableFeatureRow:
    """Create one invented chronological wearable row."""
    prediction_date = date(2025, 1, 1) + timedelta(days=index)
    return WearableFeatureRow(
        aligned=AlignedDailyObservation(
            prediction_date=prediction_date,
            prediction_cutoff=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=index),
            cycle_start_date=date(2024, 12, 20),
            cycle_day=13 + index,
            oura=None,
        ),
        feature_names=WEARABLE_FEATURE_NAMES,
        values=(float(13 + index), signal, 0.0, 80.0, 40.0, 25_000.0, 0, 1, 0, 0, 0),
        outcome_offset_days=outcome,
    )


def test_empirical_hazard_baseline_returns_exhaustive_probabilities() -> None:
    """Convert smoothed completed-cycle hazards without truncating survival."""
    forecast = forecast_with_empirical_cycle_hazard(
        row=_row(index=0, outcome=None, signal=75.0),
        completed_cycle_lengths=(27, 28, 29, 30),
    )

    assert (
        abs(
            sum((*forecast.daily_probabilities, forecast.after_horizon_probability))
            - 1.0
        )
        < 1e-9
    )


def test_neighbor_baseline_uses_only_earlier_labeled_rows() -> None:
    """Ignore censored and future rows while returning a smoothed distribution."""
    current = _row(index=3, outcome=None, signal=82.0)
    forecast = forecast_with_wearable_neighbors(
        row=current,
        training_rows=(
            _row(index=0, outcome=2, signal=80.0),
            _row(index=1, outcome=None, signal=81.0),
            _row(index=4, outcome=0, signal=82.0),
        ),
        neighbor_count=2,
    )

    assert forecast.daily_probabilities[2] > forecast.daily_probabilities[0]
