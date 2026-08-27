"""Test calibrated discrete-time wearable survival modeling."""

from datetime import UTC, date, datetime, timedelta

from cycle_forecast.data.wearable_alignment import AlignedDailyObservation
from cycle_forecast.features.wearable import WEARABLE_FEATURE_NAMES, WearableFeatureRow
from cycle_forecast.models.discrete_survival import (
    DiscreteSurvivalConfig,
    fit_discrete_survival_model,
    predict_with_discrete_survival_model,
)


def _row(*, index: int, outcome: int | None) -> WearableFeatureRow:
    """Create one invented labeled morning with a learnable temporal signal."""
    prediction_date = date(2025, 1, 1) + timedelta(days=index)
    signal = float(index % 10)
    return WearableFeatureRow(
        aligned=AlignedDailyObservation(
            prediction_date=prediction_date,
            prediction_cutoff=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=index),
            cycle_start_date=date(2024, 12, 20),
            cycle_day=index + 13,
            oura=None,
        ),
        feature_names=WEARABLE_FEATURE_NAMES,
        values=(
            float(index + 13),
            70.0 + signal,
            signal / 10,
            75.0,
            40.0,
            25_000.0,
            0,
            0,
            0,
            0,
            0,
        ),
        outcome_offset_days=outcome,
    )


def test_fits_calibrates_and_predicts_an_exhaustive_later_distribution() -> None:
    """Keep fitting, calibration, and prediction in chronological blocks."""
    rows = tuple(_row(index=index, outcome=index % 16) for index in range(45))
    model = fit_discrete_survival_model(
        training_rows=rows[:30],
        calibration_rows=rows[30:40],
        configuration=DiscreteSurvivalConfig(
            minimum_training_rows=30,
            minimum_calibration_rows=10,
        ),
    )

    forecast = predict_with_discrete_survival_model(
        model=model,
        row=_row(index=44, outcome=None),
    )

    assert (
        abs(
            sum((*forecast.daily_probabilities, forecast.after_horizon_probability))
            - 1.0
        )
        < 1e-9
    )
    assert all(0.0 <= value <= 1.0 for value in forecast.daily_probabilities)
