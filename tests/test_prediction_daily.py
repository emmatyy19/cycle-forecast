"""Tests for today's baseline-first probability forecast."""

from datetime import date
from pathlib import Path

import pytest

from cycle_forecast.prediction_daily import (
    DailyPointEstimateMethod,
    estimate_next_start_from_history,
    predict_daily_from_history,
)
from tests.test_prediction import write_test_model


def test_predict_daily_distribution_from_completed_cycle_starts(tmp_path: Path) -> None:
    """Produce exhaustive probabilities using the newest start as cycle context."""
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "cycle_start_date,period_length_days\n"
        "2025-01-01,5\n"
        "2025-01-30,6\n"
        "2025-03-01,\n",
        encoding="utf-8",
    )

    prediction = predict_daily_from_history(
        history_path=history_path,
        prediction_date=date(2025, 3, 5),
        timezone_name="America/New_York",
    )

    assert prediction.current_cycle_start_date == date(2025, 3, 1)
    assert prediction.cycle_day == 5
    probabilities = (
        *prediction.distribution.daily_probabilities,
        prediction.distribution.after_horizon_probability,
    )
    assert sum(probabilities) == pytest.approx(1.0)
    assert prediction.distribution.probability_within(days=3) <= (
        prediction.distribution.probability_within(days=14)
    )


def test_reject_daily_prediction_before_latest_start(tmp_path: Path) -> None:
    """Do not construct a forecast with an impossible nonpositive cycle day."""
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "cycle_start_date,period_length_days\n2025-01-01,5\n2025-02-01,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot precede"):
        predict_daily_from_history(
            history_path=history_path,
            prediction_date=date(2025, 1, 31),
            timezone_name="America/New_York",
        )


def test_naive_point_estimate_uses_median_when_model_is_absent(tmp_path: Path) -> None:
    """Label the robust history-only fallback and round half-days upward."""
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "cycle_start_date,period_length_days\n"
        "2025-01-01,5\n"
        "2025-01-30,5\n"
        "2025-03-01,\n",
        encoding="utf-8",
    )

    estimate = estimate_next_start_from_history(
        history_path=history_path,
        model_path=tmp_path / "missing-model.json",
    )

    assert estimate.method is DailyPointEstimateMethod.NAIVE_MEDIAN
    assert estimate.predicted_cycle_length_days == 29.5
    assert estimate.predicted_next_cycle_start_date == date(2025, 3, 31)


def test_point_estimate_prefers_packaged_phase_a_model(tmp_path: Path) -> None:
    """Use the selected Phase A model whenever its package is available."""
    history_path = Path("data/synthetic/sample_cycle_history.csv").resolve()
    model_path = tmp_path / "selected-model.json"
    write_test_model(path=model_path, history_path=history_path)

    estimate = estimate_next_start_from_history(
        history_path=history_path,
        model_path=model_path,
    )

    assert estimate.method is DailyPointEstimateMethod.PHASE_A_MODEL
    assert estimate.source_label.startswith("Selected Phase A model")
