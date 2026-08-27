"""Test exhaustive daily distributions and proper probability scores."""

from datetime import UTC, date, datetime

import pytest

from cycle_forecast.forecasting.daily import (
    DailyPeriodDistribution,
    distribution_from_hazards,
    evaluate_daily_distributions,
)


def test_hazards_form_an_exhaustive_distribution_and_inclusive_windows() -> None:
    """Retain survival mass after the explicit 15 prediction dates."""
    forecast = distribution_from_hazards(
        prediction_date=date(2025, 1, 1),
        prediction_cutoff=datetime(2025, 1, 1, 8, tzinfo=UTC),
        hazards=(0.1,) * 15,
    )

    assert sum((*forecast.daily_probabilities, forecast.after_horizon_probability)) == (
        pytest.approx(1.0)
    )
    assert forecast.probability_within(days=3) == pytest.approx(
        sum(forecast.daily_probabilities[:3])
    )


def test_scores_daily_and_after_horizon_outcomes() -> None:
    """Evaluate all 16 mutually exclusive outcomes and planning windows."""
    forecast = DailyPeriodDistribution(
        prediction_date=date(2025, 1, 1),
        prediction_cutoff=datetime(2025, 1, 1, 8, tzinfo=UTC),
        daily_probabilities=(0.05,) * 15,
        after_horizon_probability=0.25,
    )

    evaluation = evaluate_daily_distributions(
        forecasts=(forecast, forecast), outcome_offsets=(0, 20)
    )

    assert evaluation.count == 2
    assert evaluation.logarithmic_loss > 0.0
    assert evaluation.multiclass_brier_score > 0.0
    assert set(evaluation.window_brier_scores) == {1, 3, 7, 14}
