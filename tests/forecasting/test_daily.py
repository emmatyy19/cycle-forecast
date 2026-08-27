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


@pytest.mark.parametrize(
    ("probabilities", "later", "message"),
    (
        ((0.0,) * 14, 1.0, "15 values"),
        ((0.0,) * 14 + (-0.1,), 1.1, "finite and in"),
        ((0.0,) * 15, 0.5, "sum to one"),
    ),
)
def test_rejects_invalid_daily_distributions(
    probabilities: tuple[float, ...], later: float, message: str
) -> None:
    """Reject malformed probability vectors at the domain boundary."""
    with pytest.raises(ValueError, match=message):
        DailyPeriodDistribution(
            prediction_date=date(2025, 1, 1),
            prediction_cutoff=datetime(2025, 1, 1, 8, tzinfo=UTC),
            daily_probabilities=probabilities,
            after_horizon_probability=later,
        )


def test_rejects_invalid_hazards_scores_and_windows() -> None:
    """Exercise public validation for hazards, outcomes, and planning windows."""
    cutoff = datetime(2025, 1, 1, 8, tzinfo=UTC)
    with pytest.raises(ValueError, match="15 values"):
        distribution_from_hazards(
            prediction_date=date(2025, 1, 1), prediction_cutoff=cutoff, hazards=(0.1,)
        )
    with pytest.raises(ValueError, match="finite and in"):
        distribution_from_hazards(
            prediction_date=date(2025, 1, 1),
            prediction_cutoff=cutoff,
            hazards=(1.1,) * 15,
        )
    forecast = distribution_from_hazards(
        prediction_date=date(2025, 1, 1),
        prediction_cutoff=cutoff,
        hazards=(0.1,) * 15,
    )
    with pytest.raises(ValueError, match="one of"):
        forecast.probability_within(days=2)
    with pytest.raises(ValueError, match="equal nonzero"):
        evaluate_daily_distributions(forecasts=(), outcome_offsets=())
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate_daily_distributions(forecasts=(forecast,), outcome_offsets=(-1,))
    with pytest.raises(ValueError, match="timezone-aware"):
        DailyPeriodDistribution(
            prediction_date=date(2025, 1, 1),
            prediction_cutoff=datetime(2025, 1, 1),
            daily_probabilities=(0.0,) * 15,
            after_horizon_probability=1.0,
        )
