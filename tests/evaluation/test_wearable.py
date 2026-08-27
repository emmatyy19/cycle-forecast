"""Test shared-window daily probabilistic comparison and calibration."""

from datetime import UTC, date, datetime, timedelta

import pytest

from cycle_forecast.evaluation.wearable import (
    DailyForecastCandidate,
    compare_daily_forecasters,
)
from cycle_forecast.forecasting.daily import DailyPeriodDistribution


def _forecast(*, index: int, today_probability: float) -> DailyPeriodDistribution:
    """Create one invented exhaustive daily distribution."""
    remaining = 1.0 - today_probability
    return DailyPeriodDistribution(
        prediction_date=date(2025, 1, 1) + timedelta(days=index),
        prediction_cutoff=datetime(2025, 1, 1, 8, tzinfo=UTC) + timedelta(days=index),
        daily_probabilities=(today_probability, *((remaining / 28,) * 14)),
        after_horizon_probability=remaining / 2,
    )


def test_compares_shared_candidates_and_builds_calibration_tables() -> None:
    """Report proper scores and populated fixed-bin planning diagnostics."""
    dates = tuple(_forecast(index=index, today_probability=0.1) for index in range(4))
    sharper = tuple(_forecast(index=index, today_probability=0.4) for index in range(4))
    outcomes = (0, 1, 8, 20)

    comparison = compare_daily_forecasters(
        candidates=(
            DailyForecastCandidate(
                label="Baseline",
                version="baseline-v1",
                forecasts=dates,
                outcome_offsets=outcomes,
            ),
            DailyForecastCandidate(
                label="Survival",
                version="survival-v1",
                forecasts=sharper,
                outcome_offsets=outcomes,
            ),
        )
    )

    assert len(comparison.entries) == 2
    assert comparison.entries[0].evaluation.count == 4
    assert set(comparison.entries[0].calibration) == {1, 3, 7, 14}
    assert sum(bin_.count for bin_ in comparison.entries[0].calibration[1]) == 4


def test_rejects_mismatched_candidate_windows() -> None:
    """Prevent apparently comparable scores from using different dates."""
    first = _forecast(index=0, today_probability=0.1)
    second = _forecast(index=1, today_probability=0.1)

    with pytest.raises(ValueError, match="share prediction dates"):
        compare_daily_forecasters(
            candidates=(
                DailyForecastCandidate(
                    label="A", version="a-v1", forecasts=(first,), outcome_offsets=(0,)
                ),
                DailyForecastCandidate(
                    label="B", version="b-v1", forecasts=(second,), outcome_offsets=(0,)
                ),
            )
        )
