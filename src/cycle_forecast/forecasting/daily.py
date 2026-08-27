"""Represent and evaluate exhaustive daily period-start distributions."""

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite, log
from typing import Final

DAILY_FORECAST_HORIZON_DAYS: Final = 15
"""Count of explicit event dates: today through prediction day plus 14."""

DAILY_DISTRIBUTION_VERSION: Final = "daily-period-distribution-v1"
"""Semantic version of daily outcome and inclusive-window behavior."""

CALIBRATION_WINDOWS: Final = (1, 3, 7, 14)
"""Planning windows evaluated from the exhaustive distribution."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DailyPeriodDistribution:
    """Store probabilities for 15 dates and the exhaustive later outcome."""

    prediction_date: date
    prediction_cutoff: datetime
    daily_probabilities: tuple[float, ...]
    after_horizon_probability: float

    def __post_init__(self) -> None:
        """Validate probability shape, finiteness, and total mass."""
        if self.prediction_cutoff.tzinfo is None:
            raise ValueError("prediction_cutoff must be timezone-aware")
        if len(self.daily_probabilities) != DAILY_FORECAST_HORIZON_DAYS:
            raise ValueError("daily_probabilities must contain 15 values")
        values = (*self.daily_probabilities, self.after_horizon_probability)
        if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in values):
            raise ValueError("distribution probabilities must be finite and in [0, 1]")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("distribution probabilities must sum to one")

    def probability_within(self, *, days: int) -> float:
        """Return probability of a start in the inclusive planning window."""
        if days not in CALIBRATION_WINDOWS:
            raise ValueError("days must be one of 1, 3, 7, or 14")
        return sum(self.daily_probabilities[:days])


@dataclass(frozen=True, slots=True, kw_only=True)
class DailyForecastEvaluation:
    """Contain proper scores for a shared set of uncensored daily forecasts."""

    count: int
    logarithmic_loss: float
    multiclass_brier_score: float
    window_brier_scores: dict[int, float]


def distribution_from_hazards(
    *, prediction_date: date, prediction_cutoff: datetime, hazards: tuple[float, ...]
) -> DailyPeriodDistribution:
    """Convert sequential conditional hazards to an exhaustive distribution."""
    if len(hazards) != DAILY_FORECAST_HORIZON_DAYS:
        raise ValueError("hazards must contain 15 values")
    if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in hazards):
        raise ValueError("hazards must be finite and in [0, 1]")
    survival = 1.0
    probabilities: list[float] = []
    for hazard in hazards:
        probabilities.append(survival * hazard)
        survival *= 1.0 - hazard
    return DailyPeriodDistribution(
        prediction_date=prediction_date,
        prediction_cutoff=prediction_cutoff,
        daily_probabilities=tuple(probabilities),
        after_horizon_probability=survival,
    )


def evaluate_daily_distributions(
    *, forecasts: tuple[DailyPeriodDistribution, ...], outcome_offsets: tuple[int, ...]
) -> DailyForecastEvaluation:
    """Calculate proper multiclass and inclusive-window probability scores."""
    if not forecasts or len(forecasts) != len(outcome_offsets):
        raise ValueError("forecasts and outcomes must have equal nonzero length")
    if any(offset < 0 for offset in outcome_offsets):
        raise ValueError("outcome offsets must be nonnegative")
    log_losses: list[float] = []
    brier_scores: list[float] = []
    window_scores: dict[int, list[float]] = {
        window: [] for window in CALIBRATION_WINDOWS
    }
    for forecast, offset in zip(forecasts, outcome_offsets, strict=True):
        probabilities = (
            *forecast.daily_probabilities,
            forecast.after_horizon_probability,
        )
        outcome_index = min(offset, DAILY_FORECAST_HORIZON_DAYS)
        probability = max(probabilities[outcome_index], 1e-15)
        log_losses.append(-log(probability))
        brier_scores.append(
            sum(
                (value - float(index == outcome_index)) ** 2
                for index, value in enumerate(probabilities)
            )
        )
        for window in CALIBRATION_WINDOWS:
            predicted = forecast.probability_within(days=window)
            observed = float(offset < window)
            window_scores[window].append((predicted - observed) ** 2)
    count = len(forecasts)
    return DailyForecastEvaluation(
        count=count,
        logarithmic_loss=sum(log_losses) / count,
        multiclass_brier_score=sum(brier_scores) / count,
        window_brier_scores={
            window: sum(scores) / count for window, scores in window_scores.items()
        },
    )
