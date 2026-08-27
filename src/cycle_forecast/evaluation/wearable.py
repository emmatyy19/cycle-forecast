"""Compare daily probabilistic forecasters and summarize calibration."""

from dataclasses import dataclass
from datetime import date
from typing import Final

from cycle_forecast.forecasting.daily import (
    CALIBRATION_WINDOWS,
    DailyForecastEvaluation,
    DailyPeriodDistribution,
    evaluate_daily_distributions,
)

CALIBRATION_BIN_COUNT: Final = 10
"""Fixed equal-width bins used for development calibration diagnostics."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DailyForecastCandidate:
    """Pair one forecaster's distributions with exact shared outcomes."""

    label: str
    version: str
    forecasts: tuple[DailyPeriodDistribution, ...]
    outcome_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationBin:
    """Summarize predicted and observed probability within one fixed bin."""

    lower_bound: float
    upper_bound: float
    count: int
    mean_predicted_probability: float
    observed_fraction: float


@dataclass(frozen=True, slots=True, kw_only=True)
class DailyCandidateEvaluation:
    """Contain proper scores and planning-window calibration tables."""

    label: str
    version: str
    evaluation: DailyForecastEvaluation
    calibration: dict[int, tuple[CalibrationBin, ...]]


@dataclass(frozen=True, slots=True, kw_only=True)
class DailyModelComparison:
    """Contain shared-window daily candidate evaluations."""

    prediction_dates: tuple[date, ...]
    entries: tuple[DailyCandidateEvaluation, ...]


def _calibration_bins(
    *,
    forecasts: tuple[DailyPeriodDistribution, ...],
    outcome_offsets: tuple[int, ...],
    window: int,
) -> tuple[CalibrationBin, ...]:
    """Group one planning window into deterministic equal-width bins."""
    grouped: list[list[tuple[float, float]]] = [
        [] for _ in range(CALIBRATION_BIN_COUNT)
    ]
    for forecast, outcome in zip(forecasts, outcome_offsets, strict=True):
        predicted = forecast.probability_within(days=window)
        index = min(int(predicted * CALIBRATION_BIN_COUNT), CALIBRATION_BIN_COUNT - 1)
        grouped[index].append((predicted, float(outcome < window)))
    bins: list[CalibrationBin] = []
    for index, values in enumerate(grouped):
        if not values:
            continue
        count = len(values)
        bins.append(
            CalibrationBin(
                lower_bound=index / CALIBRATION_BIN_COUNT,
                upper_bound=(index + 1) / CALIBRATION_BIN_COUNT,
                count=count,
                mean_predicted_probability=(
                    sum(predicted for predicted, _ in values) / count
                ),
                observed_fraction=sum(observed for _, observed in values) / count,
            )
        )
    return tuple(bins)


def compare_daily_forecasters(
    *, candidates: tuple[DailyForecastCandidate, ...]
) -> DailyModelComparison:
    """Evaluate candidates on identical uncensored dates and outcomes.

    Parameters
    ----------
    candidates
        Two or more candidate batches covering the exact same mornings.

    Returns
    -------
    DailyModelComparison
        Proper scores and calibration diagnostics in candidate input order.

    Raises
    ------
    ValueError
        If candidate windows, outcomes, labels, or versions are invalid.
    """
    if len(candidates) < 2:
        raise ValueError("daily comparison requires at least two candidates")
    reference = candidates[0]
    prediction_dates = tuple(
        forecast.prediction_date for forecast in reference.forecasts
    )
    if not prediction_dates or len(set(prediction_dates)) != len(prediction_dates):
        raise ValueError("daily comparison dates must be nonempty and unique")
    entries: list[DailyCandidateEvaluation] = []
    for candidate in candidates:
        if not candidate.label or not candidate.version:
            raise ValueError("daily candidates require labels and versions")
        if (
            tuple(forecast.prediction_date for forecast in candidate.forecasts)
            != prediction_dates
            or candidate.outcome_offsets != reference.outcome_offsets
        ):
            raise ValueError(
                "daily candidates must share prediction dates and outcomes"
            )
        evaluation = evaluate_daily_distributions(
            forecasts=candidate.forecasts,
            outcome_offsets=candidate.outcome_offsets,
        )
        entries.append(
            DailyCandidateEvaluation(
                label=candidate.label,
                version=candidate.version,
                evaluation=evaluation,
                calibration={
                    window: _calibration_bins(
                        forecasts=candidate.forecasts,
                        outcome_offsets=candidate.outcome_offsets,
                        window=window,
                    )
                    for window in CALIBRATION_WINDOWS
                },
            )
        )
    return DailyModelComparison(
        prediction_dates=prediction_dates,
        entries=tuple(entries),
    )
