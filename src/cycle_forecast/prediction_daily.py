"""Generate today's baseline-first probability forecast from local history."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum, auto
from itertools import pairwise
from math import floor
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cycle_forecast.data.cycle_history import load_cycle_history
from cycle_forecast.forecasting.daily import DailyPeriodDistribution
from cycle_forecast.forecasting.wearable_baselines import (
    EMPIRICAL_HAZARD_BASELINE_VERSION,
    forecast_with_empirical_cycle_hazard_context,
)
from cycle_forecast.prediction import predict_from_local_files


class DailyPointEstimateMethod(StrEnum):
    """Identify the source of the longer-range next-start estimate."""

    PHASE_A_MODEL = "phase-a-model"
    NAIVE_MEDIAN = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class DailyPointEstimate:
    """Contain one explicitly sourced longer-range next-start estimate."""

    predicted_next_cycle_start_date: date
    predicted_cycle_length_days: float
    method: DailyPointEstimateMethod
    source_label: str


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryDailyPrediction:
    """Contain today's cycle context and exhaustive history-baseline forecast."""

    prediction_date: date
    current_cycle_start_date: date
    cycle_day: int
    model_version: str
    distribution: DailyPeriodDistribution


def estimate_next_start_from_history(
    *, history_path: Path, model_path: Path
) -> DailyPointEstimate:
    """Use the selected Phase A package or an explicit naive median fallback.

    Parameters
    ----------
    history_path
        Validated private cycle-history CSV.
    model_path
        Preferred packaged Phase A model path.

    Returns
    -------
    DailyPointEstimate
        Next-start estimate with an explicit method and source label.
    """
    if model_path.is_file():
        prediction = predict_from_local_files(
            model_path=model_path,
            history_path=history_path,
        )
        return DailyPointEstimate(
            predicted_next_cycle_start_date=(
                prediction.predicted_next_cycle_start_date
            ),
            predicted_cycle_length_days=prediction.predicted_cycle_length_days,
            method=DailyPointEstimateMethod.PHASE_A_MODEL,
            source_label=f"Selected Phase A model · {prediction.model_version}",
        )

    records = load_cycle_history(path=history_path)
    completed_lengths = tuple(
        (following.cycle_start_date - current.cycle_start_date).days
        for current, following in pairwise(records)
    )
    if not completed_lengths:
        raise ValueError("next-start estimate requires a completed cycle")
    median_length = float(median(completed_lengths))
    operational_length = floor(median_length + 0.5)
    return DailyPointEstimate(
        predicted_next_cycle_start_date=(
            records[-1].cycle_start_date + timedelta(days=operational_length)
        ),
        predicted_cycle_length_days=median_length,
        method=DailyPointEstimateMethod.NAIVE_MEDIAN,
        source_label="Naive median of completed cycle lengths",
    )


def predict_daily_from_history(
    *,
    history_path: Path,
    prediction_date: date,
    timezone_name: str,
    prediction_hour: int = 9,
) -> HistoryDailyPrediction:
    """Forecast today's period-start probabilities from completed cycles.

    Parameters
    ----------
    history_path
        Validated private cycle-history CSV.
    prediction_date
        Local date on which the forecast is generated.
    timezone_name
        IANA timezone used for the prediction cutoff.
    prediction_hour
        Local cutoff hour from zero through 23.

    Returns
    -------
    HistoryDailyPrediction
        Current cycle context and exhaustive daily probability distribution.

    Raises
    ------
    ValueError
        If history or prediction context is insufficient or invalid.
    """
    if not 0 <= prediction_hour <= 23:
        raise ValueError("prediction_hour must be between 0 and 23")
    try:
        timezone = ZoneInfo(timezone_name)
    except (OSError, ValueError, ZoneInfoNotFoundError) as error:
        raise ValueError("timezone must be an IANA timezone") from error
    records = load_cycle_history(path=history_path)
    if len(records) < 2:
        raise ValueError("daily prediction requires at least two period starts")
    current_start = records[-1].cycle_start_date
    cycle_day = (prediction_date - current_start).days + 1
    if cycle_day < 1:
        raise ValueError("prediction date cannot precede the newest period start")
    completed_lengths = tuple(
        (following.cycle_start_date - current.cycle_start_date).days
        for current, following in pairwise(records)
    )
    cutoff = datetime.combine(
        prediction_date,
        time(hour=prediction_hour),
        tzinfo=timezone,
    )
    distribution = forecast_with_empirical_cycle_hazard_context(
        cycle_day=cycle_day,
        prediction_date=prediction_date,
        prediction_cutoff=cutoff,
        completed_cycle_lengths=completed_lengths,
    )
    return HistoryDailyPrediction(
        prediction_date=prediction_date,
        current_cycle_start_date=current_start,
        cycle_day=cycle_day,
        model_version=EMPIRICAL_HAZARD_BASELINE_VERSION,
        distribution=distribution,
    )
