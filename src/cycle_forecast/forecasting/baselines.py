"""Provide leakage-safe cycle-history forecasting baselines."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from math import floor, isfinite
from statistics import mean, median

from cycle_forecast.data import CycleDataset

PREVIOUS_CYCLE_BASELINE_NAME = "previous-cycle"
"""Stable name of the previous-cycle baseline."""

PREVIOUS_CYCLE_BASELINE_VERSION = "previous-cycle-v1"
"""Semantic version of the previous-cycle baseline behavior."""

ROLLING_MEAN_BASELINE_VERSION = "rolling-mean-v1"
"""Semantic version of the rolling-mean baseline behavior."""

ROLLING_MEDIAN_BASELINE_VERSION = "rolling-median-v1"
"""Semantic version of the rolling-median baseline behavior."""

EXPANDING_MEAN_BASELINE_NAME = "expanding-mean"
"""Stable name of the expanding-mean baseline."""

EXPANDING_MEAN_BASELINE_VERSION = "expanding-mean-v1"
"""Semantic version of the expanding-mean baseline behavior."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CycleLengthForecast:
    """Represent one point forecast made at a cycle start.

    Parameters
    ----------
    cycle_start_date
        Prediction cutoff and start of the cycle being forecast.
    predicted_cycle_length_days
        Raw numeric point forecast retained for evaluation.
    operational_cycle_length_days
        Whole-day forecast obtained by rounding the raw value half up.
    predicted_next_cycle_start_date
        Calendar date implied by the operational whole-day forecast.
    """

    cycle_start_date: date
    predicted_cycle_length_days: float
    operational_cycle_length_days: int
    predicted_next_cycle_start_date: date


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastBatch:
    """Contain ordered forecasts and their reproducibility metadata.

    Parameters
    ----------
    forecaster_name
        Stable identifier for the forecasting method and configuration.
    forecaster_version
        Semantic version of the forecasting behavior.
    dataset_fingerprint
        Identity of the exact validated dataset used to make the forecasts.
    forecasts
        Forecasts in chronological prediction-cutoff order.
    """

    forecaster_name: str
    forecaster_version: str
    dataset_fingerprint: str
    forecasts: tuple[CycleLengthForecast, ...]


def round_cycle_length_days(*, value: float) -> int:
    """Round a positive day estimate to the nearest whole day, half up.

    Parameters
    ----------
    value
        Positive raw cycle-length forecast.

    Returns
    -------
    int
        Operational whole-day forecast.

    Raises
    ------
    ValueError
        If ``value`` is not positive.
    """
    if not isfinite(value) or value <= 0:
        message = "predicted cycle length must be positive and finite"
        raise ValueError(message)
    return floor(value + 0.5)


def _last(values: tuple[int, ...]) -> float:
    """Return the last historical value as a float.

    Parameters
    ----------
    values
        Non-empty historical cycle lengths.

    Returns
    -------
    float
        Last cycle length.
    """
    return float(values[-1])


def _mean(values: tuple[int, ...]) -> float:
    """Return the arithmetic mean of historical values.

    Parameters
    ----------
    values
        Non-empty historical cycle lengths.

    Returns
    -------
    float
        Arithmetic mean cycle length.
    """
    return mean(values)


def _median(values: tuple[int, ...]) -> float:
    """Return the median of historical values.

    Parameters
    ----------
    values
        Non-empty historical cycle lengths.

    Returns
    -------
    float
        Median cycle length.
    """
    return median(values)


def _forecast_from_history(
    *,
    dataset: CycleDataset,
    forecaster_name: str,
    forecaster_version: str,
    minimum_history: int,
    window_size: int | None,
    statistic: Callable[[tuple[int, ...]], float],
) -> ForecastBatch:
    """Construct forecasts from historical targets before each cutoff.

    Parameters
    ----------
    dataset
        Immutable completed-cycle rows in chronological order.
    forecaster_name
        Stable identifier for the configured forecasting method.
    forecaster_version
        Semantic version of the forecasting behavior.
    minimum_history
        Required number of prior completed cycles.
    window_size
        Number of most-recent targets to use, or ``None`` for all history.
    statistic
        Function that reduces non-empty historical targets to a point forecast.

    Returns
    -------
    ForecastBatch
        Chronological forecasts and their provenance metadata.
    """
    cycle_lengths = tuple(row.cycle_length_days for row in dataset.rows)
    forecasts: list[CycleLengthForecast] = []
    for position in range(minimum_history, len(dataset.rows)):
        history_start = 0 if window_size is None else position - window_size
        history = cycle_lengths[history_start:position]
        raw_prediction = statistic(history)
        operational_prediction = round_cycle_length_days(value=raw_prediction)
        cycle_start = dataset.rows[position].cycle_start_date
        forecasts.append(
            CycleLengthForecast(
                cycle_start_date=cycle_start,
                predicted_cycle_length_days=raw_prediction,
                operational_cycle_length_days=operational_prediction,
                predicted_next_cycle_start_date=(
                    cycle_start + timedelta(days=operational_prediction)
                ),
            )
        )

    return ForecastBatch(
        forecaster_name=forecaster_name,
        forecaster_version=forecaster_version,
        dataset_fingerprint=dataset.fingerprint,
        forecasts=tuple(forecasts),
    )


def forecast_with_previous_cycle(*, dataset: CycleDataset) -> ForecastBatch:
    """Forecast each cycle using the immediately preceding completed cycle.

    Parameters
    ----------
    dataset
        Immutable completed-cycle rows in chronological order.

    Returns
    -------
    ForecastBatch
        Forecasts beginning with the second completed-cycle row.

    Notes
    -----
    The preceding cycle becomes complete at the current row's start, so the
    baseline respects the prediction cutoff. The first row has no forecast.
    """
    return _forecast_from_history(
        dataset=dataset,
        forecaster_name=PREVIOUS_CYCLE_BASELINE_NAME,
        forecaster_version=PREVIOUS_CYCLE_BASELINE_VERSION,
        minimum_history=1,
        window_size=1,
        statistic=_last,
    )


def forecast_with_rolling_mean(
    *, dataset: CycleDataset, window_size: int
) -> ForecastBatch:
    """Forecast from the mean of a fixed window of preceding cycles.

    Parameters
    ----------
    dataset
        Immutable completed-cycle rows in chronological order.
    window_size
        Positive number of immediately preceding completed cycles to average.

    Returns
    -------
    ForecastBatch
        Forecasts beginning after ``window_size`` completed cycles.

    Raises
    ------
    ValueError
        If ``window_size`` is not positive.
    """
    if window_size < 1:
        message = "window_size must be positive"
        raise ValueError(message)
    return _forecast_from_history(
        dataset=dataset,
        forecaster_name=f"rolling-mean-{window_size}",
        forecaster_version=ROLLING_MEAN_BASELINE_VERSION,
        minimum_history=window_size,
        window_size=window_size,
        statistic=_mean,
    )


def forecast_with_rolling_median(
    *, dataset: CycleDataset, window_size: int
) -> ForecastBatch:
    """Forecast from the median of a fixed window of preceding cycles.

    Parameters
    ----------
    dataset
        Immutable completed-cycle rows in chronological order.
    window_size
        Positive number of immediately preceding completed cycles to summarize.

    Returns
    -------
    ForecastBatch
        Forecasts beginning after ``window_size`` completed cycles.

    Raises
    ------
    ValueError
        If ``window_size`` is not positive.
    """
    if window_size < 1:
        message = "window_size must be positive"
        raise ValueError(message)
    return _forecast_from_history(
        dataset=dataset,
        forecaster_name=f"rolling-median-{window_size}",
        forecaster_version=ROLLING_MEDIAN_BASELINE_VERSION,
        minimum_history=window_size,
        window_size=window_size,
        statistic=_median,
    )


def forecast_with_expanding_mean(*, dataset: CycleDataset) -> ForecastBatch:
    """Forecast from the mean of all preceding completed cycles.

    Parameters
    ----------
    dataset
        Immutable completed-cycle rows in chronological order.

    Returns
    -------
    ForecastBatch
        Forecasts beginning with the second completed-cycle row.
    """
    return _forecast_from_history(
        dataset=dataset,
        forecaster_name=EXPANDING_MEAN_BASELINE_NAME,
        forecaster_version=EXPANDING_MEAN_BASELINE_VERSION,
        minimum_history=1,
        window_size=None,
        statistic=_mean,
    )
