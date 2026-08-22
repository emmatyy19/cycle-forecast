"""Provide leakage-safe cycle-history forecasting baselines."""

from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise

from cycle_forecast.data import CycleDataset

PREVIOUS_CYCLE_BASELINE_NAME = "previous-cycle"
"""Stable name of the previous-cycle baseline."""

PREVIOUS_CYCLE_BASELINE_VERSION = "previous-cycle-v1"
"""Semantic version of the previous-cycle baseline behavior."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CycleLengthForecast:
    """Represent one point forecast made at a cycle start.

    Parameters
    ----------
    cycle_start_date
        Prediction cutoff and start of the cycle being forecast.
    predicted_cycle_length_days
        Forecast cycle length in whole days.
    predicted_next_cycle_start_date
        Calendar date implied by the predicted cycle length.
    """

    cycle_start_date: date
    predicted_cycle_length_days: int
    predicted_next_cycle_start_date: date


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastBatch:
    """Contain ordered forecasts and their reproducibility metadata.

    Parameters
    ----------
    forecaster_name
        Stable identifier for the forecasting method.
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


def forecast_with_previous_cycle(*, dataset: CycleDataset) -> ForecastBatch:
    """Forecast each cycle using the immediately preceding completed cycle.

    Parameters
    ----------
    dataset
        Immutable completed-cycle rows in chronological order.

    Returns
    -------
    ForecastBatch
        Forecasts beginning with the second completed-cycle row. No forecast is
        emitted for the first row because no earlier completed cycle is
        available.

    Notes
    -----
    For a row beginning at cycle start ``t``, only the target from the row
    immediately before ``t`` is used. That preceding cycle becomes complete at
    the current row's start, so the baseline respects the prediction cutoff.
    """
    forecasts = tuple(
        CycleLengthForecast(
            cycle_start_date=current.cycle_start_date,
            predicted_cycle_length_days=previous.cycle_length_days,
            predicted_next_cycle_start_date=(
                current.cycle_start_date + timedelta(days=previous.cycle_length_days)
            ),
        )
        for previous, current in pairwise(dataset.rows)
    )
    return ForecastBatch(
        forecaster_name=PREVIOUS_CYCLE_BASELINE_NAME,
        forecaster_version=PREVIOUS_CYCLE_BASELINE_VERSION,
        dataset_fingerprint=dataset.fingerprint,
        forecasts=forecasts,
    )
