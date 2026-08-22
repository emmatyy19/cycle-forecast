"""Compare chronological forecasts over a common walk-forward window."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from cycle_forecast.data import CycleDataset
from cycle_forecast.forecasting.baselines import ForecastBatch
from cycle_forecast.forecasting.metrics import ForecastEvaluation, evaluate_forecasts


@dataclass(frozen=True, slots=True, kw_only=True)
class WalkForwardEvaluation:
    """Contain comparable evaluations over identical temporal cutoffs.

    Parameters
    ----------
    dataset_fingerprint
        Identity of the dataset supplying forecasts and actuals.
    cycle_start_dates
        Chronological cutoffs present in every evaluated forecast batch.
    evaluations
        Per-forecaster evaluations in caller-supplied order, each restricted to
        ``cycle_start_dates``.
    """

    dataset_fingerprint: str
    cycle_start_dates: tuple[date, ...]
    evaluations: tuple[ForecastEvaluation, ...]


def evaluate_walk_forward(
    *,
    dataset: CycleDataset,
    forecast_batches: Sequence[ForecastBatch],
) -> WalkForwardEvaluation:
    """Evaluate forecasters over their shared chronological prediction window.

    Parameters
    ----------
    dataset
        Immutable completed-cycle rows supplying actual targets.
    forecast_batches
        One or more chronological forecast batches to compare. Each forecaster
        name and version pair must be unique.

    Returns
    -------
    WalkForwardEvaluation
        Evaluations restricted to cycle-start cutoffs present in every batch.
        If any valid batch has no forecasts or the batches do not overlap, the
        shared window and every evaluation are empty with undefined metrics.

    Raises
    ------
    ValueError
        If no forecast batches are supplied, forecaster identities are
        duplicated, or a batch fails normal forecast evaluation validation.

    Notes
    -----
    This function enforces a fair temporal comparison after prediction. Every
    supplied forecast must have been generated using information available at
    its own cycle-start cutoff. The built-in baseline forecasters satisfy that
    requirement by construction.
    """
    batches = tuple(forecast_batches)
    if not batches:
        message = "at least one forecast batch is required"
        raise ValueError(message)

    identities = tuple(
        (batch.forecaster_name, batch.forecaster_version) for batch in batches
    )
    if len(set(identities)) != len(identities):
        message = "forecast batch forecaster identities must be unique"
        raise ValueError(message)

    # Validate every complete batch before restricting it to the common window.
    # This prevents malformed forecasts outside the overlap from being hidden.
    for batch in batches:
        evaluate_forecasts(dataset=dataset, forecast_batch=batch)

    common_starts = {forecast.cycle_start_date for forecast in batches[0].forecasts}
    for batch in batches[1:]:
        common_starts.intersection_update(
            forecast.cycle_start_date for forecast in batch.forecasts
        )
    common_start_dates = tuple(sorted(common_starts))

    evaluations = tuple(
        evaluate_forecasts(
            dataset=dataset,
            forecast_batch=ForecastBatch(
                forecaster_name=batch.forecaster_name,
                forecaster_version=batch.forecaster_version,
                dataset_fingerprint=batch.dataset_fingerprint,
                forecasts=tuple(
                    forecast
                    for forecast in batch.forecasts
                    if forecast.cycle_start_date in common_starts
                ),
            ),
        )
        for batch in batches
    )
    return WalkForwardEvaluation(
        dataset_fingerprint=dataset.fingerprint,
        cycle_start_dates=common_start_dates,
        evaluations=evaluations,
    )
