"""Align cycle-length forecasts with actuals and calculate metrics in days."""

from dataclasses import dataclass
from datetime import date
from math import isfinite, sqrt
from statistics import mean, median

from cycle_forecast.data import CycleDataset
from cycle_forecast.forecasting.baselines import ForecastBatch


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastError:
    """Represent the error for one cycle-length forecast.

    Parameters
    ----------
    cycle_start_date
        Prediction cutoff identifying the forecast and actual target.
    predicted_cycle_length_days
        Raw numeric forecast used for evaluation.
    actual_cycle_length_days
        Observed whole-day cycle length.
    error_days
        Signed error calculated as prediction minus actual. Negative values are
        early or short predictions; positive values are late or long.
    absolute_error_days
        Absolute magnitude of the signed error.
    """

    cycle_start_date: date
    predicted_cycle_length_days: float
    actual_cycle_length_days: int
    error_days: float
    absolute_error_days: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastMetrics:
    """Contain aggregate point-forecast metrics in days.

    Parameters
    ----------
    forecast_count
        Number of aligned forecasts summarized.
    mean_error_days
        Mean signed error, or ``None`` when no forecasts are available.
    mean_absolute_error_days
        Mean absolute error, or ``None`` when no forecasts are available.
    median_absolute_error_days
        Median absolute error, or ``None`` when no forecasts are available.
    root_mean_squared_error_days
        Root mean squared error, or ``None`` when no forecasts are available.
    within_1_day
        Fraction with absolute error at most one day, or ``None`` when empty.
    within_2_days
        Fraction with absolute error at most two days, or ``None`` when empty.
    within_3_days
        Fraction with absolute error at most three days, or ``None`` when empty.
    within_5_days
        Fraction with absolute error at most five days, or ``None`` when empty.
    """

    forecast_count: int
    mean_error_days: float | None
    mean_absolute_error_days: float | None
    median_absolute_error_days: float | None
    root_mean_squared_error_days: float | None
    within_1_day: float | None
    within_2_days: float | None
    within_3_days: float | None
    within_5_days: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastEvaluation:
    """Tie per-cycle errors and aggregate metrics to a forecast run.

    Parameters
    ----------
    forecaster_name
        Stable identifier for the evaluated forecasting method.
    forecaster_version
        Semantic version of the evaluated forecasting behavior.
    dataset_fingerprint
        Identity of the dataset supplying both forecasts and actuals.
    errors
        Per-cycle errors in chronological prediction-cutoff order.
    metrics
        Aggregate metrics calculated from ``errors``.
    """

    forecaster_name: str
    forecaster_version: str
    dataset_fingerprint: str
    errors: tuple[ForecastError, ...]
    metrics: ForecastMetrics


def _fraction_within(
    *, absolute_errors: tuple[float, ...], threshold_days: int
) -> float:
    """Calculate the fraction of errors within an inclusive day threshold.

    Parameters
    ----------
    absolute_errors
        Non-empty absolute forecast errors.
    threshold_days
        Inclusive nonnegative error threshold.

    Returns
    -------
    float
        Fraction of errors no greater than the threshold.
    """
    within_count = sum(error <= threshold_days for error in absolute_errors)
    return within_count / len(absolute_errors)


def _summarize_errors(*, errors: tuple[ForecastError, ...]) -> ForecastMetrics:
    """Aggregate aligned per-cycle forecast errors.

    Parameters
    ----------
    errors
        Chronological per-cycle forecast errors.

    Returns
    -------
    ForecastMetrics
        Metrics in days, with undefined values represented by ``None`` when
        ``errors`` is empty.
    """
    if not errors:
        return ForecastMetrics(
            forecast_count=0,
            mean_error_days=None,
            mean_absolute_error_days=None,
            median_absolute_error_days=None,
            root_mean_squared_error_days=None,
            within_1_day=None,
            within_2_days=None,
            within_3_days=None,
            within_5_days=None,
        )

    signed_errors = tuple(error.error_days for error in errors)
    absolute_errors = tuple(error.absolute_error_days for error in errors)
    return ForecastMetrics(
        forecast_count=len(errors),
        mean_error_days=mean(signed_errors),
        mean_absolute_error_days=mean(absolute_errors),
        median_absolute_error_days=median(absolute_errors),
        root_mean_squared_error_days=sqrt(mean(error**2 for error in signed_errors)),
        within_1_day=_fraction_within(
            absolute_errors=absolute_errors,
            threshold_days=1,
        ),
        within_2_days=_fraction_within(
            absolute_errors=absolute_errors,
            threshold_days=2,
        ),
        within_3_days=_fraction_within(
            absolute_errors=absolute_errors,
            threshold_days=3,
        ),
        within_5_days=_fraction_within(
            absolute_errors=absolute_errors,
            threshold_days=5,
        ),
    )


def evaluate_forecasts(
    *, dataset: CycleDataset, forecast_batch: ForecastBatch
) -> ForecastEvaluation:
    """Align forecasts to actual cycle lengths and calculate metrics.

    Parameters
    ----------
    dataset
        Immutable completed-cycle rows supplying actual targets.
    forecast_batch
        Chronological point forecasts to evaluate.

    Returns
    -------
    ForecastEvaluation
        Per-cycle errors and aggregate metrics tied to their provenance.

    Raises
    ------
    ValueError
        If dataset fingerprints differ, forecast cutoffs are not strictly
        chronological and unique, a cutoff has no actual row, or a raw
        prediction is not positive and finite.

    Notes
    -----
    Signed error is ``prediction - actual``. Metrics use the raw numeric
    prediction rather than the rounded operational prediction.
    """
    if forecast_batch.dataset_fingerprint != dataset.fingerprint:
        message = "forecast batch fingerprint does not match dataset"
        raise ValueError(message)

    actuals_by_start = {
        row.cycle_start_date: row.cycle_length_days for row in dataset.rows
    }
    errors: list[ForecastError] = []
    previous_start: date | None = None
    for forecast in forecast_batch.forecasts:
        if previous_start is not None and forecast.cycle_start_date <= previous_start:
            message = "forecast cycle starts must be strictly chronological and unique"
            raise ValueError(message)
        previous_start = forecast.cycle_start_date

        try:
            actual = actuals_by_start[forecast.cycle_start_date]
        except KeyError as error:
            message = (
                "forecast cycle start has no actual dataset row: "
                f"{forecast.cycle_start_date.isoformat()}"
            )
            raise ValueError(message) from error

        prediction = forecast.predicted_cycle_length_days
        if not isfinite(prediction) or prediction <= 0:
            message = "predicted cycle length must be positive and finite"
            raise ValueError(message)
        signed_error = prediction - actual
        errors.append(
            ForecastError(
                cycle_start_date=forecast.cycle_start_date,
                predicted_cycle_length_days=prediction,
                actual_cycle_length_days=actual,
                error_days=signed_error,
                absolute_error_days=abs(signed_error),
            )
        )

    error_tuple = tuple(errors)
    return ForecastEvaluation(
        forecaster_name=forecast_batch.forecaster_name,
        forecaster_version=forecast_batch.forecaster_version,
        dataset_fingerprint=dataset.fingerprint,
        errors=error_tuple,
        metrics=_summarize_errors(errors=error_tuple),
    )
