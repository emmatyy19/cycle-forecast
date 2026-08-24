"""Evaluate leakage-safe prediction intervals on development forecasts."""

from dataclasses import dataclass
from datetime import date
from math import ceil, isfinite
from statistics import mean, median

from cycle_forecast.evaluation.comparison import (
    DevelopmentModelComparison,
    ForecasterComparisonEntry,
)
from cycle_forecast.forecasting import ForecastError


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionIntervalConfig:
    """Configure sequential residual prediction intervals.

    Parameters
    ----------
    coverage_levels
        Strictly increasing unique target coverage fractions between zero and
        one.
    minimum_calibration_rows
        Earlier forecast errors required before producing an interval.
    """

    coverage_levels: tuple[float, ...]
    minimum_calibration_rows: int

    def __post_init__(self) -> None:
        """Validate finite interval settings and finite-sample feasibility."""
        if (
            not self.coverage_levels
            or any(
                not isfinite(level) or not 0.0 < level < 1.0
                for level in self.coverage_levels
            )
            or tuple(sorted(set(self.coverage_levels))) != self.coverage_levels
        ):
            message = (
                "coverage_levels must contain unique finite fractions between "
                "zero and one in increasing order"
            )
            raise ValueError(message)
        if self.minimum_calibration_rows < 1:
            message = "minimum_calibration_rows must be positive"
            raise ValueError(message)
        maximum_finite_coverage = self.minimum_calibration_rows / (
            self.minimum_calibration_rows + 1
        )
        if self.coverage_levels[-1] > maximum_finite_coverage:
            message = (
                "minimum_calibration_rows is too small for the largest finite "
                "coverage level"
            )
            raise ValueError(message)


DEFAULT_PREDICTION_INTERVAL_CONFIG = PredictionIntervalConfig(
    coverage_levels=(0.5, 0.8, 0.9),
    minimum_calibration_rows=12,
)
"""Fixed interval targets and calibration warmup for development reporting."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionInterval:
    """Represent one cutoff-safe cycle-length prediction interval.

    Parameters
    ----------
    cycle_start_date
        Prediction cutoff for the target cycle.
    predicted_cycle_length_days
        Point forecast at the center of the untruncated symmetric interval.
    actual_cycle_length_days
        Observed target used only after the interval was constructed.
    nominal_coverage
        Requested long-run coverage fraction.
    lower_cycle_length_days
        Inclusive lower bound, restricted to positive cycle lengths.
    upper_cycle_length_days
        Inclusive upper bound.
    calibration_error_count
        Number of strictly earlier errors used to choose the radius.
    radius_days
        Conformal absolute-residual quantile around the point forecast.
    contains_actual
        Whether the observed cycle length fell inside the interval.
    """

    cycle_start_date: date
    predicted_cycle_length_days: float
    actual_cycle_length_days: int
    nominal_coverage: float
    lower_cycle_length_days: float
    upper_cycle_length_days: float
    calibration_error_count: int
    radius_days: float
    contains_actual: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionIntervalMetrics:
    """Summarize interval reliability and usefulness together.

    Parameters
    ----------
    nominal_coverage
        Requested coverage fraction.
    interval_count
        Number of sequentially evaluated intervals.
    empirical_coverage
        Observed fraction containing the actual cycle length.
    mean_width_days
        Average upper-minus-lower interval width.
    median_width_days
        Median upper-minus-lower interval width.
    """

    nominal_coverage: float
    interval_count: int
    empirical_coverage: float
    mean_width_days: float
    median_width_days: float


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionIntervalEvaluation:
    """Tie sequential intervals and their metrics to one forecaster."""

    forecaster_label: str
    forecaster_name: str
    forecaster_version: str
    dataset_fingerprint: str
    intervals: tuple[PredictionInterval, ...]
    metrics: PredictionIntervalMetrics


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecasterUncertaintyEvaluation:
    """Contain every requested interval level for one forecaster."""

    forecaster_label: str
    interval_evaluations: tuple[PredictionIntervalEvaluation, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class DevelopmentUncertaintyEvaluation:
    """Compare selected-model uncertainty without opening the final holdout."""

    dataset_fingerprint: str
    holdout_policy_version: str
    configuration: PredictionIntervalConfig
    cycle_start_dates: tuple[date, ...]
    forecasters: tuple[ForecasterUncertaintyEvaluation, ...]


def _conformal_radius(
    *, earlier_errors: tuple[ForecastError, ...], coverage_level: float
) -> float:
    """Calculate the finite-sample corrected absolute-residual quantile.

    Parameters
    ----------
    earlier_errors
        Forecast errors strictly before the interval being formed.
    coverage_level
        Requested coverage fraction.

    Returns
    -------
    float
        Selected nonnegative residual radius.
    """
    ordered_residuals = sorted(error.absolute_error_days for error in earlier_errors)
    rank = ceil((len(ordered_residuals) + 1) * coverage_level)
    return ordered_residuals[rank - 1]


def _evaluate_entry(
    *,
    entry: ForecasterComparisonEntry,
    configuration: PredictionIntervalConfig,
) -> ForecasterUncertaintyEvaluation:
    """Evaluate all configured interval levels for one comparison entry."""
    errors = entry.evaluation.errors
    evaluations: list[PredictionIntervalEvaluation] = []
    for coverage_level in configuration.coverage_levels:
        intervals: list[PredictionInterval] = []
        for position in range(configuration.minimum_calibration_rows, len(errors)):
            current = errors[position]
            earlier_errors = errors[:position]
            radius = _conformal_radius(
                earlier_errors=earlier_errors,
                coverage_level=coverage_level,
            )
            lower = max(1.0, current.predicted_cycle_length_days - radius)
            upper = current.predicted_cycle_length_days + radius
            intervals.append(
                PredictionInterval(
                    cycle_start_date=current.cycle_start_date,
                    predicted_cycle_length_days=(current.predicted_cycle_length_days),
                    actual_cycle_length_days=current.actual_cycle_length_days,
                    nominal_coverage=coverage_level,
                    lower_cycle_length_days=lower,
                    upper_cycle_length_days=upper,
                    calibration_error_count=position,
                    radius_days=radius,
                    contains_actual=(
                        lower <= current.actual_cycle_length_days <= upper
                    ),
                )
            )
        if not intervals:
            message = "prediction intervals require errors after calibration warmup"
            raise ValueError(message)
        interval_tuple = tuple(intervals)
        widths = tuple(
            interval.upper_cycle_length_days - interval.lower_cycle_length_days
            for interval in interval_tuple
        )
        metrics = PredictionIntervalMetrics(
            nominal_coverage=coverage_level,
            interval_count=len(interval_tuple),
            empirical_coverage=(
                sum(interval.contains_actual for interval in interval_tuple)
                / len(interval_tuple)
            ),
            mean_width_days=mean(widths),
            median_width_days=median(widths),
        )
        evaluations.append(
            PredictionIntervalEvaluation(
                forecaster_label=entry.label,
                forecaster_name=entry.evaluation.forecaster_name,
                forecaster_version=entry.evaluation.forecaster_version,
                dataset_fingerprint=entry.evaluation.dataset_fingerprint,
                intervals=interval_tuple,
                metrics=metrics,
            )
        )
    return ForecasterUncertaintyEvaluation(
        forecaster_label=entry.label,
        interval_evaluations=tuple(evaluations),
    )


def evaluate_development_uncertainty(
    *,
    comparison: DevelopmentModelComparison,
    configuration: PredictionIntervalConfig = DEFAULT_PREDICTION_INTERVAL_CONFIG,
) -> DevelopmentUncertaintyEvaluation:
    """Evaluate sequential intervals for selected Ridge and strongest baseline.

    Parameters
    ----------
    comparison
        Development-only point-forecast comparison.
    configuration
        Fixed coverage targets and calibration warmup.

    Returns
    -------
    DevelopmentUncertaintyEvaluation
        Coverage and width results over identical post-warmup dates.

    Raises
    ------
    ValueError
        If development history is too short or provenance is inconsistent.

    Notes
    -----
    Each interval radius uses only absolute errors from earlier cutoffs. The
    current actual is consulted only after interval bounds are fixed. This
    function does not access final-holdout rows.
    """
    entries = (comparison.selected_ridge, comparison.strongest_baseline)
    if any(
        entry.evaluation.dataset_fingerprint != comparison.dataset_fingerprint
        for entry in entries
    ):
        message = "uncertainty entries must match the comparison fingerprint"
        raise ValueError(message)
    if len(comparison.cycle_start_dates) <= configuration.minimum_calibration_rows:
        message = "prediction intervals require errors after calibration warmup"
        raise ValueError(message)
    cycle_start_dates = comparison.cycle_start_dates[
        configuration.minimum_calibration_rows :
    ]
    forecasters = tuple(
        _evaluate_entry(entry=entry, configuration=configuration) for entry in entries
    )
    return DevelopmentUncertaintyEvaluation(
        dataset_fingerprint=comparison.dataset_fingerprint,
        holdout_policy_version=comparison.holdout_policy_version,
        configuration=configuration,
        cycle_start_dates=cycle_start_dates,
        forecasters=forecasters,
    )
