"""Select Ridge regularization and compare models on development data."""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum, auto
from math import isfinite

from cycle_forecast.data import CycleDataset
from cycle_forecast.features import HistoryFeatureDataset
from cycle_forecast.forecasting import (
    ForecastBatch,
    ForecastEvaluation,
    TemporalHoldoutSplit,
    evaluate_walk_forward,
    forecast_with_expanding_mean,
    forecast_with_previous_cycle,
    forecast_with_rolling_mean,
    forecast_with_rolling_median,
)
from cycle_forecast.models import (
    RidgeForecastConfig,
    forecast_with_walk_forward_ridge,
)


class ForecasterKind(StrEnum):
    """Classify comparison entries without string-based control flow."""

    BASELINE = auto()
    RIDGE = auto()


class _CandidateField(StrEnum):
    """Identify candidate configuration fields without magic strings."""

    RIDGE_ALPHAS = auto()
    ROLLING_WINDOWS = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class DevelopmentComparisonConfig:
    """Configure the fixed development-only model comparison.

    Parameters
    ----------
    ridge_alphas
        Strictly increasing unique positive Ridge strengths.
    rolling_windows
        Strictly increasing unique positive mean and median baseline windows.
    minimum_ridge_training_rows
        Positive earlier feature-row count required by every Ridge candidate.
    """

    ridge_alphas: tuple[float, ...]
    rolling_windows: tuple[int, ...]
    minimum_ridge_training_rows: int

    def __post_init__(self) -> None:
        """Validate deterministic comparison candidate sets."""
        _validate_positive_candidates(
            values=self.ridge_alphas,
            field=_CandidateField.RIDGE_ALPHAS,
        )
        _validate_positive_candidates(
            values=self.rolling_windows,
            field=_CandidateField.ROLLING_WINDOWS,
        )
        if self.minimum_ridge_training_rows < 1:
            message = "minimum_ridge_training_rows must be positive"
            raise ValueError(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecasterComparisonEntry:
    """Describe one method evaluated on the shared development window.

    Parameters
    ----------
    label
        Concise human-readable report label.
    kind
        Baseline or Ridge method category.
    ridge_alpha
        Ridge strength, or ``None`` for a baseline.
    evaluation
        Metrics and per-cycle errors on the shared development dates.
    """

    label: str
    kind: ForecasterKind
    ridge_alpha: float | None
    evaluation: ForecastEvaluation


@dataclass(frozen=True, slots=True, kw_only=True)
class DevelopmentModelComparison:
    """Contain development-only candidates, selection, and provenance.

    Parameters
    ----------
    dataset_fingerprint
        Identity of the complete dataset whose holdout remains untouched.
    holdout_policy_version
        Version of the final-holdout policy applied before comparison.
    feature_version
        Semantic version of features used by every Ridge candidate.
    configuration
        Exact Ridge and baseline candidates considered.
    cycle_start_dates
        Shared chronological development cutoffs for every entry.
    entries
        Baseline and Ridge evaluations in stable report order.
    selected_ridge
        Ridge candidate with lowest MAE and deterministic stronger-alpha tie-break.
    strongest_baseline
        Baseline with lowest MAE on the same development dates.
    """

    dataset_fingerprint: str
    holdout_policy_version: str
    feature_version: str
    configuration: DevelopmentComparisonConfig
    cycle_start_dates: tuple[date, ...]
    entries: tuple[ForecasterComparisonEntry, ...]
    selected_ridge: ForecasterComparisonEntry
    strongest_baseline: ForecasterComparisonEntry


def _validate_positive_candidates(
    *, values: tuple[float, ...] | tuple[int, ...], field: _CandidateField
) -> None:
    """Validate finite, positive, unique, increasing candidate values.

    Parameters
    ----------
    values
        Numeric candidate values.
    field
        Closed configuration-field identifier for validation messages.

    Raises
    ------
    ValueError
        If candidates are empty, invalid, duplicated, or out of order.
    """
    if (
        not values
        or any(not isfinite(value) or value <= 0 for value in values)
        or tuple(sorted(set(values))) != values
    ):
        message = (
            f"{field.value} must contain unique positive finite values "
            "in increasing order"
        )
        raise ValueError(message)


DEFAULT_DEVELOPMENT_COMPARISON_CONFIG = DevelopmentComparisonConfig(
    ridge_alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    rolling_windows=(3, 6, 12),
    minimum_ridge_training_rows=12,
)
"""Predeclared candidates evaluated without consulting final holdout rows."""


def _development_batch(
    *, batch: ForecastBatch, split: TemporalHoldoutSplit
) -> ForecastBatch:
    """Remove final-holdout forecasts before metrics or selection.

    Parameters
    ----------
    batch
        Complete-dataset leakage-safe forecast batch.
    split
        Temporal partition defining the first forbidden holdout cutoff.

    Returns
    -------
    ForecastBatch
        Same forecaster restricted strictly to development cycle starts.

    Raises
    ------
    ValueError
        If batch and split fingerprints differ.
    """
    if batch.dataset_fingerprint != split.dataset_fingerprint:
        message = "forecast batch fingerprint does not match temporal split"
        raise ValueError(message)
    first_holdout_start = split.holdout_rows[0].cycle_start_date
    return ForecastBatch(
        forecaster_name=batch.forecaster_name,
        forecaster_version=batch.forecaster_version,
        dataset_fingerprint=batch.dataset_fingerprint,
        forecasts=tuple(
            forecast
            for forecast in batch.forecasts
            if forecast.cycle_start_date < first_holdout_start
        ),
    )


def _require_mae(*, entry: ForecasterComparisonEntry) -> float:
    """Return a defined MAE for candidate selection.

    Parameters
    ----------
    entry
        Evaluated comparison entry.

    Returns
    -------
    float
        Defined mean absolute error.

    Raises
    ------
    ValueError
        If shared development history produced no forecasts.
    """
    value = entry.evaluation.metrics.mean_absolute_error_days
    if value is None:
        message = "development comparison requires a nonempty shared forecast window"
        raise ValueError(message)
    return value


def _require_ridge_alpha(*, entry: ForecasterComparisonEntry) -> float:
    """Return the Ridge strength attached to a Ridge entry.

    Parameters
    ----------
    entry
        Ridge comparison entry.

    Returns
    -------
    float
        Configured regularization strength.

    Raises
    ------
    ValueError
        If a non-Ridge entry is supplied.
    """
    if entry.ridge_alpha is None:
        message = "Ridge comparison entry must include an alpha"
        raise ValueError(message)
    return entry.ridge_alpha


def compare_development_forecasters(
    *,
    dataset: CycleDataset,
    split: TemporalHoldoutSplit,
    features: HistoryFeatureDataset,
    configuration: DevelopmentComparisonConfig = (
        DEFAULT_DEVELOPMENT_COMPARISON_CONFIG
    ),
) -> DevelopmentModelComparison:
    """Evaluate Ridge candidates and baselines without using final holdout rows.

    Parameters
    ----------
    dataset
        Complete immutable dataset used only to align development actuals.
    split
        Versioned partition whose holdout rows remain excluded.
    features
        Development-only supervised historical features.
    configuration
        Fixed Ridge strengths, rolling windows, and minimum training history.

    Returns
    -------
    DevelopmentModelComparison
        Shared-window metrics, selected Ridge candidate, and strongest baseline.

    Raises
    ------
    ValueError
        If provenance differs, split rows do not reconstruct the dataset, or
        shared development history is insufficient for evaluation.

    Notes
    -----
    Ridge is selected by lowest development MAE. Exact ties prefer the larger
    alpha, which applies stronger regularization. The selection never evaluates
    or reports final-holdout predictions.
    """
    if (
        dataset.fingerprint != split.dataset_fingerprint
        or dataset.fingerprint != features.dataset_fingerprint
    ):
        message = "dataset, temporal split, and features must share a fingerprint"
        raise ValueError(message)
    if split.development_rows + split.holdout_rows != dataset.rows:
        message = "temporal split rows do not reconstruct dataset"
        raise ValueError(message)
    if features.holdout_policy_version != split.policy_version:
        message = "feature and temporal split holdout policies do not match"
        raise ValueError(message)
    if not split.holdout_rows:
        message = "temporal split must reserve at least one final holdout row"
        raise ValueError(message)

    baseline_specs: list[tuple[str, ForecastBatch]] = [
        (
            "Previous cycle",
            forecast_with_previous_cycle(dataset=dataset),
        ),
        (
            "Expanding mean",
            forecast_with_expanding_mean(dataset=dataset),
        ),
    ]
    for window in configuration.rolling_windows:
        baseline_specs.extend(
            (
                (
                    f"Rolling mean ({window})",
                    forecast_with_rolling_mean(
                        dataset=dataset,
                        window_size=window,
                    ),
                ),
                (
                    f"Rolling median ({window})",
                    forecast_with_rolling_median(
                        dataset=dataset,
                        window_size=window,
                    ),
                ),
            )
        )

    ridge_specs = tuple(
        (
            f"Ridge (alpha={alpha:g})",
            alpha,
            forecast_with_walk_forward_ridge(
                features=features,
                configuration=RidgeForecastConfig(
                    alpha=alpha,
                    minimum_training_rows=configuration.minimum_ridge_training_rows,
                ),
            ).forecast_batch,
        )
        for alpha in configuration.ridge_alphas
    )
    development_batches = tuple(
        _development_batch(batch=batch, split=split) for _, batch in baseline_specs
    ) + tuple(batch for _, _, batch in ridge_specs)
    walk_forward = evaluate_walk_forward(
        dataset=dataset,
        forecast_batches=development_batches,
    )

    baseline_count = len(baseline_specs)
    baseline_entries = tuple(
        ForecasterComparisonEntry(
            label=label,
            kind=ForecasterKind.BASELINE,
            ridge_alpha=None,
            evaluation=evaluation,
        )
        for (label, _), evaluation in zip(
            baseline_specs,
            walk_forward.evaluations[:baseline_count],
            strict=True,
        )
    )
    ridge_entries = tuple(
        ForecasterComparisonEntry(
            label=label,
            kind=ForecasterKind.RIDGE,
            ridge_alpha=alpha,
            evaluation=evaluation,
        )
        for (label, alpha, _), evaluation in zip(
            ridge_specs,
            walk_forward.evaluations[baseline_count:],
            strict=True,
        )
    )
    for entry in baseline_entries + ridge_entries:
        _require_mae(entry=entry)

    selected_ridge = min(
        ridge_entries,
        key=lambda entry: (
            _require_mae(entry=entry),
            -_require_ridge_alpha(entry=entry),
        ),
    )
    strongest_baseline = min(
        baseline_entries,
        key=lambda entry: _require_mae(entry=entry),
    )
    return DevelopmentModelComparison(
        dataset_fingerprint=dataset.fingerprint,
        holdout_policy_version=split.policy_version,
        feature_version=features.feature_version,
        configuration=configuration,
        cycle_start_dates=walk_forward.cycle_start_dates,
        entries=baseline_entries + ridge_entries,
        selected_ridge=selected_ridge,
        strongest_baseline=strongest_baseline,
    )
