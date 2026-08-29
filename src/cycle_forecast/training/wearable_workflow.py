"""Run a private local comparison of daily wearable forecasting methods."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import StrEnum, auto
from itertools import pairwise
from math import sqrt
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cycle_forecast.data.cycle_history import CycleHistoryRecord, load_cycle_history
from cycle_forecast.data.oura import RetrievedOuraDailyObservation
from cycle_forecast.data.oura_normalization import normalize_oura_snapshots
from cycle_forecast.data.oura_snapshot import LoadedOuraSnapshot, load_snapshot
from cycle_forecast.data.wearable_alignment import align_daily_observation
from cycle_forecast.evaluation.wearable import (
    DailyCandidateEvaluation,
    DailyForecastCandidate,
    DailyModelComparison,
    compare_daily_forecasters,
)
from cycle_forecast.features.wearable import (
    WearableFeatureRow,
    build_wearable_feature_row,
)
from cycle_forecast.forecasting.daily import (
    CALIBRATION_WINDOWS,
    DAILY_FORECAST_HORIZON_DAYS,
    DailyPeriodDistribution,
)
from cycle_forecast.forecasting.wearable_baselines import (
    EMPIRICAL_HAZARD_BASELINE_VERSION,
    HISTORY_TEMPERATURE_BLEND_VERSION,
    TEMPERATURE_NEIGHBOR_BASELINE_VERSION,
    WEARABLE_NEIGHBOR_BASELINE_VERSION,
    blend_history_with_temperature,
    forecast_with_empirical_cycle_hazard,
    forecast_with_temperature_neighbors,
    forecast_with_wearable_neighbors,
)
from cycle_forecast.models.discrete_survival import (
    DISCRETE_SURVIVAL_MODEL_VERSION,
    DiscreteSurvivalConfig,
    fit_discrete_survival_model,
    predict_with_discrete_survival_model,
)

WEARABLE_EVALUATION_WORKFLOW_VERSION: Final = "wearable-evaluation-v4"
"""Semantic version of local assembly, temporal partitioning, and comparison."""

HISTORY_BASELINE_LABEL: Final = "Empirical cycle hazard"
HISTORY_TEMPERATURE_LABEL: Final = "Cycle history plus temperature"
TEMPERATURE_BASELINE_LABEL: Final = "Temperature ablation"
WEARABLE_BASELINE_LABEL: Final = "Wearable nearest neighbors"
SURVIVAL_MODEL_LABEL: Final = "Calibrated discrete survival"


class WearableEvaluationMode(StrEnum):
    """Identify historical-availability assumptions for local evaluation."""

    PROSPECTIVE = auto()
    EXPLORATORY_BACKFILL = "exploratory-backfill"


class WearableEvaluationError(ValueError):
    """Indicate invalid or insufficient private wearable evaluation data."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableCalibrationDiagnostic:
    """Summarize planning-window probabilities across evaluated mornings."""

    count: int
    mean_predicted_probability: float
    observed_fraction: float


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableCycleDayDiagnostic:
    """Summarize exact-date Brier error within a cycle-day band."""

    label: str
    count: int
    mean_brier_score: float


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableCandidateDiagnostics:
    """Describe how one candidate behaves beyond aggregate proper scores."""

    label: str
    version: str
    count: int
    mean_actual_outcome_probability: float
    minimum_actual_outcome_probability: float
    root_mean_squared_offset_error: float
    mean_signed_offset_error: float
    calibration: dict[int, WearableCalibrationDiagnostic]
    cycle_day: tuple[WearableCycleDayDiagnostic, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableDataDiagnostics:
    """Describe evaluated missingness and outcome prevalence without dates."""

    missingness_rates: dict[str, float]
    outcome_window_rates: dict[int, float]
    after_horizon_rate: float


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableEvaluationDiagnostics:
    """Contain privacy-safe data and candidate diagnostic summaries."""

    data: WearableDataDiagnostics
    candidates: tuple[WearableCandidateDiagnostics, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableCycleFoldResult:
    """Contain one unseen completed-cycle comparison without its private date."""

    fold_number: int
    training_cycle_count: int
    training_row_count: int
    calibration_row_count: int
    evaluation_row_count: int
    comparison: DailyModelComparison
    diagnostics: tuple[WearableCandidateDiagnostics, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableAggregateEntry:
    """Summarize equally weighted per-cycle scores for one forecaster."""

    label: str
    version: str
    mean_logarithmic_loss: float
    mean_multiclass_brier_score: float
    mean_window_brier_scores: dict[int, float]
    log_loss_cycle_wins: int
    brier_cycle_wins: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableWalkForwardComparison:
    """Contain chronological folds and cycle-weighted aggregate scores."""

    folds: tuple[WearableCycleFoldResult, ...]
    entries: tuple[WearableAggregateEntry, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableEvaluationResult:
    """Summarize a private comparison without identifiers or raw health values."""

    workflow_version: str
    mode: WearableEvaluationMode
    optimistic_backfill_assumption: bool
    snapshot_count: int
    normalized_day_count: int
    aligned_row_count: int
    uncensored_row_count: int
    eligible_completed_cycle_count: int
    evaluation_fold_count: int
    first_fold_training_cycle_count: int
    final_fold_training_cycle_count: int
    evaluation_cycle_count: int
    evaluation_row_count: int
    walk_forward: WearableWalkForwardComparison
    diagnostics: WearableEvaluationDiagnostics


def _load_snapshots(*, directory: Path) -> tuple[LoadedOuraSnapshot, ...]:
    """Load every validated private snapshot in stable path order."""
    paths = tuple(sorted(directory.glob("*.json"))) if directory.is_dir() else ()
    if not paths:
        raise WearableEvaluationError("no Oura snapshots found")
    return tuple(load_snapshot(path=path) for path in paths)


def _timezone(*, timezone_name: str) -> ZoneInfo:
    """Load a reproducible IANA timezone for local prediction dates."""
    try:
        return ZoneInfo(timezone_name)
    except (OSError, ValueError, ZoneInfoNotFoundError) as error:
        raise WearableEvaluationError("timezone must be an IANA timezone") from error


def _next_cycle_start(
    *, prediction_date: date, history: Sequence[CycleHistoryRecord]
) -> date | None:
    """Find the first recorded period start on or after a prediction date."""
    return next(
        (
            record.cycle_start_date
            for record in history
            if record.cycle_start_date >= prediction_date
        ),
        None,
    )


def _strip_future_sleep(
    *, observation: RetrievedOuraDailyObservation, cutoff: datetime
) -> RetrievedOuraDailyObservation | None:
    """Exclude a main sleep that had not ended by an assumed backfill cutoff."""
    sleep = observation.main_sleep
    if sleep is None or datetime.fromisoformat(sleep.bedtime_end) <= cutoff:
        return observation.model_copy(update={"available_at": cutoff})
    remaining = observation.model_copy(
        update={"available_at": cutoff, "main_sleep": None}
    )
    if any(
        document is not None
        for document in (remaining.readiness, remaining.daily_sleep)
    ):
        return remaining
    return None


def _prospective_contexts(
    *,
    snapshots: tuple[LoadedOuraSnapshot, ...],
    history: tuple[CycleHistoryRecord, ...],
    timezone: ZoneInfo,
) -> tuple[tuple[datetime, date, tuple[RetrievedOuraDailyObservation, ...]], ...]:
    """Reconstruct only mornings with proven retrieval-time availability."""
    earliest_by_date: dict[date, datetime] = {}
    for cutoff in sorted({snapshot.retrieval_started_at for snapshot in snapshots}):
        earliest_by_date.setdefault(cutoff.astimezone(timezone).date(), cutoff)
    cutoffs = tuple(earliest_by_date.values())
    contexts: list[
        tuple[datetime, date, tuple[RetrievedOuraDailyObservation, ...]]
    ] = []
    for cutoff in cutoffs:
        prediction_date = cutoff.astimezone(timezone).date()
        if not any(record.cycle_start_date < prediction_date for record in history):
            continue
        observations = normalize_oura_snapshots(snapshots=snapshots, cutoff=cutoff)
        contexts.append((cutoff, prediction_date, observations))
    return tuple(contexts)


def _exploratory_contexts(
    *,
    snapshots: tuple[LoadedOuraSnapshot, ...],
    history: tuple[CycleHistoryRecord, ...],
    timezone: ZoneInfo,
    prediction_hour: int,
) -> tuple[tuple[datetime, date, tuple[RetrievedOuraDailyObservation, ...]], ...]:
    """Assume current backfill versions existed at their source-day mornings."""
    if not 0 <= prediction_hour <= 23:
        raise WearableEvaluationError("prediction_hour must be between 0 and 23")
    latest_cutoff = max(snapshot.retrieval_started_at for snapshot in snapshots)
    normalized = normalize_oura_snapshots(snapshots=snapshots, cutoff=latest_cutoff)
    contexts: list[
        tuple[datetime, date, tuple[RetrievedOuraDailyObservation, ...]]
    ] = []
    for observation in normalized:
        if not any(record.cycle_start_date < observation.day for record in history):
            continue
        cutoff = datetime.combine(
            observation.day,
            time(hour=prediction_hour),
            tzinfo=timezone,
        )
        assumed = _strip_future_sleep(observation=observation, cutoff=cutoff)
        contexts.append(
            (cutoff, observation.day, (assumed,) if assumed is not None else ())
        )
    return tuple(contexts)


def _feature_rows(
    *,
    contexts: tuple[
        tuple[datetime, date, tuple[RetrievedOuraDailyObservation, ...]], ...
    ],
    history: tuple[CycleHistoryRecord, ...],
    observed_through: date,
) -> tuple[WearableFeatureRow, ...]:
    """Align contexts and attach labels known by the evaluation as-of date."""
    rows: list[WearableFeatureRow] = []
    for cutoff, prediction_date, observations in contexts:
        if prediction_date > observed_through:
            continue
        aligned = align_daily_observation(
            prediction_date=prediction_date,
            prediction_cutoff=cutoff,
            cycle_history=history,
            oura_observations=observations,
        )
        rows.append(
            build_wearable_feature_row(
                aligned=aligned,
                next_cycle_start=_next_cycle_start(
                    prediction_date=prediction_date,
                    history=history,
                ),
                observed_through=observed_through,
            )
        )
    return tuple(rows)


def _cycle_folds(
    *,
    rows: tuple[WearableFeatureRow, ...],
    history: tuple[CycleHistoryRecord, ...],
) -> tuple[
    tuple[
        tuple[WearableFeatureRow, ...],
        tuple[WearableFeatureRow, ...],
        tuple[WearableFeatureRow, ...],
    ],
    ...,
]:
    """Build expanding folds from completed cycles without crossing boundaries."""
    labeled = tuple(row for row in rows if row.outcome_offset_days is not None)
    completed_starts = {record.cycle_start_date for record in history[:-1]}
    cycle_starts = tuple(
        dict.fromkeys(
            row.aligned.cycle_start_date
            for row in labeled
            if row.aligned.cycle_start_date in completed_starts
        )
    )
    if len(cycle_starts) < 3:
        raise WearableEvaluationError(
            "wearable evaluation requires labeled mornings from at least three "
            "completed cycles"
        )
    folds: list[
        tuple[
            tuple[WearableFeatureRow, ...],
            tuple[WearableFeatureRow, ...],
            tuple[WearableFeatureRow, ...],
        ]
    ] = []
    for evaluation_position in range(2, len(cycle_starts)):
        training_cycles = set(cycle_starts[: evaluation_position - 1])
        calibration_cycle = cycle_starts[evaluation_position - 1]
        evaluation_cycle = cycle_starts[evaluation_position]
        training = tuple(
            row for row in labeled if row.aligned.cycle_start_date in training_cycles
        )
        calibration = tuple(
            row for row in labeled if row.aligned.cycle_start_date == calibration_cycle
        )
        evaluation = tuple(
            row for row in labeled if row.aligned.cycle_start_date == evaluation_cycle
        )
        if not training or not calibration or not evaluation:
            raise WearableEvaluationError(
                "walk-forward wearable partitions must be nonempty"
            )
        folds.append((training, calibration, evaluation))
    return tuple(folds)


def _completed_cycle_lengths(
    *, history: tuple[CycleHistoryRecord, ...], before: date
) -> tuple[int, ...]:
    """Calculate completed cycle lengths strictly before an evaluation cycle."""
    starts = tuple(record.cycle_start_date for record in history)
    return tuple(
        (following - current).days
        for current, following in pairwise(starts)
        if following <= before
    )


def _distribution_probabilities(
    *, forecast: DailyPeriodDistribution
) -> tuple[float, ...]:
    """Return the exhaustive forecast vector including the later outcome."""
    return (*forecast.daily_probabilities, forecast.after_horizon_probability)


def _candidate_diagnostics(
    *,
    label: str,
    version: str,
    forecasts: tuple[DailyPeriodDistribution, ...],
    rows: tuple[WearableFeatureRow, ...],
) -> WearableCandidateDiagnostics:
    """Calculate privacy-safe behavior diagnostics for one evaluated cycle."""
    exact_outcomes = tuple(
        row.outcome_offset_days for row in rows if row.outcome_offset_days is not None
    )
    if len(exact_outcomes) != len(rows):
        raise WearableEvaluationError("diagnostics require uncensored outcomes")
    actual_probabilities: list[float] = []
    squared_offset_errors: list[float] = []
    signed_offset_errors: list[float] = []
    cycle_day_groups: dict[str, list[float]] = {
        "days 1-10": [],
        "days 11-20": [],
        "day 21+": [],
    }
    for forecast, row, outcome in zip(forecasts, rows, exact_outcomes, strict=True):
        probabilities = _distribution_probabilities(forecast=forecast)
        outcome_index = min(outcome, DAILY_FORECAST_HORIZON_DAYS)
        actual_probabilities.append(probabilities[outcome_index])
        expected_offset = sum(
            index * probability for index, probability in enumerate(probabilities)
        )
        offset_error = expected_offset - outcome_index
        signed_offset_errors.append(offset_error)
        squared_offset_errors.append(offset_error**2)
        brier = sum(
            (probability - float(index == outcome_index)) ** 2
            for index, probability in enumerate(probabilities)
        )
        cycle_day = row.aligned.cycle_day
        band = (
            "days 1-10"
            if cycle_day <= 10
            else "days 11-20"
            if cycle_day <= 20
            else "day 21+"
        )
        cycle_day_groups[band].append(brier)
    count = len(rows)
    return WearableCandidateDiagnostics(
        label=label,
        version=version,
        count=count,
        mean_actual_outcome_probability=sum(actual_probabilities) / count,
        minimum_actual_outcome_probability=min(actual_probabilities),
        root_mean_squared_offset_error=sqrt(sum(squared_offset_errors) / count),
        mean_signed_offset_error=sum(signed_offset_errors) / count,
        calibration={
            window: WearableCalibrationDiagnostic(
                count=count,
                mean_predicted_probability=sum(
                    forecast.probability_within(days=window) for forecast in forecasts
                )
                / count,
                observed_fraction=sum(outcome < window for outcome in exact_outcomes)
                / count,
            )
            for window in CALIBRATION_WINDOWS
        },
        cycle_day=tuple(
            WearableCycleDayDiagnostic(
                label=band,
                count=len(scores),
                mean_brier_score=sum(scores) / len(scores),
            )
            for band, scores in cycle_day_groups.items()
            if scores
        ),
    )


def _data_diagnostics(
    *, evaluation_rows: tuple[WearableFeatureRow, ...]
) -> WearableDataDiagnostics:
    """Summarize evaluated input coverage and outcome prevalence."""
    count = len(evaluation_rows)
    missing_indices = {
        "Readiness score": 6,
        "Temperature": 7,
        "Sleep score": 8,
        "Average HRV": 9,
        "Total sleep": 10,
    }
    outcomes = tuple(
        row.outcome_offset_days
        for row in evaluation_rows
        if row.outcome_offset_days is not None
    )
    return WearableDataDiagnostics(
        missingness_rates={
            label: sum(row.values[index] == 1.0 for row in evaluation_rows) / count
            for label, index in missing_indices.items()
        },
        outcome_window_rates={
            window: sum(outcome < window for outcome in outcomes) / count
            for window in CALIBRATION_WINDOWS
        },
        after_horizon_rate=sum(
            outcome >= DAILY_FORECAST_HORIZON_DAYS for outcome in outcomes
        )
        / count,
    )


def _evaluate_fold(
    *,
    fold_number: int,
    training: tuple[WearableFeatureRow, ...],
    calibration: tuple[WearableFeatureRow, ...],
    evaluation: tuple[WearableFeatureRow, ...],
    history: tuple[CycleHistoryRecord, ...],
    neighbor_count: int,
) -> WearableCycleFoldResult:
    """Fit and compare every forecaster for one unseen completed cycle."""
    configuration = DiscreteSurvivalConfig(
        minimum_training_rows=len(training),
        minimum_calibration_rows=len(calibration),
    )
    model = fit_discrete_survival_model(
        training_rows=training,
        calibration_rows=calibration,
        configuration=configuration,
    )
    completed_lengths = _completed_cycle_lengths(
        history=history,
        before=evaluation[0].aligned.cycle_start_date,
    )
    if not completed_lengths:
        raise WearableEvaluationError(
            "history baseline requires a completed cycle before evaluation"
        )
    history_forecasts = tuple(
        forecast_with_empirical_cycle_hazard(
            row=row,
            completed_cycle_lengths=completed_lengths,
        )
        for row in evaluation
    )
    prior_neighbor_rows = training + calibration
    neighbor_forecasts = tuple(
        forecast_with_wearable_neighbors(
            row=row,
            training_rows=prior_neighbor_rows,
            neighbor_count=neighbor_count,
        )
        for row in evaluation
    )
    temperature_forecasts = tuple(
        forecast_with_temperature_neighbors(
            row=row,
            training_rows=prior_neighbor_rows,
            neighbor_count=neighbor_count,
        )
        for row in evaluation
    )
    history_temperature_forecasts = tuple(
        blend_history_with_temperature(history=history, temperature=temperature)
        for history, temperature in zip(
            history_forecasts,
            temperature_forecasts,
            strict=True,
        )
    )
    model_forecasts = tuple(
        predict_with_discrete_survival_model(model=model, row=row) for row in evaluation
    )
    outcomes = tuple(
        row.outcome_offset_days
        for row in evaluation
        if row.outcome_offset_days is not None
    )
    comparison = compare_daily_forecasters(
        candidates=(
            DailyForecastCandidate(
                label=HISTORY_BASELINE_LABEL,
                version=EMPIRICAL_HAZARD_BASELINE_VERSION,
                forecasts=history_forecasts,
                outcome_offsets=outcomes,
            ),
            DailyForecastCandidate(
                label=HISTORY_TEMPERATURE_LABEL,
                version=HISTORY_TEMPERATURE_BLEND_VERSION,
                forecasts=history_temperature_forecasts,
                outcome_offsets=outcomes,
            ),
            DailyForecastCandidate(
                label=TEMPERATURE_BASELINE_LABEL,
                version=TEMPERATURE_NEIGHBOR_BASELINE_VERSION,
                forecasts=temperature_forecasts,
                outcome_offsets=outcomes,
            ),
            DailyForecastCandidate(
                label=WEARABLE_BASELINE_LABEL,
                version=WEARABLE_NEIGHBOR_BASELINE_VERSION,
                forecasts=neighbor_forecasts,
                outcome_offsets=outcomes,
            ),
            DailyForecastCandidate(
                label=SURVIVAL_MODEL_LABEL,
                version=DISCRETE_SURVIVAL_MODEL_VERSION,
                forecasts=model_forecasts,
                outcome_offsets=outcomes,
            ),
        )
    )
    return WearableCycleFoldResult(
        fold_number=fold_number,
        training_cycle_count=len({row.aligned.cycle_start_date for row in training}),
        training_row_count=len(training),
        calibration_row_count=len(calibration),
        evaluation_row_count=len(evaluation),
        comparison=comparison,
        diagnostics=tuple(
            _candidate_diagnostics(
                label=label,
                version=version,
                forecasts=forecasts,
                rows=evaluation,
            )
            for label, version, forecasts in (
                (
                    HISTORY_BASELINE_LABEL,
                    EMPIRICAL_HAZARD_BASELINE_VERSION,
                    history_forecasts,
                ),
                (
                    HISTORY_TEMPERATURE_LABEL,
                    HISTORY_TEMPERATURE_BLEND_VERSION,
                    history_temperature_forecasts,
                ),
                (
                    TEMPERATURE_BASELINE_LABEL,
                    TEMPERATURE_NEIGHBOR_BASELINE_VERSION,
                    temperature_forecasts,
                ),
                (
                    WEARABLE_BASELINE_LABEL,
                    WEARABLE_NEIGHBOR_BASELINE_VERSION,
                    neighbor_forecasts,
                ),
                (
                    SURVIVAL_MODEL_LABEL,
                    DISCRETE_SURVIVAL_MODEL_VERSION,
                    model_forecasts,
                ),
            )
        ),
    )


def _entry_at(
    *, fold: WearableCycleFoldResult, position: int
) -> DailyCandidateEvaluation:
    """Return one stable candidate position from a cycle fold."""
    return fold.comparison.entries[position]


def _aggregate_folds(
    *, folds: tuple[WearableCycleFoldResult, ...]
) -> tuple[WearableAggregateEntry, ...]:
    """Average scores with equal weight for every evaluated cycle."""
    if not folds:
        raise WearableEvaluationError("walk-forward evaluation produced no folds")
    candidate_count = len(folds[0].comparison.entries)
    entries: list[WearableAggregateEntry] = []
    for position in range(candidate_count):
        candidates = tuple(_entry_at(fold=fold, position=position) for fold in folds)
        reference = candidates[0]
        if any(
            candidate.label != reference.label or candidate.version != reference.version
            for candidate in candidates
        ):
            raise WearableEvaluationError(
                "walk-forward candidate identity changed across cycles"
            )
        entries.append(
            WearableAggregateEntry(
                label=reference.label,
                version=reference.version,
                mean_logarithmic_loss=sum(
                    candidate.evaluation.logarithmic_loss for candidate in candidates
                )
                / len(candidates),
                mean_multiclass_brier_score=sum(
                    candidate.evaluation.multiclass_brier_score
                    for candidate in candidates
                )
                / len(candidates),
                mean_window_brier_scores={
                    window: sum(
                        candidate.evaluation.window_brier_scores[window]
                        for candidate in candidates
                    )
                    / len(candidates)
                    for window in CALIBRATION_WINDOWS
                },
                log_loss_cycle_wins=sum(
                    candidate.evaluation.logarithmic_loss
                    == min(
                        entry.evaluation.logarithmic_loss
                        for entry in fold.comparison.entries
                    )
                    for candidate, fold in zip(candidates, folds, strict=True)
                ),
                brier_cycle_wins=sum(
                    candidate.evaluation.multiclass_brier_score
                    == min(
                        entry.evaluation.multiclass_brier_score
                        for entry in fold.comparison.entries
                    )
                    for candidate, fold in zip(candidates, folds, strict=True)
                ),
            )
        )
    return tuple(entries)


def _aggregate_diagnostics(
    *, folds: tuple[WearableCycleFoldResult, ...]
) -> tuple[WearableCandidateDiagnostics, ...]:
    """Pool diagnostic moments across all unseen evaluation mornings."""
    candidate_count = len(folds[0].diagnostics)
    aggregates: list[WearableCandidateDiagnostics] = []
    for position in range(candidate_count):
        candidates = tuple(fold.diagnostics[position] for fold in folds)
        reference = candidates[0]
        if any(
            candidate.label != reference.label or candidate.version != reference.version
            for candidate in candidates
        ):
            raise WearableEvaluationError(
                "diagnostic candidate identity changed across cycles"
            )
        count = sum(candidate.count for candidate in candidates)
        cycle_day_labels = tuple(
            dict.fromkeys(
                diagnostic.label
                for candidate in candidates
                for diagnostic in candidate.cycle_day
            )
        )
        aggregates.append(
            WearableCandidateDiagnostics(
                label=reference.label,
                version=reference.version,
                count=count,
                mean_actual_outcome_probability=sum(
                    candidate.mean_actual_outcome_probability * candidate.count
                    for candidate in candidates
                )
                / count,
                minimum_actual_outcome_probability=min(
                    candidate.minimum_actual_outcome_probability
                    for candidate in candidates
                ),
                root_mean_squared_offset_error=sqrt(
                    sum(
                        candidate.root_mean_squared_offset_error**2 * candidate.count
                        for candidate in candidates
                    )
                    / count
                ),
                mean_signed_offset_error=sum(
                    candidate.mean_signed_offset_error * candidate.count
                    for candidate in candidates
                )
                / count,
                calibration={
                    window: WearableCalibrationDiagnostic(
                        count=count,
                        mean_predicted_probability=sum(
                            candidate.calibration[window].mean_predicted_probability
                            * candidate.calibration[window].count
                            for candidate in candidates
                        )
                        / count,
                        observed_fraction=sum(
                            candidate.calibration[window].observed_fraction
                            * candidate.calibration[window].count
                            for candidate in candidates
                        )
                        / count,
                    )
                    for window in CALIBRATION_WINDOWS
                },
                cycle_day=tuple(
                    WearableCycleDayDiagnostic(
                        label=label,
                        count=sum(
                            diagnostic.count
                            for candidate in candidates
                            for diagnostic in candidate.cycle_day
                            if diagnostic.label == label
                        ),
                        mean_brier_score=(
                            sum(
                                diagnostic.mean_brier_score * diagnostic.count
                                for candidate in candidates
                                for diagnostic in candidate.cycle_day
                                if diagnostic.label == label
                            )
                            / sum(
                                diagnostic.count
                                for candidate in candidates
                                for diagnostic in candidate.cycle_day
                                if diagnostic.label == label
                            )
                        ),
                    )
                    for label in cycle_day_labels
                ),
            )
        )
    return tuple(aggregates)


def evaluate_local_wearable_models(
    *,
    history_path: Path,
    snapshot_directory: Path,
    timezone_name: str,
    mode: WearableEvaluationMode,
    observed_through: date,
    prediction_hour: int = 9,
    neighbor_count: int = 20,
) -> WearableEvaluationResult:
    """Assemble private data and compare baselines with calibrated survival.

    Parameters
    ----------
    history_path
        Private validated cycle-history CSV.
    snapshot_directory
        Directory containing immutable private Oura snapshots.
    timezone_name
        IANA timezone used to construct local prediction dates.
    mode
        Strict retrieval provenance or explicitly optimistic backfill behavior.
    observed_through
        Last local date whose period-start absence is known.
    prediction_hour
        Assumed local cutoff hour used only by exploratory backfill mode.
    neighbor_count
        Positive maximum number of earlier mornings used by the wearable baseline.

    Returns
    -------
    WearableEvaluationResult
        Privacy-safe counts and shared-window development metrics.

    Raises
    ------
    WearableEvaluationError
        If inputs are invalid or insufficient for cycle-level partitioning.
    """
    history = load_cycle_history(path=history_path)
    snapshots = _load_snapshots(directory=snapshot_directory)
    timezone = _timezone(timezone_name=timezone_name)
    latest_cutoff = max(snapshot.retrieval_started_at for snapshot in snapshots)
    normalized = normalize_oura_snapshots(snapshots=snapshots, cutoff=latest_cutoff)
    contexts = (
        _prospective_contexts(
            snapshots=snapshots,
            history=history,
            timezone=timezone,
        )
        if mode is WearableEvaluationMode.PROSPECTIVE
        else _exploratory_contexts(
            snapshots=snapshots,
            history=history,
            timezone=timezone,
            prediction_hour=prediction_hour,
        )
    )
    rows = _feature_rows(
        contexts=contexts,
        history=history,
        observed_through=observed_through,
    )
    partitions = _cycle_folds(rows=rows, history=history)
    folds = tuple(
        _evaluate_fold(
            fold_number=fold_number,
            training=training,
            calibration=calibration,
            evaluation=evaluation,
            history=history,
            neighbor_count=neighbor_count,
        )
        for fold_number, (training, calibration, evaluation) in enumerate(
            partitions, start=1
        )
    )
    aggregate_entries = _aggregate_folds(folds=folds)
    evaluated_rows = tuple(row for _, _, evaluation in partitions for row in evaluation)
    evaluated_cycle_count = len(folds)
    return WearableEvaluationResult(
        workflow_version=WEARABLE_EVALUATION_WORKFLOW_VERSION,
        mode=mode,
        optimistic_backfill_assumption=(
            mode is WearableEvaluationMode.EXPLORATORY_BACKFILL
        ),
        snapshot_count=len(snapshots),
        normalized_day_count=len(normalized),
        aligned_row_count=len(rows),
        uncensored_row_count=sum(row.outcome_offset_days is not None for row in rows),
        eligible_completed_cycle_count=evaluated_cycle_count + 2,
        evaluation_fold_count=evaluated_cycle_count,
        first_fold_training_cycle_count=folds[0].training_cycle_count,
        final_fold_training_cycle_count=folds[-1].training_cycle_count,
        evaluation_cycle_count=evaluated_cycle_count,
        evaluation_row_count=sum(fold.evaluation_row_count for fold in folds),
        walk_forward=WearableWalkForwardComparison(
            folds=folds,
            entries=aggregate_entries,
        ),
        diagnostics=WearableEvaluationDiagnostics(
            data=_data_diagnostics(evaluation_rows=evaluated_rows),
            candidates=_aggregate_diagnostics(folds=folds),
        ),
    )
