"""Run a private local comparison of daily wearable forecasting methods."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import StrEnum, auto
from itertools import pairwise
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cycle_forecast.data.cycle_history import CycleHistoryRecord, load_cycle_history
from cycle_forecast.data.oura import RetrievedOuraDailyObservation
from cycle_forecast.data.oura_normalization import normalize_oura_snapshots
from cycle_forecast.data.oura_snapshot import LoadedOuraSnapshot, load_snapshot
from cycle_forecast.data.wearable_alignment import align_daily_observation
from cycle_forecast.evaluation.wearable import (
    DailyForecastCandidate,
    DailyModelComparison,
    compare_daily_forecasters,
)
from cycle_forecast.features.wearable import (
    WearableFeatureRow,
    build_wearable_feature_row,
)
from cycle_forecast.forecasting.wearable_baselines import (
    EMPIRICAL_HAZARD_BASELINE_VERSION,
    WEARABLE_NEIGHBOR_BASELINE_VERSION,
    forecast_with_empirical_cycle_hazard,
    forecast_with_wearable_neighbors,
)
from cycle_forecast.models.discrete_survival import (
    DISCRETE_SURVIVAL_MODEL_VERSION,
    DiscreteSurvivalConfig,
    fit_discrete_survival_model,
    predict_with_discrete_survival_model,
)

WEARABLE_EVALUATION_WORKFLOW_VERSION: Final = "wearable-evaluation-v1"
"""Semantic version of local assembly, temporal partitioning, and comparison."""

HISTORY_BASELINE_LABEL: Final = "Empirical cycle hazard"
WEARABLE_BASELINE_LABEL: Final = "Wearable nearest neighbors"
SURVIVAL_MODEL_LABEL: Final = "Calibrated discrete survival"


class WearableEvaluationMode(StrEnum):
    """Identify historical-availability assumptions for local evaluation."""

    PROSPECTIVE = auto()
    EXPLORATORY_BACKFILL = "exploratory-backfill"


class WearableEvaluationError(ValueError):
    """Indicate invalid or insufficient private wearable evaluation data."""


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
    training_cycle_count: int
    calibration_cycle_count: int
    evaluation_cycle_count: int
    training_row_count: int
    calibration_row_count: int
    evaluation_row_count: int
    comparison: DailyModelComparison


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


def _partition_rows(
    *, rows: tuple[WearableFeatureRow, ...]
) -> tuple[
    tuple[WearableFeatureRow, ...],
    tuple[WearableFeatureRow, ...],
    tuple[WearableFeatureRow, ...],
]:
    """Reserve whole chronological cycles for calibration and evaluation."""
    labeled = tuple(row for row in rows if row.outcome_offset_days is not None)
    cycle_starts = tuple(dict.fromkeys(row.aligned.cycle_start_date for row in labeled))
    if len(cycle_starts) < 3:
        raise WearableEvaluationError(
            "wearable evaluation requires labeled mornings from at least three cycles"
        )
    calibration_cycle = cycle_starts[-2]
    evaluation_cycle = cycle_starts[-1]
    training = tuple(
        row for row in labeled if row.aligned.cycle_start_date < calibration_cycle
    )
    calibration = tuple(
        row for row in labeled if row.aligned.cycle_start_date == calibration_cycle
    )
    evaluation = tuple(
        row for row in labeled if row.aligned.cycle_start_date == evaluation_cycle
    )
    if not training or not calibration or not evaluation:
        raise WearableEvaluationError("temporal wearable partitions must be nonempty")
    return training, calibration, evaluation


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
    training, calibration, evaluation = _partition_rows(rows=rows)
    configuration = DiscreteSurvivalConfig(
        minimum_training_rows=len(training),
        minimum_calibration_rows=len(calibration),
    )
    model = fit_discrete_survival_model(
        training_rows=training,
        calibration_rows=calibration,
        configuration=configuration,
    )
    prior_neighbor_rows = training + calibration
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
    neighbor_forecasts = tuple(
        forecast_with_wearable_neighbors(
            row=row,
            training_rows=prior_neighbor_rows,
            neighbor_count=neighbor_count,
        )
        for row in evaluation
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
        training_cycle_count=len({row.aligned.cycle_start_date for row in training}),
        calibration_cycle_count=1,
        evaluation_cycle_count=1,
        training_row_count=len(training),
        calibration_row_count=len(calibration),
        evaluation_row_count=len(evaluation),
        comparison=comparison,
    )
