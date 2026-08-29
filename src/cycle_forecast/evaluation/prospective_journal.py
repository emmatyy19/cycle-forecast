"""Persist and score immutable private forecasts after outcomes arrive."""

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import Final, cast

from cycle_forecast.data.cycle_history import CycleHistoryRecord
from cycle_forecast.data.private_files import ensure_private_directory
from cycle_forecast.forecasting.daily import (
    CALIBRATION_WINDOWS,
    DailyForecastEvaluation,
    DailyPeriodDistribution,
    evaluate_daily_distributions,
)
from cycle_forecast.prediction_daily import DailyPointEstimate, HistoryDailyPrediction

PROSPECTIVE_JOURNAL_SCHEMA_VERSION: Final = "prospective-forecast-journal-v3"
"""Version of immutable forecast entries and delayed scoring semantics."""

DEFAULT_PROSPECTIVE_JOURNAL_PATH: Final = Path("data/private/forecast-journal.jsonl")
"""Ignored owner-private journal used by the unified daily workflow."""

_V1_ENTRY_FIELDS: Final[set[str]] = {
    "schema_version",
    "prediction_date",
    "prediction_cutoff",
    "current_cycle_start_date",
    "cycle_day",
    "probability_model_version",
    "daily_probabilities",
    "after_horizon_probability",
    "point_estimate_date",
    "point_estimate_cycle_length_days",
    "point_estimate_method",
    "model_dataset_fingerprint",
    "oura_synced_through",
}
_V2_ENTRY_FIELDS: Final[set[str]] = _V1_ENTRY_FIELDS | {
    "wearable_model_version",
    "wearable_daily_probabilities",
    "wearable_after_horizon_probability",
}
_ENTRY_FIELDS: Final[set[str]] = _V2_ENTRY_FIELDS | {
    "temperature_model_version",
    "temperature_daily_probabilities",
    "temperature_after_horizon_probability",
}


class ProspectiveJournalError(ValueError):
    """Indicate an invalid or conflicting private forecast journal."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ProspectiveForecastEntry:
    """Store exactly one morning forecast before its outcome is known."""

    schema_version: str
    prediction_date: date
    prediction_cutoff: datetime
    current_cycle_start_date: date
    cycle_day: int
    probability_model_version: str
    daily_probabilities: tuple[float, ...]
    after_horizon_probability: float
    point_estimate_date: date
    point_estimate_cycle_length_days: float
    point_estimate_method: str
    model_dataset_fingerprint: str
    oura_synced_through: date
    wearable_model_version: str | None = None
    wearable_daily_probabilities: tuple[float, ...] | None = None
    wearable_after_horizon_probability: float | None = None
    temperature_model_version: str | None = None
    temperature_daily_probabilities: tuple[float, ...] | None = None
    temperature_after_horizon_probability: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ProspectivePerformanceSummary:
    """Summarize equally weighted completed-cycle prospective performance."""

    journal_forecast_count: int
    resolved_forecast_count: int
    completed_cycle_count: int
    mean_cycle_logarithmic_loss: float | None
    mean_cycle_brier_score: float | None
    mean_cycle_window_brier_scores: dict[int, float]
    mean_cycle_point_absolute_error_days: float | None
    wearable_resolved_forecast_count: int
    wearable_completed_cycle_count: int
    wearable_mean_cycle_logarithmic_loss: float | None
    wearable_mean_cycle_brier_score: float | None
    wearable_mean_cycle_window_brier_scores: dict[int, float]
    temperature_resolved_forecast_count: int
    temperature_completed_cycle_count: int
    temperature_mean_cycle_logarithmic_loss: float | None
    temperature_mean_cycle_brier_score: float | None
    temperature_mean_cycle_window_brier_scores: dict[int, float]


def build_prospective_entry(
    *,
    prediction: HistoryDailyPrediction,
    point_estimate: DailyPointEstimate,
    model_dataset_fingerprint: str,
    oura_synced_through: date,
    wearable_model_version: str | None = None,
    wearable_distribution: DailyPeriodDistribution | None = None,
    temperature_model_version: str | None = None,
    temperature_distribution: DailyPeriodDistribution | None = None,
) -> ProspectiveForecastEntry:
    """Build a versioned journal entry from the completed daily forecast."""
    return ProspectiveForecastEntry(
        schema_version=PROSPECTIVE_JOURNAL_SCHEMA_VERSION,
        prediction_date=prediction.prediction_date,
        prediction_cutoff=prediction.distribution.prediction_cutoff,
        current_cycle_start_date=prediction.current_cycle_start_date,
        cycle_day=prediction.cycle_day,
        probability_model_version=prediction.model_version,
        daily_probabilities=prediction.distribution.daily_probabilities,
        after_horizon_probability=(prediction.distribution.after_horizon_probability),
        point_estimate_date=point_estimate.predicted_next_cycle_start_date,
        point_estimate_cycle_length_days=(point_estimate.predicted_cycle_length_days),
        point_estimate_method=point_estimate.method.value,
        model_dataset_fingerprint=model_dataset_fingerprint,
        oura_synced_through=oura_synced_through,
        wearable_model_version=wearable_model_version,
        wearable_daily_probabilities=(
            wearable_distribution.daily_probabilities
            if wearable_distribution is not None
            else None
        ),
        wearable_after_horizon_probability=(
            wearable_distribution.after_horizon_probability
            if wearable_distribution is not None
            else None
        ),
        temperature_model_version=temperature_model_version,
        temperature_daily_probabilities=(
            temperature_distribution.daily_probabilities
            if temperature_distribution is not None
            else None
        ),
        temperature_after_horizon_probability=(
            temperature_distribution.after_horizon_probability
            if temperature_distribution is not None
            else None
        ),
    )


def _serialize_entry(*, entry: ProspectiveForecastEntry) -> bytes:
    """Encode one entry as canonical newline-delimited JSON."""
    payload = asdict(entry)
    if entry.schema_version == "prospective-forecast-journal-v1":
        for field in _ENTRY_FIELDS - _V1_ENTRY_FIELDS:
            del payload[field]
    elif entry.schema_version == "prospective-forecast-journal-v2":
        for field in _ENTRY_FIELDS - _V2_ENTRY_FIELDS:
            del payload[field]
    return (
        json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _entry_from_object(*, value: object, line_number: int) -> ProspectiveForecastEntry:
    """Validate one decoded journal object without accepting coercive shapes."""
    if not isinstance(value, dict):
        raise ProspectiveJournalError(
            f"forecast journal line {line_number} must be an object"
        )
    payload = cast(dict[object, object], value)
    try:
        schema_version = payload.get("schema_version")
        expected_fields = (
            _V1_ENTRY_FIELDS
            if schema_version == "prospective-forecast-journal-v1"
            else _V2_ENTRY_FIELDS
            if schema_version == "prospective-forecast-journal-v2"
            else _ENTRY_FIELDS
        )
        if (
            any(not isinstance(key, str) for key in payload)
            or {key for key in payload if isinstance(key, str)} != expected_fields
        ):
            raise ValueError("entry fields do not match the journal schema")
        if schema_version not in {
            "prospective-forecast-journal-v1",
            "prospective-forecast-journal-v2",
            PROSPECTIVE_JOURNAL_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported schema version")
        probabilities_raw = payload["daily_probabilities"]
        if not isinstance(probabilities_raw, list):
            raise ValueError("daily probabilities must be an array")
        probability_items = cast(list[object], probabilities_raw)
        probabilities = tuple(
            float(item)
            for item in probability_items
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        )
        if len(probabilities) != len(probability_items):
            raise ValueError("daily probabilities must be numeric")
        wearable_probabilities_raw = payload.get("wearable_daily_probabilities")
        if wearable_probabilities_raw is not None and not isinstance(
            wearable_probabilities_raw, list
        ):
            raise ValueError("wearable daily probabilities must be an array")
        wearable_items = (
            cast(list[object], wearable_probabilities_raw)
            if wearable_probabilities_raw is not None
            else None
        )
        wearable_probabilities = (
            tuple(
                float(item)
                for item in wearable_items
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            )
            if wearable_items is not None
            else None
        )
        if wearable_items is not None and len(wearable_probabilities or ()) != len(
            wearable_items
        ):
            raise ValueError("wearable daily probabilities must be numeric")
        temperature_probabilities_raw = payload.get("temperature_daily_probabilities")
        if temperature_probabilities_raw is not None and not isinstance(
            temperature_probabilities_raw, list
        ):
            raise ValueError("temperature daily probabilities must be an array")
        temperature_items = (
            cast(list[object], temperature_probabilities_raw)
            if temperature_probabilities_raw is not None
            else None
        )
        temperature_probabilities = (
            tuple(
                float(item)
                for item in temperature_items
                if isinstance(item, (int, float)) and not isinstance(item, bool)
            )
            if temperature_items is not None
            else None
        )
        if temperature_items is not None and len(
            temperature_probabilities or ()
        ) != len(temperature_items):
            raise ValueError("temperature daily probabilities must be numeric")
        entry = ProspectiveForecastEntry(
            schema_version=str(schema_version),
            prediction_date=date.fromisoformat(str(payload["prediction_date"])),
            prediction_cutoff=datetime.fromisoformat(str(payload["prediction_cutoff"])),
            current_cycle_start_date=date.fromisoformat(
                str(payload["current_cycle_start_date"])
            ),
            cycle_day=int(str(payload["cycle_day"])),
            probability_model_version=str(payload["probability_model_version"]),
            daily_probabilities=probabilities,
            after_horizon_probability=float(str(payload["after_horizon_probability"])),
            point_estimate_date=date.fromisoformat(str(payload["point_estimate_date"])),
            point_estimate_cycle_length_days=float(
                str(payload["point_estimate_cycle_length_days"])
            ),
            point_estimate_method=str(payload["point_estimate_method"]),
            model_dataset_fingerprint=str(payload["model_dataset_fingerprint"]),
            oura_synced_through=date.fromisoformat(str(payload["oura_synced_through"])),
            wearable_model_version=(
                str(payload["wearable_model_version"])
                if payload.get("wearable_model_version") is not None
                else None
            ),
            wearable_daily_probabilities=wearable_probabilities,
            wearable_after_horizon_probability=(
                float(str(payload["wearable_after_horizon_probability"]))
                if payload.get("wearable_after_horizon_probability") is not None
                else None
            ),
            temperature_model_version=(
                str(payload["temperature_model_version"])
                if payload.get("temperature_model_version") is not None
                else None
            ),
            temperature_daily_probabilities=temperature_probabilities,
            temperature_after_horizon_probability=(
                float(str(payload["temperature_after_horizon_probability"]))
                if payload.get("temperature_after_horizon_probability") is not None
                else None
            ),
        )
        DailyPeriodDistribution(
            prediction_date=entry.prediction_date,
            prediction_cutoff=entry.prediction_cutoff,
            daily_probabilities=entry.daily_probabilities,
            after_horizon_probability=entry.after_horizon_probability,
        )
        if entry.wearable_daily_probabilities is not None:
            DailyPeriodDistribution(
                prediction_date=entry.prediction_date,
                prediction_cutoff=entry.prediction_cutoff,
                daily_probabilities=entry.wearable_daily_probabilities,
                after_horizon_probability=cast(
                    float, entry.wearable_after_horizon_probability
                ),
            )
        if entry.temperature_daily_probabilities is not None:
            DailyPeriodDistribution(
                prediction_date=entry.prediction_date,
                prediction_cutoff=entry.prediction_cutoff,
                daily_probabilities=entry.temperature_daily_probabilities,
                after_horizon_probability=cast(
                    float, entry.temperature_after_horizon_probability
                ),
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ProspectiveJournalError(
            f"invalid forecast journal entry on line {line_number}: {error}"
        ) from error
    expected_cycle_day = (
        entry.prediction_date - entry.current_cycle_start_date
    ).days + 1
    if (
        entry.cycle_day != expected_cycle_day
        or entry.cycle_day < 1
        or not entry.probability_model_version
        or not entry.point_estimate_method
        or not isfinite(entry.point_estimate_cycle_length_days)
        or entry.point_estimate_cycle_length_days <= 0.0
        or entry.point_estimate_date < entry.current_cycle_start_date
        or entry.oura_synced_through < entry.prediction_date
        or re.fullmatch(r"sha256:[0-9a-f]{64}", entry.model_dataset_fingerprint) is None
        or (
            (entry.wearable_model_version is None)
            != (entry.wearable_daily_probabilities is None)
        )
        or (
            (entry.wearable_model_version is None)
            != (entry.wearable_after_horizon_probability is None)
        )
        or (
            (entry.temperature_model_version is None)
            != (entry.temperature_daily_probabilities is None)
        )
        or (
            (entry.temperature_model_version is None)
            != (entry.temperature_after_horizon_probability is None)
        )
    ):
        raise ProspectiveJournalError(
            f"invalid forecast journal entry on line {line_number}"
        )
    if entry.prediction_cutoff.date() != entry.prediction_date:
        raise ProspectiveJournalError(
            f"forecast journal cutoff date mismatch on line {line_number}"
        )
    return entry


def load_prospective_journal(*, path: Path) -> tuple[ProspectiveForecastEntry, ...]:
    """Load and validate every immutable entry in chronological order."""
    if not path.exists():
        return ()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProspectiveJournalError(
            f"could not read forecast journal: {error}"
        ) from error
    entries: list[ProspectiveForecastEntry] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ProspectiveJournalError(
                f"forecast journal line {line_number} is blank"
            )
        try:
            value = cast(object, json.loads(line))
        except json.JSONDecodeError as error:
            raise ProspectiveJournalError(
                f"invalid forecast journal JSON on line {line_number}"
            ) from error
        entries.append(_entry_from_object(value=value, line_number=line_number))
    prediction_dates = tuple(entry.prediction_date for entry in entries)
    if (
        len(set(prediction_dates)) != len(prediction_dates)
        or tuple(sorted(prediction_dates)) != prediction_dates
    ):
        raise ProspectiveJournalError(
            "forecast journal entries must be unique and chronological"
        )
    return tuple(entries)


def append_prospective_forecast(*, path: Path, entry: ProspectiveForecastEntry) -> bool:
    """Append the first forecast for a local date and preserve it on reruns."""
    entries = load_prospective_journal(path=path)
    if any(existing.prediction_date == entry.prediction_date for existing in entries):
        return False
    if entries and entry.prediction_date < entries[-1].prediction_date:
        raise ProspectiveJournalError("cannot append an older forecast date")
    ensure_private_directory(directory=path.parent)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for existing in (*entries, entry):
                temporary.write(_serialize_entry(entry=existing))
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(0o600)
        load_prospective_journal(path=temporary_path)
        os.replace(temporary_path, path)
    except OSError as error:
        raise ProspectiveJournalError(
            f"could not append forecast journal: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return True


def _next_start_by_cycle(
    *, history: tuple[CycleHistoryRecord, ...]
) -> dict[date, date]:
    """Map each completed cycle start to its observed following start."""
    return {
        current.cycle_start_date: following.cycle_start_date
        for current, following in pairwise(history)
    }


def summarize_prospective_performance(
    *,
    entries: tuple[ProspectiveForecastEntry, ...],
    history: tuple[CycleHistoryRecord, ...],
) -> ProspectivePerformanceSummary:
    """Score resolved forecasts and give each completed cycle equal weight."""
    next_starts = _next_start_by_cycle(history=history)
    grouped: dict[date, list[ProspectiveForecastEntry]] = {}
    for entry in entries:
        next_start = next_starts.get(entry.current_cycle_start_date)
        if next_start is None or entry.prediction_date > next_start:
            continue
        grouped.setdefault(entry.current_cycle_start_date, []).append(entry)
    cycle_evaluations: list[DailyForecastEvaluation] = []
    cycle_point_errors: list[float] = []
    wearable_cycle_evaluations: list[DailyForecastEvaluation] = []
    temperature_cycle_evaluations: list[DailyForecastEvaluation] = []
    wearable_resolved_count = 0
    temperature_resolved_count = 0
    resolved_count = 0
    for cycle_start, cycle_entries in grouped.items():
        next_start = next_starts[cycle_start]
        forecasts = tuple(
            DailyPeriodDistribution(
                prediction_date=entry.prediction_date,
                prediction_cutoff=entry.prediction_cutoff,
                daily_probabilities=entry.daily_probabilities,
                after_horizon_probability=entry.after_horizon_probability,
            )
            for entry in cycle_entries
        )
        outcomes = tuple(
            (next_start - entry.prediction_date).days for entry in cycle_entries
        )
        cycle_evaluations.append(
            evaluate_daily_distributions(
                forecasts=forecasts,
                outcome_offsets=outcomes,
            )
        )
        wearable_entries = tuple(
            entry
            for entry in cycle_entries
            if entry.wearable_daily_probabilities is not None
        )
        if wearable_entries:
            wearable_cycle_evaluations.append(
                evaluate_daily_distributions(
                    forecasts=tuple(
                        DailyPeriodDistribution(
                            prediction_date=entry.prediction_date,
                            prediction_cutoff=entry.prediction_cutoff,
                            daily_probabilities=cast(
                                tuple[float, ...], entry.wearable_daily_probabilities
                            ),
                            after_horizon_probability=cast(
                                float, entry.wearable_after_horizon_probability
                            ),
                        )
                        for entry in wearable_entries
                    ),
                    outcome_offsets=tuple(
                        (next_start - entry.prediction_date).days
                        for entry in wearable_entries
                    ),
                )
            )
            wearable_resolved_count += len(wearable_entries)
        temperature_entries = tuple(
            entry
            for entry in cycle_entries
            if entry.temperature_daily_probabilities is not None
        )
        if temperature_entries:
            temperature_cycle_evaluations.append(
                evaluate_daily_distributions(
                    forecasts=tuple(
                        DailyPeriodDistribution(
                            prediction_date=entry.prediction_date,
                            prediction_cutoff=entry.prediction_cutoff,
                            daily_probabilities=cast(
                                tuple[float, ...],
                                entry.temperature_daily_probabilities,
                            ),
                            after_horizon_probability=cast(
                                float, entry.temperature_after_horizon_probability
                            ),
                        )
                        for entry in temperature_entries
                    ),
                    outcome_offsets=tuple(
                        (next_start - entry.prediction_date).days
                        for entry in temperature_entries
                    ),
                )
            )
            temperature_resolved_count += len(temperature_entries)
        cycle_point_errors.append(
            sum(
                abs((entry.point_estimate_date - next_start).days)
                for entry in cycle_entries
            )
            / len(cycle_entries)
        )
        resolved_count += len(cycle_entries)
    cycle_count = len(cycle_evaluations)
    if not cycle_count:
        return ProspectivePerformanceSummary(
            journal_forecast_count=len(entries),
            resolved_forecast_count=0,
            completed_cycle_count=0,
            mean_cycle_logarithmic_loss=None,
            mean_cycle_brier_score=None,
            mean_cycle_window_brier_scores={},
            mean_cycle_point_absolute_error_days=None,
            wearable_resolved_forecast_count=0,
            wearable_completed_cycle_count=0,
            wearable_mean_cycle_logarithmic_loss=None,
            wearable_mean_cycle_brier_score=None,
            wearable_mean_cycle_window_brier_scores={},
            temperature_resolved_forecast_count=0,
            temperature_completed_cycle_count=0,
            temperature_mean_cycle_logarithmic_loss=None,
            temperature_mean_cycle_brier_score=None,
            temperature_mean_cycle_window_brier_scores={},
        )
    wearable_cycle_count = len(wearable_cycle_evaluations)
    temperature_cycle_count = len(temperature_cycle_evaluations)
    return ProspectivePerformanceSummary(
        journal_forecast_count=len(entries),
        resolved_forecast_count=resolved_count,
        completed_cycle_count=cycle_count,
        mean_cycle_logarithmic_loss=sum(
            evaluation.logarithmic_loss for evaluation in cycle_evaluations
        )
        / cycle_count,
        mean_cycle_brier_score=sum(
            evaluation.multiclass_brier_score for evaluation in cycle_evaluations
        )
        / cycle_count,
        mean_cycle_window_brier_scores={
            window: sum(
                evaluation.window_brier_scores[window]
                for evaluation in cycle_evaluations
            )
            / cycle_count
            for window in CALIBRATION_WINDOWS
        },
        mean_cycle_point_absolute_error_days=sum(cycle_point_errors) / cycle_count,
        wearable_resolved_forecast_count=wearable_resolved_count,
        wearable_completed_cycle_count=wearable_cycle_count,
        wearable_mean_cycle_logarithmic_loss=(
            sum(item.logarithmic_loss for item in wearable_cycle_evaluations)
            / wearable_cycle_count
            if wearable_cycle_count
            else None
        ),
        wearable_mean_cycle_brier_score=(
            sum(item.multiclass_brier_score for item in wearable_cycle_evaluations)
            / wearable_cycle_count
            if wearable_cycle_count
            else None
        ),
        wearable_mean_cycle_window_brier_scores=(
            {
                window: sum(
                    item.window_brier_scores[window]
                    for item in wearable_cycle_evaluations
                )
                / wearable_cycle_count
                for window in CALIBRATION_WINDOWS
            }
            if wearable_cycle_count
            else {}
        ),
        temperature_resolved_forecast_count=temperature_resolved_count,
        temperature_completed_cycle_count=temperature_cycle_count,
        temperature_mean_cycle_logarithmic_loss=(
            sum(item.logarithmic_loss for item in temperature_cycle_evaluations)
            / temperature_cycle_count
            if temperature_cycle_count
            else None
        ),
        temperature_mean_cycle_brier_score=(
            sum(item.multiclass_brier_score for item in temperature_cycle_evaluations)
            / temperature_cycle_count
            if temperature_cycle_count
            else None
        ),
        temperature_mean_cycle_window_brier_scores=(
            {
                window: sum(
                    item.window_brier_scores[window]
                    for item in temperature_cycle_evaluations
                )
                / temperature_cycle_count
                for window in CALIBRATION_WINDOWS
            }
            if temperature_cycle_count
            else {}
        ),
    )
