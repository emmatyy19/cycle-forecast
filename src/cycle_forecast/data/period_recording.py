"""Safely record period starts in one private local cycle-history file."""

import csv
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from cycle_forecast.data.cycle_history import (
    CycleHistoryRecord,
    CycleHistoryValidationError,
    load_cycle_history,
)

DEFAULT_MINIMUM_CYCLE_DAYS = 15
"""Data-entry safeguard for unusually close period starts."""


class PeriodRecordingError(ValueError):
    """Indicate that a requested history update is unsafe or incomplete."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PeriodRecordingResult:
    """Describe a successful private history update without exposing all rows."""

    history_path: Path
    cycle_start_date: date
    completed_previous_cycle_days: int | None
    period_length_days: int | None
    record_count: int
    created_history: bool
    completed_existing_period: bool


def _validate_period_length(*, value: int | None, label: str) -> None:
    """Require a positive duration when a period length is supplied."""
    if value is not None and value < 1:
        raise PeriodRecordingError(f"{label} must be a positive whole number")


def _write_history_atomically(
    *, path: Path, records: tuple[CycleHistoryRecord, ...]
) -> None:
    """Replace one history CSV atomically after writing a validated temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.writer(temporary, lineterminator="\n")
            writer.writerow(("cycle_start_date", "period_length_days"))
            writer.writerows(
                (
                    record.cycle_start_date.isoformat(),
                    record.period_length_days
                    if record.period_length_days is not None
                    else "",
                )
                for record in records
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        load_cycle_history(path=temporary_path)
        os.replace(temporary_path, path)
    except OSError as error:
        raise PeriodRecordingError(
            f"Could not safely update {path}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def record_period_start(
    *,
    history_path: Path,
    cycle_start_date: date,
    recorded_on: date,
    period_length_days: int | None = None,
    previous_period_length_days: int | None = None,
    minimum_cycle_days: int = DEFAULT_MINIMUM_CYCLE_DAYS,
) -> PeriodRecordingResult:
    """Record a new start or complete the newest period in a private CSV.

    Parameters
    ----------
    history_path
        Private cycle-history CSV to create or update.
    cycle_start_date
        First calendar day of the period being recorded.
    recorded_on
        Current local date, supplied explicitly for deterministic validation.
    period_length_days
        Known duration of the period being recorded, or ``None`` while ongoing.
    previous_period_length_days
        Duration used to complete the prior pending row before a new row is added.
    minimum_cycle_days
        Smallest accepted gap between consecutive starts.

    Returns
    -------
    PeriodRecordingResult
        Privacy-safe details of the completed atomic update.

    Raises
    ------
    PeriodRecordingError
        If the date, durations, or existing history make the update unsafe.
    """
    if minimum_cycle_days < 1:
        raise PeriodRecordingError("minimum_cycle_days must be positive")
    _validate_period_length(value=period_length_days, label="period length")
    _validate_period_length(
        value=previous_period_length_days,
        label="previous period length",
    )
    if cycle_start_date > recorded_on:
        raise PeriodRecordingError("period start date cannot be in the future")
    elapsed_calendar_days = (recorded_on - cycle_start_date).days + 1
    if period_length_days is not None and period_length_days > elapsed_calendar_days:
        raise PeriodRecordingError(
            "period length cannot extend beyond the recording date; "
            f"maximum {elapsed_calendar_days} days"
        )

    created_history = not history_path.exists()
    if created_history:
        records: tuple[CycleHistoryRecord, ...] = ()
    else:
        try:
            records = load_cycle_history(
                path=history_path,
                minimum_cycle_days=minimum_cycle_days,
            )
        except CycleHistoryValidationError as error:
            raise PeriodRecordingError(str(error)) from error

    if not records:
        updated = (
            CycleHistoryRecord(
                cycle_start_date=cycle_start_date,
                period_length_days=period_length_days,
            ),
        )
        completed_existing = False
        completed_cycle_days = None
    else:
        latest = records[-1]
        if cycle_start_date < latest.cycle_start_date:
            raise PeriodRecordingError(
                "period start must be on or after the newest recorded start "
                f"({latest.cycle_start_date.isoformat()})"
            )
        if cycle_start_date == latest.cycle_start_date:
            if latest.period_length_days is not None:
                raise PeriodRecordingError("that period start is already recorded")
            if period_length_days is None:
                raise PeriodRecordingError(
                    "that start is already recorded; provide its period length to "
                    "complete it"
                )
            updated = (
                *records[:-1],
                replace(latest, period_length_days=period_length_days),
            )
            completed_existing = True
            completed_cycle_days = None
        else:
            cycle_length = (cycle_start_date - latest.cycle_start_date).days
            if cycle_length < minimum_cycle_days:
                raise PeriodRecordingError(
                    f"new start is only {cycle_length} days after the previous start; "
                    f"expected at least {minimum_cycle_days}"
                )
            prior_length = latest.period_length_days
            if prior_length is None:
                if previous_period_length_days is None:
                    raise PeriodRecordingError(
                        "the previous period length is still unknown"
                    )
                prior_length = previous_period_length_days
            if prior_length > cycle_length:
                raise PeriodRecordingError(
                    "previous period length cannot exceed the completed cycle length"
                )
            completed_latest = replace(latest, period_length_days=prior_length)
            updated = (
                *records[:-1],
                completed_latest,
                CycleHistoryRecord(
                    cycle_start_date=cycle_start_date,
                    period_length_days=period_length_days,
                ),
            )
            completed_existing = False
            completed_cycle_days = cycle_length

    _write_history_atomically(path=history_path, records=updated)
    return PeriodRecordingResult(
        history_path=history_path,
        cycle_start_date=cycle_start_date,
        completed_previous_cycle_days=completed_cycle_days,
        period_length_days=period_length_days,
        record_count=len(updated),
        created_history=created_history,
        completed_existing_period=completed_existing,
    )
