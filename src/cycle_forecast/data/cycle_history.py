"""Load and validate raw cycle-history data."""

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


class CycleHistoryValidationError(ValueError):
    """Indicate that cycle-history data violates the raw-data contract."""


@dataclass(frozen=True, slots=True)
class CycleHistoryRecord:
    """Represent one raw cycle-history observation.

    Parameters
    ----------
    cycle_start_date
        First calendar day of the period.
    period_length_days
        Number of days in the period.
    """

    cycle_start_date: date
    period_length_days: int


def load_cycle_history(
    path: str | Path,
    *,
    minimum_cycle_days: int = 15,
) -> tuple[CycleHistoryRecord, ...]:
    """Load validated cycle-history records from a CSV file.

    Parameters
    ----------
    path
        CSV file containing ``cycle_start_date`` and ``period_length_days``.
    minimum_cycle_days
        Smallest accepted gap between consecutive cycle starts. This is a
        data-entry safeguard, not a medical definition.

    Returns
    -------
    tuple[CycleHistoryRecord, ...]
        Records in their original, strictly increasing order.

    Raises
    ------
    ValueError
        If ``minimum_cycle_days`` is not positive.
    CycleHistoryValidationError
        If the file does not satisfy the raw cycle-history contract.
    """
    if minimum_cycle_days < 1:
        message = "minimum_cycle_days must be positive"
        raise ValueError(message)

    csv_path = Path(path)
    try:
        file_handle = csv_path.open(encoding="utf-8", newline="")
    except OSError as error:
        message = f"Could not read cycle-history data from {csv_path}: {error}"
        raise CycleHistoryValidationError(message) from error

    with file_handle:
        reader = csv.reader(file_handle)
        try:
            header = next(reader)
        except StopIteration as error:
            message = "CSV header must be cycle_start_date,period_length_days"
            raise CycleHistoryValidationError(message) from error
        if header != ["cycle_start_date", "period_length_days"]:
            message = "CSV header must be cycle_start_date,period_length_days"
            raise CycleHistoryValidationError(message)

        records: list[CycleHistoryRecord] = []
        for line_number, row in enumerate(reader, start=2):
            if not row:
                message = f"Missing cycle-history values on line {line_number}"
                raise CycleHistoryValidationError(message)
            if len(row) != 2:
                message = (
                    f"Expected exactly two cycle-history values on line {line_number}"
                )
                raise CycleHistoryValidationError(message)

            raw_date, raw_period_length = row
            if not raw_date.strip() or not raw_period_length.strip():
                message = f"Missing cycle-history value on line {line_number}"
                raise CycleHistoryValidationError(message)

            try:
                cycle_start_date = date.fromisoformat(raw_date)
            except ValueError as error:
                message = (
                    f"Invalid cycle_start_date on line {line_number}: {raw_date!r}; "
                    "expected YYYY-MM-DD"
                )
                raise CycleHistoryValidationError(message) from error
            if raw_date != cycle_start_date.isoformat():
                message = (
                    f"Invalid cycle_start_date on line {line_number}: {raw_date!r}; "
                    "expected YYYY-MM-DD"
                )
                raise CycleHistoryValidationError(message)

            try:
                period_length = int(raw_period_length)
            except ValueError as error:
                message = (
                    f"Invalid period_length_days on line {line_number}: "
                    f"{raw_period_length!r}; expected a positive whole number"
                )
                raise CycleHistoryValidationError(message) from error
            if str(period_length) != raw_period_length or period_length < 1:
                message = (
                    f"Invalid period_length_days on line {line_number}: "
                    f"{raw_period_length!r}; expected a positive whole number"
                )
                raise CycleHistoryValidationError(message)

            if records:
                previous_start = records[-1].cycle_start_date
                gap_days = (cycle_start_date - previous_start).days
                if gap_days == 0:
                    message = (
                        f"Duplicate cycle_start_date on line {line_number}: {raw_date}"
                    )
                    raise CycleHistoryValidationError(message)
                if gap_days < 0:
                    message = (
                        "cycle_start_date values must be in strictly increasing order; "
                        f"line {line_number} contains {raw_date} after "
                        f"{previous_start.isoformat()}"
                    )
                    raise CycleHistoryValidationError(message)
                if gap_days < minimum_cycle_days:
                    message = (
                        f"Cycle starts on lines {line_number - 1} and {line_number} "
                        f"are only {gap_days} days apart; expected at least "
                        f"{minimum_cycle_days}"
                    )
                    raise CycleHistoryValidationError(message)

                previous_period_length = records[-1].period_length_days
                if previous_period_length > gap_days:
                    message = (
                        f"period_length_days on line {line_number - 1} is "
                        f"{previous_period_length}, which extends beyond the next "
                        f"cycle start on line {line_number} after {gap_days} days"
                    )
                    raise CycleHistoryValidationError(message)

            records.append(
                CycleHistoryRecord(
                    cycle_start_date=cycle_start_date,
                    period_length_days=period_length,
                )
            )

    if not records:
        message = "CSV must contain at least one cycle-history record"
        raise CycleHistoryValidationError(message)

    return tuple(records)
