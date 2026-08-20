"""Tests for cycle-history loading and validation."""

from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest

from cycle_forecast.data.cycle_history import (
    CycleHistoryRecord,
    CycleHistoryValidationError,
    load_cycle_history,
)


def test_load_synthetic_cycle_history() -> None:
    """Load the committed synthetic example in chronological order."""
    records = load_cycle_history("data/synthetic/sample_cycle_history.csv")

    assert len(records) == 12
    assert records[0] == CycleHistoryRecord(
        cycle_start_date=date(2024, 1, 3),
        period_length_days=6,
    )
    assert records[-1] == CycleHistoryRecord(
        cycle_start_date=date(2024, 11, 12),
        period_length_days=6,
    )


@pytest.mark.parametrize(
    ("contents", "expected_message"),
    [
        ("", "header"),
        ("wrong_column\n2024-01-01\n", "header"),
        ("cycle_start_date,period_length_days\n", "at least one"),
        ("cycle_start_date,period_length_days\n\n", "Missing cycle-history"),
        ('cycle_start_date,period_length_days\n"",""\n', "Missing cycle-history"),
        (
            "cycle_start_date,period_length_days\n2024-01-01,7\n\n2024-02-01,7\n",
            "Missing cycle-history",
        ),
        (
            "cycle_start_date,period_length_days\n2024-01-01,7,extra\n",
            "exactly two",
        ),
        (
            "cycle_start_date,period_length_days\nnot-a-date,7\n",
            "Invalid cycle_start_date",
        ),
        (
            "cycle_start_date,period_length_days\n2024-1-01,7\n",
            "expected YYYY-MM-DD",
        ),
        (
            "cycle_start_date,period_length_days\n 2024-01-01,7\n",
            "expected YYYY-MM-DD",
        ),
        (
            "cycle_start_date,period_length_days\n2024-01-01,0\n",
            "positive whole number",
        ),
        (
            "cycle_start_date,period_length_days\n2024-01-01,-1\n",
            "positive whole number",
        ),
        (
            "cycle_start_date,period_length_days\n,7\n",
            "Missing cycle-history",
        ),
        (
            "cycle_start_date,period_length_days\n2024-01-01,\n",
            "Missing cycle-history",
        ),
        (
            "period_length_days,cycle_start_date\n7,2024-01-01\n",
            "header",
        ),
        (
            "cycle_start_date,period_length_days\n2024-01-01,7.5\n",
            "positive whole number",
        ),
        (
            "cycle_start_date,period_length_days\n2024-01-01,7\n2024-01-01,7\n",
            "Duplicate cycle_start_date",
        ),
        (
            "cycle_start_date,period_length_days\n2024-02-01,7\n2024-01-01,7\n",
            "strictly increasing",
        ),
        (
            "cycle_start_date,period_length_days\n2024-01-01,7\n2024-01-10,7\n",
            "only 9 days apart",
        ),
        (
            "cycle_start_date,period_length_days\n2024-01-01,32\n2024-02-01,7\n",
            "extends beyond the next cycle start",
        ),
    ],
)
def test_reject_invalid_cycle_history(
    tmp_path: Path,
    contents: str,
    expected_message: str,
) -> None:
    """Reject files that violate the documented raw-data contract."""
    csv_path = tmp_path / "cycle_history.csv"
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(CycleHistoryValidationError, match=expected_message):
        load_cycle_history(csv_path)


def test_allow_configurable_minimum_gap(tmp_path: Path) -> None:
    """Allow callers to adjust the data-entry safeguard explicitly."""
    csv_path = tmp_path / "cycle_history.csv"
    csv_path.write_text(
        "cycle_start_date,period_length_days\n2024-01-01,7\n2024-01-10,6\n",
        encoding="utf-8",
    )

    assert load_cycle_history(csv_path, minimum_cycle_days=9) == (
        CycleHistoryRecord(
            cycle_start_date=date(2024, 1, 1),
            period_length_days=7,
        ),
        CycleHistoryRecord(
            cycle_start_date=date(2024, 1, 10),
            period_length_days=6,
        ),
    )


def test_reject_nonpositive_minimum_gap() -> None:
    """Reject a nonsensical minimum-gap configuration."""
    with pytest.raises(ValueError, match="must be positive"):
        load_cycle_history("unused.csv", minimum_cycle_days=0)


def test_wrap_missing_file_error(tmp_path: Path) -> None:
    """Report unreadable input using the domain validation exception."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(CycleHistoryValidationError, match="Could not read"):
        load_cycle_history(missing_path)


def test_cycle_history_record_is_frozen() -> None:
    """Prevent mutation of a validated cycle-history record."""
    record = CycleHistoryRecord(
        cycle_start_date=date(2024, 1, 1),
        period_length_days=7,
    )
    field_name = "period_length_days"

    with pytest.raises(FrozenInstanceError):
        setattr(record, field_name, 8)
