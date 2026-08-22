"""Tests for cycle-history loading and validation."""

from dataclasses import FrozenInstanceError
from datetime import date
from inspect import Parameter, signature
from pathlib import Path

import pytest

from cycle_forecast.data.cycle_history import (
    CYCLE_DATASET_TRANSFORMATION_VERSION,
    CycleDataset,
    CycleDatasetRow,
    CycleHistoryRecord,
    CycleHistoryValidationError,
    build_cycle_dataset,
    fingerprint_cycle_dataset,
    load_cycle_history,
)


def test_load_synthetic_cycle_history() -> None:
    """Load the committed synthetic example in chronological order."""
    records = load_cycle_history(path="data/synthetic/sample_cycle_history.csv")

    assert len(records) == 121
    assert records[0] == CycleHistoryRecord(
        cycle_start_date=date(2015, 1, 4),
        period_length_days=6,
    )
    assert records[-1] == CycleHistoryRecord(
        cycle_start_date=date(2025, 2, 7),
        period_length_days=6,
    )
    assert len(build_cycle_dataset(records=records).rows) == 120


def test_public_data_api_requires_keyword_arguments() -> None:
    """Keep reusable data functions and records explicit at call sites."""
    assert signature(load_cycle_history).parameters["path"].kind is (
        Parameter.KEYWORD_ONLY
    )
    assert signature(build_cycle_dataset).parameters["records"].kind is (
        Parameter.KEYWORD_ONLY
    )
    assert signature(fingerprint_cycle_dataset).parameters["records"].kind is (
        Parameter.KEYWORD_ONLY
    )
    assert signature(CycleHistoryRecord).parameters["cycle_start_date"].kind is (
        Parameter.KEYWORD_ONLY
    )
    assert signature(CycleDatasetRow).parameters["cycle_start_date"].kind is (
        Parameter.KEYWORD_ONLY
    )
    assert signature(CycleDataset).parameters["rows"].kind is Parameter.KEYWORD_ONLY


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
        load_cycle_history(path=csv_path)


def test_allow_configurable_minimum_gap(tmp_path: Path) -> None:
    """Allow callers to adjust the data-entry safeguard explicitly."""
    csv_path = tmp_path / "cycle_history.csv"
    csv_path.write_text(
        "cycle_start_date,period_length_days\n2024-01-01,7\n2024-01-10,6\n",
        encoding="utf-8",
    )

    assert load_cycle_history(path=csv_path, minimum_cycle_days=9) == (
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
        load_cycle_history(path="unused.csv", minimum_cycle_days=0)


def test_wrap_missing_file_error(tmp_path: Path) -> None:
    """Report unreadable input using the domain validation exception."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(CycleHistoryValidationError, match="Could not read"):
        load_cycle_history(path=missing_path)


def test_cycle_history_record_is_frozen() -> None:
    """Prevent mutation of a validated cycle-history record."""
    record = CycleHistoryRecord(
        cycle_start_date=date(2024, 1, 1),
        period_length_days=7,
    )
    field_name = "period_length_days"

    with pytest.raises(FrozenInstanceError):
        setattr(record, field_name, 8)


def test_build_cycle_dataset_from_consecutive_starts() -> None:
    """Create chronological targets from consecutive validated starts."""
    records = load_cycle_history(path="data/synthetic/sample_cycle_history.csv")

    dataset = build_cycle_dataset(records=records)

    assert len(dataset.rows) == len(records) - 1
    assert dataset.rows[0] == CycleDatasetRow(
        cycle_start_date=date(2015, 1, 4),
        next_cycle_start_date=date(2015, 2, 9),
        cycle_length_days=36,
    )
    assert dataset.rows[-1] == CycleDatasetRow(
        cycle_start_date=date(2025, 1, 12),
        next_cycle_start_date=date(2025, 2, 7),
        cycle_length_days=26,
    )


@pytest.mark.parametrize("record_count", [0, 1])
def test_build_cycle_dataset_requires_a_completed_pair(record_count: int) -> None:
    """Return no targets when no cycle has a known following start."""
    records = (
        CycleHistoryRecord(
            cycle_start_date=date(2024, 1, 1),
            period_length_days=6,
        ),
    )[:record_count]

    assert build_cycle_dataset(records=records).rows == ()


def test_cycle_dataset_row_is_frozen() -> None:
    """Prevent mutation of a derived target row."""
    row = CycleDatasetRow(
        cycle_start_date=date(2024, 1, 1),
        next_cycle_start_date=date(2024, 1, 29),
        cycle_length_days=28,
    )
    field_name = "cycle_length_days"

    with pytest.raises(FrozenInstanceError):
        setattr(row, field_name, 29)


def test_build_cycle_dataset_records_provenance() -> None:
    """Attach stable transformation and input identities to derived rows."""
    records = load_cycle_history(path="data/synthetic/sample_cycle_history.csv")

    first_dataset = build_cycle_dataset(records=records)
    second_dataset = build_cycle_dataset(records=records)

    assert first_dataset == second_dataset
    assert first_dataset.transformation_version == (
        CYCLE_DATASET_TRANSFORMATION_VERSION
    )
    assert first_dataset.fingerprint.startswith("sha256:")
    assert len(first_dataset.fingerprint.removeprefix("sha256:")) == 64


def test_cycle_dataset_fingerprint_has_stable_golden_value() -> None:
    """Detect accidental changes to the canonical fingerprint payload."""
    records = (
        CycleHistoryRecord(
            cycle_start_date=date(2024, 1, 3),
            period_length_days=6,
        ),
        CycleHistoryRecord(
            cycle_start_date=date(2024, 1, 31),
            period_length_days=7,
        ),
    )

    assert (
        fingerprint_cycle_dataset(
            records=records,
            transformation_version="cycle-dataset-v1",
        )
        == "sha256:c05fd92198797dc05521c8ac45ba3c7bbb86c8a27754cf1d01a8b43511ce7a39"
    )


def test_cycle_dataset_fingerprint_covers_all_validated_input_fields() -> None:
    """Change identity when either a start date or period length changes."""
    records = (
        CycleHistoryRecord(
            cycle_start_date=date(2024, 1, 3),
            period_length_days=6,
        ),
        CycleHistoryRecord(
            cycle_start_date=date(2024, 1, 31),
            period_length_days=7,
        ),
    )
    changed_date = (
        records[0],
        CycleHistoryRecord(
            cycle_start_date=date(2024, 2, 1),
            period_length_days=7,
        ),
    )
    changed_period_length = (
        CycleHistoryRecord(
            cycle_start_date=date(2024, 1, 3),
            period_length_days=5,
        ),
        records[1],
    )

    original = fingerprint_cycle_dataset(
        records=records,
        transformation_version="cycle-dataset-v1",
    )

    assert (
        fingerprint_cycle_dataset(
            records=changed_date,
            transformation_version="cycle-dataset-v1",
        )
        != original
    )
    assert (
        fingerprint_cycle_dataset(
            records=changed_period_length,
            transformation_version="cycle-dataset-v1",
        )
        != original
    )


def test_cycle_dataset_fingerprint_covers_transformation_version() -> None:
    """Change identity when dataset semantics receive a new version."""
    records = (
        CycleHistoryRecord(
            cycle_start_date=date(2024, 1, 3),
            period_length_days=6,
        ),
    )

    version_one = fingerprint_cycle_dataset(
        records=records,
        transformation_version="cycle-dataset-v1",
    )
    version_two = fingerprint_cycle_dataset(
        records=records,
        transformation_version="cycle-dataset-v2",
    )

    assert version_one != version_two


@pytest.mark.parametrize("version", ["", "v1\nforged", "v1\rforged"])
def test_reject_invalid_transformation_version(version: str) -> None:
    """Reject versions that would make canonical payload boundaries ambiguous."""
    with pytest.raises(ValueError, match="transformation_version"):
        fingerprint_cycle_dataset(records=(), transformation_version=version)


def test_cycle_dataset_is_frozen() -> None:
    """Keep provenance attached to the exact immutable rows it identifies."""
    dataset = CycleDataset(
        rows=(),
        transformation_version="cycle-dataset-v1",
        fingerprint="sha256:example",
    )
    field_name = "fingerprint"

    with pytest.raises(FrozenInstanceError):
        setattr(dataset, field_name, "sha256:changed")
