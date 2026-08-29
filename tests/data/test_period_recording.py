"""Tests for safe private period-history recording."""

from datetime import date
from pathlib import Path

import pytest

import cycle_forecast.data.period_recording as period_recording
from cycle_forecast.data.cycle_history import load_cycle_history
from cycle_forecast.data.period_recording import (
    PeriodRecordingError,
    record_period_start,
)


def test_create_history_with_ongoing_period(tmp_path: Path) -> None:
    """Create one valid private CSV without inventing the current duration."""
    history_path = tmp_path / "private/history.csv"

    result = record_period_start(
        history_path=history_path,
        cycle_start_date=date(2025, 1, 1),
        recorded_on=date(2025, 1, 1),
    )

    assert result.created_history
    assert result.record_count == 1
    assert load_cycle_history(path=history_path)[0].period_length_days is None
    assert history_path.read_text(encoding="utf-8") == (
        "cycle_start_date,period_length_days\n2025-01-01,\n"
    )


def test_complete_previous_period_and_append_new_start(tmp_path: Path) -> None:
    """Complete a pending duration while recording the next period start."""
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "cycle_start_date,period_length_days\n2025-01-01,\n",
        encoding="utf-8",
    )

    result = record_period_start(
        history_path=history_path,
        cycle_start_date=date(2025, 1, 30),
        recorded_on=date(2025, 1, 30),
        previous_period_length_days=5,
    )

    records = load_cycle_history(path=history_path)
    assert result.completed_previous_cycle_days == 29
    assert tuple(record.period_length_days for record in records) == (5, None)


def test_complete_existing_pending_period_without_duplicate(tmp_path: Path) -> None:
    """Allow the same start date to fill its previously unknown duration."""
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "cycle_start_date,period_length_days\n2025-01-01,\n",
        encoding="utf-8",
    )

    result = record_period_start(
        history_path=history_path,
        cycle_start_date=date(2025, 1, 1),
        recorded_on=date(2025, 1, 6),
        period_length_days=5,
    )

    assert result.completed_existing_period
    assert load_cycle_history(path=history_path)[0].period_length_days == 5


def test_reject_period_duration_beyond_recording_date(tmp_path: Path) -> None:
    """Prevent a duration from claiming bleeding days that have not occurred."""
    history_path = tmp_path / "history.csv"
    original = "cycle_start_date,period_length_days\n2025-01-01,\n"
    history_path.write_text(original, encoding="utf-8")

    with pytest.raises(PeriodRecordingError, match="maximum 5 days"):
        record_period_start(
            history_path=history_path,
            cycle_start_date=date(2025, 1, 1),
            recorded_on=date(2025, 1, 5),
            period_length_days=6,
        )

    assert history_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("start", "recorded_on", "previous_length", "message"),
    [
        (date(2025, 2, 2), date(2025, 2, 1), None, "future"),
        (date(2025, 1, 1), date(2025, 2, 1), None, "already recorded"),
        (date(2025, 1, 10), date(2025, 2, 1), None, "only 9 days"),
    ],
)
def test_reject_unsafe_updates_without_changing_history(
    tmp_path: Path,
    start: date,
    recorded_on: date,
    previous_length: int | None,
    message: str,
) -> None:
    """Keep the original bytes when validation rejects a requested update."""
    history_path = tmp_path / "history.csv"
    original = "cycle_start_date,period_length_days\n2025-01-01,5\n"
    history_path.write_text(original, encoding="utf-8")

    with pytest.raises(PeriodRecordingError, match=message):
        record_period_start(
            history_path=history_path,
            cycle_start_date=start,
            recorded_on=recorded_on,
            previous_period_length_days=previous_length,
        )

    assert history_path.read_text(encoding="utf-8") == original


def test_require_pending_previous_duration_before_new_start(tmp_path: Path) -> None:
    """Do not silently invent the duration of an earlier period."""
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "cycle_start_date,period_length_days\n2025-01-01,\n",
        encoding="utf-8",
    )

    with pytest.raises(PeriodRecordingError, match="still unknown"):
        record_period_start(
            history_path=history_path,
            cycle_start_date=date(2025, 2, 1),
            recorded_on=date(2025, 2, 1),
        )


def test_atomic_replace_failure_preserves_original_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave original bytes intact and remove temporary files on write failure."""
    history_path = tmp_path / "history.csv"
    original = "cycle_start_date,period_length_days\n2025-01-01,5\n"
    history_path.write_text(original, encoding="utf-8")

    def fail_replace(_: Path, __: Path) -> None:
        """Simulate an operating-system failure at the atomic replacement."""
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(period_recording.os, "replace", fail_replace)

    with pytest.raises(PeriodRecordingError, match="safely update"):
        record_period_start(
            history_path=history_path,
            cycle_start_date=date(2025, 2, 1),
            recorded_on=date(2025, 2, 1),
        )

    assert history_path.read_text(encoding="utf-8") == original
    assert tuple(tmp_path.glob(".*.tmp")) == ()
