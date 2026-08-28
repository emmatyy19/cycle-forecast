"""Tests for immutable private prospective forecasts and delayed scoring."""

from datetime import UTC, date, datetime
from pathlib import Path
from stat import S_IMODE

import pytest

import cycle_forecast.evaluation.prospective_journal as prospective_journal
from cycle_forecast.data import CycleHistoryRecord
from cycle_forecast.evaluation.prospective_journal import (
    PROSPECTIVE_JOURNAL_SCHEMA_VERSION,
    ProspectiveForecastEntry,
    ProspectiveJournalError,
    append_prospective_forecast,
    load_prospective_journal,
    summarize_prospective_performance,
)


def _entry(
    *,
    prediction_date: date,
    cycle_start: date,
    actual_offset: int,
    perfect: bool,
    point_estimate_date: date,
) -> ProspectiveForecastEntry:
    """Build an invented exhaustive journal entry for one morning."""
    probabilities = [0.0] * 15
    after = 0.0
    if perfect:
        probabilities[actual_offset] = 1.0
    else:
        after = 1.0
    return ProspectiveForecastEntry(
        schema_version=PROSPECTIVE_JOURNAL_SCHEMA_VERSION,
        prediction_date=prediction_date,
        prediction_cutoff=datetime.combine(
            prediction_date,
            datetime.min.time().replace(hour=9),
            tzinfo=UTC,
        ),
        current_cycle_start_date=cycle_start,
        cycle_day=(prediction_date - cycle_start).days + 1,
        probability_model_version="synthetic-history-v1",
        daily_probabilities=tuple(probabilities),
        after_horizon_probability=after,
        point_estimate_date=point_estimate_date,
        point_estimate_cycle_length_days=30.0,
        point_estimate_method="phase-a-model",
        model_dataset_fingerprint="sha256:" + "a" * 64,
        oura_synced_through=prediction_date,
    )


def test_append_first_daily_forecast_idempotently_and_privately(tmp_path: Path) -> None:
    """Keep the first forecast for a date and use owner-only file permissions."""
    journal_path = tmp_path / "private/forecast-journal.jsonl"
    first = _entry(
        prediction_date=date(2025, 1, 29),
        cycle_start=date(2025, 1, 1),
        actual_offset=2,
        perfect=True,
        point_estimate_date=date(2025, 1, 31),
    )
    rerun = _entry(
        prediction_date=date(2025, 1, 29),
        cycle_start=date(2025, 1, 1),
        actual_offset=2,
        perfect=False,
        point_estimate_date=date(2025, 2, 2),
    )

    assert append_prospective_forecast(path=journal_path, entry=first)
    assert not append_prospective_forecast(path=journal_path, entry=rerun)

    assert load_prospective_journal(path=journal_path) == (first,)
    assert S_IMODE(journal_path.stat().st_mode) == 0o600
    assert S_IMODE(journal_path.parent.stat().st_mode) == 0o700


def test_reject_invalid_or_nonchronological_journal(tmp_path: Path) -> None:
    """Do not append around malformed private evaluation history."""
    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text("not json\n", encoding="utf-8")
    entry = _entry(
        prediction_date=date(2025, 1, 29),
        cycle_start=date(2025, 1, 1),
        actual_offset=2,
        perfect=True,
        point_estimate_date=date(2025, 1, 31),
    )

    with pytest.raises(ProspectiveJournalError, match="JSON"):
        append_prospective_forecast(path=journal_path, entry=entry)


def test_failed_atomic_replace_preserves_original_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep prior prospective evidence intact when final replacement fails."""
    journal_path = tmp_path / "private/journal.jsonl"
    first = _entry(
        prediction_date=date(2025, 1, 29),
        cycle_start=date(2025, 1, 1),
        actual_offset=2,
        perfect=True,
        point_estimate_date=date(2025, 1, 31),
    )
    second = _entry(
        prediction_date=date(2025, 1, 30),
        cycle_start=date(2025, 1, 1),
        actual_offset=1,
        perfect=True,
        point_estimate_date=date(2025, 1, 31),
    )
    append_prospective_forecast(path=journal_path, entry=first)
    original = journal_path.read_bytes()

    def fail_replace(_: Path, __: Path) -> None:
        """Simulate an operating-system failure during final replacement."""
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(prospective_journal.os, "replace", fail_replace)

    with pytest.raises(ProspectiveJournalError, match="could not append"):
        append_prospective_forecast(path=journal_path, entry=second)

    assert journal_path.read_bytes() == original
    assert tuple(journal_path.parent.glob(".*.tmp")) == ()


def test_delayed_scores_weight_completed_cycles_equally() -> None:
    """Prevent a cycle with more journaled mornings from dominating metrics."""
    first_start = date(2025, 1, 1)
    second_start = date(2025, 1, 31)
    third_start = date(2025, 3, 2)
    history = tuple(
        CycleHistoryRecord(cycle_start_date=start, period_length_days=5)
        for start in (first_start, second_start, third_start)
    )
    entries = (
        _entry(
            prediction_date=date(2025, 1, 29),
            cycle_start=first_start,
            actual_offset=2,
            perfect=True,
            point_estimate_date=second_start,
        ),
        _entry(
            prediction_date=date(2025, 1, 30),
            cycle_start=first_start,
            actual_offset=1,
            perfect=True,
            point_estimate_date=second_start,
        ),
        _entry(
            prediction_date=date(2025, 3, 1),
            cycle_start=second_start,
            actual_offset=1,
            perfect=False,
            point_estimate_date=date(2025, 3, 4),
        ),
    )

    summary = summarize_prospective_performance(entries=entries, history=history)

    assert summary.journal_forecast_count == 3
    assert summary.resolved_forecast_count == 3
    assert summary.completed_cycle_count == 2
    assert summary.mean_cycle_brier_score == pytest.approx(1.0)
    assert summary.mean_cycle_point_absolute_error_days == pytest.approx(1.0)
