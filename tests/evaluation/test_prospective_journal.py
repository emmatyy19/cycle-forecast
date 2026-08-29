"""Tests for immutable private prospective forecasts and delayed scoring."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from stat import S_IMODE

import pytest

import cycle_forecast.evaluation.prospective_journal as prospective_journal
from cycle_forecast.data import CycleHistoryRecord
from cycle_forecast.evaluation.prospective_journal import (
    PROSPECTIVE_JOURNAL_SCHEMA_VERSION,
    ProspectiveForecastEntry,
    ProspectiveJournalError,
    WearablePromotionStatus,
    append_prospective_forecast,
    load_prospective_journal,
    summarize_prospective_performance,
)
from cycle_forecast.forecasting.wearable_baselines import (
    STAGE_AWARE_TEMPERATURE_BLEND_VERSION,
)


def _entry(
    *,
    prediction_date: date,
    cycle_start: date,
    actual_offset: int,
    perfect: bool,
    point_estimate_date: date,
    wearable: bool = False,
    temperature: bool = False,
    temperature_perfect: bool | None = None,
) -> ProspectiveForecastEntry:
    """Build an invented exhaustive journal entry for one morning."""
    probabilities = [0.0] * 15
    after = 0.0
    if perfect:
        probabilities[actual_offset] = 1.0
    else:
        after = 1.0
    temperature_probabilities = [0.0] * 15
    temperature_after = 0.0
    candidate_is_perfect = (
        temperature_perfect if temperature_perfect is not None else perfect
    )
    if candidate_is_perfect:
        temperature_probabilities[actual_offset] = 1.0
    else:
        temperature_after = 1.0
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
        wearable_model_version="synthetic-wearable-v1" if wearable else None,
        wearable_daily_probabilities=tuple(probabilities) if wearable else None,
        wearable_after_horizon_probability=after if wearable else None,
        temperature_model_version=(
            STAGE_AWARE_TEMPERATURE_BLEND_VERSION if temperature else None
        ),
        temperature_daily_probabilities=(
            tuple(temperature_probabilities) if temperature else None
        ),
        temperature_after_horizon_probability=(
            temperature_after if temperature else None
        ),
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


def test_append_preserves_readable_version_one_entries(tmp_path: Path) -> None:
    """Upgrade new rows without rewriting the schema of prior evidence."""
    journal_path = tmp_path / "private/journal.jsonl"
    first = replace(
        _entry(
            prediction_date=date(2025, 1, 29),
            cycle_start=date(2025, 1, 1),
            actual_offset=2,
            perfect=True,
            point_estimate_date=date(2025, 1, 31),
        ),
        schema_version="prospective-forecast-journal-v1",
    )
    second = _entry(
        prediction_date=date(2025, 1, 30),
        cycle_start=date(2025, 1, 1),
        actual_offset=1,
        perfect=True,
        point_estimate_date=date(2025, 1, 31),
        wearable=True,
    )

    assert append_prospective_forecast(path=journal_path, entry=first)
    assert append_prospective_forecast(path=journal_path, entry=second)

    assert load_prospective_journal(path=journal_path) == (first, second)


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
            wearable=True,
            temperature=True,
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
    assert summary.wearable_resolved_forecast_count == 1
    assert summary.wearable_completed_cycle_count == 1
    assert summary.wearable_mean_cycle_brier_score == pytest.approx(0.0)
    assert summary.temperature_resolved_forecast_count == 1
    assert summary.temperature_completed_cycle_count == 1
    assert summary.temperature_mean_cycle_brier_score == pytest.approx(0.0)


def test_empty_journal_has_unresolved_summary() -> None:
    """Represent the prospective waiting period without invented metrics."""
    summary = summarize_prospective_performance(entries=(), history=())

    assert summary.journal_forecast_count == 0
    assert summary.completed_cycle_count == 0
    assert summary.mean_cycle_brier_score is None
    assert summary.wearable_completed_cycle_count == 0
    assert summary.temperature_completed_cycle_count == 0


@pytest.mark.parametrize(
    ("history_perfect", "candidate_perfect", "expected_status"),
    (
        (False, True, WearablePromotionStatus.PROMOTE),
        (True, False, WearablePromotionStatus.REJECT),
    ),
)
def test_promotion_review_applies_paired_predeclared_score_rules(
    *,
    history_perfect: bool,
    candidate_perfect: bool,
    expected_status: WearablePromotionStatus,
) -> None:
    """Promote or reject the frozen candidate from three paired cycles."""
    starts = (
        date(2025, 1, 1),
        date(2025, 1, 31),
        date(2025, 3, 2),
        date(2025, 4, 1),
    )
    history = tuple(
        CycleHistoryRecord(cycle_start_date=start, period_length_days=5)
        for start in starts
    )
    entries = tuple(
        _entry(
            prediction_date=next_start - timedelta(days=1),
            cycle_start=cycle_start,
            actual_offset=1,
            perfect=history_perfect,
            point_estimate_date=next_start,
            temperature=True,
            temperature_perfect=candidate_perfect,
        )
        for cycle_start, next_start in pairwise(starts)
    )

    review = summarize_prospective_performance(
        entries=entries, history=history
    ).promotion_review

    assert review.status is expected_status
    assert review.eligible_cycle_count == 3
    assert review.paired_forecast_count == 3
    assert review.cycle_availability_rates == (1.0, 1.0, 1.0)
    expected_wins = 3 if candidate_perfect else 0
    assert review.candidate_logarithmic_loss_cycle_wins == expected_wins


def test_promotion_review_requires_minimum_candidate_availability() -> None:
    """Keep a strong but operationally sparse candidate inconclusive."""
    starts = (
        date(2025, 1, 1),
        date(2025, 1, 31),
        date(2025, 3, 2),
        date(2025, 4, 1),
    )
    history = tuple(
        CycleHistoryRecord(cycle_start_date=start, period_length_days=5)
        for start in starts
    )
    entries: list[ProspectiveForecastEntry] = []
    for cycle_start, next_start in pairwise(starts):
        entries.extend(
            (
                _entry(
                    prediction_date=next_start - timedelta(days=2),
                    cycle_start=cycle_start,
                    actual_offset=2,
                    perfect=False,
                    point_estimate_date=next_start,
                ),
                _entry(
                    prediction_date=next_start - timedelta(days=1),
                    cycle_start=cycle_start,
                    actual_offset=1,
                    perfect=False,
                    point_estimate_date=next_start,
                    temperature=True,
                    temperature_perfect=True,
                ),
            )
        )

    review = summarize_prospective_performance(
        entries=tuple(entries), history=history
    ).promotion_review

    assert review.status is WearablePromotionStatus.INCONCLUSIVE
    assert review.cycle_availability_rates == (0.5, 0.5, 0.5)


def test_promotion_review_waits_for_three_eligible_cycles() -> None:
    """Report insufficient evidence even when two cycles favor the candidate."""
    starts = (date(2025, 1, 1), date(2025, 1, 31), date(2025, 3, 2))
    history = tuple(
        CycleHistoryRecord(cycle_start_date=start, period_length_days=5)
        for start in starts
    )
    entries = tuple(
        _entry(
            prediction_date=next_start - timedelta(days=1),
            cycle_start=cycle_start,
            actual_offset=1,
            perfect=False,
            point_estimate_date=next_start,
            temperature=True,
            temperature_perfect=True,
        )
        for cycle_start, next_start in pairwise(starts)
    )

    review = summarize_prospective_performance(
        entries=entries, history=history
    ).promotion_review

    assert review.status is WearablePromotionStatus.INSUFFICIENT_EVIDENCE
    assert review.eligible_cycle_count == 2
