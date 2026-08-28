"""Test the local private wearable evaluation workflow end to end."""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from cycle_forecast.data.oura_client import OuraRoute, retrieve_collection
from cycle_forecast.data.oura_snapshot import write_snapshot
from cycle_forecast.training.wearable_workflow import (
    WearableEvaluationError,
    WearableEvaluationMode,
    evaluate_local_wearable_models,
)


def _write_history(*, path: Path) -> None:
    """Write invented period starts spanning four completed short cycles."""
    path.write_text(
        "cycle_start_date,period_length_days\n"
        "2025-01-01,5\n"
        "2025-01-21,5\n"
        "2025-02-10,5\n"
        "2025-03-02,5\n"
        "2025-03-22,5\n",
        encoding="utf-8",
    )


def _write_backfill(*, directory: Path) -> None:
    """Write one invented readiness backfill through the final known start."""
    first_day = date(2025, 1, 2)
    final_day = date(2025, 3, 22)
    documents: list[dict[str, object]] = []
    day = first_day
    index = 0
    while day <= final_day:
        if day != date(2025, 1, 5):
            documents.append(
                {
                    "id": f"synthetic-readiness-{index}",
                    "contributors": {},
                    "day": day.isoformat(),
                    "score": 70 + index % 20,
                    "temperature_deviation": (index % 7 - 3) / 10,
                    "timestamp": f"{day.isoformat()}T00:00:00-05:00",
                }
            )
        day += timedelta(days=1)
        index += 1
    payload = json.dumps({"data": documents, "next_token": None}).encode()
    pages = retrieve_collection(
        route=OuraRoute.DAILY_READINESS,
        access_token="synthetic",
        start_date=first_day,
        end_date=final_day,
        transport=lambda _: payload,
    )
    write_snapshot(
        directory=directory,
        route=OuraRoute.DAILY_READINESS,
        start_date=first_day,
        end_date=final_day,
        retrieval_started_at=datetime(2025, 4, 1, 14, tzinfo=UTC),
        retrieval_completed_at=datetime(2025, 4, 1, 14, 0, 1, tzinfo=UTC),
        timezone_name="America/New_York",
        pages=pages,
    )
    late_sleep_payload = json.dumps(
        {
            "data": [
                {
                    "id": "synthetic-late-sleep",
                    "bedtime_start": "2025-01-05T02:00:00-05:00",
                    "bedtime_end": "2025-01-05T10:00:00-05:00",
                    "day": "2025-01-05",
                    "low_battery_alert": False,
                    "period": 0,
                    "time_in_bed": 28_800,
                }
            ],
            "next_token": None,
        }
    ).encode()
    sleep_pages = retrieve_collection(
        route=OuraRoute.SLEEP,
        access_token="synthetic",
        start_date=date(2025, 1, 5),
        end_date=date(2025, 1, 5),
        transport=lambda _: late_sleep_payload,
    )
    write_snapshot(
        directory=directory,
        route=OuraRoute.SLEEP,
        start_date=date(2025, 1, 5),
        end_date=date(2025, 1, 5),
        retrieval_started_at=datetime(2025, 4, 1, 14, tzinfo=UTC),
        retrieval_completed_at=datetime(2025, 4, 1, 14, 0, 1, tzinfo=UTC),
        timezone_name="America/New_York",
        pages=sleep_pages,
    )


def test_exploratory_backfill_compares_all_forecasters_by_complete_cycle(
    tmp_path: Path,
) -> None:
    """Run actual loading, assembly, fitting, calibration, and evaluation."""
    history_path = tmp_path / "history.csv"
    snapshot_directory = tmp_path / "snapshots"
    _write_history(path=history_path)
    _write_backfill(directory=snapshot_directory)

    result = evaluate_local_wearable_models(
        history_path=history_path,
        snapshot_directory=snapshot_directory,
        timezone_name="America/New_York",
        mode=WearableEvaluationMode.EXPLORATORY_BACKFILL,
        observed_through=date(2025, 3, 22),
    )

    assert result.optimistic_backfill_assumption
    assert result.snapshot_count == 2
    assert result.eligible_completed_cycle_count == 4
    assert result.evaluation_fold_count == 2
    assert result.first_fold_training_cycle_count == 1
    assert result.final_fold_training_cycle_count == 2
    assert result.evaluation_cycle_count == 2
    assert result.evaluation_row_count > 0
    assert tuple(fold.training_cycle_count for fold in result.walk_forward.folds) == (
        1,
        2,
    )
    assert len(result.walk_forward.entries) == 3
    assert len(result.diagnostics.candidates) == 3
    assert set(result.diagnostics.data.missingness_rates) == {
        "Readiness score",
        "Temperature",
        "Sleep score",
        "Average HRV",
        "Total sleep",
    }
    assert set(result.diagnostics.data.outcome_window_rates) == {1, 3, 7, 14}
    assert all(
        diagnostic.count == result.evaluation_row_count
        for diagnostic in result.diagnostics.candidates
    )
    assert all(
        diagnostic.minimum_actual_outcome_probability
        <= diagnostic.mean_actual_outcome_probability
        for diagnostic in result.diagnostics.candidates
    )


def test_prospective_mode_refuses_backfill_without_three_proven_cycles(
    tmp_path: Path,
) -> None:
    """Do not pretend one recent retrieval proves old morning availability."""
    history_path = tmp_path / "history.csv"
    snapshot_directory = tmp_path / "snapshots"
    _write_history(path=history_path)
    _write_backfill(directory=snapshot_directory)

    with pytest.raises(
        WearableEvaluationError, match="at least three completed cycles"
    ):
        evaluate_local_wearable_models(
            history_path=history_path,
            snapshot_directory=snapshot_directory,
            timezone_name="America/New_York",
            mode=WearableEvaluationMode.PROSPECTIVE,
            observed_through=date(2025, 4, 1),
        )


def test_rejects_invalid_local_workflow_configuration(tmp_path: Path) -> None:
    """Report missing snapshots, invalid timezone, and invalid assumed hour."""
    history_path = tmp_path / "history.csv"
    snapshot_directory = tmp_path / "snapshots"
    _write_history(path=history_path)
    with pytest.raises(WearableEvaluationError, match="no Oura snapshots"):
        evaluate_local_wearable_models(
            history_path=history_path,
            snapshot_directory=snapshot_directory,
            timezone_name="America/New_York",
            mode=WearableEvaluationMode.PROSPECTIVE,
            observed_through=date(2025, 4, 1),
        )

    _write_backfill(directory=snapshot_directory)
    with pytest.raises(WearableEvaluationError, match="IANA timezone"):
        evaluate_local_wearable_models(
            history_path=history_path,
            snapshot_directory=snapshot_directory,
            timezone_name="Pacific",
            mode=WearableEvaluationMode.PROSPECTIVE,
            observed_through=date(2025, 4, 1),
        )
    with pytest.raises(WearableEvaluationError, match="between 0 and 23"):
        evaluate_local_wearable_models(
            history_path=history_path,
            snapshot_directory=snapshot_directory,
            timezone_name="America/New_York",
            mode=WearableEvaluationMode.EXPLORATORY_BACKFILL,
            observed_through=date(2025, 4, 1),
            prediction_hour=24,
        )
