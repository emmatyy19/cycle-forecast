"""Test experimental daily wearable prediction assembly."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import cycle_forecast.prediction_wearable as prediction_wearable
from cycle_forecast.data import CycleHistoryRecord
from cycle_forecast.data.oura import (
    OuraDailyReadiness,
    OuraReadinessContributors,
    RetrievedOuraDailyObservation,
)
from cycle_forecast.prediction_wearable import predict_daily_with_wearable_neighbors


def _observation(*, day: date) -> RetrievedOuraDailyObservation:
    """Create an invented all-missing wearable morning."""
    return RetrievedOuraDailyObservation(
        day=day,
        available_at=datetime.combine(day, datetime.min.time(), tzinfo=UTC),
        main_sleep=None,
        readiness=OuraDailyReadiness(
            id=f"synthetic-{day.isoformat()}",
            contributors=OuraReadinessContributors(),
            day=day.isoformat(),
            score=80,
            timestamp=f"{day.isoformat()}T00:00:00+00:00",
        ),
    )


def test_predicts_from_resolved_prior_wearable_mornings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use only resolved earlier cycles to form today's shadow forecast."""
    snapshot_directory = tmp_path / "snapshots"
    snapshot_directory.mkdir()
    (snapshot_directory / "synthetic.json").touch()
    history = tuple(
        CycleHistoryRecord(cycle_start_date=start, period_length_days=5)
        for start in (date(2025, 1, 1), date(2025, 1, 29), date(2025, 2, 26))
    )
    observations = tuple(
        _observation(day=day)
        for day in (
            date(2025, 1, 20),
            date(2025, 2, 20),
            date(2025, 3, 5),
            date(2025, 3, 10),
        )
    )

    def load_snapshot_stub(**_: object) -> object:
        """Stand in for one already-validated snapshot."""
        return object()

    def load_history_stub(**_: object) -> tuple[CycleHistoryRecord, ...]:
        """Return invented chronological cycle history."""
        return history

    def normalize_stub(**_: object) -> tuple[RetrievedOuraDailyObservation, ...]:
        """Return invented normalized observations."""
        return observations

    monkeypatch.setattr(prediction_wearable, "load_snapshot", load_snapshot_stub)
    monkeypatch.setattr(prediction_wearable, "load_cycle_history", load_history_stub)
    monkeypatch.setattr(prediction_wearable, "normalize_oura_snapshots", normalize_stub)

    prediction = predict_daily_with_wearable_neighbors(
        history_path=tmp_path / "history.csv",
        snapshot_directory=snapshot_directory,
        prediction_date=date(2025, 3, 10),
        prediction_cutoff=datetime(2025, 3, 10, 9, tzinfo=UTC),
    )

    assert prediction.training_morning_count == 2
    assert prediction.model_version == "wearable-neighbor-v1"
    assert prediction.temperature_model_version == "history-temperature-blend-v1"
    assert sum(
        (
            *prediction.distribution.daily_probabilities,
            prediction.distribution.after_horizon_probability,
        )
    ) == pytest.approx(1.0)
    assert sum(
        (
            *prediction.temperature_distribution.daily_probabilities,
            prediction.temperature_distribution.after_horizon_probability,
        )
    ) == pytest.approx(1.0)


def test_rejects_unsafe_cutoff_or_missing_snapshots(tmp_path: Path) -> None:
    """Require temporal provenance and at least one validated snapshot."""
    with pytest.raises(ValueError, match="timezone-aware"):
        predict_daily_with_wearable_neighbors(
            history_path=tmp_path / "history.csv",
            snapshot_directory=tmp_path,
            prediction_date=date(2025, 3, 10),
            prediction_cutoff=datetime(2025, 3, 10, 9),
        )
    with pytest.raises(ValueError, match="requires Oura snapshots"):
        predict_daily_with_wearable_neighbors(
            history_path=tmp_path / "history.csv",
            snapshot_directory=tmp_path,
            prediction_date=date(2025, 3, 10),
            prediction_cutoff=datetime(2025, 3, 10, 9, tzinfo=UTC),
        )
