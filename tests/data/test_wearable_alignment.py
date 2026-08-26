"""Tests for leakage-safe cycle and Oura alignment."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from cycle_forecast.data.cycle_history import CycleHistoryRecord
from cycle_forecast.data.oura import OuraSleep, RetrievedOuraDailyObservation
from cycle_forecast.data.wearable_alignment import (
    WearableAlignmentError,
    align_daily_observation,
)

NEW_YORK = ZoneInfo("America/New_York")


def _history() -> tuple[CycleHistoryRecord, ...]:
    """Create invented consecutive period starts."""
    return (
        CycleHistoryRecord(cycle_start_date=date(2025, 1, 1), period_length_days=5),
        CycleHistoryRecord(cycle_start_date=date(2025, 1, 29), period_length_days=5),
    )


def _observation(*, available_hour: int = 8) -> RetrievedOuraDailyObservation:
    """Create an invented morning sleep observation."""
    return RetrievedOuraDailyObservation(
        day=date(2025, 1, 29),
        available_at=datetime(2025, 1, 29, available_hour, tzinfo=NEW_YORK),
        main_sleep=OuraSleep(
            id="synthetic-sleep",
            bedtime_start="2025-01-28T23:00:00-05:00",
            bedtime_end="2025-01-29T07:00:00-05:00",
            day="2025-01-29",
            low_battery_alert=False,
            period=0,
            time_in_bed=28_800,
            average_hrv=42,
        ),
    )


def test_aligns_start_day_to_previous_cycle_without_target_leakage() -> None:
    """Treat today's newly occurring start as an outcome, not an input start."""
    aligned = align_daily_observation(
        prediction_date=date(2025, 1, 29),
        prediction_cutoff=datetime(2025, 1, 29, 9, tzinfo=NEW_YORK),
        cycle_history=_history(),
        oura_observations=(_observation(),),
    )

    assert aligned.cycle_start_date == date(2025, 1, 1)
    assert aligned.cycle_day == 29
    assert aligned.oura is not None


def test_hides_observation_first_retrieved_after_cutoff() -> None:
    """Do not backdate a record merely because its source day matches."""
    aligned = align_daily_observation(
        prediction_date=date(2025, 1, 29),
        prediction_cutoff=datetime(2025, 1, 29, 9, tzinfo=NEW_YORK),
        cycle_history=_history(),
        oura_observations=(_observation(available_hour=10),),
    )

    assert aligned.oura is None


def test_rejects_sleep_that_ends_after_prediction_cutoff() -> None:
    """Reject a future measurement even if retrieval provenance is inconsistent."""
    observation = _observation(available_hour=8)
    assert observation.main_sleep is not None
    future_sleep = observation.main_sleep.model_copy(
        update={"bedtime_end": "2025-01-29T10:00:00-05:00"}
    )
    inconsistent = observation.model_copy(update={"main_sleep": future_sleep})

    with pytest.raises(WearableAlignmentError, match="ending after cutoff"):
        align_daily_observation(
            prediction_date=date(2025, 1, 29),
            prediction_cutoff=datetime(2025, 1, 29, 9, tzinfo=NEW_YORK),
            cycle_history=_history(),
            oura_observations=(inconsistent,),
        )


def test_rejects_duplicate_oura_days() -> None:
    """Reject ambiguous corrections instead of choosing by input order."""
    with pytest.raises(WearableAlignmentError, match="unique"):
        align_daily_observation(
            prediction_date=date(2025, 1, 29),
            prediction_cutoff=datetime(2025, 1, 29, 9, tzinfo=NEW_YORK),
            cycle_history=_history(),
            oura_observations=(_observation(), _observation()),
        )
