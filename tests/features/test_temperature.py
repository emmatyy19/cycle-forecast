"""Test cutoff-safe temperature trajectory feature construction."""

from datetime import UTC, date, datetime, timedelta

import pytest

from cycle_forecast.data.wearable_alignment import AlignedDailyObservation
from cycle_forecast.features.temperature import (
    TEMPERATURE_TRAJECTORY_FEATURE_NAMES,
    build_temperature_trajectory_rows,
)
from cycle_forecast.features.wearable import WEARABLE_FEATURE_NAMES, WearableFeatureRow


def _row(
    *,
    day: date,
    cycle_start: date,
    temperature: float | None,
    cutoff: datetime | None = None,
) -> WearableFeatureRow:
    """Create one invented wearable morning with optional temperature."""
    return WearableFeatureRow(
        aligned=AlignedDailyObservation(
            prediction_date=day,
            prediction_cutoff=(
                cutoff
                if cutoff is not None
                else datetime.combine(day, datetime.min.time(), tzinfo=UTC)
            ),
            cycle_start_date=cycle_start,
            cycle_day=(day - cycle_start).days + 1,
            oura=None,
        ),
        feature_names=WEARABLE_FEATURE_NAMES,
        values=(
            float((day - cycle_start).days + 1),
            0.0,
            temperature or 0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            float(temperature is None),
            1.0,
            1.0,
            1.0,
        ),
        outcome_offset_days=5,
    )


def test_builds_recent_trajectory_without_crossing_cycle_boundary() -> None:
    """Use observed same-cycle nights without interpolation or future leakage."""
    first_start = date(2025, 1, 1)
    second_start = date(2025, 1, 10)
    rows = (
        _row(day=first_start, cycle_start=first_start, temperature=0.1),
        _row(
            day=first_start + timedelta(days=1),
            cycle_start=first_start,
            temperature=0.2,
        ),
        _row(
            day=first_start + timedelta(days=2),
            cycle_start=first_start,
            temperature=None,
        ),
        _row(
            day=first_start + timedelta(days=3),
            cycle_start=first_start,
            temperature=0.4,
        ),
        _row(day=second_start, cycle_start=second_start, temperature=-0.2),
    )

    trajectories = build_temperature_trajectory_rows(rows=rows)

    assert trajectories[3].feature_names == TEMPERATURE_TRAJECTORY_FEATURE_NAMES
    assert trajectories[3].values[2] == pytest.approx(0.3)
    assert trajectories[3].values[3] == pytest.approx(0.7 / 3.0)
    assert trajectories[3].values[4] == pytest.approx(0.1)
    assert trajectories[3].values[5] == pytest.approx(0.0)
    assert trajectories[3].values[6] == pytest.approx(1.0)
    assert trajectories[4].values[2] == pytest.approx(-0.2)
    assert trajectories[4].values[3] == pytest.approx(-0.2)
    assert trajectories[4].values[10] == 1.0


def test_requires_strictly_chronological_trajectory_rows() -> None:
    """Reject repeated or reversed mornings before deriving rolling features."""
    start = date(2025, 1, 1)
    row = _row(day=start, cycle_start=start, temperature=0.1)

    with pytest.raises(ValueError, match="strictly chronological"):
        build_temperature_trajectory_rows(rows=(row, row))


def test_equal_retrieval_cutoffs_do_not_expose_future_source_days() -> None:
    """Exclude later source dates when a backfill proves them at one cutoff."""
    start = date(2025, 1, 1)
    shared_cutoff = datetime(2025, 1, 10, tzinfo=UTC)
    rows = (
        _row(
            day=start,
            cycle_start=start,
            temperature=0.1,
            cutoff=shared_cutoff,
        ),
        _row(
            day=start + timedelta(days=1),
            cycle_start=start,
            temperature=0.5,
            cutoff=shared_cutoff,
        ),
    )

    trajectories = build_temperature_trajectory_rows(rows=rows)

    assert trajectories[0].values[2] == pytest.approx(0.1)
    assert trajectories[1].values[2] == pytest.approx(0.3)
