"""Tests for cutoff-safe Oura snapshot normalization."""

from datetime import UTC, datetime

import pytest

from cycle_forecast.data.oura import (
    OuraDailyReadiness,
    OuraReadiness,
    OuraReadinessContributors,
    OuraSleep,
)
from cycle_forecast.data.oura_client import OuraRoute
from cycle_forecast.data.oura_normalization import (
    OuraNormalizationError,
    normalize_oura_snapshots,
)
from cycle_forecast.data.oura_snapshot import LoadedOuraSnapshot


def _readiness(*, score: int, available_hour: int) -> LoadedOuraSnapshot:
    """Create one invented readiness snapshot version."""
    return LoadedOuraSnapshot(
        route=OuraRoute.DAILY_READINESS,
        retrieval_started_at=datetime(2025, 1, 15, available_hour, tzinfo=UTC),
        fingerprint=f"sha256:{available_hour:064x}",
        documents=(
            OuraDailyReadiness(
                id="synthetic-readiness",
                contributors=OuraReadinessContributors(),
                day="2025-01-15",
                score=score,
                timestamp="2025-01-15T00:00:00-05:00",
            ),
        ),
    )


def test_selects_latest_document_version_available_at_each_cutoff() -> None:
    """Preserve an old version until the correction was actually retrieved."""
    snapshots = (
        _readiness(score=70, available_hour=8),
        _readiness(score=80, available_hour=10),
    )

    early = normalize_oura_snapshots(
        snapshots=snapshots,
        cutoff=datetime(2025, 1, 15, 9, tzinfo=UTC),
    )
    late = normalize_oura_snapshots(
        snapshots=snapshots,
        cutoff=datetime(2025, 1, 15, 11, tzinfo=UTC),
    )

    assert early[0].readiness is not None
    assert late[0].readiness is not None
    assert early[0].readiness.score == 70
    assert late[0].readiness.score == 80


def test_deduplicates_repeated_identical_historical_pulls() -> None:
    """Count an unchanged document once even when snapshots overlap."""
    snapshot = _readiness(score=70, available_hour=8)
    repeated = LoadedOuraSnapshot(
        route=snapshot.route,
        retrieval_started_at=datetime(2025, 1, 15, 9, tzinfo=UTC),
        fingerprint=f"sha256:{9:064x}",
        documents=snapshot.documents,
    )

    observations = normalize_oura_snapshots(
        snapshots=(snapshot, repeated),
        cutoff=datetime(2025, 1, 15, 10, tzinfo=UTC),
    )

    assert len(observations) == 1
    assert observations[0].available_at == repeated.retrieval_started_at


def test_uses_main_sleep_and_excludes_naps() -> None:
    """Apply the explicit detailed-sleep modeling allowlist."""
    snapshot = LoadedOuraSnapshot(
        route=OuraRoute.SLEEP,
        retrieval_started_at=datetime(2025, 1, 15, 8, tzinfo=UTC),
        fingerprint=f"sha256:{1:064x}",
        documents=(
            OuraSleep(
                id="synthetic-main",
                bedtime_start="2025-01-14T23:00:00-05:00",
                bedtime_end="2025-01-15T07:00:00-05:00",
                day="2025-01-15",
                low_battery_alert=False,
                period=0,
                time_in_bed=28_800,
            ),
            OuraSleep(
                id="synthetic-nap",
                bedtime_start="2025-01-15T13:00:00-05:00",
                bedtime_end="2025-01-15T13:30:00-05:00",
                day="2025-01-15",
                low_battery_alert=False,
                period=1,
                time_in_bed=1_800,
            ),
        ),
    )

    observations = normalize_oura_snapshots(
        snapshots=(snapshot,),
        cutoff=datetime(2025, 1, 16, tzinfo=UTC),
    )

    assert observations[0].main_sleep is not None
    assert observations[0].main_sleep.id == "synthetic-main"


def test_rejects_conflicting_versions_at_same_retrieval_instant() -> None:
    """Reject a tie whose winner would otherwise depend on input order."""
    first = _readiness(score=70, available_hour=8)
    second = _readiness(score=80, available_hour=8)

    with pytest.raises(OuraNormalizationError, match="ambiguous"):
        normalize_oura_snapshots(
            snapshots=(first, second),
            cutoff=datetime(2025, 1, 15, 9, tzinfo=UTC),
        )


def test_collapses_reissued_daily_ids_and_excludes_nested_sleep_readiness() -> None:
    """Handle aliases while retaining readiness only from its canonical route."""
    readiness = _readiness(score=70, available_hour=8)
    assert isinstance(readiness.documents[0], OuraDailyReadiness)
    readiness_alias = readiness.documents[0].model_copy(
        update={"id": "synthetic-readiness-alias"}
    )
    daily_snapshot = LoadedOuraSnapshot(
        route=readiness.route,
        retrieval_started_at=readiness.retrieval_started_at,
        fingerprint=readiness.fingerprint,
        documents=(*readiness.documents, readiness_alias),
    )
    common = OuraSleep(
        id="synthetic-main-basic",
        bedtime_start="2025-01-14T23:00:00-05:00",
        bedtime_end="2025-01-15T07:00:00-05:00",
        day="2025-01-15",
        low_battery_alert=False,
        period=0,
        time_in_bed=28_800,
    )
    richer = common.model_copy(
        update={
            "id": "synthetic-main-richer",
            "readiness": OuraReadiness(
                contributors=OuraReadinessContributors(), score=75
            ),
        }
    )
    sleep_snapshot = LoadedOuraSnapshot(
        route=OuraRoute.SLEEP,
        retrieval_started_at=readiness.retrieval_started_at,
        fingerprint=f"sha256:{2:064x}",
        documents=(common, richer),
    )

    observations = normalize_oura_snapshots(
        snapshots=(daily_snapshot, sleep_snapshot),
        cutoff=datetime(2025, 1, 15, 9, tzinfo=UTC),
    )

    assert len(observations) == 1
    assert observations[0].main_sleep is not None
    assert observations[0].main_sleep.readiness is None
    assert observations[0].readiness is not None
