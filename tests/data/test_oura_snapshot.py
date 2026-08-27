"""Tests for immutable Oura retrieval snapshots."""

import json
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from cycle_forecast.data.oura_client import OuraPage, OuraRoute, retrieve_collection
from cycle_forecast.data.oura_snapshot import (
    OuraSnapshotError,
    load_snapshot_metadata,
    write_snapshot,
)


def _pages(*, day: str = "2025-01-15") -> tuple[OuraPage, ...]:
    """Create one validated invented readiness page."""
    payload = (
        '{"data":[{"id":"synthetic","contributors":{},"day":"'
        + day
        + '","timestamp":"2025-01-15T00:00:00-05:00"}],'
        '"next_token":null}'
    ).encode()
    return retrieve_collection(
        route=OuraRoute.DAILY_READINESS,
        access_token="synthetic",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        transport=lambda _: payload,
    )


def test_write_and_validate_private_snapshot(tmp_path: Path) -> None:
    """Persist provenance and verify the deterministic fingerprint."""
    result = write_snapshot(
        directory=tmp_path,
        route=OuraRoute.DAILY_READINESS,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        retrieval_started_at=datetime(2025, 2, 1, tzinfo=UTC),
        retrieval_completed_at=datetime(2025, 2, 1, 0, 0, 1, tzinfo=UTC),
        timezone_name="America/New_York",
        pages=_pages(),
    )

    assert result.document_count == 1
    assert result.fingerprint.startswith("sha256:")
    assert load_snapshot_metadata(path=result.path).end_date == date(2025, 1, 31)
    assert result.path.parent.stat().st_mode & 0o777 == 0o700


def test_snapshot_rejects_document_outside_requested_dates(tmp_path: Path) -> None:
    """Reject a response whose source day violates the bounded query."""
    with pytest.raises(OuraSnapshotError, match="outside"):
        write_snapshot(
            directory=tmp_path,
            route=OuraRoute.DAILY_READINESS,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            retrieval_started_at=datetime(2025, 2, 1, tzinfo=UTC),
            retrieval_completed_at=datetime(2025, 2, 1, 0, 0, 1, tzinfo=UTC),
            timezone_name="America/New_York",
            pages=_pages(day="2025-02-01"),
        )


def test_snapshot_detects_local_tampering(tmp_path: Path) -> None:
    """Reject changed content rather than trusting filename metadata."""
    result = write_snapshot(
        directory=tmp_path,
        route=OuraRoute.DAILY_READINESS,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        retrieval_started_at=datetime(2025, 2, 1, tzinfo=UTC),
        retrieval_completed_at=datetime(2025, 2, 1, 0, 0, 1, tzinfo=UTC),
        timezone_name="America/New_York",
        pages=_pages(),
    )
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    payload["end_date"] = "2025-02-01"
    result.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OuraSnapshotError, match="fingerprint"):
        load_snapshot_metadata(path=result.path)


def test_snapshot_rejects_unsupported_api_provenance(tmp_path: Path) -> None:
    """Reject a self-consistently fingerprinted snapshot from another API version."""
    result = write_snapshot(
        directory=tmp_path,
        route=OuraRoute.DAILY_READINESS,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        retrieval_started_at=datetime(2025, 2, 1, tzinfo=UTC),
        retrieval_completed_at=datetime(2025, 2, 1, 0, 0, 1, tzinfo=UTC),
        timezone_name="America/New_York",
        pages=_pages(),
    )
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    payload.pop("fingerprint")
    payload["oura_api_version"] = "3"
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    payload["fingerprint"] = f"sha256:{sha256(canonical).hexdigest()}"
    result.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(OuraSnapshotError, match="invalid Oura snapshot"):
        load_snapshot_metadata(path=result.path)
