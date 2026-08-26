"""Tests for historical and incremental Oura sync orchestration."""

from datetime import UTC, date, datetime
from pathlib import Path
from urllib.request import Request

import pytest

import cycle_forecast.data.oura_sync as oura_sync
from cycle_forecast.data.oura_auth import OuraToken, save_token
from cycle_forecast.data.oura_sync import (
    OuraSyncError,
    latest_snapshot_end_date,
    resolve_sync_start_date,
    sync_oura,
)


def _token(path: Path) -> None:
    """Write an invented unexpired OAuth token."""
    save_token(
        token=OuraToken(
            access_token="synthetic-access",
            refresh_token="synthetic-refresh",
            expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        path=path,
    )


def _empty_page(_: Request) -> bytes:
    """Return a valid empty collection for any supported route."""
    return b'{"data":[],"next_token":null}'


def test_historical_sync_persists_every_validated_route(tmp_path: Path) -> None:
    """Use the same snapshot pipeline for a bounded historical import."""
    token_path = tmp_path / "token.json"
    snapshot_directory = tmp_path / "snapshots"
    _token(token_path)

    results = sync_oura(
        token_path=token_path,
        snapshot_directory=snapshot_directory,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        timezone_name="America/New_York",
        save=True,
        now=datetime(2025, 2, 1, tzinfo=UTC),
        transport=_empty_page,
    )

    assert len(results) == 3
    assert all(result.snapshot is not None for result in results)
    assert len(tuple(snapshot_directory.glob("*.json"))) == 3
    assert latest_snapshot_end_date(directory=snapshot_directory) == date(2025, 1, 31)


def test_incremental_sync_overlaps_latest_requested_day(tmp_path: Path) -> None:
    """Refetch the last requested day so corrections enter a new snapshot."""
    token_path = tmp_path / "token.json"
    snapshot_directory = tmp_path / "snapshots"
    _token(token_path)
    sync_oura(
        token_path=token_path,
        snapshot_directory=snapshot_directory,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        timezone_name="America/New_York",
        save=True,
        now=datetime(2025, 2, 1, tzinfo=UTC),
        transport=_empty_page,
    )

    assert resolve_sync_start_date(
        explicit_start_date=None,
        snapshot_directory=snapshot_directory,
    ) == date(2025, 1, 31)


def test_incremental_sync_recovers_partial_route_history(tmp_path: Path) -> None:
    """Restart from the original bound when a prior batch lacks required routes."""
    token_path = tmp_path / "token.json"
    snapshot_directory = tmp_path / "snapshots"
    _token(token_path)
    sync_oura(
        token_path=token_path,
        snapshot_directory=snapshot_directory,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        timezone_name="America/New_York",
        save=True,
        now=datetime(2025, 2, 1, tzinfo=UTC),
        transport=_empty_page,
    )
    for path in snapshot_directory.glob("*.json"):
        if not path.name.startswith("sleep-"):
            path.unlink()

    assert resolve_sync_start_date(
        explicit_start_date=None,
        snapshot_directory=snapshot_directory,
    ) == date(2025, 1, 1)


def test_check_only_validates_without_persisting_payloads(tmp_path: Path) -> None:
    """Support a privacy-safe live smoke test that writes no health response."""
    token_path = tmp_path / "token.json"
    snapshot_directory = tmp_path / "snapshots"
    _token(token_path)

    results = sync_oura(
        token_path=token_path,
        snapshot_directory=snapshot_directory,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        timezone_name="America/New_York",
        save=False,
        now=datetime(2025, 2, 1, tzinfo=UTC),
        transport=_empty_page,
    )

    assert all(result.snapshot is None for result in results)
    assert not snapshot_directory.exists()


def test_sync_refreshes_a_token_near_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refresh credentials before retrieval and persist the replacement."""
    token_path = tmp_path / "token.json"
    save_token(
        token=OuraToken(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        path=token_path,
    )
    refreshed = OuraToken(
        access_token="refreshed-access",
        refresh_token="new-refresh",
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )

    def refresh(**_: object) -> OuraToken:
        """Return an invented replacement token."""
        return refreshed

    monkeypatch.setattr(
        oura_sync,
        "load_oauth_application",
        lambda: ("synthetic-client", "synthetic-secret"),
    )
    monkeypatch.setattr(oura_sync, "refresh_access_token", refresh)

    sync_oura(
        token_path=token_path,
        snapshot_directory=tmp_path / "snapshots",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 1),
        timezone_name="America/New_York",
        save=False,
        now=datetime(2025, 2, 1, tzinfo=UTC),
        transport=_empty_page,
    )

    assert oura_sync.load_token(path=token_path) == refreshed


def test_sync_rejects_invalid_range_before_reading_credentials(tmp_path: Path) -> None:
    """Reject an inverted bounded query without touching token storage."""
    with pytest.raises(OuraSyncError, match="start_date"):
        sync_oura(
            token_path=tmp_path / "missing.json",
            snapshot_directory=tmp_path / "snapshots",
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 1),
            timezone_name="America/New_York",
            save=False,
        )


def test_sync_rejects_non_iana_timezone_before_reading_credentials(
    tmp_path: Path,
) -> None:
    """Require reproducible timezone rules rather than an abbreviation."""
    with pytest.raises(OuraSyncError, match="IANA"):
        sync_oura(
            token_path=tmp_path / "missing.json",
            snapshot_directory=tmp_path / "snapshots",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
            timezone_name="New York",
            save=False,
        )
