"""Tests for non-sensitive local Oura status inspection."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import cycle_forecast.data.oura_status as oura_status
from cycle_forecast.data.oura_auth import OuraAuthorizationError, OuraToken, save_token


def test_status_reports_valid_token_and_snapshot_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report readiness without returning credentials or health values."""
    token_path = tmp_path / "token.json"
    save_token(
        token=OuraToken(
            access_token="synthetic-access",
            refresh_token="synthetic-refresh",
            expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        ),
        path=token_path,
    )
    monkeypatch.setattr(
        oura_status, "load_oauth_application", lambda: ("client", "secret")
    )

    def latest_date(**_: object) -> date:
        """Return invented validated snapshot coverage."""
        return date(2025, 1, 31)

    monkeypatch.setattr(
        oura_status,
        "latest_snapshot_end_date",
        latest_date,
    )
    snapshot_directory = tmp_path / "snapshots"
    snapshot_directory.mkdir()
    (snapshot_directory / "synthetic.json").write_text("{}", encoding="utf-8")

    status = oura_status.inspect_oura_status(
        token_path=token_path,
        snapshot_directory=snapshot_directory,
        now=datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert status.application_credentials_available
    assert status.token_state == "valid"
    assert status.snapshot_count == 1
    assert status.latest_snapshot_end_date == date(2025, 1, 31)


def test_status_reports_missing_credentials_and_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return actionable missing states without raising or contacting Oura."""

    def missing_credentials() -> tuple[str, str]:
        """Simulate an unavailable Keychain entry."""
        raise OuraAuthorizationError("missing")

    monkeypatch.setattr(oura_status, "load_oauth_application", missing_credentials)

    status = oura_status.inspect_oura_status(
        token_path=tmp_path / "missing-token.json",
        snapshot_directory=tmp_path / "missing-snapshots",
    )

    assert not status.application_credentials_available
    assert status.token_state == "missing"
    assert status.snapshot_count == 0


def test_status_reports_invalid_token_without_exposing_file_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convert a malformed local token into a non-sensitive status value."""
    token_path = tmp_path / "token.json"
    token_path.write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(
        oura_status, "load_oauth_application", lambda: ("client", "secret")
    )

    status = oura_status.inspect_oura_status(
        token_path=token_path,
        snapshot_directory=tmp_path / "snapshots",
    )

    assert status.token_state == "invalid"
    assert status.token_expires_at is None
