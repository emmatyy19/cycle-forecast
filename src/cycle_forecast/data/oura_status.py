"""Report non-sensitive readiness of the local Oura integration."""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from cycle_forecast.data.oura_auth import (
    OuraAuthorizationError,
    load_oauth_application,
    load_token,
)
from cycle_forecast.data.oura_sync import latest_snapshot_end_date


@dataclass(frozen=True, slots=True, kw_only=True)
class OuraStatus:
    """Summarize local Oura readiness without exposing secret or health data."""

    application_credentials_available: bool
    token_state: str
    token_expires_at: datetime | None
    snapshot_count: int
    latest_snapshot_end_date: date | None


def inspect_oura_status(
    *, token_path: Path, snapshot_directory: Path, now: datetime | None = None
) -> OuraStatus:
    """Inspect Keychain, token, and snapshot metadata without network access."""
    try:
        load_oauth_application()
        credentials_available = True
    except OuraAuthorizationError:
        credentials_available = False

    token_state = "missing"
    token_expires_at: datetime | None = None
    if token_path.is_file():
        try:
            token = load_token(path=token_path)
            token_expires_at = token.expires_at
            current = now or datetime.now(tz=UTC)
            token_state = "valid" if token.expires_at > current else "expired"
        except OuraAuthorizationError:
            token_state = "invalid"

    snapshot_paths = tuple(snapshot_directory.glob("*.json"))
    latest = latest_snapshot_end_date(directory=snapshot_directory)
    return OuraStatus(
        application_credentials_available=credentials_available,
        token_state=token_state,
        token_expires_at=token_expires_at,
        snapshot_count=len(snapshot_paths),
        latest_snapshot_end_date=latest,
    )
