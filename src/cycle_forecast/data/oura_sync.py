"""Coordinate authenticated historical and incremental Oura retrieval."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cycle_forecast.data.oura_auth import (
    OuraToken,
    load_oauth_application,
    load_token,
    refresh_access_token,
    save_token,
)
from cycle_forecast.data.oura_client import (
    OuraRoute,
    Transport,
    count_documents,
    retrieve_collection,
)
from cycle_forecast.data.oura_snapshot import (
    OuraSnapshotError,
    OuraSnapshotMetadata,
    OuraSnapshotResult,
    load_snapshot_metadata,
    write_snapshot,
)

DEFAULT_OURA_SNAPSHOT_DIRECTORY = Path("data/private/oura/snapshots")


class OuraSyncError(ValueError):
    """Indicate invalid local sync configuration or provenance."""


@dataclass(frozen=True, slots=True, kw_only=True)
class OuraRouteSyncResult:
    """Summarize one route without exposing health measurements."""

    route: OuraRoute
    page_count: int
    document_count: int
    snapshot: OuraSnapshotResult | None


def _usable_token(*, token_path: Path, now: datetime) -> OuraToken:
    """Load and refresh the local bearer token when near expiry."""
    token = load_token(path=token_path)
    if token.expires_at > now + timedelta(minutes=5):
        return token
    client_id, client_secret = load_oauth_application()
    refreshed = refresh_access_token(
        refresh_token=token.refresh_token,
        client_id=client_id,
        client_secret=client_secret,
    )
    save_token(token=refreshed, path=token_path)
    return refreshed


def latest_snapshot_end_date(*, directory: Path) -> date | None:
    """Find the latest validated date bound recorded in local snapshots."""
    end_dates: list[date] = []
    for path in directory.glob("*.json") if directory.is_dir() else ():
        try:
            end_dates.append(load_snapshot_metadata(path=path).end_date)
        except OuraSnapshotError as error:
            raise OuraSyncError(f"invalid existing Oura snapshot: {path}") from error
    return max(end_dates, default=None)


def _safe_incremental_start_date(*, directory: Path) -> date | None:
    """Find a date that recovers lagging or missing route coverage."""
    metadata: list[OuraSnapshotMetadata] = []
    for path in directory.glob("*.json") if directory.is_dir() else ():
        try:
            metadata.append(load_snapshot_metadata(path=path))
        except OuraSnapshotError as error:
            raise OuraSyncError(f"invalid existing Oura snapshot: {path}") from error
    if not metadata:
        return None

    latest_by_route = {
        route: max(
            (item.end_date for item in metadata if item.route is route),
            default=None,
        )
        for route in OuraRoute
    }
    if all(end_date is not None for end_date in latest_by_route.values()):
        return min(
            end_date for end_date in latest_by_route.values() if end_date is not None
        )
    return min(item.start_date for item in metadata)


def resolve_sync_start_date(
    *, explicit_start_date: date | None, snapshot_directory: Path
) -> date:
    """Resolve historical or overlapping incremental retrieval start."""
    if explicit_start_date is not None:
        return explicit_start_date
    latest = _safe_incremental_start_date(directory=snapshot_directory)
    if latest is None:
        raise OuraSyncError("first Oura sync requires --start-date")
    return latest


def sync_oura(
    *,
    token_path: Path,
    snapshot_directory: Path,
    start_date: date,
    end_date: date,
    timezone_name: str,
    save: bool,
    now: datetime | None = None,
    transport: Transport | None = None,
) -> tuple[OuraRouteSyncResult, ...]:
    """Retrieve every initial Oura route and optionally persist snapshots."""
    if start_date > end_date:
        raise OuraSyncError("start_date must not be after end_date")
    try:
        ZoneInfo(timezone_name)
    except (OSError, ValueError, ZoneInfoNotFoundError) as error:
        raise OuraSyncError("timezone must be an IANA timezone") from error
    retrieval_started_at = now or datetime.now(tz=UTC)
    if retrieval_started_at.tzinfo is None:
        raise OuraSyncError("now must be timezone-aware")
    token = _usable_token(token_path=token_path, now=retrieval_started_at)
    results: list[OuraRouteSyncResult] = []
    for route in OuraRoute:
        if transport is None:
            pages = retrieve_collection(
                route=route,
                access_token=token.access_token,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            pages = retrieve_collection(
                route=route,
                access_token=token.access_token,
                start_date=start_date,
                end_date=end_date,
                transport=transport,
            )
        completed_at = datetime.now(tz=UTC)
        snapshot = (
            write_snapshot(
                directory=snapshot_directory,
                route=route,
                start_date=start_date,
                end_date=end_date,
                retrieval_started_at=retrieval_started_at,
                retrieval_completed_at=max(completed_at, retrieval_started_at),
                timezone_name=timezone_name,
                pages=pages,
            )
            if save
            else None
        )
        results.append(
            OuraRouteSyncResult(
                route=route,
                page_count=len(pages),
                document_count=count_documents(pages=pages),
                snapshot=snapshot,
            )
        )
    return tuple(results)
