"""Persist immutable, fingerprinted Oura retrieval snapshots locally."""

import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from cycle_forecast.data.oura import (
    OURA_OPENAPI_SPECIFICATION_VERSION,
    OuraDailyReadiness,
    OuraDailySleep,
    OuraSleep,
)
from cycle_forecast.data.oura_client import OuraPage, OuraRoute
from cycle_forecast.data.private_files import ensure_private_directory

OURA_SNAPSHOT_SCHEMA_VERSION = "oura-snapshot-v1"
Fingerprint = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]


class OuraSnapshotError(ValueError):
    """Indicate invalid snapshot metadata or an unsafe write."""


@dataclass(frozen=True, slots=True, kw_only=True)
class OuraSnapshotResult:
    """Describe a safely persisted Oura snapshot without health values."""

    path: Path
    fingerprint: str
    document_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class OuraSnapshotMetadata:
    """Describe validated snapshot coverage without exposing health values."""

    route: OuraRoute
    start_date: date
    end_date: date


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadedOuraSnapshot:
    """Contain one fully validated snapshot for local normalization."""

    route: OuraRoute
    retrieval_started_at: datetime
    fingerprint: str
    documents: tuple[OuraSleep | OuraDailySleep | OuraDailyReadiness, ...]


class _StoredSnapshot(BaseModel):
    """Validate a complete private snapshot and its retrieval provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["oura-snapshot-v1"]
    oura_api_version: Literal["2"]
    oura_openapi_specification_version: Literal["1.35"]
    route: OuraRoute
    start_date: date
    end_date: date
    retrieval_started_at: datetime
    retrieval_completed_at: datetime
    timezone: str
    pagination: list[str | None]
    documents: list[dict[str, object]]
    fingerprint: Fingerprint

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        """Require a reproducible IANA timezone name."""
        try:
            ZoneInfo(value)
        except (OSError, ValueError, ZoneInfoNotFoundError) as error:
            raise ValueError("timezone must be an IANA timezone") from error
        return value

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        """Require internally consistent bounds, timestamps, and pagination."""
        if self.start_date > self.end_date:
            raise ValueError("snapshot date range is invalid")
        for instant in (self.retrieval_started_at, self.retrieval_completed_at):
            if instant.tzinfo is None or instant.utcoffset() is None:
                raise ValueError("retrieval timestamps must be timezone-aware")
        if self.retrieval_completed_at < self.retrieval_started_at:
            raise ValueError("retrieval completed before it started")
        if not self.pagination or self.pagination[-1] is not None:
            raise ValueError("snapshot pagination sequence is incomplete")
        if any(token is None for token in self.pagination[:-1]):
            raise ValueError("snapshot pagination sequence is incomplete")

        documents_json = json.dumps(
            self.documents, separators=(",", ":"), sort_keys=True
        ).encode()
        if self.route is OuraRoute.SLEEP:
            documents = TypeAdapter(list[OuraSleep]).validate_json(documents_json)
        elif self.route is OuraRoute.DAILY_SLEEP:
            documents = TypeAdapter(list[OuraDailySleep]).validate_json(documents_json)
        else:
            documents = TypeAdapter(list[OuraDailyReadiness]).validate_json(
                documents_json
            )
        document_ids: set[str] = set()
        for document in documents:
            if document.id in document_ids:
                raise ValueError("snapshot contains a duplicate document ID")
            document_ids.add(document.id)
            if not self.start_date <= date.fromisoformat(document.day) <= self.end_date:
                raise ValueError("snapshot document is outside requested bounds")
        return self


def _snapshot_fingerprint(*, payload: dict[str, object]) -> str:
    """Fingerprint canonical content that excludes its own fingerprint."""
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return f"sha256:{sha256(canonical).hexdigest()}"


def _canonical_payload(
    *,
    route: OuraRoute,
    start_date: date,
    end_date: date,
    retrieval_started_at: datetime,
    retrieval_completed_at: datetime,
    timezone_name: str,
    pages: tuple[OuraPage, ...],
) -> dict[str, object]:
    """Build deterministic snapshot content before adding its fingerprint."""
    if not pages or any(page.route is not route for page in pages):
        raise OuraSnapshotError("snapshot requires pages from exactly one route")
    if start_date > end_date:
        raise OuraSnapshotError("snapshot date range is invalid")
    for instant in (retrieval_started_at, retrieval_completed_at):
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise OuraSnapshotError("retrieval timestamps must be timezone-aware")
    if retrieval_completed_at < retrieval_started_at:
        raise OuraSnapshotError("retrieval completed before it started")
    try:
        ZoneInfo(timezone_name)
    except (OSError, ValueError, ZoneInfoNotFoundError) as error:
        raise OuraSnapshotError("timezone_name must be an IANA timezone") from error

    documents: list[object] = []
    pagination: list[str | None] = []
    document_ids: set[str] = set()
    for page in pages:
        raw = json.loads(page.payload)
        page_documents = raw["data"]
        for document in page_documents:
            document_id = document["id"]
            if document_id in document_ids:
                raise OuraSnapshotError("snapshot contains a duplicate document ID")
            document_ids.add(document_id)
            document_day = date.fromisoformat(document["day"])
            if not start_date <= document_day <= end_date:
                raise OuraSnapshotError("snapshot document is outside requested bounds")
        documents.extend(page_documents)
        pagination.append(page.next_token)
    if pagination[-1] is not None or any(token is None for token in pagination[:-1]):
        raise OuraSnapshotError("snapshot pagination sequence is incomplete")
    return {
        "schema_version": OURA_SNAPSHOT_SCHEMA_VERSION,
        "oura_api_version": "2",
        "oura_openapi_specification_version": OURA_OPENAPI_SPECIFICATION_VERSION,
        "route": route.value,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "retrieval_started_at": retrieval_started_at.astimezone(UTC).isoformat(),
        "retrieval_completed_at": retrieval_completed_at.astimezone(UTC).isoformat(),
        "timezone": timezone_name,
        "pagination": pagination,
        "documents": documents,
    }


def write_snapshot(
    *,
    directory: Path,
    route: OuraRoute,
    start_date: date,
    end_date: date,
    retrieval_started_at: datetime,
    retrieval_completed_at: datetime,
    timezone_name: str,
    pages: tuple[OuraPage, ...],
) -> OuraSnapshotResult:
    """Write one immutable snapshot atomically without silent replacement."""
    payload = _canonical_payload(
        route=route,
        start_date=start_date,
        end_date=end_date,
        retrieval_started_at=retrieval_started_at,
        retrieval_completed_at=retrieval_completed_at,
        timezone_name=timezone_name,
        pages=pages,
    )
    fingerprint = _snapshot_fingerprint(payload=payload)
    complete = {**payload, "fingerprint": fingerprint}
    stamp = retrieval_started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        ensure_private_directory(directory=directory)
    except OSError as error:
        raise OuraSnapshotError(
            f"could not secure snapshot directory {directory}"
        ) from error
    path = directory / f"{route.value}-{stamp}.json"
    temporary = path.with_suffix(".json.tmp")
    if path.exists() or temporary.exists():
        raise OuraSnapshotError(f"snapshot already exists: {path}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(complete, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        path.chmod(0o600)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise OuraSnapshotError(
            f"could not write snapshot under {directory}"
        ) from error
    return OuraSnapshotResult(
        path=path,
        fingerprint=fingerprint,
        document_count=sum(len(json.loads(page.payload)["data"]) for page in pages),
    )


def load_snapshot_metadata(*, path: Path) -> OuraSnapshotMetadata:
    """Validate stored snapshot provenance and return non-sensitive coverage."""
    try:
        payload = path.read_bytes()
        stored = _StoredSnapshot.model_validate_json(payload)
        complete = json.loads(payload)
        fingerprint = complete.pop("fingerprint")
        if fingerprint != _snapshot_fingerprint(payload=complete):
            raise OuraSnapshotError("Oura snapshot fingerprint mismatch")
        return OuraSnapshotMetadata(
            route=stored.route,
            start_date=stored.start_date,
            end_date=stored.end_date,
        )
    except OuraSnapshotError:
        raise
    except (OSError, ValidationError, ValueError, KeyError, TypeError) as error:
        raise OuraSnapshotError(f"invalid Oura snapshot: {path}") from error


def load_snapshot(*, path: Path) -> LoadedOuraSnapshot:
    """Load validated health documents and retrieval provenance privately.

    Parameters
    ----------
    path
        Local immutable snapshot path.

    Returns
    -------
    LoadedOuraSnapshot
        Validated documents with the instant at which they became available.

    Raises
    ------
    OuraSnapshotError
        If the snapshot is unreadable, invalid, or has been modified.
    """
    try:
        payload = path.read_bytes()
        stored = _StoredSnapshot.model_validate_json(payload)
        complete = json.loads(payload)
        fingerprint = complete.pop("fingerprint")
        if fingerprint != _snapshot_fingerprint(payload=complete):
            raise OuraSnapshotError("Oura snapshot fingerprint mismatch")
        documents_json = json.dumps(
            stored.documents, separators=(",", ":"), sort_keys=True
        ).encode()
        if stored.route is OuraRoute.SLEEP:
            documents = TypeAdapter(tuple[OuraSleep, ...]).validate_json(documents_json)
        elif stored.route is OuraRoute.DAILY_SLEEP:
            documents = TypeAdapter(tuple[OuraDailySleep, ...]).validate_json(
                documents_json
            )
        else:
            documents = TypeAdapter(tuple[OuraDailyReadiness, ...]).validate_json(
                documents_json
            )
        return LoadedOuraSnapshot(
            route=stored.route,
            retrieval_started_at=stored.retrieval_started_at,
            fingerprint=fingerprint,
            documents=documents,
        )
    except OuraSnapshotError:
        raise
    except (OSError, ValidationError, ValueError, KeyError, TypeError) as error:
        raise OuraSnapshotError(f"invalid Oura snapshot: {path}") from error
