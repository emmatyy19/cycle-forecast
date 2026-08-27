"""Resolve overlapping immutable Oura snapshots without temporal leakage."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from cycle_forecast.data.oura import (
    OuraDailyReadiness,
    OuraDailySleep,
    OuraSleep,
    RetrievedOuraDailyObservation,
)
from cycle_forecast.data.oura_client import OuraRoute
from cycle_forecast.data.oura_snapshot import LoadedOuraSnapshot

OURA_NORMALIZATION_VERSION = "oura-normalization-v1"
"""Semantic version of correction and duplicate resolution."""

OuraDocument = OuraSleep | OuraDailySleep | OuraDailyReadiness


class OuraNormalizationError(ValueError):
    """Indicate ambiguous or inconsistent snapshot versions."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _DocumentVersion:
    """Pair one source document with its proven availability."""

    route: OuraRoute
    available_at: datetime
    snapshot_fingerprint: str
    document: OuraDocument


def _payload_without(*, document: OuraDocument, fields: set[str]) -> str:
    """Serialize a document deterministically after excluding named fields."""
    return document.model_dump_json(exclude=fields)


def _select_equivalent_daily(
    *, versions: tuple[_DocumentVersion, ...], day: date, route: OuraRoute
) -> _DocumentVersion | None:
    """Collapse same-day daily summaries that differ only by source ID."""
    if not versions:
        return None
    contents = {
        _payload_without(document=version.document, fields={"id"})
        for version in versions
    }
    if len(contents) > 1:
        raise OuraNormalizationError(
            f"multiple distinct {route.value} documents for source day "
            f"{day.isoformat()}"
        )
    return min(versions, key=lambda version: version.document.id)


def _select_main_sleep(
    *, versions: tuple[_DocumentVersion, ...], day: date
) -> _DocumentVersion | None:
    """Resolve main-sleep aliases after excluding duplicate nested readiness."""
    if not versions:
        return None
    measurement_contents = {
        _payload_without(document=version.document, fields={"id", "readiness"})
        for version in versions
    }
    if len(measurement_contents) > 1:
        raise OuraNormalizationError(
            f"multiple distinct main-sleep measurements for source day "
            f"{day.isoformat()}"
        )
    selected = min(versions, key=lambda version: version.document.id)
    if not isinstance(selected.document, OuraSleep):
        raise OuraNormalizationError("main-sleep candidate must be a sleep document")
    return _DocumentVersion(
        route=selected.route,
        available_at=selected.available_at,
        snapshot_fingerprint=selected.snapshot_fingerprint,
        document=selected.document.model_copy(update={"readiness": None}),
    )


def _versions_available_at(
    *, snapshots: Sequence[LoadedOuraSnapshot], cutoff: datetime
) -> tuple[_DocumentVersion, ...]:
    """Select the latest cutoff-available version of every route/document ID."""
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise OuraNormalizationError("cutoff must be timezone-aware")
    grouped: dict[tuple[OuraRoute, str], list[_DocumentVersion]] = {}
    for snapshot in snapshots:
        if snapshot.retrieval_started_at > cutoff:
            continue
        for document in snapshot.documents:
            version = _DocumentVersion(
                route=snapshot.route,
                available_at=snapshot.retrieval_started_at,
                snapshot_fingerprint=snapshot.fingerprint,
                document=document,
            )
            grouped.setdefault((snapshot.route, document.id), []).append(version)

    selected: list[_DocumentVersion] = []
    for key, versions in grouped.items():
        latest_at = max(version.available_at for version in versions)
        latest = tuple(
            version for version in versions if version.available_at == latest_at
        )
        distinct = {version.document.model_dump_json() for version in latest}
        if len(distinct) > 1:
            route, document_id = key
            raise OuraNormalizationError(
                f"ambiguous {route.value} versions for document {document_id}"
            )
        selected.append(latest[0])
    return tuple(selected)


def normalize_oura_snapshots(
    *, snapshots: Sequence[LoadedOuraSnapshot], cutoff: datetime
) -> tuple[RetrievedOuraDailyObservation, ...]:
    """Build one allowlisted, correction-aware observation per source day.

    Parameters
    ----------
    snapshots
        Fully validated immutable snapshots. Input order has no meaning.
    cutoff
        Operational or simulated retrieval cutoff.

    Returns
    -------
    tuple[RetrievedOuraDailyObservation, ...]
        Chronological observations containing only versions available by cutoff.

    Raises
    ------
    OuraNormalizationError
        If versions or same-day route records cannot be resolved uniquely.

    Notes
    -----
    Repeated pulls of identical document IDs count once. Later retrieved content
    supersedes earlier content only after its retrieval instant. Detailed sleep
    uses Oura's ``period == 0`` main-sleep record. Naps and the redundant nested
    sleep-readiness copy remain outside the initial modeling allowlist; readiness
    comes from the dedicated daily-readiness route.
    """
    versions = _versions_available_at(snapshots=snapshots, cutoff=cutoff)
    by_day: dict[date, list[_DocumentVersion]] = {}
    for version in versions:
        by_day.setdefault(date.fromisoformat(version.document.day), []).append(version)

    observations: list[RetrievedOuraDailyObservation] = []
    for day, day_versions in sorted(by_day.items()):
        readiness = tuple(
            version
            for version in day_versions
            if version.route is OuraRoute.DAILY_READINESS
        )
        daily_sleep = tuple(
            version
            for version in day_versions
            if version.route is OuraRoute.DAILY_SLEEP
        )
        main_sleep = tuple(
            version
            for version in day_versions
            if version.route is OuraRoute.SLEEP
            and isinstance(version.document, OuraSleep)
            and version.document.period == 0
        )
        selected_readiness = _select_equivalent_daily(
            versions=readiness,
            day=day,
            route=OuraRoute.DAILY_READINESS,
        )
        selected_daily_sleep = _select_equivalent_daily(
            versions=daily_sleep,
            day=day,
            route=OuraRoute.DAILY_SLEEP,
        )
        selected_main_sleep = _select_main_sleep(versions=main_sleep, day=day)
        chosen = tuple(
            version
            for version in (
                selected_readiness,
                selected_daily_sleep,
                selected_main_sleep,
            )
            if version is not None
        )
        if not chosen:
            continue
        observations.append(
            RetrievedOuraDailyObservation(
                day=day,
                available_at=max(version.available_at for version in chosen),
                readiness=(
                    selected_readiness.document
                    if selected_readiness is not None
                    and isinstance(selected_readiness.document, OuraDailyReadiness)
                    else None
                ),
                daily_sleep=(
                    selected_daily_sleep.document
                    if selected_daily_sleep is not None
                    and isinstance(selected_daily_sleep.document, OuraDailySleep)
                    else None
                ),
                main_sleep=(
                    selected_main_sleep.document
                    if selected_main_sleep is not None
                    and isinstance(selected_main_sleep.document, OuraSleep)
                    else None
                ),
            )
        )
    return tuple(observations)
