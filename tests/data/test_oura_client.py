"""Tests for bounded, paginated Oura API retrieval."""

from datetime import date
from email.message import Message
from typing import Self
from urllib.error import HTTPError
from urllib.request import Request

import pytest

import cycle_forecast.data.oura_client as oura_client
from cycle_forecast.data.oura_client import (
    OuraApiError,
    OuraRoute,
    retrieve_collection,
)


def _readiness_page(*, document_id: str, next_token: str | None) -> bytes:
    """Create one invented Oura readiness page."""
    token = "null" if next_token is None else f'"{next_token}"'
    return (
        '{"data":[{"id":"' + document_id + '","contributors":{},"day":"2025-01-15",'
        '"timestamp":"2025-01-15T00:00:00-05:00"}],"next_token":' + token + "}"
    ).encode()


def test_retrieve_collection_follows_unique_pagination_tokens() -> None:
    """Fetch all pages with bounded dates and bearer authorization."""
    payloads = iter(
        (
            _readiness_page(document_id="synthetic-1", next_token="page-2"),
            _readiness_page(document_id="synthetic-2", next_token=None),
        )
    )
    requests: list[Request] = []

    def transport(request: Request) -> bytes:
        """Record the request and return the next invented response."""
        requests.append(request)
        return next(payloads)

    pages = retrieve_collection(
        route=OuraRoute.DAILY_READINESS,
        access_token="synthetic-token",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        transport=transport,
    )

    assert len(pages) == 2
    assert "start_date=2025-01-01" in requests[0].full_url
    assert "next_token=page-2" in requests[1].full_url
    assert requests[0].get_header("Authorization") == "Bearer synthetic-token"


def test_retrieve_collection_rejects_repeated_pagination_token() -> None:
    """Stop rather than loop forever on a malformed pagination sequence."""
    payload = _readiness_page(document_id="synthetic", next_token="repeat")

    with pytest.raises(OuraApiError, match="repeated"):
        retrieve_collection(
            route=OuraRoute.DAILY_READINESS,
            access_token="synthetic-token",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            transport=lambda _: payload,
        )


def test_retrieve_collection_rejects_schema_drift() -> None:
    """Reject undocumented response fields at the external boundary."""
    with pytest.raises(OuraApiError, match="invalid Oura"):
        retrieve_collection(
            route=OuraRoute.DAILY_READINESS,
            access_token="synthetic-token",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            transport=lambda _: b'{"data":[],"next_token":null,"new":1}',
        )


def test_retrieve_collection_requires_nonempty_access_token() -> None:
    """Reject a missing credential before constructing a request."""
    with pytest.raises(ValueError, match="access_token"):
        retrieve_collection(
            route=OuraRoute.DAILY_SLEEP,
            access_token="",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
        )


def test_default_transport_reads_response_without_logging_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real transport boundary with an invented response."""

    class Response:
        """Provide a context-managed invented API response."""

        def __enter__(self) -> Self:
            """Enter the response context."""
            return self

        def __exit__(self, *_: object) -> None:
            """Exit the response context."""

        def read(self) -> bytes:
            """Return an empty valid collection."""
            return b'{"data":[],"next_token":null}'

    def urlopen(_: Request, timeout: int) -> Response:
        """Return the invented response at the network seam."""
        assert timeout == 30
        return Response()

    monkeypatch.setattr(oura_client, "urlopen", urlopen)

    pages = retrieve_collection(
        route=OuraRoute.DAILY_SLEEP,
        access_token="synthetic-token",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 2),
    )

    assert len(pages) == 1


@pytest.mark.parametrize(
    ("status", "message"),
    [(401, "expired"), (403, "forbidden"), (429, "rate limit"), (500, "HTTP 500")],
)
def test_default_transport_maps_http_errors_without_response_bodies(
    status: int, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn Oura HTTP failures into concise, non-sensitive errors."""

    def urlopen(request: Request, timeout: int) -> bytes:
        """Raise one invented Oura HTTP failure."""
        raise HTTPError(request.full_url, status, "synthetic", hdrs=Message(), fp=None)

    monkeypatch.setattr(oura_client, "urlopen", urlopen)

    with pytest.raises(OuraApiError, match=message):
        retrieve_collection(
            route=OuraRoute.DAILY_SLEEP,
            access_token="synthetic-token",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
        )
