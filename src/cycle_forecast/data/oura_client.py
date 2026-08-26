"""Retrieve and validate bounded Oura API V2 collections."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel

from cycle_forecast.data.oura import (
    OuraDailyReadinessResponse,
    OuraDailySleepResponse,
    OuraSleepResponse,
)

OURA_API_BASE_URL: Final = "https://api.ouraring.com/v2/usercollection"
MAX_PAGES: Final = 1_000


class OuraRoute(StrEnum):
    """Identify the Oura collections supported by this project."""

    SLEEP = "sleep"
    DAILY_SLEEP = "daily_sleep"
    DAILY_READINESS = "daily_readiness"


class OuraApiError(RuntimeError):
    """Indicate a transport, protocol, or response-validation failure."""


@dataclass(frozen=True, slots=True, kw_only=True)
class OuraPage:
    """Contain one validated page and its original response bytes."""

    route: OuraRoute
    payload: bytes
    model: BaseModel
    next_token: str | None


Transport = Callable[[Request], bytes]


def _default_transport(request: Request) -> bytes:
    """Execute one Oura request using the standard library."""
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as error:
        if error.code == 401:
            message = "Oura authorization expired or was revoked"
        elif error.code == 403:
            message = "Oura access was forbidden; check membership and scopes"
        elif error.code == 429:
            message = "Oura rate limit exceeded; retry later"
        else:
            message = f"Oura request failed with HTTP {error.code}"
        raise OuraApiError(message) from error
    except OSError as error:
        raise OuraApiError("Oura request failed") from error


def _validate_page(*, route: OuraRoute, payload: bytes) -> OuraPage:
    """Validate one response against its exact Pydantic boundary model."""
    model_type: type[BaseModel]
    if route is OuraRoute.SLEEP:
        model_type = OuraSleepResponse
    elif route is OuraRoute.DAILY_SLEEP:
        model_type = OuraDailySleepResponse
    else:
        model_type = OuraDailyReadinessResponse
    try:
        model = model_type.model_validate_json(payload)
    except ValueError as error:
        raise OuraApiError(f"invalid Oura {route.value} response") from error
    next_token = model.next_token
    return OuraPage(
        route=route,
        payload=payload,
        model=model,
        next_token=next_token,
    )


def retrieve_collection(
    *,
    route: OuraRoute,
    access_token: str,
    start_date: date,
    end_date: date,
    transport: Transport = _default_transport,
) -> tuple[OuraPage, ...]:
    """Retrieve every validated page in one bounded date range.

    Raises
    ------
    ValueError
        If the date range or token is invalid.
    OuraApiError
        If transport, validation, or pagination fails.
    """
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if not access_token:
        raise ValueError("access_token must be non-empty")

    pages: list[OuraPage] = []
    next_token: str | None = None
    seen_tokens: set[str] = set()
    for _ in range(MAX_PAGES):
        query = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        if next_token is not None:
            query["next_token"] = next_token
        request = Request(
            f"{OURA_API_BASE_URL}/{route.value}?{urlencode(query)}",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        page = _validate_page(route=route, payload=transport(request))
        pages.append(page)
        next_token = page.next_token
        if next_token is None:
            return tuple(pages)
        if next_token in seen_tokens:
            raise OuraApiError("Oura pagination token repeated")
        seen_tokens.add(next_token)
    raise OuraApiError("Oura pagination exceeded the safety limit")


def count_documents(*, pages: tuple[OuraPage, ...]) -> int:
    """Count documents without exposing their values."""
    return sum(len(json.loads(page.payload)["data"]) for page in pages)
