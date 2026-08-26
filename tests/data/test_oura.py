"""Tests for strict Oura API boundary models."""

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from cycle_forecast.data.oura import (
    OURA_OPENAPI_SPECIFICATION_VERSION,
    OuraDailyReadinessResponse,
    OuraSleep,
    RetrievedOuraDailyObservation,
)


def test_load_documented_daily_readiness_json() -> None:
    """Parse the exact documented response shape directly from JSON."""
    response = OuraDailyReadinessResponse.model_validate_json(
        '{"data":[{"id":"synthetic-readiness","contributors":{},'
        '"day":"2025-01-15","score":81,"temperature_deviation":0.2,'
        '"temperature_trend_deviation":null,'
        '"timestamp":"2025-01-15T00:00:00-05:00"}],"next_token":null}'
    )

    assert OURA_OPENAPI_SPECIFICATION_VERSION == "1.35"
    assert response.data[0].temperature_deviation == 0.2


def test_reject_undocumented_oura_field() -> None:
    """Expose upstream schema drift instead of silently ignoring it."""
    payload = (
        '{"data":[{"id":"synthetic-readiness","contributors":{},'
        '"day":"2025-01-15","timestamp":"2025-01-15T00:00:00-05:00",'
        '"new_metric":4}],"next_token":null}'
    )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        OuraDailyReadinessResponse.model_validate_json(payload)


def test_reject_missing_required_pagination_token() -> None:
    """Match Oura's requirement that next_token is present even when null."""
    with pytest.raises(ValidationError, match="next_token"):
        OuraDailyReadinessResponse.model_validate_json('{"data":[]}')


def test_retrieved_observation_requires_matching_day() -> None:
    """Prevent a source-day mismatch at the normalization boundary."""
    sleep = OuraSleep(
        id="synthetic-sleep",
        bedtime_end="2025-01-16T07:00:00-05:00",
        bedtime_start="2025-01-15T23:00:00-05:00",
        day="2025-01-16",
        low_battery_alert=False,
        period=0,
        time_in_bed=28_800,
    )

    with pytest.raises(ValidationError, match="must match"):
        RetrievedOuraDailyObservation(
            day=date(2025, 1, 15),
            available_at=datetime(2025, 1, 16, 8, tzinfo=UTC),
            main_sleep=sleep,
        )
