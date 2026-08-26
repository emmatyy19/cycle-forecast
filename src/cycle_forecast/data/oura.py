"""Validate Oura API V2 payloads against OpenAPI specification 1.35.

See https://api.ouraring.com/v2/static/json/openapi-1.35.json and
https://cloud.ouraring.com/v2/docs.
"""

from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

OURA_OPENAPI_SPECIFICATION_VERSION = "1.35"
"""Version of the Oura OpenAPI document represented by these models."""

NonEmptyString = Annotated[str, StringConstraints(min_length=1)]


class OuraBoundaryModel(BaseModel):
    """Provide strict, immutable validation for an Oura wire object."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OuraReadinessContributors(OuraBoundaryModel):
    """Represent documented readiness-score contributors."""

    activity_balance: int | None = None
    body_temperature: int | None = None
    hrv_balance: int | None = None
    previous_day_activity: int | None = None
    previous_night: int | None = None
    recovery_index: int | None = None
    resting_heart_rate: int | None = None
    sleep_balance: int | None = None
    sleep_regularity: int | None = None


class OuraSleepContributors(OuraBoundaryModel):
    """Represent documented sleep-score contributors."""

    deep_sleep: int | None = None
    efficiency: int | None = None
    latency: int | None = None
    rem_sleep: int | None = None
    restfulness: int | None = None
    timing: int | None = None
    total_sleep: int | None = None


class OuraReadiness(OuraBoundaryModel):
    """Represent readiness details nested in an Oura sleep document."""

    contributors: OuraReadinessContributors
    score: int | None = None
    temperature_deviation: float | None = None
    temperature_trend_deviation: float | None = None


class OuraSample(OuraBoundaryModel):
    """Represent one documented sampled Oura time series."""

    interval: float
    items: list[float | None]
    timestamp: str


class OuraSleepAlgorithmVersion(StrEnum):
    """Enumerate documented Oura sleep-algorithm versions."""

    V1 = "v1"
    V2 = "v2"


class OuraSleepAnalysisReason(StrEnum):
    """Enumerate documented reasons for Oura sleep analysis."""

    FOREGROUND_SLEEP_ANALYSIS = "foreground_sleep_analysis"
    BEDTIME_EDIT = "bedtime_edit"
    BACKGROUND_SLEEP_ANALYSIS = "background_sleep_analysis"
    BACKGROUND_CREATED_FOREGROUND_UPDATED = "background_created_foreground_updated"


class OuraSleepType(StrEnum):
    """Enumerate documented Oura sleep-period types."""

    DELETED = "deleted"
    SLEEP = "sleep"
    LONG_SLEEP = "long_sleep"
    LATE_NAP = "late_nap"
    REST = "rest"


class OuraDailyReadiness(OuraBoundaryModel):
    """Represent an Oura ``daily_readiness`` document.

    See https://cloud.ouraring.com/v2/docs#tag/Daily-Readiness-Routes.
    """

    id: NonEmptyString
    contributors: OuraReadinessContributors
    day: str
    score: int | None = None
    temperature_deviation: float | None = None
    temperature_trend_deviation: float | None = None
    timestamp: str


class OuraDailySleep(OuraBoundaryModel):
    """Represent an Oura ``daily_sleep`` document.

    See https://cloud.ouraring.com/v2/docs#tag/Daily-Sleep-Routes.
    """

    id: NonEmptyString
    contributors: OuraSleepContributors
    day: str
    score: int | None = None
    timestamp: str


class OuraSleep(OuraBoundaryModel):
    """Represent an Oura ``sleep`` document.

    See https://cloud.ouraring.com/v2/docs#tag/Sleep-Routes.
    """

    id: NonEmptyString
    average_breath: float | None = None
    average_heart_rate: float | None = None
    average_hrv: int | None = None
    awake_time: int | None = None
    bedtime_end: str
    bedtime_start: str
    day: str
    deep_sleep_duration: int | None = None
    efficiency: int | None = None
    heart_rate: OuraSample | None = None
    hrv: OuraSample | None = None
    latency: int | None = None
    light_sleep_duration: int | None = None
    low_battery_alert: bool
    lowest_heart_rate: int | None = None
    movement_30_sec: str | None = None
    period: int
    readiness: OuraReadiness | None = None
    readiness_score_delta: int | None = None
    rem_sleep_duration: int | None = None
    restless_periods: int | None = None
    sleep_algorithm_version: OuraSleepAlgorithmVersion | None = None
    sleep_analysis_reason: OuraSleepAnalysisReason | None = None
    sleep_phase_30_sec: str | None = None
    sleep_phase_5_min: str | None = None
    sleep_score_delta: int | None = None
    time_in_bed: int
    total_sleep_duration: int | None = None
    type: OuraSleepType | None = None
    ring_id: str | None = None
    app_sleep_phase_5_min: str | None = None


class OuraDailyReadinessResponse(OuraBoundaryModel):
    """Represent the documented paginated daily-readiness response."""

    data: list[OuraDailyReadiness]
    next_token: str | None


class OuraDailySleepResponse(OuraBoundaryModel):
    """Represent the documented paginated daily-sleep response."""

    data: list[OuraDailySleep]
    next_token: str | None


class OuraSleepResponse(OuraBoundaryModel):
    """Represent the documented paginated sleep response."""

    data: list[OuraSleep]
    next_token: str | None


class RetrievedOuraDailyObservation(OuraBoundaryModel):
    """Join validated Oura daily documents to their known availability time."""

    day: date
    available_at: datetime
    readiness: OuraDailyReadiness | None = None
    daily_sleep: OuraDailySleep | None = None
    main_sleep: OuraSleep | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        """Require timezone-aware availability and consistent source days.

        Returns
        -------
        Self
            The validated observation.

        Raises
        ------
        ValueError
            If availability lacks a timezone or a document has another day.
        """
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("available_at must be timezone-aware")
        documents = (self.readiness, self.daily_sleep, self.main_sleep)
        if not any(document is not None for document in documents):
            raise ValueError("at least one Oura document is required")
        if any(
            document is not None and document.day != self.day.isoformat()
            for document in documents
        ):
            raise ValueError("Oura document days must match observation day")
        return self
