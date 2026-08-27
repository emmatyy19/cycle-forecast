"""Build allowlisted daily wearable features and censoring-aware labels."""

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Final

from cycle_forecast.data.wearable_alignment import AlignedDailyObservation

WEARABLE_FEATURE_VERSION: Final = "wearable-daily-features-v1"
"""Semantic version of the daily wearable feature allowlist."""

WEARABLE_FEATURE_NAMES: Final = (
    "cycle_day",
    "readiness_score",
    "temperature_deviation_celsius",
    "sleep_score",
    "average_hrv_milliseconds",
    "total_sleep_duration_seconds",
    "readiness_missing",
    "temperature_missing",
    "sleep_score_missing",
    "average_hrv_missing",
    "total_sleep_duration_missing",
)
"""Stable model-input order with explicit missingness indicators."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableFeatureRow:
    """Pair one cutoff-safe morning vector with an exact or censored label."""

    aligned: AlignedDailyObservation
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    outcome_offset_days: int | None

    def __post_init__(self) -> None:
        """Require a stable finite feature vector and nonnegative label."""
        if self.feature_names != WEARABLE_FEATURE_NAMES:
            raise ValueError("wearable feature names do not match the versioned schema")
        if len(self.values) != len(self.feature_names):
            raise ValueError("wearable feature values do not match feature names")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("wearable feature values must be finite")
        if self.outcome_offset_days is not None and self.outcome_offset_days < 0:
            raise ValueError("outcome offset must be nonnegative")


def _value_and_missing(*, value: int | float | None) -> tuple[float, float]:
    """Represent a nullable measurement without treating missing as observed zero."""
    return (0.0, 1.0) if value is None else (float(value), 0.0)


def build_wearable_feature_row(
    *,
    aligned: AlignedDailyObservation,
    next_cycle_start: date | None,
    observed_through: date,
) -> WearableFeatureRow:
    """Transform one aligned morning and establish its censoring-safe label.

    Parameters
    ----------
    aligned
        Cutoff-safe cycle and Oura observation.
    next_cycle_start
        First period start on or after the prediction date, when recorded.
    observed_through
        Last local date whose absence of a period start is known.

    Returns
    -------
    WearableFeatureRow
        Stable numeric vector and exact offset, or ``None`` when right-censored.

    Raises
    ------
    ValueError
        If dates contradict the aligned prediction context.
    """
    if observed_through < aligned.prediction_date:
        raise ValueError("observed_through cannot precede prediction_date")
    if next_cycle_start is not None and next_cycle_start < aligned.prediction_date:
        raise ValueError("next_cycle_start cannot precede prediction_date")
    if next_cycle_start is not None and next_cycle_start > observed_through:
        raise ValueError("next_cycle_start cannot follow observed_through")
    outcome = (
        (next_cycle_start - aligned.prediction_date).days
        if next_cycle_start is not None
        else None
    )
    if outcome is None and (observed_through - aligned.prediction_date).days >= 14:
        outcome = 15

    oura = aligned.oura
    readiness_score = oura.readiness.score if oura and oura.readiness else None
    temperature = (
        oura.readiness.temperature_deviation if oura and oura.readiness else None
    )
    sleep_score = oura.daily_sleep.score if oura and oura.daily_sleep else None
    average_hrv = oura.main_sleep.average_hrv if oura and oura.main_sleep else None
    total_sleep = (
        oura.main_sleep.total_sleep_duration if oura and oura.main_sleep else None
    )
    readiness_value, readiness_missing = _value_and_missing(value=readiness_score)
    temperature_value, temperature_missing = _value_and_missing(value=temperature)
    sleep_value, sleep_missing = _value_and_missing(value=sleep_score)
    hrv_value, hrv_missing = _value_and_missing(value=average_hrv)
    total_sleep_value, total_sleep_missing = _value_and_missing(value=total_sleep)
    return WearableFeatureRow(
        aligned=aligned,
        feature_names=WEARABLE_FEATURE_NAMES,
        values=(
            float(aligned.cycle_day),
            readiness_value,
            temperature_value,
            sleep_value,
            hrv_value,
            total_sleep_value,
            readiness_missing,
            temperature_missing,
            sleep_missing,
            hrv_missing,
            total_sleep_missing,
        ),
        outcome_offset_days=outcome,
    )
