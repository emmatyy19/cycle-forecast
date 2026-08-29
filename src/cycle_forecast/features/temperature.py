"""Build cutoff-safe temperature trajectories within each menstrual cycle."""

from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from math import isfinite
from typing import Final

from cycle_forecast.data.wearable_alignment import AlignedDailyObservation
from cycle_forecast.features.wearable import WearableFeatureRow

TEMPERATURE_TRAJECTORY_FEATURE_VERSION: Final = "temperature-trajectory-features-v1"
"""Version of recent-level, slope, drop, and elevated-streak semantics."""

TEMPERATURE_TRAJECTORY_FEATURE_NAMES: Final = (
    "cycle_day",
    "temperature_deviation_celsius",
    "temperature_mean_3_days",
    "temperature_mean_7_days",
    "temperature_slope_7_days",
    "temperature_drop_from_7_day_maximum",
    "temperature_elevated_streak_days",
    "temperature_missing",
    "temperature_mean_3_days_missing",
    "temperature_mean_7_days_missing",
    "temperature_slope_7_days_missing",
    "temperature_drop_from_7_day_maximum_missing",
    "temperature_elevated_streak_missing",
)
"""Stable order for the temperature-trajectory neighbor candidate."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TemperatureTrajectoryRow:
    """Contain one cutoff-safe temperature trajectory and eventual outcome."""

    aligned: AlignedDailyObservation
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    outcome_offset_days: int | None

    def __post_init__(self) -> None:
        """Require the versioned finite feature vector and valid outcome."""
        if self.feature_names != TEMPERATURE_TRAJECTORY_FEATURE_NAMES:
            raise ValueError("temperature trajectory names do not match the schema")
        if len(self.values) != len(self.feature_names):
            raise ValueError("temperature trajectory values do not match names")
        if any(not isfinite(value) for value in self.values):
            raise ValueError("temperature trajectory values must be finite")
        if self.outcome_offset_days is not None and self.outcome_offset_days < 0:
            raise ValueError("temperature trajectory outcome must be nonnegative")


def _temperature(*, row: WearableFeatureRow) -> float | None:
    """Return the observed temperature deviation from a wearable feature row."""
    return None if row.values[7] == 1.0 else row.values[2]


def _value_and_missing(*, value: float | int | None) -> tuple[float, float]:
    """Represent a nullable trajectory value without treating zero as observed."""
    return (0.0, 1.0) if value is None else (float(value), 0.0)


def _mean(*, values: tuple[float, ...]) -> float | None:
    """Calculate an observed-value mean or preserve an empty window as missing."""
    return sum(values) / len(values) if values else None


def _slope(*, points: tuple[tuple[int, float], ...]) -> float | None:
    """Fit a least-squares daily slope when two observed dates are available."""
    if len(points) < 2:
        return None
    mean_day = sum(day for day, _ in points) / len(points)
    mean_value = sum(value for _, value in points) / len(points)
    denominator = sum((day - mean_day) ** 2 for day, _ in points)
    if denominator == 0.0:
        return None
    return (
        sum((day - mean_day) * (value - mean_value) for day, value in points)
        / denominator
    )


def _elevated_streak(*, rows: tuple[WearableFeatureRow, ...]) -> int | None:
    """Count consecutive observed nights above the personal Oura baseline."""
    if not rows or _temperature(row=rows[-1]) is None:
        return None
    streak = 0
    expected_date = rows[-1].aligned.prediction_date
    for row in reversed(rows):
        if row.aligned.prediction_date != expected_date:
            break
        temperature = _temperature(row=row)
        if temperature is None or temperature <= 0.0:
            break
        streak += 1
        expected_date -= timedelta(days=1)
    return streak


def build_temperature_trajectory_rows(
    *, rows: tuple[WearableFeatureRow, ...]
) -> tuple[TemperatureTrajectoryRow, ...]:
    """Transform chronological mornings using only same-cycle past observations.

    Parameters
    ----------
    rows
        Strictly chronological cutoff-safe wearable rows. Gaps are retained and
        never interpolated.

    Returns
    -------
    tuple[TemperatureTrajectoryRow, ...]
        One trajectory vector for every input morning in the same order.

    Raises
    ------
    ValueError
        If rows are not strictly chronological or dates repeat.
    """
    if any(
        current.aligned.prediction_date >= following.aligned.prediction_date
        for current, following in pairwise(rows)
    ):
        raise ValueError("temperature trajectory rows must be strictly chronological")
    transformed: list[TemperatureTrajectoryRow] = []
    for row in rows:
        same_cycle = tuple(
            candidate
            for candidate in rows
            if candidate.aligned.cycle_start_date == row.aligned.cycle_start_date
            and candidate.aligned.prediction_date <= row.aligned.prediction_date
            and candidate.aligned.prediction_cutoff <= row.aligned.prediction_cutoff
        )
        recent_three = tuple(
            candidate
            for candidate in same_cycle
            if (row.aligned.prediction_date - candidate.aligned.prediction_date).days
            <= 2
        )
        recent_seven = tuple(
            candidate
            for candidate in same_cycle
            if (row.aligned.prediction_date - candidate.aligned.prediction_date).days
            <= 6
        )
        observed_three: list[float] = []
        for candidate in recent_three:
            value = _temperature(row=candidate)
            if value is not None:
                observed_three.append(value)
        three_values = tuple(observed_three)
        observed_seven: list[tuple[int, float]] = []
        for candidate in recent_seven:
            value = _temperature(row=candidate)
            if value is not None:
                observed_seven.append(
                    (
                        (
                            candidate.aligned.prediction_date
                            - row.aligned.prediction_date
                        ).days,
                        value,
                    )
                )
        seven_points = tuple(observed_seven)
        seven_values = tuple(value for _, value in seven_points)
        current_temperature = _temperature(row=row)
        mean_three = _mean(values=three_values)
        mean_seven = _mean(values=seven_values)
        slope_seven = _slope(points=seven_points)
        drop_from_maximum = (
            current_temperature - max(seven_values)
            if current_temperature is not None and seven_values
            else None
        )
        elevated_streak = _elevated_streak(rows=same_cycle)
        current_value, current_missing = _value_and_missing(value=current_temperature)
        mean_three_value, mean_three_missing = _value_and_missing(value=mean_three)
        mean_seven_value, mean_seven_missing = _value_and_missing(value=mean_seven)
        slope_value, slope_missing = _value_and_missing(value=slope_seven)
        drop_value, drop_missing = _value_and_missing(value=drop_from_maximum)
        streak_value, streak_missing = _value_and_missing(value=elevated_streak)
        transformed.append(
            TemperatureTrajectoryRow(
                aligned=row.aligned,
                feature_names=TEMPERATURE_TRAJECTORY_FEATURE_NAMES,
                values=(
                    float(row.aligned.cycle_day),
                    current_value,
                    mean_three_value,
                    mean_seven_value,
                    slope_value,
                    drop_value,
                    streak_value,
                    current_missing,
                    mean_three_missing,
                    mean_seven_missing,
                    slope_missing,
                    drop_missing,
                    streak_missing,
                ),
                outcome_offset_days=row.outcome_offset_days,
            )
        )
    return tuple(transformed)
