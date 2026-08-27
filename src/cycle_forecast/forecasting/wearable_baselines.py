"""Provide daily history and wearable-informed probabilistic baselines."""

from math import sqrt
from statistics import fmean, pstdev
from typing import Final

from cycle_forecast.features.wearable import WearableFeatureRow
from cycle_forecast.forecasting.daily import (
    DAILY_FORECAST_HORIZON_DAYS,
    DailyPeriodDistribution,
    distribution_from_hazards,
)

EMPIRICAL_HAZARD_BASELINE_VERSION: Final = "empirical-cycle-hazard-v1"
"""Semantic version of the smoothed completed-cycle hazard baseline."""

WEARABLE_NEIGHBOR_BASELINE_VERSION: Final = "wearable-neighbor-v1"
"""Semantic version of the nonparametric wearable-informed baseline."""


def forecast_with_empirical_cycle_hazard(
    *,
    row: WearableFeatureRow,
    completed_cycle_lengths: tuple[int, ...],
    smoothing: float = 1.0,
) -> DailyPeriodDistribution:
    """Forecast daily start probabilities from prior completed-cycle hazards.

    Parameters
    ----------
    row
        Current cutoff-safe morning row.
    completed_cycle_lengths
        Positive cycle lengths completed strictly before the prediction cutoff.
    smoothing
        Positive beta-binomial pseudocount applied to event and survival counts.

    Returns
    -------
    DailyPeriodDistribution
        Exhaustive distribution for today through day 14 and later.
    """
    if not completed_cycle_lengths or any(
        value < 1 for value in completed_cycle_lengths
    ):
        raise ValueError("completed_cycle_lengths must be positive and nonempty")
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")
    elapsed_days = row.aligned.cycle_day - 1
    hazards: list[float] = []
    for offset in range(DAILY_FORECAST_HORIZON_DAYS):
        event_day = elapsed_days + offset
        at_risk = sum(length >= event_day for length in completed_cycle_lengths)
        events = sum(length == event_day for length in completed_cycle_lengths)
        hazards.append((events + smoothing) / (at_risk + 2.0 * smoothing))
    return distribution_from_hazards(
        prediction_date=row.aligned.prediction_date,
        prediction_cutoff=row.aligned.prediction_cutoff,
        hazards=tuple(hazards),
    )


def _scales(*, rows: tuple[WearableFeatureRow, ...]) -> tuple[float, ...]:
    """Calculate training-only scales for cycle day and five measurements."""
    indices = (0, 1, 2, 3, 4, 5)
    missing_indices = (None, 6, 7, 8, 9, 10)
    scales: list[float] = []
    for value_index, missing_index in zip(indices, missing_indices, strict=True):
        values = tuple(
            row.values[value_index]
            for row in rows
            if missing_index is None or row.values[missing_index] == 0.0
        )
        deviation = pstdev(values) if len(values) > 1 else 0.0
        scales.append(deviation if deviation > 0.0 else 1.0)
    return tuple(scales)


def _distance(
    *,
    current: WearableFeatureRow,
    candidate: WearableFeatureRow,
    scales: tuple[float, ...],
) -> float:
    """Calculate missingness-aware standardized wearable distance."""
    total = ((current.values[0] - candidate.values[0]) / scales[0]) ** 2
    for position, (value_index, missing_index) in enumerate(
        zip((1, 2, 3, 4, 5), (6, 7, 8, 9, 10), strict=True),
        start=1,
    ):
        current_missing = current.values[missing_index] == 1.0
        candidate_missing = candidate.values[missing_index] == 1.0
        if current_missing != candidate_missing:
            total += 1.0
        elif not current_missing:
            total += (
                (current.values[value_index] - candidate.values[value_index])
                / scales[position]
            ) ** 2
    return sqrt(total)


def forecast_with_wearable_neighbors(
    *,
    row: WearableFeatureRow,
    training_rows: tuple[WearableFeatureRow, ...],
    neighbor_count: int = 20,
    smoothing: float = 1.0,
) -> DailyPeriodDistribution:
    """Forecast from similar prior mornings using an empirical distribution.

    Parameters
    ----------
    row
        Current cutoff-safe wearable row.
    training_rows
        Earlier uncensored rows whose labels are known at the cutoff.
    neighbor_count
        Positive maximum number of nearest earlier rows used.
    smoothing
        Positive symmetric pseudocount for all 16 outcome categories.

    Returns
    -------
    DailyPeriodDistribution
        Smoothed wearable-informed nearest-neighbor distribution.
    """
    labeled = tuple(
        candidate
        for candidate in training_rows
        if candidate.outcome_offset_days is not None
        and candidate.aligned.prediction_cutoff < row.aligned.prediction_cutoff
    )
    if not labeled:
        raise ValueError("wearable baseline requires earlier labeled rows")
    if neighbor_count < 1 or smoothing <= 0.0:
        raise ValueError("neighbor_count and smoothing must be positive")
    scales = _scales(rows=labeled)
    neighbors = tuple(
        candidate
        for _, candidate in sorted(
            (
                (_distance(current=row, candidate=candidate, scales=scales), candidate)
                for candidate in labeled
            ),
            key=lambda item: (item[0], item[1].aligned.prediction_cutoff),
        )[:neighbor_count]
    )
    counts = [smoothing] * (DAILY_FORECAST_HORIZON_DAYS + 1)
    for candidate in neighbors:
        assert candidate.outcome_offset_days is not None
        counts[min(candidate.outcome_offset_days, DAILY_FORECAST_HORIZON_DAYS)] += 1.0
    total = fmean(counts) * len(counts)
    return DailyPeriodDistribution(
        prediction_date=row.aligned.prediction_date,
        prediction_cutoff=row.aligned.prediction_cutoff,
        daily_probabilities=tuple(
            count / total for count in counts[:DAILY_FORECAST_HORIZON_DAYS]
        ),
        after_horizon_probability=counts[-1] / total,
    )
