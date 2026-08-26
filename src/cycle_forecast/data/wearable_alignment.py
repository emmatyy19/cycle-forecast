"""Align cycle history and retrieved Oura observations without leakage."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise

from cycle_forecast.data.cycle_history import CycleHistoryRecord
from cycle_forecast.data.oura import RetrievedOuraDailyObservation

WEARABLE_ALIGNMENT_VERSION = "oura-alignment-v1"
"""Semantic version of wearable and cycle-history alignment rules."""


class WearableAlignmentError(ValueError):
    """Indicate invalid or temporally unsafe alignment input."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AlignedDailyObservation:
    """Represent one cutoff-safe daily observation within a cycle.

    Parameters
    ----------
    prediction_date
        Local calendar date being predicted.
    prediction_cutoff
        Timezone-aware retrieval cutoff.
    cycle_start_date
        Most recent period start strictly before the prediction date.
    cycle_day
        One-based day within that cycle.
    oura
        Oura data proven available by the cutoff, if present.
    """

    prediction_date: date
    prediction_cutoff: datetime
    cycle_start_date: date
    cycle_day: int
    oura: RetrievedOuraDailyObservation | None


def align_daily_observation(
    *,
    prediction_date: date,
    prediction_cutoff: datetime,
    cycle_history: Sequence[CycleHistoryRecord],
    oura_observations: Sequence[RetrievedOuraDailyObservation],
) -> AlignedDailyObservation:
    """Align one morning prediction using only cutoff-available inputs.

    Parameters
    ----------
    prediction_date
        Local date for which the start probability is predicted.
    prediction_cutoff
        Retrieval start instant; it must be timezone-aware.
    cycle_history
        Chronological period starts. Starts on or after the prediction date are
        outcomes and cannot identify the current cycle.
    oura_observations
        Validated observations with historical availability provenance.

    Returns
    -------
    AlignedDailyObservation
        The cutoff-safe cycle context and matching Oura observation.

    Raises
    ------
    WearableAlignmentError
        If input is unordered, ambiguous, or temporally inconsistent.
    """
    if prediction_cutoff.tzinfo is None or prediction_cutoff.utcoffset() is None:
        raise WearableAlignmentError("prediction_cutoff must be timezone-aware")
    starts = tuple(record.cycle_start_date for record in cycle_history)
    if any(current >= following for current, following in pairwise(starts)):
        raise WearableAlignmentError("cycle history must be strictly chronological")
    prior_starts = tuple(start for start in starts if start < prediction_date)
    if not prior_starts:
        raise WearableAlignmentError("prediction date requires an earlier cycle start")

    matching = tuple(item for item in oura_observations if item.day == prediction_date)
    if len(matching) > 1:
        raise WearableAlignmentError("Oura observation days must be unique")
    oura = matching[0] if matching else None
    if oura is not None and oura.available_at > prediction_cutoff:
        oura = None
    if oura is not None and oura.main_sleep is not None:
        try:
            sleep_end = datetime.fromisoformat(oura.main_sleep.bedtime_end)
        except ValueError as error:
            raise WearableAlignmentError(
                "sleep bedtime_end must be ISO 8601"
            ) from error
        if sleep_end.tzinfo is None or sleep_end.utcoffset() is None:
            raise WearableAlignmentError("sleep bedtime_end must include an offset")
        if sleep_end > prediction_cutoff:
            raise WearableAlignmentError("sleep ending after cutoff cannot be aligned")

    cycle_start = prior_starts[-1]
    return AlignedDailyObservation(
        prediction_date=prediction_date,
        prediction_cutoff=prediction_cutoff,
        cycle_start_date=cycle_start,
        cycle_day=(prediction_date - cycle_start).days + 1,
        oura=oura,
    )
