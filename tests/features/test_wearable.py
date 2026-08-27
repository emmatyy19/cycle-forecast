"""Test cutoff-safe daily wearable feature and censoring semantics."""

from datetime import UTC, date, datetime

from cycle_forecast.data.wearable_alignment import AlignedDailyObservation
from cycle_forecast.features.wearable import build_wearable_feature_row


def _aligned() -> AlignedDailyObservation:
    """Create one invented aligned morning with explicitly missing Oura data."""
    return AlignedDailyObservation(
        prediction_date=date(2025, 1, 10),
        prediction_cutoff=datetime(2025, 1, 10, 8, tzinfo=UTC),
        cycle_start_date=date(2025, 1, 1),
        cycle_day=10,
        oura=None,
    )


def test_builds_missingness_features_and_exact_event_offset() -> None:
    """Keep absent wearables distinct from observed zeros and attach the label."""
    row = build_wearable_feature_row(
        aligned=_aligned(),
        next_cycle_start=date(2025, 1, 12),
        observed_through=date(2025, 1, 12),
    )

    assert row.outcome_offset_days == 2
    assert row.values[0] == 10.0
    assert row.values[6:] == (1.0, 1.0, 1.0, 1.0, 1.0)


def test_retains_right_censoring_until_complete_horizon_is_observed() -> None:
    """Do not mislabel an incomplete dataset tail as no event."""
    censored = build_wearable_feature_row(
        aligned=_aligned(),
        next_cycle_start=None,
        observed_through=date(2025, 1, 20),
    )
    observed_later = build_wearable_feature_row(
        aligned=_aligned(),
        next_cycle_start=None,
        observed_through=date(2025, 1, 24),
    )

    assert censored.outcome_offset_days is None
    assert observed_later.outcome_offset_days == 15
