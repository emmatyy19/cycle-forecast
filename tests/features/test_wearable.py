"""Test cutoff-safe daily wearable feature and censoring semantics."""

from datetime import UTC, date, datetime
from math import nan

import pytest

from cycle_forecast.data.wearable_alignment import AlignedDailyObservation
from cycle_forecast.features.wearable import (
    WEARABLE_FEATURE_NAMES,
    WearableFeatureRow,
    build_wearable_feature_row,
)


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


@pytest.mark.parametrize(
    ("next_start", "observed_through", "message"),
    (
        (None, date(2025, 1, 9), "cannot precede"),
        (date(2025, 1, 9), date(2025, 1, 20), "cannot precede"),
        (date(2025, 1, 21), date(2025, 1, 20), "cannot follow"),
    ),
)
def test_rejects_inconsistent_label_dates(
    next_start: date | None, observed_through: date, message: str
) -> None:
    """Reject label dates that contradict the prediction timeline."""
    with pytest.raises(ValueError, match=message):
        build_wearable_feature_row(
            aligned=_aligned(),
            next_cycle_start=next_start,
            observed_through=observed_through,
        )


@pytest.mark.parametrize(
    ("names", "values", "outcome", "message"),
    (
        (("wrong",), (1.0,), None, "names"),
        (WEARABLE_FEATURE_NAMES, (1.0,), None, "values"),
        (WEARABLE_FEATURE_NAMES, (nan,) * 11, None, "finite"),
        (WEARABLE_FEATURE_NAMES, (0.0,) * 11, -1, "nonnegative"),
    ),
)
def test_rejects_invalid_wearable_feature_rows(
    names: tuple[str, ...],
    values: tuple[float, ...],
    outcome: int | None,
    message: str,
) -> None:
    """Enforce the versioned feature-row schema."""
    with pytest.raises(ValueError, match=message):
        WearableFeatureRow(
            aligned=_aligned(),
            feature_names=names,
            values=values,
            outcome_offset_days=outcome,
        )
