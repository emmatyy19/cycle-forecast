"""Test the fixed final temporal holdout policy."""

from datetime import date, timedelta

import pytest

from cycle_forecast.data import CycleDataset, CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.forecasting import (
    FINAL_HOLDOUT_CYCLE_COUNT,
    FINAL_HOLDOUT_POLICY_VERSION,
    split_final_temporal_holdout,
)


def _dataset(*, cycle_count: int) -> CycleDataset:
    """Build a synthetic dataset with a requested number of completed cycles.

    Parameters
    ----------
    cycle_count
        Number of completed rows to construct.

    Returns
    -------
    CycleDataset
        Synthetic chronological dataset.
    """
    starts = tuple(
        CycleHistoryRecord(
            cycle_start_date=date(2020, 1, 1) + timedelta(days=28 * position),
            period_length_days=5,
        )
        for position in range(cycle_count + 1)
    )
    return build_cycle_dataset(records=starts)


def test_final_holdout_reserves_latest_fixed_block() -> None:
    """Partition without overlap, gaps, reordering, or lost provenance."""
    dataset = _dataset(cycle_count=20)

    split = split_final_temporal_holdout(dataset=dataset)

    assert split.dataset_fingerprint == dataset.fingerprint
    assert split.policy_version == FINAL_HOLDOUT_POLICY_VERSION
    assert len(split.development_rows) == 8
    assert len(split.holdout_rows) == FINAL_HOLDOUT_CYCLE_COUNT
    assert split.development_rows + split.holdout_rows == dataset.rows
    assert (
        split.development_rows[-1].next_cycle_start_date
        == split.holdout_rows[0].cycle_start_date
    )


def test_final_holdout_boundary_keeps_one_development_cycle() -> None:
    """Allow the smallest dataset that preserves development history."""
    dataset = _dataset(cycle_count=FINAL_HOLDOUT_CYCLE_COUNT + 1)

    split = split_final_temporal_holdout(dataset=dataset)

    assert len(split.development_rows) == 1
    assert len(split.holdout_rows) == FINAL_HOLDOUT_CYCLE_COUNT


@pytest.mark.parametrize(
    "cycle_count",
    [0, FINAL_HOLDOUT_CYCLE_COUNT - 1, FINAL_HOLDOUT_CYCLE_COUNT],
)
def test_final_holdout_rejects_insufficient_history(cycle_count: int) -> None:
    """Reject datasets that cannot preserve any development rows.

    Parameters
    ----------
    cycle_count
        Insufficient completed-cycle count.
    """
    dataset = _dataset(cycle_count=cycle_count)

    with pytest.raises(ValueError, match="requires at least 13 completed cycles"):
        split_final_temporal_holdout(dataset=dataset)
