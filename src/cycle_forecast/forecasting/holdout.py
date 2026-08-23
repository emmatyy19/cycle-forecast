"""Preserve a fixed final temporal holdout for cycle-history evaluation."""

from dataclasses import dataclass

from cycle_forecast.data import CycleDataset, CycleDatasetRow

FINAL_HOLDOUT_CYCLE_COUNT = 12
"""Number of most-recent completed cycles reserved for final evaluation."""

FINAL_HOLDOUT_POLICY_VERSION = "final-temporal-holdout-v1"
"""Semantic version of the final temporal holdout rule."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalHoldoutSplit:
    """Contain development and final-holdout rows with provenance.

    Parameters
    ----------
    dataset_fingerprint
        Identity of the complete dataset that was partitioned.
    policy_version
        Semantic version of the partition rule.
    development_rows
        Chronological completed-cycle rows available for model development.
    holdout_rows
        Chronological most-recent rows reserved for one final evaluation.
    """

    dataset_fingerprint: str
    policy_version: str
    development_rows: tuple[CycleDatasetRow, ...]
    holdout_rows: tuple[CycleDatasetRow, ...]


def split_final_temporal_holdout(*, dataset: CycleDataset) -> TemporalHoldoutSplit:
    """Reserve the latest fixed block of completed cycles as final holdout.

    Parameters
    ----------
    dataset
        Immutable completed-cycle dataset in chronological order.

    Returns
    -------
    TemporalHoldoutSplit
        Non-overlapping development and final-holdout rows whose concatenation
        exactly reproduces the input rows.

    Raises
    ------
    ValueError
        If the dataset does not contain at least one development row in addition
        to the fixed holdout block.

    Notes
    -----
    The holdout count is intentionally not a function argument. Changing it is
    a policy change that requires a new version and code review. Holdout rows
    must not be consulted during feature or model selection.
    """
    if len(dataset.rows) <= FINAL_HOLDOUT_CYCLE_COUNT:
        message = (
            "final temporal holdout requires at least "
            f"{FINAL_HOLDOUT_CYCLE_COUNT + 1} completed cycles"
        )
        raise ValueError(message)

    split_position = len(dataset.rows) - FINAL_HOLDOUT_CYCLE_COUNT
    return TemporalHoldoutSplit(
        dataset_fingerprint=dataset.fingerprint,
        policy_version=FINAL_HOLDOUT_POLICY_VERSION,
        development_rows=dataset.rows[:split_position],
        holdout_rows=dataset.rows[split_position:],
    )
