"""Test leakage-safe cycle-history feature construction."""

from datetime import date, timedelta

import pytest

from cycle_forecast.data import CycleDataset, CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.features import (
    CYCLE_HISTORY_FEATURE_VERSION,
    CycleHistoryFeatureConfig,
    build_development_history_features,
    build_history_feature_vector,
)
from cycle_forecast.forecasting import (
    FINAL_HOLDOUT_POLICY_VERSION,
    WalkForwardContext,
    split_final_temporal_holdout,
)


def _dataset(*, cycle_lengths: tuple[int, ...]) -> CycleDataset:
    """Build a synthetic dataset with specified cycle lengths.

    Parameters
    ----------
    cycle_lengths
        Positive cycle lengths to encode as period starts.

    Returns
    -------
    CycleDataset
        Synthetic completed-cycle dataset.
    """
    starts = [date(2020, 1, 1)]
    for cycle_length in cycle_lengths:
        starts.append(starts[-1] + timedelta(days=cycle_length))
    return build_cycle_dataset(
        records=tuple(
            CycleHistoryRecord(
                cycle_start_date=cycle_start,
                period_length_days=5,
            )
            for cycle_start in starts
        )
    )


def test_history_feature_vector_has_named_deterministic_values() -> None:
    """Calculate lags, rolling statistics, and expanding history in order."""
    dataset = _dataset(cycle_lengths=(20, 30, 40, 50))
    configuration = CycleHistoryFeatureConfig(
        lags=(1, 3),
        rolling_windows=(2, 3),
        include_expanding_mean=True,
    )
    context = WalkForwardContext(
        cycle_start_date=dataset.rows[3].cycle_start_date,
        history=dataset.rows[:3],
    )

    vector = build_history_feature_vector(
        context=context,
        configuration=configuration,
    )

    assert vector.cycle_start_date == dataset.rows[3].cycle_start_date
    assert vector.feature_names == (
        "lag_1_cycle_length_days",
        "lag_3_cycle_length_days",
        "rolling_mean_2_cycles",
        "rolling_median_2_cycles",
        "rolling_mean_3_cycles",
        "rolling_median_3_cycles",
        "expanding_mean_cycle_length_days",
    )
    assert vector.values == (40.0, 20.0, 35.0, 35.0, 30.0, 30.0, 30.0)


def test_development_features_exclude_holdout_and_current_target() -> None:
    """Build supervised rows using only preceding development history."""
    cycle_lengths = tuple(range(20, 50))
    dataset = _dataset(cycle_lengths=cycle_lengths)
    split = split_final_temporal_holdout(dataset=dataset)
    configuration = CycleHistoryFeatureConfig(
        lags=(1,),
        rolling_windows=(3,),
        include_expanding_mean=True,
    )

    features = build_development_history_features(
        split=split,
        configuration=configuration,
    )

    assert features.dataset_fingerprint == dataset.fingerprint
    assert features.holdout_policy_version == FINAL_HOLDOUT_POLICY_VERSION
    assert features.feature_version == CYCLE_HISTORY_FEATURE_VERSION
    assert len(features.rows) == len(split.development_rows) - 3
    assert all(
        row.vector.cycle_start_date < split.holdout_rows[0].cycle_start_date
        for row in features.rows
    )
    last = features.rows[-1]
    assert last.vector.values[0] == split.development_rows[-2].cycle_length_days
    assert last.target_cycle_length_days == split.development_rows[-1].cycle_length_days


def test_training_and_prediction_use_identical_feature_transform() -> None:
    """Reproduce a supervised row from the public prediction transform."""
    dataset = _dataset(cycle_lengths=(28,) * 20)
    split = split_final_temporal_holdout(dataset=dataset)
    configuration = CycleHistoryFeatureConfig(
        lags=(1, 2),
        rolling_windows=(3,),
        include_expanding_mean=False,
    )
    features = build_development_history_features(
        split=split,
        configuration=configuration,
    )
    target_position = 3

    prediction_vector = build_history_feature_vector(
        context=WalkForwardContext(
            cycle_start_date=split.development_rows[target_position].cycle_start_date,
            history=split.development_rows[:target_position],
        ),
        configuration=configuration,
    )

    assert features.rows[0].vector == prediction_vector


def test_development_features_allow_empty_post_warmup_rows() -> None:
    """Return schema and provenance when development history is too short."""
    dataset = _dataset(cycle_lengths=(28,) * 13)
    split = split_final_temporal_holdout(dataset=dataset)
    configuration = CycleHistoryFeatureConfig(
        lags=(3,),
        rolling_windows=(),
        include_expanding_mean=False,
    )

    features = build_development_history_features(
        split=split,
        configuration=configuration,
    )

    assert not features.rows
    assert features.feature_names == ("lag_3_cycle_length_days",)


def test_history_feature_vector_rejects_insufficient_history() -> None:
    """Reject a cutoff that cannot supply every configured feature."""
    dataset = _dataset(cycle_lengths=(28, 30))
    configuration = CycleHistoryFeatureConfig(
        lags=(3,),
        rolling_windows=(),
        include_expanding_mean=False,
    )

    with pytest.raises(ValueError, match="require at least 3 completed cycles"):
        build_history_feature_vector(
            context=WalkForwardContext(
                cycle_start_date=dataset.rows[-1].next_cycle_start_date,
                history=dataset.rows,
            ),
            configuration=configuration,
        )


@pytest.mark.parametrize(
    ("lags", "rolling_windows", "include_expanding_mean", "message"),
    [
        ((), (), False, "select at least one feature"),
        ((0,), (), False, "lags must contain unique positive values"),
        ((2, 1), (), False, "lags must contain unique positive values"),
        ((1, 1), (), False, "lags must contain unique positive values"),
        ((), (0,), False, "rolling_windows must contain unique positive values"),
    ],
)
def test_history_feature_configuration_rejects_invalid_values(
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    include_expanding_mean: bool,
    message: str,
) -> None:
    """Reject ambiguous or unusable feature configurations.

    Parameters
    ----------
    lags
        Candidate lag configuration.
    rolling_windows
        Candidate rolling-window configuration.
    include_expanding_mean
        Candidate expanding-mean setting.
    message
        Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=message):
        CycleHistoryFeatureConfig(
            lags=lags,
            rolling_windows=rolling_windows,
            include_expanding_mean=include_expanding_mean,
        )
