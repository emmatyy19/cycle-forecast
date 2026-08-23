"""Test development-only walk-forward Ridge regression."""

from dataclasses import replace
from datetime import date, timedelta

import pytest

from cycle_forecast.data import CycleDataset, CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.features import (
    CYCLE_HISTORY_FEATURE_VERSION,
    CycleHistoryFeatureConfig,
    HistoryFeatureDataset,
    HistoryFeatureRow,
    HistoryFeatureVector,
    build_development_history_features,
)
from cycle_forecast.forecasting import (
    FINAL_HOLDOUT_POLICY_VERSION,
    evaluate_forecasts,
    split_final_temporal_holdout,
)
from cycle_forecast.models import (
    RIDGE_MODEL_VERSION,
    RidgeForecastConfig,
    forecast_with_walk_forward_ridge,
)


def _dataset(*, cycle_count: int) -> CycleDataset:
    """Build deterministic synthetic cycle history with mild variation.

    Parameters
    ----------
    cycle_count
        Number of completed cycles to construct.

    Returns
    -------
    CycleDataset
        Synthetic chronological cycle dataset.
    """
    cycle_lengths = tuple(28 + position % 5 for position in range(cycle_count))
    starts = [date(2015, 1, 1)]
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


def _simple_features(*, second_feature_scale: float = 1.0) -> HistoryFeatureDataset:
    """Build a small typed feature dataset for model-boundary tests.

    Parameters
    ----------
    second_feature_scale
        Multiplicative unit scale applied to the second feature.

    Returns
    -------
    HistoryFeatureDataset
        Synthetic development-only supervised feature rows.
    """
    configuration = CycleHistoryFeatureConfig(
        lags=(1, 2),
        rolling_windows=(),
        include_expanding_mean=False,
    )
    feature_names = (
        "lag_1_cycle_length_days",
        "lag_2_cycle_length_days",
    )
    rows = tuple(
        HistoryFeatureRow(
            vector=HistoryFeatureVector(
                cycle_start_date=date(2020, 1, 1) + timedelta(days=30 * position),
                feature_names=feature_names,
                values=(float(position), float(position**2) * second_feature_scale),
            ),
            target_cycle_length_days=25 + position,
        )
        for position in range(20)
    )
    return HistoryFeatureDataset(
        dataset_fingerprint="sha256:synthetic-model-features",
        holdout_policy_version=FINAL_HOLDOUT_POLICY_VERSION,
        feature_version=CYCLE_HISTORY_FEATURE_VERSION,
        configuration=configuration,
        feature_names=feature_names,
        rows=rows,
    )


def test_walk_forward_ridge_returns_development_only_forecasts() -> None:
    """Fit at each cutoff and retain complete model and data provenance."""
    dataset = _dataset(cycle_count=60)
    split = split_final_temporal_holdout(dataset=dataset)
    features = build_development_history_features(split=split)
    configuration = RidgeForecastConfig(alpha=1.0, minimum_training_rows=12)

    result = forecast_with_walk_forward_ridge(
        features=features,
        configuration=configuration,
    )

    assert result.dataset_fingerprint == dataset.fingerprint
    assert result.holdout_policy_version == FINAL_HOLDOUT_POLICY_VERSION
    assert result.feature_version == CYCLE_HISTORY_FEATURE_VERSION
    assert result.feature_configuration == features.configuration
    assert result.model_version == RIDGE_MODEL_VERSION
    assert result.model_configuration == configuration
    assert result.forecast_batch.forecaster_name == (
        "ridge-regression-alpha-1-min-training-12"
    )
    assert result.forecast_batch.forecaster_version == RIDGE_MODEL_VERSION
    assert len(result.forecast_batch.forecasts) == 24
    assert result.forecast_batch.forecasts[0].cycle_start_date == (
        features.rows[12].vector.cycle_start_date
    )
    assert result.forecast_batch.forecasts[-1].cycle_start_date < (
        split.holdout_rows[0].cycle_start_date
    )
    evaluation = evaluate_forecasts(
        dataset=dataset,
        forecast_batch=result.forecast_batch,
    )
    assert evaluation.metrics.forecast_count == 24
    assert evaluation.metrics.mean_absolute_error_days is not None


def test_walk_forward_ridge_does_not_train_on_current_target() -> None:
    """Keep a cutoff forecast fixed when only its unseen target changes."""
    features = _simple_features()
    configuration = RidgeForecastConfig(alpha=1.0, minimum_training_rows=12)
    changed_current_row = replace(
        features.rows[12],
        target_cycle_length_days=100,
    )
    changed_features = replace(
        features,
        rows=(*features.rows[:12], changed_current_row, *features.rows[13:]),
    )

    original = forecast_with_walk_forward_ridge(
        features=features,
        configuration=configuration,
    )
    changed = forecast_with_walk_forward_ridge(
        features=changed_features,
        configuration=configuration,
    )

    assert original.forecast_batch.forecasts[0] == changed.forecast_batch.forecasts[0]


def test_walk_forward_ridge_standardizes_feature_scales() -> None:
    """Keep predictions invariant when one feature changes measurement scale."""
    unscaled = _simple_features(second_feature_scale=1.0)
    rescaled = _simple_features(second_feature_scale=1_000_000.0)
    configuration = RidgeForecastConfig(alpha=1.0, minimum_training_rows=12)

    unscaled_result = forecast_with_walk_forward_ridge(
        features=unscaled,
        configuration=configuration,
    )
    rescaled_result = forecast_with_walk_forward_ridge(
        features=rescaled,
        configuration=configuration,
    )

    assert tuple(
        forecast.predicted_cycle_length_days
        for forecast in unscaled_result.forecast_batch.forecasts
    ) == pytest.approx(
        tuple(
            forecast.predicted_cycle_length_days
            for forecast in rescaled_result.forecast_batch.forecasts
        )
    )


def test_walk_forward_ridge_allows_insufficient_training_rows() -> None:
    """Return an empty forecast batch until minimum training history exists."""
    features = _simple_features()
    configuration = RidgeForecastConfig(alpha=1.0, minimum_training_rows=20)

    result = forecast_with_walk_forward_ridge(
        features=features,
        configuration=configuration,
    )

    assert not result.forecast_batch.forecasts


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan"), float("inf")])
def test_ridge_configuration_rejects_invalid_alpha(alpha: float) -> None:
    """Reject values that cannot define positive regularization.

    Parameters
    ----------
    alpha
        Invalid Ridge regularization strength.
    """
    with pytest.raises(ValueError, match="alpha must be positive and finite"):
        RidgeForecastConfig(alpha=alpha, minimum_training_rows=12)


def test_ridge_configuration_rejects_nonpositive_training_rows() -> None:
    """Reject a walk-forward configuration without prior training rows."""
    with pytest.raises(ValueError, match="minimum_training_rows must be positive"):
        RidgeForecastConfig(alpha=1.0, minimum_training_rows=0)


def test_walk_forward_ridge_rejects_feature_name_mismatch() -> None:
    """Reject a row whose positional values have a different schema."""
    features = _simple_features()
    invalid_vector = replace(
        features.rows[0].vector,
        feature_names=("unexpected_feature",),
    )
    invalid_features = replace(
        features,
        rows=(replace(features.rows[0], vector=invalid_vector), *features.rows[1:]),
    )

    with pytest.raises(ValueError, match="row names do not match"):
        forecast_with_walk_forward_ridge(features=invalid_features)


def test_walk_forward_ridge_rejects_feature_width_mismatch() -> None:
    """Reject a row whose value count differs from the shared schema."""
    features = _simple_features()
    invalid_vector = replace(features.rows[0].vector, values=(1.0,))
    invalid_features = replace(
        features,
        rows=(replace(features.rows[0], vector=invalid_vector), *features.rows[1:]),
    )

    with pytest.raises(ValueError, match="row width does not match"):
        forecast_with_walk_forward_ridge(features=invalid_features)


def test_walk_forward_ridge_rejects_nonchronological_rows() -> None:
    """Reject feature rows whose cutoffs could invalidate walk-forward order."""
    features = _simple_features()
    invalid_features = replace(
        features,
        rows=(features.rows[1], features.rows[0], *features.rows[2:]),
    )

    with pytest.raises(ValueError, match="strictly chronological and unique"):
        forecast_with_walk_forward_ridge(features=invalid_features)
