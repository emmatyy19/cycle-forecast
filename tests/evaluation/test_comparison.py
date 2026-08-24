"""Test development-only hyperparameter and baseline comparison."""

from dataclasses import replace
from datetime import date, timedelta

import pytest

from cycle_forecast.data import CycleDataset, CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.evaluation import (
    DevelopmentComparisonConfig,
    ForecasterKind,
    compare_development_forecasters,
)
from cycle_forecast.features import build_development_history_features
from cycle_forecast.forecasting import split_final_temporal_holdout


def _dataset(*, cycle_count: int = 72) -> CycleDataset:
    """Build a deterministic nonlinear synthetic cycle history."""
    lengths = tuple(
        27 + position % 5 + (position // 9) % 2 for position in range(cycle_count)
    )
    starts = [date(2015, 1, 1)]
    for length in lengths:
        starts.append(starts[-1] + timedelta(days=length))
    return build_cycle_dataset(
        records=tuple(
            CycleHistoryRecord(cycle_start_date=start, period_length_days=5)
            for start in starts
        )
    )


def test_comparison_tunes_ridge_and_ranks_baselines_on_same_development_dates() -> None:
    """Select only from predeclared candidates on one pre-holdout window."""
    dataset = _dataset()
    split = split_final_temporal_holdout(dataset=dataset)
    features = build_development_history_features(split=split)

    comparison = compare_development_forecasters(
        dataset=dataset,
        split=split,
        features=features,
    )

    assert len(comparison.entries) == 13
    assert comparison.cycle_start_dates
    assert comparison.cycle_start_dates[-1] < split.holdout_rows[0].cycle_start_date
    assert all(
        entry.evaluation.metrics.forecast_count == len(comparison.cycle_start_dates)
        and tuple(error.cycle_start_date for error in entry.evaluation.errors)
        == comparison.cycle_start_dates
        for entry in comparison.entries
    )
    ridge_entries = tuple(
        entry for entry in comparison.entries if entry.kind is ForecasterKind.RIDGE
    )
    baseline_entries = tuple(
        entry for entry in comparison.entries if entry.kind is ForecasterKind.BASELINE
    )
    assert tuple(entry.ridge_alpha for entry in ridge_entries) == (
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    )
    assert comparison.selected_ridge in ridge_entries
    assert comparison.strongest_baseline in baseline_entries
    ridge_maes = tuple(
        entry.evaluation.metrics.mean_absolute_error_days for entry in ridge_entries
    )
    baseline_maes = tuple(
        entry.evaluation.metrics.mean_absolute_error_days for entry in baseline_entries
    )
    assert all(value is not None for value in ridge_maes)
    assert all(value is not None for value in baseline_maes)
    assert comparison.selected_ridge.evaluation.metrics.mean_absolute_error_days == min(
        value for value in ridge_maes if value is not None
    )
    assert (
        comparison.strongest_baseline.evaluation.metrics.mean_absolute_error_days
        == min(value for value in baseline_maes if value is not None)
    )


@pytest.mark.parametrize(
    "ridge_alphas",
    [(), (0.0,), (1.0, 0.1), (1.0, 1.0), (float("nan"),)],
)
def test_comparison_configuration_rejects_invalid_ridge_candidates(
    ridge_alphas: tuple[float, ...],
) -> None:
    """Reject an ambiguous or unusable fixed tuning grid."""
    with pytest.raises(ValueError, match="ridge_alphas must contain"):
        DevelopmentComparisonConfig(
            ridge_alphas=ridge_alphas,
            rolling_windows=(3,),
            minimum_ridge_training_rows=12,
        )


def test_comparison_rejects_provenance_mismatch() -> None:
    """Prevent selection across unrelated datasets."""
    dataset = _dataset()
    split = split_final_temporal_holdout(dataset=dataset)
    features = build_development_history_features(split=split)

    with pytest.raises(ValueError, match="share a fingerprint"):
        compare_development_forecasters(
            dataset=dataset,
            split=split,
            features=replace(features, dataset_fingerprint="sha256:other"),
        )


def test_comparison_rejects_an_empty_shared_development_window() -> None:
    """Explain when configured training history leaves nothing to compare."""
    dataset = _dataset(cycle_count=30)
    split = split_final_temporal_holdout(dataset=dataset)
    features = build_development_history_features(split=split)
    configuration = DevelopmentComparisonConfig(
        ridge_alphas=(1.0,),
        rolling_windows=(3,),
        minimum_ridge_training_rows=100,
    )

    with pytest.raises(ValueError, match="nonempty shared forecast window"):
        compare_development_forecasters(
            dataset=dataset,
            split=split,
            features=features,
            configuration=configuration,
        )
