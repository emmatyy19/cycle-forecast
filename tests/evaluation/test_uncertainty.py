"""Test leakage-safe sequential prediction-interval evaluation."""

from dataclasses import replace
from datetime import date, timedelta
from math import ceil

import pytest

from cycle_forecast.data import CycleDataset, CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.evaluation import (
    DevelopmentModelComparison,
    PredictionIntervalConfig,
    compare_development_forecasters,
    evaluate_development_uncertainty,
)
from cycle_forecast.features import build_development_history_features
from cycle_forecast.forecasting import split_final_temporal_holdout


def _comparison() -> DevelopmentModelComparison:
    """Build a deterministic real development comparison for interval tests."""
    lengths = tuple(27 + position % 5 + (position // 8) % 2 for position in range(72))
    starts = [date(2015, 1, 1)]
    for length in lengths:
        starts.append(starts[-1] + timedelta(days=length))
    dataset: CycleDataset = build_cycle_dataset(
        records=tuple(
            CycleHistoryRecord(cycle_start_date=start, period_length_days=5)
            for start in starts
        )
    )
    split = split_final_temporal_holdout(dataset=dataset)
    features = build_development_history_features(split=split)
    return compare_development_forecasters(
        dataset=dataset,
        split=split,
        features=features,
    )


def test_uncertainty_uses_only_earlier_errors_and_reports_width_with_coverage() -> None:
    """Construct finite-sample intervals and summarize both quality dimensions."""
    comparison = _comparison()

    evaluation = evaluate_development_uncertainty(comparison=comparison)

    assert evaluation.cycle_start_dates == comparison.cycle_start_dates[12:]
    assert len(evaluation.forecasters) == 2
    for forecaster in evaluation.forecasters:
        assert tuple(
            item.metrics.nominal_coverage for item in forecaster.interval_evaluations
        ) == (0.5, 0.8, 0.9)
        assert all(
            item.metrics.interval_count == len(evaluation.cycle_start_dates)
            for item in forecaster.interval_evaluations
        )
        assert tuple(
            item.metrics.mean_width_days for item in forecaster.interval_evaluations
        ) == tuple(
            sorted(
                item.metrics.mean_width_days for item in forecaster.interval_evaluations
            )
        )

    selected_errors = comparison.selected_ridge.evaluation.errors
    first_interval = evaluation.forecasters[0].interval_evaluations[1].intervals[0]
    ordered_prior_errors = sorted(
        error.absolute_error_days for error in selected_errors[:12]
    )
    expected_rank = ceil(13 * 0.8)
    assert first_interval.radius_days == ordered_prior_errors[expected_rank - 1]
    assert first_interval.calibration_error_count == 12


def test_current_actual_cannot_change_its_already_formed_interval() -> None:
    """Keep current bounds fixed when only the unseen current outcome changes."""
    comparison = _comparison()
    original = evaluate_development_uncertainty(comparison=comparison)
    errors = comparison.selected_ridge.evaluation.errors
    changed_error = replace(
        errors[12],
        actual_cycle_length_days=100,
        error_days=errors[12].predicted_cycle_length_days - 100,
        absolute_error_days=abs(errors[12].predicted_cycle_length_days - 100),
    )
    changed_selected = replace(
        comparison.selected_ridge,
        evaluation=replace(
            comparison.selected_ridge.evaluation,
            errors=(*errors[:12], changed_error, *errors[13:]),
        ),
    )
    changed_comparison = replace(
        comparison,
        selected_ridge=changed_selected,
    )

    changed = evaluate_development_uncertainty(comparison=changed_comparison)

    original_interval = original.forecasters[0].interval_evaluations[1].intervals[0]
    changed_interval = changed.forecasters[0].interval_evaluations[1].intervals[0]
    assert changed_interval.lower_cycle_length_days == (
        original_interval.lower_cycle_length_days
    )
    assert changed_interval.upper_cycle_length_days == (
        original_interval.upper_cycle_length_days
    )
    assert changed_interval.contains_actual is False


@pytest.mark.parametrize(
    ("coverage_levels", "minimum_rows", "message"),
    [
        ((), 12, "coverage_levels must contain"),
        ((0.0,), 12, "coverage_levels must contain"),
        ((1.0,), 12, "coverage_levels must contain"),
        ((0.8, 0.5), 12, "coverage_levels must contain"),
        ((0.8, 0.8), 12, "coverage_levels must contain"),
        ((0.8,), 0, "minimum_calibration_rows must be positive"),
        ((0.99,), 12, "too small for the largest"),
    ],
)
def test_interval_configuration_rejects_invalid_values(
    coverage_levels: tuple[float, ...], minimum_rows: int, message: str
) -> None:
    """Reject ambiguous or finite-sample-impossible interval settings."""
    with pytest.raises(ValueError, match=message):
        PredictionIntervalConfig(
            coverage_levels=coverage_levels,
            minimum_calibration_rows=minimum_rows,
        )


def test_uncertainty_rejects_insufficient_post_warmup_history() -> None:
    """Require at least one honest interval after calibration."""
    comparison = _comparison()
    configuration = PredictionIntervalConfig(
        coverage_levels=(0.5,),
        minimum_calibration_rows=len(comparison.cycle_start_dates),
    )

    with pytest.raises(ValueError, match="require errors after calibration"):
        evaluate_development_uncertainty(
            comparison=comparison,
            configuration=configuration,
        )
