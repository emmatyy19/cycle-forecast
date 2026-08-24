"""Test development comparison text and visual reporting."""

from datetime import date, timedelta

from matplotlib import pyplot as plt

from cycle_forecast.data import CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.evaluation import (
    DevelopmentComparisonConfig,
    DevelopmentModelComparison,
    compare_development_forecasters,
)
from cycle_forecast.features import build_development_history_features
from cycle_forecast.forecasting import split_final_temporal_holdout
from cycle_forecast.reporting import (
    plot_development_model_comparison,
    render_development_comparison_markdown,
)


def _comparison() -> DevelopmentModelComparison:
    """Build a small real comparison for renderer integration tests."""
    lengths = tuple(27 + position % 4 for position in range(48))
    starts = [date(2020, 1, 1)]
    for length in lengths:
        starts.append(starts[-1] + timedelta(days=length))
    dataset = build_cycle_dataset(
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
        configuration=DevelopmentComparisonConfig(
            ridge_alphas=(0.1, 1.0),
            rolling_windows=(3,),
            minimum_ridge_training_rows=12,
        ),
    )


def test_markdown_report_explains_selection_and_holdout_boundary() -> None:
    """Make the report readable while warning that it is not a final score."""
    comparison = _comparison()

    report = render_development_comparison_markdown(comparison=comparison)

    assert "Development only" in report
    assert "final temporal holdout was not evaluated" in report
    assert comparison.selected_ridge.label in report
    assert comparison.strongest_baseline.label in report
    assert "| Method | Type | MAE |" in report


def test_plot_contains_rankings_and_selected_method_error_history() -> None:
    """Return a caller-controlled figure with three diagnostic panels."""
    comparison = _comparison()

    figure = plot_development_model_comparison(comparison=comparison)

    assert len(figure.axes) == 3
    assert tuple(axis.get_title() for axis in figure.axes) == (
        "Mean absolute error",
        "Forecasts within ±2 days",
        "Absolute error over time",
    )
    assert len(figure.axes[2].lines) == 2
    plt.close(figure)
