"""Test prediction-uncertainty text and visual reporting."""

from datetime import date, timedelta

from matplotlib import pyplot as plt

from cycle_forecast.data import CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.evaluation import (
    DevelopmentUncertaintyEvaluation,
    compare_development_forecasters,
    evaluate_development_uncertainty,
)
from cycle_forecast.features import build_development_history_features
from cycle_forecast.forecasting import split_final_temporal_holdout
from cycle_forecast.reporting import (
    plot_development_uncertainty,
    render_development_uncertainty_markdown,
)


def _uncertainty_evaluation() -> DevelopmentUncertaintyEvaluation:
    """Build a real uncertainty result for renderer integration tests."""
    lengths = tuple(27 + position % 5 for position in range(60))
    starts = [date(2018, 1, 1)]
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
    comparison = compare_development_forecasters(
        dataset=dataset,
        split=split,
        features=features,
    )
    return evaluate_development_uncertainty(comparison=comparison)


def test_uncertainty_report_places_coverage_beside_width() -> None:
    """Make interval reliability and practical cost readable together."""
    evaluation = _uncertainty_evaluation()

    report = render_development_uncertainty_markdown(evaluation=evaluation)

    assert "final temporal holdout remains sealed" in report
    assert "| Target coverage | Actual coverage | Coverage gap | Mean width |" in report
    assert evaluation.forecasters[0].forecaster_label in report
    assert "90%" in report


def test_uncertainty_plot_contains_reliability_and_width_panels() -> None:
    """Return a caller-controlled two-panel interval diagnostic figure."""
    evaluation = _uncertainty_evaluation()

    figure = plot_development_uncertainty(evaluation=evaluation)

    assert len(figure.axes) == 2
    assert tuple(axis.get_title() for axis in figure.axes) == (
        "Coverage reliability",
        "Prediction-window width",
    )
    assert len(figure.axes[0].lines) == 3
    assert len(figure.axes[1].lines) == 2
    plt.close(figure)
