"""Tests for cycle-history exploratory statistics."""

from datetime import date, timedelta
from inspect import Parameter, signature

import pytest
from matplotlib import pyplot as plt

from cycle_forecast.analysis.cycle_history import (
    CycleHistoryExploration,
    CycleLengthFrequency,
    explore_cycle_history,
)
from cycle_forecast.analysis.plotting import plot_cycle_history
from cycle_forecast.data import CycleDataset, CycleDatasetRow


def _dataset(*, cycle_lengths: tuple[int, ...]) -> CycleDataset:
    """Build a minimal immutable dataset with specified cycle lengths.

    Parameters
    ----------
    cycle_lengths
        Synthetic lengths to place in chronological rows.

    Returns
    -------
    CycleDataset
        Synthetic dataset for analysis tests.
    """
    start = date(2024, 1, 1)
    rows: list[CycleDatasetRow] = []
    for cycle_length in cycle_lengths:
        next_start = start + timedelta(days=cycle_length)
        rows.append(
            CycleDatasetRow(
                cycle_start_date=start,
                next_cycle_start_date=next_start,
                cycle_length_days=cycle_length,
            )
        )
        start = next_start
    return CycleDataset(
        rows=tuple(rows),
        transformation_version="cycle-dataset-v1",
        fingerprint=f"sha256:synthetic-{','.join(map(str, cycle_lengths))}",
    )


def test_explore_cycle_history_summarizes_core_dimensions() -> None:
    """Summarize distribution, variability, trend, and autocorrelation."""
    exploration = explore_cycle_history(
        dataset=_dataset(cycle_lengths=(24, 26, 28, 30, 32))
    )

    assert exploration.dataset_fingerprint == "sha256:synthetic-24,26,28,30,32"
    assert exploration.cycle_count == 5
    assert exploration.cycle_length_frequencies == (
        CycleLengthFrequency(cycle_length_days=24, count=1),
        CycleLengthFrequency(cycle_length_days=26, count=1),
        CycleLengthFrequency(cycle_length_days=28, count=1),
        CycleLengthFrequency(cycle_length_days=30, count=1),
        CycleLengthFrequency(cycle_length_days=32, count=1),
    )
    assert exploration.mean_days == 28
    assert exploration.median_days == 28
    assert exploration.minimum_days == 24
    assert exploration.maximum_days == 32
    assert exploration.standard_deviation_days == pytest.approx(3.1622776602)
    assert exploration.interquartile_range_days == 4
    assert exploration.trend_days_per_cycle == 2
    assert exploration.lag_one_autocorrelation == 1


def test_public_analysis_api_requires_keyword_arguments() -> None:
    """Keep reusable analysis functions and summaries explicit at call sites."""
    assert signature(explore_cycle_history).parameters["dataset"].kind is (
        Parameter.KEYWORD_ONLY
    )
    assert signature(plot_cycle_history).parameters["dataset"].kind is (
        Parameter.KEYWORD_ONLY
    )
    assert signature(CycleLengthFrequency).parameters["cycle_length_days"].kind is (
        Parameter.KEYWORD_ONLY
    )
    assert signature(CycleHistoryExploration).parameters[
        "dataset_fingerprint"
    ].kind is (Parameter.KEYWORD_ONLY)


def test_explore_empty_history_reports_undefined_statistics() -> None:
    """Represent statistics honestly when no completed cycles exist."""
    exploration = explore_cycle_history(dataset=_dataset(cycle_lengths=()))

    assert exploration == CycleHistoryExploration(
        dataset_fingerprint="sha256:synthetic-",
        cycle_count=0,
        cycle_length_frequencies=(),
        mean_days=None,
        median_days=None,
        minimum_days=None,
        maximum_days=None,
        standard_deviation_days=None,
        interquartile_range_days=None,
        trend_days_per_cycle=None,
        lag_one_autocorrelation=None,
    )


def test_explore_single_cycle_limits_unsupported_statistics() -> None:
    """Return location and range without inventing variability or trend."""
    exploration = explore_cycle_history(dataset=_dataset(cycle_lengths=(28,)))

    assert exploration.mean_days == 28
    assert exploration.median_days == 28
    assert exploration.minimum_days == 28
    assert exploration.maximum_days == 28
    assert exploration.standard_deviation_days is None
    assert exploration.interquartile_range_days is None
    assert exploration.trend_days_per_cycle is None
    assert exploration.lag_one_autocorrelation is None


def test_explore_constant_history_has_undefined_autocorrelation() -> None:
    """Avoid reporting a correlation when adjacent values have no variance."""
    exploration = explore_cycle_history(dataset=_dataset(cycle_lengths=(28, 28, 28)))

    assert exploration.standard_deviation_days == 0
    assert exploration.trend_days_per_cycle == 0
    assert exploration.lag_one_autocorrelation is None


def test_explore_two_cycles_has_no_lag_one_autocorrelation() -> None:
    """Require at least two adjacent pairs for a correlation estimate."""
    exploration = explore_cycle_history(dataset=_dataset(cycle_lengths=(27, 29)))

    assert exploration.lag_one_autocorrelation is None


def test_plot_cycle_history_builds_three_exploration_panels() -> None:
    """Render distribution, chronological, and lag views from shared rows."""
    dataset = _dataset(cycle_lengths=(27, 29, 28, 30))

    figure = plot_cycle_history(dataset=dataset)

    assert [axis.get_title() for axis in figure.axes] == [
        "Cycle-length distribution",
        "Cycle length over time",
        "Lag-1 relationship",
    ]
    history_labels = [line.get_label() for line in figure.axes[1].lines]
    assert "Linear trend" in history_labels
    plt.close(figure)


def test_date_ticks_are_sparse_and_include_history_boundaries() -> None:
    """Keep long histories readable while labeling both temporal endpoints."""
    dataset = _dataset(cycle_lengths=(28,) * 121)

    figure = plot_cycle_history(dataset=dataset)

    date_labels = [label.get_text() for label in figure.axes[1].get_xticklabels()]
    assert len(date_labels) == 8
    assert date_labels[0] == dataset.rows[0].cycle_start_date.isoformat()
    assert date_labels[-1] == dataset.rows[-1].cycle_start_date.isoformat()
    plt.close(figure)


def test_plot_cycle_history_adds_trailing_moving_average() -> None:
    """Smooth chronology using completed cycles at or before each position."""
    dataset = _dataset(cycle_lengths=(24, 26, 28, 30, 32, 34, 36))

    figure = plot_cycle_history(dataset=dataset, moving_average_window=3)

    history_labels = [line.get_label() for line in figure.axes[1].lines]
    assert "3-cycle trailing mean" in history_labels
    plt.close(figure)


def test_reject_nonpositive_moving_average_window() -> None:
    """Reject a moving-average definition with no completed cycles."""
    with pytest.raises(ValueError, match="moving_average_window"):
        plot_cycle_history(
            dataset=_dataset(cycle_lengths=(28,)),
            moving_average_window=0,
        )


def test_plot_cycle_history_rejects_mismatched_exploration() -> None:
    """Prevent rows from being plotted with another dataset's statistics."""
    dataset = _dataset(cycle_lengths=(27, 29, 28))
    other_exploration = explore_cycle_history(
        dataset=_dataset(cycle_lengths=(30, 31, 32))
    )

    with pytest.raises(ValueError, match="fingerprint"):
        plot_cycle_history(dataset=dataset, exploration=other_exploration)


def test_plot_empty_cycle_history() -> None:
    """Return labeled empty panels when no completed cycle is available."""
    figure = plot_cycle_history(dataset=_dataset(cycle_lengths=()))

    assert len(figure.axes) == 3
    plt.close(figure)
