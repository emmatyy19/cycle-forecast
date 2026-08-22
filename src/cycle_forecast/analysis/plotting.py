"""Render reusable cycle-history exploration figures."""

# Matplotlib's public plotting methods accept extensible keyword arguments that
# its type information exposes as Unknown. Keep that limitation isolated here.
# pyright: reportUnknownMemberType=false

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cycle_forecast.analysis.cycle_history import (
    CycleHistoryExploration,
    explore_cycle_history,
)
from cycle_forecast.data import CycleDataset


def _date_tick_positions(
    *, cycle_count: int, maximum_ticks: int = 8
) -> tuple[int, ...]:
    """Select evenly spaced chronological labels without crowding the axis.

    Parameters
    ----------
    cycle_count
        Number of chronological observations.
    maximum_ticks
        Maximum number of date labels to display.

    Returns
    -------
    tuple[int, ...]
        Zero-based positions including the first and last observations when
        any observations exist.

    Raises
    ------
    ValueError
        If ``cycle_count`` is negative or ``maximum_ticks`` is less than two.
    """
    if cycle_count < 0:
        message = "cycle_count must not be negative"
        raise ValueError(message)
    if maximum_ticks < 2:
        message = "maximum_ticks must be at least two"
        raise ValueError(message)
    if cycle_count <= maximum_ticks:
        return tuple(range(cycle_count))

    last_position = cycle_count - 1
    return tuple(
        round(tick_number * last_position / (maximum_ticks - 1))
        for tick_number in range(maximum_ticks)
    )


def plot_cycle_history(
    *,
    dataset: CycleDataset,
    exploration: CycleHistoryExploration | None = None,
    moving_average_window: int | None = 6,
) -> Figure:
    """Plot distribution, chronological history, and lag-1 relationships.

    Parameters
    ----------
    dataset
        Immutable completed-cycle rows to visualize.
    exploration
        Optional statistics previously calculated from the same dataset. The
        function calculates them when omitted.
    moving_average_window
        Number of completed cycles in the trailing moving average. Use ``None``
        to omit the line. A value of 6 is roughly half a year for visualization,
        not a selected forecasting hyperparameter.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing three reusable exploration panels.

    Raises
    ------
    ValueError
        If ``exploration`` belongs to a different dataset fingerprint or the
        moving-average window is not positive.

    Notes
    -----
    The function neither displays nor saves the figure. Notebook and application
    callers retain control over those side effects.
    """
    if moving_average_window is not None and moving_average_window < 1:
        message = "moving_average_window must be positive or None"
        raise ValueError(message)

    summary = exploration or explore_cycle_history(dataset=dataset)
    if summary.dataset_fingerprint != dataset.fingerprint:
        message = "exploration fingerprint does not match the dataset"
        raise ValueError(message)

    figure = plt.figure(figsize=(15, 4), layout="constrained")
    distribution_axis = figure.add_subplot(1, 3, 1)
    history_axis = figure.add_subplot(1, 3, 2)
    lag_axis = figure.add_subplot(1, 3, 3)

    lengths = tuple(row.cycle_length_days for row in dataset.rows)
    dates = tuple(row.cycle_start_date for row in dataset.rows)
    positions = tuple(range(len(lengths)))

    distribution_axis.bar(
        [item.cycle_length_days for item in summary.cycle_length_frequencies],
        [item.count for item in summary.cycle_length_frequencies],
        width=0.8,
    )
    distribution_axis.set(
        title="Cycle-length distribution",
        xlabel="Cycle length (days)",
        ylabel="Completed cycles",
    )

    history_axis.plot(positions, lengths, "-o", label="Observed")
    if summary.median_days is not None:
        history_axis.axhline(
            summary.median_days,
            linestyle="--",
            color="tab:gray",
            label="Median",
        )
    if summary.trend_days_per_cycle is not None and summary.mean_days is not None:
        position_mean = (len(lengths) - 1) / 2
        fitted_values = tuple(
            summary.mean_days
            + summary.trend_days_per_cycle * (position - position_mean)
            for position in range(len(lengths))
        )
        history_axis.plot(
            positions,
            fitted_values,
            "-",
            label="Linear trend",
        )
    if moving_average_window is not None and len(lengths) >= moving_average_window:
        moving_average_positions = positions[moving_average_window - 1 :]
        moving_averages = tuple(
            sum(lengths[end_position - moving_average_window + 1 : end_position + 1])
            / moving_average_window
            for end_position in moving_average_positions
        )
        history_axis.plot(
            moving_average_positions,
            moving_averages,
            linewidth=2,
            label=f"{moving_average_window}-cycle trailing mean",
        )
    history_axis.set(
        title="Cycle length over time",
        xlabel="Cycle start",
        ylabel="Cycle length (days)",
    )
    if lengths:
        history_axis.legend()
        tick_positions = _date_tick_positions(cycle_count=len(dates))
        history_axis.set_xticks(
            tick_positions,
            [dates[position].isoformat() for position in tick_positions],
            rotation=45,
            horizontalalignment="right",
        )

    current_lengths = lengths[:-1]
    following_lengths = lengths[1:]
    lag_axis.scatter(current_lengths, following_lengths)
    if current_lengths:
        all_paired_lengths = current_lengths + following_lengths
        reference_minimum = min(all_paired_lengths)
        reference_maximum = max(all_paired_lengths)
        lag_axis.plot(
            (reference_minimum, reference_maximum),
            (reference_minimum, reference_maximum),
            linestyle="--",
            color="tab:gray",
            label="Equal length",
        )
        lag_axis.legend()
    lag_axis.set(
        title="Lag-1 relationship",
        xlabel="Current cycle length (days)",
        ylabel="Following cycle length (days)",
    )

    return figure
