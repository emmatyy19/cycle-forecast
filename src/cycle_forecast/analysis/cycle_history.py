"""Summarize completed cycle-length history reproducibly."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, median, quantiles, stdev

from cycle_forecast.data import CycleDataset


@dataclass(frozen=True, slots=True, kw_only=True)
class CycleLengthFrequency:
    """Represent the observed frequency of one cycle length.

    Parameters
    ----------
    cycle_length_days
        Observed cycle length.
    count
        Number of completed cycles with that length.
    """

    cycle_length_days: int
    count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CycleHistoryExploration:
    """Contain descriptive statistics for completed cycle lengths.

    Parameters
    ----------
    dataset_fingerprint
        Identity of the exact validated dataset summarized.
    cycle_count
        Number of completed cycles.
    cycle_length_frequencies
        Counts for each observed length, ordered from shortest to longest.
    mean_days
        Arithmetic mean cycle length.
    median_days
        Median cycle length.
    minimum_days
        Shortest observed cycle length.
    maximum_days
        Longest observed cycle length.
    standard_deviation_days
        Sample standard deviation, or ``None`` with fewer than two cycles.
    interquartile_range_days
        Inclusive third quartile minus first quartile, or ``None`` with fewer
        than two cycles.
    trend_days_per_cycle
        Ordinary least-squares slope over chronological cycle position, or
        ``None`` with fewer than two cycles.
    lag_one_autocorrelation
        Pearson correlation between adjacent cycle lengths, or ``None`` when
        fewer than three cycles or zero variance makes it undefined.
    """

    dataset_fingerprint: str
    cycle_count: int
    cycle_length_frequencies: tuple[CycleLengthFrequency, ...]
    mean_days: float | None
    median_days: float | None
    minimum_days: int | None
    maximum_days: int | None
    standard_deviation_days: float | None
    interquartile_range_days: float | None
    trend_days_per_cycle: float | None
    lag_one_autocorrelation: float | None


def _linear_trend(*, values: Sequence[int]) -> float | None:
    """Calculate an ordinary least-squares slope over sequence position.

    Parameters
    ----------
    values
        Chronologically ordered observations.

    Returns
    -------
    float | None
        Change in days per cycle, or ``None`` with fewer than two values.
    """
    if len(values) < 2:
        return None

    position_mean = (len(values) - 1) / 2
    value_mean = mean(values)
    numerator = sum(
        (position - position_mean) * (value - value_mean)
        for position, value in enumerate(values)
    )
    denominator = sum(
        (position - position_mean) ** 2 for position in range(len(values))
    )
    return numerator / denominator


def _correlation(
    *,
    left: Sequence[int],
    right: Sequence[int],
) -> float | None:
    """Calculate Pearson correlation for two equal-length sequences.

    Parameters
    ----------
    left
        First sequence.
    right
        Second sequence of the same length.

    Returns
    -------
    float | None
        Pearson correlation, or ``None`` when fewer than two pairs or zero
        variance makes correlation undefined.

    Raises
    ------
    ValueError
        If the sequences have different lengths.
    """
    if len(left) != len(right):
        message = "correlation sequences must have equal lengths"
        raise ValueError(message)
    if len(left) < 2:
        return None

    left_mean = mean(left)
    right_mean = mean(right)
    left_deviations = tuple(value - left_mean for value in left)
    right_deviations = tuple(value - right_mean for value in right)
    left_sum_squares = sum(value**2 for value in left_deviations)
    right_sum_squares = sum(value**2 for value in right_deviations)
    if left_sum_squares == 0 or right_sum_squares == 0:
        return None

    cross_product = sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_deviations,
            right_deviations,
            strict=True,
        )
    )
    return cross_product / (left_sum_squares * right_sum_squares) ** 0.5


def explore_cycle_history(*, dataset: CycleDataset) -> CycleHistoryExploration:
    """Summarize distribution, variability, trend, and autocorrelation.

    Parameters
    ----------
    dataset
        Immutable completed-cycle dataset to summarize.

    Returns
    -------
    CycleHistoryExploration
        Descriptive statistics tied to the input dataset fingerprint.

    Notes
    -----
    Standard deviation uses the sample definition. Quartiles use the inclusive
    method so their endpoints represent the observed finite sample. Trend is an
    unadjusted descriptive slope, not evidence of a causal or durable change.
    """
    cycle_lengths = tuple(row.cycle_length_days for row in dataset.rows)
    if not cycle_lengths:
        return CycleHistoryExploration(
            dataset_fingerprint=dataset.fingerprint,
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

    standard_deviation = stdev(cycle_lengths) if len(cycle_lengths) >= 2 else None
    if len(cycle_lengths) >= 2:
        first_quartile, _, third_quartile = quantiles(
            cycle_lengths,
            n=4,
            method="inclusive",
        )
        interquartile_range = third_quartile - first_quartile
    else:
        interquartile_range = None

    frequencies = Counter(cycle_lengths)
    return CycleHistoryExploration(
        dataset_fingerprint=dataset.fingerprint,
        cycle_count=len(cycle_lengths),
        cycle_length_frequencies=tuple(
            CycleLengthFrequency(cycle_length_days=length, count=frequencies[length])
            for length in sorted(frequencies)
        ),
        mean_days=mean(cycle_lengths),
        median_days=median(cycle_lengths),
        minimum_days=min(cycle_lengths),
        maximum_days=max(cycle_lengths),
        standard_deviation_days=standard_deviation,
        interquartile_range_days=interquartile_range,
        trend_days_per_cycle=_linear_trend(values=cycle_lengths),
        lag_one_autocorrelation=_correlation(
            left=cycle_lengths[:-1],
            right=cycle_lengths[1:],
        ),
    )
