"""Render development-only comparison tables and diagnostic plots."""

# Matplotlib lacks complete type information for several object-oriented methods.
# pyright: reportUnknownMemberType=false

from collections.abc import Callable

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from cycle_forecast.evaluation import (
    DevelopmentModelComparison,
    ForecasterComparisonEntry,
)

_SELECTED_RIDGE_COLOR = "#2E7D32"
_STRONGEST_BASELINE_COLOR = "#1565C0"
_OTHER_COLOR = "#B0BEC5"


def _require_metric(*, value: float | None, metric_name: str) -> float:
    """Return a metric known to exist for a nonempty comparison.

    Parameters
    ----------
    value
        Optional aggregate metric value.
    metric_name
        Human-readable metric identifier for failures.

    Returns
    -------
    float
        Defined metric value.

    Raises
    ------
    ValueError
        If the comparison unexpectedly contains an undefined metric.
    """
    if value is None:
        message = f"comparison metric is undefined: {metric_name}"
        raise ValueError(message)
    return value


def _color_for(
    *, entry: ForecasterComparisonEntry, comparison: DevelopmentModelComparison
) -> str:
    """Choose a stable highlight color for a comparison entry.

    Parameters
    ----------
    entry
        Entry being rendered.
    comparison
        Comparison defining the selected methods.

    Returns
    -------
    str
        Matplotlib-compatible color.
    """
    if entry == comparison.selected_ridge:
        return _SELECTED_RIDGE_COLOR
    if entry == comparison.strongest_baseline:
        return _STRONGEST_BASELINE_COLOR
    return _OTHER_COLOR


def _format_days(*, value: float | None, metric_name: str) -> str:
    """Format a defined day-valued metric to two decimals."""
    return f"{_require_metric(value=value, metric_name=metric_name):.2f}"


def _format_percent(*, value: float | None, metric_name: str) -> str:
    """Format a defined fractional metric as a percentage."""
    return f"{_require_metric(value=value, metric_name=metric_name):.1%}"


def render_development_comparison_markdown(
    *, comparison: DevelopmentModelComparison
) -> str:
    """Render a portable Markdown summary of development model selection.

    Parameters
    ----------
    comparison
        Development-only comparison result.

    Returns
    -------
    str
        Markdown report text. The caller decides whether and where to save it.

    Notes
    -----
    This function performs no file I/O so private-data artifact paths remain an
    explicit caller decision.
    """
    selected_mae = _format_days(
        value=comparison.selected_ridge.evaluation.metrics.mean_absolute_error_days,
        metric_name="selected Ridge MAE",
    )
    baseline_mae = _format_days(
        value=(
            comparison.strongest_baseline.evaluation.metrics.mean_absolute_error_days
        ),
        metric_name="strongest baseline MAE",
    )
    lines = [
        "# Development model comparison",
        "",
        "> **Development only:** the final temporal holdout was not evaluated or ",
        "> used for model or hyperparameter selection.",
        "",
        f"- Shared forecast cycles: {len(comparison.cycle_start_dates)}",
        f"- Date range: {comparison.cycle_start_dates[0].isoformat()} to "
        f"{comparison.cycle_start_dates[-1].isoformat()}",
        f"- Selected Ridge: {comparison.selected_ridge.label} (MAE {selected_mae} days)",
        f"- Strongest baseline: {comparison.strongest_baseline.label} "
        f"(MAE {baseline_mae} days)",
        f"- Holdout policy: `{comparison.holdout_policy_version}`",
        f"- Feature version: `{comparison.feature_version}`",
        "",
        "| Method | Type | MAE | Median AE | RMSE | Mean error | Within ±1 | Within ±2 | Within ±3 | Within ±5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in comparison.entries:
        metrics = entry.evaluation.metrics
        lines.append(
            f"| {entry.label} | {entry.kind.value} | "
            f"{_format_days(value=metrics.mean_absolute_error_days, metric_name='MAE')} | "
            f"{_format_days(value=metrics.median_absolute_error_days, metric_name='median AE')} | "
            f"{_format_days(value=metrics.root_mean_squared_error_days, metric_name='RMSE')} | "
            f"{_format_days(value=metrics.mean_error_days, metric_name='mean error')} | "
            f"{_format_percent(value=metrics.within_1_day, metric_name='within 1 day')} | "
            f"{_format_percent(value=metrics.within_2_days, metric_name='within 2 days')} | "
            f"{_format_percent(value=metrics.within_3_days, metric_name='within 3 days')} | "
            f"{_format_percent(value=metrics.within_5_days, metric_name='within 5 days')} |"
        )
    return "\n".join(lines) + "\n"


def plot_development_model_comparison(
    *, comparison: DevelopmentModelComparison
) -> Figure:
    """Plot ranking, useful accuracy, and chronological selected-model errors.

    Parameters
    ----------
    comparison
        Development-only comparison result.

    Returns
    -------
    matplotlib.figure.Figure
        Three-panel figure. The caller decides whether to show or save it.
    """
    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 7))
    labels = [entry.label for entry in comparison.entries]
    colors = [
        _color_for(entry=entry, comparison=comparison) for entry in comparison.entries
    ]
    metric_extractors: tuple[
        tuple[str, str, Callable[[ForecasterComparisonEntry], float | None]], ...
    ] = (
        (
            "Mean absolute error",
            "Days (lower is better)",
            lambda entry: entry.evaluation.metrics.mean_absolute_error_days,
        ),
        (
            "Forecasts within ±2 days",
            "Fraction (higher is better)",
            lambda entry: entry.evaluation.metrics.within_2_days,
        ),
    )
    for axis, (title, x_label, extractor) in zip(
        axes[:2], metric_extractors, strict=True
    ):
        values = [
            _require_metric(value=extractor(entry), metric_name=title)
            for entry in comparison.entries
        ]
        axis.barh(labels, values, color=colors)
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)

    for entry, color in (
        (comparison.selected_ridge, _SELECTED_RIDGE_COLOR),
        (comparison.strongest_baseline, _STRONGEST_BASELINE_COLOR),
    ):
        axes[2].plot(
            comparison.cycle_start_dates,
            [error.absolute_error_days for error in entry.evaluation.errors],
            label=entry.label,
            color=color,
            marker="o",
            markersize=3,
        )
    axes[2].set_title("Absolute error over time")
    axes[2].set_xlabel("Cycle start")
    axes[2].set_ylabel("Absolute error (days)")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure
