"""Render development prediction-interval coverage and width together."""

# Matplotlib lacks complete type information for several object-oriented methods.
# pyright: reportUnknownMemberType=false

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from cycle_forecast.evaluation import DevelopmentUncertaintyEvaluation

_FORECASTER_COLORS = ("#2E7D32", "#1565C0")


def render_development_uncertainty_markdown(
    *, evaluation: DevelopmentUncertaintyEvaluation
) -> str:
    """Render interval coverage and width in one Markdown table.

    Parameters
    ----------
    evaluation
        Development-only sequential interval evaluation.

    Returns
    -------
    str
        Markdown text without file-system side effects.
    """
    lines = [
        "# Development prediction uncertainty",
        "",
        "> **Development only:** every interval used earlier forecast errors only; ",
        "> the final temporal holdout remains sealed.",
        "",
        f"- Evaluated interval cycles: {len(evaluation.cycle_start_dates)}",
        f"- Calibration warmup: {evaluation.configuration.minimum_calibration_rows} forecasts",
        f"- Date range: {evaluation.cycle_start_dates[0].isoformat()} to "
        f"{evaluation.cycle_start_dates[-1].isoformat()}",
        "",
        "| Method | Target coverage | Actual coverage | Coverage gap | Mean width | Median width | Intervals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for forecaster in evaluation.forecasters:
        for interval_evaluation in forecaster.interval_evaluations:
            metrics = interval_evaluation.metrics
            gap = metrics.empirical_coverage - metrics.nominal_coverage
            lines.append(
                f"| {forecaster.forecaster_label} | "
                f"{metrics.nominal_coverage:.0%} | "
                f"{metrics.empirical_coverage:.1%} | "
                f"{gap:+.1%} | "
                f"{metrics.mean_width_days:.2f} days | "
                f"{metrics.median_width_days:.2f} days | "
                f"{metrics.interval_count} |"
            )
    return "\n".join(lines) + "\n"


def plot_development_uncertainty(
    *, evaluation: DevelopmentUncertaintyEvaluation
) -> Figure:
    """Plot empirical coverage beside the width needed to obtain it.

    Parameters
    ----------
    evaluation
        Development-only sequential interval evaluation.

    Returns
    -------
    matplotlib.figure.Figure
        Two-panel reliability and usefulness figure controlled by the caller.
    """
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    targets = evaluation.configuration.coverage_levels
    axes[0].plot(
        targets,
        targets,
        color="#616161",
        linestyle="--",
        label="Perfect calibration",
    )
    for forecaster, color in zip(
        evaluation.forecasters, _FORECASTER_COLORS, strict=True
    ):
        empirical_coverages = tuple(
            item.metrics.empirical_coverage for item in forecaster.interval_evaluations
        )
        mean_widths = tuple(
            item.metrics.mean_width_days for item in forecaster.interval_evaluations
        )
        axes[0].plot(
            targets,
            empirical_coverages,
            color=color,
            marker="o",
            label=forecaster.forecaster_label,
        )
        axes[1].plot(
            targets,
            mean_widths,
            color=color,
            marker="o",
            label=forecaster.forecaster_label,
        )
    axes[0].set_title("Coverage reliability")
    axes[0].set_xlabel("Target coverage")
    axes[0].set_ylabel("Actual coverage")
    axes[0].set_xlim(0.4, 1.0)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend()
    axes[1].set_title("Prediction-window width")
    axes[1].set_xlabel("Target coverage")
    axes[1].set_ylabel("Mean width (days)")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure
