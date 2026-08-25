"""Make a local forecast from validated history and a packaged model."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from cycle_forecast.data import build_cycle_dataset, load_cycle_history
from cycle_forecast.forecasting import WalkForwardContext
from cycle_forecast.training import load_model_package, predict_with_model_package

NON_MEDICAL_DISCLAIMER = (
    "For personal planning only; not for diagnosis or medical decision-making."
)
"""Required warning attached to local prediction output."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalPrediction:
    """Contain one user-facing forecast and its model provenance.

    Parameters
    ----------
    current_cycle_start_date
        Most recent period start and prediction cutoff.
    predicted_cycle_length_days
        Raw numeric model prediction.
    operational_cycle_length_days
        Positive whole-day prediction used for date arithmetic.
    predicted_next_cycle_start_date
        Forecast next period start.
    model_version
        Semantic version of packaged model behavior.
    code_version
        Code revision recorded when the model was packaged.
    disclaimer
        Non-medical-use warning displayed with the result.
    """

    current_cycle_start_date: date
    predicted_cycle_length_days: float
    operational_cycle_length_days: int
    predicted_next_cycle_start_date: date
    model_version: str
    code_version: str
    disclaimer: str


def predict_from_local_files(
    *, model_path: str | Path, history_path: str | Path
) -> LocalPrediction:
    """Predict the current cycle from local package and CSV files.

    Parameters
    ----------
    model_path
        Versioned JSON model package produced by the training workflow.
    history_path
        Validated cycle-history CSV whose newest row is the current period start.

    Returns
    -------
    LocalPrediction
        Forecast dates, lengths, provenance, and non-medical disclaimer.

    Raises
    ------
    OSError
        If the model package cannot be read.
    ValueError
        If either file is invalid or history is insufficient for the model.

    Notes
    -----
    The newest raw record supplies only the current prediction cutoff. Completed
    cycle lengths derived from preceding record pairs form the model context.
    No data leaves the local process.
    """
    package = load_model_package(path=model_path)
    records = load_cycle_history(path=history_path)
    if not records:
        message = "cycle history must contain a current period start"
        raise ValueError(message)
    dataset = build_cycle_dataset(records=records)
    forecast = predict_with_model_package(
        package=package,
        context=WalkForwardContext(
            cycle_start_date=records[-1].cycle_start_date,
            history=dataset.rows,
        ),
    )
    return LocalPrediction(
        current_cycle_start_date=forecast.cycle_start_date,
        predicted_cycle_length_days=forecast.predicted_cycle_length_days,
        operational_cycle_length_days=forecast.operational_cycle_length_days,
        predicted_next_cycle_start_date=forecast.predicted_next_cycle_start_date,
        model_version=package.model_version,
        code_version=package.code_version,
        disclaimer=NON_MEDICAL_DISCLAIMER,
    )
