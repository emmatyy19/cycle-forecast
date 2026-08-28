"""Run the reproducible Phase A training workflow from local history."""

from dataclasses import dataclass
from pathlib import Path

from cycle_forecast.data import build_cycle_dataset, load_cycle_history
from cycle_forecast.evaluation import compare_development_forecasters
from cycle_forecast.features import (
    build_development_history_features,
    build_operational_history_features,
)
from cycle_forecast.forecasting import split_final_temporal_holdout
from cycle_forecast.training.delivery import (
    fit_selected_model_package,
    load_training_config,
    record_training_run,
    save_model_package,
    save_training_run,
)

DEFAULT_MODEL_FILENAME = "selected-model.json"
"""Stable filename discovered by the local prediction picker."""

DEFAULT_RUN_FILENAME = "training-run.json"
"""Stable filename for the latest reproducible training manifest."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalTrainingResult:
    """Summarize artifacts and selection metrics from one local training run."""

    model_path: Path
    run_path: Path
    dataset_fingerprint: str
    selected_ridge_alpha: float
    development_mean_absolute_error_days: float
    development_forecast_count: int


def train_from_local_history(
    *,
    history_path: str | Path,
    configuration_path: str | Path,
    output_directory: str | Path,
    code_version: str,
    replace: bool = False,
    refit_on_all_completed_cycles: bool = False,
) -> LocalTrainingResult:
    """Select, fit, and save a model using validated private history.

    Parameters
    ----------
    history_path
        Private cycle-history CSV.
    configuration_path
        Versioned Phase A TOML configuration.
    output_directory
        Local ignored directory for the package and run manifest.
    code_version
        Non-empty code or package revision recorded in both artifacts.
    replace
        Whether existing default artifacts may be replaced.
    refit_on_all_completed_cycles
        Whether to refit the selected configuration on all completed cycles
        after leakage-safe model selection and evaluation.

    Returns
    -------
    LocalTrainingResult
        Artifact paths and selected model's development performance.

    Raises
    ------
    FileExistsError
        If an output exists and ``replace`` is false.
    OSError
        If an input cannot be read or an artifact cannot be written.
    ValueError
        If data, configuration, or training invariants are invalid.

    Notes
    -----
    Model selection and metrics use development rows only. The final temporal
    holdout remains untouched during evaluation. Operational callers may then
    refit the selected configuration on all completed cycles without changing
    those development metrics.
    """
    destination = Path(output_directory)
    model_path = destination / DEFAULT_MODEL_FILENAME
    run_path = destination / DEFAULT_RUN_FILENAME
    existing = tuple(path for path in (model_path, run_path) if path.exists())
    if existing and not replace:
        names = ", ".join(str(path) for path in existing)
        message = f"refusing to replace existing artifact(s): {names}"
        raise FileExistsError(message)

    configuration = load_training_config(path=configuration_path)
    records = load_cycle_history(path=history_path)
    dataset = build_cycle_dataset(records=records)
    split = split_final_temporal_holdout(dataset=dataset)
    features = build_development_history_features(
        split=split,
        configuration=configuration.features,
    )
    comparison = compare_development_forecasters(
        dataset=dataset,
        split=split,
        features=features,
        configuration=configuration.comparison,
    )
    run = record_training_run(
        comparison=comparison,
        configuration=configuration,
        code_version=code_version,
    )
    package_features = (
        build_operational_history_features(
            dataset=dataset,
            configuration=configuration.features,
            holdout_policy_version=split.policy_version,
        )
        if refit_on_all_completed_cycles
        else features
    )
    package = fit_selected_model_package(
        features=package_features,
        comparison=comparison,
        configuration=configuration,
        code_version=code_version,
    )
    mae = comparison.selected_ridge.evaluation.metrics.mean_absolute_error_days
    if mae is None:
        message = "selected model must have a development MAE"
        raise ValueError(message)

    destination.mkdir(parents=True, exist_ok=True)
    save_model_package(package=package, path=model_path)
    save_training_run(run=run, path=run_path)
    return LocalTrainingResult(
        model_path=model_path,
        run_path=run_path,
        dataset_fingerprint=dataset.fingerprint,
        selected_ridge_alpha=run.selected_ridge_alpha,
        development_mean_absolute_error_days=mae,
        development_forecast_count=(
            comparison.selected_ridge.evaluation.metrics.forecast_count
        ),
    )
