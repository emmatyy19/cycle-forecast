"""Refresh the local Phase A package only when cycle history changes."""

import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

from cycle_forecast.data import build_cycle_dataset, load_cycle_history
from cycle_forecast.training.delivery import load_model_package
from cycle_forecast.training.workflow import (
    DEFAULT_RUN_FILENAME,
    LocalTrainingResult,
    train_from_local_history,
)


class DailyModelRefreshStatus(StrEnum):
    """Identify whether the daily flow reused, created, or refreshed a model."""

    CURRENT = auto()
    CREATED = auto()
    REFRESHED = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class DailyModelRefreshResult:
    """Describe model freshness and optional retraining performance."""

    status: DailyModelRefreshStatus
    model_path: Path
    run_path: Path
    dataset_fingerprint: str
    training: LocalTrainingResult | None


def refresh_daily_model_if_needed(
    *,
    history_path: Path,
    model_path: Path,
    configuration_path: Path,
    code_version: str,
) -> DailyModelRefreshResult:
    """Reuse a current model or train complete replacement artifacts.

    Parameters
    ----------
    history_path
        Validated private cycle-history CSV.
    model_path
        Preferred local Phase A package path.
    configuration_path
        Versioned Phase A training configuration.
    code_version
        Application version recorded in refreshed artifacts.

    Returns
    -------
    DailyModelRefreshResult
        Freshness status, artifact paths, fingerprint, and optional metrics.

    Raises
    ------
    OSError
        If inputs or replacement artifacts cannot be read or written.
    ValueError
        If history, configuration, or an existing package is invalid.

    Notes
    -----
    Training writes both artifacts into a temporary sibling directory. Existing
    artifacts are not replaced until the complete training workflow succeeds.
    """
    dataset = build_cycle_dataset(records=load_cycle_history(path=history_path))
    run_path = model_path.parent / DEFAULT_RUN_FILENAME
    model_existed = model_path.exists()
    if model_existed:
        package = load_model_package(path=model_path)
        if package.dataset_fingerprint == dataset.fingerprint and run_path.is_file():
            return DailyModelRefreshResult(
                status=DailyModelRefreshStatus.CURRENT,
                model_path=model_path,
                run_path=run_path,
                dataset_fingerprint=dataset.fingerprint,
                training=None,
            )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".daily-model-refresh-",
        dir=model_path.parent,
    ) as temporary_directory:
        temporary_result = train_from_local_history(
            history_path=history_path,
            configuration_path=configuration_path,
            output_directory=temporary_directory,
            code_version=code_version,
            refit_on_all_completed_cycles=True,
        )
        os.replace(temporary_result.model_path, model_path)
        os.replace(temporary_result.run_path, run_path)

    training = LocalTrainingResult(
        model_path=model_path,
        run_path=run_path,
        dataset_fingerprint=temporary_result.dataset_fingerprint,
        selected_ridge_alpha=temporary_result.selected_ridge_alpha,
        development_mean_absolute_error_days=(
            temporary_result.development_mean_absolute_error_days
        ),
        development_forecast_count=temporary_result.development_forecast_count,
    )
    return DailyModelRefreshResult(
        status=(
            DailyModelRefreshStatus.REFRESHED
            if model_existed
            else DailyModelRefreshStatus.CREATED
        ),
        model_path=model_path,
        run_path=run_path,
        dataset_fingerprint=dataset.fingerprint,
        training=training,
    )
