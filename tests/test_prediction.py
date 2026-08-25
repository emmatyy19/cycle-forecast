"""Test local prediction from private files and portable packages."""

from datetime import date
from pathlib import Path

from cycle_forecast.data import build_cycle_dataset, load_cycle_history
from cycle_forecast.features import (
    CYCLE_HISTORY_FEATURE_VERSION,
    DEFAULT_HISTORY_FEATURE_CONFIG,
    build_history_feature_vector,
)
from cycle_forecast.forecasting import WalkForwardContext
from cycle_forecast.models import RIDGE_MODEL_VERSION
from cycle_forecast.prediction import NON_MEDICAL_DISCLAIMER, predict_from_local_files
from cycle_forecast.training import (
    MODEL_PACKAGE_SCHEMA_VERSION,
    ModelPackage,
    save_model_package,
)


def write_test_model(*, path: Path, history_path: Path) -> None:
    """Write a deterministic constant model matching the history transform."""
    records = load_cycle_history(path=history_path)
    dataset = build_cycle_dataset(records=records)
    vector = build_history_feature_vector(
        context=WalkForwardContext(
            cycle_start_date=records[-1].cycle_start_date,
            history=dataset.rows,
        ),
        configuration=DEFAULT_HISTORY_FEATURE_CONFIG,
    )
    width = len(vector.values)
    save_model_package(
        package=ModelPackage(
            schema_version=MODEL_PACKAGE_SCHEMA_VERSION,
            code_version="git:test",
            dataset_fingerprint="sha256:test",
            holdout_policy_version="final-temporal-holdout-v1",
            feature_version=CYCLE_HISTORY_FEATURE_VERSION,
            model_version=RIDGE_MODEL_VERSION,
            feature_configuration=DEFAULT_HISTORY_FEATURE_CONFIG,
            feature_names=vector.feature_names,
            ridge_alpha=1.0,
            scaler_mean=(0.0,) * width,
            scaler_scale=(1.0,) * width,
            ridge_coefficients=(0.0,) * width,
            ridge_intercept=29.0,
        ),
        path=path,
    )


def test_predict_from_local_files_uses_newest_start_as_cutoff(tmp_path: Path) -> None:
    """Forecast the incomplete newest cycle without exposing its future target."""
    history_path = Path("data/synthetic/sample_cycle_history.csv")
    model_path = tmp_path / "model.json"
    write_test_model(path=model_path, history_path=history_path)

    prediction = predict_from_local_files(
        model_path=model_path,
        history_path=history_path,
    )

    assert prediction.current_cycle_start_date == date(2025, 2, 7)
    assert prediction.predicted_cycle_length_days == 29.0
    assert prediction.operational_cycle_length_days == 29
    assert prediction.predicted_next_cycle_start_date == date(2025, 3, 8)
    assert prediction.disclaimer == NON_MEDICAL_DISCLAIMER
