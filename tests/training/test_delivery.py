"""Test versioned configuration, run records, and portable model packages."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from cycle_forecast.data import CycleHistoryRecord, build_cycle_dataset
from cycle_forecast.evaluation import (
    DevelopmentModelComparison,
    compare_development_forecasters,
)
from cycle_forecast.features import (
    HistoryFeatureDataset,
    build_development_history_features,
    build_history_feature_vector,
)
from cycle_forecast.forecasting import (
    TemporalHoldoutSplit,
    WalkForwardContext,
    split_final_temporal_holdout,
)
from cycle_forecast.training import (
    TrainingConfig,
    fit_selected_model_package,
    load_model_package,
    load_training_config,
    predict_with_model_package,
    record_training_run,
    save_model_package,
    save_training_run,
)

PROJECT_ROOT = Path(__file__).parents[2]
"""Repository root containing the committed training configuration."""


def _training_inputs() -> tuple[
    TrainingConfig,
    HistoryFeatureDataset,
    DevelopmentModelComparison,
    TemporalHoldoutSplit,
]:
    """Build deterministic inputs spanning configuration through selection."""
    lengths = tuple(27 + position % 5 for position in range(72))
    starts = [date(2015, 1, 1)]
    for length in lengths:
        starts.append(starts[-1] + timedelta(days=length))
    dataset = build_cycle_dataset(
        records=tuple(
            CycleHistoryRecord(cycle_start_date=start, period_length_days=5)
            for start in starts
        )
    )
    split = split_final_temporal_holdout(dataset=dataset)
    configuration = load_training_config(path=PROJECT_ROOT / "configs/phase_a.toml")
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
    return configuration, features, comparison, split


def test_committed_training_configuration_matches_predeclared_candidates() -> None:
    """Load the repository's versioned settings without hidden defaults."""
    configuration = load_training_config(path=PROJECT_ROOT / "configs/phase_a.toml")

    assert configuration.schema_version == "phase-a-training-config-v1"
    assert configuration.features.lags == (1, 2, 3)
    assert configuration.comparison.ridge_alphas == (0.01, 0.1, 1.0, 10.0, 100.0)


def test_training_run_records_provenance_configuration_and_all_metrics(
    tmp_path: Path,
) -> None:
    """Capture enough immutable context to interpret a selection result."""
    configuration, _, comparison, _ = _training_inputs()

    run = record_training_run(
        comparison=comparison,
        configuration=configuration,
        code_version="git:abc123",
        recorded_at=datetime(2026, 8, 24, 12, tzinfo=UTC),
    )

    assert run.code_version == "git:abc123"
    assert run.dataset_fingerprint == comparison.dataset_fingerprint
    assert len(run.metrics) == len(comparison.entries)
    assert run.selected_ridge_alpha == comparison.selected_ridge.ridge_alpha
    assert run.recorded_at == "2026-08-24T12:00:00+00:00"
    manifest_path = tmp_path / "training-run.json"
    save_training_run(run=run, path=manifest_path)
    assert '"code_version": "git:abc123"' in manifest_path.read_text(encoding="utf-8")


def test_model_package_round_trip_reproduces_fitted_pipeline_prediction(
    tmp_path: Path,
) -> None:
    """Persist fitted parameters while retaining the shared feature transform."""
    configuration, features, comparison, split = _training_inputs()
    package = fit_selected_model_package(
        features=features,
        comparison=comparison,
        configuration=configuration,
        code_version="git:abc123",
    )
    path = tmp_path / "selected-model.json"

    save_model_package(package=package, path=path)
    loaded = load_model_package(path=path)
    context = WalkForwardContext(
        cycle_start_date=split.holdout_rows[0].cycle_start_date,
        history=split.development_rows,
    )
    prediction = predict_with_model_package(package=loaded, context=context)
    prediction_context_values = build_history_feature_vector(
        context=context,
        configuration=configuration.features,
    )
    standardized = tuple(
        (value - center) / scale
        for value, center, scale in zip(
            prediction_context_values.values,
            package.scaler_mean,
            package.scaler_scale,
            strict=True,
        )
    )
    expected = package.ridge_intercept + sum(
        coefficient * value
        for coefficient, value in zip(
            package.ridge_coefficients, standardized, strict=True
        )
    )

    assert loaded == package
    assert prediction.predicted_cycle_length_days == pytest.approx(expected)
    assert prediction.cycle_start_date == split.holdout_rows[0].cycle_start_date


def test_training_configuration_rejects_an_unknown_schema(tmp_path: Path) -> None:
    """Prevent silent interpretation of a future configuration contract."""
    path = tmp_path / "config.toml"
    path.write_text('schema_version = "future"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported training configuration"):
        load_training_config(path=path)
