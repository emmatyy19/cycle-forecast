"""Load versioned training configuration and package selected Ridge models."""

# Scikit-learn's published inline types are incomplete. Unknown estimator member
# types are contained at this adapter boundary.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Final, cast

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cycle_forecast.evaluation import (
    DevelopmentComparisonConfig,
    DevelopmentModelComparison,
)
from cycle_forecast.features import (
    CycleHistoryFeatureConfig,
    HistoryFeatureDataset,
    build_history_feature_vector,
)
from cycle_forecast.forecasting import (
    CycleLengthForecast,
    WalkForwardContext,
    round_cycle_length_days,
)
from cycle_forecast.models import RIDGE_MODEL_VERSION

TRAINING_CONFIG_SCHEMA_VERSION: Final = "phase-a-training-config-v1"
"""Supported version of committed Phase A training configuration."""

MODEL_PACKAGE_SCHEMA_VERSION: Final = "ridge-model-package-v1"
"""Supported portable selected-model package format."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingConfig:
    """Contain all versioned settings needed for Phase A model selection."""

    schema_version: str
    features: CycleHistoryFeatureConfig
    comparison: DevelopmentComparisonConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingRun:
    """Record immutable provenance and development metrics for one run."""

    recorded_at: str
    code_version: str
    dataset_fingerprint: str
    holdout_policy_version: str
    feature_version: str
    model_version: str
    configuration: TrainingConfig
    metrics: tuple[Mapping[str, object], ...]
    selected_ridge_alpha: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelPackage:
    """Store a fitted Ridge model and its shared feature contract portably."""

    schema_version: str
    code_version: str
    dataset_fingerprint: str
    holdout_policy_version: str
    feature_version: str
    model_version: str
    feature_configuration: CycleHistoryFeatureConfig
    feature_names: tuple[str, ...]
    ridge_alpha: float
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    ridge_coefficients: tuple[float, ...]
    ridge_intercept: float


def _table(*, value: object, name: str) -> dict[str, object]:
    """Narrow a TOML value to a table or raise a configuration error."""
    if not isinstance(value, dict):
        message = f"{name} must be a TOML table"
        raise ValueError(message)
    untyped_table = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in untyped_table):
        message = f"{name} must be a TOML table"
        raise ValueError(message)
    return {key: item for key, item in untyped_table.items() if isinstance(key, str)}


def _number_tuple(*, value: object, name: str) -> tuple[float, ...]:
    """Narrow a TOML array to non-boolean numeric values."""
    if not isinstance(value, list):
        message = f"{name} must be an array of numbers"
        raise ValueError(message)
    numbers: list[float] = []
    for item in cast(list[object], value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            message = f"{name} must be an array of numbers"
            raise ValueError(message)
        numbers.append(float(item))
    return tuple(numbers)


def _integer_tuple(*, value: object, name: str) -> tuple[int, ...]:
    """Narrow a TOML array to integer values."""
    if not isinstance(value, list):
        message = f"{name} must be an array of integers"
        raise ValueError(message)
    integers: list[int] = []
    for item in cast(list[object], value):
        if isinstance(item, bool) or not isinstance(item, int):
            message = f"{name} must be an array of integers"
            raise ValueError(message)
        integers.append(item)
    return tuple(integers)


def load_training_config(*, path: str | Path) -> TrainingConfig:
    """Load and validate a versioned Phase A training TOML file.

    Parameters
    ----------
    path
        Training configuration file.

    Returns
    -------
    TrainingConfig
        Validated feature and candidate-selection settings.

    Raises
    ------
    OSError
        If the file cannot be read.
    ValueError
        If its schema or values are unsupported.
    """
    with Path(path).open("rb") as file_handle:
        payload = tomllib.load(file_handle)
    if payload.get("schema_version") != TRAINING_CONFIG_SCHEMA_VERSION:
        message = "unsupported training configuration schema_version"
        raise ValueError(message)
    features = _table(value=payload.get("features"), name="features")
    comparison = _table(value=payload.get("comparison"), name="comparison")
    include_expanding_mean = features.get("include_expanding_mean")
    minimum_rows = comparison.get("minimum_ridge_training_rows")
    if not isinstance(include_expanding_mean, bool):
        message = "features.include_expanding_mean must be a boolean"
        raise ValueError(message)
    if isinstance(minimum_rows, bool) or not isinstance(minimum_rows, int):
        message = "comparison.minimum_ridge_training_rows must be an integer"
        raise ValueError(message)
    return TrainingConfig(
        schema_version=TRAINING_CONFIG_SCHEMA_VERSION,
        features=CycleHistoryFeatureConfig(
            lags=_integer_tuple(value=features.get("lags"), name="features.lags"),
            rolling_windows=_integer_tuple(
                value=features.get("rolling_windows"),
                name="features.rolling_windows",
            ),
            include_expanding_mean=include_expanding_mean,
        ),
        comparison=DevelopmentComparisonConfig(
            ridge_alphas=_number_tuple(
                value=comparison.get("ridge_alphas"),
                name="comparison.ridge_alphas",
            ),
            rolling_windows=_integer_tuple(
                value=comparison.get("rolling_windows"),
                name="comparison.rolling_windows",
            ),
            minimum_ridge_training_rows=minimum_rows,
        ),
    )


def record_training_run(
    *,
    comparison: DevelopmentModelComparison,
    configuration: TrainingConfig,
    code_version: str,
    recorded_at: datetime | None = None,
) -> TrainingRun:
    """Create a complete provenance and metrics record for a selection run."""
    if not code_version.strip():
        message = "code_version must be non-empty"
        raise ValueError(message)
    if comparison.configuration != configuration.comparison:
        message = "comparison was not produced with the supplied configuration"
        raise ValueError(message)
    selected_alpha = comparison.selected_ridge.ridge_alpha
    if selected_alpha is None:
        message = "selected Ridge entry must include an alpha"
        raise ValueError(message)
    timestamp = recorded_at or datetime.now(tz=UTC)
    if timestamp.tzinfo is None:
        message = "recorded_at must be timezone-aware"
        raise ValueError(message)
    metrics: tuple[Mapping[str, object], ...] = tuple(
        {
            "label": entry.label,
            "kind": entry.kind.value,
            "ridge_alpha": entry.ridge_alpha,
            "forecast_count": entry.evaluation.metrics.forecast_count,
            "mean_error_days": entry.evaluation.metrics.mean_error_days,
            "mean_absolute_error_days": (
                entry.evaluation.metrics.mean_absolute_error_days
            ),
            "median_absolute_error_days": (
                entry.evaluation.metrics.median_absolute_error_days
            ),
            "root_mean_squared_error_days": (
                entry.evaluation.metrics.root_mean_squared_error_days
            ),
            "within_1_day": entry.evaluation.metrics.within_1_day,
            "within_2_days": entry.evaluation.metrics.within_2_days,
            "within_3_days": entry.evaluation.metrics.within_3_days,
            "within_5_days": entry.evaluation.metrics.within_5_days,
        }
        for entry in comparison.entries
    )
    return TrainingRun(
        recorded_at=timestamp.astimezone(UTC).isoformat(),
        code_version=code_version,
        dataset_fingerprint=comparison.dataset_fingerprint,
        holdout_policy_version=comparison.holdout_policy_version,
        feature_version=comparison.feature_version,
        model_version=RIDGE_MODEL_VERSION,
        configuration=configuration,
        metrics=metrics,
        selected_ridge_alpha=selected_alpha,
    )


def fit_selected_model_package(
    *,
    features: HistoryFeatureDataset,
    comparison: DevelopmentModelComparison,
    configuration: TrainingConfig,
    code_version: str,
) -> ModelPackage:
    """Fit the selected Ridge candidate on every development feature row."""
    if not features.rows:
        message = "at least one development feature row is required"
        raise ValueError(message)
    if (
        features.dataset_fingerprint != comparison.dataset_fingerprint
        or features.feature_version != comparison.feature_version
        or features.configuration != configuration.features
    ):
        message = "features, comparison, and configuration provenance must match"
        raise ValueError(message)
    selected_alpha = comparison.selected_ridge.ridge_alpha
    if selected_alpha is None:
        message = "selected Ridge entry must include an alpha"
        raise ValueError(message)
    matrix = np.asarray([row.vector.values for row in features.rows], dtype=np.float64)
    targets = np.asarray(
        [row.target_cycle_length_days for row in features.rows], dtype=np.float64
    )
    scaler = StandardScaler()
    ridge = Ridge(alpha=selected_alpha)
    pipeline = Pipeline((("scale", scaler), ("regression", ridge)))
    pipeline.fit(matrix, targets)
    return ModelPackage(
        schema_version=MODEL_PACKAGE_SCHEMA_VERSION,
        code_version=code_version,
        dataset_fingerprint=features.dataset_fingerprint,
        holdout_policy_version=features.holdout_policy_version,
        feature_version=features.feature_version,
        model_version=RIDGE_MODEL_VERSION,
        feature_configuration=features.configuration,
        feature_names=features.feature_names,
        ridge_alpha=selected_alpha,
        scaler_mean=tuple(
            float(value)
            for value in np.asarray(scaler.mean_, dtype=np.float64).reshape(-1)
        ),
        scaler_scale=tuple(
            float(value)
            for value in np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)
        ),
        ridge_coefficients=tuple(
            float(value)
            for value in np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
        ),
        ridge_intercept=float(ridge.intercept_),
    )


def save_training_run(*, run: TrainingRun, path: str | Path) -> None:
    """Write a training-run manifest as deterministic JSON.

    Parameters
    ----------
    run
        Complete run provenance, configuration, and development metrics.
    path
        Destination under a local, ignored experiment-artifact directory.
    """
    Path(path).write_text(
        json.dumps(asdict(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def predict_with_model_package(
    *, package: ModelPackage, context: WalkForwardContext
) -> CycleLengthForecast:
    """Predict from a package using the shared cutoff-safe transformation."""
    vector = build_history_feature_vector(
        context=context,
        configuration=package.feature_configuration,
    )
    if vector.feature_names != package.feature_names:
        message = "model package feature names do not match shared transformation"
        raise ValueError(message)
    standardized = tuple(
        (value - center) / scale
        for value, center, scale in zip(
            vector.values, package.scaler_mean, package.scaler_scale, strict=True
        )
    )
    prediction = package.ridge_intercept + sum(
        coefficient * value
        for coefficient, value in zip(
            package.ridge_coefficients, standardized, strict=True
        )
    )
    if not isfinite(prediction) or prediction <= 0:
        message = "packaged model produced a nonpositive or nonfinite prediction"
        raise ValueError(message)
    operational = round_cycle_length_days(value=prediction)
    return CycleLengthForecast(
        cycle_start_date=context.cycle_start_date,
        predicted_cycle_length_days=prediction,
        operational_cycle_length_days=operational,
        predicted_next_cycle_start_date=context.cycle_start_date
        + timedelta(days=operational),
    )


def save_model_package(*, package: ModelPackage, path: str | Path) -> None:
    """Write a portable model package as deterministic JSON."""
    payload = asdict(package)
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_model_package(*, path: str | Path) -> ModelPackage:
    """Load and validate a portable JSON model package."""
    # ``json.loads`` is typed as Any; immediately contain it behind validators.
    raw = cast(object, json.loads(Path(path).read_text(encoding="utf-8")))
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") != MODEL_PACKAGE_SCHEMA_VERSION
    ):
        message = "unsupported model package schema_version"
        raise ValueError(message)
    payload = _table(value=raw, name="model package")
    feature_raw = _table(
        value=payload.get("feature_configuration"),
        name="model package feature_configuration",
    )
    include_expanding_mean = feature_raw.get("include_expanding_mean")
    if not isinstance(include_expanding_mean, bool):
        message = "model package feature_configuration must be an object"
        raise ValueError(message)
    try:
        return ModelPackage(
            schema_version=MODEL_PACKAGE_SCHEMA_VERSION,
            code_version=str(payload["code_version"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            holdout_policy_version=str(payload["holdout_policy_version"]),
            feature_version=str(payload["feature_version"]),
            model_version=str(payload["model_version"]),
            feature_configuration=CycleHistoryFeatureConfig(
                lags=_integer_tuple(
                    value=feature_raw.get("lags"), name="model package lags"
                ),
                rolling_windows=_integer_tuple(
                    value=feature_raw.get("rolling_windows"),
                    name="model package rolling_windows",
                ),
                include_expanding_mean=include_expanding_mean,
            ),
            feature_names=_string_tuple(
                value=payload.get("feature_names"), name="feature_names"
            ),
            ridge_alpha=_float_value(value=payload["ridge_alpha"], name="ridge_alpha"),
            scaler_mean=_number_tuple(
                value=payload.get("scaler_mean"), name="scaler_mean"
            ),
            scaler_scale=_number_tuple(
                value=payload.get("scaler_scale"), name="scaler_scale"
            ),
            ridge_coefficients=_number_tuple(
                value=payload.get("ridge_coefficients"),
                name="ridge_coefficients",
            ),
            ridge_intercept=_float_value(
                value=payload["ridge_intercept"], name="ridge_intercept"
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        message = "invalid model package contents"
        raise ValueError(message) from error


def _string_tuple(*, value: object, name: str) -> tuple[str, ...]:
    """Narrow a decoded JSON value to an array of strings."""
    if not isinstance(value, list):
        message = f"{name} must be an array of strings"
        raise ValueError(message)
    strings: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            message = f"{name} must be an array of strings"
            raise ValueError(message)
        strings.append(item)
    return tuple(strings)


def _float_value(*, value: object, name: str) -> float:
    """Narrow a decoded JSON value to a non-boolean number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = f"{name} must be a number"
        raise ValueError(message)
    return float(value)
