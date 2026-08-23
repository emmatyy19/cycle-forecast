"""Fit regularized linear cycle-length models in walk-forward order."""

# Scikit-learn does not publish complete inline types, and the separately
# maintained stub package lags the installed release. Keep that limitation at
# this adapter boundary while preserving strict project-facing annotations.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum, auto
from math import isfinite

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cycle_forecast.features import (
    CycleHistoryFeatureConfig,
    HistoryFeatureDataset,
)
from cycle_forecast.forecasting import (
    CycleLengthForecast,
    ForecastBatch,
    round_cycle_length_days,
)

RIDGE_MODEL_NAME = "ridge-regression"
"""Stable name of the first regularized regression model."""

RIDGE_MODEL_VERSION = "ridge-regression-v1"
"""Semantic version of Ridge fitting and prediction behavior."""


class _PipelineStep(StrEnum):
    """Identify sklearn pipeline steps without magic strings."""

    SCALE = auto()
    REGRESSION = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class RidgeForecastConfig:
    """Configure walk-forward Ridge regression.

    Parameters
    ----------
    alpha
        Positive L2 regularization strength. Larger values shrink coefficients
        more strongly toward zero.
    minimum_training_rows
        Positive number of earlier supervised feature rows required before a
        forecast is emitted.
    """

    alpha: float
    minimum_training_rows: int

    def __post_init__(self) -> None:
        """Validate Ridge configuration boundaries."""
        if not isfinite(self.alpha) or self.alpha <= 0:
            message = "alpha must be positive and finite"
            raise ValueError(message)
        if self.minimum_training_rows < 1:
            message = "minimum_training_rows must be positive"
            raise ValueError(message)


DEFAULT_RIDGE_CONFIG = RidgeForecastConfig(
    alpha=1.0,
    minimum_training_rows=12,
)
"""Initial fixed Ridge configuration for development-only evaluation."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RidgeWalkForwardResult:
    """Contain Ridge forecasts and their complete development configuration.

    Parameters
    ----------
    dataset_fingerprint
        Identity of the complete dataset whose holdout remains reserved.
    holdout_policy_version
        Version of the temporal holdout policy used before feature construction.
    feature_version
        Semantic version of the shared feature transformation.
    feature_configuration
        Exact historical features supplied to Ridge.
    model_version
        Semantic version of model fitting and prediction behavior.
    model_configuration
        Exact regularization and minimum-training-row configuration.
    forecast_batch
        Chronological development-only walk-forward forecasts.
    """

    dataset_fingerprint: str
    holdout_policy_version: str
    feature_version: str
    feature_configuration: CycleHistoryFeatureConfig
    model_version: str
    model_configuration: RidgeForecastConfig
    forecast_batch: ForecastBatch


def _forecaster_name(*, configuration: RidgeForecastConfig) -> str:
    """Construct a stable identity containing evaluation-relevant settings.

    Parameters
    ----------
    configuration
        Validated Ridge configuration.

    Returns
    -------
    str
        Configuration-specific forecaster name.
    """
    return (
        f"{RIDGE_MODEL_NAME}-alpha-{configuration.alpha:g}"
        f"-min-training-{configuration.minimum_training_rows}"
    )


def _new_pipeline(*, configuration: RidgeForecastConfig) -> Pipeline:
    """Create an unfitted scaler and Ridge pipeline for one cutoff.

    Parameters
    ----------
    configuration
        Validated Ridge configuration.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fresh, unfitted pipeline whose scaler cannot retain future information.
    """
    return Pipeline(
        steps=(
            (_PipelineStep.SCALE, StandardScaler()),
            (_PipelineStep.REGRESSION, Ridge(alpha=configuration.alpha)),
        )
    )


def _validate_feature_dataset(*, features: HistoryFeatureDataset) -> None:
    """Validate feature schema and chronological row invariants.

    Parameters
    ----------
    features
        Development-only supervised feature rows.

    Raises
    ------
    ValueError
        If a row has a mismatched schema, width, or chronological cutoff.
    """
    previous_start = None
    expected_width = len(features.feature_names)
    for row in features.rows:
        vector = row.vector
        if vector.feature_names != features.feature_names:
            message = "feature row names do not match dataset feature names"
            raise ValueError(message)
        if len(vector.values) != expected_width:
            message = "feature row width does not match dataset feature names"
            raise ValueError(message)
        if previous_start is not None and vector.cycle_start_date <= previous_start:
            message = "feature rows must be strictly chronological and unique"
            raise ValueError(message)
        previous_start = vector.cycle_start_date


def forecast_with_walk_forward_ridge(
    *,
    features: HistoryFeatureDataset,
    configuration: RidgeForecastConfig = DEFAULT_RIDGE_CONFIG,
) -> RidgeWalkForwardResult:
    """Fit on past development rows and forecast each next development row.

    Parameters
    ----------
    features
        Development-only supervised rows produced by the shared feature
        transformation after reserving the final holdout.
    configuration
        Ridge regularization and minimum-training-row settings.

    Returns
    -------
    RidgeWalkForwardResult
        Development-only forecasts and complete feature/model provenance.

    Raises
    ------
    ValueError
        If feature rows do not share one chronological schema or Ridge produces
        a nonpositive or nonfinite prediction.

    Notes
    -----
    A fresh ``StandardScaler`` and ``Ridge`` pipeline is fit at every cutoff.
    The current row supplies only its feature vector to ``predict``; its target
    joins the training set only for later cutoffs.
    """
    _validate_feature_dataset(features=features)
    forecasts: list[CycleLengthForecast] = []
    for position in range(configuration.minimum_training_rows, len(features.rows)):
        training_rows = features.rows[:position]
        current_row = features.rows[position]
        training_matrix = np.asarray(
            [row.vector.values for row in training_rows],
            dtype=np.float64,
        )
        training_targets = np.asarray(
            [row.target_cycle_length_days for row in training_rows],
            dtype=np.float64,
        )
        prediction_matrix = np.asarray(
            [current_row.vector.values],
            dtype=np.float64,
        )

        pipeline = _new_pipeline(configuration=configuration)
        pipeline.fit(training_matrix, training_targets)
        raw_prediction = float(pipeline.predict(prediction_matrix)[0])
        operational_prediction = round_cycle_length_days(value=raw_prediction)
        forecasts.append(
            CycleLengthForecast(
                cycle_start_date=current_row.vector.cycle_start_date,
                predicted_cycle_length_days=raw_prediction,
                operational_cycle_length_days=operational_prediction,
                predicted_next_cycle_start_date=(
                    current_row.vector.cycle_start_date
                    + timedelta(days=operational_prediction)
                ),
            )
        )

    forecast_batch = ForecastBatch(
        forecaster_name=_forecaster_name(configuration=configuration),
        forecaster_version=RIDGE_MODEL_VERSION,
        dataset_fingerprint=features.dataset_fingerprint,
        forecasts=tuple(forecasts),
    )
    return RidgeWalkForwardResult(
        dataset_fingerprint=features.dataset_fingerprint,
        holdout_policy_version=features.holdout_policy_version,
        feature_version=features.feature_version,
        feature_configuration=features.configuration,
        model_version=RIDGE_MODEL_VERSION,
        model_configuration=configuration,
        forecast_batch=forecast_batch,
    )
