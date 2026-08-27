"""Fit a calibrated discrete-time logistic wearable survival model."""

# Scikit-learn's installed package does not provide complete inline typing.
# Keep that limitation at this adapter boundary while exposing precise domain types.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
from itertools import pairwise
from math import isfinite
from typing import Final

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cycle_forecast.features.wearable import WearableFeatureRow
from cycle_forecast.forecasting.daily import (
    DAILY_FORECAST_HORIZON_DAYS,
    DailyPeriodDistribution,
    distribution_from_hazards,
)

DISCRETE_SURVIVAL_MODEL_VERSION: Final = "wearable-discrete-survival-v1"
"""Semantic version of model features, risk expansion, and prediction."""

PLATT_CALIBRATION_VERSION: Final = "hazard-platt-calibration-v1"
"""Semantic version of chronological hazard calibration."""


class _PipelineStep(StrEnum):
    """Identify sklearn steps without string-based control flow."""

    SCALE = auto()
    LOGISTIC = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class DiscreteSurvivalConfig:
    """Configure regularized hazard fitting and minimum sample sizes."""

    inverse_regularization_strength: float = 1.0
    minimum_training_rows: int = 30
    minimum_calibration_rows: int = 10

    def __post_init__(self) -> None:
        """Validate positive finite model settings."""
        if (
            not isfinite(self.inverse_regularization_strength)
            or self.inverse_regularization_strength <= 0.0
        ):
            raise ValueError("inverse_regularization_strength must be positive")
        if self.minimum_training_rows < 1 or self.minimum_calibration_rows < 1:
            raise ValueError("minimum row counts must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class FittedDiscreteSurvivalModel:
    """Contain a fitted hazard pipeline and separate Platt calibrator."""

    model_version: str
    calibration_version: str
    feature_names: tuple[str, ...]
    configuration: DiscreteSurvivalConfig
    fitted_through: datetime
    calibration_through: datetime
    pipeline: Pipeline
    calibrator: LogisticRegression


DEFAULT_DISCRETE_SURVIVAL_CONFIG: Final = DiscreteSurvivalConfig()
"""Initial fixed survival configuration for development evaluation."""


def _expand_rows(
    *, rows: tuple[WearableFeatureRow, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Expand labeled mornings into binary at-risk person-period rows."""
    matrix: list[tuple[float, ...]] = []
    targets: list[float] = []
    for row in rows:
        outcome = row.outcome_offset_days
        if outcome is None:
            continue
        final_offset = min(outcome, DAILY_FORECAST_HORIZON_DAYS - 1)
        for offset in range(final_offset + 1):
            matrix.append((*row.values, float(offset)))
            targets.append(float(outcome == offset))
    if not matrix:
        raise ValueError("survival fitting requires uncensored labeled rows")
    return np.asarray(matrix, dtype=np.float64), np.asarray(targets, dtype=np.int64)


def _require_binary_classes(*, targets: np.ndarray, name: str) -> None:
    """Require both event and no-event examples for logistic fitting."""
    if len(np.unique(targets)) != 2:
        raise ValueError(f"{name} requires both event and no-event examples")


def fit_discrete_survival_model(
    *,
    training_rows: tuple[WearableFeatureRow, ...],
    calibration_rows: tuple[WearableFeatureRow, ...],
    configuration: DiscreteSurvivalConfig = DEFAULT_DISCRETE_SURVIVAL_CONFIG,
) -> FittedDiscreteSurvivalModel:
    """Fit coefficients on early rows and calibrate on a later temporal block.

    Parameters
    ----------
    training_rows
        Chronological early labeled mornings used for coefficient fitting.
    calibration_rows
        Strictly later labeled mornings used only for Platt calibration.
    configuration
        Regularization and minimum sample requirements.

    Returns
    -------
    FittedDiscreteSurvivalModel
        Versioned fitted pipeline and independent hazard calibrator.
    """
    if len(training_rows) < configuration.minimum_training_rows:
        raise ValueError("insufficient survival training rows")
    if len(calibration_rows) < configuration.minimum_calibration_rows:
        raise ValueError("insufficient survival calibration rows")
    if not training_rows or not calibration_rows:
        raise ValueError("training and calibration rows must be nonempty")
    if any(
        current.aligned.prediction_cutoff >= following.aligned.prediction_cutoff
        for current, following in pairwise(training_rows)
    ) or any(
        current.aligned.prediction_cutoff >= following.aligned.prediction_cutoff
        for current, following in pairwise(calibration_rows)
    ):
        raise ValueError("survival rows must be strictly chronological")
    if (
        training_rows[-1].aligned.prediction_cutoff
        >= calibration_rows[0].aligned.prediction_cutoff
    ):
        raise ValueError("calibration rows must follow all training rows")
    feature_names = training_rows[0].feature_names
    if any(
        row.feature_names != feature_names for row in training_rows + calibration_rows
    ):
        raise ValueError("survival rows must share one feature schema")

    training_matrix, training_targets = _expand_rows(rows=training_rows)
    _require_binary_classes(targets=training_targets, name="survival training")
    pipeline = Pipeline(
        steps=(
            (_PipelineStep.SCALE, StandardScaler()),
            (
                _PipelineStep.LOGISTIC,
                LogisticRegression(
                    C=configuration.inverse_regularization_strength,
                    max_iter=2_000,
                    random_state=0,
                ),
            ),
        )
    )
    pipeline.fit(training_matrix, training_targets)

    calibration_matrix, calibration_targets = _expand_rows(rows=calibration_rows)
    _require_binary_classes(targets=calibration_targets, name="hazard calibration")
    raw_probabilities = np.asarray(
        pipeline.predict_proba(calibration_matrix), dtype=np.float64
    )[:, 1].reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=2_000, random_state=0)
    calibrator.fit(raw_probabilities, calibration_targets)
    return FittedDiscreteSurvivalModel(
        model_version=DISCRETE_SURVIVAL_MODEL_VERSION,
        calibration_version=PLATT_CALIBRATION_VERSION,
        feature_names=feature_names,
        configuration=configuration,
        fitted_through=training_rows[-1].aligned.prediction_cutoff,
        calibration_through=calibration_rows[-1].aligned.prediction_cutoff,
        pipeline=pipeline,
        calibrator=calibrator,
    )


def predict_with_discrete_survival_model(
    *, model: FittedDiscreteSurvivalModel, row: WearableFeatureRow
) -> DailyPeriodDistribution:
    """Predict a calibrated exhaustive distribution for one later morning."""
    if row.feature_names != model.feature_names:
        raise ValueError("prediction feature schema does not match fitted model")
    if row.aligned.prediction_cutoff <= model.calibration_through:
        raise ValueError("prediction row must follow the calibration block")
    matrix = np.asarray(
        [(*row.values, float(offset)) for offset in range(DAILY_FORECAST_HORIZON_DAYS)],
        dtype=np.float64,
    )
    raw = np.asarray(model.pipeline.predict_proba(matrix), dtype=np.float64)[
        :, 1
    ].reshape(-1, 1)
    calibrated = np.asarray(model.calibrator.predict_proba(raw), dtype=np.float64)[:, 1]
    hazards = tuple(float(value) for value in calibrated)
    return distribution_from_hazards(
        prediction_date=row.aligned.prediction_date,
        prediction_cutoff=row.aligned.prediction_cutoff,
        hazards=hazards,
    )
