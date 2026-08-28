"""Build lagged and rolling features from completed cycle history."""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum, auto
from statistics import mean, median

from cycle_forecast.data.cycle_history import CycleDataset
from cycle_forecast.forecasting.baselines import WalkForwardContext
from cycle_forecast.forecasting.holdout import TemporalHoldoutSplit

CYCLE_HISTORY_FEATURE_VERSION = "cycle-history-features-v1"
"""Semantic version of the historical feature definitions and ordering."""


class _FeatureConfigField(StrEnum):
    """Identify validated feature-configuration fields without magic strings."""

    LAGS = auto()
    ROLLING_WINDOWS = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class CycleHistoryFeatureConfig:
    """Configure leakage-safe cycle-history features.

    Parameters
    ----------
    lags
        Strictly increasing unique positive lag positions.
    rolling_windows
        Strictly increasing unique positive windows for both rolling mean and
        rolling median features.
    include_expanding_mean
        Whether to include the mean of all completed history.
    """

    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    include_expanding_mean: bool = True

    def __post_init__(self) -> None:
        """Validate deterministic feature configuration invariants."""
        if (
            not self.lags
            and not self.rolling_windows
            and not self.include_expanding_mean
        ):
            message = "feature configuration must select at least one feature"
            raise ValueError(message)
        _validate_positions(values=self.lags, field=_FeatureConfigField.LAGS)
        _validate_positions(
            values=self.rolling_windows,
            field=_FeatureConfigField.ROLLING_WINDOWS,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryFeatureVector:
    """Contain one named, ordered feature vector at a prediction cutoff.

    Parameters
    ----------
    cycle_start_date
        Prediction cutoff for the target cycle.
    feature_names
        Stable semantic names in model-input order.
    values
        Numeric values corresponding positionally to ``feature_names``.

    Notes
    -----
    Positional values are deliberate because downstream estimators consume
    matrices. Names travel with every vector to prevent silent column mismatch.
    """

    cycle_start_date: date
    feature_names: tuple[str, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryFeatureRow:
    """Pair a cutoff-safe feature vector with its supervised target.

    Parameters
    ----------
    vector
        Features calculated strictly from completed prior rows.
    target_cycle_length_days
        Actual cycle length attached only after feature calculation.
    """

    vector: HistoryFeatureVector
    target_cycle_length_days: int


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryFeatureDataset:
    """Contain development-only supervised historical features.

    Parameters
    ----------
    dataset_fingerprint
        Identity of the complete dataset from which the holdout was reserved.
    holdout_policy_version
        Version of the policy that excluded final rows.
    feature_version
        Semantic version of feature definitions and ordering.
    configuration
        Exact lag, rolling-window, and expanding-feature choices.
    feature_names
        Shared model-input column order for every row.
    rows
        Chronological development rows with targets.
    """

    dataset_fingerprint: str
    holdout_policy_version: str
    feature_version: str
    configuration: CycleHistoryFeatureConfig
    feature_names: tuple[str, ...]
    rows: tuple[HistoryFeatureRow, ...]


def _validate_positions(*, values: tuple[int, ...], field: _FeatureConfigField) -> None:
    """Validate positive, unique, strictly increasing feature positions.

    Parameters
    ----------
    values
        Lag positions or rolling windows.
    field
        Closed configuration-field identifier used in an error message.

    Raises
    ------
    ValueError
        If a value is nonpositive, duplicated, or not increasing.
    """
    if any(value < 1 for value in values) or tuple(sorted(set(values))) != values:
        message = (
            f"{field.value} must contain unique positive values in increasing order"
        )
        raise ValueError(message)


DEFAULT_HISTORY_FEATURE_CONFIG = CycleHistoryFeatureConfig(
    lags=(1, 2, 3),
    rolling_windows=(3, 6, 12),
    include_expanding_mean=True,
)
"""Initial candidate history features fixed before final holdout evaluation."""


def _feature_names(*, configuration: CycleHistoryFeatureConfig) -> tuple[str, ...]:
    """Construct deterministic feature names for a configuration.

    Parameters
    ----------
    configuration
        Validated feature configuration.

    Returns
    -------
    tuple[str, ...]
        Feature names in model-input order.
    """
    names = [f"lag_{lag}_cycle_length_days" for lag in configuration.lags]
    for window in configuration.rolling_windows:
        names.extend(
            (
                f"rolling_mean_{window}_cycles",
                f"rolling_median_{window}_cycles",
            )
        )
    if configuration.include_expanding_mean:
        names.append("expanding_mean_cycle_length_days")
    return tuple(names)


def _minimum_history(*, configuration: CycleHistoryFeatureConfig) -> int:
    """Calculate the completed history required by a configuration.

    Parameters
    ----------
    configuration
        Validated feature configuration.

    Returns
    -------
    int
        Minimum number of completed rows needed for all selected features.
    """
    requirements = configuration.lags + configuration.rolling_windows
    return max(requirements, default=1)


def build_history_feature_vector(
    *,
    context: WalkForwardContext,
    configuration: CycleHistoryFeatureConfig = DEFAULT_HISTORY_FEATURE_CONFIG,
) -> HistoryFeatureVector:
    """Transform cutoff-safe completed history into model-ready features.

    Parameters
    ----------
    context
        Prediction cutoff and completed rows strictly before it.
    configuration
        Exact lagged, rolling, and expanding features to calculate.

    Returns
    -------
    HistoryFeatureVector
        Named numeric features in deterministic order.

    Raises
    ------
    ValueError
        If the context does not contain enough completed history.
    """
    required_history = _minimum_history(configuration=configuration)
    if len(context.history) < required_history:
        message = (
            f"history features require at least {required_history} completed cycles"
        )
        raise ValueError(message)

    cycle_lengths = tuple(row.cycle_length_days for row in context.history)
    values = [float(cycle_lengths[-lag]) for lag in configuration.lags]
    for window in configuration.rolling_windows:
        rolling_values = cycle_lengths[-window:]
        values.extend((mean(rolling_values), median(rolling_values)))
    if configuration.include_expanding_mean:
        values.append(mean(cycle_lengths))

    return HistoryFeatureVector(
        cycle_start_date=context.cycle_start_date,
        feature_names=_feature_names(configuration=configuration),
        values=tuple(values),
    )


def build_development_history_features(
    *,
    split: TemporalHoldoutSplit,
    configuration: CycleHistoryFeatureConfig = DEFAULT_HISTORY_FEATURE_CONFIG,
) -> HistoryFeatureDataset:
    """Build supervised features exclusively from development rows.

    Parameters
    ----------
    split
        Versioned temporal split whose holdout rows must remain untouched.
    configuration
        Exact lagged, rolling, and expanding features to calculate.

    Returns
    -------
    HistoryFeatureDataset
        Chronological supervised rows after the required warm-up history.

    Notes
    -----
    The shared :func:`build_history_feature_vector` transformation is used for
    each row. The target row is attached only after its features are calculated
    from the preceding development rows. Holdout rows are never accessed.
    """
    minimum_history = _minimum_history(configuration=configuration)
    rows: list[HistoryFeatureRow] = []
    for position in range(minimum_history, len(split.development_rows)):
        target_row = split.development_rows[position]
        vector = build_history_feature_vector(
            context=WalkForwardContext(
                cycle_start_date=target_row.cycle_start_date,
                history=split.development_rows[:position],
            ),
            configuration=configuration,
        )
        rows.append(
            HistoryFeatureRow(
                vector=vector,
                target_cycle_length_days=target_row.cycle_length_days,
            )
        )

    return HistoryFeatureDataset(
        dataset_fingerprint=split.dataset_fingerprint,
        holdout_policy_version=split.policy_version,
        feature_version=CYCLE_HISTORY_FEATURE_VERSION,
        configuration=configuration,
        feature_names=_feature_names(configuration=configuration),
        rows=tuple(rows),
    )


def build_operational_history_features(
    *,
    dataset: CycleDataset,
    configuration: CycleHistoryFeatureConfig,
    holdout_policy_version: str,
) -> HistoryFeatureDataset:
    """Build deployment features from every completed cycle after evaluation.

    Parameters
    ----------
    dataset
        Complete validated cycle dataset.
    configuration
        Feature settings selected without consulting final holdout outcomes.
    holdout_policy_version
        Evaluation policy retained as package provenance.

    Returns
    -------
    HistoryFeatureDataset
        Supervised rows spanning all completed cycles after warm-up.

    Notes
    -----
    This function is for operational refitting only after model family and
    hyperparameters have been selected on development data. Evaluation metrics
    remain those calculated before the final holdout is incorporated.
    """
    minimum_history = _minimum_history(configuration=configuration)
    rows: list[HistoryFeatureRow] = []
    for position in range(minimum_history, len(dataset.rows)):
        target_row = dataset.rows[position]
        vector = build_history_feature_vector(
            context=WalkForwardContext(
                cycle_start_date=target_row.cycle_start_date,
                history=dataset.rows[:position],
            ),
            configuration=configuration,
        )
        rows.append(
            HistoryFeatureRow(
                vector=vector,
                target_cycle_length_days=target_row.cycle_length_days,
            )
        )
    return HistoryFeatureDataset(
        dataset_fingerprint=dataset.fingerprint,
        holdout_policy_version=holdout_policy_version,
        feature_version=CYCLE_HISTORY_FEATURE_VERSION,
        configuration=configuration,
        feature_names=_feature_names(configuration=configuration),
        rows=tuple(rows),
    )
