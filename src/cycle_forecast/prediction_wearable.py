"""Generate an experimental wearable-neighbor forecast for today's cutoff."""

from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path

from cycle_forecast.data.cycle_history import load_cycle_history
from cycle_forecast.data.oura_normalization import normalize_oura_snapshots
from cycle_forecast.data.oura_snapshot import load_snapshot
from cycle_forecast.data.wearable_alignment import align_daily_observation
from cycle_forecast.features.temperature import build_temperature_trajectory_rows
from cycle_forecast.features.wearable import (
    WearableFeatureRow,
    build_wearable_feature_row,
)
from cycle_forecast.forecasting.daily import DailyPeriodDistribution
from cycle_forecast.forecasting.wearable_baselines import (
    STAGE_AWARE_TEMPERATURE_BLEND_VERSION,
    WEARABLE_NEIGHBOR_BASELINE_VERSION,
    blend_history_with_stage_aware_temperature,
    forecast_with_empirical_cycle_hazard,
    forecast_with_temperature_trajectory_neighbors,
    forecast_with_wearable_neighbors,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class WearableDailyPrediction:
    """Contain full-wearable and history-temperature shadow forecasts."""

    model_version: str
    training_morning_count: int
    distribution: DailyPeriodDistribution
    temperature_model_version: str
    temperature_distribution: DailyPeriodDistribution


def predict_daily_with_wearable_neighbors(
    *,
    history_path: Path,
    snapshot_directory: Path,
    prediction_date: date,
    prediction_cutoff: datetime,
    neighbor_count: int = 20,
) -> WearableDailyPrediction:
    """Fit the neighbor baseline on resolved mornings and predict today."""
    if prediction_cutoff.tzinfo is None or prediction_cutoff.utcoffset() is None:
        raise ValueError("prediction_cutoff must be timezone-aware")
    snapshots = tuple(
        load_snapshot(path=path) for path in sorted(snapshot_directory.glob("*.json"))
    )
    if not snapshots:
        raise ValueError("wearable forecast requires Oura snapshots")
    history = load_cycle_history(path=history_path)
    observations = normalize_oura_snapshots(
        snapshots=snapshots, cutoff=prediction_cutoff
    )
    next_starts = {
        current.cycle_start_date: following.cycle_start_date
        for current, following in pairwise(history)
    }
    training_rows: list[WearableFeatureRow] = []
    context_rows: list[WearableFeatureRow] = []
    for observation in observations:
        if observation.day >= prediction_date:
            continue
        aligned = align_daily_observation(
            prediction_date=observation.day,
            prediction_cutoff=observation.available_at,
            cycle_history=history,
            oura_observations=(observation,),
        )
        next_start = next_starts.get(aligned.cycle_start_date)
        feature_row = build_wearable_feature_row(
            aligned=aligned,
            next_cycle_start=(
                next_start
                if next_start is not None and next_start <= prediction_date
                else None
            ),
            observed_through=prediction_date,
        )
        context_rows.append(feature_row)
        if next_start is not None and next_start <= prediction_date:
            training_rows.append(feature_row)
    current = align_daily_observation(
        prediction_date=prediction_date,
        prediction_cutoff=prediction_cutoff,
        cycle_history=history,
        oura_observations=observations,
    )
    current_row = build_wearable_feature_row(
        aligned=current,
        next_cycle_start=None,
        observed_through=prediction_date,
    )
    context_rows.append(current_row)
    training = tuple(training_rows)
    distribution = forecast_with_wearable_neighbors(
        row=current_row,
        training_rows=training,
        neighbor_count=neighbor_count,
    )
    trajectories = build_temperature_trajectory_rows(rows=tuple(context_rows))
    trajectories_by_date = {row.aligned.prediction_date: row for row in trajectories}
    training_trajectories = tuple(
        trajectories_by_date[row.aligned.prediction_date] for row in training
    )
    current_trajectory = trajectories_by_date[current_row.aligned.prediction_date]
    temperature_trajectory = forecast_with_temperature_trajectory_neighbors(
        row=current_trajectory,
        training_rows=training_trajectories,
        neighbor_count=neighbor_count,
    )
    completed_cycle_lengths = tuple(
        (following.cycle_start_date - prior.cycle_start_date).days
        for prior, following in pairwise(history)
        if following.cycle_start_date <= current_row.aligned.cycle_start_date
    )
    history_distribution = forecast_with_empirical_cycle_hazard(
        row=current_row,
        completed_cycle_lengths=completed_cycle_lengths,
    )
    temperature_distribution = blend_history_with_stage_aware_temperature(
        history=history_distribution,
        temperature_trajectory=temperature_trajectory,
        cycle_day=current_row.aligned.cycle_day,
    )
    return WearableDailyPrediction(
        model_version=WEARABLE_NEIGHBOR_BASELINE_VERSION,
        training_morning_count=len(training_rows),
        distribution=distribution,
        temperature_model_version=STAGE_AWARE_TEMPERATURE_BLEND_VERSION,
        temperature_distribution=temperature_distribution,
    )
