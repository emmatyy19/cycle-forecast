"""Generate an experimental wearable-neighbor forecast for today's cutoff."""

from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise
from pathlib import Path

from cycle_forecast.data.cycle_history import load_cycle_history
from cycle_forecast.data.oura_normalization import normalize_oura_snapshots
from cycle_forecast.data.oura_snapshot import load_snapshot
from cycle_forecast.data.wearable_alignment import align_daily_observation
from cycle_forecast.features.wearable import (
    WearableFeatureRow,
    build_wearable_feature_row,
)
from cycle_forecast.forecasting.daily import DailyPeriodDistribution
from cycle_forecast.forecasting.wearable_baselines import (
    HISTORY_TEMPERATURE_BLEND_VERSION,
    WEARABLE_NEIGHBOR_BASELINE_VERSION,
    blend_history_with_temperature,
    forecast_with_empirical_cycle_hazard,
    forecast_with_temperature_neighbors,
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
        if next_start is None or next_start > prediction_date:
            continue
        training_rows.append(
            build_wearable_feature_row(
                aligned=aligned,
                next_cycle_start=next_start,
                observed_through=prediction_date,
            )
        )
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
    training = tuple(training_rows)
    distribution = forecast_with_wearable_neighbors(
        row=current_row,
        training_rows=training,
        neighbor_count=neighbor_count,
    )
    temperature_ablation = forecast_with_temperature_neighbors(
        row=current_row,
        training_rows=training,
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
    temperature_distribution = blend_history_with_temperature(
        history=history_distribution,
        temperature=temperature_ablation,
    )
    return WearableDailyPrediction(
        model_version=WEARABLE_NEIGHBOR_BASELINE_VERSION,
        training_morning_count=len(training_rows),
        distribution=distribution,
        temperature_model_version=HISTORY_TEMPERATURE_BLEND_VERSION,
        temperature_distribution=temperature_distribution,
    )
