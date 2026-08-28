"""Friendly interactive and scriptable command-line interface."""

import argparse
import getpass
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date, datetime, timedelta
from enum import StrEnum, auto
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cycle_forecast import __version__
from cycle_forecast.data.cycle_history import load_cycle_history
from cycle_forecast.data.oura_auth import (
    DEFAULT_OURA_TOKEN_PATH,
    OuraAuthorizationError,
    authorize_interactively,
    save_oauth_application,
)
from cycle_forecast.data.oura_client import OuraApiError
from cycle_forecast.data.oura_status import inspect_oura_status
from cycle_forecast.data.oura_sync import (
    DEFAULT_OURA_SNAPSHOT_DIRECTORY,
    resolve_sync_start_date,
    sync_oura,
)
from cycle_forecast.data.period_recording import (
    PeriodRecordingResult,
    record_period_start,
)
from cycle_forecast.evaluation.prospective_journal import (
    DEFAULT_PROSPECTIVE_JOURNAL_PATH,
    ProspectivePerformanceSummary,
    append_prospective_forecast,
    build_prospective_entry,
    load_prospective_journal,
    summarize_prospective_performance,
)
from cycle_forecast.prediction import LocalPrediction, predict_from_local_files
from cycle_forecast.prediction_daily import (
    DailyPointEstimate,
    HistoryDailyPrediction,
    estimate_next_start_from_history,
    predict_daily_from_history,
)
from cycle_forecast.training import (
    DailyModelRefreshResult,
    DailyModelRefreshStatus,
    LocalTrainingResult,
    WearableEvaluationMode,
    WearableEvaluationResult,
    evaluate_local_wearable_models,
    load_model_package,
    refresh_daily_model_if_needed,
    train_from_local_history,
)

DEFAULT_MODEL_DIRECTORIES = (Path("artifacts"), Path("models"))
"""Local ignored directories searched for packaged models."""

DEFAULT_HISTORY_DIRECTORIES = (Path("data/raw"), Path("data/private"))
"""Local ignored directories searched for private cycle-history files."""

DEFAULT_TRAINING_CONFIGURATION = Path("configs/phase_a.toml")
"""Committed Phase A configuration used by convenient local training."""

DEFAULT_ARTIFACT_DIRECTORY = Path("artifacts")
"""Ignored destination for the selected model and run manifest."""

RULE = "─" * 54
"""Consistent visual separator for the terminal interface."""

WEARABLE_RULE = "─" * 78
"""Wider separator for readable wearable comparison tables."""


class OutputFormat(StrEnum):
    """Identify supported prediction output formats."""

    HUMAN = auto()
    JSON = auto()


class Command(StrEnum):
    """Identify scriptable CLI subcommands."""

    PREDICT = auto()
    DAILY = auto()
    PERIOD_RECORD = "period-record"
    TRAIN = auto()
    OURA_AUTHORIZE = "oura-authorize"
    OURA_SYNC = "oura-sync"
    OURA_SETUP = "oura-setup"
    OURA_STATUS = "oura-status"
    WEARABLE_EVALUATE = "wearable-evaluate"


class InteractiveAction(StrEnum):
    """Identify actions in the bare-command menu."""

    PREDICT = auto()
    DAILY = auto()
    PERIOD_RECORD = "period-record"
    TRAIN = auto()
    WEARABLE_EVALUATE = "wearable-evaluate"
    EXIT = auto()


def _parser() -> argparse.ArgumentParser:
    """Build the local command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="cycle-forecast",
        description="Forecast cycle timing locally from private history.",
    )
    subparsers = parser.add_subparsers(dest="command")
    predict = subparsers.add_parser(
        "predict",
        help="predict from a packaged model and cycle-history CSV",
    )
    daily = subparsers.add_parser(
        "daily",
        help="sync Oura, check period history, and forecast today",
    )
    daily.add_argument("--history", type=Path)
    daily.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY / "selected-model.json",
        help="preferred Phase A model (falls back to a naive history median)",
    )
    daily.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIGURATION,
        help="Phase A training configuration used when the model is stale",
    )
    daily.add_argument("--timezone", help="active IANA timezone")
    daily.add_argument("--start-date", type=date.fromisoformat)
    daily.add_argument("--token-path", type=Path, default=DEFAULT_OURA_TOKEN_PATH)
    daily.add_argument(
        "--snapshot-dir", type=Path, default=DEFAULT_OURA_SNAPSHOT_DIRECTORY
    )
    daily.add_argument(
        "--journal",
        type=Path,
        default=DEFAULT_PROSPECTIVE_JOURNAL_PATH,
        help="private prospective forecast journal",
    )
    predict.add_argument("--model", type=Path, help="model package JSON path")
    predict.add_argument("--history", type=Path, help="cycle-history CSV path")
    predict.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of a friendly summary",
    )
    period_record = subparsers.add_parser(
        "period-record",
        help="safely add a period start to private local history",
    )
    period_record.add_argument("--history", type=Path)
    period_record.add_argument("--date", type=date.fromisoformat)
    period_record.add_argument(
        "--period-length",
        type=int,
        help="known duration for this period; omit while it is ongoing",
    )
    period_record.add_argument(
        "--previous-period-length",
        type=int,
        help="complete a prior pending period while adding a new start",
    )
    period_record.add_argument(
        "--yes",
        action="store_true",
        help="save without an interactive confirmation",
    )
    train = subparsers.add_parser(
        "train",
        help="select, fit, and save a model from local cycle history",
    )
    train.add_argument("--history", type=Path, help="cycle-history CSV path")
    train.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIGURATION,
        help="versioned training TOML path (default: configs/phase_a.toml)",
    )
    train.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
        help="artifact destination (default: artifacts)",
    )
    train.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing selected model and run manifest",
    )
    authorize = subparsers.add_parser(
        "oura-authorize", help="authorize local access to one Oura account"
    )
    authorize.add_argument(
        "--redirect-uri",
        default="http://localhost:8765/callback",
        help="redirect URI registered with the Oura OAuth application",
    )
    authorize.add_argument(
        "--token-path",
        type=Path,
        default=DEFAULT_OURA_TOKEN_PATH,
        help="private local OAuth token destination",
    )
    sync = subparsers.add_parser(
        "oura-sync", help="retrieve validated Oura data into private snapshots"
    )
    sync.add_argument("--start-date", type=date.fromisoformat)
    sync.add_argument("--end-date", type=date.fromisoformat)
    sync.add_argument("--timezone", required=True, help="active IANA timezone")
    sync.add_argument("--token-path", type=Path, default=DEFAULT_OURA_TOKEN_PATH)
    sync.add_argument(
        "--snapshot-dir", type=Path, default=DEFAULT_OURA_SNAPSHOT_DIRECTORY
    )
    sync.add_argument(
        "--check-only",
        action="store_true",
        help="validate live retrieval without saving health payloads",
    )
    setup = subparsers.add_parser(
        "oura-setup", help="guide Keychain storage, authorization, and live check"
    )
    setup.add_argument("--timezone", required=True, help="active IANA timezone")
    setup.add_argument(
        "--redirect-uri",
        default="http://localhost:8765/callback",
        help="redirect URI registered with the Oura OAuth application",
    )
    setup.add_argument("--token-path", type=Path, default=DEFAULT_OURA_TOKEN_PATH)
    status = subparsers.add_parser(
        "oura-status", help="show non-sensitive local Oura integration status"
    )
    status.add_argument("--token-path", type=Path, default=DEFAULT_OURA_TOKEN_PATH)
    status.add_argument(
        "--snapshot-dir", type=Path, default=DEFAULT_OURA_SNAPSHOT_DIRECTORY
    )
    wearable = subparsers.add_parser(
        "wearable-evaluate",
        help="compare wearable baselines and survival modeling locally",
    )
    wearable.add_argument("--history", type=Path, required=True)
    wearable.add_argument(
        "--snapshot-dir", type=Path, default=DEFAULT_OURA_SNAPSHOT_DIRECTORY
    )
    wearable.add_argument("--timezone", required=True, help="evaluation IANA timezone")
    wearable.add_argument(
        "--mode",
        type=WearableEvaluationMode,
        choices=tuple(WearableEvaluationMode),
        default=WearableEvaluationMode.PROSPECTIVE,
    )
    wearable.add_argument(
        "--as-of-date",
        type=date.fromisoformat,
        help="last date known to have no later period start (default: today)",
    )
    wearable.add_argument(
        "--prediction-hour",
        type=int,
        default=9,
        help="assumed local hour for exploratory backfill (default: 9)",
    )
    wearable.add_argument(
        "--neighbors",
        type=int,
        default=20,
        help="nearest prior mornings used by wearable baseline (default: 20)",
    )
    wearable.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable local results",
    )
    return parser


def _load_timezone(*, timezone_name: str) -> ZoneInfo:
    """Load one IANA timezone or raise a concise configuration error."""
    try:
        return ZoneInfo(timezone_name)
    except (OSError, ValueError, ZoneInfoNotFoundError) as error:
        raise ValueError("timezone must be an IANA timezone") from error


def _run_oura_setup(
    *, redirect_uri: str, timezone_name: str, token_path: Path, output: TextIO
) -> None:
    """Guide one-time Keychain storage, authorization, and live validation."""
    timezone = _load_timezone(timezone_name=timezone_name)
    client_id = input("Oura client ID: ").strip()
    client_secret = getpass.getpass("Oura client secret: ").strip()
    save_oauth_application(client_id=client_id, client_secret=client_secret)
    authorize_interactively(
        redirect_uri=redirect_uri,
        input_fn=getpass.getpass,
        token_path=token_path,
    )
    end_date = datetime.now(tz=timezone).date()
    start_date = end_date - timedelta(days=1)
    results = sync_oura(
        token_path=token_path,
        snapshot_directory=DEFAULT_OURA_SNAPSHOT_DIRECTORY,
        start_date=start_date,
        end_date=end_date,
        timezone_name=timezone_name,
        save=False,
    )
    print(
        "Oura setup complete; live retrieval validated without saving payloads.",
        file=output,
    )
    for result in results:
        print(
            f"{result.route.value}: {result.document_count} documents "
            f"across {result.page_count} pages",
            file=output,
        )


def _run_oura_status(
    *, token_path: Path, snapshot_directory: Path, output: TextIO
) -> None:
    """Render non-sensitive Oura integration readiness."""
    status = inspect_oura_status(
        token_path=token_path,
        snapshot_directory=snapshot_directory,
    )
    credentials = "ready" if status.application_credentials_available else "missing"
    print(f"Application credentials  {credentials}", file=output)
    print(f"Authorization token     {status.token_state}", file=output)
    expiry = status.token_expires_at.isoformat() if status.token_expires_at else "—"
    print(f"Token expiry            {expiry}", file=output)
    print(f"Private snapshots       {status.snapshot_count}", file=output)
    latest = status.latest_snapshot_end_date or "—"
    print(f"Latest snapshot date    {latest}", file=output)


def _discover_files(*, directories: Sequence[Path], pattern: str) -> tuple[Path, ...]:
    """Find matching local files in stable path order."""
    return tuple(
        sorted(
            path
            for directory in directories
            if directory.is_dir()
            for path in directory.rglob(pattern)
            if path.is_file()
        )
    )


def _discover_model_packages(*, directories: Sequence[Path]) -> tuple[Path, ...]:
    """Find valid packaged models while ignoring other artifact JSON files."""
    candidates = _discover_files(directories=directories, pattern="*.json")
    valid: list[Path] = []
    for path in candidates:
        try:
            load_model_package(path=path)
        except (OSError, ValueError):
            continue
        valid.append(path)
    return tuple(valid)


def _choose_path(
    *,
    label: str,
    candidates: Sequence[Path],
    input_fn: Callable[[str], str],
    output: TextIO,
) -> Path:
    """Let a user select a discovered file or type a different path."""
    candidate_paths = tuple(candidates)
    print(f"\n{label}", file=output)
    if candidate_paths:
        for number, path in enumerate(candidate_paths, start=1):
            print(f"  [{number}] {path}", file=output)
        print("\nEnter a number, or paste a different path.", file=output)
    else:
        print("  No likely files found. Paste a file path to continue.", file=output)

    while True:
        prompt = "Select [1]: " if len(candidate_paths) == 1 else "Select: "
        answer = input_fn(prompt).strip()
        if not answer and len(candidate_paths) == 1:
            return candidate_paths[0]
        if answer.isdecimal():
            position = int(answer) - 1
            if 0 <= position < len(candidate_paths):
                return candidate_paths[position]
        elif answer:
            return Path(answer).expanduser()
        print("Please choose a listed number or enter a file path.", file=output)


def _choose_action(
    *, input_fn: Callable[[str], str], output: TextIO
) -> InteractiveAction:
    """Prompt for the main local workflow without requiring command knowledge."""
    print("\nCYCLE FORECAST", file=output)
    print(RULE, file=output)
    print("Private, local cycle planning", file=output)
    print("Your health data never leaves this computer.\n", file=output)
    print("What would you like to do?", file=output)
    print("  [1] Daily check-in", file=output)
    print("  [2] Record a period start", file=output)
    print("  [3] Make a packaged-model prediction", file=output)
    print("  [4] Train or update a model", file=output)
    print("  [5] Evaluate wearable models", file=output)
    print("  [6] Exit", file=output)
    choices = {
        "1": InteractiveAction.DAILY,
        "2": InteractiveAction.PERIOD_RECORD,
        "3": InteractiveAction.PREDICT,
        "4": InteractiveAction.TRAIN,
        "5": InteractiveAction.WEARABLE_EVALUATE,
        "6": InteractiveAction.EXIT,
    }
    while True:
        answer = input_fn("\nSelect: ").strip()
        action = choices.get(answer)
        if action is not None:
            return action
        print("Please choose 1, 2, 3, 4, 5, or 6.", file=output)


def _choose_wearable_mode(
    *, input_fn: Callable[[str], str], output: TextIO
) -> WearableEvaluationMode:
    """Explain and select a local wearable availability assumption."""
    print("\nEVALUATION MODE", file=output)
    print(
        "  [1] Prospective — strict; uses only real morning retrieval cutoffs",
        file=output,
    )
    print(
        "  [2] Exploratory backfill — test historical data with an optimistic "
        "availability assumption",
        file=output,
    )
    while True:
        answer = input_fn("Select [1]: ").strip() or "1"
        if answer == "1":
            return WearableEvaluationMode.PROSPECTIVE
        if answer == "2":
            return WearableEvaluationMode.EXPLORATORY_BACKFILL
        print("Please choose 1 or 2.", file=output)


def _choose_timezone(*, input_fn: Callable[[str], str], output: TextIO) -> str:
    """Prompt until the user supplies a valid IANA timezone name."""
    while True:
        timezone_name = input_fn(
            "IANA timezone (for example America/Los_Angeles): "
        ).strip()
        try:
            _load_timezone(timezone_name=timezone_name)
        except ValueError as error:
            print(str(error), file=output)
            continue
        return timezone_name


def _prompt_date(
    *, label: str, default: date, input_fn: Callable[[str], str], output: TextIO
) -> date:
    """Prompt until a calendar date is entered in ISO format."""
    while True:
        raw_value = input_fn(f"{label} [{default.isoformat()}]: ").strip()
        if not raw_value:
            return default
        try:
            return date.fromisoformat(raw_value)
        except ValueError:
            print("Please enter a date as YYYY-MM-DD.", file=output)


def _prompt_positive_int(
    *, label: str, input_fn: Callable[[str], str], output: TextIO
) -> int:
    """Prompt until a positive whole number is entered."""
    while True:
        raw_value = input_fn(f"{label}: ").strip()
        if raw_value.isdecimal() and int(raw_value) > 0:
            return int(raw_value)
        print("Please enter a positive whole number.", file=output)


def _render_period_recording_result(
    *, result: PeriodRecordingResult, output: TextIO
) -> None:
    """Confirm a private history update in friendly language."""
    print("\n✓ PERIOD HISTORY UPDATED", file=output)
    print(RULE, file=output)
    print(f"Period start       {result.cycle_start_date.isoformat()}", file=output)
    if result.completed_previous_cycle_days is not None:
        print(
            f"Completed cycle    {result.completed_previous_cycle_days} days",
            file=output,
        )
    if result.period_length_days is None:
        print("Period duration    ongoing · add it later", file=output)
    else:
        print(f"Period duration    {result.period_length_days} days", file=output)
    print(f"History records    {result.record_count}", file=output)
    print(f"Saved privately    {result.history_path}", file=output)


def _run_period_recording(
    *,
    history_path: Path | None,
    cycle_start_date: date | None,
    period_length_days: int | None,
    previous_period_length_days: int | None,
    recorded_on: date,
    assume_yes: bool,
    input_fn: Callable[[str], str],
    output: TextIO,
) -> PeriodRecordingResult | None:
    """Guide one safe new-start or period-duration history update."""
    print("\nRECORD A PERIOD", file=output)
    print(RULE, file=output)
    print("This stays in one private file on this computer.", file=output)
    resolved_history = history_path or _choose_path(
        label="CYCLE HISTORY",
        candidates=_discover_files(
            directories=DEFAULT_HISTORY_DIRECTORIES,
            pattern="*.csv",
        ),
        input_fn=input_fn,
        output=output,
    )
    resolved_date = cycle_start_date or _prompt_date(
        label="First day of this period",
        default=recorded_on,
        input_fn=input_fn,
        output=output,
    )

    existing = (
        load_cycle_history(path=resolved_history) if resolved_history.exists() else ()
    )
    resolved_period_length = period_length_days
    resolved_previous_length = previous_period_length_days
    if existing:
        latest = existing[-1]
        if (
            resolved_date == latest.cycle_start_date
            and latest.period_length_days is None
            and resolved_period_length is None
            and not assume_yes
        ):
            resolved_period_length = _prompt_positive_int(
                label="How many days did this period last?",
                input_fn=input_fn,
                output=output,
            )
        elif (
            resolved_date > latest.cycle_start_date
            and latest.period_length_days is None
            and resolved_previous_length is None
            and not assume_yes
        ):
            resolved_previous_length = _prompt_positive_int(
                label="How many days did your previous period last?",
                input_fn=input_fn,
                output=output,
            )

    print("\nPLEASE CONFIRM", file=output)
    print(f"  History file       {resolved_history}", file=output)
    print(f"  Period start       {resolved_date.isoformat()}", file=output)
    if resolved_previous_length is not None:
        print(f"  Previous duration  {resolved_previous_length} days", file=output)
    if resolved_period_length is None:
        print("  Current duration   ongoing / not known yet", file=output)
    else:
        print(f"  Current duration   {resolved_period_length} days", file=output)
    if not assume_yes:
        answer = input_fn("Save this? [Y/n] ").strip().lower()
        if answer in {"n", "no"}:
            print("Nothing was changed.", file=output)
            return None

    result = record_period_start(
        history_path=resolved_history,
        cycle_start_date=resolved_date,
        recorded_on=recorded_on,
        period_length_days=resolved_period_length,
        previous_period_length_days=resolved_previous_length,
    )
    _render_period_recording_result(result=result, output=output)
    return result


def _render_daily_prediction(
    *,
    prediction: HistoryDailyPrediction,
    point_estimate: DailyPointEstimate,
    output: TextIO,
) -> None:
    """Print today's baseline-first probability forecast for planning."""
    distribution = prediction.distribution
    daily_probabilities = distribution.daily_probabilities
    most_likely_offset = max(
        range(len(daily_probabilities)),
        key=daily_probabilities.__getitem__,
    )
    most_likely_daily_probability = daily_probabilities[most_likely_offset]
    print("\nTODAY'S FORECAST", file=output)
    print(RULE, file=output)
    friendly_start = prediction.current_cycle_start_date.strftime("%B %d, %Y").replace(
        " 0", " "
    )
    print("\nCURRENT CYCLE", file=output)
    print(f"  {'Period start':<24}{friendly_start}", file=output)
    print(f"  {'Cycle day':<24}{prediction.cycle_day}", file=output)

    print("\nSHORT-RANGE PROBABILITIES", file=output)
    print(f"  {'Window':<24}{'Chance':>8}", file=output)
    print(f"  {'─' * 32}", file=output)
    probability_rows = (
        ("Today", daily_probabilities[0]),
        ("Within 3 days", distribution.probability_within(days=3)),
        ("Within 7 days", distribution.probability_within(days=7)),
        ("Within 14 days", distribution.probability_within(days=14)),
    )
    for label, probability in probability_rows:
        print(f"  {label:<24}{probability:>8.1%}", file=output)
    if distribution.after_horizon_probability > most_likely_daily_probability:
        horizon_end = prediction.prediction_date + timedelta(days=14)
        likely_outcome = f"After {horizon_end.strftime('%B %d, %Y')}"
    else:
        likely_date = prediction.prediction_date + timedelta(days=most_likely_offset)
        likely_outcome = likely_date.strftime("%A, %B %d, %Y").replace(" 0", " ")
    print("\n15-DAY OUTLOOK", file=output)
    print(f"  {'Most likely result':<24}{likely_outcome}", file=output)
    friendly_estimate = point_estimate.predicted_next_cycle_start_date.strftime(
        "%A, %B %d, %Y"
    ).replace(" 0", " ")
    print("\nNEXT PERIOD ESTIMATE", file=output)
    print(f"  {'Estimated start':<24}{friendly_estimate}", file=output)
    print(
        f"  {'Estimated cycle length':<24}"
        f"{point_estimate.predicted_cycle_length_days:.1f} days",
        file=output,
    )
    print(f"  {'Estimate source':<24}{point_estimate.source_label}", file=output)
    print(f"  {'Interpretation':<24}Single planning guess", file=output)

    print("\nMODEL STATUS", file=output)
    print(
        f"  {'Probability model':<24}Cycle-history baseline",
        file=output,
    )
    print(f"  {'Model version':<24}{prediction.model_version}", file=output)
    print(
        f"  {'Wearable models':<24}Experimental · not used in this forecast",
        file=output,
    )


def _render_prospective_performance(
    *, summary: ProspectivePerformanceSummary, appended: bool, output: TextIO
) -> None:
    """Render delayed performance without printing private forecast dates."""
    print("\nPROSPECTIVE JOURNAL", file=output)
    saved_status = "Saved" if appended else "Already saved · original kept"
    today_label = "Today's forecast"
    print(f"  {today_label:<24}{saved_status}", file=output)
    print(f"  {'Journal forecasts':<24}{summary.journal_forecast_count}", file=output)
    print(f"  {'Resolved forecasts':<24}{summary.resolved_forecast_count}", file=output)
    print(f"  {'Completed cycles':<24}{summary.completed_cycle_count}", file=output)
    if summary.completed_cycle_count == 0:
        print(
            "  Performance             Waiting for a future period start", file=output
        )
        return
    assert summary.mean_cycle_logarithmic_loss is not None
    assert summary.mean_cycle_brier_score is not None
    assert summary.mean_cycle_point_absolute_error_days is not None
    print(
        f"  {'History log loss':<24}{summary.mean_cycle_logarithmic_loss:.3f}",
        file=output,
    )
    print(
        f"  {'History Brier':<24}{summary.mean_cycle_brier_score:.3f}",
        file=output,
    )
    print(
        f"  {'Point-estimate MAE':<24}"
        f"{summary.mean_cycle_point_absolute_error_days:.2f} days",
        file=output,
    )
    window_scores = summary.mean_cycle_window_brier_scores
    print(
        f"  {'Window Brier 1/3/7/14d':<24}"
        f"{window_scores[1]:.3f} / {window_scores[3]:.3f} / "
        f"{window_scores[7]:.3f} / {window_scores[14]:.3f}",
        file=output,
    )
    print("  Scores give every completed cycle equal weight.", file=output)


def _render_daily_model_refresh(
    *, result: DailyModelRefreshResult, output: TextIO
) -> None:
    """Explain whether today's check-in reused or retrained Phase A."""
    print("\n3. Model update", file=output)
    if result.status is DailyModelRefreshStatus.CURRENT:
        print(
            "   ✓ Existing Phase A model is current; no retraining needed.", file=output
        )
        return
    action = (
        "created" if result.status is DailyModelRefreshStatus.CREATED else "refreshed"
    )
    assert result.training is not None
    print(f"   ✓ Phase A model {action} from updated cycle history.", file=output)
    print(
        f"   Development MAE: "
        f"{result.training.development_mean_absolute_error_days:.2f} days "
        f"across {result.training.development_forecast_count} forecasts.",
        file=output,
    )


def _run_daily(
    *,
    history_path: Path | None,
    model_path: Path,
    configuration_path: Path,
    timezone_name: str | None,
    explicit_start_date: date | None,
    token_path: Path,
    snapshot_directory: Path,
    journal_path: Path,
    today: date | None,
    input_fn: Callable[[str], str],
    output: TextIO,
) -> HistoryDailyPrediction:
    """Synchronize Oura, update period history if needed, and forecast today."""
    print("\nDAILY CYCLE FORECAST", file=output)
    print(WEARABLE_RULE, file=output)
    print("One private check-in: sync, record, predict.", file=output)
    resolved_history = history_path or _choose_path(
        label="CYCLE HISTORY",
        candidates=_discover_files(
            directories=DEFAULT_HISTORY_DIRECTORIES,
            pattern="*.csv",
        ),
        input_fn=input_fn,
        output=output,
    )
    resolved_timezone = timezone_name or _choose_timezone(
        input_fn=input_fn,
        output=output,
    )
    timezone = _load_timezone(timezone_name=resolved_timezone)
    resolved_today = today or datetime.now(tz=timezone).date()
    sync_start = resolve_sync_start_date(
        explicit_start_date=explicit_start_date,
        snapshot_directory=snapshot_directory,
    )
    print(
        f"\n1. Syncing Oura: {sync_start} through {resolved_today}…",
        file=output,
    )
    sync_results = sync_oura(
        token_path=token_path,
        snapshot_directory=snapshot_directory,
        start_date=sync_start,
        end_date=resolved_today,
        timezone_name=resolved_timezone,
        save=True,
    )
    print(
        f"   ✓ Saved {sum(result.document_count for result in sync_results)} "
        "validated records across Oura routes.",
        file=output,
    )

    records = load_cycle_history(path=resolved_history)
    latest_start = records[-1].cycle_start_date
    print("\n2. Period history", file=output)
    print(f"   Latest recorded start: {latest_start.isoformat()}", file=output)
    answer = input_fn("   Did a newer period start? [y/N] ").strip().lower()
    if answer in {"y", "yes"}:
        _run_period_recording(
            history_path=resolved_history,
            cycle_start_date=None,
            period_length_days=None,
            previous_period_length_days=None,
            recorded_on=resolved_today,
            assume_yes=False,
            input_fn=input_fn,
            output=output,
        )
    else:
        print("   ✓ Period history is current.", file=output)

    refresh = refresh_daily_model_if_needed(
        history_path=resolved_history,
        model_path=model_path,
        configuration_path=configuration_path,
        code_version=f"cycle-forecast-{__version__}",
    )
    _render_daily_model_refresh(result=refresh, output=output)

    print("\n4. Producing today's forecast…", file=output)
    prediction = predict_daily_from_history(
        history_path=resolved_history,
        prediction_date=resolved_today,
        timezone_name=resolved_timezone,
    )
    point_estimate = estimate_next_start_from_history(
        history_path=resolved_history,
        model_path=refresh.model_path,
    )
    _render_daily_prediction(
        prediction=prediction,
        point_estimate=point_estimate,
        output=output,
    )
    entry = build_prospective_entry(
        prediction=prediction,
        point_estimate=point_estimate,
        model_dataset_fingerprint=refresh.dataset_fingerprint,
        oura_synced_through=resolved_today,
    )
    appended = append_prospective_forecast(path=journal_path, entry=entry)
    summary = summarize_prospective_performance(
        entries=load_prospective_journal(path=journal_path),
        history=load_cycle_history(path=resolved_history),
    )
    _render_prospective_performance(
        summary=summary,
        appended=appended,
        output=output,
    )
    print("\nFor personal planning only; not medical advice.", file=output)
    return prediction


def _render_human(*, prediction: LocalPrediction, output: TextIO) -> None:
    """Print a concise planning-oriented prediction summary."""
    friendly_date = prediction.predicted_next_cycle_start_date.strftime(
        "%A, %B %d, %Y"
    ).replace(" 0", " ")
    print("\nPREDICTION", file=output)
    print(RULE, file=output)
    print(
        f"Next period start    {friendly_date}",
        file=output,
    )
    print(
        f"Predicted length     {prediction.predicted_cycle_length_days:.1f} days",
        file=output,
    )
    print(
        f"Planning length      {prediction.operational_cycle_length_days} days",
        file=output,
    )
    print(
        f"Current cycle start  {prediction.current_cycle_start_date.isoformat()}",
        file=output,
    )
    print("\nMODEL", file=output)
    print(
        f"{prediction.model_version} · {prediction.code_version}",
        file=output,
    )
    print(f"\nNote: {prediction.disclaimer}", file=output)


def _render_json(*, prediction: LocalPrediction, output: TextIO) -> None:
    """Print a stable JSON representation suitable for local scripts."""
    print(json.dumps(asdict(prediction), default=str, sort_keys=True), file=output)


def _run_prediction(
    *,
    model_path: Path | None,
    history_path: Path | None,
    output_format: OutputFormat,
    input_fn: Callable[[str], str],
    output: TextIO,
    announce: bool = True,
) -> None:
    """Resolve missing paths interactively, predict, and render the result."""
    if announce and output_format is OutputFormat.HUMAN:
        print("\nCYCLE FORECAST", file=output)
        print(RULE, file=output)
        print("Private prediction · data stays on this computer", file=output)
    resolved_model = model_path or _choose_path(
        label="MODEL PACKAGE",
        candidates=_discover_model_packages(directories=DEFAULT_MODEL_DIRECTORIES),
        input_fn=input_fn,
        output=output,
    )
    resolved_history = history_path or _choose_path(
        label="CYCLE HISTORY",
        candidates=_discover_files(
            directories=DEFAULT_HISTORY_DIRECTORIES,
            pattern="*.csv",
        ),
        input_fn=input_fn,
        output=output,
    )
    prediction = predict_from_local_files(
        model_path=resolved_model,
        history_path=resolved_history,
    )
    if output_format is OutputFormat.JSON:
        _render_json(prediction=prediction, output=output)
    else:
        _render_human(prediction=prediction, output=output)


def _render_training_result(*, result: LocalTrainingResult, output: TextIO) -> None:
    """Print saved artifact paths and development selection performance."""
    print("\n✓ MODEL READY", file=output)
    print(RULE, file=output)
    print(f"Selected Ridge alpha  {result.selected_ridge_alpha:g}", file=output)
    print(
        "Development MAE      "
        f"{result.development_mean_absolute_error_days:.2f} days "
        f"across {result.development_forecast_count} forecasts",
        file=output,
    )
    print("\nSAVED LOCALLY", file=output)
    print(f"Model package  {result.model_path}", file=output)
    print(f"Run manifest   {result.run_path}", file=output)


def _render_wearable_evaluation(
    *, result: WearableEvaluationResult, output: TextIO
) -> None:
    """Print privacy-safe data sufficiency and probability scores."""
    print("\nWEARABLE MODEL EVALUATION", file=output)
    print(WEARABLE_RULE, file=output)
    print("\nMODE", file=output)
    print(f"  {result.mode.value}", file=output)
    if result.optimistic_backfill_assumption:
        print(
            "  ⚠ Optimistic historical assumption; these scores are not a "
            "leakage-safe estimate.",
            file=output,
        )
    print("\nDATA USED", file=output)
    print(f"  {'Validated snapshots':<24}{result.snapshot_count:>8}", file=output)
    print(f"  {'Normalized days':<24}{result.normalized_day_count:>8}", file=output)
    print(f"  {'Aligned mornings':<24}{result.aligned_row_count:>8}", file=output)
    print(
        f"  {'Usable labeled mornings':<24}{result.uncensored_row_count:>8}",
        file=output,
    )

    print("\nCYCLE-LEVEL WALK-FORWARD", file=output)
    print(
        f"  {'Eligible completed cycles':<30}"
        f"{result.eligible_completed_cycle_count:>8}",
        file=output,
    )
    print(
        f"  {'Unseen evaluation folds':<30}{result.evaluation_fold_count:>8}",
        file=output,
    )
    print(
        f"  {'Training cycles by fold':<30}"
        f"{result.first_fold_training_cycle_count:>3} → "
        f"{result.final_fold_training_cycle_count}",
        file=output,
    )
    print(f"  {'Calibration cycles per fold':<30}{1:>8}", file=output)
    print(
        f"  {'Total evaluated mornings':<30}{result.evaluation_row_count:>8}",
        file=output,
    )

    entries = result.walk_forward.entries
    if not entries:
        print("\nNo evaluated candidates.", file=output)
        return

    label_aliases = {
        "Empirical cycle hazard": "Cycle history",
        "Wearable nearest neighbors": "Wearable neighbors",
        "Calibrated discrete survival": "Survival model",
    }
    labels = {
        entry.label: label_aliases.get(entry.label, entry.label) for entry in entries
    }
    print("\nEXACT-DATE SCORES  (lower is better)", file=output)
    print(
        "  Log loss strongly penalizes confident mistakes; Brier is squared "
        "probability error.",
        file=output,
    )
    print(f"  {'Method':<24}{'Log loss':>12}{'Brier':>12}", file=output)
    print(f"  {'─' * 48}", file=output)
    for entry in entries:
        print(
            f"  {labels[entry.label]:<24}"
            f"{entry.mean_logarithmic_loss:>12.3f}"
            f"{entry.mean_multiclass_brier_score:>12.3f}",
            file=output,
        )
    best_log_loss = min(entries, key=lambda entry: entry.mean_logarithmic_loss)
    best_brier = min(entries, key=lambda entry: entry.mean_multiclass_brier_score)
    print(f"\n  Best log loss: {labels[best_log_loss.label]}", file=output)
    print(f"  Best Brier:    {labels[best_brier.label]}", file=output)
    print(
        "  Values are means of cycle scores; every cycle receives equal weight.",
        file=output,
    )

    print("\nPLANNING-WINDOW BRIER  (lower is better)", file=output)
    print("  Windows include today. Near-zero values may round to 0.000.", file=output)
    print(
        f"  {'Method':<24}{'Today':>10}{'3 days':>10}{'7 days':>10}{'14 days':>10}",
        file=output,
    )
    print(f"  {'─' * 64}", file=output)
    for entry in entries:
        scores = entry.mean_window_brier_scores
        print(
            f"  {labels[entry.label]:<24}"
            f"{scores[1]:>10.3f}{scores[3]:>10.3f}"
            f"{scores[7]:>10.3f}{scores[14]:>10.3f}",
            file=output,
        )
    print("\nPER-CYCLE EXACT-DATE BRIER", file=output)
    print(
        f"  {'Fold':>4}{'Train':>8}{'Mornings':>11}"
        f"{'History':>10}{'Neighbor':>10}{'Survival':>10}  Winner",
        file=output,
    )
    print(f"  {'─' * 74}", file=output)
    for fold in result.walk_forward.folds:
        fold_entries = fold.comparison.entries
        scores = tuple(
            entry.evaluation.multiclass_brier_score for entry in fold_entries
        )
        winner = min(
            fold_entries,
            key=lambda entry: entry.evaluation.multiclass_brier_score,
        )
        ranking = " > ".join(
            labels[entry.label]
            for entry in sorted(
                fold_entries,
                key=lambda entry: entry.evaluation.multiclass_brier_score,
            )
        )
        print(
            f"  {fold.fold_number:>4}{fold.training_cycle_count:>8}"
            f"{fold.evaluation_row_count:>11}"
            f"{scores[0]:>10.3f}{scores[1]:>10.3f}{scores[2]:>10.3f}  "
            f"{labels[winner.label]}",
            file=output,
        )
        print(f"       Ranking: {ranking}", file=output)
    print("\nCYCLE WINS", file=output)
    print(f"  {'Method':<24}{'Log loss':>12}{'Brier':>12}", file=output)
    for entry in entries:
        print(
            f"  {labels[entry.label]:<24}{entry.log_loss_cycle_wins:>12}"
            f"{entry.brier_cycle_wins:>12}",
            file=output,
        )

    diagnostics = result.diagnostics
    print("\nEVALUATED-DATA DIAGNOSTICS", file=output)
    print("  Missing wearable values among evaluated mornings:", file=output)
    for label, rate in diagnostics.data.missingness_rates.items():
        print(f"  {label:<24}{rate:>8.1%}", file=output)
    print("\n  Observed period-start prevalence:", file=output)
    for window, rate in diagnostics.data.outcome_window_rates.items():
        print(f"  {'Within ' + str(window) + ' day(s)':<24}{rate:>8.1%}", file=output)
    print(
        f"  {'After 14 days':<24}{diagnostics.data.after_horizon_rate:>8.1%}",
        file=output,
    )

    print("\nMODEL BEHAVIOR", file=output)
    print(
        "  Offset uses the distribution's expected day; after 14 days is "
        "represented as day 15.",
        file=output,
    )
    print(
        f"  {'Method':<24}{'Actual p avg':>14}{'Actual p min':>14}"
        f"{'Offset bias':>14}{'Offset RMSE':>14}",
        file=output,
    )
    print(f"  {'─' * 80}", file=output)
    for candidate in diagnostics.candidates:
        print(
            f"  {labels[candidate.label]:<24}"
            f"{candidate.mean_actual_outcome_probability:>14.3f}"
            f"{candidate.minimum_actual_outcome_probability:>14.3f}"
            f"{candidate.mean_signed_offset_error:>+14.2f}"
            f"{candidate.root_mean_squared_offset_error:>14.2f}",
            file=output,
        )

    print("\nPLANNING-WINDOW CALIBRATION", file=output)
    print("  Gap = predicted frequency minus observed frequency.", file=output)
    print(
        f"  {'Method':<24}{'Window':>8}{'Predicted':>12}{'Observed':>12}{'Gap':>10}",
        file=output,
    )
    for candidate in diagnostics.candidates:
        for window, diagnostic in candidate.calibration.items():
            gap = diagnostic.mean_predicted_probability - diagnostic.observed_fraction
            print(
                f"  {labels[candidate.label]:<24}{window:>7}d"
                f"{diagnostic.mean_predicted_probability:>12.1%}"
                f"{diagnostic.observed_fraction:>12.1%}{gap:>+10.1%}",
                file=output,
            )

    print("\nCYCLE-DAY BRIER", file=output)
    print("  Lower is better; counts are evaluated mornings in each band.", file=output)
    print(
        f"  {'Method':<24}{'Cycle days':>14}{'Mornings':>12}{'Brier':>12}",
        file=output,
    )
    for candidate in diagnostics.candidates:
        for diagnostic in candidate.cycle_day:
            print(
                f"  {labels[candidate.label]:<24}{diagnostic.label:>14}"
                f"{diagnostic.count:>12}{diagnostic.mean_brier_score:>12.3f}",
                file=output,
            )
    print(
        "\n  Exploratory results remain preliminary until strict prospective "
        "cycles accumulate.",
        file=output,
    )


def _run_wearable_evaluation(
    *,
    history_path: Path,
    snapshot_directory: Path,
    timezone_name: str,
    mode: WearableEvaluationMode,
    observed_through: date,
    prediction_hour: int,
    neighbor_count: int,
    output_format: OutputFormat,
    output: TextIO,
) -> WearableEvaluationResult:
    """Evaluate local wearable methods and render private-safe results."""
    result = evaluate_local_wearable_models(
        history_path=history_path,
        snapshot_directory=snapshot_directory,
        timezone_name=timezone_name,
        mode=mode,
        observed_through=observed_through,
        prediction_hour=prediction_hour,
        neighbor_count=neighbor_count,
    )
    if output_format is OutputFormat.JSON:
        print(json.dumps(asdict(result), default=str, sort_keys=True), file=output)
    else:
        _render_wearable_evaluation(result=result, output=output)
    return result


def _run_training(
    *,
    history_path: Path | None,
    configuration_path: Path,
    output_directory: Path,
    replace: bool,
    input_fn: Callable[[str], str],
    output: TextIO,
) -> tuple[LocalTrainingResult, Path]:
    """Resolve local history, train the selected model, and show artifacts."""
    resolved_history = history_path or _choose_path(
        label="CYCLE HISTORY",
        candidates=_discover_files(
            directories=DEFAULT_HISTORY_DIRECTORIES,
            pattern="*.csv",
        ),
        input_fn=input_fn,
        output=output,
    )
    print("\nTraining and comparing models…", file=output)
    result = train_from_local_history(
        history_path=resolved_history,
        configuration_path=configuration_path,
        output_directory=output_directory,
        code_version=f"cycle-forecast-{__version__}",
        replace=replace,
    )
    _render_training_result(result=result, output=output)
    return result, resolved_history


def _run_interactive(*, input_fn: Callable[[str], str], output: TextIO) -> None:
    """Run the menu-driven prediction or training journey."""
    action = _choose_action(input_fn=input_fn, output=output)
    if action is InteractiveAction.EXIT:
        print("Goodbye.", file=output)
        return
    if action is InteractiveAction.DAILY:
        _run_daily(
            history_path=None,
            model_path=DEFAULT_ARTIFACT_DIRECTORY / "selected-model.json",
            configuration_path=DEFAULT_TRAINING_CONFIGURATION,
            timezone_name=None,
            explicit_start_date=None,
            token_path=DEFAULT_OURA_TOKEN_PATH,
            snapshot_directory=DEFAULT_OURA_SNAPSHOT_DIRECTORY,
            journal_path=DEFAULT_PROSPECTIVE_JOURNAL_PATH,
            today=None,
            input_fn=input_fn,
            output=output,
        )
        return
    if action is InteractiveAction.PERIOD_RECORD:
        _run_period_recording(
            history_path=None,
            cycle_start_date=None,
            period_length_days=None,
            previous_period_length_days=None,
            recorded_on=date.today(),
            assume_yes=False,
            input_fn=input_fn,
            output=output,
        )
        return
    if action is InteractiveAction.PREDICT:
        _run_prediction(
            model_path=None,
            history_path=None,
            output_format=OutputFormat.HUMAN,
            input_fn=input_fn,
            output=output,
            announce=False,
        )
        return
    if action is InteractiveAction.WEARABLE_EVALUATE:
        history_path = _choose_path(
            label="CYCLE HISTORY",
            candidates=_discover_files(
                directories=DEFAULT_HISTORY_DIRECTORIES,
                pattern="*.csv",
            ),
            input_fn=input_fn,
            output=output,
        )
        mode = _choose_wearable_mode(input_fn=input_fn, output=output)
        timezone_name = _choose_timezone(input_fn=input_fn, output=output)
        timezone = _load_timezone(timezone_name=timezone_name)
        print("\nComparing wearable forecasting methods…", file=output)
        _run_wearable_evaluation(
            history_path=history_path,
            snapshot_directory=DEFAULT_OURA_SNAPSHOT_DIRECTORY,
            timezone_name=timezone_name,
            mode=mode,
            observed_through=datetime.now(tz=timezone).date(),
            prediction_hour=9,
            neighbor_count=20,
            output_format=OutputFormat.HUMAN,
            output=output,
        )
        return

    replace = False
    model_path = DEFAULT_ARTIFACT_DIRECTORY / "selected-model.json"
    run_path = DEFAULT_ARTIFACT_DIRECTORY / "training-run.json"
    if model_path.exists() or run_path.exists():
        answer = input_fn("Replace the existing local model artifacts? [y/N] ")
        replace = answer.strip().lower() in {"y", "yes"}
        if not replace:
            print("Training cancelled; existing artifacts were kept.", file=output)
            return
    result, history_path = _run_training(
        history_path=None,
        configuration_path=DEFAULT_TRAINING_CONFIGURATION,
        output_directory=DEFAULT_ARTIFACT_DIRECTORY,
        replace=replace,
        input_fn=input_fn,
        output=output,
    )
    answer = input_fn("Make a prediction with this model now? [Y/n] ")
    if answer.strip().lower() not in {"n", "no"}:
        _run_prediction(
            model_path=result.model_path,
            history_path=history_path,
            output_format=OutputFormat.HUMAN,
            input_fn=input_fn,
            output=output,
            announce=False,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Cycle Forecast command-line application.

    Parameters
    ----------
    argv
        Arguments excluding the executable name, or ``None`` for process args.

    Returns
    -------
    int
        Process exit status: zero on success and two for a user-facing error.
    """
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command is None:
            _run_interactive(input_fn=input, output=sys.stdout)
        elif arguments.command == Command.PREDICT:
            if arguments.json and (
                arguments.model is None or arguments.history is None
            ):
                message = "--json requires both --model and --history"
                raise ValueError(message)
            _run_prediction(
                model_path=arguments.model,
                history_path=arguments.history,
                output_format=(
                    OutputFormat.JSON if arguments.json else OutputFormat.HUMAN
                ),
                input_fn=input,
                output=sys.stdout,
            )
        elif arguments.command == Command.DAILY:
            _run_daily(
                history_path=arguments.history,
                model_path=arguments.model,
                configuration_path=arguments.config,
                timezone_name=arguments.timezone,
                explicit_start_date=arguments.start_date,
                token_path=arguments.token_path,
                snapshot_directory=arguments.snapshot_dir,
                journal_path=arguments.journal,
                today=None,
                input_fn=input,
                output=sys.stdout,
            )
        elif arguments.command == Command.TRAIN:
            _run_training(
                history_path=arguments.history,
                configuration_path=arguments.config,
                output_directory=arguments.output_dir,
                replace=arguments.replace,
                input_fn=input,
                output=sys.stdout,
            )
        elif arguments.command == Command.PERIOD_RECORD:
            _run_period_recording(
                history_path=arguments.history,
                cycle_start_date=arguments.date,
                period_length_days=arguments.period_length,
                previous_period_length_days=arguments.previous_period_length,
                recorded_on=date.today(),
                assume_yes=arguments.yes,
                input_fn=input,
                output=sys.stdout,
            )
        elif arguments.command == Command.OURA_AUTHORIZE:
            authorize_interactively(
                redirect_uri=arguments.redirect_uri,
                input_fn=getpass.getpass,
                token_path=arguments.token_path,
            )
            print(f"Oura authorization saved privately at {arguments.token_path}")
        elif arguments.command == Command.OURA_SYNC:
            timezone = _load_timezone(timezone_name=arguments.timezone)
            end_date = arguments.end_date or datetime.now(tz=timezone).date()
            start_date = resolve_sync_start_date(
                explicit_start_date=arguments.start_date,
                snapshot_directory=arguments.snapshot_dir,
            )
            results = sync_oura(
                token_path=arguments.token_path,
                snapshot_directory=arguments.snapshot_dir,
                start_date=start_date,
                end_date=end_date,
                timezone_name=arguments.timezone,
                save=not arguments.check_only,
            )
            action = "validated" if arguments.check_only else "saved privately"
            print(f"Oura retrieval {action}: {start_date} through {end_date}")
            for result in results:
                print(
                    f"{result.route.value}: {result.document_count} documents "
                    f"across {result.page_count} pages"
                )
        elif arguments.command == Command.OURA_SETUP:
            _run_oura_setup(
                redirect_uri=arguments.redirect_uri,
                timezone_name=arguments.timezone,
                token_path=arguments.token_path,
                output=sys.stdout,
            )
        elif arguments.command == Command.OURA_STATUS:
            _run_oura_status(
                token_path=arguments.token_path,
                snapshot_directory=arguments.snapshot_dir,
                output=sys.stdout,
            )
        elif arguments.command == Command.WEARABLE_EVALUATE:
            timezone = _load_timezone(timezone_name=arguments.timezone)
            _run_wearable_evaluation(
                history_path=arguments.history,
                snapshot_directory=arguments.snapshot_dir,
                timezone_name=arguments.timezone,
                mode=arguments.mode,
                observed_through=(
                    arguments.as_of_date or datetime.now(tz=timezone).date()
                ),
                prediction_hour=arguments.prediction_hour,
                neighbor_count=arguments.neighbors,
                output_format=(
                    OutputFormat.JSON if arguments.json else OutputFormat.HUMAN
                ),
                output=sys.stdout,
            )
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        return 2
    except (OSError, ValueError, OuraApiError, OuraAuthorizationError) as error:
        if arguments.command == Command.TRAIN:
            operation = "train a model"
        elif arguments.command == Command.DAILY:
            operation = "run the daily forecast"
        elif arguments.command == Command.PERIOD_RECORD:
            operation = "record a period"
        elif arguments.command == Command.PREDICT:
            operation = "make a prediction"
        elif arguments.command == Command.OURA_AUTHORIZE:
            operation = "authorize Oura"
        elif arguments.command == Command.OURA_SYNC:
            operation = "sync Oura data"
        elif arguments.command == Command.OURA_SETUP:
            operation = "set up Oura"
        elif arguments.command == Command.OURA_STATUS:
            operation = "inspect Oura status"
        elif arguments.command == Command.WEARABLE_EVALUATE:
            operation = "evaluate wearable models"
        else:
            operation = "run the command"
        print(f"Could not {operation}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
