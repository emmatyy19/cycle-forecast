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
from cycle_forecast.prediction import LocalPrediction, predict_from_local_files
from cycle_forecast.training import (
    LocalTrainingResult,
    WearableEvaluationMode,
    WearableEvaluationResult,
    evaluate_local_wearable_models,
    load_model_package,
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
    TRAIN = auto()
    OURA_AUTHORIZE = "oura-authorize"
    OURA_SYNC = "oura-sync"
    OURA_SETUP = "oura-setup"
    OURA_STATUS = "oura-status"
    WEARABLE_EVALUATE = "wearable-evaluate"


class InteractiveAction(StrEnum):
    """Identify actions in the bare-command menu."""

    PREDICT = auto()
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
    predict.add_argument("--model", type=Path, help="model package JSON path")
    predict.add_argument("--history", type=Path, help="cycle-history CSV path")
    predict.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of a friendly summary",
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
    print("  [1] Make a prediction", file=output)
    print("  [2] Train or update a model", file=output)
    print("  [3] Evaluate wearable models", file=output)
    print("  [4] Exit", file=output)
    choices = {
        "1": InteractiveAction.PREDICT,
        "2": InteractiveAction.TRAIN,
        "3": InteractiveAction.WEARABLE_EVALUATE,
        "4": InteractiveAction.EXIT,
    }
    while True:
        answer = input_fn("\nSelect: ").strip()
        action = choices.get(answer)
        if action is not None:
            return action
        print("Please choose 1, 2, 3, or 4.", file=output)


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

    print("\nTEMPORAL SPLIT", file=output)
    print(f"  {'Purpose':<16}{'Cycles':>8}{'Mornings':>12}", file=output)
    print(f"  {'─' * 36}", file=output)
    print(
        f"  {'Train':<16}{result.training_cycle_count:>8}"
        f"{result.training_row_count:>12}",
        file=output,
    )
    print(
        f"  {'Calibrate':<16}{result.calibration_cycle_count:>8}"
        f"{result.calibration_row_count:>12}",
        file=output,
    )
    print(
        f"  {'Evaluate':<16}{result.evaluation_cycle_count:>8}"
        f"{result.evaluation_row_count:>12}",
        file=output,
    )

    entries = result.comparison.entries
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
    for entry in result.comparison.entries:
        metrics = entry.evaluation
        print(
            f"  {labels[entry.label]:<24}"
            f"{metrics.logarithmic_loss:>12.3f}"
            f"{metrics.multiclass_brier_score:>12.3f}",
            file=output,
        )
    best_log_loss = min(entries, key=lambda entry: entry.evaluation.logarithmic_loss)
    best_brier = min(entries, key=lambda entry: entry.evaluation.multiclass_brier_score)
    print(f"\n  Best log loss: {labels[best_log_loss.label]}", file=output)
    print(f"  Best Brier:    {labels[best_brier.label]}", file=output)

    print("\nPLANNING-WINDOW BRIER  (lower is better)", file=output)
    print("  Windows include today. Near-zero values may round to 0.000.", file=output)
    print(
        f"  {'Method':<24}{'Today':>10}{'3 days':>10}{'7 days':>10}{'14 days':>10}",
        file=output,
    )
    print(f"  {'─' * 64}", file=output)
    for entry in entries:
        scores = entry.evaluation.window_brier_scores
        print(
            f"  {labels[entry.label]:<24}"
            f"{scores[1]:>10.3f}{scores[3]:>10.3f}"
            f"{scores[7]:>10.3f}{scores[14]:>10.3f}",
            file=output,
        )
    print(
        "\n  Treat this as preliminary: evaluation mornings come from "
        f"{result.evaluation_cycle_count} held-out cycle(s).",
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
        elif arguments.command == Command.TRAIN:
            _run_training(
                history_path=arguments.history,
                configuration_path=arguments.config,
                output_directory=arguments.output_dir,
                replace=arguments.replace,
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
