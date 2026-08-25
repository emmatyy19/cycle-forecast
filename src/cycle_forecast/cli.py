"""Friendly interactive and scriptable command-line interface."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from enum import StrEnum, auto
from pathlib import Path
from typing import TextIO

from cycle_forecast import __version__
from cycle_forecast.prediction import LocalPrediction, predict_from_local_files
from cycle_forecast.training import (
    LocalTrainingResult,
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


class OutputFormat(StrEnum):
    """Identify supported prediction output formats."""

    HUMAN = auto()
    JSON = auto()


class Command(StrEnum):
    """Identify scriptable CLI subcommands."""

    PREDICT = auto()
    TRAIN = auto()


class InteractiveAction(StrEnum):
    """Identify actions in the bare-command menu."""

    PREDICT = auto()
    TRAIN = auto()
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
    return parser


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
    print("  [3] Exit", file=output)
    choices = {
        "1": InteractiveAction.PREDICT,
        "2": InteractiveAction.TRAIN,
        "3": InteractiveAction.EXIT,
    }
    while True:
        answer = input_fn("\nSelect: ").strip()
        action = choices.get(answer)
        if action is not None:
            return action
        print("Please choose 1, 2, or 3.", file=output)


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
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        operation = (
            "train a model"
            if arguments.command == Command.TRAIN
            else "make a prediction"
        )
        print(f"Could not {operation}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
