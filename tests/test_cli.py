"""Test friendly interactive and scriptable command-line prediction."""

import json
from pathlib import Path

import pytest

from cycle_forecast.cli import main
from tests.test_prediction import write_test_model


def test_predict_command_prints_machine_readable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Support repeatable local scripts without interactive prompts."""
    history_path = Path("data/synthetic/sample_cycle_history.csv").resolve()
    model_path = tmp_path / "model.json"
    write_test_model(path=model_path, history_path=history_path)

    status = main(
        (
            "predict",
            "--model",
            str(model_path),
            "--history",
            str(history_path),
            "--json",
        )
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert payload["predicted_next_cycle_start_date"] == "2025-03-08"


def test_bare_command_lists_and_selects_discovered_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guide a user by number so exact paths and flags need not be remembered."""
    history_directory = tmp_path / "data/raw"
    model_directory = tmp_path / "artifacts"
    history_directory.mkdir(parents=True)
    model_directory.mkdir()
    history_path = history_directory / "my-history.csv"
    history_path.write_text(
        Path("data/synthetic/sample_cycle_history.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    model_path = model_directory / "selected-model.json"
    write_test_model(path=model_path, history_path=history_path)
    (model_directory / "training-run.json").write_text(
        '{"schema_version": "not-a-model"}\n',
        encoding="utf-8",
    )
    answers = iter(("1", "1", "1"))

    def answer_prompt(_: str) -> str:
        """Return the next simulated numbered selection."""
        return next(answers)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", answer_prompt)

    status = main(())

    captured = capsys.readouterr()
    assert status == 0
    assert "[1] artifacts/selected-model.json" in captured.out
    assert "training-run.json" not in captured.out
    assert "[1] data/raw/my-history.csv" in captured.out
    assert "Next period start    Saturday, March 8, 2025" in captured.out
    assert "not for diagnosis" in captured.out


def test_bare_command_can_train_a_discovered_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Create both artifacts through the guided menu without command flags."""
    history_directory = tmp_path / "data/raw"
    configuration_directory = tmp_path / "configs"
    history_directory.mkdir(parents=True)
    configuration_directory.mkdir()
    (history_directory / "my-history.csv").write_text(
        Path("data/synthetic/sample_cycle_history.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (configuration_directory / "phase_a.toml").write_text(
        Path("configs/phase_a.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    answers = iter(("2", "", "n"))

    def answer_prompt(_: str) -> str:
        """Return the next simulated training selection."""
        return next(answers)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", answer_prompt)

    status = main(())

    captured = capsys.readouterr()
    assert status == 0
    assert (tmp_path / "artifacts/selected-model.json").is_file()
    assert (tmp_path / "artifacts/training-run.json").is_file()
    assert "✓ MODEL READY" in captured.out
    assert "Development MAE" in captured.out


def test_prediction_error_is_concise_and_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Turn local file failures into an actionable message without a traceback."""
    status = main(
        (
            "predict",
            "--model",
            str(tmp_path / "missing.json"),
            "--history",
            str(tmp_path / "missing.csv"),
        )
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "Could not make a prediction" in captured.err
    assert "Traceback" not in captured.err
