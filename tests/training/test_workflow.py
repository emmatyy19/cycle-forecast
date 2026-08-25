"""Test the complete reproducible local training workflow."""

from pathlib import Path

import pytest

from cycle_forecast.training import (
    load_model_package,
    train_from_local_history,
)


def test_train_from_local_history_writes_model_and_manifest(tmp_path: Path) -> None:
    """Turn validated history into both private delivery artifacts."""
    result = train_from_local_history(
        history_path="data/synthetic/sample_cycle_history.csv",
        configuration_path="configs/phase_a.toml",
        output_directory=tmp_path,
        code_version="git:test",
    )

    package = load_model_package(path=result.model_path)
    assert result.model_path == tmp_path / "selected-model.json"
    assert result.run_path == tmp_path / "training-run.json"
    assert result.run_path.is_file()
    assert package.code_version == "git:test"
    assert package.ridge_alpha == result.selected_ridge_alpha
    assert result.development_mean_absolute_error_days >= 0
    assert result.development_forecast_count > 0


def test_train_from_local_history_refuses_to_replace_artifacts(tmp_path: Path) -> None:
    """Protect an existing local model unless replacement is explicit."""
    model_path = tmp_path / "selected-model.json"
    model_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        train_from_local_history(
            history_path="data/synthetic/sample_cycle_history.csv",
            configuration_path="configs/phase_a.toml",
            output_directory=tmp_path,
            code_version="git:test",
        )

    assert model_path.read_text(encoding="utf-8") == "keep me"
