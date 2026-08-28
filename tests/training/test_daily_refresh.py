"""Tests for freshness-based daily Phase A model updates."""

from pathlib import Path

import pytest

import cycle_forecast.training.daily_refresh as daily_refresh
from cycle_forecast.training import (
    DailyModelRefreshStatus,
    refresh_daily_model_if_needed,
)


def _copy_history(*, destination: Path) -> None:
    """Copy the invented synthetic history into a writable test path."""
    destination.write_text(
        Path("data/synthetic/sample_cycle_history.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def test_create_then_reuse_current_daily_model(tmp_path: Path) -> None:
    """Train once for a fingerprint and skip work on an unchanged history."""
    history_path = tmp_path / "history.csv"
    model_path = tmp_path / "artifacts/selected-model.json"
    _copy_history(destination=history_path)

    created = refresh_daily_model_if_needed(
        history_path=history_path,
        model_path=model_path,
        configuration_path=Path("configs/phase_a.toml"),
        code_version="cycle-forecast-test",
    )
    current = refresh_daily_model_if_needed(
        history_path=history_path,
        model_path=model_path,
        configuration_path=Path("configs/phase_a.toml"),
        code_version="cycle-forecast-test",
    )

    assert created.status is DailyModelRefreshStatus.CREATED
    assert created.training is not None
    assert current.status is DailyModelRefreshStatus.CURRENT
    assert current.training is None
    assert current.dataset_fingerprint == created.dataset_fingerprint


def test_refresh_model_after_new_completed_cycle(tmp_path: Path) -> None:
    """Replace a stale package after history gains another period start."""
    history_path = tmp_path / "history.csv"
    model_path = tmp_path / "artifacts/selected-model.json"
    _copy_history(destination=history_path)
    initial = refresh_daily_model_if_needed(
        history_path=history_path,
        model_path=model_path,
        configuration_path=Path("configs/phase_a.toml"),
        code_version="cycle-forecast-test",
    )
    with history_path.open(mode="a", encoding="utf-8") as history:
        history.write("2025-03-10,\n")

    refreshed = refresh_daily_model_if_needed(
        history_path=history_path,
        model_path=model_path,
        configuration_path=Path("configs/phase_a.toml"),
        code_version="cycle-forecast-test-2",
    )

    assert refreshed.status is DailyModelRefreshStatus.REFRESHED
    assert refreshed.training is not None
    assert refreshed.dataset_fingerprint != initial.dataset_fingerprint


def test_refresh_current_model_when_manifest_is_missing(tmp_path: Path) -> None:
    """Restore the model and run manifest as one complete artifact set."""
    history_path = tmp_path / "history.csv"
    model_path = tmp_path / "artifacts/selected-model.json"
    _copy_history(destination=history_path)
    created = refresh_daily_model_if_needed(
        history_path=history_path,
        model_path=model_path,
        configuration_path=Path("configs/phase_a.toml"),
        code_version="cycle-forecast-test",
    )
    created.run_path.unlink()

    refreshed = refresh_daily_model_if_needed(
        history_path=history_path,
        model_path=model_path,
        configuration_path=Path("configs/phase_a.toml"),
        code_version="cycle-forecast-test",
    )

    assert refreshed.status is DailyModelRefreshStatus.REFRESHED
    assert refreshed.run_path.is_file()


def test_failed_refresh_preserves_existing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the current package and manifest when replacement training fails."""
    history_path = tmp_path / "history.csv"
    model_path = tmp_path / "artifacts/selected-model.json"
    _copy_history(destination=history_path)
    refresh_daily_model_if_needed(
        history_path=history_path,
        model_path=model_path,
        configuration_path=Path("configs/phase_a.toml"),
        code_version="cycle-forecast-test",
    )
    run_path = model_path.parent / "training-run.json"
    original_model = model_path.read_bytes()
    original_run = run_path.read_bytes()
    with history_path.open(mode="a", encoding="utf-8") as history:
        history.write("2025-03-10,\n")

    def fail_training(**_: object) -> None:
        """Simulate failure before replacement artifacts exist."""
        raise ValueError("simulated training failure")

    monkeypatch.setattr(daily_refresh, "train_from_local_history", fail_training)

    with pytest.raises(ValueError, match="simulated training failure"):
        refresh_daily_model_if_needed(
            history_path=history_path,
            model_path=model_path,
            configuration_path=Path("configs/phase_a.toml"),
            code_version="cycle-forecast-test-2",
        )

    assert model_path.read_bytes() == original_model
    assert run_path.read_bytes() == original_run
