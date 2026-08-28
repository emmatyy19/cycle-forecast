"""Test friendly interactive and scriptable command-line prediction."""

import json
from datetime import UTC, date, datetime, tzinfo
from pathlib import Path

import pytest

import cycle_forecast.cli as cli
from cycle_forecast.cli import main
from cycle_forecast.data.oura_auth import OuraAuthorizationError, OuraToken
from cycle_forecast.data.oura_client import OuraRoute
from cycle_forecast.data.oura_status import OuraStatus
from cycle_forecast.data.oura_sync import OuraRouteSyncResult
from cycle_forecast.evaluation.wearable import (
    DailyCandidateEvaluation,
    DailyModelComparison,
)
from cycle_forecast.forecasting.daily import DailyForecastEvaluation
from cycle_forecast.training import WearableEvaluationMode, WearableEvaluationResult
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


def test_first_oura_sync_requires_an_explicit_start_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Give a concise historical-import instruction before reading a token."""
    status = main(
        (
            "oura-sync",
            "--timezone",
            "America/New_York",
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
        )
    )

    captured = capsys.readouterr()
    assert status == 2
    assert "Could not sync Oura data" in captured.err
    assert "requires --start-date" in captured.err
    assert "Traceback" not in captured.err


def test_oura_authorize_command_reports_only_private_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise the OAuth CLI without printing synthetic token values."""
    token_path = tmp_path / "token.json"
    token = OuraToken(
        access_token="do-not-print-access",
        refresh_token="do-not-print-refresh",
        expires_at=datetime(2030, 1, 1, tzinfo=UTC),
    )

    authorization_arguments: dict[str, object] = {}

    def authorize(**arguments: object) -> OuraToken:
        """Return the invented token without opening a browser."""
        authorization_arguments.update(arguments)
        return token

    monkeypatch.setattr(cli, "authorize_interactively", authorize)

    status = main(("oura-authorize", "--token-path", str(token_path)))

    captured = capsys.readouterr()
    assert status == 0
    assert str(token_path) in captured.out
    assert token.access_token not in captured.out
    assert token.refresh_token not in captured.out
    assert authorization_arguments["input_fn"] is cli.getpass.getpass


def test_oura_check_only_command_reports_counts_without_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Render privacy-safe live validation metadata."""

    def sync(**_: object) -> tuple[OuraRouteSyncResult, ...]:
        """Return privacy-safe invented retrieval metadata."""
        return (
            OuraRouteSyncResult(
                route=OuraRoute.DAILY_READINESS,
                page_count=1,
                document_count=2,
                snapshot=None,
            ),
        )

    monkeypatch.setattr(cli, "sync_oura", sync)

    status = main(
        (
            "oura-sync",
            "--start-date",
            "2025-01-01",
            "--end-date",
            "2025-01-02",
            "--timezone",
            "America/New_York",
            "--snapshot-dir",
            str(tmp_path),
            "--check-only",
        )
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "validated" in captured.out
    assert "daily_readiness: 2 documents across 1 pages" in captured.out
    assert date(2025, 1, 1).isoformat() in captured.out


def test_oura_sync_default_end_date_uses_requested_timezone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the current calendar day in the explicitly active timezone."""
    captured_arguments: dict[str, object] = {}

    class FixedDateTime(datetime):
        """Return an instant whose local date differs across US timezones."""

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            """Return the fixed instant converted to the requested timezone."""
            instant = datetime(2025, 1, 2, 4, 30, tzinfo=UTC)
            return instant if tz is None else instant.astimezone(tz)

    def sync(**arguments: object) -> tuple[OuraRouteSyncResult, ...]:
        """Capture the resolved sync range without contacting Oura."""
        captured_arguments.update(arguments)
        return ()

    def resolve_start_date(**_: object) -> date:
        """Return an invented prior snapshot boundary."""
        return date(2025, 1, 1)

    monkeypatch.setattr(cli, "datetime", FixedDateTime)
    monkeypatch.setattr(cli, "resolve_sync_start_date", resolve_start_date)
    monkeypatch.setattr(cli, "sync_oura", sync)

    status = main(
        (
            "oura-sync",
            "--timezone",
            "America/New_York",
            "--snapshot-dir",
            str(tmp_path),
            "--check-only",
        )
    )

    assert status == 0
    assert captured_arguments["end_date"] == date(2025, 1, 1)


def test_oura_authorization_error_is_concise_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Turn token schema failures into actionable CLI output."""

    def fail_authorization(**_: object) -> OuraToken:
        """Raise an invented safe authorization error."""
        raise OuraAuthorizationError("invalid token response")

    monkeypatch.setattr(cli, "authorize_interactively", fail_authorization)

    status = main(("oura-authorize",))

    captured = capsys.readouterr()
    assert status == 2
    assert "Could not authorize Oura: invalid token response" in captured.err
    assert "Traceback" not in captured.err


def test_oura_setup_guides_keychain_authorization_and_live_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Complete guided setup without printing prompted credentials."""
    saved: dict[str, str] = {}

    def save_application(*, client_id: str, client_secret: str) -> None:
        """Capture invented credentials at the Keychain boundary."""
        saved["client_id"] = client_id
        saved["client_secret"] = client_secret

    def authorize(**_: object) -> OuraToken:
        """Return an invented authorized token."""
        return OuraToken(
            access_token="synthetic-access",
            refresh_token="synthetic-refresh",
            expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        )

    def sync(**_: object) -> tuple[OuraRouteSyncResult, ...]:
        """Return invented check-only retrieval metadata."""
        return (
            OuraRouteSyncResult(
                route=OuraRoute.DAILY_READINESS,
                page_count=1,
                document_count=2,
                snapshot=None,
            ),
        )

    def client_id_prompt(_: str) -> str:
        """Return an invented client ID."""
        return "synthetic-client"

    def client_secret_prompt(_: str) -> str:
        """Return an invented client secret."""
        return "synthetic-secret"

    monkeypatch.setattr("builtins.input", client_id_prompt)
    monkeypatch.setattr(cli.getpass, "getpass", client_secret_prompt)
    monkeypatch.setattr(cli, "save_oauth_application", save_application)
    monkeypatch.setattr(cli, "authorize_interactively", authorize)
    monkeypatch.setattr(cli, "sync_oura", sync)

    status = main(
        (
            "oura-setup",
            "--timezone",
            "America/Los_Angeles",
            "--token-path",
            str(tmp_path / "token.json"),
        )
    )

    captured = capsys.readouterr()
    assert status == 0
    assert saved == {
        "client_id": "synthetic-client",
        "client_secret": "synthetic-secret",
    }
    assert "setup complete" in captured.out
    assert "synthetic-secret" not in captured.out


def test_oura_status_command_reports_only_non_sensitive_metadata(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Render readiness without showing credentials or health values."""

    def inspect(**_: object) -> OuraStatus:
        """Return invented non-sensitive status metadata."""
        return OuraStatus(
            application_credentials_available=True,
            token_state="valid",
            token_expires_at=datetime(2030, 1, 1, tzinfo=UTC),
            snapshot_count=6,
            latest_snapshot_end_date=date(2026, 8, 25),
        )

    monkeypatch.setattr(cli, "inspect_oura_status", inspect)

    status = main(("oura-status",))

    captured = capsys.readouterr()
    assert status == 0
    assert "Application credentials  ready" in captured.out
    assert "Authorization token     valid" in captured.out
    assert "Private snapshots       6" in captured.out


def test_oura_setup_rejects_non_iana_timezone_before_prompting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject ambiguous timezone input without requesting credentials."""
    status = main(("oura-setup", "--timezone", "Pacific"))

    captured = capsys.readouterr()
    assert status == 2
    assert "timezone must be an IANA timezone" in captured.err


def test_wearable_evaluate_command_renders_private_safe_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose evaluation sufficiency and the optimistic-mode warning."""
    result = WearableEvaluationResult(
        workflow_version="wearable-evaluation-v1",
        mode=WearableEvaluationMode.EXPLORATORY_BACKFILL,
        optimistic_backfill_assumption=True,
        snapshot_count=3,
        normalized_day_count=60,
        aligned_row_count=50,
        uncensored_row_count=45,
        training_cycle_count=2,
        calibration_cycle_count=1,
        evaluation_cycle_count=1,
        training_row_count=25,
        calibration_row_count=10,
        evaluation_row_count=10,
        comparison=DailyModelComparison(
            prediction_dates=(date(2025, 3, 1),),
            entries=(
                DailyCandidateEvaluation(
                    label="Empirical cycle hazard",
                    version="history-v1",
                    evaluation=DailyForecastEvaluation(
                        count=10,
                        logarithmic_loss=0.2,
                        multiclass_brier_score=0.05,
                        window_brier_scores={1: 0.0, 3: 0.01, 7: 0.02, 14: 0.03},
                    ),
                    calibration={},
                ),
                DailyCandidateEvaluation(
                    label="Wearable nearest neighbors",
                    version="neighbors-v1",
                    evaluation=DailyForecastEvaluation(
                        count=10,
                        logarithmic_loss=0.9,
                        multiclass_brier_score=0.3,
                        window_brier_scores={1: 0.01, 3: 0.02, 7: 0.05, 14: 0.2},
                    ),
                    calibration={},
                ),
            ),
        ),
    )

    def evaluate(**_: object) -> WearableEvaluationResult:
        """Return invented privacy-safe wearable evaluation metadata."""
        return result

    monkeypatch.setattr(cli, "evaluate_local_wearable_models", evaluate)

    status = main(
        (
            "wearable-evaluate",
            "--history",
            str(tmp_path / "history.csv"),
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
            "--timezone",
            "America/New_York",
            "--mode",
            "exploratory-backfill",
            "--as-of-date",
            "2025-03-22",
        )
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "Optimistic historical assumption" in captured.out
    assert "TEMPORAL SPLIT" in captured.out
    assert "EXACT-DATE SCORES" in captured.out
    assert "PLANNING-WINDOW BRIER" in captured.out
    assert "Best log loss: Cycle history" in captured.out
    assert "Normalized days" in captured.out
    assert "60" in captured.out
    assert str(tmp_path) not in captured.out


def test_bare_command_guides_wearable_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Select history, availability mode, and timezone without command flags."""
    history_directory = tmp_path / "data/raw"
    history_directory.mkdir(parents=True)
    history_path = history_directory / "history.csv"
    history_path.write_text(
        Path("data/synthetic/sample_cycle_history.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    result = WearableEvaluationResult(
        workflow_version="wearable-evaluation-v1",
        mode=WearableEvaluationMode.EXPLORATORY_BACKFILL,
        optimistic_backfill_assumption=True,
        snapshot_count=3,
        normalized_day_count=60,
        aligned_row_count=50,
        uncensored_row_count=45,
        training_cycle_count=2,
        calibration_cycle_count=1,
        evaluation_cycle_count=1,
        training_row_count=25,
        calibration_row_count=10,
        evaluation_row_count=10,
        comparison=DailyModelComparison(prediction_dates=(), entries=()),
    )
    received: dict[str, object] = {}

    def evaluate(**arguments: object) -> WearableEvaluationResult:
        """Capture guided selections and return invented aggregate results."""
        received.update(arguments)
        return result

    answers = iter(("3", "1", "2", "America/New_York"))

    def answer_prompt(_: str) -> str:
        """Return the next guided wearable selection."""
        return next(answers)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", answer_prompt)
    monkeypatch.setattr(cli, "evaluate_local_wearable_models", evaluate)

    status = main(())

    captured = capsys.readouterr()
    assert status == 0
    assert "[3] Evaluate wearable models" in captured.out
    assert "EVALUATION MODE" in captured.out
    assert "Optimistic historical assumption" in captured.out
    assert received["history_path"] == Path("data/raw/history.csv")
    assert received["mode"] is WearableEvaluationMode.EXPLORATORY_BACKFILL
    assert received["timezone_name"] == "America/New_York"
