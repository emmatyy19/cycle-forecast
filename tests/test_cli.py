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
from cycle_forecast.evaluation.prospective_journal import ProspectivePerformanceSummary
from cycle_forecast.evaluation.wearable import (
    DailyCandidateEvaluation,
    DailyModelComparison,
)
from cycle_forecast.forecasting.daily import (
    DailyForecastEvaluation,
    DailyPeriodDistribution,
)
from cycle_forecast.prediction_wearable import WearableDailyPrediction
from cycle_forecast.training import (
    DailyModelRefreshResult,
    DailyModelRefreshStatus,
    WearableAggregateEntry,
    WearableCalibrationDiagnostic,
    WearableCandidateDiagnostics,
    WearableCycleDayDiagnostic,
    WearableCycleFoldResult,
    WearableDataDiagnostics,
    WearableEvaluationDiagnostics,
    WearableEvaluationMode,
    WearableEvaluationResult,
    WearableWalkForwardComparison,
)
from tests.test_prediction import write_test_model


def _resolved_prospective_summary(**_: object) -> ProspectivePerformanceSummary:
    """Return invented resolved history and wearable performance."""
    return ProspectivePerformanceSummary(
        journal_forecast_count=20,
        resolved_forecast_count=12,
        completed_cycle_count=2,
        mean_cycle_logarithmic_loss=0.4,
        mean_cycle_brier_score=0.2,
        mean_cycle_window_brier_scores={1: 0.1, 3: 0.2, 7: 0.3, 14: 0.4},
        mean_cycle_point_absolute_error_days=1.5,
        wearable_resolved_forecast_count=10,
        wearable_completed_cycle_count=2,
        wearable_mean_cycle_logarithmic_loss=0.5,
        wearable_mean_cycle_brier_score=0.3,
        wearable_mean_cycle_window_brier_scores={
            1: 0.1,
            3: 0.2,
            7: 0.3,
            14: 0.4,
        },
        temperature_resolved_forecast_count=10,
        temperature_completed_cycle_count=2,
        temperature_mean_cycle_logarithmic_loss=0.45,
        temperature_mean_cycle_brier_score=0.25,
        temperature_mean_cycle_window_brier_scores={
            1: 0.1,
            3: 0.2,
            7: 0.3,
            14: 0.4,
        },
    )


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
    answers = iter(("3", "1", "1"))

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
    answers = iter(("4", "", "n"))

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


def test_period_record_command_appends_ongoing_period_without_prompt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Support a repeatable explicit command while leaving duration unknown."""
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "cycle_start_date,period_length_days\n2025-01-01,5\n",
        encoding="utf-8",
    )

    status = main(
        (
            "period-record",
            "--history",
            str(history_path),
            "--date",
            "2025-02-01",
            "--yes",
        )
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "✓ PERIOD HISTORY UPDATED" in captured.out
    assert "Completed cycle    31 days" in captured.out
    assert "ongoing · add it later" in captured.out
    assert history_path.read_text(encoding="utf-8").endswith("2025-02-01,\n")


def test_bare_command_completes_pending_period_intuitively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Guide duration completion through the main menu and one confirmation."""
    history_directory = tmp_path / "data/raw"
    history_directory.mkdir(parents=True)
    history_path = history_directory / "history.csv"
    history_path.write_text(
        "cycle_start_date,period_length_days\n2025-01-01,\n",
        encoding="utf-8",
    )
    answers = iter(("2", "", "2025-01-01", "5", ""))

    def answer_prompt(_: str) -> str:
        """Return the next guided recording selection."""
        return next(answers)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", answer_prompt)

    status = main(())

    captured = capsys.readouterr()
    assert status == 0
    assert "[2] Record a period start" in captured.out
    assert "How many days did this period last?" not in captured.out
    assert "Current duration   5 days" in captured.out
    assert history_path.read_text(encoding="utf-8").endswith("2025-01-01,5\n")


def test_daily_flow_syncs_checks_history_and_forecasts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Complete the primary workflow through one entry point."""
    history_path = tmp_path / "history.csv"
    history_path.write_text(
        "cycle_start_date,period_length_days\n"
        "2026-06-28,5\n"
        "2026-07-27,5\n"
        "2026-08-24,\n",
        encoding="utf-8",
    )
    received: dict[str, object] = {}

    def sync(**arguments: object) -> tuple[OuraRouteSyncResult, ...]:
        """Capture the incremental range and return privacy-safe counts."""
        received.update(arguments)
        return tuple(
            OuraRouteSyncResult(
                route=route,
                page_count=1,
                document_count=2,
                snapshot=None,
            )
            for route in OuraRoute
        )

    def resolve_start(**_: object) -> date:
        """Return a deterministic overlapping incremental start."""
        return date(2026, 8, 26)

    class FixedDateTime(datetime):
        """Provide a deterministic local date to the command."""

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            """Return the invented current instant in the requested timezone."""
            return cls(2026, 8, 27, 9, tzinfo=tz)

    def answer_no(_: str) -> str:
        """Keep the already-current period history unchanged."""
        return "n"

    def reuse_model(**_: object) -> DailyModelRefreshResult:
        """Keep this CLI test focused on orchestration and rendering."""
        return DailyModelRefreshResult(
            status=DailyModelRefreshStatus.CURRENT,
            model_path=tmp_path / "missing-model.json",
            run_path=tmp_path / "training-run.json",
            dataset_fingerprint="sha256:" + "0" * 64,
            training=None,
        )

    def wearable_shadow(**arguments: object) -> WearableDailyPrediction:
        """Return an invented experimental distribution at the shared cutoff."""
        prediction_date = arguments["prediction_date"]
        prediction_cutoff = arguments["prediction_cutoff"]
        assert isinstance(prediction_date, date)
        assert isinstance(prediction_cutoff, datetime)
        return WearableDailyPrediction(
            model_version="wearable-neighbor-v1",
            training_morning_count=42,
            distribution=DailyPeriodDistribution(
                prediction_date=prediction_date,
                prediction_cutoff=prediction_cutoff,
                daily_probabilities=(0.01,) * 15,
                after_horizon_probability=0.85,
            ),
            temperature_model_version="temperature-neighbor-v1",
            temperature_distribution=DailyPeriodDistribution(
                prediction_date=prediction_date,
                prediction_cutoff=prediction_cutoff,
                daily_probabilities=(0.02,) * 15,
                after_horizon_probability=0.7,
            ),
        )

    monkeypatch.setattr(cli, "sync_oura", sync)
    monkeypatch.setattr(cli, "resolve_sync_start_date", resolve_start)
    monkeypatch.setattr(cli, "datetime", FixedDateTime)
    monkeypatch.setattr(cli, "refresh_daily_model_if_needed", reuse_model)
    monkeypatch.setattr(cli, "predict_daily_with_wearable_neighbors", wearable_shadow)
    monkeypatch.setattr(
        cli, "summarize_prospective_performance", _resolved_prospective_summary
    )
    monkeypatch.setattr("builtins.input", answer_no)

    status = main(
        (
            "daily",
            "--history",
            str(history_path),
            "--timezone",
            "America/Los_Angeles",
            "--model",
            str(tmp_path / "missing-model.json"),
            "--token-path",
            str(tmp_path / "token.json"),
            "--snapshot-dir",
            str(tmp_path / "snapshots"),
            "--journal",
            str(tmp_path / "forecast-journal.jsonl"),
        )
    )

    captured = capsys.readouterr()
    assert status == 0
    assert received["start_date"] == date(2026, 8, 26)
    assert received["end_date"] == date(2026, 8, 27)
    assert "One private check-in: sync, record, predict." in captured.out
    assert "Latest recorded start: 2026-08-24" in captured.out
    assert "Existing Phase A model is current" in captured.out
    assert "CURRENT CYCLE" in captured.out
    assert "Cycle day               4" in captured.out
    assert "SHORT-RANGE PROBABILITIES" in captured.out
    assert "Cycle-history baseline" in captured.out
    assert "NEXT PERIOD ESTIMATE" in captured.out
    assert "Naive median of completed cycle lengths" in captured.out
    assert "EXPERIMENTAL WEARABLE SHADOW" in captured.out
    assert "Wearable nearest neighbors" in captured.out
    assert "Cycle day + temperature" in captured.out
    assert "Training mornings       42" in captured.out
    assert "PROSPECTIVE JOURNAL" in captured.out
    assert "PAIRED SHADOW COMPARISON" in captured.out
    assert "Wearable log loss       0.500" in captured.out
    assert "Temperature log loss    0.450" in captured.out
    journal_path = tmp_path / "forecast-journal.jsonl"
    assert journal_path.is_file()
    journal_entry = json.loads(journal_path.read_text(encoding="utf-8"))
    assert journal_entry["wearable_model_version"] == "wearable-neighbor-v1"
    assert journal_entry["temperature_model_version"] == "temperature-neighbor-v1"


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
    comparison = DailyModelComparison(
        prediction_dates=(date(2025, 3, 1),),
        entries=tuple(
            DailyCandidateEvaluation(
                label=label,
                version=version,
                evaluation=DailyForecastEvaluation(
                    count=10,
                    logarithmic_loss=log_loss,
                    multiclass_brier_score=brier,
                    window_brier_scores=window_scores,
                ),
                calibration={},
            )
            for label, version, log_loss, brier, window_scores in (
                (
                    "Empirical cycle hazard",
                    "history-v1",
                    0.2,
                    0.05,
                    {1: 0.0, 3: 0.01, 7: 0.02, 14: 0.03},
                ),
                (
                    "Wearable nearest neighbors",
                    "neighbors-v1",
                    0.9,
                    0.3,
                    {1: 0.01, 3: 0.02, 7: 0.05, 14: 0.2},
                ),
                (
                    "Calibrated discrete survival",
                    "survival-v1",
                    0.6,
                    0.2,
                    {1: 0.0, 3: 0.01, 7: 0.04, 14: 0.1},
                ),
            )
        ),
    )
    aggregate_entries = tuple(
        WearableAggregateEntry(
            label=entry.label,
            version=entry.version,
            mean_logarithmic_loss=entry.evaluation.logarithmic_loss,
            mean_multiclass_brier_score=entry.evaluation.multiclass_brier_score,
            mean_window_brier_scores=entry.evaluation.window_brier_scores,
            log_loss_cycle_wins=int(entry.label == "Empirical cycle hazard"),
            brier_cycle_wins=int(entry.label == "Empirical cycle hazard"),
        )
        for entry in comparison.entries
    )
    candidate_diagnostics = tuple(
        WearableCandidateDiagnostics(
            label=entry.label,
            version=entry.version,
            count=10,
            mean_actual_outcome_probability=0.2,
            minimum_actual_outcome_probability=0.01,
            root_mean_squared_offset_error=2.5,
            mean_signed_offset_error=-0.5,
            calibration={
                window: WearableCalibrationDiagnostic(
                    count=10,
                    mean_predicted_probability=0.4,
                    observed_fraction=0.3,
                )
                for window in (1, 3, 7, 14)
            },
            cycle_day=(
                WearableCycleDayDiagnostic(
                    label="days 1-10",
                    count=10,
                    mean_brier_score=entry.evaluation.multiclass_brier_score,
                ),
            ),
        )
        for entry in comparison.entries
    )
    diagnostics = WearableEvaluationDiagnostics(
        data=WearableDataDiagnostics(
            missingness_rates={"Readiness score": 0.1},
            outcome_window_rates={1: 0.1, 3: 0.2, 7: 0.3, 14: 0.4},
            after_horizon_rate=0.6,
        ),
        candidates=candidate_diagnostics,
    )
    result = WearableEvaluationResult(
        workflow_version="wearable-evaluation-v2",
        mode=WearableEvaluationMode.EXPLORATORY_BACKFILL,
        optimistic_backfill_assumption=True,
        snapshot_count=3,
        normalized_day_count=60,
        aligned_row_count=50,
        uncensored_row_count=45,
        eligible_completed_cycle_count=4,
        evaluation_fold_count=1,
        first_fold_training_cycle_count=2,
        final_fold_training_cycle_count=2,
        evaluation_cycle_count=1,
        evaluation_row_count=10,
        walk_forward=WearableWalkForwardComparison(
            folds=(
                WearableCycleFoldResult(
                    fold_number=1,
                    training_cycle_count=2,
                    training_row_count=25,
                    calibration_row_count=10,
                    evaluation_row_count=10,
                    comparison=comparison,
                    diagnostics=candidate_diagnostics,
                ),
            ),
            entries=aggregate_entries,
        ),
        diagnostics=diagnostics,
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
    assert "CYCLE-LEVEL WALK-FORWARD" in captured.out
    assert "EXACT-DATE SCORES" in captured.out
    assert "PLANNING-WINDOW BRIER" in captured.out
    assert "Best log loss: Cycle history" in captured.out
    assert "PER-CYCLE EXACT-DATE BRIER" in captured.out
    assert "CYCLE WINS" in captured.out
    assert "EVALUATED-DATA DIAGNOSTICS" in captured.out
    assert "MODEL BEHAVIOR" in captured.out
    assert "PLANNING-WINDOW CALIBRATION" in captured.out
    assert "CYCLE-DAY BRIER" in captured.out
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
        workflow_version="wearable-evaluation-v2",
        mode=WearableEvaluationMode.EXPLORATORY_BACKFILL,
        optimistic_backfill_assumption=True,
        snapshot_count=3,
        normalized_day_count=60,
        aligned_row_count=50,
        uncensored_row_count=45,
        eligible_completed_cycle_count=4,
        evaluation_fold_count=1,
        first_fold_training_cycle_count=2,
        final_fold_training_cycle_count=2,
        evaluation_cycle_count=1,
        evaluation_row_count=10,
        walk_forward=WearableWalkForwardComparison(folds=(), entries=()),
        diagnostics=WearableEvaluationDiagnostics(
            data=WearableDataDiagnostics(
                missingness_rates={},
                outcome_window_rates={},
                after_horizon_rate=0.0,
            ),
            candidates=(),
        ),
    )
    received: dict[str, object] = {}

    def evaluate(**arguments: object) -> WearableEvaluationResult:
        """Capture guided selections and return invented aggregate results."""
        received.update(arguments)
        return result

    answers = iter(("5", "1", "2", "America/New_York"))

    def answer_prompt(_: str) -> str:
        """Return the next guided wearable selection."""
        return next(answers)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", answer_prompt)
    monkeypatch.setattr(cli, "evaluate_local_wearable_models", evaluate)

    status = main(())

    captured = capsys.readouterr()
    assert status == 0
    assert "[5] Evaluate wearable models" in captured.out
    assert "EVALUATION MODE" in captured.out
    assert "Optimistic historical assumption" in captured.out
    assert received["history_path"] == Path("data/raw/history.csv")
    assert received["mode"] is WearableEvaluationMode.EXPLORATORY_BACKFILL
    assert received["timezone_name"] == "America/New_York"
