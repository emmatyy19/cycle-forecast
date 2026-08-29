"""Tests for authenticated encrypted private-data backup and restore."""

import base64
import json
from datetime import UTC, date, datetime
from pathlib import Path
from stat import S_IMODE

import pytest

from cycle_forecast.data.oura_client import OuraRoute, retrieve_collection
from cycle_forecast.data.oura_snapshot import write_snapshot
from cycle_forecast.data.private_backup import (
    PRIVATE_BACKUP_MAGIC,
    PrivateBackupError,
    create_private_backup,
    restore_private_backup,
)

_PASSWORD = "synthetic-test-password"


def _write_history(*, path: Path) -> None:
    """Write a small invented valid cycle history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "cycle_start_date,period_length_days\n2025-01-01,5\n2025-01-30,\n",
        encoding="utf-8",
    )


def _write_snapshot(*, directory: Path) -> Path:
    """Write one invented validated Oura snapshot."""
    payload = (
        b'{"data":[{"id":"synthetic","contributors":{},'
        b'"day":"2025-01-15","timestamp":"2025-01-15T00:00:00-05:00"}],'
        b'"next_token":null}'
    )
    pages = retrieve_collection(
        route=OuraRoute.DAILY_READINESS,
        access_token="synthetic",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        transport=lambda _: payload,
    )
    return write_snapshot(
        directory=directory,
        route=OuraRoute.DAILY_READINESS,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        retrieval_started_at=datetime(2025, 2, 1, tzinfo=UTC),
        retrieval_completed_at=datetime(2025, 2, 1, 0, 0, 1, tzinfo=UTC),
        timezone_name="America/New_York",
        pages=pages,
    ).path


def test_encrypted_backup_round_trip_restores_validated_private_files(
    tmp_path: Path,
) -> None:
    """Restore exact history and snapshot bytes without exposing plaintext."""
    history = tmp_path / "source/history.csv"
    snapshots = tmp_path / "source/snapshots"
    journal = tmp_path / "source/forecast-journal.jsonl"
    _write_history(path=history)
    snapshot = _write_snapshot(directory=snapshots)
    journal.write_text("", encoding="utf-8")
    backup = tmp_path / "backups/private.cfbackup"

    created = create_private_backup(
        history_path=history,
        snapshot_directory=snapshots,
        journal_path=journal,
        output_path=backup,
        password=_PASSWORD,
        created_at=datetime(2025, 2, 2, tzinfo=UTC),
    )

    encrypted = backup.read_bytes()
    assert b"cycle_start_date" not in encrypted
    assert b"synthetic" not in encrypted
    assert created.snapshot_count == 1
    assert created.journal_included
    assert S_IMODE(backup.stat().st_mode) == 0o600

    restored_history = tmp_path / "restored/history.csv"
    restored_snapshots = tmp_path / "restored/snapshots"
    restored_journal = tmp_path / "restored/journal.jsonl"
    result = restore_private_backup(
        input_path=backup,
        history_path=restored_history,
        snapshot_directory=restored_snapshots,
        journal_path=restored_journal,
        password=_PASSWORD,
    )

    assert restored_history.read_bytes() == history.read_bytes()
    assert (restored_snapshots / snapshot.name).read_bytes() == snapshot.read_bytes()
    assert restored_journal.read_bytes() == journal.read_bytes()
    assert result.snapshot_count == 1
    assert result.journal_restored
    assert result.replaced_file_count == 0


def test_restore_rejects_wrong_password_before_writing(tmp_path: Path) -> None:
    """Authenticate the entire bundle before creating destination files."""
    history = tmp_path / "source/history.csv"
    _write_history(path=history)
    backup = tmp_path / "private.cfbackup"
    create_private_backup(
        history_path=history,
        snapshot_directory=tmp_path / "missing-snapshots",
        journal_path=tmp_path / "missing-journal.jsonl",
        output_path=backup,
        password=_PASSWORD,
        created_at=datetime(2025, 2, 2, tzinfo=UTC),
    )
    destination = tmp_path / "restored/history.csv"

    with pytest.raises(PrivateBackupError, match=r"password.*damaged"):
        restore_private_backup(
            input_path=backup,
            history_path=destination,
            snapshot_directory=tmp_path / "restored/snapshots",
            journal_path=tmp_path / "restored/journal.jsonl",
            password="incorrect-password",
        )

    assert not destination.exists()


def test_restore_rejects_tampered_ciphertext_before_writing(tmp_path: Path) -> None:
    """Detect any encrypted-bundle modification through authentication."""
    history = tmp_path / "source/history.csv"
    _write_history(path=history)
    backup = tmp_path / "private.cfbackup"
    create_private_backup(
        history_path=history,
        snapshot_directory=tmp_path / "missing-snapshots",
        journal_path=tmp_path / "missing-journal.jsonl",
        output_path=backup,
        password=_PASSWORD,
        created_at=datetime(2025, 2, 2, tzinfo=UTC),
    )
    encrypted = bytearray(backup.read_bytes())
    encrypted[-1] ^= 1
    backup.write_bytes(encrypted)
    destination = tmp_path / "restored/history.csv"

    with pytest.raises(PrivateBackupError, match="damaged"):
        restore_private_backup(
            input_path=backup,
            history_path=destination,
            snapshot_directory=tmp_path / "restored/snapshots",
            journal_path=tmp_path / "restored/journal.jsonl",
            password=_PASSWORD,
        )

    assert not destination.exists()


def test_restore_requires_explicit_replacement(tmp_path: Path) -> None:
    """Preserve an existing destination unless replacement is authorized."""
    history = tmp_path / "source/history.csv"
    _write_history(path=history)
    backup = tmp_path / "private.cfbackup"
    create_private_backup(
        history_path=history,
        snapshot_directory=tmp_path / "missing-snapshots",
        journal_path=tmp_path / "missing-journal.jsonl",
        output_path=backup,
        password=_PASSWORD,
        created_at=datetime(2025, 2, 2, tzinfo=UTC),
    )
    destination = tmp_path / "restored/history.csv"
    destination.parent.mkdir()
    destination.write_text("keep me", encoding="utf-8")

    with pytest.raises(PrivateBackupError, match="explicit replacement"):
        restore_private_backup(
            input_path=backup,
            history_path=destination,
            snapshot_directory=tmp_path / "restored/snapshots",
            journal_path=tmp_path / "restored/journal.jsonl",
            password=_PASSWORD,
        )

    assert destination.read_text(encoding="utf-8") == "keep me"


def test_backup_and_restore_cannot_overwrite_their_own_inputs(tmp_path: Path) -> None:
    """Reject path collisions even when replacement is explicitly enabled."""
    history = tmp_path / "history.csv"
    _write_history(path=history)

    with pytest.raises(PrivateBackupError, match="source file"):
        create_private_backup(
            history_path=history,
            snapshot_directory=tmp_path / "missing-snapshots",
            journal_path=tmp_path / "missing-journal.jsonl",
            output_path=history,
            password=_PASSWORD,
            created_at=datetime(2025, 2, 2, tzinfo=UTC),
            replace=True,
        )

    backup = tmp_path / "private.cfbackup"
    create_private_backup(
        history_path=history,
        snapshot_directory=tmp_path / "missing-snapshots",
        journal_path=tmp_path / "missing-journal.jsonl",
        output_path=backup,
        password=_PASSWORD,
        created_at=datetime(2025, 2, 2, tzinfo=UTC),
    )
    original_backup = backup.read_bytes()

    with pytest.raises(PrivateBackupError, match="backup bundle"):
        restore_private_backup(
            input_path=backup,
            history_path=backup,
            snapshot_directory=tmp_path / "restored/snapshots",
            journal_path=tmp_path / "restored/journal.jsonl",
            password=_PASSWORD,
            replace=True,
        )

    assert backup.read_bytes() == original_backup


def test_backup_requires_strong_password_and_valid_sources(tmp_path: Path) -> None:
    """Reject weak protection and malformed private inputs before writing."""
    history = tmp_path / "history.csv"
    _write_history(path=history)
    output = tmp_path / "private.cfbackup"

    with pytest.raises(PrivateBackupError, match="at least 12"):
        create_private_backup(
            history_path=history,
            snapshot_directory=tmp_path / "missing-snapshots",
            journal_path=tmp_path / "missing-journal.jsonl",
            output_path=output,
            password="too-short",
            created_at=datetime(2025, 2, 2, tzinfo=UTC),
        )

    history.write_text("not,a,valid,history\n", encoding="utf-8")
    with pytest.raises(PrivateBackupError, match="invalid cycle history"):
        create_private_backup(
            history_path=history,
            snapshot_directory=tmp_path / "missing-snapshots",
            journal_path=tmp_path / "missing-journal.jsonl",
            output_path=output,
            password=_PASSWORD,
            created_at=datetime(2025, 2, 2, tzinfo=UTC),
        )

    assert not output.exists()


def test_backup_rejects_invalid_optional_sources_and_naive_time(tmp_path: Path) -> None:
    """Validate optional inputs and provenance before producing a bundle."""
    history = tmp_path / "history.csv"
    _write_history(path=history)
    snapshot_path = tmp_path / "snapshots"
    snapshot_path.write_text("not a directory", encoding="utf-8")
    output = tmp_path / "private.cfbackup"

    with pytest.raises(PrivateBackupError, match="regular directory"):
        create_private_backup(
            history_path=history,
            snapshot_directory=snapshot_path,
            journal_path=tmp_path / "missing-journal.jsonl",
            output_path=output,
            password=_PASSWORD,
            created_at=datetime(2025, 2, 2, tzinfo=UTC),
        )

    snapshot_path.unlink()
    snapshot_path.mkdir()
    (snapshot_path / "invalid.json").write_text("not JSON", encoding="utf-8")
    with pytest.raises(PrivateBackupError, match="invalid Oura snapshot"):
        create_private_backup(
            history_path=history,
            snapshot_directory=snapshot_path,
            journal_path=tmp_path / "missing-journal.jsonl",
            output_path=output,
            password=_PASSWORD,
            created_at=datetime(2025, 2, 2, tzinfo=UTC),
        )

    (snapshot_path / "invalid.json").unlink()
    journal = tmp_path / "journal.jsonl"
    journal.write_text("\n", encoding="utf-8")
    with pytest.raises(PrivateBackupError, match="invalid forecast journal"):
        create_private_backup(
            history_path=history,
            snapshot_directory=snapshot_path,
            journal_path=journal,
            output_path=output,
            password=_PASSWORD,
            created_at=datetime(2025, 2, 2, tzinfo=UTC),
        )

    journal.unlink()
    with pytest.raises(PrivateBackupError, match="timezone-aware"):
        create_private_backup(
            history_path=history,
            snapshot_directory=snapshot_path,
            journal_path=journal,
            output_path=output,
            password=_PASSWORD,
            created_at=datetime(2025, 2, 2),
        )


def test_restore_rejects_malformed_unencrypted_headers(tmp_path: Path) -> None:
    """Validate fixed header shape and key-derivation parameters before use."""
    history = tmp_path / "history.csv"
    _write_history(path=history)
    valid_backup = tmp_path / "valid.cfbackup"
    create_private_backup(
        history_path=history,
        snapshot_directory=tmp_path / "missing-snapshots",
        journal_path=tmp_path / "missing-journal.jsonl",
        output_path=valid_backup,
        password=_PASSWORD,
        created_at=datetime(2025, 2, 2, tzinfo=UTC),
    )
    header, ciphertext = valid_backup.read_bytes()[len(PRIVATE_BACKUP_MAGIC) :].split(
        b"\n", 1
    )
    metadata = json.loads(header)
    assert isinstance(metadata, dict)
    invalid_headers = (
        b"[]",
        json.dumps({**metadata, "scrypt_n": 1}).encode(),
        json.dumps({**metadata, "salt": base64.b64encode(b"short").decode()}).encode(),
    )

    for index, invalid_header in enumerate(invalid_headers):
        backup = tmp_path / f"invalid-{index}.cfbackup"
        backup.write_bytes(PRIVATE_BACKUP_MAGIC + invalid_header + b"\n" + ciphertext)
        with pytest.raises(PrivateBackupError, match="invalid private backup header"):
            restore_private_backup(
                input_path=backup,
                history_path=tmp_path / f"restored-{index}/history.csv",
                snapshot_directory=tmp_path / f"restored-{index}/snapshots",
                journal_path=tmp_path / f"restored-{index}/journal.jsonl",
                password=_PASSWORD,
            )


def test_backup_preserves_existing_bundle_without_replace(tmp_path: Path) -> None:
    """Require explicit replacement and then atomically replace a prior bundle."""
    history = tmp_path / "history.csv"
    _write_history(path=history)
    output = tmp_path / "private.cfbackup"

    def create(*, replace: bool = False) -> None:
        """Create the same invented bundle with optional replacement."""
        create_private_backup(
            history_path=history,
            snapshot_directory=tmp_path / "missing-snapshots",
            journal_path=tmp_path / "missing-journal.jsonl",
            output_path=output,
            password=_PASSWORD,
            created_at=datetime(2025, 2, 2, tzinfo=UTC),
            replace=replace,
        )

    create()
    original = output.read_bytes()

    with pytest.raises(PrivateBackupError, match="already exists"):
        create()

    create(replace=True)
    assert output.read_bytes() != original


def test_restore_rejects_unrelated_file_and_destination_collisions(
    tmp_path: Path,
) -> None:
    """Reject unsupported input early and require distinct restore targets."""
    unrelated = tmp_path / "not-a-backup"
    unrelated.write_bytes(b"plaintext")

    with pytest.raises(PrivateBackupError, match="not a supported"):
        restore_private_backup(
            input_path=unrelated,
            history_path=tmp_path / "restored/history.csv",
            snapshot_directory=tmp_path / "restored/snapshots",
            journal_path=tmp_path / "restored/journal.jsonl",
            password=_PASSWORD,
        )

    history = tmp_path / "source/history.csv"
    journal = tmp_path / "source/journal.jsonl"
    _write_history(path=history)
    journal.write_text("", encoding="utf-8")
    backup = tmp_path / "private.cfbackup"
    create_private_backup(
        history_path=history,
        snapshot_directory=tmp_path / "missing-snapshots",
        journal_path=journal,
        output_path=backup,
        password=_PASSWORD,
        created_at=datetime(2025, 2, 2, tzinfo=UTC),
    )
    collision = tmp_path / "restored/shared"

    with pytest.raises(PrivateBackupError, match="destinations must be distinct"):
        restore_private_backup(
            input_path=backup,
            history_path=collision,
            snapshot_directory=tmp_path / "restored/snapshots",
            journal_path=collision,
            password=_PASSWORD,
        )
