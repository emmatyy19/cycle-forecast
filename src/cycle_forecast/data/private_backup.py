"""Create and restore authenticated encrypted bundles of private local data."""

import base64
import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from cycle_forecast.data.cycle_history import load_cycle_history
from cycle_forecast.data.oura_snapshot import load_snapshot
from cycle_forecast.data.private_files import ensure_private_directory
from cycle_forecast.evaluation.prospective_journal import load_prospective_journal

PRIVATE_BACKUP_VERSION: Final = "cycle-forecast-private-backup-v1"
"""Version of encryption, manifest, and logical archive-path semantics."""

PRIVATE_BACKUP_MAGIC: Final = b"CYCLE_FORECAST_PRIVATE_BACKUP_V1\n"
"""Unencrypted prefix used to reject unrelated files before decryption."""

MINIMUM_BACKUP_PASSWORD_LENGTH: Final = 12
"""Smallest accepted backup password length."""

_SCRYPT_N: Final = 2**15
_SCRYPT_R: Final = 8
_SCRYPT_P: Final = 1
_KEY_LENGTH: Final = 32
_SALT_LENGTH: Final = 16
_NONCE_LENGTH: Final = 12
_MAXIMUM_RESTORE_BYTES: Final = 1_000_000_000
_HISTORY_ARCHIVE_PATH: Final = "history/cycle_history.csv"
_JOURNAL_ARCHIVE_PATH: Final = "journal/forecast-journal.jsonl"
_SNAPSHOT_ARCHIVE_PREFIX: Final = "oura/snapshots/"
_MANIFEST_ARCHIVE_PATH: Final = "manifest.json"


class PrivateBackupError(ValueError):
    """Indicate an unsafe, invalid, or unreadable private backup operation."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PrivateBackupResult:
    """Describe an encrypted backup without exposing private file contents."""

    output_path: Path
    snapshot_count: int
    journal_included: bool
    encrypted_byte_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PrivateRestoreResult:
    """Describe a verified restore without exposing private file contents."""

    history_path: Path
    snapshot_count: int
    journal_restored: bool
    replaced_file_count: int


def _derive_key(*, password: str, salt: bytes) -> bytes:
    """Derive one encryption key from a password and random salt."""
    return Scrypt(
        salt=salt,
        length=_KEY_LENGTH,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    ).derive(password.encode("utf-8"))


def _sha256(*, payload: bytes) -> str:
    """Return a domain-neutral SHA-256 digest for one archived payload."""
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _read_private_file(*, path: Path, label: str) -> bytes:
    """Read one regular non-symlink private source file."""
    if path.is_symlink() or not path.is_file():
        raise PrivateBackupError(f"{label} must be a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise PrivateBackupError(f"could not read {label}: {error}") from error


def _validated_backup_files(
    *, history_path: Path, snapshot_directory: Path, journal_path: Path
) -> dict[str, bytes]:
    """Load validated private inputs under stable logical archive paths."""
    try:
        load_cycle_history(path=history_path)
    except (OSError, ValueError) as error:
        raise PrivateBackupError(f"invalid cycle history: {error}") from error
    files = {
        _HISTORY_ARCHIVE_PATH: _read_private_file(
            path=history_path, label="cycle history"
        )
    }
    if snapshot_directory.exists():
        if snapshot_directory.is_symlink() or not snapshot_directory.is_dir():
            raise PrivateBackupError(
                f"snapshot directory must be a regular directory: {snapshot_directory}"
            )
        for snapshot_path in sorted(snapshot_directory.glob("*.json")):
            if snapshot_path.is_symlink():
                raise PrivateBackupError(
                    f"snapshot must not be a symbolic link: {snapshot_path}"
                )
            try:
                load_snapshot(path=snapshot_path)
            except (OSError, ValueError) as error:
                raise PrivateBackupError(
                    f"invalid Oura snapshot {snapshot_path.name}: {error}"
                ) from error
            files[f"{_SNAPSHOT_ARCHIVE_PREFIX}{snapshot_path.name}"] = (
                _read_private_file(path=snapshot_path, label="Oura snapshot")
            )
    if journal_path.exists():
        try:
            load_prospective_journal(path=journal_path)
        except (OSError, ValueError) as error:
            raise PrivateBackupError(f"invalid forecast journal: {error}") from error
        files[_JOURNAL_ARCHIVE_PATH] = _read_private_file(
            path=journal_path, label="forecast journal"
        )
    return files


def _build_archive(*, files: dict[str, bytes], created_at: datetime) -> bytes:
    """Build an in-memory ZIP containing a checksummed private manifest."""
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise PrivateBackupError("backup creation time must be timezone-aware")
    manifest = {
        "version": PRIVATE_BACKUP_VERSION,
        "created_at": created_at.astimezone(UTC).isoformat(),
        "files": [
            {"path": path, "size": len(payload), "sha256": _sha256(payload=payload)}
            for path, payload in sorted(files.items())
        ],
    }
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(
            _MANIFEST_ARCHIVE_PATH,
            json.dumps(manifest, separators=(",", ":"), sort_keys=True),
        )
        for path, payload in sorted(files.items()):
            archive.writestr(path, payload)
    return archive_buffer.getvalue()


def _write_bytes_atomically(
    *, path: Path, payload: bytes, replace: bool, secure_parent: bool
) -> None:
    """Write owner-only bytes atomically without implicit replacement."""
    if path.exists() and not replace:
        raise PrivateBackupError(f"destination already exists: {path}")
    if secure_parent:
        ensure_private_directory(directory=path.parent)
    else:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
    except OSError as error:
        raise PrivateBackupError(f"could not safely write {path}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def create_private_backup(
    *,
    history_path: Path,
    snapshot_directory: Path,
    journal_path: Path,
    output_path: Path,
    password: str,
    created_at: datetime,
    replace: bool = False,
) -> PrivateBackupResult:
    """Validate and encrypt private forecasting inputs into one portable bundle."""
    if len(password) < MINIMUM_BACKUP_PASSWORD_LENGTH:
        raise PrivateBackupError(
            f"backup password must contain at least {MINIMUM_BACKUP_PASSWORD_LENGTH} characters"
        )
    output_resolved = output_path.resolve()
    source_paths = [history_path]
    if journal_path.exists():
        source_paths.append(journal_path)
    if snapshot_directory.is_dir():
        source_paths.extend(snapshot_directory.glob("*.json"))
    if any(path.resolve() == output_resolved for path in source_paths):
        raise PrivateBackupError("backup destination cannot replace a source file")
    files = _validated_backup_files(
        history_path=history_path,
        snapshot_directory=snapshot_directory,
        journal_path=journal_path,
    )
    plaintext = _build_archive(files=files, created_at=created_at)
    salt = os.urandom(_SALT_LENGTH)
    nonce = os.urandom(_NONCE_LENGTH)
    header = json.dumps(
        {
            "version": PRIVATE_BACKUP_VERSION,
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "scrypt_n": _SCRYPT_N,
            "scrypt_r": _SCRYPT_R,
            "scrypt_p": _SCRYPT_P,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    ciphertext = AESGCM(_derive_key(password=password, salt=salt)).encrypt(
        nonce, plaintext, header
    )
    encrypted = PRIVATE_BACKUP_MAGIC + header + b"\n" + ciphertext
    _write_bytes_atomically(
        path=output_path,
        payload=encrypted,
        replace=replace,
        secure_parent=False,
    )
    return PrivateBackupResult(
        output_path=output_path,
        snapshot_count=sum(path.startswith(_SNAPSHOT_ARCHIVE_PREFIX) for path in files),
        journal_included=_JOURNAL_ARCHIVE_PATH in files,
        encrypted_byte_count=len(encrypted),
    )


def _decrypt_backup(*, input_path: Path, password: str) -> bytes:
    """Authenticate and decrypt one versioned private backup bundle."""
    payload = _read_private_file(path=input_path, label="private backup")
    if not payload.startswith(PRIVATE_BACKUP_MAGIC):
        raise PrivateBackupError("file is not a supported private backup")
    try:
        header, ciphertext = payload[len(PRIVATE_BACKUP_MAGIC) :].split(b"\n", 1)
        metadata = cast(object, json.loads(header))
        if not isinstance(metadata, dict):
            raise ValueError("backup header must be an object")
        fields = cast(dict[object, object], metadata)
        if fields != {
            "version": PRIVATE_BACKUP_VERSION,
            "salt": fields.get("salt"),
            "nonce": fields.get("nonce"),
            "scrypt_n": _SCRYPT_N,
            "scrypt_r": _SCRYPT_R,
            "scrypt_p": _SCRYPT_P,
        }:
            raise ValueError("unsupported backup header")
        salt = base64.b64decode(str(fields["salt"]), validate=True)
        nonce = base64.b64decode(str(fields["nonce"]), validate=True)
        if len(salt) != _SALT_LENGTH or len(nonce) != _NONCE_LENGTH:
            raise ValueError("invalid backup salt or nonce")
        return AESGCM(_derive_key(password=password, salt=salt)).decrypt(
            nonce, ciphertext, header
        )
    except InvalidTag as error:
        raise PrivateBackupError(
            "backup password is incorrect or the bundle is damaged"
        ) from error
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise PrivateBackupError(f"invalid private backup header: {error}") from error


def _validated_archive_files(*, plaintext: bytes) -> dict[str, bytes]:
    """Verify archive shape, size, manifest fields, and every payload checksum."""
    try:
        with zipfile.ZipFile(io.BytesIO(plaintext), mode="r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or _MANIFEST_ARCHIVE_PATH not in names:
                raise PrivateBackupError(
                    "backup archive has duplicate or missing entries"
                )
            if sum(member.file_size for member in members) > _MAXIMUM_RESTORE_BYTES:
                raise PrivateBackupError(
                    "backup archive exceeds the restore size limit"
                )
            manifest_raw = cast(
                object, json.loads(archive.read(_MANIFEST_ARCHIVE_PATH))
            )
            if not isinstance(manifest_raw, dict):
                raise PrivateBackupError("backup manifest must be an object")
            manifest = cast(dict[object, object], manifest_raw)
            if set(manifest) != {"version", "created_at", "files"}:
                raise PrivateBackupError("backup manifest fields are invalid")
            if manifest.get("version") != PRIVATE_BACKUP_VERSION:
                raise PrivateBackupError("unsupported backup manifest version")
            created_at = datetime.fromisoformat(str(manifest.get("created_at")))
            if created_at.tzinfo is None or created_at.utcoffset() is None:
                raise PrivateBackupError("backup manifest time must be timezone-aware")
            entries_raw = manifest.get("files")
            if not isinstance(entries_raw, list):
                raise PrivateBackupError("backup manifest files must be an array")
            files: dict[str, bytes] = {}
            for value in cast(list[object], entries_raw):
                if not isinstance(value, dict):
                    raise PrivateBackupError("backup manifest file must be an object")
                entry = cast(dict[object, object], value)
                if set(entry) != {"path", "size", "sha256"}:
                    raise PrivateBackupError("invalid backup manifest file fields")
                path = entry.get("path")
                size = entry.get("size")
                digest = entry.get("sha256")
                if (
                    not isinstance(path, str)
                    or not isinstance(size, int)
                    or isinstance(size, bool)
                    or not isinstance(digest, str)
                    or path in files
                ):
                    raise PrivateBackupError("invalid backup manifest file metadata")
                if path not in names or path == _MANIFEST_ARCHIVE_PATH:
                    raise PrivateBackupError(
                        "backup manifest references an invalid file"
                    )
                payload = archive.read(path)
                if len(payload) != size or _sha256(payload=payload) != digest:
                    raise PrivateBackupError("backup payload checksum mismatch")
                files[path] = payload
            if set(names) != {*files, _MANIFEST_ARCHIVE_PATH}:
                raise PrivateBackupError("backup archive contains unmanifested files")
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        if isinstance(error, PrivateBackupError):
            raise
        raise PrivateBackupError(f"invalid private backup archive: {error}") from error
    if _HISTORY_ARCHIVE_PATH not in files:
        raise PrivateBackupError("backup does not contain cycle history")
    if any(
        path not in {_HISTORY_ARCHIVE_PATH, _JOURNAL_ARCHIVE_PATH}
        and (
            not path.startswith(_SNAPSHOT_ARCHIVE_PREFIX)
            or path != f"{_SNAPSHOT_ARCHIVE_PREFIX}{Path(path).name}"
        )
        for path in files
    ):
        raise PrivateBackupError("backup contains an unsupported logical path")
    return files


def _validate_restored_payloads(*, files: dict[str, bytes]) -> None:
    """Run domain validators against decrypted payloads before any restore write."""
    with tempfile.TemporaryDirectory(prefix="cycle-forecast-restore-") as temporary:
        root = Path(temporary)
        root.chmod(0o700)
        history = root / _HISTORY_ARCHIVE_PATH
        history.parent.mkdir(parents=True)
        history.write_bytes(files[_HISTORY_ARCHIVE_PATH])
        try:
            load_cycle_history(path=history)
            for path, payload in files.items():
                if path.startswith(_SNAPSHOT_ARCHIVE_PREFIX):
                    snapshot = root / path
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.write_bytes(payload)
                    load_snapshot(path=snapshot)
            if _JOURNAL_ARCHIVE_PATH in files:
                journal = root / _JOURNAL_ARCHIVE_PATH
                journal.parent.mkdir(parents=True, exist_ok=True)
                journal.write_bytes(files[_JOURNAL_ARCHIVE_PATH])
                load_prospective_journal(path=journal)
        except (OSError, ValueError) as error:
            raise PrivateBackupError(
                f"decrypted backup contains invalid private data: {error}"
            ) from error


def restore_private_backup(
    *,
    input_path: Path,
    history_path: Path,
    snapshot_directory: Path,
    journal_path: Path,
    password: str,
    replace: bool = False,
) -> PrivateRestoreResult:
    """Authenticate, verify, and restore private data to explicit destinations."""
    files = _validated_archive_files(
        plaintext=_decrypt_backup(input_path=input_path, password=password)
    )
    _validate_restored_payloads(files=files)
    destinations = {
        _HISTORY_ARCHIVE_PATH: history_path,
        **{
            path: snapshot_directory / Path(path).name
            for path in files
            if path.startswith(_SNAPSHOT_ARCHIVE_PREFIX)
        },
    }
    if _JOURNAL_ARCHIVE_PATH in files:
        destinations[_JOURNAL_ARCHIVE_PATH] = journal_path
    destination_paths = tuple(path.resolve() for path in destinations.values())
    if len(destination_paths) != len(set(destination_paths)):
        raise PrivateBackupError("restore destinations must be distinct")
    if input_path.resolve() in destination_paths:
        raise PrivateBackupError("restore destination cannot replace the backup bundle")
    existing = tuple(path for path in destinations.values() if path.exists())
    if existing and not replace:
        raise PrivateBackupError(
            f"restore would replace {len(existing)} existing private file(s); "
            "explicit replacement is required"
        )
    for logical_path, destination in destinations.items():
        _write_bytes_atomically(
            path=destination,
            payload=files[logical_path],
            replace=replace,
            secure_parent=True,
        )
    return PrivateRestoreResult(
        history_path=history_path,
        snapshot_count=sum(path.startswith(_SNAPSHOT_ARCHIVE_PREFIX) for path in files),
        journal_restored=_JOURNAL_ARCHIVE_PATH in files,
        replaced_file_count=len(existing),
    )
