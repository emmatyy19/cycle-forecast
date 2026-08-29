# Design 006: Private Backup and Restore

## Status

Accepted

## Scope and threat model

The backup workflow protects portable copies of local cycle history, immutable
Oura snapshots, and the prospective forecast journal. These inputs are
irreplaceable or required for strict prospective evaluation. OAuth credentials
and derived model artifacts are excluded: credentials should be reauthorized on
a recovered device, and model artifacts can be regenerated from restored data.

The encrypted bundle protects confidentiality and detects modification while it
is stored or copied outside the project. It does not protect an unlocked running
process, a compromised device while the password is entered, or a password that
is stored beside the bundle. The application never stores or recovers the
backup password.

## Cryptographic envelope

The `cycle-forecast-private-backup-v1` format derives a 256-bit key from the
UTF-8 password with scrypt using a fresh 16-byte random salt, `N = 32768`,
`r = 8`, and `p = 1`. AES-GCM encrypts and authenticates the complete archive
with a fresh 12-byte nonce. The small unencrypted header contains only the
format version, salt, nonce, and fixed key-derivation parameters; it is supplied
as authenticated additional data. Wrong passwords, header changes, and
ciphertext changes therefore fail before private payloads are parsed or written.

Passwords must contain at least 12 characters. This is only a minimum safeguard;
a unique generated password stored in a password manager is preferred.

## Encrypted archive

The plaintext inside the authenticated envelope is a ZIP archive with fixed
logical paths rather than original filesystem paths. Its encrypted manifest
records the format version, UTC creation time, byte size, and SHA-256 checksum
of every payload. Backup creation runs the existing domain validator for cycle
history, every Oura snapshot, and the forecast journal before encryption.
Symbolic-link inputs are rejected so the workflow cannot silently capture data
outside the selected sources.

## Restore safety

Restore authenticates and decrypts the complete bundle, limits declared
uncompressed content to 1 GB, requires an exact manifest-to-archive match, and
verifies every size and checksum. Decrypted files are staged in an owner-only
temporary directory and passed through their existing domain validators before
any destination is written.

Logical archive paths are mapped to explicit destination arguments; ZIP paths
are never extracted directly. Destination collisions and attempts to overwrite
the backup bundle itself are rejected. Existing destination files are preserved
unless the caller supplies `--replace`. Each restored file is written through
an owner-only temporary file and atomically replaced. Extra local snapshots not
represented in the bundle are preserved rather than deleted.

The multi-file restore is validated as a unit but cannot be atomically committed
across multiple filesystem directories. An operating-system failure during the
write phase may therefore leave a partial restore; rerunning the same verified
bundle with explicit replacement completes it safely.
