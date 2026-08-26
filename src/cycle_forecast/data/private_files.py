"""Apply owner-only permissions to directories containing private local data."""

from pathlib import Path


def ensure_private_directory(*, directory: Path) -> None:
    """Create a directory and restrict access to its owner.

    Parameters
    ----------
    directory
        Directory that will contain credentials or personal data.

    Raises
    ------
    OSError
        If the directory cannot be created or its permissions cannot be secured.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
