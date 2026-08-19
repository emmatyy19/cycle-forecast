"""Package-level smoke tests."""

import cycle_forecast


def test_package_version() -> None:
    """The package exposes its current version."""
    assert cycle_forecast.__version__ == "0.1.0"
