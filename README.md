# Cycle Forecast

An end-to-end machine learning project for forecasting menstrual cycle timing
from cycle history and, eventually, wearable data.

This project is for personal experimentation and is not a medical device. It
must not be used for diagnosis or medical decision-making.

## Repository toolchain

The project uses Python 3.12 and an installable `src/` package layout. Tool and
dependency configuration is centralized in `pyproject.toml`:

- **uv** manages Python, the virtual environment, dependencies, and the lockfile.
- **Ruff** formats code, sorts imports, and enforces lint and NumPy-style
  docstring rules.
- **Pyright/Pylance** performs strict static type checking. Pyright runs from
  the command line; Pylance displays the same class of diagnostics in VS Code.
- **pytest** runs the test suite, with **pytest-testmon** selecting tests affected
  by a change during pre-commit.
- **pre-commit** applies these checks before Git creates a commit.

Exact dependency resolutions live in `uv.lock`, which is committed and should
not be edited manually.

## Development setup

Create or synchronize the locked environment:

```bash
uv sync --all-groups
```

Install the repository's Git hooks:

```bash
uv run pre-commit install
```

VS Code workspace settings select `.venv/bin/python`, enable pytest discovery,
use Pylance diagnostics, and format Python with Ruff on save. Install the
Microsoft Python and Pylance extensions and the Astral Ruff extension when VS
Code prompts for them.

## Quality checks

Run all checks manually with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

Pyright is configured in strict mode. All functions must therefore have typed
parameters and return values. Ruff enforces NumPy-style docstrings.

The pre-commit pytest hook runs:

```bash
uv run pytest --testmon
```

The first run executes the full suite and creates a local `.testmondata`
dependency cache. Later runs select tests affected by changed code. Run
`uv run pytest` without `--testmon` whenever you want the complete suite.

GitHub Actions runs the full suite with branch coverage on every pull request
and every push to `master`. It also verifies formatting, linting, strict types,
the lockfile, and package builds in a clean Linux environment.

## Test layout

Tests live under `tests/` and mirror the package structure under
`src/cycle_forecast/`. Test modules use the `test_*.py` naming convention so
pytest and VS Code can discover them.

```text
src/cycle_forecast/features/cycle_features.py
tests/features/test_cycle_features.py
```

VS Code's Testing view can discover, run, and debug the entire suite or an
individual test. Use **Test: Refresh Tests** from the Command Palette if newly
created tests do not appear immediately.

## Data privacy

Raw, private, interim, and processed health data must remain local. Generated
models, experiment artifacts, credentials, and notebook outputs containing
private data must not be committed. Safe synthetic fixtures may be committed
under `data/synthetic/` when that directory is introduced.
