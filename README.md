# Cycle Forecast

An end-to-end machine learning project for forecasting menstrual cycle timing
from cycle history and, eventually, wearable data.

This project is for personal experimentation and is not a medical device. It
must not be used for diagnosis or medical decision-making.

## Motivation

My cycles vary enough that planning around a single expected date is difficult,
even though the variation is not extreme. In my own use, apps such as Apple
Health and Oura may show possible period-start windows spanning roughly 7 to 14
days, while some peers with more regular cycles report windows closer to 5 to 7
days. This is an anecdotal observation, not a controlled comparison of those
products.

A narrower, trustworthy window would make it easier to plan travel, events, and
other activities in advance. The goal is not to create false certainty or force
a narrow estimate: any reduction in window size must retain honest, measured
coverage of the actual start date.

## Project status

The project is currently building Phase A: a history-only model that predicts
the next cycle length without using future information. See the
[project roadmap](docs/roadmap.md) for milestone progress and the
[Phase A design](docs/design/001-cycle-history-model.md) for the prediction,
data, and evaluation contracts.

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
Code prompts for them. Explicit saves also run Ruff's import organizer, using
the same isort-compatible `I` rules enforced by pre-commit and CI.

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
and every push to `main`. It also verifies formatting, linting, strict types,
the lockfile, and package builds in a clean Linux environment.

## Local prediction

Keep private cycle history under `data/raw/` or `data/private/`. Run the command
with no arguments for a guided menu that can train a model or make a prediction
through numbered choices:

```bash
uv run cycle-forecast
```

In VS Code, the same menu is available under **Run and Debug** as
**Cycle Forecast: Interactive**. It uses the integrated terminal so file
selection and confirmation prompts work normally. **Cycle Forecast: Train** is
also available as a direct training shortcut.

### Record a period without editing a spreadsheet

Run the guided command and select your existing private history file:

```bash
uv run cycle-forecast period-record
```

The command defaults to today, previews the change, and asks for confirmation.
It records a new period with an unknown duration while it is ongoing instead of
inventing a value. Run it again with that same start date to fill in the final
duration, or enter the duration when the next period starts. All updates use a
validated temporary file and atomic replacement.

For a repeatable explicit update:

```bash
uv run cycle-forecast period-record \
  --history data/raw/cycle_history.csv \
  --date 2026-08-27 \
  --yes
```

If the prior period is still pending when adding the next start, the guided
flow asks for its duration. Scripts can supply it with
`--previous-period-length`. In VS Code, use **Cycle Forecast: Record Period** in
Run and Debug.

### One-command daily check-in

The primary personal workflow synchronizes Oura through today, asks whether a
new period started, and immediately produces updated period-start probabilities:

```bash
uv run cycle-forecast daily \
  --history data/raw/cycle_history.csv \
  --timezone America/Los_Angeles
```

The command safely overlaps the last retrieved Oura date, so it tolerates
skipped days and captures later corrections without duplicating normalized
observations. On the first-ever Oura sync, also provide the historical retrieval
start with `--start-date YYYY-MM-DD`; later runs infer it from validated local
snapshots.

If no newer period has started and the latest period still has an unknown
duration, the daily check-in asks whether it has ended. Answer yes to enter the
inclusive number of bleeding days; the existing start is updated atomically
without creating a duplicate cycle. The prompt rejects a duration that would
extend beyond the current date. Answer no while it is still ongoing.

The forecast reports the chance of a period starting today and within 3, 7, and
14 days. It also prints one explicitly labeled longer-range point estimate. That
estimate uses `artifacts/selected-model.json` from Phase A when available and
otherwise falls back to the median completed cycle length; the single date is a
planning guess, not a confidence window. The short-range probabilities use the
strongest evaluated cycle-history probability baseline. Synchronized wearable
data remains available for local learning and evaluation, but wearable models
stay explicitly experimental until prospective evidence supports promotion. In
VS Code, use **Cycle Forecast: Daily** in Run and Debug.

### Encrypted private backup and restore

Create a portable encrypted bundle containing validated cycle history, Oura
snapshots, and the prospective forecast journal:

```bash
uv run cycle-forecast private-backup \
  --history data/raw/cycle_history.csv \
  --output /Volumes/EncryptedBackup/cycle-forecast.cfbackup
```

The password is requested twice without echoing. It is never stored, and it
cannot be recovered by the application, so keep it in a password manager. The
bundle uses authenticated AES-GCM encryption with a scrypt-derived key. OAuth
credentials and regenerable model artifacts are deliberately excluded. An
existing bundle is preserved unless `--replace` is supplied. See the
[private backup design](docs/design/006-private-backup.md) for the format,
threat model, and restore guarantees.

To verify and restore the bundle to the standard local paths:

```bash
uv run cycle-forecast private-restore \
  --input /Volumes/EncryptedBackup/cycle-forecast.cfbackup
```

Restore authenticates the complete encrypted bundle, verifies every manifest
checksum, and validates the cycle history, Oura snapshots, and forecast journal
before writing. It refuses to replace any existing destination file unless
`--replace` is supplied explicitly. Restore does not delete extra local
snapshots that are absent from the bundle.

Before predicting, the daily workflow fingerprints the current history. It
reuses `artifacts/selected-model.json` when that fingerprint is current and
otherwise retrains complete replacement artifacts in a temporary directory.
Model selection and reported development metrics remain leakage-safe; after
selection, the chosen Ridge configuration is refit on all completed cycles for
operational prediction. A new period start therefore teaches the point-estimate
model one newly completed cycle, while ordinary daily runs skip retraining.

After predicting, the workflow stores the first forecast made on each local
date in `data/private/forecast-journal.jsonl`. The ignored journal uses
owner-only permissions and includes forecast probabilities, the Phase A point
estimate, model/data provenance, and the Oura retrieval bound; repeated runs on
the same date preserve the original forecast. Once a later period start resolves
those forecasts, `daily` reports cycle-weighted prospective log loss, Brier
score, and point-estimate MAE. Until then it shows that prospective performance
is waiting for a future period start.

The same check-in prints and journals a wearable nearest-neighbor forecast in
clearly labeled experimental shadow mode, together with a stage-aware hybrid.
The hybrid returns cycle history unchanged through cycle day 10. Beginning on
day 11, it keeps 75% of the history probabilities and gives 25% weight to a
candidate using recent cutoff-safe Oura temperature levels, slope, maximum
drop, and elevated-night streak. Cycle history remains the official forecast,
and unavailable wearable data never blocks it. After a later period start
resolves the forecasts, the journal reports their equal-cycle scores for direct
prospective comparisons.

The training option uses `configs/phase_a.toml`, keeps the final temporal
holdout reserved, and writes `artifacts/selected-model.json` plus
`artifacts/training-run.json`. It refuses to replace them without confirmation.
The equivalent repeatable command is:

```bash
uv run cycle-forecast train --history data/raw/cycle_history.csv
```

Use `--replace` when intentionally updating existing artifacts. To predict with
explicit paths:

```bash
uv run cycle-forecast predict \
  --model artifacts/selected-model.json \
  --history data/raw/cycle_history.csv
```

Add `--json` to that direct command for machine-readable output. Prediction is
fully local: the newest CSV record is treated as the current period start, and
the command neither uploads data nor writes a forecast file.

## Local wearable evaluation

Compare the daily history baseline, wearable nearest-neighbor baseline, and
calibrated discrete-time survival model using private cycle history and Oura
snapshots. Run `uv run cycle-forecast` and choose **Evaluate wearable models**
for the guided workflow. In VS Code, choose **Cycle Forecast: Evaluate
Wearables** from Run and Debug; it prompts for history, timezone, and evaluation
mode. The equivalent repeatable command is:

```bash
uv run cycle-forecast wearable-evaluate \
  --history data/raw/cycle_history.csv \
  --timezone America/Los_Angeles
```

The default `prospective` mode uses only real snapshot retrieval cutoffs. It
will refuse evaluation until at least three cycles contain enough proven daily
observations for separate training, calibration, and evaluation partitions.

An exploratory analysis can use the current historical backfill as though each
record were available at 9:00 AM on its source day:

```bash
uv run cycle-forecast wearable-evaluate \
  --history data/raw/cycle_history.csv \
  --timezone America/Los_Angeles \
  --mode exploratory-backfill
```

This mode is intentionally labeled optimistic: historical snapshots cannot
prove when records originally arrived or whether Oura later corrected them.
Use `--prediction-hour`, `--as-of-date`, and `--neighbors` to record alternate
assumptions explicitly. Add `--json` for private local scripting. The command
prints aggregate counts and metrics but never writes health values or fitted
artifacts.

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

The committed cycle-history exploration notebook and documented results use
only independently invented synthetic data. Private analysis runs the identical
validation, dataset-construction, statistics, and plotting code locally, but its
figures, statistics, fingerprints, and notebook output must remain uncommitted.
