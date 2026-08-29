# Project Roadmap

This roadmap tracks public, milestone-level progress. Detailed implementation
work belongs in GitHub issues and pull requests; private health data and private
planning notes never belong here.

## Foundation

- [x] Create an installable Python 3.12 project with a `src/` layout
- [x] Manage and lock dependencies with uv
- [x] Configure Ruff formatting and linting
- [x] Configure strict Pyright type checking
- [x] Configure pytest, coverage, and affected-test pre-commit checks
- [x] Configure GitHub Actions for full pull-request verification
- [x] Document repository and contribution conventions

## Phase A: Cycle-history forecasting

Predict the next cycle length using only period starts and information available
at the beginning of the current cycle. See the
[Phase A design](design/001-cycle-history-model.md) for the technical contract.

### Data foundation

- [x] Define and validate the raw period-start CSV contract
- [x] Add safe synthetic data for tests and examples
- [x] Build a typed cycle-length dataset
- [x] Record reproducible dataset fingerprints
- [x] Document every cleaning and exclusion decision

### Exploration and baselines

- [x] Explore distributions, trends, variability, and autocorrelation
- [x] Implement a previous-cycle baseline
- [x] Implement rolling-mean baselines
- [x] Implement rolling-median baselines
- [x] Implement an expanding-history baseline

### Evaluation and modeling

- [x] Implement forecasting metrics in days
- [x] Implement leakage-safe walk-forward validation
- [x] Define and preserve a final temporal holdout
- [x] Build lagged and rolling historical features
- [x] Train the first regularized regression model
- [x] Compare every model against the non-ML baselines
- [x] Quantify prediction uncertainty
- [x] Evaluate prediction-window width and empirical coverage together

### Reproducibility and delivery

- [x] Store training configuration in versioned TOML files
- [x] Record code version, data fingerprint, configuration, and metrics per run
- [x] Package the selected model and shared transformations
- [x] Expose a local prediction interface

## Phase B: Wearable-informed forecasting

Update predictions from daily wearable observations and estimate the probability
of a period starting within a future window.

- [x] Define the daily prediction target and observation cutoff in the
  [Phase B prediction contract](design/002-daily-wearable-prediction.md)
- [x] Define a privacy-safe wearable-data contract in the
  [Oura data contract](design/003-oura-data-contract.md)

### Oura ingestion and alignment

- [x] Model selected Oura API V2 responses with strict Pydantic boundary models
- [x] Implement local OAuth authorization and token refresh without exposing
  credentials
- [x] Implement an Oura API client with bounded queries, pagination, error
  handling, and response validation
- [x] Store immutable private snapshots with retrieval provenance, schema
  version, and deterministic fingerprints
- [x] Import historical Oura data through the same validated snapshot pipeline
- [x] Expose a local sync command for morning incremental retrieval; see the
  [local Oura setup guide](oura-local-sync.md)
- [x] Guide Keychain-backed Oura setup and report non-sensitive local status
- [x] Validate retrieval against a real Oura account without logging, committing,
  or uploading personal payloads
- [x] Normalize overlapping snapshots by route and document ID with explicit
  version selection: use the latest version available at an operational or
  simulated prediction cutoff, preserve corrections for leakage-safe
  backtesting, and prevent repeated historical pulls from double-counting
- [x] Align validated in-memory cycle history and daily Oura observations without
  leakage using
  the [Phase B alignment contract](design/004-wearable-alignment.md)

### Baselines and modeling

- [x] Establish wearable-informed baselines
- [x] Evaluate time-to-event and survival-analysis formulations
- [x] Produce calibrated daily probability distributions
- [x] Expose prospective and explicitly optimistic backfill evaluation through
  a private local workflow
- [x] Expand evaluation to cycle-level walk-forward folds so multiple unseen
  cycles contribute per-cycle and aggregate scores without long cycles receiving
  extra weight
- [x] Add private diagnostics for wearable missingness, outcome prevalence,
  actual-outcome probability, predicted versus actual offset, calibration,
  cycle-day performance, and per-cycle rankings
- [x] Add a temperature-focused ablation to measure whether Oura temperature
  improves on cycle timing without relying on other wearable signals
- [x] Add a conservative history-plus-temperature shadow candidate that keeps
  cycle history dominant while allowing temperature to adjust probabilities
- [x] Refine temperature features after walk-forward diagnostics identify their
  early-cycle weakness: keep history unchanged through day 10, then evaluate a
  frozen cutoff-safe temperature-trajectory adjustment
- [ ] Accumulate enough morning snapshots to evaluate at least three cycles with
  strict prospective retrieval provenance

Phase B begins only after Phase A has a reproducible evaluation pipeline and a
history-only baseline worth comparing against.

### Personal daily workflow

- [x] Add an interactive period-start recorder that validates dates, prevents
  duplicates, and safely updates one private local history file without a
  monthly spreadsheet workflow
- [ ] Package a local daily forecaster with its preprocessing, calibration, and
  version metadata; keep the cycle-history baseline as the default until a
  wearable candidate demonstrates reliable prospective improvement
- [x] Add one interactive daily command that can synchronize any Oura days
  available since the previous run, load the latest period history, and print
  today's exact-date and 1-, 3-, 7-, and 14-day probabilities
- [x] Make the daily command tolerate skipped days and multi-day synchronization
  without duplicating records or requiring daily execution
- [x] Retrain after a newly recorded period completes the previous cycle, while
  reusing the current packaged model between period starts
- [x] Journal the first forecast from each daily run privately and score resolved
  forecasts automatically with equal weight for every completed cycle
- [x] Show wearable forecasts as experimental comparisons until promotion
  criteria based on prospective cycle-level evaluation are satisfied

## Later operational milestones

- [ ] Add wearable model packaging and version metadata only after walk-forward
  and prospective evidence show reliable improvement over the cycle-history
  baseline
- [ ] Add a prediction service when a stable prediction contract exists
- [ ] Containerize the service when deployment requires it
- [x] Define monitoring signals and delayed-label evaluation
- [ ] Add scheduled retraining only after a justified retraining policy exists

Tools such as MLflow, Docker, FastAPI, and a model registry are intentionally
deferred until the project reaches the problem each tool solves.
