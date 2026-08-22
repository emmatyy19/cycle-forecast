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
- [ ] Build lagged and rolling historical features
- [ ] Train the first regularized regression model
- [ ] Compare every model against the non-ML baselines
- [ ] Quantify prediction uncertainty
- [ ] Evaluate prediction-window width and empirical coverage together

### Reproducibility and delivery

- [ ] Store training configuration in versioned TOML files
- [ ] Record code version, data fingerprint, configuration, and metrics per run
- [ ] Package the selected model and shared transformations
- [ ] Expose a local prediction interface

## Phase B: Wearable-informed forecasting

Update predictions from daily wearable observations and estimate the probability
of a period starting within a future window.

- [ ] Define the daily prediction target and observation cutoff
- [ ] Define a privacy-safe wearable-data contract
- [ ] Align cycle history and daily wearable observations without leakage
- [ ] Establish wearable-informed baselines
- [ ] Evaluate time-to-event and survival-analysis formulations
- [ ] Produce calibrated daily probability distributions

Phase B begins only after Phase A has a reproducible evaluation pipeline and a
history-only baseline worth comparing against.

## Later operational milestones

- [ ] Add model packaging and version metadata
- [ ] Add a prediction service when a stable prediction contract exists
- [ ] Containerize the service when deployment requires it
- [ ] Define monitoring signals and delayed-label evaluation
- [ ] Add scheduled retraining only after a justified retraining policy exists

Tools such as MLflow, Docker, FastAPI, and a model registry are intentionally
deferred until the project reaches the problem each tool solves.
