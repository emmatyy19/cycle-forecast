# Design 001: Cycle-History Forecasting

## Status

Accepted

## Context

Phase A establishes an end-to-end forecasting and evaluation pipeline using the
smallest useful input: historical period-start dates. It provides the baseline
that future wearable-informed models must beat and creates the project's core
data, feature, training, and evaluation contracts.

This project is for learning and personal experimentation. It is not a medical
device and must not be used for diagnosis or medical decision-making.

## User need and motivation

The project is motivated by personal cycle variability that makes advance
planning difficult. In personal use, Apple Health and Oura may present possible
period-start windows spanning roughly 7 to 14 days. Anecdotally, peers with more
regular cycles have described windows closer to 5 to 7 days. These observations
are not a controlled product comparison and do not establish how either product
generates its forecasts.

The practical objective is to determine whether long-term personal history and,
later, daily wearable signals can produce a narrower useful window for planning
travel, events, and other activities. A narrower interval is valuable only when
it remains calibrated: the project must measure how often the actual start falls
inside the predicted window and must not trade honest uncertainty for apparent
precision.

## Problem statement

Given all cycle information available at the start of the current cycle,
predict the length of that cycle in whole days. Adding the predicted length to
the current period start produces a predicted next period-start date.

## Goals

- Define a clear target with an explicit prediction-time cutoff.
- Establish simple history-based baselines before training ML models.
- Prevent future information from influencing past predictions.
- Evaluate forecasts in units that are understandable to a user: days.
- Evaluate uncertainty using both interval width and empirical coverage.
- Make training and evaluation reproducible from versioned code and
  configuration.
- Preserve the privacy of all personal health data and derived datasets.

## Non-goals

- Diagnosing medical conditions or recommending care.
- Predicting fertility, ovulation, or pregnancy.
- Using wearable observations in Phase A.
- Serving a public API before the prediction contract is stable.
- Adding deep learning or operational infrastructure without evidence that it
  solves a current problem.

## Prediction contract

For cycle `t`:

```text
target_t = next_cycle_start_date_t - cycle_start_date_t
```

The target is `cycle_length_days`, represented as a positive integer. The
prediction cutoff is the start of cycle `t`. Features may use period starts and
completed cycle lengths strictly before or at that cutoff, but may not use
observations collected during the target cycle.

The initial prediction output contains:

- predicted cycle length in days
- predicted next period-start date
- model and configuration version

Prediction intervals or calibrated distributions will be added only after the
point-forecast evaluation is reliable. Their evaluation must report both how
narrow the intervals are and how frequently they contain the actual start date.

## Data contract

The private raw input is a two-column CSV stored locally at
`data/raw/cycle_history.csv`:

```csv
cycle_start_date,period_length_days
2024-01-03,6
2024-01-31,7
```

`cycle_start_date` is the first day of a period and `period_length_days` is the
number of days in that period. The loader must reject malformed dates,
non-positive or non-integer period lengths, missing values, duplicates,
nonchronological rows, unexpected columns, and explicitly defined implausible
gaps. It must also reject a period length that extends beyond the following
cycle start; this check is unavailable for the newest record until the next
cycle is recorded. It must not silently sort, deduplicate, fill, or repair
records.

The operational rationale for every current rejection, non-cleaning, and row
exclusion rule is maintained in the
[cleaning and exclusion policy](../data/cleaning-and-exclusions.md).

Only invented synthetic examples may be committed. Raw, private, interim, and
processed personal datasets remain ignored by Git because transformed
single-person health data is still identifying.

## Dataset construction

Each pair of consecutive period starts creates one completed cycle:

```text
cycle_start_date  next_cycle_start_date  cycle_length_days
2011-03-14        2011-04-11             28
2011-04-11        2011-05-10             29
```

The most recent period start has no target until the following period begins and
is excluded from supervised training rows.

Every derived dataset should carry or be accompanied by a deterministic
fingerprint based on its validated inputs and transformation version.

## Baselines

The initial evaluation compares at least:

- previous completed cycle length
- rolling mean over recent completed cycles
- rolling median over recent completed cycles
- expanding mean over all prior completed cycles

Every baseline uses targets that were complete before the prediction cutoff.
Fixed-window baselines emit no forecast until the full configured window is
available; the previous-cycle and expanding-mean baselines require one prior
completed cycle. Raw numeric predictions are retained for metric calculation.
For the operational next-start date, positive fractional predictions are
rounded to the nearest whole day with halves rounded up. The configured rolling
window is part of the forecaster name recorded with each forecast batch.

An ML model is adopted only when it improves meaningfully over the strongest
simple baseline under the same temporal evaluation.

## Features

Initial candidate features include lagged completed cycle lengths and rolling or
expanding statistics computed using historical rows only. Feature construction
must be shared between training and prediction code.

The initial fixed candidate configuration uses cycle-length lags 1, 2, and 3;
rolling means and medians over 3, 6, and 12 completed cycles; and an expanding
mean over all completed history. This requires 12 prior completed cycles before
producing a complete feature vector. Every vector carries ordered feature names,
and the identical cutoff-safe transform is used for supervised development rows
and future predictions. Supervised feature construction accepts the temporal
split and accesses development rows only; holdout targets cannot enter feature
selection or training matrices.

Calendar or trend features may be considered later, but only when their value and
prediction-time availability are explicit.

## First regularized model

The first learned model is a scikit-learn pipeline containing a
``StandardScaler`` followed by Ridge regression. The initial fixed configuration
uses an L2 regularization strength of 1.0 and requires 12 earlier supervised
feature rows before emitting a forecast. This is an initial model contract, not
a claim that those settings are optimal.

Walk-forward fitting creates a fresh pipeline at every development cutoff. Both
the scaler and Ridge estimator fit only earlier development rows, then predict
the current row from its already constructed cutoff-safe features. The current
target joins training data only at later cutoffs. This prevents the scaler's
means and variances, as well as the regression coefficients, from learning from
future rows.

The model result records its dataset fingerprint, holdout policy, feature
version and configuration, model version, regularization strength, and minimum
training history. It emits the same ``ForecastBatch`` contract as the baselines,
so the next milestone can compare them over identical dates. This stage neither
selects regularization from results nor evaluates the final holdout.

## Evaluation

Random train/test splits are prohibited. Model selection uses walk-forward
validation:

```text
train on rows before t → evaluate row t → advance one row
```

Before serious tuning begins, the most recent block of complete cycles will be
reserved as an untouched final holdout. The fixed policy reserves the 12 most
recent completed cycles, roughly one year of outcomes, and requires at least one
earlier development cycle. The partition records its policy version and the
complete dataset fingerprint. Holdout size is not runtime-configurable; changing
it requires a versioned policy change. The holdout must not be inspected during
feature or model selection and is used once after those decisions are frozen.

The primary metric is mean absolute error in days. Supporting metrics include:

- median absolute error
- root mean squared error when useful
- percentage of forecasts within ±1, ±2, ±3, and ±5 days

Metrics are reported for every baseline and model over identical evaluation
windows.

Walk-forward comparisons use the chronological intersection of prediction
cutoffs emitted by every forecaster being compared. This common window prevents
a method with a longer minimum-history requirement from being measured over a
different set of cycles. Forecast batches are fully validated before their
common dates are selected, so invalid predictions outside the overlap cannot be
silently hidden. An empty common window is valid but produces undefined metrics.
At each cutoff, the shared forecast generator supplies a predictor with only the
completed rows preceding the target cycle plus the target cycle's start date;
the row containing the eventual target is not exposed.

Point-forecast metrics use the unrounded numeric prediction. Signed error is
defined as prediction minus actual, so negative values indicate an early or
short forecast and positive values indicate a late or long forecast. Empty
forecast batches are valid when history is insufficient; their aggregate
metrics are undefined rather than zero. Forecasts must carry the same dataset
fingerprint as their actual targets and align uniquely by cycle-start cutoff.

## Reproducibility

A training run should record:

- versioned TOML configuration
- Git commit
- Python and dependency versions from `uv.lock`
- validated-data fingerprint
- feature definition and cutoff policy
- random seed, when an algorithm uses randomness
- validation windows and final metrics
- serialized model version, when an artifact is produced

MLflow or another experiment tracker may be introduced when filesystem-based run
records become difficult to compare. It is not required for the first baseline.

## Alternatives considered

### Predict the next date directly

Predicting cycle length is preferred because it provides a stable numeric target
and converts cleanly to a date using the known current period start.

### Begin with wearable data

Deferred to Phase B. A history-only system is easier to validate and supplies a
necessary benchmark for measuring the incremental value of wearables.

### Random train/test split

Rejected because future cycles could influence models evaluated on earlier
cycles, producing optimistic and operationally invalid results.

### Start with a complex model

Rejected until rolling and expanding baselines establish what additional model
complexity must improve upon.

## Open questions

- What minimum history is required before producing a forecast?
- Which rolling windows should be fixed before final evaluation?
- How large should the final temporal holdout be for the available history?
- Which data-quality thresholds should warn versus reject?
- What uncertainty representation is most useful after point forecasts are
  established?
- What coverage target provides a useful balance between planning value and
  honest uncertainty?
