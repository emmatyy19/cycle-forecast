# Design 005: Daily Survival Modeling and Evaluation

## Status

Accepted

## Dataset and labels

Each row is the cutoff-safe aligned morning observation from Design 004. Its
event offset is the number of local calendar days from the prediction date to
the next recorded period start. Offsets 0 through 14 are exact event outcomes;
an observed later start is the exhaustive `after day 14` outcome. A row with no
later start is usable as `after day 14` only when observation continues through
the end of day 14. Otherwise it is right-censored and excluded from ordinary
multiclass scores.

The modeling allowlist contains cycle day, readiness score, sleep score,
temperature deviation, average overnight HRV, total sleep duration, and an
explicit missingness indicator for every nullable wearable value. Source IDs,
raw time series, fingerprints, and future observations never enter features.
Oura's temperature deviation is retained in its documented units. Other
continuous values retain their documented units and are standardized only by
preprocessing fitted on prior training rows.

## Baselines

The daily history baseline is an empirical discrete hazard by cycle day, fitted
only from completed prior cycles. Add-one smoothing prevents zero-probability
outcomes. A wearable-informed nearest-neighbor baseline compares the current row
with prior labeled mornings using standardized cycle day and available wearable
features. Missing values match only through explicit missingness indicators.

Both baselines return probabilities for today, each of the following 14 days,
and after day 14. Sequential conditional hazards are converted into an
exhaustive distribution, so probability mass is never silently truncated.

## Model and calibration

The first Phase B model is discrete-time logistic survival regression. Training
expands each labeled morning into at-risk binary rows through its observed event
or horizon. Preprocessing is fitted inside each temporal training fold. The
model predicts one conditional event hazard per future day and converts those
hazards into the same exhaustive distribution as the baselines.

Probability calibration uses a later chronological calibration block that is
not used to fit model coefficients. Platt scaling is fitted to the block's
at-risk hazard predictions. If either calibration class is absent, calibration
fails rather than silently returning an uncalibrated model.

## Evaluation

Comparison is chronological and shared-window: every candidate is scored on the
same uncensored prediction dates. Report multiclass logarithmic loss, multiclass
Brier score, and binary Brier scores for the inclusive 1-, 3-, 7-, and 14-day
windows. Calibration tables group predicted window probabilities into fixed
bins and retain predicted mean, observed fraction, and count.

Model selection and calibration never consult an evaluation cycle. Personal
metrics and fitted artifacts remain local and ignored by Git.

## Local evaluation modes

The local workflow supports two explicit availability interpretations.
`prospective` constructs at most one row from the first real retrieval cutoff on
each local date and applies immutable snapshot provenance exactly.
`exploratory-backfill` assumes the latest normalized historical version was
available at a configured local hour on its source day. A detailed sleep ending
after that assumed cutoff is unavailable, and a morning with no remaining
wearable record is retained through missingness features. Backfill results are
always labeled optimistic and are not leakage-safe performance estimates.

Temporal partitions reserve whole completed cycles and form expanding
walk-forward folds. The earliest fold trains on the first completed cycle,
calibrates hazards on the second, and evaluates every candidate on the third.
Each later fold expands the training block by one cycle, uses the immediately
following cycle for calibration, and evaluates on the next unseen cycle. The
ongoing cycle is never an evaluation fold, even when some of its older mornings
already have observable 14-day outcomes.

Within a fold, every candidate is scored on the same mornings. Aggregate scores
first summarize each evaluation cycle and then average those cycle scores, so a
long cycle cannot outweigh a short cycle merely because it has more mornings.
The report also retains each cycle's scores and counts cycle-level wins without
printing private dates. This contract is versioned as `wearable-evaluation-v2`.

Private-safe diagnostics describe the evaluated mornings without printing dates
or health measurements. They report wearable-field missingness, observed event
prevalence for each planning window, and the fraction occurring after the
14-day horizon. For each candidate they also report the mean and minimum
probability assigned to the actual outcome, expected-offset bias and RMSE,
predicted versus observed planning-window frequencies, and exact-date Brier
scores for cycle days 1-10, 11-20, and 21 or later. The exhaustive later class
is represented as offset 15 for offset-error summaries. These diagnostic
aggregates describe mornings, while the headline model scores continue to give
each evaluation cycle equal weight. Per-cycle tables rank every candidate and
omit the private cycle dates.

## Operational daily forecast

The local `daily` workflow incrementally synchronizes every supported Oura route
through the current local date, offers the period-history recorder, and then
forecasts from the updated history in one process. Incremental retrieval
overlaps the least-complete route's latest requested date, allowing skipped
execution days and upstream corrections while immutable snapshots and
normalization prevent duplicate observations.

Until a wearable candidate meets prospective promotion criteria, the official
daily probability distribution is the empirical completed-cycle hazard
baseline. It is fitted directly from all cycle lengths completed before the
current period and reports probabilities for today, the inclusive 3-, 7-, and
14-day planning windows, and the exhaustive later outcome. Oura synchronization
still occurs first so prospective wearable evidence accumulates continuously;
the terminal report states that wearable models are not yet used for the
official forecast.

For longer-range convenience, the same report includes one separately labeled
point estimate. It prefers the selected packaged Phase A model and otherwise
uses the median completed cycle length, rounded to the nearest operational day
with halves rounded upward. This naive or model-based date does not replace the
probability distribution and is explicitly described as a planning guess rather
than a confidence window.

## Prospective forecast journal

Every successful daily workflow appends at most one immutable forecast per
local prediction date to an owner-private, git-ignored JSON Lines journal. The
first forecast is preserved on same-day reruns. Each versioned entry contains
the exhaustive history-baseline distribution, Phase A point estimate, cycle
context, model and dataset provenance, prediction cutoff, and the date through
which Oura synchronization was requested. It contains no raw wearable values.

Journal entries remain unresolved until validated cycle history contains the
following period start for their recorded current cycle. Delayed scoring uses
the same exhaustive multiclass log loss and Brier definitions as development
evaluation. Forecasts are summarized within each completed cycle first and
then averaged across cycles, preventing frequently journaled or longer cycles
from receiving extra weight. The point estimate is evaluated by absolute date
error under the same equal-cycle rule. Predictions accidentally made after a
later-reported period start are excluded because their event offset would be
negative.
