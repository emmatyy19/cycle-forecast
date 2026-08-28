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
