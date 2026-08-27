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

Model selection and calibration never consult the final temporal holdout.
Personal metrics and fitted artifacts remain local and ignored by Git.
