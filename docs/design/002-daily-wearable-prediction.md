# Design 002: Daily Wearable-Informed Prediction Contract

## Status

Accepted

## Context

Phase B updates the Phase A history-only forecast as daily wearable observations
arrive. This document fixes the prediction target and information cutoff before
the wearable-data schema, features, baselines, or models are selected. Fixing
those semantics first prevents later implementation choices from changing the
question being evaluated.

This project is for learning and personal experimentation. It is not a medical
device and must not be used for diagnosis, fertility decisions, or other medical
decision-making.

## Daily prediction contract

One prediction is made each local calendar morning after the user opens the Oura
app and completes a Ring sync. For prediction date `D`, the observation cutoff
is the recorded instant when that morning's data retrieval begins. The completed
main sleep ending before that instant may be used, including observations from
the morning of `D`. Observations collected after the cutoff may not be used.
Cycle-history information recorded before the cutoff may also be used.

The event is the locally recorded calendar date of the first bleeding day, using
the same period-start definition as Phase A. The model produces a discrete
probability distribution over these mutually exclusive outcomes:

- the period starts today (`D`)
- the period starts on each of the following 14 calendar days (`D + 1` through
  `D + 14`)
- the period starts after `D + 14`

Including the final outcome makes the distribution exhaustive rather than
silently forcing all probability mass into the 15 reported dates. The daily
probabilities must be non-negative and sum to one.

For planning-oriented presentation and evaluation, the same distribution also
produces cumulative probabilities that the period starts within the next 1, 3,
7, and 14 days. Each window includes today, so an `N`-day window covers `D`
through `D + N - 1`. These values are derived from the daily distribution and
are not independently predicted.

## Cutoff and leakage rules

- Every prediction records its timezone-aware cutoff instant. A wearable value
  may enter the prediction only when its measurement interval ends no later than
  that cutoff and the record was available from Oura when retrieval began.
- Oura's completed main sleep may enter the morning prediction even when Oura
  assigns it to day `D` or part of its measurement interval falls on `D`.
- Late-arriving or retrospectively corrected observations may be used only if a
  reproducible snapshot or availability record proves they were present at the
  historical cutoff. Otherwise evaluation must treat them as unavailable.
- The label for a prediction remains unknown until the first bleeding day is
  recorded or the complete 14-day horizon passes. It must not enter features,
  preprocessing, imputation, calibration, or model fitting at that cutoff.
- Predictions stop once the period start is recorded. A later daily prediction
  must not be generated for the cycle that has already ended.
- Calendar dates and cutoffs use one explicit local timezone per dataset. UTC
  conversion must not move observations or period starts across local dates.
- Missing observations remain distinguishable from observed values. The future
  data contract will define whether and how a cutoff-safe model handles them.

## Evaluation implications

Backtesting must recreate every morning cutoff and expose only information that
was available then. Daily distribution quality and calibration must be assessed,
not merely whether the most likely date was correct. The exact scoring rules,
minimum history, censoring policy, and comparison window will be fixed with the
wearable-informed baseline and modeling designs.

The 1-, 3-, 7-, and 14-day cumulative outcomes are positive when the recorded
start falls within the corresponding inclusive window. Predictions whose full
outcome cannot be established at the end of a dataset must not be treated as
negative examples; the later evaluation design must exclude them or handle them
with an explicit censoring method.

## Deferred decisions

The following belong to subsequent Phase B roadmap items:

- wearable sources, fields, units, timestamps, and privacy-safe file format
- rules for assigning overnight measurements to local calendar dates
- alignment, missingness, and correction handling supported by available data
- baseline definitions, statistical formulation, calibration, and scoring rules
- the operational scheduling mechanism for the morning run

Changing the event definition, cutoff, horizon, or inclusive-window semantics
requires a new version of this contract and explicit comparability handling.
