# Design 004: Leakage-Safe Oura and Cycle-History Alignment

## Status

Accepted

## Contract

Each Phase B row represents one morning prediction. It records a local
`prediction_date`, a timezone-aware `prediction_cutoff`, the most recent period
start strictly before that date, a one-based cycle day, and the Oura observation
for that date if it was demonstrably available by the cutoff.

A period start on the prediction date is the outcome for the preceding cycle. It
must not be selected as the current cycle start, because doing so would expose
the label while constructing features. This makes the actual start date eligible
for a final "starts today" prediction.

An Oura observation is eligible only when its first recorded retrieval time is
no later than the prediction cutoff. A completed main sleep must also end no
later than that cutoff. Matching source dates alone never prove historical
availability.

## Corrections and historical backfill

Oura documents are immutable within a validated retrieval snapshot, while later
snapshots may contain corrected versions. Alignment requires at most one resolved
observation per source day and cutoff. An ambiguous duplicate is rejected rather
than resolved by input order.

A historical backfill retrieved today cannot prove that its records or corrected
values were available at old morning cutoffs. It can support exploration, but it
is excluded from leakage-safe backtesting unless independent snapshot provenance
establishes availability. Daily snapshots collected going forward create that
provenance.

## Boundary and domain separation

Oura JSON is first validated by strict Pydantic models matching
[Oura OpenAPI 1.35](https://api.ouraring.com/v2/static/json/openapi-1.35.json).
The alignment layer consumes project-owned observations that pair those validated
documents with retrieval provenance. This separation keeps upstream schema
validation distinct from temporal and forecasting invariants.

Missing Oura data remains an explicit absent observation. Alignment does not
impute, interpolate, or convert missing values to zero; later baseline and
feature contracts must choose cutoff-safe missingness behavior.
