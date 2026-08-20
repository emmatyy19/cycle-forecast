# Cycle-History Cleaning and Exclusion Policy

## Status and scope

This document records every data-quality, cleaning, and supervised-row exclusion
decision currently applied to the Phase A cycle-history dataset. It covers the
raw CSV loader and the `cycle-dataset-v1` transformation. It is a reproducibility
contract, not a medical definition of plausible menstruation.

The governing principle is fail-fast validation: questionable source data is
rejected for manual review rather than silently cleaned. Accepted records retain
their values and chronological order.

## Raw-file validation decisions

The loader applies the following decisions before dataset construction:

| Input condition | Decision | Rationale |
| --- | --- | --- |
| File is missing or unreadable | Reject the file | A partial or substituted dataset must not be used silently. |
| File is empty or lacks the exact header | Reject the file | The schema must be explicit and stable. |
| Header differs from `cycle_start_date,period_length_days` | Reject the file | Missing, reordered, or unexpected columns can change field meaning. |
| File contains no data records | Reject the file | A raw history must contain at least one observation, even though one observation cannot yet form a target. |
| A row is blank, has a missing value, or does not contain exactly two values | Reject the file | Missing or extra values require source review rather than inference. |
| A start date is not canonical ISO `YYYY-MM-DD` | Reject the file | Strict formatting avoids locale and parsing ambiguity. |
| A period length is not a canonical positive whole number | Reject the file | Zero, negative, fractional, padded, or otherwise ambiguous values violate the source contract. |
| A start date is duplicated | Reject the file | Two cycles cannot have the same unique start under this contract. |
| Start dates are not strictly increasing | Reject the file | Temporal order is meaningful and must come from the source rather than hidden sorting. |
| Consecutive starts are fewer than 15 days apart by default | Reject the file | The threshold is a conservative data-entry safeguard. It is configurable explicitly because it is not a medical boundary. |
| A period length extends beyond the following start | Reject the file | Overlapping recorded periods and cycle starts are internally inconsistent under the Phase A contract. |
| The newest period length has no following start | Accept without the overlap check | The information needed for that consistency check does not exist yet. The check occurs after a later start is added. |

Validation is all-or-nothing. The loader returns the complete validated history
or raises an error; it does not return a partially accepted dataset.

## Cleaning decisions

No automatic cleaning is currently performed. Specifically, the pipeline does
not:

- sort or reorder records
- deduplicate starts
- fill or impute missing dates or period lengths
- correct malformed values
- merge nearby starts
- infer unrecorded cycles
- truncate or winsorize cycle lengths
- remove statistical outliers
- apply a maximum cycle-length threshold

These choices preserve an auditable boundary between source corrections and
modeling. A rejected record must be reviewed and corrected at its private source
when appropriate, then the entire file must be validated again.

## Supervised-dataset construction and exclusions

For each consecutive pair of validated starts, `cycle-dataset-v1` creates one
row whose target is the calendar-day difference between those starts:

```text
cycle_length_days = next_cycle_start_date - cycle_start_date
```

The transformation makes these inclusion and exclusion decisions:

| Condition | Decision | Rationale |
| --- | --- | --- |
| A start has a following observed start | Include one completed-cycle row | Both dates required to calculate the target are observed. |
| The newest start has no following observed start | Exclude it as its own supervised row | Its cycle-length target is not known; inventing one would create a false label and future leakage. |
| History contains zero or one record in the in-memory construction API | Produce zero rows | No consecutive pair exists. The CSV loader itself rejects zero records. |
| A cycle is unusually long but passed configured raw validation | Include it | There is currently no statistical or maximum-length exclusion policy. |
| `period_length_days` is present on a validated record | Exclude it from the current supervised row schema, but include it in the fingerprint | Current-period duration is not known at the prediction cutoff and is not part of the cycle-length target; retaining it in provenance detects any source-data change. |

The newest start is still part of the validated input and fingerprint because it
completes the preceding cycle. Dataset rows preserve chronological order and are
never randomly reordered during construction.

## Provenance and policy changes

The dataset fingerprint covers every validated start date and period length in
order, plus the transformation version. It therefore changes when an accepted
source value is corrected, a record is added, record order changes, or the
transformation version changes.

Any future cleaning or exclusion rule must be documented here before it is used.
If the rule can change emitted rows or their meaning for the same validated
input, `CYCLE_DATASET_TRANSFORMATION_VERSION` must be incremented. Historical
experiment metadata must retain its original fingerprint and transformation
version rather than being updated to the newest values.
