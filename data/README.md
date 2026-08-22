# Data directories

Personal menstrual-cycle data is inherently identifying and must remain local,
including cleaned or transformed versions.

## Layout

```text
data/
├── raw/          # Original private exports; ignored by Git
├── private/      # Other private source data; ignored by Git
├── interim/      # Intermediate private transformations; ignored by Git
├── processed/    # Model-ready private datasets; ignored by Git
└── synthetic/    # Non-identifying examples safe to commit
```

Do not put real health records in `synthetic/`. Synthetic files should be
invented independently rather than created by perturbing personal records.

`sample_cycle_history.csv` contains 120 completed cycles for realistic examples,
tests, and plotting density. It is generated deterministically by
`scripts/generate_synthetic_cycle_history.py`; its fixed seed and generic
parameters were chosen independently rather than fitted to personal data.

## Raw cycle-history contract

The local `data/raw/cycle_history.csv` file has exactly two columns:

```csv
cycle_start_date,period_length_days
2024-01-03,6
2024-01-31,7
```

Requirements:

- The header is exactly `cycle_start_date,period_length_days`.
- Every `cycle_start_date` is an ISO 8601 date in `YYYY-MM-DD` form.
- Every `period_length_days` is a positive whole number.
- Dates contain no missing values or duplicates.
- Dates are in strictly increasing chronological order.
- Consecutive period starts are at least 15 days apart by default. This is a
  conservative data-entry check, not a medical definition, and the loader makes
  the threshold configurable.
- A period cannot extend beyond the following cycle start. This consistency
  check cannot be applied to the newest record until the next cycle is known.

Validation failures require manual review. The loader does not sort, deduplicate,
fill, or otherwise silently modify questionable records.

The complete rationale for every current validation, non-cleaning, and
supervised-row exclusion rule is recorded in the
[cleaning and exclusion policy](../docs/data/cleaning-and-exclusions.md).

## Derived dataset provenance

Dataset construction preserves validated chronological order and creates one
target from each consecutive pair of starts. The newest start cannot produce a
target until another start is observed, so it is excluded as a supervised row.

Every constructed dataset carries a transformation version and a SHA-256
fingerprint. The fingerprint's canonical UTF-8 payload contains:

- the `cycle-forecast:cycle-dataset` domain identifier
- the transformation version
- fixed input field names
- every validated start date and period length in chronological order

File paths and original CSV formatting are excluded, so equivalent validated
records have the same identity. Any input value, record order, or transformation
version change produces a different identity. The transformation version must be
incremented whenever the construction rules or row meaning changes; equivalent
refactoring does not require a version increment.

A fingerprint identifies private data but does not anonymize it. Personal input,
derived rows, and their provenance metadata must remain private.
