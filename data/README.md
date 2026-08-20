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
