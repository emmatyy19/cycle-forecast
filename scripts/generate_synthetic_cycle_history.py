"""Generate the independently invented long cycle-history demonstration."""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SYNTHETIC_SEED = 847_201
COMPLETED_CYCLE_COUNT = 120
FIRST_START = date(2015, 1, 4)
OUTPUT_PATH = Path("data/synthetic/sample_cycle_history.csv")


def generate_rows() -> tuple[tuple[date, int], ...]:
    """Generate deterministic, independently invented period-start records.

    Returns
    -------
    tuple[tuple[date, int], ...]
        Synthetic start dates and period lengths. There is one more start than
        completed cycle so the final start completes the preceding target.

    Notes
    -----
    Parameters are generic demonstration choices and are not estimated from or
    fitted to personal health data.
    """
    random_generator = random.Random(SYNTHETIC_SEED)
    rows: list[tuple[date, int]] = []
    current_start = FIRST_START
    for record_index in range(COMPLETED_CYCLE_COUNT + 1):
        period_length = random_generator.randint(4, 7)
        rows.append((current_start, period_length))
        if record_index == COMPLETED_CYCLE_COUNT:
            break
        cycle_length = round(random_generator.triangular(24, 39, 30))
        current_start += timedelta(days=cycle_length)
    return tuple(rows)


def main() -> None:
    """Write the deterministic synthetic history to its committed CSV path."""
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.writer(file_handle, lineterminator="\n")
        writer.writerow(("cycle_start_date", "period_length_days"))
        writer.writerows(
            (cycle_start.isoformat(), period_length)
            for cycle_start, period_length in generate_rows()
        )


if __name__ == "__main__":
    main()
