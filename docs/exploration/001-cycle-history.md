# Exploration 001: Completed Cycle History

## Purpose

This exploration describes completed cycle lengths before baseline forecasting
or model fitting. It covers four questions:

1. What is the observed distribution of cycle lengths?
2. How much have completed cycles varied?
3. Is there a simple linear trend over chronological cycle position?
4. Are adjacent cycle lengths linearly associated at lag 1?

The reusable analysis lives under `src/cycle_forecast/analysis/`; private
results and identifying output must remain local. The committed results below
use only the invented synthetic example and demonstrate interpretation rather
than making a claim about a person or population.

## Definitions

- Distribution is reported as counts for every observed whole-day length, plus
  mean, median, minimum, maximum, and interquartile range (IQR).
- Variability uses sample standard deviation because the observed history is
  treated as a sample of possible cycles. IQR uses inclusive quartiles, which
  are appropriate for describing the endpoints of a finite observed sample.
- Trend is the ordinary least-squares slope of cycle length against zero-based
  chronological cycle position, reported in days per cycle. The chronological
  plot also shows a six-cycle trailing mean by default; each point uses only the
  current and five preceding completed cycles.
- Lag-1 autocorrelation is the Pearson correlation between each cycle length and
  the immediately following length.

Statistics that the available history cannot support are `None`: sample
standard deviation, IQR, and trend require two cycles; lag-1 autocorrelation
requires at least three cycles and nonzero variance in both adjacent sequences.
A zero correlation is therefore distinguishable from an undefined one.

## Reproduce locally

After placing private data at the ignored raw-data path, run the same functions
used by tests and future pipeline code:

```python
from cycle_forecast.analysis import explore_cycle_history, plot_cycle_history
from cycle_forecast.data import build_cycle_dataset, load_cycle_history

records = load_cycle_history(path="data/raw/cycle_history.csv")
dataset = build_cycle_dataset(records=records)
exploration = explore_cycle_history(dataset=dataset)
figure = plot_cycle_history(dataset=dataset, exploration=exploration)
```

The returned exploration carries `dataset.fingerprint`, allowing locally saved
results to be tied to the exact validated inputs and transformation version.
Do not commit printed private results, plots, or notebook output.

The reusable demonstration notebook is
`notebooks/cycle_history_exploration.ipynb`. It defaults to the independently
invented 120-cycle `sample_cycle_history.csv` demonstration; set the
`CYCLE_FORECAST_DATA_PATH` environment variable to use a private local CSV
without editing notebook source. Repository hooks strip all notebook output
before commit, so rerunning it privately cannot place rendered personal results
in Git.

## Synthetic demonstration

For `data/synthetic/sample_cycle_history.csv`, 121 invented starts produce 120
completed cycles. The observed cycle-length counts are:

| Length (days) | Count |
| ---: | ---: |
| 25 | 4 |
| 26 | 7 |
| 27 | 7 |
| 28 | 10 |
| 29 | 17 |
| 30 | 17 |
| 31 | 12 |
| 32 | 11 |
| 33 | 11 |
| 34 | 9 |
| 35 | 7 |
| 36 | 4 |
| 37 | 1 |
| 38 | 3 |

Summary statistics for this invented history are:

| Statistic | Synthetic result |
| --- | ---: |
| Mean | 30.73 days |
| Median | 30 days |
| Minimum / maximum | 25 / 38 days |
| Sample standard deviation | 3.08 days |
| IQR | 4.00 days |
| Linear trend | -0.01 days per cycle |
| Lag-1 autocorrelation | -0.03 |

The synthetic history is concentrated around 30 days and has essentially no
linear trend or lag-1 correlation. Those properties are consequences of its
generic independent generator and are not evidence about real cycle behavior.
The descriptive values demonstrate the workflow; predictive relationships must
still be assessed with walk-forward evaluation.

## Modeling implications

- Distribution and variability establish the scale on which forecast errors
  will later be judged.
- A trend statistic summarizes one pattern but does not justify extrapolation;
  walk-forward evaluation must determine whether trend-aware methods help.
- Lag-1 autocorrelation motivates a previous-cycle baseline, but its predictive
  value must be tested out of sample.
- Exploration must not select or repeatedly inspect the future final holdout.
  The holdout policy will be fixed before serious model comparison begins.
