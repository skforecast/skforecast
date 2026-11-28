---
name: baseline-forecasting
description: >
  Builds and evaluates simple baseline forecasts with ForecasterEquivalentDate
  (equivalent-date / seasonal-naive / moving-average) to benchmark machine
  learning and foundation models. Covers offset (int vs pandas DateOffset), n_offsets with
  agg_func, backtesting a baseline, comparing against ML with MASE, conformal
  intervals, and grid_search_equivalent_date. Use when the user wants a baseline,
  a naive or seasonal reference, or to check whether a model beats a simple rule.
  Not for choosing among ML forecasters (see choosing-a-forecaster) or tuning ML
  hyperparameters (see hyperparameter-optimization).
---

# Baseline Forecasting

## When to Use

Use this skill to build a **simple, explainable baseline** with
`ForecasterEquivalentDate` and to check whether a more complex model actually
beats it. A baseline predicts each future step from an equivalent past date
(e.g. same weekday last week), optionally aggregating several past occurrences.

Always establish a baseline **before** trusting a model, machine learning or
foundation: if it cannot beat the naive baseline, the added complexity is not
justified.

### Related skills

- **Prerequisite**: `choosing-a-forecaster` (the first comparison: LightGBM, Chronos-2 and this baseline)
- **Alongside**: `forecasting-single-series` (train the ML model you are benchmarking against)
- **Next**: `metric-selection` (use MASE to compare the model against the naive baseline)
- **Next**: `hyperparameter-optimization` (route: baselines are tuned with `grid_search_equivalent_date`, not the generic search)
- **Next**: `prediction-intervals` (add conformal intervals to the baseline)

## Stop Conditions

Scan before writing code. Each row lists a rule, the symptom when it is broken, and the recovery.

| Rule | Symptom | Recovery |
|------|---------|----------|
| Set the index frequency before fitting | `ValueError: ... DatetimeIndex with frequency` | Call `y = y.asfreq('D')` (or the correct alias) |
| A pandas `DateOffset` offset requires a `DatetimeIndex` with frequency | `TypeError: ... the index must be a pandas DatetimeIndex with frequency` | Use a `DatetimeIndex` with `asfreq`, or use an integer `offset` for a `RangeIndex` |
| `offset * n_offsets` must be smaller than the training window | `ValueError: ... offset ... is larger than the available data` (or configs silently skipped in search) | Reduce `offset`/`n_offsets`, or increase `initial_train_size` |
| Baselines have no `estimator`, `lags`, or `exog` | Error from `grid_search_forecaster` (baselines lack `set_lags`) | Tune with `grid_search_equivalent_date`; do not pass `exog` |

## Overview

`ForecasterEquivalentDate` has three modeling parameters:

- **`offset`** — how far back to look for the equivalent date. `int` (number of
  steps) or a pandas `DateOffset` (calendar-aware).
- **`n_offsets`** — how many equivalent dates to combine (default 1).
- **`agg_func`** — how to aggregate them when `n_offsets > 1` (default `np.mean`).

```python
import numpy as np
from skforecast.recursive import ForecasterEquivalentDate

# Predict each step as the value observed 7 steps earlier (seasonal-naive, weekly on daily data)
forecaster = ForecasterEquivalentDate(offset=7, n_offsets=1)
forecaster.fit(y=y_train)
predictions = forecaster.predict(steps=14)
```

## `offset`: int vs pandas `DateOffset`

This is the part users get wrong most often.

- **Integer `offset`** — a fixed number of steps back. Works with a
  `RangeIndex` or a `DatetimeIndex`. `offset=7` on daily data = "same day last
  week"; `offset=24` on hourly data = "same hour yesterday".
- **pandas `DateOffset`** — a calendar rule (e.g. `pd.offsets.Week`,
  `pd.offsets.BusinessDay`). Requires a `DatetimeIndex` with a set frequency.
  Use it when the equivalent date must follow the calendar (business days,
  month ends, holidays) rather than a fixed step count.

```python
import pandas as pd

# Integer offset: 7 steps back
forecaster = ForecasterEquivalentDate(offset=7)

# DateOffset: one calendar week back (needs a DatetimeIndex with frequency)
forecaster = ForecasterEquivalentDate(offset=pd.offsets.Week(1))
```

With a timezone-aware index that crosses a daylight saving time change, use an
integer `offset` (same results with pandas 2 and 3). On hourly data,
`pd.offsets.Day(n)` is `n * 24` hours with pandas 2 but a calendar day with
pandas 3 (different predictions after the change, or a `ValueError` for the
hour skipped in spring or repeated in autumn), and `pd.DateOffset(days=n)` can
fail or repeat hours: use `offset=24 * n` or `pd.offsets.Hour(24 * n)`. On
daily data, `offset=n` works with both versions (`pd.offsets.Day(n)` fails
with pandas 3 on `freq='24h'` and with pandas 2 on dates in local time).

## `n_offsets` and `agg_func`

Set `n_offsets > 1` to combine several equivalent dates, aggregated by
`agg_func`. `offset` and `n_offsets` are **coupled** (together they define the
window), and `agg_func` decides how the values are pooled:

```python
# Moving average of the last 7 observations: offset=1, n_offsets=7
forecaster = ForecasterEquivalentDate(offset=1, n_offsets=7, agg_func=np.mean)

# Mean of the values 7 and 14 steps back: offset=7, n_offsets=2
forecaster = ForecasterEquivalentDate(offset=7, n_offsets=2, agg_func=np.mean)
```

`agg_func` is a real, searchable parameter — do not forget it when comparing
configurations (e.g. `np.mean` vs `np.median`).

## Complete Workflow: backtest and compare against a model

```python
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold
from skforecast.metrics import mean_absolute_scaled_error

cv = TimeSeriesFold(steps=7, initial_train_size=len(y) - 70, refit=False)

metric_baseline, _ = backtesting_forecaster(
    forecaster = ForecasterEquivalentDate(offset=7, n_offsets=1),
    y          = y,
    cv         = cv,
    metric     = mean_absolute_scaled_error,
)
# Backtest the ML model with the SAME `cv` and `metric`, then compare the two
# values: the lower error wins. For the model, MASE < 1 means it beats the
# in-sample naive forecast; MASE >= 1 means it does not.
```

See `metric-selection` for why MASE is the natural yardstick against a naive
baseline.

## Several series and categorical targets

`ForecasterEquivalentDate` takes one numeric series. With **several series**,
backtest one baseline per series with the same `cv` and average the metric:

```python
import pandas as pd

# series: dict with one pandas Series per series
metric_baseline = pd.concat([
    backtesting_forecaster(
        forecaster    = ForecasterEquivalentDate(offset=7),
        y             = y_i,
        cv            = cv,
        metric        = "mean_absolute_scaled_error",
        show_progress = False,
    )[0]
    for y_i in series.values()
])["mean_absolute_scaled_error"].mean()
```

With a **categorical target** (`ForecasterRecursiveClassifier`), the baseline is a
seasonal rule: the category observed one season before. Compare it on the dates
the classifier was backtested on, with `steps` no longer than the season so that
the rule only uses values known when the forecast is made:

```python
from sklearn.metrics import balanced_accuracy_score

# predictions: second output of backtesting_forecaster for the classifier
rule = y.shift(7).loc[predictions.index]
metric_rule = balanced_accuracy_score(y.loc[predictions.index], rule)
```

## Selecting a configuration

Use `grid_search_equivalent_date` (baselines are not compatible with the generic
`grid_search_forecaster`). Because `offset`/`n_offsets` are coupled and
`agg_func` is a third searchable parameter, prefer the **list-of-configs** form
so every configuration is explicit:

```python
from skforecast.model_selection import grid_search_equivalent_date, TimeSeriesFold

cv = TimeSeriesFold(steps=7, initial_train_size=len(y) - 70, refit=False)

results = grid_search_equivalent_date(
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1),
    y          = y,
    cv         = cv,
    param_grid = [
        {'alias': '7-day moving average',     'offset': 1, 'n_offsets': 7, 'agg_func': np.mean},
        {'alias': 'mean of lag-7 and lag-14', 'offset': 7, 'n_offsets': 2, 'agg_func': np.mean},
        {'alias': 'median of lag-7 & lag-14', 'offset': 7, 'n_offsets': 2, 'agg_func': np.median},
    ],
    metric      = 'mean_absolute_error',
    return_best = True,
)
```

- `dict` form → Cartesian product (use only when every combination is
  meaningful).
- `list` form → one explicit configuration per dict (scalar values), avoiding
  meaningless `offset`/`n_offsets` combinations; each dict may carry an optional
  `alias` label that appears as the first results column.

Details and the full parameter table: `hyperparameter-optimization`
(`references/search-parameters.md`).

## Prediction intervals

`ForecasterEquivalentDate` supports **conformal** intervals only. Store
residuals at fit time, then call `predict_interval`:

```python
forecaster = ForecasterEquivalentDate(offset=7, n_offsets=1)
forecaster.fit(y=y_train, store_in_sample_residuals=True)
predictions = forecaster.predict_interval(steps=14, method='conformal', interval=[0.1, 0.9])
```

See `prediction-intervals` for the conformal method details.

## Related API

Constructor and method signatures live in the `complete-api-reference` skill
(`references/forecaster-constructors.md` and `references/forecaster-methods.md`).
This skill covers the baseline workflow and its gotchas, not the full API surface.
