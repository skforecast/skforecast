---
name: prediction-intervals
description: >
  Generates prediction intervals for time series forecasts using bootstrapping,
  conformal prediction, or built-in statistical model intervals. Covers
  interval configuration, residual management, and calibration.
  Use when the user needs uncertainty quantification for forecasts.
---

# Prediction Intervals

## When to Use

Use prediction intervals to quantify forecast uncertainty. Skforecast offers three methods:

| Method | Forecasters | Description |
|--------|-------------|-------------|
| **Bootstrapping** | Recursive, Direct, RecursiveMultiSeries, DirectMultiVariate | Resample from training residuals |
| **Conformal** | All ML forecasters (incl. RNN, EquivalentDate) | Distribution-free intervals via conformal prediction |
| **Built-in** | ForecasterStats (ARIMA, ETS) | Parametric intervals from the statistical model |

### Choosing a method

- **Conformal**: distribution-free, fast, and gives a single symmetric coverage
  guarantee. The default for multi-series forecasters and the only option for
  `ForecasterRnn` / `ForecasterEquivalentDate`. Prefer it when you need a quick,
  well-calibrated interval and do not need the full predictive distribution.
- **Bootstrapping**: resamples residuals to build the full predictive
  distribution, so it also powers `predict_quantiles()` and `predict_dist()`.
  Prefer it when you need arbitrary quantiles, a fitted distribution, or
  asymmetric intervals. Needs enough stored residuals to be reliable.
- **Built-in**: parametric intervals from the statistical model itself
  (`ForecasterStats`). No residual management required.

### Related skills

- **Before**: `forecasting-single-series` / `forecasting-multiple-series` (have a fitted forecaster before adding intervals)
- **Before**: `hyperparameter-optimization` (interval calibration assumes a tuned point forecaster)
- **After**: `drift-detection` (monitor whether residual assumptions still hold once the model is in production)

## Intervals Are Quantiles (Changed in 0.23.0)

Since skforecast 0.23.0, `interval` is expressed as **quantiles in `[0, 1]`**, not
percentiles in `[0, 100]`. For example, an 80% interval is `interval=[0.1, 0.9]`
(the default is `[0.05, 0.95]` = 90%).

- `interval` also accepts a single `float` as nominal coverage: `interval=0.95`
  is equivalent to `interval=[0.025, 0.975]`.
- Passing percentiles (e.g. `[10, 90]`) still works but emits a `FutureWarning`
  and will be removed in skforecast 0.25.0.
- Mixing scales (e.g. `[0.1, 90]`) raises a `ValueError` (ambiguous scale).
- `predict_quantiles(quantiles=...)` was already on the 0-1 scale and is unchanged.

## Stop Conditions

Scan before writing code. Each row lists a rule, the symptom when it is broken, and the recovery. Full pitfall catalog: the `troubleshooting-common-errors` skill.

| Rule | Symptom | Recovery |
|------|---------|----------|
| Fit with `store_in_sample_residuals=True` before `predict_interval(method='bootstrapping')` | `No in-sample residuals stored` | Refit: `forecaster.fit(y=y_train, store_in_sample_residuals=True)` |
| Multi-series forecasters default to `method='conformal'`, not `'bootstrapping'` | Bootstrapping silently not applied on `ForecasterRecursiveMultiSeries` / `ForecasterDirectMultiVariate` | Pass `method='bootstrapping'` explicitly when that is what you want |
| ML forecasters accept `interval=`; only `ForecasterStats` accepts `alpha=` | `TypeError` / unexpected argument | Use `interval=[lower, upper]` for ML forecasters |
| `interval` must be quantiles in `[0, 1]`, not percentiles | `FutureWarning` (percentiles deprecated, removed in 0.25.0) | Use `interval=[0.1, 0.9]` instead of `interval=[10, 90]` |
| Do not mix quantile and percentile scales in one `interval` | `ValueError`: scale is ambiguous | Use a single scale, e.g. `interval=[0.05, 0.95]` |

## Bootstrapping Method

```python
from skforecast.recursive import ForecasterRecursive
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=24,
)

# IMPORTANT: store_in_sample_residuals=True is required for bootstrapping
forecaster.fit(y=y_train, store_in_sample_residuals=True)

predictions = forecaster.predict_interval(
    steps=10,
    interval=[0.1, 0.9],             # Quantiles → 80% interval (default is [0.05, 0.95] = 90%)
    method='bootstrapping',
    n_boot=250,                      # Number of bootstrap samples (default)
    use_in_sample_residuals=True,    # Use training residuals
    use_binned_residuals=True,       # Better calibration: residuals binned by prediction level
    random_state=123,
)
# Returns: DataFrame with columns [pred, lower_bound, upper_bound]
```

## Conformal Prediction

```python
# IMPORTANT: store_in_sample_residuals=True is also required for conformal
forecaster.fit(y=y_train, store_in_sample_residuals=True)

predictions = forecaster.predict_interval(
    steps=10,
    interval=[0.1, 0.9],
    method='conformal',
    use_in_sample_residuals=True,    # Uses in_sample_residuals_ stored during fit
    use_binned_residuals=True,
)
```

## Statistical Model Intervals

```python
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima

forecaster = ForecasterStats(
    estimator=Arima(order=(1, 1, 1), seasonal_order=(1, 1, 1), m=12)
)
forecaster.fit(y=y_train)

# Uses parametric intervals from statsmodels — different interface
predictions = forecaster.predict_interval(
    steps=12,
    interval=[0.1, 0.9],       # Quantiles → 80% interval. Equivalently, alpha=0.2
)
```

## Foundation Model Intervals

`ForecasterFoundation` returns intervals and quantiles directly from the underlying foundation model's native quantile output — no bootstrapping or conformal calibration required.

```python
from skforecast.foundation import FoundationModel, ForecasterFoundation

forecaster = ForecasterFoundation(
    estimator=FoundationModel(model_id='autogluon/chronos-2-small')
)
forecaster.fit(series=y_train)

predictions = forecaster.predict_interval(steps=24, interval=[0.1, 0.9])
predictions = forecaster.predict_quantiles(
    steps=24, quantiles=[0.1, 0.5, 0.9]
)
```

TimesFM 2.5 and Moirai-2 restrict quantiles to `[0.1, 0.2, …, 0.9]`; Chronos-2, TabICL, TabPFN-TS and TFC-T0 accept any quantile in `(0, 1)`. See the `foundation-forecasting` skill for details.

## During Backtesting

```python
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

cv = TimeSeriesFold(
    steps=10,
    initial_train_size=len(y_train),
    refit=False,
)

metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric='mean_absolute_error',
    interval=[0.1, 0.9],
    interval_method='bootstrapping',   # or 'conformal'
    n_boot=250,
    use_in_sample_residuals=True,
    use_binned_residuals=True,
)
# predictions has columns: pred, lower_bound, upper_bound
```

## Multi-Series Intervals

```python
from skforecast.recursive import ForecasterRecursiveMultiSeries

# NOTE: default method for multi-series is 'conformal', not 'bootstrapping'
predictions = forecaster_multi.predict_interval(
    steps=10,
    levels=['series_1', 'series_2'],
    interval=[0.1, 0.9],
    method='conformal',
)
```

## Out-of-Sample Residuals (Better Calibration)

```python
# For better interval calibration, use out-of-sample residuals
# First, compute them via backtesting
metric, predictions_bt = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric='mean_absolute_error',
)

# Set out-of-sample residuals on the forecaster
forecaster.set_out_sample_residuals(
    y_true=data['target'].loc[predictions_bt.index],
    y_pred=predictions_bt['pred'],
)

# Now use them for intervals
predictions = forecaster.predict_interval(
    steps=10,
    interval=[0.1, 0.9],
    method='bootstrapping',
    use_in_sample_residuals=False,  # Use out-of-sample residuals
)
```

## Evaluating Interval Quality

```python
from skforecast.metrics import calculate_coverage

coverage = calculate_coverage(
    y_true=y_test,
    lower_bound=predictions['lower_bound'],
    upper_bound=predictions['upper_bound'],
)
print(f"Coverage: {coverage:.2%}")  # Should be close to 0.80 for [0.1, 0.9] interval
```

## Beyond Intervals

For bootstrapping-capable ML forecasters (Recursive, Direct, RecursiveMultiSeries,
DirectMultiVariate), two richer probabilistic outputs build on the same residual
resampling:

```python
# Arbitrary quantiles — returns one column per quantile (q_0.05, q_0.5, q_0.95)
predictions = forecaster.predict_quantiles(
    steps=10,
    quantiles=[0.05, 0.5, 0.95],
    n_boot=250,
)

# Fit a scipy distribution to the bootstrapped predictions
from scipy.stats import norm

predictions = forecaster.predict_dist(
    steps=10,
    distribution=norm,   # any scipy.stats distribution
    n_boot=250,
)
```

See [references/interval-compatibility.md](references/interval-compatibility.md) for the full method/forecaster support matrix.

## Common Mistakes

1. **Passing percentiles instead of quantiles**: Since 0.23.0, `interval` is on the 0-1 scale. Use `interval=[0.1, 0.9]`, not `[10, 90]` (percentiles are deprecated and emit a `FutureWarning`, removed in 0.25.0). Mixing scales, e.g. `[0.1, 90]`, raises a `ValueError`.
2. **Forgetting `store_in_sample_residuals=True`**: Required in `fit()` before using `predict_interval(method='bootstrapping')`.
3. **Wrong default method for multi-series**: `ForecasterRecursiveMultiSeries` and `ForecasterDirectMultiVariate` default to `method='conformal'`, not `'bootstrapping'`.
4. **Mixing `alpha` and `interval`**: `ForecasterStats` supports both `alpha` (e.g., `alpha=0.05` for 95% interval) and `interval=[lo, hi]` (quantiles). ML forecasters only support `interval`.
5. **Not evaluating coverage**: Always check if actual coverage matches nominal interval width.

## References

See [references/interval-compatibility.md](references/interval-compatibility.md) for the complete compatibility matrix: which forecaster supports which method, parameter differences, binned residuals support, and common error patterns.
