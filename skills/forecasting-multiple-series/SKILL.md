---
name: forecasting-multiple-series
description: >
  Forecasts multiple time series simultaneously using a global model with
  ForecasterRecursiveMultiSeries or ForecasterDirectMultiVariate. Covers data
  formats, encoding, per-series transformers, and multi-series backtesting.
  Use when the user has two or more related time series.
---

# Forecasting Multiple Time Series

## When to Use

- **ForecasterRecursiveMultiSeries**: A single global model learns patterns across many series. Each series is predicted independently but the model shares parameters. Best default for multi-series.
- **ForecasterDirectMultiVariate**: Uses values from multiple series as input features to predict one target series. Use when series are strongly correlated and influence each other.

## Data Formats

ForecasterRecursiveMultiSeries accepts three input formats:

1. **Wide DataFrame** — columns are series, index is datetime:
   ```python
   #            series_1  series_2  series_3
   # 2020-01-01     1.0       2.5       3.1
   # 2020-01-02     1.2       2.3       3.4
   ```

2. **Long DataFrame** — stacked format with a `series_id` column:
   ```python
   #            series_id  value
   # 2020-01-01  series_1    1.0
   # 2020-01-01  series_2    2.5
   ```

3. **Dictionary** — `{series_id: pd.Series}`:
   ```python
   {'series_1': pd.Series([1.0, 1.2, ...]), 'series_2': pd.Series([2.5, 2.3, ...])}
   ```

## Complete Workflow

```python
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import backtesting_forecaster_multiseries, TimeSeriesFold

# 1. Load data as wide DataFrame (columns = series)
series = pd.read_csv('data.csv', index_col='date', parse_dates=True)
series = series.asfreq('D')

# 2. Create forecaster
forecaster = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(n_estimators=200, random_state=123),
    lags=24,
    encoding='ordinal',       # 'ordinal', 'ordinal_category', 'onehot', or None
    transformer_series=None,  # Apply same transformer to all series
    # Or per-series: {'series_1': StandardScaler(), 'series_2': MinMaxScaler()}
    differentiation=None,
    dropna_from_series=False, # Set True if individual series may have NaN
)

# 3. Train
forecaster.fit(series=series)

# 4. Predict all series
predictions = forecaster.predict(steps=10)

# 5. Predict specific series only
predictions = forecaster.predict(steps=10, levels=['series_1', 'series_2'])

# 6. Backtesting (multi-series)
cv = TimeSeriesFold(
    steps=10,
    initial_train_size=len(series) - 100,
    refit=False,
)

metric, predictions_bt = backtesting_forecaster_multiseries(
    forecaster=forecaster,
    series=series,
    cv=cv,
    metric='mean_absolute_error',
    levels=None,  # Evaluate all series
)
print(metric)  # Shows per-series and aggregated metrics
```

## With Exogenous Variables

```python
# Exog can also be wide DataFrame, long DataFrame, or dict
forecaster.fit(series=series, exog=exog_df)
predictions = forecaster.predict(steps=10, exog=exog_test)
```

## ForecasterDirectMultiVariate

```python
from skforecast.direct import ForecasterDirectMultiVariate

# Predicts ONE target series using lags from ALL series as features
forecaster = ForecasterDirectMultiVariate(
    level='target_series',    # Name of the series to predict
    steps=10,
    estimator=LGBMRegressor(n_estimators=100, random_state=123),
    lags=24,                  # Or dict: {'series_a': 12, 'series_b': 24}
)
forecaster.fit(series=series_df)
predictions = forecaster.predict()
```

## Data Reshaping Utilities

```python
from skforecast.preprocessing import (
    reshape_series_wide_to_long,
    reshape_series_long_to_dict,
    reshape_exog_long_to_dict,
    reshape_series_exog_dict_to_long,
)

# Wide → Long
series_long = reshape_series_wide_to_long(series_wide)

# Long → Dict
series_dict = reshape_series_long_to_dict(series_long)
```

## Common Mistakes

1. **Mismatched series lengths**: ForecasterRecursiveMultiSeries handles different-length series if `dropna_from_series=True`.
2. **Wrong encoding for categorical regressor**: Use `encoding='ordinal_category'` with regressors that natively handle categoricals (LightGBM, CatBoost).
3. **Exog format mismatch**: Exog format (wide/long/dict) must match the series format.
4. **Forgetting `levels` parameter**: By default `predict()` forecasts all series. Use `levels` to limit predictions.
