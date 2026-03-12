<!-- AUTO-GENERATED FILE. DO NOT EDIT MANUALLY. -->
<!-- Source: tools/ai/llms-base.txt + tools/ai/ai_context_header.md -->
<!-- Regenerate with: python tools/ai/generate_ai_context_files.py -->

# Skforecast — Development Context

## For Contributors Working Inside This Repository

### Testing

```bash
pytest skforecast/recursive/tests/ -vv           # Run a specific module's tests
pytest --cov=skforecast --cov-report=html         # Coverage report
pytest -n auto                                    # Parallel execution (pytest-xdist)
```

Markers: `@pytest.mark.slow` for long-running tests (skip with `-m "not slow"`).

### Code Style

- NumPy-style docstrings
- Type hints for function signatures
- PEP 8 compliant (max line length 88, enforced by ruff)
- Single quotes for strings (ruff `quote-style = "single"`)
- Relative imports within package

### Dependencies

Core: numpy>=1.26, pandas>=2.1,<3.0, scikit-learn>=1.4, scipy>=1.12, optuna>=4.0, joblib>=1.3, numba>=0.59, tqdm>=4.66, rich>=13.9
Optional: statsmodels>=0.13,<0.15 (stats), matplotlib>=3.7,<3.11 + seaborn>=0.12,<0.14 (plotting), keras>=3.0,<4.0 (deep learning)

---

# Skforecast — Complete API & Workflow Reference

(The content below is the full `llms-base.txt` and applies to any user of skforecast)

# Skforecast

> Python library for time series forecasting using machine learning models

This document is for skforecast v0.21.0+. If you are using an older version, check the documentation at skforecast.org.

Skforecast is a Python library that simplifies time series forecasting using machine learning. It works with any estimator compatible with the scikit-learn API (LightGBM, XGBoost, CatBoost, Keras, etc.).

## Quick Info

- Version: 0.21.0
- License: BSD-3-Clause
- Python: 3.10, 3.11, 3.12, 3.13, 3.14
- Repository: https://github.com/skforecast/skforecast
- Documentation: https://skforecast.org
- PyPI: https://pypi.org/project/skforecast/

## Installation

```bash
pip install skforecast
```

Optional dependencies:
```bash
pip install skforecast[stats]        # For ARIMA, SARIMAX, ETS models
pip install skforecast[plotting]     # For visualization
pip install skforecast[deeplearning] # For RNN/LSTM models
```

## Project Structure

```
skforecast/
├── base/                    # ForecasterBase - abstract parent class for all forecasters
├── recursive/               # ForecasterRecursive, ForecasterRecursiveMultiSeries,
│                            # ForecasterRecursiveClassifier, ForecasterStats, ForecasterEquivalentDate
├── direct/                  # ForecasterDirect, ForecasterDirectMultiVariate
├── deep_learning/           # ForecasterRnn, create_and_compile_model
├── stats/                   # Arima, Sarimax, Ets, Arar (sklearn-compatible wrappers)
├── preprocessing/           # TimeSeriesDifferentiator, RollingFeatures, DateTimeFeatureTransformer,
│                            # QuantileBinner, ConformalIntervalCalibrator, reshape_* functions
├── model_selection/         # backtesting_forecaster, grid/random/bayesian search, TimeSeriesFold
├── feature_selection/       # select_features, select_features_multiseries
├── metrics/                 # MASE, RMSSE, sMAPE, CRPS, coverage, pinball loss
├── datasets/                # 30+ built-in datasets (fetch_dataset, load_demo_dataset)
├── drift_detection/         # RangeDriftDetector, PopulationDriftDetector
├── utils/                   # Shared validation and transformation functions
├── exceptions/              # Custom warnings and exceptions
├── plot/                    # plot_residuals, plot_prediction_intervals, plot_prediction_distribution,
│                            # plot_multivariate_time_series_corr, set_dark_theme, backtesting_gif_creator
└── experimental/            # Experimental features (API may change)
```

### Module Relationships

- **Forecasters inheriting from `ForecasterBase`**: ForecasterRecursive, ForecasterRecursiveMultiSeries, ForecasterRecursiveClassifier, ForecasterDirect, ForecasterDirectMultiVariate, ForecasterRnn
- **Standalone forecasters (no inheritance)**: ForecasterStats, ForecasterEquivalentDate
- Statistical models in `stats/` are wrapped by `ForecasterStats` (in `recursive/`)
- `model_selection/` functions work with all forecaster types
- `preprocessing/` classes can be passed to forecasters via `transformer_y`, `transformer_exog`, `window_features`

## Core Forecasters

| Forecaster | Use Case |
|------------|----------|
| ForecasterRecursive | Single series, recursive multi-step forecasting |
| ForecasterDirect | Single series, direct multi-step forecasting |
| ForecasterRecursiveMultiSeries | Multiple series forecasting (global model) |
| ForecasterDirectMultiVariate | Multivariate forecasting (multiple series as features) |
| ForecasterRnn | Deep learning (RNN/LSTM) forecasting |
| ForecasterStats | Statistical models (ARIMA, SARIMAX, ETS, ARAR) |
| ForecasterRecursiveClassifier | Classification-based forecasting |
| ForecasterEquivalentDate | Baseline forecaster using equivalent past dates |

## Basic Usage Example

```python
# Single series forecasting with ForecasterRecursive
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

# Load data — y must be a pandas Series with DatetimeIndex and frequency set
data = pd.read_csv('data.csv', index_col='date', parse_dates=True)
data = data.asfreq('h')  # IMPORTANT: always set frequency before using skforecast

# Create and train forecaster
forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=24  # Use last 24 observations as features
)
forecaster.fit(y=data['target'])

# Predict next 10 steps
predictions = forecaster.predict(steps=10)

# Define cross-validation strategy for backtesting
cv = TimeSeriesFold(
    steps=10,
    initial_train_size=len(data) - 100,
    refit=False,
    fixed_train_size=False
)

# Backtesting for model evaluation
metric, predictions_backtest = backtesting_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    metric='mean_absolute_error'
)
```

## Multi-Series Forecasting Example

```python
# Multiple series with global model
from skforecast.recursive import ForecasterRecursiveMultiSeries
from lightgbm import LGBMRegressor

# Data: DataFrame with multiple series as columns
# series = pd.DataFrame({'series_1': [...], 'series_2': [...], ...})

forecaster = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(n_estimators=100, random_state=123),
    lags=24,
    encoding='ordinal'  # 'ordinal', 'ordinal_category', 'onehot', or None
)
forecaster.fit(series=series)

# Predict all series
predictions = forecaster.predict(steps=10)

# Predict specific series
predictions = forecaster.predict(steps=10, levels=['series_1', 'series_2'])
```

## With Exogenous Variables

```python
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24
)

# Fit with exogenous variables
forecaster.fit(y=y_train, exog=exog_train)

# Predict - exog must cover the forecast horizon
predictions = forecaster.predict(steps=10, exog=exog_test)
```

## Window Features (Rolling Statistics)

```python
from skforecast.preprocessing import RollingFeatures

# Create rolling features
rolling_features = RollingFeatures(
    stats=['mean', 'std', 'min', 'max'],
    window_sizes=7  # int applies to all stats, or list with same length as stats
)

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    window_features=rolling_features
)
```

## Prediction Intervals

```python
# Predict with confidence intervals
predictions = forecaster.predict_interval(
    steps=10,
    interval=[10, 90],  # 80% prediction interval
    method='bootstrapping',  # or 'conformal'
    n_boot=500
)
# Returns DataFrame with columns: pred, lower_bound, upper_bound
```

## Backtesting

Backtesting evaluates forecaster performance using time series cross-validation.

```python
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

# Define cross-validation strategy
cv = TimeSeriesFold(
    steps=10,
    initial_train_size=len(data) - 100,
    refit=False,
    fixed_train_size=False
)

metric, predictions = backtesting_forecaster(
    forecaster=forecaster,                      # Forecaster object to evaluate
    y=data['target'],                           # Time series data (pandas Series with DatetimeIndex)
    cv=cv,                                      # TimeSeriesFold with CV configuration
    metric='mean_absolute_error',               # Metric(s): str, callable, or list
    exog=exog,                                  # Exogenous variables (optional)
    interval=[10, 90],                          # Prediction intervals as percentiles (optional)
    interval_method='bootstrapping',            # 'bootstrapping' or 'conformal'
    n_boot=250,                                 # Bootstrap iterations (only if method='bootstrapping')
    use_in_sample_residuals=True,               # Use training residuals for intervals
    use_binned_residuals=True,                  # Select residuals based on predicted values
    random_state=123,                           # Seed for reproducibility
    return_predictors=False,                    # Return predictor values used in each fold
    n_jobs='auto',                              # Parallel jobs (-1 for all cores, 'auto' for automatic)
    verbose=False,                              # Print fold information
    show_progress=True,                         # Show progress bar
    suppress_warnings=False                     # Suppress skforecast warnings
)
```

## Cross-Validation Strategies

Skforecast provides two cross-validation strategies for time series: `TimeSeriesFold` for multi-step ahead validation and `OneStepAheadFold` for one-step ahead validation.

### TimeSeriesFold

Class to split time series data into train and test folds for backtesting and hyperparameter search.

```python
from skforecast.model_selection import TimeSeriesFold

cv = TimeSeriesFold(
    steps=12,                        # (required) Number of observations to predict in each fold (forecast horizon)
    initial_train_size=100,          # Number of observations for initial training. Can be int, str (date), or pd.Timestamp
    fold_stride=None,                # Observations between consecutive test set starts. If None, equals steps (no overlap)
    window_size=None,                # Observations needed for autoregressive predictors (set automatically by forecaster)
    differentiation=None,            # Differencing order to extend last_window (set automatically by forecaster)
    refit=False,                     # Whether to refit forecaster each fold: True, False, or int (refit every n folds)
    fixed_train_size=True,           # If True, training size is fixed; if False, expands each fold
    gap=0,                           # Observations between end of training and start of test set
    skip_folds=None,                 # Folds to skip: int (every n-th fold) or list of fold indexes to skip
    allow_incomplete_fold=True,      # Whether to allow last fold with fewer observations than steps
    return_all_indexes=False,        # Whether to return all indexes or only start/end of each fold
    verbose=True                     # Whether to print information about generated folds
)

# View the folds
folds = cv.split(X=data, as_pandas=True)
print(folds)
```

**Key behaviors:**
- If `fold_stride == steps`: test sets are back-to-back without overlap
- If `fold_stride < steps`: test sets overlap (multiple forecasts for same observations)
- If `fold_stride > steps`: gaps between consecutive test sets

### OneStepAheadFold

Class for one-step-ahead forecasting validation. Faster than `TimeSeriesFold` as it doesn't require recursive predictions.

```python
from skforecast.model_selection import OneStepAheadFold

cv = OneStepAheadFold(
    initial_train_size=100,          # (required) Number of observations for initial training. Can be int, str (date), or pd.Timestamp
    window_size=None,                # Observations needed for autoregressive predictors (set automatically by forecaster)
    differentiation=None,            # Differencing order to extend last_window (set automatically by forecaster)
    return_all_indexes=False,        # Whether to return all indexes or only start/end of each fold
    verbose=True                     # Whether to print information about generated folds
)

# View the fold
fold = cv.split(X=data, as_pandas=True)
print(fold)
```

**When to use:**
- `TimeSeriesFold`: When you need to evaluate multi-step forecasting performance (realistic backtesting)
- `OneStepAheadFold`: When you need fast hyperparameter tuning (less accurate but much faster)

## Hyperparameter Tuning

Three search strategies are available: `grid_search_forecaster`, `random_search_forecaster`, and `bayesian_search_forecaster` (Optuna-based). All accept `TimeSeriesFold` or `OneStepAheadFold`. Multi-series variants: `grid_search_forecaster_multiseries`, `random_search_forecaster_multiseries`, `bayesian_search_forecaster_multiseries`.

```python
from skforecast.model_selection import bayesian_search_forecaster, TimeSeriesFold

cv = TimeSeriesFold(steps=12, initial_train_size=len(data) - 100, refit=False)

# Bayesian Search — lags can be included in search_space
def search_space(trial):
    return {
        'lags': trial.suggest_categorical('lags', [3, 5, [1, 2, 3, 20]]),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }

results, best_trial = bayesian_search_forecaster(
    forecaster=forecaster,
    y=data['target'],
    cv=cv,
    search_space=search_space,
    metric='mean_absolute_error',
    n_trials=50,
    random_state=123,
    return_best=True,
    n_jobs='auto',
    show_progress=True
)
```

## Statistical Models (ARIMA, ETS, ARAR)

Statistical models are wrapped by `ForecasterStats` for a sklearn-compatible interface. Available models: `Arima`, `Sarimax`, `Ets`, `Arar`.

```python
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima, Ets, Sarimax, Arar

# ARIMA model (order=(p,d,q), seasonal_order=(P,D,Q), m=seasonal_period)
forecaster = ForecasterStats(estimator=Arima(order=(1, 1, 1), seasonal_order=(1, 1, 1), m=12))
forecaster.fit(y=data['target'])
predictions = forecaster.predict(steps=10)

# Auto ARIMA (automatic order selection) - set order=None
forecaster = ForecasterStats(estimator=Arima(order=None, seasonal=True, m=12))

# ETS model (model string: 1st=Error, 2nd=Trend, 3rd=Seasonal; A=Add, M=Mult, N=None, Z=Auto)
forecaster = ForecasterStats(estimator=Ets(m=12, model='AAA'))
```

## Feature Selection

Use sklearn selectors (RFECV, SelectFromModel, etc.) to identify relevant lags, window features, and exogenous variables. Multi-series variant: `select_features_multiseries`.

```python
from sklearn.feature_selection import RFECV
from skforecast.feature_selection import select_features

selected_lags, selected_window_features, selected_exog = select_features(
    forecaster=forecaster,
    selector=RFECV(estimator=RandomForestRegressor(), step=1, cv=3),
    y=y_train,
    exog=exog_train,
    select_only=None,              # 'autoreg', 'exog', or None (all features)
    force_inclusion=None,          # Features to always include (list or regex str)
    subsample=0.5,
    random_state=123,
    verbose=True
)
forecaster.set_lags(selected_lags)
```

## Drift Detection

Two drift detection tools for monitoring data distribution changes during deployment.

```python
from skforecast.drift_detection import RangeDriftDetector, PopulationDriftDetector

# RangeDriftDetector — lightweight, checks out-of-range values vs training ranges
detector = RangeDriftDetector()
detector.fit(series=y_train, exog=exog_train)
flag_out_of_range, out_of_range_series, out_of_range_exog = detector.predict(
    last_window=new_data, exog=new_exog
)

# PopulationDriftDetector — statistical tests (KS, Chi-Square, Jensen-Shannon)
detector = PopulationDriftDetector(chunk_size=100, threshold=3, threshold_method='std')
detector.fit(X=reference_data)
results, summary = detector.predict(X=new_data)
```

## Key Classes and Imports

**Note:** In versions prior to 0.14.0, `ForecasterRecursive` was named `ForecasterAutoreg` and `ForecasterRecursiveMultiSeries` was named `ForecasterAutoregMultiSeries`. Always use the current names shown below.

```python
# Forecasters
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.recursive import ForecasterRecursiveClassifier
from skforecast.recursive import ForecasterStats
from skforecast.recursive import ForecasterEquivalentDate
from skforecast.direct import ForecasterDirect
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning import create_and_compile_model

# Model Selection
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import backtesting_stats
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import grid_search_forecaster_multiseries
from skforecast.model_selection import random_search_forecaster
from skforecast.model_selection import random_search_forecaster_multiseries
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.model_selection import grid_search_stats
from skforecast.model_selection import random_search_stats
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import OneStepAheadFold

# Preprocessing
from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.preprocessing import DateTimeFeatureTransformer
from skforecast.preprocessing import create_datetime_features
from skforecast.preprocessing import QuantileBinner
from skforecast.preprocessing import ConformalIntervalCalibrator
# Data reshaping utilities
from skforecast.preprocessing import reshape_series_wide_to_long
from skforecast.preprocessing import reshape_series_long_to_dict
from skforecast.preprocessing import reshape_exog_long_to_dict
from skforecast.preprocessing import reshape_series_exog_dict_to_long

# Feature Selection
from skforecast.feature_selection import select_features
from skforecast.feature_selection import select_features_multiseries

# Datasets
from skforecast.datasets import fetch_dataset
from skforecast.datasets import load_demo_dataset
from skforecast.datasets import show_datasets_info

# Metrics
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from skforecast.metrics import symmetric_mean_absolute_percentage_error
from skforecast.metrics import add_y_train_argument
from skforecast.metrics import crps_from_predictions
from skforecast.metrics import crps_from_quantiles
from skforecast.metrics import calculate_coverage
from skforecast.metrics import create_mean_pinball_loss

# Statistical models (used with ForecasterStats)
from skforecast.stats import Arima, Ets, Sarimax, Arar

# Drift Detection
from skforecast.drift_detection import RangeDriftDetector
from skforecast.drift_detection import PopulationDriftDetector

# Plotting
from skforecast.plot import plot_residuals
from skforecast.plot import plot_multivariate_time_series_corr
from skforecast.plot import plot_prediction_distribution
from skforecast.plot import plot_prediction_intervals
from skforecast.plot import calculate_lag_autocorrelation
from skforecast.plot import backtesting_gif_creator
from skforecast.plot import set_dark_theme

# Exceptions and Warnings
from skforecast.exceptions import set_warnings_style
from skforecast.exceptions import warn_skforecast_categories
```

## Available Datasets

30+ built-in datasets covering multiple frequencies (15min, 30min, hourly, daily, monthly, quarterly). Most commonly used:

```python
from skforecast.datasets import fetch_dataset, show_datasets_info

data = fetch_dataset(name='h2o')             # Monthly, single series
data = fetch_dataset(name='items_sales')     # Daily, 3 series (multi-series)
data = fetch_dataset(name='bike_sharing')    # Hourly, with exogenous variables
data = fetch_dataset(name='store_sales')     # Daily, 50 products × 10 stores
data = fetch_dataset(name='h2o_exog')        # Monthly, with exogenous variables

# See all available datasets
show_datasets_info()
```

## Tips for Best Results

1. **Always set frequency**: Use `data.asfreq('h')` or similar — skforecast requires a DatetimeIndex with frequency
2. **Handle missing values**: Forecasters don't accept NaN by default
3. **Scale data**: Use `transformer_y` for better model performance
4. **Use backtesting**: Always validate with realistic train/test splits
5. **Consider differentiation**: For non-stationary series, use `differentiation` parameter
6. **Start simple**: Begin with ForecasterRecursive before trying complex models

## Documentation

- Quick Start: https://skforecast.org/latest/quick-start/quick-start-skforecast.html
- User Guides: https://skforecast.org/latest/user_guides/table-of-contents.html
- API Reference: https://skforecast.org/latest/api/forecasterrecursive.html
- Examples: https://skforecast.org/latest/examples/examples_english.html
- Release Notes: https://skforecast.org/latest/releases/releases.html

## Citation

```
Amat Rodrigo, J., & Escobar Ortiz, J. (2026). skforecast (Version 0.21.0) [Computer software]. https://doi.org/10.5281/zenodo.8382787
```
