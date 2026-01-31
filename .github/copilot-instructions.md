# Skforecast - GitHub Copilot Instructions

This file provides context for GitHub Copilot to generate accurate and up-to-date code for the **skforecast** library.

## About Skforecast

Skforecast is a Python library (v0.20.0) for time series forecasting using machine learning and statistical models. It works with any estimator compatible with the scikit-learn API.

**Repository**: https://github.com/skforecast/skforecast  
**Documentation**: https://skforecast.org  
**Python versions**: 3.10, 3.11, 3.12, 3.13, 3.14

## Project Structure

```
skforecast/
├── base/                    # ForecasterBase - parent class for all forecasters
├── recursive/               # ForecasterRecursive, ForecasterRecursiveMultiSeries, ForecasterRecursiveClassifier, ForecasterStats
├── direct/                  # ForecasterDirect, ForecasterDirectMultiVariate
├── stats/                   # Statistical model classes: Arima, Ets, Sarimax, Arar
├── preprocessing/           # TimeSeriesDifferentiator, RollingFeatures, DateTimeFeatureTransformer
├── model_selection/         # Backtesting, grid/random/bayesian search, TimeSeriesFold
├── feature_selection/       # Feature importance and selection tools
├── metrics/                 # Custom metrics for time series
├── utils/                   # Utility functions (check_*, initialize_*, transform_*)
├── exceptions/              # Custom warnings and exceptions
├── datasets/                # Sample datasets
├── plot/                    # Plotting utilities
├── deep_learning/           # ForecasterRnn (RNN/LSTM)
├── drift_detection/         # Concept drift detection
└── experimental/            # Experimental features
```

## Detailed Project Architecture

```
skforecast/
├── base/                    # Abstract base class for all forecasters
│   ├── _forecaster_base.py  # ForecasterBase with common interface:
│   │                        #   - fit(), predict(), predict_interval()
│   │                        #   - set_params(), get_feature_importances()
│   │                        #   - Common attributes: is_fitted, last_window_, etc.
│   └── __init__.py
│
├── recursive/               # Recursive (autoregressive) forecasters
│   ├── _forecaster_recursive.py             # ForecasterRecursive
│   │                                        #   Single series, uses predictions as inputs
│   ├── _forecaster_recursive_multiseries.py # ForecasterRecursiveMultiSeries
│   │                                        #   Global model for multiple series
│   ├── _forecaster_recursive_classifier.py  # ForecasterRecursiveClassifier
│   │                                        #   Classification-based approach
│   ├── _forecaster_stats.py                 # ForecasterStats
│   │                                        #   Wrapper for statistical models (ARIMA, ETS)
│   ├── _forecaster_equivalent_date.py       # ForecasterEquivalentDate
│   │                                        #   Baseline using equivalent past dates
│   ├── tests/                               # Unit tests for recursive forecasters
│   └── __init__.py
│
├── direct/                  # Direct multi-step forecasters (one model per horizon step)
│   ├── _forecaster_direct.py           # ForecasterDirect
│   │                                   #   Single series, trains n_steps models
│   ├── _forecaster_direct_multivariate.py  # ForecasterDirectMultiVariate
│   │                                       #   Multiple series as input features
│   ├── tests/
│   └── __init__.py
│
├── deep_learning/           # Neural network forecasters
│   ├── _forecaster_rnn.py   # ForecasterRnn - RNN/LSTM/GRU forecaster
│   ├── utils.py             # create_and_compile_model() helper function
│   ├── tests/
│   └── __init__.py
│
├── stats/                   # Statistical models (sklearn-compatible wrappers)
│   ├── _arima.py            # Arima class (ARIMA + Auto-ARIMA when order=None)
│   ├── _sarimax.py          # Sarimax class (SARIMAX with exogenous)
│   ├── _ets.py              # Ets class (Exponential Smoothing)
│   ├── _arar.py             # Arar class (Autoregressive AR)
│   ├── _utils.py            # Shared utilities for stats models
│   ├── arima/               # ARIMA internals and auto-selection
│   │   └── _auto_arima.py   # Auto-ARIMA implementation
│   ├── seasonal/            # Seasonal analysis utilities
│   ├── transformations/     # Box-Cox and power transformations
│   ├── tests/
│   └── __init__.py
│
├── preprocessing/           # Data transformation and feature engineering
│   ├── preprocessing.py     # All preprocessing classes:
│   │   # - TimeSeriesDifferentiator: Apply/reverse differencing
│   │   # - RollingFeatures: Rolling window statistics as features
│   │   # - RollingFeaturesClassification: Rolling features for classifiers
│   │   # - DateTimeFeatureTransformer: Extract datetime features
│   │   # - QuantileBinner: Bin residuals for conformal prediction
│   │   # - ConformalIntervalCalibrator: Calibrate prediction intervals
│   │   # - reshape_series_wide_to_long(): Convert wide to long format
│   │   # - reshape_series_long_to_dict(): Convert long to dict format
│   │   # - reshape_exog_long_to_dict(): Convert exog long to dict
│   │   # - reshape_series_exog_dict_to_long(): Convert dict to long
│   ├── tests/
│   └── __init__.py
│
├── model_selection/         # Model evaluation and hyperparameter optimization
│   ├── _split.py            # Cross-validation splitters:
│   │                        #   - TimeSeriesFold: Multi-step ahead CV
│   │                        #   - OneStepAheadFold: Fast one-step CV
│   ├── _validation.py       # Backtesting functions:
│   │                        #   - backtesting_forecaster
│   │                        #   - backtesting_forecaster_multiseries
│   │                        #   - backtesting_stats
│   ├── _search.py           # Hyperparameter search:
│   │                        #   - grid_search_forecaster[_multiseries]
│   │                        #   - random_search_forecaster[_multiseries]
│   │                        #   - bayesian_search_forecaster[_multiseries]
│   │                        #   - grid/random_search_stats
│   ├── _utils.py            # Helper functions for model selection
│   ├── tests/
│   └── __init__.py
│
├── feature_selection/       # Feature importance and selection tools
│   ├── feature_selection.py # Functions:
│   │                        #   - select_features(): For single series
│   │                        #   - select_features_multiseries(): For multi-series
│   │                        #   Works with sklearn selectors (RFECV, etc.)
│   ├── tests/
│   └── __init__.py
│
├── metrics/                 # Time series specific evaluation metrics
│   ├── metrics.py           # Metrics:
│   │                        #   - mean_absolute_scaled_error (MASE)
│   │                        #   - root_mean_squared_scaled_error (RMSSE)
│   │                        #   - symmetric_mean_absolute_percentage_error (sMAPE)
│   │                        #   - crps_from_predictions, crps_from_quantiles
│   │                        #   - calculate_coverage
│   ├── tests/
│   └── __init__.py
│
├── drift_detection/         # Data drift monitoring for production
│   ├── _range_drift.py      # RangeDriftDetector:
│   │                        #   Lightweight, checks if values are in training range
│   ├── _population_drift.py # PopulationDriftDetector:
│   │                        #   Statistical tests (KS, Chi-Square, JS divergence)
│   ├── tests/
│   └── __init__.py
│
├── utils/                   # Shared utility functions used across all modules
│   ├── utils.py             # Key functions:
│   │   # Input validation: check_y, check_exog, check_interval, check_predict_input
│   │   # Initialization: initialize_lags, initialize_window_features, initialize_weights
│   │   # Transformation: transform_numpy, transform_dataframe, input_to_frame
│   │   # Index manipulation: expand_index, date_to_index_position
│   ├── tests/
│   └── __init__.py
│
├── exceptions/              # Custom warnings and exceptions
│   ├── exceptions.py        # Custom exceptions:
│   │                        #   - DataTransformationWarning
│   │                        #   - IgnoredArgumentWarning
│   │                        #   - MissingValuesWarning
│   │                        #   - ResidualsUsageWarning
│   │                        #   - warn_skforecast_categories()
│   ├── tests/
│   └── __init__.py
│
├── datasets/                # Sample datasets for examples and testing
│   ├── datasets.py          # fetch_dataset() function
│   │                        # 30+ datasets: h2o, bike_sharing, store_sales, etc.
│   ├── tests/
│   └── __init__.py
│
├── plot/                    # Visualization utilities
│   ├── plot.py              # Plotting functions for forecasts and diagnostics
│   ├── tests/
│   └── __init__.py
│
├── experimental/            # Experimental features (API may change without notice)
│   ├── _experimental.py
│   └── __init__.py
│
└── __init__.py              # Package version and top-level imports
```

### Module Relationships and Dependencies

```
                              ┌─────────────────┐
                              │  ForecasterBase │
                              │     (base/)     │
                              └────────┬────────┘
                                       │ inherits
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐            ┌─────────────────┐            ┌─────────────────┐
│   recursive/  │            │     direct/     │            │ deep_learning/  │
│ - Recursive   │            │ - Direct        │            │ - ForecasterRnn │
│ - MultiSeries │            │ - MultiVariate  │            └─────────────────┘
│ - Classifier  │            └─────────────────┘
└───────────────┘

┌─────────────────────────────────────────────────────────────┐
│           Standalone Forecasters (no inheritance)           │
├─────────────────────────────────────────────────────────────┤
│  ForecasterStats ─────► wraps stats/ models (Arima, ETS,   │
│                         Sarimax, Arar)                      │
│  ForecasterEquivalentDate ─► Baseline forecaster            │
└─────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────┐
        │                    Supporting Modules                        │
        ├─────────────────────────────────────────────────────────────┤
        │  preprocessing/  ──►  transformer_y, transformer_exog,      │
        │                       window_features parameters             │
        │  model_selection/ ──► backtesting, grid/bayesian search     │
        │  feature_selection/ ► select_features for lag/exog selection│
        │  metrics/         ──► Custom evaluation metrics              │
        │  drift_detection/ ──► Production monitoring                  │
        │  utils/           ──► Shared validation & transformation     │
        │  exceptions/      ──► Custom warnings across all modules     │
        └─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Inheritance**: Most forecasters inherit from `ForecasterBase` (exceptions: `ForecasterStats`, `ForecasterEquivalentDate`)
2. **Composition**: Statistical models (`stats/`) are wrapped by `ForecasterStats` for sklearn compatibility
3. **Dependency Injection**: Transformers and estimators are passed to forecasters at initialization
4. **Strategy Pattern**: Different forecasting strategies (recursive, direct) share common interface
5. **Factory Pattern**: `create_and_compile_model()` for building RNN architectures

## Correct Import Patterns

Always use these import patterns - they reflect the current API:

```python
# ✅ CORRECT Forecaster imports (v0.20.0+)
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.recursive import ForecasterRecursiveClassifier
from skforecast.recursive import ForecasterStats
from skforecast.recursive import ForecasterEquivalentDate
from skforecast.direct import ForecasterDirect
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning import create_and_compile_model

# Statistical models (used with ForecasterStats)
from skforecast.stats import Arima, Ets, Sarimax, Arar

# ❌ DEPRECATED imports - DO NOT USE
# from skforecast.ForecasterAutoreg import ForecasterAutoreg  # OLD
# from skforecast.ForecasterAutoregMultiSeries import ...     # OLD
```

```python
# ✅ CORRECT Model Selection imports
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import backtesting_forecaster_multiseries
from skforecast.model_selection import backtesting_stats
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import grid_search_forecaster_multiseries
from skforecast.model_selection import grid_search_stats
from skforecast.model_selection import random_search_forecaster
from skforecast.model_selection import random_search_forecaster_multiseries
from skforecast.model_selection import random_search_stats
from skforecast.model_selection import bayesian_search_forecaster
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import OneStepAheadFold
```

```python
# ✅ CORRECT Preprocessing imports
from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.preprocessing import DateTimeFeatureTransformer
from skforecast.preprocessing import QuantileBinner
from skforecast.preprocessing import ConformalIntervalCalibrator

# Data reshaping utilities
from skforecast.preprocessing import reshape_series_wide_to_long
from skforecast.preprocessing import reshape_series_long_to_dict
from skforecast.preprocessing import reshape_exog_long_to_dict
from skforecast.preprocessing import reshape_series_exog_dict_to_long
```

```python
# ✅ CORRECT Other imports
from skforecast.feature_selection import select_features
from skforecast.feature_selection import select_features_multiseries
from skforecast.datasets import fetch_dataset
from skforecast.metrics import mean_absolute_scaled_error
from skforecast.metrics import root_mean_squared_scaled_error
from skforecast.metrics import symmetric_mean_absolute_percentage_error
from skforecast.metrics import crps_from_predictions
from skforecast.metrics import crps_from_quantiles
from skforecast.metrics import calculate_coverage
from skforecast.drift_detection import RangeDriftDetector
from skforecast.drift_detection import PopulationDriftDetector
```

## Core Forecasters API

### ForecasterRecursive (Single Series)

```python
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures

forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=24,                                    # int, list, or array
    window_features=RollingFeatures(            # Optional rolling statistics
        stats=['mean', 'std'],
        window_sizes=7                          # int applies to all stats, or list with same length as stats
    ),
    transformer_y=None,                         # Optional transformer for y
    transformer_exog=None,                      # Optional transformer for exog
    differentiation=None,                       # Optional: int for differencing order
    fit_kwargs=None                             # Optional: kwargs for estimator.fit()
)

# Training
forecaster.fit(y=y_train, exog=exog_train)

# Prediction
predictions = forecaster.predict(steps=10, exog=exog_test)

# Prediction intervals
predictions_interval = forecaster.predict_interval(
    steps=10,
    exog=exog_test,
    interval=[10, 90],                          # 80% interval
    method='bootstrapping',                     # or 'quantile_residuals', 'conformal'
    n_boot=500
)
```

### ForecasterRecursiveMultiSeries (Multiple Series - Global Model)

```python
from skforecast.recursive import ForecasterRecursiveMultiSeries
from lightgbm import LGBMRegressor

forecaster = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(n_estimators=100, random_state=123),
    lags=24,
    encoding='ordinal',                         # 'ordinal', 'onehot', 'onehot_drop_first'
    transformer_series=None,                    # Transformer or dict of transformers
    transformer_exog=None,
    differentiation=None,
    window_features=None
)

# Training - series is a DataFrame with each column being a time series
forecaster.fit(series=series_df, exog=exog_df)

# Predict all series
predictions = forecaster.predict(steps=10, exog=exog_test)

# Predict specific series
predictions = forecaster.predict(steps=10, levels=['series_1', 'series_2'], exog=exog_test)
```

### ForecasterDirect (Direct Multi-Step)

```python
from skforecast.direct import ForecasterDirect

forecaster = ForecasterDirect(
    estimator=LGBMRegressor(),
    lags=24,
    steps=10,                                   # Must specify forecast horizon
    transformer_y=None,
    transformer_exog=None
)

forecaster.fit(y=y_train, exog=exog_train)
predictions = forecaster.predict(exog=exog_test)  # steps already defined
```

### ForecasterStats (Statistical Models)

```python
from skforecast.recursive import ForecasterStats
from skforecast.stats import Arima, Ets, Sarimax, Arar

# ARIMA model (order=(p,d,q), seasonal_order=(P,D,Q), m=seasonal_period)
arima_model = Arima(order=(1, 1, 1), seasonal_order=(1, 1, 1), m=12)
forecaster = ForecasterStats(estimator=arima_model)

# Auto ARIMA (automatic order selection) - set order=None
auto_arima = Arima(order=None, seasonal=True, m=12)
forecaster = ForecasterStats(estimator=auto_arima)

# ETS model (Error-Trend-Seasonal)
ets_model = Ets(error='add', trend='add', seasonal='add', seasonal_periods=12)
forecaster = ForecasterStats(estimator=ets_model)

# ARAR model (Autoregressive AR with memory shortening)
arar_model = Arar()
forecaster = ForecasterStats(estimator=arar_model)

forecaster.fit(y=y_train)
predictions = forecaster.predict(steps=10)
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

## Model Selection

### Backtesting

```python
from skforecast.model_selection import backtesting_forecaster, TimeSeriesFold

# Define cross-validation strategy
cv = TimeSeriesFold(
    steps=10,
    initial_train_size=len(data) - 100,
    refit=False,
    fixed_train_size=False                      # Expanding window if False
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

**Parameters explanation:**

| Parameter | Description |
|-----------|-------------|
| `forecaster` | Forecaster object (ForecasterRecursive, ForecasterDirect, ForecasterEquivalentDate, ForecasterRecursiveClassifier) |
| `y` | Training time series as pandas Series with DatetimeIndex |
| `cv` | TimeSeriesFold object defining the cross-validation strategy |
| `metric` | Metric(s) to evaluate: 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_log_error', 'mean_absolute_scaled_error', 'root_mean_squared_scaled_error', or custom callable |
| `exog` | Exogenous variables (pandas Series/DataFrame). Must cover the entire time period |
| `interval` | Prediction intervals: float (0-1 for coverage), list/tuple of percentiles [10, 90], 'bootstrapping', or scipy.stats distribution |
| `interval_method` | Method for intervals: 'bootstrapping' (resampling residuals) or 'conformal' (conformal prediction) |
| `n_boot` | Number of bootstrap samples for interval estimation (default 250) |
| `use_in_sample_residuals` | If True, use training residuals; if False, use out-of-sample residuals (must be set with `set_out_sample_residuals()`) |
| `use_binned_residuals` | If True, select residuals based on predicted value bins for better interval calibration |
| `random_state` | Seed for reproducible results |
| `return_predictors` | If True, return DataFrame with predictor values used in each fold |
| `n_jobs` | Parallel jobs: -1 (all cores), 'auto' (automatic selection), or specific int |
| `verbose` | Print information about folds and training/validation sets |
| `show_progress` | Display progress bar during backtesting |
| `suppress_warnings` | Suppress skforecast-specific warnings during execution |

### Grid Search

```python
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import grid_search_forecaster

# Define cross-validation strategy
cv = TimeSeriesFold(
    steps=12,
    initial_train_size=len(data) - 100,
    refit=False
)

# Lags grid (different lag configurations to try)
lags_grid = [3, 10, [1, 2, 3, 20]]

# Estimator hyperparameters grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, 15]
}

results = grid_search_forecaster(
    forecaster=forecaster,
    y=data['target'],
    exog=exog,
    cv=cv,
    lags_grid=lags_grid,
    param_grid=param_grid,
    metric='mean_squared_error',
    return_best=True,
    n_jobs='auto',
    show_progress=True
)
```

### Bayesian Search (Optuna-based)

```python
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import bayesian_search_forecaster

# Define cross-validation strategy
cv = TimeSeriesFold(
    steps=12,
    initial_train_size=len(data) - 100,
    refit=False
)

# NOTE: Search space function (lags can be included here for bayesian search)
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
    exog=exog,
    cv=cv,
    search_space=search_space,
    metric='mean_absolute_error',
    n_trials=50,
    random_state=123,
    return_best=True,
    n_jobs='auto',
    show_progress=True,
    suppress_warnings=False,
    output_file=None,
    kwargs_create_study={},
    kwargs_study_optimize={}
)
```

### Bayesian Search Multi-Series

```python
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import bayesian_search_forecaster_multiseries

cv = TimeSeriesFold(
    steps=12,
    initial_train_size=len(series_df) - 100,
    refit=False
)

results, best_trial = bayesian_search_forecaster_multiseries(
    forecaster=forecaster_multiseries,
    series=series,  # Wide DataFrame, Long DataFrame, dict {series_id: pd.Series}
    exog=exog,  # Wide DataFrame, Long DataFrame, dict {series_id: pd.Series or pd.DataFrame}
    cv=cv,
    search_space=search_space,
    metric='mean_absolute_error',
    n_trials=50,
    return_best=True,
    n_jobs='auto',
    show_progress=True,
    suppress_warnings=False,
    output_file=None,
    kwargs_create_study={},
    kwargs_study_optimize={}
)
```

## Preprocessing

### RollingFeatures

```python
from skforecast.preprocessing import RollingFeatures

rolling = RollingFeatures(
    stats=['mean', 'std', 'min', 'max', 'sum'],
    window_sizes=[7, 14, 30]
)

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    window_features=rolling
)
```

### TimeSeriesDifferentiator

```python
from skforecast.preprocessing import TimeSeriesDifferentiator

# Automatic differencing in forecaster
forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    differentiation=1                           # First-order differencing
)

# Manual differencing
differentiator = TimeSeriesDifferentiator(order=1)
y_diff = differentiator.fit_transform(y)
y_original = differentiator.inverse_transform(y_diff)
```

### DateTimeFeatureTransformer

```python
from skforecast.preprocessing import DateTimeFeatureTransformer

transformer = DateTimeFeatureTransformer(
    features=['year', 'month', 'week', 'day_of_week', 'hour']
)

# Use with pandas DatetimeIndex
exog = transformer.fit_transform(data.index)
```

## Feature Selection

Feature selection using sklearn selectors (RFECV, SelectFromModel, etc.) to identify the most relevant lags, window features, and exogenous variables.

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.feature_selection import select_features

# Create forecaster with many features
forecaster = ForecasterRecursive(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=24
)

# Feature selection for single series
selected_lags, selected_window_features, selected_exog = select_features(
    forecaster=forecaster,
    selector=RFECV(estimator=RandomForestRegressor(), step=1, cv=3),
    y=y_train,
    exog=exog_train,
    select_only=None,              # 'autoreg', 'exog', or None (all features)
    force_inclusion=None,          # Features to always include (list or regex str)
    subsample=0.5,                 # Proportion of data to use
    random_state=123,
    verbose=True
)

# Update forecaster with selected features
forecaster.set_lags(selected_lags)
```

### Feature Selection for Multi-Series

```python
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.feature_selection import select_features_multiseries

forecaster = ForecasterRecursiveMultiSeries(
    estimator=RandomForestRegressor(n_estimators=100, random_state=123),
    lags=24
)

selected_lags, selected_window_features, selected_exog = select_features_multiseries(
    forecaster=forecaster,
    selector=RFECV(estimator=RandomForestRegressor(), step=1, cv=3),
    series=series_df,
    exog=exog_df,
    select_only=None,              # 'autoreg', 'exog', or None (all features)
    force_inclusion=None,          # Features to always include (list or regex str)
    subsample=0.5,                 # Proportion of data to use
    random_state=123,
    verbose=True
)
```

## Drift Detection

Skforecast provides two drift detection tools for monitoring changes in data distribution during model deployment.

### RangeDriftDetector

Lightweight detector for out-of-range values based on training feature ranges. Suitable for real-time inference applications.

```python
from skforecast.drift_detection import RangeDriftDetector

# Initialize and fit the detector
detector = RangeDriftDetector()
detector.fit(series=y_train, exog=exog_train)

# Check new data for out-of-range values
flag_out_of_range, out_of_range_series, out_of_range_exog = detector.predict(
    last_window=new_data,
    exog=new_exog,
    verbose=True,
    suppress_warnings=False
)
```

### PopulationDriftDetector

Advanced detector using statistical tests (Kolmogorov-Smirnov, Chi-Square, Jensen-Shannon) to detect population drift between reference and new datasets.

```python
from skforecast.drift_detection import PopulationDriftDetector

# Initialize the detector
detector = PopulationDriftDetector(
    chunk_size=100,                     # int, str (e.g., 'D', 'W'), or None
    threshold=3,                        # Multiplier for std or quantile level
    threshold_method='std',             # 'std' or 'quantile'
    max_out_of_range_proportion=0.1     # Max allowed proportion out of range
)

# Fit with reference data
detector.fit(X=reference_data)

# Detect drift in new data
results, summary = detector.predict(X=new_data)
```

**Key Parameters:**

| Parameter | Description |
|-----------|-------------|
| `chunk_size` | Size of chunks for sequential analysis: int (observations), str (time-based like 'D', 'W'), or None (entire dataset) |
| `threshold` | Threshold for statistics. If `threshold_method='std'`: multiplier of std. If `threshold_method='quantile'`: quantile level (0-1) |
| `threshold_method` | `'std'`: mean + threshold * std. `'quantile'`: empirical quantile with leave-one-chunk-out CV |
| `max_out_of_range_proportion` | Maximum allowed proportion of out-of-range observations (0-1) |

## Datasets

```python
from skforecast.datasets import fetch_dataset
data = fetch_dataset(name='h2o')
```

**Available datasets:**

| Dataset | Frequency | Description |
|---------|-----------|-------------|
| `h2o` | Monthly | Australian health system expenditure on corticosteroid drugs (1991-2008) |
| `h2o_exog` | Monthly | Same as h2o with simulated exogenous variables |
| `fuel_consumption` | Monthly | Monthly fuel consumption in Spain (1969-2022) |
| `items_sales` | Daily | Simulated sales for 3 different items |
| `air_quality_valencia` | Hourly | Air pollutant measures at Valencia city (2019-2023) |
| `air_quality_valencia_no_missing` | Hourly | Same as above with imputed missing values |
| `website_visits` | Daily | Daily visits to cienciadedatos.net website |
| `bike_sharing` | Hourly | Bike share system usage in Washington D.C. (2011-2012) |
| `bike_sharing_extended_features` | Hourly | Same as above with engineered features |
| `australia_tourism` | Quarterly | Quarterly overnight trips in Australia (1998-2016) |
| `uk_daily_flights` | Daily | Daily number of flights in UK (2019-2022) |
| `wikipedia_visits` | Daily | Log daily page views for Peyton Manning Wikipedia page |
| `vic_electricity` | 30min | Half-hourly electricity demand for Victoria, Australia |
| `vic_electricity_classification` | Hourly | Electricity demand classified into low/medium/high |
| `store_sales` | Daily | 913,000 sales transactions for 50 products in 10 stores (2013-2017) |
| `bicimad` | Daily | Daily users of BiciMad bicycle rental in Madrid (2014-2022) |
| `m4_daily` | Daily | Time series with daily frequency from M4 competition |
| `m4_hourly` | Hourly | Time series with hourly frequency from M4 competition |
| `ashrae_daily` | Daily | Daily energy consumption from ASHRAE competition |
| `bdg2_daily` | Daily | Daily energy consumption from Building Data Genome Project 2 |
| `bdg2_daily_sample` | Daily | Sample of two buildings from bdg2_daily |
| `bdg2_hourly` | Hourly | Hourly energy consumption from Building Data Genome Project 2 |
| `bdg2_hourly_sample` | Hourly | Sample of two buildings from bdg2_hourly |
| `m5` | Daily | Daily sales data from M5 competition |
| `ett_m1` | 15min | Electricity transformer data (2016-2018) |
| `ett_m2` | 15min | Electricity transformer data from different station |
| `ett_m2_extended` | 15min | Same as ett_m2 with calendar features |
| `expenditures_australia` | Monthly | Monthly expenditure on cafes/restaurants in Victoria (1982-2024) |
| `public_transport_madrid` | Daily | Daily public transport users in Madrid (2023-2024) |
| `turbine_emission` | Hourly | Gas turbine sensor measures for emissions study (2011-2015) |

## Important Notes

1. **Data must have DatetimeIndex with frequency**: Always use `data.asfreq('h')` or similar
2. **No NaN values**: Forecasters don't accept missing values by default
3. **Exog for prediction**: Must cover the entire forecast horizon
4. **Transformers**: Applied automatically during fit/predict
5. **Differentiation**: Predictions are automatically inverse-transformed

## Code Style

- Use NumPy-style docstrings
- Type hints for function signatures
- PEP 8 compliant (max line length 88)
- Relative imports within package

## Testing

```bash
pytest skforecast/recursive/tests/ -vv
pytest --cov=skforecast --cov-report=html
```

## Dependencies

Core: numpy>=1.24, pandas>=1.5, scikit-learn>=1.2, scipy>=1.3.2, optuna>=2.10, joblib>=1.1, numba>=0.59, tqdm>=4.57, rich>=13.9

Optional:
- stats: statsmodels>=0.12, <0.15
- plotting: matplotlib>=3.3, seaborn>=0.11
- deeplearning: keras>=3.0
