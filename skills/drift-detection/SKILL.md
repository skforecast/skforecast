---
name: drift-detection
description: >
  Detects data drift in time series forecasting pipelines using
  RangeDriftDetector and PopulationDriftDetector. Covers range-based
  out-of-range detection and statistical distribution tests.
  Use when the user wants to monitor model reliability in production.
---

# Drift Detection

## When to Use

Use drift detection to monitor whether new data falls outside the patterns seen during training. This helps decide when to retrain a model.

| Detector | Speed | Use Case |
|----------|-------|----------|
| `RangeDriftDetector` | Very fast | Real-time inference — checks if values are in training range |
| `PopulationDriftDetector` | Moderate | Batch monitoring — statistical tests for distribution shifts |

## RangeDriftDetector

Checks whether new observations fall within the ranges seen during training. Lightweight and suitable for real-time scoring.

```python
from skforecast.drift_detection import RangeDriftDetector
from skforecast.recursive import ForecasterRecursive

# 1. Train the forecaster
forecaster = ForecasterRecursive(estimator=regressor, lags=24)
forecaster.fit(y=y_train, exog=exog_train)

# 2. Fit the drift detector on training data
detector = RangeDriftDetector()
detector.fit(series=y_train, exog=exog_train)

# 3. Check new data before making predictions
flag_drift, out_of_range_series, out_of_range_exog = detector.predict(
    last_window=new_data,
    exog=new_exog,
    verbose=True,           # Print drift details
    suppress_warnings=False,
)

if flag_drift:
    print("WARNING: New data contains values outside training range!")
    print(f"Out-of-range series features: {out_of_range_series}")
    print(f"Out-of-range exog features: {out_of_range_exog}")
```

## PopulationDriftDetector

Uses statistical tests to detect distribution shifts between reference (training) and new data.

```python
from skforecast.drift_detection import PopulationDriftDetector

# 1. Create detector
detector = PopulationDriftDetector(
    chunk_size=100,                     # Split data into chunks of 100 obs
    threshold=3,                        # Multiplier for std deviation
    threshold_method='std',             # 'std' or 'quantile'
    max_out_of_range_proportion=0.1,    # Max 10% out-of-range allowed
)

# 2. Fit on reference (training) data
detector.fit(X=X_train)

# 3. Detect drift in new data
results, summary = detector.predict(X=X_new)

# results: DataFrame with per-chunk drift statistics
# summary: DataFrame with per-feature drift summary
print(summary)
```

### Chunk Size Options

```python
# Fixed number of observations per chunk
detector = PopulationDriftDetector(chunk_size=100)

# Time-based chunks
detector = PopulationDriftDetector(chunk_size='W')   # Weekly
detector = PopulationDriftDetector(chunk_size='M')   # Monthly
detector = PopulationDriftDetector(chunk_size='D')   # Daily

# No chunking — compare entire datasets
detector = PopulationDriftDetector(chunk_size=None)
```

### Threshold Methods

```python
# Standard deviation method: flag if statistic > mean + threshold * std
detector = PopulationDriftDetector(
    threshold=3,
    threshold_method='std',
)

# Quantile method: flag if statistic > empirical quantile
detector = PopulationDriftDetector(
    threshold=0.95,             # 95th percentile
    threshold_method='quantile',
)
```

## Integration with Forecasting Pipeline

```python
from skforecast.recursive import ForecasterRecursive
from skforecast.drift_detection import RangeDriftDetector

# Train
forecaster = ForecasterRecursive(estimator=regressor, lags=24)
forecaster.fit(y=y_train, exog=exog_train)

# Set up monitoring
detector = RangeDriftDetector()
detector.fit(series=y_train, exog=exog_train)

# Production loop
def predict_with_monitoring(new_window, new_exog):
    flag, _, _ = detector.predict(
        last_window=new_window, exog=new_exog, verbose=False
    )
    if flag:
        print("Drift detected — consider retraining the model")
    return forecaster.predict(steps=10, exog=new_exog)
```

## Common Mistakes

1. **Fitting detector on test data**: Always fit on training data — the reference distribution.
2. **Ignoring drift signals**: Drift doesn't mean the model is wrong, but it signals degradation risk.
3. **Over-sensitive thresholds**: Start with `threshold=3` (3 sigma) and adjust based on false positive rate.
