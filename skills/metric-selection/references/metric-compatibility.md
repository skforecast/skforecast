# Metric Compatibility Matrix

Complete reference for all metrics available in skforecast: properties, compatibility, and usage guidance.

## Point Forecast Metrics (Regression)

| Metric | String name | Source | Requires `y_train` | Scale-independent | Robust to outliers | Handles zeros | Range | Interpretation |
|--------|-------------|--------|:-------------------:|:-----------------:|:------------------:|:-------------:|-------|----------------|
| Mean Absolute Error | `'mean_absolute_error'` | sklearn | — | — | ✓ | ✓ | [0, ∞) | Average absolute deviation from truth |
| Mean Squared Error | `'mean_squared_error'` | sklearn | — | — | — | ✓ | [0, ∞) | Average squared deviation; penalizes large errors |
| Median Absolute Error | `'median_absolute_error'` | sklearn | — | — | ✓✓ | ✓ | [0, ∞) | Median of absolute deviations; very robust |
| Mean Absolute Percentage Error | `'mean_absolute_percentage_error'` | sklearn | — | ✓ | — | — | [0, ∞) | Percentage error; undefined when y_true = 0 |
| Mean Squared Log Error | `'mean_squared_log_error'` | sklearn | — | — | — | ✓* | [0, ∞) | Log-scale MSE; penalizes under-predictions more |
| Symmetric MAPE | `'symmetric_mean_absolute_percentage_error'` | skforecast | — | ✓ | — | ✓ | [0, 200] % | Symmetric percentage; avoids MAPE's asymmetry |
| Mean Absolute Scaled Error | `'mean_absolute_scaled_error'` | skforecast | ✓ | ✓ | ✓ | ✓ | [0, ∞) | Ratio vs naive forecast; < 1 = better than naive |
| Root Mean Squared Scaled Error | `'root_mean_squared_scaled_error'` | skforecast | ✓ | ✓ | — | ✓ | [0, ∞) | RMSE ratio vs naive; < 1 = better than naive |

*MSLE requires y_true ≥ 0 and y_pred ≥ 0 (typically used with positive data).

## Probabilistic Forecast Metrics

| Metric | Function | Input signature | Use with | Measures |
|--------|----------|-----------------|----------|----------|
| Coverage | `calculate_coverage` | `(y_true, lower_bound, upper_bound)` | Any forecaster with intervals | Calibration (proportion of truths in interval) |
| CRPS (bootstrap) | `crps_from_predictions` | `(y_true: float, y_pred: ndarray)` | Bootstrapped predictions | Calibration + sharpness (single observation) |
| CRPS (quantiles) | `crps_from_quantiles` | `(y_true: float, pred_quantiles: ndarray, quantile_levels: ndarray)` | Foundation models, quantile predictions | Calibration + sharpness (single observation) |
| Pinball Loss | `create_mean_pinball_loss(alpha)` | Returns `func(y_true, y_pred)` | Any quantile forecast at level alpha | Single-quantile accuracy |

## Classification Metrics

| Metric | String name | Source | Best for |
|--------|-------------|--------|----------|
| Accuracy | `'accuracy_score'` | sklearn | Balanced classes |
| Balanced Accuracy | `'balanced_accuracy_score'` | sklearn | Imbalanced classes (recommended default) |
| F1 Score | `'f1_score'` | sklearn | Balance of precision and recall |
| Precision | `'precision_score'` | sklearn | Minimizing false positives |
| Recall | `'recall_score'` | sklearn | Minimizing false negatives |

## Forecaster × Metric Type Compatibility

| Forecaster | Point metrics | Probabilistic metrics | Classification metrics |
|------------|:-------------:|:---------------------:|:---------------------:|
| `ForecasterRecursive` | ✓ | ✓ (bootstrapping, conformal) | — |
| `ForecasterDirect` | ✓ | ✓ (bootstrapping, conformal) | — |
| `ForecasterRecursiveMultiSeries` | ✓ | ✓ (bootstrapping, conformal) | — |
| `ForecasterDirectMultiVariate` | ✓ | ✓ (bootstrapping, conformal) | — |
| `ForecasterRnn` | ✓ | ✓ (conformal) | — |
| `ForecasterStats` | ✓ | ✓ (native parametric) | — |
| `ForecasterFoundation` | ✓ | ✓ (native quantiles) | — |
| `ForecasterEquivalentDate` | ✓ | ✓ (conformal) | — |
| `ForecasterRecursiveClassifier` | — | — | ✓ |

## When to Use / When to Avoid

| Metric | Use when | Avoid when |
|--------|----------|------------|
| MAE | General purpose; robust default; easy to explain | Need to heavily penalize large errors |
| MSE | Large errors should be penalized disproportionally | Data has outliers (a few bad predictions dominate) |
| MedAE | Data is noisy with frequent outliers | Need sensitivity to all errors (median ignores tails) |
| MAPE | Need percentage interpretation; all values are non-zero | Data contains zeros or near-zero values |
| SMAPE | Need percentage interpretation; data may have zeros | Need strict mathematical properties (SMAPE has known biases) |
| MSLE | Positive data where relative errors matter more than absolute | Data has zeros or negative values |
| MASE | Comparing across series with different scales; need baseline reference | Very short training series (naive denominator unreliable) |
| RMSSE | Same as MASE but want to penalize large errors more | Same as MASE |
| Coverage | Evaluating prediction interval calibration | Want to measure interval width/sharpness (use CRPS) |
| CRPS | Comprehensive interval/quantile evaluation (calibration + sharpness) | Only have point forecasts |
| Pinball loss | Evaluating a specific quantile (e.g., 90th percentile for risk) | Need overall distributional assessment (use CRPS) |
| Balanced accuracy | Classification with imbalanced classes | Perfectly balanced classes (accuracy is simpler) |
| F1 | Classification where both false positives and negatives matter | Need per-class detail (use precision/recall separately) |

## Multi-Series Aggregation Options

Used in `backtesting_forecaster_multiseries` (via `add_aggregated_metric=True`) and
multi-series search functions (via `aggregate_metric` parameter).

| Aggregation | Behavior | Best for |
|-------------|----------|----------|
| `'average'` | Arithmetic mean of per-series metric values | All series equally important |
| `'weighted_average'` | Weighted by number of predicted values per level | Series with more observations contribute more |
| `'pooling'` | Metric computed on all predictions concatenated | Global accuracy regardless of series identity |

In backtesting, set `add_aggregated_metric=True` (default) to include all three.
In search functions, use `aggregate_metric` to select which to report (default: all three).
