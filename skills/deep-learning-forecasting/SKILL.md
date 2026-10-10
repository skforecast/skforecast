---
name: deep-learning-forecasting
description: >
  Forecasts time series using recurrent neural networks (RNN, LSTM, GRU) with
  ForecasterRnn and the create_and_compile_model helper. Covers model
  architecture, training, and multi-series deep learning.
  Use when the user wants to use deep learning / neural networks for
  time series forecasting.
---

# Deep Learning Forecasting (RNN/LSTM)

## When to Use

Use `ForecasterRnn` when:
- You have large datasets (thousands of observations)
- Complex nonlinear patterns that tree-based models struggle with
- Multi-series problems where series share deep temporal patterns

**Requirements**: `pip install skforecast[deeplearning]` (installs keras)

### Related skills

- **Prerequisite**: `choosing-a-forecaster` (confirm `ForecasterRnn` is the right choice for the data size and pattern)
- **Prerequisite**: `feature-engineering` (RNN models still benefit from cyclical / calendar exogenous features)
- **Next**: `hyperparameter-optimization` (tune RNN architecture and training hyperparameters)
- **Next**: `prediction-intervals` (only conformal intervals are supported for `ForecasterRnn`)

## Stop Conditions

Scan before writing code. Each row lists a rule, the symptom when it is broken, and the recovery. Full pitfall catalog: the `troubleshooting-common-errors` skill.

| Rule | Symptom | Recovery |
|------|---------|----------|
| `lags` in `ForecasterRnn` must match `create_and_compile_model(..., lags=...)` | Input shape mismatch error during fit | Use the same `lags` value in both calls |
| `ForecasterRnn` supports only `method='conformal'` for intervals | Error when calling `predict_interval(method='bootstrapping')` | Use `method='conformal'` |
| Pass `exog` to `create_and_compile_model` if exog is used in `fit()` / `predict()` | Architecture mismatch or failure when predicting with exog | Build the model with the same `exog` you train on |
| Scale inputs with `transformer_series=MinMaxScaler()` | Poor convergence; RNNs are scale-sensitive | Always set a scaler on `transformer_series` |
| `fit()` does not reset the network weights | Calling `fit()` again (or backtesting a trained forecaster) resumes training from the learned weights | Create a new `ForecasterRnn` from the untrained model (the forecaster stores a copy, so the original `model` object stays untrained) |
| Custom Keras models must use the layer names of `create_and_compile_model` | `ValueError: No such layer: series_input` (or `output_dense_td_layer`) when creating the forecaster | Name the inputs `series_input` / `exog_input`, the output layer `output_dense_td_layer`, and return shape `(steps, n_levels)` |

## Quick Start

```python
import pandas as pd
from skforecast.deep_learning import ForecasterRnn, create_and_compile_model
from sklearn.preprocessing import MinMaxScaler

# 1. Prepare data (DataFrame with DatetimeIndex, columns = series)
series = pd.read_csv('data.csv', index_col='date', parse_dates=True)
series = series.asfreq('h')

# 2. Create and compile a Keras model
model = create_and_compile_model(
    series=series,
    lags=48,
    steps=24,
    levels=series.columns.tolist(),  # All series
    recurrent_layer='LSTM',          # 'LSTM', 'GRU', or 'RNN'
    recurrent_units=[64, 32],        # Units per recurrent layer
    dense_units=[32],                # Units per dense layer
    compile_kwargs={'optimizer': 'adam', 'loss': 'mse'},
)

# 3. Create forecaster
forecaster = ForecasterRnn(
    levels=series.columns.tolist(),
    lags=48,
    estimator=model,
    transformer_series=MinMaxScaler(feature_range=(0, 1)),
    fit_kwargs={'epochs': 50, 'batch_size': 32, 'verbose': 0},
)

# 4. Train
forecaster.fit(series=series)

# 5. Predict
predictions = forecaster.predict(steps=24)
```

## Model Architecture with create_and_compile_model

```python
from skforecast.deep_learning import create_and_compile_model

# Simple LSTM
model = create_and_compile_model(
    series=series,
    lags=48,
    steps=24,
    levels='target',
    recurrent_layer='LSTM',
    recurrent_units=[64],
    dense_units=[32],
    compile_kwargs={'optimizer': 'adam', 'loss': 'mse'},
)

# Stacked LSTM (multiple recurrent layers)
model = create_and_compile_model(
    series=series,
    lags=48,
    steps=24,
    levels=series.columns.tolist(),
    recurrent_layer='LSTM',
    recurrent_units=[128, 64, 32],  # 3 stacked LSTM layers
    dense_units=[64, 32],           # 2 dense layers
    compile_kwargs={'optimizer': 'adam', 'loss': 'mse'},
)

# GRU variant (faster training)
model = create_and_compile_model(
    series=series,
    lags=48,
    steps=24,
    levels='target',
    recurrent_layer='GRU',
    recurrent_units=[64],
    dense_units=[32],
    compile_kwargs={'optimizer': 'adam', 'loss': 'mse'},
)

# Advanced: customize layer kwargs
model = create_and_compile_model(
    series=series,
    lags=48,
    steps=24,
    levels=series.columns.tolist(),
    recurrent_layer='LSTM',
    recurrent_units=[128, 64],
    recurrent_layers_kwargs={'activation': 'tanh'},   # default
    dense_units=[64],
    dense_layers_kwargs={'activation': 'relu'},        # default
    output_dense_layer_kwargs={'activation': 'linear'}, # default
    compile_kwargs={'optimizer': 'adam', 'loss': 'mse'},
    model_name='my_lstm_model',
)
```

## With Exogenous Variables

When using exogenous variables, pass `exog` to `create_and_compile_model` so it
builds the correct architecture (uses `TimeDistributed` layers internally).

```python
# exog must be a DataFrame covering the training period
exog = pd.DataFrame({'temperature': [...], 'holiday': [...]}, index=series.index)

model = create_and_compile_model(
    series=series,
    lags=48,
    steps=24,
    levels=series.columns.tolist(),
    exog=exog,                        # Passes exog info to build architecture
    recurrent_layer='LSTM',
    recurrent_units=[64, 32],
    dense_units=[32],
    compile_kwargs={'optimizer': 'adam', 'loss': 'mse'},
)

forecaster = ForecasterRnn(
    levels=series.columns.tolist(),
    lags=48,
    estimator=model,
    fit_kwargs={'epochs': 50, 'batch_size': 32, 'verbose': 0},
)

forecaster.fit(series=series, exog=exog)
predictions = forecaster.predict(steps=24, exog=exog_test)  # exog_test covers forecast horizon
```

## Custom Keras Model

`ForecasterRnn` finds the inputs and the output of the model by layer name, so a
custom model must follow the conventions of `create_and_compile_model`: input
named `series_input`, output layer named `output_dense_td_layer`, and model
output of shape `(steps, n_levels)`.

```python
import keras

# Build your own model for full control (single level, no exog)
lags, steps, n_series, n_levels = 48, 24, 1, 1

inputs = keras.layers.Input(shape=(lags, n_series), name='series_input')
x = keras.layers.LSTM(64, return_sequences=True)(inputs)
x = keras.layers.LSTM(32)(x)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dense(steps * n_levels, name='output_dense_td_layer')(x)
outputs = keras.layers.Reshape((steps, n_levels))(x)  # Output shape (steps, n_levels)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

forecaster = ForecasterRnn(
    levels='target',
    lags=48,
    estimator=model,
    transformer_series=MinMaxScaler(feature_range=(0, 1)),
    fit_kwargs={'epochs': 100, 'batch_size': 32},
)
```

> **Multi-series custom model**: For N levels, the output layer should be
> `Dense(steps * n_levels, name='output_dense_td_layer')` followed by
> `Reshape((steps, n_levels))`. For models with exogenous variables, see
> [references/architecture-options.md](references/architecture-options.md).

## Training with Validation and Early Stopping

Split the data chronologically into train, validation and test. Pass the
validation data through `fit_kwargs` with the keys `series_val` (and `exog_val`
if the model uses exog). They are not Keras arguments: the forecaster extracts
them to build the validation set and passes the remaining keys to `model.fit()`.

```python
import keras
import numpy as np
from keras.callbacks import EarlyStopping

# Chronological split (never random)
n_val, n_test = 24 * 60, 24 * 30
series_train = series.iloc[:-(n_val + n_test)]
series_val = series.iloc[-(n_val + n_test):-n_test]

keras.utils.set_random_seed(123)  # Reproducible weight initialization and training
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels=['target'], recurrent_layer='GRU'
)

forecaster = ForecasterRnn(
    estimator=model,
    levels=['target'],
    lags=48,
    transformer_series=MinMaxScaler(),
    fit_kwargs={
        'epochs': 50,            # Maximum, early stopping decides when to stop
        'batch_size': 512,
        'callbacks': [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ],
        'series_val': series_val,  # Validation series (exog_val too if exog is used)
    },
)
forecaster.fit(series=series_train)
forecaster.plot_history()  # Training vs validation loss per epoch

best_epochs = int(np.argmin(forecaster.history_['val_loss'])) + 1
```

`ForecasterRnn` stores a copy of the model it receives, and `fit()` does not
reset the weights: calling it again resumes training from the learned weights.
To train from scratch, create a new `ForecasterRnn` from the original `model`,
which is still untrained.

## Backtesting

For the final evaluation, add the validation partition to the training set.
Early stopping is no longer possible, so train a new network for the number of
epochs identified before.

```python
from skforecast.model_selection import backtesting_forecaster_multiseries, TimeSeriesFold

cv = TimeSeriesFold(
    steps=forecaster.max_step,
    initial_train_size=len(series) - n_test,  # Train + validation
    refit=False,  # Retraining RNNs is expensive; set True only if needed
)

keras.utils.set_random_seed(123)
forecaster_bt = ForecasterRnn(
    estimator=model,  # Untrained: the first forecaster trained its own copy
    levels=['target'],
    lags=48,
    transformer_series=MinMaxScaler(),
    fit_kwargs={'epochs': best_epochs, 'batch_size': 512, 'verbose': 0},
)

metric, predictions = backtesting_forecaster_multiseries(
    forecaster=forecaster_bt,
    series=series,
    cv=cv,
    metric='mean_absolute_error',
)
```

## Prediction Intervals

`ForecasterRnn` supports conformal prediction only. In-sample residuals
(`store_in_sample_residuals=True` in `fit()`, or `set_in_sample_residuals()`)
come from data the network was trained on, so they are too small and the
intervals are usually too narrow. Prefer out-of-sample residuals computed on a
calibration set that was not used to fit the network (for example, the
validation partition).

```python
# Predictions on the calibration set with the trained forecaster
# initial_train_size=None: backtesting uses the forecaster as it is, no refit
cv_cal = TimeSeriesFold(steps=forecaster.max_step, initial_train_size=None, refit=False)
_, pred_cal = backtesting_forecaster_multiseries(
    forecaster=forecaster, series=series_val, cv=cv_cal, metric='mean_absolute_error'
)
pred_target = pred_cal.loc[pred_cal['level'] == 'target', 'pred']

# y_true and y_pred in the original scale, the forecaster applies the transformation
forecaster.set_out_sample_residuals(
    y_true={'target': series_val.loc[pred_target.index, 'target']},
    y_pred={'target': pred_target},
)

predictions = forecaster.predict_interval(
    steps=24,
    last_window=series_val,       # Forecast right after the calibration period
    method='conformal',           # Only 'conformal' supported
    interval=[0.1, 0.9],
    use_in_sample_residuals=False,
    use_binned_residuals=True,    # Better calibration with binned residuals
)
```

Conformal intervals add a margin to the point prediction, so the lower bound can
be negative for non-negative series; clip it at zero if needed.

## Common Mistakes

1. **Not scaling data**: RNNs are sensitive to scale. Always use `transformer_series=MinMaxScaler()`.
2. **Fixed number of epochs without validation**: Set a maximum (50-100) and let `EarlyStopping` with `series_val` in `fit_kwargs` decide when to stop.
3. **Wrong input shape**: The `lags` parameter in `ForecasterRnn` and `create_and_compile_model` must match.
4. **Refit=True in backtesting**: Retraining RNNs at every fold is very slow; use `refit=False` or `refit=5`.
5. **No GPU**: Training is slow on CPU. Use GPU if available.
6. **Using `predict_interval(method='bootstrapping')`**: ForecasterRnn only supports `method='conformal'`.
7. **Forgetting `exog` in `create_and_compile_model`**: If you use exog in `fit()`/`predict()`, you must also pass `exog` when building the model so the architecture accounts for the extra input features.
8. **Calling `fit()` again to retrain from scratch**: `fit()` resumes from the learned weights, and backtesting a trained forecaster continues its training. Create a new `ForecasterRnn` from the untrained model instead.
9. **Intervals from in-sample residuals only**: They are usually too narrow. Use `set_out_sample_residuals()` with predictions on a calibration set.
10. **Non-reproducible results**: Call `keras.utils.set_random_seed()` before creating each model.

## References

See [references/architecture-options.md](references/architecture-options.md) for
the complete `create_and_compile_model` signature, recurrent layer types,
output shape rules, exog architecture, custom Keras model requirements,
and `fit_kwargs` options.
