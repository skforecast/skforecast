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

## References

See [references/architecture-options.md](references/architecture-options.md) for
the complete `create_and_compile_model` signature, recurrent layer types,
output shape rules, exog architecture, custom Keras model requirements,
and `fit_kwargs` options.

## When to Use

Use `ForecasterRnn` when:
- You have large datasets (thousands of observations)
- Complex nonlinear patterns that tree-based models struggle with
- Multi-series problems where series share deep temporal patterns

**Requirements**: `pip install skforecast[deeplearning]` (installs keras)

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

```python
import keras

# Build your own model for full control (single level, no exog)
# Output units = steps * n_levels. For 1 level: steps. For N levels: steps * N + Reshape.
inputs = keras.layers.Input(shape=(48, 1))  # (lags, n_features)
x = keras.layers.LSTM(64, return_sequences=True)(inputs)
x = keras.layers.LSTM(32)(x)
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(24)(x)  # steps * n_levels (here 24 * 1)

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
> `Dense(steps * n_levels)` followed by `Reshape((steps, n_levels))`.

## Prediction Intervals

```python
# ForecasterRnn supports conformal prediction only
forecaster.fit(series=series, store_in_sample_residuals=True)

predictions = forecaster.predict_interval(
    steps=24,
    method='conformal',           # Only 'conformal' supported
    interval=[10, 90],
    use_in_sample_residuals=True,
)
```

## Backtesting

```python
from skforecast.model_selection import backtesting_forecaster_multiseries, TimeSeriesFold

cv = TimeSeriesFold(
    steps=24,
    initial_train_size=len(series) - 200,
    refit=False,  # Retraining RNNs is expensive; set True only if needed
)

metric, predictions = backtesting_forecaster_multiseries(
    forecaster=forecaster,
    series=series,
    cv=cv,
    metric='mean_absolute_error',
)
```

## Common Mistakes

1. **Not scaling data**: RNNs are sensitive to scale. Always use `transformer_series=MinMaxScaler()`.
2. **Too few epochs**: Deep learning needs more training iterations. Start with 50-100 epochs.
3. **Wrong input shape**: The `lags` parameter in `ForecasterRnn` and `create_and_compile_model` must match.
4. **Refit=True in backtesting**: Retraining RNNs at every fold is very slow — use `refit=False` or `refit=5`.
5. **No GPU**: Training is slow on CPU. Use GPU if available.
6. **Using `predict_interval(method='bootstrapping')`**: ForecasterRnn only supports `method='conformal'`.
7. **Forgetting `exog` in `create_and_compile_model`**: If you use exog in `fit()`/`predict()`, you must also pass `exog` when building the model so the architecture accounts for the extra input features.
