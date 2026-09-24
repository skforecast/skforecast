# Deep Learning: Architecture Options Reference

## Contents

- create_and_compile_model signature
- Recurrent layer types
- Architecture building blocks
- Output layer shape
- Exogenous variables architecture
- Layer kwargs customization
- ForecasterRnn constructor
- Custom Keras model requirements
- Prediction intervals

## create_and_compile_model Signature

```python
from skforecast.deep_learning import create_and_compile_model

model = create_and_compile_model(
    series,                          # pd.DataFrame (required), input time series
    lags,                            # int | list[int] | np.ndarray | range (required)
    steps,                           # int (required), forecast horizon
    levels=None,                     # str | list[str] | None, output series (None = all)
    exog=None,                       # pd.Series | pd.DataFrame | None
    recurrent_layer='LSTM',          # 'LSTM' | 'GRU' | 'RNN'
    recurrent_units=100,             # int | list[int], units per recurrent layer
    recurrent_layers_kwargs={'activation': 'tanh'},   # dict | list[dict] | None
    dense_units=64,                  # int | list[int] | None
    dense_layers_kwargs={'activation': 'relu'},       # dict | list[dict] | None
    output_dense_layer_kwargs={'activation': 'linear'},  # dict | None
    compile_kwargs={'optimizer': Adam(), 'loss': MeanSquaredError()},  # dict
    model_name=None,                 # str | None
)
```

## Recurrent Layer Types

| Layer | Class | Speed | Memory | Best for |
|-------|-------|-------|--------|----------|
| `'LSTM'` | Long Short-Term Memory | Slowest | Highest | Long-range dependencies, default choice |
| `'GRU'` | Gated Recurrent Unit | Medium | Medium | Faster training, comparable performance |
| `'RNN'` | Simple RNN | Fastest | Lowest | Short sequences only, rarely used |

## Architecture Building Blocks

### Single recurrent layer

```python
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels='target',
    recurrent_layer='LSTM',
    recurrent_units=64,           # single int → 1 recurrent layer
    dense_units=32,               # single int → 1 dense layer
)
# Architecture: Input → LSTM(64) → Dense(32) → Dense(24) → Reshape(24, 1)
```

### Stacked recurrent layers

```python
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels='target',
    recurrent_layer='LSTM',
    recurrent_units=[128, 64, 32],  # list → 3 stacked LSTM layers
    dense_units=[64, 32],            # list → 2 dense layers
)
# Architecture: Input → LSTM(128) → LSTM(64) → LSTM(32) → Dense(64) → Dense(32) → Dense(24) → Reshape(24, 1)
```

### No dense layers

```python
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels='target',
    recurrent_layer='GRU',
    recurrent_units=64,
    dense_units=None,               # None → no intermediate dense layers
)
# Architecture: Input → GRU(64) → Dense(24) → Reshape(24, 1)  (output layer only)
```

## Output Layer Shape

The output layer (always named `output_dense_td_layer`) is automatically sized
based on `steps` and `levels`. The model output is always 3D, `(batch, steps, n_levels)`,
even for a single level:

| Configuration | Output layer | Output shape |
|---------------|--------------|--------------|
| 1 level, no exog | `Dense(steps)` + `Reshape((steps, 1))` | `(batch, steps, 1)` |
| M levels, no exog | `Dense(steps × M)` + `Reshape((steps, M))` | `(batch, steps, M)` |
| M levels, with exog | `TimeDistributed(Dense(M))` | `(batch, steps, M)` |

```python
# Single level
model = create_and_compile_model(
    series=series, lags=48, steps=24,
    levels='target',                 # 1 level → output: Dense(24) + Reshape(24, 1)
)

# Multiple levels
model = create_and_compile_model(
    series=series, lags=48, steps=24,
    levels=['series_1', 'series_2', 'series_3'],  # 3 levels → output: Dense(72) + Reshape(24, 3)
)
```

## Exogenous Variables Architecture

When `exog` is provided, the model has a second input, `exog_input`, with the
values of the exogenous variables for each future step. The output of the
recurrent layers is repeated once per step (`RepeatVector`), concatenated with
the exogenous variables, and processed step by step with `TimeDistributed`
dense layers:

```python
# Without exog: single input branch
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels='target',
)
# Input shape: series_input (batch, 48, n_series)

# With exog: two input branches merged
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels='target',
    exog=exog_df,                    # Must be provided at model creation
)
# Input shapes: series_input (batch, 48, n_series) + exog_input (batch, 24, n_exog)
# Architecture: series_input → LSTM → RepeatVector(24) → Concatenate(exog_input)
#               → TimeDistributed(Dense) → TimeDistributed(Dense(n_levels))
```

> **Critical:** If you plan to use exog in `fit()` and `predict()`, you MUST
> pass `exog` to `create_and_compile_model()` so the architecture includes
> the exogenous input branch.

## Layer Kwargs Customization

### Same kwargs for all layers

```python
model = create_and_compile_model(
    ...,
    recurrent_units=[128, 64],
    recurrent_layers_kwargs={'activation': 'tanh', 'dropout': 0.2},  # applied to both
    dense_units=[64, 32],
    dense_layers_kwargs={'activation': 'relu'},  # applied to both
)
```

### Different kwargs per layer

```python
model = create_and_compile_model(
    ...,
    recurrent_units=[128, 64],
    recurrent_layers_kwargs=[
        {'activation': 'tanh', 'dropout': 0.3},  # first LSTM
        {'activation': 'tanh', 'dropout': 0.1},  # second LSTM
    ],
    dense_units=[64, 32],
    dense_layers_kwargs=[
        {'activation': 'relu'},
        {'activation': 'relu'},
    ],
)
```

## ForecasterRnn Constructor

```python
ForecasterRnn(
    estimator,                       # Keras model (required), from create_and_compile_model or custom
    levels,                          # str | list[str] (required)
    lags,                            # int | list[int] | np.ndarray | range (required)
    transformer_series=MinMaxScaler(feature_range=(0, 1)),  # default: MinMaxScaler
    transformer_exog=MinMaxScaler(feature_range=(0, 1)),    # default: MinMaxScaler
    fit_kwargs=None,                 # dict, kwargs for model.fit()
    binner_kwargs=None,              # dict, kwargs for QuantileBinner (binned residuals)
    forecaster_id=None,              # str | int
)
```

The forecaster stores a copy of `estimator` (the original object is not
trained). `fit()` does not reset the weights: calling it again resumes training
from the learned weights. To train from scratch, create a new `ForecasterRnn`.

### fit_kwargs

```python
forecaster = ForecasterRnn(
    estimator=model,
    levels=series.columns.tolist(),
    lags=48,
    fit_kwargs={
        'epochs': 50,                # maximum number of training epochs
        'batch_size': 32,            # training batch size
        'verbose': 0,                # 0=silent, 1=progress bar, 2=one line per epoch
        'callbacks': [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ],                           # Keras callbacks
        'series_val': series_val,    # validation series (not a Keras argument)
        'exog_val': exog_val,        # validation exog, required if the model uses exog
    },
)
```

`series_val` and `exog_val` are extracted by the forecaster, which builds the
validation windows and passes them to Keras as `validation_data`. The remaining
keys are passed to `model.fit()`. The training history is stored in
`forecaster.history_` and can be plotted with `forecaster.plot_history()`.

## Custom Keras Model Requirements

When building a custom model instead of using `create_and_compile_model`, follow
its layer names, because `ForecasterRnn` finds the inputs and the output by name:

- Series input named `series_input`, shape `(lags, n_series)`.
- Exogenous input (optional) named `exog_input`, shape `(steps, n_exog)`, passed
  as the second model input.
- Output layer named `output_dense_td_layer`.
- Model output of shape `(steps, n_levels)`, also for a single level.

Otherwise, creating the forecaster raises `ValueError: No such layer: series_input`
(or `output_dense_td_layer`), or `predict()` fails on a 2D output.

### Single or multiple levels (no exog)

```python
import keras

inputs = keras.layers.Input(shape=(lags, n_series), name='series_input')
x = keras.layers.LSTM(64, return_sequences=True)(inputs)
x = keras.layers.LSTM(32)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dense(steps * n_levels, name='output_dense_td_layer')(x)  # units = steps × n_levels
outputs = keras.layers.Reshape((steps, n_levels))(x)   # reshape required, also for 1 level

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
```

### With exogenous variables

```python
series_input = keras.layers.Input(shape=(lags, n_series), name='series_input')
exog_input = keras.layers.Input(shape=(steps, n_exog), name='exog_input')
x = keras.layers.LSTM(64)(series_input)
x = keras.layers.RepeatVector(steps)(x)                    # (steps, 64)
x = keras.layers.Concatenate(axis=-1)([x, exog_input])     # (steps, 64 + n_exog)
x = keras.layers.TimeDistributed(keras.layers.Dense(32, activation='relu'))(x)
outputs = keras.layers.TimeDistributed(
    keras.layers.Dense(n_levels), name='output_dense_td_layer'
)(x)                                                       # (steps, n_levels)

model = keras.Model(inputs=[series_input, exog_input], outputs=outputs)
model.compile(optimizer='adam', loss='mse')
```

### Input shape reference

| Scenario | Input shape | Output shape |
|----------|-------------|-------------|
| 1 level, no exog | `(lags, n_series)` | `(steps, 1)` after Reshape |
| M levels, no exog | `(lags, n_series)` | `(steps, M)` after Reshape |
| 1 level, K exog features | Two inputs: `(lags, n_series)` + `(steps, K)` | `(steps, 1)` |
| M levels, K exog features | Two inputs: `(lags, n_series)` + `(steps, K)` | `(steps, M)` |

`n_series` is the number of series used as predictors, which can be larger than
the number of levels (predicted series).

## Prediction Intervals

ForecasterRnn only supports conformal prediction:

```python
# In-sample residuals: residuals on the training data, intervals usually too narrow
forecaster.fit(series=series, store_in_sample_residuals=True)

predictions = forecaster.predict_interval(
    steps=24,
    method='conformal',           # only valid method
    interval=[0.1, 0.9],
    use_in_sample_residuals=True,
    use_binned_residuals=True,    # Better calibration with binned residuals
)
# NOTE: 'bootstrapping' is NOT supported for ForecasterRnn
# NOTE: predict_quantiles() and predict_dist() are NOT available
```

Out-of-sample residuals, computed on data not used to fit the network, give
intervals with a coverage closer to the nominal one. Predict a calibration set
with the trained forecaster (`initial_train_size=None` avoids refitting), store
the residuals with `set_out_sample_residuals()` (values in the original scale)
and set `use_in_sample_residuals=False`. See the Prediction Intervals section of
`SKILL.md` for a complete example.
