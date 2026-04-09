# Deep Learning — Architecture Options Reference

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
# Architecture: Input → LSTM(64) → Dense(32) → Dense(24)
```

### Stacked recurrent layers

```python
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels='target',
    recurrent_layer='LSTM',
    recurrent_units=[128, 64, 32],  # list → 3 stacked LSTM layers
    dense_units=[64, 32],            # list → 2 dense layers
)
# Architecture: Input → LSTM(128) → LSTM(64) → LSTM(32) → Dense(64) → Dense(32) → Dense(24)
```

### No dense layers

```python
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels='target',
    recurrent_layer='GRU',
    recurrent_units=64,
    dense_units=None,               # None → no intermediate dense layers
)
# Architecture: Input → GRU(64) → Dense(24)  (output layer only)
```

## Output Layer Shape

The output layer is automatically sized based on `steps` and `levels`:

| Configuration | Output Dense units | Output shape |
|---------------|-------------------|--------------|
| 1 level, N steps | `steps` | `(batch, steps)` |
| M levels, N steps | `steps × M` + Reshape | `(batch, steps, M)` |

```python
# Single level
model = create_and_compile_model(
    series=series, lags=48, steps=24,
    levels='target',                 # 1 level → output: Dense(24)
)

# Multiple levels
model = create_and_compile_model(
    series=series, lags=48, steps=24,
    levels=['series_1', 'series_2', 'series_3'],  # 3 levels → output: Dense(72) + Reshape(24,3)
)
```

## Exogenous Variables Architecture

When `exog` is provided, the model creates a separate input branch with
`TimeDistributed` layers:

```python
# Without exog: single input branch
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels='target',
)
# Input shape: (batch, 48, n_series)

# With exog: two input branches merged
model = create_and_compile_model(
    series=series, lags=48, steps=24, levels='target',
    exog=exog_df,                    # Must be provided at model creation
)
# Input shapes: series_input (batch, 48, n_series) + exog_input (batch, 48, n_exog)
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
    levels,                          # str | list[str] (required)
    lags,                            # int | list[int] | np.ndarray | range (required)
    estimator=None,                  # Keras model (from create_and_compile_model or custom)
    transformer_series=MinMaxScaler(feature_range=(0, 1)),  # default: MinMaxScaler
    transformer_exog=MinMaxScaler(feature_range=(0, 1)),    # default: MinMaxScaler
    fit_kwargs=None,                 # dict, kwargs for model.fit()
    forecaster_id=None,              # str | int
)
```

### fit_kwargs

```python
forecaster = ForecasterRnn(
    levels=series.columns.tolist(),
    lags=48,
    estimator=model,
    fit_kwargs={
        'epochs': 50,                # number of training epochs
        'batch_size': 32,            # training batch size
        'verbose': 0,                # 0=silent, 1=progress bar, 2=one line per epoch
        'validation_split': 0.1,     # fraction for validation
        'callbacks': [EarlyStopping(patience=5)],  # Keras callbacks
    },
)
```

## Custom Keras Model Requirements

When building a custom model instead of using `create_and_compile_model`:

### Single level (no exog)

```python
import keras

inputs = keras.layers.Input(shape=(lags, n_series))  # (lags, number of input series)
x = keras.layers.LSTM(64, return_sequences=True)(inputs)
x = keras.layers.LSTM(32)(x)
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(steps)(x)               # units = steps

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
```

### Multiple levels (no exog)

```python
inputs = keras.layers.Input(shape=(lags, n_series))
x = keras.layers.LSTM(64, return_sequences=True)(inputs)
x = keras.layers.LSTM(32)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dense(steps * n_levels)(x)           # units = steps × n_levels
outputs = keras.layers.Reshape((steps, n_levels))(x)   # reshape required

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
```

### Input shape reference

| Scenario | Input shape | Output shape |
|----------|-------------|-------------|
| 1 level, no exog | `(lags, 1)` | `(steps,)` |
| M levels, no exog | `(lags, M)` | `(steps, M)` after Reshape |
| 1 level, K exog features | Two inputs: `(lags, 1)` + `(lags, K)` | `(steps,)` |
| M levels, K exog features | Two inputs: `(lags, M)` + `(lags, K)` | `(steps, M)` after Reshape |

## Prediction Intervals

ForecasterRnn only supports conformal prediction:

```python
forecaster.fit(series=series, store_in_sample_residuals=True)

predictions = forecaster.predict_interval(
    steps=24,
    method='conformal',           # only valid method
    interval=[10, 90],
    use_in_sample_residuals=True,
)
# NOTE: 'bootstrapping' is NOT supported for ForecasterRnn
# NOTE: predict_quantiles() and predict_dist() are NOT available
```
