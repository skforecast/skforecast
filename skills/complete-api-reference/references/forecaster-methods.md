# Forecaster Methods

Method signatures for every skforecast forecaster, plus the availability matrix.

## Contents

- fit()
- predict()
- predict_interval()
- predict_quantiles()
- predict_dist()
- set_out_sample_residuals()
- set_params() and set_lags()
- Method availability matrix

## Forecaster Methods: fit()

```python
# ForecasterRecursive, ForecasterDirect
forecaster.fit(
    y,                                # pd.Series with DatetimeIndex (required)
    exog=None,                        # pd.Series | pd.DataFrame | None
    store_last_window=True,           # bool
    store_in_sample_residuals=False,  # bool, set True before using predict_interval()
    random_state=123,                 # int, seed for residual sampling
    suppress_warnings=False           # bool
)

# ForecasterRecursiveMultiSeries
forecaster.fit(
    series,                           # pd.DataFrame | dict[str, pd.Series|pd.DataFrame] (required)
    exog=None,                        # pd.Series | pd.DataFrame | dict[str, pd.Series|pd.DataFrame] | None
    store_last_window=True,           # bool | list[str], True stores all, list stores specific series
    store_in_sample_residuals=False,  # bool, set True before using predict_interval()
    random_state=123,                 # int
    suppress_warnings=False           # bool
)

# ForecasterDirectMultiVariate, ForecasterRnn
forecaster.fit(
    series,                           # pd.DataFrame with multiple columns (required)
    exog=None,                        # pd.Series | pd.DataFrame | None
    store_last_window=True,           # bool
    store_in_sample_residuals=False,  # bool
    random_state=123,                 # int
    suppress_warnings=False           # bool
)

# ForecasterRecursiveClassifier
forecaster.fit(
    y,                                # pd.Series with DatetimeIndex (required)
    exog=None,                        # pd.Series | pd.DataFrame | None
    store_last_window=True,           # bool
    suppress_warnings=False           # bool
)
# NOTE: No store_in_sample_residuals or random_state.

# ForecasterStats
forecaster.fit(
    y,                                # pd.Series with DatetimeIndex (required)
    exog=None,                        # pd.Series | pd.DataFrame | None
    store_last_window=True,           # bool
    suppress_warnings=False           # bool
)

# ForecasterEquivalentDate
forecaster.fit(
    y,                                # pd.Series with DatetimeIndex (required)
    store_in_sample_residuals=False,  # bool
    random_state=123,                 # int
    suppress_warnings=False           # bool
)
# NOTE: No exog parameter (uses date offsets, not exogenous variables).

# ForecasterFoundation
forecaster.fit(
    series,                           # pd.Series | pd.DataFrame | dict[str, pd.Series] (required)
    exog=None,                        # pd.Series | pd.DataFrame | dict | None (Chronos-2 only)
)
# NOTE: "fit" does not train the model — it only stores the last
# context_length observations and metadata. Foundation models are
# pre-trained; training happens upstream on HuggingFace.
```

## Forecaster Methods: predict()

```python
# ForecasterRecursive
forecaster.predict(
    steps,                    # int | str | pd.Timestamp (required)
    last_window=None,         # pd.Series | pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | None
    check_inputs=True,        # bool
    suppress_warnings=False   # bool
) -> pd.Series

# ForecasterRecursiveMultiSeries
forecaster.predict(
    steps,                    # int (required)
    levels=None,              # str | list[str] | None, which series to predict
    last_window=None,         # pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | dict | None
    suppress_warnings=False,  # bool
    check_inputs=True         # bool
) -> pd.DataFrame

# ForecasterDirect
forecaster.predict(
    steps=None,               # int | list[int] | None, subset of trained steps
    last_window=None,         # pd.Series | pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | None
    check_inputs=True,        # bool
    suppress_warnings=False   # bool
) -> pd.Series

# ForecasterDirectMultiVariate
forecaster.predict(
    steps=None,               # int | list[int] | None, subset of trained steps
    last_window=None,         # pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | None
    suppress_warnings=False,  # bool
    check_inputs=True         # bool
) -> pd.DataFrame

# ForecasterRecursiveClassifier
forecaster.predict(
    steps,                    # int | str | pd.Timestamp (required)
    last_window=None,         # pd.Series | pd.DataFrame | None
    exog=None                 # pd.Series | pd.DataFrame | None
) -> pd.Series
# Also: predict_proba(steps, last_window=None, exog=None) -> pd.DataFrame

# ForecasterStats
forecaster.predict(
    steps,                    # int (required)
    last_window=None,         # pd.Series | None
    last_window_exog=None,    # pd.Series | pd.DataFrame | None, exog for last_window period
    exog=None,                # pd.Series | pd.DataFrame | None, exog for forecast period
    suppress_warnings=False   # bool
) -> pd.Series | pd.DataFrame

# ForecasterEquivalentDate
forecaster.predict(
    steps,                    # int (required)
    last_window=None,         # pd.Series | None
    check_inputs=True,        # bool
    suppress_warnings=False   # bool
) -> pd.Series

# ForecasterRnn
forecaster.predict(
    steps=None,               # int | list[int] | None, subset of trained steps
    levels=None,              # str | list[str] | None, which series to predict
    last_window=None,         # pd.DataFrame | None
    exog=None,                # pd.Series | pd.DataFrame | None
    suppress_warnings=False,  # bool
    check_inputs=True         # bool
) -> pd.DataFrame

# ForecasterFoundation
forecaster.predict(
    steps,                    # int (required)
    levels=None,              # str | list[str] | None, subset of series
    context=None,             # pd.Series | pd.DataFrame | dict | None, override stored context
    context_exog=None,        # pd.Series | pd.DataFrame | dict | None, historical exog
    exog=None,                # pd.Series | pd.DataFrame | dict | None, future exog (Chronos-2 only)
    check_inputs=True         # bool
) -> pd.DataFrame             # Long-format: columns ['level', 'pred']
# Also:
#   predict_interval(steps, ..., interval=[0.1, 0.9]) -> ['level','pred','lower_bound','upper_bound']
#   predict_quantiles(steps, ..., quantiles=[0.1, 0.5, 0.9]) -> ['level','q_0.1','q_0.5','q_0.9']
```

## Forecaster Methods: predict_interval()

```python
# ForecasterRecursive
forecaster.predict_interval(
    steps,                              # int | str | pd.Timestamp (required)
    last_window=None,                   # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    method='bootstrapping',             # 'bootstrapping' | 'conformal'
    interval=[0.05, 0.95],              # float (coverage) | list[float] | tuple[float], quantiles 0-1
    n_boot=250,                         # int, number of bootstrap samples
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterRecursiveMultiSeries (NOTE: default method='conformal')
forecaster.predict_interval(
    steps,                              # int (required)
    levels=None,                        # str | list[str] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | dict | None
    method='conformal',                 # 'bootstrapping' | 'conformal'
    interval=[0.05, 0.95],              # float (coverage) | list[float] | tuple[float], quantiles 0-1
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterDirect
forecaster.predict_interval(
    steps=None,                         # int | list[int] | None
    last_window=None,                   # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    method='bootstrapping',             # 'bootstrapping' | 'conformal'
    interval=[0.05, 0.95],              # float (coverage) | list[float] | tuple[float], quantiles 0-1
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterDirectMultiVariate (NOTE: default method='conformal')
forecaster.predict_interval(
    steps=None,                         # int | list[int] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    method='conformal',                 # 'bootstrapping' | 'conformal'
    interval=[0.05, 0.95],              # float (coverage) | list[float] | tuple[float], quantiles 0-1
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterStats (NOTE: different interface — uses alpha, no method/n_boot)
forecaster.predict_interval(
    steps,                              # int (required)
    last_window=None,                   # pd.Series | None
    last_window_exog=None,              # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    alpha=0.05,                         # float, significance level
    interval=None,                      # list[float] | tuple[float] | None, quantiles 0-1
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterEquivalentDate (NOTE: only 'conformal' method supported)
forecaster.predict_interval(
    steps,                              # int (required)
    last_window=None,                   # pd.Series | None
    method='conformal',                 # only 'conformal' supported
    interval=[0.05, 0.95],              # float (coverage) | list[float] | tuple[float], quantiles 0-1
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=None,                  # Any, ignored (API compatibility)
    exog=None,                          # Any, ignored (API compatibility)
    n_boot=None,                        # Any, ignored (API compatibility)
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterRnn (NOTE: only 'conformal' method supported)
forecaster.predict_interval(
    steps=None,                         # int | list[int] | None
    levels=None,                        # str | list[str] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    method='conformal',                 # only 'conformal' supported
    interval=[0.05, 0.95],              # float (coverage) | list[float] | tuple[float], quantiles 0-1
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    suppress_warnings=False,            # bool
    n_boot=None,                        # Any, ignored (API compatibility)
    random_state=None,                  # Any, ignored (API compatibility)
) -> pd.DataFrame

# ForecasterRecursiveClassifier: No predict_interval(). Use predict_proba() instead.
```

## Forecaster Methods: predict_quantiles()

```python
# ForecasterRecursive
forecaster.predict_quantiles(
    steps,                              # int | str | pd.Timestamp (required)
    last_window=None,                   # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    quantiles=[0.05, 0.5, 0.95],       # list[float] | tuple[float]
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterRecursiveMultiSeries
forecaster.predict_quantiles(
    steps,                              # int (required)
    levels=None,                        # str | list[str] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | dict | None
    quantiles=[0.05, 0.5, 0.95],       # list[float] | tuple[float]
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterDirect
forecaster.predict_quantiles(
    steps=None,                         # int | list[int] | None
    last_window=None,                   # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    quantiles=[0.05, 0.5, 0.95],       # list[float] | tuple[float]
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterDirectMultiVariate
forecaster.predict_quantiles(
    steps=None,                         # int | list[int] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    quantiles=[0.05, 0.5, 0.95],       # list[float] | tuple[float]
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False,            # bool
    levels=None,                        # Any, ignored (API compatibility)
) -> pd.DataFrame

# NOT available in: ForecasterRecursiveClassifier, ForecasterStats,
#                    ForecasterEquivalentDate, ForecasterRnn
```

## Forecaster Methods: predict_dist()

```python
# ForecasterRecursive
forecaster.predict_dist(
    steps,                              # int | str | pd.Timestamp (required)
    distribution,                       # scipy.stats distribution object (required)
    last_window=None,                   # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterRecursiveMultiSeries
forecaster.predict_dist(
    steps,                              # int (required)
    distribution,                       # scipy.stats distribution object (required)
    levels=None,                        # str | list[str] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | dict | None
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterDirect
forecaster.predict_dist(
    distribution,                       # scipy.stats distribution object (required)
    steps=None,                         # int | list[int] | None
    last_window=None,                   # pd.Series | pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False             # bool
) -> pd.DataFrame

# ForecasterDirectMultiVariate
forecaster.predict_dist(
    distribution,                       # scipy.stats distribution object (required)
    steps=None,                         # int | list[int] | None
    last_window=None,                   # pd.DataFrame | None
    exog=None,                          # pd.Series | pd.DataFrame | None
    n_boot=250,                         # int
    use_in_sample_residuals=True,       # bool
    use_binned_residuals=True,          # bool
    random_state=123,                   # int
    suppress_warnings=False,            # bool
    levels=None,                        # Any, ignored (API compatibility)
) -> pd.DataFrame

# NOT available in: ForecasterRecursiveClassifier, ForecasterStats,
#                    ForecasterEquivalentDate, ForecasterRnn
```

## Forecaster Methods: set_out_sample_residuals()

```python
# ForecasterRecursive, ForecasterDirect
forecaster.set_out_sample_residuals(
    y_true,                  # np.ndarray | pd.Series (required)
    y_pred,                  # np.ndarray | pd.Series (required)
    append=False,            # bool, append to existing residuals
    random_state=123         # int
) -> None

# ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate, ForecasterRnn
forecaster.set_out_sample_residuals(
    y_true,                  # dict[str, np.ndarray | pd.Series] (required)
    y_pred,                  # dict[str, np.ndarray | pd.Series] (required)
    append=False,            # bool
    random_state=123         # int
) -> None

# ForecasterEquivalentDate (same as single series)
forecaster.set_out_sample_residuals(
    y_true,                  # np.ndarray | pd.Series (required)
    y_pred,                  # np.ndarray | pd.Series (required)
    append=False,            # bool
    random_state=123         # int
) -> None

# NOT available in: ForecasterRecursiveClassifier, ForecasterStats
```

## Forecaster Methods: set_params() and set_lags()

```python
# set_params — available in all forecasters except ForecasterEquivalentDate
forecaster.set_params(
    params                   # dict[str, object] (required)
) -> None
# ForecasterStats also accepts dict[str, dict] for multiple models

# set_lags — available in all forecasters except ForecasterStats and ForecasterEquivalentDate
forecaster.set_lags(
    lags=None                # int | list[int] | np.ndarray | range | None
) -> None
# ForecasterDirectMultiVariate also accepts dict[str, int | list]
# ForecasterRnn: set_lags() exists but is a no-op for API consistency
```

## Method Availability Matrix

| Method | Recursive | Direct | RecursiveMultiSeries | DirectMultiVariate | Rnn | Stats | EquivalentDate | Classifier |
|--------|:---------:|:------:|:-------------------:|:-----------------:|:---:|:-----:|:--------------:|:----------:|
| `predict()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `predict_interval()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| `predict_quantiles()` | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| `predict_dist()` | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| `predict_proba()` | — | — | — | — | — | — | — | ✓ |
| `set_params()` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| `set_lags()` | ✓ | ✓ | ✓ | ✓ | ✓* | — | — | ✓ |
| `set_out_sample_residuals()` | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | — |

> ✓ = supported, — = not available, ✓* = exists but is a no-op

