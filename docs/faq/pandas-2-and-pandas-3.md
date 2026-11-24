# pandas 2 and pandas 3

Since version 0.28.0, skforecast supports pandas 2.3 and pandas 3. With the same data, forecasters return the same predictions, prediction intervals and metrics with both versions, with two exceptions: [text columns that contain `None`](#text-columns-str-instead-of-object) and [`ForecasterEquivalentDate` with a `pd.offsets.Day` or `pd.DateOffset` offset and a time zone](#forecasterequivalentdate-with-a-time-zone).

What does change are the defaults that pandas 3 changes for every library: the dtype of text, the resolution of dates, the frequency of indexes converted between time zones and copy-on-write. This page lists how they show up in skforecast and how to write code that works with both versions.

The backends of `MoiraiAdapter` and `TabPFNAdapter` depend on `gluonts`, which requires pandas 2: installing them in an environment with pandas 3 downgrades it to pandas 2.3. For Moirai, use the command of the [foundation models guide](../user_guides/foundation-forecasting-models.ipynb) (`pip install uni2ts "pandas>=2.3"`): `pip install uni2ts` alone installs pandas 2.1, which skforecast does not support.


## Text columns: `str` instead of `object`

pandas 3 stores text with the `str` dtype instead of `object`. It appears in the outputs of skforecast: `exog_dtypes_in_`, the `level` column of the predictions of `ForecasterRecursiveMultiSeries`, the column names of `X_train`, etc. Code such as `X_train.columns.dtype == object` is no longer true with pandas 3; use `pd.api.types.is_string_dtype` to check for text in both versions.

The detection and encoding of categorical features (`categorical_features='auto'`) work the same with both dtypes, with one difference in missing values:

+ With pandas 3, `None` in a text column is a missing value (`NaN`), as `np.nan` already was in pandas 2.

+ With pandas 2, `None` in an `object` column is a value of its own, and the encoder gives it its own category.

So, with the same data, an estimator that does not accept missing values works with pandas 2 and raises `ValueError: Input X contains NaN` with pandas 3. Use `np.nan` for missing values, or a text such as `'unknown'` if they are a category, to get the same model with both versions.

A column with the `string[python]` dtype and missing values (`pd.NA`) raises a `TypeError` when the categorical features are encoded, with both versions. Convert it first:

```python
exog['col'] = exog['col'].astype(object).fillna(np.nan)
```


## Dates: resolution in microseconds

pandas 3 parses dates in microseconds: `pd.date_range`, `pd.to_datetime` and `read_csv(parse_dates=...)` return `datetime64[us]` instead of `datetime64[ns]`. With pandas 3, skforecast keeps the resolution of the input index in its outputs (predictions, training matrices, `last_window_`, backtesting), so they are usually `datetime64[us]`. With pandas 2, the index of the predictions is `datetime64[ns]` whatever the resolution of the input.

The dates are the same, only the dtype changes. Between two indexes with the same dates and different resolutions, `==` is true in both versions, `.equals` is true only with pandas 3, and `pd.testing.assert_index_equal` fails in both. To compare with results saved with pandas 2, convert the index:

```python
predictions.index = predictions.index.as_unit('ns')
```

For the same reason, with pandas 2, `y` and `exog` with different resolutions raise a `ValueError` in `fit` saying that their indexes are not aligned. Give both the same resolution with `.as_unit()`.


## Time zones: `tz_convert` drops the frequency

In pandas 3, `pd.offsets.Day` is a calendar day, not 24 hours. As a result, `tz_convert` drops calendar frequencies such as `'D'`, `'W'` or `'MS'`, and the forecasters raise a `ValueError` asking for a frequency:

```python
idx = pd.date_range('2021-02-15', periods=90, freq='D', tz='UTC').tz_convert('Europe/Madrid')
idx.freq  # pandas 2: <Day>, pandas 3: None
```

Reassigning the frequency (`pd.DatetimeIndex(idx, freq='D')`) also fails with pandas 3 when the dates cross a daylight saving time change. Depending on what the dates represent:

+ **Fixed steps in UTC** (dates created in UTC and converted to a local time zone): use `freq='24h'` for daily data or `freq='168h'` for weekly data. pandas keeps these frequencies in both versions, and the results are the same with both. With dates at UTC midnight, they are also identical to those of `freq='D'` with pandas 2, unless UTC midnight is also local midnight in winter (for example, `'Europe/London'` or `'Europe/Lisbon'`). In those time zones, and with dates at another UTC hour (for example, 12:00 UTC), use `'24h'` with pandas 2 too: with `'D'`, the predicted dates after a daylight saving time change can be shifted by one hour, which also gives wrong backtesting metrics. If the data already has such an index, set the frequency with `data.asfreq('24h')`: `data.asfreq('D')` moves to calendar days, and the values after a daylight saving time change become missing values. Keeping the index in UTC also works, and it is the option for monthly data (`'MS'`), which has no equivalent in hours.

+ **Calendar days in local time** (every date at local midnight): create the index in the local time zone (`pd.date_range(..., freq='D', tz='Europe/Madrid')`), or call `data.asfreq('D')`.

This also affects:

+ `calculate_distance_from_holiday`: the unit of the distances comes from the frequency of the index, so with `'24h'` or `'168h'` they are in hours (in both versions). Divide by 24 or 168 to get days or weeks. An index without frequency and fewer than three dates (for example, two dates converted from UTC with pandas 3) gives a warning and distances in hours.

+ Forecasters trained with pandas 2 on daily dates converted from UTC (frequency `'D'`) and loaded with pandas 3: `predict` without `last_window` works, but a `last_window` must have the frequency `'D'` (with `'24h'` it raises a `TypeError`), and pandas 3 only accepts `'D'` on those dates if they do not cross a daylight saving time change. Train the forecaster again with `freq='24h'`.


## `ForecasterEquivalentDate` with a time zone

The `offset` of `ForecasterEquivalentDate` can be an integer (a number of steps) or a pandas `DateOffset`. With an index with a time zone that crosses a daylight saving time change, an **integer offset** gives the same results in all cases and in both versions:

+ **Hourly data**: use `offset=24 * n` or `pd.offsets.Hour(24 * n)`. With pandas 2, `pd.offsets.Day(n)` is `n * 24` hours. With pandas 3 it keeps the local hour, so in the days after a daylight saving time change the equivalent dates, and the predictions, are different, or it raises a `ValueError` about a nonexistent or ambiguous time (the hour skipped in spring or repeated in autumn). `pd.DateOffset(days=n)` keeps the local hour in both versions: it raises the same kind of error, or repeats and skips hours.

+ **Daily data in fixed UTC steps** (`freq='24h'`): use `offset=n` or `pd.offsets.Hour(24 * n)`. `pd.offsets.Day(n)` only works with pandas 2.

+ **Daily data in local time**: use `offset=n` or `pd.DateOffset(days=n)`. `pd.offsets.Day(n)` only works with pandas 3.


## Read-only arrays

With copy-on-write, the default in pandas 3, `to_numpy()` and `.values` of a `DataFrame` or `Series` can return a read-only view, also for the outputs of skforecast (such as `X_train`). Use `to_numpy(copy=True)` to get an array you can modify. Modifying the `DataFrame` itself (`X_train.loc[...] = ...`) works in both versions.

With pandas 3, the arrays of `out_sample_residuals_` stored by `set_out_sample_residuals` are also read-only (in `ForecasterRecursiveMultiSeries`, the array of each series in its dict). Copy them with `.copy()` before modifying them.


## Saving and loading forecasters

+ A forecaster saved with pandas 2 loads with pandas 3. With `joblib`, `pickle` or `cloudpickle`, the pandas 3 environment needs `pytz` (`pip install pytz`) when the dates of the forecaster have a time zone that pandas 2 stores as a `pytz` object: named time zones such as `'Europe/Madrid'`, and dates with a time zone read from Arrow or Parquet. pandas 3 does not install `pytz`, and without it the file raises a `ModuleNotFoundError`, to which `load_forecaster` adds a note explaining it. With `pytz` installed, the forecaster loads and gives the same predictions. Saving it again with these backends keeps the requirement: to remove it, train the forecaster again with pandas 3, or save it with `backend='skops'` (not available for `ForecasterStats`, `ForecasterFoundation` or `ForecasterRnn`) and check that the new file loads before uninstalling `pytz`.

+ A forecaster saved with pandas 3 with `joblib`, `pickle` or `cloudpickle` and trained with dates (a `DatetimeIndex`, with or without time zone) cannot be loaded with pandas 2: pandas fails to rebuild the index, and `load_forecaster` adds a note to the error explaining it. Forecasters trained with a `RangeIndex` load and predict, but to share forecasters between versions use `backend='skops'`.

+ `backend='skops'` works in both directions between pandas 2.3 and pandas 3, and does not need `pytz`: it stores the time zone by its name. The list of types to trust depends on the pandas version that saved the file (for example, `pandas.core.indexes.datetimes.DatetimeIndex` in pandas 2 and `pandas.DatetimeIndex` in pandas 3, and `pandas.StringDtype` with text exogenous variables), so a list written for one version does not load files of the other. Get it from each file:

```python
from skops.io import get_untrusted_types
from skforecast.utils import load_forecaster

untrusted_types = get_untrusted_types(file='forecaster.skops')
forecaster = load_forecaster('forecaster.skops', trusted=untrusted_types)
```

See [Save and load forecasters](../user_guides/save-load-forecaster.ipynb) for more details.
