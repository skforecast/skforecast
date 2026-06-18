# Calendar Features — Reference

Two ways to add calendar/datetime features (month, day of week, hour, …) to a
forecaster:

- **Delegated** — pass a `CalendarFeatures` instance to the forecaster's
  `calendar_features` parameter. The forecaster generates the features
  automatically during **training and prediction**. No manual `exog`.
  **New in skforecast 0.23.0.**
- **Manual** — build the features yourself with `CalendarFeatures`
  (`fit_transform`) or `create_calendar_features` and pass the result as `exog`
  (or wire `CalendarFeatures` as `transformer_exog`). Required for forecasters
  that do not support the `calendar_features` parameter.

Both paths use the same `CalendarFeatures` class, so the constructor and
encoding options below apply to either workflow.

## Which workflow to use

| You want… | Use |
|-----------|-----|
| Simplest workflow, no manual exog at predict time | Delegated (`calendar_features=`) |
| One of the 4 supported forecasters (see table below) | Delegated |
| A forecaster without `calendar_features` support | Manual (`exog` / `transformer_exog`) |
| Calendar features inside a `Pipeline` / `ColumnTransformer` | Manual (`CalendarFeatures` as a transformer) |
| One-shot feature creation outside any forecaster | Manual (`create_calendar_features`) |

### Forecaster support for the `calendar_features` parameter

| Forecaster | `calendar_features` param | Calendar features path |
|------------|:-------------------------:|------------------------|
| `ForecasterRecursive` | ✓ | Delegated or manual |
| `ForecasterRecursiveMultiSeries` | ✓ | Delegated or manual |
| `ForecasterDirect` | ✓ | Delegated or manual |
| `ForecasterDirectMultiVariate` | ✓ | Delegated or manual |
| `ForecasterRecursiveClassifier` | ✗ | Manual only |
| `ForecasterRnn` | ✗ | Manual only |
| `ForecasterStats` | ✗ | Manual only |
| `ForecasterFoundation` | ✗ | Manual only |
| `ForecasterEquivalentDate` | ✗ | Manual only |

> All calendar tools are built into skforecast — no `feature_engine` (or any
> other extra) dependency is required.

## CalendarFeatures Constructor

```python
from skforecast.preprocessing import CalendarFeatures

calendar = CalendarFeatures(
    features=None,                   # list[str] | None — None extracts all supported features
    features_to_encode=None,         # list[str] | None — None encodes all encodable features
    encoding="cyclical",             # 'cyclical' | 'onehot' | 'spline' | None
    max_values=None,                 # dict[str, int] | None — override per-feature period
    spline_kwargs=None,              # dict | None — args for SplineTransformer (encoding='spline')
    keep_original_columns=True,      # bool — merge with the input X's columns
)
```

> The forecaster stores a **clone** of the instance you pass to
> `calendar_features`, so the same object can be reused safely.

### Supported features

`'year'`, `'month'`, `'week'`, `'day_of_week'`, `'day_of_month'`,
`'day_of_year'`, `'weekend'`, `'hour'`, `'minute'`, `'second'`, `'quarter'`.

By default all are extracted; pass `features=[...]` to subset.

### Supported encodings

| `encoding` | Output | Notes |
|-----------|--------|-------|
| `'cyclical'` (default) | `{feature}_sin`, `{feature}_cos` | sin/cos pair per cyclical feature |
| `'onehot'` | One column per known category (e.g. `month_1` … `month_12`) | Stable schema across train / predict |
| `'spline'` | `≈ max_val` columns per feature, periodic B-splines | Smooth alternative to onehot |
| `None` | Raw integer columns | No transformation |

`'year'` and `'weekend'` are **never** encoded (they are not cyclical) and are
always kept as raw integers regardless of `encoding`.

### Choosing an encoding

| Encoding | Columns per feature | Best for |
|----------|---------------------|----------|
| `'cyclical'` | 2 (sin + cos) | Compact, smooth — good default for tree and linear models |
| `'onehot'` | `max_val` | Stable schema; tree models can split on individual categories |
| `'spline'` | `≈ max_val` (dense) | Smooth + flexible; high-cardinality features (`day_of_year`) when memory allows |
| `None` | 1 | Tree models that benefit from raw ordinal values |

For high-cardinality features (`day_of_year` → 366, `day_of_month` → 31),
`'cyclical'` or `'spline'` are typically more memory-efficient than `'onehot'`.

### `max_values` defaults

```python
{'month': 12, 'week': 53, 'day_of_week': 7, 'day_of_month': 31,
 'day_of_year': 366, 'hour': 24, 'minute': 60, 'second': 60, 'quarter': 4}
```

These handle leap years and ISO week 53 correctly. Override only the keys you
need:

```python
calendar = CalendarFeatures(
    features=['month', 'hour'],
    encoding='cyclical',
    max_values={'month': 6},   # Custom semester period; hour keeps default 24
)
```

### `spline_kwargs`

`spline_kwargs` accepts any argument of
`sklearn.preprocessing.SplineTransformer` **except** `knots` (computed
internally from `max_values`) and `sparse_output` (incompatible with the
DataFrame output).

```python
calendar = CalendarFeatures(
    features=['day_of_year'],
    encoding='spline',
    spline_kwargs={'degree': 3, 'n_knots': 12},
)
```

### `features_to_encode` — encode only some features

Extract a feature but leave it as a raw integer:

```python
calendar = CalendarFeatures(
    features=['year', 'month', 'hour'],
    features_to_encode=['month', 'hour'],   # 'year' kept as raw int
    encoding='cyclical',
)
```

## Workflow A — Delegated (`calendar_features` parameter)

The forecaster generates calendar features from the datetime index during both
training and prediction — you never build or pass a calendar `exog`.

```python
from lightgbm import LGBMRegressor
from skforecast.preprocessing import CalendarFeatures
from skforecast.recursive import ForecasterRecursive

calendar = CalendarFeatures(
    features=['month', 'day_of_week', 'hour'],
    encoding='cyclical',
    keep_original_columns=False,
)

forecaster = ForecasterRecursive(
    estimator=LGBMRegressor(),
    lags=24,
    calendar_features=calendar,
)

# Index must be a pandas DatetimeIndex with frequency set
forecaster.fit(y=y_train)                       # calendar features built automatically
predictions = forecaster.predict(steps=24)      # calendar features built automatically
```

Key points:

- **Requires a `DatetimeIndex`.** A non-datetime index raises `TypeError`.
- **No horizon coverage needed.** The forecaster derives the prediction index
  itself, so unlike manual `exog` you do not pass future calendar values to
  `predict`.
- Calendar features are added as predictors **alongside** lags, window
  features, and any `exog`.
- Works in multi-series forecasters too: the same configuration is applied to
  the (shared) datetime index, producing identical calendar values per series.

### Related attributes (after `fit`)

| Attribute | Meaning |
|-----------|---------|
| `forecaster.calendar_features` | The cloned `CalendarFeatures` instance (or `None`) |
| `forecaster.calendar_features_names` | The `features` requested (e.g. `['month', 'hour']`) |
| `forecaster.X_train_calendar_features_names_out_` | Output column names added to `X_train` |

## Workflow B — Manual (`exog` / `transformer_exog` / function)

Use this for forecasters without `calendar_features` support, or when you need
the features inside a `Pipeline` / `ColumnTransformer`.

### As an exog DataFrame

```python
calendar = CalendarFeatures(
    features=['month', 'week', 'day_of_week', 'hour'],
    encoding='cyclical',
    keep_original_columns=False,
)
exog_calendar = calendar.fit_transform(data)
# Columns: month_sin, month_cos, week_sin, week_cos,
#          day_of_week_sin, day_of_week_cos, hour_sin, hour_cos

calendar.get_feature_names_out()   # resolved output column names (sklearn API)
```

The `exog` passed to `predict()` must include future dates covering the **entire
forecast horizon**.

### Function form — `create_calendar_features`

For one-shot use without instantiating a transformer:

```python
from skforecast.preprocessing import create_calendar_features

exog_calendar = create_calendar_features(
    X=data,
    features=['month', 'day_of_week', 'hour'],
    encoding='cyclical',
    keep_original_columns=False,
)
```

All parameters match `CalendarFeatures`. Prefer the transformer when you want to
fit / re-apply the same configuration, plug it into a `Pipeline`, or pass it as
`transformer_exog`.

## Gotchas

- **Datetime index required (delegated path).** A non-`DatetimeIndex` raises
  `TypeError`. Set the frequency first (`data.asfreq('h')`).
- **Do not mix paths for the same features.** Passing `calendar_features=` and
  also adding manually-built calendar columns to `exog` with the same names
  raises `ValueError: Duplicated feature names detected in X_train`. Pick one
  path per feature.
- **`keep_original_columns` default is `True`.** When used to build `exog`
  manually, set `keep_original_columns=False` if you only want the calendar
  columns (the constructor default keeps the input `X` columns too).
- **`max_values` for `'week'` / `'day_of_year'`.** Defaults (53, 366) handle ISO
  week 53 and leap years. Use 52 / 365 only after verifying your data never
  reaches the maximum.
