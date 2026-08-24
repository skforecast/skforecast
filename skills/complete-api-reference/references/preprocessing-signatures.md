# Preprocessing and Drift Detection Signatures

Constructor signatures for drift detectors and preprocessing transformers.

## Contents

- Drift detection classes
- Preprocessing classes

## Drift Detection Classes

```python
# RangeDriftDetector — lightweight out-of-range detector
RangeDriftDetector()  # No constructor parameters

RangeDriftDetector.fit(
    series=None,             # pd.DataFrame | pd.Series | dict | None
    exog=None,               # pd.DataFrame | pd.Series | dict | None
)

RangeDriftDetector.predict(
    last_window=None,        # pd.Series | pd.DataFrame | dict | None
    exog=None,               # pd.Series | pd.DataFrame | dict | None
    verbose=True,            # bool
    suppress_warnings=False  # bool
) -> tuple[bool, list[str], list[str] | dict[str, list[str]]]

# PopulationDriftDetector — statistical tests for distribution drift
PopulationDriftDetector(
    chunk_size=None,                    # int | str | None
    threshold=3,                        # int | float
    threshold_method='std',             # 'std' | 'quantile'
    max_out_of_range_proportion=0.1     # float
)

PopulationDriftDetector.fit(X)          # Reference dataset
PopulationDriftDetector.predict(X) -> tuple[pd.DataFrame, pd.DataFrame]
```

## Preprocessing Classes

```python
RollingFeatures(
    stats,                   # str | list[str], e.g. ['mean', 'std', 'min', 'max']
    window_sizes,            # int | list[int], int applies to all stats
    min_periods=None,        # int | list[int] | None
    features_names=None,     # list[str] | None, custom names for features
    fillna=None,             # str | float | None
    kwargs_stats={'ewm': {'alpha': 0.3}}  # dict | None, kwargs for specific stats
)

TimeSeriesDifferentiator(
    order=1,                 # int, differencing order
    window_size=None         # int | None
)

CalendarFeatures(
    features=None,           # list[str] | None, e.g. ['year', 'month', 'day_of_week', 'hour']
    encoding='cyclical',     # 'cyclical' | 'onehot' | None
    max_values=None          # dict[str, int] | None, max values for cyclical encoding
)
```
