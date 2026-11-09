"""
BUG-mase-window-size: `repro/PERF-04.py` adapted to run with 0.27.x (argument
`window_size`) and with the fix (column `fold`, no `window_size`), to compare them.
With the fix (2026-11-09): MAE 4.8 s and 58 MB; MASE 2.2 s and 68 MB.

PERF-04: cost of preparing `y_train` for the scaled metrics (MASE, RMSSE) in
the metrics of the multi-series backtesting.

`_calculate_metrics_backtesting_multiseries` stacks the whole series in long
format, builds a `MultiIndex.from_product` of levels and training dates,
filters it with `isin` and drops the first `window_size` observations of each
series with `cumcount`, only to pass `y_train` to the metrics that need it.
The true values come with the predictions (QW-04), so with metrics that do not
use `y_train` none of this runs. It prints the time and the peak memory of the
metrics of a panel with MAE and with MASE: the difference is the preparation
of `y_train`.

Expected (10,000 series of 1,500 observations, 28 test dates, 280,000
predictions; branch feature/backtesting-output-columns, 2026-11-01): MAE about
4.7 s and 50 MB; MASE about 6.3 s and 2.6 GB. The training values that MASE
needs take about 117 MB. Most of the time of MAE is the loop over the series
(one metric call per series, PERF-03), not `y_train`.
"""
import time
import inspect
import skforecast
import tracemalloc
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from skforecast.metrics import add_y_train_argument, mean_absolute_scaled_error
from skforecast.model_selection._utils import (
    _calculate_metrics_backtesting_multiseries,
    _get_y_true_multiseries,
)

rng = np.random.default_rng(0)
n_series, n_obs, test, window_size = 10_000, 1_500, 28, 7
span = pd.date_range("2020-01-01", periods=n_obs, freq="D")
series = {f"s{i}": pd.Series(rng.random(n_obs), index=span) for i in range(n_series)}
index = pd.MultiIndex.from_arrays(
    [span[-test:].repeat(n_series), np.tile(list(series), test)], names=["idx", "level"]
)
predictions = pd.DataFrame(
    {"fold": 0, "y_true": _get_y_true_multiseries(series, index), "y_pred": rng.random(len(index))},
    index=index,
)
train_end = n_obs - test
folds = [[0, [0, train_end], [train_end - window_size, train_end],
          [train_end, n_obs], [train_end, n_obs], False]]

print(
    f"Python {platform.python_version()}, numpy {np.__version__}, "
    f"pandas {pd.__version__}, scikit-learn {sklearn.__version__}"
)
print(f"skforecast from {skforecast.__file__}")
has_window_size = "window_size" in inspect.signature(_calculate_metrics_backtesting_multiseries).parameters
extra = {"window_size": window_size} if has_window_size else {}
print(f"{n_series:,} series x {n_obs:,} observations, {len(index):,} predictions")
print(f"Training values needed by MASE: {n_series * (train_end - window_size) * 8 / 1e6:.0f} MB")
for name, metric in [("MAE", mean_absolute_error), ("MASE", mean_absolute_scaled_error)]:
    tracemalloc.start()
    t0 = time.perf_counter()
    _calculate_metrics_backtesting_multiseries(
        series=series, predictions=predictions, folds=folds, span_index=span,
        metrics=[add_y_train_argument(metric)], **extra,
        levels=list(series), add_aggregated_metric=True,
    )
    elapsed = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"  {name:<5}: {elapsed:6.2f} s, peak {peak / 1e6:8.1f} MB")
