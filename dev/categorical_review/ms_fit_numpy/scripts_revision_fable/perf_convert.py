"""Cost of the conversion of the training matrix to the array of the estimator.

Branch only. Times `_train_matrix_to_estimator_input` on the matrix that
`_create_train_X_y` builds, for several encodings and exog dtypes, and tells
whether the result is a view (no copy) or a new array.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import warnings
import numpy as np
import pandas as pd
import skforecast
print(skforecast.__file__)
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils import _train_matrix_to_estimator_input
import common as C
C.pin_core(12)
warnings.simplefilter("ignore")

n_series, n_obs = 300, 1000
rng = np.random.default_rng(0)
idx = pd.date_range("2018-01-01", periods=n_obs, freq="D")
series = pd.DataFrame(rng.normal(size=(n_obs, n_series)).cumsum(axis=0), index=idx,
                      columns=[f"s{i}" for i in range(n_series)])


def exog_for(kind):
    exog = pd.DataFrame(rng.normal(size=(n_obs, 10)), index=idx, columns=[f"x{i}" for i in range(10)])
    if kind == "one int":
        exog["x9"] = rng.integers(0, 7, size=n_obs)
    if kind == "one bool":
        exog["x9"] = rng.integers(0, 2, size=n_obs).astype(bool)
    if kind == "one category (auto)":
        exog["x9"] = pd.Categorical(rng.integers(0, 12, size=n_obs))
    return exog


print(f"python {sys.version.split()[0]} numpy {np.__version__} pandas {pd.__version__}")
for enc in ["ordinal", "ordinal_category", "onehot"]:
    for kind in ["all float", "one int", "one bool", "one category (auto)"]:
        if enc == "onehot" and kind != "all float":
            continue
        exog = exog_for(kind)
        f = ForecasterRecursiveMultiSeries(Ridge(), lags=24, encoding=enc)
        t = time.perf_counter()
        X_train = f._create_train_X_y(series=series, exog=exog, store_last_window=False)[0]
        t_create = time.perf_counter() - t
        times = []
        for _ in range(7):
            t = time.perf_counter()
            X = _train_matrix_to_estimator_input(X_train=X_train, estimator=f.estimator, cat_indices=[])
            times.append(time.perf_counter() - t)
        n_blocks = len({str(d) for d in X_train.dtypes})
        is_view = bool(np.shares_memory(X, X_train.iloc[:, 0].to_numpy()))
        print(f"RESULT enc={enc:17} exog={kind:20} shape={X.shape} create_train_X_y {t_create * 1000:7.1f} ms | "
              f"conversion min {min(times) * 1000:7.2f} ms med {np.median(times) * 1000:7.2f} ms | "
              f"view of X_train (no copy): {is_view} | new array {0 if is_view else X.nbytes / 1024**2:6.1f} MB | order={'F' if X.flags.f_contiguous else 'C'}")
