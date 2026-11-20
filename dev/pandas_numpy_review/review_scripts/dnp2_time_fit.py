"""
D-NP2: time and peak memory of `ForecasterRecursiveMultiSeries.fit`, one case
per process. The tree is chosen with PYTHONPATH (the current code or the
prototype). Cases: `bdg2` (the 1578 buildings of py59, see `real_data.py`) and
synthetic ones with 600 series of 500 days and 20 exog columns: `float` (one
float block), `ordinal_category`, `int10` (10 integer exog columns), `text2`
('auto' with two text columns), `none_cat2` (None with two integer `category`
columns). Best of REPS fits after a warm-up one.

Usage: python dnp2_time_fit.py CASE ESTIMATOR [REPS]   (see dnp2_run_perf.sh)
Run it alone on the machine (OMP_NUM_THREADS=1).
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import resource
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
case, est_name = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3


def estimator():
    if est_name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(verbose=-1, n_estimators=50, random_state=1, n_jobs=1)
    if est_name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=50, random_state=1, n_jobs=1)
    if est_name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=50, random_state=1)
    if est_name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=50, verbose=0, random_state=1, thread_count=1, allow_writing_files=False)
    from sklearn.linear_model import Ridge
    return Ridge()


from skforecast.recursive import ForecasterRecursiveMultiSeries  # noqa: E402

if case == "bdg2":
    from real_data import _bdg2_data
    from skforecast.preprocessing import RollingFeatures
    series, exog = _bdg2_data()
    series = {k: v.loc[:"2017-10-31 23:59:00"] for k, v in series.items()}
    exog = {k: v.loc[:"2017-10-31 23:59:00"] for k, v in exog.items()}
    make = lambda: ForecasterRecursiveMultiSeries(  # noqa: E731
        estimator(), lags=14, window_features=RollingFeatures(stats=["mean", "min", "max"], window_sizes=7),
        categorical_features="auto", encoding="ordinal")
else:
    # Synthetic: 600 series x 500 days, 20 exog columns.
    rng = np.random.default_rng(0)
    n, n_series = 500, 600
    idx = pd.date_range("2020", periods=n, freq="D")
    series = pd.DataFrame(rng.normal(size=(n, n_series)), index=idx,
                          columns=[f"s{i}" for i in range(n_series)])
    exog = pd.DataFrame(rng.normal(size=(n, 20)), index=idx, columns=[f"x{i}" for i in range(20)])
    encoding, cf = "ordinal", "auto"
    if case == "float":            # one float block
        pass
    elif case == "ordinal_category":
        encoding = "ordinal_category"
    elif case == "int10":          # 10 of the 20 exog columns are int64
        for i in range(10):
            exog[f"x{i}"] = rng.integers(0, 100, n)
    elif case == "text2":          # 'auto' with two text columns (encoded by skforecast)
        for i in range(2):
            exog[f"x{i}"] = np.array(list("abcdefgh"))[rng.integers(0, 8, n)]
    elif case == "none_cat2":      # None with two integer `category` columns
        cf = None
        for i in range(2):
            exog[f"x{i}"] = pd.Categorical(rng.integers(0, 8, n))
    else:
        raise ValueError(case)
    make = lambda: ForecasterRecursiveMultiSeries(  # noqa: E731
        estimator(), lags=7, encoding=encoding, categorical_features=cf)

T = {"est_fit": 0.0, "est_predict": 0.0}
est_cls = type(estimator())
_fit, _predict = est_cls.fit, est_cls.predict
SEEN = {}


def fit(self, X, y, *a, **k):
    SEEN["X"] = (type(X).__name__ + " " + (str(X.dtype) + (" F" if X.flags.f_contiguous and not X.flags.c_contiguous else " C")
                 + ("" if X.flags.writeable else " read-only") if isinstance(X, np.ndarray)
                 else ",".join(sorted({str(t) for t in X.dtypes}))) + f" {X.shape}")
    t = time.perf_counter()
    try:
        return _fit(self, X, y, *a, **k)
    finally:
        T["est_fit"] += time.perf_counter() - t


def predict(self, X, *a, **k):
    t = time.perf_counter()
    try:
        return _predict(self, X, *a, **k)
    finally:
        T["est_predict"] += time.perf_counter() - t


est_cls.fit, est_cls.predict = fit, predict
rows = []
for r in range(reps + 1):
    f = make()
    T["est_fit"] = T["est_predict"] = 0.0
    t = time.perf_counter()
    f.fit(series=series, exog=exog, suppress_warnings=True)
    total = time.perf_counter() - t
    if r:
        rows.append((total, T["est_fit"], T["est_predict"], total - T["est_fit"] - T["est_predict"]))
    del f
best = min(rows)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
tree = _env.tree()
print(f"{case:17}{est_name:9}{tree:6} fit {best[0]:7.2f} s (estimator.fit {best[1]:7.2f}, in-sample predict "
      f"{best[2]:6.2f}, rest {best[3]:6.2f}); peak {peak:7.0f} MB; estimator gets {SEEN['X']}")
