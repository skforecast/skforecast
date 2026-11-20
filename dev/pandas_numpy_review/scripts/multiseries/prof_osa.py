"""One-step-ahead search pieces of ForecasterRecursiveMultiSeries.
Usage: python prof_osa.py N EXOG(none|dictcat) EST(ridge|lgbm)"""
import sys, time, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import skforecast.model_selection._utils as U
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import OneStepAheadFold

N, EXOG, EST = int(sys.argv[1]), sys.argv[2], sys.argv[3]
LEN = 1000
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=LEN, freq="D")
series = {f"s{i:04d}": pd.Series(rng.normal(size=LEN).cumsum() + 50, index=idx, name=f"s{i:04d}") for i in range(N)}
exog = None
if EXOG == "dictcat":
    cats = ["alpha", "beta", "gamma", "delta"]
    exog = {k: pd.DataFrame({"e1": rng.normal(size=LEN), "e2": rng.normal(size=LEN),
                             "c1": pd.Categorical(rng.choice(cats, size=LEN), categories=cats)}, index=idx)
            for k in series}
est = Ridge() if EST == "ridge" else LGBMRegressor(verbose=-1, n_jobs=1, n_estimators=50)
fc = ForecasterRecursiveMultiSeries(est, lags=24)
cv = OneStepAheadFold(initial_train_size=800)
levels = list(series)

T = defaultdict(float)
def wrap(name):
    f = getattr(U, name)
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[name] += time.perf_counter() - t0
    setattr(U, name, w)
for name in ["_copy_rows_to_check", "_check_in_place_fit", "_calculate_metric_values",
             "cast_catboost_categorical_columns_dataframe", "_get_y_train_and_scales"]:
    wrap(name)

for rep in range(3):
    T.clear()
    t0 = time.perf_counter()
    split = fc._train_test_split_one_step_ahead(series=series, initial_train_size=800, exog=exog)
    t_split = time.perf_counter() - t0
    X_train, y_train, X_test, y_test, Xtr_enc, Xte_enc, sw, fk = split
    e = fc.estimator
    f_fit, f_pred = type(e).fit, type(e).predict
    def fit_w(*a, **k):
        t1 = time.perf_counter(); r = f_fit(e, *a[0:], **k) if False else f_fit(*a, **k); T["estimator.fit"] += time.perf_counter() - t1; return r
    def pred_w(*a, **k):
        t1 = time.perf_counter(); r = f_pred(*a, **k); T["estimator.predict"] += time.perf_counter() - t1; return r
    e.fit = lambda *a, **k: fit_w(e, *a, **k)
    e.predict = lambda *a, **k: pred_w(e, *a, **k)
    t0 = time.perf_counter()
    U._predict_and_calculate_metrics_one_step_ahead_multiseries(
        forecaster=fc, series=series, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        X_train_encoding=Xtr_enc, X_test_encoding=Xte_enc, levels=levels, metrics=["mean_absolute_error"],
        add_aggregated_metric=True, sample_weight=sw, fit_kwargs=fk, return_predictions=False)
    t_cand = time.perf_counter() - t0
    del e.fit, e.predict
    print(f"rep{rep}: split (once per lags) {t_split*1e3:.1f} ms; per candidate {t_cand*1e3:.1f} ms -> "
          + ", ".join(f"{k} {v*1e3:.1f}" for k, v in sorted(T.items(), key=lambda kv: -kv[1])))
other = t_cand - sum(T.values())
print(f"per-candidate rest (pandas groupby/loc/DataFrame/aggregation): {other*1e3:.1f} ms = {100*other/t_cand:.0f} %")
print("X_train", X_train.shape, "X_test", X_test.shape)
# cost pieces of the rest
pred = fc.estimator.predict(X_test)
def grp():
    g = pd.DataFrame({"y_true": y_test, "y_pred": pred, "_level_skforecast": Xte_enc}, index=y_test.index).groupby("_level_skforecast")
    d = {k: v for k, v in g}
    return d
import timeit
print("groupby+dict of groups:", round(min(timeit.repeat(grp, number=3, repeat=3)) / 3 * 1e3, 1), "ms")
d = grp()
print("per-level .loc y_true/y_pred:", round(min(timeit.repeat(lambda: [(v.loc[:, 'y_true'], v.loc[:, 'y_pred']) for v in d.values()], number=3, repeat=3)) / 3 * 1e3, 1), "ms")
codes = np.asarray(Xte_enc)
def numpy_alt():
    o = np.argsort(pd.factorize(codes)[0], kind="stable")
    return o
print("alt numpy argsort split:", round(min(timeit.repeat(numpy_alt, number=3, repeat=3)) / 3 * 1e3, 2), "ms")
print("X_train_encoding map (split):", round(min(timeit.repeat(lambda: X_train["_level_skforecast"].map({v: k for k, v in fc.encoding_mapping_.items()}).to_numpy(), number=3, repeat=3)) / 3 * 1e3, 2), "ms")
