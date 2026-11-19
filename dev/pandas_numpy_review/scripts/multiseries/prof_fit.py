"""Stage timing of ForecasterRecursiveMultiSeries.fit (read-only study).

Usage: python prof_fit.py N_SERIES EXOG(none|wide|dictcat) EST(ridge|lgbm) [REPEATS]
"""
import os
import sys
import time
import warnings
import cProfile
import pstats
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

import skforecast.recursive._forecaster_recursive_multiseries as M
from skforecast.recursive import ForecasterRecursiveMultiSeries

N = int(sys.argv[1])
EXOG = sys.argv[2]
EST = sys.argv[3]
REP = int(sys.argv[4]) if len(sys.argv) > 4 else 3
PROFILE = os.environ.get("PROFILE", "0") == "1"

LEN = 1000
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=LEN, freq="D")
series = {
    f"s{i:04d}": pd.Series(
        rng.normal(size=LEN).cumsum() + 50, index=idx, name=f"s{i:04d}"
    )
    for i in range(N)
}
if EXOG == "none":
    exog = None
elif EXOG == "wide":
    exog = pd.DataFrame(rng.normal(size=(LEN, 3)), index=idx, columns=["e1", "e2", "e3"])
elif EXOG == "dictcat":
    cats = ["alpha", "beta", "gamma", "delta"]
    exog = {}
    for k in series:
        exog[k] = pd.DataFrame(
            {
                "e1": rng.normal(size=LEN),
                "e2": rng.normal(size=LEN),
                "c1": pd.Categorical(rng.choice(cats, size=LEN), categories=cats),
            },
            index=idx,
        )
else:
    raise ValueError(EXOG)


def make_est():
    if EST == "ridge":
        return Ridge()
    return LGBMRegressor(verbose=-1, n_jobs=1, random_state=0)


# ---------------------------------------------------------------- patches
T = defaultdict(float)


def wrap_func(module, name, label):
    f = getattr(module, name)

    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[label] += time.perf_counter() - t0

    setattr(module, name, w)


for name in [
    "check_preprocess_series",
    "check_preprocess_exog_multiseries",
    "align_series_and_exog_multiseries",
    "transform_dataframe",
    "get_exog_dtypes",
    "_filter_nan_train_X_y",
    "configure_estimator_categorical_features",
    "cast_catboost_categorical_columns_dataframe",
    "_copy_rows_to_check",
    "_check_in_place_fit",
    "initialize_transformer_series",
]:
    wrap_func(M, name, name)

orig_concat = M.compat.concat


class _C:
    pass


compat_proxy = _C()
compat_proxy.__dict__.update(M.compat.__dict__)


def concat_w(*a, **k):
    t0 = time.perf_counter()
    try:
        return orig_concat(*a, **k)
    finally:
        T["compat.concat"] += time.perf_counter() - t0


compat_proxy.concat = concat_w
M.compat = compat_proxy

Cls = M.ForecasterRecursiveMultiSeries
for meth, label in [
    ("_create_train_X_y", "_create_train_X_y [total]"),
    ("_assemble_train_X_y", "_assemble_train_X_y [total]"),
    ("_create_train_X_y_single_series", "single_series (per series)"),
    ("create_sample_weights", "create_sample_weights"),
    ("_get_level_row_slices", "_get_level_row_slices"),
    ("_binning_in_sample_residuals", "_binning_in_sample_residuals"),
    ("fit", "fit [total]"),
]:
    f = getattr(Cls, meth)

    def mk(f, label):
        def w(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return f(self, *a, **k)
            finally:
                T[label] += time.perf_counter() - t0
        w.__wrapped__ = f
        return w

    setattr(Cls, meth, mk(f, label))


def instrument_estimator(fc):
    est = fc.estimator
    f_fit = est.fit
    f_pred = est.predict

    def fit_w(*a, **k):
        t0 = time.perf_counter()
        try:
            return f_fit(*a, **k)
        finally:
            T["estimator.fit"] += time.perf_counter() - t0

    def pred_w(*a, **k):
        t0 = time.perf_counter()
        try:
            return f_pred(*a, **k)
        finally:
            T["estimator.predict (in-sample res.)"] += time.perf_counter() - t0

    est.fit = fit_w
    est.predict = pred_w

    enc = fc.categorical_encoder
    f_ft = enc.fit_transform

    def ft_w(*a, **k):
        t0 = time.perf_counter()
        try:
            return f_ft(*a, **k)
        finally:
            T["categorical_encoder.fit_transform"] += time.perf_counter() - t0

    enc.fit_transform = ft_w


def one_run():
    fc = ForecasterRecursiveMultiSeries(estimator=make_est(), lags=24)
    instrument_estimator(fc)
    T.clear()
    fc.fit(series=series, exog=exog, suppress_warnings=True)
    return dict(T), fc


runs = [one_run() for _ in range(REP)]
best = min(runs, key=lambda r: r[0]["fit [total]"])
Tb, fc = best
tot = Tb["fit [total]"]
print(f"## N={N} exog={EXOG} est={EST}  fit total min of {REP}: {tot*1e3:.1f} ms  "
      f"(all: {[round(r[0]['fit [total]']*1e3,1) for r in runs]})")
for k, v in sorted(Tb.items(), key=lambda kv: -kv[1]):
    print(f"   {k:45s} {v*1e3:9.2f} ms  {100*v/tot:5.1f} %")

# ---------------------------------------------------- X_train facts
out = fc._create_train_X_y.__wrapped__(fc, series=series, exog=exog, store_last_window=False)
X_train, y_train = out[0], out[1]
print("   X_train shape", X_train.shape, "n_blocks", len(X_train._mgr.blocks),
      "dtypes", dict(X_train.dtypes.value_counts()))
print("   X_train flags F", X_train._mgr.blocks[0].values.flags.f_contiguous,
      "C", X_train._mgr.blocks[0].values.flags.c_contiguous)
print("   categorical_features_names_in_", fc.categorical_features_names_in_)

# --------------------------------- estimator with DataFrame vs numpy
est_df = make_est()
est_np = make_est()
fk = {}
if fc.categorical_features_names_in_ and EST == "lgbm":
    fk = {"categorical_feature": [fc.X_train_features_names_out_.index(c)
                                  for c in fc.categorical_features_names_in_]}
Xn = X_train.to_numpy()
yn = y_train.to_numpy()
t_conv = []
for _ in range(3):
    t0 = time.perf_counter(); X_train.to_numpy(); t_conv.append(time.perf_counter() - t0)
tdf, tnp = [], []
for _ in range(REP):
    t0 = time.perf_counter(); est_df.fit(X_train, y_train, **fk); tdf.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); est_np.fit(Xn, yn, **fk); tnp.append(time.perf_counter() - t0)
pdf, pnp = [], []
for _ in range(REP):
    t0 = time.perf_counter(); est_df.predict(X_train); pdf.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); est_np.predict(Xn); pnp.append(time.perf_counter() - t0)
print(f"   X_train.to_numpy(): {min(t_conv)*1e3:.2f} ms")
print(f"   est.fit(DataFrame) {min(tdf)*1e3:.1f} ms vs fit(numpy) {min(tnp)*1e3:.1f} ms; "
      f"predict(DataFrame) {min(pdf)*1e3:.1f} vs predict(numpy) {min(pnp)*1e3:.1f} ms")
same = np.allclose(est_df.predict(X_train), est_np.predict(Xn), rtol=0, atol=0)
print("   identical predictions DF vs numpy:", same)

# ------------------------------------------------ cProfile by library
if PROFILE:
    fc2 = ForecasterRecursiveMultiSeries(estimator=make_est(), lags=24)
    pr = cProfile.Profile()
    pr.enable()
    fc2.fit(series=series, exog=exog, suppress_warnings=True)
    pr.disable()
    st = pstats.Stats(pr)
    agg = defaultdict(float)
    total = 0.0
    for (fn, ln, func), (cc, nc, tt, ct, callers) in st.stats.items():
        total += tt
        if "/pandas/" in fn:
            lib = "pandas"
        elif "/numpy/" in fn:
            lib = "numpy"
        elif "/sklearn/" in fn:
            lib = "sklearn"
        elif "/lightgbm/" in fn:
            lib = "lightgbm"
        elif "/skforecast/" in fn:
            lib = "skforecast"
        elif fn == "~":
            # builtins: attribute by name
            if "pandas" in func:
                lib = "pandas(C)"
            elif "numpy" in func or "ndarray" in func or "ufunc" in func:
                lib = "numpy(C)"
            elif "_ctypes" in func or "CFuncPtr" in func:
                lib = "lightgbm(C via ctypes)"
            else:
                lib = "builtins/other(C)"
        else:
            lib = "other-py"
        agg[lib] += tt
    print(f"   cProfile tottime by library (total {total*1e3:.0f} ms, inflated):")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"      {k:28s} {v*1e3:8.1f} ms {100*v/total:5.1f} %")
    # cumulative of interesting pandas entry points
    print("   top pandas-heavy callers (cumtime) inside skforecast:")
    rows = []
    for (fn, ln, func), (cc, nc, tt, ct, callers) in st.stats.items():
        if "/skforecast/" in fn:
            rows.append((ct, f"{os.path.basename(fn)}:{ln} {func}"))
    for ct, name in sorted(rows, reverse=True)[:18]:
        print(f"      {ct*1e3:8.1f} ms  {name}")
