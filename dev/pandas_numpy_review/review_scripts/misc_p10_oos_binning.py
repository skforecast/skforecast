"""P10: out-of-sample residual binning, pandas (current) vs numpy prototype.
The tail of `set_out_sample_residuals` (rng subsampling, empty bins) is replicated
to see whether the final stored residuals are the same."""
import _env
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive

warnings.simplefilter("ignore")


def bin_pandas(y_true, y_pred, binner):
    data = pd.DataFrame({"prediction": y_pred, "residuals": y_true - y_pred}).dropna()
    y_pred = data["prediction"].to_numpy()
    residuals = data["residuals"].to_numpy()
    data["bin"] = binner.transform(y_pred).astype(int)
    return residuals, data.groupby("bin")["residuals"].apply(np.array).to_dict()


def bin_numpy(y_true, y_pred, binner, int_keys=True):
    residuals = y_true - y_pred
    keep = ~(np.isnan(y_pred) | np.isnan(residuals))
    y_pred, residuals = y_pred[keep], residuals[keep]
    bins = binner.transform(y_pred).astype(int)
    order = np.argsort(bins, kind="stable")
    keys, start = np.unique(bins[order], return_index=True)
    parts = np.split(residuals[order], start[1:]) if len(keys) else []
    keys = keys.tolist() if int_keys else list(keys)
    return residuals, dict(zip(keys, parts))


def tail(forecaster, residuals, residuals_by_bin, random_state=123):
    """Lines 3167-3207 of _forecaster_recursive.py (append=False)."""
    by_bin = dict(residuals_by_bin)
    max_samples = 10_000 // forecaster.binner.n_bins_
    rng = np.random.default_rng(seed=random_state)
    for k, v in by_bin.items():
        if len(v) > max_samples:
            by_bin[k] = rng.choice(a=v, size=max_samples, replace=False)
    for k in forecaster.binner_intervals_.keys():
        if k not in by_bin:
            by_bin[k] = np.array([])
    empty = [k for k, v in by_bin.items() if v.size == 0]
    for k in empty:
        by_bin[k] = rng.choice(a=residuals, size=min(max_samples, len(residuals)), replace=False)
    return by_bin


def dict_equal(a, b, check_key_order=True):
    if check_key_order and list(a) != list(b):
        return False
    return set(a) == set(b) and all(
        a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k], equal_nan=True) for k in a)


rng = np.random.default_rng(1)
n = 300
y = pd.Series(rng.normal(size=n).cumsum(), index=pd.date_range("2020", periods=n, freq="D"))
f = ForecasterRecursive(Ridge(), lags=3, binner_kwargs={"n_bins": 5})
f.fit(y, store_in_sample_residuals=True)
print("in-sample key types :", {type(k).__name__ for k in f.in_sample_residuals_by_bin_},
      "| binner_intervals_ key types:", {type(k).__name__ for k in f.binner_intervals_})

lo, hi = y.min(), y.max()
m = 40_000
scen = {}
yp = rng.uniform(lo, hi, m); yt = yp + rng.normal(size=m)
scen["many residuals (subsampling in every bin)"] = (yt, yp)
scen["few residuals"] = (yt[:50], yp[:50])
yp2 = rng.uniform(lo, lo + (hi - lo) * 0.3, 3000)
scen["empty bins (predictions only in low range)"] = (yp2 + rng.normal(size=3000), yp2)
yt3, yp3 = yt[:5000].copy(), yp[:5000].copy(); yt3[::7] = np.nan; yp3[::11] = np.nan
scen["NaN in y_true and in y_pred"] = (yt3, yp3)
scen["inf in y_pred"] = (np.append(yt[:20], 1.0), np.append(yp[:20], np.inf))
scen["all NaN"] = (np.full(10, np.nan), yp[:10])
scen["one residual"] = (yt[:1], yp[:1])

for name, (yt_, yp_) in scen.items():
    try:
        r_pd, d_pd = bin_pandas(yt_, yp_, f.binner)
    except Exception as e:
        r_pd, d_pd = None, f"{type(e).__name__}: {str(e)[:60]}"
    try:
        r_np, d_np = bin_numpy(yt_, yp_, f.binner)
    except Exception as e:
        r_np, d_np = None, f"{type(e).__name__}: {str(e)[:60]}"
    print(f"## {name}")
    if isinstance(d_pd, str) or isinstance(d_np, str):
        print("   pandas:", d_pd if isinstance(d_pd, str) else "ok", "| numpy:", d_np if isinstance(d_np, str) else "ok")
        continue
    print("   binned dict equal (values, dtype, order in bin, key order):", dict_equal(d_pd, d_np),
          "| keys:", list(d_pd), "| key types pandas/numpy:",
          {type(k).__name__ for k in d_pd}, {type(k).__name__ for k in d_np})
    if len(r_pd) == 0:
        continue
    fin_pd, fin_np = tail(f, r_pd, d_pd), tail(f, r_np, d_np)
    print("   final stored dict equal (after rng):", dict_equal(fin_pd, fin_np), "| final key order:", list(fin_pd))
    # faithful replica? compare with the real method
    f.out_sample_residuals_ = None; f.out_sample_residuals_by_bin_ = None
    try:
        f.set_out_sample_residuals(y_true=yt_, y_pred=yp_)
        print("   replica == real set_out_sample_residuals:", dict_equal(fin_pd, f.out_sample_residuals_by_bin_),
              "| real key types:", {type(k).__name__ for k in f.out_sample_residuals_by_bin_})
    except Exception as e:
        print("   real method:", type(e).__name__, str(e)[:80])
    # does order inside a bin matter? shuffle within bins before the rng step
    shuf = {k: np.sort(v) for k, v in d_np.items()}
    print("   same final dict if the order inside each bin changes (sorted):", dict_equal(fin_pd, tail(f, r_np, shuf)))

print("## consumers of the key type")
d_i64 = {np.int64(0): np.array([1.0])}; d_int = {0: np.array([1.0])}
print("   lookup d_i64[0]:", d_i64[0], "| d_int[np.int64(0)]:", d_int[np.int64(0)],
      "| keys ==:", list(d_i64) == list(d_int), "| repr keys:", repr(list(d_i64)), "vs", repr(list(d_int)))
print("   pickle bytes equal:", pickle.dumps(d_i64) == pickle.dumps(d_int))
import json
for d in (d_i64, d_int):
    try:
        json.dumps({k: v.tolist() for k, v in d.items()}); print("   json.dumps ok for", type(next(iter(d))).__name__)
    except TypeError as e:
        print("   json.dumps fails for", type(next(iter(d))).__name__, ":", e)
print(_env.versions())
