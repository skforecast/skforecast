"""P11: np.quantile vs DataFrame.quantile(axis=1). P12: conformal factors,
np.vectorize vs array lookup, and why a cache keyed by the dict is not safe."""
import _env
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive

rng = np.random.default_rng(0)
print("--- P11: quantiles ---")
q = np.array([0.1, 0.9])
for n_boot in (10, 25, 250, 500, 1001):
    boot = rng.normal(size=(48, n_boot)) * 1e3
    df = pd.DataFrame(boot)
    ref = df.quantile(q=q, axis=1).transpose().to_numpy()
    a = np.quantile(boot, q, axis=1).T
    print(f"n_boot={n_boot:5d} no NaN: np.quantile bitwise equal={np.array_equal(ref, a)}"
          f" max|diff|={np.abs(ref - a).max():.3g}")
for q_ in ([0.05, 0.5, 0.95], [0.0, 1.0], [0.025, 0.975], [1 / 3]):
    boot = rng.normal(size=(200, 250))
    ref = pd.DataFrame(boot).quantile(q=q_, axis=1).transpose().to_numpy()
    print(f"q={q_}: bitwise equal={np.array_equal(ref, np.quantile(boot, q_, axis=1).T)}")
# all the quantiles a user can reach: interval=[a, b], interval=float c -> 0.5 -/+ c/2
qs = np.unique(np.concatenate([np.linspace(0, 1, 201), 0.5 - np.linspace(0, 1, 101) / 2,
                               0.5 + np.linspace(0, 1, 101) / 2, rng.uniform(size=300)]))
for n_boot in (25, 250, 500):
    boot = rng.normal(size=(100, n_boot)) * 50 + 100
    ref = pd.DataFrame(boot).quantile(q=qs, axis=1).transpose().to_numpy()
    a = np.quantile(boot, qs, axis=1).T
    bad = (ref != a)
    print(f"grid of {len(qs)} quantiles, n_boot={n_boot}: cells differing={bad.sum()} of {bad.size}"
          f" | quantiles affected={bad.any(axis=0).sum()} | max rel diff={np.max(np.abs(ref - a) / np.abs(ref)):.2e}"
          f" | e.g. q={qs[bad.any(axis=0)][:4].tolist()}")
boot = rng.normal(size=(6, 250)); boot[1, 3] = np.nan; boot[2, :] = np.nan; boot[4, 100:] = np.nan
ref = pd.DataFrame(boot).quantile(q=q, axis=1).transpose().to_numpy()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    a = np.quantile(boot, q, axis=1).T
    b = np.nanquantile(boot, q, axis=1).T
print("with NaN: DataFrame.quantile rows with NaN ->", ref[[1, 2, 4]].round(3).tolist())
print("          np.quantile                     ->", a[[1, 2, 4]].round(3).tolist(), "equal:", np.array_equal(ref, a, equal_nan=True))
print("          np.nanquantile                  ->", b[[1, 2, 4]].round(3).tolist(), "bitwise equal:", np.array_equal(ref, b, equal_nan=True),
      "| warnings:", sorted({f"{x.category.__name__}: {x.message}" for x in w}))
boot_inf = rng.normal(size=(3, 50)); boot_inf[0, 0] = np.inf
print("with inf: equal:", np.array_equal(pd.DataFrame(boot_inf).quantile(q=q, axis=1).T.to_numpy(), np.quantile(boot_inf, q, axis=1).T, equal_nan=True))

print("--- P11: can bootstrapped predictions hold NaN? (backtesting sets _allow_nan_predictions) ---")
warnings.simplefilter("ignore")
from lightgbm import LGBMRegressor
n = 200
y = pd.Series(rng.normal(size=n).cumsum(), index=pd.date_range("2020", periods=n, freq="D"))
f = ForecasterRecursive(Ridge(), lags=3, binner_kwargs={"n_bins": 4})
f.fit(y, store_in_sample_residuals=True)
lw = y.iloc[-3:].copy(); lw.iloc[1] = np.nan
for allow in (False, True):
    f._allow_nan_predictions = allow
    try:
        p = f.predict_interval(steps=3, last_window=lw, method="bootstrapping", n_boot=20)
        print(f"Ridge allow_nan={allow}:", p.round(3).to_numpy().tolist())
    except Exception as e:
        print(f"Ridge allow_nan={allow}: {type(e).__name__}: {str(e)[:70]}")
f._allow_nan_predictions = False

print("--- P11: reuse of inputs; rng consumption ---")
p1 = f.predict_interval(steps=5, method="bootstrapping", n_boot=30, random_state=7)
p2 = f.predict(steps=5)
b = f.predict_bootstrapping(steps=5, n_boot=30, random_state=7)
print("pred column == predict():", np.array_equal(p1["pred"].to_numpy(), p2.to_numpy()),
      "| bounds == np.quantile(boot):",
      np.array_equal(p1[["lower_bound", "upper_bound"]].to_numpy(), np.quantile(b.to_numpy(), [0.05, 0.95], axis=1).T))
lwv = f.last_window_.to_numpy().copy()
f.predict_bootstrapping(steps=5, n_boot=30)
print("last_window_ untouched by the bootstrapping recursion:", np.array_equal(lwv, f.last_window_.to_numpy()))

print("--- P12: conformal ---")
yp = np.concatenate([rng.uniform(y.min() - 5, y.min() - 1, 250), rng.uniform(y.max() + 1, y.max() + 5, 250)])
f.set_out_sample_residuals(y_true=yp + rng.normal(size=500), y_pred=yp)
d = f.out_sample_residuals_by_bin_
print("out-of-sample dict key order:", list(d), "(not 0..n-1 when a bin was empty)")
cov = 0.8
factor = {k: np.quantile(np.abs(v), cov) for k, v in d.items()}
bins = f.binner.transform(rng.uniform(y.min(), y.max(), 50))
ref = np.vectorize(lambda x: factor[x])(bins)
by_key = np.array([factor[k] for k in range(f.binner.n_bins_)])[bins.astype(int)]
by_pos = np.array(list(factor.values()))[bins.astype(int)]
print("lookup by key  == np.vectorize:", np.array_equal(ref, by_key), "| binner.transform dtype:", bins.dtype)
print("lookup by dict position (naive) == np.vectorize:", np.array_equal(ref, by_pos))
id_before = id(f.out_sample_residuals_by_bin_)
f.set_out_sample_residuals(y_true=yp * 3, y_pred=yp, append=True)
factor2 = {k: np.quantile(np.abs(v), cov) for k, v in f.out_sample_residuals_by_bin_.items()}
print("append=True keeps the same dict object (id equal):", id(f.out_sample_residuals_by_bin_) == id_before,
      "| factors changed:", factor != factor2)
f.in_sample_residuals_by_bin_[0] = f.in_sample_residuals_by_bin_[0] * 10  # users/tests assign these attributes
print("attributes are plain (no property/setter):",
      not isinstance(getattr(type(f), "out_sample_residuals_by_bin_", None), property))
print(_env.versions())
