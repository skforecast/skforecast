"""
Ceiling of a numpy-only backtesting loop for ForecasterRecursiveMultiSeries
(refit=False): exog pre-encoded once, folds sliced by position, prediction by
`_recursive_predict` directly. Compares with backtesting_forecaster_multiseries.
Usage: python ceiling_bt_ms.py N_SERIES EXOG(float|mixed|none) EST(ridge|lgbm) [N_FOLDS] [STEPS]
"""
import os, sys, time, warnings
if os.environ.get("NO_PYARROW") == "1":
    class _B:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")
    sys.meta_path.insert(0, _B)
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries

n_series = int(sys.argv[1]); exog_kind = sys.argv[2]; est = sys.argv[3]
n_folds = int(sys.argv[4]) if len(sys.argv) > 4 else 50
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 24
n_rows = 1500; lags = 24
rng = np.random.default_rng(1)
idx = pd.date_range("2020-01-01", periods=n_rows, freq="h")
names = [f"series_{i:04d}" for i in range(n_series)]
values = np.cumsum(rng.normal(size=(n_rows, n_series)), axis=0) + 100
series = {k: pd.Series(values[:, j], index=idx, name=k) for j, k in enumerate(names)}
DOW = np.array(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
def make_exog(j):
    df = pd.DataFrame({"temp": rng.normal(size=n_rows), "promo": rng.normal(size=n_rows)}, index=idx)
    if exog_kind == "mixed":
        df["hour"] = idx.hour.to_numpy().astype("int64")
        df["dow"] = pd.Categorical(DOW[idx.dayofweek.to_numpy()], categories=list(DOW))
        df["shop_type"] = np.where(rng.random(n_rows) < 0.5, "big", "small")  # str column
    return df
exog = None if exog_kind == "none" else {k: make_exog(j) for j, k in enumerate(names)}
estimator = Ridge() if est == "ridge" else LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1, random_state=1)
f = ForecasterRecursiveMultiSeries(estimator, lags=lags, encoding="ordinal")
initial = n_rows - n_folds * steps
cv = TimeSeriesFold(steps=steps, initial_train_size=initial, refit=False)

# Current backtesting (includes initial fit). Fit time measured apart.
t = time.perf_counter()
f_fit = ForecasterRecursiveMultiSeries(estimator, lags=lags, encoding="ordinal")
f_fit.fit(series={k: v.iloc[:initial] for k, v in series.items()},
          exog=None if exog is None else {k: v.iloc[:initial] for k, v in exog.items()})
t_fit = time.perf_counter() - t
best = []
for r in range(3):
    t = time.perf_counter()
    metric, preds = backtesting_forecaster_multiseries(
        forecaster=f, series=series, exog=exog, cv=cv, metric="mean_absolute_error",
        show_progress=False, n_jobs=1, suppress_warnings=True)
    best.append(time.perf_counter() - t)
t_bt = min(best)

# Ceiling: pre-encode exog once (numpy 3D), folds by position.
f2 = f_fit
t0 = time.perf_counter()
if exog is not None:
    cat = f2.categorical_features_names_in_ or []
    ex3 = np.empty((n_rows, n_series, len(f2.X_train_exog_names_out_)))
    for j, k in enumerate(names):
        df = exog[k][f2.exog_names_in_]
        if cat:
            df = df.copy(deep=False)
            df[cat] = f2.categorical_encoder.transform(df[cat])
        ex3[:, j, :] = df.to_numpy(dtype=float)
t_pre = time.perf_counter() - t0
def run_ceiling():
    out = np.empty((n_folds * steps, n_series))
    for i in range(n_folds):
        t_start = initial + i * steps
        lw = pd.DataFrame(values[t_start - lags:t_start], columns=names, copy=False)
        ed = None if exog is None else {s + 1: ex3[t_start + s] for s in range(steps)}
        p = f2._recursive_predict(steps=steps, levels=names, last_window=lw, exog_values_dict=ed)
        out[i * steps:(i + 1) * steps] = p
    y_true = values[initial:]
    mae = np.mean(np.abs(y_true - out), axis=0)
    return out, mae
bc = []
for r in range(3):
    t = time.perf_counter(); out, mae = run_ceiling(); bc.append(time.perf_counter() - t)
t_ceil = min(bc)
# model-only time inside the ceiling: predict_fn calls
from skforecast.utils.utils import _build_predict_function
pf = _build_predict_function(f2.estimator, f2)
nfeat = len(f2.X_train_features_names_out_)
Xdummy = np.asfortranarray(np.zeros((n_series, nfeat)))
t = time.perf_counter()
for _ in range(n_folds * steps): pf(Xdummy)
t_model = time.perf_counter() - t

# identity check vs current backtesting (wide pivot of predictions)
p = preds.copy()
if "fold" in p.columns: p = p.drop(columns="fold")
wide = p.reset_index().pivot(index=p.index.name or "index", columns="level", values="pred")[names].to_numpy()
ident = np.array_equal(wide, out)
print(f"pandas={pd.__version__} pyarrow={'pyarrow' in sys.modules} n_series={n_series} exog={exog_kind} est={est} folds={n_folds} steps={steps}")
print(f"  initial fit            {t_fit*1000:9.1f} ms")
print(f"  backtesting (current)  {t_bt*1000:9.1f} ms  -> sin fit {(t_bt-t_fit)*1000:9.1f} ms, {(t_bt-t_fit)/n_folds*1000:7.2f} ms/fold")
print(f"  ceiling numpy loop     {t_ceil*1000:9.1f} ms  ({t_ceil/n_folds*1000:7.2f} ms/fold) + pre-encode once {t_pre*1000:.1f} ms")
print(f"  model only (predict_fn){t_model*1000:9.1f} ms")
print(f"  identical predictions: {ident}")
