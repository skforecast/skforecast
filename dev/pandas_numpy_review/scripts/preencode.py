"""Cost of converting the exog dict of a whole backtesting to one float array, once."""
import sys, time, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
n_series = int(sys.argv[1]); n_rows = 1500
rng = np.random.default_rng(1); idx = pd.date_range("2020", periods=n_rows, freq="h")
names = [f"series_{i:04d}" for i in range(n_series)]
DOW = np.array(["mon","tue","wed","thu","fri","sat","sun"])
series = {k: pd.Series(rng.normal(size=n_rows), index=idx) for k in names}
exog = {k: pd.DataFrame({"temp": rng.normal(size=n_rows), "promo": rng.normal(size=n_rows),
        "hour": idx.hour.to_numpy().astype("int64"),
        "dow": pd.Categorical(DOW[idx.dayofweek.to_numpy()], categories=list(DOW)),
        "shop_type": np.where(rng.random(n_rows) < .5, "big", "small")}, index=idx) for k in names}
f = ForecasterRecursiveMultiSeries(Ridge(), lags=24); f.fit(series={k: v.iloc[:500] for k, v in series.items()},
                                                         exog={k: v.iloc[:500] for k, v in exog.items()})
cat = f.categorical_features_names_in_; enc = f.categorical_encoder
def per_series():
    out = np.empty((n_rows, n_series, 5))
    for j, k in enumerate(names):
        df = exog[k].copy(deep=False); df[cat] = enc.transform(df[cat]); out[:, j] = df.to_numpy(dtype=float)
    return out
def concat_once():
    big = pd.concat([exog[k] for k in names], axis=0, ignore_index=True)
    big[cat] = enc.transform(big[cat])
    return big.to_numpy(dtype=float).reshape(n_series, n_rows, 5).transpose(1, 0, 2)
def codes_lookup():
    # per column, numpy only after pandas codes; shared CategoricalDtype -> one lookup
    out = np.empty((n_rows, n_series, 5))
    num = ["temp", "promo", "hour"]
    lookups = {}
    for j, k in enumerate(names):
        df = exog[k]
        out[:, j, :3] = df[num].to_numpy(dtype=float)
        for c_i, c in enumerate(cat):
            col = df[c]
            if isinstance(col.dtype, pd.CategoricalDtype):
                key = (c, col.dtype)
                if key not in lookups:
                    lk = pd.Index(enc.categories_[c_i]).get_indexer(col.cat.categories).astype(float)
                    lk[lk < 0] = np.nan; lookups[key] = np.append(lk, np.nan)
                out[:, j, 3 + c_i] = lookups[key][col.cat.codes.to_numpy()]
            else:
                lk = pd.Index(enc.categories_[c_i]).get_indexer(col.to_numpy()).astype(float)
                lk[lk < 0] = np.nan; out[:, j, 3 + c_i] = lk
    return out
res = {}
for name, fn in [("per_series_encoder", per_series), ("concat_once_encoder", concat_once), ("codes_lookup", codes_lookup)]:
    ts = []
    for _ in range(3):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t)
    res[name] = r
    print(f"pandas {pd.__version__} {n_series} series x {n_rows} rows: {name:22s} {min(ts)*1000:8.1f} ms")
a = res["per_series_encoder"]
print("  identical concat:", np.array_equal(a, res["concat_once_encoder"], equal_nan=True),
      " identical codes:", np.array_equal(a, res["codes_lookup"], equal_nan=True))
