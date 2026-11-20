"""P16: DMV last window, current per-column extraction vs one `to_numpy(dtype=float)`.
Compares the arrays that reach `transform_numpy` and the public result when possible."""
import _env
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.preprocessing import RollingFeatures

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
n, ws = 60, 5
idx = pd.date_range("2020-01-01", periods=n, freq="D")
names = ["a", "b", "c"]
series = pd.DataFrame(rng.normal(size=(n, 3)).cumsum(axis=0) + 50, index=idx, columns=names)


def current(lw, names_in, ws):
    """_forecaster_direct_multivariate.py:2324-2337."""
    lw = lw.iloc[-ws:, lw.columns.get_indexer(names_in)].copy()
    return [lw[s].to_numpy(copy=True) for s in names_in]


def proto_all_then_pos(lw, names_in, ws):
    """Convert the whole window once, then take column positions."""
    arr = lw.iloc[-ws:].to_numpy(dtype=float)
    pos = lw.columns.get_indexer(names_in)
    return [arr[:, j] for j in pos]


def proto_pos_then_convert(lw, names_in, ws, na_value=None):
    pos = lw.columns.get_indexer(names_in)
    kw = {} if na_value is None else {"na_value": na_value}
    arr = lw.iloc[-ws:, pos].to_numpy(dtype=float, **kw)
    return [arr[:, j] for j in range(len(names_in))]


def run(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        return f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"


def cmp(ref, out, w):
    if isinstance(ref, str) or isinstance(out, str):
        return f"current={'ERR ' + ref if isinstance(ref, str) else 'ok'} | proto={'ERR ' + out if isinstance(out, str) else 'ok'}"
    as_float = lambda r: pd.to_numeric(pd.Series(r), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    vals = all(np.array_equal(as_float(r), o, equal_nan=True) for r, o in zip(ref, out))
    shares = any(np.shares_memory(o, blk) for o in out for blk in
                 [w[c].to_numpy() for c in w.columns if w[c].dtype == "float64" and not w.columns.duplicated().any()])
    return (f"values={vals} dtypes {sorted({str(r.dtype) for r in ref})}->{sorted({str(o.dtype) for o in out})}"
            f" contiguous={all(o.flags.c_contiguous for o in out)} writeable={all(o.flags.writeable for o in out)}"
            f" shares memory with user's last_window={shares}")


lw = series.iloc[-ws:]
cases = {
    "same order, float": lw,
    "columns in other order": lw[["c", "a", "b"]],
    "extra numeric column": lw.assign(extra=1.0),
    "extra TEXT column": lw.assign(extra="x"),
    "extra datetime column": lw.assign(extra=pd.Timestamp("2020-01-01")),
    "int64 columns": lw.round().astype("int64"),
    "one int64 column, rest float": lw.assign(b=lw["b"].round().astype("int64")),
    "int64 > 2**53": lw.round().astype("int64").assign(a=2**53 + 1),
    "bool column": lw.assign(b=lw["b"] > 50),
    "Float64 nullable, no NA": lw.astype("Float64"),
    "Float64 nullable with NA": lw.astype("Float64").mask(lw > lw.max().max() - 1e-9),
    "Int64 nullable with NA": lw.round().astype("Int64").mask(lw > lw.max().max() - 1e-9),
    "NaN in a float column": lw.mask(lw > lw.max().max() - 1e-9),
    "longer than window_size": series.iloc[-12:],
    "duplicated column 'a'": pd.concat([lw, lw[["a"]]], axis=1),
    "missing column 'c' (check_inputs=False path)": lw[["a", "b"]],
    "object column with numbers": lw.assign(b=lw["b"].astype(object)),
}
for name, w in cases.items():
    ref = run(current, w, names, ws)
    print(f"## {name}")
    print("   all->pos        :", cmp(ref, run(proto_all_then_pos, w, names, ws), w))
    print("   pos->float      :", cmp(ref, run(proto_pos_then_convert, w, names, ws), w))
    print("   pos->float na=nan:", cmp(ref, run(proto_pos_then_convert, w, names, ws, na_value=np.nan), w))
    if isinstance(ref, str) is False and name.startswith("missing"):
        print("   NOTE current silently uses column position -1:", np.array_equal(ref[2], w.iloc[-ws:, -1].to_numpy()))

print("## real forecaster today: predict(window as given) vs predict(window.astype(float))")
for label, kw in (("no transformer", {}), ("StandardScaler", {"transformer_series": StandardScaler()}),
                  ("differentiation=1 + RollingFeatures", {"differentiation": 1,
                   "window_features": RollingFeatures(stats=["mean", "sum"], window_sizes=3)})):
    f = ForecasterDirectMultiVariate(Ridge(), level="a", lags=3, steps=2, **kw)
    f.fit(series=series)
    print(f"-- {label} (window_size={f.window_size})")
    for name in ("int64 columns", "one int64 column, rest float", "bool column", "Float64 nullable, no NA",
                 "object column with numbers", "columns in other order", "extra TEXT column"):
        w = cases[name].iloc[-f.window_size:]
        w_float = w[names].astype(float)
        try:
            same = np.array_equal(f.predict(last_window=w)["pred"].to_numpy(),
                                  f.predict(last_window=w_float)["pred"].to_numpy())
            print(f"   {name:30s}: bitwise equal to the float window = {same}")
        except Exception as e:
            print(f"   {name:30s}: {type(e).__name__}: {str(e).splitlines()[0][:70]}")
print(_env.versions())
