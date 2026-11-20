"""Micro-benchmarks of the pandas ops of ForecasterRecursiveMultiSeries fit
(100 series x 1000 daily points, lags 24, dict exog 2 float + 1 category)."""
import sys
import timeit
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
from skforecast.utils.utils import check_exog, check_preprocess_exog_multiseries

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
LEN, W = 1000, 24
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=LEN, freq="D")
series = {f"s{i:04d}": pd.Series(rng.normal(size=LEN), index=idx, name=f"s{i:04d}") for i in range(N)}
cats = ["alpha", "beta", "gamma", "delta"]
exog = {
    k: pd.DataFrame(
        {"e1": rng.normal(size=LEN), "e2": rng.normal(size=LEN),
         "c1": pd.Categorical(rng.choice(cats, size=LEN), categories=cats)},
        index=idx,
    )
    for k in series
}
wide = pd.DataFrame(rng.normal(size=(LEN, 3)), index=idx, columns=["e1", "e2", "e3"])


def t(stmt, number=2, repeat=3):
    return min(timeit.repeat(stmt, number=number, repeat=repeat, globals=globals())) / number


res = {}
res["per-series: series v.copy() (check_preprocess_series)"] = t(lambda: [v.copy() for v in series.values()])
res["per-series: v.isna().to_numpy().all() (check_preprocess_series)"] = t(lambda: [v.isna().to_numpy().all() for v in series.values()])
res["per-series: exog v.copy() (check_preprocess_exog_multiseries)"] = t(lambda: [v.copy() for v in exog.values()])
res["per-series: check_exog(v) (check_preprocess_exog_multiseries)"] = t(lambda: [check_exog(v, allow_nan=True) for v in exog.values()])
res["once: dtypes DataFrame + nunique (check_preprocess_exog_multiseries)"] = t(
    lambda: (pd.DataFrame({k: df.dtypes for k, df in exog.items()}).nunique(axis=1) == 1).all())
res["once: whole check_preprocess_exog_multiseries"] = t(
    lambda: check_preprocess_exog_multiseries(list(series), pd.DatetimeIndex, exog, {k: None for k in series}))
res["per-series: iat[0]/iat[-1] isna + index.equals (align)"] = t(
    lambda: [(pd.isna(v.iat[0]) or pd.isna(v.iat[-1]), v.index.equals(exog[k].index)) for k, v in series.items()])
res["per-series: exog_df.iloc[W:, ] (X_train_exog_buffer)"] = t(lambda: [e.iloc[W:, ] for e in exog.values()])
res["per-series: wide.iloc[W:, ] same object N times"] = t(lambda: [wide.iloc[W:, ] for _ in series])
buf = [e.iloc[W:, ] for e in exog.values()]
bufw = [wide.iloc[W:, ] for _ in series]
res["once: pd.concat(N exog dict, with category)"] = t(lambda: pd.concat(buf, axis=0))
res["once: pd.concat(N x wide float)"] = t(lambda: pd.concat(bufw, axis=0))
res["  alt: np.tile(wide.values[W:]) (no pandas)"] = t(lambda: np.tile(wide.to_numpy()[W:], (N, 1)))
Xe = pd.concat(buf, axis=0)
res["  alt: np.concatenate float cols + codes (dict exog)"] = t(
    lambda: (np.concatenate([e[["e1", "e2"]].to_numpy()[W:] for e in exog.values()]),
             np.concatenate([e["c1"].cat.codes.to_numpy()[W:] for e in exog.values()])))
parts = [v.index[W:] for v in series.values()]
res["once: train_index = parts[0].append(parts[1:])"] = t(lambda: parts[0].append(parts[1:]))
ti = parts[0].append(parts[1:])
res["once: X_train_exog.index.equals(train_index)"] = t(lambda: Xe.index.equals(ti))
res["once: get_exog_dtypes x2 + categorical detection"] = t(
    lambda: [c for c, d in Xe.dtypes.items() if not pd.api.types.is_numeric_dtype(d) and not pd.api.types.is_bool_dtype(d)])
enc = OrdinalEncoder(dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan,
                     encoded_missing_value=np.nan).set_output(transform="pandas")
res["once: OrdinalEncoder(set_output pandas).fit_transform(Xe[['c1']])"] = t(lambda: enc.fit_transform(Xe[["c1"]]), number=3)
enc2 = OrdinalEncoder(dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan,
                      encoded_missing_value=np.nan)
res["  same, default output (numpy)"] = t(lambda: enc2.fit_transform(Xe[["c1"]]), number=3)
res["  same, input as object ndarray"] = t(lambda: enc2.fit_transform(Xe[["c1"]].to_numpy()), number=3)
codes = Xe["c1"].cat.codes.to_numpy()


def via_codes():
    # Same categories as OrdinalEncoder (sorted unique of observed values),
    # computed from the pandas codes: no object array, no Python sort of N rows.
    c = Xe["c1"]
    observed = np.unique(c.cat.codes.to_numpy())
    observed = observed[observed >= 0]
    labels = c.cat.categories.to_numpy()[observed]
    order = np.argsort(labels, kind="stable")
    lut = np.full(len(c.cat.categories), np.nan)
    lut[observed[order]] = np.arange(len(observed), dtype=float)
    cc = c.cat.codes.to_numpy()
    return np.where(cc >= 0, lut[cc], np.nan)


res["  alt: codes + LUT (category dtype only)"] = t(via_codes)
assert np.array_equal(via_codes(), enc2.fit_transform(Xe[["c1"]])[:, 0], equal_nan=True)
Xe2 = Xe.copy()
out = enc.fit_transform(Xe[["c1"]])
res["once: X_train_exog[cats] = encoded (setitem)"] = t(lambda: Xe2.__setitem__(["c1"], out))
res["per-series: last_window_ v.iloc[-W:].copy()"] = t(lambda: {k: v.iloc[-W:].copy() for k, v in series.items()})
res["per-series: training_range_ idx[[0, -1]]"] = t(lambda: {k: v.index[[0, -1]] for k, v in series.items()})
res["  alt: idx[0], idx[-1] (scalars)"] = t(lambda: {k: (v.index[0], v.index[-1]) for k, v in series.items()})
X = pd.DataFrame(np.asfortranarray(rng.normal(size=(len(ti), 28))), index=ti)
res["once: pd.isna(X_train DataFrame) (_filter_nan)"] = t(lambda: np.asarray(pd.isna(X)).any())
Xv = X.to_numpy()
res["  alt: np.isnan(values).any()"] = t(lambda: np.isnan(Xv).any())

tot_ps = 0.0
for k, v in res.items():
    print(f"{v*1e3:9.3f} ms  {k}")
print(pd.__version__, np.__version__)
