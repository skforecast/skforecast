"""
What skforecast does in pandas 3 when the index resolution of the training
data ([ns]) and of the data passed at predict time ([us]: the default of
pd.date_range / pd.to_datetime from strings in pandas 3) differ, and the
reverse. Also TimeSeriesFold with a date as initial_train_size and
expand_index on [us] and [ns] indexes.

    bash dev/pandas3_migration/run_in_both.sh experiments/exp_mixed_resolution.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.utils import expand_index

warnings.simplefilter("ignore")
print("pandas", pd.__version__)
rng = np.random.default_rng(0)
n = 200
idx_ns = pd.date_range("2020-01-01", periods=n + 20, freq="D").as_unit("ns")
idx_us = idx_ns.as_unit("us") if hasattr(idx_ns, "as_unit") else idx_ns
y = pd.Series(rng.normal(size=n + 20).cumsum(), index=idx_ns, name="y")
exog = pd.DataFrame({"e": rng.normal(size=n + 20)}, index=idx_ns)


def attempt(label, fn):
    try:
        out = fn()
        dtype = out.index.dtype if hasattr(out, "index") else out
        print(f"{label:55s} OK   -> {dtype}")
        return out
    except Exception as e:  # noqa: BLE001
        print(f"{label:55s} FAIL {type(e).__name__}: {str(e)[:120]}")


for fit_unit, pred_unit in (("ns", "us"), ("us", "ns"), ("ns", "ns"), ("us", "us")):
    y_fit = y.iloc[:n].copy(); y_fit.index = y_fit.index.as_unit(fit_unit)
    e_fit = exog.iloc[:n].copy(); e_fit.index = e_fit.index.as_unit(fit_unit)
    e_pred = exog.iloc[n:n + 10].copy(); e_pred.index = e_pred.index.as_unit(pred_unit)
    lw = y.iloc[n - 30:n].copy(); lw.index = lw.index.as_unit(pred_unit)
    f = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    f.fit(y=y_fit, exog=e_fit)
    print(f"-- fit index [{fit_unit}] (index_freq_={f.index_freq_!r}), predict inputs [{pred_unit}]")
    attempt("  predict(exog)", lambda: f.predict(steps=10, exog=e_pred))
    attempt("  predict(exog, last_window)", lambda: f.predict(steps=10, exog=e_pred, last_window=lw))
    p = attempt("  predict(exog, last_window).index", lambda: f.predict(steps=10, exog=e_pred, last_window=lw))
    if p is not None:
        ref = f.predict(steps=10, exog=exog.iloc[n:n + 10], last_window=y.iloc[n - 30:n])
        print(f"  {'values identical to the all-[ns] prediction:':55s} {np.array_equal(p.to_numpy(), ref.to_numpy())}")

print("-- expand_index")
for unit in ("ns", "us"):
    attempt(f"  expand_index([{unit}] with freq)", lambda: expand_index(idx_ns.as_unit(unit)[:50], 5))
    attempt(f"  expand_index([{unit}] without freq, inferred)", lambda: expand_index(pd.DatetimeIndex(idx_ns.as_unit(unit)[:50].to_numpy()), 5))

print("-- TimeSeriesFold with a date as initial_train_size")
for unit in ("ns", "us"):
    y_u = y.copy(); y_u.index = y_u.index.as_unit(unit)
    cv = TimeSeriesFold(steps=10, initial_train_size="2020-06-01", refit=False)
    attempt(f"  backtesting_forecaster([{unit}], initial_train_size='2020-06-01')",
            lambda: backtesting_forecaster(ForecasterRecursive(LinearRegression(), lags=5), y=y_u, cv=cv,
                                           metric="mean_absolute_error", show_progress=False, verbose=False)[1])
    cv2 = TimeSeriesFold(steps=10, initial_train_size=pd.Timestamp("2020-06-01"), refit=False)
    attempt(f"  backtesting_forecaster([{unit}], initial_train_size=Timestamp)",
            lambda: backtesting_forecaster(ForecasterRecursive(LinearRegression(), lags=5), y=y_u, cv=cv2,
                                           metric="mean_absolute_error", show_progress=False, verbose=False)[1])

print("-- multiseries: series [ns] as dict, exog [us] as dict")
series = {"a": y.iloc[:n], "b": (y.iloc[:n] * 2).rename("b")}
exog_us = {k: exog.iloc[:n + 10].copy() for k in series}
for k in exog_us:
    exog_us[k].index = exog_us[k].index.as_unit("us")
fm = ForecasterRecursiveMultiSeries(estimator=LinearRegression(), lags=5, encoding="ordinal")
attempt("  fit(series [ns], exog [us])", lambda: (fm.fit(series=series, exog={k: v.iloc[:n] for k, v in exog_us.items()}), fm.last_window_["a"])[1])
attempt("  predict(exog [us])", lambda: fm.predict(steps=10, exog={k: v.iloc[n:n + 10] for k, v in exog_us.items()}))
X, _ = fm.create_train_X_y(series=series, exog={k: v.iloc[:n] for k, v in exog_us.items()})
print(f"  {'create_train_X_y index dtype:':55s} {X.index.dtype}")
