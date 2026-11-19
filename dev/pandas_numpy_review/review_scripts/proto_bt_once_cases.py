"""
Equivalence of `proto_bt_once.backtesting_once` with
`backtesting_forecaster_multiseries` on the cases the ceiling of the study
(`scripts/ceiling_bt_ms.py`) leaves out. For each case: predictions and metrics
(exact comparison) and the list of warnings (same warnings in the same order).

Usage: python proto_bt_once_cases.py [-v]
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import os
import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from proto_bt_blocks import backtesting_blocks
from proto_bt_once import backtesting_once
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.preprocessing import CalendarFeatures, RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries

VERBOSE = "-v" in sys.argv
# REFIT=0 (default), 1 (every fold) or an integer; PROTO=once (default) or blocks.
REFIT = int(os.environ.get("REFIT", "0"))
PROTO = os.environ.get("PROTO", "blocks" if REFIT else "once")
N_ROWS, N_SERIES, LAGS, STEPS = 320, 8, 12, 10


def make_series(kind="dict", freq="h", tz=None, ragged=False, nans=False, seed=0):
    rng = np.random.default_rng(seed)
    if freq is None:
        idx = pd.RangeIndex(N_ROWS)
    else:
        idx = pd.date_range("2021-01-01", periods=N_ROWS, freq=freq, tz=tz)
    values = np.cumsum(rng.normal(size=(N_ROWS, N_SERIES)), axis=0) + 50
    names = [f"s{j}" for j in range(N_SERIES)]
    # NOTE: With refit the NaNs of the test sets end up in a training set, and
    # Ridge cannot be fitted: the "late" variant has no NaN then.
    if nans and not (nans == "late" and REFIT):
        if nans != "late":
            values[150:153, 1] = np.nan      # in the training set
        values[245:247, 2] = np.nan          # in a last window and in a test set
        values[300:, 3] = np.nan             # NaN at the end
    wide = pd.DataFrame(values, index=idx, columns=names)
    if kind == "wide":
        if ragged:
            wide.iloc[:40, 4] = np.nan       # starts late
            wide.iloc[260:, 5] = np.nan      # ends early
            wide.iloc[:215, 6] = np.nan      # too short in the training set
        return wide
    series = {k: wide[k].copy() for k in names}
    if ragged:
        series["s4"] = series["s4"].iloc[40:]      # starts late
        series["s5"] = series["s5"].iloc[:260]     # ends before the last folds
        series["s6"] = series["s6"].iloc[215:]     # too short in the training set
        series["s7"] = series["s7"].iloc[:231]     # ends inside the first last window
    return series


def make_exog(series, kind="float", holes=False, seed=1):
    rng = np.random.default_rng(seed)
    names = list(series.keys()) if isinstance(series, dict) else list(series.columns)
    idx = series[names[0]].index if isinstance(series, dict) else series.index
    idx = pd.date_range(idx[0], periods=N_ROWS, freq=idx.freq) if isinstance(idx, pd.DatetimeIndex) else pd.RangeIndex(N_ROWS)

    def frame():
        df = pd.DataFrame(
            {"temp": rng.normal(size=N_ROWS), "promo": rng.normal(size=N_ROWS)}, index=idx
        )
        if kind == "mixed":
            df["hour"] = (np.arange(N_ROWS) % 24).astype("int64")
            dow = np.array(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])
            df["dow"] = pd.Categorical(dow[np.arange(N_ROWS) % 7], categories=list(dow))
            df["shop"] = np.where(rng.random(N_ROWS) < 0.5, "big", "small")
            df["flag"] = rng.random(N_ROWS) < 0.3
        return df

    if kind == "wide":
        return frame()
    exog = {k: frame() for k in names}
    if holes:
        exog["s0"].iloc[243:246, 0] = np.nan           # NaN in a gap / test set
        exog["s1"] = None                               # level without exog
        del exog["s2"]                                  # key missing
        exog["s3"] = exog["s3"].iloc[:275]              # ends inside a test set
        exog["s4"] = exog["s4"].drop(exog["s4"].index[251:254])  # dates missing
    return exog


def run_both(forecaster, series, exog, cv_kwargs, metric="mean_absolute_error", levels=None):
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=220, refit=REFIT if REFIT > 1 else bool(REFIT), verbose=False, **cv_kwargs)
    out = []
    for fn in ("current", "once"):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                if fn == "current":
                    res = backtesting_forecaster_multiseries(
                        forecaster=forecaster, series=series, exog=exog, cv=cv,
                        metric=metric, levels=levels, n_jobs=1, show_progress=False,
                    )
                else:
                    res = (backtesting_once if PROTO == "once" else backtesting_blocks)(
                        forecaster=forecaster, series=series, exog=exog, cv=cv,
                        metric=metric, levels=levels,
                    )
            except Exception as error:  # noqa: BLE001
                res = error
        out.append((res, [(w.category.__name__, str(w.message)) for w in rec]))
    return out


def compare(name, forecaster, series, exog=None, cv_kwargs=None, **kwargs):
    (cur, w_cur), (new, w_new) = run_both(forecaster, series, exog, cv_kwargs or {}, **kwargs)
    if isinstance(cur, Exception) or isinstance(new, Exception):
        same = type(cur) is type(new) and str(cur) == str(new)
        print(f"{name:<58} error: current={type(cur).__name__} once={type(new).__name__} same={same}")
        if VERBOSE or not same:
            print("   current:", str(cur)[:300]); print("   once   :", str(new)[:300])
        return
    res = []
    for a, b in zip(cur, new):
        try:
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            res.append("identical")
        except AssertionError as e:
            if a.shape == b.shape and all(a.columns == b.columns):
                num = a.select_dtypes("number").columns
                diff = np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))
                res.append(f"DIFFERENT (max abs diff {diff:.3g}; {str(e).splitlines()[0][:60]})")
            else:
                res.append(f"DIFFERENT shapes {a.shape} {b.shape}")
    w_same = w_cur == w_new
    w_set = sorted(set(w_cur)) == sorted(set(w_new))
    print(
        f"{name:<58} metrics={res[0]} preds={res[1]} rows={len(cur[1])} "
        f"nan_pred={int(cur[1]['pred'].isna().sum())} | warnings: current={len(w_cur)} "
        f"once={len(w_new)} same_list={w_same} same_set={w_set}"
    )
    if VERBOSE and not w_same:
        only_cur = [w for w in w_cur if w not in w_new]
        only_new = [w for w in w_new if w not in w_cur]
        for w in only_cur[:4]:
            print("   only current:", w[0], w[1][:150])
        for w in only_new[:4]:
            print("   only once   :", w[0], w[1][:150])


def lgbm():
    return LGBMRegressor(n_estimators=20, verbose=-1, n_jobs=1, random_state=1)


F = ForecasterRecursiveMultiSeries
s_dict = make_series()
s_rag = make_series(ragged=True, nans=True)
s_rag_late = make_series(ragged=True, nans="late")  # no NaN in the training set

compare("base, Ridge", F(Ridge(), lags=LAGS), s_dict)
compare("ragged + NaN, Ridge (levels dropped)", F(Ridge(), lags=LAGS), s_rag_late)
compare("ragged + NaN, LightGBM (NaN kept)", F(lgbm(), lags=LAGS), s_rag)
compare("ragged + NaN, wide DataFrame, LightGBM", F(lgbm(), lags=LAGS), make_series("wide", ragged=True, nans=True))
compare("gap=3, exog float with holes, LightGBM", F(lgbm(), lags=LAGS), s_dict, make_exog(s_dict, holes=True), {"gap": 3})
compare("gap=3, exog float with holes, Ridge", F(Ridge(), lags=LAGS), s_dict, make_exog(s_dict, holes=True), {"gap": 3})
compare("exog mixed with holes, ragged, LightGBM", F(lgbm(), lags=LAGS), s_rag, make_exog(s_rag, "mixed", holes=True))
compare("exog mixed, Ridge", F(Ridge(), lags=LAGS), s_dict, make_exog(s_dict, "mixed"))
compare("wide series + wide exog mixed, Ridge", F(Ridge(), lags=LAGS), make_series("wide"), make_exog(make_series("wide"), "wide"))
compare("dict series + wide exog (shared object)", F(Ridge(), lags=LAGS), s_dict, dict.fromkeys(s_dict, make_exog(s_dict, "wide")))
for order in (1, 2):
    compare(f"differentiation={order} + StandardScaler, Ridge", F(Ridge(), lags=LAGS, differentiation=order, transformer_series=StandardScaler()), s_rag_late)
compare("MinMaxScaler, LightGBM, ragged", F(lgbm(), lags=LAGS, transformer_series=MinMaxScaler()), s_rag)
compare("window features + calendar features, Ridge", F(Ridge(), lags=LAGS, window_features=RollingFeatures(stats=["mean", "std"], window_sizes=[7, 14]), calendar_features=CalendarFeatures(features=["hour", "day_of_week"])), s_dict)
for enc in ("onehot", "ordinal_category", None):
    compare(f"encoding={enc}, LightGBM, exog mixed", F(lgbm(), lags=LAGS, encoding=enc), s_rag, make_exog(s_rag, "mixed"))
compare("levels subset in another order", F(Ridge(), lags=LAGS), s_rag_late, levels=["s5", "s0", "s3"])
compare("dropna_from_series=True, Ridge", F(Ridge(), lags=LAGS, dropna_from_series=True), s_rag)
compare("overlapping folds (fold_stride=4)", F(Ridge(), lags=LAGS), s_rag_late, cv_kwargs={"fold_stride": 4})
compare("skip_folds=2, incomplete last fold", F(Ridge(), lags=LAGS), s_rag_late, cv_kwargs={"skip_folds": 2})
compare("metrics MAE + MASE + RMSSE", F(Ridge(), lags=LAGS), s_rag_late, metric=["mean_absolute_error", "mean_absolute_scaled_error", "root_mean_squared_scaled_error"])
compare("transformer_exog=StandardScaler, exog float", F(Ridge(), lags=LAGS, transformer_exog=StandardScaler()), s_dict, make_exog(s_dict))
compare("RangeIndex, ragged", F(Ridge(), lags=LAGS), make_series(freq=None, ragged=True, nans="late"))
compare("daily index with timezone", F(Ridge(), lags=LAGS), make_series(freq="D", tz="Europe/Madrid", ragged=True))
print(f"REFIT={REFIT} PROTO={PROTO}", _env.versions())
