import os, sys, warnings, timeit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.utils import check_predict_input, expand_index
from prof import report, wall

warnings.filterwarnings("ignore")
nser = int(sys.argv[1]) if len(sys.argv) > 1 else 10
only = sys.argv[2] if len(sys.argv) > 2 else "all"
N = 2000
idx = pd.date_range("2020-01-01", periods=N + 24, freq="h")
rng = np.random.default_rng(0)
series = pd.DataFrame(rng.normal(size=(len(idx), nser)).cumsum(axis=0) + 100, index=idx,
                      columns=[f"s{i}" for i in range(nser)])
exnum = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["e1", "e2", "e3"])
excat = pd.DataFrame({"cat": pd.Categorical(np.array(list("abcdefg"))[idx.dayofweek])}, index=idx)
exogs = {"none": None, "num3": exnum, "cat1": excat}
ests = {"ridge": Ridge(), "lgbm": LGBMRegressor(verbose=-1, n_estimators=50, n_jobs=1)}


def t(label, fn, n=1000):
    r = min(timeit.repeat(fn, number=n, repeat=5)) / n
    print(f"   {label:66s} {r*1e6:8.1f} us")
    return r


for ename, est in ests.items():
    for xname, ex in exogs.items():
        if only != "all" and only not in (ename, xname, f"{ename}-{xname}"):
            continue
        s = series.iloc[:N]
        e = ex.iloc[:N] if ex is not None else None
        ef = ex.iloc[N:] if ex is not None else None
        f = ForecasterDirectMultiVariate(est, level="s0", steps=24, lags=24, n_jobs=1)
        print(f"\n===== DMV {nser} series {ename} exog={xname} (transformer_series=StandardScaler default)")
        rep = 3 if ename == "ridge" else 1
        t_fit = wall(lambda: f.fit(s, exog=e), n=rep, warm=1 if ename == "ridge" else 0)
        t_pred = wall(lambda: f.predict(exog=ef), n=50)
        print(f"wall fit {t_fit*1e3:.1f} ms | predict {t_pred*1e3:.3f} ms")
        if ename == "ridge":
            report("fit", lambda: f.fit(s, exog=e), n=1)
        report("predict", lambda: f.predict(exog=ef), n=200)
        f.fit(s, exog=e, store_in_sample_residuals=True)
        t_pi = wall(lambda: f.predict_interval(exog=ef, n_boot=250), n=20)
        t_pb = wall(lambda: f.predict_bootstrapping(exog=ef, n_boot=250), n=20)
        t_pq = wall(lambda: f.predict_quantiles(exog=ef, n_boot=250), n=20)
        print(f"wall predict_interval(boot) {t_pi*1e3:.3f} ms | bootstrapping {t_pb*1e3:.3f} | quantiles {t_pq*1e3:.3f}")
        cv = TimeSeriesFold(steps=24, initial_train_size=N - 50 * 24, refit=False)
        def bt():
            backtesting_forecaster_multiseries(f, series.iloc[:N], cv=cv, levels="s0",
                                               exog=ex.iloc[:N] if ex is not None else None,
                                               metric="mean_absolute_error", n_jobs=1, show_progress=False)
        e800 = ex.iloc[:N - 1200] if ex is not None else None
        t_bt = wall(bt, n=rep, warm=1 if ename == "ridge" else 0)
        t_f8 = wall(lambda: f.fit(series.iloc[:N - 1200], exog=e800), n=rep, warm=0)
        print(f"wall backtesting refit=False 50 folds {t_bt*1e3:.1f} ms; initial fit {t_f8*1e3:.1f} ms; per fold excl fit {(t_bt-t_f8)/50*1e3:.3f} ms")
        if ename == "ridge":
            report("backtesting refit=False (incl initial fit)", bt, n=1)
            report("initial fit to subtract", lambda: f.fit(series.iloc[:N - 1200], exog=e800), n=1)
        if ename == "ridge":
            f.fit(s, exog=e)
            lw = f.last_window_
            print("  micro (predict pieces):")
            t("predict total", lambda: f.predict(exog=ef))
            t("check_predict_input", lambda: check_predict_input(
                forecaster_name="ForecasterDirectMultiVariate", steps=list(range(1, 25)), is_fitted=True,
                exog_in_=f.exog_in_, index_type_=f.index_type_, index_freq_=f.index_freq_,
                window_size=f.window_size, last_window=lw, exog=ef, exog_names_in_=f.exog_names_in_,
                max_step=24, series_names_in_=f.X_train_series_names_in_))
            cols = f.X_train_series_names_in_
            t("lw.iloc[-ws:, get_indexer(cols)].copy()", lambda: lw.iloc[-f.window_size:, lw.columns.get_indexer(cols)].copy())
            lw2 = lw.iloc[-f.window_size:, lw.columns.get_indexer(cols)].copy()
            t("per-series lw[col].to_numpy(copy=True) x nser", lambda: [lw2[c].to_numpy(copy=True) for c in cols])
            t("alt: lw.to_numpy(dtype=float, copy=True) once + column views", lambda: [a for a in lw.iloc[-f.window_size:, lw.columns.get_indexer(cols)].to_numpy(dtype=float, copy=True).T])
            from skforecast.utils import transform_numpy
            arrs = [lw2[c].to_numpy(copy=True) for c in cols]
            t("transform_numpy (StandardScaler fast path) x nser", lambda: [transform_numpy(a, f.transformer_series_[c]) for a, c in zip(arrs, cols)])
            t("expand_index", lambda: expand_index(lw.index, 24))
            pidx = expand_index(lw.index, 24); p = np.zeros(24)
            t("output DataFrame {level, pred}", lambda: pd.DataFrame({"level": np.tile([f.level], 24), "pred": p}, index=pidx))
            if xname == "cat1":
                enc = f.categorical_encoder
                def assign():
                    e2 = ef.copy(deep=False)
                    e2[["cat"]] = enc.transform(e2[["cat"]])
                    return e2
                t("exog copy + OrdinalEncoder.transform + assign", assign)
