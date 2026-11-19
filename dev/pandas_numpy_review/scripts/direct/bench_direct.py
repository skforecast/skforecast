import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from prof import report, wall

warnings.filterwarnings("ignore")
which = sys.argv[1] if len(sys.argv) > 1 else "all"
N = 2000
idx = pd.date_range("2020-01-01", periods=N + 1200, freq="h")
rng = np.random.default_rng(0)
yfull = pd.Series(np.sin(np.arange(len(idx)) * 2 * np.pi / 24) * 10 + rng.normal(size=len(idx)) + 50,
                  index=idx, name="y")
exnum = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["e1", "e2", "e3"])
excat = pd.DataFrame({"cat": pd.Categorical(np.array(list("abcdefg"))[idx.dayofweek])}, index=idx)
exogs = {"none": None, "num3": exnum, "cat1": excat}
ests = {"ridge": Ridge(), "lgbm": LGBMRegressor(verbose=-1, n_estimators=50, n_jobs=1)}

for ename, est in ests.items():
    for xname, ex in exogs.items():
        if which != "all" and which not in (ename, xname, f"{ename}-{xname}"):
            continue
        y = yfull.iloc[:N]
        e = ex.iloc[:N] if ex is not None else None
        e_fut = ex.iloc[N:N + 24] if ex is not None else None
        f = ForecasterDirect(est, steps=24, lags=24, n_jobs=1)
        print(f"\n===== ForecasterDirect {ename} exog={xname}")
        t_fit = wall(lambda: f.fit(y, exog=e), n=3 if ename=="ridge" else 1, warm=1 if ename=="ridge" else 0)
        t_pred = wall(lambda: f.predict(exog=e_fut), n=50)
        print(f"wall fit {t_fit*1e3:.1f} ms | predict {t_pred*1e3:.3f} ms")
        report("fit", lambda: f.fit(y, exog=e), n=2 if ename=="ridge" else 1)
        report("predict", lambda: f.predict(exog=e_fut), n=200)
        f.fit(y, exog=e, store_in_sample_residuals=True)
        t_pi = wall(lambda: f.predict_interval(exog=e_fut, n_boot=250), n=20)
        t_pq = wall(lambda: f.predict_quantiles(exog=e_fut, n_boot=250), n=20)
        t_pb = wall(lambda: f.predict_bootstrapping(exog=e_fut, n_boot=250), n=20)
        t_sir = wall(lambda: f.set_in_sample_residuals(y, exog=e), n=3 if ename=="ridge" else 1, warm=1 if ename=="ridge" else 0)
        print(f"wall predict_interval(boot) {t_pi*1e3:.3f} ms | quantiles {t_pq*1e3:.3f} | bootstrapping {t_pb*1e3:.3f} | set_in_sample_residuals {t_sir*1e3:.1f}")
        report("predict_interval(bootstrapping)", lambda: f.predict_interval(exog=e_fut, n_boot=250), n=50)
        # backtesting refit False, 50 folds
        cv = TimeSeriesFold(steps=24, initial_train_size=800, refit=False)
        def bt():
            backtesting_forecaster(f, yfull.iloc[:2000], cv=cv, exog=ex.iloc[:2000] if ex is not None else None,
                                   metric="mean_absolute_error", n_jobs=1, show_progress=False)
        e800 = ex.iloc[:800] if ex is not None else None
        nrep = 3 if ename == "ridge" else 1
        t_bt = wall(bt, n=nrep)
        t_fit800 = wall(lambda: f.fit(yfull.iloc[:800], exog=e800), n=nrep)
        print(f"wall backtesting refit=False 50 folds {t_bt*1e3:.1f} ms; initial fit(800) {t_fit800*1e3:.1f} ms; per fold excl. fit {(t_bt-t_fit800)/50*1e3:.3f} ms")
        if ename == "ridge":
            report("backtesting refit=False (incl. initial fit)", bt, n=1)
            report("fit(800) to subtract", lambda: f.fit(yfull.iloc[:800], exog=e800), n=1)
