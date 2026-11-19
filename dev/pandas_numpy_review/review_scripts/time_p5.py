"""
P5 measured on the whole single-series backtesting (the study estimated it from
the pieces of one `predict` call). 50 folds of 24 steps, no refit.

A: no validation per fold (`predict(check_inputs=False)`).
B: A + index of the predictions taken by position from the index of `y`
   instead of `expand_index`.
Both are compared with the current output. Run alone, OMP_NUM_THREADS=1.
"""
import _env  # noqa: F401
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

import skforecast.recursive._forecaster_recursive as fr
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.recursive import ForecasterRecursive

warnings.simplefilter("ignore")
N, STEPS, FOLDS = 3000, 24, 50
rng = np.random.default_rng(0)
orig_predict = ForecasterRecursive.predict
orig_expand = fr.expand_index


def best(fn, n=5):
    out, res = np.inf, None
    for _ in range(n):
        t = time.perf_counter(); res = fn(); out = min(out, time.perf_counter() - t)
    return out, res


def run(label, idx, est, with_exog):
    y = pd.Series(np.cumsum(rng.normal(size=N)), index=idx, name="y")
    exog = pd.DataFrame(rng.normal(size=(N, 3)), index=idx, columns=list("abc")) if with_exog else None
    f = ForecasterRecursive(est, lags=24)
    cv = TimeSeriesFold(steps=STEPS, initial_train_size=N - FOLDS * STEPS, refit=False, verbose=False)
    call = lambda: backtesting_forecaster(f, y=y, exog=exog, cv=cv, metric="mean_absolute_error", n_jobs=1, show_progress=False)
    t_fit, _ = best(lambda: ForecasterRecursive(est, lags=24).fit(y.iloc[: N - FOLDS * STEPS], exog=None if exog is None else exog.iloc[: N - FOLDS * STEPS]), 3)

    ForecasterRecursive.predict = orig_predict; fr.expand_index = orig_expand
    t0, r0 = best(call)

    def predict_no_check(self, *a, **k):
        k["check_inputs"] = False
        return orig_predict(self, *a, **k)
    ForecasterRecursive.predict = predict_no_check
    t_a, r_a = best(call)

    full = y.index
    def expand_by_position(index, steps):
        pos = full.get_loc(index[-1]) + 1
        return full[pos:pos + steps]
    fr.expand_index = expand_by_position
    t_b, r_b = best(call)
    ForecasterRecursive.predict = orig_predict; fr.expand_index = orig_expand

    def same(r):
        try:
            pd.testing.assert_frame_equal(r0[0], r[0], check_exact=True)
            pd.testing.assert_frame_equal(r0[1], r[1], check_exact=True)
            return "identical"
        except AssertionError as e:
            return "DIFFERENT: " + str(e).splitlines()[0][:60]
    pf = lambda t: (t - t_fit) / FOLDS * 1e3
    print(f"{label:<44} today {t0 * 1e3:6.1f} ms ({pf(t0):.2f} per fold) | A {t_a * 1e3:6.1f} ms ({(t_a / t0 - 1) * 100:+.0f} %, {same(r_a)}) "
          f"| B {t_b * 1e3:6.1f} ms ({(t_b / t0 - 1) * 100:+.0f} %, {same(r_b)}) | fit {t_fit * 1e3:.1f} ms")


hourly = pd.date_range("2020-01-01", periods=N, freq="h")
daily_tz = pd.date_range("2012-01-01", periods=N, freq="D", tz="Europe/Madrid")
run("Ridge, hourly, exog float", hourly, Ridge(), True)
run("Ridge, hourly, no exog", hourly, Ridge(), False)
run("LightGBM(50), hourly, exog float", hourly, LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1), True)
run("Ridge, daily with timezone, exog float", daily_tz, Ridge(), True)
run("LightGBM(50), daily with timezone, exog float", daily_tz, LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1), True)
print(_env.versions())
