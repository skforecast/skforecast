"""
BUG-search-osa-direct-categorical: `grid_search_forecaster` and
`grid_search_forecaster_multiseries` with `OneStepAheadFold` evaluate the
candidates of `ForecasterDirect` and `ForecasterDirectMultiVariate` with the
categorical exog as a numeric column when the estimator is configured through
`set_params` (XGBoost, HistGradientBoosting). The bayesian searches do not.

For each estimator and forecaster, prints the MAE of one candidate with a
categorical exog (`cat`) and with the same column as float (`num`), in the grid
and in the bayesian search. The bug shows as `grid cat` equal to `grid num` and
different from `bayes cat` (Direct with XGBoost: 0.555, 0.555 and 0.221).
Recursive, MultiSeries and LightGBM, which is configured through `fit`
arguments, are not affected.

Usage: python grid_osa.py  (with `PYTHONPATH` set to the repository root if the
installed skforecast is not this code; the first line printed is the path of
the package used)
"""
import warnings
import numpy as np
import pandas as pd
import skforecast
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    OneStepAheadFold, grid_search_forecaster, bayesian_search_forecaster,
    grid_search_forecaster_multiseries, bayesian_search_forecaster_multiseries,
)

warnings.simplefilter("ignore")
print(skforecast.__file__, "pandas", pd.__version__)
rng = np.random.default_rng(7)
n = 400
idx = pd.date_range("2020-01-01", periods=n, freq="D")
codes = rng.permutation(np.arange(n) % 8)
effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0, 7.0, -5.0])
cat = pd.Series(pd.Categorical(codes), index=idx)
noise = rng.normal(scale=0.3, size=n)
y = pd.Series(effect[codes] + noise, index=idx, name="y")
series = pd.DataFrame(
    {"l1": y.to_numpy(), "l2": effect[codes] * 0.5 + rng.normal(scale=0.3, size=n)},
    index=idx,
)
exog_cat = pd.DataFrame({"c": cat})
exog_num = pd.DataFrame({"c": codes.astype(float)}, index=idx)
cv = OneStepAheadFold(initial_train_size=300)
GRID = {"max_depth": [2]}


def space(trial):
    return {"max_depth": trial.suggest_categorical("max_depth", [2])}


ESTS = {
    "XGB": lambda: XGBRegressor(n_estimators=20, max_depth=2, n_jobs=1, random_state=1),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=20, max_depth=2, random_state=1),
    "LGBM": lambda: LGBMRegressor(
        n_estimators=20, max_depth=2, n_jobs=1, random_state=1, verbose=-1,
        min_data_per_group=5, cat_smooth=1,
    ),
}


def single(make_f, exog, kind):
    f = make_f()
    kw = dict(
        forecaster=f, y=y, cv=cv, metric="mean_absolute_error", exog=exog,
        return_best=False, show_progress=False, verbose=False,
    )
    if kind == "grid":
        r = grid_search_forecaster(param_grid=GRID, **kw)
    else:
        r, _ = bayesian_search_forecaster(search_space=space, n_trials=1, **kw)
    return float(r["mean_absolute_error"].iloc[0])


def multi(make_f, exog, kind):
    f = make_f()
    kw = dict(
        forecaster=f, series=series, cv=cv, metric="mean_absolute_error", exog=exog,
        return_best=False, show_progress=False, verbose=False,
    )
    if kind == "grid":
        r = grid_search_forecaster_multiseries(param_grid=GRID, **kw)
    else:
        r, _ = bayesian_search_forecaster_multiseries(
            search_space=space, n_trials=1, **kw
        )
    col = [c for c in r.columns if "mean_absolute_error" in c][0]
    return float(r[col].iloc[0])


for name, make in ESTS.items():
    FCS = {
        "Recursive": (single, lambda: ForecasterRecursive(estimator=make(), lags=3)),
        "Direct": (
            single,
            lambda: ForecasterDirect(estimator=make(), lags=3, steps=2, n_jobs=1),
        ),
        "MultiSeries": (
            multi,
            lambda: ForecasterRecursiveMultiSeries(
                estimator=make(), lags=3, encoding="ordinal"
            ),
        ),
        "DMV": (
            multi,
            lambda: ForecasterDirectMultiVariate(
                estimator=make(), level="l1", lags=3, steps=2, n_jobs=1,
                transformer_series=None,
            ),
        ),
    }
    for fname, (fn, make_f) in FCS.items():
        out = {}
        for label, exog, kind in (
            ("grid cat", exog_cat, "grid"),
            ("bayes cat", exog_cat, "bayes"),
            ("grid num", exog_num, "grid"),
            ("bayes num", exog_num, "bayes"),
        ):
            try:
                out[label] = f"{fn(make_f, exog, kind):.6f}"
            except Exception as e:
                out[label] = f"{type(e).__name__}: {str(e)[:50]}"
        print(f"{name:<5} {fname:<12} " + " | ".join(f"{k}={v}" for k, v in out.items()))
