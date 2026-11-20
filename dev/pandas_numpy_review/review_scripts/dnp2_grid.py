"""
D-NP2: grid of 336 configurations of `ForecasterRecursiveMultiSeries` (5
estimators, 4 encodings, `categorical_features` 'auto', None or a list that
omits the `category` column, 4 layouts of its categories, series as DataFrame
or as dict with different lengths and NaN). For each one: fit, predict,
in-sample residuals (also `set_in_sample_residuals`), bootstrapped intervals,
and the MAE of a one-step-ahead validation and of a backtesting of one step
without refit, which must be equal.

Run it with each tree and compare with `dnp2_cmp.py grid OUTDIR`:
    SK_TREE=<0.28.x>                 bash run3.sh dnp2_grid.py OUTDIR
    SK_TREE=<0.28.x + dnp2_proto.patch> bash run3.sh dnp2_grid.py OUTDIR
`GRID_EST=lgbm,ridge` limits the estimators. `DNP2_CATBOOST_DF=1` (prototype
only) keeps a DataFrame for CatBoost with categorical features.
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from p1_common import make_data
from skforecast.model_selection import (
    OneStepAheadFold, TimeSeriesFold, backtesting_forecaster_multiseries,
    grid_search_forecaster_multiseries,
)
from skforecast.recursive import ForecasterRecursiveMultiSeries

EST = {
    "lgbm": lambda: LGBMRegressor(verbose=-1, n_estimators=40, min_child_samples=5, random_state=1, n_jobs=1),
    "xgb": lambda: XGBRegressor(n_estimators=40, random_state=1, n_jobs=1),
    "hgb": lambda: HistGradientBoostingRegressor(max_iter=40, random_state=1),
    "catboost": lambda: CatBoostRegressor(iterations=40, verbose=0, random_state=1, thread_count=1),
    "ridge": lambda: Ridge(),
}
CATS = {"10..80": [10, 20, 30, 40, 50, 60, 70, 80], "0..7": list(range(8)),
        "perm": [3, 0, 7, 1, 5, 2, 6, 4], "text": list("abcdefgh")}


def extra(exog, exog_p, rng):
    exog["i"] = rng.integers(0, 5, len(exog))  # int column, not categorical
    exog_p["i"] = 2


def run(est, encoding, cf, cats, series_kind):
    series, exog, exog_p = make_data(CATS[cats], n_series=6, extra=extra)
    if series_kind == "dict":
        series = {k: v.iloc[10 * j:].copy() for j, (k, v) in enumerate(series.items())}
        series["a"].iloc[50:60] = np.nan
    else:
        series = pd.DataFrame(series)
    cf_value = {"auto": "auto", "None": None, "list": ["i"]}[cf]
    out = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            f = ForecasterRecursiveMultiSeries(
                EST[est](), lags=3, encoding=encoding, categorical_features=cf_value,
                dropna_from_series=False,
            )
            f.fit(series=series, exog=exog, store_in_sample_residuals=True)
            out["fit"] = "ok"
            out["resid"] = np.concatenate([np.sort(v) for v in f.in_sample_residuals_.values()])
        except Exception as e:  # noqa
            out["fit"] = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
        if out["fit"] == "ok":
            try:
                out["pred"] = f.predict(steps=8, exog=exog_p)["pred"].to_numpy()
            except Exception as e:  # noqa
                out["pred"] = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            try:
                f.set_in_sample_residuals(series=series, exog=exog)
                out["resid2"] = np.concatenate([np.sort(v) for v in f.in_sample_residuals_.values()])
            except Exception as e:  # noqa
                out["resid2"] = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            try:
                out["interval"] = f.predict_interval(
                    steps=8, exog=exog_p, method="bootstrapping", n_boot=20, random_state=1
                ).iloc[:, 1:].to_numpy(dtype=float)
            except Exception as e:  # noqa
                out["interval"] = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
        for name, cv in (
            ("osa", OneStepAheadFold(initial_train_size=220)),
            ("bt", TimeSeriesFold(initial_train_size=220, steps=1, refit=False)),
        ):
            try:
                f = ForecasterRecursiveMultiSeries(
                    EST[est](), lags=3, encoding=encoding, categorical_features=cf_value,
                    dropna_from_series=False,
                )
                if name == "osa":
                    res = grid_search_forecaster_multiseries(
                        f, series=series, exog=exog, cv=cv, param_grid={}, lags_grid=[3],
                        metric="mean_absolute_error", return_best=False, show_progress=False,
                        verbose=False,
                    )
                    out[name] = float(res.filter(like="mean_absolute_error").iloc[0, 0])
                else:
                    m, _ = backtesting_forecaster_multiseries(
                        f, series=series, exog=exog, cv=cv, metric="mean_absolute_error",
                        show_progress=False,
                    )
                    out[name] = float(m.loc[m["levels"] == "weighted_average", "mean_absolute_error"].iloc[0])
            except Exception as e:  # noqa
                out[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
    out["warnings"] = sorted({x.category.__name__ + ": " + str(x.message)[:60] for x in w
                              if "skforecast" in x.category.__module__})
    return out


results = {}
for est in (os.environ['GRID_EST'].split(',') if os.environ.get('GRID_EST') else EST):
    for encoding in ("ordinal", "ordinal_category", "onehot", None):
        for cf in ("auto", "None", "list"):
            for cats in CATS:
                for series_kind in ("frame", "dict"):
                    if series_kind == "dict" and (cats not in ("10..80", "text") or est == "ridge"):
                        continue
                    results[(est, encoding, cf, cats, series_kind)] = run(est, encoding, cf, cats, series_kind)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
suffix = "_catboost_df" if os.environ.get("DNP2_CATBOOST_DF") == "1" else ""
path = out / f"grid_{_env.tree()}{suffix}_{_env.tag()}.pkl"
pickle.dump({"versions": _env.versions(), "results": results}, open(path, "wb"))
print(len(results), "cases ->", path, _env.versions())
