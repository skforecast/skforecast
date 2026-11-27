"""The other forecasters must not change. Usage: others_unchanged.py <out.pkl>"""
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import skforecast
print(skforecast.__file__, flush=True)
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveClassifier
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold, backtesting_forecaster,
    grid_search_forecaster, bayesian_search_forecaster,
    backtesting_forecaster_multiseries, grid_search_forecaster_multiseries,
)
import common as C

results = {}


def run(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001
            out = C.summarize_exception(e)
    return out, sorted({(x.category.__name__, str(x.message)[:300]) for x in w
                        if "skforecast" in x.category.__module__})


for est_name in ["lgbm", "xgb", "hgb", "ridge", "catboost"]:
    for dtype_name in ["float", "cat_int_vals", "cat_str", "mixed", "bool", "nullable_nona"]:
        series, exog, exog_fut, cat_like, cat_cols = C.make_data(dtype_name)
        y = series["s_a"]
        for cf_mode in C.CATFEATS:
            cf = C.catfeat_value(cf_mode, cat_like, list(exog.columns))
            if cf == []:
                continue
            base_key = (est_name, dtype_name, cf_mode)
            makers = {
                "Recursive": lambda: ForecasterRecursive(
                    estimator=C.make_estimator(est_name), lags=3, categorical_features=cf),
                "Direct": lambda: ForecasterDirect(
                    estimator=C.make_estimator(est_name), lags=3, steps=3,
                    categorical_features=cf, n_jobs=1),
            }
            for name, mk in makers.items():
                res = {}
                f, _ = run(mk)
                if isinstance(f, tuple):
                    results[(name,) + base_key] = {"init": f}
                    continue
                res["fit"], res["fit_warn"] = run(lambda: f.fit(y=y, exog=exog, store_in_sample_residuals=True))
                if res["fit"] is None:
                    res["pred"], _ = run(lambda: f.predict(steps=3, exog=exog_fut).to_numpy())
                    res["interval"], _ = run(lambda: f.predict_interval(steps=3, exog=exog_fut, method="conformal", interval=0.8).to_numpy())
                f2, _ = run(mk)
                res["bt"], _ = run(lambda: backtesting_forecaster(
                    forecaster=f2, y=y, exog=exog,
                    cv=TimeSeriesFold(steps=3, initial_train_size=60, refit=False),
                    metric="mean_absolute_error", n_jobs=1, show_progress=False)[0].to_numpy())
                f3, _ = run(mk)
                res["osa"], _ = run(lambda: grid_search_forecaster(
                    forecaster=f3, y=y, exog=exog, cv=OneStepAheadFold(initial_train_size=60),
                    param_grid={}, lags_grid=[3, 4], metric="mean_absolute_error",
                    return_best=False, n_jobs=1, show_progress=False, verbose=False
                )["mean_absolute_error"].to_numpy())
                results[(name,) + base_key] = res

            # DirectMultiVariate
            res = {}
            mk = lambda: ForecasterDirectMultiVariate(
                estimator=C.make_estimator(est_name), level="s_a", lags=3, steps=3,
                categorical_features=cf, n_jobs=1)
            f, _ = run(mk)
            if isinstance(f, tuple):
                results[("DMV",) + base_key] = {"init": f}
            else:
                res["fit"], res["fit_warn"] = run(lambda: f.fit(series=series, exog=exog, store_in_sample_residuals=True))
                if res["fit"] is None:
                    res["pred"], _ = run(lambda: f.predict(steps=3, exog=exog_fut)["pred"].to_numpy())
                f2, _ = run(mk)
                res["bt"], _ = run(lambda: backtesting_forecaster_multiseries(
                    forecaster=f2, series=series, exog=exog,
                    cv=TimeSeriesFold(steps=3, initial_train_size=60, refit=False),
                    metric="mean_absolute_error", n_jobs=1, show_progress=False)[0].select_dtypes("number").to_numpy())
                f3, _ = run(mk)
                res["osa"], _ = run(lambda: grid_search_forecaster_multiseries(
                    forecaster=f3, series=series, exog=exog, cv=OneStepAheadFold(initial_train_size=60),
                    param_grid={}, lags_grid=[3, 4], metric="mean_absolute_error",
                    return_best=False, n_jobs=1, show_progress=False, verbose=False
                )["mean_absolute_error"].to_numpy())
                results[("DMV",) + base_key] = res
    print(est_name, len(results), flush=True)

# Classifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
for cname, mkc in {"logreg": lambda: LogisticRegression(max_iter=200),
                   "lgbmc": lambda: LGBMClassifier(n_estimators=10, verbose=-1, n_jobs=1, random_state=1)}.items():
    for dtype_name in ["float", "cat_int_vals", "mixed"]:
        series, exog, exog_fut, cat_like, cat_cols = C.make_data(dtype_name)
        y = pd.Series(pd.qcut(series["s_a"], 3, labels=False).to_numpy(), index=series.index, name="y")
        for cf_mode in ["auto", "none", "full"]:
            cf = C.catfeat_value(cf_mode, cat_like, list(exog.columns))
            res = {}
            mk = lambda: ForecasterRecursiveClassifier(estimator=mkc(), lags=3, categorical_features=cf)
            f, _ = run(mk)
            if isinstance(f, tuple):
                results[("Classifier", cname, dtype_name, cf_mode)] = {"init": f}
                continue
            res["fit"], res["fit_warn"] = run(lambda: f.fit(y=y, exog=exog))
            if res["fit"] is None:
                res["pred"], _ = run(lambda: f.predict(steps=3, exog=exog_fut).to_numpy())
                res["proba"], _ = run(lambda: f.predict_proba(steps=3, exog=exog_fut).to_numpy())
            results[("Classifier", cname, dtype_name, cf_mode)] = res

with open(sys.argv[1], "wb") as fh:
    pickle.dump(results, fh)
print("done", len(results))
