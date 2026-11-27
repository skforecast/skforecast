"""Searches with OneStepAheadFold: grid, random and bayesian.

For every grid candidate, the one-step-ahead metric is compared with a
backtesting with steps=1 and no refit. Results are pickled to compare trees.
Usage: searches.py <estimator> <out.pkl>
"""
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import skforecast
print(skforecast.__file__, flush=True)
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold, backtesting_forecaster_multiseries,
    grid_search_forecaster_multiseries, random_search_forecaster_multiseries,
    bayesian_search_forecaster_multiseries,
)
import common as C

est_name = sys.argv[1]
out_file = sys.argv[2]
GRID = {
    "ridge": ("alpha", [0.1, 10.0]),
    "lgbm": ("num_leaves", [4, 8]),
    "xgb": ("max_depth", [2, 3]),
    "hgb": ("max_depth", [2, 3]),
    "catboost": ("depth", [2, 3]),
}
pname, pvalues = GRID[est_name]
N_TRAIN = 60
results = {}


def run(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001
            out = C.summarize_exception(e)
    return out, sorted({(x.category.__name__, str(x.message)[:80]) for x in w})


def metric_cols(df):
    return [c for c in df.columns if c.startswith("mean_absolute_error")]


for as_dict in (False, True):
    for dtype_name, cf_modes in [("cat_int_vals", ["auto", "none", "incomplete"]),
                                 ("mixed", ["auto"]), ("float", ["auto"])]:
        if as_dict and dtype_name == "float":
            continue
        series, exog, exog_fut, cat_like, cat_cols = C.make_data(dtype_name, as_dict=as_dict)
        exog_cols = list(exog.columns) if not as_dict else list(exog["s_a"].columns)
        for cf_mode in cf_modes:
            cf = C.catfeat_value(cf_mode, cat_like, exog_cols)
            for enc in ["ordinal", "ordinal_category", "onehot", None]:
                key = (est_name, dtype_name, cf_mode, enc, as_dict)
                res = {}

                def new(lags=3, **params):
                    est = C.make_estimator(est_name)
                    est.set_params(**params)
                    return ForecasterRecursiveMultiSeries(
                        estimator=est, lags=lags, encoding=enc,
                        categorical_features=cf,
                    )

                cv = OneStepAheadFold(initial_train_size=N_TRAIN)
                f = new()
                grid, wg = run(lambda: grid_search_forecaster_multiseries(
                    forecaster=f, series=series, exog=exog, cv=cv,
                    param_grid={pname: pvalues}, lags_grid=[3, 5],
                    metric="mean_absolute_error",
                    aggregate_metric=["weighted_average", "pooling"],
                    return_best=True, n_jobs=1, show_progress=False, verbose=False,
                ))
                res["grid"] = grid
                res["grid_warn"] = wg
                if isinstance(grid, pd.DataFrame):
                    res["best_pred"], _ = run(
                        lambda: f.predict(steps=5, exog=exog_fut)["pred"].to_numpy()
                    )
                    diffs = []
                    for _, row in grid.iterrows():
                        fb = new(lags=list(row["lags"]), **row["params"])
                        bt, _ = run(lambda: backtesting_forecaster_multiseries(
                            forecaster=fb, series=series, exog=exog,
                            cv=TimeSeriesFold(steps=1, initial_train_size=N_TRAIN, refit=False),
                            metric="mean_absolute_error", n_jobs=1, show_progress=False,
                        )[0].set_index("levels")["mean_absolute_error"])
                        if isinstance(bt, tuple):
                            diffs.append(bt)
                        else:
                            diffs.append(max(
                                abs(bt["weighted_average"] - row["mean_absolute_error__weighted_average"]),
                                abs(bt["pooling"] - row["mean_absolute_error__pooling"]),
                            ))
                    res["grid_vs_bt"] = diffs

                f = new()
                res["random"], res["random_warn"] = run(lambda: random_search_forecaster_multiseries(
                    forecaster=f, series=series, exog=exog, cv=cv,
                    param_distributions={pname: pvalues}, lags_grid=[3, 5], n_iter=2,
                    random_state=7, metric="mean_absolute_error",
                    aggregate_metric="weighted_average",
                    return_best=False, n_jobs=1, show_progress=False, verbose=False,
                ))

                def space(trial):
                    return {
                        "lags": trial.suggest_categorical("lags", [3, 5]),
                        pname: trial.suggest_categorical(pname, pvalues),
                    }

                f = new()
                bay, wb = run(lambda: bayesian_search_forecaster_multiseries(
                    forecaster=f, series=series, exog=exog, cv=cv,
                    search_space=space, n_trials=5, random_state=7,
                    metric="mean_absolute_error", aggregate_metric="weighted_average",
                    return_best=True, n_jobs=1, show_progress=False, verbose=False,
                )[0])
                res["bayes"] = bay
                res["bayes_warn"] = wb
                if isinstance(bay, pd.DataFrame):
                    res["bayes_best_pred"], _ = run(
                        lambda: f.predict(steps=5, exog=exog_fut)["pred"].to_numpy()
                    )
                    res["est_params_after"] = {
                        k: v for k, v in f.estimator.get_params().items()
                        if k in ("feature_types", "enable_categorical",
                                 "categorical_features", "categorical_feature",
                                 "cat_features")
                    }
                results[key] = res
                d = res.get("grid_vs_bt")
                print(key, "grid", "ERR" if isinstance(grid, tuple) else len(grid),
                      "grid_vs_bt", d if d is None else [x if isinstance(x, tuple) else float(f"{x:.2e}") for x in d],
                      "bayes", "ERR" if isinstance(bay, tuple) else len(bay), flush=True)

# ForecasterDirectMultiVariate (wide series only)
series, exog, exog_fut, cat_like, cat_cols = C.make_data("cat_int_vals")
for cf_mode in ["auto", "none"]:
    cf = C.catfeat_value(cf_mode, cat_like, list(exog.columns))
    key = (est_name, "DMV", cf_mode)
    res = {}

    def new_dmv(lags=3, **params):
        est = C.make_estimator(est_name)
        est.set_params(**params)
        return ForecasterDirectMultiVariate(
            estimator=est, level="s_a", steps=2, lags=lags,
            categorical_features=cf, n_jobs=1,
        )

    cv = OneStepAheadFold(initial_train_size=N_TRAIN)
    f = new_dmv()
    res["grid"], res["grid_warn"] = run(lambda: grid_search_forecaster_multiseries(
        forecaster=f, series=series, exog=exog, cv=cv,
        param_grid={pname: pvalues}, lags_grid=[3, 5],
        metric="mean_absolute_error",
        return_best=True, n_jobs=1, show_progress=False, verbose=False,
    ))
    if isinstance(res["grid"], pd.DataFrame):
        res["best_pred"], _ = run(lambda: f.predict(steps=2, exog=exog_fut)["pred"].to_numpy())

    def space(trial):
        return {
            "lags": trial.suggest_categorical("lags", [3, 5]),
            pname: trial.suggest_categorical(pname, pvalues),
        }

    f = new_dmv()
    res["bayes"], res["bayes_warn"] = run(lambda: bayesian_search_forecaster_multiseries(
        forecaster=f, series=series, exog=exog, cv=cv,
        search_space=space, n_trials=4, random_state=7,
        metric="mean_absolute_error",
        return_best=False, n_jobs=1, show_progress=False, verbose=False,
    )[0])
    results[key] = res
    print(key, "grid", "ERR" if isinstance(res["grid"], tuple) else len(res["grid"]),
          "bayes", "ERR" if isinstance(res["bayes"], tuple) else len(res["bayes"]), flush=True)

with open(out_file, "wb") as fh:
    pickle.dump(results, fh)
print("done", len(results))
