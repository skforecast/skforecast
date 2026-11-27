"""T2: one-step-ahead searches vs an equivalent backtesting (steps=1, no refit).
Prints, for every config, the metric of grid/random/bayesian with
OneStepAheadFold and the MAE of backtesting_forecaster_multiseries."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
banner()
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import (
    OneStepAheadFold, TimeSeriesFold, grid_search_forecaster_multiseries,
    random_search_forecaster_multiseries, bayesian_search_forecaster_multiseries,
    backtesting_forecaster_multiseries)

ITS = 80
rows = []
for skind in ["df", "dict"]:
    for exname, cols in [("num", ["f", "i", "b"]), ("ci", ["f", "ci"])]:
        series, exog_train, _ = make_case(skind, cols, effect_col="ci" if "ci" in cols else None,
                                          exog_kind="df" if skind == "df" else "dict")
        for est in ["lgbm", "xgb", "cat", "hgb", "ridge"]:
            for enc in ["ordinal", "ordinal_category", "onehot"]:
                for cf in ["auto", None]:
                    def mk():
                        return ForecasterRecursiveMultiSeries(make_estimator(est), lags=3, encoding=enc,
                                                              categorical_features=cf)
                    out = {}
                    for name, fn in [
                        ("grid", lambda f: grid_search_forecaster_multiseries(
                            f, series, cv=OneStepAheadFold(initial_train_size=ITS), exog=exog_train,
                            param_grid={}, lags_grid=[3], metric="mean_absolute_error",
                            aggregate_metric="average", return_best=False, verbose=False, show_progress=False)),
                        ("bayes", lambda f: bayesian_search_forecaster_multiseries(
                            f, series, cv=OneStepAheadFold(initial_train_size=ITS), exog=exog_train,
                            search_space=lambda t: {"lags": t.suggest_categorical("lags", [3])},
                            metric="mean_absolute_error", aggregate_metric="average", n_trials=1,
                            return_best=False, verbose=False, show_progress=False)[0]),
                    ]:
                        f = mk()
                        st, r, w = run(lambda: fn(f))
                        out[name] = (float(r["mean_absolute_error__average"].iloc[0]) if len(r) else "EMPTY") if st == "ok" else r[:90]
                    f = mk()
                    st, r, w = run(lambda: backtesting_forecaster_multiseries(
                        f, series, cv=TimeSeriesFold(steps=1, initial_train_size=ITS, refit=False),
                        exog=exog_train, metric="mean_absolute_error",
                        show_progress=False, n_jobs=1, suppress_warnings=True))
                    out["bt"] = float(r[0].set_index("levels").loc["average", "mean_absolute_error"]) if st == "ok" else r[:90]
                    print(skind, exname, est, enc, cf, out, flush=True)

# ForecasterDirectMultiVariate: search one step ahead
series, exog_train, _ = make_case("df", ["f", "i", "b"], exog_kind="df")
for est in ["lgbm", "xgb", "ridge"]:
    f = ForecasterDirectMultiVariate(make_estimator(est), level="s0", steps=2, lags=3)
    st, r, w = run(lambda: grid_search_forecaster_multiseries(
        f, series, cv=OneStepAheadFold(initial_train_size=ITS), exog=exog_train,
        param_grid={}, lags_grid=[3, 4], metric="mean_absolute_error", return_best=False,
        verbose=False, show_progress=False))
    print("DMV", est, r["mean_absolute_error"].to_list() if st == "ok" else r)
    f = ForecasterDirectMultiVariate(make_estimator(est), level="s0", steps=2, lags=3)
    st, r, w = run(lambda: bayesian_search_forecaster_multiseries(
        f, series, cv=OneStepAheadFold(initial_train_size=ITS), exog=exog_train,
        search_space=lambda t: {"lags": t.suggest_categorical("lags", [3, 4])},
        metric="mean_absolute_error", n_trials=3, return_best=False,
        verbose=False, show_progress=False))
    print("DMV bayes", est, r[0]["mean_absolute_error"].to_list() if st == "ok" else r)
