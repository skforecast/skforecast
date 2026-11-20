"""
STUDY-osa-search-cost: where the time of the hyperparameter search of py59
goes (`bayesian_search_forecaster_multiseries` with `OneStepAheadFold`, the
article's search space, `random_state=123`, `return_best=True`), split into:

- split: `_train_test_split_one_step_ahead_search` (training and test matrices
  of skforecast, once per distinct `lags`, cached by the search);
- evaluation of each trial: `estimator.fit`, `estimator.predict`, and the rest
  (predictions grouped by level, inverse transform, metrics);
- final fit (`return_best`): `estimator.fit` and the rest of
  `ForecasterRecursiveMultiSeries.fit` (mainly `_create_train_X_y`);
- other: optuna, results table.

Usage: REAL_DATA_DIR=... [REAL_N_SERIES=300] [LGBM_N_JOBS=1] [OSA_TRIALS=10] python time_real_osa_search.py
Run it alone on the machine.
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import os
import time
import warnings

from lightgbm import LGBMRegressor

import skforecast.model_selection._search as search_module
from real_data import LGBM_N_JOBS, _bdg2_data
from skforecast.model_selection import OneStepAheadFold, bayesian_search_forecaster_multiseries
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
N_TRIALS = int(os.environ.get("OSA_TRIALS", "10"))
T = dict.fromkeys(
    ("split", "eval", "eval_est_fit", "eval_est_predict", "final_fit", "final_est_fit",
     "final_est_predict"), 0.0
)
N = {"split": 0, "eval": 0}
PHASE = [None]


def timed(fn, key, phase=None):
    def wrapper(*args, **kwargs):
        previous = PHASE[0]
        if phase is not None:
            PHASE[0] = phase
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            T[key] += time.perf_counter() - t
            if key in N:
                N[key] += 1
            PHASE[0] = previous
    return wrapper


search_module._train_test_split_one_step_ahead_search = timed(
    search_module._train_test_split_one_step_ahead_search, "split", "split")
search_module._predict_and_calculate_metrics_one_step_ahead_multiseries = timed(
    search_module._predict_and_calculate_metrics_one_step_ahead_multiseries, "eval", "eval")
ForecasterRecursiveMultiSeries.fit = timed(ForecasterRecursiveMultiSeries.fit, "final_fit", "final")
_est_fit, _est_predict = LGBMRegressor.fit, LGBMRegressor.predict


def est_fit(self, *args, **kwargs):
    t = time.perf_counter()
    try:
        return _est_fit(self, *args, **kwargs)
    finally:
        if PHASE[0] == "eval":
            T["eval_est_fit"] += time.perf_counter() - t
        elif PHASE[0] == "final":
            T["final_est_fit"] += time.perf_counter() - t


def est_predict(self, *args, **kwargs):
    t = time.perf_counter()
    try:
        return _est_predict(self, *args, **kwargs)
    finally:
        if PHASE[0] == "eval":
            T["eval_est_predict"] += time.perf_counter() - t
        elif PHASE[0] == "final":
            T["final_est_predict"] += time.perf_counter() - t


LGBMRegressor.fit, LGBMRegressor.predict = est_fit, est_predict

end_validation = "2017-10-31 23:59:00"
series_dict, exog_dict = _bdg2_data()
series = {k: v.loc[:end_validation] for k, v in series_dict.items()}
exog = {k: v.loc[:end_validation] for k, v in exog_dict.items()}
forecaster = ForecasterRecursiveMultiSeries(
    estimator=LGBMRegressor(random_state=8520, verbose=-1, **LGBM_N_JOBS),
    lags=14,
    window_features=RollingFeatures(stats=["mean", "min", "max"], window_sizes=7),
    categorical_features="auto",
    encoding="ordinal",
)


def search_space(trial):
    return {
        "lags": trial.suggest_categorical("lags", [31, 62]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 7, 8]),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


t0 = time.perf_counter()
results, _ = bayesian_search_forecaster_multiseries(
    forecaster=forecaster, series=series, exog=exog,
    cv=OneStepAheadFold(initial_train_size="2017-08-31"), search_space=search_space,
    n_trials=N_TRIALS, random_state=123, metric="mean_absolute_error",
    suppress_warnings=True, show_progress=False,
)
total = time.perf_counter() - t0

eval_rest = T["eval"] - T["eval_est_fit"] - T["eval_est_predict"]
final_rest = T["final_fit"] - T["final_est_fit"] - T["final_est_predict"]
other = total - T["split"] - T["eval"] - T["final_fit"]
model = T["eval_est_fit"] + T["eval_est_predict"] + T["final_est_fit"] + T["final_est_predict"]
not_model = total - model
best = results.iloc[0]
print(f"bdg2 OneStepAheadFold search: {len(series)} series, {N_TRIALS} trials, LightGBM "
      f"(n_jobs={forecaster.estimator.get_params()['n_jobs']}); best lags {len(best['lags'])}, "
      f"MAE {best.filter(like='mean_absolute_error').iloc[0]:.2f}")
print(f"  total                                {total:8.2f} s")
rows = [
    (f"split ({N['split']} calls, skforecast matrices)", T["split"]),
    (f"trials: estimator.fit ({N['eval']} trials)", T["eval_est_fit"]),
    ("trials: estimator.predict", T["eval_est_predict"]),
    ("trials: rest (group by level, inverse, metrics)", eval_rest),
    ("final fit: estimator.fit", T["final_est_fit"]),
    ("final fit: estimator.predict (in-sample residuals)", T["final_est_predict"]),
    ("final fit: rest (create_train_X_y...)", final_rest),
    ("other (optuna, results table)", other),
]
for name, v in rows:
    print(f"  {name:<50}{v:8.2f} s ({v / total * 100:5.1f} %)")
print(f"  -> model (estimator fit + predict): {model:.2f} s ({model / total * 100:.1f} %); "
      f"not model: {not_model:.2f} s ({not_model / total * 100:.1f} %)")
print(" ", _env.versions())
