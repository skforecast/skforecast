"""
Experiment 2: grid_search_forecaster over `lags` with scaled and unscaled
metrics, TimeSeriesFold and OneStepAheadFold.

For each candidate the search returns MASE, MAE, RMSSE and MSE (one search,
all metrics), so the ranking by MASE and by MAE come from the same predictions.
MASE with a fixed denominator (full y_train) is MAE / constant, so it ranks
exactly as MAE: any difference between the MASE and MAE rankings is caused by
the denominator.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge

from skforecast.datasets import load_demo_dataset
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, OneStepAheadFold, grid_search_forecaster
from skforecast.metrics import mean_absolute_scaled_error, root_mean_squared_scaled_error

warnings.filterwarnings("ignore")
print(f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "
      f"scikit-learn {sklearn.__version__}")

LAGS_GRID = [1, 2, 3, 6, 12, 13, 24, 36, 48]
METRICS = [mean_absolute_scaled_error, "mean_absolute_error",
           root_mean_squared_scaled_error, "mean_squared_error"]


def search(y, initial_train_size, kind, exog=None):
    if kind == "TimeSeriesFold":
        cv = TimeSeriesFold(steps=12, initial_train_size=initial_train_size, refit=False)
    else:
        cv = OneStepAheadFold(initial_train_size=initial_train_size)
    res = grid_search_forecaster(
        forecaster=ForecasterRecursive(estimator=Ridge(alpha=1.0), lags=1),
        y=y, exog=exog, cv=cv, param_grid={"alpha": [1.0]}, lags_grid=LAGS_GRID,
        metric=METRICS, return_best=False, n_jobs=1, verbose=False, show_progress=False,
    )
    res["n_lags"] = res["lags"].apply(len)
    return res.set_index("n_lags")


def rank_summary(res):
    by_mase = res.sort_values("mean_absolute_scaled_error").index.to_list()
    by_mae = res.sort_values("mean_absolute_error").index.to_list()
    by_rmsse = res.sort_values("root_mean_squared_scaled_error").index.to_list()
    by_mse = res.sort_values("mean_squared_error").index.to_list()
    tau_mae = kendalltau(res["mean_absolute_scaled_error"], res["mean_absolute_error"]).statistic
    tau_mse = kendalltau(res["root_mean_squared_scaled_error"], res["mean_squared_error"]).statistic
    return by_mase, by_mae, by_rmsse, by_mse, tau_mae, tau_mse


def report(name, y, initial_train_size, exog=None):
    for kind in ["TimeSeriesFold", "OneStepAheadFold"]:
        res = search(y, initial_train_size, kind, exog)
        by_mase, by_mae, by_rmsse, by_mse, tau_mae, tau_mse = rank_summary(res)
        print(f"\n### {name}, {kind}")
        print(f"order by MASE : {by_mase}")
        print(f"order by MAE  : {by_mae}")
        print(f"order by RMSSE: {by_rmsse}")
        print(f"order by MSE  : {by_mse}")
        print(f"best MASE={by_mase[0]} vs MAE={by_mae[0]}; best RMSSE={by_rmsse[0]} vs MSE={by_mse[0]}; "
              f"Kendall tau MASE~MAE {tau_mae:.3f}, RMSSE~MSE {tau_mse:.3f}")
        y_full = y.iloc[:initial_train_size]
        scale_full = np.mean(np.abs(np.diff(y_full)))
        tab = res[["mean_absolute_error", "mean_absolute_scaled_error"]].copy()
        tab["MASE full y_train"] = tab["mean_absolute_error"] / scale_full
        tab["implied scale"] = tab["mean_absolute_error"] / tab["mean_absolute_scaled_error"]
        print(tab.sort_index().round(5).to_string())


def synthetic(seed, n=400, trend_var=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    e = np.zeros(n)
    sd = 1.0 + trend_var * t / n
    for i in range(1, n):
        e[i] = 0.7 * e[i - 1] + rng.normal(scale=sd[i])
    y = 10 + 3 * np.sin(2 * np.pi * t / 12) + e
    return pd.Series(y, index=pd.date_range("2000-01-01", periods=n, freq="MS"), name="y")


if __name__ == "__main__":
    y = load_demo_dataset(verbose=False)
    report("h2o (variability grows over time)", y, len(y) - 36)
    report("synthetic seasonal AR(1), constant variance, seed 0", synthetic(0), 300)

    # Systematic check: many seeds, constant vs growing vs decreasing variance.
    print("\n## Many seeds (n=400, initial_train_size=300, Ridge, lags grid", LAGS_GRID, ")")
    print("| series | fold | seeds | best MASE != best MAE | best RMSSE != best MSE | mean Kendall tau MASE~MAE | mean n_lags best MASE | mean n_lags best MAE |")
    print("|---|---|---|---|---|---|---|---|")
    for label, tv in [("constant variance", 0.0), ("variance grows x3", 2.0), ("variance shrinks x1/3", -0.667)]:
        for kind in ["TimeSeriesFold", "OneStepAheadFold"]:
            diff_mae = diff_mse = 0
            taus, best_mase, best_mae = [], [], []
            seeds = range(30)
            for s in seeds:
                res = search(synthetic(s, trend_var=tv), 300, kind)
                by_mase, by_mae, by_rmsse, by_mse, tau_mae, _ = rank_summary(res)
                diff_mae += by_mase[0] != by_mae[0]
                diff_mse += by_rmsse[0] != by_mse[0]
                taus.append(tau_mae); best_mase.append(by_mase[0]); best_mae.append(by_mae[0])
            print(f"| {label} | {kind} | {len(seeds)} | {diff_mae} | {diff_mse} | {np.mean(taus):.3f} | "
                  f"{np.mean(best_mase):.1f} | {np.mean(best_mae):.1f} |")
