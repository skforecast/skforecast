"""
Option c with several NaN-native estimators.

drop : today's rule (validated against skforecast in option_c.py).
c1   : train also with rows whose missing lags are before the TRUE start of
       the series (never rows cut by the training window), predict with NaN.
c2   : train as today, predict short series with NaN lags. A series not in the
       training set gets NaN encoding (unknown level, as skforecast does).
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skforecast.model_selection import TimeSeriesFold

warnings.simplefilter("ignore")

ESTIMATORS = {
    "HGBR": lambda: HistGradientBoostingRegressor(max_iter=200, random_state=123),
    "LightGBM": lambda: LGBMRegressor(n_estimators=200, random_state=123, verbose=-1),
    "XGBoost": lambda: XGBRegressor(n_estimators=200, random_state=123),
    "CatBoost": lambda: CatBoostRegressor(iterations=200, random_state=123, verbose=0),
}
LAGS = [12, 24]


def make_data(seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=96, freq="MS")
    t = np.arange(96)

    def make(start, level, noise):
        y = level * (1 + 0.15 * np.sin(2 * np.pi * t / 12)) + rng.normal(0, noise * level, 96)
        return pd.Series(y, index=idx).loc[start:]

    series = {}
    for i in range(5):
        series[f"long_{i}"] = make("2016-01-01", rng.uniform(50, 150), 0.05)
    starts = pd.date_range("2019-01-01", "2021-01-01", freq="MS")
    for i in range(15):
        series[f"new_{i:02d}"] = make(rng.choice(starts), rng.uniform(50, 150), 0.15)
    return idx, pd.DataFrame(series).reindex(idx)


def run(seed, est_name):
    idx, Y = make_data(seed)
    names = list(Y.columns)
    Yv = Y.to_numpy()
    first = np.array([np.argmax(~np.isnan(Yv[:, j])) for j in range(len(names))])
    folds = TimeSeriesFold(steps=12, initial_train_size="2021-06-01", refit=True,
                           window_size=1, verbose=False).split(X=Y, as_pandas=True)

    # Evaluation set: observed test value and some history before the fold
    E = []
    for _, f in folds.iterrows():
        for j, name in enumerate(names):
            if (~np.isnan(Yv[f["train_start"]:f["train_end"], j])).sum() == 0:
                continue
            for p in range(f["test_start"], f["test_end"]):
                if not np.isnan(Yv[p, j]):
                    hist = Yv[f["train_start"]:f["train_end"], j]
                    E.append((name, f["fold"], idx[p], Yv[p, j], hist[~np.isnan(hist)][-1]))
    E = pd.DataFrame(E, columns=["level", "fold", "date", "y", "naive"])

    out = {}
    for lags in LAGS:
        for variant in ["drop", "c1", "c2"]:
            recs, n_rows = [], 0
            for _, f in folds.iterrows():
                tr0, tr1, te0, te1 = f["train_start"], f["train_end"], f["test_start"], f["test_end"]
                W = Yv[tr0:tr1]
                mu, sd = np.nanmean(W, axis=0), np.nanstd(W, axis=0)
                sd = np.where((sd > 0) & ~np.isnan(sd), sd, 1.0)
                Z = (W - mu) / sd
                n_obs = (~np.isnan(W)).sum(axis=0)
                in_train = n_obs > lags
                X, y = [], []
                for j in range(len(names)):
                    s_rel = first[j] - tr0      # true start relative to the window
                    for p in range(len(Z)):
                        if np.isnan(Z[p, j]):
                            continue
                        complete = p - lags >= 0 and not np.isnan(Z[p - lags:p, j]).any()
                        if variant in ("drop", "c2"):
                            if not complete:
                                continue
                        else:
                            # c1: only NaN that are before the true start
                            if not complete and not (s_rel >= 0 and p > s_rel):
                                continue
                        lag_vals = [Z[p - k, j] if p - k >= 0 else np.nan
                                    for k in range(1, lags + 1)]
                        X.append(lag_vals + [j])
                        y.append(Z[p, j])
                n_rows += len(y)
                model = ESTIMATORS[est_name]().fit(np.array(X), np.array(y))
                known = set(int(r[-1]) for r in X)

                # Recursive prediction, all series at once per step
                pred_j = [j for j in range(len(names)) if n_obs[j] > 0
                          and (variant != "drop" or in_train[j])]
                hist = {j: list(Z[:, j]) for j in pred_j}
                for h in range(te1 - te0):
                    Xp = []
                    for j in pred_j:
                        lag_vals = [hist[j][-k] for k in range(1, lags + 1)]
                        enc = j if j in known else np.nan
                        Xp.append(lag_vals + [enc])
                    yhat = model.predict(np.array(Xp, dtype=float))
                    for j, v in zip(pred_j, yhat):
                        hist[j].append(v)
                        recs.append((idx[te0 + h], names[j], f["fold"], v * sd[j] + mu[j]))
            P = pd.DataFrame(recs, columns=["date", "level", "fold", "pred"])
            out[(variant, lags)] = (E.merge(P, on=["level", "fold", "date"], how="left"), n_rows)

    def mae(df, mask):
        d = df[mask]
        return (d["pred"] - d["y"]).abs().groupby(d["level"]).mean().mean()

    covered_today = out[("drop", 24)][0]["pred"].notna().to_numpy()
    cold = ~covered_today
    row = {"seed": seed, "estimator": est_name}
    for variant in ["drop", "c1", "c2"]:
        m = {lags: mae(out[(variant, lags)][0], out[(variant, lags)][0]["pred"].notna())
             for lags in LAGS}
        row[f"adv24_{variant}"] = m[12] - m[24]
    row["cold_naive"] = (E.loc[cold, "naive"] - E.loc[cold, "y"]).abs().mean()
    for variant in ["c1", "c2"]:
        d = out[(variant, 24)][0][cold]
        row[f"cold_{variant}"] = (d["pred"] - d["y"]).abs().mean()
    # Does c1 hurt the points that are predicted today?
    for variant in ["drop", "c1"]:
        d = out[(variant, 24)][0][covered_today]
        row[f"covered_{variant}"] = (d["pred"] - d["y"]).abs().mean()
    row["rows_c1/drop"] = out[("c1", 24)][1] / out[("drop", 24)][1]
    return row


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [42, 0, 1]
    rows = []
    for est_name in ESTIMATORS:
        for seed in seeds:
            t0 = time.time()
            rows.append(run(seed, est_name))
            print(f"done {est_name} seed={seed} ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    print("\nMean over seeds:")
    print(res.drop(columns="seed").groupby("estimator", sort=False).mean().round(2).to_string())
