"""
Simulation of option c: a global recursive model that predicts series with
less history than its window by setting the missing lags to NaN, with an
estimator that supports NaN natively (HistGradientBoostingRegressor).

Variants (manual implementation, same folds, scaling and encoding):
  drop : today's rule, a series needs more than `lags` values in the train set
         of the fold and a full last window (validated against skforecast).
  c1   : rows with missing lags (NaN) are used to train AND to predict.
  c2   : train as today (complete rows only), predict with NaN lags.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
rng = np.random.default_rng(SEED)
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
names = list(series)
Y = pd.DataFrame(series).reindex(idx)          # NaN before each series starts
Yv = Y.to_numpy()

cv_kwargs = dict(steps=12, initial_train_size="2021-06-01", refit=True)
folds = TimeSeriesFold(**cv_kwargs, window_size=1).split(X=Y, as_pandas=True)

def est():
    return HistGradientBoostingRegressor(max_iter=200, random_state=123)

def fit_predict(lags, variant, fold):
    tr0, tr1 = fold["train_start"], fold["train_end"]
    te0, te1 = fold["test_start"], fold["test_end"]
    W = Yv[tr0:tr1]                               # train window (NaN = no data)
    mu = np.nanmean(W, axis=0)
    sd = np.nanstd(W, axis=0)
    sd = np.where((sd > 0) & ~np.isnan(sd), sd, 1.0)
    Z = (W - mu) / sd
    n_obs = (~np.isnan(W)).sum(axis=0)
    in_train = n_obs > lags                       # today's rule (#1418)

    X, y = [], []
    for j in range(len(names)):
        if variant in ("drop", "c2") and not in_train[j]:
            continue
        for p in range(len(Z)):
            if np.isnan(Z[p, j]):
                continue
            lag_vals = [Z[p - k, j] if p - k >= 0 else np.nan for k in range(1, lags + 1)]
            if variant in ("drop", "c2") and np.isnan(lag_vals).any():
                continue
            if variant == "c1" and np.isnan(lag_vals).all():
                continue                          # at least one observed lag
            X.append(lag_vals + [j])
            y.append(Z[p, j])
    model = est().fit(np.array(X), np.array(y))

    preds = {}
    for j in range(len(names)):
        if n_obs[j] == 0:
            continue
        if variant == "drop" and not in_train[j]:
            continue
        hist = list(Z[:, j])
        out = []
        for h in range(te1 - te0):
            lag_vals = [hist[-k] if len(hist) - k >= 0 else np.nan for k in range(1, lags + 1)]
            yhat = model.predict(np.array([lag_vals + [j]]))[0]
            hist.append(yhat)
            out.append(yhat * sd[j] + mu[j])
        preds[names[j]] = pd.Series(out, index=idx[te0:te1])
    return preds

# Evaluation set defined by the data: observed test value and some history
# before the fold.
E = []
for _, fold in folds.iterrows():
    te0, te1 = fold["test_start"], fold["test_end"]
    for j, name in enumerate(names):
        if (~np.isnan(Yv[fold["train_start"]:fold["train_end"], j])).sum() == 0:
            continue
        for p in range(te0, te1):
            if not np.isnan(Yv[p, j]):
                E.append((name, fold["fold"], idx[p], Yv[p, j]))
E = pd.DataFrame(E, columns=["level", "fold", "date", "y"])

def evaluate(pred_df):
    df = E.merge(pred_df, on=["level", "fold", "date"], how="left")
    cov = df["pred"].notna()
    mae = (df.loc[cov, "pred"] - df.loc[cov, "y"]).abs().groupby(df.loc[cov, "level"]).mean().mean()
    return df, cov.mean(), mae

rows, frames = [], {}
for lags in [3, 12, 24]:
    # skforecast real (today)
    f = ForecasterRecursiveMultiSeries(est(), lags=lags, encoding="ordinal",
                                       transformer_series=StandardScaler())
    _, p = backtesting_forecaster_multiseries(
        f, series, TimeSeriesFold(**cv_kwargs), "mean_absolute_error",
        n_jobs=1, show_progress=False)
    p = p.reset_index(names="date")[["date", "level", "fold", "pred"]]
    _, cov, mae = evaluate(p)
    rows.append(("skforecast today", lags, cov, mae))
    for variant in ["drop", "c1", "c2"]:
        recs = []
        for _, fold in folds.iterrows():
            for name, s in fit_predict(lags, variant, fold).items():
                recs += [(d, name, fold["fold"], v) for d, v in s.items()]
        pdf = pd.DataFrame(recs, columns=["date", "level", "fold", "pred"])
        df, cov, mae = evaluate(pdf)
        frames[(variant, lags)] = df
        rows.append((variant, lags, cov, mae))

res = pd.DataFrame(rows, columns=["method", "lags", "coverage", "MAE"])
table = res.pivot(index="method", columns="lags", values="MAE").round(2)
cover = res.pivot(index="method", columns="lags", values="coverage").map(lambda x: f"{x:.0%}")
order = ["skforecast today", "drop", "c1", "c2"]
print(f"seed={SEED}")
print("Average MAE per series (lower is better):")
print(table.loc[order].to_string())
print("\nCoverage of the evaluation set:")
print(cover.loc[order].to_string())

# Cold-start points: the ones that lags=24 cannot predict today
cold = frames[("drop", 24)]["pred"].isna()
print(f"\nCold-start points (not predictable with lags=24 today): {cold.sum()} of {len(cold)}")
naive_last = []
for _, r in E[cold.to_numpy()].iterrows():
    fold = folds.loc[folds["fold"] == r["fold"]].iloc[0]
    hist = Yv[fold["train_start"]:fold["train_end"], names.index(r["level"])]
    naive_last.append(hist[~np.isnan(hist)][-1])
naive_mae = np.mean(np.abs(np.array(naive_last) - E[cold.to_numpy()]["y"].to_numpy()))
print(f"  naive (last value) MAE on them: {naive_mae:.2f}")
for variant in ["c1", "c2"]:
    for lags in [3, 12, 24]:
        df = frames[(variant, lags)][cold.to_numpy()]
        print(f"  {variant} lags={lags:2d} MAE on them: {(df['pred'] - df['y']).abs().mean():.2f}")
