"""
BUG-search-short-series, experiment 2: 5 long and 15 short series (new
products), Ridge, lags 3/12/24. Compares how the candidates are ranked
today and with option a (exclude), B (common set) and d (naive fallback).
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.recursive import ForecasterRecursiveMultiSeries

warnings.simplefilter("ignore")
rng = np.random.default_rng(42)
idx = pd.date_range("2016-01-01", periods=96, freq="MS")
t = np.arange(96)

def make(start, level, noise):
    """Monthly seasonal series (period 12) that starts at `start`."""
    y = level * (1 + 0.15 * np.sin(2 * np.pi * t / 12)) + rng.normal(0, noise * level, 96)
    return pd.Series(y, index=idx).loc[start:]

series = {}
# 5 long series: full history since 2016
for i in range(5):
    series[f"long_{i}"] = make("2016-01-01", level=rng.uniform(50, 150), noise=0.05)
# 15 short series (new products): start between 2019-01 and 2021-01,
# noisier because they are new
starts = pd.date_range("2019-01-01", "2021-01-01", freq="MS")
# NOTE: avoids exactly 12 or 24 obs in the train set (ISSUE-1410, fixed in #1418).
# Kept so that the numbers of the document reproduce.
starts = starts.drop([pd.Timestamp("2019-07-01"), pd.Timestamp("2020-07-01")])
for i in range(15):
    series[f"new_{i:02d}"] = make(rng.choice(starts), level=rng.uniform(50, 150), noise=0.15)

initial_train_end = pd.Timestamp("2021-06-01")
cv_kwargs = dict(steps=12, initial_train_size="2021-06-01", refit=True)

# ---- 1. What the series look like ------------------------------------------
print("Series (one char = 2 months; '.' no data, '#' train of fold 0, '+' test)")
print(f"{'':10s} 2016      2017      2018      2019      2020      2021      2022      2023")
for name, s in list(series.items())[:5] + list(series.items())[5:12]:
    row = ""
    for d in idx[::2]:
        if d < s.index[0]:
            row += "."
        elif d <= initial_train_end:
            row += "#"
        else:
            row += "+"
    n_train = (s.index <= initial_train_end).sum()
    print(f"{name:10s} {row}  train fold 0: {n_train:3d} obs")
print("  ... (8 more new_* series)\n")

# ---- 2. Backtesting of each candidate --------------------------------------
candidates = {"lags_3": 3, "lags_12": 12, "lags_24": 24}
preds = {}
for name, lags in candidates.items():
    f = ForecasterRecursiveMultiSeries(Ridge(), lags=lags, encoding="ordinal")
    cv = TimeSeriesFold(**cv_kwargs)
    _, p = backtesting_forecaster_multiseries(
        f, series, cv, "mean_absolute_error", n_jobs=1, show_progress=False
    )
    preds[name] = p

# Evaluation set defined by the data: (level, date) with an observed value in
# the test set and some history before the fold (otherwise nobody can predict).
folds = TimeSeriesFold(**cv_kwargs, window_size=1).split(
    X=pd.DataFrame(series).reindex(idx), as_pandas=True
)
rows = []
for _, fold in folds.iterrows():
    test_dates = idx[fold["test_start"]:fold["test_end"]]
    fold_start = test_dates[0]
    for level, s in series.items():
        if (s.index < fold_start).sum() == 0:
            continue
        last_value = s.loc[:fold_start - pd.offsets.MonthBegin(1)].iloc[-1]
        for d in test_dates:
            if d in s.index and not np.isnan(s[d]):
                rows.append((level, fold["fold"], d, s[d], last_value))
E = pd.DataFrame(rows, columns=["level", "fold", "date", "y", "naive"])

def per_level_mae_avg(df, col):
    return (df[col] - df["y"]).abs().groupby(df["level"]).mean().mean()

results = []
covered_sets = {}
for name, p in preds.items():
    p = p.reset_index(names="date")[["date", "level", "fold", "pred"]]
    df = E.merge(p, on=["level", "fold", "date"], how="left")
    covered = df["pred"].notna()
    covered_sets[name] = set(df.loc[covered, ["level", "date", "fold"]].itertuples(index=False))
    n_levels = df.loc[covered, "level"].nunique()
    df["pred_fallback"] = df["pred"].fillna(df["naive"])
    results.append({
        "candidate": name,
        "series evaluated": f"{n_levels}/{E['level'].nunique()}",
        "pairs covered": f"{covered.mean():.0%}",
        "today (search)": per_level_mae_avg(df[covered], "pred"),
        "d. with naive fallback": per_level_mae_avg(df, "pred_fallback"),
        "_df": df,
    })

common = set.intersection(*covered_sets.values())
for r in results:
    df = r.pop("_df")
    mask = [tuple(x) in common for x in df[["level", "date", "fold"]].itertuples(index=False)]
    r["B. common set"] = per_level_mae_avg(df[mask], "pred")
    r["a. excluded?"] = "no" if r["pairs covered"] == "100%" else "EXCLUDED"

res = pd.DataFrame(results).set_index("candidate")
print("Average MAE per series (lower is better) under each way of comparing:")
print(res.round(2).to_string())
print(f"\nCommon set: {len(common)} of {len(E)} test points "
      f"({len(common) / len(E):.0%}), "
      f"{len({c[0] for c in common})} series.")
