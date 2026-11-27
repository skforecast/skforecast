"""Summarize the benchmark results by question of the decision tree."""

import json
import os
import sys
from pathlib import Path

import pandas as pd

RES = Path(__file__).parent / os.environ.get("BENCH_OUT", "results")
frames = [pd.read_csv(f) for f in sorted(RES.glob("*.csv")) if "classification" not in f.name]
df = pd.concat(frames, ignore_index=True)

# Best model of each family per case and history
df["family_best"] = df.groupby(["case", "history_obs", "family"])["MASE"].transform("min")
best = (
    df.loc[df["MASE"] == df["family_best"]]
    .drop_duplicates(["case", "history_obs", "family"])
    .pivot_table(index=["case", "history_obs"], columns="family", values="MASE")
)
best_model = (
    df.loc[df["MASE"] == df["family_best"]]
    .drop_duplicates(["case", "history_obs", "family"])
    .pivot_table(index=["case", "history_obs"], columns="family", values="model", aggfunc="first")
)
naive = best["Baseline"]
rel = best.div(naive, axis=0).round(3)  # < 1 beats the seasonal naive
rel["winner"] = best.drop(columns="Baseline").idxmin(axis=1)
rel["winner_overall"] = best.idxmin(axis=1)

pd.set_option("display.width", 220)
print("MASE relative to the seasonal naive (best configuration of each family):")
print(rel.to_string())
print("\nBest model of each family:")
print(best_model.to_string())

# Fixed default configurations (what the snippet would show)
fixed = df.pivot_table(index=["case", "history_obs"], columns="model", values="MASE")
fixed = fixed.div(fixed["Seasonal naive"], axis=0).round(3)
print("\nEvery model, MASE relative to the seasonal naive:")
print(fixed.to_string())

secs = df.pivot_table(index=["case", "history_obs"], columns="model", values="seconds").round(2)
print("\nSeconds of the whole backtest:")
print(secs.to_string())

out = {
    "relative_best_family": rel.reset_index().to_dict(orient="records"),
    "best_model_family": best_model.reset_index().to_dict(orient="records"),
    "relative_all": fixed.reset_index().to_dict(orient="records"),
    "seconds": secs.reset_index().to_dict(orient="records"),
    "raw": df.drop(columns="family_best").to_dict(orient="records"),
}
clf = RES / "vic_electricity_classification.csv"
if clf.exists():
    out["classification"] = pd.read_csv(clf).to_dict(orient="records")
    print("\nClassification:")
    print(pd.read_csv(clf).round(3).to_string(index=False))
(RES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
