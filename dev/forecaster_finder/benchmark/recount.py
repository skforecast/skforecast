"""
Recount of the review: the claims of the study under each protocol, by run
and by dataset, from the CSV files written by review.py (results/review/).

Usage:
    python -I recount.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results" / "review"
DATASET = {
    "vic_electricity_daily": "vic_electricity", "vic_electricity_daily_past_only": "vic_electricity",
    "bike_sharing_daily": "bike_sharing", "bike_sharing_hourly": "bike_sharing",
    "bike_sharing_hourly_past_only": "bike_sharing", "public_transport_madrid": "public_transport_madrid",
    "bicimad_weekly": "bicimad", "h2o_monthly": "h2o", "fuel_gasoline_monthly": "fuel_consumption",
    "expenditures_australia_monthly": "expenditures_australia", "ett_hourly_past_only": "ett_m2",
    "store_sales_50_items": "store_sales", "bdg2_buildings_energy": "bdg2_daily",
    "m4_hourly_50_series": "m4_hourly", "australia_tourism_quarterly": "australia_tourism",
}
LOW_FREQ = ["bicimad_weekly", "h2o_monthly", "fuel_gasoline_monthly",
            "expenditures_australia_monthly", "australia_tourism_quarterly"]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

df = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("*.csv")) if not f.name.startswith("extra")],
               ignore_index=True)
df["dataset"] = df["case"].map(DATASET)
df["full"] = df["history"] == "full"
wide = df.pivot_table(index=["dataset", "case", "history_obs", "full", "n_series"],
                      columns="model", values="rel").reset_index()
wide["past"] = wide["case"].str.contains("past_only")


def col(name, alt=None):
    """Column `name`, filled with `alt` in the cases where it does not exist."""
    out = wide[name] if name in wide else pd.Series(np.nan, index=wide.index)
    for a in ([alt] if isinstance(alt, str) else (alt or [])):
        if a in wide:
            out = out.fillna(wide[a])
    return out


ML_FROZEN = ["LightGBM", "Ridge", "LightGBM (diff)", "Ridge (diff)", "LightGBM (no exog)",
             "LightGBM drivers (DirectMultiVariate)"]
ML_REFIT = [f"{m} [refit]" for m in ML_FROZEN]
a = pd.DataFrame({
    "dataset": wide["dataset"], "case": wide["case"], "h": wide["history_obs"],
    "full": wide["full"], "n": wide["n_series"],
    "fm": col("Chronos-2", "Chronos-2 (no exog)"),
    "fm_base": col("Chronos-2 base", "Chronos-2 base (no exog)"),
    "fm_synth": col("Chronos-2 synth", "Chronos-2 synth (no exog)"),
    "lgbm_frozen": col("LightGBM", "LightGBM (no exog)"),
    "lgbm_refit": col("LightGBM [refit]", "LightGBM (no exog) [refit]"),
    "snippet_frozen": col("LightGBM snippet"),
    "snippet_refit": col("LightGBM snippet [refit]"),
    "strong": col("LightGBM+ [refit]"),
    "strong_diff": col("LightGBM+ (diff) [refit]"),
    # Without a validation period (short history), the default LightGBM is kept
    "val4": col("ML selected on validation (4)", ["LightGBM [refit]", "LightGBM (no exog) [refit]"]),
    "val6": col("ML selected on validation (6)", ["LightGBM [refit]", "LightGBM (no exog) [refit]"]),
    "oracle_frozen": wide[[c for c in ML_FROZEN if c in wide]].min(axis=1),
    "oracle_refit": wide[[c for c in ML_REFIT if c in wide]].min(axis=1),
    "stats": wide[[c for c in ["AutoETS", "AutoARIMA"] if c in wide]].min(axis=1),
    "ets": col("AutoETS"), "arima": col("AutoARIMA"),
})
a.round(3).to_csv(RES / "recount_runs.csv", index=False)
print(a.round(2).to_string(index=False))

ML_COLS = ["lgbm_frozen", "lgbm_refit", "snippet_frozen", "strong", "val4", "val6",
           "oracle_frozen", "oracle_refit"]


def count(sel, label):
    s = a[sel]
    rows = []
    for c in ML_COLS:
        ok = s[c].notna() & s["fm"].notna()
        rows.append({"ML protocol": c, "runs": int(ok.sum()),
                     "Chronos-2 < ML": int((s.fm[ok] < s[c][ok]).sum()),
                     "ML < naive": int((s[c][ok] < 1).sum()),
                     "median ML": s[c][ok].median(), "median Chronos-2": s.fm[ok].median()})
    print(f"\n--- {label}: {len(s)} runs, Chronos-2 beats the naive in {(s.fm < 1).sum()}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    return pd.DataFrame(rows).assign(scope=label)


out = [count(a.n == 1, "one series, all histories"),
       count((a.n == 1) & a.full, "one series, full history"),
       count(a.n > 1, "several series, all histories"),
       count((a.n > 1) & a.full, "several series, full history"),
       count(a.full, "all cases, full history")]

# By dataset: one vote per dataset (mean of the relative MASE of its runs)
print("\n--- By dataset (mean over the runs of the dataset; one vote each)")
by = a.groupby("dataset")[["fm", "fm_synth"] + ML_COLS + ["stats"]].mean()
by["runs"] = a.groupby("dataset").size()
print(by.round(2).to_string())
rows = []
for c in ML_COLS:
    rows.append({"ML protocol": c, "datasets": int(by[c].notna().sum()),
                 "Chronos-2 < ML": int((by.fm < by[c]).sum()),
                 "ML < naive": int((by[c] < 1).sum())})
print(pd.DataFrame(rows).to_string(index=False))
full = a[a.full].groupby("dataset")[["fm"] + ML_COLS].mean()
print("\nBy dataset, full history only:")
print(pd.DataFrame([{"ML protocol": c, "Chronos-2 < ML": int((full.fm < full[c]).sum()),
                     "ML < naive": int((full[c] < 1).sum()), "datasets": len(full)}
                    for c in ML_COLS]).to_string(index=False))
out.append(pd.DataFrame(rows).assign(scope="by dataset, all histories"))
pd.concat(out).round(3).to_csv(RES / "recount_summary.csv", index=False)

# Frequency rule: statistical models against Chronos-2 and ML in low frequency
low = a[a.case.isin(LOW_FREQ) & a.stats.notna()]
print("\n--- Low frequency (weekly, monthly, quarterly) with a statistical model")
print(low[["case", "h", "fm", "lgbm_frozen", "lgbm_refit", "strong", "val6", "ets", "arima"]]
      .round(2).to_string(index=False))
print(f"runs {len(low)}: stats < Chronos-2 in {(low.stats < low.fm).sum()}, "
      f"AutoETS < Chronos-2 in {(low.ets < low.fm).sum()}, "
      f"stats < LightGBM frozen in {(low.stats < low.lgbm_frozen).sum()}, "
      f"stats < LightGBM refit in {(low.stats < low.lgbm_refit).sum()}, "
      f"stats < LightGBM+ in {(low.stats < low.strong).sum()}")
lowd = low.groupby("dataset")[["fm", "stats", "ets", "lgbm_refit", "strong"]].mean()
print(lowd.round(2).to_string())
high = a[~a.case.isin(LOW_FREQ) & a.stats.notna()]
print(f"daily with stats, runs {len(high)}: stats < Chronos-2 in {(high.stats < high.fm).sum()}")

# Contamination control: Chronos-2 trained only on synthetic data
c = a[a.full & a.fm_synth.notna()]
print("\n--- Chronos-2 small / base / synth (full history)")
print(c[["case", "fm", "fm_base", "fm_synth", "lgbm_refit", "strong"]].round(3).to_string(index=False))
print(f"synth < small in {(c.fm_synth < c.fm).sum()} of {len(c)}; "
      f"synth < LightGBM refit in {(c.fm_synth < c.lgbm_refit).sum()}; "
      f"small < LightGBM refit in {(c.fm < c.lgbm_refit).sum()}; "
      f"synth < naive in {(c.fm_synth < 1).sum()}; "
      f"median small {c.fm.median():.3f}, base {c.fm_base.median():.3f}, synth {c.fm_synth.median():.3f}")

# Several series: other ways to aggregate (from the per-series errors)
print("\n--- Several series: aggregation across series (full history)")
for f in sorted((RES / "detail").glob("*.csv")):
    d = pd.read_csv(f)
    d = d[d.history == "full"]
    if d.level.nunique() < 2:
        continue
    p = d.pivot_table(index="level", columns="model", values="mae")
    scale = d.drop_duplicates("level").set_index("level")["scale"]
    rows = []
    for m in ["Chronos-2", "LightGBM", "LightGBM [refit]", "LightGBM+ [refit]",
              "LightGBM (diff) [refit]"]:
        if m not in p:
            continue
        r = p[m] / p["Seasonal naive"].replace(0, np.nan)
        rows.append({"model": m,
                     "mean MASE / mean MASE naive": (p[m] / scale).mean() / (p["Seasonal naive"] / scale).mean(),
                     "geometric mean of relative MAE": float(np.exp(np.log(r.dropna()).mean())),
                     "median relative MAE": r.median(),
                     "% series better than naive": 100 * (r < 1).mean(),
                     "% series better than Chronos-2": 100 * (p[m] < p["Chronos-2"]).mean()})
    print(f.stem)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

# One series: folds won by Chronos-2 against the refitted LightGBM
print("\n--- One series, full history: folds in which Chronos-2 has a lower MAE")
rows = []
for f in sorted((RES / "detail").glob("*.csv")):
    d = pd.read_csv(f)
    d = d[d.history == "full"]
    if d.level.nunique() > 1 or "fold" not in d:
        continue
    p = d.pivot_table(index="fold", columns="model", values="mae")
    fm = "Chronos-2" if "Chronos-2" in p else "Chronos-2 (no exog)"
    ml = "LightGBM [refit]" if "LightGBM [refit]" in p else "LightGBM (no exog) [refit]"
    rows.append({"case": f.stem, "folds": len(p), "Chronos-2 wins": int((p[fm] < p[ml]).sum())})
print(pd.DataFrame(rows).to_string(index=False))
