"""
Evidence behind each rule of the forecaster finder, with the extended set of
datasets (the 15 cases of the review plus the cases of cases_d4.py).

One vote per dataset in each type of data (the mean of its runs), with the
LightGBM that the finder snippet writes. It also compares AutoETS and
AutoARIMA as the statistical reference. Reads results/review/*.csv and
writes rules_runs.csv, rules_cells.csv and rules_stats.csv, which build.py
uses to check the sentences shown on the page.

Usage:
    python -I rules.py
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
FUTURE_EXOG = {"vic_electricity_daily": "weather", "bike_sharing_daily": "weather",
               "bike_sharing_hourly": "weather", "bdg2_buildings_energy": "weather"}
HOURLY = ["air_quality_8_series", "ett_m1_7_series"]
ONE_NONE = "one series, nothing known in advance"
ONE_KNOWN = "one series, known in advance"
SEVERAL = "several series"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)


def frequency(case):
    for key in ("hourly", "weekly", "monthly", "quarterly"):
        if key in case:
            return key
    return "hourly" if case in HOURLY else "daily"


frames = []
for case, dataset in DATASET.items():
    d = pd.read_csv(RES / f"{case}.csv")
    d["dataset"] = dataset
    d["exog_kind"] = FUTURE_EXOG.get(case, "")
    d["cell"] = np.where(d["n_series"] > 1, SEVERAL,
                         ONE_KNOWN if case in FUTURE_EXOG else ONE_NONE)
    d["source"] = "review"
    frames.append(d)
for path in sorted(RES.glob("d4_*.csv")):
    frames.append(pd.read_csv(path).assign(source="d4"))
df = pd.concat(frames, ignore_index=True)
df["exog_kind"] = df["exog_kind"].fillna("")
wide = df.pivot_table(index=["source", "cell", "dataset", "exog_kind", "case", "history_obs",
                             "n_series"], columns="model", values="rel").reset_index()


def col(*names):
    out = pd.Series(np.nan, index=wide.index)
    for name in names:
        if name in wide:
            out = out.fillna(wide[name])
    return out


a = wide[["source", "cell", "dataset", "exog_kind", "case", "history_obs", "n_series"]].copy()
a["freq"] = a["case"].map(frequency)
a["fm"] = col("Chronos-2", "Chronos-2 (no exog)")
a["lgbm"] = col("LightGBM [refit]", "LightGBM (no exog) [refit]")
a["strong"] = col("LightGBM+ [refit]")
# LightGBM as the snippet writes it (the cases with variables known only up to
# today have no snippet of their own: LightGBM without exog)
a["ml"] = col("LightGBM snippet v2 [refit]", "LightGBM (no exog) [refit]")
a["ets"] = col("AutoETS")
a["arima"] = col("AutoARIMA")
a["arima_exog"] = col("AutoARIMA (exog)")
a["tuned"] = col("LightGBM tuned [refit]")
a.round(3).to_csv(RES / "rules_runs.csv", index=False)

print(f"{len(a)} runs, {a.dataset.nunique()} datasets")
print(a[["cell", "dataset", "exog_kind", "case", "history_obs", "fm", "ml", "lgbm", "strong",
         "ets", "arima", "arima_exog"]].round(2).to_string(index=False))

# ==============================================================================
# By type of data: one vote per dataset
# ==============================================================================
cells = []
for cell, g in a.groupby("cell"):
    by = g.groupby("dataset")[["fm", "ml", "lgbm", "strong"]].mean()
    margin = (by.ml - by.fm) / by[["ml", "fm"]].min(axis=1)
    cells.append({
        "cell": cell, "runs": len(g), "datasets": len(by),
        "Chronos-2 ahead (datasets)": int((by.fm < by.ml).sum()),
        "ahead by more than 5%": int((margin > 0.05).sum()),
        "LightGBM ahead by more than 5%": int((margin < -0.05).sum()),
        "Chronos-2 ahead of study LightGBM": int((by.fm < by.lgbm).sum()),
        "Chronos-2 ahead of LightGBM+": f"{int((by.fm < by.strong).sum())} of {int(by.strong.notna().sum())}",
        "LightGBM >= naive (runs)": int((g.ml >= 1).sum()),
        "LightGBM >= naive (datasets)": int((by.ml >= 1).sum()),
        "Chronos-2 >= naive (runs)": int((g.fm >= 1).sum()),
        "Chronos-2 >= naive (datasets)": int((by.fm >= 1).sum()),
        "median Chronos-2": by.fm.median(), "median LightGBM": by.ml.median(),
    })
    print(f"\n--- {cell}: per dataset (mean of its runs)")
    print(by.assign(margin=margin).round(2).to_string())
cells = pd.DataFrame(cells).round(3)
print("\n--- By type of data")
print(cells.T.to_string())
cells.to_csv(RES / "rules_cells.csv", index=False)

known = a[a.cell == ONE_KNOWN]
for kind, g in known.groupby("exog_kind"):
    by = g.groupby("dataset")[["fm", "ml"]].mean()
    print(f"known in advance, {kind}: {len(by)} datasets, Chronos-2 ahead in {int((by.fm < by.ml).sum())}")

# ==============================================================================
# LightGBM with a small tuning budget (20 trials on validation), full history
# ==============================================================================
tuned = a[a.tuned.notna()]
rows = []
print("\n--- LightGBM tuned on validation against the untuned one and Chronos-2 (full history)")
for cell, g in tuned.groupby("cell"):
    by = g.groupby("dataset")[["fm", "lgbm", "tuned"]].mean()
    best = by[["lgbm", "tuned"]].min(axis=1)
    rows.append({
        "cell": cell, "datasets": len(by),
        "untuned < Chronos-2": int((by.lgbm < by.fm).sum()),
        "tuned < Chronos-2": int((by.tuned < by.fm).sum()),
        "tuned within 5% of Chronos-2 or better": int((by.tuned < 1.05 * by.fm).sum()),
        "tuning improved LightGBM": int((by.tuned < by.lgbm).sum()),
        "tuning made it worse by more than 5%": int((by.tuned > 1.05 * by.lgbm).sum()),
        "tuned >= naive": int((by.tuned >= 1).sum()),
        "untuned >= naive": int((by.lgbm >= 1).sum()),
        "median Chronos-2": by.fm.median(), "median untuned": by.lgbm.median(),
        "median tuned": by.tuned.median(),
    })
    print(f"\n{cell}")
    print(by.round(2).to_string())
tuned_cells = pd.DataFrame(rows).round(3)
print(tuned_cells.T.to_string())
tuned_cells.to_csv(RES / "rules_tuned.csv", index=False)

# ==============================================================================
# Statistical reference: AutoETS or AutoARIMA, and with which frequencies
# ==============================================================================
one = a[a.n_series == 1]
rows = []
for freq, g in one.groupby("freq"):
    for model in ("ets", "arima", "arima_exog"):
        ok = g[g[model].notna()]
        if ok.empty:
            continue
        by = ok.groupby("dataset")[["fm", "ml", model]].mean()
        rows.append({
            "freq": freq, "model": model, "runs": len(ok), "datasets": len(by),
            "< Chronos-2 (runs)": int((ok[model] < ok.fm).sum()),
            "< Chronos-2 (datasets)": int((by[model] < by.fm).sum()),
            "< LightGBM (datasets)": int((by[model] < by.ml).sum()),
            ">= naive (runs)": int((ok[model] >= 1).sum()),
            "median": ok[model].median(), "median Chronos-2": ok.fm.median(),
        })
stats = pd.DataFrame(rows).round(3)
print("\n--- Statistical models, one series")
print(stats.to_string(index=False))
stats.to_csv(RES / "rules_stats.csv", index=False)

both = one[one.ets.notna() & one.arima.notna()]
for freq, g in both.groupby("freq"):
    by = g.groupby("dataset")[["ets", "arima", "fm"]].mean()
    print(f"\n{freq}: AutoETS against AutoARIMA on the same runs ({len(g)} runs, {len(by)} datasets): "
          f"AutoETS lower in {int((g.ets < g.arima).sum())} runs and {int((by.ets < by.arima).sum())} datasets")
    print(by.round(2).to_string())

# Quarterly, one model per series (304 series)
detail = pd.read_csv(RES / "detail" / "australia_tourism_quarterly.csv")
p = detail.pivot_table(index="level", columns="model", values="mae")
print("\n--- australia_tourism, 304 quarterly series, one model per series")
for m in ["AutoETS", "AutoARIMA", "Chronos-2"]:
    if m in p:
        r = p[m] / p["Seasonal naive"]
        print(f"{m:<10} median relative error {r.median():.3f} | better than the naive in "
              f"{100 * (r < 1).mean():.0f}% | better than Chronos-2 in "
              f"{100 * (p[m] < p['Chronos-2']).mean():.0f}% | worst {r.max():.2f}")
