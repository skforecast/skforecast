"""Data of the report: chart rows and full tables, from results/."""
import glob
from pathlib import Path

RES = Path(__file__).parent / "results"
import json
import pandas as pd

LABEL = {
    "vic_electricity_daily": ("Electricity demand, daily", "1", "Known in advance"),
    "bike_sharing_daily": ("Bike rentals, daily", "1", "Known in advance"),
    "bike_sharing_hourly": ("Bike rentals, hourly", "1", "Known in advance"),
    "public_transport_madrid": ("Madrid public transport, daily", "1", "None"),
    "bicimad_weekly": ("BiciMAD users, weekly", "1", "None"),
    "h2o_monthly": ("Drug expenditure (h2o), monthly", "1", "None"),
    "fuel_gasoline_monthly": ("Gasoline consumption, monthly", "1", "None"),
    "expenditures_australia_monthly": ("Restaurant spending, monthly", "1", "None"),
    "vic_electricity_daily_past_only": ("Electricity demand, daily", "1", "Only up to today"),
    "bike_sharing_hourly_past_only": ("Bike rentals, hourly", "1", "Only up to today"),
    "ett_hourly_past_only": ("Transformer oil temp., hourly", "1", "Only up to today"),
    "store_sales_50_items": ("Store sales, daily", "50", "None"),
    "bdg2_buildings_energy": ("Building energy, daily", "40", "Known in advance"),
    "m4_hourly_50_series": ("M4 hourly", "50", "None"),
    "australia_tourism_quarterly": ("Australian tourism, quarterly", "304", "None"),
}
df = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "*.csv")) if "classif" not in f], ignore_index=True)
naive = df.loc[df.model == "Seasonal naive"].set_index(["case", "history_obs"])["MASE"]
df["rel"] = df["MASE"] / naive.reindex(pd.MultiIndex.from_frame(df[["case", "history_obs"]])).to_numpy()
full = df.loc[df.groupby("case")["history_obs"].transform("max") == df["history_obs"]]
chart = []
for case, d in full.groupby("case"):
    g = lambda names: next((float(d.loc[d.model == n, "rel"].iloc[0]) for n in names if (d.model == n).any()), None)
    st = d.loc[d.family == "Statistical"].sort_values("rel")
    lab, n, extra = LABEL[case]
    chart.append({"case": case, "label": lab, "n": n, "extra": extra, "history": int(d.history_obs.iloc[0]),
                  "fm": g(["Chronos-2", "Chronos-2 (no exog)"]), "ml": g(["LightGBM", "LightGBM (no exog)"]),
                  "ml_best": float(d.loc[d.family == "ML", "rel"].min()),
                  "ml_best_name": d.loc[d.family == "ML"].sort_values("rel").model.iloc[0],
                  "stats": float(st.rel.iloc[0]) if len(st) else None,
                  "stats_name": st.model.iloc[0] if len(st) else None,
                  "t_fm": float(d.loc[d.model.str.startswith("Chronos-2"), "seconds"].iloc[0]),
                  "t_ml": float(d.loc[d.model.isin(["LightGBM", "LightGBM (no exog)"]), "seconds"].iloc[0])})
order = list(LABEL)
chart.sort(key=lambda r: order.index(r["case"]))
table = df[["case", "history_obs", "model", "family", "MASE", "rel", "seconds"]].round(3)
table["label"] = table.case.map(lambda c: LABEL[c][0])
out = {"chart": chart, "table": table.to_dict(orient="records"),
       "classification": pd.read_csv(RES / "vic_electricity_classification.csv").round(3).to_dict(orient="records"),
       "env": json.load(open(RES / "environment.json"))}
json.dump(out, open(Path(__file__).parent / "report_data.json", "w"))
print(pd.DataFrame(chart)[["label", "n", "extra", "history", "fm", "ml", "ml_best", "ml_best_name", "stats", "t_fm", "t_ml"]].round(3).to_string(index=False))
