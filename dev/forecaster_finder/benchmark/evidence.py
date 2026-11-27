"""Evidence counts behind each rule of finder_tree.decide()."""
import glob
from pathlib import Path

RES = Path(__file__).parent / "results"
import pandas as pd

df = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "*.csv")) if "classif" not in f], ignore_index=True)
naive = df.loc[df.model == "Seasonal naive"].set_index(["case", "history_obs"])["MASE"]
df["rel"] = df["MASE"] / naive.reindex(pd.MultiIndex.from_frame(df[["case", "history_obs"]])).to_numpy()

def pick(d, names):
    for n in names:
        r = d.loc[d.model == n]
        if len(r):
            return r.iloc[0]

rows = []
for (case, h), d in df.groupby(["case", "history_obs"]):
    fm = pick(d, ["Chronos-2", "Chronos-2 (no exog)"])
    ml = pick(d, ["LightGBM", "LightGBM (no exog)"])
    mlb = d.loc[d.family == "ML"].sort_values("rel").iloc[0]
    st = d.loc[d.family == "Statistical"].sort_values("rel")
    rows.append(dict(case=case, h=h, n=d.n_series.iloc[0], fm=fm.rel, lgbm=ml.rel,
                     ml_best=mlb.rel, ml_best_name=mlb.model,
                     stats=st.iloc[0].rel if len(st) else float("nan"),
                     t_fm=fm.seconds, t_lgbm=ml.seconds))
a = pd.DataFrame(rows)
a["full"] = a.groupby("case")["h"].transform("max") == a["h"]
pd.set_option("display.width", 200)
print(a.round(3).to_string(index=False))
for name, sel in [("single, full history", (a.n == 1) & a.full), ("single, all histories", a.n == 1),
                  ("several, full history", (a.n > 1) & a.full), ("several, all histories", a.n > 1)]:
    s = a[sel]
    print(f"\n{name}: {len(s)} rows | FM < LightGBM default: {(s.fm < s.lgbm).sum()} | FM < best ML variant: {(s.fm < s.ml_best).sum()}"
          f" | FM beats naive: {(s.fm < 1).sum()} | LightGBM default beats naive: {(s.lgbm < 1).sum()} | best ML beats naive: {(s.ml_best < 1).sum()}")
    print(f"   median rel: FM {s.fm.median():.3f}, LightGBM {s.lgbm.median():.3f}, best ML {s.ml_best.median():.3f}; median seconds FM {s.t_fm.median():.1f}, LightGBM {s.t_lgbm.median():.1f}")
low = a[(a.n == 1) & a.stats.notna() & a.case.str.contains("monthly|weekly")]
print(f"\nmonthly/weekly single: stats < FM in {(low.stats < low.fm).sum()} of {len(low)}; stats < LightGBM default in {(low.stats < low.lgbm).sum()}")
