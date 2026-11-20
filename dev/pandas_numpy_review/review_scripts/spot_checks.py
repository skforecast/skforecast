"""Spot checks of two findings: HGB + categorical + bool fails in predict (P2), get_indexer vs OrdinalEncoder with None (P4)."""
import _env, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from skforecast.recursive import ForecasterRecursive
n=200; idx=pd.date_range("2020",periods=n+5,freq="D"); rng=np.random.default_rng(0)
y=pd.Series(rng.normal(size=n),index=idx[:n])
ex=pd.DataFrame({"t":rng.normal(size=n+5),"c":pd.Categorical(rng.choice(list("abc"),n+5)),"b":rng.random(n+5)<.5},index=idx)
f=ForecasterRecursive(HistGradientBoostingRegressor(max_iter=10),lags=3); f.fit(y,exog=ex.iloc[:n])
try: print("HGB + category + bool predict:", f.predict(5,exog=ex.iloc[n:]).round(2).tolist())
except Exception as e: print("HGB + category + bool predict:", type(e).__name__, str(e)[:90])
enc=OrdinalEncoder(dtype=float,handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan).set_output(transform="pandas")
s=pd.DataFrame({"c":["a",None,"b","a"]}); enc.fit(s)
cats=enc.categories_[0]; gi=pd.Index(cats).get_indexer(s["c"]).astype(float); gi[gi==-1]=np.nan
print("dtype",s["c"].dtype,"categories_",list(cats),"| OrdinalEncoder",enc.transform(s)["c"].tolist(),"| get_indexer",gi.tolist())
print(_env.versions())
