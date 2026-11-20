import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
n=100; idx=pd.date_range("2000",periods=n,freq="D"); y=pd.Series(np.random.default_rng(0).normal(size=n),index=idx)
ex=pd.DataFrame({"temp":np.arange(n,dtype=float)},index=idx)
pipe=lambda: make_pipeline(ColumnTransformer([("s",StandardScaler(),["temp"])],remainder="passthrough"),Ridge())
for name,mk,kw in [("Rec",lambda:ForecasterRecursive(pipe(),lags=2),dict(y=y,exog=ex)),
                   ("MS",lambda:ForecasterRecursiveMultiSeries(pipe(),lags=2),dict(series={"a":y,"b":y},exog=ex))]:
    f=mk()
    try:
        f.fit(**kw); print(name,"fit ok; predict:", end=" ")
        try: print(f.predict(2, exog=ex.iloc[:2].set_axis(pd.date_range(idx[-1]+idx.freq,periods=2,freq="D"))).shape)
        except Exception as e: print("ERR",type(e).__name__,str(e)[:120])
    except Exception as e: print(name,"fit ERR",type(e).__name__,str(e)[:120])
