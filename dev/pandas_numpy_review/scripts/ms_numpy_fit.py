"""Does ForecasterRecursiveMultiSeries give the same model if the estimator gets numpy instead of the DataFrame?"""
import warnings, time; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
import skforecast.recursive._forecaster_recursive_multiseries as m
n=2000; k=50; idx=pd.date_range("2000",periods=n,freq="h"); rng=np.random.default_rng(0)
DOW=np.array(["mon","tue","wed","thu","fri","sat","sun"])
series={f"s{j:03d}":pd.Series(np.cumsum(rng.normal(size=n)),index=idx) for j in range(k)}
exog={s:pd.DataFrame({"t":rng.normal(size=n),"h":idx.hour.to_numpy(),"b":idx.dayofweek>=5,
      "dow":pd.Categorical(DOW[idx.dayofweek.to_numpy()],categories=list(DOW))},index=idx) for s in series}
orig = m.cast_catboost_categorical_columns_dataframe
o_copy, o_check = m._copy_rows_to_check, m._check_in_place_fit
m._copy_rows_to_check = lambda X: o_copy(X) if hasattr(X, 'iloc') else None
m._check_in_place_fit = lambda X, X_rows: o_check(X=X, X_rows=X_rows) if hasattr(X, 'iloc') else None
def run(est, enc, to_numpy):
    m.cast_catboost_categorical_columns_dataframe = (lambda X, **kw: X.to_numpy(dtype=float)) if to_numpy else orig
    f=ForecasterRecursiveMultiSeries(est, lags=24, encoding=enc)
    t=time.perf_counter(); f.fit(series=series, exog=exog, store_in_sample_residuals=False); dt=time.perf_counter()-t
    ex_p={s:v.iloc[:12].set_axis(pd.date_range(idx[-1]+idx.freq,periods=12,freq="h")) for s,v in exog.items()}
    return f.predict(12, exog=ex_p)["pred"].to_numpy(), dt
for name,mk in [("Ridge",lambda:Ridge()),("LGBM",lambda:LGBMRegressor(n_estimators=50,verbose=-1,n_jobs=1,random_state=0)),
                ("HGB",lambda:HistGradientBoostingRegressor(max_iter=30,random_state=0))]:
    for enc in ("ordinal","ordinal_category","onehot"):
        a,ta=run(mk(),enc,False); b,tb=run(mk(),enc,True)
        print(f"{name:5s} {enc:16s} identical={np.array_equal(a,b)} fit DataFrame {ta*1000:6.0f} ms  numpy {tb*1000:6.0f} ms")
m.cast_catboost_categorical_columns_dataframe = orig
