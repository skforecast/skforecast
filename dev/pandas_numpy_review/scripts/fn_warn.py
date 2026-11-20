import warnings, numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
n=100; idx=pd.date_range("2020",periods=n,freq="D")
s=pd.DataFrame({"a":np.arange(n,dtype=float),"b":np.arange(n,dtype=float)*2},index=idx)
for est in (KNeighborsRegressor(), HistGradientBoostingRegressor(max_iter=5)):
    f=ForecasterRecursiveMultiSeries(est,lags=3); f.fit(s)
    print(type(est).__name__, "feature_names_in_:", hasattr(f.estimator,"feature_names_in_"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always"); f.predict(3)
    print("  warnings:", [str(x.message)[:90] for x in w])
