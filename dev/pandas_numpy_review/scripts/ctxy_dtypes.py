import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect
from sklearn.linear_model import Ridge
n=50; idx=pd.date_range("2000",periods=n,freq="D"); y=pd.Series(np.arange(n,dtype=float),index=idx)
ex=pd.DataFrame({"t":np.arange(n,dtype=float),"h":np.arange(n),"b":np.arange(n)%2==0},index=idx)
X,_=ForecasterRecursive(Ridge(),lags=2).create_train_X_y(y,exog=ex); print("Rec ", dict(X.dtypes.astype(str)))
X,_=ForecasterDirect(Ridge(),lags=2,steps=2).create_train_X_y(y,exog=ex); print("Dir ", dict(X.dtypes.astype(str)))
X,_=ForecasterRecursiveMultiSeries(Ridge(),lags=2).create_train_X_y({"a":y,"b2":y},exog=ex); print("MS  ", dict(X.dtypes.astype(str)))
