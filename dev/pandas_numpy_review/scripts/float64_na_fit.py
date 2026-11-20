"""A nullable `Float64` exog column with pd.NA in `fit`."""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect
n = 100; idx = pd.date_range("2020", periods=n, freq="D"); y = pd.Series(np.arange(n, dtype=float), index=idx)
ex = pd.DataFrame({"a": np.arange(n, dtype=float), "b": pd.array(np.arange(n, dtype=float), dtype="Float64")}, index=idx)
ex.loc[idx[50], "b"] = pd.NA
for est in (Ridge(), LGBMRegressor(verbose=-1, n_estimators=5)):
    for name, mk, kw in [("Rec", lambda e: ForecasterRecursive(e, lags=3), dict(y=y, exog=ex)),
                         ("Dir", lambda e: ForecasterDirect(e, lags=3, steps=2), dict(y=y, exog=ex)),
                         ("MS", lambda e: ForecasterRecursiveMultiSeries(e, lags=3), dict(series={"s": y, "t": y}, exog=ex))]:
        try:
            mk(est).fit(**kw); print(type(est).__name__, name, "fit ok")
        except Exception as e:
            print(type(est).__name__, name, "ERR", type(e).__name__, str(e)[:90])
