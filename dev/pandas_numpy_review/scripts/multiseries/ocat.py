import warnings, timeit, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
import lightgbm.basic as B
from skforecast.recursive import ForecasterRecursiveMultiSeries
rng = np.random.default_rng(0); LEN=1000; N=100
idx = pd.date_range("2020-01-01", periods=LEN, freq="D")
series = {f"s{i:03d}": pd.Series(rng.normal(size=LEN).cumsum(), index=idx) for i in range(N)}
for enc in ["ordinal", "ordinal_category"]:
    f = ForecasterRecursiveMultiSeries(LGBMRegressor(verbose=-1, n_jobs=1), lags=24, encoding=enc)
    X, y = f.create_train_X_y(series)
    sig = B._data_from_pandas.__code__.co_varnames[:B._data_from_pandas.__code__.co_argcount]
    t = min(timeit.repeat(lambda: B._data_from_pandas(X, "auto", "auto", None) if len(sig)==4 else B._data_from_pandas(X, "auto", "auto"), number=3, repeat=3))/3
    print(enc, "blocks", len(X._mgr.blocks), "lightgbm _data_from_pandas", round(t*1e3,2), "ms", "| to_numpy", round(min(timeit.repeat(X.to_numpy, number=3, repeat=3))/3*1e3,2), "ms", sig)
