import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster, backtesting_forecaster_multiseries
from prof import callsites
warnings.filterwarnings("ignore")
N = 2000
idx = pd.date_range("2020-01-01", periods=N + 24, freq="h")
rng = np.random.default_rng(0)
y = pd.Series(rng.normal(size=N + 24).cumsum(), index=idx, name="y")
series = pd.DataFrame(rng.normal(size=(len(idx), 10)).cumsum(axis=0), index=idx, columns=[f"s{i}" for i in range(10)])
exnum = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["e1", "e2", "e3"])
excat = pd.DataFrame({"cat": pd.Categorical(np.array(list("abcdefg"))[idx.dayofweek])}, index=idx)
cv = TimeSeriesFold(steps=24, initial_train_size=800, refit=False)
for xn, ex in [("num3", exnum), ("cat1", excat)]:
    print(f"\n##### ForecasterDirect Ridge {xn}")
    f = ForecasterDirect(Ridge(), steps=24, lags=24, n_jobs=1)
    f.fit(y.iloc[:N], exog=ex.iloc[:N], store_in_sample_residuals=True)
    callsites(lambda: f.predict(exog=ex.iloc[N:]), n=300, cats=("pandas", "sklearn.transformer(OrdinalEncoder/scaler)"))
    callsites(lambda: f.predict_interval(exog=ex.iloc[N:], n_boot=250), n=100, cats=("pandas",), top=10)
    callsites(lambda: backtesting_forecaster(f, y.iloc[:N], cv=cv, exog=ex.iloc[:N], metric="mean_absolute_error", n_jobs=1, show_progress=False), n=1, cats=("pandas",), top=15)
    print(f"\n##### DMV Ridge 10 series {xn}")
    d = ForecasterDirectMultiVariate(Ridge(), level="s0", steps=24, lags=24, n_jobs=1)
    d.fit(series.iloc[:N], exog=ex.iloc[:N], store_in_sample_residuals=True)
    callsites(lambda: d.predict(exog=ex.iloc[N:]), n=300, cats=("pandas", "sklearn.transformer(OrdinalEncoder/scaler)"))
    callsites(lambda: backtesting_forecaster_multiseries(d, series.iloc[:N], cv=cv, levels="s0", exog=ex.iloc[:N], metric="mean_absolute_error", n_jobs=1, show_progress=False), n=1, cats=("pandas",), top=18)
