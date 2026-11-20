"""LightGBM + categorical_features=None + exog `category` of integers [10, 20, 30]."""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
n = 600; idx = pd.date_range("2020", periods=n, freq="D"); rng = np.random.default_rng(0)
cat = np.array([10, 20, 30])[rng.integers(0, 3, n)]
effect = {10: -50.0, 20: 0.0, 30: 50.0}
y = pd.Series(np.array([effect[c] for c in cat]) + rng.normal(size=n), index=idx)
exog = pd.DataFrame({"c": pd.Categorical(cat, categories=[10, 20, 30])}, index=idx)
fidx = pd.date_range(idx[-1] + idx.freq, periods=3, freq="D")
exog_p = pd.DataFrame({"c": pd.Categorical([10, 20, 30], categories=[10, 20, 30])}, index=fidx)
mk = lambda: LGBMRegressor(n_estimators=50, verbose=-1, random_state=0)
f = ForecasterRecursive(mk(), lags=1, categorical_features=None); f.fit(y, exog=exog)
print("Recursive   predict:", f.predict(3, exog=exog_p).round(1).tolist())
fm = ForecasterRecursiveMultiSeries(mk(), lags=1, categorical_features=None)
fm.fit(series={"a": y, "b": y.copy()}, exog=exog)
print("MultiSeries predict:", fm.predict(3, levels=["a"], exog=exog_p)["pred"].round(1).tolist())
Xp = fm.create_predict_X(3, levels=["a"], exog=exog_p)
print("MultiSeries estimator.predict(create_predict_X):", np.round(fm.estimator.predict(Xp.drop(columns="level", errors="ignore")), 1).tolist())
