import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries, ForecasterRecursive
rng = np.random.default_rng(0)
n = 400
idx = pd.date_range("2020-01-01", periods=n + 10, freq="D")
cat = pd.Categorical(rng.choice([10, 20, 30], size=n + 10), categories=[10, 20, 30])
effect = pd.Series(cat).map({10: 0.0, 20: 50.0, 30: -50.0}).to_numpy(dtype=float)
series = {f"s{i}": pd.Series(rng.normal(size=n) + effect[:n], index=idx[:n]) for i in range(3)}
exog = pd.DataFrame({"c": cat}, index=idx)
est = LGBMRegressor(verbose=-1, n_estimators=50, min_child_samples=5, n_jobs=1)
f = ForecasterRecursiveMultiSeries(est, lags=3, categorical_features=None)
f.fit(series, exog=exog.iloc[:n])
Xp = f.create_predict_X(steps=5, exog=exog.iloc[n:], levels=["s0"])
print("estimator.predict(DataFrame create_predict_X):", np.round(f.estimator.predict(Xp.drop(columns="level")), 1))
print("forecaster.predict:", np.round(f.predict(steps=5, exog=exog.iloc[n:], levels=["s0"])["pred"].to_numpy(), 1))
print("pandas_categorical stored by lightgbm:", f.estimator.booster_.pandas_categorical)
fr = ForecasterRecursive(LGBMRegressor(verbose=-1, n_estimators=50, min_child_samples=5, n_jobs=1), lags=3, categorical_features=None)
fr.fit(series["s0"], exog=exog.iloc[:n])
print("ForecasterRecursive.predict:", np.round(fr.predict(steps=5, exog=exog.iloc[n:]).to_numpy(), 1))
print("true effect", effect[n:n+5])
