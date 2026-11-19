import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
rng = np.random.default_rng(0)
n = 400
idx = pd.date_range("2020-01-01", periods=n + 10, freq="D")
cat = pd.Categorical(rng.choice([10, 20, 30], size=n + 10), categories=[10, 20, 30])
effect = pd.Series(cat).map({10: 0.0, 20: 50.0, 30: -50.0}).to_numpy(dtype=float)
series = {f"s{i}": pd.Series(rng.normal(size=n) + effect[:n], index=idx[:n]) for i in range(3)}
exog = pd.DataFrame({"c": cat}, index=idx)
for est in [LGBMRegressor(verbose=-1, n_estimators=50, min_child_samples=5, n_jobs=1), HistGradientBoostingRegressor(categorical_features="from_dtype")]:
    for enc in ["ordinal", "ordinal_category"]:
        f = ForecasterRecursiveMultiSeries(est, lags=3, categorical_features=None, encoding=enc)
        f.fit(series, exog=exog.iloc[:n])
        p = f.predict(steps=10, exog=exog.iloc[n:])
        X, y = f.create_train_X_y(series, exog=exog.iloc[:n])
        Xp = f.create_predict_X(steps=10, exog=exog.iloc[n:])
        print(type(est).__name__, enc, "X_train dtypes:", dict(X.dtypes.astype(str).value_counts()))
        # expected effect for step 1..3, series s0
        p0 = p[p.level == "s0"]["pred"].to_numpy()[:5]
        print("   predict:", np.round(p0, 1), " true effect:", effect[n:n+5])
        # predicting with DataFrame of the predictors (estimator.predict on X with category dtype)
        Xp0 = Xp[Xp.level == "s0"].drop(columns="level") if "level" in Xp.columns else Xp
        print("   create_predict_X dtypes:", dict(Xp0.dtypes.astype(str).value_counts()))
