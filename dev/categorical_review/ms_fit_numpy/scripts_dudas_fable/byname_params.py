# Estimator params that refer to features by name (not categorical).
import warnings
import numpy as np, pandas as pd
import skforecast; print(skforecast.__file__)
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries, ForecasterRecursive
warnings.simplefilter("ignore")
N = 150
idx = pd.date_range("2020", periods=N, freq="D")
rng = np.random.default_rng(0)
series = pd.DataFrame({"a": rng.normal(size=N).cumsum(), "b": rng.normal(size=N).cumsum()}, index=idx)
exog = pd.DataFrame({"exog_num": rng.normal(size=N)}, index=idx)
cases = {
    "HGB monotonic_cst dict by name": lambda: HistGradientBoostingRegressor(max_iter=10, monotonic_cst={"lag_1": 1}),
    "HGB interaction_cst (positions)": lambda: HistGradientBoostingRegressor(max_iter=10, interaction_cst=[[0, 1], [2, 3, 4]]),
    "XGB monotone_constraints dict by name": lambda: XGBRegressor(n_estimators=10, monotone_constraints={"lag_1": 1}),
    "XGB interaction_constraints by name": lambda: XGBRegressor(n_estimators=10, interaction_constraints=[["lag_1", "lag_2"], ["lag_3", "_level_skforecast", "exog_num"]]),
    "XGB feature_weights / positions tuple": lambda: XGBRegressor(n_estimators=10, monotone_constraints=(1, 0, 0, 0, 0)),
}
for name, mk in cases.items():
    out = []
    for fname in ["multiseries", "recursive"]:
        try:
            if fname == "multiseries":
                f = ForecasterRecursiveMultiSeries(estimator=mk(), lags=3)
                f.fit(series=series.iloc[:-3], exog=exog.iloc[:-3])
                f.predict(steps=3, exog=exog.iloc[-3:])
            else:
                if "_level_skforecast" in str(mk().get_params()):
                    out.append("recursive: n/a"); continue
                f = ForecasterRecursive(estimator=mk(), lags=3)
                f.fit(y=series["a"].iloc[:-3], exog=exog.iloc[:-3])
                f.predict(steps=3, exog=exog.iloc[-3:])
            out.append(f"{fname}: ok")
        except Exception as e:
            out.append(f"{fname}: ERR {type(e).__name__}: {str(e)[:90]}")
    print(f"{name:40s} | " + " | ".join(out))
