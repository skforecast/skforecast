# Estimator parameters that name columns and are not categorical settings:
# do they still work when the estimator is fitted with a numpy array?
import warnings

import numpy as np
import pandas as pd
import skforecast
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

print(skforecast.__file__)
warnings.simplefilter("ignore")
rng = np.random.default_rng(123)
idx = pd.date_range("2020-01-01", periods=200, freq="D")
series = pd.DataFrame(
    {"l1": rng.normal(size=200).cumsum(), "l2": rng.normal(size=200).cumsum()},
    index=idx,
)
estimators = {
    "HGB monotonic_cst by name": lambda: HistGradientBoostingRegressor(
        max_iter=20, monotonic_cst={"lag_1": 1}
    ),
    "XGB monotone_constraints by name": lambda: XGBRegressor(
        n_estimators=20, monotone_constraints={"lag_1": 1}
    ),
    "XGB interaction_constraints by name": lambda: XGBRegressor(
        n_estimators=20, interaction_constraints=[["lag_1", "lag_2"], ["lag_3"]]
    ),
}
for name, make in estimators.items():
    for label, build, data in [
        (
            "multiseries",
            lambda: ForecasterRecursiveMultiSeries(
                estimator=make(), lags=3, encoding="ordinal"
            ),
            {"series": series},
        ),
        (
            "ForecasterRecursive",
            lambda: ForecasterRecursive(estimator=make(), lags=3),
            {"y": series["l1"]},
        ),
    ]:
        try:
            forecaster = build()
            forecaster.fit(**data)
            forecaster.predict(steps=3)
            result = "ok"
        except Exception as exc:
            result = f"{type(exc).__name__}: {str(exc)[:90]}"
        print(f"{name:38s} | {label:20s} | {result}")
