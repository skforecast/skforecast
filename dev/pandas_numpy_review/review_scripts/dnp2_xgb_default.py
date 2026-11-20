"""
D-NP2: XGBoost treats a `category` column as categorical with its default
parameters (no `enable_categorical=True`), and then predicts differently with
the DataFrame and with a numpy array of the same values.

Usage: bash run3.sh dnp2_xgb_default.py
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

rng = np.random.default_rng(0)
c = rng.integers(0, 3, 300)
X = pd.DataFrame({"x": rng.normal(size=300), "c": pd.Categorical(c * 10 + 10)})
y = np.array([-50.0, 0, 50])[c] + rng.normal(size=300)
X_numpy = np.column_stack([np.zeros(3), [10, 20, 30]])
X_frame = pd.DataFrame({"x": np.zeros(3), "c": pd.Categorical([10, 20, 30])})
for kwargs in ({}, {"enable_categorical": True}):
    try:
        model = XGBRegressor(n_estimators=20, **kwargs).fit(X, y)
        print(
            f"xgboost {xgboost.__version__} {kwargs}: fit ok, feature_types "
            f"{model.get_booster().feature_types}; predict(DataFrame 10, 20, 30) "
            f"{model.predict(X_frame).round(0)}; predict(numpy 10, 20, 30) "
            f"{model.predict(X_numpy).round(0)}; true [-50, 0, 50]"
        )
    except Exception as e:  # noqa: BLE001
        print(f"xgboost {xgboost.__version__} {kwargs}: {type(e).__name__}: {str(e)[:150]}")
print(_env.versions())
