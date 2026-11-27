"""
How ForecasterRecursive and ForecasterDirect handle the same categorical
situations (category exog with values 10/20/30, user configuration by name
or by position). The tree is chosen with PYTHONPATH.
"""
import json
import warnings

import numpy as np
import pandas as pd

import skforecast
print("skforecast from:", skforecast.__file__, flush=True)

from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

rng = np.random.default_rng(0)
n, steps = 300, 12
idx = pd.date_range("2020-01-01", periods=n + steps, freq="D")
effect = {10: 0.0, 20: 40.0, 30: -40.0}
c = rng.choice([10, 20, 30], size=n + steps)
truth = np.array([effect[v] for v in c])
y = pd.Series(truth + rng.normal(size=n + steps), index=idx, name="y")
exog = pd.DataFrame(
    {"num": rng.normal(size=n + steps),
     "cat": pd.Categorical(c, categories=[10, 20, 30])},
    index=idx,
)

cases = [
    ("lgbm", lambda: LGBMRegressor(n_estimators=60, verbose=-1), None, None, "none"),
    ("lgbm", lambda: LGBMRegressor(n_estimators=60, verbose=-1), None,
     {"categorical_feature": ["cat"]}, "fit_kwargs by name"),
    ("lgbm", lambda: LGBMRegressor(n_estimators=60, verbose=-1), None,
     {"categorical_feature": [4]}, "fit_kwargs by position"),
    ("lgbm", lambda: LGBMRegressor(n_estimators=60, verbose=-1, categorical_feature="name:cat"),
     None, None, "constructor by name"),
    ("xgb", lambda: XGBRegressor(n_estimators=60), None, None, "none"),
    ("xgb", lambda: XGBRegressor(n_estimators=60, enable_categorical=True,
                                 feature_types=["q", "q", "q", "q", "c"]),
     None, None, "constructor feature_types"),
    ("hgb", lambda: HistGradientBoostingRegressor(max_iter=60), None, None, "none"),
    ("hgb", lambda: HistGradientBoostingRegressor(max_iter=60, categorical_features=["cat"]),
     None, None, "constructor by name"),
    ("hgb", lambda: HistGradientBoostingRegressor(max_iter=60, categorical_features=[4]),
     None, None, "constructor by position"),
    ("lgbm", lambda: LGBMRegressor(n_estimators=60, verbose=-1), "auto", None, "auto"),
]

for fname, cls, extra in [("ForecasterRecursive", ForecasterRecursive, {}),
                          ("ForecasterDirect", ForecasterDirect, {"steps": steps})]:
    for est, fac, cf, fk, user in cases:
        rec = dict(forecaster=fname, est=est, cf=cf, user=user)
        try:
            f = cls(estimator=fac(), lags=3, categorical_features=cf, fit_kwargs=fk, **extra)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                f.fit(y=y.iloc[:n], exog=exog.iloc[:n])
                p = f.predict(steps=steps, exog=exog.iloc[n:])
            mae = float(np.abs(p.to_numpy() - truth[n:]).mean())
            rec.update(status="ok", mae=round(mae, 3),
                       warnings=sorted({x.category.__name__ for x in w
                                        if x.category.__module__.startswith("skforecast")}))
        except Exception as e:
            rec.update(status="error", err=f"{type(e).__name__}: {str(e)[:200]}")
        print(json.dumps(rec), flush=True)
