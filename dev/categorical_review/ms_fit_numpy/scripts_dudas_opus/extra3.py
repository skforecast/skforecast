"""
Pipeline whose first step selects columns by name (ColumnTransformer), and
`feature_names_in_` of the fitted estimator. The tree is chosen with PYTHONPATH.
"""
import json
import warnings

import numpy as np
import pandas as pd

import skforecast
print("skforecast from:", skforecast.__file__, flush=True)

from skforecast.recursive import ForecasterRecursiveMultiSeries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

rng = np.random.default_rng(3)
n = 120
idx = pd.date_range("2020-01-01", periods=n + 5, freq="D")
series = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx[:n])
exog = pd.DataFrame({"x": rng.normal(size=n + 5)}, index=idx)

ct = ColumnTransformer([("sc", StandardScaler(), ["lag_1", "x"])], remainder="passthrough")
for name, est in [("Pipeline(ColumnTransformer by name, Ridge)", make_pipeline(ct, Ridge())),
                  ("Ridge", Ridge())]:
    rec = {"case": name}
    try:
        f = ForecasterRecursiveMultiSeries(est, lags=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.fit(series=series, exog=exog.iloc[:n])
            rec["fit"] = "ok"
            rec["feature_names_in_"] = list(getattr(f.estimator, "feature_names_in_", [])) or None
            f.predict(steps=5, exog=exog.iloc[n:])
            rec["predict"] = "ok"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
    print(json.dumps(rec), flush=True)
