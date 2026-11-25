"""
DOC-lightgbm-categorical-few-rows: with few rows per category, LightGBM with a
native categorical feature (what `categorical_features='auto'` configures) can
be much worse than with the same column as numeric codes. The parameter
responsible is `min_data_per_group` (default 100).

For 15 to 240 rows per category, prints the MAE of `ForecasterRecursive` with
the native categorical feature and several values of its parameters, and with
the column as float (15 rows per category: 5.13 native, 3.28 numeric and 0.45
native with `min_data_per_group=1`; from 120 rows, the same).

Usage: python lgbm_few_rows.py  (with `PYTHONPATH` set to the repository root if the
installed skforecast is not this code; the first line printed is the path of
the package used)
"""
import warnings
import numpy as np
import pandas as pd
import lightgbm
import skforecast
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive

warnings.simplefilter("ignore")
print(skforecast.__file__, "lightgbm", lightgbm.__version__, "pandas", pd.__version__)
effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0, 7.0, -5.0])
CONFIGS = {
    "default": {},
    "min_data_per_group=1": {"min_data_per_group": 1},
    "cat_smooth=0": {"cat_smooth": 0},
    "cat_l2=0": {"cat_l2": 0},
    "min_data_per_group=1, cat_smooth=0": {"min_data_per_group": 1, "cat_smooth": 0},
    "max_cat_to_onehot=32": {"max_cat_to_onehot": 32},
}

for n_train in (120, 240, 480, 960, 1920):
    rng = np.random.default_rng(7)
    n = n_train + 80
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    codes = rng.permutation(np.arange(n) % 8)
    y = pd.Series(effect[codes] + rng.normal(scale=0.3, size=n), index=idx, name="y")
    exog_cat = pd.DataFrame({"c": pd.Categorical(codes)}, index=idx)
    exog_num = pd.DataFrame({"c": codes.astype(float)}, index=idx)
    out = {}
    for label, params in CONFIGS.items():
        for kind, exog in (("native", exog_cat), ("numeric", exog_num)):
            if kind == "numeric" and label != "default":
                continue
            f = ForecasterRecursive(
                estimator=LGBMRegressor(
                    n_estimators=50, n_jobs=1, random_state=1, verbose=-1, **params
                ),
                lags=3,
            )
            f.fit(y=y.iloc[:n_train], exog=exog.iloc[:n_train])
            p = f.predict(steps=80, exog=exog.iloc[n_train:])
            out[f"{kind} {label}"] = np.mean(np.abs(p.to_numpy() - y.iloc[n_train:].to_numpy()))
    print(f"rows per category {n_train // 8:>4}: " + " | ".join(f"{k}={v:.2f}" for k, v in out.items()))
