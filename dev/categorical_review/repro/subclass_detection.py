"""
CAT-subclass-detection: the family of the estimator is recognised by
`type(estimator).__module__`, so a subclass defined outside the package of the
estimator (in the user's code) is not recognised: the native categorical
features are not configured, with no error or warning, and the model differs.

Fits `ForecasterRecursive` with a categorical exog and the default
`categorical_features='auto'`, with `LGBMRegressor` and `CatBoostRegressor` and
with an empty subclass of each, and prints the categorical features the fitted
model has and the difference between the predictions. For the CatBoost
subclass, it also prints that `_is_fitted_catboost` does not recognise it and
that `fit` keeps the estimator object when the forecaster is fitted again.

Usage: python subclass_detection.py  (with `PYTHONPATH` set to the repository root if the
installed skforecast is not this code; the first line printed is the path of
the package used)

Session of 2026-11-25 (pandas 2.3.3, CatBoost 1.2.10, LightGBM 4.7.0, Windows):
LightGBM, `categorical_column` [3] and None with the subclass, max abs diff
2.52; CatBoost, categorical indices [3] and [] with the subclass, max abs
diff 2.03; `_is_fitted_catboost` False and the same estimator object after
the second fit.
"""
import warnings

import catboost
import lightgbm
import numpy as np
import pandas as pd
import skforecast
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.utils import _is_fitted_catboost

warnings.simplefilter("ignore")
print(skforecast.__file__, skforecast.__version__)
print("pandas", pd.__version__, "catboost", catboost.__version__, "lightgbm", lightgbm.__version__)
rng = np.random.RandomState(7)
n = 260
index = pd.date_range("2020-01-01", periods=n, freq="D")
codes = rng.permutation(np.arange(n) % 6)
effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0])
y = pd.Series(effect[codes] + rng.normal(scale=0.3, size=n), index=index, name="y")
exog = pd.DataFrame({"c": pd.Categorical(codes), "x": rng.normal(size=n)}, index=index)


class MyLGBM(LGBMRegressor):
    pass


class MyCatBoost(CatBoostRegressor):
    pass


def fit_predict(estimator):
    forecaster = ForecasterRecursive(estimator=estimator, lags=3)
    forecaster.fit(y=y.iloc[:250], exog=exog.iloc[:250])
    return forecaster, forecaster.predict(steps=5, exog=exog.iloc[250:255]).to_numpy()


kw = dict(n_estimators=30, verbose=-1, random_state=1, n_jobs=1)
plain, p_plain = fit_predict(LGBMRegressor(**kw))
sub, p_sub = fit_predict(MyLGBM(**kw))
print(
    "LightGBM, categorical_column:", plain.estimator.booster_.params.get("categorical_column"),
    "| subclass:", sub.estimator.booster_.params.get("categorical_column"),
    "| max abs diff of the predictions:", float(np.abs(p_plain - p_sub).max()),
)

kw = dict(
    iterations=30, depth=3, verbose=0, thread_count=1, random_seed=1,
    allow_writing_files=False,
)
plain, p_plain = fit_predict(CatBoostRegressor(**kw))
sub, p_sub = fit_predict(MyCatBoost(**kw))
print(
    "CatBoost, categorical indices:", list(plain.estimator.get_cat_feature_indices()),
    "| subclass:", list(sub.estimator.get_cat_feature_indices()),
    "| max abs diff of the predictions:", float(np.abs(p_plain - p_sub).max()),
)
estimator = sub.estimator
is_fitted = _is_fitted_catboost(estimator)
sub.fit(y=y.iloc[:250], exog=exog.iloc[:250])
print(
    "CatBoost subclass, _is_fitted_catboost:", is_fitted,
    "| same estimator object after the second fit:", sub.estimator is estimator,
)
