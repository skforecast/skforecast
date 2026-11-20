"""
P1, item 2. What users lose when the estimator of ForecasterRecursiveMultiSeries
is fitted with numpy: `feature_names_in_` and the documented pattern
`forecaster.estimator.predict(X DataFrame)` (docs/user_guides/
training-and-prediction-matrices.ipynb, cells 43, 46, 58 and 59).
Default `categorical_features='auto'`, `category` exog of strings.
"""
import _env
import warnings
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from p1_common import make_data, numpy_fit, attempt

series, exog, exog_p = make_data(tuple("stuvwxyz"), n_series=6)
EST = {
    "Ridge": lambda: Ridge(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=5, random_state=0, n_jobs=1),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=20, random_state=0),
    "LGBM": lambda: LGBMRegressor(n_estimators=20, verbose=-1, n_jobs=1, random_state=0),
    "XGB": lambda: XGBRegressor(n_estimators=20, n_jobs=1, random_state=0),
}


def names(est):
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    if hasattr(est, "feature_name_"):
        return est.feature_name_
    return None


def predict_df(est, X, ref):
    """estimator.predict(DataFrame): equal to the reference? which warnings?"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = attempt(lambda: est.predict(X))
    if isinstance(p, str):
        return p
    msg = sorted({str(x.message).splitlines()[0][:70] for x in w})
    return f"equal={np.allclose(p, ref)} warnings={msg}"


for name, mk in EST.items():
    for enc in ("ordinal", "ordinal_category"):
        print(f"\n# {name}, encoding={enc}")
        for mode in ("today", "fix"):
            f = ForecasterRecursiveMultiSeries(mk(), lags=2, encoding=enc)
            if mode == "fix":
                with numpy_fit():
                    f.fit(series=series, exog=exog)
            else:
                f.fit(series=series, exog=exog)
            X_train, _ = f.create_train_X_y(series=series, exog=exog)
            Xp = f.create_predict_X(3, levels=["a"], exog=exog_p).drop(columns="level")
            ref_p = f.predict(3, levels=["a"], exog=exog_p)["pred"].to_numpy()
            ref_t = f.estimator.predict(X_train.to_numpy(dtype=float))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                f.predict(3, levels=["a"], exog=exog_p)
            print(f"  {mode:5} names in estimator: {names(f.estimator)}")
            print(f"        warnings of forecaster.predict: {len(w)}")
            print(f"        estimator.predict(X_train)        -> {predict_df(f.estimator, X_train, ref_t)}")
            print(f"        estimator.predict(create_predict_X)-> {predict_df(f.estimator, Xp, ref_p)}")
            print(f"        get_feature_importances: {attempt(lambda: f.get_feature_importances()['feature'].tolist())}")

print("\n# ForecasterRecursive today (reference: its estimator is already fitted with numpy)")
for name, mk in EST.items():
    f = ForecasterRecursive(mk(), lags=2)
    f.fit(series["a"], exog=exog)
    X_train, _ = f.create_train_X_y(series["a"], exog=exog)
    print(f"  {name:12} names: {names(f.estimator)} | estimator.predict(X_train) -> "
          f"{predict_df(f.estimator, X_train, f.estimator.predict(X_train.to_numpy(dtype=float)))}")
print(_env.versions())
