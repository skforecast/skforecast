"""
P1, item 2. One-step-ahead validation against backtesting with steps=1 and no
refit (the same model and the same one-step predictions, so the same metric is
expected). The one-step-ahead path fits and predicts with DataFrames
(model_selection/_utils.py:3031 and 3038); backtesting predicts with numpy.
The prototype of the fix is also applied to the one-step-ahead path (numpy in
`fit` and in `predict`).
"""
import _env
from contextlib import contextmanager
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import skforecast.model_selection._utils as msu
from skforecast.model_selection import (
    OneStepAheadFold, TimeSeriesFold, grid_search_forecaster_multiseries,
)
from skforecast.recursive import ForecasterRecursiveMultiSeries
from p1_common import make_data, numpy_fit


@contextmanager
def numpy_osa():
    """Prototype in the one-step-ahead path: X_train and X_test as float numpy."""
    o = msu.cast_catboost_categorical_columns_dataframe, msu._copy_rows_to_check, msu._check_in_place_fit
    msu.cast_catboost_categorical_columns_dataframe = lambda X, **kw: X.to_numpy(dtype=float)
    msu._copy_rows_to_check = lambda X: None
    msu._check_in_place_fit = lambda X, X_rows: None
    try:
        yield
    finally:
        (msu.cast_catboost_categorical_columns_dataframe, msu._copy_rows_to_check,
         msu._check_in_place_fit) = o


EST = {
    "LGBM": (lambda: LGBMRegressor(n_estimators=40, verbose=-1, n_jobs=1, random_state=0), {"n_estimators": [40]}),
    "XGB": (lambda: XGBRegressor(n_estimators=40, n_jobs=1, random_state=0), {"n_estimators": [40]}),
    "HGB": (lambda: HistGradientBoostingRegressor(max_iter=40, random_state=0), {"max_iter": [40]}),
}


def metric(mk, grid, cf, cats, cv):
    series, exog, _ = make_data(cats, n_series=6)
    f = ForecasterRecursiveMultiSeries(mk(), lags=1, categorical_features=cf)
    res = grid_search_forecaster_multiseries(
        forecaster=f, series=pd.DataFrame(series), exog=exog, cv=cv, param_grid=grid, lags_grid=[1],
        metric="mean_absolute_error", aggregate_metric="average", return_best=False,
        n_jobs=1, verbose=False, show_progress=False,
    )
    return float(res.filter(like="mean_absolute_error").iloc[0, 0])


osa = OneStepAheadFold(initial_train_size=200)
bt = TimeSeriesFold(steps=1, initial_train_size=200, refit=False)
print("MAE (average of 6 series), 100 test points per series; noise sd = 1")
for name, (mk, grid) in EST.items():
    for cf, cats, label in [(None, tuple(range(10, 90, 10)), "10..80"), (None, (5, 2, 7, 0, 3, 6, 1, 4), "perm(0..7)"),
                            (None, tuple(range(8)), "0..7"), ("auto", tuple(range(10, 90, 10)), "10..80")]:
        a, b = metric(mk, grid, cf, cats, osa), metric(mk, grid, cf, cats, bt)
        with numpy_fit(), numpy_osa():
            c, d = metric(mk, grid, cf, cats, osa), metric(mk, grid, cf, cats, bt)
        print(f"  {name:4} cf={cf!s:4} categories {label:10} today: one-step-ahead {a:6.2f} backtesting {b:6.2f}"
              f" | fix: one-step-ahead {c:6.2f} backtesting {d:6.2f}")
print(_env.versions())
