"""
P1, item 3. Decision D-NP2: what to do with a `category` exog when
`categorical_features=None` once the estimator is fitted with numpy.

Reference "today (DataFrame)": the model fitted today, evaluated the way it
was fitted (estimator.predict on the DataFrame), that is, what the user
believes to have. Options, all fitted and predicted with numpy:
  (a)  codes of the OrdinalEncoder (position in the sorted categories) as a
       numeric column (literal D-NP2 (a))
  (a') the values of the categories as a numeric column (plain
       `to_numpy(dtype=float)`; what ForecasterRecursive does today)
  (c)  codes + column index given to the estimator as categorical
       (= `categorical_features='auto'`)
Columns: is the fitted model the same as today's (in-sample predictions), and
the backtesting MAE (steps=1, no refit, 100 points x 6 series, noise sd = 1).
"""
import _env
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster_multiseries
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from p1_common import make_data, numpy_fit, attempt

EST = {
    "LGBM": lambda: LGBMRegressor(n_estimators=40, verbose=-1, n_jobs=1, random_state=0),
    "XGB": lambda: XGBRegressor(n_estimators=40, n_jobs=1, random_state=0),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=40, random_state=0),
    "Ridge": lambda: Ridge(),
}
CATS = {"0..7": tuple(range(8)), "10..80": tuple(range(10, 90, 10)), "perm(0..7)": (5, 2, 7, 0, 3, 6, 1, 4),
        "uneven": (1, 2, 3, 4, 5, 6, 7, 1000)}
cv = TimeSeriesFold(steps=1, initial_train_size=200, refit=False)


def evaluate(mk, exog, cf, numpy, series):
    f = ForecasterRecursiveMultiSeries(mk(), lags=1, categorical_features=cf)
    fit = lambda: f.fit(series=series, exog=exog)
    if numpy:
        with numpy_fit():
            fit()
            m, _ = backtesting_forecaster_multiseries(
                forecaster=f, series=pd.DataFrame(series), exog=exog, cv=cv,
                metric="mean_absolute_error", n_jobs=1, verbose=False, show_progress=False)
        mae = float(m["mean_absolute_error"].mean())
    else:
        fit()
        mae = np.nan
    X, _ = f.create_train_X_y(series=series, exog=exog)
    in_sample = f.estimator.predict(X if not numpy else X.to_numpy(dtype=float))
    return in_sample, mae


for name, mk in EST.items():
    print(f"\n# {name}")
    for label, cats in CATS.items():
        series, exog, _ = make_data(cats, n_series=6)
        # Codes of the forecaster's OrdinalEncoder: position in the sorted categories.
        codes = exog.assign(c=np.searchsorted(sorted(cats), exog["c"].to_numpy()).astype(float))
        ref, _ = evaluate(mk, exog, None, False, series)
        out = []
        for opt, (ex, cf) in {"(a) codes": (codes, None), "(a') values": (exog, None), "(c) index": (exog, "auto")}.items():
            p, mae = evaluate(mk, ex, cf, True, series)
            d = np.abs(p - ref).max()
            out.append(f"{opt}: {'same model' if d < 1e-6 else f'changes (max {d:.3g})'}, MAE {mae:.2f}")
        print(f"  categories {label:10} | " + " | ".join(out))

print("\n# ForecasterRecursive today, categorical_features=None: what does the estimator get?")
series, exog, _ = make_data((5, 2, 7, 0, 3, 6, 1, 4))
f = ForecasterRecursive(Ridge(), lags=1, categorical_features=None)
X, _ = f.create_train_X_y(series["a"], exog=exog)
print("  exog c (first 5):", exog["c"].iloc[1:6].tolist(), "codes:", exog["c"].cat.codes.iloc[1:6].tolist(),
      "-> X_train c:", X["c"].iloc[:5].tolist(), "dtype", X["c"].dtype)
print(_env.versions())
