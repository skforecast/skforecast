"""
P1, item 2. Does the fitted model change if `ForecasterRecursiveMultiSeries.fit`
gives the estimator `X_train.to_numpy(dtype=float)` instead of the DataFrame?

For each estimator, encoding, `categorical_features` (cf) and category layout:
what the estimator receives today, the 8-step prediction of series 'a' today
and with the prototype (expected: EFFECT8 of p1_common, [30, -50, 10, 50, -30, 0, 40, -10]), and whether the two
fitted models give the same in-sample prediction on the float matrix.

Usage: python p1_fix_grid.py [lgbm|xgb|xgb_default|hgb|catboost|ridge ...]
"""
import _env
import sys
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursiveMultiSeries
from p1_common import make_data, spy_fit, numpy_fit, attempt, EFFECT8


def factories():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    return {
        "lgbm": lambda: LGBMRegressor(n_estimators=40, verbose=-1, n_jobs=1, random_state=0),
        "xgb": lambda: XGBRegressor(n_estimators=40, enable_categorical=True, tree_method="hist",
                                    n_jobs=1, random_state=0),
        "xgb_default": lambda: XGBRegressor(n_estimators=40, n_jobs=1, random_state=0),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=40, random_state=0),
        "catboost": lambda: CatBoostRegressor(verbose=0, iterations=30, thread_count=1, random_state=0),
        "ridge": lambda: Ridge(),
    }


def est_params(est):
    """Categorical configuration stored in the estimator after `fit`."""
    p = est.get_params()
    out = {k: p[k] for k in ("feature_types", "enable_categorical", "categorical_features",
                             "cat_features") if k in p and p[k] is not None}
    if hasattr(est, "is_categorical_"):
        out["is_categorical_"] = None if est.is_categorical_ is None else est.is_categorical_.astype(int).tolist()
    if hasattr(est, "booster_"):
        out["pandas_categorical"] = est.booster_.pandas_categorical
    if hasattr(est, "get_cat_feature_indices"):
        out["cat_idx"] = est.get_cat_feature_indices()
    return out


def one(mk, enc, cf, categories, n_cat_cols, numpy):
    series, exog, exog_p = make_data(categories, n_cat_cols=n_cat_cols, n_series=6)
    est = mk()
    f = ForecasterRecursiveMultiSeries(est, lags=1, encoding=enc, categorical_features=cf)
    log = []
    with spy_fit(type(est), log):
        if numpy:
            with numpy_fit():
                res = attempt(lambda: f.fit(series=series, exog=exog, store_in_sample_residuals=True))
        else:
            res = attempt(lambda: f.fit(series=series, exog=exog, store_in_sample_residuals=True))
    if isinstance(res, str):
        return {"fit": "fit -> " + res, "pred": "-", "probe": "-"}
    d = log[0]
    got = d["type"]
    if d["type"] == "DataFrame":
        got += str({k: v for k, v in d["dtypes"].items() if v != "float64"})
    else:
        got += f"[{d['dtypes']}]"
    # Float matrix that `predict` would build for the training rows.
    X, _ = f.create_train_X_y(series=series, exog=exog)
    X = X[f.X_train_features_names_out_].to_numpy(dtype=float)
    from skforecast.utils.utils import _build_predict_function
    probe = attempt(lambda: _build_predict_function(f.estimator, forecaster=f)(X))
    return {
        "fit": f"{got} kw={d['kwargs']} {est_params(f.estimator)}"[:230],
        "pred": attempt(lambda: f.predict(len(categories), levels=["a"], exog=exog_p)["pred"].to_numpy()),
        "probe": probe,
        "resid": attempt(lambda: float(np.abs(f.in_sample_residuals_["_unknown_level"]).mean())),
    }


def same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return "n/a"
    return "identical" if np.array_equal(a, b) else f"DIFFERENT (max abs diff {np.abs(a - b).max():.3g})"


R8, X10, PERM, STR = tuple(range(8)), tuple(range(10, 90, 10)), (5, 2, 7, 0, 3, 6, 1, 4), tuple("stuvwxyz")
CASES = [  # (encoding, cf, categories, n_cat_cols); always 6 series
    ("ordinal", None, R8, 1),
    ("ordinal", None, X10, 1),
    ("ordinal", None, PERM, 1),
    ("ordinal", None, STR, 1),
    ("ordinal", "auto", R8, 1),
    ("ordinal", "auto", X10, 1),
    ("ordinal", "auto", STR, 1),
    ("ordinal", ["c"], R8, 2),
    ("ordinal", ["c"], X10, 2),
    ("ordinal_category", "auto", X10, 1),
    ("ordinal_category", None, R8, 1),
    ("ordinal_category", None, R8, 0),
    ("onehot", "auto", X10, 1),
    ("onehot", None, R8, 1),
    (None, "auto", X10, 1),
    (None, None, R8, 1),
]

def mae(pred, cats, k):
    """Mean absolute error of the prediction of 'a' against the known effect."""
    if isinstance(pred, str):
        return pred
    expected = (EFFECT8 if k else 0 * EFFECT8) + (0.2 * EFFECT8[1] if k == 2 else 0.0)
    return f"{np.abs(pred - expected).mean():.1f}"


F = factories()
for name in (sys.argv[1:] or list(F)):
    print(f"\n# {name}   (MAE = error of predict('a') against the known effect; noise sd = 1)")
    for enc, cf, cats, k in CASES:
        a = one(F[name], enc, cf, cats, k, numpy=False)
        b = one(F[name], enc, cf, cats, k, numpy=True)
        label = {R8: "0..7", X10: "10..80", PERM: "perm(0..7)", STR: "strings"}[cats]
        label += {0: ", no category exog", 1: "", 2: ", cols c and c2"}[k]
        print(f"- encoding={enc}, cf={cf}, categories {label}")
        print(f"    today: {a['fit']}")
        print(f"    fix  : {b['fit']}")
        if isinstance(a["pred"], str) and a["pred"] == "-":
            continue
        print(f"    predict MAE today {mae(a['pred'], cats, k)} fix {mae(b['pred'], cats, k)}"
              f" | forecast {same(a['pred'], b['pred'])}"
              f" | fitted model (float matrix) {same(a['probe'], b['probe'])}"
              f" | mean abs in-sample residual today {a.get('resid')!s:.5} fix {b.get('resid')!s:.5}")
print(_env.versions())
