"""
Option (c), item 1. CatBoost in ForecasterRecursiveMultiSeries: 6 series, 300
rows, a `category` exog `c` with 8 categories (p1_common.make_data), backtesting
of 1 step without refit over the last 100 rows.

A. Today: `categorical_features=None` (alone and with `cat_features` in
   `fit_kwargs`) and `'auto'`. "predict vs as fitted" compares the backtesting
   predictions (numpy, `_build_predict_function`) with the estimator evaluated
   on the DataFrame that `fit` builds for the test rows (same cast as `fit`).
B. Prototype of (c) with `categorical_features=None`: numpy + `cat_features`
   + integer cast. Variants "codes" (pandas codes, predict maps values to
   codes) and "values". The reference is the estimator evaluated on a
   DataFrame with integer columns, which does not use `_build_predict_function`.
   They are also compared with `'auto'` + numpy fit ("sorted").
Extra layouts: 10% NaN in `c`; a category (99) that only appears in the test
rows, declared in the dtype; both.
"""
import _env
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils.utils import cast_catboost_categorical_columns_dataframe
from p1_common import make_data, numpy_fit
from p1c_common import option_c, head, backtest, test_rows, to_matrix, attempt, fmt

N_TRAIN = 200
mk = lambda: CatBoostRegressor(iterations=40, verbose=0, thread_count=1, random_state=0,
                               allow_writing_files=False)


def with_nan(exog, exog_p, rng):
    exog.loc[rng.random(len(exog)) < 0.1, "c"] = np.nan


def with_unknown(exog, exog_p, rng):
    exog["c"] = exog["c"].cat.add_categories([99])
    late = (np.arange(len(exog)) >= N_TRAIN) & (rng.random(len(exog)) < 0.3)
    exog.loc[late, "c"] = 99


def with_both(exog, exog_p, rng):
    with_nan(exog, exog_p, rng)
    with_unknown(exog, exog_p, rng)


LAYOUTS = {
    "0..7": (tuple(range(8)), None),
    "10..80": (tuple(range(10, 90, 10)), None),
    "perm(0..7)": ((5, 2, 7, 0, 3, 6, 1, 4), None),
    "-4..3": (tuple(range(-4, 4)), None),
    "10..80, 10% NaN": (tuple(range(10, 90, 10)), with_nan),
    "10..80, 99 only in test": (tuple(range(10, 90, 10)), with_unknown),
    "10..80, NaN and 99 in test": (tuple(range(10, 90, 10)), with_both),
    "-4..3, NaN": (tuple(range(-4, 4)), with_nan),
    "strings": (tuple("stuvwxyz"), None),
}


def mae(p):
    return f"{(p['pred'] - p['y_true']).abs().mean():5.2f}"


def as_fitted_today(f, series, exog):
    """Estimator on the DataFrame of the test rows, cast as in `fit` (:2467-2472)."""
    X, key = test_rows(f, series, exog, N_TRAIN)
    kw = {"cat_features": list(f.estimator.get_cat_feature_indices())}
    X = cast_catboost_categorical_columns_dataframe(
        X=X, fit_kwargs=kw, estimator=f.estimator, feature_names=list(X.columns)) if kw["cat_features"] else X
    return pd.Series(f.estimator.predict(X), index=key)


def as_fitted_proto(f, series, exog, variant):
    """Estimator on a DataFrame whose categorical columns are integers (codes or values)."""
    X, key = test_rows(f, series, exog, N_TRAIN)
    M, _ = to_matrix(X, variant)
    df = pd.DataFrame(M, columns=X.columns)
    for j in f.estimator.get_cat_feature_indices():
        df[df.columns[j]] = df.iloc[:, j].fillna(-1).astype(int)
    return pd.Series(f.estimator.predict(df), index=key)


print("# A. Today")
for label, (cats, extra) in LAYOUTS.items():
    series, exog, _ = make_data(cats, n_series=6, extra=extra)
    s_tr, e_tr = head(series, N_TRAIN), head(exog, N_TRAIN)
    print(f"- categories {label}")
    for cf, enc, fk in ((None, "ordinal", None), (None, "ordinal", {"cat_features": [3]}),
                        (None, "ordinal", {"cat_features": ["c"]}), ("auto", "ordinal", None),
                        ("auto", "ordinal_category", None), (None, "ordinal_category", {"cat_features": [1, 3]})):
        new = lambda: ForecasterRecursiveMultiSeries(mk(), lags=1, encoding=enc, categorical_features=cf,
                                                     fit_kwargs=fk)
        f = new()
        res = attempt(lambda: f.fit(series=s_tr, exog=e_tr))
        tag = f"    cf={cf!s:4} encoding={enc:16} fit_kwargs={fk!s:28}"
        if isinstance(res, str):
            print(f"{tag}: fit -> {res}")
            continue
        bt = attempt(lambda: backtest(new(), series, exog, N_TRAIN))
        if isinstance(bt, str):
            print(f"{tag}: fit ok, backtesting -> {bt}")
            continue
        ref = as_fitted_today(f, series, exog)
        print(f"{tag}: fit ok, cat_features of the model {f.estimator.get_cat_feature_indices()}, MAE {mae(bt)}"
              f" | predict vs as fitted: {fmt(np.abs(bt['pred'].reindex(ref.index) - ref).max())}"
              f" (MAE as fitted {np.abs(ref - bt['y_true'].reindex(ref.index)).mean():5.2f})")

print("\n# B. Prototype of (c), categorical_features=None")
for label, (cats, extra) in LAYOUTS.items():
    series, exog, _ = make_data(cats, n_series=6, extra=extra)
    s_tr, e_tr = head(series, N_TRAIN), head(exog, N_TRAIN)
    print(f"- categories {label}")
    for enc in ("ordinal", "ordinal_category"):
        preds = {}
        for variant, cf, ctx in (("codes", None, lambda: option_c("codes")),
                                 ("values", None, lambda: option_c("values")),
                                 ("sorted", "auto", numpy_fit)):
            new = lambda: ForecasterRecursiveMultiSeries(mk(), lags=1, encoding=enc, categorical_features=cf)
            with ctx():
                f = new()
                res = attempt(lambda: f.fit(series=s_tr, exog=e_tr))
                bt = res if isinstance(res, str) else attempt(lambda: backtest(new(), series, exog, N_TRAIN))
            tag = f"    encoding={enc:16} (c) {variant:6}"
            if isinstance(bt, str):
                print(f"{tag}: -> {bt}")
                continue
            preds[variant] = bt["pred"]
            out = f"{tag}: fit ok, cat_features {f.estimator.get_cat_feature_indices()}, MAE {mae(bt)}"
            if variant != "sorted":
                ref = as_fitted_proto(f, series, exog, variant)
                out += f" | predict vs as fitted: {fmt(np.abs(bt['pred'].reindex(ref.index) - ref).max())}"
            print(out)
        if "codes" in preds:
            print("      against (c) codes: " + ", ".join(
                f"{k} {fmt(np.abs(v - preds['codes']).max())}" for k, v in preds.items() if k != "codes"))
print(_env.versions())
