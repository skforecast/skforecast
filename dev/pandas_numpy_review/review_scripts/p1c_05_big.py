"""
Option (c), item 5. A more demanding dataset (`p1c_common.make_big`): 30
series, exog as a dict, c40 (40 categories with values 100, 107, ..., declared
in a shuffled order, effect partly ordinal), c3 (3 categories declared as
[30, 10, 20]), a numeric exog, and a category (value 240) that is declared in
the dtype, has no rows in the training window and appears in 20% of the rows
of the second half of the test window.

`categorical_features=None` in all the options but "(c) sorted":
  today      fit with the DataFrame, predict with numpy (current code)
  as fitted  today's estimator evaluated on the DataFrame of the test rows
             (what the user believes to have; what one-step-ahead reports)
  (c) codes  numpy with the pandas codes + index, predict maps values to codes
  (c) values numpy with the values + index, predict untouched
  (c) sorted `categorical_features='auto'` + numpy fit (OrdinalEncoder codes)
  (a)        OrdinalEncoder codes as a numeric column, numpy fit
  (a')       values as a numeric column, numpy fit
Columns: "model" = max abs difference of the in-sample predictions with
today's estimator (LightGBM: also the tree structure); "predict" = max abs
difference of the backtesting predictions (1 step, no refit) with "as fitted";
MAE of the backtesting against y_true (noise sd = 1), all rows / rows with a
seen category / rows with the unseen category.

Usage: python p1c_05_big.py [lgbm|xgb|hgb|catboost ...]
"""
import _env
import sys
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils.utils import _build_predict_function, cast_catboost_categorical_columns_dataframe
from p1_common import numpy_fit
from p1c_common import (
    make_big, option_c, head, backtest, test_rows, to_matrix, lgbm_structure, same_structure, attempt,
)


def factories():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    return {
        "lgbm": lambda: LGBMRegressor(n_estimators=150, verbose=-1, n_jobs=1, random_state=0),
        "xgb": lambda: XGBRegressor(n_estimators=150, n_jobs=1, random_state=0),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=150, random_state=0),
        "catboost": lambda: CatBoostRegressor(iterations=150, verbose=0, thread_count=1, random_state=0,
                                              allow_writing_files=False),
    }


def sorted_codes(exog):
    """Exog with the `category` columns replaced by the OrdinalEncoder codes (float)."""
    out = {}
    for k, df in exog.items():
        df = df.copy()
        for c in ("c40", "c3"):
            cats = np.sort(df[c].cat.categories.to_numpy())
            df[c] = np.searchsorted(cats, df[c].to_numpy()).astype(float)
        out[k] = df
    return out


def mae(p, mask):
    e = (p["pred"] - p["y_true"]).abs()
    out = f"{e.mean():6.2f} /{e[~mask].mean():6.2f}"
    return out + (f" /{e[mask].mean():6.2f}" if mask.any() else " /   n/a")


def fmt(d):
    return "identical" if d == 0 else ("same (<1e-6)" if d < 1e-6 else f"CHANGES (max {d:.3g})")


def run(name, mk, n, n_train, share, detail):
    series, exog, unseen = make_big(n=n, ordinal_share=share, unseen_from=n_train + (n - n_train) // 2)
    s_tr, e_tr = head(series, n_train), head(exog, n_train)
    new = lambda cf="none": ForecasterRecursiveMultiSeries(
        mk(), lags=3, categorical_features=None if cf == "none" else cf)
    print(f"\n## {name}: {n_train} training rows x 30 series, test {n - n_train}, ordinal share {share}"
          f"   (MAE: all / seen / unseen category)")

    # Today
    f0 = new()
    res = attempt(lambda: f0.fit(series=s_tr, exog=e_tr))
    ref = ins0 = None
    if isinstance(res, str):
        print(f"  today      : fit -> {res}")
    else:
        Xte, key = test_rows(f0, series, exog, n_train)
        mask = pd.Series(np.concatenate([unseen[k].to_numpy() for k in series]),
                         index=pd.MultiIndex.from_product([series, next(iter(series.values())).index])
                         ).swaplevel().reindex(key).to_numpy()
        Xtr = f0.create_train_X_y(series=s_tr, exog=e_tr)[0][f0.X_train_features_names_out_]
        ins0 = f0.estimator.predict(Xtr)
        bt0 = backtest(new(), series, exog, n_train)
        ref = pd.DataFrame({"y_true": bt0["y_true"].reindex(key).to_numpy(),
                            "pred": attempt(lambda: f0.estimator.predict(Xte))}, index=key)
        mask = pd.Series(mask, index=key)
        print(f"  today      : MAE {mae(bt0.reindex(key), mask)} | predict vs as fitted: "
              f"{fmt(np.abs(bt0['pred'].reindex(key) - ref['pred']).max())}")
        print(f"  as fitted  : MAE {mae(ref, mask)}")

    rows = {}
    for label, cf, ctx, ex in (
        ("(c) codes ", "none", lambda: option_c("codes"), exog),
        ("(c) values", "none", lambda: option_c("values"), exog),
        ("(c) sorted", "auto", numpy_fit, exog),
        ("(a)       ", "none", numpy_fit, sorted_codes(exog)),
        ("(a')      ", "none", numpy_fit, exog),
    ):
        with ctx():
            f = new(cf)
            res = attempt(lambda: f.fit(series=s_tr, exog=head(ex, n_train)))
            if isinstance(res, str):
                print(f"  {label} : fit -> {res}")
                continue
            bt = attempt(lambda: backtest(new(cf), series, ex, n_train))
            if isinstance(bt, str):
                print(f"  {label} : backtesting -> {bt}")
                continue
            Xtr = f.create_train_X_y(series=s_tr, exog=head(ex, n_train))[0][f.X_train_features_names_out_]
            variant = "codes" if "codes" in label else "values"
            ins = _build_predict_function(f.estimator)(to_matrix(Xtr, variant)[0])
        if ref is None:
            Xte, key = test_rows(f, series, ex, n_train)
            mask = pd.Series(np.concatenate([unseen[k].to_numpy() for k in series]),
                             index=pd.MultiIndex.from_product([series, next(iter(series.values())).index])
                             ).swaplevel().reindex(key)
        bt = bt.reindex(key)
        rows[label] = bt
        out = f"  {label} : MAE {mae(bt, mask)}"
        if ins0 is not None and detail:
            out += f" | model: {fmt(np.abs(ins - ins0).max())}"
            if name == "lgbm":
                out += f", trees {same_structure(lgbm_structure(f0.estimator), lgbm_structure(f.estimator))}"
            out += f" | predict vs as fitted: {fmt(np.abs(bt['pred'] - ref['pred']).max())}"
            m = mask.to_numpy()
            if m.any():
                out += (f" (seen rows {fmt(np.abs(bt['pred'] - ref['pred'])[~m].max())},"
                        f" unseen rows {fmt(np.abs(bt['pred'] - ref['pred'])[m].max())})")
        print(out)
    if ref is None and detail and "(c) codes " in rows:
        base = rows["(c) codes "]["pred"]
        print("  no model today; backtesting predictions against (c) codes: " + ", ".join(
            f"{k.strip()} {fmt(np.abs(v['pred'] - base).max())}" for k, v in rows.items() if "(c)" in k and "codes" not in k))


F = factories()
for name in (sys.argv[1:] or list(F)):
    print(f"\n# {name}")
    run(name, F[name], 400, 300, 0.5, True)
    run(name, F[name], 400, 300, 0.0, False)
    run(name, F[name], 400, 300, 1.0, False)
    run(name, F[name], 200, 100, 0.5, True)
print(_env.versions())
