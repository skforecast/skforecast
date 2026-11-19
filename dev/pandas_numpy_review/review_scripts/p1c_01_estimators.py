"""
Option (c), item 2. Estimator level, no forecaster: what must the numpy matrix
hold so that "numpy + column index as categorical" gives the model fitted
today with a DataFrame and a `category` column?

Reference "today": `est.fit(DataFrame with a category column)`, evaluated on
DataFrames (training rows and 600 new rows). Variants, fitted with numpy and
the index given through `configure_estimator_categorical_features` (CatBoost:
plus the integer cast), and evaluated with `_build_predict_function`:
  V  values of the categories
  C  pandas codes (position in `dtype.categories`; NaN for missing)
  S  codes of an OrdinalEncoder fitted on the training rows (position in the
     sorted observed values; NaN for missing or not observed), which is what
     `categorical_features='auto'` gives
For each: does it fit; max abs difference with today in the training rows
("in") and in the new rows ("out"); LightGBM: also the tree structure; and
"values->": difference between predicting on the VALUES (what `predict`
builds today) and on the matrix of the variant.

Usage: python p1c_01_estimators.py [lgbm|xgb|xgb_cat|hgb|catboost ...]
"""
import _env
import sys
import numpy as np
import pandas as pd
from skforecast.utils.utils import (
    configure_estimator_categorical_features, cast_catboost_categorical_columns,
    _build_predict_function,
)
from p1c_common import attempt, lgbm_structure, same_structure, to_matrix

N_TEST = 600


def factories():
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    return {
        "lgbm": lambda: LGBMRegressor(n_estimators=40, verbose=-1, n_jobs=1, random_state=0),
        "xgb": lambda: XGBRegressor(n_estimators=40, n_jobs=1, random_state=0),
        "xgb_cat": lambda: XGBRegressor(n_estimators=40, n_jobs=1, random_state=0, enable_categorical=True),
        "hgb": lambda: HistGradientBoostingRegressor(max_iter=40, random_state=0),
        "catboost": lambda: CatBoostRegressor(iterations=40, verbose=0, thread_count=1, random_state=0,
                                              allow_writing_files=False),
    }


def data(categories, n=2400, seed=0, nan_share=0.0, ordered=False, test_only=()):
    """
    y = effect of c (non monotone) + 3 x + noise; `n` training rows and N_TEST
    new rows. `test_only`: declared categories that only appear in the new rows.
    """
    rng = np.random.default_rng(seed)
    cats = list(categories)
    eff = dict(zip(cats, rng.normal(0, 30, len(cats))))
    train_cats = np.array([k for k in cats if k not in test_only], dtype=object)
    v = np.concatenate([train_cats[rng.integers(0, len(train_cats), n)],
                        np.array(cats, dtype=object)[rng.integers(0, len(cats), N_TEST)]])
    x = rng.normal(size=n + N_TEST)
    y = np.array([eff[k] for k in v]) + 3 * x + rng.normal(size=n + N_TEST)
    if nan_share:
        v = np.where(rng.random(n + N_TEST) < nan_share, None, v)
    df = pd.DataFrame({"x": x, "c": pd.Categorical(v, categories=cats, ordered=ordered),
                       "z": rng.normal(size=n + N_TEST)})
    return df.iloc[:n], y[:n], df.iloc[n:]


LAYOUTS = {
    "0..7": dict(categories=range(8)),
    "10..80": dict(categories=range(10, 90, 10)),
    "perm(0..7) declared [5,2,7,0,3,6,1,4]": dict(categories=(5, 2, 7, 0, 3, 6, 1, 4)),
    "10..90, 50 only in the new rows": dict(categories=range(10, 100, 10), test_only=(50,)),
    "negative -3..4": dict(categories=range(-3, 5)),
    "large 0..6 and 100000": dict(categories=(0, 1, 2, 3, 4, 5, 6, 100000)),
    "huge 0..6 and 3e9": dict(categories=(0, 1, 2, 3, 4, 5, 6, 3_000_000_000)),
    "floats .25 .5 .75 1.5 2.5 2.75 3.5 4.5": dict(categories=(0.25, 0.5, 0.75, 1.5, 2.5, 2.75, 3.5, 4.5)),
    "10..80 with 10% NaN": dict(categories=range(10, 90, 10), nan_share=0.1),
    "10..80 ordered=True": dict(categories=range(10, 90, 10), ordered=True),
    "40 categories 100,107,... shuffled": dict(
        categories=tuple(int(k) for k in np.random.default_rng(1).permutation(100 + 7 * np.arange(40)))),
    "40 categories, 300 training rows": dict(
        categories=tuple(int(k) for k in np.random.default_rng(1).permutation(100 + 7 * np.arange(40))), n=300),
    "300 categories 1000..1299": dict(categories=range(1000, 1300), n=9000),
}


def sorted_codes(df, observed):
    """Matrix with the codes of an OrdinalEncoder fitted on `observed` values."""
    X, _ = to_matrix(df, "values")
    v = X[:, 1]
    pos = np.searchsorted(observed, v)
    ok = np.isfinite(v) & (pos < len(observed))
    ok[ok] = observed[pos[ok]] == v[ok]
    X[:, 1] = np.where(ok, pos, np.nan)
    return X


def fit_numpy(mk, X, y):
    est = mk()
    kw = configure_estimator_categorical_features(est, ["c"], ["x", "c", "z"], {})
    est.fit(cast_catboost_categorical_columns(X, kw, est), y, **kw)
    return est


def fmt(d):
    return "identical" if d == 0 else ("same (<1e-6)" if d < 1e-6 else f"CHANGES (max {d:.3g})")


F = factories()
for name in (sys.argv[1:] or list(F)):
    mk = F[name]
    print(f"\n# {name}")
    for label, kw in LAYOUTS.items():
        df, y, df_new = data(**kw)
        ref = attempt(lambda: mk().fit(df, y))
        print(f"- {label}")
        Xv, Xv_new = to_matrix(df, "values")[0], to_matrix(df_new, "values")[0]
        if isinstance(ref, str):
            print(f"    today (DataFrame): fit -> {ref}")
            ref_in = ref_out = None
        else:
            ref_in = ref.predict(df)
            ref_out = attempt(lambda: ref.predict(df_new))
            p_np = attempt(lambda: _build_predict_function(ref)(Xv))
            bug = p_np if isinstance(p_np, str) else fmt(np.abs(p_np - ref_in).max())
            print(f"    today (DataFrame): fit ok; numpy predict on VALUES against predict on the DataFrame: {bug}"
                  + (f" | DataFrame predict on the new rows -> {ref_out}" if isinstance(ref_out, str) else ""))
        observed = np.unique(Xv[np.isfinite(Xv[:, 1]), 1])
        mats = {"V values": (Xv, Xv_new),
                "C pandas codes": (to_matrix(df, "codes")[0], to_matrix(df_new, "codes")[0]),
                "S sorted codes": (sorted_codes(df, observed), sorted_codes(df_new, observed))}
        preds = {}
        for var, (X, X_new) in mats.items():
            est = attempt(lambda: fit_numpy(mk, X, y))
            if isinstance(est, str):
                print(f"    {var:14}: fit -> {est}")
                continue
            fn = _build_predict_function(est)
            p = attempt(lambda: (fn(X), fn(X_new)))
            if isinstance(p, str):
                print(f"    {var:14}: fit ok, predict -> {p}")
                continue
            preds[var] = p
            out = f"    {var:14}: fit ok"
            if ref_in is not None:
                out += f" | vs today in: {fmt(np.abs(p[0] - ref_in).max())}"
                if not isinstance(ref_out, str):
                    out += f", out: {fmt(np.abs(p[1] - ref_out).max())}"
                if name == "lgbm":
                    out += f" | trees: {same_structure(lgbm_structure(ref), lgbm_structure(est))}"
            if var != "V values":
                pv = attempt(lambda: fn(Xv_new))
                out += " | values->: " + (pv if isinstance(pv, str) else fmt(np.abs(pv - p[1]).max()))
            print(out)
        if ref_in is None and "C pandas codes" in preds:
            print("    no model today; against C: " + ", ".join(
                f"{k[0]} in {fmt(np.abs(v[0] - preds['C pandas codes'][0]).max())},"
                f" out {fmt(np.abs(v[1] - preds['C pandas codes'][1]).max())}"
                for k, v in preds.items() if k[0] != "C"))
print(_env.versions())
