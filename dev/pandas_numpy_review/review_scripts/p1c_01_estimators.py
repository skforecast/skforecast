"""
Option (c), item 2. Estimator level, no forecaster: what must the numpy matrix
hold so that "numpy + column index as categorical" gives the model fitted
today with a DataFrame and a `category` column?

Reference "today": `est.fit(DataFrame with a category column)`, evaluated on
that DataFrame. Variants, fitted with numpy and the index given through
`configure_estimator_categorical_features` (CatBoost: plus the integer cast):
  V  values of the categories
  C  pandas codes (position in `dtype.categories`; NaN for missing)
  S  codes of the OrdinalEncoder (position in the sorted categories)
For each: does it fit, is the in-sample prediction the one of today (max abs
difference; LightGBM: also the tree structure), and is the prediction on the
matrix of VALUES (what `predict` builds today) the same (column "values->").

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


def data(categories, n=2400, seed=0, nan_share=0.0, ordered=False, used=None):
    """y = effect of c (non monotone) + 3 x + noise. `used`: categories with rows."""
    rng = np.random.default_rng(seed)
    cats = list(categories)
    used = cats if used is None else list(used)
    eff = dict(zip(used, rng.normal(0, 30, len(used))))
    v = np.array(used, dtype=object)[rng.integers(0, len(used), n)]
    x = rng.normal(size=n)
    y = np.array([eff[k] for k in v]) + 3 * x + rng.normal(size=n)
    c = pd.Categorical(v, categories=cats, ordered=ordered)
    if nan_share:
        c = pd.Categorical(np.where(rng.random(n) < nan_share, None, v), categories=cats, ordered=ordered)
    return pd.DataFrame({"x": x, "c": c, "z": rng.normal(size=n)}), y


LAYOUTS = {
    "0..7": dict(categories=range(8)),
    "10..80": dict(categories=range(10, 90, 10)),
    "perm(0..7) declared [5,2,7,0,3,6,1,4]": dict(categories=(5, 2, 7, 0, 3, 6, 1, 4)),
    "declared 0..15, used 8 (even)": dict(categories=range(16), used=range(0, 16, 2)),
    "negative -3..4": dict(categories=range(-3, 5)),
    "large 0..6 and 100000": dict(categories=(0, 1, 2, 3, 4, 5, 6, 100000)),
    "huge 0..6 and 3e9": dict(categories=(0, 1, 2, 3, 4, 5, 6, 3_000_000_000)),
    "floats .25 .5 .75 1.5 2.5 2.75 3.5 4.5": dict(categories=(0.25, 0.5, 0.75, 1.5, 2.5, 2.75, 3.5, 4.5)),
    "10..80 with 10% NaN": dict(categories=range(10, 90, 10), nan_share=0.1),
    "10..80 ordered=True": dict(categories=range(10, 90, 10), ordered=True),
    "40 categories 100,107,...": dict(categories=tuple(100 + 7 * k for k in range(40))),
    "300 categories 0..299": dict(categories=range(300), n=9000),
    "300 categories 1000..1299": dict(categories=range(1000, 1300), n=9000),
}


def sorted_codes(df):
    cats = df["c"].cat.categories
    rank = np.empty(len(cats))
    rank[np.argsort(cats.to_numpy())] = np.arange(len(cats))
    codes = df["c"].cat.codes.to_numpy()
    X, _ = to_matrix(df, "values")
    X[:, 1] = np.where(codes < 0, np.nan, rank[np.clip(codes, 0, None)])
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
        df, y = data(**kw)
        ref = attempt(lambda: mk().fit(df, y))
        print(f"- {label}")
        if isinstance(ref, str):
            print(f"    today (DataFrame): fit -> {ref}")
            ref_pred = None
        else:
            ref_pred = ref.predict(df)
            Xv, _ = to_matrix(df, "values")
            p_np = attempt(lambda: _build_predict_function(ref)(Xv))
            bug = p_np if isinstance(p_np, str) else f"max abs diff {np.abs(p_np - ref_pred).max():.3g}"
            print(f"    today (DataFrame): fit ok; its numpy predict on VALUES vs on the DataFrame: {bug}")
        mats = {"V values": to_matrix(df, "values")[0], "C pandas codes": to_matrix(df, "codes")[0],
                "S sorted codes": sorted_codes(df)}
        preds = {}
        for var, X in mats.items():
            est = attempt(lambda: fit_numpy(mk, X, y))
            if isinstance(est, str):
                print(f"    {var:14}: fit -> {est}")
                continue
            p = attempt(lambda: _build_predict_function(est)(X))
            if isinstance(p, str):
                print(f"    {var:14}: fit ok, predict -> {p}")
                continue
            preds[var] = p
            out = f"    {var:14}: fit ok"
            if ref_pred is not None:
                out += f" | vs today: {fmt(np.abs(p - ref_pred).max())}"
                if name == "lgbm":
                    out += f" | trees: {same_structure(lgbm_structure(ref), lgbm_structure(est))}"
            if var != "V values":
                pv = attempt(lambda: _build_predict_function(est)(mats["V values"]))
                out += " | values->: " + (pv if isinstance(pv, str) else fmt(np.abs(pv - p).max()))
            print(out)
        if "V values" in preds and "C pandas codes" in preds:
            print(f"    V vs C (relabelling): {fmt(np.abs(preds['V values'] - preds['C pandas codes']).max())}"
                  + (f" | S vs C: {fmt(np.abs(preds['S sorted codes'] - preds['C pandas codes']).max())}"
                     if "S sorted codes" in preds else ""))
print(_env.versions())
