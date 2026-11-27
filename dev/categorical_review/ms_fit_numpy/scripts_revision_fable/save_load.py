"""save_forecaster / load_forecaster. Usage: save_load.py <save|load> <dir>

`save` fits, saves each forecaster and its predictions. `load` loads them (with
the tree of the current PYTHONPATH), predicts and compares with the saved
predictions, and recomputes the in-sample residuals.
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import skforecast
print(skforecast.__file__, flush=True)
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.utils import save_forecaster, load_forecaster
import common as C

mode, folder = sys.argv[1], sys.argv[2]
os.makedirs(folder, exist_ok=True)
CASES = [
    ("lgbm", "cat_int_vals", "auto", "ordinal_category"),
    ("lgbm", "cat_int_vals", "none", "ordinal"),
    ("lgbm", "cat_int_codes", "none", "ordinal_category"),
    ("xgb", "cat_int_vals", "auto", "ordinal_category"),
    ("xgb", "cat_int_vals", "none", "ordinal_category"),
    ("hgb", "mixed", "auto", "ordinal_category"),
    ("hgb", "cat_int_vals", "none", "ordinal_category"),
    ("catboost", "mixed", "auto", "ordinal_category"),
    ("ridge", "float", "auto", "onehot"),
    ("ridge", "bool", "auto", None),
]
for est_name, dtype_name, cf_mode, enc in CASES:
    series, exog, exog_fut, cat_like, cat_cols = C.make_data(dtype_name)
    cf = C.catfeat_value(cf_mode, cat_like, list(exog.columns))
    tag = f"{est_name}_{dtype_name}_{cf_mode}_{enc}"
    path = os.path.join(folder, tag + ".joblib")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            if mode == "save":
                f = ForecasterRecursiveMultiSeries(
                    C.make_estimator(est_name), lags=3, encoding=enc, categorical_features=cf
                )
                f.fit(series=series, exog=exog, store_in_sample_residuals=True)
                pred = f.predict(steps=5, exog=exog_fut)["pred"].to_numpy()
                resid = {k: np.sort(v) for k, v in f.in_sample_residuals_.items()}
                save_forecaster(f, path, verbose=False)
                pickle.dump((pred, resid), open(path + ".pred", "wb"))
                f2 = load_forecaster(path, verbose=False)
                pred2 = f2.predict(steps=5, exog=exog_fut)["pred"].to_numpy()
                print(f"  {tag:45} saved; reload in same tree predicts the same: {np.array_equal(pred, pred2)}")
            else:
                pred, resid = pickle.load(open(path + ".pred", "rb"))
                f = load_forecaster(path, verbose=False)
                pred2 = f.predict(steps=5, exog=exog_fut)["pred"].to_numpy()
                f.set_in_sample_residuals(series=series, exog=exog)
                resid2 = {k: np.sort(v) for k, v in f.in_sample_residuals_.items()}
                same_resid = all(
                    len(resid[k]) == len(resid2[k]) and np.allclose(resid[k], resid2[k]) for k in resid
                )
                max_resid = max(float(np.abs(resid[k] - resid2[k]).max()) for k in resid if len(resid[k]) == len(resid2[k]))
                f.fit(series=series, exog=exog)
                pred3 = f.predict(steps=5, exog=exog_fut)["pred"].to_numpy()
                print(f"  {tag:45} loaded; predict == saved: {np.array_equal(pred, pred2)} (max diff {np.abs(pred - pred2).max():.4f}) | "
                      f"set_in_sample_residuals == those of the saved fit: {same_resid} (max diff {max_resid:.4f}) | "
                      f"refit predict == saved: {np.array_equal(pred, pred3)} (max diff {np.abs(pred - pred3).max():.4f})")
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:45} ERR {type(e).__name__}: {str(e)[:200]}")
    ws = sorted({f"{x.category.__name__}: {str(x.message)[:100]}" for x in w
                 if "DataTransformation" not in x.category.__name__ and "InputType" not in x.category.__name__})
    for x in ws:
        print("       warn:", x)
