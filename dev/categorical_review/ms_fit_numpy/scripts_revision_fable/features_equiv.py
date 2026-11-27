"""Default `categorical_features='auto'` with the other features of the forecaster.

Weights, differentiation, transformers, window features, calendar features,
RangeIndex, dict with NaN, backtesting with and without refit, intervals,
set_in_sample_residuals. Usage: features_equiv.py <estimator> <out.pkl>
"""
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import skforecast
print(skforecast.__file__, flush=True)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import (
    TimeSeriesFold, backtesting_forecaster_multiseries,
)
import common as C

est_name, out_file = sys.argv[1], sys.argv[2]
results = {}


def weight_func(index):
    return np.where(index >= index[len(index) // 2], 2.0, 1.0)


def run(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return C.summarize_exception(e)


def to_range(obj):
    if isinstance(obj, dict):
        return {k: to_range(v) for k, v in obj.items()}
    obj = obj.copy()
    return obj


VARIANTS = {
    "plain": {},
    "weights": dict(weight_func=weight_func, series_weights={"s_a": 2.0, "s_b": 1.0, "s_c": 0.5}),
    "diff": dict(differentiation=1),
    "scaler": dict(transformer_series=StandardScaler(), transformer_exog=make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include="float64")),
        remainder="passthrough", verbose_feature_names_out=False,
    ).set_output(transform="pandas")),
    "onehot_exog": dict(transformer_exog=make_column_transformer(
        (OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["x_c"]),
        remainder="passthrough", verbose_feature_names_out=False,
    ).set_output(transform="pandas")),
    "window": dict(window_features=RollingFeatures(stats=["mean", "std"], window_sizes=4)),
    "dropna": dict(dropna_from_series=True),
}

for as_dict in (False, True):
    for dtype_name in ["float", "cat_int_vals", "mixed", "bool"]:
        series, exog, exog_fut, cat_like, cat_cols = C.make_data(dtype_name, as_dict=as_dict)
        for vname, vkw in VARIANTS.items():
            if vname == "onehot_exog" and dtype_name == "bool":
                continue
            for enc in C.ENCODINGS:
                key = (est_name, dtype_name, vname, enc, as_dict)
                res = {}

                def new():
                    return ForecasterRecursiveMultiSeries(
                        estimator=C.make_estimator(est_name), lags=3,
                        encoding=enc, **vkw
                    )

                f = run(new)
                if isinstance(f, tuple):
                    results[key] = {"init": f}
                    continue
                res["fit"] = run(lambda: f.fit(series=series, exog=exog, store_in_sample_residuals=True))
                if res["fit"] is None:
                    res["pred"] = run(lambda: f.predict(steps=5, exog=exog_fut)["pred"].to_numpy())
                    res["pred_levels"] = run(lambda: f.predict(steps=3, exog=exog_fut, levels=["s_b"])["pred"].to_numpy())
                    res["boot"] = run(lambda: f.predict_bootstrapping(steps=2, exog=exog_fut, n_boot=10, random_state=1).select_dtypes("number").to_numpy())
                    res["quant"] = run(lambda: f.predict_quantiles(steps=2, exog=exog_fut, n_boot=10, random_state=1).select_dtypes("number").to_numpy())
                    res["conf"] = run(lambda: f.predict_interval(steps=2, exog=exog_fut, method="conformal", interval=0.8).select_dtypes("number").to_numpy())
                    res["resid"] = run(lambda: {k: (None if v is None else np.sort(v)) for k, v in f.in_sample_residuals_.items()})
                    res["imp"] = run(lambda: (lambda d: None if d is None else d["importance"].to_numpy())(f.get_feature_importances()))
                    # set_in_sample_residuals must reproduce those of fit
                    def set_resid():
                        f.set_in_sample_residuals(series=series, exog=exog)
                        return {k: (None if v is None else np.sort(v)) for k, v in f.in_sample_residuals_.items()}
                    res["set_resid"] = run(set_resid)
                for refit in (False, True, 2):
                    fb = new()
                    res[f"bt_refit_{refit}"] = run(lambda: (lambda m, p: (m.select_dtypes("number").to_numpy(), p.select_dtypes("number").to_numpy()))(
                        *backtesting_forecaster_multiseries(
                            forecaster=fb, series=series, exog=exog,
                            cv=TimeSeriesFold(steps=4, initial_train_size=60, refit=refit, fixed_train_size=False),
                            metric=["mean_absolute_error", "mean_absolute_scaled_error"],
                            interval=[0.1, 0.9] if refit is False else None,
                            interval_method="conformal",
                            n_jobs=1, show_progress=False,
                        )))
                results[key] = res
    print(as_dict, len(results), flush=True)

with open(out_file, "wb") as fh:
    pickle.dump(results, fh)
print("done", len(results))
