"""Behaviour matrix: estimator x encoding x categorical_features x exog dtype.

Usage: matrix.py <estimator> <out.pkl> [dict]
Stores, per configuration, predictions, intervals, one-step-ahead metric and
the metric of an equivalent backtesting (steps=1, no refit), or the error.
"""
import os
import sys
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import skforecast
print(skforecast.__file__, flush=True)

from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold, backtesting_forecaster_multiseries,
    grid_search_forecaster_multiseries,
)
import common as C

est_name = sys.argv[1]
out_file = sys.argv[2]
as_dict = len(sys.argv) > 3 and sys.argv[3] == "dict"
dtypes = C.DTYPES if not as_dict else ["float", "cat_int_vals", "mixed", "nullable"]

results = {}
t0 = time.time()
for dtype_name in dtypes:
    try:
        series, exog, exog_fut, cat_like, cat_cols = C.make_data(
            dtype_name, as_dict=as_dict
        )
    except Exception as e:  # noqa: BLE001
        print("DATA ERROR", dtype_name, repr(e)[:200], flush=True)
        continue
    exog_cols = list(exog.columns) if not as_dict else list(exog["s_a"].columns)
    for cf_mode in C.CATFEATS:
        cf = C.catfeat_value(cf_mode, cat_like, exog_cols)
        for enc in C.ENCODINGS:
            key = (est_name, dtype_name, cf_mode, enc, as_dict)
            cap = C.Capture()
            res = {}

            def new_forecaster():
                return ForecasterRecursiveMultiSeries(
                    estimator=C.make_estimator(est_name), lags=3, encoding=enc,
                    categorical_features=cf,
                )

            forecaster = cap(new_forecaster)
            if isinstance(forecaster, tuple):
                res["init"] = forecaster
                results[key] = res
                continue
            fit_out = cap(lambda: forecaster.fit(
                series=series, exog=exog, store_in_sample_residuals=True
            ))
            res["fit"] = fit_out
            if fit_out is None:
                res["pred"] = cap(lambda: forecaster.predict(
                    steps=5, exog=exog_fut
                ).to_numpy())
                res["resid"] = cap(lambda: {
                    k: (None if v is None else np.sort(v)[::7].copy())
                    for k, v in forecaster.in_sample_residuals_.items()
                })
                res["boot"] = cap(lambda: forecaster.predict_interval(
                    steps=3, exog=exog_fut, method="bootstrapping", n_boot=20,
                    interval=[0.1, 0.9], random_state=3
                ).to_numpy())
                res["conf"] = cap(lambda: forecaster.predict_interval(
                    steps=3, exog=exog_fut, method="conformal", interval=0.8,
                ).to_numpy())
                res["imp"] = cap(lambda: (
                    lambda d: None if d is None else d.to_numpy()
                )(forecaster.get_feature_importances()))
                res["names"] = list(forecaster.X_train_features_names_out_)
            res["warn_fit_pred"] = sorted(cap.warns)

            # Equivalent validation: one-step-ahead vs backtesting steps=1
            n_train = 60
            cap2 = C.Capture()
            f_bt = cap2(new_forecaster)
            res["bt"] = cap2(lambda: backtesting_forecaster_multiseries(
                forecaster=f_bt, series=series, exog=exog,
                cv=TimeSeriesFold(steps=1, initial_train_size=n_train, refit=False),
                metric="mean_absolute_error", n_jobs=1, show_progress=False,
            )[0].set_index("levels")["mean_absolute_error"].to_dict())
            f_osa = cap2(new_forecaster)
            res["osa"] = cap2(lambda: grid_search_forecaster_multiseries(
                forecaster=f_osa, series=series, exog=exog,
                cv=OneStepAheadFold(initial_train_size=n_train),
                param_grid={}, lags_grid=[3], metric="mean_absolute_error",
                aggregate_metric=["weighted_average", "average", "pooling"],
                return_best=False, n_jobs=1, show_progress=False, verbose=False,
            ).iloc[0].filter(like="mean_absolute_error").to_dict())
            res["warn_val"] = sorted(cap2.warns)
            results[key] = res
    print(dtype_name, round(time.time() - t0, 1), flush=True)

with open(out_file, "wb") as f:
    pickle.dump(results, f)
print("done", len(results), round(time.time() - t0, 1))
