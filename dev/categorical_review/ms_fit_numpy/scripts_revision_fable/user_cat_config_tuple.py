"""Categorical features configured by the user (categorical_features=None).

For each estimator and way of configuring (constructor / fit_kwargs, by name /
by position), prints one line: fit status, MAE of the 8 future steps against the
known truth, backtesting(steps=1, no refit) vs one-step-ahead metric, params
left on the estimator, and state checks (second fit, set_lags, error in fit).
"""
import sys
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

only = sys.argv[1] if len(sys.argv) > 1 else None
rng = np.random.default_rng(5)
n, n_fut = 160, 8
idx = pd.date_range("2021-01-01", periods=n + n_fut, freq="D")
codes = rng.integers(0, 3, size=n + n_fut)
x_f = rng.normal(size=n + n_fut)
effect = np.array([0.0, 30.0, -20.0])[codes]
series_all = pd.DataFrame(
    {s: 10 * (i + 1) + effect + 2 * x_f + rng.normal(scale=0.3, size=n + n_fut)
     for i, s in enumerate(["s_a", "s_b", "s_c"])}, index=idx
)
series, truth = series_all.iloc[:n], series_all.iloc[n:]


def make_exog(kind):
    vals = codes * 10 + 10
    if kind == "cat":
        col = pd.Categorical(vals, categories=[10, 20, 30])
    elif kind == "cat_rev":
        col = pd.Categorical(vals, categories=[30, 10, 20])
    else:
        col = vals.astype("int64")
    return pd.DataFrame({"x_f": x_f, "x_c": col}, index=idx)


def pos(enc, lags=3):
    return {"ordinal": lags + 2, "ordinal_category": lags + 2,
            "onehot": lags + 3 + 1, None: lags + 1}[enc]


def n_feat(enc, lags=3):
    return {"ordinal": lags + 3, "ordinal_category": lags + 3,
            "onehot": lags + 5, None: lags + 2}[enc]


def configs(enc):
    p = pos(enc)
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    lg = dict(n_estimators=30, num_leaves=8, min_child_samples=3,
              min_data_per_group=1, cat_smooth=1, cat_l2=0, verbose=-1,
              n_jobs=1, random_state=1, deterministic=True, force_row_wise=True)
    xg = dict(n_estimators=30, max_depth=3, n_jobs=1, random_state=1, tree_method="hist")
    cb = dict(iterations=30, depth=3, verbose=0, random_seed=1, thread_count=1,
              allow_writing_files=False)
    hg = dict(max_iter=30, min_samples_leaf=3, random_state=1)
    ft = ["q"] * n_feat(enc)
    ft[p] = "c"
    mask = np.zeros(n_feat(enc), dtype=bool)
    mask[p] = True
    return {
        "lgbm none": (lambda: LGBMRegressor(**lg), None),
        "lgbm fit_kwargs idx": (lambda: LGBMRegressor(**lg), {"categorical_feature": [p]}),
        "lgbm fit_kwargs name": (lambda: LGBMRegressor(**lg), {"categorical_feature": ["x_c"]}),
        "lgbm ctor idx": (lambda: LGBMRegressor(categorical_feature=[p], **lg), None),
        "lgbm ctor name": (lambda: LGBMRegressor(categorical_feature=["x_c"], **lg), None),
        "lgbm ctor str": (lambda: LGBMRegressor(categorical_feature="name:x_c", **lg), None),
        "xgb none": (lambda: XGBRegressor(**xg), None),
        "xgb feature_types": (lambda: XGBRegressor(feature_types=list(ft), enable_categorical=True, **xg), None),
        "cb none": (lambda: CatBoostRegressor(**cb), None),
        "cb fit_kwargs idx": (lambda: CatBoostRegressor(**cb), {"cat_features": [p]}),
        "cb fit_kwargs name": (lambda: CatBoostRegressor(**cb), {"cat_features": ["x_c"]}),
        "cb ctor idx": (lambda: CatBoostRegressor(cat_features=(p,), **cb), None),
        "cb ctor name": (lambda: CatBoostRegressor(cat_features=("x_c",), **cb), None),
        "hgb none": (lambda: HistGradientBoostingRegressor(**hg), None),
        "hgb idx": (lambda: HistGradientBoostingRegressor(categorical_features=[p], **hg), None),
        "hgb name": (lambda: HistGradientBoostingRegressor(categorical_features=["x_c"], **hg), None),
        "hgb mask": (lambda: HistGradientBoostingRegressor(categorical_features=mask.copy(), **hg), None),
    }


def cat_state(est):
    p = est.get_params()
    keys = ["categorical_feature", "cat_features", "feature_types",
            "enable_categorical", "categorical_features"]
    out = {}
    for k in keys:
        if k in p and p[k] is not None:
            v = p[k]
            if isinstance(v, np.ndarray):
                v = "arr" + str(v.astype(int).tolist())
            out[k] = v
    return str(out).replace("'", "")


def run(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001
            out = f"ERR {type(e).__name__}: {str(e)[:110]}"
    ws = sorted({x.category.__name__ for x in w if "skforecast" in x.category.__module__})
    return out, ws


for exog_kind in ["int", "cat", "cat_rev"]:
    exog_all = make_exog(exog_kind)
    exog, exog_fut = exog_all.iloc[:n], exog_all.iloc[n:]
    for enc in C.ENCODINGS:
        for name, (make, fk) in configs(enc).items():
            if only and not name.startswith(only):
                continue

            def new():
                return ForecasterRecursiveMultiSeries(
                    estimator=make(), lags=3, encoding=enc,
                    categorical_features=None, fit_kwargs=fk,
                )

            f, _ = run(new)
            if isinstance(f, str):
                print(f"{exog_kind:8}|{str(enc):17}|{name:21}| INIT {f}")
                continue
            state0 = cat_state(f.estimator)
            r, ws = run(lambda: f.fit(series=series, exog=exog))
            if isinstance(r, str):
                print(f"{exog_kind:8}|{str(enc):17}|{name:21}| FIT {r} | params after: {cat_state(f.estimator)} | warns {ws}")
                continue
            pred, _ = run(lambda: f.predict(steps=n_fut, exog=exog_fut))
            if isinstance(pred, str):
                mae = pred
            else:
                wide = pred.pivot(columns="level", values="pred")
                mae = f"{np.abs(wide.to_numpy() - truth.to_numpy()).mean():7.3f}"
            state1 = cat_state(f.estimator)
            # second fit: same predictions?
            r2, _ = run(lambda: (f.fit(series=series, exog=exog), f.predict(steps=n_fut, exog=exog_fut))[1])
            same2 = (
                "same" if not isinstance(r2, str) and not isinstance(pred, str)
                and np.array_equal(r2["pred"].to_numpy(), pred["pred"].to_numpy())
                else ("DIFF" if not isinstance(r2, str) else r2)
            )
            # set_lags then fit
            def lags5():
                f.set_lags(5)
                f.fit(series=series, exog=exog)
                return f.predict(steps=2, exog=exog_fut)
            r3, _ = run(lags5)
            lags_res = "ok" if not isinstance(r3, str) else r3
            state3 = cat_state(f.estimator)
            # validation equivalence
            fb = new()
            bt, _ = run(lambda: backtesting_forecaster_multiseries(
                forecaster=fb, series=series, exog=exog,
                cv=TimeSeriesFold(steps=1, initial_train_size=110, refit=False),
                metric="mean_absolute_error", n_jobs=1, show_progress=False,
            )[0].set_index("levels").loc["weighted_average", "mean_absolute_error"])
            fo = new()
            osa, _ = run(lambda: grid_search_forecaster_multiseries(
                forecaster=fo, series=series, exog=exog,
                cv=OneStepAheadFold(initial_train_size=110),
                param_grid={}, lags_grid=[3], metric="mean_absolute_error",
                aggregate_metric="weighted_average",
                return_best=False, n_jobs=1, show_progress=False, verbose=False,
            ).iloc[0]["mean_absolute_error__weighted_average"])
            if isinstance(bt, str) or isinstance(osa, str):
                val = f"bt={bt if isinstance(bt, str) else round(bt, 3)} osa={osa if isinstance(osa, str) else round(osa, 3)}"
            else:
                val = f"bt={bt:7.3f} osa={osa:7.3f} {'EQ' if abs(bt - osa) < 1e-9 else 'NEQ'}"
            print(
                f"{exog_kind:8}|{str(enc):17}|{name:21}| mae={mae} | {val} | 2nd fit {same2} | "
                f"set_lags(5) {lags_res} | params {state0} -> {state1} -> {state3} | warns {ws}"
            )
