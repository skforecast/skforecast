"""Targeted probes. Prints one block per probe; run on base and branch and diff."""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import skforecast
print(skforecast.__file__, flush=True)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import (
    TimeSeriesFold, OneStepAheadFold, backtesting_forecaster_multiseries,
    grid_search_forecaster_multiseries,
)
import common as C

only = sys.argv[1].split(",") if len(sys.argv) > 1 else None


def probe(name):
    def deco(fn):
        if only and name not in only:
            return fn
        print(f"\n===== {name}: {fn.__doc__.strip().splitlines()[0]}")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  PROBE ERROR {type(e).__name__}: {str(e)[:300]}")
        return fn
    return deco


def run(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001
            out = f"ERR {type(e).__name__}: {str(e)[:200]}"
    ws = sorted({f"{x.category.__name__}: {str(x.message)[:110]}" for x in w})
    return out, ws


def lgbm():
    return LGBMRegressor(n_estimators=20, num_leaves=8, min_child_samples=3,
                         min_data_per_group=1, cat_smooth=1, verbose=-1, n_jobs=1,
                         random_state=1, deterministic=True, force_row_wise=True)


def fmt(pred):
    if isinstance(pred, str):
        return pred
    return np.round(pred["pred"].to_numpy()[:6], 4).tolist()


@probe("P01")
def p01():
    """Reported bug: category 10/20/30 with categorical_features=None."""
    series, exog, exog_fut, *_ = C.make_data("cat_int_vals", n=150)
    for est in ["lgbm", "xgb", "catboost", "hgb"]:
        for cf in [None, ["x_k"], "auto"]:
            def bt():
                f = ForecasterRecursiveMultiSeries(C.make_estimator(est), lags=3, categorical_features=cf)
                return backtesting_forecaster_multiseries(
                    forecaster=f, series=series, exog=exog,
                    cv=TimeSeriesFold(steps=1, initial_train_size=100, refit=False),
                    metric="mean_absolute_error", n_jobs=1, show_progress=False,
                )[0].set_index("levels").loc["weighted_average", "mean_absolute_error"]

            def osa():
                f = ForecasterRecursiveMultiSeries(C.make_estimator(est), lags=3, categorical_features=cf)
                return grid_search_forecaster_multiseries(
                    forecaster=f, series=series, exog=exog, cv=OneStepAheadFold(initial_train_size=100),
                    param_grid={}, lags_grid=[3], metric="mean_absolute_error",
                    aggregate_metric="weighted_average", return_best=False, n_jobs=1,
                    show_progress=False, verbose=False,
                ).iloc[0]["mean_absolute_error__weighted_average"]
            b, _ = run(bt)
            o, _ = run(osa)
            b = b if isinstance(b, str) else round(float(b), 4)
            o = o if isinstance(o, str) else round(float(o), 4)
            print(f"  {est:8} cf={str(cf):8} backtesting MAE={b}  one-step-ahead MAE={o}")


@probe("P02")
def p02():
    """create_train_X_y / create_predict_X given by the user to forecaster.estimator.predict."""
    series, exog, exog_fut, *_ = C.make_data("cat_int_vals", n=150)
    for cf in [None, "auto"]:
        for enc in ["ordinal", "ordinal_category"]:
            f = ForecasterRecursiveMultiSeries(lgbm(), lags=3, encoding=enc, categorical_features=cf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f.fit(series=series, exog=exog, store_in_sample_residuals=True)
                X_train, y_train = f.create_train_X_y(series=series, exog=exog)
                insample_user = f.estimator.predict(X_train)
                resid_user = np.sort(y_train.to_numpy() - insample_user)
                resid_fc = np.sort(np.concatenate([f.in_sample_residuals_[k] for k in ["s_a", "s_b", "s_c"]]))
                X_pred = f.create_predict_X(steps=1, exog=exog_fut)
                p_user = f.estimator.predict(X_pred.drop(columns="level"))
                p_fc = f.predict(steps=1, exog=exog_fut)["pred"].to_numpy()
            print(f"  cf={str(cf):5} enc={enc:17} estimator.predict(create_train_X_y) == in-sample of fit: "
                  f"{np.allclose(resid_user, resid_fc)} | estimator.predict(create_predict_X) == predict: "
                  f"{np.allclose(p_user, p_fc)} (max diff {np.abs(p_user - p_fc).max():.3f}) | "
                  f"X_train dtypes x_c={X_train['x_c'].dtype}")


@probe("P03")
def p03():
    """Text that looks numeric, object columns of numbers, with None or an incomplete list."""
    series, exog, exog_fut, *_ = C.make_data("cat_int_vals", n=120)
    vals = np.asarray(exog["x_c"]).astype(int)
    vals_f = np.asarray(exog_fut["x_c"]).astype(int)
    variants = {
        "object of ints": (pd.Series(vals, index=exog.index, dtype=object), pd.Series(vals_f, index=exog_fut.index, dtype=object)),
        "object of numeric strings": (pd.Series(vals.astype(str), index=exog.index, dtype=object), pd.Series(vals_f.astype(str), index=exog_fut.index, dtype=object)),
        "default string dtype of numeric strings": (pd.Series(vals.astype(str).tolist(), index=exog.index), pd.Series(vals_f.astype(str).tolist(), index=exog_fut.index)),
        "category of numeric strings": (pd.Series(pd.Categorical(vals.astype(str)), index=exog.index), pd.Series(pd.Categorical(vals_f.astype(str), categories=["10", "20", "30"]), index=exog_fut.index)),
        "category of bool": (pd.Series(pd.Categorical(vals > 10), index=exog.index), pd.Series(pd.Categorical(vals_f > 10, categories=[False, True]), index=exog_fut.index)),
        "category int with NaN": (pd.Series(pd.Categorical(np.where(vals == 30, np.nan, vals), categories=[10, 20]), index=exog.index), pd.Series(pd.Categorical(np.where(vals_f == 30, np.nan, vals_f), categories=[10, 20]), index=exog_fut.index)),
        "category uint8": (pd.Series(pd.Categorical(vals.astype("uint8")), index=exog.index), pd.Series(pd.Categorical(vals_f.astype("uint8"), categories=np.array([10, 20, 30], dtype="uint8")), index=exog_fut.index)),
        "category Int64 categories": (pd.Series(pd.Categorical(pd.array(vals, dtype="Int64")), index=exog.index), pd.Series(pd.Categorical(pd.array(vals_f, dtype="Int64"), categories=pd.array([10, 20, 30], dtype="Int64")), index=exog_fut.index)),
        "float32": (pd.Series(vals.astype("float32"), index=exog.index), pd.Series(vals_f.astype("float32"), index=exog_fut.index)),
        "datetime tz": (pd.Series(pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(vals, "D"), index=exog.index), pd.Series(pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(vals_f, "D"), index=exog_fut.index)),
        "timedelta": (pd.Series(pd.to_timedelta(vals, "D"), index=exog.index), pd.Series(pd.to_timedelta(vals_f, "D"), index=exog_fut.index)),
        "period": (pd.Series(pd.PeriodIndex(pd.Timestamp("2020-01-01") + pd.to_timedelta(vals, "D"), freq="D"), index=exog.index), pd.Series(pd.PeriodIndex(pd.Timestamp("2020-01-01") + pd.to_timedelta(vals_f, "D"), freq="D"), index=exog_fut.index)),
        "sparse float": (pd.Series(pd.arrays.SparseArray(vals.astype(float)), index=exog.index), pd.Series(pd.arrays.SparseArray(vals_f.astype(float)), index=exog_fut.index)),
        "complex": (pd.Series(vals.astype(complex), index=exog.index), pd.Series(vals_f.astype(complex), index=exog_fut.index)),
    }
    for vname, (col, col_f) in variants.items():
        ex = exog.assign(x_c=col)
        exf = exog_fut.assign(x_c=col_f)
        for cf in [None, ["x_k"]]:
            for est_name in ["lgbm", "ridge"]:
                f = ForecasterRecursiveMultiSeries(C.make_estimator(est_name), lags=3, categorical_features=cf)
                r, ws = run(lambda: f.fit(series=series, exog=ex))
                p = "-"
                if r is None:
                    p, _ = run(lambda: f.predict(steps=2, exog=exf))
                    p = fmt(p)
                wsk = [w.split(":")[0] for w in ws if "DataType" in w]
                print(f"  {vname:40} dtype={str(ex['x_c'].dtype)[:18]:18} cf={str(cf):8} {est_name:5} fit={'ok' if r is None else r[:150]} | pred={str(p)[:120]} | {wsk}")


@probe("P04")
def p04():
    """Does the matrix given to the estimator share memory with the data of the user? In-place estimators."""
    class Writer(LinearRegression):
        def fit(self, X, y, sample_weight=None):
            self.shared_ = {k: bool(np.shares_memory(np.asarray(X), v)) for k, v in Writer.user.items()}
            self.writeable_ = getattr(np.asarray(X).flags, "writeable", None) if not isinstance(X, pd.DataFrame) else "DataFrame"
            self.type_ = type(X).__name__
            return super().fit(X, y, sample_weight=sample_weight)

    for dtype_name in ["float", "int", "mixed"]:
        for enc in C.ENCODINGS:
            for with_exog in (True, False):
                series, exog, exog_fut, *_ = C.make_data(dtype_name)
                if dtype_name == "mixed":
                    exog = exog.drop(columns="x_s")
                Writer.user = {"series": series.to_numpy(), "exog": exog["x_f"].to_numpy()}
                s0, e0 = series.copy(), exog.copy()
                f = ForecasterRecursiveMultiSeries(Writer(), lags=3, encoding=enc)
                r, _ = run(lambda: f.fit(series=series, exog=exog if with_exog else None))
                print(f"  dtype={dtype_name:6} enc={str(enc):17} exog={with_exog!s:5} fit={'ok' if r is None else r[:60]} "
                      f"X type={getattr(f.estimator, 'type_', '?')} writeable={getattr(f.estimator, 'writeable_', '?')} "
                      f"shares memory with user data={getattr(f.estimator, 'shared_', '?')} "
                      f"user data intact={series.equals(s0) and exog.equals(e0)}")
    # in-place estimators, with and without the in-sample residuals
    for est_name, mk in {"LinearRegression(copy_X=False)": lambda: LinearRegression(copy_X=False),
                         "pipe StandardScaler(copy=False)": lambda: make_pipeline(StandardScaler(copy=False), LinearRegression())}.items():
        for dtype_name in ["float", "int"]:
            for prob in (True, False):
                series, exog, exog_fut, *_ = C.make_data(dtype_name)
                s0, e0 = series.copy(), exog.copy()
                f = ForecasterRecursiveMultiSeries(mk(), lags=3)
                if not prob:
                    f._probabilistic_mode = False
                r, _ = run(lambda: f.fit(series=series, exog=exog))
                p = "-"
                if r is None:
                    p, _ = run(lambda: f.predict(steps=2, exog=exog_fut))
                    p = fmt(p)
                print(f"  {est_name:32} dtype={dtype_name:5} residuals={prob!s:5} fit={'ok' if r is None else r[:70]} pred={p} user data intact={series.equals(s0) and exog.equals(e0)}")


@probe("P05")
def p05():
    """Pipeline that selects columns by name; estimator attributes with names."""
    series, exog, exog_fut, *_ = C.make_data("float")
    pipe = make_pipeline(
        make_column_transformer((StandardScaler(), ["x_f"]), remainder="passthrough"),
        Ridge()
    )
    f = ForecasterRecursiveMultiSeries(pipe, lags=3)
    r, _ = run(lambda: f.fit(series=series, exog=exog))
    p = "-"
    if r is None:
        p, _ = run(lambda: f.predict(steps=2, exog=exog_fut))
        p = fmt(p)
    print(f"  ColumnTransformer by name: fit={'ok' if r is None else r[:150]} | predict={p}")
    for est in [lgbm(), XGBRegressor(n_estimators=5), Ridge(), HistGradientBoostingRegressor(max_iter=5)]:
        f = ForecasterRecursiveMultiSeries(est, lags=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.fit(series=series, exog=exog)
        e = f.estimator
        names = getattr(e, "feature_names_in_", None)
        extra = ""
        if hasattr(e, "feature_name_"):
            extra = f" feature_name_={e.feature_name_}"
        if hasattr(e, "get_booster"):
            extra = f" booster.feature_names={e.get_booster().feature_names}"
        print(f"  {type(e).__name__:30} feature_names_in_={None if names is None else list(names)}{extra}")
        imp, _ = run(lambda: f.get_feature_importances())
        print(f"      get_feature_importances features: {imp if isinstance(imp, str) else imp['feature'].tolist()}")


@probe("P06")
def p06():
    """State: error in the middle of fit, second fit, set_params, with 'auto' and with None."""
    series, exog, exog_fut, *_ = C.make_data("cat_int_vals", n=120)

    def state(est):
        p = est.get_params()
        return {k: p[k] for k in ("feature_types", "enable_categorical", "categorical_features") if k in p}

    for est_name, mk in {"xgb": lambda: XGBRegressor(n_estimators=5, n_jobs=1),
                         "hgb": lambda: HistGradientBoostingRegressor(max_iter=5)}.items():
        for cf in ["auto", None]:
            f = ForecasterRecursiveMultiSeries(mk(), lags=3, encoding="ordinal_category", categorical_features=cf)
            s0 = state(f.estimator)
            bad = exog.iloc[:50]  # exog shorter than series -> error before the estimator
            r, _ = run(lambda: f.fit(series=series, exog=bad))
            s_err1 = state(f.estimator)
            # error inside the estimator: y with inf
            cls = type(f.estimator)
            orig = cls.fit

            def boom(self, X, y, **kw):
                raise RuntimeError("boom")
            cls.fit = boom
            try:
                r2, _ = run(lambda: f.fit(series=series, exog=exog))
            finally:
                cls.fit = orig
            s_err2 = state(f.estimator)
            r3, w3 = run(lambda: f.fit(series=series, exog=exog))
            s1 = state(f.estimator)
            p1, _ = run(lambda: f.predict(steps=2, exog=exog_fut))
            r4, w4 = run(lambda: f.fit(series=series, exog=exog))
            p2, _ = run(lambda: f.predict(steps=2, exog=exog_fut))
            f.set_lags(5)
            r5, w5 = run(lambda: f.fit(series=series, exog=exog))
            s5 = state(f.estimator)
            p5, _ = run(lambda: f.predict(steps=2, exog=exog_fut))
            f.set_params({"random_state": 3})
            s6 = state(f.estimator)
            r6, w6 = run(lambda: f.fit(series=series, exog=exog))
            print(f"  {est_name} cf={cf}: initial {s0}\n     after error inside estimator.fit: {s_err2}\n     after fit: {s1}\n"
                  f"     2nd fit same pred: {fmt(p1) == fmt(p2)} warnings 2nd fit: {[w for w in w4 if 'Ignored' in w]}\n"
                  f"     after set_lags(5)+fit: {r5 if r5 else 'ok'} {s5} pred={fmt(p5)} warnings: {[w[:60] for w in w5 if 'Ignored' in w]}\n"
                  f"     after set_params: {s6}; fit: {r6 if r6 else 'ok'}")


@probe("P07")
def p07():
    """No exog, the four encodings; unknown level in predict; levels subset."""
    series, exog, exog_fut, *_ = C.make_data("float", n=120)
    lw = series.iloc[-3:].rename(columns={"s_a": "new"})[["new"]]
    for est_name in ["ridge", "lgbm", "xgb", "hgb", "catboost"]:
        for enc in C.ENCODINGS:
            f = ForecasterRecursiveMultiSeries(C.make_estimator(est_name), lags=3, encoding=enc)
            r, _ = run(lambda: f.fit(series=series))
            p, _ = run(lambda: f.predict(steps=2))
            pu, _ = run(lambda: f.predict(steps=2, levels=["new"], last_window=lw))
            print(f"  {est_name:8} enc={str(enc):17} fit={'ok' if r is None else r[:60]} pred={fmt(p)} unknown level={fmt(pu)}")


@probe("P08")
def p08():
    """Exog as dict where one series has no exog, or lacks a column."""
    series, exog, exog_fut, *_ = C.make_data("mixed", as_dict=True)
    for case in ["series without exog", "series without one column"]:
        ex = dict(exog)
        exf = dict(exog_fut)
        if case == "series without exog":
            ex["s_c"] = None
            exf["s_c"] = None
        else:
            ex["s_c"] = ex["s_c"].drop(columns=["x_i"])
            exf["s_c"] = exf["s_c"].drop(columns=["x_i"])
        for est_name in ["lgbm", "hgb", "xgb"]:
            for cf in ["auto", ["x_s", "x_c"]]:
                f = ForecasterRecursiveMultiSeries(C.make_estimator(est_name), lags=3, categorical_features=cf)
                r, ws = run(lambda: f.fit(series=series, exog=ex))
                p = "-"
                if r is None:
                    p, _ = run(lambda: f.predict(steps=2, exog=exf))
                    p = fmt(p)
                print(f"  {case:26} {est_name:5} cf={str(cf):15} fit={'ok' if r is None else r[:130]} pred={p}")


@probe("P09")
def p09():
    """Weights given through weight_func and series_weights reach the estimator."""
    series, exog, exog_fut, *_ = C.make_data("cat_int_vals", n=120)
    seen = {}
    orig = LGBMRegressor.fit

    def spy(self, X, y, sample_weight=None, **kw):
        seen["sw"] = None if sample_weight is None else (float(np.sum(sample_weight)), len(sample_weight))
        seen["type"] = type(X).__name__
        seen["kw"] = {k: v for k, v in kw.items()}
        return orig(self, X, y, sample_weight=sample_weight, **kw)

    for cf in ["auto", None]:
        f = ForecasterRecursiveMultiSeries(
            lgbm(), lags=3, categorical_features=cf, encoding="ordinal_category",
            weight_func=lambda idx: np.where(idx > idx[50], 3.0, 1.0),
            series_weights={"s_a": 2.0, "s_b": 1.0, "s_c": 1.0},
        )
        LGBMRegressor.fit = spy
        try:
            r, _ = run(lambda: f.fit(series=series, exog=exog))
        finally:
            LGBMRegressor.fit = orig
        print(f"  cf={cf}: fit={'ok' if r is None else r[:80]} X type={seen.get('type')} sample_weight (sum, n)={seen.get('sw')} fit kwargs={seen.get('kw')}")


@probe("P10")
def p10():
    """Category 0..5 (values == codes) with None: the model that changes without changing code."""
    rng = np.random.default_rng(11)
    n, n_fut = 400, 30
    idx = pd.date_range("2021-01-01", periods=n + n_fut, freq="D")
    codes = rng.integers(0, 6, size=n + n_fut)
    eff = np.array([0.0, 30.0, -20.0, 12.0, -35.0, 22.0])[codes]
    x_f = rng.normal(size=n + n_fut)
    series_all = pd.DataFrame(
        {s: 10 * (i + 1) + eff + 2 * x_f + rng.normal(scale=0.3, size=n + n_fut)
         for i, s in enumerate(["s_a", "s_b", "s_c"])}, index=idx)
    exog_all = pd.DataFrame({"x_f": x_f, "x_c": pd.Categorical(codes, categories=range(6))}, index=idx)
    series, truth = series_all.iloc[:n], series_all.iloc[n:]
    exog, exog_fut = exog_all.iloc[:n], exog_all.iloc[n:]
    for est_name in ["lgbm", "xgb", "hgb", "catboost"]:
        for enc in ["ordinal", "ordinal_category"]:
            out = {}
            for cf in ["auto", None]:
                est = C.make_estimator(est_name)
                if est_name == "lgbm":
                    est.set_params(n_estimators=8, num_leaves=4)
                if est_name == "xgb":
                    est.set_params(n_estimators=8, max_depth=2)
                if est_name == "hgb":
                    est.set_params(max_iter=8, max_leaf_nodes=4)
                if est_name == "catboost":
                    est.set_params(iterations=8, depth=2)
                f = ForecasterRecursiveMultiSeries(est, lags=3, encoding=enc, categorical_features=cf)
                r, _ = run(lambda: f.fit(series=series, exog=exog))
                if r is not None:
                    out[cf] = r[:80]
                    continue
                p, _ = run(lambda: f.predict(steps=n_fut, exog=exog_fut))
                if isinstance(p, str):
                    out[cf] = p[:80]
                    continue
                wide = p.pivot(columns="level", values="pred")
                out[cf] = (round(float(np.abs(wide.to_numpy() - truth.to_numpy()).mean()), 4), p["pred"].to_numpy())
            same = (
                isinstance(out["auto"], tuple) and isinstance(out[None], tuple)
                and np.array_equal(out["auto"][1], out[None][1])
            )
            show = {k: (v[0] if isinstance(v, tuple) else v) for k, v in out.items()}
            print(f"  {est_name:8} enc={enc:17} MAE auto={show['auto']} None={show[None]} | None identical to auto: {same}")
