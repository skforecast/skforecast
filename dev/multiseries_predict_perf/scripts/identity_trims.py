"""
Outputs, warnings and errors of predict and backtesting for the tree on
PYTHONPATH, to compare the base tree with the branch (identity_compare.py).

Usage: python identity_trims.py <out.pkl>      (NO_PYARROW=1 blocks pyarrow)
"""
import os
import sys

if os.environ.get("NO_PYARROW") == "1":
    class _BlockPyarrow:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")
            return None

    assert "pyarrow" not in sys.modules and "pandas" not in sys.modules
    sys.meta_path.insert(0, _BlockPyarrow)

import pickle
import warnings

import numpy as np
import pandas as pd
import skforecast
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.model_selection import (
    TimeSeriesFold, backtesting_forecaster, backtesting_forecaster_multiseries,
)
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import (
    ForecasterEquivalentDate, ForecasterRecursive, ForecasterRecursiveClassifier,
    ForecasterRecursiveMultiSeries,
)

N = 220
T = 170  # end of the training data in the predict cases


def idx_of(kind, n=N):
    if kind == "dt":
        return pd.date_range("2020-01-01", periods=n, freq="h")
    if kind == "tz_h":  # hourly, crosses the DST change of 2020-03-29
        return pd.date_range("2020-03-22", periods=n, freq="h", tz="Europe/Madrid")
    if kind == "tz_D":  # local calendar days, crosses the DST change of October
        return pd.date_range("2019-06-01", periods=n, freq="D", tz="Europe/Madrid")
    if kind == "utc_D":  # UTC-anchored days shown in a DST timezone
        return pd.date_range("2019-06-01", periods=n, freq="D", tz="UTC").tz_convert(
            "Europe/Madrid"
        )
    if kind == "utc6_D":  # days anchored at 06:00 UTC shown in a DST timezone
        return pd.date_range("2019-03-20 06:00", periods=n, freq="D", tz="UTC").tz_convert(
            "Europe/Madrid"
        )
    if kind == "D":
        return pd.date_range("2015-01-01", periods=n, freq="D")
    if kind == "MS":
        return pd.date_range("1990-01-01", periods=n, freq="MS")
    if kind == "range":
        return pd.RangeIndex(n)
    if kind == "range5":
        return pd.RangeIndex(start=10, stop=10 + 5 * n, step=5)
    raise ValueError(kind)


def exog_of(kind, idx):
    rng = np.random.default_rng(7)
    n = len(idx)
    if kind is None:
        return None
    if kind == "num":
        return pd.DataFrame(rng.normal(size=(n, 3)), index=idx, columns=["e1", "e2", "e3"])
    if kind == "series":
        return pd.Series(rng.normal(size=n), index=idx, name="e1")
    if kind == "float32":
        return pd.DataFrame(
            rng.normal(size=(n, 2)).astype("float32"), index=idx, columns=["e1", "e2"]
        )
    if kind == "int":
        return pd.DataFrame(rng.integers(0, 10, size=(n, 2)), index=idx, columns=["e1", "e2"])
    if kind == "mixed":
        return pd.DataFrame({
            "ef": rng.normal(size=n),
            "ei": rng.integers(0, 10, size=n),
            "ef32": rng.normal(size=n).astype("float32"),
            "eb": rng.integers(0, 2, size=n).astype(bool),
            "ecat": pd.Categorical(rng.integers(0, 4, size=n)),
        }, index=idx)
    if kind == "strcat":
        return pd.DataFrame({
            "ef": rng.normal(size=n),
            "ecat": pd.Categorical(np.array(["a", "b", "c", "d"])[rng.integers(0, 4, size=n)]),
        }, index=idx)
    if kind == "nullable":
        return pd.DataFrame({
            "ef": pd.array(rng.normal(size=n), dtype="Float64"),
            "ei": pd.array(rng.integers(0, 10, size=n), dtype="Int64"),
        }, index=idx)
    raise ValueError(kind)


def data_of(fkind, idx):
    rng = np.random.default_rng(3)
    n = len(idx)
    if fkind == "classifier":
        return pd.Series(rng.integers(0, 3, size=n), index=idx, name="y")
    if fkind in ("direct_mv", "multiseries"):
        return pd.DataFrame(
            np.cumsum(rng.normal(size=(n, 3)), axis=0) + 50, index=idx, columns=["s1", "s2", "s3"]
        )
    return pd.Series(np.cumsum(rng.normal(size=n)) + 50, index=idx, name="y")


CFGS = {
    "base": {},
    "tr_exog": {"transformer_exog": StandardScaler()},
    "diff": {"differentiation": 1},
    "wf": {"window_features": RollingFeatures(stats=["mean"], window_sizes=7)},
}


def make(fkind, cfg="base", lgbm=False, steps=3):
    kw = dict(CFGS[cfg])
    est = LGBMRegressor(n_estimators=5, verbose=-1, random_state=1) if lgbm else Ridge()
    if lgbm:
        kw["categorical_features"] = "auto"
    if fkind == "recursive":
        return ForecasterRecursive(estimator=est, lags=5, **kw)
    if fkind == "classifier":
        kw.pop("categorical_features", None)
        est_clf = (
            LGBMClassifier(n_estimators=5, verbose=-1, random_state=1) if lgbm
            else LogisticRegression(max_iter=200)
        )
        return ForecasterRecursiveClassifier(estimator=est_clf, lags=5, **kw)
    if fkind == "direct":
        return ForecasterDirect(estimator=est, lags=5, steps=steps, **kw)
    if fkind == "direct_mv":
        return ForecasterDirectMultiVariate(
            estimator=est, level="s1", lags=5, steps=steps, **kw
        )
    if fkind == "multiseries":
        return ForecasterRecursiveMultiSeries(estimator=est, lags=5, **kw)
    raise ValueError(fkind)


def mutate_exog(ex, mut, steps):
    """`ex` holds `steps + 6` rows from the first step to predict."""
    if ex is None or mut == "long":
        return ex
    if mut == "exact":
        return ex.iloc[:steps]
    if mut == "nan_in":
        ex = ex.iloc[:steps].copy()
        if isinstance(ex, pd.Series):
            ex.iloc[0] = np.nan
        else:
            ex.iloc[0, 0] = np.nan
        return ex
    if mut == "nan_beyond":
        ex = ex.copy()
        if isinstance(ex, pd.Series):
            ex.iloc[-1] = np.nan
        else:
            ex.iloc[-1, 0] = np.nan
        return ex
    if mut == "reordered":
        return ex.iloc[:steps, ::-1] if isinstance(ex, pd.DataFrame) else ex.iloc[:steps]
    if mut == "int_cast":
        ex = ex.iloc[:steps].copy()
        if isinstance(ex, pd.DataFrame):
            col = ex.columns[0]
            ex[col] = ex[col].astype(float).round().astype("int64")
        return ex
    if mut == "strcat_cast":
        ex = ex.iloc[:steps].copy()
        if isinstance(ex, pd.DataFrame):
            col = ex.columns[0]
            ex[col] = pd.Categorical([f"{v:.1f}" for v in ex[col].astype(float)])
        return ex
    if mut == "shifted":
        return ex.iloc[1:steps + 1]
    if mut == "short":
        return ex.iloc[:max(steps - 1, 0)]
    if mut == "extra_col":
        ex = ex.iloc[:steps].copy()
        if isinstance(ex, pd.DataFrame):
            ex["zz"] = 1.0
        return ex
    if mut == "missing_col":
        return ex.iloc[:steps, 1:] if isinstance(ex, pd.DataFrame) else ex.iloc[:steps]
    raise ValueError(mut)


RESULTS = {}


def record(name, fn):
    assert name not in RESULTS, name
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out, err = fn(), None
        except Exception as e:  # noqa: BLE001
            out, err = None, (type(e).__name__, str(e))
    RESULTS[name] = {
        "out": out, "err": err,
        "warn": [(x.category.__name__, str(x.message)) for x in w],
    }


def fit_kwargs(fkind, data, exog, end):
    key = "series" if fkind in ("direct_mv", "multiseries") else "y"
    return {key: data.iloc[:end], "exog": None if exog is None else exog.iloc[:end]}


def predict_cases():
    fkinds = ["recursive", "classifier", "direct", "direct_mv", "multiseries"]
    for fkind in fkinds:
        steps_list = [1, 3] if fkind in ("recursive", "classifier", "multiseries") else [None, 2, [1, 3]]
        cfgs = ["base", "tr_exog"] if fkind == "classifier" else ["base", "tr_exog", "diff", "wf"]

        def run(name, ikind, ekind, cfg, lw, steps, mut, method="predict", lgbm=False,
                check_inputs=True, levels=None, exog_fmt="wide", n=N, T=T):
            idx = idx_of(ikind, n)
            data, exog = data_of(fkind, idx), exog_of(ekind, idx)
            # An estimator that accepts NaN, to compare the whole output.
            lgbm = lgbm or mut == "nan_in"

            def fn():
                f = make(fkind, cfg, lgbm=lgbm)
                f.fit(**fit_kwargs(fkind, data, exog, T))
                ws = f.window_size
                n_steps = 3 if steps is None else (max(steps) if isinstance(steps, list) else steps)
                ex = None if exog is None else mutate_exog(exog.iloc[T:T + n_steps + 6], mut, n_steps)
                if ex is not None and fkind == "multiseries" and exog_fmt == "dict":
                    ex = {"s1": ex, "s2": ex.iloc[:-1] if len(ex) > 1 else ex, "s3": None}
                last_window = {
                    "stored": None, "exact": data.iloc[T - ws:T], "longer": data.iloc[T - 3 * ws:T],
                    "all": data.iloc[:T],
                }[lw]
                kw = dict(steps=steps, last_window=last_window, exog=ex)
                if fkind == "multiseries":
                    kw["levels"] = levels
                if method == "predict":
                    if not check_inputs:
                        kw["check_inputs"] = False
                    return f.predict(**kw)
                if method == "create_predict_X":
                    return f.create_predict_X(**kw)
                if method == "interval_boot":
                    f.fit(store_in_sample_residuals=True, **fit_kwargs(fkind, data, exog, T))
                    return f.predict_interval(method="bootstrapping", n_boot=10, interval=[0.1, 0.9], **kw)
                if method == "interval_conformal":
                    f.fit(store_in_sample_residuals=True, **fit_kwargs(fkind, data, exog, T))
                    return f.predict_interval(method="conformal", interval=0.8, **kw)
                if method == "proba":
                    return f.predict_proba(**kw)
                raise ValueError(method)

            record(f"P|{fkind}|{name}", fn)

        # 1. Index kinds x window x steps, without and with numeric exog.
        for ikind in ["dt", "tz_h", "tz_D", "utc_D", "D", "MS", "range", "range5"]:
            for lw in ["stored", "exact", "longer", "all"]:
                for steps in steps_list:
                    for ekind in [None, "num"]:
                        run(f"idx:{ikind} lw:{lw} steps:{steps} exog:{ekind}",
                            ikind, ekind, "base", lw, steps, "exact")
        # 1b. Window that contains a DST change, predictions that cross the next
        # one: the full window and its last rows identify different conventions.
        for lw in ["stored", "exact", "longer", "all"]:
            for ekind in [None, "num"]:
                for end in [218, 219, 220]:
                    run(f"idx:utc6_D end:{end} lw:{lw} exog:{ekind}", "utc6_D", ekind, "base",
                        lw, 3, "exact", n=240, T=end)
        # 2. Exog kinds and mutations.
        for ekind in ["num", "series", "float32", "int", "mixed", "strcat", "nullable"]:
            for mut in ["long", "nan_in", "nan_beyond", "reordered", "int_cast", "strcat_cast",
                        "shifted", "short", "extra_col", "missing_col"]:
                for lw in ["stored", "longer"]:
                    run(f"exog:{ekind} mut:{mut} lw:{lw}", "dt", ekind, "base", lw,
                        steps_list[-1], mut, lgbm=ekind in ("mixed", "strcat", "nullable"))
        # 3. Forecaster configurations.
        for cfg in cfgs:
            for ekind in [None, "num", "int"]:
                if cfg == "tr_exog" and ekind is None:
                    continue
                for lw in ["stored", "longer"]:
                    for ikind in ["dt", "tz_D", "range"]:
                        run(f"cfg:{cfg} exog:{ekind} lw:{lw} idx:{ikind}", ikind, ekind, cfg, lw,
                            steps_list[-1], "nan_beyond")
        # 4. Other methods and `check_inputs=False`.
        for ikind in ["dt", "tz_D", "range"]:
            for lw in ["stored", "longer"]:
                for mut in ["exact", "nan_in"]:
                    tag = f"idx:{ikind} lw:{lw} mut:{mut}"
                    if fkind != "classifier":
                        run(f"no_check {tag}", ikind, "num", "base", lw, steps_list[-1], mut,
                            check_inputs=False)
                        run(f"X {tag}", ikind, "num", "base", lw, steps_list[-1], mut,
                            method="create_predict_X")
                        run(f"boot {tag}", ikind, "num", "base", lw, steps_list[-1], mut,
                            method="interval_boot")
                        run(f"conformal {tag}", ikind, "num", "base", lw, steps_list[-1], mut,
                            method="interval_conformal")
                    else:
                        run(f"proba {tag}", ikind, "num", "base", lw, steps_list[-1], mut,
                            method="proba")
        # 5. Multiseries: levels and exog formats.
        if fkind == "multiseries":
            for levels in ["s1", ["s1", "s3"], None]:
                for fmt in ["wide", "dict"]:
                    for lw in ["stored", "exact", "longer"]:
                        for ikind in ["dt", "tz_D", "utc_D", "range"]:
                            for mut in ["exact", "nan_in", "shifted"]:
                                run(f"levels:{levels} fmt:{fmt} lw:{lw} idx:{ikind} mut:{mut}",
                                    ikind, "num", "base", lw, 3, mut, levels=levels, exog_fmt=fmt)


def backtesting_cases():
    def run(name, fkind, ikind, ekind, cv_kw, bt_kw=None, cfg="base", steps=3, mut=None,
            lgbm=False):
        idx = idx_of(ikind)
        data = data_of(fkind if fkind != "equivalent_date" else "recursive", idx)
        exog = exog_of(ekind, idx)
        if exog is not None and mut == "nan":
            exog = exog.copy()
            exog.iloc[N - 20, 0] = np.nan
        if mut == "nan_y":
            data = data.copy().astype(float)
            data.iloc[N - 20] = np.nan

        def fn():
            cv = TimeSeriesFold(**{"steps": steps, "initial_train_size": N - 60, **cv_kw})
            kw = {"cv": cv, "exog": exog, "n_jobs": 1, "verbose": False,
                  "show_progress": False, **(bt_kw or {})}
            if fkind == "equivalent_date":
                f = ForecasterEquivalentDate(offset=7, n_offsets=2)
                return backtesting_forecaster(forecaster=f, y=data, metric="mean_absolute_error", **kw)
            f = make(fkind, cfg, lgbm=lgbm, steps=max(steps, cv_kw.get("gap", 0) + steps))
            if fkind in ("direct_mv", "multiseries"):
                return backtesting_forecaster_multiseries(
                    forecaster=f, series=data, metric="mean_absolute_error", **kw
                )
            metric = "accuracy_score" if fkind == "classifier" else ["mean_absolute_error", "mean_squared_error"]
            return backtesting_forecaster(forecaster=f, y=data, metric=metric, **kw)

        record(f"B|{fkind}|{name}", fn)

    cvs = {
        "plain": {},
        "refit": {"refit": True},
        "refit2": {"refit": 2},
        "gap": {"gap": 2},
        "fixed": {"refit": True, "fixed_train_size": True},
        "incomplete": {"allow_incomplete_fold": False},
        "skip": {"skip_folds": 2},
        "stride": {"fold_stride": 1},
        "one_fold": {"initial_train_size": N - 3},
        "no_initial": {"initial_train_size": None},
    }
    for fkind in ["recursive", "direct", "classifier", "equivalent_date", "direct_mv", "multiseries"]:
        for ikind in ["dt", "tz_h", "tz_D", "utc_D", "MS", "range", "range5"]:
            for ekind in [None, "num"]:
                if fkind == "equivalent_date" and (ekind is not None or ikind == "MS"):
                    continue
                for steps in [1, 3, 7]:
                    run(f"idx:{ikind} exog:{ekind} steps:{steps}", fkind, ikind, ekind, {}, steps=steps)
        for cv_name, cv_kw in cvs.items():
            if cv_name == "no_initial":
                continue
            for ekind in [None, "num", "mixed"]:
                if fkind == "equivalent_date" and ekind is not None:
                    continue
                for ikind in ["dt", "range"]:
                    run(f"cv:{cv_name} exog:{ekind} idx:{ikind}", fkind, ikind, ekind, cv_kw)
        if fkind == "equivalent_date":
            continue
        for mut in ["nan", "nan_y"]:
            for cv_name in ["plain", "refit", "gap"]:
                for use_lgbm in [False, True]:
                    run(f"mut:{mut} cv:{cv_name} lgbm:{use_lgbm}", fkind, "dt", "num",
                        cvs[cv_name], mut=mut, lgbm=use_lgbm)
        for cfg in (["tr_exog"] if fkind == "classifier" else ["tr_exog", "diff", "wf"]):
            run(f"cfg:{cfg}", fkind, "dt", "num", {}, cfg=cfg)
        if fkind in ("recursive", "direct"):
            for ikind in ["dt", "range", "tz_D"]:
                run(f"interval boot idx:{ikind}", fkind, ikind, "num", {},
                    bt_kw=dict(interval=[0.1, 0.9], interval_method="bootstrapping", n_boot=10))
                run(f"interval conformal idx:{ikind}", fkind, ikind, "num", {},
                    bt_kw=dict(interval=0.8, interval_method="conformal"))
                run(f"return_predictors idx:{ikind}", fkind, ikind, "num", {},
                    bt_kw=dict(return_predictors=True))
            run("n_jobs 2", fkind, "dt", "num", {}, bt_kw=dict(n_jobs=2))


predict_cases()
backtesting_cases()
n_err = sum(r["err"] is not None for r in RESULTS.values())
n_warn = sum(bool(r["warn"]) for r in RESULTS.values())
with open(sys.argv[1], "wb") as fh:
    pickle.dump(RESULTS, fh)
print(f"pandas {pd.__version__} | {os.path.dirname(os.path.dirname(skforecast.__file__))[-24:]} | "
      f"{len(RESULTS)} cases, {n_err} errors, {n_warn} with warnings")
