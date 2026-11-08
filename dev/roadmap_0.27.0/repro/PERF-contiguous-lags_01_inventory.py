"""
PERF-contiguous-lags, script 01: memory layout of the matrices the estimator
receives in each forecaster of `recursive/` and `direct/`.

A LinearRegression (LogisticRegression for the classifier) subclass records,
in every `fit` and `predict` call, the memory layout of `X` (and of `y` in
`fit`) as the estimator sees it (`np.asarray(X)` for a DataFrame, which is what
scikit-learn validates). The records are stored as attributes of the estimator,
so they also come back from the joblib workers of Direct and DMV.

Layout codes:
- `C`: C-contiguous (row-major), `F`: F-contiguous (column-major).
- `view(a,b)`: neither C nor F contiguous, strides `(a, b)` in bytes.
- `df:` prefix: the estimator receives a pandas DataFrame.

Run from the root of the repository. By default it reports the layout of
`0.27.x` (the flag `_CONTIGUOUS_X_TRAIN` of the prototype off, if present);
with `prototype`, the layout with the flag on (C-contiguous copy):
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_01_inventory.py
    python dev/roadmap_0.27.0/repro/PERF-contiguous-lags_01_inventory.py prototype
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import (
    ForecasterRecursive,
    ForecasterRecursiveClassifier,
    ForecasterRecursiveMultiSeries,
)

import skforecast.direct._forecaster_direct as m_direct
import skforecast.direct._forecaster_direct_multivariate as m_dmv
import skforecast.recursive._forecaster_recursive as m_rec
import skforecast.recursive._forecaster_recursive_classifier as m_cls

warnings.filterwarnings("ignore")

PROTOTYPE = len(sys.argv) > 1 and sys.argv[1] == "prototype"
for _m in (m_rec, m_cls, m_direct, m_dmv):
    if hasattr(_m, "_CONTIGUOUS_X_TRAIN"):
        _m._CONTIGUOUS_X_TRAIN = PROTOTYPE
print(f"Layout {'with the prototype' if PROTOTYPE else 'of 0.27.x'}")


def layout(a) -> str:
    prefix = ""
    if isinstance(a, (pd.DataFrame, pd.Series)):
        prefix = "df:"
        a = np.asarray(a)
    if a.ndim == 1:
        return prefix + ("C" if a.flags["C_CONTIGUOUS"] else f"view({a.strides[0]})")
    if a.flags["C_CONTIGUOUS"]:
        return prefix + "C"
    if a.flags["F_CONTIGUOUS"]:
        return prefix + "F"
    return prefix + f"view({a.strides[0]},{a.strides[1]})"


class _Recorder:
    def _rec(self, name, value):
        log = getattr(self, "rec_", None)
        if log is None:
            log = []
            self.rec_ = log
        log.append((name, value))

    def fit(self, X, y, sample_weight=None):
        self.rec_ = []
        self._rec("fit_X", layout(X))
        self._rec("fit_y", layout(y))
        self._rec("fit_shape", np.asarray(X).shape)
        # The layout is already recorded: NaN (dropna_from_series=False) are
        # replaced so that the linear models can be fitted.
        return super().fit(np.nan_to_num(np.asarray(X, dtype=float)), y,
                           sample_weight=sample_weight)

    def predict(self, X):
        self._rec("predict_X", (layout(X), np.asarray(X).shape[0]))
        return super().predict(np.nan_to_num(np.asarray(X, dtype=float)))


class RecLinear(_Recorder, LinearRegression):
    pass


class RecLogistic(_Recorder, LogisticRegression):
    def predict_proba(self, X):
        self._rec("predict_X", (layout(X), np.asarray(X).shape[0]))
        return super().predict_proba(np.nan_to_num(np.asarray(X, dtype=float)))


def summarize(est):
    rec = getattr(est, "rec_", [])
    fit_X = [v for k, v in rec if k == "fit_X"]
    fit_y = [v for k, v in rec if k == "fit_y"]
    shape = [v for k, v in rec if k == "fit_shape"]
    pred = [v for k, v in rec if k == "predict_X"]
    # predict calls in fit (in-sample residuals) have as many rows as fit_X
    n_fit_rows = shape[0][0] if shape else None
    pred_in_fit = sorted({lay for lay, n in pred if n == n_fit_rows})
    return fit_X[0] if fit_X else "-", fit_y[0] if fit_y else "-", pred_in_fit


def predict_layouts(est, n_fit_rows):
    rec = getattr(est, "rec_", [])
    return sorted({lay for k, v in rec if k == "predict_X" for lay in [v[0]]
                   if v[1] != n_fit_rows})


# Data -----------------------------------------------------------------------
rng = np.random.default_rng(123)
n = 500
idx = pd.date_range("2020-01-01", periods=n, freq="h")
y_clean = pd.Series(rng.normal(size=n).cumsum() + 50, index=idx, name="y")
y_nan = y_clean.copy()
y_nan.iloc[[100, 200, 300]] = np.nan
exog_clean = pd.DataFrame(
    {"exog_1": rng.normal(size=n + 10), "exog_2": rng.normal(size=n + 10)},
    index=pd.date_range("2020-01-01", periods=n + 10, freq="h"),
)
exog_nan = exog_clean.copy()
exog_nan.iloc[[150, 250], 0] = np.nan
y_class = pd.Series(
    np.where(y_clean.diff().fillna(0) > 0, "up", "down"), index=idx, name="y"
)
y_class_nan = y_class.astype(object).copy()
y_class_nan.iloc[[100, 200, 300]] = np.nan

LAGS = 24
WF = RollingFeatures(stats=["mean", "std"], window_sizes=24)

CASES = [
    # name, kwargs of the forecaster, use NaN in y, exog (None, 'clean', 'nan')
    ("lags int", dict(lags=LAGS), False, None),
    ("lags list [1,2,3,24] (not contiguous)", dict(lags=[1, 2, 3, 24]), False, None),
    ("lags + window features", dict(lags=LAGS, window_features=WF), False, None),
    ("window features only", dict(lags=None, window_features=WF), False, None),
    ("lags + exog", dict(lags=LAGS), False, "clean"),
    ("lags, NaN in y, dropna False", dict(lags=LAGS, dropna_from_series=False), True, None),
    ("lags, NaN in y, dropna True", dict(lags=LAGS, dropna_from_series=True), True, None),
    ("lags + exog NaN, dropna False", dict(lags=LAGS, dropna_from_series=False), False, "nan"),
    ("lags + exog NaN, dropna True", dict(lags=LAGS, dropna_from_series=True), False, "nan"),
    ("lags, differentiation=1", dict(lags=LAGS, differentiation=1), False, None),
    ("lags, transformer_y", dict(lags=LAGS, transformer_y=StandardScaler()), False, None),
    ("lags, diff + transformer_y", dict(lags=LAGS, differentiation=1,
                                       transformer_y=StandardScaler()), False, None),
]


def get_exog(kind, length):
    if kind is None:
        return None
    e = exog_clean if kind == "clean" else exog_nan
    return e.iloc[:length]


rows = []


def run_single(name, Forecaster, kw, use_nan, exog_kind, extra=None):
    kw = dict(kw)
    extra = extra or {}
    is_class = Forecaster is ForecasterRecursiveClassifier
    if is_class:
        if "differentiation" in kw or "transformer_y" in kw:
            return
        kw.pop("differentiation", None)
        est = RecLogistic(max_iter=500)
        y = y_class_nan if use_nan else y_class
    else:
        est = RecLinear()
        y = y_nan if use_nan else y_clean
    if Forecaster in (ForecasterDirect,):
        kw.update(steps=3, n_jobs=extra.get("n_jobs", 1))
    f = Forecaster(estimator=est, **kw)
    exog_fit = get_exog(exog_kind, n)
    f.fit(y=y, exog=exog_fit, store_in_sample_residuals=not is_class)
    exog_pred = (
        get_exog(exog_kind, n + 10).iloc[n:] if exog_kind is not None else None
    )
    if Forecaster is ForecasterDirect:
        ests = f.estimators_
        if use_nan:
            pass
        else:
            f.predict(exog=exog_pred)
            f.predict_interval(exog=exog_pred, method="bootstrapping", n_boot=20)
        summ = {s: summarize(e) for s, e in ests.items()}
        fit_X = sorted({v[0] for v in summ.values()})
        fit_y = sorted({v[1] for v in summ.values()})
        pin = sorted({x for v in summ.values() for x in v[2]})
        nrows = {s: dict(e.rec_)["fit_shape"][0] for s, e in ests.items()}
        pred = sorted({x for s, e in ests.items() for x in predict_layouts(e, nrows[s])})
    else:
        if not use_nan:
            f.predict(steps=5, exog=exog_pred)
            if not is_class:
                f.predict_interval(steps=5, exog=exog_pred, method="bootstrapping",
                                   n_boot=20)
            else:
                f.predict_proba(steps=5, exog=exog_pred)
        fx, fy, pin = summarize(f.estimator)
        fit_X, fit_y = [fx], [fy]
        nrows = dict(f.estimator.rec_)["fit_shape"][0]
        pred = predict_layouts(f.estimator, nrows)
    rows.append(dict(
        forecaster=Forecaster.__name__ + (f" n_jobs={extra['n_jobs']}" if extra else ""),
        case=name,
        fit_X=" / ".join(fit_X),
        fit_y=" / ".join(fit_y),
        predict_in_fit=" / ".join(pin) or "-",
        predict=" / ".join(pred) if pred else "(not run)",
    ))


for Forecaster in (ForecasterRecursive, ForecasterRecursiveClassifier, ForecasterDirect):
    for name, kw, use_nan, exog_kind in CASES:
        run_single(name, Forecaster, kw, use_nan, exog_kind)

# Direct with joblib processes (default n_jobs='auto' with non-linear estimators)
run_single("lags int", ForecasterDirect, dict(lags=LAGS), False, None, {"n_jobs": 2})
run_single("lags + exog", ForecasterDirect, dict(lags=LAGS), False, "clean", {"n_jobs": 2})


# ForecasterDirectMultiVariate ------------------------------------------------
def make_series(nan=False):
    s = pd.DataFrame(
        {f"s{i}": rng.normal(size=n).cumsum() for i in range(3)}, index=idx
    )
    if nan:
        s.iloc[[100, 200], 0] = np.nan
    return s


DMV_CASES = [
    ("lags int (3 series)", dict(lags=LAGS), False, None),
    ("lags only in 1 series (dict)", dict(lags={"s0": LAGS, "s1": None, "s2": None}), False, None),
    ("lags + window features", dict(lags=LAGS, window_features=WF), False, None),
    ("lags + exog", dict(lags=LAGS), False, "clean"),
    ("lags, NaN in series, dropna False", dict(lags=LAGS, dropna_from_series=False), True, None),
    ("lags, NaN in series, dropna True", dict(lags=LAGS, dropna_from_series=True), True, None),
    ("lags, differentiation=1", dict(lags=LAGS, differentiation=1), False, None),
    ("lags, transformer_series", dict(lags=LAGS, transformer_series=StandardScaler()), False, None),
]

for name, kw, use_nan, exog_kind in DMV_CASES:
    for n_jobs in (1, 2):
        series = make_series(use_nan)
        f = ForecasterDirectMultiVariate(
            estimator=RecLinear(), level="s0", steps=3, n_jobs=n_jobs, **kw
        )
        f.fit(series=series, exog=get_exog(exog_kind, n))
        if not use_nan:
            exog_pred = get_exog(exog_kind, n + 10).iloc[n:] if exog_kind else None
            f.predict(exog=exog_pred)
        summ = {s: summarize(e) for s, e in f.estimators_.items()}
        nrows = {s: dict(e.rec_)["fit_shape"][0] for s, e in f.estimators_.items()}
        pred = sorted({x for s, e in f.estimators_.items()
                       for x in predict_layouts(e, nrows[s])})
        rows.append(dict(
            forecaster=f"ForecasterDirectMultiVariate n_jobs={n_jobs}",
            case=name,
            fit_X=" / ".join(sorted({v[0] for v in summ.values()})),
            fit_y=" / ".join(sorted({v[1] for v in summ.values()})),
            predict_in_fit=" / ".join(sorted({x for v in summ.values() for x in v[2]})) or "-",
            predict=" / ".join(pred) if pred else "(not run)",
        ))


# ForecasterRecursiveMultiSeries ---------------------------------------------
def make_series_dict(nan=False):
    d = {f"s{i}": pd.Series(rng.normal(size=n).cumsum(), index=idx, name=f"s{i}")
         for i in range(3)}
    if nan:
        d["s0"].iloc[[100, 200]] = np.nan
    return d


MS_CASES = [
    ("lags, encoding ordinal", dict(lags=LAGS), False, None),
    ("lags, encoding None", dict(lags=LAGS, encoding=None), False, None),
    ("lags, encoding onehot", dict(lags=LAGS, encoding="onehot"), False, None),
    ("lags + window features", dict(lags=LAGS, window_features=WF), False, None),
    ("lags + exog", dict(lags=LAGS), False, "clean"),
    ("lags + exog, encoding None", dict(lags=LAGS, encoding=None), False, "clean"),
    ("lags, NaN, dropna False", dict(lags=LAGS, dropna_from_series=False), True, None),
    ("lags, NaN, dropna True", dict(lags=LAGS, dropna_from_series=True), True, None),
    ("lags, differentiation=1", dict(lags=LAGS, differentiation=1), False, None),
]

for name, kw, use_nan, exog_kind in MS_CASES:
    f = ForecasterRecursiveMultiSeries(estimator=RecLinear(), **kw)
    series = make_series_dict(use_nan)
    f.fit(series=series, exog=get_exog(exog_kind, n))
    if not use_nan:
        exog_pred = get_exog(exog_kind, n + 10).iloc[n:] if exog_kind else None
        f.predict(steps=5, exog=exog_pred)
    fx, fy, pin = summarize(f.estimator)
    nrows = dict(f.estimator.rec_)["fit_shape"][0]
    pred = predict_layouts(f.estimator, nrows)
    rows.append(dict(
        forecaster="ForecasterRecursiveMultiSeries", case=name, fit_X=fx, fit_y=fy,
        predict_in_fit=" / ".join(pin) or "-",
        predict=" / ".join(pred) if pred else "(not run)",
    ))

df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.max_rows", 200)
for fc, g in df.groupby("forecaster", sort=False):
    print(f"\n## {fc}")
    g = g.drop(columns="forecaster")
    print("| " + " | ".join(g.columns) + " |")
    print("|" + "---|" * len(g.columns))
    for r in g.itertuples(index=False):
        print("| " + " | ".join(str(v) for v in r) + " |")
