"""
D-NP2: what every forecaster does today with a `category` exog column that
`categorical_features` does not handle (None, or a list that omits it), with
LightGBM. For each case: what the estimator receives in `fit`, whether the
model has categorical splits, and whether `predict` works.

Usage: bash run3.sh dnp2_today_matrix.py   (SK_TREE: a checkout of 0.28.x)
"""
import _env  # noqa: F401  (first: it can block pyarrow)
import re
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate

rng = np.random.default_rng(0)
n = 200
idx = pd.date_range("2020", periods=n, freq="D")
fidx = pd.date_range(idx[-1] + idx.freq, periods=3, freq="D")
y = pd.Series(rng.normal(size=n), index=idx, name="y")
series = pd.DataFrame({"a": y, "b": y + 1})

CATS = {
    "int 10,20,30": [10, 20, 30],
    "int 0,1,2": [0, 1, 2],
    "float 1.5,2.5,3.5": [1.5, 2.5, 3.5],
    "text": ["n", "s", "w"],
}


def exogs(cats, extra_text=False):
    pos = rng.integers(0, 3, n)
    e = pd.DataFrame({"x": rng.normal(size=n)}, index=idx)
    e["c"] = pd.Categorical(np.array(cats, dtype=object)[pos], categories=cats)
    ep = pd.DataFrame({"x": np.zeros(3)}, index=fidx)
    ep["c"] = pd.Categorical(cats, categories=cats)
    if extra_text:
        e["d"] = pd.Categorical(np.array(["u", "v", "w"], dtype=object)[pos])
        ep["d"] = pd.Categorical(["u", "v", "w"])
    return e, ep


def run(make, fit, predict):
    seen = {}
    orig = LGBMRegressor.fit

    def spy(self, X, y, **kw):
        seen["X"] = type(X).__name__ + (
            " " + ",".join(sorted({str(t) for t in X.dtypes})) if isinstance(X, pd.DataFrame)
            else " " + str(X.dtype))
        seen["kw"] = {k: v for k, v in kw.items() if k != "sample_weight"}
        return orig(self, X, y, **kw)

    LGBMRegressor.fit = spy
    out = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            f = make()
            fit(f)
            cat_model = ("categorical splits" if re.search(r"num_cat=[1-9]", f_trees(f))
                         else "numeric splits")
            out.append(f"fit ok [{seen.get('X')}; {cat_model}]")
            try:
                predict(f)
                out.append("predict ok")
            except Exception as e:  # noqa
                out.append(f"predict {type(e).__name__}: {str(e).splitlines()[0][:70]}")
        except Exception as e:  # noqa
            out.append(f"fit {type(e).__name__}: {str(e).splitlines()[0][:90]}")
        finally:
            LGBMRegressor.fit = orig
    ws = sorted({x.category.__name__ for x in w if "skforecast" in str(x.category)})
    return "; ".join(out) + (f"; warnings {ws}" if ws else "")


def f_trees(f):
    est = f.estimator if hasattr(f, "estimator") and not hasattr(f, "estimators_") else None
    if est is None or not hasattr(est, "booster_"):
        est = list(f.estimators_.values())[0]
    return est.booster_.model_to_string()


lgbm = lambda: LGBMRegressor(verbose=-1, n_estimators=20, min_child_samples=5, n_jobs=1)  # noqa
for label, cats in CATS.items():
    for cf_label, cf, extra in (("None", None, False), ("list omitting c", ["d"], True)):
        e, ep = exogs(cats, extra)
        print(f"\n{label} | categorical_features={cf_label}")
        print("  Recursive   :", run(
            lambda: ForecasterRecursive(lgbm(), lags=3, categorical_features=cf),
            lambda f: f.fit(y=y, exog=e), lambda f: f.predict(steps=3, exog=ep)))
        print("  Direct      :", run(
            lambda: ForecasterDirect(lgbm(), lags=3, steps=3, categorical_features=cf),
            lambda f: f.fit(y=y, exog=e), lambda f: f.predict(exog=ep)))
        print("  DirectMV    :", run(
            lambda: ForecasterDirectMultiVariate(lgbm(), level="a", lags=3, steps=3, categorical_features=cf),
            lambda f: f.fit(series=series, exog=e), lambda f: f.predict(exog=ep)))
        print("  MultiSeries :", run(
            lambda: ForecasterRecursiveMultiSeries(lgbm(), lags=3, categorical_features=cf),
            lambda f: f.fit(series=series, exog=e), lambda f: f.predict(steps=3, exog=ep)))
print(_env.versions())
