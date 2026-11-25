"""
BUG-predict-readonly-wrapped-catboost: with CatBoost inside a wrapper that does
not copy the matrix (`TransformedTargetRegressor`, a Pipeline with no other
step or with an identity `FunctionTransformer`), `predict` and the
bootstrapping `predict_interval` of `ForecasterRecursiveMultiSeries`, and the
bootstrapping `predict_interval` of `ForecasterRecursive`, raise
`ValueError: assignment destination is read-only`. CatBoost leaves the array
read-only after `predict`, and `_build_predict_function` only restores the flag
in the path of a bare `CatBoostRegressor`.

Prints ok or the error of each call, with and without exog, for the bare
estimator and four wrappers. No categorical feature is needed. With a scaler
before CatBoost it works.

Usage: python readonly.py  (with `PYTHONPATH` set to the repository root if the
installed skforecast is not this code; the first line printed is the path of
the package used)
"""
import warnings
import numpy as np
import pandas as pd
import skforecast
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate

warnings.simplefilter("ignore")
print(skforecast.__file__, "pandas", pd.__version__)
rng = np.random.default_rng(123)
idx = pd.date_range("2020-01-01", periods=120, freq="D")
y = pd.Series(rng.normal(size=120), index=idx, name="y")
series = pd.DataFrame({"l1": rng.normal(size=120), "l2": rng.normal(size=120)}, index=idx)
exog = pd.DataFrame({"e": rng.normal(size=140)}, index=pd.date_range("2020-01-01", periods=140, freq="D"))
exog_tr, exog_pr = exog.iloc[:120], exog.iloc[120:]


def cb():
    return CatBoostRegressor(iterations=10, verbose=0, thread_count=1, random_seed=1, allow_writing_files=False)


ESTS = {
    "bare": cb,
    "pipe(cb)": lambda: make_pipeline(cb()),
    "pipe(identity,cb)": lambda: make_pipeline(FunctionTransformer(), cb()),
    "pipe(scaler,cb)": lambda: make_pipeline(StandardScaler(), cb()),
    "TTR(cb)": lambda: TransformedTargetRegressor(regressor=cb()),
}


def run(label, fn):
    try:
        fn()
        print(f"  {label:<34} ok")
    except Exception as e:
        print(f"  {label:<34} {type(e).__name__}: {str(e)[:70]}")


for name, make in ESTS.items():
    print(name)
    for with_exog in (False, True):
        ex_tr = exog_tr if with_exog else None
        ex_pr = exog_pr if with_exog else None
        tag = "exog" if with_exog else "no exog"

        f = ForecasterRecursive(estimator=make(), lags=5)
        f.fit(y=y, exog=ex_tr, store_in_sample_residuals=True)
        run(f"Recursive predict ({tag})", lambda: f.predict(steps=5, exog=ex_pr))
        run(f"Recursive bootstrapping ({tag})", lambda: f.predict_interval(steps=5, exog=ex_pr, method="bootstrapping", n_boot=10))
        run(f"Recursive conformal ({tag})", lambda: f.predict_interval(steps=5, exog=ex_pr, method="conformal"))

        d = ForecasterDirect(estimator=make(), lags=5, steps=3, n_jobs=1)
        d.fit(y=y, exog=ex_tr)
        run(f"Direct predict ({tag})", lambda: d.predict(exog=ex_pr))

        for enc in ("ordinal", "ordinal_category", "onehot", None):
            m = ForecasterRecursiveMultiSeries(estimator=make(), lags=5, encoding=enc)
            try:
                m.fit(series=series, exog=ex_tr, store_in_sample_residuals=True)
            except Exception as e:
                print(f"  MultiSeries fit {enc} ({tag})        {type(e).__name__}: {str(e)[:70]}")
                continue
            run(f"MultiSeries predict {enc} ({tag})", lambda: m.predict(steps=5, exog=ex_pr))
            run(f"MultiSeries bootstrap {enc} ({tag})", lambda: m.predict_interval(steps=5, exog=ex_pr, method="bootstrapping", n_boot=10))

        mv = ForecasterDirectMultiVariate(estimator=make(), level="l1", lags=5, steps=3, n_jobs=1, transformer_series=None)
        mv.fit(series=series, exog=ex_tr)
        run(f"DMV predict ({tag})", lambda: mv.predict(exog=ex_pr))
