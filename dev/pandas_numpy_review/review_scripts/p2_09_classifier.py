"""
P2 in `ForecasterRecursiveClassifier` (`_forecaster_recursive_classifier.py`
:928 fit, :1622 predict): same questions as p2_01 for a few dtypes.
"""
import _env
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from skforecast.recursive import ForecasterRecursiveClassifier
import p2_proto as P

n, steps = 80, 3
rng = np.random.default_rng(0)
idx = pd.date_range("2020-01-01", periods=n + steps, freq="D")
y = pd.Series(rng.integers(0, 3, size=n + steps), index=idx, name="y")
k = np.arange(n + steps)
a = np.linspace(0, 1, n + steps)


def col(dtype, values, na=None):
    x = pd.array(values, dtype=dtype)
    if na is not None:
        x[na] = pd.NA
    return x


cases = {
    "float + bool": lambda: k % 3 == 0,
    "float + Int64": lambda: col("Int64", k % 5),
    "float + Float64 NA in fit": lambda: col("Float64", k / 7, 30),
    "float + Float64 NA in predict": lambda: col("Float64", k / 7, n + 1),
    "float + double[pyarrow]": lambda: col("double[pyarrow]", k / 7),
}
estimators = {
    "LogReg": lambda: LogisticRegression(max_iter=200),
    "LGBM": lambda: LGBMClassifier(n_estimators=5, verbose=-1, n_jobs=1, min_child_samples=3),
    "HGB": lambda: HistGradientBoostingClassifier(max_iter=5, min_samples_leaf=3),
}


def experiment(est, exog):
    f = ForecasterRecursiveClassifier(est, lags=3)
    with P.capture_fit(est) as seen:
        _, st_fit, w_fit = P.run(lambda: f.fit(y.iloc[:n], exog=exog.iloc[:n]))
    proba, st_pred, w_pred = (None, "-", [])
    if st_fit == "ok":
        proba, st_pred, w_pred = P.run(lambda: f.predict_proba(steps=steps, exog=exog.iloc[n:]))
    return seen.get("dtype", "not reached"), st_fit[:75], st_pred[:75], (w_fit, w_pred), proba


for cname, make in cases.items():
    try:
        exog = pd.DataFrame({"a": a, "x": make()}, index=idx)
    except Exception as e:  # noqa: BLE001
        print(f"\n### {cname}: SKIPPED ({type(e).__name__})")
        continue
    print(f"\n### {cname}")
    for ename, make_e in estimators.items():
        x0, f0, p0, w0, v0 = experiment(make_e(), exog)
        with P.proposal():
            x1, f1, p1, w1, v1 = experiment(make_e(), exog)
        print(f"  {ename:6s} X {x0} -> {x1} | fit {f0} -> {f1} | predict_proba {p0} -> {p1} | "
              f"eq {P.same(v0, v1)} | warn {w0} -> {'=' if w0 == w1 else w1}")
print(_env.versions())
