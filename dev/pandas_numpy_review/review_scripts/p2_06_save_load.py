"""
P2, item 9: a forecaster fitted and saved BEFORE the change (object matrix),
loaded and used AFTER it (float matrix in predict). Stored dtype attributes,
predictions and warnings.

"old -> old": fit, save, load and predict with today's code.
"old -> new": fit and save with today's code, load and predict with the proposal.
"new -> new": everything with the proposal.
"""
import _env
import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.utils import save_forecaster, load_forecaster
import p2_proto as P

n, steps = 60, 3
y = P.make_y(n + steps)
k = np.arange(n + steps)
a = np.linspace(0, 1, n + steps)
exogs = {
    "float + bool": pd.DataFrame({"a": a, "b": k % 3 == 0}, index=y.index),
    "float + Int64": pd.DataFrame({"a": a, "i": pd.array(k % 5, dtype="Int64")}, index=y.index),
    "float + bool + category ('auto')": pd.DataFrame(
        {"a": a, "b": k % 3 == 0, "c": pd.Categorical(k % 4)}, index=y.index),
}
try:
    exogs["float + double[pyarrow]"] = pd.DataFrame(
        {"a": a, "f": pd.array(k / 7, dtype="double[pyarrow]")}, index=y.index)
except Exception as e:  # noqa: BLE001
    print("float + double[pyarrow]: SKIPPED", type(e).__name__)
estimators = {
    "Ridge": lambda: Ridge(),
    "LGBM": lambda: LGBMRegressor(n_estimators=5, verbose=-1, n_jobs=1, min_child_samples=3),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=5, min_samples_leaf=3),
}
forecasters = {
    "Rec": lambda e: ForecasterRecursive(e, lags=3),
    "Dir": lambda e: ForecasterDirect(e, lags=3, steps=steps, n_jobs=1),
}
ATTRS = ("exog_type_in_", "exog_dtypes_in_", "exog_dtypes_out_", "X_train_features_names_out_")
tmp = tempfile.mkdtemp()


def fit_save(f, exog, path):
    f.fit(y.iloc[:n], exog=exog.iloc[:n])
    save_forecaster(f, path, verbose=False, suppress_warnings=True)
    return {a: repr(getattr(f, a)) for a in ATTRS}


def load_predict(path, exog):
    f = load_forecaster(path, verbose=False)
    return P.run(lambda: f.predict(steps=steps, exog=exog.iloc[n:]))


for ename, make_e in estimators.items():
    for fname, make_f in forecasters.items():
        for cname, exog in exogs.items():
            old = os.path.join(tmp, "old.joblib")
            new = os.path.join(tmp, "new.joblib")
            attrs_old = fit_save(make_f(make_e()), exog, old)
            p_oo, st_oo, w_oo = load_predict(old, exog)
            with P.proposal():
                attrs_new = fit_save(make_f(make_e()), exog, new)
                p_on, st_on, w_on = load_predict(old, exog)
                p_nn, st_nn, w_nn = load_predict(new, exog)
            print(f"{ename:5s} {fname} {cname:33s} stored attrs equal: {attrs_old == attrs_new} | "
                  f"old->old {st_oo[:40]} {w_oo} | old->new {st_on[:40]} {w_on} | new->new {st_nn[:40]} {w_nn} | "
                  f"pred old->old == old->new: {P.same(p_oo, p_on)}, == new->new: {P.same(p_oo, p_nn)}")
print(_env.versions())
