"""
P2, item 8 (decision D-NP3): exog nullable column with `pd.NA`.
Today vs proposal, in fit and in predict: exception type and message, the
frame that raises it (is it skforecast or the estimator?), and the skforecast
warnings emitted before it.
"""
import _env
import os
import traceback
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
import p2_proto as P

n, steps = 60, 3
y = P.make_y(n + steps)
a = np.linspace(0, 1, n + steps)


def exog(na_row):
    x = pd.array(np.arange(n + steps) % 7, dtype="Int64")
    x[na_row] = pd.NA
    return pd.DataFrame({"a": a, "x": x}, index=y.index)


def call(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn()
            res = "ok"
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)
            last = tb[-1]
            sk = [f for f in tb if "/skforecast/" in f.filename and "site-packages" not in f.filename]
            where = f"{os.path.basename(last.filename)}:{last.lineno} `{last.line}`"
            if sk and sk[-1] is not last:
                where += f" <- skforecast {os.path.basename(sk[-1].filename)}:{sk[-1].lineno} `{sk[-1].line[:60]}`"
            res = f"{type(e).__name__}: {str(e).splitlines()[0][:110]}\n            raised at {where}"
    ws = [f"{x.category.__name__}: {str(x.message)[:70]}" for x in w
          if x.category.__module__.startswith("skforecast")]
    return res, ws


estimators = {
    "Ridge": lambda: Ridge(),
    "LGBM": lambda: LGBMRegressor(n_estimators=5, verbose=-1, n_jobs=1, min_child_samples=3),
    "HGB": lambda: HistGradientBoostingRegressor(max_iter=5, min_samples_leaf=3),
}
forecasters = {
    "Rec": lambda e: ForecasterRecursive(e, lags=3),
    "Dir": lambda e: ForecasterDirect(e, lags=3, steps=steps, n_jobs=1),
    "DMV": lambda e: ForecasterDirectMultiVariate(e, level="y", lags=3, steps=steps, n_jobs=1),
}
for stage, na_row in (("fit", 30), ("predict", n + 1)):
    ex = exog(na_row)
    for fname, make in forecasters.items():
        for ename, est in estimators.items():
            def run():
                f = make(est())
                target = y.iloc[:n].to_frame() if fname == "DMV" else y.iloc[:n]
                kw = {"series": target} if fname == "DMV" else {"y": target}
                if stage == "fit":
                    return call(lambda: f.fit(exog=ex.iloc[:n], **kw))
                f.fit(exog=ex.iloc[:n], **kw)
                return call(lambda: f.predict(steps=steps, exog=ex.iloc[n:]))
            today, w_today = run()
            with P.proposal():
                new, w_new = run()
            print(f"\nNA in {stage} | {fname} | {ename}")
            print(f"  today   : {today}\n            warnings {w_today}")
            print(f"  proposal: {new}\n            warnings {'=' if w_new == w_today else w_new}")
print(_env.versions())
