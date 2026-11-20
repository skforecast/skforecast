"""
What a `fit` of ForecasterRecursiveMultiSeries inside a backtesting is made of
(`store_in_sample_residuals=True`, as backtesting calls it): creation of the
training matrices, the estimator and the rest. In a grid search over the
parameters of the estimator the training matrices are the same for every
candidate. Usage: python time_fit_parts.py N_SERIES EXOG EST. Run alone.
"""
import _env  # noqa: F401
import sys
import time
import warnings

from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from bench_data import make_data
from skforecast.recursive import ForecasterRecursiveMultiSeries as C

warnings.simplefilter("ignore")
n_series = int(sys.argv[1]); exog_kind = sys.argv[2]; est = sys.argv[3]
series, exog = make_data(n_series, exog_kind, 1500)
series = {k: v.iloc[:1260] for k, v in series.items()}
exog = None if exog is None else {k: v.iloc[:1260] for k, v in exog.items()}
cls = Ridge if est == "ridge" else LGBMRegressor
estimator = Ridge() if est == "ridge" else LGBMRegressor(n_estimators=50, verbose=-1, n_jobs=1, random_state=1)
acc = {}


def timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter(); out = fn(*a, **k)
        acc[name] = acc.get(name, 0.0) + time.perf_counter() - t
        return out
    return wrapper


C._create_train_X_y = timed("create_train_X_y", C._create_train_X_y)
cls.fit = timed("estimator.fit", cls.fit)
cls.predict = timed("estimator.predict (in-sample residuals)", cls.predict)
best = None
for _ in range(3):
    acc.clear()
    f = C(estimator, lags=24, encoding="ordinal")
    f._probabilistic_mode = False
    t = time.perf_counter()
    f.fit(series=series, exog=exog, store_in_sample_residuals=True)
    acc["total"] = time.perf_counter() - t
    if best is None or acc["total"] < best["total"]:
        best = dict(acc)
rest = best["total"] - sum(v for k, v in best.items() if k != "total")
print(f"n_series={n_series} exog={exog_kind} est={est}: fit {best['total'] * 1e3:.0f} ms = "
      + ", ".join(f"{k} {v * 1e3:.0f} ms ({v / best['total'] * 100:.0f} %)" for k, v in best.items() if k != "total")
      + f", rest {rest * 1e3:.0f} ms ({rest / best['total'] * 100:.0f} %)")
print(_env.versions())
