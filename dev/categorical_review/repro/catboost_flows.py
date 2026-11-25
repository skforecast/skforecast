"""
BUG-catboost-refit-categorical: which skforecast flows fit an estimator object
that is already fitted, and how much the result changes, with CatBoost and a
categorical exog.

Backtesting with and without refit of `ForecasterRecursive` (new forecaster,
fitted before, and with `one_hot_max_size=255`) against a new forecaster in
each fold (10 folds: 2.33 against 0.79 with a fixed window, 2.29 against 1.36
with an expanding one), a grid search without refit,
`ForecasterRecursiveMultiSeries` with two encodings,
`ForecasterRecursiveClassifier`, CatBoost alone with three pairs of training
sets and `ForecasterDirect`. The last block, `ForecasterDirect` with 10 steps,
takes several minutes: CatBoost needs about one second per fit with these
categorical features.

Usage: python catboost_flows.py  (with `PYTHONPATH` set to the repository root if the
installed skforecast is not this code; the first line printed is the path of
the package used)
"""
import warnings
import numpy as np
import pandas as pd
import skforecast
from catboost import CatBoostRegressor, CatBoostClassifier
from skforecast.recursive import (
    ForecasterRecursive, ForecasterRecursiveMultiSeries, ForecasterRecursiveClassifier,
)
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import (
    TimeSeriesFold, backtesting_forecaster, backtesting_forecaster_multiseries,
    grid_search_forecaster,
)

warnings.simplefilter("ignore")
print(skforecast.__file__, skforecast.__version__, "pandas", pd.__version__)
rng = np.random.default_rng(7)
n = 300
idx = pd.date_range("2020-01-01", periods=n, freq="D")
codes = rng.permutation(np.arange(n) % 6)
effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0])
y = pd.Series(effect[codes] + rng.normal(scale=0.3, size=n), index=idx, name="y")
series = pd.DataFrame(
    {"l1": y.to_numpy(), "l2": 0.5 * effect[codes] + rng.normal(scale=0.3, size=n)},
    index=idx,
)
exog = pd.DataFrame({"c": pd.Categorical(codes), "x": rng.normal(size=n)}, index=idx)


def cb(**kw):
    return CatBoostRegressor(
        iterations=30, depth=3, verbose=0, thread_count=1, random_seed=1,
        allow_writing_files=False, **kw,
    )


def mae(m):
    return round(float(m["mean_absolute_error"].iloc[0]), 4)


cv_norefit = TimeSeriesFold(steps=10, initial_train_size=200, refit=False)
cv_refit = TimeSeriesFold(steps=10, initial_train_size=200, refit=True, fixed_train_size=True)
kw = dict(metric="mean_absolute_error", n_jobs=1, show_progress=False, verbose=False)

print("ForecasterRecursive")
for label, cv in (("refit=False", cv_norefit), ("refit=True", cv_refit)):
    new = ForecasterRecursive(estimator=cb(), lags=3)
    m_new, _ = backtesting_forecaster(forecaster=new, y=y, exog=exog, cv=cv, **kw)
    fitted = ForecasterRecursive(estimator=cb(), lags=3)
    fitted.fit(y=y.iloc[:150], exog=exog.iloc[:150])
    m_fit, _ = backtesting_forecaster(forecaster=fitted, y=y, exog=exog, cv=cv, **kw)
    onehot = ForecasterRecursive(estimator=cb(one_hot_max_size=255), lags=3)
    m_oh, _ = backtesting_forecaster(forecaster=onehot, y=y, exog=exog, cv=cv, **kw)
    print(
        f"  backtesting {label}: new forecaster {mae(m_new)} | fitted before "
        f"{mae(m_fit)} | new, one_hot_max_size=255 {mae(m_oh)}"
    )

# Reference with refit: a new forecaster in each fold
errs = []
for start in range(200, n, 10):
    k = ForecasterRecursive(estimator=cb(), lags=3)
    k.fit(y=y.iloc[start - 200:start], exog=exog.iloc[start - 200:start])
    pk = k.predict(steps=10, exog=exog.iloc[start:start + 10])
    errs.append(np.abs(pk.to_numpy() - y.iloc[start:start + 10].to_numpy()))
print("  reference, new forecaster in each fold (refit=True):", round(float(np.mean(errs)), 4))

# Grid search with backtesting and return_best
f = ForecasterRecursive(estimator=cb(), lags=3)
f.fit(y=y.iloc[:150], exog=exog.iloc[:150])
res = grid_search_forecaster(
    forecaster=f, y=y, exog=exog, cv=cv_norefit, param_grid={"depth": [3]},
    metric="mean_absolute_error", return_best=False, n_jobs=1, show_progress=False,
    verbose=False,
)
print("  grid search (refit=False) on a fitted forecaster:", mae(res))

print("ForecasterRecursiveMultiSeries")
for enc in ("ordinal", "ordinal_category"):
    for label, cv in (("refit=False", cv_norefit), ("refit=True", cv_refit)):
        out = []
        for prefit in (False, True):
            m = ForecasterRecursiveMultiSeries(estimator=cb(), lags=3, encoding=enc)
            if prefit:
                m.fit(series=series.iloc[:150], exog=exog.iloc[:150])
            met, _ = backtesting_forecaster_multiseries(
                forecaster=m, series=series, exog=exog, cv=cv,
                metric="mean_absolute_error", n_jobs=1, show_progress=False,
                verbose=False,
            )
            out.append(round(float(met["mean_absolute_error"].mean()), 4))
        print(f"  {enc}, backtesting {label}: new forecaster {out[0]} | fitted before {out[1]}")
    # exog without categorical features: only the level
    out = []
    for prefit in (False, True):
        m = ForecasterRecursiveMultiSeries(estimator=cb(), lags=3, encoding=enc)
        if prefit:
            m.fit(series=series.iloc[:150], exog=exog[["x"]].iloc[:150])
        met, _ = backtesting_forecaster_multiseries(
            forecaster=m, series=series, exog=exog[["x"]], cv=cv_refit,
            metric="mean_absolute_error", n_jobs=1, show_progress=False, verbose=False,
        )
        out.append(round(float(met["mean_absolute_error"].mean()), 4))
    print(f"  {enc}, numeric exog only, refit=True: new forecaster {out[0]} | fitted before {out[1]}")

print("ForecasterRecursiveClassifier")
y_cls = pd.Series(codes % 3, index=idx, name="y")


def cbc():
    return CatBoostClassifier(
        iterations=30, depth=3, verbose=0, thread_count=1, random_seed=1,
        allow_writing_files=False,
    )


c1 = ForecasterRecursiveClassifier(estimator=cbc(), lags=3)
c1.fit(y=y_cls.iloc[50:250], exog=exog.iloc[50:250])
p1 = c1.predict_proba(steps=10, exog=exog.iloc[250:260])
c2 = ForecasterRecursiveClassifier(estimator=cbc(), lags=3)
c2.fit(y=y_cls.iloc[:200], exog=exog.iloc[:200])
c2.fit(y=y_cls.iloc[50:250], exog=exog.iloc[50:250])
p2 = c2.predict_proba(steps=10, exog=exog.iloc[250:260])
print(
    "  fit A then B vs new on B, predict_proba max abs diff:",
    round(float(np.max(np.abs(p1.to_numpy() - p2.to_numpy()))), 4),
)

print("Expanding window (fixed_train_size=False)")
cv_exp = TimeSeriesFold(steps=10, initial_train_size=200, refit=True, fixed_train_size=False)
m_exp, _ = backtesting_forecaster(forecaster=ForecasterRecursive(estimator=cb(), lags=3), y=y, exog=exog, cv=cv_exp, **kw)
errs = []
for start in range(200, n, 10):
    k = ForecasterRecursive(estimator=cb(), lags=3)
    k.fit(y=y.iloc[:start], exog=exog.iloc[:start])
    pk = k.predict(steps=10, exog=exog.iloc[start:start + 10])
    errs.append(np.abs(pk.to_numpy() - y.iloc[start:start + 10].to_numpy()))
print("  backtesting refit=True:", mae(m_exp), "| new forecaster in each fold:", round(float(np.mean(errs)), 4))

print("CatBoost alone, second fit on data that starts at the same row")
X = pd.DataFrame({"c": codes, "x": exog["x"].to_numpy()})
yy = y.to_numpy()
for label, s1, s2 in (("A=[0:200], B=[0:250]", slice(0, 200), slice(0, 250)), ("A=[0:200], B=[50:250]", slice(0, 200), slice(50, 250)), ("A=[0:200], B=[1:201]", slice(0, 200), slice(1, 201))):
    new = cb().fit(X.iloc[s2], yy[s2], cat_features=[0])
    re = cb().fit(X.iloc[s1], yy[s1], cat_features=[0])
    re.fit(X.iloc[s2], yy[s2], cat_features=[0])
    P = slice(250, 300)
    print(f"  {label}: same={np.array_equal(new.predict(X.iloc[P]), re.predict(X.iloc[P]))} MAE new={np.mean(np.abs(new.predict(X.iloc[P]) - yy[P])):.4f} refit={np.mean(np.abs(re.predict(X.iloc[P]) - yy[P])):.4f}")
print("  first categories of each start:", codes[0:6], codes[1:7], codes[50:56])

print("ForecasterDirect")
new = ForecasterDirect(estimator=cb(), lags=3, steps=10, n_jobs=1)
m_new, _ = backtesting_forecaster(forecaster=new, y=y, exog=exog, cv=cv_refit, **kw)
fitted = ForecasterDirect(estimator=cb(), lags=3, steps=10, n_jobs=1)
fitted.fit(y=y.iloc[:150], exog=exog.iloc[:150])
m_fit, _ = backtesting_forecaster(forecaster=fitted, y=y, exog=exog, cv=cv_refit, **kw)
print(f"  backtesting refit=True: new forecaster {mae(m_new)} | fitted before {mae(m_fit)}")

