"""
BUG-catboost-refit-categorical: with CatBoost and native categorical features,
a second `fit` of the same estimator object with other data gives a different
model than a new object. `ForecasterRecursive` fits the same object in every
`fit`, so a backtesting with refit is affected; `ForecasterDirect` clones the
estimator and is not.

Compares, with a categorical exog and with the same column as float: fitting
twice with the same data, fitting with A and then with B against a new object
on B, the same with `ForecasterDirect`, and a backtesting with refit against a
new forecaster in each fold (MAE 1.73 against 0.80 with the categorical exog,
identical with the numeric one). The last block repeats it with CatBoost alone.

Usage: python catboost_refit.py  (with `PYTHONPATH` set to the repository root if the
installed skforecast is not this code; the first line printed is the path of
the package used)
"""
import warnings
import numpy as np
import pandas as pd
import skforecast
from catboost import CatBoostRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster

warnings.simplefilter("ignore")
print(skforecast.__file__, "pandas", pd.__version__)
rng = np.random.default_rng(7)
n = 300
idx = pd.date_range("2020-01-01", periods=n, freq="D")
codes = rng.permutation(np.arange(n) % 6)
effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0])
y = pd.Series(effect[codes] + rng.normal(scale=0.3, size=n), index=idx, name="y")
exog_cat = pd.DataFrame(
    {"c": pd.Categorical(codes), "x": rng.normal(size=n)}, index=idx
)
exog_num = pd.DataFrame(
    {"c": codes.astype(float), "x": exog_cat["x"].to_numpy()}, index=idx
)


def cb():
    return CatBoostRegressor(
        iterations=30, depth=3, verbose=0, thread_count=1, random_seed=1,
        allow_writing_files=False,
    )


def cmp(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.array_equal(a, b):
        return "identical"
    return f"max abs diff {np.max(np.abs(a - b)):.6g}"


A, B, P = slice(0, 200), slice(50, 250), slice(250, 260)
for label, exog in (("categorical exog", exog_cat), ("numeric exog", exog_num)):
    print(label)
    # 1) Same data twice
    f = ForecasterRecursive(estimator=cb(), lags=3)
    f.fit(y=y.iloc[B], exog=exog.iloc[B])
    p1 = f.predict(steps=10, exog=exog.iloc[P])
    f.fit(y=y.iloc[B], exog=exog.iloc[B])
    p2 = f.predict(steps=10, exog=exog.iloc[P])
    print("  Recursive, fit B twice on the same object:", cmp(p1, p2))
    # 2) Fit A then B vs new on B
    g = ForecasterRecursive(estimator=cb(), lags=3)
    g.fit(y=y.iloc[A], exog=exog.iloc[A])
    g.fit(y=y.iloc[B], exog=exog.iloc[B])
    p3 = g.predict(steps=10, exog=exog.iloc[P])
    print("  Recursive, fit A then B vs new object on B:", cmp(p1, p3))
    print("    first values new   :", np.round(p1.to_numpy()[:4], 4))
    print("    first values refit :", np.round(p3.to_numpy()[:4], 4))
    # 3) Direct (clones the estimator in each fit)
    d1 = ForecasterDirect(estimator=cb(), lags=3, steps=2, n_jobs=1)
    d1.fit(y=y.iloc[B], exog=exog.iloc[B])
    q1 = d1.predict(exog=exog.iloc[P])
    d2 = ForecasterDirect(estimator=cb(), lags=3, steps=2, n_jobs=1)
    d2.fit(y=y.iloc[A], exog=exog.iloc[A])
    d2.fit(y=y.iloc[B], exog=exog.iloc[B])
    q2 = d2.predict(exog=exog.iloc[P])
    print("  Direct, fit A then B vs new object on B:", cmp(q1, q2))
    # 4) Backtesting with refit: last fold vs a new forecaster fitted on that fold
    cv = TimeSeriesFold(
        steps=10, initial_train_size=200, refit=True, fixed_train_size=True
    )
    metric, preds = backtesting_forecaster(
        forecaster=ForecasterRecursive(estimator=cb(), lags=3), y=y.iloc[:260],
        exog=exog.iloc[:260], cv=cv, metric="mean_absolute_error", n_jobs=1,
        show_progress=False, verbose=False,
    )
    h = ForecasterRecursive(estimator=cb(), lags=3)
    h.fit(y=y.iloc[50:250], exog=exog.iloc[50:250])
    p4 = h.predict(steps=10, exog=exog.iloc[P])
    print(
        "  Backtesting with refit, last fold vs new object:",
        cmp(preds["pred"].iloc[-10:], p4),
    )
    # MAE of the backtesting vs MAE with a new forecaster in each fold
    errs = []
    for start in range(200, 260, 10):
        k = ForecasterRecursive(estimator=cb(), lags=3)
        k.fit(y=y.iloc[start - 200:start], exog=exog.iloc[start - 200:start])
        pk = k.predict(steps=10, exog=exog.iloc[start:start + 10])
        errs.append(np.abs(pk.to_numpy() - y.iloc[start:start + 10].to_numpy()))
    print(
        "  MAE backtesting:", round(float(metric["mean_absolute_error"].iloc[0]), 6),
        "| MAE new object per fold:", round(float(np.mean(errs)), 6),
    )

# CatBoost alone, outside skforecast
print("CatBoost alone")
X = pd.DataFrame({"c": codes, "x": exog_cat["x"].to_numpy()})
yy = y.to_numpy()
m_new = cb().fit(X.iloc[B], yy[B], cat_features=[0])
m_re = cb().fit(X.iloc[A], yy[A], cat_features=[0])
m_re.fit(X.iloc[B], yy[B], cat_features=[0])
print(
    "  DataFrame int codes, fit A then B vs new on B:",
    cmp(m_new.predict(X.iloc[P]), m_re.predict(X.iloc[P])),
)
Xo = X.to_numpy().astype(object)
Xo[:, 0] = Xo[:, 0].astype(int)
m_new = cb().fit(Xo[B], yy[B], cat_features=[0])
m_re = cb().fit(Xo[A], yy[A], cat_features=[0])
m_re.fit(Xo[B], yy[B], cat_features=[0])
print(
    "  object array, fit A then B vs new on B:",
    cmp(m_new.predict(Xo[P]), m_re.predict(Xo[P])),
)
