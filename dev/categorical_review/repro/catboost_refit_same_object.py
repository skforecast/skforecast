"""
BUG-catboost-refit-categorical, CatBoost alone (no skforecast), the minimal
script for a CatBoost issue about the model fitted twice without a copy: with
a categorical feature encoded with target statistics, fitting again a fitted
model with other rows does not give the model of a new object.

The other form of the bug (a copy made with `pickle` or `copy.deepcopy` that
is fitted again blocks `predict`) is already reported, with CatBoost 1.2.8 on
Linux, and fixed in the CatBoost repository after version 1.2.10:
https://github.com/catboost/catboost/issues/2905. That fix trains into a new
model object, so it may cover this form too. Not checked: its regression test
fits the copy again with the same data, and with the same data this
difference does not appear.

If the issue is opened, CatBoost asks for these fields (`ISSUE_TEMPLATE.md` of
its repository): Problem, catboost version, Operating System, CPU and GPU.
Suggested title: "Fitting again a fitted model with other data and
categorical features gives a different model than a new object".

Usage: python catboost_refit_same_object.py

Session of 2026-11-26 (CatBoost 1.2.10, Windows, Python 3.13.18, numpy 2.5.5,
pandas 2.3.3), the same output in two runs:
categorical, other rows in the first fit: max abs diff 6.6794 | MAE refit 3.5309 | MAE new 0.4688
categorical, same rows in both fits   : max abs diff 0.0000 | MAE refit 0.4688 | MAE new 0.4688
categorical, one_hot_max_size=255     : max abs diff 0.0000 | MAE refit 0.2481 | MAE new 0.2481
numeric (no cat_features)             : max abs diff 0.0000 | MAE refit 0.2657 | MAE new 0.2657
"""
import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

print("catboost", catboost.__version__)
rng = np.random.default_rng(7)
n = 300
codes = rng.permutation(np.arange(n) % 6)
effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0])
X = pd.DataFrame({"c": codes, "x": rng.normal(size=n)})
y = effect[codes] + rng.normal(scale=0.3, size=n)
A, B, P = slice(0, 200), slice(50, 250), slice(250, 300)
params = dict(
    iterations=30, depth=3, verbose=0, thread_count=1, random_seed=1,
    allow_writing_files=False,
)


def compare(name, first, cat_features, **kwargs):
    # Model fitted twice (rows `first`, then rows B) against a new model
    # fitted once with rows B.
    refit = CatBoostRegressor(**params, **kwargs)
    refit.fit(X.iloc[first], y[first], cat_features=cat_features)
    refit.fit(X.iloc[B], y[B], cat_features=cat_features)
    new = CatBoostRegressor(**params, **kwargs)
    new.fit(X.iloc[B], y[B], cat_features=cat_features)
    p_refit, p_new = refit.predict(X.iloc[P]), new.predict(X.iloc[P])
    print(
        f"{name}: max abs diff {np.abs(p_refit - p_new).max():.4f}"
        f" | MAE refit {np.abs(p_refit - y[P]).mean():.4f}"
        f" | MAE new {np.abs(p_new - y[P]).mean():.4f}"
    )


compare("categorical, other rows in the first fit", A, [0])
compare("categorical, same rows in both fits   ", B, [0])
compare("categorical, one_hot_max_size=255     ", A, [0], one_hot_max_size=255)
compare("numeric (no cat_features)             ", A, None)
