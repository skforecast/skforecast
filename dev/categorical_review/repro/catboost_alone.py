"""
BUG-catboost-refit-categorical, without skforecast: a second `fit` of the same
`CatBoostRegressor` with other data and a categorical feature gives a different
model than a new object.

For several input types (DataFrame with int codes, strings or category dtype,
object array), prints whether the predictions are the same and the MAE of a new
object, of the refitted one, of a `sklearn.base.clone` fitted again and of a
`copy.deepcopy` fitted again. `clone` avoids the difference and `deepcopy` does
not; `one_hot_max_size=255` avoids it, and without categorical features the
models are identical. Then prints the predictions by category and checks the
same data in another row order.

Usage: python catboost_alone.py  (with `PYTHONPATH` set to the repository root if the
installed skforecast is not this code; the first line printed is the path of
the package used)
"""
import copy
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor, Pool
from sklearn.base import clone

print("catboost", catboost.__version__)
rng = np.random.default_rng(7)
n = 300
codes = rng.permutation(np.arange(n) % 6)
effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0])
yy = effect[codes] + rng.normal(scale=0.3, size=n)
x = rng.normal(size=n)
A, B, P = slice(0, 200), slice(50, 250), slice(250, 300)


def cb(**kw):
    return CatBoostRegressor(
        iterations=30, depth=3, verbose=0, thread_count=1, random_seed=1,
        allow_writing_files=False, **kw,
    )


def mae(m, X):
    return float(np.mean(np.abs(m.predict(X[P] if not hasattr(X, "iloc") else X.iloc[P]) - yy[P])))


def cut(X, s):
    return X.iloc[s] if hasattr(X, "iloc") else X[s]


def trial(label, X, fit_kw=None, **kw):
    fit_kw = fit_kw or {"cat_features": [0]}
    new = cb(**kw).fit(cut(X, B), yy[B], **fit_kw)
    re = cb(**kw).fit(cut(X, A), yy[A], **fit_kw)
    re.fit(cut(X, B), yy[B], **fit_kw)
    cl = cb(**kw).fit(cut(X, A), yy[A], **fit_kw)
    cl = clone(cl).fit(cut(X, B), yy[B], **fit_kw)
    dc = cb(**kw).fit(cut(X, A), yy[A], **fit_kw)
    dc = copy.deepcopy(dc).fit(cut(X, B), yy[B], **fit_kw)
    p_new, p_re = new.predict(cut(X, P)), re.predict(cut(X, P))
    print(
        f"{label:<44} same={np.array_equal(p_new, p_re)!s:<5} "
        f"maxdiff={np.max(np.abs(p_new - p_re)):.4f} "
        f"MAE new={np.mean(np.abs(p_new - yy[P])):.4f} "
        f"refit={np.mean(np.abs(p_re - yy[P])):.4f} "
        f"clone={np.mean(np.abs(cl.predict(cut(X, P)) - yy[P])):.4f} "
        f"deepcopy={np.mean(np.abs(dc.predict(cut(X, P)) - yy[P])):.4f}"
    )
    return new, re


X_int = pd.DataFrame({"c": codes, "x": x})
X_str = pd.DataFrame({"c": [f"k{c}" for c in codes], "x": x})
X_cat = pd.DataFrame({"c": pd.Categorical(codes), "x": x})
X_obj = X_int.to_numpy().astype(object)
X_obj[:, 0] = X_obj[:, 0].astype(int)

new, re = trial("DataFrame, int codes", X_int)
trial("DataFrame, strings", X_str)
trial("DataFrame, category dtype", X_cat)
trial("object array, int codes", X_obj)
trial("DataFrame, int codes, one_hot_max_size=255", X_int, one_hot_max_size=255)
trial("DataFrame, int codes, one_hot_max_size=2", X_int, one_hot_max_size=2)
trial("no categorical feature", X_int, fit_kw={"cat_features": []})

# Where do they differ: by category
p_new, p_re = new.predict(X_int.iloc[P]), re.predict(X_int.iloc[P])
by_cat = pd.DataFrame(
    {"c": codes[P], "new": p_new, "refit": p_re, "y": yy[P]}
).groupby("c").mean().round(3)
print(by_cat)
print("categories in A:", sorted(set(codes[A])), "in B:", sorted(set(codes[B])))
print("first category seen in A:", codes[A][:6], "in B:", codes[B][:6])

# Same data in another row order
perm = rng.permutation(200)
XB = X_int.iloc[B].reset_index(drop=True)
yB = yy[B]
m1 = cb().fit(XB, yB, cat_features=[0])
m2 = cb().fit(XB.iloc[perm], yB[perm], cat_features=[0])
m2.fit(XB, yB, cat_features=[0])
print(
    "fit on shuffled B then B vs new on B:",
    np.array_equal(m1.predict(X_int.iloc[P]), m2.predict(X_int.iloc[P])),
)
