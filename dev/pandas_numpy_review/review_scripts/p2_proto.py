"""
Prototype of proposal P2, without editing the package.

`proposal()` is a context manager that replaces `pd.DataFrame.to_numpy` with a
wrapper. The wrapper only acts when the call is the argument-less
`exog.to_numpy()` of the forecasters: the caller frame is `_create_train_X_y`
or `_create_predict_inputs` and the DataFrame is the local variable `exog` of
that frame (so the calendar features and the last window are not touched).
In that case, and only if `condition(exog)` holds, it returns
`exog.to_numpy(dtype=float, na_value=np.nan)`. Any other call goes to the
original method.

`condition` is the test "all columns numeric or bool" as the study words it:
`is_numeric_dtype(dtype) or is_bool_dtype(dtype)` for every column.
`strict_condition` is the narrower one this review recommends.
"""
import contextlib
import sys
import warnings

import numpy as np
import pandas as pd

_ORIG = pd.DataFrame.to_numpy
_CALLERS = {"_create_train_X_y", "_create_predict_inputs"}
calls = []  # (caller function, sorted set of the dtypes of the converted exog)


def condition(exog: pd.DataFrame) -> bool:
    return all(
        pd.api.types.is_numeric_dtype(d) or pd.api.types.is_bool_dtype(d)
        for d in exog.dtypes
    )


def strict_condition(exog: pd.DataFrame) -> bool:
    """Narrower test recommended by this review: convert only when today's
    `to_numpy()` would be `object`, that is, more than one dtype and at least
    one bool, nullable or pyarrow int/float/bool column; every column must be
    bool, int or float (numpy, nullable or pyarrow). No complex, no decimal,
    no Sparse. A frame with a single dtype is left to `to_numpy()`, which
    already gives a numeric array (and a view, without an extra copy)."""
    dtypes = set(exog.dtypes)
    if len(dtypes) < 2:
        return False
    special = False
    for d in dtypes:
        if isinstance(d, np.dtype):
            if d.kind not in "biuf":
                return False
            special |= d.kind == "b"
        elif isinstance(d, (pd.api.extensions.ExtensionDtype,)) and not isinstance(
            d, (pd.SparseDtype, pd.CategoricalDtype)
        ) and (
            pd.api.types.is_bool_dtype(d)
            or pd.api.types.is_integer_dtype(d)
            or pd.api.types.is_float_dtype(d)
        ):
            special = True
        else:
            return False
    return special


@contextlib.contextmanager
def proposal(cond=condition):
    def to_numpy(self, dtype=None, copy=False, na_value=pd.api.extensions.no_default):
        caller = sys._getframe(1)
        if (
            dtype is None
            and caller.f_code.co_name in _CALLERS
            and caller.f_locals.get("exog") is self
            and cond(self)
        ):
            out = _ORIG(self, dtype=float, na_value=np.nan)
            calls.append((caller.f_code.co_name, tuple(sorted({str(d) for d in self.dtypes}))))
            return out
        return _ORIG(self, dtype=dtype, copy=copy, na_value=na_value)

    pd.DataFrame.to_numpy = to_numpy
    try:
        yield
    finally:
        pd.DataFrame.to_numpy = _ORIG


def run(fn):
    """Run `fn`, return (result or None, 'ok' or 'ExcType: msg', sorted warning names)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            out, status = fn(), "ok"
        except Exception as e:  # noqa: BLE001
            out, status = None, f"{type(e).__name__}: {str(e).splitlines()[0][:95]}"
    names = sorted({x.category.__name__ for x in w} - {"IgnoredArgumentWarning"})
    return out, status, names


@contextlib.contextmanager
def capture_fit(estimator):
    """Record what `estimator.fit` receives (first call). The forecaster clones
    the estimator, so the method is replaced in the class and restored."""
    cls = type(estimator)
    orig = cls.fit
    seen = {}

    def fit(self, X, y, *a, **k):
        if not seen:
            seen["dtype"] = f"{type(X).__name__}[{getattr(X, 'dtype', None)}]"
            if getattr(X, "dtype", None) == object:
                seen["elem"] = sorted({type(v).__name__ for v in np.asarray(X)[:, -1]})
        return orig(self, X, y, *a, **k)

    cls.fit = fit
    try:
        yield seen
    finally:
        cls.fit = orig


def make_y(n=60, seed=0):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n).cumsum(), index=idx, name="y")


def same(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, pd.DataFrame):  # long format: keep the numeric columns
        a, b = a.select_dtypes("number"), b.select_dtypes("number")
    return bool(np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True))
