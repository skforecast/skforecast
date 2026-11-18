"""
Helpers shared by the review scripts of T1 (`REVIEW_T1.md`). They do not
modify production code: `transform_numpy` is replaced at runtime in the modules
that import it by name.

Three candidates are compared with the current `transform_numpy`:

- `plan_literal`: T1 exactly as written in PLAN.md.
- `prototypes.fast_transform_numpy`: the prototype the plan was measured with.
  It is not the same as the plan: it does not check `feature_names_in_` nor
  `mean_`, and it checks that `array` is a numpy ndarray.
- `proposed`: the conditions that REVIEW_T1.md proposes.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import skforecast.utils.utils as _utils  # noqa: E402

ORIG_TRANSFORM_NUMPY = _utils.transform_numpy

# Every module of the package that imports `transform_numpy` by name
# (`grep -n "transform_numpy" skforecast/ --include=*.py`, tests excluded).
# `prototypes._MODULES` lacks the last two.
ALL_MODULES = [
    "skforecast.recursive._forecaster_recursive_multiseries",
    "skforecast.recursive._forecaster_recursive",
    "skforecast.direct._forecaster_direct",
    "skforecast.direct._forecaster_direct_multivariate",
    "skforecast.model_selection._utils",
    "skforecast.recursive._forecaster_stats",
    "skforecast.deep_learning._forecaster_rnn",
]


_SKIPPED = set()


def import_modules(names=ALL_MODULES):
    mods = []
    for name in names:
        try:
            mods.append(importlib.import_module(name))
        except ImportError:  # keras / torch not installed
            if name not in _SKIPPED:
                _SKIPPED.add(name)
                print(f"[skip] {name} cannot be imported (optional dependency missing)")
    return mods


def plan_conditions(array, transformer, fit, inverse_transform):
    """Each condition of point 2 of T1 (PLAN.md), by name."""
    is_ss = type(transformer) is StandardScaler
    cond = {
        "fit=False": not fit,
        "type is StandardScaler": is_ss,
        "fitted (scale_/mean_)": is_ss
        and (hasattr(transformer, "scale_") or not transformer.with_std)
        and (hasattr(transformer, "mean_") or not transformer.with_mean),
        "n_features_in_==1": getattr(transformer, "n_features_in_", None) == 1,
        "no feature_names_in_": not hasattr(transformer, "feature_names_in_"),
        "float64": isinstance(array, np.ndarray) and array.dtype == np.float64,
        "1D or inverse": isinstance(array, np.ndarray)
        and (array.ndim == 1 or inverse_transform),
    }
    return cond


def prototype_conditions(array, transformer, fit, inverse_transform):
    """Conditions of `prototypes.fast_transform_numpy` (as written)."""
    return (
        not fit
        and type(transformer) is StandardScaler
        and isinstance(array, np.ndarray)
        and array.dtype == np.float64
        and getattr(transformer, "n_features_in_", None) == 1
        and hasattr(transformer, "scale_")
        and (inverse_transform or array.ndim == 1)
    )


def plan_literal(
    array, transformer, fit=False, inverse_transform=False, force_single_column=False
):
    """
    T1 exactly as written in PLAN.md: the fast path is tried right after
    `if transformer is None` (point 1), with the conditions of point 2 in their
    order and nothing else (no check of the type of `array`), the arithmetic of
    point 3, the copy of point 4 and the inf check of point 5.
    """
    if transformer is None:
        return array
    if (
        not fit
        and type(transformer) is StandardScaler
        and (hasattr(transformer, "scale_") or not transformer.with_std)
        and (hasattr(transformer, "mean_") or not transformer.with_mean)
        and getattr(transformer, "n_features_in_", None) == 1
        and not hasattr(transformer, "feature_names_in_")
        and array.dtype == np.float64
        and (array.ndim == 1 or inverse_transform)
    ):
        out = array.copy()
        if inverse_transform:
            if transformer.with_std:
                out *= transformer.scale_[0]
            if transformer.with_mean:
                out += transformer.mean_[0]
        else:
            if transformer.with_mean:
                out -= transformer.mean_[0]
            if transformer.with_std:
                out /= transformer.scale_[0]
        if not (not math.isfinite(out.sum()) and np.isinf(array).any()):
            return out
    return ORIG_TRANSFORM_NUMPY(
        array=array, transformer=transformer, fit=fit,
        inverse_transform=inverse_transform, force_single_column=force_single_column,
    )


def _is_float64_vector(value):
    return type(value) is np.ndarray and value.dtype == np.float64 and value.shape == (1,)


def proposed_applies(array, transformer, fit, inverse_transform):
    """Conditions of the shortcut proposed in REVIEW_T1.md."""
    if not (
        not fit
        and type(transformer) is StandardScaler
        and type(array) is np.ndarray
        and array.dtype == np.float64
        and (array.ndim == 1 or (inverse_transform and array.ndim == 2))
        and array.size > 0
    ):
        return False
    state = vars(transformer)
    return (
        state.get("n_features_in_") == 1
        and (not transformer.with_mean or _is_float64_vector(state.get("mean_")))
        and (not transformer.with_std or _is_float64_vector(state.get("scale_")))
    )


def proposed(
    array, transformer, fit=False, inverse_transform=False, force_single_column=False
):
    """
    Conditions proposed in REVIEW_T1.md: the same arithmetic as the plan, with
    the guards that the review found missing (exact type, dimensions and size
    of `array`, type and shape of `mean_` and `scale_`), without the
    `feature_names_in_` condition and ignoring the overflow of the sum, as
    scikit-learn does.
    """
    if transformer is None:
        return array
    if proposed_applies(array, transformer, fit, inverse_transform):
        out = array.copy()
        if inverse_transform:
            if transformer.with_std:
                out *= transformer.scale_[0]
            if transformer.with_mean:
                out += transformer.mean_[0]
        else:
            if transformer.with_mean:
                out -= transformer.mean_[0]
            if transformer.with_std:
                out /= transformer.scale_[0]
        with np.errstate(over="ignore"):
            total = out.sum()
        if math.isfinite(total) or not np.isinf(array).any():
            return out
    return ORIG_TRANSFORM_NUMPY(
        array=array, transformer=transformer, fit=fit,
        inverse_transform=inverse_transform, force_single_column=force_single_column,
    )


def proposed_isinf_first(
    array, transformer, fit=False, inverse_transform=False, force_single_column=False
):
    """
    Same conditions as `proposed`, with the infinite values looked for in the
    input before computing (no sum of the result, so no `np.errstate`).
    """
    if transformer is None:
        return array
    if (
        proposed_applies(array, transformer, fit, inverse_transform)
        and not np.isinf(array).any()
    ):
        out = array.copy()
        if inverse_transform:
            if transformer.with_std:
                out *= transformer.scale_[0]
            if transformer.with_mean:
                out += transformer.mean_[0]
        else:
            if transformer.with_mean:
                out -= transformer.mean_[0]
            if transformer.with_std:
                out /= transformer.scale_[0]
        return out
    return ORIG_TRANSFORM_NUMPY(
        array=array, transformer=transformer, fit=fit,
        inverse_transform=inverse_transform, force_single_column=force_single_column,
    )


def plan_applies(array, transformer, fit, inverse_transform):
    return all(plan_conditions(array, transformer, fit, inverse_transform).values())


class Counted:
    """Wrap a candidate and count the calls in which its shortcut applies."""

    def __init__(self, func, applies):
        self.func, self.applies = func, applies
        self.calls = self.hits = 0

    def __call__(self, array, transformer, fit=False, inverse_transform=False,
                 force_single_column=False):
        if transformer is not None:
            self.calls += 1
            self.hits += bool(self.applies(array, transformer, fit, inverse_transform))
        return self.func(array=array, transformer=transformer, fit=fit,
                         inverse_transform=inverse_transform,
                         force_single_column=force_single_column)


@contextlib.contextmanager
def patched(func, modules=None):
    """Replace `transform_numpy` by `func` in every module that imports it."""
    mods = import_modules() if modules is None else modules
    for m in mods:
        m.transform_numpy = func
    try:
        yield
    finally:
        for m in mods:
            m.transform_numpy = ORIG_TRANSFORM_NUMPY


class Tracer:
    """
    Records every call to `transform_numpy` by call site (file:line of the
    caller): arguments, input array and state of the transformer.
    """

    def __init__(self, user_transformers=()):
        self.calls = defaultdict(Counter)
        self.user_ids = {id(t) for t in user_transformers}

    def __call__(self, array, transformer, fit=False, inverse_transform=False,
                 force_single_column=False):
        frame = sys._getframe(1)
        site = f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}"
        if transformer is None:
            key = ("transformer=None",)
        else:
            cond = plan_conditions(array, transformer, fit, inverse_transform)
            is_arr = isinstance(array, np.ndarray)
            key = (
                f"fit={fit}",
                f"inverse={inverse_transform}",
                f"ndim={array.ndim}" if is_arr else type(array).__name__,
                f"dtype={array.dtype}" if is_arr else "",
                f"c_contig={array.flags.c_contiguous}" if is_arr else "",
                f"writeable={array.flags.writeable}" if is_arr else "",
                f"transformer={type(transformer).__name__}",
                f"user_object={id(transformer) in self.user_ids}",
                f"feature_names_in_={hasattr(transformer, 'feature_names_in_')}",
                f"n_features_in_={getattr(transformer, 'n_features_in_', None)}",
                "PLAN=" + ("yes" if all(cond.values()) else
                           "no (" + "; ".join(k for k, v in cond.items() if not v) + ")"),
                "PROTOTYPE=" + ("yes" if prototype_conditions(
                    array, transformer, fit, inverse_transform) else "no"),
            )
        self.calls[site][key] += 1
        return ORIG_TRANSFORM_NUMPY(
            array=array, transformer=transformer, fit=fit,
            inverse_transform=inverse_transform, force_single_column=force_single_column,
        )

    def report(self, title):
        print(f"\n##### {title}")
        for site in sorted(self.calls, key=lambda s: (s.split(":")[0], int(s.split(":")[1]))):
            for key, n in self.calls[site].items():
                print(f"{site:48s} x{n:<5d} " + " ".join(k for k in key if k))


def env_line():
    import numba
    import pandas as pd
    import sklearn

    import skforecast

    return (
        f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "
        f"scikit-learn {sklearn.__version__}, numba {numba.__version__}, "
        f"skforecast {skforecast.__version__}"
    )
