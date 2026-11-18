"""
Helpers shared by the review scripts of T1 (`REVIEW_T1.md`). They do not
modify production code: `transform_numpy` is replaced at runtime in the modules
that import it by name.

The reference ("current" in the outputs) is `transform_numpy` as it was
before T1: the function is read with git from commit `c9172e1` (`0.28.x`, the
base of the review; set `REVIEW_T1_REFERENCE` to use another one). So the
scripts keep comparing against the behavior before T1 once the shortcut is
implemented in the package.

Candidates compared with the reference:

- `PACKAGE_TRANSFORM_NUMPY` (PACKAGE): the function that is now in
  `skforecast/utils/utils.py`. Before T1 is implemented it is the reference
  itself; afterwards it is the implementation to accept.

- `plan_literal`: T1 exactly as PLAN.md was written before the review
  (commit 9d02780).
- `prototypes.fast_transform_numpy`: the prototype. During the review it was
  the one the plan was measured with, which checked neither
  `feature_names_in_` nor `mean_`. After the review it was aligned with the
  revised design, so it is now equivalent to `proposed_isinf_first`. The
  outputs in `results/review_t1_*` are those of the review: check out commit
  9d02780 to reproduce its PROTO figures.
- `proposed`: the conditions that REVIEW_T1.md proposes, with the inf check
  that was decided before the review (sum of the result).
- `proposed_isinf_first`: the same conditions, looking for inf in the input
  before computing. It is the design adopted in PLAN.md.
"""

from __future__ import annotations

import __future__

import ast
import contextlib
import importlib
import math
import os
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import skforecast.utils.utils as _utils  # noqa: E402

# The function that is in the package now
PACKAGE_TRANSFORM_NUMPY = _utils.transform_numpy
REFERENCE_COMMIT = os.environ.get("REVIEW_T1_REFERENCE", "c9172e1")


def _load_reference(commit=REFERENCE_COMMIT):
    """
    `transform_numpy` as it is in `commit`, compiled from the source that git
    stores. The function only needs numpy, pandas, warnings and
    `ColumnTransformer`. Returns None if git cannot read the commit.
    """
    try:
        source = subprocess.run(
            ["git", "-C", str(HERE), "show", f"{commit}:skforecast/utils/utils.py"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    node = next(
        n for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "transform_numpy"
    )
    namespace = {"np": np, "pd": pd, "warnings": warnings,
                 "ColumnTransformer": ColumnTransformer}
    code = compile(
        ast.Module(body=[node], type_ignores=[]), f"<utils.py at {commit}>", "exec",
        flags=__future__.annotations.compiler_flag, dont_inherit=True,
    )
    exec(code, namespace)
    return namespace["transform_numpy"]


_reference = _load_reference()
# Reference of every comparison, and fallback of the candidates
ORIG_TRANSFORM_NUMPY = _reference if _reference is not None else PACKAGE_TRANSFORM_NUMPY
REFERENCE = (
    f"transform_numpy of commit {REFERENCE_COMMIT}" if _reference is not None
    else "transform_numpy of the package (git could not read the reference commit: "
         "the comparisons of PACKAGE are trivial)"
)

# Every module of the package that imports `transform_numpy` by name
# (`grep -n "transform_numpy" skforecast/ --include=*.py`, tests excluded).
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
    """Conditions of `prototypes.fast_transform_numpy` (aligned after the review)."""
    from prototypes import scaler_shortcut_applies

    return scaler_shortcut_applies(array, transformer, fit, inverse_transform)


def plan_literal(
    array, transformer, fit=False, inverse_transform=False, force_single_column=False
):
    """
    T1 exactly as PLAN.md was written before the review (commit 9d02780): the
    fast path is tried right after `if transformer is None` (point 1), with
    the conditions of point 2 in their order and nothing else (no check of the
    type of `array`), the arithmetic of point 3, the copy of point 4 and the
    inf check of point 5.
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
    """
    Wrap a `transform_numpy` and count the calls that take the shortcut: those
    with a `StandardScaler` in which no `transform` nor `inverse_transform` of
    the scaler runs. It does not need to know the conditions, so it also
    works with the implementation in the package. `calls` counts the calls
    with a transformer, of any type and with any value of `fit`.
    """

    _METHODS = ("transform", "inverse_transform")

    def __init__(self, func):
        self.func = func
        self.calls = self.hits = 0

    def __call__(self, array, transformer, fit=False, inverse_transform=False,
                 force_single_column=False):
        if transformer is None:
            return self.func(array=array, transformer=transformer, fit=fit,
                             inverse_transform=inverse_transform,
                             force_single_column=force_single_column)
        self.calls += 1
        used = []
        originals = {name: getattr(StandardScaler, name) for name in self._METHODS}

        def spy(name):
            def method(scaler, *args, **kwargs):
                used.append(name)
                return originals[name](scaler, *args, **kwargs)
            return method

        for name in originals:
            setattr(StandardScaler, name, spy(name))
        try:
            return self.func(array=array, transformer=transformer, fit=fit,
                             inverse_transform=inverse_transform,
                             force_single_column=force_single_column)
        finally:
            for name, method in originals.items():
                setattr(StandardScaler, name, method)
            self.hits += type(transformer) is StandardScaler and not used


@contextlib.contextmanager
def patched(func, modules=None):
    """Replace `transform_numpy` by `func` in every module that imports it."""
    mods = import_modules() if modules is None else modules
    originals = [m.transform_numpy for m in mods]
    for m in mods:
        m.transform_numpy = func
    try:
        yield
    finally:
        for m, original in zip(mods, originals):
            m.transform_numpy = original


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
    import sklearn

    import skforecast

    return (
        f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "
        f"scikit-learn {sklearn.__version__}, numba {numba.__version__}, "
        f"skforecast {skforecast.__version__}\n"
        f"Reference (\"current\"): {REFERENCE}"
    )
