"""
Shared helpers of the pandas 3 migration harness: fingerprints of outputs,
memory sharing between objects, capture of pandas warnings emitted by
skforecast and a description of the running environment.

Two fingerprints are computed for every object:

- strict: values plus the pandas dtype of every column, the unit and `freq`
  of datetime indexes and the type of each index.
- normalized: values only. Datetimes are converted to integer nanoseconds,
  text is compared by value whatever its dtype (`object` or `str`), and
  categoricals by codes and categories.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import warnings
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def fingerprint(obj, strict: bool) -> str:
    """
    SHA-256 of an object, strict or normalized (see the module docstring).
    """
    h = hashlib.sha256()
    _feed(h, obj, strict)
    return h.hexdigest()


def fingerprints(obj) -> dict:
    """
    Both fingerprints of an object, plus a short description used to explain
    a difference of the strict fingerprint.
    """
    return {
        "strict": fingerprint(obj, True),
        "normalized": fingerprint(obj, False),
        "describe": describe(obj),
    }


def _describe_index(idx: pd.Index) -> str:
    text = f"{type(idx).__name__}[{idx.dtype}]"
    freq = getattr(idx, "freqstr", None)
    if freq is not None:
        text += f" freq={freq}"
    return text


def describe(obj, depth: int = 0):
    """
    Types, dtypes and index of an object (no values).
    """
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": list(obj.shape),
            "dtypes": sorted({f"{c}: {t}" for c, t in obj.dtypes.astype(str).items()}),
            "columns": _describe_index(obj.columns),
            "index": _describe_index(obj.index),
        }
    if isinstance(obj, pd.Series):
        return {
            "type": "Series",
            "len": len(obj),
            "dtype": str(obj.dtype),
            "index": _describe_index(obj.index),
        }
    if isinstance(obj, pd.Index):
        return _describe_index(obj)
    if isinstance(obj, np.ndarray):
        return f"ndarray[{obj.dtype}] {obj.shape}"
    if depth < 2 and isinstance(obj, dict):
        return {str(k): describe(v, depth + 1) for k, v in list(obj.items())[:20]}
    if depth < 2 and isinstance(obj, (list, tuple)):
        return [describe(v, depth + 1) for v in list(obj)[:20]]
    if isinstance(obj, (np.dtype, pd.api.extensions.ExtensionDtype)):
        return str(obj)
    return type(obj).__name__


def _tag(h, text: str) -> None:
    h.update(text.encode("utf-8") + b"\x00")


def _is_na_scalar(x) -> bool:
    if x is None or x is pd.NA or x is pd.NaT:
        return True
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return True
    return False


def _is_text_dtype(dtype) -> bool:
    return (isinstance(dtype, np.dtype) and dtype.kind == "O") or isinstance(
        dtype, pd.StringDtype
    )


def _dtype_family(dtype) -> str:
    """
    dtype as seen by the normalized fingerprint: `object` and `str` are both
    text, and datetimes lose their unit.
    """
    if _is_text_dtype(dtype):
        return "text"
    if isinstance(dtype, pd.DatetimeTZDtype):
        return f"datetime64[{dtype.tz}]"
    if isinstance(dtype, np.dtype) and dtype.kind == "M":
        return "datetime64"
    if isinstance(dtype, np.dtype) and dtype.kind == "m":
        return "timedelta64"
    return str(dtype)


def _feed(h, obj, strict: bool) -> None:
    if obj is None:
        _tag(h, "None")
    elif isinstance(obj, (bool, np.bool_)):
        _tag(h, f"bool:{bool(obj)}")
    elif isinstance(obj, (int, np.integer)):
        _tag(h, f"int:{int(obj)}")
    elif isinstance(obj, (float, np.floating)):
        _tag(h, f"float:{float(obj)!r}")
    elif isinstance(obj, str):
        _tag(h, f"str:{obj}")
    elif isinstance(obj, pd.DataFrame):
        _feed_frame(h, obj, strict)
    elif isinstance(obj, pd.Series):
        _tag(h, f"Series:{len(obj)}")
        _feed(h, obj.name, strict)
        if strict:
            _tag(h, f"dtype:{obj.dtype}")
        _feed_index(h, obj.index, strict)
        _feed_array(h, obj.array, strict)
    elif isinstance(obj, pd.Index):
        _feed_index(h, obj, strict)
    elif isinstance(obj, (np.ndarray, pd.api.extensions.ExtensionArray)):
        _feed_array(h, obj, strict)
    elif isinstance(obj, dict):
        _tag(h, f"dict:{len(obj)}")
        for key, value in obj.items():
            _feed(h, key, strict)
            _feed(h, value, strict)
    elif isinstance(obj, (list, tuple)):
        _tag(h, f"{type(obj).__name__}:{len(obj)}")
        for value in obj:
            _feed(h, value, strict)
    elif isinstance(obj, (set, frozenset)):
        # The iteration order of a set of strings changes between processes
        # (hash randomization): sort by the fingerprint of each element.
        _tag(h, f"set:{len(obj)}")
        for fp in sorted(fingerprint(value, strict) for value in obj):
            _tag(h, fp)
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(obj)
        _tag(h, f"Timestamp:{ts.tz}:{ts.as_unit('ns').value}")
        if strict:
            _tag(h, f"unit:{ts.unit}")
    elif isinstance(obj, (pd.Timedelta, np.timedelta64)):
        td = pd.Timedelta(obj)
        _tag(h, f"Timedelta:{td.as_unit('ns').value}")
        if strict:
            _tag(h, f"unit:{td.unit}")
    elif isinstance(obj, (np.dtype, pd.api.extensions.ExtensionDtype)):
        _tag(h, f"dtype:{obj if strict else _dtype_family(obj)}")
    elif isinstance(obj, pd.offsets.BaseOffset):
        _tag(h, f"offset:{obj.freqstr}")
    elif isinstance(obj, type):
        _tag(h, f"type:{obj.__module__}.{obj.__qualname__}")
    else:
        _tag(h, f"repr:{type(obj).__name__}:{obj!r}")


def _feed_frame(h, df: pd.DataFrame, strict: bool) -> None:
    _tag(h, f"DataFrame:{df.shape}")
    _feed_index(h, df.columns, strict)
    _feed_index(h, df.index, strict)
    for i in range(df.shape[1]):
        col = df.iloc[:, i]
        if strict:
            _tag(h, f"dtype:{col.dtype}")
        _feed_array(h, col.array, strict)


def _feed_index(h, idx: pd.Index, strict: bool) -> None:
    if isinstance(idx, pd.MultiIndex):
        _tag(h, f"MultiIndex:{idx.nlevels}:{len(idx)}")
        for name in idx.names:
            _feed(h, name, strict)
        for i in range(idx.nlevels):
            _feed_index(h, idx.get_level_values(i), strict)
        return
    _tag(h, f"Index:{len(idx)}")
    _feed(h, idx.name, strict)
    if strict:
        _tag(h, f"type:{type(idx).__name__}:dtype:{idx.dtype}")
        _tag(h, f"freq:{getattr(idx, 'freqstr', None)}")
        if isinstance(idx, pd.RangeIndex):
            _tag(h, f"range:{idx.start}:{idx.stop}:{idx.step}")
            return
    if isinstance(idx, pd.RangeIndex):
        _feed_array(h, np.arange(idx.start, idx.stop, idx.step, dtype="int64"), strict)
    else:
        _feed_array(h, idx.array, strict)


def _feed_array(h, arr, strict: bool) -> None:
    if isinstance(arr, pd.arrays.NumpyExtensionArray):
        arr = arr.to_numpy()
    dtype = arr.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        _tag(h, f"categorical:ordered:{dtype.ordered}")
        if strict:
            _tag(h, f"categories_dtype:{dtype.categories.dtype}")
        _feed_array(h, np.asarray(dtype.categories, dtype=object), strict)
        _feed_array(h, np.asarray(arr.codes, dtype="int64"), strict)
        return

    if isinstance(dtype, pd.DatetimeTZDtype) or (
        isinstance(dtype, np.dtype) and dtype.kind == "M"
    ):
        values = pd.DatetimeIndex(arr)
        _tag(h, f"datetime:{values.tz}:{values.shape}")
        if strict:
            _tag(h, f"dtype:{dtype}")
        h.update(np.ascontiguousarray(values.as_unit("ns").asi8).tobytes())
        return

    if isinstance(dtype, np.dtype) and dtype.kind == "m":
        values = pd.TimedeltaIndex(arr)
        _tag(h, f"timedelta:{values.shape}")
        if strict:
            _tag(h, f"dtype:{dtype}")
        h.update(np.ascontiguousarray(values.as_unit("ns").asi8).tobytes())
        return

    if _is_text_dtype(dtype) or not isinstance(dtype, np.dtype):
        values = np.asarray(arr, dtype=object)
        _tag(h, f"objects:{values.shape}")
        if strict:
            _tag(h, f"dtype:{dtype}")
        for x in values.ravel():
            if _is_na_scalar(x):
                _tag(h, "NA")
            else:
                _feed(h, x, strict)
        return

    values = np.ascontiguousarray(arr)
    _tag(h, f"num:{values.dtype.str}:{values.shape}")
    if values.dtype.kind == "f":
        values = values.copy()
        values[np.isnan(values)] = np.nan  # a single NaN bit pattern
    h.update(values.tobytes())


# ---------------------------------------------------------------------------
# Memory sharing
# ---------------------------------------------------------------------------

def numpy_buffers(obj, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    """
    Numpy arrays that hold the data of an object, with a path for the report.
    Only numpy-backed data is returned (extension arrays are skipped), and
    indexes are ignored: they are immutable, so sharing them is harmless.
    Obtaining the arrays never copies: `to_numpy()` of a numpy-backed column
    is a view.
    """
    found = []
    if isinstance(obj, np.ndarray):
        found.append((prefix or "array", obj))
    elif isinstance(obj, pd.DataFrame):
        for i, col in enumerate(obj.columns):
            values = obj.iloc[:, i].array
            if isinstance(values, pd.arrays.NumpyExtensionArray):
                found.append((f"{prefix}[{col}]", values.to_numpy()))
    elif isinstance(obj, pd.Series):
        if isinstance(obj.array, pd.arrays.NumpyExtensionArray):
            found.append((f"{prefix}.values", obj.array.to_numpy()))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(numpy_buffers(value, f"{prefix}[{key!r}]"))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            found.extend(numpy_buffers(value, f"{prefix}[{i}]"))
    return [(path, arr) for path, arr in found if arr.size > 0]


def shared_memory(outputs: dict, inputs: dict) -> list[str]:
    """
    Pairs `output path <-> input path` whose numpy buffers share memory.
    """
    input_buffers = []
    for name, obj in inputs.items():
        input_buffers.extend(numpy_buffers(obj, name))
    pairs = []
    for name, obj in outputs.items():
        for out_path, out_arr in numpy_buffers(obj, name):
            for in_path, in_arr in input_buffers:
                if np.may_share_memory(out_arr, in_arr) and np.shares_memory(
                    out_arr, in_arr
                ):
                    pairs.append(f"{out_path} <-> {in_path}")
    return sorted(set(pairs))


# ---------------------------------------------------------------------------
# Warnings emitted from the skforecast package
# ---------------------------------------------------------------------------

def _package_dir() -> Path:
    import skforecast

    return Path(skforecast.__file__).resolve().parent


@contextmanager
def capture_package_warnings():
    """
    Record every warning raised while the block runs and keep the ones
    attributed to a file of the skforecast package (tests excluded) whose
    category is a deprecation (FutureWarning, DeprecationWarning or pandas'
    Pandas4Warning). The result is a list of `category | file:line | message`
    strings with their count.
    """
    package = _package_dir()
    result: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        yield result
    counter: Counter = Counter()
    for w in records:
        if not issubclass(w.category, (FutureWarning, DeprecationWarning)):
            continue
        try:
            path = Path(w.filename).resolve()
            rel = path.relative_to(package)
        except (OSError, ValueError):
            continue
        if "tests" in rel.parts:
            continue
        message = str(w.message).splitlines()[0][:160]
        counter[f"{w.category.__name__} | {rel.as_posix()}:{w.lineno} | {message}"] += 1
    result.extend(f"{key} (x{n})" for key, n in sorted(counter.items()))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def environment() -> dict:
    """
    Versions of the packages that matter for the comparisons, plus the
    location of the imported skforecast (it must be the tree under test).
    """
    import importlib.metadata as md

    import skforecast

    def version(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    pkgs = ["pandas", "numpy", "scikit-learn", "scipy", "numba", "lightgbm",
            "statsmodels", "skops", "joblib", "pyarrow"]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "skforecast_file": skforecast.__file__,
        "skforecast_version": skforecast.__version__,
        "str_storage": str(pd.Series(["a"]).dtype),
        "packages": {p: version(p) for p in pkgs},
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def ensure_hash_seed() -> None:
    """
    Re-run the current script with `PYTHONHASHSEED=0` if it is not set.
    Some results depend on the iteration order of sets of strings (for
    example, `PopulationDriftDetector.predict` builds its chi2 contingency
    table from a set), which changes between processes otherwise.
    """
    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    import subprocess

    env = dict(os.environ, PYTHONHASHSEED="0")
    completed = subprocess.run([sys.executable, *sys.argv], env=env)
    raise SystemExit(completed.returncode)


def check_tree(expected_root: str | None) -> None:
    """
    Abort if skforecast is not imported from `expected_root`. Protects
    against a stale skforecast installed in site-packages.
    """
    if not expected_root:
        return
    import skforecast

    actual = Path(skforecast.__file__).resolve()
    root = Path(expected_root).resolve()
    if root not in actual.parents:
        raise SystemExit(
            f"skforecast imported from {actual}, expected a tree under {root}. "
            "Set PYTHONPATH to the tree root."
        )
