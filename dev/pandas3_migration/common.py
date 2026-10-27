"""
Helpers shared by the pandas 3 migration experiments.

Measurement protocol (used by every script in this folder):

- Each variant is a zero-argument callable. ``ab_interleaved`` runs the
  variants interleaved (A, B, A, B, ...) so that both see the same machine
  state, after ``warmup`` discarded runs, and reports median and minimum of
  ``repeats`` wall-clock times (``time.perf_counter``).
- Peak memory is measured in a separate pass with ``tracemalloc`` (Python and
  numpy allocations) and with the resident set size high-water mark of the
  process (``VmHWM`` in ``/proc/self/status``, reset before each run through
  ``/proc/self/clear_refs``). Timing and memory passes never overlap.
- ``env_info`` records Python and package versions next to every number.
- ``fingerprint`` hashes outputs (values, index, columns, dtypes) so that
  results are compared across versions with hashes, never by eye.

Run every script one at a time, never in parallel with other processes.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------
def env_info() -> dict[str, str]:
    """Versions of Python and of the packages involved in the measurements."""
    info = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": str(os.cpu_count()),
    }
    for name in [
        "pandas", "numpy", "sklearn", "scipy", "numba", "statsmodels",
        "lightgbm", "skops", "joblib", "pyarrow",
    ]:
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "?")
        except Exception:
            info[name] = "not installed"
    try:
        import skforecast
        info["skforecast"] = skforecast.__version__
        info["skforecast_path"] = os.path.dirname(skforecast.__file__)
    except Exception:
        info["skforecast"] = "not importable"
    return info


def pandas_major() -> int:
    return int(pd.__version__.split(".")[0])


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def _time_once(fn: Callable[[], Any]) -> float:
    gc.collect()
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def ab_interleaved(
    variants: dict[str, Callable[[], Any]],
    repeats: int = 7,
    warmup: int = 2,
) -> dict[str, dict[str, float | list[float]]]:
    """
    Time several variants interleaved. Returns, per variant, the list of
    times, the median and the minimum (seconds).
    """
    names = list(variants)
    for _ in range(warmup):
        for name in names:
            _time_once(variants[name])
    times: dict[str, list[float]] = {name: [] for name in names}
    for i in range(repeats):
        order = names if i % 2 == 0 else names[::-1]
        for name in order:
            times[name].append(_time_once(variants[name]))
    out: dict[str, dict[str, float | list[float]]] = {}
    for name in names:
        t = times[name]
        out[name] = {
            "median_s": statistics.median(t),
            "min_s": min(t),
            "max_s": max(t),
            "times_s": t,
        }
    return out


# ----------------------------------------------------------------------------
# Peak memory
# ----------------------------------------------------------------------------
def _read_vmhwm_kb() -> int | None:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _reset_vmhwm() -> None:
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def peak_memory(fn: Callable[[], Any], repeats: int = 3) -> dict[str, float]:
    """
    Peak memory of ``fn``: tracemalloc peak (MiB, Python + numpy allocations,
    minimum over ``repeats``) and the increase of the resident set size
    high-water mark (MiB, minimum over ``repeats``; None when not available).
    """
    tm_peaks = []
    rss_peaks = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tm_peaks.append(peak / 2**20)
        gc.collect()
    for _ in range(repeats):
        gc.collect()
        _reset_vmhwm()
        before = _read_vmhwm_kb()
        fn()
        after = _read_vmhwm_kb()
        if before is not None and after is not None:
            rss_peaks.append((after - before) / 2**10)
        gc.collect()
    return {
        "tracemalloc_peak_mib": min(tm_peaks),
        "rss_hwm_increase_mib": min(rss_peaks) if rss_peaks else None,
    }


# ----------------------------------------------------------------------------
# Fingerprints
# ----------------------------------------------------------------------------
def _hash_update_array(h: "hashlib._Hash", arr: np.ndarray, strict: bool) -> None:
    arr = np.asarray(arr)
    if arr.dtype.kind == "M":
        # Datetime: normalised to int64 nanoseconds, the unit is reported apart.
        if strict:
            h.update(str(arr.dtype).encode())
        arr = arr.astype("datetime64[ns]").astype("int64")
    elif arr.dtype.kind == "m":
        if strict:
            h.update(str(arr.dtype).encode())
        arr = arr.astype("timedelta64[ns]").astype("int64")
    elif arr.dtype.kind in "OUST":
        if strict:
            h.update(str(arr.dtype).encode())
        # Strings/objects: repr of each element (NaN and None kept distinct).
        h.update(json.dumps([repr(x) for x in arr.ravel().tolist()]).encode())
        h.update(str(arr.shape).encode())
        return
    else:
        h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())


def _index_payload(index: pd.Index, strict: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": type(index).__name__}
    if strict:
        payload["dtype"] = str(index.dtype)
        if isinstance(index, pd.DatetimeIndex):
            payload["freq"] = index.freqstr
            payload["tz"] = str(index.tz)
    payload["names"] = [str(n) for n in index.names]
    return payload


def fingerprint(obj: Any, strict: bool = True) -> str:
    """
    SHA-256 of the content of ``obj``.

    ``strict=True`` includes dtype names (``object`` vs ``str``), datetime
    units (``[ns]`` vs ``[us]``) and index frequencies. ``strict=False``
    normalises them (datetimes to nanoseconds, strings to their repr) so
    that only the values, the labels and the shape are compared.
    """
    h = hashlib.sha256()
    _fingerprint_into(h, obj, strict)
    return h.hexdigest()


def _fingerprint_into(h: "hashlib._Hash", obj: Any, strict: bool) -> None:
    if isinstance(obj, pd.DataFrame):
        h.update(b"DataFrame")
        h.update(json.dumps(_index_payload(obj.index, strict)).encode())
        _hash_update_array(h, obj.index.to_numpy(), strict)
        if isinstance(obj.index, pd.MultiIndex):
            for lvl in range(obj.index.nlevels):
                _hash_update_array(h, obj.index.get_level_values(lvl).to_numpy(), strict)
        h.update(json.dumps([str(c) for c in obj.columns]).encode())
        for col in obj.columns:
            s = obj[col] if not isinstance(obj[col], pd.DataFrame) else obj[col].iloc[:, 0]
            _fingerprint_series_values(h, s, strict)
    elif isinstance(obj, pd.Series):
        h.update(b"Series")
        h.update(json.dumps(_index_payload(obj.index, strict)).encode())
        _hash_update_array(h, obj.index.to_numpy(), strict)
        if isinstance(obj.index, pd.MultiIndex):
            for lvl in range(obj.index.nlevels):
                _hash_update_array(h, obj.index.get_level_values(lvl).to_numpy(), strict)
        h.update(str(obj.name).encode())
        _fingerprint_series_values(h, obj, strict)
    elif isinstance(obj, pd.Index):
        h.update(b"Index")
        h.update(json.dumps(_index_payload(obj, strict)).encode())
        if strict:
            h.update(("pd_dtype:" + str(obj.dtype)).encode())
        _hash_update_array(h, obj.to_numpy(), strict)
    elif isinstance(obj, np.ndarray):
        h.update(b"ndarray")
        _hash_update_array(h, obj, strict)
    elif isinstance(obj, dict):
        h.update(b"dict")
        for k in obj:
            h.update(repr(k).encode())
            _fingerprint_into(h, obj[k], strict)
    elif isinstance(obj, (list, tuple)):
        h.update(type(obj).__name__.encode())
        for x in obj:
            _fingerprint_into(h, x, strict)
    elif isinstance(obj, (np.floating, float)):
        h.update(repr(float(obj)).encode())
    elif isinstance(obj, (np.integer, int, bool, np.bool_)):
        h.update(repr(int(obj)).encode())
    elif obj is None:
        h.update(b"None")
    elif isinstance(obj, (np.dtype, pd.api.extensions.ExtensionDtype)):
        h.update(("dtype:" + str(obj)).encode() if strict else b"dtype")
    else:
        h.update(repr(obj).encode())


def _fingerprint_series_values(h: "hashlib._Hash", s: pd.Series, strict: bool) -> None:
    if strict:
        # The pandas dtype, not only the numpy one: the pandas 3 ``str`` dtype
        # converts to an object ndarray, which would hide the difference.
        h.update(("pd_dtype:" + str(s.dtype)).encode())
    if isinstance(s.dtype, pd.CategoricalDtype):
        h.update(b"category")
        if strict:
            h.update(str(s.cat.categories.dtype).encode())
            h.update(str(s.cat.ordered).encode())
        _hash_update_array(h, s.cat.categories.to_numpy(), strict)
        _hash_update_array(h, s.cat.codes.to_numpy(), strict)
    else:
        _hash_update_array(h, s.to_numpy(), strict)


def describe(obj: Any) -> dict[str, Any]:
    """Human-readable dtype/index/unit summary stored next to the hashes."""
    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": list(obj.shape),
            "index_dtype": str(obj.index.dtype),
            "index_freq": getattr(obj.index, "freqstr", None),
            "dtypes": {str(c): str(d) for c, d in obj.dtypes.items()},
        }
    if isinstance(obj, pd.Series):
        return {
            "type": "Series",
            "shape": list(obj.shape),
            "index_dtype": str(obj.index.dtype),
            "index_freq": getattr(obj.index, "freqstr", None),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, np.ndarray):
        return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {"type": "dict", "keys": [str(k) for k in obj]}
    return {"type": type(obj).__name__}


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
def save_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)


def result_path(script_name: str, suffix: str = "") -> str:
    """Path of the JSON result of a script, suffixed with the pandas version."""
    here = os.path.dirname(os.path.abspath(__file__))
    tag = f"pandas{pd.__version__}"
    if suffix:
        tag += f"_{suffix}"
    return os.path.join(here, "results", f"{script_name}_{tag}.json")
