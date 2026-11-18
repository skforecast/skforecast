"""
Which skforecast lines make pandas 3 build or convert text arrays (the `str`
dtype of names of series and columns), how many times per call and with how
many elements. Only the outermost call into the string array classes is
counted. The time is indicative (the spy adds overhead).

Usage: python str_spy.py <out.json> <operation> <n_series> <exog_kind> [fmt] [steps]
"""
import common  # noqa: F401

import json
import sys
import time
import types

import pandas as pd
import skforecast
from lineprof import build

ROOT = skforecast.__path__[0]
calls = {}   # (method, file, line) -> [count, seconds, elements]
depth = 0


def _site():
    frame = sys._getframe(2)
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename.startswith(ROOT):
            return filename[len(ROOT) + 1:].replace("\\", "/"), frame.f_lineno
        frame = frame.f_back
    return None


def _size(args):
    for a in args[:2]:
        try:
            return len(a)
        except TypeError:
            continue
    return 0


def _spy(cls_name, name, func):
    def spy(*args, **kwargs):
        global depth
        if depth:
            return func(*args, **kwargs)
        site = _site()
        if site is None:
            return func(*args, **kwargs)
        depth += 1
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            depth -= 1
            rec = calls.setdefault((f"{cls_name}.{name}", *site), [0, 0.0, 0])
            rec[0] += 1
            rec[1] += elapsed
            rec[2] += _size(args[1:] if name == "_from_sequence" else args)
    return spy


SKIP = {"__init__", "__len__", "__new__", "__class__", "dtype", "__setstate__", "__getstate__"}


def install():
    if pd.__version__ < "3":
        return []
    from pandas.core.arrays.string_ import BaseStringArray, StringArray
    classes = [BaseStringArray, StringArray]
    if "pyarrow" in sys.modules or pd.Series(["a"]).dtype.storage == "pyarrow":
        from pandas.core.arrays.arrow.array import ArrowExtensionArray
        from pandas.core.arrays.string_arrow import ArrowStringArray
        classes += [ArrowExtensionArray, ArrowStringArray]
    done = []
    for cls in classes:
        for name, attr in list(vars(cls).items()):
            if name in SKIP:
                continue
            if isinstance(attr, types.FunctionType):
                setattr(cls, name, _spy(cls.__name__, name, attr))
            elif isinstance(attr, classmethod):
                setattr(cls, name, classmethod(_spy(cls.__name__, name, attr.__func__)))
            else:
                continue
            done.append(f"{cls.__name__}.{name}")
    return done


if __name__ == "__main__":
    out, operation, n_series, exog_kind = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    fmt = sys.argv[5] if len(sys.argv) > 5 else "wide"
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    fn, n_units = build(operation, n_series, exog_kind, fmt, steps)
    fn()
    wrapped = install()
    fn()
    src = {}
    rows = []
    for (method, file, line), (count, seconds, elements) in calls.items():
        if file not in src:
            with open(f"{ROOT}/{file}", encoding="utf-8") as fh:
                src[file] = fh.read().splitlines()
        rows.append({
            "method": method, "file": file, "line": line, "calls": count / n_units,
            "us": seconds / n_units * 1e6, "elements": elements / max(count, 1),
            "code": src[file][line - 1].strip()[:80],
        })
    with open(out, "w") as fh:
        json.dump({"rows": rows, "n_units": n_units, "args": sys.argv[2:],
                   "wrapped": len(wrapped), **common.env_info()}, fh)
    by_line = {}
    for r in rows:
        key = (r["file"], r["line"], r["code"])
        rec = by_line.setdefault(key, [0.0, 0.0, set()])
        rec[0] += r["calls"]
        rec[1] += r["us"]
        rec[2].add(r["method"].split(".")[1])
    info = common.env_info()
    print(f"pandas {info['pandas']} storage {info['storage']} {sys.argv[2:]}: "
          f"{sum(v[0] for v in by_line.values()):.1f} text array calls per unit, "
          f"{sum(v[1] for v in by_line.values()):.0f} us (indicative)")
    for (file, line, code), (count, us, methods) in sorted(by_line.items(), key=lambda kv: -kv[1][1]):
        print(f"{count:7.1f} calls {us:8.1f} us  {file.split('/')[-1]}:{line}  {code[:60]}  "
              f"[{', '.join(sorted(methods))[:60]}]")
