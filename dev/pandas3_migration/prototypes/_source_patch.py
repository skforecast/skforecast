"""
Source-level monkeypatching of skforecast methods and functions, so that a
proposed change can be measured without touching the library.

A prototype module defines a list of (target, replacements) and exposes
apply() and restore(). `target` is a class attribute ("module:Class.method")
or a module function ("module:function"); `replacements` is a list of
(old, new) text pairs applied to the source of that callable, which is then
re-executed in the namespace of its module and installed in place of the
original. Decorators in the source (e.g. @manage_warnings) are resolved from
the module namespace, so they keep working.
"""

from __future__ import annotations

import importlib
import inspect

_ORIGINALS: dict[str, object] = {}


def _resolve(target: str):
    mod_name, qual = target.split(":")
    module = importlib.import_module(mod_name)
    parts = qual.split(".")
    owner = module
    for p in parts[:-1]:
        owner = getattr(owner, p)
    return module, owner, parts[-1]


def patch_source(target: str, replacements: list[tuple[str, str]]) -> None:
    module, owner, name = _resolve(target)
    original = getattr(owner, name)
    _ORIGINALS.setdefault(target, original)
    func = original
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    # The source keeps its original indentation (wrapped in `if True:`), so
    # the replacement texts are written exactly as they appear in the file.
    src = inspect.getsource(func)
    for old, new in replacements:
        if old not in src:
            raise ValueError(f"{target}: text not found:\n{old}")
        src = src.replace(old, new)
    if src[:1].isspace():
        src = "if True:\n" + src
    ns = module.__dict__
    exec(compile(src, f"<patched {target}>", "exec"), ns)  # noqa: S102
    patched = ns[name]
    if func is not original:
        # Re-apply the outermost decorator chain found in the source (the
        # exec above already applied the decorators written in the source).
        pass
    setattr(owner, name, patched)


def restore_all() -> None:
    for target, original in _ORIGINALS.items():
        _, owner, name = _resolve(target)
        setattr(owner, name, original)


def is_pandas3() -> bool:
    import pandas as pd
    return int(pd.__version__.split(".")[0]) >= 3
