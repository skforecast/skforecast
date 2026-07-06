"""
Test harness for the scikit-learn compatibility check.

Provides the primitives shared by all checks: the minimum supported
scikit-learn version, the ``CheckResult`` container, version parsing, warning
filtering, and the ``run_check`` runner that isolates scikit-learn deprecation
warnings and errors. See ``check_sklearn_compatibility.py`` for the entry point.
"""

from __future__ import annotations

import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

# Minimum scikit-learn version pinned in pyproject.toml.
MIN_SKLEARN_VERSION = "1.4"


@dataclass
class CheckResult:
    """Outcome of a single compatibility check."""

    name: str
    passed: bool
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


def version_tuple(version: str) -> tuple[int, ...]:
    """Convert a dotted version string into a comparable tuple of ints."""

    parts = []
    for chunk in version.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _collect_sklearn_warnings(record: list[warnings.WarningMessage]) -> list[str]:
    """Keep only Future/Deprecation warnings coming from scikit-learn."""

    messages = []
    for entry in record:
        if not issubclass(entry.category, (FutureWarning, DeprecationWarning)):
            continue
        filename = str(entry.filename)
        if "sklearn" not in filename and "scikit" not in filename:
            continue
        messages.append(f"{entry.category.__name__}: {entry.message}")
    return messages


def run_check(name: str, func: Callable[[], None]) -> CheckResult:
    """Run ``func`` capturing scikit-learn deprecation warnings and errors."""

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        try:
            func()
        except Exception:  # noqa: BLE001 - report any failure, do not stop the run
            return CheckResult(
                name=name,
                passed=False,
                warnings=_collect_sklearn_warnings(record),
                error=traceback.format_exc(),
            )
    return CheckResult(
        name=name, passed=True, warnings=_collect_sklearn_warnings(record)
    )
