"""
Scikit-learn compatibility check for skforecast
===============================================

Maintenance utility to evaluate whether the currently installed version of
scikit-learn works correctly with skforecast. It is meant to be run whenever a
new scikit-learn release is published, to catch deprecations, removed public
API, and behavioural changes before they reach users.

The script does not add new functionality to skforecast. It only exercises the
public scikit-learn touch points that skforecast relies on (estimator cloning,
``Pipeline``, ``ColumnTransformer``, estimator tags, feature selection,
hyperparameter search, etc.) across a representative set of forecasters, and
reports:

- Installed versions of skforecast and its core dependencies.
- Whether the installed scikit-learn satisfies the minimum pinned in
  ``pyproject.toml`` (``scikit-learn>=1.4``).
- Any ``FutureWarning`` / ``DeprecationWarning`` emitted by scikit-learn during
  typical skforecast workflows.
- Any error raised while running those workflows.

The implementation is split across sibling modules in ``dev/``:

- ``_sklearn_compat_harness``: ``CheckResult``, ``run_check`` and version utils.
- ``_sklearn_compat_data``: synthetic sample data generators.
- ``_sklearn_compat_checks``: the individual checks and the ``CHECKS`` registry.

Usage
-----
    python dev/check_sklearn_compatibility.py

Exit code is 0 when every check passes (warnings do not fail the run) and 1
when at least one check raises an error, so the script can also be wired into
CI if desired.
"""

from __future__ import annotations

import importlib
import sys

from _sklearn_compat_checks import CHECKS
from _sklearn_compat_harness import MIN_SKLEARN_VERSION, run_check, version_tuple


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _print_versions() -> str:
    """Print installed versions and return the scikit-learn version string."""

    import sklearn

    import skforecast

    print("Environment")
    print("-" * 70)
    print(f"python       : {sys.version.split()[0]}")
    print(f"skforecast   : {skforecast.__version__}")
    print(f"scikit-learn : {sklearn.__version__}")

    for package in ("numpy", "pandas", "scipy", "joblib"):
        try:
            module = importlib.import_module(package)
            print(f"{package:<12} : {module.__version__}")
        except Exception:  # noqa: BLE001
            print(f"{package:<12} : not installed")
    print()
    return sklearn.__version__


def _check_min_version(sklearn_version: str) -> bool:
    ok = version_tuple(sklearn_version) >= version_tuple(MIN_SKLEARN_VERSION)
    status = "OK" if ok else "TOO OLD"
    print(
        f"Minimum required scikit-learn: >={MIN_SKLEARN_VERSION} "
        f"(installed {sklearn_version}) -> {status}"
    )
    print()
    return ok


def main() -> int:
    print("=" * 70)
    print("skforecast <-> scikit-learn compatibility check")
    print("=" * 70)

    sklearn_version = _print_versions()
    version_ok = _check_min_version(sklearn_version)

    results = [run_check(name, func) for name, func in CHECKS.items()]

    print("Checks")
    print("-" * 70)
    n_passed = 0
    n_warned = 0
    for result in results:
        if result.passed:
            n_passed += 1
            marker = "WARN" if result.warnings else "PASS"
        else:
            marker = "FAIL"
        print(f"[{marker}] {result.name}")
        for message in result.warnings:
            n_warned += 1
            print(f"       - {message}")
        if result.error is not None:
            for line in result.error.strip().splitlines():
                print(f"       {line}")
    print()

    print("Summary")
    print("-" * 70)
    n_total = len(results)
    n_failed = n_total - n_passed
    print(f"passed        : {n_passed}/{n_total}")
    print(f"failed        : {n_failed}/{n_total}")
    print(f"sklearn warns : {n_warned}")
    print(f"min version   : {'satisfied' if version_ok else 'NOT satisfied'}")

    all_ok = version_ok and n_failed == 0
    print()
    print("Result: " + ("COMPATIBLE" if all_ok else "ACTION REQUIRED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
