"""
BUG-osa-differentiation: which tests fix the current behavior? Runs the tests of
`model_selection` (utils and search) with the prototype of the fix enabled
(`_00_fix_prototype.py`, as a pytest plugin). The tests that fail are the ones
whose expected values were computed with the cumulative sum of the predicted
differences.

Run from the repository root with PYTHONPATH pointing to the code to test.
"""
import importlib.util
import pathlib
import sys
import pytest

_spec = importlib.util.spec_from_file_location(
    "fix_prototype", pathlib.Path(__file__).with_name("BUG-osa-differentiation_00_fix_prototype.py")
)
fix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fix)

paths = sys.argv[1:] or [
    "skforecast/model_selection/tests/tests_utils",
    "skforecast/model_selection/tests/tests_search",
]
sys.exit(pytest.main([*paths, "-q", "-p", "no:cacheprovider", "-p", "no:warnings",
                      "--tb=no", "-rf"], plugins=[fix]))
