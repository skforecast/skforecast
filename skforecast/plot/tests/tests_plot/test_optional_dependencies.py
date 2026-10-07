# Unit test import of optional dependencies in skforecast.plot
# ==============================================================================
import os
import sys
import subprocess
import pytest


# NOTE: Code run before the import to simulate that a package is not installed.
# `check_optional_dependency` is replaced to check the name it receives, since
# the package is installed in the test environment.
HIDE_PACKAGE = """
import sys
import importlib.abc
import skforecast.utils

class HidePackage(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition('.')[0] == '{package}':
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)

sys.meta_path.insert(0, HidePackage())

def check_optional_dependency(package_name):
    raise ImportError(f"check_optional_dependency called with {{package_name!r}}")

skforecast.utils.check_optional_dependency = check_optional_dependency
"""


def run_import(code, pythonpath=None):
    """
    Run `code` in a new Python process and return the result.
    """
    env = os.environ.copy()
    if pythonpath is not None:
        paths = [str(pythonpath), env.get('PYTHONPATH', '')]
        env['PYTHONPATH'] = os.pathsep.join(path for path in paths if path)
    return subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, env=env
    )


@pytest.mark.parametrize("package", 
                         ['matplotlib', 'statsmodels'], 
                         ids = lambda package: f'package: {package}')
def test_import_plot_check_optional_dependency_when_package_is_not_installed(package):
    """
    Test `check_optional_dependency` is called with the name of the package 
    when matplotlib or statsmodels is not installed.
    """
    result = run_import(
        HIDE_PACKAGE.format(package=package) + "import skforecast.plot"
    )

    assert result.returncode != 0
    assert f"check_optional_dependency called with '{package}'" in result.stderr


def test_import_plot_shows_real_error_when_statsmodels_fails_to_import(tmp_path):
    """
    Test the original error is raised when statsmodels is installed but fails 
    to import. Before, the last word of the message was used as the package
    name, which raised `ModuleNotFoundError: No module named '(/path/...'`.
    """
    package = tmp_path / 'statsmodels'
    package.mkdir()
    (package / '__init__.py').write_text(
        "raise ImportError(\"cannot import name 'Int64Index' from 'pandas' (/path/pandas)\")"
    )

    result = run_import("import skforecast.plot", pythonpath=tmp_path)

    assert result.returncode != 0
    assert result.stderr.strip().splitlines()[-1] == (
        "ImportError: cannot import name 'Int64Index' from 'pandas' (/path/pandas)"
    )
