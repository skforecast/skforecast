# Unit test import of keras in skforecast.deep_learning
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


@pytest.mark.skipif(
    sys.version_info >= (3, 14), 
    reason="Python 3.14+ raises a specific error about the Keras backend."
)
@pytest.mark.parametrize("module", 
                         ['skforecast.deep_learning._forecaster_rnn', 
                          'skforecast.deep_learning.utils'], 
                         ids = lambda module: f'module: {module}')
def test_import_check_optional_dependency_when_keras_is_not_installed(module):
    """
    Test `check_optional_dependency` is called with 'keras' when keras is not 
    installed.
    """
    result = run_import(HIDE_PACKAGE.format(package='keras') + f"import {module}")

    assert result.returncode != 0
    assert "check_optional_dependency called with 'keras'" in result.stderr


@pytest.mark.skipif(
    sys.version_info >= (3, 14), 
    reason="Python 3.14+ raises a specific error about the Keras backend."
)
@pytest.mark.parametrize("module", 
                         ['skforecast.deep_learning._forecaster_rnn', 
                          'skforecast.deep_learning.utils'], 
                         ids = lambda module: f'module: {module}')
def test_import_shows_real_error_when_keras_fails_to_import(module, tmp_path):
    """
    Test the original error is raised when keras is installed but fails to 
    import. Before, the last word of the message was used as the package name.
    """
    package = tmp_path / 'keras'
    package.mkdir()
    (package / '__init__.py').write_text(
        "raise ImportError(\"cannot import name 'x' from 'backend' (/path/backend)\")"
    )

    result = run_import(f"import {module}", pythonpath=tmp_path)

    assert result.returncode != 0
    assert result.stderr.strip().splitlines()[-1] == (
        "ImportError: cannot import name 'x' from 'backend' (/path/backend)"
    )
