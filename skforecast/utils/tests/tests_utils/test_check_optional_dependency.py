# Unit test check_optional_dependency
# ==============================================================================
import re
import pytest
import tomli
from skforecast.utils import optional_dependencies
from skforecast.utils.utils import _find_optional_dependency


def test_skforecast_utils_optional_dependencies_match_dependences_in_toml():
    """
    Test that optional_dependencies in skforecast/utils/optional_dependencies.py
    match optional-dependencies in pyproject.toml
    """

    with open("./pyproject.toml", mode='rb') as fp:
        pyproject = tomli.load(fp)
    
    optional_dependencies_in_toml = {
        k: v
        for k, v in pyproject['project']['optional-dependencies'].items()
        if k not in ['full', 'all', 'docs', 'test']
    }
    assert optional_dependencies_in_toml == optional_dependencies


def test_find_optional_dependency_ValueError_when_package_not_in_optional_dependencies():
    """
    Test that _find_optional_dependency raises ValueError when the package
    name is not listed in optional_dependencies.
    """
    msg = re.escape("'not_a_real_package' is not listed in optional_dependencies.")
    with pytest.raises(ValueError, match=msg):
        _find_optional_dependency(package_name='not_a_real_package')


def test_find_optional_dependency_output_when_package_is_optional():
    """
    Test that _find_optional_dependency returns the correct extra name and
    package version string for a known optional dependency.
    """
    extra, package_version = _find_optional_dependency(package_name='statsmodels')
    assert extra == 'stats'
    assert 'statsmodels' in package_version
