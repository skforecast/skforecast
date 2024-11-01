# Unit test check_optional_dependency
# ==============================================================================
from skforecast.utils import optional_dependencies
import tomli


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
