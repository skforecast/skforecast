# Unit test skforecast version
# ==============================================================================
import tomllib
from skforecast import __version__

version = "0.28.0"


def test_version_in_init_and_pyproject_toml():
    """
    Test that version in __init__.py and pyproject.toml are the same.

    Parameters
    ----------
    version : str, optional
    """

    with open("./pyproject.toml", mode='rb') as fp:
        pyproject = tomllib.load(fp)

    version_in_tolm = pyproject['project']['version']
    version_in_init = __version__

    assert version_in_init == version
    assert version_in_tolm == version    


# Test are located inside each module's folder