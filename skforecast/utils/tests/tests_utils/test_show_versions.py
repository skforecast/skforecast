# Unit test show_versions
# ==============================================================================
from skforecast.utils import show_versions


def test_show_versions():
    """
    Test show versions function.
    """
    
    show_versions()
    s = show_versions(as_str=True)

    assert isinstance(s, str)
    assert "System" in s
    assert "Python dependencies" in s
