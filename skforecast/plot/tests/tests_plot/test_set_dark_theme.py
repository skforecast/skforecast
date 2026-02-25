# Unit test set_dark_theme
# ==============================================================================
import copy
import matplotlib
from skforecast.plot import set_dark_theme

_EXPECTED_FACECOLOR = "#001633"


def test_set_dark_theme_returns_none():
    """
    Test that set_dark_theme returns None.
    """
    saved = copy.deepcopy(dict(matplotlib.rcParams))
    try:
        result = set_dark_theme()
    finally:
        matplotlib.rcParams.update(saved)

    assert result is None


def test_set_dark_theme_modifies_facecolor():
    """
    Test that set_dark_theme sets figure.facecolor and axes.facecolor to the
    expected dark-navy colour.
    """
    saved = copy.deepcopy(dict(matplotlib.rcParams))
    try:
        set_dark_theme()
        assert matplotlib.rcParams["figure.facecolor"] == _EXPECTED_FACECOLOR
        assert matplotlib.rcParams["axes.facecolor"] == _EXPECTED_FACECOLOR
    finally:
        matplotlib.rcParams.update(saved)


def test_set_dark_theme_modifies_expected_keys():
    """
    Test that set_dark_theme updates a representative set of rcParam keys.
    """
    saved = copy.deepcopy(dict(matplotlib.rcParams))
    try:
        set_dark_theme()
        assert matplotlib.rcParams["axes.grid"] is True
        assert matplotlib.rcParams["axes.spines.left"] is False
        assert matplotlib.rcParams["axes.spines.right"] is False
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["axes.spines.bottom"] is False
        assert matplotlib.rcParams["lines.linewidth"] == 1.5
    finally:
        matplotlib.rcParams.update(saved)


def test_set_dark_theme_custom_style():
    """
    Test that custom_style values override the default dark theme.
    """
    saved = copy.deepcopy(dict(matplotlib.rcParams))
    try:
        set_dark_theme(custom_style={"lines.linewidth": 5.0})
        assert matplotlib.rcParams["lines.linewidth"] == 5.0
        # Base dark theme key should still be set
        assert matplotlib.rcParams["figure.facecolor"] == _EXPECTED_FACECOLOR
    finally:
        matplotlib.rcParams.update(saved)


def test_set_dark_theme_empty_custom_style():
    """
    Test that passing an empty dict for custom_style still applies the default
    dark theme without error.
    """
    saved = copy.deepcopy(dict(matplotlib.rcParams))
    try:
        set_dark_theme(custom_style={})
        assert matplotlib.rcParams["figure.facecolor"] == _EXPECTED_FACECOLOR
    finally:
        matplotlib.rcParams.update(saved)
