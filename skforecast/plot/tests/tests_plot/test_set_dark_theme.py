# Unit test set_dark_theme
# ==============================================================================
import copy
import pytest
import matplotlib
from skforecast.plot import set_dark_theme

_EXPECTED_FACECOLOR = "#001633"


def test_set_dark_theme_default():
    """
    Test that set_dark_theme returns None and sets the expected rcParam keys
    including facecolor, grid, spines and line width.
    """
    saved = copy.deepcopy(dict(matplotlib.rcParams))
    try:
        result = set_dark_theme()
        assert result is None
        assert matplotlib.rcParams["figure.facecolor"] == _EXPECTED_FACECOLOR
        assert matplotlib.rcParams["axes.facecolor"] == _EXPECTED_FACECOLOR
        assert matplotlib.rcParams["axes.grid"] is True
        assert matplotlib.rcParams["axes.spines.left"] is False
        assert matplotlib.rcParams["axes.spines.right"] is False
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["axes.spines.bottom"] is False
        assert matplotlib.rcParams["lines.linewidth"] == 1.5
    finally:
        matplotlib.rcParams.update(saved)


@pytest.mark.parametrize(
    "custom_style, expected_linewidth",
    [
        ({}, 1.5),
        ({"lines.linewidth": 5.0}, 5.0),
    ],
    ids=["empty_dict", "override_linewidth"],
)
def test_set_dark_theme_custom_style(custom_style, expected_linewidth):
    """
    Test that custom_style values override defaults when provided and that
    passing an empty dict still applies the default dark theme.
    """
    saved = copy.deepcopy(dict(matplotlib.rcParams))
    try:
        set_dark_theme(custom_style=custom_style)
        assert matplotlib.rcParams["figure.facecolor"] == _EXPECTED_FACECOLOR
        assert matplotlib.rcParams["lines.linewidth"] == expected_linewidth
    finally:
        matplotlib.rcParams.update(saved)
