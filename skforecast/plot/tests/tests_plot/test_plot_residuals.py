# Unit test plot_residuals
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skforecast.plot import plot_residuals


def test_plot_residuals_ValueError_when_residuals_are_not_provided():
    """
    Test ValueError if `residuals` argument is None and also `y_true` and `y_pred`
    are None.
    """
    err_msg = re.escape(
        "If `residuals` argument is None then, `y_true` and `y_pred` must be provided."
    )
    with pytest.raises(ValueError, match=err_msg):
        plot_residuals()


def test_plot_residuals_output_from_residuals_array():
    """
    Test that plot_residuals returns a Figure with 3 axes when `residuals` is a
    numpy ndarray.
    """
    matplotlib.use("Agg")
    rng = np.random.default_rng(123)
    residuals = rng.standard_normal(100)

    fig = plot_residuals(residuals=residuals)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3
    assert len(fig.axes[0].lines) >= 1    # residuals over time
    assert len(fig.axes[1].patches) >= 1  # histogram
    assert len(fig.axes[2].lines) >= 1    # ACF
    plt.close(fig)


def test_plot_residuals_output_from_y_true_y_pred():
    """
    Test that plot_residuals returns a Figure when residuals are computed
    internally from y_true and y_pred.
    """
    matplotlib.use("Agg")
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(100)
    y_pred = rng.standard_normal(100)

    fig = plot_residuals(y_true=y_true, y_pred=y_pred)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3
    assert len(fig.axes[0].lines) >= 1
    assert len(fig.axes[1].patches) >= 1
    assert len(fig.axes[2].lines) >= 1
    plt.close(fig)


def test_plot_residuals_output_from_pd_series():
    """
    Test that plot_residuals returns a Figure when `residuals` is a pandas Series.
    """
    matplotlib.use("Agg")
    rng = np.random.default_rng(123)
    residuals = pd.Series(rng.standard_normal(100))

    fig = plot_residuals(residuals=residuals)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3
    assert len(fig.axes[0].lines) >= 1
    assert len(fig.axes[1].patches) >= 1
    assert len(fig.axes[2].lines) >= 1
    plt.close(fig)


def test_plot_residuals_output_with_custom_fig():
    """
    Test that plot_residuals uses the provided Figure object and returns the
    same object (identity check).
    """
    matplotlib.use("Agg")
    rng = np.random.default_rng(123)
    residuals = rng.standard_normal(100)
    custom_fig = plt.figure()

    fig = plot_residuals(residuals=residuals, fig=custom_fig)

    assert fig is custom_fig
    plt.close(fig)


def test_plot_residuals_output_axis_titles():
    """
    Test that plot_residuals sets the expected subplot titles.
    """
    matplotlib.use("Agg")
    rng = np.random.default_rng(123)
    residuals = rng.standard_normal(100)

    fig = plot_residuals(residuals=residuals)

    assert fig.axes[0].get_title() == "Residuals"
    assert fig.axes[1].get_title() == "Distribution"
    assert fig.axes[2].get_title() == "Autocorrelation"
    plt.close(fig)
