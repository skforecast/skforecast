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


@pytest.mark.parametrize(
    "input_type",
    ["ndarray", "pd_series", "y_true_y_pred"],
)
def test_plot_residuals_output(input_type):
    """
    Test that plot_residuals returns a Figure with 3 axes (residuals over time,
    histogram, ACF) regardless of the input type: numpy ndarray, pandas Series,
    or computed internally from y_true and y_pred.
    """
    rng = np.random.default_rng(123)
    if input_type == "ndarray":
        fig = plot_residuals(residuals=rng.standard_normal(100))
    elif input_type == "pd_series":
        fig = plot_residuals(residuals=pd.Series(rng.standard_normal(100)))
    else:
        fig = plot_residuals(
            y_true=rng.standard_normal(100),
            y_pred=rng.standard_normal(100),
        )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 3
    assert len(fig.axes[0].lines) >= 1    # residuals over time
    assert len(fig.axes[1].patches) >= 1  # histogram
    assert len(fig.axes[2].lines) >= 1    # ACF


def test_plot_residuals_output_custom_fig_and_titles():
    """
    Test that plot_residuals uses a provided Figure (identity check) and sets
    the expected subplot titles.
    """
    rng = np.random.default_rng(123)
    residuals = rng.standard_normal(100)
    custom_fig = plt.figure()

    fig = plot_residuals(residuals=residuals, fig=custom_fig)

    assert fig is custom_fig
    assert fig.axes[0].get_title() == "Residuals"
    assert fig.axes[1].get_title() == "Distribution"
    assert fig.axes[2].get_title() == "Autocorrelation"
