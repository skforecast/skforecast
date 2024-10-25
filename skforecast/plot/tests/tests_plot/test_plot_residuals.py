# Unit test plot_residuals
# ==============================================================================
import re
import pytest
from skforecast.plot import plot_residuals


def test_plot_residuals_exception_when_residuals_are_not_provided():
    """
    Test exception if `residuals` argument is None and also `y_true` and `y_pred`
    are None."
    """
    err_msg = re.escape(
                "If `residuals` argument is None then, `y_true` and `y_pred` must be provided."
              )
    with pytest.raises(ValueError, match = err_msg):
        plot_residuals()