# Unit test check_residuals_input
# ==============================================================================
import re
import pytest
from skforecast.utils import check_residuals_input


@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_ValueError_when_not_in_sample_residuals(use_binned_residuals):
    """
    Test ValueError is raised when there is no in_sample_residuals_ or 
    in_sample_residuals_by_bin_.
    """

    if use_binned_residuals:
        residuals = {}
        literal = "in_sample_residuals_by_bin_"
    else:
        residuals = None
        literal = "in_sample_residuals_"

    err_msg = re.escape(
        f"`forecaster.{literal}` is None. Use `store_in_sample_residuals = True` "
        f"when fitting the forecaster to store in-sample residuals."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(
            use_in_sample_residuals      = True,
            in_sample_residuals_         = residuals,
            out_sample_residuals_        = None,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = residuals,
            out_sample_residuals_by_bin_ = None
        )


@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_ValueError_when_not_out_sample_residuals(use_binned_residuals):
    """
    Test ValueError is raised when there is no out_sample_residuals_ or 
    out_sample_residuals_by_bin_.
    """

    if use_binned_residuals:
        residuals = {}
        literal = "out_sample_residuals_by_bin_"
    else:
        residuals = None
        literal = "out_sample_residuals_"

    err_msg = re.escape(
        f"`forecaster.{literal}` is None. Use `use_in_sample_residuals = True` "
        f"or the `set_out_sample_residuals()` method before predicting."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(
            use_in_sample_residuals      = False,
            in_sample_residuals_         = None,
            out_sample_residuals_        = residuals,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = None,
            out_sample_residuals_by_bin_ = residuals
        )
