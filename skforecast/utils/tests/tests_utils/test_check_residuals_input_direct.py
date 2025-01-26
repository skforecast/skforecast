# Unit test check_residuals_input_direct
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.utils import check_residuals_input_direct


@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_direct_ValueError_when_not_in_sample_residuals_for_some_step(use_binned_residuals):
    """
    Test ValueError is raised when there is no in_sample_residuals_ or 
    in_sample_residuals_by_bin_ for some step.
    """

    if use_binned_residuals:
        residuals = {2: {1: np.array([1, 2, 3])}}
        literal = "in_sample_residuals_by_bin_"
    else:
        residuals = {2: np.array([1, 2, 3])}
        literal = "in_sample_residuals_"

    err_msg = re.escape(
        f"`forecaster.{literal}` doesn't contain residuals for steps: "
        f"{set([1, 2]) - set(residuals.keys())}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input_direct(
            steps                        = [1, 2],
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
def test_check_residuals_input_direct_ValueError_when_not_out_sample_residuals_for_some_step(use_binned_residuals):
    """
    Test ValueError is raised when there is no out_sample_residuals_ or 
    out_sample_residuals_by_bin_ for some step.
    """

    if use_binned_residuals:
        residuals = {2: {1: np.array([1, 2, 3])}}
        literal = "out_sample_residuals_by_bin_"
    else:
        residuals = {2: np.array([1, 2, 3])}
        literal = "out_sample_residuals_"

    err_msg = re.escape(
        f"`forecaster.{literal}` doesn't contain residuals for steps: "
        f"{set([1, 2]) - set(residuals.keys())}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input_direct(
            steps                        = [1, 2],
            use_in_sample_residuals      = False,
            in_sample_residuals_         = None,
            out_sample_residuals_        = residuals,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = None,
            out_sample_residuals_by_bin_ = residuals
        )


@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_direct_ValueError_when_out_sample_residuals_for_some_step_is_None(use_binned_residuals):
    """
    Test ValueError is raised when out_sample_residuals_ or
    out_sample_residuals_by_bin_ for some step is None.
    """

    if use_binned_residuals:
        residuals = {
            1: {1: np.array([1, 2, 3, 4, 5])},
            2: {2: np.array([1, 2, 3, 4, 5])},
            3: None
        }
        literal = "out_sample_residuals_by_bin_"
    else:
        residuals = {
            1: np.array([1, 2, 3, 4, 5]),
            2: np.array([1, 2, 3, 4, 5]),
            3: None
        }
        literal = "out_sample_residuals_"

    err_msg = re.escape(
        f"Residuals for step 3 are None. Check `forecaster.{literal}`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input_direct(
            steps                        = [1, 2, 3],
            use_in_sample_residuals      = False,
            in_sample_residuals_         = None,
            out_sample_residuals_        = residuals,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = None,
            out_sample_residuals_by_bin_ = residuals
        )
