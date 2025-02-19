# Unit test check_residuals_input
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.utils import check_residuals_input
from skforecast.exceptions import UnknownLevelWarning


@pytest.mark.parametrize("residuals", 
                         [None, {}, np.array([])],
                         ids = lambda res: f'residuals: {res}')
@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_ValueError_when_not_in_sample_residuals(residuals, use_binned_residuals):
    """
    Test ValueError is raised when there is no in_sample_residuals_ or 
    in_sample_residuals_by_bin_.
    """

    if use_binned_residuals:
        literal = "in_sample_residuals_by_bin_"
    else:
        literal = "in_sample_residuals_"

    err_msg = re.escape(
        f"`forecaster.{literal}` is either None or empty. Use "
        f"`store_in_sample_residuals = True` when fitting the "
        f"forecaster to store in-sample residuals."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(
            forecaster_name              = 'ForecasterRecursive',
            use_in_sample_residuals      = True,
            in_sample_residuals_         = residuals,
            out_sample_residuals_        = None,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = residuals,
            out_sample_residuals_by_bin_ = None
        )


@pytest.mark.parametrize("residuals", 
                         [None, {}, np.array([])],
                         ids = lambda res: f'residuals: {res}')
@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_ValueError_when_not_out_sample_residuals(residuals, use_binned_residuals):
    """
    Test ValueError is raised when there is no out_sample_residuals_ or 
    out_sample_residuals_by_bin_.
    """

    if use_binned_residuals:
        literal = "out_sample_residuals_by_bin_"
    else:
        literal = "out_sample_residuals_"

    err_msg = re.escape(
        f"`forecaster.{literal}` is either None or empty. Use "
        f"`use_in_sample_residuals = True` or the "
        f"`set_out_sample_residuals()` method before predicting."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(
            forecaster_name              = 'ForecasterRecursive',
            use_in_sample_residuals      = False,
            in_sample_residuals_         = None,
            out_sample_residuals_        = residuals,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = None,
            out_sample_residuals_by_bin_ = residuals
        )


@pytest.mark.parametrize("use_in_sample_residuals", 
                         [True, False],
                         ids = lambda in_sample: f'use_in_sample_residuals: {in_sample}')
def test_check_residuals_input_ValueError_when_not_in_sample_residuals_for_some_step(use_in_sample_residuals):
    """
    Test ValueError is raised when there is no in_sample_residuals_ or 
    in_sample_residuals_by_bin_ for some step in a Direct forecaster.
    """

    if use_in_sample_residuals:
        literal = "in_sample_residuals_"
    else:
        literal = "out_sample_residuals_"
    residuals = {2: np.array([1, 2, 3])}

    err_msg = re.escape(
        f"`forecaster.{literal}` doesn't contain residuals for steps: "
        f"{set([1, 2]) - set(residuals.keys())}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(
            forecaster_name              = 'ForecasterDirect',
            steps                        = [1, 2],
            use_in_sample_residuals      = use_in_sample_residuals,
            in_sample_residuals_         = residuals,
            out_sample_residuals_        = residuals,
            use_binned_residuals         = False,
            in_sample_residuals_by_bin_  = None,
            out_sample_residuals_by_bin_ = None
        )


@pytest.mark.parametrize("use_in_sample_residuals", 
                         [True, False],
                         ids = lambda in_sample: f'use_in_sample_residuals: {in_sample}')
def test_check_residuals_input_ValueError_when_residuals_for_some_step_is_None(use_in_sample_residuals):
    """
    Test ValueError is raised when residuals for some step is None or empty in 
    a Direct forecaster.
    """
    if use_in_sample_residuals:
        literal = "in_sample_residuals_"
    else:
        literal = "out_sample_residuals_"
    residuals = {
        1: {1: np.array([1, 2, 3, 4, 5])},
        2: {2: np.array([1, 2, 3, 4, 5])},
        3: None
    }

    err_msg = re.escape(
        f"Residuals for step 3 are None. Check `forecaster.{literal}`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(
            forecaster_name              = 'ForecasterDirectMultiVariate',
            steps                        = [1, 2, 3],
            use_in_sample_residuals      = use_in_sample_residuals,
            in_sample_residuals_         = residuals,
            out_sample_residuals_        = residuals,
            use_binned_residuals         = False,
            in_sample_residuals_by_bin_  = None,
            out_sample_residuals_by_bin_ = None
        )


@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_multiseries_ValueError_when_not_in_sample_residuals_for_any_level(use_binned_residuals):
    """
    Test ValueError is raised when there is no in_sample_residuals_ for any level
    in 'ForecasterRecursiveMultiSeries'.
    """
    levels = ['1', '2']
    residuals = {'1': np.array([1, 2, 3, 4, 5])}

    if use_binned_residuals:
        residuals = {
            '1': {1: np.array([1, 2, 3, 4, 5])},
            '_unknown_level': {1: np.array([1, 2, 3, 4, 5])}
        }
        literal = "in_sample_residuals_by_bin_"
    else:
        residuals = {
            '1': np.array([1, 2, 3, 4, 5]),
            '_unknown_level': np.array([1, 2, 3, 4, 5])
        }
        literal = "in_sample_residuals_"

    warn_msg = re.escape(
        f"`levels` {set('2')} are not present in `forecaster.{literal}`, "
        f"most likely because they were not present in the training data. "
        f"A random sample of the residuals from other levels will be used. "
        f"This can lead to inaccurate intervals for the unknown levels."
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        check_residuals_input(
            forecaster_name              = 'ForecasterRecursiveMultiSeries',
            levels                       = levels,
            encoding                     ='ordinal',
            use_in_sample_residuals      = True,
            in_sample_residuals_         = residuals,
            out_sample_residuals_        = None,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = residuals,
            out_sample_residuals_by_bin_ = None,
        )


@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_multiseries_ValueError_when_not_out_sample_residuals_for_any_level(use_binned_residuals):
    """
    Test ValueError is raised when there is no out_sample_residuals_ for any level
    in 'ForecasterRecursiveMultiSeries'.
    """
    levels = ['1', '2']
    residuals = {'1': np.array([1, 2, 3, 4, 5])}

    if use_binned_residuals:
        residuals = {
            '1': {1: np.array([1, 2, 3, 4, 5])},
            '_unknown_level': {1: np.array([1, 2, 3, 4, 5])}
        }
        literal = "out_sample_residuals_by_bin_"
    else:
        residuals = {
            '1': np.array([1, 2, 3, 4, 5]),
            '_unknown_level': np.array([1, 2, 3, 4, 5])
        }
        literal = "out_sample_residuals_"

    warn_msg = re.escape(
        f"`levels` {set('2')} are not present in `forecaster.{literal}`. "
        f"A random sample of the residuals from other levels will be used. "
        f"This can lead to inaccurate intervals for the unknown levels. "
        f"Otherwise, Use the `set_out_sample_residuals()` method before "
        f"predicting to set the residuals for these levels.",
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        check_residuals_input(
            forecaster_name              = 'ForecasterRecursiveMultiSeries',
            levels                       = levels,
            encoding                     ='ordinal',
            use_in_sample_residuals      = False,
            in_sample_residuals_         = None,
            out_sample_residuals_        = residuals,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = None,
            out_sample_residuals_by_bin_ = residuals,
        )


@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_ValueError_when_residuals_for_some_level_is_None(use_binned_residuals):
    """
    Test ValueError is raised when residuals for some level is None or empty
    in 'ForecasterRecursiveMultiSeries'.
    """
    levels = ['1', '2']

    if use_binned_residuals:
        residuals = {
            'l1': {1: np.array([1, 2, 3, 4, 5])},
            'l2': {2: np.array([1, 2, 3, 4, 5])},
            'l3': {},
            '_unknown_level': {1: np.array([1, 2, 3, 4, 5])}
        }
        use_in_sample_residuals = False
        literal = "out_sample_residuals_by_bin_"
    else:
        residuals = {
            'l1': np.array([1, 2, 3, 4, 5]),
            'l2': np.array([1, 2, 3, 4, 5]),
            'l3': None,
            '_unknown_level': np.array([1, 2, 3, 4, 5])
        }
        use_in_sample_residuals = True
        literal = "in_sample_residuals_"

    err_msg = re.escape(
        f"Residuals for level 'l3' are None. Check `forecaster.{literal}`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(
            forecaster_name              = 'ForecasterRecursiveMultiSeries',
            levels                       = levels,
            encoding                     ='ordinal',
            use_in_sample_residuals      = use_in_sample_residuals,
            in_sample_residuals_         = residuals,
            out_sample_residuals_        = residuals,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = residuals,
            out_sample_residuals_by_bin_ = residuals
        )
