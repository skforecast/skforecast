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
        f"`store_in_sample_residuals = True` when fitting the forecaster "
        f"or use the `set_in_sample_residuals()` method before predicting."
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


def _residuals_with_level_l3_None(use_binned_residuals):
    """
    Residuals of levels 'l1' and 'l2', None or empty for level 'l3'.
    """
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

    return residuals, use_in_sample_residuals, literal


@pytest.mark.parametrize("forecaster_name", 
                         ['ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 'ForecasterRnn'],
                         ids = lambda fn: f'forecaster_name: {fn}')
@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_ValueError_when_residuals_for_some_level_is_None(forecaster_name, use_binned_residuals):
    """
    Test ValueError is raised when residuals for a level to predict are None 
    or empty.
    """
    residuals, use_in_sample_residuals, literal = _residuals_with_level_l3_None(
        use_binned_residuals
    )

    err_msg = re.escape(
        f"Residuals for level 'l3' are None or empty. Check `forecaster.{literal}`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(
            forecaster_name              = forecaster_name,
            levels                       = ['l1', 'l3'],
            encoding                     ='ordinal',
            use_in_sample_residuals      = use_in_sample_residuals,
            in_sample_residuals_         = residuals,
            out_sample_residuals_        = residuals,
            use_binned_residuals         = use_binned_residuals,
            in_sample_residuals_by_bin_  = residuals,
            out_sample_residuals_by_bin_ = residuals
        )


@pytest.mark.parametrize("forecaster_name", 
                         ['ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 'ForecasterRnn'],
                         ids = lambda fn: f'forecaster_name: {fn}')
@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_no_error_when_residuals_None_only_for_levels_not_predicted(forecaster_name, use_binned_residuals):
    """
    Test no error is raised when residuals are None or empty only for levels 
    that are not predicted. Before, the residuals of all the levels were 
    checked, so storing residuals only for some series made every prediction
    interval fail.
    """
    residuals, use_in_sample_residuals, _ = _residuals_with_level_l3_None(
        use_binned_residuals
    )

    check_residuals_input(
        forecaster_name              = forecaster_name,
        levels                       = ['l1', 'l2'],
        encoding                     ='ordinal',
        use_in_sample_residuals      = use_in_sample_residuals,
        in_sample_residuals_         = residuals,
        out_sample_residuals_        = residuals,
        use_binned_residuals         = use_binned_residuals,
        in_sample_residuals_by_bin_  = residuals,
        out_sample_residuals_by_bin_ = residuals
    )


@pytest.mark.parametrize("use_binned_residuals", 
                         [True, False],
                         ids = lambda binned: f'use_binned_residuals: {binned}')
def test_check_residuals_input_unknown_level_uses_unknown_level_residuals_ForecasterRecursiveMultiSeries(use_binned_residuals):
    """
    Test the residuals of '_unknown_level' are checked for a level without 
    residuals in ForecasterRecursiveMultiSeries, the residuals used to 
    predict it. In the rest of multiseries forecasters, a level without 
    residuals raises a ValueError.
    """
    residuals, use_in_sample_residuals, literal = _residuals_with_level_l3_None(
        use_binned_residuals
    )
    kwargs = {
        'levels': ['l4'],
        'encoding': None,
        'use_in_sample_residuals': use_in_sample_residuals,
        'in_sample_residuals_': residuals,
        'out_sample_residuals_': residuals,
        'use_binned_residuals': use_binned_residuals,
        'in_sample_residuals_by_bin_': residuals,
        'out_sample_residuals_by_bin_': residuals
    }

    check_residuals_input(forecaster_name='ForecasterRecursiveMultiSeries', **kwargs)

    err_msg = re.escape(
        f"Residuals for level 'l4' are None or empty. Check `forecaster.{literal}`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(forecaster_name='ForecasterDirectMultiVariate', **kwargs)

    residuals['_unknown_level'] = {} if use_binned_residuals else None
    err_msg = re.escape(
        f"Residuals for level '_unknown_level' are None or empty. Check "
        f"`forecaster.{literal}`."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_residuals_input(forecaster_name='ForecasterRecursiveMultiSeries', **kwargs)
