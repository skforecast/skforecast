# Unit test _initialize_levels_model_selection_multiseries
# ==============================================================================
import re
import pytest
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection._utils import _initialize_levels_model_selection_multiseries
from skforecast.exceptions import IgnoredArgumentWarning

# Fixtures
from ..fixtures_model_selection_multiseries import series_wide_range, series_dict_range


def test_initialize_levels_model_selection_multiseries_TypeError_when_levels_not_list_str_None():
    """
    Test TypeError is raised in _initialize_levels_model_selection_multiseries when 
    `levels` is not a `list`, `str` or `None`.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator = Ridge(random_state=123), lags = 4
    )
    levels = 5

    err_msg = re.escape(
        "`levels` must be a `list` of column names, a `str` of a column "
        "name or `None` when using a forecaster of type "
        "['ForecasterRecursiveMultiSeries', 'ForecasterRnn']. If the forecaster "
        "is of type `ForecasterDirectMultiVariate`, this argument is ignored."
    )
    with pytest.raises(TypeError, match = err_msg):
        _initialize_levels_model_selection_multiseries(
            forecaster = forecaster, 
            series     = series_dict_range,
            levels     = levels
        )


def test_initialize_levels_model_selection_multiseries_IgnoredArgumentWarning_forecaster_multivariate_and_levels():
    """
    Test IgnoredArgumentWarning is raised when levels is not forecaster.level or 
    None in ForecasterDirectMultiVariate.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )
    
    levels = 'not_l1_or_None'
    
    warn_msg = re.escape(
        "`levels` argument have no use when the forecaster is of type "
        "`ForecasterDirectMultiVariate`. The level of this forecaster is "
        "'l1', to predict another level, change the `level` "
        "argument when initializing the forecaster."
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        levels = _initialize_levels_model_selection_multiseries(
                     forecaster = forecaster, 
                     series     = series_dict_range,
                     levels     = levels
                 )
        
    assert levels == ['l1']


@pytest.mark.parametrize("series_as_dict",
                         [True, False],
                         ids=lambda series_as_dict: f'series_as_dict: {series_as_dict}')
def test_initialize_levels_model_selection_multiseries_ValueError_when_levels_not_in_series(series_as_dict):
    """
    Test ValueError is raised in _initialize_levels_model_selection_multiseries when 
    `levels` is not present in `series`.
    """

    forecaster = ForecasterRecursiveMultiSeries(
        estimator = Ridge(random_state=123), lags = 4
    )
    levels = ['l3']

    err_msg = re.escape(
        "Levels ['l3'] not found in `series`, available levels are "
        "['l1', 'l2']. Review `levels` argument."
    )
    with pytest.raises(ValueError, match = err_msg):
        _initialize_levels_model_selection_multiseries(
            forecaster = forecaster, 
            series     = series_dict_range if series_as_dict else series_wide_range,
            levels     = levels
        )


@pytest.mark.parametrize("series_as_dict",
                         [True, False],
                         ids=lambda series_as_dict: f'series_as_dict: {series_as_dict}')
@pytest.mark.parametrize("levels, levels_expected",
                         [(None, ['l1', 'l2']), 
                          ('l1', ['l1']),
                          (['l1', 'l2'], ['l1', 'l2'])])
def test_initialize_levels_model_selection_multiseries_for_all_inputs(series_as_dict, levels, levels_expected):
    """
    Test initialize_levels_model_selection_multiseries when levels is None, 
    str or list.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator = Ridge(random_state=123),
                     lags      = 2
                 )
    
    levels = _initialize_levels_model_selection_multiseries(
                 forecaster = forecaster, 
                 series     = series_dict_range if series_as_dict else series_wide_range,
                 levels     = levels
             )
    
    assert levels == levels_expected
