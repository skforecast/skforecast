# Unit test __init__ ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate


def test_init_TypeError_when_level_is_not_a_str():
    """
    Test TypeError is raised when level is not a str.
    """
    level = 5
    err_msg = re.escape(f"`level` argument must be a str. Got {type(level)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterDirectMultiVariate(LinearRegression(), level=level, lags=2, steps=3)


def test_init_TypeError_when_steps_is_not_int():
    """
    Test TypeError is raised when steps is not an int.
    """
    steps = 'not_valid_type'
    err_msg = re.escape(
        f"`steps` argument must be an int greater than or equal to 1. "
        f"Got {type(steps)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        ForecasterDirectMultiVariate(LinearRegression(), level='l1', lags=2, steps=steps)


def test_init_ValueError_when_steps_is_less_than_1():
    """
    Test ValueError is raised when steps is less than 1.
    """
    steps = 0
    err_msg = re.escape(f"`steps` argument must be greater than or equal to 1. Got {steps}.")
    with pytest.raises(ValueError, match = err_msg):
        ForecasterDirectMultiVariate(LinearRegression(), level='l1', lags=2, steps=steps)


@pytest.mark.parametrize("lags", 
                         [None, {'l1': None, 'l2': None}], 
                         ids = lambda lags: f'lags: {lags}')
def test_init_ValueError_when_no_lags_or_window_features(lags):
    """
    Test ValueError is raised when no lags or window_features are passed.
    """
    err_msg = re.escape(
        ("At least one of the arguments `lags` or `window_features` "
         "must be different from None. This is required to create the "
         "predictors used in training the forecaster.")
    )
    with pytest.raises(ValueError, match = err_msg):
        ForecasterDirectMultiVariate(
            regressor       = LinearRegression(),
            level           = 'l1',
            steps           = 3,
            lags            = lags,
            window_features = None
        )


@pytest.mark.parametrize("lags, window_features, expected", 
                         [(5, None, 5), 
                          (None, True, 6), 
                          (5, True, 6)], 
                         ids = lambda dt: f'lags, window_features, expected: {dt}')
def test_init_window_size_correctly_stored(lags, window_features, expected):
    """
    Test window_size is correctly stored when lags or window_features are passed.
    """
    if window_features:
        window_features = RollingFeatures(
            stats=['ratio_min_max', 'median'], window_sizes=[5, 6]
        )

    forecaster = ForecasterDirectMultiVariate(
                     regressor       = LinearRegression(),
                     level           = 'l1',
                     steps           = 3,
                     lags            = lags,
                     window_features = window_features
                 )
    
    assert forecaster.window_size == expected
    if lags:
        np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3, 4, 5]))
        assert forecaster.lags_names == [f'lag_{i}' for i in range(1, lags + 1)]
        assert forecaster.max_lag == lags
    else:
        assert forecaster.lags is None
        assert forecaster.lags_names is None
        assert forecaster.max_lag is None
    if window_features:
        assert forecaster.window_features_names == ['roll_ratio_min_max_5', 'roll_median_6']
        assert forecaster.window_features_class_names == ['RollingFeatures']
    else:
        assert forecaster.window_features_names is None
        assert forecaster.window_features_class_names is None


@pytest.mark.parametrize("lags, expected",
    [({'l1': 3, 'l2': 5}, {'l1': np.array([1, 2, 3]), 'l2': np.array([1, 2, 3, 4, 5])}), 
    ({'l1': None, 'l2': 5}, {'l1': None, 'l2': np.array([1, 2, 3, 4, 5])})],
    ids = lambda value: f'lags: {value}')
def test_init_max_lag_stored_when_dict(lags, expected):
    """
    Test lags are correctly stored when passed as a dictionary.
    """

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', lags=lags, steps=2
    )
    
    for k in forecaster.lags:
        if lags[k] is None:
            assert forecaster.lags[k] is None
            assert forecaster.lags_names[k] is None
        else:
            np.testing.assert_array_almost_equal(forecaster.lags[k], expected[k])
            assert forecaster.lags_names[k] == [f'{k}_lag_{i}' for i in range(1, lags[k] + 1)]
    assert forecaster.max_lag == 5
    assert forecaster.window_size == 5


def test_init_when_lags_dict_with_all_None():
    """
    Test lags dict are correctly stored when all values are None.
    """
    window_features = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=[5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
                     regressor       = LinearRegression(),
                     level           = 'l1',
                     steps           = 3,
                     lags            = {'l1': None, 'l2': None},
                     window_features = window_features
                 )
    
    assert forecaster.window_size == 6
    assert forecaster.lags == {'l1': None, 'l2': None}
    assert forecaster.lags_names == {'l1': None, 'l2': None}
    assert forecaster.max_lag is None
    assert forecaster.window_features_names == ['roll_ratio_min_max_5', 'roll_median_6']
    assert forecaster.window_features_class_names == ['RollingFeatures']


@pytest.mark.parametrize("dif", 
                         [0, 0.5, 1.5, 'not_int'], 
                         ids = lambda dif: f'differentiation: {dif}')
def test_init_ValueError_when_differentiation_argument_is_not_int_or_greater_than_0(dif):
    """
    Test ValueError is raised when differentiation is not an int or greater than 0.
    """
    err_msg = re.escape(
        (f"Argument `differentiation` must be an integer equal to or "
         f"greater than 1. Got {dif}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        ForecasterDirectMultiVariate(
            regressor       = LinearRegression(),
            level           = 'l1',
            steps           = 3,
            lags            = 5,
            differentiation = dif
        )


@pytest.mark.parametrize("dif", 
                         [1, 2], 
                         ids = lambda dif: f'differentiation: {dif}')
def test_init_window_size_is_increased_when_differentiation(dif):
    """
    Test window_size is increased when including differentiation.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor       = LinearRegression(),
                     level           = 'l1',
                     steps           = 3,
                     lags            = 5,
                     differentiation = dif
                 )
    
    assert forecaster.window_size == 5 + dif


@pytest.mark.parametrize("n_jobs", 
                         [1.0, 'not_int_auto'], 
                         ids = lambda value: f'n_jobs: {value}')
def test_init_TypeError_when_n_jobs_not_int_or_auto(n_jobs):
    """
    Test TypeError is raised in when n_jobs is not an integer or 'auto'.
    """
    err_msg = re.escape(f"`n_jobs` must be an integer or `'auto'`. Got {type(n_jobs)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterDirectMultiVariate(
            LinearRegression(), level='l1', steps=2, lags=2, n_jobs=n_jobs
        )
