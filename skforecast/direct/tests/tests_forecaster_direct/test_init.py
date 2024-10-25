# Unit test __init__ ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirect


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
        ForecasterDirect(LinearRegression(), lags=2, steps=steps)


def test_init_ValueError_when_steps_is_less_than_1():
    """
    Test ValueError is raised when steps is less than 1.
    """
    steps = 0
    err_msg = re.escape(f"`steps` argument must be greater than or equal to 1. Got {steps}.")
    with pytest.raises(ValueError, match = err_msg):
        ForecasterDirect(LinearRegression(), lags=2, steps=steps)


def test_init_ValueError_when_no_lags_or_window_features():
    """
    Test ValueError is raised when no lags or window_features are passed.
    """
    err_msg = re.escape(
        ("At least one of the arguments `lags` or `window_features` "
         "must be different from None. This is required to create the "
         "predictors used in training the forecaster.")
    )
    with pytest.raises(ValueError, match = err_msg):
        ForecasterDirect(
            regressor       = LinearRegression(),
            steps           = 3,
            lags            = None,
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

    forecaster = ForecasterDirect(
                     regressor       = LinearRegression(),
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
        ForecasterDirect(
            regressor       = LinearRegression(),
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
    forecaster = ForecasterDirect(
                     regressor       = LinearRegression(),
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
        ForecasterDirect(LinearRegression(), steps=2, lags=2, n_jobs=n_jobs)
