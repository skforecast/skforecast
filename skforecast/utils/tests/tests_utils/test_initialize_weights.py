# Unit test initialize_weights
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
import inspect
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.utils.utils import initialize_weights
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


@pytest.mark.parametrize("forecaster_name", 
                         ['ForecasterRecursive', 'ForecasterDirect', 'ForecasterDirectMultiVariate'], 
                         ids=lambda fn: f'forecaster_name: {fn}')
def test_TypeError_initialize_weights_when_weight_func_is_not_a_Callable(forecaster_name):
    """
    Test TypeError is raised when weight_func is not a Callable.
    """
    weight_func = 'not_Callable'
    err_msg = re.escape(
        f"Argument `weight_func` must be a Callable. Got {type(weight_func)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_weights(
            forecaster_name = forecaster_name, 
            regressor       = LinearRegression(), 
            weight_func     = weight_func, 
            series_weights  = None
        )


def test_TypeError_initialize_weights_when_weight_func_is_not_a_Callable_or_dict():
    """
    Test TypeError is raised when weight_func is not a Callable or a dict.
    """
    weight_func = 'not_Callable_or_dict'
    err_msg = re.escape(
        f"Argument `weight_func` must be a Callable or a dict of Callables. Got {type(weight_func)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_weights(
            forecaster_name = 'ForecasterRecursiveMultiSeries', 
            regressor       = LinearRegression(), 
            weight_func     = weight_func, 
            series_weights  = None
        )


def test_TypeError_initialize_weights_when_series_weights_is_not_a_dict():
    """
    Test TypeError is raised when series_weights is not a dict.
    """
    series_weights = 'not_Callable_or_dict'
    err_msg = re.escape(
        (f"Argument `series_weights` must be a dict of floats or ints."
         f"Got {type(series_weights)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        initialize_weights(
            forecaster_name = 'ForecasterRecursiveMultiSeries', 
            regressor       = LinearRegression(), 
            weight_func     = None, 
            series_weights  = series_weights
        )


def test_IgnoredArgumentWarning_initialize_weights_when_weight_func_is_provided_and_regressor_has_not_sample_weights():
    """
    Test IgnoredArgumentWarning is created when weight_func is provided but the regressor 
    has no argument sample_weights in his fit method.
    """
    def weight_func():  # pragma: no cover
        pass

    warn_msg = re.escape(
        ("Argument `weight_func` is ignored since regressor KNeighborsRegressor() "
         "does not accept `sample_weight` in its `fit` method.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        weight_func, source_code_weight_func, _ = initialize_weights(
            forecaster_name = 'ForecasterRecursive', 
            regressor       = KNeighborsRegressor(), 
            weight_func     = weight_func, 
            series_weights  = None
        )
    
    assert weight_func is None
    assert source_code_weight_func is None


def test_IgnoredArgumentWarning_initialize_weights_when_series_weights_is_provided_and_regressor_has_not_sample_weights():
    """
    Test IgnoredArgumentWarning is created when series_weights is provided but the regressor 
    has no argument sample_weights in his fit method.
    """
    series_weights = {'series_1': 1., 'series_2': 2.}

    warn_msg = re.escape(
                ("Argument `series_weights` is ignored since regressor KNeighborsRegressor() "
                 "does not accept `sample_weight` in its `fit` method.")
            )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        weight_func, source_code_weight_func, series_weights = initialize_weights(
            forecaster_name = 'ForecasterRecursiveMultiSeries', 
            regressor       = KNeighborsRegressor(), 
            weight_func     = None, 
            series_weights  = series_weights
        )
    
    assert series_weights is None


@pytest.mark.parametrize("regressor", 
                         [LinearRegression(), RandomForestRegressor()], 
                         ids=lambda regressor: f'regressor: {type(regressor).__name__}')
def test_initialize_weights_finds_sample_weight_in_different_regressors_when_weight_func(recwarn, regressor):
    """
    Test initialize weights finds `sample_weight` attribute in different
    regressors when `weight_func`.
    """
    def weight_func():  # pragma: no cover
        pass

    weight_func, source_code_weight_func, _ = initialize_weights(
        forecaster_name = 'ForecasterRecursive', 
        regressor       = regressor, 
        weight_func     = weight_func, 
        series_weights  = None
    )
    
    # Count the number of warnings, it should be 0
    assert len(recwarn) == 0


@pytest.mark.parametrize("regressor", 
                         [LinearRegression(), RandomForestRegressor()], 
                         ids=lambda regressor: f'regressor: {type(regressor).__name__}')
def test_initialize_weights_finds_sample_weight_in_different_regressors_when_series_weights(recwarn, regressor):
    """
    Test initialize weights finds `sample_weight` attribute in different
    regressors when `series_weights`.
    """
    series_weights = {'series_1': 1., 'series_2': 2.}

    weight_func, source_code_weight_func, series_weights = initialize_weights(
        forecaster_name = 'ForecasterRecursiveMultiSeries', 
        regressor       = regressor, 
        weight_func     = None, 
        series_weights  = series_weights
    )
    
    # Count the number of warnings, it should be 0
    assert len(recwarn) == 0


def test_output_initialize_weights_source_code_weight_func_when_weight_func_not_dict():
    """
    Test source_code_weight_func output of initialize_weights when 
    weight_func is a Callable.
    """
    def test_weight_func():  # pragma: no cover
        """
        test
        """
        pass

    weight_func, source_code_weight_func, series_weights = initialize_weights(
        forecaster_name = 'ForecasterRecursiveMultiSeries', 
        regressor       = LinearRegression(), 
        weight_func     = test_weight_func, 
        series_weights  = None
    )
    
    assert source_code_weight_func == inspect.getsource(test_weight_func)


def test_output_initialize_weights_source_code_weight_func_when_weight_func_dict():
    """
    Test source_code_weight_func output of initialize_weights when 
    weight_func is a dict.
    """
    def test_weight_func():  # pragma: no cover
        """
        test
        """
        pass

    def test_weight_func_2():  # pragma: no cover
        """
        test2
        """
        pass

    weight_func = {'series_1': test_weight_func, 'series_2': test_weight_func_2}    

    weight_func, source_code_weight_func, series_weights = initialize_weights(
        forecaster_name = 'ForecasterRecursiveMultiSeries', 
        regressor       = LinearRegression(), 
        weight_func     = weight_func, 
        series_weights  = None
    )
    
    assert source_code_weight_func['series_1'] == inspect.getsource(test_weight_func)
    assert source_code_weight_func['series_2'] == inspect.getsource(test_weight_func_2)
