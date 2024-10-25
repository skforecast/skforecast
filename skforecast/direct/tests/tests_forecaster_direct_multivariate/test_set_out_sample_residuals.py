# Unit test set_out_sample_residuals ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.direct import ForecasterDirectMultiVariate
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

# Fixtures
series = pd.DataFrame({'l1': pd.Series(np.arange(15)), 
                       'l2': pd.Series(np.arange(15))})


@pytest.mark.parametrize("residuals", [[1, 2, 3], {1: [1, 2, 3, 4]}], 
                         ids=lambda residuals: f'residuals: {residuals}')
def test_set_out_sample_residuals_TypeError_when_residuals_is_not_a_dict_of_numpy_ndarray(residuals):
    """
    Test TypeError is raised when residuals is not a dict of numpy ndarrays.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2)
    err_msg = re.escape(
        (f"`residuals` argument must be a dict of numpy ndarrays in the form "
         "`{step: residuals}`. " 
         f"Got {type(residuals)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2)
    residuals = {1: np.array([1, 2, 3, 4, 5]), 
                 2: np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        ("This forecaster is not fitted yet. Call `fit` with appropriate "
         "arguments before using `set_out_sample_residuals()`.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_UserWarning_when_residuals_not_for_all_steps():
    """
    Test UserWarning is raised when residuals does not contain a residue for all steps.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2)
    forecaster.fit(series=series)
    residuals = {1: np.array([1, 2, 3])}

    err_msg = re.escape(
        (f"Only residuals of models (steps) "
         f"{set({1: None, 2: None}.keys()).intersection(set(residuals.keys()))} "
         f"are updated.")
    )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_UserWarning_when_forecaster_has_transformer_and_transform_False():
    """
    Test UserWarning is raised when forcaster has a transformer_series and transform=False.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2, transformer_series=StandardScaler())
    forecaster.fit(series=series)
    residuals = {1: np.array([1, 2, 3]), 2: np.array([1, 2, 3])}

    err_msg = re.escape(
        (f"Argument `transform` is set to `False` but forecaster was trained "
         f"using a transformer {forecaster.transformer_series_[forecaster.level]}. Ensure "
         f"that the new residuals are already transformed or set `transform=True`.")
    )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=False)


def test_set_out_sample_residuals_UserWarning_when_forecaster_has_transformer_and_transform_True():
    """
    Test UserWarning is raised when forcaster has a transformer_series and transform=True.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2, transformer_series=StandardScaler())
    forecaster.fit(series=series)
    residuals = {1: np.array([1, 2, 3]), 2: np.array([1, 2, 3])}

    err_msg = re.escape(
                (f"Residuals will be transformed using the same transformer used when "
                 f"training the forecaster ({forecaster.transformer_series_[forecaster.level]}). Ensure "
                 f"the new residuals are on the same scale as the original time series.")
            )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=True)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is False.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2, transformer_series=None)
    forecaster.fit(series=series)
    residuals = {1: np.arange(10), 2: np.arange(10)}
    new_residuals = {1: np.arange(20), 2: np.arange(20)}

    forecaster.set_out_sample_residuals(residuals=residuals)
    forecaster.set_out_sample_residuals(residuals=new_residuals, append=False)
    
    expected = {1: np.arange(20), 2: np.arange(20)}
    results = forecaster.out_sample_residuals_

    assert expected.keys() == results.keys()
    assert all(all(expected[k] == results[k]) for k in expected.keys())


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_append():
    """
    Test residuals stored when new residuals length is less than 1000 and append is True.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2, transformer_series=None)
    forecaster.fit(series=series)
    residuals = {1: np.arange(10), 2: np.arange(10)}

    forecaster.set_out_sample_residuals(residuals=residuals)
    forecaster.set_out_sample_residuals(residuals=residuals, append=True)

    expected = {1: np.append(np.arange(10), np.arange(10)),
                2: np.append(np.arange(10), np.arange(10))}
    results = forecaster.out_sample_residuals_

    assert expected.keys() == results.keys()
    assert all(all(expected[k] == results[k]) for k in expected.keys())


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000():
    """
    Test len residuals stored when its length is greater than 1000.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2, transformer_series=None)
    forecaster.fit(series=series)
    residuals = {1: np.arange(2000), 2: np.arange(2000)}

    forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals_

    assert list(results.keys()) == [1, 2]
    assert all(len(value) == 1000 for value in results.values())


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000_and_append():
    """
    Test residuals stored when new residuals length is greater than 1000 and append is True.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2, transformer_series=None)
    forecaster.fit(series=series)
    residuals = {1: np.arange(10), 2: np.arange(10)}
    residuals_2 = {1: np.arange(1200), 2: np.arange(1200)}

    forecaster.set_out_sample_residuals(residuals = residuals)
    forecaster.set_out_sample_residuals(residuals = residuals_2,
                                        append    = True)
    
    expected = {}
    for key, value in residuals_2.items():
        rng = np.random.default_rng(seed=123)
        expected_2 = rng.choice(a=value, size=1000, replace=False)
        expected[key] = np.append(np.arange(10), expected_2)[:1000]

    results = forecaster.out_sample_residuals_

    assert expected.keys() == results.keys()
    assert all(all(expected[k] == results[k]) for k in expected.keys())
    assert all(len(value) == 1000 for value in results.values())


def test_set_out_sample_residuals_when_residuals_keys_do_not_match():
    """
    Test residuals are not stored when keys does not match.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2)
    forecaster.fit(series=series)
    residuals = {3: np.arange(10), 4: np.arange(10)}

    forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals_

    assert results == {1: None, 2: None}


def test_set_out_sample_residuals_when_residuals_keys_partially_match():
    """
    Test residuals are stored only for matching keys.
    """
    forecaster = ForecasterDirectMultiVariate(LinearRegression(fit_intercept=True), level='l1',
                                               lags=3, steps=2, transformer_series=None)
    forecaster.fit(series=series)
    residuals = {1: np.arange(10), 4: np.arange(10)}

    forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals_

    assert list(results.keys()) == [1, 2]
    assert all(results[1] == np.arange(10))
    assert results[2] is None
