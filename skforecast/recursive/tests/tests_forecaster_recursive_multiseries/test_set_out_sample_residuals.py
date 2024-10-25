# Unit test set_out_sample_residuals ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.exceptions import UnknownLevelWarning
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                       'l2': pd.Series(np.arange(10))})


@pytest.mark.parametrize("residuals", [[1, 2, 3], {'1': [1, 2, 3, 4]}], 
                         ids=lambda residuals: f'residuals: {residuals}')
def test_set_out_sample_residuals_TypeError_when_residuals_is_not_a_dict_of_numpy_ndarray(residuals):
    """
    Test TypeError is raised when residuals is not a dict of numpy ndarrays.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    err_msg = re.escape(
       (f"`residuals` argument must be a dict of numpy ndarrays in the form "
        "`{level: residuals}`. " 
        f"Got {type(residuals)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    residuals = {'l1': np.array([1, 2, 3, 4, 5]), 
                 'l2': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        ("This forecaster is not fitted yet. Call `fit` with appropriate "
         "arguments before using `set_out_sample_residuals()`.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)


def test_set_out_sample_residuals_UnknownLevelWarning_when_residuals_levels_but_encoding_None():
    """
    Test UnknownLevelWarning is raised when residuals contains levels 
    but encoding is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(fit_intercept=True), 
                                              lags=3, encoding=None)
    forecaster.fit(series=series)
    residuals = {'l1': np.array([1, 2, 3])}

    err_msg = re.escape(
        ("As `encoding` is set to `None`, no distinction between levels "
         "is made. All residuals are stored in the '_unknown_level' key.")
    )
    with pytest.warns(UnknownLevelWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals_

    expected = {
        '_unknown_level': np.array([1, 2, 3])
    }

    assert expected.keys() == results.keys()
    for k in results.keys():
        np.testing.assert_array_almost_equal(expected[k], results[k])


def test_set_out_sample_residuals_IgnoredArgumentWarning_when_residuals_not_for_all_levels():
    """
    Test IgnoredArgumentWarning is raised when residuals does not contain a residue for all levels.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(fit_intercept=True), lags=3)
    forecaster.fit(series=series)
    residuals = {'l1': np.array([1, 2, 3])}

    err_msg = re.escape(
        (
            f"Only residuals of levels "
            f"{set(forecaster.series_names_in_).intersection(set(residuals.keys()))} "
            f"are updated."
        )
    )
    with pytest.warns(IgnoredArgumentWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals_

    expected = {
        'l1': np.array([1, 2, 3]), 
        'l2': None, 
        '_unknown_level': np.array([1, 2, 3])
    }

    assert expected.keys() == results.keys()
    for k in results.keys():
        if results[k] is not None:
            np.testing.assert_array_almost_equal(expected[k], results[k])
        else:
            assert results[k] is None


@pytest.mark.parametrize("encoding, expected_keys", 
                         [('ordinal', 'all'), 
                          ('onehot', 'all'),
                          ('ordinal_category', 'all'),
                          (None, '_unknown_level')],
                         ids=lambda encoding: f'encoding: {encoding}')
def test_set_out_sample_residuals_warning_when_forecaster_has_transformer_and_transform_False(encoding, expected_keys):
    """
    Test Warning is raised when forcaster has a transformer_series and transform=False.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=encoding,
                                              transformer_series=StandardScaler())
    forecaster.fit(series=series)
    residuals = {'l1': np.array([1, 2, 3]), 'l2': np.array([4, 5, 6])}

    level = 'l1' if encoding is not None else '_unknown_level'
    level_str = f"level '{level}'" if encoding is not None else 'all levels'

    err_msg = re.escape(
        (f"Argument `transform` is set to `False` but forecaster was "
         f"trained using a transformer {forecaster.transformer_series_[level]} "
         f"for {level_str}. Ensure that the new residuals are "
         f"already transformed or set `transform=True`.")
    )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=False)
    results = forecaster.out_sample_residuals_

    expected = {
        'l1': np.array([1, 2, 3]), 
        'l2': np.array([4, 5, 6]),
        '_unknown_level': np.array([1, 2, 3, 4, 5, 6])
    }
    if expected_keys != 'all':
        expected = {k: v for k, v in expected.items() 
                    if k == '_unknown_level'}

    assert expected.keys() == results.keys()
    for k in results.keys():
        np.testing.assert_array_almost_equal(expected[k], results[k])


@pytest.mark.parametrize("encoding, expected_keys", 
                         [('ordinal', 'all'), 
                          ('onehot', 'all'),
                          ('ordinal_category', 'all'),
                          (None, '_unknown_level')],
                         ids=lambda encoding: f'encoding: {encoding}')
def test_set_out_sample_residuals_warning_when_forecaster_has_transformer_and_transform_True(encoding, expected_keys):
    """
    Test Warning is raised when forcaster has a transformer_y and transform=True.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=encoding,
                                              transformer_series=StandardScaler())
    forecaster.fit(series=series)
    residuals = {'l1': np.array([1, 2, 3]), 'l2': np.array([4, 5, 6])}
    
    level = 'l1' if encoding is not None else '_unknown_level'
    level_str = f"level '{level}'" if encoding is not None else 'all levels'

    err_msg = re.escape(
        (f"Residuals will be transformed using the same transformer used "
         f"when training the forecaster for {level_str} : "
         f"({forecaster.transformer_series_[level]}). Ensure that the new "
         f"residuals are on the same scale as the original time series.")
    )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(residuals=residuals, transform=True)
    results = forecaster.out_sample_residuals_

    expected = {
        'l1': np.array([-1.21854359, -0.87038828, -0.52223297]), 
        'l2': np.array([-0.17407766,  0.17407766,  0.52223297]),
        '_unknown_level': np.array([-1.21854359, -0.87038828, -0.52223297, 
                                    -0.17407766,  0.17407766, 0.52223297])
    }
    if expected_keys != 'all':
        expected = {k: v for k, v in expected.items() 
                    if k == '_unknown_level'}

    assert expected.keys() == results.keys()
    for k in results.keys():
        np.testing.assert_array_almost_equal(expected[k], results[k])


@pytest.mark.parametrize("encoding", ['ordinal', 'onehot', 'ordinal_category'], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_no_append(encoding):
    """
    Test residuals stored when new residuals length is less than 1000 and append is False.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=encoding,
                                              transformer_series=None)
    forecaster.fit(series=series)
    residuals = {'l1': np.arange(10), 'l2': np.arange(10)}
    new_residuals = {'l1': np.arange(20), 'l2': np.arange(20)}

    forecaster.set_out_sample_residuals(residuals=residuals)
    forecaster.set_out_sample_residuals(residuals=new_residuals, append=False)
    results = forecaster.out_sample_residuals_

    expected = {
        'l1': np.arange(20), 
        'l2': np.arange(20), 
        '_unknown_level': np.concatenate((np.arange(20), np.arange(20)))
    }

    assert expected.keys() == results.keys()
    for k in results.keys():
        np.testing.assert_array_almost_equal(expected[k], results[k])


def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_encoding_None():
    """
    Test residuals stored when new residuals length is less than 1000 and 
    encoding is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=None,
                                              transformer_series=None)
    forecaster.fit(series=series)
    residuals = {'_unknown_level': np.arange(20)}
    new_residuals = {'_unknown_level': np.arange(20, 40)}

    forecaster.set_out_sample_residuals(residuals=residuals)
    forecaster.set_out_sample_residuals(residuals=new_residuals, append=False)
    results = forecaster.out_sample_residuals_

    expected = {'_unknown_level': np.arange(20, 40)}

    assert expected.keys() == results.keys()
    for k in results.keys():
        np.testing.assert_array_almost_equal(expected[k], results[k])


@pytest.mark.parametrize("encoding", ['ordinal', 'onehot', 'ordinal_category'], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_set_out_sample_residuals_when_residuals_length_is_less_than_1000_and_append(encoding):
    """
    Test residuals stored when new residuals length is less than 1000 and append is True.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=encoding,
                                              transformer_series=None)
    forecaster.fit(series=series)
    residuals = {'l1': np.arange(10), 'l2': np.arange(10)}
    
    forecaster.set_out_sample_residuals(residuals=residuals)
    forecaster.set_out_sample_residuals(residuals=residuals, append=True)
    results = forecaster.out_sample_residuals_

    expected = {
        'l1': np.concatenate((np.arange(10), np.arange(10))), 
        'l2': np.concatenate((np.arange(10), np.arange(10))), 
        '_unknown_level': np.concatenate(
            (np.arange(10), np.arange(10), np.arange(10), np.arange(10))
        )
    }

    assert expected.keys() == results.keys()
    for k in results.keys():
        np.testing.assert_array_almost_equal(expected[k], results[k])


@pytest.mark.parametrize("encoding", ['ordinal', 'onehot', 'ordinal_category'], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000(encoding):
    """
    Test len residuals stored when its length is greater than 1000.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=encoding,
                                              transformer_series=None)
    forecaster.fit(series=series)

    residuals = {'l1': np.arange(2000), 'l2': np.arange(2000)}
    forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals_

    assert list(results.keys()) == ['l1', 'l2', '_unknown_level']
    assert all(len(value) == 1000 for value in results.values())


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000_encoding_None():
    """
    Test len residuals stored when its length is greater than 1000
    and encoding is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=None,
                                              transformer_series=None)
    forecaster.fit(series=series)

    residuals = {'_unknown_level': np.arange(2000)}
    forecaster.set_out_sample_residuals(residuals=residuals)
    results = forecaster.out_sample_residuals_

    assert list(results.keys()) == ['_unknown_level']
    assert all(len(value) == 1000 for value in results.values())


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000_and_append():
    """
    Test residuals stored when new residuals length is greater than 1000 and append is True.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series)
    residuals = {'l1': np.arange(10), 'l2': np.arange(10)}
    residuals_2 = {'l1': np.arange(1200), 'l2': np.arange(1200)}

    forecaster.set_out_sample_residuals(residuals = residuals)
    forecaster.set_out_sample_residuals(residuals = residuals_2,
                                        append    = True)
    results = forecaster.out_sample_residuals_

    expected = {}
    for key, value in residuals_2.items():
        rng = np.random.default_rng(seed=123)
        expected_2 = rng.choice(a=value, size=1000, replace=False)
        expected[key] = np.hstack([np.arange(10), expected_2])[:1000]
    
    residuals_unknown_level = [
        v for k, v in expected.items() 
        if v is not None and k != '_unknown_level'
    ]
    if residuals_unknown_level:
        residuals_unknown_level = np.concatenate(residuals_unknown_level)
        if len(residuals_unknown_level) > 1000:
            rng = np.random.default_rng(seed=123)
            expected['_unknown_level'] = rng.choice(
                                             a       = residuals_unknown_level,
                                             size    = 1000,
                                             replace = False
                                         )

    assert expected.keys() == results.keys()
    for k in results.keys():
        assert len(results[k]) == 1000 
        np.testing.assert_array_almost_equal(expected[k], results[k])
