# Unit test set_out_sample_residuals ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from skforecast.exceptions import UnknownLevelWarning
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                       'l2': pd.Series(np.arange(10))})


def test_set_out_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    y_true = {'1': np.array([1, 2, 3, 4, 5]), '2': np.array([1, 2, 3, 4, 5])}
    y_pred = {'1': np.array([1, 2, 3, 4, 5]), '2': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        ("This forecaster is not fitted yet. Call `fit` with appropriate "
         "arguments before using `set_out_sample_residuals()`.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_true_is_not_dict():
    """
    Test TypeError is raised when y_true is not a dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    y_true = 'not_dict'
    y_pred = {'1': np.array([1, 2, 3, 4, 5]), '2': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        f"`y_true` must be a dictionary of numpy ndarrays or pandas Series. "
        f"Got {type(y_true)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_pred_is_not_dict():
    """
    Test TypeError is raised when y_pred is not a dict.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    y_true = {'1': np.array([1, 2, 3, 4, 5]), '2': np.array([1, 2, 3, 4, 5])}
    y_pred = 'not_dict'

    err_msg = re.escape(
        f"`y_pred` must be a dictionary of numpy ndarrays or pandas Series. "
        f"Got {type(y_pred)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_pred_and_y_true_keys_do_not_match():
    """
    Test TypeError is raised when y_pred and y_true keys do not match.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    y_true = {'1': np.array([1, 2, 3, 4, 5]), '2': np.array([1, 2, 3, 4, 5])}
    y_pred = {'3': np.array([1, 2, 3, 4, 5]), '4': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        f"`y_true` and `y_pred` must have the same keys. "
        f"Got {set(y_true.keys())} and {set(y_pred.keys())}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_true_contains_no_numpy_ndarrays_or_pandas_Series():
    """
    Test TypeError is raised when y_true contains no numpy ndarrays or pandas Series.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    y_true = {'1': 'not_ndarray'}
    y_pred = {'1': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        f"Values of `y_true` must be numpy ndarrays or pandas Series. "
        f"Got {type(y_true['1'])} for series '1'."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_pred_contains_no_numpy_ndarrays_or_pandas_Series():
    """
    Test TypeError is raised when y_pred contains no numpy ndarrays or pandas Series.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    y_true = {'1': np.array([1, 2, 3, 4, 5])}
    y_pred = {'1': 'not_ndarray'}

    err_msg = re.escape(
        f"Values of `y_pred` must be numpy ndarrays or pandas Series. "
        f"Got {type(y_pred['1'])} for series '1'."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_elements_with_different_lengths():
    """
    Test ValueError is raised when y_true and y_pred have elements with different lengths.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    y_true = {'1': np.array([1, 2, 3, 4, 5]), '2': np.array([1, 2, 3, 4, 5])}
    y_pred = {'1': np.array([1, 2, 3, 4, 5]), '2': np.array([1, 2])}

    err_msg = re.escape(
        f"`y_true` and `y_pred` must have the same length. "
        f"Got {len(y_true['2'])} and {len(y_pred['2'])} for series '2'."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_series_with_different_indexes():
    """
    Test ValueError is raised when y_true and y_pred have series with different indexes.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.is_fitted = True
    y_true = {'1': pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])}
    y_pred = {'1': pd.Series([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        "When containing pandas Series, elements in `y_true` and "
        "`y_pred` must have the same index. Error with series '1'."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_UserWarning_when_inputs_does_not_match_series_seen_in_fit():
    """
    Test UserWarning is raised when inputs does not contain keys that match any 
    series seen in fit.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    y_true = {'5': np.array([1, 2, 3])}
    y_pred = {'5': np.array([1, 2, 3])}

    err_msg = re.escape(
        "Provided keys in `y_pred` and `y_true` do not match any series seen "
        "in `fit`. Residuals are not updated."
    )
    with pytest.warns(UserWarning, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_UnknownLevelWarning_when_residuals_levels_but_encoding_None():
    """
    Test UnknownLevelWarning is raised when residuals contains levels but encoding is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3, encoding=None)
    forecaster.fit(series=series)
    y_true = {'l1': np.array([1, 2, 3, 4, 5])}
    y_pred = {'l1': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        ("As `encoding` is set to `None`, no distinction between levels "
         "is made. All residuals are stored in the '_unknown_level' key.")
    )
    with pytest.warns(UnknownLevelWarning, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    results = forecaster.out_sample_residuals_

    expected = {
        '_unknown_level': np.array([0, 0, 0, 0, 0])
    }

    assert expected.keys() == results.keys()
    for k in results.keys():
        np.testing.assert_array_almost_equal(expected[k], results[k])


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'onehot', 'ordinal_category'], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_no_append(encoding):
    """
    Test residuals stored when new residuals length is less than 10000 and append
    is False.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    LinearRegression(),
                    lags=3,
                    encoding=encoding,
                 )
    forecaster.fit(series=series)
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([2, 3, 4, 5, 6])}
    y_pred = {'l1': np.array([0, 1, 2, 3, 4]), 'l2': np.array([0, 1, 2, 3, 4])}

    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    results = forecaster.out_sample_residuals_
    expected = {
        'l1': np.array([1, 1, 1, 1, 1]),
        'l2': np.array([2, 2, 2, 2, 2]),
        '_unknown_level': np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1])
    }

    assert expected.keys() == results.keys()
    assert all(all(np.sort(expected[k]) == np.sort(results[k])) for k in expected.keys())


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'onehot', 'ordinal_category', None], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_set_out_sample_residuals_for_unknown_level(encoding):
    """
    Test residuals stored for unknown level.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3, encoding=encoding)
    forecaster.fit(series=series)
    y_true = {'_unknown_level': np.array([1, 2, 3, 5, 6])}
    y_pred = {'_unknown_level': np.array([0, 1, 2, 3, 4])}

    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    results = forecaster.out_sample_residuals_

    if encoding is None:
        expected = {
            '_unknown_level': np.array([1, 1, 1, 2, 2])
        }
    else:
        expected = {
            'l1': None,
            'l2': None,
            '_unknown_level': np.array([1, 1, 1, 2, 2])
        }

    assert expected.keys() == results.keys()
    for k in results.keys():
        if results[k] is None:
            assert results[k] == expected[k]
        else:
            np.testing.assert_array_almost_equal(expected[k], results[k])


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_encoding_None():
    """
    Test residuals stored when new residuals length is less than 10000 and 
    encoding is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    LinearRegression(),
                    lags=3,
                    encoding=None,
                 )
    forecaster.fit(series=series)
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([2, 3, 4, 5, 6])}
    y_pred = {'l1': np.array([0, 1, 2, 3, 4]), 'l2': np.array([0, 1, 2, 3, 4])}

    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    results = forecaster.out_sample_residuals_
    expected = {
        '_unknown_level': np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1])
    }

    assert expected.keys() == results.keys()
    assert all(all(np.sort(expected[k]) == np.sort(results[k])) for k in expected.keys())


@pytest.mark.parametrize("encoding", ['ordinal', 'onehot', 'ordinal_category'], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_append(encoding):
    """
    Test residuals stored when new residuals length is less than 10000 and append is True.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    LinearRegression(),
                    lags=3,
                    encoding=encoding,
                 )
    forecaster.fit(series=series)
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([2, 3, 4, 5, 6])}
    y_pred = {'l1': np.array([0, 1, 2, 3, 4]), 'l2': np.array([0, 1, 2, 3, 4])}
    
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
    results = forecaster.out_sample_residuals_

    expected = {
        'l1': np.array([1, 1, 1, 1, 1] * 2), 
        'l2': np.array([2, 2, 2, 2, 2] * 2),
        '_unknown_level': np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1] * 2)
    }

    assert expected.keys() == results.keys()
    for k in results.keys():
        np.testing.assert_array_almost_equal(np.sort(expected[k]), np.sort(results[k]))


@pytest.mark.parametrize("encoding", ['ordinal', 'onehot', 'ordinal_category'], 
                         ids=lambda encoding: f'encoding: {encoding}')
def test_set_out_sample_residuals_when_residuals_length_is_greater_than_10000(encoding):
    """
    Test len residuals stored when its length is greater than 10000.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    LinearRegression(),
                    lags=3,
                    encoding=encoding,
                 )
    forecaster.fit(series=series)
    y_true = {'l1': np.arange(20_000), 'l2': np.arange(20_000)}
    y_pred = {'l1': np.arange(20_000) + 1, 'l2': np.arange(20_000) + 2}
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    results = forecaster.out_sample_residuals_

    assert list(results.keys()) == ['l1', 'l2', '_unknown_level']
    assert all(len(value) == 10_000 for value in results.values())


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_1000_encoding_None():
    """
    Test len residuals stored when its length is greater than 1000
    and encoding is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    LinearRegression(),
                    lags=3,
                    encoding=None,
                 )
    forecaster.fit(series=series)
    y_true = {'l1': np.arange(20_000), 'l2': np.arange(20_000)}
    y_pred = {'l1': np.arange(20_000) + 1, 'l2': np.arange(20_000) + 2}
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    results = forecaster.out_sample_residuals_

    assert list(results.keys()) == ['_unknown_level']
    assert all(len(value) == 10_000 for value in results.values())


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_10000_and_append():
    """
    Test residuals stored when new residuals length is greater than 10000 and
    append is True.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    LinearRegression(),
                    lags=3,
                 )
    forecaster.fit(series=series)
    y_true = {'l1': np.random.normal(size=5_000), 'l2': np.random.normal(size=5_000)}
    y_pred = {'l1': np.random.normal(size=5_000), 'l2': np.random.normal(size=5_000)}
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    y_true = {'l1': np.random.normal(size=10_000), 'l2': np.random.normal(size=10_000)}
    y_pred = {'l1': np.random.normal(size=10_000), 'l2': np.random.normal(size=10_000)}
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
    results = forecaster.out_sample_residuals_

    assert all([len(v) == 10_000 for v in results.values()])


def test_set_out_sample_residuals_when_residuals_keys_do_not_match():
    """
    Test residuals are not stored when keys does not match.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    LinearRegression(),
                    lags=3,
                 )
    forecaster.fit(series=series)
    y_pred = {'l3': np.arange(10), 'l4': np.arange(10)}
    y_true = {'l3': np.arange(10), 'l4': np.arange(10)}
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    results = forecaster.out_sample_residuals_

    assert results == {'l1': None, 'l2': None, '_unknown_level': None}


def test_set_out_sample_residuals_when_residuals_keys_partially_match():
    """
    Test residuals are stored only for matching keys.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                    LinearRegression(),
                    lags=3,
                 )
    forecaster.fit(series=series)
    y_pred = {'l1': np.repeat(1, 5), 'l4': np.arange(10)}
    y_true = {'l1': np.arange(5), 'l4': np.arange(10)}
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    results = forecaster.out_sample_residuals_
    expected = {
        'l1': np.array([-1,  0,  1,  2,  3]),
        'l2': None,
        '_unknown_level': np.array([-1,  0,  1,  2,  3])
    }
    for key in expected.keys():
        if expected[key] is not None:
            np.testing.assert_array_almost_equal(expected[key], results[key])
        else:
            assert results[key] is None


def test_forecaster_set_outsample_residuals_when_transformer_y_and_diferentiation():
    """
    Test set_out_sample_residuals when forecaster has transformer_y and differentiation.
    Stored should equivalent to residuals calculated manually if transformer_y and
    differentiation are applied to `y_true` and `y_pred` before calculating residuals.
    """
    rng = np.random.default_rng(12345)
    series_train = {
        'l1': pd.Series(
            rng.normal(loc=0, scale=1, size=100),
            index = pd.date_range(start='1-1-2018', periods=100, freq='D')
        ),
        'l2': pd.Series(
            rng.normal(loc=0, scale=1, size=100),
            index = pd.date_range(start='1-1-2018', periods=100, freq='D')
        )
    }
    y_true  = {
        'l1': rng.normal(loc=0, scale=1, size=5),
        'l2': rng.normal(loc=0, scale=1, size=5)
    }
    y_pred = {
        'l1': rng.normal(loc=0, scale=1, size=5),
        'l2': rng.normal(loc=0, scale=1, size=5)
    }
    forecaster = ForecasterRecursiveMultiSeries(
                    regressor       = LinearRegression(),
                    lags            = 5,
                    differentiation = 1,
                    transformer_series = StandardScaler(),
                )
    forecaster.fit(series=series_train)
    forecaster.set_out_sample_residuals(
        y_true = y_true,
        y_pred = y_pred
    )
    print(forecaster.out_sample_residuals_)

    y_true['l1'] = forecaster.transformer_series_['l1'].transform(y_true['l1'].reshape(-1, 1)).flatten()
    y_true['l2'] = forecaster.transformer_series_['l2'].transform(y_true['l2'].reshape(-1, 1)).flatten()
    y_pred['l1'] = forecaster.transformer_series_['l1'].transform(y_pred['l1'].reshape(-1, 1)).flatten()
    y_pred['l2'] = forecaster.transformer_series_['l2'].transform(y_pred['l2'].reshape(-1, 1)).flatten()
    y_true['l1'] = forecaster.differentiator_['l1'].transform(y_true['l1'])[forecaster.differentiation:]
    y_true['l2'] = forecaster.differentiator_['l2'].transform(y_true['l2'])[forecaster.differentiation:]
    y_pred['l1'] = forecaster.differentiator_['l1'].transform(y_pred['l1'])[forecaster.differentiation:]
    y_pred['l2'] = forecaster.differentiator_['l2'].transform(y_pred['l2'])[forecaster.differentiation:]
    residuals = {}
    residuals['l1'] = y_true['l1'] - y_pred['l1']
    residuals['l2'] = y_true['l2'] - y_pred['l2']

    for key in residuals.keys():
        np.testing.assert_array_almost_equal(residuals[key], forecaster.out_sample_residuals_[key])
