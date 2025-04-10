# Unit test set_out_sample_residuals ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from skforecast.exceptions import ResidualsUsageWarning
from skforecast.direct import ForecasterDirectMultiVariate

# Fixtures
series = pd.DataFrame({'l1': pd.Series(np.arange(15)), 
                       'l2': pd.Series(np.arange(15))})


def test_set_out_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3
    )
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}
    y_pred = {'l1': np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `set_out_sample_residuals()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_true_is_not_dict():
    """
    Test TypeError is raised when y_true is not a dict.
    """
    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.is_fitted = True
    y_true = 'not_dict'
    y_pred = {'l1': np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}

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
    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}
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
    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}
    y_pred = {3: np.array([1, 2, 3, 4, 5]), 4: np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        f"`y_true` and `y_pred` must have the same keys. "
        f"Got {set(y_true.keys())} and {set(y_pred.keys())}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_true_contains_no_numpy_ndarrays_or_pandas_series():
    """
    Test TypeError is raised when y_true contains no numpy ndarrays or pandas series.
    """
    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': 'not_ndarray'}
    y_pred = {'l1': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        f"Values of `y_true` must be numpy ndarrays or pandas Series. "
        f"Got {type(y_true['l1'])} for series l1."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_pred_contains_no_numpy_ndarrays_or_pandas_series():
    """
    Test TypeError is raised when y_pred contains no numpy ndarrays or pandas series.
    """
    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': np.array([1, 2, 3, 4, 5])}
    y_pred = {'l1': 'not_ndarray'}

    err_msg = re.escape(
        f"Values of `y_pred` must be numpy ndarrays or pandas Series. "
        f"Got {type(y_pred['l1'])} for series l1."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_elements_with_different_lengths():
    """
    Test ValueError is raised when y_true and y_pred have elements with different lengths.
    """
    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([1, 2, 3, 4, 5])}
    y_pred = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([1, 2])}

    err_msg = re.escape(
        '`y_true` and `y_pred` must have the same length. Got 5 and 2 for series l2.'
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_series_with_different_indexes():
    """
    Test ValueError is raised when y_true and y_pred have series with different indexes.
    """
    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])}
    y_pred = {'l1': pd.Series([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        "When containing pandas Series, elements in `y_true` and "
        "`y_pred` must have the same index."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_inputs_does_not_match_the_target_level():
    """
    Test ValueError is raised when inputs does not contain keys that match any step.
    """
    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.fit(series=series)
    y_true = {'l3': np.array([1, 2, 3])}
    y_pred = {'l3': np.array([1, 2, 3])}

    err_msg = re.escape(
        "`y_pred` and `y_true` must have only the key 'l1'. Got {'l3'}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and 
    append is False.
    """
    rng = np.random.default_rng(123)
    y_true = {
        'l1':pd.Series(rng.normal(loc=10, scale=10, size=1000)), 
    }
    y_pred = {
        'l1':pd.Series(rng.normal(loc=10, scale=10, size=1000)), 
    }

    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3, transformer_series=None
    )
    forecaster.fit(series=series)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=False)
    results = {
        'l1':np.sort(forecaster.out_sample_residuals_['l1']),
    }

    expected = {
        'l1':np.sort(y_true['l1'] - y_pred['l1']),
    }

    assert forecaster.out_sample_residuals_.keys() == expected.keys()
    for key in results.keys():
        np.testing.assert_array_almost_equal(expected[key], results[key])


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and 
    append is True.
    """
    rng = np.random.default_rng(123)
    y_true = {
        'l1':pd.Series(rng.normal(loc=10, scale=10, size=1000))
    }
    y_pred = {
        'l1':pd.Series(rng.normal(loc=10, scale=10, size=1000))
    }

    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=2, lags=3, transformer_series=None
    )
    forecaster.fit(series=series)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
    results = {
        'l1':np.sort(forecaster.out_sample_residuals_['l1'])
    }

    residuals_1 = (y_true['l1'] - y_pred['l1'])
    expected = {
        'l1':np.sort(np.concatenate((residuals_1, residuals_1))),
    }

    assert forecaster.out_sample_residuals_.keys() == expected.keys()
    for key in results.keys():
        np.testing.assert_array_almost_equal(expected[key], results[key])


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_10000():
    """
    Test length residuals stored when its length is greater than 10_000.
    """
    rng = np.random.RandomState(42)
    series_fit = pd.DataFrame(
        data = {
            'l1': rng.normal(loc=10, scale=1, size=50_000),
            'l2': rng.normal(loc=10, scale=1, size=50_000)
        }
    )

    forecaster = ForecasterDirectMultiVariate(
        regressor=LinearRegression(), level='l1', steps=1, lags=1, 
        transformer_series=None, binner_kwargs={"n_bins": 10}
    )
    forecaster.fit(series=series_fit)
    X_train, y_train = forecaster.create_train_X_y(series=series_fit)
    X_train_step_1, y_train_step_1 = forecaster.filter_train_X_y_for_step(
        step=1, X_train=X_train, y_train=y_train
    )

    y_true = {'l1': y_train_step_1}
    y_pred = {'l1': forecaster.regressors_[1].predict(X_train_step_1)}
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    assert list(forecaster.out_sample_residuals_.keys()) == ['l1']
    assert len(forecaster.out_sample_residuals_['l1']) == 10_000
    for v in forecaster.out_sample_residuals_by_bin_['l1'].values():
        assert len(v) == 1_000


def test_out_sample_residuals_by_bin_and_in_sample_residuals_by_bin_equivalence():
    """
    Test out sample residuals by bin are equivalent to in-sample residuals by bin
    when training data and training predictions are passed.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 5,
                     transformer_series = None,
                     binner_kwargs      = {'n_bins': 3}
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    X_train, y_train = forecaster.create_train_X_y(series=series)

    y_true = []
    y_pred = []
    for step in range(1, forecaster.steps + 1):
        X_train_step, y_train_step = forecaster.filter_train_X_y_for_step(
            step=step, X_train=X_train, y_train=y_train
        )
        y_true.append(y_train_step.to_numpy())
        y_pred.append(forecaster.regressors_[step].predict(X_train_step))

    y_true = {'l1': np.concatenate(y_true)}
    y_pred = {'l1': np.concatenate(y_pred)}

    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    assert forecaster.in_sample_residuals_by_bin_.keys() == forecaster.out_sample_residuals_by_bin_.keys()
    assert forecaster.in_sample_residuals_by_bin_['l1'].keys() == forecaster.out_sample_residuals_by_bin_['l1'].keys()
    for k in forecaster.out_sample_residuals_by_bin_['l1'].keys():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_by_bin_['l1'][k],
            forecaster.out_sample_residuals_by_bin_['l1'][k]
        )


def test_set_out_sample_residuals_append_new_residuals_per_bin():
    """
    Test that set_out_sample_residuals append residuals per bin until it
    reaches the max allowed size of 10_000 // n_bins
    """
    rng = np.random.default_rng(12345)
    series_fit = pd.DataFrame(
        data = {
            'l1': rng.normal(loc=10, scale=1, size=1001),
            'l2': rng.normal(loc=10, scale=1, size=1001)
        },
        index = pd.date_range(start="01-01-2000", periods=1001, freq="h"),
    )

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 1,
                     lags               = 1,
                     transformer_series = None,
                     binner_kwargs      = {'n_bins': 2}
                 )
    forecaster.fit(series=series_fit)
    X_train, y_train = forecaster.create_train_X_y(series=series_fit)
    X_train_step_1, y_train_step_1 = forecaster.filter_train_X_y_for_step(
        step=1, X_train=X_train, y_train=y_train
    )

    y_true = {'l1': y_train_step_1}
    y_pred = {'l1': forecaster.regressors_[1].predict(X_train_step_1)}
    for i in range(1, 20):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
        for v in forecaster.out_sample_residuals_by_bin_['l1'].values():
            assert len(v) == min(5_000, 500 * i)


def test_set_out_sample_residuals_when_there_are_no_residuals_for_some_bins():
    """
    Test that set_out_sample_residuals works when there are no residuals for some bins.
    """
    rng = np.random.default_rng(12345)
    series_fit = pd.DataFrame(
        data = {
            'l1': rng.normal(loc=10, scale=1, size=100),
            'l2': rng.normal(loc=10, scale=1, size=100)
        },
        index = pd.date_range(start="01-01-2000", periods=100, freq="h"),
    )

    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 1,
                     lags               = 5,
                     transformer_series = None,
                     binner_kwargs      = {'n_bins': 3}
                 )
    forecaster.fit(series=series_fit)
    y = series_fit['l1'].copy()
    y_pred = {'l1': y.loc[y > 10]}
    y_true = {'l1': y_pred['l1'] + rng.normal(loc=0, scale=1, size=len(y_pred['l1']))}

    warn_msg = re.escape(
        f"The following bins of level 'l1' have no out of sample residuals: [0]. "
        f"No predicted values fall in the interval "
        f"[{forecaster.binner_intervals_['l1'][0]}]. "
        f"Empty bins will be filled with a random sample of residuals."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)

    assert len(forecaster.out_sample_residuals_by_bin_['l1'][0]) == len(y_pred['l1'])


def test_forecaster_set_out_sample_residuals_when_transformer_y_and_differentiation():
    """
    Test set_out_sample_residuals when forecaster has transformer_y and differentiation.
    Stored should equivalent to residuals calculated manually if transformer_y and
    differentiation are applied to `y_true` and `y_pred` before calculating residuals.
    """
    rng = np.random.default_rng(12345)
    series_train = pd.DataFrame({
        'l1': pd.Series(
            rng.normal(loc=0, scale=1, size=100),
            index = pd.date_range(start='1-1-2018', periods=100, freq='D')
        ),
        'l2': pd.Series(
            rng.normal(loc=0, scale=1, size=100),
            index = pd.date_range(start='1-1-2018', periods=100, freq='D')
        )
    })
    y_true  = {
        'l1':rng.normal(loc=0, scale=1, size=5)
    }
    y_pred = {
        'l1':rng.normal(loc=0, scale=1, size=5)
    }
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 2,
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     differentiation    = 1
                 )
    forecaster.fit(series=series_train)
    forecaster.set_out_sample_residuals(
        y_true = y_true,
        y_pred = y_pred
    )

    y_true['l1'] = forecaster.transformer_series_['l1'].transform(y_true['l1'].reshape(-1, 1)).flatten()
    y_pred['l1'] = forecaster.transformer_series_['l1'].transform(y_pred['l1'].reshape(-1, 1)).flatten()
    y_true['l1'] = forecaster.differentiator_['l1'].transform(y_true['l1'])[forecaster.differentiation:]
    y_pred['l1'] = forecaster.differentiator_['l1'].transform(y_pred['l1'])[forecaster.differentiation:]
    residuals = {'l1' : y_true['l1'] - y_pred['l1']}

    assert forecaster.out_sample_residuals_.keys() == residuals.keys()
    for key in residuals.keys():
        np.testing.assert_array_almost_equal(forecaster.out_sample_residuals_[key], residuals[key])
