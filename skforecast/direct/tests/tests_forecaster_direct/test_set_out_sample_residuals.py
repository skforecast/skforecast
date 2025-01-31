# Unit test set_out_sample_residuals ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from skforecast.exceptions import ResidualsUsageWarning
from skforecast.direct import ForecasterDirect

# Fixtures
y = pd.Series(np.arange(15))


def test_set_out_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    y_true = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}
    y_pred = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}

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
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.is_fitted = True
    y_true = 'not_dict'
    y_pred = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}

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
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.is_fitted = True
    y_true = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}
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
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.is_fitted = True
    y_true = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}
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
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.is_fitted = True
    y_true = {1: 'not_ndarray'}
    y_pred = {1: np.array([1, 2, 3, 4, 5])}
    
    err_msg = re.escape(
        f"Values of `y_true` must be numpy ndarrays or pandas Series. "
        f"Got {type(y_true[1])} for step 1."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_pred_contains_no_numpy_ndarrays_or_pandas_series():
    """
    Test TypeError is raised when y_pred contains no numpy ndarrays or pandas series.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.is_fitted = True
    y_true = {1: np.array([1, 2, 3, 4, 5])}
    y_pred = {1: 'not_ndarray'}
    err_msg = re.escape(
        f"Values of `y_pred` must be numpy ndarrays or pandas Series. "
        f"Got {type(y_pred[1])} for step 1."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_elements_with_different_lengths():
    """
    Test ValueError is raised when y_true and y_pred have elements with different lengths.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.is_fitted = True
    y_true = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}
    y_pred = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2])}
    
    err_msg = re.escape(
        f"`y_true` and `y_pred` must have the same length. "
        f"Got {len(y_true[2])} and {len(y_pred[2])} for step 2."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_series_with_different_indexes():
    """
    Test ValueError is raised when y_true and y_pred have series with different indexes.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.is_fitted = True
    y_true = {1: pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])}
    y_pred = {1: pd.Series([1, 2, 3, 4, 5])}
    
    err_msg = re.escape(
        "When containing pandas Series, elements in `y_true` and "
        "`y_pred` must have the same index. Error in step 1."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_inputs_does_not_match_any_step():
    """
    Test ValueError is raised when inputs does not contain keys that match any step.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=y)
    y_true = {5: np.array([1, 2, 3])}
    y_pred = {5: np.array([1, 2, 3])}

    err_msg = re.escape(
        "Provided keys in `y_pred` and `y_true` do not match any step. "
        "Residuals cannot be updated."
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
        1: pd.Series(rng.normal(loc=10, scale=10, size=1000)), 
        2: pd.Series(rng.normal(loc=10, scale=10, size=1000))}
    y_pred = {
        1: pd.Series(rng.normal(loc=10, scale=10, size=1000)), 
        2: pd.Series(rng.normal(loc=10, scale=10, size=1000))
    }

    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=y)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=False)
    results = {
        1: np.sort(forecaster.out_sample_residuals_[1]),
        2: np.sort(forecaster.out_sample_residuals_[2])
    }

    expected = {
        1: np.sort(y_true[1] - y_pred[1]),
        2: np.sort(y_true[2] - y_pred[2])
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
        1: pd.Series(rng.normal(loc=10, scale=10, size=1000)), 
        2: pd.Series(rng.normal(loc=10, scale=10, size=1000))}
    y_pred = {
        1: pd.Series(rng.normal(loc=10, scale=10, size=1000)), 
        2: pd.Series(rng.normal(loc=10, scale=10, size=1000))
    }

    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=y)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
    results = {
        1: np.sort(forecaster.out_sample_residuals_[1]),
        2: np.sort(forecaster.out_sample_residuals_[2])
    }

    residuals_1 = (y_true[1] - y_pred[1])
    residuals_2 = (y_true[2] - y_pred[2])
    expected = {
        1: np.sort(np.concatenate((residuals_1, residuals_1))),
        2: np.sort(np.concatenate((residuals_2, residuals_2)))
    }

    assert forecaster.out_sample_residuals_.keys() == expected.keys()
    for key in results.keys():
        np.testing.assert_array_almost_equal(expected[key], results[key])


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_10000():
    """
    Test length residuals stored when its length is greater than 10_000.
    """
    rng = np.random.RandomState(42)
    y_fit = pd.Series(rng.normal(loc=10, scale=10, size=50_000))

    forecaster = ForecasterDirect(
        LinearRegression(), lags=1, steps=1, binner_kwargs={"n_bins": 10}
    )
    forecaster.fit(y=y_fit)
    X_train, y_train = forecaster.create_train_X_y(y_fit)
    X_train_step_1, y_train_step_1 = forecaster.filter_train_X_y_for_step(
        step=1, X_train=X_train, y_train=y_train
    )

    y_true = {1: y_train_step_1}
    y_pred = {1: forecaster.regressors_[1].predict(X_train_step_1)}
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    assert list(forecaster.out_sample_residuals_.keys()) == [1]
    for v in forecaster.out_sample_residuals_.values():
        assert len(v) == 10_000
    for v in forecaster.out_sample_residuals_by_bin_.values():
        assert len(v) == 1_000


def test_out_sample_residuals_by_bin_and_in_sample_reseiduals_by_bin_equivalence():
    """
    Test out sample residuals by bin are quivalent to insample residuals by bin
    when training data and training predictions are passed.
    """
    forecaster = ForecasterDirect(
                     regressor     = LinearRegression(),
                     steps         = 3,
                     lags          = 5,
                     binner_kwargs = {'n_bins': 3}
                 )
    forecaster.fit(y)
    X_train, y_train = forecaster.create_train_X_y(y)

    y_true = {}
    y_pred = {}
    for step in range(1, forecaster.steps + 1):
        X_train_step, y_train_step = forecaster.filter_train_X_y_for_step(
            step=step, X_train=X_train, y_train=y_train
        )
        y_true[step] = y_train_step
        y_pred[step] = forecaster.regressors_[step].predict(X_train_step)

    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    assert forecaster.in_sample_residuals_by_bin_.keys() == forecaster.out_sample_residuals_by_bin_.keys()
    for k in forecaster.out_sample_residuals_by_bin_.keys():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_by_bin_[k],
            forecaster.out_sample_residuals_by_bin_[k]
        )


def test_set_out_sample_residuals_append_new_residuals_per_bin():
    """
    Test that set_out_sample_residuals append residuals per bin until it
    reaches the max allowed size of 10_000 // n_bins
    """
    rng = np.random.default_rng(12345)
    y_fit = pd.Series(
        data=rng.normal(loc=10, scale=1, size=1001),
        index=pd.date_range(start="01-01-2000", periods=1001, freq="h"),
    )

    forecaster = ForecasterDirect(
        LinearRegression(), lags=1, steps=1, binner_kwargs={"n_bins": 2}
    )
    forecaster.fit(y=y_fit)
    X_train, y_train = forecaster.create_train_X_y(y_fit)
    X_train_step_1, y_train_step_1 = forecaster.filter_train_X_y_for_step(
        step=1, X_train=X_train, y_train=y_train
    )

    y_true = {1: y_train_step_1}
    y_pred = {1: forecaster.regressors_[1].predict(X_train_step_1)}
    for i in range(1, 20):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
        for v in forecaster.out_sample_residuals_by_bin_.values():
            assert len(v) == min(5_000, 500 * i)


def test_set_out_sample_residuals_when_there_are_no_residuals_for_some_bins():
    """
    Test that set_out_sample_residuals works when there are no residuals for some bins.
    """
    rng = np.random.default_rng(12345)
    y = pd.Series(
            data=rng.normal(loc=10, scale=1, size=100),
            index=pd.date_range(start="01-01-2000", periods=100, freq="h"),
        )

    forecaster = ForecasterDirect(
        regressor=LinearRegression(), steps=1, lags=5, binner_kwargs={"n_bins": 3}
    )
    forecaster.fit(y)
    y_pred = {1: y.loc[y > 10]}
    y_true = {1: y_pred[1] + rng.normal(loc=0, scale=1, size=len(y_pred[1]))}

    warn_msg = re.escape(
        f"The following bins have no out of sample residuals: [0]. "
        f"No predicted values fall in the interval "
        f"[{forecaster.binner_intervals_[0]}]. "
        f"Empty bins will be filled with a random sample of residuals."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)

    assert len(forecaster.out_sample_residuals_by_bin_[0]) == len(y_pred[1])


def test_set_out_sample_residuals_when_residuals_keys_partially_match():
    """
    Test residuals are stored only for matching keys.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=y)
    y_pred = {1: np.repeat(1, 5), 4: np.arange(10)}
    y_true = {1: np.arange(5), 4: np.arange(10)}
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    results = forecaster.out_sample_residuals_
    
    expected = {1: np.array([-1,  0,  1,  2,  3]), 2: None}

    for key in results.keys():
        if results[key] is not None:
            np.testing.assert_array_almost_equal(results[key], expected[key])
        else:
            assert results[key] is None


def test_forecaster_set_outsample_residuals_when_transformer_y_and_diferentiation():
    """
    Test set_out_sample_residuals when forecaster has transformer_y and differentiation.
    Stored should equivalent to residuals calculated manually if transformer_y and
    differentiation are applied to `y_true` and `y_pred` before calculating residuals.
    """
    rng = np.random.default_rng(12345)
    y_train = pd.Series(rng.normal(loc=0, scale=1, size=100), index=range(100))
    y_true  = {
        1: rng.normal(loc=0, scale=1, size=5),
        2: rng.normal(loc=0, scale=1, size=5)
    }
    y_pred = {
        1: rng.normal(loc=0, scale=1, size=5),
        2: rng.normal(loc=0, scale=1, size=5)
    }
    forecaster = ForecasterDirect(
                     regressor       = LinearRegression(),
                     lags            = 5,
                     steps           = 2, 
                     differentiation = 1,
                     transformer_y   = StandardScaler(),
                 )
    forecaster.fit(y=y_train)
    forecaster.set_out_sample_residuals(
        y_true = y_true,
        y_pred = y_pred
    )

    y_true[1] = forecaster.transformer_y.transform(y_true[1].reshape(-1, 1)).flatten()
    y_true[2] = forecaster.transformer_y.transform(y_true[2].reshape(-1, 1)).flatten()
    y_pred[1] = forecaster.transformer_y.transform(y_pred[1].reshape(-1, 1)).flatten()
    y_pred[2] = forecaster.transformer_y.transform(y_pred[2].reshape(-1, 1)).flatten()
    y_true[1] = forecaster.differentiator.transform(y_true[1])[forecaster.differentiation:]
    y_true[2] = forecaster.differentiator.transform(y_true[2])[forecaster.differentiation:]
    y_pred[1] = forecaster.differentiator.transform(y_pred[1])[forecaster.differentiation:]
    y_pred[2] = forecaster.differentiator.transform(y_pred[2])[forecaster.differentiation:]
    residuals = {}
    residuals[1] = y_true[1] - y_pred[1]
    residuals[2] = y_true[2] - y_pred[2]

    assert forecaster.out_sample_residuals_.keys() == residuals.keys()
    for key in residuals.keys():
        np.testing.assert_array_almost_equal(forecaster.out_sample_residuals_[key], residuals[key])
