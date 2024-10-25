# Unit test set_out_sample_residuals ForecasterRecursive
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Fixtures
from .fixtures_forecaster_recursive import y

if pd.__version__ < '2.2.0':
    freq = "H"
else:
    freq = "h"


def test_set_out_sample_residuals_TypeError_when_y_true_is_not_numpy_array_or_pandas_series():
    """
    Test TypeError is raised when y_true argument is not numpy ndarray or pandas Series.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    y_true = 'invalid'
    y_pred = np.array([1, 2, 3])

    err_msg = re.escape(
        f"`y_true` argument must be `numpy ndarray` or `pandas Series`. "
        f"Got {type(y_true)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_pred_is_not_numpy_array_or_pandas_series():
    """
    Test TypeError is raised when y_pred argument is not numpy ndarray or pandas Series.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    y_true = np.array([1, 2, 3])
    y_pred = 'invalid'

    err_msg = re.escape(
        f"`y_pred` argument must be `numpy ndarray` or `pandas Series`. "
        f"Got {type(y_pred)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_different_lenght():
    """
    Test ValueError is raised when y_true and y_pred have different length.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3, 4])

    err_msg = re.escape(
        f"`y_true` and `y_pred` must have the same length. "
        f"Got {len(y_true)} and {len(y_pred)}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_different_index():
    """
    Test ValueError is raised when residuals and y_pred have different index.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    y_true = pd.Series([1, 2, 3], index=[1, 2, 3])
    y_pred = pd.Series([1, 2, 3], index=[1, 2, 4])

    err_msg = re.escape("`y_true` and `y_pred` must have the same index.")
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_10000():
    """
    Test length residuals stored when its length is greater than 10_000.
    """
    rng = np.random.RandomState(42)
    y_true = pd.Series(rng.normal(loc=10, scale=10, size=50_000))
    forecaster = ForecasterRecursive(LinearRegression(), lags=1)
    forecaster.fit(y_true)
    X_train, y_train = forecaster.create_train_X_y(y_true)
    y_pred = forecaster.regressor.predict(X_train)    
    forecaster.set_out_sample_residuals(y_true=y_train, y_pred=y_pred)

    for v in forecaster.out_sample_residuals_by_bin_.values():
        assert len(v) == 1000
    assert len(forecaster.out_sample_residuals_) == 10_000


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and append
    is False.
    """
    rng = np.random.default_rng(12345)
    y_true = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    y_pred = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y_true)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=False)
    expected = np.sort(y_true - y_pred)
    results = np.sort(forecaster.out_sample_residuals_)

    np.testing.assert_almost_equal(results, expected)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and append
    is True.
    """
    rng = np.random.default_rng(12345)
    y_true = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    y_pred = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y_true)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
    residuals = (y_true - y_pred)
    expected = np.sort(np.concatenate((residuals, residuals)))
    results = np.sort(forecaster.out_sample_residuals_)

    np.testing.assert_almost_equal(results, expected)


def test_out_sample_residuals_by_bin_and_in_sample_reseiduals_by_bin_equivalence():
    """
    Test out sample residuals by bin are quivalent to insample residuals by bin
    when training data and training predictions are passed.
    """
    forecaster = ForecasterRecursive(
                     regressor = LinearRegression(),
                     lags = 5,
                     binner_kwargs = {'n_bins': 3}
                 )
    forecaster.fit(y)
    X_train, y_train = forecaster.create_train_X_y(y)
    forecaster.regressor.fit(X_train, y_train)
    predictions = forecaster.regressor.predict(X_train)

    forecaster.set_out_sample_residuals(
        y_true=y_train,
        y_pred=predictions
    )

    for k in forecaster.out_sample_residuals_by_bin_.keys():
        np.testing.assert_almost_equal(
            forecaster.in_sample_residuals_by_bin_[k],
            forecaster.out_sample_residuals_by_bin_[k]
        )


def test_set_out_sample_residuals_stores_maximum_1000_residuals_per_bin_when_n_bin_is_10():
    """
    Test that set_out_sample_residuals stores a maximum of 1000 residuals per bin
    when n_bins = 10.
    """
    y = pd.Series(
        data=np.random.normal(loc=10, scale=1, size=20_000),
        index=pd.date_range(start="01-01-2000", periods=20_000, freq="h"),
    )
    y_pred = y
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(), lags=5, binner_kwargs={"n_bins": 10}
    )
    forecaster.fit(y)
    forecaster.set_out_sample_residuals(
        y_true=y,
        y_pred=y_pred
    )

    for v in forecaster.out_sample_residuals_by_bin_.values():
        assert len(v) <= 1000

    np.testing.assert_array_almost_equal(
        forecaster.out_sample_residuals_,
        np.concatenate(list(forecaster.out_sample_residuals_by_bin_.values())),
    )


def test_set_out_sample_residuals_append_new_residuals_per_bin():
    """
    Test that set_out_sample_residuals append residuals per bin until it
    reaches the max allowed size of 10_000 // n_bins
    """
    rng = np.random.default_rng(12345)
    y = pd.Series(
        data=rng.normal(loc=10, scale=1, size=1001),
        index=pd.date_range(start="01-01-2000", periods=1001, freq="h"),
    )
    forecaster = ForecasterRecursive(
        regressor=LinearRegression(), lags=1, binner_kwargs={"n_bins": 2}
    )
    forecaster.fit(y)   
    X_train, y_train = forecaster.create_train_X_y(y=y)
    y_pred = forecaster.regressor.predict(X_train)

    for i in range(1, 20):
        forecaster.set_out_sample_residuals(y_true=y_train, y_pred=y_pred, append=True)
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

    forecaster = ForecasterRecursive(
        regressor=LinearRegression(), lags=5, binner_kwargs={"n_bins": 3}
    )
    forecaster.fit(y)
    y_pred = y.loc[y > 10]
    y_true = y_pred + rng.normal(loc=0, scale=1, size=len(y_pred))

    warn_msg = re.escape(
            f"The following bins have no out of sample residuals: [0]. "
            f"No predicted values fall in the interval "
            f"[{forecaster.binner_intervals_[0]}]. "
            f"Empty bins will be filled with a random sample of residuals."
    )
    with pytest.warns(UserWarning, match=warn_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)

    assert len(forecaster.out_sample_residuals_by_bin_[0]) == 3333


def test_forecaster_set_outsample_residuals_when_transformer_y_and_diferentiation():
    """
    Test set_out_sample_residuals when forecaster has transformer_y and differentiation.
    Stored should equivalent to residuals calculated manually if transformer_y and
    differentiation are applied to `y_true` and `y_pred` before calculating residuals.
    """
    rng = np.random.default_rng(12345)
    y_train = pd.Series(rng.normal(loc=0, scale=1, size=100), index=range(100))
    y_true  = pd.Series(rng.normal(loc=0, scale=1, size=36), index=range(100, 136))
    y_pred = rng.uniform(low=-2.5, high=2, size=36)
    forecaster = ForecasterRecursive(
                        regressor       = LinearRegression(),
                        lags            = 5,
                        differentiation = 1,
                        transformer_y   = StandardScaler(),
                        binner_kwargs   = {"n_bins": 3}
                    )

    forecaster.fit(y=y_train)
    forecaster.set_out_sample_residuals(
        y_true = y_true,
        y_pred = y_pred
    )

    y_true = forecaster.transformer_y.transform(y_true.to_numpy().reshape(-1, 1)).flatten()
    y_pred = forecaster.transformer_y.transform(y_pred.reshape(-1, 1)).flatten()
    y_true = forecaster.differentiator.transform(y_true)[forecaster.differentiation:]
    y_pred = forecaster.differentiator.transform(y_pred)[forecaster.differentiation:]
    residuals = y_true - y_pred
    residuals = np.sort(residuals)
    out_sample_residuals_ = np.sort(forecaster.out_sample_residuals_)

    np.testing.assert_array_almost_equal(residuals, out_sample_residuals_)