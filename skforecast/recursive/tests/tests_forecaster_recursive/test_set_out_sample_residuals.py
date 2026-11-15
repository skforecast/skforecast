# Unit test set_out_sample_residuals ForecasterRecursive
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from skforecast.exceptions import ResidualsUsageWarning
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y

freq = "h"


def test_set_out_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    y_true = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}
    y_pred = {1: np.array([1, 2, 3, 4, 5]), 2: np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `set_out_sample_residuals()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_true_is_not_numpy_array_or_pandas_series():
    """
    Test TypeError is raised when y_true argument is not numpy ndarray or pandas Series.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y)
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
    forecaster.fit(y)
    y_true = np.array([1, 2, 3])
    y_pred = 'invalid'

    err_msg = re.escape(
        f"`y_pred` argument must be `numpy ndarray` or `pandas Series`. "
        f"Got {type(y_pred)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_different_length():
    """
    Test ValueError is raised when y_true and y_pred have different length.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y)
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
    forecaster.fit(y)
    y_true = pd.Series([1, 2, 3], index=[1, 2, 3])
    y_pred = pd.Series([1, 2, 3], index=[1, 2, 4])

    err_msg = re.escape("`y_true` and `y_pred` must have the same index.")
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and 
    append is False.
    """
    rng = np.random.default_rng(12345)
    y_true = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    y_pred = pd.Series(rng.normal(loc=10, scale=10, size=1000))

    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y_true)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=False)
    results = np.sort(forecaster.out_sample_residuals_)

    expected = np.sort(y_true - y_pred)

    np.testing.assert_array_almost_equal(results, expected)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and 
    append is True.
    """
    rng = np.random.default_rng(12345)
    y_true = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    y_pred = pd.Series(rng.normal(loc=10, scale=10, size=1000))

    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y_true)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
    results = np.sort(forecaster.out_sample_residuals_)
    
    residuals = (y_true - y_pred)
    expected = np.sort(np.concatenate((residuals, residuals)))

    np.testing.assert_array_almost_equal(results, expected)


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_10000():
    """
    Test length residuals stored when its length is greater than 10_000.
    """
    rng = np.random.RandomState(42)
    y_fit = pd.Series(rng.normal(loc=10, scale=10, size=50_000))

    forecaster = ForecasterRecursive(
        LinearRegression(), lags=1, binner_kwargs={"n_bins": 10}
    )
    forecaster.fit(y_fit)
    X_train, y_train = forecaster.create_train_X_y(y_fit)

    y_pred = forecaster.estimator.predict(X_train)
    forecaster.set_out_sample_residuals(y_true=y_train, y_pred=y_pred)

    assert len(forecaster.out_sample_residuals_) == 10_000
    for v in forecaster.out_sample_residuals_by_bin_.values():
        assert len(v) == 1_000


def test_out_sample_residuals_by_bin_and_in_sample_residuals_by_bin_equivalence():
    """
    Test out sample residuals by bin are equivalent to in-sample residuals by bin
    when training data and training predictions are passed.
    """
    forecaster = ForecasterRecursive(
                     estimator = LinearRegression(),
                     lags = 5,
                     binner_kwargs = {'n_bins': 3}
                 )
    forecaster.fit(y, store_in_sample_residuals=True)

    X_train, y_train = forecaster.create_train_X_y(y)
    forecaster.estimator.fit(X_train, y_train)
    predictions = forecaster.estimator.predict(X_train)

    forecaster.set_out_sample_residuals(
        y_true = y_train,
        y_pred = predictions
    )

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

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=1, binner_kwargs={"n_bins": 2}
    )
    forecaster.fit(y_fit)   
    X_train, y_train = forecaster.create_train_X_y(y=y_fit)
    y_pred = forecaster.estimator.predict(X_train)

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
        estimator=LinearRegression(), lags=5, binner_kwargs={"n_bins": 3}
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
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)

    assert len(forecaster.out_sample_residuals_by_bin_[0]) == len(y_pred)


def test_forecaster_set_out_sample_residuals_when_transformer_y_and_differentiation():
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
                     estimator       = LinearRegression(),
                     lags            = 5,
                     transformer_y   = StandardScaler(),
                     differentiation = 1,
                     binner_kwargs   = {"n_bins": 3}
                 )

    forecaster.fit(y=y_train)
    forecaster.set_out_sample_residuals(
        y_true = y_true,
        y_pred = y_pred
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning
        )
        y_true = forecaster.transformer_y.transform(y_true.to_numpy().reshape(-1, 1)).flatten()
        y_pred = forecaster.transformer_y.transform(y_pred.reshape(-1, 1)).flatten()

    y_true = forecaster.differentiator.transform(y_true)[forecaster.differentiation:]
    y_pred = forecaster.differentiator.transform(y_pred)[forecaster.differentiation:]
    residuals = y_true - y_pred
    residuals = np.sort(residuals)
    out_sample_residuals_ = np.sort(forecaster.out_sample_residuals_)

    np.testing.assert_array_almost_equal(residuals, out_sample_residuals_)


def test_set_out_sample_residuals_ResidualsUsageWarning_when_few_residuals_per_bin():
    """
    Test ResidualsUsageWarning is raised when the average number of
    out-of-sample residuals per bin is lower than 10.
    """
    rng = np.random.default_rng(12345)
    y_true = pd.Series(rng.normal(loc=10, scale=10, size=1000))
    y_pred = pd.Series(rng.normal(loc=10, scale=10, size=1000))

    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y_true)
    n_bins = forecaster.binner.n_bins_
    n_residuals = 5 * n_bins

    warn_msg = re.escape(
        f"Only {n_residuals} out-of-sample residuals are available for "
        f"{n_bins} bins, an average of 5.0 residuals per bin. "
        "With fewer than 10 residuals per bin, prediction intervals estimated "
        "with `use_binned_residuals = True` are likely to be too narrow. Consider "
        "providing more out-of-sample residuals, reducing `n_bins` in the "
        "`binner_kwargs` of the forecaster, or predicting with "
        "`use_binned_residuals = False`."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        forecaster.set_out_sample_residuals(
            y_true=y_true.iloc[:n_residuals], y_pred=y_pred.iloc[:n_residuals]
        )


def test_set_out_sample_residuals_does_not_modify_y_true_y_pred_with_in_place_transformer():
    """
    Test set_out_sample_residuals does not modify `y_true` and `y_pred` when
    `transformer_y` writes into its input (`StandardScaler(copy=False)`), and
    the residuals are those of the transformed values.
    """
    rng = np.random.default_rng(12345)
    y_train = pd.Series(rng.normal(loc=0, scale=1, size=100), index=range(100))
    y_true = pd.Series(rng.normal(loc=0, scale=1, size=36), index=range(100, 136))
    y_pred = rng.uniform(low=-2.5, high=2, size=36)
    y_true_copy = y_true.copy()
    y_pred_copy = y_pred.copy()

    forecaster = ForecasterRecursive(
                     estimator     = LinearRegression(),
                     lags          = 5,
                     transformer_y = StandardScaler(copy=False),
                     binner_kwargs = {"n_bins": 3}
                 )
    forecaster.fit(y=y_train)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    expected = np.array([-1.2772675404369869, -2.6976380487813736, -2.184354224199175,
                         -0.35768531181393204, 2.145222127722728, -0.7016170345274135,
                         -1.6005105944038125, 0.6179990785930884, 1.2259164154453306,
                         1.9371927964836586, 0.9733868646912225, 4.671599052639544,
                         2.109531213117364, -1.230686247743356, -0.6375709773369026,
                         -0.6654440551206414, 1.9205458730789169, 2.722669381564323,
                         0.19652237969909053, -0.07417050381022627, 1.0149846218057323,
                         1.8375518527833448, 2.654763007741273, -1.5348880689994,
                         -2.0072283463984095, -0.4351683512272346, -0.7640091782210141,
                         -0.2647878280519763, -1.0115408078139747, -0.2910138043566606,
                         -1.9933866092276862, -2.776331516691367, 0.980693084848252,
                         -1.9767429696902359, -2.299969542966542, -1.3864428994097509])

    pd.testing.assert_series_equal(y_true, y_true_copy)
    np.testing.assert_array_equal(y_pred, y_pred_copy)
    np.testing.assert_array_almost_equal(forecaster.out_sample_residuals_, expected)
