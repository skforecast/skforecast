# Unit test fit ForecasterRecursive
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog


def custom_weights(index):  # pragma: no cover
    """
    Return 0 if index is one of '2022-01-05', '2022-01-06', 1 otherwise.
    """
    weights = np.where(
                (index >= 20) & (index <= 40),
                0,
                1
              )
    
    return weights


def test_forecaster_y_exog_features_stored():
    """
    Test forecaster stores y and exog features after fitting.
    """
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    forecaster = ForecasterRecursive(
        LinearRegression(), lags=3, window_features=rolling
    )
    forecaster.fit(y=y, exog=exog)

    exog_in_ = True
    exog_type_in_ = type(exog)
    exog_names_in_ = ['exog']
    exog_dtypes_in_ = {'exog': exog.dtype}
    X_train_window_features_names_out_ = ['roll_ratio_min_max_4', 'roll_median_4']
    X_train_exog_names_out_ = ['exog']
    X_train_features_names_out_ = ['lag_1', 'lag_2', 'lag_3', 
                                   'roll_ratio_min_max_4', 'roll_median_4', 'exog']
    
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_exog_names_out_ == X_train_exog_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq.
    """
    serie_with_DatetimeIndex = pd.Series(
        data  = [1, 2, 3, 4, 5],
        index = pd.date_range(start='2022-01-01', periods=5)
    )
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freqstr
    results = forecaster.index_freq_

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq.
    """
    y = pd.Series(data=np.arange(10))
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=y)
    expected = y.index.step
    results = forecaster.index_freq_

    assert results == expected
    
    
def test_fit_in_sample_residuals_stored():
    """
    Test that values of in_sample_residuals_ are stored after fitting.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)))
    results = forecaster.in_sample_residuals_
    expected = np.array([0., 0.])

    assert isinstance(results, np.ndarray)
    np.testing.assert_array_equal(results, expected)


def test_fit_in_sample_residuals_by_bin_stored():
    """
    Test that values of in_sample_residuals_by_bin are stored after fitting.
    """
    forecaster = ForecasterRecursive(
                     regressor     = LinearRegression(),
                     lags          = 5,
                     binner_kwargs = {'n_bins': 3}
                 )
    forecaster.fit(y)

    X_train, y_train = forecaster.create_train_X_y(y)
    forecaster.regressor.fit(X_train, y_train)
    predictions_regressor = forecaster.regressor.predict(X_train)
    expected_1 = y_train - predictions_regressor

    expected_2 = {
        0: np.array([
                0.0334789 , -0.12428472,  0.34053202, -0.40668544, -0.29246428,
                0.16990408, -0.02118736, -0.24234062, -0.11745596,  0.1697826 ,
                -0.01432662, -0.00063421, -0.03462192,  0.41322689,  0.19077889
            ]),
        1: np.array([
                -0.07235524, -0.10880301, -0.07773704, -0.09070227,  0.21559424,
                -0.29380582,  0.03359274,  0.10109702,  0.2080735 , -0.17086244,
                0.01929597, -0.09396861, -0.0670198 ,  0.38248168, -0.01100463
            ]),
        2: np.array([
                0.44780048,  0.03560524, -0.04960603,  0.24323339,  0.12651656,
                -0.46533293, -0.17532266, -0.24111645,  0.3805961 , -0.05842153,
                0.08927473, -0.42295249, -0.32047616,  0.38902396, -0.01640072
            ])
    }

    expected_3 = {
        0: (0.31791969404305154, 0.47312737276420375),
        1: (0.47312737276420375, 0.5259220171775293),
        2: (0.5259220171775293, 0.6492244994657664)
    }

    np.testing.assert_almost_equal(
        np.sort(forecaster.in_sample_residuals_),
        np.sort(expected_1)
    )
    for k in expected_2.keys():
        np.testing.assert_almost_equal(forecaster.in_sample_residuals_by_bin_[k], expected_2[k])
    for k in expected_3.keys():
        assert forecaster.binner_intervals_[k][0] == approx(expected_3[k][0])
        assert forecaster.binner_intervals_[k][1] == approx(expected_3[k][1])


def test_fit_same_residuals_when_residuals_greater_than_10000():
    """
    Test fit return same residuals when residuals len is greater than 10_000.
    Testing with two different forecaster.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(12000)))
    results_1 = forecaster.in_sample_residuals_
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(12000)))
    results_2 = forecaster.in_sample_residuals_
    
    assert isinstance(results_1, np.ndarray)
    assert isinstance(results_2, np.ndarray)
    np.testing.assert_array_equal(results_1, results_2)


def test_fit_in_sample_residuals_not_stored():
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False`.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(5)), store_in_sample_residuals=False)
    results = forecaster.in_sample_residuals_

    assert results is None


@pytest.mark.parametrize("store_last_window", 
                         [True, False], 
                         ids=lambda lw: f'store_last_window: {lw}')
def test_fit_last_window_stored(store_last_window):
    """
    Test that values of last window are stored after fitting.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(50)), store_last_window=store_last_window)
    expected = pd.DataFrame(np.array([47, 48, 49]), index=[47, 48, 49], columns=['y'])

    if store_last_window:
        pd.testing.assert_frame_equal(forecaster.last_window_, expected)
    else:
        assert forecaster.last_window_ is None


def test_fit_model_coef_when_using_weight_func():
    """
    Check the value of the regressor coefs when using a `weight_func`.
    """
    forecaster = ForecasterRecursive(
                     regressor   = LinearRegression(),
                     lags        = 5,
                     weight_func = custom_weights
                 )
    forecaster.fit(y=y)
    results = forecaster.regressor.coef_
    expected = np.array([0.01211677, -0.20981367,  0.04214442, -0.0369663, -0.18796105])

    np.testing.assert_almost_equal(results, expected)


def test_fit_model_coef_when_not_using_weight_func():
    """
    Check the value of the regressor coefs when not using a `weight_func`.
    """
    forecaster = ForecasterRecursive(regressor=LinearRegression(), lags=5)
    forecaster.fit(y=y)
    results = forecaster.regressor.coef_
    expected = np.array([0.16773502, -0.09712939,  0.10046413, -0.09971515, -0.15849756])

    np.testing.assert_almost_equal(results, expected)
