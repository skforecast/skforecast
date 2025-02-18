
# Unit test fit ForecasterDirect
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirect

# Fixtures
from .fixtures_forecaster_direct import y
from .fixtures_forecaster_direct import exog


def test_forecaster_y_exog_features_stored():
    """
    Test forecaster stores y and exog features after fitting.
    """
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    forecaster = ForecasterDirect(
        LinearRegression(), lags=3, steps=2, window_features=rolling
    )
    forecaster.fit(y=y, exog=exog)

    exog_in_ = True
    exog_type_in_ = type(exog)
    exog_names_in_ = ['exog']
    exog_dtypes_in_ = {'exog': exog.dtype}
    X_train_window_features_names_out_ = ['roll_ratio_min_max_4', 'roll_median_4']
    X_train_exog_names_out_ = ['exog']
    X_train_direct_exog_names_out_ = ['exog_step_1', 'exog_step_2']
    X_train_features_names_out_ = [
        'lag_1', 'lag_2', 'lag_3', 
        'roll_ratio_min_max_4', 'roll_median_4', 'exog_step_1', 'exog_step_2'
    ]
    
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_exog_names_out_ == X_train_exog_names_out_
    assert forecaster.X_train_direct_exog_names_out_ == X_train_direct_exog_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq_.
    """
    serie_with_DatetimeIndex = pd.Series(
        data  = np.arange(10),
        index = pd.date_range(start='2022-01-01', periods=10)
    )
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freqstr
    results = forecaster.index_freq_

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq_.
    """
    y = pd.Series(data=np.arange(10))
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=y)
    expected = y.index.step
    results = forecaster.index_freq_

    assert results == expected
    

@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_stored(n_jobs):
    """
    Test that values of in_sample_residuals_ are stored after fitting.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(y=pd.Series(np.arange(5)))
    results = forecaster.in_sample_residuals_

    expected = {
        1: np.array([0.]),
        2: np.array([0.])
    }

    assert isinstance(results, dict)
    assert results.keys() == expected.keys()
    for k in expected.keys():
        assert isinstance(results[k], np.ndarray)
        np.testing.assert_array_almost_equal(results[k], expected[k])


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_same_residuals_when_residuals_greater_than_10000(n_jobs):
    """
    Test fit return same residuals when residuals len is greater than 10_000.
    Testing with two different forecaster.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(y=pd.Series(np.arange(12_000)))
    results_1 = forecaster.in_sample_residuals_

    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(y=pd.Series(np.arange(12_000)))
    results_2 = forecaster.in_sample_residuals_

    assert isinstance(results_1, dict)
    assert isinstance(results_2, dict)
    assert results_1.keys() == results_2.keys()
    for k in results_1.keys():
        assert isinstance(results_1[k], np.ndarray)
        assert isinstance(results_2[k], np.ndarray)
        assert len(results_1[k] == 10_000)
        assert len(results_2[k] == 10_000)
        np.testing.assert_array_equal(results_1[k], results_2[k])


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_by_bin_stored(n_jobs):
    """
    Test that values of in_sample_residuals_by_bin are stored after fitting.
    """
    forecaster = ForecasterDirect(
        LinearRegression(), lags=3, steps=2, binner_kwargs={'n_bins': 3}, n_jobs=n_jobs
    )
    forecaster.fit(y)

    expected_1 = {
        1: np.array([ 0.04393357,  0.18398148, -0.08688424,  0.51870003,  0.06216896,
            0.02452184, -0.12608551, -0.16259998,  0.23758614, -0.12067947,
            -0.38402291, -0.05926222,  0.18972306, -0.3385514 , -0.22649077,
            0.01348154,  0.00174659,  0.1596128 ,  0.32282691,  0.17531481,
            0.10745551,  0.1995734 , -0.22535703, -0.08763543, -0.29992457,
            -0.16705068,  0.13439907, -0.44182743,  0.04125504, -0.13693395,
            0.02970289, -0.08200536, -0.17358741, -0.05563861,  0.38006595,
            0.37597605, -0.0144369 ,  0.15009597, -0.45267465, -0.10006534,
            -0.12798406,  0.38508591, -0.31515422,  0.08711746,  0.41108872,
            -0.05056158]),
        2: np.array([ 0.21973412, -0.07827408,  0.46573298,  0.15786283,  0.04787789,
            -0.14081875, -0.13513316,  0.22976323, -0.07790008, -0.42298643,
            -0.14762595,  0.21302143, -0.30705996, -0.34168009, -0.03513684,
            0.03880089,  0.12527145,  0.31242716,  0.23101186,  0.13115123,
            0.22330156, -0.15775149, -0.11087341, -0.30329788, -0.19297871,
            0.09058465, -0.42792044, -0.07029422, -0.14640934,  0.02323429,
            -0.12154526, -0.19662895, -0.09436159,  0.37245731,  0.43690674,
            0.02167369,  0.12380491, -0.38864242, -0.13622859, -0.14006171,
            0.38181952, -0.28740547,  0.00152984,  0.41738958,  0.06388605,
            0.13177162])
    }

    expected_2 = {
        0: np.array([ 0.51870003,  0.02452184, -0.38402291, -0.05926222, -0.22649077,
         0.1596128 , -0.08763543, -0.16705068,  0.04125504,  0.02970289,
        -0.17358741, -0.05563861,  0.15009597, -0.10006534,  0.38508591,
         0.08711746,  0.04787789, -0.13513316, -0.42298643, -0.30705996,
         0.13115123, -0.15775149, -0.11087341, -0.19297871,  0.02323429,
         0.02167369, -0.13622859,  0.38181952,  0.00152984,  0.06388605,
         0.13177162]),
        1: np.array([ 0.04393357, -0.08688424, -0.12608551, -0.16259998,  0.23758614,
        -0.3385514 ,  0.01348154,  0.10745551,  0.13439907, -0.08200536,
         0.38006595, -0.0144369 ,  0.21973412, -0.07827408,  0.46573298,
         0.22976323, -0.07790008, -0.34168009,  0.03880089,  0.12527145,
         0.23101186,  0.22330156, -0.42792044, -0.07029422, -0.19662895,
        -0.09436159,  0.37245731,  0.43690674,  0.12380491, -0.38864242]),
        2: np.array([ 0.18398148,  0.06216896, -0.12067947,  0.18972306,  0.00174659,
         0.32282691,  0.17531481,  0.1995734 , -0.22535703, -0.29992457,
        -0.44182743, -0.13693395,  0.37597605, -0.45267465, -0.12798406,
        -0.31515422,  0.41108872, -0.05056158,  0.15786283, -0.14081875,
        -0.14762595,  0.21302143, -0.03513684,  0.31242716, -0.30329788,
         0.09058465, -0.14640934, -0.12154526, -0.14006171, -0.28740547,
         0.41738958])
    }

    expected_3 = {
        0: (0.39244612759441666, 0.4901889798207174),
        1: (0.4901889798207174, 0.5222610284825959),
        2: (0.5222610284825959, 0.6226607762583838)
    }

    for k in forecaster.in_sample_residuals_.keys():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_[k], expected_1[k]
        )
    for k in forecaster.in_sample_residuals_by_bin_.keys():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_by_bin_[k], expected_2[k]
        )
    for k in forecaster.binner_intervals_.keys():
        assert forecaster.binner_intervals_[k][0] == approx(expected_3[k][0])
        assert forecaster.binner_intervals_[k][1] == approx(expected_3[k][1])


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_not_stored(n_jobs):
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False`.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2, n_jobs=n_jobs)
    forecaster.fit(y=pd.Series(np.arange(5)), store_in_sample_residuals=False)
    expected = {1: None, 2: None}
    results = forecaster.in_sample_residuals_

    assert isinstance(results, dict)
    assert results.keys() == expected.keys()
    for k in results.keys():
        assert results[k] == expected[k]
    assert forecaster.in_sample_residuals_by_bin_ is None
    assert forecaster.binner_intervals_ is None


@pytest.mark.parametrize("store_last_window", 
                         [True, False], 
                         ids=lambda lw: f'store_last_window: {lw}')
def test_fit_last_window_stored(store_last_window):
    """
    Test that values of last window are stored after fitting.
    """
    y = pd.Series(np.arange(20), name='y')
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=y, store_last_window=store_last_window)

    expected = pd.Series(
        np.array([17, 18, 19]), index=[17, 18, 19]
    ).to_frame(name='y')

    if store_last_window:
        pd.testing.assert_frame_equal(forecaster.last_window_, expected)
    else:
        assert forecaster.last_window_ is None
