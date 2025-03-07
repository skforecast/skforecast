# Unit test _predict_interval_conformal ForecasterDirectMultiVariate
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from skforecast.direct import ForecasterDirectMultiVariate

# Fixtures
from .fixtures_forecaster_direct_multivariate import series
series_2 = pd.DataFrame({'l1': np.arange(10), 'l2': np.arange(10, 20)})


def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=3, transformer_series=None
    )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.in_sample_residuals_ = {
        1: np.full_like(forecaster.in_sample_residuals_[1], fill_value=10),
        2: np.full_like(forecaster.in_sample_residuals_[2], fill_value=20)
    }
    results = forecaster._predict_interval_conformal(
        steps=1, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=11, step=1)
               )
    expected.insert(0, 'level', np.tile(['l1'], 1))

    pd.testing.assert_frame_equal(results, expected)

    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=3, transformer_series=None
    )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.in_sample_residuals_ = {
        1: np.full_like(forecaster.in_sample_residuals_[1], fill_value=10),
        2: np.full_like(forecaster.in_sample_residuals_[2], fill_value=20)
    }
    results = forecaster._predict_interval_conformal(
        steps=2, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.],
                                       [11., -9., 31.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=12, step=1)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=3, transformer_series=None
    )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = {
        1: np.full_like(forecaster.in_sample_residuals_[1], fill_value=10),
        2: np.full_like(forecaster.in_sample_residuals_[2], fill_value=20)
    }
    results = forecaster._predict_interval_conformal(
        steps=1, nominal_coverage=0.95, use_in_sample_residuals=False, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=11, step=1)
               )
    expected.insert(0, 'level', np.tile(['l1'], 1))

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=3, transformer_series=None
    )
    forecaster.fit(series=series_2, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = {
        1: np.full_like(forecaster.in_sample_residuals_[1], fill_value=10),
        2: np.full_like(forecaster.in_sample_residuals_[2], fill_value=20)
    }
    results = forecaster._predict_interval_conformal(
        steps=2, nominal_coverage=0.95, use_in_sample_residuals=False, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.],
                                       [11., -9., 31.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=12, step=1)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_regressor_is_LinearRegression():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.63114259,  0.23482357,  1.02746162],
                              [0.3800417 , -0.05044528,  0.81052868],
                              [0.33255977, -0.0626732 ,  0.72779273]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.63114259,  0.23482357,  1.02746162],
                              [0.3800417 , -0.05044528,  0.81052868],
                              [0.33255977, -0.0626732 ,  0.72779273]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_out_sample_residuals():
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as out-sample residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster._predict_interval_conformal(
        nominal_coverage=0.95, use_in_sample_residuals=False, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.63114259,  0.23482357,  1.02746162],
                              [0.3800417 , -0.05044528,  0.81052868],
                              [0.33255977, -0.0626732 ,  0.72779273]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_binned_residuals():
    """
    Test predict output when using binned residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.63114259, 0.17603311, 1.08625208],
                              [0.3800417 , 0.12832655, 0.63175685],
                              [0.33255977, 0.08084462, 0.58427492]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_binned__out_sample_residuals():
    """
    Test predict output when using binned out-sample residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_
    results = forecaster._predict_interval_conformal(
        nominal_coverage=0.95, use_in_sample_residuals=False, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.63114259, 0.17603311, 1.08625208],
                              [0.3800417 , 0.12832655, 0.63175685],
                              [0.33255977, 0.08084462, 0.58427492]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_with_differentiation():
    """
    Test predict output when using differentiation.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     differentiation    = 1
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.75141456,  0.31429586,  1.18853326],
                              [0.64535259, -0.28024294,  1.57094811],
                              [0.63651233, -0.77584002,  2.04886467]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))
    
    pd.testing.assert_frame_equal(results, expected)
