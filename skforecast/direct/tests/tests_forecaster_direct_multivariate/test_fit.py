# Unit test fit ForecasterDirectMultiVariate
# ==============================================================================
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate

# Fixtures
from .fixtures_forecaster_direct_multivariate import series as series_fixtures
from .fixtures_forecaster_direct_multivariate import exog

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


@pytest.mark.parametrize(
    "forecaster_kwargs",
    [
        {"estimator": LinearRegression(), "level": "l1", "lags": 3, "steps": 2},
        {"estimator": LinearRegression(), "level": "l1", "lags": 3, "steps": 2,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4)},
        {"estimator": LinearRegression(), "level": "l1", "lags": 3, "steps": 2,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4),
         "transformer_series": StandardScaler(), "transformer_exog": StandardScaler()},
        {"estimator": LinearRegression(), "level": "l1", "lags": 3, "steps": 2,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4),
         "transformer_series": StandardScaler(), "transformer_exog": StandardScaler(),
         "differentiation": 1},
    ],
    ids=["base", "window_features", "transformers", "differentiation"]
)
def test_forecaster_fit_does_not_modify_series_exog(forecaster_kwargs):
    """
    Test forecaster.fit does not modify series and exog.
    """
    series_local = series_fixtures.copy()
    exog_local = exog[['exog_1']].copy()
    series_copy = series_local.copy()
    exog_copy = exog_local.copy()

    forecaster = ForecasterDirectMultiVariate(**forecaster_kwargs)
    forecaster.fit(series=series_local, exog=exog_local)

    pd.testing.assert_frame_equal(series_local, series_copy)
    pd.testing.assert_frame_equal(exog_local, exog_copy)


def test_forecaster_series_exog_features_stored():
    """
    Test forecaster stores series and exog features after fitting.
    """
    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    forecaster = ForecasterDirectMultiVariate(
                     estimator        = LinearRegression(), 
                     level            = 'l1',
                     steps            = 2,
                     lags             = 3,
                     window_features  = rolling,
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(series=series_fixtures, exog=exog)

    series_names_in_ = ['l1', 'l2']
    exog_in_ = True
    exog_type_in_ = type(exog)
    exog_names_in_ = ['exog_1', 'exog_2']
    exog_dtypes_in_ = {'exog_1': exog['exog_1'].dtype, 'exog_2': exog['exog_2'].dtype}
    # All floats
    exog_dtypes_out_ = {'exog_1': exog['exog_1'].dtype, 'exog_2_a': exog['exog_1'].dtype, 'exog_2_b': exog['exog_1'].dtype}
    X_train_series_names_in_ = ['l1', 'l2']
    X_train_window_features_names_out_ = [
        'l1_roll_ratio_min_max_4', 'l1_roll_median_4',
        'l2_roll_ratio_min_max_4', 'l2_roll_median_4'
    ]
    X_train_exog_names_out_ = ['exog_1', 'exog_2_a', 'exog_2_b']
    X_train_direct_exog_names_out_ = [
        'exog_1_step_1', 'exog_2_a_step_1', 'exog_2_b_step_1',
        'exog_1_step_2', 'exog_2_a_step_2', 'exog_2_b_step_2'
    ]
    X_train_features_names_out_ = [
        'l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_roll_ratio_min_max_4', 'l1_roll_median_4',
        'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_roll_ratio_min_max_4', 'l2_roll_median_4',
        'exog_1_step_1', 'exog_2_a_step_1', 'exog_2_b_step_1', 
        'exog_1_step_2', 'exog_2_a_step_2', 'exog_2_b_step_2'
    ]
    
    assert forecaster.series_names_in_ == series_names_in_
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.exog_dtypes_out_ == exog_dtypes_out_
    assert forecaster.X_train_series_names_in_ == X_train_series_names_in_
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_exog_names_out_ == X_train_exog_names_out_
    assert forecaster.X_train_direct_exog_names_out_ == X_train_direct_exog_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_
    assert forecaster.categorical_features_names_in_ == []


def test_fit_correct_dict_create_transformer_series():
    """
    Test fit method creates correctly all the auxiliary dicts transformer_series_.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10)), 
                           'l3': pd.Series(np.arange(10))})

    transformer_series = {'l1': StandardScaler(), 'l3': StandardScaler(), 'l4': StandardScaler()}

    forecaster = ForecasterDirectMultiVariate(
                     estimator          = LinearRegression(), 
                     level              = 'l1',
                     lags               = 3,
                     steps              = 2,
                     transformer_series = transformer_series
                 )
    
    forecaster.fit(series=series, store_in_sample_residuals=False)
    expected_transformer_series_ = {'l1': forecaster.transformer_series_['l1'], 'l2': None, 'l3': forecaster.transformer_series_['l3']}

    assert forecaster.transformer_series_ == expected_transformer_series_

    forecaster.fit(series=series[['l1', 'l2']], store_in_sample_residuals=False)
    expected_transformer_series_ = {'l1': forecaster.transformer_series_['l1'], 'l2': None}

    assert forecaster.transformer_series_ == expected_transformer_series_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test series.index.freq is stored in forecaster.index_freq.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})

    series.index = pd.date_range(start='2022-01-01', periods=10, freq='1D')

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=1
    )
    forecaster.fit(series=series)
    expected = series.index.freq
    results = forecaster.index_freq_

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq_.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2
    )
    forecaster.fit(series=series)
    expected = series.index.step
    results = forecaster.index_freq_

    assert results == expected
    

@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_stored(n_jobs):
    """
    Test that values of in_sample_residuals_ are stored after fitting.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10))})
    
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2, n_jobs=n_jobs
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    expected = {
        "l1": np.array(
                [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 
                 0.0000000e+00, 0.0000000e+00, 8.8817842e-16, 0., 0., 0., 0., 0., 0.]
              )
    }
    results = forecaster.in_sample_residuals_

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
    series = pd.DataFrame({'l1': pd.Series(np.arange(12_000)), 
                           'l2': pd.Series(np.arange(12_000))})
    
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2, n_jobs=n_jobs
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_1 = forecaster.in_sample_residuals_

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2, n_jobs=n_jobs
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_2 = forecaster.in_sample_residuals_

    assert isinstance(results_1, dict)
    assert isinstance(results_2, dict)
    assert results_1.keys() == results_2.keys()
    for k in results_1.keys():
        assert isinstance(results_1[k], np.ndarray)
        assert isinstance(results_2[k], np.ndarray)
        assert len(results_1[k]) == 10_000
        assert len(results_2[k]) == 10_000
        np.testing.assert_array_equal(results_1[k], results_2[k])


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_by_bin_stored(n_jobs):
    """
    Test that values of in_sample_residuals_by_bin are stored after fitting.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2, binner_kwargs={'n_bins': 3}, n_jobs=n_jobs
    )
    forecaster.fit(series=series_fixtures, store_in_sample_residuals=True)

    expected_1 = {
        'l1' : np.array([
                    0.61351689,  0.71846659, -0.46065047,  1.8740517 ,  0.35951685,
                    0.36635782, -0.3225934 , -0.86465358,  0.82252715, -0.2741012 ,
                    -1.70423914, -0.33664349,  1.16033533, -1.66552242, -1.26639479,
                    0.18116919, -0.35795787,  1.02694562,  1.33431561,  0.98054222,
                    0.94382542,  0.57275958, -1.52361509, -0.76119268, -0.84770863,
                    -1.07296309,  0.20287787, -1.30813619,  0.3000927 , -0.59338942,
                    0.52690419, -0.5786443 , -0.43975004,  0.23252597,  1.25096008,
                    1.05277835, -0.17913321,  0.65860599, -1.73952616,  0.00926585,
                    -0.74486859,  1.56624963, -0.89792655,  0.1841916 ,  1.34397847,
                    -0.34315036, 0.89701176, -0.41773024,  1.82467126,  0.72112853,  0.40786125,
                    -0.61189293, -0.60896505,  0.65160718, -0.30338729, -1.93442134,
                    -0.79019702,  0.95739893, -1.24036132, -1.99937927,  0.22147022,
                    0.12771388,  0.72032178,  1.67692078,  0.86091935,  0.9791868 ,
                    0.59836304, -0.8500661 , -0.88860757, -0.87948968, -0.67394226,
                    0.29203755, -1.50705801,  0.1996901 , -0.79372163,  0.47123556,
                    -0.65428718, -0.94038195, -0.16313875,  1.38627477,  1.49653973,
                    -0.01688572,  0.7562465 , -1.50022778, -0.32925933, -0.60151835,
                    1.26646515, -1.08437562, -0.21536791,  1.62120186,  0.11599908,
                    0.75439722
                ])
    }

    expected_2 = {
        'l1': {
            0: np.array([0.61351689,  0.36635782, -0.3225934 , -1.70423914,  1.16033533,
                    -1.26639479,  1.02694562,  0.94382542, -0.84770863, -1.30813619,
                    0.3000927 ,  0.52690419, -0.43975004,  0.23252597,  0.00926585,
                    -0.89792655,  0.1841916 ,  0.40786125,  0.72032178,  1.67692078,
                    0.9791868 , -0.87948968, -0.67394226, -1.50705801,  0.1996901 ,
                    0.47123556, -0.16313875,  0.7562465 , -1.50022778, -0.32925933,
                    0.75439722]),
            1: np.array([-0.46065047,  0.82252715, -0.2741012 , -0.33664349,  0.18116919,
                    1.33431561,  0.98054222, -0.76119268, -1.07296309,  0.65860599,
                    -1.73952616,  1.56624963,  0.89701176, -0.41773024,  0.72112853,
                    -0.61189293, -0.60896505, -0.30338729, -1.93442134,  0.95739893,
                    -1.24036132,  0.22147022,  0.12771388,  0.86091935, -0.8500661 ,
                    -0.94038195, -0.01688572, -1.08437562, -0.21536791,  0.11599908]),
            2: np.array([ 0.71846659,  1.8740517 ,  0.35951685, -0.86465358, -1.66552242,
                    -0.35795787,  0.57275958, -1.52361509,  0.20287787, -0.59338942,
                    -0.5786443 ,  1.25096008,  1.05277835, -0.17913321, -0.74486859,
                    1.34397847, -0.34315036,  1.82467126,  0.65160718, -0.79019702,
                    -1.99937927,  0.59836304, -0.88860757,  0.29203755, -0.79372163,
                    -0.65428718,  1.38627477,  1.49653973, -0.60151835,  1.26646515,
                    1.62120186])
        }
    }

    expected_3 = {
        'l1': {
            0: (-0.8032116858231456, -0.13623943835630237),
            1: (-0.13623943835630237, 0.17856930169525217),
            2: (0.17856930169525217, 0.8498028419581619)
        }
    }

    np.testing.assert_array_almost_equal(forecaster.in_sample_residuals_['l1'], expected_1['l1'])
    for k in forecaster.in_sample_residuals_by_bin_['l1'].keys():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_by_bin_['l1'][k], expected_2['l1'][k]
        )
    for k in forecaster.binner_intervals_['l1'].keys():
        assert forecaster.binner_intervals_['l1'][k][0] == approx(expected_3['l1'][k][0])
        assert forecaster.binner_intervals_['l1'][k][1] == approx(expected_3['l1'][k][1])


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_not_stored_probabilistic_mode_binned(n_jobs):
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False`. Binner intervals are stored.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2, binner_kwargs={'n_bins': 3}, n_jobs=n_jobs
    )
    forecaster.fit(series=series_fixtures, store_in_sample_residuals=False)

    expected = {'l1': None}
    expected_binner_intervals_ = {
        'l1' :{
            0: (-0.8032116858231456, -0.13623943835630237),
            1: (-0.13623943835630237, 0.17856930169525217),
            2: (0.17856930169525217, 0.8498028419581619)
        }
    }

    assert isinstance(forecaster.in_sample_residuals_, dict)
    assert forecaster.in_sample_residuals_.keys() == expected.keys()
    for k in forecaster.in_sample_residuals_.keys():
        assert forecaster.in_sample_residuals_[k] == expected[k]
    
    assert forecaster.in_sample_residuals_by_bin_ == {'l1': None}

    assert forecaster.binner_intervals_.keys() == expected_binner_intervals_.keys()
    for k in forecaster.binner_intervals_['l1'].keys():
        assert forecaster.binner_intervals_['l1'][k][0] == approx(expected_binner_intervals_['l1'][k][0])
        assert forecaster.binner_intervals_['l1'][k][1] == approx(expected_binner_intervals_['l1'][k][1])


@pytest.mark.parametrize("n_jobs", [1, -1, 'auto'], 
                         ids=lambda n_jobs: f'n_jobs: {n_jobs}')
def test_fit_in_sample_residuals_not_stored_probabilistic_mode_False(n_jobs):
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False` and _probabilistic_mode=False.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(5)), 
                           'l2': pd.Series(np.arange(5))})
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2, binner_kwargs={'n_bins': 3}, n_jobs=n_jobs
    )
    forecaster._probabilistic_mode = False
    forecaster.fit(series=series, store_in_sample_residuals=False)
    
    expected = {'l1': None}

    assert isinstance(forecaster.in_sample_residuals_, dict)
    assert forecaster.in_sample_residuals_.keys() == expected.keys()
    for k in forecaster.in_sample_residuals_.keys():
        assert forecaster.in_sample_residuals_[k] == expected[k]
    assert forecaster.in_sample_residuals_by_bin_ == {'l1': None}
    assert forecaster.binner_intervals_ == {}


@pytest.mark.parametrize("store_last_window", 
                         [True, ['l1', 'l2'], False], 
                         ids=lambda lw: f'store_last_window: {lw}')
def test_fit_last_window_stored(store_last_window):
    """
    Test that values of last window are stored after fitting.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(50, 60))})

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2
    )
    forecaster.fit(series=series, store_last_window=store_last_window)

    expected = pd.DataFrame({
        'l1': pd.Series(np.array([7, 8, 9])), 
        'l2': pd.Series(np.array([57, 58, 59]))
    })
    expected.index = pd.RangeIndex(start=7, stop=10, step=1)

    if store_last_window:
        pd.testing.assert_frame_equal(forecaster.last_window_, expected)
        assert forecaster.series_names_in_ == ['l1', 'l2']
        assert forecaster.X_train_series_names_in_ == ['l1', 'l2']
    else:
        assert forecaster.last_window_ is None


def test_fit_last_window_stored_when_different_lags():
    """
    Test that values of last window are stored after fitting when different lags
    configurations.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))})

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l2', steps = 2, lags = {'l1': 3, 'l2': [1, 5]}
    )
    forecaster.fit(series=series)

    expected = pd.DataFrame({
        'l1': pd.Series(np.array([5, 6, 7, 8, 9])), 
        'l2': pd.Series(np.array([105, 106, 107, 108, 109]))
    })
    expected.index = pd.RangeIndex(start=5, stop=10, step=1)

    pd.testing.assert_frame_equal(forecaster.last_window_, expected)
    assert forecaster.series_names_in_ == ['l1', 'l2']
    assert forecaster.X_train_series_names_in_ == ['l1', 'l2']


@pytest.mark.parametrize("level",
                         ['l1', 'l2'],
                         ids=lambda level: f'level: {level}')
def test_fit_last_window_stored_when_lags_dict_with_None(level):
    """
    Test that values of last window are stored after fitting when lags is a dict
    with None values.    
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(100, 110))})

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level=level, steps = 2, lags = {'l1': 3, 'l2': None}
    )
    forecaster.fit(series=series)

    expected = pd.DataFrame({'l1': pd.Series(np.array([7, 8, 9]))})
    expected.index = pd.RangeIndex(start=7, stop=10, step=1)

    pd.testing.assert_frame_equal(forecaster.last_window_, expected)
    assert forecaster.series_names_in_ == ['l1', 'l2']
    assert forecaster.X_train_series_names_in_ == ['l1']


def test_fit_resets_out_sample_residuals_on_refit():
    """
    Test that out_sample_residuals_ and out_sample_residuals_by_bin_ are reset
    to None when the forecaster is refitted.
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2
    )
    forecaster.fit(series=series_fixtures)
    forecaster.set_out_sample_residuals(
        y_true={'l1': np.arange(1, 46, dtype=float)},
        y_pred={'l1': np.zeros(45)},
    )

    assert forecaster.out_sample_residuals_ is not None
    assert forecaster.out_sample_residuals_by_bin_ is not None

    forecaster.fit(series=series_fixtures)

    assert forecaster.out_sample_residuals_ is None
    assert forecaster.out_sample_residuals_by_bin_ is None


def custom_weights(index):  # pragma: no cover
    """
    Return 0 if index is between 20 and 40 else 1.
    """
    weights = np.where(
                (index >= 20) & (index <= 40),
                0,
                1
              )
    
    return weights


def test_fit_model_coef_when_using_weight_func():
    """
    Check the value of the estimator coefs when using a `weight_func`.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator   = LinearRegression(),
                     level       = 'l1',
                     lags        = 5,
                     steps       = 2,
                     weight_func = custom_weights
                 )
    forecaster.fit(series=series_fixtures)
    results_1 = forecaster.estimators_[1].coef_
    results_2 = forecaster.estimators_[2].coef_
    expected_1 = np.array([
        -0.07798749, -0.12725508,  0.35291714,  0.31257354, -0.17962821,
        -0.2411156 , -0.33348055,  0.39950299,  0.54651497, -0.01166156
    ])
    expected_2 = np.array([
        -0.02370398,  0.36524827,  0.02372546, -0.25968417, -0.49530388,
        -0.27695513,  0.35233664,  0.57568233, -0.09067347, -0.5731565
    ])

    np.testing.assert_almost_equal(results_1, expected_1)
    np.testing.assert_almost_equal(results_2, expected_2)


def test_fit_model_coef_when_not_using_weight_func():
    """
    Check the value of the estimator coefs when not using a `weight_func`.
    """
    forecaster = ForecasterDirectMultiVariate(
                     estimator = LinearRegression(),
                     level     = 'l1',
                     lags      = 5,
                     steps     = 2
                 )
    forecaster.fit(series=series_fixtures)
    results_1 = forecaster.estimators_[1].coef_
    results_2 = forecaster.estimators_[2].coef_
    expected_1 = np.array([
         0.09489197,  0.00139308,  0.17712709, -0.07425304, -0.30147832,
        -0.25874708, -0.03541285,  0.32235909,  0.06700057, -0.16040817
    ])
    expected_2 = np.array([
        -0.01427965,  0.19641192, -0.14719057, -0.19390091, -0.27612126,
        -0.07905377,  0.30408338,  0.18286733, -0.09371034, -0.13316052
    ])

    np.testing.assert_almost_equal(results_1, expected_1)
    np.testing.assert_almost_equal(results_2, expected_2)


# ==============================================================================
# Tests: fit with categorical features and configure_estimator_categorical_features
# ==============================================================================
@pytest.mark.parametrize(
    "estimator, check_fn",
    [
        (
            CatBoostRegressor(
                iterations=10, random_seed=123, verbose=0,
                allow_writing_files=False
            ),
            None
        ),
        (
            LGBMRegressor(verbose=-1, random_state=123),
            None
        ),
        (
            XGBRegressor(random_state=123),
            lambda est, cat_idx, n_features: (
                est.get_params()['enable_categorical'] is True
                and est.get_params()['feature_types'] == [
                    'c' if i in cat_idx else 'q' for i in range(n_features)
                ]
            )
        ),
        (
            HistGradientBoostingRegressor(random_state=123),
            lambda est, cat_idx, n_features: (
                est.get_params()['categorical_features'] == cat_idx
            )
        ),
    ],
    ids=['CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_fit_configures_estimator_categorical_features(estimator, check_fn):
    """
    Test that fit correctly configures native categorical feature support
    for each supported estimator. Verifies that the individual step
    estimators in estimators_ have the categorical params set.
    """
    series_cat = pd.DataFrame({
        'l1': np.arange(20, dtype=float),
        'l2': np.arange(100, 120, dtype=float)
    })
    exog_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float),
        'exog_cat': pd.Categorical(range(20))
    })

    forecaster = ForecasterDirectMultiVariate(
        estimator=estimator, level='l1', lags=3, steps=2,
        categorical_features='auto'
    )
    forecaster.fit(series=series_cat, exog=exog_cat)

    assert forecaster.is_fitted
    assert forecaster.categorical_features_names_in_ == ['exog_cat']
    assert 'exog_cat_step_1' in forecaster.X_train_features_names_out_
    assert 'exog_cat_step_2' in forecaster.X_train_features_names_out_

    if check_fn is not None:
        # Step-specific feature names (without _step_N suffix)
        from itertools import chain
        n_lags = len(list(
            chain(*[v for v in forecaster.lags_.values() if v is not None])
        ))
        n_wf = (
            len(forecaster.X_train_window_features_names_out_)
            if forecaster.window_features is not None else 0
        )
        step_features = forecaster.X_train_features_names_out_[:n_lags + n_wf]
        step_features = step_features + forecaster.X_train_exog_names_out_
        cat_idx = [step_features.index('exog_cat')]
        n_features = len(step_features)
        for step_est in forecaster.estimators_.values():
            assert check_fn(step_est, cat_idx, n_features)

    # fit_kwargs must not be mutated
    assert forecaster.fit_kwargs == {}


@pytest.mark.parametrize(
    "estimator, param_name, default_value",
    [
        (
            XGBRegressor(random_state=123),
            'feature_types',
            None
        ),
        (
            HistGradientBoostingRegressor(random_state=123),
            'categorical_features',
            'from_dtype'
        ),
    ],
    ids=['XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_fit_resets_estimator_categorical_params_on_refit_without_categoricals(
    estimator, param_name, default_value
):
    """
    Test that fitting with categorical features and then refitting without
    categoricals resets the estimator's categorical parameters to their
    default values (XGBoost: feature_types=None,
    HistGradientBoosting: categorical_features='from_dtype').
    """
    series_cat = pd.DataFrame({
        'l1': np.arange(20, dtype=float),
        'l2': np.arange(100, 120, dtype=float)
    })
    exog_with_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float),
        'exog_cat': pd.Categorical(['a', 'b'] * 10)
    })
    exog_no_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float)
    })

    forecaster = ForecasterDirectMultiVariate(
        estimator=estimator, level='l1', lags=3, steps=2,
        categorical_features='auto'
    )

    # First fit — with categoricals
    forecaster.fit(series=series_cat, exog=exog_with_cat)
    assert forecaster.categorical_features_names_in_ == ['exog_cat']

    # Second fit — without categoricals (auto detects no categories → [])
    forecaster.fit(series=series_cat, exog=exog_no_cat)
    assert forecaster.categorical_features_names_in_ == []
    assert forecaster.estimators_[1].get_params()[param_name] == default_value


@pytest.mark.parametrize(
    "estimator",
    [
        CatBoostRegressor(
            iterations=10, random_seed=123, verbose=0,
            allow_writing_files=False
        ),
        LGBMRegressor(verbose=-1, random_state=123),
        XGBRegressor(random_state=123),
        HistGradientBoostingRegressor(random_state=123),
    ],
    ids=['CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_fit_no_categoricals_with_supported_estimators(estimator):
    """
    Test that fit works correctly with supported estimators when
    categorical_features=None (no categorical encoding).
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=estimator, level='l1', lags=3, steps=2,
        categorical_features=None
    )
    forecaster.fit(series=series_fixtures, exog=exog[['exog_1']])

    assert forecaster.is_fitted
    assert forecaster.categorical_features_names_in_ is None
