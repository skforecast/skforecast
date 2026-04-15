# Unit test _train_test_split_one_step_ahead ForecasterDirect
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirect


def test_train_test_split_one_step_ahead_when_lags_and_exog():
    """
    Test _train_test_split_one_step_ahead returns correct numpy arrays,
    sample_weight=None and fit_kwargs={} when using lags and exog with
    no weight_func, no categorical_features, no transformers.
    """
    y = pd.Series(
        np.arange(15), index=pd.date_range("2020-01-01", periods=15), dtype=float
    )
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 115, dtype=float),
            "exog_2": np.arange(1000, 1015, dtype=float),
        },
        index=pd.date_range("2020-01-01", periods=15),
    )

    forecaster = ForecasterDirect(LinearRegression(), lags=5, steps=1)

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )

    expected_X_train = np.array([
        [4.0, 3.0, 2.0, 1.0, 0.0, 105.0, 1005.],
        [5.0, 4.0, 3.0, 2.0, 1.0, 106.0, 1006.],
        [6.0, 5.0, 4.0, 3.0, 2.0, 107.0, 1007.],
        [7.0, 6.0, 5.0, 4.0, 3.0, 108.0, 1008.],
        [8.0, 7.0, 6.0, 5.0, 4.0, 109.0, 1009.],
    ])
    expected_y_train = np.array([5.0, 6.0, 7.0, 8.0, 9.0])

    expected_X_test = np.array([
        [ 9.0,  8.0,  7.0,  6.0,  5.0, 110.0, 1010.],
        [10.0,  9.0,  8.0,  7.0,  6.0, 111.0, 1011.],
        [11.0, 10.0,  9.0,  8.0,  7.0, 112.0, 1012.],
        [12.0, 11.0, 10.0,  9.0,  8.0, 113.0, 1013.],
        [13.0, 12.0, 11.0, 10.0,  9.0, 114.0, 1014.],
    ])
    expected_y_test = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_no_exog():
    """
    Test _train_test_split_one_step_ahead when exog is None. Verify that
    X_train and X_test contain only lag columns.
    """
    y = pd.Series(
        np.arange(15), index=pd.date_range("2020-01-01", periods=15), dtype=float
    )

    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=1)

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, initial_train_size=10
    )

    expected_X_train = np.array([
        [2.0, 1.0, 0.0],
        [3.0, 2.0, 1.0],
        [4.0, 3.0, 2.0],
        [5.0, 4.0, 3.0],
        [6.0, 5.0, 4.0],
        [7.0, 6.0, 5.0],
        [8.0, 7.0, 6.0],
    ])
    expected_y_train = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    expected_X_test = np.array([
        [ 9.0,  8.0,  7.0],
        [10.0,  9.0,  8.0],
        [11.0, 10.0,  9.0],
        [12.0, 11.0, 10.0],
        [13.0, 12.0, 11.0],
    ])
    expected_y_test = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_weight_func():
    """
    Test _train_test_split_one_step_ahead precomputes sample_weight when
    weight_func is provided. Verifies mixed weights are correctly mapped
    per observation.
    """
    y = pd.Series(
        np.arange(15), index=pd.date_range("2020-01-01", periods=15), dtype=float
    )

    def custom_weights(index):
        return np.where(index >= pd.Timestamp("2020-01-07"), 2.0, 1.0)

    forecaster = ForecasterDirect(
        LinearRegression(), lags=3, steps=1, weight_func=custom_weights
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, initial_train_size=10
    )

    expected_X_train = np.array([
        [2., 1., 0.],
        [3., 2., 1.],
        [4., 3., 2.],
        [5., 4., 3.],
        [6., 5., 4.],
        [7., 6., 5.],
        [8., 7., 6.],
    ])
    expected_y_train = np.array([3., 4., 5., 6., 7., 8., 9.])
    expected_X_test = np.array([
        [ 9.,  8.,  7.],
        [10.,  9.,  8.],
        [11., 10.,  9.],
        [12., 11., 10.],
        [13., 12., 11.],
    ])
    expected_y_test = np.array([10., 11., 12., 13., 14.])
    expected_sample_weight = np.array([1., 1., 1., 2., 2., 2., 2.])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    np.testing.assert_array_equal(sample_weight, expected_sample_weight)
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_window_features_and_transformers_and_differentiation():
    """
    Test _train_test_split_one_step_ahead with window_features, transformer_y,
    transformer_exog, and differentiation. Verify shapes and return types.
    """
    y = pd.Series(
        np.arange(20), index=pd.date_range("2020-01-01", periods=20), dtype=float
    )
    exog = pd.DataFrame(
        {"exog_1": np.arange(100, 120, dtype=float)},
        index=pd.date_range("2020-01-01", periods=20),
    )
    rolling = RollingFeatures(stats=['mean'], window_sizes=3)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(),
        lags=3,
        steps=1,
        window_features=rolling,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
        differentiation=1,
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    expected_X_train = np.array([
        [0.23145502, 0.23145502, 0.23145502, 0.23145502, -0.69436507],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502, -0.46291005],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502, -0.23145502],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502,  0.        ],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502,  0.23145502],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502,  0.46291005],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502,  0.69436507],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502,  0.9258201 ],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502,  1.15727512],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502,  1.38873015],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502,  1.62018517],
    ])
    expected_y_train = np.array([
        0.23145502, 0.23145502, 0.23145502, 0.23145502, 0.23145502,
        0.23145502, 0.23145502, 0.23145502, 0.23145502, 0.23145502,
        0.23145502,
    ])
    expected_X_test = np.array([
        [0.23145502, 0.23145502, 0.23145502, 0.23145502, 1.8516402 ],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502, 2.08309522],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502, 2.31455025],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502, 2.54600527],
        [0.23145502, 0.23145502, 0.23145502, 0.23145502, 2.7774603 ],
    ])
    expected_y_test = np.array([
        0.23145502, 0.23145502, 0.23145502, 0.23145502, 0.23145502,
    ])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_steps_greater_than_1():
    """
    Test _train_test_split_one_step_ahead with steps=3 filters to step 1,
    removing step-2 and step-3 exog columns while reducing row count
    accordingly.
    """
    y = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 120, dtype=float),
            "exog_2": np.arange(1000, 1020, dtype=float),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirect(LinearRegression(), lags=5, steps=3)

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    expected_X_train = np.array([
        [  4.,   3.,   2.,   1.,   0.,  105., 1005.],
        [  5.,   4.,   3.,   2.,   1.,  106., 1006.],
        [  6.,   5.,   4.,   3.,   2.,  107., 1007.],
        [  7.,   6.,   5.,   4.,   3.,  108., 1008.],
        [  8.,   7.,   6.,   5.,   4.,  109., 1009.],
        [  9.,   8.,   7.,   6.,   5.,  110., 1010.],
        [ 10.,   9.,   8.,   7.,   6.,  111., 1011.],
        [ 11.,  10.,   9.,   8.,   7.,  112., 1012.],
    ])
    expected_y_train = np.array([5., 6., 7., 8., 9., 10., 11., 12.])
    expected_X_test = np.array([
        [ 14.,  13.,  12.,  11.,  10.,  115., 1015.],
        [ 15.,  14.,  13.,  12.,  11.,  116., 1016.],
        [ 16.,  15.,  14.,  13.,  12.,  117., 1017.],
    ])
    expected_y_test = np.array([15., 16., 17.])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_lightgbm_categorical():
    """
    Test _train_test_split_one_step_ahead with LGBMRegressor and categorical
    features. Verify fit_kwargs contains 'categorical_feature' with correct
    indices.
    """
    y = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirect(
        estimator=LGBMRegressor(verbose=-1, random_state=123),
        lags=3,
        steps=1,
        categorical_features='auto',
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    expected_X_train = np.array([
        [  2.,   1.,   0., 103.,   3.],
        [  3.,   2.,   1., 104.,   4.],
        [  4.,   3.,   2., 105.,   5.],
        [  5.,   4.,   3., 106.,   6.],
        [  6.,   5.,   4., 107.,   7.],
        [  7.,   6.,   5., 108.,   8.],
        [  8.,   7.,   6., 109.,   9.],
        [  9.,   8.,   7., 110.,  10.],
        [ 10.,   9.,   8., 111.,  11.],
        [ 11.,  10.,   9., 112.,  12.],
        [ 12.,  11.,  10., 113.,  13.],
        [ 13.,  12.,  11., 114.,  14.],
    ])
    expected_y_train = np.array([3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.])
    expected_X_test = np.array([
        [ 14.,  13.,  12., 115.,  np.nan],
        [ 15.,  14.,  13., 116.,  np.nan],
        [ 16.,  15.,  14., 117.,  np.nan],
        [ 17.,  16.,  15., 118.,  np.nan],
        [ 18.,  17.,  16., 119.,  np.nan],
    ])
    expected_y_test = np.array([15., 16., 17., 18., 19.])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert 'categorical_feature' in fit_kwargs
    assert fit_kwargs['categorical_feature'] == [4]
    assert X_train.dtype != object
    assert 'categorical_feature' not in forecaster.fit_kwargs


def test_train_test_split_one_step_ahead_when_xgboost_categorical():
    """
    Test _train_test_split_one_step_ahead with XGBRegressor and categorical
    features. Verify estimator is configured with feature_types and
    enable_categorical via set_params.
    """
    y = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    estimator = XGBRegressor(random_state=123, enable_categorical=False)
    forecaster = ForecasterDirect(
        estimator=estimator,
        lags=3,
        steps=1,
        categorical_features='auto',
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    expected_X_train = np.array([
        [  2.,   1.,   0., 103.,   3.],
        [  3.,   2.,   1., 104.,   4.],
        [  4.,   3.,   2., 105.,   5.],
        [  5.,   4.,   3., 106.,   6.],
        [  6.,   5.,   4., 107.,   7.],
        [  7.,   6.,   5., 108.,   8.],
        [  8.,   7.,   6., 109.,   9.],
        [  9.,   8.,   7., 110.,  10.],
        [ 10.,   9.,   8., 111.,  11.],
        [ 11.,  10.,   9., 112.,  12.],
        [ 12.,  11.,  10., 113.,  13.],
        [ 13.,  12.,  11., 114.,  14.],
    ])
    expected_y_train = np.array([3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.])
    expected_X_test = np.array([
        [ 14.,  13.,  12., 115.,  np.nan],
        [ 15.,  14.,  13., 116.,  np.nan],
        [ 16.,  15.,  14., 117.,  np.nan],
        [ 17.,  16.,  15., 118.,  np.nan],
        [ 18.,  17.,  16., 119.,  np.nan],
    ])
    expected_y_test = np.array([15., 16., 17., 18., 19.])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert 'categorical_feature' not in fit_kwargs
    assert 'cat_features' not in fit_kwargs
    params = forecaster.estimators_[1].get_params()
    assert params['enable_categorical'] is True
    assert params['feature_types'] == ['q', 'q', 'q', 'q', 'c']


def test_train_test_split_one_step_ahead_when_histgradientboosting_categorical():
    """
    Test _train_test_split_one_step_ahead with HistGradientBoostingRegressor
    and categorical features. Verify estimator is configured with
    categorical_features via set_params.
    """
    y = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirect(
        estimator=HistGradientBoostingRegressor(random_state=123),
        lags=3,
        steps=1,
        categorical_features='auto',
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    expected_X_train = np.array([
        [  2.,   1.,   0., 103.,   3.],
        [  3.,   2.,   1., 104.,   4.],
        [  4.,   3.,   2., 105.,   5.],
        [  5.,   4.,   3., 106.,   6.],
        [  6.,   5.,   4., 107.,   7.],
        [  7.,   6.,   5., 108.,   8.],
        [  8.,   7.,   6., 109.,   9.],
        [  9.,   8.,   7., 110.,  10.],
        [ 10.,   9.,   8., 111.,  11.],
        [ 11.,  10.,   9., 112.,  12.],
        [ 12.,  11.,  10., 113.,  13.],
        [ 13.,  12.,  11., 114.,  14.],
    ])
    expected_y_train = np.array([3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.])
    expected_X_test = np.array([
        [ 14.,  13.,  12., 115.,  np.nan],
        [ 15.,  14.,  13., 116.,  np.nan],
        [ 16.,  15.,  14., 117.,  np.nan],
        [ 17.,  16.,  15., 118.,  np.nan],
        [ 18.,  17.,  16., 119.,  np.nan],
    ])
    expected_y_test = np.array([15., 16., 17., 18., 19.])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert 'categorical_feature' not in fit_kwargs
    assert 'cat_features' not in fit_kwargs
    params = forecaster.estimators_[1].get_params()
    assert params['categorical_features'] == [4]


def test_train_test_split_one_step_ahead_when_catboost_categorical():
    """
    Test _train_test_split_one_step_ahead with CatBoostRegressor and
    categorical features. Verify fit_kwargs contains 'cat_features' with
    correct indices and that categorical columns in X_train/X_test are
    cast to int (object dtype).
    """
    y = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical([0, 1, 2] * 6 + [0, 1]),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirect(
        estimator=CatBoostRegressor(
            iterations=10, random_seed=123, verbose=0,
            allow_writing_files=False
        ),
        lags=3,
        steps=1,
        categorical_features='auto',
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    expected_X_train = np.array([
        [2.0, 1.0, 0.0, 103.0, 0],
        [3.0, 2.0, 1.0, 104.0, 1],
        [4.0, 3.0, 2.0, 105.0, 2],
        [5.0, 4.0, 3.0, 106.0, 0],
        [6.0, 5.0, 4.0, 107.0, 1],
        [7.0, 6.0, 5.0, 108.0, 2],
        [8.0, 7.0, 6.0, 109.0, 0],
        [9.0, 8.0, 7.0, 110.0, 1],
        [10.0, 9.0, 8.0, 111.0, 2],
        [11.0, 10.0, 9.0, 112.0, 0],
        [12.0, 11.0, 10.0, 113.0, 1],
        [13.0, 12.0, 11.0, 114.0, 2],
    ], dtype=object)
    expected_y_train = np.array([3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.])
    expected_X_test = np.array([
        [14.0, 13.0, 12.0, 115.0, 0],
        [15.0, 14.0, 13.0, 116.0, 1],
        [16.0, 15.0, 14.0, 117.0, 2],
        [17.0, 16.0, 15.0, 118.0, 0],
        [18.0, 17.0, 16.0, 119.0, 1],
    ], dtype=object)
    expected_y_test = np.array([15., 16., 17., 18., 19.])

    np.testing.assert_array_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert 'cat_features' in fit_kwargs
    assert fit_kwargs['cat_features'] == [4]
    assert X_train.dtype == object
    assert X_test.dtype == object


def test_train_test_split_one_step_ahead_when_fit_kwargs_no_categorical():
    """
    Test _train_test_split_one_step_ahead passes user-defined fit_kwargs
    through the else branch (categorical_features=None). Verify that
    fit_kwargs is a copy of forecaster.fit_kwargs and the original is
    not mutated.
    """
    y = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical([0, 1, 2] * 6 + [0, 1]),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirect(
        estimator=LGBMRegressor(verbose=-1, random_state=123),
        lags=3,
        steps=1,
        categorical_features=None,
        fit_kwargs={'categorical_feature': 'auto'},
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    expected_X_train = np.array([
        [  2.,   1.,   0., 103.,   0.],
        [  3.,   2.,   1., 104.,   1.],
        [  4.,   3.,   2., 105.,   2.],
        [  5.,   4.,   3., 106.,   0.],
        [  6.,   5.,   4., 107.,   1.],
        [  7.,   6.,   5., 108.,   2.],
        [  8.,   7.,   6., 109.,   0.],
        [  9.,   8.,   7., 110.,   1.],
        [ 10.,   9.,   8., 111.,   2.],
        [ 11.,  10.,   9., 112.,   0.],
        [ 12.,  11.,  10., 113.,   1.],
        [ 13.,  12.,  11., 114.,   2.],
    ])
    expected_y_train = np.array([3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.])
    expected_X_test = np.array([
        [ 14.,  13.,  12., 115.,   0.],
        [ 15.,  14.,  13., 116.,   1.],
        [ 16.,  15.,  14., 117.,   2.],
        [ 17.,  16.,  15., 118.,   0.],
        [ 18.,  17.,  16., 119.,   1.],
    ])
    expected_y_test = np.array([15., 16., 17., 18., 19.])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert fit_kwargs == {'categorical_feature': 'auto'}
    assert fit_kwargs is not forecaster.fit_kwargs


def test_train_test_split_one_step_ahead_when_non_contiguous_lags():
    """
    Test _train_test_split_one_step_ahead with non-contiguous lags (gaps).
    Verify that window_size is driven by the maximum lag and that shapes
    and feature values are correct.
    """
    y = pd.Series(
        np.arange(20, dtype=float),
        index=pd.date_range("2020-01-01", periods=20),
    )
    exog = pd.DataFrame(
        {"exog_1": np.arange(100, 120, dtype=float)},
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirect(LinearRegression(), lags=[1, 7], steps=1)

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    expected_X_train = np.array([
        [  6.,   0., 107.],
        [  7.,   1., 108.],
        [  8.,   2., 109.],
        [  9.,   3., 110.],
        [ 10.,   4., 111.],
        [ 11.,   5., 112.],
        [ 12.,   6., 113.],
        [ 13.,   7., 114.],
    ])
    expected_y_train = np.array([7., 8., 9., 10., 11., 12., 13., 14.])
    expected_X_test = np.array([
        [ 14.,   8., 115.],
        [ 15.,   9., 116.],
        [ 16.,  10., 117.],
        [ 17.,  11., 118.],
        [ 18.,  12., 119.],
    ])
    expected_y_test = np.array([15., 16., 17., 18., 19.])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert sample_weight is None
    assert fit_kwargs == {}


@pytest.mark.parametrize(
    'estimator, categorical_features, expected_fit_kwarg',
    [
        (
            LGBMRegressor(verbose=-1, random_state=123, n_estimators=10),
            'auto',
            'categorical_feature',
        ),
        (
            CatBoostRegressor(
                iterations=10, random_seed=123, verbose=0,
                allow_writing_files=False
            ),
            'auto',
            'cat_features',
        ),
        (
            XGBRegressor(random_state=123, n_estimators=10),
            'auto',
            None,
        ),
        (
            HistGradientBoostingRegressor(random_state=123, max_iter=10),
            'auto',
            None,
        ),
    ],
    ids=['LightGBM', 'CatBoost', 'XGBoost', 'HistGradientBoosting']
)
def test_train_test_split_one_step_ahead_fit_predict_with_categorical(
    estimator, categorical_features, expected_fit_kwarg
):
    """
    Integration test: verify that the output of _train_test_split_one_step_ahead
    can be used directly to fit and predict with each estimator that supports
    native categorical features. This catches mismatches between categorical
    indices and actual data layout.
    """
    y = pd.Series(
        np.arange(50, dtype=float),
        index=pd.date_range("2020-01-01", periods=50),
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 150, dtype=float),
            "exog_cat": pd.Categorical([0, 1, 2] * 16 + [0, 1]),
        },
        index=pd.date_range("2020-01-01", periods=50),
    )

    forecaster = ForecasterDirect(
        estimator=estimator,
        lags=3,
        steps=1,
        categorical_features=categorical_features,
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=40
    )

    # Fit must succeed with the provided data and kwargs
    forecaster.estimators_[1].fit(X_train, y_train, **fit_kwargs)

    # Predict must succeed and return correct shape
    predictions = forecaster.estimators_[1].predict(X_test)
    assert predictions.shape == y_test.shape

    # Verify categorical config was applied
    if expected_fit_kwarg is not None:
        assert expected_fit_kwarg in fit_kwargs
    else:
        params = forecaster.estimators_[1].get_params()
        est_name = type(forecaster.estimators_[1]).__name__
        if 'XGB' in est_name:
            assert params['enable_categorical'] is True
            assert 'c' in params['feature_types']
        elif 'HistGradient' in est_name:
            assert params['categorical_features'] == [4]


@pytest.mark.parametrize(
    'initial_is_fitted',
    [True, False],
    ids=lambda v: f'is_fitted={v}'
)
def test_train_test_split_one_step_ahead_restores_is_fitted(initial_is_fitted):
    """
    Test _train_test_split_one_step_ahead preserves the original is_fitted
    state of the forecaster after execution.
    """
    y = pd.Series(
        np.arange(15), index=pd.date_range("2020-01-01", periods=15), dtype=float
    )

    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=1)
    forecaster.is_fitted = initial_is_fitted

    forecaster._train_test_split_one_step_ahead(
        y=y, initial_train_size=10
    )

    assert forecaster.is_fitted == initial_is_fitted


def test_train_test_split_one_step_ahead_NaN_filtered_when_dropna_from_series_True():
    """
    Test _train_test_split_one_step_ahead filters NaN from both train and
    test sets when dropna_from_series is True.
    """
    y = pd.Series(
        [0, 1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, np.nan, 13, 14],
        index=pd.date_range('2020-01-01', periods=15, freq='D'),
        dtype=float,
    )

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2,
        dropna_from_series=True,
    )
    forecaster.fit(
        y=pd.Series(
            np.arange(20, dtype=float),
            index=pd.date_range('2020-01-01', periods=20, freq='D'),
        )
    )

    X_train, y_train, X_test, y_test, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            y=y, initial_train_size=10
        )
    )

    expected_X_train = np.array([
        [6., 5., 4.],
        [7., 6., 5.],
    ])
    expected_y_train = np.array([7., 8.])
    expected_X_test = np.array([
        [ 9.,  8.,  7.],
        [10.,  9.,  8.],
    ])
    expected_y_test = np.array([10., 11.])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert sample_weight is None
    assert fit_kwargs == {}
