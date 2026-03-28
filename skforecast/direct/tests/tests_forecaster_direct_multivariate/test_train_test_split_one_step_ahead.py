# Unit test _train_test_split_one_step_ahead ForecasterDirectMultiVariate
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
from skforecast.direct import ForecasterDirectMultiVariate


def test_train_test_split_one_step_ahead_when_series_and_exog_are_dataframe():
    """
    Test the output of _train_test_split_one_step_ahead when series and exog are
    pandas dataframes. Returns numpy arrays and pandas Index objects.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(15),
            "series_2": np.arange(50, 65),
        },
        index=pd.date_range("2020-01-01", periods=15),
        dtype=float,
    )
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 115, dtype=float),
            "exog_2": np.arange(1000, 1015, dtype=float),
        },
        index=pd.date_range("2020-01-01", periods=15),
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), lags=5, level="series_1", steps=1,
        transformer_series=None
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=10
        )
    )

    expected_X_train = np.array([
        [4.0, 3.0, 2.0, 1.0, 0.0, 54.0, 53.0, 52.0, 51.0, 50.0, 105.0, 1005.0],
        [5.0, 4.0, 3.0, 2.0, 1.0, 55.0, 54.0, 53.0, 52.0, 51.0, 106.0, 1006.0],
        [6.0, 5.0, 4.0, 3.0, 2.0, 56.0, 55.0, 54.0, 53.0, 52.0, 107.0, 1007.0],
        [7.0, 6.0, 5.0, 4.0, 3.0, 57.0, 56.0, 55.0, 54.0, 53.0, 108.0, 1008.0],
        [8.0, 7.0, 6.0, 5.0, 4.0, 58.0, 57.0, 56.0, 55.0, 54.0, 109.0, 1009.0],
    ])

    expected_y_train = np.array([5.0, 6.0, 7.0, 8.0, 9.0])

    expected_X_test = np.array([
        [ 9.0,  8.0,  7.0,  6.0,  5.0, 59.0, 58.0, 57.0, 56.0, 55.0, 110.0, 1010.0],
        [10.0,  9.0,  8.0,  7.0,  6.0, 60.0, 59.0, 58.0, 57.0, 56.0, 111.0, 1011.0],
        [11.0, 10.0,  9.0,  8.0,  7.0, 61.0, 60.0, 59.0, 58.0, 57.0, 112.0, 1012.0],
        [12.0, 11.0, 10.0,  9.0,  8.0, 62.0, 61.0, 60.0, 59.0, 58.0, 113.0, 1013.0],
        [13.0, 12.0, 11.0, 10.0,  9.0, 63.0, 62.0, 61.0, 60.0, 59.0, 114.0, 1014.0],
    ])

    expected_y_test = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    expected_train_index = pd.DatetimeIndex(
        ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"],
        freq="D",
    )

    expected_test_index = pd.DatetimeIndex(
        ["2020-01-11", "2020-01-12", "2020-01-13", "2020-01-14", "2020-01-15"],
        freq="D",
    )

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    pd.testing.assert_index_equal(train_index, expected_train_index)
    pd.testing.assert_index_equal(test_index, expected_test_index)
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_no_exog():
    """
    Test _train_test_split_one_step_ahead when exog is None. Verify that
    X_train and X_test contain only lag columns from all series.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(15),
            "series_2": np.arange(50, 65),
        },
        index=pd.date_range("2020-01-01", periods=15),
        dtype=float,
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), lags=3, level="series_1", steps=1,
        transformer_series=None
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, initial_train_size=10
        )
    )

    # 3 lags × 2 series = 6 columns, no exog
    expected_X_train = np.array([
        [2.0, 1.0, 0.0, 52.0, 51.0, 50.0],
        [3.0, 2.0, 1.0, 53.0, 52.0, 51.0],
        [4.0, 3.0, 2.0, 54.0, 53.0, 52.0],
        [5.0, 4.0, 3.0, 55.0, 54.0, 53.0],
        [6.0, 5.0, 4.0, 56.0, 55.0, 54.0],
        [7.0, 6.0, 5.0, 57.0, 56.0, 55.0],
        [8.0, 7.0, 6.0, 58.0, 57.0, 56.0],
    ])
    expected_y_train = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    expected_X_test = np.array([
        [ 9.0,  8.0,  7.0, 59.0, 58.0, 57.0],
        [10.0,  9.0,  8.0, 60.0, 59.0, 58.0],
        [11.0, 10.0,  9.0, 61.0, 60.0, 59.0],
        [12.0, 11.0, 10.0, 62.0, 61.0, 60.0],
        [13.0, 12.0, 11.0, 63.0, 62.0, 61.0],
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
    series = pd.DataFrame(
        {
            "series_1": np.arange(15),
            "series_2": np.arange(50, 65),
        },
        index=pd.date_range("2020-01-01", periods=15),
        dtype=float,
    )

    def custom_weights(index):
        return np.where(index >= pd.Timestamp("2020-01-07"), 2.0, 1.0)

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), lags=3, level="series_1", steps=1,
        transformer_series=None, weight_func=custom_weights
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, initial_train_size=10
        )
    )

    assert sample_weight is not None
    assert isinstance(sample_weight, np.ndarray)
    assert len(sample_weight) == len(y_train)
    # train_index: 2020-01-04..2020-01-10 (7 obs, lags=3 so first at y[3])
    # weights: 2020-01-04,05,06 -> 1.0; 2020-01-07,08,09,10 -> 2.0
    expected_weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(sample_weight, expected_weights)


def test_train_test_split_one_step_ahead_when_window_features_and_transformers_and_differentiation():
    """
    Test _train_test_split_one_step_ahead with window_features,
    transformer_series, transformer_exog, and differentiation. Verify
    shapes and return types.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    )
    exog = pd.DataFrame(
        {"exog_1": np.arange(100, 120, dtype=float)},
        index=pd.date_range("2020-01-01", periods=20),
    )
    rolling = RollingFeatures(stats=['mean'], window_sizes=3)

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        lags=3,
        level="series_1",
        steps=1,
        window_features=rolling,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
        differentiation=1,
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=15
        )
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(train_index, pd.DatetimeIndex)
    assert isinstance(test_index, pd.DatetimeIndex)
    # 3 lags × 2 series + 1 window_feature × 2 series + 1 exog = 9 columns
    assert X_train.shape[1] == 9
    assert X_test.shape[1] == 9
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_steps_greater_than_1():
    """
    Test _train_test_split_one_step_ahead with steps=3 filters to step 1,
    removing step-2 and step-3 exog columns while reducing row count
    accordingly.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    )
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 120, dtype=float),
            "exog_2": np.arange(1000, 1020, dtype=float),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), lags=5, level="series_1", steps=3,
        transformer_series=None
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=15
        )
    )

    # Filtered to step 1: 5 lags × 2 series + 2 exog (step 1 only) = 12 columns
    # Train: 15 - 5(lags) - 2(steps-1) = 8 rows
    # Test: test_init = 15-5 = 10, series[10:20] = 10 obs, 10-5-2 = 3 rows
    assert X_train.shape == (8, 12)
    assert X_test.shape == (3, 12)
    assert len(y_train) == 8
    assert len(y_test) == 3
    # First train row: series_1 lags=[4,3,2,1,0], series_2 lags=[54,53,52,51,50],
    #                  exog_step1=[105,1005], y=5
    np.testing.assert_array_almost_equal(
        X_train[0],
        [4.0, 3.0, 2.0, 1.0, 0.0, 54.0, 53.0, 52.0, 51.0, 50.0, 105.0, 1005.0]
    )
    np.testing.assert_array_almost_equal(y_train[0], 5.0)
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_lightgbm_categorical():
    """
    Test _train_test_split_one_step_ahead with LGBMRegressor and categorical
    features. Verify fit_kwargs contains 'categorical_feature' with correct
    indices.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LGBMRegressor(verbose=-1, random_state=123),
        lags=3,
        level="series_1",
        steps=1,
        transformer_series=None,
        categorical_features='auto',
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=15
        )
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    # Features: 3 lags × 2 series + exog_num + exog_cat = 8 columns
    # exog_cat is the last column -> cat at index 7
    assert 'categorical_feature' in fit_kwargs
    assert fit_kwargs['categorical_feature'] == [7]
    # Data should remain float (LightGBM handles categoricals natively)
    assert X_train.dtype != object
    # The original forecaster.fit_kwargs should not be mutated
    assert 'categorical_feature' not in forecaster.fit_kwargs


def test_train_test_split_one_step_ahead_when_xgboost_categorical():
    """
    Test _train_test_split_one_step_ahead with XGBRegressor and categorical
    features. Verify estimator is configured with feature_types and
    enable_categorical via set_params.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    estimator = XGBRegressor(random_state=123, enable_categorical=False)
    forecaster = ForecasterDirectMultiVariate(
        estimator=estimator,
        lags=3,
        level="series_1",
        steps=1,
        transformer_series=None,
        categorical_features='auto',
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=15
        )
    )

    # XGBoost uses set_params in-place, not fit_kwargs
    assert 'categorical_feature' not in fit_kwargs
    assert 'cat_features' not in fit_kwargs
    # estimators_[1] should be configured (not self.estimator)
    params = forecaster.estimators_[1].get_params()
    assert params['enable_categorical'] is True
    # Features: 3 lags × 2 series + exog_num + exog_cat = 8 columns
    # exog_cat is the last column -> cat at index 7
    assert params['feature_types'] == ['q', 'q', 'q', 'q', 'q', 'q', 'q', 'c']


def test_train_test_split_one_step_ahead_when_histgradientboosting_categorical():
    """
    Test _train_test_split_one_step_ahead with HistGradientBoostingRegressor
    and categorical features. Verify estimator is configured with
    categorical_features via set_params.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical(range(20)),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=HistGradientBoostingRegressor(random_state=123),
        lags=3,
        level="series_1",
        steps=1,
        transformer_series=None,
        categorical_features='auto',
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=15
        )
    )

    # HistGradientBoosting uses set_params in-place, not fit_kwargs
    assert 'categorical_feature' not in fit_kwargs
    assert 'cat_features' not in fit_kwargs
    # estimators_[1] should be configured
    params = forecaster.estimators_[1].get_params()
    # Features: 3 lags × 2 series + exog_num + exog_cat = 8 columns
    # exog_cat is the last column -> cat at index 7
    assert params['categorical_features'] == [7]


def test_train_test_split_one_step_ahead_when_catboost_categorical():
    """
    Test _train_test_split_one_step_ahead with CatBoostRegressor and
    categorical features. Verify fit_kwargs contains 'cat_features' with
    correct indices and that categorical columns in X_train/X_test are
    cast to int (object dtype).
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical([0, 1, 2] * 6 + [0, 1]),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=CatBoostRegressor(
            iterations=10, random_seed=123, verbose=0,
            allow_writing_files=False
        ),
        lags=3,
        level="series_1",
        steps=1,
        transformer_series=None,
        categorical_features='auto',
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=15
        )
    )

    # Features: 3 lags × 2 series + exog_num + exog_cat = 8 columns
    # exog_cat is the last column -> cat at index 7
    assert 'cat_features' in fit_kwargs
    assert fit_kwargs['cat_features'] == [7]
    # CatBoost requires object dtype with int-cast categorical columns
    assert X_train.dtype == object
    assert X_test.dtype == object
    # Categorical column (index 7) should contain int values
    assert all(isinstance(v, (int, np.integer)) for v in X_train[:, 7])
    assert all(isinstance(v, (int, np.integer)) for v in X_test[:, 7])
    # Non-categorical columns should still be float
    assert all(isinstance(v, (float, np.floating)) for v in X_train[:, 0])


def test_train_test_split_one_step_ahead_when_fit_kwargs_no_categorical():
    """
    Test _train_test_split_one_step_ahead passes user-defined fit_kwargs
    through the else branch (categorical_features=None). Verify that
    fit_kwargs is a copy of forecaster.fit_kwargs and the original is
    not mutated.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 120, dtype=float),
            "exog_cat": pd.Categorical([0, 1, 2] * 6 + [0, 1]),
        },
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LGBMRegressor(verbose=-1, random_state=123),
        lags=3,
        level="series_1",
        steps=1,
        transformer_series=None,
        categorical_features=None,
        fit_kwargs={'categorical_feature': 'auto'},
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=15
        )
    )

    # User fit_kwargs passed through the else branch (no categorical config)
    assert fit_kwargs == {'categorical_feature': 'auto'}
    # fit_kwargs is a copy, not the same object
    assert fit_kwargs is not forecaster.fit_kwargs


def test_train_test_split_one_step_ahead_when_non_contiguous_lags():
    """
    Test _train_test_split_one_step_ahead with non-contiguous lags (gaps).
    Verify that window_size is driven by the maximum lag and that shapes
    and feature values are correct.
    """
    series = pd.DataFrame(
        {
            "series_1": np.arange(20),
            "series_2": np.arange(50, 70),
        },
        index=pd.date_range("2020-01-01", periods=20),
        dtype=float,
    )
    exog = pd.DataFrame(
        {"exog_1": np.arange(100, 120, dtype=float)},
        index=pd.date_range("2020-01-01", periods=20),
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), lags=[1, 7], level="series_1", steps=1,
        transformer_series=None
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=15
        )
    )

    # window_size = max_lag = 7
    # train rows: 15 - 7 = 8
    # test_init = 15 - 7 = 8, series[8:] = 12 obs -> 12 - 7 = 5 rows
    # columns: 2 lags × 2 series + exog_1 = 5
    assert X_train.shape == (8, 5)
    assert X_test.shape == (5, 5)
    assert len(y_train) == 8
    assert len(y_test) == 5
    # First training row: y[7]=7 is target
    # series_1: lag_1=y[6]=6, lag_7=y[0]=0
    # series_2: lag_1=s2[6]=56, lag_7=s2[0]=50
    # exog=107
    np.testing.assert_array_almost_equal(
        X_train[0], [6.0, 0.0, 56.0, 50.0, 107.0]
    )
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
    series = pd.DataFrame(
        {
            "series_1": np.arange(50),
            "series_2": np.arange(50, 100),
        },
        index=pd.date_range("2020-01-01", periods=50),
        dtype=float,
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 150, dtype=float),
            "exog_cat": pd.Categorical([0, 1, 2] * 16 + [0, 1]),
        },
        index=pd.date_range("2020-01-01", periods=50),
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=estimator,
        lags=3,
        level="series_1",
        steps=1,
        transformer_series=None,
        categorical_features=categorical_features,
    )

    X_train, y_train, X_test, y_test, train_index, test_index, sample_weight, fit_kwargs = (
        forecaster._train_test_split_one_step_ahead(
            series=series, exog=exog, initial_train_size=40
        )
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
            # 3 lags × 2 series + exog_num + exog_cat = 8 columns
            # exog_cat at index 7
            assert params['categorical_features'] == [7]


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
    series = pd.DataFrame(
        {
            "series_1": np.arange(15),
            "series_2": np.arange(50, 65),
        },
        index=pd.date_range("2020-01-01", periods=15),
        dtype=float,
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), lags=3, level="series_1", steps=1,
        transformer_series=None
    )
    forecaster.is_fitted = initial_is_fitted

    forecaster._train_test_split_one_step_ahead(
        series=series, initial_train_size=10
    )

    assert forecaster.is_fitted == initial_is_fitted
