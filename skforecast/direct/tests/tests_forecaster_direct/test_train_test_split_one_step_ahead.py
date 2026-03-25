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

    assert sample_weight is not None
    assert isinstance(sample_weight, np.ndarray)
    assert len(sample_weight) == len(y_train)
    # train_index[1]: 2020-01-04..2020-01-10 (7 obs, lags=3 so first at y[3])
    # weights: 2020-01-04,05,06 -> 1.0; 2020-01-07,08,09,10 -> 2.0
    expected_weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(sample_weight, expected_weights)


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

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    # 3 lags + 1 window_feature (rolling mean) + 1 exog = 5 columns
    assert X_train.shape[1] == 5
    assert X_test.shape[1] == 5
    # window_size = max(3, 3) + differentiation(1) = 4
    # train rows: 15 - 1(diff) = 14, then 14 - 3 window = 11
    # test_init = 15 - 4 = 11, y[11:] = 9 obs -> diff 8 -> -3 window = 5 rows
    assert X_train.shape[0] == 11
    assert X_test.shape[0] == 5
    assert len(y_train) == 11
    assert len(y_test) == 5
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

    # Filtered to step 1: 5 lags + 2 exog (step 1 only) = 7 columns
    # Train: 15 - 5(lags) - 2(steps-1) = 8 rows
    # Test: test_init = 15-5 = 10, y[10:20] = 10 obs, 10-5-2 = 3 rows
    assert X_train.shape == (8, 7)
    assert X_test.shape == (3, 7)
    assert len(y_train) == 8
    assert len(y_test) == 3
    # First train row: lags=[4,3,2,1,0], exog_step1=[105,1005], y=5
    np.testing.assert_array_almost_equal(
        X_train[0], [4.0, 3.0, 2.0, 1.0, 0.0, 105.0, 1005.0]
    )
    np.testing.assert_array_almost_equal(y_train[0], 5.0)
    # First test row: lags=[14,13,12,11,10], exog_step1=[115,1015], y=15
    np.testing.assert_array_almost_equal(
        X_test[0], [14.0, 13.0, 12.0, 11.0, 10.0, 115.0, 1015.0]
    )
    np.testing.assert_array_almost_equal(y_test[0], 15.0)
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

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    # Features: lag_1, lag_2, lag_3, exog_num, exog_cat -> cat at index 4
    assert 'categorical_feature' in fit_kwargs
    assert fit_kwargs['categorical_feature'] == [4]
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

    # XGBoost uses set_params in-place, not fit_kwargs
    assert 'categorical_feature' not in fit_kwargs
    assert 'cat_features' not in fit_kwargs
    # estimators_[1] should be configured (not self.estimator)
    params = forecaster.estimators_[1].get_params()
    assert params['enable_categorical'] is True
    # Features: lag_1, lag_2, lag_3, exog_num, exog_cat -> cat at index 4
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

    # HistGradientBoosting uses set_params in-place, not fit_kwargs
    assert 'categorical_feature' not in fit_kwargs
    assert 'cat_features' not in fit_kwargs
    # estimators_[1] should be configured
    params = forecaster.estimators_[1].get_params()
    # Features: lag_1, lag_2, lag_3, exog_num, exog_cat -> cat at index 4
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

    # Features: lag_1, lag_2, lag_3, exog_num, exog_cat -> cat at index 4
    assert 'cat_features' in fit_kwargs
    assert fit_kwargs['cat_features'] == [4]
    # CatBoost requires object dtype with int-cast categorical columns
    assert X_train.dtype == object
    assert X_test.dtype == object
    # Categorical column (index 4) should contain int values
    assert all(isinstance(v, (int, np.integer)) for v in X_train[:, 4])
    assert all(isinstance(v, (int, np.integer)) for v in X_test[:, 4])
    # Non-categorical columns should still be float
    assert all(isinstance(v, (float, np.floating)) for v in X_train[:, 0])


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

    # window_size = max_lag = 7
    # train rows: 15 - 7 = 8
    # test_init = 15 - 7 = 8, y[8:] = 12 obs -> 12 - 7 = 5 rows
    # columns: lag_1, lag_7, exog_1 = 3
    assert X_train.shape == (8, 3)
    assert X_test.shape == (5, 3)
    assert len(y_train) == 8
    assert len(y_test) == 5
    # First training row: y[7]=7 is target, features = [lag_1=y[6]=6, lag_7=y[0]=0, exog=107]
    np.testing.assert_array_almost_equal(X_train[0], [6.0, 0.0, 107.0])
    # First test row: y[15]=15 is target, features = [lag_1=y[14]=14, lag_7=y[8]=8, exog=115]
    np.testing.assert_array_almost_equal(X_test[0], [14.0, 8.0, 115.0])
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
