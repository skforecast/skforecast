# Unit test _train_test_split_one_step_ahead ForecasterRecursiveClassifier
# ==============================================================================
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from skforecast.recursive import ForecasterRecursiveClassifier


def test_train_test_split_one_step_ahead_when_lags_and_exog():
    """
    Test _train_test_split_one_step_ahead returns correct numpy arrays with
    ordinal-encoded lags and exog. LogisticRegression does not support native
    categoricals, so features_encoding='auto' falls back to ordinal and
    fit_kwargs={}.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 5),
        index=pd.date_range("2020-01-01", periods=15),
        name='y'
    )
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 115, dtype=float),
            "exog_2": np.arange(1000, 1015, dtype=float),
        },
        index=pd.date_range("2020-01-01", periods=15),
    )

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=5)

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=10
    )

    # y encoded: a=0, b=1, c=2. Lags are ordinal-encoded integers.
    expected_X_train = np.array([
        [1., 0., 2., 1., 0., 105., 1005.],
        [2., 1., 0., 2., 1., 106., 1006.],
        [0., 2., 1., 0., 2., 107., 1007.],
        [1., 0., 2., 1., 0., 108., 1008.],
        [2., 1., 0., 2., 1., 109., 1009.],
    ])
    expected_y_train = np.array([2, 0, 1, 2, 0])

    expected_X_test = np.array([
        [0., 2., 1., 0., 2., 110., 1010.],
        [1., 0., 2., 1., 0., 111., 1011.],
        [2., 1., 0., 2., 1., 112., 1012.],
        [0., 2., 1., 0., 2., 113., 1013.],
        [1., 0., 2., 1., 0., 114., 1014.],
    ])
    expected_y_test = np.array([1, 2, 0, 1, 2])

    np.testing.assert_array_almost_equal(X_train, expected_X_train)
    np.testing.assert_array_almost_equal(y_train, expected_y_train)
    np.testing.assert_array_almost_equal(X_test, expected_X_test)
    np.testing.assert_array_almost_equal(y_test, expected_y_test)
    assert sample_weight is None
    assert fit_kwargs == {}


def test_train_test_split_one_step_ahead_when_no_exog():
    """
    Test _train_test_split_one_step_ahead when exog is None. X_train and
    X_test contain only ordinal-encoded lag columns.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 5),
        index=pd.date_range("2020-01-01", periods=15),
        name='y'
    )

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, initial_train_size=10
    )

    # a=0, b=1, c=2. Lags only: 3 columns.
    # y = [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]
    # First trainable at index 3 (y[3]=0): lags = [y[2]=2, y[1]=1, y[0]=0]
    expected_X_train = np.array([
        [2., 1., 0.],
        [0., 2., 1.],
        [1., 0., 2.],
        [2., 1., 0.],
        [0., 2., 1.],
        [1., 0., 2.],
        [2., 1., 0.],
    ])
    expected_y_train = np.array([0, 1, 2, 0, 1, 2, 0])

    expected_X_test = np.array([
        [0., 2., 1.],
        [1., 0., 2.],
        [2., 1., 0.],
        [0., 2., 1.],
        [1., 0., 2.],
    ])
    expected_y_test = np.array([1, 2, 0, 1, 2])

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
        np.array(['a', 'b', 'c'] * 5),
        index=pd.date_range("2020-01-01", periods=15),
        name='y'
    )

    def custom_weights(index):
        return np.where(index >= pd.Timestamp("2020-01-07"), 2.0, 1.0)

    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=3, weight_func=custom_weights
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, initial_train_size=10
    )

    assert sample_weight is not None
    assert isinstance(sample_weight, np.ndarray)
    assert len(sample_weight) == len(y_train)
    # train_index: 2020-01-04..2020-01-10 (7 obs, lags=3 so first at y[3])
    # weights: 2020-01-04,05,06 -> 1.0; 2020-01-07,08,09,10 -> 2.0
    expected_weights = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(sample_weight, expected_weights)


def test_train_test_split_one_step_ahead_when_lightgbm_native_categoricals():
    """
    Test _train_test_split_one_step_ahead with LGBMClassifier and
    features_encoding='auto'. Lags should be treated as native categorical
    features (use_native_categoricals=True). fit_kwargs should contain
    'categorical_feature' with lag indices.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 7),
        index=pd.date_range("2020-01-01", periods=21),
        name='y'
    )

    forecaster = ForecasterRecursiveClassifier(
        estimator=LGBMClassifier(verbose=-1, random_state=123),
        lags=3,
        features_encoding='auto',
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, initial_train_size=15
    )

    assert forecaster.use_native_categoricals is True
    # Features: lag_1, lag_2, lag_3 -> all categorical at indices [0, 1, 2]
    assert 'categorical_feature' in fit_kwargs
    assert fit_kwargs['categorical_feature'] == [0, 1, 2]
    assert X_train.shape[1] == 3
    assert X_test.shape[1] == 3


def test_train_test_split_one_step_ahead_when_lightgbm_native_categoricals_and_exog_categorical():
    """
    Test _train_test_split_one_step_ahead with LGBMClassifier,
    features_encoding='auto', and exog with categorical columns.
    fit_kwargs['categorical_feature'] should include both lag indices and
    exog categorical indices.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 7),
        index=pd.date_range("2020-01-01", periods=21),
        name='y'
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 121, dtype=float),
            "exog_cat": pd.Categorical(['x', 'y', 'z'] * 7),
        },
        index=pd.date_range("2020-01-01", periods=21),
    )

    forecaster = ForecasterRecursiveClassifier(
        estimator=LGBMClassifier(verbose=-1, random_state=123),
        lags=3,
        features_encoding='auto',
        categorical_features='auto',
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    # Features: lag_1(0), lag_2(1), lag_3(2), exog_num(3), exog_cat(4)
    # Categorical: lags [0,1,2] + exog_cat [4] = [0, 1, 2, 4]
    assert 'categorical_feature' in fit_kwargs
    assert fit_kwargs['categorical_feature'] == [0, 1, 2, 4]
    assert X_train.shape[1] == 5
    # The original forecaster.fit_kwargs should not be mutated
    assert 'categorical_feature' not in forecaster.fit_kwargs


def test_train_test_split_one_step_ahead_when_features_encoding_ordinal():
    """
    Test _train_test_split_one_step_ahead with LGBMClassifier but
    features_encoding='ordinal'. Lags should NOT be treated as native
    categoricals even though estimator supports them. Only exog categorical
    features should appear in fit_kwargs.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 7),
        index=pd.date_range("2020-01-01", periods=21),
        name='y'
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 121, dtype=float),
            "exog_cat": pd.Categorical(['x', 'y', 'z'] * 7),
        },
        index=pd.date_range("2020-01-01", periods=21),
    )

    forecaster = ForecasterRecursiveClassifier(
        estimator=LGBMClassifier(verbose=-1, random_state=123),
        lags=3,
        features_encoding='ordinal',
        categorical_features='auto',
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    assert forecaster.use_native_categoricals is False
    # Features: lag_1(0), lag_2(1), lag_3(2), exog_num(3), exog_cat(4)
    # Only exog_cat at index 4 (lags are NOT categorical in ordinal mode)
    assert 'categorical_feature' in fit_kwargs
    assert fit_kwargs['categorical_feature'] == [4]


def test_train_test_split_one_step_ahead_when_catboost_native_categoricals():
    """
    Test _train_test_split_one_step_ahead with CatBoostClassifier and
    features_encoding='auto'. Verify fit_kwargs contains 'cat_features'
    with lag + exog indices and that categorical columns are cast to int
    (object dtype).
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 7),
        index=pd.date_range("2020-01-01", periods=21),
        name='y'
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 121, dtype=float),
            "exog_cat": pd.Categorical(['x', 'y', 'z'] * 7),
        },
        index=pd.date_range("2020-01-01", periods=21),
    )

    forecaster = ForecasterRecursiveClassifier(
        estimator=CatBoostClassifier(
            iterations=10, random_seed=123, verbose=0,
            allow_writing_files=False
        ),
        lags=3,
        features_encoding='auto',
        categorical_features='auto',
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    # Features: lag_1(0), lag_2(1), lag_3(2), exog_num(3), exog_cat(4)
    # Categorical: lags [0,1,2] + exog_cat [4] = [0, 1, 2, 4]
    assert 'cat_features' in fit_kwargs
    assert fit_kwargs['cat_features'] == [0, 1, 2, 4]
    # CatBoost requires object dtype with int-cast categorical columns
    assert X_train.dtype == object
    assert X_test.dtype == object
    # Lag columns (indices 0,1,2) and exog_cat (index 4) should be int
    for idx in [0, 1, 2, 4]:
        assert all(isinstance(v, (int, np.integer)) for v in X_train[:, idx])
        assert all(isinstance(v, (int, np.integer)) for v in X_test[:, idx])
    # Non-categorical column (exog_num at index 3) should still be float
    assert all(isinstance(v, (float, np.floating)) for v in X_train[:, 3])


def test_train_test_split_one_step_ahead_when_fit_kwargs_no_categorical():
    """
    Test _train_test_split_one_step_ahead passes user-defined fit_kwargs
    through the else branch (features_encoding='ordinal',
    categorical_features=None, use_native_categoricals=False). Verify that
    fit_kwargs is a copy and the original is not mutated.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 7),
        index=pd.date_range("2020-01-01", periods=21),
        name='y'
    )

    forecaster = ForecasterRecursiveClassifier(
        estimator=LGBMClassifier(verbose=-1, random_state=123),
        lags=3,
        features_encoding='ordinal',
        categorical_features=None,
        fit_kwargs={'categorical_feature': 'auto'},
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, initial_train_size=15
    )

    # User fit_kwargs passed through the else branch
    assert fit_kwargs == {'categorical_feature': 'auto'}
    # fit_kwargs is a copy, not the same object
    assert fit_kwargs is not forecaster.fit_kwargs


def test_train_test_split_one_step_ahead_when_non_contiguous_lags():
    """
    Test _train_test_split_one_step_ahead with non-contiguous lags.
    Verify that window_size is driven by the maximum lag and that shapes
    and feature values are correct.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 7),
        index=pd.date_range("2020-01-01", periods=21),
        name='y'
    )
    exog = pd.DataFrame(
        {"exog_1": np.arange(100, 121, dtype=float)},
        index=pd.date_range("2020-01-01", periods=21),
    )

    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=[1, 7]
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=15
    )

    # window_size = max_lag = 7
    # train rows: 15 - 7 = 8
    # test_init = 15 - 7 = 8, y[8:] = 13 obs -> 13 - 7 = 6 rows
    # columns: lag_1, lag_7, exog_1 = 3
    assert X_train.shape == (8, 3)
    assert X_test.shape == (6, 3)
    assert len(y_train) == 8
    assert len(y_test) == 6
    assert sample_weight is None
    assert fit_kwargs == {}


@pytest.mark.parametrize(
    'estimator, features_encoding, categorical_features, expected_fit_kwarg, expected_cat_indices',
    [
        (
            LGBMClassifier(verbose=-1, random_state=123, n_estimators=10),
            'auto',
            'auto',
            'categorical_feature',
            [0, 1, 2, 4],  # 3 lags + exog_cat
        ),
        (
            LGBMClassifier(verbose=-1, random_state=123, n_estimators=10),
            'ordinal',
            'auto',
            'categorical_feature',
            [4],  # only exog_cat (lags are ordinal)
        ),
        (
            CatBoostClassifier(
                iterations=10, random_seed=123, verbose=0,
                allow_writing_files=False
            ),
            'auto',
            'auto',
            'cat_features',
            [0, 1, 2, 4],  # 3 lags + exog_cat
        ),
    ],
    ids=['LightGBM-auto', 'LightGBM-ordinal', 'CatBoost-auto']
)
def test_train_test_split_one_step_ahead_fit_predict_with_categorical(
    estimator, features_encoding, categorical_features,
    expected_fit_kwarg, expected_cat_indices
):
    """
    Integration test: verify that the output of _train_test_split_one_step_ahead
    can be used directly to fit and predict with estimators that support native
    categorical features. Tests both the features_encoding (AR categoricals)
    and categorical_features (exog categoricals) configuration.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 17),
        index=pd.date_range("2020-01-01", periods=51),
        name='y'
    )
    exog = pd.DataFrame(
        {
            "exog_num": np.arange(100, 151, dtype=float),
            "exog_cat": pd.Categorical(['x', 'y', 'z'] * 17),
        },
        index=pd.date_range("2020-01-01", periods=51),
    )

    forecaster = ForecasterRecursiveClassifier(
        estimator=estimator,
        lags=3,
        features_encoding=features_encoding,
        categorical_features=categorical_features,
    )

    (
        X_train, y_train, X_test, y_test, sample_weight, fit_kwargs
    ) = forecaster._train_test_split_one_step_ahead(
        y=y, exog=exog, initial_train_size=40
    )

    # Verify categorical indices match expectations
    assert expected_fit_kwarg in fit_kwargs
    assert fit_kwargs[expected_fit_kwarg] == expected_cat_indices

    # Fit must succeed with the provided data and kwargs
    forecaster.estimator.fit(X_train, y_train, **fit_kwargs)

    # Predict must succeed and return correct number of predictions
    predictions = forecaster.estimator.predict(X_test).ravel()
    assert len(predictions) == len(y_test)


@pytest.mark.parametrize(
    'initial_is_fitted',
    [True, False],
    ids=lambda v: f'is_fitted={v}'
)
def test_train_test_split_one_step_ahead_restores_is_fitted_and_encoding_mapping(
    initial_is_fitted
):
    """
    Test _train_test_split_one_step_ahead preserves the original is_fitted
    state and encoding_mapping_ of the forecaster after execution.
    """
    y = pd.Series(
        np.array(['a', 'b', 'c'] * 5),
        index=pd.date_range("2020-01-01", periods=15),
        name='y'
    )

    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.is_fitted = initial_is_fitted
    original_encoding = {'sentinel': 999}
    forecaster.encoding_mapping_ = original_encoding

    forecaster._train_test_split_one_step_ahead(
        y=y, initial_train_size=10
    )

    assert forecaster.is_fitted == initial_is_fitted
    assert forecaster.encoding_mapping_ is original_encoding
