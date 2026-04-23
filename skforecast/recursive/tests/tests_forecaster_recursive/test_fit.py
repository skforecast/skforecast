# Unit test fit ForecasterRecursive
# ==============================================================================
import re
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from skforecast.exceptions import MissingValuesWarning

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog


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


@pytest.mark.parametrize(
    "forecaster_kwargs",
    [
        {"estimator": LinearRegression(), "lags": 3},
        {"estimator": LinearRegression(), "lags": 3,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4)},
        {"estimator": LinearRegression(), "lags": 3,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4),
         "transformer_y": StandardScaler(), "transformer_exog": StandardScaler()},
        {"estimator": LinearRegression(), "lags": 3,
         "window_features": RollingFeatures(stats=['mean'], window_sizes=4),
         "transformer_y": StandardScaler(), "transformer_exog": StandardScaler(),
         "differentiation": 1},
    ],
    ids=["base", "window_features", "transformers", "differentiation"]
)
def test_forecaster_fit_does_not_modify_y_exog(forecaster_kwargs):
    """
    Test forecaster.fit does not modify y and exog.
    """
    y_local = y.copy()
    exog_local = exog.copy()
    y_copy = y_local.copy()
    exog_copy = exog_local.copy()

    forecaster = ForecasterRecursive(**forecaster_kwargs)
    forecaster.fit(y=y_local, exog=exog_local)

    pd.testing.assert_series_equal(y_local, y_copy)
    pd.testing.assert_series_equal(exog_local, exog_copy)


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

    series_name_in_ = 'y'
    exog_in_ = True
    exog_type_in_ = type(exog)
    exog_names_in_ = ['exog']
    exog_dtypes_in_ = {'exog': exog.dtype}
    exog_dtypes_out_ = {'exog': exog.dtype}
    X_train_window_features_names_out_ = ['roll_ratio_min_max_4', 'roll_median_4']
    X_train_exog_names_out_ = ['exog']
    X_train_features_names_out_ = [
        'lag_1', 'lag_2', 'lag_3', 'roll_ratio_min_max_4', 'roll_median_4', 'exog'
    ]
    
    assert forecaster.series_name_in_ == series_name_in_
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.exog_dtypes_out_ == exog_dtypes_out_
    assert forecaster.categorical_features_names_in_ == []
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_exog_names_out_ == X_train_exog_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freq is stored in forecaster.index_freq.
    """
    serie_with_DatetimeIndex = pd.Series(
        data  = [1, 2, 3, 4, 5],
        index = pd.date_range(start='2022-01-01', periods=5)
    )
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=serie_with_DatetimeIndex)
    expected = serie_with_DatetimeIndex.index.freq
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
    forecaster.fit(y=pd.Series(np.arange(5)), store_in_sample_residuals=True)
    results = forecaster.in_sample_residuals_
    expected = np.array([0., 0.])

    assert isinstance(results, np.ndarray)
    np.testing.assert_array_almost_equal(results, expected)


def test_fit_same_residuals_when_residuals_greater_than_10000():
    """
    Test fit return same residuals when residuals len is greater than 10_000.
    Testing with two different forecaster.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(12_000)), store_in_sample_residuals=True)
    results_1 = forecaster.in_sample_residuals_
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(12_000)), store_in_sample_residuals=True)
    results_2 = forecaster.in_sample_residuals_
    
    assert isinstance(results_1, np.ndarray)
    assert isinstance(results_2, np.ndarray)
    assert len(results_1) == 10_000
    assert len(results_2) == 10_000
    np.testing.assert_array_almost_equal(results_1, results_2)


def test_fit_in_sample_residuals_by_bin_stored():
    """
    Test that values of in_sample_residuals_by_bin are stored after fitting.
    """
    forecaster = ForecasterRecursive(
                     estimator     = LinearRegression(),
                     lags          = 5,
                     binner_kwargs = {'n_bins': 3}
                 )
    forecaster.fit(y, store_in_sample_residuals=True)

    X_train, y_train = forecaster.create_train_X_y(y)
    forecaster.estimator.fit(X_train, y_train)
    predictions_estimator = forecaster.estimator.predict(X_train)
    expected_1 = y_train - predictions_estimator

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

    np.testing.assert_array_almost_equal(
        np.sort(forecaster.in_sample_residuals_),
        np.sort(expected_1)
    )
    for k in expected_2.keys():
        np.testing.assert_array_almost_equal(forecaster.in_sample_residuals_by_bin_[k], expected_2[k])
    for k in expected_3.keys():
        assert forecaster.binner_intervals_[k][0] == approx(expected_3[k][0])
        assert forecaster.binner_intervals_[k][1] == approx(expected_3[k][1])


def test_fit_in_sample_residuals_not_stored_probabilistic_mode_binned():
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False`. Binner intervals are stored.
    """
    forecaster = ForecasterRecursive(
                     estimator     = LinearRegression(),
                     lags          = 5,
                     binner_kwargs = {'n_bins': 3}
                 )
    forecaster.fit(y, store_in_sample_residuals=False)

    expected_binner_intervals_ = {
        0: (0.31791969404305154, 0.47312737276420375),
        1: (0.47312737276420375, 0.5259220171775293),
        2: (0.5259220171775293, 0.6492244994657664)
    }

    assert forecaster.in_sample_residuals_ is None
    assert forecaster.in_sample_residuals_by_bin_ is None
    assert forecaster.binner_intervals_.keys() == expected_binner_intervals_.keys()
    for k in expected_binner_intervals_.keys():
        assert forecaster.binner_intervals_[k][0] == approx(expected_binner_intervals_[k][0])
        assert forecaster.binner_intervals_[k][1] == approx(expected_binner_intervals_[k][1])


def test_fit_in_sample_residuals_not_stored_probabilistic_mode_False():
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False` and _probabilistic_mode=False.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster._probabilistic_mode = False
    forecaster.fit(y=pd.Series(np.arange(10), name='y'), store_in_sample_residuals=False)

    assert forecaster.in_sample_residuals_ is None
    assert forecaster.in_sample_residuals_by_bin_ is None
    assert forecaster.binner_intervals_ is None


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
    Check the value of the estimator coefs when using a `weight_func`.
    """
    forecaster = ForecasterRecursive(
                     estimator   = LinearRegression(),
                     lags        = 5,
                     weight_func = custom_weights
                 )
    forecaster.fit(y=y)
    results = forecaster.estimator.coef_
    expected = np.array([0.01211677, -0.20981367,  0.04214442, -0.0369663, -0.18796105])

    np.testing.assert_almost_equal(results, expected)


def test_fit_model_coef_when_not_using_weight_func():
    """
    Check the value of the estimator coefs when not using a `weight_func`.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y)
    results = forecaster.estimator.coef_
    expected = np.array([0.16773502, -0.09712939,  0.10046413, -0.09971515, -0.15849756])

    np.testing.assert_almost_equal(results, expected)


def test_fit_resets_out_sample_residuals_on_refit():
    """
    Test that out_sample_residuals_ and out_sample_residuals_by_bin_ are reset
    to None when the forecaster is refitted.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=y)
    forecaster.set_out_sample_residuals(
        y_true=np.arange(1, 46, dtype=float),
        y_pred=np.zeros(45),
    )

    assert forecaster.out_sample_residuals_ is not None
    assert forecaster.out_sample_residuals_by_bin_ is not None

    forecaster.fit(y=y)

    assert forecaster.out_sample_residuals_ is None
    assert forecaster.out_sample_residuals_by_bin_ is None


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
    for each supported estimator (LGBMRegressor, XGBRegressor,
    HistGradientBoostingRegressor).
    """
    y_cat = pd.Series(np.arange(20, dtype=float), name='y')
    exog_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float),
        'exog_cat': pd.Categorical(range(20))
    })

    forecaster = ForecasterRecursive(
        estimator=estimator, lags=3, categorical_features='auto'
    )
    forecaster.fit(y=y_cat, exog=exog_cat)

    assert forecaster.is_fitted
    assert forecaster.categorical_features_names_in_ == ['exog_cat']
    assert 'exog_cat' in forecaster.X_train_features_names_out_

    if check_fn is not None:
        cat_idx = [
            forecaster.X_train_features_names_out_.index('exog_cat')
        ]
        n_features = len(forecaster.X_train_features_names_out_)
        assert check_fn(forecaster.estimator, cat_idx, n_features)

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
    y_cat = pd.Series(np.arange(20, dtype=float), name='y')
    exog_with_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float),
        'exog_cat': pd.Categorical(['a', 'b'] * 10)
    })
    exog_no_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float)
    })

    forecaster = ForecasterRecursive(
        estimator=estimator, lags=3, categorical_features='auto'
    )

    # First fit — with categoricals
    forecaster.fit(y=y_cat, exog=exog_with_cat)
    assert forecaster.categorical_features_names_in_ == ['exog_cat']

    # Second fit — without categoricals (auto detects no categories → [])
    forecaster.fit(y=y_cat, exog=exog_no_cat)
    assert forecaster.categorical_features_names_in_ == []
    assert forecaster.estimator.get_params()[param_name] == default_value


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
    forecaster = ForecasterRecursive(
        estimator=estimator, lags=3, categorical_features=None
    )
    forecaster.fit(y=y, exog=exog)

    assert forecaster.is_fitted
    assert forecaster.categorical_features_names_in_ is None


def test_fit_with_interspersed_NaN_and_dropna_from_series_True():
    """
    Test fit works correctly with interspersed NaN in y and
    dropna_from_series=True. Estimator: LinearRegression.
    """

    y_nan = pd.Series(
        data = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        name = 'y'
    )
    forecaster = ForecasterRecursive(
                     estimator          = LinearRegression(),
                     lags               = 3,
                     dropna_from_series = True
                 )

    warn_msg = re.escape(
        "NaNs detected in `X_train`. They have been dropped."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        forecaster.fit(y=y_nan)

    assert forecaster.is_fitted
    assert not np.isnan(forecaster.last_window_.to_numpy()).any()
    predictions = forecaster.predict(steps=3)
    assert len(predictions) == 3
    assert not predictions.isna().any()


def test_fit_with_interspersed_NaN_and_dropna_from_series_False():
    """
    Test fit works correctly with interspersed NaN in y and
    dropna_from_series=False. Estimator: HistGradientBoostingRegressor
    (supports NaN natively).
    """

    y_nan = pd.Series(
        data = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        name = 'y'
    )
    forecaster = ForecasterRecursive(
                     estimator          = HistGradientBoostingRegressor(random_state=123),
                     lags               = 3,
                     dropna_from_series = False
                 )

    warn_msg = re.escape(
        "NaNs detected in `X_train`. Some estimators do not allow "
        "NaN values during training."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        forecaster.fit(y=y_nan)

    assert forecaster.is_fitted
    assert not np.isnan(forecaster.last_window_.to_numpy()).any()
    predictions = forecaster.predict(steps=3)
    assert len(predictions) == 3
    assert not predictions.isna().any()
