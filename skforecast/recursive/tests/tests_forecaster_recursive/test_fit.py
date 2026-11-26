# Unit test fit ForecasterRecursive
# ==============================================================================
import json
import os
import pickle
import re
import subprocess
import sys
import textwrap
from pathlib import Path
import pytest
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skforecast.preprocessing import RollingFeatures, CalendarFeatures
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
    
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2022-01-01', periods=len(y), freq='D')
    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(start='2022-01-01', periods=len(exog), freq='D')

    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    calendar = CalendarFeatures(
        features=['day_of_week', 'weekend'], encoding="cyclical"
    )

    forecaster = ForecasterRecursive(
        LinearRegression(), lags=3, window_features=rolling, calendar_features=calendar
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)

    series_name_in_ = 'y'
    exog_in_ = True
    exog_type_in_ = type(exog_datetime)
    exog_names_in_ = ['exog']
    exog_dtypes_in_ = {'exog': exog_datetime.dtype}
    exog_dtypes_out_ = {'exog': exog_datetime.dtype}
    X_train_window_features_names_out_ = ['roll_ratio_min_max_4', 'roll_median_4']
    X_train_calendar_features_names_out_ = ['weekend', 'day_of_week_sin', 'day_of_week_cos']
    X_train_exog_names_out_ = ['exog']
    X_train_features_names_out_ = [
        'lag_1', 'lag_2', 'lag_3', 'roll_ratio_min_max_4', 'roll_median_4', 
        'exog', 'weekend', 'day_of_week_sin', 'day_of_week_cos'
    ]
    
    assert forecaster.series_name_in_ == series_name_in_
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.exog_dtypes_out_ == exog_dtypes_out_
    assert forecaster.categorical_features_names_in_ == []
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_calendar_features_names_out_ == X_train_calendar_features_names_out_
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


@pytest.mark.parametrize(
    "estimator, check_fn",
    [
        (
            CatBoostRegressor(
                iterations=10, random_seed=123, verbose=0,
                allow_writing_files=False
            ),
            lambda model: model.get_cat_feature_indices() == []
        ),
        (
            LGBMRegressor(verbose=-1, random_state=123),
            lambda model: model.booster_.params.get('categorical_column') is None
        ),
        (
            XGBRegressor(random_state=123),
            lambda model: model.get_params()['feature_types'] is None
        ),
        (
            HistGradientBoostingRegressor(random_state=123),
            lambda model: (
                model.get_params()['categorical_features'] == 'from_dtype'
                and model.is_categorical_ is None
            )
        ),
    ],
    ids=['CatBoostRegressor', 'LGBMRegressor', 'XGBRegressor', 'HistGradientBoostingRegressor']
)
def test_fit_does_not_configure_categorical_features_when_estimator_is_Pipeline(
    estimator, check_fn
):
    """
    Test that fit does not configure native categorical features when the
    estimator is a Pipeline, whatever its last step is (CatBoostRegressor,
    LGBMRegressor, XGBRegressor, HistGradientBoostingRegressor): the previous
    steps may transform the columns, so the categorical features reach the
    last step with their ordinal encoding, as numeric columns.
    """
    y_cat = pd.Series(np.arange(20, dtype=float), name='y')
    exog_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float),
        'exog_cat': pd.Categorical(['a', 'b'] * 10)
    })

    forecaster = ForecasterRecursive(
        estimator=make_pipeline(StandardScaler(), estimator),
        lags=3,
        categorical_features='auto'
    )
    forecaster.fit(y=y_cat, exog=exog_cat)

    assert forecaster.is_fitted
    assert forecaster.categorical_features_names_in_ == ['exog_cat']
    assert 'exog_cat' in forecaster.X_train_features_names_out_
    assert check_fn(forecaster.estimator[-1])

    # fit_kwargs must not be mutated
    assert forecaster.fit_kwargs == {}


@pytest.mark.parametrize(
    "estimator, get_model",
    [
        (
            TransformedTargetRegressor(
                regressor=make_pipeline(
                    StandardScaler(),
                    LGBMRegressor(
                        n_estimators=10, min_child_samples=2,
                        random_state=123, verbose=-1
                    )
                )
            ),
            lambda estimator: estimator.regressor_[-1]
        ),
        (
            make_pipeline(
                StandardScaler(),
                TransformedTargetRegressor(
                    regressor=LGBMRegressor(
                        n_estimators=10, min_child_samples=2,
                        random_state=123, verbose=-1
                    )
                )
            ),
            lambda estimator: estimator[-1].regressor_
        ),
    ],
    ids=['TransformedTargetRegressor(Pipeline)', 'Pipeline(TransformedTargetRegressor)']
)
def test_fit_and_predict_when_Pipeline_nested_with_TransformedTargetRegressor(
    estimator, get_model
):
    """
    Test that fit and predict work when a Pipeline and a
    TransformedTargetRegressor are nested, in both orders, around
    LGBMRegressor and there is a categorical feature. Native categorical
    features are not configured, so no argument reaches the `fit` of the
    Pipeline and the categorical feature is passed with its ordinal encoding.
    """
    y_cat = pd.Series(np.arange(20, dtype=float), name='y')
    exog_cat = pd.DataFrame({
        'exog_num': np.arange(100, 120, dtype=float),
        'exog_cat': pd.Categorical(['a', 'b'] * 10)
    })
    exog_predict = pd.DataFrame(
        {'exog_num': np.arange(120, 123, dtype=float),
         'exog_cat': pd.Categorical(['a', 'b', 'a'], categories=['a', 'b'])},
        index=pd.RangeIndex(start=20, stop=23, step=1)
    )

    forecaster = ForecasterRecursive(
        estimator=estimator, lags=3, categorical_features='auto'
    )
    forecaster.fit(y=y_cat, exog=exog_cat)
    predictions = forecaster.predict(steps=3, exog=exog_predict)

    model = get_model(forecaster.estimator)
    assert model.booster_.params.get('categorical_column') is None

    expected = pd.Series(
                   data  = np.array([15.8849117279, 14.2566078107, 14.2566078107]),
                   index = pd.RangeIndex(start=20, stop=23, step=1),
                   name  = 'pred'
               )

    pd.testing.assert_series_equal(predictions, expected)


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
        "Some rows of the training matrix have missing values. 3 of 7 rows "
        "are dropped, with NaN predictors (lags, window features or exog). "
        "They come from interspersed NaNs in `y` or `exog`. To keep the "
        "rows with NaN predictors, set `forecaster.dropna_from_series = "
        "False` (the estimator must support NaN)."
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
        "Some rows of the training matrix have missing values. 3 of 7 rows "
        "are kept with NaN predictors (lags, window features or exog). They "
        "come from interspersed NaNs in `y` or `exog`. The estimator must "
        "support NaN. To drop those rows, set "
        "`forecaster.dropna_from_series = True`."
    )
    with pytest.warns(MissingValuesWarning, match=warn_msg):
        forecaster.fit(y=y_nan)

    assert forecaster.is_fitted
    assert not np.isnan(forecaster.last_window_.to_numpy()).any()
    predictions = forecaster.predict(steps=3)
    assert len(predictions) == 3
    assert not predictions.isna().any()



@pytest.mark.parametrize(
    "estimator, expected_c_contiguous",
    [
        (Ridge(), True),
        (HistGradientBoostingRegressor(max_iter=5, random_state=123), False),
    ],
    ids=lambda x: type(x).__name__ if not isinstance(x, bool) else f"C: {x}"
)
def test_fit_training_matrix_layout_received_by_estimator(
    estimator, expected_c_contiguous, monkeypatch
):
    """
    Test the memory layout of the training matrix that the estimator receives
    in `fit` when the predictors are only lags: a C-contiguous copy for the
    estimators that prefer it (Ridge), and the view of `sliding_window_view`,
    not copied, for the rest (HistGradientBoostingRegressor).
    """
    flags = []
    original_fit = type(estimator).fit

    def recording_fit(self, X, y, *args, **kwargs):
        flags.append(np.asarray(X).flags)
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(type(estimator), "fit", recording_fit)
    forecaster = ForecasterRecursive(estimator=estimator, lags=5)
    forecaster.fit(y=y)

    assert len(flags) == 1
    for flag in flags:
        assert flag.c_contiguous is expected_c_contiguous
        assert flag.owndata is expected_c_contiguous


def test_fit_predictions_equal_with_c_contiguous_training_matrix():
    """
    Test that the copy of the training matrix to a C-contiguous array does not
    change the predictions with Ridge: they are equal to the ones of skforecast
    0.27.x before the copy, when the estimator received the view of
    `sliding_window_view`, up to the rounding of the BLAS library, which differs
    between platforms (about 1e-16).
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=5)
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5).to_numpy()
    expected = np.array([0.5736183556677587, 0.4747505305408221, 0.4405594104104335,
                         0.49070097324783774, 0.48362170469052196])

    np.testing.assert_allclose(predictions, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "estimator",
    [
        CatBoostRegressor(iterations=2, verbose=0, allow_writing_files=False),
        make_pipeline(
            StandardScaler(),
            CatBoostRegressor(iterations=2, verbose=0, allow_writing_files=False)
        ),
    ],
    ids=["CatBoostRegressor", "Pipeline"]
)
def test_fit_replaces_fitted_CatBoost_estimator_with_unfitted_clone(estimator):
    """
    Test that the first fit keeps the estimator object and that a CatBoost
    estimator that is already fitted, also as the last step of a Pipeline or in
    a forecaster copied with pickle, is replaced with a clone before it is
    fitted again. No categorical features are used: CatBoost can block when a
    fitted model with categorical features is fitted again.
    """
    forecaster = ForecasterRecursive(estimator=estimator, lags=3)
    estimator_new = forecaster.estimator
    forecaster.fit(y=y)
    estimator_first_fit = forecaster.estimator
    forecaster_copy = pickle.loads(pickle.dumps(forecaster))
    estimator_copy = forecaster_copy.estimator
    forecaster.fit(y=y)
    forecaster_copy.fit(y=y)

    assert estimator_first_fit is estimator_new
    assert forecaster.estimator is not estimator_first_fit
    assert forecaster_copy.estimator is not estimator_copy
    assert forecaster.is_fitted
    assert forecaster_copy.is_fitted


@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(), LGBMRegressor(n_estimators=2, verbose=-1)],
    ids=lambda estimator: f"estimator: {type(estimator).__name__}"
)
def test_fit_keeps_estimator_object_when_fitted_again_and_not_CatBoost(estimator):
    """
    Test that an estimator that is not a CatBoost model is fitted again without
    being replaced with a clone.
    """
    forecaster = ForecasterRecursive(estimator=estimator, lags=3)
    estimator_new = forecaster.estimator
    forecaster.fit(y=y)
    forecaster.fit(y=y)

    assert forecaster.estimator is estimator_new


def test_fit_predict_output_when_CatBoost_with_categorical_features_fitted_again():
    """
    Test that a forecaster with a CatBoost estimator and categorical features
    that is fitted again, the same object or a copy made with pickle, predicts
    as a new forecaster fitted with the same data. With categorical features,
    CatBoost does not give the model of a new object when a fitted model is
    fitted again, and `predict` can block forever with a copy, so the fits run
    in a subprocess with a time limit.
    """
    code = textwrap.dedent("""
        import json
        import pickle
        import numpy as np
        import pandas as pd
        from catboost import CatBoostRegressor
        from skforecast.recursive import ForecasterRecursive

        rng = np.random.RandomState(7)
        n = 260
        index = pd.date_range("2020-01-01", periods=n, freq="D")
        codes = rng.permutation(np.arange(n) % 6)
        effect = np.array([5.0, -3.0, 9.0, 0.0, -8.0, 2.0])
        y = pd.Series(
            effect[codes] + rng.normal(scale=0.3, size=n), index=index, name="y"
        )
        exog = pd.DataFrame(
            {"c": pd.Categorical(codes), "x": rng.normal(size=n)}, index=index
        )

        def create_forecaster():
            return ForecasterRecursive(
                estimator=CatBoostRegressor(
                    iterations=30, depth=3, verbose=0, thread_count=1,
                    random_seed=1, allow_writing_files=False
                ),
                lags=3
            )

        forecaster = create_forecaster()
        forecaster.fit(y=y.iloc[:200], exog=exog.iloc[:200])
        forecasters = {
            "new": create_forecaster(),
            "pickle": pickle.loads(pickle.dumps(forecaster)),
            "same": forecaster
        }
        predictions = {}
        for name, forecaster in forecasters.items():
            forecaster.fit(y=y.iloc[50:250], exog=exog.iloc[50:250])
            predictions[name] = forecaster.predict(
                steps=5, exog=exog.iloc[250:255]
            ).to_list()
        print(json.dumps(predictions))
    """)
    repo_root = Path(__file__).resolve().parents[4]

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        encoding       = "utf-8",
        errors         = "replace",
        cwd            = repo_root,
        env            = {**os.environ, "PYTHONPATH": str(repo_root)},
        timeout        = 120
    )

    assert result.returncode == 0, result.stderr
    predictions = json.loads(result.stdout.splitlines()[-1])

    # NOTE: CatBoost results with categorical features change between platforms
    # (Linux, macOS), so the expected values are not hardcoded. They are the
    # predictions of a new forecaster fitted with the same data in the subprocess.
    expected = np.array(predictions["new"])

    assert expected.shape == (5,)
    np.testing.assert_array_almost_equal(predictions["same"], expected)
    np.testing.assert_array_almost_equal(predictions["pickle"], expected)
