# Unit test fit ForecasterRecursiveClassifier
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.recursive import ForecasterRecursiveClassifier

# Fixtures
from .fixtures_forecaster_recursive_classifier import y, y_dt
from .fixtures_forecaster_recursive_classifier import exog, exog_dt


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


def test_forecaster_y_exog_features_stored():
    """
    Test forecaster stores y and exog features after fitting.
    """
    rolling = RollingFeaturesClassification(
        stats=['proportion', 'mode'], window_sizes=4
    )
    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(), lags=3, window_features=rolling
    )
    forecaster.fit(y=y, exog=exog)

    series_name_in_ = 'y'
    exog_in_ = True
    exog_type_in_ = type(exog)
    exog_names_in_ = ['exog']
    exog_dtypes_in_ = {'exog': exog.dtype}
    exog_dtypes_out_ = {'exog': exog.dtype}
    X_train_window_features_names_out_ = [
        'roll_proportion_4_class_0.0', 'roll_proportion_4_class_1.0',
        'roll_proportion_4_class_2.0', 'roll_mode_4'
    ]
    X_train_exog_names_out_ = ['exog']
    X_train_features_names_out_ = [
        'lag_1', 'lag_2', 'lag_3', 
        'roll_proportion_4_class_0.0', 'roll_proportion_4_class_1.0',
        'roll_proportion_4_class_2.0', 'roll_mode_4', 'exog'
    ]

    classes_ = [np.int64(1), np.int64(2), np.int64(3)]
    class_codes_ = [0.0, 1.0, 2.0]
    n_classes_ = 3
    encoding_mapping_ = {np.int64(1): 0.0, np.int64(2): 1.0, np.int64(3): 2.0}
    code_to_class_mapping_ = {0.0: np.int64(1), 1.0: np.int64(2), 2.0: np.int64(3)}
    
    assert forecaster.series_name_in_ == series_name_in_
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.exog_dtypes_out_ == exog_dtypes_out_
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_exog_names_out_ == X_train_exog_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_
    assert forecaster.classes_ == classes_
    assert forecaster.class_codes_ == class_codes_
    assert forecaster.n_classes_ == n_classes_
    assert forecaster.encoding_mapping_ == encoding_mapping_
    assert forecaster.code_to_class_mapping_ == code_to_class_mapping_


def test_forecaster_y_exog_features_stored_lgbm_as_categorical():
    """
    Test forecaster stores y and exog features after fitting when using 
    LGBMClassifier.
    """
    rolling = RollingFeaturesClassification(
        stats=['proportion', 'mode'], window_sizes=4
    )
    forecaster = ForecasterRecursiveClassifier(
        LGBMClassifier(verbose=-1), lags=3, window_features=rolling
    )
    forecaster.fit(y=y_dt, exog=exog_dt)

    series_name_in_ = 'y'
    exog_in_ = True
    exog_type_in_ = type(exog_dt)
    exog_names_in_ = ['exog']
    exog_dtypes_in_ = {'exog': exog_dt.dtype}
    exog_dtypes_out_ = {'exog': exog_dt.dtype}
    X_train_window_features_names_out_ = [
        'roll_proportion_4_class_0', 'roll_proportion_4_class_1',
        'roll_proportion_4_class_2', 'roll_mode_4'
    ]
    X_train_exog_names_out_ = ['exog']
    X_train_features_names_out_ = [
        'lag_1', 'lag_2', 'lag_3', 
        'roll_proportion_4_class_0', 'roll_proportion_4_class_1',
        'roll_proportion_4_class_2', 'roll_mode_4', 'exog'
    ]

    classes_ = [np.int64(1), np.int64(2), np.int64(3)]
    class_codes_ = [0, 1, 2]
    n_classes_ = 3
    encoding_mapping_ = {np.int64(1): 0, np.int64(2): 1, np.int64(3): 2}
    code_to_class_mapping_ = {0: np.int64(1), 1: np.int64(2), 2: np.int64(3)}
    
    assert forecaster.series_name_in_ == series_name_in_
    assert forecaster.exog_in_ == exog_in_
    assert forecaster.exog_type_in_ == exog_type_in_
    assert forecaster.exog_names_in_ == exog_names_in_
    assert forecaster.exog_dtypes_in_ == exog_dtypes_in_
    assert forecaster.exog_dtypes_out_ == exog_dtypes_out_
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_exog_names_out_ == X_train_exog_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_
    assert forecaster.classes_ == classes_
    assert forecaster.class_codes_ == class_codes_
    assert forecaster.n_classes_ == n_classes_
    assert forecaster.encoding_mapping_ == encoding_mapping_
    assert forecaster.code_to_class_mapping_ == code_to_class_mapping_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test y_dt.index.freq is stored in forecaster.index_freq.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y_dt)
    expected = y_dt.index.freq
    results = forecaster.index_freq_

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=y)
    expected = y.index.step
    results = forecaster.index_freq_

    assert results == expected


@pytest.mark.parametrize("store_last_window", 
                         [True, False], 
                         ids=lambda lw: f'store_last_window: {lw}')
def test_fit_last_window_stored(store_last_window):
    """
    Test that values of last window are stored after fitting.
    """
    forecaster = ForecasterRecursiveClassifier(LogisticRegression(), lags=3)
    forecaster.fit(y=pd.Series(
        np.arange(50)), store_last_window=store_last_window
    )
    expected = pd.DataFrame(
        np.array([47, 48, 49]), index=[47, 48, 49], columns=['y']
    )

    if store_last_window:
        pd.testing.assert_frame_equal(forecaster.last_window_, expected)
    else:
        assert forecaster.last_window_ is None


def test_fit_model_coef_when_using_weight_func():
    """
    Check the value of the estimator coefs when using a `weight_func`.
    """
    forecaster = ForecasterRecursiveClassifier(
                     estimator   = LogisticRegression(),
                     lags        = 5,
                     weight_func = custom_weights
                 )
    forecaster.fit(y=y)
    results = forecaster.estimator.coef_
    expected = np.array(
        [
            [0.07295195, 0.26058532, -0.12877723, -0.70811789, 0.71544608],
            [0.16912783, -0.11071792, -0.00131558, 0.53663323, -0.83335583],
            [-0.24207979, -0.1498674, 0.13009281, 0.17148466, 0.11790975],
        ]
    )

    np.testing.assert_almost_equal(results, expected)


def test_fit_model_coef_when_not_using_weight_func():
    """
    Check the value of the estimator coefs when not using a `weight_func`.
    """
    forecaster = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(), lags=5
    )
    forecaster.fit(y=y)
    results = forecaster.estimator.coef_
    expected = np.array(
        [
            [0.33080368, 0.01291936, -0.31347922, -0.43600223, 0.25260526],
            [-0.3486165, 0.05728672, -0.06652468, 0.25286034, -0.33176274],
            [0.01781282, -0.07020608, 0.38000389, 0.18314188, 0.07915748],
        ]
    )

    np.testing.assert_almost_equal(results, expected)
