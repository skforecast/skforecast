# Unit test __init__ ForecasterRecursiveClassifier
# ==============================================================================
import re
import pytest
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from skforecast.preprocessing import RollingFeaturesClassification
from skforecast.recursive import ForecasterRecursiveClassifier


def test_init_ValueError_when_features_encoding_not_valid():
    """
    Test ValueError is raised when `features_encoding` is not valid.
    """
    err_msg = re.escape(
        "`features_encoding` must be one of ['auto', 'categorical', 'ordinal']. "
        "Got 'not_valid'."
    )
    with pytest.raises(ValueError, match = err_msg):
        ForecasterRecursiveClassifier(
            regressor         = LogisticRegression(),
            lags              = 3,
            features_encoding = 'not_valid'
        )


def test_init_ValueError_when_estimator_does_not_support_categorical_features():
    """
    Test ValueError is raised when `features_encoding='categorical'` and
    the estimator does not support categorical features natively.
    """
    err_msg = re.escape(
        f"`features_encoding='categorical'` requires a estimator that "
        f"supports native categorical features (LightGBM, CatBoost, XGBoost). "
        f"Got {type(LogisticRegression()).__name__}. Use 'auto' or 'ordinal' instead."
    )
    with pytest.raises(ValueError, match = err_msg):
        ForecasterRecursiveClassifier(
            regressor         = LogisticRegression(),
            lags              = 3,
            features_encoding = 'categorical'
        )


def test_init_ValueError_when_no_lags_or_window_features():
    """
    Test ValueError is raised when no lags or window_features are passed.
    """
    err_msg = re.escape(
        "At least one of the arguments `lags` or `window_features` "
        "must be different from None. This is required to create the "
        "predictors used in training the forecaster."
    )
    with pytest.raises(ValueError, match = err_msg):
        ForecasterRecursiveClassifier(
            regressor       = LogisticRegression(),
            lags            = None,
            window_features = None
        )


@pytest.mark.parametrize("features_encoding, estimator, expected", 
                         [('auto', LogisticRegression(), False), 
                          ('auto', LGBMClassifier(verbose=-1), True), 
                          ('categorical', LGBMClassifier(verbose=-1), True), 
                          ('ordinal', LGBMClassifier(verbose=-1), False)], 
                         ids = lambda dt: f'features_encoding, estimator, expected: {dt}')
def test_init_use_native_categoricals_set(features_encoding, estimator, expected):
    """
    Test use_native_categoricals is correctly set during initialization.
    """

    forecaster = ForecasterRecursiveClassifier(
                     regressor         = estimator,
                     lags              = 3,
                     features_encoding = features_encoding
                 )
    
    assert forecaster.use_native_categoricals == expected


@pytest.mark.parametrize("lags, window_features, expected", 
                         [(5, None, 5), 
                          (None, True, 6), 
                          ([], True, 6), 
                          (5, True, 6)], 
                         ids = lambda dt: f'lags, window_features, expected: {dt}')
def test_init_window_size_correctly_stored(lags, window_features, expected):
    """
    Test window_size is correctly stored when lags or window_features are passed.
    """
    if window_features:
        window_features = RollingFeaturesClassification(
            stats=['proportion', 'mode'], window_sizes=[5, 6]
        )

    forecaster = ForecasterRecursiveClassifier(
                     regressor       = LogisticRegression(),
                     lags            = lags,
                     window_features = window_features
                 )
    
    assert forecaster.window_size == expected
    if lags:
        np.testing.assert_array_almost_equal(forecaster.lags, np.array([1, 2, 3, 4, 5]))
        assert forecaster.lags_names == [f'lag_{i}' for i in range(1, lags + 1)]
        assert forecaster.max_lag == lags
    else:
        assert forecaster.lags is None
        assert forecaster.lags_names is None
        assert forecaster.max_lag is None
    if window_features:
        assert forecaster.window_features_names == ['roll_proportion_5', 'roll_mode_6']
        assert forecaster.window_features_class_names == ['RollingFeaturesClassification']
    else:
        assert forecaster.window_features_names is None
        assert forecaster.window_features_class_names is None
