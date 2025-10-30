# Unit test _check_categorical_support ForecasterRecursiveClassifier
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from skforecast.recursive import ForecasterRecursiveClassifier


def test_check_categorical_support_LogisticRegression():
    """
    Test _check_categorical_support method with LogisticRegression.
    """
    forecaster = ForecasterRecursiveClassifier(
        regressor         = LogisticRegression(),
        lags              = 3,
        features_encoding = 'auto'
    )

    results = forecaster._check_categorical_support(regressor=forecaster.regressor)
    assert results is False


def test_check_categorical_support_LGBMClassifier():
    """
    Test _check_categorical_support method with LGBMClassifier.
    """
    forecaster = ForecasterRecursiveClassifier(
        regressor         = LGBMClassifier(),
        lags              = 3,
        features_encoding = 'auto'
    )

    results = forecaster._check_categorical_support(regressor=forecaster.regressor)
    assert results is True
