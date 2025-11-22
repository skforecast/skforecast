# Unit test _check_categorical_support ForecasterRecursiveClassifier
# ==============================================================================
import pytest
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from skforecast.recursive import ForecasterRecursiveClassifier


@pytest.mark.parametrize("estimator", 
                         [LogisticRegression(), 
                          CalibratedClassifierCV(LogisticRegression())], 
                         ids = lambda estimator: f'estimator: {estimator}')
def test_check_categorical_support_LogisticRegression(estimator):
    """
    Test _check_categorical_support method with LogisticRegression.
    """
    forecaster = ForecasterRecursiveClassifier(
        regressor         = estimator,
        lags              = 3,
        features_encoding = 'auto'
    )

    results = forecaster._check_categorical_support(estimator=forecaster.regressor)
    assert results is False


@pytest.mark.parametrize("estimator", 
                         [LGBMClassifier(), 
                          CalibratedClassifierCV(LGBMClassifier())], 
                         ids = lambda estimator: f'estimator: {estimator}')
def test_check_categorical_support_LGBMClassifier(estimator):
    """
    Test _check_categorical_support method with LGBMClassifier.
    """
    forecaster = ForecasterRecursiveClassifier(
        regressor         = estimator,
        lags              = 3,
        features_encoding = 'auto'
    )

    results = forecaster._check_categorical_support(estimator=forecaster.regressor)
    assert results is True
