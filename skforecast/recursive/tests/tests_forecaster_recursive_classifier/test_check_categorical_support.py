# Unit test _check_categorical_support ForecasterRecursiveClassifier
# ==============================================================================
import pytest
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
        estimator         = estimator,
        lags              = 3,
        features_encoding = 'auto'
    )

    results = forecaster._check_categorical_support(estimator=forecaster.estimator)
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
        estimator         = estimator,
        lags              = 3,
        features_encoding = 'auto'
    )

    results = forecaster._check_categorical_support(estimator=forecaster.estimator)
    assert results is True


@pytest.mark.parametrize("estimator",
                         [make_pipeline(StandardScaler(), LGBMClassifier()),
                          make_pipeline(
                              StandardScaler(), CalibratedClassifierCV(LGBMClassifier())
                          ),
                          CalibratedClassifierCV(
                              make_pipeline(StandardScaler(), LGBMClassifier())
                          )],
                         ids = ['Pipeline(LGBMClassifier)',
                                'Pipeline(CalibratedClassifierCV(LGBMClassifier))',
                                'CalibratedClassifierCV(Pipeline(LGBMClassifier))'])
def test_check_categorical_support_Pipeline(estimator):
    """
    Test _check_categorical_support method returns False when the estimator is
    a Pipeline, or wraps one, even if LGBMClassifier supports categorical
    features natively: they are not configured inside a Pipeline.
    """
    forecaster = ForecasterRecursiveClassifier(
        estimator         = estimator,
        lags              = 3,
        features_encoding = 'auto'
    )

    results = forecaster._check_categorical_support(estimator=forecaster.estimator)
    assert results is False
    assert forecaster.use_native_categoricals is False
