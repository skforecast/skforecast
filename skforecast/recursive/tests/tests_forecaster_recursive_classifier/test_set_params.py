# Unit test set_params ForecasterRecursiveClassifier
# ==============================================================================
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from skforecast.recursive import ForecasterRecursiveClassifier
from packaging import version


@pytest.mark.skipif(version.parse(sklearn.__version__) >= version.parse("1.5"), 
                    reason="Requires scikit-learn < 1.5")
def test_set_params_sklearn_less_than_1_5():
    """
    Test set_params with scikit-learn < 1.5 (multi_class parameter present).
    """
    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(penalty='l1'), lags=3
    )
    new_params = {'penalty': 'elasticnet'}
    forecaster.set_params(new_params)
    expected = {
        'C': 1.0,
        'class_weight': None,
        'dual': False,
        'fit_intercept': True,
        'intercept_scaling': 1,
        'l1_ratio': None,
        'max_iter': 100,
        'multi_class': 'deprecated',
        'n_jobs': None,
        'penalty': 'elasticnet',
        'random_state': None,
        'solver': 'lbfgs',
        'tol': 0.0001,
        'verbose': 0,
        'warm_start': False
    }

    results = forecaster.estimator.get_params()
    
    assert results == expected


@pytest.mark.skipif(version.parse(sklearn.__version__) < version.parse("1.5"), 
                    reason="Requires scikit-learn >= 1.5")
def test_set_params_sklearn_1_5_or_greater():
    """
    Test set_params with scikit-learn >= 1.5 (multi_class parameter removed, 
    l1_ratio defaults to 0.0 when penalty='elasticnet').
    """
    forecaster = ForecasterRecursiveClassifier(
        LogisticRegression(penalty='l1'), lags=3
    )
    new_params = {'penalty': 'elasticnet'}
    forecaster.set_params(new_params)
    expected = {
        'C': 1.0,
        'class_weight': None,
        'dual': False,
        'fit_intercept': True,
        'intercept_scaling': 1,
        'l1_ratio': 0.0,
        'max_iter': 100,
        'n_jobs': None,
        'penalty': 'elasticnet',
        'random_state': None,
        'solver': 'lbfgs',
        'tol': 0.0001,
        'verbose': 0,
        'warm_start': False
    }

    results = forecaster.estimator.get_params()
    
    assert results == expected
