# Unit test set_params ForecasterRecursiveClassifier
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from skforecast.recursive import ForecasterRecursiveClassifier


def test_set_params():
    """
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
