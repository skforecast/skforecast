# Unit test set_fit_kwargs ForecasterRecursiveClassifier
# ==============================================================================
from lightgbm import LGBMClassifier
from skforecast.recursive import ForecasterRecursiveClassifier


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
    """
    forecaster = ForecasterRecursiveClassifier(
                     estimator  = LGBMClassifier(verbose=-1),
                     lags       = 3,
                     fit_kwargs = {'categorical_feature': 'auto'}
                 )
    
    new_fit_kwargs = {'categorical_feature': ['exog']}
    forecaster.set_fit_kwargs(new_fit_kwargs)
    results = forecaster.fit_kwargs

    expected = {'categorical_feature': ['exog']}

    assert results == expected
