# Unit test set_fit_kwargs ForecasterRecursiveMultiSeries
# ==============================================================================
from lightgbm import LGBMRegressor
from ....recursive import ForecasterRecursiveMultiSeries


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor  = LGBMRegressor(),
                     lags       = 3,
                     fit_kwargs = {'categorical_feature': 'auto'}
                 )
    
    new_fit_kwargs = {'categorical_feature': ['exog']}
    forecaster.set_fit_kwargs(new_fit_kwargs)
    results = forecaster.fit_kwargs

    expected = {'categorical_feature': ['exog']}

    assert results == expected