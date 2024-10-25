# Unit test create_sample_weights ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.direct import ForecasterDirectMultiVariate
from sklearn.linear_model import LinearRegression


def custom_weights(index): # pragma: no cover
    """
    Return 0 if index is between '2022-01-10', '2022-01-12', 1 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-10') & (index <= '2022-01-12'),
                   0,
                   1
              )
    
    return weights


def custom_weights_nan(index): # pragma: no cover
    """
    Return np.nan f index is between '2022-01-10', '2022-01-12', 1 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-10') & (index <= '2022-01-12'),
                   np.nan,
                   1
              )
    
    return weights


def custom_weights_negative(index): # pragma: no cover
    """
    Return -1 f index is between '2022-01-10', '2022-01-12', 1 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-10') & (index <= '2022-01-12'),
                   -1,
                   1
              )
    
    return weights


def custom_weights_zeros(index): # pragma: no cover
    """
    Return 0 for all elements in index
    """
    weights = np.zeros_like(index, dtype=int)

    return weights


series = pd.DataFrame(
             data = np.array([[0.12362923, 0.51328688],
                              [0.65138268, 0.11599708],
                              [0.58142898, 0.72350895],
                              [0.72969992, 0.10305721],
                              [0.97790567, 0.20581485],
                              [0.56924731, 0.41262027],
                              [0.85369084, 0.82107767],
                              [0.75425194, 0.0107816 ],
                              [0.08167939, 0.94951918],
                              [0.00249297, 0.55583355]]),
             columns= ['series_1', 'series_2'],
             index = pd.DatetimeIndex(
                         ['2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07',
                          '2022-01-08', '2022-01-09', '2022-01-10', '2022-01-11',
                          '2022-01-12', '2022-01-13'],
                         dtype='datetime64[ns]', freq='D'
                     ),
         )

# X_train for 3 lags and 3 steps
X_train = pd.DataFrame(
              data = np.array(
                          [[0.58142898, 0.65138268, 0.12362923, 0.72350895, 0.11599708, 0.51328688],
                           [0.72969992, 0.58142898, 0.65138268, 0.10305721, 0.72350895, 0.11599708],  
                           [0.97790567, 0.72969992, 0.58142898, 0.20581485, 0.10305721, 0.72350895],
                           [0.56924731, 0.97790567, 0.72969992, 0.41262027, 0.20581485, 0.10305721],
                           [0.85369084, 0.56924731, 0.97790567, 0.82107767, 0.41262027, 0.20581485]]
                     ),
               index = pd.DatetimeIndex(
                           ['2022-01-09', '2022-01-10', '2022-01-11', '2022-01-12', '2022-01-13'],
                           dtype='datetime64[ns]', freq='D'
                       ),
               columns = ['series_1_lag_1', 'series_1_lag_2', 'series_1_lag_3', 
                          'series_2_lag_1', 'series_2_lag_2', 'series_2_lag_3']
          )


def test_create_sample_weights_output():
    """
    Test sample_weights creation.
    """
    forecaster = ForecasterDirectMultiVariate(
                     level       = 'series_1',
                     lags        = 3,
                     steps       = 3,    
                     regressor   = LinearRegression(),
                     weight_func = custom_weights
                 )

    expected = np.array([1, 0, 0, 0, 1])
    results = forecaster.create_sample_weights(X_train=X_train)

    assert np.array_equal(results, expected)


def test_create_sample_weights_exceptions_when_weights_has_nan():
    """
    Test sample_weights exception when weights contains NaNs.
    """
    forecaster = ForecasterDirectMultiVariate(
                     level       = 'series_1',
                     lags        = 3,
                     steps       = 3,  
                     regressor   = LinearRegression(),
                     weight_func = custom_weights_nan
                 )

    err_msg = re.escape("The resulting `sample_weight` cannot have NaN values.")
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(X_train=X_train)
    

def test_create_sample_weights_exceptions_when_weights_has_negative_values():
    """
    Test sample_weights exception when sample_weight contains negative values.
    """
    forecaster = ForecasterDirectMultiVariate(
                     level       = 'series_1',
                     lags        = 3,
                     steps       = 3,     
                     regressor   = LinearRegression(),
                     weight_func = custom_weights_negative
                 )

    err_msg = re.escape("The resulting `sample_weight` cannot have negative values.")
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(X_train=X_train)
    

def test_create_sample_weights_exceptions_when_weights_all_zeros():
    """
    Test sample_weights exception when all weights are zeros.
    """
    forecaster = ForecasterDirectMultiVariate(
                     level       = 'series_1',
                     lags        = 3,
                     steps       = 3,    
                     regressor   = LinearRegression(),
                     weight_func = custom_weights_zeros
                 )
    
    err_msg = re.escape(
                    ("The resulting `sample_weight` cannot be normalized because "
                     "the sum of the weights is zero.")
                )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(X_train=X_train)