# Unit test create_sample_weights ForecasterRecursive
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression


def custom_weights(index):  # pragma: no cover
    """
    Return 0 if index is one of '2022-01-05', '2022-01-06', 1 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-05') & (index <= '2022-01-06'),
                   0,
                   1
              )

    return weights


def custom_weights_nan(index):  # pragma: no cover
    """
    Return np.nan if index is one of '2022-01-05', '2022-01-06', 1 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-05') & (index <= '2022-01-06'),
                   np.nan,
                   1
              )

    return weights


def custom_weights_negative(index):  # pragma: no cover
    """
    Return -1 if index is one of '2022-01-05', '2022-01-06', 1 otherwise.
    """
    weights = np.where(
                  (index >= '2022-01-05') & (index <= '2022-01-06'),
                   -1,
                   1
              )
    
    return weights


def custom_weights_zeros(index):  # pragma: no cover
    """
    Return 0 for all elements in index
    """
    weights = np.zeros_like(index, dtype=int)

    return weights


X_train = pd.DataFrame(
              data = np.array(
                          [[ 1.32689708,  1.69928888, -0.04452447],
                           [ 1.74874583,  1.32689708,  1.69928888],
                           [ 0.4900031 ,  1.74874583,  1.32689708],
                           [ 1.08116793,  0.4900031 ,  1.74874583],
                           [ 0.22790434,  1.08116793,  0.4900031 ],
                           [-0.36615834,  0.22790434,  1.08116793],
                           [-0.54452449, -0.36615834,  0.22790434]]
                     ),
               index = pd.DatetimeIndex(
                           ['2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07',
                            '2022-01-08', '2022-01-09', '2022-01-10'],
                           dtype='datetime64[ns]', freq='D'
                       ),
               columns = ['lag_1', 'lag_2', 'lag_3']
          )


def test_create_sample_weights_output():
    """
    Test sample_weights creation.
    """
    forecaster = ForecasterRecursive(
                     lags        = 3,
                     regressor   = LinearRegression(),
                     weight_func = custom_weights
                 )

    expected = np.array([1, 0, 0, 1, 1, 1, 1])
    results = forecaster.create_sample_weights(X_train=X_train)

    assert np.array_equal(results, expected)


def test_create_sample_weights_exceptions_when_weights_has_nan():
    """
    Test sample_weights exception when weights contains NaNs.
    """
    forecaster = ForecasterRecursive(
                     lags        = 3,
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
    forecaster = ForecasterRecursive(
                     lags        = 3,
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
    forecaster = ForecasterRecursive(
                     lags        = 3,
                     regressor   = LinearRegression(),
                     weight_func = custom_weights_zeros
                 )
    
    err_msg = re.escape(
                    ("The resulting `sample_weight` cannot be normalized because "
                     "the sum of the weights is zero.")
                )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.create_sample_weights(X_train=X_train)