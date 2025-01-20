# Unit test predict_dist ForecasterRecursive
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursive
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Fixtures
from .fixtures_forecaster_recursive import y
from .fixtures_forecaster_recursive import exog
from .fixtures_forecaster_recursive import exog_predict


def test_predict_dist_TypeError_when_distribution_object_is_not_valid():
    """
    Test TypeError is raise in predict_dist when `distribution` is not a valid
    probability distribution object from scipy.stats.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=y, exog=exog)
    
    class CustomObject:  # pragma: no cover
        pass
    
    err_msg = re.escape(
        "`distribution` must be a valid probability distribution object "
        "from scipy.stats, with methods `_pdf` and `fit`."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.predict_dist(
            steps                   = 2,
            exog                    = exog_predict,
            distribution            = CustomObject(),
            n_boot                  = 4,
            use_in_sample_residuals = True
        )


def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_dist when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                     binner_kwargs    = {'n_bins': 15}
                 )

    forecaster.fit(y=y, exog=exog)
    results = forecaster.predict_dist(
                  steps                   = 2,
                  exog                    = exog_predict,
                  distribution            = norm,
                  n_boot                  = 4,
                  use_in_sample_residuals = True
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.61669254, 0.15680479],
                                       [0.38256026, 0.29497167]]),
                   columns = ['loc', 'scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_dist when regressor is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                     binner_kwargs    = {'n_bins': 15}
                 )
    
    forecaster.fit(y=y, exog=exog)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_dist(
                  steps                   = 2,
                  exog                    = exog_predict,
                  distribution            = norm,
                  n_boot                  = 4,
                  use_in_sample_residuals = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.61669254, 0.15680479],
                                       [0.38256026, 0.29497167]]),
                   columns = ['loc', 'scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)
