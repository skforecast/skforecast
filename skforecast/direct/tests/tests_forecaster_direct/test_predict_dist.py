# Unit test predict_dist ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.direct import ForecasterDirect
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Fixtures
from .fixtures_forecaster_direct import y
from .fixtures_forecaster_direct import exog
from .fixtures_forecaster_direct import exog_predict


def test_predict_dist_TypeError_when_distribution_object_is_not_valid():
    """
    Test TypeError is raise in predict_dist when `distribution` is not a valid
    probability distribution object from scipy.stats.
    """
    forecaster = ForecasterDirect(
                     regressor = LinearRegression(),
                     steps     = 2,
                     lags      = 3
                 )
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
    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     steps            = 2,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
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
                   data    = np.array([[0.542745939120041, 0.20807726416842307],
                                       [0.3204638186856599, 0.13511555588800012]]),
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
    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     steps            = 2,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
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
                   data    = np.array([[0.542745939120041, 0.20807726416842307],
                                       [0.3204638186856599, 0.13511555588800012]]),
                   columns = ['loc', 'scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)
