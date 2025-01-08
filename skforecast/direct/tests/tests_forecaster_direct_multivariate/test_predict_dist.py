# Unit test predict_dist ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.direct import ForecasterDirectMultiVariate
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm

# Fixtures
from .fixtures_forecaster_direct_multivariate import series
from .fixtures_forecaster_direct_multivariate import exog
from .fixtures_forecaster_direct_multivariate import exog_predict

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


def test_predict_dist_TypeError_when_distribution_object_is_not_valid():
    """
    Test TypeError is raise in predict_dist when `distribution` is not a valid
    probability distribution object from scipy.stats.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     steps              = 2,
                     level              = 'l1',
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog)
    
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
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     steps              = 2,
                     level              = 'l1',
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_dist(
                  steps                   = 2,
                  exog                    = exog_predict,
                  distribution            = norm,
                  n_boot                  = 4,
                  use_in_sample_residuals = True
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.53808076, 0.11721631],
                                       [0.32552242, 0.1713172]]),
                   columns = ['loc', 'scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_dist_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_dist when regressor is LinearRegression,
    2 steps are predicted, using out-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     steps              = 2,
                     level              = 'l1',
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    
    forecaster.fit(series=series, exog=exog)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_dist(
                  steps                   = 2,
                  exog                    = exog_predict,
                  distribution            = norm,
                  n_boot                  = 4,
                  use_in_sample_residuals = False
              )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.53808076, 0.11721631],
                                       [0.32552242, 0.1713172]]),
                   columns = ['loc', 'scale'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))

    pd.testing.assert_frame_equal(expected, results)
