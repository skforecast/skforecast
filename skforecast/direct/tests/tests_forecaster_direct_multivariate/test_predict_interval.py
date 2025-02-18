# Unit test predict_interval ForecasterDirectMultiVariate
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


def test_check_interval_ValueError_when_method_is_not_valid_method():
    """
    Check ValueError is raised when `method` is not 'bootstrapping' or 'conformal'.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=3
    )
    forecaster.fit(series=series)

    method = 'not_valid_method'
    err_msg = re.escape(
        f"Invalid `method` '{method}'. Choose 'bootstrapping' or 'conformal'."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(steps=1, method=method)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_interval when regressor is LinearRegression,
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
    results = forecaster.predict_interval(
                  steps                   = 2,
                  exog                    = exog_predict,
                  interval                = [5, 95],
                  n_boot                  = 4,
                  use_in_sample_residuals = True
              )
    expected = pd.DataFrame(
                   data    = np.array([[0.61820497, 0.39855187, 0.67329092],
                                       [0.41314101, 0.20291844, 0.56528096]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_interval when regressor is LinearRegression,
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
    results = forecaster.predict_interval(
                  steps                   = 2,
                  exog                    = exog_predict,
                  interval                = (5, 95),
                  n_boot                  = 4,
                  use_in_sample_residuals = False
              )
    expected = pd.DataFrame(
                   data    = np.array([[0.61820497, 0.39855187, 0.67329092],
                                       [0.41314101, 0.20291844, 0.56528096]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))

    pd.testing.assert_frame_equal(expected, results)
