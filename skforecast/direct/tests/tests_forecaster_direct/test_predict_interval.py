# Unit test predict_interval ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.direct import ForecasterDirect
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Fixtures
from ....exceptions import ResidualsUsageWarning
from .fixtures_forecaster_direct import y
from .fixtures_forecaster_direct import exog
from .fixtures_forecaster_direct import exog_predict


def test_check_interval_ValueError_when_method_is_not_valid_method():
    """
    Check ValueError is raised when `method` is not 'bootstrapping' or 'conformal'.
    """
    forecaster = ForecasterDirect(LinearRegression(), steps=2, lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))

    method = 'not_valid_method'
    err_msg = re.escape(
        f"Invalid `method` '{method}'. Choose 'bootstrapping' or 'conformal'."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_interval(steps=1, method=method)


@pytest.mark.parametrize("interval", 
                         [0.90, [5, 95], (5, 95)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer(interval):
    """
    Test output of predict_interval when regressor is LinearRegression,
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
    
    n_boot = 250
    recommended_n_boot = np.max([len(v) for v in forecaster.in_sample_residuals_.values()])
    warn_msg = re.escape(
        f"`n_boot`, {n_boot}, is greater than the number of available "
        f"residuals, {recommended_n_boot}. Additional iterations don't "
        f"add new information to the bootstrapping process, but increase "
        f"the computational cost."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        results = forecaster.predict_interval(
                      steps                   = 2,
                      interval                = interval,
                      exog                    = exog_predict,
                      n_boot                  = n_boot,
                      use_in_sample_residuals = True
                  )
    
    expected = pd.DataFrame(
                   data    = np.array([[6.75235879e-01, 3.09660082e-01, 1.07596889e+00],
                                       [3.80249878e-01, 7.09041120e-04, 7.82616124e-01]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_interval when regressor is LinearRegression,
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
    
    n_boot = 250
    recommended_n_boot = np.max([len(v) for v in forecaster.in_sample_residuals_.values()])
    warn_msg = re.escape(
        f"`n_boot`, {n_boot}, is greater than the number of available "
        f"residuals, {recommended_n_boot}. Additional iterations don't "
        f"add new information to the bootstrapping process, but increase "
        f"the computational cost."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        results = forecaster.predict_interval(
                      steps                   = 2,
                      interval                = (5, 95),
                      exog                    = exog_predict,
                      n_boot                  = n_boot,
                      use_in_sample_residuals = False
                  )
    
    expected = pd.DataFrame(
                   data    = np.array([[6.75235879e-01, 3.09660082e-01, 1.07596889e+00],
                                       [3.80249878e-01, 7.09041120e-04, 7.82616124e-01]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )

    pd.testing.assert_frame_equal(expected, results)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_True_binned_residuals_is_True():
    """
    Test output when regressor is LinearRegression 5 step ahead is predicted
    using in sample binned residuals.
    """
    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     steps            = 5,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y)

    recommended_n_boot = np.max([len(v) for v in forecaster.in_sample_residuals_by_bin_.values()])
    warn_msg = re.escape(
        f"`n_boot`, 250, is greater than the total number of "
        f"residuals, {recommended_n_boot}. Additional iterations don't "
        f"add new information to the bootstrapping process, but increase "
        f"the computational cost."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        results = forecaster.predict_interval(
            steps=5, interval=(5, 95), use_in_sample_residuals=True, use_binned_residuals=True
        )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.51883519, 0.11786323, 0.96698339],
                                 [0.4584716 , 0.04248806, 0.85897968],
                                 [0.39962743, 0.16612305, 0.91127372],
                                 [0.40452904, 0.17102467, 0.85708924],
                                 [0.41534557, 0.07605488, 0.92699186]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_False_binned_residuals_is_True():
    """
    Test output when regressor is LinearRegression, steps=5, use_in_sample_residuals=False,
    binned_residuals=True.
    """
    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     steps            = 5,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = StandardScaler(),
                 )
    forecaster.fit(y=y)
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_

    recommended_n_boot = np.max([len(v) for v in forecaster.out_sample_residuals_by_bin_.values()])
    warn_msg = re.escape(
        f"`n_boot`, 250, is greater than the total number of "
        f"residuals, {recommended_n_boot}. Additional iterations don't "
        f"add new information to the bootstrapping process, but increase "
        f"the computational cost."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        results = forecaster.predict_interval(
            steps=5, interval=(5, 95), use_in_sample_residuals=False, use_binned_residuals=True
        )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.51883519, 0.11786323, 0.96698339],
                                 [0.4584716 , 0.04248806, 0.85897968],
                                 [0.39962743, 0.16612305, 0.91127372],
                                 [0.40452904, 0.17102467, 0.85708924],
                                 [0.41534557, 0.07605488, 0.92699186]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("interval", 
                         [0.95, (2.5, 97.5)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_conformal_output_when_regressor_is_LinearRegression_with_transform_y(interval):
    """
    Test predict output when using LinearRegression as regressor and StandardScaler
    and conformal prediction.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    forecaster = ForecasterDirect(
                     regressor     = LinearRegression(),
                     steps         = 3,
                     lags          = 3,
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    results = forecaster.predict_interval(
        steps=3, method='conformal', interval=interval
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [-0.07720596, -1.83615751,  1.68174558],
                              [-0.54638907, -2.30534062,  1.21256247],
                              [-0.08892596, -1.8478775 ,  1.67002558]]),
                   index = pd.RangeIndex(start=20, stop=25, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("interval", 
                         [0.95, (2.5, 97.5)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_conformal_output_when_binned_residuals(interval):
    """
    Test predict output when using LinearRegression as regressor and StandardScaler
    and conformal prediction with binned residuals.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    forecaster = ForecasterDirect(
                     regressor     = LinearRegression(),
                     steps         = 3,
                     lags          = 3,
                     transformer_y = StandardScaler()
                 )
    forecaster.fit(y=y)
    results = forecaster.predict_interval(
        steps=3, method='conformal', interval=interval, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [-0.07720596, -1.96803865,  1.81362673],
                              [-0.54638907, -3.1153822 ,  2.02260406],
                              [-0.08892596, -1.97975865,  1.80190673]]),
                   index = pd.RangeIndex(start=20, stop=25, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(results, expected)
