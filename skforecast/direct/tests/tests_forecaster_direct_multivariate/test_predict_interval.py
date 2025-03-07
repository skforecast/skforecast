# Unit test predict_interval ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from skforecast.exceptions import ResidualsUsageWarning
from skforecast.direct import ForecasterDirectMultiVariate

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
    forecaster.fit(series=series, store_in_sample_residuals=True)

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
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 2,
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    
    n_boot = 250
    recommended_n_boot = np.max([len(v) for v in forecaster.in_sample_residuals_.values()])
    warn_msg = re.escape(
        f"`n_boot`, {n_boot}, is greater than the number of available "
        f"residuals. More than {recommended_n_boot} iterations don't "
        f"add new information to the bootstrapping process, but increase "
        f"the computational cost."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        results = forecaster.predict_interval(
                      steps                   = 2,
                      exog                    = exog_predict,
                      method                  = 'bootstrapping',
                      interval                = interval,
                      n_boot                  = n_boot,
                      use_in_sample_residuals = True,
                      use_binned_residuals    = False
                  )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.61820497, 0.2246467 , 0.89301964],
                                       [0.41314101, 0.07063188, 0.76618886]]),
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
                     level              = 'l1',
                     steps              = 2,
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    
    n_boot = 250
    recommended_n_boot = np.max([len(v) for v in forecaster.in_sample_residuals_.values()])
    warn_msg = re.escape(
        f"`n_boot`, {n_boot}, is greater than the number of available "
        f"residuals. More than {recommended_n_boot} iterations don't "
        f"add new information to the bootstrapping process, but increase "
        f"the computational cost."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        results = forecaster.predict_interval(
                      steps                   = 2,
                      exog                    = exog_predict,
                      method                  = 'bootstrapping',
                      interval                = [5, 95],
                      n_boot                  = n_boot,
                      use_in_sample_residuals = False,
                      use_binned_residuals    = False
                  )
    
    expected = pd.DataFrame(
                   data    = np.array([[0.61820497, 0.2246467 , 0.89301964],
                                       [0.41314101, 0.07063188, 0.76618886]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    expected.insert(0, 'level', np.tile(['l1'], 2))

    pd.testing.assert_frame_equal(expected, results)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_True_binned_residuals_is_True():
    """
    Test output when regressor is LinearRegression 5 step ahead is predicted
    using in sample binned residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 5,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)

    recommended_n_boot = np.max([len(v) for v in forecaster.in_sample_residuals_by_bin_.values()])
    warn_msg = re.escape(
        f"`n_boot`, 250, is greater than the number of available "
        f"residuals. More than {recommended_n_boot} iterations don't "
        f"add new information to the bootstrapping process, but increase "
        f"the computational cost."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        results = forecaster.predict_interval(
            steps=5, method='bootstrapping', interval=(5, 95), 
            use_in_sample_residuals=True, use_binned_residuals=True
        )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.58307704, 0.30689592, 0.97766641],
                                 [0.40064856, 0.04683134, 0.63486761],
                                 [0.29394488, 0.13006777, 0.50853662],
                                 [0.41007329, 0.0785547 , 0.85997779],
                                 [0.4390632 , 0.10298379, 0.83922684]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_False_binned_residuals_is_True():
    """
    Test output when regressor is LinearRegression, steps=5, use_in_sample_residuals=False,
    binned_residuals=True.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 5,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_

    recommended_n_boot = np.max([len(v) for v in forecaster.out_sample_residuals_by_bin_.values()])
    warn_msg = re.escape(
        f"`n_boot`, 250, is greater than the number of available "
        f"residuals. More than {recommended_n_boot} iterations don't "
        f"add new information to the bootstrapping process, but increase "
        f"the computational cost."
    )
    with pytest.warns(ResidualsUsageWarning, match=warn_msg):
        results = forecaster.predict_interval(
            steps=5, method='bootstrapping', interval=(5, 95), 
            use_in_sample_residuals=False, use_binned_residuals=True
        )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.58307704, 0.30689592, 0.97766641],
                                 [0.40064856, 0.04683134, 0.63486761],
                                 [0.29394488, 0.13006777, 0.50853662],
                                 [0.41007329, 0.0785547 , 0.85997779],
                                 [0.4390632 , 0.10298379, 0.83922684]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("interval", 
                         [0.95, (2.5, 97.5)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_conformal_output_when_regressor_is_LinearRegression(interval):
    """
    Test predict output when using LinearRegression as regressor and StandardScaler
    and conformal prediction.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=False)
    forecaster.set_in_sample_residuals(series=series)
    results = forecaster.predict_interval(
        steps=3, method='conformal', interval=interval, 
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.63114259,  0.23482357,  1.02746162],
                              [0.3800417 , -0.05044528,  0.81052868],
                              [0.33255977, -0.0626732 ,  0.72779273]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))
    
    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("interval", 
                         [0.95, (2.5, 97.5)], 
                         ids = lambda value: f'interval: {value}')
def test_predict_interval_conformal_output_when_binned_residuals(interval):
    """
    Test predict output when using LinearRegression as regressor and StandardScaler
    and conformal prediction with binned residuals.
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = LinearRegression(),
                     level              = 'l1',
                     steps              = 3,
                     lags               = 3,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster.predict_interval(
        steps=3, method='conformal', interval=interval, 
        use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.63114259, 0.17603311, 1.08625208],
                              [0.3800417 , 0.12832655, 0.63175685],
                              [0.33255977, 0.08084462, 0.58427492]]),
                   index = pd.RangeIndex(start=50, stop=53, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    expected.insert(0, 'level', np.tile(['l1'], forecaster.steps))
    
    pd.testing.assert_frame_equal(results, expected)
