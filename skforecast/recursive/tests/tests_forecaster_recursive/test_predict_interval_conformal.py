# Unit test _predict_interval_conformal ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.recursive import ForecasterRecursive
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Fixtures
from .fixtures_forecaster_recursive import y


def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=10)
    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=11, step=1)
               )
    results = forecaster._predict_interval_conformal(steps=1, use_in_sample_residuals=True)

    pd.testing.assert_frame_equal(results, expected)

    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=10)
    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.],
                                       [11., 1., 21.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=12, step=1)
               )
    results = forecaster._predict_interval_conformal(steps=2, use_in_sample_residuals=True)

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.out_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=10)
    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=11, step=1)
               )
    results = forecaster._predict_interval_conformal(steps=1, use_in_sample_residuals=False)

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.out_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=10)
    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.],
                                       [11., 1., 21.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=12, step=1)
               )
    results = forecaster._predict_interval_conformal(steps=2, use_in_sample_residuals=False)

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_regressor_is_LinearRegression_with_transform_y():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    forecaster = ForecasterRecursive(
                     regressor     = LinearRegression(),
                     lags          = 5,
                     transformer_y = StandardScaler(),
                     binner_kwargs = {'n_bins': 15}
                 )
    forecaster.fit(y=y)
    predictions = forecaster._predict_interval_conformal(steps=5)

    expected = pd.DataFrame(
                   data = np.array([
                              [-0.1578203 , -1.91279656,  1.59715596],
                              [-0.18459942, -1.93957568,  1.57037684],
                              [-0.13711051, -1.89208677,  1.61786574],
                              [-0.01966358, -1.77463983,  1.73531268],
                              [-0.03228613, -1.78726239,  1.72269012]]),
                   index = pd.RangeIndex(start=20, stop=25, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_conformal_output_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog():
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    y = pd.Series(
            np.array([-0.59, 0.02, -0.9, 1.09, -3.61, 0.72, -0.11, -0.4])
        )
    exog = pd.DataFrame(
               {'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4],
                'col_2': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']}
           )
    exog_predict = exog.copy()
    exog_predict.index = pd.RangeIndex(start=8, stop=16)

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 5,
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster._predict_interval_conformal(steps=5, exog=exog_predict)
    expected = pd.DataFrame(
                   data = np.array([
                              [ 0.50619336,  0.50619336,  0.50619336],
                              [-0.09630298, -0.09630298, -0.09630298],
                              [ 0.05254973,  0.05254973,  0.05254973],
                              [ 0.12281153,  0.12281153,  0.12281153],
                              [ 0.00221741,  0.00221741,  0.00221741]]),
                   index = pd.RangeIndex(start=8, stop=13, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_True_binned_residuals_is_True():
    """
    Test output when regressor is LinearRegression 5 step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y)
    results = forecaster._predict_interval_conformal(
        steps=5, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.56842545, 0.16048717, 0.97636374],
                                 [0.50873285, 0.35161235, 0.66585336],
                                 [0.51189344, 0.15888611, 0.86490076],
                                 [0.51559104, 0.16258371, 0.86859837],
                                 [0.51060927, 0.15760194, 0.8636166 ]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_False_binned_residuals_is_True():
    """
    Test output when regressor is LinearRegression, steps=5, use_in_sample_residuals=False,
    binned_residuals=True.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y)
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_
    results = forecaster._predict_interval_conformal(
        steps=5, nominal_coverage=0.95, use_in_sample_residuals=False, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.56842545, 0.16048717, 0.97636374],
                                 [0.50873285, 0.35161235, 0.66585336],
                                 [0.51189344, 0.15888611, 0.86490076],
                                 [0.51559104, 0.16258371, 0.86859837],
                                 [0.51060927, 0.15760194, 0.8636166 ]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )

    pd.testing.assert_frame_equal(results, expected)
