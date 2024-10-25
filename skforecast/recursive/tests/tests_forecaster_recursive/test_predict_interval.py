# Unit test predict_interval ForecasterRecursive
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



def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=10)
    expected = pd.DataFrame(
                   data    = np.array([[10., 20., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=11, step=1)
               )
    results = forecaster.predict_interval(steps=1, use_in_sample_residuals=True)

    pd.testing.assert_frame_equal(results, expected)

    
def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=10)
    expected = pd.DataFrame(
                   data    = np.array([[10. ,20., 20.],
                                       [11., 24.33333333, 24.33333333]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=12, step=1)
               )
    results = forecaster.predict_interval(steps=2, use_in_sample_residuals=True)

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.out_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=10)
    expected = pd.DataFrame(
                   data    = np.array([[10., 20., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=11, step=1)
               )
    results = forecaster.predict_interval(steps=1, use_in_sample_residuals=False)

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_True_binned_residuals_is_True():
    """
    Test output when regressor is LinearRegression 5 step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y)
    results = forecaster.predict_interval(
        steps=5, use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.56842545, 0.25304752, 0.98664822],
                                [0.50873285, 0.05687183, 0.95641577],
                                [0.51189344, 0.23219067, 0.9780938 ],
                                [0.51559104, 0.16866501, 0.97647946],
                                [0.51060927, 0.12764843, 0.9697876 ]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.out_sample_residuals_ = np.full_like(forecaster.in_sample_residuals_, fill_value=10)
    expected = pd.DataFrame(
                   data    = np.array([[10., 20., 20.],
                                       [11., 24.33333333, 24.33333333]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=12, step=1)
               )
    results = forecaster.predict_interval(steps=2, use_in_sample_residuals=False)

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_y():
    """
    Test predict output when using LinearRegression as regressor and StandardScaler.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    transformer_y = StandardScaler()
    forecaster = ForecasterRecursive(
                     regressor     = LinearRegression(),
                     lags          = 5,
                     transformer_y = transformer_y,
                     binner_kwargs = {'n_bins': 15}
                 )
    forecaster.fit(y=y)
    predictions = forecaster.predict_interval(steps=5)

    expected = pd.DataFrame(
                   data = np.array([[-0.1578203 , -2.02104471,  1.55076389],
                                    [-0.18459942, -1.98567931,  1.49673592],
                                    [-0.13711051, -1.89299245,  1.47330827],
                                    [-0.01966358, -1.60143134,  1.59908257],
                                    [-0.03228613, -1.73050679,  1.51878244]]),
                   index = pd.RangeIndex(start=20, stop=25, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog():
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

    transformer_y = StandardScaler()
    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['col_1']),
                             ('onehot', OneHotEncoder(), ['col_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )
    forecaster = ForecasterRecursive(
                     regressor        = LinearRegression(),
                     lags             = 5,
                     transformer_y    = transformer_y,
                     transformer_exog = transformer_exog
                 )
    
    forecaster.fit(y=y, exog=exog)
    predictions = forecaster.predict_interval(steps=5, exog=exog_predict)
    expected = pd.DataFrame(
                   data = np.array([[ 0.50619336,  0.50619336,  0.50619336],
                                    [-0.09630298, -0.09630298, -0.09630298],
                                    [ 0.05254973,  0.05254973,  0.05254973],
                                    [ 0.12281153,  0.12281153,  0.12281153],
                                    [ 0.00221741,  0.00221741,  0.00221741]]),
                   index = pd.RangeIndex(start=8, stop=13, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(predictions, expected)


def test_predict_interval_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_False_binned_residuals_is_True():
    """
    Test output when regressor is LinearRegression, steps=5, use_in_sample_residuals=False,
    binned_residuals=True.
    """
    forecaster = ForecasterRecursive(LinearRegression(), lags=3, binner_kwargs={'n_bins': 15})
    forecaster.fit(y=y)
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_
    results = forecaster.predict_interval(
        steps=5, use_in_sample_residuals=False, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                   data    = np.array(
                                [[0.56842545, 0.25304752, 0.98664822],
                                 [0.50873285, 0.05687183, 0.95641577],
                                 [0.51189344, 0.23219067, 0.9780938],
                                 [0.51559104, 0.16866501, 0.97647946],
                                 [0.51060927, 0.12764843, 0.9697876]]
                            ),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=50, stop=55, step=1)
               )
    
    pd.testing.assert_frame_equal(results, expected)
