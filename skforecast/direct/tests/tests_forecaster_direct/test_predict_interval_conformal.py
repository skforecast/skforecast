# Unit test _predict_interval_conformal ForecasterDirect
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from skforecast.direct import ForecasterDirect

# Fixtures
from .fixtures_forecaster_direct import y


def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterDirect(LinearRegression(), steps=2, lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=True)
    forecaster.in_sample_residuals_ = {
        1: np.full_like(forecaster.in_sample_residuals_[1], fill_value=10),
        2: np.full_like(forecaster.in_sample_residuals_[2], fill_value=20)
    }
    results = forecaster._predict_interval_conformal(
        steps=1, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=11, step=1)
               )

    pd.testing.assert_frame_equal(results, expected)

    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_True():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterDirect(LinearRegression(), steps=2, lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=True)
    forecaster.in_sample_residuals_ = {
        1: np.full_like(forecaster.in_sample_residuals_[1], fill_value=10),
        2: np.full_like(forecaster.in_sample_residuals_[2], fill_value=20)
    }
    results = forecaster._predict_interval_conformal(
        steps=2, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.],
                                       [11., -9., 31.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=12, step=1)
               )

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_1_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and one step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterDirect(LinearRegression(), steps=2, lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = {
        1: np.full_like(forecaster.in_sample_residuals_[1], fill_value=10),
        2: np.full_like(forecaster.in_sample_residuals_[2], fill_value=20)
    }
    results = forecaster._predict_interval_conformal(
        steps=1, nominal_coverage=0.95, use_in_sample_residuals=False, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=11, step=1)
               )

    pd.testing.assert_frame_equal(results, expected)
    
    
def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_is_False():
    """
    Test output when regressor is LinearRegression and two step ahead is predicted
    using out sample residuals.
    """
    forecaster = ForecasterDirect(LinearRegression(), steps=2, lags=3)
    forecaster.fit(y=pd.Series(np.arange(10)), store_in_sample_residuals=True)
    forecaster.out_sample_residuals_ = {
        1: np.full_like(forecaster.in_sample_residuals_[1], fill_value=10),
        2: np.full_like(forecaster.in_sample_residuals_[2], fill_value=20)
    }
    results = forecaster._predict_interval_conformal(
        steps=2, nominal_coverage=0.95, use_in_sample_residuals=False, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data    = np.array([[10., 0., 20.],
                                       [11., -9., 31.]]),
                   columns = ['pred', 'lower_bound', 'upper_bound'],
                   index   = pd.RangeIndex(start=10, stop=12, step=1)
               )

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
    forecaster = ForecasterDirect(
                     regressor     = LinearRegression(),
                     steps         = 3,
                     lags          = 3,
                     transformer_y = StandardScaler(),
                     binner_kwargs = {'n_bins': 4}
                 )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [-0.07720596, -2.24497965,  2.09056772],
                              [-0.54638907, -2.98792915,  1.895151  ],
                              [-0.08892596, -1.8478775 ,  1.67002558]]),
                   index = pd.RangeIndex(start=20, stop=23, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_regressor_is_LinearRegression_with_transform_y_and_transform_exog():
    """
    Test predict output when using LinearRegression as regressor, StandardScaler
    as transformer_y and transformer_exog as transformer_exog.
    """
    y = pd.Series(
            np.array([-0.59,  0.02, -0.9 ,  1.09, -3.61,  0.72, -0.11, -0.4 ,  0.49,
                       0.67,  0.54, -0.17,  0.54,  1.49, -2.26, -0.41, -0.64, -0.8 ,
                      -0.61, -0.88])
        )
    exog = pd.DataFrame(
               {'col_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 36.5,
                          7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 87.2, 47.4, 36.5,
                          7.5, 24.4],
                'col_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b',
                          'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']}
           )
    exog_predict = exog.copy()
    exog_predict.index = pd.RangeIndex(start=20, stop=40)

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['col_1']),
                            ('onehot', OneHotEncoder(), ['col_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    forecaster = ForecasterDirect(
                     regressor        = LinearRegression(),
                     steps            = 5,
                     lags             = 3,
                     transformer_y    = StandardScaler(),
                     transformer_exog = transformer_exog
                 )
    forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        steps=None, nominal_coverage=0.95, exog=exog_predict,
        use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [ 1.33676517, -0.95182882,  3.62535915],
                              [-1.05138096, -3.25578143,  1.15301951],
                              [ 0.55115225, -1.269978  ,  2.37228251],
                              [ 0.86985865, -0.78370515,  2.52342245],
                              [ 0.44787213, -1.15612139,  2.05186566]]),
                   index = pd.RangeIndex(start=20, stop=25, step=1),
                   columns = ['pred', 'lower_bound', 'upper_bound']
               )
    
    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_when_forecaster_is_LinearRegression_steps_is_5_in_sample_residuals_is_True_binned_residuals_is_True():
    """
    Test output when regressor is LinearRegression 5 step ahead is predicted
    using in sample residuals.
    """
    forecaster = ForecasterDirect(LinearRegression(), steps=5, lags=3)
    forecaster.fit(y=y, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        steps=5, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[ 0.51883519,  0.08813429,  0.94953609],
                                 [ 0.4584716 ,  0.04326184,  0.87368136],
                                 [ 0.39962743, -0.04758052,  0.84683537],
                                 [ 0.40452904, -0.0426789 ,  0.85173698],
                                 [ 0.41534557, -0.03186238,  0.86255351]]
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
    forecaster = ForecasterDirect(LinearRegression(), steps=5, lags=3)
    forecaster.fit(y=y, store_in_sample_residuals=True)
    forecaster.out_sample_residuals_by_bin_ = forecaster.in_sample_residuals_by_bin_
    results = forecaster._predict_interval_conformal(
        steps=5, nominal_coverage=0.95, use_in_sample_residuals=False, use_binned_residuals=True
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[ 0.51883519,  0.08813429,  0.94953609],
                                 [ 0.4584716 ,  0.04326184,  0.87368136],
                                 [ 0.39962743, -0.04758052,  0.84683537],
                                 [ 0.40452904, -0.0426789 ,  0.85173698],
                                 [ 0.41534557, -0.03186238,  0.86255351]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )

    pd.testing.assert_frame_equal(results, expected)


def test_predict_interval_conformal_output_with_differentiation():
    """
    Test predict output when using differentiation.
    """
    forecaster = ForecasterDirect(
        LinearRegression(), steps=5, lags=3, differentiation=1
    )
    forecaster.fit(y=y, store_in_sample_residuals=True)
    results = forecaster._predict_interval_conformal(
        steps=5, nominal_coverage=0.95, use_in_sample_residuals=True, use_binned_residuals=False
    )

    expected = pd.DataFrame(
                    data    = np.array(
                                [[0.65659084,  0.17710854,  1.13607314],
                                 [0.6496844 , -0.40541691,  1.7047857 ],
                                 [0.61632235, -0.98830203,  2.22094673],
                                 [0.59545137, -1.53828123,  2.72918398],
                                 [0.64917089, -2.01927988,  3.31762167]]
                            ),
                    columns = ['pred', 'lower_bound', 'upper_bound'],
                    index   = pd.RangeIndex(start=50, stop=55, step=1)
                )

    pd.testing.assert_frame_equal(results, expected)
