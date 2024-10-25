# Unit test predict_bootstrapping ForecasterRecursiveMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.exceptions import UnknownLevelWarning
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries

# Fixtures
from .fixtures_forecaster_recursive_multiseries import series
from .fixtures_forecaster_recursive_multiseries import exog
from .fixtures_forecaster_recursive_multiseries import exog_predict

transformer_exog = ColumnTransformer(
                       [('scale', StandardScaler(), ['exog_1']),
                        ('onehot', OneHotEncoder(), ['exog_2'])],
                       remainder = 'passthrough',
                       verbose_feature_names_out = False
                   )


def test_predict_bootstrapping_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, levels=None)


def test_predict_bootstrapping_IgnoredArgumentWarning_when_not_available_self_last_window_for_some_levels():
    """
    Test IgnoredArgumentWarning is raised when last_window is not available for 
    levels because it was not stored during fit.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'], store_last_window=['1'])

    warn_msg = re.escape(
        ("Levels {'2'} are excluded from prediction "
         "since they were not stored in `last_window_` attribute "
         "during training. If you don't want to retrain the "
         "Forecaster, provide `last_window` as argument.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        results = forecaster.predict_bootstrapping(
                      steps                   = 1,
                      levels                  = ['1', '2'],
                      n_boot                  = 4, 
                      exog                    = exog_predict['exog_1'], 
                      use_in_sample_residuals = True
                  )

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.48128438, 0.20172898, 0.07381993, 0.38819631]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )

    expected = {'1': expected_1}

    pd.testing.assert_frame_equal(results['1'], expected['1'])


@pytest.mark.parametrize("store_last_window",
                         [['1'], False],
                         ids=lambda slw: f"store_last_window: {slw}")
def test_predict_bootstrapping_ValueError_when_not_available_self_last_window_for_levels(store_last_window):
    """
    Test ValueError is raised when last_window is not available for all 
    levels because it was not stored during fit.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=5)
    forecaster.fit(series=series, store_last_window=store_last_window)

    err_msg = re.escape(
        ("No series to predict. None of the series {'2'} are present in "
         "`last_window_` attribute. Provide `last_window` as argument "
         "in predict method.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=5, levels=['2'], last_window=None)


def test_predict_bootstrapping_IgnoredArgumentWarning_when_levels_is_list_and_different_last_index_in_self_last_window_DatetimeIndex():
    """
    Test IgnoredArgumentWarning is raised when levels is a list and have 
    different last index in last_window attribute using a DatetimeIndex.
    """
    series_2 = {
        '1': series['1'].copy(),
        '2': series['2'].iloc[:30].copy()
    }
    series_2['1'].index = pd.date_range(start='2020-01-01', periods=50)
    series_2['2'].index = pd.date_range(start='2020-01-01', periods=30)
    exog_2 = {
        '1': exog['exog_1'].copy(),
        '2': exog['exog_1'].iloc[:30].copy()
    }
    exog_2['1'].index = pd.date_range(start='2020-01-01', periods=50)
    exog_2['2'].index = pd.date_range(start='2020-01-01', periods=30)
    exog_2_pred = {
        '1': exog_predict['exog_1'].copy()
    }
    exog_2_pred['1'].index = pd.date_range(start='2020-02-20', periods=50)

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series_2, exog=exog_2)

    warn_msg = re.escape(
        ("Only series whose last window ends at the same index "
         "can be predicted together. Series that do not reach the "
         "maximum index, '2020-02-19 00:00:00', are excluded "
         "from prediction: {'2'}.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        results = forecaster.predict_bootstrapping(
                      steps                   = 1,
                      levels                  = ['1', '2'],
                      n_boot                  = 4, 
                      exog                    = exog_2_pred, 
                      use_in_sample_residuals = True
                  )

    expected = {
        '1': pd.DataFrame(
                 data    = np.array([[0.47931483, 0.21880114, 0.10692630, 0.42510201]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.date_range(start='2020-02-20', periods=1)
             )
    }

    for k in results.keys():
        pd.testing.assert_frame_equal(results[k], expected[k])


def test_predict_bootstrapping_UnknownLevelWarning_when_not_in_sample_residuals_for_level():
    """
    Test UnknownLevelWarning is raised when predicting an unknown level and 
    use_in_sample_residuals is True (using _unknown_level residuals).
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LGBMRegressor(verbose=-1), lags=3, encoding='ordinal'
    )
    forecaster.fit(series=series)
    last_window = pd.DataFrame(forecaster.last_window_)
    last_window['3'] = last_window['1']

    warn_msg = re.escape(
        ("`levels` {'3'} were not included in training. "
         "Unknown levels are encoded as NaN, which may cause the "
         "prediction to fail if the regressor does not accept NaN values.")
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        results = forecaster.predict_bootstrapping(
                      steps                   = 1,
                      levels                  = ['1', '2', '3'],
                      last_window             = last_window,
                      n_boot                  = 4,
                      use_in_sample_residuals = True
                  )

    warn_msg = re.escape(
        ("`levels` {'3'} are not present in `forecaster.in_sample_residuals_`, "
         "most likely because they were not present in the training data. "
         "A random sample of the residuals from other levels will be used. "
         "This can lead to inaccurate intervals for the unknown levels.")
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        results = forecaster.predict_bootstrapping(
                      steps                   = 1,
                      levels                  = ['1', '2', '3'],
                      last_window             = last_window,
                      n_boot                  = 4,
                      use_in_sample_residuals = True
                  )

    expected = {
        '1': pd.DataFrame(
                 data    = np.array([[0.60487599, 0.22612695, 0.07322577, 0.45340699]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '2': pd.DataFrame(
                 data    = np.array([[0.69343437, 0.67766497, 0.69444735, 0.64415095]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '3': pd.DataFrame(
                 data    = np.array([[0.26291992, 0.45547833, 0.22612695, 0.51589959]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             )
    }

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_encoding_None_unknown_level():
    """
    Test UnknownLevelWarning is raised when predicting an unknown level and 
    encoding is None (using _unknown_level residuals).
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LGBMRegressor(verbose=-1), lags=3, encoding=None
    )
    forecaster.fit(series=series)
    last_window = pd.DataFrame(forecaster.last_window_)
    last_window['3'] = last_window['1']

    results = forecaster.predict_bootstrapping(
                  steps                   = 1,
                  levels                  = ['1', '2', '3'],
                  last_window             = last_window,
                  n_boot                  = 4,
                  use_in_sample_residuals = True
              )

    expected = {
        '1': pd.DataFrame(
                 data    = np.array([[0.62339220, 0.03921391, 0.58593655, 0.45059094]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '2': pd.DataFrame(
                 data    = np.array([[0.94743870, 0.53474394, 0.38578090, 0.56857677]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '3': pd.DataFrame(
                 data    = np.array([[0.23759071, 0.42972248, 0.19361364, 0.49593259]]),
                 columns = [f"pred_boot_{i}" for i in range(4)],
                 index   = pd.RangeIndex(50, 51)
             )
    }

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_encoding_None_unknown_level_differentiation():
    """
    Test UnknownLevelWarning is raised when predicting an unknown level and 
    encoding is None (using _unknown_level residuals) and differentiation=1.
    """
    forecaster = ForecasterRecursiveMultiSeries(
        LGBMRegressor(verbose=-1), lags=3, encoding=None, differentiation=1
    )
    forecaster.fit(series=series)
    last_window = pd.DataFrame(forecaster.last_window_)
    last_window['3'] = last_window['1'] * 0.9

    results = forecaster.predict_bootstrapping(
                  steps                   = 1,
                  levels                  = ['1', '2', '3'],
                  last_window             = last_window,
                  n_boot                  = 5,
                  use_in_sample_residuals = True
              )

    expected = {
        '1': pd.DataFrame(
                 data    = np.array([[0.63878989, 0.33256144, 0.62392879, 0.37577094, 1.14849867]]),
                 columns = [f"pred_boot_{i}" for i in range(5)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '2': pd.DataFrame(
                 data    = np.array([[0.26274688, -0.00150092, 0.26392699, 0.18584762, 0.26392699]]),
                 columns = [f"pred_boot_{i}" for i in range(5)],
                 index   = pd.RangeIndex(50, 51)
             ),
        '3': pd.DataFrame(
                 data    = np.array([[0.44105054, 0.54865001, 0.26752194, 0.36550593, 0.26752194]]),
                 columns = [f"pred_boot_{i}" for i in range(5)],
                 index   = pd.RangeIndex(50, 51)
             )
    }

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ is None.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)

    err_msg = re.escape(
        ("`forecaster.out_sample_residuals_` is `None`. Use "
         "`use_in_sample_residuals=True` or the "
         "`set_out_sample_residuals()` method before predicting.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(
            steps=1, levels='1', use_in_sample_residuals=False
        )


def test_predict_bootstrapping_UnknownLevelWarning_out_sample_residuals_with_encoding_None():
    """
    Test UnknownLevelWarning is raised when encoding is None and 
    out_sample_residuals is set.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3, 
                                              encoding=None)
    forecaster.fit(series=series)

    new_residuals = {
        '1': np.array([1, 2, 3, 4, 5]), 
        '2': np.array([1, 2, 3, 4, 5])
    }
    warn_msg = re.escape(
        ("As `encoding` is set to `None`, no distinction between levels "
         "is made. All residuals are stored in the '_unknown_level' key.")
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        forecaster.set_out_sample_residuals(new_residuals)
    
    last_window = pd.DataFrame(forecaster.last_window_)
    last_window['3'] = last_window['1']
    results = forecaster.predict_bootstrapping(
                  steps                   = 1,
                  levels                  = ['1', '2', '3'],
                  last_window             = last_window,
                  n_boot                  = 4,
                  use_in_sample_residuals = False
              )

    expected = {
        '1': pd.DataFrame(
              data    = np.array([[1.48307367, 2.48307367, 1.48307367, 1.48307367]]),
              columns = [f"pred_boot_{i}" for i in range(4)],
              index   = pd.RangeIndex(50, 51)
          ),
        '2': pd.DataFrame(
              data    = np.array([[5.5061193, 3.5061193, 3.5061193, 2.5061193]]),
              columns = [f"pred_boot_{i}" for i in range(4)],
              index   = pd.RangeIndex(50, 51)
          ),
        '3': pd.DataFrame(
              data    = np.array([[4.48307367, 2.48307367, 4.48307367, 4.48307367]]),
              columns = [f"pred_boot_{i}" for i in range(4)],
              index   = pd.RangeIndex(50, 51)
          )
    }

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


@pytest.mark.parametrize("transformer_series", 
                         [None, StandardScaler()],
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_predict_bootstrapping_ValueError_when_not_level_in_out_sample_residuals(transformer_series):
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ is missing a level.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=transformer_series)
    forecaster.fit(series=series)
    residuals = {'1': np.array([1, 2, 3, 4, 5])}
    forecaster.set_out_sample_residuals(residuals = residuals)

    err_msg = re.escape(
        ("Not available residuals for level '2'. "
         "Check `forecaster.out_sample_residuals_`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=3, use_in_sample_residuals=False)


@pytest.mark.parametrize("transformer_series", 
                         [None, StandardScaler()],
                         ids = lambda tr: f'transformer_series type: {type(tr)}')
def test_predict_bootstrapping_ValueError_when_level_out_sample_residuals_value_contains_None_or_NaNs(transformer_series):
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ has a level with a None or NaN value.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=transformer_series)
    forecaster.fit(series=series)
    residuals = {'1': np.array([1, 2, 3, 4, 5]),
                 '2': np.array([1, 2, 3, 4, None])}  # StandardScaler() transforms None to NaN
    forecaster.set_out_sample_residuals(residuals = residuals)

    err_msg = re.escape(
        ("forecaster residuals for level '2' contains `None` "
         "or `NaNs` values. Check `forecaster.out_sample_residuals_`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=3, use_in_sample_residuals=False)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'])
    results = forecaster.predict_bootstrapping(
                  steps                   = 1, 
                  n_boot                  = 4, 
                  exog                    = exog_predict['exog_1'], 
                  use_in_sample_residuals = True
              )

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.48128438, 0.20172898, 0.07381993, 0.38819631]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.72210743, 0.73878038, 0.80513315, 0.57237401]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_True():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using in-sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'])
    results = forecaster.predict_bootstrapping(
                  steps                   = 2, 
                  n_boot                  = 4, 
                  exog                    = exog_predict['exog_1'], 
                  use_in_sample_residuals = True
              )

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.48128438, 0.20172898, 0.07381993, 0.38819631],
                                         [0.22472696, 0.06269684, 0.35120532, 0.66839553]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.08624916, 0.57237401, 0.69862572, 0.92542401],
                                         [0.06877403, 0.19830837, 0.10900956, 0.09861192]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )    

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_1_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    1 step ahead is predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'])
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_bootstrapping(
                  steps                   = 1, 
                  n_boot                  = 4, 
                  exog                    = exog_predict['exog_1'], 
                  use_in_sample_residuals = False,
                  suppress_warnings       = True
              )

    expected_1 = pd.DataFrame(
                     data    = np.array([[0.48128438, 0.20172898, 0.07381993, 0.38819631]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.72210743, 0.73878038, 0.80513315, 0.57237401]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=51)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
    
    
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_exog_steps_is_2_in_sample_residuals_is_False():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    2 steps ahead are predicted with exog using out-sample residuals.
    """
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    forecaster.fit(series=series, exog=exog['exog_1'])
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_bootstrapping(
                  steps                   = 2, 
                  n_boot                  = 4, 
                  exog                    = exog_predict['exog_1'], 
                  use_in_sample_residuals = False
              )
    
    expected_1 = pd.DataFrame(
                     data    = np.array([[0.48128438, 0.20172898, 0.07381993, 0.38819631],
                                         [0.22472696, 0.06269684, 0.35120532, 0.66839553]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.08624916, 0.57237401, 0.69862572, 0.92542401],
                                         [0.06877403, 0.19830837, 0.10900956, 0.09861192]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )    

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer_fixed_to_zero():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included, both
    inputs are transformed and in-sample residuals are fixed to 0.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)

    forecaster.in_sample_residuals_['1'] = np.full_like(forecaster.in_sample_residuals_['1'], fill_value=0)
    forecaster.in_sample_residuals_['2'] = np.full_like(forecaster.in_sample_residuals_['2'], fill_value=0)
    results = forecaster.predict_bootstrapping(
                  steps                   = 2, 
                  n_boot                  = 4, 
                  exog                    = exog_predict, 
                  use_in_sample_residuals = True
              )
    
    expected_1 = pd.DataFrame(
                     data    = np.array([[0.50201669, 0.50201669, 0.50201669, 0.50201669],
                                         [0.49804821, 0.49804821, 0.49804821, 0.49804821]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.52531076, 0.52531076, 0.52531076, 0.52531076],
                                         [0.51683613, 0.51683613, 0.51683613, 0.51683613]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_bootstrapping(
                  steps                   = 2, 
                  n_boot                  = 4, 
                  exog                    = exog_predict, 
                  use_in_sample_residuals = True
              )
    
    expected_1 = pd.DataFrame(
                     data    = np.array([[0.48598851, 0.24503818, 0.10778738, 0.38990777],
                                         [0.25740769, 0.06499781, 0.35491959, 0.67049609]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([[0.08931035, 0.57081569, 0.70066373, 0.96406103],
                                         [0.06876068, 0.23752454, 0.11044965, 0.10202009]]),
                     columns = [f"pred_boot_{i}" for i in range(4)],
                     index   = pd.RangeIndex(start=50, stop=52)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_output_when_window_features():
    """
    Test output of predict_bootstrapping when regressor is LGBMRegressor 
    and window features.
    """

    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=4)
    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )
    
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LGBMRegressor(verbose=-1),
                     lags               = 5,
                     window_features    = rolling,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_bootstrapping(
                  steps                   = 6, 
                  n_boot                  = 10, 
                  exog                    = exog_predict, 
                  use_in_sample_residuals = True
              )

    expected_1 = pd.DataFrame(
                     data = np.array([
                                [0.70414275, 0.33846487, 0.51525768, 0.78238922, 0.53358641,
                                 0.55496755, 0.49698665, 0.3156552 , 0.70164044, 0.51987002],
                                [0.5093462 , 0.10945852, 0.50986314, 0.53323641, 0.50986314,
                                 0.31517078, 0.54867791, 0.09113636, 0.46379211, 0.41342218],
                                [0.86504059, 0.56932758, 0.45218404, 0.44580085, 0.83119047,
                                 0.43019077, 0.68569118, 0.51340363, 0.61452721, 0.84957527],
                                [0.65524631, 0.84194486, 0.93082861, 0.7155279 , 0.83658981,
                                 1.02306043, 0.6582736 , 0.86211073, 0.56189833, 0.70564327],
                                [0.54647555, 0.75265845, 0.43334796, 0.36702491, 0.56762363,
                                 0.43184972, 0.66142918, 0.47738529, 0.57800408, 0.17483813],
                                [0.53343541, 0.56488173, 0.78929977, 0.44786181, 0.49626976,
                                 0.6850046 , 0.36524591, 0.70909355, 0.34761764, 0.39494035]]),
                     index   = pd.RangeIndex(start=50, stop=56, step=1),
                     columns = [f"pred_boot_{i}" for i in range(10)]
                 )
    expected_2 = pd.DataFrame(
                     data = np.array([
                                [0.4703539 , 0.61121897, 0.7456648 , 1.05767197, 0.85702976,
                                 0.84549833, 0.90211559, 1.05767197, 0.84549833, 0.43576797],
                                [0.17638536, 0.36323208, 0.28360047, 0.39694573, 0.42021445,
                                 0.62085666, 0.21910573, 0.1686658 , 0.03353859, 0.17638536],
                                [0.63706768, 0.95491448, 0.84158046, 0.88270255, 0.63313527,
                                 0.56008641, 0.4801817 , 0.64017749, 0.48757368, 0.70482518],
                                [0.88520648, 0.81333217, 0.27723066, 0.44112079, 0.45918293,
                                 0.57023604, 0.96036122, 0.63876782, 0.81917714, 0.82440159],
                                [0.45118834, 0.55605023, 0.84244033, 0.9414499 , 0.55014026,
                                 0.50114166, 0.25916732, 0.52348578, 0.64211623, 0.49025447],
                                [0.49198322, 0.10785838, 0.12577904, 0.12725819, 0.79780591,
                                 0.54327887, 0.78210044, 0.78482102, 0.19724282, 0.7278225 ]]),
                     index   = pd.RangeIndex(start=50, stop=56, step=1),
                     columns = [f"pred_boot_{i}" for i in range(10)]
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])


def test_predict_bootstrapping_output_when_differentiation():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals, exog is included and both
    inputs are transformed and differentiation.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = StandardScaler(),
                     transformer_exog   = transformer_exog,
                     differentiation    = 1
                 )
    forecaster.fit(series=series, exog=exog)
    results = forecaster.predict_bootstrapping(
                  steps                   = 3, 
                  n_boot                  = 10, 
                  exog                    = exog_predict, 
                  use_in_sample_residuals = True
              )
    
    expected_1 = pd.DataFrame(
                     data    = np.array([
                                   [0.86835554,  0.43568881,  0.72677672,  1.19824354,  0.44801019,
                                    0.66116832,  0.98730031,  0.49328573,  0.83294094,  0.49328573],
                                   [0.932948  , -0.04286105,  0.78881897,  0.94368941,  0.68990191,
                                    0.39479453,  0.66796447, -0.02242345,  0.79317039,  1.09345254],
                                   [0.80047137,  0.1669375 ,  0.49039454,  1.27844361,  1.00682521,
                                   -0.02294521,  0.66800615,  0.0497091 ,  0.79972101,  1.18391928]]),
                     columns = [f"pred_boot_{i}" for i in range(10)],
                     index   = pd.RangeIndex(start=50, stop=53)
                 )
    expected_2 = pd.DataFrame(
                     data    = np.array([
                                   [0.30645781,  0.38475619, -0.21650711, -0.11175337,  0.13573138,
                                    0.45197729,  0.07709315,  0.62428333,  0.42746891, -0.02426089],
                                   [0.62617546,  0.48165268, -0.50937191,  0.02315903,  0.00976064,
                                    0.37468356, -0.01104646,  0.34859438,  0.48364255,  0.07752748],
                                   [0.2132809 ,  0.24251965, -0.01663732,  0.39527145,  0.5228262 ,
                                   -0.15531628, -0.40091162,  0.07307154,  0.35477817,  0.31644122]]),
                     columns = [f"pred_boot_{i}" for i in range(10)],
                     index   = pd.RangeIndex(start=50, stop=53)
                 )

    expected = {'1': expected_1, '2': expected_2}

    for key in results.keys():
        pd.testing.assert_frame_equal(results[key], expected[key])
