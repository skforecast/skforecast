# Unit test predict_bootstrapping ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

from skforecast.preprocessing import RollingFeatures
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.direct import ForecasterDirect

# Fixtures
from .fixtures_forecaster_direct import y
from .fixtures_forecaster_direct import exog
from .fixtures_forecaster_direct import exog_predict
from .fixtures_forecaster_direct import data  # to test results when using differentiation


def test_predict_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=5)

    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.predict_bootstrapping(steps=5)


def test_predict_bootstrapping_ValueError_when_not_in_sample_residuals_for_some_step():
    """
    Test ValueError is raised when use_in_sample_residuals=True but there is no
    residuals for some step.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=pd.Series(np.arange(10)))
    forecaster.in_sample_residuals_ = {2: np.array([1, 2, 3])}

    err_msg = re.escape(
        (f"Not `forecaster.in_sample_residuals_` for steps: "
         f"{set([1, 2]) - set(forecaster.in_sample_residuals_.keys())}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=None, use_in_sample_residuals=True)


def test_predict_bootstrapping_ValueError_when_out_sample_residuals_is_None():
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ is None.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    forecaster.fit(y=pd.Series(np.arange(10)))

    err_msg = re.escape(
        ("`forecaster.out_sample_residuals_` is `None`. Use "
         "`use_in_sample_residuals=True` or the "
         "`set_out_sample_residuals()` method before predicting.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=1, use_in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_not_out_sample_residuals_for_all_steps_predicted():
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ is not available for all steps predicted.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=3)
    forecaster.fit(y=pd.Series(np.arange(15)))
    residuals = {2: np.array([1, 2, 3, 4, 5]), 
                 3: np.array([1, 2, 3, 4, 5])}
    forecaster.out_sample_residuals_ = residuals

    err_msg = re.escape(
        (f"No `forecaster.out_sample_residuals_` for steps: "
         f"{set([1, 2]) - set(forecaster.out_sample_residuals_.keys())}. "
         f"Use method `set_out_sample_residuals()`.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=[1, 2], use_in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_step_out_sample_residuals_value_is_None():
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ has a step with a None.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=3)
    forecaster.fit(y=pd.Series(np.arange(15)))
    forecaster.out_sample_residuals_ = {
        1: np.array([1, 2, 3, 4, 5]),
        2: np.array([1, 2, 3, 4, 5]),
        3: None
    }
    err_msg = re.escape(
        ("forecaster residuals for step 3 are `None`. "
         "Check forecaster.out_sample_residuals_.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.predict_bootstrapping(steps=3, use_in_sample_residuals=False)


def test_predict_bootstrapping_ValueError_when_step_out_sample_residuals_value_contains_None():
    """
    Test ValueError is raised when use_in_sample_residuals=False and
    forecaster.out_sample_residuals_ has a step with a None value.
    """
    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=3)
    forecaster.fit(y=pd.Series(np.arange(15)))
    forecaster.out_sample_residuals_ = {
        1: np.array([1, 2, 3, 4, 5]),
        2: np.array([1, 2, 3, 4, 5]),
        3: np.array([1, 2, 3, 4, None]),
    }
    err_msg = re.escape(
        (
            "forecaster residuals for step 3 contains `None` values. "
            "Check forecaster.out_sample_residuals_."
        )
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.predict_bootstrapping(steps=3, use_in_sample_residuals=False)


@pytest.mark.parametrize("steps", [2, [1, 2], None], 
                         ids=lambda steps: f'steps: {steps}')
def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_True_exog_and_transformer(steps):
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
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
    results = forecaster.predict_bootstrapping(
        steps=steps, exog=exog_predict, n_boot=4, use_in_sample_residuals=True
    )

    expected = pd.DataFrame(
                   data = np.array([
                              [0.7319542342426019, 0.6896870991312363, 0.2024868918354829, 0.5468555312708429],
                              [0.13621073709672699, 0.29541488852202646, 0.5160668468937661, 0.33416280223011985]
                          ]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_False_exog_and_transformer():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
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
    forecaster.out_sample_residuals_ = forecaster.in_sample_residuals_
    results = forecaster.predict_bootstrapping(
        steps=2, exog=exog_predict, n_boot=4, use_in_sample_residuals=False
    )
    
    expected = pd.DataFrame(
                   data = np.array([
                              [0.7319542342426019, 0.6896870991312363, 0.2024868918354829, 0.5468555312708429],
                              [0.13621073709672699, 0.29541488852202646, 0.5160668468937661, 0.33416280223011985]
                          ]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_steps_is_2_in_sample_residuals_fixed():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression,
    2 steps are predicted, using in-sample residuals that are fixed.
    """
    forecaster = ForecasterDirect(
                     regressor = LinearRegression(),
                     steps     = 2,
                     lags      = 3
                 )
    forecaster.fit(y=y, exog=exog)
    forecaster.in_sample_residuals_ = {1: pd.Series([1, 1, 1, 1, 1, 1, 1]),
                                       2: pd.Series([5, 5, 5, 5, 5, 5, 5])}
    results = forecaster.predict_bootstrapping(
        steps=2, exog=exog_predict, n_boot=4, use_in_sample_residuals=True
    )
    
    expected = pd.DataFrame(
                   data = np.array([[1.67523588, 1.67523588, 1.67523588, 1.67523588],
                                    [5.38024988, 5.38024988, 5.38024988, 5.38024988]]),
                   columns = [f"pred_boot_{i}" for i in range(4)],
                   index   = pd.RangeIndex(start=50, stop=52)
               )
    
    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_and_differentiation_is_1_steps_1():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    differentiation=1 and steps=1.
    """
    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    data_diff = diferenciator.fit_transform(data.to_numpy())
    data_diff = pd.Series(data_diff, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirect(regressor=LinearRegression(), steps=1, lags=15)
    forecaster_1.fit(y=data_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    boot_predictions_diff = forecaster_1.predict_bootstrapping(
        exog=exog_diff.loc[end_train:], n_boot=10
    )

    # Revert the differentiation
    last_value_train = data.loc[:end_train].iloc[[-1]]
    boot_predictions_1 = boot_predictions_diff.copy()
    boot_predictions_1.loc[last_value_train.index[0]] = last_value_train.values[0]
    boot_predictions_1 = boot_predictions_1.sort_index()
    boot_predictions_1 = boot_predictions_1.cumsum(axis=0).iloc[1:,]
    boot_predictions_1 = boot_predictions_1.asfreq('MS')
    
    forecaster_2 = ForecasterDirect(regressor=LinearRegression(), steps=1, lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    boot_predictions_2 = forecaster_2.predict_bootstrapping(
        exog=exog_diff.loc[end_train:], n_boot=10
    )
    
    pd.testing.assert_frame_equal(boot_predictions_1, boot_predictions_2)


def test_predict_bootstrapping_output_when_forecaster_is_LinearRegression_and_differentiation_is_1_steps_10():
    """
    Test output of predict_bootstrapping when regressor is LinearRegression and
    differentiation=1 and steps=10.
    """
    # Data differentiated
    diferenciator = TimeSeriesDifferentiator(order=1)
    data_diff = diferenciator.fit_transform(data.to_numpy())
    data_diff = pd.Series(data_diff, index=data.index).dropna()

    # Simulated exogenous variable
    rng = np.random.default_rng(9876)
    exog = pd.Series(
        rng.normal(loc=0, scale=1, size=len(data)), index=data.index, name='exog'
    )
    exog_diff = exog.iloc[1:]
    end_train = '2003-03-01 23:59:00'

    forecaster_1 = ForecasterDirect(regressor=LinearRegression(), steps=10, lags=15)
    forecaster_1.fit(y=data_diff.loc[:end_train], exog=exog_diff.loc[:end_train])
    boot_predictions_diff = forecaster_1.predict_bootstrapping(
        exog=exog_diff.loc[end_train:], n_boot=10
    )
    last_value_train = data.loc[:end_train].iloc[[-1]]
    boot_predictions_1 = boot_predictions_diff.copy()
    boot_predictions_1.loc[last_value_train.index[0]] = last_value_train.values[0]
    boot_predictions_1 = boot_predictions_1.sort_index()
    boot_predictions_1 = boot_predictions_1.cumsum(axis=0).iloc[1:,]
    boot_predictions_1 = boot_predictions_1.asfreq('MS')
    
    forecaster_2 = ForecasterDirect(regressor=LinearRegression(), steps=10, lags=15, differentiation=1)
    forecaster_2.fit(y=data.loc[:end_train], exog=exog.loc[:end_train])
    boot_predictions_2 = forecaster_2.predict_bootstrapping(
        exog=exog_diff.loc[end_train:], n_boot=10
    )
    
    pd.testing.assert_frame_equal(boot_predictions_1, boot_predictions_2)


def test_predict_bootstrapping_output_when_window_features_steps_1():
    """
    Test output of predict_bootstrapping when regressor is LGBMRegressor and
    4 steps ahead are predicted with exog and window features using in-sample residuals
    with steps=1.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2001-01-01', periods=len(y), freq='D')
    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(start='2001-01-01', periods=len(exog), freq='D')
    exog_predict_datetime = exog_predict.copy()
    exog_predict_datetime.index = pd.date_range(start='2001-02-20', periods=len(exog_predict), freq='D')
    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirect(
        LGBMRegressor(verbose=-1, random_state=123), steps=1, lags=3, window_features=rolling
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster.predict_bootstrapping(
        n_boot=10, exog=exog_predict_datetime, use_in_sample_residuals=True
    )

    expected = pd.DataFrame(
                   data    = np.array(
                                 [[0.61866004, 0.42315364, 0.54459359, 0.6327955 , 0.29239436,
                                   0.46729748, 0.3783451 , 0.18128852, 0.71765599, 0.58209158]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(10)],
                   index   = pd.date_range(start='2001-02-20', periods=1, freq='D')
               )

    pd.testing.assert_frame_equal(expected, results)


def test_predict_bootstrapping_output_when_window_features_steps_10():
    """
    Test output of predict_bootstrapping when regressor is LGBMRegressor and
    4 steps ahead are predicted with exog and window features using in-sample residuals
    with steps=10.
    """
    y_datetime = y.copy()
    y_datetime.index = pd.date_range(start='2001-01-01', periods=len(y), freq='D')
    exog_datetime = exog.copy()
    exog_datetime.index = pd.date_range(start='2001-01-01', periods=len(exog), freq='D')
    exog_predict_datetime = exog_predict.copy()
    exog_predict_datetime.index = pd.date_range(start='2001-02-20', periods=len(exog_predict), freq='D')
    rolling = RollingFeatures(stats=['mean', 'sum'], window_sizes=[3, 5])
    forecaster = ForecasterDirect(
        LGBMRegressor(verbose=-1, random_state=123), steps=10, lags=3, window_features=rolling
    )
    forecaster.fit(y=y_datetime, exog=exog_datetime)
    results = forecaster.predict_bootstrapping(
        n_boot=10, exog=exog_predict_datetime, use_in_sample_residuals=True
    )

    expected = pd.DataFrame(
                   data    = np.array(
                                 [[0.42310646, 0.63097612, 0.36178866, 0.9807642 , 0.89338916,
                                   0.43857224, 0.39804426, 0.72904971, 0.17545176, 0.72904971],
                                  [0.53155137, 0.31226122, 0.72445532, 0.50183668, 0.72445532,
                                   0.73799541, 0.42583029, 0.31226122, 0.89338916, 0.94416002],
                                  [0.68482974, 0.32295891, 0.18249173, 0.73799541, 0.73799541,
                                   0.42635131, 0.31226122, 0.39804426, 0.84943179, 0.4936851 ],
                                  [0.0596779 , 0.09210494, 0.61102351, 0.1156184 , 0.42583029,
                                   0.18249173, 0.94416002, 0.42635131, 0.73799541, 0.36178866],
                                  [0.89338916, 0.17545176, 0.17545176, 0.39804426, 0.39211752,
                                   0.36178866, 0.39211752, 0.63097612, 0.61102351, 0.73799541],
                                  [0.61102351, 0.34317802, 0.73799541, 0.36178866, 0.43857224,
                                   0.42635131, 0.53182759, 0.41482621, 0.18249173, 0.43086276],
                                  [0.63097612, 0.86630916, 0.4936851 , 0.31728548, 0.22826323,
                                   0.53155137, 0.17545176, 0.31728548, 0.53155137, 0.89338916],
                                  [0.09210494, 0.72445532, 0.36178866, 0.62395295, 0.29371405,
                                   0.41482621, 0.48303426, 0.72445532, 0.09210494, 0.09210494],
                                  [0.43086276, 0.73799541, 0.36178866, 0.4936851 , 0.48303426,
                                   0.84943179, 0.42635131, 0.62395295, 0.48303426, 0.53182759],
                                  [0.94416002, 0.32295891, 0.63097612, 0.39804426, 0.62395295,
                                   0.73799541, 0.22826323, 0.43370117, 0.94416002, 0.09210494]]
                             ),
                   columns = [f"pred_boot_{i}" for i in range(10)],
                   index   = pd.date_range(start='2001-02-20', periods=10, freq='D')
               )

    pd.testing.assert_frame_equal(expected, results)
