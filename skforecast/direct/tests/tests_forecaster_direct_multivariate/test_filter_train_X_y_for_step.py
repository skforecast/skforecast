# Unit test _create_train_X_y ForecasterDirectMultiVariate
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate


@pytest.mark.parametrize("step", [0, 4], ids=lambda step: f'step: {step}')
def test_filter_train_X_y_for_step_exception_when_step_not_in_steps(step):
    """
    Test exception is raised when step not in steps.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=3)
    X_train, y_train, *_ = forecaster._create_train_X_y(series=series)

    err_msg = re.escape(
        (f"Invalid value `step`. For this forecaster, minimum value is 1 "
         f"and the maximum step is {forecaster.steps}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.filter_train_X_y_for_step(step=step, X_train=X_train, y_train=y_train)


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_exog_is_None_for_step_1():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 for step 1.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(100, 110), dtype=float)})
    
    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l1',
                                               lags=3, steps=2, transformer_series=None)
    X_train, y_train, *_ = forecaster._create_train_X_y(series=series)
    results = forecaster.filter_train_X_y_for_step(step=1, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 102., 101., 100.],
                             [3., 2., 1., 103., 102., 101.],
                             [4., 3., 2., 104., 103., 102.],
                             [5., 4., 3., 105., 104., 103.],
                             [6., 5., 4., 106., 105., 104.],
                             [7., 6., 5., 107., 106., 105.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=9, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 7., 8.], dtype=float),
            index = pd.RangeIndex(start=3, stop=9, step=1),
            name  = 'l1_step_1'
        )
    )
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_and_exog_for_step_2():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=[1, 2, 3], steps=2, transformer_series=None)
    X_train, y_train, *_ = forecaster._create_train_X_y(series=series, exog=exog)
    results = forecaster.filter_train_X_y_for_step(step=2, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 52., 51., 50., 104.],
                             [3., 2., 1., 53., 52., 51., 105.],
                             [4., 3., 2., 54., 53., 52., 106.],
                             [5., 4., 3., 55., 54., 53., 107.],
                             [6., 5., 4., 56., 55., 54., 108.],
                             [7., 6., 5., 57., 56., 55., 109.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3',
                       'exog_step_2']
        ),
        pd.Series(
            data  = np.array([54., 55., 56., 57., 58., 59.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'l2_step_2', 
            dtype = float
        )
    )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_and_exog_for_step_2_remove_suffix():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2 with remove_suffix=True.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags=[1, 2, 3], steps=2, transformer_series=None)
    X_train, y_train, *_ = forecaster._create_train_X_y(series=series, exog=exog)
    results = forecaster.filter_train_X_y_for_step(
                  step          = 2, 
                  X_train       = X_train, 
                  y_train       = y_train,
                  remove_suffix = True
              )

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 52., 51., 50., 104.],
                             [3., 2., 1., 53., 52., 51., 105.],
                             [4., 3., 2., 54., 53., 52., 106.],
                             [5., 4., 3., 55., 54., 53., 107.],
                             [6., 5., 4., 56., 55., 54., 108.],
                             [7., 6., 5., 57., 56., 55., 109.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3',
                       'exog']
        ),
        pd.Series(
            data  = np.array([54., 55., 56., 57., 58., 59.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'l2', 
            dtype = float
        )
    )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_lags_dict_with_None_steps_2_and_exog_for_step_2():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': None, 'l2': 3}, 
                                               steps=2, transformer_series=None)
    X_train, y_train, *_ = forecaster._create_train_X_y(series=series, exog=exog)
    results = forecaster.filter_train_X_y_for_step(step=2, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data = np.array([[52., 51., 50., 104.],
                             [53., 52., 51., 105.],
                             [54., 53., 52., 106.],
                             [55., 54., 53., 107.],
                             [56., 55., 54., 108.],
                             [57., 56., 55., 109.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['l2_lag_1', 'l2_lag_2', 'l2_lag_3',
                       'exog_step_2']
        ),
        pd.Series(
            data  = np.array([54., 55., 56., 57., 58., 59.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'l2_step_2', 
            dtype = float
        )
    )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_lags_dict_with_None_steps_2_and_exog_for_step_2_remove_suffix():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2 with remove_suffix=True.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10), dtype=float), 
                           'l2': pd.Series(np.arange(50, 60), dtype=float)})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirectMultiVariate(LinearRegression(), level='l2',
                                               lags={'l1': 3, 'l2': None}, 
                                               steps=2, transformer_series=None)
    X_train, y_train, *_ = forecaster._create_train_X_y(series=series, exog=exog)
    results = forecaster.filter_train_X_y_for_step(
                  step          = 2, 
                  X_train       = X_train, 
                  y_train       = y_train,
                  remove_suffix = True
              )

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 104.],
                             [3., 2., 1., 105.],
                             [4., 3., 2., 106.],
                             [5., 4., 3., 107.],
                             [6., 5., 4., 108.],
                             [7., 6., 5., 109.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3',
                       'exog']
        ),
        pd.Series(
            data  = np.array([54., 55., 56., 57., 58., 59.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'l2', 
            dtype = float
        )
    )
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_window_features_and_exog_steps_1():
    """
    Test the output of filter_train_X_y_for_step when using window_features and 
    exog with datetime index and steps=1.
    """
    series = pd.DataFrame(
        {'l1': np.arange(15, dtype=float), 
         'l2': np.arange(50, 65, dtype=float)},
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    exog = pd.Series(
        np.arange(100, 115), name='exog', 
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), steps=1, level='l1', lags=5, window_features=rolling, transformer_series=None
    )
    X_train, y_train, *_ = forecaster._create_train_X_y(series=series, exog=exog)
    results = forecaster.filter_train_X_y_for_step(
                  step          = 1, 
                  X_train       = X_train, 
                  y_train       = y_train,
                  remove_suffix = False
              )

    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 55., 54., 53., 52., 51., 53., 53., 315., 106.],
                             [6., 5., 4., 3., 2., 4., 4., 21., 56., 55., 54., 53., 52., 54., 54., 321., 107.],
                             [7., 6., 5., 4., 3., 5., 5., 27., 57., 56., 55., 54., 53., 55., 55., 327., 108.],
                             [8., 7., 6., 5., 4., 6., 6., 33., 58., 57., 56., 55., 54., 56., 56., 333., 109.],
                             [9., 8., 7., 6., 5., 7., 7., 39., 59., 58., 57., 56., 55., 57., 57., 339., 110.],
                             [10., 9., 8., 7., 6., 8., 8., 45., 60., 59., 58., 57., 56., 58., 58., 345., 111.],
                             [11., 10., 9., 8., 7., 9., 9., 51., 61., 60., 59., 58., 57., 59., 59., 351., 112.],
                             [12., 11., 10., 9., 8., 10., 10., 57., 62., 61., 60., 59., 58., 60., 60., 357., 113.],
                             [13., 12., 11., 10., 9., 11., 11., 63., 63., 62., 61., 60., 59., 61., 61., 363., 114.]],
                             dtype=float),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
                       'exog_step_1']
        ),
        pd.Series(
                data  = np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.], dtype=float), 
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = "l1_step_1"
        )
    )
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_create_train_X_y_output_when_window_features_and_exog_steps_2():
    """
    Test the output of _create_train_X_y when using window_features and exog 
    with datetime index and steps=2.
    """
    series = pd.DataFrame(
        {'l1': np.arange(15, dtype=float), 
         'l2': np.arange(50, 65, dtype=float)},
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    exog = pd.Series(
        np.arange(100, 115), name='exog', 
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), steps=2, level='l1', lags=5, 
        window_features=rolling, transformer_series=None
    )
    X_train, y_train, *_ = forecaster._create_train_X_y(series=series, exog=exog)
    results = forecaster.filter_train_X_y_for_step(
                  step          = 2, 
                  X_train       = X_train, 
                  y_train       = y_train,
                  remove_suffix = True
              )

    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 55., 54., 53., 52., 51., 53., 53., 315., 107.],
                             [6., 5., 4., 3., 2., 4., 4., 21., 56., 55., 54., 53., 52., 54., 54., 321., 108.],
                             [7., 6., 5., 4., 3., 5., 5., 27., 57., 56., 55., 54., 53., 55., 55., 327., 109.],
                             [8., 7., 6., 5., 4., 6., 6., 33., 58., 57., 56., 55., 54., 56., 56., 333., 110.],
                             [9., 8., 7., 6., 5., 7., 7., 39., 59., 58., 57., 56., 55., 57., 57., 339., 111.],
                             [10., 9., 8., 7., 6., 8., 8., 45., 60., 59., 58., 57., 56., 58., 58., 345., 112.],
                             [11., 10., 9., 8., 7., 9., 9., 51., 61., 60., 59., 58., 57., 59., 59., 351., 113.],
                             [12., 11., 10., 9., 8., 10., 10., 57., 62., 61., 60., 59., 58., 60., 60., 357., 114.]],
                             dtype=float),
            index = pd.date_range('2000-01-08', periods=8, freq='D'),
            columns = ['l1_lag_1', 'l1_lag_2', 'l1_lag_3', 'l1_lag_4', 'l1_lag_5', 
                       'l1_roll_mean_5', 'l1_roll_median_5', 'l1_roll_sum_6',
                       'l2_lag_1', 'l2_lag_2', 'l2_lag_3', 'l2_lag_4', 'l2_lag_5',
                       'l2_roll_mean_5', 'l2_roll_median_5', 'l2_roll_sum_6',
                       'exog']
        ),
        pd.Series(
            data  = np.array([7., 8., 9., 10., 11., 12., 13., 14.], dtype=float), 
            index = pd.date_range('2000-01-08', periods=8, freq='D'),
            name  = "l1"
        )
    )
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
