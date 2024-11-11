# Unit test create_train_X_y ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirect


@pytest.mark.parametrize("step", [0, 4], ids=lambda step: f'step: {step}')
def test_filter_train_X_y_for_step_ValueError_when_step_not_in_steps(step):
    """
    Test ValueError is raised when step not in steps.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)

    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=3)
    X_train, y_train = forecaster.create_train_X_y(y)

    err_msg = re.escape(
        f"Invalid value `step`. For this forecaster, minimum value is 1 "
        f"and the maximum step is {forecaster.steps}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.filter_train_X_y_for_step(step=step, X_train=X_train, y_train=y_train)


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_exog_is_None_for_step_1():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 for step 1.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)

    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    X_train, y_train = forecaster.create_train_X_y(y=y)
    results = forecaster.filter_train_X_y_for_step(step=1, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0.],
                             [3., 2., 1.],
                             [4., 3., 2.],
                             [5., 4., 3.],
                             [6., 5., 4.],
                             [7., 6., 5.]], dtype=float),
            index   = pd.RangeIndex(start=3, stop=9, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 7., 8.], dtype=float),
            index = pd.RangeIndex(start=3, stop=9, step=1),
            name  = 'y_step_1'
        )
    )  
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_and_exog_for_step_2():
    """
    Test output of filter_train_X_y_for_step when regressor is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)
    results = forecaster.filter_train_X_y_for_step(step=2, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data = np.array([[2., 1., 0., 104.],
                             [3., 2., 1., 105.],
                             [4., 3., 2., 106.],
                             [5., 4., 3., 107.],
                             [6., 5., 4., 108.],
                             [7., 6., 5., 109.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_step_2']
        ),
        pd.Series(
            data  = np.array([4., 5., 6., 7., 8., 9.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'y_step_2', 
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
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirect(LinearRegression(), lags=3, steps=2)
    X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)
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
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog']
        ),
        pd.Series(
            data  = np.array([4., 5., 6., 7., 8., 9.]),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'y', 
            dtype = float
        )
    )
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_window_features_and_exog_steps_1():
    """
    Test the output of filter_train_X_y_for_step when using window_features and exog 
    with datetime index and steps=1.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirect(
        LinearRegression(), steps=1, lags=5, window_features=rolling
    )
    X_train, y_train = forecaster.create_train_X_y(y=y_datetime, exog=exog_datetime)
    results = forecaster.filter_train_X_y_for_step(
                  step          = 1, 
                  X_train       = X_train, 
                  y_train       = y_train,
                  remove_suffix = False
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 106.],
                             [6., 5., 4., 3., 2., 4., 4., 21., 107.],
                             [7., 6., 5., 4., 3., 5., 5., 27., 108.],
                             [8., 7., 6., 5., 4., 6., 6., 33., 109.],
                             [9., 8., 7., 6., 5., 7., 7., 39., 110.],
                             [10., 9., 8., 7., 6., 8., 8., 45., 111.],
                             [11., 10., 9., 8., 7., 9., 9., 51., 112.],
                             [12., 11., 10., 9., 8., 10., 10., 57., 113.],
                             [13., 12., 11., 10., 9., 11., 11., 63., 114.]]),
            index   = pd.date_range('2000-01-07', periods=9, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog_step_1']
        ),
        pd.Series(
            data  = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14]),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y_step_1',
            dtype = float
        )
    )
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_window_features_and_exog_steps_2():
    """
    Test the output of filter_train_X_y_for_step when using window_features and 
    exog with datetime index and steps=2.
    """
    y_datetime = pd.Series(
        np.arange(15), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='y', dtype=float
    )
    exog_datetime = pd.Series(
        np.arange(100, 115), index=pd.date_range('2000-01-01', periods=15, freq='D'),
        name='exog', dtype=float
    )
    rolling = RollingFeatures(
        stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6]
    )

    forecaster = ForecasterDirect(
        LinearRegression(), steps=2, lags=5, window_features=rolling
    )
    X_train, y_train = forecaster.create_train_X_y(y=y_datetime, exog=exog_datetime)
    results = forecaster.filter_train_X_y_for_step(
                  step          = 2, 
                  X_train       = X_train, 
                  y_train       = y_train,
                  remove_suffix = True
              )
    
    expected = (
        pd.DataFrame(
            data = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 107.],
                             [6., 5., 4., 3., 2., 4., 4., 21., 108.],
                             [7., 6., 5., 4., 3., 5., 5., 27., 109.],
                             [8., 7., 6., 5., 4., 6., 6., 33., 110.],
                             [9., 8., 7., 6., 5., 7., 7., 39., 111.],
                             [10., 9., 8., 7., 6., 8., 8., 45., 112.],
                             [11., 10., 9., 8., 7., 9., 9., 51., 113.],
                             [12., 11., 10., 9., 8., 10., 10., 57., 114.]]),
            index   = pd.date_range('2000-01-08', periods=8, freq='D'),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'roll_mean_5', 'roll_median_5', 'roll_sum_6', 
                       'exog']
        ),
        pd.Series(
            data  = np.array([7, 8, 9, 10, 11, 12, 13, 14]),
            index = pd.date_range('2000-01-08', periods=8, freq='D'),
            name  = 'y',
            dtype = float
        )
    )
 
    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
