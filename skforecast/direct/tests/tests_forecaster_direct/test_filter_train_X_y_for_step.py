# Unit test filter_train_X_y_for_step ForecasterDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirect


@pytest.mark.parametrize(
    'step', [0, 4], ids=lambda s: f'step: {s}'
)
def test_filter_train_X_y_for_step_ValueError_when_step_not_in_steps(step):
    """
    Test ValueError is raised when step not in steps.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=3
    )
    X_train, y_train = forecaster.create_train_X_y(y)

    err_msg = re.escape(
        f"Invalid value `step`. For this forecaster, minimum value is 1 "
        f"and the maximum step is {forecaster.max_step}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.filter_train_X_y_for_step(step=step, X_train=X_train, y_train=y_train)


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_exog_is_None_for_step_1():
    """
    Test output of filter_train_X_y_for_step when estimator is LinearRegression, 
    lags is 3 and steps is 2 for step 1.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2
    )
    X_train, y_train = forecaster.create_train_X_y(y=y)
    results = forecaster.filter_train_X_y_for_step(step=1, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data    = np.array([[2., 1., 0.],
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
    Test output of filter_train_X_y_for_step when estimator is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2
    )
    X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)
    results = forecaster.filter_train_X_y_for_step(step=2, X_train=X_train, y_train=y_train)

    expected = (
        pd.DataFrame(
            data    = np.array([[2., 1., 0., 104.],
                                [3., 2., 1., 105.],
                                [4., 3., 2., 106.],
                                [5., 4., 3., 107.],
                                [6., 5., 4., 108.],
                                [7., 6., 5., 109.]], dtype=float),
            index   = pd.RangeIndex(start=4, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_step_2']
        ),
        pd.Series(
            data  = np.array([4., 5., 6., 7., 8., 9.], dtype=float),
            index = pd.RangeIndex(start=4, stop=10, step=1),
            name  = 'y_step_2', 
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_lags_3_steps_2_and_exog_for_step_2_remove_suffix():
    """
    Test output of filter_train_X_y_for_step when estimator is LinearRegression, 
    lags is 3 and steps is 2 with exog for step 2 with remove_suffix=True.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2
    )
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
        estimator=LinearRegression(), steps=1, lags=5, window_features=rolling
    )
    X_train, y_train = forecaster.create_train_X_y(y=y_datetime, exog=exog_datetime)
    results = forecaster.filter_train_X_y_for_step(
                  step    = 1, 
                  X_train = X_train, 
                  y_train = y_train
              )

    expected = (
        pd.DataFrame(
            data    = np.array([[5., 4., 3., 2., 1., 3., 3., 15., 106.],
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
            data  = np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.], dtype=float),
            index = pd.date_range('2000-01-07', periods=9, freq='D'),
            name  = 'y_step_1',
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


@pytest.mark.parametrize(
    'remove_suffix, expected_columns, expected_y_name',
    [
        (
            False,
            ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
             'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog_step_2'],
            'y_step_2'
        ),
        (
            True,
            ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
             'roll_mean_5', 'roll_median_5', 'roll_sum_6', 'exog'],
            'y'
        ),
    ],
    ids=['keep_suffix', 'remove_suffix']
)
def test_filter_train_X_y_for_step_output_when_window_features_and_exog_steps_2(
    remove_suffix, expected_columns, expected_y_name
):
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
        estimator=LinearRegression(), steps=2, lags=5, window_features=rolling
    )
    X_train, y_train = forecaster.create_train_X_y(y=y_datetime, exog=exog_datetime)
    results = forecaster.filter_train_X_y_for_step(
                  step          = 2, 
                  X_train       = X_train, 
                  y_train       = y_train,
                  remove_suffix = remove_suffix
              )

    expected_X_data = np.array(
        [[5., 4., 3., 2., 1., 3., 3., 15., 107.],
         [6., 5., 4., 3., 2., 4., 4., 21., 108.],
         [7., 6., 5., 4., 3., 5., 5., 27., 109.],
         [8., 7., 6., 5., 4., 6., 6., 33., 110.],
         [9., 8., 7., 6., 5., 7., 7., 39., 111.],
         [10., 9., 8., 7., 6., 8., 8., 45., 112.],
         [11., 10., 9., 8., 7., 9., 9., 51., 113.],
         [12., 11., 10., 9., 8., 10., 10., 57., 114.]]
    )
    expected = (
        pd.DataFrame(
            data    = expected_X_data,
            index   = pd.date_range('2000-01-08', periods=8, freq='D'),
            columns = expected_columns
        ),
        pd.Series(
            data  = np.array([7., 8., 9., 10., 11., 12., 13., 14.], dtype=float),
            index = pd.date_range('2000-01-08', periods=8, freq='D'),
            name  = expected_y_name,
            dtype = float
        )
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])


def test_filter_train_X_y_for_step_output_when_y_and_exog_have_nan_and_dropna_from_series_True():
    """
    Test that filter_train_X_y_for_step removes rows where y or exog has
    NaN when dropna_from_series is True. NaN in y propagates to different
    steps due to the target offset; NaN in exog affects the step where
    that exog column appears.
    """
    y = pd.Series(
        [0, 1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        index=pd.date_range('2020-01-01', periods=13, freq='D'),
        name='y', dtype=float
    )
    exog = pd.DataFrame(
        {
            'exog_1': [10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.],
            'exog_2': [100., 101., 102., np.nan, 104., 105., 106., 107., 108., 109., 110., np.nan, 112.],
        },
        index=pd.date_range('2020-01-01', periods=13, freq='D'),
    )

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2,
        dropna_from_series=True
    )
    X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)

    # Step 1: NaN from y (lag propagation) and exog_2 NaN at 2020-01-12
    X_step1, y_step1 = forecaster.filter_train_X_y_for_step(
        step=1, X_train=X_train, y_train=y_train
    )

    expected_X_step1 = pd.DataFrame(
        data    = np.array([[6., 5., 4., 17., 107.],
                            [7., 6., 5., 18., 108.],
                            [8., 7., 6., 19., 109.],
                            [9., 8., 7., 20., 110.]]),
        index   = pd.date_range('2020-01-08', periods=4, freq='D'),
        columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1_step_1', 'exog_2_step_1']
    )
    expected_y_step1 = pd.Series(
        data  = np.array([7., 8., 9., 10.]),
        index = pd.date_range('2020-01-08', periods=4, freq='D'),
        name  = 'y_step_1'
    )

    pd.testing.assert_frame_equal(X_step1, expected_X_step1)
    pd.testing.assert_series_equal(y_step1, expected_y_step1)

    # Step 2: NaN from y (target and lag) and exog_2 NaN at 2020-01-12
    X_step2, y_step2 = forecaster.filter_train_X_y_for_step(
        step=2, X_train=X_train, y_train=y_train
    )

    expected_X_step2 = pd.DataFrame(
        data    = np.array([[ 2.,  1.,  0., 14., 104.],
                            [ 6.,  5.,  4., 18., 108.],
                            [ 7.,  6.,  5., 19., 109.],
                            [ 8.,  7.,  6., 20., 110.],
                            [10.,  9.,  8., 22., 112.]]),
        index   = pd.DatetimeIndex(
                      ['2020-01-05', '2020-01-09', '2020-01-10',
                       '2020-01-11', '2020-01-13'],
                      freq=None
                  ),
        columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1_step_2', 'exog_2_step_2']
    )
    expected_y_step2 = pd.Series(
        data  = np.array([4., 8., 9., 10., 12.]),
        index = pd.DatetimeIndex(
                    ['2020-01-05', '2020-01-09', '2020-01-10',
                     '2020-01-11', '2020-01-13'],
                    freq=None
                ),
        name  = 'y_step_2'
    )

    pd.testing.assert_frame_equal(X_step2, expected_X_step2)
    pd.testing.assert_series_equal(y_step2, expected_y_step2)


def test_filter_train_X_y_for_step_output_when_y_has_nan_and_dropna_from_series_False():
    """
    Test that filter_train_X_y_for_step removes rows where y has NaN but
    keeps rows where X has NaN when dropna_from_series is False.
    """
    y = pd.Series(
        [0, 1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        index=pd.date_range('2020-01-01', periods=13, freq='D'),
        name='y', dtype=float
    )

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2,
        dropna_from_series=False
    )
    X_train, y_train = forecaster.create_train_X_y(y=y)

    X_step1, y_step1 = forecaster.filter_train_X_y_for_step(
        step=1, X_train=X_train, y_train=y_train
    )

    expected_X_step1 = pd.DataFrame(
        data    = np.array([[np.nan,  2.,  1.],
                            [ 4., np.nan,  2.],
                            [ 5.,  4., np.nan],
                            [ 6.,  5.,  4.],
                            [ 7.,  6.,  5.],
                            [ 8.,  7.,  6.],
                            [ 9.,  8.,  7.],
                            [10.,  9.,  8.]]),
        index   = pd.date_range('2020-01-05', periods=8, freq='D'),
        columns = ['lag_1', 'lag_2', 'lag_3']
    )
    expected_y_step1 = pd.Series(
        data  = np.array([4., 5., 6., 7., 8., 9., 10., 11.]),
        index = pd.date_range('2020-01-05', periods=8, freq='D'),
        name  = 'y_step_1'
    )

    pd.testing.assert_frame_equal(X_step1, expected_X_step1)
    pd.testing.assert_series_equal(y_step1, expected_y_step1)


def test_filter_train_X_y_for_step_output_when_no_nans_unchanged():
    """
    Test that filter_train_X_y_for_step returns the same output as before
    NaN filtering was added when there are no NaN values.
    """
    y = pd.Series(np.arange(10), name='y', dtype=float)

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), lags=3, steps=2,
        dropna_from_series=True
    )
    X_train, y_train = forecaster.create_train_X_y(y=y)
    X_step, y_step = forecaster.filter_train_X_y_for_step(
        step=1, X_train=X_train, y_train=y_train
    )

    expected_X = pd.DataFrame(
        data=np.array([[2., 1., 0.],
                        [3., 2., 1.],
                        [4., 3., 2.],
                        [5., 4., 3.],
                        [6., 5., 4.],
                        [7., 6., 5.]], dtype=float),
        index=pd.RangeIndex(start=3, stop=9, step=1),
        columns=['lag_1', 'lag_2', 'lag_3']
    )
    expected_y = pd.Series(
        data=np.array([3., 4., 5., 6., 7., 8.], dtype=float),
        index=pd.RangeIndex(start=3, stop=9, step=1),
        name='y_step_1'
    )

    pd.testing.assert_frame_equal(X_step, expected_X)
    pd.testing.assert_series_equal(y_step, expected_y)
