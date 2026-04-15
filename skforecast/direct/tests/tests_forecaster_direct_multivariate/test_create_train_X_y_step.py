# Unit test _create_train_X_y_step ForecasterDirectMultiVariate
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirectMultiVariate


@pytest.mark.parametrize(
    "step, expected_y",
    [
        (1, np.array([3.0, 4.0, 5.0, 6.0, 7.0])),
        (2, np.array([4.0, 5.0, 6.0, 7.0, 8.0])),
        (3, np.array([5.0, 6.0, 7.0, 8.0, 9.0])),
    ],
    ids=["step_1", "step_2", "step_3"],
)
def test_create_train_X_y_step_output_when_exog_is_None(step, expected_y):
    """
    Test that _create_train_X_y_step returns X_train_autoreg unchanged and
    the correct y_train slice when there are no exogenous variables.
    """
    series = pd.DataFrame(
        {'l1': np.arange(10, dtype=float),
         'l2': np.arange(100, 110, dtype=float)}
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=3,
        transformer_series=None
    )
    X_train_autoreg, X_train_exog, y_train, *_ = forecaster._create_train_X_y(
        series=series
    )

    X_step, y_step = forecaster._create_train_X_y_step(
        X_train_autoreg=X_train_autoreg,
        X_train_exog=X_train_exog,
        y_train=y_train,
        step=step,
    )

    expected_X = np.array(
        [[  2.,   1.,   0., 102., 101., 100.],
         [  3.,   2.,   1., 103., 102., 101.],
         [  4.,   3.,   2., 104., 103., 102.],
         [  5.,   4.,   3., 105., 104., 103.],
         [  6.,   5.,   4., 106., 105., 104.]]
    )

    assert X_train_exog is None
    np.testing.assert_array_almost_equal(X_step, expected_X)
    np.testing.assert_array_almost_equal(y_step, expected_y)


@pytest.mark.parametrize(
    "step, expected_X, expected_y",
    [
        (
            1,
            np.array(
                [[ 2.,  1.,  0., 52., 51., 50., 103.],
                 [ 3.,  2.,  1., 53., 52., 51., 104.],
                 [ 4.,  3.,  2., 54., 53., 52., 105.],
                 [ 5.,  4.,  3., 55., 54., 53., 106.],
                 [ 6.,  5.,  4., 56., 55., 54., 107.],
                 [ 7.,  6.,  5., 57., 56., 55., 108.]]
            ),
            np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ),
        (
            2,
            np.array(
                [[ 2.,  1.,  0., 52., 51., 50., 104.],
                 [ 3.,  2.,  1., 53., 52., 51., 105.],
                 [ 4.,  3.,  2., 54., 53., 52., 106.],
                 [ 5.,  4.,  3., 55., 54., 53., 107.],
                 [ 6.,  5.,  4., 56., 55., 54., 108.],
                 [ 7.,  6.,  5., 57., 56., 55., 109.]]
            ),
            np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ),
    ],
    ids=["step_1", "step_2"],
)
def test_create_train_X_y_step_output_when_lags_3_steps_2_and_exog(
    step, expected_X, expected_y
):
    """
    Test output of _create_train_X_y_step when lags is 3 and steps is 2
    with a single exogenous variable.
    """
    series = pd.DataFrame(
        {'l1': np.arange(10, dtype=float),
         'l2': np.arange(50, 60, dtype=float)}
    )
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=2,
        transformer_series=None
    )
    X_train_autoreg, X_train_exog, y_train, *_ = forecaster._create_train_X_y(
        series=series, exog=exog
    )
    X_step, y_step = forecaster._create_train_X_y_step(
        X_train_autoreg=X_train_autoreg,
        X_train_exog=X_train_exog,
        y_train=y_train,
        step=step,
    )

    np.testing.assert_array_almost_equal(X_step, expected_X)
    np.testing.assert_array_almost_equal(y_step, expected_y)


@pytest.mark.parametrize(
    "step, expected_X, expected_y",
    [
        (
            1,
            np.array(
                [[ 2.,  1.,  0., 52., 51., 50., 103., 203.],
                 [ 3.,  2.,  1., 53., 52., 51., 104., 204.],
                 [ 4.,  3.,  2., 54., 53., 52., 105., 205.],
                 [ 5.,  4.,  3., 55., 54., 53., 106., 206.],
                 [ 6.,  5.,  4., 56., 55., 54., 107., 207.],
                 [ 7.,  6.,  5., 57., 56., 55., 108., 208.],
                 [ 8.,  7.,  6., 58., 57., 56., 109., 209.],
                 [ 9.,  8.,  7., 59., 58., 57., 110., 210.],
                 [10.,  9.,  8., 60., 59., 58., 111., 211.],
                 [11., 10.,  9., 61., 60., 59., 112., 212.]]
            ),
            np.array([3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]),
        ),
        (
            2,
            np.array(
                [[ 2.,  1.,  0., 52., 51., 50., 104., 204.],
                 [ 3.,  2.,  1., 53., 52., 51., 105., 205.],
                 [ 4.,  3.,  2., 54., 53., 52., 106., 206.],
                 [ 5.,  4.,  3., 55., 54., 53., 107., 207.],
                 [ 6.,  5.,  4., 56., 55., 54., 108., 208.],
                 [ 7.,  6.,  5., 57., 56., 55., 109., 209.],
                 [ 8.,  7.,  6., 58., 57., 56., 110., 210.],
                 [ 9.,  8.,  7., 59., 58., 57., 111., 211.],
                 [10.,  9.,  8., 60., 59., 58., 112., 212.],
                 [11., 10.,  9., 61., 60., 59., 113., 213.]]
            ),
            np.array([4., 5., 6., 7., 8., 9., 10., 11., 12., 13.]),
        ),
        (
            3,
            np.array(
                [[ 2.,  1.,  0., 52., 51., 50., 105., 205.],
                 [ 3.,  2.,  1., 53., 52., 51., 106., 206.],
                 [ 4.,  3.,  2., 54., 53., 52., 107., 207.],
                 [ 5.,  4.,  3., 55., 54., 53., 108., 208.],
                 [ 6.,  5.,  4., 56., 55., 54., 109., 209.],
                 [ 7.,  6.,  5., 57., 56., 55., 110., 210.],
                 [ 8.,  7.,  6., 58., 57., 56., 111., 211.],
                 [ 9.,  8.,  7., 59., 58., 57., 112., 212.],
                 [10.,  9.,  8., 60., 59., 58., 113., 213.],
                 [11., 10.,  9., 61., 60., 59., 114., 214.]]
            ),
            np.array([5., 6., 7., 8., 9., 10., 11., 12., 13., 14.]),
        ),
    ],
    ids=["step_1", "step_2", "step_3"],
)
def test_create_train_X_y_step_output_when_lags_3_steps_3_and_multiple_exog(
    step, expected_X, expected_y
):
    """
    Test output of _create_train_X_y_step when using multiple exogenous
    variables (DataFrame) with lags=3 and steps=3.
    """
    series = pd.DataFrame(
        {'l1': np.arange(15, dtype=float),
         'l2': np.arange(50, 65, dtype=float)}
    )
    exog = pd.DataFrame(
        {'exog_1': np.arange(100, 115, dtype=float),
         'exog_2': np.arange(200, 215, dtype=float)}
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=3, steps=3,
        transformer_series=None
    )
    X_train_autoreg, X_train_exog, y_train, *_ = forecaster._create_train_X_y(
        series=series, exog=exog
    )
    X_step, y_step = forecaster._create_train_X_y_step(
        X_train_autoreg=X_train_autoreg,
        X_train_exog=X_train_exog,
        y_train=y_train,
        step=step,
    )

    np.testing.assert_array_almost_equal(X_step, expected_X)
    np.testing.assert_array_almost_equal(y_step, expected_y)


@pytest.mark.parametrize(
    "step, expected_X, expected_y",
    [
        (
            1,
            np.array(
                [[ 4.,  3.,  2.,  1.,  0.,  2.,  2., 54., 53., 52., 51., 50., 52., 52., 105.],
                 [ 5.,  4.,  3.,  2.,  1.,  3.,  3., 55., 54., 53., 52., 51., 53., 53., 106.],
                 [ 6.,  5.,  4.,  3.,  2.,  4.,  4., 56., 55., 54., 53., 52., 54., 54., 107.],
                 [ 7.,  6.,  5.,  4.,  3.,  5.,  5., 57., 56., 55., 54., 53., 55., 55., 108.],
                 [ 8.,  7.,  6.,  5.,  4.,  6.,  6., 58., 57., 56., 55., 54., 56., 56., 109.],
                 [ 9.,  8.,  7.,  6.,  5.,  7.,  7., 59., 58., 57., 56., 55., 57., 57., 110.],
                 [10.,  9.,  8.,  7.,  6.,  8.,  8., 60., 59., 58., 57., 56., 58., 58., 111.],
                 [11., 10.,  9.,  8.,  7.,  9.,  9., 61., 60., 59., 58., 57., 59., 59., 112.],
                 [12., 11., 10.,  9.,  8., 10., 10., 62., 61., 60., 59., 58., 60., 60., 113.]]
            ),
            np.array([5., 6., 7., 8., 9., 10., 11., 12., 13.]),
        ),
        (
            2,
            np.array(
                [[ 4.,  3.,  2.,  1.,  0.,  2.,  2., 54., 53., 52., 51., 50., 52., 52., 106.],
                 [ 5.,  4.,  3.,  2.,  1.,  3.,  3., 55., 54., 53., 52., 51., 53., 53., 107.],
                 [ 6.,  5.,  4.,  3.,  2.,  4.,  4., 56., 55., 54., 53., 52., 54., 54., 108.],
                 [ 7.,  6.,  5.,  4.,  3.,  5.,  5., 57., 56., 55., 54., 53., 55., 55., 109.],
                 [ 8.,  7.,  6.,  5.,  4.,  6.,  6., 58., 57., 56., 55., 54., 56., 56., 110.],
                 [ 9.,  8.,  7.,  6.,  5.,  7.,  7., 59., 58., 57., 56., 55., 57., 57., 111.],
                 [10.,  9.,  8.,  7.,  6.,  8.,  8., 60., 59., 58., 57., 56., 58., 58., 112.],
                 [11., 10.,  9.,  8.,  7.,  9.,  9., 61., 60., 59., 58., 57., 59., 59., 113.],
                 [12., 11., 10.,  9.,  8., 10., 10., 62., 61., 60., 59., 58., 60., 60., 114.]]
            ),
            np.array([6., 7., 8., 9., 10., 11., 12., 13., 14.]),
        ),
    ],
    ids=["step_1", "step_2"],
)
def test_create_train_X_y_step_output_when_window_features_and_exog(
    step, expected_X, expected_y
):
    """
    Test output of _create_train_X_y_step when using window_features and
    exog with lags=5 and steps=2.
    """
    series = pd.DataFrame(
        {'l1': np.arange(15, dtype=float),
         'l2': np.arange(50, 65, dtype=float)},
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    exog = pd.Series(
        np.arange(100, 115, dtype=float), name='exog',
        index=pd.date_range('2000-01-01', periods=15, freq='D')
    )
    rolling = RollingFeatures(stats=['mean', 'median'], window_sizes=[5, 5])

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), steps=2, level='l1', lags=5,
        window_features=rolling, transformer_series=None
    )
    X_train_autoreg, X_train_exog, y_train, *_ = forecaster._create_train_X_y(
        series=series, exog=exog
    )
    X_step, y_step = forecaster._create_train_X_y_step(
        X_train_autoreg=X_train_autoreg,
        X_train_exog=X_train_exog,
        y_train=y_train,
        step=step,
    )

    np.testing.assert_array_almost_equal(X_step, expected_X)
    np.testing.assert_array_almost_equal(y_step, expected_y)


@pytest.mark.parametrize(
    "step, expected_X, expected_y",
    [
        (
            1,
            np.array(
                [[52., 51., 50., 103.],
                 [53., 52., 51., 104.],
                 [54., 53., 52., 105.],
                 [55., 54., 53., 106.],
                 [56., 55., 54., 107.],
                 [57., 56., 55., 108.]]
            ),
            np.array([53., 54., 55., 56., 57., 58.]),
        ),
        (
            2,
            np.array(
                [[52., 51., 50., 104.],
                 [53., 52., 51., 105.],
                 [54., 53., 52., 106.],
                 [55., 54., 53., 107.],
                 [56., 55., 54., 108.],
                 [57., 56., 55., 109.]]
            ),
            np.array([54., 55., 56., 57., 58., 59.]),
        ),
    ],
    ids=["step_1", "step_2"],
)
def test_create_train_X_y_step_output_when_lags_dict_with_None(
    step, expected_X, expected_y
):
    """
    Test output of _create_train_X_y_step when lags is a dict with None
    for one series. Only the series with lags contributes to autoreg features.
    """
    series = pd.DataFrame(
        {'l1': np.arange(10, dtype=float),
         'l2': np.arange(50, 60, dtype=float)}
    )
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=float)

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l2',
        lags={'l1': None, 'l2': 3}, steps=2, transformer_series=None
    )
    X_train_autoreg, X_train_exog, y_train, *_ = forecaster._create_train_X_y(
        series=series, exog=exog
    )
    X_step, y_step = forecaster._create_train_X_y_step(
        X_train_autoreg=X_train_autoreg,
        X_train_exog=X_train_exog,
        y_train=y_train,
        step=step,
    )

    np.testing.assert_array_almost_equal(X_step, expected_X)
    np.testing.assert_array_almost_equal(y_step, expected_y)


def test_create_train_X_y_step_matches_filter_train_X_y_for_step():
    """
    Test that _create_train_X_y_step produces the same numerical result as
    the public filter_train_X_y_for_step for every step.
    """
    series = pd.DataFrame(
        {'l1': np.arange(20, dtype=float),
         'l2': np.arange(50, 70, dtype=float)}
    )
    exog = pd.DataFrame(
        {'exog_1': np.arange(100, 120, dtype=float),
         'exog_2': np.arange(200, 220, dtype=float)}
    )

    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), level='l1', lags=5, steps=3,
        transformer_series=None
    )

    # Private path (numpy)
    X_train_autoreg, X_train_exog, y_train_private, *_ = (
        forecaster._create_train_X_y(series=series, exog=exog)
    )

    # Public path (pandas)
    X_train_public, y_train_public = forecaster.create_train_X_y(
        series=series, exog=exog
    )

    for step in [1, 2, 3]:
        X_step_private, y_step_private = forecaster._create_train_X_y_step(
            X_train_autoreg=X_train_autoreg,
            X_train_exog=X_train_exog,
            y_train=y_train_private,
            step=step,
        )
        X_step_public, y_step_public = forecaster.filter_train_X_y_for_step(
            step=step, X_train=X_train_public, y_train=y_train_public
        )

        np.testing.assert_array_almost_equal(
            X_step_private, X_step_public.to_numpy()
        )
        np.testing.assert_array_almost_equal(
            y_step_private, y_step_public.to_numpy()
        )
