# Unit test _create_train_X_y_step ForecasterDirect
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.preprocessing import RollingFeatures
from skforecast.direct import ForecasterDirect


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
    y = pd.Series(np.arange(10), name="y", dtype=float)

    forecaster = ForecasterDirect(estimator=LinearRegression(), lags=3, steps=3)
    X_train_autoreg, X_train_exog, y_train, *_ = forecaster._create_train_X_y(y=y)

    X_step, y_step = forecaster._create_train_X_y_step(
        X_train_autoreg=X_train_autoreg,
        X_train_exog=X_train_exog,
        y_train=y_train,
        step=step,
    )

    expected_X = np.array(
        [
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0],
            [5.0, 4.0, 3.0],
            [6.0, 5.0, 4.0],
        ]
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
                [
                    [2.0, 1.0, 0.0, 103.0],
                    [3.0, 2.0, 1.0, 104.0],
                    [4.0, 3.0, 2.0, 105.0],
                    [5.0, 4.0, 3.0, 106.0],
                    [6.0, 5.0, 4.0, 107.0],
                    [7.0, 6.0, 5.0, 108.0],
                ]
            ),
            np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ),
        (
            2,
            np.array(
                [
                    [2.0, 1.0, 0.0, 104.0],
                    [3.0, 2.0, 1.0, 105.0],
                    [4.0, 3.0, 2.0, 106.0],
                    [5.0, 4.0, 3.0, 107.0],
                    [6.0, 5.0, 4.0, 108.0],
                    [7.0, 6.0, 5.0, 109.0],
                ]
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
    y = pd.Series(np.arange(10), name="y", dtype=float)
    exog = pd.Series(np.arange(100, 110), name="exog", dtype=float)

    forecaster = ForecasterDirect(estimator=LinearRegression(), lags=3, steps=2)
    X_train_autoreg, X_train_exog, y_train, *_ = forecaster._create_train_X_y(
        y=y, exog=exog
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
                [
                    [2.0, 1.0, 0.0, 103.0, 203.0],
                    [3.0, 2.0, 1.0, 104.0, 204.0],
                    [4.0, 3.0, 2.0, 105.0, 205.0],
                    [5.0, 4.0, 3.0, 106.0, 206.0],
                    [6.0, 5.0, 4.0, 107.0, 207.0],
                    [7.0, 6.0, 5.0, 108.0, 208.0],
                    [8.0, 7.0, 6.0, 109.0, 209.0],
                    [9.0, 8.0, 7.0, 110.0, 210.0],
                    [10.0, 9.0, 8.0, 111.0, 211.0],
                    [11.0, 10.0, 9.0, 112.0, 212.0],
                ]
            ),
            np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ),
        (
            2,
            np.array(
                [
                    [2.0, 1.0, 0.0, 104.0, 204.0],
                    [3.0, 2.0, 1.0, 105.0, 205.0],
                    [4.0, 3.0, 2.0, 106.0, 206.0],
                    [5.0, 4.0, 3.0, 107.0, 207.0],
                    [6.0, 5.0, 4.0, 108.0, 208.0],
                    [7.0, 6.0, 5.0, 109.0, 209.0],
                    [8.0, 7.0, 6.0, 110.0, 210.0],
                    [9.0, 8.0, 7.0, 111.0, 211.0],
                    [10.0, 9.0, 8.0, 112.0, 212.0],
                    [11.0, 10.0, 9.0, 113.0, 213.0],
                ]
            ),
            np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
        ),
        (
            3,
            np.array(
                [
                    [2.0, 1.0, 0.0, 105.0, 205.0],
                    [3.0, 2.0, 1.0, 106.0, 206.0],
                    [4.0, 3.0, 2.0, 107.0, 207.0],
                    [5.0, 4.0, 3.0, 108.0, 208.0],
                    [6.0, 5.0, 4.0, 109.0, 209.0],
                    [7.0, 6.0, 5.0, 110.0, 210.0],
                    [8.0, 7.0, 6.0, 111.0, 211.0],
                    [9.0, 8.0, 7.0, 112.0, 212.0],
                    [10.0, 9.0, 8.0, 113.0, 213.0],
                    [11.0, 10.0, 9.0, 114.0, 214.0],
                ]
            ),
            np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]),
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
    y = pd.Series(np.arange(15), name="y", dtype=float)
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 115, dtype=float),
            "exog_2": np.arange(200, 215, dtype=float),
        }
    )

    forecaster = ForecasterDirect(estimator=LinearRegression(), lags=3, steps=3)
    X_train_autoreg, X_train_exog, y_train, *_ = forecaster._create_train_X_y(
        y=y, exog=exog
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
                [
                    [4.0, 3.0, 2.0, 1.0, 0.0, 2.0, 2.0, 105.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0, 3.0, 3.0, 106.0],
                    [6.0, 5.0, 4.0, 3.0, 2.0, 4.0, 4.0, 107.0],
                    [7.0, 6.0, 5.0, 4.0, 3.0, 5.0, 5.0, 108.0],
                    [8.0, 7.0, 6.0, 5.0, 4.0, 6.0, 6.0, 109.0],
                    [9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 7.0, 110.0],
                    [10.0, 9.0, 8.0, 7.0, 6.0, 8.0, 8.0, 111.0],
                    [11.0, 10.0, 9.0, 8.0, 7.0, 9.0, 9.0, 112.0],
                    [12.0, 11.0, 10.0, 9.0, 8.0, 10.0, 10.0, 113.0],
                ]
            ),
            np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]),
        ),
        (
            2,
            np.array(
                [
                    [4.0, 3.0, 2.0, 1.0, 0.0, 2.0, 2.0, 106.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0, 3.0, 3.0, 107.0],
                    [6.0, 5.0, 4.0, 3.0, 2.0, 4.0, 4.0, 108.0],
                    [7.0, 6.0, 5.0, 4.0, 3.0, 5.0, 5.0, 109.0],
                    [8.0, 7.0, 6.0, 5.0, 4.0, 6.0, 6.0, 110.0],
                    [9.0, 8.0, 7.0, 6.0, 5.0, 7.0, 7.0, 111.0],
                    [10.0, 9.0, 8.0, 7.0, 6.0, 8.0, 8.0, 112.0],
                    [11.0, 10.0, 9.0, 8.0, 7.0, 9.0, 9.0, 113.0],
                    [12.0, 11.0, 10.0, 9.0, 8.0, 10.0, 10.0, 114.0],
                ]
            ),
            np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]),
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
    y = pd.Series(
        np.arange(15),
        index=pd.date_range("2000-01-01", periods=15, freq="D"),
        name="y",
        dtype=float,
    )
    exog = pd.Series(
        np.arange(100, 115),
        index=pd.date_range("2000-01-01", periods=15, freq="D"),
        name="exog",
        dtype=float,
    )
    rolling = RollingFeatures(stats=["mean", "median"], window_sizes=[5, 5])

    forecaster = ForecasterDirect(
        estimator=LinearRegression(), steps=2, lags=5, window_features=rolling
    )
    X_train_autoreg, X_train_exog, y_train, *_ = forecaster._create_train_X_y(
        y=y, exog=exog
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
    y = pd.Series(np.arange(20), name="y", dtype=float)
    exog = pd.DataFrame(
        {
            "exog_1": np.arange(100, 120, dtype=float),
            "exog_2": np.arange(200, 220, dtype=float),
        }
    )

    forecaster = ForecasterDirect(estimator=LinearRegression(), lags=5, steps=3)

    # Private path (numpy)
    X_train_autoreg, X_train_exog, y_train_private, *_ = (
        forecaster._create_train_X_y(y=y, exog=exog)
    )

    # Public path (pandas)
    X_train_public, y_train_public = forecaster.create_train_X_y(y=y, exog=exog)

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
