# Unit test _binning_in_sample_residuals ForecasterRecursiveMultiseries
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursiveMultiSeries


def test_binning_in_sample_residuals_output():
    """
    Test that _binning_in_sample_residuals returns the expected output.
    """

    forecaster = ForecasterRecursiveMultiSeries(
        regressor=LinearRegression(), lags=5, binner_kwargs={"n_bins": 3}
    )

    series = pd.DataFrame(
        {
            "l1": pd.Series(np.arange(10)),
            "l2": pd.Series(np.arange(10)),
            "l3": pd.Series(np.arange(10)),
        }
    )
    series.index = pd.DatetimeIndex(
        [
            "2022-01-04",
            "2022-01-05",
            "2022-01-06",
            "2022-01-07",
            "2022-01-08",
            "2022-01-09",
            "2022-01-10",
            "2022-01-11",
            "2022-01-12",
            "2022-01-13",
        ],
        dtype="datetime64[ns]",
        freq="D",
    )
    forecaster.fit(series=series)

    rng = np.random.default_rng(12345)
    y_pred = rng.normal(100, 15, 20)
    y_true = rng.normal(100, 10, 20)
    forecaster.in_sample_residuals_ = {"level_1": None}
    forecaster.in_sample_residuals_by_bin_ = {"level_1": {}}
    forecaster.binner_intervals_ = {"level_1": {}}

    forecaster._binning_in_sample_residuals(
        level="level_1",
        y_pred=y_pred,
        y_true=y_true,
    )

    expected_residuals = {
        "level_1": np.array(
            [
                34.58035615,
                -21.95291202,
                22.08911948,
                -12.32822882,
                -0.451743,
                15.6081091,
                7.0808798,
                -10.55026794,
                11.83152763,
                55.47454021,
                -27.43753138,
                -6.24112163,
                1.80092458,
                -25.62685698,
                -7.11862253,
                6.32581108,
                -4.31327121,
                12.2624188,
                -20.92461257,
                -18.40910724,
            ]
        )
    }

    expected_residuals_by_bin = {
        "level_1": {
            0: np.array(
                [
                    34.58035615,
                    22.08911948,
                    15.6081091,
                    7.0808798,
                    55.47454021,
                    1.80092458,
                    12.2624188,
                ]
            ),
            1: np.array(
                [
                    -12.32822882,
                    -0.45174300,
                    11.83152763,
                    -7.11862253,
                    6.32581108,
                    -20.92461257,
                ]
            ),
            2: np.array(
                [
                    -21.95291202,
                    -10.55026794,
                    -27.43753138,
                    -6.24112163,
                    -25.62685698,
                    -4.31327121,
                    -18.40910724,
                ]
            ),
        }
    }

    expected_binner_intervals = {
        "level_1": {
            0: (70.70705405481715, 90.25638761254116),
            1: (90.25638761254116, 109.36821559391004),
            2: (109.36821559391004, 135.2111448156828),
        },
    }

    assert all(
        np.allclose(forecaster.in_sample_residuals_[key], expected_residuals[key])
        for key in forecaster.in_sample_residuals_
    )
    for k in expected_residuals_by_bin.keys():
        for bin in [0, 1, 2]:
            np.testing.assert_almost_equal(
                forecaster.in_sample_residuals_by_bin_[k][bin],
                expected_residuals_by_bin[k][bin],
            )
    assert forecaster.binner_intervals_ == expected_binner_intervals


def test_binning_in_sample_residuals_stores_maximum_10000_residuals_per_level():
    """
    Test that maximum 10_000 residuals are stored per ñevel and 10_000 / n_bins per bin.
    """

    forecaster = ForecasterRecursiveMultiSeries(
        regressor=LinearRegression(), lags=5, binner_kwargs={"n_bins": 3}
    )

    rng = np.random.default_rng(12345)
    series = pd.DataFrame(
        {
            "l1": rng.normal(100, 15, 15_000),
            "l2": rng.normal(100, 15, 15_000),
            "l3": rng.normal(100, 15, 15_000),
        },
        index=pd.date_range(start="1990-01-01", periods=15_000, freq="h"),
    )
    forecaster.fit(series=series)
    
    y_pred = rng.normal(100, 15, 15_000)
    y_true = rng.normal(100, 10, 15_000)
    forecaster.in_sample_residuals_ = {"level_1": None}
    forecaster.in_sample_residuals_by_bin_ = {"level_1": {}}

    forecaster._binning_in_sample_residuals(
        level="level_1",
        y_pred=y_pred,
        y_true=y_true,
    )

    for k in forecaster.in_sample_residuals_.keys():
        assert len(forecaster.in_sample_residuals_[k]) == 10_000

    for k in forecaster.in_sample_residuals_by_bin_.keys():
        max_size = 10_000 // 3
        for bin in [0, 1, 2]:
            assert len(forecaster.in_sample_residuals_by_bin_[k][bin]) == max_size
