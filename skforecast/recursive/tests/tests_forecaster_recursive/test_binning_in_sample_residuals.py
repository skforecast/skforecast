# Unit test _binning_in_sample_residuals ForecasterRecursive
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.recursive import ForecasterRecursive


def test_binning_in_sample_residuals_output():
    """
    Test that _binning_in_sample_residuals returns the expected output.
    """

    forecaster = ForecasterRecursive(
        regressor=object(),
        lags = 5,
        binner_kwargs={'n_bins': 3}
    )

    rng = np.random.default_rng(12345)
    y_pred = rng.normal(100, 15, 20)
    y_true = rng.normal(100, 10, 20)

    forecaster._binning_in_sample_residuals(
        y_pred=y_pred,
        y_true=y_true,
    )

    expected_1 = np.array([
                     34.58035615,  22.08911948,  15.6081091 ,   7.0808798 ,
                     55.47454021,   1.80092458,  12.2624188 , -12.32822882,
                     -0.451743  ,  11.83152763,  -7.11862253,   6.32581108,
                     -20.92461257, -21.95291202, -10.55026794, -27.43753138,
                     -6.24112163, -25.62685698,  -4.31327121, -18.40910724
                 ])

    expected_2 = {
        0: np.array([
               34.58035615, 22.08911948, 15.6081091,  7.0808798, 55.47454021,
               1.80092458, 12.2624188
           ]),
        1: np.array([
               -12.32822882,  -0.45174300,  11.83152763,  -7.11862253,
               6.32581108, -20.92461257
           ]),
        2: np.array([
               -21.95291202, -10.55026794, -27.43753138,  -6.24112163,
               -25.62685698,  -4.31327121, -18.40910724
           ])
    }

    expected_3 = {
        0: (70.70705405481715, 90.25638761254116),
        1: (90.25638761254116, 109.36821559391004),
        2: (109.36821559391004, 135.2111448156828)
    }

    np.testing.assert_almost_equal(forecaster.in_sample_residuals_, expected_1)
    for k in expected_2.keys():
        np.testing.assert_almost_equal(forecaster.in_sample_residuals_by_bin_[k], expected_2[k])
    assert forecaster.binner_intervals_ == expected_3


def test_binning_in_sample_residuals_stores_maximum_10000_residuals():
    """
    Test that maximum 10000 residuals are stored.
    """

    n = 15000
    y = pd.Series(
            data = np.random.normal(loc=10, scale=1, size=n),
            index = pd.date_range(start='01-01-2000', periods=n, freq='h')
        )
    forecaster = ForecasterRecursive(
                    regressor=LinearRegression(),
                    lags = 5,
                    binner_kwargs={'n_bins': 2}
                )
    forecaster.fit(y)
    max_residuals_per_bin = int(10_000 // forecaster.binner.n_bins_)

    for v in forecaster.in_sample_residuals_by_bin_.values():
        assert len(v) == max_residuals_per_bin
        assert len(v) > 0

    np.testing.assert_array_almost_equal(
        forecaster.in_sample_residuals_,
        np.concatenate(list(forecaster.in_sample_residuals_by_bin_.values()))
    )

    assert len(forecaster.in_sample_residuals_) == 10_000
