# Unit test _binning_in_sample_residuals ForecasterDirect
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.direct import ForecasterDirect


def test_binning_in_sample_residuals_output():
    """
    Test that _binning_in_sample_residuals returns the expected output.
    """

    forecaster = ForecasterDirect(
        LinearRegression(), lags=3, steps=2, binner_kwargs={'n_bins': 3}
    )

    rng = np.random.default_rng(123)
    y_pred = rng.normal(100, 15, 20)
    y_true = rng.normal(100, 10, 20)

    forecaster._binning_in_sample_residuals(
        y_pred=y_pred,
        y_true=y_true,
    )

    expected_residuals_binned = {
        0: np.array([
               -7.23789073,  13.79601419,  22.36597697,   4.88697987,
                10.58663414, -11.65409374,  27.51615357
           ]),
        1: np.array([
               8.35845165,  8.67514021, -5.0751785 , -0.40101616, 14.61016061,
               -7.98175116
           ]),
        2: np.array([
               -3.90257497,  -6.25576705, -10.1163358 ,   2.61102291,
               -5.62019863, -18.70551475, -14.38168435
           ])
    }
    expected_binner_intervals = {
        0.0: (77.11104390221573, 95.19313158258132),
        1.0: (95.19313158258132, 107.10836783689707),
        2.0: (107.10836783689707, 122.98049619443195)
    }

    for k in forecaster.in_sample_residuals_by_bin_.keys():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_by_bin_[k], expected_residuals_binned[k]
        )
    for k in forecaster.binner_intervals_.keys():
        assert forecaster.binner_intervals_[k][0] == approx(expected_binner_intervals[k][0])
        assert forecaster.binner_intervals_[k][1] == approx(expected_binner_intervals[k][1])


def test_binning_in_sample_residuals_stores_maximum_10000_residuals():
    """
    Test that maximum 10_000 residuals are stored.
    """

    n = 15000
    y = pd.Series(
            data = np.random.normal(loc=10, scale=1, size=n),
            index = pd.date_range(start='01-01-2000', periods=n, freq='h')
        )

    forecaster = ForecasterDirect(
        LinearRegression(), lags=3, steps=2, binner_kwargs={'n_bins': 3}
    )
    forecaster.fit(y)
    max_residuals_per_bin = int(10_000 // forecaster.binner.n_bins_)

    for v in forecaster.in_sample_residuals_.values():
        assert len(v) == 10_000
    for v in forecaster.in_sample_residuals_by_bin_.values():
        assert len(v) == max_residuals_per_bin
