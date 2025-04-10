# Unit test _binning_in_sample_residuals ForecasterDirectMultiVariate
# ==============================================================================
from pytest import approx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from skforecast.direct import ForecasterDirectMultiVariate


def test_binning_in_sample_residuals_output():
    """
    Test that _binning_in_sample_residuals returns the expected output.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=3, binner_kwargs={'n_bins': 3}
    )

    rng = np.random.default_rng(123)
    y_pred = rng.normal(100, 15, 20)
    y_true = rng.normal(100, 10, 20)

    forecaster.in_sample_residuals_ = {}
    forecaster.in_sample_residuals_by_bin_ = {}
    forecaster._binning_in_sample_residuals(
        level=forecaster.level,
        y_pred=y_pred,
        y_true=y_true,
        store_in_sample_residuals=True
    )

    expected_in_sample_residuals_ = {
        'l1': np.array([
            -7.23789073,  13.79601419,  -3.90257497,   8.35845165,
            -6.25576705, -10.1163358 ,  22.36597697,   2.61102291,
            8.67514021,   4.88697987,  -5.0751785 ,  10.58663414,
            -5.62019863, -11.65409374, -18.70551475,  -0.40101616,
            -14.38168435,  27.51615357,  14.61016061,  -7.98175116
        ])
    }

    expected_residuals_binned = {
        'l1': {
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
    }
    expected_binner_intervals = {
        'l1': {
            0: (77.11104390221573, 95.19313158258132),
            1: (95.19313158258132, 107.10836783689707),
            2: (107.10836783689707, 122.98049619443195)
        }
    }

    # In-sample residuals
    assert isinstance(forecaster.in_sample_residuals_, dict)
    assert forecaster.in_sample_residuals_.keys() == expected_in_sample_residuals_.keys()
    np.testing.assert_array_almost_equal(
        forecaster.in_sample_residuals_['l1'],
        expected_in_sample_residuals_['l1']
    )

    for k in forecaster.in_sample_residuals_by_bin_['l1'].keys():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_by_bin_['l1'][k], expected_residuals_binned['l1'][k]
        )
    for k in forecaster.binner_intervals_['l1'].keys():
        assert forecaster.binner_intervals_['l1'][k][0] == approx(expected_binner_intervals['l1'][k][0])
        assert forecaster.binner_intervals_['l1'][k][1] == approx(expected_binner_intervals['l1'][k][1])


def test_binning_in_sample_residuals_store_in_sample_residuals_False():
    """
    Test that _binning_in_sample_residuals store_in_sample_residuals False.
    """
    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=3, binner_kwargs={'n_bins': 3}
    )

    rng = np.random.default_rng(123)
    y_pred = rng.normal(100, 15, 20)
    y_true = rng.normal(100, 10, 20)

    forecaster._binning_in_sample_residuals(
        level = 'l1',
        y_pred=y_pred,
        y_true=y_true,
        store_in_sample_residuals=False
    )

    expected_binner_intervals = {
        'l1': {
            0: (77.11104390221573, 95.19313158258132),
            1: (95.19313158258132, 107.10836783689707),
            2: (107.10836783689707, 122.98049619443195)
        }
    }

    assert forecaster.in_sample_residuals_ is None
    assert forecaster.in_sample_residuals_by_bin_ is None
    for k in forecaster.binner_intervals_['l1'].keys():
        assert forecaster.binner_intervals_['l1'][k][0] == approx(expected_binner_intervals['l1'][k][0])
        assert forecaster.binner_intervals_['l1'][k][1] == approx(expected_binner_intervals['l1'][k][1])


def test_binning_in_sample_residuals_probabilistic_mode_no_binned():
    """
    Test that _binning_in_sample_residuals stores when _probabilistic_mode is
    no binned.
    """

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=3, binner_kwargs={'n_bins': 3}
    )

    rng = np.random.default_rng(123)
    y_pred = rng.normal(100, 15, 20)
    y_true = rng.normal(100, 10, 20)

    forecaster._probabilistic_mode = "no_binned"
    forecaster.in_sample_residuals_ = {}
    forecaster.in_sample_residuals_by_bin_ = {}
    forecaster.binner_intervals_ = {}
    forecaster._binning_in_sample_residuals(
        level = 'l1',
        y_pred=y_pred,
        y_true=y_true,
        store_in_sample_residuals=True
    )

    expected_in_sample_residuals_ = {
        'l1': np.array([
            -7.23789073,  13.79601419,  -3.90257497,   8.35845165,
            -6.25576705, -10.1163358 ,  22.36597697,   2.61102291,
            8.67514021,   4.88697987,  -5.0751785 ,  10.58663414,
            -5.62019863, -11.65409374, -18.70551475,  -0.40101616,
            -14.38168435,  27.51615357,  14.61016061,  -7.98175116
        ])
    }

    # In-sample residuals
    assert isinstance(forecaster.in_sample_residuals_, dict)
    assert forecaster.in_sample_residuals_.keys() == expected_in_sample_residuals_.keys()
    np.testing.assert_array_almost_equal(
        forecaster.in_sample_residuals_['l1'], expected_in_sample_residuals_['l1']
    )

    assert forecaster.in_sample_residuals_by_bin_ == {'l1': None}
    assert forecaster.binner_intervals_ == {}


def test_binning_in_sample_residuals_stores_maximum_10000_residuals():
    """
    Test that maximum 10_000 residuals are stored.
    """
    n = 15000
    series = pd.DataFrame(
        data = {
            'l1': np.random.normal(loc=10, scale=1, size=n),
            'l2': np.random.normal(loc=20, scale=2, size=n),
        },
        index = pd.date_range(start='01-01-2000', periods=n, freq='h')
    )

    forecaster = ForecasterDirectMultiVariate(
        LinearRegression(), level='l1', steps=2, lags=3, binner_kwargs={'n_bins': 3}
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    max_residuals_per_bin = int(10_000 // forecaster.binner['l1'].n_bins_)

    for v in forecaster.in_sample_residuals_.values():
        assert len(v) == 10_000
    for v in forecaster.in_sample_residuals_by_bin_['l1'].values():
        assert len(v) == max_residuals_per_bin
