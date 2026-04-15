# Unit test _binning_in_sample_residuals ForecasterRnn
# ==============================================================================
import os
import numpy as np
import pandas as pd
os.environ['KERAS_BACKEND'] = 'torch'
from skforecast.deep_learning import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn
from skforecast.preprocessing import QuantileBinner

# Fixtures
np.random.seed(123)
series = pd.DataFrame(
    {
        'l1': np.random.rand(50),
        'l2': np.random.rand(50),
    },
    index=pd.date_range(start='2000-01-01', periods=50, freq='MS')
)

model = create_and_compile_model(
    series=series,
    lags=3,
    steps=5,
    levels=['l1', 'l2'],
    recurrent_units=10,
    dense_units=5,
)


def test_binning_in_sample_residuals_output():
    """
    Test that _binning_in_sample_residuals returns the expected output.
    """
    forecaster = ForecasterRnn(
        estimator=model, levels=['l1', 'l2'], lags=3,
        binner_kwargs={'n_bins': 3}
    )

    rng = np.random.default_rng(123)
    y_pred = rng.normal(100, 15, 20)
    y_true = rng.normal(100, 10, 20)

    forecaster.in_sample_residuals_ = {}
    forecaster.in_sample_residuals_by_bin_ = {}
    forecaster._binning_in_sample_residuals(
        level='l1',
        y_pred=y_pred,
        y_true=y_true,
        store_in_sample_residuals=True
    )

    # In-sample residuals
    assert isinstance(forecaster.in_sample_residuals_, dict)
    assert 'l1' in forecaster.in_sample_residuals_
    assert len(forecaster.in_sample_residuals_['l1']) == 20

    # Binned residuals
    assert isinstance(forecaster.in_sample_residuals_by_bin_, dict)
    assert 'l1' in forecaster.in_sample_residuals_by_bin_
    assert isinstance(forecaster.in_sample_residuals_by_bin_['l1'], dict)
    assert len(forecaster.in_sample_residuals_by_bin_['l1']) == 3
    assert set(forecaster.in_sample_residuals_by_bin_['l1'].keys()) == {0, 1, 2}
    total_binned = sum(
        len(v) for v in forecaster.in_sample_residuals_by_bin_['l1'].values()
    )
    assert total_binned == 20

    # Binned residuals are a partition of total residuals
    all_binned = np.sort(np.concatenate(
        list(forecaster.in_sample_residuals_by_bin_['l1'].values())
    ))
    np.testing.assert_array_almost_equal(
        all_binned, np.sort(forecaster.in_sample_residuals_['l1'])
    )

    # Binner object
    assert isinstance(forecaster.binner['l1'], QuantileBinner)
    assert forecaster.binner['l1'].n_bins_ == 3

    # Binner intervals
    assert 'l1' in forecaster.binner_intervals_
    assert len(forecaster.binner_intervals_['l1']) == 3
    for interval in forecaster.binner_intervals_['l1'].values():
        assert isinstance(interval, tuple)
        assert len(interval) == 2


def test_binning_in_sample_residuals_store_in_sample_residuals_False():
    """
    Test that _binning_in_sample_residuals with store_in_sample_residuals False
    only stores binner intervals but not residuals.
    """
    forecaster = ForecasterRnn(
        estimator=model, levels=['l1', 'l2'], lags=3,
        binner_kwargs={'n_bins': 3}
    )

    rng = np.random.default_rng(123)
    y_pred = rng.normal(100, 15, 20)
    y_true = rng.normal(100, 10, 20)

    forecaster.in_sample_residuals_ = {}
    forecaster.in_sample_residuals_by_bin_ = {}
    forecaster._binning_in_sample_residuals(
        level='l1',
        y_pred=y_pred,
        y_true=y_true,
        store_in_sample_residuals=False
    )

    assert 'l1' not in forecaster.in_sample_residuals_
    assert 'l1' not in forecaster.in_sample_residuals_by_bin_

    # Binner and intervals are always stored
    assert isinstance(forecaster.binner['l1'], QuantileBinner)
    assert forecaster.binner['l1'].n_bins_ == 3
    assert 'l1' in forecaster.binner_intervals_
    assert len(forecaster.binner_intervals_['l1']) == 3


def test_binning_in_sample_residuals_stores_maximum_10000_residuals():
    """
    Test that _binning_in_sample_residuals stores a maximum of 10_000 residuals.
    """
    forecaster = ForecasterRnn(
        estimator=model, levels=['l1', 'l2'], lags=3,
        binner_kwargs={'n_bins': 5}
    )

    rng = np.random.default_rng(123)
    y_pred = rng.normal(100, 15, 20_000)
    y_true = rng.normal(100, 10, 20_000)

    forecaster.in_sample_residuals_ = {}
    forecaster.in_sample_residuals_by_bin_ = {}
    forecaster._binning_in_sample_residuals(
        level='l1',
        y_pred=y_pred,
        y_true=y_true,
        store_in_sample_residuals=True
    )

    assert len(forecaster.in_sample_residuals_['l1']) == 10_000
    assert len(forecaster.in_sample_residuals_by_bin_['l1']) == 5
    for v in forecaster.in_sample_residuals_by_bin_['l1'].values():
        assert len(v) <= 10_000 // 5


def test_binning_in_sample_residuals_multiple_levels():
    """
    Test that _binning_in_sample_residuals works with multiple levels.
    """
    forecaster = ForecasterRnn(
        estimator=model, levels=['l1', 'l2'], lags=3,
        binner_kwargs={'n_bins': 3}
    )

    rng = np.random.default_rng(123)

    forecaster.in_sample_residuals_ = {}
    forecaster.in_sample_residuals_by_bin_ = {}

    for level in ['l1', 'l2']:
        y_pred = rng.normal(100, 15, 50)
        y_true = rng.normal(100, 10, 50)
        forecaster._binning_in_sample_residuals(
            level=level,
            y_pred=y_pred,
            y_true=y_true,
            store_in_sample_residuals=True
        )

    assert set(forecaster.in_sample_residuals_.keys()) == {'l1', 'l2'}
    assert set(forecaster.in_sample_residuals_by_bin_.keys()) == {'l1', 'l2'}
    assert set(forecaster.binner.keys()) == {'l1', 'l2'}
    assert set(forecaster.binner_intervals_.keys()) == {'l1', 'l2'}

    # Each level has its own independent binner
    assert forecaster.binner['l1'] is not forecaster.binner['l2']
    for level in ['l1', 'l2']:
        assert len(forecaster.in_sample_residuals_by_bin_[level]) == 3
        assert len(forecaster.in_sample_residuals_[level]) == 50
