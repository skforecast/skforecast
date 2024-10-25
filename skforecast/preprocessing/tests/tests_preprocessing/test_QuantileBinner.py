# Unit test QantityBinner class
# ==============================================================================
import re
import pytest
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from ...preprocessing import QuantileBinner


def test_QuantileBinner_validate_params():
    """
    Test RollingFeatures _validate_params method.
    """

    params = {
        0: {'n_bins': 1, 'method': 'linear', 'dtype': np.float64, 'subsample': 20000, 'random_state': 123},
        1: {'n_bins': 5, 'method': 'not_valid_method', 'dtype': np.float64, 'subsample': 20000, 'random_state': 123},
        2: {'n_bins': 5, 'method': 'linear', 'dtype': 'not_valid_dtype', 'subsample': 20000, 'random_state': 123},
        3: {'n_bins': 5, 'method': 'linear', 'dtype': np.float64, 'subsample': 'not_int', 'random_state': 123},
        4: {'n_bins': 5, 'method': 'linear', 'dtype': np.float64, 'subsample': 20000, 'random_state': 'not_int'},
    }
        
    err_msg = re.escape(
        f"`n_bins` must be an int greater than 1. Got {params[0]['n_bins']}."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        QuantileBinner(**params[0])

    valid_methods = [
            "inverse_cdf",
            "averaged_inverse_cdf",
            "closest_observation",
            "interpolated_inverse_cdf",
            "hazen",
            "weibull",
            "linear",
            "median_unbiased",
            "normal_unbiased",
        ]
    err_msg = re.escape(
        f"`method` must be one of {valid_methods}. Got {params[1]['method']}."
    )
    with pytest.raises(ValueError, match = err_msg):
        QuantileBinner(**params[1])

    err_msg = re.escape(
        f"`dtype` must be a valid numpy dtype. Got {params[2]['dtype']}."
    )
    with pytest.raises(ValueError, match = err_msg):
        QuantileBinner(**params[2])
    
    err_msg = re.escape(
        f"`subsample` must be an integer greater than or equal to 1. "
        f"Got {params[3]['subsample']}."
    )
    with pytest.raises(ValueError, match = err_msg):
        QuantileBinner(**params[3])
    
    err_msg = re.escape(
        f"`random_state` must be an integer greater than or equal to 0. "
        f"Got {params[4]['random_state']}."
    )
    with pytest.raises(ValueError, match = err_msg):
        QuantileBinner(**params[4])


def test_QuantileBinner_is_equivalent_to_KBinsDiscretizer():
    """
    Test that QuantileBinner is equivalent to KBinsDiscretizer when `method='linear'`.
    """
    
    X = np.random.normal(10, 10, 10000)
    n_bins_grid = [2, 10, 20]

    for n_bins in n_bins_grid:
        binner_1 = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="quantile",
            dtype=np.float64,
            random_state=789654,
        )
        binner_2 = QuantileBinner(
            n_bins=n_bins,
            method='linear',
            dtype=np.float64,
            random_state=789654,
        )

        binner_1.fit(X.reshape(-1, 1))
        binner_2.fit(X)

        transformed_1 = binner_1.transform(X.reshape(-1, 1)).flatten()
        transformed_2 = binner_2.transform(X)

        np.testing.assert_array_almost_equal(binner_1.bin_edges_[0], binner_2.bin_edges_)
        np.testing.assert_array_almost_equal(transformed_1, transformed_2)