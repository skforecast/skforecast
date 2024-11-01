# Unit test QantityBinner class
# ==============================================================================
import re
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
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


def test_QuantileBinner_fit_ValueError_when_input_data_is_empty():
    """
    Test ValueError is raised when input data is empty during fit.
    """
    
    X = np.array([])
    binner = QuantileBinner(
        n_bins=10,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )
    
    err_msg = re.escape("Input data `X` cannot be empty.")
    with pytest.raises(ValueError, match = err_msg):
        binner.fit(X)


def test_QuantileBinner_transform_NotFittedError():
    """
    Test NotFittedError is raised when transform is used before fitting.
    """
    
    X = np.array([])
    binner = QuantileBinner(
        n_bins=10,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )
    
    err_msg = re.escape(
        "The model has not been fitted yet. Call 'fit' with training data first."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        binner.transform(X)


def test_QuantileBinner_set_params():
    """
    Test set_params method.
    """
        
    binner = QuantileBinner(
        n_bins=10,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )
    
    params = {
        'n_bins': 5,
        'method': 'inverse_cdf',
        'dtype': np.float32,
        'random_state': 123456,
    }
    
    binner.set_params(**params)
    
    assert binner.n_bins == params['n_bins']
    assert binner.method == params['method']
    assert binner.dtype == params['dtype']
    assert binner.random_state == params['random_state']


def test_QuantileBinner_fit_with_subsample():
    """
    Test QuantileBinner fit method with subsample.
    """
    
    X = np.arange(1000)
    binner = QuantileBinner(
        n_bins=10,
        method='linear',
        subsample=10,
        dtype=np.float64,
        random_state=789654,
    )
    
    binner.fit(X.reshape(-1, 1))

    expected_bin_edges_ = np.array(
        [21., 159.6, 283.8, 388.7, 429.8, 578., 725., 731.6, 734.2,
         739.4, 743.]
    )
    expected_n_bins_ = 10
    expected_intervals_ = {
        0.0: (21.0, 159.6),
        1.0: (159.6, 283.8),
        2.0: (283.8, 388.7),
        3.0: (388.7, 429.8),
        4.0: (429.8, 578.0),
        5.0: (578.0, 725.0),
        6.0: (725.0, 731.6),
        7.0: (731.6, 734.2),
        8.0: (734.2, 739.4),
        9.0: (739.4, 743.0)
    }

    np.testing.assert_array_almost_equal(binner.bin_edges_, expected_bin_edges_)
    assert binner.n_bins_ == expected_n_bins_
    assert binner.intervals_ == expected_intervals_


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
