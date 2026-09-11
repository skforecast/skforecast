# Unit test QantityBinner class
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import KBinsDiscretizer
from ..._preprocessing import QuantileBinner
from skforecast.exceptions import IgnoredArgumentWarning


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
        0: (21.0, 159.6),
        1: (159.6, 283.8),
        2: (283.8, 388.7),
        3: (388.7, 429.8),
        4: (429.8, 578.0),
        5: (578.0, 725.0),
        6: (725.0, 731.6),
        7: (731.6, 734.2),
        8: (734.2, 739.4),
        9: (739.4, 743.0)
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
            quantile_method='linear',
        )
        binner_2 = QuantileBinner(
            n_bins=n_bins,
            method='linear',
            dtype=np.float64,
            random_state=789654,
        )

        binner_1.fit(X.reshape(-1, 1))
        transformed_1 = binner_1.transform(X.reshape(-1, 1)).flatten()
        transformed_2 = binner_2.fit_transform(X)

        np.testing.assert_array_almost_equal(binner_1.bin_edges_[0], binner_2.bin_edges_)
        np.testing.assert_array_almost_equal(transformed_1, transformed_2)


def test_QuantileBinner_fit_with_duplicate_edges_raises_warning():
    """
    Test that QuantileBinner raises a warning when duplicate edges are removed
    due to repeated values in the data.
    """
    
    # Data with many repeated values that will cause duplicate edges
    # Two unique values (1 and 2) will result in 2 bins instead of 10
    X = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    binner = QuantileBinner(
        n_bins=10,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )
    
    warn_msg = re.escape(
        "The number of bins has been reduced from 10 to 2 due to duplicated "
        "edges. This happens when "
        "the values used to compute the edges of the bins are highly "
        "concentrated or contain many repeated values.",
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        binner.fit(X)
    
    # Check that n_bins_ is reduced to 2 (one bin per unique value)
    assert binner.n_bins_ == 2
    assert binner.n_bins_ < binner.n_bins


def test_QuantileBinner_fit_with_identical_values():
    """
    Test that QuantileBinner handles data with all identical values correctly,
    creating at least 1 bin.
    """
    
    X = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    binner = QuantileBinner(
        n_bins=10,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )
    
    warn_msg = re.escape(
        "The number of bins has been reduced from 10 to 1 due to duplicated "
        "edges. This happens when "
        "the values used to compute the edges of the bins are highly "
        "concentrated or contain many repeated values.",
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        binner.fit(X)
    
    # Check that at least 1 bin is created
    assert binner.n_bins_ == 1
    assert len(binner.bin_edges_) == 2
    assert binner.bin_edges_[0] == 5.0
    assert binner.bin_edges_[1] == 5.0
    
    # Check that transform works correctly
    transformed = binner.transform(X)
    np.testing.assert_array_equal(transformed, np.zeros(5))


def test_QuantileBinner_fit_no_warning_when_bins_not_reduced():
    """
    Test that QuantileBinner does not raise a warning when the number of bins
    is not reduced (no duplicate edges).
    """
    
    X = np.arange(100).astype(float)
    binner = QuantileBinner(
        n_bins=10,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        binner.fit(X)
    
    assert binner.n_bins_ == binner.n_bins


def test_QuantileBinner_fit_removes_empty_bins():
    """
    Test that QuantileBinner removes bins that contain no observations. Quantile
    interpolation can place two consecutive edges between the same pair of
    observations, creating a bin no value can fall into.
    """

    # Edges are [0, 1/30, 0.1, 0.5], all distinct, but no value falls in [1/30, 0.1)
    X = np.array([0.0, 0.0, 0.1, 0.1, 0.5])
    binner = QuantileBinner(
        n_bins=3,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )

    warn_msg = re.escape(
        "The number of bins has been reduced from 3 to 2 due to empty bins. This happens when "
        "the values used to compute the edges of the bins are highly "
        "concentrated or contain many repeated values.",
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        binner.fit(X)

    assert binner.n_bins_ == 2
    np.testing.assert_array_almost_equal(
        binner.bin_edges_, np.array([0.0, 1 / 30, 0.5])
    )
    assert sorted(binner.intervals_) == [0, 1]

    # Every remaining bin holds at least one observation
    counts = np.bincount(binner.transform(X).astype(int), minlength=binner.n_bins_)
    np.testing.assert_array_equal(counts, np.array([2, 3]))


def test_QuantileBinner_fit_removes_duplicated_edges_and_empty_bins():
    """
    Test that QuantileBinner reports both causes when the number of bins is
    reduced by duplicated edges and by empty bins at the same time.
    """

    X = np.array([0.0, 0.0, 1.0, 10.0])
    binner = QuantileBinner(
        n_bins=5,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )

    warn_msg = re.escape(
        "The number of bins has been reduced from 5 to 3 due to duplicated edges "
        "and empty bins. This happens when "
        "the values used to compute the edges of the bins are highly "
        "concentrated or contain many repeated values.",
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        binner.fit(X)

    assert binner.n_bins_ == 3
    np.testing.assert_array_almost_equal(
        binner.bin_edges_, np.array([0.0, 0.2, 4.6, 10.0])
    )

    counts = np.bincount(binner.transform(X).astype(int), minlength=binner.n_bins_)
    np.testing.assert_array_equal(counts, np.array([2, 1, 1]))


@pytest.mark.parametrize(
    "X",
    [
        np.array([0.0, 0.0, 0.1, 0.1, 0.5]),
        np.array([0.0, 0.0, 1.0, 10.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
        np.concatenate([np.zeros(50), np.array([1.0, 3.0, 20.0, 100.0])]),
        np.arange(100).astype(float),
    ],
    ids=lambda X: f"X: len={len(X)} n_unique={len(np.unique(X))}",
)
@pytest.mark.parametrize("n_bins", [2, 3, 5, 10, 20], ids=lambda n: f"n_bins: {n}")
def test_QuantileBinner_fit_every_bin_is_not_empty(X, n_bins):
    """
    Test that after fitting, every bin from 0 to n_bins_ - 1 contains at least
    one observation of the data used to fit the binner.
    """

    binner = QuantileBinner(
        n_bins=n_bins,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IgnoredArgumentWarning)
        binner.fit(X)

    counts = np.bincount(binner.transform(X).astype(int), minlength=binner.n_bins_)

    assert binner.n_bins_ >= 1
    assert len(counts) == binner.n_bins_
    assert (counts > 0).all()
    assert sorted(binner.intervals_) == list(range(binner.n_bins_))


def test_QuantileBinner_fit_every_bin_is_not_empty_when_subsampling():
    """
    Test that the non empty bin guarantee also holds when `fit` subsamples the
    input data, as the subsample is a subset of the values of `X`.
    """

    rng = np.random.default_rng(789654)
    X = np.where(rng.random(5_000) < 0.85, 0.0, rng.exponential(20, 5_000))
    binner = QuantileBinner(
        n_bins=10,
        method='linear',
        subsample=500,
        dtype=np.float64,
        random_state=789654,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IgnoredArgumentWarning)
        binner.fit(X)

    counts = np.bincount(binner.transform(X).astype(int), minlength=binner.n_bins_)

    assert (counts > 0).all()


def test_QuantileBinner_fit_every_bin_is_not_empty_with_2d_input():
    """
    Test that the non empty bin guarantee holds when `X` is a 2D array.
    """

    X = np.array([0.0, 0.0, 0.1, 0.1, 0.5]).reshape(-1, 1)
    binner = QuantileBinner(
        n_bins=3,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IgnoredArgumentWarning)
        binner.fit(X)

    assert binner.n_bins_ == 2
    np.testing.assert_array_almost_equal(
        binner.bin_edges_, np.array([0.0, 1 / 30, 0.5])
    )


def test_QuantileBinner_transform_with_reduced_bins():
    """
    Test that transform works correctly after bins have been reduced.
    """
    
    # Data that will result in fewer bins
    X_train = np.array([1, 1, 1, 5, 5, 5, 10, 10, 10])
    X_test = np.array([0, 1, 3, 5, 7, 10, 15])
    
    binner = QuantileBinner(
        n_bins=10,
        method='linear',
        dtype=np.float64,
        random_state=789654,
    )
    
    with pytest.warns(IgnoredArgumentWarning):
        binner.fit(X_train)
    
    # Transform should work and produce valid bin indices
    transformed = binner.transform(X_test)
    
    # All indices should be between 0 and n_bins_ - 1
    assert transformed.min() >= 0
    assert transformed.max() <= binner.n_bins_ - 1
