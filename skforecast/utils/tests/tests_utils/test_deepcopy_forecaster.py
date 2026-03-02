# Unit test deepcopy_forecaster
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from skforecast.utils.utils import deepcopy_forecaster
from skforecast.recursive import ForecasterRecursive
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.recursive import ForecasterStats
from skforecast.direct import ForecasterDirect
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.stats import Arima


# Fixtures
# ==============================================================================
@pytest.fixture(scope="module")
def fitted_forecaster_recursive():
    """
    ForecasterRecursive fitted with LinearRegression and in-sample residuals.
    """
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100), index=pd.date_range("2020", periods=100, freq="D"))
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=5)
    forecaster.fit(y=y, store_in_sample_residuals=True)
    return forecaster, y


@pytest.fixture(scope="module")
def fitted_forecaster_direct():
    """
    ForecasterDirect fitted with LinearRegression and in-sample residuals.
    """
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100), index=pd.date_range("2020", periods=100, freq="D"))
    forecaster = ForecasterDirect(estimator=LinearRegression(), lags=5, steps=3)
    forecaster.fit(y=y, store_in_sample_residuals=True)
    return forecaster, y


@pytest.fixture(scope="module")
def fitted_forecaster_multiseries():
    """
    ForecasterRecursiveMultiSeries fitted with LinearRegression and in-sample residuals.
    """
    rng = np.random.default_rng(123)
    series = pd.DataFrame(
        {"series_1": rng.normal(size=100), "series_2": rng.normal(size=100)},
        index=pd.date_range("2020", periods=100, freq="D"),
    )
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=5
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    return forecaster, series


@pytest.fixture(scope="module")
def fitted_forecaster_direct_multivariate():
    """
    ForecasterDirectMultiVariate fitted with LinearRegression and in-sample residuals.
    """
    rng = np.random.default_rng(123)
    series = pd.DataFrame(
        {"series_1": rng.normal(size=100), "series_2": rng.normal(size=100)},
        index=pd.date_range("2020", periods=100, freq="D"),
    )
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(), lags=5, steps=3, level="series_1"
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    return forecaster, series


@pytest.fixture(scope="module")
def fitted_forecaster_stats():
    """
    ForecasterStats fitted with Arima.
    """
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100), index=pd.date_range("2020", periods=100, freq="D"))
    forecaster = ForecasterStats(estimator=Arima(order=(1, 0, 0)))
    forecaster.fit(y=y)
    return forecaster, y


# Tests
# ==============================================================================
@pytest.mark.parametrize(
    "forecaster_fixture",
    [
        pytest.param("fitted_forecaster_recursive", id="ForecasterRecursive"),
        pytest.param("fitted_forecaster_direct", id="ForecasterDirect"),
        pytest.param("fitted_forecaster_multiseries", id="ForecasterRecursiveMultiSeries"),
        pytest.param("fitted_forecaster_direct_multivariate", id="ForecasterDirectMultiVariate"),
        pytest.param("fitted_forecaster_stats", id="ForecasterStats"),
    ],
)
def test_deepcopy_forecaster_default_returns_correct_type(forecaster_fixture, request):
    """
    Test that deepcopy_forecaster with default parameters returns a forecaster
    of the same type but a different instance.
    """
    forecaster, _ = request.getfixturevalue(forecaster_fixture)
    forecaster_copy = deepcopy_forecaster(forecaster)
    assert type(forecaster_copy) is type(forecaster)
    assert forecaster_copy is not forecaster


@pytest.mark.parametrize(
    "forecaster_fixture",
    [
        pytest.param("fitted_forecaster_recursive", id="ForecasterRecursive"),
        pytest.param("fitted_forecaster_direct", id="ForecasterDirect"),
        pytest.param("fitted_forecaster_multiseries", id="ForecasterRecursiveMultiSeries"),
        pytest.param("fitted_forecaster_direct_multivariate", id="ForecasterDirectMultiVariate"),
    ],
)
def test_deepcopy_forecaster_default_estimator_is_unfitted_clone(
    forecaster_fixture, request
):
    """
    Test that the copy's estimator is an unfitted clone with the same
    hyperparameters as the original. Applies to sklearn-based forecasters.
    Note: For Direct forecasters, the base `estimator` is never fitted
    (only `estimators_` dict values are), so we only verify the clone has
    the same hyperparameters and is unfitted.
    """
    forecaster, _ = request.getfixturevalue(forecaster_fixture)
    forecaster_copy = deepcopy_forecaster(forecaster)

    # Same hyperparameters
    assert forecaster_copy.estimator.get_params() == forecaster.estimator.get_params()
    # Unfitted
    with pytest.raises(NotFittedError):
        check_is_fitted(forecaster_copy.estimator)


def test_deepcopy_forecaster_default_estimators_dict_unfitted(
    fitted_forecaster_direct,
):
    """
    Test that ForecasterDirect copy's estimators_ dict contains unfitted clones
    for each step.
    """
    forecaster, _ = fitted_forecaster_direct
    forecaster_copy = deepcopy_forecaster(forecaster)

    assert isinstance(forecaster_copy.estimators_, dict)
    assert set(forecaster_copy.estimators_.keys()) == set(forecaster.estimators_.keys())
    for step, est in forecaster_copy.estimators_.items():
        with pytest.raises(NotFittedError):
            check_is_fitted(est)
        # Original step estimator is still fitted
        check_is_fitted(forecaster.estimators_[step])


def test_deepcopy_forecaster_default_estimators_list_stats(fitted_forecaster_stats):
    """
    Test that ForecasterStats copy's estimators_ list is a shallow-copied list
    of the unfitted estimators (not the fitted estimators_).
    """
    forecaster, _ = fitted_forecaster_stats
    forecaster_copy = deepcopy_forecaster(forecaster)

    assert isinstance(forecaster_copy.estimators_, list)
    assert len(forecaster_copy.estimators_) == len(forecaster.estimators_)
    # Copy's estimators_ are not the same object as the original's
    for i in range(len(forecaster_copy.estimators_)):
        assert forecaster_copy.estimators_[i] is not forecaster.estimators_[i]


@pytest.mark.parametrize(
    "forecaster_fixture, residual_attrs",
    [
        pytest.param(
            "fitted_forecaster_recursive",
            [
                "in_sample_residuals_",
                "in_sample_residuals_by_bin_",
                "out_sample_residuals_",
                "out_sample_residuals_by_bin_",
            ],
            id="ForecasterRecursive",
        ),
        pytest.param(
            "fitted_forecaster_direct",
            ["in_sample_residuals_"],
            id="ForecasterDirect",
        ),
        pytest.param(
            "fitted_forecaster_multiseries",
            [
                "in_sample_residuals_",
                "in_sample_residuals_by_bin_",
                "out_sample_residuals_",
                "out_sample_residuals_by_bin_",
            ],
            id="ForecasterRecursiveMultiSeries",
        ),
    ],
)
def test_deepcopy_forecaster_default_residuals_are_none(
    forecaster_fixture, residual_attrs, request
):
    """
    Test that with default parameters (include_*=False), all residual
    attributes in the copy are None.
    """
    forecaster, _ = request.getfixturevalue(forecaster_fixture)
    forecaster_copy = deepcopy_forecaster(forecaster)

    for attr in residual_attrs:
        if hasattr(forecaster_copy, attr):
            assert getattr(forecaster_copy, attr) is None, (
                f"Expected {attr} to be None in the copy, "
                f"got {type(getattr(forecaster_copy, attr))}"
            )


@pytest.mark.parametrize(
    "forecaster_fixture",
    [
        pytest.param("fitted_forecaster_recursive", id="ForecasterRecursive"),
        pytest.param("fitted_forecaster_direct", id="ForecasterDirect"),
        pytest.param("fitted_forecaster_multiseries", id="ForecasterRecursiveMultiSeries"),
        pytest.param("fitted_forecaster_stats", id="ForecasterStats"),
    ],
)
def test_deepcopy_forecaster_default_last_window_is_none(
    forecaster_fixture, request
):
    """
    Test that with default parameters the copy's last_window_ is None.
    """
    forecaster, _ = request.getfixturevalue(forecaster_fixture)
    forecaster_copy = deepcopy_forecaster(forecaster)
    assert forecaster_copy.last_window_ is None


@pytest.mark.parametrize(
    "forecaster_fixture",
    [
        pytest.param("fitted_forecaster_recursive", id="ForecasterRecursive"),
        pytest.param("fitted_forecaster_direct", id="ForecasterDirect"),
        pytest.param("fitted_forecaster_multiseries", id="ForecasterRecursiveMultiSeries"),
        pytest.param("fitted_forecaster_stats", id="ForecasterStats"),
    ],
)
def test_deepcopy_forecaster_include_last_window(forecaster_fixture, request):
    """
    Test that include_last_window=True preserves last_window_ in the copy.
    """
    forecaster, _ = request.getfixturevalue(forecaster_fixture)
    forecaster_copy = deepcopy_forecaster(forecaster, include_last_window=True)

    assert forecaster_copy.last_window_ is not None
    if isinstance(forecaster.last_window_, pd.Series):
        pd.testing.assert_series_equal(
            forecaster_copy.last_window_, forecaster.last_window_
        )
    elif isinstance(forecaster.last_window_, pd.DataFrame):
        pd.testing.assert_frame_equal(
            forecaster_copy.last_window_, forecaster.last_window_
        )
    # Verify independence: modifying the copy doesn't affect the original
    original_last_window = forecaster.last_window_.copy()
    forecaster_copy.last_window_ = None
    if isinstance(forecaster.last_window_, pd.Series):
        pd.testing.assert_series_equal(forecaster.last_window_, original_last_window)
    elif isinstance(forecaster.last_window_, pd.DataFrame):
        pd.testing.assert_frame_equal(forecaster.last_window_, original_last_window)


def test_deepcopy_forecaster_include_in_sample_residuals(fitted_forecaster_recursive):
    """
    Test that include_in_sample_residuals=True preserves in_sample_residuals_
    and in_sample_residuals_by_bin_ in the copy, while out_sample remains None.
    """
    forecaster, _ = fitted_forecaster_recursive
    forecaster_copy = deepcopy_forecaster(
        forecaster, include_in_sample_residuals=True
    )

    assert forecaster_copy.in_sample_residuals_ is not None
    assert forecaster_copy.out_sample_residuals_ is None
    assert forecaster_copy.out_sample_residuals_by_bin_ is None

    # Verify it's a deep copy (different object)
    assert forecaster_copy.in_sample_residuals_ is not forecaster.in_sample_residuals_


def test_deepcopy_forecaster_include_out_sample_residuals(fitted_forecaster_recursive):
    """
    Test that include_out_sample_residuals=True preserves out_sample_residuals_
    and out_sample_residuals_by_bin_ when they exist, while in_sample
    remains None.
    """
    forecaster, _ = fitted_forecaster_recursive

    # out_sample_residuals_ is None by default; set a dummy value to test
    forecaster_original_out = forecaster.out_sample_residuals_
    forecaster.out_sample_residuals_ = {"_unknown_level": np.array([0.1, -0.2, 0.3])}
    forecaster.out_sample_residuals_by_bin_ = {"_unknown_level": {1: np.array([0.1])}}

    try:
        forecaster_copy = deepcopy_forecaster(
            forecaster, include_out_sample_residuals=True
        )
        assert forecaster_copy.in_sample_residuals_ is None
        assert forecaster_copy.out_sample_residuals_ is not None
        np.testing.assert_array_equal(
            forecaster_copy.out_sample_residuals_["_unknown_level"],
            np.array([0.1, -0.2, 0.3]),
        )
        # Verify independence
        assert (
            forecaster_copy.out_sample_residuals_
            is not forecaster.out_sample_residuals_
        )
    finally:
        # Restore original values
        forecaster.out_sample_residuals_ = forecaster_original_out
        forecaster.out_sample_residuals_by_bin_ = None


@pytest.mark.parametrize(
    "include_in_sample_residuals, include_out_sample_residuals, include_last_window",
    [
        pytest.param(True, True, True, id="all_True"),
        pytest.param(True, False, True, id="in_sample_and_last_window"),
        pytest.param(False, True, False, id="only_out_sample"),
    ],
)
def test_deepcopy_forecaster_parameter_combinations(
    fitted_forecaster_recursive,
    include_in_sample_residuals,
    include_out_sample_residuals,
    include_last_window,
):
    """
    Test various combinations of include_* parameters produce the expected
    presence/absence of attributes.
    """
    forecaster, _ = fitted_forecaster_recursive
    forecaster_copy = deepcopy_forecaster(
        forecaster,
        include_in_sample_residuals=include_in_sample_residuals,
        include_out_sample_residuals=include_out_sample_residuals,
        include_last_window=include_last_window,
    )

    if include_in_sample_residuals:
        assert forecaster_copy.in_sample_residuals_ is not None
    else:
        assert forecaster_copy.in_sample_residuals_ is None

    if include_last_window:
        assert forecaster_copy.last_window_ is not None
    else:
        assert forecaster_copy.last_window_ is None


@pytest.mark.parametrize(
    "forecaster_fixture",
    [
        pytest.param("fitted_forecaster_recursive", id="ForecasterRecursive"),
        pytest.param("fitted_forecaster_direct", id="ForecasterDirect"),
        pytest.param("fitted_forecaster_multiseries", id="ForecasterRecursiveMultiSeries"),
        pytest.param("fitted_forecaster_direct_multivariate", id="ForecasterDirectMultiVariate"),
        pytest.param("fitted_forecaster_stats", id="ForecasterStats"),
    ],
)
def test_deepcopy_forecaster_original_unchanged(forecaster_fixture, request):
    """
    Test that the original forecaster's heavy attributes are fully restored
    after creating a copy.
    """
    forecaster, _ = request.getfixturevalue(forecaster_fixture)

    # Snapshot original state
    had_estimator = hasattr(forecaster, "estimator")
    if had_estimator:
        original_estimator = forecaster.estimator

    had_estimators_ = (
        hasattr(forecaster, "estimators_") and forecaster.estimators_ is not None
    )
    if had_estimators_:
        original_estimators_ = forecaster.estimators_

    original_last_window = forecaster.last_window_

    # Perform copy
    _ = deepcopy_forecaster(forecaster)

    # Verify restoration
    if had_estimator:
        assert forecaster.estimator is original_estimator

    if had_estimators_:
        assert forecaster.estimators_ is original_estimators_

    assert forecaster.last_window_ is original_last_window
    assert forecaster.last_window_ is not None


def test_deepcopy_forecaster_copy_is_independent(fitted_forecaster_recursive):
    """
    Test that modifications to the copy do not affect the original forecaster
    and vice versa.
    """
    forecaster, _ = fitted_forecaster_recursive
    forecaster_copy = deepcopy_forecaster(
        forecaster, include_last_window=True, include_in_sample_residuals=True
    )

    # Modify the copy
    forecaster_copy.lags = np.array([1, 2, 3])
    forecaster_copy.last_window_ = None

    # Original unchanged
    assert not np.array_equal(forecaster.lags, np.array([1, 2, 3]))
    assert forecaster.last_window_ is not None


def test_deepcopy_forecaster_preserves_metadata(fitted_forecaster_recursive):
    """
    Test that non-heavy scalar/metadata attributes are preserved in the copy.
    """
    forecaster, _ = fitted_forecaster_recursive
    forecaster_copy = deepcopy_forecaster(forecaster)

    assert forecaster_copy.is_fitted == forecaster.is_fitted
    assert forecaster_copy.window_size == forecaster.window_size
    np.testing.assert_array_equal(forecaster_copy.lags, forecaster.lags)
    assert forecaster_copy.index_type_ == forecaster.index_type_
    assert forecaster_copy.index_freq_ == forecaster.index_freq_
    pd.testing.assert_index_equal(forecaster_copy.training_range_, forecaster.training_range_)


def test_deepcopy_forecaster_stats_no_sklearn_residuals(fitted_forecaster_stats):
    """
    Test that ForecasterStats (which has no out_sample_residuals_ or
    in_sample_residuals_by_bin_) is handled correctly without errors.
    """
    forecaster, _ = fitted_forecaster_stats
    forecaster_copy = deepcopy_forecaster(forecaster)

    assert type(forecaster_copy) is ForecasterStats
    assert forecaster_copy.last_window_ is None
    assert isinstance(forecaster_copy.estimators_, list)
    # Verify the original is restored
    assert forecaster.last_window_ is not None
    assert forecaster.estimators_ is not None
