# Unit test _prefers_c_contiguous_X
# ==============================================================================
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lars, Lasso, LassoLars, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from skforecast.utils import _prefers_c_contiguous_X
from skforecast.utils.utils import _SKLEARN_C_CONTIGUOUS_ESTIMATORS

# One instance of each estimator in `_SKLEARN_C_CONTIGUOUS_ESTIMATORS`.
C_CONTIGUOUS_ESTIMATORS = [LinearRegression(), Ridge(), Lars(), LassoLars()]


def test_prefers_c_contiguous_X_test_cases_cover_the_list():
    """
    Test that the estimators of this module are the ones of
    `_SKLEARN_C_CONTIGUOUS_ESTIMATORS`, so that an estimator added to the list
    is also added to the tests.
    """
    names = {type(estimator).__name__ for estimator in C_CONTIGUOUS_ESTIMATORS}

    assert names == _SKLEARN_C_CONTIGUOUS_ESTIMATORS


@pytest.mark.parametrize(
    "estimator",
    C_CONTIGUOUS_ESTIMATORS,
    ids=lambda estimator: type(estimator).__name__
)
def test_prefers_c_contiguous_X_true(estimator):
    """
    Test that `_prefers_c_contiguous_X` returns `True` for the estimators in
    `_SKLEARN_C_CONTIGUOUS_ESTIMATORS`.
    """
    assert _prefers_c_contiguous_X(estimator) is True


@pytest.mark.parametrize(
    "estimator",
    [
        HistGradientBoostingRegressor(),
        Lasso(),
        make_pipeline(Ridge()),
    ],
    ids=["HistGradientBoostingRegressor", "Lasso", "Pipeline(Ridge)"]
)
def test_prefers_c_contiguous_X_false(estimator):
    """
    Test that `_prefers_c_contiguous_X` returns `False` for estimators that are
    not in `_SKLEARN_C_CONTIGUOUS_ESTIMATORS` and for a Pipeline, even if its
    only step is in the list (only the class of the estimator is checked).
    """
    assert _prefers_c_contiguous_X(estimator) is False
