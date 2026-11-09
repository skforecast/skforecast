# Unit test _revert_differentiation_one_step_ahead
# ==============================================================================
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from skforecast.model_selection._utils import _revert_differentiation_one_step_ahead


def test_revert_differentiation_one_step_ahead_order_1_adds_predicted_difference_to_previous_observed_value():
    """
    Test that, with differentiation of order 1 and no transformer, each
    prediction is the previous observed value plus the predicted difference,
    so the error of each prediction is the error of its difference and is not
    carried into the following ones.
    """
    y = np.array([100., 102., 101., 103., 104.])
    diff_true = np.diff(y)
    diff_pred = np.array([1., -2., 1., 0.])

    y_true, y_pred = _revert_differentiation_one_step_ahead(
        y_observed=y[1:], diff_true=diff_true, diff_pred=diff_pred, transformer=None
    )

    np.testing.assert_array_almost_equal(y_true, y[1:])
    np.testing.assert_array_almost_equal(y_pred, np.array([101., 100., 102., 103.]))
    np.testing.assert_array_almost_equal(y_true - y_pred, diff_true - diff_pred)


def test_revert_differentiation_one_step_ahead_order_2():
    """
    Test that, with differentiation of order 2, each prediction is built from
    the two previous observed values: `2 * y[t-1] - y[t-2] + diff_pred[t]`.
    """
    rng = np.random.default_rng(123)
    y = rng.normal(size=30).cumsum().cumsum()
    diff_true = np.diff(y, n=2)
    diff_pred = diff_true + rng.normal(size=len(diff_true))

    y_true, y_pred = _revert_differentiation_one_step_ahead(
        y_observed=y[2:], diff_true=diff_true, diff_pred=diff_pred, transformer=None
    )

    expected = 2 * y[1:-1] - y[:-2] + diff_pred
    np.testing.assert_array_almost_equal(y_true, y[2:])
    np.testing.assert_array_almost_equal(y_pred, expected)


@pytest.mark.parametrize("order", [1, 2], ids=lambda order: f"order: {order}")
def test_revert_differentiation_one_step_ahead_with_transformer(order):
    """
    Test that, with a transformer, the predicted difference is added in the
    transformed scale and the result is reverted to the original scale.
    """
    rng = np.random.default_rng(456)
    y = 50 + rng.normal(size=40).cumsum()
    transformer = StandardScaler().fit(y[:20].reshape(-1, 1))
    y_transformed = transformer.transform(y.reshape(-1, 1)).ravel()
    diff_true = np.diff(y_transformed, n=order)
    diff_pred = diff_true + rng.normal(scale=0.1, size=len(diff_true))

    y_true, y_pred = _revert_differentiation_one_step_ahead(
        y_observed=y[order:], diff_true=diff_true, diff_pred=diff_pred,
        transformer=transformer
    )

    expected = transformer.inverse_transform(
        (y_transformed[order:] - diff_true + diff_pred).reshape(-1, 1)
    ).ravel()
    np.testing.assert_array_almost_equal(y_true, y[order:])
    np.testing.assert_array_almost_equal(y_pred, expected)
