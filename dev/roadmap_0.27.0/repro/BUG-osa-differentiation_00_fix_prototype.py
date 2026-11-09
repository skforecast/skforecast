"""
BUG-osa-differentiation: prototype of the fix, without changing the package.

The one-step-ahead metrics revert the differentiation of `y_true` and `y_pred`
with `TimeSeriesDifferentiator.inverse_transform_next_window`, which adds the
cumulative sum of the differences to the last training values. For `y_true`
this gives the observed values (the cumulative sum of the observed differences),
but for `y_pred` the error of each prediction is carried into all the following
ones. The one-step-ahead prediction is the observed value it would be built
from plus the predicted difference, which for any order `d` is

    level_pred_t = level_true_t - diff_true_t + diff_pred_t

(the d-th difference of the observed values at t, `diff_true_t`, and the
predicted one, `diff_pred_t`, are added to the same combination of the previous
observed values). The error in the original (transformed) scale is the error of
the difference.

`enable()` replaces `inverse_transform_next_window` with a version that, only
when it is called from the two one-step-ahead functions, pairs its two
consecutive calls (first `y_true`, then `y_pred`, as both functions do for each
level) and returns `level_true - diff_true + diff_pred` for the second one.
Every other caller (`predict`, backtesting) gets the original method.
"""
import sys
import numpy as np
from skforecast.preprocessing import TimeSeriesDifferentiator

_ORIGINAL = TimeSeriesDifferentiator.inverse_transform_next_window
_OSA_FUNCTIONS = {
    "_calculate_metrics_one_step_ahead",
    "_predict_and_calculate_metrics_one_step_ahead_multiseries",
}
_pending = {}


def _patched(self, X, y=None):
    levels = _ORIGINAL(self, X)
    if sys._getframe(1).f_code.co_name not in _OSA_FUNCTIONS:
        return levels
    key = id(self)
    if key not in _pending:  # first call: y_true
        _pending[key] = (np.asarray(X, dtype=float).copy(), levels.copy())
        return levels
    diff_true, level_true = _pending.pop(key)  # second call: y_pred
    diff_pred = np.asarray(X, dtype=float).reshape(level_true.shape)
    return level_true - diff_true.reshape(level_true.shape) + diff_pred


def enable():
    TimeSeriesDifferentiator.inverse_transform_next_window = _patched


def disable():
    TimeSeriesDifferentiator.inverse_transform_next_window = _ORIGINAL
    assert not _pending, "Unpaired call of inverse_transform_next_window"


class enabled:
    def __enter__(self):
        enable()

    def __exit__(self, *exc):
        disable()


# pytest plugin: `pytest -p <this module>` runs the tests with the fix.
def pytest_configure(config):
    enable()
