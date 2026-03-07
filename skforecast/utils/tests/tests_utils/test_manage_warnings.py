# Unit test manage_warnings
# ==============================================================================
import pytest
import warnings
from skforecast.utils.utils import manage_warnings
from skforecast.exceptions import IgnoredArgumentWarning


@manage_warnings
def _dummy_function_that_warns(suppress_warnings=False):
    """Emit a skforecast warning for testing purposes."""
    warnings.warn("test warning", IgnoredArgumentWarning)
    return "result"


def test_manage_warnings_suppresses_skforecast_warnings():
    """
    When suppress_warnings=True, skforecast warnings should not be raised.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _dummy_function_that_warns(suppress_warnings=True)

    skforecast_warnings = [
        w for w in caught if issubclass(w.category, IgnoredArgumentWarning)
    ]
    assert result == "result"
    assert len(skforecast_warnings) == 0


def test_manage_warnings_does_not_suppress_when_false():
    """
    When suppress_warnings=False, skforecast warnings should be raised normally.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _dummy_function_that_warns(suppress_warnings=False)

    skforecast_warnings = [
        w for w in caught if issubclass(w.category, IgnoredArgumentWarning)
    ]
    assert result == "result"
    assert len(skforecast_warnings) == 1


def test_manage_warnings_restores_state_after_nested_calls():
    """
    An inner function with suppress_warnings=True must not leak its
    suppression into an outer function with suppress_warnings=False.
    The outer function should still see warnings emitted after the
    inner call returns.
    """

    @manage_warnings
    def _inner(suppress_warnings=False):
        warnings.warn("inner warning", IgnoredArgumentWarning)

    @manage_warnings
    def _outer(suppress_warnings=False):
        # Inner call suppresses, but outer does not
        _inner(suppress_warnings=True)
        # This warning must still be visible to the outer caller
        warnings.warn("outer warning", IgnoredArgumentWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _outer(suppress_warnings=False)

    messages = [
        str(w.message) for w in caught
        if issubclass(w.category, IgnoredArgumentWarning)
    ]
    assert not any("inner warning" in m for m in messages)
    assert any("outer warning" in m for m in messages)
