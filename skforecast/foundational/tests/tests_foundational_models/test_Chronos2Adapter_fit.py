# Unit test Chronos2Adapter fit
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational._foundational_models import Chronos2Adapter


# Fixtures
# ==============================================================================
y = pd.Series(
    data=np.arange(50, dtype=float),
    index=pd.date_range("2020-01-01", periods=50, freq="ME"),
    name="y",
)

exog = pd.DataFrame(
    {"feat_a": np.arange(50, dtype=float), "feat_b": np.arange(50, dtype=float) * 2},
    index=y.index,
)


# Tests Chronos2Adapter.fit
# ==============================================================================
def test_Chronos2Adapter_fit_sets_is_fitted():
    """
    Test that fit sets _is_fitted to True.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    assert adapter._is_fitted is False
    adapter.fit(y=y)
    assert adapter._is_fitted is True


def test_Chronos2Adapter_fit_returns_self():
    """
    Test that fit returns the adapter instance itself.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    result = adapter.fit(y=y)
    assert result is adapter


def test_Chronos2Adapter_fit_stores_full_history_when_no_context_length():
    """
    Test that fit stores the entire series when context_length is None.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(y=y)
    pd.testing.assert_series_equal(adapter._history, y)


def test_Chronos2Adapter_fit_stores_exog_when_no_context_length():
    """
    Test that fit stores the entire exog when context_length is None.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(y=y, exog=exog)
    pd.testing.assert_frame_equal(adapter._history_exog, exog)


def test_Chronos2Adapter_fit_stores_none_when_no_exog():
    """
    Test that _history_exog is None when no exog is passed to fit.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    adapter.fit(y=y)
    assert adapter._history_exog is None


def test_Chronos2Adapter_fit_trims_history_to_context_length():
    """
    Test that fit trims history to the last context_length observations.
    """
    context_length = 20
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=context_length)
    adapter.fit(y=y)
    assert len(adapter._history) == context_length
    pd.testing.assert_series_equal(adapter._history, y.iloc[-context_length:])


def test_Chronos2Adapter_fit_trims_exog_to_context_length():
    """
    Test that fit trims exog to the last context_length rows when context_length is set.
    """
    context_length = 15
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=context_length)
    adapter.fit(y=y, exog=exog)
    assert len(adapter._history_exog) == context_length
    pd.testing.assert_frame_equal(adapter._history_exog, exog.iloc[-context_length:])


def test_Chronos2Adapter_fit_exog_none_when_context_length_set_but_no_exog():
    """
    Test that _history_exog remains None when context_length is set but no exog is provided.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=20)
    adapter.fit(y=y)
    assert adapter._history_exog is None


def test_Chronos2Adapter_fit_does_not_modify_input_y():
    """
    Test that fit does not modify the original input series.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=20)
    y_copy = y.copy()
    adapter.fit(y=y)
    pd.testing.assert_series_equal(y, y_copy)


def test_Chronos2Adapter_fit_raises_TypeError_when_y_is_not_series():
    """
    Test that fit raises TypeError when y is not a pandas Series.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    err_msg = re.escape(
        "`y` must be a pandas Series with a DatetimeIndex or a RangeIndex. "
        f"Found {type(np.array([1, 2, 3]))}."
    )
    with pytest.raises(TypeError, match=err_msg):
        adapter.fit(y=np.array([1, 2, 3]))


def test_Chronos2Adapter_fit_raises_ValueError_when_y_has_nan():
    """
    Test that fit raises ValueError when y contains NaN values.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    y_nan = y.copy()
    y_nan.iloc[5] = np.nan
    err_msg = re.escape("`y` has missing values.")
    with pytest.raises(ValueError, match=err_msg):
        adapter.fit(y=y_nan)


@pytest.mark.parametrize(
    "context_length, expected_len",
    [(10, 10), (25, 25), (50, 50), (100, 50)],
    ids=lambda x: f"context_length, expected_len: {x}",
)
def test_Chronos2Adapter_fit_context_length_parametrize(context_length, expected_len):
    """
    Test that fit trims history to min(context_length, len(y)).
    When context_length >= len(y), the full series is stored.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", context_length=context_length)
    adapter.fit(y=y)
    assert len(adapter._history) == expected_len
