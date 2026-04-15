# Unit test fit FoundationModel
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._foundation_model import FoundationModel
from .fixtures_adapters import y, exog, y_wide, y_dict


# Tests fit — errors
# ==============================================================================
def test_fit_TypeError_when_series_is_invalid_type():
    """
    Test that fit raises TypeError when series is not a pd.Series,
    pd.DataFrame, or dict.
    """
    m = FoundationModel("autogluon/chronos-2-small")
    err_msg = re.escape(
        "`series` must be a pandas DataFrame or a dict of DataFrames or Series. "
        f"Got {type([1, 2, 3])}."
    )
    with pytest.raises(TypeError, match=err_msg):
        m.fit(series=[1, 2, 3])


# Tests fit — single series
# ==============================================================================
def test_fit_output_when_single_series():
    """
    Test fit on a single series: returns self, is_fitted=True,
    adapter.context_ populated, and all metadata attributes are
    correctly stored.
    """
    m = FoundationModel("autogluon/chronos-2-small")
    result = m.fit(series=y)

    assert result is m
    assert m.is_fitted is True
    assert m.is_multiple_series_ is False
    assert m.series_names_in_ == ["sales"]

    # Adapter history
    assert isinstance(m.adapter.context_, dict)
    assert len(next(iter(m.adapter.context_.values()))) == len(y)

    # Index metadata
    assert m.index_type_ is pd.DatetimeIndex
    assert m.index_freq_ == y.index.freq
    assert "sales" in m.context_range_
    pd.testing.assert_index_equal(
        m.context_range_["sales"],
        y.index[[0, -1]],
    )

    # Exog metadata
    assert m.exog_in_ is False
    assert m.exog_names_in_ is None
    assert m.exog_names_in_per_series_ is None

    # Fit date
    assert m.fit_date is not None


def test_fit_output_when_single_series_with_exog():
    """
    Test fit on a single series with exogenous variables: exog_in_,
    exog_names_in_, and exog_names_in_per_series_ are set correctly.
    """
    m = FoundationModel("autogluon/chronos-2-small")
    m.fit(series=y, exog=exog)

    assert m.exog_in_ is True
    assert set(m.exog_names_in_) == {"feat_a", "feat_b"}
    assert m.exog_names_in_per_series_["sales"] == ["feat_a", "feat_b"]


# Tests fit — multi-series
# ==============================================================================
@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"],
)
def test_fit_output_when_multi_series(series_input):
    """
    Test that fit on a wide DataFrame or dict of Series sets
    is_multiple_series_=True, returns self, stores adapter history with
    correct keys, and sets series_names_in_ correctly.
    """
    m = FoundationModel("autogluon/chronos-2-small")
    result = m.fit(series=series_input)

    assert result is m
    assert m.is_fitted is True
    assert m.is_multiple_series_ is True
    assert m.series_names_in_ == ["s1", "s2"]
    assert isinstance(m.adapter.context_, dict)
    assert list(m.adapter.context_.keys()) == ["s1", "s2"]


def test_fit_output_when_single_series_range_index():
    """
    Test fit on a single series with RangeIndex: index_type_ is
    pd.RangeIndex, index_freq_ stores the step, and context_range_ uses
    the RangeIndex values.
    """
    y_range = pd.Series(
        np.arange(30, dtype=float),
        index=pd.RangeIndex(start=0, stop=30, step=1),
        name="y",
    )
    m = FoundationModel("autogluon/chronos-2-small")
    m.fit(series=y_range)

    assert m.index_type_ is pd.RangeIndex
    assert m.index_freq_ == 1
    assert m.is_fitted is True
    pd.testing.assert_index_equal(
        m.context_range_["y"],
        y_range.index[[0, -1]],
    )


# Tests fit — does not modify input
# ==============================================================================
@pytest.mark.parametrize(
    "use_exog",
    [False, True],
    ids=["no_exog", "with_exog"],
)
def test_fit_does_not_modify_input(use_exog):
    """
    Test that fit does not modify the input series or exog.
    """
    y_local = y.copy()
    y_copy = y_local.copy()

    if use_exog:
        exog_local = exog.copy()
        exog_copy = exog_local.copy()
    else:
        exog_local = None
        exog_copy = None

    m = FoundationModel("autogluon/chronos-2-small")
    m.fit(series=y_local, exog=exog_local)

    pd.testing.assert_series_equal(y_local, y_copy)
    if use_exog:
        pd.testing.assert_frame_equal(exog_local, exog_copy)
