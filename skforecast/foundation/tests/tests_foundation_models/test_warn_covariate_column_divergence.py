# Unit test _warn_covariate_column_divergence FoundationModel
# ==============================================================================
import re
import warnings
import pytest
import numpy as np
import pandas as pd
from skforecast.exceptions import MissingExogWarning
from skforecast.foundation._foundation_model import FoundationModel


# -- Fixtures ------------------------------------------------------------------
# The helper only inspects covariate column names, never the values or index,
# so tiny frames are enough. `model_id` selects the adapter whose class name is
# reported in the message; no backend is loaded (adapters load lazily).
_EXOG_ADAPTERS = [
    "autogluon/chronos-2-small",     # ChronosAdapter (independent past/future)
    "soda-inria/tabicl",             # TabICLAdapter
    "priorlabs/tabpfn-ts",           # TabPFNAdapter
    "theforecastingcompany/t0-alpha",  # T0Adapter
    "Synthefy/Nori",                 # NoriAdapter
    "taharnbl/TS-ICL",               # TSICLAdapter
    "google/timesfm-3.0-pytorch",    # TimesFMAdapter (v3, allow_exog)
]

# Divergent: s1 has 'B' in context but not in future; s2 has 'Z' in future but
# not in context. missing_future = {'s1': ['B']}, no_history = {'s2': ['Z']}.
_ce_div = {
    "s1": pd.DataFrame({"A": [0.0, 1.0], "B": [0.0, 1.0]}),
    "s2": pd.DataFrame({"A": [0.0, 1.0]}),
}
_ex_div = {
    "s1": pd.DataFrame({"A": [9.0]}),
    "s2": pd.DataFrame({"A": [9.0], "Z": [0.0]}),
}
_names_div = ["s1", "s2"]


def _make_model(model_id="autogluon/chronos-2-small"):
    return FoundationModel(model_id)


# Tests _warn_covariate_column_divergence — fires
# ==============================================================================
@pytest.mark.parametrize(
    "model_id", _EXOG_ADAPTERS, ids=lambda m: f"model_id: {m}"
)
def test_warn_covariate_column_divergence_fires_for_all_exog_adapters(model_id):
    """
    Test the divergence warning fires for every exog-supporting adapter,
    including the independent past/future family (Chronos, TS-ICL) where a
    one-sided column is legitimate. The reported column dicts are
    adapter-independent.
    """
    m = _make_model(model_id)

    warn_msg = re.escape(
        "In context but missing from future `exog`: {'s1': ['B']}"
    )
    with pytest.warns(MissingExogWarning, match=warn_msg):
        m._warn_covariate_column_divergence(
            context_exog=_ce_div, exog=_ex_div, series_names_in=_names_div,
        )


def test_warn_covariate_column_divergence_reports_per_series_columns():
    """
    Test the message lists both divergence directions per series (missing from
    future vs. no history) and excludes series whose columns match.
    """
    m = _make_model()
    context_exog = {
        "s1": pd.DataFrame({"A": [0.0], "B": [0.0]}),
        "s2": pd.DataFrame({"A": [0.0]}),
        "s3": pd.DataFrame({"A": [0.0]}),
    }
    exog = {
        "s1": pd.DataFrame({"A": [9.0]}),
        "s2": pd.DataFrame({"A": [9.0], "Z": [0.0]}),
        "s3": pd.DataFrame({"A": [9.0]}),
    }

    with pytest.warns(MissingExogWarning) as record:
        m._warn_covariate_column_divergence(
            context_exog=context_exog, exog=exog,
            series_names_in=["s1", "s2", "s3"],
        )

    msg = str(record[0].message)
    assert "In context but missing from future `exog`: {'s1': ['B']}" in msg
    assert "In future `exog` but absent from context: {'s2': ['Z']}" in msg
    assert "s3" not in msg


def test_warn_covariate_column_divergence_reports_multiple_columns_sorted():
    """
    Test that several divergent columns for one series are reported sorted.
    """
    m = _make_model()
    context_exog = {"s1": pd.DataFrame({"A": [0.0], "C": [0.0], "B": [0.0]})}
    exog = {"s1": pd.DataFrame({"A": [9.0]})}

    warn_msg = re.escape(
        "In context but missing from future `exog`: {'s1': ['B', 'C']}"
    )
    with pytest.warns(MissingExogWarning, match=warn_msg):
        m._warn_covariate_column_divergence(
            context_exog=context_exog, exog=exog, series_names_in=["s1"],
        )


# Tests _warn_covariate_column_divergence — one-sided (None) inputs fire
# ==============================================================================
def test_warn_covariate_column_divergence_fires_when_no_context_exog():
    """
    Test a future-only covariate (no context_exog at all) is reported as
    having no history.
    """
    m = _make_model()
    exog = {"s1": pd.DataFrame({"feat": [9.0]})}

    warn_msg = re.escape(
        "In future `exog` but absent from context: {'s1': ['feat']}"
    )
    with pytest.warns(MissingExogWarning, match=warn_msg):
        m._warn_covariate_column_divergence(
            context_exog=None, exog=exog, series_names_in=["s1"],
        )


def test_warn_covariate_column_divergence_fires_when_future_exog_none():
    """
    Test a context covariate absent from an all-None future exog is reported as
    missing from the future.
    """
    m = _make_model()
    context_exog = {"s1": pd.DataFrame({"feat": [0.0]})}

    warn_msg = re.escape(
        "In context but missing from future `exog`: {'s1': ['feat']}"
    )
    with pytest.warns(MissingExogWarning, match=warn_msg):
        m._warn_covariate_column_divergence(
            context_exog=context_exog, exog={"s1": None},
            series_names_in=["s1"],
        )


# Tests _warn_covariate_column_divergence — does not fire
# ==============================================================================
@pytest.mark.parametrize(
    "context_exog, exog",
    [
        (
            {"s1": pd.DataFrame({"A": [0.0]}), "s2": pd.DataFrame({"A": [0.0]})},
            {"s1": pd.DataFrame({"A": [9.0]}), "s2": pd.DataFrame({"A": [9.0]})},
        ),
        (None, {"s1": None, "s2": None}),
        ({"s1": None, "s2": None}, None),
        (None, None),
    ],
    ids=["uniform_columns", "both_none_dicts", "context_none_dict", "both_none"],
)
def test_warn_covariate_column_divergence_does_not_fire(context_exog, exog):
    """
    Test no warning is emitted when columns match on both sides or when there
    are no covariates at all.
    """
    m = _make_model()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m._warn_covariate_column_divergence(
            context_exog=context_exog, exog=exog,
            series_names_in=["s1", "s2"],
        )

    assert not any(
        isinstance(w.message, MissingExogWarning) for w in record
    )


# Tests _warn_covariate_column_divergence — Series-valued blocks
# ==============================================================================
def test_warn_covariate_column_divergence_uses_series_name_for_blocks():
    """
    Test that pandas Series covariate blocks are compared by their `name`:
    matching names do not warn, differing names do.
    """
    m = _make_model()
    ctx_series = pd.Series([0.0, 1.0], name="feat")

    # Matching names -> no warning.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m._warn_covariate_column_divergence(
            context_exog={"s1": ctx_series},
            exog={"s1": pd.Series([9.0], name="feat")},
            series_names_in=["s1"],
        )
    assert not any(
        isinstance(w.message, MissingExogWarning) for w in record
    )

    # Differing names -> warning reporting both sides.
    with pytest.warns(MissingExogWarning) as record:
        m._warn_covariate_column_divergence(
            context_exog={"s1": ctx_series},
            exog={"s1": pd.Series([9.0], name="other")},
            series_names_in=["s1"],
        )
    msg = str(record[0].message)
    assert "In context but missing from future `exog`: {'s1': ['feat']}" in msg
    assert "In future `exog` but absent from context: {'s1': ['other']}" in msg
