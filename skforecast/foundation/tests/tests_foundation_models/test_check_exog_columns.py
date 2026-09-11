# Unit test _check_exog_columns FoundationModel
# ==============================================================================
import re
import warnings
import pytest
import pandas as pd
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.foundation._foundation_model import FoundationModel
from skforecast.foundation._adapters import (
    ChronosAdapter,
    TimesFM25Adapter,
    TimesFM3Adapter,
    MoiraiAdapter,
    TabICLAdapter,
    TabPFNAdapter,
    T0Adapter,
    TSICLAdapter,
    NoriAdapter,
)
from .fixtures_adapters import y, y_wide, exog_shared, FakePipeline


# Fixtures
# ==============================================================================
# The check only inspects covariate column names, never the values or index,
# so tiny frames are enough. `model_id` selects the adapter; no backend is
# loaded (adapters load lazily).
_EXOG_ADAPTERS = [
    "autogluon/chronos-2-small",
    "soda-inria/tabicl",
    "priorlabs/tabpfn-ts",
    "theforecastingcompany/t0-alpha",
    "Synthefy/Nori",
    "taharnbl/TS-ICL",
    "google/timesfm-3.0-pytorch",
]
_PAST_ONLY_ADAPTERS = [
    "autogluon/chronos-2-small",
    "taharnbl/TS-ICL",
    "google/timesfm-3.0-pytorch",
]
_FUTURE_ONLY_ADAPTERS = [
    "soda-inria/tabicl",
    "priorlabs/tabpfn-ts",
    "theforecastingcompany/t0-alpha",
    "Synthefy/Nori",
]

# s1 has 'B' in context but not in future (past-only column); s2 has 'Z' in
# future but not in context (no history).
_ce_div = {
    "s1": pd.DataFrame({"A": [0.0, 1.0], "B": [0.0, 1.0]}),
    "s2": pd.DataFrame({"A": [0.0, 1.0]}),
}
_ex_div = {
    "s1": pd.DataFrame({"A": [9.0]}),
    "s2": pd.DataFrame({"A": [9.0], "Z": [0.0]}),
}
_ce_past_only = {"s1": pd.DataFrame({"A": [0.0], "C": [0.0], "B": [0.0]})}
_ex_past_only = {"s1": pd.DataFrame({"A": [9.0]})}


def _make_model(model_id="autogluon/chronos-2-small"):
    return FoundationModel(model_id)


# Tests adapters declare supports_past_only_covariates
# ==============================================================================
@pytest.mark.parametrize(
    "adapter_cls, expected",
    [
        (ChronosAdapter, True),
        (TSICLAdapter, True),
        (TabICLAdapter, False),
        (TabPFNAdapter, False),
        (T0Adapter, False),
        (NoriAdapter, False),
        (MoiraiAdapter, False),
        (TimesFM25Adapter, False),
        (TimesFM3Adapter, True),
    ],
    ids=lambda x: f"{getattr(x, '__name__', x)}",
)
def test_adapters_declare_supports_past_only_covariates(adapter_cls, expected):
    """
    Test that every adapter class declares supports_past_only_covariates:
    True for adapters that use historical columns without future values as
    past-only covariates, False for those that drop them.
    """
    assert adapter_cls.supports_past_only_covariates is expected


def test_FoundationModel_supports_past_only_covariates_mirrors_adapter():
    """
    Test that FoundationModel.supports_past_only_covariates delegates to the
    adapter.
    """
    assert _make_model("autogluon/chronos-2-small").supports_past_only_covariates is True
    assert _make_model("soda-inria/tabicl").supports_past_only_covariates is False
    assert _make_model("google/timesfm-3.0-pytorch").supports_past_only_covariates is True
    assert _make_model("google/timesfm-2.5-200m-pytorch").supports_past_only_covariates is False


# Tests _check_exog_columns ValueError (future column without history)
# ==============================================================================
@pytest.mark.parametrize("model_id", _EXOG_ADAPTERS, ids=lambda m: f"model_id: {m}")
def test_check_exog_columns_ValueError_when_future_column_has_no_history(model_id):
    """
    Test that a future exog column with no historical values in the same
    series raises ValueError for every exog-supporting adapter, naming the
    series and columns.
    """
    m = _make_model(model_id)
    err_msg = re.escape(
        "`exog` contains columns with no historical values in the context "
        "for series {'s2': ['Z']}."
    )
    with pytest.raises(ValueError, match=err_msg):
        m._check_exog_columns(
            context_exog=_ce_div, exog=_ex_div, series_names_in=["s1", "s2"]
        )


@pytest.mark.parametrize(
    "context_exog",
    [None, {"s1": None}],
    ids=["context_exog_None", "context_exog_dict_of_None"],
)
def test_check_exog_columns_ValueError_when_no_context_exog(context_exog):
    """
    Test that future exog with no historical exog at all (None or an all-None
    dict) raises ValueError, since every future column lacks history.
    """
    m = _make_model()
    err_msg = re.escape("for series {'s1': ['feat']}")
    with pytest.raises(ValueError, match=err_msg):
        m._check_exog_columns(
            context_exog=context_exog,
            exog={"s1": pd.DataFrame({"feat": [9.0]})},
            series_names_in=["s1"],
        )


# Tests _check_exog_columns IgnoredArgumentWarning (historical column without future)
# ==============================================================================
@pytest.mark.parametrize("model_id", _FUTURE_ONLY_ADAPTERS, ids=lambda m: f"model_id: {m}")
def test_check_exog_columns_IgnoredArgumentWarning_when_adapter_drops_past_only(model_id):
    """
    Test that adapters with supports_past_only_covariates=False issue an
    IgnoredArgumentWarning listing the historical columns without future
    values per series, sorted by name.
    """
    m = _make_model(model_id)
    warn_msg = re.escape(
        f"{type(m.adapter).__name__} only uses covariates that also have "
        f"future values. Historical exog columns without future values are "
        f"ignored for series {{'s1': ['B', 'C']}}."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        m._check_exog_columns(
            context_exog=_ce_past_only, exog=_ex_past_only, series_names_in=["s1"]
        )


@pytest.mark.parametrize("model_id", _FUTURE_ONLY_ADAPTERS, ids=lambda m: f"model_id: {m}")
def test_check_exog_columns_IgnoredArgumentWarning_when_future_exog_none(model_id):
    """
    Test that, for adapters that drop past-only covariates, an all-None future
    exog (predict without `exog` after fit with exog) warns that every
    historical column is ignored.
    """
    m = _make_model(model_id)
    warn_msg = re.escape("ignored for series {'s1': ['feat']}")
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        m._check_exog_columns(
            context_exog={"s1": pd.DataFrame({"feat": [0.0]})},
            exog={"s1": None},
            series_names_in=["s1"],
        )


@pytest.mark.parametrize("model_id", _PAST_ONLY_ADAPTERS, ids=lambda m: f"model_id: {m}")
@pytest.mark.parametrize(
    "exog",
    [_ex_past_only, {"s1": None}, None],
    ids=["subset_of_columns", "dict_of_None", "None"],
)
def test_check_exog_columns_no_warning_for_past_only_adapters(model_id, exog):
    """
    Test that adapters with supports_past_only_covariates=True do not warn
    when historical columns have no future values: the column is a
    legitimate past-only covariate.
    """
    m = _make_model(model_id)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m._check_exog_columns(
            context_exog=_ce_past_only, exog=exog, series_names_in=["s1"]
        )
    assert len(record) == 0


# Tests _check_exog_columns no error nor warning
# ==============================================================================
@pytest.mark.parametrize(
    "context_exog, exog",
    [
        (
            {"s1": pd.DataFrame({"A": [0.0], "B": [0.0]}), "s2": pd.DataFrame({"A": [0.0]})},
            {"s1": pd.DataFrame({"B": [9.0], "A": [9.0]}), "s2": pd.DataFrame({"A": [9.0]})},
        ),
        ({"s1": pd.Series([0.0], name="feat"), "s2": None}, {"s1": pd.Series([9.0], name="feat"), "s2": None}),
        (None, {"s1": None, "s2": None}),
        ({"s1": None, "s2": None}, None),
        (None, None),
    ],
    ids=["matching_columns_any_order", "series_blocks_by_name", "both_none_dicts", "context_none_dict", "both_none"],
)
@pytest.mark.parametrize("model_id", ["soda-inria/tabicl", "autogluon/chronos-2-small"], ids=lambda m: f"model_id: {m}")
def test_check_exog_columns_passes_when_columns_match(context_exog, exog, model_id):
    """
    Test that no error or warning is raised when the columns of the future
    exog match those of the historical exog per series (order-insensitive,
    pandas Series compared by name), or when there are no covariates at all.
    """
    m = _make_model(model_id)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m._check_exog_columns(
            context_exog=context_exog, exog=exog, series_names_in=["s1", "s2"]
        )
    assert len(record) == 0


# Tests _check_exog_columns through FoundationModel.predict
# ==============================================================================
def test_predict_ValueError_zero_shot_when_exog_without_context_exog():
    """
    Test that in zero-shot mode (context given, no fit) passing `exog`
    without `context_exog` raises ValueError from the column check, before
    reaching the adapter.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    future = pd.DataFrame(
        {"feat": [1.0, 2.0, 3.0]},
        index=pd.date_range(y.index[-1] + y.index.freq, periods=3, freq=y.index.freq),
    )
    err_msg = re.escape("for series {'sales': ['feat']}")
    with pytest.raises(ValueError, match=err_msg):
        m.predict(steps=3, context=y, exog=future)


def test_predict_check_exog_columns_only_for_requested_levels():
    """
    Test that the column check runs only on the series selected with
    `levels`: a series whose future exog column has no history does not raise
    when it is not among the requested levels.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y_wide, exog={"s1": exog_shared})
    future_idx = pd.date_range(
        y_wide.index[-1] + y_wide.index.freq, periods=3, freq=y_wide.index.freq
    )
    future = {
        "s1": pd.DataFrame({"feat": [1.0, 2.0, 3.0]}, index=future_idx),
        "s2": pd.DataFrame({"feat": [1.0, 2.0, 3.0]}, index=future_idx),
    }

    with pytest.raises(ValueError, match=re.escape("{'s2': ['feat']}")):
        m.predict(steps=3, exog=future)

    predictions = m.predict(steps=3, levels=["s1"], exog=future)
    assert list(predictions["level"].unique()) == ["s1"]
