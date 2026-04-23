# Unit test __repr__, _repr_html_, _truncate_names, _format_names_repr ForecasterFoundation
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation import ForecasterFoundation
from skforecast.foundation._forecaster_foundation import ForecasterFoundation as FF

# Fixtures
from .fixtures_forecaster_foundation import (
    make_forecaster, FakePipeline, y, df_exog, series_dict,
)


# ==============================================================================
# Tests _truncate_names
# ==============================================================================
def test_truncate_names_none_and_empty():
    """
    Test that _truncate_names returns None for None input and an empty
    list for empty input.
    """
    assert FF._truncate_names(None) is None
    assert FF._truncate_names([]) == []


def test_truncate_names_not_truncated_when_within_max_items():
    """
    Test that lists with length <= max_items are returned as a copy
    without truncation.
    """
    names_short = ["a", "b", "c"]
    result_short = FF._truncate_names(names_short, max_items=10)
    assert result_short == ["a", "b", "c"]
    assert result_short is not names_short

    names_exact = [f"s{i}" for i in range(10)]
    result_exact = FF._truncate_names(names_exact, max_items=10)
    assert result_exact == names_exact
    assert "..." not in result_exact


def test_truncate_names_truncated_when_exceeds_max_items():
    """
    Test that a list exceeding max_items is truncated to
    first half + '...' + last half.
    """
    names = [f"s{i}" for i in range(20)]
    result = FF._truncate_names(names, max_items=6)
    assert len(result) == 7  # 3 + '...' + 3
    assert result[:3] == ["s0", "s1", "s2"]
    assert result[3] == "..."
    assert result[4:] == ["s17", "s18", "s19"]


# ==============================================================================
# Tests _format_names_repr
# ==============================================================================
def test_format_names_repr_none_and_empty():
    """
    Test that _format_names_repr returns None for None input and an empty
    string for an empty list.
    """
    assert FF._format_names_repr(None) is None
    assert FF._format_names_repr([]) == ""


def test_format_names_repr_comma_join_and_wrapping():
    """
    Test that a short list is comma-joined without wrapping, and a long
    string exceeding max_text_length is wrapped.
    """
    result_short = FF._format_names_repr(["a", "b", "c"])
    assert result_short == "a, b, c"

    names_long = [f"very_long_feature_name_{i}" for i in range(5)]
    result_long = FF._format_names_repr(names_long, max_text_length=20)
    assert "\n" in result_long


def test_format_names_repr_truncates_long_lists():
    """
    Test that _format_names_repr truncates long lists before formatting.
    """
    names = [f"s{i}" for i in range(100)]
    result = FF._format_names_repr(names, max_items=6)
    assert "..." in result
    assert "s0" in result
    assert "s99" in result
    assert "s50" not in result


# ==============================================================================
# Tests __repr__
# ==============================================================================
def test_repr_before_fit():
    """
    Test that __repr__ before fit contains all expected fields, None
    values for unfitted attributes, and shows forecaster_id.
    """
    from skforecast.foundation._foundation_model import FoundationModel

    forecaster = ForecasterFoundation(
        estimator=FoundationModel(
            "autogluon/chronos-2-small", pipeline=FakePipeline()
        ),
        forecaster_id="custom_id_42",
    )
    result = repr(forecaster)

    for field in [
        "ForecasterFoundation", "Model ID:", "Context length:",
        "Series names:", "Exogenous included:",
        "Context range: None", "Last fit date: None",
        "Skforecast version:", "Python version:",
        "Forecaster id: custom_id_42",
    ]:
        assert field in result


@pytest.mark.parametrize(
    "series_input, exog_input, expected_series, expected_exog_str",
    [
        (y, None, "y", "Exogenous included: False"),
        (y, df_exog, "y", "Exogenous included: True"),
        (series_dict, None, "s1", "Exogenous included: False"),
    ],
    ids=["single_no_exog", "single_with_exog", "multi_series"],
)
def test_repr_after_fit(series_input, exog_input, expected_series, expected_exog_str):
    """
    After fit, repr reflects the fitted state, series names, and exog.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_input, exog=exog_input)
    result = repr(forecaster)

    assert "Context range: None" not in result
    assert "Last fit date: None" not in result
    assert expected_series in result
    assert expected_exog_str in result
    if exog_input is not None:
        assert "feat_a" in result
        assert "feat_b" in result
    if isinstance(series_input, dict):
        assert "s2" in result


def test_repr_many_series_truncated():
    """
    When fitted with more than 50 series, repr truncates with '...'.
    """
    idx = pd.date_range("2020-01-01", periods=50, freq="ME")
    many_series = {
        f"series_{i}": pd.Series(
            np.arange(50, dtype=float), index=idx, name=f"series_{i}"
        )
        for i in range(60)
    }
    forecaster = make_forecaster()
    forecaster.fit(series=many_series)
    result = repr(forecaster)
    assert "..." in result
    assert "series_0" in result
    assert "series_59" in result


# ==============================================================================
# Tests _repr_html_
# ==============================================================================
def test_repr_html_before_fit():
    """
    Test that _repr_html_ before fit returns valid HTML with structure,
    class name, General Information, Model Parameters, Exogenous Variables,
    API Reference, and 'Not fitted' indicator.
    """
    forecaster = make_forecaster(context_length=512)
    result = forecaster._repr_html_()

    for fragment in [
        "<div", "</div>", "<details", "<summary>",
        "ForecasterFoundation", "General Information",
        "Model Parameters", "context_length", "512",
        "Exogenous Variables", "API Reference", "Not fitted",
    ]:
        assert fragment in result


@pytest.mark.parametrize(
    "series_input, exog_input, expected_fragments, unexpected_fragments",
    [
        (
            y, df_exog,
            ["2020", "feat_a", "feat_b"],
            ["Not fitted"],
        ),
        (
            series_dict, None,
            ["s1", "s2", "Context range"],
            ["Not fitted"],
        ),
    ],
    ids=["single_with_exog", "multi_series"],
)
def test_repr_html_after_fit(
    series_input, exog_input, expected_fragments, unexpected_fragments
):
    """
    After fit, _repr_html_ shows context dates, exog names comma-joined,
    and series names. 'Not fitted' no longer appears.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=series_input, exog=exog_input)
    result = forecaster._repr_html_()

    for fragment in expected_fragments:
        assert fragment in result
    for fragment in unexpected_fragments:
        assert fragment not in result


def test_repr_html_truncation_many_series_and_exog():
    """
    When fitted with many series or many exog columns, _repr_html_
    truncates both and includes '...'.
    """
    idx = pd.date_range("2020-01-01", periods=50, freq="ME")

    # Many series
    many_series = {
        f"series_{i}": pd.Series(
            np.arange(50, dtype=float), index=idx, name=f"series_{i}"
        )
        for i in range(60)
    }
    forecaster = make_forecaster()
    forecaster.fit(series=many_series)
    result_series = forecaster._repr_html_()
    assert "..." in result_series
    assert "series_0" in result_series
    assert "series_59" in result_series

    # Many exog
    many_exog = pd.DataFrame(
        {f"feat_{i}": np.arange(50, dtype=float) for i in range(60)},
        index=idx,
    )
    forecaster2 = make_forecaster()
    forecaster2.fit(series=y, exog=many_exog)
    result_exog = forecaster2._repr_html_()
    assert "\u2026" in result_exog
    assert "feat_0" in result_exog
    assert "feat_59" in result_exog


# ==============================================================================
# Tests summary
# ==============================================================================
def test_summary_prints_repr(capsys):
    """
    summary() prints the __repr__ to stdout.
    """
    forecaster = make_forecaster()
    forecaster.summary()
    captured = capsys.readouterr()
    assert "ForecasterFoundation" in captured.out
