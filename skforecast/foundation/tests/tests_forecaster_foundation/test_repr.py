# Unit test __repr__ and _repr_html_ ForecasterFoundation
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation import ForecasterFoundation

# Fixtures
from .fixtures_forecaster_foundation import make_forecaster, FakePipeline, y, df_exog


# Tests __repr__
# ==============================================================================

def test_repr_output_before_fit():
    """
    __repr__ returns a non-empty string containing the class name, all
    expected metadata fields, and None values for unfitted attributes.
    """
    forecaster = make_forecaster()
    result = repr(forecaster)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "ForecasterFoundation" in result
    assert "Model:" in result
    assert "Context length:" in result
    assert "Series names:" in result
    assert "Exogenous included:" in result
    assert "Context range: None" in result
    assert "Last fit date: None" in result
    assert "Skforecast version:" in result
    assert "Python version:" in result
    assert "Forecaster id:" in result


@pytest.mark.parametrize(
    "exog_input, exog_present_str",
    [(None, "Exogenous included: False"), (df_exog, "Exogenous included: True")],
    ids=["without_exog", "with_exog"],
)
def test_repr_output_after_fit(exog_input, exog_present_str):
    """
    After fit, repr reflects the fitted state and exog details.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=exog_input)
    result = repr(forecaster)

    assert "Context range: None" not in result
    assert "Last fit date: None" not in result
    assert "Series names: ['y']" in result
    assert exog_present_str in result
    if exog_input is not None:
        assert "feat_a" in result
        assert "feat_b" in result


def test_repr_shows_forecaster_id():
    """
    forecaster_id appears in the repr.
    """
    from skforecast.foundation._foundation_model import FoundationModel

    forecaster = ForecasterFoundation(
        estimator=FoundationModel(
            "autogluon/chronos-2-small", pipeline=FakePipeline()
        ),
        forecaster_id="custom_id_42",
    )
    result = repr(forecaster)
    assert "Forecaster id: custom_id_42" in result


# Tests _repr_html_
# ==============================================================================

def test_repr_html_output_before_fit():
    """
    _repr_html_ returns a non-empty HTML string with basic structure,
    class name, General Information section, API Reference link, and
    'Not fitted' indicator.
    """
    forecaster = make_forecaster()
    result = forecaster._repr_html_()

    assert isinstance(result, str)
    assert len(result) > 0
    assert "<div" in result
    assert "</div>" in result
    assert "<details" in result
    assert "<summary>" in result
    assert "ForecasterFoundation" in result
    assert "General Information" in result
    assert "API Reference" in result
    assert "Not fitted" in result


def test_repr_html_output_after_fit():
    """
    After fit, _repr_html_ shows the training range (year appears).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster._repr_html_()
    assert "2020" in result


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
