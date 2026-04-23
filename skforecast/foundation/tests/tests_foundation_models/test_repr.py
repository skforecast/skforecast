# Unit test __repr__, _repr_html_ FoundationModel
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._foundation_model import FoundationModel

# Fixtures
from .fixtures_adapters import FakePipeline, y, exog


def _make_model(**kwargs):
    """
    Create a FoundationModel with a fake Chronos pipeline injected.
    """
    m = FoundationModel("autogluon/chronos-2-small", **kwargs)
    m.adapter.pipeline = FakePipeline()
    return m


# ==============================================================================
# Tests __repr__
# ==============================================================================
def test_repr_output():
    """
    Test that __repr__ contains all expected fields, including model
    parameters.
    """
    model = _make_model(context_length=512)
    result = repr(model)

    for field in [
        "FoundationModel", "Model ID:", "Context length: 512",
        "Model parameters:", "Last fit date: None",
        "Skforecast version:", "Python version:",
    ]:
        assert field in result


def test_repr_shows_model_parameters():
    """
    Test that __repr__ includes adapter parameters (excluding model_id).
    """
    model = _make_model(context_length=256)
    result = repr(model)
    assert "context_length" in result
    assert "256" in result


# ==============================================================================
# Tests _repr_html_
# ==============================================================================
def test_repr_html_before_fit():
    """
    Test that _repr_html_ before fit returns valid HTML with structure,
    class name, General Information, Model Parameters, API Reference,
    User Guide, and 'Not fitted' indicator.
    """
    model = _make_model(context_length=512)
    result = model._repr_html_()

    for fragment in [
        "<div", "</div>", "<details", "<summary>",
        "FoundationModel", "General Information",
        "Model Parameters", "context_length", "512",
        "API Reference", "User Guide",
    ]:
        assert fragment in result


def test_repr_html_after_fit():
    """
    After fit, _repr_html_ no longer shows 'Not fitted'.
    """
    model = _make_model()
    model.fit(series=y)
    result = model._repr_html_()
    assert "Not fitted" not in result
