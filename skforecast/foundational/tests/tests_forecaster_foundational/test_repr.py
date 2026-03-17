# Unit test __repr__ and _repr_html_ ForecasterFoundational
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational import ForecasterFoundational

# Fixtures
from .fixtures_forecaster_foundational import make_forecaster, y, df_exog


# Tests __repr__ and _repr_html_
# ==============================================================================

def test_repr_returns_string():
    """
    __repr__ returns a non-empty string.
    """
    forecaster = make_forecaster()
    result = repr(forecaster)
    assert isinstance(result, str)
    assert len(result) > 0


def test_repr_contains_class_name():
    """
    __repr__ contains the class name 'ForecasterFoundational'.
    """
    forecaster = make_forecaster()
    result = repr(forecaster)
    assert "ForecasterFoundational" in result


def test_repr_contains_expected_fields():
    """
    __repr__ contains all expected metadata fields.
    """
    forecaster = make_forecaster()
    result = repr(forecaster)

    assert "Model:" in result
    assert "Context length:" in result
    assert "Series name:" in result
    assert "Exogenous included:" in result
    assert "Training range:" in result
    assert "Last fit date:" in result
    assert "Skforecast version:" in result
    assert "Python version:" in result
    assert "Forecaster id:" in result


def test_repr_training_range_None_before_fit():
    """
    Before fit, training range is None in the repr.
    """
    forecaster = make_forecaster()
    result = repr(forecaster)
    assert "Training range: None" in result


def test_repr_last_fit_date_None_before_fit():
    """
    Before fit, last fit date is None in the repr.
    """
    forecaster = make_forecaster()
    result = repr(forecaster)
    assert "Last fit date: None" in result


def test_repr_after_fit_without_exog():
    """
    After fit (no exog), repr reflects the fitted state.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = repr(forecaster)

    assert "Training range: None" not in result
    assert "Last fit date: None" not in result
    assert "Series name: y" in result
    assert "Exogenous included: False" in result


def test_repr_after_fit_with_exog():
    """
    After fit with exog, repr shows exog details.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y, exog=df_exog)
    result = repr(forecaster)

    assert "Exogenous included: True" in result
    assert "feat_a" in result
    assert "feat_b" in result


def test_repr_shows_forecaster_id():
    """
    forecaster_id appears in the repr.
    """
    from skforecast.foundational._foundational_model import FoundationalModel
    from .fixtures_forecaster_foundational import FakePipeline

    forecaster = ForecasterFoundational(
        estimator=FoundationalModel("autogluon/chronos-2-small", pipeline=FakePipeline()),
        forecaster_id="custom_id_42",
    )
    result = repr(forecaster)
    assert "Forecaster id: custom_id_42" in result


# Tests _repr_html_
# ==============================================================================

def test_repr_html_returns_string():
    """
    _repr_html_ returns a non-empty string.
    """
    forecaster = make_forecaster()
    result = forecaster._repr_html_()
    assert isinstance(result, str)
    assert len(result) > 0


def test_repr_html_contains_html_structure():
    """
    _repr_html_ output contains basic HTML structure.
    """
    forecaster = make_forecaster()
    result = forecaster._repr_html_()
    assert "<div" in result
    assert "</div>" in result
    assert "<details" in result
    assert "<summary>" in result


def test_repr_html_contains_class_name():
    """
    _repr_html_ output contains the class name.
    """
    forecaster = make_forecaster()
    result = forecaster._repr_html_()
    assert "ForecasterFoundational" in result


def test_repr_html_contains_general_information_section():
    """
    _repr_html_ has an open 'General Information' details section.
    """
    forecaster = make_forecaster()
    result = forecaster._repr_html_()
    assert "General Information" in result


def test_repr_html_contains_api_reference_link():
    """
    _repr_html_ contains an API Reference link.
    """
    forecaster = make_forecaster()
    result = forecaster._repr_html_()
    assert "API Reference" in result


def test_repr_html_not_fitted_shows_not_fitted():
    """
    Before fit, _repr_html_ shows 'Not fitted' in training information.
    """
    forecaster = make_forecaster()
    result = forecaster._repr_html_()
    assert "Not fitted" in result


def test_repr_html_after_fit_shows_training_range():
    """
    After fit, _repr_html_ shows the training range (year appears).
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    result = forecaster._repr_html_()
    # y starts from 2020; the training range should reference that year.
    assert "2020" in result


def test_summary_prints_repr(capsys):
    """
    summary() prints the __repr__ to stdout.
    """
    forecaster = make_forecaster()
    forecaster.summary()
    captured = capsys.readouterr()
    assert "ForecasterFoundational" in captured.out
