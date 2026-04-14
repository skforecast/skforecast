# Unit test set_params FoundationModel
# ==============================================================================
import re
import pytest
from skforecast.foundation._foundation_model import FoundationModel
from .fixtures_adapters import y, exog, FakePipeline


# Tests set_params
# ==============================================================================
def test_set_params_model_id_resets_pipeline():
    """
    Test that set_params(model_id=...) updates model_id on the adapter
    and resets the cached pipeline.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    assert m.adapter._pipeline is not None

    m.set_params(model_id="autogluon/chronos-2-large")

    assert m.adapter.model_id == "autogluon/chronos-2-large"
    assert m.model_id == "autogluon/chronos-2-large"
    assert m.adapter._pipeline is None
    assert m.is_fitted is False


def test_set_params_updates_context_length_and_model_id_attributes():
    """
    Test that set_params keeps context_length and model_id attributes in
    sync with the adapter.
    """
    m = FoundationModel("autogluon/chronos-2-small", context_length=512)
    assert m.context_length == 512
    assert m.model_id == "autogluon/chronos-2-small"

    m.set_params(context_length=256)
    assert m.context_length == 256
    assert m.adapter.context_length == 256

    m.set_params(model_id="autogluon/chronos-2-tiny")
    assert m.model_id == "autogluon/chronos-2-tiny"
    assert m.adapter.model_id == "autogluon/chronos-2-tiny"


def test_set_params_non_pipeline_key_no_reset():
    """
    Test that set_params with a non-pipeline key (cross_learning) does
    not reset the cached pipeline.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.set_params(cross_learning=True)
    assert m.adapter.cross_learning is True
    assert m.adapter._pipeline is not None
    assert m.is_fitted is False


def test_set_params_ValueError_when_invalid_key():
    """
    Test that set_params raises ValueError for invalid keys and the error
    message names 'FoundationModel', not the internal adapter class.
    """
    m = FoundationModel("autogluon/chronos-2-small")
    err_msg = re.escape("Invalid parameter")
    with pytest.raises(ValueError, match=err_msg) as exc_info:
        m.set_params(bad_param=1)
    assert "FoundationModel" in str(exc_info.value)
    assert "Chronos2Adapter" not in str(exc_info.value)


def test_set_params_resets_fitted_state_and_metadata():
    """
    Test that set_params resets is_fitted, fit_date, and all metadata
    attributes to their initial state after a model has been fitted.
    """
    m = FoundationModel("autogluon/chronos-2-small", pipeline=FakePipeline())
    m.fit(series=y, exog=exog)

    assert m.is_fitted is True
    assert m.fit_date is not None
    assert m.series_names_in_ is not None
    assert m.exog_in_ is True

    m.set_params(context_length=256)

    assert m.is_fitted is False
    assert m.fit_date is None
    assert m.index_type_ is None
    assert m.index_freq_ is None
    assert m.context_range_ is None
    assert m.series_names_in_ is None
    assert m.is_multiple_series_ is False
    assert m.exog_in_ is False
    assert m.exog_names_in_ is None
    assert m.exog_names_in_per_series_ is None
    assert m.exog_type_in_ is None
