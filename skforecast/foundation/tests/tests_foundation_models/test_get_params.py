# Unit test get_params FoundationModel
# ==============================================================================
import pytest
from skforecast.foundation._foundation_model import FoundationModel


# Tests get_params
# ==============================================================================
def test_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns a dict with the canonical parameter names,
    excludes internal attributes like 'pipeline', and reflects the values
    passed at construction time.
    """
    m = FoundationModel(
        "autogluon/chronos-2-small",
        context_length=64,
        cross_learning=True,
    )
    params = m.get_params()

    assert set(params.keys()) == {
        "model",
        "cross_learning",
        "context_length",
        "device_map",
        "torch_dtype",
        "predict_kwargs",
    }
    assert "pipeline" not in params
    assert params["model"] == "autogluon/chronos-2-small"
    assert params["context_length"] == 64
    assert params["cross_learning"] is True
