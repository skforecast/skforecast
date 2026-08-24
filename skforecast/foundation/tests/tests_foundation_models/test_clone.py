# Unit test clone FoundationModel
# ==============================================================================
import pytest
from sklearn.base import clone
from skforecast.foundation._foundation_model import FoundationModel


# Tests sklearn.base.clone compatibility
# ==============================================================================
@pytest.mark.parametrize(
    "model_id",
    [
        "autogluon/chronos-2-small",
        "google/timesfm-2.5-200m-pytorch",
        "Salesforce/moirai-2.0-R-small",
        "soda-inria/tabicl",
        "priorlabs/tabpfn-ts",
        "theforecastingcompany/t0-alpha",
        "taharnbl/TS-ICL",
        "Synthefy/Nori",
    ],
    ids=["chronos", "timesfm", "moirai", "tabicl", "tabpfn", "t0", "tsicl", "nori"],
)
def test_clone_round_trip_every_adapter_default_params(model_id):
    """
    Test that sklearn.base.clone succeeds for every registered adapter built
    with default parameters, producing an independent, unfitted copy with
    identical parameters.
    """
    model = FoundationModel(model_id)
    cloned = clone(model)

    assert cloned is not model
    assert isinstance(cloned, FoundationModel)
    assert cloned.get_params() == model.get_params()
    assert cloned.is_fitted is False


@pytest.mark.parametrize(
    "model_id, config_kwargs",
    [
        ("autogluon/chronos-2-small",       {"predict_kwargs": {"num_samples": 20}}),
        ("google/timesfm-2.5-200m-pytorch", {"forecast_config_kwargs": {"normalize_inputs": True}}),
        ("soda-inria/tabicl",               {"tabicl_config": {"n_estimators": 4}}),
        ("priorlabs/tabpfn-ts",             {"tabpfn_model_config": {"device": "cpu"}}),
        ("Synthefy/Nori",                   {"nori_config": {"model_path": "dummy"}}),
    ],
    ids=["chronos", "timesfm", "tabicl", "tabpfn", "nori"],
)
def test_clone_round_trip_with_non_empty_config_dict(model_id, config_kwargs):
    """
    Test that sklearn.base.clone produces an independent, unfitted
    FoundationModel with identical parameters when a non-empty config dict is
    passed. Adapters store config dicts by reference (not copied), so clone's
    identity check on the reconstructed parameters succeeds instead of raising
    RuntimeError.
    """
    model = FoundationModel(model_id, **config_kwargs)
    cloned = clone(model)

    key = next(iter(config_kwargs))
    assert cloned is not model
    assert isinstance(cloned, FoundationModel)
    assert cloned.get_params() == model.get_params()
    assert cloned.is_fitted is False
    # clone deep-copies params, so the config of the clone is an independent
    # object with equal content (mutating one must not affect the other).
    assert cloned.get_params()[key] == model.get_params()[key]
    assert cloned.get_params()[key] is not model.get_params()[key]
