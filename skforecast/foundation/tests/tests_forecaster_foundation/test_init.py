# Unit test __init__ ForecasterFoundation
# ==============================================================================
import re
import sys
import pytest
from sklearn.linear_model import LinearRegression
from skforecast.foundation import ForecasterFoundation, FoundationModel
from skforecast.foundation._adapters import ChronosAdapter

# Fixtures
from .fixtures_forecaster_foundation import make_forecaster, FakePipeline, y


# Tests __init__
# ==============================================================================

@pytest.mark.parametrize(
    "estimator",
    [LinearRegression(), ChronosAdapter(model_id="autogluon/chronos-2-small")],
    ids=lambda e: f"estimator: {type(e).__name__}",
)
def test_init_TypeError_when_estimator_is_not_FoundationModel(estimator):
    """
    Raise TypeError if `estimator` is not a FoundationModel instance.
    """
    err_msg = re.escape(
        f"`estimator` must be a `FoundationModel` instance. "
        f"Got {type(estimator)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        ForecasterFoundation(estimator=estimator)


def test_init_estimator_is_cloned_and_derived_attributes_correctly_stored():
    """
    estimator is cloned (not aliased) so that two forecasters wrapping the
    same FoundationModel do not share fit-derived state, and context_length,
    model_id, and window_size are correctly derived from the clone.
    """
    estimator = FoundationModel(
        "autogluon/chronos-2-small", context_length=128, pipeline=FakePipeline()
    )
    forecaster = ForecasterFoundation(estimator=estimator)

    assert forecaster.estimator is not estimator
    assert isinstance(forecaster.estimator, FoundationModel)
    assert forecaster.context_length == 128
    assert forecaster.model_id == "autogluon/chronos-2-small"
    assert forecaster.model_id == estimator.model_id
    assert forecaster.window_size == 128


@pytest.mark.parametrize(
    "model_id, config_kwargs",
    [
        ("google/timesfm-2.5-200m-pytorch", {"forecast_config_kwargs": {"normalize_inputs": True}}),
        ("soda-inria/tabicl",               {"tabicl_config": {"n_estimators": 4}}),
        ("priorlabs/tabpfn-ts",             {"tabpfn_model_config": {"device": "cpu"}}),
        ("Synthefy/Nori",                   {"nori_config": {"model_path": "dummy"}}),
    ],
    ids=["timesfm", "tabicl", "tabpfn", "nori"],
)
def test_init_clones_estimator_with_non_empty_config_dict(model_id, config_kwargs):
    """
    __init__ clones the estimator, which must not raise when the underlying
    adapter was built with a non-empty config dict. The cloned estimator
    preserves the config values.
    """
    estimator = FoundationModel(model_id, **config_kwargs)
    forecaster = ForecasterFoundation(estimator=estimator)

    assert forecaster.estimator is not estimator
    assert isinstance(forecaster.estimator, FoundationModel)
    assert forecaster.estimator.get_params() == estimator.get_params()


def test_init_two_forecasters_sharing_estimator_do_not_leak_state():
    """
    Two ForecasterFoundation instances built from the same FoundationModel
    instance must not share fit-derived state: fitting one must not affect
    the other's series_names_in_, context_range_, or exog_in_.
    """
    estimator = FoundationModel(
        "autogluon/chronos-2-small", context_length=8, pipeline=FakePipeline()
    )
    forecaster_1 = ForecasterFoundation(estimator=estimator)
    forecaster_2 = ForecasterFoundation(estimator=estimator)

    forecaster_1.fit(series=y)

    assert forecaster_1.is_fitted is True
    assert forecaster_2.is_fitted is False
    assert forecaster_2.series_names_in_ is None
    assert forecaster_2.context_range_ is None
    assert forecaster_2.exog_in_ is False


def test_init_default_attributes_before_fit():
    """
    All fit-time attributes are initialised to their 'unfitted' defaults.
    """
    forecaster = make_forecaster()

    assert forecaster.context_ is None
    assert forecaster.index_type_ is None
    assert forecaster.index_freq_ is None
    assert forecaster.context_range_ is None
    assert forecaster.series_names_in_ is None
    assert forecaster.is_multiple_series_ is False
    assert forecaster.exog_in_ is False
    assert forecaster.exog_names_in_ is None
    assert forecaster.exog_names_in_per_series_ is None
    assert forecaster.exog_type_in_ is None
    assert forecaster.is_fitted is False
    assert forecaster.fit_date is None


def test_init_delegated_properties_gated_on_is_fitted_when_estimator_state_is_stale():
    """
    Test that when `is_fitted` is manually reset to False after a real fit
    (simulating a stale/inconsistent state), all delegated properties
    consistently return their unfitted default instead of leaking stale
    data, and repr/_repr_html_ do not raise.
    """
    forecaster = make_forecaster()
    forecaster.fit(series=y)
    assert forecaster.is_fitted is True

    forecaster.is_fitted = False

    assert forecaster.context_ is None
    assert forecaster.context_exog_ is None
    assert forecaster.index_type_ is None
    assert forecaster.index_freq_ is None
    assert forecaster.context_range_ is None
    assert forecaster.series_names_in_ is None
    assert forecaster.is_multiple_series_ is False
    assert forecaster.exog_in_ is False
    assert forecaster.exog_names_in_ is None
    assert forecaster.exog_names_in_per_series_ is None
    assert forecaster.exog_type_in_ is None
    assert forecaster.fit_date is None

    assert "Context range: None" in repr(forecaster)
    assert "Not fitted" in forecaster._repr_html_()


@pytest.mark.parametrize(
    "forecaster_id, expected",
    [(None, None), ("my_forecaster", "my_forecaster")],
    ids=lambda v: f"forecaster_id={v}",
)
def test_init_forecaster_id_correctly_stored(forecaster_id, expected):
    """
    forecaster_id is stored exactly as provided (or defaults to None).
    """
    estimator = FoundationModel(
        "autogluon/chronos-2-small", pipeline=FakePipeline()
    )
    forecaster = ForecasterFoundation(
        estimator=estimator, forecaster_id=forecaster_id
    )
    assert forecaster.forecaster_id == expected


def test_init_metadata_and_tags_correctly_stored():
    """
    Metadata attributes (skforecast_version, python_version, creation_date)
    and __skforecast_tags__ are set with expected keys and values.
    """
    from skforecast import __version__ as sfv

    forecaster = make_forecaster()

    # Metadata
    assert forecaster.skforecast_version == sfv
    assert forecaster.python_version == sys.version.split(" ")[0]
    assert forecaster.creation_date is not None

    # Tags
    tags = forecaster.__skforecast_tags__
    assert tags["forecaster_name"] == "ForecasterFoundation"
    assert tags["supports_exog"] is True
    assert tags["supports_lags"] is False
    assert tags["supports_probabilistic"] is True
    assert "quantile_native" in tags["probabilistic_methods"]
