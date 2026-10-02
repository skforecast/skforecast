# Unit test _AdapterBase
# ==============================================================================
import re
import inspect
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapter_base import _AdapterBase
from skforecast.foundation._adapters import _ADAPTER_REGISTRY

# Backend objects injected through `__init__`, not exposed by `get_params`
INJECTED_BACKEND_PARAMS = {"pipeline", "module", "model"}


# ==============================================================================
# Tests _AdapterBase.__init_subclass__
# ==============================================================================
def test_AdapterBase_TypeError_when_subclass_misses_required_class_attributes():
    """
    Test that defining an adapter that does not declare every required class
    attribute in its own class body raises TypeError listing the missing ones.
    """
    err_msg = re.escape(
        "`IncompleteAdapter` must define the class attribute(s) "
        "['supports_nan_in_series', 'requires_hf_auth', 'backend_package', "
        "'default_model_id'] in its class body."
    )
    with pytest.raises(TypeError, match=err_msg):

        class IncompleteAdapter(_AdapterBase):
            SUPPORTED_QUANTILES = None
            allow_exog = True
            supports_past_only_covariates = False
            supports_categorical_covariates = False
            supports_heterogeneous_covariates = True


# ==============================================================================
# Tests contract of every registered adapter
# ==============================================================================
@pytest.mark.parametrize(
    "adapter_cls",
    list(dict.fromkeys(_ADAPTER_REGISTRY.values())),
    ids=lambda cls: cls.__name__,
)
def test_AdapterBase_every_registered_adapter_honors_the_contract(adapter_cls):
    """
    Contract test: every registered adapter inherits from _AdapterBase, can be
    created from its default model ID without the backend installed, exposes
    every constructor parameter (except the injected backend object) through
    `get_params`, survives a `set_params(**get_params())` round trip without
    any change, and `fit` stores the context and historical exog it receives
    (FoundationModel decides what to pass) and returns the adapter.
    """
    assert issubclass(adapter_cls, _AdapterBase)

    adapter = adapter_cls(adapter_cls.default_model_id)

    init_params = set(inspect.signature(adapter_cls).parameters)
    params = adapter.get_params()
    assert set(params) == init_params - INJECTED_BACKEND_PARAMS

    state_before = dict(vars(adapter))
    returned = adapter.set_params(**params)
    assert returned is adapter
    assert adapter.get_params() == params
    assert vars(adapter).keys() == state_before.keys()
    for name, value in state_before.items():
        assert vars(adapter)[name] is value, name

    index = pd.date_range("2020-01-01", periods=10, freq="D")
    context = {"s1": pd.Series(np.arange(10, dtype=float), index=index, name="s1")}
    context_exog = {
        "s1": pd.DataFrame({"feat": np.arange(10, dtype=float)}, index=index)
    }

    returned = adapter.fit(context=context, context_exog=None)
    assert returned is adapter
    assert adapter.is_fitted is True
    assert adapter.context_ is context
    assert adapter.context_exog_ is None

    adapter.fit(context=context, context_exog=context_exog)
    assert adapter.context_exog_ is context_exog
