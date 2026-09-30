################################################################################
#                       skforecast.foundation._model_info                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# Capability metadata of the foundation model adapters. Everything is read from
# the adapter classes and the license registry, so this module is the public
# view of those sources and never duplicates them. No backend library is
# imported.

from __future__ import annotations
from dataclasses import asdict, dataclass
import inspect
import pandas as pd

from ._adapters import _ADAPTER_REGISTRY, _resolve_adapter
from ._utils import _get_non_commercial_license


@dataclass(frozen=True, kw_only=True)
class FoundationModelInfo:
    """
    Capabilities and requirements of a foundation model supported by
    skforecast.

    Instances are created by `get_model_info` and `list_adapters`; they are
    immutable and can be converted to a `dict` with `dataclasses.asdict`.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID the information refers to.
    adapter : str
        Name of the adapter class that serves `model_id`.
    model_id_prefixes : tuple
        Model ID prefixes routed to `adapter`.
    default_model_id : str
        Model ID used by default for `adapter`.
    default_context_length : int
        Default maximum number of historical observations used as context.
    backend_package : str
        Package that provides the backend, as passed to `pip install`.
    allow_exog : bool
        Whether exogenous variables (covariates) are supported.
    supports_past_only_covariates : bool
        Whether covariates without future values are used as past-only
        covariates. When `False`, they are ignored with a warning.
    supports_categorical_covariates : bool
        Whether the backend supports non-numeric covariates natively, so
        they do not have to be encoded as numbers. `False` also when the
        adapter forwards them unchanged but skforecast does not verify how
        the backend handles them.
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. When `False`, the series are grouped by their
        covariate columns and the backend is called once per group.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as context.
    supported_quantiles : tuple, None
        Quantile levels accepted by the backend. `None` means any level in
        `(0, 1)`.
    requires_hf_auth : bool
        Whether the checkpoints served by `adapter` are gated on the Hugging
        Face Hub, so an authenticated account that has accepted the model
        license is needed. Unlike the license fields, it is declared per
        adapter, not resolved for `model_id`.
    license_restriction : str, None
        Name of the license that restricts commercial use of the weights, as
        also reported by `LicenseWarning` when they are loaded. `None` only
        means that no restriction is registered in skforecast, it does not
        confirm that the license permits commercial use.
    license_url : str, None
        Link to the terms of `license_restriction`. `None` when
        `license_restriction` is `None`.

    """

    model_id: str
    adapter: str
    model_id_prefixes: tuple[str, ...]
    default_model_id: str
    default_context_length: int
    backend_package: str
    allow_exog: bool
    supports_past_only_covariates: bool
    supports_categorical_covariates: bool
    supports_heterogeneous_covariates: bool
    supports_nan_in_series: bool
    supported_quantiles: tuple[float, ...] | None
    requires_hf_auth: bool
    license_restriction: str | None
    license_url: str | None


def _build_model_info(model_id: str, adapter_cls: type) -> FoundationModelInfo:
    """
    Build the `FoundationModelInfo` of `model_id` from its adapter class.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID the information refers to.
    adapter_cls : type
        Adapter class that serves `model_id`.

    Returns
    -------
    info : FoundationModelInfo
        Capabilities and requirements of `model_id`.

    """

    # The default context length lives only in the adapter signature, so it
    # is read from there instead of being declared twice.
    default_context_length = (
        inspect.signature(adapter_cls).parameters["context_length"].default
    )
    supported_quantiles = adapter_cls.SUPPORTED_QUANTILES
    license_info = _get_non_commercial_license(model_id)
    license_restriction, license_url = (
        license_info if license_info is not None else (None, None)
    )

    prefixes = tuple(
        prefix for prefix, cls in _ADAPTER_REGISTRY.items() if cls is adapter_cls
    )
    if supported_quantiles is not None:
        supported_quantiles = tuple(supported_quantiles)

    info = FoundationModelInfo(
        model_id                          = model_id,
        adapter                           = adapter_cls.__name__,
        model_id_prefixes                 = prefixes,
        default_model_id                  = adapter_cls.default_model_id,
        default_context_length            = default_context_length,
        backend_package                   = adapter_cls.backend_package,
        allow_exog                        = adapter_cls.allow_exog,
        supports_past_only_covariates     = adapter_cls.supports_past_only_covariates,
        supports_categorical_covariates   = adapter_cls.supports_categorical_covariates,
        supports_heterogeneous_covariates = adapter_cls.supports_heterogeneous_covariates,
        supports_nan_in_series            = adapter_cls.supports_nan_in_series,
        supported_quantiles               = supported_quantiles,
        requires_hf_auth                  = adapter_cls.requires_hf_auth,
        license_restriction               = license_restriction,
        license_url                       = license_url,
    )

    return info


def get_model_info(model_id: str) -> FoundationModelInfo:
    """
    Return the capabilities and requirements of a foundation model.

    The adapter is resolved from `model_id` in the same way as
    `FoundationModel` does, but neither the backend library nor the weights
    are loaded, so the backend does not need to be installed.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `'autogluon/chronos-2-small'`.

    Returns
    -------
    info : FoundationModelInfo
        Capabilities and requirements of `model_id`. The license fields are
        resolved for this specific `model_id`.

    """

    if not isinstance(model_id, str):
        raise TypeError(
            f"`model_id` must be a string. Got {type(model_id).__name__}."
        )

    adapter_cls = _resolve_adapter(model_id)

    return _build_model_info(model_id=model_id, adapter_cls=adapter_cls)


def list_adapters(
    as_frame: bool = False
) -> list[FoundationModelInfo] | pd.DataFrame:
    """
    Return the capabilities and requirements of every foundation model
    adapter supported by skforecast.

    Each adapter is described by its `default_model_id`, so the license
    fields refer to that model ID. Use `get_model_info` to resolve them for a
    different checkpoint.

    Parameters
    ----------
    as_frame : bool, default False
        If `True`, return a pandas DataFrame with one row per adapter, indexed
        by the adapter name, instead of a list of `FoundationModelInfo`.

    Returns
    -------
    adapters : list, pandas DataFrame
        One `FoundationModelInfo` per adapter, in registration order. If
        `as_frame=True`, a DataFrame with one row per adapter (index
        `adapter`) and one column per remaining field of
        `FoundationModelInfo`.

    """

    adapter_classes = list(dict.fromkeys(_ADAPTER_REGISTRY.values()))
    adapters = [
        _build_model_info(model_id=cls.default_model_id, adapter_cls=cls)
        for cls in adapter_classes
    ]

    if as_frame:
        adapters = pd.DataFrame(
            [asdict(adapter) for adapter in adapters]
        ).set_index("adapter")

    return adapters
