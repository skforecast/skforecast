################################################################################
#                     skforecast.foundation._adapter_base                      #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# Contract shared by every foundation model adapter. `FoundationModel` and
# `get_model_info` rely on it, so a change here affects all the adapters:
# modify it only when the contract itself changes. To add a new backend,
# subclass `_AdapterBase` in `_adapters.py`, declare every class attribute in
# `_REQUIRED_CLASS_ATTRIBUTES` and register the class in `_ADAPTER_REGISTRY`.

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class _AdapterBase(ABC):
    """
    Base class for all foundation model adapters. It declares the contract
    that `FoundationModel` relies on: the class attributes listed in
    `_REQUIRED_CLASS_ATTRIBUTES`, which every adapter must define in its own
    class body (checked when the subclass is defined), the abstract methods
    `predict`, `get_params` and `set_params`, and the shared `fit`.

    Adapters only translate already-normalized per-series inputs into the
    backend call. Generic handling of the inputs belongs to `FoundationModel`,
    driven by the capability class attributes.

    `requires_equal_length_context` (bool) declares whether the backend needs
    every series of a call to have the same context length. If `True`,
    `FoundationModel` left-pads `context` and `context_exog` of every call
    with NaN to the length of the longest series of the call, so the adapter
    receives the context with a positional pandas RangeIndex and must not
    rely on its timestamps. The same applies to `context_exog`, while the
    future `exog` keeps the index of the forecast horizon: history and future
    must be joined by position, never aligned by index. Boolean exog columns
    arrive as float, and NumPy integer columns of the padded series as float
    (nullable `Int64` and `Float64` columns keep their dtype and hold
    `pd.NA`). Requires `supports_nan_in_series=True`; adapters that derive
    features from the timestamps must keep it `False`.

    `supports_past_only_covariates` (bool) declares whether the backend uses
    the historical exog columns that have no future values (past-only
    covariates). If `False`, `FoundationModel` removes those columns from
    `context_exog` before every call, so the adapter only receives
    covariates that also have future values in `exog` and needs no filtering
    of its own.

    `max_series_per_call` (int, None) caps the number of series sent to the
    backend in one call. If it is an integer, `FoundationModel` splits each
    group of series into consecutive chunks of at most that size and calls
    `predict` once per chunk (the left padding above is applied per chunk).
    `None` sends every series of a group in one call. Declare it for
    backends whose cost per series, or memory, grows with the size of the
    batch.
    """

    _REQUIRED_CLASS_ATTRIBUTES = (
        "SUPPORTED_QUANTILES",
        "allow_exog",
        "supports_past_only_covariates",
        "supports_categorical_covariates",
        "supports_heterogeneous_covariates",
        "requires_equal_length_context",
        "max_series_per_call",
        "supports_nan_in_series",
        "requires_hf_auth",
        "requires_provider_auth",
        "weights_repo_id",
        "weights_in_hf_cache",
        "backend_package",
        "default_model_id",
    )

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Check that the subclass defines every required class attribute in its
        own class body. Inherited values do not count, so each adapter states
        its capabilities explicitly.
        """

        super().__init_subclass__(**kwargs)
        missing = [
            name for name in cls._REQUIRED_CLASS_ATTRIBUTES
            if name not in vars(cls)
        ]
        if missing:
            raise TypeError(
                f"`{cls.__name__}` must define the class attribute(s) "
                f"{missing} in its class body."
            )

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> _AdapterBase:
        """
        Store the training series and historical exogenous variables, and
        mark the adapter as fitted. Zero-shot adapters do no training, so
        they override `fit` only to document it and delegate here with
        `super().fit(...)`.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).
            `FoundationModel` passes `None` when no exog was given or the
            adapter does not support exogenous variables (`allow_exog=False`).

        Returns
        -------
        self : _AdapterBase

        """

        self.context_ = context
        self.context_exog_ = context_exog
        self.is_fitted = True

        return self

    @abstractmethod
    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float, ...] | None
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions with the backend.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict pandas Series
            Per-series context windows (already trimmed to `context_length`),
            left-padded with NaN to a common length and indexed by a
            positional pandas RangeIndex when `requires_equal_length_context`
            is `True`. It holds at most `max_series_per_call` series when that
            attribute is not `None`.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series past covariates (already trimmed), left-padded and
            indexed like `context` when `requires_equal_length_context` is
            `True`. When `supports_past_only_covariates` is `False`, they
            only hold columns that also have future values in `exog`.
        exog : dict pandas DataFrame, pandas Series, or None
            Per-series future covariates for the forecast horizon.
        quantiles : list of float, tuple of float, None
            Quantile levels to return. If `None`, a point forecast is produced.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)`.

        """

        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters, except the injected
        backend object.

        Returns
        -------
        params : dict
            Parameter names mapped to their current values.

        """

        pass

    @abstractmethod
    def set_params(self, **params) -> _AdapterBase:
        """
        Set adapter parameters.

        Parameters
        ----------
        **params :
            Valid keys are the ones returned by `get_params`.

        Returns
        -------
        self : _AdapterBase

        """

        pass
