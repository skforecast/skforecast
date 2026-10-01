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
    class body (checked when the subclass is defined), and the methods `fit`,
    `predict`, `get_params` and `set_params`.

    Adapters only translate already-normalized per-series inputs into the
    backend call. Generic handling of the inputs belongs to `FoundationModel`,
    driven by the capability class attributes.
    """

    _REQUIRED_CLASS_ATTRIBUTES = (
        "SUPPORTED_QUANTILES",
        "allow_exog",
        "supports_past_only_covariates",
        "supports_categorical_covariates",
        "supports_heterogeneous_covariates",
        "supports_nan_in_series",
        "requires_hf_auth",
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

    def _fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> _AdapterBase:
        """
        Store the training series and historical exogenous variables, and
        mark the adapter as fitted. Shared implementation of `fit` for
        zero-shot adapters.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : _AdapterBase

        """

        self.context_ = context
        self.context_exog_ = context_exog
        self.is_fitted = True

        return self

    @abstractmethod
    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> _AdapterBase:
        """
        Store the training series and historical exogenous variables.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : _AdapterBase

        """

        pass

    @abstractmethod
    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float] | None
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions with the backend.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict pandas Series
            Per-series context windows (already trimmed to `context_length`).
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series past covariates (already trimmed).
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
