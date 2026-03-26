################################################################################
#                           FoundationalModel                                  #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Any
import pandas as pd
from ._adapters import _resolve_adapter


class FoundationalModel:
    """
    Lightweight user-facing interface for foundational time-series models.

    Currently supports Amazon Chronos-2 only. For full skforecast
    ecosystem integration (backtesting, model selection, etc.) use
    `ForecasterFoundational` instead.

    Parameters
    ----------
    model : str
        HuggingFace model ID, e.g. "autogluon/chronos-2-small".
    **kwargs :
        Additional keyword arguments forwarded to the underlying adapter.
        Valid keys depend on the adapter selected by `model`; see the
        corresponding adapter class for the full parameter list.

    Attributes
    ----------
    context_length : int
        Maximum number of historical observations used as context. Mirrors
        `adapter.context_length`. Updated automatically when `set_params`
        is called.
    model_id : str
        HuggingFace model ID. Mirrors `adapter.model_id`. Updated
        automatically when `set_params` is called.
    adapter : object
        The underlying adapter instance, instantiated automatically based on
        the `model` prefix. The concrete type depends on the model — e.g.
        `Chronos2Adapter` for `autogluon/chronos-*` models.

    Notes
    -----
    Each adapter imports its own backend library lazily (i.e. inside the
    method that first needs it) rather than at module level. This means
    that only the library required by the adapter you actually use needs to
    be installed — other foundational-model backends remain optional.

    """

    def __init__(
        self,
        model: str,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the FoundationalModels interface.

        Parameters
        ----------
        model : str
            HuggingFace model ID, e.g. "autogluon/chronos-2-small".
        **kwargs :
            Additional keyword arguments forwarded to the underlying adapter.
            Valid keys depend on the adapter selected by `model`; see the
            corresponding adapter class for the full parameter list.
        """
        adapter_cls = _resolve_adapter(model)
        self.adapter = adapter_cls(model_id=model, **kwargs)
        self.context_length = self.adapter.context_length
        self.model_id       = self.adapter.model_id

    @property
    def is_fitted(self) -> bool:
        """
        Whether the model has been fitted.

        Returns
        -------
        bool
            `True` after `fit` has been called at least once,
            `False` otherwise.
        """
        return self.adapter._is_fitted


    def fit(
        self,
        series: pd.Series | pd.DataFrame | dict[str, pd.Series],
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> FoundationalModel:
        """
        Fit the model by storing the training series and optional exog.

        Parameters
        ----------
        series : pd.Series, pd.DataFrame, or dict of pd.Series
            Training time series.

            - `pd.Series`: single-series mode.
            - Wide `pd.DataFrame` (each column = one series): multi-series
              mode.
            - `dict[str, pd.Series]`: multi-series mode; keys are series
              names.
        exog : pd.Series, pd.DataFrame, dict, or None
            Historical exogenous variables aligned to `series`.

            - `pd.Series` or `pd.DataFrame`: broadcast to all series.
            - `dict[str, pd.DataFrame | pd.Series | None]`: per-series.

        Returns
        -------
        self : FoundationalModel
        """

        self.adapter.fit(series=series, exog=exog)
        
        return self

    def predict(
        self,
        steps: int,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
        quantiles: list[float] | tuple[float] | None = None,
        last_window: pd.Series | pd.DataFrame | dict[str, pd.Series] | None = None,
        last_window_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Generate predictions.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        exog : pd.Series, pd.DataFrame, dict, or None
            Future known exogenous variables for the forecast horizon.
            Broadcast or per-series dict; see `Chronos2Adapter.predict`.
        quantiles : list of float, optional
            Quantile levels to return, e.g. `[0.1, 0.5, 0.9]`. If None,
            returns a point forecast (median).
        last_window : pd.Series, pd.DataFrame, dict, or None
            Override the stored history with this window.

            - `pd.Series`: single-series override.
            - Wide `pd.DataFrame` or `dict[str, pd.Series]`: multi-series override.
        last_window_exog : pd.Series, pd.DataFrame, dict, or None
            Historical exog corresponding to `last_window`.

        Returns
        -------
        pd.Series
            Point forecast (single-series, no quantiles).
        pd.DataFrame
            Single-series with quantiles: columns are `q_0.1`, `q_0.5`,
            etc. Multi-series point forecast: long format with columns
            `["level", "pred"]`. Multi-series quantile forecast: long
            format with columns `["level", "q_0.1", "q_0.5", ...]`. In both
            multi-series cases the index repeats each forecast timestamp once
            per series (`n_steps x n_series` rows total).
        """

        return self.adapter.predict(
            steps            = steps,
            exog             = exog,
            quantiles        = quantiles,
            last_window      = last_window,
            last_window_exog = last_window_exog,
        )

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator (sklearn-compatible).

        Required so that `sklearn.base.clone` can create an unfitted copy
        of this object, which is used internally by `deepcopy_forecaster`
        during backtesting. The pre-loaded pipeline is intentionally excluded
        so that clones are created without copying heavy model weights; the
        pipeline is reloaded lazily on the first `predict` call.

        Parameters
        ----------
        deep : bool, default True
            Ignored. Included for sklearn API compatibility.

        Returns
        -------
        params : dict
            Parameter names mapped to their current values.
        """

        params = self.adapter.get_params()
        # Expose model_id as 'model' to match the FoundationalModel constructor
        params['model'] = params.pop('model_id')
        return params

    def set_params(self, **params) -> FoundationalModel:
        """
        Set parameters for this estimator (sklearn-compatible).

        Parameters
        ----------
        **params :
            Estimator parameters forwarded to the underlying adapter's
            `set_params`. Use `model` to change the model ID (mapped
            to `model_id` on the adapter). All other keys are
            adapter-specific.

        Returns
        -------
        self : FoundationalModel
        """

        if 'model' in params:
            params = dict(params)
            params['model_id'] = params.pop('model')
        try:
            self.adapter.set_params(**params)
        except ValueError as exc:
            raise ValueError(str(exc).replace(type(self.adapter).__name__, "FoundationalModel")) from exc
        self.context_length = self.adapter.context_length
        self.model_id       = self.adapter.model_id
        return self