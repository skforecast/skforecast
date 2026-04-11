################################################################################
#                               FoundationModel                                #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Any
import sys
import warnings
import numpy as np
import pandas as pd

from .. import __version__
from ._adapters import _resolve_adapter
from ._utils import (
    check_preprocess_exog_type,
    check_preprocess_series_foundation,
    normalize_exog_to_dict,
    validate_exog_predict,
    validate_last_window_exog,
)
from ..exceptions import IgnoredArgumentWarning
from ..utils import (
    check_preprocess_exog_multiseries,
    align_series_and_exog_multiseries
)

# TODO: en todos los check_y añadir allow_nan = True


class FoundationModel:
    """
    Lightweight user-facing interface for foundation time-series models.

    Currently supports Amazon Chronos-2 only. For full skforecast
    ecosystem integration (backtesting, model selection, etc.) use
    `ForecasterFoundation` instead.

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
    be installed — other foundation-model backends remain optional.

    """

    def __init__(
        self,
        model: str,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the FoundationModels interface.

        Parameters
        ----------
        model : str
            HuggingFace model ID, e.g. "autogluon/chronos-2-small".
        **kwargs :
            Additional keyword arguments forwarded to the underlying adapter.
            Valid keys depend on the adapter selected by `model`; see the
            corresponding adapter class for the full parameter list.
        
        """

        adapter_cls                    = _resolve_adapter(model)
        self.adapter                   = adapter_cls(model_id=model, **kwargs)
        self.context_length            = self.adapter.context_length
        self.model_id                  = self.adapter.model_id
        self.index_type_               = None
        self.index_freq_               = None
        self.training_range_           = None
        self.series_names_in_          = None
        self.exog_in_                  = False
        self.exog_names_in_            = None
        self.exog_names_in_per_series_ = None
        self.is_multiple_series_       = False
        self.creation_date             = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                  = None
        self.skforecast_version        = __version__
        self.python_version            = sys.version.split(" ")[0]

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
        return self.adapter.is_fitted

    @property
    def allow_exogenous(self) -> bool:
        """
        Whether the underlying adapter supports exogenous variables.

        Returns
        -------
        bool
            `True` if the adapter accepts and uses `exog`; `False` if it
            ignores covariates (e.g. TimesFM 2.5, Moirai-2).
        """
        return self.adapter.allow_exogenous

    def fit(
        self,
        series: pd.Series | pd.DataFrame | dict[str, pd.Series],
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> FoundationModel:
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
        self : FoundationModel

        """

        self.index_type_                = None
        self.index_freq_                = None
        self.training_range_            = None
        self.series_names_in_           = None
        self.exog_in_                   = False
        self.exog_names_in_             = None
        self.exog_names_in_per_series_  = None
        self.is_multiple_series_       = False
        self.fit_date                   = None

        series_dict, series_indexes = check_preprocess_series_foundation(series)
        series_names_in_ = list(series_dict.keys())
        is_multiple_series_ = len(series_names_in_) > 1

        series_index_type = type(series_indexes[series_names_in_[0]])
        if exog is not None:
            exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                series_names_in_  = series_names_in_,
                series_index_type = series_index_type,
                exog              = exog,
                exog_dict         = {name: None for name in series_names_in_},
            )
        else:
            exog_dict = {name: None for name in series_names_in_}

        series_dict, exog_dict = align_series_and_exog_multiseries(
                                     series_dict = series_dict,
                                     exog_dict   = exog_dict
                                 )

        self.adapter.fit(
            series_dict        = series_dict,
            exog_dict          = exog_dict,
            is_multiple_series = is_multiple_series_
        )

        self.series_names_in_    = series_names_in_
        self.is_multiple_series_ = is_multiple_series_

        self.exog_names_in_per_series_ = {
            k: list(v.columns) if v is not None else None
            for k, v in exog_dict.items()
        }
        if exog is not None:
            self.exog_in_ = len(exog_names_in_) > 0
            self.exog_names_in_ = exog_names_in_

        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.training_range_ = {k: v[[0, -1]] for k, v in series_indexes.items()}
        self.index_type_ = series_index_type
        if isinstance(series_indexes[series_names_in_[0]], pd.DatetimeIndex):
            self.index_freq_ = series_indexes[series_names_in_[0]].freq
        else:
            self.index_freq_ = series_indexes[series_names_in_[0]].step

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
    ) -> pd.DataFrame:
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
        predictions : pd.DataFrame
            Always a long-format DataFrame. Point forecast: columns
            `["level", "pred"]`. Quantile forecast: columns
            `["level", "q_0.1", "q_0.5", ...]`. The index repeats each
            forecast timestamp once per series.
        
        """

        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("`steps` must be a positive integer.")

        if quantiles is not None:
            for q in quantiles:
                if not 0.0 <= q <= 1.0:
                    raise ValueError(
                        f"All quantiles must be between 0 and 1. Got {q}."
                    )

        if not self.is_fitted and last_window is None:
            raise ValueError(
                "Call `fit` before `predict`, or pass `last_window`."
            )

        # Validate exog and last_window_exog against training metadata
        # (only when fitted AND trained with exog — foundation models are
        # zero-shot, so exog can be introduced at predict time without having
        # been present during fit).
        if self.is_fitted and self.exog_in_:
            exog = validate_exog_predict(
                exog                      = exog,
                steps                     = steps,
                last_window               = last_window,
                exog_names_in_            = self.exog_names_in_,
                exog_in_                  = self.exog_in_,
                index_freq_               = self.index_freq_,
                is_multiseries            = self.is_multiple_series_,
                training_range_           = self.training_range_,
                series_names_in_          = self.series_names_in_,
                exog_names_in_per_series_ = self.exog_names_in_per_series_,
            )

            validate_last_window_exog(
                last_window_exog = last_window_exog,
                last_window      = last_window,
                exog_in_         = self.exog_in_,
            )

        # Handle adapters that don't support exogenous variables
        if not self.allow_exogenous:
            has_exog = (
                (exog is not None)
                or (last_window_exog is not None)
            )
            if has_exog:
                warnings.warn(
                    f"{type(self.adapter).__name__} does not currently "
                    "support covariates. `exog` and `last_window_exog` "
                    "are ignored.",
                    IgnoredArgumentWarning,
                    stacklevel=2,
                )
                exog = None
                last_window_exog = None

        # Normalize exog types (long-format DataFrame → dict)
        exog = check_preprocess_exog_type(exog)
        last_window_exog = check_preprocess_exog_type(last_window_exog)

        # Resolve history
        if last_window is not None:
            history_dict, _ = check_preprocess_series_foundation(last_window)
            series_names = list(history_dict.keys())
        else:
            history_dict = self.adapter._history
            series_names = list(history_dict.keys())

        # Trim history to context_length
        history_dict = {
            name: s.iloc[-self.context_length :]
            for name, s in history_dict.items()
        }

        # Resolve past exog
        if last_window is not None:
            past_exog_dict = normalize_exog_to_dict(
                                 last_window_exog, series_names
                             )
            past_exog_dict = {
                name: (
                    e.iloc[-self.context_length :] if e is not None else None
                )
                for name, e in past_exog_dict.items()
            }
        else:
            history_exog = getattr(self.adapter, '_history_exog', None)
            if history_exog is not None:
                past_exog_dict = history_exog
            else:
                past_exog_dict = {name: None for name in series_names}

        # Resolve future exog
        future_exog_dict = normalize_exog_to_dict(exog, series_names)

        is_multiple_series = len(series_names) > 1

        predictions = self.adapter.predict(
                          steps              = steps,
                          history_dict       = history_dict,
                          past_exog_dict     = past_exog_dict,
                          future_exog_dict   = future_exog_dict,
                          quantiles          = quantiles,
                          is_multiple_series = is_multiple_series,
                      )

        return predictions

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
        # Expose model_id as 'model' to match the FoundationModel constructor
        params['model'] = params.pop('model_id')
        return params

    def set_params(self, **params) -> FoundationModel:
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
        self : FoundationModel

        """

        if 'model' in params:
            params = dict(params)
            params['model_id'] = params.pop('model')
        try:
            self.adapter.set_params(**params)
        except ValueError as exc:
            raise ValueError(str(exc).replace(type(self.adapter).__name__, "FoundationModel")) from exc
        
        self.context_length = self.adapter.context_length
        self.model_id       = self.adapter.model_id

        return self
        