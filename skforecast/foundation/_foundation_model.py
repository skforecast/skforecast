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
    align_series_and_exog_multiseries,
    expand_index,
)


class FoundationModel:
    """
    Lightweight user-facing interface for foundation time-series models.

    Currently supports Amazon Chronos-2, Google TimesFM 2.5 and Salesforce
    Moirai-2. For full skforecast ecosystem integration (backtesting, model
    selection, etc.) use `ForecasterFoundation` instead.

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
    adapter : object
        The underlying adapter instance, instantiated automatically based on
        the `model` prefix. The concrete type depends on the model — e.g.
        `Chronos2Adapter` for `autogluon/chronos-*` models.
    context_length : int
        Maximum number of historical observations used as context. Mirrors
        `adapter.context_length`. Updated automatically when `set_params`
        is called.
    model_id : str
        HuggingFace model ID. Mirrors `adapter.model_id`. Updated
        automatically when `set_params` is called.
    allow_exogenous : bool
        Whether the underlying adapter supports exogenous variables.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    training_range_ : dict
        First and last values of index of the data used during training for
        each series.
    series_names_in_ : list
        Names of the series (levels) provided by the user during training.
    exog_in_ : bool
        If the model has been trained using exogenous variable/s.
    exog_names_in_ : list
        Names of the exogenous variables used during training.
    exog_names_in_per_series_ : dict
        Names of the exogenous variables used during training for each series.
    is_multiple_series_ : bool
        Whether the model was fitted with multiple series.
    creation_date : str
        Date of creation.
    is_fitted : bool
        Tag to identify if the model has been fitted (trained).
    fit_date : str
        Date of last fit.
    skforecast_version : str
        Version of skforecast library used to create the model.
    python_version : str
        Version of python used to create the model.

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
        is_fitted : bool
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
        allow_exogenous : bool
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
        series : pandas Series, pandas DataFrame, dict
            Training time series.

            - If `pandas Series`: single-series mode.
            - If wide `pandas DataFrame` (each column = one series):
            multi-series mode.
            - If `dict[str, pandas Series]`: multi-series mode; keys are
            series names.
        exog : pandas Series, pandas DataFrame, dict, default None
            Historical exogenous variables aligned to `series`.

            - If `pandas Series` or `pandas DataFrame`: broadcast to all
            series.
            - If `dict`: per-series exogenous variables.

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
        self.is_multiple_series_        = False
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

    def _prepare_future_exog(
        self,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ),
        series_names: list[str],
        steps: int,
        last_window: pd.Series | pd.DataFrame | dict[str, pd.Series] | None,
    ) -> dict[str, pd.DataFrame | pd.Series | None]:
        """
        Normalize, validate, and convert future exog to a per-series dict.

        Encapsulates the full pipeline for the forecast-horizon covariates:
        type coercion (long-format → dict), validation against training
        metadata (if fitted), and broadcast/dict conversion.

        Parameters
        ----------
        exog : pandas Series, pandas DataFrame, dict, None
            Future exogenous variables for the forecast horizon, in any
            supported input format.
        series_names : list
            Series names that define the output dict keys.
        steps : int
            Number of steps ahead to forecast.
        last_window : pandas Series, pandas DataFrame, dict, None
            Context override passed to predict. Used to determine the
            reference end-timestamp for index alignment.

        Returns
        -------
        future_exog_dict : dict
            Per-series dict with exactly the keys in `series_names`.
            Values are pandas DataFrame, pandas Series, or None.

        """

        exog = check_preprocess_exog_type(exog)

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

        return normalize_exog_to_dict(exog, series_names)

    def _prepare_past_exog(
        self,
        last_window_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ),
        last_window: pd.Series | pd.DataFrame | dict[str, pd.Series] | None,
        series_names: list[str],
    ) -> dict[str, pd.DataFrame | pd.Series | None]:
        """
        Normalize, validate, and convert past exog to a per-series dict.

        When `last_window` is None the stored adapter history exog is
        returned directly. Otherwise the full pipeline is applied: type
        coercion (long-format → dict), validation against training
        metadata (if fitted), and broadcast/dict conversion.

        Parameters
        ----------
        last_window_exog : pandas Series, pandas DataFrame, dict, None
            Historical exogenous variables corresponding to `last_window`.
        last_window : pandas Series, pandas DataFrame, dict, None
            Context override passed to predict. When None, stored
            adapter history is used instead.
        series_names : list
            Series names that define the output dict keys.

        Returns
        -------
        past_exog_dict : dict
            Per-series dict with exactly the keys in `series_names`.
            Values are pandas DataFrame, pandas Series, or None.

        """

        if last_window is None:
            if self.adapter._history_exog is not None:
                return self.adapter._history_exog
            return {name: None for name in series_names}

        last_window_exog = check_preprocess_exog_type(last_window_exog)

        if self.is_fitted and self.exog_in_:
            validate_last_window_exog(
                last_window_exog = last_window_exog,
                last_window      = last_window,
                exog_in_         = self.exog_in_,
            )

        return normalize_exog_to_dict(last_window_exog, series_names)

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
        Predict n steps ahead.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        exog : pandas Series, pandas DataFrame, dict, default None
            Future known exogenous variables for the forecast horizon.

            - If `pandas Series` or `pandas DataFrame`: broadcast to all
            series.
            - If `dict`: per-series exogenous variables.
        quantiles : list, tuple, default None
            Quantile levels to return, e.g. `[0.1, 0.5, 0.9]`. If `None`,
            returns a point forecast (median).
        last_window : pandas Series, pandas DataFrame, dict, default None
            Override the stored history with this window.

            - If `pandas Series`: single-series override.
            - If wide `pandas DataFrame` or `dict[str, pandas Series]`:
            multi-series override.
        last_window_exog : pandas Series, pandas DataFrame, dict, default None
            Historical exog corresponding to `last_window`.

        Returns
        -------
        predictions : pandas DataFrame
            Always a long-format DataFrame. Point forecast: columns
            `["level", "pred"]`. Quantile forecast: columns
            `["level", "q_0.1", "q_0.5", ...]`. The index repeats each
            forecast timestamp once per series.
        
        """

        if not self.is_fitted and last_window is None:
            raise ValueError(
                "Call `fit` before `predict`, or pass `last_window`."
            )

        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("`steps` must be a positive integer.")

        if quantiles is not None:
            for q in quantiles:
                if not 0.0 <= q <= 1.0:
                    raise ValueError(
                        f"All quantiles must be between 0 and 1. Got {q}."
                    )

        # Resolve history first — the series context is the primary input
        # and must be established before processing exogenous variables.
        if last_window is not None:
            history_dict, _ = check_preprocess_series_foundation(last_window)
            series_names = list(history_dict.keys())
        else:
            history_dict = self.adapter._history
            series_names = list(history_dict.keys())

        # Validate last_window index consistency with training data.
        if last_window is not None and self.is_fitted:
            ref_index = next(iter(history_dict.values())).index
            if not isinstance(ref_index, self.index_type_):
                raise TypeError(
                    f"Expected index of type {self.index_type_.__name__} "
                    f"for `last_window`. Got {type(ref_index).__name__}."
                )
            if isinstance(ref_index, pd.DatetimeIndex):
                if ref_index.freq != self.index_freq_:
                    raise TypeError(
                        f"Expected frequency '{self.index_freq_}' for "
                        f"`last_window` index. Got '{ref_index.freq}'."
                    )

        # Handle adapters that don't support exogenous variables.
        # Must run before any exog validation so that unsupported exog is
        # discarded early without triggering validation errors.
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

        # Prepare exog: normalize types, validate against training
        # metadata, and convert to per-series dicts.
        future_exog_dict = self._prepare_future_exog(
                               exog         = exog,
                               series_names = series_names,
                               steps        = steps,
                               last_window  = last_window,
                           )
        past_exog_dict = self._prepare_past_exog(
                             last_window_exog = last_window_exog,
                             last_window      = last_window,
                             series_names     = series_names,
                         )

        # Trim history and past exog to context_length. Stored history
        # is already trimmed in adapter.fit(), so only trim when the
        # user provides a custom last_window.
        if last_window is not None:
            history_dict = {
                name: s.iloc[-self.context_length :]
                for name, s in history_dict.items()
            }
            past_exog_dict = {
                name: (
                    e.iloc[-self.context_length :] if e is not None else None
                )
                for name, e in past_exog_dict.items()
            }

        is_multiple_series = len(series_names) > 1

        # Adapter returns dict[str, np.ndarray] with shape (steps, n_q)
        raw_predictions = self.adapter.predict(
                              steps              = steps,
                              history_dict       = history_dict,
                              past_exog_dict     = past_exog_dict,
                              future_exog_dict   = future_exog_dict,
                              quantiles          = quantiles,
                              is_multiple_series = is_multiple_series,
                          )

        # Build long-format DataFrame from raw predictions
        per_series_indices = [
            expand_index(history_dict[name].index, steps=steps)
            for name in series_names
        ]

        if len(series_names) == 1:
            long_index = per_series_indices[0]
            level_col = np.repeat(series_names, steps)
        else:
            long_index = np.column_stack(
                [np.asarray(idx) for idx in per_series_indices]
            ).ravel()
            level_col = np.tile(series_names, steps)

        if quantiles is None:
            # Point forecast (median): single "pred" column
            pred_matrix = np.column_stack([
                raw_predictions[name][:, 0] for name in series_names
            ])
            predictions = pd.DataFrame(
                {"level": level_col, "pred": pred_matrix.ravel()},
                index=long_index,
            )
        else:
            q_columns = [f"q_{q}" for q in quantiles]
            data_dict: dict[str, np.ndarray] = {"level": level_col}
            for j, col in enumerate(q_columns):
                q_matrix = np.column_stack([
                    raw_predictions[name][:, j] for name in series_names
                ])
                data_dict[col] = q_matrix.ravel()
            predictions = pd.DataFrame(data_dict, index=long_index)

        return predictions

    def get_params(self, deep: Any = None) -> dict:
        """
        Get parameters for this estimator (sklearn-compatible).

        Parameters
        ----------
        deep : Any, default None
            Not used, present here for API consistency by convention.

        Returns
        -------
        params : dict
            Parameter names mapped to their current values.

        Notes
        -----
        Required so that `sklearn.base.clone` can create an unfitted copy
        of this object, which is used internally by `deepcopy_forecaster`
        during backtesting. The pre-loaded pipeline is intentionally excluded
        so that clones are created without copying heavy model weights; the
        pipeline is reloaded lazily on the first `predict` call.
        
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
            The same object with updated parameters.

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
