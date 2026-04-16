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
from ._utils import check_preprocess_series_foundation
from ..exceptions import IgnoredArgumentWarning, InputTypeWarning, MissingValuesWarning
from ..utils import (
    check_preprocess_exog_multiseries,
    align_series_and_exog_multiseries,
    expand_index,
)


class FoundationModel:
    """
    Scikit-learn compatible interface for foundation time-series models.

    Currently supports Amazon Chronos-2, Google TimesFM 2.5 and Salesforce
    Moirai-2. For full skforecast ecosystem integration (backtesting, model
    selection, etc.) use `ForecasterFoundation` instead.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "autogluon/chronos-2-small".
    **kwargs :
        Additional keyword arguments forwarded to the underlying adapter.
        Valid keys depend on the adapter selected by `model_id`; see the
        corresponding adapter class for the full parameter list.

    Attributes
    ----------
    adapter : object
        The underlying adapter instance, instantiated automatically based on
        the `model_id` prefix. The concrete type depends on the model — e.g.
        `Chronos2Adapter` for `autogluon/chronos-*` models.
    model_id : str
        HuggingFace model ID. Mirrors `adapter.model_id`.
    context_ : dict[str, pandas Series]
        Per-series dict of pandas Series containing the last `context_length`
        observations from the training data, stored during `fit`. Mirrors
        `adapter.context_`.
    context_exog_ : dict
        Per-series dict of pandas DataFrame containing the last `context_length`
        exog variables from the training data, stored during `fit`. `None` if
        the adapter does not support exogenous variables or no exog was
        provided. Mirrors `adapter.context_exog_`.
    context_length : int
        Maximum number of historical observations used as context. Mirrors
        `adapter.context_length`.
    allow_exog : bool
        Whether the underlying adapter supports exogenous variables.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : pandas DateOffset, int
        Frequency of the index of the input used in training. A pandas
        DateOffset for DatetimeIndex or an int step for RangeIndex.
    context_range_ : dict[str, pandas Index]
        First and last values of the index of the data used during training
        for each series.
    series_names_in_ : list
        Names of the series (levels) provided by the user during training.
    is_multiple_series_ : bool
        Whether the model was fitted with multiple series.
    exog_in_ : bool
        If the model has been trained using exogenous variable/s.
    exog_names_in_ : list
        Names of the exogenous variables used during training. `None` if no
        exog was provided.
    exog_names_in_per_series_ : dict
        Names of the exogenous variables used during training for each series.
        `None` if no exog was provided.
    exog_type_in_ : type
        Type of exogenous variable/s used in training. `None` if no exog
        was provided.
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
    be installed, other foundation-model backends remain optional.

    """

    def __init__(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> None:

        adapter_cls                    = _resolve_adapter(model_id)
        self.adapter                   = adapter_cls(model_id=model_id, **kwargs)
        self.index_type_               = None
        self.index_freq_               = None
        self.context_range_            = None
        self.series_names_in_          = None
        self.is_multiple_series_       = False
        self.exog_in_                  = False
        self.exog_names_in_            = None
        self.exog_names_in_per_series_ = None
        self.exog_type_in_             = None
        self.creation_date             = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.fit_date                  = None
        self.skforecast_version        = __version__
        self.python_version            = sys.version.split(" ")[0]

    @property
    def model_id(self) -> str:
        """
        HuggingFace model ID.

        Returns
        -------
        model_id : str
            HuggingFace model ID. Mirrors `adapter.model_id`.
        """
        return self.adapter.model_id

    @property
    def context_(self) -> dict[str, pd.Series]:
        """
        Context stored during `fit`, used as default context for `predict` if no
        override is provided.

        Returns
        -------
        context_ : dict[str, pd.Series]
            Per-series dict of pandas Series containing the last `context_length`
            observations from the training data, stored during `fit`. Mirrors
            `adapter.context_`.
        """
        return self.adapter.context_

    @property
    def context_exog_(self) -> dict[str, pd.DataFrame] | None:
        """
        Context stored during `fit`, used as default context for `predict` if no
        override is provided.

        Returns
        -------
        context_exog_ : dict[str, pd.DataFrame], None
            Per-series dict of pandas DataFrame containing the last
            `context_length` exog variables from the training data, stored
            during `fit`. `None` if the adapter does not support exogenous
            variables or no exog was provided. Mirrors `adapter.context_exog_`.
        """
        return self.adapter.context_exog_

    @property
    def context_length(self) -> int:
        """
        Maximum number of historical observations used as context.

        Returns
        -------
        context_length : int
            Maximum context length. Mirrors `adapter.context_length`.
        """
        return self.adapter.context_length

    @property
    def allow_exog(self) -> bool:
        """
        Whether the underlying adapter supports exogenous variables.

        Returns
        -------
        allow_exog : bool
            `True` if the adapter accepts and uses `exog`; `False` if it
            ignores covariates (e.g. TimesFM 2.5, Moirai-2).
        """
        return self.adapter.allow_exog

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
    
    def _check_preprocess_context(
        self,
        series: pd.Series | pd.DataFrame | dict[str, pd.Series],
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> tuple[dict[str, pd.Series], dict[str, pd.Index], list[str], dict[str, pd.DataFrame | pd.Series | None], list[str]]:
        """
        Normalize and validate context input to a per-series dict.

        Parameters
        ----------
        series : pandas Series, pandas DataFrame, dict
            Time series to normalize and validate.

            - If `pandas Series`: single-series mode.
            - If wide `pandas DataFrame` or `dict[str, pandas Series]`:
            multi-series mode.
        exog : pandas Series, pandas DataFrame, dict, default None
            Exogenous variables aligned to `series`.

            - If `pandas Series` or `pandas DataFrame`: broadcast to all
            series.
            - If `dict`: per-series exogenous variables.

        Returns
        -------
        context : dict
            Per-series dict of pandas Series, trimmed to the last
            `context_length` observations.
        series_indexes : dict
            Index of each series before trimming.
        series_names_in_ : list
            Names of the series.
        context_exog : dict or None
            Per-series dict of exogenous DataFrames trimmed to the last
            `context_length` observations. `None` if `exog` is `None`.
        exog_names_in_ : list or None
            Names of the exogenous variables. `None` if `exog` is `None`.

        """

        series_dict, series_indexes = check_preprocess_series_foundation(series)
        series_names_in_ = list(series_dict.keys())

        if exog is not None:
            exog_dict, exog_names_in_ = check_preprocess_exog_multiseries(
                series_names_in_  = series_names_in_,
                series_index_type = type(series_indexes[series_names_in_[0]]),
                exog              = exog,
                exog_dict         = {name: None for name in series_names_in_},
            )

            # NOTE: As no trim is applied to the series, it is only needed to 
            # align exog.
            series_dict, exog_dict = align_series_and_exog_multiseries(
                                         series_dict      = series_dict,
                                         exog_dict        = exog_dict,
                                         trim_series_nan  = False,
                                     )

        context = {
            name: s.iloc[-self.context_length :]
            for name, s in series_dict.items()
        }
        if exog is not None:
            context_exog = {
                name: (
                    e.iloc[-self.context_length :]
                    if e is not None
                    else None
                )
                for name, e in exog_dict.items()
            }
        else:
            context_exog = None
            exog_names_in_ = None
                                 
        return context, series_indexes, series_names_in_, context_exog, exog_names_in_

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
        self.context_range_             = None
        self.series_names_in_           = None
        self.is_multiple_series_        = False
        self.exog_in_                   = False
        self.exog_names_in_             = None
        self.exog_names_in_per_series_  = None
        self.exog_type_in_              = None
        self.fit_date                   = None
        
        context, series_indexes, series_names_in_, context_exog, exog_names_in_ = (
            self._check_preprocess_context(
                series=series,
                exog=exog,
            )
        )

        self.adapter.fit(
            context      = context,
            context_exog = context_exog,
        )

        self.series_names_in_    = series_names_in_
        self.is_multiple_series_ = len(series_names_in_) > 1

        if context_exog is not None and len(exog_names_in_) > 0:
            self.exog_in_ = True
            self.exog_names_in_ = exog_names_in_
            self.exog_names_in_per_series_ = {
                k: list(v.columns) if v is not None else None
                for k, v in context_exog.items()
            }
            self.exog_type_in_ = type(exog)

        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.context_range_ = {k: v[[0, -1]] for k, v in series_indexes.items()}
        self.index_type_ = type(series_indexes[series_names_in_[0]])
        if isinstance(series_indexes[series_names_in_[0]], pd.DatetimeIndex):
            self.index_freq_ = series_indexes[series_names_in_[0]].freq
        else:
            self.index_freq_ = series_indexes[series_names_in_[0]].step

        return self

    @staticmethod
    def _exog_to_dict(
        exog: pd.Series | pd.DataFrame | dict[str, pd.DataFrame | pd.Series | None],
        series_names_in: list[str],
    ) -> dict[str, pd.DataFrame | pd.Series | None]:
        """
        Normalize any supported exog format into a per-series dict.

        Parameters
        ----------
        exog : pandas Series, pandas DataFrame, dict
            Future exogenous variables in any supported format.

            - If `pandas Series` (flat index): broadcast to all series.
            - If `pandas Series` (MultiIndex): converted to dict, then
              keyed per series.
            - If `pandas DataFrame` (flat index): broadcast to all series.
            - If `pandas DataFrame` (MultiIndex / long-format): converted
              to dict per series ID.
            - If `dict`: used directly, missing series keys filled as
              `None`.
        series_names_in : list[str]
            Series names that define the output dict keys.

        Returns
        -------
        exog_dict : dict
            Per-series dict with exactly the keys in `series_names_in`.

        """

        if isinstance(exog, dict):
            return {name: exog.get(name, None) for name in series_names_in}

        if isinstance(exog, pd.Series):
            if isinstance(exog.index, pd.MultiIndex):
                exog = exog.to_frame()
            else:
                return {name: exog for name in series_names_in}

        # At this point exog is always a DataFrame (original or coerced)
        if isinstance(exog.index, pd.MultiIndex):
            if not isinstance(exog.index.levels[1], pd.DatetimeIndex):
                raise TypeError(
                    "The second level of the MultiIndex in `exog` must be a "
                    "pandas DatetimeIndex. "
                    f"Found {type(exog.index.levels[1])}."
                )
            per_series = {
                sid: group.droplevel(0)
                for sid, group in exog.groupby(level=0, sort=False)
            }
            warnings.warn(
                "Passing a long-format DataFrame as `exog` requires "
                "additional internal transformations, which can increase "
                "computational time. It is recommended to use a dictionary "
                "of pandas Series or DataFrames instead.",
                InputTypeWarning,
                stacklevel=3,
            )
            return {name: per_series.get(name, None) for name in series_names_in}

        return {name: exog for name in series_names_in}

    def _prepare_future_exog(
        self,
        steps: int,
        context: dict[str, pd.Series],
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ),
        series_names_in: list[str],
    ) -> dict[str, pd.DataFrame | None]:
        """
        Normalize, broadcast, and align future exogenous variables to the
        forecast horizon in a single pass.

        Performs the full pipeline for future exog:

        1. **Type coercion**: long-format MultiIndex Series/DataFrame is
        converted to a dict keyed by series ID.
        2. **Broadcast / dict normalisation**: flat Series or DataFrame is
        broadcast to every series; a dict is filled with `None` for
        missing keys; `None` input produces an all-None dict.
        3. **Temporal alignment**: each per-series exog is aligned to the
        forecast horizon using the resolved context. For `DatetimeIndex`
        data, exog is reindexed to the exact expected range (NaN-filling
        gaps). For other index types a length check and optional
        `RangeIndex` start verification are applied.

        This function is self-contained — it does not depend on any
        metadata stored at `fit` time. Alignment is driven entirely by the
        context that will be used for prediction.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict[str, pandas Series]
            Per-series resolved context. Each value is a pandas Series whose
            index provides the reference end-point and frequency for
            alignment.
        exog : pandas Series, pandas DataFrame, dict, default None
            Future exogenous variables in any supported format.

            - If `None`: returns `{name: None ...}` for every series.
            - If `pandas Series` (flat index): broadcast to all series.
            - If `pandas Series` (MultiIndex): converted to dict, then
            keyed per series.
            - If `pandas DataFrame` (flat index): broadcast to all series.
            - If `pandas DataFrame` (MultiIndex / long-format): converted
            to dict per series ID.
            - If `dict`: used directly, missing series keys filled as
            `None`.
        series_names_in : list[str]
            Series names that define the output dict keys.

        Returns
        -------
        exog_aligned : dict
            Per-series dict with exactly the keys in `series_names_in`. Each
            non-None value is a pandas DataFrame with exactly `steps` rows
            aligned to the forecast horizon. Series inputs are coerced to
            single-column DataFrames.

        Raises
        ------
        TypeError
            If `exog` is a long-format DataFrame whose second MultiIndex
            level is not a `DatetimeIndex`, or if `exog` is an unsupported
            type.
        ValueError
            If a non-DatetimeIndex exog has fewer than `steps` rows, or if
            a `RangeIndex` exog does not start at the expected position.

        """

        # Early return: no exog provided
        if exog is None:
            return {name: None for name in series_names_in}

        # Type guard
        if not isinstance(exog, (pd.Series, pd.DataFrame, dict)):
            raise TypeError(
                f"`exog` must be a pandas Series, DataFrame, dict, or None. "
                f"Got {type(exog)}."
            )

        # Normalize any input format (Series, DataFrame, dict) into a
        # per-series dict keyed by series name.
        exog_dict = self._exog_to_dict(exog, series_names_in)

        # Determine index type and freq once from the first non-empty
        # context series. All series share the same type and freq
        # (guaranteed by check_preprocess_series upstream).
        first_ctx = next(
            (ctx for ctx in context.values() if ctx is not None and len(ctx) > 0),
            None,
        )
        if first_ctx is not None:
            is_datetime_ctx = isinstance(first_ctx.index, pd.DatetimeIndex)
            if is_datetime_ctx:
                freq = first_ctx.index.freq
            elif isinstance(first_ctx.index, pd.RangeIndex):
                freq = first_ctx.index.step
            else:
                freq = None
        else:
            is_datetime_ctx = False
            freq = None

        # Align each series' exog to its forecast horizon
        exog_aligned = {}
        nan_filled_series = []
        for name in series_names_in:
            e = exog_dict.get(name)
            if e is None:
                exog_aligned[name] = None
                continue

            if isinstance(e, pd.Series):
                e = e.to_frame()

            # No context available for this series: keep exog as-is
            ctx = context.get(name)
            if ctx is None or len(ctx) == 0:
                exog_aligned[name] = e
                continue

            ref_end = ctx.index[-1]
            label = f"`exog` for series '{name}'"

            # DatetimeIndex: reindex to the exact expected date range,
            # filling gaps with NaN.
            if is_datetime_ctx and freq is not None and isinstance(e.index, pd.DatetimeIndex):
                expected_idx = pd.date_range(
                    start=ref_end + freq, periods=steps, freq=freq
                )
                e_aligned = e.reindex(expected_idx)
                if e_aligned.isnull().any(axis=None):
                    nan_filled_series.append(name)
                exog_aligned[name] = e_aligned
            else:
                # RangeIndex / other: length check + optional start validation,
                # then truncate to the forecast horizon.
                if len(e) < steps:
                    raise ValueError(
                        f"{label} must have at least {steps} values. "
                        f"Got {len(e)}."
                    )
                if isinstance(e.index, pd.RangeIndex) and freq is not None:
                    expected_start = ref_end + freq
                    if e.index[0] != expected_start:
                        raise ValueError(
                            f"To make predictions {label} must start one step "
                            f"ahead of `context`.\n"
                            f"    `context` ends at: {ref_end}.\n"
                            f"    {label} starts at: {e.index[0]}.\n"
                            f"    Expected index: {expected_start}."
                        )
                exog_aligned[name] = e.iloc[:steps]

        # Batch warning for all series whose exog had missing timestamps
        if nan_filled_series:
            warnings.warn(
                f"`exog` for series {nan_filled_series} has been reindexed "
                f"to match the expected forecast horizon. Missing timestamps "
                f"were filled with NaN.",
                MissingValuesWarning,
            )

        return exog_aligned

    def predict(
        self,
        steps: int,
        context: pd.Series | pd.DataFrame | dict[str, pd.Series] | None = None,
        context_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
        quantiles: list[float] | tuple[float] | None = None,
        check_inputs: bool = True,
    ) -> pd.DataFrame:
        """
        Predict n steps ahead.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : pandas Series, pandas DataFrame, dict, default None
            Override the stored context with this window.

            - If `pandas Series`: single-series override.
            - If wide `pandas DataFrame` or `dict[str, pandas Series]`:
            multi-series override.
        context_exog : pandas Series, pandas DataFrame, dict, default None
            Historical exog corresponding to `context`.
        exog : pandas Series, pandas DataFrame, dict, default None
            Future known exogenous variables for the forecast horizon.

            - If `pandas Series` or `pandas DataFrame`: broadcast to all
            series.
            - If `dict`: per-series exogenous variables.
        quantiles : list, tuple, default None
            Quantile levels to return, e.g. `[0.1, 0.5, 0.9]`. If `None`,
            returns a point forecast (median).
        check_inputs : bool, default True
            If `True`, the `context` and `context_exog` inputs are validated
            and normalized via `_check_preprocess_context`. If `False`,
            `context` must already be a `dict[str, pandas Series]` and
            `context_exog` must be a `dict[str, pandas DataFrame | None]`
            or `None`. This argument is created for internal use and is not
            recommended to be changed.

        Returns
        -------
        predictions : pandas DataFrame
            Value of predictions. The DataFrame includes the following columns:

            - level: Name of the series.
            - pred: Predicted values (point forecast, median).

            If `quantiles` is not `None`, the `pred` column is replaced by
            one column per quantile level (e.g., `q_0.1`, `q_0.5`, `q_0.9`).

        Notes
        -----
        Foundation models are pre-trained and do not learn from the data passed 
        to `fit`. The `fit` method only stores context (the last `context_length` 
        observations) and metadata. This leads to four distinct behaviors 
        depending on the combination of `is_fitted` and `context`:

        - **Not fitted, `context=None`**: raises `ValueError`. There is no context 
        available for prediction.
        - **Fitted, `context=None`**: uses the context and `context_exog_` stored 
        during `fit`. If the user supplies `context_exog`, it is ignored with a 
        warning.
        - **Not fitted, `context` provided (zero-shot mode)**: The model uses 
        `context` and `context_exog` (if provided) as context for prediction.
        - **Fitted, `context` provided**: Stored context is ignored, the 
        provided `context` and `context_exog` (if provided) are used for 
        prediction.

        """

        if not self.is_fitted and context is None:
            raise ValueError(
                "Call `fit` before `predict`, or pass `context`."
            )

        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("`steps` must be a positive integer.")

        if quantiles is not None:
            for q in quantiles:
                if not 0.0 <= q <= 1.0:
                    raise ValueError(
                        f"All quantiles must be between 0 and 1. Got {q}."
                    )
        
        # Context (past data)
        if context is None:
            if context_exog is not None:
                warnings.warn(
                    "`context_exog` is ignored when `context` is not provided. "
                    "The stored `context_exog_` from `fit` is used instead.",
                    IgnoredArgumentWarning,
                )
            context = self.adapter.context_
            series_names_in = self.series_names_in_
            context_exog = self.adapter.context_exog_
        elif check_inputs:
            context, _, series_names_in, context_exog, _ = self._check_preprocess_context(
                series = context,
                exog   = context_exog,
            )
        else:
            series_names_in = list(context.keys())

        # Future exog
        if not self.allow_exog:
            has_exog = (exog is not None) or (context_exog is not None)
            if has_exog:
                warnings.warn(
                    f"{type(self.adapter).__name__} does not currently "
                    "support covariates. `exog` and `context_exog` "
                    "are ignored.",
                    IgnoredArgumentWarning,
                )
                exog = None
                context_exog = None
        else:
            if check_inputs:
                exog = self._prepare_future_exog(
                           steps           = steps,
                           context         = context,
                           exog            = exog,
                           series_names_in = series_names_in,
                       )

        # Adapter returns dict[str, np.ndarray] with shape (steps, n_q)
        raw_predictions = self.adapter.predict(
                              steps        = steps,
                              context      = context,
                              context_exog = context_exog,
                              exog         = exog,
                              quantiles    = quantiles,
                          )

        # Build long-format DataFrame from raw predictions
        n_series = len(series_names_in)
        per_series_indices = [
            expand_index(context[name].index, steps=steps)
            for name in series_names_in
        ]

        if n_series == 1:
            long_index = per_series_indices[0]
            level_col = np.repeat(series_names_in, steps)
        else:
            idx_matrix = np.empty(
                (steps, n_series), dtype=per_series_indices[0].dtype
            )
            for i, idx in enumerate(per_series_indices):
                idx_matrix[:, i] = idx
            long_index = idx_matrix.ravel()
            level_col = np.tile(series_names_in, steps)

        col_names = ["pred"] if quantiles is None else [f"q_{q}" for q in quantiles]
        predictions: dict[str, np.ndarray] = {"level": level_col}
        for j, col in enumerate(col_names):
            pred_arr = np.empty((steps, n_series), dtype=np.float64)
            for i, name in enumerate(series_names_in):
                pred_arr[:, i] = raw_predictions[name][:, j]
            predictions[col] = pred_arr.ravel()

        predictions = pd.DataFrame(predictions, index=long_index)

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

        return self.adapter.get_params()

    def set_params(self, **params) -> FoundationModel:
        """
        Set parameters for this estimator (sklearn-compatible).

        After calling this method, the FoundationModel is reset to an unfitted state.

        Parameters
        ----------
        **params :
            Estimator parameters forwarded to the underlying adapter's
            `set_params`. Use `model_id` to change the model ID. All
            other keys are adapter-specific.

        Returns
        -------
        self : FoundationModel
            The same object with updated parameters.

        """

        try:
            self.adapter.set_params(**params)
        except ValueError as exc:
            raise ValueError(
                str(exc).replace(type(self.adapter).__name__, "FoundationModel")
            ) from exc

        self.index_type_               = None
        self.index_freq_               = None
        self.context_range_            = None
        self.series_names_in_          = None
        self.is_multiple_series_       = False
        self.exog_in_                  = False
        self.exog_names_in_            = None
        self.exog_names_in_per_series_ = None
        self.exog_type_in_             = None
        self.fit_date                  = None
        self.adapter.is_fitted         = False

        return self
