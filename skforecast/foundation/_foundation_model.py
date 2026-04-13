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
    context_exog_ : dict[str, pandas DataFrame]
        Per-series dict of pandas DataFrame containing the last `context_length`
        exog variables from the training data, stored during `fit`. Mirrors
        `adapter.context_exog_`.
    context_length : int
        Maximum number of historical observations used as context. Mirrors
        `adapter.context_length`.
    allow_exog : bool
        Whether the underlying adapter supports exogenous variables.
    index_type_ : type
        Type of index of the input used in training.
    index_freq_ : str
        Frequency of Index of the input used in training.
    context_range_ : dict[str, tuple]
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
        self.exog_in_                  = False
        self.exog_names_in_            = None
        self.exog_names_in_per_series_ = None
        self.is_multiple_series_       = False
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
    def context_exog_(self) -> dict[str, pd.DataFrame]:
        """
        Context stored during `fit`, used as default context for `predict` if no
        override is provided.

        Returns
        -------
        context_exog_ : dict[str, pd.DataFrame]
            Per-series dict of pandas DataFrame containing the last `context_length`
            exog variables from the training data, stored during `fit`. Mirrors
            `adapter.context_exog_`.
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
        context : pandas Series, pandas DataFrame, dict
            Context override passed to predict.

            - If `pandas Series`: single-series override.
            - If wide `pandas DataFrame` or `dict[str, pandas Series]`:
            multi-series override.
        
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
            context_exog = {name: None for name in series_names_in_}
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
        self.exog_in_                   = False
        self.exog_names_in_             = None
        self.exog_names_in_per_series_  = None
        self.is_multiple_series_        = False
        self.fit_date                   = None
        
        context, series_indexes, series_names_in_, context_exog, exog_names_in_ = self._check_preprocess_context(
            series = series,
            exog   = exog,
        )

        self.adapter.fit(
            context      = context,
            context_exog = context_exog,
        )

        self.series_names_in_    = series_names_in_
        self.is_multiple_series_ = len(series_names_in_) > 1

        self.exog_names_in_per_series_ = {
            k: list(v.columns) if v is not None else None
            for k, v in context_exog.items()
        }
        if exog is not None:
            self.exog_in_ = len(exog_names_in_) > 0
            self.exog_names_in_ = exog_names_in_

        self.fit_date = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
        self.context_range_ = {k: v[[0, -1]] for k, v in series_indexes.items()}
        self.index_type_ = type(series_indexes[series_names_in_[0]])
        if isinstance(series_indexes[series_names_in_[0]], pd.DatetimeIndex):
            self.index_freq_ = series_indexes[series_names_in_[0]].freq
        else:
            self.index_freq_ = series_indexes[series_names_in_[0]].step

        return self

    # TODO: se tiene que verificar que para cada serie las exog the futuro son 
    # las mismas que las del pasado. Si no lo son, rellenar con NaNs.
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
        steps : int
            Number of steps ahead to forecast.
        context : dict[str, pd.Series]
            Per-series resolved context. Each value is a pandas Series whose
            index provides the reference end-point and frequency for
            alignment.

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

        if exog is None:
            return {name: None for name in series_names_in}

        if not isinstance(exog, (pd.Series, pd.DataFrame, dict)):
            raise TypeError(
                f"`exog` must be a pandas Series, DataFrame, dict, or None. "
                f"Got {type(exog)}."
            )

        if isinstance(exog, dict):
            exog_dict = {name: exog.get(name, None) for name in series_names_in}

        elif isinstance(exog, pd.Series):
            if isinstance(exog.index, pd.MultiIndex):
                exog = exog.to_frame()
                # fall through to DataFrame path
            else:
                exog_dict = {name: exog for name in series_names_in}

        if isinstance(exog, pd.DataFrame):
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
                    stacklevel=2,
                )
                exog_dict = {name: per_series.get(name, None) for name in series_names_in}
            else:
                exog_dict = {name: exog for name in series_names_in}
        
        exog_aligned = {}
        for name in series_names_in:
            e = exog_dict.get(name)
            if e is None:
                exog_aligned[name] = None
                continue

            # Coerce Series to DataFrame for uniform downstream handling
            if isinstance(e, pd.Series):
                e = e.to_frame()

            ctx = context.get(name)
            if ctx is None or len(ctx) == 0:
                exog_aligned[name] = e
                continue

            ref_end = ctx.index[-1]
            if isinstance(ctx.index, pd.DatetimeIndex):
                freq = ctx.index.freq
            elif isinstance(ctx.index, pd.RangeIndex):
                freq = ctx.index.step
            else:
                freq = None

            label = f"`exog` for series '{name}'"

            if isinstance(e.index, pd.DatetimeIndex) and freq is not None:
                expected_idx = pd.date_range(
                    start=ref_end + freq, periods=steps, freq=freq
                )
                e_aligned = e.reindex(expected_idx)
                has_nans = e_aligned.isnull().any()
                if isinstance(e_aligned, pd.DataFrame):
                    has_nans = has_nans.any()
                if has_nans:
                    warnings.warn(
                        f"{label} has been reindexed to match the expected "
                        f"forecast horizon ({expected_idx[0]} — {expected_idx[-1]}). "
                        f"Missing timestamps were filled with NaN.",
                        MissingValuesWarning,
                    )
                exog_aligned[name] = e_aligned
            else:
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
    ) -> pd.DataFrame:
        """
        Predict n steps ahead.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : pandas Series, pandas DataFrame, dict, default None
            Override the stored history with this window.

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

        Returns
        -------
        predictions : pandas DataFrame
            Always a long-format DataFrame. Point forecast: columns
            `["level", "pred"]`. Quantile forecast: columns
            `["level", "q_0.1", "q_0.5", ...]`. The index repeats each
            forecast timestamp once per series.

        Notes
        -----
        Foundation models are pre-trained and do not learn from the data
        passed to `fit`. The `fit` method only stores context (the last
        `context_length` observations) and metadata. This leads to four
        distinct behaviors depending on the combination of `is_fitted`
        and `context`:

        - **Not fitted, `context=None`**: raises `ValueError`. There
        is no context available for prediction.
        - **Fitted, `context=None`**: uses the context stored during
        `fit` (`adapter.context_`). Exogenous variables are validated
        against the metadata recorded at fit time.
        - **Not fitted, `context` provided**: pure zero-shot mode.
        The model uses `context` as context without any validation
        against fit metadata (there is none).
        - **Fitted, `context` provided**: `context` fully
        overrides the stored context. No validation against fit
        metadata is performed because the pre-trained model did not
        learn from the `fit` data; the user-supplied context is
        accepted as-is.

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
            context = self.adapter.context_
            series_names_in = self.series_names_in_
            context_exog = self.adapter.context_exog_
        else:
            context, _, series_names_in, context_exog, _ = self._check_preprocess_context(
                series = context,
                exog   = context_exog,
            )

        # Future exog
        if not self.allow_exog:
            has_exog = (
                (exog is not None)
                or (context_exog is not None)
            )
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
        per_series_indices = [
            expand_index(context[name].index, steps=steps)
            for name in series_names_in
        ]

        if len(series_names_in) == 1:
            long_index = per_series_indices[0]
            level_col = np.repeat(series_names_in, steps)
        else:
            long_index = np.column_stack(
                [np.asarray(idx) for idx in per_series_indices]
            ).ravel()
            level_col = np.tile(series_names_in, steps)

        if quantiles is None:
            # Point forecast (median): single "pred" column
            pred_matrix = np.column_stack([
                raw_predictions[name][:, 0] for name in series_names_in
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
                    raw_predictions[name][:, j] for name in series_names_in
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

        return self.adapter.get_params()

    def set_params(self, **params) -> FoundationModel:
        """
        Set parameters for this estimator (sklearn-compatible).

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
        
        return self
