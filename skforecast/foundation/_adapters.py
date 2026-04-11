################################################################################
#                         Foundation Model Adapters                            #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8
# Each adapter imports its own backend library lazily (i.e. inside the method
# that first needs it) rather than at module level. This means that only the
# library required by the adapter you actually use needs to be installed, other
# foundation-model backends remain optional.

from __future__ import annotations
from typing import Any
import warnings
import numpy as np
import pandas as pd
from ..exceptions import IgnoredArgumentWarning
from ..utils import expand_index


class Chronos2Adapter:
    """
    Adapter for Amazon Chronos-2 foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "autogluon/chronos-2-small".
    context_length : int, default 2048
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `last_window` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Defaults to 2048, which matches the context window of
        all current Chronos-2 variants. Must be a positive integer.
    pipeline : BaseChronosPipeline, optional
        Pre-loaded pipeline instance. If None, the pipeline is loaded lazily on
        the first call to `predict`.
    device_map : str, optional
        Device map string forwarded to `BaseChronosPipeline.from_pretrained`
        (e.g. "cuda", "cpu").
    torch_dtype : optional
        Torch dtype forwarded to `BaseChronosPipeline.from_pretrained`.
    predict_kwargs : dict, optional
        Additional keyword arguments forwarded to the pipeline's
        `predict_quantiles` method.
    
    """

    # TODO: DEBERÍA ESTAR en el init
    allow_exogenous: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        pipeline: Any | None = None,
        context_length: int = 2048,
        predict_kwargs: dict[str, Any] | None = None,
        device_map: str | None = None,
        torch_dtype: Any | None = None,
        cross_learning: bool = False,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. "autogluon/chronos-2-small".
        pipeline : BaseChronosPipeline, optional
            Pre-loaded pipeline instance. If `None`, the pipeline is
            loaded lazily on the first call to `predict`.
        context_length : int, default 2048
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if
            `last_window` is longer than `context_length` it is trimmed to
            this length before inference; if it is shorter, all available
            observations are passed as-is and the model handles reduced
            context gracefully. Defaults to 2048, which matches the context
            window of all current Chronos-2 variants. Must be a positive
            integer.
        predict_kwargs : dict, optional
            Additional keyword arguments forwarded verbatim to the
            pipeline's `predict_quantiles` method.
        device_map : str, optional
            Device map string forwarded to
            `BaseChronosPipeline.from_pretrained` (e.g. "cuda",
            "cpu", "auto").
        torch_dtype : optional
            Torch dtype forwarded to `BaseChronosPipeline.from_pretrained`
            (e.g. `torch.bfloat16`).
        cross_learning : bool, default False
            If `True`, Chronos-2 shares information across all series in
            the batch when predicting in multi-series mode. Forwarded
            directly to `predict_quantiles`. Ignored in single-series mode.
        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )

        self.model_id            = model_id
        self._pipeline           = pipeline
        self._history            = None
        self._history_exog       = None
        self.context_length      = context_length
        self.predict_kwargs      = predict_kwargs or {}
        self.device_map          = device_map
        self.torch_dtype         = torch_dtype
        self.cross_learning      = cross_learning
        self.is_fitted           = False
        self.is_multiple_series_ = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        dict
            Keys: `model_id`, `cross_learning`, `context_length`,
            `device_map`, `torch_dtype`, `predict_kwargs`.
        """
        return {
            'model_id':       self.model_id,
            'cross_learning': self.cross_learning,
            'context_length': self.context_length,
            'device_map':     self.device_map,
            'torch_dtype':    self.torch_dtype,
            'predict_kwargs': self.predict_kwargs or None,
        }

    def set_params(self, **params) -> Chronos2Adapter:
        """
        Set adapter parameters. Resets the pipeline when a device or dtype
        param changes, since those are baked into the loaded pipeline.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `cross_learning`, `context_length`,
            `device_map`, `torch_dtype`, `predict_kwargs`.

        Returns
        -------
        self : Chronos2Adapter
        """
        valid = {
            'model_id', 'cross_learning', 'context_length',
            'device_map', 'torch_dtype', 'predict_kwargs',
        }
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for Chronos2Adapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )
        pipeline_reset_keys = {'model_id', 'device_map', 'torch_dtype'}
        if params.keys() & pipeline_reset_keys:
            self._pipeline = None
        for key, value in params.items():
            if key == 'predict_kwargs':
                self.predict_kwargs = value or {}
            elif key == 'context_length':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                self.context_length = value
            else:
                setattr(self, key, value)
        return self

    def _load_pipeline(self) -> None:
        """
        Load the Chronos-2 pipeline into `self._pipeline` if not already set.

        The pipeline is imported lazily from `chronos` and instantiated via
        `BaseChronosPipeline.from_pretrained`, which auto-dispatches to the
        correct pipeline class based on the model config. Optional
        `device_map` and `torch_dtype` stored at initialisation are
        forwarded to the constructor. This method is a no-op when
        `self._pipeline` is already populated.

        Raises
        ------
        ImportError
            If `chronos-forecasting` >=2.0 is not installed.
        """

        if self._pipeline is not None:
            return
        try:
            from chronos import BaseChronosPipeline
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting >=2.0 is required. "
                "Install it with `pip install chronos-forecasting`."
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        self._pipeline = BaseChronosPipeline.from_pretrained(self.model_id, **kwargs)

    @staticmethod
    def _to_covariate_array(col_data: Any) -> np.ndarray:
        """
        Convert a covariate column to a numpy array.

        Numeric columns (int, float) and boolean columns are cast to
        `float64`. All other dtypes (object, string, Categorical) are left
        as-is so that Chronos-2 can handle them as categorical covariates
        natively.

        Parameters
        ----------
        col_data : array-like
            A single covariate column (e.g. a pandas Series or 1-D array).

        Returns
        -------
        np.ndarray
            A 1-D numpy array. Numeric/bool → `float64`. Others → original
            dtype (typically `object` for string and categorical data).
        """

        # Handle pandas Series first to correctly process nullable extension
        # dtypes (pd.Int64Dtype, pd.Float64Dtype, pd.BooleanDtype): np.asarray()
        # on those produces dtype=object with pd.NA sentinels instead of float64.
        if isinstance(col_data, pd.Series):
            if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
                return col_data.astype(np.float64).to_numpy()
            return col_data.to_numpy()

        # Fallback for numpy arrays, lists, etc.
        arr = np.asarray(col_data)
        if arr.dtype.kind in ("i", "u", "f", "b"):  # integer, unsigned int, float, bool
            return arr.astype(np.float64)
        return arr

    def _build_chronos_input(
        self,
        target: np.ndarray,
        past_exog: pd.DataFrame | pd.Series | None = None,
        future_exog: pd.DataFrame | pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Build the input dict consumed by the pipeline's `predict_quantiles` method.

        Parameters
        ----------
        target : np.ndarray
            1-D array of observed time series values used as context. Must be
            castable to `float64`.
        past_exog : pd.DataFrame or pd.Series, optional
            Historical exogenous variables whose index is aligned to
            `target`. Each column (or the single Series, referenced by
            its name) becomes an entry in the returned
            "past_covariates" dict. Numeric and boolean columns are
            cast to `float64`; string and categorical columns are passed
            as-is and handled natively by Chronos-2.
        future_exog : pd.DataFrame or pd.Series, optional
            Future-known exogenous variables covering the forecast horizon.
            Must have exactly `prediction_length` rows. Each column
            becomes an entry in the returned "future_covariates" dict.
            Numeric and boolean columns are cast to `float64`; string and
            categorical columns are passed as-is.

        Returns
        -------
        dict
            Dictionary with mandatory key "target" (1-D `float64`
            `np.ndarray`) and optional keys "past_covariates" and
            "future_covariates", each mapping column names to 1-D
            arrays (`float64` for numeric/bool columns, `object` dtype
            for string/categorical columns).
        """

        input_dict = {"target": np.asarray(target, dtype=float)}
        if past_exog is not None:
            df = (
                past_exog
                if isinstance(past_exog, pd.DataFrame)
                else past_exog.to_frame()
            )
            input_dict["past_covariates"] = {
                col: Chronos2Adapter._to_covariate_array(df[col]) for col in df.columns
            }
        if future_exog is not None:
            df = (
                future_exog
                if isinstance(future_exog, pd.DataFrame)
                else future_exog.to_frame()
            )
            input_dict["future_covariates"] = {
                col: Chronos2Adapter._to_covariate_array(df[col]) for col in df.columns
            }
        return input_dict

    def fit(
        self,
        series_dict: dict[str, pd.Series],
        exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        is_multiple_series: bool,
    ) -> Chronos2Adapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since Chronos-2 is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        series_dict : dict of str → pd.Series
            Normalized training series, one entry per series.
        exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series historical exogenous variables (past covariates).
        is_multiple_series : bool
            `True` when multiple series are provided.

        Returns
        -------
        self : Chronos2Adapter

        """

        self.is_fitted = False
        self.is_multiple_series_ = False

        self._history = {
            name: s.iloc[-self.context_length :].copy()
            for name, s in series_dict.items()
        }

        if exog_dict is not None:
            self._history_exog = {
                name: (
                    e.iloc[-self.context_length :].copy()
                    if e is not None
                    else None
                )
                for name, e in exog_dict.items()
            }
        
        self.is_fitted = True
        self.is_multiple_series_ = is_multiple_series

        return self

    def _predict_multiseries(
        self,
        steps: int,
        history_dict: dict[str, pd.Series],
        past_exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        future_exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        quantiles: list[float] | None,
    ) -> pd.DataFrame:
        """
        Internal multi-series prediction logic.

        All dicts are already resolved and trimmed to `context_length`
        by the caller (`predict`).

        Parameters
        ----------
        steps : int
            Forecast horizon.
        history_dict : dict of str → pd.Series
            Per-series context windows (already trimmed).
        past_exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series past covariates (already trimmed).
        future_exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series future covariates.
        quantiles : list of float or None
            Quantile levels to return. If `None`, a point forecast
            (median) is produced.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame.
        """

        quantile_levels = list(quantiles) if quantiles is not None else [0.5]
        series_names = list(history_dict.keys())

        # Build list of per-series input dicts for the pipeline
        inputs_list = [
            self._build_chronos_input(
                target      = history_dict[name].to_numpy(),
                past_exog   = past_exog_dict[name],
                future_exog = future_exog_dict[name],
            )
            for name in series_names
        ]

        quantile_preds, _ = self._pipeline.predict_quantiles(
            inputs            = inputs_list,
            prediction_length = steps,
            quantile_levels   = quantile_levels,
            cross_learning    = self.cross_learning,
            **self.predict_kwargs,
        )

        # Decode all per-series quantile arrays once
        # Each decoded[i] has shape (steps, n_q)
        decoded: list[np.ndarray] = []
        for i in range(len(series_names)):
            q_arr = quantile_preds[i]
            if hasattr(q_arr, "detach"):
                q_arr = q_arr.detach().cpu().numpy()
            else:
                q_arr = np.asarray(q_arr)
            decoded.append(q_arr.squeeze(0))

        # Build shared long-format index and level column
        #  index = [t0_s0, t0_s1, t1_s0, t1_s1, …] — per-series future timestamps
        #  level = [s0, s1, s0, s1, …] (series names tiled for each step)
        n_series = len(series_names)
        per_series_forecast_indices = [
            expand_index(history_dict[name].index, steps=steps)
            for name in series_names
        ]
        long_index = np.array([
            per_series_forecast_indices[j][i]
            for i in range(steps)
            for j in range(n_series)
        ])
        level_col = np.tile(series_names, steps)

        if quantiles is None:
            # point forecast: long DataFrame with columns [level, pred]
            # quantile_levels == [0.5], so the median is always at index 0
            point_matrix = np.column_stack([decoded[i][:, 0] for i in range(n_series)])
            return pd.DataFrame(
                {"level": level_col, "pred": point_matrix.ravel()},
                index=long_index,
            )
        else:
            # quantile forecast: long DataFrame with columns [level, q_0.1, q_0.5, ...]
            q_columns = [f"q_{q}" for q in quantile_levels]
            data_dict: dict[str, np.ndarray] = {"level": level_col}
            for j, col in enumerate(q_columns):
                q_matrix = np.column_stack([decoded[i][:, j] for i in range(n_series)])
                data_dict[col] = q_matrix.ravel()
            return pd.DataFrame(data_dict, index=long_index)

    def predict(
        self,
        steps: int,
        history_dict: dict[str, pd.Series],
        past_exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        future_exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        quantiles: list[float] | tuple[float] | None,
        is_multiple_series: bool,
    ) -> pd.Series | pd.DataFrame:
        """
        Generate predictions using the Chronos-2 pipeline.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters 
        ----------
        steps : int
            Number of steps ahead to forecast.
        history_dict : dict of str → pd.Series
            Per-series context windows (already trimmed to
            `context_length`).
        past_exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series past covariates (already trimmed).
        future_exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return. If `None`, returns a point
            forecast (median).
        is_multiple_series : bool
            `True` when multiple series are provided.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame. Point forecast: columns
            `["level", "pred"]`. Quantile forecast: columns
            `["level", "q_0.1", "q_0.5", ...]`.
        
        """

        # NOTE: the pipeline is loaded lazily here so that the adapter can be
        # instantiated and fitted without requiring Chronos-2 to be installed.
        self._load_pipeline()

        if is_multiple_series:
            return self._predict_multiseries(
                steps            = steps,
                history_dict     = history_dict,
                past_exog_dict   = past_exog_dict,
                future_exog_dict = future_exog_dict,
                quantiles        = quantiles,
            )

        # Single-series path — same long-format output as multi-series
        quantile_levels = list(quantiles) if quantiles is not None else [0.5]
        name = next(iter(history_dict))
        history = history_dict[name]
        past_exog = past_exog_dict[name]
        future_exog = future_exog_dict[name]

        input_dict = self._build_chronos_input(
            target      = history.to_numpy(),
            past_exog   = past_exog,
            future_exog = future_exog,
        )

        quantile_preds, _ = self._pipeline.predict_quantiles(
            inputs            = [input_dict],
            prediction_length = steps,
            quantile_levels   = quantile_levels,
            **self.predict_kwargs,
        )

        q_arr = quantile_preds[0].squeeze(0)
        if hasattr(q_arr, "detach"):
            q_arr = q_arr.detach().cpu().numpy()

        forecast_index = expand_index(history.index, steps=steps)
        level_col = np.array([name] * steps)

        if quantiles is None:
            return pd.DataFrame(
                {"level": level_col, "pred": q_arr[:, 0]},
                index=forecast_index,
            )

        q_columns = [f"q_{q}" for q in quantile_levels]
        data_dict: dict[str, np.ndarray] = {"level": level_col}
        for j, col in enumerate(q_columns):
            data_dict[col] = q_arr[:, j]
        return pd.DataFrame(data_dict, index=forecast_index)


class TimesFM25Adapter:
    """
    Adapter for Google TimesFM 2.5 foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "google/timesfm-2.5-200m-pytorch".
    model : any, optional
        Pre-loaded and compiled TimesFM model instance. If `None`, the
        model is loaded and compiled lazily on the first `predict` call.
    context_length : int, default 512
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `last_window` is longer than `context_length` it
        is trimmed to this length; if it is shorter, all available
        observations are used as-is. Must be a positive integer. Defaults to
        512. TimesFM 2.5 supports up to 16 384.
    max_horizon : int, default 512
        Initial forecast horizon baked into the compiled decode function.
        If `predict` is called with `steps > max_horizon`, `max_horizon` is
        updated automatically and the model is recompiled transparently.
        Must be a positive integer.
    forecast_config_kwargs : dict, optional
        Additional keyword arguments forwarded verbatim to
        `timesfm.ForecastConfig` at compile time. Supported keys:
        `normalize_inputs`, `use_continuous_quantile_head`,
        `force_flip_invariance`, `infer_is_positive`,
        `fix_quantile_crossing`. Do **not** include `max_context` or
        `max_horizon` here — those are controlled by the corresponding
        adapter parameters.

    Notes
    -----
    TimesFM 2.5 supports only the fixed quantile levels
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Requesting any
    other level raises a `ValueError`.

    Covariate support (via TimesFM's `forecast_with_covariates`) is not
    yet implemented. Passing `exog` or `last_window_exog` issues an
    `IgnoredArgumentWarning` and the values are discarded.
    """

    SUPPORTED_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    allow_exogenous: bool = False

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 512,
        max_horizon: int = 512,
        forecast_config_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. "google/timesfm-2.5-200m-pytorch".
        model : any, optional
            Pre-loaded and compiled TimesFM model instance. If `None`, the
            model is loaded and compiled lazily on the first `predict` call.
        context_length : int, default 512
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `last_window` is
            longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        max_horizon : int, default 512
            Initial forecast horizon baked into the compiled decode function.
            If `predict` is called with `steps > max_horizon`, `max_horizon`
            is updated automatically and the model is recompiled
            transparently. Must be a positive integer.
        forecast_config_kwargs : dict, optional
            Additional keyword arguments forwarded verbatim to
            `timesfm.ForecastConfig` at compile time.
        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )
        if not isinstance(max_horizon, int) or max_horizon < 1:
            raise ValueError(
                f"`max_horizon` must be a positive integer. Got {max_horizon!r}."
            )

        self.model_id               = model_id
        self._model                 = model
        self.context_length         = context_length
        self.max_horizon            = max_horizon
        self.forecast_config_kwargs = dict(forecast_config_kwargs) if forecast_config_kwargs else {}
        self._history               = None
        self.is_fitted              = False
        self.is_multiple_series_    = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        dict
            Keys: `model_id`, `context_length`, `max_horizon`,
            `forecast_config_kwargs`.
        """
        return {
            'model_id':               self.model_id,
            'context_length':         self.context_length,
            'max_horizon':            self.max_horizon,
            'forecast_config_kwargs': self.forecast_config_kwargs or None,
        }

    def set_params(self, **params) -> TimesFM25Adapter:
        """
        Set adapter parameters. Resets the model when parameters that affect
        compilation change (`model_id`, `context_length`, `max_horizon`,
        `forecast_config_kwargs`).

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `max_horizon`,
            `forecast_config_kwargs`.

        Returns
        -------
        self : TimesFM25Adapter
        """
        valid = {'model_id', 'context_length', 'max_horizon', 'forecast_config_kwargs'}
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for TimesFM25Adapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )
        model_reset_keys = {'model_id', 'context_length', 'max_horizon', 'forecast_config_kwargs'}
        if params.keys() & model_reset_keys:
            self._model = None
        for key, value in params.items():
            if key == 'context_length':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                self.context_length = value
            elif key == 'max_horizon':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`max_horizon` must be a positive integer. Got {value!r}."
                    )
                self.max_horizon = value
            elif key == 'forecast_config_kwargs':
                self.forecast_config_kwargs = dict(value) if value else {}
            else:
                setattr(self, key, value)
        return self

    def _load_model(self) -> None:
        """
        Load (but do not compile) the TimesFM 2.5 model into `self._model`
        if not already set.

        The model is imported lazily from `timesfm` and loaded via
        `TimesFM_2p5_200M_torch.from_pretrained`. Compilation is deferred to
        `_ensure_compiled`, which is called from `predict` with the actual
        forecast horizon so that the compiled decode graph is sized exactly
        for the requested number of steps rather than the (much larger)
        `max_horizon` ceiling. This method is a no-op when `self._model` is
        already populated.

        Raises
        ------
        ImportError
            If `timesfm[torch]` is not installed.
        """

        if self._model is not None:
            return
        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "timesfm is required for TimesFM25Adapter. "
                "Install it with `pip install git+https://github.com/google-research/timesfm.git`."
            ) from exc

        # Workaround for a compatibility issue between huggingface_hub and
        # timesfm: huggingface_hub's `from_pretrained` passes `proxies` and
        # `resume_download` to `_from_pretrained`, but timesfm's
        # `_from_pretrained` does not declare them as explicit parameters, so
        # they fall into **model_kwargs and are forwarded to __init__, raising
        # a TypeError. A local subclass overrides `_from_pretrained` to absorb
        # those kwargs without modifying any global state.
        class _TimesFMCompat(timesfm.TimesFM_2p5_200M_torch):
            @classmethod
            def _from_pretrained(cls, *, proxies=None, resume_download=None, **kwargs):  # type: ignore[override]
                return super()._from_pretrained(**kwargs)

        self._model = _TimesFMCompat.from_pretrained(self.model_id)

    def _ensure_compiled(self, steps: int) -> None:
        """
        Compile the model for the given forecast horizon if not already
        compiled for at least `steps` steps.

        This is separated from `_load_model` so that compilation uses the
        *actual* number of requested forecast steps rather than `max_horizon`.
        TimesFM's compiled decode always runs `forecast_config.max_horizon`
        autoregressive decode iterations regardless of the requested horizon;
        the true horizon is only used to *slice* the output afterwards. When
        the compiled `max_horizon` is large (e.g. the default 512) but
        `steps` is small (e.g. 12), the model performs up to
        `(max_horizon - 1) // output_patch_len` unnecessary extra transformer
        forward passes per inference call. Compiling here with
        `max_horizon = steps` reduces those wasted passes to zero for the
        typical backtesting case where `steps` is constant across folds.

        If the model was already compiled for a horizon `>= steps` (e.g. a
        pre-compiled model passed via the `model` constructor argument), this
        method is a no-op.

        Parameters
        ----------
        steps : int
            The forecast horizon that the model must support.
        """

        fc = getattr(self._model, 'forecast_config', None)
        if fc is not None and steps <= fc.max_horizon:
            return

        import timesfm
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_length,
                max_horizon=steps,
                **self.forecast_config_kwargs,
            )
        )

    def fit(
        self,
        series_dict: dict[str, pd.Series],
        exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        is_multiple_series: bool,
    ) -> TimesFM25Adapter:
        """
        Store the training series as context.

        No model training occurs since TimesFM 2.5 is a zero-shot inference
        model.  All input normalization and validation is performed upstream
        by `FoundationModel`.

        Parameters
        ----------
        series_dict : dict of str → pd.Series
            Normalized training series, one entry per series.
        exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series exogenous variables (ignored — accepted for API
            consistency).
        is_multiple_series : bool
            `True` when multiple series are provided.

        Returns
        -------
        self : TimesFM25Adapter
        """

        if any(v is not None for v in exog_dict.values()):
            warnings.warn(
                "TimesFM25Adapter does not currently support covariates. "
                "`exog` is ignored.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )

        self.is_multiple_series_ = is_multiple_series
        self._history = {
            name: s.iloc[-self.context_length :].copy()
            for name, s in series_dict.items()
        }
        self.is_fitted = True
        return self

    def _predict_multiseries(
        self,
        steps: int,
        quantiles: list[float] | None,
        history_dict: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Internal multi-series prediction logic.

        `history_dict` is already resolved and trimmed by the caller.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        quantiles : list of float or None
            Quantile levels to return from `SUPPORTED_QUANTILES`.
        history_dict : dict of str → pd.Series
            Per-series context windows (already trimmed).

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame.
        """

        series_names = list(history_dict.keys())

        # Build list of 1-D numpy arrays for TimesFM
        inputs_list = [
            history_dict[name].to_numpy()
            for name in series_names
        ]

        point_forecast, quantile_forecast = self._model.forecast(
            horizon=steps,
            inputs=inputs_list,
        )
        # point_forecast  : (n_series, steps)
        # quantile_forecast: (n_series, steps, 10)  — idx 0 = mean, 1-9 = q0.1-q0.9

        n_series = len(series_names)
        per_series_forecast_indices = [
            expand_index(history_dict[name].index, steps=steps)
            for name in series_names
        ]
        long_index = np.array([
            per_series_forecast_indices[j][i]
            for i in range(steps)
            for j in range(n_series)
        ])
        level_col  = np.tile(series_names, steps)

        pf = np.asarray(point_forecast)[:n_series]   # (n_series, steps)
        qf = np.asarray(quantile_forecast)[:n_series] # (n_series, steps, 10)

        if quantiles is None:
            point_matrix = pf.T  # (steps, n_series)
            return pd.DataFrame(
                {"level": level_col, "pred": point_matrix.ravel()},
                index=long_index,
            )
        else:
            q_columns = [f"q_{q}" for q in quantiles]
            q_indices  = [round(q * 10) for q in quantiles]
            data_dict: dict[str, np.ndarray] = {"level": level_col}
            for j, col in enumerate(q_columns):
                q_matrix = qf[:, :, q_indices[j]].T  # (steps, n_series)
                data_dict[col] = q_matrix.ravel()
            return pd.DataFrame(data_dict, index=long_index)

    def predict(
        self,
        steps: int,
        history_dict: dict[str, pd.Series],
        past_exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        future_exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        quantiles: list[float] | tuple[float] | None,
        is_multiple_series: bool,
    ) -> pd.Series | pd.DataFrame:
        """
        Generate predictions using the TimesFM 2.5 model.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        history_dict : dict of str → pd.Series
            Per-series context windows (already trimmed to
            `context_length`).
        past_exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series past covariates (ignored by TimesFM).
        future_exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series future covariates (ignored by TimesFM).
        quantiles : list of float or None
            Quantile levels. Must be a subset of `SUPPORTED_QUANTILES`.
        is_multiple_series : bool
            `True` when multiple series are provided.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame. Point forecast: columns
            `["level", "pred"]`. Quantile forecast: columns
            `["level", "q_0.1", "q_0.5", ...]`.

        Raises
        ------
        ValueError
            If a requested quantile level is not in `SUPPORTED_QUANTILES`
            or `steps` exceeds `max_horizon`.
        """

        if quantiles is not None:
            quantile_list = list(quantiles)
            for q in quantile_list:
                if not any(abs(q - sq) < 1e-9 for sq in self.SUPPORTED_QUANTILES):
                    raise ValueError(
                        f"TimesFM 2.5 only supports quantile levels "
                        f"{self.SUPPORTED_QUANTILES}. Got {q!r}. "
                        f"Quantile interpolation is not supported."
                    )
        else:
            quantile_list = None

        self._load_model()

        if steps > self.max_horizon:
            raise ValueError(
                f"`steps` ({steps}) exceeds `max_horizon` ({self.max_horizon})."
            )

        self._ensure_compiled(steps)

        if is_multiple_series:
            return self._predict_multiseries(
                steps        = steps,
                quantiles    = quantile_list,
                history_dict = history_dict,
            )

        # Single-series path — same long-format output as multi-series
        name = next(iter(history_dict))
        history = history_dict[name]

        point_forecast, quantile_forecast = self._model.forecast(
            horizon=steps,
            inputs=[history.to_numpy()],
        )

        forecast_index = expand_index(history.index, steps=steps)
        level_col = np.array([name] * steps)

        if quantile_list is None:
            return pd.DataFrame(
                {"level": level_col, "pred": np.asarray(point_forecast[0])},
                index=forecast_index,
            )

        q_indices = [round(q * 10) for q in quantile_list]
        qf        = np.asarray(quantile_forecast[0])
        q_columns = [f"q_{q}" for q in quantile_list]
        data_dict: dict[str, np.ndarray] = {"level": level_col}
        for j, col in enumerate(q_columns):
            data_dict[col] = qf[:, q_indices[j]]
        return pd.DataFrame(data_dict, index=forecast_index)


class MoiraiAdapter:
    """
    Adapter for Salesforce Moirai-2 foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"Salesforce/moirai-2.0-R-small"`.
        Must be a `Salesforce/moirai-2.0-R-{small,base,large}` variant.
    module : any, optional
        Pre-loaded `Moirai2Module` instance. If `None`, the module is
        loaded lazily on the first call to `predict`.
    context_length : int, default 2048
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `last_window` is longer than `context_length`
        it is trimmed to this length; if it is shorter, all available
        observations are used as-is. Must be a positive integer.

    Notes
    -----
    Moirai-2 supports only the fixed quantile levels
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Requesting any
    other level raises a `ValueError`.

    Covariate support via the high-level `Moirai2Forecast.predict()` API
    is not functional: the padding/truncation loop inside `predict()`
    clips every list-valued field — including `feat_dynamic_real` — to
    `context_length`, discarding the future portion that future
    covariates require. Passing `exog` or `last_window_exog` issues an
    `IgnoredArgumentWarning` and the values are discarded.
    """

    SUPPORTED_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    allow_exogenous: bool = False

    def __init__(
        self,
        model_id: str,
        *,
        module: Any | None = None,
        context_length: int = 2048,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. `"Salesforce/moirai-2.0-R-small"`.
        module : any, optional
            Pre-loaded `Moirai2Module` instance. If `None`, the module
            is loaded lazily on the first call to `predict`.
        context_length : int, default 2048
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `last_window`
            is longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. "
                f"Got {context_length!r}."
            )

        self.model_id        = model_id
        self._module         = module
        self.context_length  = context_length
        self._forecast_obj   = None
        self._history        = None
        self.is_fitted       = False
        self.is_multiple_series_ = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        dict
            Keys: `model_id`, `context_length`.
        """
        return {
            'model_id':       self.model_id,
            'context_length': self.context_length,
        }

    def set_params(self, **params) -> MoiraiAdapter:
        """
        Set adapter parameters. Resets the module and forecast object when
        `model_id` or `context_length` changes.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`.

        Returns
        -------
        self : MoiraiAdapter
        """
        valid = {'model_id', 'context_length'}
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for MoiraiAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )
        if params.keys() & {'model_id', 'context_length'}:
            self._module = None
            self._forecast_obj = None
        for key, value in params.items():
            if key == 'context_length':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. "
                        f"Got {value!r}."
                    )
                self.context_length = value
            else:
                setattr(self, key, value)
        return self

    def _load_module(self) -> None:
        """
        Load the `Moirai2Module` into `self._module` if not already set.

        The module is imported lazily from `uni2ts` and instantiated via
        `Moirai2Module.from_pretrained`, then set to evaluation mode.
        This method is a no-op when `self._module` is already populated.

        Raises
        ------
        ImportError
            If `uni2ts` is not installed.
        """

        if self._module is not None:
            return
        try:
            from uni2ts.model.moirai2 import Moirai2Module
        except ImportError as exc:
            raise ImportError(
                "uni2ts is required for MoiraiAdapter. "
                "Install it with `pip install uni2ts`."
            ) from exc
        self._module = Moirai2Module.from_pretrained(self.model_id)
        self._module.eval()

    def _ensure_forecast_obj(self) -> None:
        """
        Build the `Moirai2Forecast` inference wrapper if not already set.

        Calls `_load_module` then wraps `self._module` in a
        `Moirai2Forecast` with `prediction_length=1` (overridden
        per-call via `hparams_context`) and sets it to evaluation mode.
        This method is a no-op when `self._forecast_obj` is already
        populated.

        Raises
        ------
        ImportError
            If `uni2ts` is not installed.
        """

        if self._forecast_obj is not None:
            return
        self._load_module()
        from uni2ts.model.moirai2 import Moirai2Forecast

        self._forecast_obj = Moirai2Forecast(
            module=self._module,
            prediction_length=1,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).eval()

    def fit(
        self,
        series_dict: dict[str, pd.Series],
        exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        is_multiple_series: bool,
    ) -> MoiraiAdapter:
        """
        Store the training series as context.

        No model training occurs since Moirai-2 is a zero-shot inference
        model.  All input normalization and validation is performed upstream
        by `FoundationModel`.

        Parameters
        ----------
        series_dict : dict of str → pd.Series
            Normalized training series, one entry per series.
        exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Ignored — accepted for API consistency.
        is_multiple_series : bool
            `True` when multiple series are provided.

        Returns
        -------
        self : MoiraiAdapter
        """

        if any(v is not None for v in exog_dict.values()):
            warnings.warn(
                "MoiraiAdapter does not support covariates. "
                "`exog` is ignored.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )

        self.is_multiple_series_ = is_multiple_series
        self._history = {
            name: s.iloc[-self.context_length :].copy()
            for name, s in series_dict.items()
        }
        self.is_fitted = True
        return self

    def _run_inference(
        self,
        inputs_list: list[np.ndarray],
        steps: int,
    ) -> np.ndarray:
        """
        Run batched inference with `Moirai2Forecast`.

        Parameters
        ----------
        inputs_list : list of np.ndarray
            List of 2-D arrays with shape `(T, 1)`, one per series.
            Each array holds `float64` values.
        steps : int
            Forecast horizon.

        Returns
        -------
        np.ndarray
            Array of shape `(n_series, 9, steps)` containing quantile
            forecasts for the 9 fixed levels in `SUPPORTED_QUANTILES`
            order.
        """

        self._ensure_forecast_obj()
        with self._forecast_obj.hparams_context(prediction_length=steps):
            raw = self._forecast_obj.predict(inputs_list)
        return raw

    def _predict_multiseries(
        self,
        steps: int,
        quantiles: list[float] | None,
        history_dict: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Internal multi-series prediction logic.

        `history_dict` is already resolved and trimmed by the caller.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        quantiles : list of float or None
            Quantile levels to return from `SUPPORTED_QUANTILES`.
        history_dict : dict of str → pd.Series
            Per-series context windows (already trimmed).

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame.
        """

        series_names = list(history_dict.keys())

        # Build batched input list: each array is (T, 1) float64
        inputs_list = [
            history_dict[name].to_numpy(dtype=float).reshape(-1, 1)
            for name in series_names
        ]

        raw = self._run_inference(inputs_list, steps)

        n_series = len(series_names)
        per_series_forecast_indices = [
            expand_index(history_dict[name].index, steps=steps)
            for name in series_names
        ]
        long_index = np.array([
            per_series_forecast_indices[j][i]
            for i in range(steps)
            for j in range(n_series)
        ])
        level_col  = np.tile(series_names, steps)

        quantile_levels = list(quantiles) if quantiles is not None else [0.5]
        q_indices = [
            next(
                i for i, sq in enumerate(self.SUPPORTED_QUANTILES)
                if abs(q - sq) < 1e-9
            )
            for q in quantile_levels
        ]

        # preds_per_series[i]: (steps, n_q)
        preds_per_series = [raw[i][q_indices, :].T for i in range(n_series)]

        if quantiles is None:
            # quantile_levels == [0.5], so index 0 is the median
            point_matrix = np.column_stack(
                [preds_per_series[i][:, 0] for i in range(n_series)]
            )
            return pd.DataFrame(
                {"level": level_col, "pred": point_matrix.ravel()},
                index=long_index,
            )

        q_columns = [f"q_{q}" for q in quantile_levels]
        data_dict: dict[str, np.ndarray] = {"level": level_col}
        for j, col in enumerate(q_columns):
            q_matrix = np.column_stack(
                [preds_per_series[i][:, j] for i in range(n_series)]
            )
            data_dict[col] = q_matrix.ravel()
        return pd.DataFrame(data_dict, index=long_index)

    def predict(
        self,
        steps: int,
        history_dict: dict[str, pd.Series],
        past_exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        future_exog_dict: dict[str, pd.DataFrame | pd.Series | None],
        quantiles: list[float] | tuple[float] | None,
        is_multiple_series: bool,
    ) -> pd.Series | pd.DataFrame:
        """
        Generate predictions using Moirai-2.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        history_dict : dict of str → pd.Series
            Per-series context windows (already trimmed to
            `context_length`).
        past_exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series past covariates (ignored by Moirai).
        future_exog_dict : dict of str → pd.DataFrame, pd.Series, or None
            Per-series future covariates (ignored by Moirai).
        quantiles : list of float or None
            Quantile levels. Must be a subset of `SUPPORTED_QUANTILES`.
        is_multiple_series : bool
            `True` when multiple series are provided.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame. Point forecast: columns
            `["level", "pred"]`. Quantile forecast: columns
            `["level", "q_0.1", "q_0.5", ...]`.

        Raises
        ------
        ValueError
            If a requested quantile level is not in `SUPPORTED_QUANTILES`.
        """

        if quantiles is not None:
            quantile_list = list(quantiles)
            for q in quantile_list:
                if not any(abs(q - sq) < 1e-9 for sq in self.SUPPORTED_QUANTILES):
                    raise ValueError(
                        f"Moirai-2 only supports quantile levels "
                        f"{self.SUPPORTED_QUANTILES}. Got {q!r}. "
                        f"Quantile interpolation is not supported."
                    )
        else:
            quantile_list = None

        if is_multiple_series:
            return self._predict_multiseries(
                steps        = steps,
                quantiles    = quantile_list,
                history_dict = history_dict,
            )

        # Single-series path — same long-format output as multi-series
        name = next(iter(history_dict))
        history = history_dict[name]

        inputs_list = [history.to_numpy(dtype=float).reshape(-1, 1)]
        raw = self._run_inference(inputs_list, steps)

        quantile_levels = quantile_list if quantile_list is not None else [0.5]
        q_indices = [
            next(
                i for i, sq in enumerate(self.SUPPORTED_QUANTILES)
                if abs(q - sq) < 1e-9
            )
            for q in quantile_levels
        ]
        result = raw[0][q_indices, :].T

        forecast_index = expand_index(history.index, steps=steps)
        level_col = np.array([name] * steps)

        if quantile_list is None:
            return pd.DataFrame(
                {"level": level_col, "pred": result[:, 0]},
                index=forecast_index,
            )

        q_columns = [f"q_{q}" for q in quantile_list]
        data_dict: dict[str, np.ndarray] = {"level": level_col}
        for j, col in enumerate(q_columns):
            data_dict[col] = result[:, j]
        return pd.DataFrame(data_dict, index=forecast_index)


_ADAPTER_REGISTRY: dict[str, type] = {
    "autogluon/chronos": Chronos2Adapter,
    "google/timesfm":    TimesFM25Adapter,
    "Salesforce/moirai": MoiraiAdapter,
    # "ibm/TTM": TTMAdapter,
}


def _resolve_adapter(model_id: str) -> type:
    """Return the adapter class for *model_id* based on prefix matching."""
    for prefix, cls in _ADAPTER_REGISTRY.items():
        if model_id.startswith(prefix):
            return cls
    raise ValueError(
        f"No adapter found for model '{model_id}'. "
        f"Registered prefixes: {list(_ADAPTER_REGISTRY)}."
    )


