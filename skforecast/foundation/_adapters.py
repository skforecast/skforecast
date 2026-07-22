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
import contextlib
import io
import numpy as np
import pandas as pd
import warnings


def _resolve_torch_device(device: str) -> str:
    """
    Resolve a device string to a concrete PyTorch device name.

    If `device` is `"auto"`, the best available accelerator is selected
    in priority order: CUDA > MPS (Apple Silicon) > CPU.

    Parameters
    ----------
    device : str
        Device string. Use `"auto"` for automatic selection, or an
        explicit name such as `"cuda"`, `"mps"`, or `"cpu"`.

    Returns
    -------
    device : str
        Resolved device name.

    """

    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChronosAdapter:
    """
    Adapter for Amazon Chronos foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "autogluon/chronos-2-small".
    pipeline : BaseChronosPipeline, default None
        Pre-loaded pipeline instance. If `None`, the pipeline is loaded
        lazily on the first call to `predict`.
    context_length : int, default 8192
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Defaults to 8192, which matches the maximum context
        window of Chronos. Must be a positive integer.
    predict_kwargs : dict, default None
        Additional keyword arguments forwarded to the pipeline's
        `predict_quantiles` method.
    device_map : str, default 'auto'
        Device placement for the model. `"auto"` selects the best
        available accelerator (CUDA > MPS > CPU). Also accepts explicit
        values such as `"cuda"`, `"mps"`, or `"cpu"`, forwarded to
        `BaseChronosPipeline.from_pretrained`.
    torch_dtype : object, default None
        Torch dtype forwarded to `BaseChronosPipeline.from_pretrained`.
    cross_learning : bool, default False
        If `True`, Chronos shares information across all series in
        the batch when predicting in multi-series mode. Forwarded
        directly to `predict_quantiles`. Ignored in single-series mode.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    predict_kwargs : dict
        Additional keyword arguments forwarded to `predict_quantiles`.
    device_map : str
        Device map string for model loading.
    torch_dtype : object
        Torch dtype for model loading.
    cross_learning : bool
        Whether cross-series learning is enabled.
    is_fitted : bool
        Whether the adapter has been fitted.

    References
    ----------
    .. [1] https://github.com/amazon-science/chronos-forecasting
    
    .. [2] https://huggingface.co/amazon/chronos-2

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        pipeline: Any | None = None,
        context_length: int = 8192,
        predict_kwargs: dict[str, Any] | None = None,
        device_map: str = "auto",
        torch_dtype: Any | None = None,
        cross_learning: bool = False,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. "autogluon/chronos-2-small".
        pipeline : BaseChronosPipeline, default None
            Pre-loaded pipeline instance. If `None`, the pipeline is
            loaded lazily on the first call to `predict`.
        context_length : int, default 8192
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if
            `context` is longer than `context_length` it is trimmed to
            this length before inference; if it is shorter, all available
            observations are passed as-is and the model handles reduced
            context gracefully. Defaults to 8192, which matches the
            maximum context window of Chronos. Must be a positive
            integer.
        predict_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to the
            pipeline's `predict_quantiles` method.
        device_map : str, default 'auto'
            Device placement for the model. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU). Also accepts
            explicit values such as `"cuda"`, `"mps"`, or `"cpu"`,
            forwarded to `BaseChronosPipeline.from_pretrained`.
        torch_dtype : object, default None
            Torch dtype forwarded to `BaseChronosPipeline.from_pretrained`
            (e.g. `torch.bfloat16`).
        cross_learning : bool, default False
            If `True`, Chronos shares information across all series in
            the batch when predicting in multi-series mode. Forwarded
            directly to `predict_quantiles`. Ignored in single-series mode.
        
        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )

        self.model_id       = model_id
        self._pipeline      = pipeline
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.predict_kwargs = predict_kwargs or {}
        self.device_map     = device_map
        self.torch_dtype    = torch_dtype
        self.cross_learning = cross_learning
        self.is_fitted      = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
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

    def set_params(self, **params) -> ChronosAdapter:
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
        self : ChronosAdapter

        """

        valid = {
            'model_id', 'cross_learning', 'context_length',
            'device_map', 'torch_dtype', 'predict_kwargs',
        }
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for ChronosAdapter: {sorted(invalid)}. "
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

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
    ) -> ChronosAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since Chronos is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : ChronosAdapter

        """

        self.context_ = context
        self.context_exog_ = context_exog
        self.is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
        exog: dict[str, pd.DataFrame | pd.Series | None],
        quantiles: list[float] | tuple[float] | None
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using the Chronos pipeline.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : dict
            Per-series past covariates (already trimmed).
        exog : dict
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return. If `None`, a point forecast
            (median, quantile 0.5) is produced.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)`.
        
        """

        # NOTE: the pipeline is loaded lazily here so that the adapter can be
        # instantiated and fitted without requiring Chronos to be installed.
        self._load_pipeline()

        series_names_in = list(context.keys())
        quantile_levels = list(quantiles) if quantiles is not None else [0.5]

        inputs_list = [
            self._build_chronos_input(
                context      = context[name].to_numpy(),
                context_exog = context_exog[name] if context_exog is not None else None,
                exog         = exog[name] if exog is not None else None,
            )
            for name in series_names_in
        ]

        quantile_preds, _ = self._pipeline.predict_quantiles(
            inputs            = inputs_list,
            prediction_length = steps,
            quantile_levels   = quantile_levels,
            cross_learning    = self.cross_learning if len(series_names_in) > 1 else False,
            **self.predict_kwargs,
        )

        predictions: dict[str, np.ndarray] = {}
        for i, name in enumerate(series_names_in):
            q_arr = quantile_preds[i].squeeze(0)
            if hasattr(q_arr, "detach"):
                q_arr = q_arr.detach().cpu().numpy()
            else:
                q_arr = np.asarray(q_arr)
            predictions[name] = q_arr

        return predictions

    def _load_pipeline(self) -> None:
        """
        Load the Chronos pipeline into `self._pipeline` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `chronos-forecasting` >=2.0 is not installed.

        Notes
        -----
        The pipeline is imported lazily from `chronos` and instantiated via
        `BaseChronosPipeline.from_pretrained`, which auto-dispatches to the
        correct pipeline class based on the model config. Optional
        `device_map` and `torch_dtype` stored at initialisation are
        forwarded to the constructor. This method is a no-op when
        `self._pipeline` is already populated.

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
        kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        
        self._pipeline = BaseChronosPipeline.from_pretrained(self.model_id, **kwargs)

    @staticmethod
    def _to_covariate_array(col_data: Any) -> np.ndarray:
        """
        Convert a covariate column to a numpy array.

        Numeric columns (int, float) and boolean columns are cast to
        `float32`. All other dtypes (object, string, Categorical) are left
        as-is so that Chronos can handle them as categorical covariates
        natively.

        Parameters
        ----------
        col_data : array-like
            A single covariate column (e.g. a pandas Series or 1-D array).

        Returns
        -------
        col_array : numpy ndarray
            A 1-D numpy array. Numeric/bool are cast to `float32`. Others
            keep their original dtype (typically `object` for string and
            categorical data).
        
        """

        # Handle pandas Series first to correctly process nullable extension
        # dtypes (pd.Int64Dtype, pd.Float64Dtype, pd.BooleanDtype): np.asarray()
        # on those produces dtype=object with pd.NA sentinels instead of float32.
        if isinstance(col_data, pd.Series):
            if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
                return col_data.astype(np.float32).to_numpy()
            return col_data.to_numpy()

        # Fallback for numpy arrays, lists, etc.
        arr = np.asarray(col_data)
        if arr.dtype.kind in ("i", "u", "f", "b"):  # integer, unsigned int, float, bool
            return arr.astype(np.float32)
        
        return arr

    def _build_chronos_input(
        self,
        context: np.ndarray,
        context_exog: pd.DataFrame | pd.Series | None = None,
        exog: pd.DataFrame | pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Build the input dict consumed by the pipeline's `predict_quantiles` method.

        Parameters
        ----------
        context : numpy ndarray
            1-D array of observed time series values used as context. Must be
            castable to `float32`.
        context_exog : pandas DataFrame, pandas Series, default None
            Historical exogenous variables whose index is aligned to
            `context`. Each column (or the single Series, referenced by
            its name) becomes an entry in the returned
            "past_covariates" dict. Numeric and boolean columns are
            cast to `float32`; string and categorical columns are passed
            as-is and handled natively by Chronos.
        exog : pandas DataFrame, pandas Series, default None
            Future-known exogenous variables covering the forecast horizon.
            Must have exactly `prediction_length` rows. Each column
            becomes an entry in the returned "future_covariates" dict.
            Numeric and boolean columns are cast to `float32`; string and
            categorical columns are passed as-is.

        Returns
        -------
        input_dict : dict
            Dictionary with mandatory key "target" (1-D `float32`
            `numpy ndarray`) and optional keys "past_covariates" and
            "future_covariates", each mapping column names to 1-D
            arrays (`float32` for numeric/bool columns, `object` dtype
            for string/categorical columns).
        
        """

        input_dict = {"target": np.asarray(context, dtype=np.float32)}
        if context_exog is not None:
            df = (
                context_exog
                if isinstance(context_exog, pd.DataFrame)
                else context_exog.to_frame()
            )
            input_dict["past_covariates"] = {
                col: ChronosAdapter._to_covariate_array(df[col]) for col in df.columns
            }
        if exog is not None:
            df = (
                exog
                if isinstance(exog, pd.DataFrame)
                else exog.to_frame()
            )
            input_dict["future_covariates"] = {
                col: ChronosAdapter._to_covariate_array(df[col]) for col in df.columns
            }
        
        return input_dict


class TimesFMAdapter:
    """
    Adapter for Google TimesFM foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "google/timesfm-2.5-200m-pytorch".
    model : object, default None
        Pre-loaded and compiled TimesFM model instance. If `None`, the
        model is loaded and compiled lazily on the first `predict` call.
    context_length : int, default 512
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it
        is trimmed to this length; if it is shorter, all available
        observations are used as-is. Must be a positive integer. Defaults to
        512. TimesFM supports up to 16_384.
    max_horizon : int, default 512
        Maximum forecast horizon. If `predict` is called with
        `steps > max_horizon`, a `ValueError` is raised. The model is
        compiled lazily for the exact requested `steps` (up to this
        ceiling) to avoid unnecessary decode iterations. Must be a
        positive integer.
    forecast_config_kwargs : dict, default None
        Additional keyword arguments forwarded verbatim to
        `timesfm.ForecastConfig` at compile time. Supported keys:
        `normalize_inputs`, `use_continuous_quantile_head`,
        `force_flip_invariance`, `infer_is_positive`,
        `fix_quantile_crossing`. Do **not** include `max_context` or
        `max_horizon` here — those are controlled by the corresponding
        adapter parameters.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Not used, present here for API consistency by convention.
    context_length : int
        Maximum number of historical observations used as context.
    max_horizon : int
        Maximum forecast horizon.
    forecast_config_kwargs : dict
        Additional keyword arguments forwarded to `ForecastConfig`.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    TimesFM supports only the fixed quantile levels
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Requesting any
    other level raises a `ValueError`.

    Covariate support (via TimesFM's `forecast_with_covariates`) is not
    yet implemented. Passing `exog` or `context_exog` issues an
    `IgnoredArgumentWarning` and the values are discarded.

    References
    ----------
    .. [1] https://github.com/google-research/timesfm

    .. [2] https://huggingface.co/google/timesfm-2.5-200m-pytorch

    """

    SUPPORTED_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    allow_exog: bool = False

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
        model : object, default None
            Pre-loaded and compiled TimesFM model instance. If `None`, the
            model is loaded and compiled lazily on the first `predict` call.
        context_length : int, default 512
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `context` is
            longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        max_horizon : int, default 512
            Maximum forecast horizon. If `predict` is called with
            `steps > max_horizon`, a `ValueError` is raised. The model
            is compiled lazily for the exact requested `steps` (up to
            this ceiling) to avoid unnecessary decode iterations. Must
            be a positive integer.
        forecast_config_kwargs : dict, default None
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
        self.context_               = None
        self.context_exog_          = None
        self.context_length         = context_length
        self.max_horizon            = max_horizon
        self.forecast_config_kwargs = dict(forecast_config_kwargs) if forecast_config_kwargs else {}
        self.is_fitted              = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `max_horizon`,
            `forecast_config_kwargs`.
        
        """
        return {
            'model_id':               self.model_id,
            'context_length':         self.context_length,
            'max_horizon':            self.max_horizon,
            'forecast_config_kwargs': self.forecast_config_kwargs or None,
        }

    def set_params(self, **params) -> TimesFMAdapter:
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
        self : TimesFMAdapter

        """

        valid = {'model_id', 'context_length', 'max_horizon', 'forecast_config_kwargs'}
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for TimesFMAdapter: {sorted(invalid)}. "
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

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: Any,
    ) -> TimesFMAdapter:
        """
        Store the training series.
        No model training occurs since TimesFM is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : Any
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : TimesFMAdapter

        """

        self.context_ = context
        self.is_fitted = True
        
        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: Any,
        exog: Any,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using the TimesFM model.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : Any
            Not used, present here for API consistency by convention.
        exog : Any
            Not used, present here for API consistency by convention.
        quantiles : list of float or None
            Quantile levels. Must be a subset of `SUPPORTED_QUANTILES`.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)`.

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
                        f"TimesFM only supports quantile levels "
                        f"{self.SUPPORTED_QUANTILES}. Got {q!r}. "
                        f"Quantile interpolation is not supported."
                    )
        else:
            quantile_list = None

        if steps > self.max_horizon:
            raise ValueError(
                f"`steps` ({steps}) exceeds `max_horizon` ({self.max_horizon})."
            )

        self._load_model()
        self._ensure_compiled(steps)

        series_names_in = list(context.keys())
        inputs_list = [
            context[name].to_numpy() for name in series_names_in
        ]

        point_forecast, quantile_forecast = self._model.forecast(
            horizon=steps,
            inputs=inputs_list,
        )
        # point_forecast  : (n_series, steps)
        # quantile_forecast: (n_series, steps, 10)  — idx 0 = mean, 1-9 = q0.1-q0.9

        predictions: dict[str, np.ndarray] = {}
        for i, name in enumerate(series_names_in):
            if quantile_list is None:
                # Point forecast: shape (steps, 1)
                predictions[name] = np.asarray(point_forecast[i]).reshape(-1, 1)
            else:
                q_indices = [round(q * 10) for q in quantile_list]
                qf = np.asarray(quantile_forecast[i])
                predictions[name] = qf[:, q_indices]  # (steps, n_quantiles)

        return predictions

    def _load_model(self) -> None:
        """
        Load (but do not compile) the TimesFM model into `self._model`
        if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `timesfm[torch]` is not installed.

        Notes
        -----
        The model is imported lazily from `timesfm` and loaded via
        `TimesFM_2p5_200M_torch.from_pretrained`. Compilation is deferred to
        `_ensure_compiled`, which is called from `predict` with the actual
        forecast horizon so that the compiled decode graph is sized exactly
        for the requested number of steps rather than the (much larger)
        `max_horizon` ceiling. This method is a no-op when `self._model` is
        already populated.
        """

        if self._model is not None:
            return
        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "timesfm is required for TimesFMAdapter. "
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

        Parameters
        ----------
        steps : int
            The forecast horizon that the model must support.

        Returns
        -------
        None

        Notes
        -----
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
        """

        fc = getattr(self._model, 'forecast_config', None)
        if fc is not None and steps <= fc.max_horizon:
            return

        import timesfm
        self._model.compile(
            timesfm.ForecastConfig(
                max_context = self.context_length,
                max_horizon = steps,
                **self.forecast_config_kwargs,
            )
        )


class MoiraiAdapter:
    """
    Adapter for Salesforce Moirai foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"Salesforce/moirai-2.0-R-small"`.
        Must be a `Salesforce/moirai-2.0-R-{small,base,large}` variant.
    module : object, default None
        Pre-loaded `Moirai2Module` instance. If `None`, the module is
        loaded lazily on the first call to `predict`.
    context_length : int, default 2048
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length`
        it is trimmed to this length; if it is shorter, all available
        observations are used as-is. Must be a positive integer.
    device : str, default 'auto'
        Device placement for the model. `"auto"` selects the best
        available accelerator (CUDA > MPS > CPU). Also accepts explicit
        values such as `"cuda"`, `"mps"`, or `"cpu"`.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Not used, present here for API consistency by convention.
    context_length : int
        Maximum number of historical observations used as context.
    device : str
        Device placement for the model.
    _forecast_obj : object
        Internal Moirai forecast object, populated at the first call to
        `predict`.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    Moirai supports only the fixed quantile levels
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Requesting any
    other level raises a `ValueError`.

    Covariate support via the high-level `Moirai2Forecast.predict()` API
    is not functional: the padding/truncation loop inside `predict()`
    clips every list-valued field — including `feat_dynamic_real` — to
    `context_length`, discarding the future portion that future
    covariates require. Passing `exog` or `context_exog` issues an
    `IgnoredArgumentWarning` and the values are discarded.

    References
    ----------
    .. [1] https://github.com/SalesforceAIResearch/uni2ts

    .. [2] https://huggingface.co/Salesforce/moirai-2.0-R-small

    """

    SUPPORTED_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    allow_exog: bool = False

    def __init__(
        self,
        model_id: str,
        *,
        module: Any | None = None,
        context_length: int = 2048,
        device: str = "auto",
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. `"Salesforce/moirai-2.0-R-small"`.
        module : object, default None
            Pre-loaded `Moirai2Module` instance. If `None`, the module
            is loaded lazily on the first call to `predict`.
        context_length : int, default 2048
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `context`
            is longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        device : str, default 'auto'
            Device placement for the model. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU). Also accepts
            explicit values such as `"cuda"`, `"mps"`, or `"cpu"`.
        
        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. "
                f"Got {context_length!r}."
            )

        self.model_id       = model_id
        self._module        = module
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.device         = device
        self._forecast_obj  = None
        self.is_fitted      = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `device`.
        """
        return {
            'model_id':       self.model_id,
            'context_length': self.context_length,
            'device':         self.device,
        }

    def set_params(self, **params) -> MoiraiAdapter:
        """
        Set adapter parameters. Resets the module and forecast object when
        `model_id` or `context_length` changes.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `device`.

        Returns
        -------
        self : MoiraiAdapter

        """

        valid = {'model_id', 'context_length', 'device'}
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for MoiraiAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )
        if params.keys() & {'model_id', 'context_length', 'device'}:
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

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: Any,
    ) -> MoiraiAdapter:
        """
        Store the training series.
        No model training occurs since Moirai is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : Any
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : MoiraiAdapter

        """

        self.context_ = context
        self.is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: Any,
        exog: Any,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using Moirai.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict pandas Series
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : Any
            Not used, present here for API consistency by convention.
        exog : Any
            Not used, present here for API consistency by convention.
        quantiles : list of float or None
            Quantile levels. Must be a subset of `SUPPORTED_QUANTILES`.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)`.

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
                        f"Moirai only supports quantile levels "
                        f"{self.SUPPORTED_QUANTILES}. Got {q!r}. "
                        f"Quantile interpolation is not supported."
                    )
        else:
            quantile_list = None

        quantile_levels = quantile_list if quantile_list is not None else [0.5]
        q_indices = [
            next(
                i for i, sq in enumerate(self.SUPPORTED_QUANTILES)
                if abs(q - sq) < 1e-9
            )
            for q in quantile_levels
        ]

        series_names_in = list(context.keys())
        inputs_list = [
            context[name].to_numpy(dtype=np.float32).reshape(-1, 1)
            for name in series_names_in
        ]

        raw = self._run_inference(inputs_list, steps)

        predictions: dict[str, np.ndarray] = {}
        for i, name in enumerate(series_names_in):
            predictions[name] = raw[i][q_indices, :].T  # (steps, n_quantiles)

        return predictions

    def _load_module(self) -> None:
        """
        Load the `Moirai2Module` into `self._module` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `uni2ts` is not installed.

        Notes
        -----
        The module is imported lazily from `uni2ts` and instantiated via
        `Moirai2Module.from_pretrained`, then set to evaluation mode.
        This method is a no-op when `self._module` is already populated.
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

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `uni2ts` is not installed.

        Notes
        -----
        Calls `_load_module` then wraps `self._module` in a
        `Moirai2Forecast` with `prediction_length=1` (overridden
        per-call via `hparams_context`), sets it to evaluation mode,
        and moves it to the device specified by `self.device`.
        This method is a no-op when `self._forecast_obj` is already
        populated.
        """

        if self._forecast_obj is not None:
            return
        
        self._load_module()
        from uni2ts.model.moirai2 import Moirai2Forecast

        self._forecast_obj = Moirai2Forecast(
            module                     = self._module,
            prediction_length          = 1,
            context_length             = self.context_length,
            target_dim                 = 1,
            feat_dynamic_real_dim      = 0,
            past_feat_dynamic_real_dim = 0,
        ).eval()

        resolved_device = _resolve_torch_device(self.device)
        if resolved_device == "mps":
            warnings.warn(
                "MPS device is not supported by Moirai because the uni2ts "
                "library uses float64 operations internally. Falling back "
                "to CPU.",
                stacklevel=6,
            )
            resolved_device = "cpu"
        self._forecast_obj.to(resolved_device)

    def _run_inference(
        self,
        inputs_list: list[np.ndarray],
        steps: int,
    ) -> np.ndarray:
        """
        Run batched inference with `Moirai2Forecast`.

        Parameters
        ----------
        inputs_list : list of numpy ndarray
            List of 2-D arrays with shape `(T, 1)`, one per series.
            Each array holds `float32` values.
        steps : int
            Forecast horizon.

        Returns
        -------
        raw : numpy ndarray
            Array of shape `(n_series, 9, steps)` containing quantile
            forecasts for the 9 fixed levels in `SUPPORTED_QUANTILES`
            order.
        
        """

        self._ensure_forecast_obj()
        with self._forecast_obj.hparams_context(prediction_length=steps):
            raw = self._forecast_obj.predict(inputs_list)
        
        return raw


class TabICLAdapter:
    """
    Adapter for TabICL zero-shot time-series foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"soda-inria/tabicl"`.
    model : object, default None
        Pre-instantiated `TabICLForecaster` instance. If `None`, a new
        instance is created lazily on the first call to `predict`. Intended
        for testing only.
    context_length : int, default 4096
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Must be a positive integer.
    point_estimate : str, default 'mean'
        Method used to derive the point forecast from the TabICL output.
        Accepted values: `'mean'`, `'median'`.
    tabicl_config : dict, default None
        Additional keyword arguments forwarded verbatim to
        `TabICLRegressor` at inference time. If `None`, defaults to empty
        dict (TabICL's own defaults).
    temporal_features : list, default None
        List of `TimeTransform` instances applied to the time series before
        inference. If `None`, TabICL uses its default transforms:
        `[IndexEncoder(), DatetimeEncoder(), AutoPeriodicEncoder()]`. Pass
        an empty list to disable all temporal feature engineering.
    show_progress : bool, default False
        If `False`, the tqdm progress bar emitted by the underlying TabICL
        dispatch loop (`GPU 0: ...`) is suppressed.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    point_estimate : str
        Point forecast method.
    tabicl_config : dict
        Additional configuration forwarded to `TabICLRegressor`.
    temporal_features : list
        Temporal feature transforms applied to the series.
    show_progress : bool
        Whether the TabICL dispatch progress bar is shown.
    is_fitted : bool
        Whether the adapter has been fitted.
    _model : object
        Internal `TabICLForecaster` instance. `None` until the first call
        to `predict`, after which it is cached for reuse.

    Notes
    -----
    TabICL supports arbitrary quantile levels (any float in `[0, 1]`),
    unlike models with fixed quantile sets such as TimesFM or Moirai.

    Covariate support is available: extra columns in `context` and `exog`
    are forwarded as covariates. TabICL uses only the intersection of columns
    present in both context and future data (missing values are filled with
    `NaN`).

    Series with a `RangeIndex` are accepted. Internally, TabICL requires
    datetime timestamps, so a synthetic daily `DatetimeIndex` (starting
    2000-01-01) is used. Calendar-based transforms
    (`DatetimeEncoder`, `AutoPeriodicEncoder`) will not be meaningful for
    such series; consider passing `temporal_features=[]` or
    `temporal_features=[IndexEncoder()]` in that case.

    References
    ----------
    .. [1] https://github.com/soda-inria/tabicl

    .. [2] https://tabicl.readthedocs.io/en/latest/

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 4096,
        point_estimate: str = "mean",
        tabicl_config: dict[str, Any] | None = None,
        temporal_features: list[Any] | None = None,
        show_progress: bool = False,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. `"soda-inria/tabicl"`.
        model : object, default None
            Pre-instantiated `TabICLForecaster` instance. If `None`, a new
            instance is created lazily on the first call to `predict`.
            Intended for testing only.
        context_length : int, default 4096
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if
            `context` is longer than `context_length` it is trimmed to
            this length before inference; if it is shorter, all available
            observations are passed as-is. Must be a positive integer.
        point_estimate : str, default 'mean'
            Method used to derive the point forecast. Accepted values:
            `'mean'`, `'median'`.
        tabicl_config : dict, default None
            Additional keyword arguments forwarded verbatim to
            `TabICLRegressor` at inference time.
        temporal_features : list, default None
            List of `TimeTransform` instances applied before inference. If
            `None`, TabICL uses its defaults. Pass `[]` to disable all
            temporal feature engineering.
        show_progress : bool, default False
            If `False`, the tqdm progress bar emitted by the underlying
            TabICL dispatch loop (`GPU 0: ...`) is suppressed by
            redirecting stderr during the `predict_df` call.

        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )
        if point_estimate not in ("mean", "median"):
            raise ValueError(
                f"`point_estimate` must be 'mean' or 'median'. Got {point_estimate!r}."
            )

        self.model_id          = model_id
        self._model            = model
        self.context_          = None
        self.context_exog_     = None
        self.context_length    = context_length
        self.point_estimate    = point_estimate
        self.tabicl_config     = dict(tabicl_config) if tabicl_config else {}
        self.temporal_features = temporal_features
        self.show_progress     = show_progress
        self.is_fitted         = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `point_estimate`,
            `tabicl_config`, `temporal_features`, `show_progress`.
            `tabicl_config` is returned as `None` when no additional
            config was set (i.e. when the internal dict is empty).

        """
        return {
            "model_id":          self.model_id,
            "context_length":    self.context_length,
            "point_estimate":    self.point_estimate,
            "tabicl_config":     self.tabicl_config or None,
            "temporal_features": self.temporal_features,
            "show_progress":     self.show_progress,
        }

    def set_params(self, **params) -> TabICLAdapter:
        """
        Set adapter parameters. Resets the model when a parameter that affects
        the `TabICLForecaster` instance changes; toggling `show_progress` does
        not reset the model.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `point_estimate`,
            `tabicl_config`, `temporal_features`, `show_progress`.

        Returns
        -------
        self : TabICLAdapter

        """

        valid = {
            "model_id", "context_length", "point_estimate",
            "tabicl_config", "temporal_features", "show_progress",
        }
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for TabICLAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )

        validated = {}
        for key, value in params.items():
            if key == "context_length":
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                validated[key] = value
            elif key == "point_estimate":
                if value not in ("mean", "median"):
                    raise ValueError(
                        f"`point_estimate` must be 'mean' or 'median'. Got {value!r}."
                    )
                validated[key] = value
            elif key == "tabicl_config":
                validated[key] = dict(value) if value else {}
            elif key == "show_progress":
                if not isinstance(value, bool):
                    raise ValueError(
                        f"`show_progress` must be a bool. Got {value!r}."
                    )
                validated[key] = value
            else:
                validated[key] = value

        model_reset_keys = {"model_id", "context_length", "point_estimate", "tabicl_config", "temporal_features"}
        actually_changed = {
            k: v for k, v in validated.items()
            if getattr(self, k) != v
        }
        if actually_changed:
            if actually_changed.keys() & model_reset_keys:
                self._model = None
            for key, value in actually_changed.items():
                setattr(self, key, value)

        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> TabICLAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since TabICL is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : TabICLAdapter

        """

        self.context_      = context
        self.context_exog_ = context_exog
        self.is_fitted     = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using TabICL.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict pandas Series
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series past covariates (already trimmed).
        exog : dict pandas DataFrame, pandas Series, or None
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return. If `None`, a point forecast is
            produced (shape `(steps, 1)`). Accepts any float in `[0, 1]`.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D numpy ndarray of
            shape `(steps, n_quantiles)`.

        """

        self._load_model()

        quantile_list = list(quantiles) if quantiles is not None else None
        tabicl_quantiles = (
            quantile_list
            if quantile_list is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        series_names_in = list(context.keys())

        first_series = next(iter(context.values()))
        is_datetime = isinstance(first_series.index, pd.DatetimeIndex)

        if not is_datetime:
            warnings.warn(
                "TabICLAdapter received series with a non-DatetimeIndex. "
                "TabICL requires datetime timestamps internally; a synthetic "
                "daily DatetimeIndex (starting 2000-01-01) will be used. "
                "Calendar-based temporal features (DatetimeEncoder, "
                "AutoPeriodicEncoder) will not be meaningful for "
                "integer-indexed data. Consider passing "
                "`temporal_features=[]` to disable calendar feature "
                "transforms.",
                # stacklevel=3: TabICLAdapter.predict → FoundationModel.predict → user
                stacklevel=3,
            )

        context_df = self._build_context_df(
                         series_names = series_names_in, 
                         context      = context, 
                         context_exog = context_exog, 
                         is_datetime  = is_datetime
                     )
        
        future_df = self._build_future_df(
                        series_names = series_names_in, 
                        context      = context, 
                        exog         = exog, 
                        steps        = steps, 
                        is_datetime  = is_datetime
                    )

        _stderr_cm = (
            contextlib.redirect_stderr(io.StringIO())
            if not self.show_progress
            else contextlib.nullcontext()
        )
        with _stderr_cm:
            result_df = self._model.predict_df(
                            context_df = context_df,
                            future_df  = future_df,
                            quantiles  = tabicl_quantiles,
                        )

        # result_df is a plain DataFrame with MultiIndex (item_id, timestamp).
        # columns: "target" (str) and quantile levels as float column names.
        predictions: dict[str, np.ndarray] = {}
        for name in series_names_in:
            group = result_df.loc[name]  # DataFrame indexed by timestamp
            if quantile_list is None:
                predictions[name] = group["target"].to_numpy().reshape(-1, 1)
            else:
                predictions[name] = group[quantile_list].to_numpy()

        return predictions

    def _load_model(self) -> None:
        """
        Load the `TabICLForecaster` into `self._model` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `tabicl[forecast]` is not installed.

        Notes
        -----
        The model is imported lazily from `tabicl` and instantiated with
        the current adapter parameters. This method is a no-op when
        `self._model` is already populated (either by a prior call or by
        the `model` test-injection parameter).
        """

        if self._model is not None:
            return
        try:
            from tabicl.forecast import TabICLForecaster
        except ImportError as exc:
            raise ImportError(
                "tabicl[forecast] is required for TabICLAdapter. "
                "Install it with `pip install tabicl[forecast]`."
            ) from exc
        
        self._model = TabICLForecaster(
                          max_context_length = self.context_length,
                          temporal_features  = self.temporal_features,
                          point_estimate     = self.point_estimate,
                          tabicl_config      = self.tabicl_config or {},
                      )

    def _get_timestamps(
        self, series: pd.Series, is_datetime: bool
    ) -> pd.DatetimeIndex:
        """
        Return datetime timestamps for a context series.

        For `DatetimeIndex` series the original index is returned. For
        `RangeIndex` series a synthetic daily `DatetimeIndex` starting at
        2000-01-01 is created so that TabICL's requirement for datetime
        timestamps is satisfied.

        Parameters
        ----------
        series : pandas Series
            The context series.
        is_datetime : bool
            Whether the series has a `DatetimeIndex`.

        Returns
        -------
        timestamps : pandas DatetimeIndex
            Datetime timestamps aligned with the series values.

        """

        if is_datetime:
            return series.index
        
        return pd.date_range("2000-01-01", periods=len(series), freq="D")

    def _get_future_timestamps(
        self, series: pd.Series, steps: int, is_datetime: bool
    ) -> pd.DatetimeIndex:
        """
        Return datetime timestamps for the forecast horizon.

        For `DatetimeIndex` series the horizon is appended at the inferred
        frequency. For `RangeIndex` series the synthetic daily timeline
        (2000-01-01 + len(context) days) is extended by `steps` days.

        Parameters
        ----------
        series : pandas Series
            The context series (used to determine the end timestamp and
            frequency).
        steps : int
            Number of steps ahead.
        is_datetime : bool
            Whether the series has a `DatetimeIndex`.

        Returns
        -------
        timestamps : pandas DatetimeIndex
            Datetime timestamps for the `steps` forecast steps.

        """

        if is_datetime:
            freq = series.index.freq
            if freq is None:
                freq = pd.tseries.frequencies.to_offset(
                    pd.infer_freq(series.index)
                )
            timestamps = pd.date_range(
                             start   = series.index[-1] + freq,
                             periods = steps,
                             freq    = freq,
                         )
        else:
            n = len(series)
            timestamps = pd.date_range(
                             start   = pd.Timestamp("2000-01-01") + pd.Timedelta(days=n),
                             periods = steps,
                             freq    = "D",
                         )
        
        return timestamps

    def _build_context_df(
        self,
        series_names: list,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | None] | None,
        is_datetime: bool,
    ) -> pd.DataFrame:
        """
        Build a long-format context DataFrame expected by TabICL.

        Each series' observations become rows with `item_id`, `timestamp`,
        `target`, and optional exogenous covariate columns.

        Parameters
        ----------
        series_names : list
            Ordered list of series names.
        context : dict pandas Series
            Per-series context windows.
        context_exog : dict or None
            Per-series historical exogenous variables.
        is_datetime : bool
            Whether the series have a `DatetimeIndex`.

        Returns
        -------
        context_df : pandas DataFrame
            Long-format DataFrame with columns `item_id`, `timestamp`,
            `target`, and any exogenous columns.

        """

        context_df = []
        for name in series_names:
            series = context[name]
            n = len(series)
            part = pd.DataFrame({
                "item_id":   np.full(n, name),
                "timestamp": np.asarray(self._get_timestamps(series, is_datetime)),
                "target":    series.to_numpy(dtype=float),
            })
            exog_entry = (
                context_exog.get(name) if context_exog is not None else None
            )
            if exog_entry is not None:
                part = pd.concat(
                    [part, exog_entry.reset_index(drop=True)], axis=1
                )
            context_df.append(part)

        context_df = pd.concat(context_df, ignore_index=True)

        return context_df

    def _build_future_df(
        self,
        series_names: list,
        context: dict[str, pd.Series],
        exog: dict[str, pd.DataFrame | None] | None,
        steps: int,
        is_datetime: bool,
    ) -> pd.DataFrame:
        """
        Build a long-format future DataFrame expected by TabICL.

        Each series' forecast horizon becomes rows with `item_id`,
        `timestamp`, and optional future exogenous covariate columns.

        Parameters
        ----------
        series_names : list
            Ordered list of series names.
        context : dict pandas Series
            Per-series context windows (used to derive future timestamps).
        exog : dict or None
            Per-series future exogenous variables covering the forecast
            horizon.
        steps : int
            Number of steps ahead.
        is_datetime : bool
            Whether the series have a `DatetimeIndex`.

        Returns
        -------
        future_df : pandas DataFrame
            Long-format DataFrame with columns `item_id`, `timestamp`, and
            any future exogenous columns.

        """

        future_df = []
        for name in series_names:
            series = context[name]
            part = pd.DataFrame({
                "item_id":   np.full(steps, name),
                "timestamp": np.asarray(
                    self._get_future_timestamps(series, steps, is_datetime)
                ),
            })
            future_exog = exog.get(name) if exog is not None else None
            if future_exog is not None:
                part = pd.concat(
                    [part, future_exog.reset_index(drop=True)], axis=1
                )
            future_df.append(part)

        future_df = pd.concat(future_df, ignore_index=True)

        return future_df


class TabPFNAdapter:
    """
    Adapter for Prior Labs TabPFN-TS zero-shot time-series foundation models.

    TabPFN-TS frames forecasting as tabular regression: the series is
    featurized (running index, calendar features, automatically detected
    seasonal features) and a TabPFN regressor predicts the forecast horizon
    zero-shot.

    Parameters
    ----------
    model_id : str
        Model ID, e.g. `"priorlabs/tabpfn-ts"`. Used only to resolve this
        adapter; the underlying checkpoint is controlled by
        `tabpfn_model_config` (key `model_path`).
    model : object, default None
        Pre-instantiated `TabPFNTSPipeline` instance. If `None`, a new
        instance is created lazily on the first call to `predict`. Intended
        for testing only.
    context_length : int, default 32768
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Defaults to 32768, which matches the TabPFN-TS ship
        configuration; lower values (e.g. 4096) speed up inference at a small
        accuracy cost. Must be a positive integer.
    mode : str, default 'local'
        Inference mode. `'local'` runs the TabPFN model locally (CUDA > MPS >
        CPU selected automatically by the library; the checkpoint is
        downloaded on first use). `'client'` sends the featurized data to the
        Prior Labs cloud API via `tabpfn-client` (no GPU needed, requires an
        account/API key).
    point_estimate : str, default 'median'
        Method used to aggregate the TabPFN ensemble output into the point
        forecast. Accepted values: `'mean'`, `'median'`, `'mode'`.
    tabpfn_model_config : dict, default None
        Additional configuration forwarded verbatim to the underlying TabPFN
        regressor (e.g. `model_path`, `device`). If `None`, the library
        defaults are used.
    temporal_features : list, default None
        List of `FeatureGenerator` instances applied to the time series
        before inference. If `None`, TabPFN-TS uses its default transforms:
        `[RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]`.
        Pass an empty list to disable all temporal feature engineering.
    show_progress : bool, default False
        If `False`, the tqdm progress bar emitted by the underlying TabPFN-TS
        dispatch loop (`Predicting time series: ...` on CPU, `GPU 0: ...` on
        GPU) is suppressed.

    Attributes
    ----------
    model_id : str
        Model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    mode : str
        Inference mode, `'local'` or `'client'`.
    point_estimate : str
        Point forecast aggregation method.
    tabpfn_model_config : dict
        Additional configuration forwarded to the TabPFN regressor.
    temporal_features : list
        Temporal feature transforms applied to the series.
    show_progress : bool
        Whether the tqdm progress bar is shown during inference.
    is_fitted : bool
        Whether the adapter has been fitted.
    _model : object
        Internal `TabPFNTSPipeline` instance. `None` until the first call
        to `predict`, after which it is cached for reuse.

    Notes
    -----
    TabPFN-TS supports arbitrary quantile levels (any float in `(0, 1)`),
    unlike models with fixed quantile sets such as TimesFM or Moirai.

    Covariate support is available for *known-future* covariates: extra
    columns present in both the historical context and the forecast horizon
    are used by the model. Covariates without future values are discarded by
    the library.

    Series with a `RangeIndex` are accepted. Internally, TabPFN-TS requires
    datetime timestamps, so a synthetic daily `DatetimeIndex` (starting
    2000-01-01) is used. Calendar-based transforms (`CalendarFeature`) will
    not be meaningful for such series; consider passing
    `temporal_features=[]` or `[RunningIndexFeature()]` in that case.

    References
    ----------
    .. [1] https://github.com/PriorLabs/tabpfn-time-series

    .. [2] https://priorlabs.ai/

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 32768,
        mode: str = "local",
        point_estimate: str = "median",
        tabpfn_model_config: dict[str, Any] | None = None,
        temporal_features: list[Any] | None = None,
        show_progress: bool = False,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            Model ID, e.g. `"priorlabs/tabpfn-ts"`.
        model : object, default None
            Pre-instantiated `TabPFNTSPipeline` instance. If `None`, a new
            instance is created lazily on the first call to `predict`.
            Intended for testing only.
        context_length : int, default 32768
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if
            `context` is longer than `context_length` it is trimmed to
            this length before inference; if it is shorter, all available
            observations are passed as-is. Must be a positive integer.
        mode : str, default 'local'
            Inference mode. Accepted values: `'local'`, `'client'`.
        point_estimate : str, default 'median'
            Method used to aggregate the TabPFN ensemble output into the
            point forecast. Accepted values: `'mean'`, `'median'`, `'mode'`.
        tabpfn_model_config : dict, default None
            Additional configuration forwarded verbatim to the underlying
            TabPFN regressor.
        temporal_features : list, default None
            List of `FeatureGenerator` instances applied before inference.
            If `None`, TabPFN-TS uses its defaults. Pass `[]` to disable all
            temporal feature engineering.
        show_progress : bool, default False
            If `False`, the tqdm progress bar emitted by the underlying
            TabPFN-TS dispatch loop (`Predicting time series: ...` on CPU,
            `GPU 0: ...` on GPU) is suppressed.

        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )
        if mode not in ("local", "client"):
            raise ValueError(
                f"`mode` must be 'local' or 'client'. Got {mode!r}."
            )
        if point_estimate not in ("mean", "median", "mode"):
            raise ValueError(
                f"`point_estimate` must be 'mean', 'median' or 'mode'. "
                f"Got {point_estimate!r}."
            )

        self.model_id            = model_id
        self._model              = model
        self.context_            = None
        self.context_exog_       = None
        self.context_length      = context_length
        self.mode                = mode
        self.point_estimate      = point_estimate
        self.tabpfn_model_config = dict(tabpfn_model_config) if tabpfn_model_config else {}
        self.temporal_features   = temporal_features
        self.show_progress       = show_progress
        self.is_fitted           = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `mode`, `point_estimate`,
            `tabpfn_model_config`, `temporal_features`, `show_progress`.
            `tabpfn_model_config` is returned as `None` when no additional
            config was set (i.e. when the internal dict is empty).

        """
        return {
            "model_id":            self.model_id,
            "context_length":      self.context_length,
            "mode":                self.mode,
            "point_estimate":      self.point_estimate,
            "tabpfn_model_config": self.tabpfn_model_config or None,
            "temporal_features":   self.temporal_features,
            "show_progress":       self.show_progress,
        }

    def set_params(self, **params) -> TabPFNAdapter:
        """
        Set adapter parameters. Resets the model when a parameter that affects
        the `TabPFNTSPipeline` instance changes; toggling `show_progress` does
        not reset the model.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `mode`,
            `point_estimate`, `tabpfn_model_config`, `temporal_features`,
            `show_progress`.

        Returns
        -------
        self : TabPFNAdapter

        """

        valid = {
            "model_id", "context_length", "mode", "point_estimate",
            "tabpfn_model_config", "temporal_features", "show_progress",
        }
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for TabPFNAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )

        validated = {}
        for key, value in params.items():
            if key == "context_length":
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                validated[key] = value
            elif key == "mode":
                if value not in ("local", "client"):
                    raise ValueError(
                        f"`mode` must be 'local' or 'client'. Got {value!r}."
                    )
                validated[key] = value
            elif key == "point_estimate":
                if value not in ("mean", "median", "mode"):
                    raise ValueError(
                        f"`point_estimate` must be 'mean', 'median' or 'mode'. "
                        f"Got {value!r}."
                    )
                validated[key] = value
            elif key == "tabpfn_model_config":
                validated[key] = dict(value) if value else {}
            elif key == "show_progress":
                if not isinstance(value, bool):
                    raise ValueError(
                        f"`show_progress` must be a bool. Got {value!r}."
                    )
                validated[key] = value
            else:
                validated[key] = value

        model_reset_keys = {
            "model_id", "context_length", "mode", "point_estimate",
            "tabpfn_model_config", "temporal_features",
        }
        actually_changed = {
            k: v for k, v in validated.items()
            if getattr(self, k) != v
        }
        if actually_changed:
            if actually_changed.keys() & model_reset_keys:
                self._model = None
            for key, value in actually_changed.items():
                setattr(self, key, value)

        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> TabPFNAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since TabPFN-TS is a zero-shot inference
        model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : TabPFNAdapter

        """

        self.context_      = context
        self.context_exog_ = context_exog
        self.is_fitted     = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using TabPFN-TS.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict pandas Series
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series past covariates (already trimmed).
        exog : dict pandas DataFrame, pandas Series, or None
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return. If `None`, a point forecast is
            produced (shape `(steps, 1)`). Accepts any float in `[0, 1]`.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D numpy ndarray of
            shape `(steps, n_quantiles)`.

        """

        self._load_model()

        quantile_list = list(quantiles) if quantiles is not None else None
        tabpfn_quantiles = (
            quantile_list
            if quantile_list is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        series_names_in = list(context.keys())

        first_series = next(iter(context.values()))
        is_datetime = isinstance(first_series.index, pd.DatetimeIndex)

        if not is_datetime:
            warnings.warn(
                "TabPFNAdapter received series with a non-DatetimeIndex. "
                "TabPFN-TS requires datetime timestamps internally; a "
                "synthetic daily DatetimeIndex (starting 2000-01-01) will be "
                "used. Calendar-based temporal features (CalendarFeature) "
                "will not be meaningful for integer-indexed data. Consider "
                "passing `temporal_features=[]` to disable calendar feature "
                "transforms.",
                # stacklevel=3: TabPFNAdapter.predict → FoundationModel.predict → user
                stacklevel=3,
            )

        context_df = self._build_context_df(
                         series_names = series_names_in,
                         context      = context,
                         context_exog = context_exog,
                         is_datetime  = is_datetime
                     )

        future_df = self._build_future_df(
                        series_names = series_names_in,
                        context      = context,
                        exog         = exog,
                        steps        = steps,
                        is_datetime  = is_datetime
                    )

        _stderr_cm = (
            contextlib.redirect_stderr(io.StringIO())
            if not self.show_progress
            else contextlib.nullcontext()
        )
        with _stderr_cm:
            result_df = self._model.predict_df(
                            context_df = context_df,
                            future_df  = future_df,
                            quantiles  = tabpfn_quantiles,
                        )

        # result_df is a DataFrame with MultiIndex (item_id, timestamp).
        # columns: "target" (str) and quantile levels as float column names.
        predictions: dict[str, np.ndarray] = {}
        for name in series_names_in:
            group = result_df.loc[name]  # DataFrame indexed by timestamp
            if quantile_list is None:
                predictions[name] = group["target"].to_numpy().reshape(-1, 1)
            else:
                predictions[name] = group[quantile_list].to_numpy()

        return predictions

    def _load_model(self) -> None:
        """
        Load the `TabPFNTSPipeline` into `self._model` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `tabpfn-time-series` is not installed.

        Notes
        -----
        The pipeline is imported lazily from `tabpfn_time_series` and
        instantiated with the current adapter parameters. This method is a
        no-op when `self._model` is already populated (either by a prior
        call or by the `model` test-injection parameter).
        """

        if self._model is not None:
            return
        try:
            from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
        except ImportError as exc:
            raise ImportError(
                "tabpfn-time-series is required for TabPFNAdapter. "
                "Install it with `pip install tabpfn-time-series`."
            ) from exc

        kwargs: dict[str, Any] = {
            "max_context_length": self.context_length,
            "tabpfn_mode": (
                TabPFNMode.LOCAL if self.mode == "local" else TabPFNMode.CLIENT
            ),
            "tabpfn_output_selection": self.point_estimate,
        }
        if self.tabpfn_model_config:
            kwargs["tabpfn_model_config"] = self.tabpfn_model_config
        if self.temporal_features is not None:
            kwargs["temporal_features"] = self.temporal_features

        self._model = TabPFNTSPipeline(**kwargs)

    def _get_timestamps(
        self, series: pd.Series, is_datetime: bool
    ) -> pd.DatetimeIndex:
        """
        Return datetime timestamps for a context series.

        For `DatetimeIndex` series the original index is returned. For
        `RangeIndex` series a synthetic daily `DatetimeIndex` starting at
        2000-01-01 is created so that TabPFN-TS's requirement for datetime
        timestamps is satisfied.

        Parameters
        ----------
        series : pandas Series
            The context series.
        is_datetime : bool
            Whether the series has a `DatetimeIndex`.

        Returns
        -------
        timestamps : pandas DatetimeIndex
            Datetime timestamps aligned with the series values.

        """

        if is_datetime:
            return series.index

        return pd.date_range("2000-01-01", periods=len(series), freq="D")

    def _get_future_timestamps(
        self, series: pd.Series, steps: int, is_datetime: bool
    ) -> pd.DatetimeIndex:
        """
        Return datetime timestamps for the forecast horizon.

        For `DatetimeIndex` series the horizon is appended at the inferred
        frequency. For `RangeIndex` series the synthetic daily timeline
        (2000-01-01 + len(context) days) is extended by `steps` days.

        Parameters
        ----------
        series : pandas Series
            The context series (used to determine the end timestamp and
            frequency).
        steps : int
            Number of steps ahead.
        is_datetime : bool
            Whether the series has a `DatetimeIndex`.

        Returns
        -------
        timestamps : pandas DatetimeIndex
            Datetime timestamps for the `steps` forecast steps.

        """

        if is_datetime:
            freq = series.index.freq
            if freq is None:
                freq = pd.tseries.frequencies.to_offset(
                    pd.infer_freq(series.index)
                )
            timestamps = pd.date_range(
                             start   = series.index[-1] + freq,
                             periods = steps,
                             freq    = freq,
                         )
        else:
            n = len(series)
            timestamps = pd.date_range(
                             start   = pd.Timestamp("2000-01-01") + pd.Timedelta(days=n),
                             periods = steps,
                             freq    = "D",
                         )

        return timestamps

    def _build_context_df(
        self,
        series_names: list,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | None] | None,
        is_datetime: bool,
    ) -> pd.DataFrame:
        """
        Build a long-format context DataFrame expected by TabPFN-TS.

        Each series' observations become rows with `item_id`, `timestamp`,
        `target`, and optional exogenous covariate columns.

        Parameters
        ----------
        series_names : list
            Ordered list of series names.
        context : dict pandas Series
            Per-series context windows.
        context_exog : dict or None
            Per-series historical exogenous variables.
        is_datetime : bool
            Whether the series have a `DatetimeIndex`.

        Returns
        -------
        context_df : pandas DataFrame
            Long-format DataFrame with columns `item_id`, `timestamp`,
            `target`, and any exogenous columns.

        """

        context_df = []
        for name in series_names:
            series = context[name]
            n = len(series)
            part = pd.DataFrame({
                "item_id":   np.full(n, name),
                "timestamp": np.asarray(self._get_timestamps(series, is_datetime)),
                "target":    series.to_numpy(dtype=float),
            })
            exog_entry = (
                context_exog.get(name) if context_exog is not None else None
            )
            if exog_entry is not None:
                part = pd.concat(
                    [part, exog_entry.reset_index(drop=True)], axis=1
                )
            context_df.append(part)

        context_df = pd.concat(context_df, ignore_index=True)

        return context_df

    def _build_future_df(
        self,
        series_names: list,
        context: dict[str, pd.Series],
        exog: dict[str, pd.DataFrame | None] | None,
        steps: int,
        is_datetime: bool,
    ) -> pd.DataFrame:
        """
        Build a long-format future DataFrame expected by TabPFN-TS.

        Each series' forecast horizon becomes rows with `item_id`,
        `timestamp`, and optional future exogenous covariate columns.

        Parameters
        ----------
        series_names : list
            Ordered list of series names.
        context : dict pandas Series
            Per-series context windows (used to derive future timestamps).
        exog : dict or None
            Per-series future exogenous variables covering the forecast
            horizon.
        steps : int
            Number of steps ahead.
        is_datetime : bool
            Whether the series have a `DatetimeIndex`.

        Returns
        -------
        future_df : pandas DataFrame
            Long-format DataFrame with columns `item_id`, `timestamp`, and
            any future exogenous columns.

        """

        future_df = []
        for name in series_names:
            series = context[name]
            part = pd.DataFrame({
                "item_id":   np.full(steps, name),
                "timestamp": np.asarray(
                    self._get_future_timestamps(series, steps, is_datetime)
                ),
            })
            future_exog = exog.get(name) if exog is not None else None
            if future_exog is not None:
                part = pd.concat(
                    [part, future_exog.reset_index(drop=True)], axis=1
                )
            future_df.append(part)

        future_df = pd.concat(future_df, ignore_index=True)

        return future_df


class T0Adapter:
    """
    Adapter for The Forecasting Company T0 foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "theforecastingcompany/t0-alpha".
    model : T0Forecaster, default None
        Pre-loaded model instance. If `None`, the model is loaded lazily
        on the first call to `predict`.
    context_length : int, default 8192
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Must be a positive integer.
    device_map : str, default 'auto'
        Device placement for the model. `"auto"` selects the best
        available accelerator (CUDA > MPS > CPU). Also accepts explicit
        values such as `"cuda"`, `"mps"`, or `"cpu"`.
    torch_dtype : object, default None
        Torch dtype the loaded model is cast to (e.g. `torch.bfloat16`).
        When `None` the model keeps its default `float32` weights.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    device_map : str
        Device map string for model loading.
    torch_dtype : object
        Torch dtype for model loading.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    T0 conditions on covariates that are known over both the context and the
    forecast horizon (future-known covariates). skforecast exogenous variables
    map exactly onto this channel: their historical values (`context_exog`,
    aligned to the context) are concatenated with their future values (`exog`,
    aligned to the horizon) to form the `[context + horizon]` covariate stream
    that T0 expects. Covariates must be numeric; encode categoricals as numbers
    before passing them. A series with no future exog is forecast without
    covariates.

    References
    ----------
    .. [1] https://github.com/theforecastingcompany/tfc-t0
    
    .. [2] https://huggingface.co/theforecastingcompany/t0-alpha

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 8192,
        device_map: str = "auto",
        torch_dtype: Any | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. "theforecastingcompany/t0-alpha".
        model : T0Forecaster, default None
            Pre-loaded model instance. If `None`, the model is loaded
            lazily on the first call to `predict`.
        context_length : int, default 8192
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if `context`
            is longer than `context_length` it is trimmed to this length
            before inference; if it is shorter, all available observations
            are passed as-is. Must be a positive integer.
        device_map : str, default 'auto'
            Device placement for the model. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU). Also accepts explicit
            values such as `"cuda"`, `"mps"`, or `"cpu"`.
        torch_dtype : object, default None
            Torch dtype the loaded model is cast to (e.g. `torch.bfloat16`).
            When `None` the model keeps its default `float32` weights.

        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )

        self.model_id       = model_id
        self._model         = model
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.device_map     = device_map
        self.torch_dtype    = torch_dtype
        self.is_fitted      = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `device_map`, `torch_dtype`.

        """
        return {
            'model_id':       self.model_id,
            'context_length': self.context_length,
            'device_map':     self.device_map,
            'torch_dtype':    self.torch_dtype,
        }

    def set_params(self, **params) -> T0Adapter:
        """
        Set adapter parameters. Resets the model when a device, dtype, or
        model_id param changes, since those are baked into the loaded model.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `device_map`,
            `torch_dtype`.

        Returns
        -------
        self : T0Adapter

        """

        valid = {'model_id', 'context_length', 'device_map', 'torch_dtype'}
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for T0Adapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )

        model_reset_keys = {'model_id', 'device_map', 'torch_dtype'}
        if params.keys() & model_reset_keys:
            self._model = None

        for key, value in params.items():
            if key == 'context_length':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                self.context_length = value
            else:
                setattr(self, key, value)

        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
    ) -> T0Adapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since T0 is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : T0Adapter

        """

        self.context_ = context
        self.context_exog_ = context_exog
        self.is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
        exog: dict[str, pd.DataFrame | pd.Series | None],
        quantiles: list[float] | tuple[float] | None
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using the T0 model.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict
            Per-series context windows (already trimmed to `context_length`).
        context_exog : dict
            Per-series past covariates (already trimmed).
        exog : dict
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return, in the requested order. If `None`, a
            point forecast (median, quantile 0.5) is produced.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)` with columns ordered to match `quantiles`.

        """

        # NOTE: the model is loaded lazily here so that the adapter can be
        # instantiated and fitted without requiring tfc-t0 to be installed.
        self._load_model()

        requested = list(quantiles) if quantiles is not None else [0.5]
        # T0 requires sorted, unique levels in (0, 1); query those, then
        # reindex the columns back to the order the caller asked for.
        query_levels = sorted(set(requested))

        series_names = list(context.keys())
        arrays = [np.asarray(context[name].to_numpy(), dtype=np.float32) for name in series_names]
        lengths = [a.shape[0] for a in arrays]
        context_length = max(lengths)

        # All series are forecast in a single batched call. Series shorter than
        # the longest are left-padded with NaN, which T0 treats as MISSING; the
        # forecast origin therefore aligns at the end of the window for every
        # series.
        context_batch = np.full((len(series_names), context_length), np.nan, dtype=np.float32)
        for row, array in zip(context_batch, arrays):
            row[context_length - array.shape[0]:] = array

        future_covariates = self._build_future_covariates(
            series_names   = series_names,
            context_exog   = context_exog,
            exog           = exog,
            context_length = context_length,
            steps          = steps,
        )

        forecast = self._model.predict(
            context           = context_batch,
            horizon           = steps,
            quantiles         = query_levels,
            future_covariates = future_covariates,
        )

        q_arr = forecast.quantiles
        if hasattr(q_arr, "detach"):
            q_arr = q_arr.detach().cpu().numpy()
        else:
            q_arr = np.asarray(q_arr)

        column_for = [query_levels.index(q) for q in requested]
        return {name: q_arr[i][:, column_for] for i, name in enumerate(series_names)}

    def _load_model(self) -> None:
        """
        Load the T0 model into `self._model` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `tfc-t0` is not installed.

        Notes
        -----
        The model is imported lazily from `t0` and loaded via
        `T0Forecaster.from_pretrained`, then moved to the resolved device and
        switched to eval mode. This method is a no-op when `self._model` is
        already populated.

        """

        if self._model is not None:
            return
        try:
            from t0 import T0Forecaster
        except ImportError as exc:
            raise ImportError(
                "tfc-t0 is required for T0Adapter. "
                "Install it with `pip install tfc-t0`."
            ) from exc

        device = _resolve_torch_device(self.device_map)
        model = T0Forecaster.from_pretrained(self.model_id).to(device)
        if self.torch_dtype is not None:
            model = model.to(self.torch_dtype)
        self._model = model.eval()

    def _build_future_covariates(
        self,
        series_names: list[str],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        context_length: int,
        steps: int,
    ) -> np.ndarray | None:
        """
        Assemble T0's batched `[n_series, n_covariates, context_length + steps]`
        covariate array from per-series past and future exogenous values.

        Covariate columns are pooled across all series (first-seen order). For
        each series and column the historical values (from `context_exog`) are
        placed flush against the forecast origin and the future values (from
        `exog`) cover the horizon. Every unfilled cell — a padded timestep, or a
        column/series that lacks that covariate — stays NaN, which T0 treats as
        missing.

        Parameters
        ----------
        series_names : list of str
            Series order defining the batch rows.
        context_exog : dict or None
            Per-series historical exogenous values aligned to each context.
        exog : dict or None
            Per-series future-known exogenous values covering the horizon.
        context_length : int
            Width of the (left-padded) context window.
        steps : int
            Number of forecast steps.

        Returns
        -------
        future_covariates : numpy ndarray or None
            Array of shape `(n_series, n_covariates, context_length + steps)`,
            or `None` when no series has future exog.

        """

        if exog is None:
            return None

        future_frames = {
            name: (e if isinstance(e, pd.DataFrame) else e.to_frame())
            for name, e in exog.items()
            if e is not None
        }
        if not future_frames:
            return None

        columns: list[str] = []
        for frame in future_frames.values():
            for col in frame.columns:
                if col not in columns:
                    columns.append(col)

        total_length = context_length + steps
        covariates = np.full(
            (len(series_names), len(columns), total_length), np.nan, dtype=np.float32
        )
        column_index = {col: j for j, col in enumerate(columns)}
        for row, name in enumerate(series_names):
            future_df = future_frames.get(name)
            if future_df is None:
                continue
            past_df = None
            if context_exog is not None and context_exog.get(name) is not None:
                ctx = context_exog[name]
                past_df = ctx if isinstance(ctx, pd.DataFrame) else ctx.to_frame()
            for col in future_df.columns:
                j = column_index[col]
                future_values = self._to_float_array(future_df[col])
                covariates[row, j, context_length:context_length + future_values.shape[0]] = future_values
                if past_df is not None and col in past_df.columns:
                    past_values = self._to_float_array(past_df[col])
                    covariates[row, j, context_length - past_values.shape[0]:context_length] = past_values

        return covariates

    @staticmethod
    def _to_float_array(col_data: pd.Series) -> np.ndarray:
        """
        Convert a numeric or boolean covariate column to a `float32` array.

        Parameters
        ----------
        col_data : pandas Series
            A single covariate column.

        Returns
        -------
        col_array : numpy ndarray
            1-D `float32` array.

        Raises
        ------
        ValueError
            If the column is neither numeric nor boolean. T0 only conditions
            on numeric covariates; categoricals must be encoded as numbers.

        """

        if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            return col_data.astype(np.float32).to_numpy()

        raise ValueError(
            f"T0Adapter supports only numeric covariates. Column "
            f"{col_data.name!r} has dtype {col_data.dtype}. Encode categorical "
            f"covariates as numeric values before passing them."
        )


class NoriAdapter:
    """
    Adapter for Synthefy Nori zero-shot tabular foundation models.

    Nori is a tabular regression foundation model that predicts via in-context
    learning: given labeled context rows it predicts query rows in a single
    forward pass, with no task-specific training or fine-tuning. This adapter
    frames forecasting as tabular regression: each series is featurized (running
    index, calendar features, Fourier seasonal terms, and optional known-future
    covariates) and a ``NoriRegressor`` predicts the forecast horizon zero-shot.

    Parameters
    ----------
    model_id : str
        Model ID, e.g. ``"Synthefy/Nori"``. Used to resolve this adapter; the
        underlying checkpoint is controlled by ``nori_config`` (key
        ``model_path``) and defaults to the public Hugging Face checkpoint.
    model : object, default None
        Pre-instantiated ``NoriRegressor`` instance. If ``None``, a new instance
        is created lazily on the first call to ``predict``. Intended for testing
        only.
    context_length : int, default 4096
        Maximum number of historical observations to use as context rows. Must be
        a positive integer.
    point_estimate : str, default 'mean'
        Point forecast from Nori's predictive distribution. One of ``'mean'``,
        ``'median'``, ``'mode'``.
    add_calendar_features : bool, default True
        Add calendar features (month, day, day-of-week, day-of-year, quarter,
        hour) when the series has a ``DatetimeIndex``. Ignored for ``RangeIndex``.
    n_fourier_terms : int, default 2
        Number of Fourier (sin/cos) seasonal harmonics on the yearly and weekly
        cycles for datetime series (or on the running index for ``RangeIndex``).
        Set ``0`` to disable.
    nori_config : dict, default None
        Extra keyword arguments forwarded verbatim to ``NoriRegressor`` (e.g.
        ``model_path``, ``device``, ``token``, ``augmentations``).

    Attributes
    ----------
    model_id, context_, context_exog_, context_length, point_estimate,
    add_calendar_features, n_fourier_terms, nori_config, is_fitted, _model

    Notes
    -----
    Nori is a pure tabular regressor and does not ship a time-series
    featurization pipeline (unlike TabPFN-TS), so this adapter builds its own
    lightweight, dependency-free features. If the maintainers prefer the
    ``temporal_features`` / ``FeatureGenerator`` convention used by
    ``TabPFNAdapter``, the featurizer can be swapped to mirror it.

    Covariate support is available for *known-future* covariates: columns present
    in both the historical context and the forecast horizon are used as features.
    Covariates without future values are ignored.

    Series with a ``RangeIndex`` are accepted; only running-index and
    Fourier(index) features are meaningful there (calendar features are skipped).

    Quantiles use Nori's native pinball head and accept any tau strictly in
    ``(0, 1)``. A ``bar_distribution`` checkpoint does not support quantiles.

    References
    ----------
    .. [1] https://github.com/Synthefy/synthefy-nori

    .. [2] https://huggingface.co/Synthefy/Nori

    .. [3] https://docs.synthefy.com/nori/

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 4096,
        point_estimate: str = "mean",
        add_calendar_features: bool = True,
        n_fourier_terms: int = 2,
        nori_config: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )
        if point_estimate not in ("mean", "median", "mode"):
            raise ValueError(
                f"`point_estimate` must be 'mean', 'median' or 'mode'. "
                f"Got {point_estimate!r}."
            )
        if not isinstance(n_fourier_terms, int) or n_fourier_terms < 0:
            raise ValueError(
                f"`n_fourier_terms` must be a non-negative integer. "
                f"Got {n_fourier_terms!r}."
            )

        self.model_id              = model_id
        self._model                = model
        self.context_              = None
        self.context_exog_         = None
        self.context_length        = context_length
        self.point_estimate        = point_estimate
        self.add_calendar_features = add_calendar_features
        self.n_fourier_terms       = n_fourier_terms
        self.nori_config           = dict(nori_config) if nori_config else {}
        self.is_fitted             = False

    def get_params(self) -> dict:
        return {
            "model_id":              self.model_id,
            "context_length":        self.context_length,
            "point_estimate":        self.point_estimate,
            "add_calendar_features": self.add_calendar_features,
            "n_fourier_terms":       self.n_fourier_terms,
            "nori_config":           self.nori_config or None,
        }

    def set_params(self, **params) -> "NoriAdapter":
        valid = {
            "model_id", "context_length", "point_estimate",
            "add_calendar_features", "n_fourier_terms", "nori_config",
        }
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for NoriAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )

        validated = {}
        for key, value in params.items():
            if key == "context_length":
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                validated[key] = value
            elif key == "point_estimate":
                if value not in ("mean", "median", "mode"):
                    raise ValueError(
                        f"`point_estimate` must be 'mean', 'median' or 'mode'. "
                        f"Got {value!r}."
                    )
                validated[key] = value
            elif key == "n_fourier_terms":
                if not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"`n_fourier_terms` must be a non-negative integer. "
                        f"Got {value!r}."
                    )
                validated[key] = value
            elif key == "nori_config":
                validated[key] = dict(value) if value else {}
            else:
                validated[key] = value

        actually_changed = {
            k: v for k, v in validated.items() if getattr(self, k) != v
        }
        if actually_changed:
            self._model = None
            for key, value in actually_changed.items():
                setattr(self, key, value)

        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> "NoriAdapter":
        self.context_      = context
        self.context_exog_ = context_exog
        self.is_fitted     = True
        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        quantile_list = list(quantiles) if quantiles is not None else None
        if quantile_list is not None and any(
            (q <= 0.0) or (q >= 1.0) for q in quantile_list
        ):
            raise ValueError(
                "NoriAdapter quantiles must lie strictly in (0, 1). "
                f"Got {quantile_list!r}."
            )

        predictions: dict[str, np.ndarray] = {}
        for name, series in context.items():
            is_datetime = isinstance(series.index, pd.DatetimeIndex)
            if not is_datetime and self.add_calendar_features:
                warnings.warn(
                    "NoriAdapter received a series with a non-DatetimeIndex; "
                    "calendar features are skipped for it. Only running-index "
                    "and Fourier(index) features are used.",
                    # stacklevel=3: NoriAdapter.predict -> FoundationModel.predict -> user
                    stacklevel=3,
                )

            ctx_exog = context_exog.get(name) if context_exog is not None else None
            fut_exog = exog.get(name) if exog is not None else None
            exog_cols = self._known_future_columns(ctx_exog, fut_exog)

            X_ctx = self._featurize(
                series, is_datetime, ctx_exog, exog_cols, offset=0, n=len(series)
            )
            X_fut = self._featurize(
                series, is_datetime, fut_exog, exog_cols, offset=len(series), n=steps
            )
            y_ctx = series.to_numpy(dtype=float)

            model = self._new_model()
            model.fit(X_ctx, y_ctx)

            if quantile_list is None:
                y_hat = model.predict(X_fut, output_type=self.point_estimate)
                predictions[name] = np.asarray(y_hat, dtype=float).reshape(-1, 1)
            else:
                q = model.predict(
                    X_fut, output_type="quantiles", quantiles=quantile_list
                )
                # Nori returns (n_quantiles, steps); skforecast expects
                # (steps, n_quantiles).
                predictions[name] = np.asarray(q, dtype=float).reshape(
                    len(quantile_list), steps
                ).T

        return predictions

    # ----------------------------------------------------------------- helpers
    def _new_model(self):
        if self._model is not None:
            return self._model
        try:
            from synthefy_nori import NoriRegressor
        except ImportError as exc:
            raise ImportError(
                "synthefy-nori is required for NoriAdapter. "
                "Install it with `pip install synthefy-nori`."
            ) from exc
        return NoriRegressor(**self.nori_config)

    @staticmethod
    def _known_future_columns(ctx_exog, fut_exog) -> list:
        if ctx_exog is None or fut_exog is None:
            return []
        c = (
            ctx_exog.columns
            if isinstance(ctx_exog, pd.DataFrame)
            else pd.Index([ctx_exog.name])
        )
        f = (
            fut_exog.columns
            if isinstance(fut_exog, pd.DataFrame)
            else pd.Index([fut_exog.name])
        )
        f_set = set(f)
        return [col for col in c if col in f_set]

    def _featurize(
        self, series, is_datetime, exog_block, exog_cols, offset, n
    ) -> np.ndarray:
        idx = np.arange(offset, offset + n, dtype=float)
        feats = [idx]  # running index

        if is_datetime and (self.add_calendar_features or self.n_fourier_terms > 0):
            ts = self._timestamps(series, offset, n)
            if self.add_calendar_features:
                feats += [
                    ts.month.to_numpy(dtype=float),
                    ts.day.to_numpy(dtype=float),
                    ts.dayofweek.to_numpy(dtype=float),
                    ts.dayofyear.to_numpy(dtype=float),
                    ts.quarter.to_numpy(dtype=float),
                    ts.hour.to_numpy(dtype=float),
                ]
            doy = ts.dayofyear.to_numpy(dtype=float)
            dow = ts.dayofweek.to_numpy(dtype=float)
            for k in range(1, self.n_fourier_terms + 1):
                feats += [
                    np.sin(2 * np.pi * k * doy / 365.25),
                    np.cos(2 * np.pi * k * doy / 365.25),
                    np.sin(2 * np.pi * k * dow / 7.0),
                    np.cos(2 * np.pi * k * dow / 7.0),
                ]
        elif self.n_fourier_terms > 0:
            period = max(len(series), 1)
            for k in range(1, self.n_fourier_terms + 1):
                feats += [
                    np.sin(2 * np.pi * k * idx / period),
                    np.cos(2 * np.pi * k * idx / period),
                ]

        X = np.column_stack(feats)

        if exog_cols and exog_block is not None:
            block = (
                exog_block.to_frame()
                if isinstance(exog_block, pd.Series)
                else exog_block
            )
            X = np.column_stack([X, block[exog_cols].to_numpy(dtype=float)])

        return X.astype(np.float32)

    def _timestamps(self, series, offset, n) -> pd.DatetimeIndex:
        if offset == 0:
            return series.index
        freq = series.index.freq
        if freq is None:
            freq = pd.tseries.frequencies.to_offset(pd.infer_freq(series.index))
        return pd.date_range(start=series.index[-1] + freq, periods=n, freq=freq)


_ADAPTER_REGISTRY: dict[str, type] = {
    "amazon/chronos":    ChronosAdapter,
    "autogluon/chronos": ChronosAdapter,
    "google/timesfm":    TimesFMAdapter,
    "Salesforce/moirai": MoiraiAdapter,
    "soda-inria/tabicl": TabICLAdapter,
    "priorlabs/tabpfn":  TabPFNAdapter,
    "Synthefy/Nori":     NoriAdapter,
    "theforecastingcompany/t0": T0Adapter,
    # "ibm/TTM": TTMAdapter,
}


def _resolve_adapter(model_id: str) -> type:
    """
    Return the adapter class for *model_id* based on prefix matching.

    Parameters
    ----------
    model_id : str
        The model ID for which to find the adapter class.

    Returns
    -------
    adapter_cls : type
        The adapter class corresponding to the given model ID.

    """

    for prefix, cls in _ADAPTER_REGISTRY.items():
        if model_id.startswith(prefix):
            return cls
    
    raise ValueError(
        f"No adapter found for model '{model_id}'. "
        f"Registered prefixes: {list(_ADAPTER_REGISTRY)}."
    )
