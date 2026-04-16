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


class Chronos2Adapter:
    """
    Adapter for Amazon Chronos-2 foundation models.

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
        window of Chronos-2. Must be a positive integer.
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
        If `True`, Chronos-2 shares information across all series in
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
            maximum context window of Chronos-2. Must be a positive
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
            If `True`, Chronos-2 shares information across all series in
            the batch when predicting in multi-series mode. Forwarded
            directly to `predict_quantiles`. Ignored in single-series mode.
        
        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )

        # TODO: Ver qué atributos podemos sacar de dentro del adapter
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

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
    ) -> Chronos2Adapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since Chronos-2 is a zero-shot inference model.

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
        self : Chronos2Adapter

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
        Generate predictions using the Chronos-2 pipeline.

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
        # instantiated and fitted without requiring Chronos-2 to be installed.
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
        Load the Chronos-2 pipeline into `self._pipeline` if not already set.

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
        as-is so that Chronos-2 can handle them as categorical covariates
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
            as-is and handled natively by Chronos-2.
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
                col: Chronos2Adapter._to_covariate_array(df[col]) for col in df.columns
            }
        if exog is not None:
            df = (
                exog
                if isinstance(exog, pd.DataFrame)
                else exog.to_frame()
            )
            input_dict["future_covariates"] = {
                col: Chronos2Adapter._to_covariate_array(df[col]) for col in df.columns
            }
        
        return input_dict


class TimesFM25Adapter:
    """
    Adapter for Google TimesFM 2.5 foundation models.

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
        512. TimesFM 2.5 supports up to 16_384.
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
    TimesFM 2.5 supports only the fixed quantile levels
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

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: Any,
    ) -> TimesFM25Adapter:
        """
        Store the training series.
        No model training occurs since TimesFM 2.5 is a zero-shot inference model.

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
        self : TimesFM25Adapter

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
        Generate predictions using the TimesFM 2.5 model.

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
                        f"TimesFM 2.5 only supports quantile levels "
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
        Load (but do not compile) the TimesFM 2.5 model into `self._model`
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
    Adapter for Salesforce Moirai-2 foundation models.

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
        Internal Moirai-2 forecast object, populated at the first call to
        `predict`.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    Moirai-2 supports only the fixed quantile levels
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
        No model training occurs since Moirai-2 is a zero-shot inference model.

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
        Generate predictions using Moirai-2.

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
                        f"Moirai-2 only supports quantile levels "
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
                stacklevel=2,
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


_ADAPTER_REGISTRY: dict[str, type] = {
    "amazon/chronos":    Chronos2Adapter,
    "autogluon/chronos": Chronos2Adapter,
    "google/timesfm":    TimesFM25Adapter,
    "Salesforce/moirai": MoiraiAdapter,
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
