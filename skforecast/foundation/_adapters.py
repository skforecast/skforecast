################################################################################
#                         Foundation Model Adapters                            #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
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

from ..utils import expand_index
from ._utils import (
    _validate_positive_int,
    _validate_model_id_prefix,
    _validate_supported_quantiles,
    _tensor_to_numpy,
    _apply_set_params,
    _warn_if_non_commercial,
    get_exog_signature,
)


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
        If `True`, Chronos shares information across the series that are
        forecast in the same batch when predicting in multi-series mode.
        Forwarded directly to `predict_quantiles`. Ignored in single-series
        mode. `FoundationModel` batches together only the series that share
        the same covariate columns, so cross-learning applies within each of
        those groups.

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
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. `False` for Chronos: `FoundationModel` groups
        the series by covariate signature and calls `predict` once per group.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context. `True` for Chronos, which treats them as missing values.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    NaN values in covariates are treated by Chronos as missing values.

    References
    ----------
    .. [1] https://github.com/amazon-science/chronos-forecasting

    .. [2] https://huggingface.co/amazon/chronos-2

    """

    allow_exog: bool = True
    supports_past_only_covariates: bool = True
    supports_heterogeneous_covariates: bool = False
    supports_nan_in_series: bool = True

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

        _validate_positive_int("context_length", context_length)

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
        Set adapter parameters. Resets the pipeline when `model_id`,
        `device_map`, or `torch_dtype` changes, since those are baked into the
        loaded pipeline.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `cross_learning`, `context_length`,
            `device_map`, `torch_dtype`, `predict_kwargs`.

        Returns
        -------
        self : ChronosAdapter

        """

        def validate(candidate_params: dict) -> dict:
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            if "predict_kwargs" in candidate_params:
                candidate_params["predict_kwargs"] = (
                    candidate_params["predict_kwargs"] or {}
                )
            return candidate_params

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                (
                    {"model_id", "device_map", "torch_dtype"},
                    lambda: setattr(self, "_pipeline", None),
                ),
            ),
        )

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
                context      = context[series_name].to_numpy(),
                context_exog = (
                    context_exog.get(series_name) if context_exog is not None else None
                ),
                exog         = exog.get(series_name) if exog is not None else None,
            )
            for series_name in series_names_in
        ]

        quantile_preds, _ = self._pipeline.predict_quantiles(
            inputs            = inputs_list,
            prediction_length = steps,
            quantile_levels   = quantile_levels,
            cross_learning    = self.cross_learning if len(series_names_in) > 1 else False,
            **self.predict_kwargs,
        )

        predictions: dict[str, np.ndarray] = {}
        for i, series_name in enumerate(series_names_in):
            q_arr = _tensor_to_numpy(quantile_preds[i].squeeze(0))
            predictions[series_name] = q_arr

        return predictions

    def _load_pipeline(self) -> None:
        """
        Load the Chronos pipeline into `self._pipeline` if not already set.

        Returns
        -------
        None

        Notes
        -----
        The pipeline is imported lazily from `chronos` and instantiated via
        `BaseChronosPipeline.from_pretrained`, which auto-dispatches to the
        correct pipeline class based on the model config. Optional
        `device_map` and `torch_dtype` stored at initialisation are
        forwarded to the constructor. This method is a no-op when
        `self._pipeline` is already populated. `chronos-forecasting` >=2.0
        must be installed; an `ImportError` is raised otherwise.

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


def _import_timesfm(adapter_name: str) -> Any:
    """
    Import the `timesfm` package lazily.

    Parameters
    ----------
    adapter_name : str
        Adapter class name, used in the raised error message.

    Returns
    -------
    timesfm : module
        The imported `timesfm` package.

    """

    try:
        import timesfm
    except ImportError as exc:
        raise ImportError(
            f"timesfm is required for {adapter_name}. "
            'Install it with `pip install "timesfm[torch]"`.'
        ) from exc

    return timesfm


class TimesFM25Adapter:
    """
    Adapter for Google TimesFM 2.5 foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"google/timesfm-2.5-200m-pytorch"`. Must
        start with `"google/timesfm-2.5"`.
    model : object, default None
        Pre-loaded model instance. If `None`, the model is loaded lazily on
        the first `predict` call. If passed directly, it should already be
        compiled.
    context_length : int, default 512
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it
        is trimmed to this length; if it is shorter, all available
        observations are used as-is. Must be a positive integer.
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
        `max_horizon` here, since those are controlled by the corresponding
        adapter parameters.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict, None
        Stored historical exogenous variables after fitting. Never used by
        this adapter, since TimesFM 2.5 does not support covariates.
    context_length : int
        Maximum number of historical observations used as context.
    max_horizon : int
        Maximum forecast horizon.
    forecast_config_kwargs : dict
        Additional keyword arguments forwarded to `ForecastConfig`.
    allow_exog : bool
        Whether this adapter accepts exogenous variables. Always `False`.
    supports_past_only_covariates : bool
        Whether historical exog columns without future values are used as
        past-only covariates. Always `False`.
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. Always `True`, since covariates are ignored.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context. Always `True`.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    TimesFM 2.5 supports only the fixed quantile levels
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Requesting any other
    level raises a `ValueError`. The point forecast is documented by TimesFM
    as the mean (in practice the 2.5 checkpoint returns a value equal to
    quantile 0.5).

    Compilation behavior. The model is compiled lazily on the first `predict`
    call, sized for the exact number of `steps` requested (not for
    `max_horizon`, which only acts as an upper bound and validation
    ceiling). When `steps` is constant across calls, as in a typical
    backtesting loop, compilation happens only once, on the first fold. A
    later `predict` that requests more `steps` than any previous call
    triggers a single recompilation for the larger horizon. To avoid any
    runtime compilation altogether, pass an already-compiled model via the
    `model` argument.

    References
    ----------
    .. [1] https://github.com/google-research/timesfm

    .. [2] https://huggingface.co/google/timesfm-2.5-200m-pytorch

    """

    SUPPORTED_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    allow_exog: bool = False
    supports_past_only_covariates: bool = False
    supports_heterogeneous_covariates: bool = True
    supports_nan_in_series: bool = True

    _MODEL_ID_PREFIX: str = "google/timesfm-2.5"

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
            HuggingFace model ID, e.g. `"google/timesfm-2.5-200m-pytorch"`.
            Must start with `"google/timesfm-2.5"`.
        model : object, default None
            Pre-loaded model instance. If `None`, the model is loaded
            lazily on the first `predict` call.
        context_length : int, default 512
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `context` is
            longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        max_horizon : int, default 512
            Maximum forecast horizon. If `predict` is called with
            `steps > max_horizon`, a `ValueError` is raised. The model is
            compiled lazily for the exact requested `steps` (up to this
            ceiling) to avoid unnecessary decode iterations. Must be a
            positive integer.
        forecast_config_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to
            `timesfm.ForecastConfig` at compile time.

        """

        _validate_model_id_prefix(model_id, self._MODEL_ID_PREFIX, type(self).__name__)
        _validate_positive_int("context_length", context_length)
        _validate_positive_int("max_horizon", max_horizon)

        self.model_id               = model_id
        self._model                 = model
        self.context_               = None
        self.context_exog_          = None
        self.context_length         = context_length
        self.max_horizon            = max_horizon
        self.forecast_config_kwargs = forecast_config_kwargs or {}
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
        loading or compilation change.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `max_horizon`,
            `forecast_config_kwargs`.

        Returns
        -------
        self : TimesFM25Adapter

        Notes
        -----
        All four parameters affect the loaded (and compiled) model, so
        changing any of them discards the cached model, which is reloaded
        and recompiled lazily on the next `predict` call.

        """

        def validate(candidate_params: dict) -> dict:
            if "model_id" in candidate_params:
                _validate_model_id_prefix(
                    candidate_params["model_id"],
                    self._MODEL_ID_PREFIX,
                    type(self).__name__,
                )
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            if "max_horizon" in candidate_params:
                _validate_positive_int("max_horizon", candidate_params["max_horizon"])
            if "forecast_config_kwargs" in candidate_params:
                candidate_params["forecast_config_kwargs"] = (
                    candidate_params["forecast_config_kwargs"] or {}
                )
            return candidate_params

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                (
                    {"model_id", "context_length", "max_horizon",
                     "forecast_config_kwargs"},
                    lambda: setattr(self, "_model", None),
                ),
            ),
        )

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> TimesFM25Adapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since TimesFM is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables. Stored for API
            consistency but never used, since TimesFM 2.5 does not support
            covariates.

        Returns
        -------
        self : TimesFM25Adapter

        """

        self.context_ = context
        self.context_exog_ = context_exog
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

        Notes
        -----
        A `ValueError` is raised if a requested quantile level is not in
        `SUPPORTED_QUANTILES` or if `steps` exceeds `max_horizon`.

        """

        quantile_list = _validate_supported_quantiles(
            quantiles, self.SUPPORTED_QUANTILES, "TimesFM"
        )

        if steps > self.max_horizon:
            raise ValueError(
                f"`steps` ({steps}) exceeds `max_horizon` ({self.max_horizon})."
            )

        self._load_model()
        self._ensure_compiled(steps)

        series_names_in = list(context.keys())
        inputs_list = [
            context[series_name].to_numpy() for series_name in series_names_in
        ]

        point_forecast, quantile_forecast = self._model.forecast(
            horizon=steps,
            inputs=inputs_list,
        )
        # point_forecast  : (n_series, steps)
        # quantile_forecast: (n_series, steps, 10), idx 0 = mean, 1-9 = q0.1-q0.9

        predictions: dict[str, np.ndarray] = {}
        for i, series_name in enumerate(series_names_in):
            if quantile_list is None:
                # Point forecast: shape (steps, 1)
                predictions[series_name] = np.asarray(point_forecast[i]).reshape(-1, 1)
            else:
                quantile_indices = [round(q * 10) for q in quantile_list]
                qf = np.asarray(quantile_forecast[i])
                # (steps, n_quantiles)
                predictions[series_name] = qf[:, quantile_indices]

        return predictions

    def _load_model(self) -> None:
        """
        Load (but do not compile) the TimesFM 2.5 model into `self._model`
        if not already set.

        Returns
        -------
        None

        Notes
        -----
        This method is a no-op when `self._model` is already populated
        (either by a prior call or by the `model` constructor argument).
        The model is imported lazily from `timesfm` and loaded via
        `TimesFM_2p5_200M_torch.from_pretrained`. Compilation is deferred to
        `_ensure_compiled`, which is called from `predict` with the actual
        forecast horizon so that the compiled decode graph is sized exactly
        for the requested number of steps rather than the (much larger)
        `max_horizon` ceiling. `timesfm` must be installed. A
        `LicenseWarning` is issued after the import succeeds, immediately
        before the weights are loaded, if `model_id` resolves to weights
        released under a non-commercial license.

        """

        if self._model is not None:
            return

        timesfm = _import_timesfm(type(self).__name__)

        _warn_if_non_commercial(self.model_id)

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
        This is separated from `_load_model` so that compilation uses
        the *actual* number of requested forecast steps rather than
        `max_horizon`. TimesFM's compiled decode always runs
        `forecast_config.max_horizon` autoregressive decode iterations
        regardless of the requested horizon; the true horizon is only used
        to *slice* the output afterwards. When the compiled `max_horizon`
        is large (e.g. the default 512) but `steps` is small (e.g. 12),
        the model performs up to `(max_horizon - 1) // output_patch_len`
        unnecessary extra transformer forward passes per inference call.
        Compiling here with `max_horizon = steps` reduces those wasted
        passes to zero for the typical backtesting case where `steps` is
        constant across folds.

        If the model was already compiled for a horizon `>= steps` (e.g. a
        pre-compiled model passed via the `model` constructor argument), this
        method is a no-op.

        """

        fc = getattr(self._model, 'forecast_config', None)
        if fc is not None and steps <= fc.max_horizon:
            return

        timesfm = _import_timesfm(type(self).__name__)
        self._model.compile(
            timesfm.ForecastConfig(
                max_context = self.context_length,
                max_horizon = steps,
                **self.forecast_config_kwargs,
            )
        )


class TimesFM3Adapter:
    """
    Adapter for Google TimesFM 3.0 foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"google/timesfm-3.0-pytorch"`. Must
        start with `"google/timesfm-3.0"`.
    model : object, default None
        Pre-loaded `TimesFM3Forecaster` instance. If `None`, the model is
        loaded lazily on the first `predict` call.
    context_length : int, default 2048
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it
        is trimmed to this length; if it is shorter, all available
        observations are used as-is. Must be a positive integer. TimesFM 3.0
        supports context lengths up to roughly 15,360.
    device : str, default 'auto'
        Device placement for the model. `"auto"` selects the best available
        accelerator (CUDA > MPS > CPU). Also accepts explicit values such as
        `"cuda"`, `"mps"`, or `"cpu"`, forwarded to
        `TimesFM3Forecaster.from_pretrained`.
    predict_kwargs : dict, default None
        Additional keyword arguments forwarded verbatim to `predict_batch`
        (e.g. `use_znorm`, `make_positive`, `use_symmetric_averaging`,
        `sort_quantiles`). Cannot include `contexts`, `horizon`,
        `return_quantiles`, `past_only_covariates`,
        `past_future_covariates`, `padding_mode`, or `ts_ids`, which are
        managed internally.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict, None
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    device : str
        Device placement for the model.
    predict_kwargs : dict
        Additional keyword arguments forwarded to `predict_batch`.
    allow_exog : bool
        Whether this adapter accepts exogenous variables. Always `True`.
    supports_past_only_covariates : bool
        Whether historical exog columns without future values are used as
        past-only covariates. Always `True`.
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. Always `False`: `predict_batch` stacks the
        covariate arrays of every series in a call, so `FoundationModel`
        groups the series by covariate signature and calls `predict` once
        per group.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context. Always `True`.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    TimesFM 3.0 supports only the fixed quantile levels
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Requesting any other
    level raises a `ValueError`. The point forecast is the median
    (quantile 0.5).

    For each series, columns present in its future `exog` become
    known-future covariates spanning `context + horizon`, built by
    concatenating the matching historical column from `context_exog` with
    the future values; columns present only in its `context_exog` become
    past-only covariates. Every series is forwarded with its own covariate
    columns only: `FoundationModel` batches together the series that share
    the same set of past-only and known-future columns and calls `predict`
    once per group, so the prediction of a series never depends on the
    covariates of the other series in the batch. Covariates must be numeric;
    encode categoricals as numbers (e.g. via `transformer_exog`) before
    passing them. NaN values inside covariates and inside the target series
    are linearly interpolated by the backend, and leading NaNs in the target
    trim the context and its covariates accordingly.

    There is no compile step and no horizon ceiling: context length and
    horizon are handled internally by `predict_batch`.

    The pre-trained weights are released under a non-commercial license, so
    loading them raises a `LicenseWarning`.

    References
    ----------
    .. [1] https://github.com/google-research/timesfm

    .. [2] https://huggingface.co/google/timesfm-3.0-pytorch

    """

    SUPPORTED_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    allow_exog: bool = True
    supports_past_only_covariates: bool = True
    supports_heterogeneous_covariates: bool = False
    supports_nan_in_series: bool = True

    _MODEL_ID_PREFIX: str = "google/timesfm-3.0"
    _RESERVED_PREDICT_KWARGS: frozenset[str] = frozenset({
        "contexts", "horizon", "return_quantiles", "past_only_covariates",
        "past_future_covariates", "padding_mode", "ts_ids",
    })

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 2048,
        device: str = "auto",
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. `"google/timesfm-3.0-pytorch"`. Must
            start with `"google/timesfm-3.0"`.
        model : object, default None
            Pre-loaded `TimesFM3Forecaster` instance. If `None`, the model
            is loaded lazily on the first `predict` call.
        context_length : int, default 2048
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `context` is
            longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        device : str, default 'auto'
            Device placement for the model. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU).
        predict_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to
            `predict_batch`.

        """

        _validate_model_id_prefix(model_id, self._MODEL_ID_PREFIX, type(self).__name__)
        _validate_positive_int("context_length", context_length)
        predict_kwargs = predict_kwargs or {}
        self._validate_predict_kwargs(predict_kwargs)

        self.model_id       = model_id
        self._model         = model
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.device         = device
        self.predict_kwargs = predict_kwargs
        self.is_fitted      = False

    @classmethod
    def _validate_predict_kwargs(cls, predict_kwargs: dict[str, Any]) -> None:
        """
        Reject `predict_kwargs` keys that the adapter manages itself.

        Parameters
        ----------
        predict_kwargs : dict
            Candidate `predict_kwargs` value.

        Returns
        -------
        None

        Notes
        -----
        A `ValueError` naming the offending keys is raised if any key in
        `_RESERVED_PREDICT_KWARGS` is present. `contexts`, `horizon`,
        `return_quantiles`, `past_only_covariates`, `past_future_covariates`
        and `padding_mode` are built internally from `context`,
        `context_exog`, `exog`, `steps`, and `quantiles`, so passing them
        explicitly would collide with the internal `predict_batch` call.
        `ts_ids` is reserved because the adapter maps the `predict_batch`
        output back to the series by position.

        """

        reserved = cls._RESERVED_PREDICT_KWARGS & set(predict_kwargs)
        if reserved:
            raise ValueError(
                f"`predict_kwargs` cannot include {sorted(reserved)}. These "
                f"arguments are managed internally by {cls.__name__}."
            )

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `device`, `predict_kwargs`.

        """

        return {
            'model_id':       self.model_id,
            'context_length': self.context_length,
            'device':         self.device,
            'predict_kwargs': self.predict_kwargs or None,
        }

    def set_params(self, **params) -> TimesFM3Adapter:
        """
        Set adapter parameters. Resets the model when parameters that affect
        loading change.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `device`,
            `predict_kwargs`.

        Returns
        -------
        self : TimesFM3Adapter

        Notes
        -----
        Only `model_id` and `device` affect the loaded model, so only those
        discard the cached model. `context_length` and `predict_kwargs` are
        applied at predict time and never trigger a reload.

        """

        def validate(candidate_params: dict) -> dict:
            if "model_id" in candidate_params:
                _validate_model_id_prefix(
                    candidate_params["model_id"],
                    self._MODEL_ID_PREFIX,
                    type(self).__name__,
                )
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            if "predict_kwargs" in candidate_params:
                candidate_params["predict_kwargs"] = (
                    candidate_params["predict_kwargs"] or {}
                )
                self._validate_predict_kwargs(candidate_params["predict_kwargs"])
            return candidate_params

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                ({"model_id", "device"}, lambda: setattr(self, "_model", None)),
            ),
        )

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> TimesFM3Adapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since TimesFM is a zero-shot inference model.

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
        self : TimesFM3Adapter

        """

        self.context_ = context
        self.context_exog_ = context_exog
        self.is_fitted = True

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
        Generate predictions using the TimesFM 3.0 model.

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
        context_exog : dict, None
            Per-series historical exogenous variables (already trimmed).
            For each series, columns not also present in its `exog` are
            forwarded as past-only covariates.
        exog : dict, None
            Per-series future exogenous variables for the forecast
            horizon. Each column is forwarded as a known-future covariate
            of that series, concatenated with its historical values.
        quantiles : list of float or None
            Quantile levels. Must be a subset of `SUPPORTED_QUANTILES`.

        Returns
        -------
        predictions : dict
            Keys are series names, in the same order as `context`. Each
            value is a 2-D array of shape `(steps, n_quantiles)`.

        Notes
        -----
        A `ValueError` is raised if a requested quantile level is not in
        `SUPPORTED_QUANTILES`. There is no horizon ceiling.

        `predict_batch` requires all series in one call to share the same
        covariate layout. `FoundationModel` guarantees it by grouping the
        series by covariate signature (`get_exog_signature`) and calling
        `predict` once per group, so the signature of the first series sets
        the column order for the whole batch. Series of different lengths
        are accepted: `predict_batch` left-pads every series and its
        covariates to the batch context length.

        With covariates present, `padding_mode="edge"` is passed to
        `predict_batch` (the default chosen here); with no covariates,
        `padding_mode="none"` is used. `predict_batch` internally rounds the
        horizon up to a multiple of the output patch length. `"edge"` repeats
        the last known-future covariate value up to that rounded length, so
        those extra positions enter the final patch unmasked, whereas
        `"none"` leaves them masked. Both modes return `steps` rows for any
        `steps` ("edge" is not required); `"edge"` matches the default used by
        TimesFM's own `predict()` and can shift the last patch's predictions
        relative to `"none"`. The point forecast (`quantiles is None`) is the
        model's median quantile.

        """

        quantile_list = _validate_supported_quantiles(
            quantiles, self.SUPPORTED_QUANTILES, "TimesFM"
        )

        self._load_model()

        names = list(context.keys())

        # Every series in a call shares the same covariate columns
        # (`FoundationModel` groups them by covariate signature), so the
        # signature of the first series sets the column order for the batch.
        past_only_cols, fut_cols = get_exog_signature(
            context_exog = context_exog.get(names[0]) if context_exog is not None else None,
            exog         = exog.get(names[0]) if exog is not None else None,
        )
        has_covariates = bool(past_only_cols) or bool(fut_cols)
        contexts = [context[series_name].to_numpy() for series_name in names]

        past_only_list: list[np.ndarray | None] = []
        past_future_list: list[np.ndarray | None] = []
        for series_name in names:
            past_only, past_future = self._build_covariates(
                context_exog   = (
                    context_exog.get(series_name) if context_exog is not None else None
                ),
                exog           = exog.get(series_name) if exog is not None else None,
                past_only_cols = past_only_cols,
                fut_cols       = fut_cols,
            )
            past_only_list.append(past_only)
            past_future_list.append(past_future)

        results = self._model.predict_batch(
            contexts               = contexts,
            horizon                = steps,
            past_only_covariates   = past_only_list if has_covariates else None,
            past_future_covariates = past_future_list if has_covariates else None,
            return_quantiles       = quantile_list is not None,
            padding_mode           = "edge" if has_covariates else "none",
            **self.predict_kwargs,
        )
        outs = dict(zip(names, results))

        if quantile_list is not None:
            quantile_indices = self._match_quantile_indices(
                list(self._model.config.quantiles), quantile_list
            )

        predictions: dict[str, np.ndarray] = {}
        for series_name in names:
            out = outs[series_name]
            if quantile_list is None:
                predictions[series_name] = np.asarray(out.forecast).reshape(-1, 1)
            else:
                predictions[series_name] = (
                    np.asarray(out.quantiles)[:, quantile_indices]
                )

        return predictions

    @staticmethod
    def _to_covariate_array(col_data: Any) -> np.ndarray:
        """
        Convert a covariate column to a `float32` numpy array.

        Parameters
        ----------
        col_data : array-like
            A single covariate column (e.g. a pandas Series or 1-D array).

        Returns
        -------
        col_array : numpy ndarray
            A 1-D `float32` numpy array.

        Notes
        -----
        Only numeric or boolean columns are accepted; a `ValueError` naming
        the offending column is raised otherwise. `predict_batch` casts
        covariates to `float32` internally and has no native categorical
        support, unlike Chronos. Encode categorical covariates as numeric
        values (e.g. via `transformer_exog`) before passing them.

        """

        if isinstance(col_data, pd.Series):
            if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
                return col_data.astype(np.float32).to_numpy()
            raise ValueError(
                f"TimesFM3Adapter supports only numeric covariates. Column "
                f"{col_data.name!r} has dtype {col_data.dtype}. Encode "
                f"categorical covariates as numeric values before passing them."
            )

        arr = np.asarray(col_data)
        if arr.dtype.kind in ("i", "u", "f", "b"):  # integer, unsigned int, float, bool
            return arr.astype(np.float32)

        raise ValueError(
            f"TimesFM3Adapter supports only numeric covariates. Got array of "
            f"dtype {arr.dtype}. Encode categorical covariates as numeric "
            f"values before passing them."
        )

    @classmethod
    def _build_covariates(
        cls,
        context_exog: pd.DataFrame | pd.Series | None,
        exog: pd.DataFrame | pd.Series | None,
        past_only_cols: tuple,
        fut_cols: tuple,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Build the past-only and known-future covariate arrays for one
        series from its own exogenous variables.

        Parameters
        ----------
        context_exog : pandas DataFrame, pandas Series, default None
            Historical exogenous variables aligned to the context for this
            series. Must contain every column in `past_only_cols` and
            `fut_cols`.
        exog : pandas DataFrame, pandas Series, default None
            Future-known exogenous variables covering the forecast horizon
            for this series. Must contain every column in `fut_cols`.
        past_only_cols : tuple
            Past-only covariate columns of this series, as returned by
            `get_exog_signature`.
        fut_cols : tuple
            Known-future covariate columns of this series, as returned by
            `get_exog_signature`.

        Returns
        -------
        past_only : numpy ndarray, None
            Array of shape `(len(past_only_cols), context_len)`, or `None`
            if `past_only_cols` is empty.
        past_future : numpy ndarray, None
            Array of shape `(len(fut_cols), context_len + steps)`, one row
            per column with the historical values followed by the future
            values, or `None` if `fut_cols` is empty.

        Notes
        -----
        No placeholder values are ever generated: every row comes from the
        series' own data. TimesFM 3.0's `predict_batch` linearly interpolates
        NaN values inside covariates, so any NaN present in the user's exog
        is filled by the backend, not by skforecast.

        """

        ctx_df = (
            context_exog.to_frame()
            if isinstance(context_exog, pd.Series)
            else context_exog
        )
        fut_df = (
            exog.to_frame()
            if isinstance(exog, pd.Series)
            else exog
        )

        past_only = (
            np.stack([cls._to_covariate_array(ctx_df[col]) for col in past_only_cols])
            if past_only_cols
            else None
        )
        past_future = (
            np.stack([
                np.concatenate([
                    cls._to_covariate_array(ctx_df[col]),
                    cls._to_covariate_array(fut_df[col]),
                ])
                for col in fut_cols
            ])
            if fut_cols
            else None
        )

        return past_only, past_future

    @staticmethod
    def _match_quantile_indices(
        quantile_grid: list[float],
        quantiles: list[float],
        tol: float = 1e-6,
    ) -> list[int]:
        """
        Map requested quantile levels to column indices in `quantile_grid`.

        Parameters
        ----------
        quantile_grid : list of float
            Quantile levels corresponding to the columns of the model's
            quantile output, in order (e.g. `self._model.config.quantiles`).
        quantiles : list of float
            Requested quantile levels.
        tol : float, default 1e-6
            Maximum allowed absolute difference between a requested
            quantile and its nearest match in `quantile_grid`.

        Returns
        -------
        indices : list of int
            Column index in `quantile_grid` for each entry in
            `quantiles`, in the same order.

        Notes
        -----
        A `ValueError` is raised if a requested quantile has no match
        within `tol`. Matching against the model's own quantile grid
        (synced from the loaded checkpoint) rather than a fixed formula
        keeps the mapping correct even if a checkpoint ships a different
        quantile grid.

        """

        indices = []
        for q in quantiles:
            diffs = [abs(g - q) for g in quantile_grid]
            best_idx = diffs.index(min(diffs))
            if diffs[best_idx] > tol:
                raise ValueError(
                    f"Quantile {q} not found in the model's quantile grid "
                    f"{quantile_grid} (tolerance {tol})."
                )
            indices.append(best_idx)

        return indices

    def _load_model(self) -> None:
        """
        Load the TimesFM 3.0 model into `self._model` if not already set.

        Returns
        -------
        None

        Notes
        -----
        This method is a no-op when `self._model` is already populated
        (either by a prior call or by the `model` constructor argument).
        The model is imported lazily from `timesfm` and loaded via
        `TimesFM3Forecaster.from_pretrained`, resolving `self.device` to a
        concrete device name first. There is no separate compile step:
        context length and horizon are handled internally by
        `predict_batch`. If the installed `timesfm` package predates 3.0
        and does not provide `TimesFM3Forecaster`, an `ImportError` prompts
        the user to upgrade. A `LicenseWarning` is issued only after both
        checks succeed, immediately before the weights are loaded.

        """

        if self._model is not None:
            return

        timesfm = _import_timesfm(type(self).__name__)

        if not hasattr(timesfm, "TimesFM3Forecaster"):
            from importlib.metadata import PackageNotFoundError, version

            try:
                installed = version("timesfm")
            except PackageNotFoundError:
                installed = None

            try:
                major = int(installed.split(".")[0]) if installed else 0
            except ValueError:
                major = 0

            if major < 3:
                raise ImportError(
                    f"TimesFM 3.0 requires `timesfm>=3.0`, but timesfm "
                    f"{installed} is installed and does not provide "
                    f"`TimesFM3Forecaster`. Upgrade with "
                    f'`pip install -U "timesfm[torch]"`.'
                )

            # timesfm>=3 is installed but TimesFM3Forecaster is missing. The
            # usual cause is that torch (an optional extra) is absent, so
            # `timesfm/__init__.py` silently skipped importing its 3.0 backend.
            # Surface the real ImportError instead of a misleading "upgrade".
            try:
                import timesfm3  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    f"TimesFM 3.0 is installed (timesfm {installed}) but its "
                    f"backend could not be imported ({exc}). This usually means "
                    f"torch is missing. Install it with "
                    f'`pip install "timesfm[torch]"`.'
                ) from exc

            raise ImportError(
                f"TimesFM 3.0 is installed (timesfm {installed}) but does not "
                f"provide `TimesFM3Forecaster`. Reinstall with "
                f'`pip install -U "timesfm[torch]"`.'
            )

        _warn_if_non_commercial(self.model_id)

        self._model = timesfm.TimesFM3Forecaster.from_pretrained(
            self.model_id,
            device=_resolve_torch_device(self.device),
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
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. `True`, since covariates are ignored.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    Moirai supports only the fixed quantile levels
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Requesting any
    other level raises a `ValueError`.

    Covariate support via the high-level `Moirai2Forecast.predict()` API
    is not functional: the padding/truncation loop inside `predict()`
    clips every list-valued field (including `feat_dynamic_real`) to
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
    supports_past_only_covariates: bool = False
    supports_heterogeneous_covariates: bool = True
    supports_nan_in_series: bool = True

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

        _validate_positive_int("context_length", context_length)

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
        `model_id`, `context_length`, or `device` changes.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `device`.

        Returns
        -------
        self : MoiraiAdapter

        """

        def validate(candidate_params: dict) -> dict:
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            return candidate_params

        def _reset_module() -> None:
            self._module = None
            self._forecast_obj = None

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                ({"model_id", "context_length", "device"}, _reset_module),
            ),
        )

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

        Notes
        -----
        A `ValueError` is raised if a requested quantile level is not in
        `SUPPORTED_QUANTILES`.

        """

        quantile_list = _validate_supported_quantiles(
            quantiles, self.SUPPORTED_QUANTILES, "Moirai"
        )

        quantile_levels = quantile_list if quantile_list is not None else [0.5]
        quantile_indices = [
            next(
                i for i, supported_quantile in enumerate(self.SUPPORTED_QUANTILES)
                if abs(q - supported_quantile) < 1e-9
            )
            for q in quantile_levels
        ]

        series_names_in = list(context.keys())
        inputs_list = [
            context[series_name].to_numpy(dtype=np.float32).reshape(-1, 1)
            for series_name in series_names_in
        ]

        raw = self._run_inference(inputs_list, steps)

        predictions: dict[str, np.ndarray] = {}
        for i, series_name in enumerate(series_names_in):
            # (steps, n_quantiles)
            predictions[series_name] = raw[i][quantile_indices, :].T

        return predictions

    def _load_module(self) -> None:
        """
        Load the `Moirai2Module` into `self._module` if not already set.

        Returns
        -------
        None

        Notes
        -----
        The module is imported lazily from `uni2ts` and instantiated via
        `Moirai2Module.from_pretrained`, then set to evaluation mode.
        This method is a no-op when `self._module` is already populated. A
        `LicenseWarning` is issued after the import succeeds, immediately
        before the weights are loaded. `uni2ts` must be installed; an
        `ImportError` is raised otherwise.
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
        _warn_if_non_commercial(self.model_id)
        self._module = Moirai2Module.from_pretrained(self.model_id)
        self._module.eval()

    def _ensure_forecast_obj(self) -> None:
        """
        Build the `Moirai2Forecast` inference wrapper if not already set.

        Returns
        -------
        None

        Notes
        -----
        Calls `_load_module`, which requires `uni2ts`, then wraps `self._module` in a
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
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. `False` for TabICL, whose long-format input
        frame holds one column set for every series: `FoundationModel`
        groups the series by covariate signature and calls `predict` once
        per group.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context. `True`: TabICL drops the rows whose target is NaN.
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
    present in both context and future data. NaN values in the future
    covariates are accepted by TabICL with a warning.

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
    supports_past_only_covariates: bool = False
    supports_heterogeneous_covariates: bool = False
    supports_nan_in_series: bool = True

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

        _validate_positive_int("context_length", context_length)
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
        self.tabicl_config     = tabicl_config or {}
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

        def validate(candidate_params: dict) -> dict:
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            if "point_estimate" in candidate_params and candidate_params[
                "point_estimate"
            ] not in ("mean", "median"):
                raise ValueError(
                    f"`point_estimate` must be 'mean' or 'median'. "
                    f"Got {candidate_params['point_estimate']!r}."
                )
            if "tabicl_config" in candidate_params:
                candidate_params["tabicl_config"] = (
                    candidate_params["tabicl_config"] or {}
                )
            if "show_progress" in candidate_params and not isinstance(
                candidate_params["show_progress"], bool
            ):
                raise ValueError(
                    f"`show_progress` must be a bool. "
                    f"Got {candidate_params['show_progress']!r}."
                )
            return candidate_params

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                (
                    {"model_id", "context_length", "point_estimate",
                     "tabicl_config", "temporal_features"},
                    lambda: setattr(self, "_model", None),
                ),
            ),
        )

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
        for series_name in series_names_in:
            group = result_df.loc[series_name]  # DataFrame indexed by timestamp
            if quantile_list is None:
                predictions[series_name] = group["target"].to_numpy().reshape(-1, 1)
            else:
                predictions[series_name] = group[quantile_list].to_numpy()

        return predictions

    def _load_model(self) -> None:
        """
        Load the `TabICLForecaster` into `self._model` if not already set.

        Returns
        -------
        None

        Notes
        -----
        The model is imported lazily from `tabicl` and instantiated with
        the current adapter parameters. This method is a no-op when
        `self._model` is already populated (either by a prior call or by
        the `model` test-injection parameter). `tabicl[forecast]` must be
        installed; an `ImportError` is raised otherwise.
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
            timestamps = expand_index(series.index, steps=steps)
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
        for series_name in series_names:
            series = context[series_name]
            n = len(series)
            part = pd.DataFrame({
                "item_id":   np.full(n, series_name),
                "timestamp": np.asarray(self._get_timestamps(series, is_datetime)),
                "target":    series.to_numpy(dtype=float),
            })
            exog_entry = (
                context_exog.get(series_name) if context_exog is not None else None
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
        for series_name in series_names:
            series = context[series_name]
            part = pd.DataFrame({
                "item_id":   np.full(steps, series_name),
                "timestamp": np.asarray(
                    self._get_future_timestamps(series, steps, is_datetime)
                ),
            })
            future_exog = exog.get(series_name) if exog is not None else None
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
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. `True`: the library handles the missing cells
        of the long-format input frame.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context.
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
    supports_past_only_covariates: bool = False
    supports_heterogeneous_covariates: bool = True
    supports_nan_in_series: bool = True

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

        _validate_positive_int("context_length", context_length)
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
        self.tabpfn_model_config = tabpfn_model_config or {}
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

        def validate(candidate_params: dict) -> dict:
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            if "mode" in candidate_params and candidate_params["mode"] not in (
                "local", "client"
            ):
                raise ValueError(
                    f"`mode` must be 'local' or 'client'. "
                    f"Got {candidate_params['mode']!r}."
                )
            if "point_estimate" in candidate_params and candidate_params[
                "point_estimate"
            ] not in ("mean", "median", "mode"):
                raise ValueError(
                    f"`point_estimate` must be 'mean', 'median' or 'mode'. "
                    f"Got {candidate_params['point_estimate']!r}."
                )
            if "tabpfn_model_config" in candidate_params:
                candidate_params["tabpfn_model_config"] = (
                    candidate_params["tabpfn_model_config"] or {}
                )
            if "show_progress" in candidate_params and not isinstance(
                candidate_params["show_progress"], bool
            ):
                raise ValueError(
                    f"`show_progress` must be a bool. "
                    f"Got {candidate_params['show_progress']!r}."
                )
            return candidate_params

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                (
                    {"model_id", "context_length", "mode", "point_estimate",
                     "tabpfn_model_config", "temporal_features"},
                    lambda: setattr(self, "_model", None),
                ),
            ),
        )

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
        for series_name in series_names_in:
            group = result_df.loc[series_name]  # DataFrame indexed by timestamp
            if quantile_list is None:
                predictions[series_name] = group["target"].to_numpy().reshape(-1, 1)
            else:
                predictions[series_name] = group[quantile_list].to_numpy()

        return predictions

    def _load_model(self) -> None:
        """
        Load the `TabPFNTSPipeline` into `self._model` if not already set.

        Returns
        -------
        None

        Notes
        -----
        The pipeline is imported lazily from `tabpfn_time_series` and
        instantiated with the current adapter parameters. This method is a
        no-op when `self._model` is already populated (either by a prior
        call or by the `model` test-injection parameter). A
        `LicenseWarning` is issued after the import succeeds, immediately
        before the pipeline is instantiated. `tabpfn-time-series` must be
        installed; an `ImportError` is raised otherwise.
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
        _warn_if_non_commercial(self.model_id)

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
            timestamps = expand_index(series.index, steps=steps)
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
        for series_name in series_names:
            series = context[series_name]
            n = len(series)
            part = pd.DataFrame({
                "item_id":   np.full(n, series_name),
                "timestamp": np.asarray(self._get_timestamps(series, is_datetime)),
                "target":    series.to_numpy(dtype=float),
            })
            exog_entry = (
                context_exog.get(series_name) if context_exog is not None else None
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
        for series_name in series_names:
            series = context[series_name]
            part = pd.DataFrame({
                "item_id":   np.full(steps, series_name),
                "timestamp": np.asarray(
                    self._get_future_timestamps(series, steps, is_datetime)
                ),
            })
            future_exog = exog.get(series_name) if exog is not None else None
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
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. `True`: T0 defines NaN as an absent covariate
        value, so the adapter pools the columns of all series and fills the
        missing cells with NaN.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context.
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

    T0 checkpoints (e.g. `theforecastingcompany/t0-alpha`) are gated on the
    Hugging Face Hub: visit the model page while logged in to accept its
    license, then authenticate locally (`hf auth login` or the `HF_TOKEN`
    environment variable) before first use.

    References
    ----------
    .. [1] https://github.com/theforecastingcompany/tfc-t0
    
    .. [2] https://huggingface.co/theforecastingcompany/t0-alpha

    """

    allow_exog: bool = True
    supports_past_only_covariates: bool = False
    supports_heterogeneous_covariates: bool = True
    supports_nan_in_series: bool = True

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

        _validate_positive_int("context_length", context_length)

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

        def validate(candidate_params: dict) -> dict:
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            return candidate_params

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                (
                    {"model_id", "device_map", "torch_dtype"},
                    lambda: setattr(self, "_model", None),
                ),
            ),
        )

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
        arrays = [
            np.asarray(context[series_name].to_numpy(), dtype=np.float32)
            for series_name in series_names
        ]
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

        q_arr = _tensor_to_numpy(forecast.quantiles)

        quantile_column_indices = [query_levels.index(q) for q in requested]
        return {
            series_name: q_arr[i][:, quantile_column_indices]
            for i, series_name in enumerate(series_names)
        }

    def _load_model(self) -> None:
        """
        Load the T0 model into `self._model` if not already set.

        Returns
        -------
        None

        Notes
        -----
        The model is imported lazily from `t0` and loaded via
        `T0Forecaster.from_pretrained`, then moved to the resolved device and
        switched to eval mode. This method is a no-op when `self._model` is
        already populated. `tfc-t0` must be installed; an `ImportError` is
        raised otherwise. An `OSError` is raised if
        `T0Forecaster.from_pretrained` fails to build the model, most
        commonly because the repository is gated on the Hugging Face Hub and
        the active credentials have not accepted its license.

        T0 checkpoints are gated on the Hugging Face Hub. When the
        repository's `config.json` cannot be downloaded (e.g. the license
        has not been accepted, or no valid token is available),
        `huggingface_hub`'s `from_pretrained` silently swallows the download
        failure and falls back to instantiating the model with no
        constructor arguments, raising a confusing `TypeError` about missing
        hyperparameters. That `TypeError` is caught here and re-raised as a
        clearer `OSError` pointing to the likely cause.

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

        try:
            model = T0Forecaster.from_pretrained(self.model_id)
        except TypeError as exc:
            raise OSError(
                f"Could not load model '{self.model_id}' from the Hugging "
                f"Face Hub. This is often caused by a gated repository "
                f"whose license has not been accepted: visit "
                f"https://huggingface.co/{self.model_id} while logged in "
                f"to accept it, then authenticate locally (`hf auth login` "
                f"or the `HF_TOKEN` environment variable) before retrying."
            ) from exc

        device = _resolve_torch_device(self.device_map)
        model = model.to(device)
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
        `exog`) cover the horizon. Every unfilled cell (a padded timestep, or a
        column/series that lacks that covariate) stays NaN, which T0 treats as
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
            series_name: (
                series_exog
                if isinstance(series_exog, pd.DataFrame)
                else series_exog.to_frame()
            )
            for series_name, series_exog in exog.items()
            if series_exog is not None
        }
        if not future_frames:
            return None

        columns: list[str] = list(
            dict.fromkeys(
                col for frame in future_frames.values() for col in frame.columns
            )
        )

        total_length = context_length + steps
        covariates = np.full(
            (len(series_names), len(columns), total_length), np.nan, dtype=np.float32
        )
        column_index = {col: j for j, col in enumerate(columns)}
        for row, series_name in enumerate(series_names):
            future_df = future_frames.get(series_name)
            if future_df is None:
                continue
            past_df = None
            if context_exog is not None and context_exog.get(series_name) is not None:
                ctx = context_exog[series_name]
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

        Notes
        -----
        A `ValueError` is raised if the column is neither numeric nor
        boolean. T0 only conditions on numeric covariates; categoricals must
        be encoded as numbers.

        """

        if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            return col_data.astype(np.float32).to_numpy()

        raise ValueError(
            f"T0Adapter supports only numeric covariates. Column "
            f"{col_data.name!r} has dtype {col_data.dtype}. Encode categorical "
            f"covariates as numeric values before passing them."
        )


class TSICLAdapter:
    """
    Adapter for EDF Lab TS-ICL foundation model.

    Parameters
    ----------
    model_id : str
        Model ID, e.g. `"taharnbl/TS-ICL"`. Used only to resolve this
        adapter; the underlying checkpoint is always downloaded from the
        `taharnbl/TS-ICL` Hugging Face repository, controlled by
        `checkpoint_version`.
    model : object, default None
        Pre-instantiated `TSICL` model instance. If `None`, a new instance
        is created lazily on the first call to `predict`. Intended for
        testing only.
    checkpoint_version : str, default 'tsicl-v1.ckpt'
        Checkpoint filename to download from the `taharnbl/TS-ICL`
        Hugging Face repository.
    context_length : int, default 4096
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Must be a positive integer.
    device : str, default 'auto'
        Device placement for inference. `"auto"` selects the best available
        accelerator (CUDA > MPS > CPU). Also accepts explicit values such as
        `"cuda"`, `"mps"`, or `"cpu"`. Note that TS-ICL currently falls back
        to CPU whenever CUDA is unavailable, so `"mps"` has no effect on
        Apple Silicon.
    allow_auto_download : bool, default True
        Whether to allow automatic download of the checkpoint from Hugging
        Face Hub if it is not already cached locally.

    Attributes
    ----------
    model_id : str
        Model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    checkpoint_version : str
        Checkpoint filename downloaded from Hugging Face Hub.
    context_length : int
        Maximum number of historical observations used as context.
    device : str
        Device placement for inference.
    allow_auto_download : bool
        Whether automatic checkpoint download is allowed.
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. `False` for TS-ICL: `FoundationModel` groups
        the series by covariate signature and calls `predict` once per group.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    TS-ICL conditions on covariates that are known over the context, the
    forecast horizon, or both. skforecast exogenous variables map directly
    onto this channel: historical values (`context_exog`) are forwarded as
    `past_covariates` and future values (`exog`) as `future_covariates`.
    Covariates must be numeric; encode categoricals as numbers before
    passing them.

    TS-ICL only supports quantile levels on a 0.01 grid in `[0.01, 0.99]`
    (i.e. a subset of `[0.01, 0.02, ..., 0.99]`); requesting any other level
    raises a `ValueError` from the underlying library.

    References
    ----------
    .. [1] https://github.com/EDF-Lab/ts-icl

    .. [2] https://huggingface.co/taharnbl/TS-ICL

    """

    allow_exog: bool = True
    supports_past_only_covariates: bool = True
    supports_heterogeneous_covariates: bool = False
    supports_nan_in_series: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        checkpoint_version: str = "tsicl-v1.ckpt",
        context_length: int = 4096,
        device: str = "auto",
        allow_auto_download: bool = True,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            Model ID, e.g. `"taharnbl/TS-ICL"`. Used only to resolve this
            adapter.
        model : object, default None
            Pre-instantiated `TSICL` model instance. If `None`, a new
            instance is created lazily on the first call to `predict`.
        checkpoint_version : str, default 'tsicl-v1.ckpt'
            Checkpoint filename to download from the `taharnbl/TS-ICL`
            Hugging Face repository.
        context_length : int, default 4096
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if
            `context` is longer than `context_length` it is trimmed to
            this length before inference; if it is shorter, all available
            observations are passed as-is. Must be a positive integer.
        device : str, default 'auto'
            Device placement for inference. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU).
        allow_auto_download : bool, default True
            Whether to allow automatic download of the checkpoint from
            Hugging Face Hub if it is not already cached locally.

        """

        _validate_positive_int("context_length", context_length)

        self.model_id             = model_id
        self._model               = model
        self._resolved_device     = None
        self.context_             = None
        self.context_exog_        = None
        self.checkpoint_version   = checkpoint_version
        self.context_length       = context_length
        self.device               = device
        self.allow_auto_download  = allow_auto_download
        self.is_fitted            = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `checkpoint_version`, `context_length`,
            `device`, `allow_auto_download`.

        """
        return {
            'model_id':             self.model_id,
            'checkpoint_version':   self.checkpoint_version,
            'context_length':       self.context_length,
            'device':               self.device,
            'allow_auto_download':  self.allow_auto_download,
        }

    def set_params(self, **params) -> TSICLAdapter:
        """
        Set adapter parameters. Resets the model when `checkpoint_version`
        or `allow_auto_download` changes, since those control which
        checkpoint is loaded.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `checkpoint_version`, `context_length`,
            `device`, `allow_auto_download`.

        Returns
        -------
        self : TSICLAdapter

        """

        def validate(candidate_params: dict) -> dict:
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            return candidate_params

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                (
                    {"checkpoint_version", "allow_auto_download"},
                    lambda: setattr(self, "_model", None),
                ),
                ({"device"}, lambda: setattr(self, "_resolved_device", None)),
            ),
        )

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
    ) -> TSICLAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since TS-ICL is a zero-shot inference model.

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
        self : TSICLAdapter

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
        Generate predictions using the TS-ICL model.

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

        # NOTE: the model is loaded lazily here so that the adapter can be
        # instantiated and fitted without requiring tsicl to be installed.
        self._load_model()

        import torch

        quantile_list = list(quantiles) if quantiles is not None else None
        query_levels = quantile_list if quantile_list is not None else [0.5]

        series_names_in = list(context.keys())
        inputs_list = [
            self._build_tsicl_input(
                context      = context[series_name].to_numpy(),
                context_exog = (
                    context_exog.get(series_name) if context_exog is not None else None
                ),
                exog         = exog.get(series_name) if exog is not None else None,
            )
            for series_name in series_names_in
        ]

        if self._resolved_device is None:
            self._resolved_device = _resolve_torch_device(self.device)
        device = torch.device(self._resolved_device)

        _, quantile_preds = self._model.forecast(
            inputs            = inputs_list,
            prediction_length = steps,
            quantile_levels   = query_levels,
            context_length    = self.context_length,
            device            = device,
            denormalize       = True,
            squeeze_output    = False,
        )

        predictions: dict[str, np.ndarray] = {}
        for i, series_name in enumerate(series_names_in):
            q_arr = _tensor_to_numpy(quantile_preds[i])
            predictions[series_name] = q_arr[0]  # drop the single-variate dim

        return predictions

    def _load_model(self) -> None:
        """
        Load the TS-ICL model into `self._model` if not already set.

        Returns
        -------
        None

        Notes
        -----
        The model is imported lazily from `tsicl` and instantiated via
        `TSICL(checkpoint_version=..., allow_auto_download=...)`, which
        downloads (or loads from cache) the checkpoint from the
        `taharnbl/TS-ICL` Hugging Face repository. This method is a no-op
        when `self._model` is already populated. A `LicenseWarning` is
        issued after the import succeeds, immediately before the checkpoint
        is loaded. `tsicl` must be installed; an `ImportError` is raised
        otherwise.

        """

        if self._model is not None:
            return
        try:
            from tsicl import TSICL
        except ImportError as exc:
            raise ImportError(
                "tsicl is required for TSICLAdapter. "
                "Install it with `pip install tsicl`."
            ) from exc

        _warn_if_non_commercial(self.model_id)

        self._model = TSICL(
            checkpoint_version   = self.checkpoint_version,
            allow_auto_download  = self.allow_auto_download,
        )

    @staticmethod
    def _to_covariate_array(col_data: Any) -> np.ndarray:
        """
        Convert a covariate column to a `float32` numpy array.

        Parameters
        ----------
        col_data : array-like
            A single covariate column (e.g. a pandas Series or 1-D array).

        Returns
        -------
        col_array : numpy ndarray
            A 1-D `float32` numpy array.

        Notes
        -----
        A `ValueError` is raised if the column is neither numeric nor
        boolean. TS-ICL only conditions on numeric covariates; categoricals
        must be encoded as numbers.

        """

        if isinstance(col_data, pd.Series):
            if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
                return col_data.astype(np.float32).to_numpy()
            raise ValueError(
                f"TSICLAdapter supports only numeric covariates. Column "
                f"{col_data.name!r} has dtype {col_data.dtype}. Encode "
                f"categorical covariates as numeric values before passing them."
            )

        arr = np.asarray(col_data)
        if arr.dtype.kind in ("i", "u", "f", "b"):  # integer, unsigned int, float, bool
            return arr.astype(np.float32)

        raise ValueError(
            f"TSICLAdapter supports only numeric covariates. Got array of "
            f"dtype {arr.dtype}. Encode categorical covariates as numeric "
            f"values before passing them."
        )

    def _build_tsicl_input(
        self,
        context: np.ndarray,
        context_exog: pd.DataFrame | pd.Series | None = None,
        exog: pd.DataFrame | pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Build the input dict consumed by `TSICL.forecast`.

        Parameters
        ----------
        context : numpy ndarray
            1-D array of observed time series values used as context. Must be
            castable to `float32`.
        context_exog : pandas DataFrame, pandas Series, default None
            Historical exogenous variables whose index is aligned to
            `context`. Each column (or the single Series, referenced by
            its name) becomes an entry in the returned "past_covariates"
            dict. Must be numeric or boolean.
        exog : pandas DataFrame, pandas Series, default None
            Future-known exogenous variables covering the forecast horizon.
            Must have exactly `steps` rows. Each column becomes an entry in
            the returned "future_covariates" dict. Must be numeric or
            boolean.

        Returns
        -------
        input_dict : dict
            Dictionary with mandatory key "target" (1-D `float32`
            `numpy ndarray`) and optional keys "past_covariates" and
            "future_covariates", each mapping column names to 1-D
            `float32` arrays.

        """

        input_dict = {"target": np.asarray(context, dtype=np.float32)}
        if context_exog is not None:
            df = (
                context_exog
                if isinstance(context_exog, pd.DataFrame)
                else context_exog.to_frame()
            )
            input_dict["past_covariates"] = {
                col: TSICLAdapter._to_covariate_array(df[col]) for col in df.columns
            }
        if exog is not None:
            df = (
                exog
                if isinstance(exog, pd.DataFrame)
                else exog.to_frame()
            )
            input_dict["future_covariates"] = {
                col: TSICLAdapter._to_covariate_array(df[col]) for col in df.columns
            }

        return input_dict


class NoriAdapter:
    """
    Adapter for Synthefy Nori zero-shot tabular foundation models.

    Nori is a tabular regression foundation model that predicts via in-context
    learning: given labeled context rows it predicts query rows in a single
    forward pass, with no task-specific training or fine-tuning. This adapter
    frames forecasting as tabular regression: each series is featurized (running
    index, calendar features, Fourier seasonal terms, and optional known-future
    covariates) and a `NoriRegressor` predicts the forecast horizon zero-shot.

    Parameters
    ----------
    model_id : str
        Model ID, e.g. `"Synthefy/Nori"` (6M), `"Synthefy/Nori-30M"` or
        `"Synthefy/Nori-100M"`. Used to resolve this adapter and, unless
        overridden in `nori_config`, to select the checkpoint downloaded from
        HuggingFace.
    model : object, default None
        Pre-instantiated `NoriRegressor` instance. If `None`, a new instance
        is created lazily on the first call to `predict`. Intended for testing
        only.
    context_length : int, default 4096
        Maximum number of historical observations to use as context rows. At fit
        time only the last `context_length` observations are stored. At predict
        time, if `context` is longer than `context_length` it is trimmed to this
        length; if it is shorter, all available observations are used as-is. Must
        be a positive integer.
    point_estimate : str, default 'mean'
        Method used to derive the point forecast from Nori's predictive
        distribution. Accepted values: `'mean'`, `'median'`, `'mode'`.
    add_calendar_features : bool, default True
        If `True`, add calendar features (month, day, day-of-week, day-of-year,
        quarter, hour) when the series has a `DatetimeIndex`. Ignored for
        `RangeIndex` series.
    n_fourier_terms : int, default 2
        Number of Fourier (sin/cos) seasonal harmonics added on the yearly and
        weekly cycles for datetime series (or on the running index for
        `RangeIndex` series). Set `0` to disable. Must be a non-negative integer.
    nori_config : dict, default None
        Keyword arguments forwarded verbatim to `NoriRegressor` at instantiation
        (e.g. `device`, `token`, `augmentations`). The checkpoint defaults to
        `model_id` and can be overridden here with `model` (a registry name such
        as `'nori-6m'` or a HuggingFace repo id) or `model_path` (a local
        checkpoint file, which takes precedence over `model`).

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
    point_estimate : str
        Point forecast method.
    add_calendar_features : bool
        Whether calendar features are added for datetime series.
    n_fourier_terms : int
        Number of Fourier seasonal harmonics added.
    nori_config : dict
        Additional configuration forwarded to `NoriRegressor`.
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. `True`: every series is fitted and predicted
        in its own `NoriRegressor` call.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context. `True`: `NoriRegressor` rejects NaN, so the adapter drops
        the context rows whose target (or any feature) is NaN before the
        in-context fit.
    is_fitted : bool
        Whether the adapter has been fitted.
    _model : object
        Internal `NoriRegressor` instance. `None` until the first call to
        `predict`, after which it is cached for reuse.

    Notes
    -----
    Nori supports arbitrary quantile levels (any float strictly in `(0, 1)`),
    unlike models with fixed quantile sets such as TimesFM or Moirai. A
    `bar_distribution` checkpoint does not support quantiles.

    Covariate support is available for *known-future* covariates: columns present
    in both the historical context and the forecast horizon are used as features.
    Covariates without future values are ignored. Covariates must be numeric;
    encode categoricals as numbers before passing them.

    Series with a `RangeIndex` are accepted; only running-index and
    Fourier(index) features are meaningful there (calendar features are skipped).

    References
    ----------
    .. [1] https://github.com/Synthefy/synthefy-nori

    .. [2] https://huggingface.co/Synthefy/Nori

    .. [3] https://docs.synthefy.com/nori/

    """

    allow_exog: bool = True
    supports_past_only_covariates: bool = False
    supports_heterogeneous_covariates: bool = True
    supports_nan_in_series: bool = True

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
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            Model ID, e.g. `"Synthefy/Nori"`.
        model : object, default None
            Pre-instantiated `NoriRegressor` instance. If `None`, a new
            instance is created lazily on the first call to `predict`.
            Intended for testing only.
        context_length : int, default 4096
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if `context`
            is longer than `context_length` it is trimmed to this length
            before inference; if it is shorter, all available observations are
            passed as-is. Must be a positive integer.
        point_estimate : str, default 'mean'
            Method used to derive the point forecast. Accepted values:
            `'mean'`, `'median'`, `'mode'`.
        add_calendar_features : bool, default True
            If `True`, add calendar features when the series has a
            `DatetimeIndex`. Ignored for `RangeIndex` series.
        n_fourier_terms : int, default 2
            Number of Fourier seasonal harmonics added. Set `0` to disable.
            Must be a non-negative integer.
        nori_config : dict, default None
            Keyword arguments forwarded verbatim to `NoriRegressor` at
            instantiation. The checkpoint defaults to `model_id` and can be
            overridden here with `model` or `model_path`.

        """

        _validate_positive_int("context_length", context_length)
        if point_estimate not in ("mean", "median", "mode"):
            raise ValueError(
                f"`point_estimate` must be 'mean', 'median' or 'mode'. "
                f"Got {point_estimate!r}."
            )
        if not isinstance(add_calendar_features, bool):
            raise ValueError(
                f"`add_calendar_features` must be a bool. "
                f"Got {add_calendar_features!r}."
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
        self.nori_config           = nori_config or {}
        self.is_fitted             = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `point_estimate`,
            `add_calendar_features`, `n_fourier_terms`, `nori_config`.
            `nori_config` is returned as `None` when no additional config was
            set (i.e. when the internal dict is empty).

        """
        return {
            "model_id":              self.model_id,
            "context_length":        self.context_length,
            "point_estimate":        self.point_estimate,
            "add_calendar_features": self.add_calendar_features,
            "n_fourier_terms":       self.n_fourier_terms,
            "nori_config":           self.nori_config or None,
        }

    def set_params(self, **params) -> NoriAdapter:
        """
        Set adapter parameters. Resets the loaded model when a parameter baked
        into the `NoriRegressor` instance changes (`model_id`, `nori_config`);
        featurization/inference-time parameters (`context_length`,
        `point_estimate`, `add_calendar_features`, `n_fourier_terms`) do not
        reset the model.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `point_estimate`,
            `add_calendar_features`, `n_fourier_terms`, `nori_config`.

        Returns
        -------
        self : NoriAdapter

        """

        def validate(candidate_params: dict) -> dict:
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            if "point_estimate" in candidate_params and candidate_params[
                "point_estimate"
            ] not in ("mean", "median", "mode"):
                raise ValueError(
                    f"`point_estimate` must be 'mean', 'median' or 'mode'. "
                    f"Got {candidate_params['point_estimate']!r}."
                )
            if "add_calendar_features" in candidate_params and not isinstance(
                candidate_params["add_calendar_features"], bool
            ):
                raise ValueError(
                    f"`add_calendar_features` must be a bool. "
                    f"Got {candidate_params['add_calendar_features']!r}."
                )
            if "n_fourier_terms" in candidate_params and (
                not isinstance(candidate_params["n_fourier_terms"], int)
                or candidate_params["n_fourier_terms"] < 0
            ):
                raise ValueError(
                    f"`n_fourier_terms` must be a non-negative integer. "
                    f"Got {candidate_params['n_fourier_terms']!r}."
                )
            if "nori_config" in candidate_params:
                candidate_params["nori_config"] = (
                    candidate_params["nori_config"] or {}
                )
            return candidate_params

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                ({"model_id", "nori_config"}, lambda: setattr(self, "_model", None)),
            ),
        )

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> NoriAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since Nori is a zero-shot inference model.

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
        self : NoriAdapter

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
        Generate predictions using Nori.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

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
        quantiles : list of float or None
            Quantile levels to return, in the requested order. Must lie
            strictly in `(0, 1)`. If `None`, a point forecast is produced
            (shape `(steps, 1)`).

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D numpy ndarray of shape
            `(steps, n_quantiles)` with columns ordered to match `quantiles`.

        Notes
        -----
        A `ValueError` is raised if a requested quantile level is not
        strictly in `(0, 1)`.

        """

        quantile_list = list(quantiles) if quantiles is not None else None
        if quantile_list is not None and any(
            (q <= 0.0) or (q >= 1.0) for q in quantile_list
        ):
            raise ValueError(
                "NoriAdapter quantiles must lie strictly in (0, 1). "
                f"Got {quantile_list!r}."
            )

        # Nori does not guarantee that the output column order matches the
        # requested quantiles, so query sorted, unique levels and reindex the
        # columns back to the caller's order afterwards.
        if quantile_list is not None:
            query_levels = sorted(set(quantile_list))
            quantile_column_indices = [query_levels.index(q) for q in quantile_list]

        self._load_model()

        first_series = next(iter(context.values()))
        is_datetime = isinstance(first_series.index, pd.DatetimeIndex)
        if not is_datetime and self.add_calendar_features:
            warnings.warn(
                "NoriAdapter received series with a non-DatetimeIndex; "
                "calendar features are skipped. Only running-index and "
                "Fourier(index) features are used.",
                # stacklevel=3: NoriAdapter.predict → FoundationModel.predict → user
                stacklevel=3,
            )

        predictions: dict[str, np.ndarray] = {}
        for series_name, series in context.items():
            ctx_exog = (
                context_exog.get(series_name) if context_exog is not None else None
            )
            fut_exog = exog.get(series_name) if exog is not None else None
            exog_cols = self._known_future_columns(ctx_exog, fut_exog)

            X_ctx = self._featurize(
                series, is_datetime, ctx_exog, exog_cols, offset=0, n=len(series)
            )
            X_fut = self._featurize(
                series, is_datetime, fut_exog, exog_cols, offset=len(series), n=steps
            )
            y_ctx = series.to_numpy(dtype=float)

            # NoriRegressor rejects NaN. Rows whose target or any feature is
            # NaN are dropped from the context: the running-index feature is
            # an absolute offset, so dropping interior rows keeps the
            # remaining rows correctly positioned in time.
            valid_rows = ~np.isnan(y_ctx) & ~np.isnan(X_ctx).any(axis=1)
            if not valid_rows.any():
                raise ValueError(
                    f"Series '{series_name}' has no context rows without NaN in the "
                    f"target and the covariates. NoriAdapter cannot predict it."
                )
            X_ctx = X_ctx[valid_rows]
            y_ctx = y_ctx[valid_rows]

            # Nori fits in-context (no gradient training); the cached model is
            # re-conditioned on each series' context rows before predicting.
            self._model.fit(X_ctx, y_ctx)

            if quantile_list is None:
                y_hat = self._model.predict(X_fut, output_type=self.point_estimate)
                predictions[series_name] = self._to_numpy(y_hat).reshape(-1, 1)
            else:
                q = self._to_numpy(
                    self._model.predict(
                        X_fut, output_type="quantiles", quantiles=query_levels
                    )
                )
                # Nori returns (n_quantiles, steps); skforecast expects
                # (steps, n_quantiles). Reorder columns to the requested order.
                q = q.reshape(len(query_levels), steps).T
                predictions[series_name] = q[:, quantile_column_indices]

        return predictions

    def _load_model(self) -> None:
        """
        Load the `NoriRegressor` into `self._model` if not already set.

        Returns
        -------
        None

        Notes
        -----
        The regressor is imported lazily from `synthefy_nori` and instantiated
        with `nori_config`. This method is a no-op when `self._model` is
        already populated (either by a prior call or by the `model`
        test-injection parameter). The same instance is reused across series
        and folds; it is re-conditioned per series via its in-context `fit`.
        `synthefy-nori` must be installed; an `ImportError` is raised
        otherwise.
        """

        if self._model is not None:
            return
        try:
            from synthefy_nori import NoriRegressor
        except ImportError as exc:
            raise ImportError(
                "synthefy-nori is required for NoriAdapter. "
                "Install it with `pip install synthefy-nori`."
            ) from exc

        # synthefy-nori has no default checkpoint. Its `model` argument accepts
        # either a registry name ('nori-6m') or a raw HuggingFace repo id, so
        # `model_id` is passed through. It is ignored when `model_path` is set.
        config = dict(self.nori_config)
        config.setdefault("model", self.model_id)

        self._model = NoriRegressor(**config)

    @staticmethod
    def _to_numpy(values: Any) -> np.ndarray:
        """
        Convert a model output to a float numpy array.

        Torch tensors (including those on a non-CPU device or requiring grad)
        are detached, moved to CPU, and converted before casting to `float`.

        Parameters
        ----------
        values : array-like
            Model output, either a numpy array or a torch tensor.

        Returns
        -------
        array : numpy ndarray
            Float numpy array.

        """

        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()

        return np.asarray(values, dtype=float)

    @staticmethod
    def _known_future_columns(
        ctx_exog: pd.DataFrame | pd.Series | None,
        fut_exog: pd.DataFrame | pd.Series | None,
    ) -> list:
        """
        Return covariate columns present in both context and future exog.

        Only known-future covariates (columns available over both the
        historical context and the forecast horizon) are usable by Nori.

        Parameters
        ----------
        ctx_exog : pandas DataFrame, pandas Series, or None
            Historical (past) covariates for a single series.
        fut_exog : pandas DataFrame, pandas Series, or None
            Future covariates for a single series.

        Returns
        -------
        columns : list
            Column names present in both `ctx_exog` and `fut_exog`, in the
            order they appear in `ctx_exog`. Empty when either input is `None`.

        """

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
        self,
        series: pd.Series,
        is_datetime: bool,
        exog_block: pd.DataFrame | pd.Series | None,
        exog_cols: list,
        offset: int,
        n: int,
    ) -> np.ndarray:
        """
        Build the tabular feature matrix for `n` rows starting at `offset`.

        Features are a running index, optional calendar features and Fourier
        seasonal harmonics (datetime series) or Fourier(index) terms
        (`RangeIndex` series), and the known-future covariate columns.

        Parameters
        ----------
        series : pandas Series
            The context series (used for its index and length).
        is_datetime : bool
            Whether the series has a `DatetimeIndex`.
        exog_block : pandas DataFrame, pandas Series, or None
            Covariate values for the rows being featurized (`context_exog` for
            the context block, `exog` for the horizon block).
        exog_cols : list
            Known-future covariate column names to include.
        offset : int
            Row offset of the block (`0` for the context, `len(series)` for the
            forecast horizon).
        n : int
            Number of rows to featurize.

        Returns
        -------
        X : numpy ndarray
            2-D `float32` feature matrix of shape `(n, n_features)`.

        """

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
            exog_values = np.column_stack(
                [self._to_float_array(block[col]) for col in exog_cols]
            )
            X = np.column_stack([X, exog_values])

        return X.astype(np.float32)

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

        Notes
        -----
        A `ValueError` is raised if the column is neither numeric nor
        boolean. Nori conditions only on numeric covariates; categoricals
        must be encoded as numbers.

        """

        if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            return col_data.astype(np.float32).to_numpy()

        raise ValueError(
            f"NoriAdapter supports only numeric covariates. Column "
            f"{col_data.name!r} has dtype {col_data.dtype}. Encode categorical "
            f"covariates as numeric values before passing them."
        )

    def _timestamps(
        self, series: pd.Series, offset: int, n: int
    ) -> pd.DatetimeIndex:
        """
        Return datetime timestamps for `n` rows starting at `offset`.

        For the context block (`offset == 0`) the series' own index is
        returned. For the forecast horizon the index is extended by `n` steps
        at the series' frequency (inferred when not set).

        Parameters
        ----------
        series : pandas Series
            The context series (used to determine the end timestamp and
            frequency).
        offset : int
            Row offset of the block. `0` selects the context index; any other
            value selects the extended forecast-horizon index.
        n : int
            Number of timestamps to return.

        Returns
        -------
        timestamps : pandas DatetimeIndex
            Datetime timestamps for the requested block.

        """

        if offset == 0:
            return series.index

        return expand_index(series.index, steps=n)


_ADAPTER_REGISTRY: dict[str, type] = {
    "amazon/chronos":     ChronosAdapter,
    "autogluon/chronos":  ChronosAdapter,
    "google/timesfm-2.5": TimesFM25Adapter,
    "google/timesfm-3.0": TimesFM3Adapter,
    "Salesforce/moirai":  MoiraiAdapter,
    "soda-inria/tabicl":  TabICLAdapter,
    "priorlabs/tabpfn":   TabPFNAdapter,
    "theforecastingcompany/t0": T0Adapter,
    "Synthefy/Nori":      NoriAdapter,
    "taharnbl/TS-ICL":    TSICLAdapter
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
