################################################################################
#                        Foundational Model Adapters                           #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8
# Each adapter imports its own backend library lazily (i.e. inside the method
# that first needs it) rather than at module level. This means that only the
# library required by the adapter you actually use needs to be installed, other
# foundational-model backends remain optional.

from __future__ import annotations
from typing import Any
import warnings
import numpy as np
import pandas as pd
from ..exceptions import IgnoredArgumentWarning
from ..utils import check_y, expand_index


class Chronos2Adapter:
    """
    Adapter for Amazon Chronos-2 foundational models.

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

        self.model_id        = model_id
        self._pipeline       = pipeline
        self._history        = None
        self._history_exog   = None
        self.context_length  = context_length
        self.predict_kwargs  = predict_kwargs or {}
        self.device_map      = device_map
        self.torch_dtype     = torch_dtype
        self.cross_learning  = cross_learning
        self._is_fitted      = False
        self._is_multiseries = False

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

    @staticmethod
    def _normalize_exog_to_dict(
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ),
        series_names: list[str],
    ) -> dict[str, pd.DataFrame | pd.Series | None]:
        """
        Normalise exog to a per-series dict keyed by series name.

        Parameters
        ----------
        exog : pd.Series, pd.DataFrame, dict, or None
            Exogenous variables to normalise.

            - `None` → all series mapped to `None`.
            - `pd.Series` or `pd.DataFrame` → broadcast identically to
              every series.
            - `dict` → values are kept per-series; series whose keys are
              absent are mapped to `None`.
        series_names : list of str
            Ordered list of series identifiers that define the output keys.

        Returns
        -------
        dict[str, pd.DataFrame | pd.Series | None]
            Per-series exog dict with exactly the keys in `series_names`.
        """

        if exog is None:
            return {name: None for name in series_names}
        if isinstance(exog, dict):
            return {name: exog.get(name, None) for name in series_names}
        # broadcast flat Series or wide DataFrame to every series
        return {name: exog for name in series_names}

    def fit(
        self,
        series: pd.Series | pd.DataFrame | dict[str, pd.Series],
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> Chronos2Adapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since Chronos-2 is a zero-shot inference model.

        Parameters
        ----------
        series : pd.Series, pd.DataFrame, or dict of pd.Series
            Training time series. Based on the type of `series`, the adapter
            operates in single-series or multi-series mode.

            - `pd.Series`: single-series mode.
            - Wide `pd.DataFrame` (each column = one series): multi-series
              mode.
            - `dict[str, pd.Series]`: multi-series mode; keys become the
              series names.
        exog : pd.Series, pd.DataFrame, dict, or None
            Historical exogenous variables. Passed as `past_covariates` in
            the Chronos-2 input.

            - `pd.Series` or `pd.DataFrame`: broadcast to all series.
            - `dict[str, pd.DataFrame | pd.Series | None]`: per-series
              covariates; series absent from the dict receive no past
              covariates.

        Returns
        -------
        self : Chronos2Adapter
        """

        # single-series path
        if isinstance(series, pd.Series):
            
            check_y(series, series_id="`series`")
            self._is_multiseries = False
            self._history = series.iloc[-self.context_length :].copy()
            self._history_exog = (
                exog.iloc[-self.context_length :].copy()
                if exog is not None
                else None
            )
    
        # multi-series path
        elif isinstance(series, (pd.DataFrame, dict)):
            
            if isinstance(series, pd.DataFrame):
                series_dict: dict[str, pd.Series] = {
                    col: series[col] for col in series.columns
                }
            else:
                series_dict = series

            if not series_dict:
                raise ValueError("`series` must contain at least one series.")

            for name, s in series_dict.items():
                if not isinstance(s, pd.Series):
                    raise TypeError(
                        f"All values in `series` must be pd.Series. "
                        f"Got {type(s)} for series '{name}'."
                    )
                check_y(s, series_id=f"'{name}'")

            series_names = list(series_dict.keys())
            exog_dict = self._normalize_exog_to_dict(exog, series_names)

            self._history = {
                name: series.iloc[-self.context_length :].copy()
                for name, series in series_dict.items()
            }
            self._history_exog = {
                name: (
                    exog.iloc[-self.context_length :].copy()
                    if exog is not None
                    else None
                )
                for name, exog in exog_dict.items()
            }

            self._is_multiseries = True
        else:
            raise TypeError(
                "`series` must be a pd.Series, a wide pd.DataFrame (one column "
                "per series), or a dict[str, pd.Series]. "
                f"Got {type(series)}. "
                "To use a long-format DataFrame, pass it to `ForecasterFoundational`, "
                "which converts it automatically before calling this adapter."
            )

        self._is_fitted = True

        return self

    def _predict_multiseries(
        self,
        steps: int,
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ),
        quantiles: list[float] | None,
        last_window: pd.DataFrame | dict[str, pd.Series] | None,
        last_window_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ),
    ) -> pd.DataFrame:
        """
        Internal multi-series prediction logic.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        exog : pd.Series, pd.DataFrame, dict, or None
            Future covariates for the forecast horizon. Broadcast or
            per-series dict; see :meth:`_normalize_exog_to_dict`.
        quantiles : list of float or None
            Quantile levels to return. If `None`, a point forecast (median)
            is produced; only `[0.5]` is requested from the pipeline.
        last_window : pd.DataFrame, dict, or None
            Override stored history. Wide DataFrame (column per series) or
            `dict[str, pd.Series]`.
        last_window_exog : pd.Series, pd.DataFrame, dict, or None
            Past covariates corresponding to `last_window`.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame. For point forecasts, columns are
            `["level", "pred"]`. For quantile forecasts, columns are
            `["level", "q_0.1", "q_0.5", ...]`. The index repeats the
            forecast timestamps once per series (`n_steps × n_series`
            rows total); the `level` column identifies the series.
        """

        quantile_levels = list(quantiles) if quantiles is not None else [0.5]

        if last_window is not None:
            if isinstance(last_window, pd.DataFrame):
                history_dict: dict[str, pd.Series] = {
                    col: last_window[col] for col in last_window.columns
                }
            elif isinstance(last_window, dict):
                history_dict = last_window
            else:
                raise TypeError(
                    "`last_window` must be a wide pd.DataFrame or a "
                    f"dict[str, pd.Series] in multi-series mode. "
                    f"Got {type(last_window)}."
                )
        else:
            history_dict = self._history

        series_names = list(history_dict.keys())

        if last_window is not None:
            past_exog_dict = self._normalize_exog_to_dict(last_window_exog, series_names)
        else:
            past_exog_dict = self._history_exog  # already a dict

        future_exog_dict = self._normalize_exog_to_dict(exog, series_names)

        # Trim to context_length when last_window is provided (backtesting)
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
        #  index = [t0, t0, t1, t1, …] (each timestamp repeated n_series times)
        #  level = [s1, s2, s1, s2, …] (series names tiled for each step)
        n_series = len(series_names)
        forecast_index = expand_index(history_dict[series_names[0]].index, steps=steps)
        long_index = np.repeat(forecast_index, n_series)
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
        Generate predictions using the Chronos-2 pipeline.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        exog : pd.Series, pd.DataFrame, dict, or None
            Future known exogenous variables for the forecast horizon. Passed
            as `future_covariates` in the Chronos-2 input. Must cover
            exactly `steps` steps for each series.

            - `pd.Series` or `pd.DataFrame` — used as-is (single-series)
              or broadcast to every series (multi-series).
            - `dict[str, pd.DataFrame | pd.Series | None]` — per-series
              future covariates (multi-series only).
        quantiles : list of float, optional
            Quantile levels to return, e.g. `[0.1, 0.5, 0.9]`. If None,
            returns a point forecast (median, i.e. the 0.5 quantile).
        last_window : pd.Series, pd.DataFrame, dict, or None
            Override the stored history with this window. When provided,
            replaces the history stored at fit time. Typically supplied by
            backtesting to pass the appropriate context per fold. If longer
            than `context_length`, only the last `context_length` observations
            are used; if shorter, all available observations are passed as-is.
            Based on the type of `last_window`, the output is generated via
            the single-series or multi-series path.

            - `pd.Series`: single-series mode.
            - Wide `pd.DataFrame` or `dict[str, pd.Series]`: multi-series mode.
        last_window_exog : pd.Series, pd.DataFrame, dict, or None
            Historical exog corresponding to `last_window`. Same
            broadcast-vs-dict semantics as `exog`.

        Returns
        -------
        pd.Series
            Point forecast when `quantiles` is None and input is a single
            series.
        pd.DataFrame
            Single-series with quantiles: columns are `q_0.1`, `q_0.5`,
            etc. Multi-series point forecast: long format with columns
            `["level", "pred"]`. Multi-series quantile forecast: long
            format with columns `["level", "q_0.1", "q_0.5", …]`. In both
            multi-series cases the index repeats each forecast timestamp once
            per series (`n_steps x n_series` rows total).
        """

        if not self._is_fitted and last_window is None:
            raise ValueError(
                "Call `fit` before `predict`, or pass `last_window`."
            )
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("`steps` must be a positive integer.")

        quantile_levels = list(quantiles) if quantiles is not None else [0.5]

        if quantiles is not None:
            for q in quantile_levels:
                if not 0.0 <= q <= 1.0:
                    raise ValueError(
                        f"All quantiles must be between 0 and 1. Got {q}."
                    )

        # NOTE: the pipeline is loaded lazily here so that the adapter can be
        # instantiated and fitted without requiring Chronos-2 to be installed.
        self._load_pipeline()

        if self._is_multiseries or isinstance(last_window, (pd.DataFrame, dict)):
            return self._predict_multiseries(
                steps            = steps,
                exog             = exog,
                quantiles        = quantiles,
                last_window      = last_window,
                last_window_exog = last_window_exog,
            )

        history = last_window if last_window is not None else self._history
        past_exog = (
            last_window_exog if last_window is not None else self._history_exog
        )

        # Trim to context_length when last_window is provided (backtesting).
        if last_window is not None:
            history = history.iloc[-self.context_length :]
            if past_exog is not None:
                past_exog = past_exog.iloc[-self.context_length :]

        input_dict = self._build_chronos_input(
            target      = history.to_numpy(),
            past_exog   = past_exog,
            future_exog = exog,
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

        if quantiles is None:
            return pd.Series(
                data  = q_arr[:, 0],
                index = forecast_index,
                name  = history.name,
            )

        columns = [f"q_{q}" for q in quantile_levels]
        return pd.DataFrame(q_arr, index=forecast_index, columns=columns)


class TimesFM25Adapter:
    """
    Adapter for Google TimesFM 2.5 foundational models.

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
        Maximum forecast horizon baked into the compiled decode function.
        A `ValueError` is raised if `predict` is called with
        `steps > max_horizon`; recreate the adapter with a larger value in
        that case. Must be a positive integer.
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
            Maximum forecast horizon baked into the compiled decode function.
            A `ValueError` is raised at predict time if
            `steps > max_horizon`. Must be a positive integer.
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
        self._is_fitted             = False
        self._is_multiseries        = False

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
        Load and compile the TimesFM 2.5 model into `self._model` if not
        already set.

        The model is imported lazily from `timesfm` and loaded via
        `TimesFM_2p5_200M_torch.from_pretrained`, then compiled with a
        `ForecastConfig` derived from `context_length`, `max_horizon`
        and `forecast_config_kwargs`. This method is a no-op when
        `self._model` is already populated.

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

        self._model.compile(
            timesfm.ForecastConfig(
                max_context=self.context_length,
                max_horizon=self.max_horizon,
                **self.forecast_config_kwargs,
            )
        )

    def fit(
        self,
        series: pd.Series | pd.DataFrame | dict[str, pd.Series],
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> TimesFM25Adapter:
        """
        Store the training series as context.

        No model training occurs since TimesFM 2.5 is a zero-shot inference
        model.

        Parameters
        ----------
        series : pd.Series, pd.DataFrame, or dict of pd.Series
            Training time series.

            - `pd.Series`: single-series mode.
            - Wide `pd.DataFrame` (each column = one series): multi-series
              mode.
            - `dict[str, pd.Series]`: multi-series mode; keys become the
              series names.
        exog : ignored
            Accepted for API compatibility. Covariate support is not yet
            implemented. Issues an `IgnoredArgumentWarning` if not `None`.

        Returns
        -------
        self : TimesFM25Adapter
        """

        if exog is not None:
            warnings.warn(
                "TimesFM25Adapter does not currently support covariates. "
                "`exog` is ignored.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )

        # single-series path
        if isinstance(series, pd.Series):
            check_y(series, series_id="`series`")
            self._is_multiseries = False
            self._history = series.iloc[-self.context_length :].copy()

        # multi-series path
        elif isinstance(series, (pd.DataFrame, dict)):
            if isinstance(series, pd.DataFrame):
                series_dict: dict[str, pd.Series] = {
                    col: series[col] for col in series.columns
                }
            else:
                series_dict = series

            if not series_dict:
                raise ValueError("`series` must contain at least one series.")

            for name, s in series_dict.items():
                if not isinstance(s, pd.Series):
                    raise TypeError(
                        f"All values in `series` must be pd.Series. "
                        f"Got {type(s)} for series '{name}'."
                    )
                check_y(s, series_id=f"'{name}'")

            self._history = {
                name: s.iloc[-self.context_length :].copy()
                for name, s in series_dict.items()
            }
            self._is_multiseries = True

        else:
            raise TypeError(
                "`series` must be a pd.Series, a wide pd.DataFrame (one column "
                "per series), or a dict[str, pd.Series]. "
                f"Got {type(series)}. "
                "To use a long-format DataFrame, pass it to `ForecasterFoundational`, "
                "which converts it automatically before calling this adapter."
            )

        self._is_fitted = True
        return self

    def _predict_multiseries(
        self,
        steps: int,
        quantiles: list[float] | None,
        last_window: pd.DataFrame | dict[str, pd.Series] | None,
    ) -> pd.DataFrame:
        """
        Internal multi-series prediction logic.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        quantiles : list of float or None
            Quantile levels to return from `SUPPORTED_QUANTILES`. If
            `None`, a point forecast is produced.
        last_window : pd.DataFrame, dict, or None
            Override stored history. Wide DataFrame (column per series) or
            `dict[str, pd.Series]`.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame. For point forecasts, columns are
            `["level", "pred"]`. For quantile forecasts, columns are
            `["level", "q_0.1", "q_0.5", ...]`. The index repeats the
            forecast timestamps once per series (`n_steps x n_series`
            rows total); the `level` column identifies the series.
        """

        if last_window is not None:
            if isinstance(last_window, pd.DataFrame):
                history_dict: dict[str, pd.Series] = {
                    col: last_window[col] for col in last_window.columns
                }
            elif isinstance(last_window, dict):
                history_dict = last_window
            else:
                raise TypeError(
                    "`last_window` must be a wide pd.DataFrame or a "
                    f"dict[str, pd.Series] in multi-series mode. "
                    f"Got {type(last_window)}."
                )
        else:
            history_dict = self._history

        series_names = list(history_dict.keys())

        # Trim to context_length when last_window is provided (backtesting)
        if last_window is not None:
            history_dict = {
                name: s.iloc[-self.context_length :]
                for name, s in history_dict.items()
            }

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
        forecast_index = expand_index(history_dict[series_names[0]].index, steps=steps)
        long_index = np.repeat(forecast_index, n_series)
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
        Generate predictions using the TimesFM 2.5 model.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast. Must not exceed `max_horizon`.
        exog : ignored
            Accepted for API compatibility. Issues an
            `IgnoredArgumentWarning` if not `None`.
        quantiles : list of float, optional
            Quantile levels to return. Must be a subset of
            `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. If
            `None`, returns a point forecast (mean).
        last_window : pd.Series, pd.DataFrame, dict, or None
            Override the stored history with this window. Typically supplied
            by backtesting to pass the appropriate context per fold. If
            longer than `context_length`, only the last
            `context_length` observations are used.

            - `pd.Series`: single-series mode.
            - Wide `pd.DataFrame` or `dict[str, pd.Series]`:
              multi-series mode.
        last_window_exog : ignored
            Accepted for API compatibility. Issues an
            `IgnoredArgumentWarning` if not `None`.

        Returns
        -------
        pd.Series
            Point forecast when `quantiles` is `None` and input is a
            single series.
        pd.DataFrame
            Single-series with quantiles: columns are `q_0.1`,
            `q_0.5`, etc. Multi-series point forecast: long format with
            columns `["level", "pred"]`. Multi-series quantile forecast:
            long format with columns `["level", "q_0.1", "q_0.5", ...]`.
            In both multi-series cases the index repeats each forecast
            timestamp once per series (`n_steps x n_series` rows total).

        Raises
        ------
        ValueError
            If `steps > max_horizon` or a requested quantile level is not
            in `SUPPORTED_QUANTILES`.
        """

        if not self._is_fitted and last_window is None:
            raise ValueError(
                "Call `fit` before `predict`, or pass `last_window`."
            )
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("`steps` must be a positive integer.")

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

        if exog is not None or last_window_exog is not None:
            warnings.warn(
                "TimesFM25Adapter does not currently support covariates. "
                "`exog` and `last_window_exog` are ignored.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )

        self._load_model()

        if steps > self.max_horizon:
            raise ValueError(
                f"`steps` ({steps}) exceeds `max_horizon` ({self.max_horizon}). "
                f"Recreate the adapter with `max_horizon >= {steps}`."
            )

        if self._is_multiseries or isinstance(last_window, (pd.DataFrame, dict)):
            return self._predict_multiseries(
                steps       = steps,
                quantiles   = quantile_list,
                last_window = last_window,
            )

        # single-series path
        history = last_window if last_window is not None else self._history

        # Trim to context_length when last_window is provided (backtesting)
        if last_window is not None:
            history = history.iloc[-self.context_length :]

        point_forecast, quantile_forecast = self._model.forecast(
            horizon=steps,
            inputs=[history.to_numpy()],
        )
        # point_forecast  : (1, steps)
        # quantile_forecast: (1, steps, 10)  — idx 0 = mean, 1-9 = q0.1-q0.9

        forecast_index = expand_index(history.index, steps=steps)

        if quantile_list is None:
            return pd.Series(
                data  = np.asarray(point_forecast[0]),
                index = forecast_index,
                name  = history.name,
            )

        q_indices = [round(q * 10) for q in quantile_list]
        qf        = np.asarray(quantile_forecast[0])  # (steps, 10)
        columns   = [f"q_{q}" for q in quantile_list]
        return pd.DataFrame(
            {col: qf[:, q_indices[j]] for j, col in enumerate(columns)},
            index=forecast_index,
        )


class MoiraiAdapter:
    """
    Adapter for Salesforce Moirai-2 foundational models.

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
        self._is_fitted      = False
        self._is_multiseries = False

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
        series: pd.Series | pd.DataFrame | dict[str, pd.Series],
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> MoiraiAdapter:
        """
        Store the training series as context.

        No model training occurs since Moirai-2 is a zero-shot inference
        model.

        Parameters
        ----------
        series : pd.Series, pd.DataFrame, or dict of pd.Series
            Training time series.

            - `pd.Series`: single-series mode.
            - Wide `pd.DataFrame` (each column = one series): multi-series
              mode.
            - `dict[str, pd.Series]`: multi-series mode; keys become the
              series names.
        exog : ignored
            Accepted for API compatibility. Covariate support is not
            implemented. Issues an `IgnoredArgumentWarning` if not `None`.

        Returns
        -------
        self : MoiraiAdapter
        """

        if exog is not None:
            warnings.warn(
                "MoiraiAdapter does not support covariates. "
                "`exog` is ignored.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )

        # single-series path
        if isinstance(series, pd.Series):
            check_y(series, series_id="`series`")
            self._is_multiseries = False
            self._history = series.iloc[-self.context_length :].copy()

        # multi-series path
        elif isinstance(series, (pd.DataFrame, dict)):
            if isinstance(series, pd.DataFrame):
                series_dict: dict[str, pd.Series] = {
                    col: series[col] for col in series.columns
                }
            else:
                series_dict = series

            if not series_dict:
                raise ValueError("`series` must contain at least one series.")

            for name, s in series_dict.items():
                if not isinstance(s, pd.Series):
                    raise TypeError(
                        f"All values in `series` must be pd.Series. "
                        f"Got {type(s)} for series '{name}'."
                    )
                check_y(s, series_id=f"'{name}'")

            self._history = {
                name: s.iloc[-self.context_length :].copy()
                for name, s in series_dict.items()
            }
            self._is_multiseries = True

        else:
            raise TypeError(
                "`series` must be a pd.Series, a wide pd.DataFrame (one column "
                "per series), or a dict[str, pd.Series]. "
                f"Got {type(series)}. "
                "To use a long-format DataFrame, pass it to "
                "`ForecasterFoundational`, which converts it automatically "
                "before calling this adapter."
            )

        self._is_fitted = True
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
        last_window: pd.DataFrame | dict[str, pd.Series] | None,
    ) -> pd.DataFrame:
        """
        Internal multi-series prediction logic.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        quantiles : list of float or None
            Quantile levels to return from `SUPPORTED_QUANTILES`. If
            `None`, a point forecast (median, q=0.5) is produced.
        last_window : pd.DataFrame, dict, or None
            Override stored history. Wide DataFrame (column per series) or
            `dict[str, pd.Series]`.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame. For point forecasts, columns are
            `["level", "pred"]`. For quantile forecasts, columns are
            `["level", "q_0.1", "q_0.5", ...]`. The index repeats the
            forecast timestamps once per series (`n_steps x n_series`
            rows total); the `level` column identifies the series.
        """

        if last_window is not None:
            if isinstance(last_window, pd.DataFrame):
                history_dict: dict[str, pd.Series] = {
                    col: last_window[col] for col in last_window.columns
                }
            elif isinstance(last_window, dict):
                history_dict = last_window
            else:
                raise TypeError(
                    "`last_window` must be a wide pd.DataFrame or a "
                    f"dict[str, pd.Series] in multi-series mode. "
                    f"Got {type(last_window)}."
                )
        else:
            history_dict = self._history

        series_names = list(history_dict.keys())

        # Trim to context_length when last_window is provided (backtesting)
        if last_window is not None:
            history_dict = {
                name: s.iloc[-self.context_length :]
                for name, s in history_dict.items()
            }

        # Build batched input list: each array is (T, 1) float64
        inputs_list = [
            history_dict[name].to_numpy(dtype=float).reshape(-1, 1)
            for name in series_names
        ]

        raw = self._run_inference(inputs_list, steps)

        n_series = len(series_names)
        forecast_index = expand_index(
            history_dict[series_names[0]].index, steps=steps
        )
        long_index = np.repeat(forecast_index, n_series)
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
        exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
        quantiles: list[float] | tuple[float] | None = None,
        last_window: (
            pd.Series | pd.DataFrame | dict[str, pd.Series] | None
        ) = None,
        last_window_exog: (
            pd.Series
            | pd.DataFrame
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Generate predictions using Moirai-2.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        exog : ignored
            Accepted for API compatibility. Issues an
            `IgnoredArgumentWarning` if not `None`.
        quantiles : list of float, optional
            Quantile levels to return. Must be a subset of
            `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. If
            `None`, returns a point forecast (median, q=0.5).
        last_window : pd.Series, pd.DataFrame, dict, or None
            Override the stored history with this window. Typically
            supplied by backtesting to pass the appropriate context per
            fold. If longer than `context_length`, only the last
            `context_length` observations are used; if shorter, all
            available observations are passed as-is.

            - `pd.Series`: single-series mode.
            - Wide `pd.DataFrame` or `dict[str, pd.Series]`:
              multi-series mode.
        last_window_exog : ignored
            Accepted for API compatibility. Issues an
            `IgnoredArgumentWarning` if not `None`.

        Returns
        -------
        pd.Series
            Point forecast when `quantiles` is `None` and input is a
            single series.
        pd.DataFrame
            Single-series with quantiles: columns are `q_0.1`,
            `q_0.5`, etc. Multi-series point forecast: long format with
            columns `["level", "pred"]`. Multi-series quantile forecast:
            long format with columns `["level", "q_0.1", "q_0.5", ...]`.
            In both multi-series cases the index repeats each forecast
            timestamp once per series (`n_steps x n_series` rows total).

        Raises
        ------
        ValueError
            If a requested quantile level is not in `SUPPORTED_QUANTILES`.
        """

        if not self._is_fitted and last_window is None:
            raise ValueError(
                "Call `fit` before `predict`, or pass `last_window`."
            )
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("`steps` must be a positive integer.")

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

        if exog is not None or last_window_exog is not None:
            warnings.warn(
                "MoiraiAdapter does not support covariates. "
                "`exog` and `last_window_exog` are ignored.",
                IgnoredArgumentWarning,
                stacklevel=2,
            )

        if self._is_multiseries or isinstance(last_window, (pd.DataFrame, dict)):
            return self._predict_multiseries(
                steps       = steps,
                quantiles   = quantile_list,
                last_window = last_window,
            )

        # single-series path
        history = last_window if last_window is not None else self._history

        # Trim to context_length when last_window is provided (backtesting)
        if last_window is not None:
            history = history.iloc[-self.context_length :]

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
        result = raw[0][q_indices, :].T  # (steps, n_q)

        forecast_index = expand_index(history.index, steps=steps)

        if quantile_list is None:
            return pd.Series(
                data  = result[:, 0],
                index = forecast_index,
                name  = history.name,
            )

        columns = [f"q_{q}" for q in quantile_list]
        return pd.DataFrame(result, index=forecast_index, columns=columns)


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


