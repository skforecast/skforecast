################################################################################
#                           FoundationalModels                                 #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from ..utils import check_y, expand_index


class Chronos2Adapter:
    """
    Adapter for Amazon Chronos-2 foundational models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "autogluon/chronos-2-small".
    context_length : int, optional
        Maximum number of historical observations to use as context. If None,
        the full history stored at fit time is used.
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

    def __init__(
        self,
        model_id: str,
        *,
        pipeline: Any | None = None,
        context_length: int | None = None,
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
        context_length : int, optional
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations
            of `series` (and `exog`) are stored. At `predict` time any
            `last_window` longer than `context_length` is trimmed to
            this length before inference. If `None`, no trimming is
            applied.
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
        self.model_id = model_id
        self._pipeline = pipeline
        self._history: pd.Series | dict[str, pd.Series] | None = None
        self._history_exog: (
            pd.DataFrame
            | pd.Series
            | dict[str, pd.DataFrame | pd.Series | None]
            | None
        ) = None
        self.context_length = context_length
        self.predict_kwargs = predict_kwargs or {}
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.cross_learning = cross_learning
        self._is_fitted = False
        self._is_multiseries = False

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
        if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating) or arr.dtype.kind == "b":
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
        input_dict: dict[str, Any] = {"target": np.asarray(target, dtype=float)}
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
        No model training occurs — Chronos-2 is a zero-shot inference model.

        Parameters
        ----------
        series : pd.Series, pd.DataFrame, or dict of pd.Series
            Training time series.

            - `pd.Series` — single-series mode.
            - Wide `pd.DataFrame` (each column = one series) — multi-series
              mode.
            - `dict[str, pd.Series]` — multi-series mode; keys become the
              series names.
        exog : pd.Series, pd.DataFrame, dict, or None
            Historical exogenous variables. Passed as `past_covariates` in
            the Chronos-2 input.

            - `pd.Series` or `pd.DataFrame` — broadcast to all series.
            - `dict[str, pd.DataFrame | pd.Series | None]` — per-series
              covariates; series absent from the dict receive no past
              covariates.

        Returns
        -------
        self : Chronos2Adapter
        """
        if isinstance(series, pd.Series):
            # --- single-series path (unchanged) ---
            check_y(series, series_id="`series`")
            self._is_multiseries = False
            if self.context_length is not None:
                self._history = series.iloc[-self.context_length :].copy()
                self._history_exog = (
                    exog.iloc[-self.context_length :].copy()
                    if exog is not None
                    else None
                )
            else:
                self._history = series.copy()
                self._history_exog = exog.copy() if exog is not None else None
        elif isinstance(series, (pd.DataFrame, dict)):
            # --- multi-series path ---
            if isinstance(series, pd.DataFrame):
                series_dict: dict[str, pd.Series] = {
                    col: series[col].copy() for col in series.columns
                }
            else:
                series_dict = {k: v.copy() for k, v in series.items()}

            if not series_dict:
                raise ValueError("`series` must contain at least one series.")

            for name, s in series_dict.items():
                if not isinstance(s, pd.Series):
                    raise TypeError(
                        f"All values in `series` must be pd.Series. "
                        f"Got {type(s)} for series '{name}'."
                    )
                check_y(s, series_id=f"'{name}'")
                series_dict[name].name = name

            series_names = list(series_dict.keys())
            exog_dict = self._normalize_exog_to_dict(exog, series_names)

            if self.context_length is not None:
                self._history = {
                    name: s.iloc[-self.context_length :].copy()
                    for name, s in series_dict.items()
                }
                self._history_exog = {
                    name: (
                        e.iloc[-self.context_length :].copy()
                        if e is not None
                        else None
                    )
                    for name, e in exog_dict.items()
                }
            else:
                self._history = series_dict
                self._history_exog = exog_dict

            self._is_multiseries = True
        else:
            raise TypeError(
                "`series` must be a pd.Series, a wide pd.DataFrame, or a "
                f"dict[str, pd.Series]. Got {type(series)}."
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
        quantile_levels: list[float],
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
        quantile_levels : list of float
            Quantile levels passed verbatim to `predict_quantiles`.
        quantiles : list of float or None
            Original `quantiles` argument from :meth:`predict`; used to
            choose between point and quantile output format.
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
        # --- resolve history dict ---
        if last_window is not None:
            if isinstance(last_window, pd.DataFrame):
                history_dict: dict[str, pd.Series] = {
                    col: last_window[col].copy() for col in last_window.columns
                }
            elif isinstance(last_window, dict):
                history_dict = {k: v.copy() for k, v in last_window.items()}
            else:
                raise TypeError(
                    "`last_window` must be a wide pd.DataFrame or a "
                    f"dict[str, pd.Series] in multi-series mode. "
                    f"Got {type(last_window)}."
                )
        else:
            history_dict = self._history  # already a dict[str, pd.Series]

        series_names = list(history_dict.keys())

        # --- resolve past exog dict ---
        if last_window is not None:
            past_exog_dict = self._normalize_exog_to_dict(last_window_exog, series_names)
        else:
            past_exog_dict = self._history_exog  # already a dict

        # --- resolve future exog dict ---
        future_exog_dict = self._normalize_exog_to_dict(exog, series_names)

        # --- context_length trimming (only when last_window is provided;
        #     _history was already trimmed at fit time) ---
        if last_window is not None and self.context_length is not None:
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

        # --- build list of per-series input dicts for the pipeline ---
        inputs_list = [
            self._build_chronos_input(
                target=history_dict[name].to_numpy(),
                past_exog=past_exog_dict[name],
                future_exog=future_exog_dict[name],
            )
            for name in series_names
        ]

        # --- single batched pipeline call ---
        quantile_preds, _ = self._pipeline.predict_quantiles(
            inputs=inputs_list,
            prediction_length=steps,
            quantile_levels=quantile_levels,
            cross_learning=self.cross_learning,
            **self.predict_kwargs,
        )

        # --- decode all per-series quantile arrays once ---
        # Each decoded[i] has shape (steps, n_q)
        decoded: list[np.ndarray] = []
        for i in range(len(series_names)):
            q_arr = quantile_preds[i]
            if hasattr(q_arr, "detach"):
                q_arr = q_arr.detach().cpu().numpy()
            else:
                q_arr = np.asarray(q_arr)
            decoded.append(q_arr.squeeze(0))

        # --- build shared long-format index and level column ---
        # Layout mirrors ForecasterRecursiveMultiSeries.predict():
        #   index  = [t0, t0, t1, t1, …] (each timestamp repeated n_series times)
        #   level  = [s1, s2, s1, s2, …] (series names tiled for each step)
        n_series = len(series_names)
        forecast_index = expand_index(history_dict[series_names[0]].index, steps=steps)
        long_index = np.repeat(forecast_index, n_series)
        level_col = np.tile(series_names, steps)

        if quantiles is None:
            # point forecast → long DataFrame with columns [level, pred]
            median_idx = quantile_levels.index(0.5)
            # shape (steps, n_series) → ravel row-major → [s1_t0, s2_t0, s1_t1, …]
            point_matrix = np.column_stack([decoded[i][:, median_idx] for i in range(n_series)])
            return pd.DataFrame(
                {"level": level_col, "pred": point_matrix.ravel()},
                index=long_index,
            )
        else:
            # quantile forecast → long DataFrame with columns [level, q_0.1, q_0.5, …]
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
            Override the stored history with this window. Used by backtesting
            to pass the appropriate context window per fold.

            - `pd.Series` — single-series mode.
            - Wide `pd.DataFrame` or `dict[str, pd.Series]` —
              multi-series mode.
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
            per series (`n_steps × n_series` rows total).
        """
        if not self._is_fitted and last_window is None:
            raise ValueError(
                "Call `fit` before `predict`, or pass `last_window`."
            )
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("`steps` must be a positive integer.")

        quantile_levels = list(quantiles) if quantiles is not None else [0.1, 0.5, 0.9]

        if quantiles is not None:
            for q in quantile_levels:
                if not 0.0 <= q <= 1.0:
                    raise ValueError(
                        f"All quantiles must be between 0 and 1. Got {q}."
                    )

        self._load_pipeline()

        if self._is_multiseries or isinstance(last_window, (pd.DataFrame, dict)):
            return self._predict_multiseries(
                steps=steps,
                exog=exog,
                quantile_levels=quantile_levels,
                quantiles=quantiles,
                last_window=last_window,
                last_window_exog=last_window_exog,
            )

        # --- single-series path ---
        history = last_window if last_window is not None else self._history
        past_exog = (
            last_window_exog if last_window is not None else self._history_exog
        )

        # When last_window is provided (backtesting), still trim to context_length.
        # When _history is used directly it was already trimmed at fit time.
        if last_window is not None and self.context_length is not None:
            history = history.iloc[-self.context_length :]
            if past_exog is not None:
                past_exog = past_exog.iloc[-self.context_length :]

        input_dict = self._build_chronos_input(
            target=history.to_numpy(),
            past_exog=past_exog,
            future_exog=exog,
        )

        quantile_preds, _ = self._pipeline.predict_quantiles(
            inputs=[input_dict],
            prediction_length=steps,
            quantile_levels=quantile_levels,
            **self.predict_kwargs,
        )

        # quantile_preds[0] shape: (n_vars, steps, n_q).
        # Univariate: n_vars == 1 — squeeze to (steps, n_q).
        q_arr = quantile_preds[0].squeeze(0)
        if hasattr(q_arr, "detach"):
            q_arr = q_arr.detach().cpu().numpy()

        forecast_index = expand_index(history.index, steps=steps)

        if quantiles is None:
            median_idx = quantile_levels.index(0.5)
            return pd.Series(
                q_arr[:, median_idx],
                index=forecast_index,
                name=history.name,
            )

        columns = [f"q_{q}" for q in quantile_levels]
        return pd.DataFrame(q_arr, index=forecast_index, columns=columns)


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
        Additional keyword arguments forwarded to `Chronos2Adapter`
        (`context_length`, `pipeline`, `device_map`, `torch_dtype`,
        `predict_kwargs`).

    """

    def __init__(
        self,
        model: str,
        *,
        cross_learning: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the FoundationalModels interface.

        Parameters
        ----------
        model : str
            HuggingFace model ID, e.g. "autogluon/chronos-2-small".
        cross_learning : bool, default False
            If `True`, Chronos-2 shares information across all series in
            the batch when predicting in multi-series mode. Ignored in
            single-series mode.
        **kwargs :
            Additional keyword arguments forwarded to `Chronos2Adapter`:
            `context_length`, `pipeline`, `device_map`,
            `torch_dtype`, `predict_kwargs`.
        """
        self.adapter = Chronos2Adapter(
            model_id=model, cross_learning=cross_learning, **kwargs
        )

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

            - `pd.Series` — single-series mode.
            - Wide `pd.DataFrame` (each column = one series) — multi-series
              mode.
            - `dict[str, pd.Series]` — multi-series mode; keys are series
              names.
        exog : pd.Series, pd.DataFrame, dict, or None
            Historical exogenous variables aligned to `series`.

            - `pd.Series` or `pd.DataFrame` — broadcast to all series.
            - `dict[str, pd.DataFrame | pd.Series | None]` — per-series.

        Returns
        -------
        self : FoundationalModels
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

            - `pd.Series` — single-series override.
            - Wide `pd.DataFrame` or `dict[str, pd.Series]` —
              multi-series override.
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
            format with columns `["level", "q_0.1", "q_0.5", …]`. In both
            multi-series cases the index repeats each forecast timestamp once
            per series (`n_steps x n_series` rows total).
        """
        return self.adapter.predict(
            steps=steps,
            exog=exog,
            quantiles=quantiles,
            last_window=last_window,
            last_window_exog=last_window_exog,
        )
