################################################################################
#                           FoundationalModels                                 #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from ..utils import check_y, expand_index


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Immutable descriptor of a foundational model adapter's architectural
    capabilities.

    These fields describe what the underlying model architecture supports
    regardless of how a particular adapter instance is configured.
    Instance-level settings such as ``context_length`` are exposed as
    regular attributes on the adapter itself.

    Attributes
    ----------
    supports_exog : bool
        Whether the adapter accepts exogenous variables (past and/or future
        covariates).
    supports_multivariate : bool
        Whether the adapter can handle multivariate (multiple-series) input.
    supports_probabilistic : bool
        Whether the adapter can produce probabilistic forecasts (prediction
        intervals or quantiles).
    """

    supports_exog: bool
    supports_multivariate: bool
    supports_probabilistic: bool


class Chronos2Adapter:
    """
    Adapter for Amazon Chronos-2 foundational models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. ``"autogluon/chronos-2-small"``.
    context_length : int, optional
        Maximum number of historical observations to use as context. If None,
        the full history stored at fit time is used.
    pipeline : BaseChronosPipeline, optional
        Pre-loaded pipeline instance. If None, the pipeline is loaded lazily on
        the first call to ``predict``.
    device_map : str, optional
        Device map string forwarded to ``BaseChronosPipeline.from_pretrained``
        (e.g. ``"cuda"``, ``"cpu"``).
    torch_dtype : optional
        Torch dtype forwarded to ``BaseChronosPipeline.from_pretrained``.
    predict_kwargs : dict, optional
        Additional keyword arguments forwarded to the pipeline's
        ``predict_quantiles`` method.
    """

    capabilities = ModelCapabilities(
        supports_exog=True,
        supports_multivariate=False,
        supports_probabilistic=True,
    )

    def __init__(
        self,
        model_id: str,
        *,
        pipeline: Any | None = None,
        context_length: int | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        device_map: str | None = None,
        torch_dtype: Any | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. ``"autogluon/chronos-2-small"``.
        pipeline : BaseChronosPipeline, optional
            Pre-loaded pipeline instance. If ``None``, the pipeline is
            loaded lazily on the first call to ``predict``.
        context_length : int, optional
            Maximum number of historical observations to retain as context.
            At ``fit`` time only the last ``context_length`` observations
            of ``y`` (and ``exog``) are stored. At ``predict`` time any
            ``last_window`` longer than ``context_length`` is trimmed to
            this length before inference. If ``None``, no trimming is
            applied.
        predict_kwargs : dict, optional
            Additional keyword arguments forwarded verbatim to the
            pipeline's ``predict_quantiles`` method.
        device_map : str, optional
            Device map string forwarded to
            ``BaseChronosPipeline.from_pretrained`` (e.g. ``"cuda"``,
            ``"cpu"``, ``"auto"``).
        torch_dtype : optional
            Torch dtype forwarded to ``BaseChronosPipeline.from_pretrained``
            (e.g. ``torch.bfloat16``).
        """
        self.model_id = model_id
        self._pipeline = pipeline
        self._history: pd.Series | None = None
        self._history_exog: pd.DataFrame | pd.Series | None = None
        self.context_length = context_length
        self.predict_kwargs = predict_kwargs or {}
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._is_fitted = False

    def _load_pipeline(self) -> None:
        """
        Load the Chronos-2 pipeline into ``self._pipeline`` if not already set.

        The pipeline is imported lazily from ``chronos`` and instantiated via
        ``BaseChronosPipeline.from_pretrained``, which auto-dispatches to the
        correct pipeline class based on the model config. Optional
        ``device_map`` and ``torch_dtype`` stored at initialisation are
        forwarded to the constructor. This method is a no-op when
        ``self._pipeline`` is already populated.

        Raises
        ------
        ImportError
            If ``chronos-forecasting`` >=2.0 is not installed.
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

    def _build_chronos_input(
        self,
        target: np.ndarray,
        past_exog: pd.DataFrame | pd.Series | None = None,
        future_exog: pd.DataFrame | pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Build the input dict consumed by the pipeline's ``predict_quantiles`` method.

        Parameters
        ----------
        target : np.ndarray
            1-D array of observed time series values used as context. Must be
            castable to ``float64``.
        past_exog : pd.DataFrame or pd.Series, optional
            Historical exogenous variables whose index is aligned to
            ``target``. Each column (or the single Series, referenced by
            its name) becomes an entry in the returned
            ``"past_covariates"`` dict. Values are cast to ``float64``.
        future_exog : pd.DataFrame or pd.Series, optional
            Future-known exogenous variables covering the forecast horizon.
            Must have exactly ``prediction_length`` rows. Each column
            becomes an entry in the returned ``"future_covariates"`` dict.
            Values are cast to ``float64``.

        Returns
        -------
        dict
            Dictionary with mandatory key ``"target"`` (1-D ``float64``
            ``np.ndarray``) and optional keys ``"past_covariates"`` and
            ``"future_covariates"``, each mapping column names to 1-D
            ``float64`` arrays.
        """
        input_dict: dict[str, Any] = {"target": np.asarray(target, dtype=float)}
        if past_exog is not None:
            df = (
                past_exog
                if isinstance(past_exog, pd.DataFrame)
                else past_exog.to_frame()
            )
            input_dict["past_covariates"] = {
                col: np.asarray(df[col], dtype=float) for col in df.columns
            }
        if future_exog is not None:
            df = (
                future_exog
                if isinstance(future_exog, pd.DataFrame)
                else future_exog.to_frame()
            )
            input_dict["future_covariates"] = {
                col: np.asarray(df[col], dtype=float) for col in df.columns
            }
        return input_dict

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | pd.Series | None = None,
    ) -> Chronos2Adapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs — Chronos-2 is a zero-shot inference model.

        Parameters
        ----------
        y : pd.Series
            Training time series.
        exog : pd.DataFrame or pd.Series, optional
            Historical exogenous variables aligned to ``y``. Passed as
            ``past_covariates`` in the Chronos-2 input.

        Returns
        -------
        self : Chronos2Adapter
        """
        check_y(y, series_id="`y`")
        if self.context_length is not None:
            self._history = y.iloc[-self.context_length :].copy()
            self._history_exog = (
                exog.iloc[-self.context_length :].copy() if exog is not None else None
            )
        else:
            self._history = y.copy()
            self._history_exog = exog.copy() if exog is not None else None
        self._is_fitted = True
        return self

    def predict(
        self,
        steps: int,
        exog: pd.DataFrame | pd.Series | None = None,
        quantiles: list[float] | tuple[float] | None = None,
        last_window: pd.Series | None = None,
        last_window_exog: pd.DataFrame | pd.Series | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Generate predictions using the Chronos-2 pipeline.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        exog : pd.DataFrame or pd.Series, optional
            Future known exogenous variables for the forecast horizon. Passed as
            ``future_covariates`` in the Chronos-2 input. Must have exactly
            ``steps`` rows. Column names are independent of those passed to
            ``fit`` — Chronos-2 treats past and future covariates separately.
        quantiles : list of float, optional
            Quantile levels to return, e.g. ``[0.1, 0.5, 0.9]``. If None,
            returns a point forecast (median, i.e. the 0.5 quantile).
        last_window : pd.Series, optional
            Override the stored history with this window. Used by backtesting
            to pass the appropriate context window per fold.
        last_window_exog : pd.DataFrame or pd.Series, optional
            Historical exog corresponding to ``last_window``.

        Returns
        -------
        pd.Series
            Point forecast when ``quantiles`` is None.
        pd.DataFrame
            Quantile forecasts with columns ``q_0.1``, ``q_0.5``, etc. when
            ``quantiles`` is provided.
        """
        if not self._is_fitted and last_window is None:
            raise ValueError(
                "Call `fit` before `predict`, or pass `last_window`."
            )
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("`steps` must be a positive integer.")

        self._load_pipeline()

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

        quantile_levels = list(quantiles) if quantiles is not None else [0.1, 0.5, 0.9]

        if quantiles is not None:
            for q in quantile_levels:
                if not 0.0 <= q <= 1.0:
                    raise ValueError(
                        f"All quantiles must be between 0 and 1. Got {q}."
                    )

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


class FoundationalModels:
    """
    Lightweight user-facing interface for foundational time-series models.

    Currently supports Amazon Chronos-2 only. For full skforecast
    ecosystem integration (backtesting, model selection, etc.) use
    ``ForecasterFoundational`` instead.

    Parameters
    ----------
    model : str
        HuggingFace model ID, e.g. ``"autogluon/chronos-2-small"``.
    **kwargs :
        Additional keyword arguments forwarded to ``Chronos2Adapter``
        (``context_length``, ``pipeline``, ``device_map``, ``torch_dtype``,
        ``predict_kwargs``).

    Examples
    --------
    >>> import pandas as pd
    >>> from skforecast.foundational import FoundationalModels
    >>> y = pd.Series(range(50), index=pd.date_range("2020", periods=50, freq="ME"))
    >>> m = FoundationalModels("autogluon/chronos-2-small")
    >>> m.fit(y)  # doctest: +SKIP
    >>> pred = m.predict(steps=12)  # doctest: +SKIP
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        """
        Initialise the FoundationalModels interface.

        Parameters
        ----------
        model : str
            HuggingFace model ID, e.g. ``"autogluon/chronos-2-small"``.
        **kwargs :
            Additional keyword arguments forwarded to ``Chronos2Adapter``:
            ``context_length``, ``pipeline``, ``device_map``,
            ``torch_dtype``, ``predict_kwargs``.
        """
        self.adapter = Chronos2Adapter(model_id=model, **kwargs)

    @property
    def is_fitted(self) -> bool:
        """
        Whether the model has been fitted.

        Returns
        -------
        bool
            ``True`` after ``fit`` has been called at least once,
            ``False`` otherwise.
        """
        return self.adapter._is_fitted

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | pd.Series | None = None,
    ) -> FoundationalModels:
        """
        Fit the model by storing the training series and optional exog.

        Parameters
        ----------
        y : pd.Series
            Training time series.
        exog : pd.DataFrame or pd.Series, optional
            Historical exogenous variables aligned to ``y``.

        Returns
        -------
        self : FoundationalModels
        """
        self.adapter.fit(y=y, exog=exog)
        return self

    def predict(
        self,
        steps: int,
        exog: pd.DataFrame | pd.Series | None = None,
        quantiles: list[float] | tuple[float] | None = None,
        last_window: pd.Series | None = None,
        last_window_exog: pd.DataFrame | pd.Series | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Generate predictions.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        exog : pd.DataFrame or pd.Series, optional
            Future known exogenous variables for the forecast horizon.
        quantiles : list of float, optional
            Quantile levels to return, e.g. ``[0.1, 0.5, 0.9]``. If None,
            returns a point forecast (median).
        last_window : pd.Series, optional
            Override the stored history with this window.
        last_window_exog : pd.DataFrame or pd.Series, optional
            Historical exog corresponding to ``last_window``.

        Returns
        -------
        pd.Series
            Point forecast when ``quantiles`` is None.
        pd.DataFrame
            Quantile forecasts when ``quantiles`` is provided.
        """
        return self.adapter.predict(
            steps=steps,
            exog=exog,
            quantiles=quantiles,
            last_window=last_window,
            last_window_exog=last_window_exog,
        )
