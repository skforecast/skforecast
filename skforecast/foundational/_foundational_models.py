################################################################################
#                           FoundationalModels                                 #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import numpy as np
import pandas as pd

from importlib.metadata import PackageNotFoundError, version

from ..utils import check_y, expand_index


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Capabilities exposed by a foundational model adapter.
    """

    supports_exog: bool
    supports_multivariate: bool
    supports_probabilistic: bool
    context_length: int | None
    min_history: int | None


class BaseAdapter:
    """
    Base adapter interface for foundational time-series models.
    """

    capabilities: ModelCapabilities

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self, data: pd.Series | pd.DataFrame, exog: pd.Series | pd.DataFrame | None = None
    ) -> None:
        raise NotImplementedError

    def forecast(
        self,
        h: int,
        exog: pd.Series | pd.DataFrame | None = None,
        quantiles: list[float] | tuple[float] | None = None,
    ) -> pd.Series | pd.DataFrame:
        raise NotImplementedError


class ChronosAdapter(BaseAdapter):
    """
    Adapter for Amazon Chronos models.
    """

    capabilities = ModelCapabilities(
        supports_exog=False,
        supports_multivariate=False,
        supports_probabilistic=True,
        context_length=None,
        min_history=None,
    )

    def __init__(
        self,
        model_id: str,
        *,
        pipeline: Any | None = None,
        context_length: int | None = None,
        num_samples: int = 20,
        predict_kwargs: dict[str, Any] | None = None,
        device_map: str | None = None,
        torch_dtype: Any | None = None,
    ) -> None:
        super().__init__(model_id=model_id)
        self._pipeline = pipeline
        self._history: pd.Series | None = None
        self.context_length = context_length
        self.num_samples = num_samples
        self.predict_kwargs = predict_kwargs or {}
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        reserved_keys = {"context", "prediction_length"}
        if reserved_keys.intersection(self.predict_kwargs):
            raise ValueError(
                "`predict_kwargs` cannot contain `context` or `prediction_length`."
            )

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return

        try:
            from chronos import ChronosPipeline, ChronosConfig
        except ImportError as exc:
            msg = (
                "Chronos is required for this model but is not installed. "
                "Install it with `pip install chronos-forecasting` (module name "
                "`chronos`), or pass a preloaded `pipeline=` when creating "
                "`FoundationalModels`."
            )
            raise ImportError(msg) from exc

        if _is_chronos_2_model(self.model_id):
            if not _chronos_supports_config_field(ChronosConfig, "input_patch_size"):
                raise TypeError(
                    "The installed `chronos-forecasting` does not support Chronos-2 "
                    "configs (`input_patch_size` missing). Upgrade with "
                    "`pip install -U chronos-forecasting transformers`, or use a "
                    "Chronos T5 checkpoint (e.g. `amazon/chronos-t5-small`)."
                )

        load_kwargs: dict[str, Any] = {}
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.torch_dtype

        try:
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id, **load_kwargs
            )
        except TypeError as exc:
            chronos_version = _get_package_version("chronos-forecasting")
            transformers_version = _get_package_version("transformers")
            msg = (
                "Chronos model config is incompatible with the installed "
                "`chronos-forecasting` package "
                f"(chronos-forecasting={chronos_version}, "
                f"transformers={transformers_version}). Try upgrading with "
                "`pip install -U chronos-forecasting transformers`, or use a "
                "Chronos model compatible with your installed versions."
            )
            raise TypeError(msg) from exc

    def _coerce_series(self, data: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("Chronos only supports univariate series for now.")
            series = data.iloc[:, 0]
            if series.name is None:
                series = series.rename(data.columns[0])
        else:
            series = data

        check_y(series, series_id="`data`")
        return series

    def _prepare_context(self) -> np.ndarray:
        if self._history is None:
            raise ValueError("Call `fit` before `forecast`.")

        values = self._history.to_numpy(dtype=float)
        if self.context_length is not None:
            values = values[-self.context_length :]

        return values

    def _to_pipeline_input(self, context: np.ndarray) -> Any:
        try:
            import torch
        except ImportError:
            return context
        return torch.as_tensor(context, dtype=torch.float32)

    def _predict_samples(self, context: np.ndarray, h: int) -> np.ndarray:
        predict_params = inspect.signature(self._pipeline.predict).parameters
        pipeline_input = self._to_pipeline_input(context)
        if "context" in predict_params:
            return self._pipeline.predict(
                context=pipeline_input,
                prediction_length=h,
                num_samples=self.num_samples,
                **self.predict_kwargs,
            )
        if "inputs" in predict_params:
            return self._pipeline.predict(
                inputs=pipeline_input,
                prediction_length=h,
                num_samples=self.num_samples,
                **self.predict_kwargs,
            )
        return self._pipeline.predict(
            pipeline_input,
            prediction_length=h,
            num_samples=self.num_samples,
            **self.predict_kwargs,
        )

    def _as_numpy(self, values: Any) -> np.ndarray:
        if isinstance(values, np.ndarray):
            return values
        if hasattr(values, "detach"):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    def _forecast_index(self, steps: int) -> pd.Index:
        if self._history is None:
            raise ValueError("Call `fit` before `forecast`.")

        index = self._history.index
        if isinstance(index, pd.DatetimeIndex) and index.freq is None:
            inferred = pd.infer_freq(index)
            if inferred is None:
                raise ValueError(
                    "DatetimeIndex must have a frequency or an inferrable frequency."
                )
            index = pd.DatetimeIndex(index, freq=inferred)

        return expand_index(index=index, steps=steps)

    def fit(
        self, data: pd.Series | pd.DataFrame, exog: pd.Series | pd.DataFrame | None = None
    ) -> None:
        if exog is not None:
            raise ValueError("Chronos models do not support `exog`.")

        series = self._coerce_series(data)
        self._history = series.copy()
        self._is_fitted = True

    def forecast(
        self,
        h: int,
        exog: pd.Series | pd.DataFrame | None = None,
        quantiles: list[float] | tuple[float] | None = None,
    ) -> pd.Series | pd.DataFrame:
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError("`h` must be a positive integer.")
        if exog is not None:
            raise ValueError("Chronos models do not support `exog`.")

        self._load_pipeline()
        context = self._prepare_context()

        raw_forecast = self._predict_samples(context=context, h=h)
        samples = self._as_numpy(raw_forecast)
        if samples.ndim == 3:
            samples = samples[0]
        if samples.ndim != 2:
            raise ValueError(
                "Unexpected Chronos output shape. Expected (n_samples, horizon)."
            )

        forecast_index = self._forecast_index(steps=h)
        series_name = (
            self._history.name if self._history is not None else "y"
        )

        if quantiles is None:
            point_forecast = np.median(samples, axis=0)
            return pd.Series(point_forecast, index=forecast_index, name=series_name)

        for q in quantiles:
            if q < 0 or q > 1:
                raise ValueError("All quantiles must be between 0 and 1.")

        quantiles = list(quantiles)
        quantile_values = np.quantile(samples, quantiles, axis=0).T
        columns = [f"q_{q}" for q in quantiles]
        return pd.DataFrame(quantile_values, index=forecast_index, columns=columns)


def _get_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _chronos_supports_config_field(config_cls: type, field_name: str) -> bool:
    try:
        params = inspect.signature(config_cls).parameters
    except (TypeError, ValueError):
        return False
    return field_name in params


def _is_chronos_2_model(model_id: str) -> bool:
    short_name = model_id.split("/", 1)[-1]
    return short_name.startswith("chronos-2")


def _resolve_adapter(model_id: str) -> type[BaseAdapter]:
    if model_id.split("/", 1)[-1].startswith("chronos"):
        return ChronosAdapter
    raise ValueError(
        "Unsupported model. Only Chronos models are supported for now."
    )


class FoundationalModels:
    """
    General interface for foundational time-series models.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        adapter_cls = _resolve_adapter(model_id=model)
        self.adapter = adapter_cls(model_id=model, **kwargs)

    @property
    def is_fitted(self) -> bool:
        return self.adapter.is_fitted

    def fit(
        self,
        data: pd.Series | pd.DataFrame,
        exog: pd.Series | pd.DataFrame | None = None,
    ) -> FoundationalModels:
        self.adapter.fit(data=data, exog=exog)
        return self

    def forecast(
        self,
        h: int,
        exog: pd.Series | pd.DataFrame | None = None,
        quantiles: list[float] | tuple[float] | None = None,
    ) -> pd.Series | pd.DataFrame:
        return self.adapter.forecast(h=h, exog=exog, quantiles=quantiles)
