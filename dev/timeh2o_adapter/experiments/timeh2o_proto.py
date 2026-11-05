"""
Prototype of a TimeH2O adapter for skforecast, used only by the experiments of
this folder. It is registered at runtime (`register()`), the package is not
modified. It was written before PR #1450 and runs with and without it: it
declares `requires_equal_length_context` (required on `0.27.x` from fc7efd0,
harmless before) and passes `device` through `_resolve_torch_device` without
the MPS fallback. The reviewed design is the skeleton of section 4 of the
report (`skeleton/`), not this file.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from skforecast.foundation import _adapters, _utils
from skforecast.foundation._adapter_base import _AdapterBase
from skforecast.foundation._adapters import _resolve_torch_device
from skforecast.foundation._utils import (
    _validate_positive_int,
    _apply_set_params,
    get_exog_signature,
)


class TimeH2OAdapter(_AdapterBase):
    """
    Prototype adapter for H2O.ai TimeH2O (h2ofm).
    """

    SUPPORTED_QUANTILES: list[float] | None = None
    allow_exog: bool = True
    supports_past_only_covariates: bool = False
    supports_categorical_covariates: bool = True
    supports_heterogeneous_covariates: bool = True
    requires_equal_length_context: bool = False
    supports_nan_in_series: bool = True
    requires_hf_auth: bool = True
    requires_provider_auth: bool = False
    weights_repo_id: str | None = None
    weights_in_hf_cache: bool = True
    backend_package: str = "h2ofm>=0.43"
    default_model_id: str = "h2oai/timeh2o-v1"

    _RESERVED_PREDICT_KWARGS = frozenset({
        "contexts", "horizon", "past_covariates", "future_covariates",
        "quantile_levels", "context_length",
    })

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 4096,
        device: str = "auto",
        batch_size: int = 256,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _validate_positive_int("context_length", context_length)
        _validate_positive_int("batch_size", batch_size)
        self.model_id       = model_id
        self._model         = model
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.device         = device
        self.batch_size     = batch_size
        self.predict_kwargs = predict_kwargs or {}
        self.is_fitted      = False

    def get_params(self) -> dict:
        return {
            "model_id":       self.model_id,
            "context_length": self.context_length,
            "device":         self.device,
            "batch_size":     self.batch_size,
            "predict_kwargs": self.predict_kwargs or None,
        }

    def set_params(self, **params) -> TimeH2OAdapter:
        return _apply_set_params(
            self, params,
            resets=(
                (
                    {"model_id", "device", "batch_size"},
                    lambda: setattr(self, "_model", None),
                ),
            ),
        )

    def fit(self, context, context_exog):
        return super().fit(context=context, context_exog=context_exog)

    def predict(self, steps, context, context_exog, exog, quantiles):
        self._load_model()
        names = list(context.keys())
        requested = list(quantiles) if quantiles is not None else [0.5]
        query_levels = sorted(set(requested))
        future_covariates = [
            self._build_future_covariates(
                context_exog = (
                    context_exog.get(name) if context_exog is not None else None
                ),
                exog         = exog.get(name) if exog is not None else None,
            )
            for name in names
        ]
        forecasts = self._model.forecast_many(
            [
                context[name].to_numpy(dtype=np.float64, na_value=np.nan)
                for name in names
            ],
            horizon           = steps,
            future_covariates = (
                future_covariates
                if any(rows is not None for rows in future_covariates)
                else None
            ),
            quantile_levels   = query_levels,
            **self.predict_kwargs,
        )
        column_indices = [query_levels.index(q) for q in requested]
        return {
            name: np.asarray(forecast.quantiles)[:, column_indices]
            for name, forecast in zip(names, forecasts)
        }

    @staticmethod
    def _build_future_covariates(context_exog, exog):
        if exog is None:
            return None
        _, fut_cols = get_exog_signature(context_exog=context_exog, exog=exog)
        if not fut_cols:
            return None
        ctx_df = (
            context_exog.to_frame() if isinstance(context_exog, pd.Series)
            else context_exog
        )
        fut_df = exog.to_frame() if isinstance(exog, pd.Series) else exog
        return [
            pd.concat([ctx_df[col], fut_df[col]], ignore_index=True)
            for col in fut_cols
        ]

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from h2ofm import TimeH2O

        self._model = TimeH2O.from_pretrained(
            self.model_id,
            device=_resolve_torch_device(self.device),
            batch_size=self.batch_size,
        )


def register() -> None:
    """
    Register the prototype and its license at runtime.
    """
    _adapters._ADAPTER_REGISTRY["h2oai/timeh2o"] = TimeH2OAdapter
    _utils._MODEL_LICENSES["h2oai/timeh2o"] = (
        "h2o-model-non-commercial-licence-v1.0",
        "https://huggingface.co/h2oai/timeh2o-v1/blob/main/LICENSE",
        True,
        "H2O Model Non-Commercial Licence v1.0",
    )
