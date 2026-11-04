"""
Prototype of a Timestral adapter for skforecast, used only by the experiments
of this folder. It is registered at runtime (`register()`), the package is not
modified.
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
    _validate_supported_quantiles,
    _tensor_to_numpy,
    _apply_set_params,
    get_exog_signature,
)

TIMESTRAL_QUANTILES = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
]


def _left_pad(arrays: list[np.ndarray], width: int) -> np.ndarray:
    """
    Stack 1-D or 2-D (rows, time) arrays right-aligned on the time axis,
    padding the start with NaN. Candidate for a shared helper (T0 does the
    same inline).
    """
    first = arrays[0]
    shape = (len(arrays),) + first.shape[:-1] + (width,)
    out = np.full(shape, np.nan, dtype=np.float64)
    for i, a in enumerate(arrays):
        out[i, ..., width - a.shape[-1]:] = a
    return out


def _is_categorical(col: pd.Series) -> bool:
    return not (
        pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)
    )


class TimestralAdapter(_AdapterBase):

    SUPPORTED_QUANTILES: list[float] = TIMESTRAL_QUANTILES
    allow_exog: bool = True
    supports_past_only_covariates: bool = True
    supports_categorical_covariates: bool = True
    supports_heterogeneous_covariates: bool = False
    supports_nan_in_series: bool = True
    requires_hf_auth: bool = False
    requires_provider_auth: bool = False
    weights_repo_id: str | None = None
    weights_in_hf_cache: bool = True
    backend_package: str = "git+https://github.com/mistralai/timestral.git"
    default_model_id: str = "mistralai/Timestral-28M"

    _RESERVED_PREDICT_KWARGS = frozenset({
        "context", "prediction_length", "observation_mask", "past_covariates",
        "known_covariates", "categorical_past_covariates",
        "categorical_known_covariates",
    })

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 8192,
        device: str = "auto",
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _validate_positive_int("context_length", context_length)
        predict_kwargs = predict_kwargs or {}
        reserved = self._RESERVED_PREDICT_KWARGS & set(predict_kwargs)
        if reserved:
            raise ValueError(f"`predict_kwargs` cannot include {sorted(reserved)}.")
        self.model_id       = model_id
        self._model         = model
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.device         = device
        self.predict_kwargs = predict_kwargs
        self.is_fitted      = False

    def get_params(self) -> dict:
        return {
            "model_id":       self.model_id,
            "context_length": self.context_length,
            "device":         self.device,
            "predict_kwargs": self.predict_kwargs or None,
        }

    def set_params(self, **params) -> TimestralAdapter:
        return _apply_set_params(
            self, params,
            resets=(({"model_id", "device"}, lambda: setattr(self, "_model", None)),),
        )

    def fit(self, context, context_exog) -> TimestralAdapter:
        return super().fit(context=context, context_exog=context_exog)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from timestral import TimestralForecaster
        device = _resolve_torch_device(self.device)
        if device == "mps":
            # timestral 1.0.0 computes its scaling statistics in float64,
            # which MPS does not support (mistralai/timestral#6).
            device = "cpu"
        self._model = TimestralForecaster.from_pretrained(self.model_id, device=device)

    def predict(self, steps, context, context_exog, exog, quantiles):
        quantile_list = _validate_supported_quantiles(
            quantiles, self.SUPPORTED_QUANTILES, "Timestral"
        )
        self._load_model()

        names = list(context.keys())
        width = max(len(context[name]) for name in names)
        batch = _left_pad(
            [context[name].to_numpy(dtype=np.float64) for name in names], width
        )

        # Every series of a call shares the same covariate columns
        # (`FoundationModel` groups them), so the first one sets the layout.
        first_ctx_exog = context_exog.get(names[0]) if context_exog is not None else None
        first_exog = exog.get(names[0]) if exog is not None else None
        past_cols, fut_cols = get_exog_signature(first_ctx_exog, first_exog)
        cat_past = [c for c in past_cols if _is_categorical(first_ctx_exog[c])]
        num_past = [c for c in past_cols if c not in cat_past]
        cat_fut = [c for c in fut_cols if _is_categorical(first_exog[c])]
        num_fut = [c for c in fut_cols if c not in cat_fut]

        def rows(name, cols, with_future, categorical):
            if not cols:
                return None
            hist = context_exog[name]
            out = []
            for col in cols:
                values = hist[col]
                if with_future:
                    values = pd.concat([values, exog[name][col]], ignore_index=True)
                if categorical:
                    codes, _ = pd.factorize(values, use_na_sentinel=True)
                    arr = np.where(codes < 0, np.nan, codes).astype(np.float64)
                else:
                    arr = values.astype(np.float64).to_numpy()
                out.append(arr)
            return np.stack(out)

        def batch_rows(cols, with_future, categorical):
            if not cols:
                return None
            per_series = [rows(n, cols, with_future, categorical) for n in names]
            return _left_pad(per_series, width + (steps if with_future else 0))

        forecast = self._model.forecast(
            context                      = batch[:, None, :],
            prediction_length            = steps,
            past_covariates              = batch_rows(num_past, False, False),
            known_covariates             = batch_rows(num_fut, True, False),
            categorical_past_covariates  = batch_rows(cat_past, False, True),
            categorical_known_covariates = batch_rows(cat_fut, True, True),
            **self.predict_kwargs,
        )
        q_arr = _tensor_to_numpy(forecast.quantiles)[:, 0]  # (B, steps, Q)
        levels = list(forecast.quantile_levels)
        requested = quantile_list if quantile_list is not None else [0.5]
        idx = [int(np.argmin([abs(g - q) for g in levels])) for q in requested]
        return {name: q_arr[i][:, idx] for i, name in enumerate(names)}


def register() -> None:
    _adapters._ADAPTER_REGISTRY["mistralai/Timestral"] = TimestralAdapter
    _utils._MODEL_LICENSES["mistralai/Timestral"] = (
        "other", "https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE",
        True, "Timestral Modified MIT License",
    )
