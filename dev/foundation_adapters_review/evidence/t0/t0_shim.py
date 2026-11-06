"""
Diagnostic shim, not a fix proposal: lets the current `T0Adapter` (which calls
`predict(context=..., quantiles=...)`, the tfc-t0<0.4 API) run on tfc-t0>=0.5
by translating the call to `predict(model_input, horizon, quantile_levels, ...)`.

Used only to measure what the new backend returns through the unchanged
skforecast pipeline (padding, covariate assembly, quantile reordering).
"""

from __future__ import annotations

from importlib.metadata import version
from typing import Any


def backend_version() -> tuple[int, ...]:
    return tuple(int(p) for p in version("tfc-t0").split(".")[:3])


class NewApiProxy:
    """
    Wrap a tfc-t0>=0.5 `T0Forecaster` so that it accepts the call of the
    current adapter. `predict_extra` is forwarded to the backend (for example
    `{"trim_leading_missing": True}`).
    """

    def __init__(self, model: Any, **predict_extra: Any) -> None:
        self.model = model
        self.predict_extra = predict_extra

    def predict(self, context, horizon, quantiles, future_covariates=None):
        return self.model.predict(
            context,
            horizon,
            quantile_levels=quantiles,
            future_covariates=future_covariates,
            **self.predict_extra,
        )


def load_backend_model(model_id: str, dtype: Any = None) -> Any:
    """
    Load `T0Forecaster` on CPU, in eval mode, as the adapter does. `dtype` is
    passed to `from_pretrained` (the route documented upstream for bf16/fp16
    autocast), not applied with `.to(dtype)`.
    """
    from t0 import T0Forecaster

    kwargs = {} if dtype is None else {"dtype": dtype}
    return T0Forecaster.from_pretrained(model_id, **kwargs).to("cpu").eval()


def make_adapter_model(model_id: str, dtype: Any = None, **predict_extra: Any) -> Any:
    """
    Object to pass as `model=` to `FoundationModel`: the backend model itself
    on tfc-t0<0.4, the proxy on tfc-t0>=0.5.
    """
    model = load_backend_model(model_id, dtype=dtype)
    if backend_version() < (0, 4, 0):
        if predict_extra:
            raise ValueError("predict_extra needs tfc-t0>=0.6")
        return model
    return NewApiProxy(model, **predict_extra)
