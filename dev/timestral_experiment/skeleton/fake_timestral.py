# Draft of FakeTimestralForecaster for fixtures_adapters.py (see section 8 of the report).
import numpy as np

TIMESTRAL_QUANTILES = (
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
)


class _FakeTimestralForecast:
    """
    Fake `timestral.Forecast` returned by `FakeTimestralForecaster.forecast`.

    `quantiles` has shape `(B, V, prediction_length, Q)` and every entry on
    the last axis equals its quantile level, making column-order assertions
    trivial. `quantile_levels` is a tuple, as in the real dataclass.
    """

    def __init__(self, quantiles, quantile_levels):
        self.quantiles = quantiles
        self.quantile_levels = tuple(quantile_levels)


class _TorchLike:
    """
    Minimal torch-tensor stand-in (`detach().cpu().numpy()`), used when
    `as_tensor=True` to cover `_tensor_to_numpy`.
    """

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakeTimestralForecaster:
    """
    Fake `timestral.TimestralForecaster` for testing without torch/timestral.

    `forecast()` validates the shapes the real backend asserts (`context`
    3-D `(B, V, T)`; past covariates `(B, C, T)`; known covariates
    `(B, C, T + prediction_length)`), records every call in `calls` (and the
    last one in `last_*`), and returns a `_FakeTimestralForecast`.
    `quantile_levels` defaults to the 21 Timestral levels but can be
    overridden to test the quantile-index mapping.
    """

    def __init__(self, quantile_levels=None, as_tensor=False):
        self.quantile_levels = tuple(
            quantile_levels if quantile_levels is not None else TIMESTRAL_QUANTILES
        )
        self.as_tensor = as_tensor
        self.last_context = None
        self.last_prediction_length = None
        self.last_past_covariates = None
        self.last_known_covariates = None
        self.last_categorical_past_covariates = None
        self.last_categorical_known_covariates = None
        self.last_kwargs = None
        self.calls = []

    @classmethod
    def from_pretrained(cls, path_or_repo_id, device="cpu", **kwargs):
        return cls()

    def forecast(
        self,
        context,
        prediction_length,
        observation_mask=None,
        past_covariates=None,
        known_covariates=None,
        categorical_past_covariates=None,
        categorical_known_covariates=None,
        symmetric_averaging=True,
        clip_nonnegative=True,
    ):
        # Same signature as the real `forecast` (no **kwargs), so an unknown
        # `predict_kwargs` key raises TypeError as with the backend.
        kwargs = {
            "symmetric_averaging": symmetric_averaging,
            "clip_nonnegative": clip_nonnegative,
        }
        context = np.asarray(context, dtype=float)
        if context.ndim != 3:
            raise AssertionError(f"context must be (B, V, T), got {context.shape}")
        b, v, t = context.shape
        covariates = {
            "past_covariates": (past_covariates, t),
            "known_covariates": (known_covariates, t + prediction_length),
            "categorical_past_covariates": (categorical_past_covariates, t),
            "categorical_known_covariates": (
                categorical_known_covariates, t + prediction_length
            ),
        }
        recorded = {}
        for name, (values, length) in covariates.items():
            if values is not None:
                values = np.asarray(values, dtype=float)
                if values.ndim != 3 or values.shape[0] != b or values.shape[-1] != length:
                    raise AssertionError(
                        f"{name} must be (B={b}, C, {length}), got {values.shape}"
                    )
            recorded[name] = values

        self.last_context = context
        self.last_prediction_length = prediction_length
        self.last_past_covariates = recorded["past_covariates"]
        self.last_known_covariates = recorded["known_covariates"]
        self.last_categorical_past_covariates = recorded["categorical_past_covariates"]
        self.last_categorical_known_covariates = recorded["categorical_known_covariates"]
        self.last_kwargs = kwargs
        self.calls.append({
            "context": context,
            "prediction_length": prediction_length,
            "observation_mask": observation_mask,
            **recorded,
            "kwargs": kwargs,
        })

        q_values = np.array(self.quantile_levels, dtype=float)
        quantiles = np.broadcast_to(
            q_values, (b, v, prediction_length, len(q_values))
        ).copy()
        if self.as_tensor:
            quantiles = _TorchLike(quantiles)

        return _FakeTimestralForecast(quantiles, self.quantile_levels)
