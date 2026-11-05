"""
Draft of the fake backend for `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`
(section 8 of the report). It mimics `h2ofm.TimeH2O` 0.43.1: the signatures
of `from_pretrained` and `forecast_many` without `**kwargs`, the shape checks
of the covariates and the validation of `quantile_levels`, so an unknown
`predict_kwargs` key raises `TypeError` as with the real backend.

To move it into `fixtures_adapters.py`, copy the two classes and add
`import types` to the imports of that module (it has no such import).
"""
import types
import numpy as np


class _FakeTimeH2OForecast:
    """
    Fake `h2ofm.timeh2o.Forecast` of one 1-D series: `quantiles` has shape
    `(horizon, len(quantile_levels))` and column `i` equals
    `quantile_levels[i]` at every step.
    """

    def __init__(self, horizon, quantile_levels):
        levels = np.asarray(quantile_levels, dtype=np.float32)
        self.quantiles = np.broadcast_to(levels, (horizon, len(levels))).copy()
        self.quantile_levels = tuple(quantile_levels)
        self.point = np.full(horizon, 0.5, dtype=np.float32)
        self.mean = self.point.copy()


class FakeTimeH2O:
    """
    Fake `h2ofm.TimeH2O` for testing without torch or h2ofm.

    `forecast_many()` returns one `_FakeTimeH2OForecast` per series and
    records the arguments of every call in `calls` (and the last ones in the
    `last_*` attributes). `quantile_levels` must be distinct and strictly
    between 0 and 1, and each `future_covariates` entry must be `None` or a
    sequence of rows of length `history + horizon`, as in the real backend.
    """

    def __init__(self, batch_size=256, device="cpu", variate_attention=True):
        self.batch_size = batch_size
        self.device = device
        # `config.variate_attention` is False for the tiny checkpoint, which
        # ignores covariates.
        self.config = types.SimpleNamespace(variate_attention=variate_attention)
        self.calls = []
        self.last_contexts = None
        self.last_horizon = None
        self.last_future_covariates = None
        self.last_past_covariates = None
        self.last_quantile_levels = None
        self.last_kwargs = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        device="auto",
        revision=None,
        token=None,
        cache_dir=None,
        local_files_only=False,
        batch_size=256,
    ):
        return cls(batch_size=batch_size, device=device)

    def forecast_many(
        self,
        contexts,
        horizon,
        *,
        past_covariates=None,
        future_covariates=None,
        quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        flip_equivariant=True,
        clip_nonnegative=False,
        snap_integer=False,
        context_length=None,
        autoregressive=None,
        patch_ensemble=True,
    ):
        levels = [float(q) for q in quantile_levels]
        if not levels or not all(0 < q < 1 for q in levels):
            raise ValueError(
                "quantile_levels must be non-empty and strictly between 0 and 1, "
                f"got {list(quantile_levels)}"
            )
        if len(set(levels)) != len(levels):
            raise ValueError("quantile_levels must not contain duplicates")
        contexts = [np.asarray(c) for c in contexts]
        for c in contexts:
            if c.ndim != 1 or c.dtype.kind not in "biuf":
                raise ValueError("A series must be a 1-D numeric array")
            if np.isinf(c).any():
                raise ValueError(
                    "Series may contain NaN for missing observations, but not infinity"
                )
        if future_covariates is not None:
            if len(future_covariates) != len(contexts):
                raise ValueError(
                    f"future_covariates has {len(future_covariates)} entries "
                    f"for {len(contexts)} items"
                )
            for i, (rows, c) in enumerate(zip(future_covariates, contexts)):
                if rows is None:
                    continue
                if any(len(row) != len(c) + horizon for row in rows):
                    raise ValueError(
                        f"future_covariates[{i}] must have shape "
                        f"(covariates, history + horizon) with history + horizon "
                        f"= {len(c) + horizon}"
                    )
        kwargs = {
            "flip_equivariant": flip_equivariant,
            "clip_nonnegative": clip_nonnegative,
            "snap_integer": snap_integer,
            "context_length": context_length,
            "autoregressive": autoregressive,
            "patch_ensemble": patch_ensemble,
        }
        self.last_contexts = contexts
        self.last_horizon = horizon
        self.last_future_covariates = future_covariates
        self.last_past_covariates = past_covariates
        self.last_quantile_levels = levels
        self.last_kwargs = kwargs
        self.calls.append({
            "contexts": contexts,
            "horizon": horizon,
            "future_covariates": future_covariates,
            "past_covariates": past_covariates,
            "quantile_levels": levels,
            "kwargs": kwargs,
        })

        return [_FakeTimeH2OForecast(horizon, levels) for _ in contexts]
