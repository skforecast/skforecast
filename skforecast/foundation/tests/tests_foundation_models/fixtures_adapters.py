# Fixtures for adapter tests (Chronos2, TimesFM25, Moirai)
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.foundation._utils import (
    check_preprocess_series_foundation,
)
from skforecast.utils import expand_index


def normalize_exog_to_dict(exog, series_names):
    """
    Simple exog normalizer for test fixtures.
    """
    if exog is None:
        return {name: None for name in series_names}
    if isinstance(exog, dict):
        return {name: exog.get(name, None) for name in series_names}
    return {name: exog for name in series_names}


# Shared time-series fixtures
# ==============================================================================
_idx_single = pd.date_range("2020-01-01", periods=50, freq="ME")
y = pd.Series(
    data=np.arange(50, dtype=float),
    index=_idx_single,
    name="sales",
)

exog = pd.DataFrame(
    {"feat_a": np.arange(50, dtype=float), "feat_b": np.arange(50, dtype=float) * 2},
    index=_idx_single,
)

_idx_ms = pd.date_range("2020-01-01", periods=30, freq="ME")
_y_s1 = pd.Series(np.arange(30, dtype=float), index=_idx_ms, name="s1")
_y_s2 = pd.Series(np.arange(30, 60, dtype=float), index=_idx_ms, name="s2")
y_wide = pd.DataFrame({"s1": _y_s1, "s2": _y_s2})
y_dict = {"s1": _y_s1.copy(), "s2": _y_s2.copy()}
exog_shared = pd.DataFrame({"feat": np.arange(30, dtype=float)}, index=_idx_ms)

# Single-column exog for predict tests (aligned to y index)
exog_single = pd.DataFrame(
    {"feat_a": np.arange(50, dtype=float)},
    index=_idx_single,
)

# Air-passengers series (longer series for value-based tests)
data = pd.Series(
    np.array([
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
    ], dtype=np.float64),
    index=pd.date_range(start="1949-01", periods=144, freq="MS"),
    name="y",
)


# Fake Chronos-2 pipeline
# ==============================================================================
class FakePipeline:
    """
    Fake Chronos-2 pipeline for testing without torch/chronos.

    Returns quantile values equal to the quantile level itself for all steps.
    Records last call arguments for inspection.
    """

    def __init__(self):
        self.last_inputs = None
        self.last_prediction_length = None
        self.last_quantile_levels = None
        self.last_kwargs = None

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        self.last_inputs = inputs
        self.last_prediction_length = prediction_length
        self.last_quantile_levels = quantile_levels
        self.last_kwargs = kwargs

        n_q = len(quantile_levels)
        q_values = np.array(quantile_levels, dtype=float)
        arr = np.broadcast_to(q_values, (1, prediction_length, n_q)).copy()
        mean = np.zeros((1, prediction_length))
        return [arr] * len(inputs), [mean] * len(inputs)


# Fake TimesFM 2.5 model
# ==============================================================================
class FakeTimesFM25Model:
    """
    Fake TimesFM 2.5 model for testing without torch/timesfm.

    `forecast()` returns zeros as point forecast and `i/10` at index `i`
    for quantile forecast. Pre-sets `forecast_config` with
    `max_horizon=16384` to prevent `_ensure_compiled` from importing
    timesfm.
    """

    class _FakeForecastConfig:
        max_horizon: int = 16384

    def __init__(self):
        self.last_horizon = None
        self.last_inputs = None
        self.forecast_config = self._FakeForecastConfig()

    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def compile(self, forecast_config):
        self.forecast_config = forecast_config

    def forecast(self, horizon, inputs):
        self.last_horizon = horizon
        self.last_inputs = inputs
        n = len(inputs)
        point_forecast = np.zeros((n, horizon))
        quantile_vals = np.array([i / 10.0 for i in range(10)])
        quantile_forecast = np.broadcast_to(
            quantile_vals[np.newaxis, np.newaxis, :],
            (n, horizon, 10),
        ).copy()
        return point_forecast, quantile_forecast


# Fake Moirai-2 forecast
# ==============================================================================
class FakeMoirai2Forecast:
    """
    Fake Moirai2Forecast for testing without uni2ts/torch.

    `predict()` returns shape `(n, 9, steps)` where
    `raw[i, q_idx, :]` equals `(q_idx + 1) / 10`.
    """

    def __init__(self):
        self.last_inputs = None
        self._last_steps = None

    class _HparamsCtx:
        def __init__(self, forecast_obj, prediction_length):
            self._obj = forecast_obj
            self._pl = prediction_length

        def __enter__(self):
            self._obj._last_steps = self._pl
            return self._obj

        def __exit__(self, *args):
            pass

    def hparams_context(self, prediction_length):
        return self._HparamsCtx(self, prediction_length)

    def predict(self, past_target):
        self.last_inputs = past_target
        n = len(past_target)
        steps = self._last_steps
        raw = np.zeros((n, 9, steps), dtype=float)
        for q_idx in range(9):
            raw[:, q_idx, :] = (q_idx + 1) / 10.0
        return raw


# Helper: prepare dicts for adapter.fit()
# ==============================================================================
def prepare_fit_args(series, exog=None, context_length=None):
    """
    Convert user-facing series/exog into the dict-based API that adapters
    expect. Returns (context, context_exog).

    When `context_length` is provided the series and exog are trimmed to
    that many trailing observations, mimicking the trimming that
    `FoundationModel._check_preprocess_context` performs upstream.
    """
    context, _ = check_preprocess_series_foundation(series)
    series_names = list(context.keys())
    if exog is not None:
        context_exog = normalize_exog_to_dict(exog, series_names)
    else:
        context_exog = None

    if context_length is not None:
        context = {
            name: s.iloc[-context_length:]
            for name, s in context.items()
        }
        if context_exog is not None:
            context_exog = {
                name: (
                    e.iloc[-context_length:] if e is not None else None
                )
                for name, e in context_exog.items()
            }

    return context, context_exog


def prepare_predict_args(adapter, steps, context=None, context_exog=None,
                         exog=None):
    """
    Convert user-facing predict args into the dict-based API that adapters
    expect. Returns (context, context_exog, exog).
    """
    if context is not None:
        lw_dict, _ = check_preprocess_series_foundation(context)
        series_names = list(lw_dict.keys())
    else:
        lw_dict = None
        series_names = list(adapter.context_.keys())

    if lw_dict is not None:
        ctx = {
            name: s.iloc[-adapter.context_length :]
            for name, s in lw_dict.items()
        }
    else:
        ctx = adapter.context_

    if lw_dict is not None and context_exog is not None:
        ctx_exog = normalize_exog_to_dict(context_exog, series_names)
        ctx_exog = {
            name: (
                e.iloc[-adapter.context_length :] if e is not None else None
            )
            for name, e in ctx_exog.items()
        }
    else:
        ctx_exog = adapter.context_exog_

    if exog is not None:
        future_exog = normalize_exog_to_dict(exog, series_names)
    else:
        future_exog = None

    return ctx, ctx_exog, future_exog


# Fake TabICL forecaster
# ==============================================================================
class FakeTabICLForecaster:
    """
    Fake TabICLForecaster for testing without tabicl.

    `predict_df()` returns a DataFrame with a (item_id, timestamp)
    MultiIndex where:

    - ``"target"`` column = 0.0 for every step.
    - Each quantile column ``q`` = ``q`` (quantile value itself) for every
      step, making assertions straightforward.

    Records the last call arguments for inspection.
    """

    def __init__(
        self,
        max_context_length=4096,
        temporal_features=None,
        point_estimate="mean",
        tabicl_config=None,
    ):
        self.max_context_length = max_context_length
        self.temporal_features = temporal_features
        self.point_estimate = point_estimate
        self.tabicl_config = tabicl_config or {}
        self.last_context_df = None
        self.last_future_df = None
        self.last_quantiles = None

    def predict_df(self, context_df, future_df, quantiles=None):
        self.last_context_df = context_df.copy()
        self.last_future_df = future_df.copy()
        self.last_quantiles = list(quantiles) if quantiles is not None else None

        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Build a MultiIndex (item_id, timestamp) result DataFrame.
        idx_tuples = []
        rows = []
        for item_id in future_df["item_id"].unique():
            item_rows = future_df[future_df["item_id"] == item_id]
            for _, row in item_rows.iterrows():
                idx_tuples.append((item_id, row["timestamp"]))
                r = {"target": 0.0}
                for q in quantiles:
                    r[q] = float(q)
                rows.append(r)

        index = pd.MultiIndex.from_tuples(idx_tuples, names=["item_id", "timestamp"])
        data = {"target": [r["target"] for r in rows]}
        for q in quantiles:
            data[q] = [r[q] for r in rows]

        return pd.DataFrame(data, index=index)
