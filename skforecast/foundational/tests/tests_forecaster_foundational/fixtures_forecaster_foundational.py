# Fixtures ForecasterFoundational
# ==============================================================================
import numpy as np
import pandas as pd
from skforecast.foundational._foundational_model import Chronos2Adapter, FoundationalModel


# ---------------------------------------------------------------------------
# Shared time-series fixtures
# ---------------------------------------------------------------------------

# 50-observation monthly series with DatetimeIndex
y = pd.Series(
    data=np.arange(50, dtype=float),
    index=pd.date_range("2020-01-01", periods=50, freq="ME"),
    name="y",
)

# Context window that follows `y` (next 20 obs)
y_lw = pd.Series(
    data=np.arange(50, 70, dtype=float),
    index=pd.date_range("2024-03-31", periods=20, freq="ME"),
    name="y",
)

# 50-observation series with RangeIndex (no frequency)
y_range = pd.Series(
    data=np.arange(50, dtype=float),
    index=pd.RangeIndex(0, 50),
    name="y",
)

# Single-column exogenous DataFrame aligned to `y`
exog = pd.DataFrame(
    {"feat_a": np.arange(50, dtype=float)},
    index=y.index,
)

# Exog for the last-window context
exog_lw = pd.DataFrame(
    {"feat_a": np.arange(50, 70, dtype=float)},
    index=y_lw.index,
)

# Future exog for the forecast horizon (5 steps)
exog_predict = pd.DataFrame(
    {"feat_a": np.arange(70, 75, dtype=float)},
    index=pd.date_range("2024-07-31", periods=5, freq="ME"),
)

# Multi-column exogenous DataFrame aligned to `y`
df_exog = pd.DataFrame(
    {
        "feat_a": np.arange(50, dtype=float),
        "feat_b": np.arange(50, dtype=float) * 2,
    },
    index=y.index,
)

# Future multi-column exog (5 steps)
df_exog_predict = pd.DataFrame(
    {
        "feat_a": np.arange(70, 75, dtype=float),
        "feat_b": np.arange(70, 75, dtype=float) * 2,
    },
    index=pd.date_range("2024-07-31", periods=5, freq="ME"),
)


# ---------------------------------------------------------------------------
# FakePipeline – avoids torch / chronos-forecasting dependency in tests
# ---------------------------------------------------------------------------

class FakePipeline:
    """
    Minimal stand-in for a Chronos-2 pipeline.

    ``predict_quantiles`` returns sample data whose *i*-th quantile column
    equals the quantile level itself (e.g. q=0.5 → 0.5 for every step).
    This makes predictions fully deterministic and easy to assert against.
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
        # shape: (n_samples=1, prediction_length, n_quantiles)
        arr = np.broadcast_to(q_values, (1, prediction_length, n_q)).copy()
        # Return one result per input series.
        mean_arr = np.zeros((1, prediction_length))
        return [arr] * len(inputs), [mean_arr] * len(inputs)


def make_forecaster(context_length: int | None = None, **kwargs) -> "ForecasterFoundational":
    """
    Return a ``ForecasterFoundational`` backed by a ``FakePipeline``.

    The fake pipeline is injected so no real Chronos model is loaded.
    """
    from skforecast.foundational import ForecasterFoundational

    estimator = FoundationalModel(
        "autogluon/chronos-2-small",
        context_length=context_length,
        pipeline=FakePipeline(),
        **kwargs,
    )
    return ForecasterFoundational(estimator=estimator)
