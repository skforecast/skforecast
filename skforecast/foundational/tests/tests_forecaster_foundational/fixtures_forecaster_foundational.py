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

# Wide-format DataFrame with two series (DatetimeIndex)
series_wide = pd.DataFrame(
    {
        "series_1": np.arange(50, dtype=float),
        "series_2": np.arange(50, 100, dtype=float),
    },
    index=pd.date_range("2020-01-01", periods=50, freq="ME"),
)

# Long-format DataFrame (MultiIndex: series ID × DatetimeIndex)
# Manually constructed to guarantee the DatetimeIndex level carries "ME" freq.
_date_idx = pd.date_range("2020-01-31", periods=50, freq="ME")
_s1 = pd.Series(np.arange(50, dtype=float), index=_date_idx, name="series_1")
_s2 = pd.Series(np.arange(50, 100, dtype=float), index=_date_idx, name="series_2")
series_long = pd.concat(
    [_s1.rename("value").to_frame().assign(dummy=0),
     _s2.rename("value").to_frame().assign(dummy=0)],
    keys=["series_1", "series_2"],
)[["value"]]

# Long-format exog (MultiIndex: series ID × DatetimeIndex) aligned to series_long
_exog_s1 = pd.DataFrame({"feat_a": np.arange(50, dtype=float)}, index=_date_idx)
_exog_s2 = pd.DataFrame({"feat_a": np.arange(50, dtype=float) * 2}, index=_date_idx)
exog_long = pd.concat([_exog_s1, _exog_s2], keys=["series_1", "series_2"])

# Wide-format DataFrame with RangeIndex (two series)
series_wide_range = pd.DataFrame(
    {
        "series_1": np.arange(50, dtype=float),
        "series_2": np.arange(50, 100, dtype=float),
    },
    index=pd.RangeIndex(0, 50),
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


def make_forecaster(context_length: int = 2048, **kwargs) -> "ForecasterFoundational":
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
