# Unit test Chronos2Adapter predict
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundational._foundational_models import Chronos2Adapter


# Fixtures
# ==============================================================================
y = pd.Series(
    data=np.arange(50, dtype=float),
    index=pd.date_range("2020-01-01", periods=50, freq="ME"),
    name="sales",
)

exog = pd.DataFrame(
    {"feat_a": np.arange(50, dtype=float), "feat_b": np.arange(50, dtype=float) * 2},
    index=y.index,
)

air_passengers = np.array([
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
], dtype=np.float64)
data = pd.Series(
    air_passengers,
    index=pd.date_range(start="1949-01", periods=len(air_passengers), freq="MS"),
    name="y",
)


class FakePipeline:
    """
    Fake Chronos-2 pipeline for testing without torch or chronos dependency.

    Returns quantile values equal to the quantile level itself (e.g. q=0.1 → 0.1)
    for all steps, making it easy to verify correctness in assertions.
    Records the last call arguments for inspection.
    """

    def __init__(self):
        self.last_inputs = None
        self.last_prediction_length = None
        self.last_quantile_levels = None

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        self.last_inputs = inputs
        self.last_prediction_length = prediction_length
        self.last_quantile_levels = quantile_levels

        n_q = len(quantile_levels)
        q_values = np.array(quantile_levels, dtype=float)
        # shape: (1, prediction_length, n_q) — univariate, n_vars=1
        arr = np.broadcast_to(q_values, (1, prediction_length, n_q)).copy()
        mean = np.zeros((1, prediction_length))
        return [arr], [mean]


# Tests Chronos2Adapter.predict — error handling
# ==============================================================================
def test_Chronos2Adapter_predict_raises_ValueError_when_not_fitted_and_no_last_window():
    """
    Test predict raises ValueError when adapter is not fitted and no last_window is passed.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small")
    err_msg = re.escape("Call `fit` before `predict`, or pass `last_window`.")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(steps=5)


@pytest.mark.parametrize(
    "steps",
    [0, -1, -10],
    ids=lambda x: f"steps: {x}",
)
def test_Chronos2Adapter_predict_raises_ValueError_when_steps_not_positive(steps):
    """
    Test predict raises ValueError when steps is 0 or negative.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    err_msg = re.escape("`steps` must be a positive integer.")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(steps=steps)


@pytest.mark.parametrize(
    "bad_quantile",
    [-0.1, 1.1, 2.0],
    ids=lambda x: f"quantile: {x}",
)
def test_Chronos2Adapter_predict_raises_ValueError_when_quantile_out_of_range(bad_quantile):
    """
    Test predict raises ValueError when a quantile level is outside [0, 1].
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    err_msg = re.escape(f"All quantiles must be between 0 and 1. Got {bad_quantile}.")
    with pytest.raises(ValueError, match=err_msg):
        adapter.predict(steps=3, quantiles=[0.5, bad_quantile])


# Tests Chronos2Adapter.predict — point forecast (quantiles=None)
# ==============================================================================
def test_Chronos2Adapter_predict_returns_series_as_point_forecast():
    """
    Test predict returns a pandas Series when quantiles is None.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    result = adapter.predict(steps=5)
    assert isinstance(result, pd.Series)


def test_Chronos2Adapter_predict_series_correct_length():
    """
    Test the returned Series has exactly `steps` observations.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    result = adapter.predict(steps=12)
    assert len(result) == 12


def test_Chronos2Adapter_predict_series_preserves_name():
    """
    Test the returned Series has the same name as the training series.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    result = adapter.predict(steps=3)
    assert result.name == "sales"


def test_Chronos2Adapter_predict_series_correct_index():
    """
    Test the returned Series index starts right after the last training date
    and covers exactly `steps` periods with the same frequency.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    result = adapter.predict(steps=3)
    expected_start = y.index[-1] + y.index.freq
    expected_index = pd.date_range(start=expected_start, periods=3, freq=y.index.freq)
    pd.testing.assert_index_equal(result.index, expected_index)


def test_Chronos2Adapter_predict_point_forecast_uses_median_quantile():
    """
    Test that the point forecast equals the 0.5 quantile value from the pipeline.
    FakePipeline returns q_level for each quantile, so the median should be 0.5.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    result = adapter.predict(steps=4)
    np.testing.assert_array_almost_equal(result.to_numpy(), np.full(4, 0.5))


# Tests Chronos2Adapter.predict — quantile forecast
# ==============================================================================
def test_Chronos2Adapter_predict_returns_dataframe_when_quantiles_provided():
    """
    Test predict returns a pandas DataFrame when quantiles is specified.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    result = adapter.predict(steps=5, quantiles=quantiles)
    assert isinstance(result, pd.DataFrame)


def test_Chronos2Adapter_predict_dataframe_has_correct_columns():
    """
    Test that quantile DataFrame columns match q_<level> naming convention.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    result = adapter.predict(steps=5, quantiles=quantiles)
    assert list(result.columns) == ["q_0.1", "q_0.5", "q_0.9"]


def test_Chronos2Adapter_predict_dataframe_correct_length():
    """
    Test that the quantile DataFrame has exactly `steps` rows.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    result = adapter.predict(steps=7, quantiles=quantiles)
    assert len(result) == 7


def test_Chronos2Adapter_predict_dataframe_values_match_quantile_levels():
    """
    Test quantile DataFrame values equal the quantile levels (FakePipeline property).
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)
    result = adapter.predict(steps=3, quantiles=quantiles)
    for q in quantiles:
        np.testing.assert_array_almost_equal(
            result[f"q_{q}"].to_numpy(), np.full(3, q)
        )


# Tests Chronos2Adapter.predict — last_window behaviour
# ==============================================================================
def test_Chronos2Adapter_predict_uses_last_window_when_provided():
    """
    Test that last_window is used as context instead of stored _history.
    The forecast index must follow from last_window's index, not _history's.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=y)

    last_window = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2025-01-01", periods=10, freq="ME"),
        name="sales",
    )
    result = adapter.predict(steps=3, last_window=last_window)
    expected_start = last_window.index[-1] + last_window.index.freq
    expected_index = pd.date_range(start=expected_start, periods=3, freq=last_window.index.freq)
    pd.testing.assert_index_equal(result.index, expected_index)


def test_Chronos2Adapter_predict_trims_last_window_to_context_length():
    """
    Test that a last_window longer than context_length is trimmed before inference.
    The pipeline must receive only context_length observations.
    """
    context_length = 10
    pipeline = FakePipeline()
    adapter = Chronos2Adapter(
        model_id="autogluon/chronos-2-small",
        pipeline=pipeline,
        context_length=context_length,
    )
    adapter.fit(y=y)

    long_window = pd.Series(
        np.arange(40, dtype=float),
        index=pd.date_range("2023-01-01", periods=40, freq="ME"),
        name="sales",
    )
    adapter.predict(steps=5, last_window=long_window)
    assert len(pipeline.last_inputs[0]["target"]) == context_length


def test_Chronos2Adapter_predict_passes_past_covariates_from_history():
    """
    Test that historical exog stored at fit time is passed as past_covariates.
    """
    pipeline = FakePipeline()
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=pipeline)
    adapter.fit(y=y, exog=exog)
    adapter.predict(steps=3)
    assert "past_covariates" in pipeline.last_inputs[0]
    assert set(pipeline.last_inputs[0]["past_covariates"].keys()) == {"feat_a", "feat_b"}


def test_Chronos2Adapter_predict_passes_past_covariates_from_last_window_exog():
    """
    Test that last_window_exog is passed as past_covariates when last_window is provided.
    """
    pipeline = FakePipeline()
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=pipeline)
    adapter.fit(y=y)

    lw = y.iloc[-10:]
    lw_exog = exog.iloc[-10:]
    adapter.predict(steps=3, last_window=lw, last_window_exog=lw_exog)
    assert "past_covariates" in pipeline.last_inputs[0]


def test_Chronos2Adapter_predict_passes_future_covariates():
    """
    Test that future exog is passed as future_covariates in the pipeline input.
    """
    pipeline = FakePipeline()
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=pipeline)
    adapter.fit(y=y)

    future = pd.DataFrame(
        {"feat_a": np.arange(6, dtype=float)},
        index=pd.date_range("2024-03-01", periods=6, freq="ME"),
    )
    adapter.predict(steps=6, exog=future)
    assert "future_covariates" in pipeline.last_inputs[0]
    assert len(pipeline.last_inputs[0]["future_covariates"]["feat_a"]) == 6


def test_Chronos2Adapter_predict_correct_steps_sent_to_pipeline():
    """
    Test that the prediction_length passed to the pipeline equals steps.
    """
    pipeline = FakePipeline()
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=pipeline)
    adapter.fit(y=y)
    adapter.predict(steps=9)
    assert pipeline.last_prediction_length == 9


def test_Chronos2Adapter_predict_correct_quantile_levels_sent_to_pipeline():
    """
    Test that the quantile_levels passed to the pipeline match the requested quantiles.
    When quantiles=None, default levels [0.1, 0.5, 0.9] are sent.
    """
    pipeline = FakePipeline()
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=pipeline)
    adapter.fit(y=y)

    adapter.predict(steps=3)
    assert pipeline.last_quantile_levels == [0.1, 0.5, 0.9]

    custom_quantiles = [0.05, 0.25, 0.75, 0.95]
    adapter.predict(steps=3, quantiles=custom_quantiles)
    assert pipeline.last_quantile_levels == custom_quantiles


# Tests Chronos2Adapter.predict — exact predicted values
# ==============================================================================
def test_Chronos2Adapter_predict_point_forecast_exact_values():
    """
    Test that the point forecast contains exactly 0.5 for all steps when
    fitted on the air-passengers series. FakePipeline always returns the
    quantile level as a constant, so the median (0.5) is expected.
    The forecast index must immediately follow the last training date.
    """
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=data)
    result = adapter.predict(steps=12)
    expected_index = pd.date_range("1961-01-01", periods=12, freq="MS")
    expected_values = np.full(12, 0.5)
    pd.testing.assert_index_equal(result.index, expected_index)
    np.testing.assert_array_almost_equal(result.to_numpy(), expected_values)


def test_Chronos2Adapter_predict_quantile_forecast_exact_values():
    """
    Test that each quantile column contains the expected constant values when
    fitted on the air-passengers series. FakePipeline returns `q_level` for
    every step, so q_0.1 must be all 0.1, q_0.5 all 0.5, q_0.9 all 0.9.
    The forecast index must immediately follow the last training date.
    """
    quantiles = [0.1, 0.5, 0.9]
    adapter = Chronos2Adapter(model_id="autogluon/chronos-2-small", pipeline=FakePipeline())
    adapter.fit(y=data)
    result = adapter.predict(steps=12, quantiles=quantiles)
    expected_index = pd.date_range("1961-01-01", periods=12, freq="MS")
    pd.testing.assert_index_equal(result.index, expected_index)
    for q in quantiles:
        np.testing.assert_array_almost_equal(
            result[f"q_{q}"].to_numpy(), np.full(12, q)
        )
