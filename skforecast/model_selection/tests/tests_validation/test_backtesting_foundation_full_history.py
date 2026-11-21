# Tests of full-history foundation backtesting.
import sys

import numpy as np
import pandas as pd
import pytest

from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.foundation.tests.tests_foundation_models.fixtures_adapters import (
    FakeTorch,
)
from skforecast.foundation.tests.tests_foundation_models.test_CIprotoAdapter import (
    FakeCIprotoModel,
    MODEL_ID,
)
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation


@pytest.mark.parametrize("context_length", [None, 64])
def test_backtesting_keeps_full_history_unless_a_limit_is_requested(
    context_length, monkeypatch
):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    initial_size = 16400
    series = pd.Series(np.arange(initial_size + 4, dtype=float), name="sales")
    exog = pd.DataFrame({"past": series + 1, "known": series + 2})
    backend = FakeCIprotoModel()
    forecaster = ForecasterFoundation(
        estimator=FoundationModel(MODEL_ID, context_length=context_length)
    )
    forecaster.estimator.adapter._model = backend
    cv = TimeSeriesFold(
        steps=2,
        initial_train_size=initial_size,
        refit=False,
        fixed_train_size=True,
        verbose=False,
    )
    metrics, predictions = backtesting_foundation(
        forecaster=forecaster,
        series=series,
        exog=exog,
        cv=cv,
        metric="mean_absolute_error",
        show_progress=False,
    )
    assert len(backend.calls) == 2
    for call, train_size in zip(backend.calls, [initial_size, initial_size + 2]):
        retained = train_size if context_length is None else context_length
        assert call["history"].shape == (1, retained)
        assert call["past_covariates"] is None
        assert call["future_covariates"].shape == (2, retained + 2)
        assert call["history"][0, 0] == train_size - retained
        assert call["future_covariates"][0, 0] == train_size - retained + 2
    pd.testing.assert_index_equal(
        predictions.index, pd.RangeIndex(initial_size, initial_size + 4)
    )
    np.testing.assert_allclose(metrics["mean_absolute_error"], 0.5)
