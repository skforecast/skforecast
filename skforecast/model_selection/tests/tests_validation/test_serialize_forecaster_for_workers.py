# Unit test _serialize_forecaster_for_workers
# ==============================================================================
import pickle
import sys
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection._validation import _serialize_forecaster_for_workers

# Fixtures
from ..fixtures_model_selection import y


def test_serialize_forecaster_for_workers_returns_forecaster_when_n_jobs_is_1():
    """
    Test that, with `n_jobs=1`, the forecaster itself is returned, so the
    folds of the backtesting share it as before.
    """
    forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)

    result = _serialize_forecaster_for_workers(forecaster=forecaster, n_jobs=1)

    assert result is forecaster


@pytest.mark.parametrize(
    "pickler",
    ["loky", "cloudpickle", "pickle"],
    ids=lambda pickler: f"pickler: {pickler}",
)
def test_serialize_forecaster_for_workers_bytes_load_same_forecaster(pickler):
    """
    Test that, with `n_jobs` greater than 1, the forecaster is returned as
    bytes that `pickle.loads` turns into a forecaster with the same
    predictions: with the pickler of loky, and, if it cannot be imported, with
    `cloudpickle`, or `pickle` if `cloudpickle` is not installed either. Except
    with `pickle`, the forecaster has a lambda as `weight_func`, which only
    `cloudpickle` can serialize.
    """
    weight_func = (
        None if pickler == "pickle" else lambda index: np.where(index < 20, 0.5, 1.0)
    )
    forecaster = ForecasterRecursive(
        estimator=Ridge(random_state=123), lags=3, weight_func=weight_func
    )
    forecaster.fit(y=y)

    patches = []
    if pickler in ("cloudpickle", "pickle"):
        patches.append(patch("skforecast.model_selection._validation.loky_dumps", None))
    if pickler == "pickle":
        patches.append(patch.dict(sys.modules, {"cloudpickle": None}))
    for p in patches:
        p.start()
    try:
        result = _serialize_forecaster_for_workers(forecaster=forecaster, n_jobs=2)
    finally:
        for p in reversed(patches):
            p.stop()

    assert isinstance(result, bytes)
    pd.testing.assert_series_equal(
        pickle.loads(result).predict(steps=5), forecaster.predict(steps=5)
    )
