# Unit test set_fit_kwargs ForecasterRnn
# ==============================================================================
import os
import numpy as np
import pandas as pd
os.environ["KERAS_BACKEND"] = "torch"
import keras
from skforecast.deep_learning import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn

# Fixtures
np.random.seed(123)
series = pd.DataFrame(
    {
        "1": np.random.rand(50),
        "2": np.random.rand(50),
        "3": np.random.rand(50),
    },
    index=pd.date_range(start='2000-01-01', periods=50, freq='MS')
)

model = create_and_compile_model(
    series=series,
    lags=3,
    steps=5,
    levels=["1", "2", "3"],
    recurrent_units=64,
    dense_units=32,
)


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
    """
    forecaster = ForecasterRnn(
        model, levels=["1", "2", "3"], lags=3, 
        fit_kwargs={"epochs": 10, "batch_size": 16}
    )
    
    new_fit_kwargs = {'epochs': 25}
    forecaster.set_fit_kwargs(new_fit_kwargs)
    results = forecaster.fit_kwargs

    expected = {'epochs': 25}

    assert results == expected
