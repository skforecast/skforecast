# Unit test set_fit_kwargs ForecasterRnn
# ==============================================================================
import os
import re
import numpy as np
import pandas as pd
import pytest
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
exog = pd.DataFrame(
    {
        "exog_1": np.random.rand(50),
        "exog_2": np.random.rand(50),
        "exog_3": np.random.rand(50),
    },
    index=pd.date_range(start='2000-01-01', periods=50, freq='MS')
)

model = create_and_compile_model(
    series=series,
    exog=exog,
    lags=3,
    steps=5,
    levels=["1", "2", "3"],
    recurrent_units=64,
    dense_units=32,
)


def test_set_fit_kwargs_exceptions_when_validation_data():
    """
    Test set_fit_kwargs method raises exceptions when validation data is provided.
    """
    
    fit_kwargs = {
        "epochs": 10, 
        "series_val": series.iloc[:10],
        "exog_val": exog.iloc[:10]
    }
    forecaster = ForecasterRnn(
        estimator=model, levels=["1", "2", "3"], lags=3, 
        fit_kwargs=fit_kwargs
    )

    wrong_fit_kwargs = {
        "epochs": 25,
        "series_val": [1, 2, 3],
        "exog_val": exog.iloc[10:20]
    } 
    err_msg = re.escape(
        "`series_val` must be a pandas DataFrame. Got <class 'list'>."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster.set_fit_kwargs(wrong_fit_kwargs)

    wrong_fit_kwargs = {
        "epochs": 25,
        "series_val": series.iloc[10:20],
    }
    err_msg = re.escape(
        "If `series_val` is provided, `exog_val` must also be "
        "provided using the `fit_kwargs` argument when the "
        "estimator has exogenous variables."
    )
    with pytest.raises(ValueError, match=err_msg):
        forecaster.set_fit_kwargs(wrong_fit_kwargs)

    wrong_fit_kwargs = {
        "epochs": 25,
        "series_val": series.iloc[10:20],
        "exog_val": 42
    }
    err_msg = re.escape(
        "`exog_val` must be a pandas Series or DataFrame. Got <class 'int'>."
    )
    with pytest.raises(TypeError, match=err_msg):
        forecaster.set_fit_kwargs(wrong_fit_kwargs)


def test_set_fit_kwargs():
    """
    Test set_fit_kwargs method.
    """
    fit_kwargs = {
        "epochs": 10, 
        "batch_size": 16,
        "series_val": series.iloc[:10],
        "exog_val": exog.iloc[:10]
    }
    new_fit_kwargs = {
        "epochs": 25
    }

    forecaster = ForecasterRnn(
        estimator=model, levels=["1", "2", "3"], lags=3, 
        fit_kwargs=fit_kwargs
    )   
    forecaster.set_fit_kwargs(new_fit_kwargs)
    results = forecaster.fit_kwargs

    expected = {'epochs': 25}

    assert results == expected
    assert forecaster.series_val is None
    assert forecaster.exog_val is None


def test_set_fit_kwargs_with_validation():
    """
    Test set_fit_kwargs method when fit_kwargs contains validation data.
    """
    fit_kwargs = {
        "epochs": 10, 
        "batch_size": 16,
        "series_val": series.iloc[:10],
        "exog_val": exog.iloc[:10]
    }
    new_fit_kwargs = {
        "epochs": 25,
        "series_val": series.iloc[10:20],
        "exog_val": exog.iloc[10:20]
    }

    forecaster = ForecasterRnn(
        estimator=model, levels=["1", "2", "3"], lags=3, 
        fit_kwargs=fit_kwargs
    )   
    forecaster.set_fit_kwargs(new_fit_kwargs)
    results = forecaster.fit_kwargs

    expected = {'epochs': 25}

    assert results == expected
    pd.testing.assert_frame_equal(forecaster.series_val, series.iloc[10:20])
    pd.testing.assert_frame_equal(forecaster.exog_val, exog.iloc[10:20])
