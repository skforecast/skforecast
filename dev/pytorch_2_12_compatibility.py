# ==============================================================================
# PyTorch Compatibility Check for ForecasterRnn
# ==============================================================================

import os
import numpy as np
import pandas as pd

# Set backend to PyTorch before importing Keras or skforecast
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model

def run_compatibility_check():
    """
    Run a full compatibility check for PyTorch backend using ForecasterRnn.
    """
    print(f"Testing PyTorch version: {torch.__version__}")
    print(f"Testing Keras version: {keras.__version__}")
    
    # Generate synthetic data
    series = pd.DataFrame({
        "level_1": pd.Series(np.random.normal(size=100)),
        "level_2": pd.Series(np.random.normal(size=100))
    })
    
    exog = pd.DataFrame({
        "exog_1": pd.Series(np.random.normal(size=100)),
        "exog_2": pd.Series(np.random.normal(size=100))
    })

    lags = 5
    steps = 3
    levels = "level_1"

    # Create and compile a Keras model
    model = create_and_compile_model(
        series=series,
        exog=exog,
        levels=levels,
        lags=lags,
        steps=steps,
        recurrent_layer="LSTM",
        recurrent_units=64,
        dense_units=32,
    )

    # Initialize ForecasterRnn
    forecaster = ForecasterRnn(estimator=model, levels=levels, lags=lags)
    forecaster.set_fit_kwargs({"epochs": 2, "verbose": 0})
    
    # Fit the model
    print("Fitting ForecasterRnn...")
    forecaster.fit(
        series=series, 
        exog=exog, 
        store_in_sample_residuals=True
    )
    
    # Check backend
    assert forecaster.keras_backend_ == "torch", f"Expected backend 'torch', got '{forecaster.keras_backend_}'"
    print("Backend confirmed as 'torch'.")

    # Predict
    print("Predicting...")
    # The last_window ends at index 99, so future exog must start at 100.
    exog_predict = pd.DataFrame({
        "exog_1": pd.Series(np.random.normal(size=steps)),
        "exog_2": pd.Series(np.random.normal(size=steps))
    }, index=pd.RangeIndex(start=100, stop=100 + steps))
    
    predictions = forecaster.predict(steps=steps, exog=exog_predict)
    assert not predictions.empty, "Predictions DataFrame should not be empty."
    assert len(predictions) == steps, f"Expected {steps} predictions, got {len(predictions)}"
    
    # Predict Interval
    print("Predicting intervals...")
    predictions_interval = forecaster.predict_interval(steps=steps, exog=exog_predict)
    assert "lower_bound" in predictions_interval.columns, "lower_bound missing from interval predictions"
    assert "upper_bound" in predictions_interval.columns, "upper_bound missing from interval predictions"
    
    print("All compatibility checks passed successfully!")

if __name__ == "__main__":
    run_compatibility_check()
