# Unit test set_in_sample_residuals ForecasterRnn
# ==============================================================================ç
import os
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
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
    lags=3,
    steps=5,
    levels=["1", "2", "3"],
    recurrent_units=64,
    dense_units=32,
)

model_exog = create_and_compile_model(
    series=series,
    exog=exog,
    lags=3,
    steps=5,
    levels=["1", "2", "3"],
    recurrent_units=64,
    dense_units=32,
)


def test_set_in_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterRnn(
        model, levels=["1", "2", "3"], lags=3
    )

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `set_in_sample_residuals()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_in_sample_residuals(series=series)


def test_set_in_sample_residuals_TypeError_when_series_not_dataframe():
    """
    Test TypeError is raised when series is not a DataFrame.
    """
    forecaster = ForecasterRnn(
        model, levels=["1", "2", "3"], lags=3
    )
    forecaster.fit(series=series, exog=exog)

    wrong_series = np.arange(10)
    err_msg = re.escape(
        f"`series` must be a pandas DataFrame. Got {type(wrong_series)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_in_sample_residuals(series=wrong_series)


@pytest.mark.parametrize("diff_index", 
                         [pd.RangeIndex(start=50, stop=100), 
                          pd.date_range(start='1991-07-01', periods=50, freq='MS')], 
                         ids=lambda idx: f'diff_index: {idx[[0, -1]]}')
def test_set_in_sample_residuals_IndexError_when_series_has_different_index_than_training(diff_index):
    """
    Test IndexError is raised when series has different index than training.
    """
    forecaster = ForecasterRnn(
        model, levels=["1", "2", "3"], lags=3
    )
    forecaster.fit(series=series, exog=exog)

    series_diff_index = series.copy()
    series_diff_index.index = diff_index
    series_diff_index_range = series_diff_index.index[[0, -1]]

    err_msg = re.escape(
        f"The index range of `series` does not match the range "
        f"used during training. Please ensure the index is aligned "
        f"with the training data.\n"
        f"    Expected : {forecaster.training_range_}\n"
        f"    Received : {series_diff_index_range}"
    )
    with pytest.raises(IndexError, match = err_msg):
        forecaster.set_in_sample_residuals(series=series_diff_index)


def test_set_in_sample_residuals_ValueError_when_X_train_features_names_out_not_the_same():
    """
    Test ValueError is raised when X_train_features_names_out are different from 
    the ones used in training.
    """
    model_2_exog = create_and_compile_model(
        series=series,
        exog=exog[['exog_1', 'exog_2']],
        lags=3,
        steps=5,
        levels=["1", "2", "3"],
        recurrent_units=64,
        dense_units=32,
    )
    
    forecaster = ForecasterRnn(
        model_2_exog, levels=["1", "2", "3"], lags=3, transformer_exog=None
    )
    forecaster.fit(series=series, exog=exog[['exog_1', 'exog_2']])

    err_msg = re.escape(
        "Feature mismatch detected after matrix creation. The features "
        "generated from the provided data do not match those used during "
        "the training process. To correctly set in-sample residuals, "
        "ensure that the same data and preprocessing steps are applied.\n"
        "    Expected output : ['lag_3', 'lag_2', 'lag_1', 'exog_1', 'exog_2']\n"
        "    Current output  : ['lag_3', 'lag_2', 'lag_1', 'exog_1', 'exog_3']"
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_in_sample_residuals(series=series, exog=exog[['exog_1', 'exog_3']])


def test_set_in_sample_residuals_store_same_residuals_as_fit():
    """
    Test that set_in_sample_residuals stores same residuals as fit.
    """
    forecaster = ForecasterRnn(
        model, levels=["1", "2", "3"], lags=3
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    scaler_id_after_fit = id(forecaster.transformer_series_["1"])
    residuals_fit = forecaster.in_sample_residuals_

    forecaster.in_sample_residuals_ = None  # Reset in-sample residuals
    forecaster.set_in_sample_residuals(series=series)
    scaler_id_after_set_in_sample_residuals = id(forecaster.transformer_series_["1"])
    residuals_in_sample = forecaster.in_sample_residuals_

    # Transformer
    assert scaler_id_after_fit == scaler_id_after_set_in_sample_residuals

    # Residuals
    assert residuals_fit.keys() == residuals_in_sample.keys()
    for level in residuals_fit.keys():
        np.testing.assert_almost_equal(residuals_fit[level], residuals_in_sample[level])


def test_set_in_sample_residuals_store_same_residuals_as_fit_exog():
    """
    Test that set_in_sample_residuals stores same residuals as fit with exogenous variables.
    """
    forecaster = ForecasterRnn(
        model_exog, levels=["1", "2", "3"], lags=3
    )
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    scaler_id_after_fit = id(forecaster.transformer_series_["1"])
    residuals_fit = forecaster.in_sample_residuals_

    forecaster.in_sample_residuals_ = None  # Reset in-sample residuals
    forecaster.set_in_sample_residuals(series=series, exog=exog)
    scaler_id_after_set_in_sample_residuals = id(forecaster.transformer_series_["1"])
    residuals_in_sample = forecaster.in_sample_residuals_

    # Transformer
    assert scaler_id_after_fit == scaler_id_after_set_in_sample_residuals

    # Residuals
    assert residuals_fit.keys() == residuals_in_sample.keys()
    for level in residuals_fit.keys():
        np.testing.assert_almost_equal(residuals_fit[level], residuals_in_sample[level])
