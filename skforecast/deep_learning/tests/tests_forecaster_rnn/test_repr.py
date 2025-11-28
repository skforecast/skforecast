# Unit test __repr__ and _repr_html_ ForecasterRnn
# ==============================================================================
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set keras backend before importing ForecasterRnn
os.environ['KERAS_BACKEND'] = 'torch'

from skforecast.deep_learning import ForecasterRnn
from skforecast.deep_learning.utils import create_and_compile_model


def test_repr_and_repr_html():
    """
    Comprehensive test for __repr__ and _repr_html_ methods.
    """
    # Test 1: Basic forecaster - not fitted
    model = create_and_compile_model(
        series=pd.DataFrame({'serie_1': np.random.rand(50)}),
        levels='serie_1',
        lags=3,
        steps=5,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster = ForecasterRnn(
        estimator=model,
        levels='serie_1',
        lags=3
    )
    
    # Test __repr__
    result_repr = repr(forecaster)
    assert isinstance(result_repr, str)
    assert len(result_repr) > 0
    assert "ForecasterRnn" in result_repr
    assert "Estimator:" in result_repr
    assert "Lags:" in result_repr
    assert "Window size:" in result_repr
    assert "Maximum steps to predict:" in result_repr
    assert "Target series (levels):" in result_repr
    assert "Training range: None" in result_repr
    assert "Last fit date: None" in result_repr
    
    # Test _repr_html_
    result_html = forecaster._repr_html_()
    assert isinstance(result_html, str)
    assert len(result_html) > 0
    assert "<div" in result_html
    assert "</div>" in result_html
    assert "ForecasterRnn" in result_html
    assert "<details" in result_html
    assert "<summary>" in result_html
    assert "General Information" in result_html
    assert "https://skforecast.org" in result_html
    assert "API Reference" in result_html or "User Guide" in result_html
    assert "Not fitted" in result_html or "None" in result_html
    
    # Test 2: Forecaster with custom lags
    model2 = create_and_compile_model(
        series=pd.DataFrame({'serie_1': np.random.rand(50)}),
        levels='serie_1',
        lags=[1, 2, 3, 5, 7],
        steps=5,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster2 = ForecasterRnn(
        estimator=model2,
        levels='serie_1',
        lags=[1, 2, 3, 5, 7]
    )
    result_repr2 = repr(forecaster2)
    assert "Lags:" in result_repr2
    assert isinstance(result_repr2, str)
    
    # Test 3: Forecaster with multiple levels
    series_multi = pd.DataFrame({
        'serie_1': np.random.rand(50),
        'serie_2': np.random.rand(50)
    })
    
    model3 = create_and_compile_model(
        series=series_multi,
        levels=['serie_1', 'serie_2'],
        lags=3,
        steps=4,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster3 = ForecasterRnn(
        estimator=model3,
        levels=['serie_1', 'serie_2'],
        lags=3
    )
    result_repr3 = repr(forecaster3)
    assert "Target series (levels):" in result_repr3
    assert "serie_1" in result_repr3
    assert "serie_2" in result_repr3
    
    # Test 4: Forecaster with forecaster_id
    model4 = create_and_compile_model(
        series=pd.DataFrame({'serie_1': np.random.rand(50)}),
        levels='serie_1',
        lags=5,
        steps=3,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster4 = ForecasterRnn(
        estimator=model4,
        levels='serie_1',
        lags=5,
        forecaster_id="test_rnn_forecaster"
    )
    result_repr4 = repr(forecaster4)
    assert "Forecaster id: test_rnn_forecaster" in result_repr4
    
    # Test 5: Fitted forecaster without exog
    series = pd.DataFrame({
        'serie_1': np.random.rand(100)
    }, index=pd.date_range('2020-01-01', periods=100, freq='D'))
    
    model5 = create_and_compile_model(
        series=series,
        levels='serie_1',
        lags=3,
        steps=4,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster5 = ForecasterRnn(
        estimator=model5,
        levels='serie_1',
        lags=3,
        transformer_series=MinMaxScaler()
    )
    forecaster5.fit(series=series)
    
    result_repr5 = repr(forecaster5)
    result_html5 = forecaster5._repr_html_()
    
    assert "Training range:" in result_repr5
    assert "Training range: None" not in result_repr5
    assert "Last fit date:" in result_repr5
    assert "Last fit date: None" not in result_repr5
    assert "2020" in result_html5
    assert "Series names:" in result_repr5
    
    # Test 6: Fitted forecaster with exog
    series = pd.DataFrame({
        'serie_1': np.random.rand(100)
    }, index=pd.date_range('2020-01-01', periods=100, freq='D'))
    exog = pd.DataFrame({
        'exog_1': np.random.rand(100),
        'exog_2': np.random.rand(100)
    }, index=pd.date_range('2020-01-01', periods=100, freq='D'))
    
    model6 = create_and_compile_model(
        series=series,
        exog=exog,
        levels='serie_1',
        lags=3,
        steps=3,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16
    )
    
    forecaster6 = ForecasterRnn(
        estimator=model6,
        levels='serie_1',
        lags=3,
        transformer_series=MinMaxScaler(),
        transformer_exog=MinMaxScaler()
    )
    forecaster6.fit(series=series, exog=exog)
    
    result_repr6 = repr(forecaster6)
    result_html6 = forecaster6._repr_html_()
    
    assert "Exogenous included: True" in result_repr6
    assert "exog_1" in result_repr6 or "Exogenous names:" in result_repr6
    assert "MinMaxScaler" in result_repr6
    assert "Transformer for series:" in result_repr6
    assert "Exogenous" in result_html6 or "Exogenous Variables" in result_html6
    
    # Test 7: Check all HTML sections are present
    expected_sections = [
        "General Information",
        "Exogenous Variables",
        "Data Transformations",
        "Training Information",
        "Estimator Parameters"
    ]
    
    for section in expected_sections:
        assert section in result_html6, f"Section '{section}' not found in HTML output"
    
    # Test 8: Test with transformer_series as dict
    series = pd.DataFrame({
        'serie_1': np.random.rand(100),
        'serie_2': np.random.rand(100)
    }, index=pd.date_range('2020-01-01', periods=100, freq='D'))
    
    model8 = create_and_compile_model(
        series=series,
        levels=['serie_1', 'serie_2'],
        lags=2,
        steps=3,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster8 = ForecasterRnn(
        estimator=model8,
        levels=['serie_1', 'serie_2'],
        lags=2,
        transformer_series={
            'serie_1': MinMaxScaler(), 
            'serie_2': StandardScaler()
        }
    )
    result_repr8 = repr(forecaster8)
    assert "Transformer for series:" in result_repr8
    
    # Test 9: Test with fit_kwargs
    model9 = create_and_compile_model(
        series=pd.DataFrame({'serie_1': np.random.rand(50)}),
        levels='serie_1',
        lags=2,
        steps=3,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster9 = ForecasterRnn(
        estimator=model9,
        levels='serie_1',
        lags=2,
        fit_kwargs={'epochs': 10, 'batch_size': 32}
    )
    result_repr9 = repr(forecaster9)
    assert "fit_kwargs:" in result_repr9
    
    # Test 10: Verify layers_names appear in repr
    model10 = create_and_compile_model(
        series=pd.DataFrame({'serie_1': np.random.rand(50)}),
        levels='serie_1',
        lags=3,
        steps=5,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster10 = ForecasterRnn(
        estimator=model10,
        levels='serie_1',
        lags=3
    )
    result_repr10 = repr(forecaster10)
    result_html10 = forecaster10._repr_html_()
    assert "Layers names:" in result_repr10
    assert "Layers names" in result_html10
    
    # Test 11: Verify keras backend appears in repr after fitting
    series = pd.DataFrame({
        'serie_1': np.random.rand(100)
    }, index=pd.date_range('2020-01-01', periods=100, freq='D'))
    
    model11 = create_and_compile_model(
        series=series,
        levels='serie_1',
        lags=3,
        steps=3,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster11 = ForecasterRnn(
        estimator=model11,
        levels='serie_1',
        lags=3
    )
    forecaster11.fit(series=series)
    
    result_repr11 = repr(forecaster11)
    result_html11 = forecaster11._repr_html_()
    assert "Keras backend:" in result_repr11
    assert "Keras backend" in result_html11
    
    # Test 12: Test with different recurrent layer types (GRU)
    model12 = create_and_compile_model(
        series=pd.DataFrame({'serie_1': np.random.rand(50)}),
        levels='serie_1',
        lags=3,
        steps=5,
        recurrent_layer='GRU',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster12 = ForecasterRnn(
        estimator=model12,
        levels='serie_1',
        lags=3
    )
    result_repr12 = repr(forecaster12)
    assert "ForecasterRnn" in result_repr12
    assert isinstance(result_repr12, str)
    
    # Test 13: Verify compile parameters appear in repr
    model13 = create_and_compile_model(
        series=pd.DataFrame({'serie_1': np.random.rand(50)}),
        levels='serie_1',
        lags=3,
        steps=5,
        recurrent_layer='LSTM',
        recurrent_units=32,
        dense_units=16,
    )
    
    forecaster13 = ForecasterRnn(
        estimator=model13,
        levels='serie_1',
        lags=3
    )
    result_repr13 = repr(forecaster13)
    assert "Compile parameters:" in result_repr13
    assert "Estimator parameters:" in result_repr13
