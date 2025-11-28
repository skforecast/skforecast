# Unit test __repr__ and _repr_html_ ForecasterRecursiveClassifier
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skforecast.recursive import ForecasterRecursiveClassifier


def test_repr_and_repr_html():
    """
    Comprehensive test for __repr__ and _repr_html_ methods.
    """
    # Test 1: Basic forecaster - not fitted
    forecaster = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=5
    )
    
    # Test __repr__
    result_repr = repr(forecaster)
    assert isinstance(result_repr, str)
    assert len(result_repr) > 0
    assert "ForecasterRecursiveClassifier" in result_repr
    assert "LogisticRegression" in result_repr
    assert "Estimator:" in result_repr
    assert "Lags:" in result_repr
    assert "Window size:" in result_repr
    assert "Classes:" in result_repr
    assert "Number of classes:" in result_repr
    assert "Feature encoding:" in result_repr
    assert "Training range: None" in result_repr
    assert "Last fit date: None" in result_repr
    
    # Test _repr_html_
    result_html = forecaster._repr_html_()
    assert isinstance(result_html, str)
    assert len(result_html) > 0
    assert "<div" in result_html
    assert "</div>" in result_html
    assert "ForecasterRecursiveClassifier" in result_html
    assert "LogisticRegression" in result_html
    assert "<details" in result_html
    assert "<summary>" in result_html
    assert "General Information" in result_html
    assert "Classification Information" in result_html
    assert "https://skforecast.org" in result_html
    assert "API Reference" in result_html or "User Guide" in result_html
    assert "Not fitted" in result_html or "None" in result_html
    
    # Test 2: Forecaster with custom lags
    forecaster2 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=[1, 2, 3, 5]
    )
    result_repr2 = repr(forecaster2)
    assert "Lags:" in result_repr2
    assert "[1 2 3 5]" in result_repr2 or "1 2 3 5" in result_repr2
    
    # Test 3: Forecaster with forecaster_id
    forecaster3 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=5,
        forecaster_id="test_classifier_forecaster"
    )
    result_repr3 = repr(forecaster3)
    assert "Forecaster id: test_classifier_forecaster" in result_repr3
    
    # Test 4: Fitted forecaster without exog
    forecaster4 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3
    )
    # Create classification target with discrete classes
    y = pd.Series(
        np.random.choice(['low', 'medium', 'high'], size=50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    forecaster4.fit(y)
    
    result_repr4 = repr(forecaster4)
    result_html4 = forecaster4._repr_html_()
    
    assert "Training range:" in result_repr4
    assert "Training range: None" not in result_repr4
    assert "Last fit date:" in result_repr4
    assert "Last fit date: None" not in result_repr4
    assert "Classes:" in result_repr4
    assert "Number of classes: 3" in result_repr4
    assert "2020" in result_html4
    
    # Test 5: Fitted forecaster with exog
    forecaster5 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3,
        transformer_exog=StandardScaler()
    )
    y = pd.Series(
        np.random.choice(['class_a', 'class_b'], size=50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    exog = pd.DataFrame({
        'exog_1': np.random.rand(50),
        'exog_2': np.random.rand(50)
    }, index=pd.date_range('2020-01-01', periods=50, freq='D'))
    
    forecaster5.fit(y, exog=exog)
    result_repr5 = repr(forecaster5)
    result_html5 = forecaster5._repr_html_()
    
    assert "Exogenous included: True" in result_repr5
    assert "exog_1" in result_repr5 or "Exogenous names:" in result_repr5
    assert "StandardScaler" in result_repr5
    assert "Exogenous included" in result_html5 or "Exogenous Variables" in result_html5
    assert "True" in result_html5
    
    # Test 6: Check all HTML sections are present
    expected_sections = [
        "General Information",
        "Classification Information",
        "Exogenous Variables",
        "Training Information",
        "Estimator Parameters"
    ]
    
    for section in expected_sections:
        assert section in result_html5, f"Section '{section}' not found in HTML output"
    
    # Test 7: Verify class encoding information appears
    forecaster7 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3
    )
    y = pd.Series(
        np.random.choice([0, 1, 2], size=50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='y'
    )
    forecaster7.fit(y)
    
    result_repr7 = repr(forecaster7)
    result_html7 = forecaster7._repr_html_()
    
    assert "Classes:" in result_repr7
    assert "Number of classes:" in result_repr7
    assert "Class encoding" in result_html7
    
    # Test 8: Forecaster with different features_encoding options
    for encoding in ['auto', 'ordinal']:
        forecaster8 = ForecasterRecursiveClassifier(
            estimator=LogisticRegression(),
            lags=3,
            features_encoding=encoding
        )
        result_repr8 = repr(forecaster8)
        assert f"Feature encoding: {encoding}" in result_repr8
    
    # Test 9: Forecaster with weight_func
    def custom_weight_func(index):
        return np.ones(len(index))
    
    forecaster9 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3,
        weight_func=custom_weight_func
    )
    result_repr9 = repr(forecaster9)
    assert "Weight function included: True" in result_repr9
    
    # Test 10: Forecaster with fit_kwargs
    forecaster10 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=2,
        fit_kwargs={'sample_weight': None}
    )
    result_repr10 = repr(forecaster10)
    assert "fit_kwargs:" in result_repr10
    
    # Test 11: Verify window_features appear in repr
    from skforecast.preprocessing import RollingFeatures
    rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=3)
    forecaster11 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3,
        window_features=rolling
    )
    result_repr11 = repr(forecaster11)
    result_html11 = forecaster11._repr_html_()
    assert "Window features:" in result_repr11
    assert "Window features" in result_html11
    
    # Test 12: Verify creation date and skforecast version appear
    forecaster12 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3
    )
    result_repr12 = repr(forecaster12)
    result_html12 = forecaster12._repr_html_()
    
    assert "Creation date:" in result_repr12
    assert "Skforecast version:" in result_repr12
    assert "Python version:" in result_repr12
    assert "Creation date" in result_html12
    assert "Skforecast version" in result_html12
    
    # Test 13: Fitted forecaster - verify series name is captured
    forecaster13 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3
    )
    y = pd.Series(
        np.random.choice(['yes', 'no'], size=50),
        index=pd.date_range('2020-01-01', periods=50, freq='D'),
        name='my_classification_target'
    )
    forecaster13.fit(y)
    
    result_repr13 = repr(forecaster13)
    assert "Series name: my_classification_target" in result_repr13
    
    # Test 14: Verify training index type for RangeIndex
    forecaster14 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3
    )
    y = pd.Series(
        np.random.choice(['A', 'B', 'C'], size=50),
        index=pd.RangeIndex(start=0, stop=50, step=1),
        name='y'
    )
    forecaster14.fit(y)
    
    result_repr14 = repr(forecaster14)
    assert "Training index type: RangeIndex" in result_repr14
    
    # Test 15: Forecaster with RandomForestClassifier
    forecaster15 = ForecasterRecursiveClassifier(
        estimator=RandomForestClassifier(n_estimators=10, random_state=123),
        lags=3
    )
    result_repr15 = repr(forecaster15)
    assert "RandomForestClassifier" in result_repr15
    
    # Test 16: Verify Fit Kwargs section in HTML
    forecaster16 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3,
        fit_kwargs={'sample_weight': None}
    )
    result_html16 = forecaster16._repr_html_()
    assert "Fit Kwargs" in result_html16
    
    # Test 17: Fitted forecaster with string classes
    forecaster17 = ForecasterRecursiveClassifier(
        estimator=LogisticRegression(),
        lags=3
    )
    y = pd.Series(
        np.random.choice(['cat', 'dog', 'bird', 'fish'], size=60),
        index=pd.date_range('2020-01-01', periods=60, freq='D'),
        name='animal'
    )
    forecaster17.fit(y)
    
    result_repr17 = repr(forecaster17)
    assert "Number of classes: 4" in result_repr17
    assert "Classes:" in result_repr17
