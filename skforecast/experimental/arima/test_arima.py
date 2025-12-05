"""
Test suite for ARIMA implementation
"""

import numpy as np
import pytest
from arima import ARIMA, check_stationarity


class TestARIMAInitialization:
    """Test ARIMA initialization"""
    
    def test_default_initialization(self):
        """Test default ARIMA(1,0,0) initialization"""
        model = ARIMA()
        assert model.order == (1, 0, 0)
        assert model.p == 1
        assert model.d == 0
        assert model.q == 0
        assert not model.is_fitted_
    
    def test_custom_order(self):
        """Test custom order initialization"""
        model = ARIMA(order=(2, 1, 2))
        assert model.order == (2, 1, 2)
        assert model.p == 2
        assert model.d == 1
        assert model.q == 2
    
    def test_invalid_order(self):
        """Test that invalid orders raise errors"""
        with pytest.raises(ValueError):
            ARIMA(order=(-1, 0, 0))
        
        with pytest.raises(ValueError):
            ARIMA(order=(0, 0, 0))


class TestARIMAFitting:
    """Test ARIMA fitting functionality"""
    
    def test_fit_ar_model(self):
        """Test fitting AR model"""
        np.random.seed(42)
        n = 200
        y = np.zeros(n)
        y[0] = np.random.randn()
        
        # Generate AR(1) process
        phi = 0.7
        for t in range(1, n):
            y[t] = phi * y[t-1] + np.random.randn() * 0.5
        
        model = ARIMA(order=(1, 0, 0))
        model.fit(y)
        
        assert model.is_fitted_
        assert len(model.ar_coef_) == 1
        assert len(model.ma_coef_) == 0
        assert model.sigma2_ > 0
        # Check AR coefficient is close to true value
        assert abs(model.ar_coef_[0] - phi) < 0.2
    
    def test_fit_ma_model(self):
        """Test fitting MA model"""
        np.random.seed(42)
        n = 200
        errors = np.random.randn(n)
        y = np.zeros(n)
        
        # Generate MA(1) process
        theta = 0.6
        y[0] = errors[0]
        for t in range(1, n):
            y[t] = errors[t] + theta * errors[t-1]
        
        model = ARIMA(order=(0, 0, 1))
        model.fit(y)
        
        assert model.is_fitted_
        assert len(model.ar_coef_) == 0
        assert len(model.ma_coef_) == 1
        assert model.sigma2_ > 0
    
    def test_fit_arma_model(self):
        """Test fitting ARMA model"""
        np.random.seed(42)
        n = 200
        errors = np.random.randn(n) * 0.5
        y = np.zeros(n)
        
        # Generate ARMA(1,1) process
        phi = 0.5
        theta = 0.4
        y[0] = errors[0]
        for t in range(1, n):
            y[t] = phi * y[t-1] + errors[t] + theta * errors[t-1]
        
        model = ARIMA(order=(1, 0, 1))
        model.fit(y)
        
        assert model.is_fitted_
        assert len(model.ar_coef_) == 1
        assert len(model.ma_coef_) == 1
        assert model.sigma2_ > 0
    
    def test_fit_with_differencing(self):
        """Test fitting with differencing"""
        np.random.seed(42)
        n = 200
        
        # Generate random walk (needs d=1)
        y = np.cumsum(np.random.randn(n) * 0.5) + 10
        
        model = ARIMA(order=(1, 1, 0))
        model.fit(y)
        
        assert model.is_fitted_
        assert len(model.y_diff_) == n - 1
        assert model.d == 1
    
    def test_fit_multiple_differencing(self):
        """Test fitting with multiple differencing"""
        np.random.seed(42)
        n = 200
        
        # Generate series with trend
        t = np.arange(n)
        y = 0.1 * t**2 + np.random.randn(n) * 0.5
        
        model = ARIMA(order=(1, 2, 0))
        model.fit(y)
        
        assert model.is_fitted_
        assert len(model.y_diff_) == n - 2
        assert model.d == 2
    
    def test_fit_insufficient_data(self):
        """Test that insufficient data raises error"""
        y = np.array([1, 2, 3])
        model = ARIMA(order=(5, 0, 0))
        
        with pytest.raises(ValueError, match="Time series too short"):
            model.fit(y)
    
    def test_fit_with_nan(self):
        """Test that NaN values raise error"""
        y = np.array([1, 2, np.nan, 4, 5])
        model = ARIMA(order=(1, 0, 0))
        
        with pytest.raises(ValueError, match="NaN"):
            model.fit(y)


class TestARIMAPrediction:
    """Test ARIMA prediction functionality"""
    
    def test_predict_ar_model(self):
        """Test prediction with AR model"""
        np.random.seed(42)
        n = 200
        y = np.zeros(n)
        y[0] = np.random.randn()
        
        # Generate AR(1) process
        phi = 0.7
        for t in range(1, n):
            y[t] = phi * y[t-1] + np.random.randn() * 0.5
        
        model = ARIMA(order=(1, 0, 0))
        model.fit(y)
        
        # Predict 10 steps ahead
        forecasts = model.predict(steps=10)
        
        assert len(forecasts) == 10
        assert not np.any(np.isnan(forecasts))
        assert not np.any(np.isinf(forecasts))
    
    def test_predict_with_differencing(self):
        """Test prediction with differencing"""
        np.random.seed(42)
        n = 200
        
        # Generate random walk
        y = np.cumsum(np.random.randn(n) * 0.5) + 10
        
        model = ARIMA(order=(1, 1, 0))
        model.fit(y)
        
        forecasts = model.predict(steps=5)
        
        assert len(forecasts) == 5
        # Forecasts should be in reasonable range of last values
        assert np.abs(forecasts[0] - y[-1]) < 10
    
    def test_predict_multiple_differencing(self):
        """Test prediction with multiple differencing"""
        np.random.seed(42)
        n = 100
        
        # Generate series with trend
        t = np.arange(n)
        y = 0.05 * t**2 + np.random.randn(n) * 0.5 + 10
        
        model = ARIMA(order=(1, 2, 0))
        model.fit(y)
        
        forecasts = model.predict(steps=10)
        
        assert len(forecasts) == 10
        assert not np.any(np.isnan(forecasts))
    
    def test_predict_without_fit(self):
        """Test that prediction without fitting raises error"""
        model = ARIMA(order=(1, 0, 0))
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(steps=5)
    
    def test_predict_invalid_steps(self):
        """Test that invalid steps raise error"""
        np.random.seed(42)
        y = np.random.randn(100)
        
        model = ARIMA(order=(1, 0, 0))
        model.fit(y)
        
        with pytest.raises(ValueError, match="steps must be at least 1"):
            model.predict(steps=0)


class TestDifferencing:
    """Test differencing operations"""
    
    def test_no_differencing(self):
        """Test d=0 returns unchanged series"""
        y = np.array([1, 2, 3, 4, 5])
        model = ARIMA(order=(1, 0, 0))
        
        y_diff, initial_values = model._difference_with_initial(y, 0)
        assert np.array_equal(y_diff, y)
        assert initial_values == []
    
    def test_first_differencing(self):
        """Test first differencing"""
        y = np.array([1, 3, 6, 10, 15])
        model = ARIMA(order=(1, 1, 0))
        
        y_diff, initial_values = model._difference_with_initial(y, 1)
        expected = np.array([2, 3, 4, 5])
        assert np.array_equal(y_diff, expected)
        assert initial_values == [15]  # Last value before differencing
    
    def test_second_differencing(self):
        """Test second differencing"""
        y = np.array([1, 3, 6, 10, 15, 21])
        model = ARIMA(order=(1, 2, 0))
        
        y_diff, initial_values = model._difference_with_initial(y, 2)
        # First diff: [2, 3, 4, 5, 6]
        # Second diff: [1, 1, 1, 1]
        expected = np.array([1, 1, 1, 1])
        assert np.array_equal(y_diff, expected)
        assert len(initial_values) == 2
        assert initial_values[0] == 21  # Last value before first diff
        assert initial_values[1] == 6   # Last value before second diff
    
    def test_inverse_differencing_d1(self):
        """Test inverse differencing for d=1"""
        y_diff = np.array([5.0])  # One forecast on differenced scale
        initial_values = [15.0]  # Last original value
        
        model = ARIMA(order=(1, 1, 0))
        y_inv = model._inverse_difference(y_diff, initial_values, 1)
        
        # Last value was 15, difference is 5, so next should be 20
        assert y_inv[0] == 20.0
    
    def test_inverse_differencing_d2(self):
        """Test inverse differencing for d=2"""
        y_diff = np.array([1.0])  # One forecast on twice-differenced scale
        # initial_values[0] = last value at level 0 (original)
        # initial_values[1] = last value at level 1 (first diff)
        initial_values = [15.0, 5.0]
        
        model = ARIMA(order=(1, 2, 0))
        y_inv = model._inverse_difference(y_diff, initial_values, 2)
        
        # First integrate: 5 + cumsum([1]) = 6
        # Second integrate: 15 + cumsum([6]) = 21
        assert y_inv[0] == 21.0


class TestStationarityCheck:
    """Test stationarity check function"""
    
    def test_stationary_series(self):
        """Test detection of stationary series"""
        np.random.seed(42)
        y = np.random.randn(200)
        
        is_stationary = check_stationarity(y)
        assert is_stationary
    
    def test_nonstationary_series(self):
        """Test detection of non-stationary series"""
        np.random.seed(42)
        # Random walk
        y = np.cumsum(np.random.randn(200) * 5)
        
        is_stationary = check_stationarity(y)
        assert not is_stationary


class TestSciKitLearnAPI:
    """Test scikit-learn API compatibility"""
    
    def test_fit_returns_self(self):
        """Test that fit returns self"""
        np.random.seed(42)
        y = np.random.randn(100)
        
        model = ARIMA(order=(1, 0, 0))
        result = model.fit(y)
        
        assert result is model
    
    def test_attributes_after_fit(self):
        """Test that expected attributes exist after fit"""
        np.random.seed(42)
        y = np.random.randn(100)
        
        model = ARIMA(order=(2, 1, 1))
        model.fit(y)
        
        # Check all expected attributes
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'ar_coef_')
        assert hasattr(model, 'ma_coef_')
        assert hasattr(model, 'intercept_')
        assert hasattr(model, 'sigma2_')
        assert hasattr(model, 'residuals_')
        assert hasattr(model, 'is_fitted_')
        
        # Check shapes
        assert len(model.ar_coef_) == 2
        assert len(model.ma_coef_) == 1
        assert isinstance(model.intercept_, float)
        assert isinstance(model.sigma2_, float)


class TestPerformance:
    """Test performance and edge cases"""
    
    def test_large_series(self):
        """Test handling of large time series"""
        np.random.seed(42)
        n = 1000
        y = np.random.randn(n)
        
        model = ARIMA(order=(2, 1, 2))
        model.fit(y)
        forecasts = model.predict(steps=50)
        
        assert len(forecasts) == 50
        assert model.is_fitted_
    
    def test_high_order_model(self):
        """Test fitting high-order models"""
        np.random.seed(42)
        y = np.random.randn(500)
        
        model = ARIMA(order=(5, 0, 5))
        model.fit(y)
        forecasts = model.predict(steps=10)
        
        assert len(forecasts) == 10
        assert len(model.ar_coef_) == 5
        assert len(model.ma_coef_) == 5
    
    def test_constant_series(self):
        """Test handling of constant series"""
        y = np.ones(100) * 5
        
        model = ARIMA(order=(1, 0, 0))
        model.fit(y)
        forecasts = model.predict(steps=5)
        
        # Should forecast approximately constant value
        assert np.allclose(forecasts, 5.0, atol=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
