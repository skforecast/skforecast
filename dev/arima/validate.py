"""
Validation and demonstration script for ARIMA implementation.
Shows usage examples and compares performance.
"""

import numpy as np
import time
from arima import ARIMA, check_stationarity


def generate_ar_process(n=200, phi=0.7, sigma=0.5, seed=42):
    """Generate an AR(1) process"""
    np.random.seed(seed)
    y = np.zeros(n)
    y[0] = np.random.randn()
    
    for t in range(1, n):
        y[t] = phi * y[t-1] + np.random.randn() * sigma
    
    return y


def generate_ma_process(n=200, theta=0.6, sigma=0.5, seed=42):
    """Generate an MA(1) process"""
    np.random.seed(seed)
    errors = np.random.randn(n) * sigma
    y = np.zeros(n)
    y[0] = errors[0]
    
    for t in range(1, n):
        y[t] = errors[t] + theta * errors[t-1]
    
    return y


def generate_arma_process(n=200, phi=0.5, theta=0.4, sigma=0.5, seed=42):
    """Generate an ARMA(1,1) process"""
    np.random.seed(seed)
    errors = np.random.randn(n) * sigma
    y = np.zeros(n)
    y[0] = errors[0]
    
    for t in range(1, n):
        y[t] = phi * y[t-1] + errors[t] + theta * errors[t-1]
    
    return y


def generate_random_walk(n=200, sigma=0.5, seed=42):
    """Generate a random walk (requires differencing)"""
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n) * sigma) + 10


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def validate_ar_model():
    """Validate AR model estimation"""
    print_section("AR(1) Model Validation")
    
    # True parameters
    true_phi = 0.7
    y = generate_ar_process(n=300, phi=true_phi, sigma=0.5)
    
    print(f"True AR coefficient: {true_phi:.4f}")
    print(f"Series length: {len(y)}")
    print(f"Stationary: {check_stationarity(y)}")
    
    # Fit model
    start = time.time()
    model = ARIMA(order=(1, 0, 0))
    model.fit(y)
    fit_time = time.time() - start
    
    print(f"\nFitting time: {fit_time:.4f} seconds")
    print(f"Estimated AR coefficient: {model.ar_coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Residual variance: {model.sigma2_:.4f}")
    print(f"Estimation error: {abs(model.ar_coef_[0] - true_phi):.4f}")
    
    # Generate forecasts
    forecasts = model.predict(steps=10)
    print(f"\n10-step ahead forecasts: {forecasts[:5]} ...")
    
    return model


def validate_ma_model():
    """Validate MA model estimation"""
    print_section("MA(1) Model Validation")
    
    # True parameters
    true_theta = 0.6
    y = generate_ma_process(n=300, theta=true_theta, sigma=0.5)
    
    print(f"True MA coefficient: {true_theta:.4f}")
    print(f"Series length: {len(y)}")
    
    # Fit model
    start = time.time()
    model = ARIMA(order=(0, 0, 1))
    model.fit(y)
    fit_time = time.time() - start
    
    print(f"\nFitting time: {fit_time:.4f} seconds")
    print(f"Estimated MA coefficient: {model.ma_coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Residual variance: {model.sigma2_:.4f}")
    print(f"Estimation error: {abs(model.ma_coef_[0] - true_theta):.4f}")
    
    # Generate forecasts
    forecasts = model.predict(steps=10)
    print(f"\n10-step ahead forecasts: {forecasts[:5]} ...")
    
    return model


def validate_arma_model():
    """Validate ARMA model estimation"""
    print_section("ARMA(1,1) Model Validation")
    
    # True parameters
    true_phi = 0.5
    true_theta = 0.4
    y = generate_arma_process(n=300, phi=true_phi, theta=true_theta, sigma=0.5)
    
    print(f"True AR coefficient: {true_phi:.4f}")
    print(f"True MA coefficient: {true_theta:.4f}")
    print(f"Series length: {len(y)}")
    
    # Fit model
    start = time.time()
    model = ARIMA(order=(1, 0, 1))
    model.fit(y)
    fit_time = time.time() - start
    
    print(f"\nFitting time: {fit_time:.4f} seconds")
    print(f"Estimated AR coefficient: {model.ar_coef_[0]:.4f}")
    print(f"Estimated MA coefficient: {model.ma_coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Residual variance: {model.sigma2_:.4f}")
    print(f"AR estimation error: {abs(model.ar_coef_[0] - true_phi):.4f}")
    print(f"MA estimation error: {abs(model.ma_coef_[0] - true_theta):.4f}")
    
    # Generate forecasts
    forecasts = model.predict(steps=10)
    print(f"\n10-step ahead forecasts: {forecasts[:5]} ...")
    
    return model


def validate_arima_model():
    """Validate ARIMA model with differencing"""
    print_section("ARIMA(1,1,0) Model Validation (Random Walk)")
    
    y = generate_random_walk(n=300, sigma=0.5)
    
    print(f"Series length: {len(y)}")
    print(f"Stationary (original): {check_stationarity(y)}")
    print(f"First few values: {y[:5]}")
    print(f"Last few values: {y[-5:]}")
    
    # Fit model
    start = time.time()
    model = ARIMA(order=(1, 1, 0))
    model.fit(y)
    fit_time = time.time() - start
    
    print(f"\nFitting time: {fit_time:.4f} seconds")
    print(f"Differenced series length: {len(model.y_diff_)}")
    print(f"Stationary (differenced): {check_stationarity(model.y_diff_)}")
    print(f"Estimated AR coefficient: {model.ar_coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Residual variance: {model.sigma2_:.4f}")
    
    # Generate forecasts
    forecasts = model.predict(steps=10)
    print(f"\n10-step ahead forecasts: {forecasts}")
    print(f"Last training value: {y[-1]:.4f}")
    print(f"Forecast continues from: {forecasts[0]:.4f}")
    
    return model


def validate_high_order_model():
    """Validate high-order ARIMA model"""
    print_section("High-Order ARIMA(3,1,2) Model")
    
    np.random.seed(42)
    # Generate more complex series
    n = 500
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 50)
    noise = np.random.randn(n) * 2
    y = trend + seasonal + noise
    
    print(f"Series length: {len(y)}")
    print(f"Mean: {np.mean(y):.4f}")
    print(f"Std: {np.std(y):.4f}")
    
    # Fit model
    start = time.time()
    model = ARIMA(order=(3, 1, 2))
    model.fit(y)
    fit_time = time.time() - start
    
    print(f"\nFitting time: {fit_time:.4f} seconds")
    print(f"AR coefficients: {model.ar_coef_}")
    print(f"MA coefficients: {model.ma_coef_}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Residual variance: {model.sigma2_:.4f}")
    
    # Generate forecasts
    forecasts = model.predict(steps=20)
    print(f"\n20-step ahead forecasts (first 10): {forecasts[:10]}")
    
    return model


def performance_benchmark():
    """Benchmark performance on various series lengths"""
    print_section("Performance Benchmark")
    
    series_lengths = [100, 200, 500, 1000, 2000]
    orders = [(1, 0, 0), (1, 1, 1), (2, 1, 2), (3, 1, 3)]
    
    print(f"{'Series Length':<15} {'Order':<15} {'Fit Time (s)':<15} {'Predict Time (s)':<15}")
    print("-" * 70)
    
    for n in series_lengths:
        y = generate_ar_process(n=n, seed=42)
        
        for order in orders:
            try:
                model = ARIMA(order=order)
                
                # Measure fit time
                start = time.time()
                model.fit(y)
                fit_time = time.time() - start
                
                # Measure predict time
                start = time.time()
                forecasts = model.predict(steps=50)
                predict_time = time.time() - start
                
                print(f"{n:<15} {str(order):<15} {fit_time:<15.6f} {predict_time:<15.6f}")
            except Exception as e:
                print(f"{n:<15} {str(order):<15} ERROR: {str(e)[:30]}")


def demonstrate_api_usage():
    """Demonstrate scikit-learn compatible API"""
    print_section("Scikit-Learn API Demonstration")
    
    # Generate data
    y = generate_arma_process(n=200, seed=42)
    
    print("# Initialize model")
    print("model = ARIMA(order=(1, 0, 1))")
    model = ARIMA(order=(1, 0, 1))
    
    print("\n# Fit model")
    print("model.fit(y)")
    model.fit(y)
    
    print("\n# Access fitted attributes")
    print(f"model.ar_coef_: {model.ar_coef_}")
    print(f"model.ma_coef_: {model.ma_coef_}")
    print(f"model.intercept_: {model.intercept_:.4f}")
    print(f"model.sigma2_: {model.sigma2_:.4f}")
    print(f"model.is_fitted_: {model.is_fitted_}")
    
    print("\n# Generate predictions")
    print("forecasts = model.predict(steps=10)")
    forecasts = model.predict(steps=10)
    print(f"forecasts: {forecasts}")
    
    print("\n# Chain operations")
    print("model2 = ARIMA(order=(2, 1, 0)).fit(y)")
    model2 = ARIMA(order=(2, 1, 0)).fit(y)
    print(f"model2.ar_coef_: {model2.ar_coef_}")


def main():
    """Run all validation tests"""
    print("\n" + "="*70)
    print(" ARIMA Implementation Validation Suite")
    print(" Optimized with Numba JIT compilation")
    print("="*70)
    
    # Run validations
    validate_ar_model()
    validate_ma_model()
    validate_arma_model()
    validate_arima_model()
    validate_high_order_model()
    
    # Performance benchmark
    performance_benchmark()
    
    # API demonstration
    demonstrate_api_usage()
    
    print("\n" + "="*70)
    print(" Validation Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
