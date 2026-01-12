"""
Test script to verify auto_arima integration in Arima class
"""
import numpy as np
import pandas as pd
from skforecast.stats import Arima

# Generate sample data
np.random.seed(42)
n = 100
trend = np.linspace(0, 10, n)
seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 1, n)
y = trend + seasonal + noise

print("=" * 70)
print("Testing Arima with auto_arima integration")
print("=" * 70)

# Test 1: Auto selection (both order and seasonal_order None)
print("\n1. Test with order=None and seasonal_order=None (full auto)")
print("-" * 70)
model_auto = Arima(m=12)
print(f"Before fitting: {model_auto.estimator_id}")
model_auto.fit(y, suppress_warnings=True)
print(f"After fitting: {model_auto.estimator_selected_id_}")
print(f"Selected order: {model_auto.order}")
print(f"Selected seasonal_order: {model_auto.seasonal_order}")
print(f"AIC: {model_auto.aic_:.4f}")
bic_auto_str = f"{model_auto.bic_:.4f}" if model_auto.bic_ is not None else "None"
print(f"BIC: {bic_auto_str}")

# Test 2: Manual specification
print("\n2. Test with specified order=(1,0,1) and seasonal_order=(0,0,0)")
print("-" * 70)
model_manual = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0), m=12)
print(f"Before fitting: {model_manual.estimator_id}")
model_manual.fit(y, suppress_warnings=True)
print(f"After fitting: {model_manual.estimator_selected_id_}")
print(f"Order: {model_manual.order}")
print(f"Seasonal order: {model_manual.seasonal_order}")
print(f"AIC: {model_manual.aic_:.4f}")
bic_str = f"{model_manual.bic_:.4f}" if model_manual.bic_ is not None else "None"
print(f"BIC: {bic_str}")

# Test 3: Predictions
print("\n3. Test predictions from auto-selected model")
print("-" * 70)
predictions = model_auto.predict(steps=5)
print(f"Predictions for next 5 steps: {predictions}")

# Test 4: Verify attributes are set correctly
print("\n4. Verify fitted attributes")
print("-" * 70)
print(f"Converged: {model_auto.converged_}")
print(f"Number of coefficients: {len(model_auto.coef_)}")
print(f"Coefficient names: {model_auto.coef_names_}")
print(f"Sigma2: {model_auto.sigma2_:.6f}")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)
