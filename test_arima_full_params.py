"""
Test script to verify all auto_arima parameters work in Arima class
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

print("=" * 80)
print("Testing Arima with all auto_arima parameters")
print("=" * 80)

# Test 1: Auto selection with custom parameters
print("\n1. Auto selection with custom parameters")
print("-" * 80)
model = Arima(
    m=12,
    max_p=3,
    max_q=3,
    max_P=1,
    max_Q=1,
    max_order=4,
    ic="bic",
    stepwise=True,
    trace=True,
    approximation=False,
    seasonal=True,
    stationary=False,
)
print(f"Before fitting: {model.estimator_id}")
model.fit(y, suppress_warnings=True)
print(f"After fitting: {model.estimator_selected_id_}")
print(f"Selected order: {model.order}")
print(f"Selected seasonal_order: {model.seasonal_order}")
print(f"AIC: {model.aic_:.4f}")

# Test 2: With Box-Cox transformation
print("\n2. Auto selection with Box-Cox transformation")
print("-" * 80)
model_bc = Arima(
    m=12,
    lambda_bc="auto",
    biasadj=True,
    max_p=2,
    max_q=2,
    trace=False,
)
model_bc.fit(y, suppress_warnings=True)
print(f"Selected model: {model_bc.estimator_selected_id_}")
print(f"Lambda: {model_bc.model_.get('lambda', 'N/A')}")
print(f"Bias adjustment: {model_bc.model_.get('biasadj', 'N/A')}")

# Test 3: With custom unit root tests
print("\n3. Auto selection with custom unit root test")
print("-" * 80)
model_test = Arima(
    m=12,
    test="adf",
    seasonal_test="seas",
    max_d=1,
    max_D=1,
    trace=False,
)
model_test.fit(y, suppress_warnings=True)
print(f"Selected model: {model_test.estimator_selected_id_}")
print(f"Order: {model_test.order}")

# Test 4: Stationary model
print("\n4. Auto selection with stationary constraint")
print("-" * 80)
model_stat = Arima(
    m=1,
    stationary=True,
    max_p=3,
    max_q=3,
    trace=False,
)
model_stat.fit(y, suppress_warnings=True)
print(f"Selected model: {model_stat.estimator_selected_id_}")
print(f"Order (should have d=0, D=0): {model_stat.order}")
print(f"Seasonal order: {model_stat.seasonal_order}")

# Test 5: Non-stepwise (exhaustive search) with small limits
print("\n5. Auto selection with exhaustive search")
print("-" * 80)
model_exhaustive = Arima(
    m=1,
    stepwise=False,
    max_p=2,
    max_q=2,
    max_P=0,
    max_Q=0,
    trace=False,
)
model_exhaustive.fit(y, suppress_warnings=True)
print(f"Selected model: {model_exhaustive.estimator_selected_id_}")

# Test 6: Verify all parameters are accessible
print("\n6. Verify all parameters are stored")
print("-" * 80)
print(f"max_p: {model.max_p}")
print(f"max_q: {model.max_q}")
print(f"max_P: {model.max_P}")
print(f"max_Q: {model.max_Q}")
print(f"max_order: {model.max_order}")
print(f"max_d: {model.max_d}")
print(f"max_D: {model.max_D}")
print(f"start_p: {model.start_p}")
print(f"start_q: {model.start_q}")
print(f"start_P: {model.start_P}")
print(f"start_Q: {model.start_Q}")
print(f"stationary: {model.stationary}")
print(f"seasonal: {model.seasonal}")
print(f"ic: {model.ic}")
print(f"stepwise: {model.stepwise}")
print(f"nmodels: {model.nmodels}")
print(f"trace: {model.trace}")
print(f"approximation: {model.approximation}")
print(f"truncate: {model.truncate}")
print(f"test: {model.test}")
print(f"test_kwargs: {model.test_kwargs}")
print(f"seasonal_test: {model.seasonal_test}")
print(f"seasonal_test_kwargs: {model.seasonal_test_kwargs}")
print(f"allowdrift: {model.allowdrift}")
print(f"allowmean: {model.allowmean}")
print(f"lambda_bc: {model.lambda_bc}")
print(f"biasadj: {model.biasadj}")

print("\n" + "=" * 80)
print("All tests completed successfully!")
print("=" * 80)
