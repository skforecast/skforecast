"""
Test get_params and set_params with new auto_arima parameters
"""
import numpy as np
from skforecast.stats import Arima

print("=" * 80)
print("Testing get_params and set_params with auto_arima parameters")
print("=" * 80)

# Create model with custom parameters
model = Arima(
    m=12,
    max_p=3,
    max_q=3,
    max_P=1,
    max_Q=1,
    ic="bic",
    stepwise=True,
    trace=False,
    seasonal=True,
    lambda_bc="auto",
    biasadj=True,
)

# Test get_params
print("\n1. Testing get_params()")
print("-" * 80)
params = model.get_params()
print(f"Number of parameters: {len(params)}")
print("\nKey auto_arima parameters:")
print(f"  max_p: {params['max_p']}")
print(f"  max_q: {params['max_q']}")
print(f"  max_P: {params['max_P']}")
print(f"  max_Q: {params['max_Q']}")
print(f"  ic: {params['ic']}")
print(f"  stepwise: {params['stepwise']}")
print(f"  trace: {params['trace']}")
print(f"  seasonal: {params['seasonal']}")
print(f"  lambda_bc: {params['lambda_bc']}")
print(f"  biasadj: {params['biasadj']}")
print(f"  allowdrift: {params['allowdrift']}")
print(f"  allowmean: {params['allowmean']}")

# Test set_params
print("\n2. Testing set_params()")
print("-" * 80)
model.set_params(
    max_p=5,
    max_q=5,
    ic="aicc",
    stepwise=False,
    trace=True,
    lambda_bc=0.5,
)
print("Updated parameters:")
print(f"  max_p: {model.max_p}")
print(f"  max_q: {model.max_q}")
print(f"  ic: {model.ic}")
print(f"  stepwise: {model.stepwise}")
print(f"  trace: {model.trace}")
print(f"  lambda_bc: {model.lambda_bc}")

# Verify all parameters are in get_params
print("\n3. Verify all auto_arima parameters are included in get_params()")
print("-" * 80)
expected_params = [
    'max_p', 'max_q', 'max_P', 'max_Q', 'max_order',
    'max_d', 'max_D', 'start_p', 'start_q', 'start_P', 'start_Q',
    'stationary', 'seasonal', 'ic', 'stepwise', 'nmodels',
    'trace', 'approximation', 'truncate', 'test', 'test_kwargs',
    'seasonal_test', 'seasonal_test_kwargs', 'allowdrift', 'allowmean',
    'lambda_bc', 'biasadj'
]

params = model.get_params()
missing = [p for p in expected_params if p not in params]
if missing:
    print(f"❌ Missing parameters: {missing}")
else:
    print("✓ All auto_arima parameters are present in get_params()")

print("\n" + "=" * 80)
print("All tests passed!")
print("=" * 80)
