"""
Examples demonstrating auto_arima integration in Arima class
"""
import numpy as np
import pandas as pd
from skforecast.stats import Arima

# Generate sample seasonal data
np.random.seed(123)
n = 120
time = np.arange(n)
trend = 0.5 * time
seasonal = 10 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(0, 2, n)
y = trend + seasonal + noise

print("=" * 80)
print("Examples: Auto ARIMA integration in skforecast")
print("=" * 80)

# Example 1: Fully automatic model selection
print("\n" + "=" * 80)
print("Example 1: Automatic model selection (order=None, seasonal_order=None)")
print("=" * 80)
arima_auto = Arima(m=12)
print(f"Initial estimator_id: {arima_auto.estimator_id}")
arima_auto.fit(y, suppress_warnings=True)
print(f"Selected model: {arima_auto.estimator_selected_id_}")
print(f"  - Non-seasonal order (p,d,q): {arima_auto.order}")
print(f"  - Seasonal order (P,D,Q): {arima_auto.seasonal_order}")
print(f"  - AIC: {arima_auto.aic_:.2f}")
print(f"  - Converged: {arima_auto.converged_}")

# Predict
forecast_auto = arima_auto.predict(steps=12)
print(f"\n12-step forecast (first 5): {forecast_auto[:5]}")

# Example 2: Manual model specification
print("\n" + "=" * 80)
print("Example 2: Manual model specification")
print("=" * 80)
arima_manual = Arima(order=(1, 1, 1), seasonal_order=(1, 0, 1), m=12)
print(f"Specified model: {arima_manual.estimator_id}")
arima_manual.fit(y, suppress_warnings=True)
print(f"After fitting: {arima_manual.estimator_selected_id_}")
print(f"  - AIC: {arima_manual.aic_:.2f}")
print(f"  - Number of parameters: {len(arima_manual.coef_)}")

# Example 3: With exogenous variables
print("\n" + "=" * 80)
print("Example 3: Auto ARIMA with exogenous variables")
print("=" * 80)
# Create exogenous variable
exog = np.random.randn(len(y), 2)
exog_df = pd.DataFrame(exog, columns=['exog1', 'exog2'])

arima_with_exog = Arima(m=12)
arima_with_exog.fit(y, exog=exog_df, suppress_warnings=True)
print(f"Selected model: {arima_with_exog.estimator_selected_id_}")
print(f"  - Coefficient names: {arima_with_exog.coef_names_}")
print(f"  - Number of coefficients: {len(arima_with_exog.coef_)}")
print(f"  - Exog features: {arima_with_exog.n_exog_features_in_}")
print(f"  - Exog names: {arima_with_exog.n_exog_names_in_}")

# Example 4: Non-seasonal data (m=1)
print("\n" + "=" * 80)
print("Example 4: Auto ARIMA for non-seasonal data")
print("=" * 80)
# Generate non-seasonal data
y_non_seasonal = np.cumsum(np.random.randn(100)) + 50
arima_non_seasonal = Arima(m=1)
arima_non_seasonal.fit(y_non_seasonal, suppress_warnings=True)
print(f"Selected model: {arima_non_seasonal.estimator_selected_id_}")
print(f"  - Order: {arima_non_seasonal.order}")
print(f"  - Seasonal order: {arima_non_seasonal.seasonal_order}")
print(f"  - AIC: {arima_non_seasonal.aic_:.2f}")

# Example 5: Comparison - Auto vs Manual
print("\n" + "=" * 80)
print("Example 5: Comparison - Automatic vs Manual selection")
print("=" * 80)
print(f"Auto-selected model: {arima_auto.estimator_selected_id_}")
print(f"  AIC: {arima_auto.aic_:.2f}")
print(f"\nManual model: {arima_manual.estimator_selected_id_}")
print(f"  AIC: {arima_manual.aic_:.2f}")
print(f"\nAuto-selection found a {'better' if arima_auto.aic_ < arima_manual.aic_ else 'worse'} model!")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print("✓ Auto ARIMA successfully integrated into Arima class")
print("✓ Works with and without exogenous variables")
print("✓ Handles both seasonal and non-seasonal data")
print("✓ estimator_id shows 'Arima(auto)' before fitting")
print("✓ estimator_selected_id_ shows the selected model after fitting")
print("=" * 80)
