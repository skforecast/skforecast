"""
Test script for prediction intervals functionality
"""
import numpy as np
import matplotlib.pyplot as plt
from arima import ARIMA

# Generate synthetic AR(2) data
np.random.seed(42)
n = 200
ar_coef_true = np.array([0.6, 0.3])
y = np.zeros(n)
y[0] = np.random.normal(0, 1)
y[1] = np.random.normal(0, 1)

for t in range(2, n):
    y[t] = ar_coef_true[0] * y[t-1] + ar_coef_true[1] * y[t-2] + np.random.normal(0, 1)

# Split into train and test
train_size = 150
y_train = y[:train_size]
y_test = y[train_size:]

# Fit ARIMA model
model = ARIMA(order=(2, 0, 0))
model.fit(y_train)

print("=" * 70)
print("ARIMA Model with Prediction Intervals")
print("=" * 70)
print(f"Order: {model.order}")
print(f"AR coefficients: {model.ar_coef_}")
print(f"True AR coefficients: {ar_coef_true}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Residual variance (σ²): {model.sigma2_:.4f}")
print()

# Test 1: Point forecasts
print("Test 1: Point Forecasts")
print("-" * 70)
forecasts = model.predict(steps=10)
print(f"10-step ahead forecasts: {forecasts[:5]}... (showing first 5)")
print()

# Test 2: Prediction intervals with default alpha
print("Test 2: 95% Prediction Intervals (alpha=0.05)")
print("-" * 70)
forecasts, lower, upper = model.predict_interval(steps=10, alpha=0.05)
print(f"{'Step':<6} {'Forecast':<12} {'Lower (95%)':<15} {'Upper (95%)':<15} {'Width':<10}")
print("-" * 70)
for i in range(10):
    width = upper[i] - lower[i]
    print(f"{i+1:<6} {forecasts[i]:<12.4f} {lower[i]:<15.4f} {upper[i]:<15.4f} {width:<10.4f}")
print()

# Test 3: Different confidence levels
print("Test 3: Different Confidence Levels")
print("-" * 70)
for confidence in [0.80, 0.90, 0.95, 0.99]:
    alpha = 1 - confidence
    _, lower, upper = model.predict_interval(steps=5, alpha=alpha)
    width_avg = np.mean(upper - lower)
    print(f"{confidence*100:.0f}% CI (α={alpha:.2f}): Average width = {width_avg:.4f}")
print()

# Test 4: Verify interval properties
print("Test 4: Interval Properties")
print("-" * 70)
forecasts_50, lower_50, upper_50 = model.predict_interval(steps=50, alpha=0.05)

# Check that intervals widen over time
widths = upper_50 - lower_50
print(f"Interval width at step 1:  {widths[0]:.4f}")
print(f"Interval width at step 10: {widths[9]:.4f}")
print(f"Interval width at step 50: {widths[49]:.4f}")
print(f"Widths increase over time: {np.all(np.diff(widths) >= 0)}")
print()

# Test 5: Coverage check
print("Test 5: Coverage Check (on test set)")
print("-" * 70)
test_steps = len(y_test)
forecasts_test, lower_test, upper_test = model.predict_interval(steps=test_steps, alpha=0.05)

# Check how many actual values fall within the intervals
coverage = np.mean((y_test >= lower_test) & (y_test <= upper_test))
print(f"Actual coverage: {coverage*100:.1f}% (expected ~95%)")
print(f"Test observations within intervals: {np.sum((y_test >= lower_test) & (y_test <= upper_test))}/{test_steps}")
print()

# Test 6: Edge cases
print("Test 6: Edge Cases")
print("-" * 70)
try:
    # Single step forecast
    f1, l1, u1 = model.predict_interval(steps=1, alpha=0.05)
    print(f"✓ Single step: forecast={f1[0]:.4f}, interval=[{l1[0]:.4f}, {u1[0]:.4f}]")
except Exception as e:
    print(f"✗ Single step failed: {e}")

try:
    # Invalid alpha
    model.predict_interval(steps=5, alpha=1.5)
    print("✗ Invalid alpha should raise error")
except ValueError as e:
    print(f"✓ Invalid alpha correctly rejected: {str(e)[:50]}...")

try:
    # Zero steps
    model.predict_interval(steps=0)
    print("✗ Zero steps should raise error")
except ValueError as e:
    print(f"✓ Zero steps correctly rejected: {str(e)[:50]}...")

print()

# Visualization
print("Generating visualization...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Forecasts with prediction intervals
forecast_steps = 30
forecasts_plot, lower_plot, upper_plot = model.predict_interval(steps=forecast_steps, alpha=0.05)
forecast_idx = np.arange(train_size, train_size + forecast_steps)

ax1.plot(np.arange(train_size), y_train, 'b-', label='Training data', alpha=0.7)
ax1.plot(forecast_idx, forecasts_plot, 'r-', label='Forecast', linewidth=2)
ax1.fill_between(forecast_idx, lower_plot, upper_plot, color='red', alpha=0.2, label='95% PI')
if train_size + forecast_steps <= len(y):
    ax1.plot(forecast_idx, y[train_size:train_size+forecast_steps], 'g--', 
             label='Actual (test)', linewidth=1.5, alpha=0.8)
ax1.axvline(x=train_size, color='black', linestyle='--', alpha=0.5, label='Train/Test split')
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.set_title('ARIMA Forecasts with 95% Prediction Intervals')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Interval width over forecast horizon
forecast_steps_long = 50
_, lower_long, upper_long = model.predict_interval(steps=forecast_steps_long, alpha=0.05)
widths_long = upper_long - lower_long

ax2.plot(np.arange(1, forecast_steps_long + 1), widths_long, 'b-', linewidth=2)
ax2.set_xlabel('Forecast Horizon (steps ahead)')
ax2.set_ylabel('Prediction Interval Width')
ax2.set_title('How Prediction Interval Width Grows with Forecast Horizon')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_intervals_demo.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved as 'prediction_intervals_demo.png'")
print()

print("=" * 70)
print("All tests completed successfully!")
print("=" * 70)
