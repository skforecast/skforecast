# ARIMA Quick Reference Card

## Installation

```bash
pip install numpy scipy numba
```

## Basic Usage

```python
from arima import ARIMA
import numpy as np

# Create model
model = ARIMA(order=(p, d, q))

# Fit to data
model.fit(y)

# Generate forecasts
forecasts = model.predict(steps=10)
```

## Model Orders

| Model | Order | Description |
|-------|-------|-------------|
| AR(p) | (p, 0, 0) | Autoregressive of order p |
| MA(q) | (0, 0, q) | Moving average of order q |
| ARMA(p,q) | (p, 0, q) | Combined AR and MA |
| ARIMA(p,d,q) | (p, d, q) | ARMA with d differences |

## Common Patterns

### Stationary Series (AR/MA/ARMA)
```python
# Use d=0
model = ARIMA(order=(1, 0, 1))
model.fit(y)
```

### Random Walk
```python
# Use d=1
model = ARIMA(order=(1, 1, 0))
model.fit(y)
```

### Trend Series
```python
# Use d=1 or d=2
model = ARIMA(order=(1, 2, 0))
model.fit(y)
```

## Accessing Results

```python
# After fitting
model.ar_coef_        # AR coefficients (length p)
model.ma_coef_        # MA coefficients (length q)
model.intercept_      # Intercept/mean
model.sigma2_         # Residual variance
model.residuals_      # Model residuals
model.is_fitted_      # True if fitted
```

## Complete Example

```python
import numpy as np
from arima import ARIMA

# Generate random walk data
np.random.seed(42)
y = np.cumsum(np.random.randn(200) * 0.5) + 10

# Split train/test
y_train = y[:150]
y_test = y[150:]

# Fit ARIMA(1,1,0)
model = ARIMA(order=(1, 1, 0))
model.fit(y_train)

# Forecast test period
forecasts = model.predict(steps=len(y_test))

# Evaluate
mae = np.mean(np.abs(forecasts - y_test))
print(f"MAE: {mae:.4f}")

# Access parameters
print(f"AR coef: {model.ar_coef_[0]:.4f}")
print(f"Variance: {model.sigma2_:.4f}")
```

## Model Selection Guide

### Choose p (AR order)
- Look at PACF (partial autocorrelation)
- PACF cuts off after lag p
- Start with p=1 or p=2

### Choose d (differencing)
- d=0: Stationary series (constant mean/variance)
- d=1: Random walk, single trend
- d=2: Polynomial trend

### Choose q (MA order)
- Look at ACF (autocorrelation)
- ACF cuts off after lag q
- Start with q=0 or q=1

## Performance Tips

1. **Use small orders**: p, q ≤ 3 for best speed
2. **Longer series**: Better parameter estimates (n > 100)
3. **Differencing**: Only use d>0 if non-stationary
4. **Validation**: Always use train/test split

## Error Handling

```python
try:
    model = ARIMA(order=(1, 1, 0))
    model.fit(y)
    forecasts = model.predict(steps=10)
except ValueError as e:
    print(f"Error: {e}")
    # Handle insufficient data, NaN values, etc.
```

## Common Issues

### Issue: Poor forecasts
**Solution**: Try different orders, check stationarity

### Issue: Slow fitting
**Solution**: Reduce p, q; check for very long series

### Issue: NaN in forecasts
**Solution**: Check input data for NaN/Inf values

### Issue: Unstable predictions
**Solution**: Check AR/MA coefficients near 1.0

## Utility Functions

```python
from arima import check_stationarity

# Check if series is stationary
is_stationary = check_stationarity(y)
if not is_stationary:
    # Use differencing
    model = ARIMA(order=(1, 1, 0))
else:
    # No differencing needed
    model = ARIMA(order=(1, 0, 0))
```

## Testing

```bash
# Run test suite
pytest test_arima.py -v

# Run validation examples
python validate.py

# Run practical examples
python examples.py
```

## Limitations

❌ No seasonal components (SARIMA)
❌ No exogenous variables (ARIMAX)
❌ No automatic order selection
❌ No prediction intervals
✅ Fast and simple for basic ARIMA

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| y_t | Time series at time t |
| φ_i | AR coefficient i |
| θ_j | MA coefficient j |
| ε_t | Error/residual at time t |
| ∇ | Difference operator |
| σ² | Error variance |

## ARIMA Equation

$$\nabla^d y_t = c + \sum_{i=1}^{p} \phi_i \nabla^d y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

Where $\epsilon_t \sim N(0, \sigma^2)$

## Resources

- **README.md**: Full documentation
- **TECHNICAL.md**: Implementation details
- **SUMMARY.md**: Project overview
- **examples.py**: 5 practical examples
- **validate.py**: Validation and benchmarks
- **test_arima.py**: 27 unit tests

## Version Info

- Python: 3.7+
- NumPy: Any recent version
- SciPy: Any recent version
- Numba: 0.50+

## Support

For issues or questions:
1. Check README.md for documentation
2. Run examples.py for usage patterns
3. Review TECHNICAL.md for implementation details
4. Check test_arima.py for test cases
