"""
Quick start example for ARIMA implementation
"""

import numpy as np
from arima import ARIMA

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def example_1_simple_forecast():
    """Example 1: Simple AR model with forecasting"""
    print("\n" + "="*60)
    print("Example 1: AR(1) Model")
    print("="*60)
    
    # Generate AR(1) data: y_t = 0.7 * y_{t-1} + noise
    np.random.seed(42)
    n = 150
    y = np.zeros(n)
    y[0] = np.random.randn()
    
    for t in range(1, n):
        y[t] = 0.7 * y[t-1] + np.random.randn() * 0.3
    
    # Split into train/test
    train_size = 130
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Fit model
    model = ARIMA(order=(1, 0, 0))
    model.fit(y_train)
    
    # Make predictions
    forecasts = model.predict(steps=len(y_test))
    
    # Print results
    print(f"\nEstimated AR coefficient: {model.ar_coef_[0]:.4f} (true: 0.7000)")
    print(f"Residual std: {np.sqrt(model.sigma2_):.4f}")
    print(f"\nForecasts: {forecasts[:5]}")
    print(f"Actuals:   {y_test[:5]}")
    
    # Calculate MAE
    mae = np.mean(np.abs(forecasts - y_test))
    print(f"\nMean Absolute Error: {mae:.4f}")


def example_2_random_walk():
    """Example 2: ARIMA with differencing for random walk"""
    print("\n" + "="*60)
    print("Example 2: Random Walk with ARIMA(1,1,0)")
    print("="*60)
    
    # Generate random walk
    np.random.seed(123)
    n = 200
    y = np.cumsum(np.random.randn(n) * 0.5) + 50
    
    # Split into train/test
    train_size = 180
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    print(f"\nOriginal series - Mean: {np.mean(y_train):.2f}, Std: {np.std(y_train):.2f}")
    print(f"First few values: {y_train[:5]}")
    print(f"Last few values:  {y_train[-5:]}")
    
    # Fit ARIMA with differencing
    model = ARIMA(order=(1, 1, 0))
    model.fit(y_train)
    
    print(f"\nAfter differencing - Series length: {len(model.y_diff_)}")
    print(f"Differenced mean: {np.mean(model.y_diff_):.4f}")
    print(f"Differenced std:  {np.std(model.y_diff_):.4f}")
    
    # Make predictions
    forecasts = model.predict(steps=len(y_test))
    
    print(f"\nForecasts: {forecasts}")
    print(f"Actuals:   {y_test}")
    
    mae = np.mean(np.abs(forecasts - y_test))
    print(f"\nMean Absolute Error: {mae:.4f}")


def example_3_arma_model():
    """Example 3: ARMA model"""
    print("\n" + "="*60)
    print("Example 3: ARMA(2,2) Model")
    print("="*60)
    
    # Generate ARMA(2,2) process
    np.random.seed(456)
    n = 300
    ar_coef = [0.6, -0.3]
    ma_coef = [0.4, 0.2]
    
    errors = np.random.randn(n) * 0.5
    y = np.zeros(n)
    
    for t in range(2, n):
        ar_term = ar_coef[0] * y[t-1] + ar_coef[1] * y[t-2]
        ma_term = ma_coef[0] * errors[t-1] + ma_coef[1] * errors[t-2]
        y[t] = ar_term + errors[t] + ma_term
    
    # Fit model
    model = ARIMA(order=(2, 0, 2))
    model.fit(y)
    
    print(f"\nTrue AR coefficients: {ar_coef}")
    print(f"Estimated AR:         {model.ar_coef_}")
    print(f"\nTrue MA coefficients: {ma_coef}")
    print(f"Estimated MA:         {model.ma_coef_}")
    print(f"\nResidual variance: {model.sigma2_:.4f}")
    
    # Generate forecasts
    forecasts = model.predict(steps=10)
    print(f"\n10-step forecast: {forecasts}")


def example_4_visualization():
    """Example 4: Visualization of forecasts"""
    print("\n" + "="*60)
    print("Example 4: Forecast Visualization")
    print("="*60)
    
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available. Skipping visualization example.")
        print("Install matplotlib to see plots: pip install matplotlib")
        return
    
    # Generate data
    np.random.seed(789)
    n = 200
    t = np.arange(n)
    
    # Trend + noise
    y = 0.1 * t + 10 + np.random.randn(n) * 2
    
    # Split
    train_size = 150
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Fit model with differencing to handle trend
    model = ARIMA(order=(2, 1, 1))
    model.fit(y_train)
    
    # Forecast
    forecast_steps = len(y_test) + 20  # Forecast beyond test set
    forecasts = model.predict(steps=forecast_steps)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(range(train_size), y_train, 'b-', label='Training Data', linewidth=2)
    
    # Plot test data
    test_range = range(train_size, train_size + len(y_test))
    plt.plot(test_range, y_test, 'g-', label='Test Data', linewidth=2)
    
    # Plot forecasts
    forecast_range = range(train_size, train_size + forecast_steps)
    plt.plot(forecast_range, forecasts, 'r--', label='Forecasts', linewidth=2)
    
    plt.axvline(x=train_size, color='k', linestyle=':', alpha=0.5, label='Train/Test Split')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('ARIMA(2,1,1) Forecasting Example', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('arima_forecast_example.png', dpi=150)
    print("\nPlot saved to 'arima_forecast_example.png'")
    
    # Print statistics
    mae_in_sample = np.mean(np.abs(forecasts[:len(y_test)] - y_test))
    print(f"\nIn-sample MAE: {mae_in_sample:.4f}")
    print(f"Last training value: {y_train[-1]:.2f}")
    print(f"First forecast: {forecasts[0]:.2f}")
    print(f"Last test value: {y_test[-1]:.2f}")
    print(f"Forecast at end of test: {forecasts[len(y_test)-1]:.2f}")


def example_5_model_comparison():
    """Example 5: Compare different ARIMA orders"""
    print("\n" + "="*60)
    print("Example 5: Model Order Comparison")
    print("="*60)
    
    # Generate AR(2) data
    np.random.seed(999)
    n = 200
    y = np.zeros(n)
    y[0] = np.random.randn()
    y[1] = np.random.randn()
    
    for t in range(2, n):
        y[t] = 0.6 * y[t-1] - 0.2 * y[t-2] + np.random.randn() * 0.5
    
    # Split
    train_size = 150
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Try different orders
    orders = [
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
        (1, 0, 1),
        (2, 0, 1),
    ]
    
    print(f"\n{'Order':<15} {'Res. Var.':<15} {'MAE':<15} {'Params':<30}")
    print("-" * 75)
    
    best_mae = float('inf')
    best_order = None
    
    for order in orders:
        try:
            model = ARIMA(order=order)
            model.fit(y_train)
            forecasts = model.predict(steps=len(y_test))
            mae = np.mean(np.abs(forecasts - y_test))
            
            params_str = f"AR:{len(model.ar_coef_)} MA:{len(model.ma_coef_)}"
            print(f"{str(order):<15} {model.sigma2_:<15.4f} {mae:<15.4f} {params_str:<30}")
            
            if mae < best_mae:
                best_mae = mae
                best_order = order
                
        except Exception as e:
            print(f"{str(order):<15} ERROR: {str(e)}")
    
    print(f"\nBest model: ARIMA{best_order} with MAE = {best_mae:.4f}")
    print(f"True model: AR(2) with coefficients [0.6, -0.2]")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print(" ARIMA Implementation - Quick Start Examples")
    print("="*60)
    
    example_1_simple_forecast()
    example_2_random_walk()
    example_3_arma_model()
    example_4_visualization()
    example_5_model_comparison()
    
    print("\n" + "="*60)
    print(" All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
