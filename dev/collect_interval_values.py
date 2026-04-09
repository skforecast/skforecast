"""Collect new expected values for test_predict_interval.py after BUG-2 fix."""
import numpy as np
import pandas as pd
import platform
import sys
sys.path.insert(0, '/home/joaquin/Documents/GitHub/skforecast')
from skforecast.stats._arima import Arima
from skforecast.stats.tests.tests_arima.fixtures_arima import (
    air_passengers, multi_seasonal, fuel_consumption
)


def ar1_series(n=100, phi=0.7, sigma=1.0, seed=123):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, size=n)
    y = np.zeros(n, dtype=float)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y

print(f"Platform: {platform.system()}")

# ===========================================================================
# Base ARIMA(1,0,1) seed=42
# ===========================================================================
y = ar1_series(100, seed=42)
model = Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0))
model.fit(y)

r = model.predict_interval(steps=10, level=(80, 95))
print("\n=== Base ARIMA(1,0,1) seed=42 ===")
print("mean[:3]:", repr(r['mean'].values[:3]))
print("lower_95[:3]:", repr(r['lower_95'].values[:3]))
print("upper_95[:3]:", repr(r['upper_95'].values[:3]))

r90 = model.predict_interval(steps=10, level=(90,))
print("lower_90[:3]:", repr(r90['lower_90'].values[:3]))
print("upper_90[:3]:", repr(r90['upper_90'].values[:3]))

rc = model.predict_interval(steps=10, level=(50, 75, 99))
print("mean_cust[:2]:", repr(rc['mean'].values[:2]))
print("lower_50[:2]:", repr(rc['lower_50'].values[:2]))
print("upper_50[:2]:", repr(rc['upper_50'].values[:2]))
print("lower_99[:2]:", repr(rc['lower_99'].values[:2]))
print("upper_99[:2]:", repr(rc['upper_99'].values[:2]))

r90s = model.predict_interval(steps=5, level=90)
print("mean_90_5:", repr(r90s['mean'].values))
print("lower_90_5:", repr(r90s['lower_90'].values))
print("upper_90_5:", repr(r90s['upper_90'].values))

r95s = model.predict_interval(steps=5, level=(95,))
print("mean_95_5:", repr(r95s['mean'].values))
print("lower_95_5:", repr(r95s['lower_95'].values))
print("upper_95_5:", repr(r95s['upper_95'].values))

# ===========================================================================
# AR(1,0,0) with 2D exog
# ===========================================================================
np.random.seed(42)
y2 = ar1_series(80, seed=42)
exog_train = np.random.randn(80, 2)
model2 = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
model2.fit(y2, exog=exog_train)
np.random.seed(42)
exog_pred = np.random.randn(10, 2)
r2 = model2.predict_interval(steps=10, exog=exog_pred, level=(95,))
print("\n=== AR(1,0,0) with 2D exog ===")
print("mean[:3]:", repr(r2['mean'].values[:3]))
print("lower_95[:3]:", repr(r2['lower_95'].values[:3]))
print("upper_95[:3]:", repr(r2['upper_95'].values[:3]))

# ===========================================================================
# Seasonal AR model
# ===========================================================================
np.random.seed(123)
n_seas = 50
t = np.arange(n_seas)
seasonal_series = 2 * np.sin(2 * np.pi * t / 12) + 0.3 * np.random.randn(n_seas)
seasonal_series = pd.Series(seasonal_series)
model_seasonal = Arima(order=(1, 0, 0), seasonal_order=(1, 0, 0), m=12)
model_seasonal.fit(seasonal_series, suppress_warnings=True)
r_seas = model_seasonal.predict_interval(steps=10, level=(95,))
print("\n=== Seasonal AR model ===")
print("mean_first3:", repr(r_seas['mean'].values[:3]))
print("lower95_first3:", repr(r_seas['lower_95'].values[:3]))
print("upper95_first3:", repr(r_seas['upper_95'].values[:3]))
print("mean_last3:", repr(r_seas['mean'].values[-3:]))
print("lower95_last3:", repr(r_seas['lower_95'].values[-3:]))
print("upper95_last3:", repr(r_seas['upper_95'].values[-3:]))

# ===========================================================================
# AR(1,1,0) with differencing
# ===========================================================================
np.random.seed(42)
y_nonstat = np.cumsum(np.random.randn(50)) - 10
y_nonstat = pd.Series(y_nonstat)
model_diff = Arima(order=(1, 1, 0), seasonal_order=(0, 0, 0))
model_diff.fit(y_nonstat)
r_diff = model_diff.predict_interval(steps=5, level=(95,))
print("\n=== AR(1,1,0) with differencing ===")
print("mean:", repr(r_diff['mean'].values))
print("lower_95:", repr(r_diff['lower_95'].values))
print("upper_95:", repr(r_diff['upper_95'].values))

# ===========================================================================
# Fuel consumption
# ===========================================================================
model_fuel = Arima(
    order=(1, 1, 1), seasonal_order=(1, 1, 1), m=12,
    fit_intercept=True, enforce_stationarity=True, method="CSS-ML",
    n_cond=None, optim_method="BFGS", optim_kwargs={"maxiter": 2000},
)
model_fuel.fit(
    y=fuel_consumption.loc[:'1989-09-01', 'y'],
    exog=fuel_consumption.loc[:'1989-09-01'].drop(columns=['y']),
    suppress_warnings=True
)
r_fuel = model_fuel.predict_interval(
    steps=5, exog=fuel_consumption.loc['1989-09-01':].drop(columns=['y']),
    level=(95, 99)
)
print(f"\n=== Fuel consumption [{platform.system()}] ===")
print("mean:", repr(r_fuel['mean'].values))
print("lower_95:", repr(r_fuel['lower_95'].values))
print("upper_95:", repr(r_fuel['upper_95'].values))
print("lower_99:", repr(r_fuel['lower_99'].values))
print("upper_99:", repr(r_fuel['upper_99'].values))

# ===========================================================================
# exog DataFrame and Series
# ===========================================================================
np.random.seed(42)
y3 = ar1_series(80, seed=42)
exog_train_df = pd.DataFrame({
    'feature1': np.random.randn(80),
    'feature2': np.random.randn(80)
})
model3 = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
model3.fit(y3, exog=exog_train_df)
np.random.seed(123)
exog_pred_df = pd.DataFrame({
    'feature1': np.random.randn(5),
    'feature2': np.random.randn(5)
})
r3 = model3.predict_interval(steps=5, exog=exog_pred_df, level=(95,))
print("\n=== exog DataFrame ===")
print("mean:", repr(r3['mean'].values))
print("lower_95:", repr(r3['lower_95'].values))
print("upper_95:", repr(r3['upper_95'].values))

np.random.seed(42)
y4 = ar1_series(80, seed=42)
exog_train_1d = pd.Series(np.random.randn(80), name='single_feature')
model4 = Arima(order=(1, 0, 0), seasonal_order=(0, 0, 0))
model4.fit(y4, exog=exog_train_1d)
exog_pred_s = pd.Series(np.random.randn(5))
r4 = model4.predict_interval(steps=5, exog=exog_pred_s, level=(95,))
print("\n=== exog Series ===")
print("mean:", repr(r4['mean'].values))
print("lower_95:", repr(r4['lower_95'].values))
print("upper_95:", repr(r4['upper_95'].values))

# ===========================================================================
# Auto-ARIMA air passengers
# ===========================================================================
model_auto = Arima(
    order=None, seasonal_order=None,
    start_p=0, start_q=0, max_p=3, max_q=3, max_P=2, max_Q=2,
    max_order=5, max_d=2, max_D=1, ic="aic", seasonal=True, test="kpss",
    nmodels=94, optim_method="BFGS", approximation=False,
    optim_kwargs={'maxiter': 5000, 'gtol': 1e-6, 'ftol': 1e-9},
    m=12, trace=False, stepwise=True,
)
model_auto.fit(air_passengers, suppress_warnings=True)
r_auto = model_auto.predict_interval(steps=5, level=(95, 99))
print(f"\n=== Auto-ARIMA air passengers [{platform.system()}] ===")
print("order:", model_auto.best_params_['order'])
print("seasonal_order:", model_auto.best_params_['seasonal_order'])
print("name:", model_auto.estimator_name_)
print("mean:", repr(r_auto['mean'].values))
print("lower_95:", repr(r_auto['lower_95'].values))
print("upper_95:", repr(r_auto['upper_95'].values))
print("lower_99:", repr(r_auto['lower_99'].values))
print("upper_99:", repr(r_auto['upper_99'].values))

# ===========================================================================
# ForecasterStats interval test
# ===========================================================================
from skforecast.recursive._forecaster_stats import ForecasterStats
y_fs = pd.Series(np.random.default_rng(42).normal(size=50))
forecaster = ForecasterStats(estimator=Arima(order=(1, 0, 1), seasonal_order=(0, 0, 0)))
forecaster.fit(y=y_fs)
preds_fs = forecaster.predict_interval(steps=5, alpha=0.05)
print("\n=== ForecasterStats interval ===")
print(repr(preds_fs.values))
