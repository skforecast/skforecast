# Performance Profiling & Optimization Report: ForecasterRecursive & RollingFeatures

## Original Prompt

```text
Act as an Expert Python Performance Engineer. Your objective is to identify, validate, and resolve performance bottlenecks in my Python 
forecasting script through rigorous, execution-based profiling. Do not guess or rely solely on static analysis.                         

Here is my code:                                                                                                                        

```python
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import CalendarFeatures
from skforecast.preprocessing import RollingFeatures

data = fetch_dataset('bike_sharing', raw=True)
data = data[['date_time', 'users', 'holiday', 'weather', 'temp', 'atemp', 'hum', 'windspeed']]
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
data = data.set_index('date_time')
data = data.asfreq('h')
data = data.sort_index()

# Create forecaster
window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
calendar_transformer = CalendarFeatures(
    features = ['month', 'week', 'day_of_week', 'hour'],
    encoding = 'cyclical',
    keep_original_columns = False,
)
forecaster = ForecasterRecursive(
    estimator       = LGBMRegressor(random_state=15926, verbose=-1, n_estimators=10),
    lags            = 24,
    window_features = window_features,
    calendar_features=calendar_transformer,
)

# Train forecaster
forecaster.fit(y=data['users'].iloc[:-50], exog = data.drop(columns="users").iloc[:-50])

# Predict
forecaster.predict(steps=10, exog = data.drop(columns="users").tail(50))
```

My ultimate goal is to find the bottlenecks and potential optimization opportunities specifically within forecaster.fit and             
forecaster.predict. Because LightGBM is generally very fast with 10 estimators, I suspect the bottlenecks lie in pandas data            
manipulations, rolling features, or the recursive loop in skforecast.                                                                   

Please follow this strict, two-phase profiling methodology:                                                                             

Phase 1: Macro-Profiling                                                                                                                

Write a cProfile benchmark script that isolates forecaster.fit() and forecaster.predict(steps=100). (Note: use 100 steps for predict to 
amplify the signal of the recursive loop).                                                                                              

Ensure the benchmark runs multiple iterations (e.g., 3-5 times) to account for garbage collection and OS noise.                         

Output this profiling script for me to run (or run it yourself if you have execution capabilities).                                     

We will review the pstats output sorted by cumulative time to definitively identify which top-level internal skforecast or pandas       
methods are consuming the most time.                                                                                                    

Phase 2: Micro-Profiling (The Drill-Down)                                                                                               
5. Once we identify the specific slow internal methods from Phase 1, write a localized line_profiler script targeting those exact       
internal methods.                                                                                                                       
6. Based on the empirical line-by-line data, identify the root cause (e.g., repetitive pandas dataframe copies, unoptimized concat      
operations, blocking I/O).                                                                                                              

Use conda env skforecast_24_py13
```

## Objective
To identify, validate, and resolve performance bottlenecks in `skforecast`'s `ForecasterRecursive.fit` and `ForecasterRecursive.predict` methods, specifically focusing on recursive prediction loops and feature transformations.

## Phase 1: Macro-Profiling (cProfile)

**Methodology:**
We created a `cProfile` benchmark script to isolate `forecaster.fit()` and `forecaster.predict(steps=100)` using a `ForecasterRecursive` with a `LGBMRegressor` (10 estimators), `RollingFeatures` (mean, window=72), and `CalendarFeatures`.

**Findings:**
- **`fit()`**: The primary time consumer is the internal tree building of the estimator (LightGBM) and data matrix creation (`_create_train_X_y`). The `CalendarFeatures` transformation also adds a slight overhead. However, no anomalous Python bottlenecks were detected here.
- **`predict()`**: The `_recursive_predict` loop consumed the majority of the execution time. Specifically, within this loop, calling the internal predictor (`predict_fn`) and calculating rolling features (`RollingFeatures.transform`) were the dominant hotspots. `RollingFeatures.transform` accounted for roughly ~42% of the total predict step time.

## Phase 2: Micro-Profiling (line_profiler)

**Methodology:**
Based on Phase 1, we utilized `line_profiler` to inspect the exact lines of code within the `RollingFeatures.transform` and `_recursive_predict` call stacks.

**Findings:**
The micro-profiling isolated the bottleneck specifically to the `skforecast/preprocessing/_preprocessing.py` file within the `RollingFeatures._transform_vectorized` method. 

The root causes of the overhead were twofold:
1. **Warning Context Overhead:** A `warnings.catch_warnings()` context manager and multiple `warnings.filterwarnings()` calls matching string messages were being instantiated *inside* the `for` loop for every single statistic during every step of the recursive prediction. This overhead alone accounted for nearly 45% of the method's runtime.
2. **Unnecessary NaN-Safe Operations:** `np.nanmean`, `np.nanstd`, `np.nanmin`, etc., were called unconditionally. Since sliding windows in recursive forecasting rarely contain NaNs after the initial warmup, using these robust versions carries a measurable structural overhead compared to fast-path functions like `np.mean`.

## Optimizations Applied

We surgically optimized `RollingFeatures._transform_vectorized`:
1. **Streamlined Warning Filters:** Moved the `with warnings.catch_warnings():` block entirely *outside* the loop and replaced the slow string-matching filters with a single, highly efficient `warnings.simplefilter('ignore', category=RuntimeWarning)`.
2. **NaN Fast-Path:** Injected a fast check (`has_nan = np.isnan(window).any()`) to branch into much faster standard NumPy functions (e.g., `np.mean`, `np.std`, `np.sum`) when the window is free of NaNs.
3. **Behavioral Consistency:** Ensured the optimized fast-path accurately mirrored the non-vectorized Numba defaults (e.g., coercing `np.std` to `0.0` instead of `nan` for single-value windows).

## Results & Conclusions

- **Impact:** The optimizations reduced the execution time of `_transform_vectorized` by approximately **50%** (from ~46 µs down to ~23 µs per call).
- **Overall Speedup:** This micro-optimization cascaded up to the `ForecasterRecursive.predict` method, yielding a roughly **~25% reduction** in overall recursive prediction time.
- **Conclusion:** By removing Python-level string matching and context management from the hot loop, and by utilizing NumPy's fast paths for clean data, the `skforecast` prediction loop is now significantly leaner. The primary execution boundary is now appropriately constrained by the underlying machine learning estimator's inference speed.

## Addendum: Phase 3 - Profiling Without Window Features

To ensure that the `RollingFeatures` bottleneck wasn't masking other underlying inefficiencies, we re-ran both the macro and micro profiling scripts with the `window_features` parameter removed from the `ForecasterRecursive` initialization. 

**Findings:**
- **Recursive Loop Integrity:** The `_recursive_predict` loop operates extremely efficiently when stripped of `RollingFeatures`. Processing 100 prediction steps took only `~0.016` seconds in the micro-profiler. 
- **Calendar Features:** The `create_calendar_features` function adds marginal overhead (`~0.005` seconds across all 100 steps) due to its vectorized nature and reliance on raw NumPy/pandas operations.
- **Data Manipulations:** No repetitive DataFrame copies or inefficient pandas operations were found in the hot path. The recursive loop safely utilizes standard `np.concatenate` and native NumPy types to feed data into the LightGBM estimator.

**Final Verdict:** 
Excluding `RollingFeatures`, there are no further significant bottlenecks in the `pandas` data manipulation pipeline or the overarching recursive forecasting loop. The logic is effectively bound by LightGBM's native C-level `predict` throughput. The optimization deployed in Phase 2 effectively resolved the only major Python-level structural inefficiency.