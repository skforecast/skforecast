### h6_boosting_refit (cold, 4 CPU, 5 reps) hgb n=2000

spec: `{"kind": "backtest", "est": "hgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 1.34 1.78 1.42 / 2.22 1.96 1.50

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 6.018 | 5.591 | 6.190 | nan | - | 45056e5c6c |
| auto | 3 | 3.421 | 3.291 | 3.525 | nan | 5.70 | 45056e5c6c |

Same result in every config: yes

### h6_boosting_refit (cold, 4 CPU, 5 reps) xgb n=2000

spec: `{"kind": "backtest", "est": "xgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 2.22 1.96 1.50 / 2.94 2.20 1.61

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.126 | 3.082 | 3.676 | nan | - | b0c15525d9 |
| auto | 3 | 3.939 | 3.653 | 4.096 | nan | 2.90 | b0c15525d9 |

Same result in every config: yes

### h6_boosting_refit (cold, 4 CPU, 5 reps) cat 300 it n=2000

spec: `{"kind": "backtest", "est": "cat", "est_params": {"iterations": 300}, "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 2.70 2.16 1.60 / 6.55 3.55 2.16

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 9.312 | 8.791 | 9.559 | nan | - | 47e44c6870 |
| auto | 3 | 8.227 | 7.806 | 8.319 | nan | 9.44 | 47e44c6870 |

Same result in every config: yes

### h6_boosting_refit (cold, 4 CPU, 5 reps) xgb n=10000 lags 48

spec: `{"kind": "backtest", "est": "xgb", "n": 10000, "steps": 24, "n_folds": 10, "refit": true, "lags": 48}`  
load before/after: 2.06 3.06 3.50 / 2.64 3.04 3.46

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.453 | 3.187 | 3.703 | nan | - | 7821c9cf2c |
| auto | 3 | 4.310 | 4.042 | 4.526 | nan | 3.09 | 7821c9cf2c |

Same result in every config: yes

### h6_boosting_refit (cold, 4 CPU, 5 reps) hgb n=10000 lags 48

spec: `{"kind": "backtest", "est": "hgb", "n": 10000, "steps": 24, "n_folds": 10, "refit": true, "lags": 48}`  
load before/after: 2.64 3.04 3.46 / 3.27 3.13 3.47

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.560 | 4.391 | 4.942 | nan | - | e54b95c16f |
| auto | 3 | 3.755 | 3.427 | 4.055 | nan | 4.14 | e54b95c16f |

Same result in every config: yes

### h6_boosting_refit (warm, 4 CPU, 5 reps) xgb n=2000 warm

spec: `{"kind": "backtest", "est": "xgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 2.76 3.03 3.43 / 2.61 2.93 3.37

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 8.345 | 8.036 | 8.673 | nan | - | b0c15525d9 |
| auto | 3 | 2.773 | 2.684 | 2.933 | nan | 7.20 | b0c15525d9 |

Same result in every config: yes

### h6_boosting_refit (cold, 4 CPU, 5 reps) cat 300 it n=10000 lags 48

spec: `{"kind": "backtest", "est": "cat", "est_params": {"iterations": 300}, "n": 10000, "steps": 24, "n_folds": 10, "refit": true, "lags": 48}`  
load before/after: 2.61 2.93 3.37 / 6.03 3.97 3.69

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 11.000 | 10.750 | 11.261 | nan | - | a0e5cd5099 |
| auto | 3 | 9.966 | 9.703 | 10.288 | nan | 10.04 | a0e5cd5099 |

Same result in every config: yes

