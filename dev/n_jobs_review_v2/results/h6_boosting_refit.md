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

