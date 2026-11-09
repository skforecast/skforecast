### h3_cpu_minus_one (cold, 4 CPU, 5 reps) RF 25 trees

spec: `{"kind": "backtest", "est": "rf", "est_params": {"n_estimators": 25}, "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 1.23 1.73 1.13 / 2.08 1.86 1.23

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 7.345 | 6.991 | 7.392 | nan | - | b8432825fe |
| auto | 3 | 4.440 | 4.312 | 4.594 | nan | 6.04 | b8432825fe |
| 4 | 4 | 3.896 | 3.748 | 3.975 | nan | - | b8432825fe |

Same result in every config: yes

### h3_cpu_minus_one (cold, 2 CPU, 5 reps) RF 25 trees

spec: `{"kind": "backtest", "est": "rf", "est_params": {"n_estimators": 25}, "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 1.91 1.83 1.23 / 1.55 1.72 1.24

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| auto | 1 | 7.438 | 7.170 | 7.757 | nan | 5.50 | b8432825fe |
| 2 | 2 | 5.782 | 5.479 | 6.198 | nan | - | b8432825fe |

Same result in every config: yes

