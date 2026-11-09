### h4_threshold (cold, 4 CPU, 5 reps) RF n_estimators=4

spec: `{"kind": "backtest", "est": "rf", "est_params": {"n_estimators": 4}, "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 1.93 1.88 0.97 / 2.31 1.96 1.04

| n_jobs | used | median s | min s | max s | peak MB | fp |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.274 | 1.057 | 1.451 | nan | 8a64b83af9 |
| 3 | 3 | 2.023 | 1.979 | 2.097 | nan | 8a64b83af9 |
| auto | 1 | 1.277 | 1.161 | 1.283 | nan | 8a64b83af9 |

Same result in every config: yes

### h4_threshold (cold, 4 CPU, 5 reps) RF n_estimators=12

spec: `{"kind": "backtest", "est": "rf", "est_params": {"n_estimators": 12}, "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 2.31 1.96 1.04 / 2.11 1.97 1.11

| n_jobs | used | median s | min s | max s | peak MB | fp |
|---|---|---|---|---|---|---|
| 1 | 1 | 3.610 | 3.402 | 3.814 | nan | bd4a8941cd |
| 3 | 3 | 2.962 | 2.849 | 3.121 | nan | bd4a8941cd |
| auto | 3 | 2.899 | 2.871 | 3.085 | nan | bd4a8941cd |

Same result in every config: yes

### h4_threshold (cold, 4 CPU, 5 reps) RF n_estimators=8

spec: `{"kind": "backtest", "est": "rf", "est_params": {"n_estimators": 8}, "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 1.64 1.87 1.09 / 1.95 1.91 1.15

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.399 | 2.382 | 2.615 | nan | - | 1eeeeb8e71 |
| 3 | 3 | 2.415 | 2.325 | 2.432 | nan | - | 1eeeeb8e71 |
| auto | 1,3 | 2.395 | 2.203 | 2.497 | nan | 1.94 | 1eeeeb8e71 |

Same result in every config: yes

### h4_threshold (warm, 4 CPU, 5 reps) RF n_estimators=1

spec: `{"kind": "backtest", "est": "rf", "est_params": {"n_estimators": 1}, "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 1.95 1.91 1.15 / 1.88 1.90 1.15

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.311 | 0.261 | 0.350 | nan | - | 61d5d72857 |
| 3 | 3 | 0.171 | 0.152 | 0.200 | nan | - | 61d5d72857 |
| auto | 1 | 0.364 | 0.335 | 0.378 | nan | 0.32 | 61d5d72857 |

Same result in every config: yes

### h4_threshold (warm, 4 CPU, 5 reps) RF n_estimators=4

spec: `{"kind": "backtest", "est": "rf", "est_params": {"n_estimators": 4}, "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 1.88 1.90 1.15 / 1.63 1.84 1.15

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.207 | 1.127 | 1.262 | nan | - | 8a64b83af9 |
| 3 | 3 | 0.544 | 0.483 | 0.585 | nan | - | 8a64b83af9 |
| auto | 1 | 1.165 | 1.048 | 1.325 | nan | 0.96 | 8a64b83af9 |

Same result in every config: yes

