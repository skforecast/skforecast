### h8_stats (cold, 4 CPU, 1 reps) pilot Arima(2,1,1)(1,1,1)24

spec: `{"kind": "backtest_stats", "est": "arima", "est_params": {"order": [2, 1, 1], "seasonal_order": [1, 1, 1], "m": 24}, "n": 1500, "steps": 24, "n_folds": 10, "refit": true}`  
load before/after: 2.28 2.07 2.66 / 1.78 1.95 2.59

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| auto | 1 | 37.121 | 37.121 | 37.121 | nan | - | 7d5d2c852a |
| 3 | 3 | 11.169 | 11.169 | 11.169 | nan | - | 7d5d2c852a |

Same result in every config: yes

### h8_stats (cold, 4 CPU, 1 reps) pilot Ets(m=24) auto model

spec: `{"kind": "backtest_stats", "est": "ets", "est_params": {"m": 24}, "n": 1500, "steps": 24, "n_folds": 10, "refit": true}`  
load before/after: 1.78 1.95 2.59 / 2.07 2.00 2.59

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| auto | 1 | 8.468 | 8.468 | 8.468 | nan | - | cffbbfe823 |
| 3 | 3 | 4.552 | 4.552 | 4.552 | nan | - | cffbbfe823 |

Same result in every config: yes

### h8_stats (cold, 4 CPU, 5 reps) Ets(m=24) auto model

spec: `{"kind": "backtest_stats", "est": "ets", "est_params": {"m": 24}, "n": 1500, "steps": 24, "n_folds": 10, "refit": true}`  
load before/after: 1.61 1.91 2.55 / 2.11 2.00 2.55

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| auto | 1 | 3.276 | 2.748 | 3.632 | nan | - | cffbbfe823 |
| 3 | 3 | 4.372 | 4.183 | 4.484 | nan | - | cffbbfe823 |

Same result in every config: yes

### h8_stats (cold, 4 CPU, 5 reps) Ets(m=1) small

spec: `{"kind": "backtest_stats", "est": "ets", "est_params": {"m": 1}, "n": 500, "steps": 12, "n_folds": 10, "refit": true}`  
load before/after: 2.11 2.00 2.55 / 2.10 2.00 2.53

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| auto | 1 | 0.599 | 0.523 | 0.709 | nan | - | ccecfd573e |
| 3 | 3 | 2.492 | 2.425 | 2.639 | nan | - | ccecfd573e |

Same result in every config: yes

### h8_stats (cold, 4 CPU, 3 reps) Arima(2,1,1)(1,1,1)24, numba cache warm (3 reps: 37 s per sequential run)

spec: `{"kind": "backtest_stats", "est": "arima", "est_params": {"order": [2, 1, 1], "seasonal_order": [1, 1, 1], "m": 24}, "n": 1500, "steps": 24, "n_folds": 10, "refit": true}`  
load before/after: 1.78 1.93 2.50 / 1.52 1.81 2.39

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| auto | 1 | 18.554 | 18.545 | 19.037 | nan | - | 7d5d2c852a |
| 3 | 3 | 11.918 | 10.596 | 11.921 | nan | - | 7d5d2c852a |

Same result in every config: yes

