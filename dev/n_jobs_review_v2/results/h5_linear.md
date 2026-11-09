### h5_linear (cold, 4 CPU, 1 reps) pilot ridge

spec: `{"kind": "backtest", "est": "ridge", "n": 20000, "steps": 24, "n_folds": 50, "refit": true, "lags": 48}`  
load before/after: 0.86 1.53 1.19 / 0.87 1.52 1.19

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.620 | 0.620 | 0.620 | nan | - | a9d7a05755 |
| 3 | 3 | 1.819 | 1.819 | 1.819 | nan | - | a9d7a05755 |
| auto | 1 | 0.553 | 0.553 | 0.553 | nan | 1.06 | a9d7a05755 |

Same result in every config: yes

### h5_linear (cold, 4 CPU, 1 reps) pilot pipe_ridge big

spec: `{"kind": "backtest", "est": "pipe_ridge", "n": 50000, "steps": 24, "n_folds": 100, "refit": true, "lags": 168, "n_exog": 5}`  
load before/after: 0.74 1.47 1.18 / 2.65 1.90 1.34

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 22.694 | 22.694 | 22.694 | 407 | - | 719afc5b00 |
| 3 | 3 | 10.716 | 10.716 | 10.716 | 1486 | - | 67e504c2d3 |
| auto | 1 | 22.081 | 22.081 | 22.081 | 407 | 26.07 | 719afc5b00 |

Same result in every config: NO {'67e504c2d3', '719afc5b00'}

### h5_linear (cold, 4 CPU, 5 reps) pipe_ridge 50k x 168 lags, 10 folds

spec: `{"kind": "backtest", "est": "pipe_ridge", "n": 50000, "steps": 24, "n_folds": 10, "refit": true, "lags": 168, "n_exog": 5}`  
load before/after: 1.65 1.74 1.31 / 1.82 1.78 1.35

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.326 | 2.211 | 2.610 | nan | - | 770d1a29a2 |
| 3 | 3 | 2.751 | 2.642 | 2.869 | nan | - | 770d1a29a2 |
| auto | 1 | 2.267 | 2.185 | 2.446 | nan | 2.23 | 770d1a29a2 |

Same result in every config: yes

### h5_linear (cold, 4 CPU, 5 reps) pipe_ridge 50k x 168 lags, 20 folds

spec: `{"kind": "backtest", "est": "pipe_ridge", "n": 50000, "steps": 24, "n_folds": 20, "refit": true, "lags": 168, "n_exog": 5}`  
load before/after: 1.68 1.75 1.35 / 2.99 2.10 1.49

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.798 | 4.296 | 4.935 | nan | - | 24bad345cf |
| 3 | 3 | 4.077 | 3.992 | 4.924 | nan | - | 24bad345cf |

Same result in every config: yes

### h5_linear (cold, 4 CPU, 5 reps) pipe_ridge 50k x 168 lags, 10 folds, re-run after merge (2926b38)

spec: `{"kind": "backtest", "est": "pipe_ridge", "n": 50000, "steps": 24, "n_folds": 10, "refit": true, "lags": 168, "n_exog": 5}`  
load before/after: 0.97 3.40 3.65 / 1.46 3.23 3.58

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.384 | 2.178 | 2.668 | nan | - | 770d1a29a2 |
| 3 | 3 | 2.811 | 2.785 | 3.288 | nan | - | 770d1a29a2 |

Same result in every config: yes

### h5_linear (cold, 4 CPU, 5 reps) pipe_ridge 50k x 168 lags, 20 folds, re-run after merge (2926b38)

spec: `{"kind": "backtest", "est": "pipe_ridge", "n": 50000, "steps": 24, "n_folds": 20, "refit": true, "lags": 168, "n_exog": 5}`  
load before/after: 1.46 3.23 3.58 / 2.24 3.12 3.52

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.573 | 4.360 | 4.819 | nan | - | 24bad345cf |
| 3 | 3 | 3.636 | 3.416 | 3.964 | nan | - | 24bad345cf |

Same result in every config: yes

### h5_linear (cold, 4 CPU, 5 reps) multiseries Ridge, 50 series x 10000, 80 folds

spec: `{"kind": "backtest_ms", "est": "ridge", "n": 10000, "n_series": 50, "steps": 24, "n_folds": 80, "refit": true, "fixed_train_size": false, "lags": 24}`  
load before/after: 1.50 1.63 2.25 / 2.90 2.13 2.36

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 12.085 | 11.090 | 12.677 | 427 | - | 9aa5ec4029 |
| 3 | 3 | 8.649 | 8.435 | 8.960 | 1473 | - | 9aa5ec4029 |

Same result in every config: yes

