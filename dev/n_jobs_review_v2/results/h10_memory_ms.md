### h10_memory_ms (cold, 4 CPU, 2 reps) Ridge, 50 series x 10000, 10 folds

spec: `{"kind": "backtest_ms", "est": "ridge", "n": 10000, "n_series": 50, "steps": 24, "n_folds": 10, "refit": true, "fixed_train_size": false, "lags": 24}`  
load before/after: 1.26 1.62 2.27 / 1.24 1.61 2.26

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.269 | 2.101 | 2.436 | 402 | - | 00ca310268 |

Same result in every config: yes

### h10_memory_ms (cold, 4 CPU, 2 reps) Ridge, 50 series x 10000, 80 folds

spec: `{"kind": "backtest_ms", "est": "ridge", "n": 10000, "n_series": 50, "steps": 24, "n_folds": 80, "refit": true, "fixed_train_size": false, "lags": 24}`  
load before/after: 1.24 1.61 2.26 / 1.63 1.66 2.26

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 12.505 | 12.378 | 12.633 | 426 | - | 9aa5ec4029 |

Same result in every config: yes

