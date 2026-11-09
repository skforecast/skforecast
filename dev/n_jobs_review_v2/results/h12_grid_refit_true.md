### h12_grid_refit_true (cold, 4 CPU, 5 reps) grid RF cheap 4 candidates, refit=True, 10 folds

spec: `{"kind": "grid", "est": "rf", "n": 2000, "steps": 24, "n_folds": 10, "refit": true, "lags": 24, "param_grid": {"n_estimators": [5, 10], "max_depth": [5, null]}}`  
load before/after: 1.71 1.95 2.22 / 1.58 1.89 2.18

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.788 | 3.655 | 3.916 | nan | - | 689b25bde3 |
| auto | 3 | 3.249 | 3.108 | 3.319 | nan | - | 689b25bde3 |

Same result in every config: yes

### h12_grid_refit_true (cold, 4 CPU, 5 reps) grid XGB 4 candidates, refit=True, 10 folds

spec: `{"kind": "grid", "est": "xgb", "n": 2000, "steps": 24, "n_folds": 10, "refit": true, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 1.45 1.86 2.17 / 2.95 2.21 2.27

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.655 | 4.489 | 4.840 | nan | - | 7f584ca72b |
| auto | 3 | 5.868 | 5.770 | 6.126 | nan | - | 7f584ca72b |

Same result in every config: yes

