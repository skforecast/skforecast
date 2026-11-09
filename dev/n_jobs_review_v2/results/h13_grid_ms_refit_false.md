### h13_grid_ms_refit_false (cold, 4 CPU, 5 reps) LGBM(n_jobs=1), 50 series x 1000, 24 steps, 20 folds, 4 candidates

spec: `{"kind": "grid_ms", "est": "lgbm", "est_params": {"n_jobs": 1}, "n": 1000, "n_series": 50, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}}`  
load before/after: 1.03 0.64 0.26 / 1.58 0.85 0.36

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.269 | 3.161 | 3.400 | nan | - | 008f9c8691 |
| auto | 3 | 4.968 | 4.960 | 5.133 | nan | - | 008f9c8691 |

Same result in every config: yes

### h13_grid_ms_refit_false (cold, 4 CPU, 5 reps) HGB, 50 series x 1000, 24 steps, 20 folds, 4 candidates

spec: `{"kind": "grid_ms", "est": "hgb", "est_params": {}, "n": 1000, "n_series": 50, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 1.58 0.85 0.36 / 2.64 1.31 0.55

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.834 | 4.667 | 4.927 | nan | - | 5e6874c4ac |
| auto | 3 | 5.022 | 4.806 | 5.586 | nan | - | 5e6874c4ac |

Same result in every config: yes

### h13_grid_ms_refit_false (warm, 4 CPU, 5 reps) LGBM(n_jobs=1), 50 series x 1000, 24 steps, 20 folds, 4 candidates

spec: `{"kind": "grid_ms", "est": "lgbm", "est_params": {"n_jobs": 1}, "n": 1000, "n_series": 50, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}}`  
load before/after: 2.59 1.32 0.56 / 1.75 1.27 0.58

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.288 | 3.083 | 3.649 | nan | - | 008f9c8691 |
| auto | 3 | 3.783 | 3.662 | 3.881 | nan | - | 008f9c8691 |

Same result in every config: yes

### h13_grid_ms_refit_false (warm, 4 CPU, 5 reps) HGB, 50 series x 1000, 24 steps, 20 folds, 4 candidates

spec: `{"kind": "grid_ms", "est": "hgb", "est_params": {}, "n": 1000, "n_series": 50, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 1.75 1.27 0.58 / 2.66 1.59 0.73

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.679 | 4.354 | 4.947 | nan | - | 5e6874c4ac |
| auto | 3 | 3.684 | 3.572 | 3.952 | nan | - | 5e6874c4ac |

Same result in every config: yes

### h13_grid_ms_refit_false (cold, 4 CPU, 5 reps) RF, 20 series x 1000, 24 steps, 20 folds, 4 candidates

spec: `{"kind": "grid_ms", "est": "rf", "n": 1000, "n_series": 20, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"n_estimators": [20, 50], "max_depth": [5, null]}}`  
load before/after: 2.15 1.53 0.72 / 1.32 1.39 0.81

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 13.053 | 12.875 | 13.432 | nan | - | 0dcf18794e |
| auto | 3 | 17.343 | 16.895 | 17.594 | nan | - | 0dcf18794e |

Same result in every config: yes

### h13_grid_ms_refit_false (cold, 4 CPU, 5 reps) HGB, 50 series x 3000, 168 steps, 10 folds, 2 candidates

spec: `{"kind": "grid_ms", "est": "hgb", "n": 3000, "n_series": 50, "steps": 168, "n_folds": 10, "refit": false, "lags": 48, "param_grid": {"max_depth": [3, 6]}}`  
load before/after: 1.18 1.36 0.81 / 2.70 1.80 1.01

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 5.628 | 5.582 | 5.760 | nan | - | 3fa17942f3 |
| auto | 3 | 4.583 | 4.304 | 4.808 | nan | - | 3fa17942f3 |

Same result in every config: yes

### h13_grid_ms_refit_false (warm, 4 CPU, 5 reps) HGB, 50 series x 3000, 168 steps, 10 folds, 2 candidates

spec: `{"kind": "grid_ms", "est": "hgb", "n": 3000, "n_series": 50, "steps": 168, "n_folds": 10, "refit": false, "lags": 48, "param_grid": {"max_depth": [3, 6]}}`  
load before/after: 2.70 1.80 1.01 / 2.88 2.01 1.13

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 5.298 | 5.013 | 5.604 | nan | - | 3fa17942f3 |
| auto | 3 | 2.933 | 2.896 | 3.235 | nan | - | 3fa17942f3 |

Same result in every config: yes

