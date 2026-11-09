### h1_grid_refit_false (cold, 4 CPU, 5 reps) RF

spec: `{"kind": "grid", "est": "rf", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"n_estimators": [20, 50], "max_depth": [5, null]}}`  
load before/after: 1.20 0.49 0.17 / 1.44 0.63 0.24

| n_jobs | used | median s | min s | max s | peak MB | fp |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.535 | 1.372 | 1.702 | nan | 96c5b73929 |
| auto | 3 | 3.338 | 3.273 | 3.398 | nan | 96c5b73929 |

Same result in every config: yes

### h1_grid_refit_false (warm, 4 CPU, 5 reps) RF

spec: `{"kind": "grid", "est": "rf", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"n_estimators": [20, 50], "max_depth": [5, null]}}`  
load before/after: 1.44 0.63 0.24 / 1.29 0.66 0.26

| n_jobs | used | median s | min s | max s | peak MB | fp |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.515 | 1.415 | 1.669 | nan | 96c5b73929 |
| auto | 3 | 1.835 | 1.780 | 1.878 | nan | 96c5b73929 |

Same result in every config: yes

### h1_grid_refit_false (cold, 4 CPU, 5 reps) hgb

spec: `{"kind": "grid", "est": "hgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 1.18 0.65 0.26 / 1.94 0.93 0.37

| n_jobs | used | median s | min s | max s | peak MB | fp |
|---|---|---|---|---|---|---|
| 1 | 1 | 2.592 | 2.212 | 2.647 | nan | 65fe60cd64 |
| auto | 3 | 2.822 | 2.714 | 2.850 | nan | 65fe60cd64 |

Same result in every config: yes

### h1_grid_refit_false (cold, 4 CPU, 5 reps) xgb

spec: `{"kind": "grid", "est": "xgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 1.94 0.93 0.37 / 2.15 1.09 0.45

| n_jobs | used | median s | min s | max s | peak MB | fp |
|---|---|---|---|---|---|---|
| 1 | 1 | 0.729 | 0.682 | 0.761 | nan | e218e55061 |
| auto | 3 | 2.748 | 2.621 | 2.760 | nan | e218e55061 |

Same result in every config: yes

### h1_grid_refit_false (warm, 4 CPU, 5 reps) hgb

spec: `{"kind": "grid", "est": "hgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 1.97 1.08 0.44 / 2.67 1.31 0.54

| n_jobs | used | median s | min s | max s | peak MB | fp |
|---|---|---|---|---|---|---|
| 1 | 1 | 2.382 | 2.271 | 2.662 | nan | 65fe60cd64 |
| auto | 3 | 1.491 | 1.399 | 1.512 | nan | 65fe60cd64 |

Same result in every config: yes

### h1_grid_refit_false (cold, 4 CPU, 5 reps) cat

spec: `{"kind": "grid", "est": "cat", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"depth": [4, 6], "iterations": [100, 200]}}`  
load before/after: 2.67 1.31 0.54 / 3.34 1.63 0.67

| n_jobs | used | median s | min s | max s | peak MB | fp |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.276 | 1.159 | 1.351 | nan | a5cd23bd2a |
| auto | 3 | 2.495 | 2.352 | 2.578 | nan | a5cd23bd2a |

Same result in every config: yes

### h1_grid_refit_false (cold, 4 CPU, 5 reps) RF, re-run after merging 0.27.x (2926b38)

spec: `{"kind": "grid", "est": "rf", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"n_estimators": [20, 50], "max_depth": [5, null]}}`  
load before/after: 0.52 3.72 3.76 / 1.06 3.45 3.67

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.700 | 1.507 | 1.849 | nan | - | f555806b88 |
| auto | 3 | 3.474 | 3.333 | 3.595 | nan | - | f555806b88 |

Same result in every config: yes

### h1_grid_refit_false (cold, 4 CPU, 5 reps) HGB, steps=168 (horizonte largo como en la revisión previa), 2 candidatos

spec: `{"kind": "grid", "est": "hgb", "n": 5000, "steps": 168, "n_folds": 10, "refit": false, "lags": 48, "param_grid": {"max_depth": [3, 6]}}`  
load before/after: 0.08 0.94 1.65 / 1.47 1.21 1.71

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.686 | 3.564 | 4.016 | nan | - | ef5dd00005 |
| auto | 3 | 2.797 | 2.627 | 2.901 | nan | - | ef5dd00005 |

Same result in every config: yes

### h1_grid_refit_false (cold, 4 CPU, 5 reps) RF, steps=168, 2 candidatos

spec: `{"kind": "grid", "est": "rf", "n": 5000, "steps": 168, "n_folds": 10, "refit": false, "lags": 48, "param_grid": {"n_estimators": [50, 100]}}`  
load before/after: 1.35 1.19 1.70 / 1.41 1.30 1.68

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 9.789 | 9.343 | 10.111 | nan | - | f45db6047a |
| auto | 3 | 11.292 | 10.955 | 11.565 | nan | - | f45db6047a |

Same result in every config: yes

### h1_grid_refit_false (cold, 4 CPU, 5 reps) LGBM(n_jobs=1)

spec: `{"kind": "grid", "est": "lgbm", "est_params": {"n_jobs": 1}, "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}}`  
load before/after: 0.43 1.02 1.55 / 0.74 1.03 1.54

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.568 | 0.514 | 0.617 | nan | - | 2c5c81df5a |
| auto | 3 | 1.990 | 1.942 | 2.037 | nan | - | 2c5c81df5a |

Same result in every config: yes

### h1_grid_refit_false (warm, 4 CPU, 5 reps) LGBM(n_jobs=1)

spec: `{"kind": "grid", "est": "lgbm", "est_params": {"n_jobs": 1}, "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]}}`  
load before/after: 0.74 1.03 1.54 / 0.78 1.03 1.53

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.486 | 0.445 | 0.573 | nan | - | 2c5c81df5a |
| auto | 3 | 0.731 | 0.665 | 0.761 | nan | - | 2c5c81df5a |

Same result in every config: yes

