### h14_hgb_predict_1thread (warm, 4 CPU, 5 reps) HGB, 50 series x 1000, 24 steps, 20 folds, 4 candidates; 1lim = n_jobs=1 with predict limited to 1 OpenMP thread

spec: `{"kind": "grid_ms", "est": "hgb", "n": 1000, "n_series": 50, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 0.18 0.91 0.87 / 1.66 1.19 0.97

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.642 | 4.459 | 4.927 | nan | - | 5e6874c4ac |
| 1lim | 1 | 4.057 | 4.000 | 4.159 | nan | - | 5e6874c4ac |
| auto | 3 | 3.631 | 3.562 | 3.704 | nan | - | 5e6874c4ac |

Same result in every config: yes

### h14_hgb_predict_1thread (cold, 4 CPU, 5 reps) HGB, 50 series x 1000, 24 steps, 20 folds, 4 candidates; 1lim = n_jobs=1 with predict limited to 1 OpenMP thread

spec: `{"kind": "grid_ms", "est": "hgb", "n": 1000, "n_series": 50, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 1.66 1.19 0.97 / 1.92 1.36 1.05

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.841 | 4.693 | 4.956 | nan | - | 5e6874c4ac |
| 1lim | 1 | 4.066 | 3.973 | 4.219 | nan | - | 5e6874c4ac |
| auto | 3 | 4.992 | 4.918 | 5.280 | nan | - | 5e6874c4ac |

Same result in every config: yes

### h14_hgb_predict_1thread (warm, 4 CPU, 5 reps) HGB, 50 series x 2000, 72 steps, 10 folds, 4 candidates; 1lim = n_jobs=1 with predict limited to 1 OpenMP thread

spec: `{"kind": "grid_ms", "est": "hgb", "n": 2000, "n_series": 50, "steps": 72, "n_folds": 10, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 1.62 1.32 1.04 / 2.89 1.84 1.25

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 5.764 | 5.559 | 5.860 | nan | - | eeca1c6f7d |
| 1lim | 1 | 4.633 | 4.146 | 4.901 | nan | - | eeca1c6f7d |
| auto | 3 | 3.918 | 3.833 | 4.035 | nan | - | eeca1c6f7d |

Same result in every config: yes

### h14_hgb_predict_1thread (cold, 4 CPU, 5 reps) HGB, 50 series x 2000, 72 steps, 10 folds, 4 candidates; 1lim = n_jobs=1 with predict limited to 1 OpenMP thread

spec: `{"kind": "grid_ms", "est": "hgb", "n": 2000, "n_series": 50, "steps": 72, "n_folds": 10, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 2.89 1.84 1.25 / 2.51 2.01 1.38

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 5.496 | 5.278 | 5.623 | nan | - | eeca1c6f7d |
| 1lim | 1 | 4.742 | 4.528 | 4.902 | nan | - | eeca1c6f7d |
| auto | 3 | 5.228 | 5.096 | 5.429 | nan | - | eeca1c6f7d |

Same result in every config: yes

### h14_hgb_predict_1thread (warm, 4 CPU, 5 reps) HGB, single series n=2000, 24 steps, 20 folds, 4 candidates (H1 case); 1lim = n_jobs=1 with predict limited to 1 OpenMP thread

spec: `{"kind": "grid", "est": "hgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 2.31 1.98 1.37 / 2.90 2.16 1.45

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.417 | 2.020 | 2.491 | nan | - | 0dd56d5ba6 |
| 1lim | 1 | 1.933 | 1.833 | 1.977 | nan | - | 0dd56d5ba6 |
| auto | 3 | 1.452 | 1.424 | 1.559 | nan | - | 0dd56d5ba6 |

Same result in every config: yes

### h14_hgb_predict_1thread (cold, 4 CPU, 5 reps) HGB, single series n=2000, 24 steps, 20 folds, 4 candidates (H1 case); 1lim = n_jobs=1 with predict limited to 1 OpenMP thread

spec: `{"kind": "grid", "est": "hgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 2.90 2.16 1.45 / 3.66 2.49 1.61

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.520 | 2.328 | 2.888 | nan | - | 0dd56d5ba6 |
| 1lim | 1 | 2.028 | 1.815 | 2.100 | nan | - | 0dd56d5ba6 |
| auto | 3 | 2.840 | 2.586 | 2.891 | nan | - | 0dd56d5ba6 |

Same result in every config: yes

