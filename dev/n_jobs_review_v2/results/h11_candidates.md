### h11_candidates (cold, 4 CPU, 5 reps) grid_search_forecaster RF 4 candidates refit=False

spec: `{"kind": "grid", "est": "rf", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"n_estimators": [50, 100], "max_depth": [10, null]}}`  
load before/after: 2.07 1.99 2.31 / 1.96 1.97 2.27

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.057 | 3.990 | 4.231 | nan | - | 43e69f4470 |
| auto | 3 | 6.407 | 6.284 | 6.726 | nan | - | 43e69f4470 |

Same result in every config: yes

### h11_candidates (cold, 4 CPU, 5 reps) prototype: candidates in parallel, RF 4 candidates refit=False

spec: `{"kind": "grid_candidates", "est": "rf", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"n_estimators": [50, 100], "max_depth": [10, null]}}`  
load before/after: 1.96 1.97 2.27 / 2.56 2.12 2.31

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 3 |  | 3.400 | 3.089 | 3.477 | nan | - | 43e69f4470 |
| 4 |  | 2.832 | 2.627 | 3.140 | nan | - | 43e69f4470 |

Same result in every config: yes

### h11_candidates (cold, 4 CPU, 5 reps) prototype, RF 12 candidates refit=False (n_jobs=1 = sequential candidates)

spec: `{"kind": "grid_candidates", "est": "rf", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"n_estimators": [20, 50, 100], "max_depth": [5, 10, 20, null]}}`  
load before/after: 2.17 2.05 2.28 / 2.19 2.05 2.26

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 8.431 | 8.112 | 9.046 | nan | - | c4734d5bec |
| 3 |  | 4.869 | 4.667 | 5.236 | nan | - | c4734d5bec |
| 4 |  | 4.277 | 4.001 | 4.478 | nan | - | c4734d5bec |

Same result in every config: yes

