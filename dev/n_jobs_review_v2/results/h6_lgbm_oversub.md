### h6_lgbm_oversub (cold, 4 CPU, 1 reps) lgbm n_jobs=None, 4 folds pilot

spec: `{"kind": "backtest", "est": "lgbm", "n": 2000, "steps": 24, "n_folds": 4, "refit": true, "lags": 24}`  
load before/after: 1.21 4.55 3.29 / 6.18 5.39 3.65

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.550 | 0.550 | 0.550 | nan | - | 7b8f7309fc |
| 3 | 3 | 52.843 | 52.843 | 52.843 | nan | - | 7b8f7309fc |

Same result in every config: yes

### h6_lgbm_oversub (cold, 4 CPU, 1 reps) TransformedTargetRegressor(LGBM), 20 folds pilot

spec: `{"kind": "backtest", "est": "ttr_lgbm", "n": 2000, "steps": 24, "n_folds": 20, "refit": true, "lags": 24}`  
load before/after: 4.42 5.04 3.57 / 4.39 5.02 3.58

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.104 | 2.104 | 2.104 | nan | - | 7488bb5f77 |
| auto | 1 | 1.922 | 1.922 | 1.922 | nan | 1.59 | 7488bb5f77 |

Same result in every config: yes

### h6_lgbm_oversub (cold, 4 CPU, 3 reps) TTR(LGBM 400 trees), 10 folds, n_jobs=1 only ('auto' > 155 s timeout)

spec: `{"kind": "backtest", "est": "ttr_lgbm", "est_params": {"n_estimators": 400}, "n": 2000, "steps": 24, "n_folds": 10, "refit": true, "lags": 24}`  
load before/after: 1.39 4.76 4.07 / 1.54 4.63 4.04

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.533 | 3.445 | 3.694 | nan | - | c3850f55e6 |

Same result in every config: yes

