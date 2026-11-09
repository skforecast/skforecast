### h15_after_p2_p5 (warm, 4 CPU, 5 reps) Package with P2 + P5. HGB, single series n=2000, 24 steps, 20 folds, 4 candidates (H1/H14 case); 3 = parallel folds as 'auto' did before P2

spec: `{"kind": "grid", "est": "hgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 0.08 0.04 0.00 / 0.99 0.26 0.08

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.002 | 1.649 | 2.042 | nan | - | 0dd56d5ba6 |
| 3 | 3 | 1.493 | 1.415 | 1.527 | nan | - | 0dd56d5ba6 |
| auto | 1 | 1.854 | 1.795 | 2.008 | nan | - | 0dd56d5ba6 |

Same result in every config: yes

### h15_after_p2_p5 (cold, 4 CPU, 5 reps) Package with P2 + P5. HGB, single series n=2000, 24 steps, 20 folds, 4 candidates (H1/H14 case); 3 = parallel folds as 'auto' did before P2

spec: `{"kind": "grid", "est": "hgb", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "param_grid": {"max_depth": [3, 6], "learning_rate": [0.05, 0.1]}}`  
load before/after: 0.99 0.26 0.08 / 2.14 0.68 0.23

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.143 | 1.888 | 2.205 | nan | - | 0dd56d5ba6 |
| 3 | 3 | 2.908 | 2.840 | 3.167 | nan | - | 0dd56d5ba6 |
| auto | 1 | 1.887 | 1.857 | 2.109 | nan | - | 0dd56d5ba6 |

Same result in every config: yes

