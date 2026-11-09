### h9_intervals_norefit (cold, 4 CPU, 5 reps) RF 100, bootstrapping 250, refit=False

spec: `{"kind": "backtest", "est": "rf", "n": 2000, "steps": 24, "n_folds": 20, "refit": false, "lags": 24, "interval": 0.8, "n_boot": 250}`  
load before/after: 0.99 1.65 2.32 / 1.41 1.67 2.30

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| auto | 1 | 2.341 | 2.203 | 2.575 | nan | 0.00 | 40773d2f2a |
| 3 | 3 | 3.562 | 3.369 | 3.725 | nan | - | 40773d2f2a |

Same result in every config: yes

