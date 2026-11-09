### h7_direct (cold, 4 CPU, 5 reps) Direct rf10, 12 steps, fit+predict

spec: `{"kind": "direct_fit", "est": "rf", "est_params": {"n_estimators": 10}, "n": 2000, "steps": 12, "lags": 24}`  
load before/after: 0.74 1.77 2.70 / 1.12 1.74 2.66

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 2.250 | 1.979 | 2.425 | nan | - | df36c234d1 |
| auto | 3 | 2.228 | 2.137 | 2.366 | nan | - | df36c234d1 |

Same result in every config: yes

### h7_direct (cold, 4 CPU, 5 reps) Direct xgb, 12 steps, fit+predict

spec: `{"kind": "direct_fit", "est": "xgb", "n": 2000, "steps": 12, "lags": 24}`  
load before/after: 1.12 1.74 2.66 / 1.98 1.87 2.66

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1.991 | 1.782 | 2.155 | nan | - | 9aab1a4084 |
| auto | 3 | 2.987 | 2.869 | 3.126 | nan | - | 9aab1a4084 |

Same result in every config: yes

### h7_direct (cold, 4 CPU, 5 reps) Direct rf10 small, 6 steps

spec: `{"kind": "direct_fit", "est": "rf", "est_params": {"n_estimators": 10}, "n": 500, "steps": 6, "lags": 12}`  
load before/after: 1.67 1.81 2.63 / 2.49 2.00 2.67

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.187 | 0.150 | 0.215 | nan | - | ff03aa5e59 |
| auto | 3 | 1.506 | 1.488 | 1.631 | nan | - | ff03aa5e59 |

Same result in every config: yes

### h7_direct (cold, 4 CPU, 5 reps) Direct hgb, 12 steps

spec: `{"kind": "direct_fit", "est": "hgb", "n": 2000, "steps": 12, "lags": 24}`  
load before/after: 2.49 2.00 2.67 / 2.94 2.18 2.70

| n_jobs | used | median s | min s | max s | peak MB | est. work s | fp |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 3.348 | 3.173 | 3.386 | nan | - | b82e204c0e |
| auto | 3 | 2.501 | 2.316 | 2.666 | nan | - | b82e204c0e |

Same result in every config: yes

