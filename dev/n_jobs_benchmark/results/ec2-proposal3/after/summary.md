# Benchmark of n_jobs in skforecast

- Machine: Linux-7.0.0-1017-aws-x86_64-with-glibc2.43, 16 logical CPUs, 8 physical cores, 130.8 GB.
- Python 3.12.13; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: b721d085d5a17d867a4849c7546bef886ccc18b2.
- Measures: 50 finished, 0 failed or timed out.
- Workers: 50 timed measures, 30 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.002 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 16 CPUs

joblib sees 16 CPUs (8 physical cores). Thread pools: [['openmp', 'libgomp-e985bcbb.so.1.0.0', 16], ['blas', 'libscipy_openblas-5f890258.so', 16], ['openmp', 'libgomp.so.1.0.0', 16], ['blas', 'libscipy_openblas64_-f48b354e.so', 16]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 3.91 | 3.84 | 5.34 | 2.27 | 1.69x | 3.07 | 7.51 | 15 | 1/1 (measured) | 0.99x | 1.68x |  | 1.00x | 0.6 | 2762 | 59.4 | 35.2 | yes |  half: 4.65/2.28 (n_jobs=8) |
| hgb | 2/2 | 19.44 | 19.39 | 7.89 | 4.86 | 3.99x | 3.03 | 4.04 | 15 | 15/15 (measured) | 1.03x | 1.03x |  | 1.00x | 1.9 | 2630 | 33.7 | 53.3 | yes |  half: 7.56/5.27 (n_jobs=8) |
| lgbm_1t | 2/2 | 3.35 | 3.36 | 6.38 | 3.28 | 1.02x | 3.10 |  | 1 | 1/1 | 1.00x | 1.02x |  | 1.00x | 1.4 | 2581 | 3.4 | 8.0 | yes |  |
| xgb | 2/2 | 2.96 | 2.82 | 5.50 | 2.34 | 1.21x | 3.16 | 18.43 | 1 | 1/1 (measured) | 0.99x | 1.22x |  | 0.99x | 2.0 | 2792 | 43.4 | 34.4 | yes |  half: 4.70/2.34 (n_jobs=8) |

## 8 CPUs

joblib sees 8 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas-5f890258.so', 8], ['blas', 'libscipy_openblas64_-f48b354e.so', 8], ['openmp', 'libgomp.so.1.0.0', 8], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 8]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 4.06 | 3.99 | 5.03 | 2.71 | 1.47x | 2.31 | 7.23 | 7 | 1/1 (measured) | 1.00x | 1.47x |  | 1.00x | 0.6 | 1317 | 31.2 | 21.2 | yes |  half: 5.12/2.82 (n_jobs=4) |
| hgb | 2/2 | 19.26 | 19.17 | 7.80 | 5.62 | 3.41x | 2.18 | 3.08 | 7 | 7/7 (measured) | 1.02x | 1.02x |  | 1.01x | 1.9 | 1256 | 33.3 | 39.2 | yes |  half: 9.89/7.63 (n_jobs=4) |
| lgbm_1t | 2/2 | 3.34 | 3.36 | 5.56 | 3.30 | 1.02x | 2.25 |  | 1 | 1/1 | 1.00x | 1.02x |  | 1.00x | 1.4 | 1232 | 3.4 | 5.8 | yes |  |
| xgb | 2/2 | 3.28 | 3.15 | 4.99 | 2.67 | 1.18x | 2.31 | 15.21 | 1 | 1/1 (measured) | 1.00x | 1.19x |  | 0.99x | 2.0 | 1332 | 23.9 | 20.0 | yes |  half: 5.25/2.90 (n_jobs=4) |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 16 | first | 4 | 29.7 | 25.1 | 18.3 | 18.1 | 2.46x (bt_norefit, hgb) | 1.91x (bt_norefit, lgbm_1t) | 1.03x (bt_norefit, hgb) |
| 16 | next | 4 | 29.4 | 12.8 | 15.0 | 12.8 | 3.99x (bt_norefit, hgb) | 1.00x (bt_norefit, catboost) | 1.68x (bt_norefit, catboost) |
| 8 | first | 4 | 29.9 | 23.4 | 18.7 | 18.5 | 2.47x (bt_norefit, hgb) | 1.66x (bt_norefit, lgbm_1t) | 1.02x (bt_norefit, hgb) |
| 8 | next | 4 | 29.7 | 14.3 | 16.3 | 14.3 | 3.41x (bt_norefit, hgb) | 1.00x (bt_norefit, catboost) | 1.47x (bt_norefit, catboost) |
