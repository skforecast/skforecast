# Benchmark of n_jobs in skforecast

- Machine: Linux-7.0.0-1017-aws-x86_64-with-glibc2.43, 16 logical CPUs, 8 physical cores, 130.8 GB.
- Python 3.12.13; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: b721d085d5a17d867a4849c7546bef886ccc18b2.
- Measures: 50 finished, 0 failed or timed out.
- Workers: 50 timed measures, 28 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.063 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 16 CPUs

joblib sees 16 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas-5f890258.so', 16], ['openmp', 'libgomp.so.1.0.0', 16], ['blas', 'libscipy_openblas64_-f48b354e.so', 16], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 16]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 3.88 | 3.81 | 5.35 | 2.25 | 1.69x | 3.10 | 7.57 | 15 | 1/1 (measured) | 1.00x | 1.70x |  | 1.00x | 0.6 | 2766 | 59.1 | 34.8 | yes |  half: 4.62/2.29 (n_jobs=8) |
| hgb | 2/2 | 19.32 | 19.08 | 7.90 | 4.87 | 3.92x | 3.04 | 4.08 | 15 | 1/1 (measured) | 2.42x | 3.92x |  | 1.01x | 1.9 | 2629 | 33.2 | 53.4 | yes |  half: 7.61/5.30 (n_jobs=8) |
| lgbm_1t | 2/2 | 3.35 | 3.36 | 6.37 | 3.28 | 1.02x | 3.09 |  | 1 | 1/1 | 1.00x | 1.02x |  | 1.00x | 1.4 | 2584 | 3.4 | 8.1 | yes |  |
| xgb | 2/2 | 2.94 | 2.83 | 5.47 | 2.37 | 1.20x | 3.10 | 18.96 | 1 | 1/1 (measured) | 1.00x | 1.21x |  | 0.98x | 2.0 | 2795 | 43.5 | 34.6 | yes |  half: 4.77/2.43 (n_jobs=8) |

## 8 CPUs

joblib sees 8 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas64_-f48b354e.so', 8], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 8], ['openmp', 'libgomp.so.1.0.0', 8], ['blas', 'libscipy_openblas-5f890258.so', 8]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 4.06 | 3.99 | 5.02 | 2.73 | 1.47x | 2.29 | 7.22 | 7 | 1/1 (measured) | 1.00x | 1.47x |  | 1.00x | 0.6 | 1317 | 31.3 | 21.3 | yes |  half: 5.14/2.84 (n_jobs=4) |
| hgb | 2/2 | 19.37 | 19.56 | 7.85 | 5.57 | 3.51x | 2.28 | 3.19 | 7 | 1/1 (measured) | 2.59x | 3.66x |  | 1.02x | 1.9 | 1256 | 33.7 | 38.9 | yes |  half: 9.86/7.69 (n_jobs=4) |
| lgbm_1t | 2/2 | 3.34 | 3.35 | 5.55 | 3.29 | 1.02x | 2.26 |  | 1 | 1/1 | 1.00x | 1.02x |  | 1.00x | 1.4 | 1236 | 3.4 | 5.8 | yes |  |
| xgb | 2/2 | 3.30 | 3.20 | 5.00 | 2.68 | 1.19x | 2.32 | 14.45 | 1 | 1/1 (measured) | 1.00x | 1.19x |  | 1.00x | 2.0 | 1332 | 24.2 | 20.0 | yes |  half: 5.19/2.91 (n_jobs=4) |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 16 | first | 4 | 29.5 | 25.1 | 29.3 | 18.1 | 2.44x (bt_norefit, hgb) | 1.90x (bt_norefit, lgbm_1t) | 2.42x (bt_norefit, hgb) |
| 16 | next | 4 | 29.1 | 12.8 | 29.1 | 12.8 | 3.92x (bt_norefit, hgb) | 1.00x (bt_norefit, catboost) | 3.92x (bt_norefit, hgb) |
| 8 | first | 4 | 30.1 | 23.4 | 31.0 | 18.6 | 2.47x (bt_norefit, hgb) | 1.66x (bt_norefit, lgbm_1t) | 2.59x (bt_norefit, hgb) |
| 8 | next | 4 | 30.1 | 14.3 | 30.9 | 14.3 | 3.51x (bt_norefit, hgb) | 1.00x (bt_norefit, catboost) | 3.66x (bt_norefit, hgb) |
