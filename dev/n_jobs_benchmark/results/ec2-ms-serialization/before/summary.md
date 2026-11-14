# Benchmark of n_jobs in skforecast

- Machine: Linux-7.0.0-1017-aws-x86_64-with-glibc2.43, 16 logical CPUs, 8 physical cores, 130.8 GB.
- Python 3.12.13; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: aa4ec3be0399552e9a29cef6a20612b142c05eff.
- Measures: 48 finished, 0 failed or timed out.
- Workers: 48 timed measures, 24 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.062 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 16 CPUs

joblib sees 16 CPUs (8 physical cores). Thread pools: [['openmp', 'libgomp.so.1.0.0', 16], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 16], ['blas', 'libscipy_openblas64_-f48b354e.so', 16], ['blas', 'libscipy_openblas-5f890258.so', 16]]. First use of the window features in a process: 0.00 s.

### Multi-series backtesting without refit (50 series, 52 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 4.33 | 4.22 | 8.04 | 5.73 | 0.74x | 2.31 |  | 15 | 1/1 | 1.00x | 1.00x |  | 0.99x | 0.9 | 2579 | 11.4 | 17.3 | yes |  |
| lgbm_1t | 2/2 | 5.14 | 5.07 | 10.47 | 9.08 | 0.56x | 1.39 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.99x | 1.5 | 2581 | 5.1 | 14.1 | yes |  |
| xgb | 2/2 | 3.32 | 3.18 | 6.70 | 8.87 | 0.36x | -2.17 |  | 1 | 1/1 | 1.00x | 1.00x |  | 5.32x | 2.3 | 2804 | 24.0 | 18.7 | yes |  |

### Multi-series backtesting with refit (50 series, 8 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 6.53 | 6.39 | 5.56 | 3.06 | 2.09x | 2.51 | 4.81 | 15 | 8/8 |  |  | 0.78 | 1.00x | 0.5 | 1665 | 40.2 | 23.2 | yes |  |
| lgbm_1t | 2/2 | 23.91 | 23.92 | 9.32 | 7.24 | 3.30x | 2.08 | 2.99 | 15 | 8/8 |  |  | 0.85 | 1.00x | 1.5 | 1636 | 24.0 | 26.5 | yes |  |
| xgb | 2/2 | 9.30 | 9.30 | 7.54 | 10.10 | 0.92x | -2.56 |  | 15 | 8/8 |  |  | 0.82 | 5.05x | 2.3 | 1706 | 136.5 | 61.0 | yes |  |

## 8 CPUs

joblib sees 8 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas-5f890258.so', 8], ['openmp', 'libgomp.so.1.0.0', 8], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 8], ['blas', 'libscipy_openblas64_-f48b354e.so', 8]]. First use of the window features in a process: 0.00 s.

### Multi-series backtesting without refit (50 series, 52 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 4.92 | 4.28 | 7.54 | 4.98 | 0.86x | 2.56 |  | 7 | 1/1 | 1.00x | 1.00x |  | 0.98x | 0.9 | 1232 | 11.6 | 15.5 | yes |  |
| lgbm_1t | 2/2 | 5.18 | 5.08 | 10.59 | 7.79 | 0.65x | 2.80 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.99x | 1.5 | 1234 | 5.1 | 10.5 | yes |  |
| xgb | 2/2 | 3.35 | 3.27 | 6.26 | 6.11 | 0.54x | 0.15 |  | 1 | 1/1 | 1.00x | 1.00x |  | 2.90x | 2.3 | 1339 | 12.5 | 11.9 | yes |  |

### Multi-series backtesting with refit (50 series, 8 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 6.51 | 6.45 | 5.65 | 3.41 | 1.89x | 2.24 | 4.75 | 7 | 7/7 | 1.00x | 1.00x | 0.77 | 1.02x | 0.5 | 1439 | 39.9 | 16.4 | yes |  |
| lgbm_1t | 2/2 | 23.88 | 23.91 | 9.21 | 7.23 | 3.31x | 1.97 | 2.83 | 7 | 7/7 | 1.00x | 1.00x | 0.85 | 1.00x | 1.5 | 1464 | 23.9 | 26.4 | yes |  |
| xgb | 2/2 | 10.22 | 10.26 | 8.41 | 8.21 | 1.25x | 0.20 | 0.99 | 7 | 7/7 | 1.00x | 1.00x | 0.81 | 2.86x | 2.3 | 1516 | 75.4 | 36.8 | yes |  |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 16 | first | 3 | 12.8 | 25.2 | 12.8 | 12.8 | 1.00x (ms_norefit, hgb) | 2.04x (ms_norefit, lgbm_1t) | 1.00x (ms_norefit, hgb) |
| 16 | next | 3 | 12.5 | 23.7 | 12.5 | 12.5 | 1.00x (ms_norefit, hgb) | 2.79x (ms_norefit, xgb) | 1.00x (ms_norefit, hgb) |
| 8 | first | 6 | 54.1 | 47.7 | 36.7 | 36.7 | 2.59x (ms_refit, lgbm_1t) | 2.04x (ms_norefit, lgbm_1t) | 1.00x (ms_norefit, hgb) |
| 8 | next | 6 | 53.3 | 37.7 | 31.5 | 31.5 | 3.31x (ms_refit, lgbm_1t) | 1.87x (ms_norefit, xgb) | 1.00x (ms_norefit, hgb) |
