# Benchmark of n_jobs in skforecast

- Machine: Linux-7.0.0-1017-aws-x86_64-with-glibc2.43, 16 logical CPUs, 8 physical cores, 130.8 GB.
- Python 3.12.13; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: aa4ec3be0399552e9a29cef6a20612b142c05eff.
- Measures: 48 finished, 0 failed or timed out.
- Workers: 48 timed measures, 24 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.018 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 16 CPUs

joblib sees 16 CPUs (8 physical cores). Thread pools: [['openmp', 'libgomp.so.1.0.0', 16], ['blas', 'libscipy_openblas64_-f48b354e.so', 16], ['blas', 'libscipy_openblas-5f890258.so', 16], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 16]]. First use of the window features in a process: 0.00 s.

### Multi-series backtesting without refit (50 series, 52 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 4.36 | 4.28 | 6.37 | 3.21 | 1.33x | 3.16 | 12.60 | 15 | 1/1 | 1.00x | 1.33x |  | 1.00x | 0.9 | 2552 | 11.4 | 15.1 | yes |  |
| lgbm_1t | 2/2 | 5.17 | 5.05 | 7.91 | 4.84 | 1.04x | 3.08 |  | 1 | 1/1 | 1.00x | 1.04x |  | 0.99x | 1.5 | 2573 | 5.1 | 9.0 | yes |  |
| xgb | 2/2 | 3.31 | 3.19 | 6.27 | 3.20 | 1.00x | 3.07 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.01x | 2.3 | 2813 | 24.1 | 22.6 | yes |  |

### Multi-series backtesting with refit (50 series, 8 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 6.52 | 6.40 | 5.19 | 2.98 | 2.15x | 2.21 | 4.14 | 15 | 8/8 |  |  | 0.78 | 0.99x | 0.5 | 1658 | 40.3 | 23.3 | yes |  |
| lgbm_1t | 2/2 | 23.93 | 23.91 | 8.89 | 6.64 | 3.60x | 2.25 | 3.12 | 15 | 8/8 |  |  | 0.85 | 0.97x | 1.5 | 1637 | 24.0 | 26.1 | yes |  |
| xgb | 2/2 | 9.39 | 9.53 | 7.52 | 5.35 | 1.78x | 2.17 | 4.95 | 15 | 8/8 |  |  | 0.81 | 1.00x | 2.3 | 1719 | 140.1 | 66.8 | yes |  |

## 8 CPUs

joblib sees 8 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas-5f890258.so', 8], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 8], ['openmp', 'libgomp.so.1.0.0', 8], ['blas', 'libscipy_openblas64_-f48b354e.so', 8]]. First use of the window features in a process: 0.00 s.

### Multi-series backtesting without refit (50 series, 52 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 4.31 | 4.21 | 5.57 | 3.21 | 1.31x | 2.36 | 9.98 | 7 | 1/1 | 1.00x | 1.31x |  | 1.06x | 0.9 | 1219 | 11.3 | 12.6 | yes |  |
| lgbm_1t | 2/2 | 5.15 | 5.07 | 7.13 | 4.85 | 1.05x | 2.28 |  | 1 | 1/1 | 1.00x | 1.05x |  | 0.99x | 1.5 | 1231 | 5.1 | 7.0 | yes |  |
| xgb | 2/2 | 3.36 | 3.33 | 5.53 | 3.22 | 1.03x | 2.30 |  | 1 | 1/1 | 1.00x | 1.03x |  | 0.99x | 2.3 | 1351 | 13.0 | 13.0 | yes |  |

### Multi-series backtesting with refit (50 series, 8 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 6.52 | 6.51 | 5.38 | 3.19 | 2.04x | 2.19 | 4.30 | 7 | 7/7 | 1.00x | 1.00x | 0.77 | 1.01x | 0.5 | 1435 | 40.1 | 16.3 | yes |  |
| lgbm_1t | 2/2 | 23.90 | 23.91 | 8.84 | 6.63 | 3.61x | 2.21 | 3.06 | 7 | 7/7 | 1.00x | 1.00x | 0.86 | 0.99x | 1.5 | 1463 | 24.5 | 25.9 | yes |  |
| xgb | 2/2 | 10.14 | 10.05 | 8.18 | 5.91 | 1.70x | 2.27 | 5.51 | 7 | 7/7 | 1.00x | 1.00x | 0.84 | 1.04x | 2.3 | 1530 | 73.8 | 38.4 | yes |  |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 16 | first | 3 | 12.8 | 20.6 | 12.8 | 12.8 | 1.00x (ms_norefit, hgb) | 1.90x (ms_norefit, xgb) | 1.00x (ms_norefit, hgb) |
| 16 | next | 3 | 12.5 | 11.3 | 12.5 | 11.2 | 1.33x (ms_norefit, hgb) | 1.00x (ms_norefit, xgb) | 1.33x (ms_norefit, hgb) |
| 8 | first | 6 | 53.4 | 40.6 | 35.2 | 35.2 | 2.71x (ms_refit, lgbm_1t) | 1.65x (ms_norefit, xgb) | 1.00x (ms_norefit, hgb) |
| 8 | next | 6 | 53.1 | 27.0 | 28.3 | 27.0 | 3.61x (ms_refit, lgbm_1t) | 1.00x (ms_norefit, hgb) | 1.31x (ms_norefit, hgb) |
