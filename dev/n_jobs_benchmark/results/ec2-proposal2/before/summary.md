# Benchmark of n_jobs in skforecast

- Machine: Linux-7.0.0-1017-aws-x86_64-with-glibc2.43, 16 logical CPUs, 8 physical cores, 130.8 GB.
- Python 3.12.13; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: 09bdbe69f9404bccaae3c7078ca9c3f505aad6a1.
- Measures: 60 finished, 0 failed or timed out.
- Workers: 60 timed measures, 36 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.003 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 16 CPUs

joblib sees 16 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas64_-f48b354e.so', 16], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 16], ['blas', 'libscipy_openblas-5f890258.so', 16], ['openmp', 'libgomp.so.1.0.0', 16]]. First use of the window features in a process: 0.00 s.

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 20.20 | 20.05 | 9.64 | 9.31 | 2.15x | 0.32 | 0.60 | 15 | 6/6 (measured) | 0.95x | 0.73x | 0.54 | 1.00x | 1.9 | 2297 | 103.2 | 54.9 | yes |  |
| lgbm_1t | 2/2 | 16.67 | 16.65 | 8.11 | 7.73 | 2.15x | 0.37 | 0.69 | 15 | 6/6 (measured) | 1.00x | 0.75x | 0.82 | 0.98x | 1.4 | 2334 | 16.7 | 29.6 | yes |  |
| xgb | 2/2 | 10.70 | 10.54 | 12.55 | 12.33 | 0.85x | 0.22 |  | 15 | 6/6 (measured) | 0.94x | 0.74x | 0.81 | 1.00x | 2.0 | 2562 | 166.5 | 81.3 | yes |  |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 60.97 | 61.21 | 29.16 | 22.48 | 2.72x | 6.68 | 10.56 | 15 | 15/15 | 1.00x | 1.00x |  | 1.01x | 1.9 | 2758 | 316.9 | 132.9 | yes |  |
| lgbm_1t | 2/2 | 53.03 | 53.03 | 24.63 | 18.50 | 2.87x | 6.13 | 9.41 | 15 | 15/15 | 1.00x | 1.00x |  | 1.00x | 1.4 | 2805 | 53.1 | 57.0 | yes |  |
| xgb | 2/2 | 32.17 | 32.01 | 37.26 | 31.00 | 1.03x | 6.26 |  | 15 | 15/15 | 1.16x | 1.00x |  | 1.01x | 2.0 | 3129 | 505.4 | 209.9 | yes |  |

## 8 CPUs

joblib sees 8 CPUs (8 physical cores). Thread pools: [['openmp', 'libgomp.so.1.0.0', 8], ['blas', 'libscipy_openblas-5f890258.so', 8], ['blas', 'libscipy_openblas64_-f48b354e.so', 8], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 8]]. First use of the window features in a process: 0.00 s.

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 20.40 | 20.19 | 9.64 | 7.16 | 2.82x | 2.48 | 3.84 | 7 | 6/6 (measured) | 0.99x | 1.00x | 0.53 | 0.99x | 1.9 | 1301 | 103.8 | 44.3 | yes |  |
| lgbm_1t | 2/2 | 16.68 | 16.67 | 8.02 | 5.81 | 2.87x | 2.20 | 3.38 | 7 | 6/6 (measured) | 1.01x | 1.01x | 0.83 | 1.00x | 1.4 | 1320 | 16.7 | 19.9 | yes |  |
| xgb | 2/2 | 13.04 | 12.99 | 12.81 | 10.65 | 1.22x | 2.16 | 11.98 | 7 | 6/6 (measured) | 1.00x | 1.00x | 0.82 | 1.01x | 2.0 | 1459 | 101.4 | 61.2 | yes |  |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 61.79 | 61.64 | 24.94 | 22.55 | 2.73x | 2.39 | 3.76 | 7 | 7/7 | 1.00x | 1.00x |  | 0.98x | 1.9 | 1317 | 317.3 | 132.4 | yes |  |
| lgbm_1t | 2/2 | 52.99 | 53.03 | 20.61 | 18.37 | 2.89x | 2.23 | 3.41 | 7 | 7/7 | 1.00x | 1.00x |  | 0.99x | 1.4 | 1338 | 53.4 | 56.7 | yes |  |
| xgb | 2/2 | 38.86 | 38.74 | 34.36 | 32.13 | 1.21x | 2.24 | 13.11 | 7 | 7/7 | 1.00x | 1.00x |  | 0.99x | 2.0 | 1496 | 302.6 | 177.5 | yes |  |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 16 | first | 6 | 193.7 | 121.3 | 118.3 | 114.4 | 2.15x (gs_refit_small, lgbm_1t) | 1.17x (bt_refit_small, xgb) | 1.16x (gs_refit_small, xgb) |
| 16 | next | 6 | 193.5 | 101.4 | 92.5 | 99.6 | 2.87x (gs_refit_small, lgbm_1t) | 1.17x (bt_refit_small, xgb) | 1.00x (gs_refit_small, hgb) |
| 8 | first | 6 | 203.8 | 110.4 | 110.3 | 110.4 | 2.57x (gs_refit_small, lgbm_1t) | 1.00x (bt_refit_small, hgb) | 1.01x (bt_refit_small, lgbm_1t) |
| 8 | next | 6 | 203.3 | 96.7 | 96.7 | 96.7 | 2.89x (gs_refit_small, lgbm_1t) | 1.00x (bt_refit_small, hgb) | 1.01x (bt_refit_small, lgbm_1t) |
