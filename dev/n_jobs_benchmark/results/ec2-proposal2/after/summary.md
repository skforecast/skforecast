# Benchmark of n_jobs in skforecast

- Machine: Linux-7.0.0-1017-aws-x86_64-with-glibc2.43, 16 logical CPUs, 8 physical cores, 130.8 GB.
- Python 3.12.13; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: 09bdbe69f9404bccaae3c7078ca9c3f505aad6a1.
- Measures: 60 finished, 0 failed or timed out.
- Workers: 60 timed measures, 36 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.002 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 16 CPUs

joblib sees 16 CPUs (8 physical cores). Thread pools: [['openmp', 'libgomp.so.1.0.0', 16], ['blas', 'libscipy_openblas64_-f48b354e.so', 16], ['blas', 'libscipy_openblas-5f890258.so', 16], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 16]]. First use of the window features in a process: 0.00 s.

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 20.12 | 20.06 | 9.01 | 6.59 | 3.04x | 2.42 | 3.61 | 15 | 6/6 (measured) | 0.99x | 1.03x | 0.54 | 1.00x | 1.9 | 1130 | 102.9 | 52.7 | yes |  |
| lgbm_1t | 2/2 | 16.73 | 16.71 | 8.07 | 5.82 | 2.87x | 2.24 | 3.44 | 15 | 6/6 (measured) | 1.00x | 1.00x | 0.82 | 1.00x | 1.4 | 1142 | 16.7 | 17.9 | yes |  |
| xgb | 2/2 | 10.66 | 10.68 | 9.87 | 7.79 | 1.37x | 2.08 | 7.67 | 15 | 6/6 (measured) | 1.02x | 1.02x | 0.81 | 1.01x | 2.0 | 1263 | 168.7 | 82.8 | yes |  |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 61.31 | 61.31 | 23.48 | 21.21 | 2.89x | 2.27 | 3.47 | 15 | 15/15 | 1.00x | 1.00x |  | 1.01x | 1.9 | 1135 | 316.5 | 170.6 | yes |  |
| lgbm_1t | 2/2 | 52.60 | 52.84 | 20.69 | 18.44 | 2.86x | 2.25 | 3.46 | 15 | 15/15 | 1.00x | 1.00x |  | 1.00x | 1.4 | 1143 | 53.4 | 57.1 | yes |  |
| xgb | 2/2 | 32.78 | 32.75 | 25.90 | 23.60 | 1.39x | 2.29 | 8.21 | 15 | 15/15 | 1.00x | 1.00x |  | 1.00x | 2.0 | 1272 | 515.9 | 253.0 | yes |  |

## 8 CPUs

joblib sees 8 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas64_-f48b354e.so', 8], ['blas', 'libscipy_openblas-5f890258.so', 8], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 8], ['openmp', 'libgomp.so.1.0.0', 8]]. First use of the window features in a process: 0.00 s.

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 20.29 | 20.28 | 9.56 | 7.18 | 2.82x | 2.37 | 3.68 | 7 | 6/6 (measured) | 1.00x | 1.00x | 0.53 | 0.99x | 1.9 | 1127 | 103.3 | 42.4 | yes |  |
| lgbm_1t | 2/2 | 16.66 | 16.66 | 8.01 | 5.80 | 2.87x | 2.21 | 3.40 | 7 | 6/6 (measured) | 1.00x | 1.00x | 0.82 | 1.00x | 1.4 | 1137 | 16.7 | 17.8 | yes |  |
| xgb | 2/2 | 13.04 | 12.98 | 12.78 | 10.59 | 1.23x | 2.19 | 11.88 | 7 | 6/6 (measured) | 1.00x | 1.00x | 0.82 | 0.98x | 2.0 | 1258 | 101.4 | 58.7 | yes |  |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 2/2 | 61.55 | 61.87 | 24.80 | 22.40 | 2.76x | 2.40 | 3.76 | 7 | 7/7 | 1.00x | 1.00x |  | 0.99x | 1.9 | 1129 | 318.5 | 131.5 | yes |  |
| lgbm_1t | 2/2 | 53.00 | 53.11 | 20.57 | 18.35 | 2.90x | 2.22 | 3.40 | 7 | 7/7 | 1.00x | 1.00x |  | 0.99x | 1.4 | 1140 | 53.2 | 56.6 | yes |  |
| xgb | 2/2 | 39.10 | 38.97 | 34.29 | 32.17 | 1.21x | 2.11 | 12.11 | 7 | 7/7 | 1.00x | 1.00x |  | 1.00x | 2.0 | 1273 | 304.5 | 177.8 | yes |  |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 16 | first | 6 | 194.2 | 97.0 | 97.1 | 97.0 | 2.61x (gs_refit_small, hgb) | 1.00x (bt_refit_small, hgb) | 1.02x (bt_refit_small, xgb) |
| 16 | next | 6 | 194.4 | 83.5 | 83.8 | 83.5 | 3.04x (bt_refit_small, hgb) | 1.00x (bt_refit_small, hgb) | 1.03x (bt_refit_small, hgb) |
| 8 | first | 6 | 203.6 | 110.0 | 110.0 | 110.0 | 2.58x (gs_refit_small, lgbm_1t) | 1.00x (bt_refit_small, hgb) | 1.00x (bt_refit_small, lgbm_1t) |
| 8 | next | 6 | 203.9 | 96.5 | 96.5 | 96.5 | 2.90x (gs_refit_small, lgbm_1t) | 1.00x (bt_refit_small, hgb) | 1.00x (bt_refit_small, hgb) |
