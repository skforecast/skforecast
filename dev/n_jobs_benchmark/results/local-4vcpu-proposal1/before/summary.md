# Benchmark of n_jobs in skforecast

- Machine: Linux-6.18.44-fc-v445-x86_64-with-glibc2.39, 4 logical CPUs, 4 physical cores, 16.5 GB.
- Python 3.12.3; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: c3138f334ad8ab74f3cd2f7d60ba14c57536fcce.
- Measures: 66 finished, 0 failed or timed out.
- Workers: 66 timed measures, 39 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.010 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 4 CPUs

joblib sees 4 CPUs (4 physical cores). Thread pools: [['blas', 'libscipy_openblas64_-f48b354e.so', 4], ['openmp', 'libgomp.so.1.0.0', 4], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 4], ['blas', 'libscipy_openblas-5f890258.so', 4]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 11.55 | 12.15 | 9.67 | 6.82 | 1.78x | 2.85 | 6.50 | 3 | 1/1 (measured) | 1.18x | 1.76x |  | 0.98x | 1.9 | 666 | 17.8 | 23.8 | yes |  half: 11.30/9.49 (n_jobs=2) |
| lgbm_1t | 1/1 | 2.35 | 2.18 | 7.88 | 4.00 | 0.55x | 3.88 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.87x | 1.4 | 649 | 2.2 | 6.0 | yes |  |
| rf | 1/1 | 13.74 | 12.88 | 25.89 | 18.71 | 0.69x | 7.18 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.94x | 40.4 | 1208 | 13.0 | 27.5 | yes |  |
| ridge | 1/1 | 0.15 | 0.12 | 1.70 | 0.10 | 1.25x | 1.60 | 8.14 | 1 | 1/1 | 1.00x | 1.25x |  | 1.17x | 0.0 | 612 | 0.1 | 0.2 | yes |  |
| xgb | 1/1 | 3.39 | 3.33 | 4.70 | 4.41 | 0.75x | 0.28 |  | 1 | 1/1 (measured) | 0.86x | 0.98x |  | 1.50x | 2.0 | 690 | 13.2 | 10.1 | yes |  half: 4.65/4.58 (n_jobs=2) |

### Backtesting with refit (16 folds, 15 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 42.86 | 44.36 | 21.77 | 20.64 | 2.15x | 1.13 | 2.11 | 3 | 3/3 (measured) | 0.89x | 0.94x | 0.60 | 0.91x | 1.9 | 692 | 132.2 | 63.6 | yes |  all: 16.94/15.90 (n_jobs=4) half: 28.17/26.46 (n_jobs=2) |
| lgbm_1t | 1/1 | 27.60 | 28.73 | 12.43 | 11.30 | 2.54x | 1.13 | 1.87 | 3 | 3/3 | 1.00x | 1.00x | 0.97 | 0.79x | 1.4 | 692 | 28.9 | 29.0 | yes |  all: 11.04/9.82 (n_jobs=4) half: 15.89/15.58 (n_jobs=2) |
| ridge | 1/1 | 0.24 | 0.23 | 1.90 | 0.11 | 2.07x | 1.79 | 3.45 | 1 | 1/1 | 1.00x | 2.07x | 0.99 | 1.06x | 0.0 | 641 | 0.2 | 0.3 | yes |  |
| xgb | 1/1 | 32.92 | 32.56 | 27.13 | 27.63 | 1.18x | -0.50 |  | 3 | 3/3 (measured) | 1.02x | 1.07x | 0.84 | 1.70x | 2.0 | 762 | 129.8 | 80.4 | yes |  all: 23.67/23.45 (n_jobs=4) half: 29.68/27.17 (n_jobs=2) |

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 13.76 | 15.15 | 9.40 | 8.39 | 1.81x | 1.01 | 2.25 | 3 | 3/3 (measured) | 1.03x | 0.97x | 0.65 | 1.00x | 1.9 | 690 | 47.5 | 24.6 | yes |  |
| lgbm_1t | 1/1 | 9.28 | 9.76 | 7.62 | 5.30 | 1.84x | 2.32 | 5.07 | 3 | 3/3 (measured) | 0.90x | 1.07x | 0.79 | 1.02x | 1.4 | 691 | 9.8 | 10.7 | yes |  |
| rf | 1/1 | 71.89 | 69.67 | 39.78 | 38.19 | 1.82x | 1.60 | 3.53 | 3 | 3/3 | 1.00x | 1.00x | 0.87 | 1.19x | 40.4 | 1215 | 70.1 | 79.7 | yes |  |
| ridge | 1/1 | 0.11 | 0.08 | 1.69 | 0.06 | 1.33x | 1.63 | 6.62 | 1 | 1/1 | 1.00x | 1.33x | 0.77 | 0.82x | 0.0 | 636 | 0.1 | 0.1 | yes |  |
| xgb | 1/1 | 12.63 | 13.09 | 13.29 | 13.16 | 0.99x | 0.13 |  | 3 | 3/3 (measured) | 1.17x | 0.97x | 0.86 | 1.66x | 2.0 | 752 | 52.0 | 31.8 | yes |  |

### Grid search without refit (3 candidates, 40 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 16.96 | 17.81 | 13.62 | 12.36 | 1.44x | 1.26 | 4.11 | 3 | 3/3 (measured) | 1.03x | 1.06x |  | 1.22x | 1.9 | 670 | 34.6 | 43.7 | yes |  half: 16.03/14.59 (n_jobs=2) |
| lgbm_1t | 1/1 | 5.78 | 4.69 | 10.67 | 9.32 | 0.50x | 1.35 |  | 1 | 1/1 (measured) | 0.91x | 1.14x |  | 1.03x | 1.4 | 651 | 4.7 | 12.5 | yes |  |
| rf | 1/1 | 33.92 | 35.01 | 40.98 | 41.23 | 0.85x | -0.25 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 1209 | 35.2 | 50.5 | yes |  |
| ridge | 1/1 | 0.19 | 0.17 | 1.93 | 0.21 | 0.82x | 1.72 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.82x | 0.0 | 612 | 0.2 | 0.5 | yes |  |
| xgb | 1/1 | 7.85 | 7.68 | 12.30 | 11.64 | 0.66x | 0.66 |  | 1 | 1/1 (measured) | 0.88x | 0.90x |  | 1.61x | 2.0 | 692 | 30.5 | 25.9 | yes |  half: 11.88/11.77 (n_jobs=2) |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 44.89 | 45.58 | 29.03 | 26.45 | 1.72x | 2.58 | 6.15 | 3 | 3/3 | 1.00x | 1.00x |  | 0.94x | 1.9 | 692 | 145.4 | 78.5 | yes |  |
| lgbm_1t | 1/1 | 32.05 | 29.27 | 17.56 | 17.60 | 1.66x | -0.04 |  | 3 | 3/3 | 1.00x | 1.00x |  | 1.26x | 1.4 | 691 | 29.4 | 36.7 | yes |  |
| ridge | 1/1 | 0.28 | 0.26 | 2.02 | 0.20 | 1.29x | 1.82 | 8.05 | 1 | 1/1 | 1.00x | 1.29x |  | 1.32x | 0.0 | 637 | 0.3 | 0.4 | yes |  |
| xgb | 1/1 | 38.73 | 37.77 | 42.69 | 43.54 | 0.87x | -0.85 |  | 3 | 3/3 | 1.10x | 1.15x |  | 1.79x | 2.0 | 750 | 150.6 | 105.3 | yes |  |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 4 | first | 23 | 423.1 | 355.6 | 311.0 | 307.4 | 2.22x (bt_refit, lgbm_1t) | 15.54x (bt_refit_small, ridge) | 1.18x (bt_norefit, hgb) |
| 4 | next | 23 | 422.5 | 321.4 | 304.2 | 292.0 | 2.54x (bt_refit, lgbm_1t) | 1.99x (gs_norefit, lgbm_1t) | 2.07x (bt_refit, ridge) |
