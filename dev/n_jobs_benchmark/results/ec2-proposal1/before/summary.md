# Benchmark of n_jobs in skforecast

- Machine: Linux-7.0.0-1017-aws-x86_64-with-glibc2.43, 16 logical CPUs, 8 physical cores, 130.8 GB.
- Python 3.12.13; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: c3138f334ad8ab74f3cd2f7d60ba14c57536fcce.
- Measures: 158 finished, 0 failed or timed out.
- Workers: 158 timed measures, 94 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.017 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 16 CPUs

joblib sees 16 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas-5f890258.so', 16], ['openmp', 'libgomp.so.1.0.0', 16], ['blas', 'libscipy_openblas64_-f48b354e.so', 16], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 16]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 3.82 | 3.81 | 5.50 | 2.39 | 1.59x | 3.11 | 8.36 | 15 | 1/1 | 1.00x | 1.59x |  | 0.99x | 0.6 | 2771 | 59.1 | 36.6 | yes |  half: 4.72/2.32 (n_jobs=8) |
| hgb | 1/1 | 19.05 | 19.17 | 13.71 | 11.40 | 1.68x | 2.31 | 5.70 | 15 | 1/1 (measured) | 1.43x | 1.73x |  | 1.00x | 1.9 | 2679 | 33.2 | 54.3 | yes |  half: 13.64/11.78 (n_jobs=8) |
| lgbm_1t | 1/1 | 3.35 | 3.36 | 12.89 | 8.70 | 0.39x | 4.19 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.01x | 1.4 | 2575 | 3.4 | 14.5 | yes |  |
| rf | 1/1 | 21.86 | 21.81 | 36.05 | 34.92 | 0.62x | 1.13 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 5377 | 21.9 | 52.4 | yes |  |
| ridge | 1/1 | 0.25 | 0.23 | 3.20 | 0.17 | 1.29x | 3.02 | 13.41 | 1 | 1/1 | 1.00x | 1.29x |  | 1.00x | 0.0 | 2395 | 0.2 | 0.7 | yes |  |
| xgb | 1/1 | 2.91 | 2.82 | 5.97 | 7.41 | 0.38x | -1.43 |  | 1 | 1/1 (measured) | 1.01x | 1.00x |  | 3.56x | 2.0 | 2790 | 43.3 | 23.2 | yes |  half: 5.25/7.28 (n_jobs=8) |

### Backtesting with refit (16 folds, 15 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 30.41 | 30.35 | 23.01 | 20.09 | 1.51x | 2.92 | 8.63 | 15 | 15/15 | 1.00x | 1.00x | 0.87 | 0.99x | 0.6 | 3200 | 480.2 | 320.2 | yes |  |
| hgb | 1/1 | 60.01 | 60.25 | 15.75 | 12.55 | 4.80x | 3.20 | 4.04 | 15 | 15/15 (measured) | 0.99x | 0.98x | 0.54 | 0.99x | 1.9 | 2793 | 280.3 | 150.0 | yes |  all: 15.89/12.56 (n_jobs=16) half: 16.41/14.03 (n_jobs=8) |
| lgbm_1t | 1/1 | 44.63 | 44.66 | 11.66 | 8.57 | 5.21x | 3.08 | 3.82 | 15 | 15/15 | 1.00x | 1.00x | 0.92 | 0.99x | 1.4 | 2783 | 44.7 | 69.9 | yes |  all: 11.67/8.52 (n_jobs=16) half: 11.85/9.63 (n_jobs=8) |
| ridge | 1/1 | 0.39 | 0.37 | 3.24 | 0.11 | 3.50x | 3.13 | 4.39 | 1 | 1/1 | 1.00x | 3.50x | 0.65 | 0.99x | 0.0 | 2512 | 0.4 | 0.7 | yes |  |
| xgb | 1/1 | 28.42 | 28.00 | 17.97 | 19.62 | 1.43x | -1.65 |  | 15 | 15/15 (measured) | 1.00x | 1.00x | 0.93 | 3.82x | 2.0 | 3093 | 442.3 | 208.1 | yes |  all: 18.01/19.53 (n_jobs=16) half: 17.71/20.11 (n_jobs=8) |

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 11.17 | 11.14 | 10.21 | 9.81 | 1.14x | 0.39 | 3.31 | 15 | 6/6 |  |  | 0.79 | 0.99x | 0.6 | 2605 | 176.2 | 136.0 | yes |  |
| hgb | 1/1 | 20.16 | 20.12 | 9.91 | 9.75 | 2.06x | 0.16 | 0.30 | 15 | 6/6 (measured) | 0.91x | 0.73x | 0.54 | 0.99x | 1.9 | 2321 | 103.3 | 55.5 | yes |  |
| lgbm_1t | 1/1 | 16.71 | 16.71 | 8.43 | 8.12 | 2.06x | 0.32 | 0.61 | 15 | 6/6 (measured) | 0.99x | 0.76x | 0.82 | 1.00x | 1.4 | 2315 | 16.7 | 30.0 | yes |  |
| rf | 1/1 | 116.75 | 116.71 | 42.12 | 41.89 | 2.79x | 0.23 | 0.35 | 15 | 6/6 |  |  | 0.83 | 1.00x | 40.4 | 4011 | 117.2 | 130.8 | yes |  |
| ridge | 1/1 | 0.14 | 0.12 | 2.41 | 1.99 | 0.06x | 0.42 |  | 1 | 1/1 | 1.00x | 1.00x | 0.60 | 1.17x | 0.0 | 2489 | 0.1 | 9.3 | yes |  |
| xgb | 1/1 | 10.79 | 10.48 | 12.78 | 16.74 | 0.63x | -3.95 |  | 15 | 6/6 (measured) | 0.92x | 1.20x | 0.79 | 3.81x | 2.0 | 2540 | 165.6 | 71.6 | yes |  |

### Grid search without refit (3 candidates, 40 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 7.77 | 7.67 | 9.23 | 6.25 | 1.23x | 2.98 | 16.11 | 15 | 15/15 (measured) | 1.19x | 0.99x |  | 1.00x | 0.6 | 2769 | 119.4 | 96.7 | yes |  half: 8.46/6.13 (n_jobs=8) |
| hgb | 1/1 | 27.27 | 27.13 | 20.74 | 18.54 | 1.46x | 2.20 | 6.95 | 15 | 15/15 (measured) | 0.99x | 1.01x |  | 0.99x | 1.9 | 2674 | 70.1 | 95.5 | yes |  half: 20.47/18.53 (n_jobs=8) |
| lgbm_1t | 1/1 | 9.41 | 9.40 | 20.66 | 18.91 | 0.50x | 1.74 |  | 1 | 1/1 (measured) | 1.00x | 1.00x |  | 1.01x | 1.4 | 2573 | 9.4 | 28.5 | yes |  |
| rf | 1/1 | 54.31 | 54.24 | 63.75 | 63.07 | 0.86x | 0.68 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.01x | 40.4 | 5377 | 54.4 | 77.5 | yes |  |
| ridge | 1/1 | 0.32 | 0.30 | 3.36 | 0.28 | 1.06x | 3.08 | 55.92 | 1 | 1/1 | 1.00x | 1.06x |  | 1.01x | 0.0 | 2397 | 0.3 | 0.9 | yes |  |
| xgb | 1/1 | 6.54 | 6.42 | 19.25 | 20.98 | 0.31x | -1.73 |  | 1 | 1/1 (measured) | 0.99x | 0.99x |  | 3.75x | 2.0 | 2784 | 99.9 | 63.7 | yes |  half: 18.41/20.61 (n_jobs=8) |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 33.49 | 33.39 | 28.47 | 23.66 | 1.41x | 4.81 | 16.50 | 15 | 15/15 | 1.00x | 1.00x |  | 1.00x | 0.6 | 3220 | 528.3 | 375.2 | yes |  |
| hgb | 1/1 | 63.07 | 62.98 | 30.05 | 23.67 | 2.66x | 6.38 | 10.22 | 15 | 15/15 | 1.00x | 1.00x |  | 0.98x | 1.9 | 2785 | 325.4 | 133.4 | yes |  |
| lgbm_1t | 1/1 | 52.80 | 52.85 | 25.33 | 19.52 | 2.71x | 5.81 | 9.22 | 15 | 15/15 | 1.00x | 1.00x |  | 1.00x | 1.4 | 2784 | 52.9 | 57.8 | yes |  |
| ridge | 1/1 | 0.40 | 0.38 | 6.37 | 0.28 | 1.35x | 6.09 | 23.47 | 1 | 1/1 | 1.00x | 1.35x |  | 0.97x | 0.0 | 2509 | 0.4 | 1.9 | yes |  |
| xgb | 1/1 | 31.83 | 31.65 | 46.79 | 45.11 | 0.70x | 1.67 |  | 15 | 15/15 | 1.47x | 1.43x |  | 3.84x | 2.0 | 3100 | 500.1 | 182.2 | yes |  |

## 8 CPUs

joblib sees 8 CPUs (8 physical cores). Thread pools: [['openmp', 'libgomp.so.1.0.0', 8], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 8], ['blas', 'libscipy_openblas-5f890258.so', 8], ['blas', 'libscipy_openblas64_-f48b354e.so', 8]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 4.08 | 4.01 | 5.10 | 2.78 | 1.44x | 2.33 | 7.56 | 7 | 1/1 | 1.00x | 1.44x |  | 0.99x | 0.6 | 1319 | 31.4 | 21.8 | yes |  half: 5.14/2.87 (n_jobs=4) |
| hgb | 1/1 | 19.54 | 19.61 | 13.42 | 11.67 | 1.68x | 1.74 | 4.30 | 7 | 1/1 (measured) | 1.46x | 1.72x |  | 1.03x | 1.9 | 1278 | 34.0 | 52.3 | yes |  half: 13.79/11.72 (n_jobs=4) |
| lgbm_1t | 1/1 | 3.34 | 3.36 | 12.73 | 7.66 | 0.44x | 5.07 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 1.4 | 1232 | 3.4 | 11.0 | yes |  |
| rf | 1/1 | 21.82 | 21.79 | 36.63 | 35.30 | 0.62x | 1.33 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 2537 | 21.9 | 52.9 | yes |  |
| ridge | 1/1 | 0.25 | 0.22 | 2.43 | 0.14 | 1.56x | 2.29 | 6.37 | 1 | 1/1 | 1.00x | 1.56x |  | 0.99x | 0.0 | 1147 | 0.2 | 0.4 | yes |  |
| xgb | 1/1 | 3.29 | 3.19 | 5.55 | 5.30 | 0.60x | 0.25 |  | 1 | 1/1 (measured) | 1.00x | 1.00x |  | 2.10x | 2.0 | 1333 | 24.2 | 16.2 | yes |  half: 5.42/5.20 (n_jobs=4) |

### Backtesting with refit (16 folds, 15 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 37.43 | 37.45 | 29.37 | 27.29 | 1.37x | 2.08 | 7.65 | 7 | 7/7 | 1.00x | 1.00x | 0.89 | 1.00x | 0.6 | 1529 | 297.6 | 218.0 | yes |  |
| hgb | 1/1 | 67.76 | 60.91 | 20.01 | 17.84 | 3.41x | 2.16 | 3.06 | 7 | 7/7 (measured) | 1.01x | 1.01x | 0.59 | 1.03x | 1.9 | 1336 | 291.2 | 103.6 | yes |  all: 16.88/14.41 (n_jobs=8) half: 22.97/20.79 (n_jobs=4) |
| lgbm_1t | 1/1 | 44.59 | 44.62 | 13.99 | 11.78 | 3.79x | 2.21 | 3.00 | 7 | 7/7 | 1.00x | 1.00x | 0.93 | 0.99x | 1.4 | 1329 | 44.7 | 50.1 | yes |  all: 11.80/9.61 (n_jobs=8) half: 16.91/14.76 (n_jobs=4) |
| ridge | 1/1 | 0.39 | 0.36 | 2.40 | 0.12 | 3.16x | 2.28 | 3.34 | 1 | 1/1 | 1.00x | 3.16x | 0.67 | 1.00x | 0.0 | 1202 | 0.4 | 0.5 | yes |  |
| xgb | 1/1 | 35.55 | 34.72 | 28.20 | 28.32 | 1.23x | -0.12 |  | 7 | 7/7 (measured) | 1.00x | 0.99x | 1.39 | 2.12x | 2.0 | 1473 | 271.8 | 137.2 | yes |  all: 21.41/21.47 (n_jobs=8) half: 24.10/24.47 (n_jobs=4) |

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 13.79 | 13.74 | 12.67 | 10.79 | 1.27x | 1.88 | 8.79 | 7 | 6/6 |  |  | 0.80 | 1.00x | 0.6 | 1484 | 109.1 | 85.8 | yes |  |
| hgb | 1/1 | 20.35 | 20.29 | 9.92 | 7.63 | 2.66x | 2.30 | 3.68 | 7 | 6/6 (measured) | 1.01x | 1.00x | 0.54 | 0.98x | 1.9 | 1315 | 102.3 | 45.1 | yes |  |
| lgbm_1t | 1/1 | 16.64 | 16.67 | 10.57 | 8.49 | 1.96x | 2.08 | 4.24 | 7 | 6/6 (measured) | 0.78x | 0.73x | 0.82 | 1.74x | 1.4 | 1310 | 16.7 | 22.7 | yes |  |
| rf | 1/1 | 116.41 | 116.36 | 41.99 | 39.76 | 2.93x | 2.23 | 3.39 | 7 | 6/6 |  |  | 0.83 | 1.00x | 40.4 | 2474 | 116.6 | 120.6 | yes |  |
| ridge | 1/1 | 0.14 | 0.13 | 2.28 | 0.99 | 0.13x | 1.30 |  | 1 | 1/1 | 1.00x | 1.00x | 0.60 | 1.24x | 0.0 | 1193 | 0.1 | 1.2 | yes |  |
| xgb | 1/1 | 13.19 | 12.98 | 12.81 | 12.86 | 1.01x | -0.05 |  | 7 | 6/6 (measured) | 1.00x | 1.00x | 0.81 | 2.10x | 2.0 | 1444 | 101.5 | 56.5 | yes |  |

### Grid search without refit (3 candidates, 40 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 8.80 | 8.73 | 9.73 | 7.47 | 1.17x | 2.26 | 15.67 | 7 | 7/7 (measured) | 1.11x | 1.00x |  | 1.00x | 0.6 | 1318 | 68.8 | 58.6 | yes |  half: 9.76/7.50 (n_jobs=4) |
| hgb | 1/1 | 27.18 | 27.29 | 20.39 | 18.40 | 1.48x | 1.98 | 6.09 | 7 | 7/7 (measured) | 1.00x | 1.00x |  | 0.98x | 1.9 | 1280 | 71.1 | 90.8 | yes |  half: 20.15/18.10 (n_jobs=4) |
| lgbm_1t | 1/1 | 9.47 | 9.43 | 19.27 | 17.05 | 0.55x | 2.22 |  | 1 | 1/1 (measured) | 0.99x | 1.00x |  | 1.00x | 1.4 | 1231 | 10.8 | 22.8 | yes |  |
| rf | 1/1 | 54.32 | 54.29 | 62.37 | 61.98 | 0.88x | 0.40 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 2538 | 54.4 | 74.4 | yes |  |
| ridge | 1/1 | 0.32 | 0.30 | 2.51 | 0.25 | 1.22x | 2.26 | 12.37 | 1 | 1/1 | 1.00x | 1.22x |  | 1.02x | 0.0 | 1147 | 0.3 | 0.6 | yes |  |
| xgb | 1/1 | 7.72 | 7.57 | 14.78 | 14.71 | 0.52x | 0.07 |  | 1 | 1/1 (measured) | 0.99x | 1.00x |  | 2.09x | 2.0 | 1328 | 58.3 | 43.6 | yes |  half: 14.65/14.52 (n_jobs=4) |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 41.25 | 41.27 | 34.07 | 31.53 | 1.31x | 2.54 | 10.76 | 7 | 7/7 | 1.00x | 1.00x |  | 1.00x | 0.6 | 1528 | 327.7 | 251.3 | yes |  |
| hgb | 1/1 | 61.35 | 61.44 | 25.98 | 23.70 | 2.59x | 2.28 | 3.71 | 7 | 7/7 | 1.00x | 1.00x |  | 0.98x | 1.9 | 1331 | 318.1 | 133.8 | yes |  |
| lgbm_1t | 1/1 | 53.10 | 53.12 | 21.68 | 19.53 | 2.72x | 2.16 | 3.41 | 7 | 7/7 | 1.00x | 1.00x |  | 1.00x | 1.4 | 1327 | 53.9 | 57.7 | yes |  |
| ridge | 1/1 | 0.40 | 0.38 | 4.24 | 0.20 | 1.87x | 4.04 | 8.68 | 1 | 1/1 | 1.00x | 1.87x |  | 1.00x | 0.0 | 1202 | 0.4 | 0.8 | yes |  |
| xgb | 1/1 | 39.02 | 38.93 | 38.96 | 39.02 | 1.00x | -0.05 |  | 7 | 7/7 | 1.00x | 1.00x |  | 2.10x | 2.0 | 1489 | 303.9 | 166.1 | yes |  |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 16 | first | 26 | 550.1 | 456.5 | 379.4 | 359.1 | 3.83x (bt_refit, lgbm_1t) | 17.02x (bt_refit_small, ridge) | 1.47x (gs_refit_small, xgb) |
| 16 | next | 26 | 548.7 | 402.8 | 346.3 | 325.3 | 5.21x (bt_refit, lgbm_1t) | 16.13x (bt_refit_small, ridge) | 3.50x (bt_refit, ridge) |
| 8 | first | 26 | 591.3 | 459.4 | 398.8 | 393.7 | 3.39x (bt_refit, hgb) | 15.92x (bt_refit_small, ridge) | 1.46x (bt_norefit, hgb) |
| 8 | next | 26 | 583.1 | 412.0 | 376.8 | 368.7 | 3.79x (bt_refit, lgbm_1t) | 7.89x (bt_refit_small, ridge) | 3.16x (bt_refit, ridge) |
