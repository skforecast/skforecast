# Benchmark of n_jobs in skforecast

- Machine: Linux-7.0.0-1017-aws-x86_64-with-glibc2.43, 16 logical CPUs, 8 physical cores, 130.8 GB.
- Python 3.12.13; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: cf3c05b099b4e2dc993976d32d408fa824328c7f.
- Measures: 158 finished, 0 failed or timed out.
- Workers: 158 timed measures, 94 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.017 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 16 CPUs

joblib sees 16 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas64_-f48b354e.so', 16], ['openmp', 'libgomp.so.1.0.0', 16], ['blas', 'libscipy_openblas-5f890258.so', 16], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 16]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 3.87 | 3.81 | 5.36 | 2.25 | 1.69x | 3.11 | 7.61 | 15 | 1/1 | 1.00x | 1.69x |  | 1.00x | 0.6 | 2765 | 59.1 | 34.9 | yes |  half: 4.63/2.26 (n_jobs=8) |
| hgb | 1/1 | 19.13 | 19.40 | 7.96 | 4.84 | 4.01x | 3.12 | 4.16 | 15 | 1/1 (measured) | 2.38x | 3.93x |  | 1.02x | 1.9 | 2628 | 33.3 | 53.1 | yes |  half: 7.65/5.29 (n_jobs=8) |
| lgbm_1t | 1/1 | 3.34 | 3.35 | 6.38 | 3.29 | 1.02x | 3.09 |  | 1 | 1/1 | 1.00x | 1.02x |  | 1.00x | 1.4 | 2581 | 3.4 | 8.0 | yes |  |
| rf | 1/1 | 21.92 | 21.94 | 33.84 | 32.51 | 0.67x | 1.33 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 4792 | 22.0 | 54.3 | yes |  |
| ridge | 1/1 | 0.25 | 0.23 | 3.18 | 0.14 | 1.59x | 3.03 | 8.20 | 1 | 1/1 | 1.00x | 1.59x |  | 0.99x | 0.0 | 2395 | 0.2 | 0.7 | yes |  |
| xgb | 1/1 | 2.97 | 2.78 | 5.46 | 2.32 | 1.20x | 3.14 | 18.92 | 1 | 1/1 (measured) | 0.98x | 1.21x |  | 0.99x | 2.0 | 2788 | 42.8 | 34.2 | yes |  half: 4.74/2.33 (n_jobs=8) |

### Backtesting with refit (16 folds, 15 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 30.36 | 30.32 | 22.96 | 20.21 | 1.50x | 2.76 | 8.27 | 15 | 15/15 | 1.00x | 1.00x | 0.87 | 1.00x | 0.6 | 3211 | 479.2 | 322.0 | yes |  |
| hgb | 1/1 | 60.44 | 60.18 | 15.15 | 11.91 | 5.05x | 3.25 | 4.05 | 15 | 15/15 (measured) | 1.00x | 1.00x | 0.54 | 0.99x | 1.9 | 2766 | 280.8 | 154.5 | yes |  all: 15.47/12.05 (n_jobs=16) half: 16.41/13.39 (n_jobs=8) |
| lgbm_1t | 1/1 | 44.66 | 44.67 | 10.82 | 7.81 | 5.72x | 3.01 | 3.65 | 15 | 15/15 | 1.00x | 1.00x | 0.92 | 0.99x | 1.4 | 2806 | 44.8 | 74.7 | yes |  all: 10.90/7.80 (n_jobs=16) half: 11.40/9.11 (n_jobs=8) |
| ridge | 1/1 | 0.39 | 0.37 | 3.25 | 0.10 | 3.59x | 3.15 | 4.37 | 1 | 1/1 | 1.00x | 3.59x | 0.65 | 0.98x | 0.0 | 2515 | 0.4 | 0.7 | yes |  |
| xgb | 1/1 | 28.91 | 28.02 | 17.86 | 14.80 | 1.89x | 3.06 | 6.49 | 15 | 15/15 (measured) | 1.00x | 1.01x | 0.91 | 1.01x | 2.0 | 3124 | 443.0 | 218.9 | yes |  all: 18.08/14.72 (n_jobs=16) half: 17.51/15.58 (n_jobs=8) |

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 11.17 | 11.12 | 10.17 | 9.70 | 1.15x | 0.47 | 3.64 | 15 | 6/6 |  |  | 0.79 | 0.99x | 0.6 | 2611 | 175.8 | 134.8 | yes |  |
| hgb | 1/1 | 20.29 | 20.32 | 9.69 | 9.24 | 2.20x | 0.45 | 0.83 | 15 | 6/6 (measured) | 0.94x | 0.72x | 0.54 | 0.98x | 1.9 | 2295 | 103.6 | 54.7 | yes |  |
| lgbm_1t | 1/1 | 16.67 | 16.70 | 8.13 | 7.76 | 2.15x | 0.37 | 0.69 | 15 | 6/6 (measured) | 0.99x | 0.75x | 0.82 | 1.00x | 1.4 | 2334 | 16.8 | 29.6 | yes |  |
| rf | 1/1 | 116.62 | 116.67 | 41.91 | 41.54 | 2.81x | 0.36 | 0.56 | 15 | 6/6 |  |  | 0.83 | 1.00x | 40.4 | 4002 | 116.9 | 130.4 | yes |  |
| ridge | 1/1 | 0.14 | 0.12 | 2.40 | 2.02 | 0.06x | 0.38 |  | 1 | 1/1 | 1.00x | 1.00x | 0.60 | 1.09x | 0.0 | 2486 | 0.1 | 9.3 | yes |  |
| xgb | 1/1 | 10.56 | 10.50 | 12.59 | 12.15 | 0.86x | 0.44 |  | 15 | 6/6 (measured) | 0.94x | 0.74x | 0.82 | 0.99x | 2.0 | 2561 | 165.8 | 80.6 | yes |  |

### Grid search without refit (3 candidates, 40 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 7.75 | 7.69 | 9.18 | 6.10 | 1.26x | 3.08 | 14.93 | 15 | 15/15 (measured) | 1.18x | 1.01x |  | 0.99x | 0.6 | 2771 | 120.0 | 94.8 | yes |  half: 8.42/6.09 (n_jobs=8) |
| hgb | 1/1 | 27.44 | 27.53 | 13.19 | 10.20 | 2.70x | 2.99 | 4.75 | 15 | 15/15 (measured) | 1.00x | 1.00x |  | 0.98x | 1.9 | 2631 | 70.8 | 92.7 | yes |  half: 12.92/10.69 (n_jobs=8) |
| lgbm_1t | 1/1 | 9.43 | 9.40 | 12.86 | 9.80 | 0.96x | 3.06 |  | 1 | 1/1 (measured) | 1.00x | 1.00x |  | 1.00x | 1.4 | 2585 | 9.4 | 18.8 | yes |  |
| rf | 1/1 | 54.32 | 54.25 | 60.88 | 59.30 | 0.91x | 1.58 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.99x | 40.4 | 4793 | 54.4 | 75.7 | yes |  |
| ridge | 1/1 | 0.32 | 0.30 | 3.37 | 0.25 | 1.20x | 3.12 | 18.68 | 1 | 1/1 | 1.00x | 1.20x |  | 1.01x | 0.0 | 2396 | 0.3 | 0.9 | yes |  |
| xgb | 1/1 | 6.55 | 6.45 | 9.23 | 6.26 | 1.03x | 2.97 |  | 1 | 1/1 (measured) | 1.04x | 1.04x |  | 1.01x | 2.0 | 2778 | 100.4 | 91.3 | yes |  half: 8.69/6.13 (n_jobs=8) |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 33.49 | 33.46 | 28.34 | 23.44 | 1.43x | 4.90 | 16.35 | 15 | 15/15 | 1.00x | 1.00x |  | 0.99x | 0.6 | 3226 | 529.6 | 371.9 | yes |  |
| hgb | 1/1 | 61.01 | 60.98 | 29.31 | 22.54 | 2.71x | 6.77 | 10.74 | 15 | 15/15 | 1.00x | 1.00x |  | 1.00x | 1.9 | 2753 | 315.1 | 131.9 | yes |  |
| lgbm_1t | 1/1 | 53.02 | 52.93 | 24.58 | 18.43 | 2.87x | 6.15 | 9.44 | 15 | 15/15 | 1.00x | 1.00x |  | 0.99x | 1.4 | 2804 | 53.1 | 57.0 | yes |  |
| ridge | 1/1 | 0.39 | 0.37 | 6.42 | 0.26 | 1.43x | 6.16 | 20.37 | 1 | 1/1 | 1.00x | 1.43x |  | 1.00x | 0.0 | 2496 | 0.4 | 1.8 | yes |  |
| xgb | 1/1 | 32.18 | 31.51 | 37.27 | 30.99 | 1.02x | 6.28 |  | 15 | 15/15 | 1.16x | 1.00x |  | 1.03x | 2.0 | 3128 | 497.9 | 210.2 | yes |  |

## 8 CPUs

joblib sees 8 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas-5f890258.so', 8], ['openmp', 'libgomp.so.1.0.0', 8], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 8], ['blas', 'libscipy_openblas64_-f48b354e.so', 8]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 4.07 | 4.00 | 5.01 | 2.77 | 1.45x | 2.25 | 7.29 | 7 | 1/1 | 1.00x | 1.45x |  | 1.00x | 0.6 | 1315 | 31.3 | 21.6 | yes |  half: 5.13/2.84 (n_jobs=4) |
| hgb | 1/1 | 19.78 | 19.80 | 7.82 | 5.61 | 3.53x | 2.21 | 3.09 | 7 | 1/1 (measured) | 2.45x | 3.47x |  | 1.03x | 1.9 | 1256 | 34.3 | 38.8 | yes |  half: 9.80/7.55 (n_jobs=4) |
| lgbm_1t | 1/1 | 3.34 | 3.35 | 5.57 | 3.28 | 1.02x | 2.29 |  | 1 | 1/1 | 1.00x | 1.02x |  | 1.00x | 1.4 | 1233 | 3.4 | 5.8 | yes |  |
| rf | 1/1 | 21.86 | 21.83 | 33.89 | 32.27 | 0.68x | 1.62 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 2265 | 21.9 | 50.9 | yes |  |
| ridge | 1/1 | 0.25 | 0.23 | 2.41 | 0.13 | 1.73x | 2.27 | 5.40 | 1 | 1/1 | 1.00x | 1.73x |  | 1.00x | 0.0 | 1147 | 0.2 | 0.4 | yes |  |
| xgb | 1/1 | 3.28 | 3.19 | 5.00 | 2.69 | 1.19x | 2.31 | 14.71 | 1 | 1/1 (measured) | 1.00x | 1.18x |  | 1.00x | 2.0 | 1330 | 24.2 | 20.1 | yes |  half: 5.16/2.90 (n_jobs=4) |

### Backtesting with refit (16 folds, 15 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 37.41 | 37.37 | 29.42 | 27.22 | 1.37x | 2.21 | 8.13 | 7 | 7/7 | 1.00x | 1.00x | 0.90 | 0.99x | 0.6 | 1535 | 296.8 | 217.4 | yes |  |
| hgb | 1/1 | 60.92 | 61.48 | 20.00 | 17.43 | 3.53x | 2.57 | 3.59 | 7 | 7/7 (measured) | 1.00x | 1.00x | 0.54 | 0.99x | 1.9 | 1321 | 288.7 | 101.6 | yes |  all: 16.23/13.73 (n_jobs=8) half: 23.46/20.56 (n_jobs=4) |
| lgbm_1t | 1/1 | 44.51 | 44.76 | 13.89 | 11.68 | 3.83x | 2.21 | 2.99 | 7 | 7/7 | 1.00x | 1.00x | 0.92 | 0.99x | 1.4 | 1339 | 44.8 | 48.9 | yes |  all: 11.39/9.04 (n_jobs=8) half: 16.76/14.58 (n_jobs=4) |
| ridge | 1/1 | 0.39 | 0.36 | 2.38 | 0.10 | 3.49x | 2.28 | 3.19 | 1 | 1/1 | 1.00x | 3.49x | 0.66 | 1.01x | 0.0 | 1202 | 0.4 | 0.5 | yes |  |
| xgb | 1/1 | 34.44 | 34.39 | 28.13 | 25.89 | 1.33x | 2.24 | 9.08 | 7 | 7/7 (measured) | 1.00x | 1.00x | 0.93 | 1.04x | 2.0 | 1488 | 268.9 | 141.2 | yes |  all: 21.35/19.03 (n_jobs=8) half: 24.31/22.22 (n_jobs=4) |

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 13.79 | 13.75 | 12.67 | 10.76 | 1.28x | 1.91 | 8.79 | 7 | 6/6 |  |  | 0.80 | 1.00x | 0.6 | 1487 | 109.1 | 85.8 | yes |  |
| hgb | 1/1 | 20.07 | 20.34 | 9.54 | 7.22 | 2.82x | 2.32 | 3.59 | 7 | 6/6 (measured) | 1.00x | 0.99x | 0.53 | 1.03x | 1.9 | 1300 | 104.2 | 44.8 | yes |  |
| lgbm_1t | 1/1 | 16.75 | 16.75 | 7.99 | 5.80 | 2.89x | 2.19 | 3.35 | 7 | 6/6 (measured) | 1.00x | 1.00x | 0.82 | 0.99x | 1.4 | 1338 | 16.8 | 18.9 | yes |  |
| rf | 1/1 | 116.36 | 116.40 | 41.75 | 39.51 | 2.95x | 2.24 | 3.40 | 7 | 6/6 |  |  | 0.83 | 1.00x | 40.4 | 2291 | 116.6 | 120.5 | yes |  |
| ridge | 1/1 | 0.14 | 0.12 | 2.30 | 0.98 | 0.13x | 1.32 |  | 1 | 1/1 | 1.00x | 1.00x | 0.60 | 1.10x | 0.0 | 1191 | 0.1 | 1.3 | yes |  |
| xgb | 1/1 | 13.02 | 12.73 | 12.78 | 10.58 | 1.20x | 2.20 | 13.04 | 7 | 6/6 (measured) | 1.01x | 1.03x | 0.82 | 1.02x | 2.0 | 1459 | 99.2 | 60.7 | yes |  |

### Grid search without refit (3 candidates, 40 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 8.79 | 8.74 | 9.74 | 7.47 | 1.17x | 2.27 | 15.59 | 7 | 7/7 (measured) | 1.10x | 0.99x |  | 1.00x | 0.6 | 1319 | 68.8 | 58.4 | yes |  half: 9.72/7.52 (n_jobs=4) |
| hgb | 1/1 | 27.65 | 27.78 | 13.16 | 10.98 | 2.53x | 2.18 | 3.61 | 7 | 7/7 (measured) | 1.00x | 1.01x |  | 0.98x | 1.9 | 1255 | 71.4 | 77.8 | yes |  half: 15.77/13.47 (n_jobs=4) |
| lgbm_1t | 1/1 | 9.43 | 9.40 | 11.98 | 9.69 | 0.97x | 2.29 |  | 1 | 1/1 (measured) | 1.00x | 1.00x |  | 1.00x | 1.4 | 1235 | 9.4 | 13.7 | yes |  |
| rf | 1/1 | 54.16 | 54.12 | 60.47 | 58.50 | 0.93x | 1.97 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 2285 | 54.3 | 70.8 | yes |  |
| ridge | 1/1 | 0.32 | 0.30 | 2.49 | 0.22 | 1.39x | 2.28 | 8.14 | 1 | 1/1 | 1.00x | 1.39x |  | 0.99x | 0.0 | 1147 | 0.3 | 0.6 | yes |  |
| xgb | 1/1 | 7.73 | 7.50 | 9.53 | 7.24 | 1.04x | 2.29 |  | 1 | 1/1 (measured) | 0.99x | 1.05x |  | 0.99x | 2.0 | 1333 | 57.6 | 54.7 | yes |  half: 9.74/7.47 (n_jobs=4) |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 41.25 | 41.22 | 33.98 | 31.91 | 1.29x | 2.07 | 9.15 | 7 | 7/7 | 1.00x | 1.00x |  | 1.00x | 0.6 | 1528 | 327.3 | 254.2 | yes |  |
| hgb | 1/1 | 62.07 | 62.23 | 24.72 | 22.42 | 2.78x | 2.30 | 3.60 | 7 | 7/7 | 1.00x | 1.00x |  | 0.98x | 1.9 | 1317 | 320.8 | 131.9 | yes |  |
| lgbm_1t | 1/1 | 52.99 | 53.11 | 20.63 | 18.36 | 2.89x | 2.27 | 3.47 | 7 | 7/7 | 1.00x | 1.00x |  | 0.99x | 1.4 | 1338 | 53.2 | 56.6 | yes |  |
| ridge | 1/1 | 0.39 | 0.37 | 4.25 | 0.20 | 1.84x | 4.05 | 8.90 | 1 | 1/1 | 1.00x | 1.84x |  | 0.99x | 0.0 | 1197 | 0.4 | 0.7 | yes |  |
| xgb | 1/1 | 39.04 | 38.65 | 34.37 | 32.11 | 1.20x | 2.26 | 13.33 | 7 | 7/7 | 1.00x | 1.00x |  | 1.00x | 2.0 | 1497 | 302.0 | 177.4 | yes |  |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 16 | first | 26 | 549.8 | 399.7 | 359.0 | 342.4 | 4.13x (bt_refit, lgbm_1t) | 16.73x (bt_refit_small, ridge) | 2.38x (bt_norefit, hgb) |
| 16 | next | 26 | 547.6 | 318.9 | 309.2 | 299.4 | 5.72x (bt_refit, lgbm_1t) | 16.39x (bt_refit_small, ridge) | 3.93x (bt_norefit, hgb) |
| 8 | first | 26 | 584.1 | 411.5 | 382.9 | 370.6 | 3.20x (bt_refit, lgbm_1t) | 16.09x (bt_refit_small, ridge) | 2.45x (bt_norefit, hgb) |
| 8 | next | 26 | 584.1 | 352.8 | 353.7 | 336.8 | 3.83x (bt_refit, lgbm_1t) | 7.95x (bt_refit_small, ridge) | 3.49x (bt_refit, ridge) |
