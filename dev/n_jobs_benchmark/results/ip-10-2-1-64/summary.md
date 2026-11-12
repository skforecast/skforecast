# Benchmark of n_jobs in skforecast

- Machine: Linux-7.0.0-1017-aws-x86_64-with-glibc2.43, 16 logical CPUs, 8 physical cores, 130.8 GB.
- Python 3.12.13; skforecast 0.27.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: 6fa488c7b2a2e12a42fa0c1df56777bc4a39313b.
- Measures: 566 finished, 0 failed or timed out.
- Workers: 494 timed measures, 264 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 30 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.032 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 16 CPUs

joblib sees 16 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas64_-f48b354e.so', 16], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 16], ['blas', 'libscipy_openblas-5f890258.so', 16], ['openmp', 'libgomp.so.1.0.0', 16]]. First use of the window features in a process: 0.00 s.

### Start of the workers

| workers | window features | first | next | start cost |
|---|---|---|---|---|
| 2 | yes | 2.47 | 0.21 | 2.26 |
| 4 | yes | 2.50 | 0.22 | 2.27 |
| 8 | yes | 2.62 | 0.23 | 2.39 |
| 15 | yes | 3.40 | 0.26 | 3.14 |
| 16 | yes | 3.46 | 0.25 | 3.20 |
| 15 | no | 3.41 | 0.25 | 3.15 |

Consecutive calls changing the number of workers:

| n_jobs | threads per worker | seconds | executor id | processes |
|---|---|---|---|---|
| 15 | 1 | 3.30 | 0 | 15 |
| 15 | 1 | 0.18 | 0 | 15 |
| 8 | 2 | 3.16 | 1 | 8 |
| 15 | 1 | 3.70 | 2 | 15 |
| 12 | 1 | 0.99 | 2 | 12 |
| 15 | 1 | 2.26 | 2 | 15 |
| 2 | 8 | 3.02 | 3 | 2 |
| 15 | 1 | 3.57 | 4 | 15 |

### Ladder of work: backtesting with refit of LightGBM with one thread (16 folds), by training rows

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 rows | 2/2 | 1.86 | 1.84 | 3.62 | 0.49 | 3.78x | 3.13 | 4.26 | 15 | 1/1 (measured) | 1.00x | 3.78x | 0.94 | 1.00x | 0.3 | 2684 | 1.8 | 2.7 | yes |  |
| 300 rows | 2/2 | 4.25 | 4.24 | 3.99 | 0.92 | 4.62x | 3.08 | 3.93 | 15 | 15/15 (measured) | 1.01x | 0.99x | 0.94 | 1.00x | 0.5 | 2819 | 4.3 | 6.1 | yes |  |
| 400 rows | 2/2 | 7.43 | 7.39 | 4.58 | 1.48 | 5.01x | 3.10 | 3.88 | 15 | 15/15 (measured) | 1.00x | 1.01x | 0.94 | 1.00x | 0.6 | 2838 | 7.4 | 10.9 | yes |  |
| 550 rows | 2/2 | 14.68 | 14.78 | 5.84 | 2.82 | 5.23x | 3.01 | 3.72 | 15 | 15/15 (measured) | 0.99x | 0.99x | 0.92 | 1.00x | 0.9 | 2875 | 14.8 | 22.3 | yes |  |
| 800 rows | 2/2 | 30.68 | 30.84 | 9.08 | 6.15 | 5.01x | 2.93 | 3.65 | 15 | 15/15 (measured) | 1.02x | 1.00x | 0.91 | 0.99x | 1.3 | 2835 | 30.9 | 46.6 | yes |  |

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 3.88 | 3.81 | 5.50 | 2.36 | 1.61x | 3.14 | 8.25 | 15 | 1/1 | 1.00x | 1.61x |  | 0.99x | 0.6 | 2923 | 58.9 | 36.3 | yes |  half: 4.76/2.31 (n_jobs=8) |
| catboost_1t | 1/1 | 13.03 | 12.83 | 14.52 | 11.48 | 1.12x | 3.04 | 28.76 | 15 | 1/1 | 1.00x | 1.12x |  | 1.00x | 0.6 | 3083 | 40.9 | 20.6 | yes |  |
| hgb | 2/2 | 18.90 | 18.95 | 13.90 | 11.81 | 1.60x | 2.08 | 5.53 | 15 | 1/1 (measured) | 1.43x | 1.70x |  | 1.00x | 1.9 | 2857 | 32.7 | 54.6 | yes |  half: 13.36/11.46 (n_jobs=8) |
| lgbm | 2/2 | 1.46 | 1.46 | 5.93 | 2.47 | 0.59x | 3.46 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 1.4 | 2670 | 10.6 | 17.9 | yes |  |
| lgbm_1t | 2/2 | 3.34 | 3.33 | 12.66 | 8.70 | 0.38x | 3.96 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 1.4 | 2669 | 3.3 | 14.5 | yes |  |
| rf | 2/2 | 21.87 | 21.84 | 36.33 | 34.98 | 0.62x | 1.35 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 5556 | 21.9 | 54.1 | yes |  |
| rf_all | 1/1 | 4.15 | 4.14 | 18.86 | 17.24 | 0.24x | 1.62 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.01x | 40.4 | 5555 | 25.7 | 55.4 | yes |  |
| ridge | 2/2 | 0.24 | 0.22 | 3.21 | 0.17 | 1.28x | 3.04 | 13.75 | 1 | 1/1 | 1.00x | 1.28x |  | 1.02x | 0.0 | 2482 | 0.2 | 0.7 | yes |  |
| xgb | 2/2 | 2.94 | 2.82 | 6.03 | 7.41 | 0.38x | -1.39 |  | 1 | 1/1 (measured) | 0.99x | 1.00x |  | 3.75x | 2.0 | 2935 | 43.3 | 23.8 | yes |  half: 5.13/7.20 (n_jobs=8) |
| xgb_1t | 1/1 | 9.09 | 9.01 | 12.34 | 9.15 | 0.99x | 3.19 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 2.0 | 3086 | 9.1 | 13.4 | yes |  |

### Backtesting with refit (16 folds, 15 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 30.36 | 30.33 | 23.05 | 20.29 | 1.50x | 2.77 | 8.36 | 15 | 15/15 | 1.00x | 1.00x | 0.87 | 1.00x | 0.6 | 3360 | 479.7 | 323.3 | yes |  |
| catboost_1t | 1/1 | 174.68 | 174.36 | 31.31 | 28.26 | 6.17x | 3.05 | 3.64 | 15 | 15/15 | 1.00x | 1.00x | 0.93 | 1.00x | 0.6 | 3482 | 218.4 | 273.6 | yes |  |
| hgb | 2/2 | 60.17 | 60.20 | 15.79 | 12.46 | 4.83x | 3.33 | 4.20 | 15 | 15/15 (measured) | 1.00x | 1.00x | 0.53 | 1.01x | 1.9 | 2887 | 281.5 | 149.7 | yes |  all: 15.96/12.43 (n_jobs=16) half: 16.25/14.07 (n_jobs=8) |
| lgbm | 2/0 | 13.97 | 13.95 |  |  |  |  |  | 1 | 1/1 (measured) | 0.98x | 0.98x | 0.91 |  | 1.4 |  | 108.6 |  | yes |  |
| lgbm_1t | 2/2 | 44.61 | 44.66 | 11.63 | 8.59 | 5.20x | 3.04 | 3.76 | 15 | 15/15 | 1.00x | 1.00x | 0.93 | 1.00x | 1.4 | 3001 | 44.7 | 70.3 | yes |  all: 12.49/9.38 (n_jobs=16) half: 11.84/9.61 (n_jobs=8) |
| rf_all | 1/1 | 29.97 | 30.09 | 28.34 | 24.56 | 1.22x | 3.78 | 20.58 | 15 | 15/15 | 1.00x | 1.00x | 0.82 | 0.98x | 40.4 | 6230 | 373.8 | 380.1 | yes |  |
| ridge | 2/2 | 0.38 | 0.36 | 3.25 | 0.11 | 3.23x | 3.14 | 4.55 | 1 | 1/1 | 1.00x | 3.23x | 0.67 | 1.01x | 0.0 | 2623 | 0.4 | 0.7 | yes |  |
| xgb | 2/2 | 28.39 | 28.15 | 18.05 | 19.63 | 1.43x | -1.58 |  | 15 | 15/15 (measured) | 1.00x | 1.00x | 0.91 | 3.78x | 2.0 | 3241 | 444.8 | 207.3 | yes |  all: 18.07/19.63 (n_jobs=16) half: 17.75/20.34 (n_jobs=8) |
| xgb_1t | 1/1 | 122.24 | 121.98 | 24.37 | 21.29 | 5.73x | 3.08 | 3.73 | 15 | 15/15 | 1.00x | 1.00x | 0.98 | 1.00x | 2.0 | 3395 | 123.2 | 198.8 | yes |  |

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 11.17 | 11.12 | 10.23 | 9.89 | 1.12x | 0.34 | 3.08 | 15 | 6/6 |  |  | 0.79 | 0.99x | 0.6 | 2871 | 176.0 | 137.1 | yes |  |
| hgb | 2/2 | 19.92 | 19.93 | 9.99 | 9.72 | 2.05x | 0.27 | 0.53 | 15 | 6/6 (measured) | 0.93x | 0.74x | 0.54 | 0.99x | 1.9 | 2490 | 101.7 | 55.4 | yes |  |
| lgbm | 2/1 | 5.16 | 5.12 | 92.19 |  |  |  |  | 1 | 1/1 | 1.00x | 1.00x | 0.81 | 0.98x | 1.4 | 1610 | 39.8 |  | yes |  |
| lgbm_1t | 2/2 | 16.67 | 16.67 | 8.41 | 8.14 | 2.05x | 0.27 | 0.53 | 15 | 6/6 (measured) | 0.99x | 0.76x | 0.82 | 1.00x | 1.4 | 2406 | 16.7 | 29.9 | yes |  |
| rf | 2/2 | 116.52 | 116.57 | 42.23 | 41.92 | 2.78x | 0.31 | 0.48 | 15 | 6/6 |  |  | 0.83 | 1.00x | 40.4 | 4175 | 117.9 | 130.8 | yes |  |
| ridge | 2/2 | 0.14 | 0.12 | 2.39 | 2.02 | 0.06x | 0.37 |  | 1 | 1/1 | 1.00x | 1.00x | 0.60 | 1.08x | 0.0 | 2593 | 0.1 | 9.3 | yes |  |
| xgb | 2/2 | 10.73 | 10.56 | 12.56 | 16.83 | 0.63x | -4.27 |  | 15 | 6/6 (measured) | 0.93x | 1.17x | 0.81 | 3.76x | 2.0 | 2664 | 166.8 | 71.8 | yes |  |

### Grid search without refit (3 candidates, 40 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 7.74 | 7.69 | 9.25 | 6.27 | 1.23x | 2.98 | 16.18 | 15 | 15/15 (measured) | 1.19x | 0.99x |  | 1.00x | 0.6 | 2919 | 119.6 | 96.9 | yes |  half: 8.50/6.14 (n_jobs=8) |
| catboost_1t | 1/1 | 34.78 | 34.72 | 36.30 | 33.34 | 1.04x | 2.96 |  | 15 | 15/15 | 1.04x | 1.00x |  | 1.00x | 0.6 | 3082 | 71.0 | 48.3 | yes |  |
| hgb | 2/2 | 26.83 | 26.82 | 20.69 | 18.63 | 1.44x | 2.06 | 6.75 | 15 | 15/15 (measured) | 1.00x | 1.00x |  | 1.00x | 1.9 | 2765 | 69.1 | 95.2 | yes |  half: 20.48/18.64 (n_jobs=8) |
| lgbm | 2/2 | 3.34 | 3.31 | 8.73 | 5.87 | 0.56x | 2.87 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 1.4 | 2671 | 25.0 | 39.5 | yes |  |
| lgbm_1t | 2/2 | 9.42 | 9.40 | 20.54 | 18.93 | 0.50x | 1.62 |  | 1 | 1/1 (measured) | 1.00x | 1.00x |  | 1.00x | 1.4 | 2669 | 9.4 | 28.6 | yes |  |
| rf | 2/2 | 54.24 | 54.27 | 64.22 | 63.59 | 0.85x | 0.63 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 5554 | 54.4 | 78.2 | yes |  |
| rf_all | 1/1 | 7.38 | 7.26 | 17.32 | 16.75 | 0.43x | 0.57 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 5557 | 63.9 | 86.8 | yes |  |
| ridge | 2/2 | 0.32 | 0.30 | 3.39 | 0.28 | 1.08x | 3.11 | 44.51 | 1 | 1/1 | 1.00x | 1.08x |  | 0.99x | 0.0 | 2482 | 0.3 | 0.9 | yes |  |
| xgb | 2/2 | 6.58 | 6.38 | 19.23 | 20.89 | 0.31x | -1.66 |  | 1 | 1/1 (measured) | 1.00x | 1.02x |  | 3.71x | 2.0 | 2928 | 99.4 | 64.1 | yes |  half: 18.23/20.53 (n_jobs=8) |
| xgb_1t | 1/1 | 25.33 | 25.23 | 28.97 | 26.24 | 0.96x | 2.73 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 2.0 | 3081 | 25.5 | 34.9 | yes |  |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 33.42 | 33.42 | 28.53 | 23.57 | 1.42x | 4.95 | 16.81 | 15 | 15/15 | 1.00x | 1.00x |  | 1.00x | 0.6 | 3522 | 528.7 | 373.8 | yes |  |
| hgb | 1/1 | 61.42 | 61.41 | 30.13 | 23.80 | 2.58x | 6.32 | 10.33 | 15 | 15/15 | 1.00x | 1.00x |  | 1.00x | 1.9 | 2982 | 317.7 | 133.7 | yes |  |
| lgbm | 1/0 | 15.93 | 15.90 |  |  |  |  |  | 1 | 1/1 | 1.00x | 1.00x |  |  | 1.4 |  | 123.8 |  | yes |  |
| lgbm_1t | 1/1 | 52.99 | 53.00 | 25.24 | 19.47 | 2.72x | 5.77 | 9.13 | 15 | 15/15 | 1.00x | 1.00x |  | 1.00x | 1.4 | 3000 | 53.1 | 57.7 | yes |  |
| ridge | 1/1 | 0.39 | 0.37 | 6.47 | 0.26 | 1.41x | 6.21 | 21.23 | 1 | 1/1 | 1.00x | 1.41x |  | 0.99x | 0.0 | 2733 | 0.4 | 1.8 | yes |  |
| xgb | 1/1 | 32.01 | 31.42 | 46.73 | 45.09 | 0.70x | 1.64 |  | 15 | 15/15 | 1.46x | 1.44x |  | 3.81x | 2.0 | 3396 | 496.5 | 183.0 | yes |  |

### Multi-series backtesting without refit (50 series, 52 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 4.31 | 4.20 | 8.20 | 5.66 | 0.74x | 2.53 |  | 15 | 1/1 | 1.00x | 1.00x |  | 1.01x | 0.9 | 2772 | 11.3 | 17.2 | yes |  |
| lgbm | 1/1 | 3.26 | 3.18 | 7.44 | 4.59 | 0.69x | 2.84 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.06x | 1.5 | 2784 | 12.6 | 20.0 | yes |  |
| lgbm_1t | 1/1 | 5.15 | 5.07 | 10.80 | 9.04 | 0.56x | 1.76 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.99x | 1.5 | 2782 | 5.1 | 13.9 | yes |  |
| ridge | 1/1 | 1.88 | 1.80 | 5.16 | 2.51 | 0.72x | 2.65 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.04x | 0.1 | 2604 | 2.2 | 4.4 | yes |  |
| xgb | 1/1 | 3.28 | 3.17 | 6.74 | 8.79 | 0.36x | -2.05 |  | 1 | 1/1 | 1.00x | 1.00x |  | 5.37x | 2.3 | 3117 | 24.0 | 18.3 | yes |  |

### Multi-series backtesting with refit (50 series, 8 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 6.46 | 6.31 | 5.78 | 5.46 | 1.16x | 0.32 | 2.38 | 15 | 8/8 |  |  | 0.79 | 1.00x | 0.5 | 3179 | 39.6 | 30.9 | yes |  |
| lgbm | 1/0 | 11.03 | 11.15 |  |  |  |  |  | 1 | 1/1 | 1.00x | 1.00x | 0.86 |  | 1.5 |  | 82.2 |  | yes |  |
| lgbm_1t | 1/1 | 23.87 | 23.88 | 9.47 | 9.33 | 2.56x | 0.14 | 0.24 | 15 | 8/8 |  |  | 0.85 | 1.00x | 1.5 | 3243 | 23.9 | 40.5 | yes |  |
| ridge | 1/1 | 0.75 | 0.78 | 3.07 | 2.67 | 0.29x | 0.40 |  | 1 | 1/1 | 1.00x | 1.00x | 0.56 | 1.03x | 0.1 | 3003 | 2.1 | 16.0 | yes |  |
| xgb | 1/1 | 9.43 | 9.32 | 8.17 | 12.72 | 0.73x | -4.55 |  | 15 | 8/8 |  |  | 0.82 | 5.02x | 2.3 | 3457 | 137.0 | 56.0 | yes |  |

### Threads of the prediction

Microseconds per call of the predict method of the estimator, by rows. `native=k`: k threads with the parameter of the library. `limit=k`: k OpenMP threads with threadpoolctl in every call.

catboost, 100 trees:

| rows | default | native=1 | native=2 | native=4 | native=8 |
|---|---|---|---|---|---|
| 1 | 721 | 603 | 607 | 607 | 634 |
| 5 | 804 | 671 | 683 | 691 | 725 |
| 10 | 796 | 682 | 695 | 699 | 720 |
| 25 | 809 | 698 | 717 | 704 | 764 |
| 50 | 831 | 725 | 726 | 725 | 746 |
| 100 | 868 | 780 | 770 | 768 | 776 |
| 500 | 1217 | 1231 | 1122 | 1077 | 1127 |

catboost, 500 trees:

| rows | default | native=1 | native=2 | native=4 | native=8 |
|---|---|---|---|---|---|
| 1 | 740 | 626 | 630 | 629 | 681 |
| 5 | 809 | 706 | 717 | 724 | 769 |
| 10 | 848 | 734 | 745 | 761 | 765 |
| 25 | 874 | 759 | 773 | 776 | 812 |
| 50 | 895 | 784 | 780 | 784 | 788 |
| 100 | 963 | 866 | 852 | 882 | 871 |
| 500 | 1521 | 1544 | 1312 | 1199 | 1262 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 0.634 | 0.634 | 0.881 | 1.00x | yes |
| no_estimator | 0.059 | 0.059 | 0.082 | 0.09x |  |
| native=1 | 0.539 | 0.538 | 0.749 | 0.85x | yes |

catboost, 1000 trees:

| rows | default | native=1 | native=2 | native=4 | native=8 |
|---|---|---|---|---|---|
| 1 | 739 | 622 | 626 | 627 | 649 |
| 5 | 840 | 725 | 740 | 747 | 784 |
| 10 | 907 | 767 | 775 | 786 | 794 |
| 25 | 887 | 794 | 801 | 811 | 852 |
| 50 | 939 | 819 | 842 | 817 | 884 |
| 100 | 1059 | 940 | 921 | 918 | 965 |
| 500 | 1785 | 1952 | 1699 | 1672 | 1735 |

hgb, 100 trees:

| rows | default | limit=1 | limit=2 | limit=4 | limit=8 |
|---|---|---|---|---|---|
| 1 | 1925 | 1602 | 1841 | 1860 | 1962 |
| 5 | 2002 | 1431 | 1850 | 1930 | 2028 |
| 10 | 2067 | 1504 | 1912 | 2007 | 2106 |
| 25 | 2101 | 1656 | 1978 | 2038 | 2136 |
| 50 | 2175 | 1872 | 2117 | 2104 | 2209 |
| 100 | 2197 | 2266 | 2360 | 2233 | 2257 |
| 500 | 2810 | 5510 | 4399 | 3372 | 2858 |

hgb, 500 trees:

| rows | default | limit=1 | limit=2 | limit=4 | limit=8 |
|---|---|---|---|---|---|
| 1 | 8620 | 7066 | 7902 | 8261 | 8700 |
| 5 | 8859 | 6290 | 8050 | 8293 | 8857 |
| 10 | 9103 | 6428 | 8201 | 8606 | 9135 |
| 25 | 9448 | 7113 | 8607 | 8763 | 9440 |
| 50 | 9692 | 8083 | 9143 | 9128 | 9721 |
| 100 | 9927 | 9932 | 10387 | 9778 | 9957 |
| 500 | 12984 | 25067 | 19749 | 15212 | 13008 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 5.101 | 5.074 | 7.085 | 1.00x | yes |
| no_estimator | 0.092 | 0.092 | 0.128 | 0.02x |  |
| outer_limit=1 | 5.057 | 5.049 | 7.023 | 0.99x | yes |
| all_threads | 6.278 | 6.264 | 8.719 | 1.23x | yes |

hgb, 1000 trees:

| rows | default | limit=1 | limit=2 | limit=4 | limit=8 |
|---|---|---|---|---|---|
| 1 | 16649 | 13511 | 15138 | 15896 | 16714 |
| 5 | 17836 | 12167 | 15932 | 16837 | 17790 |
| 10 | 18442 | 12822 | 16544 | 17701 | 18373 |
| 25 | 19070 | 14149 | 17541 | 18153 | 18999 |
| 50 | 19690 | 16148 | 18512 | 19154 | 19786 |
| 100 | 20507 | 19763 | 21049 | 20503 | 20471 |
| 500 | 26113 | 49376 | 39478 | 30519 | 26101 |

lgbm, 100 trees:

| rows | default | native=1 | native=2 | native=4 | native=8 |
|---|---|---|---|---|---|
| 1 | 74 | 71 | 72 | 73 | 74 |
| 5 | 84 | 96 | 91 | 87 | 83 |
| 10 | 95 | 129 | 108 | 99 | 95 |
| 25 | 107 | 267 | 165 | 122 | 107 |
| 50 | 130 | 473 | 280 | 176 | 129 |
| 100 | 183 | 900 | 503 | 284 | 181 |
| 500 | 721 | 4483 | 2419 | 1301 | 699 |

lgbm, 500 trees:

| rows | default | native=1 | native=2 | native=4 | native=8 |
|---|---|---|---|---|---|
| 1 | 113 | 109 | 110 | 111 | 113 |
| 5 | 151 | 296 | 205 | 160 | 148 |
| 10 | 221 | 554 | 354 | 272 | 220 |
| 25 | 303 | 1421 | 736 | 423 | 306 |
| 50 | 465 | 2773 | 1423 | 771 | 462 |
| 100 | 791 | 5438 | 2794 | 1448 | 787 |
| 500 | 3648 | 27120 | 13785 | 7083 | 3671 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 0.183 | 0.182 | 0.254 | 1.00x | yes |
| no_estimator | 0.060 | 0.060 | 0.084 | 0.33x |  |
| native=1 | 0.178 | 0.178 | 0.247 | 0.98x | yes |

lgbm, 1000 trees:

| rows | default | native=1 | native=2 | native=4 | native=8 |
|---|---|---|---|---|---|
| 1 | 162 | 158 | 160 | 160 | 161 |
| 5 | 257 | 561 | 430 | 330 | 257 |
| 10 | 407 | 1102 | 676 | 516 | 407 |
| 25 | 563 | 2774 | 1459 | 819 | 555 |
| 50 | 875 | 5477 | 2788 | 1488 | 880 |
| 100 | 1507 | 10767 | 5493 | 2808 | 1504 |
| 500 | 7101 | 53602 | 27104 | 13801 | 7197 |

xgb, 100 trees:

| rows | default | native=1 | limit=1 | native=2 | limit=2 | native=4 | limit=4 | native=8 | limit=8 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 235 | 210 | 256 | 218 | 264 | 220 | 267 | 224 | 271 |
| 5 | 294 | 279 | 327 | 279 | 329 | 275 | 324 | 276 | 326 |
| 10 | 308 | 290 | 338 | 292 | 339 | 287 | 335 | 288 | 338 |
| 25 | 341 | 330 | 378 | 330 | 379 | 328 | 376 | 330 | 378 |
| 50 | 444 | 401 | 444 | 403 | 447 | 398 | 444 | 404 | 457 |
| 100 | 471 | 550 | 599 | 454 | 502 | 442 | 497 | 445 | 494 |
| 500 | 582 | 1686 | 1733 | 1011 | 1059 | 645 | 717 | 535 | 582 |

xgb, 500 trees:

| rows | default | native=1 | limit=1 | native=2 | limit=2 | native=4 | limit=4 | native=8 | limit=8 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 330 | 303 | 348 | 312 | 359 | 315 | 360 | 319 | 365 |
| 5 | 575 | 667 | 717 | 619 | 667 | 576 | 622 | 562 | 609 |
| 10 | 635 | 723 | 774 | 673 | 724 | 629 | 674 | 619 | 670 |
| 25 | 796 | 888 | 938 | 861 | 907 | 827 | 875 | 845 | 891 |
| 50 | 1238 | 1168 | 1220 | 1128 | 1182 | 1122 | 1160 | 1132 | 1184 |
| 100 | 1432 | 1907 | 1957 | 1341 | 1394 | 1293 | 1361 | 1316 | 1385 |
| 500 | 1741 | 7297 | 7352 | 3952 | 4000 | 2173 | 2330 | 1548 | 1600 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 0.333 | 0.332 | 0.463 | 1.00x | yes |
| no_estimator | 0.059 | 0.058 | 0.081 | 0.18x |  |
| outer_limit=1 | 0.309 | 0.309 | 0.429 | 0.93x | yes |
| native=1 | 0.312 | 0.311 | 0.434 | 0.94x | yes |

xgb, 1000 trees:

| rows | default | native=1 | limit=1 | native=2 | limit=2 | native=4 | limit=4 | native=8 | limit=8 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 621 | 438 | 485 | 443 | 489 | 448 | 498 | 448 | 503 |
| 5 | 1185 | 1196 | 1251 | 1110 | 1159 | 1023 | 1107 | 1019 | 1110 |
| 10 | 1075 | 1294 | 1351 | 1222 | 1265 | 1129 | 1207 | 1139 | 1230 |
| 25 | 1576 | 1623 | 1683 | 1542 | 1582 | 1466 | 1523 | 1476 | 1562 |
| 50 | 2209 | 2168 | 2215 | 2093 | 2126 | 2005 | 2074 | 2031 | 2116 |
| 100 | 2545 | 3615 | 3675 | 2468 | 2499 | 2474 | 2471 | 2428 | 2505 |
| 500 | 3414 | 14446 | 14455 | 7680 | 7716 | 4164 | 4388 | 3006 | 3101 |

## 8 CPUs

joblib sees 8 CPUs (8 physical cores). Thread pools: [['blas', 'libscipy_openblas64_-f48b354e.so', 8], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 8], ['blas', 'libscipy_openblas-5f890258.so', 8], ['openmp', 'libgomp.so.1.0.0', 8]]. First use of the window features in a process: 0.00 s.

### Start of the workers

| workers | window features | first | next | start cost |
|---|---|---|---|---|
| 2 | yes | 2.43 | 0.21 | 2.22 |
| 4 | yes | 2.49 | 0.22 | 2.27 |
| 7 | yes | 2.53 | 0.22 | 2.31 |
| 8 | yes | 2.58 | 0.23 | 2.35 |
| 7 | no | 2.52 | 0.22 | 2.30 |

Consecutive calls changing the number of workers:

| n_jobs | threads per worker | seconds | executor id | processes |
|---|---|---|---|---|
| 7 | 1 | 2.41 | 0 | 7 |
| 7 | 1 | 0.16 | 0 | 7 |
| 4 | 2 | 2.73 | 1 | 4 |
| 7 | 1 | 2.74 | 2 | 7 |
| 4 | 2 | 2.76 | 3 | 4 |
| 7 | 1 | 2.72 | 4 | 7 |
| 2 | 4 | 2.75 | 5 | 2 |
| 7 | 1 | 2.70 | 6 | 7 |

### Ladder of work: backtesting with refit of LightGBM with one thread (16 folds), by training rows

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 rows | 2/2 | 1.86 | 1.84 | 2.78 | 0.54 | 3.38x | 2.24 | 3.18 | 7 | 1/1 (measured) | 0.99x | 3.37x | 0.94 | 1.01x | 0.3 | 1282 | 1.9 | 2.4 | yes |  |
| 300 rows | 2/2 | 4.27 | 4.25 | 3.35 | 1.20 | 3.55x | 2.15 | 3.00 | 7 | 7/7 (measured) | 1.01x | 1.01x | 0.89 | 1.00x | 0.5 | 1293 | 4.3 | 5.1 | yes |  |
| 400 rows | 2/2 | 7.39 | 7.40 | 4.15 | 1.99 | 3.72x | 2.17 | 2.96 | 7 | 7/7 (measured) | 0.99x | 0.99x | 0.94 | 1.00x | 0.6 | 1303 | 7.4 | 8.4 | yes |  |
| 550 rows | 2/2 | 14.79 | 14.68 | 6.08 | 3.89 | 3.78x | 2.19 | 2.98 | 7 | 7/7 (measured) | 1.00x | 0.98x | 0.91 | 1.00x | 0.9 | 1319 | 14.7 | 16.6 | yes |  |
| 800 rows | 2/2 | 31.09 | 31.15 | 10.60 | 8.47 | 3.68x | 2.13 | 2.93 | 7 | 7/7 (measured) | 1.00x | 1.00x | 0.93 | 1.00x | 1.3 | 1352 | 31.2 | 36.1 | yes |  |

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 4.05 | 3.98 | 5.07 | 2.75 | 1.45x | 2.32 | 7.49 | 7 | 1/1 | 1.00x | 1.45x |  | 1.00x | 0.6 | 1391 | 31.2 | 21.5 | yes |  |
| hgb | 2/2 | 19.23 | 19.21 | 13.66 | 11.80 | 1.63x | 1.86 | 4.82 | 7 | 1/1 | 1.41x | 1.63x |  | 1.00x | 1.9 | 1321 | 33.2 | 51.4 | yes |  |
| lgbm | 2/2 | 1.45 | 1.46 | 5.44 | 9.34 | 0.16x | -3.90 |  | 1 | 1/1 | 1.00x | 1.00x |  | 9.26x | 1.4 | 1277 | 10.6 | 19.4 | yes |  |
| lgbm_1t | 2/2 | 3.33 | 3.35 | 12.69 | 7.65 | 0.44x | 5.05 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 1.4 | 1274 | 3.4 | 10.9 | yes |  |
| rf | 2/2 | 21.85 | 21.82 | 36.37 | 35.10 | 0.62x | 1.28 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 2579 | 21.9 | 52.6 | yes |  |
| ridge | 2/2 | 0.25 | 0.23 | 2.42 | 0.15 | 1.51x | 2.28 | 6.76 | 1 | 1/1 | 1.00x | 1.51x |  | 0.99x | 0.0 | 1188 | 0.2 | 0.4 | yes |  |
| xgb | 2/2 | 3.28 | 3.19 | 5.56 | 5.24 | 0.61x | 0.32 |  | 1 | 1/1 | 1.00x | 1.00x |  | 2.10x | 2.0 | 1402 | 24.2 | 16.1 | yes |  |

### Backtesting with refit (16 folds, 15 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 37.29 | 37.25 | 29.60 | 27.18 | 1.37x | 2.41 | 8.93 | 7 | 7/7 | 1.00x | 1.00x | 0.89 | 0.99x | 0.6 | 1603 | 296.0 | 217.1 | yes |  |
| hgb | 2/2 | 60.49 | 60.51 | 19.98 | 17.81 | 3.40x | 2.16 | 3.07 | 7 | 7/7 | 1.00x | 1.00x | 0.53 | 0.99x | 1.9 | 1380 | 281.9 | 103.5 | yes |  |
| lgbm | 2/0 | 13.84 | 13.77 |  |  |  |  |  | 1 | 1/1 | 1.00x | 1.00x | 0.89 |  | 1.4 |  | 107.2 |  | yes |  |
| lgbm_1t | 2/2 | 44.57 | 44.66 | 13.90 | 11.81 | 3.78x | 2.09 | 2.84 | 7 | 7/7 | 1.00x | 1.00x | 0.93 | 0.99x | 1.4 | 1379 | 44.7 | 50.1 | yes |  |
| ridge | 2/2 | 0.39 | 0.36 | 2.41 | 0.10 | 3.50x | 2.30 | 3.22 | 1 | 1/1 | 1.00x | 3.50x | 0.65 | 1.00x | 0.0 | 1254 | 0.4 | 0.5 | yes |  |
| xgb | 2/2 | 34.44 | 34.31 | 28.17 | 28.24 | 1.22x | -0.07 |  | 7 | 7/7 | 1.00x | 1.00x | 0.93 | 2.10x | 2.0 | 1555 | 267.7 | 137.1 | yes |  |

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 13.77 | 13.73 | 12.75 | 10.73 | 1.28x | 2.02 | 9.23 | 7 | 6/6 |  |  | 0.80 | 1.00x | 0.6 | 1558 | 108.9 | 85.5 | yes |  |
| hgb | 2/2 | 20.18 | 20.05 | 9.92 | 7.64 | 2.63x | 2.28 | 3.68 | 7 | 6/6 (measured) | 0.99x | 0.99x | 0.54 | 1.00x | 1.9 | 1359 | 102.8 | 44.8 | yes |  |
| lgbm | 2/0 | 5.12 | 5.10 |  |  |  |  |  | 1 | 1/1 | 1.00x | 1.00x | 0.81 |  | 1.4 |  | 39.8 |  | yes |  |
| lgbm_1t | 2/2 | 16.69 | 16.69 | 8.27 | 6.19 | 2.70x | 2.09 | 3.32 | 7 | 6/6 (measured) | 1.00x | 1.00x | 0.82 | 1.00x | 1.4 | 1358 | 16.7 | 20.3 | yes |  |
| rf | 2/2 | 116.19 | 116.24 | 42.04 | 39.79 | 2.92x | 2.25 | 3.41 | 7 | 6/6 |  |  | 0.83 | 1.00x | 40.4 | 2520 | 116.7 | 120.7 | yes |  |
| ridge | 2/2 | 0.14 | 0.12 | 2.27 | 0.98 | 0.13x | 1.29 |  | 1 | 1/1 | 1.00x | 1.00x | 0.59 | 1.25x | 0.0 | 1243 | 0.1 | 1.2 | yes |  |
| xgb | 2/2 | 13.00 | 12.95 | 12.92 | 12.99 | 1.00x | -0.08 |  | 7 | 6/6 (measured) | 0.99x | 1.00x | 0.82 | 2.13x | 2.0 | 1514 | 101.0 | 57.5 | yes |  |

### Grid search without refit (3 candidates, 40 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 8.76 | 8.71 | 9.73 | 7.45 | 1.17x | 2.28 | 15.75 | 7 | 7/7 | 1.11x | 1.00x |  | 1.00x | 0.6 | 1390 | 68.5 | 58.4 | yes |  |
| hgb | 2/2 | 27.21 | 27.33 | 20.16 | 18.33 | 1.49x | 1.83 | 5.56 | 7 | 7/7 | 1.00x | 1.00x |  | 0.98x | 1.9 | 1320 | 71.0 | 91.4 | yes |  |
| lgbm | 2/2 | 3.34 | 3.30 | 22.04 | 26.95 | 0.12x | -4.91 |  | 1 | 1/1 | 1.00x | 1.00x |  | 9.25x | 1.4 | 1277 | 25.0 | 50.3 | yes |  |
| lgbm_1t | 2/2 | 9.44 | 9.42 | 19.28 | 17.06 | 0.55x | 2.22 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 1.4 | 1277 | 9.4 | 22.7 | yes |  |
| rf | 2/2 | 54.19 | 54.19 | 62.97 | 62.50 | 0.87x | 0.47 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 2579 | 54.3 | 74.7 | yes |  |
| ridge | 2/2 | 0.32 | 0.30 | 2.52 | 0.25 | 1.19x | 2.27 | 13.95 | 1 | 1/1 | 1.00x | 1.19x |  | 1.01x | 0.0 | 1188 | 0.3 | 0.6 | yes |  |
| xgb | 2/2 | 7.69 | 7.57 | 14.79 | 14.76 | 0.51x | 0.04 |  | 1 | 1/1 | 1.00x | 1.00x |  | 2.10x | 2.0 | 1399 | 58.2 | 44.4 | yes |  |

### Multi-series backtesting with refit (50 series, 8 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/0 | 6.50 | 6.41 |  |  |  |  |  | 7 | 7/7 |  |  | 0.77 |  | 0.5 |  | 39.7 |  | yes |  |
| lgbm_1t | 0/1 |  |  | 9.24 | 7.21 |  | 2.04 |  | 7 | 7/7 | 1.00x | 1.00x |  |  | 1.5 | 1466 |  | 26.6 | yes |  |

### Threads of the prediction

Microseconds per call of the predict method of the estimator, by rows. `native=k`: k threads with the parameter of the library. `limit=k`: k OpenMP threads with threadpoolctl in every call.

catboost, 100 trees:

| rows | default | native=1 | native=2 | native=4 |
|---|---|---|---|---|
| 1 | 628 | 612 | 617 | 616 |
| 5 | 694 | 674 | 679 | 700 |
| 10 | 717 | 688 | 704 | 712 |
| 25 | 716 | 699 | 712 | 707 |
| 50 | 729 | 726 | 722 | 722 |
| 100 | 774 | 779 | 762 | 758 |
| 500 | 1079 | 1248 | 1191 | 1117 |

catboost, 500 trees:

| rows | default | native=1 | native=2 | native=4 |
|---|---|---|---|---|
| 1 | 630 | 623 | 629 | 629 |
| 5 | 724 | 703 | 714 | 715 |
| 10 | 753 | 731 | 739 | 752 |
| 25 | 775 | 755 | 763 | 763 |
| 50 | 781 | 779 | 774 | 780 |
| 100 | 852 | 865 | 848 | 845 |
| 500 | 1373 | 1581 | 1371 | 1281 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 0.542 | 0.542 | 0.753 | 1.00x | yes |
| no_estimator | 0.059 | 0.059 | 0.082 | 0.11x |  |
| native=1 | 0.538 | 0.538 | 0.747 | 0.99x | yes |

catboost, 1000 trees:

| rows | default | native=1 | native=2 | native=4 |
|---|---|---|---|---|
| 1 | 631 | 619 | 624 | 625 |
| 5 | 745 | 719 | 723 | 736 |
| 10 | 788 | 761 | 770 | 774 |
| 25 | 813 | 785 | 796 | 807 |
| 50 | 809 | 811 | 805 | 809 |
| 100 | 923 | 934 | 916 | 913 |
| 500 | 1541 | 1891 | 1505 | 1472 |

hgb, 100 trees:

| rows | default | limit=1 | limit=2 | limit=4 |
|---|---|---|---|---|
| 1 | 1955 | 1630 | 1848 | 1881 |
| 5 | 1956 | 1463 | 1853 | 1889 |
| 10 | 2014 | 1521 | 1891 | 1936 |
| 25 | 2040 | 1655 | 1959 | 2021 |
| 50 | 2112 | 1852 | 2077 | 2067 |
| 100 | 2180 | 2270 | 2331 | 2191 |
| 500 | 2816 | 5528 | 4391 | 3348 |

hgb, 500 trees:

| rows | default | limit=1 | limit=2 | limit=4 |
|---|---|---|---|---|
| 1 | 8779 | 6993 | 8215 | 8321 |
| 5 | 9396 | 6243 | 8412 | 8667 |
| 10 | 9329 | 6548 | 8609 | 8994 |
| 25 | 9559 | 7173 | 9067 | 9331 |
| 50 | 9847 | 8170 | 9668 | 9631 |
| 100 | 10269 | 10162 | 10985 | 10101 |
| 500 | 13144 | 25235 | 20359 | 15538 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 5.086 | 5.066 | 7.064 | 1.00x | yes |
| no_estimator | 0.093 | 0.092 | 0.129 | 0.02x |  |
| outer_limit=1 | 5.025 | 5.014 | 6.979 | 0.99x | yes |
| all_threads | 6.393 | 6.351 | 8.879 | 1.26x | yes |

hgb, 1000 trees:

| rows | default | limit=1 | limit=2 | limit=4 |
|---|---|---|---|---|
| 1 | 17029 | 13420 | 15562 | 16105 |
| 5 | 18284 | 12310 | 16528 | 17134 |
| 10 | 18746 | 12813 | 17006 | 17720 |
| 25 | 19666 | 14234 | 16736 | 18564 |
| 50 | 20042 | 16157 | 19092 | 19336 |
| 100 | 21062 | 20133 | 21802 | 20558 |
| 500 | 26710 | 50059 | 40340 | 30749 |

lgbm, 100 trees:

| rows | default | native=1 | native=2 | native=4 |
|---|---|---|---|---|
| 1 | 76 | 72 | 73 | 74 |
| 5 | 86 | 97 | 92 | 89 |
| 10 | 95 | 131 | 103 | 99 |
| 25 | 111 | 268 | 161 | 126 |
| 50 | 132 | 473 | 277 | 177 |
| 100 | 180 | 895 | 488 | 290 |
| 500 | 685 | 4446 | 2335 | 1242 |

lgbm, 500 trees:

| rows | default | native=1 | native=2 | native=4 |
|---|---|---|---|---|
| 1 | 112 | 108 | 109 | 111 |
| 5 | 154 | 295 | 222 | 188 |
| 10 | 222 | 553 | 355 | 274 |
| 25 | 305 | 1416 | 735 | 421 |
| 50 | 457 | 2770 | 1422 | 764 |
| 100 | 774 | 5425 | 2774 | 1437 |
| 500 | 3625 | 27014 | 13600 | 7018 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 0.182 | 0.181 | 0.252 | 1.00x | yes |
| no_estimator | 0.060 | 0.060 | 0.083 | 0.33x |  |
| native=1 | 0.177 | 0.177 | 0.246 | 0.98x | yes |

lgbm, 1000 trees:

| rows | default | native=1 | native=2 | native=4 |
|---|---|---|---|---|
| 1 | 160 | 155 | 157 | 158 |
| 5 | 252 | 550 | 424 | 324 |
| 10 | 334 | 1082 | 592 | 436 |
| 25 | 539 | 2725 | 1428 | 789 |
| 50 | 865 | 5382 | 2739 | 1457 |
| 100 | 1481 | 10572 | 5394 | 2762 |
| 500 | 7002 | 52599 | 26577 | 13508 |

xgb, 100 trees:

| rows | default | native=1 | limit=1 | native=2 | limit=2 | native=4 | limit=4 |
|---|---|---|---|---|---|---|---|
| 1 | 221 | 207 | 253 | 215 | 261 | 219 | 264 |
| 5 | 506 | 281 | 327 | 274 | 323 | 284 | 320 |
| 10 | 498 | 292 | 340 | 286 | 344 | 292 | 337 |
| 25 | 356 | 326 | 374 | 327 | 376 | 323 | 372 |
| 50 | 394 | 392 | 440 | 393 | 442 | 390 | 440 |
| 100 | 526 | 545 | 596 | 444 | 499 | 443 | 493 |
| 500 | 520 | 1669 | 1716 | 1007 | 1061 | 660 | 710 |

xgb, 500 trees:

| rows | default | native=1 | limit=1 | native=2 | limit=2 | native=4 | limit=4 |
|---|---|---|---|---|---|---|---|
| 1 | 316 | 304 | 350 | 310 | 356 | 315 | 363 |
| 5 | 773 | 670 | 719 | 602 | 659 | 578 | 620 |
| 10 | 2139 | 728 | 767 | 678 | 729 | 627 | 678 |
| 25 | 1261 | 890 | 935 | 856 | 889 | 824 | 873 |
| 50 | 1141 | 1171 | 1218 | 1143 | 1185 | 1104 | 1161 |
| 100 | 1335 | 1906 | 1959 | 1313 | 1375 | 1309 | 1348 |
| 500 | 1488 | 7289 | 7347 | 3989 | 4019 | 2295 | 2228 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 0.322 | 0.321 | 0.447 | 1.00x | yes |
| no_estimator | 0.059 | 0.059 | 0.082 | 0.18x |  |
| outer_limit=1 | 0.307 | 0.306 | 0.426 | 0.95x | yes |
| native=1 | 0.311 | 0.311 | 0.432 | 0.97x | yes |

xgb, 1000 trees:

| rows | default | native=1 | limit=1 | native=2 | limit=2 | native=4 | limit=4 |
|---|---|---|---|---|---|---|---|
| 1 | 451 | 428 | 482 | 440 | 491 | 448 | 496 |
| 5 | 1028 | 1209 | 1268 | 1108 | 1171 | 1022 | 1103 |
| 10 | 1149 | 1300 | 1353 | 1192 | 1271 | 1128 | 1195 |
| 25 | 1475 | 1620 | 1677 | 1513 | 1583 | 1453 | 1534 |
| 50 | 2028 | 2170 | 2230 | 2072 | 2143 | 2022 | 2083 |
| 100 | 2444 | 3645 | 3698 | 2470 | 2506 | 2445 | 2493 |
| 500 | 2840 | 14423 | 14444 | 7663 | 7755 | 4324 | 4193 |

## 4 CPUs

joblib sees 4 CPUs (4 physical cores). Thread pools: [['blas', 'libscipy_openblas64_-f48b354e.so', 4], ['blas', 'libscipy_openblas-5f890258.so', 4], ['openmp', 'libgomp.so.1.0.0', 4], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 4]]. First use of the window features in a process: 0.00 s.

### Start of the workers

| workers | window features | first | next | start cost |
|---|---|---|---|---|
| 2 | yes | 2.43 | 0.21 | 2.22 |
| 3 | yes | 2.48 | 0.21 | 2.27 |
| 4 | yes | 2.60 | 0.22 | 2.38 |
| 3 | no | 2.49 | 0.21 | 2.28 |

Consecutive calls changing the number of workers:

| n_jobs | threads per worker | seconds | executor id | processes |
|---|---|---|---|---|
| 3 | 1 | 2.35 | 0 | 3 |
| 3 | 1 | 0.16 | 0 | 3 |
| 2 | 2 | 2.59 | 1 | 2 |
| 3 | 1 | 2.64 | 2 | 3 |
| 2 | 2 | 2.62 | 3 | 2 |
| 3 | 1 | 2.64 | 4 | 3 |
| 2 | 2 | 2.65 | 5 | 2 |
| 3 | 1 | 2.61 | 6 | 3 |

### Ladder of work: backtesting with refit of LightGBM with one thread (16 folds), by training rows

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 200 rows | 2/2 | 1.85 | 1.83 | 3.05 | 0.83 | 2.22x | 2.22 | 4.04 | 3 | 1/1 (measured) | 1.01x | 2.23x | 0.94 | 1.00x | 0.3 | 582 | 1.8 | 2.2 | yes |  |
| 300 rows | 2/2 | 4.24 | 4.25 | 3.99 | 1.77 | 2.40x | 2.23 | 3.81 | 3 | 3/3 (measured) | 1.00x | 1.01x | 0.94 | 1.00x | 0.5 | 586 | 4.3 | 4.8 | yes |  |
| 400 rows | 2/2 | 7.41 | 7.36 | 5.22 | 3.03 | 2.43x | 2.18 | 3.71 | 3 | 3/3 (measured) | 1.01x | 1.00x | 0.94 | 0.97x | 0.6 | 590 | 7.4 | 8.2 | yes |  |
| 550 rows | 2/2 | 14.74 | 14.75 | 8.14 | 5.93 | 2.49x | 2.21 | 3.69 | 3 | 3/3 (measured) | 1.00x | 1.00x | 0.91 | 0.96x | 0.9 | 598 | 14.8 | 16.0 | yes |  |
| 800 rows | 2/2 | 31.24 | 31.15 | 14.89 | 12.73 | 2.45x | 2.17 | 3.66 | 3 | 3/3 (measured) | 1.00x | 1.00x | 0.93 | 1.00x | 1.3 | 612 | 31.5 | 34.6 | yes |  |

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 5.14 | 5.07 | 6.32 | 4.11 | 1.23x | 2.21 | 11.66 | 3 | 1/1 | 1.00x | 1.23x |  | 0.99x | 0.6 | 627 | 20.1 | 16.2 | yes |  |
| hgb | 2/2 | 19.25 | 19.24 | 14.04 | 11.86 | 1.62x | 2.18 | 5.68 | 3 | 1/1 | 1.37x | 1.62x |  | 1.01x | 1.9 | 598 | 24.7 | 42.6 | yes |  |
| lgbm | 2/2 | 1.53 | 1.54 | 6.87 | 6.44 | 0.24x | 0.44 |  | 1 | 1/1 | 1.00x | 1.00x |  | 4.51x | 1.4 | 580 | 5.7 | 12.7 | yes |  |
| lgbm_1t | 2/2 | 3.34 | 3.35 | 13.02 | 7.37 | 0.45x | 5.66 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 1.4 | 579 | 3.4 | 9.6 | yes |  |
| rf | 2/2 | 21.93 | 21.89 | 37.10 | 35.36 | 0.62x | 1.74 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.99x | 40.4 | 1137 | 22.1 | 48.5 | yes |  |
| ridge | 2/2 | 0.24 | 0.22 | 2.44 | 0.19 | 1.16x | 2.25 | 16.33 | 1 | 1/1 | 1.00x | 1.16x |  | 1.01x | 0.0 | 541 | 0.6 | 0.6 | yes |  |
| xgb | 2/2 | 4.08 | 3.98 | 6.16 | 4.86 | 0.82x | 1.30 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.37x | 2.0 | 635 | 15.2 | 12.6 | yes |  |

### Backtesting with refit (16 folds, 15 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 54.36 | 54.36 | 50.68 | 48.50 | 1.12x | 2.18 | 20.24 | 3 | 3/3 | 1.00x | 1.00x | 0.91 | 1.00x | 0.6 | 729 | 216.6 | 194.0 | yes |  |
| hgb | 2/2 | 58.26 | 58.47 | 32.93 | 30.90 | 1.89x | 2.03 | 4.30 | 3 | 3/3 | 1.00x | 1.00x | 0.51 | 1.00x | 1.9 | 624 | 145.9 | 94.0 | yes |  half: 39.43/37.22 (n_jobs=2) |
| lgbm | 2/0 | 15.29 | 15.34 |  |  |  |  |  | 1 | 1/1 | 1.00x | 1.00x | 0.90 |  | 1.4 |  | 61.1 |  | yes |  |
| lgbm_1t | 2/2 | 44.52 | 44.55 | 20.04 | 17.83 | 2.50x | 2.21 | 3.68 | 3 | 3/3 | 1.00x | 1.00x | 0.92 | 1.00x | 1.4 | 625 | 44.8 | 48.4 | yes |  |
| ridge | 2/2 | 0.39 | 0.36 | 2.46 | 0.22 | 1.67x | 2.25 | 5.61 | 1 | 1/1 | 1.00x | 1.67x | 0.64 | 0.99x | 0.0 | 573 | 1.5 | 0.7 | yes |  |
| xgb | 2/2 | 46.68 | 46.56 | 44.90 | 43.75 | 1.06x | 1.15 | 19.04 | 3 | 3/3 | 1.00x | 1.00x | 0.95 | 1.38x | 2.0 | 698 | 180.4 | 129.0 | yes |  |

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 20.18 | 20.14 | 20.65 | 18.45 | 1.09x | 2.19 | 26.26 | 3 | 3/3 | 1.02x | 1.00x | 0.81 | 1.00x | 0.6 | 720 | 80.2 | 73.8 | yes |  |
| hgb | 2/2 | 19.02 | 19.01 | 14.45 | 12.19 | 1.56x | 2.26 | 6.31 | 3 | 3/3 (measured) | 1.00x | 1.00x | 0.52 | 0.98x | 1.9 | 621 | 51.7 | 34.6 | yes |  |
| lgbm | 2/0 | 5.68 | 5.65 |  |  |  |  |  | 1 | 1/1 | 1.00x | 1.00x | 0.81 |  | 1.4 |  | 22.2 |  | yes |  |
| lgbm_1t | 2/2 | 16.77 | 16.66 | 11.00 | 8.84 | 1.88x | 2.16 | 4.60 | 3 | 3/3 (measured) | 1.00x | 1.00x | 0.84 | 1.00x | 1.4 | 622 | 17.2 | 18.1 | yes |  |
| rf | 2/2 | 116.03 | 116.05 | 61.27 | 59.05 | 1.97x | 2.22 | 4.51 | 3 | 3/3 | 1.00x | 1.00x | 0.83 | 1.00x | 40.4 | 1145 | 116.5 | 118.9 | yes |  |
| ridge | 2/2 | 0.14 | 0.12 | 2.36 | 0.11 | 1.08x | 2.25 | 29.02 | 1 | 1/1 | 1.00x | 1.08x | 0.58 | 1.00x | 0.0 | 568 | 0.5 | 0.4 | yes |  |
| xgb | 2/2 | 17.91 | 17.87 | 21.68 | 20.57 | 0.87x | 1.11 |  | 3 | 3/3 (measured) | 1.21x | 1.15x | 0.82 | 1.38x | 2.0 | 693 | 69.4 | 51.5 | yes |  |

### Grid search without refit (3 candidates, 40 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 2/2 | 11.95 | 11.91 | 13.25 | 10.97 | 1.09x | 2.28 | 28.80 | 3 | 3/3 | 1.11x | 1.00x |  | 1.00x | 0.6 | 627 | 47.2 | 43.3 | yes |  |
| hgb | 2/2 | 26.66 | 26.58 | 20.61 | 18.49 | 1.44x | 2.12 | 6.97 | 3 | 3/3 | 1.00x | 1.00x |  | 1.01x | 1.9 | 598 | 43.6 | 65.8 | yes |  |
| lgbm | 2/2 | 3.67 | 3.65 | 15.66 | 16.41 | 0.22x | -0.75 |  | 1 | 1/1 | 1.00x | 1.00x |  | 4.41x | 1.4 | 579 | 14.0 | 29.0 | yes |  |
| lgbm_1t | 2/2 | 9.42 | 9.40 | 18.26 | 15.06 | 0.62x | 3.19 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 1.4 | 580 | 9.4 | 18.2 | yes |  |
| rf | 2/2 | 54.25 | 54.19 | 62.27 | 61.50 | 0.88x | 0.77 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 40.4 | 1138 | 54.3 | 71.2 | yes |  |
| ridge | 2/2 | 0.33 | 0.30 | 2.63 | 0.37 | 0.82x | 2.26 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.00x | 0.0 | 541 | 1.2 | 1.4 | yes |  |
| xgb | 2/2 | 10.12 | 10.00 | 14.73 | 13.52 | 0.74x | 1.21 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.37x | 2.0 | 632 | 38.5 | 34.4 | yes |  |

### Multi-series backtesting with refit (50 series, 8 folds)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ridge | 1/0 | 0.73 | 0.76 |  |  |  |  |  | 1 | 1/1 | 1.00x | 1.00x | 0.58 |  | 0.1 |  | 2.1 |  | yes |  |

### Threads of the prediction

Microseconds per call of the predict method of the estimator, by rows. `native=k`: k threads with the parameter of the library. `limit=k`: k OpenMP threads with threadpoolctl in every call.

catboost, 100 trees:

| rows | default | native=1 | native=2 |
|---|---|---|---|
| 1 | 606 | 606 | 608 |
| 5 | 657 | 635 | 646 |
| 10 | 666 | 646 | 657 |
| 25 | 679 | 662 | 669 |
| 50 | 689 | 683 | 684 |
| 100 | 730 | 738 | 729 |
| 500 | 1042 | 1176 | 1081 |

catboost, 500 trees:

| rows | default | native=1 | native=2 |
|---|---|---|---|
| 1 | 624 | 623 | 625 |
| 5 | 685 | 670 | 683 |
| 10 | 709 | 695 | 707 |
| 25 | 737 | 722 | 729 |
| 50 | 745 | 736 | 741 |
| 100 | 811 | 820 | 811 |
| 500 | 1188 | 1483 | 1216 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 0.543 | 0.543 | 0.754 | 1.00x | yes |
| no_estimator | 0.059 | 0.059 | 0.082 | 0.11x |  |
| native=1 | 0.541 | 0.540 | 0.751 | 1.00x | yes |

catboost, 1000 trees:

| rows | default | native=1 | native=2 |
|---|---|---|---|
| 1 | 622 | 622 | 624 |
| 5 | 711 | 686 | 698 |
| 10 | 744 | 729 | 739 |
| 25 | 770 | 755 | 764 |
| 50 | 780 | 776 | 775 |
| 100 | 885 | 892 | 884 |
| 500 | 1521 | 1882 | 1555 |

hgb, 100 trees:

| rows | default | limit=1 | limit=2 |
|---|---|---|---|
| 1 | 1635 | 1574 | 1653 |
| 5 | 1508 | 1436 | 1534 |
| 10 | 1523 | 1460 | 1543 |
| 25 | 1555 | 1620 | 1629 |
| 50 | 1614 | 1819 | 1734 |
| 100 | 1733 | 2210 | 1934 |
| 500 | 2654 | 5460 | 3689 |

hgb, 500 trees:

| rows | default | limit=1 | limit=2 |
|---|---|---|---|
| 1 | 7410 | 6913 | 7317 |
| 5 | 6905 | 6297 | 6777 |
| 10 | 6904 | 6538 | 6889 |
| 25 | 7066 | 7171 | 7172 |
| 50 | 7306 | 8080 | 7685 |
| 100 | 7963 | 10082 | 8713 |
| 500 | 12187 | 25060 | 16795 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 5.081 | 5.058 | 7.057 | 1.00x | yes |
| no_estimator | 0.092 | 0.092 | 0.128 | 0.02x |  |
| outer_limit=1 | 5.020 | 5.005 | 6.972 | 0.99x | yes |
| all_threads | 5.320 | 5.319 | 7.389 | 1.05x | yes |

hgb, 1000 trees:

| rows | default | limit=1 | limit=2 |
|---|---|---|---|
| 1 | 14405 | 13421 | 14207 |
| 5 | 13506 | 12159 | 13029 |
| 10 | 13544 | 12646 | 13337 |
| 25 | 14080 | 14230 | 14220 |
| 50 | 14618 | 16015 | 15118 |
| 100 | 15547 | 19655 | 17135 |
| 500 | 24058 | 49389 | 32947 |

lgbm, 100 trees:

| rows | default | native=1 | native=2 |
|---|---|---|---|
| 1 | 71 | 70 | 70 |
| 5 | 79 | 92 | 83 |
| 10 | 85 | 122 | 96 |
| 25 | 106 | 250 | 148 |
| 50 | 153 | 438 | 251 |
| 100 | 252 | 829 | 439 |
| 500 | 1103 | 4115 | 2122 |

lgbm, 500 trees:

| rows | default | native=1 | native=2 |
|---|---|---|---|
| 1 | 110 | 109 | 110 |
| 5 | 159 | 278 | 198 |
| 10 | 208 | 519 | 296 |
| 25 | 382 | 1323 | 688 |
| 50 | 719 | 2579 | 1332 |
| 100 | 1345 | 5043 | 2589 |
| 500 | 6518 | 25040 | 12731 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 0.178 | 0.178 | 0.248 | 1.00x | yes |
| no_estimator | 0.061 | 0.061 | 0.084 | 0.34x |  |
| native=1 | 0.176 | 0.176 | 0.245 | 0.99x | yes |

lgbm, 1000 trees:

| rows | default | native=1 | native=2 |
|---|---|---|---|
| 1 | 159 | 157 | 158 |
| 5 | 278 | 538 | 358 |
| 10 | 357 | 1048 | 547 |
| 25 | 752 | 2635 | 1381 |
| 50 | 1410 | 5195 | 2646 |
| 100 | 2682 | 10211 | 5216 |
| 500 | 13019 | 50762 | 25640 |

xgb, 100 trees:

| rows | default | native=1 | limit=1 | native=2 | limit=2 |
|---|---|---|---|---|---|
| 1 | 214 | 208 | 253 | 213 | 259 |
| 5 | 262 | 274 | 323 | 267 | 316 |
| 10 | 275 | 286 | 335 | 281 | 328 |
| 25 | 313 | 325 | 373 | 319 | 367 |
| 50 | 378 | 391 | 439 | 384 | 433 |
| 100 | 418 | 544 | 595 | 425 | 473 |
| 500 | 614 | 1669 | 1718 | 978 | 1028 |

xgb, 500 trees:

| rows | default | native=1 | limit=1 | native=2 | limit=2 |
|---|---|---|---|---|---|
| 1 | 307 | 299 | 349 | 307 | 354 |
| 5 | 540 | 658 | 715 | 578 | 632 |
| 10 | 597 | 712 | 765 | 630 | 684 |
| 25 | 787 | 876 | 929 | 814 | 867 |
| 50 | 1071 | 1158 | 1212 | 1096 | 1147 |
| 100 | 1237 | 1885 | 1941 | 1248 | 1302 |
| 500 | 2116 | 7286 | 7335 | 3881 | 3938 |

Predictions of 720 steps with the forecaster. `current`: skforecast as it is. `outer_limit=1`: one OpenMP thread, set once around the folds. `native=1`: one thread with the parameter of the library. `all_threads`: without the limit of one thread of HistGradientBoosting.

| mode | median s | min s | ms per step | vs current | same predictions |
|---|---|---|---|---|---|
| current | 0.315 | 0.314 | 0.438 | 1.00x | yes |
| no_estimator | 0.059 | 0.059 | 0.082 | 0.19x |  |
| outer_limit=1 | 0.307 | 0.307 | 0.426 | 0.97x | yes |
| native=1 | 0.308 | 0.307 | 0.427 | 0.98x | yes |

xgb, 1000 trees:

| rows | default | native=1 | limit=1 | native=2 | limit=2 |
|---|---|---|---|---|---|
| 1 | 433 | 427 | 473 | 439 | 486 |
| 5 | 952 | 1188 | 1245 | 1028 | 1087 |
| 10 | 1056 | 1287 | 1340 | 1135 | 1187 |
| 25 | 1374 | 1597 | 1659 | 1465 | 1508 |
| 50 | 1929 | 2152 | 2209 | 2003 | 2068 |
| 100 | 2243 | 3593 | 3655 | 2333 | 2382 |
| 500 | 3997 | 14294 | 14449 | 7499 | 7588 |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 16 | first | 35 | 576.5 | 605.4 | 409.0 | 388.2 | 3.84x (bt_refit, lgbm_1t) | 17.88x (bt_refit_small, lgbm) | 1.46x (gs_refit_small, xgb) |
| 16 | next | 34 | 569.4 | 445.6 | 369.9 | 348.7 | 5.20x (bt_refit, lgbm_1t) | 16.29x (bt_refit_small, ridge) | 3.23x (bt_refit, ridge) |
| 8 | first | 23 | 391.6 | 360.1 | 281.4 | 275.0 | 3.21x (bt_refit, lgbm_1t) | 15.98x (bt_refit_small, ridge) | 1.41x (bt_norefit, hgb) |
| 8 | next | 23 | 390.9 | 332.3 | 266.0 | 257.1 | 3.78x (bt_refit, lgbm_1t) | 8.17x (gs_norefit, lgbm) | 3.50x (bt_refit, ridge) |
| 4 | first | 25 | 566.2 | 515.8 | 445.4 | 434.5 | 2.22x (bt_refit, lgbm_1t) | 16.95x (bt_refit_small, ridge) | 1.37x (bt_norefit, hgb) |
| 4 | next | 25 | 565.5 | 466.9 | 422.8 | 411.6 | 2.50x (bt_refit, lgbm_1t) | 4.49x (gs_norefit, lgbm) | 1.67x (bt_refit, ridge) |
