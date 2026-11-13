# Benchmark of n_jobs in skforecast

- Machine: Linux-6.18.44-fc-v445-x86_64-with-glibc2.39, 4 logical CPUs, 4 physical cores, 16.5 GB.
- Python 3.12.3; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: 13ce144d3066235553523131297a7d6d3509465c.
- Measures: 66 finished, 0 failed or timed out.
- Workers: 66 timed measures, 39 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.029 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 4 CPUs

joblib sees 4 CPUs (4 physical cores). Thread pools: [['openmp', 'libgomp-e985bcbb.so.1.0.0', 4], ['blas', 'libscipy_openblas64_-f48b354e.so', 4], ['openmp', 'libgomp.so.1.0.0', 4], ['blas', 'libscipy_openblas-5f890258.so', 4]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 10.43 | 11.49 | 7.56 | 5.53 | 2.08x | 2.03 | 3.91 | 3 | 1/1 (measured) | 1.55x | 2.00x |  | 1.11x | 1.9 | 659 | 16.4 | 18.4 | yes |  half: 8.65/7.23 (n_jobs=2) |
| lgbm_1t | 1/1 | 1.92 | 2.41 | 4.26 | 2.16 | 1.12x | 2.10 | 20.08 | 1 | 1/1 | 1.00x | 1.12x |  | 0.99x | 1.4 | 649 | 2.4 | 3.3 | yes |  |
| rf | 1/1 | 14.47 | 14.90 | 22.57 | 17.75 | 0.84x | 4.82 |  | 1 | 1/1 | 1.00x | 1.00x |  | 1.08x | 40.4 | 1091 | 15.0 | 28.7 | yes |  |
| ridge | 1/1 | 0.18 | 0.12 | 1.67 | 0.10 | 1.16x | 1.57 | 11.08 | 1 | 1/1 | 1.00x | 1.16x |  | 1.27x | 0.0 | 613 | 0.1 | 0.3 | yes |  |
| xgb | 1/1 | 3.09 | 3.10 | 4.14 | 2.76 | 1.13x | 1.38 | 12.40 | 1 | 1/1 (measured) | 1.18x | 1.11x |  | 1.00x | 2.0 | 693 | 12.4 | 10.2 | yes |  half: 5.03/3.17 (n_jobs=2) |

### Backtesting with refit (16 folds, 15 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 43.24 | 43.96 | 19.30 | 19.20 | 2.29x | 0.09 | 0.17 | 3 | 3/3 (measured) | 1.07x | 0.97x | 0.66 | 1.04x | 1.9 | 689 | 127.7 | 59.8 | yes |  all: 16.84/14.79 (n_jobs=4) half: 26.02/23.92 (n_jobs=2) |
| lgbm_1t | 1/1 | 26.66 | 25.79 | 12.69 | 10.61 | 2.43x | 2.08 | 3.53 | 3 | 3/3 | 1.00x | 1.00x | 0.85 | 0.77x | 1.4 | 696 | 25.9 | 28.0 | yes |  all: 10.88/8.89 (n_jobs=4) half: 17.37/15.03 (n_jobs=2) |
| ridge | 1/1 | 0.23 | 0.18 | 1.72 | 0.11 | 1.67x | 1.61 | 4.01 | 1 | 1/1 | 1.00x | 1.67x | 0.98 | 1.83x | 0.0 | 641 | 0.2 | 0.3 | yes |  |
| xgb | 1/1 | 33.40 | 33.61 | 28.24 | 28.74 | 1.17x | -0.50 |  | 3 | 3/3 (measured) | 1.00x | 0.87x | 0.94 | 0.88x | 2.0 | 757 | 133.8 | 85.3 | yes |  all: 22.63/21.67 (n_jobs=4) half: 27.53/26.39 (n_jobs=2) |

### Backtesting with refit, few folds (6 folds, 5 refits)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 14.87 | 14.55 | 9.40 | 8.09 | 1.80x | 1.31 | 2.94 | 3 | 3/3 (measured) | 1.10x | 1.00x | 0.63 | 0.98x | 1.9 | 685 | 45.5 | 22.8 | yes |  |
| lgbm_1t | 1/1 | 8.51 | 10.47 | 6.67 | 5.23 | 2.00x | 1.44 | 2.87 | 3 | 3/3 (measured) | 0.99x | 0.99x | 1.02 | 0.85x | 1.4 | 695 | 10.5 | 11.1 | yes |  |
| rf | 1/1 | 69.40 | 69.59 | 37.47 | 33.71 | 2.06x | 3.76 | 7.30 | 3 | 3/3 | 1.00x | 1.00x | 0.85 | 0.95x | 40.4 | 1105 | 70.1 | 71.2 | yes |  |
| ridge | 1/1 | 0.10 | 0.08 | 1.66 | 0.06 | 1.27x | 1.60 | 7.55 | 1 | 1/1 | 1.00x | 1.27x | 0.77 | 1.25x | 0.0 | 636 | 0.1 | 0.1 | yes |  |
| xgb | 1/1 | 13.20 | 12.66 | 13.79 | 11.92 | 1.06x | 1.87 | 31.90 | 3 | 3/3 (measured) | 1.08x | 0.98x | 0.81 | 1.07x | 2.0 | 756 | 50.3 | 34.1 | yes |  |

### Grid search without refit (3 candidates, 40 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 18.84 | 16.55 | 11.99 | 10.23 | 1.62x | 1.77 | 4.63 | 3 | 3/3 (measured) | 1.04x | 1.01x |  | 1.05x | 1.9 | 659 | 33.2 | 35.5 | yes |  half: 13.49/12.77 (n_jobs=2) |
| lgbm_1t | 1/1 | 5.46 | 5.17 | 7.48 | 6.19 | 0.84x | 1.29 |  | 1 | 1/1 (measured) | 0.86x | 1.08x |  | 1.10x | 1.4 | 650 | 5.2 | 7.9 | yes |  |
| rf | 1/1 | 32.91 | 32.00 | 38.57 | 35.40 | 0.90x | 3.17 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.99x | 40.4 | 1091 | 32.2 | 43.2 | yes |  |
| ridge | 1/1 | 0.17 | 0.17 | 1.60 | 0.18 | 0.94x | 1.42 |  | 1 | 1/1 | 1.00x | 1.00x |  | 0.73x | 0.0 | 612 | 0.1 | 0.4 | yes |  |
| xgb | 1/1 | 7.51 | 8.12 | 8.78 | 7.26 | 1.12x | 1.52 | 14.41 | 1 | 1/1 (measured) | 1.08x | 1.04x |  | 0.86x | 2.0 | 695 | 32.4 | 27.5 | yes |  half: 9.53/7.68 (n_jobs=2) |

### Grid search with refit (3 candidates, 6 folds each)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hgb | 1/1 | 45.34 | 46.80 | 26.36 | 24.43 | 1.92x | 1.93 | 4.04 | 3 | 3/3 | 1.00x | 1.00x |  | 1.01x | 1.9 | 686 | 147.4 | 70.9 | yes |  |
| lgbm_1t | 1/1 | 30.09 | 32.56 | 17.54 | 16.83 | 1.93x | 0.71 | 1.46 | 3 | 3/3 | 1.00x | 1.00x |  | 1.21x | 1.4 | 695 | 32.7 | 35.4 | yes |  |
| ridge | 1/1 | 0.21 | 0.19 | 1.74 | 0.17 | 1.06x | 1.57 | 25.80 | 1 | 1/1 | 1.00x | 1.06x |  | 1.70x | 0.0 | 636 | 0.2 | 0.3 | yes |  |
| xgb | 1/1 | 38.35 | 36.98 | 40.93 | 37.82 | 0.98x | 3.12 |  | 3 | 3/3 | 1.07x | 1.02x |  | 0.91x | 2.0 | 765 | 147.5 | 107.2 | yes |  |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 4 | first | 23 | 418.6 | 326.1 | 306.0 | 295.0 | 2.24x (bt_refit, hgb) | 15.97x (bt_refit_small, ridge) | 1.55x (bt_norefit, hgb) |
| 4 | next | 23 | 421.5 | 284.5 | 279.6 | 276.4 | 2.43x (bt_refit, lgbm_1t) | 1.20x (gs_norefit, lgbm_1t) | 2.00x (bt_norefit, hgb) |
