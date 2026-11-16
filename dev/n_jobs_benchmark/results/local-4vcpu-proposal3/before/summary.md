# Benchmark of n_jobs in skforecast

- Machine: Linux-6.18.44-fc-v445-x86_64-with-glibc2.39, 4 logical CPUs, 4 physical cores, 16.5 GB.
- Python 3.12.3; skforecast 0.28.0, numpy 2.5.4, pandas 2.3.3, sklearn 1.9.1, scipy 1.18.1, joblib 1.6.0, threadpoolctl 3.8.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, numba 0.68.1, optuna 5.0.0.
- skforecast commit: 15c7be63bf76f2b55f4a00135f71c4878507c0fd.
- Measures: 11 finished, 0 failed or timed out.
- Workers: 11 timed measures, 6 in parallel. In all of them the first call started its workers and the following calls reused the same ones. In the 0 measures of the start of the workers, they saw the CPUs of their machine and the threads that joblib gives them.
- CPU in use by other processes before each measure: median 0.000, maximum 0.010 (fraction of the machine).

Times in seconds, median of the processes. `first`: first call of a new process (it pays the start of the workers). `next`: median of the following calls (workers already started). `seq`: `n_jobs=1`. `par`: `n_jobs=cpu_count() - 1`.

## 4 CPUs

joblib sees 4 CPUs (4 physical cores). Thread pools: [['openmp', 'libgomp.so.1.0.0', 4], ['blas', 'libscipy_openblas-5f890258.so', 4], ['openmp', 'libgomp-e985bcbb.so.1.0.0', 4], ['blas', 'libscipy_openblas64_-f48b354e.so', 4]]. First use of the window features in a process: 0.00 s.

### Backtesting without refit (100 folds of 24 steps)

| estimator | processes seq/par | seq first | seq next | par first | par next | speedup next | start cost | break-even | rule | auto first/next | auto vs best first | auto vs best next | estimate / real | fit after par / seq | model MB | workers peak MB | CPU s seq next | CPU s par next | same result | other configs (first/next) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 1/1 | 3.25 | 3.09 | 4.52 | 2.66 | 1.16x | 1.86 | 13.44 | 3 | 1/1 | 1.00x | 1.16x |  | 0.99x | 0.6 | 683 | 12.0 | 10.3 | yes |  half: 4.69/2.66 (n_jobs=2) |
| hgb | 1/1 | 12.85 | 13.00 | 7.76 | 6.02 | 2.16x | 1.75 | 3.25 | 3 | 1/1 (measured) | 1.74x | 2.07x |  | 0.97x | 1.9 | 658 | 18.2 | 19.6 | yes |  half: 10.01/8.36 (n_jobs=2) |
| xgb | 1/1 | 3.49 | 3.64 | 5.15 | 3.23 | 1.13x | 1.92 | 17.11 | 1 | 1/1 (measured) | 1.02x | 1.00x |  | 1.01x | 2.0 | 694 | 14.5 | 12.0 | yes |  half: 4.99/3.23 (n_jobs=2) |

## Policies

Sum of the times of the cases measured with `n_jobs=1` and `n_jobs=cpu_count() - 1`, with each policy, and worst case against the best of the two for each case. `auto`: what skforecast chooses.

| CPUs | call | cases | always 1 | always parallel | auto | best | worst always 1 | worst always parallel | worst auto |
|---|---|---|---|---|---|---|---|---|---|
| 4 | first | 3 | 19.6 | 17.4 | 20.3 | 14.5 | 1.66x (bt_norefit, hgb) | 1.48x (bt_norefit, xgb) | 1.74x (bt_norefit, hgb) |
| 4 | next | 3 | 19.7 | 11.9 | 18.8 | 11.9 | 2.16x (bt_norefit, hgb) | 1.00x (bt_norefit, catboost) | 2.07x (bt_norefit, hgb) |
