# Work threshold: /Users/javier.escobar/code/github/skforecast/dev/n_jobs_review/results/macos_m5

Cold start of k workers, median of the repetitions (s):

| payload | 2 | 4 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| forecaster | 1.07 | 1.17 | 1.53 | 1.66 | 1.82 |
| noop | 0.14 | 0.15 | 0.2 | 0.21 | 0.23 |

Single backtestings in a fresh process, median (s). `par`: `cpu_count() - 1` workers; `estimate`: `t_fit * n_fits`.

| case_id | seq | par | auto | auto_jobs | t_fit | n_folds | estimate | par/seq | auto/seq |
|---|---|---|---|---|---|---|---|---|---|
| backtest|recursive|lgbm_1|L|refit=False | 0.21 | 1.82 | 0.19 | 1 | 0.16 | 18 | 0.0 | 8.88 | 0.94 |
| backtest|recursive|lgbm_1|S|refit=False | 0.07 | 1.72 | 0.07 | 1 | 0.05 | 10 | 0.0 | 24.46 | 1.04 |
| backtest|recursive|lgbm_1|S|refit=True | 0.5 | 1.79 | 0.5 | 1 | 0.05 | 10 | 0.44 | 3.6 | 1.01 |
| backtest_ms|multiseries|lgbm_1|S|refit=True | 0.68 | 1.83 | 0.71 | 1 | 0.07 | 10 | 0.67 | 2.7 | 1.05 |
| backtest|recursive|lgbm_1|L|refit=True | 2.62 | 2.32 | 2.57 | 1 | 0.15 | 18 | 2.51 | 0.89 | 0.98 |
| backtest|recursive|hgb|L|refit=True | 7.1 | 6.64 | 7.14 | 1/9 | 0.2 | 18 | 3.33 | 0.94 | 1.01 |
| backtest|direct_fix|lgbm_1|S|refit=True | 10.71 | 4.15 | 4.1 | 1 | 1.02 | 10 | 9.14 | 0.39 | 0.38 |
| backtest_ms|multiseries|lgbm_1|L|refit=True | 14.84 | 5.91 | 6.04 | 9 | 0.59 | 20 | 11.29 | 0.4 | 0.41 |
| backtest|recursive|rf_none|L|refit=True | 32.78 | 8.41 | 8.42 | 9 | 1.46 | 18 | 24.82 | 0.26 | 0.26 |

## Threshold

- Line fitted on 6 cases with refit: parallel = 2.54 + 0.200 x sequential (cold overhead 2.54 s, speedup 5.0).
- Break-even: 3.17 s of real sequential work.
- Estimate / real sequential time: median 0.82 [0.47-0.98].
- Threshold in units of the estimate: 2.60 s, rounded to **2.5 s**.
- Bracket from the cases: parallel loses up to an estimate of 0.67 s and wins from 2.51 s.

Quick estimate from block A, for comparison:

- Cold start of 9 workers (`cpu_count() - 1`) with the forecaster: 1.66 s, median of 10 processes.
- Quick threshold: 0.7 x 1.66 = 1.16 s, rounded to **1.0 s**.

## 'auto' with the threshold of the installed skforecast

- 'auto' / best of 1 and cpu-1: median 1.02, worst 1.11 (backtest|recursive|lgbm_1|L|refit=True).
- 'auto' / n_jobs=1: worst 1.05.
