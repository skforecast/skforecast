# Work threshold: /home/joaquin/github/skforecast_njobs_wsl/dev/n_jobs_review/results/linux_wsl_d4

Cold start of k workers, median of the repetitions (s):

| payload | 2 | 4 | 8 | 16 | 21 | 22 |
|---|---|---|---|---|---|---|
| forecaster | 1.56 | 1.43 | 1.78 | 2.41 | 3.17 | 3.11 |
| noop | 0.24 | 0.23 | 0.26 | 0.3 | 0.35 | 0.33 |

Single backtestings in a fresh process, median (s). `par`: `cpu_count() - 1` workers; `estimate`: `t_fit * n_fits`.

| case_id | seq | par | auto | auto_jobs | t_fit | n_folds | estimate | par/seq | auto/seq |
|---|---|---|---|---|---|---|---|---|---|
| backtest|recursive|lgbm_1|S|refit=False | 0.11 | 2.24 | 0.14 | 1 | 0.1 | 10 | 0.0 | 19.97 | 1.27 |
| backtest|recursive|lgbm_1|L|refit=False | 0.64 | 2.9 | 0.35 | 1 | 0.49 | 18 | 0.0 | 4.52 | 0.55 |
| backtest|recursive|lgbm_1|S|refit=True | 0.84 | 2.7 | 1.23 | 1 | 0.08 | 10 | 0.71 | 3.22 | 1.47 |
| backtest_ms|multiseries|lgbm_1|S|refit=True | 1.22 | 2.48 | 2.2 | 1 | 0.12 | 10 | 1.1 | 2.04 | 1.81 |
| backtest|recursive|lgbm_1|L|refit=True | 4.33 | 4.01 | 3.96 | 18 | 0.25 | 18 | 4.26 | 0.93 | 0.91 |
| backtest|recursive|hgb|L|refit=True | 17.09 | 5.05 | 4.96 | 18 | 0.96 | 18 | 16.3 | 0.3 | 0.29 |
| backtest|direct_fix|lgbm_1|S|refit=True | 31.78 | 8.65 | 9.3 | 1 | 2.76 | 10 | 24.84 | 0.27 | 0.29 |
| backtest|recursive|rf_none|L|refit=True | 50.48 | 11.4 | 10.8 | 18 | 1.93 | 18 | 32.74 | 0.23 | 0.21 |
| backtest_ms|multiseries|lgbm_1|L|refit=True | 44.99 | 8.45 | 8.55 | 20 | 1.98 | 20 | 37.67 | 0.19 | 0.19 |

## Threshold

- Line fitted on 6 cases with refit: parallel = 2.63 + 0.154 x sequential (cold overhead 2.63 s, speedup 6.5).
- Break-even: 3.11 s of real sequential work.
- Estimate / real sequential time: median 0.87 [0.65-0.98].
- Threshold in units of the estimate: 2.72 s, rounded to **2.5 s**.
- Bracket from the cases: parallel loses up to an estimate of 1.10 s and wins from 4.26 s.

## 'auto' with the threshold of the installed skforecast

- 'auto' / best of 1 and cpu-1: median 1.01, worst 1.81 (backtest_ms|multiseries|lgbm_1|S|refit=True).
- 'auto' / n_jobs=1: worst 1.81.
