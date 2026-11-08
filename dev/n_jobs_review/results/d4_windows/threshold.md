# Work threshold: results\d4_windows

No block_A_cold.csv.

Single backtestings in a fresh process, median (s). `par`: `cpu_count() - 1` workers; `estimate`: `t_fit * n_fits`.

| case_id | seq | par | auto | auto_jobs | t_fit | n_folds | estimate | par/seq | auto/seq |
|---|---|---|---|---|---|---|---|---|---|
| backtest|recursive|lgbm_1|S|refit=False | 0.3 | 4.11 | 0.3 | 1 | 0.2 | 10 | 0.0 | 13.62 | 1.0 |
| backtest|recursive|lgbm_1|L|refit=False | 0.78 | 5.88 | 0.86 | 1 | 0.54 | 18 | 0.0 | 7.52 | 1.1 |
| backtest|recursive|lgbm_1|S|refit=True | 2.05 | 4.32 | 2.03 | 1 | 0.2 | 10 | 1.77 | 2.11 | 0.99 |
| backtest_ms|multiseries|lgbm_1|S|refit=True | 2.47 | 4.39 | 2.55 | 1 | 0.25 | 10 | 2.26 | 1.77 | 1.03 |
| backtest|recursive|hgb|L|refit=True | 8.55 | 7.03 | 6.75 | 18 | 0.25 | 18 | 4.17 | 0.82 | 0.79 |
| backtest|recursive|lgbm_1|L|refit=True | 10.91 | 6.57 | 6.91 | 18 | 0.52 | 18 | 8.89 | 0.6 | 0.63 |
| backtest|direct_fix|lgbm_1|S|refit=True | 42.2 | 11.52 | 11.37 | 1 | 3.22 | 10 | 28.98 | 0.27 | 0.27 |
| backtest_ms|multiseries|lgbm_1|L|refit=True | 45.75 | 14.78 | 13.48 | 20 | 1.74 | 20 | 33.05 | 0.32 | 0.29 |
| backtest|recursive|rf_none|L|refit=True | 80.16 | 15.42 | 16.26 | 18 | 3.57 | 18 | 60.65 | 0.19 | 0.2 |

## Threshold

- Line fitted on 6 cases with refit: parallel = 4.98 + 0.151 x sequential (cold overhead 4.98 s, speedup 6.6).
- Break-even: 5.86 s of real sequential work.
- Estimate / real sequential time: median 0.79 [0.49-0.91].
- Threshold in units of the estimate: 4.61 s, rounded to **4.5 s**.
- Bracket from the cases: parallel loses up to an estimate of 2.26 s and wins from 4.17 s.

## 'auto' with the threshold of the installed skforecast

- 'auto' / best of 1 and cpu-1: median 1.00, worst 1.10 (backtest|recursive|lgbm_1|L|refit=False).
- 'auto' / n_jobs=1: worst 1.10.
