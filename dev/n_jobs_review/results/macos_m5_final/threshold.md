# Work threshold: results/macos_m5_final

No block_A_cold.csv.

Single backtestings in a fresh process, median (s). `par`: `cpu_count() - 1` workers; `estimate`: `t_fit * n_fits`.

| case_id | seq | par | auto | auto_jobs | t_fit | n_folds | estimate | par/seq | auto/seq |
|---|---|---|---|---|---|---|---|---|---|
| backtest|recursive|lgbm_1|L|refit=False | 0.21 | 1.83 | 0.2 | 1 | 0.16 | 18 | 0.0 | 8.68 | 0.96 |
| backtest|recursive|lgbm_1|S|refit=False | 0.07 | 1.68 | 0.07 | 1 | 0.05 | 10 | 0.0 | 23.22 | 0.96 |
| backtest|recursive|lgbm_1|S|refit=True | 0.52 | 1.84 | 0.54 | 1 | 0.05 | 10 | 0.48 | 3.52 | 1.03 |
| backtest_ms|multiseries|lgbm_1|S|refit=True | 0.73 | 1.89 | 0.7 | 1 | 0.08 | 10 | 0.72 | 2.6 | 0.96 |
| backtest|recursive|lgbm_1|L|refit=True | 2.65 | 2.39 | 2.43 | 9 | 0.15 | 18 | 2.54 | 0.9 | 0.92 |
| backtest|direct_fix|lgbm_1|S|refit=True | 11.53 | 4.16 | 4.18 | 1 | 1.02 | 10 | 9.18 | 0.36 | 0.36 |
| backtest_ms|multiseries|lgbm_1|L|refit=True | 14.49 | 5.91 | 6.3 | 9 | 0.61 | 20 | 11.68 | 0.41 | 0.44 |
| backtest|recursive|hgb|L|refit=True | 18.1 | 3.43 | 3.41 | 9 | 0.9 | 18 | 15.34 | 0.19 | 0.19 |

## Threshold

- Line fitted on 5 cases with refit: parallel = 1.96 + 0.156 x sequential (cold overhead 1.96 s, speedup 6.4).
- Break-even: 2.32 s of real sequential work.
- Estimate / real sequential time: median 0.92 [0.81-0.99].
- Threshold in units of the estimate: 2.13 s, rounded to **2.0 s**.
- Bracket from the cases: parallel loses up to an estimate of 0.72 s and wins from 2.54 s.

Quick estimate from block A, for comparison:

No block_A_cold.csv: cannot estimate the threshold.

## 'auto' with the threshold of the installed skforecast

- 'auto' / best of 1 and cpu-1: median 1.00, worst 1.07 (backtest_ms|multiseries|lgbm_1|L|refit=True).
- 'auto' / n_jobs=1: worst 1.03.
