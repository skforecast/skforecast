# Work threshold: results\wsl

Cold start of k workers, median of the repetitions (s):

| payload | 2 | 4 | 8 | 16 | 21 | 22 |
|---|---|---|---|---|---|---|
| forecaster | 1.34 | 1.46 | 1.68 | 2.33 | 2.94 | 3.0 |
| noop | 0.22 | 0.19 | 0.22 | 0.23 | 0.27 | 0.33 |

Single backtestings in a fresh process, median (s). `par`: `cpu_count() - 1` workers; `estimate`: `t_fit * n_fits`.

| case_id | seq | par | t_fit | n_folds | estimate | par/seq |
|---|---|---|---|---|---|---|
| backtest|recursive|lgbm_1|S|refit=False | 0.1 | 1.96 | 0.08 | 10 | 0.0 | 19.76 |
| backtest|recursive|lgbm_1|L|refit=False | 0.37 | 2.65 | 0.27 | 18 | 0.0 | 7.2 |
| backtest|recursive|lgbm_1|S|refit=True | 0.9 | 2.1 | 0.08 | 10 | 0.72 | 2.32 |
| backtest_ms|multiseries|lgbm_1|S|refit=True | 1.21 | 2.13 | 0.12 | 10 | 1.1 | 1.75 |
| backtest|recursive|hgb|L|refit=True | 4.73 | 3.26 | 0.16 | 18 | 2.67 | 0.69 |
| backtest|recursive|lgbm_1|L|refit=True | 5.3 | 3.48 | 0.27 | 18 | 4.61 | 0.66 |
| backtest|direct_fix|lgbm_1|S|refit=True | 19.42 | 6.35 | 1.69 | 10 | 15.19 | 0.33 |
| backtest_ms|multiseries|lgbm_1|L|refit=True | 28.87 | 8.22 | 1.13 | 20 | 21.46 | 0.28 |
| backtest|recursive|rf_none|L|refit=True | 48.02 | 9.96 | 2.11 | 18 | 35.9 | 0.21 |

## Threshold

- Line fitted on 6 cases with refit: parallel = 2.31 + 0.172 x sequential (cold overhead 2.31 s, speedup 5.8).
- Break-even: 2.79 s of real sequential work.
- Estimate / real sequential time: median 0.77 [0.57-0.91].
- Threshold in units of the estimate: 2.15 s, rounded to **2.0 s**.
- Bracket from the cases: parallel loses up to an estimate of 1.10 s and wins from 2.67 s.

## 'auto' with the threshold of the installed skforecast

No 'auto' measured with a skforecast that has the threshold (no `auto_resolved`).
