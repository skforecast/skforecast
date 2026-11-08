# Work threshold: results

Cold start of k workers, median of the repetitions (s):

| payload | 2 | 4 | 8 | 16 | 21 | 22 |
|---|---|---|---|---|---|---|
| forecaster | 2.2 | 2.65 | 3.02 | 4.19 | 5.68 | 5.96 |
| noop | 0.42 | 0.45 | 0.5 | 0.63 | 0.66 | 0.7 |

Single backtestings in a fresh process, median (s). `par`: `cpu_count() - 1` workers; `estimate`: `t_fit * n_fits`.

| case_id | seq | par | t_fit | n_folds | estimate | par/seq |
|---|---|---|---|---|---|---|
| backtest|recursive|lgbm_1|S|refit=False | 0.18 | 3.5 | 0.09 | 10 | 0.0 | 19.77 |
| backtest|recursive|lgbm_1|L|refit=False | 0.55 | 4.77 | 0.31 | 18 | 0.0 | 8.7 |
| backtest|recursive|lgbm_1|S|refit=True | 1.07 | 3.86 | 0.1 | 10 | 0.86 | 3.61 |
| backtest_ms|multiseries|lgbm_1|S|refit=True | 1.56 | 3.9 | 0.15 | 10 | 1.33 | 2.5 |
| backtest|recursive|hgb|L|refit=True | 7.08 | 5.72 | 0.22 | 18 | 3.76 | 0.81 |
| backtest|recursive|lgbm_1|L|refit=True | 5.77 | 5.73 | 0.31 | 18 | 5.25 | 0.99 |
| backtest|direct_fix|lgbm_1|S|refit=True | 23.02 | 10.13 | 2.12 | 10 | 19.04 | 0.44 |
| backtest_ms|multiseries|lgbm_1|L|refit=True | 29.6 | 10.53 | 1.15 | 20 | 21.76 | 0.36 |
| backtest|recursive|rf_none|L|refit=True | 48.02 | 12.21 | 2.14 | 18 | 36.44 | 0.25 |

## Threshold

- Line fitted on 6 cases with refit: parallel = 4.18 + 0.181 x sequential (cold overhead 4.18 s, speedup 5.5).
- Break-even: 5.10 s of real sequential work.
- Estimate / real sequential time: median 0.78 [0.53-0.91].
- Threshold in units of the estimate: 4.00 s, rounded to **4.0 s**.
- Bracket from the cases: parallel loses up to an estimate of 1.33 s and wins from 3.76 s.

## 'auto' with the threshold of the installed skforecast

No 'auto' measured with a skforecast that has the threshold (no `auto_resolved`).
