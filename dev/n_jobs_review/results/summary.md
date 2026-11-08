# Resumen de medidas

## Bloque A: coste fijo de arrancar workers

| affinity | mode | payload | 2 | 3 | 4 | 7 | 8 | 16 | 21 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | cold | few | 0.42 [0.40-0.48] | 0.52 [0.48-0.55] | 0.59 [0.57-0.61] |  |  |  |  |  |
| 4 | cold | forecaster | 2.28 [2.16-2.35] | 2.86 [2.75-2.98] | 3.27 [3.21-3.46] |  |  |  |  |  |
| 4 | cold | noop | 0.44 [0.42-0.44] | 0.51 [0.47-0.52] | 0.58 [0.56-0.62] |  |  |  |  |  |
| 4 | warm | few | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] |  |  |  |  |  |
| 4 | warm | forecaster | 0.03 [0.02-0.03] | 0.03 [0.03-0.04] | 0.04 [0.04-0.05] |  |  |  |  |  |
| 4 | warm | noop | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] |  |  |  |  |  |
| 8 | cold | few | 0.44 [0.41-0.50] |  | 0.49 [0.48-0.52] | 0.62 [0.55-0.66] | 0.66 [0.62-0.67] |  |  |  |
| 8 | cold | forecaster | 2.37 [2.26-2.56] |  | 2.92 [2.77-3.00] | 3.94 [3.86-4.33] | 4.28 [4.09-4.35] |  |  |  |
| 8 | cold | noop | 0.46 [0.39-0.49] |  | 0.48 [0.46-0.52] | 0.63 [0.60-0.67] | 0.67 [0.63-0.74] |  |  |  |
| 8 | warm | few | 0.01 [0.01-0.01] |  | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] |  |  |  |
| 8 | warm | forecaster | 0.02 [0.02-0.06] |  | 0.04 [0.04-0.06] | 0.05 [0.05-0.05] | 0.08 [0.07-1.80] |  |  |  |
| 8 | warm | noop | 0.01 [0.01-0.01] |  | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] |  |  |  |
| 22 | cold | few | 0.42 [0.36-0.46] |  | 0.45 [0.44-0.48] |  | 0.51 [0.47-0.53] | 0.64 [0.57-0.67] | 0.68 [0.62-0.76] | 0.67 [0.63-0.73] |
| 22 | cold | forecaster | 2.20 [2.13-2.53] |  | 2.65 [2.62-2.88] |  | 3.02 [2.95-3.12] | 4.19 [4.10-4.50] | 5.68 [5.28-5.72] | 5.96 [5.91-6.43] |
| 22 | cold | noop | 0.42 [0.40-0.45] |  | 0.45 [0.44-0.47] |  | 0.50 [0.48-0.56] | 0.63 [0.59-0.69] | 0.66 [0.60-0.69] | 0.70 [0.63-0.74] |
| 22 | warm | few | 0.01 [0.01-0.01] |  | 0.01 [0.01-0.01] |  | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] | 0.01 [0.01-0.02] |
| 22 | warm | forecaster | 0.03 [0.03-0.03] |  | 0.05 [0.05-0.06] |  | 0.10 [0.09-0.10] | 0.13 [0.13-0.16] | 0.16 [0.16-0.17] | 0.17 [0.16-0.18] |
| 22 | warm | noop | 0.01 [0.01-0.01] |  | 0.01 [0.01-0.01] |  | 0.01 [0.01-0.01] | 0.01 [0.01-0.01] | 0.01 [0.01-0.02] | 0.01 [0.01-0.01] |

## Bloques B a G


### Bloque B, afinidad 4 (cpu_count 4)


`backtest|recursive|hgb|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.06 [1.96-2.07] | 1.00 | 2.04 |
| **cpu-1** | 3 | 0.91 [0.89-0.91] | 2.27 | 3.97 |
| -1 | 4 | 0.97 [0.89-1.06] | 2.13 | 2.80 |
| physical | 2 | 1.37 [1.32-1.52] | 1.50 | 1.49 |

`backtest|recursive|hgb|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 5.55 [5.50-5.70] | 1.00 | 5.78 |
| cpu-1 | 3 | 3.08 [2.99-3.20] | 1.80 | 3.87 |
| **-1** | 4 | 2.88 [2.83-3.11] | 1.92 | 4.11 |
| physical | 2 | 4.05 [4.00-4.14] | 1.37 | 7.08 |

`backtest|recursive|lgbm_1|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.44 [0.43-0.46] | 1.00 | 0.48 |
| cpu-1 | 3 | 0.42 [0.41-0.43] | 1.04 | 0.62 |
| **-1** | 4 | 0.42 [0.41-2.01] | 1.04 | 0.46 |
| physical | 2 | 0.47 [0.45-0.48] | 0.92 | 0.43 |

`backtest|recursive|lgbm_1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 5.60 [5.54-5.72] | 1.00 | 6.35 |
| cpu-1 | 3 | 2.97 [2.92-3.06] | 1.89 | 3.20 |
| **-1** | 4 | 2.93 [2.91-2.95] | 1.91 | 6.20 |
| physical | 2 | 3.52 [3.38-3.70] | 1.59 | 6.11 |

`backtest|recursive|ridge|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.02 [0.02-0.03] | 1.00 | 0.04 |
| cpu-1 | 3 | 0.03 [0.03-0.04] | 0.78 | 0.03 |
| -1 | 4 | 0.04 [0.03-0.05] | 0.56 | 3.36 |
| physical | 2 | 0.04 [0.03-0.04] | 0.64 | 0.04 |

`backtest|recursive|ridge|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.16 [0.15-0.18] | 1.00 | 0.15 |
| cpu-1 | 3 | 0.08 [0.07-0.08] | 2.06 | 0.36 |
| **-1** | 4 | 0.07 [0.07-0.08] | 2.15 | 3.49 |
| physical | 2 | 0.09 [0.08-0.10] | 1.76 | 2.53 |

### Bloque B, afinidad 8 (cpu_count 8)


`backtest|recursive|hgb|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.98 [2.80-4.66] | 1.00 | 2.89 |
| cpu-1 | 7 | 1.02 [1.00-1.04] | 2.92 | 5.57 |
| **-1** | 8 | 0.79 [0.74-2.62] | 3.79 | 1.14 |
| physical | 4 | 1.67 [1.62-1.99] | 1.78 | 2.11 |

`backtest|recursive|hgb|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 7.29 [7.12-7.33] | 1.00 | 7.77 |
| **cpu-1** | 7 | 2.26 [2.11-2.36] | 3.22 | 2.77 |
| -1 | 8 | 2.92 [2.77-3.43] | 2.49 | 3.45 |
| physical | 4 | 3.13 [3.00-3.27] | 2.33 | 6.66 |

`backtest|recursive|lgbm_1|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.51 [0.46-0.55] | 1.00 | 0.45 |
| cpu-1 | 7 | 0.48 [0.47-0.52] | 1.06 | 0.74 |
| **-1** | 8 | 0.41 [0.41-2.18] | 1.24 | 0.45 |
| physical | 4 | 0.62 [0.56-0.73] | 0.83 | 0.60 |

`backtest|recursive|lgbm_1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 5.87 [5.45-6.00] | 1.00 | 6.76 |
| **cpu-1** | 7 | 2.09 [1.99-2.16] | 2.80 | 2.27 |
| -1 | 8 | 2.09 [2.01-2.28] | 2.80 | 6.37 |
| physical | 4 | 2.48 [2.37-2.49] | 2.36 | 5.85 |

`backtest|recursive|ridge|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.03 [0.03-0.03] | 1.00 | 0.03 |
| cpu-1 | 7 | 0.04 [0.04-0.05] | 0.61 | 0.05 |
| -1 | 8 | 0.04 [0.03-0.05] | 0.74 | 4.42 |
| physical | 4 | 0.03 [0.03-0.03] | 0.90 | 0.03 |

`backtest|recursive|ridge|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.14 [0.13-0.15] | 1.00 | 0.13 |
| **cpu-1** | 7 | 0.06 [0.06-0.07] | 2.16 | 0.28 |
| -1 | 8 | 0.08 [0.08-0.10] | 1.74 | 4.92 |
| physical | 4 | 0.09 [0.08-0.09] | 1.61 | 3.65 |

### Bloque B, afinidad 22 (cpu_count 22)


`backtest|classifier|hgbc|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.74 [2.70-2.90] | 1.00 | 2.65 |
| **auto** | 21 | 1.34 [1.26-1.38] | 2.04 | 1.34 |
| cpu-1 | 21 | 1.41 [1.37-1.46] | 1.94 | 1.41 |
| -1 | 22 | 1.42 [1.39-1.44] | 1.93 | 1.40 |
| physical | 16 | 1.50 [1.35-1.57] | 1.83 | 3.91 |

`backtest|classifier|hgbc|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 9.97 [9.69-10.29] | 1.00 | 9.69 |
| **auto** | 21 | 2.08 [1.93-2.18] | 4.79 | 2.11 |
| cpu-1 | 21 | 2.10 [2.00-4.24] | 4.74 | 2.26 |
| -1 | 22 | 2.21 [2.15-3.69] | 4.51 | 2.40 |
| physical | 16 | 2.22 [2.19-2.25] | 4.49 | 3.32 |

`backtest|classifier|lgbmc_1|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.81 [1.74-1.90] | 1.00 | 1.87 |
| **auto** | 21 | 0.96 [0.88-2.93] | 1.88 | 1.06 |
| cpu-1 | 21 | 1.00 [0.96-3.08] | 1.82 | 1.14 |
| -1 | 22 | 1.04 [1.01-2.77] | 1.75 | 1.11 |
| physical | 16 | 1.06 [1.02-1.12] | 1.70 | 2.34 |

`backtest|classifier|lgbmc_1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 7.80 [7.69-8.15] | 1.00 | 8.31 |
| auto | 21 | 1.82 [1.81-1.96] | 4.29 | 1.82 |
| cpu-1 | 21 | 1.84 [1.80-3.96] | 4.24 | 7.25 |
| **-1** | 22 | 1.72 [1.67-3.05] | 4.53 | 1.65 |
| physical | 16 | 1.93 [1.81-2.00] | 4.04 | 3.77 |
| cap8 | 8 | 2.50 [2.44-2.62] | 3.12 | 7.14 |

`backtest|classifier|lgbmc_1|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.21 [0.20-0.25] | 1.00 | 0.20 |
| auto | 21 | 0.37 [0.35-3.23] | 0.56 | 5.25 |
| cpu-1 | 21 | 0.41 [0.39-2.95] | 0.52 | 0.57 |
| -1 | 22 | 2.39 [0.33-2.86] | 0.09 | 2.07 |
| physical | 16 | 0.37 [0.35-0.39] | 0.57 | 1.83 |

`backtest|classifier|lgbmc_1|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.34 [1.29-1.38] | 1.00 | 1.32 |
| auto | 21 | 0.60 [0.58-0.62] | 2.23 | 0.86 |
| cpu-1 | 21 | 0.66 [0.60-2.18] | 2.04 | 0.90 |
| **-1** | 22 | 0.59 [0.54-0.63] | 2.28 | 0.57 |
| physical | 16 | 0.64 [0.59-2.55] | 2.11 | 3.21 |

`backtest|classifier|logreg|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.61 [0.58-0.63] | 1.00 | 0.64 |
| auto | 1 | 0.60 [0.58-0.62] | 1.01 | 0.70 |
| cpu-1 | 21 | 0.35 [0.33-0.35] | 1.77 | 0.56 |
| -1 | 22 | 0.34 [0.33-2.95] | 1.78 | 2.29 |
| **physical** | 16 | 0.31 [0.30-0.34] | 1.97 | 1.62 |

`backtest|classifier|logreg|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 4.61 [4.12-4.77] | 1.00 | 4.40 |
| auto | 1 | 4.64 [4.29-4.77] | 0.99 | 4.21 |
| **cpu-1** | 21 | 0.98 [0.97-3.18] | 4.70 | 2.82 |
| -1 | 22 | 1.11 [1.03-3.00] | 4.15 | 1.13 |
| physical | 16 | 1.17 [1.14-1.24] | 3.93 | 2.40 |
| cap8 | 8 | 1.50 [1.47-1.56] | 3.08 | 5.33 |

`backtest|classifier|logreg|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.05 [0.05-0.05] | 1.00 | 0.05 |
| auto | 1 | 0.08 [0.07-0.09] | 0.65 | 0.07 |
| cpu-1 | 21 | 0.04 [0.04-0.05] | 1.11 | 0.05 |
| -1 | 22 | 0.06 [0.06-3.42] | 0.81 | 4.12 |
| **physical** | 16 | 0.04 [0.04-0.05] | 1.12 | 1.05 |

`backtest|classifier|logreg|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.22 [0.21-0.23] | 1.00 | 0.21 |
| auto | 1 | 0.26 [0.24-0.28] | 0.86 | 0.28 |
| **cpu-1** | 21 | 0.07 [0.07-0.08] | 3.17 | 0.08 |
| -1 | 22 | 0.10 [0.08-2.10] | 2.30 | 0.09 |
| physical | 16 | 0.08 [0.08-0.09] | 2.71 | 1.04 |

`backtest|recursive|cat_1|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 4.07 [3.73-4.15] | 1.00 | 4.10 |
| auto | 21 | 2.40 [2.19-4.65] | 1.70 | 2.59 |
| cpu-1 | 21 | 2.31 [2.27-2.54] | 1.76 | 2.56 |
| -1 | 22 | 2.25 [2.20-4.11] | 1.81 | 2.64 |
| **physical** | 16 | 2.25 [2.17-2.43] | 1.81 | 6.86 |

`backtest|recursive|cat_1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 35.27 [35.16-38.12] | 1.00 | 38.03 |
| auto | 21 | 5.93 [5.79-6.01] | 5.95 | 5.98 |
| cpu-1 | 21 | 5.84 [5.75-6.01] | 6.04 | 6.19 |
| **-1** | 22 | 5.71 [5.55-5.86] | 6.18 | 6.21 |
| physical | 16 | 8.26 [8.15-8.46] | 4.27 | 11.98 |
| cap8 | 8 | 10.26 [10.10-10.29] | 3.44 | 14.45 |

`backtest|recursive|cat_1|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.03 [1.01-1.06] | 1.00 | 1.04 |
| auto | 21 | 1.03 [0.97-1.13] | 1.00 | 1.12 |
| cpu-1 | 21 | 0.95 [0.91-1.03] | 1.08 | 0.97 |
| -1 | 22 | 0.93 [0.90-3.78] | 1.11 | 4.83 |
| **physical** | 16 | 0.93 [0.92-1.05] | 1.11 | 2.83 |

`backtest|recursive|cat_1|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 8.98 [8.77-9.30] | 1.00 | 9.78 |
| **auto** | 21 | 2.55 [2.42-2.77] | 3.53 | 2.78 |
| cpu-1 | 21 | 2.56 [2.31-2.68] | 3.51 | 2.60 |
| -1 | 22 | 2.63 [2.38-4.69] | 3.42 | 3.02 |
| physical | 16 | 2.57 [2.52-2.59] | 3.49 | 2.41 |

`backtest|recursive|et_none|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.00 [0.98-1.03] | 1.00 | 1.10 |
| auto | 21 | 0.89 [0.85-3.11] | 1.13 | 0.96 |
| cpu-1 | 21 | 0.84 [0.82-0.93] | 1.19 | 1.16 |
| -1 | 22 | 0.86 [0.84-0.88] | 1.17 | 0.82 |
| **physical** | 16 | 0.79 [0.78-0.81] | 1.27 | 1.99 |

`backtest|recursive|et_none|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 11.58 [11.36-12.03] | 1.00 | 12.40 |
| auto | 21 | 2.41 [2.26-2.47] | 4.80 | 2.62 |
| **cpu-1** | 21 | 2.07 [2.01-2.13] | 5.61 | 2.09 |
| -1 | 22 | 2.43 [2.38-2.53] | 4.77 | 2.34 |
| physical | 16 | 2.62 [2.55-2.73] | 4.41 | 3.70 |
| cap8 | 8 | 3.51 [3.42-3.79] | 3.29 | 8.09 |

`backtest|recursive|et_none|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.09 [0.09-0.10] | 1.00 | 0.09 |
| auto | 21 | 0.13 [0.12-0.14] | 0.66 | 0.14 |
| cpu-1 | 21 | 0.15 [0.12-1.87] | 0.60 | 1.27 |
| -1 | 22 | 0.13 [0.12-1.99] | 0.65 | 0.21 |
| physical | 16 | 0.12 [0.11-0.17] | 0.74 | 0.25 |

`backtest|recursive|et_none|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.66 [0.63-0.70] | 1.00 | 0.68 |
| **auto** | 21 | 0.23 [0.23-0.25] | 2.83 | 0.23 |
| cpu-1 | 21 | 0.26 [0.24-0.35] | 2.54 | 1.44 |
| -1 | 22 | 0.35 [0.24-2.82] | 1.91 | 0.38 |
| physical | 16 | 0.27 [0.27-0.29] | 2.45 | 0.27 |

`backtest|recursive|hgb|L|refit 3`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 3.98 [3.95-4.00] | 1.00 | 3.99 |
| **auto** | 1 | 3.97 [3.93-3.99] | 1.00 | 3.88 |
| -1 | 1 | 4.10 [4.06-4.18] | 0.97 | 4.21 |

`backtest|recursive|hgb|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.94 [2.89-2.97] | 1.00 | 2.92 |
| **auto** | 21 | 0.52 [0.52-0.53] | 5.61 | 0.63 |
| cpu-1 | 21 | 0.58 [0.56-0.60] | 5.07 | 1.50 |
| -1 | 22 | 0.60 [0.59-0.65] | 4.86 | 0.65 |
| physical | 16 | 0.58 [0.55-0.60] | 5.03 | 2.25 |

`backtest|recursive|hgb|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 7.46 [7.20-7.61] | 1.00 | 7.34 |
| auto | 21 | 1.40 [1.33-3.74] | 5.34 | 3.58 |
| **cpu-1** | 21 | 1.38 [1.35-3.64] | 5.41 | 3.70 |
| -1 | 22 | 1.40 [1.34-3.11] | 5.33 | 1.38 |
| physical | 16 | 1.84 [1.68-1.93] | 4.05 | 1.81 |
| cap8 | 8 | 1.99 [1.92-2.21] | 3.74 | 6.30 |

`backtest|recursive|hgb|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.36 [0.34-0.37] | 1.00 | 0.45 |
| **auto** | 21 | 0.27 [0.26-0.31] | 1.35 | 0.32 |
| cpu-1 | 21 | 0.35 [0.26-2.45] | 1.03 | 0.52 |
| -1 | 22 | 0.30 [0.28-2.14] | 1.21 | 0.36 |
| physical | 16 | 0.28 [0.27-0.37] | 1.30 | 1.47 |

`backtest|recursive|hgb|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.74 [1.71-1.77] | 1.00 | 1.78 |
| auto | 21 | 0.54 [0.49-2.75] | 3.19 | 0.69 |
| cpu-1 | 21 | 0.57 [0.44-3.56] | 3.07 | 4.51 |
| -1 | 22 | 0.48 [0.46-0.59] | 3.61 | 0.66 |
| **physical** | 16 | 0.47 [0.46-0.58] | 3.71 | 0.63 |

`backtest|recursive|lgbm_1|L|refit 3`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.42 [2.22-2.57] | 1.00 | 2.27 |
| auto | 1 | 2.21 [2.17-2.37] | 1.10 | 2.34 |
| **-1** | 1 | 1.99 [1.91-2.01] | 1.22 | 1.95 |

`backtest|recursive|lgbm_1|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.55 [0.54-0.57] | 1.00 | 0.52 |
| auto | 21 | 0.47 [0.43-0.49] | 1.17 | 0.50 |
| cpu-1 | 21 | 0.50 [0.46-0.56] | 1.10 | 1.54 |
| **-1** | 22 | 0.45 [0.44-2.06] | 1.21 | 0.48 |
| physical | 16 | 0.49 [0.45-0.49] | 1.13 | 1.43 |

`backtest|recursive|lgbm_1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 5.62 [5.47-5.80] | 1.00 | 5.72 |
| auto | 21 | 1.36 [1.33-3.34] | 4.14 | 7.00 |
| **cpu-1** | 21 | 1.33 [1.29-1.37] | 4.22 | 1.28 |
| -1 | 22 | 1.37 [1.35-1.41] | 4.10 | 1.42 |
| physical | 16 | 1.49 [1.43-1.54] | 3.76 | 2.66 |
| cap8 | 8 | 1.87 [1.84-1.91] | 3.01 | 1.87 |

`backtest|recursive|lgbm_1|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.13 [0.11-0.14] | 1.00 | 0.12 |
| auto | 21 | 0.25 [0.21-2.44] | 0.49 | 0.29 |
| cpu-1 | 21 | 0.17 [0.16-0.19] | 0.75 | 0.20 |
| -1 | 22 | 0.17 [0.16-0.19] | 0.72 | 0.17 |
| physical | 16 | 0.17 [0.16-0.19] | 0.75 | 2.22 |

`backtest|recursive|lgbm_1|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.15 [1.11-1.17] | 1.00 | 1.14 |
| auto | 21 | 0.43 [0.41-0.48] | 2.71 | 0.51 |
| cpu-1 | 21 | 0.47 [0.43-2.68] | 2.46 | 0.62 |
| -1 | 22 | 0.44 [0.41-2.25] | 2.62 | 0.40 |
| **physical** | 16 | 0.41 [0.39-0.42] | 2.82 | 1.44 |

`backtest|recursive|pipe_ridge|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.67 [0.66-0.70] | 1.00 | 0.68 |
| auto | 1 | 0.80 [0.75-0.87] | 0.85 | 0.78 |
| cpu-1 | 21 | 0.20 [0.16-2.26] | 3.40 | 0.28 |
| **-1** | 22 | 0.15 [0.13-1.87] | 4.64 | 0.19 |
| physical | 16 | 0.16 [0.14-0.17] | 4.20 | 0.18 |

`backtest|recursive|pipe_ridge|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.82 [0.81-0.84] | 1.00 | 0.87 |
| auto | 1 | 0.94 [0.90-0.94] | 0.88 | 0.96 |
| cpu-1 | 21 | 0.24 [0.18-2.61] | 3.43 | 0.33 |
| **-1** | 22 | 0.17 [0.16-0.18] | 4.94 | 0.16 |
| physical | 16 | 0.19 [0.19-0.21] | 4.24 | 1.42 |
| cap8 | 8 | 0.26 [0.23-0.26] | 3.22 | 3.96 |

`backtest|recursive|pipe_ridge|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.06 [0.06-0.07] | 1.00 | 0.06 |
| auto | 1 | 0.08 [0.07-0.08] | 0.83 | 0.07 |
| cpu-1 | 21 | 0.04 [0.04-0.06] | 1.44 | 0.12 |
| -1 | 22 | 0.04 [0.04-0.08] | 1.60 | 0.08 |
| **physical** | 16 | 0.04 [0.03-0.04] | 1.64 | 2.52 |

`backtest|recursive|pipe_ridge|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.08 [0.08-0.08] | 1.00 | 0.08 |
| auto | 1 | 0.09 [0.08-0.09] | 0.97 | 0.09 |
| cpu-1 | 21 | 0.05 [0.04-0.05] | 1.64 | 0.05 |
| **-1** | 22 | 0.04 [0.04-0.04] | 2.07 | 0.06 |
| physical | 16 | 0.04 [0.04-0.05] | 2.03 | 0.05 |

`backtest|recursive|rf_none|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.62 [2.60-2.72] | 1.00 | 2.83 |
| **auto** | 21 | 2.46 [2.41-2.51] | 1.06 | 3.83 |
| cpu-1 | 21 | 2.79 [2.62-5.11] | 0.94 | 7.82 |
| -1 | 22 | 2.55 [2.49-5.22] | 1.03 | 3.08 |
| physical | 16 | 2.51 [2.47-2.64] | 1.04 | 3.87 |

`backtest|recursive|rf_none|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 45.97 [45.52-48.20] | 1.00 | 50.08 |
| **auto** | 21 | 8.78 [8.75-8.90] | 5.23 | 8.69 |
| cpu-1 | 21 | 9.34 [9.01-9.81] | 4.92 | 9.61 |
| -1 | 22 | 9.69 [9.46-10.64] | 4.74 | 11.12 |
| physical | 16 | 11.78 [11.40-11.85] | 3.90 | 13.60 |
| cap8 | 8 | 14.45 [14.17-14.57] | 3.18 | 15.45 |

`backtest|recursive|rf_none|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.18 [0.16-0.19] | 1.00 | 0.15 |
| auto | 21 | 0.28 [0.19-2.37] | 0.67 | 0.39 |
| cpu-1 | 21 | 0.27 [0.23-0.37] | 0.68 | 0.60 |
| -1 | 22 | 0.26 [0.19-2.51] | 0.72 | 0.32 |
| **physical** | 16 | 0.18 [0.17-0.26] | 1.03 | 0.29 |

`backtest|recursive|rf_none|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.39 [1.36-1.40] | 1.00 | 1.50 |
| auto | 21 | 0.50 [0.47-2.70] | 2.76 | 0.64 |
| **cpu-1** | 21 | 0.44 [0.41-0.48] | 3.17 | 0.44 |
| -1 | 22 | 0.46 [0.45-2.37] | 3.05 | 0.65 |
| physical | 16 | 0.45 [0.43-0.46] | 3.11 | 1.70 |

`backtest|recursive|ridge|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.02 [0.02-0.03] | 1.00 | 0.02 |
| auto | 1 | 0.03 [0.02-0.04] | 0.79 | 0.03 |
| cpu-1 | 21 | 0.05 [0.04-0.05] | 0.54 | 0.12 |
| -1 | 22 | 0.05 [0.04-0.06] | 0.53 | 0.14 |
| physical | 16 | 0.05 [0.05-0.05] | 0.49 | 1.12 |

`backtest|recursive|ridge|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.11 [0.11-0.11] | 1.00 | 0.12 |
| auto | 1 | 0.11 [0.11-0.12] | 0.97 | 0.11 |
| cpu-1 | 21 | 0.08 [0.06-2.83] | 1.37 | 2.65 |
| -1 | 22 | 0.06 [0.05-0.08] | 1.74 | 0.08 |
| **physical** | 16 | 0.05 [0.05-0.07] | 2.06 | 0.08 |
| cap8 | 8 | 0.05 [0.05-0.07] | 2.05 | 4.15 |

`backtest|recursive|ridge|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.01 [0.01-0.01] | 1.00 | 0.02 |
| auto | 1 | 0.01 [0.01-0.01] | 0.92 | 0.02 |
| cpu-1 | 21 | 0.02 [0.02-0.02] | 0.75 | 1.06 |
| -1 | 22 | 0.04 [0.03-0.05] | 0.34 | 0.12 |
| physical | 16 | 0.02 [0.02-0.03] | 0.79 | 2.28 |

`backtest|recursive|ridge|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.02 [0.02-0.02] | 1.00 | 0.02 |
| **auto** | 1 | 0.02 [0.02-0.02] | 1.05 | 0.03 |
| cpu-1 | 21 | 0.03 [0.03-0.03] | 0.90 | 0.03 |
| -1 | 22 | 0.03 [0.03-0.03] | 0.80 | 0.05 |
| physical | 16 | 0.03 [0.03-0.08] | 0.86 | 1.20 |

`backtest|recursive|xgb_1|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.12 [1.10-1.15] | 1.00 | 1.16 |
| auto | 21 | 0.72 [0.69-0.87] | 1.55 | 2.87 |
| cpu-1 | 21 | 0.72 [0.70-0.82] | 1.55 | 1.89 |
| -1 | 22 | 0.89 [0.79-0.97] | 1.25 | 2.90 |
| **physical** | 16 | 0.68 [0.67-0.74] | 1.64 | 3.37 |

`backtest|recursive|xgb_1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 10.36 [10.32-10.59] | 1.00 | 11.05 |
| auto | 21 | 2.12 [2.03-2.13] | 4.90 | 3.53 |
| **cpu-1** | 21 | 2.11 [2.07-4.43] | 4.92 | 4.39 |
| -1 | 22 | 2.35 [2.24-4.74] | 4.41 | 4.95 |
| physical | 16 | 2.61 [2.55-2.68] | 3.97 | 5.21 |
| cap8 | 8 | 3.57 [3.48-3.60] | 2.90 | 8.61 |

`backtest|recursive|xgb_1|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.36 [0.35-0.37] | 1.00 | 0.41 |
| auto | 21 | 0.40 [0.36-0.54] | 0.90 | 1.42 |
| cpu-1 | 21 | 0.44 [0.38-3.56] | 0.81 | 0.45 |
| -1 | 22 | 0.65 [0.41-2.96] | 0.55 | 2.58 |
| physical | 16 | 0.36 [0.35-0.38] | 0.99 | 0.37 |

`backtest|recursive|xgb_1|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 3.40 [3.26-3.71] | 1.00 | 3.62 |
| **auto** | 21 | 0.94 [0.90-3.16] | 3.61 | 1.19 |
| cpu-1 | 21 | 0.96 [0.90-1.28] | 3.54 | 2.80 |
| -1 | 22 | 1.00 [0.90-4.35] | 3.41 | 6.55 |
| physical | 16 | 0.96 [0.91-0.98] | 3.56 | 2.67 |

### Bloque C, afinidad 4 (cpu_count 4)


`backtest|recursive|hgb|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 5.44 [5.40-5.54] | 1.00 | 5.99 |
| cpu-1 | 3 | 2.98 [2.96-3.00] | 1.82 | 6.14 |
| **-1** | 4 | 2.76 [2.72-2.99] | 1.97 | 6.49 |
| physical | 2 | 4.01 [3.90-4.10] | 1.36 | 6.82 |

`backtest|recursive|lgbm_none|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 3.73 [3.54-3.83] | 1.00 | 3.83 |
| cpu-1 | 3 | 5.67 [5.45-5.75] | 0.66 | 9.05 |
| -1 | 4 | 5.91 [5.84-6.00] | 0.63 | 9.75 |
| physical | 2 | 5.42 [5.38-5.54] | 0.69 | 8.08 |

`backtest|recursive|rf_m1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 21.28 [21.00-21.57] | 1.00 | 22.51 |
| cpu-1 | 3 | 19.28 [19.19-19.48] | 1.10 | 22.37 |
| **-1** | 4 | 19.15 [19.01-19.26] | 1.11 | 22.44 |
| physical | 2 | 19.36 [19.29-19.48] | 1.10 | 21.85 |

`backtest|recursive|xgb_none|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 7.08 [6.90-7.18] | 1.00 | 8.86 |
| cpu-1 | 3 | 6.14 [6.01-6.24] | 1.15 | 9.92 |
| **-1** | 4 | 5.96 [5.81-6.17] | 1.19 | 10.34 |
| physical | 2 | 6.28 [6.15-6.50] | 1.13 | 9.98 |

### Bloque C, afinidad 8 (cpu_count 8)


`backtest|recursive|hgb|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 6.88 [6.81-7.15] | 1.00 | 7.76 |
| cpu-1 | 7 | 2.19 [2.11-2.37] | 3.14 | 6.42 |
| **-1** | 8 | 2.17 [2.07-2.41] | 3.17 | 6.76 |
| physical | 4 | 3.08 [3.03-3.25] | 2.23 | 6.38 |

`backtest|recursive|lgbm_none|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 3.62 [3.56-3.73] | 1.00 | 4.00 |
| cpu-1 | 7 | 3.61 [3.47-3.73] | 1.00 | 7.80 |
| **-1** | 8 | 3.20 [2.97-3.63] | 1.13 | 7.80 |
| physical | 4 | 3.86 [3.54-4.01] | 0.94 | 6.47 |

`backtest|recursive|rf_m1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 20.42 [19.83-21.42] | 1.00 | 22.78 |
| cpu-1 | 7 | 16.50 [13.32-16.61] | 1.24 | 21.42 |
| **-1** | 8 | 13.23 [13.09-13.40] | 1.54 | 18.13 |
| physical | 4 | 13.34 [13.02-13.83] | 1.53 | 16.03 |

`backtest|recursive|xgb_none|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 10.02 [7.98-10.62] | 1.00 | 13.01 |
| **cpu-1** | 7 | 3.98 [3.90-4.08] | 2.52 | 9.26 |
| -1 | 8 | 5.36 [4.87-6.07] | 1.87 | 9.88 |
| physical | 4 | 5.73 [5.00-6.35] | 1.75 | 9.25 |

### Bloque C, afinidad 22 (cpu_count 22)


`backtest|recursive|cat_m1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 10.04 [9.95-10.09] | 1.00 | 10.60 |
| auto | 21 | 4.99 [4.64-6.46] | 2.01 | 11.01 |
| **cpu-1** | 21 | 4.51 [4.43-6.01] | 2.23 | 10.15 |
| -1 | 22 | 4.74 [4.66-6.57] | 2.12 | 10.55 |
| physical | 16 | 5.03 [4.93-5.41] | 2.00 | 10.44 |
| 4 | 4 | 6.04 [5.70-6.54] | 1.66 | 9.02 |
| 8 | 8 | 5.30 [5.20-5.43] | 1.90 | 8.79 |

`backtest|recursive|et_m1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.73 [2.70-2.81] | 1.00 | 2.87 |
| auto | 21 | 1.59 [1.56-3.83] | 1.71 | 7.15 |
| **cpu-1** | 21 | 1.57 [1.54-3.80] | 1.73 | 7.04 |
| -1 | 22 | 1.60 [1.57-3.93] | 1.71 | 7.12 |
| physical | 16 | 1.61 [1.59-1.68] | 1.70 | 6.46 |
| 4 | 4 | 1.68 [1.63-1.71] | 1.62 | 4.53 |
| 8 | 8 | 1.61 [1.57-1.62] | 1.70 | 5.11 |

`backtest|recursive|hgb|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 9.17 [8.45-9.50] | 1.00 | 11.14 |
| auto | 21 | 1.75 [1.64-4.05] | 5.25 | 7.29 |
| **cpu-1** | 21 | 1.67 [1.52-4.21] | 5.49 | 7.27 |
| -1 | 22 | 1.79 [1.70-4.24] | 5.14 | 7.35 |
| physical | 16 | 2.03 [1.89-2.15] | 4.53 | 7.49 |
| 4 | 4 | 4.28 [4.11-4.64] | 2.14 | 7.72 |
| 8 | 8 | 2.46 [2.22-2.53] | 3.72 | 5.89 |

`backtest|recursive|lgbm_4|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 3.96 [3.91-4.47] | 1.00 | 4.70 |
| auto | 1 | 3.86 [3.60-4.17] | 1.02 | 4.14 |
| cpu-1 | 21 | 1.67 [1.64-3.56] | 2.37 | 6.86 |
| -1 | 22 | 1.75 [1.65-3.77] | 2.27 | 7.16 |
| physical | 16 | 1.89 [1.60-2.14] | 2.10 | 6.39 |
| 4 | 4 | 1.82 [1.81-1.88] | 2.17 | 4.71 |
| **8** | 8 | 1.62 [1.49-1.79] | 2.45 | 4.80 |

`backtest|recursive|lgbm_m1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 5.11 [5.05-5.26] | 1.00 | 4.95 |
| auto | 1 | 5.34 [5.22-5.52] | 0.96 | 5.27 |
| cpu-1 | 21 | 3.67 [3.27-5.07] | 1.39 | 9.69 |
| -1 | 22 | 3.77 [3.61-5.55] | 1.36 | 10.08 |
| physical | 16 | 3.67 [3.52-3.84] | 1.39 | 8.83 |
| 4 | 4 | 3.73 [3.71-3.84] | 1.37 | 6.68 |
| **8** | 8 | 3.39 [3.32-3.50] | 1.51 | 6.84 |

`backtest|recursive|lgbm_none|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 3.67 [3.48-3.72] | 1.00 | 3.57 |
| auto | 1 | 3.64 [3.60-3.92] | 1.01 | 3.76 |
| cpu-1 | 21 | 3.20 [2.92-5.21] | 1.15 | 8.84 |
| -1 | 22 | 3.08 [2.89-5.24] | 1.19 | 8.92 |
| physical | 16 | 2.87 [2.69-2.94] | 1.28 | 8.62 |
| 4 | 4 | 3.08 [2.98-3.20] | 1.19 | 6.05 |
| **8** | 8 | 2.76 [2.53-2.92] | 1.33 | 6.27 |

`backtest|recursive|rf_m1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 9.79 [9.58-10.10] | 1.00 | 10.76 |
| auto | 21 | 6.81 [6.74-8.82] | 1.44 | 12.80 |
| **cpu-1** | 21 | 6.77 [6.74-9.31] | 1.45 | 12.91 |
| -1 | 22 | 7.17 [6.92-9.81] | 1.37 | 13.11 |
| physical | 16 | 6.89 [6.85-7.03] | 1.42 | 11.76 |
| 4 | 4 | 7.15 [7.11-7.62] | 1.37 | 10.36 |
| 8 | 8 | 7.00 [6.97-7.08] | 1.40 | 10.78 |

`backtest|recursive|xgb_m1|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 10.10 [9.91-10.66] | 1.00 | 11.37 |
| **auto** | 21 | 2.38 [2.35-5.34] | 4.24 | 9.94 |
| cpu-1 | 21 | 2.61 [2.50-5.69] | 3.87 | 10.09 |
| -1 | 22 | 2.40 [2.32-5.48] | 4.21 | 10.25 |
| physical | 16 | 2.87 [2.82-2.90] | 3.53 | 9.63 |
| 4 | 4 | 3.37 [3.30-3.45] | 2.99 | 7.42 |
| 8 | 8 | 2.93 [2.89-2.98] | 3.45 | 7.98 |

`backtest|recursive|xgb_none|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 7.33 [6.90-7.37] | 1.00 | 8.11 |
| auto | 21 | 2.87 [2.81-5.54] | 2.55 | 10.87 |
| **cpu-1** | 21 | 2.66 [2.48-5.35] | 2.76 | 9.21 |
| -1 | 22 | 2.77 [2.72-5.56] | 2.65 | 10.26 |
| physical | 16 | 3.17 [3.08-3.34] | 2.31 | 10.38 |
| 4 | 4 | 4.93 [4.59-4.96] | 1.49 | 8.91 |
| 8 | 8 | 4.00 [3.90-4.11] | 1.83 | 8.85 |

### Bloque D, afinidad 4 (cpu_count 4)


`fit_direct|direct|lgbm_1|L|refit `, folds/estimadores 48

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 11.33 [11.26-11.64] | 1.00 | 12.76 |
| cpu-1 | 3 | 7.58 [7.34-7.79] | 1.49 | 10.42 |
| **-1** | 4 | 7.15 [7.03-7.29] | 1.58 | 10.65 |
| physical | 2 | 8.62 [8.57-8.76] | 1.31 | 10.72 |

`fit_direct|direct|ridge|L|refit `, folds/estimadores 48

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.35 [0.28-0.38] | 1.00 | 0.34 |
| cpu-1 | 3 | 0.42 [0.42-0.43] | 0.82 | 0.60 |
| -1 | 4 | 0.47 [0.39-1.80] | 0.75 | 0.51 |
| physical | 2 | 0.40 [0.39-0.40] | 0.88 | 2.72 |

### Bloque D, afinidad 22 (cpu_count 22)


`backtest_ms|dmv_fix|lgbm_1|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 21.0 | 7.94 [7.83-8.07] | 4.45 | 7.83 |

`backtest_ms|dmv_fix|ridge|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 1.0 | 0.20 [0.20-0.21] | 1.65 | 0.20 |

`backtest_ms|dmv|lgbm_1|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| auto|auto | 1 / 21.0 | 17.82 [17.80-18.14] | 1.98 | 17.72 |
| 1|1 | 1 / 1.0 | 35.34 [34.81-36.38] | 1.00 | 37.53 |
| **cpu-1|1** | 21 / 1.0 | 12.38 [12.30-12.66] | 2.86 | 12.44 |

`backtest_ms|dmv|ridge|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| auto|auto | 1 / 1.0 | 0.23 [0.22-0.24] | 1.44 | 0.27 |
| 1|1 | 1 / 1.0 | 0.33 [0.29-0.39] | 1.00 | 0.47 |
| **cpu-1|1** | 21 / 1.0 | 0.11 [0.10-0.23] | 3.10 | 0.22 |

`backtest|direct_fix|hgb|L|refit True`, folds/estimadores 5

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 21.0 | 10.92 [10.72-11.08] | 5.85 | 13.15 |

`backtest|direct_fix|hgb|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 21.0 | 4.29 [4.25-4.31] | 9.88 | 8.81 |

`backtest|direct_fix|lgbm_1|L|refit True`, folds/estimadores 5

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 21.0 | 13.17 [11.90-13.74] | 4.16 | 12.61 |

`backtest|direct_fix|lgbm_1|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 21.0 | 5.74 [5.40-6.03] | 3.79 | 5.79 |

`backtest|direct_fix|ridge|L|refit True`, folds/estimadores 5

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 1.0 | 0.68 [0.66-0.77] | 1.08 | 0.69 |

`backtest|direct_fix|ridge|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 1.0 | 0.22 [0.22-0.25] | 1.01 | 0.22 |

`backtest|direct|hgb|L|refit False`, folds/estimadores 5

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 21.0 | 2.83 [2.71-2.90] | 4.25 | 9.18 |
| 1|1 | 1 / 1.0 | 12.04 [11.48-12.24] | 1.00 | 13.81 |
| cpu-1|1 | 21 / 1.0 | 18.41 [16.38-22.05] | 0.65 | 19.90 |
| physical|1 | 16 / 1.0 | 14.76 [13.92-16.93] | 0.82 | 17.84 |
| -1|1 | 22 / 1.0 | 14.87 [14.25-15.16] | 0.81 | 17.02 |
| cpu-1|cpu-1 | 21 / 21.0 | 3.47 [3.42-3.53] | 3.47 | 3.58 |

`backtest|direct|hgb|L|refit True`, folds/estimadores 5

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| auto|auto | 1 / 21.0 | 77.29 [75.76-79.42] | 0.83 | 88.84 |
| 1|1 | 1 / 1.0 | 63.90 [61.98-64.41] | 1.00 | 56.11 |
| cpu-1|1 | 21 / 1.0 | 29.07 [27.93-30.33] | 2.20 | 30.51 |
| -1|1 | 22 / 1.0 | 29.35 [28.15-30.16] | 2.18 | 33.28 |
| **cpu-1|cpu-1** | 21 / 21.0 | 18.79 [18.46-19.11] | 3.40 | 22.77 |

`backtest|direct|hgb|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 21.0 | 0.67 [0.64-0.70] | 8.17 | 0.86 |
| 1|1 | 1 / 1.0 | 5.46 [5.38-6.47] | 1.00 | 5.47 |
| cpu-1|1 | 21 / 1.0 | 5.55 [5.17-5.90] | 0.98 | 5.79 |
| physical|1 | 16 / 1.0 | 5.31 [5.22-5.61] | 1.03 | 5.95 |
| -1|1 | 22 / 1.0 | 5.81 [5.71-9.25] | 0.94 | 9.95 |
| cpu-1|cpu-1 | 21 / 21.0 | 2.29 [2.02-2.42] | 2.38 | 2.43 |

`backtest|direct|hgb|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| auto|auto | 1 / 21.0 | 43.06 [41.26-44.05] | 0.99 | 48.08 |
| 1|1 | 1 / 1.0 | 42.43 [40.01-43.43] | 1.00 | 37.75 |
| cpu-1|1 | 21 / 1.0 | 10.44 [9.83-10.47] | 4.07 | 10.63 |
| physical|1 | 16 / 1.0 | 10.84 [10.51-11.16] | 3.92 | 11.32 |
| -1|1 | 22 / 1.0 | 10.86 [10.39-11.13] | 3.91 | 11.50 |
| **cpu-1|cpu-1** | 21 / 21.0 | 7.92 [7.66-10.14] | 5.36 | 8.26 |

`backtest|direct|lgbm_1|L|refit False`, folds/estimadores 5

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 21.0 | 2.69 [2.58-2.89] | 4.37 | 2.46 |
| 1|1 | 1 / 1.0 | 11.74 [11.25-12.04] | 1.00 | 12.63 |
| cpu-1|1 | 21 / 1.0 | 15.66 [14.50-17.05] | 0.75 | 15.63 |
| physical|1 | 16 / 1.0 | 14.23 [13.90-15.06] | 0.83 | 16.30 |
| -1|1 | 22 / 1.0 | 14.50 [14.27-15.62] | 0.81 | 15.79 |
| cpu-1|cpu-1 | 21 / 21.0 | 4.13 [4.04-4.39] | 2.84 | 4.50 |

`backtest|direct|lgbm_1|L|refit True`, folds/estimadores 5

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| auto|auto | 1 / 21.0 | 70.67 [69.03-73.63] | 0.77 | 69.08 |
| 1|1 | 1 / 1.0 | 54.72 [54.47-57.49] | 1.00 | 58.60 |
| cpu-1|1 | 21 / 1.0 | 31.86 [31.12-32.29] | 1.72 | 34.06 |
| **cpu-1|cpu-1** | 21 / 21.0 | 10.45 [10.19-10.97] | 5.24 | 12.05 |

`backtest|direct|lgbm_1|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 21.0 | 0.52 [0.49-0.59] | 3.80 | 0.55 |
| 1|1 | 1 / 1.0 | 1.96 [1.94-2.02] | 1.00 | 1.96 |
| cpu-1|1 | 21 / 1.0 | 3.63 [3.52-6.61] | 0.54 | 7.12 |
| physical|1 | 16 / 1.0 | 3.71 [3.46-3.87] | 0.53 | 3.85 |
| -1|1 | 22 / 1.0 | 3.38 [3.35-3.53] | 0.58 | 3.52 |
| cpu-1|cpu-1 | 21 / 21.0 | 1.83 [1.78-4.29] | 1.07 | 3.96 |

`backtest|direct|lgbm_1|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| auto|auto | 1 / 21.0 | 34.14 [28.84-35.51] | 0.64 | 27.98 |
| 1|1 | 1 / 1.0 | 21.77 [21.42-22.42] | 1.00 | 24.19 |
| cpu-1|1 | 21 / 1.0 | 8.06 [7.94-8.42] | 2.70 | 8.51 |
| physical|1 | 16 / 1.0 | 8.58 [8.32-9.07] | 2.54 | 9.09 |
| **cpu-1|cpu-1** | 21 / 21.0 | 5.45 [5.31-5.53] | 3.99 | 5.97 |

`backtest|direct|ridge|L|refit False`, folds/estimadores 5

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **auto|auto** | 1 / 1.0 | 0.14 [0.13-0.15] | 1.06 | 0.15 |
| 1|1 | 1 / 1.0 | 0.15 [0.14-0.16] | 1.00 | 0.15 |
| cpu-1|1 | 21 / 1.0 | 0.16 [0.16-0.17] | 0.91 | 0.19 |
| physical|1 | 16 / 1.0 | 0.18 [0.16-0.18] | 0.84 | 1.18 |
| -1|1 | 22 / 1.0 | 0.14 [0.14-0.15] | 1.04 | 0.15 |
| cpu-1|cpu-1 | 21 / 21.0 | 0.81 [0.75-0.84] | 0.18 | 2.62 |

`backtest|direct|ridge|L|refit True`, folds/estimadores 5

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| auto|auto | 1 / 1.0 | 0.90 [0.84-0.91] | 0.81 | 0.87 |
| 1|1 | 1 / 1.0 | 0.73 [0.71-0.80] | 1.00 | 0.79 |
| cpu-1|1 | 21 / 1.0 | 0.55 [0.39-3.53] | 1.33 | 0.72 |
| **physical|1** | 16 / 1.0 | 0.39 [0.36-0.39] | 1.89 | 1.52 |
| -1|1 | 22 / 1.0 | 0.55 [0.41-2.16] | 1.33 | 0.62 |
| cpu-1|cpu-1 | 21 / 21.0 | 1.03 [1.02-1.09] | 0.71 | 1.01 |

`backtest|direct|ridge|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| auto|auto | 1 / 1.0 | 0.04 [0.03-0.05] | 0.94 | 0.05 |
| **1|1** | 1 / 1.0 | 0.04 [0.04-0.04] | 1.00 | 0.03 |
| cpu-1|1 | 21 / 1.0 | 1.79 [0.05-2.14] | 0.02 | 0.36 |
| cpu-1|cpu-1 | 21 / 21.0 | 0.09 [0.09-0.11] | 0.40 | 0.40 |

`backtest|direct|ridge|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| auto|auto | 1 / 1.0 | 0.24 [0.23-0.28] | 0.91 | 0.28 |
| 1|1 | 1 / 1.0 | 0.22 [0.22-0.23] | 1.00 | 0.23 |
| **cpu-1|1** | 21 / 1.0 | 0.11 [0.09-2.24] | 2.01 | 0.22 |
| cpu-1|cpu-1 | 21 / 21.0 | 0.19 [0.18-0.20] | 1.15 | 0.19 |

`fit_direct|direct|hgb|L|refit `, folds/estimadores 48

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 13.33 [12.92-14.01] | 1.00 | 12.59 |
| **cpu-1** | 21 | 2.36 [2.33-2.42] | 5.65 | 2.42 |
| physical | 16 | 2.56 [2.54-2.58] | 5.22 | 3.56 |
| cap8 | 8 | 3.32 [3.20-3.39] | 4.02 | 6.93 |

`fit_direct|direct|hgb|S|refit `, folds/estimadores 24

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 5.50 [5.39-5.82] | 1.00 | 4.70 |
| auto | 21 | 0.48 [0.46-0.50] | 11.51 | 0.48 |
| cpu-1 | 21 | 0.63 [0.60-0.65] | 8.78 | 0.60 |
| **-1** | 22 | 0.45 [0.44-0.46] | 12.16 | 0.50 |
| physical | 16 | 0.59 [0.58-0.71] | 9.36 | 2.00 |

`fit_direct|direct|lgbm_1|L|refit `, folds/estimadores 48

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 13.54 [13.31-14.21] | 1.00 | 14.57 |
| auto | 21 | 3.24 [3.16-3.35] | 4.17 | 3.12 |
| **cpu-1** | 21 | 3.12 [3.03-3.23] | 4.34 | 5.59 |
| -1 | 22 | 3.41 [3.24-3.47] | 3.98 | 3.39 |
| physical | 16 | 3.37 [3.34-3.59] | 4.02 | 4.50 |
| cap8 | 8 | 4.48 [4.27-4.64] | 3.02 | 7.48 |

`fit_direct|direct|lgbm_1|S|refit `, folds/estimadores 24

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.69 [2.60-2.84] | 1.00 | 3.05 |
| auto | 21 | 0.79 [0.73-0.82] | 3.41 | 7.97 |
| cpu-1 | 21 | 0.72 [0.67-0.79] | 3.74 | 0.68 |
| -1 | 22 | 0.73 [0.65-2.26] | 3.69 | 0.72 |
| **physical** | 16 | 0.62 [0.59-0.63] | 4.36 | 1.58 |

`fit_direct|direct|lgbm_none|L|refit `, folds/estimadores 48

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 7.95 [7.86-8.11] | 1.00 | 7.44 |
| **auto** | 1 | 5.91 [5.73-6.06] | 1.35 | 5.49 |
| cpu-1 | 21 | 7.60 [7.52-8.85] | 1.05 | 7.30 |
| -1 | 22 | 8.41 [8.19-8.79] | 0.95 | 8.50 |
| cap8 | 8 | 6.79 [6.47-7.28] | 1.17 | 9.98 |

`fit_direct|direct|lgbm_none|S|refit `, folds/estimadores 24

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 3.07 [2.47-3.26] | 1.00 | 2.82 |
| **auto** | 1 | 1.89 [1.79-1.98] | 1.63 | 1.78 |
| cpu-1 | 21 | 3.10 [2.99-3.69] | 0.99 | 9.90 |
| -1 | 22 | 2.71 [2.68-2.90] | 1.13 | 3.41 |
| physical | 16 | 2.28 [2.20-2.49] | 1.34 | 3.60 |

`fit_direct|direct|rf_none|S|refit `, folds/estimadores 24

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 4.86 [4.38-5.13] | 1.00 | 4.18 |
| auto | 21 | 0.65 [0.62-0.70] | 7.51 | 0.64 |
| **cpu-1** | 21 | 0.63 [0.58-0.68] | 7.69 | 0.64 |
| physical | 16 | 0.69 [0.67-0.75] | 7.06 | 1.86 |

`fit_direct|direct|ridge|L|refit `, folds/estimadores 48

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.17 [0.17-0.21] | 1.00 | 0.17 |
| cpu-1 | 21 | 0.78 [0.75-0.79] | 0.22 | 0.84 |
| cap8 | 8 | 0.59 [0.58-0.61] | 0.29 | 4.34 |

`fit_direct|direct|ridge|S|refit `, folds/estimadores 24

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.03 [0.03-0.04] | 1.00 | 0.04 |
| **auto** | 1 | 0.03 [0.03-0.06] | 1.03 | 0.05 |
| cpu-1 | 21 | 0.11 [0.09-2.05] | 0.30 | 0.20 |
| -1 | 22 | 0.08 [0.07-1.68] | 0.40 | 0.16 |
| physical | 16 | 0.08 [0.07-0.09] | 0.41 | 0.09 |

`fit_dmv|dmv|lgbm_1|L|refit `, folds/estimadores 24

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 33.87 [33.16-35.18] | 1.00 | 36.31 |
| **auto** | 21 | 6.55 [6.49-6.81] | 5.17 | 6.70 |
| cpu-1 | 21 | 6.65 [6.49-6.80] | 5.09 | 6.59 |
| physical | 16 | 7.39 [7.14-7.64] | 4.58 | 8.60 |

`fit_dmv|dmv|lgbm_1|S|refit `, folds/estimadores 12

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 3.82 [3.64-3.89] | 1.00 | 4.07 |
| cpu-1 | 21 | 1.18 [1.14-1.29] | 3.23 | 1.05 |
| **physical** | 16 | 0.82 [0.78-0.88] | 4.68 | 1.89 |

`fit_dmv|dmv|ridge|L|refit `, folds/estimadores 24

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.28 [0.26-0.33] | 1.00 | 0.30 |
| **cpu-1** | 21 | 0.27 [0.26-0.31] | 1.07 | 0.31 |

`fit_dmv|dmv|ridge|S|refit `, folds/estimadores 12

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.02 [0.02-0.03] | 1.00 | 0.03 |
| auto | 1 | 0.03 [0.02-0.03] | 0.87 | 0.07 |
| cpu-1 | 21 | 0.04 [0.04-0.06] | 0.54 | 0.09 |
| physical | 16 | 0.04 [0.04-0.05] | 0.54 | 1.09 |

### Bloque E, afinidad 4 (cpu_count 4)


`backtest_ms|multiseries|lgbm_1|L|refit False`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 1.61 [1.58-1.69] | 1.00 | 1.57 |
| cpu-1 | 3 | 1.67 [1.59-1.69] | 0.96 | 1.88 |
| -1 | 4 | 1.68 [1.61-3.14] | 0.96 | 1.76 |
| physical | 2 | 1.66 [1.63-1.73] | 0.97 | 4.17 |

`backtest_ms|multiseries|lgbm_1|L|refit True`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 27.34 [26.25-27.70] | 1.00 | 28.93 |
| cpu-1 | 3 | 14.98 [14.77-15.00] | 1.83 | 18.02 |
| **-1** | 4 | 13.69 [13.67-13.77] | 2.00 | 17.33 |
| physical | 2 | 17.74 [17.47-18.23] | 1.54 | 20.14 |

### Bloque E, afinidad 22 (cpu_count 22)


`backtest_ms|multiseries|hgb|L|refit False`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.91 [1.85-1.93] | 1.00 | 1.98 |
| **cpu-1** | 21 | 1.34 [1.27-3.34] | 1.43 | 6.63 |

`backtest_ms|multiseries|hgb|L|refit True`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 11.94 [11.84-12.21] | 1.00 | 11.92 |
| **cpu-1** | 21 | 5.51 [5.44-5.63] | 2.17 | 5.65 |

`backtest_ms|multiseries|hgb|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.47 [0.44-0.47] | 1.00 | 0.53 |
| **cpu-1** | 21 | 0.34 [0.32-0.35] | 1.38 | 0.43 |

`backtest_ms|multiseries|hgb|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.47 [2.35-2.49] | 1.00 | 2.27 |
| **cpu-1** | 21 | 0.70 [0.64-0.74] | 3.53 | 0.67 |

`backtest_ms|multiseries|lgbm_1|L|refit False`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.95 [1.82-2.02] | 1.00 | 1.88 |
| **cpu-1** | 21 | 1.83 [1.82-1.86] | 1.07 | 1.93 |

`backtest_ms|multiseries|lgbm_1|L|refit True`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 28.68 [27.99-29.91] | 1.00 | 30.11 |
| **cpu-1** | 21 | 6.43 [6.18-6.52] | 4.46 | 6.60 |

`backtest_ms|multiseries|lgbm_1|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.24 [0.24-0.25] | 1.00 | 0.26 |
| cpu-1 | 21 | 0.26 [0.25-0.28] | 0.92 | 0.28 |

`backtest_ms|multiseries|lgbm_1|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.62 [1.53-1.83] | 1.00 | 1.79 |
| **cpu-1** | 21 | 0.57 [0.55-0.60] | 2.85 | 0.56 |

`backtest_ms|multiseries|lgbm_none|L|refit False`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 1.29 [1.24-1.31] | 1.00 | 1.19 |
| cpu-1 | 21 | 1.33 [1.24-1.39] | 0.97 | 1.44 |

`backtest_ms|multiseries|lgbm_none|L|refit True`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 12.89 [11.80-14.07] | 1.00 | 12.06 |
| **cpu-1** | 21 | 7.38 [7.13-7.68] | 1.75 | 7.20 |

`backtest_ms|multiseries|lgbm_none|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.22 [0.21-0.24] | 1.00 | 0.19 |
| **cpu-1** | 21 | 0.22 [0.21-0.25] | 1.02 | 0.28 |

`backtest_ms|multiseries|lgbm_none|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 1.00 [0.96-1.01] | 1.00 | 1.03 |
| cpu-1 | 21 | 1.06 [0.97-1.14] | 0.95 | 1.12 |

`backtest_ms|multiseries|ridge|L|refit False`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.61 [0.58-0.66] | 1.00 | 0.62 |
| cpu-1 | 21 | 0.82 [0.76-0.94] | 0.75 | 0.84 |

`backtest_ms|multiseries|ridge|L|refit True`, folds/estimadores 20

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.68 [1.64-1.75] | 1.00 | 1.72 |
| **cpu-1** | 21 | 1.32 [1.24-1.33] | 1.28 | 1.19 |

`backtest_ms|multiseries|ridge|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.07 [0.06-0.07] | 1.00 | 0.06 |
| cpu-1 | 21 | 0.08 [0.07-0.08] | 0.87 | 0.08 |

`backtest_ms|multiseries|ridge|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 0.11 [0.10-0.12] | 1.00 | 0.11 |
| **cpu-1** | 21 | 0.09 [0.09-0.13] | 1.16 | 0.12 |

### Bloque F, afinidad 22 (cpu_count 22)


`backtest_stats|stats|arima|L|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.84 [2.76-2.94] | 1.00 | 2.85 |
| **cpu-1** | 21 | 1.23 [1.11-1.34] | 2.32 | 1.14 |

`backtest_stats|stats|arima|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 1.00 [0.96-1.02] | 1.00 | 0.96 |
| **cpu-1** | 21 | 0.44 [0.32-4.83] | 2.29 | 6.54 |

`backtest_stats|stats|sarimax|L|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 12.89 [12.70-13.53] | 1.00 | 14.57 |

`backtest_stats|stats|sarimax|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 3.79 [3.76-3.95] | 1.00 | 3.91 |
| **cpu-1** | 21 | 1.49 [1.40-1.50] | 2.55 | 1.48 |

`backtest|equivalent_date|none|L|refit False`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.02 [0.02-0.03] | 1.00 | 0.02 |
| cpu-1 | 21 | 0.03 [0.02-0.03] | 0.74 | 0.03 |

`backtest|equivalent_date|none|L|refit True`, folds/estimadores 18

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.02 [0.02-0.03] | 1.00 | 0.02 |
| cpu-1 | 21 | 0.03 [0.03-0.03] | 0.89 | 0.03 |

`backtest|equivalent_date|none|S|refit False`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.01 [0.01-0.01] | 1.00 | 0.01 |
| cpu-1 | 21 | 0.01 [0.01-0.01] | 0.50 | 0.01 |

`backtest|equivalent_date|none|S|refit True`, folds/estimadores 10

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| **1** | 1 | 0.01 [0.01-0.01] | 1.00 | 0.01 |
| cpu-1 | 21 | 0.01 [0.01-0.03] | 0.50 | 0.01 |

### Bloque G, afinidad 22 (cpu_count 22)


`grid_ms|multiseries|lgbm_1|L|refit False`, folds/estimadores 80

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 6.08 [5.90-6.48] | 1.00 | 6.45 |
| **cpu-1** | 21 | 5.96 [5.93-6.08] | 1.02 | 6.43 |

`grid|recursive|hgb|L|refit False`, folds/estimadores 72

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 16.21 [15.98-16.45] | 1.00 | 16.94 |
| **cpu-1** | 21 | 3.42 [3.34-3.49] | 4.75 | 3.50 |

`grid|recursive|hgb|L|refit True`, folds/estimadores 72

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 38.97 [37.82-39.66] | 1.00 | 36.90 |
| **cpu-1** | 21 | 6.67 [6.17-6.83] | 5.85 | 15.28 |

`grid|recursive|lgbm_1|L|refit False`, folds/estimadores 72

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 2.06 [1.95-2.21] | 1.00 | 1.90 |
| **cpu-1** | 21 | 1.38 [1.28-1.54] | 1.50 | 1.52 |

`grid|recursive|lgbm_1|L|refit True`, folds/estimadores 72

| n_jobs | resuelto | mediana s [mín-máx] | ×(vs 1) | 1.ª llamada s |
|---|---|---|---|---|
| 1 | 1 | 14.69 [14.00-15.45] | 1.00 | 15.35 |
| **cpu-1** | 21 | 3.46 [3.26-3.53] | 4.25 | 3.31 |

### Equivalencia de predicciones frente a n_jobs=1

575 celdas comparadas; 5 con diferencia > 1e-8.

| block | affinity | case_id | n_jobs_label | max_abs_diff |
|---|---|---|---|---|
| B | 22 | op=backtest|forecaster=classifier|estimator=logreg|size=L|refit=True | -1 | 1.0 |
| B | 22 | op=backtest|forecaster=classifier|estimator=logreg|size=L|refit=True | cap8 | 1.0 |
| B | 22 | op=backtest|forecaster=classifier|estimator=logreg|size=L|refit=True | cpu-1 | 1.0 |
| B | 22 | op=backtest|forecaster=classifier|estimator=logreg|size=L|refit=True | physical | 1.0 |
| F | 22 | op=backtest_stats|forecaster=stats|estimator=sarimax|size=S|refit=True | cpu-1 | 0.006373431323964951 |