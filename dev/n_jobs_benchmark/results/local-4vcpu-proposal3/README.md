# Propuesta 3 mínima (decidir por el tiempo de predicción sin refit): 4 vCPU

- `before/`: rama `perf/serialize-forecaster-once` (propuesta 1).
- `after/`: rama `perf/n-jobs-auto-prediction-workload` (commit 79fde32).

Contenedor Linux de 4 vCPU, Python 3.12.3, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, joblib 1.6.0. Una repetición:

```bash
python bench_n_jobs.py run --estimators hgb catboost xgb --reps 1 --blocks bt_norefit --max-priority 4 --out <carpeta>
```

- HistGB con `'auto'` (100 folds sin refit): antes un proceso, 13,5 s en
  frío y 12,5 en caliente; después 3 procesos, 8,1 y 6,5 (1,7 y 1,9 veces
  más rápido), lo mismo que con `n_jobs=3` a mano (7,8 y 6,0, dentro del
  ruido). Predecir el primer fold en el proceso principal no se nota.
- XGBoost con `'auto'`: un proceso antes y después.
- Predicciones idénticas en las 11 medidas.

```
cpus case           est      cfg   nj |  1st b  1st a    r |  nxt b  nxt a    r | fitAfter b/a | pickleMB
   4 bt_norefit     catboost 1      1 |   3.25   3.19 0.98 |   3.09   2.83 0.92 |  1.95/ 1.72 |  0.6
   4 bt_norefit     catboost half   2 |   4.69   4.69 1.00 |   2.66   2.69 1.01 |  1.98/ 1.86 |  0.6
   4 bt_norefit     catboost max    3 |   4.52   4.27 0.95 |   2.66   2.45 0.92 |  1.93/ 1.81 |  0.6
   4 bt_norefit     hgb      1      1 |  12.85  12.08 0.94 |  13.00  11.26 0.87 |  1.82/ 1.86 |  1.9
   4 bt_norefit     hgb      auto   1 |  13.49   8.14 0.60 |  12.45   6.50 0.52 |  1.74/ 1.81 |  1.9
   4 bt_norefit     hgb      half   2 |  10.01  10.26 1.03 |   8.36   8.02 0.96 |  1.92/ 1.65 |  1.9
   4 bt_norefit     hgb      max    3 |   7.76   8.49 1.09 |   6.02   6.31 1.05 |  1.76/ 1.69 |  1.9
   4 bt_norefit     xgb      1      1 |   3.49   3.21 0.92 |   3.64   3.12 0.86 |  2.26/ 2.42 |  2.0
   4 bt_norefit     xgb      auto   1 |   3.54   3.38 0.95 |   3.22   3.37 1.04 |  2.20/ 2.36 |  2.0
   4 bt_norefit     xgb      half   2 |   4.99   5.00 1.00 |   3.23   3.50 1.08 |  2.43/ 2.42 |  2.0
   4 bt_norefit     xgb      max    3 |   5.15   4.60 0.89 |   3.23   2.97 0.92 |  2.27/ 2.10 |  2.0

Cells compared: 11
```
