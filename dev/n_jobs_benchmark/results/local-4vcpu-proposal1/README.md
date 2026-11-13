# Propuesta 1 (serializar el forecaster una vez): antes y después, 4 vCPU

Máquina: contenedor Linux de 4 vCPU (4 núcleos físicos). Python 3.12.3,
numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, joblib 1.6.0, numba 0.68.1.

- `before/`: código de `0.28.x` (commit c3138f3).
- `after/`: rama `perf/serialize-forecaster-once` (commit 13ce144).

Mismo plan en las dos pasadas, una repetición, sin emulación de CPU:

```bash
python bench_n_jobs.py run --max-priority 4 --estimators ridge lgbm_1t xgb hgb rf --reps 1 --blocks bt_norefit bt_refit gs_norefit gs_refit_small --out <carpeta>
```

Tabla generada con `python compare.py <carpeta con before y after>`. Columnas: segundos de la primera llamada y
mediana de las siguientes, antes y después; `fitAfter`: un ajuste en el
proceso principal después de la medida; `pickleMB`: tamaño del forecaster
serializado.

## Resumen

- Control: las medidas con un proceso (código no afectado) suman lo mismo
  antes y después (423,1 → 418,6 s la primera llamada; 422,5 → 421,5 s las
  siguientes). Ruido por medida: ±10 a 15 %.
- Medidas en paralelo (33): 524,9 → 484,1 s la primera llamada (−8 %),
  480,2 → 426,0 s las siguientes (−11 %). Medidas con `'auto'` (10):
  118,6 → 120,8 s y 116,2 → 106,2 s.
- Sin refit en paralelo: LightGBM con un hilo −46 %, XGBoost −38 %,
  HistGB −19 %; búsquedas sin refit −34 % (LightGBM) y −38 % (XGBoost).
- Con refit: 0 a 16 % menos, en el límite del ruido.
- XGBoost, ajuste posterior en el proceso principal: 3,2 a 3,8 s antes,
  1,9 a 2,2 s después (lo mismo que sin paralelizar).
- RandomForest (40 MB serializado, que ahora viaja entero en cada fold): no
  empeora, 5 a 14 % menos.
- Predicciones idénticas en las 66 medidas.

## Tabla

```
case           est      cfg   nj | 1st bef 1st aft ratio | nxt bef nxt aft ratio |  fitAfter b/a | pickleMB
bt_norefit     hgb      1      1 |  11.55   10.43  0.90 |  12.15   11.49  0.95 |   2.06/  1.82 |    1.9
bt_norefit     hgb      auto   1 |  11.39   11.71  1.03 |  12.01   11.09  0.92 |   1.90/  1.75 |    1.9
bt_norefit     hgb      half   2 |  11.30    8.65  0.76 |   9.49    7.23  0.76 |   2.21/  1.81 |    1.9
bt_norefit     hgb      max    3 |   9.67    7.56  0.78 |   6.82    5.53  0.81 |   2.03/  2.02 |    1.9
bt_norefit     lgbm_1t  1      1 |   2.35    1.92  0.82 |   2.18    2.41  1.10 |   1.81/  1.65 |    1.4
bt_norefit     lgbm_1t  max    3 |   7.88    4.26  0.54 |   4.00    2.16  0.54 |   1.57/  1.62 |    1.4
bt_norefit     rf       1      1 |  13.74   14.47  1.05 |  12.88   14.90  1.16 |  12.38/ 11.60 |   40.4
bt_norefit     rf       max    3 |  25.89   22.57  0.87 |  18.71   17.75  0.95 |  11.58/ 12.57 |   40.4
bt_norefit     ridge    1      1 |   0.15    0.18  1.21 |   0.12    0.12  1.02 |   0.01/  0.01 |    0.0
bt_norefit     ridge    max    3 |   1.70    1.67  0.98 |   0.10    0.10  1.09 |   0.01/  0.01 |    0.0
bt_norefit     xgb      1      1 |   3.39    3.09  0.91 |   3.33    3.10  0.93 |   2.24/  2.03 |    2.0
bt_norefit     xgb      auto   1 |   2.90    3.65  1.26 |   3.27    3.07  0.94 |   2.13/  1.84 |    2.0
bt_norefit     xgb      half   2 |   4.65    5.03  1.08 |   4.58    3.17  0.69 |   3.27/  2.16 |    2.0
bt_norefit     xgb      max    3 |   4.70    4.14  0.88 |   4.41    2.76  0.62 |   3.36/  2.04 |    2.0
bt_refit       hgb      1      1 |  42.86   43.24  1.01 |  44.36   43.96  0.99 |   1.86/  1.78 |    1.9
bt_refit       hgb      all    4 |  16.94   16.84  0.99 |  15.90   14.79  0.93 |   1.82/  1.75 |    1.9
bt_refit       hgb      auto   3 |  19.27   20.69  1.07 |  19.44   18.72  0.96 |   1.89/  1.80 |    1.9
bt_refit       hgb      half   2 |  28.17   26.02  0.92 |  26.46   23.92  0.90 |   2.09/  1.80 |    1.9
bt_refit       hgb      max    3 |  21.77   19.30  0.89 |  20.64   19.20  0.93 |   1.71/  1.85 |    1.9
bt_refit       lgbm_1t  1      1 |  27.60   26.66  0.97 |  28.73   25.79  0.90 |   2.07/  1.76 |    1.4
bt_refit       lgbm_1t  all    4 |  11.04   10.88  0.99 |   9.82    8.89  0.91 |   1.80/  1.82 |    1.4
bt_refit       lgbm_1t  half   2 |  15.89   17.37  1.09 |  15.58   15.03  0.96 |   1.62/  1.78 |    1.4
bt_refit       lgbm_1t  max    3 |  12.43   12.69  1.02 |  11.30   10.61  0.94 |   1.63/  1.35 |    1.4
bt_refit       ridge    1      1 |   0.24    0.23  0.97 |   0.23    0.18  0.79 |   0.01/  0.01 |    0.0
bt_refit       ridge    max    3 |   1.90    1.72  0.91 |   0.11    0.11  0.98 |   0.01/  0.01 |    0.0
bt_refit       xgb      1      1 |  32.92   33.40  1.01 |  32.56   33.61  1.03 |   1.89/  2.44 |    2.0
bt_refit       xgb      all    4 |  23.67   22.63  0.96 |  23.45   21.67  0.92 |   3.41/  2.16 |    2.0
bt_refit       xgb      auto   3 |  27.58   28.21  1.02 |  29.69   24.90  0.84 |   3.40/  2.15 |    2.0
bt_refit       xgb      half   2 |  29.68   27.53  0.93 |  27.17   26.39  0.97 |   3.28/  1.98 |    2.0
bt_refit       xgb      max    3 |  27.13   28.24  1.04 |  27.63   28.74  1.04 |   3.22/  2.14 |    2.0
bt_refit_small hgb      1      1 |  13.76   14.87  1.08 |  15.15   14.55  0.96 |   1.92/  1.83 |    1.9
bt_refit_small hgb      auto   3 |   9.71   10.36  1.07 |   8.11    8.06  0.99 |   1.84/  2.19 |    1.9
bt_refit_small hgb      max    3 |   9.40    9.40  1.00 |   8.39    8.09  0.96 |   1.93/  1.78 |    1.9
bt_refit_small lgbm_1t  1      1 |   9.28    8.51  0.92 |   9.76   10.47  1.07 |   1.88/  1.97 |    1.4
bt_refit_small lgbm_1t  auto   3 |   6.84    6.57  0.96 |   5.66    5.19  0.92 |   1.44/  1.67 |    1.4
bt_refit_small lgbm_1t  max    3 |   7.62    6.67  0.88 |   5.30    5.23  0.99 |   1.91/  1.68 |    1.4
bt_refit_small rf       1      1 |  71.89   69.40  0.97 |  69.67   69.59  1.00 |  10.55/ 11.91 |   40.4
bt_refit_small rf       max    3 |  39.78   37.47  0.94 |  38.19   33.71  0.88 |  12.60/ 11.28 |   40.4
bt_refit_small ridge    1      1 |   0.11    0.10  0.96 |   0.08    0.08  0.92 |   0.01/  0.01 |    0.0
bt_refit_small ridge    max    3 |   1.69    1.66  0.98 |   0.06    0.06  0.96 |   0.01/  0.01 |    0.0
bt_refit_small xgb      1      1 |  12.63   13.20  1.04 |  13.09   12.66  0.97 |   2.05/  2.07 |    2.0
bt_refit_small xgb      auto   3 |  14.75   14.27  0.97 |  12.74   11.66  0.92 |   3.28/  1.99 |    2.0
bt_refit_small xgb      max    3 |  13.29   13.79  1.04 |  13.16   11.92  0.91 |   3.41/  2.22 |    2.0
gs_norefit     hgb      1      1 |  16.96   18.84  1.11 |  17.81   16.55  0.93 |   1.82/  1.82 |    1.9
gs_norefit     hgb      auto   3 |  13.98   12.53  0.90 |  13.06   10.33  0.79 |   1.80/  1.78 |    1.9
gs_norefit     hgb      half   2 |  16.03   13.49  0.84 |  14.59   12.77  0.88 |   1.85/  1.65 |    1.9
gs_norefit     hgb      max    3 |  13.62   11.99  0.88 |  12.36   10.23  0.83 |   2.23/  1.91 |    1.9
gs_norefit     lgbm_1t  1      1 |   5.78    5.46  0.95 |   4.69    5.17  1.10 |   1.73/  1.59 |    1.4
gs_norefit     lgbm_1t  auto   1 |   5.27    4.73  0.90 |   5.33    5.56  1.04 |   1.61/  1.73 |    1.4
gs_norefit     lgbm_1t  max    3 |  10.67    7.48  0.70 |   9.32    6.19  0.66 |   1.78/  1.76 |    1.4
gs_norefit     rf       1      1 |  33.92   32.91  0.97 |  35.01   32.00  0.91 |  12.26/ 11.40 |   40.4
gs_norefit     rf       max    3 |  40.98   38.57  0.94 |  41.23   35.40  0.86 |  12.31/ 11.29 |   40.4
gs_norefit     ridge    1      1 |   0.19    0.17  0.89 |   0.17    0.17  0.98 |   0.01/  0.01 |    0.0
gs_norefit     ridge    max    3 |   1.93    1.60  0.83 |   0.21    0.18  0.85 |   0.01/  0.01 |    0.0
gs_norefit     xgb      1      1 |   7.85    7.51  0.96 |   7.68    8.12  1.06 |   2.19/  2.35 |    2.0
gs_norefit     xgb      auto   1 |   6.89    8.08  1.17 |   6.92    7.58  1.09 |   1.67/  2.06 |    2.0
gs_norefit     xgb      half   2 |  11.88    9.53  0.80 |  11.77    7.68  0.65 |   3.31/  2.10 |    2.0
gs_norefit     xgb      max    3 |  12.30    8.78  0.71 |  11.64    7.26  0.62 |   3.52/  2.02 |    2.0
gs_refit_small hgb      1      1 |  44.89   45.34  1.01 |  45.58   46.80  1.03 |   2.00/  1.79 |    1.9
gs_refit_small hgb      max    3 |  29.03   26.36  0.91 |  26.45   24.43  0.92 |   1.88/  1.80 |    1.9
gs_refit_small lgbm_1t  1      1 |  32.05   30.09  0.94 |  29.27   32.56  1.11 |   1.48/  1.62 |    1.4
gs_refit_small lgbm_1t  max    3 |  17.56   17.54  1.00 |  17.60   16.83  0.96 |   1.86/  1.95 |    1.4
gs_refit_small ridge    1      1 |   0.28    0.21  0.75 |   0.26    0.19  0.72 |   0.01/  0.01 |    0.0
gs_refit_small ridge    max    3 |   2.02    1.74  0.87 |   0.20    0.17  0.88 |   0.01/  0.01 |    0.0
gs_refit_small xgb      1      1 |  38.73   38.35  0.99 |  37.77   36.98  0.98 |   2.09/  2.09 |    2.0
gs_refit_small xgb      max    3 |  42.69   40.93  0.96 |  43.54   37.82  0.87 |   3.75/  1.90 |    2.0

Cells compared: 66; only before: 0; only after: 0
```
