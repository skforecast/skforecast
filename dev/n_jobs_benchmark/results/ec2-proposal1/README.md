# Propuesta 1 (serializar el forecaster una vez): antes y después en la EC2

Máquina: EC2 r5a.4xlarge, AMD EPYC 7571, 16 CPU (8 núcleos); 8 CPU emulados
con la afinidad. Python 3.12.13, numpy 2.5.4, pandas 2.3.3,
scikit-learn 1.9.1, joblib 1.6.0, lightgbm 4.7.0, xgboost 3.4.1,
catboost 1.2.10.

- `before/`: commit c3138f3 (`0.28.x` antes del cambio).
- `after/`: rama `perf/serialize-forecaster-once` (commit cf3c05b).

Mismo plan en las dos pasadas, una repetición, 158 medidas cada una:

```bash
python bench_n_jobs.py run --cpus 16 8 --max-priority 4 --estimators ridge lgbm_1t xgb hgb rf catboost --reps 1 --blocks bt_norefit bt_refit gs_norefit gs_refit_small --out <carpeta>
```

Tabla generada con `python compare.py .` (columnas como en
`../local-4vcpu-proposal1/README.md`).

## Resumen

- 0 fallos, carga externa máxima 1,7 %, predicciones idénticas en las 158
  medidas.
- Control: las medidas con un proceso suman lo mismo antes y después
  (16 CPU: 676,5 → 675,4 s; 8 CPU: 713,2 → 714,3 s en caliente). El ruido
  es mucho menor que en la máquina de 4 vCPU.
- Medidas en paralelo (40 por tamaño), en caliente: 605,6 → 475,6 s (−21 %)
  con 16 CPU y 628,0 → 543,9 s (−13 %) con 8. En frío: −12 % y −9 %.
- Medidas con `'auto'` (11 por tamaño), en caliente: 121,0 → 101,1 s (−16 %)
  y 139,1 → 125,3 s (−10 %). En frío: −7 % y −6 %.
- XGBoost, ajuste posterior en el proceso principal: 6,4 → 1,7 s con 16 CPU
  y 4,4 → 2,1 s con 8: la contaminación desaparece por completo.
- Búsqueda con refit de XGBoost, 16 CPU (la mayor pérdida de `'auto'`):
  45,1 → 31,0 s en caliente (un proceso: 31,5) y 46,8 → 37,3 en frío
  (un proceso: 31,8). Con 8 CPU, 39,0 → 32,1 (un proceso: 38,7).
- XGBoost con refit y 6 folds, 16 CPU, `'auto'` (6 procesos): en caliente
  12,6 → 7,7 s (un proceso: 10,5); antes perdía, ahora gana 1,4 veces.
- Sin refit en paralelo: XGBoost y LightGBM con un hilo de 1,9 a 3,3 veces
  más rápidos en caliente, HistGB de 1,6 a 2,4 veces.
- CatBoost (0,6 MB serializado) no cambia. RandomForest (40 MB) mejora del
  1 al 9 %.
- Con refit y 16 folds, HistGB y LightGBM mejoran del 1 al 9 %: domina el
  ajuste de cada fold.

## Sin refit: un proceso frente al paralelo, después del cambio

Segundos en frío / en caliente. La regla actual elige un proceso en el
backtesting suelto y paraleliza solo las búsquedas de HistGB y CatBoost.

| CPU | Caso | Estimador | Un proceso | En paralelo | `'auto'` |
|---|---|---|---|---|---|
| 16 | backtesting | HistGB | 19,1 / 19,4 | 8,0 / 4,8 | 1 proceso |
| 8 | backtesting | HistGB | 19,8 / 19,8 | 7,8 / 5,6 | 1 proceso |
| 16 | backtesting | XGBoost | 3,0 / 2,8 | 5,5 / 2,3 | 1 proceso |
| 16 | backtesting | LightGBM 1 hilo | 3,3 / 3,4 | 6,4 / 3,3 | 1 proceso |
| 16 | backtesting | CatBoost | 3,9 / 3,8 | 5,4 / 2,3 | 1 proceso |
| 16 | backtesting | RandomForest | 21,9 / 21,9 | 33,8 / 32,5 | 1 proceso |
| 16 | búsqueda | HistGB | 27,4 / 27,5 | 13,2 / 10,2 | paralelo |
| 16 | búsqueda | CatBoost | 7,8 / 7,7 | 8,4 / 6,1 | paralelo |
| 16 | búsqueda | XGBoost | 6,6 / 6,5 | 8,7 / 6,1 | 1 proceso |

El backtesting suelto de HistGB sin refit es ahora el caso en el que
`'auto'` más pierde: 2,4 veces en frío y 4 veces en caliente (antes, 1,4 y
1,7). Es la evidencia para la propuesta 3.

## Tabla completa

```
cpus case           est      cfg   nj |  1st b  1st a    r |  nxt b  nxt a    r | fitAfter b/a | pickleMB
  16 bt_norefit     catboost 1      1 |   3.82   3.87 1.01 |   3.81   3.81 1.00 |  1.73/ 1.71 |  0.6
  16 bt_norefit     catboost half   8 |   4.72   4.63 0.98 |   2.32   2.26 0.97 |  1.71/ 1.71 |  0.6
  16 bt_norefit     catboost max   15 |   5.50   5.36 0.97 |   2.39   2.25 0.94 |  1.71/ 1.71 |  0.6
  16 bt_norefit     hgb      1      1 |  19.05  19.13 1.00 |  19.17  19.40 1.01 |  2.18/ 2.19 |  1.9
  16 bt_norefit     hgb      auto   1 |  19.66  18.93 0.96 |  19.70  19.01 0.97 |  2.19/ 2.18 |  1.9
  16 bt_norefit     hgb      half   8 |  13.64   7.65 0.56 |  11.78   5.29 0.45 |  2.18/ 2.24 |  1.9
  16 bt_norefit     hgb      max   15 |  13.71   7.96 0.58 |  11.40   4.84 0.42 |  2.19/ 2.23 |  1.9
  16 bt_norefit     lgbm_1t  1      1 |   3.35   3.34 1.00 |   3.36   3.35 1.00 |  3.00/ 2.99 |  1.4
  16 bt_norefit     lgbm_1t  max   15 |  12.89   6.38 0.50 |   8.70   3.29 0.38 |  3.03/ 2.99 |  1.4
  16 bt_norefit     rf       1      1 |  21.86  21.92 1.00 |  21.81  21.94 1.01 | 19.39/19.44 | 40.4
  16 bt_norefit     rf       max   15 |  36.05  33.84 0.94 |  34.92  32.51 0.93 | 19.45/19.42 | 40.4
  16 bt_norefit     ridge    1      1 |   0.25   0.25 1.00 |   0.23   0.23 1.01 |  0.01/ 0.01 |  0.0
  16 bt_norefit     ridge    max   15 |   3.20   3.18 0.99 |   0.17   0.14 0.82 |  0.01/ 0.01 |  0.0
  16 bt_norefit     xgb      1      1 |   2.91   2.97 1.02 |   2.82   2.78 0.99 |  1.80/ 1.68 |  2.0
  16 bt_norefit     xgb      auto   1 |   2.93   2.91 0.99 |   2.83   2.81 0.99 |  1.72/ 1.70 |  2.0
  16 bt_norefit     xgb      half   8 |   5.25   4.74 0.90 |   7.28   2.33 0.32 |  6.39/ 1.69 |  2.0
  16 bt_norefit     xgb      max   15 |   5.97   5.46 0.91 |   7.41   2.32 0.31 |  6.39/ 1.66 |  2.0
  16 bt_refit       catboost 1      1 |  30.41  30.36 1.00 |  30.35  30.32 1.00 |  1.73/ 1.72 |  0.6
  16 bt_refit       catboost max   15 |  23.01  22.96 1.00 |  20.09  20.21 1.01 |  1.72/ 1.72 |  0.6
  16 bt_refit       hgb      1      1 |  60.01  60.44 1.01 |  60.25  60.18 1.00 |  2.18/ 2.17 |  1.9
  16 bt_refit       hgb      all   16 |  15.89  15.47 0.97 |  12.56  12.05 0.96 |  2.18/ 2.15 |  1.9
  16 bt_refit       hgb      auto  15 |  15.55  15.21 0.98 |  12.36  11.88 0.96 |  2.20/ 2.21 |  1.9
  16 bt_refit       hgb      half   8 |  16.41  16.41 1.00 |  14.03  13.39 0.95 |  2.19/ 2.19 |  1.9
  16 bt_refit       hgb      max   15 |  15.75  15.15 0.96 |  12.55  11.91 0.95 |  2.17/ 2.16 |  1.9
  16 bt_refit       lgbm_1t  1      1 |  44.63  44.66 1.00 |  44.66  44.67 1.00 |  3.00/ 3.00 |  1.4
  16 bt_refit       lgbm_1t  all   16 |  11.67  10.90 0.93 |   8.52   7.80 0.91 |  2.99/ 2.98 |  1.4
  16 bt_refit       lgbm_1t  half   8 |  11.85  11.40 0.96 |   9.63   9.11 0.95 |  2.99/ 2.99 |  1.4
  16 bt_refit       lgbm_1t  max   15 |  11.66  10.82 0.93 |   8.57   7.81 0.91 |  2.99/ 2.98 |  1.4
  16 bt_refit       ridge    1      1 |   0.39   0.39 1.00 |   0.37   0.37 1.00 |  0.01/ 0.01 |  0.0
  16 bt_refit       ridge    max   15 |   3.24   3.25 1.00 |   0.11   0.10 0.98 |  0.01/ 0.01 |  0.0
  16 bt_refit       xgb      1      1 |  28.42  28.91 1.02 |  28.00  28.02 1.00 |  1.68/ 1.69 |  2.0
  16 bt_refit       xgb      all   16 |  18.01  18.08 1.00 |  19.53  14.72 0.75 |  6.38/ 1.72 |  2.0
  16 bt_refit       xgb      auto  15 |  18.03  17.94 1.00 |  19.58  14.90 0.76 |  6.40/ 1.71 |  2.0
  16 bt_refit       xgb      half   8 |  17.71  17.51 0.99 |  20.11  15.58 0.77 |  6.43/ 1.74 |  2.0
  16 bt_refit       xgb      max   15 |  17.97  17.86 0.99 |  19.62  14.80 0.75 |  6.43/ 1.72 |  2.0
  16 bt_refit_small catboost 1      1 |  11.17  11.17 1.00 |  11.14  11.12 1.00 |  1.74/ 1.72 |  0.6
  16 bt_refit_small catboost max   15 |  10.21  10.17 1.00 |   9.81   9.70 0.99 |  1.71/ 1.71 |  0.6
  16 bt_refit_small hgb      1      1 |  20.16  20.29 1.01 |  20.12  20.32 1.01 |  2.20/ 2.19 |  1.9
  16 bt_refit_small hgb      auto   6 |   9.04   9.14 1.01 |   7.16   6.66 0.93 |  2.19/ 2.19 |  1.9
  16 bt_refit_small hgb      max   15 |   9.91   9.69 0.98 |   9.75   9.24 0.95 |  2.17/ 2.15 |  1.9
  16 bt_refit_small lgbm_1t  1      1 |  16.71  16.67 1.00 |  16.71  16.70 1.00 |  3.00/ 3.00 |  1.4
  16 bt_refit_small lgbm_1t  auto   6 |   8.37   8.08 0.97 |   6.21   5.84 0.94 |  3.00/ 2.99 |  1.4
  16 bt_refit_small lgbm_1t  max   15 |   8.43   8.13 0.96 |   8.12   7.76 0.96 |  3.00/ 2.99 |  1.4
  16 bt_refit_small rf       1      1 | 116.75 116.62 1.00 | 116.71 116.67 1.00 | 19.41/19.43 | 40.4
  16 bt_refit_small rf       max   15 |  42.12  41.91 1.00 |  41.89  41.54 0.99 | 19.40/19.34 | 40.4
  16 bt_refit_small ridge    1      1 |   0.14   0.14 1.01 |   0.12   0.12 1.00 |  0.01/ 0.01 |  0.0
  16 bt_refit_small ridge    max   15 |   2.41   2.40 0.99 |   1.99   2.02 1.01 |  0.01/ 0.01 |  0.0
  16 bt_refit_small xgb      1      1 |  10.79  10.56 0.98 |  10.48  10.50 1.00 |  1.69/ 1.69 |  2.0
  16 bt_refit_small xgb      auto   6 |   9.95   9.96 1.00 |  12.60   7.73 0.61 |  6.43/ 1.70 |  2.0
  16 bt_refit_small xgb      max   15 |  12.78  12.59 0.99 |  16.74  12.15 0.73 |  6.43/ 1.68 |  2.0
  16 gs_norefit     catboost 1      1 |   7.77   7.75 1.00 |   7.67   7.69 1.00 |  1.72/ 1.72 |  0.6
  16 gs_norefit     catboost auto  15 |   9.22   9.12 0.99 |   6.21   6.15 0.99 |  1.71/ 1.72 |  0.6
  16 gs_norefit     catboost half   8 |   8.46   8.42 1.00 |   6.13   6.09 0.99 |  1.71/ 1.71 |  0.6
  16 gs_norefit     catboost max   15 |   9.23   9.18 1.00 |   6.25   6.10 0.98 |  1.72/ 1.70 |  0.6
  16 gs_norefit     hgb      1      1 |  27.27  27.44 1.01 |  27.13  27.53 1.01 |  2.21/ 2.21 |  1.9
  16 gs_norefit     hgb      auto  15 |  20.61  13.23 0.64 |  18.68  10.15 0.54 |  2.15/ 2.18 |  1.9
  16 gs_norefit     hgb      half   8 |  20.47  12.92 0.63 |  18.53  10.69 0.58 |  2.18/ 2.19 |  1.9
  16 gs_norefit     hgb      max   15 |  20.74  13.19 0.64 |  18.54  10.20 0.55 |  2.19/ 2.17 |  1.9
  16 gs_norefit     lgbm_1t  1      1 |   9.41   9.43 1.00 |   9.40   9.40 1.00 |  2.99/ 2.99 |  1.4
  16 gs_norefit     lgbm_1t  auto   1 |   9.42   9.42 1.00 |   9.38   9.41 1.00 |  2.98/ 2.99 |  1.4
  16 gs_norefit     lgbm_1t  max   15 |  20.66  12.86 0.62 |  18.91   9.80 0.52 |  3.01/ 2.98 |  1.4
  16 gs_norefit     rf       1      1 |  54.31  54.32 1.00 |  54.24  54.25 1.00 | 19.42/19.45 | 40.4
  16 gs_norefit     rf       max   15 |  63.75  60.88 0.95 |  63.07  59.30 0.94 | 19.53/19.34 | 40.4
  16 gs_norefit     ridge    1      1 |   0.32   0.32 1.01 |   0.30   0.30 1.01 |  0.01/ 0.01 |  0.0
  16 gs_norefit     ridge    max   15 |   3.36   3.37 1.00 |   0.28   0.25 0.89 |  0.01/ 0.01 |  0.0
  16 gs_norefit     xgb      1      1 |   6.54   6.55 1.00 |   6.42   6.45 1.01 |  1.71/ 1.69 |  2.0
  16 gs_norefit     xgb      auto   1 |   6.48   6.83 1.05 |   6.34   6.53 1.03 |  1.69/ 1.72 |  2.0
  16 gs_norefit     xgb      half   8 |  18.41   8.69 0.47 |  20.61   6.13 0.30 |  6.39/ 1.69 |  2.0
  16 gs_norefit     xgb      max   15 |  19.25   9.23 0.48 |  20.98   6.26 0.30 |  6.39/ 1.71 |  2.0
  16 gs_refit_small catboost 1      1 |  33.49  33.49 1.00 |  33.39  33.46 1.00 |  1.73/ 1.73 |  0.6
  16 gs_refit_small catboost max   15 |  28.47  28.34 1.00 |  23.66  23.44 0.99 |  1.72/ 1.71 |  0.6
  16 gs_refit_small hgb      1      1 |  63.07  61.01 0.97 |  62.98  60.98 0.97 |  2.22/ 2.17 |  1.9
  16 gs_refit_small hgb      max   15 |  30.05  29.31 0.98 |  23.67  22.54 0.95 |  2.17/ 2.17 |  1.9
  16 gs_refit_small lgbm_1t  1      1 |  52.80  53.02 1.00 |  52.85  52.93 1.00 |  2.99/ 3.00 |  1.4
  16 gs_refit_small lgbm_1t  max   15 |  25.33  24.58 0.97 |  19.52  18.43 0.94 |  2.99/ 2.99 |  1.4
  16 gs_refit_small ridge    1      1 |   0.40   0.39 0.99 |   0.38   0.37 0.99 |  0.01/ 0.01 |  0.0
  16 gs_refit_small ridge    max   15 |   6.37   6.42 1.01 |   0.28   0.26 0.93 |  0.01/ 0.01 |  0.0
  16 gs_refit_small xgb      1      1 |  31.83  32.18 1.01 |  31.65  31.51 1.00 |  1.68/ 1.67 |  2.0
  16 gs_refit_small xgb      max   15 |  46.79  37.27 0.80 |  45.11  30.99 0.69 |  6.44/ 1.72 |  2.0
   8 bt_norefit     catboost 1      1 |   4.08   4.07 1.00 |   4.01   4.00 1.00 |  2.19/ 2.18 |  0.6
   8 bt_norefit     catboost half   4 |   5.14   5.13 1.00 |   2.87   2.84 0.99 |  2.17/ 2.18 |  0.6
   8 bt_norefit     catboost max    7 |   5.10   5.01 0.98 |   2.78   2.77 1.00 |  2.17/ 2.17 |  0.6
   8 bt_norefit     hgb      1      1 |  19.54  19.78 1.01 |  19.61  19.80 1.01 |  2.19/ 2.21 |  1.9
   8 bt_norefit     hgb      auto   1 |  19.65  19.14 0.97 |  20.04  19.44 0.97 |  2.21/ 2.15 |  1.9
   8 bt_norefit     hgb      half   4 |  13.79   9.80 0.71 |  11.72   7.55 0.64 |  2.19/ 2.19 |  1.9
   8 bt_norefit     hgb      max    7 |  13.42   7.82 0.58 |  11.67   5.61 0.48 |  2.25/ 2.28 |  1.9
   8 bt_norefit     lgbm_1t  1      1 |   3.34   3.34 1.00 |   3.36   3.35 1.00 |  3.00/ 2.99 |  1.4
   8 bt_norefit     lgbm_1t  max    7 |  12.73   5.57 0.44 |   7.66   3.28 0.43 |  3.00/ 2.98 |  1.4
   8 bt_norefit     rf       1      1 |  21.82  21.86 1.00 |  21.79  21.83 1.00 | 19.36/19.41 | 40.4
   8 bt_norefit     rf       max    7 |  36.63  33.89 0.93 |  35.30  32.27 0.91 | 19.37/19.41 | 40.4
   8 bt_norefit     ridge    1      1 |   0.25   0.25 1.02 |   0.22   0.23 1.02 |  0.01/ 0.01 |  0.0
   8 bt_norefit     ridge    max    7 |   2.43   2.41 0.99 |   0.14   0.13 0.92 |  0.01/ 0.01 |  0.0
   8 bt_norefit     xgb      1      1 |   3.29   3.28 1.00 |   3.19   3.19 1.00 |  2.11/ 2.11 |  2.0
   8 bt_norefit     xgb      auto   1 |   3.30   3.27 0.99 |   3.20   3.17 0.99 |  2.12/ 2.11 |  2.0
   8 bt_norefit     xgb      half   4 |   5.42   5.16 0.95 |   5.20   2.90 0.56 |  4.41/ 2.07 |  2.0
   8 bt_norefit     xgb      max    7 |   5.55   5.00 0.90 |   5.30   2.69 0.51 |  4.44/ 2.11 |  2.0
   8 bt_refit       catboost 1      1 |  37.43  37.41 1.00 |  37.45  37.37 1.00 |  2.18/ 2.19 |  0.6
   8 bt_refit       catboost max    7 |  29.37  29.42 1.00 |  27.29  27.22 1.00 |  2.17/ 2.18 |  0.6
   8 bt_refit       hgb      1      1 |  67.76  60.92 0.90 |  60.91  61.48 1.01 |  2.18/ 2.22 |  1.9
   8 bt_refit       hgb      all    8 |  16.88  16.23 0.96 |  14.41  13.73 0.95 |  2.26/ 2.24 |  1.9
   8 bt_refit       hgb      auto   7 |  20.27  19.91 0.98 |  18.09  17.50 0.97 |  2.16/ 2.15 |  1.9
   8 bt_refit       hgb      half   4 |  22.97  23.46 1.02 |  20.79  20.56 0.99 |  2.29/ 2.32 |  1.9
   8 bt_refit       hgb      max    7 |  20.01  20.00 1.00 |  17.84  17.43 0.98 |  2.24/ 2.20 |  1.9
   8 bt_refit       lgbm_1t  1      1 |  44.59  44.51 1.00 |  44.62  44.76 1.00 |  3.00/ 3.00 |  1.4
   8 bt_refit       lgbm_1t  all    8 |  11.80  11.39 0.97 |   9.61   9.04 0.94 |  2.98/ 2.99 |  1.4
   8 bt_refit       lgbm_1t  half   4 |  16.91  16.76 0.99 |  14.76  14.58 0.99 |  3.00/ 2.98 |  1.4
   8 bt_refit       lgbm_1t  max    7 |  13.99  13.89 0.99 |  11.78  11.68 0.99 |  2.98/ 2.99 |  1.4
   8 bt_refit       ridge    1      1 |   0.39   0.39 1.00 |   0.36   0.36 1.00 |  0.01/ 0.01 |  0.0
   8 bt_refit       ridge    max    7 |   2.40   2.38 0.99 |   0.12   0.10 0.91 |  0.01/ 0.01 |  0.0
   8 bt_refit       xgb      1      1 |  35.55  34.44 0.97 |  34.72  34.39 0.99 |  2.09/ 2.09 |  2.0
   8 bt_refit       xgb      all    8 |  21.41  21.35 1.00 |  21.47  19.03 0.89 |  4.46/ 2.11 |  2.0
   8 bt_refit       xgb      auto   7 |  28.14  28.26 1.00 |  28.16  25.85 0.92 |  4.45/ 2.09 |  2.0
   8 bt_refit       xgb      half   4 |  24.10  24.31 1.01 |  24.47  22.22 0.91 |  4.42/ 2.14 |  2.0
   8 bt_refit       xgb      max    7 |  28.20  28.13 1.00 |  28.32  25.89 0.91 |  4.43/ 2.17 |  2.0
   8 bt_refit_small catboost 1      1 |  13.79  13.79 1.00 |  13.74  13.75 1.00 |  2.18/ 2.18 |  0.6
   8 bt_refit_small catboost max    7 |  12.67  12.67 1.00 |  10.79  10.76 1.00 |  2.17/ 2.18 |  0.6
   8 bt_refit_small hgb      1      1 |  20.35  20.07 0.99 |  20.29  20.34 1.00 |  2.20/ 2.18 |  1.9
   8 bt_refit_small hgb      auto   6 |  10.05   9.56 0.95 |   7.61   7.16 0.94 |  2.16/ 2.17 |  1.9
   8 bt_refit_small hgb      max    7 |   9.92   9.54 0.96 |   7.63   7.22 0.95 |  2.15/ 2.25 |  1.9
   8 bt_refit_small lgbm_1t  1      1 |  16.64  16.75 1.01 |  16.67  16.75 1.01 |  3.00/ 3.01 |  1.4
   8 bt_refit_small lgbm_1t  auto   6 |   8.29   8.01 0.97 |   6.21   5.80 0.93 |  2.98/ 2.98 |  1.4
   8 bt_refit_small lgbm_1t  max    7 |  10.57   7.99 0.76 |   8.49   5.80 0.68 |  5.22/ 2.99 |  1.4
   8 bt_refit_small rf       1      1 | 116.41 116.36 1.00 | 116.36 116.40 1.00 | 19.38/19.38 | 40.4
   8 bt_refit_small rf       max    7 |  41.99  41.75 0.99 |  39.76  39.51 0.99 | 19.35/19.39 | 40.4
   8 bt_refit_small ridge    1      1 |   0.14   0.14 1.00 |   0.13   0.12 0.98 |  0.01/ 0.01 |  0.0
   8 bt_refit_small ridge    max    7 |   2.28   2.30 1.01 |   0.99   0.98 0.99 |  0.01/ 0.01 |  0.0
   8 bt_refit_small xgb      1      1 |  13.19  13.02 0.99 |  12.98  12.73 0.98 |  2.11/ 2.09 |  2.0
   8 bt_refit_small xgb      auto   6 |  12.79  12.95 1.01 |  12.89  10.92 0.85 |  4.48/ 2.15 |  2.0
   8 bt_refit_small xgb      max    7 |  12.81  12.78 1.00 |  12.86  10.58 0.82 |  4.44/ 2.12 |  2.0
   8 gs_norefit     catboost 1      1 |   8.80   8.79 1.00 |   8.73   8.74 1.00 |  2.18/ 2.18 |  0.6
   8 gs_norefit     catboost auto   7 |   9.75   9.64 0.99 |   7.50   7.39 0.98 |  2.17/ 2.17 |  0.6
   8 gs_norefit     catboost half   4 |   9.76   9.72 1.00 |   7.50   7.52 1.00 |  2.18/ 2.17 |  0.6
   8 gs_norefit     catboost max    7 |   9.73   9.74 1.00 |   7.47   7.47 1.00 |  2.18/ 2.19 |  0.6
   8 gs_norefit     hgb      1      1 |  27.18  27.65 1.02 |  27.29  27.78 1.02 |  2.23/ 2.21 |  1.9
   8 gs_norefit     hgb      auto   7 |  20.42  13.13 0.64 |  18.45  11.05 0.60 |  2.19/ 2.18 |  1.9
   8 gs_norefit     hgb      half   4 |  20.15  15.77 0.78 |  18.10  13.47 0.74 |  2.20/ 2.24 |  1.9
   8 gs_norefit     hgb      max    7 |  20.39  13.16 0.65 |  18.40  10.98 0.60 |  2.20/ 2.17 |  1.9
   8 gs_norefit     lgbm_1t  1      1 |   9.47   9.43 1.00 |   9.43   9.40 1.00 |  2.99/ 2.99 |  1.4
   8 gs_norefit     lgbm_1t  auto   1 |   9.42   9.44 1.00 |   9.39   9.42 1.00 |  2.99/ 3.00 |  1.4
   8 gs_norefit     lgbm_1t  max    7 |  19.27  11.98 0.62 |  17.05   9.69 0.57 |  2.99/ 2.99 |  1.4
   8 gs_norefit     rf       1      1 |  54.32  54.16 1.00 |  54.29  54.12 1.00 | 19.42/19.37 | 40.4
   8 gs_norefit     rf       max    7 |  62.37  60.47 0.97 |  61.98  58.50 0.94 | 19.34/19.38 | 40.4
   8 gs_norefit     ridge    1      1 |   0.32   0.32 1.00 |   0.30   0.30 1.00 |  0.01/ 0.01 |  0.0
   8 gs_norefit     ridge    max    7 |   2.51   2.49 1.00 |   0.25   0.22 0.88 |  0.01/ 0.01 |  0.0
   8 gs_norefit     xgb      1      1 |   7.72   7.73 1.00 |   7.57   7.50 0.99 |  2.12/ 2.09 |  2.0
   8 gs_norefit     xgb      auto   1 |   7.67   7.68 1.00 |   7.55   7.57 1.00 |  2.11/ 2.11 |  2.0
   8 gs_norefit     xgb      half   4 |  14.65   9.74 0.66 |  14.52   7.47 0.51 |  4.42/ 2.09 |  2.0
   8 gs_norefit     xgb      max    7 |  14.78   9.53 0.64 |  14.71   7.24 0.49 |  4.44/ 2.07 |  2.0
   8 gs_refit_small catboost 1      1 |  41.25  41.25 1.00 |  41.27  41.22 1.00 |  2.18/ 2.18 |  0.6
   8 gs_refit_small catboost max    7 |  34.07  33.98 1.00 |  31.53  31.91 1.01 |  2.18/ 2.19 |  0.6
   8 gs_refit_small hgb      1      1 |  61.35  62.07 1.01 |  61.44  62.23 1.01 |  2.26/ 2.21 |  1.9
   8 gs_refit_small hgb      max    7 |  25.98  24.72 0.95 |  23.70  22.42 0.95 |  2.22/ 2.18 |  1.9
   8 gs_refit_small lgbm_1t  1      1 |  53.10  52.99 1.00 |  53.12  53.11 1.00 |  3.00/ 3.00 |  1.4
   8 gs_refit_small lgbm_1t  max    7 |  21.68  20.63 0.95 |  19.53  18.36 0.94 |  3.00/ 2.98 |  1.4
   8 gs_refit_small ridge    1      1 |   0.40   0.39 0.97 |   0.38   0.37 0.97 |  0.01/ 0.01 |  0.0
   8 gs_refit_small ridge    max    7 |   4.24   4.25 1.00 |   0.20   0.20 0.99 |  0.01/ 0.01 |  0.0
   8 gs_refit_small xgb      1      1 |  39.02  39.04 1.00 |  38.93  38.65 0.99 |  2.11/ 2.09 |  2.0
   8 gs_refit_small xgb      max    7 |  38.96  34.37 0.88 |  39.02  32.11 0.82 |  4.43/ 2.08 |  2.0

Cells compared: 158
```
