# Revisión final: arreglos de la serialización única, 4 vCPU

Benchmark de los arreglos de la revisión final de la pull request #1503
(16 de noviembre de 2026). Compara tres versiones con `bench_review.py`:

- `0.28.x`: `origin/0.28.x` en 5bc9bcb, sin los cambios de la pull request.
- `fd48114`: la rama antes de los arreglos de la revisión.
- `head`: la rama con los arreglos (b7cd287): el forecaster serializado viaja
  como array de numpy, que joblib comparte como memmap; no se serializa si
  joblib usa un proceso; se deserializa antes de registrar los avisos; y
  `backtesting_stats` también serializa una vez.

Contenedor Linux de 4 vCPU, Python 3.12.3, numpy 2.5.5, pandas 3.1.1,
scikit-learn 1.9.1, joblib 1.6.0, LightGBM 4.7.0, XGBoost 3.4.1. Una
repetición. Cada caso corre en un proceso nuevo y llama dos veces a la
operación: "first" arranca los workers, "next" los reutiliza. `shm_MB` es el
pico de uso de `/dev/shm` (memmaps de joblib).

```bash
python bench_review.py run   # con los worktrees wt-028 y wt-fd junto al script
```

Casos (sin refit y `n_jobs=3`, salvo los dos últimos):

- `knn40`, `knn480`: `KNeighborsRegressor`, 20.000 filas y 240 lags (modelo
  de 38 MB), 40 y 480 folds de 5 pasos.
- `rf40`, `rf480`: `RandomForestRegressor` de 100 árboles, 8.000 filas y 20
  lags (73 MB), 40 y 480 folds de 5 pasos. `rf480` no se mide en 0.28.x:
  según la revisión final, llena `/dev/shm`.
- `lgbm2000`: `LGBMRegressor` de 500 árboles, 2000 folds de un paso.
- `xgb_grid_refit`: búsqueda en rejilla de `XGBRegressor` (3 candidatos, 6
  folds con refit), `n_jobs='auto'`.
- `hgb_norefit100`: `HistGradientBoostingRegressor`, 100 folds de 36 pasos,
  `n_jobs='auto'`.

Resultados:

- Predicciones idénticas en las tres versiones en los 7 casos (`hash`).
- Los arreglos no empeoran ningún caso respecto a `fd48114`.
- Frente a 0.28.x, `head` gana en todos los casos salvo KNN: RandomForest con
  40 folds, de 22,2 a 14,0 s en frío; LightGBM con 2000 folds, de 116,6 a
  10,8; HGB con `'auto'`, de 6,7 a 4,7.
- KNN sigue de 2 a 3 veces más lento que 0.28.x en caliente (480 folds: 8,6
  frente a 3,2 s). 0.28.x enviaba sus arrays como memmaps de solo lectura que
  los workers usaban sin copiar; ahora cada fold copia el modelo al
  deserializarlo. Solo afecta a modelos que guardan arrays grandes, con
  `n_jobs` entero y sin refit (con `'auto'` van en un proceso). Queda en
  "Siguientes pasos" de `REPORT.md`.

```
scenario        version   first    next  shm_MB  hash
knn40           0.28.x     2.15    0.43    38.0  307845
knn40           fd48114    5.58    4.50     0.0  307845
knn40           head       3.78    1.21    38.1  307845
knn480          0.28.x     5.07    3.16    38.0  987871
knn480          fd48114   43.73   21.68     0.0  987871
knn480          head      10.62    8.57    38.1  987871
rf40            0.28.x    22.22   19.68     0.0  125568
rf40            fd48114   19.02   17.42     0.0  125568
rf40            head      14.04   11.50    72.7  125568
rf480           fd48114   88.91   85.18     0.0  823911
rf480           head      29.45   28.21    72.7  823911
lgbm2000        0.28.x   116.57   54.02     0.0  812808
lgbm2000        fd48114   11.44    9.06     0.0  812808
lgbm2000        head      10.76    8.95     1.4  812808
xgb_grid_refit  0.28.x    11.07    9.15     0.0  303585
xgb_grid_refit  fd48114   10.51    7.90     0.0  303585
xgb_grid_refit  head       9.82    8.26     0.0  303585
hgb_norefit100  0.28.x     6.67    7.17     0.0  845000
hgb_norefit100  fd48114    5.23    3.42     0.0  845000
hgb_norefit100  head       4.74    2.83     0.0  845000
```
