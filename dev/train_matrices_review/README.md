# Revisión de la creación de las matrices de entrenamiento (2026-11-28)

Scripts de los hallazgos de la revisión de cómo crea cada forecaster sus matrices de
entrenamiento, hecha al escribir la página de la FAQ
`docs/faq/how-forecasters-create-training-matrices.md`. Las filas del roadmap que los
usan son BUG-rnn-short-exog, PERF-ms-window-features-batch y PERF-ms-binner-batch, y
la línea del backlog sobre las máscaras de NaN de cada paso de los forecasters
directos. MAINT-train-exog-helper salió de la misma revisión, pero es lectura de código
y no tiene script.

**Base medida:** `0.28.x` en `f85edb77f`, cargada desde el repositorio con `PYTHONPATH`.
**Entorno:** Windows 11, Python 3.13, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1,
LightGBM del entorno `skforecast_28_pd3`; `rnn_short_exog.py`, con keras 3.15.1 y torch.
Los tiempos son de una máquina de escritorio sin aislar: léanse como órdenes de magnitud.

Ejecución, desde una carpeta fuera del repositorio:

```bash
PYTHONPATH=<ruta del repositorio> python -X utf8 dev/train_matrices_review/<script>.py
```

## Scripts

| Script | Fila | Qué mide o reproduce |
|---|---|---|
| `rnn_short_exog.py` | BUG-rnn-short-exog | Con `exog.iloc[window_size:]`, `exog_train` sale con 23 filas frente a 26 de `X_train`, y `fit` falla en keras |
| `window_features_share.py` | PERF-ms-window-features-batch y backlog (máscaras de NaN de Direct) | Tiempo de `_create_window_features` dentro del `create_train_X_y` multiserie (5000 series de 300 días y 500 de 1000), y de `_filter_nan_X_y_step` dentro del `fit` de `ForecasterDirect` |
| `ms_fit_breakdown.py` | PERF-ms-window-features-batch | `fit` multiserie con LightGBM, con y sin window features, frente al estimador solo, y el prototipo de las 6 window features calculadas para todas las series a la vez con un `rolling` sobre el DataFrame ancho |
| `ms_fit_profile.py` | PERF-ms-binner-batch | cProfile del `fit` multiserie sin window features: reparto entre LightGBM, matrices, predicción in-sample y binners por serie |
| `binner_edges_batch.py` | PERF-ms-binner-batch | Bucle actual de `QuantileBinner.fit` por serie frente a un `np.percentile` 2D por grupo de longitud; comprueba que los bordes son idénticos |
| `binner_batch_prototype.py` | PERF-ms-binner-batch | Prototipo completo (bordes, bordes repetidos, bins vacíos e `intervals_`) en cuatro escenarios; comprueba que `bin_edges_`, `n_bins_`, `intervals_` y `transform` son idénticos |
| `binner_warnings_fit.py` | PERF-ms-binner-batch | `fit` real con 2000 series intermitentes: peso del binning y número de avisos de bins reducidos |

## Resultados del 2026-11-28

**Window features en multiserie** (`create_train_X_y`, Ridge, lags 1 a 14, 21 y 28,
`RollingFeatures` con los 6 estadísticos de py61):

| Caso | Total | `_create_window_features` |
|---|---|---|
| 5000 series de 300 días | 4,04 s | 3,20 s (79 %) |
| 500 series de 1000 días | 0,46 s | 0,35 s (76 %) |

**`fit` multiserie con LightGBM** (5000 series de 300 días, 200 árboles):

| Fase | Sin window features | Con window features |
|---|---|---|
| `fit` total | 3,8 s | 7,5 s |
| Matrices (`create_train_X_y`) | 0,45 s | 4,2 s |
| LightGBM solo | 1,7 s | 1,7 s |
| `predict` de 28 pasos | 0,28 s | 0,24 s |

El prototipo con un `rolling` sobre el DataFrame ancho calcula las 6 window features de
las 5000 series en 0,84 s, frente a 3,2 s serie a serie. No admite series de distinta
longitud ni `fillna` por serie: es una cota de la ganancia, no la implementación.

**Binners por serie** (5000 series; bucle actual frente al prototipo; en los cuatro
casos, `bin_edges_`, `n_bins_`, `intervals_` y `transform` idénticos):

| Caso | Bucle | Prototipo | Ganancia |
|---|---|---|---|
| Predicciones continuas, 272 filas | 0,444 s | 0,090 s | x4,9 |
| Mitad de las series constantes | 0,399 s | 0,090 s | x4,4 |
| Valores repetidos (todas pierden bins) | 0,399 s | 0,163 s | x2,4 |
| Longitudes de 100 a 1000 (casi todas distintas) | 0,473 s | 0,306 s | x1,5 |

En el bucle, `np.percentile` es la mitad del tiempo (0,21 s de 0,44 s con 272 filas);
crear los objetos cuesta 0,003 s. En un `fit` real con 2000 series intermitentes y
LightGBM de 100 árboles, el binning es 0,31 s de 1,49 s (21 %) y emite 1210 avisos de
bins reducidos, uno por serie, con 12 textos distintos.

**Máscaras de NaN de Direct** (`ForecasterDirect`, 20 000 filas, 24 pasos, 24 lags, 10
exógenas, Ridge, `n_jobs=1`): con `dropna_from_series=True`, `_filter_nan_X_y_step` es
0,010 s de un `fit` de 0,21 s (5 %).

## Notas para quien implemente PERF-ms-binner-batch

- El prototipo agrupa las series por número de filas para que `np.percentile(X_2d, q,
  axis=1)` sea la misma función de numpy que hoy: los bordes salen idénticos bit a bit
  con cualquier `method`. Con longitudes casi todas distintas la ganancia baja a x1,5;
  la alternativa (relleno con NaN, orden en bloque e interpolación propia) exige
  comprobar la identidad bit a bit frente a la interpolación de numpy.
- Las series sin bordes repetidos comprueban los bins vacíos en bloque, contando los
  valores por debajo de cada borde interno (equivale al `searchsorted(..., side='right')`
  de `QuantileBinner.fit`). Las demás siguen las reglas de `QuantileBinner.fit` sobre los
  11 bordes ya calculados.
- El prototipo crea los objetos con `QuantileBinner.__new__`; en la versión final, un
  constructor privado a partir de los bordes. Las series con más filas que `subsample`
  siguen por el camino actual.
- El aviso de bins reducidos pasa a ser uno solo, con el número de series afectadas.
