# BUG-mase-window-size: el denominador de MASE y RMSSE depende del modelo

Fila del roadmap: ninguna todavía. Ya visto en FE-05 (punto 2 de "Medido con el prototipo",
`roadmap.md`), en `work/lens_outputs-metrics.md` ("Fricciones observadas") y relacionado con
PERF-04 (que hoy propone **mantener** la exclusión).

Diagnóstico (2026-11-09) en local, desde `docs/backtesting-y-true` (base `0.27.x`).
Implementación en la rama `fix/mase-scale-per-origin` desde `0.27.x` (sección 8), sin PR.
Scripts: `repro/BUG-mase-window-size_01_h2o.py`, `_02_search.py`, `_02b_seeds.py`,
`_03_paths.py`, `_03b_osa_multiseries.py` y `_04_per_origin.py` (los nombres `mase_*.py`
de las secciones 3 y 4 son los mismos scripts). Las variantes del Ridge de 3.1
(`mase_01b_ridge_variants.py`) no se guardaron.

Python 3.14.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, scipy 1.18.1, statsmodels
0.14.6.

## 0. Veredicto

**Bug.** La exclusión fue deliberada, pero se apoya en una premisa equivocada.

- **Definición de referencia.** La escala de MASE es el MAE del naive sobre los datos de
  entrenamiento de la serie: es una propiedad de la serie, no del modelo. skforecast la hace
  depender del `window_size` de cada forecaster.
- **El MASE deja de comparar modelos.** Con dos modelos sobre la misma serie y el mismo
  `cv`, el cociente de sus MASE no es el de sus MAE. En una búsqueda, la métrica escalada
  puede elegir otro candidato que la no escalada (medido: sí ocurre, ver 3.2).
- **Ni siquiera es coherente dentro de skforecast:**
  - `backtesting_stats` y `backtesting_foundation` no excluyen nada.
  - El multiserie excluye filas, no observaciones: la misma serie da otro MASE según venga
    en un DataFrame ancho o en un dict (0.7537 frente a 0.7153).
  - `ForecasterDirect` con `OneStepAheadFold` pierde además las últimas `steps - 1`
    observaciones.

El argumento original fue: "estas no participan realmente en el entrenamiento" (nota de un
notebook de desarrollo, 2024-09-09, ver 2). Confunde las filas de la matriz de
entrenamiento con los datos in-sample.

Hay además un **segundo mecanismo**, fuera de la hipótesis y más grande en h2o: con
`refit=True`, `y_train` es la unión de los entrenamientos de todos los folds. La escala
incluye entonces el test de los folds anteriores. `backtesting_stats` (`Arima`, `Ets`) y
`backtesting_foundation` fuerzan `refit=True`, así que **ETS y Chronos-2 se escalan siempre
con más datos que un ML con `refit=False`**. Es un comportamiento intencionado pero
discutible (decisión 2, en 5.1).

**La hipótesis se confirma con un matiz.** El 0.664 del ETS en h2o no viene de `window_size`
(`backtesting_stats` no excluye nada). Viene del `refit=True` forzado: `y_train` es
`y.iloc[:192]`, no `y.iloc[:168]`. Por eso, quitar solo la exclusión de `window_size` **no
deshace la inversión entre Ridge y ETS** de la observación previa (ver 3.1).

## 1. Definición de referencia

- **Hyndman y Koehler (2006)**, "Another look at measures of forecast accuracy", IJF 22(4),
  679-688. El error escalado es `q_j = e_j / ((1/(T-1)) * sum_{t=2..T} |y_t - y_{t-1}|)`
  sobre los datos de entrenamiento: "scaling the errors based on the *training* MAE from a
  simple forecast method". Cita y fórmula tomadas de fpp3, sección 5.8
  (https://otexts.com/fpp3/accuracy.html), que reproduce el artículo. El PDF del artículo no
  se pudo extraer en esta sesión (ver 7). Para series estacionales, fpp3 usa el naive
  estacional (`t = m+1..T`); skforecast usa siempre el lag 1, lo que es legítimo y no es lo
  que se analiza aquí.
- **sktime** (`sktime/forecasting/model_evaluation/_functions.py`, `_evaluate_window`):
  `score = metric(y_test, y_pred, y_train=y_train)`. Ahí, `y_train` es el entrenamiento del
  fold tal cual sale de `cv.split_series(y)`, el mismo que recibe `forecaster.fit`, sin
  recorte. La métrica se calcula por fold.
- **utilsforecast (Nixtla)** (`utilsforecast/losses.py`, `mase(df, models, seasonality,
  train_df, ...)`): la escala es la media de `|y_t - y_{t-seasonality}|` sobre todo
  `train_df`. Con `cutoff_col`, usa las filas con `time_col <= cutoff` de cada serie. No
  quita filas iniciales, salvo las `seasonality` primeras, que no tienen retardo.

Ninguna de las tres hace depender la escala del modelo evaluado.

## 2. Historia

- **2024-06-24/07-10** (`6bff16540`, `b7b7985e1`): módulo `metrics` con MASE y RMSSE
  (0.13.0). `y_train` es lo que pase el usuario.
- **2024-09-09** (`1661368e1`, Joaquín Amat, "ongoing for ForecasterAutoregMultiVariate"):
  `_calculate_metrics_multiseries` recibe `window_size` "to remove the first `window_size`
  values from y_train since they are not part of the training matrix". En
  `dev/params_search_one_step_ahead.ipynb`, la nota: "En _calculate_metrics_multiseries se
  está metiendo como y_train sin eliminar las primeras window_size filas. Estas no
  participan realmente en el entrenamiento."
- **2024-09-13** (`820fecc5f`, "ongoing unit test"): lo mismo en el backtesting de una serie
  (`train_iloc_start = fold[0][0] + window_size`). Es el desarrollo de `OneStepAheadFold`
  (0.14.0).
- **2025-03-05** (`61394cb02`): refactor del multiserie, comentario "Exclude first
  window_size observations used to create predictors".
- **2026-03-13/14** (`801c48210`, `2034fc2ac`): se mueve el comentario y nace
  `_any_metric_needs_y_train` (sin cambio de criterio).

**Motivo:** que `backtesting(steps=1)` y `OneStepAheadFold` den las mismas métricas. En el
OSA, `y_train` es el objetivo de la matriz de entrenamiento, que no tiene las primeras
`window_size` filas. Lo fijan
`test_grid_search_forecaster_equivalence_backtesting_one_step_ahead` y
`test_evaluate_grid_hyperparameters_equivalent_outputs_backtesting_one_step_ahead`
(`assert_frame_equal` con MASE y RMSSE). Se alineó el backtesting con el OSA, cuando lo
correcto era alinear el OSA con la definición.

**No hay nota de versión** en `docs/releases/releases.md` que mencione la exclusión, ni la
documentan los docstrings públicos. Solo el comentario interno de `metrics.py:203-205` y
`:288-290` ("When using this metric in validation, `y_train` doesn't include..."). Issues y
PRs de GitHub no revisados (ver 7).

## 3. Experimentos

### 3.1 h2o: observación previa y denominador aislado (`mase_01_h2o.py`)

`load_demo_dataset()`, `n_train = 168`, `TimeSeriesFold(steps=12, initial_train_size=168,
refit=False)`. Hay 3 folds, cuyos entrenamientos terminan en 168, 180 y 192.

| Modelo | `window_size` | `refit` | MAE | MASE backtesting | MASE con `y.iloc[:168]` | RMSSE backtesting | RMSSE con `y.iloc[:168]` |
|---|---|---|---|---|---|---|---|
| Seasonal naive (`offset=12`) | 12 | False | 0.0668 | **0.7121** | **0.7274** | 0.5749 | 0.5856 |
| Ridge `lags=24` + mes (numérico) | 24 | False | 0.0658 | 0.6871 | 0.7171 | 0.5358 | 0.5535 |
| ETS `m=12` | 1 | True (forzado) | 0.0649 | **0.6640** | **0.7069** | 0.4884 | 0.5198 |

- **Seasonal naive y ETS:** se reproducen exactamente (0.712 frente a 0.727 y 0.664 frente
  a 0.707).
- **Ridge "24 lags y el mes":** no se reproduce exactamente (0.667 frente a 0.696). Se
  probaron 5 variantes (mes numérico o one-hot, con y sin `StandardScaler`), con MASE entre
  0.615 y 0.687. El cociente 0.696 / 0.667 = 1.0435 sí coincide con el de las escalas con
  `window_size` 24 y sin él (0.09577 / 0.09177 = 1.0436). Con esos números, el Ridge del
  enunciado (MAE ≈ 0.0639) gana al ETS (MAE 0.0649) en MAE y pierde en el MASE del
  backtesting: la inversión es real.

**¿De dónde sale el 0.664 del ETS?** `backtesting_stats` no excluye `window_size`, pero
fuerza `refit=True` con `Ets`, y `y_train` es la unión de los tres entrenamientos:

| `y_train` | escala (MAE naive) | MASE ETS |
|---|---|---|
| `y.iloc[:168]` | 0.09177 | 0.7069 |
| `y.iloc[:180]` | 0.09418 | 0.6889 |
| `y.iloc[:192]` (lo que usa) | 0.09771 | **0.6640** |

La escala del ETS sube un 6.5 % por `refit`; la del Ridge, un 4.4 % por `window_size`. Si
solo se quita la exclusión de `window_size`, el Ridge del enunciado pasaría a 0.696 y el ETS
seguiría en 0.664: la inversión se mantiene.

**Mismo `y`, mismas predicciones, solo cambia el inicio de `y_train`** (`y.iloc[ws:168]`):

| `ws` | escala MAE naive | escala RMSE naive | MASE naive | MASE Ridge | MASE ETS | Ridge / ETS |
|---|---|---|---|---|---|---|
| 0 | 0.09177 | 0.14639 | 0.7274 | 0.7171 | 0.7069 | 1.0143 |
| 1 | 0.09215 | 0.14681 | 0.7244 | 0.7141 | 0.7040 | 1.0143 |
| 12 | 0.09375 | 0.14910 | 0.7121 | 0.7019 | 0.6920 | 1.0143 |
| 24 | 0.09577 | 0.15125 | 0.6970 | 0.6871 | 0.6774 | 1.0143 |
| 48 | 0.09859 | 0.15553 | 0.6771 | 0.6675 | 0.6580 | 1.0143 |

Con una escala común, el cociente de los MASE es el de los MAE (1.0143). skforecast
compara Ridge (escala de `ws=24`) con ETS (escala `[0:192]`) y da 1.0349. Entre `ws=0` y
`ws=48`, la escala cambia un 7.4 % en h2o, cuya variabilidad crece con el tiempo.

### 3.2 Búsquedas sobre `lags` (`mase_02_search.py`, `mase_02b_seeds.py`)

`grid_search_forecaster`, `Ridge(alpha=1)` y `lags_grid = [1, 2, 3, 6, 12, 13, 24, 36,
48]`. Cada búsqueda calcula a la vez MASE, MAE, RMSSE y MSE, así que los rankings salen de
las mismas predicciones. Un MASE con escala fija es MAE / constante y ordena igual que el
MAE: cualquier diferencia de orden la causa el denominador.

**h2o** (`initial_train_size = 168`; `TimeSeriesFold(steps=12)`):

| `cv` | Orden por MASE | Orden por MAE | Mejor MASE / MAE | Orden por RMSSE | Orden por MSE | Kendall MASE~MAE |
|---|---|---|---|---|---|---|
| `TimeSeriesFold` | 24, **48, 36**, 13, 12, 6, 3, 2, 1 | 24, **36, 48**, 13, 12, 6, 3, 2, 1 | 24 / 24 | 24, 48, 36, 13, 12, ... | 24, 36, 48, 12, 13, ... | 0.944 |
| `OneStepAheadFold` | **48**, 24, 36, 13, 12, 6, 3, 2, 1 | **24**, 36, 48, 12, 13, 6, 2, 3, 1 | **48 / 24** | 48, 36, 24, ... | 48, 36, 24, ... | 0.778 |

**Con `OneStepAheadFold`, el mejor candidato cambia.** MASE elige `lags=48` (MAE 0.06569)
frente a `lags=24` (MAE 0.06537), porque la escala de `lags=48` es 0.09859 y la de
`lags=24`, 0.09577. Con `TimeSeriesFold`, el mejor coincide, pero 36 y 48 se cruzan.

Detalle de h2o con `OneStepAheadFold` (escala implícita = MAE / MASE):

| `lags` | MAE | MASE skforecast | MASE con `y.iloc[:168]` | escala implícita |
|---|---|---|---|---|
| 12 | 0.07219 | 0.77001 | 0.78658 | 0.09375 |
| 24 | 0.06537 | 0.68255 | **0.71227** | 0.09577 |
| 36 | 0.06538 | 0.68383 | 0.71237 | 0.09560 |
| 48 | 0.06569 | **0.66628** | 0.71576 | 0.09859 |

**Series sintéticas, 30 semillas.** Seno estacional (periodo 12) más AR(1) con `phi=0.7`,
`n = 400`, `initial_train_size = 300` y burn-in de 200 (estacionaria desde `t = 0`). La
desviación del ruido es constante, crece linealmente hasta x3 o decrece hasta x1/3 a lo
largo de la serie.

| Serie | `cv` | Mejor MASE ≠ mejor MAE | Pérdida de MAE del elegido por MASE (mediana / máx.) | Mejor RMSSE ≠ mejor MSE | Kendall medio MASE~MAE | `lags` medio del mejor: MASE / MAE | escala(`ws=48`) / escala(`ws=0`), media [mín, máx] |
|---|---|---|---|---|---|---|---|
| varianza constante | `TimeSeriesFold` | 6 / 30 | 0.79 % / 1.71 % | 4 / 30 | 0.974 | 33.6 / 34.8 | 1.002 [0.949, 1.031] |
| varianza constante | `OneStepAheadFold` | 7 / 30 | 0.32 % / 2.26 % | 5 / 30 | 0.943 | 21.9 / 21.5 | 1.002 [0.949, 1.031] |
| varianza crece x3 | `TimeSeriesFold` | 6 / 30 | 0.43 % / 1.86 % | 8 / 30 | 0.870 | **30.6 / 27.1** | 1.048 [1.004, 1.074] |
| varianza crece x3 | `OneStepAheadFold` | 8 / 30 | 0.23 % / **4.62 %** | 9 / 30 | 0.741 | **16.4 / 12.2** | 1.048 [1.004, 1.074] |
| varianza decrece x1/3 | `TimeSeriesFold` | 2 / 30 | 1.67 % / 3.31 % | 1 / 30 | 0.981 | 33.3 / 33.6 | 0.986 [0.935, 1.014] |
| varianza decrece x1/3 | `OneStepAheadFold` | 8 / 30 | 0.26 % / 1.67 % | 5 / 30 | 0.950 | 25.3 / 27.0 | 0.986 [0.935, 1.014] |

Lectura:

- **El cambio del mejor candidato ocurre en un 7-27 % de las semillas**, también con
  varianza constante. Ahí la escala de `ws=48` se mueve un ±5 % por azar.
- **Casi siempre entre candidatos casi empatados:** el elegido por MASE pierde una mediana
  de 0.2-1.7 % de MAE, con un máximo de 4.6 %.
- **El sesgo es sistemático cuando la variabilidad tiene tendencia.** Si crece, el MASE
  favorece ventanas más largas (`lags` medio del mejor: 16.4 frente a 12.2 con OSA); si
  decrece, ligeramente las más cortas.
- **`OneStepAheadFold` es más sensible:** las diferencias de MAE entre candidatos son
  menores a un paso, así que un 3-5 % de escala las cruza más (Kendall 0.74-0.95 frente a
  0.87-0.98).

## 4. Caminos que pasan `y_train` a una métrica escalada

Comprobado con `mase_03_paths.py` y `mase_03b_osa_multiseries.py`. La escala implícita
(MAE / MASE) se busca entre todas las ventanas `y[a:b]` y se da la que la reproduce
exactamente (h2o, `n_train = 168`, folds de 12).

| Camino | Código | `y_train` usado (iloc) | ¿Depende de `window_size`? | Notas |
|---|---|---|---|---|
| `backtesting_forecaster`, `refit=False` | `_validation.py:716-728` | `[ws:168]` (Recursive `lags=24`: `[24:168]`; `lags=3`: `[3:168]`) | **Sí** | |
| ... `refit=True` | ídem | `[24:192]` | Sí | Unión de entrenamientos: incluye el test de los folds 1 y 2 |
| ... `refit=True, fixed_train_size=True` | ídem | `[24:192]` | Sí | La unión de ventanas móviles es contigua: igual que sin `fixed_train_size` |
| ... `refit=2` (intermitente) | ídem | `[24:192]` | Sí | |
| ... `gap=5` | ídem | `[24:168]` | Sí | `gap` no afecta |
| ... `differentiation=1` (`lags=24`) | ídem | `[25:168]` | Sí | `window_size` incluye el orden de diferenciación; `y_train` está en la escala original |
| `ForecasterDirect` `lags=24`, `steps=12` | ídem | `[24:168]` | Sí | |
| `ForecasterEquivalentDate` `offset=12` / `n_offsets=3` | ídem | `[12:168]` / `[36:168]` | **Sí** | El baseline se escala distinto que el modelo |
| `backtesting_stats`, `Sarimax` `refit=False` | `_validation.py:2433-2445` | `[0:168]` | No | |
| `backtesting_stats`, `Sarimax` `refit=True`; `Arima` / `Ets` (refit forzado) | ídem | `[0:192]` | No | Siempre la unión con `Arima` y `Ets` |
| `backtesting_foundation`, una serie | `_validation.py:3070-3078` | `[0:192]` (por lectura) | No | Fuerza `refit=True` (`:2825`), así que también usa la unión. No ejecutado (ver 7) |
| `backtesting_foundation`, varias series | `_validation.py:3045` → `_utils.py` | unión, sin exclusión (por lectura) | No | Pasa `window_size=0` (su `window_size` es el contexto, p. ej. 8192) |
| `backtesting_forecaster_multiseries`, por serie, dict | `_utils.py:2051-2083` | `[24:168]`; serie que empieza en 30: `[54:168]` | **Sí** | `cumcount() >= window_size` por serie |
| ... DataFrame ancho (o dict con NaN iniciales) | ídem | `[24:168]`; serie que empieza en 30: **`[30:168]`** | **Sí, pero mal** | `cumcount` cuenta filas, no observaciones: la exclusión cae sobre el relleno NaN. MASE de esa serie: 0.7537 en ancho frente a 0.7153 en dict |
| ... filas agregadas (`average`, `weighted_average`) | ídem | las de cada serie | Sí | Medias de los MASE por serie: heredan las dos rarezas anteriores |
| ... `pooling` | `_utils.py:2163-2185` | lista por serie, mismas ventanas; escala = media de las diferencias concatenadas | Sí | Ancho: escala 0.14621 = s1 y s2 sin 24 + s3 completa (comprobado) |
| `grid/random/bayesian_search_forecaster` con `TimeSeriesFold` | `_search.py` → `_backtesting_forecaster` | = backtesting | Sí | |
| ... con `OneStepAheadFold` | `_utils.py:1078-1172` (`_calculate_metrics_one_step_ahead`) | objetivo de la matriz: Recursive `[24:168]`, `differentiation=1`: `[25:168]` | **Sí** | Con diferenciación o `transformer_y`, `y_train` se reconstruye con `inverse_transform_training` |
| ... `ForecasterDirect` `steps=12` con `OneStepAheadFold` | ídem | **`[24:157]`** | Sí, y además `steps - 1` | La matriz del paso 1 pierde las últimas 11 filas. Los tests de equivalencia usan `steps=1` y no lo ven |
| `grid/random/bayesian_search_forecaster_multiseries` con `OneStepAheadFold` | `_utils.py:2214-2540` | objetivo de la matriz por serie: `[24:168]`; serie que empieza en 30: `[54:168]` (ancho y dict) | Sí | En ancho, **OSA ≠ backtesting**: `[54:168]` frente a `[30:168]` |
| ... con `TimeSeriesFold` | = backtesting multiserie | ancho `[30:168]` / dict `[54:168]` | Sí | |

**Resumen:** `window_size` interviene en todo lo que pasa por `_backtesting_forecaster`, el
backtesting multiserie y los dos OSA. No interviene en `backtesting_stats` ni en
`backtesting_foundation`. Con `refit=True` (o forzado), además, la escala usa datos del test
de los folds anteriores.

## 5. Arreglo mínimo propuesto (sin aplicar)

**Principio:** la escala depende solo de la serie y del momento en que se predice. Nunca de
la configuración del modelo: ni de `window_size`, ni de `refit`, ni de `fixed_train_size`.
La decisión 1 aplica la primera parte; la decisión 2 fija qué observaciones entran. La
propuesta es una escala por origen: cada error se escala con todas las observaciones
anteriores a su predicción (5.1).

La alternativa fija, `y.iloc[:initial_train_size]` para todos los puntos, también es
correcta. Es la regla que ya proponía FE-05 y la que usa `comparing-forecasting-models.ipynb`
en su `summarize` (`y_train = series[level].iloc[:n_train]`).

**Decisión 1 (`window_size`): quitar la exclusión.** Recomendado: es el bug.

1. `_backtesting_forecaster` (`_validation.py:716-728`): `train_iloc_start = fold[1][0]`,
   sin `+ window_size`; borrar el NOTE.
2. `_calculate_metrics_backtesting_multiseries` (`_utils.py:2073-2077`): quitar el filtro
   `cumcount() >= window_size` y el argumento `window_size` (llamadas en
   `_validation.py:1717` y `:3045`). Resuelve también la diferencia entre DataFrame ancho y
   dict.
3. `_calculate_metrics_one_step_ahead` (`_utils.py:1078`):
   - recibir aparte el `y_train` de las métricas, `y.iloc[:initial_train_size]` en la escala
     original, calculado una vez antes del bucle de candidatos (llamadas en
     `_search.py:573` y `:969`);
   - quitar la reconstrucción con `inverse_transform_training` y `inverse_transform` de
     `y_train`.

   Resuelve también el `[24:157]` de `ForecasterDirect`.
4. `_predict_and_calculate_metrics_one_step_ahead_multiseries` (`_utils.py:2214`): igual,
   `y_train` de cada serie desde `series` hasta `initial_train_size` (llamadas en
   `_search.py:1735` y `:2267`). Se elimina la reconstrucción por nivel (reindexado y
   `inverse_transform_training`).
5. `metrics.py:203-205` y `:288-290`: borrar el NOTE; en el docstring de `y_train`, "all the
   observations of the training set".

Los tests de equivalencia entre backtesting con `steps=1` y OSA siguen valiendo, porque
cambian los dos lados. Los puntos 3 y 4 usan la escala fija (opción B). Con la decisión 2
(opción A), la sustituye la escala por punto de 5.1.

**Decisión 2 (`refit`): sustituir la unión de entrenamientos por una escala por origen**, en
las cuatro funciones y en los dos OSA. Detalle en 5.1. Sin ella, ETS y Chronos-2 (refit
forzado) siguen escalándose con `[0:192]` frente a `[0:168]` del ML con `refit=False`, y la
inversión de h2o se mantiene (3.1).

### 5.1 Decisión 2: escala por origen

**Por qué la unión de hoy es incorrecta:**

- El mismo modelo se escala distinto según se reentrene o no, y `refit` es una decisión
  sobre el modelo, no sobre la serie.
- Los errores del primer fold se escalan con datos posteriores a su predicción (el test de
  los folds siguientes).

**Opciones correctas:**

| Opción | Escala de un punto del fold `k` | A favor | En contra |
|---|---|---|---|
| **A. Por origen (recomendada)** | Todas las observaciones anteriores al origen del fold `k` | Es literalmente la definición: cada error se escala con lo conocido al predecir. Es lo que hace utilsforecast con `cutoff_col` (`time_col <= cutoff`, ver 1) | El MASE ya no es MAE / constante, sino un MAE ponderado (con los mismos pesos para todos los modelos, así que el ranking sigue siendo justo) |
| B. Fija | `y.iloc[:initial_train_size]` para todos los puntos | Más simple; conserva la relación exacta con el MAE; coincide con el cálculo a mano de la guía | En los últimos folds no aprovecha datos ya conocidos al predecir (estimación algo más pobre de la escala, no un error) |
| C. Hoy | Unión de entrenamientos (y sin `window_size`) | Ninguna | Depende de `refit`; usa datos posteriores a la predicción |

**Definición (opción A).** Cada punto evaluado `t` del fold `k` tiene su error escalado
`q_t = |e_t| / s_k`, donde `s_k` es el MAE del naive (lag 1) sobre las observaciones
anteriores al origen del fold `k`. El MASE es la media de los `q_t` sobre todos los puntos
evaluados:

`MASE = Σ_t q_t / n = Σ_k n_k · MASE_k / Σ_k n_k`

Es decir, la media de los MASE de cada fold ponderada por sus `n_k` puntos evaluados. No hay
que ponderar a mano: sale de promediar punto a punto. Es la definición de Hyndman y Koehler
(media de los `|q_t|`) y encaja con la métrica única sobre todas las predicciones y con
`n_evaluated`.

No vale la media simple de los MASE de cada fold (lo que da promediar los resultados de
sktime fold a fold): un fold con 1 punto válido de 12 (NaN en el test) o el último fold
incompleto (`allow_incomplete_fold`) pesarían lo mismo que un fold completo.

**RMSSE** sigue la misma lógica con cuadrados, con la media punto a punto antes de la raíz:
`RMSSE = sqrt(Σ_t e_t² / s2_k / n) = sqrt(Σ_k n_k · RMSSE_k² / Σ_k n_k)`, donde `s2_k` es el
MSE del naive. No es la media de los RMSSE de cada fold.

**Casos:**

- **Origen del fold:** el final de la última ventana, `fold[2][1]`, no el final del
  entrenamiento. Comprobado con `TimeSeriesFold(steps=5, initial_train_size=20, gap=2,
  refit=False)`: los folds son `[0, [0, 20], [17, 20], [20, 27], [22, 27], True]`, `[1, [0,
  20], [22, 25], [25, 32], [27, 32], False]`..., así que los orígenes son 20, 25, 30 y 35,
  aunque el entrenamiento acabe siempre en 20. En consecuencia:
  - `refit` y `fixed_train_size` no influyen.
  - Las fechas del `gap` no entran en la escala (no se conocen al predecir).
- **NaN en `y_true` del test:** el punto no tiene error y no entra (`n_k` baja). Es lo que
  ya hace `_calculate_metrics_backtesting` desde #1458.
- **NaN o huecos antes del origen:** la escala usa solo diferencias entre dos observaciones
  consecutivas (`nanmean` de `diff`), como hoy ("`y_train` includes the intercepted NaNs").
  No se tiende un puente sobre el hueco: la diferencia entre un dato y otro 5 pasos después
  sería un error a 5 pasos e inflaría la escala de un naive definido a 1 paso.
- **Folds solapados (`fold_stride < steps`):** de cada fecha se queda la predicción del
  último fold, como ahora. Cada punto se escala con el origen de su fold (columna `fold` de
  las predicciones), no con el de la fecha.
- **`OneStepAheadFold`:** cada predicción usa los valores reales hasta `t - 1`, así que su
  origen es el propio punto: escala con `y[:t]`. Es lo que daría el backtesting con
  `steps=1` (un fold por punto), así que los tests de equivalencia entre backtesting y OSA
  se mantienen. El modelo se entrena una sola vez, pero eso es configuración del modelo y
  no cuenta.
- **Multiserie:** una escala por serie y origen, `s_{serie,k}`, y media punto a punto por
  serie. A nivel agregado, `weighted_average` (ponderada por `n_evaluated`) pasa a ser
  exactamente la media de los `q_t` de todos los puntos de todas las series.
- **Escala sin definir:** una serie sin dos observaciones consecutivas antes del origen
  (escala NaN) o constante (escala 0, MASE infinito). Pasa ya hoy. Lo coherente es dejar
  esos puntos fuera, como un `y_true` NaN, y avisar, en lugar de devolver NaN o infinito sin
  más.

**Decisión 3 (`pooling` en el multiserie).** Hoy es un MAE conjunto dividido por una escala
conjunta (las diferencias de todas las series concatenadas). Opciones:

1. Mantenerlo, con escalas por origen.
2. Alinearlo con la definición punto a punto. Entonces coincidiría con
   `weighted_average`, y `pooling` sobraría para las métricas escaladas.

Sin recomendación todavía: depende de qué se quiera que signifique `pooling`.

**Decisión 4 (métricas propias con `y_train`).** Una métrica del usuario recibe un solo
`y_train`, así que no puede escalarse por origen, y no se sabe cómo combinar sus valores por
fold (la ponderación de arriba solo vale para MASE y RMSSE). **Recomendación:** pasarle el
`y_train` fijo de la opción B, `y.iloc[:initial_train_size]` (sin datos posteriores a ninguna
predicción), y documentarlo. `mean_absolute_scaled_error` y `root_mean_squared_scaled_error`
se reconocen por la función, como ya hace `pooling` por nombre.

**Implementación.** La escala de cualquier origen sale en O(n) por serie, sin recortar
`y_train` en cada fold: `s(o) = cumsum(|diff|)[o] / cumsum(notna(diff))[o]`, con los NaN a 0
en la primera suma. Después, `q_t = |e_t| / s(origen_t)`, y la media sobre los puntos
evaluados. Lo mismo con `diff²` para RMSSE. Un helper común para las cuatro funciones y los
dos OSA. También sirve a PERF-04: no hay que materializar `y_train` en formato largo, basta
la suma acumulada de cada serie.

**h2o** (`mase_04_per_origin.py`, mismas predicciones, `refit=False` salvo ETS; Ridge
`lags=24` sin exógenas, MAE 0.0650):

| Modelo | MAE | MASE hoy | B: fija `y[:168]` | A: por origen (`y[:168]`, `y[:180]`, `y[:192]`) |
|---|---|---|---|---|
| Seasonal naive | 0.0668 | 0.7121 | 0.7274 | 0.7042 |
| Ridge `lags=24` | 0.0650 | 0.6784 | 0.7080 | 0.6898 |
| ETS | 0.0649 | 0.6640 | 0.7069 | 0.6843 |

Con A y con B, el orden es el del MAE, y la diferencia entre las dos es pequeña y la misma
para todos los modelos. Hoy, la escala del ETS es la de `[0:192]` y la del Ridge la de
`[24:168]`.

### 5.2 Impacto, tests y documentación

**Qué cambia para el usuario:**

- Los valores de MASE y RMSSE de `backtesting_forecaster` y
  `backtesting_forecaster_multiseries` (por serie y agregados) y de las búsquedas con
  `TimeSeriesFold` y `OneStepAheadFold`. Con la decisión 2, también los de
  `backtesting_stats` (con `refit=False` también: con varios folds, la escala por origen
  difiere de `y.iloc[:initial_train_size]`) y los de `backtesting_foundation`.
- En h2o (tabla de 5.1): el Ridge pasa de 0.678 a 0.690 (A) o 0.708 (B); el seasonal
  naive, de 0.712 a 0.704 o 0.727; el ETS, de 0.664 a 0.684 o 0.707.
- La dirección del cambio depende de la serie (con A, la escala de los últimos folds crece
  si la variabilidad crece).
- **El mejor candidato de una búsqueda con métrica escalada puede cambiar** (y con él el
  forecaster de `return_best=True`). A cambio, el ranking es el del MAE (con B) o el de un
  MAE ponderado igual para todos los candidatos (con A); con RMSSE, lo mismo con el RMSE.
- Con B, el MASE de skforecast coincide con el calculado a mano con `y_train =
  y.iloc[:initial_train_size]`. Con A, coincide con utilsforecast con `cutoff_col` (lag 1).

**Tests a tocar** (los que usan MASE o RMSSE en los caminos afectados; no se ha medido
cuántos fallan, ver 7):

- `tests_validation/test_backtesting_forecaster.py` (1),
  `test_backtesting_forecaster_refit.py` (1) y `test_backtesting_forecaster_multiseries.py`
  (10).
- `test_backtesting_stats.py` (1, solo con la decisión 2).
- `tests_utils/test_calculate_metrics_backtesting_multiseries.py` (6). Entre ellos,
  `..._with_window_size_variation`, que fija la exclusión y habría que reescribir para
  comprobar que el resultado no depende de `window_size`.
- `test_calculate_metrics_one_step_ahead.py` (2) y
  `test_predict_and_calculate_metrics_one_step_ahead_multiseries.py` (6).
- `tests_search/`: `test_grid_search_forecaster_multiseries.py` (4),
  `test_bayesian_search_forecaster_multiseries.py` (6),
  `test_evaluate_grid_hyperparameters_multiseries.py` (3), y los de equivalencia de
  `test_grid_search_forecaster.py`, `test_evaluate_grid_hyperparameters.py` y
  `test_bayesian_search_forecaster.py` (deberían seguir pasando).
- `test_calculate_metrics_backtesting.py` recibe `y_train` ya hecho: no cambia.

Tests nuevos:

1. Con las mismas predicciones, el MASE del backtesting no depende del forecaster: igual
   para Recursive con `lags` 3 y 24, Direct y `ForecasterEquivalentDate`, y calculado a
   mano (`Σ_t |e_t| / s(origen_t) / n`).
2. Una serie que empieza tarde da el mismo MASE en ancho y en dict.
3. OSA con `ForecasterDirect(steps=12)` usa todas las observaciones (no `[24:157]`).
4. Misma escala con `refit=True` y `False`, y con `fixed_train_size=True`.
5. Ponderación: un fold con NaN en el test y el último fold incompleto pesan por sus
   puntos evaluados (MASE igual a la media ponderada de los MASE por fold, no a la simple).
6. Con `gap`, el origen es `fold[2][1]`: las fechas del `gap` no entran en la escala.
7. Folds solapados: cada punto se escala con el origen de su fold.
8. RMSSE: raíz de la media ponderada de los cuadrados, no media de los RMSSE por fold.
9. Escala sin definir (serie constante o sin dos observaciones consecutivas antes del
   origen): esos puntos quedan fuera, con aviso.
10. Equivalencia entre backtesting con `steps=1` y OSA con la escala por punto (ya cubierta
    por los tests actuales, que deberían seguir pasando).

**Nota de versión:** sí, en **Changed** de 0.27.0 (texto para la opción A). "MASE and RMSSE
in backtesting and hyperparameter search scale each error with all the observations
before its forecast origin, the same for every forecaster and independent of `refit`.
Before, the first `window_size` observations of the forecaster were left out and, with
`refit=True`, the training sets of later folds were included, so the values of two models
were not comparable. Values change, and so may the best candidate of a search with a scaled
metric." Más una línea en **Fixed** para la diferencia entre DataFrame ancho y dict.

**Docs de IA:**

- `skills/metric-selection/SKILL.md` (bloque "y_train is handled automatically") y
  `llms-base.txt`: una frase con la regla. La línea 80 de `llms-base.txt` ("passing the same
  `TimeSeriesFold` to all of them makes the results comparable") solo es cierta para MASE
  con el arreglo (y la decisión 2).
- Regenerar con `ai-context-sync`.
- La `summarize` de `comparing-forecasting-models.ipynb` podría usar entonces la métrica del
  backtesting (opcional).

**Coordinación:** PERF-04 dice "quitar las primeras `window_size` observaciones de cada
una"; con este arreglo, su `y_train` por serie se simplifica a un corte por posición. Hay
que hacerlo antes que PERF-04 o en el mismo PR.

## 6. Fuera de alcance

- MASE con `initial_train_size=None` (forecaster ya entrenado): devuelve NaN
  (`lens_outputs-metrics.md`). Con la opción B, `y_train` estaría vacío. Con la A, la escala
  saldría de lo que `y` tenga antes de cada origen (al menos la última ventana), así que
  probablemente quede definida. Sin comprobar.
- skforecast escala con el naive de lag 1, no con el estacional (fpp3, utilsforecast con
  `seasonality`). Es otra definición válida; no se toca.

## 7. Lo que no se ha comprobado

- **`backtesting_foundation`:** ni con una serie ni con varias se ha ejecutado (requiere
  cargar el modelo). La fila de la tabla 4 sale de leer el código: `refit=True` forzado en
  `:2825`, la unión de `fold[1]` en `:3070-3078` y `window_size=0` en `:3045`.
- **Multiserie con `refit=True`:** no ejecutado; por lectura, es la misma unión de folds que
  en una serie (`_utils.py:2060-2068`).
- **No ejecutados:** `ForecasterDirectMultiVariate`, `ForecasterRnn`, `fold_stride` y folds
  solapados. Pasan por los mismos helpers, pero no se han ejecutado.
- **La configuración exacta del Ridge** que dio 0.667 / 0.696 en la observación previa no se
  ha reproducido (ver 3.1).
- **Hyndman y Koehler (2006):** el PDF no se pudo extraer. La fórmula y la cita vienen de
  fpp3, de los mismos autores.
- **Issues y PRs de GitHub:** no revisados (trabajo solo en local). La historia viene de
  `git log -S` / `-G` y de las notas de versión.
- **Número exacto de tests que fallarían con el arreglo:** no medido (no se ha modificado
  código). La lista de 5.2 sale de buscar las métricas escaladas en los tests.
- **Escala por origen (5.1):** la tabla de h2o se calculó fuera de skforecast, con las
  predicciones del backtesting y `gap=0` (origen = inicio del test de cada fold). No se ha
  prototipado:
  - la escala por punto del OSA;
  - el caso con `gap`, folds solapados o NaN;
  - el multiserie;
  - la implementación con sumas acumuladas.

## 8. Implementación (2026-11-09)

Rama `fix/mase-scale-per-origin` (worktree `.claude/worktrees/fix-mase-scale`), desde
`origin/0.27.x` (`ecbe7756a`). Decisiones del plan: escala por origen (decisión 2, opción A),
`pooling` de las escaladas = media de los errores escalados de todos los puntos (decisión
3), métricas propias con `y_train` = la serie antes del primer origen (decisión 4), puntos
con escala nula o NaN fuera de MASE y RMSSE con `RuntimeWarning` (NaN si no queda ninguno).

**Literatura del `pooling` (decisión 3).**

- M4 promedia el MASE de cada serie (sktime/M4, arXiv 2005.08067).
- GluonTS (`evaluation/_base.py`, `"MASE": "mean"`) y sktime (`multilevel="uniform_average"`)
  promedian por serie.
- M5 usa una media ponderada de los RMSSE por serie (WRMSSE).

Es decir, `average` y `weighted_average`. El `pooling` de antes (suma de errores entre suma
de escalas) mezclaba unidades; agrupar los errores escalados es la definición sobre todos
los puntos y, para MASE, coincide con `weighted_average`.

**Código.**

- `metrics/metrics.py`:
  - `_naive_scales(y, origins)`: escalas por origen en O(n), con sumas acumuladas de
    `|diff|`, `diff²` y diferencias válidas; sin puente sobre los huecos.
  - `_scaled_metric_per_origin`.
  - `_is_scaled_metric`, por identidad.
  - Fuera los NOTE de `window_size`. Las funciones públicas no cambian.
- `model_selection/_utils.py`:
  - `_get_y_train_and_scales(y, origins, metrics)`: `y_train` fijo desde el primer valor
    observado, más las escalas.
  - `_calculate_metric_values`: sustituye el bucle de métricas en todos los caminos.
  - `_warn_undefined_scale`.
  - `_calculate_metrics_backtesting` recibe `scale_abs` y `scale_sq`.
  - `_calculate_metrics_backtesting_multiseries`: sin `window_size` y sin el formato largo
    (`stack`, `MultiIndex.from_product`, `isin`, `cumcount`). Recibe la columna `fold`,
    reindexa cada serie al span y concatena las escalas para el `pooling`.
  - Los dos OSA: sin la reconstrucción de `y_train` (`inverse_transform_training`).
- `model_selection/_validation.py`:
  - `origin_ilocs = fold[2][1]` de cada predicción, filtrado igual que los folds solapados,
    en `_backtesting_forecaster` y `_backtesting_stats` (uno y varios `estimator_id`);
  - en `backtesting_foundation` (una serie), por el mapa `fold → fold[2][1]`;
  - el multiserie (forecaster y foundation) pasa `fold`;
  - fuera la unión de entrenamientos y el `+ window_size`.
- `_train_test_split_one_step_ahead` de `ForecasterRecursive`, `ForecasterDirect` y
  `ForecasterRecursiveClassifier` devuelve un 7.º elemento, `test_index`: `y_test` es numpy
  y pierde filas con NaN, así que hace falta para el origen. `_search.py` lo pasa, junto con
  `y`, a `_calculate_metrics_one_step_ahead`.
- Docstrings del parámetro `metric` de las funciones públicas de backtesting y búsqueda (19
  sitios).

**Tests.**

Nuevos (valores esperados con la fórmula de referencia en numpy, no copiados de la salida):

- `test_naive_scales.py` (2) y `test_scaled_metric_per_origin.py` (2).
- `test_backtesting_forecaster.py`:
  - 9 casos frente a la referencia: `lags` 3 y 10, diferenciación, Direct, `refit`,
    `refit` con `fixed_train_size`, `gap`, `fold_stride` e incompleto;
  - predicciones iguales con 3 y 10 lags (estimador del lag 1), mismas métricas;
  - métrica propia que recibe `y.iloc[:30]` con y sin `refit`;
  - serie constante: `RuntimeWarning` y NaN.
- `test_backtesting_forecaster_multiseries.py`: ancho = dict con una serie que empieza
  tarde, y referencia por serie.
- `test_calculate_metrics_backtesting_multiseries.py`: referencia por serie y en `pooling`
  (MASE y RMSSE); sustituye al test que fijaba la exclusión de `window_size`.
- `test_calculate_metrics_one_step_ahead.py`: Recursive y Direct (`steps=5`) frente a
  `y[:t]`.
- `test_backtesting_stats.py`: Sarimax con y sin `refit`.
- `test_train_test_split_one_step_ahead.py`: `test_index` (incluido el caso con NaN de
  Direct, alineado con `y_test`).

Actualizados:

- **Tests unitarios de los helpers:** los valores nuevos se comprobaron contra la
  referencia antes de sustituirlos.
  - multiserie: 4 tests;
  - OSA: 2 tests;
  - OSA multiserie: 3 tests.
- **Tests de integración (34 en 7 archivos):** los actualizó un script que exige que solo
  cambien columnas de MASE o RMSSE. El orden de los candidatos de las búsquedas no cambió en
  ninguno.
- Los tests de equivalencia entre backtesting con `steps=1` y OSA siguen pasando sin
  cambios.

**Verificación** (`verify`):

- ruff limpio en los archivos cambiados.
- 4542 tests pasan, 3 saltados: `direct`, `direct_multivariate`, `metrics`,
  `model_selection` (search, utils, validation), `recursive`, `recursive_classifier`,
  `recursive_multiseries` y `foundation/tests_foundation_models`.
- Checks pasan: `generate_ai_context_files.py --check`, `page_meta.py` y
  `execute_notebooks --check`.
- `metrics.ipynb` reejecutado con el código del worktree.

**Docs.**

- `metrics.ipynb`: la frase de `pooling` frente a las medias.
- `skills/metric-selection/SKILL.md` y `llms-base.txt`, regenerados.
- `releases.md`: un destacado Fix, una línea en "Before upgrading", una entrada en Changed y
  otra en Fixed (ancho frente a dict), sin enlace hasta que haya PR.

**Resultado** (scripts de `repro/` con el arreglo):

| h2o, `TimeSeriesFold(steps=12, initial_train_size=168, refit=False)` | MAE | MASE `0.27.x` | MASE arreglo | A mano por origen |
|---|---|---|---|---|
| Seasonal naive (`offset=12`) | 0.0668 | 0.7121 | 0.7042 | 0.7042 |
| Ridge `lags=24` (sin exógenas) | 0.0650 | 0.6784 | 0.6898 | 0.6898 |
| Ridge `lags=24` + mes | 0.0658 | 0.6871 | 0.6984 | |
| ETS `m=12` (refit forzado) | 0.0649 | 0.6640 | 0.6843 | 0.6843 |

- Con el arreglo, ETS (0.6843) queda por debajo del Ridge sin exógenas (0.6898), como en
  MAE (0.0649 frente a 0.0650).
- Ancho y dict dan lo mismo: la serie que empieza tarde, 0.7335 en los dos (antes 0.7537
  frente a 0.7153).
- El `pooling` de MASE es igual a `weighted_average` (0.7755).
- `ForecasterDirect` con OSA ya no usa `y[24:157]`.

Búsquedas sobre `lags` (`_02_search.py`, sin burn-in):

| Caso | `0.27.x`: mejor MASE / MAE, Kendall | Arreglo |
|---|---|---|
| h2o, `TimeSeriesFold` | 24 / 24, 0.944 (36 y 48 cruzados) | 24 / 24, 1.000 |
| h2o, `OneStepAheadFold` | **48** / 24, 0.778 | 36 / 24, 0.944 |
| Sintética, varianza constante, semilla 0 (TSF y OSA) | 24 / 24 y 48 / 48, 1.000 | igual |

`_02b_seeds.py` (30 semillas con burn-in, como la tabla de 3.2): veces en que el mejor por
MASE no es el mejor por MAE, `TimeSeriesFold` / `OneStepAheadFold`.

| Serie | `0.27.x` | Arreglo | Pérdida máxima de MAE, `0.27.x` / arreglo | Kendall medio MASE~MAE, `0.27.x` / arreglo |
|---|---|---|---|---|
| Varianza constante | 6 / 7 | **0 / 0** | 2.26 % / - | 0.974 y 0.943 / 1.000 y 0.998 |
| Varianza crece x3 | 6 / 8 | 1 / 2 | 4.62 % / 0.18 % | 0.870 y 0.741 / 0.981 y 0.989 |
| Varianza decrece x1/3 | 2 / 8 | 1 / 0 | 3.31 % / 0.02 % | 0.981 y 0.950 / 0.998 y 0.996 |

En h2o con OSA, el mejor por MASE (36) y el mejor por MAE (24) tienen MAE 0.06538 y
0.06537. Los cambios que quedan son empates: la escala por origen pondera más los primeros
puntos (escala menor), con los mismos pesos para todos los candidatos, así que ya no
dependen del `window_size`.

### 8.1 Comprobaciones antes del commit (2026-11-09)

**`backtesting_foundation` con Chronos-2 small en CPU** (`_05_foundation.py`, h2o,
`TimeSeriesFold(steps=12, initial_train_size=168)`, `refit` forzado a `True`; Python 3.14.3,
numpy 2.5.4, pandas 2.3.3, torch 2.14.0). Referencia: escala con los valores anteriores a
`168 + 12 * fold`.

| Caso | MASE `0.27.x` | MASE arreglo = referencia | RMSSE `0.27.x` | RMSSE arreglo = referencia |
|---|---|---|---|---|
| Una serie | 0.5509 | 0.5685 | 0.4270 | 0.4390 |
| Una serie, `gap=2` | 0.5480 | 0.5668 | 0.4344 | 0.4479 |
| Una serie, `quantiles` (`q_0.5`) | 0.5509 | 0.5685 | 0.4270 | 0.4390 |
| Tres series (s3 empieza 30 fechas después): s1 / s2 / s3 | 0.5509 / 0.6082 / 0.5531 | 0.5685 / 0.6289 / 0.5723 | 0.4270 / 0.4848 / 0.4189 | 0.4390 / 0.4986 / 0.4315 |
| `pooling` (`weighted_average`) | 0.5777 (0.5707) | 0.5899 (0.5899) | 0.4565 | 0.4574 |

- Con el arreglo, todos los casos coinciden con la referencia, y ancho y dict dan lo mismo
  (también en `0.27.x`, donde foundation pasaba `window_size=0`).
- En `0.27.x`, la escala era la unión de los entrenamientos (`[0:192]`), y una métrica
  propia con `y_train` recibía 192 valores, con el test de los dos primeros folds. Con el
  arreglo recibe 168.

**PERF-04** (`_06_perf04.py`, la misma medición de `repro/PERF-04.py`, adaptada a las dos
firmas: 10.000 series de 1.500 observaciones, 28 fechas de test y 280.000 predicciones;
solo las métricas del backtesting multiserie).

| | MAE | MASE |
|---|---|---|
| `0.27.x` | 4.99 s, 57 MB | 6.44 s, 2618 MB |
| Arreglo (primera versión) | 4.77 s, 58 MB | 3.19 s, 450 MB |
| Arreglo (final) | 4.83 s, 58 MB | **2.15 s, 68 MB** |

- La primera versión seguía construyendo el `y_train` fijo de cada serie, y su
  concatenación para el `pooling`, aunque las métricas escaladas no lo usan. Ahora solo se
  construye cuando lo pide una métrica que no es escalada (`_get_y_train_and_scales`).
- Con ese cambio también se corrige un `pd.concat` vacío en el `pooling` (los dos
  multiserie) cuando ningún nivel tiene `y_train`.
- MASE es ahora más rápido que MAE: se calcula en numpy, sin la validación de scikit-learn
  por serie (PERF-03).
- PERF-04 queda resuelto por este arreglo. Su fila del roadmap proponía mantener la
  exclusión de `window_size`, que ya no existe.
- Nota de versión: entrada de rendimiento en Changed.

**Pendiente:**

- PR y enlaces en `releases.md`.
- Fila de PERF-04 en `roadmap.md`: marcarla como resuelta por este arreglo.
- La `summarize` de `comparing-forecasting-models.ipynb`, que ya podría usar la métrica del
  backtesting (la guía está en espera por este bug).
