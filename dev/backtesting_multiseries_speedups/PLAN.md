# Plan: extracción de folds y métricas del backtesting multiserie

Plan de dos mejoras de rendimiento del backtesting de `ForecasterRecursiveMultiSeries`. Salen de la revisión del 2026-11-19 del procesamiento de la exog, hecha en la rama `perf/multiseries-predict-backtesting`. La revisión concluyó que el alineamiento de la exog pesa poco y que el tiempo fuera del modelo se va en dos sitios: la extracción de los datos de cada fold y el cálculo de las métricas por nivel.

- **M1 (métricas):** pasar a las métricas cortes de arrays de numpy en vez de un grupo de pandas por nivel. Fila del roadmap: segunda parte de **PERF-03**.
- **F (folds):** montar la última ventana de cada fold a partir de los valores (F1) y extraer las series de entrenamiento por posición (F2). Fila del roadmap: **PERF-backtesting-folds-positions**.

**Estado:** plan escrito, sin implementar ni programar. Nada del paquete se ha tocado. Los prototipos se aplicaron en memoria (sustituyendo las funciones en tiempo de ejecución) y los scripts de medida eran de usar y tirar; el código de los prototipos está en el anexo y el método, en la sección 1.

**Dependencias:**

- F toca `_extract_data_folds_multiseries`, que la rama `perf/multiseries-predict-backtesting` (PERF-multiseries-names-pyarrow) ya modifica: no recorta el conjunto de entrenamiento si sus límites no cambian y corta una sola vez la exog compartida por varias series. F se implementa después de fusionar esa rama.
- M1 no depende de esa rama: `_calculate_metrics_backtesting_multiseries` es igual en las dos.
- Las baterías de identidad y los casos de tiempos que usa la verificación (sección 5) están en `dev/multiseries_predict_perf/scripts/` de esa rama (commit `2525c97`, con su `README.md`).

## 1. Medidas de partida

**Árbol medido:** commit `2525c97` de `perf/multiseries-predict-backtesting`, es decir, `0.28.x` (`3721a02`) más los cambios 1, 2, 4 y 6 de PERF-multiseries-names-pyarrow. Las cifras de "hoy" son de ese árbol, no de `0.28.x`: sin esos cambios la extracción de folds y `predict` son más lentos. Al implementar hay que volver a medir sobre el árbol de ese momento.

**Entorno:** Python 3.12.3, pandas 3.1.1 (con `pyarrow`), numpy 2.5.5, scikit-learn 1.9.1, lightgbm 4.7.0, 4 núcleos, máquina compartida (ruido de varios puntos). Solo este entorno: falta pandas 2 y pandas 3 sin `pyarrow`.

### 1.1 Escenario de referencia

500 series de 1000 observaciones, 5 exog float en diccionario (un DataFrame por serie), índice horario sin nombre y con `freq`. `LGBMRegressor` por defecto, `lags=24`, `encoding="ordinal"`. Backtesting de 10 folds de 24 pasos, métrica MAE, `n_jobs="auto"` (con LightGBM resuelve a 1).

Reparto actual:

| Etapa | Sin refit (4,39 s) | Con refit (25,65 s) |
|---|---|---|
| Ajuste de LightGBM | 49 % | 74 % |
| Extraer folds (`_extract_data_folds_multiseries`) | 16 % | 11 % |
| `_create_train_X_y` | 7 % | 9,5 % |
| Métricas (`_calculate_metrics_backtesting_multiseries`) | 8,5 % | 1,5 % |
| `check_predict_input` | 6 % | 1 % |
| Resto de `_create_predict_inputs` | 3 % | 0,6 % |

Impacto de los prototipos, acumulados. El total es la mediana del ajuste de LightGBM (1,93 s sin refit, 19,25 s con refit) más el mínimo del tiempo fuera del modelo (4 repeticiones sin refit, 2 con refit):

| Prototipos acumulados | Sin refit: fuera del modelo | Sin refit: total | Con refit: fuera del modelo | Con refit: total |
|---|---|---|---|---|
| Hoy | 2,29 s | 4,22 s | 6,32 s | 25,58 s |
| + M1 (cortes de numpy, misma función de scikit-learn) | 2,00 s | 3,93 s (−6,9 %) | 6,01 s | 25,27 s (−1,2 %) |
| + F1, F2 y F3 (ventana, entrenamiento y exog por posición) | 1,46 s | 3,40 s (−19,5 %) | 4,70 s | 23,95 s (−6,4 %) |

Calcular además MAE con su fórmula en numpy (lo que propone PERF-03) quita 0,10 s más sin refit. Con refit hay solo 2 repeticiones: las diferencias de menos de 0,2 s no son fiables.

Identidad: con `Ridge`, métricas y predicciones `equals` a las de hoy con y sin refit. Con LightGBM no se puede comprobar bit a bit: dos ejecuciones idénticas del código actual ya difieren en 3e-16 (el modelo no es reproducible entre ejecuciones con varios hilos).

### 1.2 Desglose de las métricas

1000 series, 10 folds de 24 pasos (240 000 predicciones), sin exog, `Ridge`. `_calculate_metrics_backtesting_multiseries` con MAE: 797 ms, el 34 % de ese backtesting.

| Paso del bucle por nivel | ms | % |
|---|---|---|
| Llamada a la métrica de scikit-learn con Series | 448 | 60 |
| `y_true_y_pred_grouped.get_group(level)` | 204 | 28 |
| `group['y_true']` y `group['y_pred']` | 56 | 8 |
| `predictions.index.get_level_values('level').unique()` | 19 | 3 |
| Resto (`groupby`, `dropna`, pertenencia, agregados) | 14 | 2 |

- Una llamada con 240 valores: MAE de scikit-learn con Series 293 µs, con arrays 175 µs, `np.mean(np.abs(a - b))` 3,3 µs. `get_group` 124 µs.
- Con tres métricas (MAE, MSE y MASE) la función tarda 1419 ms; con MASE sola, 375 ms (skforecast ya la calcula con numpy).
- Bucle completo (hoy unos 715 ms): 203 ms con cortes de numpy y la misma función; 7,3 ms con cortes y la fórmula directa; 1,1 ms vectorizado con `bincount` (este último no es idéntico bit a bit).

### 1.3 Desglose de la extracción de folds

1000 series de 600 filas, 5 exog en diccionario, ventana de 24:

| Paso | Sin refit, 10 folds | Con refit, 5 folds |
|---|---|---|
| Exog de test: `.loc` por serie | 516 ms (33 %) | 185 ms (9,5 %) |
| Ventana: `pd.DataFrame(dict)` | 476 ms (30 %) | 147 ms (7,5 %) |
| Ventana: `.loc` por serie | 316 ms (20 %) | 434 ms (22 %) |
| Series de entrenamiento | 216 ms (14 %), solo el primer fold | 992 ms (51 %) |
| Exog de entrenamiento: `.loc` por serie | 31 ms (2 %) | 175 ms (9 %) |
| Total de la función | 1706 ms | 2096 ms |

En las series de entrenamiento con refit: `first_valid_index` y `last_valid_index` 451 ms, segundo `.loc` 226 ms, primer `.loc` 164 ms, comprobación de todo NaN 151 ms.

Alternativas, 10 folds con sus resultados vivos, 1000 series:

| Pieza | Hoy | Alternativa |
|---|---|---|
| Última ventana | de 557 a 639 ms | 6,9 ms rellenando un array por fold desde los valores de cada serie; 2,8 ms cortando una matriz común |
| Series de entrenamiento de un fold | 199 ms | 22 ms con posiciones válidas precalculadas y un `iloc` |
| Exog de test | 357 ms | 258 ms con `iloc` |

**Recolector de basura.** Desactivándolo, el backtesting completo (Ridge, 1000 series) baja de 4213 a 3423 ms (19 %) y la extracción de folds sin refit, de 1706 a 1089 ms (36 %). Interpretación sin verificar aparte: `data_folds` mantiene vivos unos 2000 objetos de pandas por fold y cada recolección los recorre. Consecuencias: las cifras por paso son orientativas (la recolección se carga al paso que la dispara) y quitar objetos ahorra más en contexto que en aislado.

## 2. Cambio M1: métricas con cortes de numpy

Fichero: `skforecast/model_selection/_utils.py`, función `_calculate_metrics_backtesting_multiseries`. La llaman `_backtesting_forecaster_multiseries` y `_backtesting_foundation` (`skforecast/model_selection/_validation.py`).

### 2.1 Qué hace hoy

```python
for level in levels:
    if level in levels_in_predictions and level in y_true_y_pred_grouped.groups:
        group = y_true_y_pred_grouped.get_group(level)      # un DataFrame por nivel
        positions = y_true_y_pred_grouped.indices[level]
        y_true = group['y_true']                            # Series
        y_pred = group['y_pred']
        metrics_level, n_undefined = _calculate_metric_values(...)
        n_undefined_levels[level] = (n_undefined, len(group))
```

### 2.2 Qué va a hacer

```python
y_true_values = y_true_y_pred['y_true'].to_numpy()          # una vez, antes del bucle
y_pred_values = y_true_y_pred['y_pred'].to_numpy()
...
positions = y_true_y_pred_grouped.indices[level]            # ya se calcula hoy
y_true = y_true_values[positions]
y_pred = y_pred_values[positions]
```

`len(group)` pasa a `len(positions)`, y `group['lower_bound']` y `group['upper_bound']` se sustituyen igual.

**Por qué da lo mismo.** `indices[level]` son las posiciones de las filas del nivel en `y_true_y_pred` (el código ya confía en ello: `scale_abs[positions]`). Entran los mismos valores, en el mismo orden, a la misma función. `_scaled_metric_per_origin` y `_calculate_interval_metrics` ya convierten sus entradas con `np.asarray(..., dtype=float)`.

**Comprobado.** Las nueve métricas de regresión de `_get_metric` dan el mismo valor bit a bit y el mismo tipo de resultado con Series y con arrays (200 grupos aleatorios de 240 valores por métrica, pandas 3.1.1): `mean_squared_error`, `root_mean_squared_error`, `mean_absolute_error`, `mean_absolute_percentage_error`, `mean_squared_log_error`, `mean_absolute_scaled_error`, `root_mean_squared_scaled_error`, `median_absolute_error` y `symmetric_mean_absolute_percentage_error`.

### 2.3 Condición del camino nuevo: métricas propias

Una métrica del usuario recibe hoy Series de pandas con las fechas en el índice. Con arrays, una función que use `.index`, `.iloc` o `.values` fallaría o cambiaría. La documentación no promete un tipo, y la validación de un paso adelante (`_predict_and_calculate_metrics_one_step_ahead_multiseries`) ya pasa a las métricas el array de `estimator.predict`.

Propuesta: **arrays solo si todas las métricas son conocidas**; con alguna métrica propia, el bucle de hoy sin cambios. Así no cambia el comportamiento de nadie y se mantiene la identidad de resultados.

- Una métrica es conocida si `getattr(m, '__wrapped__', m)` es una de las nueve funciones de la lista (`add_y_train_argument` envuelve con `functools.wraps`). Cubre también a quien pasa la función de scikit-learn en vez de su nombre.
- El conjunto de funciones conocidas sale del diccionario de `_get_metric` (`skforecast/metrics/metrics.py`), que habría que subir a constante de módulo o exponer con una función auxiliar. Solo las nueve de regresión comprobadas, no las de clasificación.

### 2.4 Micro-cambios opcionales en la misma función

Solo si la batería de identidad pasa; si no, fuera sin más:

- `level in y_true_y_pred_grouped.groups` crea un índice por grupo la primera vez (16 ms con 1000 niveles). `level in y_true_y_pred_grouped.indices` usa el diccionario que ya hace falta.
- `levels_in_predictions` (19 ms) es redundante: un nivel con filas en `y_true_y_pred` está en `predictions`. Quitarlo exige comprobar nombres de nivel que no son texto (enteros, mezclados).

### 2.5 Relación con PERF-03

PERF-03 (métricas pedidas por nombre calculadas en numpy, medida en su día con la búsqueda `OneStepAheadFold`) y M1 quitan costes distintos del mismo bucle y se suman:

- M1 quita `get_group` y la conversión de Series que scikit-learn hace al validar: el bucle baja de unos 715 a 203 ms con 1000 niveles.
- PERF-03 quita el resto de la validación de scikit-learn (175 µs por llamada con arrays): con MAE, de 203 a 7,3 ms, idéntico bit a bit en los niveles comparados.

Datos de esta revisión que sirven a PERF-03: exige, métrica a métrica, identidad bit a bit con scikit-learn en pandas 2 y 3, también en el agregado `pooling` (cientos de miles de valores, donde puede cambiar el orden de las sumas), y decidir qué pasa con los errores y avisos de validación de scikit-learn (por ejemplo, `mean_squared_log_error` con valores negativos). Solo se ha comprobado MAE.

La condición "métricas conocidas" de 2.3 es la misma que necesita PERF-03, así que conviene implementarlos en la misma PR o en commits seguidos.

### 2.6 Tests

En `skforecast/model_selection/tests/tests_utils/test_calculate_metrics_backtesting_multiseries.py` (22 tests hoy), con valores esperados fijos:

- Cada una de las nueve métricas conocidas, sola y en lista, con y sin intervalos.
- Lista con una métrica conocida y una propia: se usa el bucle de hoy y la propia recibe Series (el test comprueba el tipo que recibe).
- Métrica propia con `y_train`.
- Función de scikit-learn pasada como callable.
- Niveles con nombres enteros, y un nivel sin filas evaluadas.

## 3. Cambio F: extracción de folds por posición

Fichero: `skforecast/model_selection/_utils.py`, función `_extract_data_folds_multiseries`, **solo la rama de diccionario**. La rama de tabla ancha (la de `ForecasterDirectMultiVariate` y `ForecasterRnn`) no se toca.

Llamantes: `_backtesting_forecaster_multiseries`, `_backtesting_foundation` (con `window_size=0`) y `ForecasterRecursiveMultiSeries._train_test_split_one_step_ahead`. Las condiciones se comprueban dentro de la función, así que los tres siguen siendo correctos sin cambios.

### 3.1 Idea

Todas las series tienen la misma `freq` (o el mismo `step`), así que cada una es un tramo contiguo de `span_index`. Con la posición de inicio de cada serie en `span_index`, un corte por etiqueta (`v.loc[a:b]`) es un corte por posición, y no hace falta crear un Series por serie y fold para montar la ventana.

### 3.2 Preparación, una vez antes del bucle de folds

```python
start = span_index.get_indexer([v.index[0] for v in series.values()])   # -1 si no está
length = np.array([len(v) for v in series.values()])
end = start + length
arrays = [v.to_numpy() for v in series.values()]     # vistas, sin copia con float
valid_pos = [np.flatnonzero(v.notna().to_numpy()) for v in series.values()]
```

Coste medido en el escenario de referencia: 14 ms, una vez.

### 3.3 Condiciones

Si alguna no se cumple, toda la llamada usa el camino actual (todo o nada, sin mezclar caminos entre series).

**Comunes a F1 y F2** (la serie está en la rejilla de `span_index`):

- Todos los valores de `series` son `pd.Series`.
- El índice de cada serie es del tipo de `span_index`, con su mismo dtype (unidad y zona horaria) y su misma `freq` o `step`.
- La primera etiqueta de cada serie está en `span_index` (`start >= 0`). La misma `freq` no garantiza la misma fase: una serie horaria que empieza a y media no está en la rejilla.

**Solo F1** (la ventana tiene que salir idéntica a `pd.DataFrame(dict)`):

- Todas las series comparten un `np.dtype` numérico (clases `f`, `i` o `u`). Hoy cada columna conserva el dtype de su serie; con dtypes distintos, de extensión u `object`, camino actual.
- Todos los índices tienen el mismo nombre. Hoy la ventana hereda el nombre si es común (`'date'`) y `None` si difieren; `span_index` no tiene nombre, así que el nuevo índice es `span_index[a:b].rename(nombre)`.
- El intervalo de la ventana del fold mide `window_size`. Si no, ese fold usa el camino actual (con cobertura parcial, `pd.DataFrame(dict)` une índices y rellena con NaN).

### 3.4 F1: última ventana

**Hoy:**

```python
for k, v in series.items():
    v_last_window = v.loc[last_window_loc_start:last_window_loc_end]
    if ((externally_fitted or k in series_train) and len(v_last_window) >= window_size):
        series_last_window[k] = v_last_window
series_last_window = pd.DataFrame(series_last_window)
```

**Propuesta:**

```python
covered = (np.minimum(lw_end, end) - np.maximum(lw_start, start)) >= window_size
keep = [j for j, k in enumerate(names) if covered[j] and (externally_fitted or k in series_train)]
out = np.empty((lw_end - lw_start, len(keep)), dtype=common_dtype, order='F')
for col, j in enumerate(keep):
    out[:, col] = arrays[j][lw_start - start[j]:lw_end - start[j]]
series_last_window = pd.DataFrame(
    out, index=span_index[lw_start:lw_end].rename(index_name), columns=columns_of(keep), copy=False
)
```

`dropna(axis=1, how="any")` y `list(series_last_window.columns)` quedan como hoy.

Decisiones de diseño, con lo comprobado:

- **Sin matriz común.** Una matriz `len(span_index) × n_series` da 2,8 ms por 10 folds, pero duplica la memoria de las series (400 MB con 10 000 series de 5000 observaciones). Rellenar un array por fold desde `v.to_numpy()` (vista sin copia) da 6,9 ms y no usa memoria extra. Se elige el relleno por fold.
- **Etiquetas de columna.** `pd.DataFrame(dict)` y `pd.DataFrame(array, columns=lista)` no dan las mismas columnas con cualquier clave: con claves `[1, 2, 3]` el primero da `RangeIndex` y el segundo `Index`; con tuplas, `MultiIndex` frente a `Index` de tuplas. Con texto y con claves mezcladas son `identical`. Por eso `columns_of(keep)` construye las columnas por el mismo camino que hoy (`pd.DataFrame(dict.fromkeys(nombres), index=[0]).columns`) y las guarda por conjunto de series conservadas, que casi nunca cambia entre folds. Comprobado en pandas 3.1.1: `identical` a las columnas de hoy con texto, enteros consecutivos y no consecutivos, `numpy.int64`, claves mezcladas y tuplas. Alternativa más simple: camino nuevo solo si todos los nombres son `str`.
- **Ninguna serie conservada.** Hoy `pd.DataFrame({})` da una tabla `(0, 0)` con `RangeIndex`. El camino nuevo devuelve lo mismo, no una tabla de `window_size` filas sin columnas.
- **`window_size=0`** (`_backtesting_foundation`). Hoy sale una tabla de 0 filas con una columna por serie y un índice de fechas vacío con `freq`. Comprobado en el caso simple (pandas 3.1.1): el relleno por fold da una tabla `equals` y con el índice `identical`. Se cubre con un test; si en pandas 2 no saliera idéntico, se excluye (`window_size > 0`).
- **Comprobado en el caso simple** (series float de igual longitud): `equals`, índice `identical` (con `freq`), columnas `identical` y dtypes iguales.

### 3.5 F2: series de entrenamiento

**Hoy** (cuando cambian los límites del entrenamiento):

```python
v_train = v.loc[train_loc_start:train_loc_end]
if not v_train.isna().to_numpy().all():
    first_valid_index = v_train.first_valid_index()
    last_valid_index  = v_train.last_valid_index()
    if first_valid_index is not None and last_valid_index is not None:
        v_train = v_train.loc[first_valid_index : last_valid_index]
        if len(v_train) > window_size:
            series_train[k] = v_train
```

**Propuesta:**

```python
i0 = min(max(train_start - start[j], 0), length[j])
j0 = min(max(train_end - start[j], 0), length[j])
p, q = np.searchsorted(valid_pos[j], [i0, j0])
if q > p:
    first, last = valid_pos[j][p], valid_pos[j][q - 1]
    if last - first + 1 > window_size:
        series_train[k] = v.iloc[first:last + 1]
```

- No depende del dtype: `valid_pos` usa `notna`, el mismo criterio que `isna`. Solo necesita las condiciones comunes.
- Un tramo vacío o todo NaN da `q == p`, igual que hoy (`.all()` de un tramo vacío es `True`).
- **Trampa de los límites.** Hoy el final es `span_index[train_iloc_end - 1]`: con `train_iloc_end == 0` el índice `-1` da la última fecha y el corte abarca toda la serie. Con `initial_train_size=None` los límites son `[0, window_size]` (comprobado) y no ocurre, pero el camino nuevo debe ir al actual si `train_iloc_end <= train_iloc_start`.
- Comprobado en el caso simple: mismas claves, valores e índices `identical`.

### 3.6 F3 (opcional): exog por posición

Cortar cada exog con `iloc` en vez de `.loc`, con las condiciones comunes aplicadas a cada objeto (una exog sin `freq` sigue con `.loc`). Gana un 28 % de ese paso (de 357 a 258 ms por 10 folds y 1000 series).

Recomendación: fuera de la primera entrega. Ahorra poco y la ganancia de fondo en la exog exige que `predict` reciba un bloque en vez de N tablas, que es el rediseño de PERF-predict-numpy. Nota: en el prototipo medido en el escenario de referencia, F3 iba junto a F1 y F2, así que la cifra de −0,53 s lo incluye.

### 3.7 Tests

En `skforecast/model_selection/tests/tests_utils/test_extract_data_folds_multiseries.py`, comparando con valores esperados fijos y cubriendo los dos caminos:

- Series de distinta longitud: una que empieza tarde, una que acaba antes y una más corta que la ventana.
- NaN en los extremos del tramo de entrenamiento, NaN dentro de la ventana (con `dropna_last_window` a `True` y `False`) y una serie todo NaN en un fold.
- `externally_fitted=True`, `gap > 0`, folds solapados y `RangeIndex`.
- `window_size=0`.
- Ningún nivel con ventana válida en un fold (tabla vacía).
- Nombres de serie enteros, mezclados y tuplas; índice con nombre, y series con nombres de índice distintos.
- Series enteras, series con dtypes distintos entre sí y una serie con dtype de extensión (`Float64`): camino actual.
- Serie fuera de la rejilla (misma `freq`, otra fase): camino actual.
- El conjunto de entrenamiento se sigue compartiendo entre folds con los mismos límites (el test existente debe pasar sin cambios).

## 4. Orden y forma de trabajo

Un commit por cambio, revisado antes de pasar al siguiente: explicación previa (qué hace hoy el código, qué va a hacer, por qué da lo mismo y cuándo se usa cada camino), implementación con sus tests, comprobaciones de la sección 5 y ficha de revisión con el diff, la identidad, los tiempos y el riesgo que queda.

1. **M1**, con PERF-03 o justo antes. Pocas líneas en una función, sin condiciones sobre los datos, y no depende de ninguna otra rama.
2. **F2**, después de fusionar PERF-multiseries-names-pyarrow. Solo necesita las condiciones comunes.
3. **F1.** Es el que más ahorra sin refit y el que más condiciones tiene.
4. F3, solo si se decide (sección 6).

Dependencias internas: F1 y F2 comparten la preparación (3.2) y las condiciones comunes (3.3); M1 es independiente.

## 5. Verificación de cada cambio

Las baterías de identidad (`run_identity.sh`, `identity_*.py`) y los casos de tiempos (`bench_ms.py`) son los de `dev/multiseries_predict_perf/scripts/`, que hoy solo están en la rama `perf/multiseries-predict-backtesting` (commit `2525c97`; su `README.md` explica cómo prepararlos). Si esa carpeta se ha borrado al implementar, se recuperan de ese commit.

1. **Identidad** contra el árbol del commit anterior en los tres entornos (pandas 2, pandas 3 con y sin `pyarrow`), con las cinco baterías (`ms`, `trims`, `dtypes`, `exog` y `labels`). Criterio: 0 diferencias en valores, dtypes, índices, avisos y errores.
2. **Casos que hay que añadir a las baterías**, porque hoy no los cubren:
   - `identity_ms.py` solo usa `mean_absolute_error` y la pareja MAE con MASE. Añadir las nueve métricas, una métrica propia (con y sin `y_train`), una lista mezclada y backtesting con intervalos.
   - Backtesting con nombres de serie que no son texto, índice de las series con nombre y series de dtype entero.
3. **Tests** de los módulos tocados con pandas 2.3 y pandas 3: `skforecast/model_selection/tests/tests_utils/`, los de backtesting de `skforecast/model_selection/tests/tests_validation/` y los de `_train_test_split_one_step_ahead`. Los de `backtesting_foundation`, si se pueden ejecutar sin los paquetes de deep learning.
4. **Tiempos** con `bench_ms.py`: los casos `backtesting|...` (20, 100, 600 y 1000 series; con y sin exog; `steps=1` y `steps=24`; con refit), frente al árbol del commit anterior. Añadir un caso con varias métricas y uno con series de distinta longitud. Criterio: al menos un 3 % en pandas 3 en los casos a los que afecta y ninguno peor fuera del ruido.
5. **`verify`** antes de dar el cambio por terminado.
6. Los casos en los que el camino nuevo no se aplica (métrica propia, dtypes mezclados, serie fuera de la rejilla) pagan las comprobaciones de 3.3: medirlos como adversos.

## 6. Decisiones abiertas

1. **Métricas propias.** Bucle de hoy cuando hay alguna (recomendado, sin cambio de comportamiento) o arrays para todas con nota de versión.
2. **Etiquetas de columna en F1.** Columnas construidas por el camino de pandas y guardadas por conjunto de series (cubre cualquier nombre) o camino nuevo solo con nombres `str` (más simple).
3. **F3.** Si se prototipa después o se deja para PERF-predict-numpy.
4. **Versión de PERF-backtesting-folds-positions.** Hoy "por decidir".
5. **Nota de versión.** No cambia la API pública ni los resultados; si se documenta, es una línea de rendimiento en `docs/releases/releases.md`.

## 7. Fuera de alcance

- `check_predict_input` en cada fold (6 % sin refit en el escenario de referencia): separar las comprobaciones que no cambian entre folds de los avisos que sí. Es el bloque B de PERF-predict-numpy, que hoy se limita a `ForecasterRecursive` y `ForecasterDirect`.
- Exog en bloque (un array para todas las series en vez de un DataFrame por serie): rediseño que arrastra a `check_predict_input` y `_create_predict_inputs`.
- `identical` por `equals` al alinear la exog en `predict` (candidato 4b del seguimiento de PERF-multiseries-names-pyarrow): no afecta al escenario de referencia; con el índice de la exog con nombre, +0,34 s fuera del modelo (de 2,07 a 2,41 s).
- Arreglos pequeños de `_create_train_X_y` (`iloc[self.window_size:, ]` con coma, comprobación de dtypes de la exog en diccionario, copias profundas): estimación de 40 a 60 ms por ajuste en el escenario de referencia, sin medir en contexto.
- Reducir los objetos vivos en `data_folds` para aliviar el recolector de basura: lo consiguen en parte F1 y F2; no se plantea desactivar ni ajustar el recolector.

## Anexo: código de los prototipos

Referencia para la implementación, no código final. Solo se han probado con series float de igual longitud, exog en diccionario con `freq`, MAE y sin intervalos.

### A.1 Métricas (M1)

Sustituciones sobre el código de `_calculate_metrics_backtesting_multiseries`:

```python
# antes de `metrics_levels = []`
_yt_values = y_true_y_pred['y_true'].to_numpy()
_yp_values = y_true_y_pred['y_pred'].to_numpy()

# dentro del bucle, en lugar de get_group y las dos columnas
positions = y_true_y_pred_grouped.indices[level]
y_true = _yt_values[positions]
y_pred = _yp_values[positions]

# len(group) -> len(positions), en sus dos usos
```

Falta en el prototipo: la condición de métricas conocidas (2.3) y los límites del intervalo.

### A.2 Extracción de folds (F1, F2 y F3)

```python
def extract_p2(series, folds, span_index, window_size, exog=None,
               dropna_last_window=False, externally_fitted=False):
    names_ = list(series)
    off = np.array([span_index.get_loc(v.index[0]) for v in series.values()])
    lens = np.array([len(v) for v in series.values()]); end = off + lens
    mat = np.full((len(span_index), len(names_)), np.nan)
    valid_pos = {}
    for j, (k, v) in enumerate(series.items()):
        vals = v.to_numpy(dtype=float); mat[off[j]:end[j], j] = vals
        valid_pos[k] = np.flatnonzero(~np.isnan(vals))
    eoff = (
        {k: (None if v is None else span_index.get_loc(v.index[0])) for k, v in exog.items()}
        if exog is not None else None
    )
    previous = None; exog_train = None
    for fold in folds:
        (a, b), (lwa, lwb), (ta, tb) = fold[1], fold[2], fold[3]
        extract_train = (a, b) != previous; previous = (a, b)
        if extract_train:
            series_train = {}
            for j, (k, v) in enumerate(series.items()):
                i0 = max(a - off[j], 0); j0 = min(b - off[j], lens[j])
                if j0 <= i0:
                    continue
                vp = valid_pos[k]; p = np.searchsorted(vp, i0); q = np.searchsorted(vp, j0)
                if q > p:
                    first, last = vp[p], vp[q - 1]
                    if last - first + 1 > window_size:
                        series_train[k] = v.iloc[first:last + 1]
        covered = (np.minimum(lwb, end) - np.maximum(lwa, off)) >= window_size
        keep = [
            j for j, k in enumerate(names_)
            if covered[j] and (externally_fitted or k in series_train)
        ]
        series_last_window = pd.DataFrame(
            mat[lwa:lwb][:, keep], index=span_index[lwa:lwb],
            columns=[names_[j] for j in keep]
        )
        if dropna_last_window:
            series_last_window = series_last_window.dropna(axis=1, how="any")
        levels_last_window = list(series_last_window.columns)
        exog_test = None
        if exog is not None:
            if extract_train:
                exog_train = {}
                for k, v in exog.items():
                    if v is None:
                        exog_train[k] = None
                    else:
                        v_train = v.iloc[max(a - eoff[k], 0):max(b - eoff[k], 0)]
                        if len(v_train) > 0:
                            exog_train[k] = v_train
            exog_test = {}; by_id = {}
            for k, v in exog.items():
                if v is None:
                    exog_test[k] = None
                elif externally_fitted or k in exog_train:
                    v_test = by_id.get(id(v))
                    if v_test is None:
                        v_test = v.iloc[max(ta - eoff[k], 0):max(tb - eoff[k], 0)]
                        by_id[id(v)] = v_test
                    if not v_test.empty:
                        exog_test[k] = v_test
        yield series_train, series_last_window, levels_last_window, exog_train, exog_test, fold
```

Diferencias con lo que propone el plan: usa una matriz común (el plan, relleno por fold, 3.4), `get_loc` por serie (el plan, un `get_indexer`), no comprueba ninguna condición de 3.3, no pone el nombre del índice ni construye las columnas por el camino de pandas, y no trata la tabla vacía, `window_size=0` ni la trampa de los límites.
