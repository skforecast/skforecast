# Optimización del `fit()` de `ForecasterRecursiveMultiSeries`: informe de cierre

Rama `refactor/optimize_multiseries_fit`, commits sobre `082aa0966` (`0.26.x`):

| Commit | Contenido |
|---|---|
| `b81622fd6` | Filas de cada serie por tramos (residuos y pesos) y exógenas antes del bucle por serie |
| `cdd51597f` | `X_train` en un único bloque float; columnas `'onehot'` en el bloque; fix de `predict` con `'onehot'` |
| `641010ae8` | Variables de calendario en el bloque |
| posterior a `b6e4b4d0d` | Columnas `'onehot'` leídas como vista al localizar las filas de cada serie |

Este archivo empezó como plan y se ha reescrito al terminar: cuenta qué se ha mejorado, por
qué, con qué datos y cómo se ha comprobado que los resultados no cambian. El plan completo,
con el diario de cada commit, los prototipos y las medidas intermedias, está en
`git show 641010ae8:dev/PLAN_multiseries_fit_optimizations.md`. Cómo se construyen ahora las
matrices de entrenamiento, los casos límite y los invariantes:
`dev/multiseries_train_matrices.md`.

## 1. Resumen

- Solo cambia el `fit()` de `ForecasterRecursiveMultiSeries` (y todo lo que pasa por
  `_create_train_X_y`: `create_train_X_y`, `set_in_sample_residuals`, `OneStepAheadFold`).
  `predict()` y el resto de forecasters no cambian de velocidad.
- La ganancia absoluta es de unas décimas de segundo a varios segundos por `fit()`; el
  porcentaje depende del estimador: es grande con estimadores ligeros y se diluye con
  estimadores pesados. Los casos con más ganancia son `series_weights`, `'onehot'` con muchas
  series y `calendar_features`.
- Los resultados no cambian (`X_train`, `y_train`, predicciones, residuos e intervalos
  bit a bit idénticos), salvo tres cambios documentados en las matrices de
  `create_train_X_y` (el índice de `X_train` toma siempre el nombre del índice de `series`, y
  las columnas `'onehot'` y las de calendario enteras pasan de `int64` a `float64`) y las
  predicciones con `'onehot'` y series en orden no alfabético, que eran erróneas (2.5).
- Por el camino se corrigieron cuatro errores, uno de ellos en las predicciones con
  `'onehot'` desde la 0.22.0 (sección 2.5).

Resultado final, código real antes (`082aa0966`) frente a después (`641010ae8`), 500 series
de 2,000 observaciones salvo donde se indica (escenarios en la sección 2, método y tabla
completa en la 3):

| Caso | `fit()` antes | después | Cambio |
|---|---|---|---|
| A: sin exógenas | 2.69 s | 2.09 s | -22% |
| B: 10 exógenas float | 3.22 s | 2.79 s | -13% (memoria pico de `create_train_X_y` 953 a 500 MB) |
| C: 5 exógenas float y 5 category | 4.03 s | 3.63 s | -10% |
| A con `calendar_features` (20 columnas) | 4.23 s | 3.44 s | -19% (memoria pico 1,095 a 430 MB) |
| A con `encoding='onehot'`, 300 series | 9.80 s | 3.34 s | -66% (2.9 veces más rápido) |
| A con `series_weights` | 33.1 s | 2.40 s | -93% (14 veces más rápido) |
| A con LightGBM de 100 árboles | 6.58 s | 5.70 s | -13% |
| A con `Ridge` | 2.05 s | 1.09 s | -47% (1.9 veces más rápido) |
| A con `encoding=None` | 2.34 s | 2.46 s | sin cambio (ruido, ver 3.2) |

La ganancia absoluta no baja con estimadores más pesados (0.6 s con 25 árboles, 0.9 s con
100), pero sí su peso relativo. Con `'onehot'`, un cambio posterior a la tabla (2.4) reduce
además el `fit()` un 11% y su pico de memoria de 2,845 a 1,675 MB.

## 2. Qué se ha mejorado y por qué

Escenarios del estudio (`dev/profiling_multiseries_fit/common.py`, `make_data`): 500 series
de 2,000 observaciones diarias. A: sin exógenas. B: 10 exógenas `float64`. C: 5 `float64` y
5 `category`. Forecaster: `lags=24`, `RollingFeatures(['mean', 'std'], [7, 28])`,
`encoding='ordinal'`, LightGBM de 25 árboles. `X_train` de unas 986,000 filas.

### 2.1 Filas de cada serie por tramos (`b81622fd6`)

**Problema.** Tres sitios recorrían todas las filas de `X_train` una vez por serie:

- `fit` y `set_in_sample_residuals`, al repartir los residuos in-sample por serie: una
  máscara booleana de todas las filas por serie (`X_train['_level_skforecast'] == código`).
  Era la única etapa no lineal del `fit()`: 0.15 a 0.23 s con 500 series, 3.1 s con 1,000
  series de 4,000 observaciones (20 a 25% del `fit()`).
- `create_sample_weights` con `series_weights`: contaba las filas de cada serie con el `sum`
  de Python sobre una `Series` booleana de un millón de elementos. Era el 88% del `fit()`
  (21.5 de 24.4 s).
- `create_sample_weights` con `weight_func`: una máscara por serie (0.4 s).

**Solución.** `_create_train_X_y` escribe las filas serie a serie y los descartes de filas
con NaN conservan el orden, así que las filas de cada serie son contiguas.
`_get_level_row_slices` lee la codificación una sola vez, localiza los cambios de serie y
devuelve `{serie: slice}`. Los tramos son vistas y seleccionan los mismos elementos en el
mismo orden que las máscaras, de ahí la identidad bit a bit. Si las filas de una serie no
son contiguas lanza `ValueError` (solo puede ocurrir si un usuario pasa a
`create_sample_weights` una matriz reordenada).

**Efecto** (código real, A/B en el mismo proceso): el reparto de residuos es 3.5 veces más
rápido con 500 series y 11 veces con 1,000 (1.62 a 0.145 s); `create_sample_weights` pasa de
22.3 s a 7 ms con `series_weights` y de 0.5 s a 21 ms con `weight_func`; el `fit()` es un 4 a
10% más rápido sin pesos.

### 2.2 Exógenas antes del bucle por serie (`b81622fd6`)

Refactor sin cambio de rendimiento (A/B: 0.97 a 1.00), necesario para el bloque único: para
reservar el bloque hay que saber antes cuántas columnas exógenas son `float64`, así que las
exógenas se procesan (concatenación, `transformer_exog`, codificación de categóricas) antes
del bucle por serie, no dentro. `_create_train_X_y_single_series(y)` queda solo con lags y
window features. La comprobación de longitud de las series se movió a `_create_train_X_y`,
antes de las exógenas: si no, con todas las series demasiado cortas el codificador de
categóricas se ajustaba con 0 filas y el usuario veía un error de sklearn en lugar del de
skforecast.

### 2.3 `X_train` en un único bloque float (`cdd51597f`)

**Problema.** Los lags y las window features se escribían en un array y la columna del
nivel se añadía después como un segundo bloque de pandas.

- Sin exógenas, `X_train` llegaba al estimador con dos bloques float y era el estimador quien
  los intercalaba, una vez en `fit` y otra en `predict(X_train)` de la etapa de residuos
  (177 ms por conversión; `predict(X_train)` 477 frente a 298 ms).
- Con exógenas, el `pd.concat(axis=1)` final copiaba la matriz entera para consolidar los
  bloques float: 278 MB en unos 255 ms (29% del método en B) y con las dos copias vivas a la
  vez, el doble de memoria pico.

**Solución.** Un único array `float64` reservado al principio con columnas contiguas
(`order='F'`, el layout de un bloque de pandas): lags, window features, nivel y exógenas
`float64`. El bucle escribe cada serie en su tramo de filas, las exógenas float se copian
columna a columna y el DataFrame se crea sin copia. Las columnas de otros tipos (enteras,
`category`, el nivel de `'ordinal_category'`) se insertan después en su posición, cada una
como un bloque propio.

El ensamblado anterior con `pd.concat` se conserva en tres casos, cada uno por un motivo
medido:

- **Más columnas insertadas que columnas en el bloque.** `insert` copia cada columna que
  inserta y el `pd.concat` no; con 3 lags y 99 exógenas enteras el bloque era 1.39 veces más
  lento. El punto de equilibrio está entre 2 y 3.5 columnas insertadas por columna float; la
  regla elegida (como mucho una) renuncia a ganancias del 5% o menos.
- **100 o más columnas insertadas.** pandas emite un `PerformanceWarning` de fragmentación
  en la inserción número 100.
- **`encoding=None` sin exógenas ni calendario.** No hay ganancia (el `fit` elimina la
  columna del nivel y esa operación ya devuelve un bloque) y el array que recibe el estimador
  pasaría de contiguo por filas a contiguo por columnas, lo que cambia los coeficientes de
  `LinearRegression` en el último bit.

**Efecto** (código real, frente al commit anterior): `_create_train_X_y` 0.91 / 0.71 / 0.82
(A / B / C, nuevo / anterior), `fit()` 0.85 / 0.91 / 0.94, pico de memoria de
`create_train_X_y` 286 a 260 MB, 954 a 500 MB y 925 a 471 MB.

### 2.4 `'onehot'` y calendario en el bloque (`cdd51597f`, `641010ae8`)

- **`'onehot'`.** Antes la matriz de columnas por serie se creaba aparte como `int64`
  (`np.eye(n_series)[códigos]`, filas x series) y se unía al resto con `pd.concat`. El
  DataFrame resultante mezclaba bloques int y float, así que el estimador tenía que
  convertirlo a un array float nuevo en `fit` y otra vez en `predict(X_train)`. Ahora el
  bloque se reserva con ceros y una sola escritura pone los unos. Con 300 series,
  `_create_train_X_y` pasa de 3.02 a 0.79 s y el `fit()` de 9.14 a 3.39 s. Las columnas pasan
  a ser `float64`, como en `create_predict_X`.
- **`'onehot'` como vista** (después de `b6e4b4d0d`). Con las columnas `'onehot'` ya en el
  bloque, `predict(X_train)` dejó de copiar la matriz, y el pico de memoria del `fit()` pasó a
  ser la copia que hacía `_get_level_row_slices` al seleccionarlas por nombre (filas x series:
  1.36 GB y 418 ms con 300 series). Ahora se leen como un tramo de columnas, una vista del
  bloque (25 ms, 5 MB), y si no son contiguas o están en otro orden (una matriz reordenada por
  el usuario) se seleccionan por nombre como antes. `X_train_series_names_in_` usa los mismos
  tramos en lugar de sumar una columna por serie (191 ms). Medido en el mismo proceso contra
  `b6e4b4d0d`: `_create_train_X_y` 0.78 / 0.72, `fit()` 0.89 / 0.86, pico de memoria del
  `fit()` de 2,845 a 1,675 MB (2,958 MB en `082aa0966`). Huellas de 4.1 idénticas.
- **`calendar_features`.** Antes las variables se calculaban por fecha única, se expandían
  con `reindex` a todas las filas (un DataFrame más, con columnas enteras y float) y se unían
  con `pd.concat`. Ahora se calculan una vez por fecha única (`train_index.factorize()`) y
  se escriben en las últimas columnas del bloque como `float64`, como ya hacen
  `ForecasterRecursive`, `ForecasterDirect` y `ForecasterDirectMultiVariate`. Con el
  calendario por defecto (20 columnas), `_create_train_X_y` pasa de 1.23 a 0.63 s, el `fit()`
  de 4.30 a 3.44 s y el pico de memoria de 1,095 a 430 MB.

### 2.5 Errores corregidos por el camino

Los cuatro tienen test y entrada en la sección Fixed de `docs/releases/releases.md`.

- **`predict` con `'onehot'`** (desde la 0.22.0). El entrenamiento ordena las columnas de
  serie alfabéticamente (`encoding_mapping_`), pero la predicción ponía el 1 según el orden
  de entrada: con series en orden no alfabético, cada serie se predecía con la columna de
  otra. Además, si una serie perdía todas sus filas por NaN, la matriz de predicción tenía
  una columna menos y `predict` fallaba. Lo arregla `_encode_levels_onehot`, usado por
  `predict`, los métodos probabilísticos y `create_predict_X`. Ningún test lo detectaba
  porque todas las fixtures usaban nombres en orden alfabético.
- **`set_in_sample_residuals`** no convertía `y_train` a numpy: con `RangeIndex` y más de
  10,000 residuos lanzaba `KeyError` (con `DatetimeIndex`, un `FutureWarning`).
- **`encoding_mapping_`** se actualizaba en lugar de reconstruirse: en las búsquedas con
  `OneStepAheadFold` sobre un forecaster ya entrenado con otras series, las filas podían
  asignarse a la serie equivocada.
- **`select_features_multiseries` con `'onehot'`**: la columna de una serie sin filas en
  `X_train` llegaba al selector como si fuera una exógena.

## 3. Datos

### 3.1 Método

- **Entorno:** Windows 11, Intel Core Ultra 9 185H, Python 3.13.14, numpy 2.4.6, pandas
  2.3.3, scikit-learn 1.7.2, LightGBM 4.7.0 (`n_jobs=4`).
- **A/B en el mismo proceso.** Entre procesos el `fit()` varía un 10 a 20% en esta máquina
  (turbo, fallos de página), así que antes y después se ejecutan en el mismo proceso,
  alternados y cambiando el orden en cada ronda, tras una ejecución de calentamiento
  (`common.ab_interleaved`). Se dan mediana y mínimo.
- **El "antes" es el código real anterior**, no una réplica: el módulo se lee con
  `git show <commit>:skforecast/recursive/_forecaster_recursive_multiseries.py` y se ejecuta
  en su propio espacio de nombres con `__package__ = "skforecast.recursive"`, sobre las
  utilidades actuales (válido porque nada más usado por `fit` ha cambiado; el script lo
  comprueba).
- **Identidad antes de cronometrar.** Cada caso comprueba primero que las dos versiones dan
  el mismo `X_train`, `y_train`, predicciones, `X_train_series_names_in_` y
  `binner_intervals_`.
- **Ruido.** Con el mismo código en los dos lados, el `fit()` dio ratios de 0.88 a 1.14 en
  tandas de 5 repeticiones; los tiempos de `_create_train_X_y` son estables (0.97 a 1.02).
  Por eso el `fit()` no distingue efectos menores de un 5 a 10%, y donde la ganancia es
  pequeña lo que demuestra la mejora es el tiempo del componente.
- **Memoria:** pico de `create_train_X_y` con `tracemalloc`.

### 3.2 Resultado final

`dev/profiling_multiseries_fit/13_final_ab.py`, resultados en `results/final_ab.json`.
Tiempos mediana / mínimo; "después / antes" menor que 1 es mejora.

| Caso | `_create_train_X_y` antes | después | después / antes | `fit()` antes | después | después / antes | Pico de `create_train_X_y` |
|---|---|---|---|---|---|---|---|
| A: sin exógenas | 0.49 / 0.48 s | 0.46 / 0.44 s | 0.93 / 0.91 | 2.69 / 2.55 s | 2.09 / 1.99 s | 0.78 / 0.78 | 286 a 260 MB |
| B: 10 exógenas float | 0.86 / 0.85 s | 0.60 / 0.59 s | 0.70 / 0.70 | 3.22 / 3.18 s | 2.79 / 2.68 s | 0.87 / 0.84 | 953 a 500 MB |
| C: 5 float y 5 category | 1.77 / 1.67 s | 1.45 / 1.40 s | 0.82 / 0.84 | 4.03 / 4.00 s | 3.63 / 3.51 s | 0.90 / 0.88 | 925 a 471 MB |
| A, `encoding=None` | 0.57 / 0.52 s | 0.55 / 0.52 s | 0.97 / 1.00 | 2.34 / 2.27 s | 2.46 / 2.39 s | 1.05 / 1.05 | 414 a 414 MB |
| A, `calendar_features` por defecto (20 columnas) | 1.20 / 1.16 s | 0.60 / 0.58 s | 0.50 / 0.50 | 4.23 / 4.18 s | 3.44 / 3.13 s | 0.81 / 0.75 | 1,095 a 430 MB |
| A, `encoding='onehot'`, 300 series | 2.97 / 2.88 s | 0.77 / 0.75 s | 0.26 / 0.26 | 9.80 / 9.48 s | 3.34 / 3.19 s | 0.34 / 0.34 | 1,854 a 1,675 MB |
| A, `series_weights` (250 de 500 series) | como A | | | 33.08 / 31.95 s | 2.40 / 2.36 s | 0.07 / 0.07 | |
| A, LightGBM de 100 árboles | como A | | | 6.58 / 6.52 s | 5.70 / 5.59 s | 0.87 / 0.86 | |
| A, `Ridge` | como A | | | 2.05 / 1.99 s | 1.09 / 1.04 s | 0.53 / 0.53 | |

7 repeticiones de `_create_train_X_y` y 5 de `fit()` (3 con `'onehot'` y `series_weights`).

- **`encoding=None` sin exógenas** ejecuta el mismo código en los dos lados (mismo
  ensamblado, sin reparto de residuos por serie), así que no debe cambiar. El 1.05 de esta
  tanda es ruido: repetido con 11 repeticiones dio 0.98 / 0.92 y 1.01 / 1.09, y el mismo
  código en los dos lados, 0.99 / 0.95.
- **C:** el `fit()` (0.90 / 0.88) está en el borde del ruido; la ganancia del componente
  (0.82 / 0.84) sí es estable.
- **`'onehot'`:** la memoria pico solo baja un 10% porque la propia matriz de columnas por
  serie (filas x series) domina. La fila es anterior a la lectura de esas columnas como vista
  (2.4), que por sí sola da 0.89 en el `fit()`. No se repitió contra `082aa0966`: ese día el
  lado "antes" dio tiempos erráticos (`fit()` de 15 a 19 s frente a los 9.8 s de la tabla).

### 3.3 Medidas por cambio

Cada cambio se midió al implementarlo contra el commit anterior, con el mismo método
(números de las secciones 2.1 a 2.4). Las previsiones de los prototipos del estudio, que
están en la versión anterior de este archivo, no se repiten aquí: se confirmaron en A, se
quedaron cortas en B (`_create_train_X_y` x1.40 frente a x1.77 previsto) y en C la ganancia
del `fit()` quedó dentro del ruido.

## 4. Cómo se ha comprobado que los resultados no cambian

### 4.1 Identidad

- **Huellas de referencia** (`dev/profiling_multiseries_fit/11_snapshot_outputs.py`): 19
  configuraciones (escenarios A, B y C, las cuatro codificaciones, NaN con y sin
  `dropna_from_series`, longitudes distintas, `transformer_series`, `differentiation`,
  `series_weights` y `weight_func`). Guarda el sha1 de cada columna o array de `X_train`,
  `y_train`, `predict`, `predict_interval`, `predict_bootstrapping`, `binner_intervals_`,
  residuos in-sample, pesos y la partición de `OneStepAheadFold`. Se generaron sobre el
  código base (dos pasadas idénticas: deterministas) y `--check` dio las 19 idénticas tras
  cada commit. Las 3 configuraciones `'onehot'` se regeneraron en `cdd51597f` porque cambia
  el dtype de sus columnas de serie.
- **Comparación con el módulo anterior en configuraciones pequeñas** (el método de 3.1, sin
  cronometrar):
  - bloque único: 2,040 configuraciones (las cuatro codificaciones; exógenas float, int,
    float32, bool, object, category e `Int64`; exógenas ausentes en una serie o columna;
    `transformer_exog`; `categorical_features`; calendario; window features con y sin lags;
    `differentiation`; `transformer_series`; NaN con los dos valores de
    `dropna_from_series`), más 240 con índices con nombre;
  - `'onehot'`: 15; calendario: 54 (con `LinearRegression` y XGBoost);
  - revisión final: 160 configuraciones contra `082aa0966`, incluyendo `fit`, `predict`,
    `create_predict_X` y residuos.

  Todas idénticas salvo los cambios documentados (dtypes y nombre del índice).
- **El A/B final** comprueba la identidad a tamaño real (500 series) en cada caso antes de
  cronometrar.

### 4.2 Tests

Los valores esperados de los tests existentes no cambiaron, salvo el dtype de las columnas
`'onehot'` y de `weekend` en el calendario, y un valor erróneo que ocultaba una aserción que
siempre pasaba (al final de esta sección). Tests nuevos, en
`skforecast/recursive/tests/tests_forecaster_recursive_multiseries/` salvo el último:

| Archivo | Qué fija |
|---|---|
| `test_create_train_X_y.py` | orden de columnas y dtypes con exógenas mezcladas (float, int, category); NaN en lags y en una exógena category con `dropna_from_series`; exógena `object` distinta por serie; layout en memoria (`np.shares_memory`, `strides`, `ctypes.data`) por codificación, con exógenas no float y con calendario; los dos lados de cada límite del camino `pd.concat` (3 y 4 columnas, 99 y 100) sin `PerformanceWarning`; calendario igual por los dos caminos y con el mismo dtype que `create_predict_X`; nombre del índice; error de nombre duplicado; error de longitud antes que el de exógenas; `X_train_series_names_in_` con series desordenadas y una descartada |
| `test_get_level_row_slices.py` | tramos con series desordenadas, de distinta longitud y una descartada; `'onehot'` con columnas int y float, no contiguas o desordenadas; matriz vacía; `ValueError` si no son contiguas |
| `test_fit.py`, `test_set_in_sample_residuals.py` | residuos por serie iguales al cálculo con máscaras escrito en el test; `set_in_sample_residuals` igual que `fit` y con más de 10,000 residuos |
| `test_create_sample_weights.py` | `series_weights` y `weight_func` con series desordenadas, de distinta longitud y una descartada |
| `test_predict.py`, `test_predict_bootstrapping.py`, `test_create_predict_X.py` | `'onehot'` con series en orden no alfabético, con una serie sin filas y con un nivel desconocido |
| `test_train_test_split_one_step_ahead.py` | forecaster ya entrenado con otras series |
| `skforecast/feature_selection/.../test_select_features_multiseries.py` | `'onehot'` con una serie sin filas |

Además se corrigieron 8 aserciones `np.all(<generador>)` que siempre pasaban (`test_fit.py`,
`test_binning_in_sample_residuals.py`); una ocultaba un valor esperado erróneo.

En la revisión final pasaron, en secuencia, 714 tests (carpeta del forecaster,
`feature_selection` y `check_preprocess_exog_multiseries`) y 194 de `model_selection`
(backtesting y búsquedas multiserie).

## 5. Descartado

| Idea | Motivo |
|---|---|
| Window features por lotes (`rolling` sobre un DataFrame ancho) | pandas tarda lo mismo por columna que por serie suelta: ahorra unos 0.1 s por `fit()` (5% con 25 árboles, menos del 1% con 500) por 1 a 2 días de trabajo y riesgo medio |
| `encoding=None` sin exógenas en el bloque | 35 ms menos y las predicciones de `Ridge` cambian en torno a 1e-11 |
| Columnas int y bool de las exógenas dentro del bloque float | cambiaría los dtypes de `X_train` y de `exog_dtypes_out_` |
| `pd.concat(copy=True)` o copy-on-write | la copia se desplaza a `estimator.fit` |
| Comprobaciones de NaN sobre numpy en lugar del DataFrame | sin ganancia con un bloque (15 frente a 16 ms) |

## 6. Limitaciones y pendientes

- **Casos límite aceptados** (detalle en `dev/multiseries_train_matrices.md`, sección 11):
  estimadores que modifican `X` en el sitio (`LinearRegression(copy_X=False)`) dan
  residuos in-sample erróneos o un `KeyError`; una exógena llamada `_level_skforecast` con
  `'ordinal_category'` da el error de pandas en lugar del de skforecast.
- **pandas 3** (copy-on-write por defecto; hoy `pandas<3.0`): el bloque depende de que
  `pd.DataFrame(X, copy=False)` no copie. Al levantar el pin, repetir los tests de layout.
- **Siguientes costes del `fit()`**, fuera de este trabajo: el estimador (45 a 55% con 25
  árboles), el codificador de categóricas de sklearn en el escenario C (prototipo en el stash
  "FastOrdinalEncoder categorical exog") y el ajuste de `transformer_series` por serie.
  `_train_test_split_one_step_ahead` (`OneStepAheadFold`) todavía selecciona por nombre las
  columnas `'onehot'` de `X_train` y `X_test` (código anterior a esta rama). `predict()` no se
  ha tocado.

## 7. Cómo reproducirlo

Desde la raíz del repositorio, con el intérprete del entorno y `PYTHONIOENCODING=utf-8`:

```powershell
# Identidad a tamaño real, tiempos y memoria, antes frente a después (unos 15 minutos)
python dev\profiling_multiseries_fit\13_final_ab.py --base 082aa0966
```

`--base` admite cualquier commit en el que solo haya cambiado el módulo del forecaster (el
script lo comprueba), así que sirve también para medir un cambio futuro contra el anterior.

Para un cambio futuro que deba dejar los resultados idénticos, las huellas se usan así:
`11_snapshot_outputs.py` antes de tocar el código (genera `results/snapshot/<caso>.json`) y
`11_snapshot_outputs.py --check` después. Las huellas dependen del entorno (versiones de
LightGBM, numpy y pandas) y no se versionan.

Los scripts del estudio original (`REPORT.md`, prototipos, presupuesto por etapa y
escalado) y los A/B de cada commit no están en el repositorio: el informe y los prototipos
se guardaron en un stash local del autor, y los A/B por commit se sustituyen por
`13_final_ab.py`, que usa el mismo método contra cualquier commit base.
