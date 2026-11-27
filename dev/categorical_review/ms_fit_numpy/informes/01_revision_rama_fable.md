# Revisión independiente de `fix/ms-fit-numpy-estimator` (cambios sin commit)

Carpeta de trabajo (las rutas de scripts y salidas son relativas a ella):
`<scratchpad>\indep_review`

- **Base**: `a7c7a869d` (merge base y HEAD de la rama), extraída con `git archive` en `base/`. No usé la punta de `origin/0.28.x` (`32d429277`) porque va un PR por delante (#1547, solo `datasets.py` y documentación) y mezclaría ese cambio.
- **Rama**: el árbol de trabajo, con `PYTHONPATH` al repositorio. Cada script imprime `skforecast.__file__`.
- **Entornos**: Python 3.13.18, numpy 2.5.5, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, pyarrow 26.0.1; pandas 3.1.1 (`skforecast_27_py13_p3`) y pandas 2.3.3 (`skforecast_27_py13`). Windows 11.
- **Repositorio**: solo lectura. `git status` al terminar es el mismo que al empezar (31 entradas; los 2 ficheros sin seguimiento son los de la rama), mismo HEAD, stash sin tocar. Ejecuté con `PYTHONDONTWRITEBYTECODE=1` y `-p no:cacheprovider`. Única excepción: el primer comando (versiones) no llevaba esa variable y pudo refrescar `__pycache__`, que git ignora. Ningún test dejó ficheros.
- Un comando mío se quedó esperando entrada estándar y lo paré; comprobé que no quedó ningún proceso huérfano.

## 1. Veredictos

1. **¿Es correcto?** Sí en lo que dice arreglar, y no encontré ninguna regresión con `categorical_features='auto'`; pero no está terminado: introduce dos fallos menores y una regresión de memoria (H1, H2, H5), y varios cambios de comportamiento están sin recoger o son discutibles (H3, H4, H6, H7).
2. **¿Está optimizado?** En tiempo no hay diferencia medible (ratios entre 0,92 y 1,04, dentro de la dispersión). En memoria es igual salvo con columnas no float: XGBoost gasta 78 MB más en `fit` (+53 % del pico de la operación) y las búsquedas one-step-ahead con LightGBM y Ridge gastan 65 MB menos. La copia que causa lo primero es evitable.
3. **¿Es una mejora para skforecast?**
   - (a) Solo por méritos: no lo fusionaría tal cual. Para arreglar el bug prefiero la alternativa pequeña (31 líneas frente a 543 netas), que da las mismas predicciones que la rama donde las dos funcionan y no cambia nada a quien no sufre el bug.
   - (b) Con el criterio del proyecto (numpy por dentro): sí lo fusionaría, con los cambios de la sección 5. Aviso: la rama cumple ese criterio solo en la frontera con el estimador, no en la construcción de la matriz.

## 2. Hallazgos por gravedad

Clases: (a) fallo introducido por la rama, (b) ya existía, (c) cambio no recogido en la nota de versión, (d) recogido pero discutible.

### H1. (a) Las matrices públicas ya no son lo que recibe el estimador cuando hay una `category` sin gestionar. Gravedad media. Ejecutado.

- Dónde: `skforecast/recursive/_forecaster_recursive_multiseries.py`, `fit` L2652 (`_train_matrix_to_estimator_input`), `create_train_X_y` L1789 y `create_predict_X` L3859 (`compat.astype(X_predict, self.exog_dtypes_out_)`).
- Qué pasa: con `categorical_features=None` (o lista incompleta) y una exog `category` de valores 10, 20, 30, `create_train_X_y` sigue devolviendo la columna como `category`, pero el estimador se ajustó con los valores. Quien pasa esa matriz al estimador (el flujo de la guía `training-and-prediction-matrices.ipynb`, o cualquier análisis con SHAP) obtiene predicciones distintas sin aviso: LightGBM lee los códigos.
- Script: `probes.py` (P02). LightGBM, `encoding='ordinal'`, `None`:
  - Base: `estimator.predict(create_train_X_y) == in-sample of fit: True`.
  - Rama: `estimator.predict(create_train_X_y) == in-sample of fit: False`.
  - Con `'auto'`: `True` en las dos.
- El docstring lo menciona de pasada; la nota de versión no. Es el mismo bug que se arregla, movido al usuario.
- Arreglo: convertir la columna a sus valores al crear la matriz (`_create_train_X_y`), no en la frontera. En el prototipo de la alternativa (`alt/`, P02) sale `True` y `X_train['x_c']` es `float64`.

### H2. (a) Texto con aspecto numérico y números complejos pasan en silencio como números. Gravedad media-baja. Ejecutado.

- Dónde: `skforecast/utils/utils.py`, `_train_matrix_to_numpy`, L958 a L983. Solo se rechazan de antemano fechas, timedeltas y `category` no entera; el resto se prueba con `to_numpy(dtype=float)`, que convierte `'10'` en `10.0` y descarta la parte imaginaria.
- Script: `probes.py` (P03), LightGBM, columna fuera de la lista (`categorical_features=['x_k']`):
  - `object` de cadenas numéricas. Base: `ValueError: pandas dtypes must be int, float or bool. Fields with bad pandas dtypes: x_c: object`. Rama: `fit=ok`, sin ningún aviso.
  - dtype `str` de pandas 3. Base: el mismo error. Rama: `fit=ok`, sin aviso.
  - `complex128`. Base: error con LightGBM y `Complex data not supported` con Ridge. Rama: `fit=ok` en los dos.
- Contradice el docstring de la función ("A `TypeError` that names the columns is raised if any of them cannot be passed as numeric values: text..."). Con Ridge el texto numérico ya pasaba en la base (lo convierte scikit-learn): para los lineales es (b).
- Arreglo: rechazar por dtype los de texto (`str`, `string`, `object` de cadenas) y `kind == 'c'`. Falta el test.

### H3. (d) Con `None` y una `category` cuyos valores son sus códigos, el modelo cambia y puede empeorar. Ejecutado.

- La nota lo recoge ("the model changes, also with ... categories 0, 1, 2 ..., whose predictions were right"). Es el coste real para los usuarios: la receta histórica de categóricas nativas (exog `category` más `None`) daba predicciones correctas e idénticas a `'auto'`.
- Script: `probes.py` (P10), categorías 0..5 con efecto no monótono, MAE de 30 pasos, `auto` / `None`:
  - Base: LightGBM 10,38 / 10,38; XGBoost 3,59 / 3,59; HGB 10,38 / 10,38 (idénticos); CatBoost 2,30 / error.
  - Rama: LightGBM 10,38 / 15,26; XGBoost 3,59 / 9,63; HGB 10,38 / 15,26; CatBoost 2,30 / 6,95.
- Lo que la nota afirma se cumple: `'auto'` en la rama reproduce el modelo anterior. Pero el usuario solo ve un `DataTypeWarning`; quien silencia avisos pasa de un modelo bueno a uno peor al actualizar. Es consecuencia de la decisión de tratarla como numérica, no de implementarla con numpy: la alternativa da los mismos números.

### H4. (d) Una `category` de decimales o booleanos fuera de la lista pasa a dar `TypeError` donde la base funcionaba bien. Ejecutado.

- Dónde: `utils.py` L958 a L965 y L987.
- Script: `matrix.py`, pandas 3: 36 configuraciones `ok -> err` (dtype `cat_float`, lista incompleta; Ridge, LinearRegression, RandomForest, HGB, `Pipeline`, `TransformedTargetRegressor` y LightGBM). Con Ridge, RandomForest y `Pipeline` la base ajustaba y predecía con los valores, bien. Solo era incorrecta con LightGBM y XGBoost.
- La nota lo recoge ("Its categories must be integers"). Es coherente con `predict`, pero es una restricción nueva para un caso que funcionaba.

### H5. (a) XGBoost con columnas no float usa 78 MB más en `fit`. Ejecutado.

- Dónde: `utils.py` L969 (`X_train.to_numpy(dtype=float, na_value=np.nan)`), provocado por `level_in_block = self.encoding != 'ordinal_category'` (`_forecaster_recursive_multiseries.py` L1186) y por las exog `int` y `bool` que se insertan como bloques aparte.
- Scripts: `perf_mem.py` y `perf_convert.py`; números en la sección 4. La base daba a XGBoost el DataFrame y este lo leía por columnas; la rama crea antes un array float64 completo.

### H6. (c) Seguir la guía con un estimador de scikit-learn da un aviso nuevo. Ejecutado.

- Script: `probes2.py` (P11): `forecaster.estimator.predict(X_train)` con la matriz de `create_train_X_y`.
  - Base: sin avisos con Ridge, HGB, LightGBM y XGBoost.
  - Rama: `UserWarning: X has feature names, but Ridge was fitted without feature names` (igual con HGB, y con `OrdinalEncoder` si `ordinal_category`). LightGBM y XGBoost, sin aviso.
- La guía `docs/user_guides/training-and-prediction-matrices.ipynb` usa LightGBM, así que sus salidas no cambian, pero no se ha tocado y no dice nada. La nota solo habla de `feature_names_in_`.
- Efecto positivo no recogido: `estimator.predict(create_predict_X(...))` con `ordinal_category` fallaba en la base con LightGBM (`train and valid dataset categorical_feature do not match`) y con XGBoost, y en la rama funciona.

### H7. (c) Imprecisiones y omisiones de la nota de versión. Ejecutado.

- "fit raises a `TypeError` that names the column": con `None` no la nombra. El error sale antes, de `check_exog_dtypes` (`utils.py` L1644): `Categorical dtypes in exog must contain only integer values. In the forecasters with the argument...`. Solo la nombra con una lista. Salida en `logs/errors_matrix_p3.txt`.
- "Results with `categorical_features='auto'` (the default) are unchanged" convive con la excepción de `TransformedTargetRegressor` y `Pipeline` con `ordinal_category`, que está en otro párrafo. Conviene unirlas.
- Mejoras que la rama trae y la nota no cuenta (todas `err -> ok` en `matrix.py` o `probes.py`): exog con dtypes de pyarrow y LightGBM (la base falla con `pandas dtypes must be int, float or bool`); nullable con NA y CatBoost; CatBoost con `None` y una `category` entera, o con `ordinal_category`; diccionario con una serie sin exog y lista de categóricas (P08); `cat_features` por nombre en `fit_kwargs` de CatBoost.
- Forecasters guardados: la nota dice que hay que reajustarlos y es cierto, pero no hay guarda. `save_load.py`: guardado con la base (`None`, 10/20/30, LightGBM y XGBoost) y cargado con la rama, `predict` da lo mismo que antes y `set_in_sample_residuals` da residuos distintos de los del ajuste (diferencia máxima 24,7 y 47,1) sin aviso.

### H8. Diseño y código: partes más complicadas de lo necesario. Deducido del código y, donde se indica, ejecutado.

- `_resolve_user_categorical_config` (`utils.py` L1054 a L1245, 190 líneas) reimplementa las convenciones de cuatro librerías. Parte de lo que soporta no lo soporta ni la librería: nombres en el constructor de LightGBM, sus cinco alias y los formatos `'0,1,2'` y `'name:c1,c2'` (L1142 a L1181). Ejecutado (`user_cat_config.py`, exog entera): en la base `LGBMRegressor(categorical_feature=['x_c'])` falla con `LightGBMError: categorical_feature is not a number`, así que nadie lo usa hoy. Es superficie nueva que mantener, no compatibilidad que conservar.
- Tres mecanismos distintos para los parámetros categóricos que viven en el estimador: `configure_estimator_categorical_features` los escribe y los deja (con `'auto'`); la rama los pone y los restaura (con `None`); y las búsquedas bayesianas tienen `_get/_restore_estimator_categorical_set_params`. Ejecutado (P06): con `'auto'`, un error dentro de `estimator.fit` deja `feature_types` y `categorical_features` modificados, igual que en la base; con `None` se restauran.
- El bloque poner, ajustar y restaurar está duplicado: `_forecaster_recursive_multiseries.py` L2671 a L2689 y `model_selection/_utils.py` L3052 a L3070.
- El split de multiserie devuelve 9 elementos y el de `ForecasterDirectMultiVariate` 8, y se iguala con `split = (*split, None)` (`_utils.py` L1134). `feature_names` es obligatorio para un forecaster e ignorado para el otro.
- `make_writeable` (`utils.py` L943 a L953) vuelve a poner como escribible la vista que pandas 3 marca de solo lectura. Funciona y tiene salida segura, pero se apoya en un detalle interno de pandas para ahorrar una copia.
- `_train_matrix_to_estimator_input` devuelve un DataFrame para CatBoost con categóricas (L1040 a L1051): "numpy en `fit`" no es universal.
- Convenciones: `ruff check` sin errores en los ficheros cambiados (salvo F401 ya existentes en `utils/__init__.py`); sin guiones largos en las líneas añadidas; las dos líneas de más de 88 caracteres son código movido. `test_train_matrix_to_estimator_input.py` prueba dos funciones en un fichero. No encontré código muerto.

### H9. Tests nuevos y modificados. Ejecutado.

- Tests de la rama contra el código de la base (`base_newtests/`): 87 fallan y 280 pasan. Fallan por la razón correcta los de API pública: `test_predict_output_when_category_exog_not_in_categorical_features` (2 casos), `test_fit_estimator_receives_float_numpy_array` (4), `test_fit_DataTypeWarning...` (2), `test_fit_TypeError...` (3), `test_fit_adds_level_to_user_categorical_config...` (5), `test_fit_and_predict_when_names_have_characters_not_allowed...` (2) y otros. Muchos de `test_predict_and_calculate_metrics_one_step_ahead_multiseries.py` fallan solo porque cambia la firma privada (9 elementos), así que no demuestran nada sobre la base.
- Tests que no fallan sin el cambio:
  - `test_set_in_sample_residuals_same_as_fit_when_category_exog_not_in_categorical_features[categorical_features: ['exog_cat_2']]` pasa en la base: el efecto de la categoría es monótono, y categórica y numérica dan los mismos árboles. No prueba el caso de la lista.
  - `test_fit_when_CatBoost_cat_features_in_constructor_include_level` pasa en la base y solo comprueba `notna()`.
- Casos importantes sin test: H1, H2, búsquedas por la API pública (rejilla, aleatoria, bayesiana) con `None` y una `category`, dtypes de pyarrow en `fit`, `save_forecaster` y `load_forecaster`, y un caso con más de 3 a 5 categorías no monótonas (con pocas, categórica y numérica coinciden por casualidad, como me pasó en `matrix.py`).

### H10. (b) Ya existían, no cambian con la rama. Ejecutado.

- Con `'auto'`, XGBoost y HGB dan `IgnoredArgumentWarning` en el segundo `fit` y tras `set_lags` (CAT-T4 del roadmap). P06, igual en base y rama.
- `CatBoostRegressor(cat_features=[...])` con lista no se puede clonar (`RuntimeError: Cannot clone object`); con tupla sí.
- Configuración por posición del usuario tras `set_lags`: queda desfasada (error de XGBoost y HGB, o un lag como categórico). Es inherente a configurar por posición; por nombre la rama lo resuelve.

## 3. Áreas sin fallos: qué ejecuté exactamente

- **Matriz de comportamiento** (`matrix.py`, `compare_matrix.py`, `logs/compare_matrix_p3.txt`, `logs/compare_matrix_p2.txt`): 11 estimadores (LightGBM, XGBoost, CatBoost, HGB, Ridge, LinearRegression, RandomForest, `Pipeline` con Ridge y con LightGBM, `TransformedTargetRegressor` con Ridge y con LightGBM) x 4 codificaciones x 4 modos de `categorical_features` x 16 dtypes de exog (float, int, bool, nullable con y sin NA, pyarrow con y sin NA, `category` de enteros iguales a sus códigos, 10/20/30, declarada en otro orden, decimales y texto, object, str, fechas, mezcla) x series como DataFrame y como dict con longitudes distintas y NaN. 3520 configuraciones con pandas 3 y 1600 con pandas 2 (5 estimadores). En cada una: `fit`, `predict`, residuos in-sample, intervalos bootstrapping y conformal, importancias, backtesting de un paso sin refit y búsqueda one-step-ahead.
  - `'auto'` y lista completa, pandas 3: 1704 de 1760 idénticas bit a bit; 48 pasan de error a funcionar; 8 iguales hasta 1e-9 (`TransformedTargetRegressor` con LightGBM y `ordinal_category`). Ninguna distinta, ninguna deja de funcionar. Pandas 2: 768 idénticas y 32 de error a funcionar.
  - One-step-ahead frente a backtesting, pandas 3: en la base, 84 configuraciones con métricas distintas y 12 donde la búsqueda funciona y el backtesting falla. En la rama, 0 y 0 (2292 iguales; el resto falla en las dos vías). Pandas 2: base 52 y 15; rama 0 y 0.
  - En la rama no hay ningún caso que ajuste y luego falle en `predict` o en los intervalos (en la base, 36).
- **Lo que recibe el estimador** (`spy_consistency.py`, estimador que graba sus entradas): en la rama, 236 de 236 configuraciones reciben un `ndarray` float64 en `fit` y la fila de `predict` es idéntica a la de entrenamiento. En la base, 36 difieren al leer los códigos. Igual con pandas 2.
- **Categóricas configuradas por el usuario** (`user_cat_config.py`, `user_cat_config_tuple.py`): 17 formas (LightGBM por `fit_kwargs` y constructor, por posición, nombre y cadena; XGBoost `feature_types`; CatBoost por `fit_kwargs` y constructor; HGB por posición, nombre y máscara) x 4 codificaciones x 3 exog (int, `category` 10/20/30, `category` en otro orden). Base: 70 con métricas distintas entre búsqueda y backtesting. Rama: 0 distintas y 180 iguales; segundo `fit` idéntico; parámetros del estimador restaurados; `set_lags` funciona con nombres. Las 24 restantes son el constructor de CatBoost con lista (H10); con tupla funcionan, salvo con `ordinal_category`, que da el `ValueError` documentado.
- **Estado** (`probes.py` P06, `probes2.py` P18): excepción dentro de `estimator.fit`, segundo `fit`, `set_lags`, `set_params`, `deepcopy`. Con `None`, los parámetros del usuario quedan intactos.
- **Búsquedas** (`searches.py`): rejilla con `lags_grid`, aleatoria y bayesiana con lags en el espacio, 5 estimadores, 4 codificaciones, DataFrame y dict, y `ForecasterDirectMultiVariate`. En la rama, 656 candidatos de rejilla comparados uno a uno con un backtesting de un paso sin refit: 656 iguales (base: 128 distintos). Con `'auto'`, tablas y predicciones del mejor modelo idénticas a la base. Con pandas 2 (LightGBM y XGBoost), 288 de 288.
- **Resto de funciones con `'auto'`** (`features_equiv.py`): pesos (`weight_func`, `series_weights`), diferenciación, `transformer_series` y `transformer_exog`, one-hot de la exog, window features, `dropna_from_series`, backtesting con `refit` False, True y 2, intervalos, `predict_quantiles`, `set_in_sample_residuals`. 1080 configuraciones (5 estimadores), todas idénticas a la base. Pandas 2: 432 (LightGBM y Ridge), idénticas.
- **`set_in_sample_residuals` con `None`** (P19): igual que los residuos de `fit` en 20 de 20.
- **Backtesting con refit en paralelo** (P13): `n_jobs=1` y `n_jobs=2` dan las mismas predicciones.
- **`predict` con la exog en otro dtype que en `fit`** (P14: `int64`, float, `category` en otro orden o con menos categorías): mismas predicciones.
- **`save_forecaster` y `load_forecaster`** (`save_load.py`): 10 casos, rama a rama, idénticos. Guardado con la base y `'auto'`, cargado con la rama: idéntico.
- **Otros forecasters** (`others_unchanged.py`): `ForecasterRecursive`, `ForecasterDirect`, `ForecasterDirectMultiVariate` y `ForecasterRecursiveClassifier`, 333 configuraciones: todo idéntico salvo el texto del error de `check_exog_dtypes`. Igual con pandas 2.
- **Memoria compartida** (P04): la matriz que recibe el estimador no comparte memoria con las series ni la exog del usuario, y los datos del usuario quedan intactos también con estimadores que escriben en su entrada.
- `tools/ai/generate_ai_context_files.py --check`: "All generated files are up to date".

## 4. Rendimiento

Python 3.13.18, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1. Un hilo (`OMP_NUM_THREADS=1`, `n_jobs=1`), proceso fijado al núcleo 12 con prioridad alta. 300 series x 1000 observaciones, `lags=24`, 10 exog, 292.800 filas x 35 columnas. "float": todo float, `encoding='ordinal'`. "mixed": `ordinal_category` y exog con `int64`, `int32`, `bool` y dos `category`.

**Tiempo** (`perf_time.py`): base y rama cargadas en el mismo proceso, bloques alternados, 5 repeticiones tras un calentamiento. Mínimo en segundos, base / rama (ratio de mínimos). Solo pandas 3.

| Estimador | Datos | fit | predict (24 pasos) | backtesting (10 folds) | búsqueda one-step-ahead |
|---|---|---|---|---|---|
| Ridge | float | 0,213 / 0,215 (1,01) | 0,006 / 0,006 | 0,731 / 0,718 (0,98) | 1,106 / 1,028 (0,93) |
| Ridge | mixed | 0,331 / 0,305 (0,92) | 0,009 / 0,008 | 0,880 / 0,888 (1,01) | 1,248 / 1,165 (0,93) |
| LightGBM | float | 2,728 / 2,669 (0,98) | 0,016 / 0,016 | 2,729 / 2,737 (1,00) | 4,557 / 4,462 (0,98) |
| LightGBM | mixed | 2,820 / 2,873 (1,02) | 0,022 / 0,020 | 2,908 / 2,842 (0,98) | 4,715 / 4,634 (0,98) |
| HGB | float | 2,083 / 2,168 (1,04) | 0,016 / 0,016 | 2,488 / 2,399 (0,96) | 3,729 / 3,831 (1,03) |
| HGB (200 series) | mixed | 1,714 / 1,747 (1,02) | 0,038 / 0,040 | 1,925 / 1,864 (0,97) | 2,626 / 2,651 (1,01) |
| XGBoost | float | 2,588 / 2,661 (1,03) | 0,016 / 0,016 | 2,714 / 2,747 (1,01) | 4,362 / 4,495 (1,03) |
| XGBoost | mixed | 3,326 / 3,223 (0,97) | 0,021 / 0,023 | 3,367 / 3,347 (0,99) | 5,099 / 4,908 (0,96) |
| CatBoost (3 rep.) | mixed | 5,745 / 5,832 (1,02) | 0,043 / 0,041 | 6,053 / 6,108 (1,01) | 9,902 / 9,833 (0,99) |

La dispersión entre mínimo y máximo de cada celda es del 3 al 10 %, y mayor en el `fit` de Ridge, que dura 0,2 s (hasta el 56 %). Ninguna diferencia sale de ese ruido. La única que se repite es Ridge con datos mixtos, algo más rápido en la rama (0,92 en `fit`, 0,93 en la búsqueda). HGB "mixed" con 300 series falla en las dos por el límite de 255 categorías de HGB; lo repetí con 200. Los `predict` duran de 6 a 43 ms y su ratio no es fiable.

**Conversión de la matriz** (`perf_convert.py`, solo rama, 292.800 x 35): con todo float y `ordinal` u `onehot` es una vista, 0,02 ms y 0 MB. Con `ordinal_category`, o con una sola exog `int` o `bool`, es un array nuevo de 78,2 MB y de 15 a 18 ms, frente a 78 a 86 ms de `create_train_X_y`.

**Memoria** (`perf_mem.py`, `mem_summary.py`): un proceso nuevo por medida, pico del conjunto de trabajo menos RSS antes de la operación, MB, dos ejecuciones por árbol alternadas.

| Estimador | Datos | Operación | Base | Rama | Diferencia |
|---|---|---|---|---|---|
| XGBoost | mixed | fit | 148,1 / 148,4 | 226,2 / 226,0 | **+77,8** |
| XGBoost | mixed | búsqueda | 157,7 / 155,7 | 170,0 / 171,6 | +14,1 |
| LightGBM | mixed | búsqueda | 259,6 / 259,8 | 194,9 / 193,8 | **-65,4** |
| Ridge | mixed | búsqueda | 244,2 / 246,3 | 177,6 / 179,0 | **-66,9** |
| LightGBM | mixed | fit | 255,8 / 256,2 | 254,9 / 256,2 | -0,4 |
| Ridge | mixed | fit | 246,0 / 245,4 | 245,3 / 245,3 | -0,4 |
| CatBoost | mixed | fit | 295,1 / 294,8 | 298,1 / 300,0 | +4,1 |
| HGB | mixed | búsqueda | 314,9 / 313,3 | 316,2 / 317,6 | +2,8 |
| Los 4 | float | fit y búsqueda | | | entre -1,6 y +0,4 |

**Trabajo evitable en el código nuevo**

1. La copia de 78 MB con `ordinal_category`: la columna del nivel se crea como `pd.Categorical`, se inserta como bloque aparte y luego se convierte toda la matriz a float. Si el nivel fuera en el bloque float (y la `category` solo en la matriz pública) no habría copia ni los 78 MB de XGBoost. Lo mismo para las exog `int` y `bool`.
2. `to_numpy(dtype=float, na_value=np.nan)` con `na_value` hace una pasada más (`isna` de toda la matriz) aunque no haya dtypes de extensión. Deducido del código de pandas; no lo medí aparte.
3. `_resolve_estimator_categorical_config` se llama en cada candidato de la búsqueda y con `'auto'` casi nunca hace nada. Coste despreciable frente al ajuste; es complejidad, no tiempo.

## 5. ¿Es una mejora? Diseño frente a la alternativa

**Qué medí de la alternativa.** La prototipé en `alt/` (copia de la base más `patch_alt.py`, 31 líneas en `_create_train_X_y`): al crear la matriz, una `category` entera que `categorical_features` no gestiona se sustituye por sus valores (float), con `DataTypeWarning`, y una no entera da `TypeError` con el nombre. El DataFrame sigue llegando al estimador. Misma matriz de 2240 configuraciones (7 estimadores, pandas 3) y mismos scripts:

- Arregla el bug igual: 0 discrepancias entre one-step-ahead y backtesting; P01 da los mismos MAE que la rama; `user_cat_config.py`, 0 discrepancias.
- Con `None` o lista incompleta, donde las dos ajustan (469 configuraciones) las predicciones son idénticas en 468 y difieren en 1e-9 en 1. En las 212 que tienen una `category` entera sin gestionar, 212 idénticas.
- Mantiene lo que la rama rompe: P02 da `True` (H1), el estimador conserva `feature_names_in_`, no hay avisos nuevos de scikit-learn (H6), no cambia el modelo de `TransformedTargetRegressor` ni `Pipeline` con `ordinal_category`, no cambia la comprobación in situ, y XGBoost no gasta 78 MB más.
- No da lo que la rama da de más: 83 configuraciones que en la base fallan y en la rama funcionan (pyarrow con LightGBM, nullable con NA y CatBoost, CatBoost con `None` y `ordinal_category`), los nombres con `[`, `]`, `<`, y `cat_features` por nombre en `fit_kwargs` de CatBoost. Son arreglos independientes del bug.
- H3 es igual en las dos: el modelo con categorías 0..n-1 cambia igual.

**Complejidad y mantenimiento.** Rama: 742 líneas añadidas y 199 borradas en el paquete (543 netas), más 1752 y 373 en tests. El grueso es `_resolve_user_categorical_config`, que existe solo porque un array no tiene nombres ni dtypes: hay que traducir a posiciones lo que LightGBM, XGBoost, CatBoost y HGB hacen solos con un DataFrame, y seguirlas cuando cambien. A favor de la rama: la garantía es estructural (el estimador no puede leer distinto en `fit` y en `predict`), mientras que la alternativa cierra el caso conocido y depende de que ninguna librería empiece a interpretar otro dtype del DataFrame.

**Coherencia.** La rama iguala multiserie con los demás forecasters, que ya ajustan con numpy. Es su mejor argumento.

**Coste para los usuarios con la rama**

- Pierden los nombres en el estimador: `feature_names_in_` desaparece, `feature_name_` de LightGBM pasa a `Column_0...`, `booster.feature_names` a `None`. `get_feature_importances` no cambia (P05).
- Cambian de modelo sin cambiar su código: `None` con `category` (H3, con aviso), y `TransformedTargetRegressor` o `Pipeline` con `ordinal_category` con el valor por defecto `'auto'` (sin aviso; en mi matriz de 3 series la diferencia es de 1e-9, el roadmap cita MAE 1,78 a 4,23).
- Dejan de funcionar: `category` de decimales fuera de la lista (H4); `LinearRegression(copy_X=False)` y `StandardScaler(copy=False)` con exog no float (P04: en la base ajustaban, en la rama `ValueError`). Un `Pipeline` que selecciona columnas por nombre falla en `fit`, pero en la base ya fallaba en `predict`, así que no se pierde nada real.

**Sobre el criterio "numpy por dentro".** En este caso se cumple a medias:

- `_assemble_train_X_y` sigue construyendo un DataFrame (L1284 a L1318), con el nivel como `pd.Categorical` (L1302) y las exog no float como bloques insertados. La conversión es un `DataFrame.to_numpy` al final, con copia.
- `_get_fit_config` necesita `X_train.dtypes` para decidir el aviso (L2467).
- `_train_test_split_one_step_ahead` crea dos DataFrames y los convierte (L2091 a L2100).
- CatBoost recibe otro DataFrame creado sobre el array (`utils.py` L1040).

La dependencia de pandas sigue donde pesa, en la construcción. Lo que se gana es que el estimador no ve pandas, y eso se paga con `_resolve_user_categorical_config`. Si el objetivo es numpy por dentro, el paso que lo cumple es construir el bloque float completo y derivar de él la matriz pública, y la rama no lo da.

**Qué haría**

- (a) Sin el criterio: no la fusionaría tal cual para la 0.28.0. El bug es P1 y silencioso, y la alternativa lo arregla con 31 líneas y sin tocar a quien no lo sufre. Dejaría la rama para la versión siguiente, como cambio de arquitectura con su propio aviso.
- (b) Con el criterio: la fusionaría con estos cambios.
  1. H1: convertir la `category` sin gestionar a sus valores en `_create_train_X_y` (las 31 líneas de la alternativa caben dentro de la rama), para que `create_train_X_y`, `create_predict_X` y `exog_dtypes_out_` digan lo que recibe el estimador.
  2. H2: rechazar texto y complejos en `_train_matrix_to_numpy`, con test.
  3. H5: nivel de `ordinal_category` y exog `int` y `bool` dentro del bloque float, para quitar la copia.
  4. Recortar `_resolve_user_categorical_config` a lo que ya funcionaba y cubren los tests (nombres en `fit_kwargs` de LightGBM y CatBoost, nombres de HGB, añadir el nivel) y quitar constructor, alias y cadenas de LightGBM.
  5. Un solo mecanismo para los parámetros que viven en el estimador: que `'auto'` también ponga y restaure. De paso arregla el estado tras un `fit` fallido y el aviso de CAT-T4.
  6. Nota de versión y guía: H6, H7 y una frase clara sobre H3.
  7. Tests de H9.

## 6. Lo que no he podido comprobar

- **Linux y macOS**: todo es Windows.
- **Suite completa**: no la lancé, por las reglas. Solo lo de la sección 7.
- **GPU** (`set_cpu_gpu_device`, XGBoost en `cuda`): no hay ruta de prueba aquí.
- **Versiones mínimas de dependencias**: solo las instaladas.
- **Tiempo y memoria con pandas 2**: medí solo pandas 3. La corrección sí está en los dos.
- **Datos reales y tamaños grandes**: todo es sintético; el máximo fue 292.800 filas. No reproduje el caso de 1.035.168 filas del roadmap.
- **Memoria de HGB en `fit` con datos mixtos**: la medida de 300 series falló por el límite de 255 categorías y no la repetí con 200.
- **`TransformedTargetRegressor` y `Pipeline` con `ordinal_category` y muchas series**: con 3 series el cambio es de 1e-9. No medí el caso del roadmap.
- **Búsqueda bayesiana con trials en hilos** (`kwargs_study_optimize={'n_jobs': 2}`): el forecaster se comparte entre hilos y el poner y restaurar parámetros no es atómico. La base ya comparte y modifica ese forecaster, así que no es nuevo. Deducido del código, sin ejecutar.
- **`predict` de XGBoost por la vía de `DMatrix`** (booster `gblinear` u otro dispositivo): usa `self.feature_types`, que tras el ajuste vuelve a ser el del usuario. Deducido, sin ejecutar.
- **Notebooks**: no ejecuté ninguno.
- **Mi matriz usa 3 categorías**: con tan pocas, categórica y numérica dan a menudo árboles idénticos, así que las cifras "idéntico" con `None` y categorías iguales a sus códigos no dicen que el modelo no cambie. P10, con 6 categorías, es el que lo mide.
- **La alternativa** es un prototipo de 31 líneas probado con mis scripts, no con la suite de tests del repositorio.

## 7. Comandos de test ejecutados

Desde `<repo>`, `python.exe -m pytest <rutas> -q -p no:cacheprovider`, en secuencia (logs en `logs/pytest_*.log`).

| Rutas | pandas 3.1.1 | pandas 2.3.3 |
|---|---|---|
| `skforecast/recursive/tests/tests_forecaster_recursive_multiseries` (carpeta) | 904 passed | 904 passed |
| 10 ficheros cambiados o nuevos: en `utils/tests/tests_utils/`, `test_resolve_user_categorical_config.py`, `test_train_matrix_to_estimator_input.py`, `test_check_in_place_fit.py`, `test_check_exog_dtypes.py`; en `model_selection/tests/tests_utils/`, `test_predict_and_calculate_metrics_one_step_ahead_multiseries.py`, `test_train_test_split_one_step_ahead_search.py`; y `test_create_train_X_y.py` de direct, direct_multivariate, recursive y recursive_classifier | 541 passed, 1 skipped | 542 passed |
| `skforecast/model_selection/tests/tests_search/*multiseries*.py` (4 ficheros) | 156 passed | 156 passed |
| Tests de la rama contra el código de la base (`base_newtests/`): `test_fit.py`, `test_predict.py`, `test_set_in_sample_residuals.py`, `test_create_predict_X.py` de multiserie, `test_predict_and_calculate_metrics_one_step_ahead_multiseries.py`, `test_check_in_place_fit.py` | 87 failed, 280 passed, 1 skipped (esperado, ver H9) | no ejecutado |

Además: `ruff check --no-fix --no-cache` sobre los ficheros cambiados, "All checks passed!" salvo `utils/__init__.py` (F401 ya existentes); y `python tools/ai/generate_ai_context_files.py --check`, correcto.

Scripts propios: `run_matrix.sh`, `run_batch2.sh`, `run_chain3.sh`, `run_chain4.sh` y `run_perf.sh` lanzan todo en secuencia; los resultados están en `out/*.pkl` y `logs/`.

