Modelo: Opus 5.5 (`claude-opus-5-5`).

# Revisión independiente de `fix/ms-fit-numpy-estimator` (cambios sin commit sobre `a7c7a869d`)

## Veredictos

1. **¿Es correcto?** Sí, en lo que promete. El fallo descrito está corregido (verificado con LightGBM, XGBoost, CatBoost, HGB y TTR). Con `'auto'` los resultados son idénticos a la base salvo en un caso, `TransformedTargetRegressor` con `ordinal_category`. Pero la rama introduce un modo de fallo nuevo: las categóricas que el usuario configura en el estimador reciben valores en lugar de códigos, y con XGBoost y etiquetas grandes `fit` se dispara en tiempo y memoria. Además, la nota de versión contiene una afirmación falsa.
2. **¿Está optimizado?** Es neutro. Tiempo dentro de ±4 % en fit, predict, backtesting y one-step-ahead, con un −6 % en el fit de Ridge con exog mixta. En memoria, +17 MB (tracemalloc) en el fit de XGBoost con exog mixta y −19 MB de pico en las búsquedas one-step-ahead.
3. **¿Es una mejora?** Sí en el fondo: unificar lo que recibe el estimador en `fit` y en `predict` cierra una clase entera de fallos, no solo el reportado. Pero no la fusionaría tal cual: hacen falta los cambios concretos de la sección "Pregunta 3".

Carpeta de trabajo (W): `<scratchpad>\indep_review_opus`.

Entorno: Python 3.13.18, numpy 2.5.5, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, pyarrow 26.0.1. pandas 3.1.1 y 2.3.3.

## Hallazgos, de más a menos grave

### 1. (a) Fallo introducido: categóricas configuradas por el usuario reciben valores en lugar de códigos (ejecutado)

**Dónde:** `utils.py:903-1052` (`_train_matrix_to_numpy` y `_train_matrix_to_estimator_input`) y `utils.py:1054` (`_resolve_user_categorical_config`).

**Caso:** `categorical_features=None` y una columna `category` que el usuario declara como categórica nativa (`fit_kwargs={'categorical_feature': ['ci']}`, `feature_types` de XGBoost, `categorical_features` de HGB). La rama pasa al estimador los valores de la columna como categorías.

**Comprobación:** relabeling de las categorías (0,1,2 → 10,20,30 / −5,0,7 / 10⁶). Un modelo categórico correcto debe ser invariante a esa reetiqueta.

| Caso | Base | Rama |
|---|---|---|
| LightGBM, 10,20,30 | no invariante, max\|d\| = 3,39 (el fallo que se corrige) | invariante |
| LightGBM, −5,0,7 | no invariante, max\|d\| = 3,36 | **no invariante, max\|d\| = 1,10**: LightGBM trata los negativos como faltantes, sin aviso |
| XGBoost, −5,0,7 | fit ok, predicción errónea (5,12) | **`XGBoostError: Invalid categorical value` en `fit`** |
| XGBoost, máximo 30.000 | fit 0,14 s | fit 1,12 s |
| XGBoost, máximo 300.000 | fit 0,12 s | fit 16,3 s |
| XGBoost, máximo 3.000.000 | fit 0,1 s | **más de 5 min, 3,3 GB de working set; lo maté** |
| HGB | invariante | invariante |

**Scripts:** `W\t8_user_cat_values.py` (salidas `t8_base.txt` y `t8b_branch.txt`), `W\t8b.py`, `W\t8c_xgb_large.py`.

**Valoración:** la base ya daba predicciones erróneas en este camino. Ahora el error silencioso pasa a ser un cuelgue, un fallo de memoria o un modelo degradado sin aviso. La nota de versión no lo recoge; solo aparece como fila de documentación en el roadmap (`DOC-categorical-index-value-range`).

**Arreglo propuesto:** pasar los códigos en `fit` y en `predict` para las columnas `category` que el estimador trata como categóricas, o rechazarlas con un error que pida incluirlas en `categorical_features`.

### 2. (d) Con `'auto'` cambia el modelo con `TransformedTargetRegressor` y `ordinal_category`, sin aviso (ejecutado)

**Script:** batería `W\battery.py`, comparación con `W\compare.py`. Salidas `cmp_p3.txt` y `cmp_p2.txt`.

**Resultado:** las 32 diferencias que hay con `'auto'` o con la lista completa son todas `ttr_lgbm` con `ordinal_category` (max\|diff\| de hasta 2,67), con pandas 3 y con pandas 2.

**Causa:** en la base, TTR pasaba el DataFrame a LightGBM y la columna `category` del identificador de serie se trataba como categórica. En la rama llega como numérica: el usuario pide `ordinal_category` y obtiene en silencio el comportamiento de `ordinal`.

**Contradicción en la nota:**
- La línea 103 de `releases.md` dice: "Results with `categorical_features='auto'` (the default) are unchanged". Es falso para este caso.
- La línea 55 (aviso "Before upgrading") sí lo menciona.

**Propuesta:** emitir un aviso cuando se pide `ordinal_category` con un metaestimador o un `Pipeline`, y corregir la línea 103.

### 3. (d) Se pierde el tratamiento categórico por dtype con `None` (ejecutado, recogido en la nota)

**Caso:** `None` (o una lista incompleta) con LightGBM (`categorical_feature='auto'` por defecto), HGB (`from_dtype`) o XGBoost (`enable_categorical=True` por defecto en la 3.x). Las columnas `category` pasan a ser numéricas, con un `DataTypeWarning`.

**Casos donde la base era correcta y el modelo cambia igualmente:**
- HGB siempre (`t1b_relabel.py`: la base es invariante a la reetiqueta).
- LightGBM y XGBoost cuando los códigos coinciden con los valores (exog `ci0` de la batería).

**Otros efectos:**
- El test `test_create_predict_X`, que documentaba el patrón `transformer_exog` → `category` + `None` + LightGBM nativo, se reescribió con `'auto'`.
- `test_predict_output_when_estimator_is_Pipeline_that_keeps_...` invierte su expectativa.

Es un cambio documentado, pero real y silencioso salvo por el aviso.

### 4. (c) Cambios que la nota de versión no recoge (ejecutado salvo donde se indica)

**Correcciones no anunciadas:**
- **Exog con dtypes pyarrow numéricos y LightGBM o TTR(LightGBM):** la base fallaba en `fit` incluso con `'auto'` (`pandas dtypes must be int, float or bool`, 32 configuraciones). La rama funciona.
- **CatBoost con una columna `category` no gestionada, o con `ordinal_category` y `None`:** la base daba `CatBoostError` (64 configuraciones). La rama ajusta.
- **CatBoost con `fit_kwargs={'cat_features': ['nombre']}`:** la base daba `TypeError: list indices must be integers` (`t3_state.py`).

**Retirada sin aviso:** `cast_catboost_categorical_columns_dataframe` era pública (sin guion bajo, exportada por `from .utils import *`, presente en v0.26.0 y v0.27.0). Se elimina sin nota ni deprecación.

**Imprecisiones de la nota:**
- El "Fixed" de la línea 142 dice que CatBoost daba predicciones erróneas. Con una columna no gestionada, la base fallaba en `fit`. Solo daba predicciones erróneas si el usuario configuraba la columna como categórica (deducido).
- TTR(LightGBM) también daba predicciones erróneas en la base (`t1b_base.txt`) y no se menciona.
- "Forecasters saved with a previous version must be fitted again" es más alarmista que la realidad: un forecaster guardado con la base se carga en la rama, predice igual y `set_in_sample_residuals` coincide (`t5_saveload.py`).

### 5. (c) Uso externo de `forecaster.estimator` con el DataFrame de `create_train_X_y` (patrón SHAP) (ejecutado, `t6_external_use.py`)

- **Avisos nuevos:** HGB y Ridge emiten el `UserWarning` de scikit-learn de "feature names" en cada llamada.
- **Desfase invertido con `None` y exog `category`:** LightGBM y XGBoost leen los códigos del DataFrame y discrepan de la predicción del forecaster (max\|d\| de 3,5 a 5,1). Es el desfase de la base al revés: antes fallaba `predict`, ahora falla la explicación del usuario.

### 6. Ruido y diseño menor (ejecutado, `t10_warn_bt.py` y `t7_consistency.py`)

- **`DataTypeWarning` de más:** se emite también con estimadores cuyo modelo no cambia (Ridge, RandomForest y pipelines con escalador ya usaban los valores). En un backtesting con refit sale una vez por fold (8 de 8).
- **Inconsistencia entre forecasters:** con la rama, `ForecasterRecursiveMultiSeries` es el único que acepta la configuración categórica por nombre con `None`. `ForecasterRecursive`, `ForecasterDirect` y `ForecasterDirectMultiVariate` fallan con nombres, en la base y en la rama.

### 7. Código (deducido del código)

- **Bloque duplicado:** el `try/finally` de `set_params`/restauración está copiado en `_forecaster_recursive_multiseries.py:2671-2689` y en `model_selection/_utils.py:3052`.
- **Tuplas de longitud distinta:** `_utils.py:1134` rellena con `None` el split de `ForecasterDirectMultiVariate` para igualar las 9 posiciones del de multiserie.
- **Reimplementación frágil:** `_resolve_user_categorical_config` (unas 150 líneas) reimplementa la sintaxis de parámetros de LightGBM (alias, `'name:...'`, `'auto'`, `''`), CatBoost, XGBoost y HGB. Depende de detalles de terceros.
- **Envoltorio fino:** `_resolve_estimator_categorical_config` es un envoltorio privado del forecaster que `model_selection` llama desde fuera del módulo.
- **`make_writeable` (`utils.py:943`):** activa `writeable` en vistas de bloques de pandas 3, que es justo la protección de CoW. Hoy es seguro porque la matriz es interna y `_check_in_place_fit` detecta las escrituras, pero es frágil.
- **Alcance:** la nota de LightGBM sobre `min_data_per_group` (notebook, skill y `llms-base.txt`) no tiene que ver con el fallo; debería ir en otro PR.
- **Convenciones:** sin rayas, y las líneas de más de 88 caracteres son código movido.

### Áreas sin fallos, con lo que ejecuté en cada una

- **Batería principal (pandas 3):** 2.720 configuraciones por versión: 10 estimadores (LightGBM, XGBoost, CatBoost, HGB, Ridge, LinearRegression, RandomForest, `Pipeline(StandardScaler, Ridge)`, `Pipeline(StandardScaler, LightGBM)`, `TTR(LightGBM)`), las 4 codificaciones, los 4 modos de `categorical_features` y 11 juegos de exog (float/int/bool, `category` de enteros con y sin coincidencia código-valor, con negativos, de decimales y de texto, object, `str`, nullable con y sin NA, pyarrow, fechas), con series DataFrame y dict de longitudes distintas con NaN. Se compara `predict`, `predict_interval` bootstrapping, los residuos in-sample y `set_in_sample_residuals`.
  - En la rama, `set_in_sample_residuals` coincide con los residuos de `fit` en todas las configuraciones.
  - Con `'auto'`: 688 configuraciones idénticas y las 32 diferencias del hallazgo 2.
- **Batería con pandas 2.3.3:** subconjunto de 6 estimadores. Mismo patrón.
- **Búsquedas one-step-ahead** (`t2_search.py`, `t2_cmp.txt`): rejilla y bayesiana, frente a un backtesting de un paso sin refit, con 5 estimadores × 3 codificaciones × `'auto'`/`None` × series DataFrame y dict.
  - En la rama la métrica de la búsqueda coincide con la del backtesting en todos los casos. En la base no coincidía con LightGBM y XGBoost con exog `ci` y `None`.
  - `ForecasterDirectMultiVariate` (rejilla y bayesiana) da resultados idénticos en base y rama.
- **Otros casos** (`t4_misc.py`): `weight_func` y `series_weights` (36 configuraciones), backtesting con y sin refit y `n_jobs` 1 y 2 con intervalos conformal (42), y `ForecasterRecursive`, `ForecasterDirect` y `ForecasterDirectMultiVariate`. Todo idéntico entre base y rama.
- **Estado entre ajustes** (`t3_state.py`): `set_lags`, segundo `fit`, `set_params`, parámetros del usuario restaurados tras el ajuste, y residuos iguales a los de `booster.inplace_predict` de XGBoost. Todo bien.
- **Afirmaciones de la nota** (`t9_claims.py`): las tres se cumplen.
  - Nombres con `[`, `<`: antes fallaba, ahora funciona.
  - `copy_X=False` con exog int: ahora da `ValueError`.
  - `Pipeline` que selecciona columnas por nombre: ahora falla en `fit`, pero en la base ya fallaba en `predict`, así que no se pierde nada real.

## Rendimiento

**Montaje:** pandas 3.1.1, `OMP_NUM_THREADS=1`, `n_jobs=1`, proceso fijado al núcleo 12 con prioridad alta. Datos: 300 series × 400 filas, 24 lags. Base y rama alternadas, 2 vueltas × 3 repeticiones = 6 muestras por celda, más un calentamiento.

**Resultados:** mediana rama/base (rango en segundos en `W\bench_summary.txt`).

| Config | fit | predict | backtesting | one-step-ahead (3 candidatos) |
|---|---|---|---|---|
| LightGBM, exog float | 0,998 | 1,000 | 1,005 | 1,001 |
| LightGBM, exog mixta | 0,974 | 0,987 | 1,011 | 0,982 |
| LightGBM, float, `ordinal_category` | 0,997 | 1,017 | 0,997 | 0,988 |
| Ridge, float | 0,997 | 0,986 | 0,997 | 0,971 |
| Ridge, mixta | **0,937** | 0,971 | 1,004 | 0,982 |
| HGB, mixta | 1,006 | 0,994 | 1,006 | 0,990 |
| XGBoost, float | 0,979 | 1,002 | 0,967 | 0,985 |
| XGBoost, mixta | 1,020 | 0,996 | 1,007 | 1,017 |

La dispersión intra-celda es de un 2-5 %, así que todo es neutro salvo el −6 % del fit de Ridge con exog mixta.

**Memoria** (`W\bench_mem.py`, `mem_results.jsonl`; procesos separados, 2 por celda; pico de tracemalloc / pico de working set):

| Caso | Base | Rama |
|---|---|---|
| Fit XGBoost, exog mixta | 41,4 / 343 MB | **58,3 / 368 MB**: XGBoost leía las columnas del DataFrame sin crear una copia float64 |
| One-step-ahead LightGBM, exog mixta | WS 292 MB | WS 273 MB |
| One-step-ahead LightGBM, `ordinal_category` | WS 278 MB | WS 259 MB |
| One-step-ahead Ridge, exog mixta | tracemalloc 72,5 MB | tracemalloc 53,1 MB |
| Resto | igual (±2 MB) | igual (±2 MB) |

**Trabajo redundante (deducido):**
- `to_numpy(dtype=float, na_value=np.nan)` hace una pasada `isna` completa en el camino mixto, igual que `_exog_to_numpy` en `predict`.
- `get_params()` se llama en cada candidato. Es despreciable frente al ajuste.
- La conversión única por split en las búsquedas es una mejora real, pero con 3 candidatos apenas se nota. No medí búsquedas con muchos candidatos.

## Pregunta 3: diseño

**Frente a la alternativa** (mantener el DataFrame en `fit` y convertir solo las columnas `category` no gestionadas):
- La alternativa arreglaría el fallo reportado con unas 20 líneas y conservaría los nombres de las features en el estimador y la configuración por nombre sin el parser de 150 líneas.
- Pero dejaría vivo el desfase DataFrame en `fit` / numpy en `predict`, que es el origen de toda la clase de fallos. La batería muestra que esa clase es mayor que el fallo reportado: pyarrow con LightGBM, `category` con CatBoost, nombres con caracteres especiales, y la discrepancia entre las métricas one-step-ahead y `predict`.
- La rama alinea `ForecasterRecursiveMultiSeries` con los demás forecasters, que ya entrenan con numpy.

**Coste para los usuarios:**
- Modelos que cambian sin tocar el código, solo con un aviso: hallazgos 2 y 3.
- `copy_X=False` con exog no float pasa a dar error.
- Se pierde `feature_names_in_` en el estimador.
- Las categorías decimales en una lista incompleta pasan a dar `TypeError` (antes daban predicciones erróneas en silencio).
- El hallazgo 1.

**Lo innecesariamente complicado:**
- `_resolve_user_categorical_config`. Recomiendo aceptar nombres solo en `fit_kwargs`, y en el constructor solo posiciones, con un error claro en otro caso.
- El `try/finally` duplicado.
- El relleno de la tupla del split.

**(a) Sin el criterio numpy:** la fusionaría con cambios.
1. Arreglar el hallazgo 1: códigos en `fit` y `predict` para las `category` configuradas como categóricas, o error claro.
2. Avisar con `ordinal_category` cuando el estimador es un metaestimador o un `Pipeline`.
3. Corregir la nota: líneas 103 y 142, la retirada de la función pública y las correcciones no anunciadas.
4. Simplificar el resolver y factorizar el set/restore.
5. Sacar la nota de `min_data_per_group` a otro PR.

**(b) Con el criterio numpy:** sí con más motivo, con los mismos cambios. Pero el criterio solo se cumple a medias:
- La matriz se sigue construyendo como DataFrame (`_create_train_X_y`) y se convierte al final con una pasada de pandas.
- CatBoost sigue recibiendo un DataFrame (`utils.py:1040`).
- `_get_fit_config` decide los avisos con dtypes de pandas.

Es una frontera numpy, no numpy interno. Su valor es que habilita construir `X_train` directamente en numpy más adelante. El coste para los usuarios de los hallazgos 2 y 3 no viene del criterio numpy sino de la decisión de tratar las `category` no gestionadas como numéricas, y la alternativa lo compartiría.

## Lo que no pude comprobar

- GPU.
- Linux y macOS.
- Versiones mínimas de las dependencias.
- Búsqueda bayesiana con `kwargs_study_optimize={'n_jobs': k}` (trials en hilos que comparten el estimador). El set/restore temporal no es seguro entre hilos, pero los trials ya mutaban el estimador en la base (deducido).
- `ForecasterRecursiveClassifier`.
- Búsquedas con muchos candidatos.
- Memoria con pandas 2.
- `LinearRegression` y otros lineales con series dict con NaN: fallan igual en las dos versiones por mis datos.
- La medida del pico de memoria del caso XGBoost con etiquetas de 3·10⁶: el proceso se mató a mano.

## Tests ejecutados

- **Rama, 16 ficheros** (los modificados y nuevos de `recursive`, `model_selection`, `utils` y `direct`), con `python.exe -m pytest <ficheros> -q -p no:cacheprovider`:
  - pandas 3: **1057 passed, 1 skipped**.
  - pandas 2: **1058 passed**.
- **Tests de la rama contra el código base** (copiados en `W\base` y restaurados después), 5 ficheros (`test_fit`, `test_predict`, `test_set_in_sample_residuals`, `test_create_predict_X`, `test_check_in_place_fit`): **41 failed, 280 passed**.
  - Fallan los tests nuevos clave, por ejemplo `test_predict_output_when_category_exog_not_in_categorical_features` y `test_set_in_sample_residuals_same_as_fit_when_category_exog_not_in_categorical_features`.
  - Ningún test nuevo cubre el hallazgo 1, el 2 ni pyarrow.

## Higiene

- No escribí nada en el repositorio. `git status` sigue con las mismas 31 entradas y no hay ficheros nuevos posteriores a mi inicio; solo cambió la fecha de `.git` porque `git status` refresca el índice.
- Maté procesos colgados míos (t8, t8b y una batería parada).
- No queda ningún proceso de Python ni de bash mío en marcha.

