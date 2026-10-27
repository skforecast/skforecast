# Migración de skforecast a pandas 3 manteniendo pandas 2: estudio

Fecha: 2026-10-27. Rama `0.27.x`, skforecast 0.27.0 (commit `6aee84b`).
Autor del estudio: sesión de Claude Code (cloud). Sin cambios en `skforecast/`,
`pyproject.toml` ni tests: todo lo que aquí se mide vive en
`dev/pandas3_migration/` (scripts, prototipos por monkeypatch y resultados).
Cómo reproducir cada número: `dev/pandas3_migration/README.md`.

Entornos usados (todos con Python 3.12.3, numpy 2.5.4, scikit-learn 1.9.1,
scipy 1.18.1, numba 0.68.0, statsmodels 0.15.0, lightgbm 4.7.0, skops 0.16.0,
joblib 1.5.3, pyarrow 26.0.0; sin torch ni keras):

| Entorno | pandas | Ruta |
|---|---|---|
| `p2` | 2.3.3 | `/root/.venvs/skforecast` (el de la sesión cloud, skforecast instalado en modo editable) |
| `p3` | 3.1.0 (PyPI, 2026-10-22) | `/root/.venvs/skforecast-p3`, mismas versiones de todo lo demás, ejecutado con `PYTHONPATH=<raíz del repo>` |

Máquina: contenedor Linux de 4 vCPU y 15 GB de RAM (`nproc`, `free -g`). Las
medidas de tiempo se han ejecutado de una en una con la máquina en reposo
(sección 4 indica el protocolo). Las notas de versión de pandas se citan por
fichero y línea de las fuentes RST (`doc/source/whatsnew/v3.0.0.rst` y
`v3.1.0.rst` del repositorio `pandas-dev/pandas`, clonado en
`/home/user/pandas-dev/pandas`), nunca de memoria.

<!-- TODO: resumen ejecutivo al final -->

## 1. Resumen ejecutivo

<!-- TODO -->

## 2. Línea base: la suite completa con pandas 3.1.0

Comando (una sola vez, secuencial, sin `-n`, excluyendo `skforecast/deep_learning`
porque importa `keras` al cargar los módulos de test):

```bash
PYTHONPATH=/home/user/skforecast ~/.venvs/skforecast-p3/bin/python -m pytest skforecast -q \
    -p warnrec --continue-on-collection-errors --ignore=skforecast/deep_learning \
    -rfE --junitxml=junit.xml --durations=40
```

(`warnrec` es un plugin de 20 líneas, `results/baseline_pandas3.1.0/warnrec.py`,
que vuelca cada aviso capturado por pytest como una línea JSON.)

Resultado: **100 fallos, 6933 pasados, 1 omitido, 4016 avisos en 9 min 49 s**
(`results/baseline_pandas3.1.0/pytest_full.log`). De los 100 fallos, 10 son
`ModuleNotFoundError: No module named 'torch'` en
`skforecast/foundation/tests/tests_foundation_models/test_TSICLAdapter.py` (el
entorno no tiene torch) y no tienen que ver con pandas. **Los otros 90 pasan
todos con pandas 2.3.3** en el mismo checkout (verificado:
`results/baseline_pandas3.1.0/failing_under_p2.log`, "90 passed").

Con pandas 3.0.6 (comprobación del 2026-10-04 citada en el encargo) eran 42
fallos, todos en multiseries. pandas 3.1.0 añade los fallos de la familia G3
(cambio de `Day`/zonas horarias y comparación de offsets) y amplía G1 y G2 a
más módulos; la sección 2.2 indica, fallo por fallo, cuáles son exclusivos de
3.1. <!-- TODO: confirmar con la clasificación -->

### 2.1 Fallos por familia (causa raíz)

| Familia | Nº tests | Directorios | Síntoma |
|---|---|---|---|
| G1 resolución del índice (`datetime64[ns]` esperado, `[us]` obtenido) | 46 | `recursive/tests_forecaster_recursive_multiseries` (38), `model_selection/tests_validation` (4), `model_selection/tests_utils` (1), `utils/tests_utils` (3: `expand_index` tz-aware, `skops_decompose_compose`) | `AssertionError: DataFrame.index are different` |
| G2 dtype `str` por defecto (`object` esperado) | 32 | `recursive/tests_forecaster_recursive_multiseries` (12), `preprocessing/tests_preprocessing` (5), `drift_detection` (1), `utils/tests_utils` (6), `model_selection/tests_validation` (8) | `assert <StringDtype(na_value=nan)> == dtype('O')`, `DataFrame.columns are different`, mensajes de error con el dtype impreso (`Regex pattern did not match`) |
| G3 zonas horarias, `Day` y comparación de offsets | 12 | `direct/tests_forecaster_direct` (2), `model_selection/tests_split` (3), `preprocessing` (1), `utils/tests_utils` (5), `recursive/tests_forecaster_stats` (1) | `ValueError: ... DatetimeIndex without a frequency`, `TypeError: '<' not supported between instances of 'Day' and 'Day'`, `Cannot infer dst time`, `If steps is a date, the index must have a frequency` |
| G4 torch ausente | 10 | `foundation` | `ModuleNotFoundError` (no es pandas) |

El detalle por test está en `results/baseline_pandas3.1.0/failures_by_group.json`.

### 2.2 Causas raíz, test a test

<!-- TODO: tabla de causas raíz (classification workflow) -->

### 2.3 Avisos

De los 4016 avisos capturados, 1793 son `pandas.errors.Pandas4Warning` (las
deprecaciones de pandas 3 que se retiran en pandas 4), 1761 de ellos emitidos
desde el paquete (el resto desde código de tests). No hay `FutureWarning` ni
`DeprecationWarning` de pandas. El resto son avisos propios de skforecast
(`DataTransformationWarning` 717, `IgnoredArgumentWarning` 368...), `UserWarning`
(325, de sklearn y statsmodels) y `ConvergenceWarning` (21), iguales que con
pandas 2. Tabla completa en `results/baseline_pandas3.1.0/warnings_summary.tsv`.

`Pandas4Warning` por punto de emisión en el paquete (recuento en la suite):

| Nº | Fichero:línea | Aviso | Entrada del whatsnew |
|---|---|---|---|
| 674 | `recursive/_forecaster_recursive_multiseries.py:1567` | `copy` keyword deprecated (`pd.concat(..., axis=0, copy=False)`) | v3.0.0 "Copy keyword" (l. 993); `concat` no aparece en esa lista pero el aviso lo emite `pandas/core/reshape/concat.py:408-410` de 3.1.0 |
| 235 + 219 | `preprocessing/_calendar.py:202`, `:247` | `DatetimeIndex.dayofweek` deprecated, usar `day_of_week` | v3.1.0 "Other deprecations" l. 524 |
| 105 | `preprocessing/_calendar.py:247` | `DatetimeIndex.dayofyear` deprecated, usar `day_of_year` | v3.1.0 l. 524 |
| 160 | `drift_detection/_population_drift.py:502` | `copy` keyword (`Series.astype("float64", copy=False)`) | v3.0.0 l. 993 |
| 35 | `preprocessing/_preprocessing.py:774` | `copy` keyword (`DataFrame.astype(new_dtypes, copy=False)`) | v3.0.0 l. 993 |
| 30 + 26 | `recursive/_forecaster_recursive.py:2042`, `:1109` | `copy` keyword (`astype(..., copy=False)` en `create_predict_X` y `create_train_X_y`) | v3.0.0 l. 993 |
| 27 + 18 | `direct/_forecaster_direct.py:2407`, `:1520` | ídem | v3.0.0 l. 993 |
| 24 + 24 | `direct/_forecaster_direct_multivariate.py:2586`, `:1637` | ídem | v3.0.0 l. 993 |
| 15 + 8 | `recursive/_forecaster_recursive_classifier.py:1843`, `:1129` | ídem | v3.0.0 l. 993 |
| 23 + 2 | `recursive/_forecaster_recursive_multiseries.py:1333`, `:3552` | `pd.concat(..., axis=1, copy=False)` y `astype(..., copy=False)` | v3.0.0 l. 993 |
| 22 + 9 | `drift_detection/_population_drift.py:394`, `:238` | `isinstance(obj, DateOffset)` deprecated para offsets que no son `DateOffset`; usar `pd.offsets.BaseOffset` | v3.1.0 "Other deprecations" l. 580 |
| 7 + 6 + 6 + 6 + 3 + 3 | `recursive/_forecaster_equivalent_date.py:218`, `:382`, `:412`, `:506`, `:670`, `:892` | ídem | v3.1.0 l. 580 |
| 16 x 4 | `foundation/_adapters.py:4995`, `:4996`, `:5000`, `:5001` | `dayofweek` / `dayofyear` | v3.1.0 l. 524 |
| 8 | `utils/utils.py:2458` | `pd.infer_freq` devolverá un `BaseOffset` en vez de `str` | v3.1.0 "Inferred frequencies will be returned as offsets" l. 495 |
| 2 | `preprocessing/_calendar.py:849` | la opción `future.no_silent_downcasting` está deprecada | v3.1.0 l. 520-588 (otras deprecaciones) |

Los avisos emitidos desde tests (32) son `assert_index_equal` comprobará
`freq` por defecto (v3.1.0 l. 523), el valor por defecto `exact="equiv"` de
`assert_index_equal` deprecado (v3.1.0 l. 566) y `dayofweek` en fixtures.

## 3. Análisis por dimensión

<!-- TODO: 3.1 copias y CoW; 3.2 cadenas; 3.3 fechas y frecuencias; 3.4 APIs retiradas e internals; 3.5 tests y CI; 3.6 rendimiento -->

## 4. Tabla de medidas

<!-- TODO -->

## 5. Resultados idénticos: comprobación con huellas

Script: `identity_cases.py` (21 casos, 110 salidas) y `identity_compare.py`.
Cada salida (matrices de entrenamiento, predicciones, intervalos, residuos,
métricas, tablas de búsqueda, dtypes almacenados, predicciones tras
guardar/cargar) se reduce a dos huellas SHA-256: la **estricta** incluye el
dtype de pandas de cada columna, la unidad del índice datetime y su `freq`; la
**normalizada** solo los valores (fechas pasadas a nanosegundos enteros,
cadenas por su `repr`). Casos: `ForecasterRecursive` (exógenas numéricas,
calendario, categóricas con LightGBM, texto con LightGBM, texto sin
`categorical_features`), `ForecasterDirect`, `ForecasterRecursiveMultiSeries`
(DataFrame ancho con `ordinal`, `ordinal_category` y `onehot`; dict de series y
dict de exógenas; formato largo con `reshape_series_long_to_dict` y
`reshape_exog_long_to_dict`), `ForecasterDirectMultiVariate`,
`ForecasterRecursiveClassifier`, `ForecasterStats(Arima)`,
`ForecasterEquivalentDate`, `backtesting_forecaster` (conformal),
`backtesting_forecaster_multiseries` (bootstrapping), `backtesting_stats`,
`grid_search_forecaster`, `grid_search_forecaster_multiseries`
(`OneStepAheadFold`), y guardado/carga con joblib y skops. Intervalos:
bootstrapping, bootstrapping con residuos por bin y conformal.

Datos de entrada: índice `pd.date_range("2020-01-01", periods=300, freq="D")`,
que en pandas 2 es `datetime64[ns]` y en pandas 3 `datetime64[us]` (ver 3.3).
Por eso se ejecuta dos veces en pandas 3: tal cual y con el índice forzado a
`[ns]` (`--index-unit ns`), para separar lo que hace pandas de lo que hace
skforecast.

| Comparación | Salidas | Huella estricta igual | Solo valores iguales | Distintas |
|---|---|---|---|---|
| p2 frente a p3 (índice de entrada tal cual: `[ns]` frente a `[us]`) | 110 | 37 | 69 | 4 (tamaños de fichero skops/joblib, no deterministas; eliminados del script después) |
| p2 frente a p3 con índice de entrada `[ns]` | 110 | 79 | 27 | 4 (ídem) |
| p2 frente a p2 con prototipo `no_copy_keyword` | 110 | 110 | 0 | 0 |
| p3 frente a p3 con prototipo `no_copy_keyword` | 110 | 108 | 0 | 2 (tamaños de fichero) |
| p2 frente a p2 con prototipo `dataframe_copy_false` | 110 | 110 | 0 | 0 |
| p3 frente a p3 con prototipo `dataframe_copy_false` | 110 | 110 | 0 | 0 |

(Tablas completas: `results/identity_pandas2.3.3_vs_pandas3.1.0.md`,
`results/identity_pandas2.3.3_vs_pandas3.1.0_ns.md`,
`results/identity_pandas*_vs_<prototipo>.md`.)

**Conclusión: todos los valores son idénticos bit a bit entre pandas 2 y
pandas 3** en los 21 casos (predicciones, intervalos, residuos, residuos por
bin, métricas, matrices, slices por nivel, tablas de búsqueda). Las 27
diferencias que quedan con índice `[ns]` son solo de dtype:

| Salida | pandas 2 | pandas 3 | Origen |
|---|---|---|---|
| Columna `level` de `predict`, `predict_interval` y `backtesting_forecaster_multiseries` (predicciones), columna `levels` de la tabla de métricas, `ForecasterDirectMultiVariate` incluido | `object` | `str` | skforecast construye la columna a partir de una lista de nombres (`pd.DataFrame`/`pd.Series` con cadenas): pandas 3 infiere `str` |
| `exog_dtypes_in_` de una exógena de texto (`recursive_text_exog_lgbm`, también tras `load_forecaster`) | `{'exog_text': dtype('O')}` | `{'exog_text': <StringDtype(na_value=nan)>}` | el dtype viene del DataFrame del usuario |
| `reshape_exog_long_to_dict` con columna de texto | `object` | `str` | ídem |
| `lags`, `params`, `levels` de las tablas de `grid_search_*` | `object` | `str` | artefacto del script (convierte esas columnas con `astype(str)` para poder hashearlas; en la librería siguen siendo `object` con listas y dicts en ambas versiones) |

Con el índice de entrada `[us]` (lo que obtiene un usuario de pandas 3 con
`pd.date_range` o `pd.to_datetime` desde cadenas), todas las salidas con índice
temporal salen en `[us]`: skforecast propaga la unidad de la entrada, no la
cambia (sección 3.3 y `experiments/exp_mixed_resolution.py`).

### 5.1 Resolución mezclada entre `fit` y `predict`

`experiments/exp_mixed_resolution.py` (resultados en
`results/exp_mixed_resolution_pandas*.txt`): con pandas 3, entrenar con índice
`[ns]` y predecir con `exog`/`last_window` en `[us]`, o al revés, no lanza
ningún error y los valores son idénticos a la predicción todo-`[ns]`; el índice
de la predicción toma la unidad de `last_window` (la almacenada en `fit` si no
se pasa otra). `expand_index` conserva la unidad del índice que recibe
(`[ns]` → `[ns]`, `[us]` → `[us]`), con y sin `freq`. `TimeSeriesFold` con
`initial_train_size` como fecha (`str` o `Timestamp`) funciona con ambas
unidades. Multiseries con series `[ns]` y exógenas `[us]` en dict: funciona y
la matriz de entrenamiento sale en `[ns]`.

### 5.2 Persistencia entre versiones

`identity_crossload.py` guarda en una versión y carga en la otra
(`results/identity_crossload_*.json`):

| Backend | Guardado en p2, cargado en p3 | Guardado en p3, cargado en p2 |
|---|---|---|
| joblib (pickle) | Funciona; predicciones idénticas; `exog_dtypes_in_` conserva el `object` del fichero | **Falla**: `AssertionError` en `pandas/core/indexes/datetimes.py:93` (`_new_DatetimeIndex`) al reconstruir el `DatetimeIndex` |
| skops | Funciona; predicciones idénticas | Funciona; predicciones idénticas (`exog_dtypes_in_` llega como `str`) |

Es una limitación de pandas, no de skforecast: `experiments/exp_pickle_compat.py`
muestra que un `DatetimeIndex`, un `DataFrame` con índice temporal o uno con
una columna `category` serializados con `pickle` en pandas 3.1.0 no se cargan en
pandas 2.3.3 (`NotImplementedError` al reconstruir), mientras que los de pandas
2 se cargan en pandas 3 (una `Series` `str` de pandas 3 sí se carga en 2.3.3, que
ya conoce `StringDtype(na_value=nan)`). pandas no garantiza compatibilidad
hacia atrás de pickles; skops serializa por contenido y funciona en ambos
sentidos. Hay que documentarlo (sección 6).

## 6. Cambios de comportamiento visibles que no se pueden evitar

<!-- TODO -->

## 7. Plan de implementación

<!-- TODO -->

## 8. Decisiones pendientes

<!-- TODO -->

## 9. Preguntas abiertas y lo no verificado

<!-- TODO -->
