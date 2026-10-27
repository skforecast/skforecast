# Migración de skforecast a pandas 3 manteniendo pandas 2: estudio

Fecha: 2026-10-27. Rama `0.27.x`, skforecast 0.27.0 (commit `6aee84b`).
Autor del estudio: sesión de Claude Code (cloud). Sin cambios en `skforecast/`,
`pyproject.toml` ni tests: todo lo que aquí se mide vive en
`dev/pandas3_migration/` (scripts, prototipos por monkeypatch y resultados).
Los scripts, prototipos (monkeypatch) y ficheros de resultados que se citan
(`results/...`, `prototypes/...`, `experiments/...`) se generaron en la sesión
y **no se versionan**: este documento es la única entrega. El anexo final
recoge los comandos para reproducir cada número.

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

## Índice

1. Resumen ejecutivo (las respuestas, las cifras y la decisión)
2. Línea base: la suite completa con pandas 3.1.0 (fallos y avisos por causa raíz; la suite sobre el prototipo)
3. Análisis por dimensión (3.1 copias y CoW; 3.2 cadenas; 3.3 fechas y frecuencias; 3.4 APIs retiradas e internals; 3.5 tests y CI; 3.6 rendimiento)
4. Tabla de medidas (tiempo y memoria pico, pandas 2 y 3, antes y después)
5. Resultados idénticos: comprobación con huellas (y persistencia entre versiones)
6. Cambios de comportamiento visibles y texto para las notas de versión
7. Plan de implementación por PRs
8. Decisiones que debes tomar tú
9. Preguntas abiertas y lo no verificado
10. Pendiente para retomar

## 1. Resumen ejecutivo

**¿Se puede migrar a pandas 3? Sí, manteniendo pandas 2.** El mismo código
funciona con pandas 2.2, 2.3 y 3.x sin ramas por versión en los forecasters:
la única rama necesaria se aísla en un módulo de compatibilidad de unas 40
líneas (`skforecast/utils/_pandas_compat.py`: el keyword `copy=` solo en
pandas 2 y `pd.infer_freq` con la opción nueva de 3.1). El pin propuesto es
`pandas>=2.2, <4.0` con `requires-python>=3.10` sin cambios: pandas 3 exige
Python 3.11 y numpy 2.0.2, así que en Python 3.10 el resolutor seguirá
instalando pandas 2.3.

**¿Hay un refactoring grande? No.** El paquete no usa ninguna API eliminada
en 3.0 (83 comprobaciones sin ocurrencias), no escribe in situ en ningún
array obtenido de pandas (el riesgo típico de Copy-on-Write) y no depende de
los internals de pandas más allá de lo que ya controla (el bloque F único de
`ForecasterRecursiveMultiSeries` se conserva en pandas 3). El código que
cambia son unos 25 puntos concretos, listados con `fichero:línea` en la
sección 7: 13 llamadas con `copy=False`, 7 renombrados (`dayofweek`,
`DateOffset`), 1 alias de frecuencia retirado en un dataset, 4 guardias de
una o dos líneas para índices tz-aware y para columnas de texto, y 4
constructores de `DataFrame` que en pandas 3 copian sin necesidad. La parte
grande es la suite: 90 tests cuyas expectativas son específicas de pandas 2
(dtype `object`, resolución `[ns]`, textos de mensajes). Esfuerzo estimado:
3 a 4 días, la mitad en tests.

**¿Se pierde rendimiento? pandas 2 queda igual; pandas 3 pierde algo de
tiempo frente a pandas 2 incluso con los cambios propuestos, pero no memoria.** pandas 3 copia
los arrays de numpy al construir un `DataFrame` y los `astype` y `concat`
fragmentan igual que antes: `create_train_X_y` de los forecasters que
construyen la matriz en numpy es entre 1,6 y 3,1 veces más lento en pandas 3
y usa entre un 43 % y un 75 % más de memoria pico; `fit` univariante es un
16 % más rápido, `predict` en bucle un 40 % más lento (0,2 ms por llamada),
y el backtesting se mueve entre 0,97 y 1,41 veces. Con `copy=False` en esos
constructores y el `astype` solo de las columnas que cambian, aplicados
únicamente bajo pandas 3 desde el módulo de compatibilidad, pandas 3 baja a
entre 0,47 y 0,74 del tiempo y entre 0,57 y 0,76 de la memoria en esas
funciones, y pandas 2 conserva exactamente su ruta de hoy (quitar el
keyword `copy=False` sin más costaría hasta 1,9 veces el tiempo y un 22 %
más de memoria, medido). Algunos caminos ya mejoran con pandas 3 sin tocar
nada: `ForecasterRecursiveMultiSeries` con `onehot` y dicts (27 % menos
memoria pico), la ruta `pd.concat` (20 % menos) y las window features (25 %
menos). Comparado con pandas 2, pandas 3 con los cambios queda entre 1,25 y 1,74
veces en tiempo en la construcción de matrices y 1,4 en `predict` en bucle
y en el backtesting de folds de un paso, con la misma memoria (1,00 a 1,09):
lo que resta es coste propio de pandas 3 por llamada (seguimiento de
referencias de Copy-on-Write, constructores e indexación), no copias
evitables desde skforecast. En otros caminos pandas 3 ya gana (`fit`
univariante 0,84; multiseries con dicts y window features con un 20 % a 27 %
menos de memoria). Dos resultados A/B quedan sin explicar y la PR debe
medirlos con el cambio real (3.6). Las cifras, con su protocolo y versiones,
están en 3.6 y 4.

**¿Se mantiene la compatibilidad de resultados? Sí, bit a bit.** 21 casos
que cubren los cinco forecasters autorregresivos, `ForecasterStats`,
`ForecasterEquivalentDate`, exógenas numéricas, categóricas y de texto,
dict y formato largo, window y calendar features, los tres tipos de
intervalo, backtesting, búsqueda y guardado/carga dan 106 salidas con los
mismos valores en pandas 2.2.3, 2.3.3 y 3.1.0, antes y después de los
cambios propuestos. Las únicas diferencias son de tipo, heredadas de pandas
3 y aceptadas: las columnas de texto y la columna `level` salen con dtype
`str` en vez de `object`, y los índices temporales conservan la unidad de la
entrada (`datetime64[us]` cuando el usuario crea las fechas desde cadenas,
que es el nuevo valor por defecto de pandas). Hay una excepción, encontrada
y acotada: MASE y RMSSE de `backtesting_forecaster_multiseries` con
`ForecasterDirectMultiVariate` y un `DataFrame` ancho con NaN en la serie
objetivo, porque `DataFrame.stack()` deja de descartar los NaN; el valor de
pandas 3 es el correcto y se propone adoptarlo en ambas versiones con nota
de versión.

**Otros resultados que conviene conocer**

- La suite con pandas 3.1.0 da 100 fallos (10 por falta de torch). Los 90
  restantes pasan con pandas 2.3.3 y fallan igual con pandas 3.0.6: 47 son
  solo expectativas de test, 32 son dtypes o textos de mensaje heredados de
  pandas y 11 son fallos reales del paquete, todos en índices tz-aware
  anclados en UTC (pandas 3 descarta la frecuencia `D` al hacer
  `tz_convert`) o en la ordenación de offsets `Day`. Con los cambios
  propuestos aplicados por monkeypatch, la suite da 96 fallos (los 4 del
  paquete que tienen arreglo), ninguno nuevo, y los avisos de deprecación
  emitidos por el paquete pasan de 1761 a 81 (los que quedan son tres
  renombrados pendientes en módulos no importables aquí o cubiertos por el
  plan).
- Dos errores previos salen a la luz: con `transformer_series` o
  `transformer_y` que transforman in situ (`StandardScaler(copy=False)`),
  pandas 2 guarda `last_window_` ya transformado y predice mal (y
  `ForecasterRecursive.fit` muta la serie del usuario en ambas versiones);
  pandas 3 lo corrige solo en el camino multiseries. Y `TimeSeriesFold` con
  series de frecuencias no comparables lanza `TypeError` en vez del
  `ValueError` previsto también con pandas 2.
- Persistencia: un fichero joblib escrito con pandas 3 no se carga con
  pandas 2 (límite de pandas, no de skforecast); skops funciona en los dos
  sentidos. Hay que documentarlo.
- El job semanal `unit-tests-latest-deps.yml` ya instala pandas 3 (su
  `--upgrade` ignora el pin) y lleva ocho ejecuciones seguidas en rojo.
- La ruta correcta para series diarias tz-aware ancladas en UTC pasa a ser
  `freq="24h"` (idéntica a `"D"` de pandas 2 en valores); hay que
  documentarla.

**Decisión recomendada**: abrir el soporte de pandas 3 en 0.27 con el pin
`pandas>=2.2, <4.0`, en el orden de PRs de la sección 7 (renombrados y
módulo de compatibilidad, correcciones de robustez, tests neutrales, pin y
CI, documentación y contexto de IA), con la comprobación de identidad de la
sección 5 como criterio de aceptación de cada PR. Las decisiones que
quedan en tu mano (mínimo de pandas 2, política de avisos en CI, qué hacer
con los dos errores previos y con `stack()`) están en la sección 8; lo que no
se ha podido verificar, en la 9.

| Cifra clave | Valor |
|---|---|
| Suite con pandas 3.1.0, código actual | 100 fallos (90 por pandas), 6933 pasados, 1761 `Pandas4Warning` del paquete |
| Suite con pandas 3.1.0, cambios propuestos | 96 fallos, 0 nuevos, 81 avisos del paquete |
| Fallos reales del paquete / solo tests / dtype o texto heredado | 11 / 47 / 32 |
| Salidas comparadas por huella (pandas 2.2.3, 2.3.3, 3.1.0, antes y después) | 106 de 106 con los mismos valores |
| Puntos de código que cambian | unos 25 (sección 7) más un módulo de compatibilidad |
| Tests que cambian | 90 |
| Esfuerzo | 3 a 4 días |

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
fallos, todos en multiseries. No es un efecto de pandas 3.1: los 90 fallos se
reproducen con pandas 3.0.6 en el árbol actual (2.2); la diferencia está en
los tests añadidos a `0.27.x` desde esa fecha.

### 2.1 Fallos por familia (causa raíz)

| Familia | Nº tests | Directorios | Síntoma |
|---|---|---|---|
| G1 resolución del índice (`datetime64[ns]` esperado, `[us]` obtenido) | 46 | `recursive/tests_forecaster_recursive_multiseries` (38), `model_selection/tests_validation` (4), `model_selection/tests_utils` (1), `utils/tests_utils` (3: `expand_index` tz-aware, `skops_decompose_compose`) | `AssertionError: DataFrame.index are different` |
| G2 dtype `str` por defecto (`object` esperado) | 32 | `recursive/tests_forecaster_recursive_multiseries` (12), `preprocessing/tests_preprocessing` (5), `drift_detection` (1), `utils/tests_utils` (6), `model_selection/tests_validation` (8) | `assert <StringDtype(na_value=nan)> == dtype('O')`, `DataFrame.columns are different`, mensajes de error con el dtype impreso (`Regex pattern did not match`) |
| G3 zonas horarias, `Day` y comparación de offsets | 12 | `direct/tests_forecaster_direct` (2), `model_selection/tests_split` (3), `preprocessing` (1), `utils/tests_utils` (5), `recursive/tests_forecaster_stats` (1) | `ValueError: ... DatetimeIndex without a frequency`, `TypeError: '<' not supported between instances of 'Day' and 'Day'`, `Cannot infer dst time`, `If steps is a date, the index must have a frequency` |
| G4 torch ausente | 10 | `foundation` | `ModuleNotFoundError` (no es pandas) |

El detalle por test está en `results/baseline_pandas3.1.0/failures_by_group.json`.

### 2.2 Causas raíz, test a test

Cada familia la ha analizado un agente (reproducción mínima en los dos
entornos, lectura del test y del paquete, cita del whatsnew) y la ha revisado
un segundo agente escéptico que volvió a ejecutar cada fragmento y cada test.
Los JSON completos, con los fragmentos y sus salidas, están en
`results/baseline_pandas3.1.0/classification/`. Resumen:

| Id | Causa raíz | Tests | Tipo | Visible al usuario | Entra por | Arreglo |
|---|---|---|---|---|---|---|
| G1-RC1a | `pd.date_range(...)` y `pd.DatetimeIndex([cadenas])` dan `datetime64[us]` en pandas 3 (v3.0.0 l. 435); las expectativas de los tests se construyen así y se comparan con fixtures persistidos en `[ns]` (`fixtures_forecaster_recursive_multiseries.py:141-153`, joblib; `fixtures_model_selection_multiseries.py:132-141`, parquet) | 38 | solo test | No: el paquete propaga la unidad de la entrada (`expand_index`, `utils.py:2484-2488`, usa `pd.date_range(start=index[-1], ...)`, que conserva la unidad del `Timestamp`) | tests | Expectativas con `pd.date_range(..., unit="ns")` (kwarg desde pandas 2.0) o `pd.DatetimeIndex([...], dtype="datetime64[ns]")`; verificado 79/79 aserciones copiadas en 2.3.3, 3.0.6 y 3.1.0 |
| G1-RC1b | Expectativa fijada a `dtype="datetime64[ns]"` mientras la entrada se construye con `pd.date_range` sin unidad (`[us]` en pandas 3): `test_predict.py` de `ForecasterStats` (l. 257-263) y `test_backtesting_forecaster_multiseries.py:4166-4170` (6 casos de `ForecasterDirectMultiVariate` con window features) | 7 | solo test | No | tests | Quitar el pin de dtype o usar `.as_unit(entrada.index.unit)` |
| G1-RC2 | `_decompose_index` (`utils.py:2948-3014`) guarda `index.asi8` y `index.unit`; el test espera `unit: "ns"` y enteros en ns | 2 | solo test | No: el payload es fiel y el round trip 2.3.3 ↔ 3.1.0 es exacto (pandas 2.2+ soporta `[us]`) | tests | Fixture con `unit="ns"`; `tz_zoneinfo` calculado con `isinstance(index.tz, zoneinfo.ZoneInfo)` |
| G1-RC3 | `DatetimeIndex.tz_convert` descarta `freq` en pandas 3 salvo para offsets `Tick` (no documentado en el whatsnew; código `arrays/datetimes.py:937-939` en 3.0.6 e `indexes/datetimes.py:1558-1559` en 3.1.0; en 2.3.3 lo conservaba siempre). `expand_index` (`utils.py:2456-2463`) infiere entonces la frecuencia sobre la hora local y, para un índice anclado en UTC con `MS`, obtiene `ME` (índice de predicción un día antes) y con `QS` `None` (`ValueError`) | 1 | **cambia el resultado / rompe** | Sí, con índices tz-aware anclados en UTC construidos con `tz_convert` | paquete | Inferir también sobre `index.tz_convert("UTC")` y usar esa inferencia cuando `_is_utc_anchored_index` lo confirma (prototipo del agente verificado con la copia completa de `test_expand_index.py` en 2.3.3, 3.0.6 y 3.1.0); sin rama por versión, porque en pandas 2 el `freq` ya viene informado y el bloque no se ejecuta. Aplicar la misma inferencia donde se exige `index.freq` (`utils.py:2049`) |
| G2-RC1 | dtype `str` por defecto (v3.0.0 l. 27): las columnas de texto del usuario llegan como `str` y el paquete lo almacena tal cual en `exog_dtypes_in_` (`utils.py:1109-1112`, `get_exog_dtypes`) y en los dicts que devuelve (`reshape_exog_long_to_dict`, `_preprocessing.py:656-666`); los tests esperan `np.dtype("O")` (`test_create_train_X_y.py:854,917,...`, `test_fit.py:290`, `test_reshape_exog_long_to_dict.py:116,165,267`, `test_check_preprocess_exog_multiseries.py:515`, `test_align_series_and_exog_multiseries.py:728`) | 19 | cambia el resultado (heredado de la entrada) | Sí: quien inspeccione `exog_dtypes_in_` ve `str` | entrada del usuario | Ninguno en el paquete (normalizar a `object` perdería información). Tests: un `STR_DTYPE = pd.Series(["a"]).dtype` compartido (`dtype("O")` en 2.x, `StringDtype(na_value=nan)` en 3.x; `pd.api.types.pandas_dtype(str)` no sirve: en 2.3.3 da `<U`) |
| G2-RC2 | El `Index` de columnas con etiquetas de texto es `str` en pandas 3: `assert_frame_equal` contra `columns=[]` (`object`, `inferred_type` "empty") falla (`test_create_train_X_y.py:4798-4820`, 6 casos de exógenas enteras insertadas) y el fixture pickle del drift detector (`tests_population_drift/test_predict.py:15-16, 218-219`) tiene columnas `object` | 7 | cambia el resultado (universal en pandas 3) | Sí (`X_train.columns.dtype`) | pandas | Ninguno en el paquete. Tests: `columns=pd.Index(float_cols, dtype=STR_DTYPE)`; regenerar o comparar el pickle con `check_column_type=False` |
| G2-RC3 | `__module__` público (v3.0.0 l. 850): `f"{type(index)}"` en mensajes de error imprime `<class 'pandas.RangeIndex'>` en vez de `<class 'pandas.core.indexes.range.RangeIndex'>` (`utils.py:4373, 4555, 4579-4580`, `_preprocessing.py:857-858`) | 4 | cambia el texto de un mensaje | Sí (cosmético) | paquete | Aceptar (opcional: imprimir `type(x).__name__`). Tests: construir la expectativa con `f"{pd.RangeIndex}"` (y `pd.Index`, no `pd.RangeIndex`, en `test_reshape_series_exog_dict_to_long.py:80-81`) |
| G2-RC4 | `MultiIndex.from_product([..., range(n)])` tiene segundo nivel `RangeIndex` (v3.0.0 "Performance improvements" l. 1220, `Categorical.categories` devuelve `RangeIndex`); el mensaje "Found <class ...>" cambia | 2 (solapados con RC3) | cambia el texto de un mensaje | Sí (cosmético) | pandas | Ninguno. Tests: `f"Found {type(series.index.levels[1])}."` |
| G2-RC5 | `Day` ya no es `Tick` (v3.0.0 l. 723): `sorted({Day, 2*Day})` lanza `TypeError`; `check_preprocess_series` (`utils.py:4466-4471`) cae en su `except TypeError: sorted(key=str)` y el mensaje lista `[<2 * Days>, <Day>]` en vez de `[<Day>, <2 * Days>]` | 2 | cambia el texto de un mensaje | Sí (cosmético) | paquete | Opción A: nada; opción B: `sorted(indexes_freq, key=lambda f: (type(f).__name__, getattr(f, "n", f)))`, idéntico en 2.3.3, 3.0.6 y 3.1.0 |
| G3-RC1a | Misma causa que G1-RC3 (`tz_convert` descarta `freq=<Day>`): `pd.date_range(..., freq="D", tz="UTC").tz_convert(tz)` llega al paquete sin frecuencia y las validaciones lanzan (`utils.py:2047-2051` `check_extract_values_and_index`, `_split.py:375-380`, `utils.py:4455-4462`, `utils.py:2395-2399`) | 7 | **rompe** | Sí: con pandas 3 una serie diaria tz-aware anclada en UTC construida con `tz_convert` no se puede ajustar, y el usuario tampoco puede reponer el `freq` (`idx.freq = "D"` lanza `ValueError` en pandas 3) | paquete (validación correcta) y pandas | Las validaciones son correctas. En pandas 3 la convención "pasos fijos de 24 h en UTC" se escribe `freq="24h"`, que conserva el `freq` tras `tz_convert` en ambas versiones y da resultados bit a bit idénticos a `"D"` de pandas 2 (verificado). Documentar en `troubleshooting-common-errors` y en la guía; tests con `"24h"` (14/14 aserciones en ambas versiones). Ver también la decisión D6 |
| G3-RC1b | `reshape_series_wide_to_long` no valida `freq is None` (`_preprocessing.py:471-477`) y llama `asfreq(None)`, que pandas interpreta como día de calendario: `AmbiguousTimeError` en otoño o NaN silenciosos en primavera | 2 | **rompe** | Sí | paquete | Guardia barata: `ValueError` si `data.index.freq is None` (el docstring ya exige frecuencia); pandas 2 queda idéntico para toda entrada válida |
| G3-RC2 | `Day` no ordenable: `_split.py:383-388` hace `sorted(indexes_freq)` sin el `except TypeError` que sí tiene `utils.py:4466-4471`: `TypeError` en lugar del `ValueError` previsto | 1 | **rompe** (tipo de excepción) | Sí | paquete | Replicar el fallback `key=str` (o la clave determinista de G2-RC5) |
| G3-RC3 | Igual que G1-RC2 (`_decompose_index`, unidad y `zoneinfo`) | 2 | solo test | No | tests | `unit="ns"` en la fixture |
| G4 | torch ausente | 10 | entorno | - | - | - |

Dos de los 90 tests (`test_check_preprocess_series.py::test_TypeError_check_preprocess_series_when_series_is_pandas_DataFrame_MultiIndex_without_datetime` y `test_check_preprocess_exog_multiseries.py::test_TypeError_check_preprocess_exog_multiseries_when_exog_is_not_dict_and_series_is_dict`) tienen dos causas (G2-RC3 y G2-RC4); los dos de skops aparecen en G1-RC2 y G3-RC3. Recuento neto: 47 solo test, 32 cambio de dtype o de texto de mensaje heredado de pandas, 11 fallos reales del paquete (G1-RC3, G3-RC1a, G3-RC1b, G3-RC2), todos en índices tz-aware o en comparaciones de offsets.

**Ninguna causa es exclusiva de pandas 3.1**: los 90 tests fallan igual con
pandas 3.0.6 (`results/baseline_pandas3.1.0/failing_under_pandas3.0.6.log`,
entorno con las mismas versiones del resto de paquetes). La diferencia con los
42 fallos del 2026-10-04 no es de pandas sino del árbol: desde esa fecha han
cambiado 484 ficheros de tests en `0.27.x` y, en particular, los tests de
índices tz-aware anclados en UTC y la maquinaria `_is_utc_anchored_index`
entraron con la PR #1405 el 2026-10-15.

Lo que el paquete hace bien y conviene conservar: no fabrica resoluciones ni
dtypes (propaga los de la entrada), y `_decompose_index` guarda la unidad real,
con lo que un skops escrito con pandas 3 (`unit: "us"`, `ZoneInfo`) se carga en
pandas 2 sin pérdida.

### 2.3 La suite completa sobre el prototipo

Misma orden que la línea base, con todos los cambios de código propuestos en la
sección 7 aplicados por monkeypatch al arrancar pytest (prototipo
`all_proposed`: helper `copy=` solo en pandas 2, `DataFrame(copy=False)` y
`astype` de las columnas que cambian, copia de `y` en multiseries, orden de
offsets, guardia de `reshape_series_wide_to_long`, `object`/`str` equivalentes
en `check_preprocess_exog_multiseries`, inferencia en UTC en `expand_index`,
`day_of_week`/`BaseOffset`; sin tocar ningún test):

| | Línea base (código actual) | Prototipo |
|---|---|---|
| Resultado | 100 fallos, 6933 pasados, 1 omitido, 4016 avisos, 9 min 49 s | **96 fallos, 6937 pasados, 1 omitido, 2342 avisos, 10 min 0 s** |
| Fallos nuevos | - | **0** |
| Fallos arreglados | - | 4: `_split.py` `TypeError` (G3-RC2), orden de offsets ×2 (G2-RC5), `expand_index` `MS` anclado en UTC (G1-RC3) |
| `Pandas4Warning` desde el paquete | 1761 | **81**: `foundation/_adapters.py:4995-5001` (64, módulo no importable aquí: mismo renombrado pendiente), `pd.infer_freq` en `expand_index` (15, helper de 3.3 pendiente), `_calendar.py:849` (2, ruta numpy pendiente) |

Los 86 fallos restantes relacionados con pandas son los esperados: 45 de G1 y
30 de G2 son expectativas de los tests (dtype `object`, unidad `[ns]`,
mensajes), y los 11 de G3 son los 7 de índices tz-aware construidos con
`tz_convert` (por diseño: sin frecuencia no hay predicción; los tests pasan a
`freq="24h"`), los 2 de `reshape_series_wide_to_long` (ahora un `ValueError`
claro en vez de NaN silenciosos; mismos tests) y los 2 de skops (`unit`).
Log y lista de fallos en `results/final_suite_all_proposed/`.

### 2.4 Avisos

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

Método: siete agentes, uno por dimensión, recorrieron el paquete fichero a
fichero (no por muestreo), ejecutaron cada fragmento de verificación en los
dos entornos y citaron el whatsnew (229 hallazgos en total). Un segundo
agente por dimensión intentó refutar cada hallazgo volviendo a ejecutar los
fragmentos; por límite de uso de la sesión solo terminaron dos de esos
verificadores (`recursive/`: 10 confirmados, 8 corregidos en matices, 0
refutados, 8 comprobaciones añadidas; cadenas: 12 confirmados, 3 corregidos
en líneas, 0 refutados, 8 comprobaciones añadidas) y los de la clasificación
de fallos (sección 2.2). Las dimensiones de `direct/base/utils`,
`model_selection`, fechas, APIs retiradas y tests/CI quedan con la
verificación por fragmentos de su propio agente (ejecutada en ambos
entornos), sin la segunda revisión independiente; la sección 9 lo recoge.
Las correcciones de los verificadores están incorporadas en el texto.

### 3.1 Copias y Copy-on-Write

Hechos de pandas verificados (`experiments/exp_cow_basics.py`, inventario
`cow_recursive` s1-s14 y `cow_direct_base_utils` s01-s16; whatsnew v3.0.0
"Consistent copy/view behaviour with Copy-on-Write" l. 83 y guía
`copy_on_write.rst`):

| Operación | pandas 2.3.3 | pandas 3.1.0 |
|---|---|---|
| `pd.DataFrame(ndarray)`, `pd.Series(ndarray)` | no copia (comparte memoria) | **copia** salvo `copy=False` (`migration.rst` l. 159-164); el keyword `copy` del constructor no está deprecado |
| `df.to_numpy()`, `.values`, `np.asarray(df)` con un solo bloque | vista escribible | vista **de solo lectura** (`writeable=False`); con varios bloques o dtypes mezclados: copia escribible, como antes |
| `pd.concat([...], axis=1)` | 2 bloques (sin consolidar), copia los datos | 2 bloques, sin copia eager; `copy=` emite `Pandas4Warning` (`concat.py:408-410`); en pandas 2 `copy=False` no ahorra memoria (tracemalloc igual con y sin él, `cow_direct_base_utils` F08) |
| `df.astype(dtypes)` | copia todas las columnas; con `copy=False` solo las que cambian | perezoso; `copy=` emite `Pandas4Warning` (v3.0.0 l. 993) |
| `df[lista_de_columnas]` | copia | vista (CoW) |
| `df.iloc[a:b]` | vista | vista |
| `df.drop(columns=...)` | copia consolidada (1 bloque) | perezoso: un bloque por columna (`nblocks` = nº de columnas) hasta que algo lo materializa |
| `DataFrame.insert` | un bloque nuevo por inserción en ambas (6 bloques tras 5 inserciones) | ídem |
| Asignación encadenada, `SettingWithCopyWarning` | avisa | nunca escribe en el original; el aviso desaparece |

Inventario del paquete (180 hallazgos: 121 en `recursive/`, `direct/`,
`base/`, `preprocessing/`, `utils/` y 59 en `model_selection/`, `metrics/`,
`feature_selection/`, `stats/`, `drift_detection/`, `datasets/`, `plot/`,
`foundation/`, `deep_learning/`, `experimental/`; todos verificados por
fragmento en los dos entornos salvo los de `foundation/` y `deep_learning/`,
leídos sin ejecutar):

- **Nada rompe.** Ningún camino de `fit`, `predict`, `predict_interval`,
  `predict_bootstrapping`, `create_train_X_y`, `create_predict_X`,
  `set_out_sample_residuals`, `get_feature_importances`, transformadores,
  window features, diferenciación, calibrador conformal ni `QuantileBinner`
  escribe in situ en un array obtenido de pandas: todas las escrituras van a
  buffers propios (`np.empty`/`np.full`, el bloque F, `last_window`
  concatenado) o a copias (`to_numpy(copy=True)`). Con `-W error` sobre
  `Pandas4Warning` solo fallan los sitios con `copy=` de la tabla de 2.4 y los
  `dayofweek`/`dayofyear`. Las funciones `@njit` (lags, window features,
  diferenciación, intervalos) reciben arrays creados por skforecast, no vistas
  de pandas. Los datos del usuario no se modifican en ninguna versión (probado
  con `np.shares_memory` y comparando antes y después).
- **Los 114 `.copy()`**: 16 en `recursive/` (10 necesarios en ambas versiones,
  6 redundantes en ambas) y 21 en `direct/`, `base/`, `preprocessing/` y
  `utils/` (necesarios: `last_window_`, la copia del calibrador conformal y
  los `to_numpy(copy=True)`; el resto son copias defensivas profundas antes de
  sustituir columnas, que admiten `copy(deep=False)` en ambas versiones sin
  mutar al usuario). Las copias profundas de `check_preprocess_series` y
  `check_preprocess_exog_multiseries` (`utils.py:4367-4369, 4387-4392,
  4418-4421, 4544-4558, 4602-4606`) siguen protegiendo en pandas 2 frente a
  transformadores in situ; en pandas 3 CoW ya protege. Candidatos a
  `copy(deep=False)` seguros en ambas y con ahorro real: `utils.py:812-818`,
  `:4387-4392`, `_preprocessing.py:538-540`, `:656-657` (y 674, 744),
  `_forecaster_direct.py:1114-1126`, `_forecaster_direct_multivariate.py:1194-1206`.
  Redundantes en ambas (eliminar sin cambiar resultados):
  `_forecaster_equivalent_date.py:640, 668, 672, 868, 894`,
  `_forecaster_recursive_multiseries.py:2756-2761, 2793-2795, 2934-2935`,
  `_forecaster_stats.py:637`. Tabla completa en `results/inventory/*.md`.
- **`copy=` (35 sitios)**: 18 son `pd.Series(..., copy=False)` o
  `to_numpy(copy=True)` (no deprecados). Los deprecados son los 13 de la tabla
  de 2.3 (`astype(..., copy=False)` ×10, `pd.concat(..., copy=False)` ×2,
  `Series.astype("float64", copy=False)` ×1). Quitar el keyword sin más
  cuesta en pandas 2 una copia completa de `X_train` en cada
  `create_train_X_y`/`create_predict_X` con exógenas (medido por el agente con
  tracemalloc: +1× la matriz si todo es float, +0,56× con exógenas enteras o
  categóricas): de ahí el helper del módulo de compatibilidad (`copy=False`
  solo bajo pandas 2) o, mejor, castear solo las columnas cuyo dtype cambia
  (prototipo `astype_changed_only`, 0 copias del bloque float en ambas y frame
  consolidado). Para `pd.concat` el keyword se quita sin más (gratis en pandas
  2, medido en 3.6).
- **Copias de más en pandas 3** (`pd.DataFrame(ndarray)` sin `copy=False`):
  `create_train_X_y` de los cuatro forecasters que construyen la matriz en
  numpy (`_forecaster_recursive.py:1100-1104`, `_forecaster_recursive_classifier.py:1120-1124`,
  `_forecaster_direct.py:1497-1510`, `_forecaster_direct_multivariate.py:1614-1627`),
  los `DataFrame`/`Series` de los caminos de `predict` (`_forecaster_recursive.py:951, 1111, 2033, 2139, 2287, 2421`,
  `_forecaster_recursive_multiseries.py:1025, 2939, 3524, 3659, 3841, 4011`,
  `_forecaster_recursive_classifier.py:955, 1023, 1131, 1144, 1834, 1919, 1999`,
  `_forecaster_stats.py:946-955, 1160-1164`) y `transform_dataframe`
  (`utils.py:2816-2830`, copia de la salida del transformador). Todos son
  arrays privados: `copy=False` es seguro en ambas versiones y neutro en pandas
  2 (prototipo `dataframe_copy_false` para los cuatro primeros; medido en 3.6).
  Excepción: `_forecaster_recursive.py:1100` sin exógenas envuelve la vista
  `sliding_window_view` de los lags, que en pandas 2 llega al usuario como un
  `DataFrame` de solo lectura (`X_train.iloc[0, 0] = 1` lanza `ValueError`,
  hallazgo F36) y en pandas 3 se copia; materializar la vista
  (`np.ascontiguousarray`) y después `copy=False` iguala las dos versiones y
  arregla el caso de pandas 2.
- **El bloque F único** de `_assemble_train_X_y`
  (`_forecaster_recursive_multiseries.py:1136-1340`) se conserva en pandas 3:
  `nblocks=1`, F-contiguo, `to_numpy()` comparte memoria con el bloque (solo
  pasa a `writeable=False`, y el estimador no escribe). Con exógenas enteras
  insertadas el frame tiene 2 bloques en ambas versiones y `to_numpy()` copia
  en ambas: sin cambio. La ruta `pd.concat` da `nblocks=1` en ambas versiones
  en los casos probados por el agente (`out_s6_fblock_*`).
- **`encoding=None` con exógenas**: `X_train.drop(columns="_level_skforecast")`
  (`_forecaster_recursive_multiseries.py:2048-2049, 2460-2464, 4598-4602`)
  copia una vez en pandas 2 (bloque único) pero en pandas 3 es perezoso y deja
  un bloque por columna; el estimador interleava en `fit` y otra vez al
  calcular los residuos (o en cada candidato de una búsqueda one-step-ahead):
  dos copias de `X_train` en vez de una. Medido en 3.6
  (`ms_fit_linear_encoding_none`); los coeficientes y predicciones coinciden
  dígito a dígito.
- **Un cambio de resultado a favor de pandas 3** (F26):
  `_forecaster_recursive_multiseries.py:990` toma `y.to_numpy()` sin copia y
  lo pasa al `transformer_series`; con un transformador in situ
  (`StandardScaler(copy=False)`) pandas 2 transforma la copia interna de la
  serie, `last_window_` guarda valores ya transformados y `predict` los vuelve
  a transformar: `[88.69, 91.70, ...]` frente a `[101.98, 104.53, ...]` con
  `copy=True`. En pandas 3 la vista es de solo lectura, scikit-learn copia y
  el resultado es correcto. El prototipo `multiseries_y_copy`
  (`y.to_numpy(copy=True)`) iguala las dos versiones (verificado:
  predicciones idénticas a `copy=True` en ambas; coste neutro, 3.6). El
  verificador añade un matiz importante: **`ForecasterRecursive.fit` tiene el
  mismo problema en las dos versiones, y peor**, porque pasa el `DataFrame`
  de `y` a scikit-learn (`transform_dataframe`, `_forecaster_recursive.py:814-820`)
  y, con `transformer_y=StandardScaler(copy=False)`, scikit-learn 1.9.1 vuelve
  escribible la vista de solo lectura (`force_writeable` pone
  `flags.writeable = True` cuando la base lo permite, en vez de copiar) y
  **muta la serie del usuario** en pandas 2 y en pandas 3 (verificado:
  predicciones `[147.32, 129.08, ...]` frente a `[101.30, 101.54, ...]`).
  Es un error previo, no de pandas 3, pero el mismo arreglo (copiar antes de
  un transformador in situ, o comprobar `transformer.copy`) lo cierra; va en
  la PR 3. Por la misma razón la copia de `_forecaster_stats.py:737-743` no
  es redundante en pandas 3 (sin ella un transformador in situ mutaría
  `last_window`). La comprobación de "el estimador modificó la matriz in situ"
  (`utils.py:853-896`) no cubre los transformadores. Ningún test lo cubre.
- **`df[lista]` en `_get_level_row_slices`**: el camino normal lee la columna
  de nivel o un slice `iloc` de las columnas one-hot (vista en ambas); la
  alternativa `X_train[encoding_cols]` solo se usa si las columnas no están
  contiguas, y en pandas 3 es una vista. Sin cambio que hacer; coste medido en
  3.6.
- **Bucle de `predict`**: por llamada, `ForecasterRecursive.predict` hace 1
  `DataFrame`, 1 `copy(deep=True)` (de `last_window`), 5 `to_numpy`, 2
  `expand_index` y 1 `check_predict_input`, en las dos versiones por igual;
  `ForecasterRecursiveMultiSeries.predict` con `exog` dict añade 3 `reindex`,
  3 `reindex_like`, 4 `copy(deep=False)` y 1 `pd.concat`. No hay copias por
  paso dentro del bucle recursivo (todo es numpy). pandas 3 no cambia el
  recuento; el coste por llamada se mide en 3.6 (`predict_loop`,
  `expand_index_check_predict_input`).
- **sklearn y arrays de solo lectura**: `check_array(force_writeable=True)`
  (scikit-learn ≥ 1.6) copia cuando recibe una vista de solo lectura, así que
  los estimadores que escriben en `X` (`copy_X=False`, pipelines con
  `copy=False`) no fallan en pandas 3; con scikit-learn 1.9.1 verificado, con
  1.6 (mínimo soportado) no.
- **`model_selection/`, `stats/`, `drift_detection/`, `datasets/`,
  `feature_selection/`, `plot/`**: ninguna ruta rompe (backtesting con refit,
  conformal, bootstrapping y predictores; clasificador; multiseries con dict y
  NaN; las tres búsquedas con los dos folds; `backtesting_stats` y
  `grid_search_stats` con Arima, Arima automático con exógenas, Sarimax, Ets y
  Arar; los dos detectores de drift; `select_features`; `acf`/`pacf`; plots;
  `fetch_dataset`) y los datos del usuario no cambian (hash de `y`, `exog`,
  `series` antes y después). Los objetos por fold (`y.iloc`, `exog.iloc`,
  `v.loc`, `series.iloc`) son vistas en ambas versiones y nunca se mutan;
  `drop`, `dropna(axis=1)`, `set_index`, `[[cols]]` y `rename` pasan de copiar
  a vistas perezosas; `pd.concat` copia en ambas. Las funciones de `stats`
  copian antes de escribir (`y.copy()`, `astype`), y numba, scipy y
  matplotlib solo leen; el único efecto es que `Arima.coef_` (`_arima.py:673`)
  queda de solo lectura en pandas 3 (guardia opcional: `.copy()`). Copias
  redundantes en ambas versiones: `_range_drift.py:427-428` (una copia
  profunda del `exog` por serie, solo leída) y `_forecaster_rnn.py:1352-1356`.
  Avisos no cubiertos por la suite pero presentes en el código:
  `isinstance(x, pd.DateOffset)` en `_split.py:304, 1051` y
  `_utils.py:319-321` (mismo arreglo que 3.3). `_utils.py:1535-1541, 1599-1603`
  usan `series.stack()`, que en pandas 3 conserva los NaN (en pandas 2 los
  descartaba): la API pública convierte las series a dict antes
  (`_validation.py:1882`) y los 19 tests del fichero pasan en ambas versiones,
  pero `stack(future_stack=True)` (pandas ≥ 2.1) fija el comportamiento nuevo
  en las dos.

Recomendación de conjunto: (1) un módulo `skforecast/utils/_pandas_compat.py`
con `concat`, `astype`, `frame_from_array`, `writable_array`,
`is_string_dtype` e `infer_freq` (borrador en `prototypes/pandas_compat_draft.py`),
único sitio que mira `pd.__version__`; (2) quitar `copy=False` de los dos
`pd.concat`; (3) `astype` solo de las columnas que cambian; (4) `copy=False`
en los `DataFrame`/`Series` construidos sobre arrays privados; (5) la copia de
`y` en multiseries (F26); (6) consolidar tras `drop` solo si la medida de 3.6
lo justifica; (7) dejar los `.copy()` defensivos de `check_preprocess_*` como
están (protegen en pandas 2) y pasar a `copy(deep=False)` los redundantes,
en una PR aparte y medida.

### 3.2 Cadenas: el dtype `str` por defecto

Hechos (v3.0.0 "Dedicated string data type by default" l. 27-75;
`migration.rst` l. 453-880; 14 fragmentos del inventario `strings` ejecutados
en los dos entornos):

| | pandas 2.3.3 | pandas 3.1.0 |
|---|---|---|
| `pd.Series(["a"]).dtype`, columnas de texto de un `DataFrame`, `Index` de etiquetas de texto (columnas) | `object` | `StringDtype(na_value=nan)` (repr `str`); el valor ausente es `np.nan` |
| `dtype == np.dtype(object)` sobre esa columna | `True` | `False`; `pd.api.types.is_string_dtype` es `True` en ambas |
| `pd.CategoricalDtype(categories=ndarray_object)` | categorías `object` | categorías `str` (v3.0.0 l. 1276); `CategoricalDtype` con categorías `object` y con `str` **no son iguales** |
| `pd.concat` de una columna `object` y una `str` | `object` | `object` |
| `pd.Series(dtype=str)` | `object` | `str` (neutral para construir frames vacíos) |
| `pd.api.types.pandas_dtype(str)` | `<U` (inútil) | `str` |
| `fillna("texto")` sobre una `Series` float | silencioso, pasa a `object` | `Pandas4Warning` (v3.1.0 l. 530), pasa a `object` |

En skforecast (30 hallazgos: 1 rompe, 11 cambian un dtype, 1 aviso latente,
2 solo tests, 15 sin problema):

- **No hay comprobaciones frágiles**: cero ocurrencias de `== object`,
  `is_object_dtype`, `kind == "O"`, `select_dtypes("object")`, `.str` ni
  `np.issubdtype` sobre dtypes de pandas. `get_exog_dtypes` y
  `check_exog_dtypes` (`utils.py:1092-1115`) aceptan y rechazan lo mismo con
  los mismos mensajes; `categorical_features="auto"` codifica las columnas
  `str` igual que las `object`, y LightGBM y `HistGradientBoosting` predicen
  idéntico (sección 5).
- **Un fallo real (S05)**: `check_preprocess_exog_multiseries`
  (`utils.py:4645-4660`) exige que cada exógena tenga el mismo dtype en todas
  las series (`exog_dtypes_buffer.nunique(axis=1) == 1`). En pandas 3 una
  columna de texto es `str` en una serie y `object` en otra cuando en esta es
  toda `None` (o cuando el usuario mezcla `astype(object)` con texto inferido):
  `TypeError: Exog/s: ['txt'] have different dtypes in different series`,
  mientras que pandas 2 continúa (`object` en ambas). Guardia barata y neutra
  en pandas 2: tratar `object` y `str` como el mismo dtype en esa comparación
  (`pd.api.types.is_string_dtype(d) and not isinstance(d, pd.CategoricalDtype)`)
  y mencionar `object`/`str` en el mensaje. Ningún test lo cubre.
- **Dtypes que cambian sin cambiar valores** (aceptar y documentar, sección
  6): `exog_dtypes_in_`/`exog_dtypes_out_` (`utils.py:1092-1115`,
  `_forecaster_recursive.py:1106-1109`); las columnas `level`/`levels` de
  `predict*`, backtesting y búsqueda (`_forecaster_recursive_multiseries.py:3532, 3846, 4016`,
  `_utils.py:1533-1541, 1566, 1602-1613, 1646-1664, 1695-1702`); las
  predicciones de `ForecasterRecursiveClassifier` con clases de texto
  (`_forecaster_recursive_classifier.py:790-845, 1454-1456, 1919`); las salidas
  de `reshape_exog_long_to_dict` (`_preprocessing.py:595-782`),
  `transform_dataframe` (`utils.py:2812-2825`), `PopulationDriftDetector.predict`
  (`_population_drift.py:482-516, 560-585, 730-790`) y `fetch_dataset`
  (`datasets.py:800, 869`).
- **Dos guardias de una palabra**: `_validation.py:2848-2851`
  (`_backtesting_foundation`): el frame vacío de un fold saltado usa
  `pd.Series(dtype=object)` y degrada `level` a `object` al concatenar; con
  `pd.Series(dtype=str)` queda `object` en pandas 2 y `str` en pandas 3, como
  los demás folds. `_forecaster_recursive_multiseries.py:2026-2056`:
  `fillna("_unknown_level")` sobre una `Series` que puede ser float (nombres
  de serie enteros y nivel no visto) avisará en pandas 3.1; construir la
  `Series` con `dtype=object` antes del `fillna` lo evita (hoy el camino no se
  alcanza desde `_train_test_split_one_step_ahead`, es latente). El verificador
  añade el caso real: con `encoding="onehot"` y nombres de serie enteros
  (`:2028, 2052-2057`) la `Series` es `object` sin NaN y en pandas 2 el
  `fillna` hace un *downcast* silencioso a `int64` con `FutureWarning`
  ("Downcasting object dtype arrays on .fillna"); en pandas 3 queda `object`.
  Los valores son los mismos; el `dtype=object` explícito y un `astype`
  posterior fijan el resultado en ambas versiones.
- **Persistencia (S15, S16)**: joblib y skops de pandas 2 cargan y predicen en
  pandas 3 (las categorías se recomponen como `str` y los dtypes nativos
  coinciden). Un `.skops` escrito con pandas 3 con exógenas de texto añade
  `pandas.StringDtype` (y los nombres públicos `pandas.DataFrame`,
  `pandas.DatetimeIndex`) a la lista de tipos no confiables que devuelve
  `skops.io.get_untrusted_types`: quien pase `trusted=[...]` revisado debe
  ampliarla. Opción: descomponer `pd.StringDtype` en `_decompose_dtype`
  (`utils.py:3147-3205`) y recomponerlo con `pd.Series([], dtype=str).dtype`,
  para un fichero neutro. Los `Series`/`DataFrame` con columnas `str` de pandas
  3 sí se despicklean en 2.3.3 (el fallo de joblib 3 → 2 es por el
  `DatetimeIndex`, 5.2). pandas 2.2 no verificado para `StringDtype(na_value=nan)`.
- **Tests y docs**: 26 de los fallos de la línea base son de esta dimensión
  (G2-RC1 y RC2). Forma neutra: `STR_DTYPE = pd.Series(["a"]).dtype` o
  `exog["col"].dtype` en las expectativas, `pd.Series(dtype=str)` para frames
  vacíos, `columns=pd.Index([], dtype=STR_DTYPE)`; evitar `pandas_dtype(str)`
  y `pd.DataFrame(columns=[])`. En `docs/`, 4 notebooks muestran `dtype:
  object` en 10 salidas y 2 listan dtypes: cosmético al reejecutar.
- `__repr__` y `summary()` de los forecasters no imprimen dtypes.

### 3.3 Fechas, zonas horarias y frecuencias

Hechos verificados (`experiments/exp_datetime_resolution.py`,
`experiments/exp_mixed_resolution.py`, clasificación G1 y G3):

| Construcción | pandas 2.3.3 | pandas 3.1.0 | Entrada del whatsnew |
|---|---|---|---|
| `pd.date_range("2020-01-01", periods=n, freq="D")`, con o sin hora en la cadena, con `tz=` | `datetime64[ns]` | `datetime64[us]` | v3.0.0 "Datetime/timedelta resolution inference" l. 435 (l. 446: "The new default resolution when parsing strings is microseconds"); l. 975 (`date_range` sin `unit`) |
| `pd.date_range(pd.Timestamp("2020-01-01"), ...)`, `pd.date_range(datetime(...), ...)` | `[ns]` | `[us]` (el `Timestamp` desde cadena es `[us]`, el `datetime` de la biblioteca estándar también) | ídem |
| `pd.DatetimeIndex([cadenas])`, `pd.to_datetime([cadenas])`, `read_csv(parse_dates=)`, `pd.Series([datetime])`, `.asfreq()` sobre ellos | `[ns]` | `[us]` | ídem |
| `pd.to_datetime(["2020-01-01 00:00:00.123456789"])` | `[ns]` | `[ns]` (la cadena lo exige) | ídem |
| `pd.DatetimeIndex(np.array([...], dtype="datetime64[D]"))` | `[s]` | `[s]` | ya era así en 2.x |
| `pd.date_range(..., unit="ns")`, `.as_unit("ns")`, `pd.DatetimeIndex([...], dtype="datetime64[ns]")` | `[ns]` | `[ns]` | existen desde pandas 2.0 (verificado en 2.2.3, 2.3.3 y 3.1.0) |
| `pd.date_range(..., freq="D", tz="UTC").tz_convert(tz).freq` | `<Day>` | `None` (`tz_convert` conserva `freq` solo si es `Tick`, y `Day` dejó de serlo; código `arrays/datetimes.py:937-940` en 3.0.x, `indexes/datetimes.py:1558-1559` en 3.1.0) | v3.0.0 "Changed behavior of `pd.offsets.Day` to always represent calendar-day" l. 723 (consecuencias l. 749-755); la pérdida del `freq` en `tz_convert` no está documentada |
| `pd.date_range(..., freq="24h", tz="UTC").tz_convert(tz).freq` | `<24 * Hours>` | `<24 * Hours>` | ídem |
| `pd.offsets.Day(1) < pd.offsets.Day(2)`; `sorted({Day, 2*Day})` | `True`; ordena | `TypeError` (solo `Tick` define las comparaciones) | v3.0.0 l. 723-755 (la pérdida del orden no figura en la lista) |
| `pd.infer_freq(idx)` | `"D"` | `"D"` más `Pandas4Warning` (devolverá un `BaseOffset`; opción `future.infer_freq_returns_offset`) | v3.1.0 l. 495 |
| `DatetimeIndex.dayofweek`, `.dayofyear`, `.weekday` | ok | `Pandas4Warning`: usar `day_of_week`, `day_of_year` (existen desde pandas 1.1) | v3.1.0 l. 524, l. 563 |
| `isinstance(pd.offsets.Day(1), pd.DateOffset)` | `True` | `True` más `Pandas4Warning` (será `False`; usar `pd.offsets.BaseOffset`) | v3.1.0 l. 580 |
| `pd.date_range(..., tz="Europe/Madrid").tz` | `pytz` | `zoneinfo.ZoneInfo` | v3.0.0 l. 512 |
| `pd.to_datetime(["... +01:00", "... +02:00"])` (desfases mezclados) | `FutureWarning` y dtype `object` | `ValueError: Mixed timezones detected. Pass utc=True ...` (con `utc=True` funciona en ambas) | v3.0.0 "Other Removals" l. 1134 (verificado: `scratchpad` `mixed_offsets.py`, reproducido en 3.4) |

Qué hace skforecast con cada uno:

- **Resolución.** El paquete no fabrica resoluciones: `expand_index` crea el
  índice de predicción con `pd.date_range(start=index[-1], periods, freq)`, que
  conserva la unidad del `Timestamp` de partida (`utils.py:2484-2488`), y
  `_decompose_index` guarda `index.unit` (`utils.py:2994-2995`). Con entradas
  `[ns]` todas las salidas son `[ns]` y bit a bit idénticas a pandas 2 (sección
  5); con entradas `[us]` salen `[us]`. Mezclar unidades entre `fit` y `predict`
  funciona y la unidad de salida es la de `last_window` (5.1). Recomendación:
  no normalizar a `[ns]` en la librería (sería ir contra pandas y costaría una
  conversión por llamada); aceptar el cambio y documentarlo (sección 6). Los
  tests fijan `unit="ns"` donde comparan con fixtures persistidos en `[ns]`.
- **`index_freq_`.** Los forecasters guardan `index.freqstr` (cadena) y
  `check_predict_input` compara cadenas: no le afecta que `pd.infer_freq`
  pase a devolver offsets. `expand_index` (`utils.py:2458`) es el único sitio
  que llama a `pd.infer_freq`, y ya envuelve el resultado con `to_offset`, que
  acepta cadena y offset: el aviso de 3.1 desaparece activando la opción o,
  mejor, con un helper del módulo de compatibilidad que llame a
  `pd.infer_freq` bajo `pd.option_context("future.infer_freq_returns_offset", True)`
  cuando la opción exista (una sola rama, aislada). `_calendar.py:862` tiene el
  mismo uso.
- **Índices tz-aware anclados en UTC** (la convención de `_is_utc_anchored_index`,
  `utils.py:2214-2273`, PR #1405): con pandas 3 un índice diario construido
  con `tz_convert` llega sin `freq`, el usuario no puede reponerlo
  (`idx.freq = "D"` lanza `ValueError` en pandas 3 porque `Day` ya no encaja
  en el índice local) y las validaciones del paquete lanzan (G3-RC1a, 7 tests;
  `utils.py:2047-2051`, `_split.py:375-380`, `utils.py:4455-4462`, `:2395-2399`).
  La convención correcta en pandas 3 es `freq="24h"`, que conserva el `freq` y
  produce exactamente los mismos índices, posiciones, predicciones y métricas que
  `"D"` en pandas 2 (verificado por el agente de G3). `expand_index` tiene el
  caso adicional `MS`/`QS` anclado en UTC sin `freq` (G1-RC3): infiere sobre la
  hora local y obtiene `ME` o nada; el arreglo propuesto, inferir también sobre
  `index.tz_convert("UTC")` y usarlo cuando `_is_utc_anchored_index` lo
  confirma, no necesita rama por versión y es bit a bit idéntico donde el
  código actual funciona (verificado por el agente y su verificador con la copia
  completa de `test_expand_index.py` en 2.3.3, 3.0.6 y 3.1.0; única diferencia:
  un índice anclado en UTC con `2D` sin `freq` devuelve `freq=<2 * Days>` en vez
  de `<48 * Hours>`, mismos valores).
- **`reshape_series_wide_to_long`** llama a `asfreq(None)` cuando el índice no
  tiene `freq` (`_preprocessing.py:471-477`), que pandas interpreta como día de
  calendario: `AmbiguousTimeError` o NaN silenciosos (G3-RC1b). Guardia de una
  línea.
- **Orden de offsets** en mensajes de error: `_split.py:383-388` necesita el
  mismo `except TypeError` que ya tiene `utils.py:4466-4471` (G3-RC2). Con
  pandas 2 ese sitio ya lanza `TypeError` en vez del `ValueError` previsto
  cuando las frecuencias no son comparables (`Day` y `MonthBegin`, o una
  frecuencia y un paso entero), así que el arreglo también corrige pandas 2.
  Prototipo `fix_offsets_sort` (verificado en ambas versiones): se mantiene el
  orden natural cuando existe y, si no, los offsets se ordenan por (tipo, `n`)
  y el resto por su texto; los mensajes de pandas 2 no cambian
  (`[<Day>, <2 * Days>]`, `[<Day>, <MonthBegin>]`, `[1, <Day>]`) y pandas 3
  los reproduce (G2-RC5); solo `Day` frente a `Hour`, sin test, sale
  `[<Day>, <Hour>]` en pandas 3 y `[<Hour>, <Day>]` en pandas 2.
- **`pytz`.** El paquete no importa `pytz` (0 ocurrencias). Los ficheros skops
  guardan la zona por nombre (`utils.py:2976`), así que un fichero de pandas 2
  (`pytz`) se carga en pandas 3 (`ZoneInfo`) y al revés (verificado).
- **Alias de frecuencia retirados** (`M`, `Q`, `Y`, `H`, `T`, `S`...; v3.0.0
  l. 1061): tres ocurrencias, todas verificadas. `datasets.py:209` registra
  `australia_tourism` con `'freq': 'Q'`: con pandas 3 `fetch_dataset` devuelve
  la serie **sin frecuencia** (aviso de pandas) y cualquier forecaster lanza
  después el `ValueError` de "índice sin frecuencia"; cambiar a `'QE'` (válido
  desde pandas 2.2). `skills/drift-detection/SKILL.md:96` (y su copia
  generada en `docs/llms-full.txt:6237`) muestra
  `PopulationDriftDetector(chunk_size='M')`, que en pandas 3 lanza
  `ValueError`: cambiar a `'ME'` y regenerar. `experimental/_splitter.py:67`
  usa `freq='d'` en un docstring (`Pandas4Warning`). En tests, notebooks,
  `.md` y `tools/ai/llms-base.txt`: 0 alias retirados.
- **Resolución mezclada en `fit`**: `DatetimeIndex.equals` entre `[ns]` y
  `[us]` es `False` en pandas 2 y `True` en pandas 3 (no documentado;
  verificado por fragmento), así que `fit(y [ns], exog [us])` lanza el
  `ValueError` de alineación de `_forecaster_recursive.py:919-925` en pandas 2
  y funciona en pandas 3. Solo afecta a quien mezcle unidades a mano en
  pandas 2; una nota en `troubleshooting-common-errors` (`.as_unit`) basta.
- **`ForecasterEquivalentDate(offset=pd.offsets.Day(n))`** con índice tz-aware
  que cruza un cambio de hora: en pandas 2 `Day` eran 24 h fijas y la fecha
  equivalente no existía (`KeyError`); con el `Day` de calendario de pandas 3
  funciona (v3.0.0 l. 723). Mejora gratuita; `pd.DateOffset(days=n)` da el
  mismo resultado en ambas versiones y se puede recomendar en la docs.
- **`_calendar.py:860-873`** (`_freq_to_timedelta_unit` y la normalización de
  frecuencias) usa `pd.infer_freq` y compara cadenas: funciona en 3.x con aviso
  y romperá cuando `infer_freq` devuelva offsets (pandas 4); normalizar con
  `to_offset(...).freqstr` (una línea) lo deja listo para ambas.
- **Tests y docs (recuentos de la dimensión)**: 201 `pd.DatetimeIndex([cadenas])`
  en 42 ficheros de test, 1721 `pd.date_range` en 188 (uno con `unit=`), 18
  literales `'datetime64[ns]'` en 7, 24 aserciones de `freq` en 12, 0
  `as_unit`. De todos ellos solo fallan los 48 de la sección 2 (los demás
  comparan entradas y salidas construidas de la misma forma, que cambian de
  unidad a la vez). `assert_index_equal(exact="equiv")` no ignora la unidad;
  `check_index_type=False` sí (verificado). En `docs/`, 7 líneas de 3
  notebooks imprimen `datetime64[ns]` (cosmético al reejecutar); 0 en
  `skills/` y `llms-base.txt`.
- **`DateOffset` en `isinstance`**: `_forecaster_equivalent_date.py:218, 382,
  412, 506, 670, 892` y `_population_drift.py:238, 394`; sustituir por
  `pd.offsets.BaseOffset` (existe en pandas 2) sin cambio de comportamiento.
- **`dayofweek`/`dayofyear`**: `_calendar.py:199-202, 247`,
  `_adapters.py:4995-5001`; sustituir por `day_of_week`/`day_of_year`.

### 3.4 APIs retiradas o deprecadas y dependencias de internals

Método: el agente recorrió entrada por entrada las secciones "Removal of
prior version deprecations/changes" (v3.0.0 l. 1058-1216), "Other API
changes" (v3.0.0 l. 956-989, v3.1.0 l. 355-385), "Deprecations" (v3.0.0
l. 990-1057, v3.1.0 l. 386-588) y los cambios incompatibles no cubiertos por
3.2 y 3.3, buscando cada API en el paquete, los tests y los notebooks.

- **Ninguna API eliminada en 3.0 se usa**: 83 comprobaciones con cero
  ocurrencias (`applymap`, `swapaxes` de pandas, `Series.view`/`ravel`,
  `DataFrame.bool`, `is_categorical_dtype`, `fillna(method=)`, `inplace=True`,
  `_mgr`/`_data`, `Series.__getitem__` posicional, alias `M`/`Q`/`Y`/`H`/`T`
  en código, `mode.use_inf_as_na`, `pd.value_counts`, `iteritems`, `lookup`,
  `groupby(axis=)`, `read_*` con keywords retirados...). Los `swapaxes` del
  paquete son de numpy.
- **Deprecaciones de 3.x presentes** (todas ya en 2.4 y 3.3, aquí solo el
  resumen): `copy=` en `astype` (11 llamadas) y `pd.concat` (2);
  `future.no_silent_downcasting` + `fillna(False)` sobre float en
  `_calendar.py:849-850` (sustituir por una ruta numpy, idéntica en 2.2+ y 3);
  `pd.infer_freq` (`utils.py:2458`, `_calendar.py:862`); `dayofweek`/
  `dayofyear` (7 sitios); `isinstance(x, pd.DateOffset)` (12 sitios, que en
  pandas 4 devolverá `False` para `Day`, `Week` o `MonthBegin` y romperá
  `ForecasterEquivalentDate`, `PopulationDriftDetector` y los folds con
  `window_size` como offset: cambiar a `pd.offsets.BaseOffset`).
- **`DataFrame.stack()` conserva los NaN en pandas 3** (el comportamiento
  `future_stack=True` pasa a ser el único; v3.0.0 "Other Removals"). Es el
  **único cambio de valores** encontrado en todo el estudio:
  `model_selection/_utils.py:1536` (`_calculate_metrics_backtesting_multiseries`)
  apila la `series` ancha para calcular las métricas por nivel; en pandas 2
  las filas NaN desaparecen y `y_train` de MASE y RMSSE queda con huecos
  cerrados, en pandas 3 se conservan. Afecta a `backtesting_forecaster_multiseries`
  con un `DataFrame` ancho cuya serie objetivo tiene NaN en la parte de
  entrenamiento y métricas escaladas, solo en el camino que no convierte las
  series a dict (`ForecasterDirectMultiVariate` y `ForecasterRnn`;
  `ForecasterRecursiveMultiSeries` convierte antes, `_validation.py:1882`).
  Reproducido (`HistGradientBoostingRegressor`, 160 obs, 20 NaN en la serie
  objetivo, 8 folds de 5 pasos): MASE 30.37 y RMSSE 15.78 en pandas 2 frente
  a 36.95 y 37.67 en pandas 3, con MAE idéntico. `stack(future_stack=True)`
  (pandas ≥ 2.1) hace que pandas 2 devuelva los valores de pandas 3
  (prototipo `fix_stack_future`, verificado). El valor de pandas 3 es el
  correcto: es el que da el camino dict y el que respeta los huecos del
  entrenamiento. Se propone adoptarlo en ambas versiones con nota de versión
  (decisión D12). `_search.py:1576, 1935, 2036, 2456` apilan tablas de
  métricas (sin NaN en el uso normal; con una métrica NaN pandas 2 fallaba y
  pandas 3 no): mismo cambio por coherencia.
- **`groupby(level=0)` con un nivel categórico**: `observed` pasa a `True`
  por defecto (v3.0.0 "Improved behavior in groupby for `observed=False`"
  l. 353): `_population_drift.py:682, 857` y `_foundation_model.py:738`
  deben fijar `observed=True` para el mismo resultado en ambas versiones (y
  quitar el `FutureWarning` de pandas 2).
- **`pd.concat` con frames vacíos** (`_validation.py:2847-2857`): el dtype de
  `level` depende de si hubo folds vacíos (3.2); una palabra.
- **Internals**: `pd.concat(axis=1)` ya no consolida y `to_numpy()` copia,
  pero para las matrices reales de skforecast (`ForecasterRecursive` con y sin
  exógenas; `ForecasterRecursiveMultiSeries` con los tres `encoding`) el número
  de bloques, el reparto de memoria y la contigüidad son idénticos en 2.3.3 y
  3.1.0; el umbral de `PerformanceWarning` por fragmentación son 100 bloques en
  ambas (de ahí el `< 100` de `_assemble_train_X_y`). Los constructores
  `pd.DataFrame(ndarray)`, `pd.Series(ndarray)` e `Index(ndarray)` copian por
  defecto en pandas 3 (el de `Index` documentado en v3.0.0 l. 985; los otros
  dos no tienen entrada propia, verificados por fragmento): 3.1 inventaría los
  sitios. `X.flags.writeable = True` (`utils.py:4308`) solo recibe arrays
  creados por skforecast. Un análisis AST del paquete no encuentra ninguna
  mutación in situ de arrays derivados de pandas.
- **Dependencias**: pandas 3.1 exige numpy ≥ 2.0.2 y Python ≥ 3.11; el pin
  `numpy>=1.26.1` y `requires-python>=3.10` siguen siendo válidos con
  `pandas>=2.2,<4` (el resolutor elige pandas 2.3 en Python 3.10 o con numpy
  1.26). `unit-tests-min-deps.yml` seguirá resolviendo pandas 2.2.0 con numpy
  1.26.1 y nunca ejercitará pandas 3 (3.5).
- **Tests**: `assert_index_equal` sin `exact=` (`RangeIndex` frente a
  `Index[int64]`) y `assert_frame_equal` con niveles `DatetimeIndex` de
  distinta `freq` emiten `Pandas4Warning` en 3.1 (v3.1.0 l. 523, 566); solo
  avisos, sin cambio de resultado. 352 `assert_index_equal`, 1239
  `assert_frame_equal` y 332 `assert_series_equal` en la suite.

### 3.5 Tests y CI

Recuentos en la suite (grep del agente de tests/CI, verificados con
fragmentos de `pd.testing` en las dos versiones):

| Patrón | Ocurrencias | Efecto con pandas 3 |
|---|---|---|
| `assert_frame_equal` / `assert_series_equal` / `assert_index_equal` | 1239 / 332 / 352 | Solo fallan las 90 de la sección 2; `check_dtype=False` (12 usos) absorbe `object`/`str` pero **no** la unidad `[ns]`/`[us]` (verificado) |
| `dtype=object` en expectativas | 53 líneas en 11 ficheros (`test_bayesian_search_forecaster_multiseries.py` 14, `test_create_train_X_y.py` multiseries 9, `test_bayesian_search_forecaster.py` 7...) | Fallan solo las `Series`/`DataFrame` con `object` explícito comparadas con texto; los `np.array(..., dtype=object)` se reinfieren a `str` |
| `np.dtype("O")` para `exog_dtypes_in_` | 4 (`test_create_train_X_y.py:917, 1002, 1963, 2196`) | Fallan: esperar `exog["col"].dtype` |
| Literales `dtype="datetime64[ns]"` | 18 en 7 ficheros (`test_backtesting_forecaster_multiseries.py` 8, `test_create_sample_weights.py` ×5 ficheros, `test_predict.py` de stats) | Fallan aunque haya `check_dtype=False`; quitar el literal |
| `pd.DatetimeIndex([cadenas])` / `pd.to_datetime([...])` / listas de `pd.Timestamp` | 95 / 9 / 3 (frente a 1721 `pd.date_range`, uno con `unit=`) | `Pandas4Warning` de `freq` y de `exact` en `assert_index_equal` (3.1); el esperado debe llevar `freq`; `check_freq` no existe en `assert_index_equal` de pandas 2 |
| `pytest.warns(FutureWarning)` / `DeprecationWarning` | 14 / 1, todos de avisos propios de skforecast | `Pandas4Warning` hereda de `DeprecationWarning`: un test que espere `DeprecationWarning` podría capturar el de pandas; revisar el único caso |
| `pd.option_context("future.no_silent_downcasting")` | 1 (`test_calculate_distance_from_holiday.py:416-417`) | Deprecado en 3.1; desaparece con el cambio de `_calendar.py:849` |
| Fixtures binarios | 10 (9 joblib, 1 parquet de pandas 2.2.2; 1,9 MB en total) | Cargan sin avisos en ambas versiones y conservan `object` y `[ns]`; `tests_population_drift/test_predict.py:218-219` falla por dtype de columnas |
| Parametrización por versión de pandas, `conftest.py` | ninguna (solo el `conftest` de `plot`, backend Agg) | - |
| `deep_learning`: 18 ficheros importan `keras` sin guardia | - | Sin cambio (la CI instala keras y torch) |

Cómo queda hoy la CI (`.github/workflows/`):

| Workflow | Disparo | Matriz | Instalación | pandas que obtiene hoy |
|---|---|---|---|---|
| `unit-tests.yml` | PR a `main` | Python 3.10 a 3.14 × ubuntu, macOS, Windows | `uv pip install -e ".[test]"` | 2.3.3 en todas (pin `<3.0`) |
| `unit-tests-min-deps.yml` | PR a `main` | Python 3.10, ubuntu | `uv pip install --resolution lowest-direct -e ".[test]"` | 2.2.0 con numpy 1.26.1 |
| `unit-tests-latest-deps.yml` | lunes 07:00 UTC | Python 3.12 a 3.14, ubuntu | `-e ".[test]"` y después `uv pip install --upgrade numpy pandas scikit-learn ...` | **3.1.0**: `uv pip install --upgrade pandas` no respeta el pin del paquete ya instalado (dry run verificado: "- pandas==2.3.3, + pandas==3.1.0"). El job semanal lleva ejecutando pandas 3 desde que salió 3.0.0 (enero de 2026): sus últimas 8 ejecuciones programadas (del 2026-09-07 al 2026-10-26, runs 10 a 17 sobre `main`) terminaron en `failure`; no se ha revisado si el motivo es solo pandas 3 |
| `codecov.yml` | push y PR a `main` | Python 3.14 | `-e ".[test]"` | 2.3.3 |
| `benchmarks.yml` | manual | ubuntu, `OMP_NUM_THREADS=1` | `benchmarks/requirements_benchmarks.txt` | 2.3.3 (pin exacto) |

Restricciones de pandas 3: Python ≥ 3.11 (v3.0.0 l. 868, metadatos de PyPI
`requires_python >=3.11` en 3.0.0, 3.0.6 y 3.1.0) y numpy ≥ 2.0.2 desde 3.1
(v3.1.0 l. 594; 3.0.x acepta numpy ≥ 1.26). pandas 2.2.x y 2.3.x funcionan con
numpy 2.x (los entornos `p22` y `p2` de este estudio usan numpy 2.5.4).

Propuesta de matriz con el pin `pandas>=2.2, <4.0` (sección 7):

- `unit-tests.yml`: sin cambios en la matriz. Con el nuevo pin `uv` resuelve
  pandas 3.x en Python 3.11 a 3.14 y pandas 2.3.x en 3.10 (pandas 3 no se
  instala en 3.10), así que los dos mayores quedan cubiertos sin pasos nuevos.
  Para no depender de esa resolución implícita, añadir una entrada explícita de
  pandas 2 en una versión moderna de Python:

  ```yaml
  matrix:
    python-version: ['3.10', '3.11', '3.12', '3.13', '3.14']
    platform: [ubuntu-latest, macos-26, windows-latest]
    pandas: ['']            # el que resuelva el pin
    include:
      - python-version: '3.12'
        platform: ubuntu-latest
        pandas: '<3'        # pandas 2.3.x en un Python que admite 3.x
  ...
  - name: Install Python dependencies
    run: |
      uv pip install -e ".[test]"
      if [[ -n "${{ matrix.pandas }}" ]]; then uv pip install "pandas${{ matrix.pandas }}"; fi
  ```

- `unit-tests-min-deps.yml`: sin cambios (Python 3.10, pandas 2.2.0, numpy
  1.26.1 siguen siendo los mínimos). Si se sube el mínimo de pandas 2 a 2.2.3,
  cambiar el pin, no el workflow.
- `unit-tests-latest-deps.yml`: sin cambios; con el pin nuevo el `--upgrade`
  instalará pandas 3.x (y avisará de los `Pandas4Warning` nuevos cada lunes,
  que es su función). Conviene añadir `-W error::pandas.errors.Pandas4Warning`
  a su `pytest` para que una deprecación nueva falle en ese job y no en el
  release (opcional; ver decisión D5).
- `codecov.yml`: Python 3.14 obtiene pandas 3.x; cobertura de las ramas de
  pandas 2 solo a través de `unit-tests.yml`. Aceptable.
- `benchmarks/requirements_benchmarks.txt`: mantener `pandas==2.3.3` hasta que
  la tabla de la sección 4 esté medida con pandas 3 en CI; después pasar a
  `pandas==3.1.x` y registrar la versión de pandas junto a los números (ver 3.6).
- Ninguna de las plataformas necesita cambios: pandas 3 publica ruedas para
  ubuntu, macOS y Windows en 3.11 a 3.14 (verificado en PyPI para 3.1.0).

### 3.6 Rendimiento

Protocolo (`bench_hotspots.py` y `common.py`): 20 benchmarks de los puntos
que dependen de los internals de pandas, cada uno con datos deterministas;
para cada variante, 2 ejecuciones de calentamiento y 5 medidas
(`time.perf_counter`) intercaladas A/B en el mismo proceso cuando se compara
un prototipo con el código actual (así las dos variantes ven el mismo estado
de la máquina); se dan la mediana y el mínimo. La memoria pico se mide en
una pasada aparte con `tracemalloc` (cuenta las reservas de numpy; mínimo de
2 repeticiones). Un proceso cada vez, con la máquina en reposo (agentes
detenidos; el arnés registra la carga media y la marca si supera 1,5: las
marcas que quedan en la tabla corresponden a los hilos BLAS del propio
proceso en los `fit` de `LinearRegression`). Un **control nulo** (el mismo
método reejecutado sin ningún cambio, `null_patch`) acota el ruido del A/B
entre 0,94 y 1,14 en benchmarks de 30 a 90 ms: diferencias menores no son
significativas. Entornos: Python 3.12.3, numpy 2.5.4, scikit-learn 1.9.1,
numba 0.68.0, 4 vCPU; pandas 2.3.3 frente a 3.1.0. Las salidas de todas las
variantes son idénticas (huella normalizada) en todos los casos.

**pandas 3 frente a pandas 2 con el código actual** (tabla completa en la
sección 4):

- Construcción de la matriz pública (`create_train_X_y`) en los cuatro
  forecasters que la crean en numpy: entre 1,6 y 3,1 veces más lenta y entre
  un 43 % y un 75 % más de memoria pico (`recursive_create_train_X_y_float`:
  29 → 89 ms, 50 → 87 MiB). Causa: el constructor de `DataFrame` copia el
  array en pandas 3 (3.1) y el `astype` fragmenta el frame igual que antes.
- `ForecasterRecursiveMultiSeries`: la matriz con bloque único es 1,17
  veces más lenta (93 frente a 79 ms) con la misma memoria; la ruta `concat`
  1,23 veces más lenta pero con un 20 % menos de memoria; la ruta con dicts y
  `onehot` igual de rápida y con un 27 % menos de memoria; `fit` 1,18 (ruido:
  BLAS) y `encoding=None` 1,11 con la misma memoria (D7: no consolidar).
- `fit` univariante 0,84 (más rápido en pandas 3), `predict` en bucle 1,40
  (153 frente a 110 ms por 200 llamadas + 1 de 500 pasos: unos 0,2 ms más por
  llamada, en `check_predict_input`, `expand_index` y los constructores),
  `expand_index`/`check_predict_input` aislados 1,13, backtesting 0,97 a
  1,41 (el de 1000 folds de un paso: 1,41, por el coste por llamada de
  `predict`), window features igual de rápidas con un 25 % menos de memoria.

**Prototipos** (A/B intercalado, mismo proceso):

- `train_matrix_combo` (`DataFrame(..., copy=False)` + `astype` solo de las
  columnas que cambian, saltándolo si no cambia ninguna) en **pandas 3**:
  `create_train_X_y` entre 0,47 y 0,74 del tiempo y entre 0,57 y 0,76 de la
  memoria en `ForecasterRecursive` y `ForecasterDirect` (todo float: 79 → 37
  ms y 87 → 50 MiB); en `ForecasterDirectMultiVariate` y el clasificador el
  tiempo no cambia (1,02 y 0,98) y la memoria baja un 30 %; `backtesting`
  0,90. En `predict` en bucle sale 1,18: la comprobación de dtypes por
  llamada cuesta más de lo que ahorra en frames pequeños, así que en
  `create_predict_X` no debe hacerse el escaneo (usar un indicador calculado
  en `fit`, o el `astype` perezoso de pandas 3 sin más).
- El mismo prototipo en **pandas 2** sale entre 1,07 y 1,59 veces más lento
  con memoria idéntica, por encima del ruido del control en los forecasters
  directos y el clasificador, y sin una causa identificada (el constructor
  no copia en pandas 2 en ningún caso y el `astype` hace el mismo trabajo).
  Conclusión práctica: **en pandas 2 no se toca la ruta actual**; el módulo
  de compatibilidad aplica `copy=False` en el constructor y el `astype`
  reducido solo bajo pandas 3, y bajo pandas 2 deja exactamente las llamadas
  de hoy (`astype(todas, copy=False)`). Así pandas 2 no puede empeorar y
  pandas 3 obtiene las ganancias medidas. La PR debe repetir esta medida con
  el cambio real en el fichero (sin monkeypatch).
- **pandas 3 con los cambios frente a pandas 2 con el código actual** (lo
  que verá un usuario al cambiar de versión una vez hecha la migración):

  | Benchmark | Tiempo p3 con cambios / p2 | Memoria p3 con cambios / p2 |
  |---|---|---|
  | `recursive_create_train_X_y` | 1,39 (101,9 frente a 73,3 ms) | 1,03 |
  | `recursive_create_train_X_y_float` | 1,25 (36,6 frente a 29,3 ms) | 1,00 |
  | `direct_create_train_X_y` | 1,74 (49,0 frente a 28,1 ms) | 1,09 |
  | `direct_multivariate_create_train_X_y` | 1,37 | 1,02 |
  | `classifier_create_train_X_y` | 1,52 | 1,00 |
  | `predict_loop` (sin cambio aplicable) | 1,40 | 1,08 |
  | `backtesting_recursive` (1000 folds de 1 paso) | 1,44 | 1,07 |
  | `recursive_fit_linear` | 0,84 | 1,00 |
  | multiseries `onehot` con dicts, ruta `concat`, window features | 1,00 a 1,23 | 0,73 a 0,80 |

  La memoria queda igualada; el tiempo restante es coste por llamada de
  pandas 3 (Copy-on-Write, constructores, indexación) que skforecast no
  puede evitar sin reescribir los caminos de `predict` en numpy puro, fuera
  del alcance de esta migración.
- `no_copy_keyword_both` (quitar `copy=False` también en pandas 2, lo que
  costaría no tener el helper): entre 1,10 y 1,90 veces más lento y hasta un
  22 % más de memoria en las matrices multiseries de pandas 2 (confirma la
  nota del encargo y el microbenchmark aislado: en pandas 2, `concat(axis=1,
  copy=False)` cuesta 0,2 ms frente a 45 ms y `astype(copy=False)` 10 frente
  a 72 ms para 200k × 30).
- `no_copy_keyword` en pandas 3 (quitar el keyword donde pandas 3 avisa):
  el microbenchmark aislado da tiempos iguales con y sin keyword (`concat`
  0,15 frente a 0,14 ms; `astype` 9,3 frente a 8,8 ms) y la memoria es
  idéntica, pero el A/B sobre las funciones completas de multiseries sale
  entre 1,05 y 1,59 veces más lento, sin causa identificada (el control nulo
  descarta el mecanismo de parcheo en `recursive/`, no en multiseries). Como
  el keyword hay que quitarlo de todos modos (desaparece en pandas 4), la PR
  2 debe medir el cambio real y, si el coste se confirma, sustituir los dos
  `pd.concat` por el ensamblado en numpy que ya usa la ruta de bloque único.
- `multiseries_y_copy` (copiar cada serie antes del transformador): 1,04 a
  1,10 en tiempo, memoria idéntica, dentro del ruido; se adopta por
  corrección (F26).


## 4. Tabla de medidas

Mediana y mínimo en milisegundos, memoria pico en MiB (`tracemalloc`); 5
medidas tras 2 de calentamiento; Python 3.12.3, numpy 2.5.4, scikit-learn
1.9.1, numba 0.68.0; pandas 2.3.3 (`p2`) y 3.1.0 (`p3`); máquina de 4 vCPU en
reposo. "Salida idéntica" compara la huella normalizada de las salidas.

### 4.1 Código actual, pandas 2 frente a pandas 3

| benchmark | p2.3.3 mediana ms | p2.3.3 mín ms | p2.3.3 pico MiB | p3.1.0 mediana ms | p3.1.0 mín ms | p3.1.0 pico MiB | ratio tiempo p3/p2 | ratio memoria p3/p2 | salida idéntica p2/p3 |
|---|---|---|---|---|---|---|---|---|---|
| ms_create_train_X_y_block | 79.3 | 78.0 | 71.0 | 92.9 | 87.0 | 69.6 | 1.17 | 0.98 | sí |
| ms_create_train_X_y_concat | 44.8 | 44.5 | 41.9 | 55.3 | 42.6 | 33.7 | 1.23 | 0.80 | sí |
| ms_create_train_X_y_onehot_dict | 133.5 | 124.1 | 136.4 | 133.5 | 117.2 | 99.9 | 1.00 | 0.73 | sí |
| ms_fit_linear | 301.4 | 290.1 | 136.0 | 356.1 | 285.2 | 136.1 | 1.18 | 1.00 | sí |
| ms_fit_linear_encoding_none | 243.6 | 233.2 | 133.6 | 270.2 | 256.9 | 128.6 | 1.11 | 0.96 | sí |
| ms_get_level_row_slices | 8.5 | 8.0 | 1.7 | 11.4 | 10.4 | 1.7 | 1.33 | 1.01 | sí |
| ms_predict_levels | 7.8 | 6.9 | 0.3 | 7.7 | 7.4 | 0.3 | 0.99 | 0.90 | sí |
| recursive_create_train_X_y | 73.3 | 71.1 | 62.2 | 150.4 | 142.9 | 93.1 | 2.05 | 1.50 | sí |
| recursive_create_train_X_y_float | 29.3 | 27.7 | 49.6 | 89.4 | 62.2 | 87.0 | 3.05 | 1.75 | sí |
| direct_create_train_X_y_float | 20.0 | 19.6 | 46.4 | 58.1 | 56.5 | 73.6 | 2.90 | 1.59 | sí |
| recursive_fit_linear | 226.8 | 197.6 | 93.8 | 190.1 | 185.6 | 93.8 | 0.84 | 1.00 | sí |
| direct_create_train_X_y | 28.1 | 27.3 | 52.5 | 62.9 | 52.5 | 75.1 | 2.24 | 1.43 | sí |
| direct_multivariate_create_train_X_y | 51.9 | 48.5 | 65.9 | 81.7 | 77.7 | 95.6 | 1.57 | 1.45 | sí |
| classifier_create_train_X_y | 36.2 | 34.5 | 30.3 | 62.0 | 61.1 | 45.0 | 1.71 | 1.48 | sí |
| predict_loop | 109.5 | 104.2 | 0.6 | 153.1 | 136.3 | 0.7 | 1.40 | 1.08 | sí |
| expand_index_check_predict_input | 1148.5 | 1090.3 | 8.8 | 1292.7 | 1098.4 | 9.0 | 1.13 | 1.03 | sí |
| backtesting_recursive | 725.3 | 681.1 | 10.9 | 1022.7 | 972.9 | 11.7 | 1.41 | 1.07 | sí |
| backtesting_recursive_refit | 1309.6 | 1122.4 | 10.7 | 1273.7 | 1199.3 | 10.8 | 0.97 | 1.01 | sí |
| backtesting_multiseries | 1929.9 | 1828.0 | 21.9 | 2098.0 | 2052.0 | 22.5 | 1.09 | 1.03 | sí |
| window_features | 145.9 | 133.6 | 18.3 | 145.4 | 120.2 | 13.7 | 1.00 | 0.75 | sí |

### 4.2 Prototipo `train_matrix_combo` (constructor sin copia + `astype` de las columnas que cambian), pandas 3

| benchmark | current mediana ms | patched mediana ms | ratio | current mín ms | patched mín ms | current pico MiB | patched pico MiB | ratio memoria | salida idéntica |
|---|---|---|---|---|---|---|---|---|---|
| recursive_create_train_X_y | 137.7 | 101.9 | 0.741 | 123.5 | 95.9 | 93.1 | 63.8 | 0.685 | sí |
| recursive_create_train_X_y_float | 78.5 | 36.6 | 0.466 | 64.4 | 34.0 | 87.0 | 49.7 | 0.571 | sí |
| direct_create_train_X_y_float | 49.6 | 35.2 | 0.709 | 45.7 | 29.9 | 73.6 | 50.8 | 0.691 | sí |
| direct_create_train_X_y | 69.0 | 49.0 | 0.710 | 50.7 | 45.9 | 75.1 | 57.0 | 0.759 | sí |
| direct_multivariate_create_train_X_y | 70.1 | 71.3 | 1.016 | 65.2 | 52.0 | 95.6 | 67.0 | 0.701 | sí |
| classifier_create_train_X_y | 56.1 | 55.0 | 0.980 | 54.3 | 52.8 | 45.0 | 30.4 | 0.676 | sí |
| predict_loop | 148.4 | 174.9 | 1.178 | 139.6 | 136.0 | 0.7 | 0.8 | 1.177 | sí |
| backtesting_recursive | 1158.5 | 1044.3 | 0.901 | 985.2 | 985.7 | 11.7 | 11.8 | 1.006 | sí |

### 4.3 El mismo prototipo en pandas 2 (no se adopta en pandas 2: ver 3.6)

| benchmark | current mediana ms | patched mediana ms | ratio | current mín ms | patched mín ms | current pico MiB | patched pico MiB | ratio memoria | salida idéntica |
|---|---|---|---|---|---|---|---|---|---|
| recursive_create_train_X_y | 79.4 | 90.1 | 1.134 | 74.7 | 88.8 | 62.2 | 62.3 | 1.001 | sí |
| recursive_create_train_X_y_float | 25.8 | 32.6 | 1.263 | 24.5 | 32.2 | 49.6 | 49.7 | 1.002 | sí |
| direct_create_train_X_y_float | 22.3 | 31.4 | 1.409 | 19.7 | 25.7 | 46.4 | 46.3 | 0.997 | sí |
| direct_create_train_X_y | 30.5 | 48.4 | 1.587 | 28.1 | 42.3 | 52.5 | 52.5 | 1.001 | sí |
| direct_multivariate_create_train_X_y | 57.7 | 69.4 | 1.204 | 50.2 | 56.2 | 65.9 | 65.9 | 1.000 | sí |
| classifier_create_train_X_y | 34.0 | 49.5 | 1.457 | 33.9 | 43.1 | 30.3 | 30.4 | 1.003 | sí |
| predict_loop | 104.4 | 113.9 | 1.090 | 101.5 | 112.6 | 0.6 | 0.7 | 1.150 | sí |
| backtesting_recursive | 714.8 | 760.9 | 1.065 | 676.6 | 657.7 | 11.0 | 11.0 | 1.007 | sí |

### 4.4 Control nulo en pandas 2 (mismo código reejecutado sin cambios: ruido del A/B)

| benchmark | current mediana ms | patched mediana ms | ratio | current mín ms | patched mín ms | current pico MiB | patched pico MiB | ratio memoria | salida idéntica |
|---|---|---|---|---|---|---|---|---|---|
| recursive_create_train_X_y | 88.5 | 89.0 | 1.006 | 78.7 | 82.2 | - | - | - | sí |
| recursive_create_train_X_y_float | 33.6 | 32.1 | 0.957 | 26.1 | 28.5 | - | - | - | sí |
| direct_create_train_X_y | 31.2 | 35.5 | 1.136 | 30.4 | 34.6 | - | - | - | sí |
| classifier_create_train_X_y | 44.9 | 42.2 | 0.939 | 38.5 | 40.0 | - | - | - | sí |

### 4.5 Quitar `copy=False` también en pandas 2 (`no_copy_keyword_both`): el coste de no tener el helper

| benchmark | current mediana ms | patched mediana ms | ratio | current mín ms | patched mín ms | current pico MiB | patched pico MiB | ratio memoria | salida idéntica |
|---|---|---|---|---|---|---|---|---|---|
| ms_create_train_X_y_block | 84.2 | 118.7 | 1.410 | 78.1 | 106.2 | 71.0 | 71.2 | 1.003 | sí |
| ms_create_train_X_y_concat | 43.1 | 82.0 | 1.900 | 41.2 | 76.6 | 41.9 | 51.2 | 1.222 | sí |
| ms_create_train_X_y_onehot_dict | 143.4 | 168.9 | 1.178 | 130.6 | 149.3 | 136.4 | 136.6 | 1.001 | sí |
| ms_fit_linear | 315.1 | 399.0 | 1.266 | 282.4 | 322.6 | 136.0 | 136.2 | 1.001 | sí |
| ms_fit_linear_encoding_none | 239.3 | 262.9 | 1.098 | 230.5 | 256.9 | 133.6 | 133.8 | 1.001 | sí |

### 4.6 Quitar `copy=False` solo en pandas 3 (`no_copy_keyword`)

| benchmark | current mediana ms | patched mediana ms | ratio | current mín ms | patched mín ms | current pico MiB | patched pico MiB | ratio memoria | salida idéntica |
|---|---|---|---|---|---|---|---|---|---|
| ms_create_train_X_y_block | 89.4 | 120.9 | 1.352 | 83.8 | 113.1 | 69.6 | 69.8 | 1.003 | sí |
| ms_create_train_X_y_concat | 45.0 | 68.6 | 1.523 | 41.6 | 65.8 | 33.6 | 33.8 | 1.006 | sí |
| ms_create_train_X_y_onehot_dict | 124.6 | 197.7 | 1.587 | 117.4 | 152.1 | 99.9 | 100.0 | 1.002 | sí |
| ms_fit_linear | 325.3 | 346.8 | 1.066 | 310.1 | 306.3 | 136.1 | 136.3 | 1.002 | sí |
| ms_fit_linear_encoding_none | 290.0 | 304.0 | 1.048 | 259.8 | 270.7 | 128.5 | 128.8 | 1.002 | sí |

### 4.7 Copia de cada serie antes del transformador (`multiseries_y_copy`)

pandas 2:

| benchmark | current mediana ms | patched mediana ms | ratio | current mín ms | patched mín ms | current pico MiB | patched pico MiB | ratio memoria | salida idéntica |
|---|---|---|---|---|---|---|---|---|---|
| ms_create_train_X_y_block | 74.9 | 82.4 | 1.099 | 72.1 | 74.8 | 71.0 | 71.0 | 1.000 | sí |
| ms_fit_linear | 307.3 | 318.0 | 1.035 | 296.3 | 304.3 | 136.0 | 136.0 | 1.000 | sí |

pandas 3:

| benchmark | current mediana ms | patched mediana ms | ratio | current mín ms | patched mín ms | current pico MiB | patched pico MiB | ratio memoria | salida idéntica |
|---|---|---|---|---|---|---|---|---|---|
| ms_create_train_X_y_block | 87.8 | 93.7 | 1.067 | 84.7 | 86.1 | 69.6 | 69.6 | 1.000 | sí |
| ms_fit_linear | 317.2 | 346.3 | 1.092 | 282.2 | 284.5 | 136.1 | 136.1 | 1.000 | sí |

Identidad de salidas entre pandas 2 y pandas 3 y entre antes y después de
cada cambio: sección 5 (106 de 106 salidas con los mismos valores).

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
| pandas 2.2.3 frente a 2.3.3 | 106 | 106 | 0 | 0 |
| p2 frente a p3 (índice de entrada tal cual: `[ns]` frente a `[us]`) | 106 | 37 | 69 | 0 |
| p2 frente a p3 con índice de entrada `[ns]` | 106 | 79 | 27 | 0 |
| p2 frente a p2 con todos los cambios propuestos (`all_proposed`, sección 7) | 106 | 106 | 0 | 0 |
| p3 frente a p3 con todos los cambios propuestos | 106 | 106 | 0 | 0 |
| p2 con los cambios frente a p3 con los cambios, índice `[ns]` | 106 | 79 | 27 | 0 |

Avisos de pandas emitidos por el paquete durante los 21 casos con pandas 3:
12 `Pandas4Warning` con el código actual (`copy` keyword ×7, `isinstance(...,
DateOffset)` ×5 en `ForecasterEquivalentDate`, `dayofweek` ×1) y **0** con
todos los cambios propuestos.

**Conclusión: todos los valores son idénticos bit a bit entre pandas 2.2.3,
2.3.3 y 3.1.0, antes y después de los cambios propuestos**, en los 21 casos
(predicciones, intervalos, residuos, residuos por bin, métricas, matrices,
slices por nivel, tablas de búsqueda, predicciones tras guardar y cargar). La
única excepción conocida está fuera de estos casos y se describe en 3.4: las
métricas escaladas (MASE, RMSSE) de `backtesting_forecaster_multiseries` con
`ForecasterDirectMultiVariate` (o `ForecasterRnn`) y un `DataFrame` ancho
cuya serie objetivo tiene NaN en el entrenamiento, por el cambio de
`DataFrame.stack()`; el valor de pandas 3 es el correcto y se propone
adoptarlo en ambas versiones. Las 27 diferencias que quedan con índice `[ns]`
son solo de dtype:

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

Todos vienen de pandas, no de una decisión de skforecast, y solo aparecen al
ejecutar con pandas 3. Ninguno cambia un valor numérico (sección 5).

| Cambio | Qué ve el usuario | Normalizar en la librería o aceptar |
|---|---|---|
| Resolución `datetime64[us]` | Los índices de predicciones, matrices de entrenamiento, backtesting y `last_window_` tienen la unidad de sus datos de entrada; con pandas 3 la mayoría de las construcciones habituales dan `[us]` (3.3). Comparar un índice `[us]` con uno `[ns]` con `pd.testing.assert_index_equal` falla por dtype; `==`/`.equals` entre unidades compara bien los valores | **Aceptar.** Convertir a `[ns]` en cada salida iría contra el comportamiento de pandas 3, costaría una conversión por llamada y rompería la coherencia con la entrada del usuario. Documentar |
| dtype `str` en columnas de texto | `exog_dtypes_in_` y `exog_dtypes_out_` guardan `StringDtype(na_value=nan)` (repr `str`) en vez de `object` para columnas de texto; la columna `level` de `predict`, `predict_interval`, `predict_bootstrapping` y de las predicciones de `backtesting_forecaster_multiseries`, la columna `levels` de sus métricas, y el `Index` de columnas de `X_train` son `str`. `X_train.columns.dtype == object` deja de ser cierto | **Aceptar.** Es el dtype por defecto de pandas 3 para cualquier texto; forzar `object` perdería información y costaría una conversión. Lo que sí hay que garantizar (y ya ocurre) es que la detección de categóricas y las comprobaciones de dtype no dependan de `object` (3.2) |
| Mensajes de error que imprimen tipos de índice | `<class 'pandas.RangeIndex'>` en vez de `<class 'pandas.core.indexes.range.RangeIndex'>` (v3.0.0 l. 850); `Found <class 'pandas.RangeIndex'>` para el segundo nivel de un `MultiIndex` creado desde `range` | **Aceptar** (texto generado por pandas). Opcional: imprimir `type(x).__name__` en `utils.py:4373, 4555, 4579-4580` y `_preprocessing.py:857-858` para que el texto sea estable entre versiones |
| Orden de las frecuencias en el mensaje "Found frequencies: [...]" | `[<2 * Days>, <Day>]` en vez de `[<Day>, <2 * Days>]` cuando hay offsets `Day` (ya no ordenables) | **Normalizar** con una clave de ordenación determinista (`utils.py:4466-4471` y `_split.py:383-388`): mismo texto en las dos versiones, sin rama |
| Índices diarios tz-aware construidos con `tz_convert` | Llegan sin `freq` y `fit`, `TimeSeriesFold` y `predict` lanzan `ValueError` pidiendo una frecuencia; `idx.freq = "D"` también lanza en pandas 3 | **No se puede evitar en la librería** sin adivinar la convención del usuario. Documentar la forma correcta en pandas 3 (`freq="24h"` para pasos fijos de 24 h en UTC, o `asfreq("D")` sobre la hora local) en `troubleshooting-common-errors` y en la guía de datos; `expand_index` recupera por su cuenta los casos `MS`/`QS` anclados en UTC (3.3) |
| Ficheros joblib escritos con pandas 3 | No se cargan con pandas 2 (`AssertionError`/`NotImplementedError` al reconstruir índices temporales o columnas `category`, es un límite de pandas); los de pandas 2 sí se cargan con pandas 3 | **Documentar** en `save_forecaster`/`load_forecaster` y en la guía de persistencia: para compartir entre versiones, `backend="skops"` funciona en los dos sentidos |
| MASE y RMSSE de `backtesting_forecaster_multiseries` con `ForecasterDirectMultiVariate`/`ForecasterRnn`, `DataFrame` ancho y NaN en la serie objetivo | Valores distintos entre pandas 2 y 3 (3.4); con pandas 3, los NaN del entrenamiento se respetan como en el camino dict | **Normalizar** con `stack(future_stack=True)`: pandas 2 pasa a dar el valor de pandas 3. Es un cambio de resultado en pandas 2 para ese caso: nota de versión (D12) |
| Avisos `Pandas4Warning` | Ninguno tras el plan de la sección 7 | - |

Texto propuesto para `docs/releases/releases.md` (sección de la versión en
desarrollo, estilo del fichero: viñetas cortas con badge y enlace a la PR,
más un bloque "Before upgrading"):

```markdown
+ <span class="badge text-bg-enhancement">Enhancement</span> skforecast supports pandas 3 (3.0 and 3.1) and keeps supporting pandas 2.2 and 2.3. No results change between the two majors: predictions, intervals, residuals, metrics and training matrices are bit-identical. With pandas 3, text columns are stored and returned with the `str` dtype instead of `object`, and the index of the outputs keeps the resolution of the input index (`datetime64[us]` for dates parsed from strings, the pandas 3 default), as explained in the user guide. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))

+ <span class="badge text-bg-fix">Fix</span> <code>[ForecasterRecursiveMultiSeries]</code> with a `transformer_series` that modifies its input in place (`StandardScaler(copy=False)`) stored the transformed values in `last_window_` with pandas 2 and predicted wrong values. Each series is now copied before transforming it, as <code>[ForecasterRecursive]</code> already did. ([#XXXX](...))

+ <span class="badge text-bg-fix">Fix</span> `expand_index` and the forecasters infer the frequency of a timezone-aware index anchored in UTC (fixed steps of 24 hours, month or quarter starts in UTC) in UTC, so the prediction index with `MS` or `QS` is right with pandas 3, where `tz_convert` no longer keeps the frequency of calendar-day offsets. ([#XXXX](...))

!!! warning "Before upgrading"

    With pandas 3, a daily timezone-aware index created with
    `pd.date_range(..., freq="D", tz="UTC").tz_convert(tz)` has no frequency
    (pandas 3 made `Day` a calendar-day offset, which `tz_convert` cannot keep).
    skforecast requires a frequency: use `freq="24h"` for fixed steps of 24
    hours in UTC, or `.asfreq("D")` on the local index for calendar days. The
    results with `freq="24h"` are identical to those of `freq="D"` with pandas 2.

    Files saved with `save_forecaster(..., backend="joblib")` under pandas 3
    cannot be loaded under pandas 2 (a pandas limitation); files saved under
    pandas 2 load under pandas 3. Use `backend="skops"` to share forecasters
    between the two.
```

Las guías de usuario que cambian: `docs/user_guides/` (datos de entrada y
frecuencias; persistencia; categóricas), `skills/troubleshooting-common-errors`
(índice sin frecuencia en pandas 3, joblib entre versiones) y
`skills/forecasting-single-series` (nota sobre `freq="24h"`); además
`tools/ai/llms-base.txt` (dependencias e instalación) y
`tools/ai/ai_context_header.md` (línea de dependencias), con la regeneración
de los ficheros de contexto de IA (`python tools/ai/generate_ai_context_files.py`
y `--check`). Detalle en la sección 7.

## 7. Plan de implementación

Ordenado de menor a mayor riesgo. Cada PR se abre contra `0.27.x`, pasa
`verify` (lint, tests tocados, contexto de IA) y, antes de mezclar, la
comprobación de identidad de la sección 5 en los dos entornos (huellas
iguales a las del código anterior en pandas 2 y 3) y los tests que cambian
ejecutados con pandas 2.2.0, 2.3.3 y 3.1.0. Esfuerzo total estimado: 3 a 4
días de trabajo (la mitad en los tests), más un ciclo de CI por PR.

| # | PR | Cambios (fichero:línea) | Tests que cambian | Verificación | Bit a bit en pandas 2 |
|---|---|---|---|---|---|
| 1 | Renombrados sin cambio de comportamiento | `dayofweek`/`dayofyear` → `day_of_week`/`day_of_year` en `preprocessing/_calendar.py:199-202` y `foundation/_adapters.py:4995-5001`; `isinstance(x, pd.DateOffset)` → `pd.offsets.BaseOffset` en `model_selection/_split.py:304, 1051`, `model_selection/_utils.py:319-321`, `recursive/_forecaster_equivalent_date.py:218, 382, 412, 506, 670, 892`, `drift_detection/_population_drift.py:238, 288, 318, 394`; `'freq': 'Q'` → `'QE'` en `datasets/datasets.py:209`; `freq='d'` → `'D'` en `experimental/_splitter.py:67`; `groupby(..., observed=True)` en `_population_drift.py:682, 857` y `foundation/_foundation_model.py:738`; `fillna(False)` + `future.no_silent_downcasting` → ruta numpy en `_calendar.py:849-850`; `stack(future_stack=True)` en `model_selection/_search.py:1576, 1935, 2036, 2456` | Fixtures y tests que usan `dayofweek` (`fixtures_model_selection_multiseries.py:140-143`, 10 tests de `direct`/`recursive` listados en 2.3); el test de `_calendar` que usa `future.no_silent_downcasting`; el test de `fetch_dataset` de `australia_tourism` si comprueba `freq` | Tests tocados en los tres pandas; identidad | Sí |
| 2 | Módulo de compatibilidad y keyword `copy` | Nuevo `skforecast/utils/_pandas_compat.py` (`PANDAS_GE_3`, `concat`, `astype`, `frame_from_array`, `writable_array`, `is_string_dtype`, `infer_freq`; borrador en `prototypes/pandas_compat_draft.py`); `pd.concat(copy=False)` → `compat.concat` en `_forecaster_recursive_multiseries.py:1333, 1567`; `astype(..., copy=False)` → `compat.astype` (bajo pandas 2 exactamente la llamada actual, `astype(todas, copy=False)`; bajo pandas 3 solo las columnas cuyo dtype cambia, saltando la llamada si no cambia ninguna, y en `create_predict_X` sin escaneo por llamada: indicador calculado en `fit` o `astype` perezoso) en `_forecaster_recursive.py:1109, 2042`, `_forecaster_recursive_classifier.py:1129, 1843`, `_forecaster_direct.py:1520, 2407`, `_forecaster_direct_multivariate.py:1637, 2586`, `_forecaster_recursive_multiseries.py:3552`, `_preprocessing.py:774`; `Series.astype("float64", copy=False)` → `to_numpy(dtype="float64")` en `_population_drift.py:502`; `pd.infer_freq` → `compat.infer_freq` en `utils.py:2458` y `_calendar.py:862` (con `to_offset(...).freqstr` en `_freq_to_timedelta_unit`) | Ninguno (sin cambio de resultados) | Identidad en p2 y p3; `-W error::pandas.errors.Pandas4Warning` sobre `recursive/`, `direct/`, `preprocessing/` en p3; benchmark de 3.6 (`train_matrix_combo` en p2: tiempo y memoria iguales o mejores) | Sí (medido) |
| 3 | Copias evitables | `compat.frame_from_array` (`copy=False` solo bajo pandas 3; bajo pandas 2 el constructor por defecto, que ya no copia) sobre arrays privados en `create_train_X_y` de `_forecaster_recursive.py:1100-1104` (materializando antes la vista de lags cuando no hay exógenas: `np.ascontiguousarray`), `_forecaster_recursive_classifier.py:1120-1124`, `_forecaster_direct.py:1497-1510`, `_forecaster_direct_multivariate.py:1614-1627`; `copy=False` en los `DataFrame`/`Series` de los caminos de `predict` listados en 3.1; `y.to_numpy(copy=True)` en `_forecaster_recursive_multiseries.py:990` (corrección F26, nota de versión); copias redundantes a `copy(deep=False)` o eliminadas (`_forecaster_equivalent_date.py:640, 668, 672, 868, 894`, `_forecaster_recursive_multiseries.py:2756-2761, 2793-2795, 2934-2935`, `_range_drift.py:427-428`, `utils.py:812-818, 4387-4392`, `_preprocessing.py:538-540, 656-657, 674, 744`) | Test nuevo para F26 (`transformer_series=StandardScaler(copy=False)` predice igual que `copy=True`); test de que `create_train_X_y` sin exógenas devuelve una matriz escribible | Identidad; benchmark 3.6 (`dataframe_copy_false`, `multiseries_y_copy`); tracemalloc de las copias eliminadas | Sí (medido); F26 cambia solo el caso con transformador in situ |
| 4 | Robustez con pandas 3 (código) | Clave de ordenación de offsets en `_split.py:383-388` (con `try/except` como `utils.py:4466-4471`) y en `utils.py:4466-4471`; guardia `freq is None` en `_preprocessing.py:471`; `object`/`str` equivalentes en `check_preprocess_exog_multiseries` (`utils.py:4645-4660`); inferencia en UTC en `expand_index` (`utils.py:2456-2463`) y, si se adopta D6, mensaje con `freq="24h"` en `utils.py:2047-2051`; `pd.Series(dtype=str)` en `_validation.py:2850`; `dtype=object` antes del `fillna` en `_forecaster_recursive_multiseries.py:2026-2056`; `stack(future_stack=True)` en `_utils.py:1536` (D12, nota de versión); mensaje claro en `load_forecaster` para ficheros joblib de pandas 3 cargados en pandas 2 (D8) | Tests nuevos: `_split.py` con `Day`/`MonthBegin` y `Day`/paso (hoy `TypeError` también en pandas 2), `reshape_series_wide_to_long` sin `freq`, exógena de texto `object`/`str` entre series, `expand_index` `MS`/`QS` anclado en UTC sin `freq`, MASE/RMSSE con NaN en la serie objetivo ancha | Tests nuevos en los tres pandas; identidad | Sí salvo D12 y los mensajes de `_split.py` (antes `TypeError`) |
| 5 | Tests neutrales entre versiones | `STR_DTYPE = pd.Series(["a"]).dtype` compartido y expectativas con él (19 tests de G2-RC1, ficheros en 2.2); `columns=pd.Index(float_cols, dtype=STR_DTYPE)` y `check_column_type=False` o fixture regenerado (G2-RC2, 7); mensajes construidos con `f"{pd.RangeIndex}"`/`{type(x.index.levels[1])}` (G2-RC3/RC4, 6); `unit="ns"` en las expectativas que comparan con fixtures persistidos (G1-RC1a, 38: `test_create_predict_inputs.py` 8 `date_range`, `test_predict.py`, `test_create_predict_X.py`, `test_predict_interval*.py`, `test_backtesting_forecaster_multiseries.py`, `test_predict_and_calculate_metrics_one_step_ahead_multiseries.py`) y sin el pin `[ns]` donde la entrada se construye con `date_range` (G1-RC1b, 7); `unit="ns"` y `tz_zoneinfo` calculado en `test_skops_decompose_compose.py:60-90`; `freq="24h"` en los 7 tests de índices anclados en UTC (G3-RC1a) y en `test_reshape_series_wide_to_long.py:72-74`; `exact=`/`check_freq=` explícitos donde 3.1 avisa | 90 tests (la lista exacta en `results/baseline_pandas3.1.0/failures_by_group.json`) | Suite completa en p3 **en verde** (salvo torch) y en p2; `unit-tests-min-deps.yml` | n/a |
| 6 | `pyproject.toml` y CI | `pandas>=2.2, <4.0`; `"pandas[parquet]>=2.2"` sin cambio; matriz de 3.5 (`include` de pandas 2 en Python 3.12; opcional `-W error::pandas.errors.Pandas4Warning` en `unit-tests-latest-deps.yml`); `benchmarks/requirements_benchmarks.txt` según D9 | - | CI en verde en las 15 combinaciones; `unit-tests-min-deps.yml` | n/a |
| 7 | Documentación y contexto de IA | `docs/releases/releases.md` (texto de la sección 6); guías de usuario de datos de entrada (resolución y `freq="24h"`), persistencia (joblib frente a skops entre versiones) y categóricas (`str`); `skills/troubleshooting-common-errors` (índice sin frecuencia en pandas 3; joblib entre versiones; `.as_unit` en pandas 2), `skills/forecasting-single-series` (nota `24h`), `skills/drift-detection/SKILL.md:96` (`'ME'`); `tools/ai/llms-base.txt` y `tools/ai/ai_context_header.md` (línea de dependencias: `pandas>=2.2,<4`); `python tools/ai/generate_ai_context_files.py` y `--check`; reejecutar los notebooks afectados cuando el entorno de docs use pandas 3 (3 notebooks imprimen `[ns]`, 4 `dtype: object`) | - | `ai-context-check.yml`, `docs-page-meta-check.yml` | n/a |
| 8 | Opcionales medidos | Consolidar tras `drop` con `encoding=None` solo si 3.6 lo justifica (D7); descomponer `pd.StringDtype` en `_decompose_dtype` para skops neutros (3.2); mensajes con `type(x).__name__` (D10) | Según cambio | Identidad; benchmark | Sí |

Qué queda para pandas 4 (no bloquea ahora): quitar el helper `copy=` del
módulo de compatibilidad y el `copy=False` de los constructores cuando se
abandone pandas 2; `pd.infer_freq` devolverá offsets (el helper ya lo
absorbe); `future.infer_string` y `mode.copy_on_write` desaparecen (no se
usan); la lista de tipos confiables de skops incluirá `pandas.StringDtype`.

Orden recomendado: 1 y 2 juntas abren la compatibilidad (con ellas pandas 3
ya no emite ningún `Pandas4Warning` desde el paquete salvo los avisos de
tests); 4 y 5 ponen la suite en verde; 6 cambia el pin cuando 5 está
mezclada; 3, 7 y 8 pueden ir en paralelo. Las PR 1 a 5 no cambian ninguna
salida en pandas 2 salvo las tres correcciones señaladas (F26, D12 y el
`TypeError` de `_split.py`), que llevan nota de versión.

## 8. Decisiones que debes tomar tú

| Id | Decisión | Opciones | Recomendación |
|---|---|---|---|
| D1 | Pin de pandas y de Python | (a) `pandas>=2.2, <4.0` y `requires-python >=3.10` (pandas 3 solo se instala en 3.11+, pandas 2.3 queda para 3.10); (b) además subir `requires-python` a 3.11 y `pandas>=2.2`; (c) esperar a 0.28 | **(a)**: no deja fuera a nadie, la CI cubre ambos mayores (3.5) y el código no necesita ramas salvo el módulo de compatibilidad. Subir Python a 3.11 no aporta nada hoy (la rama 3.10 + pandas 2.3 seguirá existiendo mientras numpy y scikit-learn la soporten) |
| D2 | Mínimo de pandas 2 | (a) mantener 2.2.0; (b) subir a 2.2.3 (última 2.2, 2024-09); (c) subir a 2.3 (última serie 2.x, la única que aún recibe correcciones) | **(a)** para esta migración: nada del plan necesita una API posterior a 2.2 (`date_range(unit=)`, `as_unit`, `day_of_week`, `pd.offsets.BaseOffset`, `stack(future_stack=True)` existen en 2.2), `unit-tests-min-deps.yml` sigue en verde con 2.2.0 y numpy 1.26.1, y las 106 salidas del estudio son idénticas en 2.2.3 y 2.3.3. (c) solo aporta `StringDtype(na_value=nan)` (opción de skops neutros, 3.2) y una serie menos que mantener, a cambio de dejar fuera entornos fijados en 2.2; si se quiere, anunciarlo para 0.28 o para cuando se retire Python 3.10 |
| D3 | Dónde vive la rama por versión | (a) un módulo `skforecast/utils/_pandas_compat.py` con `PANDAS_GE_3` y los helpers (`concat`, `astype`, `frame_from_array`, `writable_array`, `is_string_dtype`, `infer_freq`); (b) comprobaciones `pd.__version__` donde hagan falta; (c) sin rama, aceptando la copia extra de `astype` en pandas 2 | **(a)**; con el prototipo `astype_changed_only` la única rama real que queda es la de `infer_freq` (opción `future.infer_freq_returns_offset`) |
| D4 | dtype `str` en salidas y atributos | (a) aceptar y documentar; (b) normalizar a `object` en `exog_dtypes_in_`, columna `level`, columnas de `X_train` | **(a)**: (b) cuesta conversiones, va contra pandas 3 y no cambia ningún valor |
| D5 | Avisos de deprecación en CI | (a) `-W error::pandas.errors.Pandas4Warning` solo en `unit-tests-latest-deps.yml` (semanal); (b) también en `unit-tests.yml`; (c) nada | **(a)**: el job semanal avisa de deprecaciones nuevas sin bloquear PRs; con el plan de la sección 7 el paquete no emite ninguna hoy |
| D6 | Índices tz-aware anclados en UTC sin `freq` (G3-RC1a) | (a) documentar `freq="24h"` y mejorar el mensaje de error de `check_extract_values_and_index` (`utils.py:2047-2051`) con esa sugerencia; (b) además inferir la frecuencia en UTC en las validaciones (como el arreglo de `expand_index`) y aceptar la serie sin `freq`; (c) nada | **(a)**: la librería ya exige `freq` explícita por diseño; (b) adivina la convención del usuario y añade una rama en cuatro sitios |
| D7 | `encoding=None` con exógenas: dos copias en pandas 3 por el `drop` perezoso | (a) nada; (b) consolidar tras `drop` cuando `nblocks > 1` | Según 3.6: si la diferencia medida es menor del 10% del `fit`, **(a)** |
| D8 | Persistencia entre versiones | (a) documentar (joblib: solo hacia delante; skops: ambos sentidos); (b) añadir una comprobación en `load_forecaster` que convierta el `AssertionError`/`NotImplementedError` de pandas en un mensaje claro | **(a) + (b)**: (b) son unas líneas en `utils.py:3621-3700` y evita un traceback opaco |
| D9 | Benchmarks (`benchmarks/`) | (a) mantener `pandas==2.3.3` hasta la primera release con pandas 3 y después pasar a 3.1.x; (b) dos entornos | **(a)**, registrando la versión de pandas en los resultados |
| D10 | Mensajes de error con tipos de índice y orden de offsets (G2-RC3, RC5) | (a) aceptar el texto de pandas 3; (b) estabilizar (`type(x).__name__`, clave de ordenación) | **(b)** solo para el orden de offsets (una línea, mismo texto en ambas); (a) para los nombres de tipo |
| D11 | Dónde aplicar el estudio | (a) todo en `0.27.x`; (b) solo la compatibilidad (PR 1 a 4 de la sección 7) en 0.27 y las optimizaciones en 0.28 | **(a)** si 0.27 no está cerrada: las optimizaciones son pequeñas y están medidas; si no, (b) |
| D12 | `stack()` con NaN en las métricas multiseries (3.4) | (a) `stack(future_stack=True)` en `_utils.py:1536` y `_search.py` (pandas 2 adopta el valor de pandas 3 para MASE/RMSSE con NaN en la serie objetivo ancha; nota de versión); (b) `dropna()` tras el `stack` (pandas 3 adopta el valor de pandas 2, que cierra los huecos del entrenamiento); (c) nada (resultados distintos por versión) | **(a)**: es el valor del camino dict y el que respeta los huecos; (c) rompe la promesa de resultados iguales entre versiones |

## 9. Preguntas abiertas y lo no verificado

- **Los 42 fallos del 2026-10-04 con pandas 3.0.6.** En el árbol actual los 90
  fallos relacionados con pandas se reproducen con 3.0.6 (entorno con las
  mismas versiones del resto). Parte de la diferencia son tests añadidos
  después (PR #1405, 2026-10-15, índices anclados en UTC), pero el verificador
  de G3 comprobó que el test de `_split.py:387` ya existía el 2026-10-03, así
  que no todo se explica por fecha. No se ha vuelto a ejecutar la suite
  completa con 3.0.6 (solo los 90 tests).
- **Deep learning y foundation.** `skforecast/deep_learning` no se ha
  ejecutado (keras/torch no instalados) y los 10 tests de `TSICLAdapter`
  tampoco; el inventario estático de `_forecaster_rnn.py` y `_adapters.py`
  es de lectura, no de ejecución. `_adapters.py:4995-5001` usa
  `dayofweek`/`dayofyear` (aviso en 3.1) y crea índices sintéticos con
  `pd.date_range("2000-01-01", ...)` (`:2597, :2630, :3266, :3299`), que
  serán `[us]` en pandas 3; no se ha comprobado si algún adaptador compara
  unidades.
- **`datasets.py:820/870`** (`pd.to_datetime(df.index, format=...)`) devolverá
  índices `[us]` con pandas 3; requiere red y no se ha ejecutado. Afecta a los
  notebooks de la documentación (sus salidas impresas mostrarán `[us]` cuando
  se vuelvan a ejecutar con pandas 3).
- **scikit-learn 1.6** (mínimo soportado): `check_array(force_writeable=True)`
  copia las vistas de solo lectura en 1.9.1; en 1.6 no se ha verificado. Los
  estimadores que escriben en `X` ya están prohibidos por la comprobación de
  `utils.py:853-896`, así que el riesgo es bajo.
- **Memoria pico con `rss_hwm_increase_mib`**: el alto de agua de RSS se
  reinicia escribiendo en `/proc/self/clear_refs`, que no está disponible en
  todos los sistemas; los números de la sección 4 usan `tracemalloc` (que sí
  cuenta las reservas de numpy) y dan el RSS solo como contraste.
- **Verificación independiente incompleta.** De los siete verificadores
  escépticos del inventario terminaron dos (`recursive/` y cadenas) antes de
  detener el flujo por el límite de uso de la sesión; los otros cinco y el
  crítico de completitud no se ejecutaron. Los hallazgos de esas dimensiones
  están verificados por fragmento por su propio agente, pero no por un
  segundo agente.
- **Dos resultados A/B sin causa identificada** (3.6): el prototipo de la
  matriz pública sale más lento en pandas 2 (1,07 a 1,59) con memoria
  idéntica, y quitar `copy=False` en pandas 3 sale más lento en las
  funciones completas de multiseries (1,05 a 1,59) aunque el microbenchmark
  aislado da tiempos iguales. El plan los neutraliza (pandas 2 conserva su
  ruta; la PR 2 mide el cambio real sin monkeypatch), pero la causa no está
  establecida.
- **Interferencia en las medidas.** Las medidas definitivas (sección 4) se
  tomaron con la máquina en reposo tras detener todos los agentes; el arnés
  registra la carga media del minuto antes y después de cada benchmark y
  marca los que superan 1,5 (este proceso aporta 1,0).
- **`pd.date_range(..., unit=)`** se ha verificado en 2.2.3, 2.3.3 y 3.1.0
  (no en 2.2.0, el mínimo exacto del pin; el kwarg existe desde pandas 2.0
  según su documentación, no verificado aquí).
- **`concat(copy=)`**: la deprecación la emite el código de pandas
  (`core/reshape/concat.py:408-410` en 3.1.0 y también en 3.0.6 según la
  comprobación del 2026-10-04), pero `concat` no aparece en la lista "Copy
  keyword" del whatsnew 3.0.0 (l. 993-1017) ni en 3.1.0.
- **Notebooks de la documentación**: no se han ejecutado con pandas 3; el
  recuento de alias de frecuencia retirados y de salidas `dtype: object`
  impresas está en 3.4/3.5 (grep), no su ejecución.

## 10. Pendiente para retomar

Lo que quedó sin cerrar por el límite de uso de la sesión, con lo necesario
para continuarlo en otra sesión (el espacio temporal de esta no se conserva;
los scripts se reconstruyen a partir del anexo y de las descripciones de la
sección 7):

1. **Segunda verificación de cinco dimensiones del inventario** (3.1
   `direct/base/utils` y `model_selection`, 3.2 ya verificada, 3.3 fechas,
   3.4 APIs retiradas, 3.5 tests/CI). Método que funcionó en las dos
   dimensiones verificadas: un agente por dimensión que, para cada hallazgo
   de la sección 3 (todos llevan `fichero:línea` y la afirmación sobre
   pandas 2 y 3), abre el fichero, reejecuta un fragmento mínimo en los dos
   entornos del anexo y da un veredicto confirmado/corregido/refutado; en las
   dos dimensiones completadas no se refutó ninguno y se corrigieron matices
   de líneas. Prioridad: 3.3 y 3.4 (contienen los fallos reales y las
   deprecaciones).
2. **Crítico de completitud**: comprobar contra los puntos obligatorios del
   encargo (compatibilidad dual, CoW y copias, rendimiento, resultados
   idénticos) qué ficheros de `skforecast/` con uso de pandas no aparecen en
   ningún hallazgo (`rg -l "pd\." skforecast --type py -g '!**/tests*/**'`
   frente a los ficheros citados en la sección 3) y cuántos de los 114
   `.copy()` quedan sin clasificar (3.1 cubre 37 explícitamente; el resto
   está en los JSON no versionados: rehacer el recuento con
   `rg -n "\.copy\(\)"`).
3. **Los dos A/B sin causa** (3.6): medir con el cambio real en los ficheros
   (no por monkeypatch) `create_train_X_y` de `ForecasterRecursive` y
   `ForecasterDirect` en pandas 2 con el `astype` reducido, y las matrices de
   `ForecasterRecursiveMultiSeries` en pandas 3 sin `copy=False` en los dos
   `pd.concat`; criterio: dentro de la banda del control nulo (0,94 a 1,14).
   Si el segundo se confirma, sustituir los `pd.concat` por el ensamblado en
   numpy de la ruta de bloque único.
4. **Tres avisos pendientes de la suite sobre el prototipo** (2.3): renombrar
   `dayofweek`/`dayofyear` en `foundation/_adapters.py:4995-5001`, el helper
   de `pd.infer_freq` (3.3) y la ruta numpy de `_calendar.py:849-850`; son
   parte de las PR 1 y 2.
5. **Dataset `australia_tourism`** (`datasets.py:209`, `'freq': 'Q'`): el
   único fallo de datos; una línea en la PR 1.

## Anexo: cómo reproducir los números

Entorno de pandas 3 (fuera del repositorio; el de pandas 2 es el de trabajo):

```bash
uv pip freeze --python <python de pandas 2> | grep -v -iE "skforecast|^-e|^pandas==" > /tmp/p3.txt
echo "pandas>=3.1" >> /tmp/p3.txt
uv venv ~/.venvs/skforecast-p3 --python 3.12
uv pip install --python ~/.venvs/skforecast-p3/bin/python -r /tmp/p3.txt
# todo se ejecuta con PYTHONPATH=<raíz del repo> para no importar un skforecast instalado
```

Línea base (una vez; `warnrec` es un plugin de 20 líneas que vuelca cada
aviso capturado por pytest como JSON: `pytest_warning_recorded`):

```bash
PYTHONPATH=$PWD ~/.venvs/skforecast-p3/bin/python -m pytest skforecast -q \
  --continue-on-collection-errors --ignore=skforecast/deep_learning -rfE --junitxml=junit.xml
# los 90 node ids fallidos, con pandas 2 (pasan) y con pandas 3.0.6 (fallan):
xargs -d '\n' -a failing_nodeids.txt <python> -m pytest -q -p no:cacheprovider -o addopts=""
```

Identidad (21 casos, 106 salidas): construir los datos con
`pd.date_range("2020-01-01", periods=300, freq="D")` (y `.as_unit("ns")`
para la comparación como con como), 250 observaciones de entrenamiento, 10
pasos, `LinearRegression`/`Ridge`/`LGBMRegressor(n_estimators=20)`/
`DecisionTreeClassifier`/`Arima(1,0,1)`, `n_boot` 50 (30 en multiseries),
`random_state=123`; huella SHA-256 de cada salida con dos variantes:
estricta (incluye `str(dtype)` de pandas por columna, unidad y `freq` del
índice) y normalizada (fechas a enteros en ns, cadenas por `repr`,
categorías por códigos y categorías). Comparar los JSON de cada entorno.

Rendimiento: para cada benchmark, dos ejecuciones de calentamiento y cinco
medidas con `time.perf_counter`, variantes intercaladas (A, B, B, A, ...) en
el mismo proceso, mediana y mínimo; memoria pico con `tracemalloc`
(`reset_peak` antes, `get_traced_memory` después, mínimo de dos pasadas) en
una pasada aparte; un proceso cada vez y máquina en reposo; `os.getloadavg()`
antes y después como testigo. Tamaños: multiseries 100 series × 2000 obs,
lags 24, exógenas 3 float + 1 int + 1 category (bloque único) o 10 int (ruta
`concat`), 50 series en dict con `onehot` y window features; univariante
200k obs, lags 24; directo 50k obs, 12 pasos; multivariante 5 × 30k, 6 pasos;
clasificador 100k; `predict` 200 llamadas de 1 paso + 1 de 500; backtesting
3000 obs/1000 folds de 1 paso sin refit, 20 600 obs/60 folds con refit, 20
series × 1500 obs/100 folds; window features 300k obs. Los prototipos se
aplicaron por monkeypatch del código fuente del método (reejecución del
texto modificado en el espacio de nombres del módulo), con un control nulo
sin cambios para acotar el ruido.

Persistencia: `save_forecaster`/`load_forecaster` con `backend="joblib"` y
`"skops"` en cada versión, carga cruzada en la otra y comparación de las
huellas de `predict`; pickles puros de `DatetimeIndex`, `Series` `str`,
`DataFrame` con `category` entre versiones.
