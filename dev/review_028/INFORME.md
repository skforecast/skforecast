# Revisión de la 0.28.x frente a la 0.27.x

Revisión independiente de todo lo que cambia de `origin/0.27.x` a `origin/0.28.x` sin
`dev/` (commit `6d99b88`, 2026-11-20/21): 171 ficheros, de ellos 23 del paquete
(+1876 / −432 líneas) y 98 de tests. Cierra REVIEW-post-1516 del roadmap. Las filas que abre
están en `dev/ROADMAP.md` (abreviatura REV28) y sus scripts en [`repro/`](repro/).

## Método

- Cuatro revisores en paralelo, uno por área: (A) pandas 3, `utils` y dependencias; (B)
  `model_selection` y paralelización; (C) forecasters; (D) resto del paquete, documentación
  y contexto IA. Cada uno leyó el código completo de su área, comprobó cada entrada de la
  sección 0.28.0 de `docs/releases/releases.md` y confirmó cada hallazgo con un script.
- Comparaciones: 0.27.x frente a 0.28.x con pandas 2.3.3 (la 0.27 no admite pandas 3), y
  0.28.x con pandas 2.3.3 frente a pandas 3.1.1. Python 3.12.3, numpy 2.5.5, scikit-learn
  1.9.1, scipy 1.18.1, numba 0.68.1, statsmodels 0.15.0, optuna 5.1.0; para `ForecasterRnn`,
  keras 3.15.1 y torch 2.15.0 (CPU).
- Máquina de 4 CPU compartida por los cuatro: no se comprobaron las cifras de velocidad ni de
  memoria de las notas, solo que los resultados sean correctos e iguales.

## Resultado

No hay regresiones en los resultados. Lo comprobado:

- **0.27.x frente a 0.28.x:** 0 diferencias en valores, índices, dtypes, avisos y mensajes
  de error en unos 1000 escenarios de forecasters (C) y 972 configuraciones de backtesting y
  búsquedas (B): `predict` y todos los métodos de intervalos, refit (`False`, `True`,
  entero), `gap`, `skip_folds`, `fixed_train_size`, multiserie con dict de distinta longitud,
  exog en todos sus formatos, encodings, transformers, diferenciación, window features,
  búsquedas grid, random y bayesiana con `TimeSeriesFold` y `OneStepAheadFold`. La única
  diferencia es una mejora: `predict` de un multiserie sin ajustar con exog en formato largo
  da `NotFittedError` en lugar de `TypeError`.
- **pandas 2.3 frente a 3.1:** valores idénticos; solo cambian los dtypes documentados
  (`object` a `str`) y lo de la sección 9.
- **`n_jobs`:** resultados idénticos con 1, 2 y `'auto'`; el forecaster del usuario no se
  modifica (528 configuraciones dan los mismos bytes antes y después de la llamada); el
  fichero temporal se borra al terminar, si falla un fold y con `KeyboardInterrupt`, y las
  llamadas simultáneas no se pisan (nombre con uuid por llamada).
- **Entradas y estado:** ningún forecaster modifica `y`, `series`, `exog` ni `last_window`
  (pandas 2 y 3), y no hay cachés obsoletas al encadenar `fit`, `predict` y los `set_*`.
- **Forecasters guardados con 0.27:** cargan en 0.28 (pandas 2 y 3) y dan los mismos
  resultados en `predict`, intervalos, backtesting, refit, `set_params` y `set_lags`, salvo
  `ForecasterRnn` (documentado; sección 5).
- **Versiones mínimas:** un entorno con todas ellas (Python 3.11, numpy 2.0.0, pandas 2.3.0,
  scikit-learn 1.6.0, scipy 1.13.0, numba 0.60.0, optuna 4.0.0, joblib 1.3.0, statsmodels
  0.14.6, matplotlib 3.8.4, skops 0.14.0, lightgbm 4.4.0) pasa los tests de `utils`.
- **Contexto IA:** `generate_ai_context_files.py --check` dice que todo está al día.
- **Tests de las áreas:** pasan con pandas 2 y 3; los únicos fallos son de entorno (torch
  ausente en `test_TSICLAdapter.py`, backend TensorFlow en `test_fit_tensorflow`). No se ha
  relajado ninguna aserción ni se ha borrado ningún test.

| Sección | Fila del roadmap | Severidad |
|---|---|---|
| 1 | BUG-load-pytz | Importante |
| 2 | BUG-skops-tz-dst | Importante (previo) |
| 3 | BUG-splitter-long-phase | Menor (regresión) |
| 4 | PERF-njobs-auto-first-fold | Importante (rendimiento) |
| 5 | BUG-rnn-old-pickle | Menor |
| 6 | BUG-njobs-temp-folder | Menor |
| 7 | DOC-028-review | Importante (punto 1) a menor |
| 8 | Filas existentes | |
| 9 | Sin fila | Detalles |

## 1. BUG-load-pytz

pandas 2 guarda una zona horaria con nombre (`'Europe/Madrid'`) como objeto de `pytz`; pandas
3 ya no depende de `pytz` y ninguna dependencia de skforecast lo instala. Un forecaster
guardado con pandas 2 con joblib, pickle o cloudpickle no carga en un entorno limpio de pandas
3: `ModuleNotFoundError: No module named 'pytz'`, sin nota, porque `load_forecaster` solo
anota `AssertionError`, `NotImplementedError` y `TypeError`. Sin zona horaria o con `'UTC'`
carga con todos los backends; con skops también (guarda la zona por nombre); con `pytz`
instalado el fichero carga y predice igual.

Contradice `docs/faq/pandas-2-and-pandas-3.md` (línea 84, "loads with pandas 3, with any
backend"), `docs/user_guides/save-load-forecaster.ipynb` (hacia la línea 700) y
`skills/troubleshooting-common-errors/SKILL.md` (línea 323). P3 verificó skops, no joblib
(`pandas3_migration/INFORME.md`, línea 705).

**Arreglo:** en `load_forecaster`, nota para `ModuleNotFoundError` con `exc.name == 'pytz'`
("el fichero se guardó con pandas 2: instala `pytz` o vuelve a guardarlo con skops"); y
corregir los tres documentos.

Script: `repro/load_pytz.py` (`save` con pandas 2, `load` con pandas 3 sin `pytz`).

## 2. BUG-skops-tz-dst

Con `backend='skops'`, un forecaster con índice diario (`freq='D'`) creado en UTC y convertido
a una zona local se guarda, pero no carga cuando su `last_window_` cruza un cambio de hora:
`ValueError: Inferred frequency None from passed values does not conform to passed frequency
D`, en `_compose_index` (`skforecast/utils/utils.py`), que reconstruye el índice con
`pd.DatetimeIndex(..., freq=...)` y pandas valida la frecuencia.

- Previo: el mismo error en 0.27.x, y guardando y cargando con pandas 2.3. No es de pandas 3.
- Con pandas 3 ese índice no puede tener `freq='D'` (`tz_convert` la quita); con `'24h'`, lo
  que recomiendan las notas, carga en las dos direcciones.
- Con un `last_window_` de unos 183 días o más el cruce es seguro (`lags=400` da "Inferred
  frequency 24h ... does not conform to passed frequency D"). El revisor A lo vio también con
  datos mensuales (`'MS'`, UTC a `America/New_York`) al cargar con pandas 3; sin comprobar con
  pandas 2.
- Importa porque se puede guardar un modelo que después no se puede cargar, y la FAQ (líneas
  84 y 88) dice que skops funciona en las dos direcciones.

**Arreglo:** reconstruir el índice sin validar la frecuencia y asignarla después, como hace
el unpickling; o convertir `Day(n)`/`Week` en `Hour(24n)`/`Hour(168)` para los índices
anclados en UTC (`_is_utc_anchored_index`). Como mínimo, documentarlo.

Script: `repro/skops_tz_dst.py`.

## 3. BUG-splitter-long-phase

`TimeSeriesSplitter` con `output_format='long'`: cuando la unión de los índices no tiene
frecuencia, `_wide_with_freq` (`skforecast/experimental/_splitter.py`) llama a
`asfreq(series_freq)` desde la fecha mínima de la unión, y esa rejilla descarta todas las
fechas de una serie con la misma frecuencia y otra fase. Las salidas `'dict'` y `'wide'` están
bien.

Valores no NaN de la serie `b` en train (se esperan 7):

| Caso (frecuencia + desfase de `b`) | 0.27.x | 0.28.x |
|---|---|---|
| `7D` + 3 días | 7 | 0 (regresión) |
| `D` + 6 horas | 0 | 0 |
| `3h` + 1 hora | 0 (y `a` 1 de 7) | 0 |
| `MS` + 14 días | 7 | 7 |

**Arreglo:** construir la salida larga desde el diccionario de cada serie, o extender la
rejilla a la unión de los `date_range` de cada serie, o lanzar un error si `asfreq` descarta
valores no NaN.

Script: `repro/splitter_long_phase.py`.

**Nota posterior (2026-11-23):** arreglado en #1534 (rama `fix/splitter-long-phase`), con la
primera opción detrás del camino que había. `_long_from_dict` sustituye a `_wide_with_freq`:
genera las fechas de la rejilla común y, si no contiene todas las del split (o no se puede
construir por un cambio de hora), apila cada serie con sus propias fechas, como en `'dict'`;
si las contiene, la salida es la de antes. En la rama, `b` tiene 7 valores en los cuatro
casos, con pandas 2.3.3 y 3.1.1. El arreglo cubre también las uniones a las que pandas da una
frecuencia que las fechas no siguen (`BusinessHour`, `MS` con horas distintas), que este
informe no recogía. Al hacerlo aparecieron cuatro fallos previos, iguales en 0.27.0, con fila
propia en el roadmap y script en `repro/`: BUG-wide-to-long-inferred-freq,
BUG-utc-midnight-anchored-freq, BUG-splitter-positional-cuts y
BUG-splitter-pandas2-dst-union (este, solo con pandas 2).

## 4. PERF-njobs-auto-first-fold

Con `n_jobs='auto'`, el backtesting sin refit de `HistGradientBoostingRegressor` o
`CatBoostRegressor` predice el fold 0 en el proceso principal y paraleliza si `n_folds × tiempo
del fold 0` supera 6 s (12 s en Windows) (`_validation.py`, hacia las líneas 850 a 879;
`_utils.py`, 1002 a 1047). Pero el primer `predict` de una sesión es mucho más lento que los
demás, y el coste extra es de skforecast (el `predict` del estimador solo tarda 0,4 ms la
primera vez).

Medido el 2026-11-21 (HGB con `max_iter=50`, `lags=24`, 1000 folds de 1 paso, `OMP_NUM_THREADS=1`):

| Llamada | Fold 0 | Mediana del resto | Estimación | `n_jobs` |
|---|---|---|---|---|
| Primera de la sesión | 12,5 ms | 0,91 ms | 12,5 s | 3 |
| Segunda | 1,5 ms | 0,68 ms | 1,45 s | 1 |

Con unos 600 folds o más (1200 en Windows), la primera llamada siempre paraleliza, cueste lo
que cueste predecir. En los workers cada fold además deserializa el forecaster (0,75 ms para
este, lo mismo que su `predict`), así que esa llamada puede ser más lenta que en 0.27 (en
serie), el caso que la 0.27.0 había corregido ("up to 20 times slower", `releases.md`). El
efecto en tiempo real no se midió (máquina compartida). NJOBS ya vio que la decisión cambia
entre llamadas cerca del umbral (propuesta 3); esto muestra que pasa lejos de él.

**Arreglo:** estimar con un fold caliente: predecir los folds 0 y 1 en el proceso principal y
usar el tiempo del segundo (o el mínimo), o hacer una llamada de calentamiento antes de medir.

Script: `repro/njobs_auto_first_fold.py` (sustituye `Parallel` por uno secuencial: no arranca
workers, solo registra el `n_jobs` pedido).

## 5. BUG-rnn-old-pickle

Un `ForecasterRnn` guardado con 0.27 predice en 0.28 (mismos valores) y hace backtesting con
`initial_train_size=None`, pero:

- `fit`: `AttributeError: 'ForecasterRnn' object has no attribute 'warm_start'`.
- `set_params`: `... '_initial_compile_config'`.
- Backtesting con `initial_train_size` y `grid_search_forecaster_multiseries`:
  `... '_initial_weights'` (`deepcopy_forecaster` y `_reset_estimator`).

Las notas lo documentan ("must be created again"); falta que el error lo diga. **Arreglo:**
comprobar `hasattr(self, '_initial_weights')` en `fit`, `set_params` y `_reset_estimator`, y
lanzar un `ValueError` que diga que el forecaster se guardó con skforecast < 0.28 y hay que
crearlo de nuevo.

Script: `repro/rnn_old_pickle.py` (`save` con 0.27.x, `load` con 0.28.x; necesita keras y
torch).

## 6. BUG-njobs-temp-folder

Desde #1503, un forecaster de más de 1 MB va siempre a los workers por la carpeta temporal de
joblib (`_validation.py`, hacia las líneas 241 a 243). Si esa carpeta no se puede escribir, el
backtesting falla con un `PicklingError: Could not pickle the task to send it to the workers`
que no dice la causa. La 0.27 funcionaba con el mismo forecaster (`RandomForestRegressor`, sin
arrays de más de 1 MB). **Arreglo:** capturar el error y repetir con `max_nbytes=None`, o al
menos un mensaje claro.

Script: `repro/njobs_temp_folder.py` (`JOBLIB_TEMP_FOLDER` apuntando a una ruta no
escribible).

**Nota posterior (2026-11-23):** arreglado en #1535 (rama `fix/njobs-temp-folder`), con la
segunda opción: el error explica la causa y tres remedios (`JOBLIB_TEMP_FOLDER`,
`joblib.parallel_config(max_nbytes=None)` o `n_jobs=1`), sin reintento. El script no
reproducía en Windows (`/dev/null/not-writable` es allí una ruta válida y joblib la creaba);
ahora usa una ruta que cuelga de un fichero. La 0.27 ya fallaba igual cuando algún array de
los argumentos pasaba de 1 MB (los datos, o un KNN ajustado).

## 7. DOC-028-review: correcciones de documentación

Script de los puntos 2, 4 y 5: `repro/doc_checks.py`. Al terminar, regenerar el contexto IA
(skill `ai-context-sync`).

**Release notes, "Before upgrading" (`docs/releases/releases.md`, sección 0.28.0):**

1. **Línea 29, "`pip install -U skforecast` keeps the installed pandas":** falso para quien
   tiene pandas < 2.3, que la 0.27 admitía (mínimo 2.2). pip sube un requisito no cumplido a
   la última versión permitida: con pandas 2.2.3 y numpy 1.26.4 instala pandas 3.1.1 y numpy
   2.5.5, no pandas 2.3 (`pip install --dry-run -U` con un paquete de prueba con los mismos
   requisitos). Decir que pandas < 2.3 pasa a pandas 3 y dar la alternativa `pip install -U
   skforecast "pandas<3"`. P3 lo comprobó solo con pandas 2.3.3 instalado
   (`pandas3_migration/INFORME.md`, línea 2634).
2. **Faltan cambios que rompen código existente** (step 3 de la skill `release-note`):
   `n_jobs=0` lanza `ValueError` en `ForecasterDirect`, `ForecasterDirectMultiVariate` y el
   backtesting (antes usaba todos los CPU); los `n_jobs` negativos siguen la
   regla de joblib (`-2` = `cpu_count() - 1`); con el backend de hilos, un forecaster que no
   se puede serializar falla. Están en Changed (líneas 81 y 88), no en "Before upgrading".
3. **`calculate_distance_from_holiday` cambia valores** sin línea en "Before upgrading": +1
   día al cruzar el cambio de hora de primavera con fechas diarias locales con zona horaria
   (5 pasa a 6), y días en lugar de horas en índices sin frecuencia anclados en UTC.
4. **Línea 31, "with `'24h'`, the results are the same as with `'D'`":** no en
   `calculate_distance_from_holiday`, que con `'24h'` da horas (la FAQ sí lo dice).
5. **Línea 33, `ForecasterEquivalentDate` con `Day(n)`:** además de cambiar las predicciones,
   con pandas 3 puede lanzar `ValueError: Cannot infer dst time ...` (o "nonexistent time")
   cerca de un cambio de hora (backtesting de datos horarios de `Europe/Madrid` desde
   2021-10-24 12:00 con `Day(1)`, `n_offsets=2`; el error sale de `date - self.offset`). La
   FAQ lo documenta. Opcional: relanzarlo con la pista `offset=24 * n`.
6. **Línea 41 y la entrada de Fixed de `reshape_series_wide_to_long` / `TimeSeriesSplitter`:**
   - "daily data with missing dates raises" vale para cualquier frecuencia: días laborables
     con un festivo ahora lanzan `ValueError` (la 0.27 devolvía NaN los fines de semana).
     Decidir si es lo deseado y documentarlo.
   - El consejo `data.asfreq('D')` (también en el mensaje de error) vuelve a meter las filas
     NaN de los fines de semana; con datos de días laborables, `'B'`.
   - En `TimeSeriesSplitter` la frecuencia sale de las series (`check_preprocess_series` la
     exige), no se infiere de las fechas, y nunca lanza el `ValueError` nuevo. Con entradas
     anchas (días laborables, horarias, mensuales) la salida es idéntica a la 0.27; solo
     cambian los dict de series con huecos entre tramos (mensual: de 2434 filas a 82).
7. **Línea 75 (#1516), `check_predict_input` "now returns the index ... instead of `None`":**
   sigue devolviendo `None` cuando `exog` es `None` (su docstring lo dice).
8. Opcional: #1512 cambia dos casos límite, a mejor: IDs de serie enteros con algún ID
   ausente dan claves `int` (antes `float`), y una primera columna con nombre duplicado
   funciona (antes `TypeError`).
9. Convenciones de la skill `release-note` (Changed y Fixed sin agrupar por área, 4 entradas
   de más de 80 palabras): se resuelven con `release-note consolidate` al publicar.

**FAQ, guías y skills:**

10. `docs/faq/pandas-2-and-pandas-3.md`, líneas 84 y 88, `save-load-forecaster.ipynb` y
    `skills/troubleshooting-common-errors/SKILL.md`, línea 323: los ficheros de pandas 2 no
    cargan "with any backend" en pandas 3 (secciones 1 y 2). Línea 86: un forecaster guardado
    con pandas 3 con joblib o pickle sí carga en pandas 2 si su índice es un `RangeIndex`
    (una serie y multiserie); solo fallan los de `DatetimeIndex`.
11. Backends de foundation que bajan pandas: la FAQ (línea 7) y las notas (línea 46) dicen
    "installing them downgrades pandas to 2.3". Con Moirai solo es así con el `"pandas>=2.3"`
    documentado: `uni2ts` solo resuelve gluonts 0.14.4 (`pandas<2.2`), es decir pandas 2.1.4,
    por debajo del mínimo de skforecast (`uv pip compile`). La tabla de instalación de la guía
    de foundation, `skills/foundation-forecasting/SKILL.md` (línea 60) y `llms-base.txt` no
    avisan de que TabPFN y Moirai pasan de pandas 3 a 2.3.
12. `skills/complete-api-reference/references/forecaster-constructors.md`, líneas 143 a 153:
    falta `warm_start=False` en el constructor de `ForecasterRnn` (entre `binner_kwargs` y
    `forecaster_id`). `skills/deep-learning-forecasting` lo tiene.
13. `forecasting-with-deep-learning-rnn-lstm.ipynb`, celda "Using Custom Loss Functions": dice
    que la pérdida propia hay que registrarla "to save and load a model"; hace falta ya para
    crear el `ForecasterRnn`, que copia el modelo (Keras da `TypeError: Could not locate
    function`, igual en 0.27; `repro/rnn_custom_loss.py`).
14. Notas "tirex-2 requires Python >= 3.11", que ya siempre se cumple (la de T0 se quitó, estas
    no): `foundation/_adapters.py` (hacia las líneas 6083, 6518 y 6528),
    `skills/foundation-forecasting/SKILL.md` (65 y 287), `adapter-parameters.md` (227), la guía
    de foundation y `llms-base.txt` (460).
15. Opcional: el "Quick Info" de `llms-base.txt` da las versiones de Python, pero no los mínimos
    de numpy y pandas ni el soporte de pandas 3.

**Docstrings:**

16. `check_backtesting_input` (`model_selection/_utils.py`, hacia la línea 315): `n_jobs='auto'`
    cita `select_n_jobs_fit_forecaster`; es `select_n_jobs_backtesting`.
17. `_validation.py`, hacia las líneas 197 a 203: dice que los procesos comparten el
    forecaster; comparten los bytes, pero cada fold lo deserializa y copia sus arrays (ver
    PERF-n_jobs-next, punto 1).
18. Ejemplo de `TimeSeriesSplitter`: `from skforecast.utils.splitter import
    TimeSeriesSplitter` no existe (`experimental/_splitter.py`, línea 63), y la clase no se
    exporta en `skforecast.experimental`. Previo.
19. `ForecasterEquivalentDate`: `offset : int, pandas.tseries.offsets.DateOffset` (líneas 48 y
    86); el código acepta cualquier `pd.offsets.BaseOffset`, y en pandas 3 `Day` no es un
    `DateOffset`.
20. `ForecasterRnn._reset_estimator`: la NOTE dice que `safe_mode=False` permite lambdas como
    pérdida; una lambda que usa globales ya falla en el primer `fit` (0.27 y 0.28), porque
    Keras la reconstruye desde el bytecode al copiar.

## 8. Añadido a filas que ya existían

- **ISSUE-1507:** confirmado otra vez (`_freq_to_timedelta_unit` da horas con `SME`, `SMS`,
  `BQE`, `BQS`, `BYE` y `BYS`; igual en 0.27). Arreglo propuesto por D: como `freq` ya es
  siempre un offset, devolver `'D'` salvo que sea un `Tick` distinto de `Day`.
- **PERF-n_jobs-next, puntos 1 y 5:** confirmado. Ridge con un atributo de 16 MB, 200 folds sin
  refit y `n_jobs=2`: 200 deserializaciones y una copia privada del array en 0.28, frente a 56 y
  un memmap compartido en 0.27.
- **PERF-exog-float-matrix:** reproducido otra vez (exog `bool` y texto, `categorical_features=
  'auto'` y HGB con `categorical_features='from_dtype'`: `TypeError: ufunc 'isnan'`; igual en
  0.27).
- **BUG-ms-exog-none-bool:** mismo patrón en `predict` con pandas 2, en `_create_predict_inputs`
  (`pd.concat` de la exog de todos los niveles): cuando a una serie le faltan columnas o no tiene
  exog, `FutureWarning` del `concat` y un `DataTypeWarning` espurio (una `int` pasa a `object`).
  Valores idénticos; con pandas 3 no hay avisos.

## 9. Sin fila

Detalles que no justifican una fila; resolverlos de paso si se toca el código:

- `ForecasterDirect` y `ForecasterDirectMultiVariate` con `RangeIndex`: la predicción tiene
  índice `Index[int64]` en pandas 2 (también en 0.27) y `RangeIndex` en pandas 3, por
  `index_all_steps[np.array(steps) - 1]`. `ForecasterRecursive` da `RangeIndex` en los dos.
  `assert_frame_equal` pasa con `check_index_type='equiv'`. Podría ir en la FAQ de pandas 2 y 3.
- Backtesting de `ForecasterRecursiveClassifier` cuando una clase aparece en un fold posterior:
  su columna `_proba` va la última, después de los predictores si `return_predictors=True`
  (igual en 0.27; el docstring nuevo describe el caso sin decir el orden). Arreglo: ordenar las
  columnas de probabilidad por clase antes de los predictores.
- El guard de reutilización de `prediction_index` en multiserie y DMV es correcto pero no tiene
  test propio.
- `reshape_series_wide_to_long`: faltan tests de días laborables y de menos de 3 fechas.

## 10. Qué no se comprobó

- Las cifras de velocidad y memoria de las notas (máquina compartida).
- Las versiones mínimas de catboost 1.2.8, xgboost 2.1, keras 3.3 y torch 2.4.
- La ejecución de los modelos foundation (sin torch ni backends) y de los notebooks.
- Las entradas de `_split.py` de Changed y Fixed (ordenar frecuencias con `key=str`,
  `BaseOffset`): solo lectura.

## Cómo ejecutar los scripts

Desde `dev/review_028/repro/`. Los ficheros que pasan de un entorno a otro van a
`$REVIEW028_OUT` (por defecto, una carpeta en el directorio temporal del sistema, nunca dentro
del repositorio). Para ejecutar la 0.27.x: `git worktree add --detach <ruta> origin/0.27.x` y
`PYTHONPATH=<ruta> python script.py`, en un entorno con pandas 2.3 (la 0.27 no admite pandas 3).
`rnn_old_pickle.py` y `rnn_custom_loss.py` necesitan keras y torch (`KERAS_BACKEND=torch`).
