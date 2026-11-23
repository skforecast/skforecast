# Roadmap de skforecast

Roadmap único y vivo: todo lo pendiente, con la prioridad de cada elemento y el orden de
trabajo (sección 11). Sustituye a los roadmaps por versión (`dev/roadmap_0.27.0/` y `dev/roadmap_0.28.0/`, borrados). Creado
el 2026-11-19 a
partir del inventario de `dev/` sobre `origin/0.28.x` (`3c0711b8a`): el estado de cada
elemento se comprobó entonces con `gh`, `git log` y el código, y los bugs marcados como
"reproducido" se ejecutaron contra ese commit.

## Cómo se usa

- **Una fila por elemento**, con un ID estable. El detalle largo vive en la carpeta o el
  documento que enlaza la columna "Detalle"; aquí va lo necesario para priorizar y empezar.
- **Sin versión objetivo**: cada versión sale con lo que esté hecho al publicarla. Lo que sí
  depende de una publicación (código final, publicar, deprecaciones anunciadas) está en la
  sección 1, con su momento en lugar de nivel.
- **Prioridad**, por el efecto en los usuarios y en sus resultados:
  - `P1`: resultados erróneos sin aviso con una configuración habitual o documentada, y
    regresiones frente a la última versión publicada (estas bloquean la publicación).
  - `P2`: errores visibles en usos plausibles, resultados erróneos en configuraciones poco
    habituales, documentación incorrecta que los usuarios siguen, y ganancias de
    rendimiento grandes, medidas y baratas.
  - `P3`: funcionalidades nuevas, mejoras de rendimiento medianas o sin medir, y
    mantenimiento.
  - `P4`: solo si aparece demanda (nicho, poco valor medido o probable descarte).

  Un requisito hereda el nivel más alto de lo que desbloquea y lo dice: `P1 (por X)`. Lo
  bloqueado por algo externo conserva su nivel y lo dice en "Estado" (`Bloqueado: ...`).
  Subir o bajar algo es cambiar esta celda, no mover texto; dentro de cada tabla, las filas
  van por nivel.
- **Requiere**: los elementos o decisiones (sección 10) sin los que no se puede hacer bien.
  Solo requisitos de verdad: los cambios que conviene hacer seguidos, que comparten tests o
  que no deben ir en paralelo están en las cadenas de la sección 11.
- **Al terminar un elemento**: su fila pasa a "Hecho desde la última versión publicada" con
  el número de la PR. Al publicar una versión, esa sección se vacía (lo publicado queda en
  `docs/releases/releases.md`).
- **Una carpeta de `dev/` existe solo mientras tenga alguna fila abierta aquí.** La PR que
  cierra la última fila borra también la carpeta, y si hace falta citarla después se cita por
  commit.
- Las líneas de código citadas en los documentos de detalle son de su fecha: buscar el
  código por nombre de clase o función.
- Esfuerzo: XS (menos de una hora), S (horas a 1 día, con tests y docs), M (2 a 5 días), L
  (más de una semana).

### Abreviaturas de la columna "Detalle"

Los documentos borrados se leen con `git show <commit>:<ruta>`.

| Abreviatura | Documento |
|---|---|
| R27 §N | `402be62d6:dev/roadmap_0.27.0/roadmap.md`, sección N (borrado) |
| R27-repro/X, R27-verify/X | `402be62d6:dev/roadmap_0.27.0/repro/X`, `402be62d6:dev/roadmap_0.27.0/work/verify/X` (borrados) |
| TRANSP §N | `d24e5b852:dev/roadmap_0.27.0/BUG-search-short-series-transparency.md`, sección N (borrado) |
| SHORT | `d24e5b852:dev/roadmap_0.27.0/BUG-search-short-series.md` (borrado) |
| UTILS §N | `35a8d976d:dev/handoff_utils_pendiente.md`, sección N (borrado); revisión completa en `2377adb24:dev/informe_utils_review.md` |
| R28 | `bdc9ae521:dev/roadmap_0.28.0/roadmap.md` (borrado) |
| P3 §N | `dev/pandas3_migration/INFORME.md`, sección N (congelado) |
| FAR | `dev/foundation_adapters_review/` |
| NJOBS | `dev/n_jobs_benchmark/REPORT.md` |
| PREP | `dev/preprocessing_speedups/PLAN.md` |
| MSPERF | `30195692f:dev/multiseries_predict_perf/` (borrado; `SEGUIMIENTO.md`, `PLAN.md`, `MEDIDAS.md` con el informe final en §17, y `scripts/`) |
| BTSPEED §N | `dev/backtesting_multiseries_speedups/PLAN.md`, sección N |
| NPREV §N | `dev/pandas_numpy_review/REVISION.md`, sección N (revisión independiente, con dos rondas, de `INFORME.md`, en la misma carpeta; donde difieran vale la revisión) |
| REV28 §N | `dev/review_028/INFORME.md`, sección N (revisión de la 0.28.x frente a la 0.27.x, 2026-11-21); los scripts, en `dev/review_028/repro/` |

### Carpetas de `dev/`

| Carpeta o fichero | Qué es | Se borra cuando |
|---|---|---|
| `ROADMAP.md` | Este documento | Nunca |
| `00_template.ipynb` | Plantilla de notebook de trabajo | Nunca |
| `foundation_adapters_review/` | Planes 02 a 10 de alineación de los adaptadores foundation y su evidencia | Se cierren FOUND-plan-02 a FOUND-plan-10 |
| `timeh2o_adapter/` | Especificación del adaptador TimeH2O y sus experimentos (`experiments/skeleton/` hace falta para implementarlo) | Se implemente o se descarte FOUND-timeh2o |
| `n_jobs_benchmark/` | Informe y scripts de medida de `n_jobs` (#1503) | Se cierre PERF-n_jobs-next |
| `preprocessing_speedups/` | Plan y kit de validación de los atajos de preprocesado (en curso) | Se cierren PERF-categorical-codes, STUDY-fit-insample-binning y PERF-one-step-ahead-inverse |
| `backtesting_multiseries_speedups/` | Plan de PERF-backtesting-folds-positions y de la segunda parte de PERF-03, con las medidas y los prototipos de la revisión del 2026-11-19 | Se cierren PERF-backtesting-folds-positions y PERF-03 |
| `pandas_numpy_review/` | Estudio del uso de pandas frente a numpy dentro de los forecasters (`INFORME.md`), su revisión independiente (`REVISION.md`), y los prototipos, scripts y medidas de los dos (2026-11-19); la sección 11 de la revisión prueba la opción (a') de D-NP2 con un parche (2026-11-20) | Se cierren BUG-ms-fit-dataframe-categorical, PERF-exog-float-matrix y MAINT-ms-predict-core, y se decida PERF-backtesting-ms-fast-path (el prototipo y su batería son su especificación) |
| `review_028/` | Informe de la revisión de la 0.28.x frente a la 0.27.x (2026-11-21) y los scripts mínimos de sus hallazgos | Se cierren BUG-skops-tz-dst, BUG-rnn-old-pickle, DOC-028-review, BUG-wide-to-long-inferred-freq, BUG-utc-midnight-anchored-freq, BUG-splitter-positional-cuts y BUG-splitter-pandas2-dst-union |
| `pandas3_migration/` | Estudio, decisiones y arnés de la migración a pandas 3, y su informe final (`INFORME_0.28_pandas3.md`, DOC-pandas3-report) | Se cierre MAINT-pandas3-final-crossenv (el A/B de las búsquedas puede usar el arnés); antes, mover la lista "Al retirar pandas 2" a MAINT-drop-pandas2. El arnés lo usan también PERF-predict-numpy y otros: decidir entonces si pasa a `benchmarks/` o `tools/`. Después, el informe se cita por commit |

---

## 1. Ligado a una publicación

Lo que no se ordena por prioridad porque depende de una publicación concreta: se hace con
el código final, al publicar, o lo promete un aviso de deprecación. Además, antes de
publicar una versión tienen que estar cerradas sus regresiones frente a la última versión
publicada (P1.1 de la sección 11) y hay que pasar `/release-note consolidate`.

| ID | Qué | Cuándo | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| MAINT-pandas3-final-crossenv | Comparación final p3 frente a p2 sobre la 0.28.x definitiva: el arnés y las búsquedas | Con el código final de la 0.28.0 (aplazado el 2026-11-20: hacerlo o no, por decidir) | S | Arnés hecho el 2026-11-20 sobre `bdc9ae521` (31 benchmarks, p2 y p3 con y sin `pyarrow`; tabla en el apéndice A de `pandas3_migration/INFORME_0.28_pandas3.md`, los resultados en bruto no se versionan): `predict` en bucle 1,20 a 1,43 con `pyarrow`, el resto entre 0,79 y 1,20 salvo la matriz de solo lags (D13). Pendientes las búsquedas, que el arnés no mide: un informe externo (2026-11-20, pandas 3.1.1 con `pyarrow`, una sola ronda) vio del 10 al 20 % más de tiempo con pandas 3 en las de cinco artículos (py42, py44, py48, py56 y py66, con LightGBM, XGBoost y HistGradientBoosting), sin confirmar: con la variabilidad de LightGBM hace falta un A/B | P3 §10; `dev/pandas3_migration/harness/` |
| MAINT-bench-pandas3 | Pasar `benchmarks/` a pandas 3 con una medición puente: tanda con `pandas==2.3.3`, PR que cambia solo ese pin y guarda la versión de `pyarrow`, segunda tanda (sin `pyarrow`, como el entorno de la CI), una tercera con `pyarrow` instalado (donde pandas 3 perdía más frente a pandas 2 y donde más gana PERF-multiseries-names-pyarrow: es la cifra que verán muchos usuarios), filtrar por `pandas_version` en `docs/faq/skforecast-benchmark.ipynb` y comprobar que su entorno tiene pandas 3 (un `benchmark.joblib` escrito con pandas 3 no se abre con pandas 2) | Con el código final de la 0.28.0 | S | Pendiente: al final de la 0.28.0, con todo el código fusionado. Segundo estudio el 2026-11-20 (rama temporal `perf/bench-study-0.28-b`, borrada): código de `36cf2ce0c` en tres campañas de 5 ejecuciones (pandas 2.3.3; 3.1.1 sin `pyarrow`; 3.1.1 con `pyarrow` 26.0.1), frente a las 15 ejecuciones de la 0.27.0 del histórico, misma CPU y mismas versiones (Python 3.12.16, numpy 2.4.6, scikit-learn 1.9.0). Media geométrica de los 75 benchmarks frente a la 0.27.0: 0,76 con pandas 2, 0,82 con pandas 3 y 0,85 con `pyarrow` (backtesting 0,71, `predict` 0,78 y `fit` 0,97 con `pyarrow`); pandas 3 frente a pandas 2 con el mismo código, 1,03 sin `pyarrow` y 1,11 con él. Caminos reales que empeoran frente a la 0.27.0 con `pyarrow`: `check_predict_inputs` 1,21 a 1,24, `_create_train_X_y` de una serie 1,13 a 1,19, `ForecasterStats` (`_create_predict_inputs`, `predict`, `predict_interval`) 1,12 a 1,17, `fit` de `ForecasterRecursive` y del clasificador 1,12 a 1,14. `ForecasterRecursiveMultiSeries__create_lags` salía 1,43 sin `pyarrow` y 2,06 con él: el benchmark lo llamaba con `X_as_pandas=True`, que ningún forecaster usa, y medía la copia que hace el constructor de `DataFrame` de pandas 3 de una vista de numpy (como D13). La PR #1531 lo pasa a `X_as_pandas=False` (unos 10 µs con pandas 2 y con pandas 3); tiene que fusionarse antes de la tanda puente, porque cambia el `function_hash` de ese benchmark. Los `_create_lags` de una serie son solo numpy y sus razones son ruido | R28 "MAINT-bench-pandas3" (pasos y estudio previo del 2026-11-16) |
| MAINT-docs-pandas3 | Reejecutar con pandas 3 los 5 notebooks cuyas salidas cambian: `[ns]` a `[us]` en `faq/cyclical-features-time-series`, `user_guides/forecasting-statistical-models` y `user_guides/forecasting-with-deep-learning-rnn-lstm`; `object` a `str` en `user_guides/categorical-features` y `user_guides/autoregressive-classification-forecasting` | Antes de publicar la 0.28.0 | S | Pendiente: cuando el entorno de la documentación use pandas 3 | R28 tabla de la sección 1 |
| REL-conda-forge-028 | Feedstock de conda-forge al publicar: Python >=3.11, numpy >=2.0, pandas >=2.3,<4.0, scipy >=1.13, numba >=0.60, statsmodels >=0.14.6, matplotlib >=3.8.4 | Al publicar la 0.28.0 | XS | Pendiente: al publicar | R28 "DEP-python310" y "DEP-numpy2-pandas23" |
| REL-skforecast-ai-python311 | Abrir un issue en `skforecast/skforecast-ai` para subir su mínimo a Python 3.11 y, después, actualizar `docs/skforecast-ai/skforecast-ai.md` (líneas 3 y 30 dicen 3.10) | Al publicar la 0.28.0 | XS | Pendiente | R28 "DEP-python310" |
| DEP-029-removals | Quitar las deprecaciones de la 0.27.0: DEP-cv-differentiation (#1423) y DEP-datasets-version (#1440) | En la 0.29.0 (lo anuncian los avisos de deprecación) | S | | #1423, #1440 |

## 2. Dependencias y mantenimiento

| ID | Qué | Prioridad | Requiere | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|---|
| MAINT-docs-deps | Subir `mkdocs-material`, `mkdocstrings`, `mkdocstrings-python` y `pymdown-extensions` (`black` ya se subió en `a1a837b4a`); no subir `mkdocs-jupyter` 0.27.0 (carga MathJax desde un CDN, choca con KaTeX y con el plugin `privacy`). Añadir `pymdown-extensions` al `ignore` de `.github/dependabot.yml` (#1517 está abierta contra `main`) | P3 | | S | Opcional: medido, sin efecto visible | R28 "MAINT-docs-deps" |
| MAINT-zensical | `mkdocs-material` llega a fin de vida el 2027-05-05; sucesor: Zensical (https://zensical.org/docs/compatibility/mkdocs/migration/) | P3 | | M a L | Sin analizar | R28 "MAINT-docs-deps" |
| MAINT-test-deps | Decidir si `pandas[parquet]>=2.3` del extra `test` lleva también `<4.0` (el core lo tiene). El resto de la limpieza está hecho | P3 | | XS | Pendiente de decidir | R28 "MAINT-test-deps" |
| DEP-statsmodels-016 | statsmodels 0.15 avisa (`FutureWarning`) de que `kpss` y `adfuller` devolverán un objeto de resultado desde la 0.16. skforecast los llama en `stats/seasonal/_differencing.py` y `stats/arima/_auto_arima.py`, silencia el aviso y lee el resultado por posición. Al subir el límite `<0.16`: comprobar que el objeto nuevo admite `[0]` y `[1]` y, cuando el mínimo lo permita, pasar `result_object` de forma explícita | P3 | | XS | Bloqueado: hasta subir el límite `<0.16`. Sin efecto hoy: el aviso está silenciado y el límite `<0.16` lo cubre. Visto el 2026-11-20 (informe externo, statsmodels 0.15.0), sin comprobar aquí; según el informe, `result_object` no existe en el mínimo (0.14.6) | Esta fila |
| MAINT-tensorflow-223 | Cuando salga tensorflow 2.22 final: `<2.23`, revisar el marcador `python_version < "3.14"` de `pyproject.toml` y los `skipif` de los tres tests de TensorFlow | P3 | | XS | Bloqueado: hoy solo hay 2.22.0rc0 | R28 §2 |
| TEST-flaky-bayesian-threads | `test_bayesian_search_forecaster_warnings_of_folds_in_processes_with_trials_in_threads` falló una vez en el job de mínimos (esperaba "in 5 of 8 trials", recibió "in 4 of 8"); probable carrera entre los hilos de los trials | P3 | | S | Visto una vez (run 43472223734) | R28 "DEP-numpy2-pandas23" |
| TEST-flaky-backtesting-threading-warnings | `test_backtesting_forecaster_warnings_of_folds_in_parallel_same_as_n_jobs_1[backend: threading]` falló una vez durante #1513, que no cambia ese caso (`n_jobs=2`); luego pasó 25 de 25 veces solo y en las ejecuciones completas. Probable dependencia del orden en que los hilos emiten los avisos, como TEST-flaky-bayesian-threads | P3 | | S | Visto una vez (2026-11-16); sin reproducir | #1513 (descripción) |
| TEST-unused-imports | 15 imports sin usar en 6 ficheros de tests (`ruff --select F401,F811`): multiserie `test_recursive_predict_bootstrapping.py`, stats `test_predict_interval.py`, clasificador `test_create_predict_X.py`, `test_align_series_and_exog_multiseries.py`, `test_initialize_transformer_series.py` | P3 | | XS | Comprobado el 2026-11-19 | UTILS §4 (T9, N-08) |
| TEST-inplace-transformer-fast-path | Seis tests de #1504 que dicen comprobar que un transformador que escribe en su entrada no la modifica no pasan por la copia de `transform_numpy`: usan `StandardScaler(copy=False)` con `fit=False`, que toma el camino rápido (`_transform_numpy_fast_path` calcula en un array nuevo) y sale antes de copiar. Son `test_set_out_sample_residuals_does_not_modify_y_true_y_pred_with_in_place_transformer` de Recursive, Direct, MultiSeries y DMV, el caso `StandardScaler(copy=False)` de `test_predict_and_calculate_metrics_one_step_ahead_multiseries_ForecasterDirectMultiVariate_with_in_place_transformer` y `test_revert_differentiation_one_step_ahead_with_in_place_transformer`. Añadirles un caso con un `FunctionTransformer` in situ (como en `test_transform_numpy`), con valores esperados fijos | P3 | | XS | Sin empezar. Comprobado el 2026-11-23: sin las tres copias de #1504, esos seis pasan con pandas 2 y 3 (el de `ForecasterRnn` sí falla). La copia la siguen vigilando los tests de las funciones auxiliares y los de `predict`, backtesting y `_calculate_metrics_one_step_ahead` | Revisión de #1504 del 2026-11-23 (scripts no versionados) |
| MAINT-numpy-timedelta-warning | `DeprecationWarning` de numpy (unidad `generic` de timedelta) en `foundation/_adapters.py`, probablemente de `pd.Timedelta(days=n)` en TabICL y TabPFN | P3 | | XS | Sin comprobar que siga | P3 §10 (línea 2234 aprox.); coincide con FOUND-plan-07 |
| MAINT-far-freeze-files | `make_env.sh` de `foundation_adapters_review/evidence/common/` escribe `requirements-freeze.txt` dentro del repo y reactiva Dependabot: escribirlos fuera o no commitearlos | P3 | | XS | Con FOUND-plan-02 (cadena C4) | FAR `evidence/README.md` |
| MAINT-drop-pandas2 | Retirar pandas 2: módulo de compatibilidad, tests condicionados (`PANDAS_GE_3`), celda `pandas<3` de CI, pin y documentación | P3 | | M | Bloqueado: `gluonts` (backends de `MoiraiAdapter` y `TabPFNAdapter`) exige pandas 2. La lista tiene errores (`compat.infer_freq` tiene 2 usos, no 3; el "orden por duración" es `sorted(key=str)`) | P3 §10 "Al retirar pandas 2: lista viva" (moverla aquí antes de borrar `pandas3_migration/`) |
| MAINT-latest-deps-pandas4 | Actualizar el filtro de `unit-tests-latest-deps.yml` cuando salga pandas 4 | P3 | | XS | Bloqueado: hasta que salga pandas 4. Anotado en un comentario del workflow | #1501 |
| MAINT-codecov-pandas2 | codecov mide con pandas 3: las ramas de pandas 2 de `skforecast/utils/_pandas_compat.py` saldrán sin cubrir | P3 | | XS | Sin comprobar | P3 §10 "PR 6" |
| DOC-gif-shap-dayofweek | `tools/docs/figures/gif_shap_values.ipynb` usa `index.dayofweek` | P4 | | XS | | P3 §10 |
| MAINT-numpy-int-args | Aceptar enteros de NumPy (`np.integer`, sin `bool`) en todos los argumentos enteros, de forma uniforme. Hoy `lags`, `steps` e `initial_train_size` de `TimeSeriesFold`, `steps` de los modelos de `stats`, `window_size`, `order` y `nlags` los aceptan, pero `steps` de `ForecasterDirect` y `n_jobs` (`_resolve_n_jobs`, todos los backtesting, búsquedas y forecasters Direct) dan `TypeError`, aunque joblib acepta `np.int64` | P4 | | S | No empezado. Antes, revisar qué otros argumentos enteros los rechazan. En `_resolve_n_jobs` basta aceptar `np.integer` (ya devuelve `int(n_jobs)`). Ya pasaba en 0.27; nadie lo ha reportado | Revisión de #1513 (2026-11-23) |

## 3. Foundation

| ID | Qué | Prioridad | Requiere | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|---|
| BUG-04 | `clone` descarta el pipeline precargado por el usuario (modelos afinados): resultado erróneo silencioso | P1 | | M | No empezado. Falta escribir el plan nuevo a partir del anexo A (la rama del primer arreglo se borró) | Anexo A; R27-repro/BUG-04.py; R27-verify/BUG-04-repro.md |
| FOUND-adapter-arg | Argumento `adapter` en `FoundationModel` (nombre o clase) para cargar checkpoints locales o de repositorios propios; `model_id` describe los pesos | P2 | | S a M | No empezado (`_resolve_adapter` sigue por prefijo) | Anexo A |
| FOUND-plan-02 | Cambios comunes: error con contexto todo NaN (D11), tabla `_UNSUPPORTED_MODEL_IDS` (D-REG), cuantiles a `float` (D-Q), helper de versión mínima (D-VER), `stacklevel` dinámico (el helper ya existe: `skforecast/utils/_warnings.py:_find_stack_level`), textos X-8 | P2 (por FOUND-plan-03) | D-FAR | S a M | No empezado. Va antes que 03 a 09 | FAR `02_common.md` |
| FOUND-plan-03 | TS-ICL: atributo `requires_single_series_with_exog`, `supports_past_only_covariates=False`, cuantiles vía `Decimal`, guarda de horizonte >672, aviso MPS, `context_length>4096` | P2 | FOUND-plan-02 | S a M | No empezado, prioridad alta. TSI-P2 hay que reescribirlo: desde #1479 `FoundationModel` ya quita las columnas solo pasadas | FAR `03_tsicl.md` |
| BUG-06 | `predict()` da la media y `predict_interval()` la mediana en TabICL, Nori y TabPFN con `point_estimate` distinto de `'median'`; `backtesting_foundation` con cuantiles no devuelve `pred` | P2 | | M | No empezado. Requisito de FE-05 y de GAP-foundation-backtesting-interval | R27 §4; R27-repro/BUG-06.py |
| QW-02 | Aviso cuando se ajusta con exog y `predict` no la recibe (Chronos-2, TimesFM 3, TS-ICL pierden el efecto en silencio) | P2 | D-QW02 | S | Decisión pendiente D-QW02 | R27 §2 |
| FOUND-finetune-small | Cargar Chronos-2 con `Chronos2Pipeline.from_pretrained` (LoRA, probar con `peft`); comprobar que `model_path` existe en TabICL y TS-ICL; exponer `model_path` en TS-ICL solo si se pide | P3 | | S | No empezado | Anexo A |
| FOUND-moirai-context-length | Con objeto precargado, buscar `context_length` lo descarta; en Moirai-2 se evita con unas 6 líneas | P3 | BUG-04 | S | Después de BUG-04 | Anexo A |
| FOUND-plan-04 | TimesFM 2.5 y 3: helper de import, `timesfm>=3.0`, quitar `_TimesFMCompat`, "media" por "mediana", `padding_mode`, kwargs reservados | P3 | FOUND-plan-02 | S a M | No empezado | FAR `04_timesfm.md` |
| FOUND-plan-05 | Chronos-2: mínimo `>=2.2`, contrato `cross_learning`, `pd.Categorical` numérica, `device_map` | P3 | FOUND-plan-02 | S a M | No empezado | FAR `05_chronos.md` |
| FOUND-plan-06 | Moirai-2: `uni2ts>=2.0`, no recargar pesos en `set_params`, guarda de cuantiles, `stacklevel` de MPS | P3 | FOUND-plan-02 | S a M | No empezado | FAR `06_moirai.md` |
| FOUND-plan-07 | TabICL: helpers comunes (reemitir avisos ocultos por `redirect_stderr`, timestamps sin `Timedelta(days=n)`), mínimo de versión, cuantiles, docs | P3 | FOUND-plan-02 | S a M | No empezado. Va antes que 08 | FAR `07_tabicl.md` |
| FOUND-plan-08 | TabPFN-TS: helpers de 07, `pipeline_kwargs` (D17), categóricas (D18), mínimo `>=1.4` (D19) | P3 | FOUND-plan-07 | S a M | No empezado (TPF-P4 hecho en #1480) | FAR `08_tabpfn.md` |
| FOUND-plan-09 | Nori: conservar las filas de contexto con covariables NaN (D23), `synthefy-nori>=0.22`, quitar workarounds, docs de Nori-1-5 | P3 | FOUND-plan-02 | S a M | No empezado. NORI-P9 ya no aplica (#1479 quitó `_known_future_columns`) | FAR `09_nori.md` |
| FOUND-plan-10-ci | Caché de los pesos de TabPFN en `foundation-backends.yml` (3.5), entrada `nori15` (3.6), README de `tools/foundation_compat` (3.7) | P3 | | S | No empezado | FAR `10_docs_ci.md` |
| FOUND-plan-10-rerun | Repetir las comprobaciones 3.2 y 3.3 del plan 10 y reejecutar la guía de foundation cuando se fusionen los planes 02 a 09 (la reejecución de #1480 solo incluía el 01) | P3 | FOUND-plan-02 a 09 | S | Después de 02 a 09 | FAR `10_docs_ci.md` |
| QW-05 | `save_forecaster` sin los pesos foundation cargados tras `predict` (0,01 MB frente a 111,81 MB) | P3 | BUG-04 | S | No empezado. Con BUG-04 basta comparar por identidad en `__getstate__` | R27 §2; anexo A |
| FE-02 | `past_only_exog` en `backtesting_foundation` y `bayesian_search_foundation` | P3 | | S | No empezado (BUG-10, requisito, hecho en #1479) | R27 §3 |
| GAP-foundation-backtesting-interval | `interval` en `backtesting_foundation` (hoy solo `quantiles`) | P3 | BUG-06 | S | Después de BUG-06 | R27 §6 |
| PERF-01 | `backtesting_foundation` llama al modelo una vez por fold aunque es zero-shot (3,4 veces más rápido con una serie, nada con 50) | P3 | | M | No empezado | R27 §4; R27-repro/PERF-01.py |
| FOUND-timeh2o | Adaptador TimeH2O (h2ofm, `h2oai/timeh2o-v1`). Especificación completa con esqueleto ejecutable (#1453). Antes de implementar, revisión ligera: base `0.28.x`, Python 3.11, la exog solo pasada ya la filtra `FoundationModel`, Timestral se fusionó con su propio `_warn_if_constant_known_covariates` (D15), recontar adaptadores, requisitos de §11 en `a1a837b4a^` | P3 | D-TIMEH2O | M | Decisión pendiente D-TIMEH2O (D1, licencia no comercial, bloquea el resto) | `dev/timeh2o_adapter/research_timeh2o_adapter.md` (§0 "Estado y cómo retomar") |
| FOUND-ci-ideas | XC-7 (invariantes genéricas en el smoke), XC-12 (carril de CI con la versión mínima de cada backend), matriz del workflow generada desde `adapters.toml` | P4 | | S a M | Ideas sin plan propio | FAR `evidence/cross_cutting.md`; `10_docs_ci.md` §8 |
| FOUND-t0-options | Opciones de `tfc-t0` que #1478 dejó en el valor por defecto: `symmetric_averaging`, `clip_nonnegative`, `snap_integers`, covariables HISTORICAL, `drop_constant_covariates` (T0-P8, D5) | P4 | | S | Pospuesto en #1478 | `91a8d2f13:dev/foundation_adapters_review/01_t0.md`; `evidence/t0/options_060.log` |
| FOUND-suppress-warnings | `suppress_warnings` en `ForecasterFoundation.predict*` (hueco de consistencia con ML) | P4 | | S | No empezado | R27 §5 |
| FOUND-skops | Guardar `ForecasterFoundation` con skops | P4 | QW-05 | S | Posible cuando el backend quede fuera del estado (QW-05) | R27 §5 |

## 4. Backtesting, búsquedas, métricas y paralelización

| ID | Qué | Prioridad | Requiere | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|---|
| BUG-02 | Con `fold_stride < steps` las métricas solo miden los primeros horizontes y las búsquedas eligen por el error a 1 paso | P2 | D-BUG02 | S | Decisión pendiente D-BUG02 | R27 §4; R27-repro/BUG-02.py |
| BUG-01 | `backtesting` con `initial_train_size=None` sobre un forecaster ajustado evalúa in-sample sin aviso (métricas 2,5 veces optimistas) | P2 | | S | Reproducido. Solo aviso silenciable (uso documentado) | R27 §4; R27-repro/BUG-01.py |
| BUG-11 | `backtesting_forecaster` con `ForecasterEquivalentDate` e `interval` da `AttributeError` (`predict_quantiles`) con el método por defecto | P2 | | S | Reproducido | R27 §4; R27-repro/BUG-11.py |
| BUG-backtesting-refit-skipfolds | Backtesting multiserie con refit intermitente y `skip_folds`: `_extract_data_folds_multiseries` vuelve a extraer el entrenamiento cuando cambian sus límites aunque ese fold no se ajuste, y una serie puede predecirse con un modelo que no la vio (`UnknownLevelWarning`) | P2 | D-SKIPFOLDS | S | Visto el 2026-11-19 en un caso (`refit=3`, `skip_folds=2`). Falta la reproducción mínima y decidir si es intencionado | NPREV §9.1 |
| PERF-expand-index-utc | `_is_utc_anchored_index` se ejecuta en cada `expand_index` (0,4 ms con índice diario con zona horaria): −70 % en el backtesting de una serie sin refit con ese índice (Ridge, 50 folds: de 52 a 65 ms a 15 a 20 ms). Hacerlo con la formulación exacta (aritmética entera, sin `normalize` ni `unique`); una caché por serie cambia resultados, porque el valor depende de la ventana | P2 | | S | Medido el 2026-11-19. La formulación exacta coincide con la actual en todas las ventanas probadas; su tiempo no está medido | NPREV §4 (P6) y §9.5 |
| BUG-backtesting-ytrue-missing-date | `backtesting_forecaster` (y `backtesting_stats`) obtiene la columna `y_true` con `y.index.get_indexer(backtest_predictions.index)` y `y.iloc[positions]` (`_validation.py`, buscar `get_indexer`): si una fecha predicha no está en `y`, `get_indexer` devuelve -1 y `y_true` toma la última observación de `y`, sin aviso. La métrica y la columna `step` salen mal. Arreglo: error claro (o NaN y aviso) cuando haya algún -1 | P2 | | XS | Reproducido el 2026-11-23 con BUG-daily-utc-steps-ambiguous: 2 de 40 fechas sin correspondencia, MAE 3,8048 en lugar de 3,5444 (pandas 2.3.3, igual en 0.27.0 y 0.28.x). Cualquier otro caso en que `expand_index` genere fechas que no están en `y` acaba igual | Revisión de #1505 del 2026-11-23 (scripts no versionados) |
| FE-05 | Comparar varios forecasters de cualquier familia en un solo backtesting, sobre los mismos puntos y con la misma escala | P3 | D-FE05, BUG-06 | M | Diseño analizado y prototipado; decisión pendiente D-FE05. Requisito: BUG-06 | R27 §3 ("FE-05: análisis de diseño"); R27-repro/FE-05-prototype.py |
| PERF-03 | Métricas pedidas por nombre en numpy (búsqueda `OneStepAheadFold` con 1000 series: 2,6 veces más rápida con Ridge); incluir RMSE. Segunda parte (M1): en el backtesting multiserie, pasar a las métricas cortes de arrays por nivel en lugar de un grupo de pandas (1000 series y MAE: el bucle por nivel baja de unos 715 a 203 ms con la misma función de scikit-learn, y a 7 ms con la primera parte) | P3 | D-PERF03 | S | No empezado (`_get_metric`). M1: plan escrito y prototipado en memoria, idéntico con las nueve métricas de regresión. M1 medido otra vez el 2026-11-19 (600 series, 50 folds): de 480 a 680 ms a 36 a 40 ms por backtesting, y por cada candidato de una búsqueda. La primera parte vale por sí sola (búsquedas one-step-ahead) y va antes del camino rápido; M1 sola es un 5 % de la llamada y su peso aparece con PERF-backtesting-ms-fast-path (las métricas pasan a ser del 30 al 40 % de lo que queda): va dentro de su paso 2. La fórmula directa no siempre es idéntica bit a bit (4e-16 en un caso) Con LightGBM y datos reales (búsqueda de py59), las métricas son parte de un 3 % de la búsqueda: el 2,6 veces es con Ridge (NPREV §10.6). | R27 §4; R27-repro/PERF-03.py; BTSPEED §2; NPREV §9.3 |
| FE-04 | `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries`, con los mismos folds que el modelo global | P3 | | M | No empezado | R27 §3; R27-verify/FE-04-risk.md, FE-04-value.md |
| BUG-refit-directmultivariate-short | Búsquedas de `ForecasterDirectMultiVariate` con series cortas y refit | P3 | | S a M | No empezado, no reproducido | R27 §4; TRANSP §14.1 |
| BUG-09b | Resto de BUG-09 en `backtesting_stats` (Sarimax, `refit=False`, `fold_stride`); la parte a se hizo en #1419 | P3 | | S | Reproducido. Nicho | R27 §4; R27-repro/BUG-09.py |
| PERF-backtesting-grouped-warnings | Un solo aviso "(in 6 of 7 folds)" en lugar del mismo aviso por fold con `refit=True`, con cualquier `n_jobs`, como ya hacen las búsquedas | P3 | | M | No empezado. Cambia lo que se ve con `n_jobs=1`. Decidir el orden con PERF-backtesting-ms-fast-path: ese camino tiene los avisos de cada fold en máscaras, así que agruparlos allí es trivial | TRANSP §15.5 (alternativa E) |
| PERF-backtesting-folds-positions | Extracción de folds del backtesting multiserie por posición: última ventana montada desde los valores de las series y series de entrenamiento con un solo `iloc` (500 series y LightGBM, 10 folds: −0,5 s de 4,2 s sin refit y −1,3 s de 25,6 s con refit, con un prototipo que incluye la exog por posición) | P3 | | M | Plan escrito y prototipado en memoria, sin implementar. PERF-multiseries-names-pyarrow, que toca la misma función, ya está fusionada (#1530). Con PERF-backtesting-ms-fast-path, F1 (ventana) y F3 (exog de test) quedan absorbidas en el backtesting multiserie: solo hacerlas antes si ese camino se retrasa (siguen sirviendo al backtesting foundation y a la validación one-step-ahead). F2 (series de entrenamiento) mantiene su valor: con refit en cada fold son 3,2 a 3,5 s de 20 a 25 s (600 series, Ridge). 2026-11-19: con el camino rápido aplazado (NPREV §10), F1 y F3 recuperan su valor: mejoran el camino actual sin un segundo camino | BTSPEED §3; NPREV §9.1 |
| PERF-residuals-numpy | Binning de los residuos out-of-sample en numpy (ordenación estable y claves `int` ascendentes: el `rng` se consume en ese orden); búsqueda en array de los factores conformal en lugar de `np.vectorize` (sin caché: no hay punto de invalidación); en los intervalos por bootstrapping, reutilizar las entradas y la predicción puntual. No cambiar `DataFrame.quantile` por `np.quantile`: no es idéntico en pandas 2.3 | P3 | | S | Corrección comprobada con prototipos en los tres entornos; las ganancias del estudio, sin verificar | NPREV §4 (P10, P11, P12) |
| PERF-create-predict-X-astype | `create_predict_X` (Recursive, Classifier, Direct y DMV) restaura los dtypes de la exog con `compat.astype` sin `changed_only`, así que convierte todas las columnas aunque no cambie ninguna (el 90 % de cada llamada, con 10 exog float); el backtesting con `return_predictors=True` la llama una vez por fold. (1) `changed_only=True` en esas llamadas. (2) Que `changed_only` asigne solo las columnas que cambian sobre una copia superficial: hoy filtra el diccionario, pero `astype` con un diccionario recorre todas las columnas del DataFrame y lo reconstruye con `concat`, un bloque por columna. 200 folds sin refit, mejor de 3: LinearRegression con 10 exog float, (1) baja de 0,60 a 0,17 s (p2) y de 0,73 a 0,20 s (p3); con 8 float, una `int` y una `category`, (1) ahorra del 0 al 10 % y (1)+(2) el 36 %; LightGBM con 10 exog float, del 33 % (p2) al 48 % (p3). Con refit, dentro del ruido. Comprobar que (2) no escribe en la entrada con pandas 2, mantener el atajo de columnas duplicadas y revisar `create_train_X_y` y `reshape_exog_long_to_dict`, que ya usan `changed_only` | P3 | | S | Medido con un prototipo el 2026-11-21 (máquina compartida); predictores iguales con `assert_frame_equal` en p2 y p3. Ningún cambio en el paquete | REV28 (fuera de la revisión); `repro/predict_x_astype.py` |
| STUDY-fit-insample-binning | Predicción sobre `X_train` en `fit` para los residuos in-sample: parte sin cambio de resultados (PERF-backtesting-skip-fit-binning) y tope de filas para los bordes de los bins | P3 | | S a M | En curso: #1520 abierta. Decisión de Joaquín sobre el tope (10.000 filas una serie, 1.000 por serie en multiserie) Dato de py59 (NPREV §10.6): en el ajuste final de la búsqueda (`return_best`), la predicción in-sample cuesta de 7 a 9 s (2,5 a 4,6 % de la búsqueda). | PREP (T3) |
| SEARCH-initial-trials | `initial_trials` en las búsquedas bayesianas | P3 | | S | En curso: #1492 abierta | #1492 |
| PERF-n_jobs-next | Siguientes pasos de `n_jobs` tras #1503: (1) arrays grandes sin copia (pickle 5 con `buffer_callback`; KNN, SVR y GP son 2 a 3 veces más lentos con `n_jobs` entero y sin refit); (2) no contar el arranque con los workers vivos; (3) umbral de 12 s de Windows con más máquinas; (4) XGBoost con refit y 4 CPU; (5) caché del forecaster por worker (LightGBM con 2000 folds: 14 de 15 s deserializando); (6) ¿sobra `mmap_mode='c'` en `backtesting_stats`?; (7) reinicio de loky por el tope de folds; (8) backend de hilos y clases de `__main__`; (9) avisos perdidos cuando falla un fold (`return_as="generator"`); (10) umbral en macOS sin medir; (11) propuesta 3 completa (más estimadores que HGB y CatBoost, multiserie sin refit); (12) XGBoost con un hilo al predecir pocas filas (`extra/bench_xgb_predict_threads.py`); (13) umbral de 6 s de Linux, que con pocas CPU parece alto | P3 | | M a L (por partes) | No empezados. (1) y (5) confirmados el 2026-11-21: 200 deserializaciones y una copia privada de un array de 16 MB en 200 folds, frente a 56 y un memmap compartido en 0.27. Dato para (3), de #1538 (Windows, 22 CPU, primera llamada de un proceso nuevo): 365 folds de 24 pasos con HGB, 6 a 8 s de trabajo real (bajo el umbral de 12 s), tardan 6,0 a 6,5 s en paralelo y 6,8 a 7,1 s en un proceso; el umbral parece conservador en esa máquina. Dato para (13), de la revisión de #1538 (Linux, 4 CPU, pandas 3.1.1, primera llamada de un proceso nuevo): el mismo caso, unos 5 s de trabajo estimados con acierto (363 folds por 11 a 15 ms), tarda 4,1 s en paralelo y 5,2 s en un proceso; con 16 CPU, 4 s de CatBoost iban más lentos en paralelo (#1503). El equilibrio parece estar entre 4 y 6 s según la máquina | NJOBS "Siguientes pasos", "Límites", propuestas 3 y 6; REV28 §8 |
| PERF-njobs-auto-first-fit | Con refit, `n_jobs='auto'` estima la carga con el tiempo del `fit` inicial, y el primer `fit` de un proceso también paga costes de arranque (compilar las funciones de numba de las window features): la estimación sale inflada y un backtesting pequeño puede arrancar los workers sin que compense. Medir antes el efecto en tiempo real; si lo hay, estimar con un `fit` caliente o descontar el arranque | P3 | | S | Sin empezar. Medido el 2026-11-23 en Windows, solo el `fit`: con `RollingFeatures(stats='ewm')`, 683 ms el primero frente a 89 ms los siguientes; sin window features, 123 frente a 83 ms. Sin comprobar: el efecto en tiempo real y si pasa igual en la 0.27 (es de esperar: el umbral por carga con refit es de #1475) | Hallazgo al medir #1538, sin script guardado; PERF-njobs-auto-first-fold (sección 13) |
| NJOBS-cpu-count | `n_jobs=-1` usa `cpu_count() - 1`: mantenerlo y documentar el margen, o pasar a `cpu_count()` | P3 | D-NJOBS | XS | Decisión pendiente D-NJOBS | NJOBS propuesta 4 |
| PERF-predict-numpy | Bloque B: que el backtesting valide solo el primer fold cuando los datos lo garantizan (`y` y `exog` sin NaN, mismo índice, sin zona horaria con frecuencia diaria o mayor), limitado a `ForecasterRecursive` y `ForecasterDirect`, con test de equivalencia. El resto de la propuesta (multiserie) está en PERF-backtesting-ms-fast-path | P4 | | M a L | Bloque A hecho (#1516). Bloque B pendiente. Medido sobre el backtesting completo el 2026-11-19 (50 folds, Ridge, exog float): −19 a −21 % sin validar por fold y −25 a −33 % con el índice por posición, es decir, 5 a 7 ms por backtesting; lo que pesa es el índice diario con zona horaria (PERF-expand-index-utc). La condición "cuando los datos lo garantizan" no hace falta: los avisos por fold se reproducen desde máscaras calculadas una vez. La parte de multiserie pasa a PERF-backtesting-ms-fast-path. D-PREDICT-PUBLIC cerrada el 2026-11-20: `check_inputs` sigue siendo de uso interno, como dice su docstring; no se documenta ni se recomienda, y los caminos rápidos del backtesting son internos | Anexo B; P3 §11; R28 fila del mismo ID; NPREV §2.2 y §9.5 |
| PERF-backtesting-ms-fast-path | Backtesting y búsquedas de `ForecasterRecursiveMultiSeries` sin refit que validan y preparan una vez (matriz de series, exog codificada solo en las filas de test, máscaras para los avisos) y recorren los folds en numpy, sobre el núcleo de MAINT-ms-predict-core, con la preparación reutilizable entre los candidatos de una búsqueda. Datos sintéticos (600 series, exog mixta, 50 folds), llamada completa: Ridge de 10 a 12 s a 1,7 a 2,0 s (1,3 s con PERF-03); LightGBM de 12 a 14 s a 3,7 a 4,2 s; búsqueda de 20 candidatos con Ridge de 66 a 77 s a 22 a 24 s. **Casos reales (tres artículos de cienciadedatos.net): x1,0 a x1,8 con sus 9 a 31 folds, porque el `fit` domina; x2,8 a x3,3 con 61 folds (1578 edificios, `steps=1`)**; con refit en cada fold, 1,05 a 1,2 veces | P4 | D-NP4, MAINT-ms-predict-core, PERF-03 (primera parte) | L | Prototipado en memoria (`proto_bt_once.py`, `proto_bt_blocks.py`): 24 de 25 casos idénticos y uno con el mismo error, con y sin refit, y salidas idénticas en todos los casos reales con LightGBM en un hilo, en pandas 2.3 y pandas 3 sin y con `pyarrow`. **Aplazado (2026-11-19, casos reales):** es una mejora para evaluaciones de muchos folds con modelos baratos o medios, no para los flujos de la documentación, y empeora la calidad del código (un segundo camino que reproduce los avisos de `predict`). Hacerlo cuando haya ese perfil de uso o demanda, después de MAINT-ms-predict-core y de la primera parte de PERF-03, con M1 dentro. Decisión D-NP4. Condiciones: un solo núcleo numpy, vuelta al camino actual en todo lo que D-NP4 deja fuera, y la batería de `proto_bt_once_cases.py` y los casos reales convertidos en tests. Aceptación por tiempo de la llamada completa | NPREV §2, §6, §9 y §10 |
| PERF-search-shared-train-matrices | Búsquedas sobre los parámetros del estimador: reutilizar las matrices de entrenamiento entre los candidatos que comparten `lags` y datos. Con Ridge, `_create_train_X_y` es del 62 al 74 % de cada `fit` (600 series) y, tras PERF-backtesting-ms-fast-path y PERF-03, los `fit` son el 75 % de la búsqueda | P4 | | por estimar | Estimado a partir de las piezas (la búsqueda de 20 candidatos bajaría de 22 a 24 s a unos 10 s), sin prototipo Con `OneStepAheadFold` ya existe: la búsqueda cachea la división por `lags` (2 llamadas para 10 trials en py59, NPREV §10.6). Queda para búsquedas con backtesting y modelos baratos: con LightGBM, el resto del `fit` final de py59 es 3,7 s frente a 52 s del estimador. | NPREV §9.2 |
| PERF-ms-osa-metrics | Validación one-step-ahead multiserie por candidato sin `groupby` por nivel: cortes por nivel calculados una vez y arrays. No es "como el DMV": hay que conservar la inversa por nivel, los niveles ausentes y las Series que reciben las métricas propias | P4 | PERF-one-step-ahead-inverse | S a M | Lectura de código; la ganancia del estudio (21 a 29 ms de 75 a 100 por candidato con 100 series), sin verificar. Junto con PERF-one-step-ahead-inverse Medido el 2026-11-19 con la búsqueda de py59 (NPREV §10.6): agrupar por nivel, deshacer la transformación y métricas son 0,9 s por trial con 1578 series, el 3 % de la búsqueda con LightGBM. Solo vale con modelos baratos. | NPREV §4 (P13) |
| DOC-njobs-faq | FAQ de paralelización: LightGBM con `n_jobs=1` y folds en paralelo a partir de 8 núcleos; HGB predice a 7 ms por paso | P4 | | XS | La FAQ ya recomienda un solo nivel de paralelismo; quizá baste | NJOBS propuesta 5 |
| BUG-05 | grid/random con todas las combinaciones fallidas devuelven un DataFrame vacío y el forecaster sin ajustar. Pulido barato: `ValueError(...) from first_exc`, como `bayesian_search` | P4 | | S | Comportamiento deliberado y testeado desde la 0.21; requiere acuerdo de mantenedores | R27 §5; R27-repro/BUG-05.py |
| SEARCH-validate-cv-once | Validar el cv una vez antes del bucle de grid | P4 | | S | Junto con BUG-05 | R27 §5 |
| FE-01 | Función pública de métricas de backtesting por fold, paso, serie y grupos | P4 | BUG-02 | M | Veredicto partido; decidir según demanda (#720). Requiere BUG-02 | R27 §5; R27-verify/FE-01-*.md |
| METRIC-wape | WAPE como métrica (de QW-08) | P4 | | S | sMAPE ya cubre las series con ceros | R27 §5 |
| SEARCH-equivalent-date-ms | `grid_search_equivalent_date` multiserie | P4 | FE-04 | S | Después de FE-04 | R27 §5 |
| ISSUE-995 | Pruning de Optuna en la búsqueda bayesiana | P4 | | M | Sin verificar; pregunta abierta desde 2025-07 | [#995](https://github.com/skforecast/skforecast/issues/995) |
| SHORT-c2-cold-start | Predecir series con menos historia que la ventana (cold start), solo como argumento explícito del usuario | P4 | | M | Aparcada | SHORT |
| PERF-njobs-small-forecaster-bytes | Backtesting en paralelo: enviar el forecaster serializado como `bytes` cuando es pequeño y como array (fichero compartido de joblib) cuando es grande, con un umbral propio. De paso, los forecasters pequeños de más de 1 MB dejarían de depender de la carpeta temporal de joblib (BUG-njobs-temp-folder) | P4 | | S | Propuesta sin validar (2026-11-23). Tres medidas en Windows, 20 folds sin refit, con fichero frente a sin él: 2,7 MB, 0,26 frente a 0,20 s; 54 MB, 0,71 frente a 1,11 s; 180 MB, 2,5 frente a 5,0 s. Antes de hacer nada, validar: dónde está el cruce, Linux (joblib escribe en `/dev/shm`), cuánto pesa en una búsqueda real sin refit, y que no anula `joblib.parallel_config(max_nbytes=...)`. Ganancia esperable pequeña (0,06 s por llamada con 2,7 MB) | Relacionado con PERF-n_jobs-next (1) y (5); medidas de la sesión de BUG-njobs-temp-folder, sin documento |
| FE-njobs-temp-folder-retry | Backtesting en paralelo: si joblib no puede escribir en su carpeta temporal, repetir la llamada con `max_nbytes=None` y un aviso, en lugar de fallar con el error explicado | P4 | | S | Probable descarte: al hacer BUG-njobs-temp-folder se eligió el mensaje claro sin reintento (2026-11-23). Propuesta sin validar: los fallos que la descartaron salen de una sola revisión y no están todos reproducidos (con `ForecasterDirect` los workers se reinician dos veces por candidato; la barra de progreso queda en rojo en notebooks; un `PicklingError` dentro de un fold con `n_jobs=1` o hilos repetiría todos los folds; los datos grandes pasan a copiarse en cada worker; necesita estado en el proceso). Medido: fallar y reintentar cuesta 2,84 s por llamada, frente a 0,15 s de fallar. Validar esos fallos antes de retomarla, y solo si hay demanda | BUG-njobs-temp-folder (#1535, sección 13); ninguna otra librería revisada reintenta (scikit-learn, pandas, optuna, sktime) |

## 5. Forecasters y multiserie

| ID | Qué | Prioridad | Requiere | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|---|
| BUG-ms-exog-rangeindex | `ForecasterRecursiveMultiSeries` con series de RangeIndex: una exog de `predict` con el índice reseteado (empieza en 0) se acepta sin aviso y usa los valores de entrenamiento. Ejemplo: predice `[-2.83, -0.72, -0.79]` frente a los correctos `[-1.77, -0.39, -0.99]`. Tratarlo con QW-06 si el diseño lo permite. Comprobar si lo cambia la rama de MSPERF, que toca la alineación de la exog en `_create_predict_inputs` | P1 | | S | Reproducido el 2026-11-19: resultado erróneo silencioso | R27 §5; R27-verify/QW-06-qw.md |
| BUG-ms-fit-dataframe-categorical | `ForecasterRecursiveMultiSeries` da un DataFrame al estimador en `fit` y numpy en `predict`. Con `categorical_features=None` (o una lista que omite una columna) y una exog `category` cuyos valores no coinciden con sus códigos (10, 20, 30; `0..n-1` declaradas en otro orden), LightGBM y XGBoost (con xgboost 3.4.1, también con sus parámetros por defecto) entrenan con los códigos y predicen con los valores: predicciones erróneas sin aviso. La validación one-step-ahead no lo ve: MAE 3,9, frente a 31,1 de un backtesting equivalente (`steps=1`, sin refit). También falla la vía documentada para `None`, configurar las categóricas en el estimador por índice (LightGBM 2,98 frente a 34,7; XGBoost con `feature_types`; CatBoost con `cat_features`). Con categorías de texto o decimales, `None` da un `TypeError` claro en `fit`; con una lista incompleta, el texto falla en `predict` y los decimales dan el mismo error silencioso. Con el valor por defecto `'auto'` es correcto, y HistGradientBoosting no está afectado. **Arreglo (D-NP2 = (a'), decidida el 2026-11-20):** el estimador recibe en `fit` los mismos valores que en `predict`, y una `category` que `categorical_features` no gestiona se trata como numérica, como en los demás forecasters. Incluye: (1) numpy al estimador en `fit`, en la predicción in-sample, en `set_in_sample_residuals` y en la validación one-step-ahead (aquí, convertir una vez en la división, no por candidato); (2) aviso `DataTypeWarning` en `fit` cuando una `category` queda sin gestionar, que recomiende `'auto'`, en todos los forecasters (los de una serie ya la tratan como numérica, sin avisar); (3) traducir a índices las categóricas que el usuario configura por nombre (`categorical_feature` y `cat_features` en `fit_kwargs`, `categorical_features` de HistGradientBoosting): hoy funcionan en multiserie y las cubren 6 tests; (4) con `encoding='ordinal_category'` y `None`, añadir el índice de `_level_skforecast` a la configuración del usuario, sin sustituirla; (5) CatBoost con categóricas: mantener el DataFrame con los valores (no los códigos) en lugar del array `object` de los forecasters de una serie, que cuesta de un 11 a un 22 % más de `fit` y 260 MB más de pico (300.000 filas por 28 columnas); (6) pasar el bloque numpy con el que ya se construye `X_train`, no una vista de pandas: en pandas 3 es de solo lectura, cambia la comprobación de modificación in situ y Ridge hace una copia más; (7) error claro para una `category` de texto fuera de la lista, también en los forecasters de una serie (hoy, `could not convert string to float`); (8) nota de versión en "Before upgrading": quien use `None` con una `category` recupera el modelo de hoy con `'auto'`, y los forecasters guardados con el bug hay que reajustarlos; actualizar el docstring, la guía `categorical-features.ipynb`, `llms-base.txt` y el mensaje de `check_exog_dtypes` | P1 | PERF-exog-float-matrix, CAT-T2, CAT-T3 | M | Reproducido el 2026-11-19 (pandas 2.3 y 3, también en `0.28.x`). **Opción (a') revisada con un prototipo el 2026-11-20** (`review_scripts/dnp2_proto.patch`, pandas 2.3, 3 sin y con `pyarrow`): con `'auto'`, idéntico al código actual en 112 de 112 configuraciones; one-step-ahead y backtesting coinciden en las 264 que se ejecutan (hoy difieren en 48); el modelo que hoy se ajusta con `None` es idéntico al de `'auto'` en 36 de 36, así que la migración es cambiar ese argumento; `fit` sin coste con LightGBM, XGBoost, HistGradientBoosting y Ridge (caso real de 1.035.168 filas por 40 columnas: de 12,8 a 13,1 s hoy, de 12,2 a 13,1 s con el prototipo). Con el prototipo fallan los tests que fijan el comportamiento que cambia: 2 de la receta antigua (`OrdinalEncoder` más `category` con `None`), que pasan a `'auto'`, y los de modificación in situ (punto 6). Requisitos: CAT-T2 (un LightGBM dentro de `TransformedTargetRegressor` con `ordinal_category` pierde el nivel categórico: MAE 1,78 a 4,23) y CAT-T3 (con `Pipeline`, el índice del nivel provoca su fallo en un caso que hoy funciona). Después de PERF-exog-float-matrix (conversión común de la exog) y, si se hace, de la segunda parte de STUDY-fit-insample-binning (el tope de filas cambia la predicción in-sample de `fit`; la PR #1520 abierta solo cambia `_validation.py` y no interfiere, comprobado el 2026-11-20). El estimador pierde `feature_names_in_` y los nombres propios de LightGBM, XGBoost y CatBoost (`get_feature_importances` no cambia) | NPREV §3.1, §9.4 y §11 |
| QW-ms-category-warning | Paso opcional y provisional de BUG-ms-fit-dataframe-categorical, por si el arreglo tarda: solo el aviso. En el `fit` de `ForecasterRecursiveMultiSeries`, `DataTypeWarning` cuando una columna `category` de la exog queda fuera de `categorical_features` (`None` o lista incompleta): "sus predicciones pueden ser erróneas con LightGBM, XGBoost o CatBoost; usa `categorical_features='auto'`". No cambia ningún resultado y el consejo es el mismo que dará el arreglo, que después solo cambia el texto. Unas 10 líneas en `_create_train_X_y`, más tests | P1 si BUG-ms-fit-dataframe-categorical se retrasa | | XS a S | Propuesto el 2026-11-20, sin implementar. Sobra si el arreglo entra en la misma versión | NPREV §11.9 |
| PERF-exog-float-matrix | Exog `bool`, nullable o numérica de `pyarrow` mezclada con otros dtypes: `exog.to_numpy()` da un array `object` en Recursive, Direct, DMV, Classifier y en el `predict` de multiserie. `fit` de 5 a 6 veces más lento con Ridge (100.000 filas) y, con HistGradientBoosting, `categorical_features='auto'`, una categórica y una `bool`, `predict` falla (`TypeError: ufunc 'isnan' not supported`). Convertir a float solo cuando hoy saldría `object` y con dtypes explícitos (ni `complex` ni `decimal`) | P1 (por BUG-ms-fit-dataframe-categorical) | | S | Medido, y fallo reproducido, el 2026-11-19 en los tres entornos; con el prototipo pasan las cuatro carpetas de tests de esos forecasters. No hay tests de estos dtypes: escribirlos. D-NP3 cerrada el 2026-11-20: todo a float. Los NA de columnas nullable pasan a NaN y decide el estimador, y los enteros mayores que 2^53, que hoy salen exactos de `create_train_X_y`, se convierten también (se redondean: documentarlo). Al implementar: `UInt64` cerca de 2^64 mezclado con float da `TypeError: cannot safely cast` con el prototipo, y `ForecasterRnn` no se pudo ejecutar (sin torch) | NPREV §3.2 |
| CAT-T2 (antes HANDOFF-T2) | Las categóricas nativas no llegan al estimador dentro de `CalibratedClassifierCV` ni de `TransformedTargetRegressor` (`configure_estimator_categorical_features` solo desenvuelve `Pipeline`). Incluir `estimator_has_native_nan_support` con TTR | P1 (por BUG-ms-fit-dataframe-categorical) | | S | Reproducido el 2026-11-19. Requisito de BUG-ms-fit-dataframe-categorical: hoy, en multiserie, un LightGBM envuelto recibe las categóricas por dtype gracias al DataFrame, y con numpy en `fit` las pierde | UTILS §4 T2; revisión §5.7 |
| CAT-T3 (antes HANDOFF-T3) | `Pipeline` con LightGBM o CatBoost y exog categórica: `fit` falla (`Pipeline.fit does not accept the categorical_feature parameter`) | P1 (por BUG-ms-fit-dataframe-categorical) | | S | Reproducido. D-T3 cerrada el 2026-11-20: (a), con un `Pipeline` no se configuran las categóricas nativas y las columnas pasan con su codificación ordinal, como con cualquier otro estimador. Los `Pipeline` con XGBoost o HistGradientBoosting pierden la configuración nativa que hoy reciben: nota en Changed. Al implementar, un test en multiserie con un `Pipeline` que conserva el DataFrame (`set_output(transform="pandas")`), que hoy puede recibir las categóricas por dtype (sin medir). Requisito de BUG-ms-fit-dataframe-categorical. Visto otra vez en NPREV §3.1 y §11.7 | UTILS §4 T3 |
| QW-01 | Aviso cuando `predict()` con `levels=None` deja fuera series que no llegan a la última fecha | P2 | | S | Reproducido (una serie excluida sin aviso) | R27 §2; R27-verify/QW-01-qw.md |
| QW-06 | Alinear `exog` por índice en `predict` de Recursive y Direct | P2 | | S | Reproducido (`ValueError`). Junto con QW-14 (mismo camino) | R27 §2; R27-verify/QW-06-qw.md |
| BUG-07 | El formato largo con MultiIndex solo funciona si sale de `reshape_series_wide_to_long` (de paso, la errata "If it a") | P2 | | S | Reproducido | R27 §4; R27-repro/BUG-07.py |
| BUG-rollingfeatures-nan | `RollingFeatures.transform` ignora los NaN y `min_periods` en predicción; `transform_batch` los respeta. Hacerlo coherente mejora un 2 % en el experimento 5 | P2 | | S | No empezado | TRANSP §7 (experimento 5) |
| ISSUE-1506 | Con un estimador que modifica su entrada en `fit` (`StandardScaler(copy=False)`), `ForecasterDirect` y `ForecasterDirectMultiVariate` predicen mal y `ForecasterRecursive` da residuos in-sample erróneos, sin aviso. MultiSeries ya lo detecta (`_check_in_place_fit`) | P2 | | S a M | Abierto | [#1506](https://github.com/skforecast/skforecast/issues/1506) |
| ISSUE-1508 | `transformer_exog` que devuelve numpy con una columna de texto: todas las exógenas pasan a `object` y `categorical_features='auto'` codifica también las numéricas, sin aviso | P2 | | S | Abierto | [#1508](https://github.com/skforecast/skforecast/issues/1508) |
| BUG-ms-exog-none-bool | Multiserie: si una serie no tiene exog, las columnas `int` pasan a `float64` y una `bool` a `object`; con `categorical_features='auto'` la `bool` se codifica entonces como categórica (con todas las series presentes, no), y con `None` LightGBM falla en `fit` | P2 | | S | Reproducido el 2026-11-19. Relacionado con ISSUE-1509. Mismo patrón en `predict` con pandas 2 (REV28 §8): si a una serie le faltan columnas o no tiene exog, el `pd.concat` de `_create_predict_inputs` da un `FutureWarning` y un `DataTypeWarning` espurio (una `int` pasa a `object`); valores idénticos y sin avisos con pandas 3 | NPREV §5 (I8) y §9.6; REV28 §8 |
| CAT-T4 (antes HANDOFF-T4) | `categorical_features='auto'` sin categóricas resetea la configuración del usuario y da un aviso falso en cada refit; la guía `categorical-features.ipynb` dice "will have no effect" | P2 | BUG-ms-fit-dataframe-categorical | S | Reproducido. D-T4 cerrada el 2026-11-20: (a), avisar una vez en `__init__` cuando el usuario trae su propia configuración categórica y `categorical_features` no es `None`, y quitar los avisos de `fit`; corregir la nota de la guía. Después de BUG-ms-fit-dataframe-categorical: el aviso recomienda `categorical_features=None` para conservar la configuración propia, y en multiserie eso hoy da predicciones erróneas (NPREV §11.2). Visto otra vez en NPREV §9.4 | UTILS §4 T4 |
| QW-14 | Un solo aviso de `exog` con NaN en `predict` de Recursive y Direct (hoy dos, uno habla de "Fitting") | P3 | | S | Reproducido | R27 §2; script en `d24e5b852:dev/roadmap_0.27.0/repro/PR-2c_joblib_warnings.py` |
| QW-15 | Avisos del `fit` en paralelo de Direct y DMV (no usan `_reemit_worker_warnings`) | P3 | | S | Sin medir | R27 §2; TRANSP §15.7 |
| BUG-classifier-nan-last-window | Clasificador sin soporte de NaN y NaN en la última ventana: error genérico de scikit-learn | P3 | | S | Reproducido. Prioridad baja | R27 §4 |
| PERF-reshape-long-to-dict | Usar `_split_long_format` (#1512) en la rama MultiIndex de `reshape_series_long_to_dict` y `reshape_exog_long_to_dict` | P3 | | S | Candidata: medir antes con pandas 2, pandas 3 sin y con `pyarrow` | R28 fila del mismo ID |
| PERF-dmv-last-window | `ForecasterDirectMultiVariate.predict`: ventana con un único `to_numpy` tras seleccionar las columnas y con copia explícita (en pandas 3 sale una vista de solo lectura del `last_window` del usuario); no llamar a `drop(columns=[])` en cada fold (solo lo alcanza el DMV) | P3 | | S | Valores idénticos comprobados; la ganancia del estudio (40 % del `predict`), sin verificar | NPREV §4 (P16, P17) |
| PERF-ms-fit-small | `fit` multiserie: `training_range_` con `v[np.array([0, -1])]` (de 40 a 55 µs a 8 a 10 µs por serie, índice idéntico); exog ancha cortada una vez (solo acierta si todas las series tienen el índice de la exog); dtypes de la exog en diccionario comparados por columna, sin construir un DataFrame | P3 | | S | La primera, medida; las otras dos, comprobadas en corrección | NPREV §4 (P7, P8, P9) |
| PERF-ms-train-pandas3 | Tres costes de `_create_train_X_y` de multiserie que pandas 3 con `pyarrow` encarece. Con las entradas de py59 (1578 series, 22 exógenas en diccionario, 3 de texto), `create_train_X_y` pasa de 2,5 s con pandas 2 a 3,2 s con pandas 3. (1) `check_preprocess_exog_multiseries` hace una copia profunda de la exog de cada serie (`k: v.copy()`): con `PANDAS_GE_3`, `copy(deep=False)`; la copia profunda se queda en pandas 2. (2) `RollingFeatures.transform_batch` y `RollingFeaturesClassification.transform_batch` asignan la lista `self.features_names` a las columnas en cada llamada (una por serie) y pandas 3 construye cada vez un `Index` de texto Arrow: asignar un `pd.Index` creado una sola vez. (3) El `OrdinalEncoder` convierte cada columna de texto Arrow a `object` varias veces (9 conversiones para 3 columnas): convertirla una vez antes de codificar | P3 | | S | Propuesta externa del 2026-11-20 (evaluación de #1530 con 33 artículos de cienciadedatos.net; Linux, pandas 3.1.1 con `pyarrow`), revisada ese día solo por lectura de código: sin medir aquí. Tiempos acumulados de cProfile, pandas 2 / pandas 3: (1) 0,20 / 0,55 s; (2) 0,16 / 0,28 s; (3) sin dato / 0,30 s. Sin perfilador la diferencia total es de 0,7 s, el 1 % del `fit` con LightGBM: pesa con modelos baratos y con refit en cada fold. Por valor: (1), después (3), y (2) solo de paso (cProfile infla las funciones pequeñas; su ganancia real serán decenas de ms con 1578 series). (1) ya estaba visto en P3 §3.1 ("en pandas 3 CoW ya protege") y se dejó como estaba (recomendación 7; decisión del 2026-11-14: como rendimiento, solo los constructores y las 9 copias). Al implementar: en (1), medir también no copiar, y ampliar `test_forecaster_fit_does_not_modify_series_exog_dict` (hoy solo exog numérica) con columnas de texto, un transformador in situ y una sola serie; en (2), `features_names` es un atributo público, `RollingFeaturesClassification` lo reescribe en `fit` y un objeto guardado con una versión anterior no tendrá el `Index` (crearlo en el primer uso), y comprobar antes si poner nombre a cada `Series` evita la construcción (el `concat` puede crear el mismo `Index` desde los nombres); en (3), las cuatro llamadas al codificador (dos al crear las matrices, dos en `predict`) y `categories_` idéntico con valores faltantes. (3) se solapa con la validación V1 de PERF-categorical-codes (ruta para texto). Medir con pandas 2, pandas 3 sin `pyarrow` y con él. Junto con PERF-ms-fit-small (misma función) | Esta fila; P3 §3.1; PREP (V1). El informe y sus scripts (`prof_fit.py`, `callers_setattr.py`) no están en el repositorio: pedirlos a su autor |
| BUG-equivalent-date-nan | `ForecasterEquivalentDate` con NaN en `y`: con `offset` entero el NaN se propaga; con `DateOffset` depende de la función (`np.mean` y `np.max` lo saltan, `np.median` no, una función propia da otro valor), y los residuos in-sample y `predict` del mismo forecaster no coinciden. Decidir antes de vectorizar; `agg_func(axis=...)` solo es idéntico hasta 7 offsets | P3 | D-NP6 | S a M | Reproducido el 2026-11-19. Decisión pendiente D-NP6 | NPREV §4 (P14) y §5 (I3) |
| MAINT-ms-predict-core | Partir el `predict` de `ForecasterRecursiveMultiSeries` en validar, preparar y un núcleo numpy, sin cambio de comportamiento (antes, paso 1 de PERF-backtesting-ms-fast-path). Por calidad, no por tiempo: `_create_predict_inputs` tiene unas 420 líneas, mezcla la validación, la alineación de la exog y las matrices, y la usan cuatro métodos; `_recursive_predict` pide un DataFrame del que solo usa los valores; los atajos de MSPERF (comparaciones `identical`, cachés por `id`) viven dentro porque hoy se valida y alinea en cada llamada | P3 | PERF-exog-float-matrix, BUG-ms-fit-dataframe-categorical | M | Lectura de código, sin prototipo. Después de PERF-exog-float-matrix y BUG-ms-fit-dataframe-categorical, para que el núcleo herede una conversión de la exog común y correcta. Esbozar antes las firmas de las tres etapas para comprobar que simplifica y no solo reparte la complejidad. Red de seguridad: las baterías de identidad de MSPERF | NPREV §8 y §10.5 |
| BUG-ms-missing-categorical-exog (antes N-03) | Multiserie: si falta una columna categórica en la exog de `predict` (ancha o dict), `TypeError: ufunc 'isnan' not supported` del `OrdinalEncoder`; con una numérica el error es claro | P3 | | S | Reproducido el 2026-11-19 (N-03) | Revisión de utils §5.5 (`2377adb24`) |
| BUG-xgbclassifier-lag-only-class | `XGBClassifier` falla en `fit` si una clase solo está en los lags y no es el último código (#1514 arregló el resto del caso) | P3 | | M | Sin issue. Arreglo: renumerar el objetivo | NJOBS "Fallos previos", punto 2 |
| BUG-pandas2-shared-memory | Tres bugs previos que solo pasan con pandas 2: alias entre `X_train` e `y_train` (también los `y_train` de cada paso de `ForecasterDirect`); `X_train` comparte memoria con la salida de una window feature propia; `filter_train_X_y_for_step` sin exógenas cambia el índice del `X_train` del usuario | P3 | | S | Sin issue | P3 §10 "PR 3a", Pendiente, punto 4 |
| MAINT-exog-shallow-copy-note | Los 9 comentarios de la copia superficial de `exog` antes de codificar las categóricas sin `transformer_exog` (`exog.copy(deep=False)`, de #1502: `fit` y `predict` de Recursive, Classifier, Direct y DMV, y `predict` de multiserie) explican que asignar las columnas codificadas las sustituye, pero no avisan de que la garantía depende de la forma `exog[cols] = ...`: con `exog.loc[:, cols] = ...`, pandas 2 escribe en los arrays del usuario sin aviso (pandas 3 no los toca por CoW, y con columnas categóricas o de texto da `TypeError`). Añadir a los 9 comentarios una frase como "Keep the `exog[cols] = ...` form: `exog.loc[:, cols] = ...` writes into the user's arrays with pandas 2", y decirlo en el docstring de `test_predict_does_not_modify_exog_with_categorical_features` | P3 | | XS | Sin empezar. Solo comentarios, sin efecto hoy: con esa mutación fallan los tests `test_predict_does_not_modify_exog_with_categorical_features`, 10 de 20 con pandas 2 (exog del usuario modificada) y 20 de 20 con pandas 3 (`TypeError`), comprobado el 2026-11-23. Va en la próxima PR que toque esos ficheros | Revisión de #1502 del 2026-11-23 (scripts no versionados) |
| REVIEW-dmv-categorical | Revisar `ForecasterDirectMultiVariate` (lags dict, intervalos, NaN al inicio, `differentiation`) y las exógenas categóricas, que el análisis de la 0.27.0 no cubrió | P3 | | M | | R27 §0 "Huecos del análisis", §7 "Cobertura" |
| PERF-one-step-ahead-inverse | Inversa del escalador con `transform_numpy` en la validación one-step-ahead multiserie: la rama de `ForecasterRecursiveMultiSeries` de `_predict_and_calculate_metrics_one_step_ahead_multiseries` llama a `transformer.inverse_transform(predictions_level[["y_true"]])` (y a la de `y_pred`) en lugar de `transform_numpy`, como ya hace la rama de `ForecasterDirectMultiVariate` desde #1504. Dos motivos: (1) rendimiento, porque el camino rápido de `StandardScaler` (#1519) evita la validación de scikit-learn con un DataFrame; (2) protección: hoy no modifica el `y_test` que comparten los candidatos solo porque `pd.DataFrame(dict)` y el `groupby` copian. Sin esas copias, con pandas 3 y un escalador con `copy=False` lo modificaría (la selección `[["y_true"]]` es una vista CoW y scikit-learn la hace escribible), y ningún test lo detecta | P3 | | S | Sin empezar. Subida de P4 (aplazada en #1520 porque `transformer_series` es `None` por defecto) el 2026-11-23, tras medir (Linux, Python 3.12.3, numpy 2.5.5, scikit-learn 1.9.1): una llamada con `StandardScaler`, de 1003 a 15 µs con pandas 3 y de 572 a 6 µs con pandas 2; `grid_search_forecaster_multiseries` con `OneStepAheadFold`, 200 series de 1000 filas, Ridge y 5 candidatos, de 3,7-3,9 a 1,4-1,6 s con pandas 3 y de 2,8-3,2 a 1,4-1,6 s con pandas 2, con las mismas métricas; arrays idénticos bit a bit con `StandardScaler()`, `StandardScaler(copy=False)` y un `FunctionTransformer` in situ. Sin las copias de aguas arriba, `y_test` cambia con pandas 3 y `StandardScaler(copy=False)`. Falta lo que pide PREP (T4) (`MinMaxScaler`, `Pipeline`, avisos, dtype float32 de `y_pred`) y añadir un test con un transformador in situ en esta rama. Antes de PERF-ms-osa-metrics, que quita el `groupby` | PREP (T4); revisión de #1504 del 2026-11-23 (scripts no versionados) |
| PERF-classifier-fast-predict | `ForecasterRecursiveClassifier`: `predict_proba` del estimador en cada paso sin camino rápido, y `predict` sin `check_inputs`. Cada rama de clasificador es código nuevo. Unificar `(a == b).all()` a `.equals` cambia un caso en pandas 2.3 (`datetime64[s]` frente a `[ns]`) | P4 | | M | Lectura de código; ganancia sin medir | NPREV §4 (P15) y §5 (I4, I5) |
| PERF-categorical-codes | Codificar las exógenas `category` por sus códigos en lugar de `OrdinalEncoder` (x1,8 a x1,9 con pandas 3 y `pyarrow`) | P4 | | M | No empezado. Revisado el 2026-11-19: `get_indexer` difiere del `OrdinalEncoder` cuando `fit` vio un valor faltante (`categories_` acaba en NaN; incluye una columna de texto con `None` en pandas 3) y usar `.cat.codes` directamente es incorrecto con otro orden de categorías. En multiserie, una vez por tabla y no serie a serie. Pierde valor con PERF-backtesting-ms-fast-path (el codificador deja de ejecutarse en cada fold): le quedan el `predict` de una serie con categóricas y el `fit` | PREP (T2); NPREV §4 (P4) |
| PERF-scalers-extension | Atajo de `transform_numpy` para `MinMaxScaler` y `RobustScaler` (V3) | P4 | | S | Ampliación de #1519. Pierde valor en el backtesting multiserie con PERF-backtesting-ms-fast-path (el escalador se aplica una vez por serie, no por fold) | PREP (T1, V3) |
| PERF-transformer-copy-allowlist | Desde #1504, `transform_numpy`, `transform_series` y `transform_dataframe` copian siempre su entrada cuando hay transformador (protege de los que escriben en ella o la devuelven tal cual). Con los escaladores habituales esa copia es redundante: `StandardScaler()`, con su `copy=True`, ya copia dentro de `check_array`. Idea: no copiar con una lista de tipos exactos de scikit-learn que se sabe que no escriben en su entrada con `copy=True` (`StandardScaler`, `MinMaxScaler`, `RobustScaler`...), con el mismo criterio de "tipo exacto, no subclase" de `_transform_numpy_fast_path`. **Baja prioridad: valorar en detalle si merece la pena antes de hacerla** (qué tipos entran y cómo se verifica cada uno, versiones de scikit-learn, mantenimiento de la lista, tests de no modificación por tipo) | P4 | | S | Sin empezar. Medido el 2026-11-23 (Linux, Python 3.12.3, numpy 2.5.5, scikit-learn 1.9.1; actual frente a sin copia, mínimo de 3 procesos): `transform_dataframe` aislada, 200.000 x 10, de 1,4 a 2,0 veces el tiempo (se ahorraría la mitad); en flujos reales, de 0 a 5 % (`fit` con Ridge y HGBR, 200 `predict`, backtesting sin refit) y 12 % en `create_train_X_y` multiserie (200 x 2000, 5 exógenas) solo con pandas 2, la versión que se retirará; memoria pico de `fit` sin cambio. La PR #1504 descartó copiar según `transformer.copy` (no protege de transformadores propios que declaren `copy=True` y escriban) | Revisión de #1504 del 2026-11-23 (scripts no versionados) |
| TODO-multivariate-gap | Formato largo con varios niveles y `gap>0` en multivariante (`_validation.py`, buscar el TODO) | P4 | | S | Sin verificar: comprobar antes si es un caso real | R27 §5 |
| BUG-direct-fit-temp-folder | `ForecasterDirect.fit` y `ForecasterDirectMultiVariate.fit` con `n_jobs > 1` y matrices de entrenamiento de más de 1 MB: si joblib no puede escribir en su carpeta temporal, fallan con `PicklingError: Could not pickle the task to send it to the workers.`, sin explicar. El backtesting ya lo explica (BUG-njobs-temp-folder), pero solo dentro de sus folds: no en un `fit` directo ni en el ajuste inicial del backtesting. Propuesta: llamar ahí a `_explain_joblib_temp_folder_error` (moverla a `utils`) | P4 | | XS a S | Propuesta sin validar (2026-11-23): deducido del código y de la revisión de BUG-njobs-temp-folder, sin reproducir con un caso real. Ya pasaba en la 0.27 (no es regresión). Validar antes: reproducirlo y ver si el caso es real (carpeta temporal no escribible). Si se hace, un mismo error puede pasar dos veces por la función (el `fit` dentro de un fold en el proceso principal): hay que evitar que el mensaje se amplíe dos veces | BUG-njobs-temp-folder (#1535, sección 13); `_explain_joblib_temp_folder_error` en `model_selection/_validation.py` |
| BUG-direct-n_jobs-resolved-at-init | `ForecasterDirect` y `ForecasterDirectMultiVariate` guardan en `self.n_jobs` el número ya resuelto en `__init__`: con `n_jobs=-1` en una máquina de 32 CPU guardan 32, y al cargarlo en otra de 4 CPU ajustan con 32 procesos (más lento, no falla). Opciones: guardar lo pedido y resolverlo en `fit`, o volver a resolverlo al cargar | P4 | | S | Ya pasaba en 0.27 (guardaba `cpu_count()`); #1513 solo lo documenta ("always stores the resolved number"). Sin medir; ver antes si `set_params` y `load_forecaster` lo tocan | Revisión de #1513 (2026-11-23) |

## 6. Preprocessing

| ID | Qué | Prioridad | Requiere | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|---|
| ISSUE-1507 | `calculate_distance_from_holiday` da horas en lugar de días con `SME`, `SMS`, `BQE`, `BQS`, `BYE` y `BYS` (faltan en `_freq_to_timedelta_unit`) | P2 | | XS | Abierto. Confirmado otra vez el 2026-11-21 (igual en 0.27): devolver `'D'` salvo que `freq` sea un `Tick` distinto de `Day`. PR externas: #1533 (cerrada) y #1537 (abierta, contra `main`; la base debe ser `0.28.x`) | [#1507](https://github.com/skforecast/skforecast/issues/1507); REV28 §8 |
| ISSUE-1509 | `reshape_exog_long_to_dict` deja una columna booleana como `object` en unas series y `bool` en otras, y MultiSeries rechaza el diccionario | P2 | | S | Abierto | [#1509](https://github.com/skforecast/skforecast/issues/1509) |
| BUG-wide-to-long-inferred-freq | `reshape_series_wide_to_long` con un índice sin frecuencia pierde valores sin aviso cuando pandas infiere una frecuencia que no es la de los datos: con fechas horarias en dos tramos de 9 horas separados por un hueco, `infer_freq` devuelve `BusinessHour` en lugar de `None`, no salta el `ValueError` de fechas irregulares y quedan 2 de 36 valores (con y sin zona horaria, pandas 2.3.3 y 3.1.1). En 0.27 también quedaban 2, con la frecuencia diaria que asumía. `TimeSeriesSplitter` ya no depende de ello (BUG-splitter-long-phase) | P2 | | S | Reproducido el 2026-11-22. Posible arreglo: aceptar la frecuencia inferida solo si su rejilla contiene todas las fechas del índice | `review_028/repro/wide_to_long_inferred_freq.py` |
| QW-11 | `hour_of_week` y días de fin de semana del mes en `CalendarFeatures`, aviso con features anuales y menos de un año de datos, guía de codificación | P3 | D-QW11 | S | Medido (`hour_of_week`: -24 % de MAE con Ridge en horarias). Decisión pendiente D-QW11 | R27 §2 ("QW-11: análisis"); R27-repro/QW-11-calendar-analysis.py |
| FE-06 | Festivos en `CalendarFeatures` con `holidays` como dependencia opcional | P3 | D-FE06 | M | Decisión pendiente D-FE06 | R27 §3 |
| FE-03 | Ventanas estacionales en `RollingFeatures` (`season_length`) | P4 | | M | Mejora el MAE un 2 % y solo con árboles | R27 §3; R27-verify/FE-03-*.md |
| DISC-1153 | Asimetría en `RollingFeatures` y errores difíciles de diagnosticar en window features propias | P4 | | S a M | Sin verificar. FE-03 reduce parte de la necesidad | [Discusión 1153](https://github.com/skforecast/skforecast/discussions/1153) |
| TODO-rolling-classification | TODO de `RollingFeaturesClassification` en `skforecast/preprocessing/_preprocessing.py` | P4 | | XS | Menor | R27 `work/phase1_product_context.md` (`402be62d6`) |

## 7. Stats

| ID | Qué | Prioridad | Requiere | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|---|
| BUG-stats-last-window (antes HANDOFF-STATS-LW) | `ForecasterStats.fit(store_last_window=False)` y después `predict(last_window=...)`: `AttributeError: 'NoneType' object has no attribute 'name'` (el `rename` que añadió T6). Pasa con Arima y Sarimax | P2 | | S | Reproducido el 2026-11-19 | UTILS §5; R27 §5 (HANDOFF-STATS-LW) |
| PERF-02 | Ajuste de Ets y Sarimax de 3 a 70 veces más lento con carga en la máquina | P3 | | S | Volver a medir: BUG-08 (#1425) cambió el escenario | R27 §4; R27-repro/PERF-02.py |
| DEP-arima-dictstyle | `FutureWarning` sin versión desde la 0.22 (`_arima_base.py`) | P3 | D-ARIMA | S | Decisión pendiente D-ARIMA | R27 §6 |
| PERF-arima-seasonal-kalman | Kalman de Arima estacional más rápido (14 ms por evaluación, 729 evaluaciones con n=5000 y m=24) | P4 | | L | | R27 §5; `402be62d6:dev/roadmap_0.27.0/work/lens_stats.md` |
| MAINT-arima-unused-constants | Dos constantes de `_arima_base.py` que no usa nadie: las funciones `@njit` escriben el valor a mano. `_JACOBIAN_STEP_SIZE` (`h = 1e-3` en `compute_arima_transform_gradient`, con una raya larga en el comentario) y `_DIFFUSE_INNOVATION_VARIANCE_THRESHOLD` (`F < 1e4`, dos veces en `_arima_kalman_core`). Cambiar la constante no cambia nada. Usarlas en las funciones o borrarlas | P4 | | XS | Encontrado el 2026-11-23 (revisión de BUG-njobs-temp-folder). Sin efecto en los resultados. Si se usan en las funciones, comprobar que los resultados no cambian (numba toma las constantes del módulo al compilar, con `cache=True`) | Buscar los dos nombres en `skforecast/stats/arima/_arima_base.py` |

## 8. Utilidades, documentación, experimental y deep learning

| ID | Qué | Prioridad | Requiere | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|---|
| DOC-028-review | Correcciones de la revisión: "Before upgrading" (`pip install -U` lleva a pandas 3 a quien tiene pandas < 2.3, y `uv pip install -U` y `--upgrade-strategy eager` lo hacen también con pandas 2.3: la frase "keeps the installed pandas" de la línea 29 solo vale para `pip install -U` con pandas 2.3; comprobado el 2026-11-23 con `pip install --dry-run` y el wheel de `20db7cd`, y la alternativa `pip install -U skforecast "pandas<3"` deja pandas 2.3.3; faltan `n_jobs=0`, `calculate_distance_from_holiday` y el `ValueError` de `ForecasterEquivalentDate` con `Day(n)`), FAQ de pandas 2 y 3, skills, contexto IA y docstrings (20 puntos). Además (revisión de #1510, 2026-11-23): la línea 53 de `docs/faq/pandas-2-and-pandas-3.md` y el punto de `tz_convert` de "Before upgrading" dicen que con `'24h'` los resultados son los mismos que con `'D'` y pandas 2; no es así con datos en pasos UTC a una hora que no es medianoche UTC, donde pandas 2 con `'D'` da un backtesting erróneo (MAE 3,80 frente a 3,54 con `'24h'`; BUG-daily-utc-steps-ambiguous): decir que coinciden con pasos a medianoche UTC y que `'24h'` evita ese fallo de pandas 2. Incluye un cambio de código: el mensaje de error de `reshape_series_wide_to_long` cuando no puede inferir la frecuencia sugiere `data.asfreq('D')`, que con días laborables vuelve a meter los fines de semana como NaN (los días laborables con un festivo dan ese `ValueError` desde #1497); sugerir también `asfreq('B')`, recogerlo en "Before upgrading" y añadir los tests de `B` y de menos de 3 fechas (REV28 §7.6 y §9). Además (revisión de #1535, 2026-11-23): la nota de versión de #1503 dice que, si joblib no puede escribir en su carpeta temporal, el error explica cómo resolverlo, pero no que ese caso funcionaba en 0.27: un forecaster que pasa de 1 MB sin que lo haga ninguno de sus arrays (un `RandomForestRegressor`) ahora necesita esa carpeta, y con ella no escribible falla donde 0.27 daba resultados (`n_jobs=2`, 2,1 MB: 0.27 da MAE 2,5933, 0.28 `PicklingError` con la explicación). Decirlo en la nota | P1 | | S | Pendiente. Regenerar el contexto IA al terminar | REV28 §7; `repro/doc_checks.py` |
| BUG-utc-midnight-anchored-freq | Un índice con zona horaria, frecuencia anclada (`W`, `MS`) y fechas a una hora local que es medianoche UTC (las 19:00 de `America/New_York` en invierno, las 20:00 en verano), sin cambio de hora dentro, se toma por un índice en pasos UTC (`_is_utc_anchored_index`) y `_date_range_from_index` genera las fechas en UTC con una frecuencia cuyo ancla es del calendario local (los domingos locales son lunes en UTC). `reshape_series_wide_to_long` y `reshape_series_long_to_dict` conservan 0 de 6 valores, y las predicciones de `ForecasterRecursive` salen con otras fechas (sábados en lugar de domingos, último día del mes en lugar del primero), sin aviso. Las salidas largas de `TimeSeriesSplitter` ya no dependen de ello (BUG-splitter-long-phase) | P2 | | S | Reproducido el 2026-11-23: igual en 0.27.0 y 0.28.x, pandas 2.3.3 y 3.1.1. Con `D` y con `MS` a las 19:00 en verano sale bien. Posible arreglo: usar los pasos UTC solo si la fecha inicial en UTC cumple la frecuencia (`freq.is_on_offset`). Código compartido: ejecutar los tests de los llamadores | `review_028/repro/utc_midnight_anchored_freq.py` |
| BUG-daily-utc-steps-ambiguous | Índice con zona horaria en pasos UTC a una hora que no es medianoche UTC (datos diarios a las 12:00 UTC en `Europe/Madrid`) cuya ventana (`last_window`) no contiene un cambio de hora: `_is_utc_anchored_index` no lo puede distinguir de uno en pasos locales y elige los locales, así que las fechas predichas que cruzan el cambio salen desplazadas una hora (13:00 en lugar de 14:00), sin aviso, con pandas 2 y 3. Con pandas 3, `predict` con el `exog` real lanza `ValueError`. Con pandas 2, que admite `freq='D'` en ese índice, `predict` acepta el `exog` (el atajo de `_check_exog_alignment` ve la misma frecuencia) y el backtesting da métricas erróneas sin aviso (BUG-backtesting-ytrue-missing-date): basta un fold cuyo horizonte cruce el cambio sin que su ventana lo contenga. Con `freq='24h'` sale bien en las dos versiones. Es la ambigüedad de la regla de reserva de `_is_utc_anchored_index` (solo reconoce pasos UTC a medianoche UTC); BUG-utc-midnight-anchored-freq es el caso contrario | P2 | | S a M | Reproducido el 2026-11-23: igual en 0.27.0 y 0.28.x; no afecta a `calculate_distance_from_holiday` (los festivos son fechas del índice). Posible arreglo: decidir la convención en `fit` con el índice de entrenamiento, que suele contener un cambio de hora, y guardarla, en lugar de deducirla de cada ventana; como mínimo, documentar `'24h'` para datos en pasos UTC. Junto con BUG-utc-midnight-anchored-freq (misma función) | Revisión de #1505 del 2026-11-23 (scripts no versionados) |
| BUG-splitter-positional-cuts | `TimeSeriesSplitter` corta todas las series de un grupo por las posiciones de la primera: con un diccionario de series con tramos distintos, `split_by_date(end_train='2023-01-07')` deja en train fechas hasta el 11 de enero de una serie que empieza el día 5, y `split_by_size(train_size=7)` con series de 10 y 14 valores deja 4 valores de la larga fuera de todos los splits, sin aviso. El docstring pide "identical indexes" en un diccionario, pero nada lo comprueba | P2 | | S | Reproducido el 2026-11-23: igual en 0.27.0 y 0.28.x, pandas 2.3.3 y 3.1.1. Las mismas series en un DataFrame largo con MultiIndex se rechazan (`ValueError`, sin frecuencia). Decidir: error si los índices no son idénticos, o cortar cada serie por fecha. Del mismo módulo, errores visibles e iguales en 0.27.0: las salidas largas con `RangeIndex` y `split_by_date` con un split vacío dan `TypeError` (`data index must be a pandas DatetimeIndex`), y `split_by_size` con un split vacío devuelve un `'long_multi_index'` con solo el nivel `series_id` | `review_028/repro/splitter_positional_cuts.py` |
| BUG-splitter-pandas2-dst-union | Con pandas 2, las salidas `'wide'`, `'long'` y `'long_multi_index'` de `TimeSeriesSplitter` dan un `AssertionError` sin mensaje con un diccionario de series diarias con zona horaria y tramos distintos, cuando un split une dos series solapadas y la primera acaba el día de un cambio de hora. Viene de `DatetimeIndex.union` (lo llama `pd.DataFrame.from_dict`): en pandas 2 `'D'` es un paso fijo de 24 horas y las fechas de `date_range` con zona horaria son días de calendario. `'dict'` está bien | P2 | | XS | Reproducido el 2026-11-23: igual en 0.27.0 y 0.28.x con pandas 2.3.3; con pandas 3.1.1 no falla. `ForecasterRecursiveMultiSeries` (`fit`, `predict` y backtesting) con el mismo diccionario funciona. Solo con tramos distintos: si BUG-splitter-positional-cuts exige índices idénticos, desaparece. Si no, construir la unión de fechas sin `union` (`append`, `unique` y `sort_values`) o esperar a MAINT-drop-pandas2 | `review_028/repro/splitter_pandas2_dst_union.py` |
| DOC-skills-percentiles-stale | Skills que aún hablan de percentiles (dan `ValueError` desde la 0.25; `skills/prediction-intervals/SKILL.md` dice `FutureWarning`) y de que EquivalentDate no tiene `set_params` (sí lo tiene) | P2 | | S | Junto a QW-07 (b); regenerar el contexto IA | R27 §6 |
| DOC-warning-panel-paths | Los paneles de avisos publican en la documentación la ruta absoluta de la máquina que ejecutó los notebooks (unas 20 guías). Que el ejecutor de notebooks muestre la ruta relativa al repositorio | P2 | | S | No empezado | R28 fila del mismo ID |
| BUG-skops-tz-dst | skops guarda pero no carga un forecaster con índice diario (`freq='D'`) de UTC a una zona local cuya `last_window_` cruza un cambio de hora: `ValueError ... does not conform to passed frequency D` en `_compose_index`. Reconstruir el índice sin validar la frecuencia, o documentarlo | P2 | | S | Reproducido el 2026-11-21: previo (igual en 0.27.x) y también de pandas 2 a pandas 2. Con `'24h'` carga en las dos direcciones. Visto también con `'MS'` de UTC a `America/New_York` al cargar con pandas 3. Mirar a la vez (2026-11-23, revisión de BUG-load-pytz; medido con `_decompose_index` y `_compose_index`, sin reproducir con ficheros): un forecaster guardado con pandas 2 y cargado en pandas 3 con zona de `pytz` `'Etc/UTC'` o `pytz.FixedOffset` da `ValueError` ("cannot be rebuilt from its name") al guardarlo con skops; con la misma versión de pandas al guardar y al cargar, `'Etc/UTC'` funciona | REV28 §2; `repro/skops_tz_dst.py` |
| BUG-rnn-old-pickle | `ForecasterRnn` guardado con 0.27: `fit`, `set_params` y el backtesting con `initial_train_size` dan un `AttributeError` (`warm_start`, `_initial_compile_config`, `_initial_weights`). Las notas lo documentan; falta un `ValueError` que diga que hay que crearlo de nuevo | P2 | | XS | Reproducido el 2026-11-21 (keras 3.15.1, torch 2.15) | REV28 §5; `repro/rnn_old_pickle.py` |
| QW-07 | Mensajes de error accionables en cuatro puntos | P3 | | S | (a) y (b) reproducidos con los mensajes antiguos. El prototipo no se guardó | R27 §2; R27-verify/QW-07-qw.md |
| DOC-rnn-exog-example | El ejemplo con exógenas de `forecasting-with-deep-learning-rnn-lstm.ipynb` no es reproducible entre versiones de keras y torch (`EarlyStopping` corta en la época 12 en lugar de la 37; MAE de 47,75 a 76,46). Hacerlo robusto y reejecutar | P3 | | S | Prioridad baja | R28 fila del mismo ID |
| DOC-crps-guide-arviz | `faq/probabilistic-forecasting-crps-score` falla al importar (`arviz-plots` 1.0.0 usa `matplotlib.style.core`, ausente en matplotlib 3.11). Esperar a `arviz-plots` o calcular el CRPS sin `pymc_marketing` | P3 | | S | Las salidas guardadas siguen limpias | R28 fila del mismo ID |
| DOC-lightgbm-reproducibility | Con `n_jobs=None` (su valor por defecto), LightGBM 4.7 entrena con todos los núcleos físicos, ignora `OMP_NUM_THREADS` y sus resultados cambian de una ejecución a otra: dos backtestings idénticos del código actual difieren; con 600 árboles y submuestreo el MAE de algunas series cambia hasta 0,4 entre dos ejecuciones, y hasta un 8 % frente a los valores publicados en el artículo. Nota en la documentación (FAQ de reproducibilidad o guías con LightGBM): fijar `n_jobs` (comprobado con `n_jobs=1`) o `deterministic=True` (documentado por LightGBM, sin comprobar aquí). En los tests, 211 creaciones de `LGBMRegressor` no fijan `n_jobs`; con sus datos pequeños no se han visto fallos, pero es lo primero que mirar ante uno intermitente | P3 | | XS | Visto el 2026-11-19 con el artículo py59. Visto otra vez el 2026-11-20 (informe externo, Linux, 7 ejecuciones de la búsqueda de py59 con `OneStepAheadFold`, igual en la 0.27.0 y en `0.28.x`): el mejor trial y el tercero no cambian, pero los puestos 5, 6 y 10 cambian en el primer o segundo decimal, así que el orden de una búsqueda con trials igualados puede cambiar. Al escribir la nota: LightGBM recomienda `force_row_wise=True` o `force_col_wise=True` junto a `deterministic=True` (sin comprobar aquí) | NPREV §10.2 |
| ISSUE-1131 | Exógenas as-of. FE-02 cubre la parte foundation; #1476 (guía de exógenas pronosticadas, Refs #1131) está abierta | P3 | D-1131 | L | Decisión pendiente D-1131 (alcance del resto) | [#1131](https://github.com/skforecast/skforecast/issues/1131); #1476 |
| MAINT-save-versions | Guardar en `save_forecaster` las versiones de pandas, numpy, scikit-learn y Python y avisar en `load_forecaster` solo si cambia una versión mayor | P3 | | S | Propuesta. Sin atributos nuevos obligatorios | P3 §12 |
| UTILS-save-path (antes HANDOFF-SAVE-PATH) | `save_forecaster` devuelve la ruta final del fichero (hoy `None`) | P4 | | XS | Propuesta | UTILS §5; revisión de utils §5.3 |
| FE-load-pytz-zoneinfo | Cargar sin `pytz` un forecaster guardado con una zona horaria de `pytz`, reconstruyéndola como `zoneinfo`, o una utilidad que convierta un forecaster ya cargado. Hoy el remedio es `pip install pytz` (nota de BUG-load-pytz): volver a guardar con joblib, pickle o cloudpickle en pandas 3 conserva la dependencia, y `ForecasterStats`, `ForecasterRnn` y `ForecasterFoundation` no tienen la salida de skops | P4 | | S a M | Solo con demanda: con `pytz` instalado el fichero carga y predice igual. Prototipo del 2026-11-23, sin guardar: un `pickle.Unpickler` que convierte `pytz._p` en `zoneinfo.ZoneInfo` carga 5 forecasters (`.pkl`) con las mismas predicciones. Costes: joblib necesita su `NumpyUnpickler` privado o un módulo falso global en `sys.modules`; hay que cubrir también `pytz._UTC` y `pytz.FixedOffset`; la clase de la zona cambia sin avisar, y un `datetime` localizado por `pytz` en la hora repetida de un cambio de hora cambia de instante (medido por el revisor) | Descripción de [#1536](https://github.com/skforecast/skforecast/pull/1536) (BUG-load-pytz) |
| UTILS-equivalent-date-last-window | `ForecasterEquivalentDate` guarda la serie entera en `last_window_` (offset mayor que los datos) | P4 | | S | Dudoso: probablemente descartar | Revisión de utils §5.8 (`2377adb24`) |
| BR-synthetic-data | Generador sintético de paneles (rama `feature_synthetic_data_generator` en origin, muy por detrás): rebase y decidir si va en `experimental` o `datasets` | P4 | | M | Sin verificar | R27 §5 |
| ISSUE-974 | skforecast como motor de reducción de sktime | P4 | | L | Sin verificar; estratégico, sin actividad | [#974](https://github.com/skforecast/skforecast/issues/974) |

## 9. Backlog sin fila (P4; aplazados en el triaje de la 0.27.0)

Una línea cada uno; retomar si aparece demanda. Detalle en R27 §7.

- Agregar los avisos por serie en paneles grandes (M; junto a QW-14 y CAT-T4).
- Columna de tiempo por combinación en grid y random (valor bajo: `bayesian_search` ya lo da en `study.trials_dataframe()["duration"]`).
- Calibración conformal de los cuantiles nativos de foundation (M; medir antes la cobertura con `coverage`).
- `search_space="auto"` (M; rangos opinables y mantenimiento continuo).
- `static_features` en modelos globales (M a L; existe la alternativa de exog en dict).
- Pesos de decaimiento temporal para `weight_func` (nicho; la alternativa son 3 líneas).
- Compilación numba de Arima en el primer uso (11,5 s, se paga una vez por entorno).

## 10. Decisiones pendientes

Las de producto, con la propuesta que se hizo. Cerrar antes de implementar el elemento.
Cerradas el 2026-11-20, cada una en la fila de su elemento: D-NP2 = (a') (BUG-ms-fit-dataframe-categorical), D-NP3 = todo a float, NA a NaN (PERF-exog-float-matrix), D-T3 = (a) (CAT-T3), D-T4 = (a) (CAT-T4) y D-PREDICT-PUBLIC = no (PERF-predict-numpy).

| ID | Elemento | Pregunta y propuesta |
|---|---|---|
| D-BUG02 | BUG-02 | Quedarse con la última predicción de cada fecha es deliberado. Propuesta: parámetro nuevo en `TimeSeriesFold` con el comportamiento actual por defecto y aviso cuando `fold_stride < steps`; cambiar el defecto, solo anunciado y en otra versión |
| D-QW02 | QW-02 | Invierte un silencio documentado. Propuesta: hacerlo; en ML el mismo olvido ya da error |
| D-FE05 | FE-05 | Nombre (propuesta: `compare_forecasters` en `model_selection`); métricas sobre los puntos comunes por defecto (propuesta: sí, con aviso y `common_points=False`); alcance de la v1 (propuesta: una serie con todas las familias, y multiserie con modelos globales y foundation; modelos locales por serie en la v2). Entrada como dict `{nombre: forecaster}` |
| D-QW11 | QW-11 | Qué features entran. Las nuevas no entran en la lista por defecto (`features=None`). Los indicadores `month_start`, `month_end`, etc., solo por paridad con feature-engine y mlforecast: medido, no aportan en las 8 series diarias |
| D-FE06 | FE-06 | Si se hace ya: es lo que más mejora las series diarias medidas, pero solo hay 2 series con festivos y el alcance es M |
| D-ARIMA | DEP-arima-dictstyle | Retirar el `FutureWarning` o fijar la versión en que se cumple |
| D-1131 | ISSUE-1131 | Alcance de lo que no cubren FE-02 y #1476 |
| D-NJOBS | NJOBS-cpu-count | `cpu_count()` frente a `cpu_count() - 1` |
| D-TIMEH2O | FOUND-timeh2o | D1 (licencia no comercial y no productiva) y D2 a D15 de la especificación (campo "Decisión" vacío) |
| D-FAR | FOUND-plan-02 a 09 | Unas 27 decisiones de la sección 4 de `foundation_adapters_review/README.md` (D1 a D3 ya se tomaron en #1478, opción a) |
| D-NP4 | PERF-backtesting-ms-fast-path | Alcance de la primera versión. Propuesta: sin refit, `interval=None`, sin `return_predictors`, `n_jobs=1`, y `transformer_exog=None` o una lista blanca de transformadores fila a fila; el resto, por el camino actual |
| D-NP6 | BUG-equivalent-date-nan | NaN en `y`: propagarlo en los dos caminos (`offset` entero y `DateOffset`) o saltarlo en los dos |
| D-SKIPFOLDS | BUG-backtesting-refit-skipfolds | Qué significa saltar un fold con refit intermitente y `skip_folds`: si es intencionado que el entrenamiento se vuelva a extraer aunque ese fold no se ajuste, y que una serie se prediga con un modelo que no la vio |
| D-PERF03 | PERF-03 | Si las métricas en numpy pueden diferir en el último bit de las de scikit-learn (4e-16 en un caso) |
| D-ORDEN | Todo | Nivel de cada fila, columna "Requiere" y orden y cadenas de la sección 11 (propuesta del 2026-11-21, por revisar) |

## 11. Orden de trabajo

Cómo se elige lo siguiente:

1. **Lo que está en curso no adelanta a lo de un nivel superior**: espera su turno por nivel.
   STUDY-fit-insample-binning (#1520), SEARCH-initial-trials (#1492) y la guía de #1476
   (ISSUE-1131) son P3 y van después de P1 y P2.
2. **Las decisiones (sección 10) se abren en cuanto se sabe que hacen falta**: cuestan días de
   calendario, no esfuerzo. Primero las que bloquean un P1 o un P2: D-BUG02, D-SKIPFOLDS,
   D-QW02 y D-FAR.
3. **Por nivel, de P1 a P4**, con el nivel heredado de los requisitos. Dentro de un nivel:
   primero los resultados erróneos sin aviso, después los errores visibles, la documentación
   y el rendimiento; a igualdad, lo más barato y de menos riesgo, lo medido antes que lo
   estimado, y machine learning y foundation antes que stats y deep learning.
4. **Las cadenas mandan sobre el orden dentro del nivel**: los cambios de una cadena van
   seguidos y en su orden, y dos cadenas que tocan el mismo código no van en paralelo.
5. **Antes de publicar**: cerradas las regresiones de P1.1 y lo ligado a la publicación
   (sección 1).

### P1, en este orden

1. **Regresiones y fallos nuevos de la 0.28.x frente a la 0.27.0** (bloquean la publicación):
   cerrados (sección 13). Falta DOC-028-review: corrige las notas de la versión, que deben
   recoger también esos arreglos (cadena C9).
2. BUG-ms-exog-rangeindex (cadena C1).
3. BUG-ms-fit-dataframe-categorical con sus requisitos PERF-exog-float-matrix, CAT-T2 y
   CAT-T3 (cadena C2). Si la cadena se retrasa, QW-ms-category-warning protege mientras tanto.
4. BUG-04 (M; falta escribir el plan; cadena C3).

### P2, por grupos

- **Resultados erróneos sin aviso en configuraciones poco habituales:** ISSUE-1507 (XS),
  BUG-wide-to-long-inferred-freq, BUG-utc-midnight-anchored-freq,
  BUG-splitter-positional-cuts, BUG-rollingfeatures-nan, ISSUE-1506, ISSUE-1508, BUG-01
  (solo un aviso), QW-01, BUG-ms-exog-none-bool (justo después de PERF-exog-float-matrix,
  C2), BUG-backtesting-refit-skipfolds (tras D-SKIPFOLDS), QW-02 (tras D-QW02) y BUG-02 (tras
  D-BUG02, C6).
- **Errores visibles:** BUG-stats-last-window, BUG-11, BUG-07, ISSUE-1509 (C2), QW-06 (C1),
  BUG-skops-tz-dst (C10), BUG-rnn-old-pickle (XS), BUG-splitter-pandas2-dst-union (XS,
  tras decidir BUG-splitter-positional-cuts) y CAT-T4 (C2).
- **Documentación:** DOC-skills-percentiles-stale (con QW-07 (b), C9) y
  DOC-warning-panel-paths.
- **Foundation:** BUG-06 (C5), FOUND-plan-02 y FOUND-plan-03 (tras D-FAR, C4) y
  FOUND-adapter-arg (C3).
- **Rendimiento:** PERF-expand-index-utc.

### P3 y P4

Sin orden fijo: se eligen por las cadenas, el esfuerzo y la demanda. El rendimiento, en el
orden de la subsección "Rendimiento y numpy interno".

### Cadenas

- **C1. Exog de `predict`:** BUG-ms-exog-rangeindex, con QW-06 y QW-14 si el diseño lo permite
  (QW-06 y QW-14 comparten camino en Recursive y Direct). BUG-ms-exog-rangeindex toca el
  `_create_predict_inputs` de multiserie que dejó #1530, igual que C2: no en paralelo con C2.
- **C2. Dtypes y categóricas de la exog multiserie:** PERF-exog-float-matrix,
  BUG-ms-exog-none-bool (comparten la matriz de tests de dtypes, `bool`, nullable y
  `pyarrow`, que hoy no existe y escribe la primera; ISSUE-1509 es el mismo problema en
  `reshape_exog_long_to_dict`: mirarlo a la vez), CAT-T2 y CAT-T3,
  BUG-ms-fit-dataframe-categorical, CAT-T4, MAINT-ms-predict-core y, cuando haya demanda,
  PERF-backtesting-ms-fast-path. Interferencia: el tope de filas de
  STUDY-fit-insample-binning, si se hace, cambia la predicción in-sample de `fit`, así que
  va antes o después de BUG-ms-fit-dataframe-categorical, no a la vez (la PR #1520 abierta
  solo cambia `_validation.py` y no interfiere, comprobado el 2026-11-20).
- **C3. Foundation afinados:** BUG-04, después FOUND-adapter-arg y FOUND-finetune-small; también
  después de BUG-04, FOUND-moirai-context-length y QW-05 (y tras este, FOUND-skops).
  FOUND-adapter-arg convive con la tabla de ids no soportados de FOUND-plan-02: no en
  paralelo.
- **C4. Planes de foundation:** D-FAR, FOUND-plan-02 (con MAINT-far-freeze-files), después 03,
  04, 05, 06 y 09 en cualquier orden, y 07 antes de 08 (MAINT-numpy-timedelta-warning, con
  07); al final, FOUND-plan-10-rerun. FOUND-plan-10-ci es independiente.
- **C5. Backtesting foundation:** BUG-06 antes de GAP-foundation-backtesting-interval y de
  FE-05. FE-02 y PERF-01 tocan también `backtesting_foundation`: seguidas, no en paralelo con
  BUG-06.
- **C6. Métricas del backtesting:** D-BUG02, BUG-02 y después FE-01; BUG-02 va antes de M1 (la
  segunda parte de PERF-03). La primera parte de PERF-03 (tras D-PERF03) va antes de
  PERF-backtesting-ms-fast-path, que lleva M1 dentro. PERF-backtesting-grouped-warnings:
  decidir su orden con PERF-backtesting-ms-fast-path (ese camino tiene los avisos de cada
  fold en máscaras y agruparlos allí es trivial).
- **C7. `fit` multiserie:** PERF-ms-fit-small y PERF-ms-train-pandas3 (misma función),
  seguidas; el punto (3) de la segunda se solapa con la validación V1 de
  PERF-categorical-codes.
- **C8. `n_jobs` (#1503):** PERF-n_jobs-next (sus medidas, con #1535 y #1538 ya
  fusionados) y PERF-njobs-auto-first-fit (el mismo bloque de `_validation.py` que
  #1538).
- **C9. Notas, mensajes y contexto IA:** DOC-028-review después de los arreglos de P1.1;
  QW-07 (b) y DOC-skills-percentiles-stale juntos. El contexto IA se regenera una vez por PR,
  al final.
- **C10. Guardar y cargar:** BUG-skops-tz-dst (después de #1536, que toca la misma FAQ y
  `save-load-forecaster.ipynb`). MAINT-save-versions, UTILS-save-path y
  FE-load-pytz-zoneinfo, sobre el mismo código, si se hacen.
- **C11. Búsquedas:** BUG-05 con SEARCH-validate-cv-once; FE-04 antes de
  SEARCH-equivalent-date-ms.

### Rendimiento y numpy interno

Resultado de `dev/pandas_numpy_review/` (2026-11-19). La ganancia grande
no viene de "usar numpy por dentro" en general (la recursión de `predict` y las matrices de
entrenamiento ya son numpy), sino de dejar de repetir trabajo de pandas en cada fold y en
cada serie. Con datos sintéticos (600 series, 50 folds) la llamada completa ganaba de 3 a 9
veces; con los backtestings reales de tres artículos de cienciadedatos.net (de 9 a 31 folds)
gana de x1,0 a x1,8, porque el `fit` domina, y x3 solo con muchos folds (61). Pequeña o nula
con refit, con modelos caros y en los forecasters de una serie, y en la búsqueda con
`OneStepAheadFold` del artículo el 91 al 98 % del tiempo es LightGBM (NPREV §10). La mayor ganancia
de calidad no es el camino rápido, sino una sola conversión de la exog a numpy en `fit` y en
`predict` (los arreglos del paso 1) y partir `predict` en etapas (MAINT-ms-predict-core). El
mismo orden, con el impacto y las condiciones de cada elemento, está en NPREV §8.

Criterios: corrección antes que velocidad; las decisiones se abren pronto (cuestan días de
calendario, no esfuerzo); un experimento barato que puede cambiar un plan caro va antes que
ese plan; dos cambios sobre el mismo código, seguidos y primero el que fija el comportamiento;
impacto = ganancia por cuántos usuarios la notan, y medido antes que estimado; y un segundo
camino de código es un coste permanente, así que entra con condiciones.

- **P2:** PERF-expand-index-utc (−70 % en el backtesting de una serie con índice diario y
  zona horaria: la única ganancia grande para los forecasters de una serie).
- **P3, por este orden:** PERF-backtesting-folds-positions completa (F1, F2 y F3: mejoran el
  camino actual sin un segundo camino; F2 es la que ayuda con refit), PERF-ms-fit-small y
  PERF-ms-train-pandas3 (C7; medir antes la segunda: sus cifras son de un informe externo),
  PERF-residuals-numpy, PERF-create-predict-X-astype (S; solo con `return_predictors=True` y sin
  refit), PERF-dmv-last-window (medir antes: su ganancia no está verificada),
  DOC-lightgbm-reproducibility y la primera parte de PERF-03. MAINT-ms-predict-core (calidad,
  no tiempo) va en C2.
- **P4, solo con muchos folds o modelos baratos, cuando haya demanda:**
  PERF-backtesting-ms-fast-path (L, con M1 dentro y después de MAINT-ms-predict-core; si
  D-BUG02 está decidida, BUG-02 antes de M1), PERF-search-shared-train-matrices (búsquedas con
  backtesting) y PERF-ms-osa-metrics. Con LightGBM y pocos folds, que es el uso de los
  artículos, todo esto junto es menos del 10 % (NPREV §10.6).
- **P4, poco valor:** PERF-categorical-codes, PERF-scalers-extension, el bloque B de
  PERF-predict-numpy (5 a 7 ms por backtesting; con datos reales, de 3 a 9 % en una serie),
  PERF-classifier-fast-predict y la vectorización de BUG-equivalent-date-nan (ganancia
  pequeña o sin verificar).

## 12. Descartado (para no repetir trabajo)

- **Aviso de infracobertura** en backtesting: con la columna `coverage` no aporta; el umbral
  es arbitrario (2026-10-18).
- **DEP-stats-alpha**: `alpha` se mantiene como alias de `interval` en `ForecasterStats` y
  `backtesting_stats`; pasar los dos da `ValueError` (2026-10-19).
- **Comprobar el dominio de las métricas** sobre `y` antes de entrenar: el error de
  scikit-learn ya es claro y no sirve para `y_pred` ni para métricas propias (2026-10-22).
- **`finetuned_checkpoint` aparte** y detectar la familia del modelo por sus ficheros: los
  sustituye FOUND-adapter-arg (2026-11-11).
- **Fine-tuning desde skforecast** y TimesFM 2.5 afinado con transformers: fuera de alcance
  (2026-11-11).
- **Adaptador TinyTimeMixer** (IBM Granite TTM, antes PLAN-TTM): descartado el 2026-11-19.
  El plan y el código candidato eran de la 0.24.0, anteriores a `_AdapterBase`, y se
  borraron; si se retoma, empezar de cero. Tres lecciones de aquel código: el `forward`
  devuelve los cuantiles en layout (B, Q, T, C); los cuantiles nativos solo existen con
  `multi_quantile_head` en la configuración del checkpoint; y los parámetros de tipo lista
  se guardan por referencia en `__init__` (sin `list()` ni `sorted()`) para que `clone`
  los reconozca.
- **Camino rápido numérico de `predict`** (PERF-predict-numpy): pierde un `TypeError`
  (2026-11-17).
- **De `dev/pandas_numpy_review/`** (2026-11-19, NPREV): predecir todos los folds de un
  backtesting sin refit en una sola pasada recursiva (medido: más lento con Ridge, igual con
  LightGBM); bloques numpy separados por dtype (con la exog codificada a float no hacen
  falta); una caché de `_is_utc_anchored_index` (cambia resultados); `np.quantile` en lugar
  de `DataFrame.quantile` como cambio idéntico (no lo es en pandas 2.3); `RollingFeatures` en
  numba (no es idéntico bit a bit); y, para D-NP2, tanto la opción (a) presentada como
  arreglo sin cambio de modelo como "pasar solo el índice" de la columna, y la opción (c)
  (códigos de pandas en `fit` y en `predict` solo para LightGBM, XGBoost y CatBoost): su
  única ventaja era conservar el modelo de quien hoy acierta con `None`, y ese modelo es
  idéntico al de `'auto'` (NPREV §11.2).
- **Prefijo del paso para las categóricas nativas dentro de un `Pipeline`** (opción (b) de
  D-T3): con `StandardScaler` delante de LightGBM, MAE 5,60 frente a 0,12 y sin aviso; con
  `ColumnTransformer` los índices apuntan a otra columna (UTILS §4 T3; 2026-11-20).
- **`check_inputs=False` como parámetro público** de `predict` (D-PREDICT-PUBLIC): la
  ganancia importa dentro del backtesting, y ahí se resuelve por dentro (2026-11-20).
- Otros descartes de la 0.27.0 (TODOs internos, modelos locales de stats en multiserie,
  `cudf.pandas`, deep learning, categorías nuevas en `predict`, fricciones de foundation,
  ensembles, reconciliación jerárquica, diferenciación estacional, ventanas expanding): R27 §7.

## 13. Hecho desde la última versión publicada (0.27.0)

Se vacía al publicar la 0.28.0. Todo fusionado en `0.28.x`.

- Migración a pandas 3: #1495, #1496 (Python 3.10 retirado), #1497, #1498, #1499, #1500
  (numpy 2.0 y pandas 2.3 como mínimos), #1501 (`pandas<4.0`), #1502, #1504, #1505, #1510.
  Revisiones independientes y CI completa sobre `12898d190` (R28 "MAINT-review-pandas3").
- DOC-pandas3-report: informe de la migración para revisarla sin leer las PRs, con un anexo
  por PR (`dev/pandas3_migration/INFORME_0.28_pandas3.md`, 2026-11-20).
- PERF-long-to-dict: #1512.
- BUG-splitter-long-phase: #1534 (las salidas largas de `TimeSeriesSplitter` conservan las
  series desfasadas, cada una con sus fechas; regresión de la 0.28.x). Hallazgos previos,
  iguales en 0.27.0, en BUG-wide-to-long-inferred-freq, BUG-utc-midnight-anchored-freq,
  BUG-splitter-positional-cuts y BUG-splitter-pandas2-dst-union.
- BUG-njobs-temp-folder: #1535 (si joblib no puede escribir en su carpeta temporal, el
  `PicklingError` del backtesting en paralelo explica la causa y tres remedios:
  `JOBLIB_TEMP_FOLDER`, `joblib.parallel_config(max_nbytes=None)` o `n_jobs=1`; sin
  reintento, y en las búsquedas el aviso de candidatos saltados lleva la explicación).
  Comprobado en Windows con Python 3.13 y 3.14 y joblib 1.3.0 y 1.6.0; Linux y macOS, sin
  comprobar. Lo que queda, sin validar, en PERF-njobs-small-forecaster-bytes,
  FE-njobs-temp-folder-retry y BUG-direct-fit-temp-folder.
- BUG-load-pytz: #1536 (un forecaster guardado con pandas 2 con joblib, pickle o
  cloudpickle y una zona horaria de `pytz` no carga en pandas 3 sin `pytz`: el
  `ModuleNotFoundError` lleva una nota que dice instalarlo, y la FAQ, la guía y el skill
  ya no dicen que carga con cualquier backend). Con `pytz` carga y predice igual; volver a
  guardar con skops queda en la FAQ como opción a comprobar. Comprobado ocultando `pytz`,
  con joblib 1.3.0 y 1.6.0; un entorno sin `pytz`, `ForecasterRnn`, `ForecasterFoundation`,
  Linux y macOS, sin comprobar. Lo que queda, en FE-load-pytz-zoneinfo y en la fila
  BUG-skops-tz-dst.
- PERF-njobs-auto-first-fold: #1538 (con `n_jobs='auto'`, sin refit y 4 folds o más,
  `backtesting_forecaster` predice los dos primeros folds en el proceso principal y estima
  la carga con el más rápido; el fold 0 la inflaba por sí solo: 45 ms de crear el
  `ThreadpoolController` en Windows, 1,2 s con window features de numba y unos 5 ms con
  CatBoost). Primera llamada de un proceso nuevo en Windows (22 CPU; pandas 2.3.3 y 3.1.1
  con y sin `pyarrow`, sin diferencias): 1000 folds de 1 paso, de 5,0 a 5,3 s a 1,0 a
  1,1 s (1,1 s en la 0.27); 300 folds de 6 pasos, de 5,3 a 5,4 s a 2,0 a 2,2 s. Batería de
  968 casos idéntica entre base y rama con pandas 2 y 3. Costes aceptados: con 3 folds ya
  no se paraleliza, y con pocos folds muy caros un fold más se predice en el proceso
  principal. Linux y macOS, sin comprobar. Script en REV28
  `repro/njobs_auto_first_fold.py`. De paso, las notas de la 0.27.0 y la 0.28.0 pierden
  los detalles de las medidas (hardware, tamaños, tiempos absolutos) y el skill
  `release-note` recoge ese criterio.
- PERF-predict-numpy, bloque A: #1516.
- PERF-multiseries-names-pyarrow: #1530 (`predict` y backtesting de multiserie, del 21 al 86 %,
  mismos resultados; informe en MSPERF `MEDIDAS.md` §17). Hallazgos previos sin tocar, en MSPERF
  `SEGUIMIENTO.md`: `levels=[]` con ventana y exog lanza un error de pandas, `predict` con exog
  float y nullable con NA lanza `TypeError` (cerca de PERF-exog-float-matrix) y errores sin
  analizar de la batería de recortes.
- `n_jobs` (serialización única, no más workers que folds, umbral por carga): #1503; arreglos
  #1511, #1513, #1514; BUG-rnn-set-params y refit de `ForecasterRnn`: #1515.
- Atajo de `StandardScaler` en `transform_numpy` (PERF-transform-numpy-scalers): #1518 (plan),
  #1519.
- REVIEW-post-1516: revisión de todo el diff de la 0.27.x a la 0.28.x (2026-11-21, cuatro
  revisores por área): sin regresiones en los resultados frente a la 0.27 ni entre pandas 2.3 y 3;
  sus hallazgos están en las secciones 4 y 8 (REV28).
- STUDY-osa-search-cost: #1524. La búsqueda de py59 con `OneStepAheadFold` es del 91 al 98 %
  LightGBM; lo que no es modelo daría como mucho x1,10 (NPREV §10.6).

---

## Anexo A: BUG-04 y modelos afinados

Copiado de la sección 3.1 de R28 (2026-11-11).

**Decisión.** El arreglo de BUG-04 no entró en la 0.27.0: el tema (usar y cargar modelos
foundation afinados) resultó más grande que el bug. Se planifica entero en la 0.28.0. El plan
antiguo (`PLAN_UnifiedFinetunedCheckpoint.md`, parámetro común `finetuned_checkpoint`) se
eliminó el 2026-11-11 por desactualizado: **falta crear un plan nuevo** a partir de lo de
abajo (la revisión propone otro diseño: un argumento `adapter`).

**Lo que se hizo (descartado).** Hubo una rama local `fix/foundation-clone-preloaded-model`
con el arreglo (4 commits, sin push ni PR), borrada el 2026-11-11. Su diseño, como punto de
partida:

- Contrato: dos atributos obligatorios nuevos en `_REQUIRED_CLASS_ATTRIBUTES`,
  `backend_attribute` (`'_pipeline'`, `'_module'` o `'_model'`) y `backend_param`
  (`'pipeline'`, `'module'`, `'model'` o `None`). Todos los adaptadores aceptan un objeto
  precargado. `_AdapterBase` no es pública.
- `FoundationModel.__sklearn_clone__` clona `get_params()` y pasa al clon la misma
  referencia del objeto del usuario (anotado en `adapter._preloaded_backend`), sin copiar
  pesos; un backend cargado de forma perezosa no pasa al clon. Meterlo en `get_params` se
  descartó: `clone` haría un `deepcopy` (Chronos-2 small: 112 MB y unos 0,1 s por copia; 5
  `deepcopy_forecaster` suman +20 MB de RSS con la referencia frente a +562 MB copiando).
- `set_params` que cambia un parámetro fijado en el backend (`model_id`, dispositivo,
  dtype...) descarta el objeto con `IgnoredArgumentWarning`. El repr muestra solo la clase.
- `save_forecaster` guarda el objeto con sus pesos. Para QW-05 basta comparar por identidad
  con `_preloaded_backend` en `__getstate__` (prototipo: 0,01 MB con backend perezoso,
  111,81 MB con el del usuario).
- `BUG-04.py` (R27-repro): las dos partes dan lo esperado con la rama (A7 cambiado para
  comprobar que `clone` comparte el objeto; parte B con `ft.model.eval()`).

**Pendiente de la revisión del 2026-11-11 sobre esa rama:**

1. Llamar a `eval()` sobre el objeto del usuario antes de predecir: `BaseChronosPipeline.fit`
   (chronos-forecasting 2.3.2) devuelve el modelo en modo entrenamiento y, con dropout
   activo, dos `predict` seguidos difieren hasta 0,35. Los adaptadores ya lo hacen cuando
   cargan ellos el modelo (Moirai-2, T0).
2. Con un objeto precargado, buscar `context_length` lo descarta (con aviso) en TimesFM 2.5,
   Moirai-2, TabICL y TabPFN-TS, porque ese parámetro reinicia el backend: documentarlo; en
   Moirai-2 se evita con unas 6 líneas (FOUND-moirai-context-length).
3. El código de la rama se fusionaba sin conflictos con #1479.

**Revisión de los backends (2026-11-11).** Fuentes: código de chronos-forecasting 2.3.2,
timesfm 3.0.1, uni2ts 2.0.0, tabicl 2.2.0, tabpfn-time-series 1.4.0 con tabpfn 9.2.0, tfc-t0
0.6.1, synthefy-nori 0.23.0, tsicl 0.2.1, timestral git a534bdb, tirex-2 0.5.0, y un
fine-tuning real de Chronos-2 small.

| Backend | Afinar | Ruta local o repositorio propio con la carga actual del adaptador | Checkpoint que dice su familia |
|---|---|---|---|
| Chronos-2 | Oficial: `pipeline.fit(...)`, completo o LoRA | Sí, el completo (`save_pretrained`, probado). LoRA probablemente no: `BaseChronosPipeline.from_pretrained` no tiene la rama de LoRA de `Chronos2Pipeline.from_pretrained` (sin probar, falta `peft`) | Sí (`architectures`, `chronos_pipeline_class`) |
| TimesFM 2.5 | Ejemplo oficial con LoRA sobre el port de transformers | Sí, pero el resultado del ejemplo tiene otro formato (4 de 232 tensores coinciden): no se carga | No |
| TimesFM 3 | No (solo inferencia) | Sí (directorio o `.safetensors`/`.pth`) | No |
| Moirai-2 | No (uni2ts solo afina Moirai 1.x) | Sí | No |
| TabICL | Oficial, pero sobre tablas (`FinetunedTabICLRegressor`) | No por `model_id` (repositorio fijo); sí con `tabicl_config={"model_path": ...}` | En parte |
| TabPFN-TS | Solo el de `tabpfn`, sobre tablas | No por `model_id`; sí con `model_path` en su configuración, solo en modo local | En parte |
| T0 | Lo escribe el usuario (`forward`) | Sí (`PyTorchModelHubMixin`) | No |
| Nori | No aplica (aprendizaje en contexto) | No por `model_id`; sí con `nori_config={"model_path": ...}` | En parte |
| TS-ICL | No aplica (aprendizaje en contexto) | No: el backend tiene `model_path`, skforecast no lo expone | No |
| Timestral | No | Sí (directorio) | No |
| TiRex-2 | Solo en la versión de pago | Sí (directorio) | No |

**En skforecast:**

- `_resolve_adapter` elige el adaptador por el prefijo de `model_id`: una ruta local o un
  repositorio propio (`./chronos2-ft`, `miorg/chronos2-ft`) da `ValueError: No adapter
  found`, aunque el backend sepa cargarlo. El único camino hoy es el objeto precargado, con
  `model_id` como simple etiqueta.
- `_warn_if_non_commercial` también va por el prefijo de `model_id`: con una ruta local se
  perdería el aviso de licencia (copias de TimesFM 3, Moirai-2 o TS-ICL, cuya licencia cubre
  también las versiones afinadas).
- TabICL y TS-ICL descargan en silencio los pesos preentrenados en `model_path` si no existe:
  una ruta mal escrita da el modelo base sin aviso.
- skforecast no afina: el `fit` de los adaptadores solo guarda el contexto.

**Qué necesitan los usuarios:** (1) usar en memoria un modelo recién afinado (BUG-04 con el
punto 1 de arriba); (2) cargar un checkpoint desde disco o desde su propio repositorio
(producción, reproducibilidad, entornos sin internet, copias locales de los pesos base), que
hoy solo bloquea el prefijo de `model_id`.

**Propuesta:**

- BUG-04 con el diseño de arriba, más los puntos 1 y 2.
- FOUND-adapter-arg: argumento `adapter` en `FoundationModel` (nombre o clase) que sustituye
  a la resolución por prefijo: `FoundationModel("./chronos2-ft", adapter="chronos")`.
  `model_id` describe los pesos y llega tal cual a la carga del backend (sirve sin más en
  Chronos-2, TimesFM 2.5 y 3, Moirai-2, T0, Timestral y TiRex-2), y el forecaster guardado
  lleva la ruta, no los pesos (encaja con QW-05). S a M: `_resolve_adapter`, `get_params`,
  repr, `get_model_info`, el aviso de licencia por adaptador y origen de los pesos, y
  convivencia con la tabla de ids no soportados de FOUND-plan-02.
- FOUND-finetune-small: Chronos-2 con `Chronos2Pipeline.from_pretrained` para aceptar LoRA
  (probar con `peft`); comprobar que `model_path` existe antes de pasárselo a TabICL y TS-ICL;
  exponer `model_path` en TS-ICL solo si se pide.
- Descartados y fuera de alcance: sección 12.

## Anexo B: PERF-predict-numpy, lo que queda

Resumen de la fila de R28 (2026-11-17). El detalle y las medidas están en P3 §11 y en la
fila de R28.

- **Bloque A hecho** (#1516): recortes que no cambian qué se valida, idénticos bit a bit.
  Ganancia en el bucle de `predict` con exog: 30 % en `ForecasterRecursive`, 25 % en
  `ForecasterDirect`, 12 a 16 % en el clasificador, 4 a 9 % en DMV y MultiSeries.
- **Bloque B** (PR aparte): que el backtesting valide solo el primer fold cuando los datos lo
  garantizan ("unos 60 a 95 ms más por 1000 folds", según R28), limitado a `ForecasterRecursive` y
  `ForecasterDirect`, con un test de equivalencia frente a validar todos los folds. Ojo: P3
  §11.3 dice que el backtesting no valida en cada fold, y es falso (`check_inputs=True`).
- **Hallazgos sin tocar:** `exog.to_numpy()[:steps]` convierte toda la exog antes de cortar;
  `_is_utc_anchored_index` se ejecuta en cada `expand_index` (0,6 ms con índice diario con
  zona horaria); en los dos forecasters de varias series, con pandas 2 y un índice diario
  anclado a una hora UTC mostrado en una zona con cambio de hora, la validación y la
  predicción crean fechas distintas al cruzar un cambio de hora (2026-11-17: no se abre issue
  por ahora).
- **Resto de la propuesta: diseñado y prototipado para multiserie el 2026-11-19** (NPREV §2,
  §6 y §9; fila PERF-backtesting-ms-fast-path). Respuestas a las preguntas que quedaban:
  todo se valida una vez y los avisos de cada fold (también los NaN de `exog` en el `gap`) se
  emiten desde máscaras, con el mismo texto y en el mismo orden; `transformer_exog` y el
  codificador se aplican una vez a las filas de test (por tramo entre refits); el índice de
  las predicciones sale por posición del índice común; `n_jobs > 1`, los intervalos y
  `return_predictors` quedan fuera de la primera versión (D-NP4). Aceptación: salidas y
  avisos idénticos, y tiempo de la llamada completa.
