# Roadmap de skforecast

Roadmap único y vivo: todo lo pendiente, con la versión objetivo de cada elemento. Sustituye
a los roadmaps por versión (`dev/roadmap_0.27.0/`, borrado, y `dev/roadmap_0.28.0/`,
congelado hasta escribir el informe de la migración a pandas 3). Creado el 2026-11-19 a
partir del inventario de `dev/` sobre `origin/0.28.x` (`3c0711b8a`): el estado de cada
elemento se comprobó entonces con `gh`, `git log` y el código, y los bugs marcados como
"reproducido" se ejecutaron contra ese commit.

## Cómo se usa

- **Una fila por elemento**, con un ID estable. El detalle largo vive en la carpeta o el
  documento que enlaza la columna "Detalle"; aquí va lo necesario para priorizar y empezar.
- **Versión**: `0.28.0` (entra en la versión en desarrollo), `0.29.0`, `posterior` (decidido
  que no entra en la actual), `por decidir` (elemento nuevo sin priorizar) o `futura` (sin
  fecha, depende de algo externo). Aplazar algo es cambiar esta celda, no mover texto.
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
| R28 | `dev/roadmap_0.28.0/roadmap.md` (congelado: fuente del informe de pandas 3) |
| P3 §N | `dev/pandas3_migration/INFORME.md`, sección N (congelado) |
| FAR | `dev/foundation_adapters_review/` |
| NJOBS | `dev/n_jobs_benchmark/REPORT.md` |
| PREP | `dev/preprocessing_speedups/PLAN.md` |
| MSPERF | `30195692f:dev/multiseries_predict_perf/` (borrado; `SEGUIMIENTO.md`, `PLAN.md`, `MEDIDAS.md` con el informe final en §17, y `scripts/`) |
| BTSPEED §N | `dev/backtesting_multiseries_speedups/PLAN.md`, sección N |
| NPREV §N | `dev/pandas_numpy_review/REVISION.md`, sección N (revisión independiente, con dos rondas, de `INFORME.md`, en la misma carpeta; donde difieran vale la revisión) |

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
| `pandas3_migration/` | Estudio, decisiones y arnés de la migración a pandas 3 | Tras DOC-pandas3-report; antes, mover la lista "Al retirar pandas 2" a MAINT-drop-pandas2. El arnés lo usan también PERF-predict-numpy y otros: decidir entonces si pasa a `benchmarks/` o `tools/` |
| `roadmap_0.28.0/` | Roadmap anterior, con el detalle de dependencias y pandas 3 | Tras DOC-pandas3-report, con BUG-04 ya resumido en el anexo A de este documento |

---

## 1. Publicación de la 0.28.0 y migración a pandas 3

| ID | Qué | Versión | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| DOC-pandas3-report | Informe detallado de todos los cambios de la migración a pandas 3 y su porqué. Fuentes: P3 (decisiones D1 a D15, ojo: §1, §7, §8 y §11.3 no recogen lo que se fusionó al final), R28 §1 y §2, cuerpos de las PRs, `docs/releases/releases.md` 0.28.0, `docs/faq/pandas-2-and-pandas-3.md` y los resultados locales del arnés (`harness/results/`, no versionados; la tabla p3/p2 de `pr6_028/cross_env/cross_env_summary.md` solo está ahí) | 0.28.0 | M | Pendiente: después de PERF-multiseries-names-pyarrow. Decidir antes el público (interno en `dev/` o publicado) | PRs: #1443, #1444, #1445, #1448 (0.27.x); #1495, #1496, #1497, #1498, #1499, #1500, #1501, #1502, #1504, #1505, #1510 (0.28.x); relacionadas #1512, #1516 |
| MAINT-pandas3-final-crossenv | Comparación final p3 frente a p2 del arnés sobre la 0.28.x definitiva (la última es de antes de #1502) | 0.28.0 | S | Pendiente, para DOC-pandas3-report | P3 §10; `dev/pandas3_migration/harness/` |
| MAINT-bench-pandas3 | Pasar `benchmarks/` a pandas 3 con una medición puente: tanda con `pandas==2.3.3`, PR que cambia solo ese pin y guarda la versión de `pyarrow`, segunda tanda (sin `pyarrow`, como el entorno de la CI), una tercera con `pyarrow` instalado (donde pandas 3 perdía más frente a pandas 2 y donde más gana PERF-multiseries-names-pyarrow: es la cifra que verán muchos usuarios), filtrar por `pandas_version` en `docs/faq/skforecast-benchmark.ipynb` y comprobar que su entorno tiene pandas 3 (un `benchmark.joblib` escrito con pandas 3 no se abre con pandas 2) | 0.28.0 | S | Pendiente: al final de la 0.28.0, con todo el código fusionado | R28 "MAINT-bench-pandas3" (pasos y estudio previo del 2026-11-16) |
| MAINT-docs-pandas3 | Reejecutar con pandas 3 los 5 notebooks cuyas salidas cambian: `[ns]` a `[us]` en `faq/cyclical-features-time-series`, `user_guides/forecasting-statistical-models` y `user_guides/forecasting-with-deep-learning-rnn-lstm`; `object` a `str` en `user_guides/categorical-features` y `user_guides/autoregressive-classification-forecasting` | 0.28.0 | S | Pendiente: cuando el entorno de la documentación use pandas 3 | R28 tabla de la sección 1 |
| REL-conda-forge-028 | Feedstock de conda-forge al publicar: Python >=3.11, numpy >=2.0, pandas >=2.3,<4.0, scipy >=1.13, numba >=0.60, statsmodels >=0.14.6, matplotlib >=3.8.4 | 0.28.0 | XS | Pendiente: al publicar | R28 "DEP-python310" y "DEP-numpy2-pandas23" |
| REL-skforecast-ai-python311 | Abrir un issue en `skforecast/skforecast-ai` para subir su mínimo a Python 3.11 y, después, actualizar `docs/skforecast-ai/skforecast-ai.md` (líneas 3 y 30 dicen 3.10) | 0.28.0 | XS | Pendiente | R28 "DEP-python310" |
| REVIEW-post-1516 | El código fusionado en `0.28.x` después del 2026-11-16 (#1511 a #1519 y lo que entre después) no tiene revisión independiente | 0.28.0 | S | Por decidir si se hace antes de la PR de la versión | R28 "MAINT-review-pandas3" |
| MAINT-drop-pandas2 | Retirar pandas 2: módulo de compatibilidad, tests condicionados (`PANDAS_GE_3`), celda `pandas<3` de CI, pin y documentación | futura | M | Bloqueado: `gluonts` (backends de `MoiraiAdapter` y `TabPFNAdapter`) exige pandas 2. La lista tiene errores (`compat.infer_freq` tiene 2 usos, no 3; el "orden por duración" es `sorted(key=str)`) | P3 §10 "Al retirar pandas 2: lista viva" (moverla aquí antes de borrar `pandas3_migration/`) |
| MAINT-latest-deps-pandas4 | Actualizar el filtro de `unit-tests-latest-deps.yml` cuando salga pandas 4 | futura | XS | Anotado en un comentario del workflow | #1501 |
| MAINT-codecov-pandas2 | codecov mide con pandas 3: las ramas de pandas 2 de `skforecast/utils/_pandas_compat.py` saldrán sin cubrir | por decidir | XS | Sin comprobar | P3 §10 "PR 6" |

## 2. Dependencias y mantenimiento

| ID | Qué | Versión | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| MAINT-docs-deps | Subir `mkdocs-material`, `mkdocstrings`, `mkdocstrings-python` y `pymdown-extensions` (`black` ya se subió en `a1a837b4a`); no subir `mkdocs-jupyter` 0.27.0 (carga MathJax desde un CDN, choca con KaTeX y con el plugin `privacy`). Añadir `pymdown-extensions` al `ignore` de `.github/dependabot.yml` (#1517 está abierta contra `main`) | 0.28.0 | S | Opcional: medido, sin efecto visible | R28 "MAINT-docs-deps" |
| MAINT-zensical | `mkdocs-material` llega a fin de vida el 2027-05-05; sucesor: Zensical (https://zensical.org/docs/compatibility/mkdocs/migration/) | por decidir | M a L | Sin analizar | R28 "MAINT-docs-deps" |
| MAINT-test-deps | Decidir si `pandas[parquet]>=2.3` del extra `test` lleva también `<4.0` (el core lo tiene). El resto de la limpieza está hecho | 0.28.0 | XS | Pendiente de decidir | R28 "MAINT-test-deps" |
| MAINT-tensorflow-223 | Cuando salga tensorflow 2.22 final: `<2.23`, revisar el marcador `python_version < "3.14"` de `pyproject.toml` y los `skipif` de los tres tests de TensorFlow | futura | XS | Bloqueado: hoy solo hay 2.22.0rc0 | R28 §2 |
| TEST-flaky-bayesian-threads | `test_bayesian_search_forecaster_warnings_of_folds_in_processes_with_trials_in_threads` falló una vez en el job de mínimos (esperaba "in 5 of 8 trials", recibió "in 4 of 8"); probable carrera entre los hilos de los trials | por decidir | S | Visto una vez (run 43472223734) | R28 "DEP-numpy2-pandas23" |
| TEST-unused-imports | 15 imports sin usar en 6 ficheros de tests (`ruff --select F401,F811`): multiserie `test_recursive_predict_bootstrapping.py`, stats `test_predict_interval.py`, clasificador `test_create_predict_X.py`, `test_align_series_and_exog_multiseries.py`, `test_initialize_transformer_series.py` | por decidir | XS | Comprobado el 2026-11-19 | UTILS §4 (T9, N-08) |
| MAINT-numpy-timedelta-warning | `DeprecationWarning` de numpy (unidad `generic` de timedelta) en `foundation/_adapters.py`, probablemente de `pd.Timedelta(days=n)` en TabICL y TabPFN | por decidir | XS | Sin comprobar que siga | P3 §10 (línea 2234 aprox.); coincide con FOUND-plan-07 |
| DOC-gif-shap-dayofweek | `tools/docs/figures/gif_shap_values.ipynb` usa `index.dayofweek` | por decidir | XS | | P3 §10 |
| MAINT-far-freeze-files | `make_env.sh` de `foundation_adapters_review/evidence/common/` escribe `requirements-freeze.txt` dentro del repo y reactiva Dependabot: escribirlos fuera o no commitearlos | con FOUND-plan-02 | XS | | FAR `evidence/README.md` |
| DEP-029-removals | Quitar las deprecaciones de la 0.27.0: DEP-cv-differentiation (#1423) y DEP-datasets-version (#1440) | 0.29.0 | S | | #1423, #1440 |

## 3. Foundation

| ID | Qué | Versión | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| BUG-04 | `clone` descarta el pipeline precargado por el usuario (modelos afinados): resultado erróneo silencioso | 0.28.0 | M | No empezado. Falta escribir el plan nuevo a partir del anexo A (la rama del primer arreglo se borró) | Anexo A; R27-repro/BUG-04.py; R27-verify/BUG-04-repro.md |
| FOUND-adapter-arg | Argumento `adapter` en `FoundationModel` (nombre o clase) para cargar checkpoints locales o de repositorios propios; `model_id` describe los pesos | 0.28.0 | S a M | No empezado (`_resolve_adapter` sigue por prefijo) | Anexo A |
| FOUND-finetune-small | Cargar Chronos-2 con `Chronos2Pipeline.from_pretrained` (LoRA, probar con `peft`); comprobar que `model_path` existe en TabICL y TS-ICL; exponer `model_path` en TS-ICL solo si se pide | 0.28.0 | S | No empezado | Anexo A |
| FOUND-moirai-context-length | Con objeto precargado, buscar `context_length` lo descarta; en Moirai-2 se evita con unas 6 líneas | 0.28.0 | S | Después de BUG-04 | Anexo A |
| FOUND-plan-02 | Cambios comunes: error con contexto todo NaN (D11), tabla `_UNSUPPORTED_MODEL_IDS` (D-REG), cuantiles a `float` (D-Q), helper de versión mínima (D-VER), `stacklevel` dinámico (el helper ya existe: `skforecast/utils/_warnings.py:_find_stack_level`), textos X-8 | 0.28.0 | S a M | No empezado. Va antes que 03 a 09 | FAR `02_common.md` |
| FOUND-plan-03 | TS-ICL: atributo `requires_single_series_with_exog`, `supports_past_only_covariates=False`, cuantiles vía `Decimal`, guarda de horizonte >672, aviso MPS, `context_length>4096` | 0.28.0 | S a M | No empezado, prioridad alta. TSI-P2 hay que reescribirlo: desde #1479 `FoundationModel` ya quita las columnas solo pasadas | FAR `03_tsicl.md` |
| FOUND-plan-04 | TimesFM 2.5 y 3: helper de import, `timesfm>=3.0`, quitar `_TimesFMCompat`, "media" por "mediana", `padding_mode`, kwargs reservados | 0.28.0 | S a M | No empezado | FAR `04_timesfm.md` |
| FOUND-plan-05 | Chronos-2: mínimo `>=2.2`, contrato `cross_learning`, `pd.Categorical` numérica, `device_map` | 0.28.0 | S a M | No empezado | FAR `05_chronos.md` |
| FOUND-plan-06 | Moirai-2: `uni2ts>=2.0`, no recargar pesos en `set_params`, guarda de cuantiles, `stacklevel` de MPS | 0.28.0 | S a M | No empezado | FAR `06_moirai.md` |
| FOUND-plan-07 | TabICL: helpers comunes (reemitir avisos ocultos por `redirect_stderr`, timestamps sin `Timedelta(days=n)`), mínimo de versión, cuantiles, docs | 0.28.0 | S a M | No empezado. Va antes que 08 | FAR `07_tabicl.md` |
| FOUND-plan-08 | TabPFN-TS: helpers de 07, `pipeline_kwargs` (D17), categóricas (D18), mínimo `>=1.4` (D19) | 0.28.0 | S a M | No empezado (TPF-P4 hecho en #1480) | FAR `08_tabpfn.md` |
| FOUND-plan-09 | Nori: conservar las filas de contexto con covariables NaN (D23), `synthefy-nori>=0.22`, quitar workarounds, docs de Nori-1-5 | 0.28.0 | S a M | No empezado. NORI-P9 ya no aplica (#1479 quitó `_known_future_columns`) | FAR `09_nori.md` |
| FOUND-plan-10-ci | Caché de los pesos de TabPFN en `foundation-backends.yml` (3.5), entrada `nori15` (3.6), README de `tools/foundation_compat` (3.7) | 0.28.0 | S | No empezado | FAR `10_docs_ci.md` |
| FOUND-plan-10-rerun | Repetir las comprobaciones 3.2 y 3.3 del plan 10 y reejecutar la guía de foundation cuando se fusionen los planes 02 a 09 (la reejecución de #1480 solo incluía el 01) | 0.28.0 | S | Después de 02 a 09 | FAR `10_docs_ci.md` |
| FOUND-ci-ideas | XC-7 (invariantes genéricas en el smoke), XC-12 (carril de CI con la versión mínima de cada backend), matriz del workflow generada desde `adapters.toml` | posterior | S a M | Ideas sin plan propio | FAR `evidence/cross_cutting.md`; `10_docs_ci.md` §8 |
| FOUND-t0-options | Opciones de `tfc-t0` que #1478 dejó en el valor por defecto: `symmetric_averaging`, `clip_nonnegative`, `snap_integers`, covariables HISTORICAL, `drop_constant_covariates` (T0-P8, D5) | posterior | S | Pospuesto en #1478 | `91a8d2f13:dev/foundation_adapters_review/01_t0.md`; `evidence/t0/options_060.log` |
| BUG-06 | `predict()` da la media y `predict_interval()` la mediana en TabICL, Nori y TabPFN con `point_estimate` distinto de `'median'`; `backtesting_foundation` con cuantiles no devuelve `pred` | 0.28.0 | M | No empezado. Requisito de FE-05 y de GAP-foundation-backtesting-interval | R27 §4; R27-repro/BUG-06.py |
| QW-05 | `save_forecaster` sin los pesos foundation cargados tras `predict` (0,01 MB frente a 111,81 MB) | 0.28.0 | S | No empezado. Con BUG-04 basta comparar por identidad en `__getstate__` | R27 §2; anexo A |
| QW-02 | Aviso cuando se ajusta con exog y `predict` no la recibe (Chronos-2, TimesFM 3, TS-ICL pierden el efecto en silencio) | 0.28.0 | S | Decisión pendiente D-QW02 | R27 §2 |
| FE-02 | `past_only_exog` en `backtesting_foundation` y `bayesian_search_foundation` | 0.28.0 | S | No empezado (BUG-10, requisito, hecho en #1479) | R27 §3 |
| GAP-foundation-backtesting-interval | `interval` en `backtesting_foundation` (hoy solo `quantiles`) | 0.28.0 | S | Después de BUG-06 | R27 §6 |
| PERF-01 | `backtesting_foundation` llama al modelo una vez por fold aunque es zero-shot (3,4 veces más rápido con una serie, nada con 50) | 0.28.0 | M | No empezado | R27 §4; R27-repro/PERF-01.py |
| FOUND-suppress-warnings | `suppress_warnings` en `ForecasterFoundation.predict*` (hueco de consistencia con ML) | posterior | S | No empezado | R27 §5 |
| FOUND-skops | Guardar `ForecasterFoundation` con skops | posterior | S | Posible cuando el backend quede fuera del estado (QW-05) | R27 §5 |
| FOUND-timeh2o | Adaptador TimeH2O (h2ofm, `h2oai/timeh2o-v1`). Especificación completa con esqueleto ejecutable (#1453). Antes de implementar, revisión ligera: base `0.28.x`, Python 3.11, la exog solo pasada ya la filtra `FoundationModel`, Timestral se fusionó con su propio `_warn_if_constant_known_covariates` (D15), recontar adaptadores, requisitos de §11 en `a1a837b4a^` | por decidir | M | Decisión pendiente D-TIMEH2O (D1, licencia no comercial, bloquea el resto) | `dev/timeh2o_adapter/research_timeh2o_adapter.md` (§0 "Estado y cómo retomar") |

## 4. Backtesting, búsquedas, métricas y paralelización

| ID | Qué | Versión | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| FE-05 | Comparar varios forecasters de cualquier familia en un solo backtesting, sobre los mismos puntos y con la misma escala | 0.28.0 | M | Diseño analizado y prototipado; decisión pendiente D-FE05. Requisito: BUG-06 | R27 §3 ("FE-05: análisis de diseño"); R27-repro/FE-05-prototype.py |
| BUG-02 | Con `fold_stride < steps` las métricas solo miden los primeros horizontes y las búsquedas eligen por el error a 1 paso | 0.28.0 | S | Decisión pendiente D-BUG02 | R27 §4; R27-repro/BUG-02.py |
| PERF-03 | Métricas pedidas por nombre en numpy (búsqueda `OneStepAheadFold` con 1000 series: 2,6 veces más rápida con Ridge); incluir RMSE. Segunda parte (M1): en el backtesting multiserie, pasar a las métricas cortes de arrays por nivel en lugar de un grupo de pandas (1000 series y MAE: el bucle por nivel baja de unos 715 a 203 ms con la misma función de scikit-learn, y a 7 ms con la primera parte) | 0.28.0 | S | No empezado (`_get_metric`). M1: plan escrito y prototipado en memoria, idéntico con las nueve métricas de regresión. M1 medido otra vez el 2026-11-19 (600 series, 50 folds): de 480 a 680 ms a 36 a 40 ms por backtesting, y por cada candidato de una búsqueda. La primera parte vale por sí sola (búsquedas one-step-ahead) y va antes del camino rápido; M1 sola es un 5 % de la llamada y su peso aparece con PERF-backtesting-ms-fast-path (las métricas pasan a ser del 30 al 40 % de lo que queda): va dentro de su paso 2. La fórmula directa no siempre es idéntica bit a bit (4e-16 en un caso) Con LightGBM y datos reales (búsqueda de py59), las métricas son parte de un 3 % de la búsqueda: el 2,6 veces es con Ridge (NPREV §10.6). | R27 §4; R27-repro/PERF-03.py; BTSPEED §2; NPREV §9.3 |
| BUG-01 | `backtesting` con `initial_train_size=None` sobre un forecaster ajustado evalúa in-sample sin aviso (métricas 2,5 veces optimistas) | 0.28.0 | S | Reproducido. Solo aviso silenciable (uso documentado) | R27 §4; R27-repro/BUG-01.py |
| BUG-11 | `backtesting_forecaster` con `ForecasterEquivalentDate` e `interval` da `AttributeError` (`predict_quantiles`) con el método por defecto | 0.28.0 | S | Reproducido | R27 §4; R27-repro/BUG-11.py |
| FE-04 | `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries`, con los mismos folds que el modelo global | 0.28.0 | M | No empezado | R27 §3; R27-verify/FE-04-risk.md, FE-04-value.md |
| BUG-refit-directmultivariate-short | Búsquedas de `ForecasterDirectMultiVariate` con series cortas y refit | 0.28.0 | S a M | No empezado, no reproducido | R27 §4; TRANSP §14.1 |
| BUG-09b | Resto de BUG-09 en `backtesting_stats` (Sarimax, `refit=False`, `fold_stride`); la parte a se hizo en #1419 | 0.28.0 | S | Reproducido. Nicho | R27 §4; R27-repro/BUG-09.py |
| PERF-backtesting-grouped-warnings | Un solo aviso "(in 6 of 7 folds)" en lugar del mismo aviso por fold con `refit=True`, con cualquier `n_jobs`, como ya hacen las búsquedas | 0.28.0 | M | No empezado. Cambia lo que se ve con `n_jobs=1`. Decidir el orden con PERF-backtesting-ms-fast-path: ese camino tiene los avisos de cada fold en máscaras, así que agruparlos allí es trivial | TRANSP §15.5 (alternativa E) |
| PERF-predict-numpy | Bloque B: que el backtesting valide solo el primer fold cuando los datos lo garantizan (`y` y `exog` sin NaN, mismo índice, sin zona horaria con frecuencia diaria o mayor), limitado a `ForecasterRecursive` y `ForecasterDirect`, con test de equivalencia. El resto de la propuesta (multiserie) está en PERF-backtesting-ms-fast-path | 0.28.0 | M a L | Bloque A hecho (#1516). Bloque B pendiente. Medido sobre el backtesting completo el 2026-11-19 (50 folds, Ridge, exog float): −19 a −21 % sin validar por fold y −25 a −33 % con el índice por posición, es decir, 5 a 7 ms por backtesting; lo que pesa es el índice diario con zona horaria (PERF-expand-index-utc). La condición "cuando los datos lo garantizan" no hace falta: los avisos por fold se reproducen desde máscaras calculadas una vez. La parte de multiserie pasa a PERF-backtesting-ms-fast-path. D-PREDICT-PUBLIC cerrada el 2026-11-20: `check_inputs` sigue siendo de uso interno, como dice su docstring; no se documenta ni se recomienda, y los caminos rápidos del backtesting son internos | Anexo B; P3 §11; R28 fila del mismo ID; NPREV §2.2 y §9.5 |
| PERF-backtesting-folds-positions | Extracción de folds del backtesting multiserie por posición: última ventana montada desde los valores de las series y series de entrenamiento con un solo `iloc` (500 series y LightGBM, 10 folds: −0,5 s de 4,2 s sin refit y −1,3 s de 25,6 s con refit, con un prototipo que incluye la exog por posición) | por decidir | M | Plan escrito y prototipado en memoria, sin implementar. PERF-multiseries-names-pyarrow, que toca la misma función, ya está fusionada (#1530). Con PERF-backtesting-ms-fast-path, F1 (ventana) y F3 (exog de test) quedan absorbidas en el backtesting multiserie: solo hacerlas antes si ese camino se retrasa (siguen sirviendo al backtesting foundation y a la validación one-step-ahead). F2 (series de entrenamiento) mantiene su valor: con refit en cada fold son 3,2 a 3,5 s de 20 a 25 s (600 series, Ridge). 2026-11-19: con el camino rápido aplazado (NPREV §10), F1 y F3 recuperan su valor: mejoran el camino actual sin un segundo camino | BTSPEED §3; NPREV §9.1 |
| PERF-backtesting-ms-fast-path | Backtesting y búsquedas de `ForecasterRecursiveMultiSeries` sin refit que validan y preparan una vez (matriz de series, exog codificada solo en las filas de test, máscaras para los avisos) y recorren los folds en numpy, sobre el núcleo de MAINT-ms-predict-core, con la preparación reutilizable entre los candidatos de una búsqueda. Datos sintéticos (600 series, exog mixta, 50 folds), llamada completa: Ridge de 10 a 12 s a 1,7 a 2,0 s (1,3 s con PERF-03); LightGBM de 12 a 14 s a 3,7 a 4,2 s; búsqueda de 20 candidatos con Ridge de 66 a 77 s a 22 a 24 s. **Casos reales (tres artículos de cienciadedatos.net): x1,0 a x1,8 con sus 9 a 31 folds, porque el `fit` domina; x2,8 a x3,3 con 61 folds (1578 edificios, `steps=1`)**; con refit en cada fold, 1,05 a 1,2 veces | por decidir | L | Prototipado en memoria (`proto_bt_once.py`, `proto_bt_blocks.py`): 24 de 25 casos idénticos y uno con el mismo error, con y sin refit, y salidas idénticas en todos los casos reales con LightGBM en un hilo, en pandas 2.3 y pandas 3 sin y con `pyarrow`. **Aplazado (2026-11-19, casos reales):** es una mejora para evaluaciones de muchos folds con modelos baratos o medios, no para los flujos de la documentación, y empeora la calidad del código (un segundo camino que reproduce los avisos de `predict`). Hacerlo cuando haya ese perfil de uso o demanda, después de MAINT-ms-predict-core y de la primera parte de PERF-03, con M1 dentro. Decisión D-NP4. Condiciones: un solo núcleo numpy, vuelta al camino actual en todo lo que D-NP4 deja fuera, y la batería de `proto_bt_once_cases.py` y los casos reales convertidos en tests. Aceptación por tiempo de la llamada completa | NPREV §2, §6, §9 y §10 |
| PERF-search-shared-train-matrices | Búsquedas sobre los parámetros del estimador: reutilizar las matrices de entrenamiento entre los candidatos que comparten `lags` y datos. Con Ridge, `_create_train_X_y` es del 62 al 74 % de cada `fit` (600 series) y, tras PERF-backtesting-ms-fast-path y PERF-03, los `fit` son el 75 % de la búsqueda | por decidir | por estimar | Estimado a partir de las piezas (la búsqueda de 20 candidatos bajaría de 22 a 24 s a unos 10 s), sin prototipo Con `OneStepAheadFold` ya existe: la búsqueda cachea la división por `lags` (2 llamadas para 10 trials en py59, NPREV §10.6). Queda para búsquedas con backtesting y modelos baratos: con LightGBM, el resto del `fit` final de py59 es 3,7 s frente a 52 s del estimador. | NPREV §9.2 |
| BUG-backtesting-refit-skipfolds | Backtesting multiserie con refit intermitente y `skip_folds`: `_extract_data_folds_multiseries` vuelve a extraer el entrenamiento cuando cambian sus límites aunque ese fold no se ajuste, y una serie puede predecirse con un modelo que no la vio (`UnknownLevelWarning`) | por decidir | S | Visto el 2026-11-19 en un caso (`refit=3`, `skip_folds=2`). Falta la reproducción mínima y decidir si es intencionado | NPREV §9.1 |
| PERF-expand-index-utc | `_is_utc_anchored_index` se ejecuta en cada `expand_index` (0,4 ms con índice diario con zona horaria): −70 % en el backtesting de una serie sin refit con ese índice (Ridge, 50 folds: de 52 a 65 ms a 15 a 20 ms). Hacerlo con la formulación exacta (aritmética entera, sin `normalize` ni `unique`); una caché por serie cambia resultados, porque el valor depende de la ventana | por decidir | S | Medido el 2026-11-19. La formulación exacta coincide con la actual en todas las ventanas probadas; su tiempo no está medido | NPREV §4 (P6) y §9.5 |
| PERF-residuals-numpy | Binning de los residuos out-of-sample en numpy (ordenación estable y claves `int` ascendentes: el `rng` se consume en ese orden); búsqueda en array de los factores conformal en lugar de `np.vectorize` (sin caché: no hay punto de invalidación); en los intervalos por bootstrapping, reutilizar las entradas y la predicción puntual. No cambiar `DataFrame.quantile` por `np.quantile`: no es idéntico en pandas 2.3 | por decidir | S | Corrección comprobada con prototipos en los tres entornos; las ganancias del estudio, sin verificar | NPREV §4 (P10, P11, P12) |
| PERF-ms-osa-metrics | Validación one-step-ahead multiserie por candidato sin `groupby` por nivel: cortes por nivel calculados una vez y arrays. No es "como el DMV": hay que conservar la inversa por nivel, los niveles ausentes y las Series que reciben las métricas propias | por decidir | S a M | Lectura de código; la ganancia del estudio (21 a 29 ms de 75 a 100 por candidato con 100 series), sin verificar. Junto con PERF-one-step-ahead-inverse Medido el 2026-11-19 con la búsqueda de py59 (NPREV §10.6): agrupar por nivel, deshacer la transformación y métricas son 0,9 s por trial con 1578 series, el 3 % de la búsqueda con LightGBM. Solo vale con modelos baratos. | NPREV §4 (P13) |
| STUDY-fit-insample-binning | Predicción sobre `X_train` en `fit` para los residuos in-sample: parte sin cambio de resultados (PERF-backtesting-skip-fit-binning) y tope de filas para los bordes de los bins | 0.28.0 | S a M | En curso: #1520 abierta. Decisión de Joaquín sobre el tope (10.000 filas una serie, 1.000 por serie en multiserie) Dato de py59 (NPREV §10.6): en el ajuste final de la búsqueda (`return_best`), la predicción in-sample cuesta de 7 a 9 s (2,5 a 4,6 % de la búsqueda). | PREP (T3) |
| SEARCH-initial-trials | `initial_trials` en las búsquedas bayesianas | 0.28.0 | S | En curso: #1492 abierta | #1492 |
| PERF-n_jobs-next | Siguientes pasos de `n_jobs` tras #1503: (1) arrays grandes sin copia (pickle 5 con `buffer_callback`; KNN, SVR y GP son 2 a 3 veces más lentos con `n_jobs` entero y sin refit); (2) no contar el arranque con los workers vivos; (3) umbral de 12 s de Windows con más máquinas; (4) XGBoost con refit y 4 CPU; (5) caché del forecaster por worker (LightGBM con 2000 folds: 14 de 15 s deserializando); (6) ¿sobra `mmap_mode='c'` en `backtesting_stats`?; (7) reinicio de loky por el tope de folds; (8) backend de hilos y clases de `__main__`; (9) avisos perdidos cuando falla un fold (`return_as="generator"`); (10) umbral en macOS sin medir; (11) propuesta 3 completa (más estimadores que HGB y CatBoost, multiserie sin refit); (12) XGBoost con un hilo al predecir pocas filas (`extra/bench_xgb_predict_threads.py`) | posterior | M a L (por partes) | No empezados | NJOBS "Siguientes pasos", "Límites", propuestas 3 y 6 |
| NJOBS-cpu-count | `n_jobs=-1` usa `cpu_count() - 1`: mantenerlo y documentar el margen, o pasar a `cpu_count()` | por decidir | XS | Decisión pendiente D-NJOBS | NJOBS propuesta 4 |
| DOC-njobs-faq | FAQ de paralelización: LightGBM con `n_jobs=1` y folds en paralelo a partir de 8 núcleos; HGB predice a 7 ms por paso | por decidir | XS | La FAQ ya recomienda un solo nivel de paralelismo; quizá baste | NJOBS propuesta 5 |
| BUG-05 | grid/random con todas las combinaciones fallidas devuelven un DataFrame vacío y el forecaster sin ajustar. Pulido barato: `ValueError(...) from first_exc`, como `bayesian_search` | posterior | S | Comportamiento deliberado y testeado desde la 0.21; requiere acuerdo de mantenedores | R27 §5; R27-repro/BUG-05.py |
| SEARCH-validate-cv-once | Validar el cv una vez antes del bucle de grid | posterior | S | Junto con BUG-05 | R27 §5 |
| FE-01 | Función pública de métricas de backtesting por fold, paso, serie y grupos | posterior | M | Veredicto partido; decidir según demanda (#720). Requiere BUG-02 | R27 §5; R27-verify/FE-01-*.md |
| METRIC-wape | WAPE como métrica (de QW-08) | posterior | S | sMAPE ya cubre las series con ceros | R27 §5 |
| SEARCH-equivalent-date-ms | `grid_search_equivalent_date` multiserie | posterior | S | Después de FE-04 | R27 §5 |
| ISSUE-995 | Pruning de Optuna en la búsqueda bayesiana | posterior | M | Sin verificar; pregunta abierta desde 2025-07 | [#995](https://github.com/skforecast/skforecast/issues/995) |
| SHORT-c2-cold-start | Predecir series con menos historia que la ventana (cold start), solo como argumento explícito del usuario | posterior | M | Aparcada | SHORT |

## 5. Forecasters y multiserie

| ID | Qué | Versión | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| BUG-ms-exog-rangeindex | `ForecasterRecursiveMultiSeries` con series de RangeIndex: una exog de `predict` con el índice reseteado (empieza en 0) se acepta sin aviso y usa los valores de entrenamiento. Ejemplo: predice `[-2.83, -0.72, -0.79]` frente a los correctos `[-1.77, -0.39, -0.99]`. Tratarlo con QW-06 si el diseño lo permite. Comprobar si lo cambia la rama de MSPERF, que toca la alineación de la exog en `_create_predict_inputs` | 0.28.0 (propuesta) | S | Reproducido el 2026-11-19: resultado erróneo silencioso | R27 §5; R27-verify/QW-06-qw.md |
| QW-01 | Aviso cuando `predict()` con `levels=None` deja fuera series que no llegan a la última fecha | 0.28.0 | S | Reproducido (una serie excluida sin aviso) | R27 §2; R27-verify/QW-01-qw.md |
| QW-06 | Alinear `exog` por índice en `predict` de Recursive y Direct | 0.28.0 | S | Reproducido (`ValueError`). Junto con QW-14 (mismo camino) | R27 §2; R27-verify/QW-06-qw.md |
| QW-14 | Un solo aviso de `exog` con NaN en `predict` de Recursive y Direct (hoy dos, uno habla de "Fitting") | 0.28.0 | S | Reproducido | R27 §2; script en `d24e5b852:dev/roadmap_0.27.0/repro/PR-2c_joblib_warnings.py` |
| QW-15 | Avisos del `fit` en paralelo de Direct y DMV (no usan `_reemit_worker_warnings`) | 0.28.0 | S | Sin medir | R27 §2; TRANSP §15.7 |
| BUG-07 | El formato largo con MultiIndex solo funciona si sale de `reshape_series_wide_to_long` (de paso, la errata "If it a") | 0.28.0 | S | Reproducido | R27 §4; R27-repro/BUG-07.py |
| BUG-classifier-nan-last-window | Clasificador sin soporte de NaN y NaN en la última ventana: error genérico de scikit-learn | 0.28.0 | S | Reproducido. Prioridad baja | R27 §4 |
| BUG-rollingfeatures-nan | `RollingFeatures.transform` ignora los NaN y `min_periods` en predicción; `transform_batch` los respeta. Hacerlo coherente mejora un 2 % en el experimento 5 | 0.28.0 | S | No empezado | TRANSP §7 (experimento 5) |
| PERF-reshape-long-to-dict | Usar `_split_long_format` (#1512) en la rama MultiIndex de `reshape_series_long_to_dict` y `reshape_exog_long_to_dict` | 0.28.0 | S | Candidata: medir antes con pandas 2, pandas 3 sin y con `pyarrow` | R28 fila del mismo ID |
| ISSUE-1506 | Con un estimador que modifica su entrada en `fit` (`StandardScaler(copy=False)`), `ForecasterDirect` y `ForecasterDirectMultiVariate` predicen mal y `ForecasterRecursive` da residuos in-sample erróneos, sin aviso. MultiSeries ya lo detecta (`_check_in_place_fit`) | 0.28.0 | S a M | Abierto | [#1506](https://github.com/skforecast/skforecast/issues/1506) |
| ISSUE-1508 | `transformer_exog` que devuelve numpy con una columna de texto: todas las exógenas pasan a `object` y `categorical_features='auto'` codifica también las numéricas, sin aviso | 0.28.0 | S | Abierto | [#1508](https://github.com/skforecast/skforecast/issues/1508) |
| BUG-ms-fit-dataframe-categorical | `ForecasterRecursiveMultiSeries` da un DataFrame al estimador en `fit` y numpy en `predict`. Con `categorical_features=None` (o una lista que omite una columna) y una exog `category` cuyos valores no coinciden con sus códigos (10, 20, 30; `0..n-1` declaradas en otro orden), LightGBM y XGBoost (con xgboost 3.4.1, también con sus parámetros por defecto) entrenan con los códigos y predicen con los valores: predicciones erróneas sin aviso. La validación one-step-ahead no lo ve: MAE 3,9, frente a 31,1 de un backtesting equivalente (`steps=1`, sin refit). También falla la vía documentada para `None`, configurar las categóricas en el estimador por índice (LightGBM 2,98 frente a 34,7; XGBoost con `feature_types`; CatBoost con `cat_features`). Con categorías de texto o decimales, `None` da un `TypeError` claro en `fit`; con una lista incompleta, el texto falla en `predict` y los decimales dan el mismo error silencioso. Con el valor por defecto `'auto'` es correcto, y HistGradientBoosting no está afectado. **Arreglo (D-NP2 = (a'), decidida el 2026-11-20):** el estimador recibe en `fit` los mismos valores que en `predict`, y una `category` que `categorical_features` no gestiona se trata como numérica, como en los demás forecasters. Incluye: (1) numpy al estimador en `fit`, en la predicción in-sample, en `set_in_sample_residuals` y en la validación one-step-ahead (aquí, convertir una vez en la división, no por candidato); (2) aviso `DataTypeWarning` en `fit` cuando una `category` queda sin gestionar, que recomiende `'auto'`, en todos los forecasters (los de una serie ya la tratan como numérica, sin avisar); (3) traducir a índices las categóricas que el usuario configura por nombre (`categorical_feature` y `cat_features` en `fit_kwargs`, `categorical_features` de HistGradientBoosting): hoy funcionan en multiserie y las cubren 6 tests; (4) con `encoding='ordinal_category'` y `None`, añadir el índice de `_level_skforecast` a la configuración del usuario, sin sustituirla; (5) CatBoost con categóricas: mantener el DataFrame con los valores (no los códigos) en lugar del array `object` de los forecasters de una serie, que cuesta de un 11 a un 22 % más de `fit` y 260 MB más de pico (300.000 filas por 28 columnas); (6) pasar el bloque numpy con el que ya se construye `X_train`, no una vista de pandas: en pandas 3 es de solo lectura, cambia la comprobación de modificación in situ y Ridge hace una copia más; (7) error claro para una `category` de texto fuera de la lista, también en los forecasters de una serie (hoy, `could not convert string to float`); (8) nota de versión en "Before upgrading": quien use `None` con una `category` recupera el modelo de hoy con `'auto'`, y los forecasters guardados con el bug hay que reajustarlos; actualizar el docstring, la guía `categorical-features.ipynb`, `llms-base.txt` y el mensaje de `check_exog_dtypes` | por decidir | M | Reproducido el 2026-11-19 (pandas 2.3 y 3, también en `0.28.x`). **Opción (a') revisada con un prototipo el 2026-11-20** (`review_scripts/dnp2_proto.patch`, pandas 2.3, 3 sin y con `pyarrow`): con `'auto'`, idéntico al código actual en 112 de 112 configuraciones; one-step-ahead y backtesting coinciden en las 264 que se ejecutan (hoy difieren en 48); el modelo que hoy se ajusta con `None` es idéntico al de `'auto'` en 36 de 36, así que la migración es cambiar ese argumento; `fit` sin coste con LightGBM, XGBoost, HistGradientBoosting y Ridge (caso real de 1.035.168 filas por 40 columnas: de 12,8 a 13,1 s hoy, de 12,2 a 13,1 s con el prototipo). Con el prototipo fallan los tests que fijan el comportamiento que cambia: 2 de la receta antigua (`OrdinalEncoder` más `category` con `None`), que pasan a `'auto'`, y los de modificación in situ (punto 6). Requisitos: CAT-T2 (un LightGBM dentro de `TransformedTargetRegressor` con `ordinal_category` pierde el nivel categórico: MAE 1,78 a 4,23) y CAT-T3 (con `Pipeline`, el índice del nivel provoca su fallo en un caso que hoy funciona). Después de PERF-exog-float-matrix (conversión común de la exog) y, si se hace, de la segunda parte de STUDY-fit-insample-binning (el tope de filas cambia la predicción in-sample de `fit`; la PR #1520 abierta solo cambia `_validation.py` y no interfiere, comprobado el 2026-11-20). El estimador pierde `feature_names_in_` y los nombres propios de LightGBM, XGBoost y CatBoost (`get_feature_importances` no cambia) | NPREV §3.1, §9.4 y §11 |
| QW-ms-category-warning | Paso opcional y provisional de BUG-ms-fit-dataframe-categorical, por si el arreglo tarda: solo el aviso. En el `fit` de `ForecasterRecursiveMultiSeries`, `DataTypeWarning` cuando una columna `category` de la exog queda fuera de `categorical_features` (`None` o lista incompleta): "sus predicciones pueden ser erróneas con LightGBM, XGBoost o CatBoost; usa `categorical_features='auto'`". No cambia ningún resultado y el consejo es el mismo que dará el arreglo, que después solo cambia el texto. Unas 10 líneas en `_create_train_X_y`, más tests | por decidir | XS a S | Propuesto el 2026-11-20, sin implementar. Sobra si el arreglo entra en la misma versión | NPREV §11.9 |
| PERF-exog-float-matrix | Exog `bool`, nullable o numérica de `pyarrow` mezclada con otros dtypes: `exog.to_numpy()` da un array `object` en Recursive, Direct, DMV, Classifier y en el `predict` de multiserie. `fit` de 5 a 6 veces más lento con Ridge (100.000 filas) y, con HistGradientBoosting, `categorical_features='auto'`, una categórica y una `bool`, `predict` falla (`TypeError: ufunc 'isnan' not supported`). Convertir a float solo cuando hoy saldría `object` y con dtypes explícitos (ni `complex` ni `decimal`) | por decidir | S | Medido, y fallo reproducido, el 2026-11-19 en los tres entornos; con el prototipo pasan las cuatro carpetas de tests de esos forecasters. No hay tests de estos dtypes: escribirlos. D-NP3 cerrada el 2026-11-20: todo a float. Los NA de columnas nullable pasan a NaN y decide el estimador, y los enteros mayores que 2^53, que hoy salen exactos de `create_train_X_y`, se convierten también (se redondean: documentarlo). Al implementar: `UInt64` cerca de 2^64 mezclado con float da `TypeError: cannot safely cast` con el prototipo, y `ForecasterRnn` no se pudo ejecutar (sin torch) | NPREV §3.2 |
| BUG-ms-exog-none-bool | Multiserie: si una serie no tiene exog, las columnas `int` pasan a `float64` y una `bool` a `object`; con `categorical_features='auto'` la `bool` se codifica entonces como categórica (con todas las series presentes, no), y con `None` LightGBM falla en `fit` | por decidir | S | Reproducido el 2026-11-19. Relacionado con ISSUE-1509 | NPREV §5 (I8) y §9.6 |
| PERF-dmv-last-window | `ForecasterDirectMultiVariate.predict`: ventana con un único `to_numpy` tras seleccionar las columnas y con copia explícita (en pandas 3 sale una vista de solo lectura del `last_window` del usuario); no llamar a `drop(columns=[])` en cada fold (solo lo alcanza el DMV) | por decidir | S | Valores idénticos comprobados; la ganancia del estudio (40 % del `predict`), sin verificar | NPREV §4 (P16, P17) |
| PERF-ms-fit-small | `fit` multiserie: `training_range_` con `v[np.array([0, -1])]` (de 40 a 55 µs a 8 a 10 µs por serie, índice idéntico); exog ancha cortada una vez (solo acierta si todas las series tienen el índice de la exog); dtypes de la exog en diccionario comparados por columna, sin construir un DataFrame | por decidir | S | La primera, medida; las otras dos, comprobadas en corrección | NPREV §4 (P7, P8, P9) |
| BUG-equivalent-date-nan | `ForecasterEquivalentDate` con NaN en `y`: con `offset` entero el NaN se propaga; con `DateOffset` depende de la función (`np.mean` y `np.max` lo saltan, `np.median` no, una función propia da otro valor), y los residuos in-sample y `predict` del mismo forecaster no coinciden. Decidir antes de vectorizar; `agg_func(axis=...)` solo es idéntico hasta 7 offsets | por decidir | S a M | Reproducido el 2026-11-19. Decisión pendiente D-NP6 | NPREV §4 (P14) y §5 (I3) |
| PERF-classifier-fast-predict | `ForecasterRecursiveClassifier`: `predict_proba` del estimador en cada paso sin camino rápido, y `predict` sin `check_inputs`. Cada rama de clasificador es código nuevo. Unificar `(a == b).all()` a `.equals` cambia un caso en pandas 2.3 (`datetime64[s]` frente a `[ns]`) | por decidir | M | Lectura de código; ganancia sin medir | NPREV §4 (P15) y §5 (I4, I5) |
| MAINT-ms-predict-core | Partir el `predict` de `ForecasterRecursiveMultiSeries` en validar, preparar y un núcleo numpy, sin cambio de comportamiento (antes, paso 1 de PERF-backtesting-ms-fast-path). Por calidad, no por tiempo: `_create_predict_inputs` tiene unas 420 líneas, mezcla la validación, la alineación de la exog y las matrices, y la usan cuatro métodos; `_recursive_predict` pide un DataFrame del que solo usa los valores; los atajos de MSPERF (comparaciones `identical`, cachés por `id`) viven dentro porque hoy se valida y alinea en cada llamada | por decidir | M | Lectura de código, sin prototipo. Después de PERF-exog-float-matrix y BUG-ms-fit-dataframe-categorical, para que el núcleo herede una conversión de la exog común y correcta. Esbozar antes las firmas de las tres etapas para comprobar que simplifica y no solo reparte la complejidad. Red de seguridad: las baterías de identidad de MSPERF | NPREV §8 y §10.5 |
| CAT-T2 (antes HANDOFF-T2) | Las categóricas nativas no llegan al estimador dentro de `CalibratedClassifierCV` ni de `TransformedTargetRegressor` (`configure_estimator_categorical_features` solo desenvuelve `Pipeline`). Incluir `estimator_has_native_nan_support` con TTR | 0.28.0 | S | Reproducido el 2026-11-19. Requisito de BUG-ms-fit-dataframe-categorical: hoy, en multiserie, un LightGBM envuelto recibe las categóricas por dtype gracias al DataFrame, y con numpy en `fit` las pierde | UTILS §4 T2; revisión §5.7 |
| CAT-T3 (antes HANDOFF-T3) | `Pipeline` con LightGBM o CatBoost y exog categórica: `fit` falla (`Pipeline.fit does not accept the categorical_feature parameter`) | 0.28.0 | S | Reproducido. D-T3 cerrada el 2026-11-20: (a), con un `Pipeline` no se configuran las categóricas nativas y las columnas pasan con su codificación ordinal, como con cualquier otro estimador. Los `Pipeline` con XGBoost o HistGradientBoosting pierden la configuración nativa que hoy reciben: nota en Changed. Al implementar, un test en multiserie con un `Pipeline` que conserva el DataFrame (`set_output(transform="pandas")`), que hoy puede recibir las categóricas por dtype (sin medir). Requisito de BUG-ms-fit-dataframe-categorical. Visto otra vez en NPREV §3.1 y §11.7 | UTILS §4 T3 |
| CAT-T4 (antes HANDOFF-T4) | `categorical_features='auto'` sin categóricas resetea la configuración del usuario y da un aviso falso en cada refit; la guía `categorical-features.ipynb` dice "will have no effect" | 0.28.0 | S | Reproducido. D-T4 cerrada el 2026-11-20: (a), avisar una vez en `__init__` cuando el usuario trae su propia configuración categórica y `categorical_features` no es `None`, y quitar los avisos de `fit`; corregir la nota de la guía. Después de BUG-ms-fit-dataframe-categorical: el aviso recomienda `categorical_features=None` para conservar la configuración propia, y en multiserie eso hoy da predicciones erróneas (NPREV §11.2). Visto otra vez en NPREV §9.4 | UTILS §4 T4 |
| BUG-ms-missing-categorical-exog (antes N-03) | Multiserie: si falta una columna categórica en la exog de `predict` (ancha o dict), `TypeError: ufunc 'isnan' not supported` del `OrdinalEncoder`; con una numérica el error es claro | por decidir | S | Reproducido el 2026-11-19 (N-03) | Revisión de utils §5.5 (`2377adb24`) |
| BUG-xgbclassifier-lag-only-class | `XGBClassifier` falla en `fit` si una clase solo está en los lags y no es el último código (#1514 arregló el resto del caso) | por decidir | M | Sin issue. Arreglo: renumerar el objetivo | NJOBS "Fallos previos", punto 2 |
| BUG-pandas2-shared-memory | Tres bugs previos que solo pasan con pandas 2: alias entre `X_train` e `y_train` (también los `y_train` de cada paso de `ForecasterDirect`); `X_train` comparte memoria con la salida de una window feature propia; `filter_train_X_y_for_step` sin exógenas cambia el índice del `X_train` del usuario | por decidir | S | Sin issue | P3 §10 "PR 3a", Pendiente, punto 4 |
| PERF-categorical-codes | Codificar las exógenas `category` por sus códigos en lugar de `OrdinalEncoder` (x1,8 a x1,9 con pandas 3 y `pyarrow`) | 0.28.0 | M | No empezado. Revisado el 2026-11-19: `get_indexer` difiere del `OrdinalEncoder` cuando `fit` vio un valor faltante (`categories_` acaba en NaN; incluye una columna de texto con `None` en pandas 3) y usar `.cat.codes` directamente es incorrecto con otro orden de categorías. En multiserie, una vez por tabla y no serie a serie. Pierde valor con PERF-backtesting-ms-fast-path (el codificador deja de ejecutarse en cada fold): le quedan el `predict` de una serie con categóricas y el `fit` | PREP (T2); NPREV §4 (P4) |
| PERF-one-step-ahead-inverse | Inversa del escalador con `transform_numpy` en la validación one-step-ahead multiserie | posterior | S | Aplazada en #1520 (`transformer_series` en multiserie es poco habitual) | PREP (T4) |
| PERF-scalers-extension | Atajo de `transform_numpy` para `MinMaxScaler` y `RobustScaler` (V3) | por decidir | S | Ampliación de #1519. Pierde valor en el backtesting multiserie con PERF-backtesting-ms-fast-path (el escalador se aplica una vez por serie, no por fold) | PREP (T1, V3) |
| TODO-multivariate-gap | Formato largo con varios niveles y `gap>0` en multivariante (`_validation.py`, buscar el TODO) | posterior | S | Sin verificar: comprobar antes si es un caso real | R27 §5 |
| REVIEW-dmv-categorical | Revisar `ForecasterDirectMultiVariate` (lags dict, intervalos, NaN al inicio, `differentiation`) y las exógenas categóricas, que el análisis de la 0.27.0 no cubrió | posterior | M | | R27 §0 "Huecos del análisis", §7 "Cobertura" |

## 6. Preprocessing

| ID | Qué | Versión | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| QW-11 | `hour_of_week` y días de fin de semana del mes en `CalendarFeatures`, aviso con features anuales y menos de un año de datos, guía de codificación | 0.28.0 | S | Medido (`hour_of_week`: -24 % de MAE con Ridge en horarias). Decisión pendiente D-QW11 | R27 §2 ("QW-11: análisis"); R27-repro/QW-11-calendar-analysis.py |
| FE-06 | Festivos en `CalendarFeatures` con `holidays` como dependencia opcional | 0.28.0 | M | Decisión pendiente D-FE06 | R27 §3 |
| FE-03 | Ventanas estacionales en `RollingFeatures` (`season_length`) | 0.28.0 | M | Mejora el MAE un 2 % y solo con árboles | R27 §3; R27-verify/FE-03-*.md |
| ISSUE-1507 | `calculate_distance_from_holiday` da horas en lugar de días con `SME`, `SMS`, `BQE`, `BQS`, `BYE` y `BYS` (faltan en `_freq_to_timedelta_unit`) | 0.28.0 | XS | Abierto | [#1507](https://github.com/skforecast/skforecast/issues/1507) |
| ISSUE-1509 | `reshape_exog_long_to_dict` deja una columna booleana como `object` en unas series y `bool` en otras, y MultiSeries rechaza el diccionario | 0.28.0 | S | Abierto | [#1509](https://github.com/skforecast/skforecast/issues/1509) |
| DISC-1153 | Asimetría en `RollingFeatures` y errores difíciles de diagnosticar en window features propias | posterior | S a M | Sin verificar. FE-03 reduce parte de la necesidad | [Discusión 1153](https://github.com/skforecast/skforecast/discussions/1153) |
| TODO-rolling-classification | TODO de `RollingFeaturesClassification` en `skforecast/preprocessing/_preprocessing.py` | posterior | XS | Menor | R27 `work/phase1_product_context.md` (`402be62d6`) |

## 7. Stats

| ID | Qué | Versión | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| BUG-stats-last-window (antes HANDOFF-STATS-LW) | `ForecasterStats.fit(store_last_window=False)` y después `predict(last_window=...)`: `AttributeError: 'NoneType' object has no attribute 'name'` (el `rename` que añadió T6). Pasa con Arima y Sarimax | por decidir | S | Reproducido el 2026-11-19 | UTILS §5; R27 §5 (HANDOFF-STATS-LW) |
| PERF-02 | Ajuste de Ets y Sarimax de 3 a 70 veces más lento con carga en la máquina | 0.28.0 | S | Volver a medir: BUG-08 (#1425) cambió el escenario | R27 §4; R27-repro/PERF-02.py |
| DEP-arima-dictstyle | `FutureWarning` sin versión desde la 0.22 (`_arima_base.py`) | 0.28.0 | S | Decisión pendiente D-ARIMA | R27 §6 |
| PERF-arima-seasonal-kalman | Kalman de Arima estacional más rápido (14 ms por evaluación, 729 evaluaciones con n=5000 y m=24) | posterior | L | | R27 §5; `402be62d6:dev/roadmap_0.27.0/work/lens_stats.md` |

## 8. Utilidades, documentación y deep learning

| ID | Qué | Versión | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| QW-07 | Mensajes de error accionables en cuatro puntos | 0.28.0 | S | (a) y (b) reproducidos con los mensajes antiguos. El prototipo no se guardó | R27 §2; R27-verify/QW-07-qw.md |
| DOC-skills-percentiles-stale | Skills que aún hablan de percentiles (dan `ValueError` desde la 0.25; `skills/prediction-intervals/SKILL.md` dice `FutureWarning`) y de que EquivalentDate no tiene `set_params` (sí lo tiene) | 0.28.0 | S | Junto a QW-07 (b); regenerar el contexto IA | R27 §6 |
| DOC-rnn-exog-example | El ejemplo con exógenas de `forecasting-with-deep-learning-rnn-lstm.ipynb` no es reproducible entre versiones de keras y torch (`EarlyStopping` corta en la época 12 en lugar de la 37; MAE de 47,75 a 76,46). Hacerlo robusto y reejecutar | 0.28.0 | S | Prioridad baja | R28 fila del mismo ID |
| DOC-warning-panel-paths | Los paneles de avisos publican en la documentación la ruta absoluta de la máquina que ejecutó los notebooks (unas 20 guías). Que el ejecutor de notebooks muestre la ruta relativa al repositorio | 0.28.0 | S | No empezado | R28 fila del mismo ID |
| DOC-crps-guide-arviz | `faq/probabilistic-forecasting-crps-score` falla al importar (`arviz-plots` 1.0.0 usa `matplotlib.style.core`, ausente en matplotlib 3.11). Esperar a `arviz-plots` o calcular el CRPS sin `pymc_marketing` | 0.28.0 | S | Las salidas guardadas siguen limpias | R28 fila del mismo ID |
| DOC-lightgbm-reproducibility | Con `n_jobs=None` (su valor por defecto), LightGBM 4.7 entrena con todos los núcleos físicos, ignora `OMP_NUM_THREADS` y sus resultados cambian de una ejecución a otra: dos backtestings idénticos del código actual difieren; con 600 árboles y submuestreo el MAE de algunas series cambia hasta 0,4 entre dos ejecuciones, y hasta un 8 % frente a los valores publicados en el artículo. Nota en la documentación (FAQ de reproducibilidad o guías con LightGBM): fijar `n_jobs` (comprobado con `n_jobs=1`) o `deterministic=True` (documentado por LightGBM, sin comprobar aquí). En los tests, 211 creaciones de `LGBMRegressor` no fijan `n_jobs`; con sus datos pequeños no se han visto fallos, pero es lo primero que mirar ante uno intermitente | por decidir | XS | Visto el 2026-11-19 con el artículo py59 | NPREV §10.2 |
| ISSUE-1131 | Exógenas as-of. FE-02 cubre la parte foundation; #1476 (guía de exógenas pronosticadas, Refs #1131) está abierta | 0.28.0 | L | Decisión pendiente D-1131 (alcance del resto) | [#1131](https://github.com/skforecast/skforecast/issues/1131); #1476 |
| MAINT-save-versions | Guardar en `save_forecaster` las versiones de pandas, numpy, scikit-learn y Python y avisar en `load_forecaster` solo si cambia una versión mayor | por decidir | S | Propuesta. Sin atributos nuevos obligatorios | P3 §12 |
| UTILS-save-path (antes HANDOFF-SAVE-PATH) | `save_forecaster` devuelve la ruta final del fichero (hoy `None`) | por decidir | XS | Propuesta | UTILS §5; revisión de utils §5.3 |
| UTILS-equivalent-date-last-window | `ForecasterEquivalentDate` guarda la serie entera en `last_window_` (offset mayor que los datos) | por decidir | S | Dudoso: probablemente descartar | Revisión de utils §5.8 (`2377adb24`) |
| BR-synthetic-data | Generador sintético de paneles (rama `feature_synthetic_data_generator` en origin, muy por detrás): rebase y decidir si va en `experimental` o `datasets` | posterior | M | Sin verificar | R27 §5 |
| ISSUE-974 | skforecast como motor de reducción de sktime | posterior | L | Sin verificar; estratégico, sin actividad | [#974](https://github.com/skforecast/skforecast/issues/974) |

## 9. Backlog sin versión (aplazados en el triaje de la 0.27.0)

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
| D-FE06 | FE-06 | Si entra en la 0.28.0: es lo que más mejora las series diarias medidas, pero solo hay 2 series con festivos y el alcance es M |
| D-ARIMA | DEP-arima-dictstyle | Retirar el `FutureWarning` o fijar la versión en que se cumple |
| D-1131 | ISSUE-1131 | Alcance de lo que no cubren FE-02 y #1476 |
| D-NJOBS | NJOBS-cpu-count | `cpu_count()` frente a `cpu_count() - 1` |
| D-TIMEH2O | FOUND-timeh2o | D1 (licencia no comercial y no productiva) y D2 a D15 de la especificación (campo "Decisión" vacío) |
| D-FAR | FOUND-plan-02 a 09 | Unas 27 decisiones de la sección 4 de `foundation_adapters_review/README.md` (D1 a D3 ya se tomaron en #1478, opción a) |
| D-NP4 | PERF-backtesting-ms-fast-path | Alcance de la primera versión. Propuesta: sin refit, `interval=None`, sin `return_predictors`, `n_jobs=1`, y `transformer_exog=None` o una lista blanca de transformadores fila a fila; el resto, por el camino actual |
| D-NP6 | BUG-equivalent-date-nan | NaN en `y`: propagarlo en los dos caminos (`offset` entero y `DateOffset`) o saltarlo en los dos |
| D-ORDEN | Todo | Orden de la sección 11 |

## 11. Orden propuesto (por confirmar)

Con las prioridades de la 0.27.0 (machine learning y foundation primero) y los resultados
erróneos silenciosos delante:

1. Lo que está en curso: DOC-pandas3-report;
   STUDY-fit-insample-binning (#1520), SEARCH-initial-trials (#1492), #1476.
2. Resultados erróneos silenciosos: BUG-ms-exog-rangeindex (con QW-06 y QW-14), BUG-04 y
   FOUND-adapter-arg, ISSUE-1506, ISSUE-1508, BUG-ms-fit-dataframe-categorical (con CAT-T2 y
   CAT-T3 antes, y CAT-T4 después)
   y PERF-exog-float-matrix (arregla un fallo de `predict` con la configuración por defecto).
3. Planes 02 y 03 de foundation, y BUG-06 (requisito de FE-05).
4. FE-05, después BUG-02. PERF-03, en el orden de rendimiento de abajo.
5. QW-01 y BUG-01 (avisos).
6. QW-11 y FE-06.
7. Dependencias (sección 2): independientes, en cualquier momento.
8. Publicación (sección 1): MAINT-bench-pandas3, MAINT-docs-pandas3, REL-conda-forge-028.
9. El resto.

### Rendimiento y numpy interno: orden por lo que aporta

Resultado de `dev/pandas_numpy_review/` (2026-11-19), sin versión asignada. La ganancia grande
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

0. **Decisiones, ya (sin código):** D-BUG02 (D-NP2 y D-NP3 se cerraron el 2026-11-20), qué significa saltar un
   fold con refit (BUG-backtesting-refit-skipfolds) y si las métricas en numpy pueden diferir
   en el último bit de las de scikit-learn (PERF-03: 4e-16 en un caso).
1. **Corrección.**
   1. Sobre el `_create_predict_inputs` de multiserie que dejó #1530 (PERF-multiseries-names-pyarrow),
      donde están la conversión de la exog de `predict` (PERF-exog-float-matrix) y su
      alineación (BUG-ms-exog-rangeindex).
      PERF-exog-float-matrix y BUG-ms-exog-none-bool, seguidas: las dos tratan los dtypes de
      la exog al pasar a numpy, y comparten la matriz de tests de dtypes (`bool`, nullable,
      `pyarrow`) que hoy no existe y que también necesita el siguiente. La primera arregla un
      fallo de `predict` con la configuración por defecto.
   2. BUG-ms-fit-dataframe-categorical (M, con CAT-T2 y CAT-T3 antes y CAT-T4 después):
      predicciones erróneas sin aviso, pero solo con
      `categorical_features=None` o una lista incompleta. Si se retrasa, QW-ms-category-warning
      (solo el aviso) protege mientras tanto. La PR #1520 no interfiere (solo cambia
      `_validation.py`); el tope de filas de STUDY-fit-insample-binning, si se hace, sí cambia
      la predicción in-sample de `fit`: uno detrás del otro.
   3. BUG-backtesting-refit-skipfolds: configuración rara (refit intermitente con
      `skip_folds`).
2. **Calidad del código de multiserie:** MAINT-ms-predict-core (M), partir `predict` en
   validar, preparar y núcleo numpy. Después de 1.1 y 1.2, para que el núcleo herede una
   conversión de la exog común y correcta.
3. **Baratas e independientes, en cualquier hueco, por este orden:** PERF-expand-index-utc
   (−70 % en el backtesting de una serie con índice diario y zona horaria: la única ganancia
   grande para los forecasters de una serie), PERF-backtesting-folds-positions completa (F1, F2
   y F3: mejoran el camino actual sin un segundo camino; F2 es la que ayuda con refit),
   PERF-ms-fit-small, PERF-residuals-numpy, PERF-dmv-last-window (medir antes: su ganancia no
   está verificada), DOC-lightgbm-reproducibility y la primera parte de PERF-03.
4. **Rendimiento solo con muchos folds o modelos baratos, cuando haya demanda:**
   PERF-backtesting-ms-fast-path (L, con M1 dentro y después de 2; si D-BUG02 está decidida,
   BUG-02 antes de M1), PERF-search-shared-train-matrices (búsquedas con backtesting) y
   PERF-ms-osa-metrics. Con LightGBM y pocos folds, que es el uso de los artículos, todo
   esto junto es menos del 10 % (NPREV §10.6).
5. **Poco valor, no ahora:** PERF-categorical-codes, PERF-scalers-extension, el bloque B de
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
- PERF-long-to-dict: #1512.
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
- STUDY-osa-search-cost: #1524. La búsqueda de py59 con `OneStepAheadFold` es del 91 al 98 %
  LightGBM; lo que no es modelo daría como mucho x1,10 (NPREV §10.6).

---

## Anexo A: BUG-04 y modelos afinados

Copiado de la sección 3.1 de R28 (2026-11-11), que se borrará con `dev/roadmap_0.28.0/`.

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
