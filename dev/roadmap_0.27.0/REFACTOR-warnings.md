# REFACTOR-warnings: organización y nombres del código de los avisos

Fecha: 2026-11-08. Rama: `fix/joblib-worker-warnings` (PR #1467 contra `0.27.x`).
Estado: **aplicado** (2026-11-08), en local y sin push, con las decisiones D1 a D6 de la
sección 7 aprobadas (D7 a D12 quedan fuera):

| Commit | Contenido |
|:--|:--|
| 5346b78e1 | 1. Mover sin renombrar a `skforecast/utils/_warnings.py`, con sus tests |
| 70f5a37ff | 2. Renombrar `fold` → `worker` (D5), con "records" en el docstring de `_SearchWarnings` (nota 5 de la sección 5) |
| 4516a4b0f | 3. Renombrar `_stacklevel_user_code` → `_find_stack_level` (D6) |
| e34e62e60 | Tras la revisión: `WORKER_WARNINGS_ATTRIBUTE` → `_WORKER_WARNINGS_ATTRIBUTE` (sin `_`, la reexportación de `skforecast/utils/__init__.py` lo hacía parecer API pública; la propuesta de la sección 4.4 lo omitía) y alineación de `function` en las tres llamadas de `_validation.py` |

Superficie de `skforecast.utils`: además de `P`, `R`, `wraps`, `ParamSpec` y `TypeVar`,
deja de existir `skforecast.utils.warn_skforecast_categories` (también
`skforecast.utils.utils.warn_skforecast_categories`). Estaba en `v0.26.0` solo por el
`import *` de un `utils.py` sin `__all__`, no está documentada y nada del repositorio la
usaba; la ruta documentada es `skforecast.exceptions`. Se acepta sin deprecación. El
refactor no añade ningún nombre público a `skforecast.utils`. La comprobación del contexto
de IA no cubre `utils` (no está en su lista de subpaquetes), así que no lo detectaría.

`git blame -C` con el algoritmo por defecto (Myers) no encuentra el bloque de
`manage_warnings` en las 5700 líneas de `utils/utils.py` y lo atribuye al commit 1. Con
`git -c diff.algorithm=histogram blame -C -C` (o `--minimal`) todas las líneas movidas se
atribuyen a sus commits originales; solo la cabecera y los imports son del commit 1.

Versiones del análisis: Python 3.14.3, scikit-learn 1.9.1, pandas 2.3.3, numpy 2.5.4,
statsmodels 0.14.6, joblib 1.6.0, optuna 5.0.0. Última release: `v0.26.0`.

## 0. Resumen

- **Tu hipótesis se confirma en parte.** Sí conviene crear `skforecast/utils/_warnings.py`
  con los mecanismos genéricos, pero **sin `_SearchWarnings`**. Esa clase no es genérica:
  emite el aviso de los candidatos saltados (`_warn_skipped_candidates_search`), reescribe
  `LongTrainingWarning` con el número de fits de la búsqueda y habla de candidatos y
  trials. Si estuviera en `utils`, `utils` tendría que importar de `model_selection` (un
  ciclo) o recibir los textos como parámetros. Ni hoy ni con QW-15 la usa nadie fuera de
  `model_selection`.
- **Se mueven** a `skforecast/utils/_warnings.py`:
  - los 5 helpers de los procesos de joblib del PR 2c, que QW-15 necesita desde los
    forecasters;
  - los 2 helpers que localizan el código del usuario;
  - `manage_warnings`, reexportado para que sus importaciones actuales sigan funcionando.
- **Se quedan donde están:**
  - los textos de cada aviso, junto a su código (`model_selection/_utils.py`,
    `utils/utils.py`, `datasets`, `foundation`);
  - `_SearchWarnings` y sus satélites, en `model_selection/_utils.py`;
  - las clases y `set_warnings_style`, en `skforecast.exceptions`.
- **Renombres que propongo:**
  - `fold` → `worker` en los helpers del PR 2c. Con QW-15 también los usará el `fit` de
    Direct y DMV, que no tiene folds. Sus nombres nacen en este PR, así que hoy cuesta
    poco.
  - `_stacklevel_user_code` → `_find_stack_level`. Tiene la misma semántica que el
    `find_stack_level` de pandas y sigue el estilo `_verbo_nombre`.
  - Ningún otro nombre cambia.
- **Comprobado** en una copia del paquete en el scratchpad, con el movimiento aplicado y
  los nombres sin cambiar:
  - 5 órdenes de importación sin ciclos;
  - **1666 tests pasan**: `model_selection/tests/tests_utils`, `test_manage_warnings.py`,
    búsquedas grid, random y bayesian, backtesting de una y de varias series, y
    `direct/tests`.
- **Tamaño: S.** Toca unos 8 archivos y mueve unas 280 líneas. Propongo **hacerlo en este
  PR**, en commits que solo muevan o solo renombren. No veo una razón fuerte en contra
  (sección 6).

## 1. Inventario

PRs del ciclo:

| Trabajo | PR |
|:--|:--|
| QW-13 | #1466 |
| PR 1 | #1447 |
| PR 2a | #1452 |
| PR 2b | #1456 |
| PR 2c | #1467, esta rama, sin fusionar |

Otros PRs que aparecen en las tablas: #1415 (niveles no predichos), #1422 (error de todas
las muestras eliminadas), #1423, #1440 y #1454.

La columna "en v0.26.0" se ha comprobado con `git grep` sobre el tag.

### 1.1 `skforecast/exceptions/exceptions.py` (502 líneas)

| Nombre | Líneas | Tipo | Público | En v0.26.0 | Lo usan (código) | Tests |
|:--|:--|:--|:--|:--|:--|:--|
| `runtime_deprecated` | 21-98 | decorador | sí (`exceptions/__init__`, `llms-base.txt`) | sí | ningún módulo del paquete | `tests_exceptions/test_runtime_deprecated.py` |
| 15 clases `*Warning` (`DataTypeWarning` ... `LicenseWarning`) | 102-337 | clases | sí (`__init__`, `docs/api/exceptions.md`, `llms-base.txt`) | sí, las 15 | forecasters, `model_selection`, `utils`, ... | indirectos |
| `warn_skforecast_categories` | 341-357 | lista | sí (`__init__`, `llms-base.txt`, docstrings de `_search.py`) | sí | `manage_warnings` y los forecasters | `test_format_warning_handler.py` |
| `format_warning_handler` | 364-420 | función | por nombre, no exportada en `__init__` | sí | nadie (`set_warnings_style` usa la de rich) | `test_format_warning_handler.py` |
| `rich_warning_handler` | 423-477 | función | por nombre, no exportada | sí | `set_warnings_style` | indirectos |
| `set_warnings_style` | 480-500, se llama en la 502 al importar | función | sí, documentada | sí | se ejecuta al importar el módulo y reemplaza `warnings.showwarning` | indirectos |

Todo lo de este módulo es público y está publicado: **se queda como está.** Hay tres
rarezas, pero no se tocan sin un ciclo de deprecación y quedan fuera de este trabajo:

- `warn_skforecast_categories` se llama como una función, pero es una lista.
- `format_warning_handler` no se usa.
- `runtime_deprecated` no lo usa ningún módulo del paquete.

### 1.2 `skforecast/utils/utils.py` (5712 líneas)

| Nombre | Líneas | Tipo | Público | En v0.26.0 | Origen | Lo usan (código) | Tests |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `P`, `R` | 53-54 | `ParamSpec` y `TypeVar` | no (sin uso fuera de `manage_warnings`) | sí | | `manage_warnings` | |
| `manage_warnings` | 2845-2886 | decorador | **de facto sí**: sin `_`, exportado por `from .utils import *` (no hay `__all__`). No está en `docs/api` ni en `llms-base.txt` | sí (ya en v0.22.0) | | 12 módulos: `from ..utils import manage_warnings` en recursive (5), direct (2), `deep_learning`, `drift_detection/_range_drift.py`, `model_selection/_search.py` y `_validation.py`, y `utils.py` (2 usos) | `utils/tests/tests_utils/test_manage_warnings.py` (importa `skforecast.utils.utils.manage_warnings`), `test_run_fold_recording_warnings.py` |
| `_all_samples_removed_message` | 5365 | texto de un **error** | privado | no | #1422 | 5 forecasters | `test_all_samples_removed_message.py` |
| `_warn_nan_training_rows` | 5450 | texto y emisión de un aviso | privado, reexportado | no | QW-13 (#1466) | `utils.py` | `test_warn_nan_training_rows.py` |
| `_filter_nan_train_X_y` | 5562 | filtra y avisa | privado, reexportado | no | QW-13 (#1466) | recursive (3) | `test_filter_nan_train_X_y.py` |
| `_warn_nan_training_rows_direct` | 5634 | cuenta y avisa | privado, reexportado | no | QW-13 (#1466) | direct (2) | `test_warn_nan_training_rows_direct.py` |

Las tres funciones de QW-13 son lógica del dominio (contar las filas con NaN de la matriz
de entrenamiento) que además avisa. **Se quedan en `utils.py`.**

### 1.3 `skforecast/model_selection/_utils.py` (4325 líneas)

**Textos y lógica de los avisos de las búsquedas y del backtesting**

| Nombre | Línea | Tipo | En v0.26.0 | Origen | Lo usan (código) | Tests |
|:--|:--|:--|:--|:--|:--|:--|
| `N_EVALUATED_WARNING_THRESHOLD` | 67 | constante | no | PR 1 (#1447) | `_utils.py` | |
| `LEVELS_NOT_PREDICTED_WARNING` | 71 | constante (inicio de un texto) | no | PR 2a (#1452) | `_search.py` | |
| `_warn_levels_not_predicted_multiseries` | 1448 | avisa | no | #1415 | `_validation.py` | `test_warn_levels_not_predicted_multiseries.py` |
| `_format_lags_warning` | 2923 | texto | no | PR 1 (#1447) | `_utils.py` | `test_format_lags_warning.py` |
| `_format_names_warning` | 2955 | texto | no | PR 1 (#1447) | `_utils.py` | indirectos |
| `_is_affected_n_evaluated` | 2992 | lógica del aviso | no | PR 1 | `_utils.py` | indirectos |
| `_n_evaluated_covers_levels_not_predicted` | 3016 | lógica del aviso | no | PR 2a | `_search.py` | `test_warn_n_evaluated_search.py` |
| `_warn_n_evaluated_search` | 3053 | avisa | no | PR 1 y PR 2b (#1456) | `_search.py` | `test_warn_n_evaluated_search.py` |
| `_warn_skipped_candidates_search` | 3412 | avisa | no | PR 2a (#1452) | `_SearchWarnings.emit` | `test_warn_skipped_candidates_search.py` |

**Registro de los avisos de las búsquedas**

| Nombre | Línea | Tipo | En v0.26.0 | Origen | Lo usan (código) | Tests |
|:--|:--|:--|:--|:--|:--|:--|
| `_SearchWarnings` | 3516-3862 | context manager | no | PR 2a (#1452) | `_search.py` | `test_search_warnings.py`, `test_reemit_fold_warnings.py`, `test_calculate_metrics_backtesting_multiseries.py` |
| `_TrialEvaluationError` | 3865 | error | no | PR 2b (#1456) | `_utils.py` | `test_optimize_study.py` |
| `_skip_trial` | 3874 | registra un trial saltado | no | PR 2b | `_search.py` | indirectos |
| `_FailedTrialsLogFilter` | 3906 | filtro de logging | no | PR 2b | `_utils.py` | `test_optimize_study.py` |
| `_optimize_study` | 3967 | ignora un aviso de Optuna | no | PR 2b | `_search.py` | `test_optimize_study.py` |
| `_split_first_sentence` | 4033 | texto | no | PR 2a | `_utils.py` | indirectos |
| `_warn_recorded` | 4061 | emisión | no | PR 2a | `_SearchWarnings.emit` | indirectos |

**Mecanismos genéricos (los candidatos a moverse)**

| Nombre | Línea | Tipo | En v0.26.0 | Origen | Lo usan (código) | Tests |
|:--|:--|:--|:--|:--|:--|:--|
| `_first_frame_outside_skforecast` | 4091 | **genérico** | no | PR 2a | `_SearchWarnings`, `_stacklevel_user_code` | indirectos (`test_search_warnings.py`) |
| `_stacklevel_user_code` | 4126 | **genérico** | no | PR 2a | 9 llamadas, todas en `_utils.py` | indirectos |
| `FOLD_WARNINGS_ATTRIBUTE` | 4145 | **genérico** | no | PR 2c | `_validation.py` (3) | `test_run_fold_recording_warnings.py` |
| `_run_fold_recording_warnings` | 4148 | **genérico** | no | PR 2c | `_validation.py` (3) | `test_run_fold_recording_warnings.py`, `test_reemit_fold_warnings.py` |
| `_serializable_fold_warnings` | 4211 | **genérico** | no | PR 2c | `_utils.py` | indirectos |
| `_module_name_from_filename` | 4246 | **genérico** | no | PR 2c | `_utils.py` | indirectos |
| `_reemit_fold_warnings` | 4270-4325 | **genérico** | no | PR 2c | `_validation.py` (6) | `test_reemit_fold_warnings.py` |

### 1.4 Otros avisos fuera de los tres módulos

Todos son textos junto a su código y se quedan:

| Nombre | Archivo y línea | En v0.26.0 | Origen |
|:--|:--|:--|:--|
| `_warn_deprecated_version` | `datasets/datasets.py:679` | no | #1440 |
| `TimeSeriesFold._warn_deprecated_params` | `model_selection/_split.py:162` | no | #1423 |
| `_warn_if_non_commercial` | `foundation/_utils.py:410` | sí | |
| `_warn_if_constant_known_covariates` | `foundation/_adapters.py:5826` | no | #1454 |
| `RangeDriftDetector._display_warnings` | `drift_detection/_range_drift.py:243` | sí | |

También hay usos ad hoc de `warnings.catch_warnings` o `filterwarnings` en 21 archivos del
paquete. Los que más tienen son los forecasters recursive y direct, que silencian avisos
del estimador. Son locales y no forman parte de esta propuesta.

### 1.5 Nombres públicos y API

Lo **público y publicado** es esto:

- las 15 clases, `warn_skforecast_categories`, `set_warnings_style` y `runtime_deprecated`;
- las rutas `skforecast.exceptions.exceptions.*`, por `docs/api/exceptions.md`;
- `manage_warnings`, accesible como `skforecast.utils.manage_warnings` y
  `skforecast.utils.utils.manage_warnings`.

Ninguno se renombra. `manage_warnings` mantiene sus dos rutas si se mueve.

Todo lo nuevo de este ciclo es privado (`_`). Nada aparece en `docs/`, `skills/`,
`tools/ai/` ni en las reglas o skills del harness. Solo lo citan los documentos de `dev/`
(`roadmap.md`, `QW-13.md` y `BUG-search-short-series-transparency.md`). El generador del
contexto de IA ignora los nombres privados (`generate_ai_context_files.py:425`), así que
reexportarlos no cambia `--check`.

## 2. Cómo lo hacen otras librerías (código instalado)

| Librería | Clases | Mecanismos | Nombres de los mecanismos | Módulos privados de utilidades |
|:--|:--|:--|:--|:--|
| scikit-learn 1.9.1 | `sklearn/exceptions.py`, plano, con `__all__` (`ConvergenceWarning`, `FitFailedWarning`, ...) | `sklearn/utils/parallel.py` (público): `Parallel`, `delayed`, `_FuncWrapper`, `_with_config_and_warning_filters` (l.24). `sklearn/utils/_testing.py`: `ignore_warnings` (l.73) y `_IgnoreWarnings` (l.117). `sklearn/utils/deprecation.py`: clase `deprecated`. Fallos de los folds en `model_selection/_validation.py`: `_insert_error_scores` (l.430) y `_warn_or_raise_about_fit_failures` (l.461) | Frases verbales privadas (`_warn_or_raise_about_*`, `_with_config_and_warning_filters`, `_get_*`). Sin helper de `stacklevel`: enteros fijos | `utils/_<tema>.py` con sustantivos: `_param_validation`, `_testing`, `_show_versions`, `_pprint`, `_tags`, `_set_output`, `_optional_dependencies`, ... No hay `_warnings.py` |
| pandas 2.3.3 | `pandas/errors/__init__.py`, con reexportaciones y un `__all__` | `pandas/util/_exceptions.py`: `find_stack_level()` (l.34), `rewrite_warning` (l.64; `catch_warnings(record=True)` y después `warn_explicit`) y `rewrite_exception`. `pandas/util/_decorators.py`: `deprecate*`. `pandas/_testing/_warnings.py`: `assert_produces_warning` | Sin `_` dentro de módulos privados: `find_*`, `rewrite_*`, `deprecate_*`. `find_stack_level` se importa en unos 80 módulos desde su ruta privada | `util/_decorators`, `_exceptions`, `_validators`, `_print_versions`, ... |
| numpy 2.5.4 | `numpy/exceptions.py` (nuevo en 1.25: "General exceptions used by NumPy") | `numpy/_utils/__init__.py` (`set_module`, `_rename_parameter`). `skip_file_prefixes` en línea en `lib/_twodim_base_impl.py` | `_rename_parameter` | Paquete privado `_utils/` |
| statsmodels 0.14.6 | `tools/sm_exceptions.py` (jerarquía `ModelWarning`; `simplefilter("always")` al importar) | Ninguno genérico | Ad hoc | Módulos públicos en su mayoría |
| joblib 1.6.0 | | **No reenvía los avisos de los procesos**: solo las excepciones (`_ExceptionWithTraceback`, `_rebuild_exc` en loky) | | |
| optuna 5.0.0 (dependencia de skforecast) | `optuna/exceptions.py` | **`optuna/_warnings.py`**: `optuna_warn` y `_OPTUNA_MODULE_ROOT`, con `skip_file_prefixes` en Python 3.12 o posterior; se importa en 48 módulos | `<paquete>_warn` | `_deprecated.py`, `_experimental.py`, `_imports.py`, `_warnings.py` |

Conclusiones aplicables a skforecast:

1. **Clases:** un `exceptions.py` público y plano (sklearn, numpy, optuna), como
   `skforecast.exceptions`. No hay nada que cambiar.
2. **Mecanismos:** van en módulos privados del paquete de utilidades: `pandas/util/_exceptions.py`,
   `sklearn/utils/parallel.py` y `_testing.py`, `optuna/_warnings.py`. Ninguno los pone
   junto a las clases públicas.
3. **Localizar el código del usuario:** el único helper que recorre los frames es el
   `find_stack_level` de pandas. Hace lo mismo que `_first_frame_outside_skforecast` y
   `_stacklevel_user_code`: sale del paquete y cuenta los tests como código del usuario.
   La alternativa moderna es `skip_file_prefixes` (Python 3.12 o posterior), que usan
   optuna y numpy. skforecast admite `>=3.10`, así que todavía no sirve.
4. **Avisos de los procesos de joblib:** ninguna librería los reenvía. sklearn solo envía
   los filtros del padre al proceso (`_FuncWrapper.__call__`). Lo más parecido a
   `_reemit_fold_warnings` es el `rewrite_warning` de pandas (registrar y reemitir con
   `warn_explicit`). Lo más parecido a `_SearchWarnings` es el
   `_warn_or_raise_about_fit_failures` de sklearn (agrupa los errores iguales con un
   `Counter`).
5. **Nombre `_warnings.py`:** lo usan optuna (mecanismos), `shap/utils/_warnings.py`,
   `pluggy` y `bs4` (clases), y `pandas/_testing/_warnings.py` (aserciones). Los he
   comprobado así:
   - El módulo C `_warnings` de CPython está en `sys.builtin_module_names` y
     `BuiltinImporter` va antes que el buscador por rutas.
   - Un `_warnings.py` en el directorio actual **no** lo sombrea: con un `_warnings.py` que
     lanza `SystemExit` en el cwd, `import _warnings` devolvió el builtin.
   - El riesgo que queda es solo de lectura: un traceback o un `grep` muestra
     `_warnings.py`.

## 3. Convenciones de skforecast

**Prefijos de las funciones del paquete** (sin tests):

| Prefijo | Funciones |
|:--|:--|
| `check_` | 19 |
| `_get_` | 13 |
| `_check_` | 9 |
| `_warn_` | 8 |
| `initialize_` | 7 |
| `get_` | 7 |
| `_fit_` | 7 |
| `select_` | 4 |
| `_validate_` | 4 |
| `_is_` | 4 |
| `_calculate_` | 4 |
| `_build_` | 4 |
| `prepare_` | 3 |
| `create_` | 3 |

El estilo es `_verbo_nombre`, con un sufijo de contexto.

**Sufijos de contexto:**

| Sufijo | Funciones | Ejemplos |
|:--|:--|:--|
| `_multiseries` | 23 | `_extract_data_folds_multiseries`, `_select_fit_folds_multiseries` |
| `_search` | 4 | `_check_metric_names_search`, `_train_test_split_one_step_ahead_search` |
| `_direct` | 3 | `exog_to_direct`, `prepare_steps_direct` |
| `_foundation` | 4 | |

Los sufijos se combinan (`_get_n_evaluated_search_multiseries`). Son dos ejes distintos:
`_search` dice dónde se usa y `_multiseries`, para qué familia de forecasters.

**Textos de mensajes:** `*_message` (`_all_samples_removed_message`,
`_SearchWarnings._long_training_message`) y `_format_*` (`_format_lags_warning`,
`_format_names_warning`).

**Módulos privados:**

- `_utils.py` está en `model_selection`, `foundation` y `stats`. `deep_learning/utils.py`
  y `utils/utils.py` son públicos.
- Los demás se nombran con un sustantivo del tema: `_split.py`, `_search.py`,
  `_validation.py`, `_calendar.py`, `_differencing.py`, `_adapters.py`,
  `_multi_estimator_mixin.py`.
- `utils/` todavía no tiene ningún módulo privado.

**Tests:** van en `<pkg>/tests/tests_<módulo sin _>/test_<función>.py`, con `__init__.py`.
`.claude/skills/verify/affected.py` aplica esa regla (`filename[:-3].lstrip("_")`), así que
`utils/_warnings.py` se asocia sin configurar nada a `utils/tests/tests_warnings/`.

## 4. Propuesta de organización

### 4.1 Qué se mueve a `skforecast/utils/_warnings.py`

| Nombre actual | Nombre propuesto | Por qué se mueve |
|:--|:--|:--|
| `manage_warnings` (con `P` y `R`) | igual | Es el mecanismo de silenciado. Con él en el mismo módulo, todo lo que decide qué avisos se ven y dónde apuntan queda junto, y los helpers de los procesos dependen de su comportamiento: los avisos silenciados dentro de la función no se registran |
| `_first_frame_outside_skforecast` | igual | Genérico, como el `find_stack_level` de pandas |
| `_stacklevel_user_code` | `_find_stack_level` | Genérico. Más adelante podría sustituir los `stacklevel=2` o `3` fijos de QW-13 (eso cambiaría el comportamiento, así que va aparte) |
| `FOLD_WARNINGS_ATTRIBUTE` | `_WORKER_WARNINGS_ATTRIBUTE` | QW-15 lo usará desde `direct` |
| `_run_fold_recording_warnings` | `_run_recording_worker_warnings` | QW-15 |
| `_serializable_fold_warnings` | `_serializable_warnings` | Lo usa la función anterior |
| `_module_name_from_filename` | igual | Lo usa `_reemit_*` |
| `_reemit_fold_warnings` | `_reemit_worker_warnings` | QW-15 |

Son unas 280 líneas: 235 de `model_selection/_utils.py` y 43 de `utils/utils.py`. El
módulo solo importa la biblioteca estándar y `from ..exceptions import
warn_skforecast_categories`, que necesita `manage_warnings`.

### 4.2 Qué se queda y por qué

- **`_SearchWarnings`, `_warn_recorded`, `_split_first_sentence`, los cuatro helpers de
  Optuna y los textos de las búsquedas** se quedan en `model_selection/_utils.py`:
  - Son específicos de la búsqueda: candidatos, trials, `LongTrainingWarning` con el
    número de fits de la búsqueda, y el aviso de los candidatos saltados, que `emit` llama
    directamente.
  - Ningún módulo de fuera de `model_selection` los necesita, tampoco con QW-15.
  - Si se movieran a `utils`, `utils` importaría de `model_selection` y se invertiría la
    dirección de los imports.
- **QW-13** (`_warn_nan_training_rows*`, `_filter_nan_train_X_y`) se queda en
  `utils/utils.py`: es lógica del dominio cuyo texto va con ella.
- **Las clases, `warn_skforecast_categories`, los handlers, `set_warnings_style` y
  `runtime_deprecated`** se quedan en `skforecast/exceptions/exceptions.py`: son públicos y
  están documentados, y así lo hacen sklearn, numpy y optuna.

### 4.3 Dirección de los imports (sin ciclos)

```
exceptions  ←  utils/_warnings  ←  utils/utils  ←  forecasters (recursive, direct, ...)  ←  model_selection
                     ↑_______________________________________________↑ (QW-15)               ↑ (_utils, _validation)
```

`utils/_warnings.py` no importa nada de `utils/utils.py`. `utils/utils.py` importa
`manage_warnings` de `._warnings`: lo usa en dos decoradores y así mantiene la ruta
`skforecast.utils.utils.manage_warnings`.

Lo he comprobado en una copia del paquete (`git archive HEAD` en el scratchpad, con
`PYTHONPATH` apuntando a ella) con el movimiento aplicado y los nombres sin cambiar:

- **Importaciones:** importando primero, en un intérprete nuevo cada vez,
  `skforecast.utils._warnings`, `skforecast.direct._forecaster_direct`,
  `skforecast.model_selection`, `skforecast.utils` y `skforecast.recursive`, todo carga y
  `model_selection._utils._reemit_fold_warnings is utils._warnings._reemit_fold_warnings`.
  Lo mismo para `manage_warnings` por sus tres rutas.
- **Tests:** `pytest -m "not slow"` de `model_selection/tests/tests_utils`,
  `utils/tests/tests_utils/test_manage_warnings.py`, `test_grid_search_forecaster.py`,
  `test_evaluate_grid_hyperparameters.py`, `test_bayesian_search_forecaster.py`,
  `test_backtesting_forecaster.py`, `test_backtesting_forecaster_multiseries.py` y
  `direct/tests` dio **1666 passed, 1 deselected** en 94 s. Entre ellos están:
  - el test de un fold que falla en un proceso de loky, que comprueba el pickle por
    referencia de la función desde su nueva ruta;
  - los tests de `_SearchWarnings` que comprueban que el aviso apunta al código del usuario,
    que comprueban `_first_frame_outside_skforecast` desde su nueva ubicación.

`_first_frame_outside_skforecast` calcula el directorio del paquete con
`dirname(dirname(abspath(__file__)))`, así que depende de la profundidad del módulo.
`utils/_warnings.py` está a la misma profundidad que `model_selection/_utils.py`, y los
tests lo confirman. En otro commit, que no sea de movimiento, se podría calcular desde
`skforecast.__file__`, como pandas (`os.path.dirname(pd.__file__)`), para que no dependa
de dónde esté. Es opcional: decisión D8.

### 4.4 Reexportación

**`skforecast/utils/__init__.py`**, con el mismo patrón que los privados actuales:

```python
from .utils import *
from .utils import (
    ...  # sin cambios
)
from ._warnings import (
    manage_warnings,
    _find_stack_level,
    _first_frame_outside_skforecast,
    _module_name_from_filename,
    _reemit_worker_warnings,
    _run_recording_worker_warnings,
    _serializable_warnings,
    _WORKER_WARNINGS_ATTRIBUTE
)
```

`manage_warnings` ya llegaría por `from .utils import *`, porque `utils.py` lo importa y no
tiene `__all__`. Aun así es mejor importarlo explícitamente, para que no dependa de ese
efecto.

**Consumidores:**

- `model_selection/_utils.py` importa de `..utils` `_first_frame_outside_skforecast` y
  `_find_stack_level`.
- `model_selection/_validation.py` importa de `..utils` los tres nombres de los procesos,
  en lugar de importarlos de `._utils`.
- Los 12 módulos de `manage_warnings` no cambian.
- `skforecast.model_selection._utils` deja de ofrecer los nombres de los procesos. Ningún
  test los importaría ya de ahí (ver 4.5).

### 4.5 Tests

Nueva carpeta `skforecast/utils/tests/tests_warnings/` (con `__init__.py`):

| Ahora | Después |
|:--|:--|
| `model_selection/tests/tests_utils/test_run_fold_recording_warnings.py` (8 tests) | `utils/tests/tests_warnings/test_run_recording_worker_warnings.py` |
| `model_selection/tests/tests_utils/test_reemit_fold_warnings.py`, 4 de sus 5 tests | `utils/tests/tests_warnings/test_reemit_worker_warnings.py` |
| `test_reemit_fold_warnings_collected_by_SearchWarnings_with_candidate` | Pasa a `model_selection/tests/tests_utils/test_search_warnings.py`: prueba la integración con el registro de la búsqueda, y así los tests de `utils` no importan de `model_selection` |
| `utils/tests/tests_utils/test_manage_warnings.py` | `utils/tests/tests_warnings/test_manage_warnings.py`. Sigue importando `skforecast.utils.utils.manage_warnings`, lo que prueba esa ruta |

`_first_frame_outside_skforecast` y `_find_stack_level` no tienen tests directos: los
cubren los de `_SearchWarnings`. Añadir `test_find_stack_level.py` sería un cambio de
tests, no un movimiento, y puede ir en un commit aparte (decisión D7).

## 5. Revisión de los nombres nuevos del ciclo

Criterio: proponer un cambio solo si el nombre engaña o va a engañar, o si rompe una
convención clara.

**QW-13 (#1466)**

| Nombre | ¿Describe lo que hace? | Propuesta | Motivo |
|:--|:--|:--|:--|
| `_warn_nan_training_rows` | Sí | **mantener** | `_warn_*`, como los otros 8 |
| `_warn_nan_training_rows_direct` | Sí: cuenta por paso y avisa | **mantener** | `_direct`, como `exog_to_direct` y `prepare_steps_direct` |
| `_filter_nan_train_X_y` | Sí: filtra y avisa | **mantener** | `train_X_y`, como `create_train_X_y`. "train" frente a "training" no justifica el ruido |

**PR 1 (#1447)**

| Nombre | ¿Describe lo que hace? | Propuesta | Motivo |
|:--|:--|:--|:--|
| `N_EVALUATED_WARNING_THRESHOLD` | Sí | **mantener** | |
| `_format_lags_warning`, `_format_names_warning` | Casi. El sufijo admite dos lecturas ("formatear el aviso de los lags" o "formatear los lags para un aviso"). El verbo `_format_` sí dice que devuelve texto y no emite | **mantener** | Ver la nota 1. Mejora pequeña frente a renombrar un archivo de test |

**PR 2a (#1452)**

| Nombre | ¿Describe lo que hace? | Propuesta | Motivo |
|:--|:--|:--|:--|
| `LEVELS_NOT_PREDICTED_WARNING` | Sí: el inicio del texto del aviso | **mantener** | |
| `_SearchWarnings` | Sí | **mantener** | Context manager privado en CamelCase, como `_IgnoreWarnings` de sklearn |
| `_SearchWarnings` (métodos) `set_candidate`, `add_skipped`, `discard`, `emit`, `_record`, `_long_training_message` | Sí | **mantener** | `emit` es el término de `logging.Handler.emit`; `_record`, el de `catch_warnings(record=True)` |
| `_warn_recorded` | Sí: emite un aviso registrado con otro texto | **mantener** | Solo lo usa `emit` |
| `_split_first_sentence` | Sí | **mantener** | |
| `_warn_skipped_candidates_search` | Sí | **mantener** | El sufijo `_search` sobra (solo hay candidatos saltados en las búsquedas), pero es coherente con `_warn_n_evaluated_search` |
| `_first_frame_outside_skforecast` | Sí | **mantener** | Devuelve el frame y el número de frames saltados, que es más que un `stacklevel` |
| `_stacklevel_user_code` | Sí, pero es un sintagma nominal (lo único sin verbo de la lista) | **`_find_stack_level`** | Misma semántica que `pandas.util._exceptions.find_stack_level`, el nombre que conocen quienes vienen de pandas. Sigue `_verbo_nombre`. Si se mueve a `utils` como herramienta del paquete, un nombre reconocible ayuda a que se use. 9 apariciones, todas en `_utils.py`, sin tests directos |

**PR 2b (#1456)**

| Nombre | ¿Describe lo que hace? | Propuesta | Motivo |
|:--|:--|:--|:--|
| `_warn_n_evaluated_search` | Sí | **mantener** | |
| `_n_evaluated_covers_levels_not_predicted`, `_is_affected_n_evaluated` | Sí | **mantener** | Predicados; `_is_*` como los otros 4 |
| `_TrialEvaluationError`, `_skip_trial`, `_FailedTrialsLogFilter`, `_optimize_study` | Sí | **mantener** | |

**#1415, del ciclo**

| Nombre | ¿Describe lo que hace? | Propuesta | Motivo |
|:--|:--|:--|:--|
| `_warn_levels_not_predicted_multiseries` | Sí | **mantener** | Ver la nota 2 |

**PR 2c (#1467, esta rama)**

| Nombre | ¿Describe lo que hace? | Propuesta | Motivo |
|:--|:--|:--|:--|
| `FOLD_WARNINGS_ATTRIBUTE` (valor `"_skforecast_fold_warnings"`) | Hoy sí; con QW-15, no | **`_WORKER_WARNINGS_ATTRIBUTE`** (valor `"_skforecast_worker_warnings"`) | Ver la nota 3. El valor es un atributo interno de la excepción; ningún test usa el literal |
| `_run_fold_recording_warnings(fold_function, main_pid, **kwargs)` | Hoy sí; con QW-15, no | **`_run_recording_worker_warnings(function, main_pid, **kwargs)`** | Mantiene el verbo y la forma "ejecutar registrando", y dice de quién son los avisos. El parámetro pasa a `function`, como en `joblib.delayed(function)`. `fold_function` aparece 9 veces en el código y 8 en los tests |
| `_serializable_fold_warnings` | Hoy sí | **`_serializable_warnings`** | No tiene nada de fold: convierte lo que devuelve `catch_warnings(record=True)` |
| `_module_name_from_filename` | Sí | **mantener** | |
| `_reemit_fold_warnings` | Hoy sí | **`_reemit_worker_warnings`** | Ver la nota 4 |

Notas:

1. **`_warn_*` frente a `_format_*_warning`.** No son incoherentes: tienen papeles
   distintos. `_warn_*` emite y `_format_*` devuelve texto. Si alguna vez se renombran, los
   nombres claros serían `_format_lags_for_warning` y `_format_names_for_warning`, pero no
   lo propongo.
2. **`_search` frente a `_multiseries`.** No son incoherentes: son ejes distintos (dónde se
   usa frente a qué familia de forecasters), y la casa los combina.
3. **`fold` frente a `worker`.** Con QW-15, el `fit` de Direct y DMV usará estos helpers
   para los pasos, que no son folds. Los docstrings ya dicen "worker process of joblib".
   Los nombres nacen en este PR, así que cambiarlos ahora evita que lleguen a `0.27.x` con
   un nombre que dejará de ser cierto unas semanas después.
4. **`reemit`.** Es preciso: es el mismo aviso (categoría, ubicación y módulo), emitido
   otra vez con `warn_explicit`, como hace `rewrite_warning` en pandas. `show` chocaría con
   `warnings.showwarning`, que es otra capa.
5. **`record` frente a `collect`.** El código usa `record` en todas partes (`_record`,
   `records`, `_run_*_recording_*`), que es el término de la biblioteca estándar
   (`catch_warnings(record=True)`). `collect` solo aparece en dos docstrings ("collects the
   warnings", "so they are collected too"). Si se quiere uniformar, basta cambiar esas dos
   palabras en el commit de renombres.

Las variables locales y los textos de los docstrings que dicen "fold warnings" en estos
helpers se ajustan en el mismo commit. Los docstrings de `_SearchWarnings` y de
`_validation.py` citan `_reemit_fold_warnings` y se actualizan con él.

## 6. Coste, riesgos y en qué PR

### Archivos que cambian

| Archivo | Cambio |
|:--|:--|
| `skforecast/utils/_warnings.py` | Nuevo |
| `skforecast/utils/utils.py` | Sale `manage_warnings` con `P` y `R`; entra el import |
| `skforecast/utils/__init__.py` | Reexporta los nombres movidos |
| `skforecast/model_selection/_utils.py` | Salen unas 235 líneas; cambian los imports y las 9 llamadas a `_stacklevel_user_code` |
| `skforecast/model_selection/_validation.py` | Imports y 3 bloques de `Parallel` (solo nombres) |
| Tests | 3 archivos que se mueven o se reparten, 1 `__init__.py` nuevo y `test_search_warnings.py`, que recibe un test |
| `dev/roadmap_0.27.0/roadmap.md` (QW-15) y `BUG-search-short-series-transparency.md` (15.8) | Nombres nuevos; `dev/` no necesita `verify` |

No cambian ni `docs/`, ni `skills/`, ni `tools/ai/`, ni las notas de la versión, porque todo
es privado y no hay cambio de comportamiento.

### Ruido en `git blame`

Se mueven unas 280 líneas: 235 del PR 2a y el 2c, y 43 de `manage_warnings`, que es
antiguo. Para limitarlo:

- El commit de movimiento no debe tocar ni una línea movida.
- Con eso, `git blame -C` atribuye las líneas a su commit original, porque el archivo de
  origen cambia en el mismo commit.
- Los renombres van en otro commit. El repositorio no tiene `.git-blame-ignore-revs` y no
  hace falta crearlo para esto.

### API pública

Solo `manage_warnings`, que mantiene sus rutas `skforecast.utils.manage_warnings` y
`skforecast.utils.utils.manage_warnings`. Los demás nombres son privados y, salvo
`manage_warnings`, ninguno se ha publicado.

### Riesgos comprobados

- **Pickle de `delayed(_run_*)` en loky:** la función se serializa por referencia. El test
  con loky pasa con la nueva ruta.
- **Profundidad de `__file__` en `_first_frame_outside_skforecast`:** es la misma, y los
  tests de la ubicación de los avisos pasan.
- **Sombra del `_warnings` builtin:** no ocurre (sección 2, punto 5).
- **Ciclos de import:** no hay (sección 4.3).

### Riesgos no comprobados

- **Otras ramas abiertas** que toquen el final de `model_selection/_utils.py` tendrán
  conflictos. No he mirado otras ramas locales.
- **Renombres:** el experimento solo movió. Son mecánicos, y he comprobado que los nombres
  nuevos no existen ya en `skforecast/`.

### ¿En este PR o en otro?

Recomiendo **este PR**. A favor:

- Los cinco nombres de los procesos nacen en este PR. Moverlos y renombrarlos antes de
  fusionar evita que `0.27.x` tenga dos versiones de la historia y deja a QW-15 un import
  directo desde `utils`.
- El coste es S y el comportamiento no cambia.

En contra, sin que pese lo bastante:

- El diff que se revisa crece en unas 280 líneas movidas. Se mitiga con commits que solo
  muevan o solo renombren y con `git diff --color-moved`.
- Si hubiera que revertir el PR 2c, el refactor se iría con él. Es poco probable, y un
  revert de un PR fusionado con merge commit lo revierte entero de todas formas.
- La descripción del PR #1467 debería mencionar el movimiento en una línea. Eso lo decides
  tú, porque no toco el PR.

Plan de commits (en local, sin push, con `verify` antes de cada uno):

1. **Mover** sin renombrar:
   - los 7 helpers y `manage_warnings` a `utils/_warnings.py`;
   - reexportarlos;
   - cambiar los imports de `_utils.py`, `_validation.py` y `utils.py`;
   - mover los tests a `utils/tests/tests_warnings/` (con el reparto de 4.5).
2. **Renombrar** `fold` → `worker`: 4 nombres, el parámetro `function` y el valor del
   atributo, con sus tests, archivos de test y docstrings.
3. **Renombrar** `_stacklevel_user_code` → `_find_stack_level`. Puede ir con el 2 si
   prefieres un solo commit de renombres.
4. **Docs de `dev/`:** QW-15 en `roadmap.md` y la sección 15.8 de
   `BUG-search-short-series-transparency.md`.

## 7. Decisiones abiertas

| # | Decisión | Recomendación | Tamaño |
|:--|:--|:--|:--|
| D1 | Crear `skforecast/utils/_warnings.py` para los mecanismos genéricos | **Sí** | S |
| D2 | Mover `_SearchWarnings` (y sus satélites) a `utils` | **No**: depende de los textos de la búsqueda y no tiene consumidores fuera de `model_selection` | |
| D3 | Mover `manage_warnings` a `_warnings.py`, reexportado en `utils.py` y `utils/__init__.py` | **Sí**. Es la parte más prescindible: el beneficio es solo de cohesión, y el riesgo es bajo porque se mantienen las dos rutas | XS |
| D4 | Nombre del módulo: `_warnings.py`, `_warning_utils.py` o `_parallel.py` | **`_warnings.py`**. Tiene precedentes en optuna, shap y pandas, y no sombrea el builtin (comprobado). `_parallel.py` dejaría fuera `manage_warnings` y la localización del código del usuario | |
| D5 | Renombrar `fold` → `worker` en los 4 nombres y el parámetro del PR 2c | **Sí, en este PR** | S |
| D6 | Renombrar `_stacklevel_user_code` → `_find_stack_level` | **Sí** (opcional: mejora la coherencia, no corrige nada) | XS |
| D7 | Añadir tests directos de `_find_stack_level` y `_first_frame_outside_skforecast` | Sí, en un commit de tests aparte después de los movimientos, o en QW-15 | XS |
| D8 | Calcular el directorio del paquete desde `skforecast.__file__` en lugar de `dirname(dirname(__file__))` | Opcional. Va en un commit que no sea de movimiento; quita la dependencia de la profundidad | XS |
| D9 | Una función que envuelva `Parallel`, el `try`/`except` y la reemisión (como `sklearn.utils.parallel`), para las 3 copias de `_validation.py` y las 2 de QW-15 | **No ahora**: no es un movimiento. Se decide en QW-15, con la medición que ya pide | S |
| D10 | Sacar los avisos de las búsquedas de `model_selection/_utils.py` a `model_selection/_search_warnings.py` | **No**: no hay un segundo consumidor, crecería el ruido en `git blame`, y el tamaño de `_utils.py` es un problema más amplio que este | M |
| D11 | Sustituir el recorrido de frames por `skip_file_prefixes` | **No** mientras skforecast admita Python 3.10 y 3.11 (requiere 3.12 o posterior) | |
| D12 | Rarezas públicas de `exceptions.py` (`warn_skforecast_categories` como nombre de lista, `format_warning_handler` sin uso) | **Fuera de alcance**: necesitarían deprecación | |
