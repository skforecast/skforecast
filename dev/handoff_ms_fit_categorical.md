# Handoff: BUG-ms-fit-dataframe-categorical, arreglo con pandas y migración a numpy

## 1. Estado

- Fecha: 2026-11-27.
- Rama `fix/ms-fit-numpy-estimator`, creada desde `0.28.x` en `a7c7a869d` y
  subida a `origin`. Va 3 commits por detrás de `origin/0.28.x` (los de #1547),
  sin conflictos. No tiene PR y no es para fusionar tal cual.
- El trabajo de numpy está en su commit `082cf9c52`: 22 ficheros de
  `skforecast/` (2 de test nuevos y 1 eliminado), su documentación
  (`releases.md`, la guía de categóricas, `llms-base.txt`, el skill
  `feature-engineering`) y los ficheros de contexto IA regenerados.
- El roadmap, este handoff y `dev/categorical_review/ms_fit_numpy/` están en
  `0.28.x`.
- `verify` en verde sobre el árbol de numpy el 2026-11-27 (ver sección 5).

## 2. Objetivo y decisiones

**El bug.** `ForecasterRecursiveMultiSeries` da al estimador un DataFrame en
`fit` y un array float en `predict`. Con `categorical_features=None`, o una
lista que omite una columna, y una exog `category` cuyos valores no coinciden
con sus códigos (por ejemplo 10, 20, 30), LightGBM y XGBoost entrenan con los
códigos y predicen con los valores: predicciones erróneas sin aviso, que la
validación one-step-ahead no ve.

**Decisión del 2026-11-27 (Joaquín): dos PR en lugar de una.**

1. **PR de arreglo, con pandas.** El estimador sigue recibiendo el DataFrame y
   solo se convierten a sus valores las exog `category` que
   `categorical_features` no gestiona.
2. **PR de migración a numpy**, a partir de esta rama, con la gestión de
   categóricas restringida.

Por qué, tras dos revisiones independientes
([informes](categorical_review/ms_fit_numpy/informes/)):

- El arreglo con pandas, implementado por los dos revisores, corrige todos los
  escenarios con predicciones erróneas en silencio (78 de 218 en la matriz de
  uno y 96 de 352 en la del otro) con +76 a +85 líneas en 3 ficheros. La rama
  necesita +742 / -199 en 5 ficheros y cambia contratos internos (la división
  one-step-ahead devuelve 9 elementos).
- Unas 280 de las líneas de la rama sirven a la configuración categórica que
  el usuario hace por su cuenta, que solo hay que traducir porque el estimador
  recibe numpy.
- La rama tiene una regresión que tres revisiones no habían visto: dejan de
  funcionar los parámetros del estimador que nombran columnas y no son
  categóricos (`monotonic_cst`, `monotone_constraints`,
  `interaction_constraints`). Reproducido:
  [`ms_byname_params.py`](categorical_review/repro/ms_byname_params.py).
- Lo que solo arregla numpy son fallos explícitos de `fit`, no resultados
  erróneos: exog pyarrow con LightGBM, nombres con `[`, `]`, `<` en LightGBM y
  XGBoost, y CatBoost con `None` y `ordinal_category`.

**Alcance acordado de la PR de arreglo (pandas):**

- Convertir la `category` sin gestionar a sus valores en `fit`,
  `set_in_sample_residuals` y `_train_test_split_one_step_ahead`.
- No romper CatBoost con `cat_features` en el constructor (tupla) y una exog
  `category`: la primera versión de los dos prototipos lo rompía.
- `DataTypeWarning` con texto neutro, que no dependa de la configuración del
  usuario en el estimador.
- `cat_features` de CatBoost por nombre en `fit_kwargs` (hoy `TypeError: list
  indices must be integers`).
- `TypeError` que nombre la columna cuando la `category` sin gestionar no se
  puede tratar como número (texto).
- Tests que fijen lo que la PR de numpy puede romper: `monotonic_cst` e
  `interaction_constraints` por nombre en multiserie, y el nivel categórico en
  un estimador envuelto con `ordinal_category`.
- Las notas de DOC-lightgbm-categorical-few-rows y
  DOC-categorical-index-value-range, y la nota de versión reescrita.

**Fuera de la PR de arreglo:** las restricciones de categóricas (no hacen falta
si el estimador recibe el DataFrame), CatBoost con `None` y `ordinal_category`,
el cambio de mensaje de `check_exog_dtypes` (función compartida, tests de cinco
forecasters), el aviso repetido en cada fold, pyarrow y los nombres con
caracteres especiales.

**Para la PR de numpy: regla de las dos puertas** (propuesta que a Joaquín le
gusta; confirmar antes de implementarla):

- Puerta principal, `categorical_features` del forecaster (`'auto'` o lista):
  no se toca.
- Puerta manual, `categorical_features=None`: skforecast no interpreta ni
  completa la configuración del usuario.
  1. No fusionar el nivel de `ordinal_category` con la configuración propia del
     usuario: sin configuración propia, el forecaster configura solo el nivel;
     con ella, error que remite a `categorical_features=[...]`.
  2. Traducir nombres a posiciones solo en `fit_kwargs`. Lo que está en el
     objeto estimador va por posición. Se pierde
     `HistGradientBoostingRegressor(categorical_features=['col'])` con `None`,
     que hoy funciona.
  3. Quitar los alias y formatos de texto de LightGBM en el constructor, que en
     la base nunca funcionaron.
- Con 1 y 2 debería desaparecer el bloque que fija y restaura parámetros del
  estimador alrededor de `fit` (deducido del código, sin probar).

**Más adelante, si un perfil lo justifica:** meter todas las columnas en el
bloque float de la matriz y reconstruir el DataFrame solo en
`create_train_X_y`. La alineación, la unión de exógenas, `transformer_exog` y
la codificación se quedan en pandas. Es viable (ida y vuelta de dtypes idéntica
en 24 combinaciones por versión de pandas,
[`roundtrip.py`](categorical_review/ms_fit_numpy/scripts_dudas_fable/roundtrip.py))
y no simplifica la gestión de categóricas.

**Descartado antes, en la primera revisión (Joaquín, 2026-11-27):** rechazar
exog de números complejos, anotar la eliminación de
`cast_catboost_categorical_columns_dataframe`, y un aviso en el código por
etiquetas grandes o negativas en columnas configuradas como categóricas (solo
la nota en la guía).

## 3. Hecho

**Rama de numpy (`fix/ms-fit-numpy-estimator`, commit `082cf9c52`).**

- `skforecast/utils/utils.py`: `_train_matrix_to_numpy`,
  `_train_matrix_to_estimator_input`, `_resolve_user_categorical_config`;
  `_copy_rows_to_check` y `_check_in_place_fit` aceptan arrays.
- `skforecast/recursive/_forecaster_recursive_multiseries.py`: `fit`,
  `set_in_sample_residuals`, `_train_test_split_one_step_ahead` (devuelve las
  matrices ya convertidas y los nombres), `_get_fit_config` con el
  `DataTypeWarning`.
- `skforecast/model_selection/_utils.py` y `_search.py`: métricas
  one-step-ahead de multiserie y desempaquetado de la división.
- Tests: carpetas de multiserie, `utils` y `model_selection`, más
  `test_resolve_user_categorical_config.py` y
  `test_train_matrix_to_estimator_input.py` (nuevos).
- Nota de versión, guía, skill y `llms-base.txt`: escritos para el camino
  numpy. Faltan los enlaces a la PR en 5 entradas.

**Revisiones y prototipos** (en
[`dev/categorical_review/ms_fit_numpy/`](categorical_review/ms_fit_numpy/README.md)):
cuatro informes, los diff del arreglo con pandas de cada revisor y sus scripts.

**Roadmap** (`dev/ROADMAP.md`), actualizado el 2026-11-27:

- BUG-ms-fit-dataframe-categorical es ahora el arreglo con pandas (P1, S) y el
  paso a numpy tiene su fila, MAINT-ms-fit-numpy-estimator (P2, propuesta por
  valorar), con la regla de las dos puertas y lo que hay que decidir.
- DOC-lightgbm-categorical-few-rows y DOC-categorical-index-value-range van con
  el arreglo con pandas.
- Cinco filas de propuestas de la sesión: API-public-matrices-none-category,
  QW-check-exog-dtypes-column-names, QW-exog-text-not-managed-warning,
  PERF-ms-xgboost-mixed-copy (requiere MAINT-ms-fit-numpy-estimator) y
  QW-ms-category-warning-per-fold.
- La sección 11 (selección de la 0.28.0, P1 y cadena C2) recoge el replanteo.
- Sin revisar: MAINT-ms-predict-core sigue teniendo como requisito
  BUG-ms-fit-dataframe-categorical, y la fila de
  BUG-predict-readonly-wrapped-catboost cita los 48 casos que la PR 1 con numpy
  dejaba en ese error. Comprobar si siguen siendo ciertos con el arreglo con
  pandas.

## 4. Pendiente

**Para el autor:**

1. Confirmar la variante del arreglo: convertir siempre (recomendada: es la
   semántica ya decidida, D-NP2 = (a'), y la que tendrá numpy) o no convertir
   cuando los códigos ya son los valores (ningún test de la base cambia, pero
   es más difícil de explicar).
2. Confirmar la regla de las dos puertas antes de la PR de numpy.
3. Decidir la prioridad de MAINT-ms-fit-numpy-estimator (en el roadmap, P2
   como propuesta) y si entra en la 0.28.0.

**Para la siguiente sesión (PR de arreglo con pandas), en orden:**

1. Comprobar que el roadmap, este handoff y
   `dev/categorical_review/ms_fit_numpy/` ya están en `0.28.x`.
2. Rama nueva desde `origin/0.28.x` con `--no-track`.
3. Plan del arreglo, a partir de
   [`fable_minimo.diff`](categorical_review/ms_fit_numpy/arreglo_pandas/fable_minimo.diff)
   y [`opus_minimo.diff`](categorical_review/ms_fit_numpy/arreglo_pandas/opus_minimo.diff).
   Puntos a resolver en el plan:
   - Una `category` de decimales sin gestionar: el 2026-11-26 se decidió
     `TypeError`, pero pensando en numpy. Comprobar si con pandas es coherente
     entre `fit` y `predict` antes de mantener el error.
   - El cambio de modelo que el arreglo comparte con la rama: con `None`,
     HistGradientBoosting con categorías 10, 20, 30 y LightGBM con categorías
     0, 1, 2 pasan de categórico a numérico. El test
     `test_create_predict_X_when_categorical_features_auto_detect_LGBMRegressor`
     falla y hay que adaptarlo. Va en "Before upgrading".
   - LightGBM con `'name:col'` en el constructor pasa de resultado erróneo en
     silencio a `LightGBMError`.
4. Tests: primero los que fallan sin el arreglo. Los de comportamiento de la
   rama ya pasan contra los prototipos (predicciones con `None` iguales a las
   de la columna entera, residuos de `set_in_sample_residuals`, one-step-ahead
   igual a backtesting con LightGBM, XGBoost e HistGradientBoosting); el de
   equivalencia con backtesting necesita adaptar la firma interna.
5. Documentación: revisar qué celdas de `categorical-features.ipynb`, del skill
   y de `llms-base.txt` siguen siendo ciertas con pandas; reescribir la nota de
   versión (casi todas las entradas de Changed de la rama dejan de aplicar).
6. `verify`, con los dos pandas.

**Para la PR de numpy, después:**

- Rehacerla con esta rama como referencia: el arreglo de pandas toca los
  mismos tres puntos y la regla de las dos puertas reescribe la configuración
  de categóricas. Se reutilizan la conversión a numpy, la comprobación in situ
  con arrays, el paso de nombres en `model_selection` y los tests de
  comportamiento.
- Decidir a conciencia y documentar: los parámetros por nombre que no son
  categóricos, el nivel numérico en estimadores envueltos con
  `ordinal_category` (también con `'auto'`), la pérdida de `feature_names_in_`
  y la comprobación in situ más estricta con dtypes mezclados.
- Sin comprobar por nadie: versiones mínimas de XGBoost, LightGBM y CatBoost,
  Linux y macOS, GPU, la suite completa, las búsquedas en paralelo con el
  bloque que fija y restaura parámetros, y los forecasters guardados con una
  versión anterior.

## 5. Verificación

Estado del árbol de numpy al final de la sesión (2026-11-27):

- `python .claude/skills/verify/affected.py`: 31 ficheros cambiados antes de
  añadir lo de `dev/`.
- Lint: los dos ficheros de test editados el 2026-11-27 pasan ruff; en el resto
  de ficheros cambiados hay 35 avisos que ya estaban (30 F401, 4 E721, 1 F403).
- Tests afectados (carpetas de `tests_search`, `tests_utils` de
  `model_selection`, multiserie, `utils` y los `test_create_train_X_y.py` de
  cuatro forecasters): 3424 passed con pandas 2.3.3; 3421 passed y 3 skipped
  con pandas 3.1.1.
- `python tools/ai/generate_ai_context_files.py --check`,
  `python tools/docs/hooks/page_meta.py` y `execute_notebooks.py --check` de
  `categorical-features.ipynb`: en verde. El notebook no se reejecutó.
- No se ha ejecutado la suite completa.
