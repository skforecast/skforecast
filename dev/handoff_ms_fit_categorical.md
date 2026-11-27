# Handoff: BUG-ms-fit-dataframe-categorical, arreglo con pandas y migración a numpy

## 1. Estado

- Fecha: 2026-11-27.
- **Arreglo con pandas: implementado el 2026-11-27** en la rama
  `fix/ms-category-exog-values`, creada desde `origin/0.28.x` (`ca0f2f4af`).
  **PR #1548 abierta** ese día contra `0.28.x`: commits `210086978` (código
  y tests) y `9d83b5664` (nota de versión, guía y contexto IA), más el del
  enlace en la nota de versión y los `dev:` del roadmap y de este handoff,
  que va en la PR por decisión de Joaquín (llega a `0.28.x` al fusionar).
  `verify` con pandas 2.3.3 y 3.1.1 en la sección 5.
- Rama `fix/ms-fit-numpy-estimator`, creada desde `0.28.x` en `a7c7a869d` y
  subida a `origin`. Va 4 commits por detrás de `origin/0.28.x` (los 3 de
  #1547 y el del roadmap, `ca0f2f4af`) y 2 por delante (`082cf9c52` y
  `ed6b152ee`, que es el mismo commit del roadmap, con el mismo contenido en
  `dev/`), sin conflictos. No tiene PR y no es para fusionar tal cual.
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

**Decisiones de la PR de arreglo (Joaquín, 2026-11-27, con medidas):**

- **Convertir siempre** la `category` sin gestionar a sus valores, no solo
  cuando sus códigos difieren de ellos. Con la variante, CatBoost con `None`
  fallaba con categorías `0..5` y funcionaba con 10..60, y el tipo de modelo
  dependía de qué valores había en el entrenamiento (con `astype('category')`,
  LightGBM categórico con `0..5` y numérico si falta un valor).
  `ForecasterRecursive` ya da un modelo numérico en ese caso.
- **Una `category` de decimales fuera de la lista se convierte a sus valores**,
  sin `TypeError` (cambia lo decidido el 2026-11-26): con pandas es coherente
  entre `fit` y `predict` en los 6 estimadores probados, es lo que hace
  `ForecasterRecursive`, y el `TypeError` rompía HistGradientBoosting y Ridge,
  que en la base funcionaban bien. Con `None` no cambia nada: salta antes el
  `TypeError` de `check_exog_dtypes`. El `TypeError` que nombra la columna
  queda para las categorías que no son números (texto, fechas, booleanos).
- **`DataTypeWarning`** de texto neutro en `fit` y, una vez, en la división
  one-step-ahead; no en `set_in_sample_residuals`.
- **Descartado exigir que la lista incluya todas las `category`**: no
  simplifica esta PR y rompe configuraciones que hoy son correctas. Queda como
  propuesta para todos los forecasters (API-categorical-list-all-category).
- Matriz de 48 escenarios (6 estimadores, 4 tipos de categoría, `None` y lista;
  one-step-ahead frente a un backtesting equivalente), igual con los dos
  pandas: base, 20 coherentes, 13 incoherentes y 15 errores de `fit`; arreglo,
  42, 0 y 6 (los 6, `None` con decimales).

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

**Arreglo con pandas (rama `fix/ms-category-exog-values`, PR #1548).**

- `skforecast/utils/utils.py`: `_category_columns_to_values` (nueva), y
  `cast_catboost_categorical_columns_dataframe` acepta `cat_features` por
  nombre y usa el del constructor del estimador si no viene en `fit_kwargs`.
- `skforecast/recursive/_forecaster_recursive_multiseries.py`: cuatro llamadas
  (`fit`, `set_in_sample_residuals`, `_train_test_split_one_step_ahead` y,
  desde la revisión final, `create_train_X_y`), las columnas `category` fuera
  del `astype` final de `create_predict_X` y docstrings. La división sigue devolviendo 8 elementos y `model_selection`
  no se toca. En total, +164 / -15 líneas en 3 ficheros.
- Tests, escritos antes del arreglo: sin él, 30 fallaban, 10 pasaban (las
  guardas) y los unitarios de la función nueva no se podían importar. Nuevos
  en `test_fit.py`, `test_predict.py`, `test_set_in_sample_residuals.py` y
  `test_train_test_split_one_step_ahead.py` de multiserie, en
  `test_predict_and_calculate_metrics_one_step_ahead_multiseries.py`, en
  `test_cast_catboost_categorical_columns_dataframe.py` y en
  `test_category_columns_to_values.py` (nuevo). Fijan lo que el paso a numpy
  puede romper: `monotonic_cst`, `monotone_constraints` e
  `interaction_constraints` por nombre, y el nivel categórico en un LightGBM
  envuelto con `ordinal_category`.
- Test adaptado: `test_create_predict_X_when_categorical_features_auto_detect_LGBMRegressor`
  usa su parámetro (`'auto'` y la lista) en lugar de `None` y conserva los
  valores esperados: `'auto'` reproduce el modelo anterior.
- Documentación: nota de versión (una línea en "Before upgrading", una entrada
  en Changed y dos en Fixed, sin enlace a la PR), las dos notas y la tabla de
  `categorical-features.ipynb` (celdas de texto, sin reejecutar),
  `llms-base.txt`, el skill `feature-engineering` y los ficheros generados.

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
- Revisado el 2026-11-27 con el arreglo con pandas: la fila de
  BUG-predict-readonly-wrapped-catboost está corregida (los 48 casos que la
  PR 1 con numpy dejaba en ese error siguen fallando en `fit` con el arreglo
  con pandas); el requisito de MAINT-ms-predict-core se mantiene sin cambios
  (el arreglo no toca `predict`, deducido del código).
- Fila nueva API-categorical-list-all-category, y corregidas en la fila del
  bug la lista del cambio de modelo y la frase de los decimales.

## 4. Pendiente

**Para el autor:**

1. Hecho el 2026-11-27: commits, rama subida y PR #1548 abierta contra
   `0.28.x`. Queda revisarla y fusionarla.
2. Hecho el 2026-11-27: enlace a #1548 en las 3 entradas de la nota de
   versión y en la línea de "Before upgrading". Este handoff va en la PR.
3. Al fusionar: mover BUG-ms-fit-dataframe-categorical y las dos filas DOC a
   la sección de hechos del roadmap, y retirar QW-ms-category-warning.
4. Confirmar la regla de las dos puertas antes de la PR de numpy.
5. Decidir la prioridad de MAINT-ms-fit-numpy-estimator (en el roadmap, P2
   como propuesta) y si entra en la 0.28.0.

**Encontrado durante el arreglo (2026-11-27), sin tocar:**

- Lista del cambio de modelo: con `None` también cambian XGBoost (con sus
  parámetros por defecto) e HistGradientBoosting con categorías `0..k`, no
  solo los dos casos que citaba este handoff.
- `CatBoostRegressor(cat_features=[...])` con una lista en el constructor no
  se puede clonar (`RuntimeError` de scikit-learn en el `__init__` de
  cualquier forecaster). Con una tupla funciona. Es ajeno a este cambio.
- LightGBM con `'name:col'` en el constructor y `None`: pasa de resultado
  erróneo en silencio (MAE 31,9) a `LightGBMError` en `fit`.
- Una `category` de decimales fuera de la lista: `fit` y `predict` funcionan,
  pero si la exog futura trae un dtype `category` con otras categorías,
  `predict` lanza el `TypeError` de `check_exog_dtypes`. Igual en
  `ForecasterRecursive` hoy.
- El `DataTypeWarning` sale también en la división one-step-ahead. Medido
  (filtros `default` y `always`): 1 aviso en `fit`; 1 en una búsqueda con
  `OneStepAheadFold` de 3 configuraciones de lags por 2 de parámetros, con y
  sin `return_best`; 1 en una búsqueda con `TimeSeriesFold` sin refit; 8 en un
  backtesting de 8 folds con refit. La repetición es solo la de
  QW-ms-category-warning-per-fold.
- LightGBM con `categorical_feature` en el constructor y `None` queda como en
  `ForecasterRecursive` hoy: por posición funciona (MAE 0,87; en la base,
  58,7) y con `'name:col'` da el mismo `LightGBMError`.
- En la sesión, `python` del `PATH` era el del sistema, sin pandas
  (`SKFORECAST_PYTHON_ENV` sin efecto): se llamó al intérprete de cada
  entorno por su ruta.

**Revisión final (2026-11-27):** dos lecturas del diff (la de la sesión y una
independiente, sin conocer las conclusiones de la primera) y un profiling. Sin
fallos en el arreglo: `fit`, `predict`, `set_in_sample_residuals`, la división
one-step-ahead y el backtesting dan al estimador los mismos valores en todos
los casos probados, con los dos pandas (NaN en la `category`, categorías sin
usar, ordenadas, `Int64` y `uint8`, las cuatro codificaciones, `Pipeline`,
`TransformedTargetRegressor`, exog en diccionario y en formato largo).

- Corregido tras la revisión:
  - La nota de versión y el docstring de la clase decían que el `TypeError`
    que nombra la columna sale siempre. Solo sale con una lista: con `None`
    salta antes el de `check_exog_dtypes`, que no la nombra y que también
    rechaza los decimales (medido con texto, decimales, booleanos y fechas).
  - La frase de `llms-base.txt` y del skill sobre configurar las categóricas
    en el estimador con `None` pasa a condicional, como en la guía. En
    `ForecasterRecursive` solo funcionan las posiciones en `fit_kwargs`
    (medido por la revisión independiente).
  - Nota en el docstring de `create_predict_X` (la `category` sin gestionar
    conserva su dtype), que solo estaba en `create_train_X_y`.
  - Test nuevo
    `test_create_predict_X_output_when_category_exog_not_in_categorical_features`
    (`None` y lista): al adaptar el test anterior, lo que dice la nota de
    versión de `create_predict_X` se había quedado sin test.
- Resuelto después en la misma rama, a petición de Joaquín (2026-11-27),
  con los tests antes (14 casos fallaban sin el cambio):
  - Matrices públicas con una `category` sin gestionar: no reproducían lo
    que recibe el estimador (LightGBM y CatBoost fallaban y XGBoost
    difería en silencio). Ahora `create_train_X_y` y `create_predict_X`
    devuelven la columna con sus valores, como float: una llamada a
    `_category_columns_to_values` al final de cada uno, sin tocar
    `_create_train_X_y` ni `exog_dtypes_out_`. Joaquín reconsideró la
    decisión de no hacerlo solo en multiserie. Los demás forecasters
    siguen devolviendo `category` (API-public-matrices-none-category).
    Cambia también la columna en los predictores que devuelve el
    backtesting (`return_predictors`), que usa `create_predict_X`
    (deducido del código, sin ejecutar ese caso).
  - `cat_features` de CatBoost sobre una columna float con decimales (una
    exog, o un lag): `cast_catboost_categorical_columns_dataframe` lanza
    un `TypeError` que nombra la columna, en lugar de truncar. Medido en
    la base: por posición en `fit_kwargs` truncaba en silencio (0,2, 0,4
    y 0,6 pasaban a 0), por nombre daba un `TypeError` de índices y en el
    constructor, `CatBoostError`. Aplica a las dos vías. Coste de la
    comprobación: en torno a 1 ms por columna categórica con 300 000
    filas y 12 ms con 3 millones (medido sin fijar el núcleo). Límite: si
    los decimales llegan solo en la exog de `predict`, se siguen
    truncando (camino de predicción, sin tocar).
  - `create_predict_X` devolvía NaN donde el estimador recibe un valor que
    no es una categoría del entrenamiento (la exog de `predict` trae la
    columna con otras categorías o como entero): la vuelta al dtype
    `category` de `exog_dtypes_out_` lo perdía, también en la base. Medido
    en 45 casos (4 codificaciones, `None` y lista, categorías enteras y
    decimales, exog en diccionario, con un booleano, NaN), igual con los dos
    pandas: 15 diferían en la base y en la rama, 0 con el arreglo. Arreglo:
    esas columnas quedan fuera del `astype`, que ya no necesita la llamada a
    `_category_columns_to_values`; la condición del `astype` no cambia,
    porque sacarlas también de ella hacía float las columnas enteras (medido).
    Test: el de las matrices públicas de `create_predict_X`, parametrizado
    con un valor no visto como `category` y como entero (4 casos que fallaban
    sin el arreglo). Los demás forecasters tienen el mismo `astype`: anotado
    en API-public-matrices-none-category. Con backtesting y
    `return_predictors` no ocurre (la exog sale de un único DataFrame).
- Riesgos que quedan, sin tocar:
  - Con CatBoost y categóricas gestionadas (`'auto'` o lista), las dos
    matrices públicas dan `CatBoostError` al pasarlas al estimador: esas
    columnas salen como float y CatBoost las quiere enteras. Ya pasa en
    la base (medido en la base y en la rama).
  - Tres reglas para una `category` aceptable: la función nueva admite
    categorías enteras o decimales; `check_exog_dtypes` y la conversión de
    `predict`, solo enteras. Con lista, decimales y un `pd.NA` en la exog de
    `predict`: `TypeError` (también en la base). Va con
    QW-check-exog-dtypes-column-names.
- Profiling (base y rama en el mismo proceso, bloques alternados, núcleo
  fijado; Python 3.13.18, numpy 2.5.5, scikit-learn 1.9.1, lightgbm 4.7.0;
  23 800 y 292 800 filas de entrenamiento, 24 lags, 4 exógenas; script y
  salidas no versionados):
  - Sin nada que convertir (sin `category`, o con `'auto'`): la función
    nueva tarda de 0,1 a 0,7 ms en total, del 0,01 al 0,25 % de `fit`, de la
    división one-step-ahead (dos llamadas) y de `set_in_sample_residuals`.
  - Convirtiendo dos columnas `category` (`None`): de 0,8 a 1,7 ms con
    23 800 filas y de 6 a 12 ms con 292 800; del 0,3 al 0,4 % de `fit` con
    LightGBM y del 1,2 al 1,4 % con Ridge.
  - Base frente a rama: diferencias de signo cambiante, dentro del ruido (con
    pandas 3.1.1, de -4,0 a +3,1 % en los mínimos de la pasada limpia; la
    primera pasada y la de pandas 2.3.3 tienen bloques con más dispersión).
  - Salidas tempranas de `cast_catboost_categorical_columns_dataframe`:
    0,2 µs sin CatBoost y 2,4 µs con CatBoost sin `cat_features` (0,1 µs en
    la base).
  - Alternativa medida y descartada: indexar las categorías ya convertidas
    con los códigos es 2 veces más rápido (1,5 frente a 2,7 ms por columna con
    300 000 filas; 15 frente a 30 ms con 3 millones), pero ahorra en torno
    al 0,2 % de un `fit` con LightGBM y al 1 % con Ridge, y solo en la
    configuración que avisa.

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

Estado del árbol del arreglo con pandas (`fix/ms-category-exog-values`,
2026-11-27, Python 3.13.18, numpy 2.5.5, lightgbm 4.7.0, xgboost 3.4.1,
catboost 1.2.10, scikit-learn 1.9.1):

- `python .claude/skills/verify/affected.py`: 19 ficheros cambiados antes de
  añadir lo de `dev/`.
- Lint (ruff 0.17.6): los ficheros cambiados pasan, salvo
  `skforecast/utils/__init__.py`, con 29 avisos del patrón que ya tenía (28
  F401 y 1 F403; eran 27 y 1, la exportación nueva suma uno).
- Tests afectados (carpetas de multiserie, `utils`, y `tests_utils` y
  `tests_search` de `model_selection`): 3017 passed con pandas 2.3.3; 3014
  passed y 3 skipped con pandas 3.1.1. Sin ejecutar: las carpetas de los demás
  forecasters, `preprocessing`, `stats` y el resto de paquetes (las dos
  funciones cambiadas de `utils` solo las usan multiserie y
  `model_selection/_utils.py`).
- `python tools/ai/generate_ai_context_files.py --check`,
  `python tools/docs/hooks/page_meta.py` y `execute_notebooks.py --check` de
  `categorical-features.ipynb`: en verde. El notebook no se reejecutó (solo
  cambian celdas de texto).
- Tras la revisión final y los dos cambios posteriores (matrices públicas y
  comprobación de enteros en CatBoost): lint igual; tests afectados, ahora
  con las carpetas `tests_validation` de `model_selection` y
  `feature_selection` (usan `create_predict_X` y `create_train_X_y`), 3558
  passed con pandas 2.3.3, y 3555 passed y 3 skipped con pandas 3.1.1; las
  tres comprobaciones de documentación y contexto IA, en verde.
- Antes de los commits de la PR #1548, sobre el árbol con el arreglo de
  `create_predict_X` para los valores no vistos: lint igual; los mismos
  tests afectados, 3562 passed con pandas 2.3.3, y 3558 passed, 3 skipped y
  1 failed con pandas 3.1.1. El fallo es
  `test_bayesian_search_forecaster_warnings_of_folds_in_processes_with_trials_in_threads`
  (`ForecasterRecursive` con Ridge; el intermitente que ya tiene fila en el
  roadmap): repetido solo, 5 de 5. Las tres comprobaciones de documentación
  y contexto IA, en verde.
- Tests nuevos frente a la base: 61 casos. Con el código de `0.28.x`
  (pandas 2.3.3), 43 fallan, los 7 de `test_category_columns_to_values.py`
  no se pueden importar y 11 pasan (las guardas).
- No se ha ejecutado la suite completa, ni Linux, macOS o las versiones
  mínimas de las dependencias.

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
