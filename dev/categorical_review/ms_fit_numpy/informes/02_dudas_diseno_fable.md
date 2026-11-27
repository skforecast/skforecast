# Opinión independiente sobre las tres dudas del cambio `fix/ms-fit-numpy-estimator`

Fecha: 2026-11-27. Árbol revisado: rama `fix/ms-fit-numpy-estimator` (cambios sin confirmar sobre `a7c7a869d`).
Entorno: Python 3.13, pandas 3.1.1, numpy 2.5.5, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10 (y pandas 2.3.3 donde se indica).

Convención: **[E]** = lo he ejecutado; **[D]** = lo deduzco leyendo el código.

## Resumen en cinco líneas

1. **Duda 1**: sí, la gestión de categóricas es más amplia de lo necesario. Soporta más que los otros tres forecasters y, en varios puntos, más que la base. Con tres restricciones se elimina `_resolve_user_categorical_config` casi entera (192 líneas, 66 casos de test) perdiendo casos de uso muy estrechos.
2. **Duda 2**: construir en numpy es viable y la heterogeneidad de las series no lo impide. Pero no simplifica la parte cara de este cambio (la configuración de categóricas); solo elimina la conversión en la frontera. Es un paso posterior, no un requisito.
3. **Duda 3**: recomiendo separar. El arreglo mínimo con pandas existe ya como prototipo: 85 líneas añadidas en 3 ficheros, mismas predicciones que la rama en todos los casos del bug, y con una variante pasa los tests de la base sin tocar ninguno.
4. Hallazgo no recogido en las notas de versión: en la rama dejan de funcionar los parámetros del estimador que nombran columnas y no son categóricos (`monotonic_cst` de HistGradientBoosting, `monotone_constraints` e `interaction_constraints` de XGBoost por nombre). En la base funcionaban.
5. Aviso sobre el entorno: la carpeta `base028` no es exactamente HEAD. Dos ficheros de test (`test_predict.py` y `test_set_in_sample_residuals.py` de multiseries) son los de la rama. El código fuente sí coincide. He usado mi propia exportación (`git archive HEAD`) como base.

---

## Duda 1. ¿Es demasiado amplia la estrategia de categóricas?

### Respuesta

Sí. La rama no se limita a reproducir con numpy lo que la base hacía con el DataFrame: añade traducción de nombres a posiciones para cuatro librerías, fusión del identificador de serie con la configuración del usuario, y un mecanismo de poner y restaurar parámetros del estimador alrededor de `fit`. Parte de eso no lo soportaba la base, y nada de eso lo soportan `ForecasterRecursive`, `ForecasterDirect` ni `ForecasterDirectMultiVariate`.

### Evidencia

**Coste en código [E, contado sobre el diff]**

| Pieza | Líneas | Tests |
|---|---|---|
| `_resolve_user_categorical_config` (`utils.py` L1054-1245) | 192 (174 no vacías) | 50 casos en su fichero propio |
| `_resolve_estimator_categorical_config` (multiseries L2357-2398) | 42 | |
| Poner y restaurar parámetros en `fit` (L2666-2689) y en `_predict_and_calculate_metrics_one_step_ahead_multiseries` | unas 45 | |
| Tests de `test_fit.py` dedicados solo a esto (añadir nivel, restaurar, CatBoost en constructor, nombres en constructor, aviso consciente de la configuración) | | 16 casos |
| Total del cambio en código fuente | +742 / -199 en 5 ficheros | 2283 tests recogidos frente a 2176 en las carpetas afectadas |

**Qué soporta cada árbol [E]**. Matriz de 352 escenarios (8 familias de estimador, configuración del usuario en constructor o `fit_kwargs` por nombre o posición, 4 valores de `categorical_features`, 4 `encoding`), ejecutada dos veces (series en DataFrame ancho y en dict de longitudes distintas, mismos recuentos). Cada escenario comprueba en 12 filas que `estimator.predict` sobre la matriz de `fit` coincide con `forecaster.predict`.

| Árbol | Correctos | Inconsistentes (el bug) | Error |
|---|---|---|---|
| Base | 200 | 96 | 56 |
| Rama | 326 | 0 | 26 |
| Arreglo mínimo con pandas | 320 | 0 | 32 |

De los errores, 24 son comunes a los tres árboles: CatBoost con `cat_features` en el constructor y `categorical_features` distinto de `None` (CatBoost rechaza que los de `fit` difieran). Con `'auto'` o lista completa, los tres árboles coinciden en qué columnas trata el modelo como categóricas para LightGBM, XGBoost y HistGradientBoosting sin envolver.

**Comparación con los otros forecasters [E, `others.py`]**. Con `categorical_features=None`, los tres se comportan igual entre sí:

- Configuración del usuario por **posición**: funciona (LightGBM en constructor y `fit_kwargs`, `feature_types` de XGBoost, HistGradientBoosting, `cat_features` de CatBoost en `fit_kwargs`).
- Configuración por **nombre**: falla siempre con el error de la librería (LightGBM, HistGradientBoosting, CatBoost).
- Columna `category` no gestionada: pasa como numérica **sin ningún aviso**. La rama añade un `DataTypeWarning` que los otros no emiten.

**Soporte que la rama añade y la base no tenía [E]**

- Nombres en el constructor de LightGBM. LightGBM los rechaza por sí mismo (`categorical_feature is not a number... add the prefix "name:"`). En la base "funcionaban" solo por accidente: si había columnas `category`, LightGBM descartaba la lista del usuario y usaba todas las `category`. Con columnas enteras, la base da ese mismo error. La rama lo traduce, incluidos 5 alias y el formato `'name:c1,c2'`.
- `cat_features` de CatBoost por nombre en `fit_kwargs` (en la base, `TypeError`).
- CatBoost con `categorical_features=None` y `encoding='ordinal_category'`.
- Fusión del nivel con la configuración del usuario. En la base, con una lista explícita del usuario, el nivel quedaba **numérico** (`lgbm|fit_pos|None|ordinal_category` da `[exog_cat]` en la base y `[nivel, exog_cat]` en la rama). Es un cambio de modelo, no una conservación.

**CatBoost con `cat_features` en el constructor [E]**. Como lista, el forecaster no se puede ni crear: `clone(estimator)` (multiseries L469) lanza `RuntimeError: Cannot clone object`. Ocurre igual en la base. Solo funciona como tupla. La rama dedica a este caso un `ValueError` propio y 9 casos de test.

**Asimetría [E, `byname_params.py`]**. La rama traduce los nombres de las categóricas pero no los demás parámetros por nombre:

| Parámetro por nombre | Base | Rama |
|---|---|---|
| `HistGradientBoostingRegressor(monotonic_cst={'lag_1': 1})` | funciona | `ValueError` en `fit` |
| `XGBRegressor(monotone_constraints={'lag_1': 1})` | funciona | `ValueError` en `fit` |
| `XGBRegressor(interaction_constraints=[['lag_1', 'lag_2'], ...])` | funciona | `ValueError` en `fit` |

Es decir, "el estimador recibe numpy, todo va por posición" ya es la regla de facto en la rama salvo para las categóricas.

### Restricciones candidatas

Las cifras de líneas y casos de esta tabla son estimaciones mías sobre el diff, no un recuento tras implementarlas.

| | Restricción | Qué elimina | Caso de uso que se pierde |
|---|---|---|---|
| R1 | Con `None` + `ordinal_category`, el forecaster configura solo el nivel con la función existente `configure_estimator_categorical_features`, que sustituye la configuración del usuario con el `IgnoredArgumentWarning` actual | Parámetro `level_name`, la fusión en 4 bloques, el `ValueError` de CatBoost, casi todo el bloque de XGBoost: unas 35 líneas y unos 30 casos de test | Tener a la vez nivel categórico y categóricas propias configuradas a mano. Alternativa equivalente: `categorical_features=[...]` |
| R2 | Solo posiciones para la configuración del usuario (como los otros tres forecasters) | `to_indices`, el análisis de textos y alias de LightGBM, las ramas `has_names`: unas 30 líneas y unos 20 casos | Nombres que la base sí aceptaba en multiseries: `fit_kwargs` de LightGBM, HistGradientBoosting, tupla de CatBoost. Fallan con error explícito de la librería, no en silencio |
| R3 | Aviso que no depende de la configuración del usuario (texto neutro), o ninguno como en los otros forecasters | La necesidad de leer la configuración de cada librería solo para decidir si avisar | Un aviso algo ruidoso para quien declaró la columna por posición |

Con R1 + R2 + R3 desaparecen `_resolve_user_categorical_config`, `_resolve_estimator_categorical_config` y los dos bloques de poner y restaurar: unas 280 de las 742 líneas añadidas y unos 66 casos de test. Lo único que queda por leer es `cat_features` de CatBoost para el cast a entero, que ya hace `cast_catboost_categorical_columns` en los otros forecasters.

R1 es la que más quita y menos cuesta. R2 es la discutible. Alternativa intermedia a R2: mantener los nombres solo donde la librería los acepta de forma documentada (`fit_kwargs` de LightGBM, HistGradientBoosting, CatBoost) y quitar constructor de LightGBM, alias y `'name:'`.

### Regresiones de la rama frente a la base que conviene decidir a conciencia [E]

- **Estimador envuelto + `ordinal_category`, con la configuración por defecto** (`'auto'`): `TransformedTargetRegressor(LGBMRegressor())` o un `Pipeline` trataban el nivel como categórico por su dtype; en la rama es numérico. Está en las notas de versión.
- **Patrón antiguo**: `transformer_exog` que devuelve columnas `category` con códigos 0, 1, 2 y `categorical_features=None`. En la base el modelo era correcto y categórico; en la rama es numérico con aviso. La rama cambió el test `test_create_predict_X_when_categorical_features_auto_detect_LGBMRegressor` de `None` a `'auto'` para adaptarlo. Una búsqueda de `astype('category')` en `docs/` no devuelve nada.
- Los parámetros por nombre de la tabla anterior (no documentado).

### Confianza

Alta en los hechos (ejecutados). Media en la valoración de cuánta gente usa cada caso: no tengo datos de uso, solo que `'auto'` es el valor por defecto y que todos los casos afectados exigen escribir `categorical_features=None` a propósito.

### Qué me haría cambiar de opinión

- Evidencia de uso real (issues, foro) de configuración por nombre con `None` en multiseries: entonces R2 no compensa.
- Que se quiera dar el mismo soporte de nombres a los otros tres forecasters: entonces la función es una inversión compartida y no un coste solo de multiseries.

---

## Duda 2. ¿Sería más sencillo crear las matrices directamente en numpy?

### Respuesta

Es viable, y la heterogeneidad (longitudes, exógenas por serie, dtypes) no es el obstáculo. Pero simplificaría poco de **este** cambio: quita la conversión en la frontera, no la gestión de categóricas. Es un paso posterior e independiente del arreglo del bug.

### Evidencia

**Lo que ya es numpy [D]**. `_assemble_train_X_y` (L1109-1371) reserva un único bloque float en orden F y escribe en él lags, window features, nivel (`ordinal`, `onehot`, `None`), exógenas float64 y calendario. Solo se insertan como columnas de pandas las exógenas que no son float64 y el nivel de `ordinal_category` (L1291-1318), "para conservar su dtype". `_create_train_X_y_single_series` es numpy puro.

**Lo que depende de pandas y por qué [D]**

| Paso | Dónde | Por qué necesita pandas | ¿Evitable? |
|---|---|---|---|
| Entrada, alineación por índice, recorte de NaN | `check_preprocess_series`, `align_series_and_exog_multiseries` | Índices de fechas, `reindex` | No |
| Unir las exógenas de todas las series | `compat.concat(..., axis=0)` L1603 | Unión de columnas cuando una serie no tiene alguna, relleno con NaN y cambio de dtype | No de forma razonable |
| `transformer_exog` | L1626 | `ColumnTransformer` selecciona por nombre | No |
| Detección `'auto'` y `OrdinalEncoder` | L1639-1668 | Decide por dtype; codifica texto | No |
| Columnas no float y nivel `category` | L1291-1318 | Solo para conservar el dtype en la matriz | **Sí** |
| Método público `create_train_X_y` | L1789 | Devuelve DataFrame con dtypes | Se reconstruye, como hace `ForecasterRecursive` (L1094-1110) |

La heterogeneidad se resuelve antes del ensamblado. Las categorías distintas por serie ni siquiera llegan: `check_preprocess_exog_multiseries` (L5687-5715) ya exige el mismo dtype, categorías incluidas, y lanza `TypeError`.

**Prototipo [E, `roundtrip.py`]**. Convertir la matriz a float64 y reconstruir el DataFrame con los dtypes guardados da un resultado idéntico (`assert_frame_equal`) en 24 combinaciones por versión de pandas (3.1.1 y 2.3.3): 4 `encoding`, `'auto'` y `None`, tres casos de heterogeneidad (todas con exog, una serie sin exog, una serie sin algunas columnas), con columnas float32, int8, int64, bool, `Int64` con NA, `boolean`, pyarrow y `category` de enteros. El método público podría devolver lo mismo que hoy.

**Qué simplificaría en este cambio [E, `boundary_copy.py`]**. La conversión de la frontera es gratis si todo es float64 y copia la matriz entera en cuanto una columna no lo es:

| Configuración (matriz de 42,7 MB) | Memoria nueva en `_train_matrix_to_numpy` |
|---|---|
| `ordinal`, exog float (por defecto) | 0 MB |
| `ordinal`, exog de texto con `'auto'` | 0 MB |
| `ordinal`, una exog int | 42,7 MB |
| `ordinal`, una exog bool | 42,7 MB |
| `ordinal_category`, exog float | 42,7 MB |

Una construcción numpy eliminaría esa copia y dejaría `_train_matrix_to_numpy` (95 líneas) en casi nada. No he medido si en la base el estimador hace una copia equivalente al recibir el DataFrame, así que no afirmo que la rama use más memoria que la base.

**Qué no simplificaría [D]**. `_resolve_user_categorical_config` existe porque el estimador recibe numpy, no por dónde se crea ese numpy. Seguiría igual.

**Coste y riesgo [D]**. Tocaría unos 10 métodos que hoy indexan `X_train` por nombre de columna: `_get_level_row_slices`, `create_sample_weights`, `_train_test_split_one_step_ahead`, `fit`, `set_in_sample_residuals`, el filtrado de NaN. Casos límite: enteros por encima de 2^53 pierden precisión (ya documentado en `_exog_to_numpy`); una columna bool con ausentes queda como `object` y volvería como 1.0/0.0 en lugar de True/False (`assert_frame_equal` no lo distingue); y una columna no convertible tendría que dar error al crear la matriz en lugar de en `fit`.

### Confianza

Alta en la viabilidad (prototipo) y en qué depende de pandas. Media en el coste: no lo he implementado.

### Qué me haría cambiar de opinión

Un perfil que muestre que la copia de la frontera pesa en casos reales con muchas series y `ordinal_category`. Entonces subiría de prioridad, pero seguiría siendo un PR aparte.

---

## Duda 3. ¿Arreglar ahora con pandas y dejar numpy para después?

### Respuesta

Sí, lo recomiendo. La comparación ya no es asimétrica: he implementado el arreglo mínimo sobre una copia de la base y lo he medido.

### El arreglo mínimo, medido [E]

Un helper `_category_columns_to_values` que sustituye por sus valores (float) las exógenas `category` no gestionadas, llamado en `fit`, `set_in_sample_residuals` y `_train_test_split_one_step_ahead`, más 9 líneas para que el cast de CatBoost acepte nombres y `cat_features` del constructor.

| | Arreglo mínimo | Rama |
|---|---|---|
| Código fuente | +85 / -3 en 3 ficheros | +742 / -199 en 5 ficheros |
| Escenarios inconsistentes (de 352) | 0 | 0 |
| Tests de la base, pandas 3 (multiseries + 2 ficheros one-step-ahead + 2 de utils) | 938 pasan, 2 fallan | no aplica (cambia 373 líneas de tests) |
| Tests de la base, pandas 2.3.3 (multiseries + 1 fichero one-step-ahead) | 904 pasan, 2 fallan (los mismos) | |
| Variante que no convierte si los códigos ya son los valores (multiseries + 2 ficheros one-step-ahead, pandas 3) | **915 pasan, 0 fallan** | |

Los 2 fallos son el test del patrón antiguo, cuya expectativa la rama también tuvo que cambiar. La variante (una condición más) deja intacto todo lo que esos tests cubren.

**Mismas predicciones que la rama en los casos del bug [E, `cases.py`]**

| Caso (`None`, categorías 10/20/30) | Base | Rama | Mínimo |
|---|---|---|---|
| LightGBM, MAE a 20 pasos | 6,40 | 0,91 | 0,91 |
| XGBoost `enable_categorical`, MAE | 4,93 | 0,19 | 0,19 |
| HistGradientBoosting, MAE | 0,90 | 0,90 | 0,90 |
| LightGBM, one-step-ahead frente a backtesting | 0,91 frente a 6,35 | iguales | iguales |
| CatBoost `fit_kwargs` por posición, one-step frente a backtesting | 0,50 frente a 6,90 | iguales | iguales |
| XGBoost `feature_types` por posición | distintos | iguales | iguales |

También coinciden rama y mínimo en `set_in_sample_residuals` (diferencia 0 con los residuos de `fit`), en series como dict con una serie sin exog o sin una columna, y en `predict_interval`.

**Agujeros propios del arreglo mínimo [E]**

- Primera versión: rompía CatBoost con `cat_features` en el constructor (tupla) y exog `category`, que en la base funcionaba (MAE 0,74). Corregido con las 9 líneas citadas; verificado.
- El `DataTypeWarning` no sabe si el usuario declaró la columna en el estimador y avisa igualmente. Se arregla con un texto neutro.
- LightGBM con nombres en el constructor pasa de "resultado incorrecto en silencio" a `LightGBMError` (3 escenarios). Lo considero mejora.
- No arregla CatBoost + `None` + `ordinal_category` si el usuario no incluye el nivel (fallo que ya tenía la base, con mensaje claro de CatBoost).
- HistGradientBoosting con categorías 10/20/30 y `None` pasa de categórico a numérico, igual que en la rama.

**Lo que solo arregla el camino numpy [E]**

| Fallo en `fit` | Base | Mínimo | Rama |
|---|---|---|---|
| LightGBM con exog de dtype pyarrow | error | error | funciona |
| LightGBM con `[`, `]`, `<` o caracteres JSON en nombres de serie (`onehot`) o de exog | error | error | funciona |
| XGBoost con `[`, `]`, `<` en nombres | error | error | funciona |
| CatBoost + `None` + `ordinal_category` | error | error | funciona |

Son fallos ruidosos, no predicciones erróneas en silencio. No tienen la urgencia del bug.

### Cambios visibles de la rama que el arreglo mínimo no introduce [E salvo indicación]

1. El estimador pierde los nombres: `feature_names_in_` desaparece, LightGBM da `Column_0...`, XGBoost `None`. `forecaster.estimator.predict(X_train)` con el DataFrame de `create_train_X_y` emite el aviso de scikit-learn. `get_feature_importances` sigue dando nombres.
2. Nivel numérico en estimadores envueltos con `ordinal_category`, también con `'auto'`.
3. Con `None` + `ordinal_category` + configuración propia, el nivel pasa a categórico (cambia el modelo).
4. `LinearRegression(copy_X=False)` con una exog entera: funcionaba, ahora `ValueError`.
5. Parámetros por nombre no categóricos (restricciones monótonas y de interacción): funcionaban, ahora fallan. **No está en las notas de versión.**
6. "Forecasters guardados deben reajustarse": no lo he comprobado.

Afirmaciones de las notas de versión que he verificado: predicciones erróneas en LightGBM, XGBoost y CatBoost; one-step-ahead ocultaba el bug; los tres fallos de `fit` arreglados; HistGradientBoosting daba predicciones correctas en la base; con `'auto'` las predicciones y métricas coinciden entre base y rama en los casos que probé (LightGBM, XGBoost, HistGradientBoosting, CatBoost).

Una que matizo: "un `Pipeline` que selecciona columnas por nombre falla en `fit`". En la base ya fallaba, pero en `predict`. No es una pérdida; la rama falla antes.

### Trabajo duplicado o desechado con dos pasos [D]

- Se desecha el helper (unas 65 líneas) y sus 3 llamadas cuando llegue numpy.
- Los tests de comportamiento de la rama (predicción con `category` no gestionada, residuos, aviso) no dependen de la representación y ya pasan contra el arreglo mínimo. El de equivalencia con backtesting necesita adaptar la firma interna.
- Todo lo de numpy de la rama (`_train_matrix_to_numpy`, `_train_matrix_to_estimator_input`, la comprobación in situ para arrays, el paso de nombres en `model_selection`) se reutiliza tal cual en el segundo PR.

### Riesgo para la versión

De los 123 tests de la rama que fallan contra el arreglo mínimo (primera versión), 86 fallan solo por el contrato interno: 74 por la tupla de 9 elementos y 12 por el argumento `feature_names`. Los otros 37 mezclan aserciones específicas de numpy y comportamientos que la rama añade a propósito. Eso mide cuánto contrato interno cambia la rama en un PR que nace como arreglo.

### Confianza

Alta en que el arreglo mínimo corrige el bug con el mismo resultado numérico. Media-alta en la recomendación de separar: depende de un juicio de proceso, no solo de hechos.

### Qué me haría cambiar de opinión

- Que los dos pasos vayan a entrar seguro en la misma 0.28.0 y no se quiera revisar dos veces: el usuario vería lo mismo y la separación solo aporta revisión más limpia y capacidad de bisección.
- Que aparezca otro desajuste entre `fit` con DataFrame y `predict` con numpy distinto del de `category`. He probado bool, `Int64` con NA y pyarrow sin encontrar ninguno, pero no es una prueba exhaustiva. Ese es el argumento más fuerte a favor de la rama: quita la asimetría de raíz.

---

## Recomendación global

1. **Ahora**: PR pequeño con el arreglo mínimo con pandas. Elegir variante según el calendario:
   - si la migración a numpy puede irse a otra versión, la variante que no convierte cuando códigos y valores coinciden (ningún test de la base cambia);
   - si las dos cosas entran en la misma versión, la variante simple, que ya deja el mismo comportamiento que tendrá numpy (un solo cambio para el usuario).
2. **Después, PR propio de migración a numpy** partiendo de esta rama, con R1 y R3 aplicadas y R2 decidida expresamente. Añadir a las notas de versión los parámetros por nombre no categóricos.
3. **Más adelante, si se mide que compensa**: construcción numpy de la matriz (duda 2).

Si se prefiere fusionar la rama tal cual, lo mínimo que haría antes: aplicar R1, documentar el punto 5 de la lista de cambios visibles, y decidir a conciencia el caso de estimadores envueltos con `ordinal_category`.

### Dónde un revisor razonable podría discrepar

- **La rama ya está hecha y pasa sus tests** (1047 en los ficheros afectados, ejecutado sobre una copia). Separar es trabajo adicional y un rebase con conflictos en `fit`, el split y `set_in_sample_residuals`.
- **Arreglo estructural frente a parche**: la rama elimina la clase entera de errores "el estimador ve cosas distintas en `fit` y `predict`"; el mínimo solo el caso conocido.
- **Dirección del proyecto hacia numpy**: la rama avanza en ella; el arreglo mínimo añade 85 líneas de pandas que luego se quitan.
- **R2**: los nombres son más cómodos y la base los aceptaba en multiseries. Mantenerlos cuesta unas 30 líneas.
- **La variante dependiente de los valores** (`[0,1,2]` sigue categórica, `[10,20,30]` pasa a numérica) es la menos invasiva pero la más difícil de explicar.
- **El aviso**: yo lo dejaría neutro; también es defendible no avisar, como los otros forecasters, o avisar en los cuatro.

---

## Scripts ejecutados

Carpeta: `<scratchpad>/doubts_a`

| Script | Qué hace |
|---|---|
| `scenarios.py`, `run_scenarios.sh`, `summarize.py` | Matriz de escenarios por árbol y comparación. Resultados en `out/*_wide.json`, `out/*_dict.json`, `out/*_wide_int.json`; resúmenes en `out/summary_wide.md`, `out/summary_dict.md`, `out/summary_wide_int.md` |
| `patch_scen.py`, `patch_scen2.py` | Ajustes de `scenarios.py` (comprobación en 12 filas, tuplas de CatBoost, exógenas enteras) |
| `others.py` | Los otros tres forecasters en las mismas situaciones (`out/others_branch.log`) |
| `cases.py`, `cmp_cases.py`, `show_d.py` | Casos concretos: fallos de `fit`, nombres, residuos, one-step-ahead, dict, MAE (`out/cases_*.json`) |
| `apply_minimal.py`, `apply_minimal2.py` | Arreglo mínimo sobre la copia de la base (`minimal/`); diff en `out/minimal_vs_base.diff` |
| `apply_minimal3.py` | Variante que conserva `category` si códigos y valores coinciden (`minimal3/`) |
| `dbg_catboost.py`, `dbg_catboost2.py` | CatBoost con `cat_features` en constructor y en `fit_kwargs` |
| `dbg_e.py` | Dónde falla el `Pipeline` que selecciona por nombre |
| `byname_params.py` | Parámetros por nombre no categóricos |
| `boundary_copy.py` | Memoria de la conversión en la frontera |
| `roundtrip.py` | Prototipo de ida y vuelta float64 a DataFrame con dtypes |

Árboles en la misma carpeta: `base_head/` (exportación de HEAD), `minimal/` (arreglo mínimo final), `minimal_bt/` (primera versión del arreglo + tests de la rama), `minimal3/` (variante), `branch_copy/` (copia de la rama para ejecutar sus tests fuera del repositorio).

Registros de pytest en `out/`: `tests_minimal2_basetests.log`, `tests_minimal2_basetests_pandas2.log`, `tests_minimal3_basetests.log`, `tests_minimal_branchtests.log`, `tests_branch_branchtests.log`. La carpeta `out_v1/` contiene una primera pasada de la matriz con una comprobación de una sola fila, que infravaloraba el bug (64 inconsistentes en lugar de 96); está superada. `out_m1/` guarda los resultados de CatBoost de la primera versión del arreglo mínimo.

Incidencia durante el trabajo: detuve cuatro procesos míos (una pasada de `scenarios.py` en segundo plano) para relanzar la matriz con la comprobación corregida. No toqué procesos ajenos ni nada dentro del repositorio.

## Lo que no he comprobado

- Tiempos de ejecución (otra revisión compartía la máquina). La única medida de memoria es la de `boundary_copy.py`, de la conversión aislada.
- Linux y macOS.
- Forecasters guardados con una versión anterior y cargados en la rama.
- La rama con pandas 2 (solo el arreglo mínimo y el prototipo de ida y vuelta).
- Los tests de la rama contra la versión final del arreglo mínimo: los 123 fallos son de la primera versión, antes del parche de CatBoost.
- La variante del arreglo mínimo con pandas 2 y con la matriz completa de escenarios (solo LightGBM, formato ancho).
- Búsquedas bayesianas en paralelo con el poner y restaurar parámetros de XGBoost y HistGradientBoosting.
- Estimadores clasificadores, `ForecasterRecursiveClassifier`, exog en formato largo, `select_features_multiseries`.
- Los notebooks de documentación y los ficheros de contexto de IA modificados.
- Si el arreglo mínimo se puede llevar a 0.27.x.
- No es una revisión de código de la rama en busca de bugs: he leído el diff y ejecutado sus tests, nada más.
- No he leído `dev/`, los planes ni las otras carpetas del scratchpad (salvo `base028`).

