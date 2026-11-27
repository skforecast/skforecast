# Revisión independiente: categóricas en ForecasterRecursiveMultiSeries (rama `fix/ms-fit-numpy-estimator`)

- **Base:** HEAD `a7c7a869d` (0.28.x). **Rama:** cambios sin commit del árbol de trabajo.
- **Entorno:** pandas 3.1.1, xgboost 3.4.1, lightgbm 4.7.0, catboost 1.2.10, scikit-learn 1.9.1. LightGBM y XGBoost también con pandas 2.3.3: los 222 resultados coinciden con los de pandas 3.
- **Notación:** **[E]** es lo ejecutado (con número); **[D]** es lo deducido leyendo código.

## Resumen

1. **El bug se arregla igual de bien con un cambio pequeño en pandas.**
   - Arreglo mínimo: +79 líneas, unas 40 de código.
   - Rama: +742/-199 líneas de producción y unas 1.400 de tests.
   - Las 78 combinaciones con predicciones erróneas silenciosas pasan a ser correctas con los dos caminos. En 182 de los 186 escenarios en que ambos funcionan, el MAE es el mismo.
2. **Casi toda la complejidad de la rama viene de pasar el estimador a numpy, no del bug.**
   - Al perder nombres y dtypes, hay que reimplementar a mano lo que pandas daba gratis: la configuración del usuario por nombre y el nivel `ordinal_category` como categórico.
   - Además, la rama añade soporte que la base nunca tuvo.
3. **Construir la matriz directamente en numpy es viable**, porque la heterogeneidad se resuelve antes, en el DataFrame de exógenas. Pero no simplifica esa parte difícil. Es un refactor posterior, no un requisito del arreglo.
4. **Recomiendo dos pasos:** arreglo mínimo en pandas ahora y, en un PR aparte, el paso a numpy en el límite con el estimador, con el alcance de las categóricas recortado.
5. **El arreglo mínimo no sale gratis.** Mi primera versión rompía CatBoost con `cat_features` en el constructor, que en la base funcionaba. Lo corregí con 7 líneas. Quedan fallos previos que solo arregla la rama.

---

## Duda 1: ¿es demasiado amplia la gestión de categóricas?

### Respuesta

Sí. Para arreglar el bug basta con que `fit` reciba los valores de la `category`, igual que `predict`.

Lo que más código cuesta es otra cosa:
- traducir la configuración categórica propia del usuario: por nombre, en el constructor o en `fit_kwargs`, con los alias y formatos de texto de LightGBM;
- fusionarla con el nivel cuando hay `ordinal_category` y `categorical_features=None`.

Propongo dos restricciones con poca pérdida (R-A y R-C) y una tercera discutible (R-B).

### Evidencia

**[E] Matriz de 218 escenarios** (`scenarios.py`):
- 7 configuraciones de estimador: LGBM, XGB, HGB, CatBoost, Ridge, `Pipeline(StandardScaler, LGBM)` y `TTR(LGBM)`;
- × 4 encodings, × `categorical_features` en `'auto'`, `None` o una lista sin la columna;
- más 13 configuraciones del usuario con `None`;
- cada una con series en DataFrame ancho y en dict.

La exógena `cat` es `category` con categorías 10/20/30 y efectos de 0, +40 y -40. Un MAE mayor que 10 es un error silencioso.

| árbol | correctos | erróneos silenciosos | errores |
|---|---|---|---|
| base | 100 | **78** | 40 |
| rama | 202 | 0 | 16 |
| mínimo v2 | 186 | 0 | 32 |

- **Los 16 errores comunes a los tres árboles** son artefactos de mi lista `['num']`, que es float: HGB falla por cardinalidad mayor de 255 y Ridge por NaN.
- **Los 16 errores que solo tiene el mínimo** ya existían en la base:
  - CatBoost con `cat_features` por nombre en `fit_kwargs` (8): `TypeError: list indices must be integers`.
  - CatBoost con `ordinal_category` y `None` (2): `CatBoostError` por `_level_skforecast` de tipo `category`.
  - LightGBM con `'name:cat'` en el constructor (6): en la base era un error silencioso; en el mínimo es un error explícito.
- **`'auto'`, el valor por defecto,** da el mismo MAE en los tres árboles.
- **HGB con `None` cambia de modelo con ambos arreglos** (0.644 → 0.621), aunque en la base ya acertaba porque `from_dtype` codifica por valores.
  - Pasa lo mismo con LightGBM y categorías `0,1,2...`.
  - El test de la base `test_create_predict_X_when_categorical_features_auto_detect_LGBMRegressor` (`transformer_exog` que devuelve `category`, con `None`) falla con los dos arreglos. La rama lo cambió a `'auto'`.
  - Es una pérdida de caso de uso común a los dos caminos. Viene de la semántica "valores como numéricos", no de numpy.

**[E] Otros forecasters** (`single_series.py`, sobre la base; la rama no los toca):
- `ForecasterRecursive` y `ForecasterDirect` con `None` y `category` 10/20/30 predicen bien: pasan los valores como numéricos y no avisan.
- La configuración del usuario por posición funciona.
- Por nombre falla: `TypeError: Wrong type(str) or unknown name(cat)` en LGBM y `ValueError ... fitted on data without feature names` en HGB.

**[E] Lo que la base multiseries ya soportaba bien** (`extra.py`, exógena `int64`):
- Funcionaba por nombre: LGBM con `fit_kwargs={'categorical_feature': ['cat']}` (MAE 0.407) y HGB con `categorical_features=['cat']` (0.428). Quitar los nombres sería una regresión para ellos.
- Nunca funcionó: CatBoost por nombre, ni LGBM con `'name:cat'` en el constructor.

**[D] Qué código sirve cada escenario en la rama:**

| Escenario | Código en la rama | Comportamiento en la base |
|---|---|---|
| `'auto'` o lista | `configure_estimator_categorical_features` (ya existía y ya daba posiciones) | Igual. |
| `category` no gestionada: valores, aviso y `TypeError` si las categorías no son enteras | `_train_matrix_to_numpy` (utils.py L903-998) y el aviso en `_get_fit_config` (multiseries L2400-2489) | Error silencioso, o error explícito con CatBoost. |
| CatBoost con columnas enteras | `_train_matrix_to_estimator_input` (L999-1053). Sustituye a `cast_catboost_categorical_columns_dataframe`. | Equivalente. |
| Configuración del usuario por nombre o posición, con `None` | `_resolve_user_categorical_config` (L1054-1247, unas 194 líneas) | Por nombre solo funcionaba con LGBM en `fit_kwargs` y con HGB. |
| Nivel `ordinal_category` añadido a la configuración del usuario, con set/restore de XGB/HGB | `level_name` en la función anterior, `_resolve_estimator_categorical_config` (L2357-2399) y un `try/finally` duplicado en `fit` y en `_predict_and_calculate_metrics_one_step_ahead_multiseries` | Implícito por el dtype `category` con LGBM/XGB/HGB. CatBoost fallaba. |

- **Coste de la parte "configuración del usuario y nivel" [D]:** unas 290 líneas de producción y unas 600 de tests (`test_resolve_user_categorical_config.py` tiene 319, más unos 8 tests de `test_fit.py`). Es aproximadamente la mitad de la rama.
- **Dos ciclos de vida distintos para los mismos parámetros [D]:** los de XGB/HGB son persistentes con `'auto'` o lista, y se fijan y restauran alrededor de `fit` con `None`.

### Restricciones candidatas

- **R-A (recomendada): no fusionar el nivel con la configuración propia del usuario.**
  - Con `None` y `ordinal_category` sin configuración propia, el forecaster configura solo el nivel. Si no lo hiciera, habría regresión: en la base el nivel era categórico por su dtype.
  - Si el usuario sí configuró algo, se lanza un error claro que proponga `categorical_features=[...]`.
  - **Se elimina:** la fusión, el `ValueError` de CatBoost con `cat_features` en el constructor, la comprobación de longitud de `feature_types`, gran parte del set/restore y unos 7 tests.
  - **Se pierde:** una combinación que es nueva en la rama. La lista `categorical_features=[...]` cubre el mismo modelo, salvo en el manejo de categorías desconocidas y NaN.
- **R-C (recomendada): no aceptar formatos de texto ni alias de LightGBM en el constructor** (`'name:...'`, `'0,1'`, `cat_feature`...). Nunca funcionaron en la base. Ahorra unas 25 líneas y 2 o 3 tests.
- **R-B (discutible): aceptar solo posiciones, como `ForecasterRecursive` y `ForecasterDirect`.**
  - Sería una regresión explícita (error, no silencio) para quien usa nombres con LGBM en `fit_kwargs` o con HGB.
  - Las posiciones son frágiles con `lags_grid` y con `onehot`.
  - No la recomiendo, salvo que prime la coherencia con los otros forecasters.
- **No restringir** la semántica "valores numéricos con `DataTypeWarning`": es coherente con `ForecasterRecursive` [E].

### Confianza y qué me haría cambiar de opinión

- **Confianza:** alta en las medidas; media en la atribución de líneas a cada escenario.
- **Cambiaría de opinión si:**
  - aparecen usuarios reales que combinan `None`, `ordinal_category` y configuración propia;
  - o el mantenedor quiere que `None` signifique "categórico por dtype". En ese caso habría que generalizar `level_name` a todas las columnas `category`, y R-A dejaría de tener sentido.

---

## Duda 2: ¿matrices directamente en numpy?

### Respuesta

Es viable: la heterogeneidad no lo impide. Pero no simplifica la parte difícil del cambio. Es un refactor independiente del bug y posterior a él.

### Evidencia [D, salvo donde se indica]

- **Series de distinta longitud e índice.** Se resuelven en `check_preprocess_series` y `align_series_and_exog_multiseries`. Los lags y las window features ya son numpy por serie (L964-1055) y se escriben en un bloque float (L1149-1271).
- **Exógenas por serie.** Pasan por `compat.concat(axis=0)` (unión de columnas con NaN, L1570-1584), `transformer_exog`, el `OrdinalEncoder` por nombre (L1620-1649) y `check_exog_dtypes` (L1651). Es tabular por naturaleza: debería quedarse en pandas.
- **Categorías distintas por serie [E].** Ya se rechazan en la base y en la rama (`TypeError ... different dtypes in different series`, utils.py hacia L5692-5711).
- **Mezcla de dtypes.** Es lo único que hoy obliga a insertar columna a columna (L1273-1299 y la heurística `single_block`, L1149-1190). Bastaría con pasar las exógenas a float tras codificarlas, como ya hacen `ForecasterRecursive` (`_exog_to_numpy`, L929) y `predict` de multiseries (L3051-3083).
- **Partes ligadas al DataFrame:**
  - `create_train_X_y` es pública y devuelve dtypes. Habría que restaurarlos como hace `create_predict_X` (`compat.astype`, L3668).
  - `_get_level_row_slices`, `create_sample_weights` y la extracción de encodings del split leen columnas e índice. Necesitarían metadatos posicionales.
- **Qué simplificaría:**
  - desaparecería `_train_matrix_to_numpy` (96 líneas, que en parte duplican `_exog_to_numpy`);
  - desaparecería la inserción por dtype;
  - se ahorraría una copia completa de la matriz en `fit` cuando alguna exógena no es float64.
- **Qué no simplificaría:** la traducción de nombres a posiciones, el nivel configurado a mano, las columnas enteras de CatBoost y el set/restore de XGB/HGB. Todo eso existe porque el estimador recibe numpy, se construya donde se construya la matriz.
- **Coste y riesgo:**
  - ida y vuelta de dtypes, con enteros por encima de 2^53;
  - cambios en la ruta caliente optimizada en PRs de rendimiento recientes;
  - cambios en varios métodos privados.

No hice ningún prototipo.

### Confianza y qué me haría cambiar de opinión

- **Confianza:** media-alta.
- **Cambiaría de opinión con** un perfil que muestre que la copia extra pesa en casos reales con muchas series. Aun así, seguiría siendo independiente del bug.

---

## Duda 3: ¿pandas ahora y numpy después?

### Respuesta

Sí. El mínimo corrige todos los errores silenciosos medidos con una fracción del cambio y sin tocar interfaces privadas. Lo que añade la rama son mejoras y cambios de comportamiento que merecen su propio PR y su propia nota de versión.

### Evidencia

**Tamaño del cambio [E]:**
- **Rama:**
  - Producción: +742/-199 líneas en 5 ficheros (utils.py +380/-73, multiseries +284/-88, `model_selection/_utils.py` +67/-33, `_search.py` +8/-5, `__init__` +3).
  - Tests: +1.243/-373, más 529 líneas en dos ficheros nuevos. Además, docs y contexto de IA.
  - El tuple privado del split pasa de 8 a 9 elementos.
- **Mínimo:**
  - +79 líneas en 3 ficheros, ver `minimal_fix.diff`.
  - Un helper, `_category_exog_to_values`, llamado desde `fit`, `_train_test_split_one_step_ahead` y `set_in_sample_residuals`.
  - Sin cambios de interfaz. Usa el mismo `DataTypeWarning`.

**Comportamiento [E]:**
- **`OneStepAheadFold` frente a backtesting con `steps=1` y `None`:**

  | Estimador | base | rama | mínimo |
  |---|---|---|---|
  | LGBM | 0.774 frente a 27.78 | 0.77 = 0.77 | 0.77 = 0.77 |
  | XGB | 0.892 frente a 41.06 | 0.851 = 0.851 | 0.851 = 0.851 |

- **`set_in_sample_residuals`** da lo mismo que `fit` en los tres árboles.
- **Tests de la base sobre el mínimo** (carpeta multiseries, los dos ficheros de one-step-ahead y `tests_search`; 1.286 tests):
  - Fallan 2. Son el cambio de modelo de `create_predict_X` con autodetección que la rama también asume.
  - Aviso: en `base028`, `test_predict.py` y `test_set_in_sample_residuals.py` no coinciden con HEAD. En mi copia usé los de HEAD.
- **Tests de la rama sobre el mínimo.** Añadí una capa de 10 líneas, solo en la copia de tests, para la interfaz (tuple de 9 y `feature_names`). Resultado: 886 pasan y 74 fallan.
  - **54 por representación:** exigen X numpy (40 del helper del split y 5 de "recibe float numpy"), detección in-place con indexación numpy o dtypes mixtos (6), y 3 más (`feature_names` obligatorio, dtypes de CatBoost, nivel en un Pipeline).
  - **8 por precisión de diagnósticos:** el aviso sale aunque el usuario configuró la columna (4), y el `TypeError` de categorías no enteras (4).
  - **12 por escenarios que solo cubre la rama:** fusión del nivel (5), restore de parámetros (1), CatBoost por nombre (1), CatBoost con constructor sin el nivel (1), LGBM `'name:'` (2) y nombres con `[ ] <` (2).
  - **Todos los tests de la rama centrados en el bug pasan con el mínimo:** predict, residuos, y equivalencia one-step-ahead frente a backtesting con LGBM, XGB, HGB y XGB con `ordinal_category`, LGBM por nombre y lista sin la columna.

**Agujeros del mínimo [E]:**
- **La v1 rompía CatBoost** con `cat_features` en el constructor y una `category`.
  - En la base funcionaba (MAE 3.52, 2.28 y 4.20). La v1 daba `CatBoostError: Invalid type for cat_feature ... 10.0`.
  - La v2 conserva el dtype entero cuando no hay NaN (+7 líneas) y vuelve a dar 3.52, 2.28 y 4.20.
- **El aviso es impreciso** cuando el usuario configuró la columna.
- **LGBM con `'name:cat'`** pasa de error silencioso a error explícito.
- **No arregla** CatBoost por nombre ni CatBoost con `ordinal_category` y `None`. [D] Ambos se podrían arreglar en pandas con unas 15 líneas.

**Fallos que solo arregla la ruta numpy [E].** Todos ya existían en la base, son explícitos y no forman parte del bug:
- exógena pyarrow con LGBM: `pandas dtypes must be int, float or bool`;
- nombres de exógena o de serie (`onehot`) con `[`, `]` o `<` en LGBM y XGB.

**Cambios visibles solo con la rama:**
- El estimador pierde `feature_names_in_` [E].
- Un `Pipeline` que selecciona columnas por nombre falla en `fit` [E]. En la base ya fallaba en `predict` (`Specifying the columns using strings is only supported for dataframes`), así que no es una regresión real.
- La comprobación in-place es más estricta con exógenas no float [D].
- El nivel pasa a ser numérico con TTR o Pipeline [D; el MAE no cambia en mi matriz].
- El nivel se añade a la configuración del usuario [E].
- `TypeError` con categorías no enteras en una lista [E].

**Trabajo duplicado y riesgo [D]:**
- Con dos pasos se tirarían unas 60 líneas del helper del mínimo y sus tests.
- Los tests de equivalencia valen para los dos caminos, y la rama puede aparcarse y rebasarse como paso 2.
- La rama toca `fit` y las funciones internas de las búsquedas. El mínimo toca 3 puntos y ninguna interfaz.

### Confianza y qué me haría cambiar de opinión

- **Confianza:** alta en el comportamiento; media en la valoración del riesgo.
- **Cambiaría de opinión si** 0.28.0 no sale hasta que estén los dos pasos. En ese caso los usuarios ven un único cambio de todas formas, aunque separar los PRs seguiría ayudando a revisar.

---

## Recomendación global

1. **PR 1 (arreglo):**
   - El mínimo v2: valores con dtype entero conservado, más `DataTypeWarning`, más los tests de equivalencia de la rama.
   - Opcional: las unas 15 líneas de CatBoost (nombres a posiciones, y el nivel en `cat_features`).
   - Nota de versión: el arreglo, y el cambio de modelo con `None` para HGB y para categorías `0..k`.
2. **PR 2 (refactor a numpy en el límite con el estimador):** reutilizar la rama con R-A y R-C aplicadas. Valorarlo por sus méritos: pyarrow, caracteres especiales, coherencia con `ForecasterRecursive` y dirección del proyecto. Su nota va en "Changed".
3. **Más adelante y aparte:** el ensamblado numpy-first, solo si un perfil lo justifica.

Si se prefiere ir a numpy ya: mantener la rama, aplicar R-A y R-C, y separar en commits distintos el arreglo y el cambio de representación.

### Dónde un revisor razonable podría discrepar

- **Dirección del proyecto:** si el destino es numpy, el arreglo en pandas es código que se tirará, y habrá dos notas en lugar de una.
- **El mínimo no es trivial:** necesitó corregir la v1, y la rama cubre mejor los casos límite.
- **Nombres frente a posiciones:** los nombres son valiosos con `lags_grid`, así que se puede defender que la rama hace bien en soportarlos.
- **Semántica de `None`:** "valores como numéricos" rompe el patrón de `transformer_exog` que devuelve `category` con la autodetección de LightGBM, que en la base era correcto con categorías `0..k`. "Valores pero categóricos" lo conservaría, con más código.

---

## Scripts

Todos están en `<scratchpad>/doubts_b/`.

- **Matriz de escenarios:**
  - `scenarios.py`, lanzado con `run_scenarios.sh`. Resultados en `res_<tag>_<familia>.json`.
  - Tags: `base`, `branch`, `minimal`, `minimal2`; con pandas 2, `p2base`, `p2branch`, `p2minimal2`.
  - Análisis: `compare.py`, `summary.py` y `p2p3.py` (222 de 222 iguales entre pandas 2 y 3).
- **Experimentos sueltos:**
  - `single_series.py`: `ForecasterRecursive` y `ForecasterDirect`.
  - `extra.py`, lanzado con `run_extra.sh`: configuración por nombre, categorías por serie, pyarrow, caracteres especiales, residuos y one-step-ahead.
  - `extra2.py`: CatBoost con constructor; salidas v1 y v2.
  - `extra3.py`: Pipeline por nombre y `feature_names_in_`.
- **Árboles y tests:**
  - Árboles: `minimal/` (diff en `minimal_fix.diff`) y `minimal_bt/` (tests de la rama más la capa de interfaz).
  - Tests: `run_tests.sh` y `run_round2.sh`.
  - Salidas: `tests_minimal_bt.txt` (123 fallos sin capa), `tests_minimal_bt_shim.txt` (74 fallos con capa) y `tests_minimal_basetests.txt`.
  - Repetición de 3 ficheros con los tests de HEAD: 2 fallos.

## Qué no comprobé

- Rendimiento y memoria: hay otro revisor en la máquina.
- Forecasters guardados con versiones anteriores (pickle).
- La suite completa, `ForecasterDirectMultiVariate`, el clasificador y la búsqueda bayesiana de principio a fin con categóricas.
- Si en la base LGBM con `categorical_feature` explícito y `ordinal_category` trataba el nivel como categórico (supongo que no).
- Las predicciones exactas con TTR o Pipeline y `ordinal_category`: solo comprobé el MAE.
- XGBoost anterior a 3.x, dependencias mínimas, macOS y Linux.
- Los notebooks de documentación.
- La identidad entre `base028` y HEAD: difieren en 2 ficheros de tests, que sustituí por los de HEAD en mi copia.

