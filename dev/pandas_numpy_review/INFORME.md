# Estudio: uso de pandas frente a numpy dentro de skforecast

Fecha: 2026-11-19. Árbol medido: rama `perf/multiseries-predict-backtesting` en `ab62227` (es `0.28.x` más los cambios 1, 2, 4, 6 y 7 de PERF-multiseries-names-pyarrow). No se ha tocado nada de `skforecast/`.

Entornos (todas las medidas, salvo que se diga otra cosa): Python 3.12.3, numpy 2.5.5, scikit-learn 1.9.1, numba 0.68.1, LightGBM 4.7.0, `OMP_NUM_THREADS=1`, máquina de 4 núcleos compartida (ruido de varios puntos; los subagentes midieron a la vez). Tres entornos de pandas: **p2** = pandas 2.3.3; **p3** = pandas 3.1.1 con la importación de `pyarrow` bloqueada; **p3 pa** = pandas 3.1.1 con `pyarrow` 26.0.1.

Scripts y resultados en [scripts/](scripts/) (ver [scripts/README.md](scripts/README.md)).

**Revisado en [REVISION.md](REVISION.md)** (revisión independiente, dos rondas). Donde difieran vale la revisión: en particular la ganancia alcanzable de §5.1 y de P3, el alcance del bug P1 y la decisión D-NP2, las condiciones de P2, y las propuestas P4, P6, P11, P14 y P16. Las filas de §9 se han llevado a `dev/ROADMAP.md` ya corregidas.

## Resumen

**Veredicto.** La filosofía "pandas en los bordes, numpy dentro" es correcta y, donde se aplica, funciona: el bucle recursivo de `predict` es numpy puro en todos los forecasters y el estimador recibe arrays (salvo en el `fit` de multiserie). Lo que falla es **dónde está el borde**. Hoy es cada llamada pública, y el backtesting y las búsquedas llaman a los métodos públicos en cada fold. En multiserie, además, cada fold crea objetos de pandas **por serie**. Por eso el coste de pandas escala con folds × series y el modelo es una parte pequeña del tiempo:

| Backtesting multiserie sin refit, 50 folds de 24 pasos (ms por fold, p3 pa) | Hoy | Bucle en numpy (techo) | Solo el modelo |
|---|---|---|---|
| Ridge, 100 series, exog mixta (float, int, `category`, texto) | 35,5 | 0,24 | 0,05 |
| Ridge, 600 series, exog mixta | 248 | 0,61 | 0,15 |
| LightGBM, 100 series, exog mixta | 40,1 | 4,2 | 2,7 |
| LightGBM, 600 series, exog mixta | 256 | 17,9 | 16,8 |

Predicciones idénticas bit a bit en los 36 casos medidos (§5.1). El techo no valida ni avisa por fold; lo alcanzable estará entre las dos columnas, pero el margen es de uno a dos órdenes de magnitud con modelos baratos y de 3 a 14 veces con LightGBM.

**Las categóricas no son lo que obliga a mantener pandas.** Tras el `OrdinalEncoder` todo es float, y LightGBM, CatBoost, XGBoost y HistGradientBoosting reciben las categóricas por índice de columna, no por dtype (§4.1). Lo que retiene pandas en multiserie es la estructura "un DataFrame por serie y por fold", la validación con sus avisos, `transformer_exog` y el refit. Codificar la exog una sola vez antes del bucle de folds cuesta 47 ms con 100 series (un `pd.concat` y una llamada al codificador); hoy se pagan 35 ms en cada fold.

**Cinco propuestas principales** (detalle en §5, filas del roadmap en §9):

1. **Bug de corrección en multiserie (nuevo).** El estimador recibe un DataFrame en `fit` y numpy en `predict`. Con LightGBM, `categorical_features=None` y una exog `category` de enteros, `predict` da −50 para las tres categorías cuando lo correcto es −50, 0 y +50 (reproducido en p2 y p3; `ForecasterRecursive` acierta). Arreglo: dar numpy al estimador en `fit`, como ya hacen los otros forecasters; mismo modelo en los 9 casos medidos y 30 a 60 ms menos por `fit` con Ridge. S a M.
2. **Exog a float de una vez en los forecasters de una serie y en el directo multivariante (nuevo; rendimiento y bug).** Una columna `bool`, `Int64`, `Float64` o numérica de `pyarrow` convierte `X_train` en un array `object`: el `fit` de `ForecasterRecursive` con Ridge pasa de 29 a 231 ms (100.000 filas) y el pico de memoria de 43 a 126 MB, con el mismo resultado. Y con NA en una columna nullable, `ForecasterRecursive` y `ForecasterDirect` fallan en `fit` con un `TypeError` de numpy incluso con LightGBM. `to_numpy(dtype=float, na_value=np.nan)` cuando todas las columnas son numéricas o `bool`, como ya hacen los adaptadores foundation. S.
3. **Backtesting multiserie sin refit con un camino interno en numpy (amplía PERF-predict-numpy y PERF-backtesting-folds-positions).** Preparar una vez la matriz de series y la exog codificada; en cada fold, cortes por posición y `_recursive_predict`. Es el cambio de mayor ganancia de todo el estudio (tabla de arriba). L. Recomiendo hacerlo **antes** que la versión univariante que propone P3 §11.2: en una serie el fold ya cuesta 0,3 a 0,8 ms; en multiserie, de 10 a 250 ms.
4. **Codificar las categóricas por códigos (PERF-categorical-codes, ya en el roadmap), y una sola vez por tabla.** En `predict` de `ForecasterRecursive` el codificador cuesta ~1 ms de 1,45 ms; en `fit` multiserie, 21 ms de 99 ms con 100 series en p3. En multiserie conviene codificar la tabla concatenada, no serie a serie: con 600 series, 343 ms frente a 661 ms con la búsqueda por códigos serie a serie y 1223 ms con el codificador serie a serie.
5. **Validar una vez por backtesting y sacar el índice de `y` por posición en los univariantes (bloque B de PERF-predict-numpy, ya en el roadmap).** −17 a −31 % en backtesting con Ridge y exog float; −68 % con índice diario con zona horaria, por `_is_utc_anchored_index` en cada `expand_index` (0,4 ms por llamada).

Además hay una docena de cambios pequeños y baratos (§5.2), entre ellos uno que quita el 40 % del `predict` de `ForecasterDirectMultiVariate` (P16), y varias incoherencias entre forecasters (§3), tres de ellas con resultados o errores distintos (`ForecasterEquivalentDate` con NaN según el tipo de `offset`; el bug 1; la exog nullable con NA).

## 1. Método

- Inventario de la fase 1 repartido en tres subagentes (familia recursiva, familia directa, multiserie) con perfiles cProfile y cronómetros por etapa. Las conclusiones que entran en este informe se comprobaron con scripts propios (los de [scripts/](scripts/)); se marca lo no verificado.
- Fase 2 y 3: prototipos aplicados en memoria (sustituyendo funciones o llamando a métodos privados), nunca en el paquete. Cada prototipo compara su salida con la del código actual (`np.array_equal`).
- Se partió del trabajo previo y no se repitió: `dev/multiseries_predict_perf/` (MSPERF: PLAN, SEGUIMIENTO, MEDIDAS), `origin/0.28.x:dev/backtesting_multiseries_speedups/PLAN.md` (BTPLAN), `dev/pandas3_migration/INFORME.md` §11 (P3) y `dev/preprocessing_speedups/PLAN.md` (PREP). Cada propuesta indica su relación con ellos.

## 2. Fase 1: inventario por forecaster

"Frontera" es el punto donde los datos pasan a numpy. "Bucle" significa que la operación se repite por fold, por paso o por serie; "una vez", una vez por llamada pública. Líneas de `ab62227`.

### 2.1 Familia recursiva de una serie

| Forecaster / camino | Frontera | pandas que queda después | Por qué | Coste medido (3000 puntos diarios, lags 24, steps 24, p3 pa) |
|---|---|---|---|---|
| `ForecasterRecursive.fit` | `y`: `check_extract_values_and_index` (`_forecaster_recursive.py:827`); exog: `exog.to_numpy()` (`:921`) | Una vez: `input_to_frame` de `y`, `transform_dataframe`, `get_exog_dtypes` ×2, detección de categóricas por dtype, `OrdinalEncoder` con salida pandas (`:903-913`), `pd.Series` para las window features (`:955`), `last_window_` y `training_range_` | Transformador de `y` ajustado con DataFrame (nombres), dtypes para detectar categóricas, contrato de `transform_batch` | `_create_train_X_y`: 0,14 ms sin exog, 0,35 con exog float, 1,83 con una `category`; `Ridge.fit` 0,75 a 0,9 ms. Solo importa con refit y modelos baratos |
| `ForecasterRecursive.predict` | ventana `:1605`; exog `:1659` (convierte toda la exog antes de cortar a `steps`) | Una vez: `check_predict_input` (isna, columnas, índices), `columns.tolist()` ×2 y `dtypes.tolist()` (~22 µs), `expand_index` con `date_range`, `pd.Series` de salida; con categóricas, el codificador (~1 ms). **Nada dentro del bucle recursivo** (`:1769-1803`) | Índice de salida, mensajes de error, nombres de columnas, codificación | Ridge: 0,16 ms sin exog, 0,30 con exog float (0,22 sin validar), 1,45 con categórica; la recursión son 0,07 ms. LightGBM: 0,91 / 1,1 / 2,25 ms |
| `predict_bootstrapping`, `predict_quantiles`, `predict_interval`, `predict_dist` | como `predict` | DataFrame de 250 columnas (`:2307`), `DataFrame.quantile(axis=1)` (`:2574`, `:2683`), `pd.concat`, `DataFrame.apply` de la distribución. El intervalo por bootstrapping prepara las entradas y recorre la recursión dos veces (`:2555-2572`) | Salidas | Ridge: 1,3 / 2,0 / 2,6 ms, pandas ~55 % |
| Conformal | numpy | `np.quantile` por bin y `np.vectorize` en cada llamada (`:2412-2416`) | Código numpy sin cachear | 1,12 ms frente a 0,36 ms de `predict` |
| `set_out_sample_residuals` | numpy en la entrada | `pd.DataFrame(...).dropna()` y `groupby('bin').apply(np.array)` (`:3143-3150`) | Histórico | 1,47 ms; en numpy 0,013 ms |
| Backtesting sin refit (`_validation.py`) | dentro de `predict`, en cada fold | Bucle: `y.iloc`/`exog.iloc` (`:96-118`), `check_predict_input` completo (`check_inputs=True`, `:457`), `pd.Series` de salida. Una vez: `pd.concat`, métricas con Series | El forecaster solo tiene la API pública con pandas | 50 folds, Ridge: 13,6 / 24,5 / ~90 ms (sin exog / float / categórica); pandas 63 % con exog float. LightGBM con exog float: pandas 16 % |
| Búsqueda `OneStepAheadFold` | `_create_train_X_y` dos veces por valor de lags | Por candidato: `get_indexer`, `estimator.predict` | | Ridge: 2,1 ms por candidato, pandas 26 a 37 % |
| `ForecasterRecursiveClassifier` | como `ForecasterRecursive` | Además: `(exog_index == y_index).all()` en vez de `.equals` (`:931`, 13 µs frente a 1), `encoder.transform` de la ventana (137 µs). **`estimator.predict_proba` en cada paso sin camino rápido** (`:1740-1746`) | | LGBMClassifier: 11 ms por `predict` (620 µs por paso; el booster, 28 µs). pandas 5 a 15 %: el problema no es pandas |
| `ForecasterEquivalentDate`, `offset` entero | `:640` | `expand_index`, `pd.Series`, `np.apply_along_axis` | | 0,11 a 0,24 ms |
| `ForecasterEquivalentDate`, `DateOffset` | `:694` | Bucle Python por fecha con aritmética de `Timestamp` (`:677-684`), `reindex`, `isnull` ×2, `DataFrame.apply(axis=1)` (`:727`) | Offsets de calendario | 2,1 a 3,3 ms; backtesting 115 a 132 ms, pandas 71 a 76 % |
| `ForecasterEquivalentDate.fit` con bins | `:515` | `shift`, `loc`, `concat`, `groupby`; `np.apply_along_axis(agg_func)` por fila (`:526`) | | 11 ms, 8,4 de ellos en `apply_along_axis` (vectorizado: 0,05 ms) |
| `ForecasterStats` | ninguna: los modelos reciben pandas | `transform_series`, `transform_dataframe`, salidas | statsmodels y sktime necesitan el índice | Domina el modelo. Se queda como está |

### 2.2 Familia directa

Los dos forecasters ya entrenan y recorren los pasos en numpy, y la búsqueda one-step-ahead también. Lo que queda de pandas está en cada `predict` (y por tanto en cada fold sin refit). Caso: serie horaria de 2000 puntos, lags 24, steps 24 (24 modelos), exog de 3 columnas float ("num3") o una `category` de 7 niveles ("cat1"), Ridge y `LGBMRegressor(n_estimators=50)`; DMV con 10 o 20 series y `StandardScaler`; backtesting de 50 folds de 24 pasos sin refit; p3 pa.

| Forecaster / camino | Frontera | pandas que queda | Por qué | Coste |
|---|---|---|---|---|
| `ForecasterDirect.fit` | `y`: `_forecaster_direct.py:1041`; exog: `exog.to_numpy()` `:1137`; lags `:855-874` | Antes de la frontera, una vez: `transform_dataframe`, `OrdinalEncoder.fit_transform` (`:1118-1129`); `pd.Series` para las window features (`:1176`). Por paso: solo un índice si hay NaN o `weight_func` | API de window features; transformadores con nombres de columna | `_create_train_X_y` 0,40 / 0,67 / 2,73 ms (sin exog / num3 / cat1) de un `fit` de 28 a 32 ms. El `fit` es estimador (68 a 75 %) y `clone` (15 %, clonado doble en `:1898` y `:1922`) |
| `ForecasterDirectMultiVariate.fit` | por serie `to_numpy(copy=True)` (`_forecaster_direct_multivariate.py:1247`) y `transform_numpy`; exog `:1217` | Como el directo | | menos del 3 % del `fit`. Lo que pesa no es pandas: 24 `np.concatenate` por paso (`:1415-1473`), 38 de 178 ms con 10 series |
| `ForecasterDirect.predict` | ventana `:2162`, exog `:2236` | Por llamada: `check_predict_input` (~120 µs), **codificador sobre DataFrame (`:2218-2228`, ~1,0 ms)**, comparación de dtypes, `pd.Series` de salida | Índice y frecuencia, API del codificador | 0,205 / 0,371 / 1,58 ms (sin exog / num3 / cat1) |
| `ForecasterDirectMultiVariate.predict` | por serie `:2336` | Por llamada: **`iloc[...].copy()` de la ventana (`:2324`) más `[col].to_numpy(copy=True)` por columna: 397 µs con 10 series, 525 con 20**; `check_predict_input` vuelve a seleccionar las columnas (`utils.py:1619`, 163 µs); DataFrame de salida (100 µs); codificador como en el directo | Copias defensivas, orden de columnas | 1,29 ms (10 series, sin exog); pandas 80 % |
| Intervalos (los dos) | numpy hasta el final | Un DataFrame de 250 columnas; `predict_interval` y `predict_quantiles` llaman otra vez a `predict` (`_forecaster_direct.py:2948`, DMV `:3135`) y usan `DataFrame.quantile`, `transpose` y `concat` | Histórico | `predict_interval` del directo 1,71 ms frente a 0,58 del bootstrapping solo |
| Backtesting del directo | por `predict` | Por fold: `y.iloc`/`exog.iloc` (~10 µs) | | por fold ≈ `predict`: 0,22 / 0,42 / 1,97 ms. Con cat1 el codificador es el 30 % del backtesting |
| Backtesting del DMV | por `predict` | Por fold: `series.iloc` y **`.drop(columns=series_to_drop)` aunque la lista esté vacía** (`model_selection/_utils.py:1685`, 124 µs; compartido con multiserie) | | 1,77 ms por fold frente a 1,29 de `predict`; pandas 55 % |
| One-step-ahead | `_train_test_split_one_step_ahead`; camino numpy del DMV (`_utils.py:2890`) | ninguno | | |

**MSPERF §5 está desactualizado:** decía que el `StandardScaler` por serie dominaba el `predict` del DMV (90 µs por serie). Hoy `transform_numpy` tiene un atajo (`utils.py:2611-2696`, 31 µs para 10 series en total). Lo que domina es la extracción de la ventana y la selección duplicada de columnas, que el bloque A de PERF-predict-numpy (#1516) no tocó: por eso el DMV ganó solo un 4 a 9 %.

**Categóricas en el DMV (REVIEW-dmv-categorical).** El código es una copia del directo (`:1175-1215` y `:2391`); LightGBM recibe los índices correctos. Con categórica y `bool`, el DMV con LightGBM da las mismas predicciones que con la exog codificada de antemano. Sin probar: categóricas con diferenciación, con lags en diccionario y con intervalos conformal.

**`ForecasterRnn` y foundation (solo lectura de código; torch no está instalado).** `ForecasterRnn` sigue la filosofía en `predict` (frontera en `_forecaster_rnn.py:1388`, exog con `to_numpy()` sin dtype en `:1419`) pero se queda en pandas por serie en `fit` (`transform_series`, `_create_lags` con Series); no admite categóricas. Los adaptadores foundation guardan el contexto como Series y convierten en la frontera del modelo con `to_numpy(dtype=np.float32, na_value=np.nan)` (`_adapters.py:495`, `:1548`, `:4021`): **son los únicos que tratan bien los dtypes nullable**. TabICL y TabPFN construyen DataFrames largos a propósito porque esos modelos los piden. En los dos domina el modelo.

### 2.3 Multiserie (`ForecasterRecursiveMultiSeries`)

El camino de `predict` y del backtesting ya está medido en MSPERF y BTPLAN; aquí solo se completa lo que faltaba.

| Camino | Frontera | pandas que queda | Por qué | Coste |
|---|---|---|---|---|
| `fit`, series | `y.to_numpy()` por serie (`_forecaster_recursive_multiseries.py:994`) | Por serie: `copy` e `isna().all()` (`check_preprocess_series`), recorte con `.loc` (`align_series_and_exog_multiseries`), `index[ws:]`, `last_window_` (`:1740`) y **`training_range_` con `v[[0, -1]]`** (`:2502`) | Alineación, recorte de NaN, atributos guardados | 100 series: `training_range_` 4,4 a 7,7 ms (44 µs por serie; con `v[np.array([0, -1])]`, 11 µs e índice `identical`) |
| `fit`, window features | vuelve a pandas por serie (`:1029`) | `pd.Series`, `transform_batch` devuelve DataFrame, `iloc` | Contrato público de las window features | `RollingFeatures` con 3 estadísticos, 100 series: 55 de 126 ms del `fit` |
| `fit`, exog | **nunca del todo**: las columnas float64 se copian al bloque float (`:1249-1254`) y el bloque vuelve a ser un DataFrame (`:1258`) | Por serie: `exog.iloc[ws:]` (`:1565`), también con exog ancha (el mismo objeto en todas las series). Una vez: `concat` (`:1576`), `transform_dataframe`, codificador, `insert` de las columnas no float64 (`:1284-1291`) | `transformer_exog` y el codificador trabajan con DataFrame; las columnas no float64 conservan su dtype | 100 series, exog dict con `category`: comprobación de dtypes 4,3 ms (`DataFrame` más `nunique`), `iloc` por serie 5,8 ms, `concat` 2,6 a 3,5, codificador 21 ms (p2: 9 a 12) |
| `fit`, estimador | **el estimador recibe un DataFrame** (`:2482-2489`), y la predicción in-sample también (`:2525`) | `pd.isna(X_train)`, copia de filas para comprobar que el estimador no modifica `X_train` | Ninguna necesaria (§4.1) | Con un solo bloque float64, `fit(DataFrame)` cuesta lo mismo que `fit(numpy)`; con `ordinal_category`, LightGBM tarda 2,7 ms frente a 0,16 en convertirlo |
| `predict` (`_create_predict_inputs`) | ventana `:2904`, exog `:3037` (luego ida y vuelta numpy, DataFrame, numpy: `:3005`, `:3170`) | Ver MSPERF | | MSPERF |
| `predict_interval` (bootstrapping), `predict_quantiles` | igual | `_create_predict_inputs` dos veces (`:4190-4208`), `DataFrame.quantile` | Reutilizar los métodos públicos | 2 a 3 % del total |
| `set_out_sample_residuals` | numpy | Por nivel: `DataFrame.dropna` y `groupby.apply` (`:4959`, `:5009`) | | 100 niveles × 1000 residuos: 165 ms; el `groupby` 470 µs por nivel frente a 33 µs en numpy |
| Backtesting con refit | cada fold es un `fit` completo | Como `fit` | | Ridge, 100 series: lo que no es el estimador es el 60 % (sin exog) al 80 % (dict con `category`) de cada fold |
| One-step-ahead, por candidato | **nunca**: `estimator.fit(DataFrame)` y `predict(X_test DataFrame)` (`_utils.py:3031`, `:3038`) | Por nivel: DataFrame más `groupby`, `.loc` de `y_true`/`y_pred`, métricas con Series, `inverse_transform` sobre DataFrame (PREP T4) | | Ridge, 100 series: 75 a 100 ms por candidato = estimador 16 + métricas 37 a 40 (PERF-03) + **otro pandas 21 a 29 ms** |

Perfil del `fit` (Ridge, series en diccionario de 1000 puntos diarios, lags 24, p3 pa):

| Caso | `fit` | Estimador | numpy de skforecast | Codificador | Resto (pandas) |
|---|---|---|---|---|---|
| 20 series, sin exog | 14,6 ms | 34 % | 30 % | | 35 % |
| 100 series, sin exog | 53,9 ms | 31 % | 37 % | | 33 % |
| 100 series, 3 exog float anchas | 65,9 ms | 27 % | 32 % | | 41 % |
| 100 series, dict con 2 float y 1 `category` | 98,9 ms | 18 % | 19 % | 23 % | 38 % |

Con LightGBM el estimador es el 83 a 86 % del `fit` y la predicción in-sample el 10 a 13 % (PREP T3): todo skforecast es menos del 4 %. **Pasar el `fit` a numpy solo compensa con modelos baratos y con refit.**

## 3. Incoherencias entre forecasters

| # | Incoherencia | Efecto | Comprobado |
|---|---|---|---|
| I1 | Multiserie da un **DataFrame** al estimador en `fit` y **numpy** en `predict`; los demás, numpy en los dos | Bug de la propuesta 1 (predicciones erróneas con LightGBM y `category` sin `categorical_features`); la exog nullable con NA funciona en `fit` y falla en `predict` (`TypeError`, ya en SEGUIMIENTO); el estimador tiene `feature_names_in_` solo en multiserie | Sí, p2 y p3 |
| I2 | En los forecasters de una serie y en el directo multivariante, una exog `bool`, nullable o numérica de `pyarrow` hace `X_train` `object`; en multiserie queda como bloque aparte del DataFrame | `fit` 8 veces más lento con Ridge, 3 veces más memoria; mismo resultado | Sí |
| I3 | `ForecasterEquivalentDate` agrega con `np.apply_along_axis` con `offset` entero (el NaN se propaga) y con `DataFrame.apply(np.mean)` con `DateOffset` (salta el NaN) | Con un NaN en la ventana: `offset=7, n_offsets=2` da `nan`; `Day(7)` da 46,0 | Sí, p2 y p3 |
| I4 | `ForecasterRecursiveClassifier` no usa `_build_predict_function` en el bucle recursivo, ni `_prefers_c_contiguous_X`; no tiene `check_inputs` y `predict` no lleva `@manage_warnings` | LGBMClassifier 7 veces más lento por paso; el backtesting valida siempre | Lectura de código (`:1740-1746`) |
| I5 | El clasificador compara índices con `(a == b).all()`; los demás con `.equals` | 13 veces más lento; con zona horaria frente a sin ella lanza `TypeError` en vez del `ValueError` previsto | Según el subagente; no reproducido |
| I6 | Validación one-step-ahead: el directo multivariante es numpy con `transform_numpy` (`_utils.py:2895-2930`); multiserie, `groupby` y Series (`:3002-3220`) | 21 a 29 ms por candidato con 100 series | Lectura de código y perfil |
| I7 | Binning de residuos: in-sample en numpy; out-of-sample (y el in-sample de `ForecasterEquivalentDate`) con `groupby` | 14 a 100 veces más lento | Perfil |
| I8 | En multiserie, si alguna serie no tiene exog, las columnas `int` pasan a `float64` en el `concat` y `exog_dtypes_in_` depende de eso | Dtypes guardados distintos para los mismos datos | Según el subagente |
| I9 | `ForecasterStats` guarda `index_freq_` como texto (`freqstr`); los demás, como offset | Solo de forma | Lectura de código |
| I10 | Exog nullable con NA en `fit`: `ForecasterRecursive` y `ForecasterDirect` fallan con un `TypeError` de numpy (`float() argument must be ... not 'NAType'`), **también con LightGBM**, que acepta NaN; multiserie con LightGBM entrena (y falla después en `predict`, I1); los adaptadores foundation convierten con `na_value=np.nan` | Error crudo sin decir qué columna, y comportamiento distinto según el forecaster | Sí, p2 y p3 ([scripts/float64_na_fit.py](scripts/float64_na_fit.py)) |
| I11 | Transformador de `y`: los univariantes lo ajustan con DataFrame y lo aplican a arrays (dependen de silenciar el aviso de nombres); los de varias series usan numpy en los dos | `transformer_y.feature_names_in_` es `['y']` en unos y no existe en otros | Lectura de código |
| I12 | Ventana en `predict`: el directo la copia una vez; el DMV dos veces más una por columna, y `check_predict_input` vuelve a seleccionar las columnas | El DMV es el más lento de los tres (§2.2) | Perfil |

## 4. Fase 2: multiserie y exógenas categóricas

### 4.1 Qué recibe el estimador

Comprobado capturando la llamada a `estimator.fit` ([scripts/capture_fit.py](scripts/capture_fit.py)) con exog float, float+int, float+bool, float+`category`, float+texto y float+`Int64`, con Ridge y LightGBM:

- `ForecasterRecursive`, `ForecasterDirect` y `ForecasterDirectMultiVariate`: **array numpy**; float64 salvo con `bool` o `Int64`, que dan `object` (I2).
- `ForecasterRecursiveMultiSeries`: **DataFrame**, con las columnas no float64 (int, bool, `Int64`) como bloques aparte.
- Con categóricas, **todos** pasan a LightGBM `categorical_feature` con **índices de columna**, y la columna ya es float (salida del `OrdinalEncoder`). `configure_estimator_categorical_features` (`utils.py:610`) hace lo mismo con CatBoost (`cat_features`), XGBoost (`feature_types`) y HistGradientBoosting (`categorical_features`): siempre por índice.
- En `predict`, todos los forecasters (multiserie incluido) pasan al estimador una matriz float de numpy (`_build_predict_function`, `utils.py:4413`).

Prueba directa ([scripts/ms_numpy_fit.py](scripts/ms_numpy_fit.py)): en el `fit` de multiserie, pasar `X_train.to_numpy(dtype=float)` al estimador en lugar del DataFrame da **predicciones idénticas** con Ridge, LightGBM y HistGradientBoosting y con las codificaciones `ordinal`, `ordinal_category` y `onehot` (exog con float, int, bool y `category`). El `fit` con Ridge baja de 132 a 94 ms (`ordinal`) y de 147 a 86 ms (`ordinal_category`) con 50 series de 2000 filas; con LightGBM y HistGradientBoosting, dentro del ruido.

Tampoco hay una función que dependa del DataFrame en el estimador: un `Pipeline` con `ColumnTransformer` que selecciona columnas por nombre falla en `ForecasterRecursive` en el `fit` y en multiserie en el `predict` ([scripts/ct_names.py](scripts/ct_names.py)). Es decir, hoy no funciona en ninguno.

### 4.2 Dónde está el DataFrame solo por el dtype

En el `fit` de multiserie, y solo allí:

1. `encoding='ordinal_category'`: `_level_skforecast` es `pd.Categorical(categories=range(n))`. Funciona porque las categorías coinciden con los códigos (nota en `:1271-1274`).
2. `categorical_features=None` con una exog `category`: la columna llega al estimador como `category` y los estimadores que leen el dtype (LightGBM con `categorical_feature='auto'`, HistGradientBoosting con `'from_dtype'`) la tratan como categórica **en `fit`, pero no en `predict`**, que recibe los valores. Con categorías que no son `0..n-1` el modelo predice mal (propuesta 1):

   ```text
   LightGBM, categorical_features=None, exog category [10, 20, 30], efecto real −50 / 0 / +50
   ForecasterRecursive.predict                       [-50.0, 0.2, 50.2]
   ForecasterRecursiveMultiSeries.predict            [-50.1, -49.9, -50.1]
   estimator.predict(create_predict_X(...))          [-50.1, 0.3, 49.8]
   ```

   ([scripts/verify_cat_bug.py](scripts/verify_cat_bug.py); igual en p2 y p3.) No está en el roadmap. Probablemente pasa también con una lista de `categorical_features` que omite una columna `category` (no comprobado).
3. Columnas nullable con NA: el `fit` las acepta (scikit-learn convierte el DataFrame) y el `predict` falla con `TypeError` (hallazgo ya anotado en SEGUIMIENTO).

En los tres casos, el DataFrame en `fit` no da una función: da una diferencia entre `fit` y `predict`.

### 4.3 Qué impide hoy trabajar en numpy en multiserie

| Obstáculo | ¿Es real? | Dónde pesa |
|---|---|---|
| Arrays con dtypes mixtos | Solo antes de codificar. Tras el codificador, todo cabe en float64 (`bool` → 0/1, enteros exactos hasta 2^53) | `isna` por serie de una tabla mixta: 15 → 43 µs por serie (MSPERF §9), `to_numpy` de una tabla mixta da `object` |
| `OrdinalEncoder` | Es fila a fila y no depende de los vecinos: aplicarlo una vez a toda la exog da lo mismo que por fold (comprobado bit a bit, §5.1) | 21 ms de 99 en `fit` (100 series, p3); ~1 ms por `predict` univariante |
| Categorías por columna y por serie | Cada serie puede traer su `CategoricalDtype` (el `check_preprocess_exog_multiseries` exige las mismas categorías) | Solo en la codificación |
| NaN en categóricas y enteros nullable | Se resuelve codificando: NaN y categoría desconocida van a NaN (lo que ya hace el codificador) | `pd.NA` en un array `object` rompe el bucle de `predict` |
| `str` frente a `object` en pandas 3 | Solo afecta a operaciones de pandas: con `pyarrow`, convertir nombres de columna cuesta 4,5 a 11 µs por serie (MSPERF §9). En numpy desaparece | En el techo, p2, p3 y p3 pa tardan lo mismo (0,19 a 0,24 ms por fold) |
| Categóricas nativas del estimador | No: se pasan por índice (§4.1) | |
| `_level_skforecast` | No: son códigos enteros; `ordinal_category` solo necesita el dtype si el estimador lo infiere | |
| `transformer_exog` | **Sí.** Recibe un DataFrame y puede seleccionar por nombre o no ser fila a fila | Impide preparar la exog una vez salvo que sea `None` o fila a fila conocido |
| Refit | **Sí.** El codificador se reajusta en cada fold con refit y sus `categories_` pueden cambiar | Impide codificar una vez para todos los folds; se puede por tramo entre refits |
| Validación y avisos | **Sí.** Los avisos son por fold y por serie (NaN en exog, niveles desconocidos), en un orden concreto | Hay que decidir qué se valida una vez (P3 §11.6, punto 1) |
| Entradas como diccionario de series de distinta longitud | Parcial: se alinean en una matriz con NaN sobre la unión de índices | Coste una vez, no por fold |

### 4.4 Alternativas evaluadas

| Alternativa | Qué garantiza | Qué rompe o cuesta | API o `save_forecaster` | Relación |
|---|---|---|---|---|
| **(a) Codificar las categóricas a float al entrar** y guardar aparte los metadatos (categorías, orden de columnas, dtypes) | Desde ahí todo es float64; ningún dtype mixto aguas abajo | Nada, si el codificador público sigue siendo el `OrdinalEncoder` ajustado como hoy (PREP T2 lo resuelve así). Ojo con `transformer_exog` que devuelve numpy con texto (ISSUE-1508) | Ninguno | Es PERF-categorical-codes más "una vez por tabla". **Recomendada como base de todo lo demás** |
| (b) Bloques numpy separados por dtype que se unen al final | Conserva dtypes | Lo que ya hace multiserie con su bloque float más columnas insertadas: complejidad sin ganancia una vez codificado | Ninguno | Descartada: (a) la hace innecesaria |
| (c) Índice como array `int64`/`datetime64` y valores en numpy, sin DataFrame intermedio | Elimina la construcción de objetos por fold | El índice de salida tiene que ser idéntico (frecuencia, zona horaria, nombre): se construye una vez al final por posición de `y.index` (P3 §11.6, punto 3) | Ninguno | Necesaria para el backtesting en numpy (propuesta 3) |
| (d) pandas solo en la validación y en la salida | Mensajes y salidas iguales | Hay que separar "validar" de "preparar" en `check_predict_input` y `_create_predict_inputs` | `check_inputs` sigue privado | Es el marco: (a) y (c) son cómo se implementa |

## 5. Fase 3: propuestas medidas

### 5.1 El techo del backtesting multiserie en numpy

[scripts/ceiling_bt_ms.py](scripts/ceiling_bt_ms.py): `backtesting_forecaster_multiseries` con `refit=False`, 50 folds de 24 pasos, series en diccionario de 1500 filas horarias, lags 24, `encoding='ordinal'`, exog en diccionario (un DataFrame por serie): "float" = 2 columnas float; "mixed" = 2 float, una `int` (hora), una `category` de texto (día de la semana) y una columna de texto. El techo ajusta el mismo forecaster, codifica la exog una vez y recorre los folds cortando arrays por posición y llamando a `_recursive_predict`. "Solo el modelo" son las llamadas a la función de predicción del estimador.

| Estimador | Series | Exog | Hoy, ms por fold (p2 / p3 / p3 pa) | Techo, ms por fold (p2 / p3 / p3 pa) | Solo el modelo (p3 pa) | Codificar una vez, serie a serie (p3 pa) |
|---|---|---|---|---|---|---|
| Ridge | 100 | ninguna | 9,7 / 9,6 / 10,1 | 0,19 / 0,20 / 0,22 | 0,05 | |
| Ridge | 100 | float | 20,6 / 21,6 / 24,0 | 0,23 / 0,23 / 0,23 | 0,05 | 22 ms |
| Ridge | 100 | mixed | 29,6 / 37,5 / 35,5 | 0,23 / 0,22 / 0,24 | 0,05 | 228 ms |
| Ridge | 600 | ninguna | 56,5 / 59,2 / 62,0 | 0,47 / 0,44 / 0,47 | 0,11 | |
| Ridge | 600 | float | 115,8 / 135,8 / 141,2 | 0,51 / 0,50 / 0,53 | 0,11 | 134 ms |
| Ridge | 600 | mixed | 192,6 / 210,7 / 248,4 | 0,56 / 0,53 / 0,61 | 0,15 | 1375 ms |
| LightGBM (50 árboles) | 100 | ninguna | 12,4 / 14,0 / 13,9 | 4,3 / 4,3 / 4,6 | 3,2 | |
| LightGBM | 100 | float | 24,1 / 25,9 / 27,8 | 4,6 / 4,7 / 4,7 | 2,4 | 22 ms |
| LightGBM | 100 | mixed | 34,6 / 39,3 / 40,1 | 4,2 / 4,6 / 4,2 | 2,7 | 220 ms |
| LightGBM | 600 | ninguna | 70,6 / 76,2 / 77,8 | 20,8 / 22,3 / 21,9 | 16,5 | |
| LightGBM | 600 | float | 138,1 / 146,3 / 170,8 | 21,9 / 22,7 / 23,4 | 16,4 | 131 ms |
| LightGBM | 600 | mixed | 217,5 / 236,4 / 255,6 | 18,4 / 17,7 / 17,9 | 16,8 | 1451 ms |

Predicciones idénticas bit a bit en los 36 casos (`np.array_equal` frente a las del backtesting actual).

Lectura:

- Con Ridge, hoy el modelo es el 0,1 al 0,5 % del fold. Con LightGBM, del 6 al 23 %. Todo lo demás es preparación de entradas en pandas.
- El techo es casi independiente del entorno: **el camino en numpy también elimina la pérdida de pandas 3** (hoy p3 pa tarda de un 3 a un 29 % más que p2).
- La codificación "una vez" del prototipo se hizo serie a serie, que es lenta. Con un único `pd.concat` y una sola llamada al codificador ([scripts/preencode.py](scripts/preencode.py), 1500 filas por serie):

  | | 100 series (p2 / p3 pa) | 600 series (p2 / p3 pa) |
  |---|---|---|
  | Codificador serie a serie | 140 / 194 ms | 1022 / 1223 ms |
  | Un `concat` y un codificador | **33 / 47 ms** | **256 / 343 ms** |
  | Búsqueda por códigos serie a serie (PREP T2) | 44 / 109 ms | 319 / 661 ms |

  Las tres dan el mismo array. En multiserie, la regla es **una operación sobre la tabla de todas las series, no una rápida por serie**: el coste fijo de pandas por serie domina a la operación.
- Qué no está en el techo: la validación (que tendría que hacerse una vez), el índice de salida (una vez), las métricas (BTPLAN M1 las lleva a numpy) y los avisos. Ninguna de esas piezas es por fold y por serie si se diseña bien, así que lo alcanzable debería quedar cerca del techo más un coste fijo por backtesting.

### 5.2 Todas las propuestas

Estado: **nueva**, **existe** (fila del roadmap o plan), **amplía** (cambia el alcance de algo existente).

| # | Propuesta | Dónde | Ganancia medida | Riesgo | Esfuerzo | Estado |
|---|---|---|---|---|---|---|
| P1 | Dar numpy al estimador en el `fit` de multiserie; codificar siempre las `category` (o pasar sus índices) aunque `categorical_features=None` | `_forecaster_recursive_multiseries.py:2444-2489`, `:2525`, one-step-ahead `_utils.py:3031` | **Corrige predicciones erróneas** (§4.2). Ridge `fit` −30 a −60 ms con 50 series × 2000 filas | Medio: estimadores que infieren categóricas por dtype; con `ordinal_category` hay que pasar el índice de `_level_skforecast` explícitamente; `create_train_X_y` sigue devolviendo el DataFrame | S a M | nueva |
| P2 | `exog.to_numpy(dtype=float, na_value=np.nan)` cuando todas las columnas son numéricas o `bool` tras codificar | `ForecasterRecursive` `:921`, `:1659`; `ForecasterDirect` `:1137`, `:2236`; `ForecasterDirectMultiVariate` `:1217`, `:2409`; `ForecasterRecursiveClassifier` `:928`, `:1622`; `ForecasterRnn` `:1419` | `ForecasterRecursive`, una columna `bool`, 100.000 filas: `fit` con Ridge 231 → 27 ms y pico de memoria 126 → 45 MB; LightGBM 851 → 587 ms. `ForecasterDirect` (2000 filas, 24 pasos): `fit` 84 → 32 ms, `predict` 515 → 397 µs. `predict` con exog `Float64` o de `pyarrow`: 0,58 → ~0,30 ms. Resultado idéntico (Ridge y LightGBM convierten `object` a float64 igual). **Además corrige I10** (exog nullable con NA hace fallar el `fit`) | Bajo: enteros mayores que 2^53 (ya se pierden al juntarse con los lags float); las columnas de texto siguen por el camino actual; los NA pasan a NaN y decide el estimador, como en foundation (D-NP3) | S | nueva |
| P3 | Backtesting multiserie sin refit con preparación única: matriz de series alineada, exog codificada una vez (un `concat`, un codificador), cortes por posición, `_recursive_predict` directo, índice y salida una vez | `_backtesting_forecaster_multiseries`, `_extract_data_folds_multiseries`, `_create_predict_inputs` | Techo §5.1: Ridge 40 a 400 veces por fold, LightGBM 3 a 14 veces. Total de un backtesting de 50 folds con 600 series y exog mixta: Ridge ~12 s → ~0,4 s más la preparación | Alto si se intenta todo de golpe. Condiciones de la primera versión: `refit=False`, `transformer_exog=None` (o una lista blanca de transformadores fila a fila), sin intervalos, sin `return_predictors`; el resto por el camino actual. Aceptación bit a bit con las baterías de MSPERF | L | amplía PERF-predict-numpy (P3 §11.2 decía empezar por los univariantes) y absorbe PERF-backtesting-folds-positions (F1 a F3), B1 y B3 |
| P4 | Codificar las categóricas por códigos o con `pd.Index(categories_[i]).get_indexer(valores)` (−1 → NaN), manteniendo el `OrdinalEncoder` público para el ajuste; en multiserie, sobre la tabla concatenada | Los 5 forecasters (tabla de PREP T2) | Transformación del directo 1028 → 143 µs por llamada (comprobado idéntico con `category`, `str`, `object` y `string[pyarrow]`, con valores no vistos y `None`): `predict` del directo con cat1 1,58 → ~0,7 ms, fold 1,97 → ~1,1 ms; `fit` multiserie: codificador 21 ms de 99 (100 series, p3) | El de PREP T2; además categorías numéricas o de tipos mezclados sin probar, y se pierde la validación de nombres del codificador | M | existe (PERF-categorical-codes); ampliar a `str`/`object` en `predict` y "una vez por tabla" en multiserie |
| P5 | Validar una vez por backtesting; índice de predicción por posición de `y.index`; no recalcular entradas y recursión tras el bootstrapping | `_validation.py:431-459` | Ridge con exog float: −17 % (sin validar por fold), −31 % (más el índice); índice diario con zona horaria: −68 % | Avisos por fold (`MissingValuesWarning`), mensajes con el fold, pasos del `gap` | M | existe (PERF-predict-numpy, bloque B) |
| P6 | Cachear o simplificar `_is_utc_anchored_index` en `expand_index` | `utils.py:2575` | 0,4 ms por llamada con índice diario con zona horaria (expand_index 578 → 37 µs) | Bajo | XS a S | existe como hallazgo (anexo B del roadmap); sin fila |
| P7 | `training_range_` con `v[np.array([0, -1])]` | multiserie `:2502`, `:4653` | 44 → 11 µs por serie: 4 a 7 ms por `fit` con 100 series, ~45 ms con 1000; índice `identical` en los tipos probados | Trivial | XS | nueva |
| P8 | Exog ancha en el `fit` multiserie: cortar una vez el objeto compartido (caché por `id`, como el cambio 6 en `predict`) | multiserie `:1563-1565` | 4,7 → 0,05 ms con 100 series (~7 % del `fit` con Ridge), ~54 ms con 1000 | Bajo (se conserva el `concat`) | XS | nueva |
| P9 | Comprobar los dtypes de la exog en diccionario comparando listas, y solo si difieren construir el DataFrame del mensaje | `check_preprocess_exog_multiseries` | 4,3 ms por 100 series, 37 ms por 1000, en cada `fit` y fold con refit | Bajo; la igualdad de `CategoricalDtype` debe coincidir con `nunique` (sin comprobar) | S | nueva |
| P10 | Binning de residuos out-of-sample en numpy (máscara de NaN, `argsort` estable y `split`) | `ForecasterRecursive` `:3143-3150`, `ForecasterDirect`, `ForecasterDirectMultiVariate`, multiserie `:4959`, `:5009`, `ForecasterEquivalentDate` | 1,47 → 0,1 ms por llamada; multiserie 470 → 33 µs por nivel | Bajo: mismas claves y orden; las claves pasan de `np.int64` a `int` (decidir) | S | nueva |
| P11 | Intervalos por bootstrapping: `np.quantile` sobre el array (bit a bit igual que `DataFrame.quantile` sin NaN), reutilizar las entradas y las predicciones puntuales en vez de llamar otra vez a `predict`, un solo DataFrame de salida en vez de `concat` | `ForecasterRecursive` `:2555-2683`, `ForecasterDirect` `:2937-2958`, `:3069`, DMV `:3125-3147`, `:3264`, multiserie `:4190-4210` | −0,5 ms por llamada univariante; `predict_interval` del directo 1,7-2,3 → ~0,9-1,2 ms (−45 %); 2 a 3 % en multiserie | `DataFrame.quantile` salta NaN; `np.quantile` no (usar `nanquantile` o comprobar) | S | nueva |
| P12 | Conformal: cachear los factores por bin y sustituir `np.vectorize` por una búsqueda en array | `ForecasterRecursive` `:2412-2417`, `ForecasterEquivalentDate` `:953-959` | −0,6 ms por llamada (Ridge) | Bajo (solo numpy) | S | nueva |
| P13 | One-step-ahead multiserie por candidato: cortes por nivel calculados una vez y `y_true`/`y_pred` en numpy, como el directo multivariante | `_utils.py:3002-3220` | Hasta −21 a −29 ms de 75 a 100 ms por candidato (100 series, Ridge) | Métricas propias reciben Series hoy: mantenerlas por el camino actual (BTPLAN §2.3) | S a M | amplía PERF-03 y PERF-one-step-ahead-inverse |
| P14 | `ForecasterEquivalentDate`: `agg_func` vectorizada con `axis` cuando se pueda; con `DateOffset`, posiciones vectorizadas con `get_indexer` | `:526`, `:658`, `:677-727` | `fit` con bins 11 → ~2 ms; `predict` con `DateOffset` ~10 veces (estimado) | Depende de resolver I3 antes (semántica de NaN) | S a M | nueva |
| P16 | DMV: extraer la ventana con un único `to_numpy(dtype=float)` y posiciones de columna, sin `get_indexer` si el orden ya coincide, y reutilizar el array en la comprobación de NaN de `check_predict_input` | `_forecaster_direct_multivariate.py:2324-2340`, `utils.py:1617-1624` | 397/525 → 69 µs (10/20 series) más 163 µs de la comprobación: **~40 % del `predict` del DMV** y de cada fold sin refit; salidas idénticas | Bajo: dtypes nullable (`na_value`), copia escribible para las window features, el mensaje de la diferenciación necesita el índice | S | nueva |
| P17 | No llamar a `.drop(columns=series_to_drop)` con la lista vacía | `model_selection/_utils.py:1685` | 124 µs por fold (~7 % de un fold del DMV; también multiserie con DataFrame ancho) | Ninguno | XS | nueva |
| P15 | Clasificador: camino rápido de predicción (`_build_predict_function` con `predict_proba`) y `check_inputs` | `_forecaster_recursive_classifier.py:1740-1746` | LGBMClassifier 11 → ~1,5 ms por `predict` (estimado) | Medio; no es pandas pero es la mayor ganancia de ese forecaster | M | nueva |

### 5.3 Lo que no compensa

- Pasar a numpy el `fit` con modelos caros: con LightGBM el estimador es el 83 a 95 % del `fit` (univariante y multiserie). P1 se justifica por corrección, no por tiempo.
- Una llamada suelta a `predict`: 0,16 a 0,30 ms con Ridge en una serie, de los que 0,07 son el modelo. Solo importa en bucles.
- `ForecasterStats`: los modelos estadísticos necesitan el índice y dominan el tiempo.
- Las salidas públicas (`pd.Series` de salida: 8,5 µs) y `create_train_X_y`/`create_predict_X`, que devuelven DataFrames y restauran los dtypes.
- Las window features: su contrato público es Series de entrada y DataFrame de salida. Una versión numba de `RollingFeatures` ahorraría hasta 44 de 126 ms del `fit` con 100 series, pero los resultados de rolling de pandas no son idénticos bit a bit y la semántica de NaN está abierta (BUG-rollingfeatures-nan).

## 6. Fase 4: veredicto

**¿Es correcta la filosofía?** Sí. Las medidas lo confirman de forma rotunda: donde ya se aplica (construcción de `X_train` en los univariantes, bucle recursivo, estimador), el coste de pandas es irrelevante; donde no se aplica (preparación de entradas por fold y por serie), pandas es más del 95 % del tiempo con modelos baratos. Además, numpy hace desaparecer la diferencia entre pandas 2 y pandas 3: en el techo, los tres entornos tardan lo mismo.

**Qué hay que corregir de cómo se aplica:**

1. **El borde está en el sitio equivocado para los bucles.** "pandas en los bordes" se ha interpretado como "en cada método público", y el backtesting y las búsquedas llaman a los métodos públicos en cada fold. El borde de un backtesting es el propio backtesting: validar y preparar una vez, iterar en numpy, construir la salida una vez.
2. **"Convertir a numpy" no basta: hay que convertir a float64.** `to_numpy()` de una tabla con `bool`, nullable o `pyarrow` da `object`, y un array `object` es más lento que el DataFrame del que salió (I2, P2). Regla propuesta: tras codificar, `to_numpy(dtype=float)`; nunca `object` en una matriz de predictores.
3. **En multiserie, "una vez para todas las series", no "rápido por serie".** El coste fijo de pandas por serie (5 a 45 µs por operación) domina a la propia operación: optimizar la operación por serie gana poco (P4 serie a serie frente a tabla única).
4. **Las categóricas no justifican pandas aguas abajo.** Tras codificar son float y los estimadores las reciben por índice. El único sitio donde el dtype se usaba de verdad (el `fit` multiserie) es un bug (P1).

**¿Hace falta una representación interna común?** No una general. Recomiendo:

- **Disciplina caso por caso** en los métodos públicos, con dos reglas escritas: convertir a float64 tras codificar (P2) y dar siempre numpy al estimador (P1).
- **Un contenedor interno solo para el backtesting multiserie** (P3): la matriz de valores (tiempo × series, NaN donde una serie no tiene datos), la exog codificada (tiempo × series × exog, o su equivalente largo con desplazamientos), el índice común y los nombres. Vive dentro de `_backtesting_forecaster_multiseries` y se construye una vez. No es API.
- No hace falta para los univariantes: allí P5 (validar una vez y el índice por posición) da la mayor parte de lo alcanzable sin estructuras nuevas.

**¿Qué se simplifica al retirar pandas 2 (MAINT-drop-pandas2)?** El módulo `_pandas_compat.py` y los tests condicionados, pero **ninguna propuesta depende de ello**, y los caminos en numpy reducen la sensibilidad a la versión de pandas. No hay razón para esperar a retirar pandas 2.

**Orden recomendado:**

| Cuándo | Propuestas | Por qué |
|---|---|---|
| 0.28.x (ya) | P1 (bug), P2 (rendimiento y bug), P7, P8, P16, P17 | Corrección y cambios pequeños, aislados y bit a bit iguales (salvo los casos de error de P2, a decidir) |
| 0.28.x si da tiempo, si no 0.29.0 | P4 (PERF-categorical-codes, ya planificado), P9, P10, P11, P12 | Baratos, solo numpy |
| 0.29.0 | P3 (después de fusionar MSPERF y BTPLAN, que tocan las mismas funciones), P5, P6, P13 | El cambio grande; el diseño de P3 decide parte de P5 |
| Posterior o por decidir | P14 (tras decidir I3), P15 | No son de pandas frente a numpy, o necesitan una decisión antes |

## 7. Decisiones pendientes (con mi recomendación)

| ID | Decisión | Opciones | Recomendación |
|---|---|---|---|
| D-NP1 | Orden de PERF-predict-numpy: univariantes primero (P3 §11.2) o multiserie primero | (a) univariantes; (b) multiserie | **(b)**: en una serie el fold cuesta 0,3 a 0,8 ms y P5 se lleva la mayor parte; en multiserie, 10 a 250 ms por fold y 40 a 400 veces de margen |
| D-NP2 | P1: qué hacer con una `category` cuando `categorical_features=None` | (a) codificarla igualmente con el `OrdinalEncoder` como numérica ordinal (sin pasar índices al estimador); (b) error pidiendo `categorical_features`; (c) pasar sus índices como categóricas | **(a)**: es lo que recibe hoy `predict`, así que `fit` y `predict` coinciden sin cambiar el `predict` de nadie; documentarlo. (c) cambia el significado de `None` |
| D-NP3 | P2: exog nullable (`Int64`, `Float64`, `boolean`) con NA, en `fit` y en `predict` | (a) conservar el `TypeError` actual; (b) convertir NA a NaN y que el estimador decida (lo que hacen los adaptadores foundation y, en `fit`, multiserie) | (b): hoy el error es de numpy, no dice qué columna y aparece incluso con estimadores que aceptan NaN. Los NaN ya tienen su aviso (`MissingValuesWarning`) |
| D-NP4 | P3: alcance de la primera versión | Ver condiciones en P3 | `refit=False`, `transformer_exog=None`, sin intervalos ni `return_predictors`; validar una vez (incluidos los NaN de la exog en el `gap`, P3 §11.6) |
| D-NP5 | D-PREDICT-PUBLIC (`check_inputs` público) | | No hace falta: P3 y P5 usan caminos internos. Mantenerlo privado |
| D-NP6 | I3, NaN en `ForecasterEquivalentDate` | (a) propagar NaN en los dos caminos; (b) saltarlos en los dos | Decidirlo antes de P14; es un cambio de resultados en uno de los dos caminos |

## 8. Descartado (con su motivo)

- **Alternativa (b), bloques numpy por dtype:** con la codificación a float (a) no hace falta.
- **Codificar serie a serie con búsqueda por códigos en multiserie:** más lento que codificar la tabla concatenada (661 frente a 343 ms con 600 series en p3 pa).
- **Sustituir el `concat` de la exog ancha del `fit` multiserie por `np.tile`:** `transformer_exog` se ajusta sobre la tabla repetida, así que el resultado podría cambiar. P8 corta una vez sin quitar el `concat`.
- **`RollingFeatures` en numba:** no es bit a bit igual (§5.3).
- **Quitar pandas de `ForecasterStats`, de `create_train_X_y`, de `create_predict_X` y de las salidas:** son API o los necesita el modelo.
- **"Camino rápido numérico de `predict`" (descartado el 2026-11-17 porque perdía un `TypeError`):** la razón sigue valiendo para sustituir el `predict` público. No aplica a P3 ni a P5, que son caminos internos del backtesting y validan antes de entrar.

## 9. Filas propuestas para `dev/ROADMAP.md`

No se ha editado el roadmap. Filas nuevas (formato de la sección 4 y 5):

| ID | Qué | Versión | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|---|
| BUG-ms-fit-dataframe-categorical | Multiserie: el estimador recibe un DataFrame en `fit` y numpy en `predict`; con LightGBM, `categorical_features=None` y una exog `category` de enteros que no son `0..n-1`, `predict` es erróneo. Dar numpy al estimador en `fit` (P1) | 0.28.0 | S a M | Reproducido el 2026-11-19 (p2 y p3). Decisión D-NP2 | `dev/pandas_numpy_review/` §4.2, P1 |
| PERF-exog-float-matrix | Exog `bool`, nullable o numérica de `pyarrow` da `X_train` `object` en Recursive, Direct, DMV, Classifier y Rnn: `to_numpy(dtype=float, na_value=np.nan)` tras codificar (P2). Corrige además el `TypeError` de numpy en `fit` con NA en una columna nullable (I10) | 0.28.0 | S | Medido: `fit` 2,6 a 8 veces más rápido con Ridge, 3 veces menos memoria; error reproducido el 2026-11-19 (p2 y p3). Decisión D-NP3 | §5.2 P2, §3 I10 |
| PERF-dmv-last-window | DMV: ventana de `predict` con un único `to_numpy` y sin selección duplicada de columnas (P16); `drop` vacío por fold (P17) | 0.28.0 | S | Medido: ~40 % del `predict` del DMV | §2.2, §5.2 P16, P17 |
| PERF-ms-fit-small | `training_range_` sin lista (P7), exog ancha cortada una vez en `fit` (P8), dtypes de la exog en diccionario sin DataFrame (P9) | 0.28.0 | S | Medido salvo P9 | §5.2 |
| PERF-residuals-numpy | Binning out-of-sample en numpy en todos los forecasters (P10); `np.quantile` y entradas reutilizadas en los intervalos por bootstrapping (P11); factores conformal cacheados (P12) | 0.29.0 | S | Medido por micro-benchmark | §5.2 |
| PERF-ms-osa-metrics | One-step-ahead multiserie por candidato sin `groupby` por nivel (P13) | 0.29.0 | S a M | Con PERF-03 | §5.2 P13 |
| BUG-equivalent-date-nan | `ForecasterEquivalentDate`: `offset` entero propaga NaN y `DateOffset` los salta. Decidir (D-NP6) y vectorizar (P14) | por decidir | S a M | Reproducido el 2026-11-19 | §3 I3 |
| PERF-classifier-fast-predict | Clasificador: camino rápido de predicción por paso, `check_inputs`, `.equals` en vez de `(a == b).all()` (P15, I4, I5) | por decidir | M | Lectura de código y perfil | §3 I4, I5 |

Filas existentes que cambian:

- **PERF-predict-numpy:** añadir P3 (backtesting multiserie en numpy) con D-NP1 (multiserie primero) y D-NP4 (alcance); anotar que absorbe B3 y que B1, F1 a F3 son sus primeros pasos. Añadir P6 (`_is_utc_anchored_index`).
- **PERF-categorical-codes:** añadir "en multiserie, sobre la tabla concatenada, no serie a serie" (§5.1).
- **D-PREDICT-PUBLIC:** recomendación de mantenerlo privado (D-NP5).
- **REVIEW-dmv-categorical:** anotar lo revisado en §2.2 (índices correctos, mismas predicciones que con la exog codificada de antemano) y lo que falta (diferenciación, lags en diccionario, conformal).
- **PERF-multiseries-names-pyarrow:** MSPERF §5 está desactualizado sobre el `predict` del DMV (§2.2).

## Anexo: cómo reproducir

```bash
# Entornos (sesión en la nube; en local, los equivalentes de conda)
uv venv ~/.venvs/skforecast-p2 --python 3.12
uv pip install --python ~/.venvs/skforecast-p2/bin/python -e ".[test]" "pandas<3" lightgbm pyarrow
# p3: el entorno de la sesión; p3 sin pyarrow: NO_PYARROW=1 (bloquea la importación)

bash dev/pandas_numpy_review/scripts/run_ceiling.sh > ceiling_results.txt   # §5.1, unos 15 minutos
python dev/pandas_numpy_review/scripts/preencode.py 600                     # §5.1, codificar una vez
python dev/pandas_numpy_review/scripts/verify_cat_bug.py                    # §4.2, bug P1
python dev/pandas_numpy_review/scripts/ms_numpy_fit.py                      # §4.1, numpy en el fit multiserie
python dev/pandas_numpy_review/scripts/bool_exog_fit.py                     # P2
python dev/pandas_numpy_review/scripts/float64_na_fit.py                    # P2, I10
python dev/pandas_numpy_review/scripts/eqdate_nan.py                        # I3
```
