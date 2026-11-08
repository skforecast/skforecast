# PERF-contiguous-lags: matriz de entrenamiento contigua para el estimador

Rama `perf/contiguous-lag-matrix` desde `origin/0.27.x` (17219ad65), solo en local.
Medición con recomendación: el código del paquete no cambia. El prototipo está en
`repro/PERF-contiguous-lags_prototype.diff` y no está aplicado en la rama.

**Estado (2026-11-08, fase 2):** medición cerrada; opción D elegida, pendiente de aprobar
la implementación.

- **Lista final** (sección 14): `LinearRegression`, `Ridge`, `Lars` y `LassoLars`.
- **Pipeline:** cuenta su primer paso, porque es el que recibe la matriz. Un Pipeline que
  empieza por `StandardScaler` no se copia.
- **Dónde se copia:** en Recursive, Direct y DMV. Classifier no cambia.
- **Datos:** la tabla comparativa limpia está en la sección 15 y la implementación,
  descrita sin aplicar, en la sección 16.
- **Tamaño S.** La fila propuesta está en la sección 11; no se ha añadido a `roadmap.md`.

Las secciones 0 a 12 son la fase 1. Se medía con la máquina cargada (carga de 38 a 76) y
con batería, y la recomendación era provisional. Donde la fase 2 corrige algo, lo dice.

Origen: QW-13, sección 8.6. Con solo lags y sin NaN, `ForecasterRecursive` pasa al
estimador una vista invertida de `sliding_window_view` (strides `(8, -8)`, ni C ni F
contigua). Con 20.000 x 24, el `fit` de Ridge tarda 1,35 ms con la vista y 0,82 ms con
una copia contigua.

## 0. Resumen

**Fase 2 (secciones 13 a 17), en condiciones limpias:** carga por debajo de 4 antes de
cada fila y equipo enchufado.

- **Lista de la opción D.** Entran `LinearRegression`, `Ridge`, `Lars` y `LassoLars`. Cumplen
  los tres requisitos:
  - ganan más de un 5 % en al menos dos de los tres tamaños;
  - suben la memoria pico un 12 % de la matriz como mucho;
  - dan predicciones idénticas.
- **Fuera de la lista:**
  - RidgeCV, BayesianRidge y ARDRegression: ganan, pero suman una matriz entera de pico
    (+100 %).
  - HuberRegressor, PoissonRegressor y TweedieRegressor: ganan mucho por iteración, pero
    sin escalar no convergen y la copia cambia sus predicciones. Además suman +100 % de
    memoria.
  - RidgeClassifier: ganancia inestable (5 % o menos en limpio).
  - Los demás: no ganan.
- **Pipeline.** La regla de mirar el último paso, que proponía la fase 1, suma una matriz
  de pico con `StandardScaler` (+100 %). Se mira el primer paso: un Pipeline que empieza
  por un transformador no se copia.
- **Ganancia de la opción D** frente a `0.27.x` (sección 15), con 20.000 filas o más:
  - `fit` de Recursive: entre un 4 y un 30 % menos.
  - `fit` de Direct (5 pasos): entre un 7 y un 43 % menos.
  - Backtesting con `refit=True` y búsquedas con muchos `fit`: entre un 6 y un 33 % menos.
  - Controles (HistGradientBoosting, LightGBM, Pipelines con escalador): sin cambio.
- **Resultados.** Las predicciones son idénticas en todas las filas. Con el prototipo en el
  modo D pasan 3.766 tests sin fallos.

**Fase 1:**

- **Dónde llega una vista al estimador** (sección 2). Recursive, Classifier y Direct con
  `lags` entero y sin `exog`, window features ni calendario. Da igual que haya
  `differentiation` o `transformer_y`. Lo mismo con solo window features (otra vista
  no contigua) y en DMV cuando solo una serie tiene lags. La lista de lags no consecutivos
  llega en orden F. MultiSeries pasa un DataFrame en orden F, salvo con `encoding=None`, que
  lo pasa en orden C. Direct y DMV con `n_jobs > 1` reciben una copia C porque joblib
  serializa la matriz. Las matrices de `predict` siempre son contiguas.
- **Quién gana.** Solo los modelos lineales que operan con `X` directamente:
  - Ridge, con 20.000 filas o más: entre un 19 y un 29 % menos en el `fit` completo de
    Recursive, y entre un 31 y un 42 % en Direct.
  - LinearRegression: entre un 3 y un 17 %.
  - Los dos dentro de un `Pipeline` con `StandardScaler`: entre un 4 y un 17 %.

  La ganancia sale del `fit` del estimador, del `predict` sobre `X_train` que el `fit`
  hace para los residuos in-sample (la vista es hasta 4 veces más lenta) y de `pd.isna`.
- **Dónde se nota más.** En backtesting con `refit=True` y en búsquedas con Ridge:
  - backtesting de 417 folds: un 15 a 17 % menos;
  - backtesting de Direct con 24 pasos: un 30 % menos;
  - `grid_search_forecaster` con `refit=True`: un 20 % menos;
  - búsqueda con `OneStepAheadFold`: entre un 31 y un 34 % menos.
- **Quién no gana.**
  - Lasso, ElasticNet, KNeighbors, RandomForest, LightGBM, XGBoost, CatBoost y
    LogisticRegression: la diferencia queda entre 0,95 y 1,05, porque ya copian o
    convierten la matriz por su cuenta.
  - HistGradientBoosting pierde un 4 % con 200.000 x 168: su `predict` es 3 veces más lento
    sobre la copia C que sobre la vista.
- **Resultados.** Las predicciones puntuales son idénticas bit a bit en los 88 casos del
  `fit` completo. Con modelos lineales, los residuos in-sample y los intervalos cambian
  menos de 2e-13. La métrica de `OneStepAheadFold` cambia 1,9e-15. Con el prototipo pasan
  todos los tests: 1.586 de los cuatro forecasters y 1.195 de `model_selection`.
- **Memoria.** La copia suma una matriz más al pico (256 MB con 200.000 x 168) con los
  estimadores que no copian por su cuenta: Lasso, ElasticNet y HistGradientBoosting. Con
  Ridge suma 0 y con LinearRegression y LightGBM, alrededor del 12 % de la matriz.
- **Recomendación.** Copiar solo con los lineales que ganan (opción D, sección 10). Copiar
  siempre (opción B) es más simple, pero tiene coste sin beneficio con los demás
  estimadores. Construir los lags en orden C desde el principio (opción C) cuesta lo mismo
  que copiar. El orden F (sección 9) queda descartado.

## 1. Entorno y método

- Python 3.14.3, numpy 2.5.4 (Accelerate), pandas 2.3.3, scikit-learn 1.9.1, scipy 1.18.1,
  joblib 1.6.0, LightGBM 4.6.0, XGBoost 3.2.0, CatBoost 1.2.10.
- Apple M5 (4 núcleos de rendimiento y 6 de eficiencia, 16 GB), macOS 27.2, con batería.
- **Carga.** Otra sesión trabajaba en paralelo en el repositorio principal y la carga del
  sistema estuvo entre 38 y 76. Los estimadores con hilos OpenMP fueron mucho más lentos de
  lo normal. Por ejemplo, el `fit` de HistGradientBoosting con 1.000 x 24 tarda 1,8 s con
  todos los hilos y 0,8 s con `OMP_NUM_THREADS=4`.
  - Los tiempos absolutos de los árboles no son representativos. La comparación sí lo es:
    las dos versiones se alternan en el mismo proceso.
  - Algunas bajadas sueltas de 0,85 a 0,89 en árboles con 20.000 x 168 no se repiten ni en
    el script 06 ni en la variante F, así que son ruido (sección 4).
- **Método.** Las dos versiones corren en el mismo proceso. El prototipo añade a los cuatro
  forecasters un indicador de módulo `_CONTIGUOUS_X_TRAIN`, y cada medición lo alterna:
  - Indicador apagado: el camino de `0.27.x` más un `if`.
  - Indicador encendido: `np.ascontiguousarray` cuando la matriz es un solo bloque (solo
    lags o solo window features). En Recursive y Classifier se aplica antes del filtrado de
    NaN; en Direct y DMV, a `X_train_autoreg` cuando no hay `exog` ni calendario.
  - Cada medición es la media de varios `fit` seguidos (al menos 0,2 s). Se toman entre 2 y
    7 rondas por versión, alternando, según lo que tarde el `fit`. Las tablas dan el mínimo.
    Las medianas están en los CSV.
- Ningún benchmark corrió en paralelo con otro (scripts uno detrás de otro).

Scripts (`repro/`, ejecutar desde la raíz del repositorio; los marcados con (p) necesitan
el prototipo, `git apply dev/roadmap_0.27.0/repro/PERF-contiguous-lags_prototype.diff`):

| Script | Qué mide |
|---|---|
| `PERF-contiguous-lags_01_inventory.py` | Disposición de lo que recibe el estimador en `fit` y `predict` (argumento `prototype` para verlo con el prototipo) |
| `PERF-contiguous-lags_02_copy_methods.py` | Coste de las formas de obtener los lags en orden C y de `pd.isna` |
| `PERF-contiguous-lags_03_fit.py` (p) | `forecaster.fit` completo; `_03_fit.csv` y, con `F`, `_03_fit_F.csv` |
| `PERF-contiguous-lags_04_backtesting_search.py` (p) | Backtesting con `refit=True` y búsquedas con Ridge; `_04_backtesting_search.csv` |
| `PERF-contiguous-lags_05_memory.py` (p) | Pico de memoria del `fit` (`tracemalloc`) |
| `PERF-contiguous-lags_06_estimator_fit_predict.py` | `fit` y `predict` del estimador solo, con vista, F y C |
| `PERF-contiguous-lags_07_residuals.py` (p) | Residuos in-sample e intervalos, vista frente a C |
| `PERF-contiguous-lags_08_pipeline.py` (p) | `Pipeline(StandardScaler, Ridge o LinearRegression)` |

## 2. Inventario: qué recibe el estimador

Script 01, `0.27.x`, 500 observaciones, `lags=24`. Notación:

- `vista(a,b)`: ni C ni F contigua, strides `(a, b)` en bytes.
- `df:` el estimador recibe un DataFrame; se da la disposición de `np.asarray`, que es lo
  que valida scikit-learn.
- "predict en fit": el `predict` sobre `X_train` que el `fit` hace para los residuos
  in-sample. Classifier no lo hace.

| Caso | Recursive | Classifier | Direct (`n_jobs=1`) |
|---|---|---|---|
| `lags` entero | vista(8,-8) | vista(8,-8) | vista(8,-8) |
| `lags=[1, 2, 3, 24]` (no consecutivos) | F | F | F |
| lags + window features | C | C | C |
| solo window features | vista(8,4000) | vista(8,4000) | vista(8,3984) |
| lags + `exog` | C | C | C |
| NaN en `y`, `dropna_from_series` False o True | C (el filtrado copia) | C | C |
| NaN en `exog`, False o True | C | C | C |
| `differentiation=1` | vista(8,-8) | - | vista(8,-8) |
| `transformer_y=StandardScaler()` | vista(8,-8) | - | vista(8,-8) |
| `differentiation` + `transformer_y` | vista(8,-8) | - | vista(8,-8) |

- **Predict en fit.** Recibe la misma matriz que el `fit` en todos los casos.
- **`y` del `fit`.** Siempre C, también la columna de cada paso en Direct.
- **Direct y DMV con `n_jobs > 1`.** Siempre reciben C, porque joblib serializa la matriz o
  la pasa por memmap a partir de 1 MB. Es lo que eligen por defecto con árboles
  (`n_jobs='auto'` da `cpu_count() - 1`). Con modelos lineales y con LightGBM eligen
  `n_jobs=1`, y la vista llega al estimador.
- **ForecasterDirectMultiVariate** (3 series): C en todos los casos (los lags de cada serie
  se concatenan), salvo con lags en una sola serie (`lags={"s0": 24, "s1": None, ...}`),
  que da vista(8,-8) con `n_jobs=1`.
- **ForecasterRecursiveMultiSeries:** DataFrame en orden F, porque es el bloque que se
  construye con orden F a propósito. La excepción es `encoding=None` sin `exog`, que da C
  (el `drop` de la columna del nivel copia en orden C; es el caso del comentario hacia la
  línea 1150). Según el script 06, el orden F rinde como el C con los lineales, así que
  MultiSeries no necesita cambios.
- **`_create_predict_inputs` y predicción.** Recursive y Classifier predicen con un `X`
  preasignado de una fila (C). El bootstrapping usa `X` de `(n_boot, n_features)` en orden
  F a propósito (comentario hacia la línea 1838). Direct, DMV y MultiSeries usan matrices
  nuevas. No hay vistas en ningún `predict`.
- **Fuera de alcance:** `ForecasterRnn`, `stats`, `foundation` y
  `ForecasterEquivalentDate` (sin estimador).

Con el prototipo (`... _01_inventory.py prototype`), todo lo que llega al `fit` y al
predict en fit de Recursive, Classifier, Direct y DMV es C. MultiSeries no cambia.

## 3. Coste de la copia

Script 02 (y la columna F, sección 9). Lags de Recursive, mínimo de 7 repeticiones, en ms.

| Filas | Lags | Vista (hoy) | `np.ascontiguousarray` | Bucle por columnas (C) | `y` invertida + `ascontiguousarray` | `np.take` 2D | `np.asfortranarray` | `pd.isna(X).any()` vista / C |
|---|---|---|---|---|---|---|---|---|
| 1.000 | 24 | 0,004 | 0,012 | 0,013 | 0,015 | 0,039 | 0,006 | 0,011 / 0,003 |
| 1.000 | 168 | 0,004 | 0,061 | 0,102 | 0,062 | 0,318 | 0,019 | 0,071 / 0,016 |
| 20.000 | 24 | 0,004 | 0,33 | 0,57 | 0,32 | 0,98 | 0,20 | 0,19 / 0,045 |
| 20.000 | 168 | 0,004 | 2,46 | 4,75 | 2,39 | 7,27 | 1,38 | 1,49 / 0,62 |
| 200.000 | 24 | 0,004 | 3,43 | 22,3 | 3,55 | 10,4 | 2,08 | 2,13 / 0,86 |
| 200.000 | 168 | 0,003 | 25,5 | 180 | 26,7 | 70,1 | 16,9 | 14,7 / 6,3 |

- `np.ascontiguousarray` sobre la vista es la forma más barata de tener los lags en orden C.
  Invertir `y` primero cuesta lo mismo. El bucle por columnas y `np.take` son de 2 a 7 veces
  más lentos.
- Con `lag_1` en la primera columna, ninguna vista de `y` puede ser C contigua. Por tanto,
  "construir los lags en orden C desde el principio" cuesta lo mismo que copiar después: la
  diferencia entre las opciones está en dónde se copia, no en cómo.
- La copia recupera parte de su coste en el `pd.isna` del filtrado de NaN (QW-13, 8.6): con
  20.000 x 24 son 0,15 ms de los 0,33 que cuesta.

## 4. `forecaster.fit` completo

Script 03. Tiempo en ms con la vista → con la copia C (prototipo), y entre paréntesis
`C / vista` (mínimos). Las predicciones (`predict(steps=24)`, `predict()` en Direct y
`predict_proba` en Classifier) son idénticas bit a bit en los 88 casos. KNeighbors no se mide
con 200.000 filas (el predict en fit tarda minutos). Direct usa `steps=5` y `n_jobs='auto'`:
1 proceso con los lineales y LightGBM, y procesos con HistGradientBoosting (que ya recibe
una copia C).

**ForecasterRecursive**

| Estimador | 1.000 x 24 | 1.000 x 168 | 20.000 x 24 | 20.000 x 168 | 200.000 x 24 | 200.000 x 168 |
|---|---|---|---|---|---|---|
| LinearRegression | 0,67 → 0,60 (0,90) | 4,06 → 3,62 (0,89) | 4,59 → 3,97 (0,86) | 47,1 → 45,5 (0,97) | 48,1 → 43,6 (0,91) | 709 → 683 (0,96) |
| Ridge | 0,44 → 0,42 (0,97) | 1,11 → 0,84 (0,75) | 2,47 → 1,86 (0,75) | 17,5 → 12,4 (0,71) | 22,7 → 18,4 (0,81) | 148 → 110 (0,75) |
| Lasso | 1,24 → 1,19 (0,96) | 13,0 → 12,8 (0,99) | 6,11 → 6,31 (1,03) | 483 → 461 (0,95) | 192 → 202 (1,05) | 4.486 → 4.648 (1,04) |
| ElasticNet | 1,58 → 1,56 (0,99) | 20,6 → 20,9 (1,01) | 10,6 → 10,6 (1,00) | 525 → 549 (1,05) | 326 → 317 (0,97) | 5.343 → 5.473 (1,02) |
| KNeighbors | 2,45 → 2,65 (1,08) | 3,04 → 2,72 (0,89) | 125 → 124 (0,99) | 370 → 358 (0,97) | - | - |
| RandomForest (10 árboles, profundidad 10) | 39,8 → 38,3 (0,96) | 223 → 226 (1,01) | 992 → 1.000 (1,01) | 7.926 → 7.067 (0,89)* | 15.251 → 15.420 (1,01) | 110.773 → 99.059 (0,89)* |
| HistGradientBoosting | 2.074 → 1.879 (0,91)* | 2.014 → 2.007 (1,00) | 1.310 → 1.274 (0,97) | 2.884 → 2.458 (0,85)* | 3.239 → 3.232 (1,00) | 4.055 → 4.218 (1,04) |
| LightGBM | 399 → 402 (1,01) | 468 → 465 (0,99) | 389 → 391 (1,01) | 901 → 786 (0,87)* | 1.109 → 1.115 (1,00) | 2.901 → 2.876 (0,99) |
| XGBoost | 158 → 166 (1,05) | 371 → 371 (1,00) | 166 → 165 (0,99) | 824 → 714 (0,87)* | 435 → 408 (0,94) | 1.676 → 1.745 (1,04) |
| CatBoost | 119 → 115 (0,97) | 332 → 325 (0,98) | 158 → 159 (1,00) | 563 → 546 (0,97) | 533 → 549 (1,03) | 1.483 → 1.533 (1,03) |

\* Ruido, por cuatro motivos:

- La diferencia es de 30 a 11.700 ms, y la copia cuesta de 0,01 a 25 ms.
- Son 2 o 3 rondas, con medianas muy dispersas.
- El script 06 no lo repite con el estimador solo: `fit` de 0,99 a 1,03 con 20.000 x 168.
- La variante F, que hace la misma copia, da 0,98 a 1,02 en esos casos.

**ForecasterDirect** (`steps=5`)

| Estimador | 1.000 x 24 | 1.000 x 168 | 20.000 x 24 | 20.000 x 168 | 200.000 x 24 | 200.000 x 168 |
|---|---|---|---|---|---|---|
| LinearRegression | 2,41 → 2,33 (0,97) | 19,0 → 17,6 (0,93) | 23,1 → 20,0 (0,87) | 201 → 175 (0,87) | 219 → 182 (0,83) | 3.283 → 3.087 (0,94) |
| Ridge | 1,88 → 1,75 (0,93) | 4,74 → 3,27 (0,69) | 12,8 → 8,86 (0,69) | 66,1 → 40,1 (0,61) | 93,3 → 62,1 (0,67) | 701 → 407 (0,58) |
| HistGradientBoosting (9 procesos) | 120 → 130 (1,08) | 680 → 674 (0,99) | 459 → 453 (0,99) | 2.011 → 2.014 (1,00) | 2.507 → 2.534 (1,01) | 15.094 → 15.366 (1,02) |
| LightGBM (1 proceso) | 1.563 → 1.576 (1,01) | 1.886 → 1.889 (1,00) | 1.967 → 1.970 (1,00) | 3.243 → 3.220 (0,99) | 4.049 → 4.085 (1,01) | 15.168 → 14.858 (0,98) |

En Direct se hace una copia que comparten los 5 estimadores, por eso Ridge gana más.

**ForecasterRecursiveClassifier** (LogisticRegression)

| 1.000 x 24 | 1.000 x 168 | 20.000 x 24 | 20.000 x 168 | 200.000 x 24 | 200.000 x 168 |
|---|---|---|---|---|---|
| 6,13 → 6,14 (1,00) | 28,6 → 28,6 (1,00) | 53,4 → 52,9 (0,99) | 463 → 470 (1,01) | 770 → 775 (1,01) | 4.666 → 4.671 (1,00) |

LogisticRegression valida `X` con `order="C"` y copia la vista por su cuenta, así que no
cambia nada.

**Pipeline con `StandardScaler`** (script 08, Recursive)

| Estimador | 1.000 x 24 | 20.000 x 24 | 20.000 x 168 | 200.000 x 168 |
|---|---|---|---|---|
| `Pipeline(StandardScaler, Ridge)` | 0,93 → 0,85 (0,91) | 5,16 → 4,27 (0,83) | 29,7 → 24,6 (0,83) | 317 → 279 (0,88) |
| `Pipeline(StandardScaler, LinearRegression)` | 0,92 → 0,86 (0,93) | 7,42 → 6,55 (0,88) | 58,5 → 53,3 (0,91) | 838 → 808 (0,96) |

El escalador recibe la vista y le pasa una matriz nueva al estimador. Aun así, el `fit` y
el `transform` del escalador y el predict en fit ganan. Las predicciones son idénticas.

## 5. De dónde sale la ganancia

Script 06: `fit` y `predict` del estimador solo, sin el forecaster, con las tres
disposiciones que producen hoy los forecasters. Mínimo (mediana) en ms; 3 a 9 rondas
alternando.

| Filas x lags | Estimador | Llamada | Vista | F | C | C / vista |
|---|---|---|---|---|---|---|
| 20.000 x 24 | LinearRegression | fit | 3,99 (4,92) | 3,50 (3,96) | 3,24 (3,61) | 0,81 |
| 20.000 x 24 | LinearRegression | predict | 0,48 (0,54) | 0,21 (0,23) | 0,24 (0,26) | 0,49 |
| 20.000 x 24 | Ridge | fit | 1,76 (1,85) | 1,37 (1,46) | 1,25 (1,71) | 0,71 |
| 20.000 x 24 | Ridge | predict | 0,50 (0,63) | 0,21 (0,27) | 0,24 (0,32) | 0,48 |
| 20.000 x 168 | LinearRegression | predict | 3,36 (3,59) | 0,86 (0,93) | 0,86 (0,93) | 0,25 |
| 20.000 x 168 | Ridge | fit | 10,3 (11,3) | 6,51 (7,53) | 7,05 (7,64) | 0,69 |
| 20.000 x 168 | Ridge | predict | 3,28 (3,43) | 0,85 (0,96) | 0,85 (1,10) | 0,26 |
| 20.000 x 168 | HistGradientBoosting | fit | 1.970 (1.976) | 1.969 (1.979) | 1.962 (1.973) | 1,00 |
| 20.000 x 168 | HistGradientBoosting | predict | 17,3 (17,5) | 17,9 (18,0) | 26,1 (26,3) | 1,51 |
| 20.000 x 168 | LightGBM | predict | 12,4 (12,9) | 10,9 (11,1) | 9,58 (10,0) | 0,78 |
| 200.000 x 168 | LinearRegression | predict | 34,8 (39,0) | 7,97 (15,7) | 8,02 (8,04) | 0,23 |
| 200.000 x 168 | Ridge | fit | 108 (122) | 60,4 (70,3) | 64,9 (69,0) | 0,60 |
| 200.000 x 168 | Ridge | predict | 32,6 (36,8) | 7,91 (8,66) | 7,72 (8,62) | 0,24 |
| 200.000 x 168 | HistGradientBoosting | predict | 83,9 (92,2) | 72,2 (77,2) | 263 (312) | 3,14 |
| 200.000 x 168 | LightGBM | predict | 131 (142) | 68,6 (74,3) | 64,4 (71,5) | 0,49 |

Las demás filas que imprime el script (RandomForest, XGBoost y el resto de `fit` y
`predict` de los árboles) quedan entre 0,93 y 1,11, y el `fit` de LinearRegression entre
0,81 y 0,94. Hay dos excepciones con medianas muy dispersas: el `fit` de LightGBM con
200.000 x 168 (0,83) y el de HistGradientBoosting (1,12). No se repiten en el `fit`
completo (sección 4: 0,99 y 1,04).

- **Lineales.** Para `X @ coef` y `X.T @ X` sobre la vista, numpy tiene que copiar la vista
  antes de llamar a BLAS, y esa copia es más lenta que `np.ascontiguousarray`. Es la
  explicación más probable, y cuadra con que los coeficientes no cambien (sección 8). El
  `predict` sobre la vista es entre 2 y 4,3 veces más lento. En el `fit` del forecaster
  hay tres consumidores de la vista (`pd.isna`, `estimator.fit` y el predict en fit) y
  cada uno paga su conversión. Una sola copia al principio sirve a los tres. F rinde
  como C.
- **HistGradientBoosting `predict`.** Es 3 veces más lento en C que en la vista o en F con
  200.000 x 168. Al recorrer los árboles por filas, cada fila salta entre columnas.
  - La vista solapa sus filas y ocupa n + k valores (1,6 MB), que caben en caché.
  - En F, filas vecinas de una misma columna comparten línea de caché.
  - En C (256 MB), cada salto toca una línea distinta.

  Explica el 1,04 de HistGradientBoosting con 200.000 x 168 en la sección 4: 163 ms más en
  el `fit` completo, frente a 179 ms del predict en fit.
- **LightGBM `predict`** gana con C (copia la vista por su cuenta), pero en el `fit`
  completo son unos pocos ms frente a segundos.

## 6. Backtesting y búsquedas

Script 04, `ForecasterRecursive` salvo donde se indica. `n_jobs='auto'` da 1 proceso con
lineales y con LightGBM. Mínimo en s, vista → C.

| Caso | `fit` | Vista → C | C / vista | Ahorro por `fit` | Diferencia máxima |
|---|---|---|---|---|---|
| Backtesting Ridge, `lags=24`, 1.000 de entrenamiento, `steps=12` | 84 | 0,046 → 0,046 | 1,00 | 0 ms | 0 |
| Backtesting LinearRegression, igual | 84 | 0,056 → 0,054 | 0,97 | 0,02 ms | 0 |
| Backtesting Ridge, `lags=168`, igual | 84 | 0,086 → 0,077 | 0,89 | 0,11 ms | 0 |
| Backtesting LinearRegression, `lags=168`, igual | 84 | 0,327 → 0,319 | 0,98 | 0,09 ms | 0 |
| Backtesting LightGBM (50 árboles), `lags` 24 / 168 | 84 | 13,9 → 14,6 / 19,1 → 22,7 | 1,05 / 1,19* | - | 0 |
| Backtesting Ridge, `lags=24`, 10.000 de entrenamiento, `steps=24` | 417 | 0,583 → 0,493 | 0,85 | 0,22 ms | 0 |
| Backtesting LinearRegression, igual | 417 | 1,093 → 0,990 | 0,91 | 0,25 ms | 0 |
| Backtesting Ridge, `lags=168`, igual | 417 | 2,555 → 2,127 | 0,83 | 1,03 ms | 0 |
| Backtesting LinearRegression, `lags=168`, igual | 417 | 8,207 → 7,710 | 0,94 | 1,19 ms | 0 |
| Backtesting Direct (`steps=24`) Ridge, `lags=24`, 2.500 de entrenamiento | 2.520 | 1,111 → 0,791 | 0,71 | 0,13 ms | 0 |
| Igual con `lags=168` | 2.520 | 3,475 → 2,427 | 0,70 | 0,42 ms | 0 |
| `grid_search_forecaster` Ridge, 10 `alpha` x `lags` [24, 168], `refit=True`, 42 folds | 840 | 0,707 → 0,564 | 0,80 | 0,17 ms | 0 |
| Igual, `refit=False`, 2.000 observaciones | 20 | 0,076 → 0,067 | 0,89 | 0,44 ms | 0 |
| Igual, `refit=False`, 20.000 observaciones | 20 | 0,456 → 0,444 | 0,97 | 0,64 ms | 0 |
| Igual, `OneStepAheadFold`, 2.000 observaciones | 20 | 0,021 → 0,015 | 0,69 | 0,33 ms | 1,9e-15 |
| Igual, `OneStepAheadFold`, 20.000 observaciones | 20 | 0,100 → 0,066 | 0,66 | 1,68 ms | 6,7e-16 |

\* Ruido: la diferencia es de 8 y 42 ms por `fit`, y la copia de 1.000 x 168 cuesta 0,06
ms. LightGBM iba lento por la carga (165 ms por fold con 50 árboles y 1.000 filas).

- Con `refit=False`, la búsqueda solo hace 20 `fit` y el tiempo se va en las predicciones,
  que no cambian.
- En `OneStepAheadFold` también gana el `predict` sobre `X_test`, que hoy es otra vista.
- Con 1.000 filas de entrenamiento y 24 lags, el `fit` de Ridge es demasiado barato para
  notar nada.

## 7. Memoria

Script 05: pico de `tracemalloc` en `ForecasterRecursive.fit`. Se toma el mínimo de 2
ejecuciones por versión, en MB. `tracemalloc` no ve las asignaciones en C/C++ de LightGBM,
que son iguales en las dos versiones; sí ve la copia del prototipo.

| Filas x lags (matriz) | LinearRegression | Ridge | Lasso | HistGradientBoosting | LightGBM |
|---|---|---|---|---|---|
| 20.000 x 168 (26 MB) | 52 → 55 (+3) | 52 → 52 (0) | 26 → 52 (+26) | 56 → 81 (+26) | 26 → 29 (+3) |
| 200.000 x 24 (37 MB) | 76 → 81 (+5) | 76 → 76 (0) | 41 → 78 (+37) | 87 → 122 (+35) | 40 → 43 (+3) |
| 200.000 x 168 (256 MB) | 516 → 548 (+32) | 516 → 516 (0) | 261 → 517 (+256) | 324 → 580 (+256) | 259 → 290 (+31) |

- **La vista no ocupa memoria:** comparte la de `y`. La copia ocupa la matriz entera.
- **Ridge y LinearRegression** ya copian la matriz dentro del `fit` (centrado o conversión
  para BLAS), así que el pico apenas sube: entre 0 y +32 MB con una matriz de 256 MB.
- **Lasso** (y ElasticNet, misma implementación, no medido aquí) copia a orden F por su
  cuenta (`copy_X=True`). Con la copia C previa conviven las dos y el pico sube una matriz
  entera.
- **HistGradientBoosting** no copia una matriz float64 C, así que la copia del forecaster
  es memoria nueva: +1 matriz.
- **Pipelines** con `StandardScaler`: no medidos. El escalador ya crea una matriz nueva,
  así que la copia previa suma como mucho una matriz al pico.
- **"La copia duplica la matriz de entrenamiento"** solo es cierto para los estimadores que
  no copian por su cuenta. Con los que se benefician, el coste es casi nulo.

## 8. Resultados: predicciones y tests

- **Predicciones puntuales.** Idénticas bit a bit en los 88 casos del script 03 (Recursive,
  Direct y Classifier) y en los 8 del script 08 (pipelines). Ridge y LinearRegression ya
  convertían la vista a C dentro del `fit` antes de llamar a BLAS, así que los coeficientes
  son los mismos.
- **Residuos in-sample e intervalos** (script 07, 20.000 filas, valores en torno a 50).
  Cambian en el último decimal con todos los lineales, Lasso incluido. En cambio, el
  `predict` sobre la vista y sobre la copia C suma en otro orden.

  | Forecaster | Estimador | Residuos in-sample | `predict_interval` bootstrapping | `predict_interval` conformal |
  |---|---|---|---|---|
  | Recursive | LinearRegression, Ridge, Lasso | 3e-14 a 1,9e-13 | 3e-14 a 2,2e-13 | 7e-15 a 1,4e-13 |
  | Direct (`steps=12`) | LinearRegression, Ridge, Lasso | 3e-14 a 2,0e-13 | 1e-14 a 6e-14 | 1e-14 a 4e-14 |
  | Recursive y Direct | HistGradientBoosting, LightGBM | 0 | 0 | 0 |

- **Métricas de búsqueda.** Con `OneStepAheadFold` cambian 1,9e-15 y 6,7e-16 (el `predict`
  sobre `X_test`). Con `TimeSeriesFold` no cambian.
- **Tests con el prototipo** (pytest secuencial, sin `-n`):

  | Directorio | Resultado |
  |---|---|
  | `recursive/tests/tests_forecaster_recursive` | 362 passed |
  | `recursive/tests/tests_forecaster_recursive_classifier` | 262 passed, 1 skipped |
  | `direct/tests/tests_forecaster_direct` | 451 passed |
  | `direct/tests/tests_forecaster_direct_multivariate` | 511 passed |
  | `model_selection/tests` (`-m "not slow"`) | 1.195 passed, 2 skipped, 2 deselected |

  **No falla ningún test.** Los valores fijos de residuos e intervalos se comparan con
  tolerancia (`assert_allclose`, `assert_frame_equal`), y cambios de 1e-13 sobre valores
  de orden 1 a 100 quedan dentro. No se ha corrido MultiSeries, porque el prototipo no lo
  toca.

## 9. Variante F (descartada)

El orden F es más barato de construir: cada columna de lags es un trozo contiguo de `y`, y
`np.asfortranarray` cuesta entre un 30 y un 65 % menos que la copia C (sección 3). Además,
en el script 06 rinde como C con los lineales y, en `predict`, como la vista con
HistGradientBoosting. Medido con el mismo prototipo (indicador `"F"`, script 03 con `F`,
`_03_fit_F.csv`):

- **Tiempo.** Ridge gana algo más que con C:
  - Recursive: de 0,98 a 0,63.
  - Direct: de 0,96 a 0,56.

  En el resto:
  - LinearRegression igual que con C.
  - Lasso, HistGradientBoosting, LightGBM y LogisticRegression, de 0,97 a 1,05.
  - HistGradientBoosting con 200.000 x 168 tarda 1,03, así que el orden F tampoco lo salva
    en el `fit` completo.
  - **XGBoost con 200.000 x 168 tarda 1,15:** reordena a C por su cuenta.
- **Predicciones.** Cambian en los 24 casos con LinearRegression y Ridge (Recursive y
  Direct), de 4e-14 a 1e-11. Con C eran idénticas. Es lo que advierte el comentario de MultiSeries.
- **Tests.** Pasan todos (2.781 passed, 3 skipped, 2 deselected), porque tienen tolerancia.
- **Memoria.** La misma que con C (script 05 con `F`).

Cambiar los decimales de los lineales para ganar lo mismo o algo más con Ridge, y perder
con XGBoost, no compensa.

## 10. Opciones

Los porcentajes son del `fit` completo (sección 4) y del backtesting (sección 6). La
memoria es la de la sección 7.

| Opción | Qué hace | Beneficio | Coste |
|---|---|---|---|
| A. No hacer nada | - | - | Ridge paga de un 23 a un 41 % más en el `fit` de Recursive desde 1.000 x 168, y hasta un 72 % en Direct (701 frente a 407 ms). LinearRegression, de un 3 a un 20 %. El backtesting y la búsqueda con lineales y muchos `fit`, de un 10 a un 52 % más. Memoria: 0. |
| B. Copiar siempre | `np.ascontiguousarray` cuando la matriz es un solo bloque (prototipo). En Direct y DMV, solo sin `exog` ni calendario. | Lo de D. | Copia que nadie aprovecha con los demás estimadores (de 0,01 a 25 ms por `fit`); HistGradientBoosting un 4 % más lento con 200.000 x 168; +1 matriz de pico con Lasso, ElasticNet y HistGradientBoosting (+256 MB con 200.000 x 168); residuos in-sample de los lineales con cambios de hasta 2e-13. Dos líneas por forecaster. |
| C. Construir los lags en orden C desde el principio (como MultiSeries) | Copiar en `_create_lags` | Lo de B. | Lo de B, más una copia doble cuando los lags se concatenan con `exog`, window features o calendario (hoy el `np.concatenate` ya da C), salvo que se limite al caso de un solo bloque, y entonces es B. No hay un método más barato que `ascontiguousarray` (sección 3). |
| D. Copiar solo cuando el estimador gana | B, pero solo si el estimador (o el primer paso del `Pipeline`, fase 2) es `LinearRegression`, `Ridge`, `Lars` o `LassoLars` (lista final de la fase 2; en la fase 1 se proponían `LinearRegression`, `Ridge` y `RidgeCV`) | Recursive: Ridge de un 3 a un 29 % menos, LinearRegression de un 3 a un 14 %. Direct: de un 7 a un 42 % y de un 3 a un 17 %. Pipelines: de un 4 a un 17 %. Backtesting y búsqueda con Ridge y muchos `fit` (417 folds, Direct, `refit=True`, `OneStepAheadFold`): de un 15 a un 34 % menos. | Memoria: entre 0 y +32 MB con una matriz de 256 MB. Residuos in-sample de esos estimadores con cambios de hasta 2e-13; las predicciones no cambian. Con el resto de estimadores, ningún cambio. Una función en `utils` y una llamada en 4 forecasters. |

**Recomendación: D.** Tiene sentido:

- La ganancia es real y estable en todos los tamaños con los modelos lineales más usados.
  Son también los que más se usan en búsquedas y backtesting con muchos `fit`, que es donde
  más se nota.
- Las predicciones no cambian y no falla ningún test.
- Es barato de implementar.

En valor absoluto es poco por `fit` (de 0,1 a 1 ms con tamaños habituales). Por eso es una
mejora de prioridad baja, no un arreglo: va con los PERF de la release, sin bloquear nada.

Detalles para la implementación (fase 1; sustituidos por la sección 16):

- **Criterio.** Lista cerrada de lo medido (`LinearRegression`, `Ridge`, `RidgeCV`, que
  usa el mismo cálculo). Hay que mirar el último paso del `Pipeline`, como
  `select_n_jobs_fit_forecaster`. **Corregido en la fase 2:** RidgeCV suma una matriz de
  pico, y el último paso de un Pipeline con escalador también; se mira el primer paso.
  - Las demás clases de `LinearModel` sin descenso por coordenadas (BayesianRidge, ARD,
    Lars, Huber, TheilSen, OMP, QuantileRegressor, RidgeClassifier) no están medidas.
    Se pueden añadir midiendo con el script 03.
  - Lasso, ElasticNet y sus variantes CV y MultiTask quedan fuera: no ganan tiempo y
    suman una matriz de memoria.
  - LogisticRegression queda fuera: copia por su cuenta.
- **Dónde.** En Recursive y Classifier, en la rama `len(X_train) == 1` de
  `_create_train_X_y`, antes de `_filter_nan_train_X_y`, para que `pd.isna` también vaya
  sobre la copia. En Direct y DMV, a `X_train_autoreg` cuando no hay `exog` ni calendario.
  Antes de `Parallel`: con `n_jobs > 1` la serialización ya copia, pero con lineales
  `n_jobs` es 1.
  - Si la matriz ya es C (concatenaciones, filas quitadas por NaN), `np.ascontiguousarray`
    no hace nada.
  - Cubre también `OneStepAheadFold` (`_train_test_split_one_step_ahead` usa
    `_create_train_X_y`) y el `X_test` de esa búsqueda.
- **Tests.**
  - Uno por forecaster que compruebe que la matriz que llega al `fit` es C con Ridge y
    que sigue siendo la vista con un árbol. Puede usar un estimador que registre la
    disposición, como el script 01.
  - Los tests con valores fijos no cambian (sección 8).
- **Release note.** "Changed" de rendimiento en `releases.md`, sin cambio de API.
- **AI context.** No cambia la API pública, así que no hace falta tocar `llms-base.txt` ni
  las skills.
- **MultiSeries.** No se toca: pasa F, que rinde como C con los lineales.

## 11. Fila propuesta para `roadmap.md` (no añadida)

Para la tabla de "Alcance" de la sección 0:

| Tema | Elementos | Notas |
|---|---|---|
| Rendimiento con modelos lineales | PERF-contiguous-lags | S. Nuevo (2026-11-08). Con solo lags, Recursive, Direct y DMV pasan al estimador una vista no contigua de `sliding_window_view`. Copiarla a orden C solo cuando el estimador (o el primer paso del `Pipeline`) es LinearRegression, Ridge, Lars o LassoLars reduce el `fit` con 20.000 filas entre un 9 y un 30 % en Recursive y entre un 14 y un 40 % en Direct (hasta un 43 % con 200.000), y el backtesting con `refit=True` y las búsquedas con estos modelos entre un 6 y un 33 %. Las predicciones no cambian (residuos in-sample: menos de 2e-13) y la memoria pico sube un 12 % de la matriz como mucho (solo con LinearRegression). Con los demás estimadores no se copia. Una lista y una función privada en `utils`, y dos líneas en tres forecasters. Medición, lista y cambios en [`PERF-contiguous-lags.md`](PERF-contiguous-lags.md), secciones 14 a 16 |

## 12. Prototipo

`repro/PERF-contiguous-lags_prototype.diff` (+138 líneas: los 4 forecasters y `utils`), no
aplicado en la rama. Añade el indicador de módulo `_CONTIGUOUS_X_TRAIN`:

- `"D"` (por defecto en el parche, fase 2): opción D. Usa la lista y la función de la
  sección 16, tal como se proponen.
- `True`: copia C siempre.
- `"F"`: copia F.
- `False`: camino de `0.27.x`.

En el parche, el modo D también está en Classifier, para medir RidgeClassifier. Con la
lista final no se copia con ningún clasificador.

Es para medir, no para fusionar: la opción D no usaría un indicador de módulo, sino el
criterio del estimador. No se ha creado ningún worktree temporal para comparar: las dos
versiones se alternan en el mismo proceso.

## 13. Fase 2: condiciones de medición

Las mismas versiones que en la fase 1: Python 3.14.3, numpy 2.5.4 (Accelerate), pandas 2.3.3,
scikit-learn 1.9.1, scipy 1.18.1, LightGBM 4.6.0, XGBoost 3.2.0 y CatBoost 1.2.10, en un Apple
M5 con macOS 27.2.

- **Carga.** Antes de cada fila, los scripts 09 a 11 esperan a que la carga de 1 minuto
  baje de 4 (`wait_for_load` en `repro/PERF-contiguous-lags_common.py`). Si no baja en 30
  minutos, se paran sin medir. La carga queda anotada en cada fila de los CSV: entre 2,1 y
  4,0 al empezar cada medición.
- **Alimentación.** Equipo enchufado (`pmset -g batt`: AC) en todas las filas.
- **La otra sesión.** Siguió trabajando en el repositorio principal, pero sin procesos
  pesados durante las mediciones.
- **Método.** El mismo que en la fase 1: las dos versiones alternan en el mismo proceso y
  se toma el mínimo de entre 2 y 7 rondas. Las medianas están en los CSV.
  - La memoria pico se mide en una pasada aparte con `tracemalloc`.
  - Direct con estimadores que no son `LinearModel` entrena en procesos de joblib, que
    `tracemalloc` no ve. Su memoria solo cubre el proceso principal.

Scripts nuevos (en `repro/`; aplicar antes el parche del prototipo):

| Script | Qué mide |
|---|---|
| `PERF-contiguous-lags_common.py` | Funciones comunes: espera de carga, entorno, `run_case`, memoria pico |
| `PERF-contiguous-lags_09_more_estimators.py` | `fit` completo con 16 estimadores más (copia siempre frente a `0.27.x`); `_09_more_estimators.csv` |
| `PERF-contiguous-lags_10_memory_and_iterations.py` | `memory`: memoria pico de los candidatos; `iterations`: iteraciones y convergencia de Huber, Poisson y Tweedie |
| `PERF-contiguous-lags_11_option_d.py` | Opción D frente a `0.27.x`: filas de la tabla comparativa; `_11_option_d.csv` |

## 14. Lista de estimadores

### 14.1 `fit` completo con copia siempre (script 09)

Tiempo en ms, de `0.27.x` → con copia C, y entre paréntesis `C / 0.27.x` (mínimos).
Direct usa `steps=5` y `n_jobs='auto'`. Con `n_jobs=9`, los pasos se entrenan en procesos
de joblib, que ya reciben una copia C en las dos versiones.

**ForecasterRecursive**

| Estimador | 20.000 x 24 | 20.000 x 168 | 200.000 x 168 | Predicciones |
|---|---|---|---|---|
| BayesianRidge | 5,43 → 5,05 (0,93) | 61,3 → 57,2 (0,93) | 1.054 → 1.062 (1,01) | idénticas |
| ARDRegression | 12,6 → 12,2 (0,97) | 44,4 → 40,4 (0,91) | 332 → 301 (0,91) | idénticas |
| HuberRegressor | 116 → 97,4 (0,84) | 700 → 512 (0,73) | 8.549 → 6.998 (0,82) | cambian hasta 0,71 |
| SGDRegressor | 228 → 229 (1,00) | 867 → 872 (1,01) | 19.479 → 19.475 (1,00) | idénticas |
| PoissonRegressor | 77,7 → 28,2 (0,36) | 552 → 107 (0,19) | 5.229 → 964 (0,18) | cambian hasta 16 |
| TweedieRegressor | 80,2 → 27,7 (0,34) | 579 → 114 (0,20) | 5.676 → 1.025 (0,18) | cambian hasta 1,2 |
| LinearSVR | 672 → 673 (1,00) | 6.066 → 6.074 (1,00) | 71.138 → 71.428 (1,00) | idénticas |
| KernelRidge | 13.584 → 14.247 (1,05) | - | - | cambian hasta 4,3e-8 |
| Lars | 3,35 → 2,63 (0,78) | 23,9 → 19,7 (0,82) | 155 → 115 (0,74) | idénticas |
| LassoLars | 3,04 → 2,37 (0,78) | 15,5 → 11,1 (0,72) | 145 → 105 (0,73) | idénticas |
| QuantileRegressor (`solver="highs"`) | 12.802 → 12.799 (1,00) | - | - | idénticas |
| RidgeCV | 5,27 → 4,76 (0,90) | 36,4 → 32,2 (0,89) | 381 → 356 (0,93) | idénticas |
| LinearRegression(positive=True) | 4,65 → 4,07 (0,87) | 177 → 170 (0,96) | 1.658 → 1.630 (0,98) | idénticas |

**ForecasterDirect** (`steps=5`)

| Estimador | `n_jobs` | 20.000 x 24 | 20.000 x 168 | 200.000 x 168 | Predicciones |
|---|---|---|---|---|---|
| BayesianRidge | 1 | 25,9 → 22,7 (0,88) | 291 → 265 (0,91) | 5.081 → 4.844 (0,95) | idénticas |
| ARDRegression | 1 | 69,9 → 68,0 (0,97) | 261 → 233 (0,89) | 2.773 → 2.561 (0,92) | idénticas |
| HuberRegressor | 1 | 578 → 480 (0,83) | 3.346 → 2.381 (0,71) | 42.013 → 32.632 (0,78) | cambian hasta 0,066 |
| SGDRegressor | 9 | 495 → 477 (0,96) | 902 → 923 (1,02) | 33.684 → 33.583 (1,00) | idénticas |
| PoissonRegressor | 9 | 165 → 167 (1,01) | 314 → 315 (1,00) | 2.897 → 2.556 (0,88) | idénticas |
| TweedieRegressor | 9 | 169 → 158 (0,93) | 303 → 303 (1,00) | 2.936 → 2.876 (0,98) | idénticas |
| LinearSVR | 1 | 3.347 → 3.342 (1,00) | 30.720 → 30.569 (1,00) | - | idénticas |
| Lars | 1 | 15,0 → 11,8 (0,79) | 106 → 79,2 (0,75) | 740 → 438 (0,59) | idénticas |
| LassoLars | 1 | 14,1 → 10,5 (0,74) | 72,8 → 47,2 (0,65) | 684 → 399 (0,58) | idénticas |
| QuantileRegressor (`solver="highs"`) | 1 | 64.030 → 63.836 (1,00) | - | - | idénticas |
| RidgeCV | 1 | 30,8 → 27,3 (0,89) | 191 → 165 (0,86) | 1.889 → 1.712 (0,91) | idénticas |
| LinearRegression(positive=True) | 1 | 27,1 → 24,1 (0,89) | 932 → 907 (0,97) | 8.144 → 7.986 (0,98) | idénticas |

**ForecasterRecursiveClassifier** (se comparan las clases predichas; RidgeClassifier y
LinearSVC no tienen `predict_proba`)

| Estimador | 20.000 x 24 | 20.000 x 168 | 200.000 x 168 | Predicciones |
|---|---|---|---|---|
| RidgeClassifier | 4,70 → 4,56 (0,97) | 16,7 → 15,7 (0,94) | 185 → 175 (0,95) | idénticas |
| LinearSVC | 55,2 → 55,4 (1,00) | 517 → 513 (0,99) | 11.055 → 11.104 (1,00) | idénticas |
| SGDClassifier | 117 → 116 (0,99) | 663 → 651 (0,98) | 3.324 → 3.346 (1,01) | idénticas |

Tamaños que faltan:

- KernelRidge, solo 20.000 x 24 y solo en Recursive. Su núcleo de 20.000 x 20.000 ocupa
  3,2 GB, y Direct lo entrenaría en 9 procesos a la vez.
- QuantileRegressor, solo 20.000 x 24. Tarda 13 s por `fit` en Recursive y 64 s en Direct,
  y no cambia.
- LinearSVR, sin 200.000 x 168 en Direct, porque su primer `fit` con 20.000 x 168 tardó más
  de 30 s.

### 14.2 Huber, Poisson y Tweedie: por qué cambian las predicciones (script 10)

Recursive, `predict(steps=24)`. Las dos versiones hacen el mismo número de iteraciones.

| Estimador | Filas x lags | Iteraciones (0.27.x / C) | Converge | ms por iteración (0.27.x / C) | Diferencia máxima |
|---|---|---|---|---|---|
| HuberRegressor | 20.000 x 24 | 100 / 100 | no / no | 1,15 / 1,00 | 0,060 |
| HuberRegressor | 20.000 x 168 | 100 / 100 | no / no | 7,07 / 5,47 | 0,36 |
| PoissonRegressor | 20.000 x 24 | 100 / 100 | no / no | 0,76 / 0,28 | 16 |
| PoissonRegressor | 20.000 x 168 | 100 / 100 | no / no | 5,32 / 1,08 | 0,32 |
| TweedieRegressor | 20.000 x 24 | 100 / 100 | no / no | 0,78 / 0,29 | 1,2 |
| TweedieRegressor | 20.000 x 168 | 100 / 100 | no / no | 5,78 / 1,18 | 0,043 |
| `Pipeline(StandardScaler, Huber)` | 20.000 x 24 / x 168 | 100 / 100 | no / no | 1,02 / 1,05 y 5,50 / 5,55 | 0 |
| `Pipeline(StandardScaler, Poisson)` | 20.000 x 24 / x 168 | 52 / 52 y 100 / 100 | sí y no | 0,37 / 0,35 y 1,26 / 1,22 | 0 |
| `Pipeline(StandardScaler, Tweedie)` | 20.000 x 24 / x 168 | 12 / 12 y 18 / 18 | sí | 0,69 / 0,60 y 2,41 / 2,12 | 0 |

- **La ganancia es real.** Con el mismo número de iteraciones, cada iteración tarda entre
  1,2 veces menos (Huber) y 4,9 veces menos (Poisson y Tweedie) con la copia C. lbfgs
  calcula `X @ coef` y `X.T @ g` en cada iteración, y con la vista numpy copia `X` cada
  vez.
- **Las predicciones cambian porque no converge.** Sin escalar (lags en torno a 50), lbfgs
  se para en `max_iter=100` con un `ConvergenceWarning`. La diferencia de redondeo entre la
  vista y la copia C se acumula durante 100 iteraciones de un problema mal condicionado.
- **Con un `StandardScaler` delante,** que es lo habitual con estos modelos, el estimador
  recibe una matriz nueva en las dos versiones. Ni gana tiempo ni cambia nada.
- **No entran en la lista.** Cambiarían resultados (los de un ajuste que no converge, pero
  visibles: hasta 16 unidades sobre valores de 50) y suman una matriz de pico (14.3). Podrían
  entrar más adelante con un aviso o una nota propia; queda fuera de este cambio.

### 14.3 Memoria pico de los candidatos (script 10, `memory`)

Pico de `tracemalloc` en `fit`, `0.27.x` → copia C, y en la última columna el aumento como
porcentaje de la matriz de entrenamiento (26 MB con 20.000 x 168; 256 MB con 200.000 x 168).
Se dan los picos de Recursive; Direct da el mismo aumento, con una base de hasta 12 MB más.

| Estimador | 20.000 x 168 (MB) | 200.000 x 168 (MB) | Aumento / matriz |
|---|---|---|---|
| Ridge | 52 → 52 | 516 → 516 | 0 % |
| Lars | 53 → 53 | 517 → 517 | 0 % |
| LassoLars | 53 → 53 | 517 → 517 | 0 % |
| RidgeClassifier (Classifier) | 79 → 79 | 789 → 789 | 0 % |
| LinearRegression (también `positive=True`) | 52 → 55 | 516 → 548 | +12 % |
| RidgeCV | 79 → 104 | 783 → 1.040 | +100 % |
| BayesianRidge | 52 → 78 | 519 → 775 | +100 % |
| ARDRegression | 53 → 79 | 517 → 773 | +100 % |
| `Pipeline(StandardScaler, Ridge)` | 52 → 77 | 516 → 772 | +100 % |
| `Pipeline(StandardScaler, LinearRegression)` | 55 → 80 | 548 → 804 | +100 % |
| HuberRegressor | 41 → 66 | 408 → 664 | +100 % |
| PoissonRegressor, TweedieRegressor (solo Recursive) | 3 → 29 | 34 → 290 | +100 % |

Con Ridge, Lars y LassoLars, la copia previa no sube el pico: la copia que hacían ellos
mismos (la conversión de la vista en cada producto) era temporal y del mismo tamaño. Con
RidgeCV, BayesianRidge y ARD el pico sube una matriz entera. La explicación probable es que
mantienen durante el ajuste otra matriz del tamaño de `X` (la copia centrada o la
descomposición), y la copia previa convive con ella. No se ha comprobado línea a línea en
scikit-learn.

### 14.4 Decisión

Criterio: gana más de un 5 % en el `fit` completo en al menos dos de los tres tamaños y no
sube la memoria pico más de un 15 % de la matriz. También se exige que las predicciones no
cambien, porque la opción D promete no cambiar resultados.

| Estimador | Gana > 5 % en dos tamaños | Memoria | Predicciones | Entra |
|---|---|---|---|---|
| Ridge | sí (fase 1 y sección 15) | 0 % | idénticas | **sí** |
| LinearRegression | sí | +12 % | idénticas | **sí** |
| Lars | sí (Recursive y Direct) | 0 % | idénticas | **sí** |
| LassoLars | sí (Recursive y Direct) | 0 % | idénticas | **sí** |
| RidgeClassifier | 0,94 y 0,95 en el script 09; 0,95, 0,95 y 0,99 en limpio (sección 15) | 0 % | idénticas | no: ganancia inestable, en el límite del 5 % |
| RidgeCV, BayesianRidge, ARDRegression | sí | +100 % | idénticas | no: memoria |
| HuberRegressor, PoissonRegressor, TweedieRegressor | sí (Recursive) | +100 % | cambian | no: memoria y resultados |
| Pipelines que empiezan por un transformador | sí (de un 5 a un 17 %) | +100 % | idénticas | no: memoria |
| SGDRegressor, LinearSVR, KernelRidge, QuantileRegressor, LinearSVC, SGDClassifier | no | - | - | no |
| Lasso, ElasticNet, LogisticRegression, árboles, boosting (fase 1) | no | hasta +100 % | - | no |

`LinearRegression(positive=True)` solo gana más de un 5 % con 20.000 x 24 (0,87 y 0,89).
Pero la lista va por clase, no por parámetros, y la copia le cuesta lo mismo que a
`LinearRegression`: +12 % de memoria y ninguna pérdida de tiempo. Se copia igual que en el
caso habitual; no merece una excepción por parámetro.

## 15. Opción D frente a `0.27.x` (condiciones limpias, script 11)

Prototipo en el modo `"D"`, con la lista final y la regla del primer paso del Pipeline.
`fit` en ms, `0.27.x` → opción D, con la ganancia entre paréntesis. La memoria es la pico
de `tracemalloc` en MB, `0.27.x` → opción D, en 20.000 x 24 / 20.000 x 168 / 200.000 x 168.
Las predicciones son idénticas bit a bit en todas las filas de `fit` y backtesting.

| Forecaster | Estimador | 20.000 x 24 | 20.000 x 168 | 200.000 x 168 | Memoria pico (MB) |
|---|---|---|---|---|---|
| Recursive | Ridge | 2,28 → 1,74 (−24 %) | 14,0 → 9,84 (−30 %) | 148 → 111 (−25 %) | 8 → 8 / 52 → 52 / 516 → 516 |
| Recursive | LinearRegression | 4,41 → 3,87 (−12 %) | 42,5 → 38,6 (−9 %) | 672 → 642 (−4 %) | 8 → 8 / 52 → 55 / 516 → 548 |
| Recursive | Lars | 3,15 → 2,58 (−18 %) | 23,6 → 19,3 (−18 %) | 159 → 120 (−25 %) | 8 → 8 / 53 → 53 / 517 → 517 |
| Recursive | LassoLars | 2,83 → 2,29 (−19 %) | 15,3 → 11,1 (−28 %) | 149 → 108 (−27 %) | 8 → 8 / 53 → 53 / 517 → 517 |
| Direct (5 pasos) | Ridge | 11,4 → 8,20 (−28 %) | 64,1 → 38,4 (−40 %) | 683 → 387 (−43 %) | 8 → 8 / 52 → 52 / 522 → 522 |
| Direct (5 pasos) | LinearRegression | 22,1 → 18,9 (−15 %) | 198 → 170 (−14 %) | 3.214 → 2.989 (−7 %) | 9 → 9 / 53 → 56 / 528 → 560 |
| Direct (5 pasos) | Lars | 15,0 → 11,7 (−22 %) | 106 → 79,1 (−25 %) | 733 → 443 (−40 %) | 8 → 8 / 56 → 56 / 525 → 525 |
| Direct (5 pasos) | LassoLars | 13,8 → 10,4 (−25 %) | 72,5 → 47,5 (−35 %) | 702 → 399 (−43 %) | 8 → 8 / 54 → 54 / 524 → 524 |
| Recursive | `Pipeline(StandardScaler, Ridge)` (no se copia) | 5,33 → 5,29 (−1 %) | 29,5 → 29,3 (−1 %) | 319 → 318 (0 %) | sin cambio |
| Recursive | `Pipeline(StandardScaler, LinearRegression)` (no se copia) | 7,34 → 7,36 (0 %) | 57,9 → 58,5 (+1 %) | 844 → 852 (+1 %) | sin cambio |
| Recursive | HistGradientBoosting (control) | - | 1.571 → 1.571 (0 %) | 3.731 → 3.761 (+1 %) | 56 → 56 / 324 → 324 |
| Recursive | LightGBM (control) | - | 623 → 624 (0 %) | 2.811 → 2.807 (0 %) | 26 → 26 / 259 → 259 |
| Direct (5 pasos) | HistGradientBoosting (control, 9 procesos) | - | 1.819 → 1.821 (0 %) | 14.784 → 14.893 (+1 %) | 32 → 32 / 49 → 49 (solo el proceso principal) |
| Direct (5 pasos) | LightGBM (control) | - | 3.207 → 3.212 (0 %) | 14.925 → 14.964 (0 %) | 27 → 27 / 266 → 266 |

Ridge y LinearRegression con los otros tres tamaños, en ms:

| Forecaster | Estimador | 1.000 x 24 | 1.000 x 168 | 200.000 x 24 |
|---|---|---|---|---|
| Recursive | Ridge | 0,54 → 0,53 (−1 %) | 1,03 → 0,79 (−24 %) | 22,3 → 17,8 (−20 %) |
| Recursive | LinearRegression | 0,57 → 0,54 (−5 %) | 3,91 → 3,69 (−6 %) | 45,9 → 41,6 (−9 %) |
| Direct (5 pasos) | Ridge | 1,98 → 1,77 (−11 %) | 4,77 → 3,30 (−31 %) | 89,8 → 59,9 (−33 %) |
| Direct (5 pasos) | LinearRegression | 2,67 → 2,48 (−7 %) | 19,1 → 17,9 (−7 %) | 212 → 181 (−15 %) |

En los Pipelines, la opción D no copia. Con la copia siempre, los Pipelines ganarían entre
un 5 y un 17 % a cambio de una matriz más de pico:

- Ridge: −17 %, −16 % y −12 %.
- LinearRegression: −12 %, −7 % y −5 %.
- Pico: de 8 a 11 MB, de 52 a 77 MB y de 516 a 772 MB.

Backtesting y búsquedas (muchos `fit`), en s:

| Caso | `fit` | `0.27.x` → opción D | Ganancia | Memoria pico (MB) | Diferencia máxima |
|---|---|---|---|---|---|
| Backtesting Recursive Ridge, `lags=24`, 10.000 de entrenamiento, 417 folds | 417 | 0,577 → 0,466 | −19 % | 6 → 6 | 0 |
| Igual, `lags=168` | 417 | 2,264 → 1,896 | −16 % | 28 → 28 | 0 |
| Backtesting Recursive LinearRegression, `lags=24`, 417 folds | 417 | 1,037 → 0,949 | −8 % | 6 → 7 | 0 |
| Igual, `lags=168` | 417 | 8,158 → 7,702 | −6 % | 28 → 29 | 0 |
| Backtesting Direct (`steps=24`) Ridge, `lags=24`, 2.500 de entrenamiento, 105 folds | 2.520 | 1,001 → 0,769 | −23 % | 2 → 2 | 0 |
| Igual, `lags=168` | 2.520 | 3,460 → 2,319 | −33 % | 8 → 8 | 0 |
| `grid_search_forecaster` Ridge, 10 `alpha` x `lags` [24, 168], `refit=True`, 42 folds | 840 | 0,630 → 0,539 | −14 % | 3 → 3 | 0 |
| Igual, `OneStepAheadFold`, 2.000 observaciones | 20 | 0,019 → 0,015 | −23 % | 4 → 4 | 1,9e-15 |
| Igual, `OneStepAheadFold`, 20.000 observaciones | 20 | 0,098 → 0,066 | −33 % | 39 → 45 | 6,7e-16 |
| Backtesting Recursive LightGBM (control), 84 folds | 84 | 13,37 → 13,39 | 0 % | 3 → 3 | 0 |

**Usuario típico** (la configuración más habitual de la documentación: Ridge o
LinearRegression, `lags` entre 24 y 168, de 10.000 a 50.000 filas):

- Un `fit` de Recursive tarda entre un 24 y un 30 % menos con Ridge y entre un 9 y un 12 %
  menos con LinearRegression. En Direct, entre un 14 y un 40 %.
- En un solo `fit` son milisegundos (de 0,5 a 4 ms en Recursive): no se nota.
- Se nota en lo que repite el `fit`. Con 417 folds y `refit=True`, el backtesting tarda
  entre un 6 y un 19 % menos. Con Direct de 24 pasos, entre un 23 y un 33 % menos (hasta
  1,1 s de 3,5). Las búsquedas tardan entre un 14 y un 33 % menos.

**Recomendación: para 0.27.0.** Es un cambio S, aislado (una lista y una función privada en
`utils`, y dos líneas en tres forecasters). No cambia la API ni las predicciones, y no
cuesta memoria con los estimadores de la lista. Encaja con las mejoras de rendimiento que
ya lleva la release (#1460, #1462 y #1464). Si el alcance de 0.27.0 ya está cerrado, puede
ir en la siguiente sin coste: nada depende de él.

## 16. Qué cambia en el código (opción D, sin aplicar)

El prototipo en el modo `"D"` (`repro/PERF-contiguous-lags_prototype.diff`) usa exactamente
la lista y la función de abajo. La implementación quita el indicador de módulo y deja la
llamada directa. Las líneas son las de `0.27.x` (17219ad65).

### 16.1 La lista y la función (`skforecast/utils/utils.py`)

La lista va junto a `_SKLEARN_NAN_TOLERANT_ESTIMATORS` (línea 60), con el mismo estilo:
`frozenset` de nombres de clase y un comentario que explica el criterio.

```python
# sklearn estimators whose `fit` (and the `predict` of the in-sample residuals)
# is faster when the training matrix is C-contiguous. With only lags, the
# forecasters build that matrix as a view of `sliding_window_view` (neither C
# nor F contiguous), which these estimators copy again in every product with
# `X`, and free each copy, so one copy before the fit adds little memory.
# Criterion: at least 5 % faster `fit` in two of three sizes, extra peak memory
# up to 15 % of the training matrix and identical predictions.
_SKLEARN_C_CONTIGUOUS_ESTIMATORS = frozenset({
    'Lars',
    'LassoLars',
    'LinearRegression',
    'Ridge',
})
```

La función va junto a `estimator_has_native_nan_support` (línea 5320). Es privada (no es API
pública) y se exporta, como las demás privadas, en la lista explícita de
`skforecast/utils/__init__.py`.

```python
def _estimator_benefits_from_c_contiguous_X(estimator: object) -> bool:
    """
    Check whether the training matrix should be copied to a C-contiguous array
    before it is passed to the estimator.

    With only lags (or only window features), the training matrix is a view of
    `sliding_window_view` that is neither C nor F contiguous. Some linear
    models of scikit-learn copy that view again in every product with `X`
    (`fit` and the `predict` of the in-sample residuals), so one copy before
    the fit is faster. Other estimators already copy or convert the matrix
    (trees, boosting, coordinate descent) and the copy would only add memory.

    Parameters
    ----------
    estimator : object
        Estimator object. If the estimator is a Pipeline, its first step is
        used, since it is the step that receives the training matrix.

    Returns
    -------
    benefits : bool
        `True` if the estimator (or the first step of a Pipeline) is one of the
        scikit-learn estimators in `_SKLEARN_C_CONTIGUOUS_ESTIMATORS`,
        otherwise `False`.

    Notes
    -----
    Detection is based on the class name and the module (`sklearn`), as in
    `estimator_has_native_nan_support`. Subclasses defined outside scikit-learn
    and other meta-estimators (for example `TransformedTargetRegressor`) are
    not detected, and the matrix is passed as it is. A Pipeline that starts
    with a transformer (for example `StandardScaler`) is not copied either: the
    transformer returns a new array, so the copy would only add memory.

    """

    if isinstance(estimator, Pipeline):
        estimator = estimator[0]

    module = type(estimator).__module__.split('.')[0]

    return (
        module == 'sklearn'
        and type(estimator).__name__ in _SKLEARN_C_CONTIGUOUS_ESTIMATORS
    )
```

- **Identificación.** Por nombre de clase y módulo `sklearn`, como
  `estimator_has_native_nan_support`, sin importar las clases.
  - Una subclase definida fuera de scikit-learn no se copia: es la opción conservadora.
  - Se diferencia de las reglas de `n_jobs='auto'` (544ba97f4), que usan
    `isinstance(estimator, LinearModel)`. Allí la familia entera se comporta igual. Aquí
    no: Lasso, ElasticNet, RidgeCV, BayesianRidge y ARD son `LinearModel` y no deben
    copiarse.
- **Pipeline: el primer paso, no el último.** Las reglas de `n_jobs` y de NaN miran el
  último paso, porque dependen de qué modelo se entrena. Aquí importa quién recibe la
  matriz. Con `StandardScaler` delante, mirar el último paso sumaría una matriz de pico
  (+100 %, sección 15).
- **Otros meta-estimadores** (`TransformedTargetRegressor`, `MultiOutputRegressor`,
  `GridSearchCV`, `VotingRegressor`, `StackingRegressor`): no se copia.
  `TransformedTargetRegressor(regressor=Ridge())` pasaría `X` sin cambiar a Ridge y
  ganaría, pero no está medido y la regla de NaN tampoco lo desenvuelve. Se puede añadir
  después.
- **Organización de `utils`.** La función no depende de los avisos ni de un posible módulo
  privado de avisos.
  - Es una comprobación de capacidades del estimador, como
    `estimator_has_native_nan_support` y `select_n_jobs_fit_forecaster`, y si `utils` se
    reparte en módulos debería ir con ellas.
  - Solo habría que cambiar el `from ..utils import` de los tres forecasters y la lista de
    `utils/__init__.py`. No afecta a lo que esté revisando la otra sesión.

### 16.2 Los forecasters

En los tres, la llamada va en `_create_train_X_y`. Así también la usan
`create_train_X_y`, `_train_test_split_one_step_ahead` (las búsquedas con
`OneStepAheadFold`, en `X_train` y `X_test`) y `set_in_sample_residuals`.

**ForecasterRecursive** (`_forecaster_recursive.py`, líneas 980 a 983, antes de
`_filter_nan_train_X_y` en la línea 989):

```python
        if len(X_train) == 1:
            X_train = X_train[0]
            if _estimator_benefits_from_c_contiguous_X(self.estimator):
                # Only lags (or only window features): a view of
                # `sliding_window_view`, neither C nor F contiguous.
                X_train = np.ascontiguousarray(X_train)
        else:
            X_train = np.concatenate(X_train, axis=1)
```

- **Antes del filtrado de NaN.** Así el `pd.isna` de `_filter_nan_train_X_y` va sobre la
  copia, 4 veces más rápido (0,045 frente a 0,19 ms con 20.000 x 24, sección 3). Ese es
  el caso habitual, sin NaN.
  - Con NaN y filas quitadas, el filtrado vuelve a copiar (la indexación booleana) y se
    hacen dos copias en vez de una. Es el caso raro, y cuesta una copia más (0,3 ms con
    20.000 x 24).
  - Ponerla después tendría el coste contrario: el `pd.isna` iría siempre sobre la vista.
- **`exog`, window features con lags y calendario.** Son varios bloques y `np.concatenate`
  ya da una matriz C. La función ni se llama.
- **Solo window features.** También es una vista no contigua (sección 2) y se copia igual.
- **`differentiation` y `transformer_y`.** `y_values` es un array nuevo, pero los lags
  siguen siendo una vista de él y se copian igual. No cambia nada más.

**ForecasterDirect** (`_forecaster_direct.py`). Detrás del bloque de calendario (líneas 1211
y siguientes) y antes del comentario `# NOTE: NaN rows are counted per step` (línea 1246) y
de `_warn_nan_training_rows_direct` (línea 1251):

```python
        if (
            X_train_exog is None
            and X_train_calendar is None
            and _estimator_benefits_from_c_contiguous_X(self.estimator)
        ):
            # Every step uses `X_train_autoreg` as it is (no `np.concatenate`),
            # so one copy serves all the steps.
            X_train_autoreg = np.ascontiguousarray(X_train_autoreg)
```

- **Por qué ahí y no en la rama `len(X_train_autoreg) == 1` (línea 1188).** Hasta
  después no se sabe si hay `exog` o calendario.
  - Con `exog` o calendario, `_create_train_X_y_step` concatena por paso y la matriz de
    cada paso ya es C: copiar `X_train_autoreg` sería una copia perdida.
  - Sin ellos, todos los pasos usan `X_train_autoreg` tal cual. Una copia sirve a todos,
    y por eso Direct gana más que Recursive.
- **Antes de `_warn_nan_training_rows_direct`,** para que su `pd.isna` vaya sobre la copia.
- **`n_jobs`.** Los cuatro estimadores de la lista son `LinearModel`, así que `n_jobs='auto'`
  da 1 y no hay serialización. Con `n_jobs > 1` explícito, joblib recibiría la copia C (no
  medido; hoy recibe la vista y la copia al serializarla).
- **NaN.** `_filter_nan_X_y_step` filtra por paso y copia solo si quita filas.

**ForecasterDirectMultiVariate** (`_forecaster_direct_multivariate.py`): el mismo bloque,
detrás del calendario (línea 1324 y siguientes) y antes del comentario de la línea 1359 y
de `_warn_nan_training_rows_direct` (línea 1364).

- **Cuándo actúa.** Solo cuando `X_train_autoreg` es un único bloque, es decir, con lags
  en una sola serie y sin window features (sección 2).
- **En los demás casos** los lags de cada serie se concatenan, la matriz ya es C y
  `np.ascontiguousarray` no hace nada. Basta la misma condición.

**ForecasterRecursiveClassifier: sin cambios.** Ningún clasificador cumple el criterio:

- RidgeClassifier gana un 5 % o menos;
- LogisticRegression ya copia la matriz a C por su cuenta;
- LinearSVC y SGDClassifier no ganan.

Si un clasificador entra en la lista, el cambio es el mismo que en Recursive (líneas 968 a
971 del clasificador).

**ForecasterRecursiveMultiSeries: sin cambios.** Pasa un DataFrame en orden F, que rinde
como C con los lineales (sección 5).

### 16.3 Qué ve el usuario

- **API.** No cambian los parámetros, los atributos ni los tipos devueltos.
- **Predicciones.** Idénticas bit a bit (88 casos en la fase 1 y todas las filas de la
  sección 15).
- **Residuos in-sample** (`in_sample_residuals_` e `in_sample_residuals_by_bin_`) y los
  intervalos que salen de ellos (bootstrapping y conformal). Con los estimadores de la
  lista cambian entre 1e-14 y 2e-13 sobre valores en torno a 50 (sección 8): el `predict`
  sobre la copia C suma en otro orden. Las métricas de las búsquedas con `OneStepAheadFold`
  cambian como mucho 1,9e-15.
- **`create_train_X_y`.** Devuelve los mismos valores. Con los estimadores de la lista, el
  DataFrame tiene su propia copia en orden C. Hoy comparte memoria con los valores de `y`:
  es un DataFrame sobre la vista de `sliding_window_view`, y lo comprueba
  `np.shares_memory`. Nadie debería depender de eso, pero modificar ese DataFrame ya no
  toca los datos de `y`.
- **Nota de la versión.** Sí, en `docs/releases/releases.md`, sección 0.27.0, en
  **Changed**, junto a las de rendimiento (#1460 y #1462). Texto propuesto:

  > + <code>[ForecasterRecursive]</code>, <code>[ForecasterDirect]</code> and
  > <code>[ForecasterDirectMultiVariate]</code> fit `LinearRegression`, `Ridge`, `Lars`
  > and `LassoLars` faster when the predictors are only lags: the training matrix is passed
  > as a C-contiguous array instead of a non-contiguous view. `fit` takes 9 to 30% less time
  > in Recursive and 14 to 40% less in Direct (20,000 rows, 24 or 168 lags), and backtesting
  > with `refit=True` 6 to 33% less; predictions are unchanged and in-sample residuals can
  > differ in the last decimals. ([#XXXX](https://github.com/skforecast/skforecast/pull/XXXX))

- **AI context.** No hace falta tocar `llms-base.txt` ni las skills, porque no cambia la
  API pública.

### 16.4 Tests

Nuevos:

- **`skforecast/utils/tests/tests_utils/test_estimator_benefits_from_c_contiguous_X.py`**,
  parametrizado.
  - Devuelven `True`: `Ridge`, `LinearRegression`, `LinearRegression(positive=True)`,
    `Lars`, `LassoLars` y `Pipeline([("ridge", Ridge())])`.
  - Devuelven `False`:
    - `Lasso`, `ElasticNet`, `RidgeCV`, `BayesianRidge`, `ARDRegression`,
      `HuberRegressor`, `PoissonRegressor`, `RidgeClassifier` y `LogisticRegression`;
    - `HistGradientBoostingRegressor` y `LGBMRegressor`;
    - `make_pipeline(StandardScaler(), Ridge())`;
    - `TransformedTargetRegressor(regressor=Ridge())`;
    - una subclase de `Ridge` definida en el test (módulo distinto de `sklearn`).
- **Disposición de la matriz**, en `test_create_train_X_y.py` de Recursive, Direct y DMV.
  Se miran los flags del `X_train` de `_create_train_X_y`, que es lo que el `fit` pasa al
  estimador.
  - Solo lags:
    - con `Ridge`, `X_train.flags.c_contiguous` es `True`;
    - con `HistGradientBoostingRegressor`, es `False` y la matriz no es dueña de sus datos
      (la vista);
    - con `make_pipeline(StandardScaler(), Ridge())`, también `False`.
  - Con `exog`: C en los dos casos.
  - En Direct, la matriz de cada paso.
  - En DMV, lags en una sola serie.
  - `differentiation` y `transformer_y`: con Ridge siguen saliendo C.
- **Predicciones iguales,** uno por forecaster en `test_fit.py`.
  - Se ajusta el mismo forecaster con Ridge dos veces: una normal y otra con
    `_estimator_benefits_from_c_contiguous_X` parcheada (`monkeypatch`) para que devuelva
    `False`.
  - `predict` tiene que coincidir bit a bit (`np.testing.assert_array_equal`).
  - `in_sample_residuals_` tiene que coincidir con `assert_allclose(rtol=0, atol=1e-10)`.
- **`create_train_X_y`.** Con Ridge, mismos valores que con un árbol
  (`pd.testing.assert_frame_equal`).

Tests existentes que podrían notar los 2e-13 (residuos in-sample o intervalos con
`LinearRegression` o `Ridge` y solo lags):

- `test_fit.py`, `test_predict_interval.py`, `test_predict_interval_conformal.py`,
  `test_predict_bootstrapping.py`, `test_predict_quantiles.py`, `test_predict_dist.py` y
  `test_set_in_sample_residuals.py` de Recursive, Direct y DMV.
- `test_recursive_predict_bootstrapping.py` de Recursive.
- En `model_selection/tests/tests_validation`, `test_backtesting_forecaster_refit.py` y
  `test_backtesting_forecaster_no_refit.py` (los de intervalos).

`test_binning_in_sample_residuals.py` recibe los residuos ya calculados y no cambia. Todos
comparan con tolerancia (`assert_allclose` y `assert_frame_equal`, con `rtol` de 1e-7 o
1e-5). Con el prototipo en el modo D no falla ninguno (sección 17).

## 17. Tests con el prototipo en el modo D

Con `_CONTIGUOUS_X_TRAIN = "D"` por defecto en los cuatro módulos y la lista final, sin
cambiar ningún test (pytest secuencial, sin `-n`, `-m "not slow"`):

| Directorios | Resultado |
|---|---|
| `recursive/tests/tests_forecaster_recursive`, `tests_forecaster_recursive_classifier`, `direct/tests/tests_forecaster_direct`, `tests_forecaster_direct_multivariate`, `model_selection/tests`, `utils/tests` | 3.766 passed, 3 skipped, 2 deselected |

No falla ninguno.
