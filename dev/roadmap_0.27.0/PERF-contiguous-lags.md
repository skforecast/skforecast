# PERF-contiguous-lags: matriz de entrenamiento contigua para el estimador

Rama `perf/contiguous-lag-matrix` desde `origin/0.27.x` (17219ad65), solo en local.
Medición con recomendación: el código del paquete no cambia. El prototipo está en
`repro/PERF-contiguous-lags_prototype.diff` y no está aplicado en la rama.

**Estado (2026-11-08):** medido y pendiente de decisión. Recomendación (sección 10):
copiar la matriz a orden C solo cuando el estimador (o el último paso del `Pipeline`) es
`LinearRegression`, `Ridge` o `RidgeCV`. Tamaño S. Fila propuesta en la sección 11; no se
ha añadido a `roadmap.md`.

Origen: QW-13, sección 8.6. Con solo lags y sin NaN, `ForecasterRecursive` pasa al
estimador una vista invertida de `sliding_window_view` (strides `(8, -8)`, ni C ni F
contigua). Con 20.000 x 24, el `fit` de Ridge tarda 1,35 ms con la vista y 0,82 ms con
una copia contigua.

## 0. Resumen

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
| D. Copiar solo cuando el estimador gana | B, pero solo si el estimador (o el último paso del `Pipeline`) es `LinearRegression`, `Ridge` o `RidgeCV` | Recursive: Ridge de un 3 a un 29 % menos, LinearRegression de un 3 a un 14 %. Direct: de un 7 a un 42 % y de un 3 a un 17 %. Pipelines: de un 4 a un 17 %. Backtesting y búsqueda con Ridge y muchos `fit` (417 folds, Direct, `refit=True`, `OneStepAheadFold`): de un 15 a un 34 % menos. | Memoria: entre 0 y +32 MB con una matriz de 256 MB. Residuos in-sample de esos estimadores con cambios de hasta 2e-13; las predicciones no cambian. Con el resto de estimadores, ningún cambio. Una función en `utils` y una llamada en 4 forecasters. |

**Recomendación: D.** Tiene sentido:

- La ganancia es real y estable en todos los tamaños con los modelos lineales más usados.
  Son también los que más se usan en búsquedas y backtesting con muchos `fit`, que es donde
  más se nota.
- Las predicciones no cambian y no falla ningún test.
- Es barato de implementar.

En valor absoluto es poco por `fit` (de 0,1 a 1 ms con tamaños habituales). Por eso es una
mejora de prioridad baja, no un arreglo: va con los PERF de la release, sin bloquear nada.

Detalles para la implementación:

- **Criterio.** Lista cerrada de lo medido (`LinearRegression`, `Ridge`, `RidgeCV`, que
  usa el mismo cálculo). Hay que mirar el último paso del `Pipeline`, como
  `select_n_jobs_fit_forecaster`.
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
| Rendimiento con modelos lineales | PERF-contiguous-lags | S. Nuevo (2026-11-08). Con solo lags, Recursive, Classifier, Direct y DMV pasan al estimador una vista no contigua de `sliding_window_view`. Copiarla a orden C solo cuando el estimador (o el último paso del `Pipeline`) es LinearRegression, Ridge o RidgeCV reduce el `fit` completo con Ridge de un 19 a un 29 % (Recursive, 20.000 filas o más) y hasta un 42 % (Direct), y el backtesting y las búsquedas con Ridge de un 15 a un 34 %, sin cambiar las predicciones (residuos in-sample: menos de 2e-13). Con los demás estimadores no se copia: no ganan, y la copia suma una matriz de memoria y hace un 4 % más lento HistGradientBoosting con 200.000 x 168. Medición y opciones en [`PERF-contiguous-lags.md`](PERF-contiguous-lags.md) |

## 12. Prototipo

`repro/PERF-contiguous-lags_prototype.diff` (+58 líneas en los 4 forecasters), no
aplicado en la rama. Añade el indicador de módulo `_CONTIGUOUS_X_TRAIN`:

- `True` (por defecto en el parche): copia C.
- `"F"`: copia F.
- `False`: camino de `0.27.x`.

Es para medir, no para fusionar: la opción D no usaría un indicador de módulo, sino el
criterio del estimador. No se ha creado ningún worktree temporal para comparar: las dos
versiones se alternan en el mismo proceso.
