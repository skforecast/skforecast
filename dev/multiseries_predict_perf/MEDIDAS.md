# Medidas: `ForecasterRecursiveMultiSeries` con pandas 2 y pandas 3

Medidas del 2026-11-17 sobre el commit `cb03fbe96` (paquete idéntico al de `0.28.x` en `c694e1915`). Sirven de línea base para los cambios de [PLAN.md](PLAN.md).

## Método

- Tres entornos: pandas 2.3.3 (p2), pandas 3.1.1 sin `pyarrow` (p3 sin pa, la importación bloqueada) y pandas 3.1.1 con `pyarrow` 26.0.0 (p3 con pa). Python 3.13.18, numpy 2.5.4, scikit-learn 1.9.1, lightgbm 4.7.0.
- Proceso fijado a un procesador lógico rápido con prioridad alta, BLAS con un hilo, rondas alternando entornos y árboles, mínimo de las repeticiones. Dispersión entre rondas: mediana 1,7 %, percentil 90 del 4,3 %.
- Regresión lineal con 24 lags y 3 columnas de exog, salvo donde se indica LightGBM (100 árboles). Las entradas se construyen fuera de la zona medida.
- Backtesting: sin refit, tiempo por fold sin el `fit` inicial.
- Perfil por líneas: tiempo de cada línea de skforecast con las llamadas a pandas que hace. Infla los tiempos alrededor de un 16 %.

## 1. Línea base

Tiempos con 20 y 100 series (1 paso):

| Caso | Series | p2 | p3 sin pa | p3 con pa |
|---|---|---|---|---|
| `predict` sin exog (µs) | 20 | 667 | 768 (+15 %) | 910 (+36 %) |
| | 100 | 802 | 906 (+13 %) | 1100 (+37 %) |
| `predict` exog ancha (µs) | 20 | 1136 | 1247 (+10 %) | 1468 (+29 %) |
| | 100 | 1835 | 1787 (−3 %) | 2049 (+12 %) |
| `predict` exog diccionario (µs) | 20 | 2362 | 2613 (+11 %) | 3018 (+28 %) |
| | 100 | 6880 | 7618 (+11 %) | 8801 (+28 %) |
| `predict` exog formato largo (µs) | 20 | 3532 | 4316 (+22 %) | 5143 (+46 %) |
| | 100 | 11538 | 15146 (+31 %) | 17453 (+51 %) |
| Backtesting sin exog (ms por fold) | 20 | 5,8 | 5,9 (+2 %) | 6,1 (+7 %) |
| | 100 | 24,4 | 24,2 (−1 %) | 24,9 (+2 %) |
| Backtesting exog ancha (ms por fold) | 20 | 8,7 | 9,1 (+4 %) | 9,6 (+11 %) |
| | 100 | 37,7 | 39,3 (+4 %) | 40,4 (+7 %) |
| `fit` exog ancha (ms) | 20 | 12,8 | 13,1 (+2 %) | 13,6 (+6 %) |
| `create_train_X_y` exog ancha (ms) | 20 | 3,7 | 4,1 (+13 %) | 4,4 (+21 %) |
| `create_train_X_y` formato largo (ms) | 20 | 6,0 | 6,8 (+14 %) | 7,8 (+30 %) |

Otros datos:

- Con 1000 series: `predict` sin exog 2386 / 2383 / 2869 µs, exog ancha 9318 / 7640 / 8370, diccionario 63976 / 71416 / 81476, formato largo 110549 / 147112 / 165906; backtesting con exog en diccionario 448 / 465 / 478 ms por fold.
- Con LightGBM y 24 pasos, `predict` con 20 series y exog ancha: 3658 / 3799 / 4011 µs (+4 % y +10 %).
- El formato de `series` (ancho, diccionario o largo) no cambia el coste por fold del backtesting: se convierte a diccionario al principio, y la exog también.
- El encoding (`ordinal`, `onehot`, `None`) no cambia los tiempos de `predict`.
- `fit` pierde entre 0 y 8 % con pandas 3; con formato largo y exog, hasta 13 %.

## 2. Coste fijo y coste por serie en `predict`

Ajuste con 20 y 100 series (µs por llamada):

| Exog | | p2 | p3 sin pa | p3 con pa |
|---|---|---|---|---|
| Sin exog | fijo | 634 | 734 | 862 |
| | por serie | 1,7 | 1,7 | 2,4 |
| Ancha | fijo | 962 | 1112 | 1323 |
| | por serie | 8,7 | 6,8 | 7,3 |
| Diccionario | fijo | 1233 | 1362 | 1572 |
| | por serie | 56 | 63 | 72 |
| Formato largo | fijo | 1531 | 1608 | 2065 |
| | por serie | 100 | 135 | 154 |

- Sin exog o con exog ancha, la pérdida de pandas 3 es fija por llamada: 100 a 150 µs sin `pyarrow` y 230 a 360 µs con él.
- Con exog en diccionario o en formato largo, la pérdida crece con el número de series: con 100 series, entre el 82 y el 98 % es coste por serie.
- Backtesting: cada fold cuesta 1,1 a 1,4 ms fijos más 0,23 ms por serie sin exog y 0,37 ms con exog, casi igual en los tres entornos.

## 3. En qué líneas pierde pandas 3

`predict`, 20 series, exog ancha (µs por llamada en el perfil):

| Línea | p2 | p3 sin pa | p3 con pa |
|---|---|---|---|
| `last_window_to_check[levels]` (`check_predict_input`) | 174 | 175 | 225 |
| `last_window.iloc[-ws:, get_indexer(levels)]` | 179 | 236 | 268 |
| DataFrame de salida | 89 | 100 | 148 |
| `pd.DataFrame(..., columns=levels)` | 51 | 59 | 90 |
| `exog.iloc[:steps, :]` | 30 | 58 | 58 |
| Tres `isna` de exog | 94 | 116 | 125 |
| Cuatro `columns.to_list()` | 4 | 14 | 59 |
| `pd.concat` de la exog repetida por serie | 239 | 184 | 192 |

- Con `pyarrow`, la pérdida (360 µs) se reparte entre las conversiones de nombres de series y columnas.
- Sin `pyarrow` (96 µs), casi todo es `DataFrame.iloc` con un corte, unos 25 a 30 µs más caro en pandas 3. Es la misma causa en `create_train_X_y` (24 → 46 µs por serie) y en el formato largo (30 → 59 µs por serie, la mayor pérdida).
- Con exog en diccionario se suman, por serie: `isna` (14 → 20 a 22 µs), `columns.to_list()` (0,5 → 2,4 y 10 µs) y la comparación de la primera fecha (7 → 10 a 11 µs).

## 4. Dónde se va el tiempo, también en pandas 2

- Fold de backtesting sin refit, 20 series con exog (10,1 ms en el perfil): 3,7 ms (36 %) en recortar y validar el conjunto de entrenamiento de cada serie, que sin refit es siempre el mismo; 0,5 ms en la exog de entrenamiento; 1,4 ms en montar la última ventana; 0,5 ms en cortar la exog de test serie a serie. Sin exog, el recorte del entrenamiento es el 54 % del fold.
- `predict` con exog en diccionario, 20 series (2,8 ms): `reindex_like` por serie 0,62 ms, la plantilla vacía 0,28 ms, `pd.concat` 0,28 ms. El backtesting pasa siempre por aquí.
- `predict` con exog ancha: el `pd.concat` de la misma exog repetida una vez por serie explica casi todo el coste por serie.

## 5. `ForecasterDirectMultiVariate`

Tiene las mismas líneas de nombres, pero pesan poco: `predict` con 20 series cuesta 3212 / 3207 / 3575 µs (0 % y +11 %) y con 100, 12310 / 11761 / 12751 (−4 % y +4 %). Lo domina el `StandardScaler` por serie (90 µs por serie, igual en los tres entornos).

## 6. Prototipo de los once cambios

Diff en [proto_patch/](proto_patch/), aplicado sobre una copia del árbol base. Base → prototipo con 20 series:

| Caso | p2 | p3 sin pa | p3 con pa |
|---|---|---|---|
| `predict` sin exog (µs) | 661 → 264 | 768 → 306 | 939 → 385 |
| `predict` exog ancha (µs) | 1153 → 409 | 1256 → 470 | 1486 → 589 |
| `predict` exog diccionario, un objeto por serie (µs) | 2393 → 1108 | 2616 → 1221 | 3090 → 1360 |
| `predict` exog diccionario, objeto compartido (µs) | 2381 → 642 | 2642 → 646 | 3048 → 782 |
| `predict` exog formato largo (µs) | 3611 → 2007 | 4408 → 2269 | 5267 → 2976 |
| `predict` ventana guardada, exog ancha (µs) | 1453 → 605 | 1525 → 654 | 1820 → 792 |
| Backtesting 1 paso sin exog (ms por fold) | 5,8 → 1,9 | 5,9 → 1,9 | 6,2 → 2,1 |
| Backtesting 1 paso exog ancha (ms por fold) | 8,6 → 2,3 | 9,0 → 2,4 | 9,6 → 2,5 |
| Backtesting LightGBM 24 pasos, exog ancha (ms por fold) | 12,9 → 6,9 | 13,4 → 7,0 | 14,4 → 7,5 |

Con 100 series: `predict` exog ancha 1851 → 533 / 1778 → 617 / 2060 → 756 µs; exog en diccionario con un objeto por serie 7013 → 3680 / 7717 → 4109 / 8913 → 4234; backtesting sin exog 24,4 → 6,5 / 24,3 → 6,6 / 24,7 → 6,7 ms por fold; con exog ancha 38,0 → 7,6 / 39,8 → 7,7 / 41,1 → 7,8.

Identidad del prototipo contra la base: 4591 de 4591 casos (p2) y 4510 de 4510 (p3 con y sin `pyarrow`) en la batería de varias series, y 2911 de 2911 en la del bloque anterior, en los tres entornos.

Ganancia de cada cambio por separado: tabla del bloque A de [PLAN.md](PLAN.md).

## 7. Lo que queda con 100 series tras los once cambios

µs por llamada o por fold en el perfil del prototipo:

| Dónde | Línea | p2 | p3 sin pa | p3 con pa | % de lo que queda |
|---|---|---|---|---|---|
| `predict`, exog por serie | `isna` de la exog de cada serie | 1294 | 1873 | 1978 | 24 / 32 / 30 |
| `predict`, exog por serie | comparar la primera fecha de cada exog | 924 | 1226 | 1306 | 17 / 21 / 20 |
| `predict`, exog por serie | `pd.concat` de la exog de todas las series | 820 | 604 | 653 | 15 / 10 / 10 |
| Fold de backtesting | cortar la ventana de cada serie y unirlas en una tabla | 5296 | 5233 | 5346 | 59 / 58 / 56 |

Son los candidatos del bloque B. El benchmark oficial (`benchmarks/benchmarks/bench_forecaster_recursive_multiseries.py`) usa 600 series con exog en diccionario, que es este caso.

## 8. Descartado por medida

- No repetir el `isna` de la exog ancha cuando no se transforma: 2,7 / 3,1 / 2,8 %, por debajo del corte.
- Construir de otra forma la columna `level` de la salida: 3 a 8 µs con 20 series (menos del 1 %); con nombres de series de tipos mezclados el dtype cambia.
- Memorizar la conversión de columnas por objeto: en pandas 3 un corte de filas de un DataFrame crea un índice de columnas nuevo, así que nunca acierta.

## 9. Prototipo con 600 y 1000 series y con exog de varios tipos (2026-11-18)

`@scale` (una ronda) y los casos `exogdt=mixed` (exog con una columna float, una `int` y una `category` de enteros; 20 y 100 series en dos rondas), base → prototipo. µs por llamada en `predict`, ms por fold en backtesting; p2 / p3 sin pa / p3 con pa. Entre paréntesis, lo que tarda pandas 3 sin y con `pyarrow` frente a pandas 2 en el mismo árbol.

| Caso | Base | Prototipo |
|---|---|---|
| `predict` 1000 series, exog ancha | 10230 / 8077 / 8900 (−21 / −13 %) | 1905 / 2064 / 2438 (+8 / +28 %) |
| `predict` 1000 series, exog diccionario, un objeto por serie | 72104 / 77825 / 90041 (+8 / +25 %) | 39330 / 46830 / 61823 (+19 / +57 %) |
| `predict` 600 series, exog diccionario, un objeto por serie | 43366 / 44458 / 50546 (+3 / +17 %) | 22852 / 28143 / 36790 (+23 / +61 %) |
| `predict` 1000 series, formato largo | 118889 / 152489 / 174943 (+28 / +47 %) | 70467 / 79993 / 103624 (+14 / +47 %) |
| `predict` 1000 series, exog ancha mezclada | 15871 / 12414 / 14679 | 2852 / 3071 / 3740 |
| `predict` 1000 series, exog diccionario mezclada | 130769 / 125819 / 143141 | 75957 / 87392 / 98388 (+15 / +30 %) |
| `predict` 1000 series, formato largo mezclado | 180388 / 226894 / 282820 (+26 / +57 %) | 117956 / 140941 / 160106 (+19 / +36 %) |
| Backtesting 600 series, sin exog | 170 / 166 / 173 | 59 / 59 / 63 |
| Backtesting 600 series, exog ancha | 260 / 282 / 308 | 68 / 66 / 70 |
| Backtesting 600 series, exog diccionario | 260 / 287 / 286 | 105 / 115 / 127 |
| Backtesting 600 series, exog diccionario mezclada | 328 / 353 / 372 | 146 / 152 / 161 |

- El prototipo gana a escala en los tres entornos: 70 a 82 % con exog ancha, 27 a 48 % con diccionario o formato largo, 56 a 77 % por fold en backtesting. Con exog mezclada gana algo menos (30 a 43 % en diccionario y largo).
- La exog mezclada cuesta mucho más que la float con muchas series: con 1000 series en diccionario, 72 → 131 ms en la base y 39 → 76 ms en el prototipo (unos 37 µs más por serie). Ese coste no estaba en el perfil (solo con float).
- Con exog en diccionario, el prototipo gana menos en pandas 3 que en pandas 2 también en valor absoluto (1000 series: 32,8 ms en p2, 31,0 sin pa, 28,2 con pa), así que la diferencia relativa crece. Con `pyarrow` encaja con el `columns.tolist()` por serie del cambio 4 (unos 4,5 µs por serie con tres columnas; `columns.equals` contra un índice guardado no es más barato: 0,17 / 0,82 / 4,49 µs frente a 2,49 / 3,40 / 4,92). Sin confirmar con el perfil.
- Identidad del prototipo con la batería de exog (`identity_exog.py`): 424 / 396 / 424 casos, 0 diferencias, ninguna exog del usuario modificada.

Perfil por líneas del prototipo, `predict` con 100 series y exog en diccionario (un objeto por serie), exog mezclada frente a float (µs por llamada de más en el perfil, que infla los tiempos; p2 / p3 sin pa / p3 con pa; `EXOG_DTYPES=mixed bash run_prof.sh ...`):

| Línea | De más con exog mezclada | Tipo de coste |
|---|---|---|
| `isna` de la exog de cada serie (`check_predict_input`) | 2769 / 2331 / 2446 (15 → 43 µs por serie en p2) | por serie |
| `pd.concat` de la exog de todas las series | 840 / 677 / 644 | por serie |
| `categorical_encoder.transform` y la selección y asignación de sus columnas | 1154 / 1124 / 1567 | por llamada (crece con las filas) |
| comparar la primera fecha de cada exog | 161 / 46 / 0 | por serie |

En formato largo se suma `data_sorted.iloc[start:end]` del corte de cada serie (+632 / +824 / +823). Con la exog mezclada, el `isna` por serie es el mayor coste que queda: `DataFrame.isna` trabaja bloque a bloque (float, `int`, `category`) y `to_numpy` de una tabla de varios dtypes da un array `object`.

Dónde pierde pandas 3 con `pyarrow` en ese caso (prototipo, exog mezclada, +2612 µs frente a p2 en el perfil): `exog_to_check.columns.to_list()` de `check_predict_input` (0,7 → 11,2 µs por serie, el 40 %), el `columns.tolist()` que añade el cambio 4 (0,5 → 6,2 µs por serie, el 22 %), el `isna` (+518) y la primera fecha (+270). Sin `pyarrow`, pandas 3 empata con pandas 2 (+26).

## 10. Cambio 4 y B4: variantes (2026-11-18)

Medido en una copia del árbol de la rama (`e31e3e843`) con las variantes conmutables por variable de entorno; p3 con pa / p3 sin pa / p2.

**Coste por serie** (µs, exog de tres columnas mezcladas y 10 filas, mínimo de 7 × 20000):

| Operación | p3 con pa | p3 sin pa | p2 |
|---|---|---|---|
| `reindex_like` de hoy | 29,4 | 30,0 | 35,0 |
| `index.identical` + `columns.identical` (cambio 4) | 8,8 | 9,5 | 10,1 |
| `index.equals` + `columns.tolist() ==` (plan, no idéntico) | 6,2 | 2,1 | 2,9 |
| `columns.to_list()` | 4,6 | 0,9 | 0,2 |
| `list(columns)` (iterar) | 4,3 | 1,2 | 0,7 |
| `columns.equals` contra un índice guardado | 2,9 | 3,4 | 2,5 |

**Identidad de las variantes.** Sonda de 6920 casos (hoy la batería `labels`): nombres de columna que no son texto, en índices de varios tipos, iguales o solo iguales entre `fit` y `predict`, con y sin variables categóricas, `StandardScaler` y `ColumnTransformer`. Diferencias frente a la base (p2 / p3 con pa / p3 sin pa):

| Variante | Diferencias | Qué cambia |
|---|---|---|
| Plan: `index.equals` + `columns.tolist() ==` | 120 / 116 / 64 | `TypeError` en vez de `ValueError` con nombres `pd.NA`; con `numpy.str_` y zonas horarias, otros avisos de scikit-learn y otras predicciones con `ColumnTransformer` |
| `index.equals` + `columns.equals` | 36 / 104 / 60 | lo mismo sin el `TypeError` |
| `equals` y etiquetas de entrenamiento tras el `concat` | 56 / 0 / 16 | con `identical`, pandas conserva las etiquetas del usuario |
| `identical` (cambio 4) | 0 / 0 / 0 | |
| Iterar las columnas en `check_predict_input` (B4) | ninguna nueva | |

Las cuatro baterías de siempre (`ms trims dtypes exog`) dan 0 diferencias con `identical` en los tres entornos.

**Tiempos en situación real** (`bench_ms.py`, 2 rondas, núcleo 12; ganancia frente al código de la rama):

| Caso | `identical` | `identical` + B4 | Plan (no idéntico) |
|---|---|---|---|
| `predict` 20 series, exog en diccionario | 12,8 / 17,1 / 21,5 | 11,0 / 17,7 / 20,5 | 13,2 / 23,6 / 28,2 |
| `predict` 100 series | 18,5 / 22,3 / 26,9 | 21,0 / 23,6 / 26,4 | 22,4 / 34,3 / 39,3 |
| `predict` 100 series, exog mezclada | 13,5 / 18,5 / 26,5 | 19,5 / 18,6 / 23,5 | 19,1 / 24,2 / 28,1 |
| `predict` 1000 series | 26,4 / 31,4 / 32,7 | 29,4 / 31,8 / 30,4 | 30,4 / 41,9 / 44,6 |
| `predict` 1000 series, exog mezclada | 28,5 / 27,8 / 39,1 | 31,0 / 28,2 / 38,8 | 33,0 / 34,2 / 44,7 |
| Backtesting 100 series, exog en diccionario (por fold) | 11,1 / 13,9 / 9,9 | 12,6 / 14,4 / 8,4 | 13,1 / 17,6 / 14,8 |

B4 sobre el cambio 4 gana del 3 al 7 % en pandas 3 con `pyarrow` con 100 o más series, es neutro sin `pyarrow` y pierde hasta un 4 % en pandas 2.

## 11. Cambio 6 y coste de `empty_exog` (2026-11-19)

Prototipo sobre una copia de `f4537f1b5`, medido frente a esa misma copia sin el cambio (`MS_BASE_TREE`), solo con los casos por los que pasa el cambio (`C6` y `C6_SCALE`). Mínimo de 2 rondas (1 con 600 y 1000 series), núcleo 12; ganancia en p2 / p3 sin pa / p3 con pa.

| Caso | 20 series | 100 series | 600 / 1000 series |
|---|---|---|---|
| `predict`, diccionario que repite un objeto | 34,9 / 41,3 / 42,7 | 61,3 / 68,8 / 70,8 | 69,7 / 77,4 / 79,4 (600); 68,2 / 76,4 / 79,0 (1000) |
| Ídem, 24 pasos | 31,7 / 37,9 / 38,8 | 58,2 / 66,3 / 67,2 | |
| Backtesting por fold, exog ancha, 1 paso | 30,3 / 33,5 / 33,6 | 43,2 / 47,5 / 48,9 | 35,3 / 43,4 / 43,1 (600) |
| Ídem, exog mezclada | 33,5 / 33,7 / 34,2 | 51,1 / 53,3 / 54,2 | 46,6 / 50,3 / 51,3 (600) |
| Ídem, 24 pasos | 19,4 / 19,5 / 21,1 | 25,1 / 28,5 / 27,9 | |
| Ídem, LightGBM, 24 pasos | 13,2 / 15,6 / 15,9 | | |

Casos con un objeto por serie (diccionario, formato largo, backtesting con diccionario, adversos `extra`, `early` y `named`, refit): de −3,4 a +2,8 % en p2, de −3,0 a +3,8 % sin pa y de −2,1 a +1,1 % con pa; con 600 y 1000 series, de −2,5 a +5,1 %. La primera ronda salió con la máquina ocupada (de −9,5 a +9,6 %). Repetidos en p2 con el orden de los árboles invertido, de −1,7 a +0,2 %, y el −5,6 % de backtesting con 600 series y exog mezclada, 0,0 %.

Identidad frente a la base (`cb03fbe96`), cinco baterías: 0 diferencias en los tres entornos (ms 4591 / 4510 / 4510 casos, trims 2911, dtypes 312, exog 424 / 396 / 424, labels 7866 / 3110 / 8106).

**Coste de crear `empty_exog`** (cambio 7; µs por llamada, `timeit` aislado, núcleo 12; en situación real, unas 1,6 veces más):

| Exog | p2 | p3 sin pa | p3 con pa |
|---|---|---|---|
| 3 columnas float | 218 a 224 | 231 a 237 | 262 a 265 |
| 3 columnas mezcladas | 315 a 321 | 338 a 340 | 373 a 379 |

Desglose con exog mezclada (p2 / p3 sin pa / p3 con pa): los tres `pd.Series` vacíos, 70 µs; `DataFrame(dict)` sin índice, 191 / 232 / 265; con el índice de las predicciones (hoy), 298 / 326 / 383; solo las columnas, `DataFrame({c: []}).columns`, 46 / 56 / 76. Tras el cambio 6 es en torno al 20 % de un `predict` con exog compartida y 20 series (1,2 a 1,4 ms) y del 7 al 8 % de un fold de backtesting con exog ancha.

## 12. Cambio 7: plantilla vacía de exog solo si hace falta (2026-11-19)

Prototipo sobre una copia de `2525c976d` (el fichero del cambio es idéntico en `d3d6e7ca3`), medido frente a esa misma copia sin el cambio (`MS_BASE_TREE`), solo con los casos por los que pasa el cambio y el adverso nuevo (`C7` y `C7_SCALE`). Núcleo 12; ganancia en p2 / p3 sin pa / p3 con pa.

**Sonda de etiquetas** (`scripts/probe_labels7.py`). Columnas candidatas frente a `empty_exog.columns` de hoy, con 22 listas de nombres (texto, enteros consecutivos o no, con paso, descendentes, uno solo, `numpy` enteros y `numpy.str_`, float, NaN, booleanos, `None`, fechas con y sin zona, tuplas) en índices de columnas de varios tipos (`object`, `category`, `string` con y sin `pyarrow`, `str` con `pyarrow`, `RangeIndex`) y tres índices de predicción; y `X.reindex(index=..., columns=candidata)` frente a `X.reindex_like(empty_exog)` con tablas del usuario de todos esos tipos, con el índice de las predicciones, con nombre, sin `freq` y con una fila más (unos 96 000 casos por candidata entre los tres entornos, errores incluidos).

| Candidata | Diferencias (p2 / p3 sin pa / p3 con pa) | µs, 3 columnas float (p2 / p3 sin pa / p3 con pa) |
|---|---|---|
| Hoy: `DataFrame` de `Series` vacías con el índice de las predicciones | | 236 / 245 / 249 (mezcladas: 326 / 338 / 362) |
| `pd.Index(list(claves))` | 0 / 126 / 126 (`Index[int64]` en vez de `RangeIndex` con enteros consecutivos) | 14 / 21 / 28 |
| `pd.DataFrame({col: [] ...}, copy=False).columns` (implementada) | 0 / 0 / 0 | 41 / 53 / 72 |

`empty_exog.index is prediction_index` en todos los casos. Control: con `pd.Index(list(claves))`, la batería `labels` da 0 diferencias en los tres entornos; la diferencia no llega a las salidas (solo cambia el camino: unas columnas `RangeIndex` del usuario dejarían de tomar el atajo del 4).

**Identidad** frente a la base (`cb03fbe96`), cinco baterías: 0 diferencias en los tres entornos (ms 4591 / 4510 / 4510 casos, trims 2911, dtypes 312, exog 424 / 396 / 424, labels 7866 / 3110 / 8106), para la variante perezosa y para la implementada.

**Tiempos de la implementada** (2 rondas; mediana de la dispersión entre rondas, 1,019):

| Caso | 3 series | 20 series | 100 series |
|---|---|---|---|
| `predict`, diccionario que repite un objeto | 22 / 21 / 17 | 20 / 19 / 14 | 11 / 13 / 8 |
| Ídem, 24 pasos | | 21 / 17 / 14 | 10 / 10 / 6 |
| `predict`, un objeto por serie | 21 / 21 / 16 | 12 / 12 / 7 | 4 / 5 / 0 |
| Ídem, exog mezclada | | 10 / 6 / 7 | 3 / 1 / 2 |
| Adversos del 4 (`extra=10`, `early=1`, `named`) | | 7 a 10 / 4 a 9 / 7 a 11 | 1 a 5 / −1 a 1 / 4 a 5 |
| Formato largo | 16 / 16 / 9 | 8 / 9 / 3 | 3 / 5 / 0 |
| Ídem, exog mezclada | | 6 / 4 / 4 | 1 / 3 / 1 |
| Backtesting por fold, exog ancha, 1 paso | | 10 / 10 / 6 | 3 / 4 / 1 |
| Ídem, mezclada | | 8 / 7 / 8 | 5 / 1 / 4 |
| Backtesting, exog en diccionario, 1 paso | | 3 / 7 / 4 | 4 / 2 / 1 |
| Backtesting, 24 pasos (ancha / diccionario) | | 3 / 5 / 5 y 4 / 4 / 3 | 0 / 2 / 0 y 2 / 2 / 1 |
| Backtesting, LightGBM 24 pasos | | 1 / 5 / 3 | |
| Backtesting con refit | | 5 / 3 / 3 | 2 / 3 / 0 |
| Control sin el cambio (`predict` con exog ancha) | | 1 / 2 / −1 | |

Con 600 y 1000 series (3 rondas, una en cada orden de los árboles), de −5 a +4 % sin patrón: el ahorro, unos 0,3 ms, queda por debajo del ruido de llamadas de 7 a 210 ms.

**Adverso `missing=1|lgbm`** (una serie sin exog en el diccionario; 3 rondas con el orden de los árboles invertido):

| Variante | 20 series | 100 series |
|---|---|---|
| Plantilla perezosa dentro del bucle (columnas y plantilla) | −3,7 / −3,7 / −3,6 (6 rondas, de −1,5 a −5,6) | −1,9 / +0,9 / −2,0 |
| Implementada (plantilla como hoy si falta alguna exog) | −1,2 / +0,4 / 0,0 | +0,3 / +0,4 / +3,3 |

La primera tanda, frente a `d3d6e7ca3`, se descartó: la máquina estaba ocupada por otros tests (el control se movía de −13 a +20 % y la dispersión de la base llegaba a 1,35).

## 13. Cambio 4b: exog con las fechas de las predicciones y el índice con nombre o sin `freq` (2026-11-19)

Prototipo sobre una copia de `1587f1069` (el paquete de `ab62227dc` es idéntico), medido frente a esa misma copia sin el cambio (`MS_BASE_TREE`), solo con los casos por los que pasa el cambio, los adversos y un control (`C4B`, `C4B_ADV` y `C4B_AFF` en `$MS_WORK/cambio4b_regex.sh`). Núcleo 12, 2 rondas; ganancia en p3 con pa / p3 sin pa / p2 (pandas 3 primero, criterio del 2026-11-19).

**Qué hace `reindex`** (pandas 2.3.3 y 3.1.1): si los dos ejes no son `identical`, `_reindex_multi` o `_reindex_axes` llaman a `Index.reindex(destino)` en cada eje, que con `self.equals(destino)` (receptor: el índice del usuario) da `indexer = None`; `reindex_indexer` copia entonces los bloques del usuario (copia profunda en p2, superficial en p3) con el objeto de destino como eje. El camino nuevo hace lo mismo con una copia superficial.

**Sonda de etiquetas** (`scripts/probe_labels4b.py`): 0 diferencias frente al código del cambio 4 y frente a `reindex` (valores, dtypes, índice y columnas `identical` y del mismo tipo, `freq`, `attrs`, `flags`, errores; el objeto del usuario no cambia) en 210 880 / 218 008 / 132 600 casos (p2 / p3 con pa / p3 sin pa), de los que 7348 / 16 100 / 9460 van por el camino nuevo. La primera versión del prototipo dejaba las columnas del usuario y dio 176 diferencias en p2: con nombres `['a', None, 'c']`, las columnas del usuario son `identical` a las de entrenamiento `['a', nan, 'c']` (pandas 2 trata `None` y NaN como iguales y las dos son `object`), y `reindex` pone las de destino. Poner también las columnas del entrenamiento cuesta 1 µs (11,0 frente a 9,7 µs en p3; `reindex`, 30 µs).

**Identidad** frente a la base (`cb03fbe96`), cinco baterías: 0 diferencias en los tres entornos (ms 4591 / 4510 / 4510 casos, trims 2911, dtypes 312, exog 424 / 424 / 396, labels 7866 / 8106 / 3110). Casos límite por la API pública (`scripts/edges4b.py`, 7472 por entorno: Series, columnas que faltan, sobran o en otro orden, categóricas, `transformer_exog`, niveles sin exog, un nivel, objeto compartido, índice con zona, en `us` y `RangeIndex`, `predict_interval`, backtesting): 0 diferencias en los tres entornos, 460 casos sin error por el camino nuevo.

**Tiempos** (versión implementada, orden de los árboles invertido; mediana de la dispersión entre rondas, 1,011):

| Caso | 20 series | 100 series |
|---|---|---|
| `predict`, exog en diccionario con el índice con nombre | 11,0 / 15,1 / 18,1 | 15,6 / 19,4 / 26,7 |
| Ídem, sin `freq` | 9,1 / 11,7 / 14,2 | 13,1 / 15,6 / 18,6 |
| Backtesting por fold, exog en diccionario con nombre | 5,9 / 7,6 / 9,3 | 13,7 / 13,5 / 10,8 |
| Ídem, sin `freq` | 5,9 / 6,7 / 7,9 | 9,1 / 9,7 / 10,1 |
| Control (`predict` con exog ancha) | −0,1 / −1,3 / −1,0 | |

Primera tanda, con la primera versión (sin poner las columnas): los mismos casos de 7 a 21 % en pandas 3; backtesting con exog ancha, con nombre o sin `freq`, de 0 a 4,5 % (el objeto compartido se alinea una vez por fold); adversos con 20 series de −0,7 a +0,3 % en pandas 3.

**Adversos con 100 series** (`extra=10`, `early=1`, `early=1|named`: otras fechas, ahora con un `equals` antes del `reindex`, de 1,0 a 1,4 µs por serie en `timeit`): la primera tanda dio de −2,2 a +1,0 % en p3 con pa, de −3,0 a +2,1 % sin pa y de −4,4 a −1,3 % en p2; repetidos con el orden de los árboles invertido, de −1,1 a −0,7 / −0,7 a −0,4 / −1,6 a −1,1. Una guarda `len(exog_values) == len(prediction_index)` antes del `equals` (también idéntica en la sonda) daba de −1,4 a −1,3 / −0,8 a +0,5 / −0,5 a +0,2: no mejora en pandas 3 con pa y el resto queda dentro del ruido, no se implementa.

**Revisión final** (`timeit` del bloque de alineación de un nivel, antes y después, núcleo 12, mínimo de 9 tandas alternas; exog de 3 filas y 3 columnas): con nombre, de 30,6 a 17,8 µs en pandas 3 y de 32,2 a 14,6 en pandas 2; sin `freq`, de 36,8 a 21,6 y de 38,9 a 19,3; el atajo del cambio 4 no cambia (6,9 y 10,2 µs). Un adverso que no estaba en las tandas, las mismas fechas con otras columnas (falta una; va a `reindex` después del `equals` y de comparar las columnas): de +0,6 a +2,4 µs sobre 104 a 112 µs en pandas 3 y de +3,6 a +4,1 sobre 79 a 94 en pandas 2, del orden del de otras fechas (de +1,4 a +2,2 sobre 74, y de +2,8 a +3,6 sobre 75). En los dos caminos los ejes del resultado son los objetos de destino (`index is prediction_index`, `columns is exog_columns`); la única diferencia con `reindex` es que en pandas 2 el resultado comparte los datos con la exog del usuario en vez de copiarlos (en pandas 3 ya los compartía), como el atajo del cambio 4, y después solo se lee.

## 14. Cambio 8: cortar cada serie de un formato largo sin el MultiIndex (2026-11-19)

Prototipo sobre una copia de `1be8a4d90`, medido frente a esa misma copia sin el cambio (`MS_BASE_TREE`). Protocolo ligero: solo pandas 3.1.1 con `pyarrow` (numpy 2.5.4), los casos afectados (exog en formato largo con 20 y 100 series) y dos controles (`predict` con exog ancha y `create_train_X_y` con series largas sin exog, que van por la rama Series y no cambian); expresión regular `C8` en `$MS_WORK/cambio8_regex.sh` (fuera del repositorio). Núcleo 12, máquina libre (carga del 6 %), 2 rondas en cada orden de los árboles.

**Orden de los árboles.** Con la base primero, el control con exog ancha ganaba un 2,9 % en las dos rondas (la base, de 3 a 5 % más lenta); con el orden invertido, −0,3 %. Se da la media de los dos órdenes.

| Caso (p3 con pa) | Base primero | Prototipo primero | Media |
|---|---|---|---|
| `predict`, exog larga, 20 series (4350 → 3441 µs) | 20,9 | 18,5 | 19,7 |
| Ídem, 100 series (15 642 → 10 827 µs) | 30,8 | 24,9 | 27,9 |
| `predict`, exog larga mezclada, 20 series (6510 → 5699 µs) | 12,4 | 12,0 | 12,2 |
| Ídem, 100 series (21 986 → 17 820 µs) | 18,9 | 19,4 | 19,2 |
| `create_train_X_y`, exog larga, 20 series (7,8 → 6,9 ms) | 12,2 | 8,4 | 10,3 |
| Ídem, 100 series (31,3 → 26,5 ms) | 15,5 | 13,9 | 14,7 |
| `fit`, exog larga, 20 series | 3,5 | 0,2 | 1,9 |
| Ídem, 100 series | 5,5 | 5,0 | 5,3 |
| Backtesting por fold, exog larga, 20 series | 1,3 | −2,7 | −0,7 |
| Ídem, 100 series | 1,2 | −1,6 | −0,2 |
| Control: `predict` con exog ancha, 20 series | 2,9 | −0,3 | 1,3 |
| Control: `create_train_X_y`, series largas sin exog, 20 series | 1,4 | −3,2 | −0,9 |

En backtesting la exog larga se corta una vez, al convertirla en diccionario, no en cada fold: su fila es ruido. En `fit` pesa más el modelo. El prototipo antiguo (§6) daba 8 / 19 / 16 % en `predict` con exog larga y de 3 a 15 % en `create_train_X_y`; sobre el árbol actual la ganancia en pandas 3 es del mismo orden o algo mayor.

**Identidad** frente a la base (`cb03fbe96`), cinco baterías: 0 diferencias en los tres entornos (ms 4591 / 4510 / 4510 casos, trims 2911, dtypes 312, exog 424 / 424 / 396, labels 7866 / 8106 / 3110; p2 / p3 con pa / p3 sin pa). Casos límite del formato largo (57 variantes, 483 llamadas por entorno: corte directo, también con `series_ids`; `create_train_X_y`, `fit`, `predict`, `predict_interval` y backtesting), frente a `1be8a4d90`: 0 diferencias en los tres entornos, y los datos del usuario no cambian.

Pendiente de la verificación final de la rama: los tiempos en pandas 2 y en pandas 3 sin `pyarrow`.

## 15. Cambio 5: repetir la exog ancha con numpy (2026-11-20)

Prototipo sobre una copia de `aae5b99be`, medido frente a esa misma copia sin el cambio (`MS_BASE_TREE`). Protocolo ligero: solo pandas 3.1.1 con `pyarrow` (numpy 2.5.4, Python 3.13.18), los casos afectados (`predict` con exog ancha, 20 y 100 series, todas sus variantes) y dos controles (`predict` sin exog y backtesting con exog ancha, que va por la rama de diccionario y no cambia); expresión regular `C5` en `$MS_WORK/cambio5_regex.sh` (fuera del repositorio). Núcleo 12, 2 rondas en cada orden de los árboles.

**Tanda descartada.** La primera se lanzó durante una llamada de Zoom (carga del 36 al 63 %): los controles se movían de −6,8 a +8,3 %. Repetida con la carga entre el 4 y el 12 %; los dos órdenes coinciden.

**Coste aislado** (`timeit`, mínimo de 7 tandas, sin fijar el proceso; solo para dimensionar): `pd.concat` de la misma tabla una vez por nivel más `to_numpy`, 151 µs con 20 series y exog float, 414 µs con exog mezclada (float, `int`, `category`) y 560 / 1280 µs con 100 series; `np.tile`, de 4 a 25 µs.

| Caso (p3 con pa) | Base primero | Prototipo primero | Media |
|---|---|---|---|
| `predict`, exog ancha, 20 series (1248 → 1038 µs) | 17,2 | 16,5 | 16,9 |
| Ídem, 100 series (1825 → 1206 µs) | 34,2 | 33,7 | 34,0 |
| Sin codificación (`encoding=None`), 20 / 100 series | 16,4 / 36,8 | 15,1 / 34,1 | 15,8 / 35,5 |
| `onehot`, 20 / 100 series | 18,6 / 34,2 | 16,4 / 34,9 | 17,5 / 34,6 |
| Exog mezclada, 20 / 100 series | 12,5 / 30,0 | 12,5 / 29,8 | 12,5 / 29,9 |
| Más filas que `steps` (`extra=10`), 20 / 100 series | 17,9 / 33,9 | 16,9 / 34,9 | 17,4 / 34,4 |
| LightGBM, 20 / 100 series | 12,7 / 30,2 | 15,5 / 29,8 | 14,1 / 30,0 |
| Ventana guardada, 20 / 100 series | 14,5 / 22,6 | 11,1 / 19,9 | 12,8 / 21,3 |
| 24 pasos, 20 / 100 series | 14,3 / 28,6 | 12,8 / 29,0 | 13,6 / 28,8 |
| 24 pasos con LightGBM, 20 / 100 series | 5,5 / 8,0 | 7,8 / 7,8 | 6,7 / 7,9 |
| Control: `predict` sin exog, 20 series | −0,6 | −0,7 | −0,7 |
| Control: backtesting por fold, exog ancha, 20 series | 0,0 | −0,9 | −0,5 |

El prototipo antiguo (§6) daba 18 / 13 / 11 % en `predict` con exog ancha y 20 series; sobre el árbol actual, con el coste fijo ya recortado por los cambios anteriores, el `concat` pesa más y la ganancia en pandas 3 con `pyarrow` es mayor.

**Identidad** frente a la base (`cb03fbe96`), cinco baterías: 0 diferencias en los tres entornos (ms 4591 / 4510 / 4510 casos, trims 2911, dtypes 312, exog 424 / 424 / 396, labels 7866 / 8106 / 3110; p2 / p3 con pa / p3 sin pa). Casos límite de la exog ancha (frente a `aae5b99be`, 16 030 / 16 030 / 15 038 llamadas): 0 diferencias, también en el diccionario de arrays de `_create_predict_inputs`; la exog del usuario no cambia.

Pendiente de la verificación final de la rama: los tiempos en pandas 2 y en pandas 3 sin `pyarrow`.
