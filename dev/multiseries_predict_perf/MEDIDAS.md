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
