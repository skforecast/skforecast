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
