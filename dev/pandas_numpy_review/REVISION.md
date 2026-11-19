# Revisión independiente del estudio "pandas frente a numpy"

Fecha: 2026-11-19. Revisa [INFORME.md](INFORME.md) (`fddf0de`). Árbol medido: `perf/multiseries-predict-backtesting` en `ab62227`, el mismo del informe. No se ha modificado nada de `skforecast/`. Scripts en [review_scripts/](review_scripts/) (ver su [README](review_scripts/README.md)).

**Entornos.** Python 3.12.3, numpy 2.5.5, scikit-learn 1.9.1, numba 0.68.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, 4 núcleos. **p2** = pandas 2.3.3 (con `pyarrow` 26.0.1 instalado); **p3** = pandas 3.1.1 con la importación de `pyarrow` bloqueada; **p3 pa** = pandas 3.1.1 con `pyarrow` 26.0.1. Las cifras van siempre en el orden p2 / p3 / p3 pa. Los tiempos se midieron uno a uno, sin nada más en marcha y con `OMP_NUM_THREADS=1` (salvo donde se dice); las comprobaciones de corrección las hicieron tres subagentes en paralelo y sin medir tiempos. Lo que afirmo de sus informes lo he vuelto a ejecutar yo cuando contradice al informe (se indica); lo demás se marca como "subagente".

## Resumen

1. **El techo de §5.1 es correcto como techo del bucle, pero el titular está sobrevendido.** Se reproduce (Ridge, 600 series, exog mixta: 179 / 200 / 225 ms por fold hoy, 0,57 / 0,57 / 0,64 en el techo). Lo que no dice el informe es cuánto pesa lo que el techo deja fuera. He escrito un prototipo que valida y prepara una vez y devuelve **las mismas métricas, las mismas predicciones y los mismos avisos en el mismo orden** que el código actual (25 casos, tres entornos). Con él, el mismo backtesting pasa de 10,1 / 11,1 / 12,0 s a 1,9 / 1,9 / 2,2 s: **5 a 6 veces la llamada completa (7 a 8 sin el fit), no 40 a 400**. Con LightGBM, 2,8 a 3,3 veces. Con 5 folds en vez de 50, 1,3 a 1,5 veces con Ridge y 1,0 con LightGBM. El bucle de folds queda en 1,2 a 1,6 ms por fold; lo que domina después son costes fijos que el informe despacha en una frase: las métricas (0,56 a 0,60 s) y la preparación (0,47 a 0,62 s).
2. **La buena noticia que el informe no da:** los avisos por fold y su orden, que el informe cuenta como obstáculo real (§4.3), se reproducen exactamente desde máscaras calculadas una vez. La validación no impide el camino rápido.
3. **P1 es un bug real y más amplio de lo descrito, y la decisión D-NP2 recomendada es la equivocada.** También afecta a XGBoost y a categorías `0..n-1` declaradas en otro orden; HistGradientBoosting **no** está afectado (el informe dice que sí); la validación one-step-ahead lo oculta (MAE 3,9 frente a 31,1 en backtesting). El arreglo "numpy en `fit`" no es "mismo modelo": con `categorical_features=None` cambia el modelo de quien hoy acierta, y la opción (a) convierte en silencio un modelo categórico en uno numérico. La opción (c) conserva el modelo de hoy y arregla `predict`.
4. **P2 es correcta en su objetivo, con riesgo mayor que "bajo".** La condición tal como está escrita acepta `complex128`, redondea enteros grandes que hoy salen exactos de `create_train_X_y`, añade copias donde hoy hay vistas y olvida el `predict` de multiserie. Además arregla un bug que el informe no vio (HistGradientBoosting con categórica y `bool` falla en `predict`).
5. **De las propuestas pequeñas, seis necesitan cambios para ser idénticas:** P4 (NaN visto en `fit`), P6 (una caché cambia resultados), P11 (no es bit a bit en pandas 2.3), P14 (solo hasta 7 offsets), P16 (vista de solo lectura en pandas 3) y P12 (caché sin punto de invalidación). I5 no se reproduce como se describe y la parte de multiserie de P17 es falsa.
6. **Orden:** de acuerdo con multiserie antes que univariantes (D-NP1), pero las métricas en numpy (BTPLAN M1 y PERF-03) tienen que ir **antes o con** P3, no después: tras P3 son el 40 % de lo que queda. Y el contenedor debe vivir por encima del backtesting (lo usan las búsquedas, que hoy repiten todo por candidato) y apoyarse en un único núcleo numpy compartido con `predict`.

## 1. Nota de la fase 1 (a ciegas)

Escrita y guardada en el commit `30cb520` antes de abrir `INFORME.md`. Se reproduce tal cual.


Código leído en `perf/multiseries-predict-backtesting` (`ab62227`):
`_validation.py:1263-2072`, `_utils.py:1559-1749`,
`_forecaster_recursive_multiseries.py:2634-3235` y `3634-3730`.

### Dónde se crean hoy objetos de pandas por fold y por serie (sin refit)

1. `_extract_data_folds_multiseries` (`_utils.py:1702-1711`): con `series` dict, un
   `v.loc[...]` por serie y fold, y después `pd.DataFrame(series_last_window)`, que alinea
   N Series. Con exog dict, un `v.loc[...]` por objeto distinto y fold (`_utils.py:1733-1744`).
2. `_backtesting_forecaster_multiseries` (`_validation.py:1842-1846`): `series[level].loc[...]`
   por fold y nivel dentro de un `any` (corta pronto en el caso normal, es N x F en el peor).
3. `predict` -> `_create_predict_inputs`: `check_predict_input` (comprobaciones por nivel),
   un `reindex` por objeto exog (`:2997`), `pd.concat` de N DataFrames (`:3014`),
   `transform_dataframe` y `categorical_encoder.transform` sobre el concatenado,
   `dtypes.to_dict()`, `pd.DataFrame(last_window_matrix)` (`:3005`) y el DataFrame largo
   de salida con `np.tile(levels, n_steps)` (array de objetos) y `np.repeat(prediction_index)`
   (`:3725`).
4. Al final, un `pd.concat` de F DataFrames, `get_indexer`, un MultiIndex (`set_index('level',
   append=True)`) y `_get_y_true_multiseries`.

### Cómo lo haría

Idea: "planificar una vez, ejecutar sobre arrays". El bucle de folds solo hace cortes de
numpy y llama a `_recursive_predict`; pandas aparece una vez antes y una vez después.

- **Antes del bucle** (una vez por modelo ajustado, es decir, una vez si no hay refit):
  - Matriz `Y` (T_span x N, float64), con NaN fuera del rango de cada serie; para un dict,
    cada serie se coloca con `span_index.get_indexer`. De ella salen, vectorizados, primer
    y último valor válido por serie y por tanto qué niveles entran en cada fold
    (reglas de `_utils.py:1689-1711`: en train con más de `window_size` valores, ventana
    completa, sin NaN si `dropna_last_window`).
  - `Y` transformada con el `transformer_series_` de cada nivel **una vez** (el transformador
    no cambia entre folds sin refit). La diferenciación se hace por ventana en numpy (o una vez
    sobre la serie entera: los valores coinciden, y la inversa solo necesita los últimos
    `order` valores de la ventana, que están en la matriz).
  - Exog: un array `E` (n_objetos_distintos x T x n_exog_out) ya pasado por `transformer_exog`
    y por el codificador de categóricas una sola vez sobre el exog completo, deduplicado por
    `id()` como ya hace `_extract_data_folds_multiseries`, más un mapa nivel -> objeto.
  - Validación (`check_predict_input`) una sola vez sobre el conjunto, registrando qué folds
    tendrían aviso (NaN de exog en el tramo de test, niveles desconocidos).
- **En cada fold**: `Y_t[lw_start:lw_end, cols]`, `E[obj, test_start:test_end]`, llamada a un
  núcleo numpy (`_recursive_predict` ya lo es salvo el `last_window.to_numpy()` de `:3170`),
  inversa de diferenciación y de transformación, y escritura en arrays preasignados
  (`pred`, `fold`, `step`, código de nivel, posición en `span_index`).
- **Después**: un único DataFrame (`span_index.take(pos)`, niveles con `take` sobre un array
  de objetos) y las métricas agrupando por código de nivel, sin MultiIndex.

Para no mantener dos implementaciones, el núcleo numpy debería ser el mismo que usa
`predict()`: `predict` = validar + convertir + núcleo + envolver; backtesting = validar una
vez + núcleo en bucle + envolver una vez.

### Condiciones que exigiría (si no se cumplen, camino actual)

- `type(forecaster) is ForecasterRecursiveMultiSeries` (ni subclases que redefinan `predict`,
  ni Direct ni Rnn).
- Sin refit para cachear las transformaciones; con refit entero el cacheo vale entre refits
  (se recalcula en cada fit, coste despreciable frente al fit).
- `interval is None` (o conforme, que es predicción puntual más residuos); bootstrapping aparte.
- `transformer_series` y `transformer_exog` **fila a fila** (el resultado de transformar una
  fila no depende de las demás). No se puede comprobar en general: un `FunctionTransformer`
  con una media móvil lo incumple. Lista blanca de sklearn o `None`; lo demás, camino actual.
- Exog con los mismos dtypes en todos los niveles y numérica tras codificar; índices de series
  y exog sobre la misma rejilla que `span_index`.
- `return_predictors=False` (o construir X una vez al final).

### Lo difícil de conservar

- **Avisos**: hoy se emiten por fold, dentro de `predict` (`MissingValuesWarning` de exog,
  niveles desconocidos, exog tras transformar `:2890`). Número, orden y texto cambian si se
  valida una vez. Con el filtro por defecto el usuario ve uno por sitio de llamada, pero los
  tests con `pytest.warns`/`recwarn` cuentan.
- **Errores**: qué fold falla primero y el mensaje con paso y fecha (`:3217-3225`); errores
  de `check_predict_input` (exog que no cubre el test del último fold, columnas que faltan).
- **Semántica de `reindex`**: fechas o columnas que faltan en la exog de un nivel se rellenan
  con NaN, y los enteros pasan a float; la exog `None` de un nivel usa `empty_exog` con los
  dtypes de entrenamiento. El dtype del `to_numpy()` final (`:3037`) depende de la mezcla de
  dtypes del concatenado: si alguna columna queda como `category` u `object`, el array no es
  float y eso llega al estimador.
- **Orden**: niveles en el orden de `levels` (`_validation.py:1386`), filas por paso y dentro
  por nivel; folds solapados conservan duplicados.
- **Niveles desconocidos** en un fold (transformador `_unknown_level`, código NaN) y
  forecaster ya ajustado (`initial_train_size=None`).
- **Igualdad bit a bit**: transformar la serie entera frente a la ventana es elemento a
  elemento en los escaladores de sklearn, así que debería ser idéntico; hay que probarlo, no
  suponerlo (PowerTransformer, QuantileTransformer, y cualquier BLAS de por medio).
- `n_jobs > 1` con entero explícito: hoy se respeta aun sin refit; habría que trocear folds.

### Contraste con el informe, una vez leído

- Coincidimos en el diseño (matriz de series, exog codificada una vez, cortes por posición, `_recursive_predict` directo) y en las condiciones (`refit=False`, transformadores fila a fila).
- La nota anticipaba tres cosas que el informe no trata: que el array final de la exog puede no ser float (`:3037`, lo confirma el subagente de P2: es `object` con `bool` o nullable), que el núcleo debería ser único para `predict` y backtesting, y que `n_jobs` entero explícito hay que respetarlo.
- La nota se equivocaba en una: daba los avisos por fold como lo más difícil de conservar. El prototipo los reproduce (sección 2.2).
- Ni la nota ni el informe vieron lo que resultó decisivo al medir: los costes fijos por backtesting (sección 2.3).

## 2. El techo de §5.1

### 2.1 Reproducción

`scripts/ceiling_bt_ms.py`, 600 series, exog mixta, 50 folds de 24 pasos ([results/ceiling_repro_600_mixed.txt](review_scripts/results/ceiling_repro_600_mixed.txt)):

| Estimador | Hoy, ms por fold (p2 / p3 / p3 pa) | Informe | Techo, ms por fold | Informe | Solo el modelo (p3 pa) |
|---|---|---|---|---|---|
| Ridge | 179 / 200 / 225 | 193 / 211 / 248 | 0,57 / 0,57 / 0,64 | 0,56 / 0,53 / 0,61 | 0,12 |
| LightGBM (50 árboles) | 195 / 213 / 247 | 218 / 236 / 256 | 16,1 / 15,7 / 17,7 | 18,4 / 17,7 / 17,9 | 17,2 |

**De acuerdo: se reproduce**, con predicciones idénticas. Mis cifras de "hoy" son un 5 a 10 % menores (máquina sin carga). La pérdida de p3 pa frente a p2 (25 %) también se reproduce.

Tres observaciones sobre cómo está construida la cifra:

1. "Hoy, ms por fold" es `(backtesting - fit) / n_folds`, así que reparte entre los folds todo lo que ocurre una sola vez: el preprocesado de la función pública (`_validation.py:2307-2317`), la extracción del primer fold, `pd.concat`, `y_true`, las métricas. El techo no incluye nada de eso. Con el desglose de la sección 2.3, lo que es de verdad por fold son 202 / 203 / 228 ms; el resto (1,0 a 1,2 s por backtesting, un 9 %) es fijo y **sigue ahí después de P3**.
2. El techo usa una matriz densa `values` que ya existe en el script, series de la misma longitud, sin transformador, sin diferenciación, sin window features, `encoding='ordinal'` y todos los niveles en todos los folds. No demuestra que esos casos se puedan cubrir (lo demuestra la sección 2.2).
3. Todo se midió con `OMP_NUM_THREADS=1`. Con los 4 hilos de LightGBM el modelo baja de 15,6 a 7,5 ms por fold ([results/proto_bt_batched.txt](review_scripts/results/proto_bt_batched.txt)), de modo que en uso real la parte que no es el modelo pesa más de lo que dice la tabla del informe. Esto favorece a P3, no la perjudica.

### 2.2 Qué deja fuera el techo, caso por caso

[review_scripts/proto_bt_once.py](review_scripts/proto_bt_once.py) es un backtesting sin refit que valida y prepara una vez, recorre los folds con numpy y construye la salida una vez. Llama a `_recursive_predict` tal cual (con un objeto mínimo en lugar del DataFrame de la ventana, por el `to_numpy()` de `:3170`) y a `_calculate_metrics_backtesting_multiseries` tal cual. [review_scripts/proto_bt_once_cases.py](review_scripts/proto_bt_once_cases.py) lo compara con `backtesting_forecaster_multiseries` en 25 casos: `assert_frame_equal(check_exact=True)` de métricas y predicciones, y la lista de avisos (categoría y texto) en el mismo orden.

**Resultado: 24 de 25 casos idénticos en métricas, predicciones y avisos, en los tres entornos; el caso 25 lanza el mismo error en los dos** (Ridge con NaN en la exog de entrenamiento).

| Punto pedido | ¿En el techo del informe? | En el prototipo | Veredicto |
|---|---|---|---|
| Series de distinta longitud (empiezan tarde, acaban antes) | No | Idéntico. Matriz con NaN más una máscara de "fecha presente en el índice de la serie" (`len(v_last_window) >= window_size` de `_utils.py:1705` depende del índice, no de los NaN) | Alcanzable |
| NaN intercalados | No | Idéntico con Ridge (el nivel se cae del fold, `_utils.py:1710`) y con LightGBM (se conserva y se avisa) | Alcanzable |
| Niveles que no están en todos los folds | No | Idéntico, incluido `_warn_levels_not_predicted_multiseries` | Alcanzable |
| `gap` y NaN de la exog en el gap | No | Idéntico con `gap=3`. El aviso de NaN en el gap se emite igual que hoy (hoy se avisa aunque esos pasos se descartan, `_validation.py:1456`) | Alcanzable |
| `skip_folds`, folds solapados, último fold incompleto | No | Idéntico | Alcanzable |
| `refit` entero o `True` | No | **No cubierto.** Las transformaciones y el codificador cambian en cada ajuste: la preparación sería por tramo entre refits y solo de las filas de test de ese tramo | Fuera de la primera versión; el informe lo dice |
| Diferenciación (orden 1 y 2) | No | Idéntico (`np.diff` por eje y la misma suma acumulada de `inverse_transform_next_window`, `_preprocessing.py:363-367`) | Alcanzable. Diferenciación en diccionario (un orden por serie): no cubierto |
| `transformer_series` | No | Idéntico con `StandardScaler` y `MinMaxScaler` transformando la serie entera una vez e invirtiendo una vez por nivel al final | Alcanzable con transformadores elemento a elemento; hace falta lista blanca |
| Window features | No | Idéntico (`RollingFeatures`): ya son numpy dentro de `_recursive_predict` | Alcanzable |
| Calendar features | No | Idéntico calculándolas una vez sobre `span_index` y cortando | Alcanzable |
| `encoding` `onehot`, `ordinal_category`, `None` | No (solo `ordinal`) | Idéntico los tres | Alcanzable |
| Niveles desconocidos | No | No pueden darse con `initial_train_size`: un nivel solo se predice si está en el entrenamiento (`_utils.py:1705`). Solo con forecaster ya ajustado (`initial_train_size=None`), **no cubierto** | Camino actual |
| `dropna_from_series` | No | Idéntico (solo afecta al `fit`) | Alcanzable |
| `transformer_exog` | No | Idéntico con `StandardScaler` aplicado una vez a la tabla | El informe lo excluye (D-NP4); una lista blanca es viable |
| Exog con huecos: clave con `None`, serie sin exog, exog que acaba antes, fechas que faltan | No | Idéntico, avisos incluidos | Alcanzable |
| `levels` como subconjunto y en otro orden | No | Idéntico | Alcanzable |
| `RangeIndex`; índice diario con zona horaria | No | Idéntico | Alcanzable |
| Métricas con `y_train` (MASE, RMSSE) | No (MAE en numpy) | Idéntico (misma función) | |
| `n_jobs > 1` | No | **No cubierto.** Sin refit `'auto'` ya resuelve a 1 (`_utils.py:836`), pero un entero explícito hoy se respeta | Decidir: ignorarlo con aviso o trocear folds |
| `return_predictors` | No | **No cubierto.** `_recursive_predict` no devuelve la matriz de predictores | Camino actual, como dice D-NP4 |
| Intervalos | No | **No cubierto** | Camino actual, como dice D-NP4 |
| Avisos por fold y su orden | No | **Idénticos** (ver abajo) | Alcanzable: en desacuerdo con §4.3 |

**Avisos.** El informe (§4.3) da la validación y los avisos como obstáculo real "que hay que decidir". No hace falta decidir nada: los avisos de `check_predict_input` (`utils.py:1624`, `:1674`, `:1715`, `:1725`, y el de `_check_exog_alignment`) dependen de tres máscaras por objeto de exog y por fecha (hay NaN, la fecha existe en su índice) y de la máscara de NaN de la ventana. Calculadas una vez, se emiten por fold con el mismo texto y orden; el coste con datos limpios es una operación vectorizada por fold. Un detalle que costó encontrar: el texto de `MissingExogWarning` imprime un `set`, cuyo orden depende de cómo se construye; hay que construirlo igual (`set(levels) - set(keys)`). No cubiertos, y validables una vez porque son propiedades del objeto y no del fold: columnas que faltan, Series con nombre desconocido, índice duplicado o de otro tipo, dtypes distintos de los de entrenamiento.

**Lo que el prototipo enseña sobre el coste de implementar.** Son unas 460 líneas que duplican la semántica de `_extract_data_folds_multiseries`, `check_predict_input`, `_create_predict_inputs` y la cola de `predict`. Funciona, pero es un segundo camino que hay que mantener en paridad (sección 6.2).

### 2.3 Lo alcanzable frente al techo

[review_scripts/time_bt_once.py](review_scripts/time_bt_once.py): mismos datos que el techo, 600 series, exog mixta, 50 folds de 24 pasos ([results/time_bt_once_600_mixed.txt](review_scripts/results/time_bt_once_600_mixed.txt)). El desglose de "hoy" es una ejecución con cronómetros en las funciones; el total, el mejor de 3 sin cronómetros.

**Ridge** (ms, p2 / p3 / p3 pa):

| Etapa | Hoy | Validar una vez (prototipo) |
|---|---|---|
| Total de la llamada | 10 059 / 11 094 / 12 041 | **1940 / 1884 / 2226** |
| `fit` del primer fold | 446 / 850 / 938 | 448 / 340 / 626 |
| Preprocesado de la función pública (una vez, por serie) | 97 / 107 / 84 | 86 / 113 / 81 |
| Extraer folds (por fold y serie) | 4986 / 4958 / 5520 | (primer fold y `cv.split`) 270 / 188 / 197 |
| `check_predict_input` | 2354 / 3110 / 2840 | |
| Resto de `_create_predict_inputs` | 2642 / 1982 / 2900 | |
| Preparación única (matriz, exog codificada del tramo de test, máscaras) | | 474 / 558 / 618 |
| Bucle de folds | (`_recursive_predict`) 29 / 30 / 29 | 59 / 78 / 66 (1,2 a 1,6 ms por fold) |
| Salida (DataFrame, `y_true`, `concat`) | 400 / 425 / 414 | 47 / 50 / 34 |
| Métricas (`_calculate_metrics_backtesting_multiseries`) | 610 / 585 / 768 | 556 / 559 / 604 |
| **Sin el `fit`** | 11 120 / 11 197 / 12 557 | **1491 / 1545 / 1600** |
| Aceleración sin el `fit` / de la llamada | | **7,5 / 7,2 / 7,8** y **5,2 / 5,9 / 5,4** |

**LightGBM (50 árboles):** llamada 12 476 / 14 091 / 14 997 ms hoy, 4463 / 4612 / 4524 con el prototipo: **2,8 / 3,1 / 3,3 veces** (3,9 / 4,8 / 5,0 sin el `fit`). El bucle son 18,3 / 19,4 / 17,5 ms por fold, casi todo el modelo.

**Otros escenarios** ([results/time_bt_once_other.txt](review_scripts/results/time_bt_once_other.txt)):

| Escenario | Aceleración sin el `fit` | Aceleración de la llamada |
|---|---|---|
| Ridge, 600 series, exog mixta, **5 folds** (p2 / p3 / p3 pa) | 1,5 / 1,7 / 1,5 | 1,5 / 1,4 / 1,3 |
| LightGBM, 600 series, exog mixta, **5 folds** | 1,6 / 1,6 / 1,6 | 1,0 / 1,1 / 1,0 |
| Ridge, 100 series, exog mixta, 50 folds (p3 pa) | 7,0 | 6,1 |
| Ridge, 600 series, **sin exog**, 50 folds (p3 pa) | 3,3 | 3,0 (las métricas son 641 de los 958 ms que quedan) |

**Veredicto sobre §5.1: de acuerdo con cambios.**

- El techo es correcto y no he encontrado nada mejor para el bucle (sección 6.2).
- La frase "lo alcanzable debería quedar cerca del techo más un coste fijo por backtesting" es cierta en la forma y engañosa en el fondo: **el coste fijo es de 1,0 a 1,5 s, entre 20 y 50 veces el bucle**. El resultado útil de §5.1 no es "40 a 400 veces por fold" sino "5 a 6 veces la llamada con Ridge y 50 folds, 3 con LightGBM, 1 a 1,5 con 5 folds".
- La fila de P3 ("Ridge ~12 s → ~0,4 s más la preparación") no se alcanza con P3 sola: mido 12,0 → 2,2 s. Para bajar de 1 s hacen falta además las métricas con la fórmula directa (BTPLAN M1 con PERF-03: de unos 600 ms a unos 10 según BTPLAN §1.2) y una preparación más barata.
- "Codificar una vez" tiene que limitarse a las filas de los tramos de test. El prototipo ya lo hace; con la exog entera (lo que mide `preencode.py`: 343 ms con 1500 filas por serie) una historia larga con pocos folds sería más lenta que hoy.

## 3. P1 y P2

### 3.1 P1 (bug de `fit` con DataFrame) y D-NP2

Scripts `review_scripts/p1_*.py` (subagente). He vuelto a ejecutar `verify_cat_bug.py`, `p1_osa.py` y `p1_dnp2.py`. Salida igual en los tres entornos y también en `0.28.x`. Datos: 6 series, 300 filas, 8 categorías con efecto no monótono; el ejemplo del informe (3 categorías, 2 series) es demasiado fácil y esconde las diferencias entre modelo categórico y numérico.

**El bug: de acuerdo, con cambios en el alcance.**

- Mecanismo confirmado. Con `categorical_features=None` la columna `category` no se codifica (`_forecaster_recursive_multiseries.py:1612-1644`), llega al estimador con su dtype (`:1284-1291`, `:2482-2489`) y LightGBM entrena con los códigos; `predict` le da los valores (`:3037`, `:3172`).
- **También XGBoost** (MAE de `predict` 45,8 con categorías 10..80 frente a 0,3 con 0..7). El informe solo nombra LightGBM y HistGradientBoosting.
- **HistGradientBoosting no está afectado.** §4.2 dice que trata la columna como categórica "en `fit`, pero no en `predict`". Es falso: guarda un preprocesador que vuelve a asignar los valores a sus categorías (MAE 2,56 en one-step-ahead y en backtesting, con cualquier disposición de categorías).
- **Categorías `0..n-1` declaradas en otro orden** (`categories=[2, 0, 1]`): predicciones permutadas en silencio. El informe da a entender que con `0..n-1` no pasa.
- **Lista de `categorical_features` que omite una `category`** (el informe: "no comprobado"): confirmado, mal con enteros; con texto, `fit` funciona y `predict` falla con `ValueError`.
- **El valor por defecto es `'auto'`** (`:433`) y con él todo es correcto. El bug exige pasar `None` o una lista incompleta: conviene decirlo en la fila del roadmap para calibrar la urgencia.
- **La validación one-step-ahead lo oculta**, porque ajusta y predice con DataFrame (`model_selection/_utils.py:3024-3038`). MAE medio de 6 series, `None`, categorías 10..80:

  | Estimador | One-step-ahead | Backtesting (1 paso, sin refit) |
  |---|---|---|
  | LightGBM | 3,93 | 31,11 |
  | XGBoost | 1,65 | 42,01 |
  | HistGradientBoosting | 2,56 | 2,56 |

  Una búsqueda con `OneStepAheadFold` devuelve hoy una métrica que `predict` no puede reproducir. Es un riesgo que el informe no recoge.

**El arreglo (`X_train.to_numpy(dtype=float)`): de acuerdo en la dirección, en desacuerdo con "mismo modelo en los 9 casos".** `ms_numpy_fit.py` solo probó `'auto'`. Con `p1_fix_grid.py`:

| Caso | LightGBM | XGBoost (`enable_categorical=True`) | HistGradientBoosting (`from_dtype`) | CatBoost | Ridge |
|---|---|---|---|---|---|
| `'auto'`, cualquier `encoding` | idéntico | idéntico | idéntico | idéntico solo con el cast a entero | idéntico |
| `None`, categorías 0..7 (hoy acierta) | **cambia**: MAE 2,2 → 7,5 | **cambia**: 0,3 → 0,6 | **cambia**: 1,9 → 7,7 | hoy falla en `fit` → ajusta como numérica | idéntico |
| `None`, 10..80 | 36,3 → 7,5 (arreglado, como numérica) | 45,8 → 0,6 | **1,9 → 7,7 (acertaba y empeora)** | falla → ajusta | idéntico |
| `ordinal_category` con `None` | **el nivel deja de ser categórico**: 1,5 → 7,5 | cambia | 0,5 → 7,7 | falla → ajusta | idéntico |

Lo que el arreglo tiene que resolver y la fila P1 no menciona:

- `ordinal_category`: el índice de `_level_skforecast` solo se pasa cuando `categorical_features is not None` (`:2451-2464`). Con `None` el tratamiento categórico viene solo del dtype, que es el propósito documentado de esa codificación (`:111`). Hay que pasar el índice siempre.
- CatBoost: una matriz float con `cat_features` falla (`'data' is numpy array of floating point numerical type`). Hay que usar `cast_catboost_categorical_columns` (`utils.py:734`), como los univariantes.
- Dos sitios más: `set_in_sample_residuals` predice con el DataFrame (`:4695`) y la validación one-step-ahead predice sobre `X_test` en `_utils.py:3038`.
- `fit_kwargs={'categorical_feature': ['c']}` **por nombre** funciona hoy solo en multiserie; con numpy lanza `TypeError`, como ya hace `ForecasterRecursive`.
- El estimador pierde `feature_names_in_`; en LightGBM, `feature_name_` pasa a `Column_0...` (gráficos de importancia y volcados del árbol tomados del estimador). El patrón documentado `forecaster.estimator.predict(create_predict_X(...))` (`docs/user_guides/training-and-prediction-matrices.ipynb`) sigue dando los mismos valores, pero con estimadores de scikit-learn emite `X has feature names, but Ridge was fitted without feature names`, igual que hoy `ForecasterRecursive`. Con `ordinal_category`, ese patrón hoy **falla** en LightGBM y XGBoost y con el arreglo funciona. SHAP no se pudo comprobar (no está instalado).
- Forecasters guardados con versiones anteriores: `predict` idéntico tras `save_forecaster` y `load_forecaster` (el arreglo no toca `predict`); el modelo solo cambia al reajustar.

**D-NP2: en desacuerdo con recomendar (a).** La justificación del informe ("es lo que recibe hoy `predict`, así que `fit` y `predict` coinciden sin cambiar el `predict` de nadie") falla dos veces: `predict` recibe los **valores**, no los códigos del `OrdinalEncoder` (con (a) literal cambian los modelos lineales con categorías no equiespaciadas, diferencia máxima 21), y sí cambia las predicciones de quien hoy acierta. Referencia: el modelo de hoy evaluado como se ajustó.

| Opción | LightGBM | XGBoost | HistGradientBoosting |
|---|---|---|---|
| (a) códigos como numérica | cambia (hasta 19,5), MAE 3,93 → 4,33 | cambia, 1,65 → 2,14 | cambia, 2,56 → 4,29 |
| (a') valores como numérica (lo que hace `ForecasterRecursive`) | igual que (a) | igual que (a) | igual que (a) |
| (c) pasar el índice como categórica | **mismo modelo que hoy**, 3,93 | mismo modelo, 1,65 | mismo modelo, 2,56 |

Mi recomendación: **(c) limitada a las columnas `category` y a la columna de nivel**: mismo modelo ajustado que hoy y `predict` pasa a ser correcto. El argumento del informe contra (c) ("cambia el significado de `None`") es real, pero ese significado ya es hoy "categórica por dtype" en multiserie. Si se prefiere que `None` signifique "numérica" como en `ForecasterRecursive`, entonces (a') o (b), publicado como cambio de comportamiento con nota de versión y un aviso cuando una `category` se encuentra con `None`. Lo que no es aceptable es (a) presentada como arreglo sin cambio de modelo.

Aparte (bug existente en todos los forecasters, visto por el subagente, no verificado por mí): un `Pipeline` cuyo último paso es LightGBM, con `'auto'` y una exog categórica, falla en `fit` porque `configure_estimator_categorical_features` pone `categorical_feature` sin el prefijo del paso (`utils.py:672-682`).

### 3.2 P2 (exog a float) y D-NP3

Scripts `review_scripts/p2_*.py` (subagente; el prototipo sustituye `DataFrame.to_numpy` solo en la llamada sin argumentos sobre `exog` dentro de `_create_train_X_y` y `_create_predict_inputs`). Salidas en `review_scripts/results/p2_*.txt`. He comprobado yo la ganancia y el bug de HistGradientBoosting.

**De acuerdo con cambios.** La ganancia se reproduce (`fit` de `ForecasterRecursive` con Ridge, 100 000 filas, una columna `bool` entre tres float: 232 / 217 / 251 ms frente a 37 / 36 / 40 con la columna pasada a float; [results/time_small.txt](review_scripts/results/time_small.txt)) y las predicciones son idénticas donde hoy funciona (Recursive, Direct y DMV con Ridge, LightGBM, HistGradientBoosting, XGBoost y CatBoost). Pero "resultado idéntico" y "riesgo bajo" no se sostienen tal como está escrita la condición ("todas las columnas numéricas o `bool`"):

| Caso | Hoy | Con P2 según el informe | Qué hacer |
|---|---|---|---|
| `complex128` | El estimador lo rechaza ("Complex data not supported") | `fit` funciona y la parte imaginaria se pierde, solo con un `ComplexWarning` de numpy | Comprobar dtypes explícitos (bool, enteros y float de numpy, nullable o `pyarrow`), no `is_numeric_dtype` |
| Enteros mayores que 2^53 mezclados con `bool`, nullable o `pyarrow` | La matriz es `object` y **salen exactos** de `create_train_X_y` y `create_predict_X`. "Ya se pierden" solo es cierto para lo que ve el estimador | Se redondean. `UInt64` cerca de 2^64 más float: `create_train_X_y` pasa a lanzar `TypeError: cannot safely cast` | Decidir y documentar, o excluir |
| `decimal128[pyarrow]` | Exacto en la matriz `object` | Pierde exactitud | Excluir |
| Tablas que hoy **no** son `object`: todo `int64`, todo `bool`, `float32`, un solo dtype nullable | `to_numpy()` devuelve una vista | Una copia float64 más antes de `np.concatenate` | Convertir solo cuando hoy saldría `object` (más de un dtype y alguno `bool` o de extensión) |
| `predict` de multiserie | `:3037` da un array `object` con `bool`, `boolean`, `Int64`, `Float64` o `pyarrow` | **No está en la lista de sitios de P2** | Añadirlo |
| `bool[pyarrow]` | | `is_numeric_dtype` es `False` | Incluir `is_bool_dtype` |
| Texto, fechas, `timedelta` | No cumplen la condición: sin cambio | Sin cambio (pero un `to_numpy(dtype=float)` sin condición convertiría fechas a enteros, con valores distintos en pandas 2.3 y 3) | Mantener la condición |

Dos cosas a favor que el informe no cuenta:

- **P2 arregla un bug existente.** `HistGradientBoostingRegressor` con `categorical_features='auto'` (el valor por defecto), una columna categórica y una `bool` o nullable ajusta y falla en `predict` con `TypeError: ufunc 'isnan' not supported` (comprobado por mí en los tres entornos, [review_scripts/spot_checks.py](review_scripts/spot_checks.py)). Con `ForecasterRecursiveClassifier` y `HistGradientBoostingClassifier` basta una `bool` mezclada con float. Un forecaster guardado antes del cambio sigue fallando hasta reajustarlo.
- No rompe tests: con el prototipo activo pasan las carpetas de tests de Recursive, Classifier, Direct y DMV (1675 en pandas 3, 1677 en pandas 2.3). También significa que **no hay ningún test** que cubra estos dtypes: hay que escribirlos.

**I10 y D-NP3 (NA a NaN): de acuerdo con (b), con los hechos corregidos.** Un solo dtype nullable ya da hoy float64 con NaN (el `TypeError` exige mezcla de dtypes); en `predict` el `TypeError` lo lanza skforecast (`_forecaster_recursive.py:1789`); con `dropna_from_series=True` hoy ya funciona; y multiserie con LightGBM y una columna `pyarrow` **falla en `fit`** (`pandas dtypes must be int, float or bool`), en contra de "multiserie con LightGBM entrena". Con (b), Ridge pasa de `TypeError` a `ValueError: Input X contains NaN`, que es lo que ya ocurre con un NaN float.

`ForecasterRnn` no se pudo ejecutar (sin torch). Por lectura, no tiene codificación de categóricas ni `check_exog_dtypes`: la premisa "tras codificar" no existe allí y hay que decidirlo aparte.

## 4. Veredicto por propuesta

"Subagente" = comprobado por un subagente con el script indicado, en los tres entornos; "yo" = ejecutado por mí.

| # | Veredicto | Prueba |
|---|---|---|
| P1 | **De acuerdo con cambios** (bug real y más amplio; el arreglo cambia modelos con `None`; faltan sitios) y **en desacuerdo con D-NP2 (a)** | Sección 3.1 |
| P2 | **De acuerdo con cambios** (condición más estricta, convertir solo si hoy sale `object`, añadir multiserie `:3037`, decidir enteros grandes) | Sección 3.2 |
| P3 | **De acuerdo con cambios**: viable y con salidas y avisos idénticos, pero la ganancia real es 5 a 6 veces la llamada (Ridge, 50 folds), no 40 a 400; necesita las métricas en numpy para lucir; preparación solo del tramo de test; esfuerzo L solo para el alcance de D-NP4 | Sección 2; yo |
| P4 | **De acuerdo con cambios.** `get_indexer` difiere del `OrdinalEncoder` cuando `fit` vio un valor faltante (`categories_` acaba en NaN): el codificador da NaN y `get_indexer` el último código. Incluye el caso por defecto de pandas 3: columna de texto con `None` (dtype `str`): `[0, nan, 1, 0]` frente a `[0, 2, 1, 0]`. En pandas 2.3 (`object`) coinciden. También difiere con `bool` en `fit` e `int` en `predict`. Usar `.cat.codes` directamente es incorrecto en 5 casos; una búsqueda por categoría (`get_indexer(s.cat.categories)[codes]`) coincide en todos. Hay que quitar los NaN de `categories_` antes de indexar | `misc_p4_get_indexer.py` (subagente, 38 casos); el caso `str` con `None`, yo (`spot_checks.py`) |
| P5 | **No verificado** más allá del coste base: fold univariante con Ridge y exog float 0,37 / 0,41 / 0,45 ms, con LightGBM 0,71 / 1,08 / 1,34 ms (`time_small.py`, yo). Los porcentajes (-17, -31, -68 %) no los he reproducido. Los avisos por fold se pueden conservar igual que en multiserie (sección 2.2) | |
| P6 | **De acuerdo con cambios.** `_is_utc_anchored_index` depende de la ventana: con un índice diario anclado a las 06:00 UTC mostrado en Europe/Madrid, el índice entero da `True` y, de 27 ventanas de 14 filas, 2 dan `True` y 25 `False`. Una caché por serie o por backtesting **cambia resultados**. Hay una formulación exacta sin caché (`asi8 % día` en UTC y en local), sin medir | `misc_p6_utc_anchored.py` (subagente) |
| P7 | **De acuerdo.** `identical` en DatetimeIndex (con y sin `freq`, zona horaria, nombre), RangeIndex e Index entero, tres entornos; nada lee `training_range_.freq`. 55 / 43 / 40 µs frente a 8 / 9 / 10 µs | `misc_p7_training_range.py` (subagente); tiempos, yo |
| P8 | **De acuerdo, con su condición.** La caché por `id` solo acierta cuando todas las series tienen exactamente el índice de la exog; con una serie que empieza más tarde, exog más larga que las series, exog desordenada o un dict construido del mismo DataFrame, los objetos ya son distintos (`utils.py:5024-5056`, `:5170-5192`) y la caché falla, correctamente. La ganancia de la fila solo vale en el primer caso | `misc_p8_wide_exog_identity.py` (subagente) |
| P9 | **De acuerdo.** `a == b` entre dtypes coincide con `nunique` en 18 pares. Tres trampas: `str(dtype)` no sirve (todas las categóricas imprimen `category`); comparar por nombre de columna, no listas enteras (series con columnas distintas se aceptan hoy); conservar la segunda pasada para los dtypes de texto (`utils.py:5080-5087`) | `misc_p9_dtype_equality.py` (subagente) |
| P10 | **De acuerdo con cambios.** La versión numpy da el mismo diccionario y el mismo resultado final tras el submuestreo, pero exige ordenación estable y claves ascendentes (el `rng` se consume en ese orden). **Las claves ya son `int` de Python hoy**: no hay decisión "`np.int64` a `int`"; al revés, hay que hacer `.tolist()` para no introducir `np.int64`. Sin ejecutar para multiserie ni `ForecasterEquivalentDate` | `misc_p10_oos_binning.py` (subagente) |
| P11 | **En desacuerdo con "bit a bit igual".** En pandas 3 sí (563 cuantiles). **En pandas 2.3 no**: 51 a 75 de 563 cuantiles difieren (hasta 6e-14 relativo), entre ellos `0.5 - 0.95 / 2`, es decir `interval=0.95`, y `1/3`. Los literales 0,025, 0,05, 0,1 coinciden. Con NaN, `np.nanquantile` coincide pero emite `RuntimeWarning: All-NaN slice`, y las filas todo NaN son alcanzables en backtesting (`_allow_nan_predictions`). Reutilizar entradas y predicción puntual sí es seguro (el `rng` solo muestrea residuos antes de la recursión) | `misc_p11_p12_intervals.py` (subagente; reejecutado por mí) |
| P12 | **De acuerdo con cambios.** La búsqueda en array por clave es idéntica; por posición del diccionario es incorrecta (orden `[0, 3, 1, 2]` cuando hubo un bin vacío). La caché es la mitad arriesgada: `set_out_sample_residuals(append=True)` modifica el mismo diccionario, los residuos se asignan directamente en `_validation.py:157-159` y no existe ningún punto de invalidación. Hacer solo la búsqueda en array | Mismo script |
| P13 | **De acuerdo con cambios** (lectura de código). "Como el DMV" vale para la representación, no para la lógica: multiserie revierte diferenciación y transformación por nivel, maneja niveles ausentes y longitudes distintas, pasa Series a las métricas y hace pooling. La cifra no está verificada | `model_selection/_utils.py:3002-3220` frente a `:2891-2990` |
| P14 | **De acuerdo con cambios.** `agg_func(axis=...)` es bit a bit igual a `np.apply_along_axis` para `np.median` siempre, y para `np.mean`, `np.nanmean`, `np.sum`, `np.std` **solo hasta 7 valores**: desde 8 difiere (suma por pares frente a secuencial). Una función de usuario sin `axis` lanza `TypeError`; hace falta lista blanca | `misc_i3_p14_eqdate.py` (subagente) |
| P15 | **De acuerdo en la lectura; cifra sin verificar.** `_build_predict_function` está escrita para regresores: cada rama de clasificador es código nuevo (probabilidades colocadas con `proba_columns`, LightGBM binario devuelve solo P(clase 1), y `LogisticRegression.predict` usa `decision_function`, no el argmax de probabilidades). Esfuerzo M es optimista | `_forecaster_recursive_classifier.py:1700-1746`, `utils.py:4413` |
| P16 | **De acuerdo con cambios.** Valores idénticos, pero en **pandas 3** un único `to_numpy(dtype=float)` de una tabla float devuelve una **vista de solo lectura que comparte memoria con el `last_window` del usuario**: `copy=True` obligatorio. Seleccionar columnas antes de convertir (una columna de texto de más, que hoy funciona, haría fallar la conversión) y conservar una guarda: con una columna que falta y `check_inputs=False`, `get_indexer` da -1 y se tomaría la última columna en silencio | `misc_p16_dmv_last_window.py` (subagente) |
| P17 | **De acuerdo para el DMV; la parte de multiserie es falsa.** `ForecasterRecursiveMultiSeries` nunca llega a ese `drop`: la función pública convierte las series en diccionario antes (`_validation.py:2307-2308`). Sin efecto observable en 26 configuraciones. 67 / 61 / 75 µs por llamada | `misc_p17_drop_empty.py` (subagente); tiempo, yo |

## 5. Veredicto por incoherencia

| # | Veredicto | Prueba |
|---|---|---|
| I1 | **De acuerdo con cambios**: cierto, pero HistGradientBoosting no está afectado y XGBoost sí (sección 3.1) | `p1_mechanism.py`, `p1_fix_grid.py` |
| I2 | **De acuerdo con dos precisiones**: una tabla solo `bool` o de un solo dtype nullable no es `object` (hace falta mezcla); y el `predict` de multiserie también produce `object` (`:3037`) | `p2_01`, `p2_07`, `p2_08` |
| I3 | **De acuerdo con cambios.** Se reproduce, pero el cuadro es más ancho: con `DateOffset` el resultado depende de si la función de numpy delega en el método de pandas (`np.mean` y `np.max` saltan el NaN, `np.median` no, y una función propia `sum/len` da 23,0, que no es ninguna de las dos convenciones). Saltar fechas equivalentes que faltan es diseño probado (`test_predict.py:201-238`); lo no cubierto es el NaN dentro de `y`. Además, dentro del mismo forecaster con `DateOffset`, los residuos in-sample (`:526`) propagan y `predict` (`:727`) salta | `misc_i3_p14_eqdate.py` |
| I4 | **De acuerdo**, con precisiones: `_create_predict_inputs` y `create_predict_X` sí aceptan `check_inputs`, y `predict_proba` sí lleva `@manage_warnings`; el que no tiene ninguno es `predict` (`:1877-1882`) | Lectura |
| I5 | **En desacuerdo con el efecto descrito.** Con zona horaria frente a sin ella, `==` devuelve todo `False` y los dos forecasters lanzan el mismo `ValueError` previsto; no hay `TypeError`. La diferencia real es otra y solo en pandas 2.3: mismos instantes como `datetime64[s]` y `[ns]`, que `ForecasterRecursive` rechaza y el clasificador acepta. Cambiar a `.equals` no es idéntico en pandas 2.3. Además multiserie también usa `==` para las window features (`:942`) | `misc_i5_index_eq.py` |
| I6 | **De acuerdo con cambios** (ver P13) | Lectura |
| I7 | **De acuerdo** en que el camino difiere; tiempos no verificados (ver P10) | `misc_p10_oos_binning.py` |
| I8 | **De acuerdo, y peor de lo descrito.** Con una serie sin exog: `int64` pasa a `float64`; una columna `bool` pasa a `object`, y entonces con `categorical_features=None` LightGBM falla en `fit`, y con `'auto'` la columna **se codifica en silencio como categórica** (con todas las series presentes sigue siendo `bool` y no se codifica): cambia la semántica del modelo | `misc_i8_exog_none_dtypes.py` |
| I9 | **De acuerdo** | `misc_i9_i11_attrs.py` |
| I10 | **De acuerdo con cambios** (sección 3.2) | `p2_05`, `p2_08` |
| I11 | **De acuerdo con una corrección**: `feature_names_in_` es el nombre de la serie, no literalmente `['y']` | `misc_i9_i11_attrs.py` |
| I12 | **De acuerdo** en la lectura del código (`_forecaster_direct_multivariate.py:2324-2337`, `utils.py:1617-1619`); tiempos no verificados | Lectura |

## 6. El veredicto y el orden del informe

### 6.1 D-NP1: multiserie antes que los univariantes

**De acuerdo.** Un fold univariante cuesta 0,37 / 0,41 / 0,45 ms con Ridge y 0,71 / 1,08 / 1,34 ms con LightGBM (50 folds, exog float); uno de multiserie con 600 series, 180 a 250 ms. Dos matices:

- No son excluyentes: P5 es independiente y de esfuerzo M. El orden importa por el diseño (el núcleo numpy de multiserie decide parte de P5), no por las horas.
- La comparación del informe es por fold; por llamada, lo que gana el usuario de multiserie depende del número de folds y del coste del `fit` (sección 2.3).

### 6.2 ¿Hay una alternativa mejor que el contenedor interno?

El contenedor es la idea correcta. Tres cambios sobre cómo lo plantea el informe:

1. **Un solo núcleo numpy, compartido con `predict`.** El informe coloca el contenedor "dentro de `_backtesting_forecaster_multiseries`" y hace que el backtesting llame a `_recursive_predict` directamente. Eso obliga a reimplementar fuera del forecaster lo que hay entre la validación y la recursión (`_create_predict_inputs:2903-3043`) y la cola de `predict` (`:3710-3728`): mi prototipo son 460 líneas de esa duplicación. Mejor partir `predict` en validar, preparar y un núcleo que recibe arrays (ventana transformada, exog por paso, calendario) y devuelve arrays; `predict` y el backtesting comparten el núcleo y solo difieren en quién prepara. Así una corrección futura en `predict` llega sola al backtesting.
2. **El contenedor debe vivir por encima del backtesting.** Las búsquedas llaman a la función pública `backtesting_forecaster_multiseries` por candidato (`_search.py:1815`, `:2269`), que repite el preprocesado (84 a 220 ms con 600 series) y repetiría la preparación de P3 (0,5 a 0,6 s). La matriz de series y la alineación de la exog no dependen del candidato. En una búsqueda de 50 candidatos sin refit eso es más tiempo que todos los bucles de folds juntos.
3. **Preparación perezosa y acotada**: solo las filas de los tramos de test, y con límite de memoria. La exog codificada ocupa filas de test × objetos × columnas × 8 bytes: 29 MB en el caso medido, pero 10 000 series × 8760 filas × 20 columnas son 14 GB, que el camino actual nunca materializa a la vez. Hace falta trocear o volver al camino actual.

Alternativas que he considerado y descartado, con medida:

- **Predecir todos los folds en una sola pasada recursiva** (sin refit todos usan el mismo modelo: apilar los folds como filas y llamar al estimador `steps` veces en vez de `n_folds × steps`). [review_scripts/proto_bt_batched.py](review_scripts/proto_bt_batched.py), 600 series, exog mixta, p3 pa: con Ridge es **más lento** (0,41 → 0,69 ms por fold, por caché); con LightGBM, igual (15,6 → 17,9 ms con un hilo; 7,5 → 7,8 con cuatro). Predicciones idénticas. No compensa: el techo del informe es un techo.
- **Solo los pasos baratos de BTPLAN (F1 a F3 y M1) sin camino nuevo:** la extracción es el 40 a 45 % del tiempo y `check_predict_input` el 19 a 28 %; F1 a F3 atacan lo primero. Es menos ganancia (BTPLAN mide -20 % con LightGBM) pero sin segundo camino. No es alternativa a P3, es su primer paso.

### 6.3 ¿Está bien estimado el esfuerzo de P3?

**L es correcto solo para el alcance de D-NP4, y con una condición.** A favor: el prototipo cubre ese alcance (más `transformer_series`, diferenciación, window y calendar features, todas las codificaciones y los avisos) con salidas idénticas, así que el riesgo técnico es menor de lo que sugiere "Alto". En contra, lo que el prototipo no hace y la versión real sí necesita: detección de condiciones y vuelta al camino actual, forecaster ya ajustado y niveles desconocidos, diferenciación en diccionario, `n_jobs` explícito, lista blanca de transformadores, límite de memoria, y una batería de paridad entre los dos caminos que hay que mantener para siempre. Con refit entero o intervalos es XL. La condición: si no se hace antes el núcleo compartido (6.2, punto 1), el coste de mantenimiento posterior es mayor que el de escribirlo.

Sobre D-NP4: `transformer_exog=None` es más restrictivo de lo necesario (un `StandardScaler` aplicado una vez da resultados idénticos), pero es una primera versión razonable.

### 6.4 El resto del veredicto

- "numpy hace desaparecer la diferencia entre pandas 2 y pandas 3": cierto en el bucle (1,2 / 1,6 / 1,3 ms por fold), no del todo en la llamada, porque la preparación y las métricas siguen en pandas (prototipo sin el `fit`: 1491 / 1545 / 1600 ms).
- "Las categóricas no justifican pandas aguas abajo": de acuerdo, con la salvedad de P1: con `categorical_features=None` el dtype es hoy lo único que hace categórica a una columna en multiserie, y eso es comportamiento, no solo un bug.
- "P1 se justifica por corrección, no por tiempo": de acuerdo.

## 7. Riesgos que el informe no recoge

1. **Expectativa de ganancia.** Quien lea el resumen espera uno a dos órdenes de magnitud. Con salidas idénticas se obtienen 5 a 6 veces (Ridge, 50 folds), 3 (LightGBM) y 1 a 1,5 (5 folds). Conviene fijar el criterio de aceptación de P3 en tiempo de la llamada completa, no por fold.
2. **Regresión por preparar de más** con pocos folds o historia larga, y **memoria** de la exog materializada (6.2, punto 3).
3. **Dos caminos de predicción** que pueden divergir, y las búsquedas, que no se benefician si el contenedor vive dentro del backtesting (6.2).
4. **D-NP2 (a) cambia en silencio el modelo** de quien hoy usa `category` de enteros `0..n-1` con `categorical_features=None`, y de todo usuario de HistGradientBoosting con una `category` (3.1).
5. **Métricas de búsqueda no reproducibles hoy:** one-step-ahead y backtesting difieren por el bug P1 (3,9 frente a 31,1). Cualquier arreglo cambia una de las dos; hay que decirlo en la nota de versión.
6. **Efectos del arreglo de P1 fuera de las predicciones:** se pierde `feature_names_in_` y los nombres de LightGBM, `fit_kwargs` con `categorical_feature` por nombre deja de funcionar, CatBoost necesita el cast a entero, y `ordinal_category` necesita su índice siempre.
7. **P2:** `complex128` aceptado, enteros grandes redondeados en `create_train_X_y`, copias nuevas, y forecasters con HistGradientBoosting guardados que siguen fallando hasta reajustar.
8. **Propuestas "bit a bit iguales" que no lo son:** P11 en pandas 2.3, P14 desde 8 offsets, P4 con NaN visto en `fit` (incluido el dtype `str` de pandas 3), P6 con caché, P16 con vista de solo lectura en pandas 3.
9. **Cobertura de tests inexistente** para exog `bool`, nullable y `pyarrow`: P2 no rompe ningún test porque ninguno los prueba.
10. **Condiciones de medida del informe:** `OMP_NUM_THREADS=1`, datos sintéticos con series iguales y completas, y medidas hechas con subagentes a la vez. Las cifras de §5.1 se reproducen, pero el reparto modelo frente a resto cambia con varios hilos.
11. **I8 con `bool`:** que una serie no tenga exog cambia que una columna `bool` se trate como categórica. Es un cambio de modelo que depende de los datos y merece fila propia.

## 8. Orden recomendado

| Cuándo | Qué | Por qué, y qué cambia respecto al informe |
|---|---|---|
| 0.28.x | **P1 con D-NP2 decidida explícitamente**: (c) para `category` y columna de nivel, o cambio de comportamiento anunciado. Incluir `:4695`, `_utils.py:3038`, el cast de CatBoost y el índice de `ordinal_category` | El informe lo da como S a M y sin cambio de modelo; con `None` sí lo hay. Antes de fusionar: tests de paridad one-step-ahead frente a backtesting |
| 0.28.x | **P2 con condición estricta** (dtypes explícitos; solo si hoy sale `object`), más multiserie `:3037`, más tests de dtypes | Corrige además el fallo de HistGradientBoosting en `predict`. Decidir D-NP3 (b) y los enteros grandes |
| 0.28.x | P7, P17 (solo DMV), P8 (con su condición), P16 (con `copy=True` y selección previa de columnas) | Como el informe, con las condiciones de la sección 4 |
| 0.28.x o 0.29.0 | **BTPLAN M1 con PERF-03 (métricas) y F1 a F3 (extracción por posición)** | **Antes** de P3, no absorbidos por ella: dan ganancia sin segundo camino, y sin M1 la mitad de lo que deja P3 son métricas |
| 0.29.0 | P9, P10 (claves `int`, orden estable), P12 (solo la búsqueda en array), P11 **solo la reutilización de entradas**; el cambio a `np.quantile`, únicamente si se acepta que no es idéntico en pandas 2.3 | El informe los da como "baratos, solo numpy"; tres de los cuatro necesitan cambios |
| 0.29.0 | **P3 en dos pasos:** (1) partir `predict` en validar, preparar y núcleo numpy, sin cambio de comportamiento; (2) camino rápido del backtesting sobre ese núcleo, con el contenedor disponible para las búsquedas y preparación acotada | El informe lo plantea como un contenedor dentro del backtesting. Aceptación: salidas y avisos idénticos (la batería de `proto_bt_once_cases.py` es un punto de partida) y tiempo de la llamada completa |
| 0.29.0 | P5, P6 (formulación exacta sin caché), P13 | Como el informe |
| Por decidir | P4 (tras definir el tratamiento de NaN en `categories_`), P14 (tras I3 y con lista blanca), P15, I8 (`bool` con series sin exog), I5 (unificar a `.equals` es un cambio en pandas 2.3) | |

## Anexo: qué no se ha verificado

- `ForecasterRnn` y los adaptadores foundation (sin torch ni keras): solo lectura de código.
- SHAP sobre el estimador tras el arreglo de P1 (no instalado); versiones de XGBoost en las que `enable_categorical` es `False` por defecto.
- Los porcentajes de P5 y las ganancias de P6, P8, P9, P10, P12, P13, P15 y P16 (solo corrección o lectura).
- P10 en multiserie y `ForecasterEquivalentDate`; P11 con filas parcialmente NaN.
- El prototipo de P3 no cubre refit, intervalos, `return_predictors`, `n_jobs > 1`, forecaster ya ajustado ni diferenciación en diccionario; tampoco se ha probado con transformadores que no sean elemento a elemento ni con más de 600 series.
- La suite completa de tests no se ha lanzado. El subagente de P2 ejecutó cuatro carpetas de tests con su prototipo activo, en pandas 3 y pandas 2.3.

## Anexo: cómo reproducir

```bash
git worktree add /home/user/sk_perf ab62227          # árbol medido
uv venv ~/.venvs/skforecast-p2 --python 3.12
uv pip install --python ~/.venvs/skforecast-p2/bin/python -e ".[test]" "pandas<3" lightgbm pyarrow xgboost catboost
cd dev/pandas_numpy_review/review_scripts

bash run3.sh proto_bt_once_cases.py                  # 2.2: 25 casos, salidas y avisos
bash run3.sh time_bt_once.py 600 mixed ridge 50 24   # 2.3 (a solas; unos 2 minutos por entorno)
bash run3.sh time_bt_once.py 600 mixed lgbm 50 24
bash run3.sh time_bt_once.py 600 mixed ridge 5 24    # pocos folds
python proto_bt_batched.py 600 mixed lgbm            # 6.2 (sin OMP_NUM_THREADS=1 para ver 4 hilos)
bash run3.sh time_small.py                           # D-NP1, P2, P7, P17
bash run3.sh p1_osa.py; bash run3.sh p1_dnp2.py; bash run3.sh p1_fix_grid.py lgbm xgb hgb catboost ridge
bash run3.sh p2_01_dtype_matrix.py                   # y el resto de p2_*.py
bash run3.sh misc_p11_p12_intervals.py               # y el resto de misc_*.py
```
