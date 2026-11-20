# Revisión independiente del estudio "pandas frente a numpy"

Fecha: 2026-11-19 (segunda ronda y prueba con casos reales añadidas el mismo día, secciones 9 y 10). Revisa [INFORME.md](INFORME.md) (`fddf0de`). Árbol medido: `perf/multiseries-predict-backtesting` en `ab62227`, el mismo del informe. No se ha modificado nada de `skforecast/`. Scripts en [review_scripts/](review_scripts/) (ver su [README](review_scripts/README.md)).

**Entornos.** Python 3.12.3, numpy 2.5.5, scikit-learn 1.9.1, numba 0.68.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, 4 núcleos. **p2** = pandas 2.3.3 (con `pyarrow` 26.0.1 instalado); **p3** = pandas 3.1.1 con la importación de `pyarrow` bloqueada; **p3 pa** = pandas 3.1.1 con `pyarrow` 26.0.1. Las cifras van siempre en el orden p2 / p3 / p3 pa. Los tiempos se midieron uno a uno, sin nada más en marcha y con `OMP_NUM_THREADS=1` (salvo donde se dice); las comprobaciones de corrección las hicieron tres subagentes en paralelo y sin medir tiempos. Lo que afirmo de sus informes lo he vuelto a ejecutar yo cuando contradice al informe (se indica); lo demás se marca como "subagente".

Las filas, decisiones y el orden por valor que salen de esta revisión están en [../ROADMAP.md](../ROADMAP.md) (abreviatura NPREV).

## Resumen

1. **El techo de §5.1 es correcto como techo del bucle, pero el titular está sobrevendido.** Se reproduce (Ridge, 600 series, exog mixta: 179 / 200 / 225 ms por fold hoy, 0,57 / 0,57 / 0,64 en el techo). Lo que no dice el informe es cuánto pesa lo que el techo deja fuera. He escrito un prototipo que valida y prepara una vez y devuelve **las mismas métricas, las mismas predicciones y los mismos avisos en el mismo orden** que el código actual (25 casos, tres entornos). Con él, el mismo backtesting pasa de 10,1 / 11,1 / 12,0 s a 1,9 / 1,9 / 2,2 s: **5 a 6 veces la llamada completa (7 a 8 sin el fit), no 40 a 400**. Con LightGBM, 2,8 a 3,3 veces. Con 5 folds en vez de 50, 1,3 a 1,5 veces con Ridge y 1,0 con LightGBM. El bucle de folds queda en 1,2 a 1,6 ms por fold; lo que domina después son costes fijos que el informe despacha en una frase: las métricas (0,56 a 0,60 s) y la preparación (0,47 a 0,62 s).
2. **La buena noticia que el informe no da:** los avisos por fold y su orden, que el informe cuenta como obstáculo real (§4.3), se reproducen exactamente desde máscaras calculadas una vez. La validación no impide el camino rápido.
3. **P1 es un bug real y más amplio de lo descrito, y la opción (a) de D-NP2 no es un arreglo neutro.** También afecta a XGBoost, a CatBoost (por `fit_kwargs`) y a categorías `0..n-1` declaradas en otro orden; HistGradientBoosting **no** está afectado (el informe dice que sí); la validación one-step-ahead lo oculta (MAE 3,9 frente a 31,1 en backtesting). El arreglo "numpy en `fit`" no es "mismo modelo": con `categorical_features=None` cambia el modelo de quien hoy acierta. **Corrección de la segunda ronda:** la opción (c) que recomendé en la primera ("pasar el índice") tampoco basta tal como la formulé; la decisión es un compromiso real y mi recomendación pasa a ser (a') anunciada como cambio de comportamiento (sección 9.4).
4. **P2 es correcta en su objetivo, con riesgo mayor que "bajo".** La condición tal como está escrita acepta `complex128`, redondea enteros grandes que hoy salen exactos de `create_train_X_y`, añade copias donde hoy hay vistas y olvida el `predict` de multiserie. Además arregla un bug que el informe no vio (HistGradientBoosting con categórica y `bool` falla en `predict`).
5. **De las propuestas pequeñas, seis necesitan cambios para ser idénticas:** P4 (NaN visto en `fit`), P6 (una caché cambia resultados), P11 (no es bit a bit en pandas 2.3), P14 (solo hasta 7 offsets), P16 (vista de solo lectura en pandas 3) y P12 (caché sin punto de invalidación). I5 no se reproduce como se describe y la parte de multiserie de P17 es falsa.
6. **Orden:** de acuerdo con multiserie antes que univariantes (D-NP1), pero las métricas en numpy (BTPLAN M1 y PERF-03) tienen que ir **antes o con** P3, no después: tras P3 son el 40 % de lo que queda. Y el contenedor debe vivir por encima del backtesting (lo usan las búsquedas, que hoy repiten todo por candidato) y apoyarse en un único núcleo numpy compartido con `predict`.
7. **Segunda ronda (sección 9): P3 solo rinde sin refit.** Con refit en cada fold el prototipo gana 1,05 a 1,2 veces, porque domina el `fit` (y dentro de él, con Ridge, la creación de las matrices de entrenamiento: 62 a 74 %). En una búsqueda de 20 candidatos gana 1,8 veces por sí solo, 2,2 a 2,4 con la preparación compartida y 3,0 a 3,4 con las métricas en numpy; lo que queda es otra vez el `fit`, que reconstruye las mismas matrices en cada candidato. Con 3000 series la ganancia y la memoria se mantienen. Los porcentajes de P5 se reproducen.
8. **Casos reales (sección 10): el prototipo da salidas idénticas, pero la ganancia depende de los folds.** En los backtestings de tres artículos de cienciadedatos.net (de 5 a 1578 series, de 9 a 31 folds) la llamada completa gana de x1,0 a x1,8, porque el `fit` domina; con una evaluación de 61 folds (1578 edificios, `steps=1`) gana x2,8 a x3,3. El camino rápido pasa a ser una mejora para evaluaciones de muchos folds, su primer paso (partir `predict`) se justifica por calidad del código, y en la búsqueda con `OneStepAheadFold` que usan esos artículos el 91 al 98 % del tiempo es LightGBM: lo que no es modelo, aunque costara cero, daría como mucho x1,10. Lo que más aporta este trabajo es la corrección (P1, P2, I8) y la calidad del código; el rendimiento, solo con muchos folds o modelos baratos.

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
| `refit` entero o `True` | No | No en `proto_bt_once.py`. **Cubierto en la segunda ronda** (`proto_bt_blocks.py`): idéntico con `refit=True` y `refit=3`, preparando por tramo entre refits | Alcanzable, pero rinde poco (9.1) |
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

**Recomendación (corregida en la segunda ronda, ver 9.4).** En la primera ronda recomendé (c) "limitada a las columnas `category` y a la columna de nivel". Un segundo subagente, encargado de romperla, demostró que esa formulación no basta: hay que codificar a los códigos de pandas en `fit` y en `predict`, solo para LightGBM, XGBoost y CatBoost sin envolver, y ni así conserva el modelo en todos los casos. También corrige mi frase "(a) es una degradación silenciosa": es un cambio silencioso de modelo cuyo signo depende del estimador y de los datos. Lo que sigue en pie es que (a), presentada como arreglo sin cambio de modelo, no es aceptable.

Aparte (bug existente en todos los forecasters, visto por el subagente, no verificado por mí; ya está en el roadmap como CAT-T3): un `Pipeline` cuyo último paso es LightGBM, con `'auto'` y una exog categórica, falla en `fit` porque `configure_estimator_categorical_features` pone `categorical_feature` sin el prefijo del paso (`utils.py:672-682`).

### 3.2 P2 (exog a float) y D-NP3

Scripts `review_scripts/p2_*.py` (subagente; el prototipo sustituye `DataFrame.to_numpy` solo en la llamada sin argumentos sobre `exog` dentro de `_create_train_X_y` y `_create_predict_inputs`). Las salidas se regeneran con `run3.sh`. He comprobado yo la ganancia y el bug de HistGradientBoosting.

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

"Subagente" = comprobado por un subagente con el script indicado, en los tres entornos; "yo" = ejecutado por mí. En la segunda ronda volví a ejecutar yo los scripts de P6, P8, P9, P10, P12, P14, P16, I3, I5 e I8: todos dan lo que dice la tabla (sección 9.6).

| # | Veredicto | Prueba |
|---|---|---|
| P1 | **De acuerdo con cambios** (bug real y más amplio; el arreglo cambia modelos con `None`; faltan sitios). D-NP2: **en desacuerdo con (a) presentada como neutra**; recomiendo (a') como cambio de comportamiento anunciado, o (c) redefinida si se quiere conservar el modelo | Secciones 3.1 y 9.4 |
| P2 | **De acuerdo con cambios** (condición más estricta, convertir solo si hoy sale `object`, añadir multiserie `:3037`, decidir enteros grandes) | Sección 3.2 |
| P3 | **De acuerdo con cambios**: viable y con salidas y avisos idénticos, pero la ganancia real es 5 a 6 veces la llamada (Ridge, 50 folds), no 40 a 400; necesita las métricas en numpy para lucir; preparación solo del tramo de test; esfuerzo L solo para el alcance de D-NP4 | Sección 2; yo |
| P4 | **De acuerdo con cambios.** `get_indexer` difiere del `OrdinalEncoder` cuando `fit` vio un valor faltante (`categories_` acaba en NaN): el codificador da NaN y `get_indexer` el último código. Incluye el caso por defecto de pandas 3: columna de texto con `None` (dtype `str`): `[0, nan, 1, 0]` frente a `[0, 2, 1, 0]`. En pandas 2.3 (`object`) coinciden. También difiere con `bool` en `fit` e `int` en `predict`. Usar `.cat.codes` directamente es incorrecto en 5 casos; una búsqueda por categoría (`get_indexer(s.cat.categories)[codes]`) coincide en todos. Hay que quitar los NaN de `categories_` antes de indexar | `misc_p4_get_indexer.py` (subagente, 38 casos); el caso `str` con `None`, yo (`spot_checks.py`) |
| P5 | **De acuerdo** (verificado en la segunda ronda sobre el backtesting completo, no sobre las piezas de un `predict`): sin validar por fold -19 / -21 / -21 %, más el índice por posición -25 / -31 / -33 % (Ridge, exog float); índice diario con zona horaria -70 %. Salidas idénticas. En términos absolutos son 5 a 7 ms de una llamada de 18 a 23 ms (36 a 46 ms con zona horaria); con LightGBM, dentro del ruido salvo con zona horaria (-27 a -30 %) | `time_p5.py`, yo; sección 9.5 |
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
2. **El contenedor debe vivir por encima del backtesting.** Las búsquedas llaman a la función pública `backtesting_forecaster_multiseries` por candidato (`_search.py:1815`, `:2269`), que repite el preprocesado y repetiría la preparación de P3. Medido en la segunda ronda (9.2): en una búsqueda de 20 candidatos con Ridge, compartir la preparación lleva la ganancia de 1,8 a 2,2 o 2,4 veces. Es una mejora real pero menor de lo que yo suponía: lo que domina después es el `fit` de cada candidato.
3. **Preparación perezosa y acotada**: solo las filas de los tramos de test. Sobre la memoria, en la primera ronda avisé de un riesgo que la medida no confirma en tamaños normales: con 3000 series y 480 filas de test el prototipo usa **menos** memoria que el código actual (pico 3244 frente a 3391 MB con Ridge; 9.7), porque hoy `data_folds` mantiene vivos los objetos de todos los folds. El riesgo queda para tramos de test muy largos (la exog codificada ocupa filas de test × objetos × columnas × 8 bytes: 10 000 series × 8760 filas × 20 columnas son 14 GB, una cuenta, no una medida).

Alternativas que he considerado y descartado, con medida:

- **Predecir todos los folds en una sola pasada recursiva** (sin refit todos usan el mismo modelo: apilar los folds como filas y llamar al estimador `steps` veces en vez de `n_folds × steps`). [review_scripts/proto_bt_batched.py](review_scripts/proto_bt_batched.py), 600 series, exog mixta, p3 pa: con Ridge es **más lento** (0,41 → 0,69 ms por fold, por caché); con LightGBM, igual (15,6 → 17,9 ms con un hilo; 7,5 → 7,8 con cuatro). Predicciones idénticas. No compensa: el techo del informe es un techo.
- **Solo los pasos baratos de BTPLAN (F1 a F3 y M1) sin camino nuevo:** la extracción es el 40 a 45 % del tiempo y `check_predict_input` el 19 a 28 %; F1 a F3 atacan lo primero. Es menos ganancia (BTPLAN mide -20 % con LightGBM) pero sin segundo camino. De ellos, M1 y F2 valen también con P3; F1 y F3 dejan de notarse en multiserie cuando P3 exista.

### 6.3 ¿Está bien estimado el esfuerzo de P3?

**L es correcto solo para el alcance de D-NP4, y con una condición.** A favor: el prototipo cubre ese alcance (más `transformer_series`, diferenciación, window y calendar features, todas las codificaciones y los avisos) con salidas idénticas, así que el riesgo técnico es menor de lo que sugiere "Alto". En contra, lo que el prototipo no hace y la versión real sí necesita: detección de condiciones y vuelta al camino actual, forecaster ya ajustado y niveles desconocidos, diferenciación en diccionario, `n_jobs` explícito, lista blanca de transformadores, límite de memoria, y una batería de paridad entre los dos caminos que hay que mantener para siempre. Con intervalos es XL. El refit está prototipado en la segunda ronda y es viable, pero apenas gana (9.1), así que no justifica ampliar el alcance. La condición: si no se hace antes el núcleo compartido (6.2, punto 1), el coste de mantenimiento posterior es mayor que el de escribirlo.

Sobre D-NP4: `transformer_exog=None` es más restrictivo de lo necesario (un `StandardScaler` aplicado una vez da resultados idénticos), pero es una primera versión razonable.

### 6.4 El resto del veredicto

- "numpy hace desaparecer la diferencia entre pandas 2 y pandas 3": cierto en el bucle (1,2 / 1,6 / 1,3 ms por fold), no del todo en la llamada, porque la preparación y las métricas siguen en pandas (prototipo sin el `fit`: 1491 / 1545 / 1600 ms).
- "Las categóricas no justifican pandas aguas abajo": de acuerdo, con la salvedad de P1: con `categorical_features=None` el dtype es hoy lo único que hace categórica a una columna en multiserie, y eso es comportamiento, no solo un bug.
- "P1 se justifica por corrección, no por tiempo": de acuerdo.

## 7. Riesgos que el informe no recoge

1. **Expectativa de ganancia.** Quien lea el resumen espera uno a dos órdenes de magnitud. Con salidas idénticas se obtienen 5 a 6 veces (Ridge, 50 folds), 3 (LightGBM) y 1 a 1,5 (5 folds). Conviene fijar el criterio de aceptación de P3 en tiempo de la llamada completa, no por fold.
2. **Regresión por preparar de más** con pocos folds o historia larga (6.2, punto 3). La memoria no es un problema en los tamaños medidos (9.7).
3. **Dos caminos de predicción** que pueden divergir, y las búsquedas, que no se benefician si el contenedor vive dentro del backtesting (6.2).
4. **Cualquier arreglo de P1 con `None` cambia algo**: (a) y (a') cambian el modelo de quien hoy usa `category` de enteros `0..n-1` ordenados y de todo usuario de HistGradientBoosting con una `category`; (c) lo conserva solo en parte y es bastante más compleja (3.1 y 9.4). Tiene que ir con nota de versión y aviso.
5. **Métricas de búsqueda no reproducibles hoy:** one-step-ahead y backtesting difieren por el bug P1 (3,9 frente a 31,1). Cualquier arreglo cambia una de las dos; hay que decirlo en la nota de versión.
6. **Efectos del arreglo de P1 fuera de las predicciones:** se pierde `feature_names_in_` y los nombres de LightGBM, `fit_kwargs` con `categorical_feature` por nombre deja de funcionar, CatBoost necesita el cast a entero, y `ordinal_category` necesita su índice siempre.
7. **P2:** `complex128` aceptado, enteros grandes redondeados en `create_train_X_y`, copias nuevas, y forecasters con HistGradientBoosting guardados que siguen fallando hasta reajustar.
8. **Propuestas "bit a bit iguales" que no lo son:** P11 en pandas 2.3, P14 desde 8 offsets, P4 con NaN visto en `fit` (incluido el dtype `str` de pandas 3), P6 con caché, P16 con vista de solo lectura en pandas 3.
9. **Cobertura de tests inexistente** para exog `bool`, nullable y `pyarrow`: P2 no rompe ningún test porque ninguno los prueba.
10. **Condiciones de medida del informe:** `OMP_NUM_THREADS=1`, datos sintéticos con series iguales y completas, y medidas hechas con subagentes a la vez. Las cifras de §5.1 se reproducen, pero el reparto modelo frente a resto cambia con varios hilos.
11. **I8 con `bool`:** que una serie no tenga exog cambia que una columna `bool` se trate como categórica. Es un cambio de modelo que depende de los datos y merece fila propia.

## 8. Orden recomendado, por importancia e impacto

Orden final tras las dos rondas y la prueba con casos reales, incluida la búsqueda con `OneStepAheadFold` (la sección 9 explica los cambios que introdujo la segunda ronda y la 10, los de los casos reales). Va de lo que más aporta a lo que menos; dentro de cada grupo, también. Es el orden a fecha de esta revisión (2026-11-19) y no se actualiza: el que se mantiene vivo es el de `dev/ROADMAP.md`, sección 11, y si difieren vale el del roadmap.

Criterios: corrección antes que velocidad; las decisiones se abren pronto (D-NP2, D-NP3, D-NP4, D-BUG02, el significado de saltar un fold con refit y si las métricas en numpy pueden diferir en el último bit), porque cuestan días de calendario y no esfuerzo; un experimento barato que puede cambiar un plan caro va antes que ese plan; dos cambios sobre el mismo código van seguidos, primero el que fija el comportamiento; impacto es ganancia por cuántos usuarios la notan, y lo medido va antes que lo estimado; y un segundo camino de código es un coste permanente, así que entra con condiciones.

| # | Qué | Impacto | Esfuerzo | Condiciones, y qué cambia respecto al informe |
|---|---|---|---|---|
| | **Grupo 1: corregir lo que hoy da resultados malos o falla** | | | |
| 1 | **P2**, exog a float, seguida de **I8** (`bool` que pasa a categórica si una serie no tiene exog) | P2 arregla un fallo de HistGradientBoosting en `predict` con la configuración por defecto (categórica más `bool`) y da de 5 a 6 veces en el `fit` con una columna `bool`; I8 es un cambio silencioso de modelo que depende de los datos | S y S | Las dos tratan los dtypes de la exog al pasar a numpy y comparten la matriz de tests de dtypes (`bool`, nullable, `pyarrow`) que hoy no existe y que también necesita P1. P2 con condición estricta (dtypes explícitos; solo si hoy sale `object`), con el `predict` de multiserie (`:3037`), D-NP3 y los enteros grandes. I8 no estaba como propuesta en el informe |
| 2 | **P1**, numpy al estimador en el `fit` de multiserie | Predicciones erróneas silenciosas (MAE 3,9 frente a 31,1), solo con `categorical_features=None` o una lista incompleta | M | Decidir antes D-NP2: (a') con aviso y nota de versión, o (c) redefinida (9.4). Requisito: CAT-T2. Incluir `:4695`, `_utils.py:3038`, el cast de CatBoost y el índice de `ordinal_category`. Tests de paridad entre one-step-ahead y backtesting. El informe lo daba como S a M y sin cambio de modelo |
| 3 | Nivel predicho por un modelo que no lo vio (refit intermitente con `skip_folds`, 9.1) | Predicciones de una serie como nivel desconocido, en una configuración rara | S | Falta la reproducción mínima y decidir si es intencionado. No está en el informe |
| | **Grupo 2: calidad del código de multiserie** | | | |
| 4 | **Refactor del núcleo de `predict`** (antes, paso 1 de P3): validar, preparar y núcleo numpy, sin cambio de comportamiento | Calidad, no tiempo: `_create_predict_inputs` tiene unas 420 líneas, mezcla validación, alineación de la exog y matrices, y la usan cuatro métodos; `_recursive_predict` pide un DataFrame del que solo usa los valores; los atajos de MSPERF (comparaciones `identical`, cachés por `id`) viven dentro porque hoy se valida y alinea en cada llamada | M | Después de 1 y 2, para que el núcleo herede una conversión de la exog ya común y correcta. Esbozar antes las firmas de las tres etapas para comprobar que simplifica y no solo reparte. Red de seguridad: las baterías de identidad de MSPERF |
| | **Grupo 3: baratas e independientes, en cualquier hueco** | | | |
| 5 | **P6**, `_is_utc_anchored_index` | -70 % en el backtesting de una serie con índice diario y zona horaria: la única ganancia grande para los forecasters de una serie | S | Con la formulación exacta, no con caché (una caché cambia resultados) |
| 6 | BTPLAN F1, F2 y F3: ventana, series de entrenamiento y exog de test por posición | Mejoran el camino actual sin un segundo camino: F2, con refit en cada fold, 3,2 a 3,5 s de 20 a 25 s (600 series, Ridge); las tres, −0,5 s de 4,2 s sin refit (500 series, LightGBM, 10 folds, BTSPEED) | M | Con P3 aplazado (sección 10), F1 y F3 recuperan su valor. Si P3 se hace más tarde, quedan absorbidas |
| 7 | **P7, P8, P9**, `fit` de multiserie | Unos milisegundos por `fit` con cientos de series | S | P8 solo acierta si todas las series tienen el índice de la exog; P9 comparando por columna |
| 8 | **P10, P12** y la reutilización de entradas de **P11** | Décimas de milisegundo por llamada | S | P10 con claves `int` y orden estable; P12 solo la búsqueda en array; de P11, no el cambio a `np.quantile` |
| 9 | **P16 y P17**, ventana del DMV | El informe mide un 40 % del `predict` del DMV (sin verificar) | S | Medir antes. `copy=True` y selección de columnas antes de convertir. P17 solo afecta al DMV |
| 10 | **Métricas en numpy, primera parte** (PERF-03: métricas por nombre en numpy) | Búsquedas one-step-ahead 2,6 veces más rápidas con Ridge y 1000 series (sintéticos); con LightGBM y datos reales, las métricas son parte de un 3 % de la búsqueda (10.6). Es la pieza que usaría P3 | S | La fórmula directa no siempre es idéntica bit a bit (4e-16 en un caso): decidir si se acepta |
| | **Grupo 4: rendimiento solo con muchos folds o modelos baratos, cuando haya demanda (sección 10)** | | | |
| 11 | **P3, camino rápido** sobre ese núcleo, con **BTPLAN M1** dentro | Solo con evaluaciones de muchos folds y modelos de coste bajo o medio: en los flujos de los artículos (9 a 31 folds) de x1,0 a x1,8 en la llamada completa; con 61 folds (evaluación diaria) de x2,8 a x3,3. Empeora la calidad: un segundo camino que debe reproducir los avisos de `predict` (unas 500 líneas en `proto_bt_blocks.py`, sin contar las funciones de avisos que toma de `proto_bt_once.py`) | L | Cuando haya ese perfil de uso o demanda. Condiciones: un solo núcleo, vuelta al camino actual en lo que D-NP4 deja fuera, y la batería de `proto_bt_once_cases.py` y los casos reales convertidos en tests. Si D-BUG02 está decidida, BUG-02 antes de M1 |
| 12 | Reutilizar las matrices de entrenamiento entre los candidatos de una búsqueda | Estimado: otras 2 veces sobre lo anterior en búsquedas con modelos baratos | Por estimar | Con `OneStepAheadFold` ya existe (la búsqueda cachea la división por `lags`, 10.6). Queda para búsquedas con backtesting y modelos baratos. Prototipo antes de decidir. No está en el informe |
| 13 | **PERF-ms-osa-metrics**: validación one-step-ahead multiserie sin `groupby` por nivel | Agrupar por nivel, deshacer la transformación y métricas: 0,9 s por trial con 1578 series, el 3 % de la búsqueda de py59 con LightGBM (10.6) | S a M | Conservar la inversa por nivel, los niveles ausentes y las métricas propias |
| | **Grupo 5: poco valor, no ahora** | | | |
| 14 | **P4**, categóricas por códigos | Pierde su ganancia en multiserie cuando exista P3; le quedan el `predict` de una serie con categóricas y el `fit` | M | Antes hay que definir el tratamiento de NaN en `categories_` |
| 15 | **P5**, validar una vez en los forecasters de una serie | 5 a 7 ms por backtesting; con datos reales (py39), de 3 a 9 % de la llamada | M | Lo que pesa de P5 es P6 |
| 16 | **P13, P14, P15**, el cambio a `np.quantile` de P11 y unificar a `.equals` (I5) | Pequeño o sin verificar | | P14 tras decidir D-NP6 y con lista blanca; I5 es un cambio de comportamiento en pandas 2.3 |

## 9. Segunda ronda: lo que quedaba por revisar

Siete puntos que la primera ronda dejaba sin medir o apoyados solo en un subagente. Mismo árbol y mismos entornos; tiempos medidos a solas con `OMP_NUM_THREADS=1`. Prototipo nuevo: [review_scripts/proto_bt_blocks.py](review_scripts/proto_bt_blocks.py), que añade a `proto_bt_once.py` el refit (preparación estática una vez y preparación del modelo tras cada ajuste), una preparación que se puede compartir entre candidatos y métricas en numpy para MAE y MSE. Pasa la misma batería de 25 casos: **24 idénticos y uno con el mismo error, con `refit=False`, `refit=True` y `refit=3`, en los tres entornos** (`REFIT=1 PROTO=blocks bash run3.sh proto_bt_once_cases.py`).

### 9.1 Backtesting con refit

600 series, exog mixta, pasos de 24 ([results/time_bt_blocks_ridge.txt](review_scripts/results/time_bt_blocks_ridge.txt), [results/time_bt_blocks_other.txt](review_scripts/results/time_bt_blocks_other.txt)). Salidas idénticas en todos.

| Escenario | Hoy | Prototipo | Aceleración | Dónde está el tiempo hoy |
|---|---|---|---|---|
| Ridge, refit en cada fold, 20 folds (p2 / p3 / p3 pa) | 23,6 / 25,6 / 29,8 s | 20,4 / 21,0 / 25,5 s | **1,16 / 1,22 / 1,17** | `fit` 15,6 / 17,7 / 21,8 s; extraer folds 4,7 / 5,3 / 6,0 s; `predict` 1,6 / 1,9 / 2,2 s |
| LightGBM, refit en cada fold, 10 folds (p3 pa) | 68,7 s | 65,8 s | **1,05** | `fit` 62,1 s |
| Ridge, refit cada 5 folds, 50 folds (p3 pa) | 21,4 s | 10,6 s | **2,0** | `fit` 8,4 s; extraer folds 7,0 s; `predict` 4,8 s |

Con refit en cada fold, P3 quita casi todo lo que no es entrenar (el `predict` y la extracción de la ventana y de la exog de test) y aun así la llamada apenas cambia, porque lo que queda es el `fit` y la extracción del conjunto de entrenamiento (3,2 a 3,5 s en el prototipo, que conserva el código actual por serie; es lo que ataca BTPLAN F2).

De qué está hecho ese `fit` ([results/time_fit_parts.txt](review_scripts/results/time_fit_parts.txt), 600 series, 1260 filas, exog mixta):

| Estimador | `fit` | `_create_train_X_y` | Estimador |
|---|---|---|---|
| Ridge (p2 / p3 / p3 pa) | 737 / 792 / 894 ms | 494 / 493 / 660 ms (67 / 62 / 74 %) | 206 / 268 / 205 ms |
| LightGBM, 50 árboles (p3 pa) | 5515 ms | 729 ms (13 %) | 4756 ms (86 %) |

**Veredicto.** El informe acierta al decir que con Ridge lo que no es el estimador es la mayor parte de cada fold con refit, y al dejar el refit fuera de la primera versión de P3. Lo que no dice, y es la conclusión útil, es que **P3 no es la propuesta que ayuda al usuario que reajusta**: para ese caso las palancas son la creación de las matrices de entrenamiento (`_create_train_X_y`) y la extracción del entrenamiento (BTPLAN F2), con modelos baratos, y ninguna con modelos caros. Conviene decirlo junto a la fila de P3 para no vender una mejora general del backtesting.

**Rareza del código actual, encontrada al hacer coincidir el prototipo.** Con refit intermitente y `skip_folds`, `_extract_data_folds_multiseries` vuelve a extraer el conjunto de entrenamiento cuando cambian sus límites (`_utils.py:1647-1649`), aunque ese fold no se ajuste porque el fold del refit se saltó. Los niveles de la última ventana se filtran entonces con un entrenamiento que el modelo en uso no ha visto. En el caso `refit=3, skip_folds=2` de la batería, la serie `s6` (demasiado corta en el primer entrenamiento) se predice con el modelo del primer fold como nivel desconocido, con un `UnknownLevelWarning`. Las predicciones del prototipo coinciden; ese aviso no lo emite. No está en el informe ni en el roadmap.

### 9.2 Búsqueda de hiperparámetros

`grid_search_forecaster_multiseries`, 600 series, exog mixta, sin refit, 10 folds de 24 pasos ([results/time_search.txt](review_scripts/results/time_search.txt)). El prototipo se llama una vez por candidato; las métricas coinciden con las de la búsqueda (diferencia máxima 0).

| | Ridge, 20 candidatos (p2 / p3 / p3 pa) | LightGBM, 10 candidatos (p3 pa) |
|---|---|---|
| Hoy | 65,6 / 73,3 / 76,6 s | 81,9 s |
| Prototipo por candidato, sin compartir nada | 36,8 / 39,8 / 43,1 s (1,8 veces) | 65,2 s (1,3) |
| Con la preparación compartida entre candidatos | 29,4 / 30,0 / 34,5 s (2,2 / 2,4 / 2,2) | 61,1 s (1,3) |
| Además con las métricas en numpy | 21,5 / 21,6 / 24,0 s (3,0 / 3,4 / 3,2) | 54,6 s (1,5) |
| De eso, los `fit` | 16,5 / 15,8 / 17,8 s | 50,2 s |

**Veredicto sobre mi propia afirmación de 6.2.** Compartir la preparación entre candidatos se confirma, pero vale menos de lo que escribí: de 1,8 a 2,2 o 2,4 veces, no "más que todos los bucles de folds juntos". Lo que la medida destapa es otra cosa: **tras P3 y las métricas, el 75 % de la búsqueda con Ridge son los `fit`, y dos tercios de cada `fit` son `_create_train_X_y`, que da las mismas matrices para todos los candidatos que comparten `lags`** (una rejilla sobre los parámetros del estimador). Son unos 10 a 13 s de los 21 a 24 que quedan. Reutilizar las matrices de entrenamiento entre candidatos dejaría la búsqueda en torno a 10 s, unas 7 veces menos que hoy. Es una estimación a partir de las piezas, no un prototipo, y no está en el informe ni en el roadmap; merece una fila.

### 9.3 Métricas en numpy antes de P3

600 series, exog mixta, 50 folds, sin refit; llamada completa (p2 / p3 / p3 pa):

| | Ridge | LightGBM |
|---|---|---|
| Hoy | 9,6 / 11,5 / 12,2 s | 12,1 / 13,7 / 14,4 s |
| Prototipo, métricas de hoy | 1,77 / 1,74 / 2,00 s (5,4 / 6,6 / 6,1 veces) | 3,67 / 4,03 / 4,18 s (3,3 / 3,4 / 3,4) |
| Prototipo, métricas en numpy | **1,25 / 1,29 / 1,31 s (7,7 / 8,9 / 9,3)** | 3,83 / 3,44 / 3,60 s (3,2 / 4,0 / 4,0) |

Las métricas pasan de 476 a 682 ms a 36 a 40 ms. **Se confirma la recomendación de adelantarlas**, con dos precisiones. La cifra de BTPLAN ("de unos 600 ms a unos 10") es del orden correcto. Y la fórmula directa no es siempre idéntica bit a bit: en estos escenarios las métricas salieron idénticas, pero con refit cada 5 folds la agregada difirió en 4,4e-16, como avisa BTPLAN. Aun con todo, el "~12 s → ~0,4 s" de la fila de P3 no se alcanza: quedan 1,3 s, de los que 0,4 son el `fit` y el resto preparación (alinear la exog 0,2 a 0,3 s, preparar el modelo 0,14 a 0,2 s, extraer el entrenamiento 0,12 a 0,2 s, preprocesado 0,08 s).

### 9.4 D-NP2: la opción (c), puesta a prueba

Scripts `review_scripts/p1c_*.py` (subagente con el encargo de romperla; he reejecutado `p1c_01_estimators.py`). Salida igual en los tres entornos salvo donde se dice.

**"Pasar el índice" no basta.** Qué tiene que contener la matriz para que el modelo ajustado sea el de hoy:

| Estimador | Valores más índice | Códigos de pandas más índice |
|---|---|---|
| LightGBM | Idéntico con enteros no negativos. **Cambia** con negativos (los trata como faltantes) y con valores desde 2^31. Un valor grande dispara la memoria del modelo (12,6 MB de texto con el valor 5 000 000 frente a 0,11 MB con 7) | Idéntico en todas las disposiciones |
| XGBoost | **Error en `fit`** con negativos y con 3e9; cambia con NaN. Alrededor de 1 GB de memoria por cada millón del valor máximo (un entorno) | Idéntico, dentro y fuera de la muestra. Con los códigos ordenados del `OrdinalEncoder` de `'auto'` **no** lo es fuera de la muestra |
| HistGradientBoosting | Idéntico siempre (recodifica por dentro) | Idéntico, pero entonces `estimator.predict(create_predict_X(...))`, que hoy funciona, deja de coincidir |
| CatBoost | Hoy no ajusta con `None`; cualquier entero sirve | Referencia |

Por tanto (c), para ser implementable, es: codificar cada columna `category` (y la de nivel) a los códigos de pandas de su dtype de entrenamiento, **en `fit` y en `predict`**, con desconocidos y NaN a NaN, y pasar los índices; solo para LightGBM, XGBoost y CatBoost sin envolver; valores para HistGradientBoosting y para el resto. Las categorías de entrenamiento ya están en `exog_dtypes_out_`.

**Y ni así conserva siempre el modelo:**

- LightGBM con una `category` **ordenada**: hoy la trata como numérica sobre los códigos; con el índice cambia el modelo.
- Una subclase de `LGBMRegressor` o un `TransformedTargetRegressor`: hoy categórica por dtype, pero el índice no se puede pasar (`utils.py:651-652` comprueba el módulo).
- La configuración del usuario. Hoy, con `None`, skforecast no toca nada (`_forecaster_recursive_multiseries.py:2463-2464`) y la documentación lo da como la vía prevista ("When `categorical_features` is not `None` do not set categorical features directly on the estimator", `tools/ai/llms-base.txt:179`). Si (c) sobrescribe como hace `'auto'`, cambia el modelo de quien configuró el estimador (por ejemplo HistGradientBoosting con `categorical_features=None`, sin aviso). (c) tendría que traducir esa configuración, no sustituirla.
- El bug alcanza también a **CatBoost** por esta vía: `fit_kwargs={'cat_features': [idx]}` ajusta con códigos y predice con valores (MAE 29,6 frente a 4,0).

**"(a) es una degradación silenciosa" era demasiado fuerte.** Con 30 series, dos categóricas (40 y 3 categorías), 300 filas, 5 semillas y efecto de la categoría nada, medio o del todo ordinal, MAE de (c) frente a (a'):

| Estimador | (c) | (a') |
|---|---|---|
| LightGBM | 3,55 / 3,21 / 3,38 | 3,98 / 3,44 / 3,19 |
| HistGradientBoosting | 3,45 / 3,22 / 3,38 | 4,02 / 3,41 / 3,19 |
| XGBoost | 4,36 / 3,91 / 4,26 | 4,21 / 3,54 / 3,52 |
| CatBoost | 3,50 / 3,23 / 3,47 | 2,94 / 2,63 / 2,23 |

El tratamiento categórico gana en LightGBM y HistGradientBoosting salvo con efecto ordinal; el numérico gana en XGBoost y CatBoost. Son datos sintéticos con hiperparámetros por defecto: sirven para desmentir una regla general, no para fijar otra.

**La incoherencia de fondo.** `None` significa hoy dos cosas: en `ForecasterRecursive`, `ForecasterDirect` y `ForecasterDirectMultiVariate` el estimador recibe los valores y no hace ningún corte categórico; en multiserie recibe el DataFrame y los hace. El docstring de multiserie dice "If `None`: no categorical encoding is applied internally" (`:139`). (c) mantiene a multiserie como excepción; (a') unifica.

**Recomendación corregida.** **(a')**: valores como numérica, que es lo que ya recibe `predict` y lo que hacen los demás forecasters, respetando la configuración que el usuario ponga por índice en el estimador o en `fit_kwargs`; publicada como cambio de comportamiento, con nota de versión y un aviso cuando una columna `category` se encuentra con `categorical_features=None` ("se trata como numérica; usa `'auto'` o configura el estimador"). Motivos: es simple, coherente con el resto de forecasters y con el docstring, y el valor por defecto es `'auto'`, así que afecta a pocos usuarios. Su coste es el que ya señalé: cambia el modelo de quien hoy acierta (HistGradientBoosting, y LightGBM o XGBoost con categorías `0..n-1` ordenadas). (c) redefinida es la alternativa si se prioriza no cambiar esos modelos, a cambio de una tabla de reglas por estimador y de los tres casos de arriba en los que tampoco lo consigue. Retiro la recomendación de (c) tal como la formulé en la primera ronda. Lo que no cambia: (a) con códigos presentada como "sin cambio para nadie" es incorrecta.

Hallazgo aparte del subagente (código existente, `'auto'`; ya está en el roadmap como CAT-T4): si no se detecta ninguna columna categórica, `feature_types` y `categorical_features` del usuario se restablecen sin aviso (`utils.py:654-662`). No verificado por mí.

### 9.5 P5 sobre el backtesting univariante completo

El informe estimó P5 a partir de las piezas de una llamada a `predict`. [review_scripts/time_p5.py](review_scripts/time_p5.py) lo mide sobre `backtesting_forecaster` entero, 50 folds de 24 pasos, sin refit ([results/time_p5.txt](review_scripts/results/time_p5.txt)). A = sin validar por fold; B = A más el índice de las predicciones por posición. Salidas idénticas en todos los casos.

| Caso (p2 / p3 / p3 pa) | Hoy | A | B |
|---|---|---|---|
| Ridge, horario, exog float | 17,9 / 19,4 / 22,7 ms | -19 / -21 / -21 % | -25 / -31 / -33 % |
| Ridge, horario, sin exog | 11,9 / 11,3 / 12,5 ms | -6 / -4 / -5 % | -16 / -15 / -10 % |
| LightGBM, horario, exog float | 125 / 129 / 147 ms | -4 / +3 / -12 % | -2 / +6 / -12 % |
| Ridge, diario con zona horaria | 51,7 / 63,8 / 65,3 ms | -5 / -18 / -18 % | **-70 / -70 / -70 %** |
| LightGBM, diario con zona horaria | 171 / 178 / 192 ms | +3 / -8 / -8 % | -30 / -30 / -27 % |

**De acuerdo: los tres porcentajes del informe se reproducen** (-17, -31 y -68 % frente a mis -19 a -21, -25 a -33 y -70). Matiz de escala: son 5 a 7 ms por backtesting en el caso horario; el caso que de verdad cambia algo es el índice diario con zona horaria (36 a 46 ms, por `_is_utc_anchored_index`), que es P6. D-NP1 se mantiene.

### 9.6 Comprobaciones de los subagentes, repetidas por mí

He vuelto a ejecutar en los tres entornos los scripts de P6, P8, P9, P10, P12, P14, P16, I3, I5 e I8 y he leído sus salidas: todas dan lo que recoge la tabla de las secciones 4 y 5. Dos detalles que afinan lo escrito:

- **P16:** en pandas 3 la vista de solo lectura aparece también seleccionando las columnas antes de convertir; en pandas 2.3, convertir toda la tabla y seleccionar después da una vista **escribible** del `last_window` del usuario. En los dos casos hace falta la copia explícita.
- **I8:** confirmado el caso `bool` con `'auto'`: con una serie sin exog, `exog_dtypes_in_` es `object` y `exog_dtypes_out_` `float64` (la columna se ha codificado como categórica); con todas las series presentes, `bool` en los dos.

### 9.7 Escala y memoria

3000 series, exog mixta, 20 folds de 24 pasos, sin refit, p3 pa ([results/time_scale_3000.txt](review_scripts/results/time_scale_3000.txt)). Memoria = pico del proceso (cada variante en un proceso aparte).

| | Hoy | Prototipo | Prototipo, métricas en numpy | Pico de memoria hoy / prototipo |
|---|---|---|---|---|
| Ridge | 31,0 s | 10,8 s (2,9 veces) | 8,6 s (3,6) | 3391 / 3244 MB |
| LightGBM | 47,9 s | 29,2 s (1,6) | 26,1 s (1,8) | 2760 / 2586 MB |

- El coste actual por fold escala de forma lineal con las series (unos 236 ms por fold con 600 y unos 1285 con 3000).
- La ganancia relativa baja respecto a 600 series y 50 folds porque hay menos folds y el `fit` pesa más (4,6 a 5,3 s con Ridge, 20,8 s con LightGBM).
- **Corrijo el riesgo de memoria de la primera ronda:** en este tamaño el prototipo usa algo menos de memoria que el código actual. El aviso de los 14 GB sigue siendo una cuenta para tramos de test muy largos, no algo observado.

### 9.8 Qué cambia en el orden recomendado

La tabla de la sección 8 ya recoge estos cambios; aquí queda el porqué de cada uno.

- **P3:** mantenerla, pero con el alcance dicho en claro: mejora el backtesting y las búsquedas **sin refit**. Criterio de aceptación en tiempo de la llamada completa.
- **Métricas en numpy (BTPLAN M1 y PERF-03):** confirmado que van antes o con P3. Precisado después: la primera parte de PERF-03 va antes, porque vale por sí sola en las búsquedas one-step-ahead; M1 va dentro del paso 2 de P3, porque sola es un 5 % de la llamada de hoy (0,6 s de 10 a 12) y su peso aparece con P3.
- **Nueva, para las búsquedas:** reutilizar las matrices de entrenamiento entre los candidatos que comparten `lags` y datos. Es la mayor palanca que queda en una búsqueda con modelos baratos (estimación: de 66 a 77 s a unos 10 s junto con P3 y las métricas).
- **Con refit:** BTPLAN F2 (extracción del entrenamiento por posición) y `_create_train_X_y`; P3 no aporta.
- **P1:** D-NP2 con (a') anunciada, o (c) redefinida; en los dos casos con nota de versión.
- **P5 y P6:** verificadas; P6 es la parte que más cambia y hay que hacerla con la formulación exacta, no con caché.
- **Para el roadmap:** la rareza de 9.1 (nivel predicho por un modelo que no lo vio, con refit intermitente y `skip_folds`).

## 10. Prueba con casos reales

Hecha el 2026-11-19 para cumplir la condición que el orden de la sección 8 ponía al camino rápido: probar el prototipo con casos reales antes de invertir una L. Los casos son los backtestings de tres artículos de cienciadedatos.net, reconstruidos con sus datos, su preprocesado, sus forecasters y sus folds ([review_scripts/real_data.py](review_scripts/real_data.py)):

- **py44** (multiserie): `store` (50 artículos de una tienda, tabla ancha, HistGradientBoosting, sin exog, 31 folds de 7 pasos) y `demo` (5 series de distinta longitud con exog en diccionario y columnas quitadas, 22 folds).
- **py59** (modelos escalables): `bdg2_daily`, 1578 edificios, 22 exógenas (dos categóricas de texto y una `category`), 9 folds de 7 pasos. Con el LightGBM por defecto y lags 14 (el forecaster antes de la búsqueda) y con el del artículo tras la búsqueda (600 árboles, lags 1 a 62).
- **py39** (una serie): `bike_sharing` horario con exógenas y LightGBM, unos 80 folds de 36 pasos. Usa `bike_sharing_extended_features`, la versión ya calculada de las variables del artículo, que no es idéntica.

Y tres variantes que no están en los artículos, sobre los mismos edificios: `steps=1` (evaluación diaria, 61 folds) con los dos modelos, y la búsqueda con `TimeSeriesFold` (9 folds de validación, 4 candidatos de LightGBM) en lugar de `OneStepAheadFold`. Scripts: [time_real_cases.py](review_scripts/time_real_cases.py), [time_real_search.py](review_scripts/time_real_search.py) y [check_real_mae.py](review_scripts/check_real_mae.py); salidas en `review_scripts/results/real_*.txt`. Mismo árbol (`ab62227`), mismos entornos y `OMP_NUM_THREADS=1`.

### 10.1 Fidelidad a los artículos

Con el código actual y pandas 2.3.3, como en los artículos ([results/real_check_mae.txt](review_scripts/results/real_check_mae.txt)):

| Caso | Valor del artículo | Aquí |
|---|---|---|
| `store`, item_1 / item_2 / item_3 | 5,5126 / 9,2185 / 7,3106 | iguales (diferencia de 1e-7 o menos) |
| `demo`, id_1000 / id_1001 / id_1003 / id_1004 | 150,89 / 863,16 / 265,13 / 921,15 | iguales (diferencia de 1e-7 o menos) |
| edificios, media | 307 | 307,25 |
| edificios, por edificio | 1225,57 / 238,25 / 32,17 | de +0,25 a +8,2 % (efecto de los hilos de LightGBM, 10.2) |
| `bike` | 49,44 | 50,97 (+3,1 %: otras variables) |

### 10.2 Salidas idénticas, y los hilos de LightGBM

Con LightGBM en un hilo (`LGBM_N_JOBS=1`), **el prototipo devuelve las mismas métricas, las mismas predicciones y los mismos avisos que el código actual en todos los casos y variantes, en los tres entornos**.

Las primeras tandas, con el `n_jobs=None` de los artículos, daban diferencias. Su causa no es el prototipo: en LightGBM 4.7.0, `n_jobs=None` entrena con todos los núcleos físicos e ignora `OMP_NUM_THREADS`, y el orden de las sumas cambia entre ejecuciones. Dos ejecuciones del código actual, en el mismo proceso y con los mismos datos en memoria, ya difieren (13 de 96 258 predicciones, en 2e-13, con el modelo por defecto); con 600 árboles y submuestreo la diferencia crece hasta 0,2 a 0,4 en el MAE de algunos edificios, y hasta un 8 % frente al artículo. Afecta a cualquier usuario que repita un notebook con LightGBM (fila DOC-lightgbm-reproducibility del roadmap).

### 10.3 Tiempos

Llamada completa, prototipo con métricas en numpy frente a hoy; rango sobre p2 / p3 / p3 pa. "Resto" es lo que no es modelo (ni `fit` del forecaster ni la función de predicción del estimador). LightGBM en un hilo salvo donde se indica.

| Caso | Series | Folds | Hoy | Resto hoy | Resto con el prototipo | Llamada completa |
|---|---|---|---|---|---|---|
| `demo` | 5 | 22 | 0,22 a 0,33 s | 0,07 a 0,12 s | 0,02 s | x1,4 a 1,8 |
| `store` | 50 | 31 | 1,15 a 1,21 s | 0,19 a 0,21 s | 0,03 s | x1,15 a 1,28 |
| edificios, por defecto, varios hilos | 1578 | 9 | 11,8 a 14,2 s | 5,7 a 7,3 s | 1,7 a 2,1 s | x1,53 a 1,63 |
| edificios, por defecto | 1578 | 9 | 17,4 a 20,5 s | 5,3 a 7,5 s | 1,7 a 2,0 s | x1,23 a 1,29 |
| edificios, artículo (600 árboles) | 1578 | 9 | 172 a 179 s | 5,7 a 7,2 s | 1,4 a 2,1 s | x0,99 a 1,03 |
| **edificios, por defecto, `steps=1`** | **1578** | **61** | **40,4 a 50,5 s** | **27,3 a 35,4 s** | **1,5 a 2,6 s** | **x2,83 a 3,25** |
| edificios, artículo, `steps=1` | 300 | 61 | 52,0 a 54,2 s | 5,1 a 6,3 s | 0,27 a 0,45 s | x1,10 a 1,15 |
| búsqueda `TimeSeriesFold`, 4 candidatos | 300 | 9 | 18,8 a 20,2 s | 4,7 a 5,6 s | 0,9 a 1,1 s (preparación compartida) | x1,30 a 1,41 |
| `bike`, una serie, P5 | 1 | 80 | 0,87 a 1,03 s | 0,27 a 0,31 s | 0,23 a 0,26 s | x1,03 a 1,09 |

Las dos variantes caras usan una muestra fija de 300 edificios (`REAL_N_SERIES=300`). La muestra es prudente: con el modelo por defecto y `steps=1` da x2,35 a 3,00 frente a x2,83 a 3,25 con los 1578, y con 9 folds, x1,26 a 1,43 frente a x1,23 a 1,29.

### 10.4 Lectura

1. **El prototipo hace lo que promete en todos los casos**: el resto baja de 3 a 20 veces y, después, un backtesting cuesta casi lo que cuesta el modelo.
2. **Lo que decide la llamada completa es cuánto pesa hoy ese resto frente al `fit`**, y eso depende de folds por series frente al coste del modelo. En los flujos de los artículos (9 a 31 folds) el ahorro va de décimas de segundo a unos 5 s, y la llamada gana de x1,0 a x1,8. Con una evaluación de muchos folds (61, diaria) ahorra unos 30 s y gana x3. Con un modelo caro no gana nunca.
3. **La condición de la sección 8 no se cumple en los flujos de los artículos y sí en las evaluaciones de muchos folds.** Los datos sintéticos (600 series, 50 folds, Ridge o LightGBM de 50 árboles) eran el caso favorable.
4. **La parte cara de py44 y py59 es la búsqueda, y la hacen con `OneStepAheadFold`, que el camino rápido no toca.** Con una búsqueda por backtesting la ganancia es moderada (x1,3 a 1,4 con 9 folds de validación).
5. **P5 se confirma como poco valor** con datos reales (3 a 9 %).
6. **Medido después (10.6):** el reparto de la búsqueda con `OneStepAheadFold` de py59.

### 10.5 Qué cambia

- **El camino rápido deja de ser la pieza central del rendimiento**: es una L para evaluaciones de muchos folds con modelos baratos o medios, que se hace cuando haya ese perfil de uso o demanda. Además empeora la calidad del código (un segundo camino que reproduce los avisos de `predict`), así que solo lo justifica la velocidad.
- **Su paso 1 se separa y se justifica por calidad**: partir `predict` en validar, preparar y núcleo numpy ordena una función de unas 420 líneas que usan cuatro métodos, estrecha la entrada de `_recursive_predict` y da un sitio a los atajos que hoy se acumulan dentro. Va después de los arreglos de la exog, que son la mayor ganancia de calidad: una sola conversión a numpy en `fit` y en `predict` elimina la clase de errores de P1, P2 e I8.
- **La búsqueda con `OneStepAheadFold` de py59, medida en 10.6**, deja PERF-03 (primera parte), PERF-ms-osa-metrics y PERF-search-shared-train-matrices como mejoras solo para modelos baratos.
- **Hallazgo para la documentación**: la reproducibilidad con LightGBM (10.2).

### 10.6 La búsqueda con `OneStepAheadFold` (STUDY-osa-search-cost)

La búsqueda de py59 tal como la hace el artículo: `bayesian_search_forecaster_multiseries` con `OneStepAheadFold(initial_train_size='2017-08-31')`, su espacio de búsqueda (lags 31 o 62, de 100 a 800 árboles), 10 trials, `random_state=123` y `return_best=True` ([time_real_osa_search.py](review_scripts/time_real_osa_search.py), [results/real_osa_search.txt](review_scripts/results/real_osa_search.txt) y `real_osa_search_1thread_n300.txt`).

| | 1578 edificios, configuración del artículo (LightGBM con sus hilos), p3 pa | 300 edificios, LightGBM en un hilo, p2 / p3 / p3 pa |
|---|---|---|
| Total | 280 s | 195 a 198 s |
| **Modelo**: `fit` y `predict` de LightGBM en los trials y en el ajuste final (incluida la predicción in-sample para los residuos) | **255 s (91 %)** | **191 a 194 s (98 %)** |
| Matrices de skforecast (`_train_test_split_one_step_ahead_search`, 2 llamadas: una por valor de `lags`) | 11,6 s (4,2 %) | 1,7 a 2,2 s (1 %) |
| Por trial: agrupar por nivel, deshacer la transformación y métricas | 8,6 s (3,1 %; 0,9 s por trial) | 1,5 a 1,7 s (0,8 %) |
| Ajuste final: resto del `fit` (sobre todo `_create_train_X_y`) | 3,7 s (1,3 %) | 0,5 a 0,7 s (0,3 %) |
| Optuna y tabla de resultados | 0,7 s | 0,1 s |

1. **En el flujo que recomienda la documentación, el 91 al 98 % del tiempo es LightGBM.** Aunque lo que no es modelo costara cero, la búsqueda real ganaría como mucho x1,10 (de 280 a 255 s). Ninguna tarea de pandas o numpy cambia mucho la experiencia de estos usuarios.
2. **Compartir las matrices entre candidatos ya existe con `OneStepAheadFold`**: la búsqueda cachea la división por `lags` (2 llamadas para 10 trials). PERF-search-shared-train-matrices solo tiene sentido en búsquedas con backtesting y con modelos baratos; con LightGBM, el resto del `fit` es 3,7 s frente a 52 s del estimador (con Ridge, en los datos sintéticos, `_create_train_X_y` era el 62 a 74 % del `fit`).
3. **PERF-ms-osa-metrics y la primera parte de PERF-03** atacan la fila de agrupar y métricas: como mucho un 3 % aquí. Su "2,6 veces" se midió con Ridge y datos sintéticos.
4. **El ajuste final predice sobre todo el entrenamiento para los residuos in-sample**: de 7 a 9 s (2,5 a 4,6 % de la búsqueda). Es lo que estudia STUDY-fit-insample-binning (#1520).
5. **Conclusión sobre "pandas frente a numpy"**: lo que más aporta es, por este orden, la corrección (P1, P2 e I8, con una sola conversión de la exog a numpy), la calidad del código (MAINT-ms-predict-core) y, solo con muchos folds o modelos baratos, el rendimiento. En el uso típico (LightGBM, pocos folds, búsqueda con `OneStepAheadFold`) la mejora de tiempo sería menor del 10 %.

## Anexo: qué no se ha verificado

- `ForecasterRnn` y los adaptadores foundation (sin torch ni keras): solo lectura de código.
- SHAP sobre el estimador tras el arreglo de P1 (no instalado); versiones de XGBoost en las que `enable_categorical` es `False` por defecto.
- Las ganancias de tiempo de P6, P8, P9, P10, P12, P13, P15 y P16 (solo corrección o lectura).
- P10 en multiserie y `ForecasterEquivalentDate`; P11 con filas parcialmente NaN.
- Los prototipos de P3 no cubren intervalos, `return_predictors`, `n_jobs > 1`, forecaster ya ajustado ni diferenciación en diccionario; con refit intermitente y `skip_folds` no emiten el `UnknownLevelWarning` del código actual (9.1). No se han probado con transformadores que no sean elemento a elemento ni con más de 3000 series.
- Casos reales (sección 10): sin refit, con un solo `fit` por backtesting, y solo con modelos de árboles (HistGradientBoosting y LightGBM); con un modelo barato (Ridge, lineales) lo que no es modelo pesaría más, y no se ha medido con datos reales.
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

# Segunda ronda (sección 9)
REFIT=1 PROTO=blocks bash run3.sh proto_bt_once_cases.py   # y REFIT=3, REFIT=0
bash run3.sh time_bt_blocks.py 600 mixed ridge 20 24 1     # N EXOG EST FOLDS STEPS REFIT [FILAS]
bash run3.sh time_bt_blocks.py 600 mixed ridge 50 24 0     # métricas de hoy y en numpy
bash run3.sh time_search.py 600 mixed ridge 20 10          # búsqueda, unos 5 minutos por entorno
bash run3.sh time_fit_parts.py 600 mixed ridge
bash run3.sh time_p5.py
python time_bt_blocks.py 3000 mixed ridge 20 24 0          # escala y memoria
bash run3.sh p1c_01_estimators.py                          # y el resto de p1c_*.py
bash run3.sh p1_osa.py; bash run3.sh p1_dnp2.py; bash run3.sh p1_fix_grid.py lgbm xgb hgb catboost ridge
bash run3.sh p2_01_dtype_matrix.py                   # y el resto de p2_*.py
bash run3.sh misc_p11_p12_intervals.py               # y el resto de misc_*.py

# Casos reales (sección 10)
uv pip install --python ~/.venvs/skforecast-p2/bin/python --no-deps feature_engine   # y en el entorno de pandas 3
python real_data.py download DIR                     # una vez, con pyarrow
export REAL_DATA_DIR=DIR
bash run3.sh time_real_cases.py demo 5               # y store 5, bike 5, bdg2_small 2, bdg2 1
LGBM_N_JOBS=1 bash run3.sh time_real_cases.py bdg2_small 2              # salidas exactas: un hilo
LGBM_N_JOBS=1 bash run3.sh time_real_cases.py bdg2_small_steps1 2
REAL_N_SERIES=300 LGBM_N_JOBS=1 bash run3.sh time_real_cases.py bdg2_steps1 1
REAL_N_SERIES=300 bash run3.sh time_real_search.py
python check_real_mae.py                             # con pandas 2.3, como los artículos
REAL_N_SERIES=300 LGBM_N_JOBS=1 bash run3.sh time_real_osa_search.py   # 10.6
python time_real_osa_search.py                       # 10.6, configuración del artículo
```
