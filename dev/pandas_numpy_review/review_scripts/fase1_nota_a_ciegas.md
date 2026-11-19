# Fase 1: nota a ciegas (escrita antes de leer INFORME.md)

Código leído en `perf/multiseries-predict-backtesting` (`ab62227`):
`_validation.py:1263-2072`, `_utils.py:1559-1749`,
`_forecaster_recursive_multiseries.py:2634-3235` y `3634-3730`.

## Dónde se crean hoy objetos de pandas por fold y por serie (sin refit)

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

## Cómo lo haría

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

## Condiciones que exigiría (si no se cumplen, camino actual)

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

## Lo difícil de conservar

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
