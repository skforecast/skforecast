# BUG-search-short-series: informar de qué compara cada candidato (propuesta vigente)

Fecha: 2026-10-29. Base: rama `0.27.x` (con #1415, #1418, #1427, #1430, #1431 y #1432
fusionados). Estado: **propuesta, sustituye a la opción c2** de
[`BUG-search-short-series.md`](BUG-search-short-series.md). Ese documento queda como
investigación: el problema, los experimentos 1 a 5 y por qué se descartaron las demás
opciones. Este documento solo trata la solución elegida y la revisión de los avisos y errores
de las búsquedas.

**Bloqueado por QW-12** (decisión del 2026-10-29): antes se hace, por separado, la columna con
el número de observaciones que entran en cada métrica del backtesting (por nivel en
multiserie). Este arreglo la reutiliza en las búsquedas y no la redefine: el nombre, la
cuenta y su documentación son de QW-12.

Entorno de medida: Python 3.14.3, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, optuna 5.0.0,
lightgbm del entorno local de desarrollo. Scripts en
`dev/roadmap_0.27.0/repro/BUG-search-short-series_*.py` (sección 9).

---

## 1. Resumen

- **Problema** (sección 2 del documento anterior): las búsquedas multiserie evalúan cada
  candidato sobre los pares (serie, fold) que puede predecir. Con distinto `window_size`, la
  métrica que se optimiza se calcula sobre puntos distintos en cada candidato, y el usuario
  no lo ve.
- **Decisión:** no cambiar cómo se evalúa ni decidir por el usuario. **Informar** de cuántos
  puntos entran en la métrica de cada candidato y avisar una sola vez, al final de la
  búsqueda, cuando no sean los mismos. El usuario decide: restringir `levels`, quitar
  candidatos o aceptarlo sabiendo lo que compara.
- **De paso, ordenar los avisos y errores de las búsquedas** (sección 5, medido en el
  experimento 6): hoy unos no salen nunca, otros dependen del orden del grid, otros se
  repiten en cada candidato o trial, y un candidato sin datos aborta toda la búsqueda con
  `OneStepAheadFold`.
- **Requisito: QW-12**, la columna de observaciones en las métricas del backtesting. Va
  antes y aparte; este arreglo la lleva a las búsquedas.
- No cambia ningún resultado. Cambio de salida: la columna de QW-12 por candidato en los
  resultados de las búsquedas. Esfuerzo S a M (sin contar QW-12).
- **c2 queda aparcada** como posible opción explícita del usuario (sección 7).

## 2. Por qué no c2

1. **c2 decide por el usuario.** Evalúa "configuración + lo que haga el estimador con los
   lags a NaN". Es menos arbitrario que el fallback naive de la opción d, pero es la misma
   objeción: mete en la métrica una decisión de modelado que el usuario no ha tomado.
2. **No es lo que se despliega.** En producción, un producto de una semana no se predice con
   un modelo de `lags=365` casi todo NaN; se usa una estrategia de cold start propia (otro
   modelo, un baseline, productos análogos, reparto desde la categoría). Medido en M5: en
   cold start, c2 pierde frente a la media de los últimos 28 días (sección 6.4.2 del
   documento anterior). Una métrica con esas predicciones no es fiable.
3. **Coste alto para el tamaño real del problema** (sección 3): `fit`, `predict`,
   transformers, `RollingFeatures`, diferenciación, una línea en "Before upgrading", y solo
   con estimadores que aceptan NaN.

Las tres objeciones a la opción A (avisar) del documento anterior ya no se sostienen:

| Objeción de entonces | Ahora |
|---|---|
| Mirar qué series tienen métricas NaN no detecta el sesgo por folds | se cuentan **puntos**, no series |
| No es accionable | lo es si la información está donde se decide (la tabla de resultados) y el aviso dice qué hacer |
| Depende de la opción B y habría que rehacerlo | no se hace B |

## 3. Tamaño real del problema

Experimento 7 (sin modelo): fracción de puntos de test que deja fuera cada ventana en las
30 490 series de M5 (`fetch_dataset("m5")`).

| periodo de backtesting | `lags=7` | `lags=28` | `lags=56` | `lags=90` | `lags=365` |
|---|---|---|---|---|---|
| últimos 84 días (lo habitual) | 0.00% | 0.01% | 0.03% | 0.06% | 1.61% |
| segundo semestre de 2014 | 0.14% | 0.70% | 1.50% | 2.59% | 12.97% |

En el periodo final, solo el 1.85% de las series se lanzó en el último año. El experimento 5
del documento anterior daba un 8.6% con `lags=56` porque se eligieron a propósito 60 productos
nuevos de 100: servía para ver el mecanismo, no el tamaño.

De qué depende:

1. **De la proporción de series jóvenes respecto a la ventana más larga del grid**, no del
   número de series: muchas series no lo diluyen.
2. **Del rango de ventanas del grid.** 7 frente a 56 es despreciable en un panel maduro;
   con 365 (estacionalidad anual en datos diarios) queda fuera toda serie de menos de un año.
3. **De lo distintos que sean los puntos descartados.** El sesgo es aproximadamente la
   fracción descartada por la diferencia de error entre esos puntos y el resto, y no tiene
   dirección fija (en M5 los productos nuevos venden poco y tienen menos error).
4. **De lo cerca que estén los candidatos.** Suelen diferir un 1% a 3%, así que un sesgo
   pequeño invierte el ranking; por lo mismo, equivocarse cuesta poco (experimento 5, HGBR:
   0.3% de MAE).

Dónde importa: catálogos con mucha rotación (moda, e-commerce, marketplaces, altas de
tiendas o sensores), paneles con poca historia y lags anuales, backtesting en un periodo con
muchos lanzamientos. Y dos daños que no dependen del tamaño: la configuración elegida no
puede predecir esas series en producción, y la métrica reportada es optimista justo en el
segmento de las series nuevas.

**Prioridad:** importante en paneles con rotación o con lags largos; despreciable en paneles
maduros con lags cortos. Lo barato (la columna y el aviso) se justifica igual: es la única
forma de saber en qué caso está el usuario.

## 4. Propuesta: cuántos puntos entran en la métrica de cada candidato

### 4.1 La cuenta

- **Puntos evaluables** de un fold: fechas del test (sin el `gap`) con valor real observado en
  series con algún valor antes del inicio del test. Los definen los datos y el `cv`, no el
  candidato: son los mismos para todos.
- **Puntos evaluados** de un candidato: los evaluables con predicción no NaN, es decir, los
  que entran en sus métricas. Faltan los de series descartadas (última ventana corta, fuera
  del entrenamiento, NaN en la última ventana) y los de predicciones NaN (#1432, y
  `ForecasterDirectMultiVariate` con huecos, sección 7.1 del documento anterior).
- Con el mismo `cv`, dos candidatos son comparables solo si evalúan los mismos puntos. Basta
  con comparar los recuentos: el denominador es común.

### 4.2 Dónde se muestra

1. **Requisito, QW-12 (va antes, aparte):** columna en la tabla de métricas del backtesting
   con las observaciones que entran en cada métrica, por nivel en multiserie, y la suma en
   `average`, `weighted_average` y `pooling`. `_calculate_metrics_backtesting_multiseries`
   ya cuenta las predicciones por nivel para `weighted_average`; QW-12 cuenta las que entran
   en las métricas (predicción y valor real no NaN) y las devuelve. Lo que este documento
   necesita de QW-12: que la cuenta sea la de los puntos evaluados de la sección 4.1 y que el
   agregado sea la suma, porque es lo que se compara entre candidatos. El nombre lo fija QW-12
   (aquí, `n_evaluated` como nombre provisional), alineado con FE-05, que en su salida
   propone `n_predictions`; no puede ser `coverage`, que ya es la cobertura de los intervalos
   (QW-09).
2. **Este arreglo, búsquedas multiserie** (`grid`, `random`, `bayesian`; `TimeSeriesFold` y
   `OneStepAheadFold`): la columna de QW-12 por candidato (la del agregado). En bayesian,
   también como `user_attr` del trial, como las métricas. Con `TimeSeriesFold` sale de la
   tabla de métricas del backtesting de cada candidato. `OneStepAheadFold` no pasa por el
   backtesting: hay que contar lo mismo en
   `_predict_and_calculate_metrics_one_step_ahead_multiseries`, con la misma definición.

### 4.3 Un solo aviso al final de la búsqueda

Si `n_evaluated` no es el mismo en todos los candidatos evaluados, un `MissingValuesWarning`
al final, una vez, fuera del bloque que silencia los avisos:

```
MissingValuesWarning: The candidates are not evaluated on the same points, so their metrics
are not comparable. The most complete candidates are evaluated on 17,224 points; 4 of 12
candidates on fewer (lags=56: 15,748, 91.4%; ...). The selected best candidate is one of
them. Common causes: series shorter than the window of the candidate, missing values in the
last window or NaN predictions. To compare them on the same points, pass in `levels` the
series that every candidate predicts or remove the candidates with larger windows. See the
`n_evaluated` column of the results.
```

- Una línea por candidato con menos puntos, hasta un máximo (por ejemplo 5) y "... y N más":
  la columna tiene el detalle.
- Si `return_best=True` y el mejor tiene menos puntos que el máximo, decirlo: es el caso en
  que el ranking puede estar sesgado a favor del elegido.
- Respeta `suppress_warnings`.
- **Umbral, decisión pendiente** (sección 8): con cualquier diferencia o a partir de un
  porcentaje. En M5, con el periodo habitual, `lags=56` deja fuera el 0.03% de los puntos:
  avisar ahí sería ruido.

### 4.4 Lo que no hace

- No corrige el ranking. Con `return_best=True` se sigue reajustando el mejor por su métrica;
  el aviso dice si el elegido se evaluó con menos puntos.
- Quien tiene mucha rotación no obtiene una comparación justa automática, sino la
  información para hacerla (sección 6, documentación).

## 5. Revisión de los avisos y errores de las búsquedas

### 5.1 Comportamiento actual (experimento 6)

Lo que ve el usuario con los filtros por defecto de Python (`catch_warnings(record=True)`
sin cambiar filtros; con `simplefilter("always")` salen los mismos números, así que ningún
aviso se pierde por el registro de Python: la repetición es real).

| Escenario | Lo que ve | Problema |
|---|---|---|
| grid multiserie, serie corta, `lags_grid=[3, 24]` | ningún aviso de series descartadas | el aviso de #1415 solo sale para el primer candidato (`_evaluate_grid_hyperparameters_multiseries` silencia todas las categorías de skforecast tras él) |
| el mismo grid con `lags_grid=[24, 3]` | 1 `MissingValuesWarning` | depende del orden del grid |
| grid multiserie con `OneStepAheadFold`, serie corta | ningún aviso | el one-step-ahead descarta en silencio |
| bayesian multiserie, `lags` en `search_space`, 10 trials | 6 `MissingValuesWarning` | uno por trial que descarta (bayesian no silencia); el texto incluye el `window_size`, así que no se agrupan |
| grid multiserie, Ridge con un hueco en `exog` (#1432), 6 candidatos afectados | 1 `MissingValuesWarning` | solo el del primer candidato |
| bayesian multiserie, el mismo caso, 10 trials | 10 `MissingValuesWarning` | uno por trial |
| grid multiserie, 2 de 3 `lags` mayores que las series, 3 `alpha` | 6 `RuntimeWarning` "Parameters skipped: ..." con el error completo | uno por combinación, aunque la causa sea el candidato de `lags` (el error se repite para cada `alpha`); `suppress_warnings=True` no los silencia (no es una categoría de skforecast) |
| grid multiserie con `OneStepAheadFold`, un candidato de `lags` sin series que entrenar | **`ValueError` y la búsqueda entera se aborta** | `_train_test_split_one_step_ahead` se llama por candidato de `lags` fuera del `try`; con `TimeSeriesFold` el mismo candidato se salta ("Parameters skipped") |
| grid de una serie, `refit=True`, 60 folds, 6 candidatos | 6 `LongTrainingWarning` con el mismo texto | `_evaluate_grid_hyperparameters` filtra tras cada candidato `category=RuntimeWarning, message="The forecaster will be fit.*"`, pero el aviso es un `LongTrainingWarning`: el filtro no hace nada |
| bayesian de una serie, el mismo caso, 6 trials | 6 `LongTrainingWarning` | uno por trial |
| random multiserie con `n_iter` mayor que el espacio | 1 `UserWarning` de scikit-learn: "... For exhaustive searches, use GridSearchCV." | menor: remite a una clase de scikit-learn, no a `grid_search_forecaster_multiseries` |

### 5.2 Propuesta

Principio: **en una búsqueda, cada problema se cuenta una vez y al final, con cuántos
candidatos afecta.** Los avisos de cada candidato no se muestran sueltos.

1. **Recoger en lugar de silenciar.** Evaluar cada candidato (o trial) dentro de
   `warnings.catch_warnings(record=True)` y, al terminar, emitir una vez cada aviso distinto,
   con el número de candidatos en que salió. Sustituye al silenciado tras el primer candidato
   de grid y random, y a la repetición de bayesian. Agrupar por categoría y mensaje; los
   avisos cuyo texto cambia por candidato (los de #1415 y #1432 incluyen `window_size` o
   recuentos) los cubre el aviso de la sección 4.3, así que no hace falta que su texto
   coincida. Los avisos dentro de procesos de joblib no llegan al proceso principal, pero los
   de #1415 y #1432 ya se emiten en él.
2. **Aviso de la sección 4.3** para la comparabilidad.
3. **"Parameters skipped" agrupado.** Un solo aviso al final con los candidatos saltados,
   agrupados por mensaje de error ("6 candidates skipped, lags = 80, 90 (any params): ...").
   Decidir si `suppress_warnings` lo silencia: hoy no, y un candidato que falta en los
   resultados es información importante; propuesta, mantenerlo visible.
4. **`OneStepAheadFold` por candidato.** Mover `_train_test_split_one_step_ahead` dentro del
   tratamiento por candidato: si un candidato de `lags` no tiene nada que entrenar o evaluar,
   se salta con el aviso del punto 3, como con `TimeSeriesFold`. Ya anotado como pendiente en
   el roadmap (#1430) y en la sección 7 del documento anterior.
5. **`LongTrainingWarning` una vez por búsqueda**, con el total de ajustes (candidatos ×
   ajustes por candidato), antes de empezar cuando se conoce (grid, random) o en el primer
   trial (bayesian). Quitar el filtro con la categoría equivocada.
6. **Menor:** capturar el `UserWarning` de `ParameterSampler` y emitir el equivalente con el
   nombre de la búsqueda de skforecast.

Otras búsquedas comparten parte de estos patrones, aunque no tengan el problema de
comparabilidad (sección 5.3): "Parameters skipped" uno por combinación en las de una serie,
`*_stats` y `grid_search_equivalent_date`; y el mismo filtro sin efecto para
`LongTrainingWarning` (`category=RuntimeWarning`) en `_evaluate_grid_hyperparameters` y
`_evaluate_grid_hyperparameters_stats`. Aplicarles la política de los puntos 1, 3 y 5.

### 5.3 Búsquedas sin problema de comparabilidad

- **Una serie** (`grid_search_forecaster`, ...): el `window_size` solo cambia las filas de
  entrenamiento; los puntos de test son los del `cv` para todos los candidatos. Las
  predicciones NaN dan error desde #1432.
- **`*_stats`** y **`bayesian_search_foundation`**: la ventana o el contexto no dependen del
  candidato.

## 6. Documentación

- Guía de búsqueda de hiperparámetros (`docs/user_guides/hyperparameter-tuning-and-lags-selection.ipynb`)
  y skill `hyperparameter-optimization`: sección "Series de distinta
  longitud: comparar candidatos sobre los mismos puntos". Qué es `n_evaluated`, por qué puede
  diferir y cómo comparar sobre los mismos puntos (restringir `levels`, limitar el rango de
  ventanas). Y que el cold start de series nuevas es una decisión del usuario fuera de la
  búsqueda (otro modelo, un baseline).
- La columna del backtesting (skills `backtesting-configuration` y
  `forecasting-multiple-series`, `llms-base.txt`) se documenta en QW-12. Aquí solo se añade
  que también sale por candidato en las búsquedas. `ai-context-sync`.
- Release note: **Changed**, la columna de QW-12 también en los resultados de las búsquedas
  multiserie (si QW-12 sale en la misma versión, una sola entrada); y **Fixed**:
  `OneStepAheadFold` ya no aborta la búsqueda por un candidato sin datos; avisos de las
  búsquedas que no salían o se repetían. Sin "Before upgrading": no cambia ningún resultado.

## 7. Aparcado: c2 como opción explícita

Si algún día se quiere predecir series con menos historia que la ventana (cold start), debe
ser **un argumento del usuario**, no un comportamiento por defecto que cambie sus métricas.
Lo medido en el documento anterior sigue valiendo como punto de partida:

- Requisito: escalado propio de cada serie aunque no aporte filas (`_unknown_level` lo hace
  inservible; sección 6.4.1).
- El relleno inicial con NaN funciona con diferenciación; los encodings dan igual.
- Pendiente: intervalos con residuos out-of-sample.

**Independiente de c2, a separar como elemento propio del roadmap:**
`RollingFeatures.transform` (predicción) ignora los NaN y `min_periods`, mientras que
`transform_batch` (entrenamiento) los respeta. Pasa hoy cuando la última ventana tiene NaN y
el estimador los acepta. Medido: hacerlo coherente mejora un 2% en el experimento 5.

## 8. Decisiones pendientes y qué medir al implementar

1. **Umbral del aviso de la sección 4.3**: cualquier diferencia, o un porcentaje (por
   ejemplo 1%) o un número mínimo de puntos. Medir con M5 completo en el periodo habitual
   (0.03% con `lags=56`) y en uno con rotación (1.5%), y con el experimento 2 (34%).
2. **`suppress_warnings` y "Parameters skipped"** (sección 5.2, punto 3).
3. **Nombre:** lo decide QW-12 (con FE-05: `n_evaluated` o `n_predictions`); aquí se
   reutiliza.
4. **Bayesian:** la columna informa al final, pero Optuna ya se ha guiado por métricas
   calculadas sobre puntos distintos. Decidir si basta con avisar (propuesta) o si un trial
   con menos puntos debería marcarse en Optuna (`user_attr` ya lo permite sin cambiar la
   optimización).
5. **Coste:** contar puntos y recoger avisos por candidato no debería notarse; medirlo con
   200 series y `OneStepAheadFold`, el camino más rápido.
6. **Tests:** el orden del grid no cambia los avisos; bayesian no repite; `OneStepAheadFold`
   salta el candidato sin datos; `LongTrainingWarning` sale una vez; la columna de cada
   candidato coincide con la del backtesting de QW-12 para la misma configuración, y con
   `OneStepAheadFold` con los puntos de sus métricas. Los casos de la cuenta en sí
   (`fold_stride < steps`, con puntos duplicados que las métricas deduplican, y `gap > 0`)
   son tests de QW-12.

## 9. Scripts

En `dev/roadmap_0.27.0/repro/`:

| Script | Qué muestra | Tiempo aprox. |
|---|---|---|
| `BUG-search-short-series_01_ranking_flip.py` | el ranking se invierte y no hay avisos (`TimeSeriesFold` y `OneStepAheadFold`) | segundos |
| `BUG-search-short-series_02_options.py` | con muchas series cortas, cobertura de cada candidato (20/20 series, 66% de los puntos) | segundos |
| `BUG-search-short-series_06_search_warnings.py` | experimento 6: avisos y errores que ve el usuario en cada búsqueda (sección 5.1) | ~1 min |
| `BUG-search-short-series_07_m5_dropped_share.py` | experimento 7: fracción de puntos que deja fuera cada ventana en M5 completo (sección 3); descarga ~116 MB | ~1 min |

Los scripts 3 a 5 (c1 y c2) están descritos en el documento anterior.

## 10. Relación con el roadmap

- **QW-12:** **bloquea este arreglo**; va antes y aparte (sección 4.2, punto 1). Este
  documento la lleva a las búsquedas.
- **QW-09:** `coverage` es la cobertura de los intervalos; de ahí el nombre `n_evaluated`.
- **FE-05:** mismo dilema al comparar forecasters; usar la misma columna y el mismo nombre.
- **#1415 y #1432:** sus avisos siguen en el backtesting; en las búsquedas los resume el aviso
  de la sección 4.3.
- **#1430:** el error del one-step-ahead sin nada que entrenar o evaluar pasa a tratarse por
  candidato (sección 5.2, punto 4).
- **#1431 (BUG-03):** la dirección de la búsqueda no cambia; el aviso usa el mismo criterio
  para decir cuál es el mejor candidato.
- **FE-04:** mismo principio de comparar sobre los mismos puntos.
