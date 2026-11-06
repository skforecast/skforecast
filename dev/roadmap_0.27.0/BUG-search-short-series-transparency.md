# BUG-search-short-series: informar de qué compara cada candidato (propuesta vigente)

Fecha: 2026-10-29. Base: rama `0.27.x` (con #1415, #1418, #1427, #1430, #1431 y #1432
fusionados). Estado: **propuesta, sustituye a la opción c2** de
[`BUG-search-short-series.md`](BUG-search-short-series.md). Ese documento queda como
investigación: el problema, los experimentos 1 a 5 y por qué se descartaron las demás
opciones. Este documento solo trata la solución elegida y la revisión de los avisos y errores
de las búsquedas.

**PR 1 fusionado en `0.27.x` el 2026-11-04 (#1447):** ver la sección 11. El PR 2 se divide en
dos (sección 8.1): **PR 2a fusionado en `0.27.x` el 2026-11-06 (#1452,** rama
`fix/search-warnings`; sección 12; también `n_evaluated` en todas las búsquedas, 12.8).
**Siguiente: PR 2b** (candidatos que fallan y pendientes de las búsquedas, sección 12.5) y
después el PR 2c (avisos de los procesos de joblib, 12.6 punto 8).

**Desbloqueado el 2026-11-03:** QW-12 está hecho con #1437. Antes estaba **bloqueado por
QW-12** (decisión del 2026-10-29): antes se hace, por separado, la columna con
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

1. **Umbral del aviso de la sección 4.3. Decidido el 2026-11-03: 95%** (primero se decidió
   el 99% y se cambió el mismo día, tras implementarlo; sección 11.5). Avisar cuando algún
   candidato evalúa menos del 95% de los puntos del candidato más completo; la columna da
   siempre el dato exacto. Con M5, el 1.5% de `lags=56` en 2014 ya no avisa.
2. **`suppress_warnings` y "Parameters skipped". Decidido el 2026-11-03:** no lo silencia (un
   candidato que falta en los resultados es información importante); se agrupa en un solo
   aviso en el PR 2 (sección 5.2, punto 3).
3. **Nombre:** decidido el 2026-10-30, `n_evaluated` (QW-12, hecho con #1437,
   fusionado en `0.27.x` el 2026-11-03: última columna de la tabla de
   métricas, la suma en los agregados). Aquí se reutiliza. Es un nombre reservado en los
   backtesting (`ValueError`): sin reservarlo, con una serie salen dos columnas
   `n_evaluated`, y en multiserie la cuenta sustituye en silencio a la métrica del usuario.
   **Hecho en el PR 1 (sección 11).** Antes era pendiente: en las búsquedas, una métrica llamada `n_evaluated` hace fallar el
   backtesting de cada candidato, así que todos se saltan con un "Parameters skipped" y el
   resultado está vacío. Validarlo al principio de las búsquedas, junto a la unicidad de los
   nombres que ya comprueban, cuando la columna llegue a sus resultados.
4. **Bayesian. Decidido el 2026-11-03:** avisar al final y guardar `n_evaluated` como
   `user_attr` de cada trial, sin penalizar el trial en Optuna ni cambiar la optimización.
5. **Coste:** contar puntos y recoger avisos por candidato no debería notarse; medirlo con
   200 series y `OneStepAheadFold`, el camino más rápido.
6. **Tests:** el orden del grid no cambia los avisos; bayesian no repite; `OneStepAheadFold`
   salta el candidato sin datos; `LongTrainingWarning` sale una vez; la columna de cada
   candidato coincide con la del backtesting de QW-12 para la misma configuración, y con
   `OneStepAheadFold` con los puntos de sus métricas. Los casos de la cuenta en sí
   (`fold_stride < steps`, con puntos duplicados que las métricas deduplican, y `gap > 0`)
   son tests de QW-12.

### 8.1 División en dos PRs (2026-11-03)

- **PR 1, el bug:** `n_evaluated` por candidato en grid, random y bayesian multiserie (con
  `TimeSeriesFold` y `OneStepAheadFold`), el aviso único de la sección 4.3, la validación
  del nombre reservado al principio de las búsquedas, docs y nota de la versión.
- **PR 2, los avisos de las búsquedas:** sección 5.2, puntos 1, 3, 4, 5 y 6 (recoger en
  lugar de silenciar, "Parameters skipped" agrupado, `OneStepAheadFold` por candidato,
  `LongTrainingWarning` una vez, `ParameterSampler`), también en las búsquedas de una serie,
  `*_stats` y `grid_search_equivalent_date`.
- **El PR 2 se divide en dos (decisión del 2026-11-04):** PR 2a, los avisos (puntos 1, 3, 5
  y 6; no cambia ningún resultado), y PR 2b, los candidatos que fallan (punto 4, los trials
  de bayesian que fallan y el one-step-ahead sin ningún nivel de `levels` en el test), que
  reutiliza el "Parameters skipped" agrupado del 2a. Ver la sección 12.

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

## 11. PR 1: implementación (2026-11-03)

Rama local `fix/search-short-series-n-evaluated`, desde `0.27.x` con los commits del roadmap
sin subir. Cinco commits: (1) `pooling` y `n_evaluated` en las métricas del one-step-ahead
multiserie, (2) nombre `n_evaluated` validado en todas las búsquedas, (3) columna por
candidato, `user_attr` y aviso, (4) guía, contexto de IA y nota de la versión, (5) este
documento y el roadmap. Entorno: Python 3.14.3, numpy 2.5.4, pandas 2.3.3, scikit-learn
1.9.1, optuna 5.0.0.

### 11.1 Línea base (antes de tocar nada, después de #1437 y #1438)

- Script 01: los mismos números que la sección 2.3 del documento anterior (`average` 2.45
  frente a 4.07 con `TimeSeriesFold`, 1.79 frente a 4.19 con `OneStepAheadFold`, gana
  `lags_24`, 0 avisos). #1438 cambió el agregado que ordena por defecto, pero el script usa
  `aggregate_metric=["average"]`, así que no le afecta.
- Script 06: los once escenarios dan los mismos avisos y el mismo `ValueError` que la tabla
  de la sección 5.1.
- Script 07 (M5): los mismos porcentajes que la sección 3.

### 11.2 Qué hace

- **Columna `n_evaluated`** en los resultados de grid, random y bayesian multiserie, después
  de las métricas y antes de las columnas de los parámetros, como en el backtesting (última
  columna de las métricas). También en el `output_file` de grid y random (escrita como entero:
  `metrics.loc[0, :]` la convertía en float). Es la fila de la primera agregación (la suma por
  nivel; es la misma en las tres) o la del nivel si solo hay uno, y es la misma para todas las
  métricas: una sola columna aunque haya varias métricas o varios `aggregate_metric`.
- **`OneStepAheadFold`:** `_predict_and_calculate_metrics_one_step_ahead_multiseries` devuelve
  `n_evaluated` con la misma definición (`_count_evaluated`) y en la misma posición que el
  backtesting; las tablas de los dos coinciden con `steps=1` y `refit=False` (tests de
  equivalencia, ahora sin quitar la columna).
- **Bayesian:** `n_evaluated` como `user_attr` de cada trial; el valor optimizado no cambia.
  El `output_file` de bayesian es el log de Optuna, así que la columna no va ahí.
- **Aviso:** un `MissingValuesWarning` al final, cuando algún candidato evalúa menos del 95%
  de los puntos del más completo. Texto final (ejemplo del experimento 2; versión de la
  sección 11.5):

  ```
  2 of 3 candidates are evaluated on fewer points than the most complete one, so their
  metrics are not comparable. The best candidate of the ranking is one of them, so the
  ranking may favor it. To compare the candidates on the same points, pass the series
  listed below in `levels` or remove the candidates with larger windows. See the column
  `n_evaluated` of the results.

  Candidates with less than 95% of the 600 points of the most complete one:
    - lags: lags_24, params: {'alpha': 1.0}: 396 (66.0%)
    - lags: lags_12, params: {'alpha': 1.0}: 540 (90.0%)

  Series that every candidate evaluates on the same points:
    levels=['long_0', 'long_1', 'long_2', 'long_3', 'long_4', 'new_06', 'new_07', 'new_12']
  ```

  Los candidatos van en el orden del ranking (máximo 5 y "... and N more"); en bayesian, con
  el número de trial. Arrays de más de 6 lags se abrevian (`[1, 2, ..., 24]`). Si el mejor no
  está afectado, lo dice ("is not one of them"). Respeta `suppress_warnings` (decorador
  `manage_warnings`).
- **Silenciado:** en grid y random, la evaluación de los candidatos y el reajuste van dentro
  de su propio `warnings.catch_warnings()`, así que el silenciado de las categorías de
  skforecast tras el primer candidato no alcanza al aviso, y el reajuste sigue silenciado como
  antes. Un test comprueba que `warnings.filters` es el mismo después de la búsqueda.
- **Nombre reservado:** `_check_metric_names_search` sustituye a la comprobación de nombres
  únicos en las siete funciones que la hacían (grid/random y bayesian de una serie y
  multiserie, `_evaluate_grid_hyperparameters_stats`, `grid_search_equivalent_date` y
  `bayesian_search_foundation`), con el mismo mensaje de unicidad y el del backtesting para
  `n_evaluated`. Antes: DataFrame vacío en grid, random, stats y equivalent date; fallo del
  primer trial con la traza de Optuna en las bayesianas. `coverage` e `interval_width` no hace
  falta reservarlos: las búsquedas llaman al backtesting con `interval=None` (comprobado).

### 11.3 Comprobaciones

- Script 01 (con un script propio sobre sus datos): `n_evaluated` 132 frente a 198 con los
  dos `cv`, igual que el backtesting de `lags=24` (132), y un aviso con el grid en los dos
  órdenes y en bayesian. El script 01 sigue imprimiendo 0 porque solo cuenta los avisos "not
  predicted".
- Experimento 2 (34% de puntos fuera): avisa. Con `TimeSeriesFold`, 396 / 540 / 600 (66%,
  90%, 100%, como la sección 2.4 del documento anterior); con `OneStepAheadFold`, 240 / 450 /
  600 (40%, 75%), más que con backtesting porque el one-step-ahead entrena una vez con el
  entrenamiento inicial. Con `levels` restringido a las 8 series con más de 24 observaciones
  antes del primer fold, los tres candidatos evalúan 240 puntos, no hay aviso y con
  `TimeSeriesFold` gana `lags_12` (8.85 frente a 8.87 de `lags_24`).
- M5 (script 07, sin modelo): con el umbral relativo al candidato más completo (`lags=7`), el
  periodo habitual no avisa con `lags=56` (99.97%) y avisa con `lags=365` (98.4%); el segundo
  semestre de 2014 avisa con `lags=56` (98.6%). Es aritmética sobre las fracciones del
  script: la búsqueda real sobre 30 490 series no cabe en memoria.
- Coste, 200 series de 730 días, Ridge, `OneStepAheadFold`, mediana de 7 repeticiones en dos
  rondas alternas (base `0.27.x` en un worktree aparte):

  | búsqueda | `0.27.x` | rama |
  |---|---|---|
  | grid, 6 candidatos | 0.342 s / 0.331 s | 0.361 s / 0.337 s |
  | bayesian, 6 trials | 0.336 s / 0.337 s | 0.343 s / 0.339 s |

  Entre +1% y +5%, de 1 a 3 ms por candidato, del orden del ruido entre rondas.
- Tests: 655 en `tests_search` y `tests_utils` de `model_selection` y el del split
  one-step-ahead de `ForecasterRecursiveMultiSeries`. Nuevos: columna igual a la del
  backtesting (con `levels` None, parcial y un nivel; los dos `cv`, también el `pooling` con
  `OneStepAheadFold` y `levels` parcial), aviso una vez con el grid invertido, sin aviso por
  debajo del umbral o con `suppress_warnings=True`, `user_attr` en bayesian, `ValueError` en
  cada familia y tests unitarios de los helpers. Los DataFrames esperados de 65 tests se
  actualizaron de forma mecánica (valores sacados de la propia salida).

### 11.4 No previsto

1. **`pooling` del one-step-ahead con `levels` parcial (bug, arreglado aquí por decisión
   del 2026-11-03).** `_predict_and_calculate_metrics_one_step_ahead_multiseries` agregaba en
   `pooling` todas las series de `X_test`, no solo `levels` (`levels=['a', 'b']` de tres
   series daba 2.90, el valor de las tres; los niveles, 0.74 y 0.92). Con `TimeSeriesFold`
   estaba bien. Sin arreglarlo, `n_evaluated` (suma de `levels`) no habría sido el número de
   puntos de `pooling`. Cambia resultados solo en ese caso: línea en "Before upgrading".
2. **El consejo de `levels` del aviso, tal como lo proponía la sección 4.3, no basta.** En el
   experimento 2 las 20 series tienen métricas en los tres candidatos: el sesgo está en los
   pares (serie, fold). Hay que restringir a las series que todos predicen **en todos los
   folds** (las que tienen más observaciones que el mayor `window_size` antes del primer
   fold). Texto cambiado y medido (11.3).
3. **La búsqueda vacía con una métrica `n_evaluated` es de este ciclo** (el nombre se reservó
   en #1437/#1438), así que en la nota de la versión no va en Fixed, sino en la entrada de
   Changed de la columna, según la convención de la skill `release-note`.
4. **Ningún test existente cubría el one-step-ahead con `levels` parcial y `pooling`**: el
   arreglo no cambió ningún resultado esperado; lo cubre el test nuevo de equivalencia.
5. **La guía de búsqueda solo trataba una serie**: la sección nueva usa `items_sales` con
   `item_3` recortado a octubre de 2014 (61 observaciones antes de la validación); con las
   tres series gana `lags=90` evaluado sobre 64 puntos de 96, con `levels` de las dos largas
   gana `lags=7`. Las salidas de otros notebooks con búsquedas multiserie (multi-series,
   métricas, paralelización...) no se reejecutaron: mostrarán la columna en su próxima
   ejecución.
6. **No comprobado:** `bayesian_search_foundation` con `context_length` en el `search_space`
   podría evaluar puntos distintos por candidato (series sin valores observados en la ventana
   de contexto). Según la sección 5.3 no tiene el problema y aquí solo lleva la validación.

### 11.5 Revisión del aviso (2026-11-03, sexto commit)

1. **Umbral al 95%.** El sesgo de la métrica es aproximadamente f × (r − 1), con f la
   fracción de puntos que faltan y r el error de esos puntos frente al resto. Medido en el
   experimento 2, con `lags_3` (predice todo) sobre los puntos de los otros candidatos:
   r = 1.1 (puntos de `lags_12`) y 1.3 (de `lags_24`), aunque las series nuevas tienen el
   triple de ruido; en M5, r < 1. Con un 1% fuera el sesgo es del 0.1% al 1%, por debajo de
   la diferencia habitual entre candidatos (1% a 3%), y cuando están tan cerca equivocarse
   cuesta poco (experimento 5: 0.3%). Con un 5%, del 0.5% al 5%: puede invertir el ranking.
   En M5 avisan `lags=365` en 2014 (13%); ya no avisan `lags=56` en 2014 (1.5%) ni
   `lags=365` en el periodo habitual (1.6%). Experimentos 1 y 2: avisan igual.
2. **Series para `levels`.** Con el `n_evaluated` por nivel de cada candidato (ya está en la
   tabla de métricas de cada candidato), las series que todos evalúan sobre los mismos puntos
   son las que tienen en todos el máximo; pasarlas en `levels` iguala los totales. En el
   experimento 2 son 8 de 20, las mismas que la regla "más de 24 observaciones antes del
   primer fold", y cubre también huecos y predicciones NaN. El aviso muestra la lista más
   corta (esas series, o "all except" las demás) si tiene 10 nombres o menos; si no, el
   recuento y algunos ejemplos de las demás. Sin series comunes, dice que restringir
   `levels` no ayuda; con un solo nivel (`ForecasterDirectMultiVariate`), no sugiere
   `levels`. No se devuelve la lista en código (`results.attrs` es experimental en pandas y
   un valor de retorno nuevo rompería la API); queda para FE-05 si hace falta.
3. **Formato (séptimo commit), siguiendo el `FitFailedWarning` de scikit-learn** (qué
   pasa, la consecuencia y una acción; el detalle al final, tras una línea en blanco):
   primer párrafo con el recuento, si el mejor está afectado (en singular con un solo
   candidato: "It is (not) the best candidate of the ranking") y la acción; después, la
   lista de candidatos, con el 95% y los puntos del más completo en su cabecera (sin el
   "fewer than 570", que repetía lo que dicen los porcentajes); al final, las series para
   `levels` como `levels=[...]` si son 10 o menos, `levels: every series except [...]` si
   lo son las demás, o su número con ejemplos de las que quedan fuera. Se quitaron las
   "Common causes" (no deciden nada; la guía las explica) y, con ellas, la fecha y el
   `window_size` del texto. Sin series comunes, la acción dice que restringir `levels` no
   ayuda y no hay bloque de series.
4. **Revisión final (2026-11-04).** Una revisión independiente del diff encontró una
   regresión: con `kwargs_create_study` que carga un study guardado (`load_if_exists`),
   bayesian daba `KeyError` al final, porque los trials de llamadas anteriores no tienen la
   cuenta por serie (ni `n_evaluated`, si son de una versión anterior). Arreglado: el aviso
   usa solo los trials de la llamada y la columna lee `n_evaluated` con NaN por defecto.
   Descartado tras comprobarlo: el `pooling` del one-step-ahead cuando ningún nivel de
   `levels` está en el test falla igual que en `0.27.x` (antes, en la media ponderada); el
   candidato se salta, y queda para el PR 2 (one-step-ahead por candidato).

## 12. PR 2a: avisos de las búsquedas (2026-11-04)

Rama `fix/search-warnings` desde `0.27.x` (con #1447); PR #1452 abierto el 2026-11-05 y
fusionado el 2026-11-06. Tres commits:
la primera frase fija del aviso de #1432, el registro de los avisos en las búsquedas (con los
tests) y la documentación.

### 12.1 Línea base y resultado (script 06)

| Escenario | `0.27.x` | PR 2a |
|---|---|---|
| grid multi, serie corta, lags [3, 24] | `MissingValuesWarning` ×1 (solo `n_evaluated`; el de #1415 silenciado) | ×2: #1415 "(in 2 of 4 candidates)" y `n_evaluated` |
| el mismo grid, lags [24, 3] | ×2 | ×2, el mismo texto que en [3, 24] |
| bayesian multi, serie corta, 10 trials | ×7 | ×2: #1415 agrupado "(in 6 of 10 trials)" y `n_evaluated` |
| bayesian multi, hueco en `exog` (#1432), 10 trials | ×10 | ×1, tal cual (mismo texto en todos los trials) |
| grid multi, 2 de 3 lags fallan | `RuntimeWarning` ×6 | ×1, agrupado por error ("every params (3 candidates)") |
| grid de una serie, `refit=True`, 6 candidatos | `LongTrainingWarning` ×6 | ×1: "Each candidate fits the forecaster about 60 times, so the search (6 candidates) fits it about 360 times..." |
| bayesian de una serie, el mismo caso | ×6 | ×1 ("6 trials") |
| random multi con `n_iter` mayor que el espacio | `UserWarning` que remite a `GridSearchCV` | ×1, remite a `grid_search_forecaster_multiseries` |
| grid multi one-step-ahead, un candidato sin datos | `ValueError`, se aborta | sin cambios (PR 2b) |

### 12.2 Qué hace

- **Registro único por búsqueda** (`_SearchWarnings` en `_utils.py`): envuelve el bucle de
  candidatos (en bayesian, `study.optimize`) y, aparte, el `fit` del refit. Sustituye
  `warnings.showwarning` y marca cada aviso con el candidato del hilo (`threading.local`;
  en bayesian, el número del trial, que fija el objetivo al empezar). Los avisos del refit
  o de fuera de los candidatos (split del one-step-ahead) van sin candidato: salen, pero no
  cuentan. Al final, fuera del contexto, cada grupo se emite con `warnings.warn`, que pasa
  por los filtros del usuario y de `suppress_warnings`.
- **Agrupación:** mismo texto en todos los candidatos, o solo fuera de ellos: tal cual; en
  algunos: "(in N of M candidates)" tras la primera frase; textos distintos con la misma
  primera frase: la frase, el recuento y "Run the backtesting of a candidate to see the
  details." (decisión A, opción 2). El aviso de #1432 empieza ahora con una frase fija
  ("Some predictions are NaN... (5 in total)"); #1432 es de este ciclo, sin "Before
  upgrading".
- **Filtros:** dentro del registro, las acciones que muestran un aviso una vez (`default`,
  `module`, `once`) pasan a `always` y se añade `always` al final, para contar las
  repeticiones; `ignore` y `error` se respetan (lo ignorado no se registra). Sin esto, un
  filtro general `default` (`-W default`, `PYTHONWARNINGS`) daba recuentos falsos ("in 1 of
  4").
- **"Parameters skipped"** (decisión D): un solo `RuntimeWarning` al final, agrupado por
  error (tipo y texto), con el prefijo de antes para no romper filtros; si todos los
  parámetros de un `lags` fallan con el mismo error, una línea "every params (N
  candidates)"; hasta 5 errores y 5 candidatos por error. Sustituye también a "No valid
  parameter combinations found..." ("N of N candidates raise an error, so the results are
  empty"). `suppress_warnings` no lo silencia. En `*_stats` y `grid_search_equivalent_date`,
  solo `params`.
- **`LongTrainingWarning`:** el primero se pasa en directo al `showwarning` anterior (rich
  o el del usuario), reescrito con el total de la búsqueda (lee el número de ajustes del
  texto del backtesting; con `ForecasterDirect`, "(60 folds * 10 estimators)"); los demás
  se descartan. Se quitan los filtros con la categoría equivocada de
  `_evaluate_grid_hyperparameters` y `_evaluate_grid_hyperparameters_stats`.
- **`ParameterSampler`:** `_sample_param_grid` silencia el aviso de scikit-learn y emite uno
  que remite a `grid_search_forecaster`, `grid_search_forecaster_multiseries` o
  `grid_search_stats` ("for each value of `lags_grid`" solo si se pasa `lags_grid`).
- Búsquedas cubiertas: grid y random de una serie y multiserie, las tres bayesian (también
  `bayesian_search_foundation`), `*_stats` y `grid_search_equivalent_date`. El silenciado
  tras el primer candidato del grid multiserie desaparece.
- Docstrings de `suppress_warnings`, nota de la versión en Changed, punto 9 de "Common
  mistakes" en la skill `hyperparameter-optimization`.

### 12.3 Comprobaciones

- Tests: 824 en `tests_search`, `tests_utils` y los backtesting multiserie y de foundation.
  Nuevos: `_SearchWarnings` (tal cual, recuento, agrupado, `n_candidates` de `emit`,
  filtros respetados y restaurados, filtros `default`/`module`/`once`,
  `LongTrainingWarning`, hilos, saltados), `_warn_skipped_candidates_search`,
  `_sample_param_grid`; en las búsquedas: el grid multiserie da los mismos avisos en los dos
  órdenes (y el del refit sale tal cual), `suppress_warnings=True` conserva "Parameters
  skipped", bayesian multiserie con `n_jobs` de Optuna 1 y 2 (repetido 8 veces sin fallos),
  `LongTrainingWarning` una vez en grid y bayesian de una serie, y el aviso de `n_iter` en
  random multiserie.
- Coste (200 series × 730 días, Ridge, `OneStepAheadFold`, mediana de 7 en dos rondas
  alternas frente a `0.27.x`): grid de 6 candidatos 0.346/0.350 s → 0.344/0.345 s; bayesian
  de 6 trials 0.349/0.344 s → 0.344/0.347 s. Sin diferencia medible. Python 3.14.3, numpy
  2.5.4, pandas 2.3.3, scikit-learn 1.9.1, optuna 5.0.0.

### 12.4 Decisiones y no previsto

1. **Hilos (decisión C):** un único contexto por búsqueda, abierto en el hilo principal, en
   lugar de un `catch_warnings` por candidato (que toca el estado global en cada trial). Los
   `catch_warnings` de `manage_warnings` dentro de los trials en hilos ya competían antes por
   los filtros globales; no cambia. Con Python 3.14 free-threaded (`context_aware_warnings`)
   no se ha probado.
2. **Optuna (decisión B, para el PR 2b):** recomienda el argumento `catch` de
   `study.optimize` o devolver NaN (el trial queda `FAIL` sin abortar). Propuesta: solo la
   evaluación dentro del `try`, relanzada como una excepción privada que se añade al
   `catch` del usuario; los errores de configuración siguen abortando.
3. El `DataTransformationWarning` de un modelo lineal sin `transformer_series` sale al crear
   el forecaster, fuera de la búsqueda: no se agrupa (no hace falta).
4. Los avisos de los procesos de joblib (backtesting con `n_jobs` mayor que 1, que
   `n_jobs='auto'` elige para algunos forecasters, también con `refit=False`) siguen sin
   llegar, como antes.

### 12.5 Pendiente: PR 2b (candidatos que fallan)

1. `_train_test_split_one_step_ahead` dentro del `try` de cada candidato en grid y random,
   de una serie y multiserie: un candidato sin datos se salta con el "Parameters skipped"
   agrupado, también cuando ningún nivel de `levels` tiene puntos en el test.
2. Bayesian deja de abortar cuando falla la evaluación de un trial (12.4, punto 2).
3. El `ValueError` de `initial_train_size` (12.6, punto 7): separar los casos y mostrar
   el valor recibido; comprobar el límite `initial_train_size == window_size` (el código
   lo acepta, el mensaje dice "greater than"). Nota en Changed.
4. El refit del mejor candidato falla con `ValueError: Length of 'short' must be greater
   than the maximum window size...` cuando una serie tiene `window_size` observaciones o
   menos en todos los datos: el backtesting la deja fuera en cada fold, pero `fit` la
   rechaza, y la búsqueda termina con un error tras evaluar todos los candidatos. Visto en
   los tests del 12.7 (serie de 6 valores, `lags=8`). No es barato: el error llega tras
   evaluar todos los candidatos. Fallar antes (al empezar, si una serie es más corta que la
   ventana de algún candidato) sería demasiado estricto: el backtesting multiserie deja
   fuera esas series a propósito (#1415) y la búsqueda solo falla si el mejor candidato es
   uno de los de ventana grande. **Propuesta:** el refit deja fuera, con un aviso, las series
   con `window_size` observaciones o menos (las mismas que el backtesting no predice nunca
   con ese candidato); el forecaster devuelto no las conoce. Confirmado por el usuario
   (2026-11-05).
5. **Aviso de `n_evaluated` en las búsquedas de una serie** (12.8, hallazgo 1; decidido el
   2026-11-05): con `OneStepAheadFold`, `dropna_from_series=True` y NaN en `y`, los
   candidatos de ventana grande se evalúan sobre menos puntos. Variante del aviso sin
   `levels`: qué candidatos tienen menos del 95% de los puntos y que la solución es rellenar
   los huecos de `y` o quitar las ventanas grandes. Mismo umbral y agrupación por `lags` que
   el multiserie (reutilizar `_warn_n_evaluated_search_multiseries` con un parámetro, o
   separar el texto). En grid, random y bayesian de una serie; test con el ejemplo del 12.8.
6. Tests, nota de la versión (Fixed: el one-step-ahead abortaba la búsqueda) y roadmap.

### 12.6 Revisión como usuario (notebook 08, 2026-11-04)

`repro/BUG-search-short-series_08_search_warnings.ipynb`: un escenario por aviso, con datos
reales (`store_1`, `store_2` y `new_product` de 120 días). Revisado como un usuario que
conoce skforecast pero no sus detalles.

Bien: `LongTrainingWarning` (claro, a tiempo, accionable), "Parameters skipped" (error
primero, candidatos debajo), `n_iter` (remite a la función de skforecast), #1432 tal cual, y
los dos órdenes del grid dan los mismos avisos.

**Aprobado para el PR 2a (implementado el 2026-11-05):**

1. **Agrupado sin perder lo accionable:** con textos distintos y la misma primera frase, se
   muestra el texto completo del primero con "(in N of M candidates; details of the first
   one)" en lugar de "Run the backtesting of a candidate to see the details.". Antes se
   perdía el consejo del `ConvergenceWarning` de scikit-learn y la serie del aviso de #1415.
2. **Ubicación:** los avisos reemitidos ("Parameters skipped", `n_iter`, los de otras
   librerías) apuntan a la línea del usuario, no a `_utils.py` con
   `warnings.warn("\n\n".join(blocks), RuntimeWarning)`.
3. "Parameters skipped": "lags: [1, 2, ..., 400], every params (2 candidates)" →
   "lags: [1, 2, ..., 400] (2 candidates)" (la línea agrupada solo sale si fallan todos
   los parámetros de esos `lags`).
4. ~~Redondear los floats de `params` en el aviso de `n_evaluated` del PR 1~~: sobra con
   el punto 6, que quita `params` de las líneas.

**Propuesto después (5 y 6 implementados en el PR 2a el 2026-11-05; 7 y 8, pendientes):**

5. **Texto del aviso de #1415 (aprobado)** (de este ciclo, sin publicar): la línea
   "- `window_size` (35) observations or fewer in the training set: 'new_product' (never
   predicted)." no se entiende ni se engancha con la frase anterior.
6. **Aviso de `n_evaluated` agrupado por `lags` (aprobado):** en el grid salen tres líneas
   iguales ("lags: [1, 2, ..., 35], params: {...}: 184 (66.7%)"), una por `alpha`. Se
   agrupan los candidatos con los mismos `lags` y el mismo `n_evaluated`, sin `params`, con
   el número de fila de la tabla de resultados (grid y random) o del trial (bayesian), como
   mucho 5 por línea: "lags: [1, 2, ..., 35], 3 candidates (rows 0, 1, 2): 184 (66.7%)",
   "lags: [1, 2, ..., 35], 60 trials (5, 7, 8, 12, 15, ...): 184 (66.7%)". Como mucho 5
   líneas y "... and N more", como ahora. Sin `params`, el redondeo del punto 4 sobra.
7. **`ValueError` de `initial_train_size`** (publicado, en `_utils.py` ×3 y `_split.py`):
   con `lags=400` y 365 observaciones dice "If `initial_train_size` is an integer, it must
   be greater than the `window_size` (400) and smaller than the length of `y` (365)..."
   aunque se pasó una fecha, y no hay ningún valor que lo cumpla: el problema es la ventana.
   Separar los casos (ventana mayor que la serie, `initial_train_size` menor que la
   ventana, mayor que la serie) e incluir el valor recibido. **Va al PR 2b.**
8. **Avisos de los procesos de joblib** (ya existía, también en el backtesting): con
   `n_jobs='auto'` cada proceso imprime sus avisos (21 paneles en el escenario 3; 3 con
   `n_jobs=1`). Arreglo posible: las funciones de cada fold en `_validation.py` registran
   sus avisos y los devuelven, y el proceso principal los reemite tras `Parallel`.
   **Va al PR 2c** (después del 2b). Desde el proceso, rich no detecta Jupyter ni 256 colores: los paneles
   salen amarillos (paleta de 16 colores), con 80 columnas y con `�` cuando un carácter del
   borde se corta entre dos trozos de la salida (revisión del 2026-11-05). Reemitidos desde
   el proceso principal, salen como los demás.

Fuera de alcance, ya existía: con todos los candidatos saltados, los resultados vacíos
tienen la columna `params` duplicada.

**Implementación (2026-11-05):**

- La ubicación se calcula subiendo por la pila hasta el primer frame fuera del paquete
  (`_first_frame_outside_skforecast`, `_stacklevel_user_code`); los tests de skforecast
  cuentan como código del usuario. Vale para Python 3.10 (`skip_file_prefixes` es de 3.12 y
  no excluye los tests). `LongTrainingWarning` usa la ubicación guardada al entrar en el
  registro, en el hilo principal, porque los trials de Optuna pueden ir en hilos.
- Con textos distintos y la misma primera frase, todos fuera de los candidatos (refit), se
  muestran tal cual.
- El aviso de `n_evaluated` usa el nombre del candidato también en el resumen y la cabecera
  ("6 of 10 trials are evaluated...", "Trials with less than 95%...").
- Nota de la versión: la entrada de Changed dice que los avisos apuntan a la línea del
  usuario. El notebook 08 se ha vuelto a ejecutar con los textos nuevos. La salida de la guía
  `hyperparameter-tuning-and-lags-selection.ipynb` muestra todavía el formato anterior del
  aviso de `n_evaluated`.

### 12.7 Un solo aviso para las series cortas en las búsquedas (2026-11-05)

En los escenarios 1 y 2 salían dos avisos que decían casi lo mismo: el de series no predichas
(#1415, la causa: "'new_product' too short for the `window_size` (35)") y el de
`n_evaluated` (la consecuencia para el ranking y la acción). Las búsquedas multiserie dejan
fuera el primero cuando el segundo lo cubre:

- **Regla** (`_n_evaluated_covers_levels_not_predicted`): sale el aviso de `n_evaluated` y
  todas las series tienen puntos evaluados en algún candidato. Las series que algunos
  candidatos no predicen son las que faltan en `levels=[...]`, y la acción es la misma.
- **Se mantienen los dos** cuando una serie no tiene puntos en ningún candidato (más corta
  que la ventana más pequeña): el aviso de `n_evaluated` la deja fuera a propósito, y sin el
  otro desaparecería sin aviso. Con una serie sin datos en el test pasaría igual, pero el de
  #1415 no la menciona (no hay nada que evaluar); queda como antes, con los dos.
- **Sale solo el de #1415** cuando los candidatos son comparables (todos la misma ventana, o
  diferencia menor del 5%): es el único que dice que una serie no entra en las métricas.
  Escenario 1b del notebook 08: un producto de temporada con 2 puntos en el test.
- Se pierde, en la búsqueda, la distinción entre ventana corta y huecos en la última ventana;
  la acción del aviso de `n_evaluated` vale para las dos, y el backtesting de un candidato
  muestra el detalle.
- Código: `_SearchWarnings.discard(category, text_start)` y la constante
  `LEVELS_NOT_PREDICTED_WARNING`, con la que empieza el aviso de #1415. Sin cambios en el
  texto del aviso de `n_evaluated`. Tests: `discard`, la regla (4 casos), el grid con las
  dos series cortas (solo `n_evaluated`) y el grid y bayesian (también con hilos) con una
  serie de un punto en el test (solo #1415, con el recuento). Skill: el punto 9 ya no dice
  "run the backtesting of a candidate".

### 12.8 `n_evaluated` en todas las búsquedas (2026-11-05, en #1452)

Decisión del usuario: la columna sale siempre, también en las búsquedas de una serie, para que
`results` tenga el mismo esquema en todas (como `cv_results_` de scikit-learn) y coincida con
el backtesting, que ya la da siempre. El aviso sigue solo en las multiserie: la columna es un
dato; el aviso, un problema.

- Búsquedas: grid, random y bayesian de una serie, `grid_search_stats`, `random_search_stats`,
  `grid_search_equivalent_date` y `bayesian_search_foundation` (una serie y multiserie). Tras
  las métricas, también en el `output_file` y como `user_attr` (NaN para los trials de un
  estudio de una versión anterior). `_calculate_metrics_one_step_ahead` devuelve
  `(metric_values, n_evaluated)`.
- Tests: 37 tests de una serie con `expected_results.insert(pos, 'n_evaluated', [n] * k)`
  (valores tomados de los resultados; todos constantes, también con `OneStepAheadFold`); el
  test que comprobaba que no estaba la columna ahora comprueba que coincide con el
  backtesting; foundation (una serie y multiserie); el helper con NaN en `y`. 836 tests.
- Docs: `llms-base.txt`, `references/search-parameters.md`, nota de #1447 (ahora "the
  results of the hyperparameter searches include `n_evaluated`"), el highlight y "Before
  upgrading".
- **Hallazgo 1 (en el PR 2b, 12.5 punto 5):** con una serie, `OneStepAheadFold`,
  `dropna_from_series=True` y NaN en `y`, `n_evaluated` sí cambia entre candidatos: las filas
  de test con NaN en los lags se eliminan, más cuanto mayor la ventana. Ejemplo: 200 días, 3
  NaN, test de 57 puntos: `lags=14` evalúa 15 y `lags=3` 48, y gana `lags=14`. Es el mismo
  problema de comparabilidad, sin aviso. El aviso multiserie no sirve tal cual (habla de
  `levels`); haría falta una variante de una serie ("rellena los huecos o quita las ventanas
  grandes"). Con `TimeSeriesFold` no pasa: los puntos de test son los mismos.
- **Hallazgo 2 (en el roadmap como BUG-backtesting-nan-y-true, sección 4):** `backtesting_forecaster` con NaN en `y` dentro del
  test falla con `ValueError: Input contains NaN.` de la métrica de scikit-learn (con
  `Ridge` y con LightGBM): las fechas con `y_true` NaN no se quitan antes de las métricas,
  como sí hace el multiserie. Por eso una búsqueda de una serie con NaN en el test se queda
  vacía con `TimeSeriesFold`.

## 13. Al final de todas las PRs (2b, 2c)

1. Volver a ejecutar `docs/user_guides/hyperparameter-tuning-and-lags-selection.ipynb`: su
   salida muestra el formato anterior del aviso de `n_evaluated` (una línea por candidato,
   con `params`), y las tablas de las búsquedas de una serie no tienen
   `n_evaluated` (12.8).
2. **`n_evaluated` en las búsquedas de una serie (12.8):** al volver a ejecutar la guía de
   búsqueda, explicar la columna en la primera búsqueda (de una serie): los puntos sobre los
   que se calcula la métrica de cada candidato, los mismos que da el backtesting; hoy solo
   se explica en la sección multiserie. Las demás guías con resultados de búsquedas también
   ganan la columna al volver a ejecutarlas (`forecasting-statistical-models`,
   `forecasting-baseline` con `grid_search_equivalent_date`, foundation...): revisar que
   ningún texto cuente las columnas o las seleccione por posición.
3. Usar la columna `y_true` de las predicciones del backtesting (#1437) en **todas** las
   páginas con backtesting, en lugar de buscar los valores reales con `data.loc[...]` o de
   pintar el test entero aparte. Es grande, por eso va al final. Páginas que llaman a
   `backtesting_*(`: 32 guías de `docs/user_guides/`, 6 de `docs/faq/`
   (`cyclical-features-time-series`, `forecasting-time-series-with-missing-values`,
   `forecasting-with-delayed-historical-data`, `non-negative-predictions`,
   `parameters-search-backtesting-vs-one-step-ahead`, `parallelization-skforecast`) y
   `docs/quick-start/quick-start-skforecast`. Las que ya usan el patrón
   `data.loc[predictions.index, ...]` (número de apariciones):
   `autoregressive-classification-forecasting` (8),
   `probabilistic-forecasting-bootstrapped-residuals` (3), `comparing-forecasting-models`
   (2), `forecasting-with-deep-learning-rnn-lstm` (2),
   `probabilistic-forecasting-conformal-prediction` (2), `probabilistic-forecasting-metrics`
   (2), `probabilistic-forecasting-global-models` (1). Revisar cada caso: `y_true` solo está
   en las predicciones del backtesting, no en las de `predict`, y en multiserie va por fila
   (formato largo). Volver a ejecutar cada página tocada.

## 14. PR 2b: candidatos que fallan y pendientes de las búsquedas (2026-11-06)

Rama `fix/search-failing-candidates` desde `0.27.x` (con #1447 y #1452). Alcance: 12.5.
Script de la línea base: `repro/BUG-search-short-series_09_failing_candidates.py` (un
escenario por punto; avisos con los filtros por defecto, primera línea).

### 14.1 Línea base en `0.27.x`

| Punto | Escenario | `0.27.x` |
|---|---|---|
| 1 | grid y random de una serie, OSA, `lags_grid=[3, 80]` (67 obs. de train) | `ValueError: Length of y must be greater than the maximum window size`, se aborta |
| 1 | grid y random multiserie, OSA, `lags_grid=[3, 80]` | `ValueError: No series can be used to train the forecaster`, se aborta |
| 1 | grid multiserie, OSA, `levels=['short']`, `lags_grid=[3, 24]` (`short` no entra con 24) | no se aborta: 2 filas con `n_evaluated=0` y métricas NaN, un `FutureWarning` de pandas (concat vacío) y el aviso de `n_evaluated`. Con `TimeSeriesFold`, esos 2 candidatos se saltan ("Parameters skipped") |
| 1 | bayesian de una serie y multiserie, OSA, `lags` en [3, 80] | el mismo `ValueError`, se aborta el study |
| 2 | bayesian de una serie y multiserie, TSF, `lags` en [3, 80] | `ValueError` de `initial_train_size`, se aborta; además Optuna imprime en stderr "Trial 0 failed..." con la traza (el logger de optuna no tiene handler y cae en `lastResort`) |
| 2 | bayesian, claves de `search_space` distintas / `search_space` que lanza / `alpha='x'` | se aborta (`ValueError`, `ZeroDivisionError`, `InvalidParameterError`) |
| 3 | backtesting, `lags=80`, `initial_train_size='2021-07-01'` (67 obs.) | "If `initial_train_size` is an integer, it must be greater than the `window_size` of the forecaster (80) and smaller than the length of `y` (96)..." (se pasó una fecha) |
| 3 | backtesting, `lags=100`, `len(y)=96` | el mismo texto: ningún valor lo cumple |
| 3 | `initial_train_size == window_size` (3): backtesting, multiserie y OSA | lo acepta el chequeo y falla después: "Length of `y` must be greater than the maximum window size" (una serie, OSA) y "No fold of the backtesting has a level to predict" (multiserie). Con 4 funciona |
| 4 | grid, random y bayesian multiserie, TSF y OSA, `tiny` de 6 valores, `lags=8`, `return_best=True` | tras evaluar todo, `ValueError: Length of 'tiny' must be greater than the maximum window size` en el refit |
| 5 | grid, random y bayesian de una serie, OSA, `dropna_from_series=True`, 3 NaN en el test (57 puntos), `lags_grid=[3, 14]` | `lags=14` evalúa 12 puntos y `lags=3` 45; gana `lags=14`, sin aviso |

Otros hallazgos de la línea base:

- En bayesian con OSA, el aviso de los NaN quitados del split sale "(in 2 of 6 trials)": el
  split se cachea por `lags` y solo se calcula (y avisa) en el primer trial de cada valor.
  En grid sale tal cual (el split va fuera de los candidatos).
- **`initial_train_size == window_size` en los demás forecasters** (TSF, `refit=False`,
  sección 3b del script): ya fallaba, con errores tardíos y distintos, en
  `ForecasterDirect` ("Minimum length of `y`... is 6", también con `window_size + 1`, porque
  necesita `window_size + steps`), `ForecasterDirectMultiVariate` y `ForecasterRnn` ("No
  fold of the backtesting has a level to predict"), `ForecasterEquivalentDate` y
  `ForecasterRecursiveClassifier` ("Length of `y` must be greater than the maximum window
  size"). **`ForecasterStats` no fallaba**: su `window_size` es 1 y se ignora; un Arima se
  ajusta con 1 observación. Para él se mantiene el límite de antes (`<`).
- **Refit con series cortas en otros forecasters multiserie** (sección 4b):
  - `ForecasterDirectMultiVariate` (DataFrame ancho, `tiny` con NaN salvo 6 valores): el
    backtesting quita `tiny` del train de cada fold (es corta) y los candidatos se
    evalúan, pero el refit la incluye y `Ridge` falla con `Input X contains NaN`. No es la
    regla de longitud de `fit` (`ForecasterDirectMultiVariate` no la tiene): con un
    estimador que admite NaN, el refit funcionaría con `tiny` casi vacía como predictor.
    **Fuera de este PR** (otra causa; anotarlo en el roadmap).
  - `ForecasterRnn`: no se puede usar en las búsquedas: `set_params` clona el modelo de
    keras y falla (`TypeError: Cannot clone object '<Functional ...>'`), con o sin `tiny`;
    todos los candidatos se saltan. No aplica. Aparte, su backtesting con una serie corta
    en el DataFrame falla con "Number of series in `series` (2) does not match the number
    of series expected by the model architecture (3)": `_extract_data_folds_multiseries`
    quita la columna corta y el modelo espera todas. Ya existía; fuera de este PR.
    `set_params` falla siempre, también fuera de las búsquedas: al roadmap como
    BUG-rnn-set-params.

### 14.2 Decisiones (aprobadas el 2026-11-06)

1. **One-step-ahead:** el split, dentro de un `try` por cada `lags` en grid y random; si
   falla, los `params` de ese `lags` se saltan con su error. Tras el split, si ningún nivel
   de `levels` tiene puntos en el test, `ValueError` y se saltan (antes: filas con
   `n_evaluated=0` y métricas NaN). En OSA, `initial_train_size > window_size` se comprueba
   por candidato con el mismo helper del punto 3.
2. **Bayesian:** abortan `search_space(trial)`, la comprobación de sus claves y
   `set_params`/`set_lags` (errores del espacio de búsqueda). Solo la evaluación (split,
   backtesting, ajuste y métricas) va en el `try`, relanzada como `_TrialEvaluationError`,
   que se añade al `catch` del usuario. En "Parameters skipped", cada trial por su número.
   `add_skipped` con `self._lock` (hilos de Optuna). Si fallan todos, resultados vacíos.
   El split de OSA va fuera del candidato (`set_candidate(None)`) y se cachea también su
   error. También en `bayesian_search_foundation`.
3. **Logs de Optuna:** filtro en el logger `optuna.study._optimize` (un filtro en `optuna`
   no se aplica a los registros de sus hijos), solo para los trials que fallan con
   `_TrialEvaluationError`, quitado en un `finally`, y solo sin `output_file` (en el
   fichero se mantienen).
4. **`initial_train_size`:** helper con tres casos (ventana demasiado grande, valor menor o
   igual que la ventana, valor mayor o igual que la longitud) y el valor recibido. Límite
   `<=` en los chequeos de `_utils.py` y en el camino explícito de `_split.py`; no en el de
   `initial_train_size=None` (forecaster ya ajustado: `initial_train_size = window_size` a
   propósito), ni en `ForecasterStats` ni en `ForecasterFoundation`. Changed: error más
   claro y temprano.
5. **Refit sin series cortas:** solo `ForecasterRecursiveMultiSeries`; las series que `fit`
   rechazaría (`len(y) <= window_size` tras `align_series_and_exog_multiseries`), con la
   regla compartida con `fit`. `exog` solo se filtra si es un dict. El forecaster queda como
   un `fit` sin esas series (sin atributos nuevos). El aviso **no** lo silencia
   `suppress_warnings`: describe el forecaster devuelto, como "Parameters skipped" describe
   los resultados.
6. **`n_evaluated` de una serie:** `_warn_n_evaluated_search_multiseries` pasa a
   `_warn_n_evaluated_search` con `levels=None` para una serie (otro texto de acción).

### 14.3 Resultado (script 09 y notebook 09)

| Punto | Escenario | PR 2b |
|---|---|---|
| 1 | grid y random de una serie y multiserie, OSA, `lags_grid=[3, 80]` | 2 filas (lags 3); "Parameters skipped: 2 of 4 candidates" con el error de `initial_train_size` (comprobado por candidato) |
| 1 | grid multiserie, OSA, `levels=['short']`, `lags_grid=[3, 24]` | 2 filas (lags 3); los de 24, saltados: "No level of `levels` can be evaluated in the test set with this `window_size` (24)..." Sin el `FutureWarning` de pandas |
| 1 y 2 | bayesian de una serie y multiserie, OSA y TSF, `lags` en [3, 80] | trials de 80 en `FAIL`; "Parameters skipped: 2 of 6 trials", cada uno por su número; sin las trazas de Optuna en stderr |
| 2 | claves de `search_space` distintas / `search_space` que lanza | siguen abortando (con el log de Optuna de antes) |
| 2 | `alpha='x'` (falla en el `fit`) | se salta, como en grid |
| 2 | todos los trials fallan, `return_best=True` | resultados vacíos y el aviso "so the results are empty" (antes, `ValueError` del primer trial) |
| 3 | `lags=80`, fecha | "`initial_train_size` must be greater than the `window_size` of the forecaster (80), so that the first training set has data to train with. Got '2021-07-01', 67 observations in the training set." |
| 3 | `lags=100`, 96 obs. | "The `window_size` of the forecaster (100) leaves no valid `initial_train_size`, which must be greater than the `window_size` and smaller than the length of `y` (96). Reduce the `window_size` (lags, window features, differentiation...) or use more data." |
| 3 | `initial_train_size == window_size` | el mismo error que con un valor menor, en el chequeo (backtesting, multiserie y OSA) |
| 4 | grid, random y bayesian multiserie, TSF y OSA, `tiny` de 6 valores, `lags=8` | la búsqueda termina; `series_names_in_ == ['l1', 'l2']`; `RuntimeWarning` "1 series is left out of the refit of the best candidate: ... Series left out: 'tiny'." |
| 5 | grid, random y bayesian de una serie, OSA, NaN en `y` | `MissingValuesWarning` "... The rows with missing values in `y` or in its lags are dropped (`dropna_from_series=True`), more with larger windows. To compare the candidates on the same points, fill the gaps of `y` or remove the candidates with larger windows..." |

Commits (rama `fix/search-failing-candidates`, en local): el mensaje de `initial_train_size`;
los candidatos sin datos del one-step-ahead; los trials que fallan en bayesian; el refit sin
series cortas; el aviso de `n_evaluated` con una serie; la documentación (nota de la versión,
skill, `llms-base.txt`); y el texto del error de los niveles sin puntos. Tests: 2499 en
`model_selection` (`tests_search`, `tests_split`, `tests_utils`), `tests_forecaster_recursive_multiseries`
y `utils/tests_utils`; y, tras el punto 3, `tests_validation` (463) y los backtesting de
`deep_learning`, `foundation`, `plot`, `ForecasterEquivalentDate` y `stats` (242).

### 14.4 Implementación y no previsto

- **Helpers nuevos** (`model_selection/_utils.py`): `_check_initial_train_size` (los tres
  casos; `window_size=None` para `ForecasterFoundation`, `allow_window_size` para
  `ForecasterStats`), `_train_test_split_one_step_ahead_search` (chequeo de la ventana,
  split y niveles con puntos; lo usan grid y bayesian), `_TrialEvaluationError`,
  `_skip_trial`, `_FailedTrialsLogFilter` y `_optimize_study` (las tres bayesian; también
  ignora el aviso de Optuna de las categorías, antes repetido), `_drop_series_too_short_refit`.
  En `utils`, `_series_too_short_multiseries`, la regla de `fit` compartida.
  `_warn_n_evaluated_search_multiseries` pasa a `_warn_n_evaluated_search` (`levels=None`
  para una serie). `add_skipped` guarda el número del trial (`candidate_id`) con el lock, y
  "Parameters skipped" dice "trials" en bayesian y los ordena por número.
- **Categoría del aviso del refit:** `suppress_warnings` silencia todas las categorías de
  skforecast (`MissingValuesWarning` incluida), así que el aviso es `RuntimeWarning`, como
  "Parameters skipped".
- **Errores de configuración en bayesian:** abortan `search_space(trial)`, la comprobación
  de sus claves y `set_params`/`set_lags`. Un nombre de parámetro que no existe aborta en
  bayesian y se salta en grid (todos los candidatos): se deja así (decisión 2).
- **Resultados vacíos con la columna `params` duplicada** (ya existía, 12.6): sigue; el test
  de bayesian multiserie con todos los trials fallidos compara las columnas sin duplicados.
- **Hallazgo (al roadmap como BUG-osa-initial-window-check):** con `OneStepAheadFold`,
  `check_one_step_ahead_input` comprueba la ventana del forecaster inicial aunque
  `lags_grid` no la use: `ForecasterRecursive(lags=70)` con `lags_grid=[3]` y 60
  observaciones de train aborta; con `TimeSeriesFold` funciona. Ya existía. Desde este PR
  la ventana se comprueba por candidato, así que bastaría con quitarla del chequeo inicial.
  **Hecho en este PR, en su propio commit (decisión del 2026-11-06):**
  `check_one_step_ahead_input` solo comprueba que `initial_train_size` sea menor que la
  longitud de los datos (`window_size=None` en `_check_initial_train_size`); la ventana
  queda para `_train_test_split_one_step_ahead_search`. Tests: el ejemplo en grid y en
  bayesian, y el test de `check_one_step_ahead_input` pasa los casos de la ventana a "sin
  error". La entrada de Fixed del one-step-ahead se amplía. Con `lags_grid=None` y una
  ventana demasiado grande, la búsqueda ya no aborta: salta los candidatos con el aviso,
  como con `TimeSeriesFold`.
- **Hallazgo (al roadmap como BUG-refit-directmultivariate-short):** la sección 4b de 14.1.

### 14.5 Revisión como usuario (notebook 09, 2026-11-06)

`repro/BUG-search-short-series_09_failing_candidates.ipynb`, un escenario por punto con los
datos del notebook 08 (`store_1`, `store_2` y `new_product` de 120 días).

- Bien: el valor recibido en `initial_train_size` (con la fecha y las observaciones), el caso
  de la ventana mayor que la serie, los trials por número y en orden, el aviso de una serie
  (causa y acción), y el refit (`series_names_in_` sin la serie y el aviso apunta a la
  línea del usuario).
- **Cambiado tras la revisión:** "No level of `levels` can be evaluated in the test set:
  none of them has more than `window_size` (35) observations in the training set, can be
  predicted after it and has observed values in the test set." mezclaba tres negaciones.
  Ahora: "No level of `levels` can be evaluated in the test set with this `window_size`
  (35). A level is evaluated if it has more than `window_size` observations in the training
  set, its last window can be used to predict, and it has observed values in the test set.
  Review `levels`, `initial_train_size` of `cv` or the span of the series."
- En el escenario 4 salen dos avisos sobre `new_product`: el de #1415 (no se predice en la
  búsqueda) y el del refit (el forecaster devuelto no la conoce). Dicen cosas distintas; se
  dejan los dos.
- Ya existía, fuera de este PR: `OneStepAheadValidationWarning` apunta a `_utils.py:794`
  (sale en `check_one_step_ahead_input`, antes del registro de avisos), y los
  `RuntimeWarning` ("Parameters skipped", el del refit) salen sin el panel de rich, que solo
  formatea las categorías de skforecast.
- Revisión final del código: la construcción de las líneas de "Parameters skipped" mezclaba
  cadenas y diccionarios (`isinstance`); ahora un prefijo con el identificador.
