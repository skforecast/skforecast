# Roadmap de skforecast 0.27.0

Fecha: 2026-10-18. Base: rama `0.26.x` (`__version__ = "0.26.0"`, publicada el 2026-10-08).
Fuentes: inventario de fase 1 (`work/phase1_open_work.md`, `work/phase1_product_context.md`),
7 lentes (`work/lens_*.md`), triaje (`work/triage.md`), 26 verificaciones (`work/verify/*.md`)
y scripts de reproducción (`repro/*.py`). Las rutas son relativas a la raíz del repositorio
salvo que se indique otra cosa.

Entorno de medida: Python 3.12.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1,
lightgbm 4.7.0, numba 0.68.0, scipy 1.18.1, optuna 5.0.0, torch 2.14.1+cpu,
chronos-forecasting 2.3.2, tabicl 2.2.0. Máquina de 4 núcleos compartida con otros agentes
(los tiempos son medianas de 3 repeticiones).

Esfuerzo: S (horas a 1 día, con tests y docs), M (2 a 5 días), L (más de una semana).

---

## 0. Revisión: lo que queremos hacer en 0.27.0

Revisión del 2026-10-18 sobre este informe. **Esta sección manda sobre la sección 1**, que
mete en 0.27.0 unos 40 elementos porque el análisis se hizo sin límite de tiempo. Las
secciones 1 a 7 quedan como referencia: evidencia, antes y después, y esbozos de cada arreglo.

> **MUY IMPORTANTE: BUG-search-short-series** (añadido el 2026-10-19). Las búsquedas
> multiserie (`grid_search`, `random_search` y `bayesian_search`, con `TimeSeriesFold` y
> `OneStepAheadFold`) evalúan cada candidato sobre las series y folds que puede predecir.
> Con distintos `lags`, una ventana grande deja fuera las series cortas y la búsqueda elige
> por una métrica calculada sobre puntos distintos, sin ningún aviso. Medido: el ranking se
> invierte, también con ventas reales de M5. Diseño propuesto (opción c2: predecir con los
> lags que faltan a NaN cuando el estimador los acepta, con escalado propio de cada serie),
> prototipo validado contra skforecast, opciones descartadas y mediciones pendientes en
> [`BUG-search-short-series.md`](BUG-search-short-series.md).

**Comprobado en local** (Python 3.14.3, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1,
numba 0.68.0, statsmodels 0.14.6): BUG-03, BUG-01 y BUG-stats-interval-ignored se
reproducen con sus scripts. QW-03 se confirma en el código: el `ValueError` de
`skforecast/model_selection/_utils.py:270-288` exige repetir `differentiation` en el cv,
pero `_validation.py:513-518` la sobrescribe después con la del forecaster. El resto de
cifras son las de la sesión en la nube y no se han vuelto a medir.

### Estado y prioridad (revisión del 2026-10-27)

**Hecho en `0.27.x`:** PR-1400 (compilación de Ets), #1412 (ISSUE-1357, IDs enteros),
#1415 (ISSUE-1406), #1416 (BUG-stats-interval-ignored), #1418 (ISSUE-1410), #1419 (BUG-09 a),
#1420 (QW-09, #1409), #1422 (#1411, mensaje cuando los NaN dejan sin muestras de
entrenamiento), #1387 (ISSUE-1386, PR externo: `plot_residuals` con `fig`) y #1423 (QW-03 y
DEP-cv-differentiation, 2026-10-20: el cv ya no exige repetir la `differentiation` del
forecaster; `differentiation` de los dos folds y `window_size` de `OneStepAheadFold` quedan
deprecados, sin efecto, con `FutureWarning` y eliminación en la 0.29.0; además corrige las
docstrings de los folds y quita `externally_fitted` de `OneStepAheadFold.split`) y #1424
(parte de QW-08: `metric='root_mean_squared_error'` por nombre en backtesting y búsquedas).
**Paneles con series que terminan y empiezan más tarde** (2026-10-21, sin elemento previo en el
roadmap; ver sección 6): #1427 (folds sin series en el backtesting multiserie) y #1430
(`initial_train_size` contra el span de las series y test set del one-step-ahead).
**BUG-08** hecho con #1425 (rama `fix/stats-freeze-params`, fusionado en `0.27.x` el 2026-10-20):
`_backtesting_stats` pasa `best_params_` del primer ajuste a las plantillas
`forecaster.estimators` (opción a). Medido en local (Python 3.14.3, numpy 2.5.3, pandas
2.3.3, scipy 1.18.1, numba 0.68.0, `n_jobs=1`): repro (800 puntos, m=7, 20 folds) Arima
automático de 15,7 s a 4,3 s (3,6x, 20 búsquedas a 1) y Ets de 3,2 s a 0,38 s (8,4x);
`grid_search_stats` con 2 candidatos, de 40 búsquedas a 2. h2o (`steps=12`, 7 folds, ventana
creciente): Arima de 5,0 s a 2,8 s (MAE 0,06190 a 0,05955) y Ets de 0,83 s a 0,14 s (MAE
0,06279 a 0,06130). `freeze_params=False` no cambia. PERF-02 se puede volver a medir.
Segundo commit: `Ets.best_params_` guarda la lambda de Box-Cox del ajuste (`lambda_auto=False`),
como `ets(y, model = fit)` de R y como `Arima`; antes Guerrero se repetía en cada fold con los
parámetros de suavizado congelados (h2o con `lambda_auto=True`: lambda de 0,34 a 0,01 entre
folds; MAE 0,0621 con lambda reestimada, 0,0643 congelada).
**BUG-03** hecho con #1431 (rama `fix/search-metric-direction`, fusionado en `0.27.x` el
2026-10-22): las búsquedas siguen la dirección de la métrica (decisión 6).
**Búsquedas con `lags=None`** (ISSUE-1355): #1433 (fusionado el 2026-10-26; sustituye a
PR-1388 y conserva su commit) y #1434 (2026-10-26: con un `None` en `lags_grid` y sin
`window_features`, `ValueError` antes de evaluar ningún candidato).
**ISSUE-1356** hecho con #1435 (2026-10-26): en `ForecasterRecursiveMultiSeries`, el orden de
`X_train_series_names_in_`, de las claves de los residuos y de las series en la salida de
`predict` con `levels=None` sigue siempre el orden de entrada (antes dependía de `encoding`).
**ISSUE-1354** hecho en el commit `6a5d933a4` (Joaquín, directo a `0.27.x` el 2026-10-27, sin
PR): `set_in_sample_residuals` con CatBoost y exógenas categóricas aplica la misma conversión
a enteros que `fit`, en los cuatro forecasters de ML. Sustituye a PR-1389.
**Predicciones NaN silenciosas con modelos lineales** (2026-10-27, sin elemento previo en el
roadmap): #1432. El atajo `np.dot` de los modelos lineales de scikit-learn devolvía NaN sin
error cuando un predictor tenía un hueco (por ejemplo en `exog`), y el NaN se propagaba por
los lags. Ahora `predict` da `ValueError` con el predictor, el paso, la fecha y la serie; con
`differentiation`, un último valor NaN de la serie (o de `last_window`) también da error, con
cualquier estimador. El backtesting sigue prediciendo (trabaja sobre una copia con
`_allow_nan_predictions`): en multiserie avisa con `MissingValuesWarning` del número de
predicciones NaN por serie que quedan fuera de las métricas; con una serie da un error claro
antes de las métricas. Va en "Before upgrading". Añade QW-12 (sección 2) y anota en
[`BUG-search-short-series.md`](BUG-search-short-series.md) su relación con ese bug.
**Pendiente antes de la release:**
- (BUG-08) Volver a ejecutar `docs/user_guides/comparing-forecasting-models.ipynb`: usa
  AutoARIMA y AutoETS (con `refit` forzado a `True`), así que sus métricas y tiempos cambian
  con #1425. No se ejecutó en el PR porque también corre modelos foundation, RNN y LightGBM.
- (ISSUE-1354) Cerrar PR-1389 en GitHub como sustituido por `6a5d933a4`, agradeciendo la
  contribución, y cerrar el issue #1354 si no se cerró solo. La nota de la versión de ese
  arreglo no enlaza el issue: añadir `([#1354](https://github.com/skforecast/skforecast/issues/1354))`.

**Orden recomendado para lo que queda:**

1. **BUG-search-short-series: aplazado (decisión del 2026-10-20)** hasta terminar los
   arreglos rápidos (S) de los puntos 2 a 8. Sigue siendo el resultado erróneo silencioso más
   grave que queda y el elemento más largo (M a L), así que conviene retomarlo pronto para que
   no retrase la release.
2. ~~**BUG-03**~~ **Hecho con #1431** (fusionado el 2026-10-22). La
   dirección sale de la métrica (decisión 6): atributo `greater_is_better` y convención de
   nombres solo para `sklearn.metrics`; si no se conoce, la dirección de la tarea, que ya
   minimiza las métricas de skforecast (todas de regresión). Las bayesianas ordenan por
   `study.direction` y avisan si `direction` contradice la métrica. Helpers
   `_metric_greater_is_better` (`skforecast/metrics/metrics.py`), `_get_search_direction` y
   `_check_study_direction` (`skforecast/model_selection/_utils.py`), pensados para que
   BUG-search-short-series no tenga que tocar la dirección. Además:
   `grid_search_stats`, `random_search_stats` y `grid_search_equivalent_date` convierten los
   nombres de métrica al principio, como las demás búsquedas (un nombre inválido da
   `ValueError` antes de evaluar candidatos), y `add_y_train_argument` devuelve tal cual una
   función ya procesada: antes, al volver a envolverla en el backtesting, quedaba marcada con
   `_needs_y_train = True` y se preparaba `y_train` en cada fold sin necesitarlo (búsqueda
   multiserie con 200 series: un 19% menos de tiempo con `OneStepAheadFold` y un 8% con
   `TimeSeriesFold`).
3. ~~**QW-03**~~ y ~~**DEP-cv-differentiation**~~ **Hechos con #1423** (fusionado el
   2026-10-20): elimina un `ValueError` en todo backtesting con `differentiation`, cuyo peor
   caso era una búsqueda que devuelve un DataFrame vacío, y depreca los argumentos del cv que
   no cambian los folds.
4. **BUG-04 y BUG-10** (S cada uno): foundation es foco, y los dos hacen que se prediga con
   otro modelo u otros datos sin aviso.
5. **QW-08 + PERF-03**, y después el bloque de salidas del backtesting: QW-04, BUG-02 + QW-10,
   anunciados juntos. De QW-08, el RMSE por nombre ya está hecho (#1424); quedan la
   validación de nombres de métrica repetidos en los backtesting y los nombres de pinball
   con el cuantil.
6. **QW-01 y BUG-01** (S, avisos).
7. **FE-05** (comparar varios modelos en un backtesting, nuevo): después de BUG-06 (punto de
   foundation con intervalos). QW-03, que era bloqueante porque un mismo `cv` no servía para
   forecasters con distinta `differentiation`, ya está hecho (#1423). QW-04 y QW-10 no bloquean, pero sus
   columnas deben coincidir.
8. **QW-11** (más features en `CalendarFeatures`, nuevo): S e independiente de todo lo
   demás; se puede hacer en cualquier momento. Alcance revisado con el análisis de la
   sección 2: `hour_of_week` y los días de fin de semana del mes, no los indicadores de
   inicio y fin de periodo.
9. Resto del alcance: QW-05, QW-02, BUG-07, ~~BUG-08~~ (hecho), PERF-02, PRs externos y QW-07. BUG-06
   no está aquí porque va antes de FE-05 (punto 7).

### Alcance

| Tema | Elementos | Notas |
|---|---|---|
| Resultados erróneos silenciosos | BUG-03, BUG-01, BUG-stats-interval-ignored, BUG-04, BUG-10, ISSUE-1406 | Todos S. BUG-01 solo como aviso silenciable: el backtesting sobre datos de entrenamiento es un uso documentado. Hechos (2026-10-19): ISSUE-1406 con #1415 (sustituye a PR-1407, cerrado) y BUG-stats-interval-ignored con #1416; DEP-stats-alpha descartado (decisión 5). Hecho (2026-10-22): BUG-03 con #1431. Hecho (2026-10-27, nuevo): predicciones NaN silenciosas con modelos lineales y huecos en los predictores, con #1432 |
| Backtesting probabilístico (titular de la release) | QW-03, QW-09, QW-04, QW-08, PERF-03, BUG-02, QW-10 | Es lo que se pide desde 2024 (#885, #902, #1253, #1409, D1119). Tres cambios de salida independientes que se fusionan por separado y se anuncian juntos en `releases.md`: QW-09 (`coverage` e `interval_width`, #1409) va primero porque no depende de QW-04; QW-04 (`y_true`) y QW-10 (`step`) después. PERF-03 (métricas en numpy: la búsqueda con `OneStepAheadFold` y 1000 series, 2,6 veces más rápida con Ridge y 1,4 con LightGBM) va junto a QW-08 o justo detrás. Hechos: QW-09 con #1420 y QW-03 con #1423 (2026-10-20) |
| Comparar modelos | FE-05 | M. Nuevo (2026-10-20). Una llamada para el backtesting de varios forecasters de cualquier familia sobre los mismos puntos. Sustituye el código que hoy repite la guía `comparing-forecasting-models.ipynb`. Diseño analizado y prototipado (sección 3). Requisitos: QW-03 (hecho, #1423) y BUG-06 |
| Feature engineering | QW-11 | S. Nuevo (2026-10-20). `hour_of_week` y el número de días de fin de semana del mes en `CalendarFeatures`, siempre opcionales, más un aviso y una guía de codificación. Medido en 18 series: `hour_of_week` reduce el MAE una mediana del 24% con Ridge en series horarias. Los festivos, que son lo que más aporta en series diarias, quedan como feature aparte (FE-06) |
| Foundation para producción | QW-05, BUG-06, QW-02 | QW-02 pendiente de decisión |
| **Búsquedas multiserie con series cortas (MUY IMPORTANTE)** | BUG-search-short-series, QW-12 | M a L. Resultado erróneo silencioso en un uso habitual de los modelos globales. Diseño en [`BUG-search-short-series.md`](BUG-search-short-series.md); antes o durante la implementación, las mediciones de su sección 8.2 (sobre todo los intervalos con residuos out-of-sample). No bloquea QW-09, pero cambia qué puntos entran en sus métricas |
| Paneles multiserie | QW-01, BUG-07, ISSUE-1410 | Hechos: ISSUE-1357 (IDs enteros) con #1412, que sustituye a PR-1408 (cerrado), e ISSUE-1410 con #1418 (2026-10-19); folds sin series con #1427 y `initial_train_size` contra el span con #1430 (2026-10-21, sección 6) |
| Stats: corrección y rendimiento | BUG-08, PERF-02, PR-1400 | PR-1400 ya está fusionado en `0.27.x`. BUG-08 hecho con #1425 (fusionado el 2026-10-20). Volver a medir PERF-02 después de BUG-08 |
| PRs externos | PR-1402, PR-1387, PR-1388, PR-1389 | Hecho: PR-1387 fusionado en `0.27.x` (2026-10-20); PR-1388 sustituido por #1433, que conserva su commit (fusionado el 2026-10-26); PR-1389 sustituido por el commit `6a5d933a4` (ISSUE-1354, 2026-10-27), queda cerrarlo en GitHub. PR-1402 sigue abierto y apunta a `0.26.x`, no revisado línea a línea |
| Pequeños | QW-07, DEP-python310 (anunciar la retirada en 0.28), DEP-arima-dictstyle | |

El orden dentro de cada tema sigue las dependencias de la sección 1. En particular:

- ISSUE-1406 va antes que QW-09 y QW-04 (hecho con #1415): toca el mismo bloque de
  `_backtesting_forecaster_multiseries` y decide qué series entran en los agregados. Las
  series que no se predicen en ningún fold tienen métricas NaN y quedan fuera de los
  agregados; `coverage` e `interval_width` deben seguir la misma regla.
- BUG-stats-interval-ignored va antes que QW-09 (hecho con #1416). Corrección (2026-10-19): `backtesting_stats`
  no estaba afectado (allí `alpha=None` por defecto y `interval` se respetaba); el bug solo
  afectaba a `ForecasterStats.predict_interval` llamado directamente. Lo que aporta a QW-09 es
  que, tras el arreglo, `backtesting_stats` traduce `alpha` a `interval` al principio, así que
  el intervalo nominal está siempre en `interval`.
- ISSUE-1410 conviene antes (mismo fichero), pero no bloquea (hecho con #1418).
- BUG-02 va antes que QW-10.
- PERF-03 va junto a QW-08 o justo detrás: los dos tocan `_get_metric`
  (`skforecast/metrics/metrics.py`), así que no conviene tener dos PRs abiertas sobre la
  misma función. `root_mean_squared_error` ya se acepta por nombre (#1424), así que PERF-03
  tiene que incluir su versión en numpy.

Orden de QW-09: ~~ISSUE-1406 (#1415) → BUG-stats-interval-ignored (#1416) → ISSUE-1410
(#1418)~~ (hechos el 2026-10-19) → ~~QW-09 (#1420)~~ (hecho el 2026-10-20) → **QW-08
(siguiente)** (validar también los nombres reservados `coverage` e `interval_width`) → QW-04
→ BUG-02 + QW-10 → FE-05 → FE-04 → foundation con `interval` (tras BUG-06).

### Decisiones tomadas (2026-10-18)

1. **QW-09 separado de QW-04 y de QW-10.** #1409 no necesita la columna `y_true`: el valor real
   ya está en el cálculo de métricas (`_validation.py:669`, `_utils.py:1160-1166`).
2. **`coverage` e `interval_width` sin flag:** se añaden siempre que `interval` tenga dos cotas.
3. **Aviso de infracobertura descartado definitivamente** (sección 7).
4. **QW-10 (columna `step`) se hará:** es barata y no depende de QW-04 ni de QW-09.
5. **DEP-stats-alpha descartado (2026-10-19).** `alpha` se mantiene como alias de `interval`
   en `ForecasterStats.predict_interval` y `backtesting_stats`: es el nombre habitual para
   quien viene de estadística (statsmodels, pmdarima) y los estimadores `Arima` y `Sarimax`
   también lo aceptan. El bug lo causaba su valor por defecto (0,05), no el alias. En su lugar,
   pasar `alpha` e `interval` a la vez da `ValueError`, como ya hace `Arima.predict_interval`
   con `alpha` y `level`, y desaparece la regla de precedencia.
6. **BUG-03: dirección de la métrica (2026-10-21).** La dirección de las búsquedas sale de la
   primera métrica, no de la tarea: el atributo `greater_is_better` de la función (el nombre
   de `make_scorer` de sklearn y de `make_forecasting_scorer` de sktime) y, solo para
   funciones de `sklearn.metrics`, su convención de nombres (`*_score` se maximiza;
   `*_error`, `*_loss` y `*_deviance` se minimizan; `matthews_corrcoef` se maximiza). Una
   función desconocida usa la dirección de la tarea, sin aviso (como Optuna y sktime). Con
   `direction` en `kwargs_create_study`, la tabla y `return_best` siguen a `study.direction`
   y se avisa si contradice una métrica de dirección conocida. Descartados: deducirla solo
   del nombre (la propuesta inicial: en forecasting muchas reglas que se minimizan se llaman
   "score", como Winkler o el interval score, y no cubre `partial`, lambdas ni la pinball) y
   un argumento nuevo en las 10 búsquedas (duplicaría `direction` en las bayesianas).

### Decisiones pendientes (cerrarlas antes de implementar)

1. **BUG-02** (`fold_stride < steps`): quedarse con la última predicción de cada fecha es
   deliberado. Propuesta: parámetro nuevo en `TimeSeriesFold` con el comportamiento actual
   por defecto y aviso cuando `fold_stride < steps`. Cambiar el defecto, solo anunciado y en
   otra versión.
2. **QW-04** (columna `y_true` sin flag): cambia la salida del backtesting y obliga a
   actualizar unas 160 aserciones. Propuesta: hacerlo sin flag, como `fold` en 0.18.0, y
   anunciarlo en las notas de la release.
3. **QW-02** (aviso en foundation cuando se ajusta con exog y `predict` no la recibe):
   invierte un silencio documentado. Propuesta: hacerlo; en ML el mismo olvido ya da error.
4. **HANDOFF-T2/T3/T4** (`dev/handoff_utils_pendiente.md`): pendientes de decisión de Joaquín.
5. **PR-1341** (TiRex-2): aceptarlo o no; la revisión es grande (L).
6. **BUG-search-short-series**: confirmar la opción c2 (o pasar a c1) con las mediciones de
   la sección 8.2 de [`BUG-search-short-series.md`](BUG-search-short-series.md). También
   decidir si `predict` admite una `last_window` más corta que la ventana y si `fit` acepta
   series con `window_size` valores o menos para ajustar su transformer (requisito medido).
   Cambia métricas y rankings existentes con estimadores que aceptan NaN: necesita una línea
   en "Before upgrading".
7. **FE-05** (comparar modelos): diseño recomendado en "FE-05: análisis de diseño"
   (sección 3). Falta decidir: el nombre (propuesta: `compare_forecasters` en
   `model_selection`); si las métricas se calculan por defecto sobre los puntos comunes
   (propuesta: sí, con aviso y `common_points=False` como alternativa); y el alcance de la
   v1 (propuesta: una serie con todas las familias, y multiserie con modelos globales y
   foundation; los modelos locales por serie, en la v2). Entrada como dict
   `{nombre: forecaster}`, porque con una lista no hay nombres únicos para tres
   `ForecasterRecursive` con estimadores distintos.
8. **QW-11** (`CalendarFeatures`): qué features entran. Propuesta en la sección 2, con el
   análisis medido. Las nuevas no entran en la lista por defecto (`features=None`), para que
   la salida de `CalendarFeatures()` no cambie. Los indicadores `month_start`, `month_end`,
   `quarter_start`, etc. solo como comodidad (paridad con feature-engine y mlforecast), no
   por precisión: medido, no aportan en ninguna de las 8 series diarias.
9. **FE-06** (festivos en `CalendarFeatures` con el paquete `holidays` como dependencia
   opcional): si entra en 0.27.0 o en 0.28. Es lo que más mejora las series diarias medidas,
   pero solo hay 2 series con festivos y el alcance es M.

### Para después de 0.27.0

FE-02, FE-03, FE-04, PERF-01 (gana 3,4 veces con una serie y nada con 50), QW-06, BUG-09 (b)
(la parte (a) se adelantó a 0.27.0, ver la tabla de bugs), BUG-11, GAP-foundation-backtesting-interval (depende de BUG-06), ISSUE-1131 y todo lo de la
sección 5.

### Huecos del análisis a cubrir al implementar

- **Foundation**: solo se ejecutaron Chronos-2 y TabICL con pesos reales. BUG-04, QW-05 y
  BUG-06 se extienden al resto de adaptadores leyendo el código: comprobarlos con cada
  adaptador al arreglarlos.
- **ML**: `ForecasterDirectMultiVariate` apenas se revisó y las exógenas categóricas no se
  revisaron.
- **Repositorio**: antes de publicar 0.27.0, decidir si `dev/roadmap_0.27.0/` (sobre todo
  `work/`) se queda, porque `0.27.x` se fusiona en `main`.

---

## 1. Propuesta para 0.27.0

**Temas**

1. **Backtesting en el que se puede confiar.** Hoy hay cuatro maneras de obtener en silencio
   una métrica o un modelo erróneos en el flujo central (evaluar in-sample sin aviso,
   `fold_stride < steps`, métricas "mayor es mejor" en las búsquedas, agregados multiserie
   sin una serie), y evaluar intervalos exige reconstruir a mano valor real, paso del
   horizonte y cobertura. La 0.27 corrige lo primero e integra lo segundo con piezas que ya
   existen (`calculate_coverage`, columna `fold`).
2. **Paneles multiserie reales sin fricción:** series que terminan antes (desaparecen de
   `predict`), que empiezan tarde, con IDs enteros, en formato largo con `set_index`, con
   `differentiation`. Tres PR externos ya cubren parte.
3. **Foundation listo para producción:** el modelo ajustado no se pierde, el punto es el
   mismo en `predict` y en los intervalos, las covariables se usan como dice la
   documentación, el fichero guardado pesa KB y el backtesting es hasta 3,4 veces más rápido.

**Orden propuesto** (valor para el usuario frente a esfuerzo, sin recorte por calendario)

| Bloque | Elementos, en orden |
|---|---|
| A. Resultados silenciosamente erróneos, esfuerzo S | BUG-03 (#1431, hecho), BUG-01, BUG-02, ISSUE-1406 (#1415, hecho), BUG-04, BUG-10, BUG-stats-interval-ignored (#1416, hecho) |
| B. Quick wins diarios y PR listos | QW-03 y DEP-cv-differentiation (#1423, hechos), QW-01, QW-09, QW-04, QW-10 (tres cambios de salida, un solo anuncio), PR-1389, PR-1388 (#1433, hecho), ISSUE-1410 (#1418, hecho), QW-02, QW-05, ISSUE-1385, PR-1400, PR-1402, PR-1387 (#1387, hecho) |
| C. Corrección de más alcance, STATS y quick wins secundarios | BUG-06, BUG-08, PERF-02, QW-07 (+ DOC-skills-percentiles-stale), BUG-07, QW-08, PERF-03, QW-06, BUG-11, BUG-09 ((a) hecho con #1419; queda (b)) |
| D. Features y gaps | FE-02, PERF-01, GAP-foundation-backtesting-interval, FE-04, FE-03 |
| E. Mantenimiento y decisiones | DEP-python310 (anunciar), DEP-arima-dictstyle, HANDOFF-T2/T3/T4, PR-1341 (TiRex-2), ISSUE-1131 (alcance) |

**Por qué esta selección y este orden**

- Primero, todo resultado silenciosamente erróneo en un flujo común: los siete elementos del
  bloque A están reproducidos, son S y casi todos se arreglan en menos de 20 líneas.
  BUG-03 reentrena con el peor modelo con `return_best=True` (valor por defecto); BUG-01 da
  métricas 2,5 veces optimistas sin aviso; BUG-02 hace que las búsquedas elijan por el error
  a 1 paso; BUG-04 y BUG-10 hacen que foundation prediga con otro modelo u otros datos.
- Después, la fricción semanal con arreglos mínimos: QW-03 son 2 líneas y quita un
  `ValueError` (o un DataFrame vacío) de todo backtesting con `differentiation` (hecho con
  #1423, junto con la deprecación DEP-cv-differentiation);
  QW-09, QW-04 y QW-10 son el quick win de calibración que se pide desde 2024 (#885, #902,
  #1253, #1409, D1119) y solo reutilizan funciones existentes.
- Dependencias que fijan el orden interno: BUG-10 antes de FE-02; BUG-06 antes de aceptar
  `interval` en `backtesting_foundation`; ISSUE-1406 y BUG-stats-interval-ignored antes de
  QW-09 (hechos); QW-04, QW-09, ISSUE-1406 e ISSUE-1410 antes de FE-04; BUG-02 antes de QW-10 (columna `step`);
  BUG-09 (a) no dependía de BUG-02 y se adelantó (#1419, 2026-10-19): al hacer BUG-02, la
  deduplicación de `_backtesting_stats` debe conservar la clave (fecha, `estimator_id`) con
  varios estimadores; BUG-04 y QW-05 comparten la declaración del atributo de backend.
- STATS entra solo por corrección y rendimiento medidos (intervalo ignorado; BUG-08, de 5 a
  15 veces más rápido; PERF-02, de 3 a 70 veces con carga); BUG-09 va al final por ser nicho.
  Deep learning queda fuera por diseño; de los módulos restantes solo entran PR-1402 y PR-1387.
- Las features van detrás porque aportan menos por unidad de esfuerzo o tienen más riesgo:
  FE-03 mejora el MAE un 2% y solo con árboles; FE-04 no tiene peticiones explícitas, pero
  corrige un sesgo real al comparar con el baseline.
- Cambios de comportamiento que hay que anunciar en `releases.md`: columnas nuevas `y_true` y
  `step` y métricas `coverage`/`interval_width` sin flag (precedente: `fold` en 0.18.0); `alpha`
  deprecado en stats; resultados distintos con `freeze_params=True` (Fixed); aviso nuevo en
  foundation sin exog futura; `save_forecaster` deja de guardar los pesos foundation; columna
  `mean_pinball_loss_q_{alpha}`.

---

## 2. Quick wins

| Item | Área | Valor para el usuario | A cuántos usuarios afecta | Esfuerzo | Evidencia | Esbozo del cambio / API |
|---|---|---|---|---|---|---|
| **QW-03** Heredar `differentiation` en `TimeSeriesFold` y `OneStepAheadFold` | model_selection | No hay que repetir `differentiation` en el cv; deja de fallar el código generado con la propia referencia IA; desaparece el peor caso (grid devuelve DataFrame vacío y forecaster sin ajustar) | Flujo habitual: todo backtesting o búsqueda con `differentiation` (recomendada con árboles), una o varias series | S (2 líneas, docstrings, 2 tests) | `skforecast/model_selection/_utils.py:274-288,602-620` lanza el error y `_validation.py:513-518,1286-1291` sobrescribe después `cv.differentiation` con `forecaster.differentiation_max`; `tools/ai/llms-base.txt:340,370` ya dice "set automatically"; prototipo: predicciones idénticas (`DataFrame.equals`), 220 tests pasan | **Hecho: #1423, fusionado en `0.27.x` el 2026-10-20.** La propuesta inicial (comprobar solo si `cv.differentiation is not None` y mantener el `ValueError` si el valor es distinto) se sustituyó por DEP-cv-differentiation: el cv ya no comprueba `differentiation`, porque el argumento no cambia los folds. Quitado el comentario "must match" de 5 celdas en 3 guías |
| **DEP-cv-differentiation** Deprecar `differentiation` en `TimeSeriesFold` y `OneStepAheadFold`, y `window_size` en `OneStepAheadFold` (añadido el 2026-10-20, misma PR que QW-03) | model_selection | Menos argumentos que no hacen nada: `differentiation` no cambia ningún fold (solo restaba observaciones en la salida de `verbose`) y llevaba a error, porque `window_size=max_lag, differentiation=d` da una última ventana más corta que la del backtesting; `window_size` no cambia el fold de `OneStepAheadFold` | Quien pasa estos argumentos al cv (las guías lo hacían en 5 celdas, y el código generado por IA también) | S | `_split.py`: `differentiation` solo en `_print_info`; el `window_size` del forecaster ya incluye la diferenciación (`_forecaster_recursive.py:415`); ningún código del paquete fija `window_size` ni llama a `split()` en `OneStepAheadFold`. Predicciones idénticas antes y después (backtesting de una serie, directo, multiserie con dict y búsquedas con los dos folds) | **Hecho: #1423, fusionado en `0.27.x` el 2026-10-20.** `FutureWarning` en `__init__` y en `set_params` cuando el valor no es `None`; el valor se ignora. Sin la comprobación de conflicto de QW-03 (`_check_cv_differentiation`, eliminada) ni la línea de differentiation en `verbose`. Los tests ya no pasan estos argumentos salvo los de la propia deprecación. Los parámetros deprecados se declaran en el atributo de clase `_deprecated_params` y no se validan, porque se ignoran. **Pendiente para la 0.29.0:** quitar los parámetros, sus atributos, `_deprecated_params` y `_warn_deprecated_params`, y los tests de la deprecación (`test_basefold.py`, `test_timeseriesfold.py`, `test_onestepaheadfold.py` y los de grid y backtesting que comprueban que se ignoran) |
| **QW-01** Aviso cuando `predict()` con `levels=None` excluye series | recursive (multiseries) | Las series cuya última observación no llega a la fecha máxima dejan de desaparecer en silencio del forecast de producción | Flujo habitual en paneles de retail o demanda construidos desde formato largo o con NaN al final | S (~15 líneas) | `skforecast/utils/utils.py:4680` (`if input_levels_is_list and ...`); medido: 6 series entrenadas, 4 predichas, 0 avisos; store_sales 50 entrenadas, 47 predichas. Misma familia que #1406 / PR-1407 (aviso en backtesting), pero en `predict` y con código distinto | `IgnoredArgumentWarning` también con `levels=None`, con lista ordenada y truncada a 10, la fecha máxima y cómo predecirlas; un solo aviso en `predict_interval(method='bootstrapping')` |
| **QW-09** Cobertura y anchura del intervalo en las métricas de backtesting (#1409) | model_selection / metrics | `coverage` e `interval_width` en la tabla de métricas cuando `interval` tiene dos cotas, por nivel y en los agregados (`average`, `weighted_average`, `pooling`), sin flag | Flujo habitual: todo backtesting probabilístico de ML y stats (las búsquedas pasan `interval=None`, `_search.py:454,1454,2778,3033`, y no cambian) | S (~1 día con tests y docs) | https://github.com/skforecast/skforecast/issues/1409; las métricas solo usan `pred` (`_validation.py:688`, `_utils.py:1067`); `skforecast/metrics/metrics.py:459` (`calculate_coverage`, no ignora NaN); medido en bike_sharing (80% nominal): cobertura de LGBM con residuos in-sample 0,53 (bootstrapping) y 0,26 (conformal) frente a 0,80 y 0,62 con residuos out-of-sample, sin ninguna señal (`work/verify/FE-01-value.md`) | Helper privado que calcula las dos métricas si la salida tiene `lower_bound` y `upper_bound`, sobre las mismas filas que el resto de métricas (tras `keep='last'`) y sin las filas con `y_true` NaN (en multiserie solo `pred` se pone a NaN, `_validation.py:1499`); columnas detrás de las métricas del usuario. En multiserie, pasar las cotas a `_calculate_metrics_backtesting_multiseries` (hoy solo `[['pred']]`, `_validation.py:1510`); las series nunca predichas dan NaN y quedan fuera de los agregados, como el resto de métricas desde #1415. `_backtesting_stats` ya solo recibe `interval` (`alpha` se quitó de las funciones privadas el 2026-10-19). Cubre `backtesting_forecaster`, `_multiseries` (incluido `ForecasterRnn`) y `_stats` (uno o varios estimadores); foundation cuando acepte `interval`. `ValueError` si una métrica del usuario se llama `coverage` o `interval_width`. Sin aviso de infracobertura (descartado, sección 7) |
| **QW-10** Columna `step` en las predicciones de backtesting (D1119) | model_selection | Paso del horizonte de cada predicción, para analizar error y cobertura por horizonte sin reconstruirlo a mano | Flujo habitual: todo informe por horizonte | S | https://github.com/skforecast/skforecast/discussions/1119; `_validation.py:660,1471,2140,2574` (solo `fold`); `cumcount()+1` por fold falla con `gap>0` (con `gap=3` los pasos reales son 4..13, `work/verify/FE-01-risk.md`) | Columna `step` (de `gap + 1` a `gap + steps`) junto a `fold` en los cuatro backtesting; receta documentada por `step` y `level`. Después de BUG-02; independiente de QW-04 y QW-09. Cambio de salida anunciado |
| **QW-04** Columna `y_true` en las predicciones de backtesting | model_selection | Valor real junto a la predicción en los cuatro backtesting: residuos, gráficos y métricas propias sin merge (que hoy cambia según dict, ancho o largo, y falla con el snippet habitual en DataFrame ancho) | Flujo habitual: casi todo usuario que inspecciona o grafica el backtesting | S de código (~40 líneas); ~1 día con la actualización mecánica de ~160 aserciones | `y_true` se calcula y se descarta en `_validation.py:669,2167,2180,2681` y `_utils.py:1149-1166`; coste medido 0,1 a 0,4% del tiempo (20 a 97 ms sobre 9 a 36 s); 21 usos de `data.loc[pred.index, col]` en 8 guías | Columna `y_true` justo antes de `pred`, sin flag. Rompe código que compara el DataFrame entero o indexa por posición. En multiserie ML, poner a NaN todas las columnas de predicción sin observación (hoy solo `pred`: la cobertura sale 0,63 frente a 0,73) |
| **QW-02** Aviso en foundation si se ajustó con exog y `predict` no la recibe | foundation | Hoy el efecto de promociones o festivos desaparece sin señal con Chronos-2 (adaptador por defecto), TimesFM 3.0 y TS-ICL; en ML el mismo olvido da `ValueError` | Flujo habitual de producción con foundation y exógenas conocidas a futuro (no afecta al backtesting) | S (~20 líneas) | `skforecast/foundation/_foundation_model.py:973-993` (solo avisa si `supports_past_only_covariates=False`); `_adapters.py:170,1154,4028`; medido con Chronos-2 small: días con promo 10,08/10,03 sin exog frente a 15,08/15,27 con exog | `MissingExogWarning` (ya existe) en `FoundationModel._check_exog_columns` cuando no hay exog futura y las columnas se usan solo como pasadas. Requiere visto bueno: invierte un silencio documentado (`skills/foundation-forecasting/references/adapter-parameters.md:226`) y hay que reescribir 6 parametrizaciones de `test_check_exog_columns.py:182-201` |
| **QW-05** `save_forecaster` sin los pesos foundation cargados tras `predict` | foundation / utils | El fichero guardado tras validar pasa de cientos de MB a KB, no queda atado a torch ni al dispositivo y deja de fallar con TimesFM 2.5 | Flujo habitual de paso a producción, cualquier adaptador | S (~30 líneas, 2 a 4 h) | Medido con chronos-2-small: 0,01 MB antes de `predict` y 111,81 MB después; carga 4,25 s frente a 0,33 s; TimesFM 2.5: `PicklingError` por clase local (`_adapters.py:970-975`); no hay `__getstate__` en `skforecast/foundation/` | `__getstate__` en `_AdapterBase` que vacía el backend cargado de forma perezosa (flag en los 9 `_load_*`) y las cachés `_forecast_obj`/`_resolved_device`, conservando el objeto que pasó el usuario (BUG-04). Documentar que el primer `predict` tras cargar recarga los pesos |
| **QW-07** Mensajes de error accionables en cuatro puntos | utils / model_selection | Frecuencia inferida, intervalos en percentiles, residuos ausentes en multiserie y función equivocada para el tipo de forecaster pasan a decir cómo resolverse (hoy grid/random con la familia equivocada devuelven un DataFrame vacío sin excepción) | Usuarios nuevos, quien migra desde 0.23/0.24 y quien pasa de ML a foundation; una vez por usuario, muy frecuente | S (+142/-3 en el prototipo) | `utils.py:2046-2050`, `:1222-1229`, `:1875-1884,1936-1947`; `_search.py:474,519`; `_validation.py:866-871,1695-1700,2318-2323,2825-2830`; prototipo: 615 + 365 tests pasan sin cambios. (b) amplía DOC-skills-percentiles-stale | (a) `pd.infer_freq` y receta `.asfreq`; (b) frase de migración si el límite está en (1, 100]; (c) un dict con todos los valores None cuenta como vacío y reutiliza el mensaje accionable; (d) `check_forecaster_type_model_selection` al inicio de cada backtesting y búsqueda, nombrando las funciones correctas. Corregir `skills/troubleshooting-common-errors/SKILL.md:72,184` |
| **QW-08** RMSE como string y nombres de métrica únicos | metrics | `metric='root_mean_squared_error'` sin importar sklearn; nombres de pinball con el cuantil; error claro en backtesting con dos métricas del mismo nombre | RMSE: flujo habitual (ahorro diario pequeño); pinball con varios cuantiles: nicho | S (+7/-1 y validación) | `skforecast/metrics/metrics.py:45-62` (solo la variante escalada, `:52`) y `:555-558`; medido: dos pinball dan columnas duplicadas, y en multiserie `TypeError` (`_utils.py:1247`); las búsquedas ya validan nombres únicos (`_search.py:394-400`) | Añadir RMSE a `allowed_metrics`; validar unicidad en los backtesting; `__name__ = f"mean_pinball_loss_q_{alpha}"` (Changed: renombra una columna). Completar los docstrings de `metric` (MedAE, sMAPE) |
| **QW-06** Alinear `exog` por índice en `predict` de Recursive y Direct | recursive / direct | Pasar un único DataFrame de exog (pasado y futuro) sin recortarlo, como ya hacen MultiSeries y Foundation | Flujo habitual de producción con calendario o meteo; coherencia entre forecasters | S (+36/-1 en el prototipo) | `utils.py:1656` (`align_by_index` solo en MultiSeries) y `:1738-1745`; prototipo: mismos resultados que el recorte manual, 871 tests pasan (1 a adaptar); con RangeIndex reseteado daría `[-1.331, ...]` frente a `[1.005, ...]` en silencio | Solo con `DatetimeIndex`: si el índice contiene el primer paso esperado, `exog.iloc[pos:]` antes del chequeo de NaN y de `transform_dataframe`; también Classifier y DirectMultiVariate; RangeIndex mantiene el chequeo estricto; `fit` no cambia. Toca el mismo camino de `predict` que HANDOFF-DOUBLE-WARN (sección 5): resolver el doble aviso a la vez si sale barato |
| **QW-11** Más features en `CalendarFeatures` (añadido el 2026-10-20, revisado con el análisis de abajo) | preprocessing | La hora de la semana (día de la semana por hora) en series horarias y los días de fin de semana del mes en series mensuales, que hoy hay que construir a mano como exógenas; un aviso cuando una feature anual se usa con menos de un año de datos; y una guía de codificación por tipo de modelo | Series horarias con modelos lineales (el mayor efecto medido) y series mensuales con modelos lineales; con árboles el efecto es pequeño | S (cada feature es una entrada en `datetime_attrs` y en `_FEATURE_KNOWN_CATEGORIES`; tests y docs) | `skforecast/preprocessing/_calendar.py:20-30,193-205` (11 features: `year`, `month`, `week`, `day_of_week`, `day_of_month`, `day_of_year`, `weekend`, `hour`, `minute`, `second`, `quarter`); análisis medido en 18 series (`repro/QW-11-calendar-analysis.py`, resultados en `.csv`) y revisión de 7 librerías, ver "QW-11: análisis" debajo. Sin peticiones en GitHub | Añadir, sin cambiar la lista por defecto: `hour_of_week` (0 a 167, codificable con 168) y `weekend_days_in_month` (8 a 10, sin codificar; además, valorar el conteo de cada día de la semana en el mes, 7 columnas). Opcional por paridad: `is_month_start`, `is_month_end` y `days_in_month`. No añadir `is_quarter_*` ni `is_year_*` (redundantes con `month` y `day_of_year`), ni Fourier con K armónicos (la codificación spline ya existente rinde igual). Aviso en `fit` si una feature anual (`month`, `week`, `day_of_year`, `quarter`) se usa con menos de un año de datos. Guía: one-hot o spline con modelos lineales, enteros (`encoding=None`) con árboles |
| **QW-12** Número de predicciones evaluadas por serie en las métricas del backtesting multiserie (añadido el 2026-10-27, a hacer con BUG-search-short-series) | model_selection | Señal robusta de que una serie o un candidato se evaluó con menos puntos de los predichos. Hoy la única señal es un aviso: `suppress_warnings=True` lo silencia y grid y random search solo lo muestran para el primer candidato | Backtesting y búsquedas multiserie con series cortas, huecos en las series o, desde #1432, modelos lineales con huecos en `exog` | S | #1432: con `suppress_warnings=True`, 10 predicciones NaN y 0 avisos; `_evaluate_grid_hyperparameters_multiseries` (`_search.py`) silencia los avisos tras el primer candidato (BUG-search-short-series, sección 2.2); `_calculate_metrics_backtesting_multiseries` (`model_selection/_utils.py`) ya cuenta las predicciones por nivel para `weighted_average`, pero no las devuelve | Columna (por ejemplo `n_evaluated`) en la tabla de métricas con los puntos que entran en las métricas de cada nivel (predicción y valor real no NaN), y la suma en los agregados; en las búsquedas, la del agregado por candidato. Hacerlo junto a BUG-search-short-series: c2 cambia qué puntos entran y la columna es la que comprueba que todos los candidatos se evalúan sobre los mismos puntos (secciones 2.2 y 7.1 de su documento). Cambio de salida anunciado, como QW-09; nombre reservado frente a las métricas del usuario, como `coverage` |

### QW-03: antes y después

```python
# Antes (0.26.0)
forecaster = ForecasterRecursive(LGBMRegressor(), lags=24, differentiation=1)
cv = TimeSeriesFold(steps=24, initial_train_size=n_train)
backtesting_forecaster(forecaster, y=y, cv=cv, metric="mean_absolute_error")
# ValueError: The differentiation included in the forecaster (1) differs from ...
# grid_search_forecaster con el mismo cv: DataFrame vacío y forecaster sin ajustar
cv = TimeSeriesFold(steps=24, initial_train_size=n_train, differentiation=1)  # hay que repetirlo
```

```python
# Después
cv = TimeSeriesFold(steps=24, initial_train_size=n_train)  # se hereda del forecaster
backtesting_forecaster(forecaster, y=y, cv=cv, metric="mean_absolute_error")
# Mismo resultado que con differentiation=1 explícito; un valor explícito (aunque sea distinto)
# da FutureWarning y se ignora (DEP-cv-differentiation)
```

### DEP-cv-differentiation: antes y después

```python
# Antes (0.26.0)
cv = TimeSeriesFold(steps=24, initial_train_size=n_train, window_size=24, differentiation=1)
cv.split(X=y, as_pandas=True)  # última ventana de 24, aunque el backtesting usa 25
```

```python
# Después
cv = TimeSeriesFold(steps=24, initial_train_size=n_train, window_size=24, differentiation=1)
# FutureWarning: `differentiation` is deprecated since version 0.27.0 and has no effect on the
# folds of `TimeSeriesFold`: the `window_size` of the forecaster already includes its
# differentiation order. It will be removed in version 0.29.0.
cv = TimeSeriesFold(steps=24, initial_train_size=n_train, window_size=forecaster.window_size)
```

### QW-01: antes y después

```python
# Antes
forecaster.fit(series=series_dict)   # 2 artículos sin registro el último día
p = forecaster.predict(steps=7)        # sin ningún aviso
faltan = set(forecaster.series_names_in_) - set(p["level"])  # hay que saber que esto pasa
# {'item_3', 'item_5'}: para predecirlas hay que llamar aparte con levels=[...]
```

```python
# Después
p = forecaster.predict(steps=7)
# IgnoredArgumentWarning: Only series whose last window ends at the same index can be
# predicted together. 2 series do not reach the maximum index, '2024-07-18 00:00:00',
# and are excluded from prediction: ['item_3', 'item_5']. To predict them, pass their
# names in `levels` (their predictions start after their own last observation) or
# provide `last_window`.
# Se silencia con suppress_warnings=True o pasando levels.
```

### QW-09: antes y después

```python
# Antes
metric, preds = backtesting_forecaster_multiseries(
    forecaster, series=series, cv=cv, metric="mean_absolute_error", interval=[0.1, 0.9]
)
# metric: levels, mean_absolute_error (las cotas del intervalo no se evalúan)
y_long = series.stack().rename("y_true")                         # solo DataFrame ancho
y_long.index.names = ["date", "level"]
preds = preds.rename_axis("date").reset_index().merge(y_long.reset_index(), on=["date", "level"])
preds = preds.dropna(subset=["y_true"])  # si no, calculate_coverage cuenta NaN como no cubierto
# ... bucle por level con calculate_coverage y (upper_bound - lower_bound).mean()
# Ninguna señal de que la cobertura in-sample con LightGBM sea 0,53 frente al 0,80 nominal
```

```python
# Después
metric, preds = backtesting_forecaster_multiseries(
    forecaster, series=series, cv=cv, metric="mean_absolute_error", interval=[0.1, 0.9]
)
# metric: levels, mean_absolute_error, coverage, interval_width
#         (una fila por nivel y average, weighted_average, pooling)
# Sin `interval`, o con cuantiles, bootstrapping o una distribución, la tabla no cambia.
```

### QW-10: antes y después

```python
# Antes
preds["step"] = preds.groupby(["fold", "level"]).cumcount() + 1   # incorrecto si gap > 0
```

```python
# Después
# preds: level, fold, step, pred, lower_bound, upper_bound   (+ y_true con QW-04)
by_step = preds.groupby(["level", "step"]).apply(lambda g: pd.Series({
    "mae": mean_absolute_error(g.y_true, g.pred),
    "coverage": calculate_coverage(g.y_true, g.lower_bound, g.upper_bound),
}), include_groups=False)   # receta documentada, sin función nueva
```

### QW-11: antes y después

```python
# Antes: la hora de la semana se construye a mano como exógena (también para predecir)
how = y.index.dayofweek * 24 + y.index.hour
exog = pd.get_dummies(pd.Categorical(how, categories=range(168)), prefix="how", dtype=int)
exog.index = y.index
forecaster = ForecasterRecursive(Ridge(), lags=24)
forecaster.fit(y=y, exog=exog)
```

```python
# Después
forecaster = ForecasterRecursive(
    Ridge(), lags=24,
    calendar_features=CalendarFeatures(features=["hour_of_week"], encoding="onehot"),
)
forecaster.fit(y=y)   # predict calcula la feature para las fechas futuras
```

### QW-11: análisis de features de calendario (2026-10-20)

**Método.** 18 series: 5 horarias (bike_sharing, vic_electricity, NO2 de Valencia, 2 edificios
de bdg2), 8 diarias (bike_sharing, vic_electricity, metro, autobús y tren de Madrid, BiciMAD,
Wikipedia y el total de store_sales) y 5 mensuales (h2o, 3 combustibles y comercio minorista
de Australia). `ForecasterRecursive` con Ridge y con LightGBM, backtesting con `refit=False`
(horarias: lags 1 a 24 y 168, `steps=24`, 8 semanas de test; diarias: lags 1 a 14,
`steps=7`, 52 semanas; mensuales: lags 1 a 12, `steps=12`, 48 meses). Cada configuración se
compara con una base de buenas prácticas con lo que `CalendarFeatures` ofrece hoy: one-hot
con Ridge y enteros con LightGBM. Las features se construyen desde el índice y entran como
exógenas, para poder probar las que todavía no existen. Python 3.14.3, numpy 2.5.3,
pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.6.0. Script y resultados por serie:
`repro/QW-11-calendar-analysis.py` y `.csv`.

**Resultados** (cambio del MAE frente a la base; mediana, y entre paréntesis cuántas series
mejoran o empeoran más de un 1%):

| Frecuencia | Configuración | Ridge | LightGBM |
|---|---|---|---|
| Horaria | `hour_of_week` en lugar de `day_of_week` + `hour` | **-23,8%** (4 mejor, 0 peor) | +4,9% (1, 4) |
| Horaria | Base + `hour_of_week` | **-23,8%** (4, 0) | -2,6% (3, 1) |
| Horaria | Base + festivo (solo 2 series) | +0,3% | -1,1% (1, 0) |
| Horaria | Codificación cíclica (seno y coseno, la que se usa por defecto) en lugar de one-hot | +3,4% (0, 4), hasta +12,6% | -0,1% |
| Horaria | Spline existente en lugar de one-hot | 0,0% (idéntico) | +2,8% |
| Horaria | Base + `day_of_year` (Fourier, K=3) con menos de un año de datos (bdg2) | +65% y +140% | +11% y +150% |
| Diaria | Base + `is_month_start` / `is_month_end` / días hasta fin de mes | +0,2% / 0,0% / +0,5% (ninguna mejora) | 0,0% / 0,0% / -1,5% (resultados mixtos) |
| Diaria | Base + `week_of_month` / `day_of_month` | -0,1% / +1,6% | -1,2% / -0,8% (mixtos) |
| Diaria | Base + festivo (solo 2 series) | **-4,5%** (vic: -8,0%) | **-6,3%** (vic: -12,5%) |
| Diaria | `day_of_year` cíclico (K=1) en lugar de `month` | +7,4% (1, 6) | -3,6% (5, 1) |
| Diaria | `day_of_year` Fourier K=6 en lugar de `month` | -0,2% | -7,1% (7, 1) |
| Diaria | `day_of_year` spline existente (13 nodos) en lugar de `month` | -0,5% | -6,3% (5, 2) |
| Diaria | `day_of_year` entero existente en lugar de `month` | +7,5% | -4,9% (5, 2) |
| Diaria | Base + `day_of_year` spline existente | -2,0% (6, 2) | -3,4% (7, 1) |
| Mensual | Base + `days_in_month` | +0,2% | +0,5% |
| Mensual | Base + días de fin de semana del mes | **-1,3%** (4, 0), hasta -11,8% | +2,0% (0, 4) |
| Mensual | Base + número de cada día de la semana del mes | -0,8% (2, 0), hasta -12,8% | -0,8% (2, 2) |

**Conclusiones**

1. **`hour_of_week` es la feature nueva que más aporta:** una mediana del 24% menos de MAE con
   Ridge en series horarias, sin empeorar ninguna. Un modelo lineal no puede representar la
   interacción entre día de la semana y hora con dos features separadas; los árboles sí (con
   LightGBM, añadirla da un -2,6%, y sustituir las dos por ella empeora). La tienen sktime y
   statsmodels (`CalendarSeasonality`).
2. **Los indicadores de inicio y fin de periodo no mejoran la precisión.** En las 8 series
   diarias, `is_month_start`, `is_month_end` y los días hasta fin de mes no mejoran con Ridge y
   dan resultados mixtos con LightGBM. Los de trimestre y año son redundantes con `month` y
   `day_of_year`. Tiene sentido añadirlos solo como comodidad, porque feature-engine y la rama
   principal de mlforecast los tienen. Salvedad: ninguna de las series tiene efectos de nómina
   o de cierre de mes (pagos, retiradas de efectivo), donde sí podrían aportar.
3. **Los festivos son lo que más mejora las series diarias** (del 4,5% al 12,5% en las dos
   series con festivos), pero `CalendarFeatures` no los genera. darts, sktime, Prophet y
   pytimetk usan el paquete `holidays`; gluonts usa kernels de distancia al festivo. Pasa a
   FE-06 (sección 3): es el hueco más claro frente a otras librerías.
4. **Los días de fin de semana del mes ayudan a los modelos lineales en series mensuales**
   (4 de 5 series mejoran, hasta un 12%), con LightGBM no. Son los regresores de días
   laborables de X-13 y del paquete forecast de R; ninguna librería de Python los ofrece.
5. **La codificación importa tanto como las features:** la cíclica (seno y coseno con un
   armónico), que es la de `CalendarFeatures` por defecto, empeora los modelos lineales (+3,4%
   de mediana en horarias, hasta +12,6%; +7,4% en el ciclo anual de las diarias). One-hot y la
   spline existente rinden igual entre sí. No hace falta añadir Fourier con K armónicos
   (Prophet, sktime, statsmodels): la spline ya cubre ese caso. Hace falta documentarlo:
   one-hot o spline con modelos lineales, enteros con árboles.
6. **Con árboles en series diarias, la posición en el año a resolución diaria mejora frente a
   `month`** (de -4,9% a -7,1% según la codificación, en 5 a 7 de 8 series), y eso ya se puede
   hacer con `day_of_year`. Es guía, no código nuevo.
7. **Las features anuales con menos de un año de historia son peligrosas:** en los dos
   edificios de bdg2 (9 meses de datos), añadir `day_of_year` empeora el error entre un 11% y
   un 150%, porque el test cae en una parte del año que no se ha visto. Merece un aviso.

**Salvedades.** Un modelo por familia con hiperparámetros fijos; los festivos, solo en 2
series; en store_sales (datos simulados), LightGBM mejora un 40% con cualquier feature que
resuelva el año por días, y eso arrastra los mejores casos de varias filas diarias.

### QW-04: antes y después

```python
# Antes
_, preds = backtesting_forecaster_multiseries(
    forecaster, series=series, cv=cv, metric="mean_absolute_error", interval=[0.1, 0.9]
)
# series dict:
y = pd.concat(series, names=["level", "date"]).rename("y_true").reset_index()
# series DataFrame ancho (pd.concat falla con TypeError):
# y = series.stack().rename("y_true").rename_axis(["date", "level"]).reset_index()
preds = preds.rename_axis("date").reset_index().merge(y, on=["level", "date"], how="left").set_index("date")
preds = preds.dropna(subset=["y_true"])  # si no, calculate_coverage cuenta NaN como no cubierto
```

```python
# Después (backtesting_forecaster, _multiseries, _stats y _foundation)
_, preds = backtesting_forecaster_multiseries(
    forecaster, series=series, cv=cv, metric="mean_absolute_error", interval=[0.1, 0.9]
)
preds[["level", "fold", "y_true", "pred", "lower_bound", "upper_bound"]]
residuals = preds["y_true"] - preds["pred"]
# Las filas sin valor observado tienen NaN en y_true y en todas las columnas de predicción.
```

### QW-02: antes y después

```python
# Antes
forecaster = ForecasterFoundation(FoundationModel("autogluon/chronos-2-small", context_length=256))
forecaster.fit(series=y, exog=exog_train)   # exog_train con la columna 'promo'
predictions = forecaster.predict(steps=7)   # sin aviso: 'promo' solo se usa como covariable pasada
```

```python
# Después
predictions = forecaster.predict(steps=7)
# MissingExogWarning: `exog` is None, so the historical exog columns are used only as
# past covariates by ChronosAdapter for series {'y': ['promo']}. If their future values
# are known (calendar, promotions, holidays...), pass them in `exog`. If they are
# past-only covariates on purpose, ignore this warning.
```

### QW-05: antes y después

```python
# Antes
forecaster = ForecasterFoundation(FoundationModel("autogluon/chronos-2-small"))
forecaster.fit(series=y)
backtesting_foundation(forecaster, series=y, cv=cv, metric="mean_absolute_error")  # carga los pesos
save_forecaster(forecaster, "forecaster.joblib")  # 112 MB
# Con "google/timesfm-2.5-200m-pytorch": PicklingError y fichero corrupto de 0 a 2 bytes
```

```python
# Después
save_forecaster(forecaster, "forecaster.joblib")  # 0,01 MB con cualquier adaptador
forecaster = load_forecaster("forecaster.joblib")
forecaster.predict(steps=7)  # recarga los pesos desde la caché de HF (0,33 s), mismas predicciones
# Un objeto pasado por el usuario (FoundationModel(..., pipeline=pipe_ajustado)) se conserva.
```

### QW-07: antes y después

```python
# Antes
y = pd.read_csv("data.csv", parse_dates=["date"], index_col="date")["y"]
forecaster.fit(y)
# ValueError: `y` has a pandas DatetimeIndex without a frequency. To avoid this error, set the frequency of the DatetimeIndex.
forecaster.predict_interval(steps=12, interval=[10, 90])
# ValueError: Lower interval bound (10) must be >= 0 and < 1.
forecaster_ms.fit(series)                 # store_in_sample_residuals=False por defecto
forecaster_ms.predict_interval(steps=12)
# ValueError: Residuals for level 'x1' are None or empty. (con encoding=None: '_unknown_level')
grid_search_forecaster(forecaster_ms, y, cv, param_grid, metric="mean_absolute_error")
# un RuntimeWarning por combinación y DataFrame vacío, sin excepción
grid_search_forecaster(forecaster_foundation, y, cv, {"context_length": [256, 512]}, metric="mean_absolute_error")
# AttributeError: 'ForecasterFoundation' object has no attribute 'lags'
```

```python
# Después
# ValueError: ... Inferred frequency: 'MS'. Use `.asfreq('MS')` to set it.
# ValueError: Lower interval bound (10) must be >= 0 and < 1. Since version 0.23.0, `interval`
#   and `quantiles` are expressed as quantiles in [0, 1], not percentiles: e.g. `interval = [0.1, 0.9]`.
# ValueError: `forecaster.in_sample_residuals_by_bin_` is either None or empty. Use
#   `store_in_sample_residuals = True` when fitting the forecaster or `set_in_sample_residuals()`.
# TypeError: ... Got ForecasterRecursiveMultiSeries. For `ForecasterRecursiveMultiSeries` use:
#   `backtesting_forecaster_multiseries`, `grid_search_forecaster_multiseries`, ...
# TypeError: ... Got ForecasterFoundation. For `ForecasterFoundation` use:
#   `backtesting_foundation`, `bayesian_search_foundation`.
```

### QW-08: antes y después

```python
# Antes
from sklearn.metrics import root_mean_squared_error
metric, preds = backtesting_forecaster(forecaster, y=y, cv=cv, metric=[root_mean_squared_error, "mean_absolute_error"])
# metric="root_mean_squared_error" -> ValueError: Allowed metrics are: [...]
q1, q9 = create_mean_pinball_loss(0.1), create_mean_pinball_loss(0.9)
backtesting_forecaster(..., metric=[q1, q9])              # columnas duplicadas 'mean_pinball_loss_q'
backtesting_forecaster_multiseries(..., metric=[q1, q9])  # TypeError: Axis must be specified ...
```

```python
# Después
metric, preds = backtesting_forecaster(forecaster, y=y, cv=cv, metric=["root_mean_squared_error", "mean_absolute_error"])
create_mean_pinball_loss(0.1).__name__  # 'mean_pinball_loss_q_0.1': columnas distintas
# Dos métricas con el mismo __name__ (lambdas, partials) -> ValueError claro, igual que en las búsquedas
```

### QW-06: antes y después

```python
# Antes
forecaster.fit(y=y, exog=exog_all.loc[y.index])
start = forecaster.last_window_.index[-1] + forecaster.last_window_.index.freq
predictions = forecaster.predict(steps=24, exog=exog_all.loc[start:])
# predict(steps=24, exog=exog_all) -> ValueError: `exog` must start one step ahead of `last_window`
```

```python
# Después (solo DatetimeIndex; RangeIndex mantiene el chequeo estricto)
forecaster.fit(y=y, exog=exog_all.loc[y.index])
predictions = forecaster.predict(steps=24, exog=exog_all)
```

---

## 3. Features y gaps

| Item | Área | Valor para el usuario | A cuántos usuarios afecta | Esfuerzo | Evidencia | Esbozo del cambio / API | Riesgo | Rompe la API |
|---|---|---|---|---|---|---|---|---|
| **FE-02** Covariables solo pasadas en `backtesting_foundation` (`past_only_exog`). Amplía ISSUE-1131 (solo la parte foundation) | foundation / model_selection | Evaluar de forma honesta variables observadas a posteriori o indicadores adelantados, y ajustar `context_length` con ellas, sin bucle manual | Nicho dentro de foundation (Chronos-2, TS-ICL, TimesFM 3.0), que es foco principal | S (30 a 40 líneas, ~4 tests) | `skforecast/model_selection/_validation.py:2521,2596,2606` (todas las columnas de exog van como futuras); `_search.py:2328-2339`; medido con Chronos-2 small: MAE 1,186 con la columna como futura frente a 2,919 honesto; como solo pasada mejora un 6% frente a descartarla (3,104) | `past_only_exog: str or list[str] or None = None` en `backtesting_foundation` y `bayesian_search_foundation`; se quitan de `exog_test` y se conservan en `context_exog`; `ValueError` si no existen; `IgnoredArgumentWarning` con adaptadores sin soporte | Bajo si BUG-10 se corrige antes; sin él, TabICL y TabPFN-TS corrompen la predicción en silencio. Presentarlo como hueco, no como fuga (el contrato de exog es "conocida a futuro") | No |
| **FE-04** `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries` | model_selection | Baseline en una llamada con métricas por serie y agregadas y, sobre todo, con los mismos pliegues que el modelo global (el bucle manual compara periodos distintos sin avisar) | Flujo habitual en multiserie para quien valida contra un baseline; sin peticiones explícitas | M (~120 líneas, 12 a 15 tests, docs) | `TypeError` en `_validation.py:1687-1701`; medido: con una serie que empieza 60 días tarde, el bucle evalúa 40 predicciones desde 2022-12-27 y el modelo global 100 desde 2022-10-28; prototipo: filas (idx, level, fold) idénticas, 50 series en 0,63 s | Función privada `_backtesting_equivalent_date_multiseries` con pliegues sobre el índice común y un clon por nivel; solo conformal; errores claros con bootstrapping, `return_predictors` y `use_in_sample_residuals=False`; arreglo de `_split.py:985` (DateOffset con dict). Ya en docs: `initial_train_size` como fecha en el bucle manual | Corrección: pliegues, series cortas (misma frontera `window_size` que #1410, `_forecaster_equivalent_date.py:433`), DateOffset con dict. Conflictos con QW-04 y QW-09 (ISSUE-1406 e ISSUE-1410 ya resueltos con #1415 y #1418): hacerlo después. Las dos lentes la ven de prioridad media-baja | No |
| **FE-03** Ventanas estacionales en `RollingFeatures` (`season_length`) | preprocessing | Medias estacionales (misma hora de los últimos días, mismo día de las últimas semanas) sin clases propias frágiles que además no se guardan | Series horarias o diarias con estacionalidad múltiple y modelos de árboles | M (código S, ~40 líneas; el resto es docs y contexto IA) | `skforecast/preprocessing/_preprocessing.py:1604,1657,1673` (solo ventanas contiguas); `skforecast/utils/utils.py:3599-3611` (las clases propias no se guardan); medido con LGBM: -2,2% de MAE en bike_sharing y -1,9% en vic_electricity frente a lags estacionales; con Ridge, 0%; prototipo: diferencia transform/transform_batch 3e-14 | `season_length=None` como último parámetro; `window_sizes` sigue siendo el tramo en pasos; validar `window_sizes % season_length == 0`; atributo de clase `season_length = None` para pickles de 0.26; nombres `roll_mean_168_s24`. Documentar ya los lags estacionales múltiples (`skills/feature-engineering/SKILL.md:250-257`) | Bajo con la semántica de tramo (la API propuesta al principio rompía `window_size` de los forecasters). En Direct la ventana se alinea con el origen; `window_size` grande excluye series cortas. Duda: mejora modesta y solo con árboles | No |
| **FE-05** Comparar varios forecasters en un solo backtesting (añadido el 2026-10-20; diseño analizado y prototipado, ver "FE-05: análisis de diseño") | model_selection | Comparar baseline, ML, stats y foundation en una llamada, sobre los mismos puntos de test y con la misma escala en las métricas, con una tabla por modelo (y por serie) y el tiempo de cada uno. Hoy hay que llamar a cuatro funciones distintas, renombrar las columnas `q_*` de foundation, medir tiempos a mano y escribir una función de resumen propia | Flujo habitual: elegir modelo frente a un baseline es el primer paso de casi todo proyecto. Sin peticiones explícitas en GitHub | M (~300 líneas: dispatch a las cuatro funciones existentes, normalización de la salida, métricas centralizadas sobre puntos comunes; tests de equivalencia con las llamadas individuales; reescribir la guía de comparación) | `docs/user_guides/comparing-forecasting-models.ipynb` (celdas 5 y 9 a 13): `summarize` propio, un `time.perf_counter()` por modelo, `rename` de `q_0.1`/`q_0.5`/`q_0.9` en foundation y bucles por serie para los modelos locales. Prototipo: `repro/FE-05-prototype.py`. Precedente de varios modelos en un backtesting: `ForecasterStats` con varios estimadores y columna `estimator_id` (#1093) | `compare_forecasters(forecasters: dict[str, Forecaster], series, cv, metric, ...)`; ver el diseño completo en el análisis | Medio: cuatro familias con argumentos y salidas distintos. Requisitos: QW-03 (bloqueante, medido; hecho con #1423) y BUG-06; BUG-11 se rodea dentro | No (función nueva) |
| **FE-06** Festivos en `CalendarFeatures` (añadido el 2026-10-20) | preprocessing | Festivos nacionales y regionales como features (indicador, días antes y después, puentes) sin construir la columna a mano | Flujo habitual en series diarias y horarias de demanda, movilidad y energía | M (dependencia opcional `holidays`, país y región, ventanas, tests y docs) | "QW-11: análisis" (sección 2): mediana de -4,5% con Ridge y -6,3% con LightGBM en las 2 series diarias con festivos (vic_electricity: -8,0% y -12,5%). Lo ofrecen darts (`holidays_timeseries`), sktime (`HolidayFeatures`, con ventanas y puentes), Prophet (`add_country_holidays`, con ventanas) y pytimetk; gluonts usa kernels de distancia al festivo. skforecast solo tiene `calculate_distance_from_holiday`, que necesita la columna de festivos ya construida | En `CalendarFeatures`: `holidays_country`, `holidays_subdiv` y `holidays_window=(días antes, días después)`, o una lista de fechas propia; columnas `is_holiday` y, con ventana, una por desplazamiento. Reutilizar `calculate_distance_from_holiday` para la distancia | Medio: dependencia nueva (opcional); los festivos cambian entre versiones del paquete, así que los resultados solo se reproducen fijando la versión. Solo 2 series medidas (decisión pendiente 9) | No (parámetros nuevos) |

### FE-02: antes y después

```python
# Antes: bucle manual (x_observed solo pasada, weekend conocida a futuro)
preds = []
for k, end in enumerate(range(initial_train_size, len(y) - steps + 1, steps)):
    p = forecaster.predict(
        steps=steps, context=y.iloc[:end], context_exog=exog.iloc[:end],
        exog=exog[["weekend"]].iloc[end:end + steps],
    )
    preds.append(p.assign(fold=k))
# Multiserie, gap, cuantiles o bayesian_search_foundation: sin alternativa
```

```python
# Después
metric, preds = backtesting_foundation(
    forecaster, series=y, exog=exog, cv=cv, metric="mean_absolute_error",
    past_only_exog=["x_observed"],
)
results, best = bayesian_search_foundation(
    forecaster, series=y, exog=exog, cv=cv, search_space=search_space,
    metric="mean_absolute_error", past_only_exog=["x_observed"],
)
```

### FE-04: antes y después

```python
# Antes: bucle por serie; con initial_train_size entero, las series que empiezan tarde
# se evalúan en otro periodo que el modelo global (sin aviso)
preds = []
for level in series.columns:
    _, p = backtesting_forecaster(ForecasterEquivalentDate(offset=7), y=series[level], cv=cv,
                                  metric="mean_absolute_error", show_progress=False)
    preds.append(p.assign(level=level))
# métricas agregadas (average, weighted_average, pooling): a mano
```

```python
# Después
metrics, preds = backtesting_forecaster_multiseries(
    forecaster=ForecasterEquivalentDate(offset=7),
    series=series, cv=cv,                      # mismo cv que el modelo global -> mismos pliegues
    metric=["mean_absolute_error", "mean_absolute_scaled_error"],
    interval=[0.1, 0.9],                       # solo conformal
)
```

### FE-03: antes y después

```python
# Antes: clase propia (no se guarda con save_forecaster y es fácil desalinear transform y transform_batch)
window_features = [RollingFeatures(["mean"], [24]), SeasonalRolling(24, 7), SeasonalRolling(168, 4)]
```

```python
# Después
window_features = RollingFeatures(
    stats         = ["mean", "mean", "mean"],
    window_sizes  = [24, 168, 672],    # tramo en pasos de tiempo (mismo significado que hoy)
    season_length = [None, 24, 168],   # None = ventana contigua
)
# roll_mean_168_s24 = media de y[t-24], y[t-48], ..., y[t-168] (7 puntos)
```

### FE-05: antes y después

```python
# Antes (guía comparing-forecasting-models): una función por familia, tiempos y resumen a mano
predictions, elapsed = {}, {}
start = time.perf_counter()
_, predictions["Seasonal naive"] = backtesting_forecaster(naive, y=y, cv=cv, ...)
elapsed["Seasonal naive"] = time.perf_counter() - start
# ... lo mismo con backtesting_forecaster (Ridge, LightGBM), backtesting_stats (ARIMA, ETS)
_, pred = backtesting_foundation(chronos, series=y, cv=cv, quantiles=[0.1, 0.5, 0.9], ...)
predictions["Chronos-2"] = pred.rename(
    columns={"q_0.1": "lower_bound", "q_0.5": "pred", "q_0.9": "upper_bound"}
)
results = summarize(predictions, elapsed, ...)   # función propia de ~40 líneas
```

```python
# Después
metrics, predictions = compare_forecasters(
    forecasters = {
        "Seasonal naive": ForecasterEquivalentDate(offset=7),
        "LightGBM": ForecasterRecursive(LGBMRegressor(), lags=7),
        "ARIMA": ForecasterStats(Arima(m=7)),
        "Chronos-2": ForecasterFoundation(FoundationModel("autogluon/chronos-2-small")),
    },
    series   = y,
    cv       = cv,
    metric   = ["mean_absolute_error", "mean_absolute_scaled_error"],
    interval = [0.1, 0.9],
)
# metrics: forecaster, level, n_predictions, mean_absolute_error,
#          mean_absolute_scaled_error, coverage, interval_width, elapsed_time
```

### FE-05: análisis de diseño (2026-10-20)

**Cómo difieren hoy las cuatro funciones** (comprobado en `0.27.x`):

| | `backtesting_forecaster` | `backtesting_forecaster_multiseries` | `backtesting_stats` | `backtesting_foundation` |
|---|---|---|---|---|
| Datos | `y` | `series` | `y` | `series` (Series, DataFrame o dict) |
| Intervalo | `interval`, `interval_method='bootstrapping'` por defecto | `interval`, `interval_method='conformal'` por defecto | `interval` o `alpha` | solo `quantiles` |
| Salida de predicciones | `fold, pred, lower_bound, upper_bound` | `level, fold, pred, ...` | `fold, pred, ...` (+ `estimator_id` con varios estimadores) | `level, fold, pred`; con cuantiles, `q_*` y **sin `pred`** (BUG-06) |
| Tabla de métricas | sin columna de nivel | columna `levels` (en plural) | `estimator_id` con varios estimadores | sin columna de nivel con una serie |
| Particularidades | `ForecasterEquivalentDate` solo admite conformal (BUG-11) | | | la primera llamada carga el modelo (16,7 s frente a menos de 1 s después, Chronos-2 small en CPU) |

**Medido con el prototipo** (`repro/FE-05-prototype.py`, h2o, 36 puntos de test):

1. **Con un solo `cv` no se pueden comparar forecasters con distinta `differentiation`.** Con
   `cv.differentiation=None`, el forecaster con `differentiation=1` da `ValueError`; con
   `cv.differentiation=1`, el que no diferencia da `ValueError`. QW-03 es requisito (el
   prototipo lo rodea fijando la `differentiation` del cv para cada forecaster). Resuelto con
   #1423: el cv ya no usa `differentiation`, así que un mismo cv sirve para todos.
2. **MASE cambia con el `window_size` de cada forecaster:** con las mismas predicciones,
   0,7051, 0,6931 y 0,6784 según `y_train` empiece en la observación 1, 12 o 24 (un 4%).
   Cada función recorta `y_train` con su propio `window_size` (`_validation.py:683-695`), así
   que cada modelo se mide con una escala distinta. Además, el `window_size` de foundation es
   la longitud de contexto (8192 con Chronos-2, en una serie de 204 puntos), así que "el mayor
   `window_size`" no sirve como regla común. La regla correcta: todas las observaciones
   anteriores al primer punto de test, iguales para todos.
3. **Puntos comunes:** en un panel en el que una serie empieza tarde, el modelo con
   `lags=72` no puede predecirla y el resto sí. Si se evalúa a cada modelo sobre sus puntos,
   la comparación no es justa; si se usan los puntos comunes, esa serie desaparece de la
   comparación para todos. Es el mismo dilema que BUG-search-short-series, y la opción c2 de
   ese documento lo reduciría.
4. **La cobertura compara configuraciones, no solo modelos:** LightGBM con los residuos
   in-sample por defecto cubre un 11% con un intervalo nominal del 80%, frente al 58% de
   ARIMA o el 78% de Chronos-2. Una tabla de comparación lo pone a la vista; hay que
   documentar la receta de residuos out-of-sample, no cambiar defaults.

**Alternativas evaluadas**

| Opción | Ventajas | Inconvenientes | Veredicto |
|---|---|---|---|
| A. Lista de forecasters en `backtesting_forecaster` y variantes | Sin función nueva | Solo compara dentro de una familia; el tipo de retorno cambia según la entrada; los argumentos de intervalo y residuos son de una sola familia | Descartada |
| B. Función nueva que llama a la función de cada familia y centraliza las métricas | Compara familias; reutiliza las funciones existentes sin reimplementarlas; mismas reglas para todos | Hay que mapear argumentos por familia | **Recomendada** |
| C. Clase con estado (`ForecastingBenchmark` de sktime, `leaderboard` de AutoGluon) | Permite añadir modelos, gráficos y caché | Mucha más API; no encaja con el estilo funcional de `model_selection` | Descartada |
| D. Solo un resumen de resultados ya calculados (`{nombre: predicciones}` → tabla) | Muy barata; el usuario conserva todo el control | Siguen siendo N llamadas, sin tiempos y sin mapeo de intervalos | Es la mitad interna de B; no exponerla en la v1 |
| E. Predicciones en formato ancho, una columna por modelo (Nixtla) | Cómodo para gráficos | Choca con el formato largo de skforecast (`level`, `estimator_id`) | Formato largo con columna `forecaster` |

**Diseño recomendado (opción B)**

```python
metrics, predictions = compare_forecasters(
    forecasters           = {"Seasonal naive": ..., "LightGBM": ..., "ARIMA": ..., "Chronos-2": ...},
    series                = y,                  # Series, DataFrame ancho o dict, como backtesting_foundation
    cv                    = cv,                 # initial_train_size obligatorio
    metric                = ["mean_absolute_error", "mean_absolute_scaled_error"],
    exog                  = exog,
    interval              = [0.1, 0.9],         # solo dos cotas, o None
    levels                = None,
    add_aggregated_metric = True,
    common_points         = True,
    backtesting_kwargs    = {"LightGBM": {"use_in_sample_residuals": False},
                             "ETS": {"exog": None}},
    show_progress         = True,
    suppress_warnings     = False,
)
```

Reglas:

1. **Llama a las funciones públicas existentes sin modificarlas.** Test clave: las
   predicciones de cada modelo son idénticas a las de su llamada individual.
2. **Un solo argumento `series`.** Los forecasters de una serie (Recursive, Direct,
   EquivalentDate, Stats) reciben la serie si hay una; con varias series, `ValueError` en la
   v1 (los modelos locales por serie, en la v2). Los globales y foundation reciben el panel.
3. **Intervalo común de dos cotas.** ML: `interval`; EquivalentDate: con
   `interval_method='conformal'` forzado (BUG-11); stats: `interval`; foundation: `quantiles`
   `[lo, 0.5, hi]` mapeados a `lower_bound` y `upper_bound`, con el punto de `pred` (requiere
   BUG-06; hoy solo hay `q_0.5`, que no es el punto de TabICL ni de Nori). Distribuciones,
   bootstrapping o varios cuantiles quedan fuera de la v1.
4. **`backtesting_kwargs` por nombre** para lo propio de cada familia (`interval_method`,
   `use_in_sample_residuals`, `n_boot`, `freeze_params`, `exog`, `n_jobs`...). Se valida
   contra la firma de la función de destino; las claves que controla `compare_forecasters`
   (`cv`, `metric`, `series`/`y`, `interval`, `quantiles`) dan `ValueError`.
5. **Métricas centralizadas:** se descartan las de cada función y se recalculan sobre las
   predicciones, con la misma deduplicación de folds solapados y el mismo `y_train` para
   todos (todas las observaciones anteriores al primer punto de test). Reutilizar
   `_calculate_metrics_backtesting_multiseries`, que ya da los agregados `average`,
   `weighted_average` y `pooling`, y `coverage` e `interval_width` (#1420).
6. **Puntos comunes por defecto**, con un aviso que diga cuántos puntos se quitan a cada
   modelo; `common_points=False` evalúa a cada uno sobre los suyos. La columna
   `n_predictions` siempre está.
7. **`initial_train_size` obligatorio:** con `None`, el inicio del test depende del
   `window_size` de cada forecaster y la evaluación sería in-sample (BUG-01).
8. **`ForecasterStats` con varios estimadores:** una fila por estimador, `"ARIMA (Arima)"`;
   el tiempo es el del forecaster completo y se indica.
9. **Errores:** se interrumpe en el primer fallo, con el nombre del modelo en el mensaje
   (`raise ValueError(f"Forecaster 'LightGBM': ...") from e`). No se silencian, para no
   repetir BUG-05.
10. **Ejecución secuencial entre modelos;** cada función conserva su `n_jobs`. `elapsed_time`
    incluye la carga del modelo foundation en la primera llamada: documentarlo.
11. **Fuera de la v1:** forecasters de clasificación (mezclar tareas no tiene sentido),
    modelos locales por serie en paneles, gráfico de comparación y paralelismo entre
    modelos. `ForecasterRnn` y `ForecasterDirectMultiVariate` entran sin coste por la función
    multiserie.

**Salidas:** `metrics` con `forecaster`, `level` (singular, también con una serie, aunque
`backtesting_forecaster_multiseries` use `levels`), `n_predictions`, las métricas, `coverage`,
`interval_width` y `elapsed_time`; `predictions` en formato largo, con índice de fechas y las
columnas `forecaster`, `level`, `fold`, `pred`, `lower_bound` y `upper_bound` (y `y_true` y
`step` cuando lleguen QW-04 y QW-10).

**Dependencias y orden:** QW-03 (bloqueante, hecho con #1423) → BUG-06 (para intervalos de
foundation) → FE-05. QW-04 y QW-10 no bloquean, porque las métricas se recalculan aquí, pero conviene que las
columnas de salida coincidan: si FE-05 va antes, que ya incluya `y_true` y `step` con los
mismos nombres. Docs: reescribir `comparing-forecasting-models.ipynb` con la función (quita la
función `summarize`, los tiempos y los `rename`) y añadirla a `llms-base.txt` y a las skills
`choosing-a-forecaster` y `backtesting-configuration`.

---

## 4. Bugs a corregir

Todos reproducidos con un script en `repro/` (vueltos a ejecutar en la crítica: BUG-03,
BUG-07, BUG-10, BUG-11 y BUG-stats-interval-ignored). BUG-10, BUG-11 y
BUG-stats-interval-ignored son hallazgos laterales (verificación de FE-02, de FE-04 y fase 1).

| ID | Descripción | Área | Esfuerzo | Script de reproducción | Esbozo del arreglo |
|---|---|---|---|---|---|
| **BUG-03** | Las búsquedas eligen el peor modelo con métricas en las que mayor es mejor (`r2_score`): el orden y la dirección de Optuna dependen del tipo de tarea, no de la métrica, y `return_best=True` reentrena con el peor | model_selection | S | `dev/roadmap_0.27.0/repro/BUG-03.py` | (1) En las 3 búsquedas bayesianas, ordenar por `study.direction` (`_search.py:916,2049,2427`). (2) Helper `_infer_metric_direction` con la convención de sklearn (`*_score` se maximiza; `*_error`, `*_loss`, `*_deviance` se minimizan) usado en todas las ordenaciones (`_search.py:514,916,1537,2049,2427,2817,3077`) y como `direction` por defecto (`:846,1953,2369`). (3) Docstrings de `metric` y `kwargs_create_study`. **Hecho: #1431, fusionado en `0.27.x` el 2026-10-22**, con otro diseño para (2): la convención de nombres solo se aplica a `sklearn.metrics` y el resto usa el atributo `greater_is_better` o la tarea (decisión 6) |
| **BUG-01** | `backtesting` con `initial_train_size=None` sobre un forecaster ya ajustado evalúa in-sample sin ningún aviso | model_selection | S | `dev/roadmap_0.27.0/repro/BUG-01.py` | En la rama `else` de `check_backtesting_input` (`skforecast/model_selection/_utils.py:370-380`), comparar la primera fecha de test (`index[window_size + gap]`) con `forecaster.training_range_` (dict por serie en multiserie) y emitir un aviso silenciable con `suppress_warnings` con el número y rango de predicciones in-sample. Que el `NotFittedError` sugiera primero `initial_train_size` |
| **BUG-02** | Con `fold_stride < steps` la métrica solo mide los horizontes `1..fold_stride` (con 1, el error a 1 paso) y todas las búsquedas seleccionan por ese error | model_selection | S | `dev/roadmap_0.27.0/repro/BUG-02.py` | Parámetro en `TimeSeriesFold`, p. ej. `overlapping_metric="last"` o `"all"`, aplicado en los 5 puntos de deduplicación (`_validation.py:662-667,1488-1493,2159-2164,2643-2648,2674-2679`; números de línea anteriores a #1409; el de stats usa (fecha, `estimator_id`) con varios estimadores tras BUG-09 (a)); aviso mientras el defecto sea `"last"` (como `OneStepAheadValidationWarning`, `_utils.py:653-658`); no cambiar el defecto sin anunciarlo (la receta de meses naturales depende de `"last"`); corregir `skills/backtesting-configuration/SKILL.md:202-208`, la guía y `llms-base.txt`. Coordinar con BUG-09 |
| **BUG-04** | Un pipeline o modelo precargado (p. ej. Chronos-2 con fine-tuning) se descarta en silencio al crear `ForecasterFoundation`: `clone` lo pierde y se recarga el checkpoint base | foundation | S | `dev/roadmap_0.27.0/repro/BUG-04.py` | `FoundationModel.__sklearn_clone__`: clon estándar y, si el usuario pasó el objeto, copiar la referencia al clon sin copiar pesos. Así lo heredan `ForecasterFoundation.__init__` (`_forecaster_foundation.py:156`), `deepcopy_forecaster` (`utils.py:4999`) y `bayesian_search_foundation`. Cada adaptador declara su atributo de backend (compartido con QW-05). Requisito de PLAN-CKPT (`dev/PLAN_UnifiedFinetunedCheckpoint.md`) |
| **BUG-10** | TabICL (medido) y TabPFN-TS (mismo código) no ignoran las covariables solo pasadas aunque `IgnoredArgumentWarning` diga que sí: predicciones absurdas en silencio al ajustar con exog y predecir sin ella o con menos columnas | foundation | S | `dev/roadmap_0.27.0/repro/BUG-10.py` (requiere `tabicl`) | En `FoundationModel.predict`, si `adapter.supports_past_only_covariates` es False, quitar de `context_exog` las columnas sin futuro antes de llamar al adaptador (arreglo centralizado). Causa: `_adapters.py:2679-2685` (TabICL) y `:3347-3353` (TabPFN) concatenan toda la exog de contexto; T0 (`:3869-3878`) y Nori (`:4795-4799`) ya filtran bien |
| **BUG-stats-interval-ignored** | `ForecasterStats.predict_interval` ignora `interval` porque `alpha=0.05` por defecto tiene precedencia: `interval=[0.1, 0.9]` devuelve el intervalo del 95%. La skill lo usa como ejemplo de intervalo del 80% | stats | S | `dev/roadmap_0.27.0/repro/BUG-stats-interval-ignored.py` | **Hecho: #1416, fusionado en `0.27.x` el 2026-10-19.** `alpha=None` por defecto; se usa `interval` si se pasa y el 95% si no se pasa ninguno; `alpha` sigue como alias (sin deprecar, decisión 5) y pasar los dos da `ValueError`, en `predict_interval` y en `backtesting_stats`. `backtesting_stats` traduce `alpha` a `interval` al principio. `interval` pasa a ir antes que `alpha` en las dos firmas (una llamada posicional antigua falla con `TypeError`, no en silencio; no hay ninguna en el repo). Ahora también se valida `alpha` (entre 0 y 1) en `predict_interval`. Corregidos `skills/statistical-models/SKILL.md`, `skills/prediction-intervals/` y `skills/complete-api-reference/`, la guía `forecasting-statistical-models.ipynb` (re-ejecutada, mismas predicciones) y `releases.md` (Fixed, Changed, highlight y "Before upgrading") |
| **BUG-06** | En TabICL y Nori (`point_estimate='mean'` por defecto) y TabPFN con `point_estimate` distinto de `'median'`, `predict()` da la media pero `predict_interval()['pred']` y la métrica de `backtesting_foundation` con `quantiles` usan `q_0.5`; además la salida pierde `pred` | foundation | M | `dev/roadmap_0.27.0/repro/BUG-06.py` | `FoundationModel.predict(quantiles=..., include_point=True)` devuelve `pred` junto a `q_*` (reutiliza `q_0.5` si `point_is_median`; TabICL y TabPFN-TS ya calculan el punto en la misma llamada; Nori necesita otra). `predict_interval` (`_forecaster_foundation.py:946-960`) y `_backtesting_foundation` (`_validation.py:2590-2599,2650-2653,2683-2686`) usan ese `pred`. Corregir el docstring (`_foundation_model.py:1041,1058`) y la tabla de la guía (TimesFM 2.5 usa la mediana) |
| **BUG-08** | `freeze_params=True` (por defecto) no congela nada en `backtesting_stats` ni en `grid/random_search_stats`: la búsqueda automática de Arima/Ets se repite en cada fold y en cada candidato | model_selection (stats) | S | `dev/roadmap_0.27.0/repro/BUG-08.py` | En `_backtesting_stats`, tras el primer ajuste y antes de `Parallel`, propagar `best_params_` a las plantillas: `for est, est_ in zip(forecaster.estimators, forecaster.estimators_): if getattr(est_, "best_params_", None): est._set_params(**est_.best_params_)`. Causa: `_validation.py:2097-2100` congela `estimators_` y `_forecaster_stats.py:521` los vuelve a clonar desde las plantillas. Regenerar valores esperados de tests automáticos con `refit=True`; nota Fixed |
| **BUG-07** | El formato largo con MultiIndex que documenta la guía solo funciona si el MultiIndex sale de `reshape_series_wide_to_long`; con `df.set_index(["series_id", "date"])` fallan `fit`, backtesting, búsquedas multiserie y foundation | recursive (multiseries) / utils | S | `dev/roadmap_0.27.0/repro/BUG-07.py` | En la rama MultiIndex de `check_preprocess_series` (`utils.py:4199-4202`), si el grupo no tiene `freq`, inferirla con `pd.infer_freq` (al menos 3 observaciones) y aplicar `asfreq`; si no se puede, `ValueError` propio del MultiIndex que sugiera `reshape_series_long_to_dict(..., freq=...)`. Corregir la errata "If it a" (`:4271,4285`) y alinear `skills/forecasting-multiple-series/SKILL.md:52` con la guía. PR-1408 tocaba `check_preprocess_series`, pero se cerró y #1412, que lo sustituye, no toca `utils.py`: sin conflicto |
| **BUG-11** | `backtesting_forecaster(ForecasterEquivalentDate, interval=...)` con el `interval_method='bootstrapping'` por defecto lanza `AttributeError: ... no attribute 'predict_quantiles'` en lugar de un error claro (la guía lo esquiva fijando `conformal`) | model_selection | S | `dev/roadmap_0.27.0/repro/BUG-11.py` | `ForecasterEquivalentDate` está en `forecasters_boot_conformal` (`skforecast/model_selection/_utils.py:222-228`) pero solo admite conformal: lista propia de forecasters solo conformal y `ValueError` que indique `interval_method="conformal"` (o usar conformal por defecto para este forecaster) |
| **BUG-09** | `backtesting_stats` falla con `fold_stride != steps` con varios estimadores, y con Sarimax y `refit=False` (también con `refit` entero >= 3 y `fold_stride` por defecto); `grid_search_stats` devuelve 0 filas. Nicho: prioridad baja dentro de STATS | model_selection (stats) | S (~10 líneas) | `dev/roadmap_0.27.0/repro/BUG-09.py` | (a) **Hecho (2026-10-19)** con #1419 (rama `fix/stats-multi-estimator-overlapping-folds`, base `0.27.x`): deduplica por (fecha, `estimator_id`) con varios estimadores; test parametrizado (`fold_stride` 1 y 2, con y sin `interval`) que compara cada estimador con su backtesting por separado, falla sin el arreglo; nota en `releases.md` (Fixed). Desde #1409 también rompía `coverage` e `interval_width`; aplicado sobre #1409, cada estimador tiene las suyas. (b) `last_window` desde el origen del fold anterior hasta el del actual (`_validation.py:1825-1830`). Validado: diferencia 0,0 frente a la referencia; 52 tests de stats pasan con el parche |

**Rendimiento (reproducido)**

| ID | Descripción | Área | Esfuerzo | Script de reproducción | Esbozo del arreglo |
|---|---|---|---|---|---|
| **PERF-02** | El ajuste de Ets y Sarimax (L-BFGS-B de scipy >= 1.15) se vuelve de 3 a 70 veces más lento con cualquier otra carga en la máquina, y `n_jobs='auto'` deja en serie todo backtesting de Arima/Ets/Arar (refit obligatorio) | stats / model_selection (stats) | S | `dev/roadmap_0.27.0/repro/PERF-02.py` | (a) Envolver `Ets.fit`, `Sarimax.fit` (y `Arima.fit`) en `threadpoolctl.threadpool_limits(1, user_api="blas")` (dependencia de scikit-learn; 1,45 ms por llamada, 0,007 ms con un `ThreadpoolController` reutilizado). (b) `select_n_jobs_backtesting` (`_utils.py:706`): `max(1, cpu_count() - 1)` para `ForecasterStats` con `refit` True o 1. Volver a medir tras BUG-08 |
| **PERF-01** | `backtesting_foundation` llama al modelo una vez por fold aunque es zero-shot, y prepara los datos con 0,87 a 1,33 ms por serie y fold | foundation / model_selection | M | `dev/roadmap_0.27.0/repro/PERF-01.py` | (1) Posiciones de fold con `searchsorted` y sumas acumuladas de `notna` una vez por serie, en lugar de `_extract_data_folds_multiseries`. (2) Pseudo-series (serie, fold) en trozos de ~1024 por llamada; una llamada por fold con `cross_learning=True` (mezclar folds filtraría futuro). (3) Cachear `expand_index` y comprobar NaN con numpy en `_prepare_future_exog` (`_foundation_model.py:870,877,1261`). Sin cambios de API |
| **PERF-03** | Las métricas pedidas como texto (`mean_absolute_error`, `mean_squared_error`) usan las funciones de scikit-learn, cuya validación cuesta unos 150 µs por llamada; se llaman una vez por serie, métrica y candidato. Ocupan el 42 a 51% del tiempo de `grid_search_forecaster_multiseries` con `OneStepAheadFold` (más un 9% de MASE con 3 métricas) y el 5 a 11% del backtesting multiserie | metrics / model_selection | S (~1 día con tests de igualdad frente a scikit-learn y benchmark) | `dev/roadmap_0.27.0/repro/PERF-03.py` | En `_get_metric` (`skforecast/metrics/metrics.py:45-73`), versiones privadas en numpy de MAE y MSE (y RMSE con QW-08) con el mismo `__name__`, para que las columnas no cambien. Conservar la validación de scikit-learn: error con NaN y con longitudes distintas, y aplanar a 1D (si no, `(n,)` frente a `(n, 1)` daría por broadcasting una matriz `n x n` sin avisar). MASE y RMSSE con arrays de numpy en lugar de Series (hoy restan Series, que alinean por índice). MAPE y MSLE siguen en scikit-learn (`eps` del denominador, error con negativos) hasta medir su uso. Quien pase la función de scikit-learn como callable no gana nada. `get_group` (1 a 2%) no merece la pena; volver a medir después. Coordinar con QW-08: los dos tocan `_get_metric`. Medir de nuevo después de #1431, que evita preparar `y_train` en cada fold cuando la métrica no lo usa (un 19% menos en la búsqueda multiserie con `OneStepAheadFold` y 200 series) |

### Observado frente a esperado

- **BUG-03.** Observado (h2o, Ridge, lags=12): `grid_search_forecaster(metric=[r2_score, ...])`
  pone en la fila 0 `alpha=100` (R2 -0,806) y deja `forecaster.estimator.alpha=100`; la
  bayesiana minimiza R2; con `direction="maximize"`, `study.best_value` es 0,857 pero se
  reentrena con `alpha=766,6` (R2 -1,084); el clasificador con `zero_one_loss` reentrena con
  `C=0.0001` (0,333 frente a 0,317). Esperado: maximizar R2 (`alpha=0.01`, R2 0,857) y
  respetar `direction`. Evidencia: `_search.py:514` (`ascending=True if is_regression`).
- **BUG-01.** Observado (h2o, LGBM, lags=12): `fit(y)` y luego `TimeSeriesFold(steps=12)` da
  MAE 0,0291 con 192 de 192 predicciones dentro de `training_range_` y 0 avisos; en el mismo
  periodo, 0,0384 in-sample frente a 0,0966 fuera de muestra (2,5 veces); multiserie 0,0378
  frente a 0,1047. Esperado: aviso con el número y rango de predicciones in-sample que
  recomiende `initial_train_size`; el uso intencionado (`backtesting.ipynb`, "Backtesting on
  training data") sigue funcionando con `suppress_warnings=True`.
- **BUG-02.** Observado (bike_sharing horario, LGBM, steps=24): `fold_stride=1` devuelve MAE
  35,37 (igual al del paso 1) frente a 72,62 sobre todos los pares origen-fecha; solo usa 720
  de 17004 predicciones; `grid_search` elige `lags=24` (real 72,62) en lugar de los lags
  estacionales (real 58,97, un 23% mejor); multiserie igual. Esperado: métrica del horizonte
  configurado (72 a 76) sea cual sea `fold_stride`, o al menos un aviso. La skill pública
  recomienda `fold_stride=1` (`skills/backtesting-configuration/SKILL.md:202-208`).
  Origen de `fold_stride`: https://github.com/skforecast/skforecast/issues/764.
- **BUG-04.** Observado (Chronos-2 small real ajustado 30 pasos): `FoundationModel` ajustado
  da `[10.231 9.79 9.795 ...]`; `ForecasterFoundation` con ese modelo da
  `[12.083 12.797 11.061 ...]`, idéntico al base (diferencia máxima 3,0 en una serie de nivel
  ~10); `adapter._pipeline is None`, 0 avisos; `deepcopy_forecaster` lo vuelve a perder.
  Esperado: predecir con el objeto del usuario (o, como mínimo, avisar). Evidencia:
  `skforecast/foundation/tests/tests_forecaster_foundation/fixtures_forecaster_foundation.py:196-211`
  ya reconoce el problema.
- **BUG-10.** Observado (TabICL real, y alrededor de -16): `predict(context=y, context_exog=exog)`
  sin exog futura da `[5.21 ... 5.06]` frente a `[-17.29 ... -18.22]` sin exog; backtest de 5
  folds MAE 15,5 frente a 3,6; el aviso dice que las columnas se ignoran. Esperado: mismo
  resultado que sin exog, como promete `_foundation_model.py:973-993`.
- **BUG-stats-interval-ignored.** Observado (h2o, `Arima(order=(1, 1, 0))`, medido hoy):
  sin argumentos y con `interval=[0.1, 0.9]` los límites del primer paso son idénticos
  (0,4631 / 1,0723); solo con `alpha=None` sale el 80% (0,5685 / 0,9668). Esperado: el
  intervalo pedido en `interval`. `backtesting_stats` no se ve afectado (`alpha=None` allí).
- **BUG-06.** Observado: Nori falso con serie lognormal, `predict` 1,438 frente a
  `predict_interval` 1,003; MAE de backtesting 1,1693 sin cuantiles frente a 1,0834 con
  cuantiles y sin columna `pred`; TabICL real: el MAE cambia +1,5% (h2o), -4,7%
  (bike_sharing), +3,9% (items_sales), +3,5% (Poisson). Esperado, como en ML:
  `predict_interval()["pred"] == predict()["pred"]` y pedir cuantiles no cambia la métrica.
  TimesFM 2.5 no está afectado.
- **BUG-08.** Observado (800 puntos, m=7, 20 folds, `refit=True`): 20 llamadas a `auto_arima`
  con `freeze_params` True y False (44,8 s y 43,4 s, MAE idéntico 1,559485; 15 órdenes
  distintos en 20 folds); Ets automático 13,1 s y 11,4 s; `grid_search_stats` con 2
  candidatos hace 40 búsquedas (76,4 s). Esperado: una búsqueda; con el arreglo, Arima 8,5 s
  (5,3x) y Ets 0,9 s (15x). Presente desde la 0.20.0.
- **BUG-07.** Observado: MultiIndex creado con `set_index` o `reset_index().set_index()` sobre
  los datos de la guía tiene `levels[1].freq=None` y `fit` lanza "If `series` is a
  dictionary, all series must have ... Found series with no frequency or step"; los rodeos
  dan las mismas predicciones. Esperado: inferir la frecuencia por serie. La guía usa el
  MultiIndex como entrada principal (`independent-multi-time-series-forecasting.ipynb`,
  celdas 5 y 11). No es silencioso, de ahí su posición.
- **BUG-11.** Observado (h2o, `ForecasterEquivalentDate(offset=12)`, medido hoy):
  `interval=[0.1, 0.9]` da `AttributeError: 'ForecasterEquivalentDate' object has no attribute
  'predict_quantiles'`; con `interval_method="conformal"` funciona. Esperado: `ValueError`
  claro o conformal automático.
- **BUG-09.** Observado: `ForecasterStats([Arima, Ets])` con `fold_stride=2` o 1 lanza
  `ValueError: Length of values (2) does not match length of index (1)` tras hacer todos los
  ajustes; Sarimax con `refit=False` y `fold_stride` 1, 3 o 10, o con `refit=3`, lanza
  "`last_window` has to start at the end of the index seen by the forecaster". Esperado: una
  fila por estimador (Arima 6,2311, Ets 8,3975) y predicciones iguales a la referencia
  (diferencia 0,0 con el arreglo).
- **PERF-02.** Observado (1000 puntos, 30 folds): `n_jobs` 1 frente a 3: Arima 6,59 frente a
  2,59 s, Ets automático 23,3 frente a 8,37 s, Sarimax 20,4 frente a 6,73 s, predicciones
  idénticas; el mismo backtest de Ets tardó 194 s con carga externa frente a 24,1 s en
  reposo; `multiprocessing.Pool(4)` con 8 series de Ets 73,6 s frente a 2,56 s con el
  límite BLAS; `setulb` ocupa el 84% del tiempo. Esperado: sin degradación por carga externa
  y paralelismo por defecto donde cada fold es independiente. Precedente aguas arriba:
  https://github.com/scipy/scipy/issues/22438 (optuna 5.0 ya lo parchea).
- **PERF-01.** Observado (chronos-2-small, CPU): 1 serie y 200 folds de 24 pasos, 5,68 s
  frente a 1,68 s agrupado (3,4x), diferencia máxima 3,8e-06 y MAE idéntico; 5 series 1,4x;
  50 series 1,1x (dentro del ruido); con 2000 series simuladas la preparación baja de 17,45 s
  a 2,70 s sin exog y de 26,69 s a 8,13 s con exog. Esperado: una o pocas llamadas
  agrupadas. Solo medido Chronos-2 en CPU; TabICL, TabPFN-TS y Nori solo ganarían con (1)
  y (3).
- **PERF-03.** Observado (series de 400 días, `lags=7`, MAE y MSE, 8 candidatos, prototipo en
  memoria; Python 3.14.3, numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, lightgbm 4.6.0, 10
  núcleos, medido en local el 2026-10-20). Los tiempos son de la llamada completa
  (`grid_search_forecaster_multiseries` o `backtesting_forecaster_multiseries`, con ajuste y
  predicción), mediana de 3; solo cambia la función de la métrica, así que la diferencia es
  toda del cálculo de métricas. Con la versión en numpy que se implementaría (conversión a
  `float` y a 1D, y `ValueError` con longitudes distintas, sin datos o con NaN o infinitos,
  como hoy scikit-learn; mismos resultados y mismos errores comprobados):

  | Series | Caso | scikit-learn | numpy con comprobaciones | Ganancia |
  |--:|:--|--:|--:|--:|
  | 100 | Grid `OneStepAheadFold`, Ridge | 0,35 s | 0,14 s | 2,6x |
  | 100 | Grid `OneStepAheadFold`, LightGBM | 1,81 s | 1,58 s | 1,15x |
  | 1000 | Grid `OneStepAheadFold`, Ridge | 3,36 s | 1,30 s | 2,59x |
  | 1000 | Grid `OneStepAheadFold`, LightGBM | 5,95 s | 4,22 s | 1,41x |
  | 1000 | Backtesting, Ridge, `refit=False` | 1,99 s | 1,69 s | 1,17x |

  (Las filas de 100 series se midieron con la versión sin comprobaciones; con 1000 series, las
  comprobaciones solo añaden unos 0,02 s. Con LightGBM la ganancia varía entre ejecuciones: de
  1,41x a 1,56x con 1000 series.) Métricas idénticas a las de scikit-learn en todos los casos
  (diferencia máxima 0). `dev/roadmap_0.27.0/repro/PERF-03.py` reproduce las filas de 1000
  series y el coste por llamada.

  - **Coste por llamada** (Series de 100 valores): scikit-learn 138 µs, numpy sin
    comprobaciones 5,4 µs, con comprobaciones 6,4 µs. Las comprobaciones no cambian la
    ganancia.
  - **El ahorro en segundos no depende del estimador**: series x métricas x candidatos x ~130
    µs (con 1000 series, 2 métricas y 8 candidatos, 16.000 llamadas, unos 2,1 s, igual con
    Ridge que con LightGBM). Lo que cambia es el porcentaje: más de la mitad de la búsqueda con
    un estimador rápido y muchas series, un tercio con LightGBM y 1000 series, poco con pocas
    series o modelos caros; crece con más métricas o más candidatos.
  - **Con perfilador**, las métricas de scikit-learn ocupan el 41,7% (1 métrica) y el 51,0%
    (3 métricas, más MASE 9,2%) de la búsqueda con `OneStepAheadFold`, y del 5,2 al 10,6% del
    backtesting con Ridge o LightGBM; `get_group`, del 0,9 al 2,1%.
  - **Sin medir todavía**: MASE y RMSSE en numpy (el prototipo no los toca; con 3 métricas la
    ganancia sería algo mayor), RMSE como texto (cuando entre QW-08). MAPE y MSLE se quedan en
    scikit-learn, sin ganancia.

  Esperado: que el coste de la métrica sea el del cálculo, no el de la validación de
  scikit-learn.

---

## 5. Para versiones posteriores

| Item | Estado | Motivo y condición para retomarlo |
|---|---|---|
| BUG-05: grid/random con todas las combinaciones fallidas devuelven un DataFrame vacío y el forecaster sin ajustar | Verificado (`later`), `repro/BUG-05.py` | No es silencioso: cada aviso lleva la causa, y el comportamiento es deliberado y testeado desde la 0.21 (commit c66130afa). Pulido barato: `ValueError(...) from first_exc`, como sklearn y `bayesian_search`. QW-03 (hecho, #1423) y QW-07 (d) quitan los disparadores más frecuentes. Un nombre de métrica inválido ya no llega aquí en stats ni equivalent date (#1431: dan `ValueError` al principio, como el resto de búsquedas). Requiere acuerdo de mantenedores |
| FE-01 completo: función pública `evaluate_backtesting` / `calculate_backtesting_metrics` por fold, paso, serie y grupos | Verificado con veredicto partido | Lente de valor: tras QW-04, QW-09 y QW-10 aporta 2 o 3 líneas por uso y basta una receta. Lente de riesgo: versión reducida M (sin `y`, sin aviso, sin MASE, total distinto del backtesting con folds solapados). Decidir en 0.28 según demanda (única petición: https://github.com/skforecast/skforecast/issues/720); requiere BUG-02 |
| WAPE (de QW-08) | Verificado (`later`) | Función nueva con tests y docs; sMAPE ya cubre las series con ceros |
| `suppress_warnings` en `ForecasterFoundation.predict*` | Hallazgo de QW-02 | Hueco de consistencia con ML |
| Validar el cv una vez antes del bucle de grid | Hallazgo de QW-03 | Junto con BUG-05. La `differentiation` del cv ya no se comprueba (DEP-cv-differentiation la depreca y la ignora); queda el resto de comprobaciones del cv |
| MultiSeries con exog de RangeIndex reseteado da un resultado erróneo en silencio | Hallazgo de QW-06, medido | Poco frecuente; anotar y tratarlo con QW-06 si el diseño lo permite |
| skops en `ForecasterFoundation` | Hallazgo de QW-05 | Posible cuando el backend quede fuera del estado |
| PLAN-CKPT (`finetuned_checkpoint`) y PLAN-TTM (adaptador TinyTimeMixer) | Plan sin código (`dev/PLAN_*.md`), no verificado | BUG-04 es requisito de PLAN-CKPT |
| `grid_search_equivalent_date` multiserie | Fuera del alcance de FE-04 | Después de FE-04 |
| Kalman de Arima estacional más rápido | `work/lens_stats.md` | 14 ms por evaluación, 729 evaluaciones con n=5000 y m=24; esfuerzo L |
| HANDOFF-STATS-LW: `fit(store_last_window=False)` y `predict(last_window=...)` dan `AttributeError` | No verificado | `skforecast/recursive/_forecaster_stats.py:626`; sin decisión |
| HANDOFF-DOUBLE-WARN: doble aviso de NaN en la exog de `predict` de Recursive/Direct | No verificado | Mismo patrón que #1375 en multiserie; junto con QW-06 (mismo camino de `predict`) o con los avisos agregados |
| DISC-1153: asimetría en `RollingFeatures` y errores difíciles de diagnosticar en window features propias | No verificado | https://github.com/skforecast/skforecast/discussions/1153 (14 comentarios); FE-03 reduce parte de la necesidad |
| ISSUE-995: pruning de Optuna en la búsqueda bayesiana | No verificado | https://github.com/skforecast/skforecast/issues/995; pregunta abierta desde 2025-07 |
| TODO-multivariate-gap: formato largo con varios niveles y `gap>0` en multivariante | No verificado | `skforecast/model_selection/_validation.py:1099`; comprobar antes si es un caso real |
| BR-synthetic-data: generador sintético de paneles | WIP, 288 commits por detrás | Rebase y decidir si va en `experimental` o `datasets`; útil para docs y benchmarks |
| ISSUE-974: skforecast como motor de reducción de sktime | No verificado | https://github.com/skforecast/skforecast/issues/974; estratégico, sin actividad |
| Elementos aplazados en el triaje | Ver sección 7 | Avisos agregados, columna de tiempo, calibración conformal foundation, `search_space='auto'`, `static_features`, decaimiento temporal, compilación de Arima |

---

## 6. Ya en curso o identificado

| ID | Referencia | Estado | Acción en 0.27.0 |
|---|---|---|---|
| PR-1400 (rama `perf/ets-numba-compile`) | https://github.com/skforecast/skforecast/pull/1400 | Terminada, etiquetada 0.27.0 | Fusionar: primera compilación de Ets de 12 s a 7 s, mismas estimaciones. Crea la sección 0.27.0 de `releases.md` (conflicto trivial con #1402) |
| PR-1402 (rama `fix/fetch-dataset-website-visits-m5`) | https://github.com/skforecast/skforecast/pull/1402 | Terminada, con tests, etiquetada 0.27.0; sigue abierta y apunta a `0.26.x` (cambiar la base a `0.27.x`) | Fusionar: `website_visits` y `m5` sin `DatetimeIndex`; m5 de más de 15 GB a ~10 GB |
| PR-1407 / ISSUE-1406 | https://github.com/skforecast/skforecast/pull/1407, https://github.com/skforecast/skforecast/pull/1415, https://github.com/skforecast/skforecast/issues/1406 | Resuelto: PR-1407 se cerró el 2026-10-19 y lo sustituyó #1415, fusionado en `0.27.x` el mismo día, que cubre más casos (series omitidas solo en algunos folds, NaN en la última ventana, foundation) | Nada. Un único `MissingValuesWarning` con las series no predichas por causa y fold; las nunca predichas tienen métricas NaN y quedan fuera de los agregados |
| PR-1408 / ISSUE-1357 | https://github.com/skforecast/skforecast/pull/1408, https://github.com/skforecast/skforecast/pull/1412, https://github.com/skforecast/skforecast/issues/1357 | Resuelto: PR-1408 se cerró sin fusionar el 2026-10-18 y lo sustituyó #1412, fusionado en `0.27.x` el mismo día (#1357 cerrada) | Nada que hacer. Los nombres enteros se conservan como enteros: QW-04 debe usar la serie ya preprocesada |
| PR-1389 / ISSUE-1354 | https://github.com/skforecast/skforecast/pull/1389, https://github.com/skforecast/skforecast/issues/1354 | PR externo, sustituido | Hecho en el commit `6a5d933a4` (directo a `0.27.x` el 2026-10-27): `set_in_sample_residuals` con CatBoost y exog categórica aplica la conversión de `fit`. Queda cerrar PR-1389 como sustituido y enlazar #1354 en la nota de la versión |
| PR-1388 / ISSUE-1355 | https://github.com/skforecast/skforecast/pull/1388, https://github.com/skforecast/skforecast/issues/1355 | PR externo | Hecho con #1433 (fusionado en `0.27.x` el 2026-10-26): búsquedas con `lags=None` y solo window features; #1388 se fusionó en una rama intermedia y se conservó en la historia con `-s ours` |
| PR-1387 / ISSUE-1386 | https://github.com/skforecast/skforecast/pull/1387, https://github.com/skforecast/skforecast/issues/1386 | Resuelto: fusionado en `0.27.x` el 2026-10-20 con la autoría del contribuidor, más un commit nuestro (nota de la versión en la 0.27.0 y test de `fig=None`); #1386 cerrada | Nada. `plot_residuals` usa `fig.add_subplot`; ninguna otra función que acepte `fig` o `ax` (`plot_multivariate_time_series_corr`, `plot_prediction_intervals`, `ForecasterRnn.plot_history`) tiene el mismo bug |
| ISSUE-1410 | https://github.com/skforecast/skforecast/issues/1410, https://github.com/skforecast/skforecast/pull/1418 | Resuelto: #1418, fusionado en `0.27.x` el 2026-10-19 | Nada. Una serie con `window_size` valores o menos en el entrenamiento de un fold queda fuera de ese fold; `backtesting_foundation` usa `window_size=0`. FE-04 debe usar la misma frontera |
| Folds sin series | https://github.com/skforecast/skforecast/pull/1427 | Resuelto: fusionado en `0.27.x` el 2026-10-21; sin issue (hallado desde skforecast-ai) | Nada. Un fold sin series que entrenar o predecir se salta y se lista en el `MissingValuesWarning`; si no se puede predecir ni evaluar ningún fold, `ValueError` antes de entrenar. También: `weighted_average` ya no es NaN cuando un nivel se predice sin valores observados (es la métrica que ordena las búsquedas por defecto), y la extracción de folds no retiene el entrenamiento de los folds sin refit (2000 series y 100 folds: pico de memoria de 1383 a 1022 MB) |
| `initial_train_size` contra el span | https://github.com/skforecast/skforecast/pull/1430 | Resuelto: fusionado en `0.27.x` el 2026-10-21; sin issue (hallado desde skforecast-ai) | Nada. `check_backtesting_input` y `check_one_step_ahead_input` validan contra el span de un dict de series, no contra la serie más larga; `OneStepAheadFold.split` devolvía el número de series como final del test con un dict; el test del one-step-ahead deja fuera las series sin valores tras el entrenamiento (antes la búsqueda entera fallaba), con `ValueError` claro si no hay nada que entrenar o evaluar. Pendiente con BUG-search-short-series: tratar por candidato el error de la búsqueda one-step-ahead (nota en su sección 7) |
| ISSUE-1385 (T0) | https://github.com/skforecast/skforecast/issues/1385; `skforecast/foundation/_adapters.py:3724-3729` | Sin PR; no reproducido aquí (sin `tfc-t0`) | Adaptar a `tfc-t0>=0.4` (`predict` sin `context` ni `quantiles`; torch como extra desde 0.6.0) |
| ISSUE-1409 + DISC-1119 + GAP-backtesting-interval-metrics | https://github.com/skforecast/skforecast/issues/1409, https://github.com/skforecast/skforecast/discussions/1119 | Sin PR | Se implementan por separado como QW-09 (`coverage` e `interval_width`, #1409) y QW-10 (columna `step`, D1119) (sección 2). Sin aviso de infracobertura (sección 7) |
| GAP-foundation-backtesting-interval | `skforecast/model_selection/_validation.py:2712-2720` (`backtesting_foundation` solo tiene `quantiles`); `skforecast/model_selection/_search.py:2091` (única búsqueda foundation) | Asimetría de API, comprobada en las firmas; sin medir | Aceptar `interval` en `backtesting_foundation` después de BUG-06; `grid_search_foundation` opcional |
| PR-1341 / ISSUE-1190 (TiRex-2) | https://github.com/skforecast/skforecast/pull/1341, https://github.com/skforecast/skforecast/issues/1190 | PR externo +1376 líneas, blocked; no revisado | Decidir: foco foundation y demanda recurrente (TabPFN, T0 y Nori ya entraron así); revisión L |
| ISSUE-1131 (exógenas as-of) | https://github.com/skforecast/skforecast/issues/1131 | Aplazado en marzo "a la release siguiente"; no verificado | Decidir el alcance: FE-02 cubre las variables nunca conocidas a futuro en foundation; las exógenas que son previsiones publicadas (propuesta `exog_builder`) siguen abiertas (L) |
| DEP-stats-alpha | `skforecast/recursive/_forecaster_stats.py:1047-1055`; `backtesting_stats` | Descartado el 2026-10-19 (decisión 5) | Nada: `alpha` queda como alias de `interval` y pasar los dos da `ValueError` (hecho con BUG-stats-interval-ignored) |
| DEP-python310 | `pyproject.toml:26,65`; `.github/workflows/unit-tests-min-deps.yml:30` | Python 3.10 llega a fin de vida en octubre de 2026 (PEP 619) | Anunciar en 0.27 y retirar en 0.28 (precedente: 3.9 en la 0.19.0) |
| DEP-arima-dictstyle | `skforecast/stats/arima/_arima_base.py:271-290` | `FutureWarning` sin versión desde la 0.22; única deprecación activa | Retirar en 0.27 o fijar la versión; valorar usar `runtime_deprecated` (`skforecast/exceptions/exceptions.py:21`), que no se usa |
| DOC-skills-percentiles-stale | `skills/prediction-intervals/SKILL.md:55-57,70,265`; `references/interval-compatibility.md:23,205`; `skills/complete-api-reference/references/forecaster-methods.md:432,455` (`set_params` marcado como ausente en EquivalentDate) | Desactualizado (percentiles dan `ValueError` desde 0.25) | Corregir junto a QW-07 (b) y regenerar el contexto IA |
| HANDOFF-T2 | `dev/handoff_utils_pendiente.md` (T2) | Pendiente, marcado "para la 0.27"; no reverificado | Implementar: categóricas nativas no llegan al estimador dentro de `CalibratedClassifierCV` o `TransformedTargetRegressor` |
| HANDOFF-T3 / HANDOFF-T4 | `dev/handoff_utils_pendiente.md` (T3, T4) | Pendiente de decisión de Joaquín; no reverificado | Decidir; T3 (Pipeline con LightGBM o CatBoost y exog categórica: `fit` falla) es un error claro en un flujo común |
| ISSUE-1356 | https://github.com/skforecast/skforecast/issues/1356 | Hecho | Hecho con #1435 (fusionado en `0.27.x` el 2026-10-26): el orden de `X_train_series_names_in_` y de las salidas sigue el de entrada con cualquier `encoding` |
| Mantenimiento | rama `ci/minimum-versions-job`; https://github.com/skforecast/skforecast/pull/1404 | Fusionada; PR de dependabot contra `main` | Borrar la rama; fusionar #1404 o reabrirla contra `0.26.x` |

---

## 7. Descartado

**Descartes de la verificación y de fase 1**

- Aviso de infracobertura en backtesting (`ResidualsUsageWarning` si la cobertura queda más de
  0,10 por debajo de la nominal; propuesto en FE-01 y en la primera versión de QW-09):
  descartado definitivamente el 2026-10-18. Con `coverage` en la tabla de métricas no aporta
  información, el umbral y el mínimo de filas son arbitrarios, no está claro si saltaría por
  serie o por agregado, y su consejo (residuos out-of-sample) no basta: LGBM con conformal y
  residuos out-of-sample cubre 0,62 frente al 0,80 nominal.
- DEP-stats-alpha (deprecar `alpha` en favor de `interval` en `ForecasterStats` y
  `backtesting_stats`): descartado el 2026-10-19. Se mantiene como alias para quien viene de
  estadística; la ambigüedad se resuelve con un `ValueError` si se pasan los dos (decisión 5).
- Duplicado "grid/random devuelven un DataFrame vacío" (lente ml-multi): fusionado en BUG-05.
- Comprobar el dominio de las métricas (MSLE, desviaciones de Poisson, Gamma o Tweedie) sobre
  `y` antes de entrenar, para que el backtesting falle antes (comentario externo revisado con
  BUG-03): descartado el 2026-10-22. El error de sklearn ya es claro (en grid y random sale en
  cada aviso "Parameters skipped"), el caso es raro, el dominio también aplica a `y_pred` (que
  no se puede comprobar antes), habría que copiar las reglas de sklearn y no sirve para
  métricas propias. Lo que pierde tiempo en grid y random es BUG-05.
- Duplicado "propagar `differentiation` a `TimeSeriesFold`" (ml-multi): fusionado en QW-03.
- Duplicado "aviso de exog en foundation" (consistency): fusionado en QW-02.
- TODO-reset-forecaster (`_forecaster_recursive.py:1289`, `_forecaster_recursive_multiseries.py:2345`): refactor interno sin valor para el usuario.
- TODO-search-exog-wide (`_search.py:1320,1722`): limpieza interna sin efecto visible.
- TODO-stats-aggregate (`_forecaster_stats.py:219`): funcionalidad nueva en STATS, fuera de foco.
- Vía de modelos locales para `ForecasterStats` en backtesting multiserie (lente de FE-04): funcionalidad nueva en STATS, fuera de foco.
- ISSUE-1084 (https://github.com/skforecast/skforecast/issues/1084, `cudf.pandas`): sin actividad desde abril y sin demanda.
- BR-review-deep-learning e ISSUE-1065: deep learning fuera de alcance por diseño.
- Categorías nuevas de una exógena en `predict` codificadas como NaN sin aviso (lente ml-multi): nicho.
- Fricciones de consistencia en foundation (formato largo con `level` en una sola serie, intervalo por defecto del 80% frente al 90% de ML, nombres `fit(series)`/`context`): cambiarlas rompe código sin ganancia clara.
- Ensembles de forecasters: medido -0,7% de MAE con dos LightGBM; no justifica API nueva.
- Reconciliación jerárquica, diferenciación estacional y ventanas expanding: sin demanda en issues o bajo valor medido (`work/lens_competitors.md`).
- Ya existen (lente competitors): calendario y festivos, `ewm`, Box-Cox y escaladores por serie, diferenciación por serie, estrategias de refit, importancia por horizonte, series nuevas en modelos globales, covariables en foundation.

**Descartados o aplazados en el triaje**

- Agregar los avisos por serie en paneles grandes: esfuerzo M y los avisos responden a errores reales; abordar junto a HANDOFF-DOUBLE-WARN y HANDOFF-T4.
- Columna de tiempo por combinación en grid/random: valor bajo; `bayesian_search` ya lo da en `study.trials_dataframe()["duration"]`.
- Calibración conformal de los cuantiles nativos foundation: esfuerzo M con diseño abierto; primero medir cobertura con QW-09.
- `search_space="auto"`: comodidad de esfuerzo M, rangos opinables y mantenimiento continuo.
- `static_features` en modelos globales: esfuerzo M a L y existe la alternativa de exog en dict.
- Pesos de decaimiento temporal para `weight_func`: nicho; la ganancia depende de la semivida (-2,8% con 365 días, +3,2% con 90) y la alternativa son 3 líneas.
- Compilación numba de Arima en el primer uso (11,5 s): `njit(cache=True)` (`_arima_base.py:578`), se paga una vez por entorno; seguimiento natural de PR-1400.

**Afirmaciones corregidas en la verificación**

- BUG-01: el "23 veces más optimista" no se reproduce; en el mismo periodo son 2,5 veces.
- BUG-06: TimesFM 2.5 no está afectado (timesfm devuelve `q0.5` como punto); lo erróneo es la tabla de `docs/user_guides/foundation-forecasting-models.ipynb:466` ("Mean").
- FE-03: el -4,1% se midió sin lags estacionales y con una clase propia con un bug; frente a la base justa es -2%.
- FE-02: las cifras originales (0,484 / 2,992) no mostraban valor de la covariable; las nuevas son 1,186 / 2,919 / 3,104.
- PERF-01: 3,4x (no 3,6x) con una serie; en CPU solo 1,1x con 50 series.
- PERF-02: `threadpool_limits` cuesta 1,45 ms por llamada, no 0,1 ms.
- QW-07 (d): el caso grave es el DataFrame vacío sin excepción; `backtesting_forecaster` con Foundation ya da `TypeError`.
- BUG-07: la exog larga creada con `set_index` ya funciona; el mensaje sí menciona el MultiIndex.
- FE-04: iterar `backtesting_forecaster` por nivel con el mismo cv reproduciría el desalineamiento de pliegues.

### Cobertura del análisis (lo que no se pudo comprobar)

- **Deep learning** (`deep_learning`, `ForecasterRnn`): fuera de alcance por diseño; no se analizó la rama `review/deep-learning-module`.
- **GitHub**: GraphQL y la API de búsqueda (`gh search`, `/search/issues`) bloqueadas en la sesión. Issues y PR se listaron por REST y MCP; las discusiones se leyeron por REST (34 en total, solo 3 en los últimos 9 meses: D1119, D1153, D1191). La búsqueda de duplicados y de demanda se hizo con la búsqueda semántica del MCP y con los títulos de 1376 issues y PR; puede faltar demanda no reportada con esos términos.
- **Foundation**: solo se ejecutaron con pesos reales `autogluon/chronos-2-small` (CPU) y TabICL 2.2.0 (instalado con `uv pip install --no-deps`). TimesFM 2.5/3.0, Moirai-2, TabPFN-TS, T0, Nori y TS-ICL no se probaron con backend real: BUG-04, QW-05 y BUG-06 se extienden a ellos leyendo el código (mismo patrón), y TimesFM 2.5 en QW-05 se probó con un paquete simulado. BUG-10 en TabPFN-TS no está medido. ISSUE-1385 (T0) no se reprodujo. Sin GPU: las ganancias de PERF-01 en GPU y la carga en CPU de un pickle hecho en GPU no están medidas.
- **Rendimiento**: 4 núcleos compartidos con otros agentes; se repitieron las medidas contaminadas y se descartaron las que no se pudieron repetir (Sarimax con m=24 y n=5000). El problema BLAS solo se midió con scipy 1.18.1 (no con scipy < 1.15).
- **ML**: no hay cifras de cobertura con RandomForest (escenario interrumpido por CPU); `random_search` multiserie no se ejecutó (comparte el patrón de grid); `ForecasterDirectMultiVariate` se revisó poco (lags dict, intervalos, NaN al inicio, `differentiation`). Categóricas en exog no se revisaron (cubiertas por HANDOFF-T2/T3/T4, que no se reverificaron).
- **Competidores**: funciones de mlforecast y darts comprobadas en la web; las de autogluon, sktime y neuralforecast, de memoria. Los experimentos usan un solo dataset por caso (bike_sharing o vic_electricity), así que las mejoras de FE-03 son indicativas.
- **PR externos** (#1341, #1388, #1389, #1407, #1408): inventariados, no revisados línea a línea.
