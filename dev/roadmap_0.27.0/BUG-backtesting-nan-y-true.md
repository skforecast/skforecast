# BUG-backtesting-nan-y-true: NaN en `y` dentro del test del backtesting de una serie

Fila del roadmap: sección 4 de `roadmap.md`. Encontrado en #1452 (sección 12.8, hallazgo 2,
de `BUG-search-short-series-transparency.md`).

Rama `fix/backtesting-nan-y-true` desde `0.27.x` (con #1447, #1452 y #1456).
**Fusionado el 2026-11-07 (#1458).**
Script: `repro/BUG-backtesting-nan-y-true.py` (un escenario por caso; avisos con los filtros
por defecto, primera línea).

## 1. Línea base en `0.27.x` (2026-11-06)

Python 3.14.3, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, LightGBM 4.6.0, statsmodels
0.14.6, optuna 5.0.0.

Datos: 200 días, NaN en las posiciones 150, 170 y 185, `TimeSeriesFold(steps=10,
initial_train_size=140)`: 60 fechas de test, 57 con valor. Ridge con
`dropna_from_series=True`, LightGBM con `dropna_from_series=False`.

Dos simulaciones, sin tocar el paquete:

- **S1 (el esbozo del arreglo):** una métrica que ignora las fechas con `y_true` NaN.
  `_check_nan_predictions_backtesting` va antes de las métricas, así que S1 da lo que daría
  el arreglo.
- **S2 (alinear una serie con el multiserie):** S1 y, además,
  `_check_nan_predictions_backtesting` sustituido por una función que no lanza (las
  predicciones NaN se quedan fuera de las métricas).

| # | Escenario | `0.27.x` | S1 (arreglo) |
|---|---|---|---|
| 1 | Ridge `lags=3`, `refit=False` / `True` | `ValueError: Input contains NaN.` | MAE 1.9563 / 1.98, `n_evaluated` 57 (igual que el multiserie con una serie) |
| 1 | LightGBM `lags=3`, `refit=False` / `True` | `Input contains NaN.` | MAE 1.6074 / 1.8107, 57 (igual que el multiserie) |
| 1 | LightGBM `lags=14`, `refit=False` / `True` | `Input contains NaN.` (y el aviso de `last_window` con NaN, x3) | MAE 1.491 / 1.4681, 57 |
| 1 | Ridge `lags=14`, `refit=False` / `True` | "29 backtesting predictions are NaN at dates with observed values (first: 2022-06-10)..." (#1432) | **el mismo error**: no cambia |
| 1 | LightGBM `lags=3`, `interval=[0.1, 0.9]` | `Input contains NaN.` | MAE 1.6074, `coverage` 0.7544 (43/57), `interval_width` 5.363, 57: las métricas del intervalo ya quitan los NaN |
| 2 | grid, random y bayesian, Ridge, `lags_grid=[3, 14]` | resultados vacíos: "Parameters skipped: 4 of 4 candidates (6 of 6 trials) raise an error" | 2 filas (`lags=3`, 57 puntos); los de `lags=14`, saltados con el error de #1432 |
| 2 | grid, random y bayesian, LightGBM | resultados vacíos | 4 filas (6 trials), todas con 57 puntos; sin aviso de `n_evaluated` |
| 3 | `backtesting_stats`, `Arima` y `Ets` (`refit` se fuerza a `True`), `Sarimax` con `refit=True` | "`y` has missing values." (`ForecasterStats.fit` rechaza NaN) | igual: no llega a las métricas |
| 3 | `backtesting_stats`, `Sarimax`, `refit=False` | `Input contains NaN.` | MAE 2.7433, 57 |
| 3 | `backtesting_foundation`, `chronos-2-small` (en caché), una serie | `Input contains NaN.` | MAE 1.3338, 57 |
| 4 | LightGBM, NaN solo en `y_train` (60 y 100), MAE + MASE + RMSSE | funciona (MASE 0.8715, RMSSE 0.9327, 60): el denominador usa `np.nanmean` | sin cambio |
| 4 | NaN en `y_train` y en el test, o solo en el test, con MAE | `Input contains NaN.` | (lo cubrirán los tests) |
| 4 | NaN solo en el test, **solo MASE y RMSSE** | **funciona**: 0.9094 y 0.9716, 57 (igual que el multiserie) | sin cambio |
| 4 | grid con MASE; `backtesting_stats` (`Sarimax`) con MASE y RMSSE | funcionan | sin cambio |
| 5 | multiserie, 1 serie: Ridge `lags=3`, LightGBM `lags=3` | MAE 1.9563 / 1.6074 (`refit=True`: 1.98 / 1.8107), 57 | |
| 5 | multiserie, 1 serie: Ridge `lags=14` | MAE 1.2976 (`refit=True`: 1.2253), **28 puntos**, aviso "Some levels are not predicted in folds where they have observed values..."; las predicciones tienen 30 filas (la serie no se predice en los folds con NaN en la última ventana) | |
| 5 | multiserie, 2 series | igual para `y`; `y_2` con 60 puntos | |
| extra | todo el test sin valores | `Input contains NaN.` (una serie); "No level predicted in the backtesting has observed values in the test set..." (multiserie) | con S1 sería un error de scikit-learn por 0 muestras |
| extra | `ForecasterDirect`, Ridge `lags=3` | `Input contains NaN.` (mismo `_backtesting_forecaster`) | |

S2 (alineado con el multiserie), Ridge `lags=14`: MAE 1.2976 (`refit=True`: 1.2253) con 28
puntos, igual que el multiserie; las predicciones mantienen las 60 filas, 30 con `pred` NaN.
En las búsquedas, los candidatos de `lags=14` (28 puntos) ganan a los de `lags=3` (57) y salta
el aviso de `n_evaluated` (ver 2.2).

Otras observaciones:

- **MASE y RMSSE ya ignoraban los NaN de `y_true` por accidente:** `np.mean(np.abs(y_true -
  y_pred))` con dos `pd.Series` llama a `Series.mean`, que salta los NaN. Por eso una
  búsqueda con solo MASE funciona y la misma con MAE se queda vacía. Con el arreglo todas las
  métricas reciben los mismos puntos.
- `y_train` de las métricas escaladas con NaN: funciona en la línea base (`np.nanmean` de las
  diferencias); el multiserie hace lo mismo ("y_train includes the intercepted NaNs"). No
  hace falta tocarlo.
- `ForecasterRecursiveClassifier` con NaN en `y` (LightGBM): `TypeError: '<' not supported
  between instances of 'str' and 'float'`. **Corregido en la implementación:** se anotó como
  un fallo del ajuste, pero `fit` funciona con NaN en `y`; el error sale de `accuracy_score`,
  que ordena las clases con el NaN de `y_true`. Es el mismo bug, y el arreglo lo resuelve
  (ver 5).

## 2. Interacciones

### 2.1 #1432 (predicciones NaN)

Sí: con `refit=False` (y con `refit=True`, porque la última ventana es la misma), el NaN de la
posición 150 entra en la última ventana de los folds siguientes si `lags` la alcanza. Con
`lags=14` y Ridge, los folds que empiezan en 160 (ventana 146-159, con el 150), 180 (con el
170) y 190 (con el 185) se predicen NaN: 10 + 9 + 10 = 29 predicciones NaN en fechas con valor
(el 185 no tiene valor), y `_check_nan_predictions_backtesting` lanza su error. Con LightGBM no
pasa (admite NaN).

Que Ridge con `lags=3` pase y con `lags=14` falle depende de **dónde caen los huecos respecto
al inicio de los folds**, no de la longitud de la ventana: los folds empiezan cada 10 fechas
(140, 150, ..., 190) y ningún hueco cae en las 3 fechas anteriores a un inicio. Con un solo
hueco en la posición 159, justo antes del fold que empieza en 160, `lags=3` también falla con
el error de #1432 (10 predicciones NaN; escenario añadido al script). Una ventana más larga
solo hace más probable que alcance un hueco.

Arreglar las métricas **no traslada el fallo de los casos que ya funcionan con
predicciones válidas** (LightGBM, Ridge con ventanas cortas, foundation, Sarimax): pasan del
`Input contains NaN.` a métricas correctas. **Los casos de un estimador sin NaN cuya ventana
alcanza el hueco ya fallan hoy con el error de #1432** (el chequeo va antes de las métricas,
así que nunca llegaban al `Input contains NaN.`) y siguen igual: el arreglo no los mueve de un
error a otro. El error es claro y da la causa y el remedio. En las búsquedas, esos candidatos se
saltan con ese error; el resto se evalúa.

El multiserie llega por otro camino: no predice la serie en los folds con NaN en la última
ventana (`_extract_data_folds_multiseries`), y avisa ("Some levels are not predicted in
folds..."). Alinear una serie con él (S2) sería un cambio de comportamiento: decisión 1.

### 2.2 Aviso de `n_evaluated` de las búsquedas de una serie (#1456)

Con el arreglo y el error de #1432 (decisión 1, opción A), **el aviso no salta** con
`TimeSeriesFold`: los puntos del test son los mismos para todos los candidatos, y las fechas
con `y_true` NaN se descuentan igual en todos (57 en los 4 candidatos y en los 6 trials). Los
candidatos que no pueden predecir el hueco se saltan con error. No hay que adaptar el texto.

Solo saltaría con S2 (opción B), y entonces el texto no encaja:

> 1 of 2 candidates is evaluated on fewer points than the most complete one, so its metrics
> are not comparable. It is the best candidate of the ranking, so the ranking may favor it.
> The rows with missing values in `y` or in its lags are dropped (`dropna_from_series=True`),
> more with larger windows. To compare the candidates on the same points, fill the gaps of `y`
> or remove the candidates with larger windows. ...

La causa que describe es la del `OneStepAheadFold`. Con `TimeSeriesFold` no se quitan filas:
la última ventana tiene NaN y el estimador no los admite (con cualquier
`dropna_from_series`). La acción ("fill the gaps of `y` or remove the candidates with larger
windows") sí vale, y faltaría "or use an estimator that accepts NaN".

## 3. Plan y decisiones

Arreglo (común a todas las opciones): en `_backtesting_forecaster`, tras
`_check_nan_predictions_backtesting` (que ya ignora las fechas con `y_true` NaN), calcular las
métricas (también las del intervalo) sin las filas con `y_true` NaN; `n_evaluated` no cambia
(ya las descuenta). Las predicciones mantienen todas sus filas, con `step` e `y_true` NaN. Lo
mismo en `backtesting_stats` (una y varias estimaciones) y en `backtesting_foundation` con una
serie.

Decisiones (**aprobadas el 2026-11-06** con las recomendaciones: 1A, 2, 3, 4 y 5; ver 5):

1. **Predicciones NaN en fechas con valor en una serie.** (A) Mantener el error de #1432; (B)
   alinearlo con el multiserie: avisar y dejarlas fuera. **Recomendación: A.** El arreglo
   resuelve todos los casos con predicciones válidas sin cambiar comportamiento; con B, una
   búsqueda elegiría `lags=14` con la mitad de los puntos (28 de 57), justo el problema de
   comparabilidad de BUG-search-short-series, y habría que reescribir el aviso de
   `n_evaluated` y la sección de `troubleshooting-common-errors` que describe la diferencia.
2. **Test sin ningún valor observado** (todas las fechas de test con `y_true` NaN). Sin nada,
   saldría un error de scikit-learn por 0 muestras (o de MASE). **Recomendación:** un
   `ValueError` claro en una serie, como el del multiserie ("No date of the test set has an
   observed value in `y`, so the metrics cannot be calculated..."). En las búsquedas, el
   candidato se salta con ese error.
3. **Alcance de `backtesting_stats`.** Solo `Sarimax` con `refit=False` llega a las métricas
   (los demás rechazan NaN en `fit`). **Recomendación:** arreglarlo igual (mismo código, en el
   mismo commit que foundation); no tocar el rechazo de `ForecasterStats.fit`.
4. **Aviso de `n_evaluated`:** con A, no cambia. Un test que compruebe que una búsqueda con
   `TimeSeriesFold` y huecos no avisa.
5. **Docs de IA:** `backtesting-configuration/SKILL.md` y `llms-base.txt` ya dicen que
   `n_evaluated` son los puntos con valor y predicción, pero solo hablan de `y_true` NaN en el
   multiserie. **Recomendación:** añadir una frase: en las cuatro funciones, las fechas sin
   valor observado se quedan en las predicciones (`y_true` NaN) y fuera de las métricas. La
   sección de `troubleshooting-common-errors` sigue siendo correcta con A.

## 4. Fuera de alcance (al roadmap)

- El hallazgo del `TypeError` de `ForecasterRecursiveClassifier` era este mismo bug (ver 1).
  Lo que sí queda fuera, como **BUG-classifier-nan-last-window** (prioridad baja): con un
  estimador que no admite NaN (`LogisticRegression`) y un NaN en la última ventana de un
  fold, el backtesting de `ForecasterRecursiveClassifier` lanza el error genérico de
  scikit-learn (`Input X contains NaN. LogisticRegression does not accept missing values...`)
  en la predicción. Los regresores lineales dan en ese caso predicciones NaN y el error de
  #1432, con la fecha, la causa y el remedio. Ya existía.

## 5. Implementación (2026-11-06)

Ajustes aprobados con el plan: un solo helper para las métricas de una serie, tests de MASE y
RMSSE que no dependan de que `np.mean` sobre una `pd.Series` ignore los NaN, la nota sobre
`lags=3` (2.1) y el clasificador al roadmap (ver 4: era el mismo bug).

- **Helper** `_calculate_metrics_backtesting` (`model_selection/_utils.py`), usado en los 4
  sitios que repetían chequeo, métricas y `n_evaluated`: `_backtesting_forecaster`,
  `_backtesting_stats` con uno y con varios estimadores (uno por `estimator_id`) y
  `backtesting_foundation` con una serie (`pred_column='q_0.5'` con `quantiles`). Orden: el
  chequeo de #1432 (sin cambios), quitar las filas con `y_true` NaN, `ValueError` si no queda
  ninguna, las métricas, las del intervalo y `n_evaluated`. Las predicciones no se tocan.
- **Error sin valores observados:** "The series has no observed values in the test set of the
  folds, so the metrics cannot be calculated. Review `initial_train_size` of `cv`, or the span
  of the series." Misma estructura que el del multiserie, sin `levels`, `fixed_train_size` ni
  `refit`, que en una serie no cambian las fechas del test. Sale tras predecir (el del
  multiserie, antes de ajustar).
- **Tests** (todos fallan sin el arreglo, comprobado en un worktree de `0.27.x`): el helper
  (valores a mano con MAE, MASE, RMSSE, cobertura y anchura; una métrica espía que comprueba
  que no recibe NaN; las predicciones sin modificar; los dos errores y su orden);
  `backtesting_forecaster` con `refit` False y True (MAE, MASE y RMSSE a mano, con `y_train`
  con NaN con `refit=True`, y la métrica espía), el `ValueError` sin valores y
  `ForecasterRecursiveClassifier` (`accuracy_score` a mano); `backtesting_stats` con `Sarimax`
  y `refit=False`, uno y dos estimadores; el test de `backtesting_foundation` que usaba una
  métrica que ignora NaN ahora usa `mean_absolute_error` y comprueba el valor; grid (vía
  `_evaluate_grid_hyperparameters`) y bayesian con `TimeSeriesFold` y huecos: resultados
  completos, `n_evaluated` igual en todos y sin el aviso de `n_evaluated`.
- **Docs:** entrada en Fixed (sin enlace hasta que exista el PR); `backtesting-configuration`
  y `llms-base.txt` dicen que, en las cuatro funciones, las fechas sin valor quedan en las
  predicciones con `y_true` NaN y fuera de las métricas. `troubleshooting-common-errors` no
  cambia (decisión 1A).

### 5.1 Resultado (script, con el arreglo)

| Escenario | `0.27.x` | Con el arreglo |
|---|---|---|
| Ridge `lags=3`, `refit` False / True | `Input contains NaN.` | MAE 1.9563 / 1.98, 57 puntos (como el multiserie) |
| LightGBM `lags=3`, `refit` False / True | `Input contains NaN.` | MAE 1.6074 / 1.8107, 57 |
| LightGBM `lags=14`, `refit` False / True | `Input contains NaN.` | MAE 1.491 / 1.4681, 57 |
| Ridge `lags=14`, `refit` False / True | error de #1432 (29 predicciones NaN) | igual |
| Ridge `lags=3`, un hueco en la posición 159 | error de #1432 (10 predicciones NaN) | igual |
| Test sin valores observados | `Input contains NaN.` | `ValueError` "The series has no observed values in the test set of the folds..." |
| LightGBM `lags=3` con `interval` | `Input contains NaN.` | MAE 1.6074, `coverage` 0.7544, `interval_width` 5.363, 57 |
| `ForecasterRecursiveClassifier` (LightGBM) | `TypeError: '<' not supported...` | `accuracy_score` 0.614, 57 |
| grid y random, Ridge `[3, 14]` | vacíos ("4 of 4 candidates raise an error") | 2 filas (`lags=3`, 57); los de 14, saltados con el error de #1432 |
| bayesian, Ridge | vacío ("6 of 6 trials") | 2 filas (`lags=3`); 4 trials saltados |
| grid, random y bayesian, LightGBM | vacíos | 4 / 4 / 6 filas, todas con 57 puntos; sin aviso de `n_evaluated` |
| `backtesting_stats`, `Sarimax` `refit=False` | `Input contains NaN.` | MAE 2.7433, 57 |
| `backtesting_stats`, `Arima`, `Ets`, `Sarimax` `refit=True` | "`y` has missing values." | igual (`ForecasterStats.fit`) |
| `backtesting_foundation`, `chronos-2-small` | `Input contains NaN.` | MAE 1.3338, 57 |
| MAE + MASE + RMSSE, NaN en el test (y en `y_train`) | `Input contains NaN.` | `refit=False`: 1.6074 / 0.9094 / 0.9716 (con NaN también en `y_train`: 1.5814 / 0.8928 / 0.9491), 57 |
| Solo MASE y RMSSE; grid con MASE; NaN solo en `y_train` | funcionaban | igual, mismos valores (MASE 0.9094 = el de arriba y el del multiserie) |
| Multiserie (1 y 2 series) | funciona | igual |

S2 (simulación de alinear con el multiserie, no implementada) da lo mismo que en la línea base:
Ridge `lags=14` con 28 puntos y las búsquedas con el aviso de `n_evaluated`.
