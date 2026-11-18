# Revisión crítica de T1: atajo de `StandardScaler` en `transform_numpy`

**Fecha:** 2026-11-18.
**Qué se revisa:** la tarea T1 de `dev/preprocessing_speedups/PLAN.md` (líneas 55 a 120), antes de implementar nada.
**Base:** rama `perf/transform-numpy-known-scalers` en `2d396cf`. `git diff c694e19 HEAD -- skforecast/` está vacío, así que los números de línea del plan siguen siendo válidos.
**Alcance:** no se ha tocado `skforecast/` ni `PLAN.md`. Todo lo ejecutado está en `dev/preprocessing_speedups/review_t1_*.py` y sus salidas en `results/review_t1_*`.

> **Estado posterior (2026-11-18).** Joaquín revisó los hallazgos y decidió:
> 1. quitar la condición de `feature_names_in_` (hallazgo 1);
> 2. buscar `inf` en la entrada antes de calcular (hallazgo 8);
> 3. sacar la validación one-step-ahead multiserie a una tarea propia, T4 (hallazgo 7);
> 4. adoptar los criterios de aceptación propuestos (hallazgo 2).
>
> `PLAN.md` ya incorpora esas decisiones y el resto de hallazgos, y `prototypes.fast_transform_numpy` se alineó con el diseño revisado. Este documento y las salidas de `results/review_t1_*` describen el plan y el prototipo tal como estaban en el commit `9d02780`. Al ejecutar hoy los scripts, «PLAN» sigue siendo el plan anterior a la revisión y PROTO coincide con PROPOSED-B (0 diferencias con el camino actual).

## Veredicto

**T1 es viable y su aritmética es correcta, pero el plan no debe implementarse tal como está escrito.** Hay que corregir tres cosas antes de empezar:

1. La condición «no tiene `feature_names_in_`» (línea 78) deja el atajo sin efecto en 4 de los 6 forecasters, y el prototipo con el que se midió no la aplica (hallazgo 1).
2. Las condiciones no cierran el contrato de la entrada: con un array vacío, una `Series` o ciertos `ndim`, el atajo devuelve un resultado donde hoy hay un error. El caso del array vacío se reproduce por la API pública (hallazgos 3, 4, 9 y 10).
3. Los criterios de aceptación de rendimiento no se sostienen: el ×1,5 con Ridge está en el techo de lo medible y el ×1,18 de backtesting es ruido (hallazgo 2).

Lo que sí se confirma: en float64, el atajo es idéntico byte a byte al camino actual en 1.176 casos, con scikit-learn 1.6.0, 1.7.2, 1.8.0 y 1.9.1, y el `ValueError` de `inf` es el mismo (sección «Lo que se confirma»).

## Entornos

| Entorno | Python | numpy | pandas | scikit-learn | numba | Otros |
|---|---|---|---|---|---|---|
| Principal (`~/.venvs/skforecast`) | 3.12.3 | 2.5.5 | 3.1.1 | 1.9.1 | 0.68.1 | lightgbm 4.7.0, xgboost 3.4.1, skops 0.16.0, pyarrow 26.0.1, sin keras |
| Mínimos (`~/.venvs/skforecast-min`, `--resolution lowest-direct`) | 3.11.18 | 2.0.0 | 2.3.0 | 1.6.0 | 0.60.0 | lightgbm 4.4.0, skops 0.14.0, keras 3.3.0, torch 2.4.0+cpu |
| scikit-learn 1.7 (`~/.venvs/skforecast-sk1.7`) | 3.12.3 | 2.5.5 | 3.1.1 | 1.7.2 | 0.68.1 | solo para el contrato |
| scikit-learn 1.8 (`~/.venvs/skforecast-sk1.8`) | 3.12.3 | 2.5.5 | 3.1.1 | 1.8.0 | 0.68.1 | solo para el contrato |
| skforecast 0.27.0 (`~/.venvs/skforecast-0270`, de PyPI) | 3.12.3 | 2.5.5 | 2.3.3 | 1.9.1 | 0.68.1 | skops 0.16.0; solo para guardar forecasters |

CPU: Intel Xeon a 2,10 GHz, 4 núcleos lógicos, contenedor en la nube. El entorno principal no se ha modificado. Todas las cifras de tiempo son del entorno principal salvo que se diga otra cosa.

Las tres variantes que se comparan con el `transform_numpy` actual (definidas en `review_t1_common.py`):

- **PLAN:** T1 exactamente como está escrito en `PLAN.md` (puntos 1 a 5).
- **PROTO:** `prototypes.fast_transform_numpy`, el prototipo con el que se midió el plan.
- **PROPOSED:** las condiciones que propone esta revisión, con el chequeo de `inf` decidido. **PROPOSED-B** es igual, pero busca `inf` en la entrada antes de calcular (hallazgo 8).

---

## Hallazgos de gravedad alta

### 1. La condición «sin `feature_names_in_`» anula el atajo en 4 de los 6 forecasters, y el prototipo no la aplica

**Qué pasa.** El punto 2 del diseño (`PLAN.md:78`) exige que el escalador no tenga `feature_names_in_`. Pero solo los dos forecasters multiserie ajustan el escalador con un array:

| Forecaster | Cómo ajusta el escalador | `feature_names_in_` | ¿Atajo según el plan? |
|---|---|---|---|
| `ForecasterRecursive` | `transform_dataframe` (`recursive/_forecaster_recursive.py:819`) | `['y']` | No |
| `ForecasterDirect` | `transform_dataframe` (`direct/_forecaster_direct.py:1033`) | `['y']` | No |
| `ForecasterStats` | `transform_series` (`recursive/_forecaster_stats.py:563`) | `['y']` | No |
| `ForecasterRnn` | `transform_series` (`deep_learning/_forecaster_rnn.py:800`) | nombre de la serie | No |
| `ForecasterRecursiveMultiSeries` | `transform_numpy(fit=True)` (`recursive/_forecaster_recursive_multiseries.py:997`) | no tiene | Sí |
| `ForecasterDirectMultiVariate` | `transform_numpy(fit=True)` (`direct/_forecaster_direct_multivariate.py:1249`) | no tiene | Sí |

El prototipo (`prototypes.py:130-138`) no comprueba `feature_names_in_`: solo `n_features_in_ == 1` y `hasattr(transformer, "scale_")`. Por tanto:

- las cifras del plan y `check_equivalence.py` validan el prototipo, no el diseño escrito;
- el caso 9 de `check_equivalence.py` (`ForecasterRecursive`) pasa por el atajo con el prototipo y no pasaría con el plan;
- «El cambio dentro de `transform_numpy` beneficia a todos sin tocarlos» (`PLAN.md:69`), la validación V4 (`PLAN.md:265`) y la nota de versión («`transformer_series` / `transformer_y`», `PLAN.md:120`) serían falsas para `transformer_y`.

**Evidencia.**

- Inventario en ejecución (`review_t1_inventory.py`, salida en `results/review_t1_inventory_latest.txt` y `_min.txt`): en `ForecasterRecursive`, `ForecasterDirect`, `ForecasterStats` y `ForecasterRnn`, todas las llamadas salen `PLAN=no (no feature_names_in_) PROTOTYPE=yes`.
- Llamadas en las que aplica cada variante (`review_t1_forecasters.py`): `ForecasterRecursive`, PLAN 0 de 55, PROTO 55 de 55; `ForecasterStats`, PLAN 0 de 2; `ForecasterRnn`, PLAN 0 de 19.
- Coste de mantener la condición (`review_t1_bench.py others`, 15 rondas intercaladas, mediana):

  | Flujo | Actual | PLAN | PROTO | PROPOSED |
  |---|---|---|---|---|
  | `ForecasterRecursive.predict(24)` | 1,06 ms | 1,05 ms (×1,00) | 0,66 ms (×1,60) | 0,66 ms (×1,60) |
  | `ForecasterRecursive`, backtesting de 1.000 folds sin refit | 396 ms | 398 ms (×1,00) | 175 ms (×2,26) | 180 ms (×2,20) |
  | `ForecasterDirect.predict(5)` | 1,07 ms | 1,06 ms (×1,01) | 0,62 ms (×1,74) | 0,63 ms (×1,71) |
  | `ForecasterDirect`, backtesting de 200 folds | 95,5 ms | 92,9 ms (×1,03) | 48,5 ms (×1,97) | 50,5 ms (×1,89) |
  | `ForecasterStats.predict(24)` (Arima) | 1,02 ms | 1,02 ms (×1,00) | 0,77 ms (×1,33) | 0,75 ms (×1,35) |

- La condición no hace falta para la identidad. Con un escalador ajustado con nombres y una entrada numpy, scikit-learn solo emite el aviso «X does not have valid feature names», que `transform_numpy` ya silencia (`skforecast/utils/utils.py:2647-2652`), e `inverse_transform` ni siquiera valida los nombres (usa `check_array`, no `validate_data`). El resultado es idéntico byte a byte con y sin nombres (`review_t1_contract.py`, escalador «fitted with a DataFrame», 0 diferencias; `review_t1_forecasters.py`, 15 configuraciones idénticas).

**Cambio propuesto en `PLAN.md`.** Decisión de Joaquín, la recomendada primero:

- **(Recomendada) Quitar la condición** de la línea 78 y dejar solo `n_features_in_ == 1`. Añadir al punto 2 la frase: «un escalador ajustado con nombres de columna también usa el atajo: la entrada de `transform_numpy` es siempre un array y scikit-learn solo emite un aviso que la función ya silencia».
- Alternativa: mantenerla y reescribir el alcance: T1 solo acelera `ForecasterRecursiveMultiSeries` y `ForecasterDirectMultiVariate`. Habría que corregir las líneas 69, 120 y 265 y la fila de la hoja de ruta (línea 339).

En los dos casos, alinear el prototipo con el plan antes de medir nada más.

### 2. Los criterios de aceptación de rendimiento no se sostienen

**Qué pasa.** Son tres problemas distintos.

**2a. El ×1,5 con Ridge está en el techo de lo medible.** En M, las 600 llamadas a `transform_numpy` de un `predict_168` suman entre el 29 % y el 30 % del tiempo. Aunque el atajo costase cero, la ganancia máxima sería de ×1,40 a ×1,43.

| `ForecasterRecursiveMultiSeries`, M, 300 series | Total | Dentro de `transform_numpy` | Techo |
|---|---|---|---|
| `predict_168`, Ridge | 153,2 ms | 44,2 ms (28,8 %), 600 llamadas, 74 µs por llamada | ×1,40 |
| `predict_24`, Ridge | 183,2 ms | 57,5 ms (31,4 %), 600 llamadas | ×1,46 |
| `interval_conformal_24`, Ridge | 285,0 ms | 58,1 ms (20,4 %), 600 llamadas | ×1,26 |
| `interval_bootstrapping_24` (`n_boot=100`), Ridge | 698,1 ms | 132,7 ms (19,0 %), 1.200 llamadas | ×1,23 |
| `predict_168`, LightGBM | 252,2 ms | 52,9 ms (21,0 %), 600 llamadas | ×1,27 |

(`results/review_t1_macro_M_ridge.txt` y `_lgbm.txt`, mejor de 3 ejecuciones. En una ejecución anterior de la misma medida con Ridge, `predict_168` dio 55,0 de 181,7 ms, el 30,3 %, techo ×1,43.)

Medido de extremo a extremo con las variantes intercaladas en cada ronda (`review_t1_bench.py macro`, razón actual / variante pareada por ronda, mediana y cuartiles):

| Flujo | Rondas | PLAN | PROTO | PROPOSED | Criterio del plan |
|---|---|---|---|---|---|
| `predict_168`, Ridge | 25 | ×1,44 [1,35; 1,65] | ×1,44 [1,32; 1,76] | ×1,46 [1,29; 1,78] | ≥ ×1,5: **no se cumple** |
| `predict_24`, Ridge | 25 | ×1,54 [1,47; 1,93] | ×1,69 [1,46; 2,10] | ×1,62 [1,40; 1,98] | |
| `interval_conformal_24`, Ridge | 25 | ×1,22 [1,11; 1,34] | ×1,33 [1,05; 1,56] | ×1,27 [1,10; 1,37] | (el plan dice ×1,56) |
| `predict_168`, LightGBM | 15 | ×1,27 [1,19; 1,33] | ×1,22 [1,17; 1,33] | ×1,24 [1,18; 1,35] | ≥ ×1,1: se cumple |
| `predict_24`, LightGBM | 15 | ×1,39 [1,33; 1,78] | ×1,49 [1,27; 1,77] | ×1,57 [1,25; 1,72] | |

Los cuartiles son anchos: en esta máquina el ruido de una ronda a otra (±15 %) es del orden del margen entre el umbral y el techo. Parte de las rondas supera el techo calculado, lo que indica ruido o efectos de segundo orden (menos objetos temporales), no una ganancia mayor garantizada.

**2b. El ×1,77 de referencia sale de una línea base ruidosa.** En `results/bench_M_ridge.json`, `predict_168` actual es `[0,168; 0,198; 0,194]` s y con P2 `[0,109; 0,109; 0,110]` s. Con el mínimo del actual sería ×1,54, no ×1,77. Al repetir el mismo `bench.py` hoy (`results/review_t1_bench_py_M_ridge.txt`): 0,194 s frente a 0,143 s, ×1,36, con una dispersión de 0,042 s.

**2c. El ×1,18 de backtesting (`PLAN.md:64`) es ruido.** Las llamadas que el atajo puede sustituir son una parte mínima del backtesting (`review_t1_bench.py backtesting`, misma llamada que `bench.py`, M, Ridge):

| Backtesting | Total | Llamadas sustituibles | Techo | Lo que dice el plan | `bench.py` repetido hoy |
|---|---|---|---|---|---|
| `refit=False` | 6,54 s | 4.200 llamadas, 0,377 s (5,8 %) | ×1,06 | ×1,00 | ×0,90 |
| `refit=True` | 22,31 s | 4.200 llamadas, 0,540 s (2,4 %) | ×1,025 | ×1,18 | ×0,89 |

`bench.py` mide el backtesting con una sola repetición, y dos ejecuciones dan ×1,18 y ×0,89 para el mismo cambio. Con refit, además, el ajuste de los escaladores (`fit=True`, 2.100 llamadas, 1,44 s) cuesta casi el triple que la parte que cubre T1, y queda fuera del atajo.

**Cambio propuesto en `PLAN.md`.**

- Línea 64: sustituir «Backtesting con refit (Ridge): de 22,7 a 19,2 s (×1,18)» por «Backtesting: sin ganancia apreciable (las llamadas que cubre el atajo son el 2,4 % con refit y el 5,8 % sin refit en M)». Quitar «backtesting» de la nota de versión (línea 120) o limitarlo a «backtesting de una serie con muchos folds» si se adopta el hallazgo 1 (×2,2 con 1.000 folds).
- Líneas 61 a 63 y fila de la hoja de ruta (339): sustituir ×1,77 y ×1,16 por «×1,4 a ×1,5 con Ridge y ×1,2 a ×1,3 con LightGBM (mediana pareada de 15 a 25 rondas)», y el conformal con Ridge por ×1,2 a ×1,3.
- Líneas 115 a 118, criterios de aceptación. Sustituir el segundo por criterios que no dependan del ruido:
  1. el atajo se aplica en el 100 % de las llamadas elegibles de los flujos de referencia (contador: 600 de 600 en `predict_168` de M; 2.000 de 2.000 en el backtesting de `ForecasterRecursive`);
  2. el coste por llamada de `transform_numpy` con un array 1D de 168 valores baja de 55 a 92 µs a menos de 10 µs;
  3. de extremo a extremo, con las variantes intercaladas y al menos 15 rondas, la mediana de la razón pareada de `predict_168` en M es de al menos ×1,3 con Ridge y ×1,1 con LightGBM.
- Sección 4: añadir que `bench.py` no sirve para decidir diferencias de menos de un 20 % en `predict` ni de un 10 % en backtesting, y remitir a `review_t1_bench.py` (o incorporar a `bench.py` el intercalado y la razón pareada).

### 3. Un array vacío deja de ser un error (regresión reproducible por la API pública)

**Qué pasa.** scikit-learn rechaza un array sin muestras (`check_array`, `ensure_min_samples=1`). El atajo, sin una condición de tamaño, devuelve un array vacío. Ni el plan ni el prototipo lo contemplan, y contradice la frase de la sección de riesgos «el peor caso es "sin ganancia", no "resultado distinto"» (`PLAN.md:349`).

**Evidencia** (`review_t1_empty_inputs.py`, `results/review_t1_empty_inputs.txt`):

```text
ForecasterRecursive.set_out_sample_residuals(empty arrays)
    current:  ValueError: Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required by StandardScaler.
    PROTO:    no error
ForecasterRecursiveMultiSeries.set_out_sample_residuals(empty arrays)
    current:  ValueError: Found array with 0 sample(s) (shape=(0, 1)) ...
    PROTO:    no error
          out_sample_residuals_ after the call: {'s0': (0,), 's1': None, '_unknown_level': (0,)}
          predict_interval(use_in_sample_residuals=False): IndexError: index -1 is out of bounds for axis 0 with size 0
```

Con el atajo, el forecaster queda con residuos vacíos y el fallo aparece más tarde, en `predict_interval`, con un `IndexError` que no señala la causa. En `transform_numpy` directamente: arrays `(0,)`, `(0, 3)` y `(3, 0)` dan `ValueError` hoy y un array vacío con PLAN y PROTO (`review_t1_contract.py`, sección 5).

**Cambio propuesto en `PLAN.md`.** Añadir al punto 2 la condición `array.size > 0` y, a la lista de tests, «array vacío (1D y 2D): el mismo `ValueError` que hoy» y «`set_out_sample_residuals` con arrays vacíos sigue lanzando `ValueError`» en un forecaster.

Nota aparte (fuera de T1): que la validación de `set_out_sample_residuals` dependa de un error de scikit-learn dentro del escalador es frágil. Sin `transformer_y` el array vacío ya se acepta hoy. Merece su propia comprobación de entrada, en otra tarea.

---

## Hallazgos de gravedad media

### 4. El punto 1 coloca el atajo antes de la comprobación de tipo de `array`, y el punto 2 no exige un `ndarray`

**Qué pasa.** Según `PLAN.md:73`, `transform_numpy` llama al atajo «justo después de `if transformer is None`», es decir, antes de la comprobación de `skforecast/utils/utils.py:2623-2626`. Las condiciones del punto 2 usan `array.dtype` y `array.ndim` sin exigir que `array` sea un `ndarray`. `transform_numpy` es una función pública documentada (`docs/api/utils.md:20`).

**Evidencia** (`review_t1_contract.py`, sección 5, igual en los cuatro entornos):

| `array` (escalador ajustado, `fit=False`) | Hoy | PLAN literal | PROTO |
|---|---|---|---|
| `pd.Series` float64 | `TypeError: array argument must be a numpy ndarray` | devuelve una `Series` | igual que hoy |
| lista | el mismo `TypeError` | `AttributeError: 'list' object has no attribute 'dtype'` | igual que hoy |

El test actual no lo detectaría: `test_transform_numpy_TypeError_when_array_is_not_numpy_ndarray` (`skforecast/utils/tests/tests_utils/test_transform_numpy.py:33-46`) usa una `Series` int64 con `fit=True`, que nunca entra en el atajo.

**Cambio propuesto en `PLAN.md`.** En el punto 1, «`transform_numpy` la llama después de la comprobación de tipo de `array`». En el punto 2, añadir `type(array) is np.ndarray` (cubre también el hallazgo 10). En los tests, un caso con una `Series` float64 y una lista, escalador ajustado y `fit=False`, en los dos sentidos.

### 5. Los tests propuestos pasan aunque el atajo no se use nunca

**Qué pasa.** Todos los casos de `PLAN.md:100-111` comparan la salida con la de scikit-learn. Si las condiciones no se cumplen nunca, la función cae en el camino actual y todos pasan. Es justo lo que ocurriría con el hallazgo 1: implementar el plan literal dejaría los tests en verde y el atajo desactivado en `ForecasterRecursive`. Tampoco hay ningún test que falle si el atajo se aplica donde no debe y da, por casualidad, el mismo valor.

**Cambio propuesto en `PLAN.md`.** Añadir un test del camino tomado, en los dos sentidos:

- elegible: con `monkeypatch`, `StandardScaler.transform` e `inverse_transform` lanzan un error centinela, y `transform_numpy` sigue devolviendo el resultado (el atajo no los llama);
- no elegible (subclase, `Pipeline`, float32, int64, array vacío, 2D sin inversa, escalador con 2 columnas, `fit=True`): con el mismo `monkeypatch`, se propaga el error centinela.

El resto de huecos está en la sección «Tests».

### 6. El kit de validación no cubre lo que el plan le atribuye

**Qué pasa.**

- `prototypes._MODULES` (`prototypes.py:112-118`) no parchea `skforecast.recursive._forecaster_stats` ni `skforecast.deep_learning._forecaster_rnn`, así que `use_fast_scaler` nunca ha ejercitado esos dos forecasters. `PLAN.md:265` dice que T1 los cubre.
- `check_equivalence.py` solo prueba `ForecasterRecursiveMultiSeries` y un `predict` de `ForecasterRecursive`. No prueba `ForecasterDirect`, `ForecasterDirectMultiVariate`, `ForecasterStats`, `ForecasterRnn`, la diferenciación, `encoding=None`, un dict de transformadores, series con NaN, `set_out_sample_residuals`, `predict_quantiles`, la búsqueda one-step-ahead ni ningún caso de error.
- Compara con `np.testing.assert_array_equal` (`check_equivalence.py:33`), que no distingue `-0.0` de `0.0` (hallazgo 12).
- No comprueba que el atajo se aplique.

**Evidencia de que la equivalencia se mantiene al ampliarlo.** `review_t1_forecasters.py` construye y ejecuta desde cero cada configuración con el camino actual y con cada variante, y compara por bytes las salidas de hasta 11 flujos (ajuste, predicción, intervalos por bootstrapping y conformal, cuantiles, residuos fuera de muestra, backtesting con y sin intervalos, búsqueda one-step-ahead y `inf` en `last_window`):

- entorno principal: 14 configuraciones por 4 variantes, 56 de 56 idénticas;
- entorno de mínimos (incluye `ForecasterRnn` con keras 3.3.0): 15 por 4, 60 de 60 idénticas.

**Cambio propuesto en `PLAN.md`.** En la sección 4, sustituir `check_equivalence.py` por `review_t1_forecasters.py` como comprobación de T1 (o fusionarlos), añadir los dos módulos a `_MODULES`, y marcar V4 como comprobada en equivalencia para los seis forecasters.

### 7. Hay llamadas al escalador, serie a serie, que no pasan por `transform_numpy`

**Qué pasa.** La validación one-step-ahead de `ForecasterRecursiveMultiSeries` llama a `transformer.inverse_transform` directamente, con un `DataFrame`, dos veces por serie y candidato (`skforecast/model_selection/_utils.py:3046` y `:3049`). T1 no las alcanza, y son más caras que las de `transform_numpy` porque la entrada es pandas. El inventario del plan (`PLAN.md:69`) cuenta solo las llamadas a `transform_numpy`.

**Evidencia** (`review_t1_one_step_ahead.py`, `grid_search_forecaster_multiseries` con `OneStepAheadFold`, M, Ridge, 3 candidatos): 9,70 s en total; 1.800 llamadas directas a `StandardScaler.inverse_transform`, 1,06 s (10,9 % de la búsqueda, 587 µs por llamada). La parte crece con el número de candidatos, porque `X_train` se construye una sola vez.

**Cambio propuesto en `PLAN.md`.** Añadir al inventario estas dos llamadas y decidir si entran en T1: sustituirlas por `transform_numpy(array=predictions_level["y_true"].to_numpy(), ..., inverse_transform=True)` son dos líneas en `model_selection/_utils.py` y dejarían el coste en unos 3 µs por llamada. Si entran, hace falta un test de identidad de la búsqueda one-step-ahead multiserie. Si no, anotarlo como tarea aparte en la hoja de ruta.

---

## Hallazgos de gravedad baja

### 8. `out.sum()` emite un `RuntimeWarning` que scikit-learn silencia

**Qué pasa.** Fallo concreto del chequeo decidido en el punto 5. Con valores finitos cuya suma desborda, `out.sum()` emite `RuntimeWarning: overflow encountered in reduce`. scikit-learn hace la misma suma dentro de `np.errstate(over="ignore")` (`sklearn/utils/validation.py:132` en 1.9.1, `:115` en 1.6.0), así que hoy no hay aviso. Con `warnings.simplefilter("error")`, el atajo lanzaría una excepción donde hoy hay un resultado.

**Evidencia** (`review_t1_contract.py`, sección 4): con `np.full(48, 1.5e308)`, PLAN y PROTO devuelven el mismo resultado con el aviso; el camino actual, sin él. Hacen falta magnitudes del orden de 1e306, así que el impacto real es mínimo.

**Dos arreglos, con su coste** (`review_t1_bench.py micro`, µs por llamada del chequeo solo):

| Entrada | Decidido | Decidido con `np.errstate(over="ignore")` | `np.isinf(array).any()` antes de calcular | Versión anterior del plan |
|---|---|---|---|---|
| 1D, 48 valores | 0,48 | 1,53 | 0,65 | 1,70 |
| 1D, 168 valores | 0,40 | 1,51 | 0,61 | 1,55 |
| 1D, 48 valores con NaN | 1,18 | 2,35 | 0,75 | 1,48 |
| 1D, 168 valores con NaN | 1,03 | 2,37 | 0,81 | 1,85 |
| 2D (168, 250) | 9,47 | 11,02 | 8,82 | 42,83 |
| 2D (168, 250) con NaN | 17,92 | 19,41 | 8,45 | 99,44 |

Las cifras del plan para el chequeo decidido (0,6 µs sin NaN y 1,5 µs con NaN, `PLAN.md:96`) se confirman. `np.errstate` cuesta más que el propio chequeo: entre 1,0 y 1,5 µs en los dos entornos (numpy 2.5.5 y 2.0.0).

**Cambio propuesto en `PLAN.md`.** Decisión de Joaquín sobre un detalle del punto 5, no sobre la decisión en sí (volver al camino actual cuando hay `inf`):

- **(Recomendada) Buscar `inf` en la entrada antes de calcular:** `if np.isinf(array).any(): return None`. Coste plano (0,6 a 0,8 µs en 1D, haya o no NaN; la mitad que el decidido en 2D con NaN), ningún aviso, ninguna aritmética en el camino del error y una sola condición que testear. Equivale exactamente a la condición de scikit-learn (lanza si y solo si la entrada tiene `inf`).
- Alternativa: mantener el chequeo decidido dentro de `np.errstate(over="ignore")`.
- Alternativa: mantenerlo tal cual y documentar el aviso.

Las tres dan el mismo `ValueError`. En cualquier caso, añadir el test «valores finitos enormes: mismo resultado y mismos avisos que hoy».

### 9. «`inverse_transform=True` con cualquier `ndim`» cambia errores por resultados

**Qué pasa.** `PLAN.md:80` acepta cualquier dimensión en la inversa. El camino actual no:

| Entrada, inversa | Hoy | PLAN y PROTO |
|---|---|---|
| array 0-d | `IndexError: tuple index out of range` | devuelve un array 0-d |
| 3D `(4, 1, 2)` | `ValueError: Found array with dim 3` | devuelve el array |
| 3D `(4, 3, 2)` | funciona (se aplana) | igual |

**Cambio propuesto en `PLAN.md`.** Línea 80: «`array.ndim == 1`, o `inverse_transform=True` y `array.ndim == 2`». Ninguna llamada del paquete usa otra dimensión (inventario, sección siguiente).

### 10. Subclases de `ndarray`

**Qué pasa.** Con un `np.ma.MaskedArray`, una subclase propia o un `np.matrix`, el atajo devuelve la subclase (o acepta el `np.matrix` que scikit-learn rechaza con `TypeError`). El camino actual devuelve siempre un `ndarray` base. Ningún forecaster pasa subclases, pero la función es pública.

**Cambio propuesto en `PLAN.md`.** `type(array) is np.ndarray` en el punto 2 (el mismo cambio que el hallazgo 4).

### 11. El estado «ajustado» no se puede comprobar con `hasattr`

**Qué pasa.** `PLAN.md:77` define «ajustado» como «tiene `scale_` (o `with_std=False`) y `mean_` (o `with_mean=False`)». En un escalador ajustado esos atributos existen siempre, aunque valgan `None`:

| Escalador ajustado | `mean_` | `scale_` |
|---|---|---|
| por defecto | float64 `(1,)` | float64 `(1,)` |
| `with_mean=False` | float64 `(1,)` (se calcula igual) | float64 `(1,)` |
| `with_std=False` | float64 `(1,)` | `None` |
| `with_mean=False, with_std=False` | `None` | `None` |
| serie constante, casi constante o de una observación | float64 | `1.0` |
| serie toda NaN | `nan` | `nan` |
| serie float32 o int64 | float64 | float64 |

(`review_t1_contract.py`, sección 1; igual en los cuatro entornos. Atributos presentes: `mean_`, `scale_`, `var_`, `n_samples_seen_`, `n_features_in_` y, si se ajustó con pandas, `feature_names_in_`.)

Consecuencia: si se cambia `with_std` con `set_params` después de ajustar, el atajo lanza `TypeError: 'NoneType' object is not subscriptable` y scikit-learn otra excepción (`AttributeError` en 1.9.1, `UFuncTypeError` en 1.6.0). Las dos fallan; es un caso exótico.

**Cambio propuesto en `PLAN.md`.** Línea 77: «si `with_mean`, `mean_` es un `ndarray` float64 de forma `(1,)`; si `with_std`, `scale_` también». Cubre el estado ajustado, el dtype y la forma con una sola comprobación, y hace innecesario razonar sobre `hasattr`.

### 12. `assert_array_equal` no es «bit a bit», y dos choques con las convenciones de tests

- `np.testing.assert_array_equal` trata `-0.0` y `0.0` como iguales. Para sostener «idéntico bit a bit» hay que comparar `tobytes()` (o añadir `np.signbit`), y usar `strict=True` para que compare también dtype y forma. La revisión compara por bytes.
- `PLAN.md:100` propone comparar «frente a llamar directamente al método de scikit-learn». `.github/instructions/testing.instructions.md` pide valores esperados escritos a mano, no calculados en el test. Aquí la propiedad que se prueba es precisamente la igualdad con scikit-learn en la versión instalada, así que la excepción está justificada, pero conviene decirlo en el plan y añadir al menos un caso con valores escritos a mano.
- El mismo fichero pide «un fichero de tests por método público»: los casos nuevos van en `test_transform_numpy.py`, no en un fichero nuevo `test_transform_numpy_known_scalers.py`.

### 13. float32: la exclusión es correcta, pero por un motivo más fuerte que el del plan

**Qué pasa.** `PLAN.md:79` excluye float32 porque «scikit-learn mantiene float32». Hay más: scikit-learn cambió su aritmética float32 en la versión 1.8.

- 1.6.0 y 1.7.2: `X -= self.mean_` (`sklearn/preprocessing/_data.py:1083` en 1.6.0), con `mean_` float64.
- 1.8.0 y 1.9.1: `X -= xp.astype(self.mean_, X.dtype)` (`:1132` en 1.9.1), con `mean_` convertido antes a float32.

Con el mismo escalador y la misma entrada float32, el resultado de `transform` difiere entre 1.7.2 y 1.8.0 (`review_t1_contract.py`, sección 3: huellas `fa8b3e2fda68` y `4c424d9a20d1`). En float64 las cuatro versiones dan la misma huella (`d39360efbc7ecf54`).

**Consecuencias para el alcance:**

- un atajo float32 tendría que depender de la versión de scikit-learn: no ampliarlo nunca sin esa bifurcación;
- las predicciones float32 no ganan nada: `XGBRegressor` en la validación one-step-ahead (`model_selection/_utils.py:1453`, comprobado: `dtype=float32`) y `ForecasterRnn` en la inversa de las predicciones (`_forecaster_rnn.py:1637`) y en `y_pred` de `set_out_sample_residuals` (`:2358`);
- con una serie float32 o int64, la transformación de la última ventana va por el camino actual y la inversa (float64) por el atajo, igual que hoy en resultado.

**Cambio propuesto en `PLAN.md`.** Sustituir la justificación de la línea 79 por la anterior, y añadir a V3 (`PLAN.md:264`) que cualquier ampliación tiene que comprobar float32 por versión.

### 14. Texto del plan

- `PLAN.md:87`: el mensaje del `ValueError` no es el mismo en los dos sentidos. `transform`: `Input X contains infinity or a value too large for dtype('float64').`; `inverse_transform`: `Input contains infinity or a value too large for dtype('float64').` (sin «X»). Los dos son iguales en 1.6.0 y 1.9.1. Los tests deben comprobar los dos.
- `PLAN.md:95`: el truco de la suma está en `skforecast/utils/utils.py:4378`; la línea 4280 es el inicio de `_build_predict_function`.
- `PLAN.md:57`: «del orden de 0,1 ms por llamada» se confirma (55 a 92 µs con 1.9.1; 47 a 82 µs con 1.6.0).
- `PLAN.md:86`: «una sola copia; scikit-learn hacía dos» es coherente con el código (`utils.py:2633` más `copy=True` del escalador), pero no se ha medido la memoria.

### Informativo: con `set_output(transform="pandas")` y pandas 3, la salida actual es de solo lectura

Con un escalador con `set_output(transform="pandas")` (o `sklearn.set_config(transform_output="pandas")`) y pandas 3, `transform_numpy` devuelve hoy un array de solo lectura (`DataFrame.to_numpy()` es una vista). El atajo devuelve un array escribible con los mismos bytes. Con pandas 2.3 no hay diferencia. No es un riesgo (la dirección es la segura), pero es la única diferencia observable que queda con las condiciones propuestas, y conviene anotarla en el plan para que nadie compare `flags.writeable` en un test.

---

## Lo que se confirma (sin cambios)

| Aspecto | Resultado | Evidencia |
|---|---|---|
| Orden de las operaciones | `transform`: `X -= mean_`, luego `X /= scale_`. Inversa: `X *= scale_`, luego `X += mean_`. Igual en 1.6.0 (`_data.py:1083-1085`, `1125-1127`) y 1.9.1 (`1132-1134`, `1176-1178`) | fuente de cada versión |
| Identidad en float64 | 1.176 casos por variante, 0 diferencias de valor, en 1.6.0, 1.7.2, 1.8.0 y 1.9.1 | `review_t1_contract.py`, sección 2 |
| Escalar `mean_[0]` frente al array `(1,)` | idéntico byte a byte, incluidos `-0.0`, subnormales, 1e300 y longitudes 1 a 1.000 | misma sección |
| `with_mean` y `with_std` | las cuatro combinaciones, idénticas | misma sección |
| Varianza cero | `scale_ = 1.0`; idéntico | secciones 1 y 2 |
| `copy=False` | idéntico; la entrada no se modifica | sección 2 |
| NaN | se aceptan y se propagan; idéntico | sección 2 |
| `inf` | mismo `ValueError`, mismo mensaje, en 1D y 2D, con `+inf`, `-inf`, ambos, con NaN y con cualquier escalador | sección 4 |
| `sklearn.config_context(assume_finite=True)` | hoy `inf` no lanza; con el atajo tampoco (porque se vuelve al camino actual) | sección 4 |
| Arrays no contiguos, con paso negativo, de solo lectura, en orden F | idénticos en valores, forma y flags; salida C contigua y escribible | sección 2 |
| La entrada no se modifica (#1504) | 5.016 comparaciones por entorno, ninguna modifica la entrada ni comparte memoria con ella | todas las secciones |
| Inversa 2D aplanada | idéntica para `(48, 250)`, `(24, 3)`, `(n, 1)`, `(1, k)`, cortes de un cubo y vistas traspuestas | sección 2 |
| Subclase de `StandardScaler`, `Pipeline`, escalador sin ajustar, escalador de 2 columnas | camino actual, mismo resultado o mismo error | sección 5 |
| Forecasters guardados | 45 ficheros (joblib, pickle y skops; guardados con 0.28.0, con 0.27.0 de PyPI y con scikit-learn 1.6.0), cargados con el código actual: predicciones idénticas con las tres variantes; tras cargar, `type(t) is StandardScaler` y `mean_`/`scale_` son `ndarray` float64 `(1,)` | `review_t1_saved.py` |
| Origen de los transformadores | `transformer_series_`: clones (`utils.py:448`) o `deepcopy` del dict (`:456`). `transformer_y`: el objeto del usuario, sin clonar (`_forecaster_recursive.py:328`, `_forecaster_direct.py:500`, `_forecaster_stats.py:227`). El atajo lee `mean_` y `scale_` en cada llamada, así que un reajuste del usuario se refleja igual que hoy | inventario |

Entre entornos, las únicas salidas float64 que cambian de bytes son las de escaladores con `mean_` o `scale_` igual a NaN (serie toda NaN, o valores de 1e200): todas las salidas son NaN y cambia el patrón de bits del NaN entre numpy 2.0.0 y 2.5.5. Dentro de cada entorno, atajo y camino actual coinciden.

---

## Inventario de llamadas a `transform_numpy`

39 llamadas en 7 módulos (el plan dice lo mismo, `PLAN.md:69`), más 2 llamadas directas a `inverse_transform` que no pasan por la función (hallazgo 7). Todas usan `fit=False` salvo las dos de ajuste. `C` es contigua en C, `W` escribible; «usuario» indica que el transformador es el objeto que pasó el usuario.

| Módulo y línea | Uso | `inverse` | `ndim` | dtype | C / W | Transformador | PLAN | PROPOSED |
|---|---|---|---|---|---|---|---|---|
| `recursive/_forecaster_recursive.py:1606` | última ventana | no | 1 | float64 (int64 si la serie es entera) | sí / sí | `transformer_y`, usuario, con nombres | no | sí (no con int64) |
| `:2153` | predicciones | sí | 1 | float64 | sí / sí | igual | no | sí |
| `:2300` | bootstrapping | sí | 2 `(steps, n_boot)` | float64 | sí / sí | igual | no | sí |
| `:2435` | conformal | sí | 2 `(steps, 3)` | float64 | sí / sí | igual | no | sí |
| `:3124`, `:3130` | `set_out_sample_residuals` | no | 1 | float64 | sí / no | igual | no | sí |
| `direct/_forecaster_direct.py:2163`, `:2530`, `:2677`, `:2813`, `:3566`, `:3572` | los mismos seis usos | igual | igual | float64 | igual | `transformer_y`, usuario, con nombres | no | sí |
| `recursive/_forecaster_recursive_multiseries.py:997` | ajuste (`fit=True` en `fit`; `fit=False` si ya está ajustado o con `encoding=None`) | no | 1 | float64 | sí / no | `transformer_series_[serie]`, clon, sin nombres | sí con `fit=False` | sí con `fit=False` |
| `:2894` | última ventana por nivel | no | 1 | float64 | sí / no | clon o `_unknown_level` | sí | sí |
| `:3649` | predicciones por nivel | sí | 1 | float64 | **no** / sí | igual | sí | sí |
| `:3832` | bootstrapping por nivel | sí | 2 `(steps, n_boot)` | float64 | **no** / sí | igual | sí | sí |
| `:4002` | conformal por nivel | sí | 2 `(steps, 3)` | float64 | **no** / sí | igual | sí | sí |
| `:4872`, `:4878` | `set_out_sample_residuals` | no | 1 | float64 | sí / no | igual | sí | sí |
| `direct/_forecaster_direct_multivariate.py:1249` | ajuste | no | 1 | float64 | sí / sí | clon, sin nombres | sí con `fit=False` | igual |
| `:2335`, `:2711`, `:2861`, `:2999`, `:3817`, `:3823` | última ventana, predicciones, bootstrapping, conformal, residuos | como arriba | 1 o 2 | float64 | sí / sí o no | clon, sin nombres | sí | sí |
| `deep_learning/_forecaster_rnn.py:1393` | última ventana | no | 1 | float64 | sí / sí | clon, **con nombres** (`transform_series`) | no | sí |
| `:1637` | predicciones | sí | 1 | **float32** | no / sí | igual | no | no (float32) |
| `:1790` | conformal | sí | 2 | float64 | no / sí | igual | no | sí |
| `:2352` | `y_true` de residuos | no | 1 | float64 | sí / sí | igual | no | sí |
| `:2358` | `y_pred` de residuos | no | 1 | **float32** | sí / sí | igual | no | no (float32) |
| `recursive/_forecaster_stats.py:938` | predicciones | sí | 1 | float64 | sí / sí o no | `transformer_y`, usuario, con nombres | no | sí |
| `:1153` | intervalos | sí | 2 `(steps * n, 3)` | float64 | sí / sí | igual | no | sí |
| `model_selection/_utils.py:1329`, `:1339` | one-step-ahead con diferenciación | no y sí | 1 | float64 | sí / sí | el del forecaster (copia) | según forecaster | sí |
| `:1447` | one-step-ahead, `y_true` | sí | 1 | float64 | sí / sí o no | `transformer_y` (copia) | no | sí |
| `:1453` | one-step-ahead, `y_pred` | sí | 1 | float64 (**float32** con XGBoost) | sí / sí | igual | no | sí (no con float32) |
| `:2905`, `:2911` | one-step-ahead, `ForecasterDirectMultiVariate` | sí | 1 | float64 | sí / no o sí | `transformer_series_[level]` | sí | sí |
| `:3046`, `:3049` | one-step-ahead, `ForecasterRecursiveMultiSeries` | **llamada directa a `inverse_transform` con un `DataFrame`** | | | | `transformer_series_[level]` | no pasa por `transform_numpy` | no pasa |

Observaciones:

- Ninguna llamada del paquete usa `ndim` distinto de 1 o 2, ni `fit=True` con `inverse_transform=True`.
- Las inversas 2D son siempre `(steps, n_boot)` o `(steps, 3)`. En multiserie llegan como cortes no contiguos de un cubo (`boot_predictions[:, i, :]`); el resultado y la disposición de la salida coinciden con el camino actual.
- Con un dict de transformadores, las series sin transformador pasan `None` y vuelven sin copiar, como hoy.
- En el backtesting el forecaster se copia, así que el escalador ya no es el objeto del usuario, pero conserva los nombres.

Salida completa: `results/review_t1_inventory_latest.txt` y `results/review_t1_inventory_min.txt` (este incluye `ForecasterRnn`).

---

## Tests: plan frente a tests existentes

| Test del plan (`PLAN.md:102-111`) | Tests existentes que lo tocan | Veredicto |
|---|---|---|
| Transformar e invertir, 1D y 2D (inversa) | `..._StandardScaler` (con `fit=True`: no entra en el atajo), `..._inverse_transform_is_True`, `..._multiple_columns_equivalent...`, `..._preserves_shape` (todos con `assert_array_almost_equal`, 6 decimales) | Necesario: ninguno comprueba igualdad exacta |
| Las cuatro combinaciones de `with_mean` y `with_std` | ninguno | Necesario. Añadir que con las dos en `False` la salida es una copia |
| NaN en la entrada | ninguno | Necesario. Añadir serie toda NaN y NaN con `inf` |
| `inf` en la entrada | ninguno (tampoco en los forecasters: hoy ese error no está testeado) | Necesario. Faltan los dos mensajes, 2D, `-inf` y `assume_finite=True` |
| float32 | ninguno | Necesario. Añadir int64 (salida float64) |
| La entrada no se modifica | `test_transform_numpy_does_not_modify_input_array` ya cubre `StandardScaler(copy=False)` con `fit=False` en los dos sentidos y `np.shares_memory` | **Redundante en 1D.** Ampliar ese parametrizado con la inversa 2D y un array de solo lectura |
| Subclase de `StandardScaler` | ninguno | Necesario |
| Sin ajustar: mismo `NotFittedError` | ninguno | Necesario. Añadir `StandardScaler(with_mean=False, with_std=False)` sin ajustar |
| `Pipeline([StandardScaler])` | el parametrizado anterior incluye `make_pipeline(StandardScaler(copy=False))`, sin comprobar el camino | Parcialmente redundante: unirlo al test del camino tomado |
| `transformer=None` devuelve la entrada sin copiar | `test_transform_numpy_when_transformer_is_None` (solo `almost_equal`) | Añadir `assert results is input_array` al test existente, no un test nuevo |

**Casos que faltan en el plan:**

1. El camino tomado, en los dos sentidos (hallazgo 5).
2. Escalador ajustado con un `DataFrame` (hallazgo 1), según la decisión que se tome.
3. Array vacío, 1D y 2D (hallazgo 3).
4. `array` que no es un `ndarray`, con `fit=False` y escalador ajustado (hallazgo 4).
5. `ndim` 0 y 3; transformación (no inversa) de un array 2D: `(n, 1)` funciona por el camino actual y `(n, k)` lanza el `ValueError` de columnas.
6. Escalador de 2 columnas, serie constante (`scale_ = 1.0`) y `set_output(transform="pandas")`.
7. El escalador no se modifica: `mean_` y `scale_` iguales antes y después.
8. Valores finitos enormes: mismos avisos que hoy (hallazgo 8).
9. En los forecasters: `inf` en `last_window` lanza el mismo error; un forecaster cargado con pickle y con skops predice igual; `set_out_sample_residuals` con arrays vacíos sigue lanzando.

Para respetar las convenciones del repositorio: todo en `test_transform_numpy.py`, parametrizado (un test de identidad con escalador, entrada y sentido como parámetros; un test de errores; un test del camino tomado).

---

## Alternativas de diseño

| Alternativa | Números | Veredicto |
|---|---|---|
| **A.** Dejar el camino actual dentro de `sklearn.config_context(assume_finite=True, skip_parameter_validation=True)` | 55 a 80 µs por llamada frente a 55 a 92 µs | Descartada: el coste no es la comprobación de finitos, sino el resto de `validate_data` y `check_array` |
| **B.** Buscar `inf` en la entrada antes de calcular | 2,3 a 3,1 µs por llamada completa, plano con NaN; frente a 3,7 a 5,1 µs del decidido con `errstate` y 2,0 a 2,9 µs del decidido tal cual (con el aviso) | **Recomendada** como arreglo del hallazgo 8: más simple y sin avisos |
| **C.** Vectorizar por series en los forecasters (una operación sobre la matriz de niveles) | Tras el atajo quedan entre 1,5 y 5 ms dentro de `transform_numpy` en un `predict` de 120 ms en M (600 llamadas) | Descartada: como mucho un 4 % más, tocando 5 forecasters |
| **D.** Atajo también para `fit=True` | En backtesting con refit, el ajuste son 1,44 s (6,4 %) frente a 0,54 s (2,4 %) de lo que cubre T1 | Fuera de T1: habría que reproducir `mean_`, `var_`, `scale_` y `n_samples_seen_` bit a bit. Solo como estudio aparte |
| **E.** Llevar las dos llamadas directas de la validación one-step-ahead a `transform_numpy` | 1,06 s de 9,70 s (10,9 %) en M con 3 candidatos | **Recomendada** como ampliación pequeña (hallazgo 7) |
| **F.** Guardar `mean_` y `scale_` en caché por transformador | ahorra menos de 0,5 µs por llamada | Descartada: añade un problema de invalidación con `transformer_y`, que es el objeto del usuario |

El diseño del plan (una función privada que devuelve `None` si no aplica) es el más simple de los razonables. Lo que cambia son las condiciones. Coste por llamada de cada variante (`review_t1_bench.py micro`, 1D de 168 valores, transformación): actual 88,6 µs; PLAN 2,2 µs; PROTO 2,2 µs; PROPOSED 4,0 µs; PROPOSED-B 2,7 µs. Las guardas añadidas cuestan unos 0,5 µs y `np.errstate` 1,3 µs. Inversa 2D `(168, 250)`: actual 128 µs; atajo 32 a 36 µs.

Para los transformadores que no usan el atajo (`MinMaxScaler`, `FunctionTransformer`, `fit=True`), las comprobaciones añaden menos de 2 µs por llamada, medido a través de un envoltorio que exagera el coste.

---

## Texto propuesto para el punto 2 de T1

Sustituye a `PLAN.md:74-80` si se aceptan los hallazgos 1, 3, 4, 9, 10 y 11:

> **El atajo aplica solo si se cumple todo lo siguiente** (cualquier otro caso va por scikit-learn). Se evalúa después de la comprobación de tipo de `array` que ya hace `transform_numpy`:
> - `fit=False`;
> - `type(transformer) is StandardScaler`: el tipo exacto, para que las subclases sigan el camino actual;
> - `type(array) is np.ndarray`, `array.dtype == np.float64` y `array.size > 0`. Las subclases de `ndarray`, float32 (scikit-learn cambió su aritmética float32 en la versión 1.8), los enteros y los arrays vacíos (que scikit-learn rechaza) van por el camino actual;
> - `array.ndim == 1`, o `inverse_transform=True` y `array.ndim == 2`;
> - `transformer.n_features_in_ == 1`. Un escalador ajustado con nombres de columna también usa el atajo;
> - si `with_mean`, `mean_` es un `ndarray` float64 de forma `(1,)`; si `with_std`, `scale_` también.

Con estas condiciones (variante PROPOSED), las 5.016 comparaciones de `review_t1_contract.py` dan 0 diferencias con el camino actual en los cuatro entornos, salvo el flag de solo lectura con `set_output(transform="pandas")` y pandas 3.

---

## Lo que no se pudo comprobar

- **Forecasters guardados con versiones anteriores a la 0.27.0** y con scikit-learn anterior a 1.6. Solo se probó la 0.27.0 de PyPI (con scikit-learn 1.9.1 y pandas 2.3.3) y la 0.28.0 guardada con scikit-learn 1.6.0.
- **Otras plataformas.** Todo es Linux x86-64. La identidad bit a bit se apoya en que las cuatro operaciones son IEEE con redondeo correcto, pero no se ha ejecutado en macOS (Apple Silicon) ni en Windows.
- **Combinaciones cruzadas de versiones.** pandas 2.3 solo con scikit-learn 1.6.0; scikit-learn 1.7.2 y 1.8.0 solo con numpy 2.5.5 y pandas 3.1.1. No se probaron las versiones de numpy entre 2.0.0 y 2.5.5 ni Python 3.13 o 3.14.
- **`ForecasterRnn` con las últimas versiones.** Solo con keras 3.3.0 y torch 2.4.0 (entorno de mínimos); el entorno principal no tiene keras.
- **La suite `benchmarks/` (V7)** no se ejecutó: tiene sus propios requisitos fijados.
- **El tamaño L.** Las cifras del plan para L (×2,1 y ×1,8 junto con T2) no se han repetido.
- **Tiempos de extremo a extremo fiables.** La máquina es un contenedor compartido de 4 núcleos; las razones tienen cuartiles de ±15 a 20 %. Las cifras por llamada y el porcentaje dentro de `transform_numpy` son estables; las razones de `predict`, orientativas.
- **Memoria.** No se midió que el atajo haga una copia en lugar de dos.
- **Backtesting en paralelo (`n_jobs > 1`)** y la API de arrays de scikit-learn (`array_api_dispatch=True`).
- **`check_equivalence.py` con pandas 2.3 y scikit-learn 1.9.** Se ejecutó con pandas 3.1.1 más scikit-learn 1.9.1 y con pandas 2.3.0 más scikit-learn 1.6.0 (las dos pasan).

---

## Cómo reproducir

Desde la raíz del repositorio. Los scripts no modifican `skforecast/`: sustituyen `transform_numpy` en tiempo de ejecución.

| Script | Qué comprueba | Salida en `results/` |
|---|---|---|
| `review_t1_common.py` | Variantes PLAN, PROPOSED y PROPOSED-B, trazador y contador | |
| `review_t1_inventory.py` | Inventario en ejecución de las llamadas | `review_t1_inventory_latest.txt`, `_min.txt` |
| `review_t1_contract.py` | Contrato de `StandardScaler`, identidad por bytes y casos límite | `review_t1_contract_sklearn_latest.txt` (1.9.1), `_min.txt` (1.6.0), `_1.7.txt`, `_1.8.txt` |
| `review_t1_forecasters.py` | Equivalencia de extremo a extremo en los seis forecasters | `review_t1_forecasters_latest.txt`, `_min.txt` |
| `review_t1_saved.py` | Forecasters guardados con joblib, pickle y skops | `review_t1_saved_forecasters.txt` |
| `review_t1_empty_inputs.py` | Arrays vacíos por la API pública | `review_t1_empty_inputs.txt` |
| `review_t1_one_step_ahead.py` | Llamadas directas a `inverse_transform` | `review_t1_one_step_ahead_M.txt` |
| `review_t1_bench.py` | Coste por llamada, extremo a extremo, backtesting y resto de forecasters | `review_t1_micro_latest.txt`, `_min.txt`, `review_t1_macro_M_ridge.txt`, `_lgbm.txt`, `review_t1_backtesting_M_ridge.txt`, `review_t1_others_latest.txt` |
| `check_equivalence.py`, `bench.py` (los del plan) | Repetidos sobre el `0.28.x` actual | `review_t1_check_equivalence_latest.txt`, `_min.txt`, `review_t1_bench_py_M_ridge.txt`, `_lgbm.txt`, `review_t1_bench_M_ridge.json`, `_lgbm.json` |

```bash
python dev/preprocessing_speedups/review_t1_contract.py
```

```bash
~/.venvs/skforecast-min/bin/python dev/preprocessing_speedups/review_t1_contract.py
```

```bash
python dev/preprocessing_speedups/review_t1_forecasters.py
```

```bash
python dev/preprocessing_speedups/review_t1_bench.py macro --size M --est ridge --rounds 25
```

Los JSON de referencia del plan (`results/bench_M_*.json`) no se han modificado: las repeticiones de `bench.py` están en `review_t1_bench_M_*.json`.
