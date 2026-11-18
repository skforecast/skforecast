# Plan: optimizaciones del preprocesado (codificación de categóricas, atajo de escaladores, predicción in-sample en `fit` y validación one-step-ahead)

**Fecha:** 2026-11-18.
**Origen:** el estudio de viabilidad de polars en `ForecasterRecursiveMultiSeries` (descartado; ver la sección 1).
**Para:** quien implemente estos cambios más adelante.
**Base comprobada:** `0.28.x` en `c694e19` (2026-11-17). Los números de línea citados son de ese commit: vuelve a localizarlos antes de editar.
**Revisión de T1:** 2026-11-18, en [`REVIEW_T1.md`](REVIEW_T1.md). La sección de T1 ya incorpora sus hallazgos y las decisiones tomadas; T4 sale de esa revisión.
**Entorno de medida:** Python 3.12.3, numpy 2.5.5, pandas 3.1.1 (con pyarrow 26.0.1), scikit-learn 1.9.1, numba 0.68.1, lightgbm 4.7.0. CPU Intel Xeon a 2,10 GHz, 4 núcleos lógicos, 15,7 GiB de RAM (contenedor en la nube). La equivalencia se comprobó también con pandas 2.3.3.

> **Cómo usar este documento**
> 1. Empieza por la tarea que toque (T1, T2, T3 o T4): cada una trae el problema, el diseño, los tests, los criterios de aceptación y la nota de versión.
> 2. Las tareas T1, T2 y T4 son independientes y van en PRs separados. T3 es un estudio que termina en una decisión, no en código.
> 3. Antes de empezar, vuelve a ejecutar los scripts de la sección 4 sobre el `0.28.x` del momento, porque el código puede haber cambiado: `check_equivalence.py` y `bench.py` para T2, y los `review_t1_*.py` para T1 y T4.
> 4. Las validaciones pendientes (sección 3) indican qué tarea bloquean.
> 5. Las decisiones marcadas «decisión de Joaquín» no se implementan hasta que él elija una opción. La recomendada va primero.

---

## 1. Contexto y decisión

Se estudió con números medidos si polars aceleraría `ForecasterRecursiveMultiSeries`. El caso objetivo:

- **Series:** de 100 a 1.000, con 1.000 a 5.000 observaciones horarias cada una y longitudes e inicios distintos.
- **Exógenas:** 5 numéricas y 3 categóricas `category` (de 5 a 50 categorías), unas comunes a todas las series y otras específicas.
- **Configuración:** `lags=48`, `RollingFeatures` con 3 estadísticos y `StandardScaler` como `transformer_series`.
- **Estimadores:** LightGBM, HGBR y Ridge.

**Decisión: no adoptar polars.** Los cuellos de botella no son operaciones tabulares:

- Con LightGBM, el estimador es el 83 % del `fit` en todos los tamaños, así que el techo de Amdahl es ×1,19.
- Medido de extremo a extremo con 1.000 series:
  - polars, `fit` de 37,8 s a 34,3 s (estimado);
  - la alternativa con numpy, 32,9 s. Es una sola repetición: lo esperable es ahorrar unos 3 s, lo que cuesta de menos construir `X_train`.
- En `predict`, solo convertir la entrada a polars ya cuesta más que el `predict` optimizado con numpy.

Los ficheros del estudio de polars se han borrado. Siguen en el historial de la rama local `chore/polars-study`, que no se subió a GitHub, mientras no se borre.

Lo que sí compensa son dos cambios pequeños, sin dependencias nuevas y con resultados idénticos bit a bit (T1 y T2), más un estudio de diseño (T3) y un tercer cambio pequeño que salió de la revisión de T1 (T4). Las cifras de partida (`ForecasterRecursiveMultiSeries`, encoding ordinal, entrada como dict, mediana de varias repeticiones):

| Tamaño | Filas de entrenamiento | `fit` LightGBM | `fit` Ridge | Parte de skforecast en `fit` | Codificador en `fit` | `predict` 168 (LightGBM) |
|---|---|---|---|---|---|---|
| S (100 series) | 134.137 | 2,34 s | 0,42 s | 0,4 s | 0,12 s | 0,12 s (0,03 s estimador) |
| M (300 series) | 880.369 | 10,41 s | 2,25 s | 1,7 s | 0,76 s | 0,27 s (0,08 s estimador) |
| L (1.000 series) | 2.895.871 | 37,53 s | 7,61 s | 6,0 s | 2,8 s | 0,97 s (0,22 s estimador) |

**Dónde se va el tiempo de skforecast** (cProfile, L, Ridge):

1. **`OrdinalEncoder` sobre columnas `category`:** convierte cada columna a un array de objetos (`Categorical.__array__`, 1,0 s; con pandas 3 y pyarrow, las categorías de texto están en Arrow) y la mapea con diccionarios de Python (`_map_to_integer`, `_check_unknown` y `_unique_python`, 1,4 s). En el backtesting con refit (L, Ridge) suma 19 s de 71,8 s, más que el propio estimador (14,8 s). Su pico de memoria transitoria es ~0,46 veces el tamaño de `X_train` (0,6 GiB en L).
2. **`StandardScaler` serie a serie:** en cada `predict`, unas 2.000 llamadas (transformar la última ventana e invertir las predicciones de cada serie). Cada una pasa por `validate_data` / `check_array` de scikit-learn: 0,57 s de `transform_numpy` sobre 1,5 s de `predict` medido con cProfile (0,64 s sin perfilador) en L con Ridge.
3. **`fit` predice sobre todo `X_train`** para ajustar los bins de residuos (`_probabilistic_mode='binned'` por defecto), aunque luego no se pidan intervalos: 13 % del `fit` con LightGBM y 17 % con HGBR en L.

---

## 2. Tareas

### T1. Atajo en `transform_numpy` para un `StandardScaler` ajustado

**Estado: revisada el 2026-11-18.** La revisión crítica está en [`REVIEW_T1.md`](REVIEW_T1.md) (hallazgos, evidencia y cifras). Decisiones de Joaquín tras revisarla, ya incorporadas a esta sección:

1. El atajo se aplica también a los escaladores ajustados con nombres de columna (`feature_names_in_`).
2. Los valores infinitos se buscan en la entrada, antes de calcular. Sustituye a la comprobación con la suma del resultado.
3. Las dos llamadas directas a `inverse_transform` de la validación one-step-ahead multiserie quedan fuera de T1: son la tarea T4.
4. Los criterios de aceptación de rendimiento se sustituyen por otros que no dependen del ruido de la máquina.

**Problema.** `transform_numpy` (`skforecast/utils/utils.py`, línea 2580) llama a `transformer.transform` o `inverse_transform` en cada serie. Para un `StandardScaler`, el cálculo es una resta y una división, pero scikit-learn valida la entrada en cada llamada (`validate_data`, `check_array`, `_assert_all_finite`): entre 55 y 92 µs por llamada con scikit-learn 1.9.1 (entre 47 y 82 µs con 1.6.0). En `predict` multiserie hay dos llamadas por serie, y en el backtesting dos por serie y fold.

**Ganancia medida** (`review_t1_bench.py`, entorno de la cabecera; detalle en `REVIEW_T1.md`):

- **Por llamada:** de 55 a 92 µs a entre 2 y 3 µs con un array 1D de 48 o 168 valores. Inversa 2D de `(168, 250)`: de 128 µs a unos 33 µs.
- **`ForecasterRecursiveMultiSeries`, M (300 series).** Mediana de la razón actual / atajo pareada por ronda, con las variantes intercaladas (25 rondas con Ridge, 15 con LightGBM). El techo es lo que se ganaría si las llamadas a `transform_numpy` costasen cero:

  | Flujo | Ridge | Techo con Ridge | LightGBM | Techo con LightGBM |
  |---|---|---|---|---|
  | `predict` de 168 pasos | ×1,4 a ×1,5 | ×1,40 (29 % del tiempo) | ×1,2 a ×1,3 | ×1,27 (21 %) |
  | `predict` de 24 pasos | ×1,5 a ×1,7 | ×1,46 (31 %) | ×1,4 a ×1,6 | ×1,47 (32 %) |
  | Conformal de 24 pasos | ×1,2 a ×1,3 | ×1,26 (20 %) | ×1,2 a ×1,35 | ×1,23 (19 %) |
  | Bootstrapping de 24 pasos (`n_boot=100`) | ×1,3 a ×1,5 | ×1,23 (19 %) | ×1,1 | ×1,08 (8 %) |

  Las razones que superan el techo son ruido: en la máquina de medida, los cuartiles de la razón quedan hasta un 20 % por encima o por debajo de la mediana.
- **Forecasters de una serie** (Ridge, 15 rondas): `ForecasterRecursive.predict(24)`, ×1,6; backtesting de 1.000 folds sin refit, ×2,2; `ForecasterDirect`, backtesting de 200 folds, ×1,9; `ForecasterStats.predict(24)` con Arima, ×1,35; `ForecasterDirectMultiVariate`, backtesting de 200 folds, ×1,2.
- **Backtesting multiserie:** sin ganancia apreciable. En M con Ridge, las llamadas que cubre el atajo son el 5,8 % del backtesting sin refit y el 2,4 % con refit (techos ×1,06 y ×1,025). El ×1,18 que daba la versión anterior de este plan era ruido de una sola repetición.
- **Junto con T2** (`bench.py`, dos ejecuciones de 2 o 3 repeticiones; orientativo): `predict` de 168 pasos entre ×1,7 y ×2,2 con Ridge y entre ×1,4 y ×1,8 con LightGBM. En L, el estudio inicial midió ×2,1 y ×1,8; no se ha repetido.

El atajo funciona igual con exógenas de texto, porque no depende de ellas.

**Dónde.** Una sola función, `transform_numpy`, a la que se llama 39 veces desde 7 módulos: `recursive/_forecaster_recursive_multiseries.py` (7 veces), `recursive/_forecaster_recursive.py` (6), `direct/_forecaster_direct.py` (6), `direct/_forecaster_direct_multivariate.py` (7), `deep_learning/_forecaster_rnn.py` (5), `recursive/_forecaster_stats.py` (2) y `model_selection/_utils.py` (6). El inventario de cada llamada (argumentos, dimensiones, dtype y origen del transformador) está en `REVIEW_T1.md`. Tres matices:

- Los forecasters de una serie y `ForecasterRnn` ajustan el escalador con un objeto de pandas (`transform_dataframe` o `transform_series`), así que su escalador tiene `feature_names_in_`. Los dos multiserie lo ajustan con `transform_numpy(fit=True)`, sin nombres. El atajo vale para los dos casos (punto 2 del diseño).
- No ganan nada las llamadas con datos float32, que siguen por scikit-learn: las predicciones de `XGBRegressor` en la validación one-step-ahead (`model_selection/_utils.py`, línea 1453) y las de `ForecasterRnn` (`deep_learning/_forecaster_rnn.py`, líneas 1637 y 2358). Tampoco la transformación de la última ventana cuando la serie es entera (int64).
- La validación one-step-ahead de `ForecasterRecursiveMultiSeries` no pasa por `transform_numpy` (`model_selection/_utils.py`, líneas 3046 y 3049): es la tarea T4.

**Diseño.** Prototipo de referencia: `fast_transform_numpy` en `dev/preprocessing_speedups/prototypes.py`, alineado con este diseño.

1. **Ruta rápida:** una función privada, `_transform_numpy_fast_path(array, transformer, inverse_transform)`. El nombre dice para qué está (es la ruta rápida de `transform_numpy`) y sigue valiendo cuando se añadan otros escaladores o un `Pipeline`. Devuelve el array transformado si el atajo aplica y `None` si no. `transform_numpy` la llama, solo con `fit=False`, **después de la comprobación de tipo de `array`** (líneas 2623 a 2626) y antes de copiarlo (línea 2633). Con `None`, sigue por el camino actual.
2. **El atajo aplica solo si se cumple todo lo siguiente** (cualquier otro caso va por scikit-learn):
   - `fit=False`;
   - `type(transformer) is StandardScaler`: el tipo exacto, para que las subclases sigan el camino actual;
   - `type(array) is np.ndarray`: con una subclase (`MaskedArray`, `np.matrix`), el camino actual devuelve un `ndarray` base o la rechaza;
   - `array.dtype == np.float64`. Con float32, scikit-learn mantiene float32 y, además, cambió su aritmética float32 en la versión 1.8 (convierte antes `mean_` y `scale_` a float32), así que el resultado depende de la versión. Los enteros también van por el camino actual;
   - `array.size > 0`: scikit-learn rechaza un array vacío con un `ValueError`, y hoy ese error es el que impide que `set_out_sample_residuals` acepte arrays vacíos;
   - `array.ndim == 1`, o `inverse_transform=True` y `array.ndim == 2` (el camino actual de la inversa multicolumna se reduce a una columna y vuelve). Ninguna llamada del paquete usa otra dimensión;
   - `transformer.n_features_in_ == 1`. **No se exige que falte `feature_names_in_`:** la entrada de `transform_numpy` es siempre un array y, con un escalador ajustado con nombres, scikit-learn solo emite un aviso que la función ya silencia (líneas 2647 a 2652). El resultado es idéntico;
   - el estado del escalador es el esperado: si `with_mean`, `mean_` es un `ndarray` float64 de forma `(1,)`; si `with_std`, `scale_` también. No vale `hasattr`: en un escalador ajustado los dos atributos existen siempre, y valen `None` cuando su parámetro es `False`.
3. **La misma aritmética, en el mismo orden, que scikit-learn:**
   - transformación: `out = array.copy()`; `if with_mean: out -= mean_[0]`; `if with_std: out /= scale_[0]`;
   - inversa: `if with_std: out *= scale_[0]`; `if with_mean: out += mean_[0]`.

   El orden es el mismo en scikit-learn 1.6.0 y 1.9.1 (`sklearn/preprocessing/_data.py`). Operar con el escalar `mean_[0]` da el mismo resultado IEEE que el array de forma `(1,)` de scikit-learn: comprobado byte a byte en 1.176 casos (las cuatro combinaciones de `with_mean` y `with_std`, NaN, `-0.0`, subnormales, valores de 1e300, varianza cero, arrays no contiguos, de solo lectura y en orden F) con scikit-learn 1.6.0, 1.7.2, 1.8.0 y 1.9.1.
4. **Conservar la copia de la entrada.** La #1504 (`fix/inplace-transformers`) garantiza que `transform_numpy` no modifica el array del llamante. El atajo hace una sola copia (`array.copy()`, contigua en C); scikit-learn hacía dos. La salida nunca comparte memoria con la entrada, y el atajo no escribe en `mean_` ni en `scale_`.
5. **Valores infinitos (decidido el 2026-11-18 y ajustado ese día tras la revisión).** scikit-learn acepta NaN pero lanza `ValueError` con `inf`, tanto en `transform` como en `inverse_transform`. Con algún `inf` en la entrada, el atajo vuelve al camino actual, que lanza ese mismo error:

     ```python
     if np.isinf(array).any():
         return None  # camino actual: el mismo ValueError que scikit-learn
     ```

   - **Por qué en la entrada y antes de calcular:** scikit-learn valida la entrada, así que la condición es exactamente la suya. Si la entrada es finita y el cálculo desborda a `inf`, scikit-learn devuelve ese `inf`, y el atajo también. Como se vuelve al camino actual en lugar de lanzar el error desde el atajo, `sklearn.config_context(assume_finite=True)` se sigue respetando.
   - **El mensaje no es el mismo en los dos sentidos:** `Input X contains infinity or a value too large for dtype('float64').` en `transform` e `Input contains infinity or a value too large for dtype('float64').` en `inverse_transform`. Los dos son iguales en scikit-learn 1.6.0 y 1.9.1.
   - **Coste medido:** de 0,6 a 0,8 µs con arrays de 48 y 168 valores, haya o no NaN; unos 8,5 µs con una matriz de `(168, 250)`.
   - **Versión anterior, sustituida:** `if not math.isfinite(out.sum()) and np.isinf(array).any()`, el truco del atajo de los modelos lineales (`_build_predict_function`, `skforecast/utils/utils.py`, línea 4378). Costaba de 0,4 a 0,5 µs sin NaN y de 1,0 a 1,2 µs con NaN, pero `out.sum()` emite un `RuntimeWarning` de overflow con valores finitos enormes, que scikit-learn silencia con `np.errstate(over="ignore")`. Silenciarlo igual costaba entre 1,0 y 1,5 µs más.
   - **Alternativa descartada:** aceptar la diferencia y documentarla.
6. **Fuera del alcance de T1:**
   - `MinMaxScaler`, `RobustScaler` y los `Pipeline`: se añaden después de la validación V3;
   - float32 y enteros (punto 2);
   - `fit=True`: habría que reproducir bit a bit el estado ajustado (`mean_`, `var_`, `scale_`, `n_samples_seen_`). En el backtesting multiserie con refit pesa más que lo que cubre T1 (1,44 s frente a 0,54 s en M), así que solo tendría sentido como estudio aparte;
   - la validación one-step-ahead multiserie (T4).
7. **Única diferencia observable conocida:** con `set_output(transform="pandas")` (o `sklearn.set_config(transform_output="pandas")`) y pandas 3, el camino actual devuelve un array de solo lectura y el atajo uno escribible, con los mismos bytes. Los tests no comparan los flags.

**Tests.** En `skforecast/utils/tests/tests_utils/test_transform_numpy.py`, parametrizados (un fichero por función pública: sin fichero nuevo). La propiedad que se prueba es la igualdad con scikit-learn en la versión instalada, así que el valor esperado se calcula llamando al método de scikit-learn, como excepción a la norma de valores escritos a mano; se añade además un caso con valores escritos a mano. Las comparaciones son por bytes (`tobytes()`) o con `np.testing.assert_array_equal(..., strict=True)`, que comprueba también dtype y forma. `assert_array_equal` por sí solo no distingue `-0.0` de `0.0`.

1. **Identidad** (escalador, entrada y sentido como parámetros):
   - transformar e invertir en 1D; invertir en 2D con `(steps, n_boot)`, `(steps, 3)`, `(n, 1)` y `(1, k)`;
   - las cuatro combinaciones de `with_mean` y `with_std` (con las dos en `False`, la salida es una copia);
   - escalador ajustado con un array y con un `DataFrame` (con `feature_names_in_`), con `copy=False` y con `set_output(transform="pandas")`;
   - serie constante (`scale_ = 1.0`);
   - NaN en la entrada y entrada toda NaN;
   - arrays no contiguos (columna de una matriz, corte de un cubo), en orden F y de solo lectura.
2. **El camino tomado**, en los dos sentidos, con `monkeypatch` para que `StandardScaler.transform` e `inverse_transform` lancen un error centinela. Sin este test, todos los demás pasan aunque el atajo no se use nunca:
   - casos elegibles: `transform_numpy` devuelve el resultado, porque el atajo no llama a scikit-learn;
   - casos no elegibles: se propaga el error centinela. Subclase de `StandardScaler`, `Pipeline([("s", StandardScaler())])`, float32, int64, array vacío, transformación de un array 2D, escalador de dos columnas y `fit=True`.
3. **Mismos resultados y errores que hoy fuera del atajo:**
   - float32: salida float32; int64: salida float64;
   - `inf`: los dos mensajes del punto 5, en 1D y 2D, con `+inf`, `-inf` y junto con NaN. Con `sklearn.config_context(assume_finite=True)` no hay error, como hoy;
   - escalador sin ajustar: el mismo `NotFittedError`, también con `StandardScaler(with_mean=False, with_std=False)`;
   - array vacío, 1D y 2D: el mismo `ValueError`;
   - `array` que no es un `ndarray` (una `Series` float64 y una lista), con el escalador ajustado y `fit=False`: el mismo `TypeError`. El test actual usa una `Series` int64 con `fit=True` y no llegaría al atajo;
   - arrays de 0 y de 3 dimensiones en la inversa: el mismo comportamiento que hoy.
4. **Sin efectos secundarios:**
   - la entrada no se modifica y la salida no comparte memoria con ella. Se amplía `test_transform_numpy_does_not_modify_input_array`, que ya cubre el caso 1D, con la inversa 2D y un array de solo lectura;
   - `mean_` y `scale_` no cambian;
   - con valores finitos del orden de 1e308 no aparece ningún aviso que no aparezca hoy.
5. **`transformer=None`** devuelve el mismo objeto: añadir `assert results is input_array` al test existente.

En los forecasters:

- `inf` en `last_window` lanza el mismo error en un forecaster de una serie y en uno multiserie (hoy ese error no tiene ningún test);
- `set_out_sample_residuals` con arrays vacíos sigue lanzando `ValueError`;
- un forecaster guardado con `save_forecaster` (pickle y skops) y cargado predice igual.

Además: los tests de los módulos de los forecasters afectados y `review_t1_forecasters.py` (equivalencia por bytes de los seis forecasters; para T1 sustituye a `check_equivalence.py`).

**Aceptación.** En los scripts `review_t1_*`, la fila `PACKAGE` compara la función del paquete con la anterior a T1 y cuenta las llamadas que toman el atajo, así que sirven tal cual para aceptar la implementación:

- identidad byte a byte en todos los tests y en `review_t1_forecasters.py`, con pandas 2.3 y 3 y con scikit-learn 1.6 (entorno de versiones mínimas). El prototipo ya la cumple con scikit-learn 1.6.0, 1.7.2, 1.8.0 y 1.9.1 (V2);
- el atajo se aplica en el 100 % de las llamadas elegibles de los flujos de referencia, medido con un contador: 600 de 600 en `predict` de 168 pasos de `ForecasterRecursiveMultiSeries` en M, y 2.000 de 2.000 en un backtesting de 1.000 folds de `ForecasterRecursive`;
- el coste por llamada de `transform_numpy` con un array 1D de 168 valores baja a menos de 10 µs (hoy, de 55 a 92 µs);
- de extremo a extremo, `predict` de 168 pasos de `ForecasterRecursiveMultiSeries` en M es al menos ×1,3 más rápido con Ridge y ×1,1 con LightGBM, medido con las variantes intercaladas, al menos 15 rondas y la mediana de la razón pareada por ronda (`review_t1_bench.py macro`). No se exige ganancia en el backtesting multiserie;
- ninguna regresión de más del 5 % en `benchmarks/` (V7).

**Nota de versión** (sección *Changed*, área de rendimiento): "`transform_numpy` applies a fitted `StandardScaler` without scikit-learn's input validation, with the same result. `predict` and `predict_interval` of the forecasters with `transformer_y` or `transformer_series` are faster (about N times with M series and a linear estimator), and so is the backtesting of a single series with many folds." Pon N y M al medir la implementación. No menciones el backtesting multiserie.

---

### T2. Codificación de las exógenas `category` por códigos

**Problema.** Los forecasters codifican las exógenas categóricas con un `OrdinalEncoder` de scikit-learn:

```python
OrdinalEncoder(dtype=float, handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan).set_output(transform="pandas")
```

Con columnas `category`, scikit-learn las convierte a arrays de objetos y las mapea con diccionarios. El resultado equivale a "posición de la categoría en `categories_`", que se obtiene con los códigos enteros de pandas y una tabla de búsqueda.

**Ganancia medida con el prototipo:**
- construcción de `X_train` en L: de 6,41 a 3,43 s (×1,9);
- en M: de 1,71 a 0,97 s con Ridge (×1,76) y de 1,82 a 1,01 s con LightGBM (×1,81);
- `fit` de Ridge en M: de 2,74 a 1,91 s (×1,43);
- backtesting con refit (L, Ridge, junto con T1): de 71,8 a 48,1 s. Lo midió un prototipo anterior que además construía lags y window features por lotes, cambio que no dio ganancia medible;
- debería eliminar el pico de memoria transitoria del codificador (0,6 GiB en L). Está por medir (V9).

**Dónde** (números de línea en `c694e19`):

| Fichero | Creación del codificador | `fit_transform` / `transform` en `_create_train_X_y` | `transform` en la predicción |
|---|---|---|---|
| `recursive/_forecaster_recursive.py` | 406 | 904, 910 | 1648 |
| `recursive/_forecaster_recursive_classifier.py` | 383 | 911, 917 | 1611 |
| `recursive/_forecaster_recursive_multiseries.py` | 534 | 1632, 1638 | 2847 (exog ancha o larga), 2957 (exog en dict) |
| `direct/_forecaster_direct.py` | 597 | 1120, 1126 | 2225 |
| `direct/_forecaster_direct_multivariate.py` | 553 | 1200, 1206 | 2398 |

**Diseño propuesto.** Prototipo de referencia: `FastOrdinalEncoder` en `dev/preprocessing_speedups/prototypes.py`.

1. **No cambiar el atributo público.** `self.categorical_encoder` sigue siendo un `OrdinalEncoder` de scikit-learn con los mismos parámetros. Así no cambian:
   - la serialización (`save_forecaster` / `load_forecaster` con pickle y skops, que reconstruye tipos de scikit-learn conocidos);
   - `__repr__`, `get_params` y la documentación;
   - los forecasters guardados con versiones anteriores, que tienen el codificador ajustado sobre todos los datos y deben poder predecir con la ruta nueva.

   Descartado: una subclase de `OrdinalEncoder`, porque cambia el tipo de un atributo público y obliga a registrar el tipo en skops.
2. **Dos funciones privadas en `skforecast/utils/utils.py`** (nombres orientativos):
   - `_fit_transform_categorical(encoder, X)`: si todas las columnas de `X` son `pd.CategoricalDtype`, ajusta `encoder` sobre una tabla pequeña y codifica con la ruta rápida; si no, `encoder.fit_transform(X)`.
   - `_transform_categorical(encoder, X)`: si todas las columnas son `category`, usa la ruta rápida; si no, `encoder.transform(X)`.

   Los 5 forecasters cambian `self.categorical_encoder.fit_transform(...)` y `.transform(...)` por estas funciones (12 llamadas en total, tabla de arriba).
3. **Ajuste equivalente.**
   - Por cada columna, se toma de los datos una fila de cada categoría presente y, si hay valores faltantes, una fila con NaN: `np.unique(codes, return_index=True)` da la primera posición de cada código, y `-1` es el faltante.
   - Las columnas se rellenan repitiendo su primera fila hasta igualar longitudes.
   - Se ajusta `encoder` sobre esa tabla. scikit-learn ve los mismos valores y el mismo dtype que con los datos completos, así que `categories_` es idéntico, en valores, orden (alfabético, con NaN al final) y dtype (por ejemplo `int64` con categorías enteras), y también lo son `n_features_in_` y `feature_names_in_`.
   - Ojo: la primera versión del prototipo construía la tabla como `object` y obtenía `categories_` de dtype `object` con categorías enteras. La salida era igual, pero el estado ajustado distinto. La versión actual lo evita, y `check_equivalence.py` comprueba ya el dtype.
4. **Codificación rápida.** Por columna:

   ```python
   lookup = pd.Index(encoder.categories_[j]).get_indexer(X[col].cat.categories).astype(float)
   lookup[lookup < 0] = np.nan          # categoría del dtype no vista en el ajuste
   lookup = np.append(lookup, np.nan)   # el código -1 (faltante) cae en esta última posición
   codificada = lookup[X[col].cat.codes.to_numpy()]
   ```

   El resultado es un `DataFrame` float64 con las mismas columnas e índice que devuelve `set_output(transform="pandas")`. Funciona aunque las categorías del dtype estén en otro orden, haya categorías nuevas en `predict` o la columna sea ordenada.
5. **Columnas de texto (`str` u `object`) y mezclas:** en la primera versión, la ruta rápida solo se usa si todas las columnas categóricas son `category`; en cualquier otro caso se usa scikit-learn, como hoy. Una ruta rápida para texto queda para la validación V1.

**Tests.** Fichero nuevo `skforecast/utils/tests/tests_utils/test_categorical_encoding.py`. En cada caso, `assert_frame_equal(check_exact=True)` de la salida y comparación de `categories_` (valores, NaN y dtype) frente a `OrdinalEncoder` directamente:

- categorías de texto y enteras;
- columna ordenada;
- NaN en el ajuste y en la transformación;
- categoría vista en el dtype pero ausente en el ajuste;
- categoría nueva en `transform`;
- categorías del dtype en otro orden en `transform`;
- varias columnas con distinto número de categorías;
- una sola fila;
- columnas mezcladas (`category` y `str`): camino actual;
- pandas 2.3 y 3, con y sin pyarrow (V5).

En los forecasters:
- un test por forecaster que compare `create_train_X_y` y `predict` con y sin la ruta rápida;
- un test de `save_forecaster` / `load_forecaster`, con pickle y con skops, de un forecaster ajustado con la ruta nueva;
- un forecaster ajustado con la versión actual (codificador ajustado con todos los datos) que prediga con la ruta nueva (V8).

**Aceptación:**
- identidad bit a bit de `X_train`, `y_train`, predicciones, intervalos y `categories_` (valores y dtype);
- construcción de `X_train` en M al menos ×1,5 más rápida con exógenas `category`;
- sin regresión con exógenas de texto (deben ir por el mismo camino de hoy);
- `benchmarks/` sin regresiones (V7).

**Nota de versión:** "The categorical exogenous variables with `category` dtype are encoded with their integer codes instead of scikit-learn's `OrdinalEncoder` conversion to object arrays, with the same result. Training matrices are built up to N times faster and use less memory." Pon N al medir.

---

### T3. Estudio: la predicción sobre `X_train` en `fit`

**Qué pasa hoy.** Con `_probabilistic_mode` distinto de `False`, que es el valor por defecto (`"binned"`), `fit`:

- predice sobre todo `X_train` (`self.estimator.predict(X_train_estimator)`);
- ajusta un `QuantileBinner` por serie con esas predicciones (`_binning_in_sample_residuals`);
- guarda los bordes (`binner_intervals_`);
- si `store_in_sample_residuals=True`, guarda una muestra de residuos por bin;
- copia filas de `X_train` para comprobar que el estimador no lo modifica (`_copy_rows_to_check` y `_check_in_place_fit`, ISSUE-1506).

`backtesting_*` ya pone `_probabilistic_mode = False` cuando no hay `interval` (`model_selection/_validation.py`, líneas 752 y 1737), así que no lo pagan ni el backtesting ni las búsquedas basadas en él (sin intervalos). Lo pagan los `fit` directos del usuario. Las búsquedas con `OneStepAheadFold` no pasan por `fit`: confírmalo en el estudio.

**Coste medido** (`ForecasterRecursiveMultiSeries`, `fit` sin `store_in_sample_residuals`):

| Tamaño | LightGBM | HGBR | Ridge |
|---|---|---|---|
| S | 9 % | 10 % | 2 % |
| M | 14 % | 19 % | 3 % |
| L | 13 % (4,8 s) | 17 % (7,7 s) | 3 % |

**Para qué se usa el binner** (`recursive/_forecaster_recursive_multiseries.py`):
- bootstrapping con residuos por bins (líneas 3303 y 3764);
- conformal con residuos por bins (3962);
- `set_out_sample_residuals`, que agrupa los residuos fuera de muestra por los bins del ajuste (4819 y 4938).

La misma lógica está en `ForecasterRecursive` (línea 1394), `ForecasterDirect` (1936), `ForecasterDirectMultiVariate` (2084), `ForecasterRnn` (1134), `ForecasterEquivalentDate` (455) y `ForecasterRecursiveClassifier`.

**Pregunta que hay que contestar.** ¿Se puede evitar este coste cuando el usuario no va a pedir intervalos, sin cambiar ningún resultado cuando sí los pide y sin romper la API?

**Opciones a evaluar:**

| Opción | Descripción | A favor | En contra |
|---|---|---|---|
| O1 | Parámetro explícito (en `__init__` o en `fit`) para no preparar los intervalos | Sin cambios de resultado | Con el valor por defecto actual no gana nada; con otro valor por defecto cambia el comportamiento |
| O2 | No preparar nada en `fit` y calcularlo bajo demanda con `set_in_sample_residuals(series, exog)` (ya existe) la primera vez que se pide un intervalo | Gana en todos los `fit` directos | Cambio de API: `predict_interval` sin preparación previa tendría que lanzar un error claro o recalcular con datos que ya no tiene |
| O3 | Ajustar el binner con una muestra de filas | Gana siempre | Cambia los bordes de los bins y, por tanto, los intervalos. Habría que medir la cobertura |
| O4 | Mantenerlo y documentarlo, con el coste en la guía de rendimiento | Sin riesgo | Sin ganancia |

**Qué medir:**
1. El porcentaje del `fit` en `ForecasterRecursive`, `ForecasterDirect` (que predice con cada estimador de cada paso: el coste puede multiplicarse por `steps`) y `ForecasterDirectMultiVariate`, con LightGBM, HGBR, XGBoost y Ridge.
2. Qué flujos de usuario dependen de la preparación en `fit`: los intervalos justo después de `fit` y `set_out_sample_residuals` con bins. Cuántos aparecen en la documentación (`docs/` y `skills/prediction-intervals/`).
3. Para O3, la diferencia en cobertura y en anchura de los intervalos con varias fracciones de muestreo.

**Criterio de decisión (decisión de Joaquín):** si con la opción elegida el coste sigue siendo de más del 10 % del `fit` con árboles, y el cambio de API se puede introducir con un aviso de deprecación en una versión, avanzar con O2 u O1. Si no, O4.

**Entregable:** una fila en la hoja de ruta (sección 5) con la decisión y, si procede, la tarea de implementación con sus propios tests y nota de versión.

---

### T4. Validación one-step-ahead multiserie: la inversa del escalador, por `transform_numpy`

**Origen.** Hallazgo 7 de [`REVIEW_T1.md`](REVIEW_T1.md). Decidido el 2026-11-18: tarea y PR propios, fuera de T1.

**Problema.** En `_predict_and_calculate_metrics_one_step_ahead_multiseries` (`skforecast/model_selection/_utils.py`, línea 2707), la rama de `ForecasterRecursiveMultiSeries` deshace la transformación llamando directamente a `transformer.inverse_transform` con un `DataFrame` de una columna, dos veces por serie y candidato (líneas 3046 y 3049). El resto del paquete lo hace con `transform_numpy` y un array, incluida la rama de `ForecasterDirectMultiVariate` de la misma función (líneas 2905 y 2911) y la rama con diferenciación (`_revert_differentiation_one_step_ahead`, línea 1281).

**Coste medido** (`review_t1_one_step_ahead.py`; `grid_search_forecaster_multiseries` con `OneStepAheadFold`, M, Ridge, 3 candidatos): 1.800 llamadas, 1,06 s de 9,70 s (10,9 % de la búsqueda), 587 µs por llamada. La parte crece con el número de candidatos, porque `X_train` se construye una sola vez.

**Diseño propuesto.** Sustituir las dos llamadas por `transform_numpy`, como en la rama de `ForecasterDirectMultiVariate`:

```python
predictions_level["y_true"] = transform_numpy(
    array             = predictions_level["y_true"].to_numpy(),
    transformer       = transformer,
    fit               = False,
    inverse_transform = True
)
```

y lo mismo con `y_pred`.

**Ganancia esperada.** No depende de T1: solo con pasar por `transform_numpy`, cada llamada costaría lo que el camino numpy actual, de 55 a 92 µs (estimado con el coste medido en T1, no medido en este flujo). Con T1, unos 3 µs.

**Qué validar antes** (la identidad de este cambio no se ha comprobado):

- la entrada del transformador pasa de `DataFrame` a array para cualquier transformador, no solo `StandardScaler`. Comprobar por bytes las métricas de la búsqueda con `StandardScaler`, `MinMaxScaler`, un `FunctionTransformer` y un `Pipeline`;
- los escaladores de `transformer_series_` se ajustan sin nombres y hoy reciben aquí un `DataFrame` con la columna `y_true` o `y_pred`: comprobar qué aviso emite scikit-learn hoy y que el cambio no altera los avisos que ve el usuario;
- el dtype de `y_pred` (float32 con `XGBRegressor`) y el de la columna asignada;
- un transformador que escribe en su entrada: `transform_numpy` le pasa una copia (#1504). Hay un test equivalente para la rama de `ForecasterDirectMultiVariate` (`..._ForecasterDirectMultiVariate_with_in_place_transformer`).

**Tests.** `skforecast/model_selection/tests/tests_utils/test_predict_and_calculate_metrics_one_step_ahead_multiseries.py` ya tiene casos de `ForecasterRecursiveMultiSeries` con `transformer_series=StandardScaler()` y valores esperados escritos a mano, que deben seguir pasando sin tocarlos. Añadir un dict de transformadores con una serie sin transformador y un transformador que escribe en su entrada.

**Aceptación:**

- métricas idénticas a las de hoy en los tests existentes y en la comprobación por bytes;
- el tiempo de esas llamadas baja al menos un 80 % en el caso medido (de 1,06 s a menos de 0,2 s);
- ninguna regresión de más del 5 % en `benchmarks/` (V7).

**Nota de versión** (sección *Changed*, área de rendimiento): "The one-step-ahead validation of `ForecasterRecursiveMultiSeries` reverts the transformation of each series with numpy arrays instead of DataFrames, which speeds up the searches with `OneStepAheadFold` and `transformer_series` (about N% with M series)." Pon N y M al medir.

---

## 3. Validaciones pendientes

| ID | Qué validar | Bloquea | Cómo |
|---|---|---|---|
| V1 | Exógenas categóricas de texto (`str` en pandas 3, `object` en pandas 2) | Una posible ampliación de T2 | Con texto, el coste actual es el mismo que con `category` (construcción en M: 1,69 s frente a 1,71 s), y la ruta rápida de T2 no aplica. Prototipar con `pd.factorize` (o `pd.Categorical(col, categories=encoder.categories_[j])` en `transform`) y comprobar la identidad bit a bit con `OrdinalEncoder`, incluido el orden de `categories_` con mayúsculas, acentos y números como texto. Medir con pandas 2.3, pandas 3 sin pyarrow y pandas 3 con pyarrow |
| V2 | scikit-learn 1.6 (versión mínima): la aritmética de `StandardScaler` y el orden de `categories_` de `OrdinalEncoder` pueden cambiar entre versiones | T2 (hecha para T1) | **T1, hecha el 2026-11-18:** el atajo es idéntico byte a byte al camino actual con scikit-learn 1.6.0, 1.7.2, 1.8.0 y 1.9.1 (`review_t1_contract.py`; con pandas 2.3.0 y 3.1.1). **T2, pendiente:** entorno con las versiones mínimas (`uv pip install --resolution lowest-direct -e ".[test]"`, ver `unit-tests-min-deps.yml`) y `check_equivalence.py` |
| V3 | Otros escaladores: `MinMaxScaler` (`X *= scale_; X += min_`, `clip`), `RobustScaler` (`center_`, `scale_`, `with_centering`, `with_scaling`) | Ampliación de T1 | Leer su `transform` / `inverse_transform` en la versión mínima y en la última; mismos tests que T1. Comprobar float32 en cada versión soportada o dejarlo fuera: scikit-learn 1.8 cambió la aritmética float32 de `StandardScaler` (`REVIEW_T1.md`, hallazgo 13) |
| V4 | El resto de forecasters: `ForecasterRecursive`, `ForecasterDirect`, `ForecasterDirectMultiVariate`, `ForecasterRecursiveClassifier` (T2), `ForecasterRnn` y `ForecasterStats` (T1) | T2 (hecha para T1) | **T1, hecha el 2026-11-18:** equivalencia por bytes en los seis forecasters (`review_t1_forecasters.py`, 15 configuraciones) y ganancia medida en cada uno (`review_t1_bench.py others`). **T2, pendiente:** medir antes y después con `benchmarks/` y con un caso equivalente al de `bench.py`. `check_equivalence.py` ya cubre `ForecasterRecursive` |
| V5 | Sin pyarrow: con pandas 3 sin pyarrow, las categorías de texto no son Arrow y el codificador actual cuesta menos | T2 (para documentar la ganancia) | `bench.py` en un entorno sin pyarrow. Con pandas 2.3, la construcción en M bajó de 1,34 a 0,93 s con T2 (×1,45), frente a ×1,8 con pandas 3 y pyarrow |
| V6 | Estimadores con categóricas nativas (LightGBM `categorical_feature`, HGBR `categorical_features`, XGBoost `enable_categorical`, CatBoost vía `cast_catboost_categorical_columns_dataframe`) | T2 | Si `X_train` es idéntico bit a bit, su ajuste también lo es. Comprobarlo con un caso por estimador en `check_equivalence.py` (CatBoost convierte las columnas a enteros: confirmar que recibe los mismos códigos) |
| V7 | Suite `benchmarks/` antes y después | T1, T2, T4 | Ver MAINT-bench-pandas3 en la hoja de ruta; diferencias de menos del 5 % son ruido |
| V8 | Serialización y compatibilidad hacia atrás | T2 | Guardar con `save_forecaster` (pickle y skops) un forecaster ajustado con la versión actual, cargarlo con la nueva y predecir: mismas predicciones. Y al revés |
| V9 | Memoria | T2 (para la nota de versión) | `tracemalloc` alrededor de `_create_train_X_y` antes y después: se espera eliminar ~0,46 veces el tamaño de `X_train` |
| V10 | La exógena en dict en `predict` multiserie (`reindex_like` por serie y `pd.concat`, ~0,3 s con 1.000 series) | Ninguna (otra tarea) | Encaja en PERF-multiseries-names-pyarrow, que la menciona sin analizar |
| V11 | Identidad bit a bit de T1 fuera de Linux x86-64 (macOS, Windows) y con Python 3.13 y 3.14 | Ninguna (la cubre el CI) | La revisión solo se ejecutó en Linux x86-64 con Python 3.11 y 3.12. Los tests de T1 comparan por bytes con scikit-learn en la versión instalada, así que la matriz de `unit-tests.yml` (ubuntu, macOS y Windows; Python 3.11 a 3.14) lo valida en el PR del release a `main` |
| V12 | Forecasters guardados con versiones anteriores a la 0.27.0 | Ninguna | La revisión cargó ficheros joblib, pickle y skops guardados con la 0.27.0 y la 0.28.0 (45 ficheros, predicciones idénticas). El atajo solo lee `mean_`, `scale_`, `n_features_in_`, `with_mean` y `with_std` del escalador, y cae en el camino actual si no tienen el tipo esperado |
| V13 | Identidad del cambio de T4 (la entrada del transformador pasa de `DataFrame` a array) | T4 | Ver «Qué validar antes» en T4 |

---

## 4. Kit de validación: `dev/preprocessing_speedups/`

| Fichero | Qué hace |
|---|---|
| `data.py` | Datos sintéticos del estudio (tamaños T, S, M y L), el forecaster de referencia y el informe del entorno |
| `prototypes.py` | `FastOrdinalEncoder` (T2) y `fast_transform_numpy` (T1, alineado con el diseño revisado), aplicados en tiempo de ejecución sin tocar `skforecast/` |
| `check_equivalence.py` | Identidad bit a bit de los dos prototipos: base (Ridge y LightGBM, ordinal y onehot), NaN en categóricas, categoría no vista, categorías reordenadas, categorías ordenadas y enteras, exógenas que faltan, columna de texto (camino actual), backtesting con refit y `ForecasterRecursive` |
| `bench.py` | Actual frente a T2 (`P1`), T1 (`P2`) y ambos, en la construcción de `X_train`, `fit`, `predict` (24 y 168), conformal y backtesting (refit `False` y `True`). Mediana, dispersión y primera llamada. `--text-cats` para V1 |
| `results/` | JSON de las mediciones de referencia de este documento y salidas de la revisión de T1 (`review_t1_*`) |
| `REVIEW_T1.md` | Revisión crítica de T1: hallazgos por gravedad, inventario de llamadas, contrato de `StandardScaler` y lo que no se pudo comprobar |
| `review_t1_common.py` | Referencia y variantes que comparan los scripts, trazador y contador de llamadas. La referencia (`current` en las salidas) es `transform_numpy` tal como estaba antes de T1, leído con git del commit `c9172e1`. La variante `PACKAGE` es la función que hay ahora en el paquete: antes de implementar T1 coincide con la referencia, y después es la implementación que hay que aceptar |
| `review_t1_inventory.py` | Inventario en ejecución de las llamadas a `transform_numpy` en cada forecaster |
| `review_t1_contract.py` | Contrato de `StandardScaler`, identidad por bytes y casos límite. Se ejecuta en cada entorno de scikit-learn |
| `review_t1_forecasters.py` | Equivalencia por bytes de extremo a extremo en los seis forecasters (para T1 sustituye a `check_equivalence.py`) |
| `review_t1_saved.py` | Forecasters guardados con joblib, pickle y skops, también con una versión anterior |
| `review_t1_empty_inputs.py` | Arrays vacíos por la API pública |
| `review_t1_bench.py` | Coste por llamada (`micro`), extremo a extremo con variantes intercaladas y razón pareada (`macro`), parte del backtesting que cubre el atajo (`backtesting`) y resto de forecasters (`others`) |
| `review_t1_one_step_ahead.py` | Llamadas directas a `inverse_transform` en la búsqueda one-step-ahead multiserie (T4) |

Comandos (desde la raíz del repositorio):

```bash
python dev/preprocessing_speedups/check_equivalence.py
```

```bash
python dev/preprocessing_speedups/bench.py --size M --est ridge --reps 3
```

Para T1 (y T4):

```bash
python dev/preprocessing_speedups/review_t1_contract.py
```

```bash
python dev/preprocessing_speedups/review_t1_forecasters.py
```

```bash
python dev/preprocessing_speedups/review_t1_bench.py macro --size M --est ridge --rounds 25
```

`review_t1_contract.py` y `review_t1_forecasters.py` se ejecutan también con el intérprete del entorno de versiones mínimas (`~/.venvs/skforecast-min/bin/python`, ver `tools/ai/ai_context_header.md`).

Al implementar T1 o T2, el prototipo deja de ser necesario: la comprobación de equivalencia se convierte en los tests de la tarea, y `bench.py` sirve para medir la rama frente a `0.28.x` (ejecutándolo en cada una). Para T1, mide con `review_t1_bench.py`: `bench.py` hace 2 o 3 repeticiones por bloques (una en el backtesting) y no distingue diferencias de menos de un 20 % en `predict` ni de un 10 % en backtesting. Mide siempre con la máquina libre: LightGBM y HGBR usan OpenMP y se degradan mucho con otra carga (en el estudio se vio una repetición de 202 s frente a 4,7 s).

**Resultados de referencia** (M, `bench.py`, pandas 3.1.1 con pyarrow):

| Estimador | Flujo | Actual | T2 (P1) | T1 (P2) | T1 + T2 |
|---|---|---|---|---|---|
| Ridge | `build_X_train` | 1,707 s | 0,971 s (×1,76) | 1,678 s (×1,02) | 0,891 s (×1,92) |
| Ridge | `fit` | 2,735 s | 1,913 s (×1,43) | 2,505 s (×1,09) | 1,678 s (×1,63) |
| Ridge | `predict_24` | 0,221 s | 0,200 s (×1,10) | 0,141 s (×1,57) | 0,146 s (×1,51) |
| Ridge | `predict_168` | 0,194 s | 0,164 s (×1,18) | 0,109 s (×1,77) | 0,090 s (×2,15) |
| Ridge | `interval_conformal_24` | 0,345 s | 0,315 s (×1,10) | 0,222 s (×1,56) | 0,250 s (×1,38) |
| Ridge | `backtesting_refit=False` | 5,417 s | 4,369 s (×1,24) | 5,426 s (×1,00) | 4,729 s (×1,15) |
| Ridge | `backtesting_refit=True` | 22,694 s | 16,430 s (×1,38) | 19,178 s (×1,18) | 15,374 s (×1,48) |
| LightGBM | `build_X_train` | 1,815 s | 1,005 s (×1,81) | 1,936 s (×0,94) | 1,185 s (×1,53) |
| LightGBM | `fit` | 10,545 s | 10,781 s (×0,98) | 10,647 s (×0,99) | 10,105 s (×1,04) |
| LightGBM | `predict_24` | 0,224 s | 0,214 s (×1,05) | 0,152 s (×1,48) | 0,148 s (×1,52) |
| LightGBM | `predict_168` | 0,264 s | 0,249 s (×1,06) | 0,226 s (×1,16) | 0,193 s (×1,36) |
| LightGBM | `interval_conformal_24` | 0,354 s | 0,332 s (×1,07) | 0,302 s (×1,17) | 0,278 s (×1,27) |
| Ridge, categóricas como texto | `build_X_train` | 1,686 s | 1,979 s (×0,85) | 1,854 s (×0,91) | 1,826 s (×0,92) |
| Ridge, categóricas como texto | `fit` | 2,487 s | 2,502 s (×0,99) | 2,450 s (×1,02) | 2,520 s (×0,99) |
| Ridge, categóricas como texto | `predict_24` | 0,264 s | 0,253 s (×1,04) | 0,144 s (×1,83) | 0,153 s (×1,73) |
| Ridge, categóricas como texto | `predict_168` | 0,166 s | 0,177 s (×0,94) | 0,127 s (×1,30) | 0,139 s (×1,19) |
| Ridge, categóricas como texto | `interval_conformal_24` | 0,330 s | 0,440 s (×0,75) | 0,265 s (×1,24) | 0,274 s (×1,20) |

Cómo leerla:

- **Repeticiones:** `build_X_train`, `predict` y conformal son la mediana de 3 repeticiones (2 con LightGBM); `fit` con LightGBM y el backtesting, de 1 (más la primera llamada). Diferencias de menos de un 10 % en esas filas son ruido.
- **Columna T1 (P2) y filas de backtesting: no las uses para T1.** Al repetir el mismo `bench.py` el 2026-11-18, `predict_168` con Ridge dio ×1,36 (la tabla dice ×1,77) y el backtesting con refit ×0,89 (la tabla dice ×1,18). Las cifras de T1 que valen son las de su sección, medidas con `review_t1_bench.py`. La columna T2 (P1) no se ha revisado.
- **Fila de texto `build_X_train` (×0,85):** es ruido. Remedido intercalando las variantes, 7 repeticiones: 1,76 s actual y 1,78 s con P1. Con texto, la ruta rápida no aplica y P1 sigue el camino actual.
- **`fit` con LightGBM no cambia de forma apreciable,** porque manda el estimador. Lo esperable es ahorrar el tiempo de construcción (~0,8 s de 10,5 s en M, ~3 s de 37,5 s en L).

---

## 5. Orden, PRs y hoja de ruta

1. **T1** (rama `perf/transform-numpy-known-scalers`, desde `origin/0.28.x`): el cambio más pequeño y localizado. Revisada (`REVIEW_T1.md`), con V2 y V4 hechas para el escalador. Pendiente: V7.
2. **T4** (rama `perf/one-step-ahead-inverse-numpy`): dos llamadas en `model_selection/_utils.py`. No depende de T1 y puede ir antes o después. Validaciones V7 y V13.
3. **T2** (rama `perf/categorical-codes-encoding`): V2 (codificador), V4, V5, V6, V8 y V9.
4. **T3** (estudio, sin rama de código hasta la decisión).

Cada PR va contra `0.28.x` (o la rama de la versión en desarrollo), con su nota en `docs/releases/releases.md`, el lint y los tests de los módulos tocados antes del commit. Ninguno cambia la API pública. Si T3 la cambia, hay que actualizar también la referencia de la API (`tools/ai/llms-base.txt`) y la documentación de usuario (`skills/`), y regenerar los ficheros derivados.

Filas propuestas para `dev/roadmap_0.28.0/roadmap.md` (o la hoja de ruta de la versión siguiente), en el formato de la sección de rendimiento:

| ID | Qué | Esfuerzo | Estado | Referencia |
|---|---|---|---|---|
| PERF-transform-numpy-scalers | Atajo en `transform_numpy` para un `StandardScaler` ajustado (misma aritmética, sin la validación de scikit-learn): `predict` multiserie de 168 pasos con 300 series entre ×1,4 y ×1,5 con Ridge y entre ×1,2 y ×1,3 con LightGBM; backtesting de una serie con 1.000 folds ×2,2; sin ganancia apreciable en el backtesting multiserie | S | Candidata, revisada; prototipo idéntico byte a byte con scikit-learn 1.6 a 1.9 | `dev/preprocessing_speedups/PLAN.md` (T1) y `REVIEW_T1.md` |
| PERF-one-step-ahead-inverse | En la validación one-step-ahead de `ForecasterRecursiveMultiSeries`, deshacer la transformación con `transform_numpy` en lugar de llamar a `inverse_transform` con un DataFrame por serie y candidato (10,9 % de una búsqueda con 300 series y 3 candidatos) | S | Candidata, identidad por comprobar | `dev/preprocessing_speedups/PLAN.md` (T4) |
| PERF-categorical-codes | Codificar las exógenas `category` con sus códigos en lugar de `OrdinalEncoder` (mismo `categories_` y misma salida): construcción de `X_train` ×1,8 a ×1,9 con pandas 3, ×1,45 con pandas 2.3; −0,46 veces `X_train` de memoria transitoria | S a M | Candidata, prototipo idéntico bit a bit | `dev/preprocessing_speedups/PLAN.md` (T2) |
| STUDY-fit-insample-binning | `fit` predice sobre todo `X_train` para los bins de residuos aunque no se pidan intervalos (13 a 17 % del `fit` con árboles y 1.000 series) | S (estudio) | Pendiente de decisión | `dev/preprocessing_speedups/PLAN.md` (T3) |

---

## 6. Riesgos

- **Identidad numérica:** T1 y T2 están diseñados para ser idénticos bit a bit, y el kit lo comprueba. El riesgo está en versiones de scikit-learn distintas de la 1.9.1 (V2). Para T1 ya está comprobado en float64 con scikit-learn 1.6.0, 1.7.2, 1.8.0 y 1.9.1; para T2 sigue pendiente. Si una versión cambia la aritmética del escalador o el orden de `categories_`, la ruta rápida debe seguirla o desactivarse para esa versión. Hay un precedente: scikit-learn 1.8 cambió la aritmética float32 de `StandardScaler`, que por eso queda fuera del atajo.
- **Atributos públicos:** `categorical_encoder` sigue siendo el mismo tipo y queda ajustado igual. Ningún atributo cambia de valor.
- **Superficie:** T2 toca 12 llamadas en 5 forecasters; T1 una sola función; T4 dos llamadas en `model_selection/_utils.py`, donde cambia el tipo de la entrada para cualquier transformador. Las dos rutas rápidas caen en el camino actual ante cualquier caso que no reconozcan, de modo que el peor caso es "sin ganancia", no "resultado distinto". Esto solo se cumple si las condiciones cubren también las entradas que hoy son un error: con la primera versión de T1, un array vacío o una `Series` daban un resultado donde hoy hay una excepción (`REVIEW_T1.md`, hallazgos 3 y 4). Al diseñar una ruta rápida hay que repasar los casos de error del camino actual, no solo los de éxito, y añadir un test que compruebe qué camino se toma.
- **Expectativas:** con LightGBM o HGBR, el `fit` apenas mejora (lo que ahorra la construcción: ~8 %, dentro del ruido de una repetición) porque manda el estimador. La ganancia visible está en `predict`, los intervalos, los modelos lineales y el backtesting con refit (este último por T2: T1 no acelera el backtesting multiserie). Conviene decirlo así en la nota de versión.
- **Medidas:** en la máquina del estudio, dos ejecuciones de `bench.py` del mismo cambio dieron ×1,18 y ×0,89 en el backtesting con refit. Con pocas repeticiones no se pueden decidir diferencias pequeñas: usa variantes intercaladas y la razón pareada por ronda (`review_t1_bench.py`), o criterios que no dependan del reloj (contar las llamadas que usan la ruta rápida, coste por llamada).
