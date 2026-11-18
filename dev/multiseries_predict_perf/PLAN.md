# Plan: `ForecasterRecursiveMultiSeries` más rápido en `predict` y backtesting (fila PERF-multiseries-names-pyarrow)

## Contexto

`ForecasterRecursiveMultiSeries` es el forecaster más usado y con pandas 3 pierde en `predict` entre un 10 y un 51 % frente a pandas 2 (20 series, según exog y `pyarrow`). La fase 1 midió dónde se va el tiempo y prototipó once cambios pequeños en una copia del árbol. Con todos ellos, pandas 3 queda por debajo de lo que tarda hoy pandas 2 y pandas 2 gana lo mismo; los resultados, avisos y errores son idénticos a la base en los tres entornos.

Medido con 20 series (base → prototipo; pandas 2 / pandas 3 sin `pyarrow` / con `pyarrow`):

| Caso | p2 | p3 sin pa | p3 con pa |
|---|---|---|---|
| `predict` sin exog (µs) | 661 → 264 | 768 → 306 | 939 → 385 |
| `predict` exog ancha (µs) | 1153 → 409 | 1256 → 470 | 1486 → 589 |
| `predict` exog diccionario, un objeto por serie (µs) | 2393 → 1108 | 2616 → 1221 | 3090 → 1360 |
| `predict` exog formato largo (µs) | 3611 → 2007 | 4408 → 2269 | 5267 → 2976 |
| Backtesting sin refit, 1 paso, exog ancha (ms por fold) | 8,6 → 2,3 | 9,0 → 2,4 | 9,6 → 2,5 |
| Backtesting LightGBM, 24 pasos, exog ancha (ms por fold) | 12,9 → 6,9 | 13,4 → 7,0 | 14,4 → 7,5 |

Esta carpeta: el estado de cada cambio y las decisiones están en [SEGUIMIENTO.md](SEGUIMIENTO.md) (lo primero que hay que leer al retomar el trabajo), las medidas de la fase 1 y del prototipo en [MEDIDAS.md](MEDIDAS.md), los scripts en [scripts/](scripts/) y el diff del prototipo (453 líneas, tres ficheros) en [proto_patch/](proto_patch/). Árbol base de las comparaciones: commit `cb03fbe96`, cuyo paquete es idéntico al de `0.28.x` (`c694e1915`).

## Revisión final del plan (2026-11-17)

**Conclusión.** Los once cambios son correctos y seguros, pero no bastan para el caso de muchas series: quitan sobre todo coste fijo por llamada, que es lo que domina con 20 series. Con 100 series y una exog por serie, lo que queda es casi todo coste por serie, y el plan original no lo tocaba. Añado el bloque B para eso.

**Comprobado en el código (riesgos del bloque A):**
- `transform_numpy` copia el array cuando hay transformador y lo devuelve sin tocar cuando no lo hay, así que dejar de copiar la ventana (cambio 2) no expone los datos del usuario.
- Los tres llamantes de `_extract_data_folds_multiseries` (`_backtesting_forecaster_multiseries`, `_backtesting_foundation`, `_train_test_split_one_step_ahead`) crean diccionarios nuevos a partir de lo que reciben; ninguno modifica el conjunto de entrenamiento que el cambio 1 reutiliza.
- El cambio 3 está en código compartido con `ForecasterDirectMultiVariate` y `ForecasterRnn`; la batería del bloque anterior (2911 casos, que incluye el directo multivariante) da idéntico.

**Dos correcciones para que ningún caso empeore:**
- Cambio 2: comparar `len(levels)` con el número de columnas antes de convertir las columnas a lista. Sin esto, predecir unos pocos niveles con una ventana de muchas columnas pagaría una conversión de más (unos 0,5 µs por columna con `pyarrow`).
- Cambio 4: comprobar primero el índice (`index.equals`, que falla rápido si las longitudes difieren) y solo después las columnas. Sin esto, una exog por serie más larga que `steps` pagaría la conversión de columnas y además el realineado.

**Lo que queda con 100 series tras el bloque A** (perfil del prototipo, µs por llamada o fold; p2 / p3 sin pa / p3 con pa):

| Dónde | Línea | Coste | % de lo que queda |
|---|---|---|---|
| `predict`, exog por serie | `isna` de la exog de cada serie (`check_predict_input`) | 1294 / 1873 / 1978 | 24 / 32 / 30 |
| `predict`, exog por serie | comparar la primera fecha de cada exog (`_check_exog_alignment`) | 924 / 1226 / 1306 | 17 / 21 / 20 |
| `predict`, exog por serie | `pd.concat` de la exog de todas las series | 820 / 604 / 653 | 15 / 10 / 10 |
| Fold de backtesting | cortar la ventana de cada serie y unirlas en una tabla | 5296 / 5233 / 5346 | 59 / 58 / 56 |

Las dos primeras líneas concentran además la pérdida que le queda a pandas 3 frente a pandas 2 con muchas series: suman +881 µs sin `pyarrow` y +1066 µs con él, cuando el neto de toda la llamada es +508 y +1150 µs (`pd.concat` es más rápido en pandas 3 y compensa una parte). El benchmark oficial del proyecto (`benchmarks/benchmarks/bench_forecaster_recursive_multiseries.py`) usa 600 series con exog en diccionario: es justo este caso.

**Buenas prácticas que añado:**
- El corte del 3 % con 20 series esconde el coste por serie: el bloque B se evalúa con 100 series y se informa también con 600 y 1000.
- Usar el benchmark oficial (600 series) como medida de aceptación antes y después, además de mis casos.
- Añadir a los tiempos cuatro casos adversos (pocos niveles con ventana ancha, exog más larga que `steps` en ancho y en diccionario, exog en diccionario desalineada) para demostrar que nada empeora.
- Documentar en una nota del código que el conjunto de entrenamiento que devuelve la extracción de folds se comparte entre folds y no debe modificarse.

## Decisiones (con mi recomendación)

1. **Cambio 6 (exog compartida por varias series): incluirlo.** Es el más invasivo (tres funciones), pero da el salto de 3,3 a 2,3 ms por fold en backtesting con exog ancha.
2. **`isna` repetido de la exog ancha: fuera.** Gana 2,7 / 3,1 / 2,8 %, por debajo del corte.
3. **`ForecasterDirectMultiVariate`: solo el cambio en código compartido** (cambio 3). Con pandas 3 pierde 0 % y +11 % con 20 series, y lo domina el escalado por serie.
4. **Una sola PR, con un commit por cambio** (no por bloque) y revisión tuya de cada uno antes de pasar al siguiente. **Pedido por ti el 2026-11-17.** La suite completa con pandas 2 y 3 y el arnés se pasan una sola vez, al final.
5. **Bloque B en la misma PR, después de verificar el bloque A**, con el corte evaluado con 100 series (3 % en dos de tres entornos). **Decidido por ti el 2026-11-17.** Cada candidato se prototipa y mide antes de decidir; el que no pase, o no sea idéntico, se queda fuera. Si no pasa ninguno, la PR es el bloque A.

## Bloque A: once cambios medidos

Ganancia en % del tiempo del caso base con 20 series (p2 / p3 sin pa / p3 con pa). Todos pasan el corte de 3 % en dos de tres entornos.

| # | Cambio | Dónde | Caso | Ganancia | Riesgo |
|---|---|---|---|---|---|
| 1 | No recortar el conjunto de entrenamiento (series y exog) en cada fold si sus límites no cambian | `_extract_data_folds_multiseries` | fold sin exog | 53 / 52 / 50 | Bajo. El diccionario se reutiliza entre folds: ningún llamante lo modifica (comprobado) |
| 2 | No reordenar ni copiar la ventana si sus columnas ya son los niveles, en el mismo orden | `_create_predict_inputs` | `predict` sin exog | 22 / 26 / 24 | Medio. Se deja de copiar la ventana; solo se lee |
| 3 | No seleccionar columnas de la ventana si ya son las que hay que comprobar | `check_predict_input` | `predict` sin exog | 21 / 19 / 19 | Bajo. Código compartido con `ForecasterDirectMultiVariate` y `ForecasterRnn` |
| 4 | No realinear la exog de una serie que ya tiene las fechas y columnas esperadas | `_create_predict_inputs` | exog diccionario | 20 / 19 / 16 | Bajo |
| 5 | Repetir la exog ancha con numpy en vez de `pd.concat` de la misma tabla | `_create_predict_inputs` | exog ancha | 18 / 13 / 11 | Bajo. Sin niveles se conserva el camino antiguo, que lanza el mismo error |
| 6 | Exog compartida por varias series: cortarla una vez por fold, comprobarla una vez si no da avisos y compararla una vez | `_extract_data_folds_multiseries`, `check_predict_input`, `_check_exog_alignment`, `_create_predict_inputs` | fold con exog ancha | 12 / 14 / 16 | Medio. Se identifica por `id` del objeto; si da un aviso se comprueba para cada serie, como hoy |
| 7 | Crear la plantilla vacía de exog solo si alguna serie la necesita | `_create_predict_inputs` | exog diccionario | 10 / 9 / 9 | Bajo |
| 8 | Cortar cada serie de un formato largo sobre una tabla sin el MultiIndex | `_split_long_format` | exog formato largo | 8 / 19 / 16 | Bajo. También mejora `create_train_X_y` (3 a 15 %) |
| 9 | No cortar la exog ancha si ya tiene `steps` filas | `_create_predict_inputs` | exog ancha | 3,6 / 5,4 / 4,7 | Bajo |
| 10 | Reutilizar el índice de columnas de la ventana en vez de convertir la lista de nombres | `_create_predict_inputs` | `predict` sin exog | 3,9 / 4,3 / 6,2 | Bajo. La tabla es interna; la salida no cambia |
| 11 | Ventana guardada en `fit`: no extraer la última fecha de cada serie si comparten índice | `preprocess_levels_self_last_window_multiseries` | ventana guardada | 3,8 / 4,1 / 3,6 | Bajo |

Ficheros del paquete: `skforecast/recursive/_forecaster_recursive_multiseries.py`, `skforecast/utils/utils.py`, `skforecast/model_selection/_utils.py`.

### Detalles que la implementación debe respetar

- **Cambio 1**: clave `(train_iloc_start, train_iloc_end)`; se aplica a la rama de diccionario y a la de tabla ancha (la de `ForecasterDirectMultiVariate`). Nota en el código: el conjunto devuelto se comparte entre folds.
- **Cambio 2**: camino rápido solo si `len(levels)` es el número de columnas, la lista de columnas es igual a `levels` y no hay nombres duplicados (con duplicados se mantiene el camino actual, que lanza el error de siempre). La longitud se compara antes de convertir las columnas.
- **Cambio 3**: la comprobación de NaN no depende del orden: basta el mismo conjunto, sin duplicados, comparando antes las longitudes.
- **Cambio 4**: primero `index.equals(prediction_index)`, después `columns.tolist() == nombres de entrenamiento`. Nada de memorizar por `id` del índice de columnas: en pandas 3 cada corte de filas crea un índice de columnas nuevo.
- **Cambio 5**: `np.tile(exog_values.to_numpy(), (n_levels, 1))` solo si la exog no es un diccionario y hay niveles.
- **Cambio 6**: `_check_exog_alignment` pasa a devolver si ha avisado; en `check_predict_input` solo se salta un objeto ya comprobado sin avisos ni errores.
- **Cambio 8**: `data_sorted.index = pd.RangeIndex(len(positions))` tras el `iloc[positions]`; el índice de cada serie se asigna después, como ahora.
- Los comentarios `PROTO Pxx` del prototipo se reescriben como notas `# NOTE:` del estilo del módulo, sin guiones largos.

## Bloque B: coste por serie (candidatos, se miden antes de decidir)

Cotas superiores del perfil con 100 series; la ganancia real se mide con el prototipo de cada uno.

| # | Candidato | Dónde | Cota con 100 series | Riesgo |
|---|---|---|---|---|
| B1 | Montar la última ventana de cada fold a partir de los valores cuando todos los cortes comparten índice (como ya hace `preprocess_levels_self_last_window_multiseries`), y cortar por posición las series que cubren todo el intervalo | `_extract_data_folds_multiseries` | hasta 35 a 40 % del fold | Medio. Hay que conservar dtype por columna, nombre y frecuencia del índice; con series de distinta longitud o dtypes de extensión se mantiene el camino actual |
| B2 | Calcular una vez la primera fecha esperada, en vez de una vez por serie | `check_predict_input`, `_check_exog_alignment` | 6 a 7 % de `predict` con exog por serie | Bajo |
| B3 | Comprobar los NaN de la exog de cada serie sobre sus valores numpy cuando la tabla es pequeña | `check_predict_input` | hasta 18 a 23 % de `predict` con exog por serie | Alto. Una revisión anterior vio que convertir a numpy una exog larga de varios dtypes es más lento (13,5 → 19,8 ms); solo entra si ningún caso de pandas 2 empeora |

Descartado de entrada: sustituir el `pd.concat` de la exog por serie (10 a 15 %) por numpy. Con transformador o variables categóricas hace falta la tabla, y la comprobación de dtypes depende de cómo los combina `concat`.

## Fuera de esta PR

- `isna` repetido de la exog ancha (por debajo del corte).
- Construir de otra forma la columna `level` de la salida: menos del 1 % y arriesga el dtype con nombres mixtos. La salida no se toca.
- Comparar los dtypes de exog como listas: no prototipado, cota de 2 %.
- El `predict` de `ForecasterDirectMultiVariate`.
- Rediseñar la extracción de folds sobre una matriz única de todas las series (dejaría el coste por fold casi independiente del número de series): es un rediseño, no un cambio pequeño. Candidato para la fila PERF-predict-numpy.
- Hallazgo sin tocar: `predict` con `levels=[]`, ventana dada y exog lanza `No objects to concatenate` (error de pandas). Se conserva; candidato a issue.

## Pasos

1. **Preparación** (sin tocar el paquete):
   - Confirmar que el paquete de `origin/0.28.x` sigue igual que `cb03fbe96` (`git diff --stat cb03fbe96 origin/0.28.x -- skforecast`); si ha cambiado, exportar de nuevo el árbol base.
   - Conjuntos de casos de tiempos en `scripts/bench_ms.py` (hecho): `@core` (24 casos con 20 y 100 series: `predict` sin exog, ancha, diccionario compartido y con un objeto por serie, largo, ventana guardada; backtesting sin exog, con exog ancha y en diccionario; LightGBM 24 pasos; `fit` y `create_train_X_y` en ancho y largo), `@adverse` (8 casos en los que un camino rápido no se aplica) y `@scale` (11 casos con 600 y 1000 series, una sola ronda).
2. **Rama** `perf/multiseries-predict-backtesting` desde `origin/0.28.x` con `--no-track` (creada el 2026-11-17).
   **Primer commit de la rama (solo `dev/`)**: carpeta `dev/multiseries_predict_perf/` con:
   - `PLAN.md`: este plan.
   - `SEGUIMIENTO.md`: una fila por cambio (estado: pendiente, en revisión, commit hecho con su hash, o descartado; ganancia medida al implementarlo; fecha) y el registro de las decisiones de cada revisión. Se actualiza al cerrar cada cambio, dentro del commit de ese cambio.
   - `MEDIDAS.md`: el informe de la fase 1 (método, tablas de tiempos, coste fijo y por serie, líneas donde pierde pandas 3) y los resultados del prototipo.
   - `scripts/`: lo necesario para repetir las medidas (tiempos, perfil por líneas, batería de identidad y comparador), con las rutas en variables de entorno. Los resultados no se versionan.
   - El diff del prototipo como referencia (`proto_patch/`).

   La carpeta se borra al fusionar la PR, cuando su contenido útil esté en la fila del roadmap; queda en el historial de git.
3. **Bloque A, un cambio cada vez**, en el orden de la tabla (1 a 11, que es el de ganancia medida y respeta las dependencias: el 7 necesita el 4, y el 6 necesita el 1 y el 4). Para cada cambio sigo el ciclo de la sección "Ciclo de cada cambio" y no empiezo el siguiente hasta que el anterior tenga tu visto bueno y su commit.
4. **Cierre del bloque A**: identidad en los tres entornos y tiempos con el conjunto corto sobre el conjunto de los once commits, para comprobar que la suma reproduce el prototipo.
5. **Bloque B**: B2, B1 y B3 en ese orden (de menor a mayor riesgo). Primero prototipo y medidas de cada uno en una copia del árbol; te las presento y decides si entra. Los que entren siguen el mismo ciclo, un commit cada uno.
6. **Tests**: van en el commit del cambio que cubren, no al final (leer antes `.claude/rules/` de tests; valores esperados fijos, API pública):
   - `utils/tests/tests_utils/test_check_predict_input.py`: ventana con las columnas en otro orden y NaN; exog compartida con NaN avisa una vez por serie; exog compartida limpia no avisa.
   - `recursive/tests/tests_forecaster_recursive_multiseries/test_predict.py` y `test_create_predict_inputs.py`: niveles en otro orden que las columnas, columnas duplicadas (mismo error), exog en diccionario alineada y desalineada, exog ancha con más filas que `steps`, objeto compartido entre series.
   - `model_selection/tests/tests_utils/test_extract_data_folds_multiseries.py`: varios folds sin refit dan lo mismo que cada fold por separado; con refit los conjuntos de entrenamiento cambian; series de distinta longitud.
   - `utils/tests/tests_utils/test_split_long_format.py` y `test_preprocess_levels_self_last_window_multiseries.py`: los existentes deben pasar sin cambios; añadir solo si falta el caso.
7. **Nota de versión** en la sección 0.28.0 de `docs/releases/releases.md` (skill `release-note`) y actualizar la fila PERF-multiseries-names-pyarrow de `dev/roadmap_0.28.0/roadmap.md`, en un último commit. No cambia la API pública: no hay que tocar el contexto de IA.

## Ciclo de cada cambio

Cada cambio es un commit que se puede revisar, probar y revertir solo.

1. **Antes de tocar nada, te explico el cambio**: qué hace hoy el código (con las líneas), qué va a hacer, por qué el resultado es el mismo y en qué condiciones se usa el camino nuevo y en cuáles se queda el actual. Si ves un fallo en el planteamiento, se para aquí.
2. **Implemento solo ese cambio** y sus tests.
3. **Compruebo** (unos 10 minutos por cambio):
   - batería de identidad contra el árbol base en los tres entornos (los resultados de la base ya están guardados, solo corre la rama);
   - tests del módulo tocado, existentes y nuevos, con pandas 2 y pandas 3;
   - lint de los ficheros cambiados;
   - tiempos de los casos a los que afecta y de los casos adversos, en los tres entornos, para aislar la ganancia de ese cambio y ver que ninguno empeora.
4. **Te entrego la ficha de revisión**: el diff, la justificación de equivalencia, los casos límite revisados, los resultados de identidad, tests y tiempos, el riesgo que queda y el mensaje de commit que propongo.
5. **Tú revisas**. Si pides cambios, vuelvo al paso 2. Con tu visto bueno, el commit lo haces tú o me lo pides de forma explícita para ese cambio; no hago commits por mi cuenta.
6. Si un cambio no reproduce su ganancia, no es idéntico o no te convence, se descarta (`git restore`) y se pasa al siguiente; los demás no dependen de él salvo las dependencias indicadas.

Si quieres una segunda opinión independiente, al cerrar el bloque A puedes lanzar `/code-review` sobre la rama, como en las PRs de la migración.

## Verificación final (una vez, con todos los commits)

1. **Identidad** contra el árbol base en los tres entornos, con `scripts/run_identity.sh` (batería multiserie de 4591 casos, 1 a 2 minutos por pasada, y la del bloque anterior de 2911). Criterio: 0 diferencias en resultados, dtypes, índices, avisos y errores.
2. **Tiempos** con el conjunto corto, 2 rondas alternadas, proceso fijado al núcleo 12, base frente a rama en los tres entornos. Criterio: ningún caso empeora en ningún entorno (fuera del ruido de ±4 %), incluidos los cuatro adversos, y las ganancias reproducen las del prototipo. Informe con 20, 100, 600 y 1000 series.
3. **Benchmark oficial** (`bench_forecaster_recursive_multiseries.py`, 600 series): las funciones de `predict`, `check_predict_input`, `_create_predict_inputs` y backtesting, base frente a rama, en pandas 2 y pandas 3.
4. **`verify`** (lint de los ficheros cambiados y tests afectados) en pandas 2 y pandas 3.
5. **Suite completa** con pandas 2 y con pandas 3, una vez, al final. Línea base: solo los 3 fallos conocidos de `test_fit_tensorflow`.
6. **Arnés** `dev/pandas3_migration/harness/run_ab.py` (identidad, tiempo y memoria) contra `origin/0.28.x` en p2 y p3.
7. Informe final con las tablas de tiempos y versiones (Python, numpy, pandas, scikit-learn, `pyarrow`).

Sin push hasta que lo pidas. Los commits, uno por cambio y solo tras tu revisión.
