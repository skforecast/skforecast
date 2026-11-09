# Revisión de `n_jobs='auto'` en skforecast: medidas en Windows con una CPU híbrida

El estudio (scripts, resultados e informe) está en `dev/n_jobs_review/`. Empezó el 7 de noviembre sobre `origin/0.27.x` @ `76e4500dd` (skforecast 0.27.0), en una rama local, y entra en el repositorio con el PR del umbral (rama `perf/n-jobs-auto-work-threshold`). Los demás cambios de código que salieron de él fueron en PRs aparte (sección 14). Cuando se cierre macOS, la carpeta se puede retirar de `dev/` con un PR pequeño antes de publicar la versión.

**Estado a 9 de noviembre de 2026.**
- Las secciones 1 a 11 describen el estudio en Windows tal como se cerró el 7 de noviembre, con notas donde algo cambió después.
- La sección 12 es la medida en Linux (WSL2, D7); la 13, un fallo que apareció durante ella (D9).
- La sección 14 resume el estado de cada decisión. Todas están aplicadas. D8 (el cuaderno de la documentación) va en la misma rama que D4 y D5.
- El umbral de trabajo (D4 y D5) está implementado con los valores de Windows y Linux. Falta medir macOS: la sección 16 explica cómo hacerlo y qué cambiar después.

Etiquetas: **[probado]** medido en esta máquina con los scripts de esta carpeta; **[verificado en código]** leído en el código fuente (skforecast o sus dependencias); **[documentado]** dicho por la documentación de una librería; **[inferido]** deducción sin medida directa.

## 1. Resumen

**Máquina**: Intel Core Ultra 9 185H (16 núcleos físicos, 22 lógicos, híbrida), Windows 11, Python 3.13, joblib 1.6.0, scikit-learn 1.9.1, lightgbm 4.7.0. Además, portátiles emulados con afinidad (2c/4t y 4c/8t). unas 4.100 llamadas cronometradas (1 calentamiento y 5 repeticiones por celda).

**Conclusión principal.** Las reglas por tipo de forecaster y estimador eligen bien *cuántos* workers usar cuando paralelizar compensa. Pero no deciden *si* compensa, y en Windows eso lo marca el coste de arrancar los workers: entre 3 y 6 s en la primera llamada de cada sesión. En un backtesting suelto, `'auto'` es hoy hasta 20 veces más lento que no paralelizar en cargas del tamaño de la documentación [probado, sección 4.1].

La estrategia recomendada combina:
- las reglas actuales, corregidas;
- un umbral sobre el trabajo secuencial estimado a partir del `fit` inicial que el backtesting ya hace;
- decidir una sola vez en las búsquedas.

Con ella, en frío, la pérdida total frente al óptimo baja de 211 a 42 s (L) y de 174 a 15 s (S); en caliente, de 133 a 19 s (L) [sección 10].

| Regla actual | Veredicto | Sección |
|---|---|---|
| `refit` entero → 1 | **Mantener** (corrección, no rendimiento) | 5.1 |
| Recursive con lineal → 1 | **Cambiar** a "barato": `Pipeline(Ridge)` (×4,9) y `LogisticRegression` (×4,7) quedan mal clasificados | 5.2, 11.3 |
| Recursive con no lineal → `cpu_count() - 1` | **Mantener** en caliente (×4-6 con refit); en frío, **añadir umbral** de trabajo | 5.3, 10 |
| Direct/DMV: `fit` paralelo, backtesting con 1 | **Mantener la regla del código, arreglar** `estimators_` (hoy, con refit, es más lento que no paralelizar: 77 frente a 64 s; con el arreglo, 11 s) y **corregir la docstring**, que describe la opción contraria | 5.4, 11.1 |
| RecursiveMultiSeries → `cpu_count() - 1` | **Mantener** para no lineales; **cambiar** lineales a 1 | 5.5, 11.3 |
| Stats, EquivalentDate → 1 | **Mantener** (Stats ganaría en caliente, pero no en un backtesting suelto) | 5.6 |
| Foundation → 1 | **Eliminar**: rama inalcanzable | 8.2 |
| LightGBM `n_jobs=1` → `cpu_count() - 1`; `n_jobs != 1` → 1 | **Mantener**: con 4 procesadores evita ×1,5; con 22 cuesta ×1,3 | 5.7 |
| Pipeline: decidir con el último paso | **Cambiar**: válido para detectar hilos, no para "barato" | 5.8 |
| `cpu_count() - 1` (frente a -1, físicos o tope) | **Mantener**: 21 ≈ 22 > 16 > 8; también con 4 y 8 procesadores | 7 |
| XGBoost, CatBoost, RF/ET con hilos propios | **Sin regla nueva**: `'auto'` queda a menos del 11 % del mejor | 6 |

**Hallazgos fuera de la heurística:**
1. Reajustar un `ForecasterDirect` ya entrenado con `n_jobs > 1` serializa los estimadores del ajuste anterior en cada tarea (×5-11 más lento) [probado].
2. `backtesting_stats` con Sarimax y `n_jobs > 1` falla con series de unas 2.000 filas, porque joblib pasa los arrays como memmaps de solo lectura [probado].
3. Las docstrings de `select_n_jobs_backtesting`, de las funciones públicas y el cuaderno de la documentación no coinciden con el código (secciones 8 y 9).

**Decisiones**: sección 10.4 (D1 a D9). **Cambios con diff**: sección 11.

**Estado a 8 de noviembre** (detalle en la sección 14):
- Fusionados en `0.27.x`: D1 (#1460), D3 (#1462), D9 (#1464) y D2 (#1468).
- Medido: D7. En Linux (WSL2) el arranque en frío de 21 workers cuesta la mitad que en Windows (2,9 s frente a 5,7 s), así que el punto de equilibrio baja de unos 5 s a unos 2,8 s de trabajo secuencial. Las reglas siguen acertando (sección 12).
- Implementado, PR pendiente: D4 y D5 en la rama `perf/n-jobs-auto-work-threshold`. Con `'auto'`, un backtesting suelto usa un solo job si el trabajo estimado desde el `fit` inicial no llega a 4 s en Windows o a 2 s en Linux; las búsquedas resuelven `'auto'` con las reglas de cada candidato, sin umbral (sección 11.4).
- Pendiente:
  - macOS: medir el arranque y fijar su umbral, que hoy usa el de Windows (sección 16);
  - D8, el cuaderno de la documentación.

## 2. Máquina, sistema y versiones

| | |
|---|---|
| CPU | Intel Core Ultra 9 185H (híbrida: 6 núcleos P con hyperthreading, 8 núcleos E, 2 núcleos LP-E) |
| Núcleos | 16 físicos, 22 lógicos. `joblib.cpu_count()` = 22, `joblib.cpu_count(only_physical_cores=True)` = 16 [probado] |
| Topología (procesador lógico) | P: 0-1, 10-19 (pares de hilos); E: 2-9; LP-E: 20-21 (`GetLogicalProcessorInformationEx`, clase de eficiencia 1 = P, 0 = E y LP-E) [probado] |
| RAM | 63,4 GB |
| Sistema | Windows 11 Enterprise 10.0.26200, plan de energía "Equilibrado", conectado a la corriente |
| Python y librerías | Python 3.13.18, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0 (loky), numba 0.68.0, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, statsmodels 0.15.0, psutil 7.2.2, threadpoolctl 3.7.0 |
| Entorno | conda `skforecast_27_py13`, skforecast importado desde el repositorio (`PYTHONPATH`, comprobado en cada script) |

Tres hechos de la máquina condicionan todo el estudio:

1. **`cpu_count() - 1` = 21, no 15.** La máquina tiene 16 núcleos físicos, pero joblib cuenta procesadores lógicos, así que `'auto'` lanza 21 workers [probado].
2. **Los núcleos no son iguales.** Un hilo de LightGBM (100 árboles, 1.713 × 288) tarda 1,83 s en un núcleo P, 2,96 s en uno E (×1,6) y 4,93 s en uno LP-E (×2,7). Un bucle de Python, 0,085 / 0,100 / 0,150 s [probado, `exp_core_speed.py`, mediana de 3 procesos por tipo]. Sin afinidad, un proceso de un hilo corre a velocidad de núcleo P (1,79 s): Windows no lo relega a los núcleos E. Con 21 workers, en cambio, parte del trabajo cae en núcleos E y LP-E, así que la aceleración ideal no es 21.
3. **Windows arranca los workers con `spawn`.** Cada worker es un intérprete nuevo que importa lo necesario para deserializar sus tareas (sección 4).

## 3. Método

- **Operaciones medidas**: solo las que skforecast paraleliza:
  - `backtesting_forecaster`, `backtesting_forecaster_multiseries` y `backtesting_stats`, paralelos por fold;
  - `fit` de `ForecasterDirect` y `ForecasterDirectMultiVariate`, paralelo por paso;
  - `grid_search_forecaster` y `grid_search_forecaster_multiseries`, un backtesting por candidato.
- **Valores de `n_jobs`**: 1, `'auto'`, `cpu_count() - 1` (21), -1 (22) y núcleos físicos (16), más 8 como tope candidato en las celdas grandes con refit. `'auto'` se mide siempre aparte y se registra el valor al que se resuelve. En el backtesting de Direct se combinan el `n_jobs` externo (folds) y el interno (`fit`).
- **Tamaños** (datos sintéticos con semilla: estacionalidad diaria y semanal, tendencia, ruido AR(1) y 2 exógenas):

  | Familia | S | L |
  |---|---|---|
  | Una serie (Recursive, Classifier, EquivalentDate, búsquedas) | 1.000 filas, lags 24, steps 24, 10 folds | 10.000 filas, lags 48, steps 168, 18 folds |
  | RecursiveMultiSeries | 5 series × 1.000, lags 24, steps 24, 10 folds | 50 series × 2.000, lags 48, steps 48, 20 folds |
  | Direct (`fit`) | 1.000 filas, 24 pasos | 5.000 filas, 48 pasos |
  | Direct (backtesting) | 10 folds × 24 pasos | 5 folds × 48 pasos |
  | DirectMultiVariate | 5 series, lags 12, 12 pasos | 10 series, lags 24, 24 pasos |
  | Stats | 360 filas, m = 12, 10 folds | 2.000 filas, m = 7, 10 folds |

- **Estimadores**:
  - Ridge, `Pipeline(StandardScaler, Ridge)`, LogisticRegression;
  - HistGradientBoosting (50 iteraciones; el clasificador, 100);
  - LightGBM con `n_jobs` = 1, None, -1 y 4 (100 árboles);
  - XGBoost con `n_jobs` = 1, None y -1 (100 árboles, profundidad 6);
  - RandomForest y ExtraTrees con `n_jobs` = None y -1 (25 árboles, profundidad 10);
  - CatBoost con `thread_count` = 1 y -1 (100 iteraciones);
  - LGBMClassifier(1) y HGBClassifier;
  - Arima y Sarimax de skforecast.
- **Repeticiones**:
  - 1 llamada de calentamiento y 5 cronometradas con `time.perf_counter()`, sin contar la construcción del forecaster.
  - Se reporta la mediana y el rango [mín-máx] de las 5.
  - El calentamiento se guarda aparte como "1.ª llamada": incluye arrancar los workers cuando cambia `n_jobs`.
  - El orden de las celdas se baraja con semilla, para que la deriva térmica no se alinee con una configuración.
- **Corrección**: en cada celda se guardan las predicciones y se comparan con las de `n_jobs=1` (sección 6, "Equivalencia de resultados").
- **Sobresuscripción** (bloque C): cada celda corre en su propio proceso, con un tiempo límite de 10 veces el secuencial (mínimo 30 s por llamada).
- **Afinidad**:
  - Se aplica con `psutil.Process().cpu_affinity()` antes de importar joblib.
  - 4 = 2 núcleos P × 2 hilos (portátil 2c/4t); 8 = 4 núcleos P × 2 hilos (4c/8t).
  - Puerta G1: `joblib.cpu_count()` devuelve 4 y 8 en el proceso principal y en los workers, que heredan la máscara [probado, `probe_workers.py`, en cada ejecución].
  - En el proceso principal se limitan OpenBLAS y OpenMP al número de procesadores con threadpoolctl, porque si no arrancan 22 hilos aunque la afinidad sea de 4 [probado]. No se usan variables de entorno porque loky las heredaría (sección 4).
- **Calibración**:
  - Un piloto de 56 min (1 llamada por celda con n_jobs 1 y `cpu-1`) fijó los tamaños y dio la estimación del barrido completo.
  - El piloto no se usa en las tablas.

## 4. Coste fijo de paralelizar: el arranque de los workers

joblib (motor loky) reparte las tareas entre procesos worker. En la **primera llamada en frío** hay que crearlos: en Windows cada uno es un intérprete nuevo (`spawn`) que importa numpy, pandas, scikit-learn, LightGBM y skforecast para deserializar su tarea. **En caliente**, loky reutiliza los workers de la llamada anterior y solo se paga enviar los datos.

Tiempo de la llamada a `Parallel`, mediana y [mín-máx] de 5 repeticiones (en frío, cada repetición en un proceso nuevo con las librerías ya importadas en el padre) [probado, `bench_spawn.py`]:

| Caso | k = 2 | 4 | 8 | 16 | 21 | 22 |
|---|---|---|---|---|---|---|
| En frío, workers que importan skforecast y LightGBM | 2,20 [2,13-2,53] | 2,65 [2,62-2,88] | 3,02 [2,95-3,12] | 4,19 [4,10-4,50] | **5,68** [5,28-5,72] | 5,96 [5,91-6,43] |
| En frío, tareas triviales | 0,42 | 0,45 | 0,50 | 0,63 | 0,66 | 0,70 |
| En frío, 4 tareas triviales con k workers | 0,42 | 0,45 | 0,51 | 0,64 | 0,68 | 0,67 |
| En caliente, forecaster | 0,03 | 0,05 | 0,10 | 0,13 | 0,16 | 0,17 |

Con afinidad, en frío y con forecaster: 4 procesadores: k = 2 → 2,28 s, 3 → 2,86 s, 4 → 3,27 s. 8 procesadores: k = 2 → 2,37 s, 4 → 2,92 s, 7 → 3,94 s, 8 → 4,28 s [probado].

Conclusiones:

1. **Arrancar en frío cuesta entre 2 y 6 s, y crece con k** (unos 0,18 s por worker) [probado]. Es el coste que una llamada suelta de `'auto'` tiene que amortizar.
2. **loky arranca los k workers aunque haya menos tareas**: 4 tareas con k = 21 cuestan lo mismo que 21 [probado].
3. **En caliente el coste es despreciable** (0,03-0,17 s) [probado].
4. **Se arranca en frío** en la primera llamada paralela de cada sesión, cuando cambia `n_jobs` y tras 300 s sin usar los workers (`idle_worker_timeout=300`, `joblib/_parallel_backends.py:637`) [verificado en código].
5. **El grueso del coste son las importaciones, no crear el proceso**: con tareas triviales cuesta de 0,4 a 0,7 s [probado].
6. **En Linux cuesta la mitad**: 2,94 s con 21 workers frente a 5,68 s [probado en WSL2, sección 12]. loky también lanza un intérprete nuevo (`fork_exec`, `loky/backend/popen_loky_posix.py:126-128`) [verificado en código], así que las importaciones se repiten, pero crear procesos y abrir archivos es más barato.

### 4.1 Backtestings sueltos en frío (V1)

Cada repetición se ejecuta en un proceso nuevo, con las librerías importadas y los datos construidos antes de cronometrar, como en un script de usuario. Mediana [mín-máx] de 5 procesos [probado, `bench_cold.py`]:

| Backtesting suelto, máquina entera | n_jobs = 1 | `'auto'` | `'auto'` / 1 | En caliente, `'auto'` |
|---|---|---|---|---|
| Recursive LGBM(1), S, sin refit | 0,18 | 3,50 [3,40-3,54] | **×19,8** | 0,25 |
| Recursive LGBM(1), S, con refit | 1,07 | 3,86 [3,58-4,04] | ×3,6 | 0,43 |
| Recursive LGBM(1), L, sin refit | 0,55 | 4,77 [4,53-4,91] | ×8,7 | 0,47 |
| MultiSeries LGBM(1), S, con refit | 1,56 | 3,90 [3,73-4,06] | ×2,5 | 0,57 |
| Recursive LGBM(1), L, con refit | 5,77 | 5,73 [5,37-6,15] | ×0,99 | 1,36 |
| Recursive HGB, L, con refit | 7,08 | 5,72 [5,69-6,77] | ×0,81 | 1,40 |
| Direct LGBM(1), S, con refit (con el arreglo) | 23,02 | 10,13 [10,00-10,53] | ×0,44 | 5,74 |
| MultiSeries LGBM(1), L, con refit | 29,60 | 10,53 [10,36-10,54] | ×0,36 | 6,43 |
| Recursive RF, L, con refit | 48,02 | 12,21 [11,92-13,62] | ×0,25 | 8,78 |

Conclusiones:

1. **En un backtesting suelto en Windows, `'auto'` empeora todo lo que en secuencial dura menos de unos 6 s**, hasta 20 veces en el tamaño de los ejemplos de la documentación, empata hacia los 6 s y solo gana por encima [probado].
2. **El sobrecoste real en frío es de 3,3 a 4,4 s**, el 72 % (mediana) del arranque medido en el bloque A. Los primeros workers empiezan a trabajar mientras arrancan los demás [probado]. Ese factor se usa para estimar el escenario en frío de las demás celdas (sección 10).
3. **El `fit` inicial, que el backtesting ya hace en el proceso principal, estima bien el trabajo con refit** [probado]. `t_fit × n_folds` da entre el 57 y el 100 % del tiempo secuencial real:
   - LGBM(1) L: 5,6 s frente a 5,8;
   - RF L: 38,5 frente a 48;
   - MultiSeries L: 23 frente a 29,6;
   - Direct S: 21,2 frente a 23;
   - HGB L, el peor: 4,0 frente a 7,1.

## 5. Reglas actuales, una a una

Tiempos en segundos, mediana de 5 llamadas en caliente con la máquina entera (22 procesadores), salvo que se indique otra cosa; `'auto'` entre paréntesis indica a qué valor se resuelve. Las tablas completas, con rango y primera llamada, están en `results/summary.md`. **Precisión**: dos celdas que en la práctica son la misma configuración (`fit` de Direct con LGBM(None), `'auto'` y `1`) dieron 5,91 y 7,95 s en momentos distintos, así que con estimadores multihilo las diferencias por debajo de un 30-35 % no se consideran significativas [probado].

### 5.1 `refit` entero distinto de 1 → 1

| Caso (L, refit = 3) | 1 | `'auto'` (1) | -1 (forzado a 1) |
|---|---|---|---|
| Recursive HGB | 3,98 | 3,97 | 4,10 |
| Recursive LGBM(1) | 2,42 | 2,21 | 1,99 |

Veredicto: **mantener**. Es una regla de corrección, no de rendimiento (sección 8.10), y el `n_jobs` explícito se fuerza a 1 con aviso [verificado en código]. Las diferencias de la tabla son ruido.

### 5.2 ForecasterRecursive (y Classifier) con estimador lineal → 1

| Caso | 1 | 21 | Mejor |
|---|---|---|---|
| Ridge, S/L, con y sin refit | 0,01-0,11 | 0,02-0,08 | Indistinto (décimas) |
| `Pipeline(StandardScaler, Ridge)`, L, con refit | 0,82 | 0,24 | -1: 0,17 (×4,9) |
| `Pipeline(StandardScaler, Ridge)`, L, sin refit | 0,67 | 0,20 | -1: 0,15 (×4,5) |
| `LogisticRegression` (Classifier), L, con refit | 4,61 | **0,98** | 21 (×4,7) |
| `LogisticRegression`, L, sin refit | 0,61 | 0,35 | 16: 0,31 |

- Con Ridge acierta: el trabajo es de décimas y en frío paralelizar costaría segundos.
- **Falla en dos casos.** `LogisticRegression` se ajusta por iteraciones, no por solución cerrada [probado]. Un `Pipeline` pierde el atajo de predicción con `np.dot` de `_build_predict_function` (`skforecast/utils/utils.py:4194`, que solo se aplica si el estimador es un `LinearModel` de scikit-learn) [verificado en código]. En la predicción recursiva (168 pasos × 18 folds), un `Pipeline.predict` por paso cuesta más que el ajuste.

Veredicto: **cambiar** el criterio "lineal" por "barato": regresores lineales de scikit-learn (`LinearModel`) que no estén dentro de un `Pipeline`. O dejar que lo decida el umbral de trabajo de la sección 10 (decisión D4).

### 5.3 ForecasterRecursive con estimador no lineal → `cpu_count() - 1`

| Caso | 1 | `'auto'` (21) | Mejor |
|---|---|---|---|
| LGBM(1), L, con refit | 5,62 | 1,36 | 21: 1,33 (×4,2) |
| XGB(1), L, con refit | 10,36 | 2,12 | 21: 2,11 (×4,9) |
| RF(None), L, con refit | 45,97 | **8,78** | auto (×5,2) |
| ET(None), L, con refit | 11,58 | 2,41 | 21: 2,07 (×5,6) |
| CatBoost(1), L, con refit | 35,27 | 5,93 | -1: 5,71 (×6,2) |
| HGB, L, con refit | 7,46 | 1,40 | 21: 1,38 (×5,4) |
| LGBMClassifier(1), L, con refit | 7,80 | 1,82 | -1: 1,72 (×4,5) |
| HGBClassifier, S, con refit | 9,97 | 2,08 | auto (×4,8) |
| LGBM(1), S, sin refit | **0,13** | 0,25 | 1 |
| XGB(1), S, sin refit | **0,36** | 0,40 | 1 |
| RF(None), S, sin refit | **0,18** | 0,28 | 1 |
| HGB, L, sin refit | 2,94 | **0,52** | auto (×5,6) |
| CatBoost(1), L, sin refit | 4,07 | 2,40 | 16: 2,25 |

Veredicto:
- **En caliente, mantener**: `'auto'` queda a menos del 17 % del mejor valor en todas las celdas L y en las S con refit [probado].
- En S sin refit pierde, pero son décimas.
- **En frío** (una llamada suelta) esta regla es la que más daño hace: suma 5,7 s a trabajos que en secuencial cuestan entre 0,1 y 6 s (secciones 4 y 10).

### 5.4 ForecasterDirect y DirectMultiVariate

**`fit`** (paralelo por pasos; regla: lineal → 1, LGBM con `n_jobs != 1` → 1, resto → 21):

| Caso | 1 | `'auto'` | 21 | Mejor |
|---|---|---|---|---|
| Direct Ridge, S / L | **0,03 / 0,17** | 1 | 0,11 / 0,78 | 1 |
| Direct LGBM(1), S (24 pasos) / L (48) | 2,69 / 13,54 | 21 | 0,72 / 3,12 | 21 (×4,3) |
| Direct HGB, S / L | 5,50 / 13,33 | 21 | 0,63 / 2,36 | 21 (×5,6-8,7) |
| Direct LGBM(None), S / L | 3,07 / 7,95 | 1: 1,89 / 5,91 | 3,10 / 7,60 | 1 |
| Direct RF(None), S | 4,86 | 21: 0,65 | 0,63 | 21 (×7,7) |
| DMV Ridge, S / L | 0,02 / 0,28 | 1 | 0,04 / 0,27 | Indistinto |
| DMV LGBM(1), S / L | 3,82 / 33,87 | 21: n/m / 6,55 | 1,18 / 6,65 | 21 (×5,2) |

Veredicto del `fit`: **mantener** las tres reglas [probado].

Un detalle: HGB en secuencial es muy lento con modelos pequeños (5,5 s frente a 0,6 s) porque cada `fit` usa OpenMP con 22 hilos en el proceso principal; en un worker, loky lo limita a 1 hilo [probado, `probe_workers.py`].

**Backtesting con `refit=True`** (formato externo|interno; la regla actual da 1|21 para no lineales):

| Caso | 1\|1 | Actual 1\|21 | **Actual + arreglo** | 21\|1 | 21\|21 |
|---|---|---|---|---|---|
| HGB, L (48 pasos, 5 folds) | 63,90 | 77,29 | **10,92** | 29,07 | 18,79 |
| LGBM(1), L | 54,72 | 70,67 | 13,17 | 31,86 | **10,45** |
| HGB, S (24 pasos, 10 folds) | 42,43 | 43,06 | **4,29** | 10,44 | 7,92 |
| LGBM(1), S | 21,77 | 34,14 | 5,74 | 8,06 | **5,45** |
| DMV LGBM(1), S | 35,34 | 17,82 | **7,94** | 12,38 | n/m |
| Ridge, S / L | 0,22 / 0,73 | 0,24 / 0,90 | 0,22 / 0,68 | **0,11 / 0,55** | 0,19 / 1,03 |

**Backtesting sin refit** (solo el ajuste inicial, en el proceso principal):

| Caso | 1\|1 | Actual 1\|21 | 21\|1 | 21\|21 |
|---|---|---|---|---|
| HGB, L | 12,04 | **2,83** | 18,41 | 3,47 |
| LGBM(1), L | 11,74 | **2,69** | 15,66 | 4,13 |
| HGB, S | 5,46 | **0,67** | 5,55 | 2,29 |

Veredicto: **mantener la regla del código** (1 externo, el `fit` decide), **aplicar el arreglo de `estimators_`** (sección 11.1) y **corregir la docstring**, que describe la alternativa 21|1.
- Con el código actual, la regla de `'auto'` con refit es **más lenta que no paralelizar** en L (77,3 s frente a 63,9 s con HGB) [probado].
- Con el arreglo es la mejor o queda a menos del 26 % del anidado [probado].
- Sin refit, paralelizar por folds es entre 5 y 6 veces peor que la regla actual, porque cada worker recibe un forecaster con 48 modelos para hacer predicciones baratas [probado].

**El problema de `estimators_`** [probado, `exp_direct_refit.py`, 4 reajustes por variante, 21 workers]:
- `fit` envía `forecaster=self` a cada tarea de `Parallel` (`_forecaster_direct.py:1907-1920`) y no reinicia `estimators_` [verificado en código].
- Al reajustar un forecaster ya ajustado, cada tarea serializa los estimadores del ajuste anterior: 13,7 MB con LGBM y 24,9 MB con HGB, una vez por paso.

| Reajuste de Direct L (48 pasos) | Código actual | `estimators_` vaciado | n_jobs = 1 |
|---|---|---|---|
| LGBM(1) | 18,4-19,4 s | 2,1 s | 11,6 s |
| HGB (100 iteraciones) | 32,5-36,3 s | 3,0 s | 15-18 s |
| DMV HGB (20 series, 480 columnas) | 29,2 s | 20,9 s | 76 s |
| DMV LGBM(1) (20 series) | 13,5 s | 13,1 s | 67 s |

El efecto es grande cuando cada `fit` es rápido en relación con el tamaño serializado del modelo, y casi nulo cuando cada `fit` es caro (DMV LGBM con 480 columnas).

### 5.5 ForecasterRecursiveMultiSeries → `cpu_count() - 1` (también con lineales)

| Caso | 1 | 21 | `'auto'` hoy |
|---|---|---|---|
| Ridge, L, sin refit | **0,61** | 0,82 | 21 |
| Ridge, L, con refit | 1,68 | **1,32** | 21 |
| Ridge, S | 0,07-0,11 | 0,08-0,09 | 21 |
| LGBM(1), L, con refit | 28,68 | **6,43** (×4,5) | 21 |
| LGBM(1), S, con refit | 1,62 | **0,57** | 21 |
| LGBM(1), L, sin refit | 1,95 | 1,83 | 21 |
| LGBM(None), L, con refit | 12,89 | **7,38** | 1 |
| HGB, L, con refit / sin refit | 11,94 / 1,91 | **5,51 / 1,34** | 21 |

Veredicto:
- **No lineales: mantener en caliente**; en frío, ver la sección 10.
- **Lineales: cambiar a 1**, como en Recursive. Ganan como mucho 0,4 s en caliente, pierden sin refit, y en frío sumarían 5,7 s a un trabajo de entre 0,1 y 1,7 s [probado].

### 5.6 ForecasterStats, ForecasterEquivalentDate y ForecasterFoundation → 1

| Caso (refit=True, 10 folds) | 1 | 21 |
|---|---|---|
| Arima, S / L | 1,00 / 2,84 | 0,44 / 1,23 |
| Sarimax, S / L | 3,79 / 12,89 | 1,49 / **falla** (sección 8.11) |
| EquivalentDate, S / L, con y sin refit | 0,01-0,02 | 0,01-0,03 |

Veredicto:
- **Mantener** las tres.
- EquivalentDate: el trabajo es de centésimas.
- Stats: en caliente 21 workers ganan ×2,3-2,5, pero en un backtesting suelto en frío pierde (Arima L: 1,2 + 5,7 ≈ 6,9 s frente a 2,8 s) [inferido, combinando secciones 4 y 5]. El umbral de trabajo de la sección 10 lo resolvería sin una regla propia.
- La rama de Foundation es inalcanzable (sección 8.2).

### 5.7 LightGBM: `n_jobs=1` → `cpu_count() - 1`; `n_jobs != 1` → 1

Recursive L con refit, `n_jobs` externo (máquina entera):

| Estimador | 1 | `'auto'` | 4 | 8 | 16 | 21 | 22 |
|---|---|---|---|---|---|---|---|
| LGBM `n_jobs=None` (16 hilos) | 3,67 | 3,64 (1) | 3,08 | **2,76** | 2,87 | 3,20 | 3,08 |
| LGBM `n_jobs=-1` (22 hilos) | 5,11 | 5,34 (1) | 3,73 | **3,39** | 3,67 | 3,67 | 3,77 |
| LGBM `n_jobs=4` | 3,96 | 3,86 (1) | 1,82 | **1,62** | 1,89 | 1,67 | 1,75 |
| LGBM `n_jobs=1` (bloque B) | 5,62 | 1,36 (21) | | 1,87 | 1,49 | **1,33** | 1,37 |

Con afinidad 4 (portátil 2c/4t), LGBM `n_jobs=None` (4 hilos): **1 → 3,73**; 2 → 5,42; 3 → 5,67; 4 → 5,91 [probado].

Con afinidad 8 (portátil 4c/8t) [probado]:

| Estimador | 1 | 4 (físicos) | 7 (`cpu_count() - 1`) | 8 |
|---|---|---|---|---|
| LGBM `n_jobs=None` (8 hilos) | 3,62 | 3,86 | 3,61 | 3,20 |
| XGB `n_jobs=None` | 10,02 | 5,73 | **3,98** | 5,36 |
| RF `n_jobs=-1` | 20,42 | 13,34 | 16,50 | **13,23** |
| HGB | 6,88 | 3,08 | 2,19 | **2,17** |

Con 8 procesadores, paralelizar por fuera con LightGBM y sus propios hilos ni gana ni pierde: es el punto de cruce entre 4 procesadores (pierde ×1,5) y 22 (gana ×1,1-1,3).

Veredicto: **mantener**.
- Con 4 procesadores la regla evita perder ×1,5-1,6 por sobresuscripción.
- Con 22 deja sin aprovechar ×1,3-1,6, porque allí la sobresuscripción no penaliza (336 hilos sobre 22 procesadores) [probado].
- En Linux es mucho peor [probado en WSL2, sección 12.4]. Un backtesting con `LGBMRegressor()` (11 hilos), `refit=True` y 21 workers explícitos pasa de 600 s. Incluso sin paralelizar por fuera, `LGBMRegressor(n_jobs=-1)` (22 hilos) tarda varios cientos de veces más que el valor por defecto en un backtesting sin refit, casi todo en el `fit`. La regla protege más en Linux que en Windows.
- La mejor combinación medida es un estimador de un hilo con 21 workers: LGBM(1) con 21 workers tarda 1,33 s, frente a 3,64 s de LightGBM por defecto con `'auto'`. `'auto'` no puede cambiar el estimador del usuario, pero la documentación puede recomendarlo para backtestings y búsquedas con refit.
- En el `fit` de Direct la regla también acierta: LGBM(None) tarda 5,91 s con 1 y 8,41 s con -1.

### 5.8 `Pipeline`: decidir con el último paso

[verificado en código] Las dos funciones desenvuelven el `Pipeline` y deciden con el último paso.
- Para detectar hilos propios (LightGBM) es lo correcto.
- Para "lineal", no (sección 5.2): el `Pipeline(StandardScaler, Ridge)` se trata como lineal y se le asigna 1, pero predice sin el atajo de `np.dot` y paralelizarlo es entre 4,5 y 4,9 veces más rápido [probado].

Veredicto: **cambiar**. Desenvolver para detectar hilos, pero no considerar "barato" un `Pipeline`.

### 5.9 Búsquedas (heredan la regla del backtesting de cada candidato)

| `grid_search` (L, 4 candidatos) | 1 | 21 (= `'auto'`) | 1.ª búsqueda, 21 |
|---|---|---|---|
| Recursive HGB, sin refit | 16,21 | **3,42** (×4,7) | 3,50 |
| Recursive HGB, con refit | 38,97 | **6,67** (×5,8) | 15,28 |
| Recursive LGBM(1), sin refit | 2,06 | 1,38 | 1,52 |
| Recursive LGBM(1), con refit | 14,69 | **3,46** (×4,2) | 3,31 |
| MultiSeries LGBM(1), sin refit | 6,08 | 5,96 | 6,43 |

Veredicto: **mantener**. Es donde `'auto'` funciona mejor: la búsqueda reutiliza los workers entre candidatos y paga el arranque una vez [probado]. Si se adopta un umbral de trabajo (sección 10), en una búsqueda hay que aplicarlo al trabajo total (candidatos × folds), no al de cada candidato.

## 6. Casos que la regla no contemplaba

| Caso | Qué hace `'auto'` hoy | Medido | Veredicto |
|---|---|---|---|
| XGBoost con `n_jobs` = None / -1 | 21 (no lineal) | 2,87 / 2,38 s, frente a 7,33 / 10,10 con 1; mejor 21-22 | Sin regla nueva: XGBoost respeta el límite de OpenMP que loky pone en cada worker [probado] |
| CatBoost `thread_count=-1` | 21 | 4,99 s frente a 10,04 con 1; mejor 21 | Sin regla nueva |
| RF / ET con `n_jobs=-1` | 21 | 6,81 / 1,59 s frente a 9,79 / 2,73 con 1 | Sin regla nueva; el joblib anidado del estimador no causa problemas [probado] |
| RF / ET con `n_jobs=None` | 21 | Sección 5.3 | Correcto |
| `ForecasterRecursiveClassifier` | Igual que Recursive | LGBMClassifier y HGBClassifier ganan ×4,5-4,8; LogisticRegression, sección 5.2 | Mantener, salvo el cambio de 5.2 |
| DirectMultiVariate con lineal | 1 (en `fit` y en backtesting) | `fit` con Ridge: 0,02-0,28 s con cualquier valor; backtesting S: 0,11-0,33 s | Mantener 1 |
| `ForecasterRnn` | 1 (rama por defecto) | No medido | Mantener; documentar |
| Lineales que no son `LinearModel` (`SGDRegressor`, `PoissonRegressor`, `TweedieRegressor`, `TransformedTargetRegressor(Ridge)`) | 21 | No medido (sección 8.6) | Con el umbral de trabajo de la sección 10 no hace falta enumerarlos |

**Equivalencia de resultados** [probado, `summary_equivalence.csv`]:
- 533 de 535 celdas dan predicciones idénticas a las de `n_jobs=1`.
- Las dos excepciones son modelos con optimización iterativa, `LogisticRegression` (probabilidades hasta 0,09 de diferencia, 1 clase distinta en 3.024 predicciones) y Sarimax S (hasta 0,006). Se reprodujo fuera de skforecast: con 1 o con 8 hilos de BLAS, la misma regresión logística termina en otra iteración (184 frente a 152) y da probabilidades distintas. El proceso principal usa 22 hilos de BLAS y cada worker 1.
- No es un fallo de skforecast: cualquier solver iterativo es sensible a ese redondeo.

## 7. ¿`cpu_count() - 1`, -1, núcleos físicos o un tope?

Mejor valor por celda, con refit (L), mediana de 5 en caliente [probado]:

| Máquina | Caso | 1 | Físicos | `cpu_count() - 1` | -1 | Tope 8 |
|---|---|---|---|---|---|---|
| 22 lógicos / 16 físicos | LGBM(1) | 5,62 | 1,49 | **1,33** | 1,37 | 1,87 |
| | CatBoost(1) | 35,27 | 8,26 | 5,84 | **5,71** | 10,26 |
| | RF(None) | 45,97 | 11,78 | **9,34** | 9,69 | 14,45 |
| | XGB(1) | 10,36 | 2,61 | **2,11** | 2,35 | 3,57 |
| | HGB | 7,46 | 1,84 | **1,38** | 1,40 | 1,99 |
| 4 lógicos / 2 físicos (2c/4t) | LGBM(1) | 5,60 | 3,52 | 2,97 | **2,93** | |
| | HGB | 5,55 | 4,05 | 3,08 | **2,88** | |
| | MultiSeries LGBM(1) | 27,34 | 17,74 | 14,98 | **13,69** | |
| | `fit` Direct LGBM(1) | 11,33 | 8,62 | 7,58 | **7,15** | |
| 8 lógicos / 4 físicos (4c/8t) | LGBM(1) | 5,87 | 2,48 | **2,09** | **2,09** | |
| | HGB | 7,29 | 3,13 | **2,26** | 2,92 | |

Veredicto: **mantener `cpu_count() - 1`**.
- **Frente a -1**: la diferencia está dentro del ruido en todas las máquinas, y dejar un procesador libre no penaliza.
- **Frente a núcleos físicos**: los procesadores lógicos (hyperthreading y núcleos E) aportan entre un 10 y un 30 % más que limitarse a los físicos.
- **Frente a un tope de 8**: con 22 procesadores, 8 workers son entre 1,3 y 1,8 veces más lentos.

Esta CPU híbrida no cambia la conclusión: un hilo en un núcleo E es 1,6 veces más lento y en uno LP-E 2,7 veces (sección 2), pero los núcleos E siguen sumando.

## 8. Discrepancias entre código, docstrings y casos no cubiertos

1. **Direct y DirectMultiVariate con `refit=True`** [verificado en código]:
   - La docstring de `select_n_jobs_backtesting` (`skforecast/model_selection/_utils.py:825-828`) dice que usan `n_jobs = cpu_count() - 1`.
   - El código (`:874-876`) devuelve 1 con cualquier `refit` y deja la paralelización al `fit` del forecaster.
   - Los tests (`test_select_n_jobs_backtesting.py:38-45`) fijan 1, y el cuaderno copia la versión de la docstring.
   - La sección 5 mide qué alternativa es mejor.
2. **`ForecasterFoundation`** [verificado en código]:
   - La rama de `select_n_jobs_backtesting` (`:854`) es inalcanzable: `backtesting_forecaster` lo rechaza (`_validation.py:907`), `backtesting_forecaster_multiseries` tampoco lo admite y `backtesting_foundation` no tiene `n_jobs`.
   - La docstring no la menciona.
3. **`ForecasterRnn`** [verificado en código]: `backtesting_forecaster_multiseries` lo admite y cae en la rama por defecto (`n_jobs = 1`), que es lo razonable con Keras, pero la docstring no lo dice.
4. **`ForecasterRecursiveClassifier` y `LGBMClassifier`** [verificado en código]: tienen reglas en el código (`:867-871`) pero no en la docstring.
5. **Lineal en Recursive frente a MultiSeries** [verificado en código]: un estimador lineal da 1 en Recursive pero `cpu_count() - 1` en RecursiveMultiSeries, y la docstring no explica la diferencia (sección 5).
6. **Qué cuenta como lineal** [probado]:
   - Lineales (`LinearModel`): Ridge, Lasso, ElasticNet, LinearRegression, BayesianRidge, HuberRegressor, QuantileRegressor y LinearSVR.
   - Tratados como no lineales (21 workers): `SGDRegressor`, `PoissonRegressor`, `TweedieRegressor` y cualquier envoltorio, como `TransformedTargetRegressor(Ridge())`.
   - El `Pipeline` sí se desenvuelve hasta el último paso.
7. **`n_jobs` de Direct se fija en `__init__`** [verificado en código]:
   - `ForecasterDirect.__init__` (`_forecaster_direct.py:633-643`) guarda un entero, y `set_params` no lo recalcula (`:3165`).
   - Consecuencias: un forecaster guardado en una máquina de 22 procesadores lleva `n_jobs = 21` al cargarlo en un portátil de 4; y una búsqueda que cambia el `n_jobs` de un LightGBM no cambia el del forecaster.
8. **Docstrings de las funciones públicas** [verificado en código]: el parámetro `n_jobs` remite a `skforecast.utils.select_n_jobs_backtesting`, pero la función vive en `skforecast.model_selection._utils` (solo `select_n_jobs_fit_forecaster` está en `skforecast.utils`).
9. **`n_jobs=-1` en skforecast es `cpu_count()`** [verificado en código]: equivale a todos los procesadores lógicos (22), igual que en joblib.
10. **`refit` entero distinto de 1** [verificado en código]:
    - Fuerza `n_jobs = 1` por corrección, no por rendimiento: el forecaster ajustado en un fold se reutiliza en los siguientes, y ese estado no viaja entre workers.
    - Si se pasa un `n_jobs` explícito, se avisa con `IgnoredArgumentWarning` y se usa 1.
11. **Fallo: `backtesting_stats` con Sarimax y `n_jobs > 1` falla con series de cierto tamaño** [probado, `exp_sarimax_parallel.py`]:
    - Con 2.000 filas (bloque F, tamaño L) y 4 o 21 workers, la primera tarea falla con `BrokenProcessPool: A task has failed to un-serialize`. La traza del worker termina en `statsmodels/tsa/statespace/_kalman_filter.pyx`, `dKalmanFilter.__setstate__` → `ValueError: buffer source array is read-only`.
    - Con 360 filas funciona.
    - Causa: joblib pasa a los workers los arrays de más de 1 MB como memmaps de solo lectura (`max_nbytes='1M'`, `mmap_mode='r'` por defecto, `joblib/parallel.py:100-101`) [verificado en código], y el filtro de Kalman de statsmodels no se puede reconstruir desde memoria de solo lectura [probado].
    - Con `joblib.parallel_config(max_nbytes=None)` funciona: 5,5 s con 4 workers en caliente frente a 11,5 s secuencial, con predicciones idénticas [probado].
    - Con `'auto'` no se ve, porque Stats usa 1; solo afecta a quien pasa `n_jobs` explícito.
    - También falla en Linux (WSL2, sección 12.5).
    - Arreglado en #1468 con `mmap_mode='c'`, no con `max_nbytes=None` (sección 11.2).

## 9. El cuaderno `docs/faq/parallelization-skforecast.ipynb`

No se ha ejecutado ni modificado. Las salidas guardadas son de skforecast 0.26.0, Python 3.14.3 y macOS arm con 10 núcleos sin hyperthreading [verificado en el cuaderno].

### 9.1 Carencias como método [verificado en el cuaderno]

1. **Una sola medida por caso**, con `time.time()`, sin calentamiento ni repeticiones (16 llamadas a `time.time()` por celda). La primera llamada paralela incluye arrancar los workers, que en esta máquina cuesta de 0,4 a 5,5 s (sección 4), y nada separa ese coste de la señal.
2. **Solo compara `n_jobs=-1` frente a 1.** Nunca mide `'auto'` ni `cpu_count() - 1`, así que no valida las reglas que describe.
3. **Cronometra operaciones que no se paralelizan**: `fit` y `create_train_X_y` de `ForecasterRecursive` y de `ForecasterRecursiveMultiSeries`, y `create_train_X_y` de los Direct. Ocupan la mitad de las barras del gráfico.
4. **El benchmark de Direct no controla el `n_jobs` del forecaster.** Lo crea con el valor por defecto (`'auto'`):
   - el caso "no parallel" (`n_jobs=1` en el backtesting) paraleliza igualmente el `fit` de cada fold;
   - el caso "parallel" (`n_jobs=-1`) anida loky dentro de loky.
   
   Ninguno de los dos es lo que dice su etiqueta.
5. **Pocos estimadores**: Ridge, LightGBM con `n_jobs` = 1 y -1, e HGB. No hay XGBoost, RandomForest, ExtraTrees ni CatBoost, que son justo los que tienen hilos propios, ni Classifier, Stats o EquivalentDate.
6. **Un solo tamaño de datos aleatorios** (ruido uniforme, 5.000 filas, 10 exógenas, 10 series), sin caso pequeño de documentación. Las búsquedas repiten candidatos idénticos (`{'alpha': [0.1, 0.1, 0.1]}`).
7. **Una sola máquina**, un Mac de 10 núcleos homogéneos. Allí `cpu_count()` = núcleos físicos y el arranque de workers es más barato que en Windows, así que sus conclusiones no se trasladan a una CPU híbrida con hyperthreading (secciones 2 y 7).

### 9.2 Texto de reglas frente a las docstrings actuales [verificado en código]

| Punto | Cuaderno | Código / docstring |
|---|---|---|
| Direct y DMV con `refit=True` | `cpu_count() - 1` | El código devuelve 1; la docstring dice `cpu_count() - 1` (la docstring está mal, sección 8.1) |
| `ForecasterRecursiveClassifier` | No aparece | Mismas reglas que Recursive, incluida la de `LGBMClassifier` (solo en el código) |
| `Pipeline` | No aparece | Se usa el último paso para decidir (código de las dos funciones) |
| `ForecasterFoundation` | No aparece | Rama en el código, inalcanzable (sección 8.2) |
| `ForecasterRnn` | No aparece | Cae en la rama por defecto, 1 (sección 8.3) |
| "Forecaster Fitting: otherwise `cpu_count() - 1`" | Sugiere que cualquier `fit` no lineal se paraleliza | Solo se paraleliza el `fit` de Direct y DMV; el resto ignora `n_jobs` |
| Regla de LightGBM | Solo `LGBMRegressor` | En backtesting también `LGBMClassifier` |

### 9.3 Propuesta de reescritura (PR posterior, siguiendo `.claude/rules/docs.md`)

**Texto de reglas.** No volver a copiar las reglas: que remita a las docstrings de `select_n_jobs_backtesting` y `select_n_jobs_fit_forecaster`, enlazadas en la referencia de la API, y que explique solo lo que no cabe en una docstring:

- Qué se paraleliza: los folds del backtesting y de las búsquedas, y los pasos del `fit` de Direct y DMV. El resto no usa `n_jobs`.
- Que `'auto'` no ve los hilos internos de los estimadores, salvo LightGBM, y cómo decidir a mano.
- El coste fijo de arrancar procesos, mayor en Windows (sección 4).
- Que joblib respeta la afinidad del proceso y `LOKY_MAX_CPU_COUNT`, útil para limitar núcleos en máquinas compartidas.
- Que en CPUs con hyperthreading o núcleos híbridos `cpu_count()` cuenta procesadores lógicos.

**Benchmark ligero**, de menos de 5 minutos, para que pueda ejecutarse al construir la documentación:

- Una función `time_call(fn, n_reps=5)`, con calentamiento, que devuelva la mediana y el rango.
- Compara `n_jobs` ∈ {1, `'auto'`, `cpu_count() - 1`} e incluye en la tabla el valor al que se resuelve `'auto'`.
- Solo operaciones que se paralelizan: `backtesting_forecaster` con refit `True` y `False`, `backtesting_forecaster_multiseries` y `ForecasterDirect.fit`. En Direct, fija el `n_jobs` del forecaster de forma explícita.
- Tres estimadores: Ridge, `LGBMRegressor(n_jobs=1)` y HGB, más un caso de LightGBM con hilos propios para ilustrar la regla.
- Un dataset de la propia documentación más uno sintético grande.
- Junto a la tabla, el sistema operativo, la CPU (física y lógica) y las versiones de Python, skforecast, scikit-learn, joblib y lightgbm. Un texto que diga que los números son de esa máquina.

Dos advertencias que este estudio aconseja incluir en el benchmark del cuaderno:

1. **Separar la primera llamada de las siguientes.** En Windows, la primera llamada paralela de una sesión cuesta entre 3 y 6 s más (secciones 4 y 4.1). Un benchmark sin calentamiento mide sobre todo eso.
2. **Medir en la misma ejecución el caso del usuario típico, un backtesting suelto, y el de una búsqueda.** Las conclusiones son opuestas en las cargas pequeñas.

Si se adopta la estrategia de la sección 10, el benchmark debería mostrar también que `'auto'` elige 1 en las cargas pequeñas.

## 10. Estrategia para elegir `n_jobs` automáticamente

### 10.1 Qué decide de verdad si paralelizar compensa

Con todos los bloques juntos, el tipo de forecaster y de estimador explican menos que una sola magnitud: **el trabajo secuencial frente al coste fijo de arrancar los workers**.
- Con refit y trabajo grande, paralelizar gana de 4 a 6 veces con cualquier forecaster y estimador (secciones 5.3-5.5 y 5.9).
- Sin refit o con trabajo pequeño, apenas gana en caliente y pierde en frío (sección 4.1).
- El punto de equilibrio en frío en Windows está en unos 6 s de trabajo secuencial, tanto con 22 procesadores como con 8 o con 4 [probado con 22, sección 4.1; inferido con 4 y 8, combinando los tiempos en caliente con el arranque medido, por ejemplo LGBM(1) L con refit: 2,1 + 3,9 ≈ 6,0 s con 8 y 3,0 + 2,9 ≈ 5,9 s con 4, frente a 5,9 y 5,6 s secuenciales].

El umbral es parecido en las tres máquinas porque arranque y ganancia escalan juntos: más workers cuestan más de arrancar, pero dividen más el trabajo.

### 10.2 Estrategias comparadas (V2, `strategies.py`)

- **E0**: la regla actual.
- **E1**: la regla actual con tres cambios:
  - el arreglo de `estimators_` en Direct y DMV;
  - "barato" en lugar de "lineal": solo `LinearModel` sin `Pipeline`;
  - RecursiveMultiSeries con lineal → 1.
- **E2**: E1 con un tope en el número de tareas (`min(cpu_count() - 1, n_folds)`; en Direct, pasos).
- **E3(T)**: E2, pero 1 si el trabajo secuencial estimado es menor que T segundos.

Se evalúan sobre las 100 celdas medidas con la máquina entera (bloques B, D, E, F y G):
- **En caliente**: las medianas medidas.
- **En frío**: la mediana en caliente más el 72 % del arranque de k workers medido en el bloque A (factor calibrado con V1).

"Pérdida" es la suma, sobre las celdas, de (tiempo de la estrategia − mejor tiempo medido con el código actual). Una pérdida negativa significa mejor que la mejor opción del código actual, gracias al arreglo de DMV.

| Estrategia | En caliente: pérdida L / S (s) | En caliente: peor frente a n_jobs=1 | En frío: pérdida L / S (s) | En frío: peor frente a n_jobs=1 |
|---|---|---|---|---|
| E0 (actual) | 133,4 / 74,1 | ×1,34 / ×2,02 | 211,0 / 173,5 | ×8,3 / **×63,8** |
| E1 | **6,9 / −1,9** | ×1,07 / ×1,93 | 92,7 / 105,8 | ×8,4 / ×83,3 |
| E2 | 6,9 / −1,9 | ×1,07 / ×1,93 | 80,2 / 61,8 | ×7,2 / ×48,9 |
| E3(5 s) | 18,6 / 19,0 | ×1,0 / ×1,0 | **42,1 / 15,1** | ×1,66 / **×1,0** |
| E3(10 s) | 35,1 / 42,9 | ×1,0 / ×1,0 | 44,1 / 26,1 | ×1,0 / ×1,0 |

[probado en caliente; inferido y calibrado en frío]

Lectura:

1. **E1 corrige casi todo en caliente.** La pérdida baja de 133 s a 7 s en L; el grueso venía de Direct con refit.
2. **En frío, solo un umbral de trabajo evita que `'auto'` sea peor que no paralelizar.** El tope por tareas (E2) ayuda poco: los workers que sobran ya estaban de más.
3. **E3(5 s)** es la mejor en frío (pérdida de 42 y 15 s, nunca más lenta que no paralelizar en S, ×1,66 en el peor caso L). En caliente cuesta 12-21 s frente a E1, porque renuncia a ganancias en cargas de menos de 5 s (por ejemplo LogisticRegression L: 4,6 s frente a 1,0).
4. **E3(10 s)** nunca es más lenta que no paralelizar, pero en caliente deja escapar más (35-43 s).
5. **Límites de la evaluación de E3** [inferido]:
   - Usa como trabajo estimado el tiempo secuencial medido, una estimación perfecta.
   - La implementación real lo extrapola del primer `fit`, que da entre el 57 y el 100 % del real (sección 4.1). Es decir, se comporta como un umbral efectivo algo mayor que el nominal: un umbral nominal de 5 s equivale a uno real de entre 5 y 9 s.
   - El peor caso de E3(5 s) en frío (×1,66) es la búsqueda MultiSeries sin refit, que no gana nada con paralelizar aunque su trabajo supere el umbral.

### 10.3 Estrategia recomendada

1. **Reglas por tipo (E1), siempre**, para decidir cuántos workers usar si se paraleliza:
   - Recursive, Classifier y MultiSeries no baratos → `cpu_count() - 1`;
   - Direct y DMV → paralelizar el `fit` (con el arreglo);
   - lineales sin `Pipeline`, LightGBM con hilos propios, Stats, EquivalentDate, Rnn y refit entero → 1.
2. **Umbral de trabajo para backtestings sueltos.**
   - Cronometrar el `fit` inicial que el backtesting ya hace en el proceso principal.
   - Estimar el trabajo como `t_fit × número de ajustes`.
   - Si queda por debajo de unos 5 s, usar 1. Si no, `min(n_jobs, n_folds)`.
   - Sin refit no hay ajustes que repartir y se usa 1: es el caso que más pierde en frío (×8,7-19,8) y en caliente gana como mucho 2,4 s (HGB L).
3. **Búsquedas: decidir una vez con las reglas (E1), sin umbral.** El arranque se paga una vez y se reparte entre todos los candidatos (sección 5.9). Hoy las búsquedas pasan `'auto'` a cada backtesting, así que, si se adopta el punto 2, deben resolver `'auto'` antes y pasar un entero.
4. **`cpu_count() - 1` como número de workers**: sin cambios (sección 7).

### 10.4 Decisiones que te corresponden

| # | Decisión | Opciones | Recomendación | Estado (8 de noviembre) |
|---|---|---|---|---|
| D1 | Arreglo de `estimators_` en Direct y DMV | (a) PR propio ya; (b) con el resto | **(a)**: independiente de la heurística, el riesgo es bajo y el impacto, de ×5-11 | (a), fusionado en #1460 |
| D2 | Fallo de Sarimax en paralelo | (a) `max_nbytes=None` en `_backtesting_stats`; (b) en todos los `Parallel` de skforecast; (c) solo documentarlo | **(a)** en un PR de fix. (b) evitaría casos parecidos con otros modelos, pero a cambio de enviar copias completas de los arrays a cada worker; no se ha medido su coste | Fusionado en #1468 con otra solución: `mmap_mode='c'`, solo con Sarimax (sección 11.2) |
| D3 | "Lineal" → "barato" y MultiSeries con lineal → 1 (E1) | (a) aplicar; (b) dejarlo y que lo resuelva el umbral (D4) | **(a)**: cambio pequeño, mejora en caliente y en frío | (a) sin el cambio de `Pipeline`, fusionado en #1462 (sección 11.3) |
| D4 | Umbral de trabajo en backtestings sueltos | (a) no, solo reglas (E1); (b) umbral fijo de unos 5 s; (c) umbral según la plataforma; (d) umbral configurable por el usuario | **(b)**, después de medir en Linux (D7). Cambia el resultado de `'auto'` en cargas pequeñas (de 21 a 1), así que necesita nota de versión. No cambia la API. La opción (d) añadiría un parámetro público, y no lo recomiendo sin necesidad | (c), implementado en `perf/n-jobs-auto-work-threshold`, PR pendiente: 4 s en Windows, 2 s en Linux y, hasta medirlo, 4 s en macOS (secciones 11.4 y 16) |
| D5 | Búsquedas: resolver `'auto'` una vez | (a) sí, junto con D4; (b) no | **(a)**, solo necesario si se adopta D4 | (a) por candidato, en la misma rama que D4 (sección 11.4) |
| D6 | Regla de LightGBM con hilos propios | (a) mantener; (b) permitir paralelizar si `cpu_count() ≥ 16` | **(a)**: protege en máquinas pequeñas (×1,5 con 4) y en la grande pierde poco (×1,3) | (a), sin cambios. WSL2 la refuerza (sección 12.4) |
| D7 | Medir en Linux (WSL2) antes de D4 | (a) subconjunto (bloque A + V1 + celdas clave, unas 2 h); (b) no | **(a)**: el umbral de 5 s está calibrado en Windows; si en Linux el arranque es mucho menor, D4 podría pasar a (c) | (a), hecho (31 min, sección 12) |
| D8 | Documentación | (a) reescribir el cuaderno (sección 9.3) y recomendar estimadores de un hilo con refit; (b) solo corregir el texto de las reglas | **(a)**, en un PR de docs posterior a D1-D5 | Aplicada en la rama de D4 y D5 (el texto de reglas ya se corrigió en #1462) |
| D9 | Las predicciones de LightGBM ignoran `n_jobs` (nueva, sección 13) | (a) pasar `num_threads` al booster; (b) siempre 1 hilo; (c) 1 hilo solo con una fila | **(a)** | (a), fusionado en #1464 |

No se proponen cambios en el valor por defecto de `n_jobs` (`'auto'`) ni en la API pública.

## 11. Cambios propuestos

Diffs propuestos el 7 de noviembre sobre `origin/0.27.x` @ `76e4500dd`, antes de aplicarlos. Cada uno se aplicó después en su propia rama, con sus tests y la skill `verify`.

*Estado*: los cuatro cambios están aplicados, cada uno en su PR (11.4, pendiente de abrir). 11.2, 11.3 y 11.4 se aplicaron con cambios respecto a lo que se propone aquí; las diferencias están en una nota al final de cada subsección.

### 11.1 Direct y DMV: vaciar `estimators_` antes del `fit` paralelo (D1)

```diff
--- a/skforecast/direct/_forecaster_direct.py
+++ b/skforecast/direct/_forecaster_direct.py
@@ def fit(  (bloque de reinicio, línea 1886)
         self.is_fitted                            = False
         self.fit_date                             = None
+        # Unfitted estimators, as after __init__. `forecaster=self` is sent to
+        # every step task of Parallel, so the estimators of a previous fit
+        # would be pickled once per step.
+        self.estimators_ = {step: clone(self.estimator) for step in self.steps}
```

Lo mismo en `skforecast/direct/_forecaster_direct_multivariate.py`, después de `self.fit_date = None` (línea 2038).

- **Impacto** [probado]:
  - reajuste de Direct L con 21 workers: LGBM 18,4 → 2,1 s; HGB 34 → 3,0 s;
  - backtesting Direct L con refit y `'auto'`: HGB 77,3 → 10,9 s; LGBM 70,7 → 13,2 s;
  - DMV S: 17,8 → 7,9 s.
- **Riesgo**: bajo. `estimators_` se reasigna al final del `fit` y cada paso se ajusta con su propio `clone(self.estimator)` [verificado en código]. Deja el mismo estado que `__init__`.
- **Test**: con un `_fit_one_step_estimator` interceptado, comprobar que en un reajuste las tareas reciben un forecaster con `estimators_` sin ajustar. Las predicciones no cambian.

**Aplicado** en #1460 (`a50b2317c`), como se propone aquí. Nota de versión: con `refit=True` y `n_jobs='auto'`, `backtesting_forecaster` es entre 5 y 7 veces más rápido.

### 11.2 `backtesting_stats` con Sarimax en paralelo (D2)

```diff
--- a/skforecast/model_selection/_validation.py
+++ b/skforecast/model_selection/_validation.py
@@ def _backtesting_stats(  (línea 2344)
-    results = Parallel(n_jobs=n_jobs)(
+    # NOTE: max_nbytes=None disables joblib's automatic memmapping. Large
+    # arrays would reach the workers as read-only memmaps, and the Kalman
+    # filter of statsmodels (Sarimax) cannot be unpickled from them.
+    results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
```

- **Impacto**: Sarimax con 2.000 filas y `n_jobs=4` pasa de fallar a funcionar: 5,5 s frente a 11,5 s secuencial, con predicciones idénticas [probado].
- **Test**: `backtesting_stats` con Sarimax, una serie de unas 2.000 filas, `refit=True` y `n_jobs=2`; comparar con `n_jobs=1`.
- **Nota de versión**: en "Fixed".

**Aplicado con otra solución** en #1468 (`2daa0e836`), tras comparar las opciones de joblib [probado]. La medida, en Windows y en WSL2, se hizo con `Sarimax(order=(1,1,1), seasonal_order=(1,0,1,7))`, 2.000 filas, `refit=True` y 10 folds:

| Opción | Resultado |
|---|---|
| Por defecto (`mmap_mode='r'`) | Falla con `BrokenProcessPool`, en Windows y en Linux |
| `max_nbytes=None` (la propuesta de arriba) | Funciona, pero cada tarea recibe una copia completa. Con 21 workers, el pico de memoria es de 6,2 GB frente a 5,7 GB; con `m=24`, 11,0 frente a 7,3 GB; con 50.000 filas, 25 GB |
| **`mmap_mode='c'`** (aplicada) | Funciona y los datos se siguen compartiendo. Windows: 12,2 s secuencial → 5,4-6,1 s con 4 workers y 4,2-7,3 s con 21. WSL2: 12,0 s → 7,6 s con 4 y 5,1 s con 21 |
| `mmap_mode='r+'` | Funciona, pero las escrituras llegan al archivo compartido |
| `prefer='threads'` | No vale: todas las tareas ajustarían el mismo objeto a la vez |
| Predecir el fold 0 en el proceso principal | Funciona, pero obliga a reestructurar `_backtesting_stats` |

- El argumento solo se pasa si algún estimador es un `Sarimax`. Un `Parallel` con un `mmap_mode` distinto del valor por defecto hace que loky reinicie los workers de una llamada anterior. Eso cuesta 1,1 s en Windows y 0,6 s en WSL2 sin importaciones, y más con las de skforecast.
- El tamaño de los arrays es de unos `nobs × k_states² × 8` bytes. Por eso el fallo depende del orden estacional además de la longitud: con `(1,1,1)(1,0,1,24)` bastan unas 200 filas. Un umbral por tamaño de serie no serviría.
- Sin probar: el ARIMA de sktime (pmdarima), que envuelve statsmodels y podría fallar igual. No está instalado.
- Las predicciones con `n_jobs > 1` son idénticas a las de `n_jobs=1`.

### 11.3 `select_n_jobs_backtesting`: "barato" en lugar de "lineal" (D3)

```diff
--- a/skforecast/model_selection/_utils.py
+++ b/skforecast/model_selection/_utils.py
@@ def select_n_jobs_backtesting(
-    forecaster_name = type(forecaster).__name__
-
-    if forecaster_name in ('ForecasterStats', 'ForecasterFoundation'):
-        n_jobs = 1
-        return n_jobs
-
-    if isinstance(forecaster.estimator, Pipeline):
-        estimator = forecaster.estimator[-1]
-    else:
-        estimator = forecaster.estimator
-
-    refit = False if refit == 0 else refit
-    if not isinstance(refit, bool) and refit != 1:
-        n_jobs = 1
-    else:
-        ... (ramas por forecaster)
-
-    return n_jobs
+    forecaster_name = type(forecaster).__name__
+    refit = False if refit == 0 else refit
+    if not isinstance(refit, bool) and refit != 1:
+        # Intermittent refit reuses the fitted forecaster between folds.
+        return 1
+
+    if forecaster_name not in {
+        'ForecasterRecursive',
+        'ForecasterRecursiveClassifier',
+        'ForecasterRecursiveMultiSeries',
+    }:
+        # ForecasterDirect and ForecasterDirectMultiVariate parallelize the
+        # fit of their steps instead (see `select_n_jobs_fit_forecaster`).
+        # ForecasterStats, ForecasterEquivalentDate and ForecasterRnn are not
+        # parallelized.
+        return 1
+
+    estimator = forecaster.estimator
+    last_step = estimator[-1] if isinstance(estimator, Pipeline) else estimator
+
+    if isinstance(estimator, LinearModel):
+        # Closed-form linear regressors are cheap to fit and predict with a dot
+        # product. A Pipeline is not cheap: it predicts step by step.
+        return 1
+
+    if (
+        type(last_step).__name__ in {'LGBMRegressor', 'LGBMClassifier'}
+        and last_step.n_jobs != 1
+    ):
+        # LightGBM with its own threads would oversubscribe the cores.
+        return 1
+
+    return max(1, cpu_count() - 1)
```

- **Cambios de comportamiento** [verificado en código]:
  - `Pipeline` con lineal: 1 → 21;
  - `LogisticRegression` y otros `LinearClassifierMixin`: 1 → 21;
  - RecursiveMultiSeries con `LinearModel`: 21 → 1.
  
  El resto es igual. `ForecasterRnn` sigue en 1, ahora de forma explícita: una lista de exclusión le daría 21 workers a un modelo de Keras por accidente. Desaparece la rama inalcanzable de Foundation.
- **Impacto**:
  - en caliente, `Pipeline(Ridge)` L 0,94 → 0,24 s y LogisticRegression L 4,6 → 1,0 s;
  - en frío, MultiSeries Ridge S deja de pagar unos 4 s por un trabajo de 0,07 s [probado e inferido].
- **Tests**: actualizar 5 casos de `test_select_n_jobs_backtesting.py` (líneas 21, 31, 33, 46 y 48; los de refit entero no cambian) y añadir casos con `Pipeline(LGBMRegressor())` y `ForecasterRnn`.
- **Docstring**: reescribir la lista de reglas para que describa este código (sección 8.1) y corregir la ruta `skforecast.utils.select_n_jobs_backtesting` en las docstrings del parámetro `n_jobs` de backtesting y búsquedas (sección 8.8).

`select_n_jobs_fit_forecaster` no cambia de lógica, solo de docstring. En el `fit` de Direct las tres reglas aciertan (sección 5.4) y un `Pipeline` lineal no es caro, porque cada paso predice una sola vez de forma vectorizada.

**Aplicado con un cambio** en #1462 (`1a3eadc51`). El `Pipeline` se sigue decidiendo por su último paso, así que `Pipeline(Ridge)` sigue en 1. Lo decidiste así por dos razones:
- su trabajo es de menos de 1 s;
- en un backtesting suelto en frío en Windows, 21 workers costarían unos 5 s.

El cambio de `Pipeline` (1 → 21) queda para D4: con un umbral de trabajo no hace falta decidirlo por tipo.

Lo demás se aplicó como se propone:
- `LogisticRegression` y otros clasificadores lineales: 1 → 21;
- RecursiveMultiSeries con `LinearModel`, también dentro de un `Pipeline`: 21 → 1;
- se elimina la rama de Foundation y se reescribe la docstring, incluida la regla de Direct con `refit=True` (sección 8.1);
- 18 docstrings remiten ahora a la ruta correcta, `skforecast.model_selection._utils.select_n_jobs_backtesting` (sección 8.8);
- el texto de reglas del cuaderno se rehízo para que coincida con las docstrings.

### 11.4 Umbral de trabajo en backtestings sueltos (D4 y D5)

```diff
--- a/skforecast/model_selection/_utils.py
+++ b/skforecast/model_selection/_utils.py
+# Sequential work (seconds) below which starting the joblib workers does not
+# pay off. Measured on Windows, where starting 21 workers that import
+# skforecast costs 3-4 s inside a backtesting (dev/n_jobs_review).
+_MIN_PARALLEL_WORKLOAD_SECONDS = 5.0
+
+
+def limit_n_jobs_by_workload(
+    n_jobs: int, n_tasks: int, n_fits: int, fit_seconds: float
+) -> int:
+    """
+    Reduce the number of jobs chosen with `'auto'` when the work is too small
+    to pay the start of the workers, and cap it by the number of tasks.
+    """
+    estimated_work = n_fits * fit_seconds
+    if n_jobs <= 1 or n_tasks <= 1 or estimated_work < _MIN_PARALLEL_WORKLOAD_SECONDS:
+        return 1
+    return min(n_jobs, n_tasks)

--- a/skforecast/model_selection/_validation.py
+++ b/skforecast/model_selection/_validation.py
+import time
@@ def _backtesting_forecaster(  (y análogo en _backtesting_forecaster_multiseries y _backtesting_stats)
+    auto_n_jobs = n_jobs == 'auto'
     if n_jobs == 'auto':
         n_jobs = select_n_jobs_backtesting(...)
@@
+    fit_seconds = 0.0
     if initial_train_size is not None:
+        t_start = time.perf_counter()
         forecaster.fit(...)
+        fit_seconds = time.perf_counter() - t_start
@@
+    if auto_n_jobs:
+        n_fits = sum(fold[5] for fold in folds)   # folds that are refitted
+        n_jobs = limit_n_jobs_by_workload(
+            n_jobs=n_jobs, n_tasks=len(fold_items), n_fits=n_fits,
+            fit_seconds=fit_seconds,
+        )
     backtest_predictions = Parallel(n_jobs=n_jobs)(

--- a/skforecast/model_selection/_search.py
+++ b/skforecast/model_selection/_search.py
@@ (antes del bucle de candidatos de cada búsqueda)
+    if n_jobs == 'auto':
+        # Decide once for the whole search: the workers started by the first
+        # candidate are reused by the rest, so the work threshold of a single
+        # backtesting does not apply.
+        n_jobs = select_n_jobs_backtesting(forecaster=forecaster, refit=cv.refit)
```

- **Impacto estimado** (E3 con umbral de 5 s frente a la regla actual, sección 10.2):
  - en frío, la pérdida total baja de 211 a 42 s (L) y de 174 a 15 s (S), y el peor caso pasa de ×63,8 a ×1,66;
  - en caliente se renuncia a entre 12 y 21 s frente a E1.
  
  [inferido y calibrado con V1]
- **Riesgos**:
  - el `fit` inicial puede ser poco representativo si los folds crecen mucho (HGB L: estima 4,0 s frente a 7,1 s reales);
  - sin refit siempre se usa 1, aunque en caliente HGB L ganaría 2,4 s;
  - el umbral está calibrado en Windows. En WSL2 el equilibrio está en unos 2,8 s (sección 12.3), así que la constante depende de la plataforma.
- **Por resolver al implementarlo**: la forma exacta de contar `n_fits` con `refit` entero (aunque en ese caso ya es 1), con `fold_stride` y con folds saltados, y si el `fit` de Direct, cuyo `n_jobs` se fija en `__init__`, debe aplicar el mismo umbral. Con el tiempo del primer paso, el `fit` de Direct S con LGBM(1) (2,7 s secuencial) pasaría a 1 en frío [inferido].

**Aplicado** en la rama `perf/n-jobs-auto-work-threshold` (desde `0.27.x` @ `2daa0e836`, PR pendiente), con estas diferencias respecto al boceto:

- **`limit_n_jobs_by_workload(n_jobs, n_tasks, n_fits, fit_seconds)`** (`_utils.py`, documentada en la página de la API junto a `select_n_jobs_backtesting`):
  - devuelve 1 si `n_jobs` es 1, si hay una sola tarea o si ningún fold reajusta (sin refit, aunque el `fit` inicial sea largo);
  - devuelve 1 si `n_fits × fit_seconds` no llega al mínimo de la plataforma: `{"win32": 4.0, "linux": 2.0}`, y 4,0 en cualquier otra, macOS incluido, hasta medirlo (sección 16);
  - si no, `min(n_jobs, n_tasks)`;
  - si el `fit` inicial no se cronometró (`fit_seconds=None`), no estima el trabajo y solo limita por tareas. Pasa en MultiSeries cuando ningún nivel del primer fold se predice con su modelo y ese fold no se ajusta.
- **`n_fits`** son los folds que reajustan, sin el `fit` inicial: `fold[5] is True` tras marcar el primero como ajustado en una serie, y `sum(fit_folds)` en MultiSeries, que ya descuenta los folds saltados. `fold_stride` y los folds saltados no necesitan nada más. Con `refit` entero, `select_n_jobs_backtesting` ya da 1.
- **Los valores, 4 s y 2 s, están en unidades de la estimación**, no del tiempo real. La estimación con `n_fits` (sin el `fit` inicial) da entre el 53 y el 91 % del tiempo secuencial real, con mediana del 78 % [probado, V1 de las dos plataformas]. `threshold.py` lo calcula igual en todas las plataformas:
  - ajusta `paralelo = a + b × secuencial` sobre los casos de V1 con refit, sin Direct: en Windows, `a` = 4,18 s y `b` = 0,181 (×5,5); en WSL2, 2,31 s y 0,172 (×5,8);
  - el equilibrio en tiempo real es `a / (1 − b)`: 5,10 s en Windows y 2,79 s en WSL2;
  - multiplicado por la mediana de la estimación (0,78 y 0,77), da 4,0 y 2,15 s, que se redondean a 0,5 s: **4,0 y 2,0 s** [inferido de medidas, `results/threshold.md` y `results/wsl/threshold.md`].
- **Búsquedas (D5)**: resuelven `'auto'` con `select_n_jobs_backtesting` en cada candidato, no una vez para toda la búsqueda, porque la regla de LightGBM depende de los parámetros del candidato. Pasan el entero al backtesting, así que el umbral no se aplica. Afecta a `grid_search_forecaster`, `random_search_forecaster`, `bayesian_search_forecaster` y sus versiones multiseries. Las de Stats y EquivalentDate no cambian: sus reglas ya dan 1.
- **El `fit` de Direct no cambia**: su `n_jobs` se fija en `__init__` y su paralelismo se decide fuera del backtesting. Su backtesting ya usaba 1 por fuera.
- **Tests**: 14 de `limit_n_jobs_by_workload` (casos sin nada que repartir, mínimo por plataforma con `sys.platform` parcheado, límite por tareas); 4 por función de backtesting y 3 por búsqueda, con `Parallel` sustituido por uno secuencial, que comprueban el `n_jobs` que recibe. Los de backtesting fallan sin el cambio (6 de 8; los 2 que pasan son los de `n_jobs` entero, que no cambia).
- **Docstrings**: el `n_jobs` de `backtesting_forecaster` y `backtesting_forecaster_multiseries` (públicas y privadas) explica el umbral; la de `select_n_jobs_backtesting` remite a `limit_n_jobs_by_workload`. Nota de versión en `0.27.0`, apartado Changed.

**Sin refit con predicciones caras** [probado]. El umbral manda a 1 todo backtesting sin refit, también con intervalos por bootstrapping, cuyo coste de predicción no estima. Con 10.000 filas, 20 folds y `n_boot=250` (LGBM(1)), o 50 series × 2.000 filas sin y con bootstrapping, el secuencial tarda de 2,2 a 4,4 s; con 21 workers, de 7,5 a 9,5 s en frío y de 1,2 a 6,4 s en caliente. Solo un caso gana en caliente (2,2 → 1,2 s). Con cargas mucho mayores sí ganaría; quien las tenga puede pasar un `n_jobs` entero.

**Comprobación en Windows con el umbral** [probado, `results/d4_windows/`, `bench_cold.py --values 1,auto,cpu-1`, mediana de 3 procesos por celda]. La máquina estaba más cargada que en la medida del estudio (los secuenciales tardan entre 1,5 y 2 veces más), pero los tres valores de `n_jobs` se midieron intercalados:

| Backtesting suelto | n_jobs = 1 | 21 workers | `'auto'` (jobs) | `'auto'` / 1 | Trabajo estimado |
|---|---|---|---|---|---|
| Recursive LGBM(1), S, sin refit | 0,30 | 4,11 | 0,30 (1) | ×1,00 | 0 |
| Recursive LGBM(1), L, sin refit | 0,78 | 5,88 | 0,86 (1) | ×1,10 | 0 |
| Recursive LGBM(1), S, con refit | 2,05 | 4,32 | 2,03 (1) | ×0,99 | 1,77 |
| MultiSeries LGBM(1), S, con refit | 2,47 | 4,39 | 2,55 (1) | ×1,03 | 2,26 |
| Recursive HGB, L, con refit | 8,55 | 7,03 | 6,75 (18) | ×0,79 | 4,17 |
| Recursive LGBM(1), L, con refit | 10,91 | 6,57 | 6,91 (18) | ×0,63 | 8,89 |
| Direct LGBM(1), S, con refit | 42,20 | 11,52 | 11,37 (1, `fit` en paralelo) | ×0,27 | 28,98 |
| MultiSeries LGBM(1), L, con refit | 45,75 | 14,78 | 13,48 (20) | ×0,29 | 33,05 |
| Recursive RF, L, con refit | 80,16 | 15,42 | 16,26 (18) | ×0,20 | 60,65 |

1. **`'auto'` nunca tarda más de 1,10 veces lo que la mejor de las dos opciones** (mediana ×1,00). Antes del umbral, `'auto'` era la columna de 21 workers: hasta ×13,6 frente a `n_jobs=1` en esta medida y ×19,8 en la del estudio.
2. **Los jobs siguen las reglas**: 1 sin refit y por debajo de 4 s; el número de folds (18 o 20, menos que 21) por encima.
3. **El caso más cercano al umbral, HGB L (4,17 s estimados), queda por encima y gana** (×0,79). En la medida del estudio, con la máquina menos cargada, estimaba 3,76 s y habría quedado por debajo, perdiendo 1,4 s: es el error esperado cerca del umbral (sección 12.3).
4. Con estos datos, `threshold.py` sugiere 4,5 s, porque la carga sube el coste de arrancar (`a` = 4,98 s). Se mantiene 4,0 s, el valor de la medida limpia, que además queda dentro de la horquilla de esta (pierde hasta 2,26 s, gana desde 4,17 s).

**Con 4 procesadores** [probado, `results/d4_windows_aff4/`, `bench_cold.py --affinity 4`, una repetición de 18 celdas; la medida se interrumpió porque con estas bastaba]. El umbral se calibró con 21 workers, pero vale igual en una máquina pequeña, porque arrancar menos workers cuesta menos (2,9 s con 3) y la aceleración máxima también es menor, y los dos efectos se compensan en el punto de equilibrio:
- Sin refit, 3 workers siguen siendo 17 y 6 veces más lentos que `n_jobs=1` (S: 0,20 frente a 3,42 s; L: 0,63 frente a 4,03 s); `'auto'` usa 1 job (0,23 y 0,59 s).
- MultiSeries S con refit: 1,96 s secuencial, 4,97 s con 3 workers, 2,16 s con `'auto'` (1 job).
- Recursive LGBM(1) L con refit, estimación de 7 s, por encima del umbral: `'auto'` usa 3 jobs y tarda 7,36 s frente a 7,13 s secuencial, el empate esperado cerca del punto de equilibrio.
- Los casos grandes ganan menos que con 21 workers, pero ganan: MultiSeries L 38,3 → 23,8 s y RF L 63,7 → 30,8 s con 3 workers.

**Comprobación en Linux (WSL2) con `run_platform.sh`** [probado, `results/linux_wsl_d4/`, 1 repetición en V1]. Es la ejecución completa (`--full`) de la que Javier hará en macOS (sección 16), con el código de la rama copiado al clon de WSL2. Sirve para comprobar el script en un sistema POSIX; con una repetición, los tiempos son orientativos.
- El script completo tarda 10 min: 28 tests pasan, el bloque A tarda 4,5 min y V1, 5 min.
- Arranque en frío de 21 workers: 3,17 s (2,94 s en la medida de la sección 12).
- `'auto'` usa 1 job en los cuatro casos pequeños y 18 o 20 en los grandes, como se espera con el umbral de 2 s. En los casos sin refit, `'auto'` / 1 es ×0,55 y ×1,27, y en los pequeños con refit ×1,47 y ×1,81, aunque `'auto'` también usó 1 job: es ruido de una sola repetición, no del umbral.
- `threshold.py` sugiere 2,5 s, con la horquilla entre 1,10 y 4,26 s. Se mantiene 2,0 s, el valor de la medida de 5 repeticiones, que también queda dentro.

### 11.5 Resumen del impacto

| Cambio | Tipo | Impacto principal | Tests |
|---|---|---|---|
| 11.1 `estimators_` | perf | Direct/DMV con refit, ×5-11 | 1 nuevo |
| 11.2 `mmap_mode='c'` (aplicado; se propuso `max_nbytes=None`) | fix | Sarimax en paralelo deja de fallar | 1 nuevo |
| 11.3 "barato" | heurística | Pipeline y LogReg ×4-5; MultiSeries lineal sin arranque inútil | 5 actualizados, 2 nuevos |
| 11.4 umbral | heurística | backtestings sueltos pequeños hasta ×20 más rápidos en Windows | 14 para `limit_n_jobs_by_workload`, 8 en los backtestings y 12 en las búsquedas |
| Docstrings y cuaderno | docs | Reglas correctas, benchmark reproducible | `verify` (docs) |

## 12. Linux: medida en WSL2 (D7)

### 12.1 Método

- **Misma máquina**, con WSL2:
  - Ubuntu 24.04 y kernel 6.6.87.2-microsoft-standard-WSL2;
  - 22 procesadores lógicos y 31 GB de RAM visibles.
- **El invitado ve 11 núcleos con 2 hilos cada uno, sin distinción P/E.** Por eso `joblib.cpu_count(only_physical_cores=True)` = 11, y `LGBMRegressor()` usa 11 hilos (16 en Windows) [probado, `results/wsl/machine.json`].
- **Entorno** conda `skforecast_27_py13_wsl`, con Python 3.13.18 y las mismas versiones que en Windows (sin torch).
- **Código**: un clon del repositorio en el sistema de archivos de Linux, en `1a3eadc51` (`0.27.x` con D1 y D3).
  - Los scripts de esta carpeta se adaptaron a Linux con `wsl_patch.py`: topología desde `/sys`, carpeta de resultados configurable y `bench.py --subset`.
  - El clon lleva aplicada la primera versión del arreglo de D9 (`wsl_copy_changes.diff`). Sin él, los tiempos secuenciales de LightGBM no eran válidos (sección 13). Con `LGBMRegressor(n_jobs=1)`, la versión fusionada usa el mismo hilo.
- **Subconjunto** (`run_wsl.sh full`, 31 min), con las mismas repeticiones que en Windows:
  - la puerta G1;
  - el bloque A completo y V1;
  - las celdas L clave de los bloques B a E;
  - la reproducción del fallo de Sarimax.

  No hay bloque H: el invitado no tiene topología híbrida que emular.
- **Workers** [probado, `results/wsl/probe.csv`]: loky fija `OMP_NUM_THREADS` y las variables de BLAS de cada worker a `cpu_count() // n_jobs` (11 con 2 workers, 1 con 21), igual que en Windows.
- **Equivalencia** [probado]: las 28 celdas de los bloques B a E dan predicciones idénticas a las de `n_jobs=1` (diferencia máxima 0).

### 12.2 Arranque de los workers (bloque A)

Tiempo de la llamada a `Parallel`, mediana de 5 repeticiones, en segundos [probado, `results/wsl/block_A_*.csv`]:

| Caso | k = 2 | 4 | 8 | 16 | 21 | 22 |
|---|---|---|---|---|---|---|
| En frío, forecaster, WSL2 | 1,34 | 1,46 | 1,68 | 2,33 | **2,94** | 3,00 |
| En frío, forecaster, Windows | 2,20 | 2,65 | 3,02 | 4,19 | **5,68** | 5,96 |
| En frío, tareas triviales, WSL2 | 0,22 | 0,19 | 0,22 | 0,23 | 0,27 | 0,33 |
| En frío, tareas triviales, Windows | 0,42 | 0,45 | 0,50 | 0,63 | 0,66 | 0,70 |
| En caliente, forecaster, WSL2 | 0,02 | 0,03 | 0,04 | 0,07 | 0,12 | 0,10 |
| En caliente, forecaster, Windows | 0,03 | 0,05 | 0,10 | 0,13 | 0,16 | 0,17 |

1. **En WSL2, arrancar en frío cuesta la mitad que en Windows**: unos 0,08 s por worker, frente a 0,18 s [probado].
2. **El grueso sigue siendo importar las librerías.** Con tareas triviales cuesta de 0,2 a 0,3 s [probado].
3. **En caliente, el coste es despreciable en las dos plataformas** [probado].

### 12.3 Backtestings sueltos en frío (V1)

Cada repetición se ejecuta en un proceso nuevo, igual que en la sección 4.1. Mediana de 5 procesos, en segundos [probado, `results/wsl/block_V_cold.csv`]:

| Backtesting suelto | n_jobs = 1 | `'auto'` | `'auto'` / 1 | Windows: `'auto'` / 1 |
|---|---|---|---|---|
| Recursive LGBM(1), S, sin refit | 0,10 | 1,96 | **×19,6** | ×19,8 |
| Recursive LGBM(1), L, sin refit | 0,37 | 2,65 | ×7,2 | ×8,7 |
| Recursive LGBM(1), S, con refit | 0,90 | 2,10 | ×2,3 | ×3,6 |
| MultiSeries LGBM(1), S, con refit | 1,21 | 2,13 | ×1,8 | ×2,5 |
| Recursive HGB, L, con refit | 4,73 | 3,26 | ×0,69 | ×0,81 |
| Recursive LGBM(1), L, con refit | 5,30 | 3,48 | ×0,66 | ×0,99 |
| Direct LGBM(1), S, con refit | 19,42 | 6,35 | ×0,33 | ×0,44 |
| MultiSeries LGBM(1), L, con refit | 28,87 | 8,22 | ×0,28 | ×0,36 |
| Recursive RF, L, con refit | 48,02 | 9,96 | ×0,21 | ×0,25 |

1. **El sobrecoste real en frío es de 2,2 a 2,5 s** [probado]. Es lo que cuesta `'auto'` en frío menos lo que cuesta en caliente, en las cuatro celdas que tienen las dos medidas (LGBM(1) L con y sin refit, HGB L con refit y MultiSeries L con refit). En Windows, esas mismas celdas dan entre 4,1 y 4,4 s.
2. **En los casos pequeños, `'auto'` pierde lo mismo en proporción** (×19,6 frente a ×19,8), porque en WSL2 también es más rápido el trabajo secuencial [probado].
3. **El punto de equilibrio baja a unos 2,5-3 s de trabajo secuencial**, frente a unos 6 s en Windows [inferido].
   - Con una aceleración en caliente de ×4,5-5 y 2,3 s de arranque, `T / 4,7 + 2,3 = T` da T ≈ 2,9 s. La misma cuenta da unos 5,5 s en Windows.
   - Las medidas lo acotan: con 1,21 s secuenciales pierde, con 4,73 s gana.
4. **La estimación desde el `fit` inicial también vale en Linux** [probado]: `t_fit × n_folds` da entre el 61 y el 100 % del tiempo secuencial real. HGB L vuelve a ser el peor caso (2,9 s estimados frente a 4,7 s reales).

**Consecuencia para D4.** Un umbral único de 5 s, calibrado en Windows, dejaría en Linux sin paralelizar trabajos de 3 a 5 s que sí ganan [inferido, con las cifras de V1]:
- HGB L con refit: el `fit` estima 2,9 s, así que usaría 1 (4,73 s) en lugar de `'auto'` (3,26 s);
- LGBM(1) L con refit: el `fit` estima 4,9 s, así que usaría 1 (5,30 s) en lugar de 3,48 s.

El error opuesto es parecido. Con un umbral de 3 s en Windows, un trabajo de 3 s tardaría unos 4,9 s en paralelo. En los dos casos se pierde entre 1,5 y 2 s cerca del umbral, así que la diferencia no es dramática. Aun así, el umbral por plataforma (opción c) acierta en las dos. La recomendación pasa a ser **(c)**. Los valores aplicados, 4 s en Windows y 2 s en Linux, están en unidades de la estimación del código, que da un 80 % del tiempo real (sección 11.4). macOS se medirá aparte (sección 16).

### 12.4 Reglas en caliente (bloques B a E)

Mediana de 5 llamadas en caliente, en segundos, con la máquina entera [probado, `results/wsl/block_*.csv`]:

| Caso (L) | WSL2: 1 | WSL2: 21 | Windows: 1 | Windows: 21 |
|---|---|---|---|---|
| Recursive Ridge, sin refit / con refit | 0,02 / 0,12 | 0,03 / 0,05 | 0,02 / 0,11 | 0,05 / 0,08 |
| Recursive LGBM(1), sin refit | 0,32 | 0,33 | 0,55 | 0,50 |
| Recursive LGBM(1), con refit | 4,71 | **1,03** (×4,6) | 5,62 | **1,33** (×4,2) |
| Recursive HGB, sin refit | 1,67 | **0,40** (×4,2) | 2,94 | **0,58** (×5,1) |
| Recursive HGB, con refit | 4,85 | **0,98** (×4,9) | 7,46 | **1,38** (×5,4) |
| Recursive RF(`n_jobs=-1`), con refit | 7,12 | **5,93** | 9,79 | **6,77** |
| `fit` de Direct, Ridge | **0,10** | 0,57 | **0,17** | 0,78 |
| `fit` de Direct, LGBM(1) | 12,15 | **1,98** (×6,1) | 13,54 | **3,12** (×4,3) |
| MultiSeries LGBM(1), sin refit | 1,50 | 1,52 | 1,95 | 1,83 |
| MultiSeries LGBM(1), con refit | 28,97 | **6,01** (×4,8) | 28,68 | **6,43** (×4,5) |

Con 4 y 8 workers (bloque C), el orden es el mismo en las dos plataformas: 1 > 4 > 8 > 21. HGB con refit tarda 5,58 / 2,42 / 1,83 / 1,14 s en WSL2.

1. **Todos los veredictos de la sección 5 se mantienen en Linux** [probado]: las mismas reglas aciertan, con aceleraciones parecidas (×4,2-6,1). `cpu_count() - 1` también es 21.
2. **LightGBM con hilos propios es mucho peor en Linux** [probado, medidas de D9 en WSL2, 10.000 filas, 48 pasos]:
   - `LGBMRegressor(n_jobs=-1)` (22 hilos) tarda entre 63 y 87 s en un backtesting sin refit con `n_jobs=1`, casi todo en el `fit`. Con `LGBMRegressor()` (11 hilos) tarda entre 0,15 y 0,36 s (3 repeticiones).
   - Con `refit=True`, `LGBMRegressor()` y 21 workers explícitos, el backtesting pasa de 600 s, por sobresuscripción en el `fit`. Pasaba antes y después del arreglo de D9.
   - Con `'auto'` no ocurre: la regla de LightGBM (D6) da 1.

   Esto refuerza D6 (mantener la regla). Para D8, la documentación debería recomendar `n_jobs=1` en el estimador cuando se paraleliza el backtesting, y advertir de `n_jobs=-1` en Linux.

### 12.5 Sarimax

El fallo de la sección 8.11 se reproduce igual en Linux: 2.000 filas, 4 workers, `BrokenProcessPool` con `ValueError: buffer source array is read-only` en `dKalmanFilter.__setstate__` [probado, `results/wsl/exp_sarimax_parallel.txt`]. Con el arreglo de #1468 funciona: 12,0 s secuencial → 7,6 s con 4 workers y 5,1 s con 21 [probado].

### 12.6 Límites

- **WSL2 es una máquina virtual sobre Hyper-V.** No se ha medido Linux nativo, que probablemente arranque igual o más rápido [inferido].
- **Una sola máquina.**
- **macOS no se ha medido.** Javier Escobar lo medirá con el bloque A y V1, que bastan para fijar su umbral. Los scripts ya funcionan en Linux y macOS sin parches (sección 16).

## 13. Fallo nuevo: las predicciones de LightGBM ignoraban `n_jobs` (D9)

**Cómo apareció.** En el piloto de WSL2, el backtesting Recursive LGBM(1) L sin refit con `n_jobs=1` tardaba 16,0 s, frente a 0,55 s en Windows [probado].

**Causa** [verificado en código]:
- `_build_predict_function` (`skforecast/utils/utils.py`) prepara un atajo de predicción para `LGBMRegressor` que llama a `booster_.predict(X)` sin `num_threads`.
- Sin ese argumento, el booster usa todos los hilos de OpenMP del proceso (22), sea cual sea el `n_jobs` del estimador.
- La predicción recursiva lo llama una vez por paso, con una sola fila.

En los workers no ocurría: loky fija `OMP_NUM_THREADS=1`.

Coste por llamada, de 3.000 predicciones de una fila con LightGBM 4.7.0 [probado, `results/wsl/booster_threads_probe.py`]:

| Llamada | Windows | WSL2 |
|---|---|---|
| `estimator.predict` | 0,353 ms | 0,318 ms |
| `booster_.predict(X, num_threads=1)` | 0,043 ms | 0,038 ms |
| `booster_.predict(X)` (antes del arreglo) | 0,060 ms | **6,581 ms** |

En WSL2, la espera activa de libgomp con 22 hilos sobre procesadores virtuales multiplica por 170 el coste de una predicción de una fila [inferido para la causa, probado para el efecto].

**Arreglo** (#1464, `40042d1b1`):
- `num_threads` es el del ajuste (`booster.params["num_threads"]`), limitado por `OMP_NUM_THREADS` si es un entero positivo.
- La primera versión, `_process_n_jobs(n_jobs)` (`results/wsl/d9_proposal.md`), tenía dos fallos:
  - no veía `num_threads=3` pasado al constructor;
  - en los workers, su valor explícito anulaba el límite de loky. En WSL2, `LGBMRegressor()` con 21 workers y sin refit pasaba de 0,29 s a 5,4 s.

**Impacto** [probado]:
- `backtesting_forecaster` con `n_jobs=1` y `LGBMRegressor()` o `LGBMRegressor(n_jobs=1)` es entre 17 y 28 veces más rápido en WSL2 (10.000 filas, 48 pasos, sin refit). Por ejemplo, con `LGBMRegressor()`: 4,5 s → 0,18 s.
- En Windows es neutro: 0,24 s → 0,23 s.
- Las predicciones son idénticas.

**Alternativas descartadas** [probado]:
- **Siempre 1 hilo.** Solo es lo mejor con una fila. Con entre 50 y 50.000 filas por llamada, un hilo es entre 2,6 y 8,8 veces más lento por llamada que los hilos del ajuste. Esas llamadas aparecen en el bootstrapping (`n_boot`) y con varios niveles en MultiSeries.
- **1 hilo solo con una fila.** Gana unos 0,03 s por backtesting.

**Efecto sobre las tablas de Windows** [inferido]: se midieron antes del arreglo. En Windows, el atajo costaba como mucho un 10 % en LightGBM secuencial sin refit, por debajo de la precisión de la sección 5, así que ninguna conclusión cambia. Las medidas de WSL2 ya llevaban el arreglo.

## 14. Estado y pasos siguientes

| Decisión | PR | Commit en `0.27.x` | Qué cambia para el usuario (nota de versión) |
|---|---|---|---|
| D1 | #1460 | `a50b2317c` | Backtesting de Direct y DMV con `refit=True` y `'auto'`, entre 5 y 7 veces más rápido |
| D3 | #1462 | `1a3eadc51` | `'auto'` paraleliza Classifier con `LogisticRegression` (×4,7) y deja en 1 MultiSeries con lineal |
| D9 | #1464 | `40042d1b1` | Predicciones con `LGBMRegressor`, entre 17 y 28 veces más rápidas en WSL2 con `n_jobs=1` |
| D2 | #1468 | `2daa0e836` | `backtesting_stats` con Sarimax y `n_jobs > 1` deja de fallar |
| D4 y D5 | pendiente | rama `perf/n-jobs-auto-work-threshold` | Con `'auto'`, un backtesting suelto pequeño ya no arranca los workers (sección 11.4) |

Pendiente, por orden:

1. **Revisar y abrir el PR de D4 y D5** (sección 11.4), rama `perf/n-jobs-auto-work-threshold`. Lleva también esta carpeta, en un commit aparte del código.
2. **macOS**: medir y fijar su umbral, que hoy es el de Windows (sección 16). Puede ir en un PR pequeño después del de D4.
3. **D8, el cuaderno `docs/faq/parallelization-skforecast.ipynb`**: reescrito el 9 de noviembre en la rama de D4 y D5, siguiendo la sección 9.3.
   - El texto explica qué se paraleliza y muestra las reglas de `'auto'` en una tabla de decisión, un diagrama HTML (como el del cuaderno de foundation models; Mermaid no está configurado en el sitio) y una celda que llama a `select_n_jobs_backtesting` con varios forecasters, de modo que las reglas que se ven son las de la versión instalada. Cita el umbral de trabajo de Windows (4 s) y Linux (2 s) y no el de macOS, pendiente de medir. Explica el arranque de los workers, los hilos de los estimadores, el aviso de LightGBM con `n_jobs=-1` en Linux y `LOKY_MAX_CPU_COUNT`, y cierra con cuatro recomendaciones.
   - El benchmark usa `bike_sharing` y `store_sales` (50 series de una tienda), compara 1, `'auto'` y `cpu_count() - 1` y separa la primera llamada (con los workers parados) de la mediana de 3 más, con un gráfico de barras por caso. Tarda unos 5 min en esta máquina. Las salidas guardadas son de la tercera corrida del 9 de noviembre: la segunda coincidió con carga de fondo y salió el doble de lenta en todos los casos (y `'auto'` paralelizó el caso de 14 folds), y el `fit` secuencial de Direct varía hasta un 60 % entre llamadas incluso con la máquina tranquila; las conclusiones lo dicen.
   - Las cifras de Linux no se citan: el texto solo dice que arrancar los workers cuesta más en Windows.
   - Hallazgo al calibrarlo: cada worker importa las librerías con su primera tarea, así que una búsqueda con menos folds que workers reparte ese coste entre los primeros candidatos (4 candidatos de 6 folds: 10 s la primera vez frente a 3,6 en secuencial). Limitar el `n_jobs` de cada candidato al número de folds lo evitaría; no está medido ni aplicado.

## 15. Reproducción

```bash
# Desde la raíz del repositorio, con el entorno conda del proyecto
set PYTHONPATH=<raíz del repositorio>
python dev/n_jobs_review/run_all.py --affinity 22,4,8   # barrido completo (unas 6 h)
python dev/n_jobs_review/run_all.py --resume            # continuar una ejecución interrumpida
python dev/n_jobs_review/exp_direct_refit.py           # reajuste de Direct (unos 30 min)
python dev/n_jobs_review/exp_core_speed.py             # velocidad por tipo de núcleo
python dev/n_jobs_review/summarize.py                  # tablas en results/summary.md
```

| Archivo | Contenido |
|---|---|
| `common.py` | Topología, afinidad, información de la máquina, CSV con reanudación |
| `cases.py` | Datos sintéticos, estimadores y operaciones cronometradas |
| `patched.py` | Direct con el arreglo de `estimators_` emulado desde fuera del paquete |
| `bench.py` | Bloques B a G (celdas, calentamiento, repeticiones, tiempo límite) |
| `bench_spawn.py` | Bloque A (coste de arrancar workers) |
| `probe_workers.py` | Puerta G1: afinidad, `cpu_count` y límites de hilos dentro de los workers |
| `run_all.py`, `estimate.py`, `summarize.py` | Orquestación, estimación desde el piloto, resúmenes |
| `bench_cold.py` | V1: backtestings sueltos en un proceso nuevo (`--values 1,auto,cpu-1`; con el umbral, registra en `auto_resolved` los jobs que usó `'auto'`; `--cases` para un subconjunto, `--affinity N` para emular una máquina de N procesadores, solo Windows) |
| `threshold.py` | Umbral de una plataforma: rápido desde el bloque A (0,7 × arranque de `cpu_count() − 1` workers) y, si hay V1, el del método completo y la comprobación de `'auto'` (sección 11.4) |
| `run_platform.sh` | Medida en macOS o Linux: información de la máquina, tests, bloque A y umbral (unos 6 min); con `--full`, también V1 (sección 16) |
| `results/*.csv` | Una fila por llamada (`phase` = warmup, rep, preds o timeout) |
| `results/machine.json` | Máquina, topología, plan de energía y versiones |
| `results/threshold.md`, `results/wsl/threshold.md` | Salida de `threshold.py` con los datos de V1 del estudio (Windows y WSL2) |
| `results/d4_windows/` | V1 con el umbral implementado (Windows): `block_V_cold.csv`, `machine.json`, `threshold.md` |
| `results/d4_windows_aff4/` | V1 con el umbral implementado y 4 procesadores (afinidad), parcial: 18 celdas de una repetición |
| `results/linux_wsl_d4/` | `run_platform.sh` en WSL2 con el umbral implementado (1 repetición en V1): bloque A, V1, tests y `threshold.md` |
| `results/wsl/` | Medida en WSL2 (sección 12): CSV de los bloques A a E y V1, `probe.csv`, `machine.json` y `exp_sarimax_parallel.txt`, con sus registros (`wsl_pilot.log`, `wsl_full.log`) |
| `results/wsl/*.sh`, `wsl_patch*.py` | Instalación del entorno, clon del repositorio, orquestación (`run_wsl.sh pilot\|full`) y los parches que se aplicaron al clon |
| `results/wsl/booster_threads_probe.py`, `d9_proposal.md`, `wsl_copy_changes.diff` | D9: coste por llamada del booster, primera propuesta de arreglo (sustituida en #1464) y parche aplicado en el clon de WSL2 |

Los scripts funcionan en Windows, Linux y macOS. La afinidad (máquinas emuladas) solo existe en Windows. Dos variables de entorno cambian dónde leen y escriben:
- `NJOBS_RESULTS_DIR`: carpeta de resultados (por defecto, `results/`);
- `NJOBS_REPO_ROOT`: repositorio desde el que se debe importar skforecast (por defecto, el que contiene esta carpeta). Sirve para medir otra rama, por ejemplo en un worktree.

Para repetir la medida de WSL2, ejecuta `wsl_install.sh` y `wsl_clone.sh`, y después `bash run_wsl.sh full` dentro de WSL. `wsl_clone.sh` es el que se usó: fija el commit de `0.27.x` medido y copia esta carpeta desde la rama local en la que empezó el estudio (`docs/n-jobs-review`); para repetirlo, cambia esas dos líneas por un `git checkout` de la rama o el commit que quieras medir, que ya lleva la carpeta. Lánzalo desde un archivo de script: Git Bash altera las asignaciones `VAR=...` de `wsl -- bash -lc`. `wsl_patch.py` adaptó la versión anterior de los scripts a Linux; ya no hace falta, porque la topología de Linux, `NJOBS_RESULTS_DIR` y `bench.py --subset` están en los scripts. `wsl_patch2.py` emulaba el arreglo de D9 antes de #1464.

Para medir el umbral en cualquier Linux o macOS basta con `run_platform.sh` (sección 16).

## 16. Completar macOS (instrucciones para Javier)

El umbral de trabajo de `'auto'` (sección 11.4) está calibrado en Windows (4 s) y Linux (2 s). macOS usa provisionalmente el de Windows. Faltan dos cosas, que se resuelven con una sola ejecución de unos 6 minutos:
1. comprobar que la lógica nueva funciona bien en macOS (los tests);
2. medir lo que cuesta arrancar los workers y fijar el umbral de macOS (bloque A).

Solo hace falta medir el arranque porque el umbral es `coste de arranque / (1 − b) × (estimación / real)`, y los otros dos factores dependen del trabajo, no de la máquina: la pendiente `b` fue 0,151 en Windows y 0,154 en WSL2, y la relación estimación/real 0,79 y 0,87. Con ellos, el umbral es 0,7 veces el arranque en frío de `cpu_count() − 1` workers, y ese factor reproduce el método completo en las dos plataformas medidas: Windows 5,68 s → 4,0 s (método completo 4,0 a 4,5 s); WSL2 3,17 s → 2,2 s (2,0 a 2,5 s). La medida completa (V1, unos 40 minutos) queda como opcional, con `--full`, para confirmar el valor caso por caso.

### 16.1 Qué necesitas

- **El código con el umbral y esta carpeta**: los dos están en la rama `perf/n-jobs-auto-work-threshold` y, cuando se fusione, en `0.27.x`.
  ```bash
  git fetch origin
  git checkout perf/n-jobs-auto-work-threshold   # o 0.27.x, si ya está fusionada
  ```
- **Un entorno conda** con las dependencias de desarrollo de skforecast (`pip install -e ".[test]"` o las de tu entorno de siempre), más `psutil`. LightGBM tiene que funcionar: en macOS necesita `libomp` (`brew install libomp`, o LightGBM de conda-forge). Los scripts importan skforecast del repositorio con `PYTHONPATH`, no el instalado en el entorno, y paran si no es así.
- **El Mac en condiciones de medida**: enchufado, sin otras aplicaciones pesadas abiertas y sin tocarlo mientras mide.

### 16.2 Ejecutar

```bash
conda activate <tu entorno>
bash dev/n_jobs_review/run_platform.sh macos_<chip>      # por ejemplo, macos_m3_pro
```

El script hace, por orden:
1. Comprueba que skforecast se importa del repositorio y que tiene el umbral, y guarda la máquina en `results/macos_<chip>/machine.json`.
2. Pasa los tests del umbral y de las búsquedas (`tests.log`).
3. Bloque A: arranque en frío y en caliente de k workers (`block_A_cold.csv`, `block_A_warm.csv`), unos 5 minutos.
4. Solo con `--full`: V1, nueve backtestings sueltos, cada uno en un proceso nuevo, con `n_jobs=1`, `'auto'` y `cpu_count() - 1` workers, 5 repeticiones (`block_V_cold.csv`), unos 35 minutos.
5. Escribe el análisis en `threshold.md`: el umbral rápido a partir del bloque A y, con `--full`, también el del método completo y la comprobación de `'auto'`.

Con `PYTHON=/ruta/a/python bash ...` eliges el intérprete. Con `--full --reps 3` haces V1 con 3 repeticiones en vez de 5.

Si falla el paso 1 por la información de la máquina (la topología se lee con `sysctl` y no se ha probado en un Mac), el script avisa y sigue: la medida no depende de ella.

### 16.3 Comprobar que la lógica funciona

En `tests.log`, todos los tests deben pasar (28). Es la comprobación de la lógica en macOS: la selección por plataforma (`sys.platform == "darwin"`), el ajuste en las dos funciones de backtesting y la resolución por candidato en las búsquedas. Sin `--full`, no hay nada más que comprobar.

Con `--full`, además, en la tabla de `threshold.md` la columna `auto_jobs` dice cuántos jobs usó `'auto'` en cada caso:
- **Sin refit** (las dos primeras filas, `estimate` = 0): `auto_jobs` = 1, y `auto/seq` cerca de 1. Antes del umbral, estos casos eran entre 7 y 20 veces más lentos con `'auto'` (columna `par/seq`).
- **Con refit y trabajo estimado por encima del umbral provisional (4 s)**: `auto_jobs` = `min(cpu_count() - 1, número de folds)`, y `auto` cerca de `par`.
- **Por debajo del umbral**: `auto_jobs` = 1, y `auto` cerca de `seq`.
- **Direct**: `auto_jobs` = 1 siempre (paraleliza el `fit` de los pasos, no los folds), y `auto` cerca de `par`.

La última sección de `threshold.md` resume la comprobación: el peor caso de `'auto'` frente al mejor de `n_jobs=1` y `cpu-1`. En Windows y Linux ese peor caso está cerca del umbral, donde paralelizar o no cuesta casi lo mismo (sección 11.4). Un peor caso muy por encima de los de Windows y Linux, o un `auto_jobs` que no sigue las reglas anteriores, indica un problema: guarda la carpeta de resultados y avisa.

### 16.4 Fijar el umbral de macOS

La sección "Quick threshold" de `threshold.md` da el valor recomendado: 0,7 veces el arranque en frío de `cpu_count() − 1` workers con el forecaster (mediana de las repeticiones del bloque A), redondeado a 0,5 s. Es el valor `X` de abajo. Compáralo con los arranques conocidos: 5,68 s en Windows (umbral 4 s) y 3,17 s en WSL2 (2 s); lo esperable en un Mac es algo intermedio.

Si has ejecutado `--full`, la sección "Threshold" da además el valor del método completo, el que dio 4 s en Windows y 2 s en Linux:
- ajusta `paralelo = a + b × secuencial` sobre los casos con refit (sin Direct);
- el punto de equilibrio es `a / (1 − b)` segundos de trabajo real;
- se pasa a unidades de la estimación del código (`t_fit × refits`, alrededor del 80 % del real) y se redondea a 0,5 s.

Los dos valores deberían coincidir o diferir en 0,5 s; en ese caso, quédate con el del método completo. Comprueba también la horquilla que da al final ("parallel loses up to... and wins from..."): el valor debería quedar dentro de ella. Si no, o si el ajuste avisa de una pendiente rara, mira la tabla antes de fijar nada.

Con el valor `X`, en `skforecast/model_selection/_utils.py`:

```diff
-# workload) and from about 2.8 s on Linux (2 s). macOS has not been measured
-# yet and uses the value of Windows.
-_MIN_PARALLEL_WORKLOAD_SECONDS = {"win32": 4.0, "linux": 2.0}
+# workload), from about 2.8 s on Linux (2 s) and from about <T> s on macOS
+# (<X> s, <chip>).
+_MIN_PARALLEL_WORKLOAD_SECONDS = {"win32": 4.0, "linux": 2.0, "darwin": <X>}
 _MIN_PARALLEL_WORKLOAD_SECONDS_DEFAULT = 4.0
```

Y además:
- la docstring de `limit_n_jobs_by_workload`, que dice "4 seconds on Windows and macOS, 2 seconds on Linux": pon los valores de cada plataforma;
- el test `test_limit_n_jobs_by_workload_output_with_the_minimum_work_of_each_platform` (`skforecast/model_selection/tests/tests_utils/test_limit_n_jobs_by_workload.py`): los dos casos de `"darwin"` deben quedar a cada lado de `X` con 9 ajustes (por ejemplo, `X / 9 − 0,1` y `X / 9 + 0,1`);
- la nota de versión del umbral en `docs/releases/releases.md`, si nombra las plataformas.

Si `X` sale igual que en Windows, añade igualmente la clave `"darwin"`, para que el valor quede medido y no heredado, y cambia el comentario.

Después:
1. Opcional, si `X` difiere de los 4 s heredados: comprueba `'auto'` con el valor nuevo en los cuatro casos pequeños y uno grande de V1 (unos 5 minutos) y mira la última sección de `threshold.md` (sección 16.3):
   ```bash
   cd dev/n_jobs_review
   PYTHONPATH=../.. NJOBS_RESULTS_DIR=results/macos_<chip>_final python bench_cold.py --reps 3 --values 1,auto,cpu-1 --cases "lgbm_1|S,lgbm_1|L,hgb"
   python threshold.py --results-dir results/macos_<chip>_final
   ```
2. Pasa la skill `verify` y abre el PR contra la rama de la versión (`0.27.x`).
3. Añade al informe una sección como la 12 con las tablas de `threshold.md` y actualiza la sección 14. Copia `results/macos_<chip>/` a esta carpeta.

### 16.5 Qué dejar en el informe

- La máquina: chip, núcleos de rendimiento y de eficiencia, RAM, versión de macOS y de Python y librerías (está en `machine.json`).
- El arranque en frío de `cpu_count() − 1` workers, frente a 5,68 s en Windows y 3,17 s en WSL2, y el umbral elegido.
- Con `--full`, la tabla de V1 y el umbral del método completo, con su horquilla.
- Cualquier diferencia de comportamiento: por ejemplo, si LightGBM con sus propios hilos se comporta como en Linux (sección 12.4).
