# Estudio de velocidad: skforecast 0.26.0 (PyPI) frente a 0.27 (`origin/0.27.x`)

## Objetivo

Medir cuánto cambia la velocidad de un usuario que pasa de skforecast 0.26.0 a 0.27
en el flujo de trabajo del artículo
[py29, Forecasting energy demand with machine learning](https://cienciadedatos.net/documentos/py29-forecasting-electricity-power-demand-python)
(dataset `vic_electricity` horario con sus variables exógenas), con seis estimadores:
`Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor` (scikit-learn),
`LGBMRegressor`, `XGBRegressor` y `CatBoostRegressor`.

La diferencia principal entre versiones es la selección automática de `n_jobs`
(`n_jobs='auto'`), pero 0.27 trae otros cambios que afectan a la velocidad. El estudio
debe separar ambos efectos y decir, para cada estimador y flujo, si 0.27 es más
rápida, igual o más lenta, por qué, y con qué incertidumbre. El resultado es un
informe en español con números reproducibles, no un cambio de código.

## Antes de empezar: lee esto (en este orden)

1. `docs/releases/releases.md`, sección 0.27.0, entradas **Changed** y **Fixed** que
   hablan de velocidad. Las relevantes aquí:
   - `n_jobs='auto'` en `backtesting_forecaster`: secuencial sin refit, o cuando el
     trabajo estimado con el ajuste inicial cronometrado es menor que el umbral
     (4 s en Windows, 2 s en Linux y macOS): `limit_n_jobs_by_workload`.
   - Búsquedas de hiperparámetros sin refit: secuenciales con `'auto'`.
   - `ForecasterDirect` refit en paralelo ya no envía los estimadores anteriores
     (#1460).
   - `LGBMRegressor`: la predicción respeta los hilos del estimador (#1464).
   - `HistGradientBoostingRegressor`: predice con 1 hilo hasta 100 filas.
   - Ajuste rápido de modelos lineales solo con lags (#1469): **comprueba si se
     activa aquí** (con exógenas, probablemente no).
   - Avisos de los workers reenviados al proceso principal (#1467): posible coste.
2. Código de la regla en cada versión:
   - 0.27: `skforecast/model_selection/_utils.py` (`select_n_jobs_backtesting`,
     `limit_n_jobs_by_workload`) y sus llamadas en `_validation.py` y `_search.py`;
     `skforecast/utils/utils.py` (`select_n_jobs_fit_forecaster`,
     `_build_predict_function`).
   - 0.26: `git show v0.26.0:skforecast/model_selection/_utils.py` (y los mismos
     ficheros con `git show v0.26.0:...`).
3. Contexto previo (solo secciones 1 y 2 de cada uno, no repitas esos estudios):
   `dev/n_jobs_review_v2/informe.md` y `dev/n_jobs_review/informe_n_jobs.md`, y la FAQ
   `docs/faq/parallelization-skforecast.ipynb` (no la leas entera con Read: es grande;
   extrae las celdas Markdown con un script). Hechos medidos que condicionan el diseño:
   arrancar 21 workers de loky que importan skforecast cuesta ~5.7 s en esta máquina
   Windows; CatBoost (`thread_count=-1`), LightGBM y `n_jobs=-1` anidado no respetan el
   límite de hilos de loky dentro de los workers (sobresuscripción); XGBoost se
   ralentiza en el mismo proceso tras una ejecución paralela (segundo equipo de
   libgomp).

Trata esos hechos como contexto, no como resultados esperados: mide y deja que los
números digan lo que pasa.

## Precondiciones (compruébalas y para si alguna falla)

- `origin/0.27.x` contiene el trabajo de la rama `perf/n-jobs-auto-work-threshold`
  (busca `def limit_n_jobs_by_workload` y `_MIN_PARALLEL_WORKLOAD_SECONDS` en
  `git show origin/0.27.x:skforecast/model_selection/_utils.py`, y el límite de 1 hilo
  de HGB en `_build_predict_function`). Haz `git fetch` primero. Si no está, para y
  pregunta al usuario qué referencia usar. Anota el hash exacto medido.
- Entorno: `C:\Users\Joaquin\miniconda3\envs\skforecast_27_py13\python.exe` con
  `PYTHONIOENCODING=utf-8`. Nunca `conda run` (falla con unicode en Windows). Las dos
  versiones se miden **en este mismo entorno**, con las mismas versiones de numpy,
  pandas, scikit-learn, lightgbm, xgboost, catboost, joblib y optuna; lo único que
  cambia es el código de skforecast.
- `astral` y `feature-engine` no están instalados (el artículo los usa para las
  variables de luz solar y las ventanas de temperatura). Pregunta al usuario si se
  pueden instalar en el entorno; antes, enséñale el resultado de
  `pip install --dry-run astral feature-engine` para confirmar que no cambia ningún
  paquete núcleo. Si dice que no, calcula las ventanas de temperatura con pandas
  (equivalente a `WindowFeatures`: `rolling(ventana).agg(f).shift(1)`, mismos nombres
  `Temperature_window_1D_mean`...) y omite las de luz solar; dilo en el informe.
- La máquina debe estar tranquila: comprueba la carga de CPU antes de la ejecución
  completa (y regístrala durante, si `psutil` está disponible) y pide al usuario que
  no use el equipo durante las mediciones largas.

## Montaje de las dos versiones (sin tocar el site-packages)

El entorno tiene un skforecast no editable en site-packages: ignóralo y selecciona la
versión con la variable de entorno `PYTHONPATH` (no con `sys.path` en el código: los
workers de loky son procesos nuevos y solo heredan el entorno).

- 0.26.0: `python -m pip install --no-deps --target <dir>/skf_026 skforecast==0.26.0`.
- 0.27: `git archive <hash de origin/0.27.x> skforecast | tar -x -C <dir>/skf_027`
  (export limpio, no el checkout de trabajo).
- Comprueba en cada versión, **en el proceso principal y dentro de un worker de
  joblib** (`Parallel(n_jobs=2)(delayed(f)() for _ in range(2))`), que
  `skforecast.__version__` y `skforecast.__file__` apuntan a la copia correcta.
- Haz una ejecución de calentamiento por versión (descartada) para que se creen
  `__pycache__` y las cachés de numba antes de medir.

## Datos (una sola vez, compartidos por ambas versiones)

Prepara los datos una vez, con la versión 0.27, y guárdalos en parquet (`pyarrow`
está instalado): así las dos versiones reciben exactamente la misma entrada y el
tiempo de preparación no entra en las medidas. Reproduce el artículo:

```python
data = fetch_dataset(name='vic_electricity', raw=True)
data['Time'] = pd.to_datetime(data['Time'], utc=True)
data = data.set_index('Time').tz_convert('Australia/Melbourne').sort_index().asfreq('30min')
data = data.drop(columns='Date')
data = (
    data.resample(rule='h', closed='left', label='right')
    .agg({'Demand': 'mean', 'Temperature': 'mean', 'Holiday': 'mean'})
)
data = data.loc[:'2014-12-31 23:00:00', :].copy()
end_train = '2013-12-31 23:59:00'
end_validation = '2014-09-30 23:59:00'
```

Exógenas del artículo:

- `CalendarFeatures(features=['month', 'week', 'day_of_week', 'hour'], encoding='cyclical', keep_original_columns=False)` de `skforecast.preprocessing`.
- Luz solar con `astral` (`LocationInfo(latitude=-37.8, longitude=144.95, timezone='Australia/Melbourne')`, amanecer y atardecer por día redondeados a la hora): `sunrise_hour_sin/cos`, `sunset_hour_sin/cos`, `daylight_hours`, `is_daylight`.
- Festivos: `Holiday` como int, `holiday_previous_day = shift(24)`, `holiday_next_day = shift(-24)`.
- `WindowFeatures(variables=['Temperature'], window=['1D', '7D'], functions=['mean', 'max', 'min'], freq='h')` (feature_engine).
- Concatenar, quitar las primeras `7 * 24` filas y las últimas 24.
- Interacciones: `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).set_output(transform='pandas')` sobre las 24 columnas anteriores (las 22 nuevas más `Temperature` y `Holiday`), quitar las columnas originales, prefijo `poly_` y espacios por `__`.
- Selección: columnas que acaban en `_sin`/`_cos`, las que empiezan por `Temperature_` y por `holiday_`, más `Temperature`, `Holiday`, `daylight_hours`, `is_daylight` (en el artículo son 90 exógenas).
- `data = data[['Demand']].merge(exog, left_index=True, right_index=True, how='inner').astype('float32')`.

Si hace falta ver el artículo completo: la herramienta de fetch devuelve una página
vacía; `curl -sL -A "Mozilla/5.0" <url>` descarga el HTML (8 MB) y el código está en
los bloques `<pre>` (extráelos con un script, `html.unescape` y quitando etiquetas).
Guarda la descarga en su propia carpeta y procésala con `python -I`.

Comprueba y anota: filas (~26 000), número de exógenas, ausencia de NaN (Ridge no los
admite) y particiones (train ~17 500, validación ~6 550, test ~2 200).

## Diseño del experimento

### Estimadores (mismos hiperparámetros en las dos versiones)

Configuración de hilos **por defecto** de cada librería, como la usaría un usuario.
Tamaños orientativos; ajústalos en el piloto (ver abajo) y anota los finales:

| Clave | Estimador |
|---|---|
| `ridge` | `Ridge(alpha=1.0)` |
| `rf` | `RandomForestRegressor(n_estimators=100, max_depth=10, max_features=0.3, random_state=123)` |
| `hgb` | `HistGradientBoostingRegressor(max_iter=200, random_state=123)` |
| `lgbm` | `LGBMRegressor(n_estimators=200, random_state=123, verbose=-1)` |
| `xgb` | `XGBRegressor(n_estimators=200, max_depth=6, random_state=123)` |
| `catboost` | `CatBoostRegressor(iterations=200, depth=6, random_seed=123, verbose=0, allow_writing_files=False)` |

Forecaster base (artículo): `lags=[1, 2, 3, 23, 24, 25, 47, 48, 49, 167, 168, 169]`
(o el que elijas, igual para todos), `window_features=RollingFeatures(stats=['mean'], window_sizes=24 * 3)`, todas las exógenas.

### Flujos (adaptados del artículo)

| Clave | Flujo | Por qué |
|---|---|---|
| `fit_predict` | `ForecasterRecursive.fit` hasta `end_validation` + `predict(steps=24)` | Sin paralelismo: aísla los cambios no relacionados con `n_jobs`. |
| `bt_norefit` | `backtesting_forecaster`, `TimeSeriesFold(steps=24, initial_train_size=len(data.loc[:end_validation]), refit=False)` (92 folds, el patrón del artículo) | 0.26 reparte las predicciones en workers; 0.27 va secuencial. |
| `bt_norefit_interval` | Igual, con `interval=[0.05, 0.95]`, `interval_method='conformal'`, `use_binned_residuals=True` (residuos in-sample) | Intervalos del artículo. |
| `bt_refit` | `TimeSeriesFold(steps=24 * 7, refit=True, fixed_train_size=False)` sobre test (~14 folds) | Caso en que paralelizar sí puede pagar; prueba el umbral de trabajo. |
| `direct_bt` | `ForecasterDirect(steps=24, n_jobs='auto' / 1)` + `bt_norefit` | Paralelismo en el `fit` de Direct (#1460). |
| `bayes_norefit` | `bayesian_search_forecaster` como el artículo: `TimeSeriesFold(steps=24, initial_train_size=len(data.loc[:end_train]), refit=False)` sobre `:end_validation` (~273 folds), `lags` en el `lags_grid` del artículo, espacio de búsqueda propio de cada estimador, `n_trials` reducido (p. ej. 10) | Búsquedas sin refit: 0.26 paraleliza cada candidato, 0.27 no. |
| `grid_refit` | `grid_search_forecaster` con 4 candidatos (2 lags x 2 valores de un hiperparámetro) y `TimeSeriesFold(steps=24 * 7, refit=True)` sobre validación | Búsquedas con refit (0.27 resuelve `'auto'` por candidato, sin umbral). |

Opcional, solo si sobra tiempo: búsqueda con `OneStepAheadFold` como control (ignora
`n_jobs`, debería empatar) y variante de estimadores de un solo hilo
(`n_jobs=1`/`thread_count=1`) para separar la sobresuscripción.

Las búsquedas deben evaluar **los mismos candidatos en las dos versiones**: semilla
fija del sampler (`random_state` de la búsqueda, comprueba que existe con el mismo
significado en 0.26.0) y verifica que las listas de parámetros evaluados coinciden.

### Factores y medida

- Versión: `0.26.0`, `0.27`.
- `n_jobs`: `'auto'` (lo que ve el usuario, el resultado principal) y `1` (control:
  la diferencia entre versiones con `n_jobs=1` son los cambios que no son de
  paralelismo).
- Cada medida en un **proceso nuevo** (subprocess) que carga el parquet, construye el
  caso y llama al flujo **dos veces**: la primera es "fría" (incluye el arranque de
  los workers, como un script) y la segunda "caliente" (workers reutilizados, como
  re-ejecutar una celda de un notebook). Cronometra solo la llamada
  (`time.perf_counter`), no los imports ni la carga de datos.
- Repeticiones: al menos 3 por celda (5 si la celda dura menos de 10 s), con el orden
  de versión y `n_jobs` barajado dentro de cada bloque de repeticiones. Nunca dos
  medidas a la vez: los subprocesos van de uno en uno.
- Tiempo máximo por subproceso (p. ej. 15 min): un timeout se registra como tal y es
  un resultado, no un error que haya que esquivar.
- Resultados en un CSV que se escribe tras cada medida, para poder reanudar (saltar
  celdas ya hechas) si algo se corta.
- Pasada aparte, sin cronometrar: registra el `n_jobs` que llega realmente a
  `joblib.Parallel` en cada celda y versión (envuelve `joblib.Parallel.__init__` en el
  subproceso; skforecast importa la clase con `from joblib import Parallel`).

### Corrección

Para cada celda, compara las predicciones (`pred`) de 0.26 y 0.27, y de `'auto'` frente
a `1` dentro de cada versión: diferencia máxima absoluta. En las búsquedas, compara el
ranking y la métrica de cada candidato. 0.27 añade columnas a la salida (`step`,
`y_true`, `n_evaluated`): compara solo las comunes. Cualquier diferencia mayor que el
ruido de coma flotante es un hallazgo para el informe, no algo que arreglar.

### Piloto y presupuesto

1. Piloto: una repetición de cada celda con 0.27 y `n_jobs=1` (y la llamada caliente)
   para conocer el tiempo secuencial. Ajusta los tamaños (folds, `n_trials`,
   `n_estimators`) para que cada celda secuencial dure entre ~5 y ~90 s; conserva a
   propósito celdas a ambos lados del umbral de 4 s en `bt_refit`. No reduzcas los
   flujos sin refit del artículo por debajo de lo realista: son el caso de uso.
2. Enseña al usuario la tabla de tamaños finales y la duración estimada de la
   ejecución completa (objetivo: menos de 3 h) y **espera su aprobación** antes de
   lanzarla.
3. Lánzala en segundo plano y no la consultes en bucle; cuando termine, revisa
   timeouts, valores atípicos y la carga de la máquina antes de analizar.

## Entregables

Todo en `dev/benchmark_026_vs_027/` (scripts, `results/*.csv`, informe), sin
commits ni pushes: lo deja todo en el árbol de trabajo y el usuario decide.

`informe.md` en español:

1. Resumen ejecutivo: veredicto en 5-8 líneas, con los factores de aceleración (o
   pérdida) más importantes y su rango.
2. Máquina (CPU, procesadores lógicos, RAM, Windows, plan de energía), versiones
   (Python, numpy, pandas, scikit-learn, scipy, joblib, threadpoolctl, optuna,
   lightgbm, xgboost, catboost) y hash exacto de 0.27.
3. Método (lo de arriba, breve) y tamaños finales de cada flujo.
4. Tabla principal por flujo y estimador: tiempo 0.26 `'auto'`, 0.27 `'auto'`,
   aceleración (0.26/0.27) con mediana y rango, frío y caliente, y el `n_jobs`
   resuelto por cada versión. Tabla de control con `n_jobs=1`.
5. Un gráfico de las aceleraciones (estimador x flujo); usa la skill `dataviz` antes
   de escribir el código del gráfico.
6. Explicación de cada diferencia relevante, atribuida al cambio concreto de 0.27
   (con su entrada de las release notes) y, si alguna no tiene explicación, dilo.
7. Casos en que 0.27 es igual o más lenta, y por qué.
8. Corrección: diferencias numéricas entre versiones.
9. Limitaciones (una sola máquina Windows de 22 procesadores lógicos, umbral distinto
   en Linux y macOS, hilos por defecto de los estimadores...).
10. Un párrafo de 3-4 frases con cifras defendibles, apto para las release notes o un
    post (sin editar `releases.md`).
11. Reproducción: comandos exactos.

## Reglas

- No modifiques nada en `skforecast/` ni en `docs/`. Si encuentras un bug o una mejora
  posible, descríbela en el informe con la evidencia.
- No lances tests de la suite ni uses `pytest -n`; este estudio no los necesita.
- Mide tiempos solo en `skforecast_27_py13`.
- Sin `git commit`, `git push` ni PRs.
- Ante una decisión que cambie el diseño (tamaños, flujos, instalar paquetes, duración
  larga), pregunta antes.
