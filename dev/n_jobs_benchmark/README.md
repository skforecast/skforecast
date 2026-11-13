# Benchmark de la estrategia de paralelización (`n_jobs`) de skforecast

Mide cuándo compensa repartir en procesos los folds de un backtesting o de una
búsqueda de hiperparámetros, y lo compara con lo que elige `n_jobs='auto'`.
Se centra en los regresores más usados: Ridge, LightGBM, XGBoost,
HistGradientBoosting, CatBoost y RandomForest.

Es un único script, `bench_n_jobs.py`, pensado para lanzarse en una máquina
Linux sin supervisión. No modifica nada fuera de esta carpeta.

## Requisitos

- Linux (usa la afinidad de CPU para emular máquinas más pequeñas).
- La máquina libre: nada más en ejecución durante las medidas.
- Acceso a internet la primera vez (paquetes y dos datasets de skforecast).

## Preparación

```bash
git clone --depth 1 -b 0.28.x https://github.com/skforecast/skforecast.git
```

```bash
cd skforecast
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
~/.local/bin/uv venv --python 3.12 ~/.venvs/skforecast-bench
```

```bash
~/.local/bin/uv pip install --python ~/.venvs/skforecast-bench/bin/python -e . lightgbm xgboost catboost
```

```bash
source ~/.venvs/skforecast-bench/bin/activate
```

```bash
cd dev/n_jobs_benchmark
```

Si falta `git`, instálalo antes (`sudo dnf install -y git` en Amazon Linux,
`sudo apt-get install -y git` en Ubuntu). Si LightGBM no encuentra
`libgomp.so.1`, instala `libgomp` (`sudo dnf install -y libgomp`, o
`sudo apt-get install -y libgomp1`).

## Ejecución

1. Comprobar el entorno (1 minuto). Descarga los datos y avisa de cualquier
   problema: una librería que no se importa, la máquina ocupada, etc.

   ```bash
   python bench_n_jobs.py check
   ```

2. Prueba rápida (unos 10 minutos). Recorre todos los caminos del script con
   tamaños mínimos; sus tiempos no significan nada. Debe terminar sin ninguna
   línea `ERROR`, `TIMEOUT` ni `WARNING`.

   ```bash
   python bench_n_jobs.py run --smoke
   ```

3. El benchmark. Para al llegar al presupuesto de tiempo (`--budget-minutes`,
   120 por defecto) y guarda cada medida al terminarla, así que se puede
   cortar y reanudar con el mismo comando: solo ejecuta lo pendiente.

   ```bash
   nohup python bench_n_jobs.py run --budget-minutes 170 > run.log 2>&1 &
   ```

   ```bash
   tail -f run.log
   ```

4. Resultados. Están en `results/<nombre de la máquina>/`: `results.jsonl`
   (todas las medidas), `machine.json` (máquina y versiones) y `summary.md`
   (tablas). El resumen se regenera con:

   ```bash
   python bench_n_jobs.py summary
   ```

   Para devolverlos, lo más simple es empaquetarlos y descargarlos:

   ```bash
   tar czf n_jobs_results.tgz results/ run.log
   ```

## Duración

Estimada a partir de una pasada en una máquina de 4 CPU; en otra máquina
puede variar un 25 %.

| Presupuesto | Qué incluye |
|---|---|
| 2 h | Etapas 1 y 2: los casos principales, una vez, en la máquina entera y en las de 8 y 4 CPU |
| 2 h 45 min | Además la etapa 3: los casos extendidos en la máquina entera |
| 4 h 45 min | Además las etapas 4 y 5: segunda repetición de los casos principales en los tres tamaños |
| 7 h | El plan completo |

Con una sola repetición, las diferencias de menos de un 15 % entre dos
configuraciones deben tratarse como un empate.

## Cómo se mide

**Una medida cada vez.** El script lanza una medida, espera a que termine y
solo entonces lanza la siguiente. Un bloqueo impide dos ejecuciones
simultáneas del benchmark en la misma carpeta. Antes de cada medida se
registra la CPU ocupada por otros procesos, y se avisa si supera el 5 %.

**En frío y en caliente.** Cada medida se ejecuta en un proceso de Python
nuevo, que llama a la operación varias veces:

- La primera llamada (`first`) es en frío: en un proceso nuevo no existe
  ningún worker de joblib, así que tiene que arrancarlos, como un script.
- Las siguientes (`next`) son en caliente: van seguidas en el mismo proceso y
  reutilizan los workers, como las celdas de un notebook.

No se da por supuesto: antes y después de cada llamada se guarda el estado del
ejecutor de joblib (su identificador y los procesos vivos). El resumen indica
si alguna primera llamada encontró workers ya arrancados, o si alguna llamada
siguiente no reutilizó exactamente los mismos. Al terminar cada medida se mata
todo su grupo de procesos, para que no quede ningún worker.

**Tamaños de máquina.** Todos los CPU, y 8 y 4 CPU emulados fijando la
afinidad del proceso antes de importar ninguna librería (un hilo por núcleo
físico mientras quedan núcleos). joblib, OpenMP, LightGBM y CatBoost ven
entonces una máquina de ese tamaño. En el bloque `start` se comprueba además
desde dentro de los workers: que ven los CPU de su máquina y los hilos que
joblib les asigna (`cpu_count() // n_jobs`).

**Configuraciones de `n_jobs`.** `1`, `max` (`cpu_count() - 1`, lo que usa
`'auto'` cuando paraleliza) y `auto`. En algunos casos, también `half` (la
mitad de los CPU) y `all`: con refit, para ver cuál es el mejor número de
procesos; sin refit, para ver si menos procesos, que arrancan antes, conservan
la ganancia.

## Qué mide

| Bloque | Casos | Qué regla evalúa |
|---|---|---|
| `start`, `resize` | Arranque de k procesos; cambio de su número entre llamadas | Umbral de carga de Linux; `min(n_jobs, n_tasks)` |
| `ladder_*` | Backtesting con refit de LightGBM con un hilo, con 5 tamaños de serie (de 200 a 800 filas) | Umbral de carga (`_MIN_PARALLEL_WORKLOAD_SECONDS`) |
| `bt_norefit` | Backtesting sin refit, 100 folds | Regla "sin refit, un proceso" |
| `bt_refit`, `bt_refit_small` | Backtesting con refit, 16 y 6 folds | Regla por estimador; estimación de la carga; menos folds que procesos |
| `gs_norefit`, `gs_refit_small` | Búsqueda en rejilla de 3 candidatos | Excepciones de HistGB y CatBoost; búsquedas sin umbral |
| `predict*` | Predicción por número de filas e hilos, con 100, 500 y 1.000 árboles | Límite de un hilo de HistGB; hilos de XGBoost, LightGBM y CatBoost |
| `ms_*` | Backtesting multiserie, 50 series | Las mismas reglas en `ForecasterRecursiveMultiSeries` |

**Estimadores.** Con los parámetros por defecto de cada librería, salvo el
número de árboles: 500 en LightGBM, XGBoost, HistGradientBoosting y CatBoost
(`--trees`), y 100 en RandomForest (`--rf-trees`), que es su valor por defecto:
con sus árboles sin límite de profundidad, cada ajuste tarda unos 12 segundos
y el modelo ocupa 40 MB. Por ese coste, RandomForest con un hilo no entra en
los casos con más ajustes (`bt_refit` y `gs_refit_small`); `bt_refit_small` lo
cubre.

Además de los seis estimadores con sus hilos por defecto, se miden variantes
de un hilo (`lgbm_1t`, `xgb_1t`, `catboost_1t`) y de todos los hilos
(`rf_all`), para comparar los hilos del estimador con los procesos de los
folds.

**Datos.** `bike_sharing_extended_features` (horario, 5.112 filas de
entrenamiento, 8 exógenas, 48 lags y una media móvil de 72 horas) y
`store_sales` (50 series diarias).

**`ForecasterDirect`** queda fuera del plan por defecto: con 500 árboles,
ajustar sus pasos en paralelo compensa siempre. Se incluye con `--direct`.

Además de los tiempos, cada medida guarda:

- el `n_jobs` que llega a `joblib.Parallel`, y lo que recibe y devuelve
  `limit_n_jobs_by_workload`;
- la CPU consumida en cada llamada por toda la máquina (procesos e hilos), para
  distinguir una opción más rápida de una que solo gasta más;
- la duración de los ajustes del proceso principal, y un ajuste suelto al final
  (para ver si una ejecución en paralelo ralentiza los ajustes posteriores);
- el tamaño del forecaster serializado, y la memoria del proceso y de sus
  workers (al final y su pico);
- una firma de las predicciones, para comprobar que no cambian.

`machine.json` guarda la máquina (CPU, núcleos, memoria), el commit de
skforecast y la lista completa de paquetes del entorno con sus versiones.

## Orden y presupuesto

Las medidas se ordenan por etapas, de modo que una ejecución cortada conserve
la cobertura más amplia. Dentro de cada etapa el orden es aleatorio (semilla
fija).

| Etapa | Contenido |
|---|---|
| 1 | Casos principales en la máquina entera, repetición 1 |
| 2 | Casos principales en 8 y 4 CPU, repetición 1 |
| 3 | Casos extendidos en la máquina entera (variantes de hilos, multiserie, búsqueda con refit, número de procesos) |
| 4 y 5 | Repetición 2 de los casos principales, en la máquina entera y en 8 y 4 CPU |
| 6 | Casos extendidos en 8 y 4 CPU |
| 7 | Repetición 3 de los casos principales en la máquina entera |

`python bench_n_jobs.py run --list` muestra el plan sin ejecutar nada.

Opciones útiles: `--cpus 16 4` (tamaños de máquina), `--reps N`, `--trees 300`,
`--rf-trees 500`, `--size L` (10.776 filas de entrenamiento),
`--blocks bt_refit ladder` (solo esos casos), `--estimators xgb hgb`,
`--max-priority 2` (sin los casos extendidos). Con otro `--trees`, `--rf-trees`
o `--size`, los resultados van a otra carpeta.

## Límites

- Solo Linux: el coste de arrancar los procesos en Windows y macOS es distinto,
  y sus umbrales no se pueden calibrar aquí.
- Las máquinas emuladas comparten la CPU, la memoria y la caché de la máquina
  real: emulan el número de núcleos, no otro hardware.
- Un único tamaño de datos por defecto. Las decisiones dependen del trabajo
  por fold, que los casos cubren con varias cargas, no del dataset concreto.
- No se miden los intervalos de predicción, los modelos estadísticos ni el
  refit intermitente (que siempre usa un proceso, por corrección).
