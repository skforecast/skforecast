# Arnés de verificación de la migración a pandas 3

Scripts para comprobar, en cada PR de la migración (`../INFORME.md`, sección
7), que los resultados no cambian y que el rendimiento se mantiene, en pandas 2
y en pandas 3. Reconstruye y amplía el arnés del estudio (secciones 3.6 y 5),
que no se versionó.

| Fichero | Qué hace |
|---|---|
| `identity_cases.py` | Ejecuta los casos de identidad (los 21 del estudio más la ampliación de la sección 5) y escribe un JSON con las huellas de cada salida y las comprobaciones |
| `identity_compare.py` | Compara dos JSON de `identity_cases.py` |
| `bench_hotspots.py` | Mide en un proceso los benchmarks (los 20 del estudio y el `fit` de los cuatro forecasters de matriz numpy) |
| `run_ab.py` | Orquesta la comparación de una rama con su base en uno o varios entornos: identidad, tiempo y memoria; escribe `summary.md` |
| `warnrec.py` | Plugin de pytest que vuelca los avisos capturados a JSON (línea base de avisos) |
| `common.py` | Huellas, memoria compartida, captura de avisos y descripción del entorno |

Los resultados van a `results/` (ignorado por git) o al directorio que se
indique. No se versiona ninguna huella de referencia: las huellas son bit a
bit y dependen de la plataforma, así que cada comparación genera las de la
base y las de la rama en la misma máquina.

## Entornos

Clones del entorno de desarrollo que solo cambian pandas (`../INFORME.md`,
sección 10, "Decisiones y arranque del 2026-11-03"):

| Nombre | pandas | conda |
|---|---|---|
| `p2` | 2.3.3 | `skforecast_27_py13` |
| `p3` | 3.1.0 | `skforecast_27_py13_p3` |
| `p22` | 2.2.3 | `skforecast_27_py13_p22` |

Se llama siempre al `python.exe` de cada entorno: los lanzadores
`Scripts\*.exe` de los clones apuntan al intérprete del entorno original.
Antes de medir, el `pip freeze` de los entornos solo puede diferir en pandas.

## Uso habitual en una PR

Desde la raíz del repositorio, con la rama de la PR en el árbol de trabajo
(se incluyen los cambios sin commit):

```bash
python dev/pandas3_migration/harness/run_ab.py --identity --bench control --memory \
    --python p2=<ruta>/skforecast_27_py13/python.exe \
    --python p3=<ruta>/skforecast_27_py13_p3/python.exe
```

- `--base` es la referencia git de la base (por defecto `origin/0.27.x`); se
  exporta con `git archive` a `<out-dir>/base_tree`.
- `--bench` acepta `all`, `control` (matrices, `predict_loop`,
  `backtesting_recursive` y los cuatro `fit`; lo que pide la sección 7 para
  las PRs que no deberían cambiar el rendimiento) o una lista separada por
  comas.
- `--rounds 4 --warmup 2 --repeat 5` por defecto: 4 rondas de procesos
  alternos (A B B A ...), 20 muestras por árbol. El estadístico de
  referencia es el mínimo.
- `--index-unit ns` fuerza el índice de los datos de identidad a
  nanosegundos, para comparar pandas 2 con pandas 3 sin la diferencia de
  resolución (`[ns]` frente a `[us]`).

`summary.md` contiene:

- **Identidad, base frente a rama en cada entorno (modo estricto)**: tiene
  que dar `Result: OK`. Si la PR cambia una salida a propósito, la diferencia
  tiene que aparecer aquí y estar explicada en la PR.
- **Identidad, pandas 2 frente a pandas 3 con el código de la rama (modo
  normalizado, informativo)**: ningún valor puede diferir; las diferencias de
  dtype (`object` frente a `str`, `[ns]` frente a `[us]`) son las esperadas
  (`../INFORME.md`, sección 5).
- **Tiempo y memoria**: la tabla que se pega en la descripción de la PR,
  junto con las versiones de la cabecera.

## Qué comprueban las huellas

Cada salida se reduce a dos SHA-256:

- **Estricta**: valores, dtype de pandas de cada columna, tipo, dtype y
  `freq` de los índices.
- **Normalizada**: solo valores. Las fechas se pasan a nanosegundos enteros,
  el texto se compara por valor (sea `object` o `str`) y las categóricas por
  códigos y categorías.

Además, en cada caso:

- `inputs_modified`: entradas del usuario cuya huella estricta cambia
  durante el caso (nunca debe ocurrir).
- `repeat_mismatch`: métodos que dan resultados distintos al llamarlos dos
  veces seguidas con los mismos argumentos.
- `shared_memory`: pares salida y entrada cuyos arrays de numpy comparten
  memoria (los índices se ignoran: son inmutables). Entre base y rama, un par
  nuevo hace fallar la comparación estricta.
- `warnings`: avisos de deprecación atribuidos a ficheros de `skforecast/`.

`identity_cases.py` y `bench_hotspots.py` se relanzan con
`PYTHONHASHSEED=0` si no está fijado: `PopulationDriftDetector.predict`
construye su tabla chi2 a partir de un `set`, y sin semilla fija los últimos
bits de `chi2_statistic` cambian entre procesos.

## Uso directo de los scripts

```bash
# Identidad de un árbol (el de PYTHONPATH) y comparación
PYTHONPATH=<raíz del árbol> python identity_cases.py --out p2.json --label p2
PYTHONPATH=<raíz del árbol> <python de p3> identity_cases.py --out p3.json --index-unit ns
python identity_compare.py p2.json p3.json --mode normalized

# Benchmarks de un árbol
PYTHONPATH=<raíz del árbol> python bench_hotspots.py --out t.json --bench control

# Avisos de la suite de tests (línea base de la sección 2.4)
WARNREC_OUT=warnings.jsonl PYTHONPATH="<raíz>;<raíz>/dev/pandas3_migration/harness" \
    <python de p3> -m pytest skforecast -q -p warnrec
```

En Windows, el separador de `PYTHONPATH` es `;`.

## Diferencias con el arnés del estudio

- Los casos de identidad pasan de 21 a 30: diferenciación,
  `predict_quantiles`/`predict_dist` con residuos fuera de muestra,
  `encoding=None`, los dos detectores de drift, `select_features` y
  Sarimax, Ets y Arar en `ForecasterStats`. Las salidas incluyen los
  atributos que guarda `fit` (`exog_dtypes_in_`, `last_window_`, residuos,
  `training_range_`...).
- El A/B de rendimiento alterna procesos de dos copias del paquete (el método
  de la tabla 4.8), en lugar de parchear el código en el mismo proceso, que
  dio dos resultados falsos.
- La memoria pico se mide solo con `tracemalloc`. El pico de RSS del estudio
  (`/proc/self/clear_refs`) no existe en Windows, y en lugar de
  `os.getloadavg()` se registra el uso de CPU de la máquina antes de cada
  benchmark.
- Los datos de los benchmarks siguen los tamaños del anexo, con la misma
  forma pero no los mismos valores: las cifras absolutas no son comparables
  con las de la sección 4, solo los órdenes de magnitud y las razones.
