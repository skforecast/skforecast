# Estudio de migración a pandas 3: cómo reproducir cada número

El informe es `INFORME.md`. Esta carpeta no toca la librería: los cambios
propuestos se prueban por monkeypatch (`prototypes/`, ver su README).

## Entornos

- pandas 2 (`p2`): el entorno de trabajo (`/root/.venvs/skforecast` en la
  sesión cloud; en local, el conda `skforecast_24_py13`).
- pandas 3 (`p3`): un entorno aparte con las mismas versiones de todo lo demás
  y pandas 3.1.0. En la sesión cloud se creó así (`uv` en `~/.local/bin`):

```bash
uv pip freeze --python /root/.venvs/skforecast/bin/python \
    | grep -v -iE "skforecast|^-e|^pandas==" > /tmp/p3_requirements.txt
echo "pandas>=3.1" >> /tmp/p3_requirements.txt
uv venv ~/.venvs/skforecast-p3 --python 3.12
uv pip install --python ~/.venvs/skforecast-p3/bin/python -r /tmp/p3_requirements.txt
```

skforecast no se instala en `p3`: todos los scripts se ejecutan con
`PYTHONPATH=<raíz del repo>`. `run_in_both.sh` lo hace por ti; las rutas de los
intérpretes se cambian con las variables `SKF_P2` y `SKF_P3`.

## Ficheros

| Fichero | Qué hace | Resultado |
|---|---|---|
| `common.py` | Protocolo de medida compartido: `ab_interleaved` (A/B intercalado, mediana y mínimo), `peak_memory` (tracemalloc y VmHWM), `fingerprint` (huellas SHA-256 estricta y normalizada), `env_info` | - |
| `run_in_both.sh <script> [args]` | Ejecuta un script en `p2` y luego en `p3`, guardando la salida en `results/<script>_pandas<versión>.txt` | - |
| `identity_cases.py` | 21 casos (todos los forecasters, exógenas numéricas/categóricas/texto, dict y formato largo, window y calendar features, intervalos, backtesting, búsqueda, joblib y skops) reducidos a huellas | `results/identity_pandas<versión>[_<tag>].json` |
| `identity_compare.py A B` | Compara dos ejecuciones de `identity_cases.py` | `results/identity_*_vs_*.md` |
| `identity_crossload.py --other <versión>` | Carga en la versión actual los forecasters guardados por la otra (joblib y skops) y compara predicciones | `results/identity_crossload_*.json` |
| `bench_hotspots.py` | 17 benchmarks de los puntos calientes (matrices de entrenamiento de cada forecaster, bloque F de multiseries, `_get_level_row_slices`, `predict` en bucle, `expand_index`/`check_predict_input`, folds del backtesting, window features), con `--patch <prototipo>` para el A/B | `results/bench_pandas<versión>[_<prototipo>].json` |
| `bench_report.py` | Tabla Markdown a partir de los JSON de `bench_hotspots.py` | `results/bench_table.md` |
| `experiments/exp_cow_basics.py` | Semántica de copias de pandas 2 y 3 (constructores, `to_numpy` de solo lectura, `concat`, `insert`, `df[lista]`) | `results/exp_cow_basics_pandas*.txt` |
| `experiments/exp_datetime_resolution.py` | Qué construcciones dan `datetime64[us]` en pandas 3 | `results/exp_datetime_resolution_pandas*.txt` |
| `experiments/exp_mixed_resolution.py` | `fit` con `[ns]` y `predict` con `[us]` (y al revés), `expand_index`, `TimeSeriesFold` con fechas | `results/exp_mixed_resolution_pandas*.txt` |
| `experiments/exp_pickle_compat.py save|load` | Compatibilidad de pickles de pandas entre versiones | `results/exp_pickle_compat_pandas*.txt`, `results/pickles/` (no versionado) |
| `prototypes/` | Monkeypatches de los cambios propuestos (`apply()`/`restore()`), `pandas_compat_draft.py` (borrador del módulo de compatibilidad) | - |
| `results/baseline_pandas3.1.0/` | Línea base: log completo de pytest, fallos agrupados, resumen de avisos, plugin `warnrec.py`, log de los 90 tests fallidos ejecutados con pandas 2 | - |

## Línea base (una sola vez, autorizada por el encargo)

```bash
cd <raíz del repo>
mkdir -p /tmp/baseline && cp dev/pandas3_migration/results/baseline_pandas3.1.0/warnrec.py /tmp/baseline/
PYTHONPATH=$PWD:/tmp/baseline WARNREC_OUT=/tmp/baseline/warnings.jsonl \
  ~/.venvs/skforecast-p3/bin/python -m pytest skforecast -q -p warnrec \
  --continue-on-collection-errors --ignore=skforecast/deep_learning -rfE \
  --junitxml=/tmp/baseline/junit.xml --durations=40 -o cache_dir=/tmp/baseline/.pytest_cache \
  > /tmp/baseline/pytest_full.log 2>&1
```

Los 90 fallos relacionados con pandas se volvieron a ejecutar con pandas 2
(todos pasan) con `xargs -d '\n' -a <lista de node ids> python -m pytest -q`
(los node ids llevan espacios en los parámetros).

## Identidad

```bash
/root/.venvs/skforecast/bin/python dev/pandas3_migration/identity_cases.py
PYTHONPATH=. /root/.venvs/skforecast-p3/bin/python dev/pandas3_migration/identity_cases.py
PYTHONPATH=. /root/.venvs/skforecast-p3/bin/python dev/pandas3_migration/identity_cases.py --index-unit ns --tag ns
python dev/pandas3_migration/identity_compare.py results/identity_pandas2.3.3.json results/identity_pandas3.1.0_ns.json
# con un prototipo aplicado desde el arranque
... identity_cases.py --patch no_copy_keyword --tag no_copy_keyword
# persistencia cruzada (después de las dos primeras ejecuciones)
/root/.venvs/skforecast/bin/python dev/pandas3_migration/identity_crossload.py --other 3.1.0
PYTHONPATH=. /root/.venvs/skforecast-p3/bin/python dev/pandas3_migration/identity_crossload.py --other 2.3.3
```

## Rendimiento

Un proceso cada vez, máquina en reposo (sin agentes ni tests en segundo
plano). Siete repeticiones intercaladas tras dos de calentamiento; memoria
pico en una pasada aparte (tres repeticiones, mínimo).

```bash
for py in /root/.venvs/skforecast/bin/python /root/.venvs/skforecast-p3/bin/python; do
  PYTHONPATH=. $py dev/pandas3_migration/bench_hotspots.py --repeats 7
  PYTHONPATH=. $py dev/pandas3_migration/bench_hotspots.py --repeats 7 --patch no_copy_keyword
  PYTHONPATH=. $py dev/pandas3_migration/bench_hotspots.py --repeats 7 --patch dataframe_copy_false
done
python dev/pandas3_migration/bench_report.py
```
