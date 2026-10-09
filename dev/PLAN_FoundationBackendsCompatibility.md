# Plan: workflow de compatibilidad de los backends de foundation models

## Contexto

Cada adapter de `skforecast/foundation/_adapters.py` envuelve una librería externa
(`chronos-forecasting`, `timesfm`, `uni2ts`, `tabicl`, `tabpfn-time-series`,
`tfc-t0`, `synthefy-nori`, `tsicl`) que publica versiones con frecuencia y cambia su
API sin aviso. Hoy no hay nada que detecte esas roturas:

- Los tests unitarios usan fakes que copian la API del backend en el momento de
  escribir el adapter (`FakeT0Forecaster.predict(self, context, horizon, quantiles,
  ...)` en `tests_foundation_models/fixtures_adapters.py:447`). Si el backend cambia,
  los tests siguen en verde.
- La user guide `foundation-forecasting-models.ipynb` solo se ejecuta en las releases,
  necesita varios entornos (Moirai no convive con el resto) y es lenta en CPU.

Caso real (2026-10-09): `tfc-t0` 0.4.0 (2026-09-15) renombró `context` a
`model_input` y 0.5.0 renombró `quantiles` a `quantile_levels`; `T0Adapter` de
skforecast 0.26.0 falla con `TypeError` desde entonces y nadie lo supo hasta ejecutar
la guía. Borrador de la issue en `dev/issue_t0_tfc_t0_compat.md`.

## Objetivo

Un único workflow de GitHub Actions que, para cada adapter y en su propio entorno,
compruebe que el adapter funciona con la versión del backend, y muestre el resultado
en una sola tabla por ejecución.

Decisiones ya tomadas:

- **Un solo workflow** (`.github/workflows/foundation-backends.yml`) con una
  **matriz**: un job por adapter (y variante), cada uno con su venv, más un job
  `summary` que agrega todo en una tabla en la página de la ejecución.
  Motivos frente a un único job en bucle: paralelismo, un fallo (OOM, cuelgue) no
  tumba al resto (`fail-fast: false`), se puede relanzar solo el job fallido y los
  logs quedan separados.
- **Dos niveles de test**: contrato (sin pesos, segundos) y smoke (pesos reales,
  minutos).
- **Dos variantes de entorno**: `latest` (última versión del backend, detecta
  roturas nuevas) y `pinned` (versiones conocidas como buenas, para saber si el fallo
  es nuestro o del backend).
- La user guide deja de ser el test; queda como documentación que se ejecuta en las
  releases con los mismos ficheros de entorno.

## Estructura de ficheros

```
tools/foundation_compat/
├── adapters.toml                    # Fuente única: adapter -> instalación, overrides, secretos
├── envs/
│   ├── chronos/requirements.txt     # Variante pinned (Dependabot los actualiza)
│   ├── timesfm25/requirements.txt
│   ├── ...
│   └── moirai/
│       ├── requirements.txt
│       └── overrides.txt            # scipy>=1.12 (uni2ts fija scipy<1.12)
├── run.py                           # Crea el venv, instala, lanza pytest y escribe results/<job>.json
└── summarize.py                     # Lee results/*.json y escribe la tabla Markdown

skforecast/foundation/tests/tests_backends/
├── conftest.py                      # Opción --backend <adapter>; sin ella todo se salta
├── contracts.py                     # BACKEND_CONTRACTS: qué importa y llama cada adapter
├── test_contract.py                 # Nivel 1
└── test_smoke.py                    # Nivel 2

.github/workflows/foundation-backends.yml
.github/dependabot.yml               # Nueva entrada para tools/foundation_compat/envs/*
```

Principio: **el workflow es fino**. Toda la lógica está en `run.py`, de modo que el
mismo comando reproduce en local exactamente lo que hace CI:

```bash
python tools/foundation_compat/run.py --adapter t0 --variant latest
python tools/foundation_compat/run.py --adapter moirai --variant pinned --level contract
```

## Paso 1: `adapters.toml` (fuente única)

Una entrada por adapter. La variante `latest` se instala a partir de
`backend_package` del adapter (no se duplica el nombre del paquete); la `pinned`, del
`requirements.txt` del adapter.

```toml
[chronos]
adapter = "ChronosAdapter"
model_id = "autogluon/chronos-2-small"

[moirai]
adapter = "MoiraiAdapter"
model_id = "Salesforce/moirai-2.0-R-small"
overrides = "envs/moirai/overrides.txt"   # uv pip install --override

[tabpfn]
adapter = "TabPFNAdapter"
model_id = "priorlabs/tabpfn-ts"
secrets = ["TABPFN_TOKEN"]                # Sin el secreto: resultado "skipped (no token)"
init_kwargs = { mode = "local" }

# timesfm25, timesfm3, tabicl, t0, nori, tsicl ...
```

`run.py` lee el adapter de `skforecast.foundation._adapters._ADAPTER_REGISTRY` para
obtener `backend_package`, así que si cambia en el código, el workflow lo sigue solo.

Un test unitario normal (en `tests_foundation_models/`, sin backends) comprueba que
**cada adapter registrado tiene entrada en `adapters.toml` y en `BACKEND_CONTRACTS`**.
Así un adapter nuevo no puede quedarse fuera del workflow.

## Paso 2: `run.py`

Para un `--adapter` y `--variant`:

1. `uv venv` en un directorio temporal con Python 3.12.
2. Instalar:
   - torch CPU: `uv pip install torch --torch-backend cpu` (evita las ruedas CUDA,
     varios GB). En `pinned`, la versión de torch va en el `requirements.txt`.
   - skforecast desde el checkout: `uv pip install -e ".[test]"`.
   - Backend: `latest` → `backend_package` (más `--upgrade-package`); `pinned` →
     `-r envs/<adapter>/requirements.txt`. Si hay `overrides`, `--override`.
3. Guardar en el JSON las versiones instaladas del backend, torch, numpy y scipy
   (`importlib.metadata`).
4. Si falta un secreto requerido: escribir `status = "skipped"` con el motivo y salir
   con código 0.
5. `pytest skforecast/foundation/tests/tests_backends --backend <adapter>
   --level <contract|smoke|all> --junitxml ...`.
6. Escribir `results/<adapter>-<variant>.json`: adapter, variante, versiones, estado
   de cada nivel (`passed`/`failed`/`skipped`), primera línea del error de cada test
   fallido y duración.
7. Código de salida: el de pytest (el job queda en rojo si falla).

Lecciones del entorno de hoy que conviene tener en cuenta:

- Si una instalación no resuelve (como `uni2ts` sin override), el JSON debe recoger
  `install: failed` y el mensaje de uv, no solo un job en rojo sin explicación.
- Fijar `HF_HUB_DISABLE_PROGRESS_BARS=1` y `TQDM_DISABLE=1` para que los logs se
  puedan leer.

## Paso 3: tests de contrato (nivel 1)

`contracts.py` describe, por adapter, cada símbolo del backend que usa el adapter y
los argumentos con nombre que le pasa. Se extrae de `_adapters.py` (estado en 0.26.0):

| Adapter | Import | Llamada y argumentos con nombre |
|---|---|---|
| Chronos | `chronos.BaseChronosPipeline` | `from_pretrained`; `predict_quantiles(inputs, prediction_length, quantile_levels, cross_learning)` |
| TimesFM 2.5 | `timesfm.TimesFM_2p5_200M_torch`, `timesfm.ForecastConfig` | `from_pretrained`; `forecast(horizon, inputs)`; `compile(...)`; `ForecastConfig(max_context, max_horizon)` |
| TimesFM 3.0 | `timesfm.TimesFM3Forecaster` | `from_pretrained(device)`; `predict_batch(contexts, horizon, past_only_covariates, past_future_covariates, return_quantiles, padding_mode)` |
| Moirai | `uni2ts.model.moirai2.Moirai2Module`, `Moirai2Forecast` | `from_pretrained`; `Moirai2Forecast(module, prediction_length, context_length, target_dim, feat_dynamic_real_dim, past_feat_dynamic_real_dim)`; `hparams_context(prediction_length)`; `predict` |
| TabICL | `tabicl.forecast.TabICLForecaster` | `__init__(max_context_length, temporal_features, point_estimate, tabicl_config)`; `predict_df(context_df, future_df, quantiles)` |
| TabPFN-TS | `tabpfn_time_series.TabPFNTSPipeline`, `TabPFNMode` | `__init__(max_context_length, tabpfn_mode, tabpfn_output_selection, tabpfn_model_config, temporal_features)`; `predict_df(context_df, future_df, quantiles)` |
| T0 | `t0.T0Forecaster` | `from_pretrained`; `predict(context, horizon, quantiles, future_covariates)` |
| Nori | `synthefy_nori.NoriRegressor` | `__init__(model, ...)`; `fit(X, y)`; `predict(X, output_type, quantiles)` |
| TS-ICL | `tsicl.TSICL` | `__init__(checkpoint_version, allow_auto_download)`; `forecast(inputs, prediction_length, quantile_levels, context_length, device, denormalize, squeeze_output)` |

`test_contract.py`, parametrizado sobre esa tabla:

- Importar el símbolo (falla si se movió o renombró).
- `inspect.signature(fn).bind_partial(**{k: None for k in kwargs})` para cada llamada:
  falla con `TypeError` si un argumento ya no existe. Es exactamente el fallo de T0.
- Si la firma tiene `**kwargs`, `bind_partial` siempre pasa: marcar ese caso en el
  resultado como "contrato débil" (lo cubre el smoke).
- Además, revisar la lista de la tabla contra el código antes de darlo por bueno
  (`grep -n "self._model\.\|self._pipeline\.\|from_pretrained" _adapters.py`).

No descarga pesos ni necesita secretos: corre en segundos.

## Paso 4: smoke tests (nivel 2)

`test_smoke.py`, un único fichero parametrizado por adapter (no un fichero por
adapter), usando solo la API pública (`FoundationModel` + `ForecasterFoundation` +
`backtesting_foundation`), para recorrer el mismo camino que el usuario:

- Datos sintéticos: 2 series horarias de ~300 observaciones (tendencia + estacionalidad
  diaria + ruido, semilla fija) y una exógena numérica.
- `context_length=64`, `steps=12`.
- Casos, activados según `get_model_info(model_id)` (`allow_exog`, etc.):
  - `fit` + `predict` de una serie: forma, índice esperado, valores finitos.
  - `predict_interval([0.1, 0.9])`: `lower_bound <= pred <= upper_bound`.
  - Multi-serie (`levels=None`).
  - Con `exog` si el adapter lo admite.
  - `backtesting_foundation` con 2 folds.
- No se comparan valores numéricos con referencias: cambian con cada versión y
  hardware. Solo invariantes.

Coste esperado: predicciones en CPU de 5 a 25 s por llamada en la ejecución de hoy
(4 núcleos), más la descarga de pesos. Con 2 folds, unos pocos minutos por adapter.
`timeout-minutes: 30` por job.

## Paso 5: el workflow

```yaml
name: Foundation backends compatibility

on:
  schedule:
    - cron: '0 6 * * 1'            # Lunes, antes de foundation-models-metadata
  workflow_dispatch:
    inputs:
      variant: { type: choice, options: [both, latest, pinned], default: both }
  pull_request:
    branches: [main, '*.x']
    paths:
      - 'skforecast/foundation/**'
      - 'tools/foundation_compat/**'
      - '.github/workflows/foundation-backends.yml'

permissions:
  contents: read

jobs:
  backend:
    name: ${{ matrix.adapter }} (${{ matrix.variant }})
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        adapter: [chronos, timesfm25, timesfm3, moirai, tabicl, tabpfn, t0, nori, tsicl]
        variant: [latest, pinned]
    steps:
      # checkout, setup-uv (Python 3.12), cache de ~/.cache/huggingface
      # (clave: adapter + model_id), y:
      - run: python tools/foundation_compat/run.py --adapter ${{ matrix.adapter }} --variant ${{ matrix.variant }}
        env:
          TABPFN_TOKEN: ${{ secrets.TABPFN_TOKEN }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
      - uses: actions/upload-artifact@...   # results/*.json, if: always()

  summary:
    needs: backend
    if: always()
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: write                          # Solo para el paso de la issue
    steps:
      # download-artifact (todos los results), y:
      - run: python tools/foundation_compat/summarize.py results >> "$GITHUB_STEP_SUMMARY"
      # Si event_name == 'schedule' y hay fallos: crear o actualizar la issue
```

Detalles:

- En PRs, ejecutar solo `pinned` (determinista: detecta regresiones nuestras); en el
  cron y en `workflow_dispatch`, las dos variantes. Se filtra la matriz con un
  `exclude`/`if` según `github.event_name`.
- Versiones de las actions: las mismas que ya usa `foundation-models-metadata.yml`
  (`actions/checkout@v7`, `astral-sh/setup-uv@v10.1.0`), `persist-credentials: false`.
- **Secretos**: `TABPFN_TOKEN` y `HF_TOKEN` como secretos del repo. Los PRs de
  Dependabot no ven los secretos normales: añadir `TABPFN_TOKEN` también como
  *Dependabot secret*. En PRs desde forks no hay secretos: TabPFN sale como
  "skipped (no token)", no como fallo.
- **Issue automática** (solo `schedule`): buscar una issue abierta con la etiqueta
  `backend-compat`; si existe, comentar la nueva tabla; si no, crearla. Cerrarla a mano
  cuando se arregle (o automáticamente cuando el cron salga verde, opcional).

## Paso 6: `summarize.py`

Lee `results/*.json` y escribe una tabla Markdown:

| Adapter | Variante | Backend | Versión | torch | Instalación | Contrato | Smoke | Detalle |
|---|---|---|---|---|---|---|---|---|
| T0 | latest | tfc-t0 | 0.5.0 | 2.9.1 | ✓ | ✗ | – | `predict()`: argumento `context` no existe |
| T0 | pinned | tfc-t0 | 0.3.2 | 2.9.1 | ✓ | ✓ | ✓ | |
| Moirai | latest | uni2ts | 2.0.0 | 2.4.1 | ✓ (override scipy) | ✓ | ✓ | |
| TabPFN-TS | latest | tabpfn-time-series | 1.3.0 | 2.9.1 | ✓ | ✓ | skipped | sin `TABPFN_TOKEN` |

Debe funcionar aunque falten JSON (job cancelado o caído): esa fila sale como
"sin resultado", con el enlace al job.

## Paso 7: variante `pinned` y Dependabot

Ficheros `tools/foundation_compat/envs/<adapter>/requirements.txt` con el backend y
torch fijados. Versiones conocidas como buenas de la ejecución de la guía del
2026-10-09 (skforecast 0.26.0, Python 3.12, CPU):

| Adapter | Fichero pinned |
|---|---|
| chronos | `chronos-forecasting==2.3.2`, `torch==2.9.1` |
| timesfm25, timesfm3 | `timesfm[torch]==3.0.2`, `torch==2.9.1` |
| moirai | `uni2ts==2.0.0`, `torch==2.4.1`, `numpy==1.26.4`; override `scipy==1.14.1` |
| tabicl | `tabicl[forecast]==2.2.0`, `torch==2.9.1` |
| tabpfn | `tabpfn-time-series==1.3.0`, `tabpfn==9.1.0`, `torch==2.9.1` (sin validar: no había token) |
| t0 | `tfc-t0==0.3.2`, `torch==2.9.1` |
| nori | `synthefy-nori==0.18.2`, `torch==2.9.1` |
| tsicl | `tsicl==0.2.1`, `torch==2.9.1` (`tsicl` limita torch a 2.9.x) |

Nueva entrada en `.github/dependabot.yml`:

```yaml
  - package-ecosystem: "pip"
    directories:
      - "/tools/foundation_compat/envs/*"
    schedule:
      interval: "weekly"
    labels: ["dependencies", "foundation"]
    commit-message:
      prefix: "dependabot"
    ignore:
      - dependency-name: "torch"   # torch lo movemos a mano (lo limitan los backends)
```

Cada versión nueva de un backend abre un PR que toca `tools/foundation_compat/**`, el
workflow corre y el PR muestra si la nueva versión es compatible antes de que la
usen los usuarios. Si pasa, se mergea y `pinned` avanza; si no, el PR documenta la
rotura (y se abre la issue).

## Fases (cada una se puede mergear por separado)

1. **Contrato + workflow + resumen, solo `latest`**: `adapters.toml`, `run.py`,
   `contracts.py`, `test_contract.py`, `summarize.py`, el workflow y el test de
   cobertura de adapters. Es lo más barato y ya habría detectado T0 el 15 de
   septiembre.
2. **Smoke tests**: `test_smoke.py`, caché de Hugging Face y secretos.
3. **Variante `pinned` + Dependabot + issue automática.**
4. Opcional: script para ejecutar la user guide por segmentos con estos mismos
   entornos en cada release (lo que se hizo a mano el 2026-10-09: ejecutar cada
   bloque de modelo en su venv, fusionar las salidas y renumerar `execution_count`).

## Verificación en local antes de subir

- Cada fase: `ruff check` de los ficheros nuevos y `pytest` de los tests unitarios
  nuevos (el de cobertura de adapters) con el entorno normal.
- `run.py` en local para al menos tres adapters representativos:
  - `--adapter t0 --variant latest` → contrato en rojo con el mensaje de `context`.
  - `--adapter t0 --variant pinned` → todo en verde.
  - `--adapter moirai --variant latest` → instalación con override y verde.
- `summarize.py` sobre esos JSON y revisar la tabla.
- `pytest skforecast/foundation/tests/tests_backends` **sin** `--backend` → todo
  `skipped` (no debe afectar a la suite normal ni descargar pesos).
- El workflow solo se puede probar de verdad en GitHub: lanzarlo con
  `workflow_dispatch` desde la rama antes de abrir el PR, o con
  [`act`](https://github.com/nektos/act) para la sintaxis.

## Fuera de alcance

- Arreglar `T0Adapter` para `tfc-t0>=0.4` (issue aparte, `dev/issue_t0_tfc_t0_compat.md`).
- Cambiar los fakes de los tests unitarios: siguen siendo útiles para la lógica del
  adapter; el contrato es lo que vigila la API del backend.
- GPU: todo en CPU. Los runners con GPU son de pago y no aportan para compatibilidad.
