# dev/foundation_adapters_review/evidence

Material de la revisión (2026-11-06) de la alineación del código de 9 adaptadores foundation de skforecast con la última versión de sus backends y checkpoints. Base: rama `docs/foundation-adapters-review` desde `origin/0.27.x` (skforecast 0.27.0 en desarrollo), commit `d9b8f4f`. Solo diagnóstico: nada fuera de `dev/foundation_adapters_review/` se ha modificado. Los planes de implementación están en la carpeta superior (`../README.md`); las rutas relativas de este fichero parten de `evidence/`. Timestral y TimeH2O quedan fuera. No contiene medidas de velocidad ni comparaciones de precisión entre versiones o variantes.

Todos los comandos se ejecutan desde la raíz del repositorio.

## Estructura

```
dev/foundation_adapters_review/
├── README.md                       índice de los planes de implementación
├── 01_t0.md ... 10_docs_ci.md      un plan por pull request
└── evidence/
    ├── README.md                   este fichero
    ├── diagnosis_report.md         diagnóstico completo y verificado
    ├── cross_cutting.md            revisión transversal
    ├── common/                     utilidades compartidas
    │   ├── make_env.sh             crea ~/.venvs/fm-review-<key> y escribe evidence/<key>/requirements-freeze.txt
    │   ├── pypi_info.py            historial de versiones de un paquete en PyPI
    │   ├── hub_info.py             listado del Hub (sha, fechas, licencia, gated, tamaño)
    │   └── backtest_guide.py       backtesting de la tabla comparativa de la guía de usuario
    ├── cross_cutting/              sondas del código común de skforecast/foundation
    │   ├── probe_shared.py
    │   └── probe_shared.log
    └── <key>/                      una carpeta por adaptador
        ├── env.log, requirements-freeze.txt   creación del entorno y versiones exactas
        ├── ci_baseline*.log        extracto del job del CI (run 42212742430, 2026-11-06)
        ├── contract.log, smoke.log, unit_tests.log
        ├── compare_direct.py/.log  adaptador frente a llamada directa al backend
        ├── *.py / *.log            comprobaciones específicas (cada script imprime sus versiones)
        └── verification/       segunda pasada independiente: verification.md, scripts y logs
```

Claves (`<key>`, iguales a las entradas de `tools/foundation_compat/adapters.toml`): `t0`, `tsicl`, `timesfm25`, `timesfm3`, `chronos`, `moirai`, `tabpfn`, `tabicl`, `nori`.

Los `__pycache__/` de `common/` y `t0/` no forman parte del material. Los logs (`*.log`) se añaden con `git add -f`.

## Reglas de ejecución

- Los entornos viven fuera del repositorio, en `~/.venvs/fm-review-<key>`; nunca se usa el python del entorno principal.
- Hugging Face: el token, si hace falta, lo inyecta el proxy de red. No definir `HF_TOKEN` ni llamar a `login()`.
- TabPFN: ejecutar con `TABPFN_DISABLE_TELEMETRY=1 TABPFN_NO_BROWSER=1`. Sin `TABPFN_TOKEN` los pesos de TabPFN-3.5 no se descargan y el smoke real falla con `TabPFNLicenseError`.
- Código upstream: clones git en `~/fm-review-src/<repo>` (fuera del repositorio), solo para lectura.
- Comprobar el disco (`df -h ~`) antes de descargar checkpoints grandes.
- `common/backtest_guide.py` solo se usa para medir antes y después de un cambio de código que pueda alterar predicciones.

## Versiones usadas (de cada `<key>/requirements-freeze.txt`)

Todos los entornos: Python 3.12.3, skforecast 0.27.0 editable, Linux x86_64, 4 CPU (Intel Xeon 2.10 GHz), 15 GB de RAM, sin GPU.

| key | Backend | torch | numpy | pandas | huggingface-hub | Otros |
|---|---|---|---|---|---|---|
| t0 | tfc-t0 0.6.0 | 2.15.0+cpu | 2.5.4 | 2.3.3 | 2.4.0 | scikit-learn 1.9.1, scipy 1.18.1 |
| t0 (aux. `-032`) | tfc-t0 0.3.2 | 2.15.0+cpu | 2.5.4 | 2.3.3 | 2.4.0 | `t0/requirements-freeze-032.txt` |
| tsicl | tsicl 0.2.1 | 2.9.1+cpu | 2.5.4 | 2.3.3 | 2.4.0 | scikit-learn 1.9.1 |
| timesfm25 | timesfm 3.0.2 | 2.15.0+cpu | 2.5.4 | 2.3.3 | 2.4.0 | scikit-learn 1.9.1 |
| timesfm3 | timesfm 3.0.2 | 2.15.0+cpu | 2.5.4 | 2.3.3 | 2.4.0 | scikit-learn 1.9.1 |
| chronos | chronos-forecasting 2.3.2 | 2.15.0+cpu | 2.5.4 | 2.3.3 | 2.4.0 | transformers 5.22.1, accelerate 1.15.0 |
| moirai | uni2ts 2.0.0 | 2.4.1+cpu | 1.26.4 | 2.3.3 | 2.4.0 | scipy 1.17.1 (override), gluonts 0.14.3 |
| tabpfn | tabpfn-time-series 1.4.0, tabpfn 9.2.0 | 2.15.0+cpu | 2.5.4 | 2.3.3 | 1.33.0 | tabpfn-client 0.7.1, gluonts 0.17.0, scikit-learn 1.9.0 |
| tabicl | tabicl 2.2.0 | 2.15.0+cpu | 2.5.4 | 2.3.3 | 2.4.0 | scikit-learn 1.9.1 |
| nori | synthefy-nori 0.23.0 | 2.13.0+cpu | 2.5.4 | 2.3.3 | 1.33.0 | scikit-learn 1.9.1, scipy 1.18.1 |

## Recrear los entornos

Uso general: `bash dev/foundation_adapters_review/evidence/common/make_env.sh <key> "<requisito del backend>" <python> [requisitos extra]`.

```bash
# T0: con tfc-t0 0.6.0 el paquete base no trae torch; se añadió el runtime después
bash dev/foundation_adapters_review/evidence/common/make_env.sh t0 "tfc-t0" 3.12
uv pip install --python ~/.venvs/fm-review-t0/bin/python --torch-backend cpu \
    torch einops rotary-embedding-torch safetensors
# T0 de referencia (0.3.2)
uv venv ~/.venvs/fm-review-t0-032 --python 3.12
uv pip install --python ~/.venvs/fm-review-t0-032/bin/python --torch-backend cpu \
    -e . pytest "tfc-t0==0.3.2"

bash dev/foundation_adapters_review/evidence/common/make_env.sh tsicl "tsicl" 3.12
bash dev/foundation_adapters_review/evidence/common/make_env.sh timesfm25 "timesfm[torch]" 3.12
bash dev/foundation_adapters_review/evidence/common/make_env.sh timesfm3 "timesfm[torch]" 3.12
bash dev/foundation_adapters_review/evidence/common/make_env.sh chronos "chronos-forecasting" 3.12
UV_OVERRIDE=dev/foundation_adapters_review/evidence/moirai/overrides.txt \
    bash dev/foundation_adapters_review/evidence/common/make_env.sh moirai "uni2ts" 3.12
bash dev/foundation_adapters_review/evidence/common/make_env.sh tabpfn "tabpfn-time-series>=1.3" 3.12
bash dev/foundation_adapters_review/evidence/common/make_env.sh tabicl "tabicl[forecast]" 3.12
bash dev/foundation_adapters_review/evidence/common/make_env.sh nori "synthefy-nori" 3.12
```

Entornos auxiliares para versiones antiguas (mismo patrón; `<v>` es la versión):

```bash
uv venv ~/.venvs/<nombre> --python 3.12
uv pip install --python ~/.venvs/<nombre>/bin/python --torch-backend cpu -e . pytest "<paquete>==<v>"
```

| Entorno | Paquetes | Para |
|---|---|---|
| `fm-review-timesfm3-v300` | `timesfm[torch]==3.0.0` | mensaje de import (TFM3-F3) |
| `fm-review-timesfm25-oldhf` | `timesfm[torch]==2.0.0/2.0.1/2.0.2/3.0.2` con `huggingface_hub==0.36.0` | workaround y cuantiles autorregresivos (T25-F5, F6) |
| `fm-review-chronos-old` | `chronos-forecasting==2.1.0` y `==2.2.0` | mínimo real (CHR-F3) |
| `fm-review-tabicl-<v>` | `tabicl[forecast]==2.0.2` y `==2.0.3` | mínimo real (TAB-F2) |
| `fm-review-tabpfn-130` | `tabpfn-time-series==1.3.0` (tabpfn 9.0.0) | mínimo declarado (TPF-F7) |
| `fm-review-nori-old` | `synthefy-nori==0.21.1` y `==0.18.2` | Nori-1-5 y NaN en X (NORI-F4, F11) |

## Volver a ejecutar las comprobaciones

Con `PY=~/.venvs/fm-review-<key>/bin/python` y `KEY=<key>`:

```bash
# Contrato y smoke de tools/foundation_compat
$PY -m pytest tools/foundation_compat/tests/test_contract.py --backend $KEY -q -p no:cacheprovider
$PY -m pytest tools/foundation_compat/tests/test_smoke.py --backend $KEY -q -p no:cacheprovider
# Opcional: flujo completo del CI (crea su propio entorno temporal)
python tools/foundation_compat/run.py --adapter $KEY
# Tests unitarios del adaptador (nombres de fichero abajo)
$PY -m pytest skforecast/foundation/tests/tests_foundation_models/test_<Adapter>.py -q -p no:cacheprovider
# PyPI y Hub
$PY dev/foundation_adapters_review/evidence/common/pypi_info.py <paquete>
$PY dev/foundation_adapters_review/evidence/common/hub_info.py <autor> --prefix <prefijo>
```

Ficheros de tests unitarios: `test_T0Adapter.py`, `test_TSICLAdapter.py`, `test_TimesFM25Adapter.py`, `test_TimesFM3Adapter.py`, `test_ChronosAdapter.py`, `test_MoiraiAdapter.py`, `test_TabPFNAdapter.py`, `test_TabICLAdapter.py`, `test_NoriAdapter.py`.

Scripts por adaptador (todos con `~/.venvs/fm-review-<key>/bin/python <ruta>` salvo que se indique):

- **t0**: `t0/compare_direct.py` (también con `fm-review-t0-032`), `t0/options_060.py`, `t0/registry_and_loading.py`, `t0/check_load_errors.py`, `t0/check_config_ignored.py`, `t0/adapter_no_torch.py`, `t0/backtest_t0.py` (antes y después de `trim_leading_missing`, usa `t0/t0_shim.py`); verificación: `t0/verification/verify_t0.py`, `no_torch_import.py`, `config_fields.py`, `load_errors.py`, `past_covariates.py`, `hetero_trim.py`.
- **tsicl**: `tsicl/compare_direct.py`, `check_covariates_rollout.py`, `check_padding_option.py`, `check_quantile_grid.py`, `check_registry_options.py`; verificación: `tsicl/verification/verify_backend_direct.py`, `check_quantile_exact_types.py`.
- **timesfm25**: `timesfm25/compare_direct.py`, `registry_and_loading.py`, `hub_checks.py`, `check_import_errors.py`, `check_flip_ar_quantiles.py` (también con `fm-review-timesfm25-oldhf`), `check_workaround.py` (con `fm-review-timesfm25-oldhf`); verificación: `timesfm25/verification/verify_direct.py`, `verify_registry_checkpoints.py`, `verify_batch_size.py`, `verify_all_nan_context.py`, `verify_all_nan_old_numpy.py <ruta a timesfm_2p5_base.py>`.
- **timesfm3**: `timesfm3/compare_direct.py`, `batch_invariance.py`, `check_import_errors.py` y `check_import_fix.py` (también con `fm-review-timesfm3-v300`); verificación: `timesfm3/verification/check_p1_probe.py` y `check_guide_fold0.py`.
- **chronos**: `chronos/compare_direct.py [model_id ...]`, `capabilities_check.py`, `all_nan_context.py`, `signature_check.py`, `old_backend_check.py` (con `fm-review-chronos-old`), `backtest_patch.py` (backtest de la guía con el parche de CHR-P4); verificación: `chronos/verification/single_series_cross_learning.py`.
- **moirai**: `moirai/compare_direct.py`, `covariates_probe.py`, `set_params_reload.py`, `misc_checks.py`, `all_nan_context.py`; verificación: `moirai/verification/extra_checks.py`.
- **tabpfn** (con `TABPFN_DISABLE_TELEMETRY=1 TABPFN_NO_BROWSER=1`): `tabpfn/compare_direct.py`, `capabilities_check.py`, `heterogeneous_check.py`, `probe_int_timestamps.py`; verificación: `tabpfn/verification/feature_level_check.py`, `nontrivial_check.py`, `real_weights_v2_check.py` (checkpoint público `Prior-Labs/TabPFN-v2-reg`, sin token), `categorical_v2_check.py`, `test_p1_pytest_warns.py` (con `-m pytest`).
- **tabicl**: `tabicl/compare_direct.py`, `capabilities_check.py`, `resolve_adapter.py`, `stderr_redirect_check.py <True|False>`, `proto_fixes.py`; verificación: `tabicl/verification/verify_tabicl.py`.
- **nori**: `nori/compare_direct.py`, `edge_cases.py [model_id]`, `patch_keep_nan_covariates.py nan-check` y `patch_keep_nan_covariates.py guide --variant current|patched --out ...`; verificación: `nori/verification/verify_nori.py`, `dataframe_numeric_check.py`, `old_backend_check.py` y `old_backend_nan_x.py` (con el entorno antiguo).
- **transversal**: `~/.venvs/fm-review-chronos/bin/python dev/foundation_adapters_review/evidence/cross_cutting/probe_shared.py`.

Backtest de la guía (solo antes y después de un cambio que altere predicciones):

```bash
~/.venvs/fm-review-<key>/bin/python dev/foundation_adapters_review/evidence/common/backtest_guide.py \
    --model-id <model_id> --exog --out dev/foundation_adapters_review/evidence/<key>/<nombre>.csv
```

## Notas

- Algunos `env.log` contienen la ruta del intérprete del sistema (`/usr/bin/python3.12`) tal como la imprime `uv`.
- Los entornos auxiliares marcados como temporales se borraron por falta de disco; se recrean con el patrón anterior.
- Los ficheros `upstream_*.log` (issues, cuerpos de issues, diffs y títulos de commits) reproducen el texto upstream tal cual, incluidos guiones largos (U+2013, U+2014) que no se han sustituido para no alterar la evidencia.
