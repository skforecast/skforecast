# Verificación adversarial de la revisión de MoiraiAdapter

Fecha: 2026-11-06. Rama `docs/foundation-adapters-review` (desde `origin/0.27.x`, skforecast 0.27.0).
Entorno reutilizado: `~/.venvs/fm-review-moirai` (Python 3.12.3, uni2ts 2.0.0, torch 2.4.1+cpu,
numpy 1.26.4, scipy 1.17.1, pandas 2.3.3). CPU x86_64, 4 núcleos, sin GPU. Código upstream:
`~/fm-review-src/uni2ts` (tag 2.0.0 = 8062ef5, main = cfd46d4).

No se ha ejecutado ningún benchmark de velocidad ni de precisión, ni `backtest_guide.py`
(ninguna propuesta cambia las predicciones: MOI-P2 da diferencia 0).

## Qué se ha re-ejecutado

| Fichero | Contenido |
|---|---|
| `rerun_compare_direct.log` | `compare_direct.py` del investigador: adapter vs llamada directa (diff 0 en todos los casos), rejilla de cuantiles, NaN, contextos largos, enrutado de ids |
| `rerun_covariates_probe.log` | `feat_dynamic_real` da RuntimeError en `moirai2/module.py:144`; `past_feat_dynamic_real` corre y cambia la salida 2.1e-2 |
| `rerun_set_params_reload.log` | 2 llamadas a `from_pretrained`, diff 0 entre módulo recargado y reutilizado, módulo inyectado descartado |
| `rerun_misc_checks.log` | mensaje de ImportError sin `uni2ts.model.moirai2`; ubicación del aviso MPS (línea 90 vía FoundationModel, 71 vía ForecasterFoundation) |
| `rerun_all_nan_context.log` | ventana de contexto toda NaN: cuantiles ~0 sin aviso |
| `rerun_tests.log` | contrato 5 passed, smoke 5 passed 1 skipped, unitarios 36 passed, device_handling -k moirai 12 passed |
| `resolve_check.log` | `uv pip compile`: 3.12 -> uni2ts 2.0.0; 3.13 -> uni2ts 1.1.1 y torch 2.15.0+cpu; `uni2ts>=2.0` en 3.13 no tiene solución; skforecast + uni2ts sin override no tiene solución (scipy); con override scipy 1.17.1, pandas 2.3.3, gluonts 0.14.3 |
| `pip_dry_run.log` | pip 26.3, Python 3.12, pandas 2.3.3 y scipy 1.17.1 preinstalados: `pip install --dry-run uni2ts` instalaría scipy 1.11.4, pandas 2.1.4, gluonts 0.14.4 |
| `extra_checks.py` / `extra_checks.log` | contexto 12000 (> 512 parches x 16 = 8192): finito y diff 0 con la llamada directa; ubicación del LicenseWarning |

Comprobaciones adicionales hechas a mano (sin log propio):

- Wheel de PyPI uni2ts 2.0.0 descomprimido frente a `git archive 2.0.0 src/uni2ts`: idénticos (`diff -rq`).
- `git diff --stat 2.0.0 origin/main`: 37 ficheros, todos fuera de `src/` y `pyproject.toml` (project/moirai-agent, etc.).
- PyPI: últimas versiones 1.1.0, 1.1.1, 1.2.0, 2.0.0 (2025-11-04). 1.1.1 exige `torch>=2.1` sin techo; 1.2.0 y 2.0.0 `torch>=2.1,<2.5`; todas `scipy~=1.11.3` y `numpy~=1.26.0`.
- `moirai2` ausente en los tags 1.1.0 y 1.2.0 y en el wheel 1.1.1.
- gluonts 0.14.4 exige `pandas<2.2.0`; 0.14.3 `pandas<3`.
- Hub: `moirai-2.0-R-small` sha 30f43ff, licencia cc-by-nc-4.0, no gated, sin LICENSE; pesos en f2ed354 (2025-08-06), commits posteriores solo README. `-base` y `-large` dan 404. `config.json`: `max_seq_len=512`, `patch_size=16`, `quantile_levels` 0.1..0.9.
- Issues upstream (search/issues): #216 y #256 (PR) abiertos; #251, #224 y #133 (issues) abiertos.
- CI de skforecast, job 127875169007 (run 42212742430): uni2ts 2.0.0, torch 2.4.1+cpu, scipy 1.17.1, override `scipy>=1.12`, contrato y smoke en verde.
- Código upstream (tag 2.0.0): `forecast.py` predict (imputación, recorte/relleno con `past_is_pad`, float32), `hparams_context` acepta `context_length`; `module.py` usa `PackedStdScaler` (que hace `target.double()`, `packed_scaler.py:46`) y `RotaryProjection` con `max_len=max_seq_len`, que se amplía en `forward` vía `_init_freq(seq_id.max() + 1)`; la salida se ordena por `module.quantile_levels`.

## Veredictos (resumen)

- MOI-F1, F2, F3, F4, F7, F8, F9, F10, F11, F12, F13, F15: confirmados.
- MOI-F5: corregido (matiz): el registro acepta cualquier id que empiece por `Salesforce/moirai-2`, no solo `moirai-2.0-R-*`.
- MOI-F6: corregido (matiz): no todos los `Salesforce/moirai*` públicos son cc-by-nc-4.0: `moirai-1.5-llama-kvcache` no declara licencia y es gated manual. No afecta al adapter (no se enruta).
- MOI-F14: confirmado, con un matiz para la tabla: el checkpoint declara `max_seq_len=512` parches (8192 observaciones); más allá el RoPE se amplía sin error (probado con 12000), pero si el modelo se entrenó con ese tope no es verificable (paper bloqueado).
- MOI-P1: confirmado, con dos matices: el mensaje de error y los comandos de instalación deben poner el especificador entre comillas (`pip install "uni2ts>=2.0"`; sin comillas el shell redirige a un fichero `=2.0`), como ya hace TabPFNAdapter; y el test unitario existente sigue valiendo.
- MOI-P2: confirmado; añadir que hay que actualizar `test_MoiraiAdapter_set_params_updates_and_resets_module`, que hoy afirma que `context_length` y `device` resetean `_module`.
- MOI-P3, P4, P5, P6, P7: confirmados.

## Hallazgos que faltaban

- MOI-V1 [probado]: el `LicenseWarning` de Moirai apunta a `skforecast/foundation/_adapters.py:2069` (`self._load_module()` dentro de `_ensure_forecast_obj`), no a la línea del usuario: `_warn_if_non_commercial` usa `stacklevel=3` fijo. Visible también en el log del CI (job 127875169007). Cosmético, de la misma familia que MOI-F13; el helper es común, conviene revisarlo para todos los adapters a la vez.
- MOI-V2 [verificado en código + probado]: para la corrección de la tabla de la guía (MOI-F14/P4), el dato del checkpoint es `max_seq_len=512` parches x `patch_size=16` = 8192 observaciones; el backend no falla por encima (12000 probado, diff 0 con la llamada directa). Propuesta: "sin tope duro en uni2ts (el checkpoint declara 8192)" en lugar de solo "sin límite".
