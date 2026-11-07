# Verificación adversarial: TimesFM3Adapter (timesfm 3.0.2)

Fecha: 2026-11-06. Entorno: ~/.venvs/fm-review-timesfm3 (Python 3.12.3, timesfm 3.0.2,
torch 2.15.0+cpu, numpy 2.5.4, pandas 2.3.3, huggingface_hub 2.4.0, skforecast 0.27.0
editable, git d9b8f4f) y ~/.venvs/fm-review-timesfm3-v300 (timesfm 3.0.0) solo para el
mensaje de import. Código upstream en ~/fm-review-src/timesfm (tags v3.0.0, v3.0.2,
origin/master e51928e, `git fetch` hecho hoy: sin commits nuevos).
Sin mediciones de velocidad ni comparaciones de precisión entre versiones o variantes.

## Lo que se ha re-ejecutado

| Script / consulta | Log | Resultado |
|---|---|---|
| `compare_direct.py` del investigador | `compare_direct_rerun.log` | Idéntico al log original (maxdiff 0.0 adaptador vs `predict_batch`; secciones A-H iguales) |
| `batch_invariance.py` del investigador | `batch_invariance_rerun.log` | Idéntico (0.20 vía `levels`, 0.32 con 100 vs 2048) |
| `check_import_errors.py` (3.0.2 y 3.0.0) | salida en consola, coincide con los logs originales | 3.0.2 da el mensaje genérico; 3.0.0 da "torch is missing" |
| `check_p1_probe.py` (nuevo) | `check_p1_probe.log` | La sonda propuesta recupera la causa en 3.0.2 y 3.0.0; simulación de PR #552 |
| `check_guide_fold0.py` (nuevo) | `check_guide_fold0.log` | MAE del CSV = 162.4352 = guía; pliegue 0 reproducido exacto (0.0) |
| contrato + smoke (`--backend timesfm3`) | `contract_smoke_verify.log` | 9 passed |
| tests unitarios `test_TimesFM3Adapter.py` | `unit_tests_verify.log` | 57 passed |
| Hub `hub_info.py google --search timesfm` | `hub_google_timesfm_verify.log` | Solo `google/timesfm-3.0-pytorch` (sha 43046b85, gated false, license other) |
| Commits del repo del Hub y blob de `model.safetensors` | `hub_commits_verify.log` | Mismo LFS sha256 (a7592b0a8432...) en los 10 commits desde 2026-08-24 |
| Issues/PR upstream desde 2026-09-05 | `upstream_recent_verify.log` | #546, #515, #552 abiertos; #549 abierto (use_sdpa) |
| Log CI job 127875169049 (API de GitHub Actions) | (no guardado) | contract 3 passed, smoke 6 passed, status passed |

## Veredictos

- **TFM3-F1** confirmado [probado]. PyPI: 3.0.0 (08-28), 3.0.1 (09-02), 3.0.2 (09-09), nada más.
  Nota: 3.0.1 no tiene tag git; la wheel 3.0.1 no contiene la corrección #483 (solo 3.0.2).
- **TFM3-F2** confirmado [verificado en código + probado]. Firmas iguales; #483 (commit 0ac3663)
  y #492 (8680259) entran en 3.0.2. `TimesFM3Torch.forward` no pasa `decode_cache`.
  Matiz: una llamada directa con las covariables en otro orden difiere ~1e-3 (sobre ~5600):
  el adaptador ordena las columnas (`get_exog_signature`); con el mismo orden la igualdad es exacta.
- **TFM3-F3** confirmado [probado]. `timesfm3/__init__.py` en v3.0.2 usa `__getattr__` perezoso.
  Sobre #552 (abierto, no fusionado): es cierto que `hasattr` no captura el ImportError, pero lo que
  se propagaría es el mensaje accionable de upstream, no un fallo opaco.
- **TFM3-F4** corregido (menor) [verificado en código]. Todo cierto salvo "TimesFM3Evaluator impone
  'none'": su docstring dice "Enforces", pero el código solo usa 'none' como valor por defecto y
  reenvía `padding_mode` a `super().predict_batch` (evaluator.py v3.0.2 líneas 58, 97, 195, 257).
  Los ejemplos del README no pasan `padding_mode`, así que usan 'none'.
- **TFM3-F5** confirmado [probado]. La frase solo está en `references/adapter-parameters.md` (línea 82).
- **TFM3-F6** confirmado [probado].
- **TFM3-F7** confirmado [probado]. LICENSE sha256 3e36db72...; README upstream (líneas 34-36 de la
  sección "License notice") confirma Apache-2.0 para el código y BigQuery ML para uso comercial.
- **TFM3-F8** confirmado [probado; #515/#546 documentado]. Placeholders `np.zeros_like` sin máscara
  verificados en v3.0.2.
- **TFM3-F9** confirmado [verificado en código]. MPS/CUDA no comprobables aquí.
- **TFM3-F10** corregido [verificado en código]. `use_sdpa` de `_ModelConfig` no se lee nunca en
  `timesfm3_forecaster.py` v3.0.2 (solo la definición en la línea 86); al cargar del Hub la
  arquitectura sale de config.json. Es justo lo que pide arreglar el PR abierto #549. Reenviarlo
  no tendría efecto. `force_download` sí se respeta y faltaba en la lista. Otros campos de
  arquitectura (`use_variate_attention`, etc.) se sobrescriben con config.json (issue #502/#505).
- **TFM3-F11** confirmado [verificado en código + probado].
- **TFM3-F12** confirmado [probado]: MAE recalculado del CSV y pliegues 0 y 29 reproducidos
  exactamente; no se relanzó el backtesting completo. `global_context` = 15360 y truncado en
  `_Query.format`/`batch_context` verificado en código.

## Propuestas

- **TFM3-P1** confirmada. La sonda propuesta funciona en 3.0.0 y 3.0.2 (`check_p1_probe.log`).
  Riesgo bajo y no rompe nada. Al implementarla hay que mantener la comprobación de versión
  (<3) antes de la sonda.
- **TFM3-P2** confirmada; needs_decision. Matiz para (c): el código pasa `padding_mode` explícitamente junto con
  `**self.predict_kwargs`, así que habría que sacarlo de `predict_kwargs` (pop/get) para evitar un
  TypeError por argumento duplicado. Corregir también en la propuesta lo de "impone" (ver F4).
- **TFM3-P3** corregida. La afirmación `(predict(levels=[x]) equals predict() filtered to x)` solo
  está en `skills/foundation-forecasting/references/adapter-parameters.md`; las Notes del
  docstring solo dicen que la predicción no depende de las covariables de otras series, que es
  cierto. En el docstring se trataría de añadir el matiz, no de quitar una frase.
- **TFM3-P4** confirmada (recomendación (a)). Matiz: `per_core_batch_size` cambia las
  predicciones multiserie (F5, compare_direct sección C), así que no es solo una opción de carga.
  No incluir `use_sdpa` (no tiene efecto en 3.0.2).
- **TFM3-P5** confirmada. Riesgo nulo.

## Resumen

La fila resumen es correcta: alineado, prioridad media. No hay versión nueva, checkpoint nuevo
ni cambio de licencia. Los cambios a hacer son la sonda de import (P1), el docstring de
`padding_mode` (P2) y textos de docs (P3, P5).
