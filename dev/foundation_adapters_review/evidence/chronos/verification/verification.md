# Verificación adversarial: ChronosAdapter (chronos-forecasting 2.3.2)

Fecha: 2026-11-06. Entornos: ~/.venvs/fm-review-chronos (Python 3.12.3, chronos-forecasting 2.3.2, torch 2.15.0+cpu, transformers 5.22.1, accelerate 1.15.0, numpy 2.5.4, pandas 2.3.3) y ~/.venvs/fm-review-chronos-old (chronos-forecasting 2.1.0, transformers 4.57.6; se instaló 2.2.0 temporalmente y se dejó de nuevo en 2.1.0). Código upstream: clon en ~/fm-review-src/chronos-forecasting (main 1224faf, tags hasta v2.3.2). CPU, sin GPU. No se hicieron medidas de tiempo ni comparaciones de precisión entre versiones o variantes.

## Qué se volvió a ejecutar

| Comprobación | Log | Resultado |
|---|---|---|
| compare_direct.py con chronos-2-small y chronos-2-synth | compare_direct_rerun.log | A, B (cross_learning False/True), H (bf16): diff 0. C: Categorical int -> int32, diff 0 frente a numérica y 0.348 frente a categórica. D: TypeError de FoundationModel. E: recorte exacto, solo log del backend. F: UserWarning >1024. G: 0.540 entre batch_size 256 y 2 |
| all_nan_context.py | all_nan_context_rerun.log | q0.1 hasta -94680, q0.9 hasta 86309, sin aviso, por predict y por fit+predict |
| old_backend_check.py con 2.1.0 y 2.2.0 | old_backend_2.1.0_rerun.log, old_backend_2.2.0_rerun.log | 2.1.0: TypeError cross_learning; 2.2.0: OK |
| capabilities_check.py | capabilities_check_rerun.log | atributos veraces; esquema heterogéneo -> ValueError; past-only diff 0 |
| contrato + smoke + unit tests | contract_smoke_rerun.log, unit_tests_rerun.log | 9 passed; 55 passed |
| Hub (sha, licencias, gated, blobs por revisión) | hub_recheck.log, configs_recheck.log | igual que el investigador |
| backtest_guide antes/después de CHR-P4 | backtest_before_rerun.log, backtest_after_rerun.log | MAE 171.267 en ambos, diff máxima 0.0 |
| cross_learning con una serie; cross_learning en predict_kwargs | single_series_cross_learning.py/.log | diff 0.0; TypeError "multiple values" |
| condición de CHR-P3 | contract_p3_check.log | 2.3.2 declara cross_learning en predict; 2.1.0 no |

CI: el log del job 127875169074 (run 42212742430) leído con la API de GitHub Actions confirma chronos-forecasting 2.3.2, torch 2.15.0+cpu, transformers 5.22.1, contrato 3 passed, smoke 6 passed.

## Veredictos (resumen)

- CHR-F1 a CHR-F11, CHR-F13 a CHR-F15: confirmados.
- CHR-F5: confirmado; el atributo se llama `supports_categorical_covariates`, no `supports_categorical_features`.
- CHR-F6: confirmado; matiz: un lote puede superar batch_size (se añade una tarea entera mientras el acumulado sea menor que batch_size). El docstring de upstream ya dice que con cross_learning "Results become dependent on batch size" y recomienda batch_size ~100 [verificado en código, v2.3.2 pipeline.py].
- CHR-F12: corregido en dos detalles: (1) forzar cross_learning=False con una sola serie no cambia nada en el backend (diff 0.0 probado), es inocuo pero no necesario; (2) que un array bool se trate como categoría no es nuevo en 2.3.0: en 2.2.2 dataset.py ya usaba `np.issubdtype(dtype, np.number)`.
- CHR-P1 a CHR-P8: confirmados; la medida de CHR-P4 es reproducible.

## Cosas no señaladas por el investigador (verificadas)

- En 2.0.x y 2.1.x el cross-learning existía con el nombre `predict_batches_jointly` (renombrado en 3cb8f88, PR #418, v2.2.0); 2.2+ aún acepta el nombre viejo con aviso. No cambia la recomendación de CHR-P2 (a).
- chronos-forecasting 2.2.x fija transformers<5 y 2.3.0 añade transformers 5 (PR #416); relevante para elegir entre >=2.2 y >=2.3 en CHR-P2.
- Con torch_dtype=None, 2.3.x pasa dtype="auto" a transformers (base.py, v2.3.2): el modelo se carga en el dtype del checkpoint (float32 en los cuatro). Sin impacto.
- Documentación del prefijo: la tabla de llms-base.txt (línea 488) solo lista `autogluon/chronos-2` y SKILL.md (línea 165) `autogluon/chronos-2-*`, que no cubre `amazon/chronos-2` ni `autogluon/chronos-2` sin sufijo; adapter-parameters.md (línea 32) sí lista los dos prefijos.
- device_map='auto' en Mac: accelerate.get_max_memory añade "mps" y entonces no añade "cpu" (accelerate 1.15.0, utils/modeling.py), es decir, todo va a MPS (o disco), no a CPU. Refuerza CHR-F7; solo leído en código.
