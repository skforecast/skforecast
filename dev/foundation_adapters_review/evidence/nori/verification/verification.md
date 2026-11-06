# Verificación adversarial de la revisión de NoriAdapter

Fecha: 2026-11-06. Rama docs/foundation-adapters-review (desde origin/0.27.x, skforecast 0.27.0 en desarrollo).

Entorno: ~/.venvs/fm-review-nori (Python 3.12.3, synthefy-nori 0.23.0, synthefy 8.0.0, torch 2.13.0+cpu, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, huggingface-hub 1.33.0, skforecast 0.27.0 editable). Hardware: Intel Xeon 2.10GHz, 4 CPU lógicas, sin GPU. Código upstream: clon en ~/fm-review-src/synthefy-nori, tag synthefy-nori-v0.23.0 (f88b73f), más tags antiguos leídos con git show.

Entornos temporales (borrados después): ~/.venvs/fm-review-nori-old con synthefy-nori 0.21.1 y después con 0.18.2 (uv venv --python 3.12; uv pip install --torch-backend cpu "synthefy-nori==X").

## Scripts y logs de esta verificación

- verify_nori.py / verify_nori.log: enrutado, adapter frente a llamada directa (datos propios, distintos de los del investigador, diarios con dos covariables), orden de cuantiles desordenados y duplicados, mode frente a median, config de inferencia guardada en el checkpoint, NaN en covariables (filas pasadas a fit, variante "conservar NaN" frente a llamada directa), Nori-Causal, salida por stdout.
- old_backend_check.py / old_backend_check.log: Nori-1-5 con synthefy-nori 0.21.1.
- old_backend_nan_x.py / old_backend_nan_x_0.18.2.log: NaN en X y en y con synthefy-nori 0.18.2.
- dataframe_numeric_check.py / dataframe_numeric_check.log: matriz numpy frente a DataFrame numérico en NoriRegressor; DataFrame con columna de texto.
- pypi_info.log, hub_info.log, hub_configs.log: PyPI, Hub y config.json de los cinco repos Synthefy/Nori*.
- contract.log, smoke.log, unit_tests.log: re-ejecución de contrato, smoke y unit tests del adapter.
- backtest_current.log / backtest_patched.log (+ csv): re-ejecución del backtest de la guía antes y después del prototipo NORI-P2.

## Veredictos

| Id | Veredicto | Etiqueta | Nota |
|----|-----------|----------|------|
| Fila resumen | confirmado | probado | Job 127875169149 (run 42212742430, rama 0.27.x) leído: synthefy-nori 0.23.0, torch 2.13.0+cpu, numpy 2.5.4, pandas 2.3.3, huggingface_hub 1.33.0, contract 4, smoke 6, held_back false. 0.24.0 = 24 Aug 2026 en releases.md; 0.18.2 (20 Aug) era la última en PyPI ese día, 0.19.0 salió el 25 Aug (inferido). |
| NORI-F1 | confirmado | probado | 4 passed, 6 passed, 43 passed re-ejecutados. |
| NORI-F2 | confirmado | verificado en código | Firmas extraídas con ast en v0.18.2 y v0.23.0: mismos posicionales; solo kwargs nuevos (encode_id_cols, large_context_holdout, large_context_max_calls, multi_target_prediction_strategy/policy en __init__; multi_target_prediction_policy en predict). Sin cambios en ningún valor por defecto del constructor. resolve_model_repo devuelve NORI_MODELS.get(model, model). |
| NORI-F3 | confirmado | probado | Con otros datos (diarios, 2 covariables, context 256, 12 pasos): diferencia 0.0 en mean, median, mode y cuantiles desordenados, para Synthefy/Nori y Synthefy/Nori-1-5; nombre de registro frente a repo id: 0.0. Matiz: con repo id crudo el backend no valida la config de inferencia esperada (get_expected_inference_config_path devuelve None), pero Nori-1-5 guarda su config (model_v3_inference.json) en el checkpoint y se usa igual. |
| NORI-F4 | confirmado | probado | Repetido con 0.21.1: Synthefy/Nori ok; Synthefy/Nori-1-5 TypeError get_x_encoder() ... prompt_quantile_ple_bins. Extra: nori-1-5 (nombre de registro) con 0.21.1 da CheckpointAccessError "access denied" (se trata como repo id crudo). Ningún código de 0.23.0 lee min_synthefy_nori_version. |
| NORI-F5 | confirmado | probado | Se enruta a NoriAdapter, la construcción no falla y predict lanza RemoteEntryNotFoundError 404. config.json: model_type noricausal, salida (T, T). |
| NORI-F6 | confirmado | probado | Hub: apache-2.0 y gated false en los cinco; _utils.py:321. |
| NORI-F7 | confirmado | probado | Shas y fechas re-consultados; _resolve_model_path llama download_checkpoint(model=, token=) sin revision y NoriRegressor no tiene argumento revision. |
| NORI-F8 | confirmado | probado | Cabeza 999-quantile pinball en los cuatro config.json; api.py v0.23.0: output_type "mode" usa quantile_collapse="median"; mode == median ejecutado en 6M y 1.5. |
| NORI-F9 | confirmado | verificado en código + probado | El bucle for i, level in enumerate(q_levels) existe ya en v0.11.0, v0.15.0, v0.16.0 ... v0.23.0. Niveles [0.8, 0.2, 0.5, 0.05] y duplicados [0.5, 0.9, 0.5, 0.1] devueltos en el orden pedido. |
| NORI-F10 | confirmado | verificado en código + probado | _predict_point y _predict_distribution convierten a numpy float64 (desde v0.15.0 al menos). Salida real ndarray float64. |
| NORI-F11 | confirmado | probado | 68 de 256 filas con NaN en x1: el adapter pasa 188 filas; la variante que solo filtra NaN en y pasa 256 y coincide (0.0) con la llamada directa. También acepta NaN en X synthefy-nori 0.18.2 (probado), así que el cambio no depende de una versión mínima nueva. NaN en y: ValueError en 0.23.0 y en 0.18.2. |
| NORI-F12 | confirmado | verificado en código + probado | Commit 00def60 (2026-08-17), primer tag synthefy-nori-v0.18.1. Probado: DataFrame con columna de texto y nivel no visto en la consulta da predicciones finitas. |
| NORI-F13 | confirmado | verificado en código | _default_device y _as_device en api.py v0.23.0; en esta máquina device_ = cpu. CUDA/MPS solo leído. |
| NORI-F14 | confirmado | probado | print en predictor.py:321; capturado en stdout. |
| NORI-F15 | confirmado | verificado en código | Bucle por item_id en nori_ts/core.py:267; extra forecasting con datasets, gluonts y statsmodels; issue #194 abierto (búsqueda gh). No se probó el ModuleNotFoundError. |
| NORI-F16 | confirmado | documentado | Metadatos de PyPI re-consultados (son metadatos publicados, no código). |
| NORI-F17 | confirmado | verificado en código | Docstrings y guía (celda 11 con docs.synthefy.com/forecasting-api/overview). Matiz: la insignia Docs del README upstream apunta a docs.synthefy.com/nori/, la misma que usa el docstring del adapter. docs.synthefy.com no es accesible desde aquí (proxy 403). |
| NORI-F18 | confirmado | verificado en código | fixtures_adapters.py:593-635; test_NoriAdapter.py:247 comprueba además que el fake recibe niveles ordenados. |
| NORI-F19 | ver sección backtest | probado | Re-ejecutado (ver abajo). |

## Propuestas

- NORI-P1: confirmado. Riesgo bajo correcto; la opción (a) es coherente con .claude/rules/foundation.md.
- NORI-P2: confirmado y reforzado: 0.18.2 también acepta NaN en X, de modo que el cambio no exige subir el mínimo del backend. Cambia predicciones solo con NaN en covariables del contexto.
- NORI-P3: corregido. backend_package solo se usa en el mensaje de ImportError, en run.py y en el test de coherencia con adapters.toml; no hay comprobación de versión en tiempo de ejecución. Poner synthefy-nori>=0.22 NO obliga a actualizar a quien ya tiene 0.21.1 instalado: seguiría viendo el TypeError opaco. Además, el mensaje de ImportError de NoriAdapter no entrecomilla backend_package (TabPFNAdapter sí: pip install "tabpfn-time-series>=1.3"); con >= sin comillas el shell redirigiría la salida. Para dar un error claro hace falta una comprobación de versión en carga (como TimesFM3Adapter con importlib.metadata) o, mejor por la regla de adapters finos, aceptar que (a) solo arregla instalaciones nuevas y documentarlo.
- NORI-P4: confirmado.
- NORI-P5: confirmado; cambio sin efecto en las predicciones con todas las versiones del backend desde v0.11.0 (código leído). Habrá que cambiar test_NoriAdapter.py:247.
- NORI-P6: corregido. El riesgo indicado ("el preprocesado de DataFrame también trata columnas numéricas, las predicciones podrían cambiar aunque no haya categóricas") no se sostiene: _transform_tabular (libs/synthefy/src/synthefy/featurize.py) pasa las columnas numéricas como float32 sin transformar, y probado: numpy frente a DataFrame numérico da diferencia 0.0 en punto y cuantiles para 6M y 1.5. Las predicciones solo cambiarían para entradas que hoy lanzan ValueError (columnas no numéricas). Riesgos reales: columnas object con cardinalidad mayor que max_categorical_cardinality lanzan error en el backend; predict exige DataFrame con las mismas columnas y dtypes.
- NORI-P7: corregido en un detalle: añadir una entrada a adapters.toml obliga también a añadirla a la matriz de .github/workflows/foundation-backends.yml (test_workflow_matrix_matches_adapters_toml).
- NORI-P8, NORI-P9, NORI-P10: confirmados (P10: el README upstream usa nori-1-5 en todos los ejemplos y DEFAULT_DOWNLOAD_MODEL_NAME = "nori-1-5" en hf.py).

## Cosas no detectadas por el investigador (verificadas)

1. El mensaje de ImportError de NoriAdapter (_adapters.py, _load_model) no entrecomilla backend_package; relevante si se adopta NORI-P3 (a).
2. backend_package no se aplica en tiempo de ejecución (ver P3).
3. Con synthefy-nori < 0.22, nori_config={"model": "nori-1-5"} produce CheckpointAccessError "access denied" engañoso (old_backend_check.log). Afecta solo a quien use el nombre de registro; el adapter pasa el repo id crudo.

## Backtest de la guía (NORI-F19 / NORI-P2)

Re-ejecutado con patch_keep_nan_covariates.py guide --variant current|patched (scripts del investigador, Synthefy/Nori, exog, context_length=500, steps=24): MAE 185.9507 en ambos, 720 predicciones, diferencia máxima 0.0 entre variantes y 0.0 frente al csv del investigador. Python 3.12.3, synthefy-nori 0.23.0, torch 2.13.0+cpu, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, huggingface_hub 1.33.0; Intel Xeon 2.10GHz, 4 CPU lógicas, sin GPU. vic_electricity no tiene NaN en Temperature ni Holiday (comprobado), así que el backtest solo demuestra que no hay regresión sin NaN; el efecto con NaN se ve en verify_nori.log (sección D). NORI-F19: confirmado.
