# Verificación de la revisión de T0Adapter

Pasada de verificación adversarial de los hallazgos T0-F1 a T0-F16 y de las propuestas T0-P1 a T0-P8. Solo diagnóstico: no se ha tocado nada fuera de `dev/foundation_adapters_review/evidence/t0/verification/`.

## Entorno

- `~/.venvs/fm-review-t0`: Python 3.12.3, tfc-t0 0.6.0, torch 2.15.0+cpu, huggingface_hub 2.4.0, numpy 2.5.4, pandas 2.3.3, skforecast 0.27.0 (editable).
- `~/.venvs/fm-review-t0-032`: igual, con tfc-t0 0.3.2.
- Hardware: 4 CPU Intel Xeon @ 2.10GHz, Linux, sin GPU.
- Código upstream: clon de theforecastingcompany/tfc-t0 en `~/fm-review-src/tfc-t0` (actualizado con `git fetch`). Commits de los tags: v0.3.0 626d5b5, v0.4.0 ceef753, v0.5.0 a2eca8d, v0.6.0 4871c64; 0.3.2 no tiene tag (d22a507, `version = "0.3.2"` en pyproject.toml). El investigador cita 5766e50 para 0.4.0: también declara 0.4.0, pero el tag apunta a ceef753 (diferencia sin consecuencias).
- PyPI: la última versión sigue siendo 0.6.0 (2026-10-13). Upstream solo hay un commit posterior a v0.6.0 (5896313, 2026-10-14) y solo cambia un notebook.

## Scripts y logs de esta pasada

| Script | Log | Qué comprueba |
|---|---|---|
| `verify_t0.py` | `verify_t0_060.log`, `verify_t0_032.log` | F1, F4, F5, F6, F9 con datos distintos de los del investigador (demanda hasta 2014-06-30, contexto 300, 12 pasos, serie sintética). En 0.6.0 parchea `T0Adapter.predict` con la llamada de P1 (no usa el proxy del investigador). |
| `no_torch_import.py` | `no_torch_import.log` | F2: bloquea `import torch` con un meta path finder y muestra el error del adaptador. |
| `config_fields.py` | `config_fields.log` | F3: campos de config.json de t0-beta y t0-gamma que se pierden con 0.3.2. |
| `load_errors.py` | `load_errors.log` | F7 y F8: repo ONNX y repo inexistente. |
| `past_covariates.py` | `past_covariates.log` | F10: forma de la salida y tipo de variable de las filas con contexto [B, V, T]. |
| `hetero_trim.py` | `hetero_trim.log` | Si `trim_leading_missing` corrige también F5 (no lo hace). |
| (tests) | `contract_060.log`, `contract_032.log`, `unit_tests_060.log` | Contrato de compatibilidad y tests unitarios de T0. |
| (backtest del investigador) | `backtest_trim_rerun.log` | Reproducción del MAE antes y después de P2. |
| (Hub) | `hub_info.log`, `hub_configs.log`, `hub_commits.log` | Checkpoints, licencias, configs y commits. |

## Resultados por hallazgo

- **T0-F1 confirmado [probado].** Con 0.6.0, `TypeError: T0Forecaster.predict() got an unexpected keyword argument 'context'`. Con la llamada de P1 (contexto posicional y `quantile_levels`), la diferencia máxima entre adaptador y llamada directa es 0.0 sin y con exog, con cuantiles pedidos desordenados. El contrato con 0.6.0 y torch falla solo en `T0Forecaster.predict` (1 fallo, 2 pasan). Con 0.3.2 pasa 3/3.
- **T0-F2 confirmado [probado].** Sin torch, `import t0` funciona y el adaptador lanza `tfc-t0 is required for T0Adapter. Install it with pip install tfc-t0`, cuya causa es el ImportError de `t0.torch` que pide `'tfc-t0[torch]'`. El log del job 127875169126 (CI, 2026-11-06) instala tfc-t0 0.6.0 con `torch: null`: el contrato falla 2/3 y el smoke no llega a ejecutarse.
- **T0-F3 confirmado [probado].** Con 0.3.2, t0-beta pierde `scaler_eps` y `scaler_eps_mode`, y t0-gamma además `scaler_zero_no_value` y `xpos_scale_layout`; los dos predicen sin error. Con 0.6.0 no se pierde ningún campo. En d22a507, `__init__` acepta `**_: object`, y `T0Config` no tiene esos campos.
- **T0-F4 confirmado [probado].** La magnitud depende de los datos: aquí la diferencia relativa es 0.0039 (0.3.2 y 0.6.0), frente al 0.0144 del investigador. Con `trim_leading_missing=True` o con `mask` con PAD baja a 1.7e-07 (con PAD en 0.3.2, 8e-06). `trim_leading_missing` no cambia nada en una serie sola sin NaN iniciales (diferencia 0.0).
- **T0-F5 confirmado [probado].** Una fila de covariables toda NaN no equivale a no tener covariables: diferencia de 0.012 (0.6.0) y 0.014 (0.3.2) aquí, frente al 5.8-6.4% del investigador (la magnitud depende de los datos). Con el atributo a False la diferencia es 0. En `t0/data.py` (v0.6.0), `future_nan` pasa a 0 con `MaskType.MISSING`.
- **T0-F6 confirmado [probado], con un matiz.** `model.to(torch.bfloat16)` falla con `mat1 and mat2 must have the same dtype` en 0.3.2 y en 0.6.0. `from_pretrained(dtype=torch.bfloat16)` deja los parámetros en float32 y predice. Matiz: falla con cualquier dtype distinto de float32 (con float32, `.to()` no hace nada). En CUDA pasaría lo mismo [verificado en código: `_as_array` convierte siempre la entrada a float32 y `_predict_scope` solo usa autocast si `dtype` llegó al constructor].
- **T0-F7 y T0-F8 confirmados [probado].** El repo ONNX y el repo inexistente acaban en el mismo OSError ("could not be downloaded"). La causa es un TypeError (faltan 7 argumentos con ONNX, porque su config.json trae `quantile_levels`, y 8 con el repo inexistente). La búsqueda de issues upstream (2 issues y 16 PR) no tiene ninguno sobre esto.
- **T0-F9 confirmado [probado].** Con 0.3.2, q0.01 == q0.1 y q0.99 == q0.9; con 0.6.0, no. El docstring de 0.3.2 decía "interpolated", pero lo que hacía era recortar. `validate_quantile_levels` (v0.6.0) coincide con lo que dice el hallazgo.
- **T0-F10 corregido.** El README llama "past covariates" a las variables extra de `context` [B, V, T], pero el backend las trata como objetivos que se predicen conjuntamente: la salida es `(2, 3, 16, 1)` y todas las filas del contexto son `VariateType.TARGET` (0); las de `future_covariates` son FUTURE (2) [probado]. `from_array` nunca asigna HISTORICAL: solo se llega a ese tipo construyendo `TimeSeries` a mano [verificado en código]. Por tanto, no son covariables solo de pasado en el sentido de skforecast.
- **T0-F11 confirmado [verificado en código].** Ver la firma de `predict` y "Recommended inference settings" del README (v0.6.0).
- **T0-F12 confirmado [probado].** Los tres repos son apache-2.0, no gated, con un solo commit del 2026-10-15. El README dice que se descargan sin autenticación.
- **T0-F13 confirmado [verificado en código].**
- **T0-F14 confirmado [probado].** 20/20 con 0.6.0. `FakeT0Forecaster.predict(self, context, horizon, quantiles, future_covariates=None)`.
- **T0-F15 confirmado [verificado en código].** Celdas 11 y 84 de la guía, SKILL.md línea 61.
- **T0-F16 corregido.** `_build_future_covariates` tiene un bucle por serie y une las columnas de todas las series rellenando con NaN. Ese relleno es justo el tratamiento heterogéneo propio del adaptador que provoca F5, así que el adaptador no está del todo libre de relleno propio. Con P3 a False, esa unión de columnas y la rama `future_df is None: continue` dejan de hacer falta.

## Propuestas

- **P1 confirmada.** El riesgo y la incompatibilidad están bien evaluados. Ya hay un patrón de comprobación de versión en `TimesFM3Adapter._load_model` (`importlib.metadata.version("timesfm")`). La issue #1385 está asignada y tiene un comentario del 2026-11-06 que pide las comillas en el mensaje del ImportError.
- **P2 confirmada.** He repetido el backtest del investigador: MAE de 147.5073 antes y después, con las mismas versiones y el mismo hardware. La invariancia ante el lote también se reproduce. Ojo: el comentario del mantenedor en #1385 pide "confirmar que el NaN inicial se lee como missing"; F4 demuestra que precisamente eso es el problema.
- **P3 confirmada.** Además, P2 no corrige F5: con `trim_leading_missing=True` la diferencia sigue siendo 0.012 (`hetero_trim.log`). `_trim_leading_missing` marca PAD solo hasta la primera observación del objetivo. Son dos cambios independientes y hacen falta los dos.
- **P4 confirmada.** Hay que cambiar también el docstring ("the loaded model is cast to"): los pesos siguen en fp32.
- **P5, P6 y P7 confirmadas.** A P6 le faltan lugares (ver omisiones).
- **P8 corregida.** Las covariables solo de pasado por la vía [B, V, T] harían que el modelo las prediga como objetivos. La vía correcta sería construir `TimeSeries` con filas HISTORICAL (con >=0.6, porque 0.6.0 corrige su tratamiento en el horizonte). Prioridad baja.

## Omisiones encontradas

1. El docstring de `FoundationModel` (`skforecast/foundation/_foundation_model.py`, lista de `model_id` y referencia [12]) solo nombra t0-alpha. [verificado en código]
2. `skills/foundation-forecasting/SKILL.md` línea 130 enumera los adaptadores con `supports_heterogeneous_covariates=False` y dice que con ellos "the forecast of a series never depends on the exog of the others". Con P3, habrá que añadir T0. Además, la línea 283 de SKILL.md y la línea 458 de `tools/ai/llms-base.txt` dan `tfc-t0` como paquete a instalar, lo que desde 0.6.0 no instala torch. [verificado en código]
3. `trim_leading_missing` no sustituye a P3 (ver arriba). [probado]

## Fuera de alcance o no comprobado

- No he medido velocidad ni precisión entre versiones o variantes.
- CUDA, MPS y MLX: solo lectura de código.
- PR abiertas upstream (#16 cuenta las celdas CENSORED en el escalado, #18 exportador ONNX de t0-gamma): no publicadas, sin efecto hoy.
