# Verificacion adversarial: TimesFM25Adapter (timesfm 3.0.2)

Fecha: 2026-11-06. Rama docs/foundation-adapters-review (desde origin/0.27.x, skforecast 0.27.0).
Alcance: alineacion del codigo del adapter con su backend y checkpoints. No se ha medido
velocidad ni precision.

## Entornos

- `~/.venvs/fm-review-timesfm25` (del investigador, reutilizado): Python 3.12.3, timesfm 3.0.2,
  torch 2.15.0+cpu, huggingface_hub 2.4.0, numpy 2.5.4, pandas 2.3.3.
- `~/.venvs/fm-review-timesfm25-oldhf` (del investigador, sin modificar): timesfm 2.0.2 + huggingface_hub 0.36.0.
- `~/.venvs/fm-review-timesfm25-verify` (propio): `uv venv --python 3.12` + `uv pip install --torch-backend cpu -e . "timesfm[torch]==X" "huggingface_hub==Y"`,
  instalando en sucesion (2.0.0, 0.36.0), (2.0.1, 0.36.0), (3.0.2, 0.36.0), (3.0.2, 1.0.0) y (3.0.2, 0.29.3). Queda con la ultima combinacion.
- Venvs temporales solo con numpy 1.26.4, 2.1.3 y 2.2.0 (borrados tras usarse).
- Hardware: 4 CPU, sin GPU, Linux x86_64.
- Codigo upstream: `~/fm-review-src/timesfm` (tags v2.0.1, v2.0.2, v3.0.0, v3.0.2; origin/master e51928e).

## Scripts y logs de esta verificacion

| Script | Log | Que comprueba |
|---|---|---|
| verify_direct.py | verify_direct.log | adapter vs llamada directa (otros datos: NaN iniciales e internos, serie mas corta que el contexto), canal de la puntual, config.json, casos limite de forecast_config_kwargs, Float64 con pd.NA |
| (check_flip_ar_quantiles.py del investigador) | rerun_flip_ar_quantiles.log | orden de cuantiles en pasos > 128 con 2.0.2 y 3.0.2 |
| (check_workaround.py del investigador) | rerun_workaround.log | carga sin `_TimesFMCompat` en 4 combinaciones de versiones |
| (check_import_errors.py del investigador) | rerun_import_errors.log | timesfm sin torch |
| hfhub_029 (comando en linea) | hfhub_029_import.log | timesfm 3.0.2 + huggingface_hub 0.29.3 |
| verify_registry_checkpoints.py | verify_registry_checkpoints.log | commits del repo -pytorch, tensores de -transformers (cabecera safetensors, sin descargar), -flax, enrutado de prefijos |
| verify_batch_size.py | verify_batch_size.log | per_core_batch_size=4 vs 1 a traves del adapter |
| verify_all_nan_context.py | verify_all_nan_context.log | contexto recortado totalmente NaN |
| verify_all_nan_old_numpy.py | verify_all_nan_old_numpy.log | mismo caso con numpy 1.26.4, 2.1.3, 2.2.0 y 2.5.4 |
| pytest contract + smoke, unit tests | rerun_contract_smoke.log, rerun_unit_tests.log | 11 passed y 1 skipped; 58 passed (seleccion -k timesfm25) |
| hub_info.py | hub_rerun.log | listado del Hub de google/timesfm |

## Veredictos por hallazgo

- T25-F1 confirmado [probado]. PyPI: 3.0.2 (2026-09-09) es la ultima. `git diff --stat v3.0.2 origin/master -- src/timesfm` vacio; los cambios posteriores son README y src/timesfm3/mlx. Log del job 127875168998 leido via GitHub: timesfm 3.0.2, torch 2.15.0+cpu, huggingface_hub 2.4.0, contract 6 passed, smoke 5 passed y 1 skipped.
- T25-F2 corregido [verificado en codigo]. Las firmas usadas no cambian entre v2.0.1 y v3.0.2, pero NO es cierto que los unicos cambios sean el import de timesfm3 y af8f595: entre v2.0.1 y v2.0.2 tambien entran 1e89ad1 y f83f134 (torch.compile se aplica a `forward` en lugar de reasignar `self`, que era un no-op, y se respeta el kwarg `torch_compile` en `load_checkpoint`) y 2011cfb (descarga previa de config.json en `_from_pretrained`). Entre v2.0.2 y v3.0.2 si es solo af8f595 mas el import de timesfm3.
- T25-F3 confirmado [probado]. Con datos distintos (3 series diarias, una con 20 NaN iniciales y 10 internos, otra de 90 puntos con context_length=320, steps=30, los 9 cuantiles): diferencia maxima 0.0 en puntual y cuantiles.
- T25-F4 confirmado [probado]. point == canal 5 exacto; canal 0 difiere hasta 0.30. Tabla de la guia (celda 11, columna "Point forecast": "Mean (dedicated output)"), Notes del adapter y adapter-parameters.md linea 53 ("mean (in practice equal to quantile 0.5)"). La skill upstream timesfm-forecasting/SKILL.md (v3.0.2, linea 239) dice "the median (0.5 quantile)".
- T25-F5 confirmado [probado]. 2.0.0 + hf 0.36.0 sin workaround: TypeError `proxies`; 2.0.1/3.0.2 + hf 0.36.0 y 3.0.2 + hf 1.0.0 cargan sin el. 2f1625c esta en v2.0.1 (PR #435, fusionado 2026-06-08). Matiz: con huggingface_hub < 0.30 `import timesfm` ya falla (ver TV-1), asi que el workaround solo actua con timesfm 2.0.0 y 0.30 <= huggingface_hub < 1.0 [verificado en codigo: `paper_url` tambien en la rueda 2.0.0].
- T25-F6 confirmado [probado]. Reproducido: 2.0.2 ordenado 47 % y 50 % en pasos 129-384, anchura 0.024 y 0.046; 3.0.2 al 100 %. af8f595 llega por el PR #463 (merge 3dae50b) y solo esta en v3.0.0 y v3.0.2.
- T25-F7 confirmado [probado]. AttributeError con torch oculto. timesfm 1.3.0 no contiene `TimesFM_2p5` ni `ForecastConfig` [verificado en codigo de la rueda].
- T25-F8 confirmado [probado en parte]. -flax: RemoteEntryNotFoundError 404. -transformers: 272 tensores, faltan 228 de los 232 esperados (cabecera safetensors); el fallo de `load_state_dict(strict=True)` tras descargar 925 MB es [verificado en codigo], no ejecutado. timesfm-2.0 rechazado, timesfm-3.0 -> TimesFM3Adapter. Los tres 2.5 apache-2.0 y no gated.
- T25-F9 confirmado [probado]. Commits del repo -pytorch: ultimo 1d95242 (2025-10-02, README); pesos d418f3e (2025-10-01). config.json con `torch_compile: false`; instancia cargada con torch_compile False.
- T25-F10 confirmado [probado]. max_context: TypeError "multiple values" en el primer predict (no en __init__); return_backcast: ValueError de broadcast (318 filas frente a 30 con context_length=320, coherente con 216 frente a 24 del investigador). window_size con TODO en configs.py v3.0.2.
- T25-F11 confirmado [probado]. 16384 falla ("16384 + 128 > 16384"), 16000 funciona; limite 1024 con continuous quantile head en compile() v3.0.2.
- T25-F12 confirmado [verificado en codigo]. 0d4d514 (2025-10-28) en v2.0.1+, y forecast_with_covariates esta en la rueda 2.0.0; beta_hat unico con jnp.linalg.pinv sobre los objetivos aplanados de todo el lote; extra xreg = jax[cuda], scikit-learn; exige return_backcast=True. No ejecutado.
- T25-F13 confirmado [verificado en codigo]. cuda:0 o cpu (lineas 72-77), float32 (428-430); PR #415 cerrado 2026-10-31 sin merge.
- T25-F14 confirmado [probado]. Diferencia 1.5e-5 (float32) con 5 series a traves del adapter; global_batch_size 1 vs 4. Sin tiempos.
- T25-F15 confirmado [verificado en codigo y probado]. Pero el atributo supports_nan_in_series=True tiene un caso limite no cubierto (TV-2).
- T25-F16 confirmado [verificado en codigo].
- T25-F17 confirmado [verificado en codigo].

## Veredictos por propuesta

- T25-P1 confirmado con matiz. "Misma cota que TimesFM3Adapter" es inexacto: TimesFM3Adapter declara `backend_package = "timesfm[torch]"` sin cota y solo comprueba la version en tiempo de ejecucion cuando falta `TimesFM3Forecaster`. Riesgo bajo y "no cambia predicciones" correctos (carga directa sin workaround == adapter). Requiere decision.
- T25-P2 confirmado.
- T25-P3 confirmado (la clave de licencia y el registro usan `startswith`, asi que el prefijo exacto sigue resolviendo la licencia).
- T25-P4 confirmado. Anadir `window_size` (no implementado upstream) a la lista de claves a rechazar o documentar.
- T25-P5 confirmado; conviene que el helper comun capture tambien el TypeError de TV-1.
- T25-P6 confirmado: el criterio de orden de cuantiles falla con 2.0.2 y pasa con 3.0.2 en esta maquina.
- T25-P7 confirmado (requiere decision).
- T25-P8 confirmado.

## Lo que el investigador no vio

- TV-1 [probado]. timesfm 3.0.2 declara `huggingface_hub>=0.28.0`, pero con 0.28/0.29 `import timesfm` lanza `TypeError: ModelHubMixin.__init_subclass__() got an unexpected keyword argument 'paper_url'`. `_import_timesfm` solo captura ImportError, asi que el usuario ve ese TypeError (afecta tambien a TimesFM3Adapter). PR upstream #531 abierto (sube el minimo a 0.30.0). Prioridad baja.
- TV-2 [probado]. Si el contexto recortado de una serie es todo NaN, `linear_interpolation` de timesfm 2.5 hace `if non_nans_values:` sobre un array vacio: con numpy >= 2.2 da `ValueError: The truth value of an empty array is ambiguous` (mensaje criptico, todo el predict falla); con numpy < 2.2 (skforecast admite >= 1.26.1) rellena con ceros y predice desde ceros sin avisar. Sin arreglo upstream (PR #486 abierto). Recomendacion: comprobacion central en FoundationModel (contexto sin ningun valor observado -> ValueError claro), no en el adapter.
- TV-3 [verificado en codigo]. `_ensure_compiled` solo compara `max_horizon`; con un `model` precompilado con `max_context` menor que `context_length`, el backend trunca en silencio a `max_context` (`value[-context:]` en forecast). Prioridad baja: documentar o avisar.
