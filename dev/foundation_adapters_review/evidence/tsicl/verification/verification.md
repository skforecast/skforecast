# Verificacion adversarial: TSICLAdapter (tsicl)

Fecha: 2026-11-06. Rama docs/foundation-adapters-review (skforecast 0.27.0).
Entorno reutilizado: ~/.venvs/fm-review-tsicl (Python 3.12.3, tsicl 0.2.1, torch 2.9.1+cpu, numpy 2.5.4, pandas 2.3.3, Linux x86_64, 4 CPU, sin CUDA ni MPS).
Codigo upstream: ~/fm-review-src/ts-icl (git fetch hoy: sin commits ni tags nuevos; v0.2.1 = 61d1e89, main = 349f3ea).

## Que se ha re-ejecutado

| Script / comprobacion | Log | Resultado |
|---|---|---|
| `verify_backend_direct.py` (nuevo, llamadas directas a `TSICL.forecast` con datos distintos a los del investigador) | `verify_backend_direct.log` | Reproduce F1, F2, F3, F4, F5, F7, F8, F9, F10 y el tipo de salida de F14 |
| `check_quantile_exact_types.py` (nuevo) | `check_quantile_exact_types.log` | Los 8 niveles rechazados SI se pueden pasar como `Decimal`/`Fraction` (refuta una parte de F7) |
| `check_padding_option.py` del investigador (re-ejecutado sin cambios) | `rerun_check_padding_option.log` | Mismos numeros que F6 |
| Contrato (`test_contract.py --backend tsicl`) | `contract.log` | 3 passed |
| Smoke (`test_smoke.py --backend tsicl`) y unitarios `test_TSICLAdapter.py` | `tests.log` | 6 passed; 54 passed |
| Registro y licencia (`FoundationModel` con 4 model_id, `_MODEL_LICENSES`, firmas del backend) | `registry_license.log` | Igual que F11 |
| Hub: ficheros por revision (f01f3869 vs 19c94031) | `hub_revisions.log` | El checkpoint (LFS sha256 a67ae9f694c2) no cambia; solo LICENSE |
| PyPI (`pypi_info.py tsicl`, JSON de cada release) | (consola) | 0.2.1 sigue siendo la ultima; todas las releases exigen Python >=3.12 y torch <2.10 |
| Issues upstream (search/issues is:issue / is:pull-request) | (consola) | #1 cerrado 2026-07-17; #2 a #23 abiertos; ninguno sobre la comparacion exacta de floats de cuantiles |
| Log del job CI 127875169087 (API de GitHub Actions) | (consola) | tsicl 0.2.1, torch 2.9.1+cpu, contract 3 passed, smoke 6 passed |

No se ha ejecutado codigo de main upstream sin publicar ni benchmarks. No se ha ejecutado `backtest_guide.py`: ninguna propuesta verificada cambia la prediccion del caso de la guia (una serie, exog con valores futuros, steps=24, cuantil 0.5).

## Veredictos de hallazgos

- **TSI-F1 confirmado [probado]**: adaptador vs llamada directa, max_abs_diff=0 con series de longitudes distintas, cuantiles [0.05, 0.5, 0.95] y exog pasado+futuro. Diff v0.2.1..main en `src/` solo toca `_rollout_f`.
- **TSI-F2 confirmado [probado]**: backend directo, solo pasado vs sin exog = 0.000e+00; pasado+futuro vs sin exog = 0.995. Codigo: `pipeline.py@v0.2.1` linea 401 (`covar_t = None` si no hay futuro) y 1364.
- **TSI-F3 confirmado [probado]**: mixto vs solo la covariable con futuro = 0.000e+00; `data_adapter.py@v0.2.1` linea 184.
- **TSI-F4 confirmado [probado]**: serie a junto a una b mas corta == a sin exog (0.000e+00) y difiere 0.995 de a sola con exog; con igual longitud, 7.6e-06. `data_adapter.py@v0.2.1` linea 288. Matiz: la longitud que cuenta es la que llega al backend tras recortar a `context_length`, asi que el fallo aparece cuando alguna serie del grupo es mas corta que `context_length` (4096 por defecto: caso frecuente).
- **TSI-F5 confirmado [probado]**: NaN en la exog pasada de b cambia a en 7.9e-03 (ruido sin NaN 7.6e-06). `perceiver.py@v0.2.1` linea 561 (`isnan().sum(0)`).
- **TSI-F6 confirmado [probado]**: re-ejecucion identica (con relleno: s1 7.6e-06, s2 8.3e-02 con exog y 2.3e-01 sin exog).
- **TSI-F7 corregido [probado]**: los 8 niveles se rechazan (comparacion exacta de `100*x` con `np.linspace(1, 99, 99)`), y es cierto que ningun float vecino sirve. Pero el backend solo usa el valor para buscar el indice: pasando `Decimal(str(q))` o `Fraction(str(q))` el indice es el correcto (6, 13, 27, 28, 54, 55, 56, 57), la llamada funciona, q0.06 y q0.08 salen identicos a la referencia y q0.07 queda entre ambos. Por tanto el adaptador si puede sortearlo traduciendo el valor.
- **TSI-F8 confirmado [probado + verificado en codigo]**: steps=700 sin exog OK; 1344 sin exog ValueError (pipeline.py:1270); 700 con exog RuntimeError (pipeline.py:1101). El rollout solo se usa con steps > 672 (linea 1068). Arreglo en main (a60e7c7, 2e6624d; PR #1 fusionado 2026-07-17) leido en codigo: `pred_len = prediction_length - start` y `dim=-2`. Sin release en PyPI.
- **TSI-F9 confirmado [verificado en codigo + probado en CPU]**: `_get_device(mps) -> cpu`. `_resolve_torch_device` esta en `_adapters.py` (linea 34) con `mps_fallback_message`; TSICLAdapter no lo usa.
- **TSI-F10 confirmado [probado]**: `context_length=10000 -> this_context_length=4096`.
- **TSI-F11 confirmado [probado]**: resolucion de los 4 model_id igual; `hf_repo_ID = 'taharnbl/TS-ICL'` fijo (`pipeline.py@v0.2.1` linea 88); Hub sin gating, licencia other / tsicl-v1-license-v1.0; el ultimo commit del Hub (19c94031, 2026-07-07) solo cambia LICENSE.
- **TSI-F12 corregido [documentado]**: la LICENSE del Hub es la de main (solo difiere el salto de linea final). Pero la version del 9 de junio YA incluia el codigo de inferencia en "TS-ICL Model" y YA restringia los Outputs a fines no comerciales ("Outputs may be used only for Non-Commercial Purposes"). Los cambios reales son: definicion de "Derivative" mas amplia (incluye destilar con Outputs), redaccion de "Non-Commercial Purpose" y de la clausula de Outputs, y "data produced" pasa a "Outputs" en 3.a. No hay cambio material para skforecast; la entrada de `_MODEL_LICENSES` sigue correcta.
- **TSI-F13 confirmado [documentado]**: todas las releases (0.1.0 a 0.2.1) declaran `requires_python >=3.12` y `torch<2.10,>=2.5.1`; skforecast declara `>=3.10`. Tambien exige scikit-learn>=1.8.0, numpy>=2.1.3 y huggingface-hub>=1.19.0 (0.2.1), mas altos que los minimos de skforecast.
- **TSI-F14 confirmado [probado]**: backend con `squeeze_output=False` devuelve Tensor (2, 1, 24, 1) con igual longitud y lista de (1, 24, 1) con longitudes distintas; `FakeTSICL` devuelve siempre lista. No hay tests de los casos citados.
- **TSI-F15 confirmado [verificado en codigo]**: firma de `TSICL.forecast` con `batch_size=64`, `point_estimator='mean'`, `allow_auto_complete`, `allow_covar_forecast`. Issues #2, #9, #21, #23 existen y estan abiertos (titulos comprobados).
- **TSI-F16 confirmado [verificado en codigo]**: adaptador fino; `squeeze_output=True` hace `.squeeze()` total (`pipeline.py@v0.2.1` lineas 779-781), asi que `squeeze_output=False` sigue siendo necesario.
- **TSI-F17 corregido [probado]**: (a)-(e) confirmados (guia celda 11: "No hard limit, set via steps"; `_foundation_model.py` linea 1114: "never depends on the exog columns of the other series"). Falta en la lista: `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` celda 2 ("Does batching series together change their forecasts? Not for Chronos-2, TS-ICL and TabICL") y celda 21 (tabla: TS-ICL "Past-only covariates: Yes"; texto: "Past-only covariates are supported"), y `skills/foundation-forecasting/references/adapter-parameters.md` linea 249 (TS-ICL entre los que usan covariables solo pasadas).

## Veredictos de propuestas

- **TSI-P1 confirmado**: (a) y (b) se comportan como dice (V4: igual longitud 7.6e-06; V5: el NaN persiste con igual longitud). (c) descartada con numeros reproducidos. Matiz: si se elige (a), agrupar por longitud tras el recorte a `context_length`. Decision del mantenedor.
- **TSI-P2 confirmado**: cambio de atributo solo anade el `IgnoredArgumentWarning` (`_foundation_model.py` linea 992); las predicciones no cambian (V2/V3 diff 0). Tambien cambia la salida de `get_model_info`/`list_adapters` y hay que actualizar `test_check_exog_columns.py` (`(TSICLAdapter, True)`) y la guia heterogenea. (b) no cubre el caso mixto: confirmado en codigo (linea 411 exige `not covar_on_future`).
- **TSI-P3 corregido**: falta una opcion (d): traducir en el adaptador cada nivel a `Decimal(str(q))` antes de llamar al backend; hace veraz la rejilla completa de 99 niveles sin quitar niveles, no cambia ningun otro cuantil (indices identicos) y se retira cuando upstream corrija la comparacion. Es un workaround que depende de un detalle interno (solo se usa el indice), por eso conviene un test. Recomendacion revisada: (d) + reportar upstream; (a) si se prefiere no depender del detalle interno.
- **TSI-P4 confirmado**: el guard debe aplicarse solo a steps > 672 (con exog futura, o multiplo de 672); steps <= 672 no usa rollout.
- **TSI-P5 confirmado**: `mps_fallback_message` existe y lo usan MoiraiAdapter (linea 2083) y TimestralAdapter (linea 5885).
- **TSI-P6 confirmado**.
- **TSI-P7 corregido**: anadir la guia heterogenea (celdas 2 y 21) y `adapter-parameters.md` linea 249; el matiz de licencia "cubre tambien el codigo de inferencia" no es nuevo (ver F12) y no justifica cambio.
- **TSI-P8 confirmado**.

## Omisiones verificadas

- **TSI-V1 [probado]**: la guia de series heterogeneas afirma para TS-ICL independencia del lote y soporte de covariables solo pasadas, ambas falsas con tsicl 0.2.1 (F2, F4, F5).
- **TSI-V2 [probado]**: workaround de cuantiles con `Decimal` (ver F7/P3).
