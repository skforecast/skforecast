# Revisión transversal de los adaptadores foundation (skforecast 0.27.x, 2026-11-06)

Alcance: según la petición del mantenedor, el objetivo es que el **código** de los adaptadores siga alineado con su backend y sus checkpoints. Queda fuera todo lo que mida velocidad o precisión de modelos o de versiones. TimestralAdapter no tiene revisión propia en esta tanda; solo aparece cuando usa código compartido.

## 1. Estado por adaptador (después de la verificación adversarial)

| Adaptador | Backend probado / último | Estado | Prioridad | Lo esencial |
|:--|:--|:--|:--|:--|
| T0 | tfc-t0 0.3.2 / 0.6.0 | **roto** | alta | La API de predict cambió y torch pasó a ser un extra (#1385). Además, la salida de una serie depende de las demás del lote (relleno con NaN y covariables heterogéneas). |
| TS-ICL | tsicl 0.2.1 / 0.2.1 | **desfasado** | alta | Tres capacidades declaradas no se cumplen: covariables solo pasadas, cuantiles y la independencia entre series. |
| TimesFM 3.0 | timesfm 3.0.2 / 3.0.2 | alineado | media | Comprobación de import con código muerto, docstring de `padding_mode` falso, la skill afirma independencia entre series. |
| TimesFM 2.5 | timesfm 3.0.2 / 3.0.2 | alineado | media | Falta un mínimo `>=3.0` (con 2.0.x los cuantiles de más de 128 pasos salen mal). `_TimesFMCompat` es eliminable. Los docs dicen "media" donde es la mediana. |
| Chronos-2 | 2.3.2 / 2.3.2 | alineado | media | Mínimo real 2.2 no declarado. Un contexto todo NaN da cuantiles de ±1e5. Un Categorical con categorías numéricas se trata como numérico. |
| Moirai-2 | uni2ts 2.0.0 / 2.0.0 | alineado | media | Con Python 3.13+ se instala uni2ts 1.1.1 y el error de import engaña. Con pip se rompe el entorno (scipy, pandas). |
| TabPFN-TS | 1.4.0 / 1.4.0 | alineado | media | La redirección de stderr oculta los warnings. En CUDA, `compile` no se puede controlar. Las salidas de la guía son anteriores a 1.3. |
| TabICL | 2.2.0 / 2.2.0 | alineado | media | Mismo problema de stderr. Rechaza niveles de cuantil int o float32. El mínimo real (2.0.3) no está declarado. |
| Nori | 0.23.0 / 0.23.0 | alineado | media | El prefijo acepta Nori-Causal. Nori-1-5 exige el backend `>=0.22`. Se descartan filas con covariables NaN. Hay dos workarounds redundantes. |

En todos los adaptadores, la salida a través de FoundationModel coincide con la llamada directa al backend (diferencia 0, o ruido float32 en TabICL). La traducción de entradas y salidas es correcta; los problemas están en las capacidades declaradas, los mínimos de versión, el código compartido y la documentación.

## 2. Hallazgos transversales (verificados en esta pasada)

### XC-1. Una ventana de contexto toda NaN llega al backend sin aviso [probado]
- **Qué pasa.** FoundationModel solo rechaza las series enteramente NaN (`check_preprocess_series`, `skforecast/utils/utils.py` ~4445). Después recorta con `iloc[-context_length:]` (`_foundation_model.py` 591), y no comprueba que en la ventana quede algún valor observado. En `probe_shared.log` (sección 1), con `context_length=50` y las últimas 80 observaciones NaN, `predict` se ejecuta y el backend recibe un contexto todo NaN.
- **Qué hace cada backend:**
  - Chronos-2 devuelve cuantiles de ±1e5 (CHR-F4, probado; issue upstream #544 abierta).
  - Moirai devuelve valores en torno a 0 (MOI-F12, probado).
  - TimesFM 2.5 lanza un `ValueError` críptico con numpy >=2.2, o predice a partir de ceros con numpy <2.2 (TV-2, probado; PR upstream #486 abierto).
  - TabICL y TabPFN rellenan con 0 cuando hay como mucho un valor válido (TAB-F9 y TPF-F15).
- **Docs y comentarios falsos:**
  - El comentario de `skforecast/model_selection/_validation.py` 2874-2882 dice que la ruta del usuario las rechaza.
  - La celda 2 de `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` dice "A context window that is entirely NaN is rejected".
  - `backtesting_foundation` sí salta esas series.
- **Dónde arreglarlo.** En FoundationModel.predict, después del recorte y para todos los adaptadores; es lo que pide `.claude/rules/foundation.md`. Fusiona CHR-P1, MOI-P5 y TV-2.

### XC-2. Mínimos de backend no declarados ni comprobados en ejecución; mensajes de instalación inconsistentes [verificado en código; probado en sección 4]
- **`backend_package` es solo informativo.** Se usa en el mensaje de ImportError, en `get_model_info` y en `tools/foundation_compat/run.py`. No existe un helper común de versión: solo TimesFM3Adapter compara la versión con `importlib.metadata` (`_adapters.py` 1656-1696).
- **Por eso un mínimo en `backend_package` no protege instalaciones existentes.** Lo vieron NORI-P3, TAB-P3 y T25-P1 (corrección del verificador).
- **Faltan mínimos en seis backends:**

  | Backend | Mínimo necesario | Hallazgo |
  |:--|:--|:--|
  | tfc-t0 | `[torch]>=0.6` | T0 |
  | timesfm para 2.5 | `>=3.0` | T25-F6 |
  | chronos-forecasting | `>=2.2` | CHR-F3 |
  | uni2ts | `>=2.0` | MOI-F3 |
  | tabicl | `>=2.0.3` | TAB-F2 |
  | synthefy-nori, para Nori-1-5 | `>=0.22` | NORI-F4 |

  TabPFN ya declara `>=1.3`, aunque TPF-V1 y TPF-V2 dan argumentos para subir a 1.4.
- **Comillas en el ImportError.**
  - Sin comillas: Chronos (451), Moirai (2042), TabICL (2559), T0 (3787), TSICL (4300) y Nori (4890).
  - Con comillas: `_import_timesfm`, TimesFM3, TabPFN y Timestral.
  - En cuanto un `backend_package` lleve `>=` o un extra, el comando sin comillas se rompe en la shell (MOI-P1 y NORI-V1).
  - Chronos escribe además ">=2.0" fijo en el texto.
- **Para TimesFM, `_import_timesfm` no basta.**
  - Ya entrecomilla el mensaje, pero no traduce el `TypeError` de `paper_url` con huggingface_hub <0.30 (TV-1).
  - TimesFM 2.5 sin torch da un `AttributeError` (T25-F7).
  - La sonda de TimesFM3 tiene una rama muerta con 3.0.2 (TFM3-F3).
- **El compat solo prueba la última versión.** Ningún mínimo declarado se prueba en CI (TAB-F19).
- **Dónde arreglarlo.** Un helper común en `_utils.py`, por ejemplo `_require_backend(adapter_cls, distribution, min_version, extra_hint)`. Lo llamaría `_load_model` de cada adaptador; leería la versión con `importlib.metadata`, distinguiría "no instalado", "versión antigua" y "falta el runtime o el extra", y entrecomillaría siempre `pip install "..."`.
  - `packaging` está disponible de forma transitiva porque optuna, dependencia core, declara `packaging>=20.0` (probado). Aun así, conviene una comparación numérica propia, como `version_key` de run.py, para no depender de una dependencia transitiva (corrección: `skforecast/utils/utils.py` ya importa `packaging` en ejecución, no hace falta una comparación propia).
  - Con el helper, `_TimesFMCompat` y la lógica ad hoc de TimesFM3 desaparecen o se simplifican.

### XC-3. Los prefijos del registro aceptan ids que no son forecasters cargables [probado]
`probe_shared.log`, sección 3. Todos estos ids resuelven a un adaptador:
- `Synthefy/Nori-Causal` a Nori: es un modelo causal y falla con un 404 al predecir (NORI-F5).
- `theforecastingcompany/t0-alpha-onnx-int8` a T0: falla con un OSError engañoso (T0-F7).
- `google/timesfm-2.5-200m-flax` y `-transformers` a TimesFM25: 404, o fallo después de descargar 925 MB (T25-F8).
- `google/timesfm-3.0-flax` a TimesFM3: no existe en el Hub.
- `soda-inria/tabicl-v1-does-not-exist`, `priorlabs/tabpfn-v2` y `taharnbl/TS-ICL-v2`: el `model_id` es solo una etiqueta y se cargan los pesos fijos (TAB-F7, TPF-F12, TSI-F11).

**Dónde arreglarlo.** En `_resolve_adapter`: una tabla `_UNSUPPORTED_MODEL_IDS` (patrón y motivo) que se consulta antes de los prefijos, o prefijos exactos. La entrada de licencia sigue funcionando por prefijo más largo (verificado en `_get_model_license`). Para TabICL, TabPFN y TS-ICL basta con documentarlo.

### XC-4. Ocultar la barra tqdm redirigiendo stderr oculta también los warnings del backend [probado por TPF y TAB; verificado en código]
- `_adapters.py` 2512 (TabICL) y 3176 (TabPFN) usan `contextlib.redirect_stderr(io.StringIO())` con `show_progress=False`, que es el valor por defecto.
- Los warnings se siguen emitiendo, pero `showwarning` escribe en el StringIO, así que el usuario no ve, por ejemplo, el aviso de NaN en las covariables futuras (corrección TPF-F6).
- **Dónde arreglarlo.** Un context manager común, por ejemplo `_suppress_backend_progress()` en `_utils.py`: graba los warnings con `catch_warnings(record=True)` dentro de la redirección y los reemite con `warn_explicit`, conservando fichero y línea, al salir. Hay que tener en cuenta dos matices de los verificadores:
  - Sin `registry`, el warning se repite en cada fold del backtesting.
  - El test debe comprobar stderr o `showwarning`, no `pytest.warns`, que hoy ya pasa.

### XC-5. Cuantiles: ni la normalización de tipos ni la validación contra SUPPORTED_QUANTILES están centralizadas [probado]
- **Tipos.** `FoundationModel.predict` solo comprueba `0 <= q <= 1` (`_foundation_model.py` 1135-1145) y pasa los niveles tal cual. `[0, 0.5, 1]` llega al backend como `int` y los `np.float32` como `float32` (sección 2 del log).
  - TabICL los rechaza (TAB-F10).
  - Los float32 generan además columnas `q_0.10000000149011612`.
- **Validación.** `_validate_supported_quantiles` se llama dentro de TimesFM25, TimesFM3 y Moirai. TS-ICL no valida, y 8 de sus niveles declarados fallan en el backend (TSI-F7).
- **Dónde arreglarlo.** En FoundationModel:
  - pasar al adaptador `float(q)`, sin cambiar los nombres de columna que pidió el usuario;
  - validar contra `adapter.SUPPORTED_QUANTILES` cuando no es None, una sola vez;
  - quitar las llamadas por adaptador.

  La traducción `Decimal` de TS-ICL (TSI-V2) sería, en cambio, un detalle propio del adaptador.

### XC-6. La documentación promete que la predicción de una serie no depende de las otras del lote, y no es cierto en cuatro backends
Falla en:
- **T0:** el relleno se lee como MISSING en vez de PAD, y una fila toda NaN de covariables no equivale a no tenerla (T0-F4/F5, probado).
- **TS-ICL:** con longitudes distintas se pierden las covariables futuras, y un NaN en la exog de una serie afecta a otra (TSI-F4/F5, probado).
- **TimesFM 3.0:** cada trozo de `per_core_batch_size` se rellena hasta la serie más larga (TFM3-F5, probado).
- **Chronos con `cross_learning=True`:** el resultado depende de `batch_size` (CHR-F6; upstream lo documenta).

Textos afectados:
- El docstring de `FoundationModel.predict` (1114: "never depends on the exog columns of the other series").
- `tools/ai/llms-base.txt` 531.
- La sección de exog de `skills/foundation-forecasting/SKILL.md`.
- `references/adapter-parameters.md` 82 ("predict(levels=[x]) equals predict() filtered").
- Celdas 2 y 21 de la guía de series heterogéneas (TSI-V1).
- El comentario de `_adapters.py` 3718 sobre T0.

**Dónde arreglarlo.** Las capacidades se corrigen con atributos y FoundationModel: un atributo nuevo de agrupación (TSI-P1), `supports_heterogeneous_covariates=False` en T0 y `trim_leading_missing` en T0. Los textos deben distinguir "no recibe la exog de otras series", que FoundationModel sí garantiza con la agrupación, de "la salida es idéntica a predecirla sola", que depende del backend.

### XC-7. Los tests no detectan las divergencias del backend [verificado en código]
- Las fakes divergen del backend real en T0, TimesFM25, Chronos, TabICL, TabPFN, Nori y TS-ICL.
- `test_smoke.py` usa `STEPS = 12` y solo comprueba forma y finitud.
- Ninguno de los defectos T0-F1 (en unit), TSI-F2/F4/F7/F8, T25-F6 ni TAB-F10 pone el CI en rojo.
- **Dónde arreglarlo.** Invariantes genéricas en `tools/foundation_compat/tests/test_smoke.py`, activadas según los atributos del adaptador:
  - una serie predicha sola da lo mismo que dentro de un lote de longitudes distintas, con y sin exog;
  - con cuantiles desordenados se respeta el orden;
  - la predicción puntual es igual a q0.5 cuando el adaptador lo declara;
  - si hay exog futura, la exog cambia la predicción;
  - horizonte largo (más de 128 pasos para TimesFM 2.5).

  Hay que acercar además cada fake a la firma real.

### XC-8. Los `stacklevel` fijos apuntan a líneas internas [probado en Moirai; verificado en código el patrón]
- `_warn_if_non_commercial` usa `stacklevel=3` (`_utils.py` 439) y se llama desde la carga perezosa, así que en Moirai apunta a `_adapters.py:2069` (MOI-V1).
- Hay más valores fijos: 3 (TabICL, TabPFN, Nori), 5 y 6 (Timestral) y 7 (fallback MPS de Moirai, que falla por un marco vía FoundationModel; MOI-F13).
- **Dónde arreglarlo.** Un helper tipo `find_stack_level`, que recorra los marcos hasta el primero fuera de `skforecast/`, usado por estos avisos. `skip_file_prefixes` exige Python 3.12, y el mínimo de skforecast es 3.10. Prioridad baja.

### XC-9. `_get_future_timestamps` está duplicado y avisa con numpy 2.5 [probado]
- `pd.Timestamp("2000-01-01") + pd.Timedelta(days=n)` emite un `DeprecationWarning` con numpy 2.5.4 y pandas 2.3.3 (sección 5 del log). Está en TabICL (2631) y TabPFN (3305).
- `pd.date_range("2000-01-01", periods=n + steps, freq="D")[n:]` no avisa y da las mismas fechas (probado).
- **Dónde arreglarlo.** Un helper común en `_utils.py` para los dos adaptadores (TPF-P5 y TPF-V4).

### XC-10. `_resolve_torch_device` y la elección de dispositivo [verificado en código]
- La usan TimesFM3 (1702), Moirai (2081, con fallback), T0 (3804), TSICL (4252, **sin** el `mps_fallback_message` aunque el backend cae a CPU en silencio; TSI-F9) y Timestral (con fallback).
- Chronos pasa `device_map="auto"` directo a transformers y accelerate, que no equivale a "CUDA > MPS > CPU" (CHR-F7).
- Nori, TabICL y TabPFN delegan la elección en el backend, con un orden coherente con sus docs.
- No hace falta cambiar el helper. Solo dos cosas: añadir el aviso en TSICL (TSI-P5) y documentar o resolver el caso de Chronos (CHR-P6).
- CUDA y MPS no se han podido ejecutar: no hay GPU en la máquina.

### XC-11. Licencias [verificado en código, probado por cada adaptador]
- Las entradas de `_MODEL_LICENSES` coinciden con las model cards en los nueve adaptadores revisados, y los atributos de autenticación y de pesos son veraces. No hay que cambiar ninguna entrada.
- Solo hay matices de texto:
  - TimesFM 3.0 es "no comercial y no de producción" (TFM3-P5).
  - La guía no menciona la licencia CC-BY-NC de Moirai (MOI-F6).
  - La guía muestra un LicenseWarning de TabPFN anterior a 1.3 (TPF-F17).

### XC-12. Infraestructura de `tools/foundation_compat` [verificado en código]
- **Python fijo en 3.12.** `run.py --python` vale 3.12 por defecto y el workflow usa 3.12. Es la única versión que sirve a la vez para uni2ts (necesita Python <=3.12) y tsicl (>=3.12). Subir el valor por defecto rompería Moirai sin aviso. Conviene una clave opcional `python` por entrada en `adapters.toml`.
- **README desfasado.** La frase "tfc-t0 and timesfm ship an MLX runtime with other signatures" ya no es cierta: los dos runtimes MLX tienen la misma API (T0-F13, TFM3-F9).
- **Caché de TabPFN.** El workflow solo cachea `~/.cache/huggingface`, y TabPFN guarda los pesos en `~/.cache/tabpfn` (TPF-F11, TPF-P7).
- **Issue #1385 desfasada.** Menciona `--variant latest` y la clave `pinned`, que ya no existen en run.py ni en adapters.toml.

## 3. Lista final de propuestas, por prioridad

Las ids son las originales. Las propuestas transversales llevan el id XC y, entre paréntesis, las ids que fusionan. Una "D" remite a una decisión de la sección 5.

### Alta
1. **T0-P1, T0-P7 y T0-P6** (API 0.6, extra `[torch]`, mínimo de versión, fakes, contrato y docs). Rotura real con la instalación por defecto. **Bloqueada:** la está haciendo otra persona (#1385 sigue abierta y no hay PR). D6.
2. **XC-1 (CHR-P1, MOI-P5, TV-2):** comprobar en FoundationModel que la ventana de contexto tenga algún valor observado. Hoy hay predicciones absurdas sin aviso en al menos cinco backends. D1.
3. **TSI-P1:** atributo de capacidad nuevo para que FoundationModel prediga una serie por llamada cuando hay exog, o agrupe por longitud. Hoy se pierde la exog en silencio en multiserie. D2.
4. **TSI-P2:** `supports_past_only_covariates=False` en TS-ICL. El atributo es falso y no hay aviso. Hay que actualizar también `test_check_exog_columns.py`, `test_get_model_info` y la guía de series heterogéneas. D3.
5. **T0-P3 y T0-P2** (complementarios, no alternativos: TV-3 de T0): `supports_heterogeneous_covariates=False` y `trim_leading_missing=True`; con P3 se simplifica `_build_future_covariates`. Es la invariancia de lote de T0. P2 depende de T0-P1. D8.

### Media
6. **XC-2:** helper común de versión mínima y de mensaje de instalación entrecomillado, más la decisión de mínimos por backend. Fusiona T25-P1 (incluida la eliminación de `_TimesFMCompat`), T25-P5, TV-1, TFM3-P1, CHR-P2, MOI-P1, TAB-P3, NORI-P3 y NORI-V1, TPF-P8 y la parte (b) de TSI-P4. D6.
7. **CHR-P3:** declarar `cross_learning` en el contrato de `Chronos2Pipeline.predict`. No necesita decisión.
8. **XC-5 (TAB-P2, TSI-P3):** normalizar los cuantiles a `float` y validar SUPPORTED_QUANTILES de forma central. Para TS-ICL hay que elegir entre traducir a `Decimal` y filtrar la lista. D4.
9. **XC-3 (NORI-P1, T0-P5, T25-P3):** excluir en el registro los ids que no son forecasters. D7.
10. **XC-4 (TPF-P1, TAB-P1):** reemitir los warnings del backend ocultos por la redirección de stderr. Sin decisión.
11. **XC-6 y TFM3-P3:** corregir todos los textos de "independencia entre series", junto con TSI-V1 y TSI-P7.
12. **TSI-P4:** guard con ValueError claro para `steps > 672` con exog futura o múltiplo de 672 mientras tsicl sea <=0.2.1. D5.
13. **T0-P4:** `torch_dtype` mediante `from_pretrained(dtype=...)`. Hoy falla siempre. Debe ir en el PR de #1385 o justo después, coordinado.
14. **TFM3-P2:** corregir el docstring de `padding_mode` y, opcionalmente, permitir cambiarlo. D11.
15. **CHR-P4:** enviar un `pd.Categorical` como categórico aunque sus categorías sean números. Cambia predicciones solo en ese caso. D9.
16. **NORI-P2:** conservar las filas de contexto con covariables NaN. No cambia el MAE de la guía. D10.
17. **TPF-P2:** dar control sobre `compile` y las opciones nuevas de TabPFNTSPipeline. No verificable sin GPU. D12.
18. **T25-P2:** "media" pasa a "mediana (q0.5)" en la guía, el docstring y la skill.
19. **TPF-P4:** volver a ejecutar las celdas de TabPFN-TS de la guía con `TABPFN_TOKEN`. Necesita el token del mantenedor.
20. **XC-7 (TSI-P8, T25-P6, NORI-P7, NORI-P8, TAB-P6, TPF-P9):** invariantes en el smoke y fakes fieles al backend. Para TabPFN, usar el checkpoint público TabPFN-v2-reg (TPF-V3) o la comparación de features, no el ModelSpecs aleatorio.

### Baja
21. **XC-9 (TPF-P5, TPF-V4):** helper común de timestamps sintéticos.
22. **XC-8 (MOI-P6, MOI-V1):** helper de `stacklevel`.
23. **XC-12:** clave `python` por entrada, frase de MLX en el README, caché de TabPFN (TPF-P7) y entrada de Nori-1-5 en compat (NORI-P7, que también hay que añadir a la matriz del workflow).
24. **TSI-P5:** aviso de caída a CPU en MPS. **TSI-P6:** aviso si el contexto supera 4096.
25. **MOI-P2:** no recargar los pesos con `set_params(context_length, device)`. **MOI-P3:** guardarraíl de la rejilla de cuantiles del checkpoint.
26. **T25-P4** (y window_size): validar `forecast_config_kwargs`. **TV-3:** avisar si un modelo precompilado tiene `max_context` menor que `context_length`.
27. **NORI-P5:** quitar los workarounds redundantes (orden de cuantiles y conversión de torch) y adaptar los tests.
28. Docs de cada adaptador no cubiertos por otros PR: CHR-P5, CHR-P7, CHR-V3; MOI-P4 y MOI-V2 (contexto "sin tope duro, el checkpoint declara 8192"); TAB-P4, TAB-P5, TAB-V2 y TAB-V1 (`context_length` cuenta filas, NaN incluidos); TPF-P3; NORI-P4 y NORI-V2; TFM3-P5; T25-P7 y T25-P8; CHR-P6 (a).
29. **CHR-P8:** tests de Chronos, incluido `cross_learning` dentro de `predict_kwargs`.
30. Decisiones de producto que se recomienda aplazar: T0-P8 (en su forma corregida), TPF-P6, NORI-P6, NORI-P9, NORI-P10, MOI-P7, TFM3-P4 y una política común de `revision`/`model_kwargs` (TFM3-F10, T0-F12, NORI-F7). D13 y D14.

### Descartado o fuera de alcance
- Todas las cifras de MAE, tiempos, bf16 o fp32 y `batch_size` entre variantes o versiones: TFM3-P2 opción (b), las medidas de T0 y TimesFM3 de una pasada anterior (retiradas), la ForecastConfig "precisa" de TimesFM 2.5 y los ajustes recomendados de T0 sin medir.
- **Corregidas por los verificadores** (se usan en su forma corregida):
  - T0-P8, parte de covariables: `[B, V, T]` predice las variables como objetivos conjuntos, no como covariables.
  - TFM3-P3: la frase falsa solo está en `adapter-parameters.md`.
  - TFM3-F10: `use_sdpa` no tiene efecto; sí cuenta `force_download`.
  - T25-P1: TimesFM3 no declara `>=3.0` en `backend_package`.
  - NORI-P6: el riesgo para columnas numéricas no existe.
  - TSI-F7: el rechazo de niveles sí se puede sortear (`Decimal`).
  - TPF-P1 y TPF-P9: cómo probarlos.
  - TAB-P3: un mínimo en `backend_package` no protege sin una comprobación en ejecución.
  - TSI-F12: la licencia no tiene cambios materiales.
- **No verificable aquí:** el comportamiento en CUDA y MPS (TPF-F5, TPF-P2, CHR-P6 (b), T0-F13, TFM3-F9, TSI-F9); TabPFN-3.5 con los pesos reales (TPF-P6 y el smoke local); el arreglo de main de tsicl (PR #1, no ejecutado); MLX; los papers en arXiv.

## 4. Pull requests propuestos (base `0.27.x`, en orden)

Las pruebas comunes a todos los PR son estas:
- La skill `verify`, que con `python .claude/skills/verify/affected.py` da los ficheros afectados.
- `pytest skforecast/foundation/tests/tests_foundation_models/<ficheros tocados> -q`, y la carpeta completa una vez al final.
- `python tools/ai/generate_ai_context_files.py --check` cuando cambian docs o la skill.
- Una entrada en `docs/releases/releases.md`, en la sección de 0.27.0.

Para el compat: `~/.venvs/fm-review-<key>/bin/python -m pytest tools/foundation_compat/tests/test_contract.py tools/foundation_compat/tests/test_smoke.py --backend <key> -q`, o `python tools/foundation_compat/run.py --adapter <key>`.

| # | Rama | Título | Propuestas | Ficheros | Tests | ¿Rompe? | Depende de / bloqueo |
|:--|:--|:--|:--|:--|:--|:--|:--|
| PR-0 | (de otra persona) | T0Adapter con tfc-t0 >=0.6 (#1385) | T0-P1, T0-P7, T0-P6; sugerir incluir T0-P4 y la parte T0 de XC-2 | `_adapters.py` (T0), `fixtures_adapters.py`, `test_T0Adapter.py`, `test_get_model_info.py`, `contracts.py`, `adapters.toml`, guía (celdas 11 y 84), `SKILL.md` 61/283, `llms-base.txt` 458/494, docstring de FoundationModel | `test_T0Adapter.py`; contract y smoke `--backend t0` | Sí: pide `tfc-t0[torch]>=0.6` | **Bloqueado** por el trabajo en curso; D6 (tope `<0.7`) |
| PR-1 | `fix/foundation-all-nan-context` | Rechazar las ventanas de contexto sin valores observados | XC-1 (CHR-P1, MOI-P5, TV-2) | `_foundation_model.py`, `_validation.py` (comentario), `test_predict.py`, `test_backtesting_foundation.py` (no debe cambiar), guía heterogénea celda 2, `llms-base.txt`, `SKILL.md` | `test_predict.py`, `test_fit.py`, `skforecast/model_selection/tests/tests_validation/test_backtesting_foundation.py` | Sí, solo para casos que hoy dan valores absurdos | **D1** |
| PR-2 | `fix/foundation-quantile-levels` | Normalización y validación central de los niveles de cuantil | XC-5 (TAB-P2, TSI-P3), MOI-P3 | `_foundation_model.py`, `_adapters.py` (quitar validaciones duplicadas; TSICL, Moirai), `_utils.py`, tests de predict, TSICL, Moirai, TimesFM y TabICL, `SKILL.md` 159, `llms-base.txt` 492/496 | `test_predict.py`, `test_TSICLAdapter.py`, `test_MoiraiAdapter.py`, `test_TabICLAdapter.py`, `test_TimesFM*Adapter.py`; smoke de tsicl, tabicl y moirai | No | **D4** (TS-ICL) |
| PR-3 | `fix/foundation-tsicl-capabilities` | TS-ICL: agrupación central y capacidades veraces | TSI-P1, TSI-P2, TSI-P4, TSI-P5, TSI-P6, TSI-P7, TSI-V1 | `_adapter_base.py` (atributo nuevo en `_REQUIRED_CLASS_ATTRIBUTES`, así que todos los adaptadores lo declaran), `_foundation_model.py`, `_adapters.py`, `test_AdapterBase.py`, `test_predict.py`, `test_TSICLAdapter.py`, `test_check_exog_columns.py`, `test_get_model_info.py`, guía heterogénea (celdas 2 y 21), guía principal (celda 11), skill y `llms-base.txt` | Los anteriores y la carpeta `tests_foundation_models/`; smoke `--backend tsicl`; `check_covariates_rollout.py` (E3/E4) en `~/.venvs/fm-review-tsicl` | Cambia el contrato de `_AdapterBase` (atributo nuevo) y corrige predicciones multiserie con exog | **D2, D3, D5**; después de PR-2 (mismo adaptador) |
| PR-4 | `fix/foundation-backend-minimums` | Helper común de versión mínima y mensajes de instalación | XC-2 (T25-P1, T25-P5, TV-1, TFM3-P1, CHR-P2, CHR-P3, MOI-P1, TAB-P3, NORI-P3, NORI-V1, TPF-P8); T0 se engancha en PR-0 o justo después | `_utils.py`, `_adapters.py` (`_load_*` de Chronos, TimesFM, Moirai, TabICL, Nori, TabPFN y TSICL; eliminar `_TimesFMCompat`), `adapters.toml`, `contracts.py`, `test_*Adapter.py`, `test_get_model_info.py`, `test_backends_compat_config.py`, guía (celda 11), `SKILL.md`, `llms-base.txt` | Unit de cada adaptador tocado; contract y smoke de chronos, timesfm25, timesfm3, moirai, tabicl, nori y tabpfn; prueba manual con versiones antiguas (envs de revisión `-old`) | Sí, para quien tenga un backend por debajo del mínimo | **D6**; no tocar T0 hasta cerrar PR-0 |
| PR-5 | `fix/foundation-registry-unsupported-ids` | Rechazar al resolver los ids que no son forecasters | XC-3 (NORI-P1, T0-P5, T25-P3); docs de TAB-P4 | `_adapters.py` (registro y `_resolve_adapter`), `test_resolve_adapter.py`, `llms-base.txt` y la columna de prefijos de la skill | `test_resolve_adapter.py`, `test_list_adapters.py`, `test_utils.py` (cobertura de `_MODEL_LICENSES`) | No (esos ids ya fallan) | **D7**; la parte T0, conviene después de PR-0 |
| PR-6 | `fix/foundation-tabular-adapters-warnings` | TabICL y TabPFN: reemitir warnings ocultos y timestamps sin deprecación | XC-4 (TPF-P1, TAB-P1), XC-9 (TPF-P5, TPF-V4), TPF-P3, fakes de TAB-P6 y TPF-P9 | `_utils.py`, `_adapters.py` (TabICL, TabPFN), `fixtures_adapters.py`, `test_TabICLAdapter.py`, `test_TabPFNAdapter.py` | Esos dos ficheros; smoke tabicl (el smoke de tabpfn necesita token) | No | Ninguna |
| PR-7 | `fix/t0-batch-invariance` | T0: el relleno como PAD y covariables agrupadas | T0-P2, T0-P3, simplificación de `_build_future_covariates`, comentario de 3718, T0-F5 en la skill | `_adapters.py` (T0), `test_T0Adapter.py`, `fixtures_adapters.py`, `SKILL.md` 130, `adapter-parameters.md` | `test_T0Adapter.py`; smoke `--backend t0`; `compare_direct.py` y `options_060.py` (comprobaciones 2 y 3) | No (corrige predicciones multiserie) | **Después de PR-0** (`trim_leading_missing` exige >=0.6); **D8** |
| PR-8 | `fix/chronos-adapter-alignment` | Chronos-2: categóricas con códigos numéricos y docs | CHR-P4, CHR-P5, CHR-P6 (a), CHR-P7, CHR-V3, CHR-P8 | `_adapters.py` (Chronos), `test_ChronosAdapter.py`, guía (celdas 0 y 11), skill, `llms-base.txt` 488, docstring de FoundationModel 152 | `test_ChronosAdapter.py`; smoke chronos; `backtest_guide.py` antes y después (ya hecho: MAE 171.267 en ambos) | Cambia predicciones solo con exog category numérica | **D9** |
| PR-9 | `fix/nori-adapter-alignment` | Nori: covariables NaN, Nori-1-5 y limpieza | NORI-P2, NORI-P4, NORI-P5, NORI-P8, NORI-V2 | `_adapters.py` (Nori), `fixtures_adapters.py`, `test_NoriAdapter.py`, docstring de FoundationModel, `adapter-parameters.md` | `test_NoriAdapter.py`; smoke nori; `patch_keep_nan_covariates.py` (MAE 185.9507 antes y después) | Cambia predicciones solo con covariables NaN en el contexto | **D10**; mínimo 0.22 vía PR-4 |
| PR-10 | `fix/timesfm-adapters-alignment` | TimesFM 2.5 y 3.0: kwargs reservados, `padding_mode` y docs | T25-P2, T25-P4, TV-3, T25-P7, T25-P8, TFM3-P2, TFM3-P3, TFM3-P5 | `_adapters.py` (TimesFM), `test_TimesFM25Adapter.py`, `test_TimesFM3Adapter.py`, guía (celda 11), skill, `llms-base.txt` | Esos tests; smoke timesfm25 y timesfm3 | No con la opción (a)/(c) de D11 | **D11**; después de PR-4 (mismo `_load_model`) |
| PR-11 | `fix/moirai-adapter-alignment` | Moirai: no recargar pesos con set_params y docs | MOI-P2, MOI-P4, MOI-V2 | `_adapters.py` (Moirai), `test_MoiraiAdapter.py`, guía (celdas 11 y 65), `SKILL.md` 168/284, `adapter-parameters.md` 218-231 | `test_MoiraiAdapter.py`, `test_set_params.py`; smoke moirai | No | Después de PR-2 (MOI-P3) |
| PR-12 | `chore/foundation-compat-checks` | Invariantes en el smoke y ajustes de la infraestructura de compat | XC-7 (TSI-P8, T25-P6, NORI-P7), XC-12 (python por entrada, README de MLX, caché TPF-P7) | `tools/foundation_compat/**`, `.github/workflows/foundation-backends.yml`, `test_backends_compat_config.py` | `test_backends_compat_config.py`; `run.py --adapter <cada uno>` | No | Después de PR-3 y PR-7, o con `xfail` documentado para T0 y TS-ICL; las invariantes fallarían hoy |
| PR-13 | `chore/foundation-warning-stacklevel` | Helper de stacklevel para los avisos de licencia y de dispositivo | XC-8 (MOI-P6, MOI-V1) | `_utils.py`, `_adapters.py` (todas las llamadas con `stacklevel` fijo), `test_device_handling.py`, `test_utils.py` | Esos tests y la carpeta | No | Mejor al final (toca muchos adaptadores) |
| PR-14 | `docs/foundation-guide-tabpfn-rerun` | Volver a ejecutar las celdas de TabPFN-TS de la guía | TPF-P4 | guía (celdas 78-83 y 108) | Comprobaciones de docs de `verify` | No | Necesita `TABPFN_TOKEN` (mantenedor) |

Orden recomendado: PR-1, PR-6, PR-2, PR-3, PR-5, PR-4, PR-8, PR-9, PR-11, PR-10; después, PR-0 seguido de PR-7; al final PR-12, PR-13 y PR-14.
- PR-1, PR-2 y PR-6 no dependen de ninguna decisión difícil, salvo D1 y D4.
- PR-0 no depende de nadie de esta lista; PR-7 va detrás de él.
- Los PR que tocan `_adapters.py` chocarán entre sí en zonas cercanas: conviene fusionarlos en serie, no en paralelo.

## 5. Decisiones del mantenedor

**D1. Ventana de contexto sin valores observados (XC-1).**
- Opciones:
  - (a) ValueError que nombre las series.
  - (b) Warning y predicción NaN para esas series, como hace el backtesting al saltarlas.
  - (c) Solo documentarlo.
- **Recomendación: (a).** Es coherente con el rechazo actual de series enteramente NaN y no cambia `backtesting_foundation`.

**D2. Política de independencia entre series (TSI-P1 y XC-6).**
- Opciones:
  - (a) Atributo de agrupación por firma de exog y longitud de contexto ya recortado.
  - (b) Atributo para predecir una serie por llamada cuando hay exog.
  - (c) Usar `requires_equal_length_context` (descartada: altera la serie rellenada entre 0.08 y 0.23).
  - (d) Resolverlo en el adaptador (contrario a las reglas del módulo).
- **Recomendación: (b) para TS-ICL**, en FoundationModel, revisable cuando upstream cierre #8 y #10. TimesFM 3.0 se queda en documentarlo: aplicarle el atributo cambiaría predicciones y multiplicaría llamadas, así que sería una decisión aparte.

**D3. `supports_past_only_covariates` de TS-ICL.**
- Opciones:
  - (a) False hasta que upstream publique #4 y #6.
  - (b) `allow_covar_forecast=True`: no cubre el caso mixto y arrastra #21.
  - (c) Dejarlo y documentarlo.
- **Recomendación: (a).**

**D4. Niveles de cuantil de TS-ICL.**
- Opciones:
  - (a) Filtrar SUPPORTED_QUANTILES.
  - (d) Traducir a `Decimal(str(q))` en el adaptador, con un test que fije el comportamiento.
  - En ambos casos, reportarlo upstream.
- **Recomendación:** (d) más el reporte upstream; (a) si no se quiere depender de un detalle interno del backend.

**D5. Horizonte de TS-ICL por encima de 672 pasos.**
- Opciones:
  - (a) Guard y documentación ahora.
  - (b) Esperar la release de upstream y subir el mínimo.
- **Recomendación: (a) ahora y (b) cuando salga la release.**

**D6. Mínimos de versión de los backends y si se comprueban en ejecución.**
- **Recomendación general:** el helper XC-2 con comprobación en ejecución, sin topes superiores salvo en T0, donde se puede valorar `<0.7` (la decisión es de quien lleva #1385).
- Por backend:

  | Backend | Opciones | Recomendación | Motivo |
  |:--|:--|:--|:--|
  | tfc-t0 | `[torch]>=0.6` con o sin tope `<0.7` | `>=0.6`; el tope lo decide #1385 | - |
  | timesfm para 2.5 | `>=3.0` o `>=2.0.1` | `>=3.0` | Cuantiles AR incorrectos en 2.0.x; misma familia que TimesFM3 |
  | chronos-forecasting | `>=2.2` o `>=2.3` | `>=2.3` | 2.2.x fija transformers<5 (CHR-V1) e incluye la corrección de bf16 (lo propuesto por el investigador era >=2.2) |
  | uni2ts | `>=2.0` más documentar Python <=3.12 y el conflicto de pip | `>=2.0` con esa documentación | - |
  | tabicl | `>=2.0.3` o `>=2.2` | `>=2.2` | La CI solo prueba esa; 2.0.2 estuvo una semana |
  | synthefy-nori | `>=0.22` o sin mínimo | `>=0.22` | - |
  | tabpfn-time-series | mantener `>=1.3` o subir a `>=1.4` | `>=1.4` | Con 1.3 y varias GPU, los cuantiles personalizados se sustituyen por los por defecto (TPF-V1), y 1.3 permite tabpfn 9.0 sin #1310 y #1215 |

- Sub-decisión: si se añade un carril de compat con la versión mínima. Recomendación: sí, solo para el nivel de contrato.

**D7. Ids no soportados en el registro.**
- Opciones:
  - (a) Tabla central de exclusiones con su motivo.
  - (b) Prefijos exactos.
  - (c) Solo documentarlo.
- **Recomendación: (a).** Para TimesFM 2.5 basta añadir `-flax` y `-transformers` a la tabla.

**D8. Invariancia de lote en T0.**
- Opciones: aplicar P2 (`trim_leading_missing`) y P3 (heterogeneous False), solo una de las dos, o documentarlo.
- **Recomendación: aplicar las dos.** Son complementarias.

**D9. Categóricas con códigos numéricos en Chronos.**
- Opciones: (a) tratarlas como categóricas; (b) documentar el comportamiento actual.
- **Recomendación: (a).** Coincide con `predict_df` de upstream y requiere nota en la release.

**D10. Covariables NaN en Nori.**
- Opciones: (a) conservar esas filas; (b) mantener el filtrado y corregir solo el docstring.
- **Recomendación: (a).**

**D11. `padding_mode` en TimesFM 3.0.**
- Opciones:
  - (a) Mantener `"edge"` y corregir el docstring.
  - (b) Pasar a `"none"`, que cambia predicciones.
  - (c) Permitir que el usuario lo cambie (sacarlo de `_RESERVED_PREDICT_KWARGS` con `pop`).
- **Recomendación: (a) y (c).**

**D12. Opciones de `TabPFNTSPipeline`.**
- Opciones:
  - (A) `pipeline_kwargs` que se reenvía tal cual.
  - (B) Solo `compile`.
  - (C) Documentarlo.
- **Recomendación: (A)**, reenviado solo si no es None.

**D13. Fijar revisión y opciones de carga en todos los adaptadores** (TFM3-P4, T0-F12, NORI-F7).
- Opciones: un `model_kwargs`/`revision` común, o nada.
- **Recomendación: posponerlo y decidirlo para todos a la vez.** Nada por adaptador.

**D14. Decisiones menores; la recomendación en todas es mantener el estado actual o aplazar.**
- Nori: checkpoint por defecto (mantener 6M), categóricas nativas, delegar en `NoriTSForecaster` (no).
- TabPFN: `supports_categorical_covariates` (mantener False hasta probarlo con TabPFN-3.5).
- Moirai: covariables (mantener False).
- TimesFM 2.5: XReg (no implementarlo, documentarlo).
- TabICL: `model_id` (solo documentarlo).
- T0: opciones de inferencia de 0.6 (aplazarlo).
- Chronos: `device_map="auto"` (documentarlo).
- TS-ICL: contexto mayor que 4096 (aviso más docstring).

---

Ficheros relevantes:
- `dev/foundation_adapters_review/evidence/cross_cutting/probe_shared.py`
- `dev/foundation_adapters_review/evidence/cross_cutting/probe_shared.log`
