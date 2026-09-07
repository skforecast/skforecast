# Review: adopción de TimesFM 3.0 en `TimesFMAdapter`

**Rama**: `refactor/timesfm-3.0` (comparada con `0.25.x`, merge-base `9a1033579`)
**Commits relevantes**: `8f1b075ac` (LicenseWarning), `a5a397a46` (TimesFM 3.0), `4bac32a35`, `5b78e311d` (docs)
**Plan**: `dev/PLAN_TimesFM3Adapter.md`
**Alcance de esta revisión**: adopción de TimesFM 3.0 en `TimesFMAdapter` y el mecanismo `LicenseWarning` que la acompaña.

---

## 0. Resumen ejecutivo

El objetivo del cambio es dar soporte a TimesFM 3.0 (`google/timesfm-3.0-pytorch`) manteniendo el camino v2.5, habilitando covariables (exog) para v3.0 y avisando de la licencia no comercial de los pesos. La implementación sigue el plan de forma fiel, el código es legible, los 650 tests del módulo `foundation` pasan, `ruff` no reporta nada y `generate_ai_context_files.py --check` está limpio.

He verificado las suposiciones del adaptador contra el código fuente de `timesfm==3.0.1` (`timesfm3/timesfm3_forecaster.py`, `timesfm3/model.py`) y he ejecutado una prueba en vivo con los pesos reales en el entorno `skforecast_py14` (Apple Silicon, `torch 2.13.0`, MPS disponible). Los detalles de la prueba están en la sección 9.

**Veredicto: no mergear todavía.** Hay dos problemas de corrección reproducibles con la API pública que afectan al caso de uso nuevo (exog), y una decisión de diseño que conviene reconsiderar antes de que la API quede fijada en una release:

| # | Severidad | Hallazgo | Sección |
|---|-----------|----------|---------|
| 1 | **Alta** | Multi-serie con covariables heterogéneas (una serie con exog y otra sin, o distinto número de columnas) explota dentro de `timesfm` con `ValueError: all input arrays must have the same shape`. Reproducido con la API pública. | 1.1 |
| 2 | **Alta** | Cuando una columna de `exog` futuro no tiene histórico en `context_exog`, el adaptador rellena el pasado con NaN y TimesFM 3 lo convierte silenciosamente en una covariable constante (extrapolación de `np.interp`). Se activa con `fit(series)` + `predict(exog=...)` o con columnas distintas entre `fit` y `predict`. Ningún aviso ni error. | 1.2 |
| 3 | Media | El error "TimesFM 3.0 requires `timesfm>=3.0`... upgrade" se emite también cuando `timesfm 3.0.1` está instalado pero falta `torch` (extra opcional `timesfm[torch]`), porque `timesfm/__init__.py` traga el `ImportError`. Las docs recomiendan `pip install timesfm` a secas. | 1.3 |
| 4 | Media | Un `ForecasterFoundation` serializado con 0.24/0.25 y cargado con esta rama falla con `AttributeError: '_backend'` en `predict` y `'device'` en `get_params`. | 5.1 |
| 5 | Media | Las docs de skills afirman que `context_length` recarga el modelo en TimesFM 3.0; el código y el docstring de `bayesian_search_foundation` dicen lo contrario. | 7.1 |
| 6 | Media | `padding_mode="edge"` se presenta como obligatorio con covariables; no lo es, y cambia el punto hasta un 14 % de la escala frente a `"none"` con `steps=12`. El usuario no puede elegir porque la clave está reservada. | 3.2 |
| 7 | Diseño | Dos backends con parámetros, semántica del punto, capacidad de exog y licencia distintos conviven en una clase con `if self._backend == "v3"` repartidos por ocho métodos. El registro por prefijo ya permitiría dos adaptadores. | 2.1 |

El resto son observaciones menores, huecos de tests y detalles de documentación.

---

## 1. Corrección

### 1.1 [Alta] Covariables heterogéneas entre series revientan en el backend

`_predict_v3` ([_adapters.py:1065-1077](../skforecast/foundation/_adapters.py#L1065-L1077)) construye una lista por serie de `past_only` / `past_future` con entradas `None` para las series sin exog, y decide `has_covariates = any(...)` a nivel de lote. El backend (`timesfm3_forecaster.py:688-704`) rellena los `None` con `np.zeros_like(batched_tgt[j])`, es decir, con forma `(1, context_len)`, y luego hace `np.stack` con los arrays reales de forma `(n_cov, context_len + horizon)`. Formas distintas, `ValueError`.

Reproducción (API pública, pesos reales):

```python
series = {"a": y_train, "b": y_train * 1.5 + 10}
fc = ForecasterFoundation(estimator=FoundationModel(model_id="google/timesfm-3.0-pytorch"))
fc.fit(series=series, exog={"a": exog_train})          # exog solo para 'a' (soportado por la API)
fc.predict(steps=12, exog={"a": exog_test})
# ValueError: all input arrays must have the same shape   (timesfm3_forecaster.py:702)
```

Mismo resultado si ambas series tienen exog pero con distinto número de columnas (`{"a": 2 columnas, "b": 1 columna}`). Ambos inputs son válidos según la documentación de `FoundationModel.fit` ("dict: per-series exogenous variables"; "missing series keys filled as None").

Es un bug del backend, pero el adaptador es quien expone la API y quien conoce la restricción. Opciones, de menor a mayor esfuerzo:

1. Validar en `_predict_v3` que todas las series tienen exactamente el mismo conjunto de columnas past-only y known-future, y lanzar un `ValueError` claro ("TimesFM 3.0 requires the same covariates for every series in a batch") antes de llamar al backend.
2. Agrupar las series por "firma" de covariables y llamar a `predict_batch` una vez por grupo. Mantiene la funcionalidad a costa de más pasadas.
3. Para las series sin covariables, rellenar con ceros de la forma correcta (`(n_cov, context_len + horizon)`) y máscara. El backend no expone máscaras desde `predict_batch`, así que estaríamos inyectando covariables falsas a cero; no lo recomiendo.

La opción 1 es suficiente para esta PR; la 2 es el comportamiento deseable a medio plazo.

### 1.2 [Alta] Relleno silencioso con NaN del pasado de una covariable futura

`_build_v3_covariates` ([_adapters.py:1192-1198](../skforecast/foundation/_adapters.py#L1192-L1198)):

```python
past_part = (
    cls._to_covariate_array(ctx_df[col])
    if ctx_df is not None and col in ctx_df.columns
    else np.full(context_len, np.nan, dtype=np.float32)
)
```

El patrón está copiado de `ChronosAdapter` (línea 1196 en 0.25.x), pero la semántica del backend es distinta. Chronos-2 trata NaN como "missing" de forma nativa. TimesFM 3 pasa las covariables por `linear_interpolation` (`timesfm3_forecaster.py`), que usa `np.interp`; para NaN al principio del array, `np.interp` extrapola con el primer valor válido. Resultado: todo el pasado de la covariable se convierte en una constante igual al primer valor futuro. El modelo ve una covariable que fue plana durante todo el contexto y de repente empieza a moverse en el horizonte. No es "missing", es información fabricada.

Verificado en vivo (sección 9, caso 6): `fit(series=y_train)` sin exog seguido de `predict(steps=12, exog=exog_test)` envía al backend un array `(2, 195)` con 366 NaN en la parte histórica, no emite ningún aviso y devuelve una predicción distinta tanto de la de "sin exog" como de la de "con exog completo". Con el caso 7 (fit con `exog_1`, predict con `exog_2`) tampoco hay error ni aviso.

Esto ocurre porque `FoundationModel.predict` no comprueba que las columnas de `exog` coincidan con `exog_names_in_` (a diferencia de `ForecasterRecursive` y compañía, que validan nombres de exog en `predict`). Es un problema preexistente de `FoundationModel`, pero TimesFM 3 es el primer adaptador en el que el resultado es una covariable fabricada en vez de un valor "missing", así que esta PR es el momento de tratarlo.

Recomendación mínima para esta PR: en `_build_v3_covariates`, si una columna de `exog` no está en `context_exog`, lanzar `ValueError` explicando que TimesFM 3.0 necesita el histórico de cada covariable futura (o, si se prefiere no romper, `warnings.warn(..., MissingValuesWarning)` y documentar la extrapolación). Y abrir un issue para validar `exog` contra `exog_names_in_` en `FoundationModel.predict`, como hacen el resto de forecasters.

Un caso relacionado que sí funciona bien: NaN intercalados en el objetivo se interpolan y los NaN iniciales del objetivo recortan también las covariables (`timesfm3_forecaster.py:188-199`). Esto no está documentado en el docstring; merece una línea en Notes.

### 1.3 [Media] Mensaje de error engañoso cuando falta `torch`

`_load_model_v3` ([_adapters.py:1362-1367](../skforecast/foundation/_adapters.py#L1362-L1367)):

```python
if not hasattr(timesfm, "TimesFM3Forecaster"):
    raise ImportError("TimesFM 3.0 requires `timesfm>=3.0`, but the installed version does not provide `TimesFM3Forecaster`. Upgrade with `pip install -U timesfm`.")
```

`timesfm/__init__.py` (3.0.1) hace:

```python
try:
  from timesfm3 import TimesFM3Forecaster, TimesFM3Torch
except ImportError:
  pass
```

y `timesfm3/timesfm3_forecaster.py` importa `torch` a nivel de módulo. `torch` es un extra opcional (`torch>=2.0.0; extra == "torch"`). Un usuario que siga la documentación (`pip install timesfm`, SKILL.md:53, notebook de la guía) en un entorno sin torch recibirá "upgrade timesfm" con `timesfm 3.0.1` ya instalado. Sugerencias:

- Distinguir los dos casos: `importlib.metadata.version("timesfm")` < 3 → "upgrade"; en caso contrario intentar `import timesfm3` directamente y propagar el `ImportError` real (que dirá "No module named 'torch'").
- Cambiar la sugerencia de instalación a `pip install "timesfm[torch]"` en el mensaje y en las docs. El docstring de `_load_model` en 0.25.x ya hablaba de `timesfm[torch]`; la rama lo ha simplificado a `timesfm`.

El mismo silenciamiento afecta al camino v2.5 (`TimesFM_2p5_200M_torch` desaparece del namespace si falta torch y el usuario ve un `AttributeError`), pero eso es preexistente.

### 1.4 [Baja] Detección de backend por subcadena

`_detect_timesfm_backend` ([_adapters.py:507](../skforecast/foundation/_adapters.py#L507)) usa `"timesfm-3" in model_id or "3.0" in model_id`. Verificado:

```
my-org/timesfm-2.5-finetuned-v3.0  -> v3
/models/timesfm-2p5/3.0            -> v3
google/timesfm-3-large             -> v3   (pero no dispara LicenseWarning: el registro usa prefijo "google/timesfm-3.0")
```

Hoy el radio de acción es pequeño porque `_ADAPTER_REGISTRY` solo enruta `google/timesfm*`, pero `TimesFM3Forecaster.from_pretrained` acepta rutas locales y el plan `PLAN_UnifiedFinetunedCheckpoint.md` va en esa dirección. Un regex anclado al patrón `timesfm-(\d+)(?:\.(\d+))?` que use la versión mayor sería más robusto y coherente con el prefijo del registro de licencias. Además, la detección de backend y la de licencia usan lógicas distintas (subcadena vs. prefijo), lo que ya produce la incoherencia de `google/timesfm-3-large`.

### 1.5 [Baja] `set_params(model_id=...)` no re-resuelve `context_length`

El centinela `context_length=None` se resuelve solo en `__init__` ([_adapters.py:699-700](../skforecast/foundation/_adapters.py#L699-L700)). Verificado:

```python
a = TimesFMAdapter(model_id="google/timesfm-2.5-200m-pytorch")   # context_length -> 512
a.set_params(model_id="google/timesfm-3.0-pytorch")               # backend v3, context_length sigue en 512
```

Es coherente con "un int explícito siempre se respeta", pero el adaptador no distingue entre "el usuario pasó 512" y "512 fue el default de v2.5". En `bayesian_search_foundation` con `model_id` en el espacio de búsqueda, v3.0 correría con 512 de contexto en vez de los 2048 documentados. Basta con guardar si el valor fue explícito, o documentar que al cambiar de backend por `set_params` hay que fijar `context_length` a mano.

### 1.6 [Baja] `set_params` en v2.5: `device` recarga un modelo que no usa `device`

`reload_keys` para v2.5 incluye `device` ([_adapters.py:819-821](../skforecast/foundation/_adapters.py#L819-L821)) y el docstring afirma que "for the v2.5 backend `model_id`, `context_length`, `max_horizon`, `forecast_config_kwargs`, and `device` all affect the loaded (and compiled) model". `_load_model_v25` no lee `device`. Es inofensivo (una recarga innecesaria) pero el docstring es falso. Igualmente, `predict_kwargs` se valida contra las claves reservadas de v3 aunque el backend sea v2.5.

### 1.7 [Baja] `stacklevel` de `LicenseWarning`

`_warn_if_non_commercial` usa `stacklevel=3` ([_utils.py:182](../skforecast/foundation/_utils.py#L182)). Desde `_load_model_v3` el aviso apunta a `_adapters.py:1283` (`_load_model`), no al código del usuario. Desde `MoiraiAdapter._load_module` apunta a `predict`. Como el aviso pasa por `rich_warning_handler`, el usuario ve el archivo y línea de skforecast en vez de su llamada. No es grave; si se quiere apuntar al usuario, el nivel correcto varía por adaptador y probablemente convenga aceptar que apunte a skforecast y quitar la pretensión de precisión.

### 1.8 Cosas que he verificado y están bien

- `ForecastOutput.forecast` es la mediana (`raw[:horizon, median_quantile_index]`, índice 4 sincronizado desde el checkpoint) y `quantiles` tiene forma `(horizon, 9)`. El adaptador indexa `[:, q_indices]` correctamente.
- `_match_quantile_indices` contra `self._model.config.quantiles` es lo correcto: `from_pretrained` sobreescribe `config.quantiles` con `model.quantiles` del checkpoint.
- `predict_batch` recorta `[:horizon]`, por lo que `steps` no múltiplo de 64 devuelve `steps` filas. Verificado en vivo con `steps=700` (sección 9).
- Las covariables se detrendan y normalizan por variate dentro del modelo (`model.py:decode`, RevIN por patch), así que la escala de exog no requiere preprocesado. `bool` e `Int64` con NA se convierten bien (`[1., 0., 1.]`, `[1., nan, 3.]`); `object` lanza el `ValueError` esperado.
- 40 covariables known-future funcionan (el `max_variates=32` de `TransformerConfig` no limita la inferencia). Verificado en vivo.
- El gate `steps > max_horizon` queda solo en `_predict_v25`.
- `_apply_set_params` compara valores antes de disparar resets, así que `set_params` con el mismo valor no recarga (hay test).
- `backtesting_foundation(..., suppress_warnings=True)` silencia `LicenseWarning` porque está en `warn_skforecast_categories`. Verificado en vivo.
- `LicenseWarning.__str__` añade la instrucción de supresión, coherente con las otras categorías.

---

## 2. Diseño de la API

### 2.1 Un adaptador para dos backends que no comparten casi nada

Tras el cambio, `TimesFMAdapter` tiene:

- Seis parámetros públicos de los cuales dos son "v2.5 only" (`max_horizon`, `forecast_config_kwargs`) y dos son "v3.0 only" (`device`, `predict_kwargs`). `get_params()` devuelve siempre los seis, así que el `repr` de un `FoundationModel` v3 muestra `max_horizon=512` y `forecast_config_kwargs=None` que no hacen nada, y el de v2.5 muestra `device='auto'` que tampoco.
- `allow_exog` como atributo de instancia que contradice el atributo de clase (`TimesFMAdapter.allow_exog is False` sigue siendo cierto mientras `TimesFMAdapter("google/timesfm-3.0-pytorch").allow_exog is True`). Ningún otro adaptador hace esto.
- Semántica del punto distinta (mediana vs. media, ver 3.1), cuantiles con distinta disposición interna, un backend con compilación y otro sin ella, uno con `LicenseWarning` y otro sin, y `reload_keys` distintos en `set_params`.
- Ocho métodos con ramas `if self._backend == "v3"` (`__init__`, `set_params`, `predict`, `_load_model`, y los cuatro `_*_v3` / `_*_v25`).

Esto es, en la práctica, dos adaptadores pegados con un `_backend`. El plan lo decidió explícitamente ("Decisions locked with the user: dual-support") y entiendo el argumento de mantener un solo prefijo `google/timesfm`. Pero `_resolve_adapter` ya funciona por `startswith` sobre un dict ordenado, y la propia PR introduce un segundo registro (`_NON_COMMERCIAL_LICENSES`) que distingue `google/timesfm-3.0` de `google/timesfm-2.5`. Con dos entradas en `_ADAPTER_REGISTRY` (`"google/timesfm-3": TimesFM3Adapter`, `"google/timesfm": TimesFMAdapter`) se obtendría:

- Superficies de parámetros limpias por clase, sin "v2.5 only / v3.0 only" en cada docstring.
- `allow_exog` como atributo de clase, como en los otros siete adaptadores.
- Ningún `_backend`, ninguna re-detección en `set_params`, ningún `reload_keys` condicional.
- Tests separados por clase en vez de `make_adapter` / `make_v3_adapter` en el mismo archivo.

Si se mantiene la clase única, al menos: (a) `get_params` debería devolver solo las claves relevantes al backend activo, o documentar por qué no; (b) pasar explícitamente un parámetro del otro backend (por ejemplo `max_horizon=100` con un id v3) debería avisar con `IgnoredArgumentWarning`, hoy se ignora en silencio.

Nota sobre `_resolve_adapter`: usa el primer prefijo que coincide en orden de inserción, no el más largo. Si se opta por dos clases, hay que insertar `google/timesfm-3` antes que `google/timesfm` o cambiar a longest-prefix como ya hace `_warn_if_non_commercial`. Sería razonable unificar ambos mecanismos.

### 2.2 `context_length=None` como centinela

Correcto y bien documentado. La firma pública cambia de `int = 512` a `int | None = None`, pero `get_params` devuelve siempre un int, así que `clone` y la serialización no se ven afectados. Ver 1.5 para el efecto en `set_params`.

### 2.3 `predict_kwargs` con claves reservadas

Buena decisión rechazar `padding_mode`, `contexts`, etc. Añadiría `use_symmetric_averaging` a la lista de "claves que conviene documentar con cuidado": con covariables activas, el backend niega también las covariables (`sym_po.append(-po_val)`), lo cual es discutible para covariables binarias o de calendario. No es un bug del adaptador pero el usuario que active esa opción con exog no tiene forma de saberlo.

### 2.4 `device` vs `device_map`

Consistente con `MoiraiAdapter` y `TSICLAdapter`. Bien.

### 2.5 `per_core_batch_size`

No se expone (el plan lo deja como follow-up). `from_pretrained(**kwargs)` lo acepta directamente, así que sería una línea. Ver 4.2 para el impacto.

### 2.6 `LicenseWarning`

El diseño (categoría propia + registro por prefijo + una llamada en cada loader) es limpio y proporcionado. Dos observaciones:

- La nota de la release dice "raised the first time a foundation model ... is loaded". En realidad se emite en cada carga de pesos: cada instancia nueva de adaptador, y en `bayesian_search_foundation` cada trial que cambie `model_id` o `device`. El filtro por defecto de Python deduplica por (mensaje, categoría, módulo, línea) dentro de una sesión, así que en la práctica el usuario lo ve una vez, pero conviene que la frase de la release sea exacta.
- Los nombres y URLs de licencia están hardcodeados en `_utils.py` y pueden quedar desactualizados sin que ningún test lo detecte (las cuatro URLs responden 200 a fecha de hoy). Un comentario con la fecha de verificación ayudaría al siguiente que lo toque.

---

## 3. Corrección estadística

### 3.1 Punto de predicción: mediana en v3, "media" en v2.5

El docstring dice: "the v2.5 point forecast is the mean, while the v3.0 point forecast is the median". Lo segundo está verificado en el código fuente. Lo primero viene de la documentación de TimesFM 2.5 (índice 0 = mean), pero en la prueba en vivo con `google/timesfm-2.5-200m-pytorch` sobre `h2o`, `np.allclose(q_0.5, pred)` devuelve `True`, así que en la práctica el checkpoint 2.5 con `use_continuous_quantile_head` parece devolver la mediana también. No cambia nada en el código, pero recomiendo suavizar la afirmación ("documented as the mean by TimesFM") o verificarla con una serie asimétrica antes de dejarla como hecho en el docstring.

Lo importante: `ForecasterFoundation.predict_interval` construye `pred` a partir de `q_0.5` ([_forecaster_foundation.py:869-882](../skforecast/foundation/_forecaster_foundation.py#L869-L882)). Con v3 el punto de `predict` y el de `predict_interval` coinciden exactamente (verificado en vivo). Con v2.5, si el índice 0 fuese realmente la media, no coincidirían. Es un argumento más a favor de que la mediana sea el punto en ambos, y de que se documente que `predict` y `predict_interval` devuelven el mismo `pred` para v3.

### 3.2 `padding_mode="edge"` cuando hay covariables

El docstring y el plan afirman que `"edge"` es obligatorio ("Without it, an arbitrary `steps` misaligns/errors"). Leyendo `decode` (`model.py:364-`), no es así: si `past_future_covariates` tiene longitud `context + steps`, `decode` infiere `horizon = steps` y rellena el resto de patches con ceros enmascarados (`pf_future_masks` a `True`). Con `"edge"`, el backend repite el último valor futuro conocido hasta completar el múltiplo de 64 **antes** de `decode`, y esas posiciones entran sin máscara.

Diferencia práctica: con `"edge"` el último patch del horizonte mezcla valores reales y repetidos en la misma embedding. Con `"none"` esas posiciones van enmascaradas. Para `steps=12` y patches de 32, hasta 20 posiciones del patch que contiene los 12 pasos reales son valores repetidos. La atención es causal en el eje temporal, pero el patch es la unidad de embedding, así que la elección puede afectar a las predicciones dentro de `steps`. He medido la diferencia en la sección 9 (caso D).

Medido en vivo (sección 9, caso D) con `h2o_exog`, `steps=12`, dos covariables known-future: ambos modos funcionan sin error, la diferencia máxima entre puntos es 0.132 (un 14 % de la escala de la serie, cuyo nivel medio es ~0.95), y contra el test real `edge` da MAE 0.0454 y `none` 0.0493. Con `steps=64` (sin padding interno) ambos coinciden exactamente, y sin covariables también, lo que confirma que la diferencia viene únicamente de las 52 posiciones repetidas.

Conclusiones:

- La afirmación "`padding_mode='edge'` is required whenever covariates are present" del docstring, del plan y de las Notes de `_predict_v3` es falsa. `"none"` funciona para cualquier `steps`.
- La elección no es neutra: un 14 % de diferencia en el punto no es ruido. En este ejemplo `edge` sale mejor, pero es una serie y un horizonte; no hay base para afirmar que sea mejor en general. El usuario debería poder elegir. Recomiendo sacar `padding_mode` de `_V3_RESERVED_PREDICT_KWARGS`, dejar `"edge"` como default (coincide con el uso que hace Google en `predict()`) y documentar en una frase qué hace cada modo.
- Independientemente de la decisión, el docstring debe explicar que `"edge"` repite el último valor futuro conocido hasta el múltiplo de 64 y que esos valores entran en el último patch sin máscara.

### 3.3 Normalización y detrending de covariables

El modelo aplica detrending lineal condicional (`use_linear_detrending`, umbral 0.5) y RevIN por variate a las covariables igual que al objetivo. Para covariables binarias o categóricas codificadas como enteros (lo que el adaptador recomienda hacer "via `transformer_exog`") esto significa que el modelo puede ajustar y extrapolar una tendencia lineal sobre un dummy 0/1. No es un problema del adaptador pero es una advertencia razonable para la guía de usuario: TimesFM 3 trata todas las covariables como continuas.

### 3.4 Mezcla de longitudes de contexto en un lote

`_Query.format` rellena por la izquierda con ceros y máscara `True` las series más cortas del lote; las covariables se rellenan igual. Correcto. El adaptador no necesita hacer nada.

### 3.5 Cuantiles

`sort_quantiles=True` por defecto en el backend garantiza monotonía (verificado en vivo). El adaptador lo deja pasar por `predict_kwargs`, lo cual está bien. Los niveles se validan dos veces (contra `SUPPORTED_QUANTILES` en `predict` y contra `config.quantiles` en `_match_quantile_indices`); la segunda es la que protege contra checkpoints con otra malla y tiene test.

---

## 4. Rendimiento

### 4.1 Camino v3

`_predict_v3` es una sola llamada a `predict_batch` por `predict`; el trabajo en Python (construcción de covariables, `to_numpy`) es despreciable. El backend hace `gc.collect()` al final de cada `predict_batch` (`try_gc`), lo que en un backtesting de muchos folds es un coste fijo por fold. No es del adaptador, pero explica parte del tiempo por fold en CPU.

### 4.2 `per_core_batch_size=4`

Con el default, 64 series son 16 pasadas del transformer por `predict`; en `backtesting_foundation` multi-serie con muchos folds el coste escala linealmente. Medido en CPU con 64 series y `steps=12` (sección 9, caso F): 0.96 s con 4, 0.44 s con 32, 0.34 s con 64. Es decir, entre 2.2x y 2.8x solo por el tamaño de lote. Exponerlo como parámetro del adaptador (`from_pretrained(..., per_core_batch_size=...)`) es trivial y debería estar en el conjunto de `reload_keys`. Nota: no se puede sortear pasando un modelo preconfigurado por `model=` desde `ForecasterFoundation`, porque `ForecasterFoundation.__init__` hace `clone(estimator)` y `get_params` no incluye `model`.

### 4.3 `device="auto"` en Apple Silicon

`_resolve_torch_device("auto")` elige MPS; el default nativo de TimesFM 3 es CUDA o CPU, nunca MPS. Funciona: diferencia máxima MPS vs CPU de 4.8e-07 sobre 12 pasos (sección 9, caso A). Con una serie de 200 puntos no hay ventaja de tiempo (0.08 s vs 0.11 s por `predict`, y la primera llamada es más lenta en MPS por la transferencia de pesos); la ganancia aparecerá con lotes grandes. Merece una línea en el docstring de `device` diciendo que MPS se selecciona automáticamente y cómo forzar CPU.

### 4.4 `context_length` por defecto 2048

Razonable. El límite duro del backend es 15360 (`_MAX_CONTEXT_LENGTH`), redondeado a múltiplo de 32. El docstring dice "roughly 15,360"; podría decir el número exacto y que se redondea al patch.

### 4.5 `set_params(context_length=...)` en v3 no recarga

Correcto y es una mejora real para `bayesian_search_foundation` respecto a v2.5. Pero ver 7.1: las docs dicen lo contrario.

---

## 5. Compatibilidad hacia atrás

### 5.1 [Media] Objetos serializados con versiones anteriores

`ForecasterFoundation` se serializa con `joblib`/`pickle`/`cloudpickle` (`save_forecaster` lo permite explícitamente). Un `TimesFMAdapter` guardado con 0.25.x no tiene `_backend`, `device`, `predict_kwargs` ni `allow_exog` de instancia. Reproducido con un objeto que imita el estado de 0.25.x:

```
old_adapter.predict(...)   -> AttributeError: 'TimesFMAdapter' object has no attribute '_backend'
old_adapter.get_params()   -> AttributeError: 'TimesFMAdapter' object has no attribute 'device'
```

`load_forecaster` avisa con `SkforecastVersionWarning` si la versión no coincide, pero el fallo posterior es un `AttributeError` opaco. Un `__setstate__` que rellene los defaults (`_backend` desde `model_id`, `device="auto"`, `predict_kwargs={}`) cuesta diez líneas y evita el problema. Si la política del proyecto es no garantizar compatibilidad de pickles entre minors, al menos debería constar en la release note como "Breaking".

### 5.2 Cambios de firma y comportamiento observables

- `context_length` pasa de `int = 512` a `int | None = None`. Comportamiento idéntico para v2.5. OK.
- `model_id` no reconocido (`google/timesfm-1.0-*`) ahora lanza `ValueError` en `__init__`; antes fallaba al cargar. Mejora.
- `fit` ahora guarda `context_exog_` también en v2.5 (antes quedaba `None`). Sin efecto visible porque `FoundationModel.predict` descarta exog cuando `allow_exog` es `False`.
- El test `("context_length", None)` se ha eliminado de la parametrización de errores; correcto dado el centinela.

### 5.3 `timesfm` 2.x instalado

El camino v2.5 sigue funcionando con `timesfm<3` (no toca `TimesFM3Forecaster`). Un id v3 con `timesfm<3` da el `ImportError` de "upgrade" (correcto en ese caso; ver 1.3 para el caso incorrecto).

---

## 6. Pruebas

Lo que hay es sólido para el camino feliz: dispatch de backend, `allow_exog`, punto y cuantiles mono y multi-serie, mapeo de cuantiles contra una malla invertida e incompleta, covariables past-only + known-future con formas correctas, `padding_mode`, covariable no numérica, claves reservadas, `set_params` por backend, y los tres escenarios de `_load_model_v3`. Los tests de `LicenseWarning` en Moirai, TabPFN y TS-ICL mockean el paquete real para no descargar nada. Bien.

Huecos que dejarían pasar los bugs de la sección 1:

1. **Ningún test de integración** con un id v3 a través de `FoundationModel`, `ForecasterFoundation` o `backtesting_foundation`. Todos los tests v3 llaman a `adapter.predict` con los helpers `prepare_*`. Los bugs 1.1 y 1.2 solo se ven cuando `FoundationModel.predict` monta `context_exog` y `exog` desde la API pública.
2. **`FakeTimesFM3Forecaster.predict_batch` no hace `np.stack`** de las covariables, así que el fake acepta lotes heterogéneos que el backend real rechaza. Si el fake replicara la validación de formas del backend (tres líneas), el test de 1.1 fallaría hoy.
3. **`last_kwargs` se registra pero no se asserta nunca**: el reenvío de `predict_kwargs` a `predict_batch` está sin test.
4. **`from_pretrained(device=...)`** no se verifica: `FakeTimesFM3Forecaster.from_pretrained` acepta `device` y lo descarta.
5. La rama `np.full(context_len, np.nan)` (columna futura sin histórico) no tiene test. Tampoco el caso `context_exog` como `pd.Series` en vez de `DataFrame`, ni series con exog y series sin exog en el mismo lote.
6. `test_warn_if_non_commercial_uses_longest_prefix_match` no prueba longest-prefix: no hay dos prefijos del registro que compartan raíz, así que el test pasa con cualquier estrategia de matching. Para probarlo hay que parchear el registro con prefijos solapados (`monkeypatch.setitem`).
7. `test_TimesFMAdapter_fit_stores_context_exog` dice "regardless of backend" y solo prueba v3.
8. Los tests que sustituyen `sys.modules["timesfm"]` lo hacen con `try/finally` manual; `monkeypatch.setitem(sys.modules, ...)` es más corto y no deja estado si el test falla a mitad. Estilo, no bloqueante.

---

## 7. Documentación

### 7.1 [Media] Contradicción sobre `context_length` y recarga en v3

El código (`reload_keys = {"model_id", "device"}` para v3) y el docstring de `bayesian_search_foundation` ([_search.py:2153-2156](../skforecast/model_selection/_search.py#L2153-L2156)) dicen que `context_length` **no** recarga el modelo en TimesFM 3.0. Las skills dicen lo contrario en cuatro sitios:

- `skills/foundation-forecasting/SKILL.md:246`: "context_length does the same on TimesFM (2.5 and 3.0)".
- `skills/foundation-forecasting/references/adapter-parameters.md:175`: "Changing these forces a model reload: **all of them**" (falso para `predict_kwargs` en ambos backends y para `context_length`, `max_horizon`, `forecast_config_kwargs` en v3).
- `adapter-parameters.md:187` y `:188`: "It reloads the model on TimesFM (2.5 and 3.0)".
- `skills/hyperparameter-optimization/references/search-parameters.md:136`.

Además `adapter-parameters.md:188` afirma que `TimesFMAdapter` "reset[s] whenever the key is passed, even if the value is unchanged"; `_apply_set_params` compara valores y hay un test (`test_TimesFMAdapter_set_params_no_reset_when_value_unchanged`) que demuestra lo contrario. Esto es preexistente pero la PR ha editado esa misma línea.

Estas skills se inyectan a agentes de IA como fuente de verdad; una contradicción entre skill y código es peor que una omisión.

### 7.2 Instalación

`pip install timesfm` (SKILL.md:53, notebook, mensajes de error) no instala `torch`. Ver 1.3. Debería ser `pip install "timesfm[torch]"` o mencionar torch aparte, como hacía el docstring anterior.

### 7.3 Docstrings

- Ver 3.1 (afirmación "mean" en v2.5) y 3.2 (justificación de `"edge"`).
- `set_params` Notes: ver 1.6.
- `_build_v3_covariates` documenta el relleno con NaN como si fuese inocuo; ver 1.2.
- Falta mencionar que NaN en el objetivo se interpolan y que NaN iniciales recortan contexto y covariables.
- `device`: mencionar MPS y que difiere del default del backend.

### 7.4 Notebook y release notes

- El notebook `foundation-forecasting-models.ipynb` ya está ejecutado con la sección de TimesFM 3.0 y las salidas de `LicenseWarning`; bien. La tabla de modelos sigue con `pip install timesfm`.
- `releases.md`: entradas correctas y enlaces `[TimesFMAdapter]` y `[LicenseWarning]` definidos. Matizar "the first time" (ver 2.6) y añadir la nota de compatibilidad de pickles si no se hace 5.1.
- `AGENTS.md` / `llms-full.txt` regenerados y `--check` limpio.

### 7.5 Ruido en el diff

`dev/PLAN_TinyTimeMixerAdapter.md` (1660 líneas) y `dev/PLAN_UnifiedFinetunedCheckpoint.md` (754 líneas) no tienen relación con esta PR y llegan en la misma rama (commits `c2fe2883e`, `0cb00c35b`, `4ba0fe1e9`). Sugiero sacarlos a su propia rama o, si `dev/` es zona libre, al menos no mezclarlos en el mismo merge para que `git log` de la feature quede limpio.

---

## 8. Mantenibilidad

- `_adapters.py` pasa de ~3.900 a ~4.500 líneas. `TimesFMAdapter` sola son ~900. Ver 2.1: dos clases serían más fáciles de leer que una con ocho ramas.
- Hay ahora dos registros por prefijo con semánticas distintas: `_ADAPTER_REGISTRY` (primer match, en `_adapters.py`) y `_NON_COMMERCIAL_LICENSES` (longest match, en `_utils.py`). Unificar la función de lookup evitaría la incoherencia de 1.4.
- Tres métodos `_to_covariate_array` distintos (Chronos, TimesFM, TS-ICL) con contratos parecidos pero no iguales (Chronos deja pasar `object`, los otros dos no). Un helper compartido con un flag `allow_object` reduciría duplicación; no bloqueante.
- El registro de licencias necesita mantenimiento manual sin red de seguridad; ver 2.6.
- Los helpers `make_adapter` / `make_v3_adapter` y las dos familias de fakes en el mismo archivo de tests son el reflejo del punto 2.1.

---

## 9. Verificación realizada

Entorno: conda `skforecast_py14`, Python 3.14, `timesfm 3.0.1`, `torch 2.13.0`, macOS Apple Silicon (MPS disponible, sin CUDA). Pesos `google/timesfm-3.0-pytorch` (1.32 GB) descargados para la prueba.

Estático:

- `pytest skforecast/foundation/tests -q`: 650 passed.
- `ruff check` sobre los archivos modificados: limpio.
- `python tools/ai/generate_ai_context_files.py --check`: limpio.
- Lectura de `timesfm3/timesfm3_forecaster.py` (`predict_batch`, `_Query.format`, `linear_interpolation`) y `timesfm3/model.py` (`decode`) para contrastar cada suposición del adaptador.
- Las cuatro URLs de `_NON_COMMERCIAL_LICENSES` responden HTTP 200.

En vivo (script en scratchpad, resultados resumidos):

| Caso | Resultado |
|------|-----------|
| Punto v3 en `device="auto"` (MPS) y en `cpu` | OK en ambos. `auto` resuelve a `mps`. Diferencia máxima MPS vs CPU: 4.8e-07 (relativa 4.6e-07). Primer `predict` 4.7 s (MPS) / 2.1 s (CPU) incluyendo carga; siguientes 0.08 s / 0.11 s |
| Cuantiles `[0.1, 0.5, 0.9]`, `predict_interval`, `q_0.5 == pred`, monotonía, cuantil fuera de malla | OK. `q_0.5 == pred` es `True`; cuantiles monótonos; `predict_interval` devuelve `['level', 'pred', 'lower_bound', 'upper_bound']`; `0.05` lanza el `ValueError` esperado |
| `steps=700` con `max_horizon` por defecto 512 | OK, forma `(700, 2)` en 0.12 s. Sin gate de `max_horizon` en v3 |
| Exog known-future vs sin exog (`h2o_exog`) | Predicciones distintas; MAE 0.0454 con exog vs 0.0627 sin exog |
| `fit` sin exog + `predict(exog=...)` | Sin aviso ni error; 366 NaN enviados al backend en la parte histórica; predicción distinta de ambas anteriores (1.2) |
| Columnas de exog distintas entre `fit` y `predict` | Sin aviso ni error (1.2) |
| Multi-serie, exog solo en una serie | `ValueError: all input arrays must have the same shape` en `timesfm3_forecaster.py:702` (1.1) |
| Multi-serie, distinto número de columnas de exog | Mismo error (1.1) |
| 40 covariables known-future | OK, forma `(12, 2)` |
| `backtesting_foundation` con exog, con y sin `suppress_warnings` | OK; `LicenseWarning` emitido y suprimido respectivamente |
| Coerción de dtypes (`bool`, `Int64` con NA, `object`) | `[1,0,1]`, `[1,nan,3]`, `ValueError` claro |
| Regresión v2.5 (`google/timesfm-2.5-200m-pytorch`, pesos en caché) | OK; sin `LicenseWarning`; `q_0.5 == pred` es `True` (ver 3.1) |
| `padding_mode="edge"` vs `"none"` con exog, `steps=12` (backend directo) | Ambos modos funcionan sin error. Punto `edge` vs `none`: diferencia máxima 0.132 (14 % de la escala de la serie). MAE contra el test real: `edge` 0.0454, `none` 0.0493, sin exog 0.0626. Con `steps=64` (sin padding) coinciden; sin covariables coinciden (3.2) |
| Parte histórica NaN tras `linear_interpolation` del backend | Toda la parte histórica queda como una constante igual al primer valor futuro (`[1.402]` y `[1.461]`), confirmando 1.2 |
| `per_core_batch_size` 4 vs 32, 64 series, CPU | 64 series, `steps=12`, CPU: 0.96 s con 4, 0.44 s con 32, 0.34 s con 64 (2.2x a 2.8x) |

---

## 10. Acciones sugeridas antes de mergear

Bloqueantes:

1. Validar homogeneidad de covariables entre series en `_predict_v3` (o agrupar por firma) con un error claro. Añadir el `np.stack` al fake y un test de integración vía `ForecasterFoundation`.
2. Decidir qué hacer con las columnas futuras sin histórico (`ValueError` recomendado) y documentar la interpolación del backend. Abrir issue para validar nombres de exog en `FoundationModel.predict`.
3. Corregir las cuatro afirmaciones de las skills sobre recarga por `context_length` en v3 y el "all of them".

Recomendadas:

4. Mensaje de error y docs de instalación con `timesfm[torch]`; distinguir "versión antigua" de "falta torch".
5. `__setstate__` con defaults para pickles antiguos, o nota "Breaking" en la release.
6. Reconsiderar la división en dos adaptadores antes de que la API de `TimesFMAdapter` con seis parámetros mixtos quede publicada.
7. Tests: reenvío de `predict_kwargs`, `device` en `from_pretrained`, longest-prefix real, rama NaN.
8. Documentar `padding_mode="edge"` con precisión (ver 3.2) y valorar exponerlo.

Menores:

9. Regex anclado en `_detect_timesfm_backend`; unificar lookup por prefijo.
10. `reload_keys` de v2.5 sin `device`; corregir docstring de `set_params`.
11. Exponer `per_core_batch_size`.
12. Sacar los dos `PLAN_*.md` ajenos de la rama.
