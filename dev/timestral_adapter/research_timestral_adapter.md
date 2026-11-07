# Timestral (Mistral AI): especificación de un adapter para `skforecast.foundation`

Documento de trabajo. Es la especificación completa para implementar el adapter en una sesión nueva: esa sesión no tendrá esta conversación, ni el scratchpad, ni el entorno del experimento, ni el clon de Timestral. Todo lo necesario está aquí o en `dev/timestral_adapter/experiments/`.

Leyenda de la procedencia de cada dato:

- **[probado]**: comprobado ejecutándolo (en el contenedor de esta sesión, con los pesos reales salvo que se diga otra cosa).
- **[verificado en código]**: leído en el código fuente de `timestral` en el commit fijado (`a534bdb`) o en el código de skforecast de la rama de referencia.
- **[documentado]**: lo dice una fuente escrita (README, model card, LICENSE o `config.json` de Hugging Face, issues de GitHub, prensa), sin comprobarlo aparte.
- **[inferido]**: deducción propia a partir de lo anterior.

## 0. Estado y cómo retomar

- **Fecha:** 29/10/2026 (última actualización: 06/11/2026, PR B fusionado, [skforecast/skforecast#1454](https://github.com/skforecast/skforecast/pull/1454), y `foundation-backends.yml` en verde para `timestral`).
- **Estado del plan:** los PR A (#1450) y B (#1454) están fusionados en `0.27.x`. El adapter está terminado y validado en CI. D6 aceptada con (a) el 06/11/2026: el PR C está hecho y sin commitear en la rama `perf/foundation-max-series-per-call` (sección 7), con `max_series_per_call = 50` para Timestral tras medir en GPU.
- **Rama de referencia:** `0.27.x` (skforecast 0.27.0 en desarrollo), commit `9fc0fa12b9d0db55f695755d034f312c88233280` de `origin/0.27.x` (29/10/2026 12:58 +0100). Todas las referencias al código de skforecast son a ese commit. Revalidado el 04/11/2026 sobre `d3db96f` (59 commits más tarde): en `skforecast/foundation/` solo cambió una línea ajena al plan (`observed=True` en `FoundationModel._exog_to_dict`), `_backtesting_foundation` sigue descartando las ventanas de contexto enteras a NaN y `run_checks.py` pasa los 32 chequeos.
- **Qué está commiteado:** este documento y `dev/timestral_adapter/experiments/`, fusionados en `0.27.x` con [skforecast/skforecast#1449](https://github.com/skforecast/skforecast/pull/1449) (merge `924ab7f`, 04/11/2026). Los logs van incluidos con `git add -f`, porque `.gitignore` excluye `*.log`. El PR A (#1450, merge `fc7efd0`, 05/11/2026) y el PR B (#1454, merge `09433c35a`, 06/11/2026) están fusionados en `0.27.x`; la rama `feature/timestral-adapter` se borró del remoto tras el merge.
- **Qué hay en `dev/timestral_adapter/experiments/`:** el prototipo inicial, los seis experimentos (`exp1` a `exp6`) con sus logs y resultados, los requisitos del entorno y `skeleton/`, que extrae el esqueleto de este documento y lo ejecuta contra el backend real (sección 12).
- **Estado del diseño:** el esqueleto de la sección 4, extraído de este documento, pasa los 32 chequeos de `skeleton/run_checks.py` con los pesos reales y `ruff check` con la configuración del repo **[probado]**. D5 y D11 están aceptadas (30/10/2026), D7 (04/11/2026) y las demás decisiones del PR B (05/11/2026); solo D6 (PR C, opcional) sigue sin decidir. La revisión final del PR A (04/11/2026, anexo A, filas 91 a 96) aplicó el plan a una copia de `0.27.x` y ejecutó sus tests **[probado]**. El PR A se fusionó en `0.27.x` el 05/11/2026 (#1450, merge `fc7efd0`, con merge commit); su revisión (anexo A, fila 97) midió el coste con el modelo real de T0 y cambió el relleno de las series a NumPy **[probado]**. La rama del PR B, `feature/timestral-adapter`, está creada desde `fc7efd0` con `--no-track`. La revisión previa al PR B (05/11/2026, anexo A, filas 98 a 101) comprobó el esqueleto contra el código fusionado y el estado de upstream.
- **PR B (fusionado el 06/11/2026, #1454):** hechos los pasos B1 a B7 (B5 con la opción (b)), con `verify` en verde (1057 tests de `skforecast/foundation` y de los llamantes, y `check_foundation_models_metadata.py` en verde para los cuatro checkpoints). `_match_quantile_indices` va en `_utils.py` justo después de `_validate_supported_quantiles`, junto a los otros helpers de cuantiles. Tres commits: el adapter (`71b8cb479`), el enlace del PR en la release note (`cdc75d590`) y un arreglo del `AI Context Check` (`e0d249544`: `extract_urls` comprueba la página del repositorio de los requisitos VCS de pip, porque la URL `.git` del requisito de `timestral` redirige). Con `timestral` 1.0.0 instalado desde git en el entorno de desarrollo (Windows, Python 3.13.18, torch 2.14.1 CPU, numpy 2.5.4, pandas 2.3.3):
  - Prueba con los pesos reales: `experiments/check_real_adapter.py` (los chequeos de `run_checks.py` que necesitan el backend, contra el adapter del paquete y sin parches) pasa 18/18 (`check_real_adapter.log`) **[probado]**.
  - Paso B5, opción (b): las celdas nuevas de Timestral (209M, `context_length=500`, MAE de backtesting 135.67, el mejor de la tabla; igual que en `exp2`), `list_adapters` y la comparación se ejecutaron en un notebook temporal con las celdas de preparación, y la comparación toma las métricas de los demás modelos de sus salidas guardadas. El resto de celdas conserva sus salidas. En el notebook de series heterogéneas solo cambian celdas de texto y un comentario de código. El fold del aviso del issue #7 (`Holiday`) es el que empieza el 2014-12-24 **[probado]**.
  - CI: `foundation-backends.yml` lanzado a mano sobre `0.27.x` tras el merge (06/11/2026). `timestral` pasa instalación, contrato y smoke test con `timestral` 1.0.0 desde git y torch 2.15.0+cpu **[probado]**. La columna "Latest on PyPI" sale vacía porque el paquete no está en PyPI (`latest_version_on_pypi` devuelve `None` y no se aplica la comprobación de "held back"); cuando se publique, esa comprobación empezará a comparar la versión instalada desde git con la de PyPI (checklist de la sección 10). El único fallo de la matriz es `t0` (issue #1385, ajeno al plan).
- **Fuera del plan:** el arreglo de `T0Adapter` para `tfc-t0>=0.4` (issue #1385) lo hace otra persona. El 06/11/2026 se le pasó la nota de traspaso en el issue: el relleno ya está centralizado en `FoundationModel` (#1450), el PR B toca los mismos ficheros de docs, skills y `llms-base.txt`, y dos detalles del issue han quedado desfasados (no hay clave `pinned` en `adapters.toml`; el comando pip del `ImportError` va entre comillas).

**Qué queda:**

1. PR C (sección 7), abierto el 07/11/2026 como #1459 (commit `8a3d45f72`, con los scripts y logs de `exp7*` y `exp8*`) y enlazado en la entrada de `TimestralAdapter` de la release note: antes de fusionar, `git merge origin/0.27.x` (regenerar los `llms-full.txt` si dan conflicto) y repetir `verify` (un atributo obligatorio nuevo es un conflicto semántico que git no ve; TimeH2O tendrá que declararlo).
2. Vigilar la checklist de upstream (sección 10): release en PyPI (cambia `backend_package` y activa la comprobación de "held back" en `foundation-backends.yml`), cambios de API o de licencia, y respuesta a los issues #6 y #7.

Si se añaden logs nuevos a `dev/timestral_adapter/experiments/`, `.gitignore` excluye `*.log`: hay que añadirlos con `git add -f`.

## 1. Resumen y recomendación

**Recomendación: implementarlo**, instalando el backend desde git hasta que llegue a PyPI, con la `LicenseWarning`, con un aviso para el caso de las covariables planas (issue #7) y con dos cambios previos de código compartido (relleno centralizado en `FoundationModel` y MPS en `_resolve_torch_device`) y, con el propio adapter, `_match_quantile_indices` movido a `_utils.py` y un ajuste de `license_name` en `tools/check_foundation_models_metadata.py`.

Motivos:

- **Es el modelo más preciso que hemos medido en la configuración de comparación de la guía de usuario** (`vic_electricity` horario con exog, diciembre de 2014). Timestral-209M con 4096 de contexto da un MAE de 96.89; Chronos-2 da 134.05 con el mismo contexto, y el mejor valor de la tabla de comparación de la guía, TS-ICL con 500 de contexto, es 140.88 **[probado]**. Con contexto corto (500), cada tamaño de Timestral rinde como el Chronos-2 de tamaño parecido; la ventaja aparece al aprovechar el contexto largo. Las cifras que publica Mistral en GIFT-Eval, TIME y fev-bench apuntan en la misma dirección, pero no hay una evaluación independiente.
- **Encaja en el contrato con tres cambios pequeños de código compartido.** Admite exog futuras y pasadas, categóricas y NaN, con 21 cuantiles fijos. El adapter es un `TimesFM3Adapter` sin bucles ni relleno propios: el relleno por la izquierda que necesita (lote rectangular) se centraliza en `FoundationModel`, que T0 también aprovecha. El esqueleto funciona de principio a fin con `FoundationModel`, `ForecasterFoundation` y `backtesting_foundation`, y la predicción de cada serie no depende de las demás series del lote (diferencia máxima de 3e-5) **[probado]**.
- **Funciona bien en CPU.** Pesos públicos sin token, de 21 MB a 837 MB. Una serie de 512 pasos tarda unos 50 ms con el 28M y 100 series, 0,8 s **[probado]**.

Lo que pesa en contra (decisiones de la sección 6):

- **No está en PyPI.** Sería el único backend instalable solo desde git (`timestral @ git+https://github.com/mistralai/timestral.git`).
- **La licencia de los pesos** no da derechos a las empresas con más de 20 M USD de ingreso mensual consolidado (también si es el del empleador).
- **Covariables futuras planas.** Una covariable futura constante en el contexto que cambia en el horizonte (por ejemplo, un precio, o un indicador 0/1 que vale 0 en todo el contexto) dispara los cuantiles altos (con un precio, en los tres checkpoints medidos; con un indicador, en el 28M y el 209M) y, en varios casos, también la mediana (el 209M con un precio, el 28M con un indicador), sin avisar (issue #7, reproducido).
- **Backend joven.** Un solo commit, sin releases, no aceptan pull requests y no han respondido a ningún issue. No funciona en MPS.

Si prefieres no depender de una instalación desde git ni de un backend sin respuestas de sus mantenedores, la alternativa razonable es **esperar** a que Mistral publique en PyPI y responda a los issues #5 y #7. El diseño seguiría valiendo tal cual: solo cambiaría `backend_package`.

## 2. Ficha del modelo

### Qué es

- Familia de foundation models para previsión probabilística, univariante y multivariante, con covariables pasadas, futuras y categóricas **[documentado]**.
- Transformer causal por parches de 32 pasos (`patch_size = 32` en los cuatro `config.json`) **[verificado en código]**. Se entrena con *Decoupled Horizon Forecasting* (DHF): la previsión sale de tokens de predicción separados que leen el contexto codificado causalmente **[documentado]** **[verificado en código]**.
- Las capas siguen el patrón `time, time, time, time_variate`: tres capas solo temporales y una `time_variate` (atención temporal y después entre variables), es decir, cuatro atenciones temporales por cada atención entre variables (2, 3, 5 y 6 capas `time_variate` en 5M, 28M, 123M y 209M). Las covariables pasan información a los objetivos, pero nunca la leen (`masks.variate_firewall`) **[verificado en código]**.
- Cuatro checkpoints, todos con pesos en float32 **[probado]**:

| Checkpoint | Parámetros | Pesos | Capas | `d_model` | Revisión en HF (fijada) |
|:-----------|-----------:|------:|------:|----------:|:------------------------|
| `mistralai/Timestral-5M` | 5,2 M | 21 MB | 8 | 192 | `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` |
| `mistralai/Timestral-28M` | 27,8 M | 111 MB | 12 | 384 | `49b26409ac87b5935539a5d8ffe7f397b720db47` |
| `mistralai/Timestral-123M` | 122,6 M | 490 MB | 20 | 640 | `5a147287addda85f6acd8da3fb8858fe9af7e8f7` |
| `mistralai/Timestral-209M` | 209,3 M | 837 MB | 24 | 768 | `9874d252ab0aae16f7f7c999f98f98d6ad7d8365` |

- Publicado el 15/10/2026: un único commit en GitHub ("Timestral release"), versión 1.0.0, sin releases ni tags **[probado]**. Los repositorios de Hugging Face se crearon el 14/10/2026 **[documentado]** (API del Hub).
- Madurez: 1.712 líneas de código y 102 de tests. Los tests usan un modelo diminuto con pesos aleatorios, no los checkpoints publicados **[verificado en código]**. Mistral no acepta pull requests y responde a los issues "best-effort" (`CONTRIBUTING.md`) **[documentado]**. Hay 7 issues abiertos y ninguna respuesta de mantenedores **[documentado]**.
- El informe técnico ("Timestral", HF Papers 2610.37585) se publicó el 27/10/2026 **[documentado]**. No lo he leído: arXiv está bloqueado en este entorno. La lista de datos de entrenamiento sigue sin publicarse (issue #3) **[documentado]**.

### Licencia

- Código: Apache 2.0 **[documentado]**.
- Pesos: "Modified MIT License", con el mismo `LICENSE` en los cuatro repositorios (1.695 bytes, sha256 `648194727930df730890e9f9affdbcc5c930d406d6829559b25ca3d5a973fc8c`) **[probado]**. Es una MIT con dos cláusulas añadidas **[documentado]**:
  - Atribución obligatoria ("Attribution notice: 2026 - Mistral AI").
  - "You are not authorized to exercise any rights under this license if the global consolidated monthly revenue of your company (or that of your employer) exceeds $20 million (or its equivalent in another currency) for the preceding month." La restricción se extiende a derivados y obras combinadas. Esas empresas necesitan una licencia comercial (sales@mistral.ai) o usar los servicios alojados de Mistral. La cláusula está numerada "2." pero se refiere a sí misma como "(b)".
- En Hugging Face la tarjeta declara `license: other` sin `license_name` ni `license_link` **[probado]**.

### Distribución

- **No está en PyPI**: `https://pypi.org/pypi/timestral/json` devuelve 404, y también los nombres parecidos **[probado]**. Se instala con `pip install git+https://github.com/mistralai/timestral.git` **[documentado]**. Para skforecast hay que usar la forma PEP 508, `"timestral @ git+https://github.com/mistralai/timestral.git"`, porque `tools/foundation_compat/run.py` saca el nombre del paquete del principio de la cadena (`distribution_name`) y con la URL sola obtendría `"git"` **[probado]**.
- Dependencias: Python >= 3.10, `torch>=2.4`, `numpy>=1.24`, `safetensors>=0.4`, `huggingface_hub>=0.23`, sin límites superiores **[verificado en código]**.
- Pesos en el Hub de Hugging Face (`config.json` y `model.safetensors`), **no gated**, sin token ni cuenta de proveedor; se guardan en la caché estándar del Hub (`snapshot_download` con `cache_dir=None`) **[probado]** **[verificado en código]**. Con `HF_HUB_OFFLINE=1` o `local_files_only=True` se leen de la caché **[probado]**.
- No hace falta la API de Mistral. El ajuste fino se ofrece aparte, en Mistral Forge **[documentado]** (prensa).

### API de inferencia

```python
from timestral import TimestralForecaster

forecaster = TimestralForecaster.from_pretrained(
    path_or_repo_id,                  # repo del Hub o directorio local
    device="cpu",                     # str o torch.device; "auto" no existe (RuntimeError de torch)
    max_context=None,                 # tope de contexto; None = 32768 (config.context_length)
    per_shot_horizon_patches=32,      # parches por bloque; > 32 se recorta a 32 sin avisar
    quantile_real_cap_k=1e4,          # recorte de cuantiles extremos; <= 0 lo desactiva
    revision=None, cache_dir=None, token=None, local_files_only=False,
)
forecast = forecaster.forecast(
    context,                          # (B, V, T), (B, T) o (T,); numpy o torch
    prediction_length,                # int >= 1 (assert)
    observation_mask=None,            # bool, mismo número de elementos que context
    past_covariates=None,             # (B, C_past, T)
    known_covariates=None,            # (B, C_known, T + prediction_length)
    categorical_past_covariates=None, # (B, C, T), códigos numéricos, NaN = missing
    categorical_known_covariates=None,# (B, C, T + prediction_length)
    symmetric_averaging=True,         # promedia con la previsión de -context
    clip_nonnegative=True,            # recorta a 0 si el contexto observado es >= 0
)
forecast.quantiles        # torch.Tensor float32 (B, V, prediction_length, 21), en el device del modelo
forecast.quantile_levels  # tuple de 21 float
forecast.median           # property: (B, V, prediction_length)
forecast.quantile(0.9)    # (B, V, prediction_length); ValueError si el nivel no existe
```

Fuente: `src/timestral/forecaster.py` y `loading.py` **[verificado en código]**; firmas y formas comprobadas ejecutándolas **[probado]**. Notas que afectan al adapter:

- `forecast` va decorado con `@torch.inference_mode()`, el modelo está en `eval()` y no guarda estado entre llamadas. Dos llamadas iguales dan exactamente el mismo resultado en CPU (`exp5.log`) **[probado]**.
- Casi toda la validación de entradas son `assert` (formas, `prediction_length >= 1`), que desaparecen con `python -O` **[verificado en código]**. Lanzan `ValueError`: códigos categóricos infinitos, un nivel de cuantil inexistente y una salida con NaN.
- Todo se convierte a float32 (`torch.as_tensor(..., dtype=torch.float32)`): una entrada float64 da el mismo resultado que una float32 **[probado]**. Los códigos categóricos se quedan en float64 en CPU.

### Entrada

- Un tensor rectangular `(B, V, T)`: todas las series del lote tienen la misma longitud `T` **[verificado en código]**.
- Las `V` variables de una serie se predicen conjuntamente. Cada una de las `B` series es independiente: la atención entre variables solo mezcla filas de la misma serie (`group_ids`). Aun así, el cálculo es conjunto para todas las filas de la llamada **[verificado en código]**.
- Contexto máximo de 32.768 pasos; se queda con los más recientes (`_assemble_rows`) **[documentado]** **[probado]**. Contexto mínimo de 1 paso: se rellena hasta múltiplo de 32 **[probado]**.
- No hay límite de horizonte. Se predice en bloques de 32 parches (1.024 pasos) y los horizontes más largos se encadenan realimentando la mediana; el horizonte se redondea al alza a parches completos **[verificado en código]** **[probado]**. Durante esa decodificación el contexto crece con cada bloque, hasta `max_context`: con horizontes de más de 1.024 pasos, el modelo usa más contexto que el `context_length` del adapter **[verificado en código]**.
- NaN marca un valor ausente. En el objetivo también se tratan como ausentes los infinitos (`isfinite`); en las covariables numéricas, solo los NaN (un `inf` entra como valor) **[verificado en código]**.
- Los parches del principio sin ninguna observación son relleno y nunca se atienden (`masks.compute_enc_key_validity`) **[verificado en código]**. Por eso se pueden juntar series de distinta longitud rellenando con NaN por la izquierda: **la predicción coincide con la de cada serie sola** (diferencia máxima de 1,1e-5 en 5M, 28M y 209M, `exp1.log`) **[probado]**.
- Una serie con todo el contexto a NaN no da error: devuelve cuantiles en torno a 0 (`exp5.log`) **[probado]**. En skforecast puede llegar al backend: `FoundationModel.fit` rechaza las series con todos los valores a NaN, pero lo comprueba antes de recortar a `context_length`, así que una serie con sus últimos `context_length` valores a NaN pasa (lo mismo con un `context` pasado a `predict`; el backtesting descarta esas ventanas en `_backtesting_foundation`) **[probado]**. Ver riesgos (sección 14).
- No hay ningún parámetro de frecuencia: el modelo no recibe el índice temporal **[verificado en código]**.

### Salida

- 21 cuantiles fijos y que no se cruzan: `0.01, 0.05, 0.1, 0.15, ..., 0.85, 0.9, 0.95, 0.99` (`quantile_knots`, iguales en los cuatro `config.json`; monotonía por construcción en `monotone_quantiles`) **[verificado en código]** **[probado]**. Tiene mediana, pero ni media ni muestras.

### Covariables

- Pasadas (`past_covariates`, solo en el contexto) y futuras conocidas (`known_covariates`, contexto + horizonte) **[documentado]** **[verificado en código]**. Los cuatro checkpoints admiten ambas (`covariate_enabled: true`) **[verificado en código]**.
- Categóricas: códigos numéricos (por ejemplo, de `pandas.factorize`, mapeando el -1 de los ausentes a NaN). Con un solo objetivo por serie, el backend aplica una *target encoding* suavizada por serie (`covariate_encoding.target_encode`, `smooth=1`): NaN es una categoría propia y una categoría no vista recibe la media de la serie. Con varios objetivos usa los códigos tal cual **[verificado en código]**.
- Las covariables **futuras** con valores 0/1 se tratan como indicadores, sin normalizar; las pasadas siempre se normalizan **[verificado en código]**. En la pasada simétrica (`symmetric_averaging`) un indicador {0, 1} se convierte en {0, -1} y deja de tratarse como indicador **[verificado en código]**.
- Las filas de las covariables futuras tienen atención temporal bidireccional sobre toda la ventana, horizonte incluido (`masks.build_two_stream_time_mask` con `bidirectional_rows`), los objetivos las leen a través de la atención entre variables, y además se normalizan con toda la fila (`instance_norm_stats`) **[verificado en código]**. Consecuencia: con covariables futuras, los primeros pasos de la previsión dependen de `steps` (28M, h=24 frente a h=100: diferencia de 0,023 en los 24 primeros pasos, frente a 1,5e-5 sin covariables o con una pasada; `exp5.log`) y de los valores futuros de la covariable en pasos posteriores. Es así por diseño y no es una fuga de información, pero un backtesting con otro `steps` no es comparable paso a paso **[probado]**.
- Todas las series de una llamada tienen que tener el mismo número de covariables `C`, porque son tensores **[verificado en código]**.
- `symmetric_averaging` (activo por defecto) hace una segunda pasada con el contexto y **todas** las covariables negadas, y promedia; cuesta algo más del doble (209M, 50 series x 500: 2,03 s frente a 0,92 s; `exp5.log`) **[verificado en código]** **[probado]**. Todos los tiempos de este documento son con el valor por defecto (`True`).
- **Issue #7.** Con una covariable futura constante en el contexto que cambia en el horizonte, los cuantiles altos se disparan en el 5M, el 28M y el 209M. Pasa igual con un indicador 0/1 que vale 0 en todo el contexto (un festivo o una promoción en el horizonte) en el 28M y el 209M; en el 5M, no. La mediana sube en varios casos: mucho en el 209M con un precio (194 frente a 2) y con un indicador (136), y claramente en el 28M con un indicador (18 frente a 2). Lo he reproducido (sección 2, Experimento) **[probado]**. Según el issue hay dos causas: la pasada simétrica, que niega el precio, y la normalización de una fila de varianza cero **[documentado]**. La causa no está confirmada: un indicador 0/1 no se normaliza y también falla **[probado]**.

### Hardware

- Funciona en CPU **[probado]** y, según el README, en GPU **[documentado]** (no hay GPU en este entorno).
- **No funciona en MPS** (Apple Silicon): las estadísticas se calculan en float64 en el device de la entrada (`scaler.compute_causal_statistics`, `forecaster._refine_block`) y MPS no admite float64 **[verificado en código]**. Lo confirma el issue #6, sin corregir; un fork no oficial (`paschembri/timestral`, rama `mps-support`) lo arregla **[documentado]**.
- La atención entre variables se calcula sobre todas las filas de la llamada: máscara `(N_enc + N_dec, 1, R, R)` con `R = B x (V + C_past + C_known + C_cat)`. Existe también sin covariables, en cuanto `R > 1` **[verificado en código]** **[probado]**. El coste crece de forma cuadrática con el número de series de una llamada. El issue #6 cuenta un intento de reservar 62 GiB con 635 series y covariables en una sola llamada **[documentado]**, coherente con el tamaño de esas matrices **[inferido]**.

### Resultados publicados

Valores de la model card del 209M, evaluación zero-shot **[documentado]**. Los publica el propio Mistral, que no ha publicado la lista de datos de entrenamiento: no son una evaluación independiente.

| Benchmark | Métrica | Timestral-209M | t0-gamma | TimesFM-3 | Toto-2.0-2.5B | Chronos-2 |
|:----------|:--------|---------------:|---------:|----------:|--------------:|----------:|
| GIFT-Eval | MASE (menor es mejor) | **0.658** | 0.661 | 0.667 | 0.696 | 0.698 |
| GIFT-Eval | CRPS (menor es mejor) | **0.446** | 0.452 | 0.456 | 0.476 | 0.485 |
| TIME | MASE (menor es mejor) | **0.626** | 0.637 | 0.640 | 0.642 | 0.662 |
| TIME | CRPS (menor es mejor) | **0.522** | 0.534 | 0.536 | 0.539 | 0.556 |
| fev-bench | SQL skill (mayor es mejor) | **50.39** | 49.42 | 48.99 | 44.42 | 47.28 |
| fev-bench | MASE skill (mayor es mejor) | **39.49** | 37.99 | 37.82 | 32.54 | 35.50 |

Escalado en GIFT-Eval (MASE): 5M 0.705, 28M 0.678, 123M 0.659, 209M 0.658 **[documentado]**. Moirai no aparece en la comparación.

### Experimento en skforecast [probado]

Entorno aparte (`~/.venvs/timestral-exp`, fuera del repositorio): Python 3.12.3, torch 2.14.1 (wheel de PyPI con CUDA, ejecutado en CPU con 4 hilos), numpy 2.5.4, pandas 2.3.3, timestral 1.0.0 (`a534bdb`), chronos-forecasting 2.3.2 y skforecast 0.27.0 (editable, rama `0.27.x`). Versiones exactas y cómo recrearlo: sección 11.

**Backtesting** (`exp2_backtest.py`, `results_exp2.csv`, `exp2.log`; las 15 configuraciones en una sola ejecución). Misma configuración que `docs/user_guides/foundation-forecasting-models.ipynb`:

- `vic_electricity` horario (`resample(rule="h", closed="left", label="right")`), `Demand` con `Temperature` y `Holiday` como exog.
- Datos de 2012-01-01 00:00 a 2014-12-30 23:00, `end_train = "2014-11-30 23:59:00"`: test del 01/12/2014 00:00 al 30/12/2014 23:00 (30 folds de 24 pasos, 720 predicciones), `refit=False`.
- `backtesting_foundation` con el prototipo registrado en tiempo de ejecución. Se mide el MAE de la mediana, la cobertura empírica del intervalo `[0.1, 0.9]` (nominal 80 %) y el tiempo del backtesting sin contar la carga de los pesos.

| Modelo | `context_length` | Exog | MAE | Cobertura 80 % | Tiempo (s) |
|:-------|-----------------:|:----:|----:|---------------:|-----------:|
| Chronos-2-small | 500 | Sí | 171.27 | 0.731 | 1.4 |
| Chronos-2 (120M) | 500 | Sí | 137.49 | 0.740 | 4.3 |
| Timestral-5M | 500 | Sí | 176.04 | 0.772 | 1.1 |
| Timestral-28M | 500 | Sí | 170.95 | 0.769 | 2.9 |
| Timestral-123M | 500 | Sí | 142.96 | 0.778 | 8.9 |
| Timestral-209M | 500 | Sí | 135.67 | 0.746 | 12.8 |
| Timestral-28M | 500 | No | 193.63 | 0.722 | 1.4 |
| Timestral-209M | 500 | No | 177.65 | 0.704 | 5.4 |
| Chronos-2-small | 4096 | Sí | 158.55 | 0.683 | 5.0 |
| Chronos-2 (120M) | 4096 | Sí | 134.05 | 0.721 | 19.1 |
| Timestral-28M | 4096 | Sí | 134.90 | 0.764 | 11.6 |
| Timestral-209M | 4096 | Sí | **96.89** | 0.718 | 73.0 |
| Chronos-2 (120M) | 8192 | Sí | 133.28 | 0.749 | 35.5 |
| Timestral-28M | 8192 | Sí | 129.76 | 0.765 | 19.5 |
| Timestral-209M | 8192 | Sí | **96.70** | 0.694 | 147.7 |

- La configuración reproduce la de la guía: Chronos-2-small con 500 de contexto da 171.27, el mismo valor que el notebook publicado (que se ejecutó con torch 2.9.1+cpu). Las métricas se calculan sobre las 720 horas del test.
- Como referencia, la tabla de comparación del notebook publica con 500 de contexto: TS-ICL 140.88, T0 145.53, TimesFM 3.0 162.44, Chronos-2-small 171.27, Nori 185.76, TabPFN-TS 186.42, TimesFM 2.5 191.93 (sin exog), Moirai-2 201.82 (sin exog) y TabICL 215.80 **[verificado en código]** (salidas del notebook). El barrido de `context_length` del mismo notebook (Chronos-2-small, 123.64 con 5000) usa otra ventana de test (`end_train` de la sección multiserie), así que no es comparable.
- Con 500 de contexto, cada tamaño de Timestral rinde como el Chronos-2 de tamaño parecido (28M: 170.95 frente a 171.27 de Chronos-2-small; 209M: 135.67 frente a 137.49 de Chronos-2 120M).
- Timestral aprovecha mucho mejor el contexto largo. Con 4096, el MAE del 209M baja de 135.67 a 96.89 (un 29 % menos), un 28 % por debajo de Chronos-2 con el mismo contexto (134.05). Pasar a 8192 no aporta nada más (96.70) y cuesta el doble (147.7 s frente a 73.0 s). El 28M sí mejora algo (de 134.90 a 129.76).
- Las exog ayudan: con 500 de contexto, el MAE baja un 12 % en el 28M y un 24 % en el 209M.
- Ningún modelo llega al 80 % nominal de cobertura: todos están entre 0.68 y 0.78. Con exog, los Timestral pequeños y medianos cubren más que Chronos-2 (0.76 a 0.78); el 209M con contexto largo cubre como Chronos-2 o algo menos (0.718 frente a 0.721 con 4096; 0.694 frente a 0.749 con 8192).
- Es **un solo dataset**, horario y con estacionalidad marcada. Indica la tendencia, pero no sustituye a un benchmark.

**Comprobaciones del diseño** (`exp1_smoke.py` y `exp1.log`, `exp3_multiseries.py` y `exp3.log`, `skeleton/run_checks.py`):

- Una serie sola frente a la misma serie dentro de un lote rellenado con NaN por la izquierda: diferencia máxima de 1,1e-5 en 5M, 28M y 209M (`exp1.log`).
- Tres series heterogéneas a través de `FoundationModel`: 900, 400 y 700 observaciones, una con NaN; dos con una exog numérica y otra categórica, la tercera solo con la numérica. Comparadas con cada serie sola, la diferencia máxima es de 2,7e-5 con el prototipo (`exp3.log`) y de 3,1e-5 con el esqueleto de la sección 4 y el relleno en `FoundationModel` (`run_checks.py`). El agrupado por columnas y la factorización de la categórica funcionan, con una llamada al backend por grupo.
- `quantiles=[0.33]` da `ValueError` con la lista de niveles admitidos.
- Tiempo en CPU de una llamada sin covariables y 24 pasos (`exp1.log`; carga desde la caché local, sin descarga):

| Checkpoint | Carga | 1 serie x 512 | 100 series x 512 | 1 serie x 8192 |
|:-----------|------:|--------------:|-----------------:|---------------:|
| 5M | 0.1 s | 23 ms | 0.38 s | 0.07 s |
| 28M | 0.4 s | 46 ms | 0.80 s | 0.15 s |
| 209M | 1.3 s | 218 ms | 4.24 s | 0.84 s |

**Coste de las covariables y troceado de lotes** (`exp3_multiseries.py` y `exp3.log`, `exp4_cost_and_issue7.py` y `exp4.log`). Una llamada con 1 objetivo y 2 covariables futuras, 512 de contexto y 24 pasos:

| Checkpoint | Series | Una llamada | Trozos de 100 | Pico de RSS |
|:-----------|-------:|------------:|--------------:|------------:|
| 28M | 10 / 100 / 200 / 400 | 0.50 / 5.47 / 12.28 / 31.57 s | | 0,8 / 1,1 / 1,4 / 1,8 GB |
| 209M | 100 | 34.7 s | 33.5 s | 2,1 GB |
| 209M | 400 | 187.9 s | **123.6 s** | 3,3 GB |

- Con dos covariables, cada serie cuesta unas 7 u 8 veces más que sin ellas: 100 series pasan de 0.80 s a 5.47 s en el 28M (6,8 veces; una covariable numérica y otra categórica, `exp3.log`) y de 4.24 s a 34.7 s en el 209M (8,2 veces; dos numéricas, `exp4.log`). El modelo procesa el contexto y el horizonte enteros, con un token de decodificación por parche, en tres filas por serie, y la atención entre variables crece con el cuadrado de las filas **[verificado en código]**.
- El coste por serie crece con el tamaño del lote. Trocear en grupos de 100 reduce un 34 % el tiempo con 400 series. En GPU, además, evitaría reservas de memoria enormes (issue #6) **[inferido]**.

**Issue #7 reproducido** (`exp4_cost_and_issue7.py` y `exp4.log`). 32 series diarias Poisson(2), 1024 de contexto, 28 pasos, 209M; el precio es una covariable futura de 2.50 en todo el contexto que cambia a 2.56 en el horizonte:

| Caso | Mediana máxima | q99 máximo | Series con q99 > 100 |
|:-----|---------------:|-----------:|---------------------:|
| Sin covariable | 2.02 | 7.54 | 0/32 |
| Precio plano en el contexto | **194.04** | **14955.49** | 32/32 |
| Precio plano, `symmetric_averaging=False` | 5.30 | **14955.49** | 32/32 |
| Precio que ya cambia dentro del contexto | 2.02 | 19.66 | 0/32 |
| Precio plano con el histórico a NaN | 2.02 | 7.70 | 0/32 |
| Precio plano, `quantile_real_cap_k=10` | 13.49 | 24.17 | 0/32 |

- Con una sola serie a través de `FoundationModel` pasa lo mismo: mediana máxima de 37.05 y q99 de 14059.03 con el promedio simétrico; con `symmetric_averaging=False`, la mediana se queda en 2.60 y el q99 no cambia (`exp3.log`).
- El q99 que se ve es el recorte `quantile_real_cap_k=1e4`: la previsión sin recortar diverge todavía más **[inferido]**.
- `symmetric_averaging=False` no es un remedio general (`exp5.log`) **[probado]**. Con el precio, baja la mediana del 209M de 194.04 a 5.30 (frente a 2.02 sin covariable), pero los cuantiles altos no cambian; en el 5M y el 28M no arregla nada, y en el 5M empeora el q99. Con el indicador, lo arregla en el 209M y solo en parte en el 28M. Resultados por checkpoint:

| Checkpoint | Caso | Mediana máxima | q99 máximo | Series con q99 > 100 |
|:-----------|:-----|---------------:|-----------:|---------------------:|
| 5M | Sin covariable | 2.13 | 6.41 | 0/32 |
| 5M | Precio plano | 4.51 | 9015.71 | 28/32 |
| 5M | Precio plano, `symmetric_averaging=False` | 6.99 | 14581.06 | 27/32 |
| 5M | Precio plano con el histórico a NaN | 2.21 | 6.71 | 0/32 |
| 28M | Sin covariable | 2.05 | 7.10 | 0/32 |
| 28M | Precio plano | 2.86 | 14955.49 | 32/32 |
| 28M | Precio plano, `symmetric_averaging=False` | 2.50 | 14955.49 | 30/32 |
| 28M | Precio plano con el histórico a NaN | 2.06 | 7.16 | 0/32 |
| 5M | Indicador 0 en el contexto y 1 en dos pasos del horizonte | 2.11 | 6.49 | 0/32 |
| 5M | Indicador, `symmetric_averaging=False` | 2.17 | 6.67 | 0/32 |
| 5M | Indicador con el histórico a NaN | 2.21 | 6.71 | 0/32 |
| 28M | Indicador 0 en el contexto y 1 en dos pasos del horizonte | 18.05 | 9149.63 | 32/32 |
| 28M | Indicador, `symmetric_averaging=False` | 9.72 | 3752.83 | 26/32 |
| 28M | Indicador con el histórico a NaN | 2.06 | 7.16 | 0/32 |
| 209M | Indicador | 136.06 | 7481.13 | 32/32 |
| 209M | Indicador, `symmetric_averaging=False` | 2.02 | 7.94 | 0/32 |
| 209M | Indicador con el histórico a NaN | 2.02 | 7.70 | 0/32 |

- En la guía de usuario (`vic_electricity`, 500 de contexto), `Holiday` vale 0 en todo el contexto y 1 en el horizonte en uno de los 30 folds (origen 2014-12-23 23:00): el aviso del adapter saltará ahí al volver a ejecutar el notebook **[probado]**.
- Una covariable futura constante en el contexto que cambia en el horizonte rompe la previsión sin avisar, y es justo el caso de un precio en retail: el issue cuenta que el 38 % de las combinaciones serie-origen de M5 FOODS_3 tienen el precio plano en 1.024 días de contexto **[documentado]**. Lo único que lo arregla en todos los casos medidos (el precio en el 5M, el 28M y el 209M; el indicador en el 28M y el 209M) es poner a NaN el histórico plano de la covariable.

## 3. Tabla de capacidades frente a los adapters existentes

Valores de los atributos de clase en `skforecast/foundation/_adapters.py` y de `_MODEL_LICENSES` en `_utils.py` (rama `0.27.x`) **[verificado en código]**; estado en PyPI **[probado]**. La fila de Timestral es la propuesta de la sección 4.

| Adapter | Exog | Solo pasadas | Categóricas nativas | Exog heterogéneas en un lote | NaN en la serie | Cuantiles | `context_length` por defecto | Uso comercial restringido | Backend en PyPI |
|:--------|:----:|:------------:|:-------------------:|:----------------------------:|:---------------:|:----------|-----------------------------:|:-------------------------:|:---------------:|
| Chronos-2 | Sí | Sí | Sí | No | Sí | Cualquiera | 8192 | No | Sí |
| TimesFM 2.5 | No | No | No | Sí | Sí | 0.1 ... 0.9 | 512 | No | Sí |
| TimesFM 3.0 | Sí | Sí | No | No | Sí | 0.1 ... 0.9 | 2048 | Sí | Sí |
| Moirai-2 | No | No | No | Sí | Sí | 0.1 ... 0.9 | 2048 | Sí | Sí |
| TabICL | Sí | No | No | No | Sí | Cualquiera | 4096 | No | Sí |
| TabPFN-TS | Sí | No | No | Sí | Sí | Cualquiera | 32768 | Sí | Sí |
| T0 | Sí | No | No | Sí | Sí | Cualquiera | 8192 | No | Sí |
| Nori | Sí | No | No | Sí | Sí | Cualquiera | 4096 | No | Sí |
| TS-ICL | Sí | Sí | No | No | Sí | 0.01 ... 0.99 (paso 0.01) | 4096 | Sí | Sí |
| **Timestral (propuesta)** | **Sí** | **Sí** | **Sí** (el adapter traduce las etiquetas a códigos) | **No** | **Sí** | **21 fijos: 0.01, 0.05, 0.1 ... 0.9 (paso 0.05), 0.95, 0.99** | **4096** (máximo del modelo: 32768) | **Sí** (empresas con más de 20 M USD de ingreso mensual) | **No** (solo git) |

Lo que más se parece:

- **Capacidades:** Chronos-2. Es el único otro adapter con covariables pasadas, futuras y categóricas a la vez.
- **Cuantiles fijos, `device` como texto y `predict_kwargs` con claves reservadas:** TimesFM 3.0, que es el único con `_RESERVED_PREDICT_KWARGS`.
- **Lote rectangular rellenado con NaN por la izquierda:** T0. Lo hacía dentro del propio adapter (`T0Adapter.predict` y `T0Adapter._build_future_covariates`) hasta el PR A (#1450); desde entonces lo hace `FoundationModel` (`requires_equal_length_context=True`).

Con la rejilla de 21 cuantiles, Timestral admite los intervalos del 80 % (`[0.1, 0.9]`, el valor por defecto de `predict_interval`), 90 % (`[0.05, 0.95]`) y 98 % (`[0.01, 0.99]`) **[probado]**. Con la rejilla `0.1 ... 0.9`, TimesFM y Moirai no pueden dar intervalos del 90 % ni del 98 %.

Nota al margen: `TSICLAdapter` declara `SUPPORTED_QUANTILES`, pero no llama a `_validate_supported_quantiles` (solo lo hacen TimesFM 2.5, TimesFM 3.0 y Moirai) **[verificado en código]**. No afecta a Timestral.

## 4. Propuesta de diseño

El esqueleto de esta sección es código completo, no un resumen. `dev/timestral_adapter/experiments/skeleton/` lo extrae de este documento y lo ejecuta contra el backend real (sección 12). Las versiones anteriores del diseño están en `dev/timestral_adapter/experiments/timestral_proto.py` (prototipo de los experimentos), que ya no coincide con este esqueleto.

### Plantilla

`TimesFM3Adapter`: cuantiles fijos validados con `_validate_supported_quantiles`, parámetro `device` como texto resuelto con `_resolve_torch_device`, `predict_kwargs` con claves reservadas (`_validate_predict_kwargs`), objeto `model` inyectable para los tests, `set_params` con un cierre `validate` y reinicio del modelo cuando cambian `model_id` o `device`. Las covariables se construyen por serie con `_build_covariates`, usando `get_exog_signature` de la primera serie (fija el orden de las columnas). Hay dos diferencias con TimesFM 3.0: Timestral pide un lote rectangular (lo resuelve `FoundationModel`, ver "Cambios de código compartido") y admite categóricas.

### Registro

- Prefijo en `_ADAPTER_REGISTRY`: `"mistralai/Timestral"`, al final del diccionario (el orden del registro es el de `list_adapters()`). Cubre los cuatro tamaños y no captura otros modelos de Mistral.
- `default_model_id`: `"mistralai/Timestral-28M"` (decisión D8). Sigue el criterio de Chronos (`autogluon/chronos-2-small`): un checkpoint pequeño para la documentación y las herramientas. Con 4096 de contexto rinde como Chronos-2 (120M) en un 40 % menos de tiempo (134.90 en 11.6 s frente a 134.05 en 19.1 s). Para la mejor precisión, la documentación recomendaría el 209M.
- Entrada en `_MODEL_LICENSES` (decisión D2):

```python
    "mistralai/Timestral": (
        "other",
        "https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE",
        True,
        "Timestral Modified MIT License (no rights for companies with a monthly "
        "revenue above USD 20 million)",
    ),
```

  El primer campo es `"other"` porque la tarjeta no tiene `license_name` (el comentario de `_MODEL_LICENSES` dice que se usa el `license_name` cuando la licencia no es estándar). El URL es fijo porque los cuatro repositorios publican el mismo `LICENSE`.

### Atributos de capacidad

| Atributo | Valor | Justificación |
|:---------|:------|:--------------|
| `SUPPORTED_QUANTILES` | Los 21 niveles | `quantile_knots` es idéntico en los cuatro `config.json` **[probado]**. Un nivel fuera de la rejilla da `ValueError`, como en TimesFM. La correspondencia con las columnas se hace con `forecast.quantile_levels`, no con la lista fija. |
| `allow_exog` | `True` | `past_covariates` y `known_covariates`. |
| `supports_past_only_covariates` | `True` | Las columnas del exog histórico sin valores futuros van a `past_covariates` (o `categorical_past_covariates`). |
| `supports_categorical_covariates` | `True` (decisión D4) | El backend trata las categóricas como tales (*target encoding* por serie). El adapter solo traduce las etiquetas a códigos con `pd.factorize` sobre el histórico y el futuro juntos de cada serie, para que una etiqueta tenga el mismo código en el contexto y en el horizonte; los ausentes pasan a NaN. |
| `supports_heterogeneous_covariates` | `False` | Las covariables son tensores `(B, C, T)`: todas las series de una llamada necesitan las mismas `C` columnas. Rellenar las columnas que faltan con filas de NaN, como hace T0, cambiaría la predicción: los tokens de decodificación de una fila vacía siguen siendo claves válidas en la atención entre variables (`masks.build_two_stream_variate_mask`, `dec_valid`); una covariable entera a NaN mueve la previsión 0,022 sobre un nivel de 50 (28M, `exp5.log`) **[verificado en código]** **[probado]**. Con `False`, `FoundationModel` agrupa las series por columnas y la predicción de una serie nunca depende de las exog de otra. |
| `requires_equal_length_context` (nuevo) | `True` | Atributo nuevo del contrato (decisión D5): `forecast` necesita la misma longitud de contexto para todas las series de la llamada, y `FoundationModel` rellena cada grupo por la izquierda con NaN. |
| `supports_nan_in_series` | `True` | NaN marca un valor ausente **[documentado]** **[probado]**. |
| `requires_hf_auth` | `False` | Repositorios no gated **[probado]**. |
| `requires_provider_auth` | `False` | No hay que aceptar ninguna licencia ni tener cuenta fuera del Hub. |
| `weights_repo_id` | `None` | Los pesos se descargan del propio `model_id`. |
| `weights_in_hf_cache` | `True` | `snapshot_download` usa la caché estándar del Hub **[verificado en código]**. |
| `backend_package` | `"timestral @ git+https://github.com/mistralai/timestral.git"` mientras no esté en PyPI; después, `"timestral"` (decisión D1) | Es lo que el usuario pasa a `pip install` y lo que instala `tools/foundation_compat/run.py`. La forma PEP 508 hace falta para que `run.distribution_name` devuelva `timestral` **[probado]**. |
| `default_model_id` | `"mistralai/Timestral-28M"` | Ver "Registro". |

### Parámetros del constructor

| Parámetro | Por defecto | Comentario |
|:----------|:------------|:-----------|
| `model_id` | | Comprobado con `_validate_model_id_prefix`, como TimesFM 3.0. |
| `model` | `None` | `TimestralForecaster` ya cargado. Para los tests. |
| `context_length` | `4096` (decisión D9) | El modelo admite hasta 32.768 pasos. En el experimento, pasar de 500 a 4096 baja el MAE del 209M un 29 %, y pasar a 8192 no mejora nada y cuesta el doble. Es el mismo valor que TabICL, Nori y TS-ICL; Chronos y T0 usan 8192. |
| `device` | `"auto"` | CUDA > MPS > CPU, resuelto por `_resolve_torch_device(..., mps_fallback_message=...)`: si sale MPS (con `"auto"` o explícito), avisa y usa CPU, como Moirai (decisión D7). El backend no acepta `"auto"`. |
| `predict_kwargs` | `None` | Se pasan a `forecast()`: `symmetric_averaging` y `clip_nonnegative`. Quedan reservadas `context`, `prediction_length`, `observation_mask`, `past_covariates`, `known_covariates`, `categorical_past_covariates` y `categorical_known_covariates`. |

Las opciones de carga de `from_pretrained` (`max_context`, que ya cubre `context_length`; `per_shot_horizon_patches`; `quantile_real_cap_k`; `revision`; `cache_dir`) no se exponen en la primera versión. Otros adapters sí exponen opciones de carga (Chronos y T0 `torch_dtype`, TS-ICL `checkpoint_version`, TimesFM 2.5 `max_horizon`), pero ninguno expone `revision` ni opciones de caché. Se pueden añadir después si hacen falta. Tampoco se pasa `max_context=context_length` a `from_pretrained`: obligaría a recargar el modelo al cambiar `context_length`, y solo importa con horizontes de más de 1.024 pasos (sección 2, Entrada).

### Cambios de código compartido

Son los únicos cambios fuera del adapter. Los dos primeros van en un pull request previo (PR A de la sección 7, decisión D11), que para el usuario solo cambia un caso: Moirai con `device="mps:0"`, que fallaba y pasa a CPU con el aviso (entrada en *Fixed*). El tercero y el cuarto van en el PR del adapter (PR B), porque solo Timestral los necesita: el tercero tiene sentido cuando aparece su segundo usuario, y el cuarto solo se puede probar con él.

**1. Relleno por la izquierda en `FoundationModel` (decisión D5).** La regla `.claude/rules/foundation.md` prohíbe el relleno por adapter. T0 lo hace hoy dentro del adapter y Timestral lo necesita igual. La opción centralizada:

- Un atributo de clase nuevo y obligatorio, `requires_equal_length_context: bool`, añadido a `_AdapterBase._REQUIRED_CLASS_ATTRIBUTES` y declarado en los nueve adapters actuales y en Timestral (`True` en T0 y Timestral; `False` en el resto). Solo tiene sentido con `supports_nan_in_series=True`.
- Un helper en `_utils.py`, `_left_pad_context`, que rellena cada serie con NaN por delante hasta la longitud de la más larga de su grupo, y su `context_exog` igual. El relleno es posicional: todas las series del grupo pasan a un `RangeIndex(width)`. No se generan fechas nuevas, así que no depende de la frecuencia ni de la zona horaria de cada serie.
- En `FoundationModel.predict`, dentro del bucle que ya recorre los grupos (`for series_names_group in series_groups`), el grupo se rellena antes de llamar a `adapter.predict` cuando el atributo es `True`. El índice de la salida se sigue construyendo con el `context` sin rellenar, así que no cambia nada fuera del adapter.

El documento anterior decía que esto no se podía hacer a nivel de pandas porque las series de un grupo pueden terminar en fechas distintas. Es falso: cada serie se rellena sobre su propio índice, así que la fecha final no importa. `exp6_padding_cost.py` lo comprueba con 400 series de 500 a 4096 puntos y 269 fechas finales distintas: el contexto y las exog apiladas con `_left_pad_context` (extraído de este documento) son idénticos a los del relleno con numpy, y el coste es de unos 170 ms por llamada frente a 48 ms (`exp6.log`). Es menos del 0,1 % de los 187.9 s que tarda el 209M con 400 series de solo 512 pasos (`exp4.log`) **[probado]**. Con backends más rápidos (T0, o GPU) su peso relativo es mayor, y en backtesting se paga en cada fold: con un T0 falso y 100 series x 512 con 4 exog, un revisor midió que la versión anterior del helper llevaba la parte de skforecast de `predict` de 74 ms a 201 ms. Por eso el helper solo convierte las columnas booleanas que existen y solo reindexa las series que necesitan relleno; la versión actual da la misma salida que la anterior (comprobado con `assert_series_equal` y `assert_frame_equal`, sin modificar las entradas) y es unas 2,5 veces más rápida **[probado]**. `run_checks.py` comprueba además, con el bucle de este documento insertado en `FoundationModel.predict`, que T0 recibe exactamente los mismos arrays que con su relleno actual, con un índice diario con zona horaria que cruza un cambio de hora y una exog booleana **[probado]**.

Detalles importantes:

- Una columna booleana no admite NaN (pandas la convertiría en `object` y T0 la rechazaría por no numérica), así que `_left_pad_context` convierte las columnas booleanas (también las `boolean` nullable) a float en todas las series. Las columnas enteras de NumPy pasan a float solo en las series que se rellenan, así que dentro de un grupo una columna entera puede llegar como int o como float según la serie; las de tipos nullable (`Int64`, `Float64`) conservan su tipo y llevan `pd.NA` en el relleno (T0 las convierte igual que hoy, comprobado en la revisión final del PR A). Los adapters solo deben fijarse en si una columna es numérica. Las categóricas y de texto conservan su tipo. Comprobado también con pandas 2.2.0 (la mínima admitida) y 3.1.0, sin avisos, por un revisor **[probado]**.
- Las series del contexto que se rellenan y tienen un dtype de NumPy se construyen con `np.full` y una asignación por posición, unas 5 veces más rápido que `reindex` (revisión del PR #1450, anexo A, fila 97): un dtype float se conserva y cualquier otro pasa a float64 (un `int64` sale como `float64`, igual que con `reindex`). Las de tipos nullable (`Float64`, `Int64`) y las que no necesitan relleno siguen por `set_axis`/`reindex`, que conserva su tipo. Las exog no cambian: su `reindex` es lo que conserva las categóricas y los nullable.
- Una primera versión generaba fechas hacia atrás con la frecuencia de cada serie. Falló dos veces en la revisión independiente: desplazaba una hora el índice diario con zona horaria al cruzar un cambio de hora (la serie rellenada quedaba entera a NaN), y `pandas.date_range` lanza `NonExistentTimeError` cuando una fecha de relleno cae en un cambio de hora a medianoche (por ejemplo, el 08/09/2024 en `America/Santiago`), algo que T0 hoy no sufre porque rellena con numpy. Además necesitaba `freq`, que puede faltar en el camino de `check_inputs=False`. El relleno posicional evita los tres problemas, y `run_checks.py` cubre los dos casos de cambio de hora y el de un índice sin `freq` **[probado]**.

<!-- skeleton:shared_utils -->
```python
def _left_pad_context(
    context: dict[str, pd.Series],
    context_exog: dict[str, pd.DataFrame | None] | None,
) -> tuple[dict[str, pd.Series], dict[str, pd.DataFrame | None] | None]:
    """
    Left-pad the context window of every series with NaN to the length of the
    longest one, so that all of them end at their own forecast origin.

    Parameters
    ----------
    context : dict
        Per-series context windows.
    context_exog : dict, None
        Per-series historical exogenous variables, aligned to the index of
        their context.

    Returns
    -------
    context : dict
        Per-series context windows, all of length `width` (the longest
        context) and indexed by `pandas.RangeIndex(width)`.
    context_exog : dict, None
        Per-series historical exogenous variables padded and indexed in the
        same way, or `None` if `context_exog` is `None`.

    Notes
    -----
    The padding is positional: the original timestamps are replaced by a
    `RangeIndex`, so the result does not depend on the frequency or on the
    time zone of each series (no timestamps are generated, which could not
    exist or be ambiguous around a daylight saving change). Adapters that
    declare `requires_equal_length_context` only use the values;
    `FoundationModel` builds the index of the predictions from the unpadded
    context.

    A padded context series with a NumPy dtype is returned as float (a float
    dtype is kept, any other becomes float64). Boolean exog columns (also the
    nullable `boolean` dtype) are cast to float in every series, because a
    bool column cannot hold NaN (pandas would turn it into an `object`
    column). NumPy integer columns become float only in the series that are
    padded, so within a group an integer column can be int or float depending
    on the series. Nullable extension dtypes (`Int64`, `Float64`), in the
    context and in the exog, keep their dtype and hold `pd.NA` in the padded
    rows. Adapters must only rely on whether a column is numeric. The other
    columns keep their dtype. The input objects are not modified.

    """

    width = max(len(series_values) for series_values in context.values())
    index = pd.RangeIndex(width)
    padded_context: dict[str, pd.Series] = {}
    padded_exog: dict[str, pd.DataFrame | None] | None = (
        {} if context_exog is not None else None
    )
    for series_name, series_values in context.items():
        n_pad = width - len(series_values)
        positions = pd.RangeIndex(n_pad, width)
        if n_pad > 0 and isinstance(series_values.dtype, np.dtype):
            # NumPy dtypes: fill an array by position, about 5 times faster
            # than `reindex`. A float dtype is kept, any other becomes float64.
            dtype = (
                series_values.dtype if series_values.dtype.kind == "f" else np.float64
            )
            values = np.full(width, np.nan, dtype=dtype)
            values[n_pad:] = series_values.to_numpy()
            padded_context[series_name] = pd.Series(
                values, index=index, name=series_values.name
            )
        else:
            series_values = series_values.set_axis(positions)
            padded_context[series_name] = (
                series_values.reindex(index) if n_pad > 0 else series_values
            )
        if padded_exog is not None:
            series_exog = context_exog.get(series_name)
            if series_exog is not None:
                series_exog = series_exog.set_axis(positions)
                for col, dtype in series_exog.dtypes.items():
                    if pd.api.types.is_bool_dtype(dtype):
                        series_exog[col] = series_exog[col].astype(float)
                if n_pad > 0:
                    series_exog = series_exog.reindex(index)
            padded_exog[series_name] = series_exog

    return padded_context, padded_exog


def _match_quantile_indices(
    quantile_grid: list[float],
    quantiles: list[float],
    tol: float = 1e-6,
) -> list[int]:
    """
    Map requested quantile levels to column indices in `quantile_grid`.

    Parameters
    ----------
    quantile_grid : list of float
        Quantile levels corresponding to the columns of the model's
        quantile output, in order (e.g. the quantile grid of the loaded
        checkpoint).
    quantiles : list of float
        Requested quantile levels.
    tol : float, default 1e-6
        Maximum allowed absolute difference between a requested
        quantile and its nearest match in `quantile_grid`.

    Returns
    -------
    indices : list of int
        Column index in `quantile_grid` for each entry in
        `quantiles`, in the same order.

    Notes
    -----
    A `ValueError` is raised if a requested quantile has no match
    within `tol`. Matching against the model's own quantile grid
    (synced from the loaded checkpoint) rather than a fixed formula
    keeps the mapping correct even if a checkpoint ships a different
    quantile grid.

    """

    indices = []
    for q in quantiles:
        diffs = [abs(g - q) for g in quantile_grid]
        best_idx = diffs.index(min(diffs))
        if diffs[best_idx] > tol:
            raise ValueError(
                f"Quantile {q} not found in the model's quantile grid "
                f"{quantile_grid} (tolerance {tol})."
            )
        indices.append(best_idx)

    return indices
```

Cambio en el bucle de `FoundationModel.predict` (código actual en `skforecast/foundation/_foundation_model.py`, `FoundationModel.predict`, después de `group_series_by_exog_signature`). Sustituye el bloque que va desde el comentario `# Adapter returns dict[str, np.ndarray] with shape (steps, n_q)` hasta, sin incluirlo, `# Build long-format DataFrame from raw predictions`, y necesita añadir `_left_pad_context` a la importación `from ._utils import (...)` del módulo. `run_checks.py` ejecuta exactamente este bloque, insertado en el código actual de `FoundationModel.predict`:

<!-- skeleton:foundation_loop -->
```python
        # Adapter returns dict[str, np.ndarray] with shape (steps, n_q)
        raw_predictions: dict[str, np.ndarray] = {}
        for series_names_group in series_groups:
            group_context = {
                series_name: context[series_name]
                for series_name in series_names_group
            }
            group_context_exog = (
                {
                    series_name: context_exog[series_name]
                    for series_name in series_names_group
                }
                if context_exog is not None else None
            )
            # Backends that take the context of a call as one array need the
            # same length for every series: left-pad the group with NaN.
            if self.adapter.requires_equal_length_context:
                group_context, group_context_exog = _left_pad_context(
                                                        context      = group_context,
                                                        context_exog = group_context_exog,
                                                    )
            raw_predictions.update(
                self.adapter.predict(
                    steps        = steps,
                    context      = group_context,
                    context_exog = group_context_exog,
                    exog         = (
                        {
                            series_name: exog[series_name]
                            for series_name in series_names_group
                        }
                        if exog is not None else None
                    ),
                    quantiles    = quantiles,
                )
            )
```

Cambio en T0 (mismo PR): `T0Adapter.predict` deja de crear `context_batch` con `np.full` y de copiar cada fila. Pasa a `np.stack` de los arrays (todos de la misma longitud). `T0Adapter._build_future_covariates` no cambia: coloca el histórico pegado al origen y une las columnas de todas las series, que es la traducción en la que se apoya su `supports_heterogeneous_covariates=True`.

**2. MPS en `_resolve_torch_device` (decisión D7).** `_resolve_torch_device` está en `_adapters.py` (no en `_utils.py`). Los tests la parchean como `skforecast.foundation._adapters._resolve_torch_device`, así que debe seguir allí. Se añaden `mps_fallback_message` y `stacklevel`. Si se pasa el mensaje y el device resuelto es MPS (con `"auto"` o explícito), lo lanza como `UserWarning` y devuelve `"cpu"`. El texto y el `stacklevel` los fija quien llama, porque cada adapter está a una profundidad distinta de la llamada del usuario. Moirai pasa a usar el helper con su mensaje actual ("MPS device is not supported by Moirai because the uni2ts library uses float64 operations internally. Falling back to CPU.") y `stacklevel=7` (su 6 actual más el marco del helper), y se elimina su bloque propio. Su aviso sale igual y en el mismo sitio. La única diferencia es que ahora también cubre `"mps:0"`, que antes llegaba a uni2ts y fallaba. Timestral usa `stacklevel=6`: `run_checks.py` comprueba que el aviso apunta a la llamada del usuario a `ForecasterFoundation.predict` **[probado]**.

<!-- skeleton:resolve_device -->
```python
def _resolve_torch_device(
    device: str,
    mps_fallback_message: str | None = None,
    stacklevel: int = 2,
) -> str:
    """
    Resolve a device string to a concrete PyTorch device name.

    If `device` is `"auto"`, the best available accelerator is selected
    in priority order: CUDA > MPS (Apple Silicon) > CPU.

    Parameters
    ----------
    device : str
        Device string. Use `"auto"` for automatic selection, or an
        explicit name such as `"cuda"`, `"mps"`, or `"cpu"`.
    mps_fallback_message : str, default None
        For backends that do not run on MPS. If given and the resolved
        device is MPS (`"auto"` on Apple Silicon without CUDA, or an
        explicit `"mps"` or `"mps:<index>"`), this message is issued as a
        `UserWarning` and `"cpu"` is returned. If `None`, MPS is allowed.
    stacklevel : int, default 2
        `stacklevel` of the warning, so that it points to the user's call.

    Returns
    -------
    device : str
        Resolved device name.

    """

    if device == "auto":
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if mps_fallback_message is not None and device.startswith("mps"):
        warnings.warn(mps_fallback_message, stacklevel=stacklevel)
        device = "cpu"

    return device
```

**3. `_match_quantile_indices` a `_utils.py` (en el PR B, paso B2).** Hoy es un método estático de `TimesFM3Adapter`. Se mueve a `_utils.py` cuando Timestral pasa a ser su segundo usuario (está en el bloque del punto 1), y `TimesFM3Adapter.predict` lo importa de allí. Código y docstring son los actuales, Notes incluidas; solo cambia el ejemplo de `quantile_grid` (`self._model.config.quantiles`), que fuera de la clase no tiene sentido. Ningún test lo llama directamente **[verificado en código]**.

**4. `tools/check_foundation_models_metadata.py` (en el PR B, paso B2).** En `check_model`, cuando la tarjeta dice `license: other` se compara con `license_name`. Si falta, como en Timestral, se compara `"none"` con `"other"` y falla. Cambio: `hub_license = card_data.get("license_name") or "other"`.

### Esqueleto del adapter

Va en `skforecast/foundation/_adapters.py`, después de `NoriAdapter` (la última clase de adapter del módulo) y justo antes de `_ADAPTER_REGISTRY`. Importaciones que hay que añadir al principio del módulo:

```python
from ..exceptions import FeatureOutOfRangeWarning
from ._utils import (
    ...,                       # las actuales
    _match_quantile_indices,   # movido desde TimesFM3Adapter (cambio compartido 3)
)
```

<!-- skeleton:adapter -->
```python
class TimestralAdapter(_AdapterBase):
    """
    Adapter for Mistral AI Timestral foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"mistralai/Timestral-28M"`. Must start
        with `"mistralai/Timestral"`.
    model : object, default None
        Pre-loaded `TimestralForecaster` instance. If `None`, the model is
        loaded lazily on the first `predict` call.
    context_length : int, default 4096
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it
        is trimmed to this length; if it is shorter, all available
        observations are used as-is. Must be a positive integer. Timestral
        supports context lengths up to 32,768: with a larger value, the
        backend only uses the last 32,768 observations.
    device : str, default 'auto'
        Device placement for the model. `"auto"` selects the best available
        accelerator (CUDA > MPS > CPU). Also accepts explicit values such as
        `"cuda"`, `"mps"`, or `"cpu"`, forwarded to
        `TimestralForecaster.from_pretrained`. MPS is not supported by
        Timestral 1.0.0: it falls back to CPU with a warning.
    predict_kwargs : dict, default None
        Additional keyword arguments forwarded verbatim to `forecast` (e.g.
        `symmetric_averaging`, `clip_nonnegative`). Cannot include
        `context`, `prediction_length`, `observation_mask`,
        `past_covariates`, `known_covariates`, `categorical_past_covariates`
        or `categorical_known_covariates`, which are managed internally.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict, None
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    device : str
        Device placement for the model.
    predict_kwargs : dict
        Additional keyword arguments forwarded to `forecast`.
    allow_exog : bool
        Whether this adapter accepts exogenous variables. Always `True`.
    supports_past_only_covariates : bool
        Whether historical exog columns without future values are used as
        past-only covariates. Always `True`.
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. Always `False`: `forecast` takes the
        covariates of every series as one array, so `FoundationModel`
        groups the series by covariate signature and calls `predict` once
        per group.
    requires_equal_length_context : bool
        Whether the backend needs every series of a call to have the same
        context length. Always `True`: `forecast` takes the context of every
        series as one array, so `FoundationModel` left-pads the context and
        the historical exog of each group with NaN, which Timestral treats
        as missing values.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context. Always `True`.
    supports_categorical_covariates : bool
        Whether the backend supports non-numeric covariates natively, so
        they do not have to be encoded as numbers. `True`: the adapter maps
        the labels of each non-numeric column to integer codes and Timestral
        encodes them per series.
    SUPPORTED_QUANTILES : list, None
        Quantile levels accepted by the backend. `None` means any level in
        `(0, 1)`.
    requires_hf_auth : bool
        Whether the checkpoints served by this adapter are gated on the
        Hugging Face Hub, so an authenticated account that has accepted the
        model license is needed. Declared per adapter, not per checkpoint.
    requires_provider_auth : bool
        Whether the model provider requires its own account or license
        acceptance, outside the Hugging Face Hub, before the weights can be
        used.
    weights_repo_id : str, None
        Hugging Face repository the backend downloads the weights from when
        it is not `model_id`. `None` means the repository is `model_id`.
        `None` for Timestral: the weights are downloaded from `model_id`.
    weights_in_hf_cache : bool
        Whether the downloaded weights are stored in the Hugging Face Hub
        cache, under `weights_repo_id`.
    backend_package : str
        Package that provides the backend, as passed to `pip install`.
    default_model_id : str
        Model ID used by default for this adapter (e.g. by skforecast
        documentation and tools that need a concrete checkpoint).
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    Timestral supports only 21 fixed quantile levels: 0.01, 0.05, every
    multiple of 0.05 up to 0.95, and 0.99. Requesting any other level raises
    a `ValueError`. The point forecast is the median (quantile 0.5).

    Each series is forecast as an independent target. For each series,
    columns present in its future `exog` become known covariates spanning
    `context + horizon`, built by concatenating the matching historical
    column from `context_exog` with the future values; columns present only
    in its `context_exog` become past covariates. Numeric and boolean columns
    are passed as numeric covariates. Any other column is passed as a
    categorical covariate: its labels are mapped to integer codes over the
    history and the horizon of each series, and missing labels become NaN.
    Whether a column is numeric is decided by its dtype, which is the same
    in every series. NaN values in the series and in the numeric covariates
    are treated as missing values by the backend; in a categorical
    covariate, a missing label is encoded as a category of its own (except
    in the left padding, which is ignored). Infinite values in numeric
    covariates are turned into NaN.

    A known covariate that is constant over the context and changes over the
    forecast horizon (for example a price, or a 0/1 indicator that is 0 over
    the whole context) can make Timestral 1.0.0 return very large quantiles
    (mistralai/timestral#7). The adapter raises a `FeatureOutOfRangeWarning`
    naming the affected series and columns. Setting the historical values of
    those columns to NaN avoids the issue.

    The pre-trained weights are released under a license that excludes
    companies above a revenue threshold, so loading them raises a
    `LicenseWarning`.

    References
    ----------
    .. [1] https://github.com/mistralai/timestral

    .. [2] https://huggingface.co/collections/mistralai/timestral

    """

    SUPPORTED_QUANTILES: list[float] = [
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
    ]
    allow_exog: bool = True
    supports_past_only_covariates: bool = True
    supports_categorical_covariates: bool = True
    supports_heterogeneous_covariates: bool = False
    requires_equal_length_context: bool = True
    supports_nan_in_series: bool = True
    requires_hf_auth: bool = False
    requires_provider_auth: bool = False
    weights_repo_id: str | None = None
    weights_in_hf_cache: bool = True
    backend_package: str = "timestral @ git+https://github.com/mistralai/timestral.git"
    default_model_id: str = "mistralai/Timestral-28M"

    _MODEL_ID_PREFIX: str = "mistralai/Timestral"
    _RESERVED_PREDICT_KWARGS: frozenset[str] = frozenset({
        "context", "prediction_length", "observation_mask", "past_covariates",
        "known_covariates", "categorical_past_covariates",
        "categorical_known_covariates",
    })

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 4096,
        device: str = "auto",
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. `"mistralai/Timestral-28M"`. Must
            start with `"mistralai/Timestral"`.
        model : object, default None
            Pre-loaded `TimestralForecaster` instance. If `None`, the model
            is loaded lazily on the first `predict` call.
        context_length : int, default 4096
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `context` is
            longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        device : str, default 'auto'
            Device placement for the model. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU). MPS falls back to CPU
            with a warning.
        predict_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to `forecast`.

        """

        _validate_model_id_prefix(model_id, self._MODEL_ID_PREFIX, type(self).__name__)
        _validate_positive_int("context_length", context_length)
        predict_kwargs = predict_kwargs or {}
        self._validate_predict_kwargs(predict_kwargs)

        self.model_id       = model_id
        self._model         = model
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.device         = device
        self.predict_kwargs = predict_kwargs
        self.is_fitted      = False

    @classmethod
    def _validate_predict_kwargs(cls, predict_kwargs: dict[str, Any]) -> None:
        """
        Reject `predict_kwargs` keys that the adapter manages itself.

        Parameters
        ----------
        predict_kwargs : dict
            Candidate `predict_kwargs` value.

        Returns
        -------
        None

        Notes
        -----
        A `ValueError` naming the offending keys is raised if any key in
        `_RESERVED_PREDICT_KWARGS` is present. They are built internally
        from `context`, `context_exog`, `exog` and `steps`, so passing them
        explicitly would collide with the internal `forecast` call.

        """

        reserved = cls._RESERVED_PREDICT_KWARGS & set(predict_kwargs)
        if reserved:
            raise ValueError(
                f"`predict_kwargs` cannot include {sorted(reserved)}. These "
                f"arguments are managed internally by {cls.__name__}."
            )

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `device`, `predict_kwargs`.

        """

        return {
            "model_id":       self.model_id,
            "context_length": self.context_length,
            "device":         self.device,
            "predict_kwargs": self.predict_kwargs or None,
        }

    def set_params(self, **params) -> TimestralAdapter:
        """
        Set adapter parameters. Resets the model when parameters that affect
        loading change.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `device`,
            `predict_kwargs`.

        Returns
        -------
        self : TimestralAdapter

        Notes
        -----
        Only `model_id` and `device` affect the loaded model, so only those
        discard the cached model. `context_length` and `predict_kwargs` are
        applied at predict time and never trigger a reload.

        """

        def validate(candidate_params: dict) -> dict:
            if "model_id" in candidate_params:
                _validate_model_id_prefix(
                    candidate_params["model_id"],
                    self._MODEL_ID_PREFIX,
                    type(self).__name__,
                )
            if "context_length" in candidate_params:
                _validate_positive_int(
                    "context_length", candidate_params["context_length"]
                )
            if "predict_kwargs" in candidate_params:
                candidate_params["predict_kwargs"] = (
                    candidate_params["predict_kwargs"] or {}
                )
                self._validate_predict_kwargs(candidate_params["predict_kwargs"])
            return candidate_params

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                ({"model_id", "device"}, lambda: setattr(self, "_model", None)),
            ),
        )

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> TimestralAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since Timestral is a zero-shot inference
        model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : TimestralAdapter

        """

        return super().fit(context=context, context_exog=context_exog)

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float, ...] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using the Timestral model.

        All input normalization, validation, context trimming and padding is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict
            Per-series context windows (already trimmed to
            `context_length` and left-padded with NaN to the same length).
        context_exog : dict, None
            Per-series historical exogenous variables (already trimmed and
            padded). For each series, columns not also present in its `exog`
            are forwarded as past covariates.
        exog : dict, None
            Per-series future exogenous variables for the forecast
            horizon. Each column is forwarded as a known covariate of that
            series, concatenated with its historical values.
        quantiles : list of float or None
            Quantile levels. Must be a subset of `SUPPORTED_QUANTILES`.

        Returns
        -------
        predictions : dict
            Keys are series names, in the same order as `context`. Each
            value is a 2-D array of shape `(steps, n_quantiles)`.

        Notes
        -----
        A `ValueError` is raised if a requested quantile level is not in
        `SUPPORTED_QUANTILES`.

        `forecast` requires all series in one call to share the same
        context length and covariate layout. `FoundationModel` guarantees
        both: it groups the series by covariate signature
        (`get_exog_signature`), left-pads each group and calls `predict`
        once per group, so the signature of the first series sets the
        column order for the whole batch. The point forecast
        (`quantiles is None`) is the model's median quantile.

        """

        quantile_list = _validate_supported_quantiles(
            quantiles, self.SUPPORTED_QUANTILES, "Timestral"
        )

        # NOTE: the model is loaded lazily here so that the adapter can be
        # instantiated and fitted without requiring timestral to be installed.
        self._load_model()

        names = list(context.keys())

        # Every series in a call shares the same covariate columns
        # (`FoundationModel` groups them by covariate signature), so the
        # signature of the first series sets the column order for the batch.
        past_only_cols, fut_cols = get_exog_signature(
            context_exog = context_exog.get(names[0]) if context_exog is not None else None,
            exog         = exog.get(names[0]) if exog is not None else None,
        )
        categorical_cols = self._categorical_columns(
            context_exog   = context_exog.get(names[0]) if context_exog is not None else None,
            exog           = exog.get(names[0]) if exog is not None else None,
            past_only_cols = past_only_cols,
            fut_cols       = fut_cols,
        )
        if fut_cols:
            self._warn_if_constant_known_covariates(
                names        = names,
                context_exog = context_exog,
                exog         = exog,
                fut_cols     = fut_cols,
            )

        covariates = [
            self._build_covariates(
                context_exog     = (
                    context_exog.get(series_name) if context_exog is not None else None
                ),
                exog             = exog.get(series_name) if exog is not None else None,
                past_only_cols   = past_only_cols,
                fut_cols         = fut_cols,
                categorical_cols = categorical_cols,
            )
            for series_name in names
        ]
        covariate_kwargs = {
            key: np.stack([series_covariates[key] for series_covariates in covariates])
            for key in covariates[0]
        }

        forecast = self._model.forecast(
            context           = np.stack([
                                    context[series_name].to_numpy(dtype=np.float64)
                                    for series_name in names
                                ])[:, np.newaxis, :],
            prediction_length = steps,
            **covariate_kwargs,
            **self.predict_kwargs,
        )

        # Shape (n_series, steps, n_levels): one target per series.
        q_arr = _tensor_to_numpy(forecast.quantiles)[:, 0]
        quantile_indices = _match_quantile_indices(
            list(forecast.quantile_levels),
            quantile_list if quantile_list is not None else [0.5],
        )

        return {
            series_name: q_arr[i][:, quantile_indices]
            for i, series_name in enumerate(names)
        }

    @staticmethod
    def _is_categorical(col_data: pd.Series) -> bool:
        """
        Whether a covariate column is passed to Timestral as categorical.

        Parameters
        ----------
        col_data : pandas Series
            A single covariate column.

        Returns
        -------
        is_categorical : bool
            `False` for numeric and boolean columns, `True` otherwise.

        """

        return not (
            pd.api.types.is_numeric_dtype(col_data)
            or pd.api.types.is_bool_dtype(col_data)
        )

    @classmethod
    def _categorical_columns(
        cls,
        context_exog: pd.DataFrame | pd.Series | None,
        exog: pd.DataFrame | pd.Series | None,
        past_only_cols: tuple,
        fut_cols: tuple,
    ) -> set:
        """
        Return the covariate columns that are passed as categorical
        covariates, from the exogenous variables of one series.

        Parameters
        ----------
        context_exog : pandas DataFrame, pandas Series, None
            Historical exogenous variables of the first series of the batch.
        exog : pandas DataFrame, pandas Series, None
            Future exogenous variables of the first series of the batch.
        past_only_cols : tuple
            Past-only covariate columns of the batch.
        fut_cols : tuple
            Known-future covariate columns of the batch.

        Returns
        -------
        categorical_cols : set
            Past-only columns that are not numeric in `context_exog` and
            known-future columns that are not numeric in `exog`.

        Notes
        -----
        One series is enough: `FoundationModel` rejects exog columns whose
        dtype differs between series (`check_preprocess_exog_multiseries`).

        """

        ctx_df = (
            context_exog.to_frame()
            if isinstance(context_exog, pd.Series)
            else context_exog
        )
        fut_df = (
            exog.to_frame()
            if isinstance(exog, pd.Series)
            else exog
        )

        return {
            col for col in past_only_cols if cls._is_categorical(ctx_df[col])
        } | {
            col for col in fut_cols if cls._is_categorical(fut_df[col])
        }

    @staticmethod
    def _to_float_array(col_data: pd.Series) -> np.ndarray:
        """
        Convert a numeric or boolean covariate column to a `float64` array.

        Parameters
        ----------
        col_data : pandas Series
            A single covariate column.

        Returns
        -------
        col_array : numpy ndarray
            1-D `float64` array, with NaN for missing and infinite values.

        Notes
        -----
        Timestral masks infinite values in the target but not in the
        numeric covariates, so they are turned into NaN here.

        """

        col_array = col_data.to_numpy(dtype=np.float64, na_value=np.nan)

        return np.where(np.isinf(col_array), np.nan, col_array)

    @staticmethod
    def _to_codes(col_data: pd.Series) -> np.ndarray:
        """
        Map the labels of a categorical covariate column to integer codes.

        Parameters
        ----------
        col_data : pandas Series
            A single categorical covariate column, with the history followed
            by the future values for known covariates, so that a label gets
            the same code in both.

        Returns
        -------
        col_array : numpy ndarray
            1-D `float64` array with the code of each label
            (`pandas.factorize`), and NaN for missing labels.

        """

        codes, _ = pd.factorize(col_data, use_na_sentinel=True)

        return np.where(codes < 0, np.nan, codes).astype(np.float64)

    @classmethod
    def _build_covariates(
        cls,
        context_exog: pd.DataFrame | pd.Series | None,
        exog: pd.DataFrame | pd.Series | None,
        past_only_cols: tuple,
        fut_cols: tuple,
        categorical_cols: set,
    ) -> dict[str, np.ndarray]:
        """
        Build the covariate arrays for one series from its own exogenous
        variables, keyed by the `forecast` argument they are passed to.

        Parameters
        ----------
        context_exog : pandas DataFrame, pandas Series, None
            Historical exogenous variables aligned to the context for this
            series. Must contain every column in `past_only_cols` and
            `fut_cols`.
        exog : pandas DataFrame, pandas Series, None
            Future-known exogenous variables covering the forecast horizon
            for this series. Must contain every column in `fut_cols`.
        past_only_cols : tuple
            Past-only covariate columns, as returned by `get_exog_signature`
            for the first series of the group.
        fut_cols : tuple
            Known-future covariate columns, as returned by
            `get_exog_signature` for the first series of the group.
        categorical_cols : set
            Columns passed as categorical covariates, as returned by
            `_categorical_columns`.

        Returns
        -------
        covariates : dict
            Keys among `past_covariates` (shape `(n_cols, context_len)`),
            `known_covariates` (shape `(n_cols, context_len + steps)`),
            `categorical_past_covariates` and `categorical_known_covariates`.
            A key is present only when it has at least one column.

        """

        ctx_df = (
            context_exog.to_frame()
            if isinstance(context_exog, pd.Series)
            else context_exog
        )
        fut_df = (
            exog.to_frame()
            if isinstance(exog, pd.Series)
            else exog
        )

        rows: dict[str, list[np.ndarray]] = {}
        for col in past_only_cols:
            if col in categorical_cols:
                key, values = "categorical_past_covariates", cls._to_codes(ctx_df[col])
            else:
                key, values = "past_covariates", cls._to_float_array(ctx_df[col])
            rows.setdefault(key, []).append(values)
        for col in fut_cols:
            if col in categorical_cols:
                key = "categorical_known_covariates"
                values = cls._to_codes(
                    pd.concat([ctx_df[col], fut_df[col]], ignore_index=True)
                )
            else:
                key = "known_covariates"
                values = np.concatenate([
                    cls._to_float_array(ctx_df[col]),
                    cls._to_float_array(fut_df[col]),
                ])
            rows.setdefault(key, []).append(values)

        return {key: np.stack(key_rows) for key, key_rows in rows.items()}

    @staticmethod
    def _warn_if_constant_known_covariates(
        names: list[str],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
        exog: dict[str, pd.DataFrame | pd.Series | None],
        fut_cols: tuple,
    ) -> None:
        """
        Warn when a numeric known covariate is constant over the context and
        takes other values over the forecast horizon.

        Parameters
        ----------
        names : list of str
            Series of the batch.
        context_exog : dict
            Per-series historical exogenous variables.
        exog : dict
            Per-series future exogenous variables.
        fut_cols : tuple
            Known-future covariate columns of the batch.

        Returns
        -------
        None

        Notes
        -----
        Timestral 1.0.0 can return very large quantiles in that case
        (mistralai/timestral#7). Only numeric columns are checked.

        """

        affected = []
        for series_name in names:
            ctx_df = context_exog[series_name]
            fut_df = exog[series_name]
            if isinstance(ctx_df, pd.Series):
                ctx_df = ctx_df.to_frame()
            if isinstance(fut_df, pd.Series):
                fut_df = fut_df.to_frame()
            for col in fut_cols:
                if not pd.api.types.is_numeric_dtype(fut_df[col]):
                    continue
                history = ctx_df[col].dropna()
                if (
                    history.nunique() == 1
                    and (fut_df[col].dropna() != history.iloc[0]).any()
                ):
                    affected.append((series_name, col))

        if affected:
            warnings.warn(
                f"Known covariates that are constant over the context and "
                f"change over the forecast horizon, as (series, column): "
                f"{affected}. Timestral can return very large quantiles in "
                f"this case (mistralai/timestral#7). Setting their historical "
                f"values to NaN avoids it.",
                FeatureOutOfRangeWarning,
                stacklevel=5,
            )

    def _load_model(self) -> None:
        """
        Load the Timestral model into `self._model` if not already set.

        Returns
        -------
        None

        Notes
        -----
        This method is a no-op when `self._model` is already populated
        (either by a prior call or by the `model` constructor argument).
        The model is imported lazily from `timestral` and loaded via
        `TimestralForecaster.from_pretrained`, resolving `self.device` to a
        concrete device name first. Timestral 1.0.0 computes its scaling
        statistics in float64, which MPS does not support
        (mistralai/timestral#6), so MPS falls back to CPU with a warning.
        `timestral` must be installed; an `ImportError` is raised otherwise.
        A `LicenseWarning` is issued immediately before the weights are
        loaded.

        """

        if self._model is not None:
            return
        try:
            from timestral import TimestralForecaster
        except ImportError as exc:
            raise ImportError(
                f"timestral is required for TimestralAdapter. "
                f'Install it with `pip install "{self.backend_package}"`.'
            ) from exc

        _warn_if_non_commercial(self.model_id)

        self._model = TimestralForecaster.from_pretrained(
            self.model_id,
            device=_resolve_torch_device(
                self.device,
                mps_fallback_message=(
                    "MPS device is not supported by Timestral because the "
                    "timestral library uses float64 operations internally. "
                    "Falling back to CPU."
                ),
                stacklevel=6,
            ),
        )
```

Cada serie va como un objetivo independiente (`V = 1`), el comportamiento por defecto del resto de adapters (Chronos-2 con `cross_learning=False`). El modo multivariante de Timestral, con varias series como variables de un mismo objeto que se predicen conjuntamente, sería el equivalente de `cross_learning`. Queda fuera de la primera versión: las exog de cada serie tendrían que apilarse como filas de covariables compartidas, y con varios objetivos el backend ya no aplica la *target encoding* a las categóricas.

### Decisiones de diseño y su justificación

1. **Relleno por la izquierda:** en `FoundationModel` (cambio compartido 1). El adapter solo apila (`np.stack`).
2. **Coste de los lotes grandes.** La atención entre variables se calcula sobre todas las filas de la llamada, y `FoundationModel` llama al adapter una vez por grupo de columnas, sin límite de series. Con 400 series y covariables, trocear en grupos de 100 ahorra un 34 % en CPU, y en GPU evitaría reservar memoria de más (issue #6). La opción centralizada es un atributo de clase nuevo y obligatorio (por ejemplo, `max_series_per_call: int | None`, con `None` en el resto de adapters), aplicado en el mismo bucle de grupos de `FoundationModel`. Recomendación: un PR aparte, después del adapter (decisión D6). Sin él, el adapter funciona, pero los lotes grandes con covariables van más lentos.
3. **MPS:** en el helper compartido (cambio compartido 2), con el comportamiento de Moirai. Se puede quitar cuando Mistral corrija el issue #6.
4. **Índices de cuantiles:** helper compartido (cambio compartido 3).
5. **Issue #7 (decisión D3).** Es un fallo del backend, no del contrato. El adapter no debe esconderlo cambiando las covariables del usuario, porque sería un preprocesado propio de un solo backend. Recomendación:
   - Mantener los valores por defecto del backend (`symmetric_averaging=True`), como hacen los demás adapters.
   - Lanzar un aviso barato, `FeatureOutOfRangeWarning` ("feature out of the range seen during training"), cuando una covariable futura numérica es constante en el contexto y cambia en el horizonte (`_warn_if_constant_known_covariates`). El aviso nombra las series y columnas afectadas y el remedio.
   - Documentar el problema y el único remedio que funciona en todos los casos medidos: poner a NaN el histórico de la covariable. `predict_kwargs={"symmetric_averaging": False}` no es un remedio general: con un precio solo baja la mediana del 209M y en el 5M empeora el q99; con un indicador lo arregla en el 209M y solo en parte en el 28M (`exp5.log`).
   - Afecta también a los indicadores 0/1 (festivos, promociones), también en el 28M, el `default_model_id` propuesto. El aviso puede saltar en casos en los que la previsión no se rompe (por ejemplo, `Holiday` en un fold de la guía de usuario), así que el texto del aviso habla de riesgo, no de error.
   - Revisarlo cuando Mistral responda al issue.
   - El bucle por series del aviso es una comprobación propia del backend, no manejo genérico de la entrada, así que se queda en el adapter.
6. **Categóricas.** `pd.factorize` por serie traduce valores al formato del backend, igual que `ChronosAdapter._to_covariate_array` o el bucle por serie de `TimesFM3Adapter`. Es traducción, así que se queda en el adapter. El tipo de cada columna se decide con la primera serie del grupo: `FoundationModel` ya rechaza las columnas con distinto dtype entre series (`check_preprocess_exog_multiseries`) **[probado]**.
7. **Infinitos en las covariables numéricas:** se convierten a NaN (`_to_float_array`), porque el backend los enmascara en el objetivo pero no en las covariables.
8. **`_validate_predict_kwargs`:** se copia de `TimesFM3Adapter` (classmethod del adapter). Con dos adapters no compensa un helper compartido; si llega un tercero, se puede mover a `_utils.py`.

`_AdapterBase` cambia en un único punto: el atributo obligatorio `requires_equal_length_context` (y `max_series_per_call` si se hace el PR C).

## 5. Alineación con los adapters existentes

Comparación del esqueleto con `TimesFM3Adapter`, `ChronosAdapter` (el adapter de Chronos-2) y `T0Adapter`, en el commit de referencia **[verificado en código]**. "Distinto, a corregir" ya no aparece: las filas que lo eran en la versión anterior del documento están corregidas en el esqueleto (anexo A).

| Aspecto | TimesFM3 / Chronos / T0 | Esqueleto | Clasificación |
|:--------|:------------------------|:----------|:--------------|
| Docstring de clase | Parameters, Attributes, Notes, References. TimesFM3 documenta todos los atributos de capacidad, también `allow_exog` y `supports_past_only_covariates` (Chronos y T0 los omiten). | Estructura de TimesFM3, más `requires_equal_length_context`. Notes: cuantiles fijos, mediana, covariables, categóricas, NaN, issue #7 y licencia. | Igual (TimesFM3) |
| Orden y anotación de los atributos de clase | Orden fijo, todos anotados | El mismo orden y tipos, con `requires_equal_length_context` tras `supports_heterogeneous_covariates` | Igual (el atributo nuevo lo declaran todos tras el PR A) |
| `_MODEL_ID_PREFIX`, `_RESERVED_PREDICT_KWARGS` | `_MODEL_ID_PREFIX` en TimesFM 2.5 y TimesFM 3.0; `_RESERVED_PREDICT_KWARGS` solo en TimesFM 3.0; tras los atributos públicos | Los dos | Igual (TimesFM3) |
| Firma del constructor | TimesFM3 `(model_id, *, model=None, context_length=2048, device="auto", predict_kwargs=None)`. Chronos `(model_id, *, pipeline, context_length, predict_kwargs, device_map, torch_dtype, cross_learning)`. T0 `(model_id, *, model, context_length, device_map, torch_dtype)`. | `(model_id, *, model=None, context_length=4096, device="auto", predict_kwargs=None)` | Igual que TimesFM3. Frente a Chronos y T0: `device` y no `device_map`, y sin `torch_dtype`; distinto y justificado, porque `from_pretrained` de Timestral recibe `device` y convierte todo a float32. |
| `context_length` por defecto | 2048 / 8192 / 8192 | 4096 | Distinto y justificado (medido, sección 2) |
| Docstring y validación de `__init__` | TimesFM3: "Initialise the adapter." con Parameters; valida prefijo, entero positivo y claves reservadas | Igual | Igual (TimesFM3) |
| Bloque de asignación | `=` alineados; orden `model_id, _model, context_, context_exog_, context_length, device, predict_kwargs, is_fitted` | Igual | Igual |
| `get_params` | Claves `model_id, context_length, device, predict_kwargs`, `predict_kwargs or None`, docstring "Keys: ..."; comillas simples | Igual, con comillas dobles | Distinto y justificado: el repo pide comillas dobles (`tools/ai/ai_context_header.md`, ruff `quote-style = "double"`); el código existente usa simples porque el formateador no se aplica (`ruff check` solo selecciona E y F) |
| `set_params` | Cierre `validate` (prefijo, entero positivo, `predict_kwargs or {}` y claves reservadas), `resets` en varias líneas, docstring con Notes | Igual | Igual (TimesFM3) |
| `fit` | Anotado, docstring, `return super().fit(...)` | Igual | Igual |
| Orden de `predict` | TimesFM3: valida cuantiles, carga, `names`, firma de la primera serie, covariables por serie, llamada, `_match_quantile_indices` | Igual, más la detección de categóricas y el aviso del issue #7 antes de las covariables | Igual (TimesFM3) |
| Previsión puntual | TimesFM3 usa `out.forecast`; Chronos y T0 piden `[0.5]` | Columna de la mediana (`[0.5]`) | Distinto de TimesFM3 y justificado (el backend no da una salida puntual aparte); igual que Chronos y T0 |
| Comentario `# NOTE: the model is loaded lazily here ...` | Chronos, T0 y TS-ICL; TimesFM3 no | Presente | Igual (patrón mayoritario) |
| Carga perezosa | `_load_model` (TimesFM3, T0), `_load_pipeline` (Chronos); `_warn_if_non_commercial` justo antes de `from_pretrained` | `_load_model`, aviso de licencia antes de `from_pretrained`, device resuelto en la llamada | Igual (TimesFM3) |
| Texto del `ImportError` | `_import_timesfm`: "timesfm is required for ... Install it with `pip install "{backend_package}"`". T0 y TS-ICL: "{backend_package} is required for XAdapter ..." | "timestral is required for TimestralAdapter. Install it with `pip install "{backend_package}"`." | Igual que TimesFM; distinto de T0 y justificado (la URL no sirve como sujeto de la frase, y las comillas son necesarias por los espacios de la forma PEP 508) |
| MPS | Moirai: `UserWarning` "MPS device is not supported by Moirai ..." y CPU | El mismo comportamiento, desde el helper compartido | Igual (Moirai), centralizado |
| Errores y avisos | `ValueError` de cuantiles (helper), `LicenseWarning` (helper), `ValueError` por covariables no numéricas (TimesFM3, T0) | `ValueError` y `LicenseWarning` de los helpers; `FeatureOutOfRangeWarning` para el issue #7; sin error de no numéricas (admite categóricas) | Igual en lo común; el aviso nuevo es distinto y justificado |
| Conversores | `_to_covariate_array` (Chronos, TimesFM3, TS-ICL), `_to_float_array` (T0, Nori; recibe `pd.Series`) | `_to_float_array` (como T0), más `_to_codes`, `_is_categorical` y `_categorical_columns` para las categóricas | Igual en el nombre; los helpers de categóricas son distintos y justificados |
| `_build_covariates` | TimesFM3: classmethod por serie que devuelve una tupla `(past_only, past_future)` | Classmethod por serie que devuelve un dict con las claves de `forecast` que tienen columnas | Distinto y justificado (cuatro argumentos del backend en vez de dos) |
| Relleno | T0 rellena dentro del adapter | Ninguno: lo hace `FoundationModel` | Distinto de T0; T0 se migra en el PR A |
| Llamadas con argumentos alineados | `context_exog   = ...`, `exog           = ...` | Igual | Igual |
| Densidad de comentarios | TimesFM3: comentarios cortos antes de los pasos no evidentes | Igual | Igual |

## 6. Decisiones abiertas

Cada decisión tiene opciones, una recomendación y lo que cambia según la respuesta. El esqueleto de la sección 4 implementa las recomendaciones.

### D1. Instalación del backend

- **Opciones:**
  - (a) Publicar ya el adapter con `backend_package = "timestral @ git+https://github.com/mistralai/timestral.git"`, sin fijar commit, y cambiar a `"timestral"` cuando salga en PyPI.
  - (b) Igual que (a), pero fijando el commit (`...timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`).
  - (c) Esperar a PyPI (issue #5, sin respuesta).
- **Recomendación:** (a). Ningún otro backend se fija a una versión, y el check semanal de `tools/foundation_compat` sirve precisamente para detectar si un cambio de `main` rompe el adapter. Con (b), ese check no vería los cambios y los usuarios se quedarían en el commit fijado. sktime y darts lo integran con la URL de git (PR abiertos).
- **Qué cambia:** con (b), solo la cadena (en la clase, en `adapters.toml` y en la documentación). Con (c), se aplaza todo; el diseño no cambia salvo `backend_package = "timestral"`. En cualquier caso, los usuarios detrás de un firewall que bloquee git no podrán instalarlo hasta que esté en PyPI.
- **Decisión:** (a), aceptada el 05/11/2026.

### D2. Licencia

- **Opciones:** (a) Registrarla con `commercial_use_restricted=True`, para que salga la `LicenseWarning` al cargar los pesos. (b) `False`, porque no es una licencia "no comercial" en sentido estricto.
- **Recomendación:** (a). Deja fuera a las empresas grandes, también para su uso interno, y el docstring de `LicenseWarning` habla de "restrictions on commercial or production use". TabPFN es un precedente: está registrada como restringida con "non-commercial without an enterprise license".
- **Qué cambia:** con (a), la entrada de `_MODEL_LICENSES` de la sección 4, los tests de licencia (sección 8) y las tablas de licencias de la documentación. El texto que todavía dice "non-commercial" (el docstring de `_warn_if_non_commercial` y la skill `foundation-forecasting`) se puede ajustar a "commercial-use restriction". Con (b), el tercer campo pasa a `False`, el cuarto a `None`, y desaparece el aviso.
- **Decisión:** (a), aceptada el 05/11/2026.

### D3. Issue #7 (covariables futuras planas)

- **Opciones:**
  - (a) Solo documentarlo.
  - (b) Documentarlo y avisar con `FeatureOutOfRangeWarning` cuando una covariable futura numérica es constante en el contexto y cambia en el horizonte.
  - (c) Que el adapter ponga a NaN el histórico de esas covariables.
  - Aparte, mantener o no `symmetric_averaging=True` por defecto.
- **Recomendación:** (b) y mantener `symmetric_averaging=True`, el valor de Mistral y de sus benchmarks. (c) cambia los datos del usuario sin decírselo y sería un preprocesado propio de un backend.
- **Qué cambia:** con (a), se eliminan `_warn_if_constant_known_covariates`, su llamada en `predict`, la importación de `FeatureOutOfRangeWarning` y sus tests, y el párrafo de Notes cambia. Con (c), el método pasa a enmascarar el histórico en lugar de avisar. Para desactivar el promedio por defecto, `__init__` pondría `{"symmetric_averaging": False}` en `predict_kwargs` (y `get_params` dejaría de devolver `None`).
- **Decisión:** (b) y `symmetric_averaging=True`, aceptada el 05/11/2026.

### D4. Categóricas

- **Opciones:** (a) `supports_categorical_covariates=True`, con el adapter traduciendo las etiquetas a códigos. (b) `False`, rechazando las columnas no numéricas como TimesFM 3.0.
- **Recomendación:** (a). El backend tiene tratamiento específico para las categóricas, y Timestral sería el segundo adapter (tras Chronos-2) que las admite.
- **Qué cambia:** con (b), se eliminan `_is_categorical`, `_categorical_columns`, `_to_codes` y las claves `categorical_*` de `_build_covariates`, y `_to_float_array` lanza `ValueError` con columnas no numéricas, con el texto de T0. Hay que retocar el docstring y los tests. Con (a), hay que corregir las frases que dicen que solo Chronos-2 admite categóricas (sección 9).
- **Decisión:** (a), aceptada el 05/11/2026.

### D5. Relleno por la izquierda

- **Opciones:**
  - (a) Atributo obligatorio `requires_equal_length_context` en el contrato, con el relleno en `FoundationModel` y T0 migrado.
  - (b) Un helper compartido de numpy en `_utils.py` (por ejemplo, `_left_pad_batch(arrays, width)`) llamado por T0 y por Timestral.
  - (c) Relleno dentro de cada adapter, como hace hoy T0.
- **Recomendación:** (a). Es la opción centralizada que pide `.claude/rules/foundation.md`, está comprobada (sección 4) y deja los dos adapters sin relleno propio. (b) cumple la regla solo a medias: el relleno sigue dentro de los adapters. (c) la incumple.
- **Qué cambia:**
  - Con (b): no hay atributo nuevo. `predict` del adapter calcula `width = max(len(s) for s in context.values())`, rellena el contexto y cada matriz de covariables con el helper (las conocidas, a `width + steps`), y `_build_covariates` recibe la anchura. En T0 se sustituye el relleno en línea por el helper.
  - Con (a) o (b): `test_T0Adapter_predict_batches_all_series_in_one_call` (que llama directamente al adapter con series de distinta longitud) pasa a probarse a través de `FoundationModel` en (a), o sigue igual en (b).
- **Decisión:** (a), aceptada el 30/10/2026, tras una revisión final independiente (anexo A, filas 83 a 90). Con dos ajustes que ya están en este documento: `_left_pad_context` optimizado (misma salida, unas 2,5 veces más rápido) y el docstring del atributo del paso A3.

### D6. Troceado de lotes grandes

- **Opciones:** (a) Un PR aparte, después del adapter, con un atributo obligatorio `max_series_per_call` en el contrato aplicado en el bucle de grupos de `FoundationModel`. (b) Incluirlo en el PR del adapter. (c) No hacerlo por ahora.
- **Recomendación:** (a). Ahorra un 34 % con 400 series en CPU y protege la memoria en GPU, pero cambia el contrato de los diez adapters y no hace falta para que el adapter funcione.
- **Qué cambia:** con (a) o (b), un atributo más en `_REQUIRED_CLASS_ATTRIBUTES` (todos lo declaran; valor propuesto para Timestral: 100, a medir en GPU) y un bucle por trozos dentro del bucle de grupos. El adapter no cambia.
- **Decisión:** (a), aceptada el 06/11/2026. La medida en GPU del paso C3 cambió el valor de Timestral de 100 a 50 (sección 7, PR C).

### D7. MPS

- **Opciones:** (a) Si sale MPS, con `"auto"` o explícito, avisar y usar CPU desde `_resolve_torch_device(device, mps_fallback_message=..., stacklevel=...)`, como hace hoy Moirai, y migrar Moirai al helper con su mensaje actual. (b) Error claro (`ValueError`) con `device="mps"` explícito, y `"auto"` que se salte MPS sin avisar.
- **Recomendación:** (a), por el precedente de Moirai y porque centraliza el código que hoy está dentro de `MoiraiAdapter`.
- **Qué cambia:** con (b), el helper compartido no avisa; el adapter valida `device` en `__init__` y en el `validate` de `set_params` (un classmethod `_validate_device`) y los tests de MPS esperan `ValueError`. Moirai no se toca.
- **Decisión:** (a), aceptada el 04/11/2026, tras la revisión final del PR A (anexo A, filas 91 a 96). El paso A6 la implementa en el PR A.

### D8. `default_model_id`

- **Opciones:** (a) `"mistralai/Timestral-28M"`. (b) `"mistralai/Timestral-209M"`.
- **Recomendación:** (a). Con contexto largo rinde como Chronos-2 en un 40 % menos de tiempo, descarga 111 MB y sigue el criterio de Chronos (el pequeño por defecto). La documentación recomendaría el 209M para la mejor precisión.
- **Qué cambia:** con (b), cambian la clase, el valor esperado en `test_list_adapters_output`, el `model_id` de los ejemplos y `EXTRA_MODEL_IDS` (el 28M pasaría a la lista y el 209M saldría).
- **Decisión:** (a), aceptada el 05/11/2026.

### D9. `context_length` por defecto

- **Opciones:** (a) 4096. (b) 8192, como Chronos y T0.
- **Recomendación:** (a): da casi toda la ganancia por la mitad del coste con el 209M (73.0 s frente a 147.7 s) y un 60 % con el 28M (11.6 s frente a 19.5 s) (sección 2).
- **Qué cambia:** la firma, los docstrings, la línea de `FoundationModel` ("Commonly used kwargs by model"), el valor esperado en `test_get_model_info_output_for_each_adapter` y en `test_TimestralAdapter_init_default_params`, y las tablas de la documentación.
- **Decisión:** (a), aceptada el 05/11/2026.

### D10. Modelo del smoke test de `tools/foundation_compat`

- **Opciones:** (a) `"mistralai/Timestral-5M"` en `adapters.toml` (21 MB). (b) `"mistralai/Timestral-28M"`, el `default_model_id`.
- **Recomendación:** (a). El check semanal descarga menos y corre más rápido. Todas las entradas actuales usan el `default_model_id`; ningún test lo exige, pero es una convención y hay que decirlo en el PR.
- **Qué cambia:** solo la línea `model_id` de `[timestral]` en `adapters.toml`.
- **Decisión:** (a), aceptada el 05/11/2026.

### D11. Orden de los pull requests

- **Opciones:**
  - (a) Primero un PR con los cambios de código compartido (PR A), después el del adapter (PR B) y, si se acepta D6, el del troceado (PR C).
  - (b) Los cambios compartidos y el adapter en un solo PR.
- **Recomendación:** (a). El PR A toca T0, Moirai y el contrato de los nueve adapters actuales sin cambiar su comportamiento, salvo el arreglo de Moirai con `device="mps:0"` (entrada en *Fixed*); revisarlo aparte es más fácil, y si hay que revertir algo, no arrastra al adapter. Hay precedente de PR solo de contrato (#1329, rama `chore/foundation_adapters_base`) y también de cambio de contrato junto a un adapter (#1299, TimesFM 3.0); aquí se separa porque el adapter depende de un backend sin release en PyPI y sin respuesta de sus mantenedores, que podría retrasarlo.
- **Qué cambia:** con (b), los pasos A y B de la sección 7 van en la misma rama, y el PR lleva dos entradas en la release note: *Added* para `TimestralAdapter` y *Fixed* para Moirai con `"mps:0"`.
- **Coordinación:** BUG-04 y QW-05 del roadmap 0.27.0 (`dev/roadmap_0.27.0/roadmap.md`), pendientes el 30/10/2026, hacen que cada adapter declare su atributo de backend y QW-05 toca `_AdapterBase`: las mismas líneas que el paso A3, y el contrato que debe cumplir `TimestralAdapter`. Acordar el orden. Si se fusionan antes, `TimestralAdapter` declara también ese atributo y el mensaje esperado de `test_AdapterBase` los incluye. Los tests unitarios no corren en CI en los PR a `X.Y.x` y un conflicto semántico (un atributo obligatorio nuevo) no se ve en git: antes de fusionar cada PR, `git merge origin/0.27.x` en la rama y repetir `verify`.
- **Decisión:** (a), aceptada el 30/10/2026, tras una revisión final independiente (anexo A). Con los ajustes ya incorporados a la sección 7: el cambio de `tools/check_foundation_models_metadata.py` pasa al PR B, la rama del PR A es `chore/`, el PR B se actualiza con merge (nunca rebase) y hay que coordinarse con BUG-04 y QW-05 del roadmap. El 04/11/2026, tras la revisión final del PR A, el movimiento de `_match_quantile_indices` (antes paso A2) también pasa al PR B (paso B2).

## 7. Plan de implementación

Pasos en orden. Cada paso indica los archivos, los símbolos, el cambio concreto y los tests que lo cubren. Ramas desde `origin/0.27.x` (o la `X.Y.x` vigente), con nombre `<type>/<topic>`; los PR van contra la rama de release, nunca contra `main`.

### PR A: cambios de código compartido (rama propuesta `chore/foundation-equal-length-context`)

Código compartido. Para el usuario solo cambia el caso de Moirai con `device="mps:0"` (ver la comprobación al final de la tabla). El tipo de rama es `chore/`, como el precedente equivalente (#1329, `chore/foundation_adapters_base`, que también llevaba un arreglo pequeño con su release note). Antes de fusionarlo: `git merge origin/0.27.x` y repetir `verify` (coordinación con BUG-04 y QW-05, decisión D11).

| Paso | Archivo y símbolo | Cambio | Tests |
|:-----|:------------------|:-------|:------|
| A1 | `skforecast/foundation/_utils.py`: `_left_pad_context` (nuevo) | El código de la sección 4 (cambio compartido 1) | `test_utils.py`: `test_left_pad_context_output` (longitudes y fechas finales distintas; índice diario con zona horaria que cruza un cambio de hora y series en `America/Santiago` a ambos lados del 08/09/2024; `DatetimeIndex` sin `freq`; exog `None`; columnas bool, int, categóricas y de texto; todas las series con `RangeIndex(width)`; los últimos valores de cada serie rellenada son los originales) y `test_left_pad_context_single_length_unchanged_values` |
| A2 | Pasa al PR B (paso B2) | Revisión final del 04/11/2026: mover `_match_quantile_indices` solo tiene sentido cuando Timestral es su segundo usuario. Se conserva la numeración para no romper las referencias a A3 y siguientes | No aplica |
| A3 | `skforecast/foundation/_adapter_base.py`: `_AdapterBase._REQUIRED_CLASS_ATTRIBUTES` y docstrings del módulo y de la clase; `skforecast/foundation/_adapters.py`: los nueve adapters | En un solo paso, porque `_AdapterBase.__init_subclass__` comprueba cada clase al definirla y el paquete no se importa si falta el atributo en alguna: añadir `"requires_equal_length_context"` a la tupla justo después de `"supports_heterogeneous_covariates"`, y declarar `requires_equal_length_context: bool = False` (`True` en `T0Adapter`) en los nueve adapters, en la misma posición, documentándolo en Attributes como los demás. Es un atributo interno: no se añade a `FoundationModelInfo` ni a `get_model_info`. Documentación del contrato (`_AdapterBase` no tiene sección Attributes, así que va como párrafo en el docstring de la clase, y se repite en Attributes de `T0Adapter` y, en el PR B, de `TimestralAdapter`): "`requires_equal_length_context` : bool. Whether the backend needs every series of a call to have the same context length. If `True`, `FoundationModel` left-pads `context` and `context_exog` of every call with NaN to the length of the longest series of the call, so the adapter receives the context with a positional `pandas.RangeIndex` and must not rely on its timestamps. The same applies to `context_exog`, while the future `exog` keeps the index of the forecast horizon: history and future must be joined by position, never aligned by index. Boolean exog columns arrive as float, and NumPy integer columns of the padded series as float (nullable `Int64` and `Float64` columns keep their dtype and hold `pd.NA`). Requires `supports_nan_in_series=True`; adapters that derive features from the timestamps must keep it `False`." Además, en los parámetros `context` y `context_exog` del docstring de `_AdapterBase.predict` (y de `T0Adapter.predict`), añadir: "left-padded with NaN to a common length and indexed by a positional `pandas.RangeIndex` when `requires_equal_length_context` is `True`" | `test_AdapterBase.py::test_AdapterBase_TypeError_when_subclass_misses_required_class_attributes`: el mensaje esperado es una lista fija y pasa a empezar por `'requires_equal_length_context', 'supports_nan_in_series', ...`; `test_AdapterBase_every_registered_adapter_honors_the_contract`: comprobar que el atributo es `bool` y que, si es `True`, `supports_nan_in_series` también lo es. No se añade a `bool_attributes` de `test_list_adapters_every_adapter_declares_its_capabilities`, que solo recorre lo que expone `FoundationModelInfo` |
| A4 | `skforecast/foundation/_foundation_model.py`: `FoundationModel.predict` y la importación `from ._utils import (...)` | Importar `_left_pad_context`; sustituir el bucle de grupos por el bloque de la sección 4 (rellena cada grupo cuando el atributo es `True`); documentarlo en Notes de `predict` junto al párrafo de agrupado | `test_predict.py`: nuevo `test_predict_left_pads_groups_when_adapter_requires_equal_length_context` (adapter falso que registra lo que recibe; series de distinta longitud y fecha final, en `America/Santiago` a ambos lados del cambio de hora, y por el camino de `check_inputs=False`; la salida conserva el índice de cada serie) y el caso `False` (no rellena) |
| A5 | `skforecast/foundation/_adapters.py`: `T0Adapter.predict` | Sustituir `arrays`, `lengths`, el `np.full` y el bucle de copia por `context_batch = np.stack([context[series_name].to_numpy(dtype=np.float32) for series_name in series_names])` y `context_length = context_batch.shape[1]` (`arrays` y `lengths` quedarían sin uso); el comentario pasa a decir que `FoundationModel` rellena el contexto (`requires_equal_length_context=True`) con NaN, que T0 trata como ausente. En el docstring de `predict`, el de los parámetros `context` y `context_exog` (paso A3). `_build_future_covariates` no cambia: su desplazamiento `context_length - past_values.shape[0]` pasa a valer siempre 0, pero sigue siendo correcto | `test_T0Adapter.py::test_T0Adapter_predict_batches_all_series_in_one_call`: pasa a construir un `FoundationModel` con el fake (el adapter solo ya no rellena). El resto de tests de T0 usan series de la misma longitud |
| A6 | `skforecast/foundation/_adapters.py`: `_resolve_torch_device`, `MoiraiAdapter._ensure_forecast_obj` | Nuevo `_resolve_torch_device(device, mps_fallback_message=None, stacklevel=2)` (sección 4, cambio compartido 2). Moirai llama a `_resolve_torch_device(self.device, mps_fallback_message="MPS device is not supported by Moirai because the uni2ts library uses float64 operations internally. Falling back to CPU.", stacklevel=7)` y se elimina su bloque de MPS | `test_device_handling.py`: nuevos `test_resolve_torch_device_auto_falls_back_to_cpu_with_mps_fallback_message` y `test_resolve_torch_device_explicit_mps_falls_back_to_cpu_with_mps_fallback_message` (torch simulado con `patch.dict("sys.modules", {"torch": mock_torch})`, `pytest.warns(UserWarning, match="MPS device is not supported")`, también `"mps:0"`), y que sin mensaje se mantiene `"mps"`. `test_MoiraiAdapter_ensure_forecast_obj_auto_mps_falls_back_to_cpu` hoy parchea el helper para que devuelva `"mps"`: pasa a simular torch, para que el aviso salga del helper. `test_MoiraiAdapter_ensure_forecast_obj_calls_to_with_resolved_device` parchea el helper para que devuelva `"cuda"` y comprueba `assert_called_once_with("cuda")`: cambia la aserción a la llamada nueva, con `mps_fallback_message` y `stacklevel` |

Comprobación del PR A: `verify` (sección 13) con los tests de `skforecast/foundation/tests/` y además los de los llamantes de `FoundationModel.predict` y `T0Adapter.predict`, que `verify` no relaciona con estos módulos: `pytest skforecast/model_selection/tests/tests_validation/test_backtesting_foundation.py skforecast/model_selection/tests/tests_search/test_bayesian_search_foundation.py -q`. El aviso de Moirai sale con el mismo texto y en el mismo sitio, pero hay un cambio visible: Moirai con `device="mps:0"` fallaba dentro de uni2ts y ahora pasa a CPU con el aviso. Es un fallo de una versión publicada, así que según la skill `release-note` va en *Fixed* de la 0.27.0, por ejemplo: "`FoundationModel` with Moirai-2 and `device='mps:0'` failed inside uni2ts; it now falls back to CPU with the same warning as `device='mps'`."

### PR B: el adapter (rama propuesta `feature/timestral-adapter`)

La rama se crea desde `origin/0.27.x` después de fusionar el PR A (el esqueleto usa los argumentos nuevos de `_resolve_torch_device` y el relleno de `FoundationModel`). Hecho el 05/11/2026: `feature/timestral-adapter` desde `fc7efd0`, con `--no-track` para que un Sync de VS Code no publique en `0.27.x`. Para ponerla al día, `git merge origin/0.27.x` y un push normal; nunca rebase, porque una rama ya publicada necesitaría un force push, que `push_guard.py` bloquea. El PR B se abre contra `0.27.x`. Orden de los pasos: los tests de cada paso solo pasan con los anteriores hechos; registrar la clase sin la licencia, `adapters.toml` y la matriz pone en rojo los tests que recorren el registro, así que esas entradas van en el mismo paso.

Código nuevo del adapter, más la documentación y el contexto de IA.

| Paso | Archivo y símbolo | Cambio | Tests |
|:-----|:------------------|:-------|:------|
| B1 | `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`; `skforecast/foundation/_utils.py`: `_MODEL_LICENSES` | `FakeTimestralForecaster` y `_FakeTimestralForecast` (sección 8); la entrada de licencia de la sección 4 | Ninguno todavía (no hay prefijo registrado que la use) |
| B2 | `skforecast/foundation/_adapters.py`: `TimestralAdapter`, `_ADAPTER_REGISTRY`; los tests que enumeran los adapters; `tools/foundation_compat/adapters.toml`, `tools/foundation_compat/tests/contracts.py`, `.github/workflows/foundation-backends.yml`, `tools/foundation_compat/README.md`, `tools/check_foundation_models_metadata.py` (`EXTRA_MODEL_IDS` y `check_model`) | En un solo paso: mover `_match_quantile_indices` de `TimesFM3Adapter` a `_utils.py` (sección 4, cambio compartido 3: importarlo en `_adapters.py` y cambiar `self._match_quantile_indices(...)` por `_match_quantile_indices(...)` en `TimesFM3Adapter.predict`), el esqueleto de la sección 4 tras `NoriAdapter`, la entrada `"mistralai/Timestral": TimestralAdapter` al final del registro, `FeatureOutOfRangeWarning` importado, las entradas de la sección 8 en los tests y en `tools/`, y el cambio de `license_name` de la sección 4 (cambio compartido 4): solo la asignación de dentro de `if hub_license == "other":` pasa a `hub_license = card_data.get("license_name") or "other"` (la primera, `card_data.get("license")`, no cambia: si cambiara, rompería todas las licencias SPDX), y la viñeta del docstring del módulo sobre `license_name` menciona que sin él se compara con `"other"` | `test_TimestralAdapter.py` (sección 8), los tests que enumeran los adapters, `test_backends_compat_config.py`, `test_list_adapters.py`, `test_utils.py` (también `test_match_quantile_indices_output` y `test_match_quantile_indices_ValueError_when_quantile_missing`; los de TimesFM3 siguen pasando); el workflow `foundation-models-metadata.yml`, que corre en los PR a `*.x` que tocan `_adapters.py`, `_utils.py` o el script, comprueba la licencia contra la tarjeta real de Timestral |
| B3 | `skforecast/foundation/_foundation_model.py`: docstrings de `FoundationModel` | Las entradas de la sección 9 | No aplica |
| B4 | Documentación y AI context a mano | La lista de la sección 9 | Checks de docs del `verify` |
| B5 | `docs/user_guides/foundation-forecasting-models.ipynb` y `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` | Cambios de la sección 9. El primero ejecuta todos los modelos de la guía y necesita todos sus backends (uni2ts con `scipy>=1.12` como override, `TABPFN_TOKEN` para TabPFN-TS), así que hay dos opciones: (a) un entorno con todos los backends y `python tools/docs/execute_notebooks/execute_notebooks.py user_guides/foundation-forecasting-models.ipynb` (está en `EXCLUDE_NOTEBOOKS`, hay que pasarlo explícitamente); (b) ejecutar en un kernel las celdas de preparación (importaciones, datos, `end_train`), las celdas nuevas de Timestral y las que cambian (45 y 102), conservando el resto de salidas. La celda 102 usa `metrics_chronos` ... `metrics_tsicl`, que solo existen si se han ejecutado los backtestings de todos los modelos: en (b) hay que reconstruirlas en ese kernel con los valores de las salidas guardadas de cada celda de backtesting, y decirlo en el PR. Recomendación: (a) si hay un entorno con todos los backends; si no, (b). El segundo notebook solo cambia en una celda de código (11, la lista comentada de `model_id`) y en celdas de texto: volver a ejecutarlo con Chronos-2, o limitar el cambio a las celdas de texto. En ambos, `execute_notebooks.py --check` en verde | No aplica |
| B6 | Archivos generados | `python tools/ai/generate_ai_context_files.py` y después `--check` (skill `ai-context-sync`) | `ai-context-check` |
| B7 | `docs/releases/releases.md` | Entrada en la 0.27.0 (skill `release-note`, sección 9) | No aplica |

### PR C (opcional, decisión D6): troceado de lotes (rama propuesta `perf/foundation-max-series-per-call`)

| Paso | Archivo y símbolo | Cambio | Tests |
|:-----|:------------------|:-------|:------|
| C1 | `_adapter_base.py` y los diez adapters, en un mismo paso (como A3) | Atributo obligatorio `max_series_per_call: int \| None` (`None` en todos salvo Timestral) | `test_list_adapters.py`, `test_AdapterBase.py` (mensaje esperado) |
| C2 | `_foundation_model.py`: `FoundationModel.predict` | Dentro del bucle de grupos, partir cada grupo en trozos de como mucho `max_series_per_call` series; el relleno del PR A se aplica por trozo | `test_predict.py`: número de llamadas y tamaños de los trozos con un adapter falso |
| C3 | Medida | Medir en GPU el valor para Timestral (en CPU, 100 series ahorran un 34 % con 400) | No aplica |

**Hecho el 06/11/2026** en la rama `perf/foundation-max-series-per-call`, creada desde `origin/0.27.x` (`c75a63789`) con `--no-track`. Sin commitear.

- **C1:** `"max_series_per_call"` va en `_REQUIRED_CLASS_ATTRIBUTES` justo después de `"requires_equal_length_context"`, documentado como párrafo en el docstring de `_AdapterBase` y en el parámetro `context` de `_AdapterBase.predict`. Los diez adapters lo declaran después de `requires_equal_length_context` y lo documentan en Attributes. Es interno, como `requires_equal_length_context`: no se añade a `FoundationModelInfo` ni a `get_model_info`, así que `test_list_adapters.py` no cambia. `test_AdapterBase.py`: mensaje esperado y, en el test de contrato, que el valor sea `None` o un entero positivo (no `bool`). `test_TimestralAdapter_init_default_params` comprueba el valor 50.
- **C2:** el bucle de grupos de `FoundationModel.predict` pasa a recorrer una lista de llamadas: cada grupo se parte en trozos consecutivos de como mucho `max_series_per_call` series (`None`: el grupo entero), y el relleno se aplica por llamada. Notes de `predict` actualizadas. Tests nuevos en `test_predict.py`: `test_predict_splits_groups_into_chunks_of_max_series_per_call` (cinco series de longitudes distintas en dos grupos por exog, con el fake de Timestral y el atributo a 2: llamadas `[s0, s1]`, `[s2]` y `[s3, s4]`, cada una rellenada a su propia anchura, mismas predicciones que sin trocear; con `check_inputs` `True` y `False`) y `test_predict_TimestralAdapter_calls_backend_in_chunks_of_50_series` (230 series: llamadas de 50, 50, 50, 50 y 30).
- **C3, valor 50 en lugar de 100.** Medido en el entorno `skforecast_27_gpu` (Windows, Python 3.13.18, torch 2.14.1+cu130, timestral 1.0.0, numpy 2.5.4, pandas 2.3.3; NVIDIA RTX 2000 Ada Laptop, 8 GB) **[probado]**, `exp7_gpu_chunk_size.py` (backend directo, 24 pasos, mínimo de dos repeticiones tras un calentamiento; logs `exp7.log`, `exp7_long.log`, `exp7_gpu_extra.log`, `exp7_cpu.log` y `exp7_cpu_extra.log`). Una llamada que no cabe en los 8 GB no da OOM en Windows: el driver desborda a la memoria compartida del sistema y la llamada se vuelve de 10 a 16 veces más lenta.

  | Checkpoint | Contexto | Covariables | Series | Una llamada | Trozos de 50 | Trozos de 100 |
  |:-----------|---------:|------------:|-------:|------------:|-------------:|--------------:|
  | 28M | 512 | 2 | 1000 | 54.55 s, 18.3 GB | 5.72 s, 0.23 GB | 5.60 s, 0.41 GB |
  | 28M | 4096 | 2 | 400 | | 14.82 s, 1.01 GB | 17.46 s, 2.42 GB |
  | 209M | 512 | 2 | 400 | 17.21 s, 6.99 GB | 11.86 s, 1.02 GB | 12.40 s, 1.38 GB |
  | 209M | 512 | 2 | 1000 | 305.59 s, 35.6 GB | 29.59 s, 1.02 GB | 31.02 s, 1.38 GB |
  | 209M | 2048 | 2 | 400 | 747.36 s, 24.5 GB | 47.22 s, 1.67 GB | 47.48 s, 3.02 GB |
  | 209M | 4096 | 2 | 400 | | 93.67 s, 2.53 GB | 94.93 s, 5.22 GB |
  | 209M | 4096 | 4 | 100 | 160.76 s, 10.9 GB (= trozos de 100) | 39.40 s, 4.14 GB | |
  | 209M | 4096 | 0 | 400 | 15.64 s, 4.37 GB | 14.87 s, 1.09 GB | 14.92 s, 1.39 GB |
  | 28M, CPU | 512 | 2 | 400 | | 17.73 s | 17.21 s |
  | 28M, CPU | 2048 | 2 | 100 | 14.71 s (= trozos de 100) | 14.55 s | |

  Con covariables, 50 no es más lento que 100 (salvo un 2 % en algún caso) y usa la mitad de memoria; con el `context_length` por defecto (4096) y cuatro covariables, 100 series ya desbordan una GPU de 8 GB y 50 no. Sin covariables y contexto corto, 50 es algo más lento que 100 (28M, 512, 1000 series: 1.21 s frente a 1.02 s; CPU, 400 series: 2.75 s frente a 2.50 s), en tiempos absolutos pequeños. Trozos de 25 apenas cambian el tiempo frente a 50 con covariables (un 4 % como mucho) y sin ellas son claramente más lentos (28M, 1000 series: 2.38 s frente a 1.21 s). La memoria crece con el número de filas por serie (objetivo más covariables) al cuadrado, así que ningún número fijo de series la acota del todo; 50 deja margen para unas cuatro covariables con contexto 4096 en 8 GB.
- **Equivalencia y coste a través del paquete** (`exp8_chunk_equivalence.py`, `exp8_cpu.log` y `exp8_cuda.log`; `FoundationModel` con `Timestral-28M`, 230 series diarias de 300 a 700 puntos con una covariable conocida, 14 pasos, cuantiles 0.1, 0.5 y 0.9) **[probado]**: las predicciones con trozos de 50 o 100 difieren de una sola llamada en 1.5e-05 como máximo (1.6e-07 relativo a la mayor predicción; mediana 0), redondeo de float32. Tiempo: CPU 6.61 s una llamada, 7.15 s con 50 (+8 %), 6.83 s con 100; GPU 1.29 s, 1.02 s y 1.07 s. Con pocas series el troceado cuesta poco y, con muchas, evita el desbordamiento.
- **Documentación y contexto de IA:** la viñeta de Timestral en `foundation-forecasting-with-heterogeneous-series.ipynb` (solo una celda de texto) y la de los atributos de clase en `skills/foundation-forecasting/references/adapter-parameters.md`; regenerados los archivos de contexto de IA. Release note: la entrada de `TimestralAdapter` en *Added* de la 0.27.0 (es del mismo ciclo, no va entrada aparte); falta añadir el enlace del PR C cuando se abra.
- **Coordinación con TimeH2O** (`dev/timeh2o_adapter/`): el prototipo de ese adapter no declara `max_series_per_call`; si se fusiona este PR antes, tiene que declararlo (`None`, salvo que se mida otra cosa).

## 8. Plan de tests

Basado en cómo están escritos los tests de los adapters actuales **[verificado en código]**. Ningún test descarga pesos: todos usan objetos falsos de `fixtures_adapters.py`, inyectados con `model=` o asignando `adapter._model` (`ForecasterFoundation.__init__` clona el estimador y pierde el `model=` inyectado, así que en los tests a través del forecaster hay que asignar `forecaster.estimator.adapter._model = fake` después de crearlo, como hace `make_forecaster` en `fixtures_forecaster_foundation.py`).

### Objeto falso: `FakeTimestralForecaster` (en `fixtures_adapters.py`)

Sigue a `FakeT0Forecaster`/`FakeT0Forecast` (objeto de salida cuyo último eje son los niveles) y a `FakeTimesFM3Forecaster` (rejilla configurable, registro de la última llamada y comprobación de formas):

- `_FakeTimestralForecast(quantiles, quantile_levels)`: `quantiles` de forma `(B, V, H, Q)` con `quantiles[..., i] == quantile_levels[i]` (`np.broadcast_to(levels, shape).copy()`), y `quantile_levels` como `tuple`. No hacen falta `median` ni `quantile()`, que el adapter no lee.
- `FakeTimestralForecaster(quantile_levels=None, as_tensor=False)`:
  - `quantile_levels` es una tupla, por defecto los 21 niveles. Con `as_tensor=True`, la salida se envuelve en un objeto con `detach()`, `cpu()` y `numpy()`, como un tensor.
  - Registra `last_context`, `last_prediction_length`, `last_past_covariates`, `last_known_covariates`, `last_categorical_past_covariates`, `last_categorical_known_covariates`, `last_kwargs` y una lista `calls`.
  - `@classmethod from_pretrained(cls, path_or_repo_id, device="cpu", **kwargs)` devuelve `cls()`. Los tests de device lo subclasifican para registrar `device`, como `_RecordingForecaster` en `test_TimesFM3Adapter.py`.
  - `forecast(self, context, prediction_length, observation_mask=None, past_covariates=None, known_covariates=None, categorical_past_covariates=None, categorical_known_covariates=None, symmetric_averaging=True, clip_nonnegative=True)`: la misma firma que el backend, sin `**kwargs`, para que una clave desconocida de `predict_kwargs` dé `TypeError` como con el backend real. Guarda `symmetric_averaging` y `clip_nonnegative` en `last_kwargs`.
  - Como el backend real, exige `context.ndim == 3`, `(B, C, T)` en las pasadas y `(B, C, T + prediction_length)` en las conocidas, y lanza `AssertionError` si no se cumplen.

Un borrador de este fake, probado con el esqueleto de la sección 4 (chequeos `fake: ...` del bloque 7 de `run_checks.py`), está en `dev/timestral_adapter/experiments/skeleton/fake_timestral.py`.

### `test_TimestralAdapter.py` (nuevo)

Imports como `test_TimesFM3Adapter.py` (`re`, `sys`, `types`, `warnings`, `pytest`, `numpy`, `pandas`, `LicenseWarning`, `FeatureOutOfRangeWarning` y los fixtures). Helper `make_adapter(**kwargs)` con `model=FakeTimestralForecaster()`.

| # | Test | Qué comprueba |
|--:|:-----|:--------------|
| 1 | `test_TimestralAdapter_init_default_params` | `context_length == 4096`, `device == "auto"`, `predict_kwargs == {}`, `_model`, `context_` y `context_exog_` a `None`, `is_fitted` `False`; atributos de clase (exog, pasadas, categóricas `True`; heterogéneas `False`; `requires_equal_length_context` y NaN `True`); `SUPPORTED_QUANTILES` igual a los 21 niveles |
| 2 | `test_TimestralAdapter_init_ValueError_when_invalid_context_length` | `[0, -1, None, "a"]` |
| 3 | `test_TimestralAdapter_init_ValueError_when_model_id_not_timestral` | `"mistralai/Mistral-7B-v0.1"`, `"google/timesfm-3.0-pytorch"`, `"autogluon/chronos-2-small"`, con el mensaje exacto de `_validate_model_id_prefix` |
| 4 | `test_TimestralAdapter_init_TypeError_when_loader_only_kwargs` | `max_context`, `per_shot_horizon_patches`, `quantile_real_cap_k`, `revision`: "unexpected keyword argument" |
| 5 | `test_TimestralAdapter_init_ValueError_when_predict_kwargs_has_reserved_key` | Parametrizado con las 7 claves reservadas |
| 6 | `test_TimestralAdapter_init_predict_kwargs_stored_by_reference` | `{"symmetric_averaging": False}` es el mismo objeto |
| 7 | `test_TimestralAdapter_get_params_returns_expected_keys_and_values` | Claves `{model_id, context_length, device, predict_kwargs}`; `predict_kwargs` `None` cuando está vacío |
| 8 | `test_TimestralAdapter_set_params_ValueError_when_invalid` | `context_length=-1`, parámetros desconocidos, `max_context` |
| 9 | `test_TimestralAdapter_set_params_ValueError_when_model_id_not_timestral` | El adapter queda intacto |
| 10 | `test_TimestralAdapter_set_params_ValueError_when_predict_kwargs_has_reserved_key` | `predict_kwargs={"known_covariates": ...}` |
| 11 | `test_TimestralAdapter_set_params_updates_and_resets_model` | `model_id` y `device`: mismo valor no reinicia; valor nuevo pone `_model` a `None` y devuelve `self` |
| 12 | `test_TimestralAdapter_set_params_no_reset_for_non_reload_keys` | `context_length` y `predict_kwargs` conservan `_model` |
| 13, 14 | `test_TimestralAdapter_fit_output_single_series`, `..._multi_series` | Copiados de TimesFM3 |
| 15 | `test_TimestralAdapter_predict_ValueError_for_unsupported_quantile` | `[0.33, 0.02, 0.12, 0.98, 1.1, -0.1]` (no reutilizar la lista de TimesFM 3.0: 0.05, 0.15, 0.25 y 0.95 son válidos aquí) |
| 16 | `test_TimestralAdapter_predict_point_forecast_single_series` | Forma `(12, 1)` con valor 0.5 (la columna de la mediana del fake); `context.shape == (1, 1, n)` y las cuatro covariables `None` |
| 17 | `test_TimestralAdapter_predict_quantile_forecast_single_series` | `[0.9, 0.01, 0.5]` devuelve las columnas en ese orden |
| 18 | `test_TimestralAdapter_predict_all_supported_quantiles` | Forma `(steps, 21)` |
| 19 | `test_TimestralAdapter_predict_point_and_quantile_multi_series` | Un array por serie, en el orden de `context` |
| 20 | `test_TimestralAdapter_predict_no_horizon_ceiling` | `steps=2000` da `(2000, 1)` |
| 21 | `test_TimestralAdapter_predict_forwards_predict_kwargs` | `last_kwargs == {"symmetric_averaging": False, "clip_nonnegative": False}` |
| 22 | `test_TimestralAdapter_predict_covariates_builds_past_and_known` | `past_covariates.shape == (1, 1, ctx)`, `known_covariates.shape == (1, 1, ctx + steps)`, valores = histórico seguido del futuro |
| 23 | `test_TimestralAdapter_predict_categorical_covariates_factorized` | Una columna de texto y una `category` van a `categorical_known_covariates` (o `categorical_past_covariates` si solo están en el histórico), con códigos comunes a histórico y futuro y NaN para los ausentes; bool, `Int64`, `Float64` y `boolean` van a las numéricas, con NaN en los NA |
| 24 | `test_TimestralAdapter_build_covariates_per_series` | Prueba directa: claves presentes solo con columnas; dict vacío sin covariables |
| 25 | `test_TimestralAdapter_predict_inf_in_numeric_covariate_becomes_nan` | `np.inf` en una covariable numérica llega como NaN |
| 26 | `test_TimestralAdapter_predict_stacks_equal_length_series_in_one_call` | Dos series de la misma longitud (ya rellenadas por `FoundationModel`): una llamada, `context.shape == (2, 1, n)`, columnas en el orden de la firma |
| 27 | `test_FoundationModel_Timestral_predict_left_pads_and_groups_heterogeneous_exog` | A través de `FoundationModel(..., model=fake)` con series de distinta longitud y columnas distintas (como `_make_heterogeneous_inputs` de `test_predict.py`): una llamada por firma, NaN delante en las series cortas (contexto y covariables), y los arrays de una serie predicha sola con `levels` iguales a los del lote sin el relleno. Las dos predicciones con el mismo `steps`: con covariables futuras, Timestral cambia los primeros pasos si cambia el horizonte (sección 2), así que con distinto `steps` no son comparables (un revisor lo encontró en un fold de backtesting en el que la serie sola se predecía con 4 pasos y en el lote con 12) |
| 28 | `test_TimestralAdapter_predict_quantile_index_mapping_uses_model_quantile_grid` | Con la rejilla del fake invertida, 0.01 y 0.99 se eligen por valor |
| 29 | `test_TimestralAdapter_predict_ValueError_when_quantile_missing_from_model_grid` | Rejilla sin 0.5: "Quantile 0.5 not found in the model's quantile grid" |
| 30 | `test_TimestralAdapter_predict_FeatureOutOfRangeWarning_when_known_covariate_constant_in_context` | Precio constante en el contexto que cambia en el horizonte: un aviso que nombra `(serie, columna)`; sin aviso si el futuro sigue constante, si el histórico ya varía o si la columna es categórica. A través de `ForecasterFoundation.predict`, `record[0].filename` es el del test (el aviso apunta a la llamada del usuario, `stacklevel=5`) |
| 31 | `test_TimestralAdapter_load_model_LicenseWarning_and_forwards_device` | Módulo `timestral` falso con `types.ModuleType` y `monkeypatch.setitem(sys.modules, ...)`; `pytest.warns(LicenseWarning, match="Timestral Modified MIT License")` con el URL del 209M; `from_pretrained` recibe `device="cpu"`; una segunda llamada no avisa |
| 32 | `test_TimestralAdapter_load_model_ImportError_when_timestral_not_installed_no_LicenseWarning` | `monkeypatch.setitem(sys.modules, "timestral", None)`: `ImportError` con `re.escape("timestral is required for TimestralAdapter")` y `backend_package` en el mensaje, sin `LicenseWarning` |
| 33 | `test_TimestralAdapter_load_model_mps_falls_back_to_cpu_with_warning` | Torch simulado (cuda `False`, mps `True`): con `"auto"` y con `"mps"`, `from_pretrained` recibe `"cpu"` y sale el `UserWarning` "MPS device is not supported by Timestral" |
| 34 | `test_TimestralAdapter_load_model_noop_when_already_set` | `_model is fake` tras `_load_model()` |
| 35 | `test_TimestralAdapter_predict_series_without_exog_key_has_no_covariates` | `context_exog={}` y `exog={}` (camino del backtesting): las cuatro covariables `None`, sin `KeyError` |
| 36 | `test_TimestralAdapter_predict_handles_tensor_output` | `FakeTimestralForecaster(as_tensor=True)` devuelve `np.ndarray` |
| 37 | `test_ForecasterFoundation_Timestral_predict_interval_supported_levels` | Parametrizado con `interval=[0.05, 0.95]`, `[0.01, 0.99]`, el valor por defecto `[0.1, 0.9]` y las formas `interval=0.9` y `interval=0.98`, que dan 0.04999999999999999 y 0.010000000000000009 y comprueban las tolerancias de `_validate_supported_quantiles` (1e-9) y `_match_quantile_indices` (1e-6). Desde la 0.23.0, `interval` va en cuantiles, no en percentiles |

Con D4 = (b), los tests 23 y 24 se sustituyen por el `ValueError` de no numéricas (`test_TimesFM3Adapter_predict_non_numeric_covariate_raises_ValueError`). Con D3 = (a), desaparece el 30. Con D7 = (b), el 33 pasa a esperar `ValueError` con `"mps"` y ningún aviso con `"auto"`.

### Tests existentes que enumeran los adapters

| Archivo | Símbolo y cambio |
|:--------|:-----------------|
| `test_resolve_adapter.py` | Importar `TimestralAdapter`; añadir los cuatro `model_id` a `test_resolve_adapter_returns_correct_class`; añadir `"mistralai/Timestral": TimestralAdapter` al `expected` de `test_ADAPTER_REGISTRY_contains_all_expected_entries`. Opcional: `"mistralai/Mistral-7B-v0.1"` en `test_resolve_adapter_ValueError_when_unknown_prefix` |
| `test_list_adapters.py` | `("TimestralAdapter", "mistralai/Timestral-28M")` al final del `expected` de `test_list_adapters_output`. Los otros dos tests recorren el registro |
| `test_get_model_info.py` | Una fila en `test_get_model_info_output_for_each_adapter`, `test_get_model_info_supported_quantiles` (tupla de 21 niveles), `test_get_model_info_license_fields_match_LicenseWarning_registry` (grupo `True`), `test_get_model_info_license_url` (URL fijo del 209M también para el 5M) y `test_get_model_info_weights_location_for_each_adapter` (`True, False`); `"timestral"` en `backends` de `test_get_model_info_does_not_import_backend_libraries`. Recomendado: `test_get_model_info_output_Timestral` como `test_get_model_info_output_TimesFM3`, con `license="other"` |
| `test_check_exog_columns.py` | Importar; `"mistralai/Timestral-28M"` en `_EXOG_ADAPTERS` y `_PAST_ONLY_ADAPTERS` (no en `_FUTURE_ONLY_ADAPTERS`); `(TimestralAdapter, True)` en `test_adapters_declare_supports_past_only_covariates` |
| `test_clone.py` | `"mistralai/Timestral-28M"` en `test_clone_round_trip_every_adapter_default_params` y `("mistralai/Timestral-28M", {"predict_kwargs": {"symmetric_averaging": False}})` en `test_clone_round_trip_with_non_empty_config_dict`, añadiendo en los dos el id `"timestral"` a su lista `ids=[...]` (si no, pytest da un error de colección) |
| `test_utils.py` | `"mistralai/Timestral-28M"` (id `"timestral"`) en `test_warn_if_non_commercial_warns_for_registered_prefixes`. `test_get_model_license_every_registered_prefix_has_a_license` recorre el registro |
| `test_set_params.py` | Opcional: `("mistralai/Timestral-28M", "is served by TimestralAdapter")` en `test_set_params_ValueError_when_model_id_served_by_other_adapter`, con el id `"timestral_adapter"` en su lista `ids=[...]` |
| `test_predict.py` | Opcional: `test_predict_timestral_one_forecast_call_per_exog_signature`, como `test_predict_timesfm_v3_one_predict_batch_call_per_exog_signature` (el test 27 ya lo cubre) |
| `tests_forecaster_foundation/test_init.py` | `("mistralai/Timestral-28M", True)` en `test_init_tag_supports_categorical_features_follows_adapter` (si D4 = a) |
| `test_AdapterBase.py`, `test_backends_compat_config.py` | Sin cambios por Timestral: recorren el registro, `adapters.toml` y la matriz, y fallan si falta alguna entrada. Recomendado en `test_backends_compat_config.py`: `test_adapters_toml_backend_package_is_a_PEP_508_requirement`, que habría detectado el problema de `"git"`. Sin importar `tools/foundation_compat/run.py` (no es importable desde los tests del paquete y usa `tomllib`, que no existe en Python 3.10): comprobar que `packaging.requirements.Requirement(backend_package).name` coincide con el principio de la cadena según la misma expresión que `run.distribution_name` (`[A-Za-z0-9._-]+`) |
| `test_device_handling.py`, `test_fit.py`, `test_get_params.py`, `test_init.py`, `test_repr.py`, `test_prepare_future_exog.py`, tests de `model_selection` | Sin cambios (no enumeran adapters o solo usan Chronos) |

### `tools/foundation_compat` y CI

`tools/foundation_compat/adapters.toml`:

```toml
[timestral]
adapter = "TimestralAdapter"
backend_package = "timestral @ git+https://github.com/mistralai/timestral.git"
model_id = "mistralai/Timestral-5M"
```

`backend_package` tiene que ser idéntico al atributo de la clase (`test_adapters_toml_entries_match_the_adapter_classes`).

`tools/foundation_compat/tests/contracts.py`, en `BACKEND_CONTRACTS` (comprobado contra timestral 1.0.0 instalado **[probado]**):

```python
    "timestral": [
        {
            "target": "timestral:TimestralForecaster.from_pretrained",
            "n_positional": 1,
            "kwargs": ["device"],
        },
        {
            "target": "timestral:TimestralForecaster.forecast",
            "kwargs": [
                "context",
                "prediction_length",
                "past_covariates",
                "known_covariates",
                "categorical_past_covariates",
                "categorical_known_covariates",
            ],
        },
        {"target": "timestral:Forecast", "kwargs": ["quantiles", "quantile_levels"]},
    ],
```

`.github/workflows/foundation-backends.yml`: añadir `timestral` al final de la lista de una sola línea `adapter: [chronos, timesfm25, timesfm3, moirai, tabicl, tabpfn, t0, nori, tsicl]`. `test_workflow_matrix_matches_adapters_toml` la lee con una expresión regular de una línea, y su contenido ordenado debe coincidir con las claves de `adapters.toml`.

`tools/foundation_compat/README.md`: añadir `timestral` a la lista de paquetes del primer párrafo.

`tools/check_foundation_models_metadata.py`: `"mistralai/Timestral-5M"`, `"mistralai/Timestral-123M"` y `"mistralai/Timestral-209M"` en `EXTRA_MODEL_IDS` (el 28M ya lo cubre `list_adapters()`), además del cambio de `license_name` (paso B2).

Aviso del harness: `.claude/skills/verify/affected.py` ejecuta `python -m pytest tools/foundation_compat/tests -q` cuando se toca `tools/foundation_compat/`, y ese comando falla porque `conftest.py` declara `--backend` como obligatorio. Es un fallo previo, ajeno a Timestral: hay que decirlo en el PR y no confundirlo con un fallo del cambio **[probado]** (05/11/2026: `error: the following arguments are required: --backend`). Esa carpeta solo tiene los tests que se ejecutan contra un backend instalado (`run.py`); la configuración (`adapters.toml`, la matriz y las clases) la comprueba `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py`, que sí entra en los tests del PR.

## 9. Puntos que hay que tocar

Lista obtenida con `git grep -il` sobre el árbol (sin `dev/`) buscando `TSICLAdapter`, `tsicl`, `TS-ICL`, `taharnbl`, `NoriAdapter`, `Nori` (palabra completa) y `synthefy`, más una segunda búsqueda de los archivos que enumeran TabPFN, T0 o Moirai sin nombrar a TS-ICL ni a Nori. Se ha contrastado con los commits que añadieron TS-ICL (PR #1265, #1267, #1268) y Nori (PR #1264, #1266, #1268) y con los posteriores que añadieron listas por adapter (`8f1b075a`, `aa7a098c`, `0101ce99`, `e0fc9c1a`, `b7a58ecd`, `1d56893f`, `c4178901`) **[verificado en código]**. Los tests y `tools/` están en la sección 8.

### Paquete (a mano)

| Archivo | Dónde | Qué añadir |
|:--------|:------|:-----------|
| `skforecast/foundation/_adapters.py` | Clase nueva y `_ADAPTER_REGISTRY` | Paso B2 |
| `skforecast/foundation/_utils.py` | `_MODEL_LICENSES` | Paso B1. Opcional: el docstring de `_warn_if_non_commercial` dice "non-commercial license" (D2) |
| `skforecast/foundation/_foundation_model.py` | Docstring de `FoundationModel`: frase inicial ("Currently supports ..."), lista de `model_id` por proveedor, "Commonly used kwargs by model" (`- **Mistral AI Timestral** (`TimestralAdapter`): `context_length` (int, default 4096), `device` (str, default `'auto'`), `predict_kwargs` (dict, default None).`), viñeta de Notes sobre el device ("`device` string ...: Moirai, TS-ICL, TimesFM 3.0") y References (GitHub y colección de HF) | Timestral en todas |
| `skforecast/foundation/_foundation_model.py` | Docstrings de las propiedades `supports_past_only_covariates` (lista de los que lo admiten) y `supports_heterogeneous_covariates` (lista de los que agrupan) | Timestral en ambas |
| `skforecast/model_selection/_search.py` | Docstring de `bayesian_search_foundation`, "Tunable parameters per model" y la nota de recarga | Opcional: Timestral solo ajusta `context_length` (ya cubierto por "All models"), que no recarga el modelo |

`_adapter_base.py` (salvo el PR A), `_model_info.py`, `__init__.py` y `pyproject.toml` no cambian: `_model_info.py` copia una lista fija de atributos de la clase (`_build_model_info`), que Timestral declara como los demás; `requires_equal_length_context` no se expone. Las clases de los adapters no se exportan y ningún backend de foundation es un extra de skforecast.

### Documentación (a mano)

| Archivo | Dónde | Qué añadir |
|:--------|:------|:-----------|
| `docs/api/FoundationModel.md` | Bloques mkdocstrings, tras `TSICLAdapter` | `::: skforecast.foundation._adapters.TimestralAdapter` con `options: show_bases: false`, como los demás |
| `docs/user_guides/foundation-forecasting-models.ipynb` | Celda 9 (diagrama HTML con una caja por modelo); celda 11 ("### Supported Foundation Models", tabla `forecasting-table-premium`: modelo, proveedor, `model_id`, covariables, contexto por defecto y máximo, horizonte máximo, previsión puntual, cross learning, backend, comando de instalación con botón de copia, enlaces); celda 17 ("### Input Data Formats", modelos con exog); celda 45 (`model_ids_allow_exog`); sección nueva bajo "## Other foundation models", como "### Synthefy Nori" (celdas 90 a 95) y "### EDF-Lab TS-ICL" (celdas 96 a 100), con `metrics_timestral`; celda 102 ("### Model comparison") | Todo, y volver a ejecutar solo ese notebook. Incluir la advertencia del issue #7 y la nota de que, con covariables futuras, la previsión depende de `steps` |
| `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` | Celda 2 (tabla "In short", fila sobre si el agrupado cambia la previsión); celda 11 (lista comentada de `model_id`); celda 13 (modelos con covariables); celda 18 ("## How it works", viñeta "Same covariate columns"); celda 21 ("## Reference: requirements and limitations per backend", fila y viñeta). La frase "Only Chronos-2 accepts string and categorical columns natively" deja de ser cierta | Todo |
| `docs/quick-start/ai-assisted-forecasting.md` | Viñeta "Foundation models (zero-shot)" y fila `foundation-forecasting` de la tabla de skills | Timestral |
| `docs/overrides/home.html` | Fila `.models`: "Chronos-2 / TimesFM / Moirai-2 / TabPFN-TS / +5" | `+5` pasa a `+6` (no sale en el grep: es un número) |
| `README.md`, `docs/introduction-forecasting/introduction-forecasting.md`, `docs/user_guides/global-forecasting-overview.md`, `docs/animations/foundation-models.html`, `tools/docs/hooks/examples.yml`, `tools/docs/hooks/page_meta.yml` | Listas "such as ..." | Opcional: no nombran a TS-ICL ni a Nori |
| `mkdocs.yml` | | Nada: la navegación apunta a páginas, no a modelos |

### Release notes (a mano, skill `release-note`)

| Archivo | Dónde | Qué añadir |
|:--------|:------|:-----------|
| `docs/releases/releases.md` | `## 0.27.0 <small>In development</small>`: un destacado en "The main changes in this release are:" y una entrada en `**Added**` | Con el formato de TS-ICL y Nori (sección `## 0.24.0`): `+ <span class="badge text-bg-feature">Feature</span> New <code>[TimestralAdapter]</code> in the <code>foundation</code> module wrapping Mistral AI's Timestral models, registered under the 'mistralai/Timestral' model_id prefix. Supports ... [User guide](...) ([#NNNN](...))`, y el mismo texto sin la insignia en `**Added**`. Si `**Added**` pasa de unas 10 entradas, agrupar bajo `*Foundation models*` |
| `docs/releases/releases.md` | Definiciones de enlaces al final, tras `[TSICLAdapter]: ...` | `[TimestralAdapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.TimestralAdapter` |

Las entradas de versiones ya publicadas (por ejemplo, las de la 0.25.0 que enumeran modelos) no se tocan.

### Contexto de IA, fuentes (a mano)

| Archivo | Dónde | Qué añadir |
|:--------|:------|:-----------|
| `tools/ai/llms-base.txt` | Comentario del árbol de `## Project Structure`; lista de clases de `### Module Relationships`; fila de ForecasterFoundation en `## Core Forecasters`; en `## Foundation Models (Zero-Shot)`: lista de paquetes, fila de la tabla "Supported adapters" (adapter, prefijo, exog, `context_length`, cuantiles), viñetas de `LicenseWarning`, de covariables solo pasadas y de agrupado | Todo |
| `llms.txt` (raíz; fuente de `docs/llms.txt`) | `## Core Forecasters`, `## Foundation Models`, `## API Reference` | Timestral en las tres listas |
| `skills/foundation-forecasting/SKILL.md` | `description` del frontmatter; `## Stop Conditions` (filas de exog y de licencia); bloque pip de `## Installation`; título y texto de `## With Exogenous Variables (...)`; `## Prediction Intervals and Quantiles` (21 niveles); fila de la tabla y párrafo de licencias de `## Choosing a Model`; `## Common Mistakes`, puntos 3, 4, 6, 8 (dice que solo Chronos-2 admite categóricas) y 10 ("Only Chronos-2, TS-ICL and TimesFM 3.0 use them as past-only covariates"); lista de `## References` | Todo, más el issue #7 en Common Mistakes |
| `skills/foundation-forecasting/references/adapter-parameters.md` | `## Contents`; sección nueva `## TimestralAdapter: Mistral AI Timestral` (prefijo, tabla de parámetros); fila de `## Tunable Parameters and Model Reload Cost`; en `## Common Behavior`, la lista de agrupado, la frase de `supports_categorical_covariates` y una mención de `requires_equal_length_context` (T0 y Timestral: `FoundationModel` rellena cada grupo con NaN por la izquierda), que se puede añadir ya en el PR A; listas y paquetes de `### Exog column validation at predict time` | Todo |
| `skills/choosing-a-forecaster/SKILL.md` | Filas de ForecasterFoundation en `## Step 2a` y `## Step 2b`; `## Decision Flowchart` | Tres sitios |
| `skills/prediction-intervals/SKILL.md` | Frase de restricción de cuantiles en `## Foundation Model Intervals` | La rejilla de 21 niveles |
| `skills/complete-api-reference/SKILL.md` | Viñeta de ForecasterFoundation en `### Forecaster Constructors` | Timestral en la lista |
| `skills/complete-api-reference/references/forecaster-constructors.md` | Bloque de `### ForecasterFoundation`, comentarios de kwargs por modelo | `device, predict_kwargs (Timestral)`; la nota "predict_kwargs (TimesFM 3.0 only)" ya era inexacta (Chronos también lo tiene) |
| `skills/hyperparameter-optimization/references/search-parameters.md` | Tabla "Worth searching" de `### Foundation Function` | Opcional: Timestral en la fila de solo `context_length` (TS-ICL tampoco está) |
| `tools/ai/README.md` | Fila `foundation-forecasting` de la tabla `## Skills` | Timestral en la lista |

### Contexto de IA, generados (no se editan)

Se regeneran con `python tools/ai/generate_ai_context_files.py` y se comprueban con `python tools/ai/generate_ai_context_files.py --check` (skill `ai-context-sync`): `AGENTS.md`, `.github/copilot-instructions.md`, `llms-full.txt`, `docs/llms-full.txt` y `docs/llms.txt`. El bloque `SKILLS-LIST` de `CLAUDE.md` no cambia (no hay skills nuevas).

### Problemas encontrados en los archivos de referencia (ajenos a Timestral)

Se pueden arreglar en el PR B si se tocan esos archivos, o dejarlos:

- `foundation-forecasting-models.ipynb`: la celda 101 dice "the five foundation models" (son nueve) y `model_ids_allow_exog` de la celda 45 no incluye `google/timesfm-3.0-pytorch`.
- La celda 2 del notebook de series heterogéneas nombra a TS-ICL pero no a Nori en la fila sobre el agrupado (sin verificar si es una omisión).

## 10. Revisión de upstream antes de empezar

La sesión de implementación debe revisar esto primero. Estado a 29/10/2026 **[probado]** salvo donde se indica.

Revisado de nuevo el 05/11/2026 **[probado]**: sin cambios. `timestral` sigue sin estar en PyPI (404); `main` sigue en el único commit `a534bdb`, sin tags ni releases; las cuatro revisiones de Hugging Face son las de la sección 2, con `license: other`, sin `license_name` y sin acceso restringido; siguen los 7 issues, sin PR y sin respuesta de Mistral (el último comentario, de la comunidad, es del 28/10/2026), y las integraciones de sktime y darts siguen abiertas. No se han revisado el informe técnico ni el anuncio.

| Punto | Dónde mirarlo | Estado el 29/10/2026 | Si ha cambiado |
|:------|:--------------|:---------------------|:---------------|
| Release en PyPI | https://pypi.org/pypi/timestral/json | 404 (no publicado) | `backend_package = "timestral"` (o `"timestral>=X"`) en la clase y en `adapters.toml`; quitar la nota de instalación desde git de la documentación y del `ImportError`; resolver D1 |
| Commits nuevos en `main` | https://github.com/mistralai/timestral/commits/main (o `git ls-remote https://github.com/mistralai/timestral.git`) | Un solo commit, `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` (15/10/2026) | Comparar con `a534bdb` `src/timestral/forecaster.py` y `loading.py`: firmas de `from_pretrained` y `forecast`, atributos de `Forecast`. Pasar el contrato de `contracts.py` contra la versión nueva y repetir `skeleton/run_checks.py` |
| Releases y tags | https://github.com/mistralai/timestral/releases | Ninguno | Si hay tags, valorar fijar `backend_package` a la versión |
| Licencia de los pesos | https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE (y los otros tres) | Modified MIT con el límite de 20 M USD, idéntica en los cuatro (sha256 `6481947279...`) | Revisar D2, la entrada de `_MODEL_LICENSES` y los textos de licencia de la documentación |
| `license` y `license_name` de la tarjeta | https://huggingface.co/api/models/mistralai/Timestral-209M (`cardData`) | `license: other`, sin `license_name` | Si aparece `license_name`, usarlo como primer campo de `_MODEL_LICENSES` (lo compara `check_foundation_models_metadata.py`) |
| Revisión de los pesos | API del Hub, campo `sha` de cada modelo | Las cuatro revisiones de la sección 2 | Repetir `exp2_backtest.py` con el 28M y el 209M y actualizar las cifras de la documentación |
| Issue #5 (PyPI) | https://github.com/mistralai/timestral/issues/5 | Abierto; un comentario de la comunidad (sktime), sin respuesta de Mistral | Ver "Release en PyPI" |
| Issue #6 (MPS y memoria) | https://github.com/mistralai/timestral/issues/6 | Abierto, sin respuesta; fork no oficial con MPS | Si se corrige: quitar `mps_fallback_message` y `stacklevel` de la llamada del adapter, las notas de MPS y el test 33 |
| Issue #7 (covariables planas) | https://github.com/mistralai/timestral/issues/7 | Abierto, sin comentarios ni respuesta | Si se corrige: valorar quitar `_warn_if_constant_known_covariates` (D3) y repetir `exp4_cost_and_issue7.py` |
| Issue #3 (datos de entrenamiento) | https://github.com/mistralai/timestral/issues/3 | Abierto; dos comentarios, sin respuesta | Solo afecta al texto sobre los benchmarks |
| Issues #1, #2 y #4 | https://github.com/mistralai/timestral/issues | Abiertos, sin respuesta (#1 entrada con DataFrames, #2 verificación de evals, #4 MPS) | #4: como el #6 |
| Issues o PR nuevos | https://github.com/mistralai/timestral/issues?q=is:issue | 7 issues, 0 PR, ningún comentario de Mistral | Leer los nuevos; cualquier cambio de API o de comportamiento con covariables cambia el diseño |
| Informe técnico | https://huggingface.co/papers/2610.37585 | Publicado el 27/10/2026 en HF Papers; no leído (arXiv bloqueado en este entorno) | Leerlo: datos de entrenamiento, recomendaciones de uso de covariables |
| Integraciones de terceros | https://github.com/sktime/sktime/pull/11569, https://github.com/unit8co/darts/pull/3270 | Abiertas, sin fusionar | Revisar cómo tratan MPS, la licencia y la instalación |
| Anuncio de Mistral | https://mistral.ai/news/timestral | No leído: el entorno no permite `mistral.ai` | Leerlo si la red lo permite |

## 11. Versiones fijadas y entorno

### Versiones

| Componente | Versión |
|:-----------|:--------|
| timestral | 1.0.0, commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` |
| Pesos en HF | Revisiones de la tabla de la sección 2 |
| Python | 3.12.3 |
| torch | 2.14.1 (wheel de PyPI con CUDA 13.0, `2.14.1+cu130`, ejecutado en CPU) |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| chronos-forecasting | 2.3.2 |
| huggingface-hub | 2.4.0 |
| safetensors | 0.8.0 |
| skforecast | 0.27.0, editable, commit `9fc0fa1` de `0.27.x` |

`dev/timestral_adapter/experiments/requirements.txt` fija lo necesario y `requirements-freeze.txt` es el `uv pip freeze` completo del entorno (Linux x86_64, incluye los paquetes `nvidia-*` del wheel de torch).

### Recrear el entorno en la nube (claude.ai/code)

En este entorno el índice de PyTorch solo para CPU no estaba disponible (`download-r2.pytorch.org` bloqueado), así que torch se instala desde PyPI (wheel con CUDA, unos GB, que funciona en CPU). Si el índice de CPU está disponible, se puede añadir `--torch-backend cpu`.

```bash
uv venv ~/.venvs/timestral-exp --python 3.12
```

```bash
uv pip install --python ~/.venvs/timestral-exp/bin/python -r dev/timestral_adapter/experiments/requirements.txt
```

```bash
uv pip install --python ~/.venvs/timestral-exp/bin/python -e .
```

Para el PR B no hace falta este entorno salvo para volver a ejecutar el notebook y `run_checks.py`: los tests usan objetos falsos. Para los tests basta el entorno del SessionStart hook; `ruff` está en `~/.venvs/skforecast/bin/ruff`.

### Recrear el entorno en local (conda)

```bash
conda create -n timestral-exp python=3.12
```

```bash
conda activate timestral-exp
```

En Linux, para no descargar el wheel con CUDA, instalar antes torch desde el índice de CPU; en macOS el wheel de PyPI ya es de CPU (MPS no funciona con timestral 1.0.0):

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r dev/timestral_adapter/experiments/requirements.txt
```

```bash
pip install -e .
```

## 12. Reproducción de los experimentos

Todo se ejecuta desde `dev/timestral_adapter/experiments/` con el entorno de la sección 11; ningún script depende de rutas del scratchpad. Los pesos se descargan la primera vez a la caché del Hub (21 MB el 5M, 111 MB el 28M, 490 MB el 123M, 837 MB el 209M; Chronos-2 para `exp2`). Con los pesos ya en caché, `HF_HUB_OFFLINE=1` evita las consultas al Hub. Tiempos aproximados en CPU con 4 hilos.

| Archivo | Qué hace | Cómo se ejecuta | Resultado esperado |
|:--------|:---------|:----------------|:-------------------|
| `timestral_proto.py` | Prototipo del adapter usado por `exp2` y `exp3` (registrado en tiempo de ejecución con `register()`). Es una versión anterior del diseño: rellena dentro del adapter y no tiene el aviso del issue #7 | Lo importan los scripts | No aplica |
| `exp1_smoke.py` | Backend directo: lote con relleno NaN frente a cada serie sola, tiempos de una llamada sin covariables (5M, 28M, 209M) | `~/.venvs/timestral-exp/bin/python exp1_smoke.py > exp1.log` (unos 2 minutos) | `exp1.log`: diferencias del orden de 1e-5 y los tiempos de la sección 2 (los tiempos varían con la máquina) |
| `exp2_backtest.py` | Backtesting de la guía con Chronos-2 y Timestral, 15 configuraciones, sobre las 720 horas del test | `~/.venvs/timestral-exp/bin/python exp2_backtest.py > exp2.log` (unos 15 minutos sin descargas) | Añade filas a `results_exp2.csv` y salta las que ya están: para repetirlo desde cero, borrar o renombrar el CSV antes. El CSV y `exp2.log` actuales salen de una sola ejecución completa (29/10/2026). Los tiempos varían con la carga de la máquina |
| `exp3_multiseries.py` | Lote heterogéneo por `FoundationModel` frente a cada serie sola, cuantil no soportado, coste de lotes con covariables (28M), promedio simétrico con una covariable plana (209M) | `~/.venvs/timestral-exp/bin/python exp3_multiseries.py > exp3.log` (unos 2 minutos) | `exp3.log` |
| `exp4_cost_and_issue7.py` | Coste de una llamada frente a trozos de 100 series (209M) e issue #7 con sus remedios | `~/.venvs/timestral-exp/bin/python exp4_cost_and_issue7.py > exp4.log` (unos 10 minutos) | `exp4.log` |
| `exp5_backend_checks.py` | Backend directo: issue #7 con los datos de `exp4` en el 5M, el 28M y el 209M (precio plano e indicador 0/1), dependencia de `prediction_length` con covariables futuras, fila de covariable a NaN, coste del promedio simétrico, float32 frente a float64, contexto a NaN | `~/.venvs/timestral-exp/bin/python exp5_backend_checks.py > exp5.log` (unos 5 minutos) | `exp5.log` |
| `exp6_padding_cost.py` | Coste de `_left_pad_context` (extraído de este documento) frente al relleno con numpy que hace T0 hoy, 400 series de 500 a 4096 puntos con 2 exog; comprueba que el contexto y las exog apiladas son idénticos | Primero `skeleton/extract_skeleton.py` (fila siguiente), después, desde esta carpeta, `~/.venvs/timestral-exp/bin/python -W ignore exp6_padding_cost.py > exp6.log` (segundos) | `exp6.log` |
| `exp7_gpu_chunk_size.py` | PR C, paso C3: tiempo y pico de memoria de GPU de una llamada frente a trozos de 25, 50, 100 y 200 series (backend directo, 28M y 209M, contexto 512 a 4096, 0 a 4 covariables conocidas); un modo por ejecución (`gpu`, `gpu-long`, `gpu-extra`, `cpu`, `cpu-extra`) | En el entorno conda `skforecast_27_gpu` (CUDA), desde esta carpeta: `python -W ignore exp7_gpu_chunk_size.py <modo> > <log>` (la docstring da cada log). El modo `gpu` tarda unos 25 minutos en una GPU de 8 GB, por las llamadas sin trocear que desbordan la memoria | `exp7.log`, `exp7_long.log`, `exp7_gpu_extra.log`, `exp7_cpu.log`, `exp7_cpu_extra.log` |
| `exp8_chunk_equivalence.py` | PR C: predicciones y tiempo de `FoundationModel` con Timestral-28M sin trocear, con trozos de 50 y de 100 (230 series de 300 a 700 puntos, una covariable) | Desde la raíz del repo, con `PYTHONPATH` en la raíz y el entorno `skforecast_27_gpu`: `python -W ignore dev/timestral_adapter/experiments/exp8_chunk_equivalence.py cpu` (y `cuda`) | `exp8_cpu.log`, `exp8_cuda.log` |
| `skeleton/extract_skeleton.py` | Extrae de este documento los bloques marcados `<!-- skeleton:... -->` y escribe `timestral_skeleton.py` y `shared_changes.py` (con importaciones absolutas) y `foundation_loop.txt` (el bucle de `FoundationModel.predict`) | `cd skeleton && ~/.venvs/timestral-exp/bin/python extract_skeleton.py ../../research_timestral_adapter.md` | Los tres archivos generados (no se guardan) |
| `skeleton/run_checks.py` | 32 chequeos del esqueleto con el backend real: contrato y parámetros, smoke test con `FoundationModel`, `ForecasterFoundation` y `backtesting_foundation`, lote heterogéneo, cuantiles, tipos de covariables, MPS (también a través de `ForecasterFoundation`, con `torch.backends.mps.is_available` parcheado), aviso del issue #7, extensión de índices, T0 con el relleno de `FoundationModel` (backend falso) y el fake. Inserta el bucle de `foundation_loop.txt` en el código actual de `FoundationModel.predict` | `cd skeleton && HF_HUB_OFFLINE=1 ~/.venvs/timestral-exp/bin/python -W ignore run_checks.py` (unos 2 minutos; usa el 28M y el 5M) | `32/32 checks passed` (`run_checks.log`) |
| `skeleton/fake_timestral.py` | Borrador de `FakeTimestralForecaster` (sección 8), probado con el esqueleto en `run_checks.py` | Referencia para `fixtures_adapters.py` | No aplica |
| Lint | `ruff` sobre los módulos extraídos | `cd skeleton && ~/.venvs/skforecast/bin/ruff check --config ../../../../pyproject.toml timestral_skeleton.py shared_changes.py` | `All checks passed!` |

## 13. Definición de terminado de la implementación

Para cada PR:

1. Las decisiones de la sección 6 que le afectan están resueltas y aplicadas.
2. Tests de los archivos tocados, de uno en uno y sin `-n`: `pytest skforecast/foundation/tests/tests_foundation_models/test_TimestralAdapter.py -x -q` mientras se itera, y al final `pytest skforecast/foundation/tests/ -q` (el módulo tocado entero) y los tests de los llamantes de `FoundationModel.predict`, que `verify` no relaciona con el módulo: `pytest skforecast/model_selection/tests/tests_validation/test_backtesting_foundation.py skforecast/model_selection/tests/tests_search/test_bayesian_search_foundation.py -q`. No lanzar la suite completa.
3. `ruff check` sin errores en los archivos tocados.
4. Skill `verify` en verde antes del commit y antes de abrir el PR (incluye lint, tests afectados, AI context y release notes). El check de `tools/foundation_compat` de `verify` falla por un problema previo (sección 8): decirlo, no ocultarlo.
5. PR B además:
   - Release note en `docs/releases/releases.md`, sección 0.27.0 (skill `release-note`).
   - AI context sincronizado: fuentes editadas a mano (sección 9), `python tools/ai/generate_ai_context_files.py` y `--check` en verde (skill `ai-context-sync`).
   - Los dos notebooks de la sección 9 ejecutados como dice el paso B5, con `python tools/docs/execute_notebooks/execute_notebooks.py --check` en verde.
   - `python tools/check_foundation_models_metadata.py` en verde para los cuatro checkpoints (necesita red).
   - Prueba con el backend real en CI: el workflow `foundation-backends.yml` no corre en los PR a `X.Y.x`, pero tiene `workflow_dispatch`. Tras añadir `timestral` a la matriz: `gh workflow run foundation-backends.yml --ref feature/timestral-adapter`, y el resultado en la sección de verificación del PR (opcional también en el PR A, para T0 y Moirai en CPU). Sus smoke tests usan series de la misma longitud, así que no prueban el relleno: eso lo cubren los tests del paso A4. En local, si el entorno lo permite: `python tools/foundation_compat/run.py --adapter timestral` (instala el backend desde git; sin verificar en este entorno).
   - `skeleton/run_checks.py` adaptado al adapter real (registrado en el paquete, sin la simulación del relleno) o un smoke test equivalente con los pesos reales.
6. Commits y PR sin atribución de IA (`attribution_guard.py`), contra la rama `X.Y.x`.

## 14. Riesgos

- **Mantenimiento del backend.** Un solo commit, sin releases ni tags, no aceptan pull requests y no han respondido a ningún issue. La API es pequeña y clara (`from_pretrained` y `forecast`), pero puede cambiar sin aviso antes de llegar a PyPI. El check semanal de `tools/foundation_compat` lo detectaría.
- **Benchmarks sin verificar.** Las cifras de la model card son de Mistral. No se sabe qué datos se usaron para entrenar ni si se excluyeron las ventanas de test de los benchmarks (issue #3). Nuestro experimento es un solo dataset.
- **Fallo silencioso con covariables planas** (issue #7, reproducido). Es el más grave para los usuarios de retail con precios como exog. El aviso propuesto lo hace visible, pero no lo corrige.
- **Coste en CPU con covariables.** Unas 7 u 8 veces más por serie. Esos tiempos ya incluyen el promedio simétrico, activo por defecto, que por sí solo duplica el coste: desactivarlo lo reduciría a la mitad, a cambio de cambiar la previsión. El 209M con 4096 de contexto tarda 73.0 s en 30 folds de una sola serie. En backtesting con muchas series hará falta GPU o un checkpoint pequeño.
- **MPS no funciona** (issue #6). En Apple Silicon solo se puede usar la CPU.
- **Cobertura de los intervalos.** Por debajo del nominal en el experimento (0.69 a 0.78 para un 80 %), como en Chronos-2.
- **Previsión dependiente de `steps` con covariables futuras.** Cambiar el horizonte cambia los primeros pasos (sección 2, Covariables). Hay que documentarlo para que nadie compare backtestings con distinto `steps` paso a paso.
- **Contexto entero a NaN.** Una serie cuya ventana de contexto queda entera a NaN tras recortar a `context_length` llega al backend, que devuelve una previsión en torno a 0 sin avisar (sección 2, Entrada). Pasa con `fit` y con `predict(context=...)`; el backtesting ya está protegido, porque `_backtesting_foundation` descarta las series cuya ventana recortada es entera a NaN. Afecta a cualquier adapter con `supports_nan_in_series=True`, así que la solución natural es una comprobación compartida en `FoundationModel`, siguiendo el modelo de `_backtesting_foundation`. Queda fuera de estos PR: propuesta para un issue aparte.
- **Validación del backend con `assert`.** Con `python -O` desaparecen las comprobaciones de formas. El adapter construye las formas correctas, pero un error en el adapter llegaría como un fallo poco claro dentro del backend.
- **No leídos:** el anuncio de Mistral (la red del entorno no permite `mistral.ai`) y el informe técnico (arXiv bloqueado). Lo que se cita del anuncio sale de la prensa y de la model card.

## 15. Fuentes

Oficiales:

- Repositorio y README: https://github.com/mistralai/timestral (commit `a534bdb`, 15/10/2026). Código leído: `src/timestral/forecaster.py`, `model.py`, `masks.py`, `layers.py`, `scaler.py`, `config.py`, `loading.py`, `covariate_encoding.py`, `tests/test_forecaster.py`, `pyproject.toml`, `CONTRIBUTING.md`, `LICENSE`.
- Colección de Hugging Face: https://huggingface.co/collections/mistralai/timestral
- Model card del 209M (benchmarks): https://huggingface.co/mistralai/Timestral-209M
- Configuraciones de los cuatro checkpoints: `https://huggingface.co/mistralai/Timestral-{5M,28M,123M,209M}/resolve/main/config.json`
- Licencia de los pesos: https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE
- Metadatos del Hub (no gated, fechas, `license: other`, revisiones): https://huggingface.co/api/models/mistralai/Timestral-209M
- Informe técnico: https://huggingface.co/papers/2610.37585 (no leído)
- PyPI (sin publicar, 404): https://pypi.org/project/timestral/
- Anuncio: https://mistral.ai/news/timestral (no leído: la red de este entorno no permite `mistral.ai`)

Issues del repositorio (todos abiertos y sin respuesta de Mistral el 29/10/2026):

- #3 Datos de entrenamiento y posible fuga de los benchmarks: https://github.com/mistralai/timestral/issues/3
- #5 Publicar en PyPI: https://github.com/mistralai/timestral/issues/5
- #6 Fallo en MPS por float64 y memoria de la atención entre variables: https://github.com/mistralai/timestral/issues/6
- #7 Cuantiles superiores disparados con covariables de signo fijo y constantes a trozos: https://github.com/mistralai/timestral/issues/7

Integraciones en curso: sktime (https://github.com/sktime/sktime/pull/11569) y darts (https://github.com/unit8co/darts/pull/3270), ambas abiertas. No he encontrado ninguna en AutoGluon ni en Nixtla, ni ninguna evaluación independiente.

Prensa (repite las cifras de Mistral):

- https://aiweekly.co/alerts/mistral-opens-timestral-209m-time-series-model-beating-timesfm-3
- https://www.startuphub.ai/ai-news/ai-research/2026/mistral-says-209m-timestral-leads-on-three-benchmarks.md
- https://tech-insider.org/mistral-timestral-209m-tops-timesfm-3-2026/
- https://braindetox.kr/en/posts/timestral_mistral_time_series_model.html

skforecast (commit `9fc0fa1` de `0.27.x`): `.claude/rules/foundation.md`, `skforecast/foundation/_adapter_base.py`, `_adapters.py`, `_foundation_model.py`, `_model_info.py`, `_utils.py`, `skforecast/exceptions/exceptions.py`, `skforecast/foundation/tests/`, `tools/foundation_compat/`, `tools/check_foundation_models_metadata.py`, `.github/workflows/foundation-backends.yml`, `pyproject.toml`, `skills/foundation-forecasting/`, `tools/ai/llms-base.txt` y los notebooks de `docs/user_guides/` sobre foundation models.

## Anexo A. Revisión del 29/10/2026: errores corregidos

Errores de la versión anterior de este documento, encontrados al revisarlo contra el código de skforecast, el código de Timestral, los logs y la ejecución del esqueleto, y cómo quedan.

**Esqueleto y diseño**

| # | Antes | Después |
|--:|:------|:--------|
| 1 | `set_params` usaba `validate=self._validate_params`, que no existe en ningún adapter | Cierre `validate` como `TimesFM3Adapter.set_params` |
| 2 | `__init__` llamaba a `self._validate_predict_kwargs`, que no estaba definido en la clase | Classmethod copiado de `TimesFM3Adapter` |
| 3 | El esqueleto llamaba a `self._build_covariates`, `_left_pad_batch`, `_match_quantile_indices` (como función) y `_resolve_torch_device(..., allow_mps=False)`, que no existían | Código completo: `_build_covariates` y los conversores en el adapter; los helpers, marcados como cambios de código compartido (sección 4) |
| 4 | Al ejecutarlo, una exog booleana de una serie rellenada se convertía en `object` al concatenar histórico y futuro, y el adapter lanzaba `ValueError` **[probado]** | Histórico y futuro se convierten por separado y se concatenan como arrays; el relleno convierte las columnas booleanas a float |
| 5 | Recomendaba un helper de relleno llamado por cada adapter, y decía que el relleno no se podía hacer en `FoundationModel` porque las series terminan en fechas distintas. Lo primero incumple `.claude/rules/foundation.md` y lo segundo es falso | Atributo `requires_equal_length_context` y relleno en `FoundationModel`, comprobado con 269 fechas finales distintas y con T0 |
| 6 | `backend_package = "git+https://github.com/mistralai/timestral.git"`: `run.distribution_name` devolvería `"git"` | `"timestral @ git+https://github.com/mistralai/timestral.git"` (PEP 508) |
| 7 | Situaba `_resolve_torch_device` en `_utils.py` | Está en `_adapters.py`, y los tests la parchean allí |
| 8 | MPS: proponía un error o un aviso sin citar el precedente de Moirai | El aviso y el paso a CPU de Moirai, centralizados en el helper |
| 9 | Sin docstrings, sin type hints en `fit` y `predict`, sin el comentario `# NOTE` de la carga perezosa y con comillas simples | Docstrings completos como TimesFM3, type hints, comentario y comillas dobles |
| 10 | Las columnas categóricas se decidían por el dtype de la primera serie, sin justificarlo | Justificado: `FoundationModel` rechaza columnas con distinto dtype entre series **[probado]** |
| 11 | No trataba los infinitos de las covariables numéricas, que el backend no enmascara | `_to_float_array` los convierte a NaN |
| 12 | "Ningún otro adapter expone opciones de carga" | Chronos y T0 (`torch_dtype`), TS-ICL y TimesFM 2.5 sí; ninguno expone `revision` ni la caché |

**Cifras**

| # | Antes | Después |
|--:|:------|:--------|
| 13 | "Pasar de 500 a 4096 baja el MAE del 209M un 28 %" | Un 29 % (de 135.67 a 96.89 con las cifras actuales). El 28 % es frente a Chronos-2 con 4096 |
| 14 | "Timestral cubre algo más que Chronos-2" | Solo los tamaños pequeños y medianos; el 209M con contexto largo cubre igual o menos |
| 15 | Tabla de tiempos de `exp1` sin log que la respaldara (cargas de 3, 4 y 12 s) | Repetido, con `exp1.log`; tabla nueva (carga desde la caché) |
| 16 | "Unos 30 ms por serie con el 28M" | 46 ms una serie de 512 y 0,8 s cien series (`exp1.log`) |
| 17 | "Con covariables, cada serie cuesta unas 8 veces más" | Entre 7 y 8 veces (6,8 en el 28M y 8,2 en el 209M) |
| 18 | La tabla del issue #7 citaba `exp3` y `exp4` | Es de `exp4.log`; el caso de una serie de `exp3.log` va aparte con sus cifras |
| 19 | "El mejor valor publicado en la guía es 140.9" | El mejor de la tabla de comparación; el barrido de contexto usa otra ventana de test |
| 20 | La lista de referencia no decía que TimesFM 2.5 y Moirai-2 se evalúan sin exog | Indicado |
| 21 | "TimesFM y Moirai solo admiten el 80 %" | No pueden dar el 90 % ni el 98 % |

**Afirmaciones sobre Timestral**

| # | Antes | Después |
|--:|:------|:--------|
| 22 | "Alterna atención temporal y atención entre variables" | Tres capas temporales por cada capa entre variables |
| 23 | "`R = B * (1 + covariables)`" | `R = B x (V + C_past + C_known + C_cat)`; la máscara existe también sin covariables |
| 24 | "NaN marca un valor ausente en el objetivo y en las covariables" | En el objetivo, NaN e infinitos; en las covariables numéricas, solo NaN |
| 25 | "Las covariables futuras con valores 0/1 se tratan como indicadores" | Solo las futuras, y la pasada simétrica lo rompe |
| 26 | "`symmetric_averaging=False` solo arregla la mediana" | Baja la mediana del 209M, pero no la deja en su nivel sin covariable, y en el 5M y el 28M no arregla nada (`exp5.log`) |
| 27 | "`per_shot_horizon_patches` (<= 32)" | Valores mayores se recortan a 32 sin avisar |
| 28 | "Unas 1.800 líneas" | 1.712 de código y 102 de tests |
| 29 | "El informe técnico está anunciado como coming soon" | Publicado el 27/10/2026 en HF Papers (no leído) |
| 30 | `device` sin más detalle | El valor por defecto es `"cpu"` y `"auto"` no es válido en el backend: lo resuelve el adapter |
| 31 | No mencionaba que con covariables futuras la previsión depende de `steps`, ni que el contexto crece con horizontes de más de 1.024 pasos | Añadido a la ficha y a los riesgos |

**Lista de archivos**

| # | Antes | Después |
|--:|:------|:--------|
| 32 | Faltaban el notebook de series heterogéneas, `docs/overrides/home.html`, los docstrings de dos propiedades de `FoundationModel`, `_search.py`, `search-parameters.md`, la lista `backends` de `test_get_model_info.py`, el enlace al final de `releases.md`, el `description` de la skill y `test_init.py` (etiqueta de categóricas) | Sección 9 completa, obtenida con grep |
| 33 | `test_device_handling.py` como test que enumera los adapters | No los enumera: lleva tests nuevos del helper |
| 34 | El smoke test con el 5M sin decir que rompe una convención | Decisión D10 |

Comprobado y correcto, sin cambios: todas las celdas de los adapters existentes en la tabla de capacidades, las nueve cifras de la guía, el fallo de `check_foundation_models_metadata.py` sin `license_name`, el formato de la entrada de `_MODEL_LICENSES` y la mayoría de las afirmaciones sobre la API de Timestral (unas 50).

**Revisión independiente** (cuatro agentes que recibieron solo este documento y las rutas del repositorio). Todos los hallazgos se han comprobado antes de aplicarlos; ninguno se ha descartado, aunque varios eran el mismo con otro enfoque y se han agrupado.

| # | Hallazgo | Comprobación | Corrección |
|--:|:---------|:-------------|:-----------|
| 35 | `_extend_index_backwards` desplazaba una hora el índice diario con zona horaria al cruzar un cambio de hora: la serie rellenada quedaba entera a NaN, también en T0 | Reproducido | Generación hacia atrás con `date_range(end=...)` y el índice original añadido; chequeo en `run_checks.py` |
| 36 | El helper fallaba con un `DatetimeIndex` sin `freq` (camino de `check_inputs=False`) | Confirmado en `expand_index` | Infiere la frecuencia como `expand_index`; `ValueError` si no puede |
| 37 | El aviso de MPS de Moirai cambiaba de texto y de `stacklevel`, y el documento decía que no | Confirmado en `MoiraiAdapter._ensure_forecast_obj` | Mensaje y `stacklevel` los pasa quien llama; Moirai conserva los suyos; comprobado que el de Timestral apunta a la llamada del usuario |
| 38 | `test_MoiraiAdapter_ensure_forecast_obj_calls_to_with_resolved_device` comprueba la llamada exacta al helper | Confirmado | Paso A6: cambiar la aserción |
| 39 | El bucle de `FoundationModel.predict` usaba `_left_pad_context` sin importarlo, y `run_checks.py` no lo ejecutaba (simulaba el relleno) | Confirmado | Paso A4 con la importación; `run_checks.py` inserta el bloque del documento en el código real del método |
| 40 | El adapter no va después de `TSICLAdapter` (la última clase es `NoriAdapter`) | Confirmado | Corregido |
| 41 | Añadir el atributo obligatorio antes de declararlo en los adapters rompe la importación del paquete | Confirmado en `__init_subclass__` | Pasos A3 y A4 fusionados (y C1) |
| 42 | `test_AdapterBase_TypeError_when_subclass_misses_required_class_attributes` tiene la lista esperada fija | Confirmado | Indicado en el paso A3 |
| 43 | El test 37 usaba percentiles (`[5, 95]`), rechazados desde la 0.23.0 | Confirmado en el docstring de `predict_interval` | Cuantiles y formas `interval=0.9` y `0.98` |
| 44 | El notebook de la guía ejecuta todos los modelos: no se puede volver a ejecutar solo con el backend de Timestral | Confirmado (celdas de todos los backends, `EXCLUDE_NOTEBOOKS`) | Paso B5 con dos opciones y `--check` |
| 45 | El test PEP 508 recomendado importaría `run.py`, que usa `tomllib` (no existe en Python 3.10) | Confirmado | Comprobación con la misma expresión, sin importar `run.py` |
| 46 | El fake aceptaba cualquier `**kwargs`, y el backend no | Confirmado con la firma real | Firma explícita; chequeo en `run_checks.py` |
| 47 | El notebook de series heterogéneas cambia una celda de código y no se volvía a ejecutar | Confirmado (celda 11) | Paso B5 |
| 48 | `adapter-parameters.md` no mencionaba el atributo nuevo en `## Common Behavior` | Confirmado | Sección 9 |
| 49 | "Con el 5M, `symmetric_averaging=False` lo arregla del todo" era falso con los datos de `exp4` | Reproducido en `exp5.log` (el 5M empeora) | Tabla nueva por checkpoint; el único remedio documentado es el histórico a NaN |
| 50 | El issue #7 también afecta a los indicadores 0/1, y el mecanismo de normalización no lo explica | Reproducido en `exp5.log` | Ficha, aviso y docstring; causa marcada como no confirmada; el aviso saltará en un fold de la guía (comprobado) |
| 51 | Una ventana de contexto entera a NaN sí llega al backend (la comprobación de `fit` es anterior al recorte) | Reproducido con `FoundationModel.fit` | Ficha y riesgo nuevo |
| 52 | "Entre 7 y 8 veces" con un 6,8, filas de la tabla con covariables distintas, y el promedio simétrico contado dos veces | Recalculado | Texto corregido |
| 53 | La capa `time_variate` también tiene atención temporal | Confirmado en `config.LAYER_KIND_AXES` | Texto corregido |
| 54 | Con `context_length` mayor que 32.768, el backend recorta sin avisar | Confirmado en `_assemble_rows` | Docstring del adapter |
| 55 | Varias cifras marcadas como probadas sin script ni log en `dev/timestral_adapter/experiments/` (0,039; 0,043; 3,5 s frente a 1,8 s; 4 hilos; 239 ms) | Confirmado | `exp5.log` y `exp6.log` con las cifras nuevas; la de los hilos se ha quitado |
| 56 | Tres filas de `results_exp2.csv` (las de 4096) sin log | Confirmado | Superado por el 77: el backtesting se ha repetido entero en una sola ejecución con log |
| 57 | La mediana del 209M con `symmetric_averaging=False` (5.30) no está "arreglada" frente a 2.02 | Confirmado | Texto corregido |
| 58 | El resumen hablaba de dos cambios compartidos y hay cuatro; D11 hablaba de diez adapters en el PR A | Confirmado | Corregido |
| 59 | El fake se había probado con el esqueleto, no con el prototipo | Confirmado | Corregido |

**Segunda ronda de revisión independiente** (los mismos cuatro enfoques, sobre el documento ya corregido). Todos los hallazgos se han comprobado y ninguno se ha descartado.

| # | Hallazgo | Comprobación | Corrección |
|--:|:---------|:-------------|:-----------|
| 60 | Las fechas de relleno generadas hacia atrás fallan (`NonExistentTimeError`, `AmbiguousTimeError`) cuando caen en un cambio de hora a la hora de la serie, por ejemplo a medianoche en `America/Santiago`; T0 hoy no falla | Reproducido | Relleno posicional (`RangeIndex` común al grupo), sin generar fechas; desaparece `_extend_index_backwards`; chequeo nuevo en `run_checks.py` |
| 61 | Los pasos del PR B dejaban los tests en rojo (la clase registrada sin licencia, fake ni `adapters.toml`) | Confirmado en `_build_model_info` y en los tests que recorren el registro | B1 (fake y licencia) y B2 (clase, registro, tests y `tools/` en un solo paso) |
| 62 | La opción (b) de ejecución del notebook no sirve para la celda 102, que usa las métricas de todos los modelos | Confirmado | Reconstruir `metrics_*` desde las salidas guardadas, o la opción (a) |
| 63 | Las parametrizaciones de `test_clone.py` y `test_set_params.py` tienen listas `ids` explícitas | Confirmado | Ids indicados |
| 64 | El punto 10 de `## Common Mistakes` de la skill también enumera los adapters con covariables solo pasadas | Confirmado | Añadido a la sección 9 |
| 65 | `_model_info.py` no lee los atributos por introspección, y el documento no decidía si el atributo nuevo se expone | Confirmado (`_build_model_info` copia una lista fija) | Atributo interno, no expuesto; comprobado en el test de contrato de `test_AdapterBase.py` |
| 66 | El aviso del issue #7 (`stacklevel=4`) apuntaba dentro de `_forecaster_foundation.py` | Reproducido | `stacklevel=5`; chequeo de que apunta a la llamada del usuario |
| 67 | El paso A7 no decía cuál de las dos asignaciones de `hub_license` cambia | Confirmado | Especificado (ahora en el paso B2) |
| 68 | Faltaban los tests de los llamantes en `model_selection` | Confirmado en `affected.py` | Añadidos a la comprobación del PR A y a la sección 13 |
| 69 | El caso de Moirai con `"mps:0"` es un fallo de una versión publicada: va en *Fixed* | Confirmado con la skill `release-note` | Entrada propuesta |
| 70 | El PR B depende de nombres que solo existen tras el PR A | Confirmado | Rama del PR B tras fusionar el PR A |
| 71 | En backtesting no llegan ventanas enteras a NaN (`_backtesting_foundation` las descarta) | Confirmado en `_validation.py` | Ficha y riesgo limitados a `fit` y `predict(context=...)` |
| 72 | `_MODEL_ID_PREFIX` también lo usa TimesFM 2.5 | Confirmado | Tabla de alineación corregida |
| 73 | El resumen del issue #7 decía que solo la mediana del 209M se rompe y que `symmetric_averaging=False` no arregla nada | Confirmado en `exp5.log` (28M con indicador; 209M con indicador sí lo arregla) | Texto reescrito |
| 74 | La dependencia de `steps` se atribuía solo a la normalización; la causa principal es la atención bidireccional de las filas de covariables futuras | Confirmado en `masks.build_two_stream_time_mask` | Texto corregido |
| 75 | El caso del indicador no se había medido en el 5M | Confirmado | Añadido a `exp5_backend_checks.py` |
| 76 | El docstring decía que NaN es un ausente también en las covariables categóricas (allí es una categoría propia) | Confirmado en `covariate_encoding` | Docstring corregido |
| 77 | `exp2_backtest.py` calculaba las métricas sobre 719 puntos (`.iloc[1:]` quitaba la primera hora del test); la diferencia con el 171.27 de la guía venía de ahí, no de torch | Confirmado | Script corregido y backtesting repetido entero en una sola ejecución; todas las cifras actualizadas |
| 78 | "Menos de la mitad del coste" en D9 solo vale para el 209M | Confirmado | Corregido |
| 79 | "Los Timestral pequeños y medianos cubren más que Chronos-2" solo vale con exog | Confirmado | Corregido |
| 80 | `exp6` medía su propio relleno, no el helper del documento, y no comparaba las exog | Confirmado | `exp6` usa `_left_pad_context` extraído y compara contexto y exog |
| 81 | El estado decía "cuatro experimentos" | Confirmado | Seis |
| 82 | El fake no es el último chequeo de `run_checks.py` | Confirmado | Corregido |

**Revisión final de D5 y D11** (30/10/2026, cuatro revisores independientes: corrección por lectura de código, comparación con alternativas, estrategia de PR y ejecución de casos límite). Veredicto de los cuatro: las dos recomendaciones son correctas y la mejor opción; ninguno propuso otra. La ejecución comparó T0 antes y después del cambio en 47 escenarios (arrays, salidas y avisos idénticos), Timestral en lote frente a cada serie sola con el checkpoint real (diferencias relativas de 1e-7), `backtesting_foundation` con `refit` y `bayesian_search_foundation`, y pandas 2.2.0 y 3.1.0. Mejoras aplicadas, todas comprobadas:

| # | Hallazgo | Comprobación | Corrección |
|--:|:---------|:-------------|:-----------|
| 83 | `_left_pad_context` copiaba cada exog con `astype` aunque no hubiera columnas booleanas y reindexaba series sin relleno: varias veces más lento de lo necesario, con más peso en T0 y en backtesting | Medido: misma salida, de 327 ms a 130 ms con 200 series; `exp6`, de 485 ms a 170 ms | Solo convierte las columnas booleanas que existen y solo reindexa si hay relleno |
| 84 | El texto decía que las columnas enteras pasan a float; solo pasa en las series rellenadas | Reproducido | Docstring del helper y sección 4 corregidos |
| 85 | El docstring del atributo cubría solo `context` y no tenía sitio en `_AdapterBase` | Confirmado (sin sección Attributes) | Texto completo en el paso A3: `context_exog`, `exog` por posición, dtypes, `supports_nan_in_series`, adapters que usan fechas, y los parámetros de `predict` |
| 86 | Un test de lote frente a serie sola debe usar el mismo `steps` | Reproducido en un fold de backtesting | Test 27 |
| 87 | El cambio de `license_name` en `check_foundation_models_metadata.py` sería código muerto en el PR A | Confirmado: solo Timestral lo necesita | Pasa al paso B2, donde lo prueba el workflow `foundation-models-metadata.yml` |
| 88 | "Se rebasa sobre `0.27.x`" exigiría un force push, que está bloqueado | Confirmado en `push_guard.py` y en el historial (merge commits) | Merge de `origin/0.27.x`, nunca rebase |
| 89 | BUG-04 y QW-05 del roadmap tocan las mismas declaraciones de los adapters, y un atributo obligatorio nuevo rompe la importación sin conflicto en git | Confirmado en `dev/roadmap_0.27.0/roadmap.md` | Nota de coordinación en D11 y merge de `0.27.x` antes de fusionar cada PR |
| 90 | D11 decía que el PR A no tenía cambios visibles; la rama `feature/` no encajaba; el PR B no tenía prueba con el backend real en CI; la sesión siguiente no tendría el documento | Confirmado | Coherencia con el *Fixed* de Moirai; rama `chore/`; `workflow_dispatch` de `foundation-backends.yml`; recomendación de subir el documento y los logs (`git add -f`) a `0.27.x` antes del PR A |

**Revisión final del PR A** (04/11/2026). El PR A de la sección 7 se aplicó a una copia de `0.27.x` (`924ab7f`), sin tocar el repositorio, con Python 3.12.3, pandas 2.3.3, numpy 2.5.4 y torch 2.14.1:

- `ruff check` limpio. Con los tests de `skforecast/foundation/tests`, `test_backtesting_foundation.py` y `test_bayesian_search_foundation.py`: 936 pasan y 4 fallan, los cuatro que el plan ya cambia (el mensaje de `test_AdapterBase`, que sale exactamente como dice A3; `test_T0Adapter_predict_batches_all_series_in_one_call`, y los dos de dispositivo de Moirai).
- T0 recibe los mismos arrays (contexto, covariables e índice de las predicciones) antes y después del cambio en 10 escenarios: exog `float`, `int64`, `Int64` y `boolean` con y sin NA, `Float64`, `bool`, exog `Series` con una serie sin exog en un índice horario con zona horaria, sin exog y `levels` parcial.
- El aviso de MPS de Moirai sale en la misma línea de la llamada del usuario con `"auto"` y `"mps"`; con `"mps:0"`, que antes iba a MPS sin aviso, ahora pasa a CPU con el aviso.
- Coste: con T0 y un backend falso, 100 series de 300 a 512 puntos y 4 exog, la parte de skforecast de `predict` pasa de 56 ms a 86 ms. Con el modelo real pesa mucho menos que la inferencia.
- Nombres (`requires_equal_length_context`, `_left_pad_context`, `mps_fallback_message`, `group_context`, `group_context_exog`) claros y en la línea de los atributos actuales; el diseño centralizado sigue siendo el más sencillo que permite `.claude/rules/foundation.md`.

| # | Hallazgo | Comprobación | Corrección |
|--:|:---------|:-------------|:-----------|
| 91 | El paso A6 implementaba D7 (a) con D7 todavía abierta | Confirmado | D7 (a) aceptada el 04/11/2026 |
| 92 | El docstring de `_match_quantile_indices` decía "Moved unchanged" pero perdía la sección Notes del original; la frase, además, es historia y no debe ir en el código | Comparado con `TimesFM3Adapter._match_quantile_indices` | Docstring original con Notes, sin la frase; solo cambia el ejemplo de `quantile_grid` |
| 93 | Mover `_match_quantile_indices` en el PR A no servía a nada del PR A: su segundo usuario es Timestral | Confirmado | Pasa al paso B2 del PR B |
| 94 | El docstring del helper, el de A3 y la sección 4 decían que las columnas enteras de las series rellenadas llegan como float; las nullable `Int64` y `Float64` conservan su tipo con `pd.NA` | Reproducido; T0 las convierte igual que hoy | Textos corregidos |
| 95 | A5 no concretaba el cambio, y `arrays` y `lengths` quedarían sin uso | Confirmado en el prototipo | A5 detallado |
| 96 | D11 decía que el PR A toca TimesFM 3.0, y la sección 0 seguía diciendo que este documento estaba pendiente de fusionar | Confirmado | Corregido (#1449 fusionado) |

**Revisión del PR #1450** (05/11/2026, PR A abierto contra `0.27.x`, rama al día con `origin/0.27.x`). En el entorno de desarrollo (Windows, Python 3.13.18, pandas 2.3.3, numpy 2.5.4, torch 2.14.1 CPU con 16 hilos), árbol base `924ab7f` exportado con `git archive` frente a la rama del PR:

- `ruff check` limpio; `skforecast/foundation/tests`, `test_backtesting_foundation.py` y `test_bayesian_search_foundation.py` en verde (954 tests), y los tests cambiados también con pandas 3.1.0 y 2.2.3. Los archivos generados del contexto de IA al día.
- T0 recibe los mismos arrays (contexto y covariables) y el mismo índice de predicciones en 13 escenarios con un backend falso: exog `float`, `int64`, `Int64` y `boolean` con y sin NA, `Float64`, `bool`, dos columnas, sin exog, `levels` parcial (2), índice horario con zona horaria y exog `Series` con una serie sin exog.
- Moirai con `"auto"`, `"mps"` y `"mps:0"`: el aviso sale en la línea de la llamada del usuario (`stacklevel=7` correcto) y el modelo va a CPU; en el árbol base, `"mps:0"` iba a MPS sin aviso.
- Coste con el modelo real (`tfc-t0` 0.3.2, T0-alpha de 101,6M de parámetros, CPU, mínimo de 3 procesos intercalados x 5 repeticiones, 24 pasos, 3 cuantiles). Parte de skforecast de `predict` (total menos el tiempo dentro de `T0Forecaster.predict`), base frente a PR antes de la fila 97: 1 serie de 512 sin exog, 1 ms frente a 1 ms (total 37 ms); 100 series iguales de 512 con 4 exog, 107 frente a 129 ms (total 5,4 s); 100 series de 300 a 512 sin exog, 10 frente a 22 ms (total 1,2 s); 100 series de 300 a 512 con 4 exog, 98 frente a 149 ms (total 5,7 s). Es decir, menos del 1 % de `predict`; la inferencia también varía entre procesos (hasta 0,6 s) por ruido de la máquina, no por el PR: las predicciones del modelo real son idénticas en base y PR en los cinco escenarios **[probado]**. Para el modelo real hace falta `tfc-t0<0.4`: la 0.6.0 cambia la firma de `predict` (issue #1385).
- En el test nuevo de `test_predict.py`, las dos `assert_index_equal` que comparaban el índice de las predicciones (sin `freq`, como todo índice largo de varias series) con `future_long`/`future_short` (con `freq`) emitían `Pandas4Warning` en pandas 3.1 y fallarían cuando pandas 4 compruebe `freq` por defecto. `check_freq` no existe en `assert_index_equal` de pandas 2.x; el esperado se construye ahora con `pd.DatetimeIndex(future_long, freq=None)`, sin aviso en pandas 2.2.3, 2.3.3 y 3.1.0 **[probado]**. El mismo aviso sale en un test anterior del archivo (`assert_index_equal(result.index.unique(), expected_index)`, línea 275), ajeno al PR.
- Pendiente, fuera del PR: el mensaje del `ImportError` de `T0Adapter._load_model` dice `pip install tfc-t0`, que instala una versión incompatible (issue #1385); la guía ya dice `pip install "tfc-t0<0.4"`. Fijar `backend_package = "tfc-t0<0.4"` arreglaría el mensaje, pero cambia el valor público de `FoundationModelInfo.backend_package`, obliga a cambiar `tools/foundation_compat/adapters.toml` (lo comprueba `test_backends_compat_config.py`) y el contexto de IA, y lleva release note. Es mejor resolverlo con el propio #1385 (adaptar `T0Adapter` a la API nueva de `tfc-t0`).

| # | Hallazgo | Comprobación | Corrección |
|--:|:---------|:-------------|:-----------|
| 97 | `_left_pad_context` rellenaba cada serie con `set_axis` + `reindex`, unas 5 veces más lento que el `np.full` que T0 usaba antes (6,4 ms frente a 1,2 ms por 100 series sin exog; 74 frente a 16 ms con 1000) | Medido en el helper aislado y en `predict` completo; salidas idénticas con `assert_series_equal` | Camino de NumPy para las series con dtype de NumPy (float se conserva, el resto a float64; los nullable siguen por `reindex`), código de la sección 4. El helper pasa de 6,4 a 1,8 ms (100 series sin exog) y de 19,3 a 16,0 ms (con 4 exog); el test fija un `int64` que sale `float64` y un `float32` que se conserva |

**Revisión previa al PR B** (05/11/2026, `0.27.x` en `fc7efd0`, con el PR A fusionado). Mismo entorno de desarrollo, sin `timestral` instalado (backend falso):

- `skforecast/foundation/tests` en verde sobre `0.27.x` (886 tests).
- Las funciones compartidas del esqueleto (`_left_pad_context`, `_resolve_torch_device`) tienen el mismo código que las del paquete (comparado con `ast`), y el bucle de `foundation_loop.txt` está entero en `FoundationModel.predict`. `_match_quantile_indices` solo difiere en lo que cambia el paso B2 (deja de ser `staticmethod` y el ejemplo de `quantile_grid`).
- El esqueleto, registrado con `FakeTimestralForecaster` y un módulo `timestral` falso, sin parchear `FoundationModel`: cumple el contrato de `_AdapterBase`; series de distinta longitud y fecha final con exog numéricas y categóricas llegan como lote rectangular rellenado por la izquierda (una llamada por firma de exog); cuantiles en el orden pedido; `LicenseWarning` al cargar; MPS (`"auto"` y `"mps:0"`) pasa a CPU con aviso; `backtesting_foundation` con series heterogéneas y cuantiles **[probado]**.
- Upstream sin cambios (sección 10). BUG-04 y QW-05 del roadmap siguen sin fusionar en `0.27.x`: la coordinación de D11 sigue vigente.

| # | Hallazgo | Comprobación | Corrección |
|--:|:---------|:-------------|:-----------|
| 98 | Tras la fila 97, `_left_pad_context` usa `np`, pero la cabecera de `shared_changes.py` que escribe `extract_skeleton.py` no lo importaba: `ruff` da F821 y `run_checks.py` fallaría con `NameError` en cuanto rellenara una serie | Reproducido | `import numpy as np` en `HEADER_SHARED`; además, lectura y escritura en UTF-8 (en Windows fallaba al leer este documento con cp1252) |
| 99 | El docstring de `_resolve_torch_device` del esqueleto decía "raised as a `UserWarning`" y el del paquete, "issued" | Comparado con el paquete | "issued" |
| 100 | Los avisos del issue #7 (`stacklevel=5`) y de MPS (`stacklevel=6`) apuntan a la línea del usuario con `ForecasterFoundation.predict`, pero llamando a `FoundationModel.predict` directamente se pasan del marco del usuario (`<sys>:0`). Moirai (`stacklevel=7`, ya en `0.27.x`) hace lo mismo, y TabICL, TabPFN y Nori, al revés (`stacklevel=3`: aciertan con `FoundationModel` y apuntan dentro de skforecast con `ForecasterFoundation`) | Reproducido con el backend falso | Sin cambio: ningún `stacklevel` fijo sirve para los dos caminos, el aviso sale igual y `ForecasterFoundation` es el camino principal, como en Moirai. El test 30 fija el camino de `ForecasterFoundation` |
| 101 | `run_checks.py` sigue insertando el bucle de `foundation_loop.txt` en `FoundationModel.predict` y declarando `requires_equal_length_context` en los adapters, cosas que ya hace el paquete | Confirmado (las marcas que busca siguen en `_foundation_model.py`, así que no falla) | Sin cambio ahora: el punto 5 de la sección 13 ya pide adaptarlo al adapter real en el PR B |
