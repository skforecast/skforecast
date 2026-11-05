# TimeH2O (H2O.ai): especificación de un adapter para `skforecast.foundation`

Documento de trabajo. Es la especificación completa para implementar el adapter sin acceso al entorno de los experimentos: todo lo necesario está aquí o en `dev/timeh2o_adapter/experiments/`. Sigue la estructura y el método de `dev/timestral_adapter/research_timestral_adapter.md` (informe del adapter de Timestral, fusionado en `0.27.x` con el PR #1449).

Leyenda de la procedencia de cada dato:

- **[probado]**: comprobado ejecutándolo (en el entorno de los experimentos de la sección 11, con los pesos reales salvo que se diga otra cosa).
- **[verificado en código]**: leído en el código fuente de `h2ofm` 0.43.1 (wheel y sdist de PyPI) o en el código de skforecast de la rama de referencia.
- **[documentado]**: lo dice una fuente escrita (README del paquete, model card, `LICENSE` o `config.json` de Hugging Face, metadatos de PyPI o del Hub), sin comprobarlo aparte.
- **[inferido]**: deducción propia a partir de lo anterior.

## 0. Estado y cómo retomar

- **Fecha:** 05/11/2026.
- **Rama de referencia:** `0.27.x` (skforecast 0.27.0 en desarrollo), commit `fc7efd0db2958d6956358575a3aa4fa9d1627e0f` de `origin/0.27.x` (05/11/2026 15:51 +0100, fusión del PR #1450). Las referencias al código de skforecast son a ese commit. Los experimentos `exp1` a `exp9` se ejecutaron con el commit anterior, `924ab7f` (fusión del PR #1449), sin el PR #1450. A `exp1` y a los experimentos de `exp3` a `exp9` el PR no les cambia nada de lo que usan (el prototipo de TimeH2O declara el atributo nuevo y no usa el paso a CPU desde MPS); `exp2` importa además el prototipo de Timestral, que no declara el atributo y con `fc7efd0` no se puede importar (sección 12). El esqueleto y `run_checks.py` se comprobaron con `fc7efd0`.
- **PR #1450** (rama `chore/foundation-equal-length-context`, último commit `d5832f40ce21655bcd3d9e09a0b55ee72da83f26`), **fusionado en `0.27.x` el 05/11/2026** (`fc7efd0`, mismo código en `skforecast/` que ese último commit): añade el atributo obligatorio `requires_equal_length_context`, el relleno por la izquierda en `FoundationModel` y el paso a CPU desde MPS en `_resolve_torch_device`. El diseño se apoya en él (sección 4).
- **Qué está commiteado:** este documento y `dev/timeh2o_adapter/experiments/`, en la rama `docs/timeh2o-adapter-research` (05/11/2026), que se fusiona en `0.27.x` con un PR propio antes del PR del adapter. Se añaden con `git add dev/timeh2o_adapter`, después de borrar los tres módulos que genera `skeleton/extract_skeleton.py`, que no se guardan y que `.gitignore` no excluye (`rm -f dev/timeh2o_adapter/experiments/skeleton/{shared_changes,timeh2o_skeleton,license_entry}.py`), y los logs, que `.gitignore` excluye (`*.log`), con `git add -f dev/timeh2o_adapter/experiments/*.log dev/timeh2o_adapter/experiments/skeleton/run_checks.log`; nunca con `git add -f` sobre la carpeta entera, que añadiría las cachés de `ruff` y `__pycache__`. No se ha modificado nada fuera de `dev/timeh2o_adapter/`.
- **Qué hay en `dev/timeh2o_adapter/experiments/`:** el prototipo del adapter, los nueve experimentos (`exp1` a `exp9`; los tres últimos, sobre las covariables planas) con sus logs y resultados, los requisitos del entorno y `skeleton/`, que extrae el esqueleto de este documento y lo ejecuta contra el backend real (sección 12).
- **Estado del diseño:** el esqueleto de la sección 4, extraído de este documento, pasa los 62 chequeos de `skeleton/run_checks.py` con los pesos reales y `0.27.x` en `fc7efd0` (con el PR #1450), y `ruff check` con la configuración del repo **[probado]**. Una segunda revisión del documento contra las fuentes, por temas y con una comprobación adicional de cada hallazgo, confirmó 56 problemas y descartó 8; una tercera revisión confirmó 30 hallazgos y descartó 11 (filas 51 a 58 del anexo A), y un ensayo de implementación del plan, salvo los notebooks (paso 5 de la sección 7), en una copia de `fc7efd0` (tests, contexto de IA, `tools/foundation_compat` y backtesting con los pesos reales en verde) señaló 18 puntos más (filas 59 a 68). Todos están corregidos. Las quince decisiones de la sección 6 están abiertas, todas con una recomendación.

**Primeros pasos para implementarlo:**

1. **Preguntar qué entorno de Python usar, antes de ejecutar nada.** No elegirlo por cuenta propia, no listar los entornos (`conda env list`) y no crear ninguno sin que se pida. Hay dos usos, que pueden ser el mismo entorno o dos distintos:
   - **Desarrollo y tests del adapter** (`pytest`, `ruff`, skill `verify`, regenerar el contexto de IA): el entorno habitual de skforecast. Los tests usan el objeto falso y no importan `h2ofm`, así que vale cualquier versión de Python que admita skforecast.
   - **Backend real** (`skeleton/run_checks.py`, los experimentos de la sección 12 y el notebook de la guía con la opción b del paso 5 de la sección 7; la opción a necesita además todos los backends de la guía): Python >= 3.12 con `h2ofm` y torch, skforecast instalado en modo editable desde la rama de trabajo y una cuenta de Hugging Face que haya aceptado las condiciones de los dos checkpoints. Si no existe, se crea como dice la sección 11 (conda o uv), con el nombre que se indique.

   Después, usar el entorno elegido en todos los comandos (`python`, `pytest`, `pip install`); los comandos de este documento escriben `python` y dan por hecho que está activado (`conda activate <entorno>` o `source ~/.venvs/<entorno>/bin/activate`; si la shell que ejecuta los comandos no lo hereda, llamar al intérprete por su ruta o con `conda run -n <entorno>`). Si skforecast está instalado en modo editable desde otra copia del repositorio, comprobar que se importa la rama de trabajo con `python -c "import skforecast; print(skforecast.__file__)"` ejecutado fuera de la raíz (por ejemplo, desde `dev/timeh2o_adapter/experiments/skeleton`, donde se lanza `run_checks.py`; desde la raíz, `python -c` importa siempre la copia local). Si no es la rama de trabajo, reinstalarla en modo editable desde su raíz o lanzar `run_checks.py` y los experimentos con `PYTHONPATH=<raíz de la rama>`. Los tests, con `python -m pytest` desde la raíz.
2. Leer la sección 6 y comprobar que todas las decisiones tienen el campo `Decisión:` relleno. Si falta alguna, preguntar antes de escribir código. D1 (licencia) condiciona todo lo demás.
3. Comprobar el estado del adapter de Timestral (si su PR ya está fusionado, reutilizar su helper del aviso de covariables planas o sustituir su método por el helper; decisión D15).
4. Repasar la checklist de upstream (sección 10): versión de `h2ofm`, licencia, gating, revisión de los pesos.
5. Comprobar que esta carpeta está en `origin/0.27.x` (`git ls-tree origin/0.27.x dev/timeh2o_adapter`): el paso 1 de la sección 7 parte de `skeleton/fake_timeh2o.py`, y el punto 10 de la sección 13 ejecuta `skeleton/run_checks.py --package`. Si no está, fusionar antes el PR de este documento o copiar la carpeta de la rama `docs/timeh2o-adapter-research`. Después, crear la rama `feature/timeh2o-adapter` desde `origin/0.27.x` y seguir el plan de la sección 7 en orden.

## 1. Resumen y recomendación

**Recomendación: implementarlo**, después del PR #1450 y condicionado a la decisión D1 sobre la licencia, con `context_length=16384` por defecto, el `LicenseWarning` y una documentación que deje claro el acceso gated, el requisito de Python >= 3.12 y que la licencia excluye el uso en producción.

Motivos:

- **Es, con Timestral-209M, el más preciso que hemos medido en la configuración de comparación de la guía de usuario** (`vic_electricity` horario con exog, diciembre de 2014), **y el mejor calibrado de los dos; con 16.384 de contexto, también el más rápido.** Con 16.384 de contexto, TimeH2O-v1 da un MAE de 84.24 y Timestral-209M, 84.32; TimeH2O cubre el 82,6 % con el intervalo del 80 % (Timestral, el 73,3 %) y tarda 114 s frente a 307 s **[probado]**. Con contextos más cortos, Timestral-209M tarda menos: con 4096, 68,8 s frente a 194,6 s (sección 2, Experimento); en el informe de Timestral, 12,8 s con 500 y 147,7 s con 8192, frente a 59,2 s y 166,5 s de TimeH2O. Con contexto corto la ventaja es clara: con 500, 103.06 frente a 135.67 de Timestral-209M y 137.49 de Chronos-2. Como referencia, Chronos-2 con 4096 da 134.05 y el mejor de la tabla de la guía es TS-ICL con 500 (140.88). Su cobertura del 80 % está entre 0.787 y 0.826 en todas las configuraciones con exog y el *ensemble*, frente a 0.67 a 0.78 de todos los demás modelos medidos aquí y en el informe de Timestral (el 0.67 es el checkpoint tiny). Las cifras de GIFT-Eval de la model card (por delante de Timestral-209M por poco) se reproducen con los resultados del leaderboard, pero los envía H2O; nuestro experimento es un solo dataset.
- **Es el adapter más delgado de todos.** Aparte del PR #1450, solo necesita un helper compartido: el aviso de covariables planas, que Timestral necesita igual (D15). `forecast_many` recibe una lista de series de longitudes distintas, cada una con sus covariables (también categóricas, codificadas por el backend), acepta cualquier nivel de cuantil y trocea los lotes por sí mismo. El adapter solo construye las filas de covariables (histórico seguido del futuro) y reordena las columnas de cuantiles; declara `requires_equal_length_context=False` porque el relleno de `FoundationModel` cambiaría la previsión. Una serie da la misma previsión en un lote heterogéneo que sola (diferencia máxima de 7,6e-6, `exp3.log`) **[probado]**.
- **Está en PyPI**, con el código de inferencia bajo Apache-2.0 y un wheel puro de Python.

Lo que pesa en contra (decisiones de la sección 6):

- **La licencia de los pesos** solo permite uso no comercial **y no productivo**: es la más restrictiva de las registradas en skforecast, revocable, y con una definición de derivado amplia (D1).
- **Acceso gated**: el primero en skforecast. El usuario necesita cuenta en Hugging Face, aceptar las condiciones y un token; el CI, una cuenta que las acepte (D3, sección 7).
- **Python >= 3.12** para `h2ofm`, mientras que skforecast admite 3.10.
- **Coste en CPU**: el *ensemble* de parches, activo por defecto, multiplica el tiempo hasta por 25 con historias cortas (100 series de 512 pasos: 93 s frente a 3,7 s sin él). Con 16.384 de contexto es una sola pasada (D6, D7).
- **Backend joven y sin repositorio público del código**: tres versiones en 15 días y un valor por defecto cambiado entre versiones menores. El código (Apache-2.0) solo se publica en PyPI, y los fallos se reportan en `h2oai/h2ofm`, el repositorio de TabH2O. No funciona en MPS.
- **Covariables futuras planas en el contexto**: como en Timestral (su issue #7), una covariable constante o casi constante en el contexto que cambia en el horizonte (un precio, un festivo o una promoción que no aparecen en el contexto) dispara los cuantiles altos, sin aviso del backend. En la guía pasa en un fold con 500 de contexto (el festivo de Navidad, que en el índice, en UTC, empieza el 24/12 a las 14:00). Propuesta: un aviso compartido con Timestral (D15).

Si la licencia no encaja con el uso que se espera de skforecast (D1), la alternativa es **esperar** a que H2O publique una licencia más abierta o el repositorio del código (la model card dice "GitHub (coming soon)"). El diseño seguiría valiendo tal cual.

## 2. Ficha del modelo

### Qué es

- Familia de foundation models de H2O.ai para previsión probabilística *zero-shot*, con covariables futuras en el checkpoint grande **[documentado]** (model card).
- Transformer de solo decodificador con atención temporal causal por parches y RoPE, y atención entre variables (las filas de una misma serie: objetivo y covariables) en una de cada cuatro capas (`variate_attention_every = 4`: 7 de las 28 capas del checkpoint grande) **[verificado en código]** (`_model/model.py`, `config.json`). La atención entre variables solo mezcla filas del mismo grupo (`group_mask = group_ids[:, None] == group_ids[None, :]`), es decir, de la misma serie **[verificado en código]**.
- **Parches de tamaño variable** (`variable_patch_sizes: true`): el checkpoint grande se entrenó con parches de 2, 4, 8 y 16 pasos y una ventana fija de 1.024 tokens (`context_length // patch_size`), es decir, 2.048, 4.096, 8.192 y 16.384 pasos de historia. Por defecto (`patch_ensemble=True`) cada serie se predice con todos los tamaños cuya ventana contiene su historia y se promedian las previsiones, una pasada por tamaño **[verificado en código]** (`_pipeline/pipeline.py`, `_fitting_patch_sizes`). Con 500 pasos de historia son cuatro pasadas; con 4.096, tres (4, 8 y 16); con 8.192, dos; con más de 8.192, una.
- Equivariante a la reflexión por construcción (`flip_equivariant: true`): la previsión de `-x` es la reflexión de la de `x`, sin pasada adicional **[verificado en código]** (`_model/flipeq.py`, `_api.py`, `_options`). Normalización causal por parches con transformación `asinh` y refinamiento iterativo de la escala en el horizonte (*iterative RevIN*) **[verificado en código]** (`_model/scaler.py`). Cada token de una covariable futura incorpora los 6 parches siguientes de esa covariable (`lookahead_patches: 6`) **[verificado en código]**.
- Dos checkpoints. Los pesos se cargan siempre en float32 (`build_model` convierte cada tensor a float32) **[verificado en código]**:

| Checkpoint | Parámetros | Pesos | Capas | `d_model` | Parches | Covariables | Revisión en HF (fijada) |
|:-----------|-----------:|------:|------:|----------:|:--------|:------------|:------------------------|
| `h2oai/timeh2o-v1` | 108.764.558 | 435 MB | 28 | 768 | 2, 4, 8, 16 | Futuras | `1310be782553943a4c520d2810a98ce8f56b2709` |
| `h2oai/timeh2o-v1-tiny` | 5.825.636 | 23 MB | 8 | 384 | 8 | No | `33e4aa27a3d8c971419c2c4a0cdab9c0a10a687e` |

  Parámetros contados en el modelo cargado y arquitectura de cada `config.json` **[probado]**. Tamaño y sha256 de `model.safetensors` (API del Hub): 435.111.800 bytes y `f9332b58bbf1b64e4fc91585b2163c32c9ceaf2a1434c18c67846e00c57e4e47` (v1); 23.316.064 bytes y `3516caa916d4f76e225ed815d5c9a464ee1b11226b2cefb8859e8220d043a6c6` (tiny) **[probado]**.
- Los repositorios de Hugging Face se crearon el 22/10/2026 (v1) y el 11/10/2026 (tiny), y se modificaron por última vez el 27/10/2026 **[documentado]** (API del Hub). El 05/11/2026 tenían 1.102 y 770 descargas **[probado]**.
- No hay informe técnico ni repositorio del modelo: la model card enlaza un blog y dice "GitHub (coming soon)" y "Paper (coming soon)" **[documentado]**. No he leído el blog (`h2o.ai/blog/2026/timeh2o-foundation-models`).
- Datos de entrenamiento: la model card enumera ocho fuentes públicas (`autogluon/chronos_datasets`, `Salesforce/GiftEvalPretrain`, `Salesforce/lotsa_data`, `hq-bench/quito-corpus`, `Dingdong-Inc/FreshRetailNet-LT`, Azure Functions 2019, Iowa Liquor Sales y BLS Average Prices) y afirma que se excluyeron los datasets que se solapan con fev-bench, GIFT-Eval y TIME **[documentado]**.
- `h2ofm` reconoce en `THIRD_PARTY.md` que el pipeline, la construcción de lotes y la decodificación autorregresiva están adaptados de `chronos-forecasting` 2.3.1, y que los métodos del modelo se apoyan en Toto 2.0, TiRex, TimesFM 3.0 (RevIN iterativo y *lookahead* de covariables futuras), Chronos-2 (atención entre variables) y FlexiViT (redimensionado de parches) **[documentado]**.

### Licencia

- **Pesos: "H2O MODEL NON-COMMERCIAL LICENCE v1.0"**, con el mismo `LICENSE` en los dos repositorios (12.391 bytes, sha256 `3209e42bba7aed041dd10a16521d06c2bc6c6d587e6e35aebdcbe526d8225a76`) **[probado]**. La tarjeta declara `license: other`, `license_name: h2o-model-non-commercial-licence-v1.0` y `license_link: LICENSE` **[probado]**. Puntos que afectan a skforecast **[documentado]** (texto completo leído):
  - Solo para "Non-Commercial Purpose" (cláusula 1.4): pruebas, evaluación, *benchmarking* e investigación sin fines comerciales, también la evaluación interna de una empresa, "provided the Outputs are not used in live business decisions, client deliverables, or paid products or services". Excluye expresamente cualquier actividad que genere ingresos, el uso en producción o en apoyo de un sistema en producción, y entrenar o destilar modelos para uso comercial. Es más restrictiva que las demás licencias registradas: no hay umbral de facturación (Timestral), y el texto solo menciona que un acuerdo escrito o una licencia comercial aparte de H2O la sustituirían (cláusula 12.2), sin condiciones ni contacto (TabPFN ofrece una licencia empresarial); prohíbe además el uso en producción aunque no sea comercial (Schedule 1, punto 1: "To deploy in production, or to make, inform or execute any live business decision").
  - "Derivative" (1.2) incluye cualquier obra que "incorporates, utilises, or is otherwise based on or derived from a Model"; las salidas (previsiones) no son derivados. La cláusula 5.1(b) y el punto 5 del Schedule 1 prohíben distribuir el modelo o un derivado a terceros "or to embed them in any product or service you provide to others". skforecast no distribuye los pesos: cada usuario los descarga del Hub tras aceptar la licencia con su propia cuenta (repositorio gated). Si un adapter que carga el modelo a petición del usuario es un "derivado" que skforecast "pone a disposición de terceros" es una cuestión jurídica que este documento no resuelve (decisión D1) **[inferido]**.
  - Publicación de resultados de evaluación (cláusula 4): permitida si se identifica el modelo y la versión evaluada, la fecha y la metodología, sin presentar los resultados de forma engañosa; H2O pide (sin exigirlo) que se le avise en legal@h2o.ai. Afecta a los resultados de este documento y de `experiments/`, que identifican la revisión de los pesos, la versión de `h2ofm`, la fecha y la metodología (sección 2), y a la tabla de comparación de la guía de usuario (decisión D1) **[documentado]**.
  - Licencia revocable "at any time on notice" (7.3), terminación automática por incumplimiento (7.2) y versiones nuevas de la licencia que solo rigen las copias obtenidas después (12.4) **[documentado]**.
  - Marcas (cláusula 11): se puede decir que se evaluó un modelo, usando su nombre solo para eso, sin sugerir respaldo de H2O **[documentado]**.
- **Código del backend (`h2ofm`): Apache-2.0** (`License-Expression: Apache-2.0` en PyPI y `LICENSE` Apache en el wheel) **[probado]**. La licencia de los pesos dice que "Model" incluye el "inference code" obtenido de un repositorio de H2O, pero el repositorio del Hub solo contiene `config.json`, `model.safetensors`, `README.md` y `LICENSE`; el código de inferencia se distribuye aparte, en PyPI, con su propia licencia Apache-2.0 **[inferido]**.
- La página `https://huggingface.co/h2oai/timeh2o-v1/blob/main/LICENSE` se ve sin autenticarse; la descarga de `config.json` sin autenticarse devuelve 401 **[probado]** (petición sin credenciales). Los metadatos del Hub (`/api/models/h2oai/timeh2o-v1/revision/main`) son públicos **[probado]**.

### Distribución

- **PyPI, paquete `h2ofm`** ("H2O.ai Foundation Models, standalone inference"; también incluye TabH2O). Tres versiones: 0.42.0 (13/10/2026), 0.43.0 (22/10/2026) y 0.43.1 (27/10/2026), sin versiones retiradas **[probado]**. Wheel puro de Python de 77 KB **[probado]**.
- Dependencias de 0.43.1: **Python >= 3.12**, `numpy>=1.26`, `pandas>=2.0`, `torch>=2.10,<3`, `safetensors>=0.6.2`, `huggingface-hub>=0.34`, `scikit-learn>=1.3` **[probado]** (metadatos de PyPI). skforecast admite Python >= 3.10: con Python 3.10 o 3.11, `pip install h2ofm` no encuentra ninguna versión **[inferido]**. `h2ofm` usa `warnings.warn(..., skip_file_prefixes=...)`, que solo existe desde Python 3.12 **[verificado en código]** (`timeh2o/_warnings.py`).
- **h2ofm 0.42.0 no carga `timeh2o-v1`**: `ValueError: Unsupported config.json fields: ['variable_patch_sizes', 'variate_attention_every']` (sí carga el tiny) **[probado]** (código de 0.42.0 del sdist, ejecutado con el intérprete de los experimentos). Hace falta `h2ofm>=0.43`.
- La API cambió entre 0.42.0 y 0.43.0: el valor por defecto de `flip_equivariant` pasó de `False` a `True` y aparecieron `snap_integer`, `patch_ensemble` y `Forecast.mean` **[verificado en código]** (diff de los sdists). Entre 0.43.0 y 0.43.1 no cambió nada en `h2ofm/timeh2o/` (solo TabH2O y `_validation.py`) **[verificado en código]**.
- **Sin repositorio público del código.** PyPI solo enlaza `https://h2oai.com/`, y la model card dice "GitHub (coming soon)" **[documentado]**. `github.com/h2oai/h2ofm` existe y es público (8 commits, el último el 22/09/2026, Apache-2.0), pero es el repositorio de acompañamiento de TabH2O (API alojada, *skill* para agentes, integraciones de Excel y Google Sheets, ejemplos) y no contiene el código de `h2ofm` **[probado]** (clon con git). Tiene 0 issues abiertos **[probado]** (listado por defecto, 05/11/2026) y 8 pull requests (refs `refs/pull/1` a `9`, sin la 4, que será un issue cerrado o borrado) **[probado]**. Su `CONTRIBUTING.md` pide abrir allí también los fallos del modelo ("we'll triage internally") **[documentado]**. Lo he leído con un clon de git y en la web de GitHub.
- **Pesos en el Hub de Hugging Face, gated (`gated: auto`)**: hay que aceptar las condiciones con una cuenta (la aprobación es automática) y autenticarse con un token **[probado]**. `from_pretrained` descarga solo `config.json` y `model.safetensors` con `snapshot_download` a la caché estándar del Hub; también acepta un directorio local **[verificado en código]** (`h2ofm/_checkpoints.py`). Con los pesos en caché, `HF_HUB_OFFLINE=1` funciona (todos los experimentos se ejecutaron así, salvo la primera ejecución de `exp2`, que se lanzó con conexión y descargó los pesos de Timestral-209M) **[probado]**.
- Sin token o sin haber aceptado las condiciones: `h2ofm` vuelve a lanzar `GatedRepoError` tal cual y convierte `RepositoryNotFoundError` en `FileNotFoundError("'h2oai/timeh2o-v1' is neither a local checkpoint nor an accessible Hugging Face repo")` **[verificado en código]**. `huggingface_hub` lanza `GatedRepoError` cuando el Hub marca el error como `GatedRepo` y `RepositoryNotFoundError` con un 401 sin esa marca **[verificado en código]** (`huggingface_hub/utils/_http.py`). Qué recibe exactamente un usuario sin token no se ha probado: los experimentos se ejecutaron con una cuenta autenticada (comprobarlo en una máquina sin token, decisión D3) **[inferido]**.
- No hace falta cuenta de H2O ni API externa: el modelo corre en local. H2O ofrece una API alojada con otras condiciones (TabH2O), que el adapter no usa **[documentado]**.

### API de inferencia

```python
from h2ofm import TimeH2O

model = TimeH2O.from_pretrained(
    pretrained_model_name_or_path,  # repo del Hub ("owner/name") o directorio local
    *,
    device="auto",                  # "auto" (CUDA si existe, si no CPU), "cpu", "cuda", "cuda:N"
    revision=None, token=None, cache_dir=None, local_files_only=False,
    batch_size=256,                 # filas (objetivos y covariables) por pasada; no cambia la previsión
)
forecasts = model.forecast_many(
    contexts,                       # lista de series 1-D (o 2-D multiobjetivo), longitudes distintas
    horizon,                        # int >= 1
    *,
    past_covariates=None,           # una entrada por serie; los checkpoints v1 las ignoran (aviso)
    future_covariates=None,         # una entrada por serie: None o filas de longitud historia + horizonte
    quantile_levels=(0.1, ..., 0.9),# niveles distintos en (0, 1), en el orden pedido
    flip_equivariant=True,          # sin efecto en los dos checkpoints (equivariantes por construcción)
    clip_nonnegative=False,         # recorta a 0 si la historia es no negativa
    snap_integer=False,             # redondea si la historia es entera
    context_length=None,            # acorta la ventana del checkpoint (16.384)
    autoregressive=None,            # None: el valor del checkpoint (False: una sola pasada)
    patch_ensemble=True,            # promedia los tamaños de parche que caben en la historia
)
forecasts[i].point            # (horizon,) float32: la mediana (también .median)
forecasts[i].quantiles        # (horizon, len(quantile_levels)) float32, NumPy en CPU
forecasts[i].quantile_levels  # tuple, en el orden pedido
forecasts[i].mean             # (horizon,): media aproximada integrando los 9 cuantiles entrenados
forecasts[i].quantile(0.9)    # (horizon,); KeyError si el nivel no se pidió
```

Fuente: `h2ofm/timeh2o/_api.py` y `h2ofm/_checkpoints.py` **[verificado en código]**; formas y tipos comprobados ejecutándolos (`exp1.log`) **[probado]**. También existen `forecast` (una serie) y `forecast_df` (DataFrame largo, con el horizonte como entero o como DataFrame de covariables futuras); el adapter usa `forecast_many`, que recibe una lista y no exige DataFrames ni fechas. Notas que afectan al adapter:

- `forecast_many` va decorado con `@torch.no_grad()` y el modelo está en `eval()`. Lo único que guarda entre llamadas es la bandera de un aviso (`_warned_covariates`) **[verificado en código]**.
- La validación usa excepciones explícitas (`ValueError`, `TypeError`), no `assert`, en todo el camino de `_api.py` **[verificado en código]**. Las funciones internas de interpolación de cuantiles sí usan `assert`, pero sus entradas ya están validadas.
- Todo se convierte a float32 (`_real_tensor`). Una entrada float64 da exactamente lo mismo que la misma en float32, dos llamadas iguales dan exactamente lo mismo en CPU, y la salida es float32 (`exp5.log`) **[probado]**.

### Entrada

- Una lista de series 1-D de **longitudes distintas**. `forecast_many` agrupa las series en lotes de unas `batch_size` filas (objetivos más covariables; añade series enteras hasta alcanzar el umbral, así que un lote puede pasarlo en las filas de una serie, y el tiny solo cuenta los objetivos) y rellena cada lote con NaN por la izquierda hasta la serie más larga del lote (`_data/dataset.py`, `iter_batches` y `_left_pad_and_cat`) **[verificado en código]**. Los parches sin observaciones no se atienden como claves y las posiciones de RoPE empiezan en el primer parche observado, así que el relleno no cambia la previsión: una serie dentro de un lote de longitudes 300, 3.000 y 9.000 da lo mismo que sola (diferencia máxima de 1,5e-5 en v1 y de 2,3e-5 en tiny, `exp1.log`) **[probado]**.
- **El relleno hecho por quien llama sí cambia la previsión.** El conjunto de tamaños de parche del *ensemble* depende de la longitud de la historia, NaN incluidos: una serie de 300 puntos rellenada con NaN hasta 9.000 pasa de cuatro tamaños (2, 4, 8 y 16) a uno (16). En v1, la previsión de las series de 300 y 3.000 puntos rellenadas a 9.000 difiere hasta 0,25 (un 0,5 % del nivel) de la de cada serie sin rellenar; en tiny, que tiene un solo tamaño de parche, no cambia (`exp1.log`) **[probado]**. Por eso el adapter debe declarar `requires_equal_length_context = False` (sección 4).
- NaN marca un valor ausente. Un objetivo entero a NaN se sustituye por ceros (*cold start*) **[verificado en código]** (`_inference_slice`). Un contexto entero a NaN da una mediana de exactamente 0 y los demás cuantiles prácticamente 0 (como mucho 2,2e-11 en v1) (`exp5.log` y una comprobación aparte) **[probado]**. En skforecast puede llegar al backend: `FoundationModel.predict(context=...)` rechaza un contexto entero a NaN ("All values of series 'y' are NaN"), pero `FoundationModel.fit` lo comprueba antes de recortar a `context_length`, así que una serie con sus últimos `context_length` valores a NaN pasa y la previsión es 0 sin avisar (con `context_length=500` y los últimos 600 valores a NaN, `exp5.log`) **[probado]**. El backtesting descarta esas ventanas (`_backtesting_foundation`, informe de Timestral). Con 200 NaN dentro de la serie la previsión es finita y cambia poco (0,050 sobre un nivel de unos 60), y 800 NaN al principio dan lo mismo que la serie sin ellos (1,5e-5), porque el conjunto de tamaños de parche no cambia (`exp5.log`) **[probado]**.
- Un infinito en la serie lanza `ValueError("Series may contain NaN for missing observations, but not infinity")` **[verificado en código]**. `FoundationModel` no los filtra: a través de él llega el mismo error (`exp4.log`) **[probado]**.
- Contexto máximo: 16.384 pasos (`context_length` de los dos `config.json`). Con una historia más larga, cada tamaño de parche usa solo los últimos `1.024 x p` pasos, sin avisar; la opción `context_length` de `forecast_many` por encima del máximo avisa y se reduce al máximo **[verificado en código]**. Comprobado: una serie de 20.000 puntos da exactamente lo mismo que sus últimos 16.384, sin aviso, y la opción `context_length=20000` avisa ("The specified context_length 20000 is greater than the model's maximum context length 16384. Resetting context_length to 16384.") (`exp5.log`) **[probado]**.
- No hay límite de horizonte. Con `autoregressive=False` (el valor de los dos checkpoints) se predice todo el horizonte en una pasada de `ceil(horizon / p)` parches de salida; con `True`, en bloques de `max_horizon // patch_size` parches de salida (en v1, 64 parches de `p` pasos: 128, 256, 512 o 1.024 pasos para p = 2, 4, 8 y 16; en el tiny, 128 parches de 8 pasos), realimentando una trayectoria por cuantil **[verificado en código]** (`_pipeline/decoding.py`). Con 4096 de contexto, horizontes de 1.024, 2.048 y 4.096 pasos dan previsiones finitas en una pasada, en 3,7 s, 4,6 s y 7,2 s; con `autoregressive=True`, 2.048 pasos tardan 138,7 s (30 veces más) y la previsión cambia (hasta 0,60 en los primeros 1.024 pasos y 3,6 después, sobre un nivel de unos 90), porque con parches pequeños el primer bloque autorregresivo cubre menos pasos (`exp5.log`) **[probado]**. El adapter deja el valor del checkpoint (una pasada).
- No recibe frecuencia ni fechas en la API de arrays **[verificado en código]**.

### Salida

- **Cualquier nivel de cuantil en (0, 1).** El modelo produce 9 cuantiles entrenados (0.1 a 0.9, paso 0.1), ordenados para que no se crucen (`preds.sort(dim=1)`). Los niveles intermedios se interpolan linealmente entre los entrenados; los de fuera de [0.1, 0.9] se extrapolan con una cola gaussiana anclada en la mediana y en el cuantil extremo: `q(τ) = q(0.5) + (q(borde) - q(0.5)) · Φ⁻¹(τ) / Φ⁻¹(borde)` **[verificado en código]** (`_pipeline/utils.py`, `extrapolate_quantiles`). Pedir el 0.5 devuelve exactamente la mediana del modelo (`exp1.log`) **[probado]**.
- Los niveles deben ser distintos y estar estrictamente entre 0 y 1: un nivel repetido, 0 o 1 lanzan `ValueError` **[verificado en código]**. Comprobado: `forecast_many` lanza "quantile_levels must not contain duplicates" y "quantile_levels must be non-empty and strictly between 0 and 1"; a través de `FoundationModel`, con el prototipo, niveles como 0.02, 0.33 y 0.98 salen en el orden pedido y sin cruzarse, y `predict_interval` funciona con `[0.1, 0.9]`, `0.9`, `0.98` y `[0.01, 0.99]` (`exp3.log`) **[probado]**. `FoundationModel` construye las columnas en un diccionario, así que con niveles repetidos devuelve una sola columna por nivel (`[0.5, 0.1, 0.5]` da `q_0.5` y `q_0.1`) **[probado]**.
- `point` es la mediana; `mean` es una media aproximada que integra los 9 cuantiles entrenados con colas constantes **[verificado en código]**. La salida se recorta a `[mín. del contexto − 1e4·escala, máx. + 1e4·escala]` **[verificado en código]**.
- Arrays de NumPy en CPU, float32, con la forma `(horizon, n_niveles)` para una serie 1-D **[probado]**.
- `clip_nonnegative` es `False` por defecto (en Timestral, `True`): con conteos Poisson(0.7), el cuantil 0.1 baja hasta -0,023; con `clip_nonnegative=True` se queda en 0, y `snap_integer=True` redondea a enteros (`exp5.log`) **[probado]**. La media aproximada (`mean`) difiere de la mediana en 0,014 en una serie horaria simétrica (`exp5.log`).

### Covariables

- **Solo futuras.** Los dos checkpoints v1 no usan covariables pasadas: `forecast_many` las valida y las ignora con un `UserWarning` ("This checkpoint does not use past covariates; past_covariates are validated and ignored. Pass covariates known through the horizon as future covariates instead.") **[verificado en código]** **[documentado]** (tabla de checkpoints del README). La previsión es idéntica con y sin la covariable pasada, y a través de `FoundationModel` con el prototipo una columna solo pasada da `IgnoredArgumentWarning` y la misma previsión que sin ella (`exp4.log`) **[probado]**.
- `future_covariates`: una entrada por serie, `None` o una secuencia de filas de longitud `historia + horizonte` (por ejemplo, una lista de `pandas.Series`) **[verificado en código]**.
- **Tipos.** Una fila con dtype numérico que no sea booleano es numérica (float32, NaN como ausente). Cualquier otra (booleana, texto, `object`, `category`) son etiquetas: el backend las codifica con una *target encoding* suavizada contra la historia del objetivo de esa misma serie (media del objetivo por categoría con `smooth=1`, con la media de la serie como prior); una categoría que no aparece en la historia recibe la media de la serie y `None` o NaN son una categoría propia ("missing") **[verificado en código]** (`_api.py`, `_covariate_row`; `_data/encoding.py`). La codificación de una serie no depende de las demás series del lote **[verificado en código]**. **Las booleanas son categóricas**, no numéricas (README: "Rows of any other dtype (strings, objects, booleans, `pd.Categorical`) are categorical") **[documentado]**. En un caso sintético con tres etiquetas que mueven el nivel (-5, 0 y +5), las etiquetas como texto bajan el MAE de 3.09 (sin covariable) a 0.43; como códigos enteros (numéricos), solo a 1.35; `category` da exactamente lo mismo que texto; una etiqueta no vista en la historia, en todo el horizonte, da 3.11 (como sin covariable), y una etiqueta `None` no da error. Un indicador booleano que sube el nivel da la misma previsión que el mismo indicador como float 0/1, salvo redondeo (7,6e-6: la parte 2 de `exp4_covariates.py` con más decimales; `exp4.log` lo imprime con tres, 0.000) **[probado]**: con las dos etiquetas en la historia, la codificación es una transformación afín creciente de 0/1, que el escalado elimina **[inferido]**.
- Un infinito en una covariable numérica lanza `ValueError("future_covariates may contain NaN, but not infinity")` **[verificado en código]**.
- **Una covariable numérica entera a NaN en la historia se descarta**, aunque tenga valores en el horizonte (`_inference_slice`: `missing = history.isnan().all(dim=-1)`); una de etiquetas con toda la historia ausente se codifica antes como una fila constante (la media de la serie), que no aporta nada pero no se descarta **[verificado en código]**. Comprobado con una numérica: la previsión es idéntica a la de sin covariable; con 100 NaN en la historia y 3 en el horizonte no hay errores (`exp4.log`) **[probado]**.
- **Covariables distintas en cada serie de un mismo lote**: cada serie lleva sus propias filas y la atención entre variables solo mezcla filas de la misma serie **[verificado en código]**. Comprobado a través de `FoundationModel` con el prototipo (`supports_heterogeneous_covariates=True`): tres series de 900, 400 y 700 puntos, que terminan en fechas distintas, una con 30 NaN, con exog distintas (numérica y de etiquetas, solo numérica, ninguna), van en una sola llamada y cada una difiere como mucho 7,6e-6 de la misma serie predicha sola; agrupándolas por columnas (tres llamadas) da lo mismo (`exp3.log`) **[probado]**.
- Con covariables futuras, la previsión depende del horizonte: cada token de una covariable futura lee los 6 parches siguientes de esa covariable, y los que caen más allá del horizonte se rellenan con NaN **[verificado en código]** (`PatchEmbed._ahead`, `_predict_step`). Comprobado: sin covariables, los primeros 24 pasos con `horizon=24` y con `horizon=100` coinciden (1,5e-5); con una covariable futura difieren desde el primer paso (hasta 0,38 sobre un nivel de 77) (`exp5.log`) **[probado]**. Es así por diseño, como en Timestral, pero dos backtestings con distinto `steps` no son comparables paso a paso.
- El checkpoint tiny no tiene atención entre variables: ignora las covariables futuras con un `UserWarning` ("This checkpoint does not use future covariates; they are validated and ignored."), solo la primera vez para cada modelo cargado (`_warned_covariates`) **[verificado en código]**. Comprobado: la previsión es idéntica con y sin covariable, y la segunda llamada ya no avisa (`exp4.log`) **[probado]**.
- Modo multivariante (`with h2ofm.timeh2o._api.multivariate():`): los objetivos de una serie multiobjetivo se ven entre sí. Es el equivalente de `cross_learning` y queda fuera de la primera versión del adapter **[verificado en código]**.
- **Covariables futuras planas en el contexto (el caso del issue #7 de Timestral).** TimeH2O tiene el mismo problema, con menos intensidad **[probado]** (`exp4.log`; los mismos datos que el informe de Timestral: 32 series diarias Poisson(2), 1.024 de contexto, 28 pasos, otra semilla):
  - Un precio de 2.50 en todo el contexto que pasa a 2.56 en el horizonte dispara los cuantiles altos: el 0.9 llega a 181.73 y el 0.99 a 326.79 (28 de 32 series por encima de 100), frente a 4.29 y 6.07 sin covariable; la mediana sube de 2.09 a 4.76. Un indicador 0/1 numérico que vale 0 en todo el contexto y 1 en dos pasos del horizonte, también: 0.9 hasta 78.70, mediana hasta 6.05.
  - No pasa si el precio ya se mueve en el contexto (0.9 hasta 4.44), ni con el indicador como booleana (categórica en `h2ofm`): la categoría `True` no aparece en la historia, recibe la media de la serie igual que `False` y la fila codificada queda constante, así que no se dispara, pero tampoco usa el evento (0.9 hasta 4.03). Poner la historia a NaN lo evita porque el backend descarta la fila: es lo mismo que no pasar la covariable, y a diferencia de Timestral no conserva la información futura.
  - **Mecanismo** (`scaler.py`, `model.py`) **[verificado en código]** **[probado]**: cada parche se normaliza con momentos causales acumulados que incluyen el propio parche, y los valores futuros de la covariable forman parte de la secuencia. Con un contexto constante, la escala se recorta a 1e-6 y el valor normalizado del contexto es 0; el primer parche del horizonte, sea cual sea el cambio, queda a unas `sqrt(n_contexto / p)` escalas (8 con p=16 y 22,6 con p=2 para 1.024 pasos de contexto, igual con un escalón de 1e-4, de 0.06 o de 10; comprobación aparte). Por eso el tamaño del cambio no importa. Lo confirma `exp7.log` **[probado]**: un ruido de 1e-6 en el contexto lo empeora (0.9 hasta 393.22), un escalón de solo 1e-4 en el horizonte dispara igual que uno de 0.06 (182.92 frente a 181.73), expresar el precio relativo a su último valor no cambia nada (181.73) y `patch_ensemble=False` lo empeora (664.56). El caso de `exp7` con un escalón de 1e-7 al final del contexto no prueba nada: en float32, 2,5000001 es 2,5.
  - **En la guía de usuario** (500 de contexto), `Holiday` vale 0 en todo el contexto y 1 en el horizonte en un fold (el que empieza el 24/12/2014 00:00; el festivo de Navidad empieza a las 14:00 en el índice, que va en UTC) (`exp7.log`) **[probado]**. Con `Holiday` como float, la mediana aprovecha el festivo (MAE 228.94 frente a 275.77 solo con `Temperature` y 309.08 sin exog), pero el intervalo del 80 % se triplica (anchura media 1365.3 frente a 479.2) y el cuantil 0.9 llega a 6512.9 con una demanda máxima de 4496.4; como booleana (categórica), el festivo no visto se ignora (271.90, anchura 456.3). Con 4096 de contexto no hay ningún fold así.
  - **¿Cuánto tiene que variar la covariable en el contexto para que no pase?** (`exp8.log`, `exp9.log`) **[probado]**. Con el mismo precio y un ruido de desviación `s` en el contexto, el cuantil 0.9 máximo es 253.49 con un cambio de 600 desviaciones (`s` = 1e-4), 264.21 con 60, 39.03 con 6, 6.51 con 2 y 4.15 con 0,6 (4.29 sin covariable). Un indicador con 1, 4, 16 y 64 unos en el contexto (unas 32, 16, 8 y 4 desviaciones de distancia) da medianas de 6.77, 2.65, 2.23 y 2.10 (2.09 sin covariable) y cuantiles 0.9 de 16.92, 5.72, 4.71 y 4.29. En la guía, `Holiday` se separa del contexto entre 6 y 13 desviaciones en los tres folds de Navidad con 4096 y con 16.384 de contexto, y ahí no se dispara nada: anchuras normales, y `Holiday` mejora la mediana en 5 de esos 6 casos (por ejemplo, con 4096 en el fold del 24/12, MAE 224.13 frente a 262.02 solo con `Temperature`; empeora en el del 26/12 con 4096, 74.44 frente a 44.27).
  - **Umbral del aviso** (decisión D15): ninguno separa del todo los casos dañinos de los inocuos (un precio con ruido y un cambio de 6 desviaciones multiplica por 9 el cuantil 0.9, y un festivo a 13 desviaciones no hace nada). Con 20 desviaciones se avisa en los casos medidos de 32 desviaciones o más (contexto plano, casi plano y un indicador visto una sola vez) y en ninguno de la guía con contexto largo; no se avisa en el precio de 6 desviaciones ni en el indicador de 16 (que sube algo la mediana).

### Hardware

- CPU y CUDA. `device="auto"` elige CUDA si está disponible y, si no, CPU; nunca MPS **[verificado en código]** (`h2ofm/_checkpoints.py`, `resolve_device`). Comprobado en el entorno de los experimentos, sin GPU: `"auto"` carga en CPU, `"cuda"` lanza `ValueError("CUDA was requested but is not available; use device='cpu'")` y `"mps"` lanza `ValueError("device must be 'auto', 'cpu', 'cuda' or 'cuda:N'")` también en Linux (`exp5.log`) **[probado]**.
- **No funciona en MPS** (Apple Silicon): `resolve_device` rechaza cualquier dispositivo que no sea CPU o CUDA (`ValueError("device must be 'auto', 'cpu', 'cuda' or 'cuda:N'")`), y además el escalado y la decodificación usan float64 (`.double()` en `scaler.py` y `model.py`), que MPS no admite **[verificado en código]**. No está probado en un Mac (Apple Silicon).
- En CUDA con capacidad de cómputo >= 8.0 y Triton instalado, el refinamiento de la escala usa un kernel de Triton (`_model/_refinement.py`) **[verificado en código]**. No está probado en GPU.
- Siempre float32: no hay opción de dtype ("All inference uses float32 weights") **[documentado]** **[verificado en código]**.
- `batch_size` (por defecto 256) fija el tamaño aproximado de cada pasada, en filas (sección 2, Entrada): cambia el tiempo y la memoria, no la previsión **[documentado]**. Comprobado: 40 series de 512 pasos dan exactamente lo mismo con `batch_size=8` y con 256 (1,57 s frente a 1,18 s, `exp5.log`) **[probado]**.
- Los avisos de `h2ofm` usan `skip_file_prefixes` para apuntar al primer marco fuera de `h2ofm` y de `torch` **[verificado en código]**: cuando los dispare el adapter, apuntarán a la línea de `_adapters.py` que llama a `forecast_many`, no a la llamada del usuario **[inferido]**.

### Resultados publicados

Valores de la model card, evaluación zero-shot **[documentado]**. Los publica H2O; las columnas de los demás modelos también vienen de la tarjeta.

| Benchmark | Métrica | TimeH2O-v1 | Timestral-209M | t0-gamma | TimesFM-3 | Chronos-2 |
|:----------|:--------|-----------:|---------------:|---------:|----------:|----------:|
| GIFT-Eval | CRPS (menor es mejor) | **0.441** | 0.446 | 0.452 | 0.456 | 0.485 |
| GIFT-Eval | MASE (menor es mejor) | **0.654** | 0.658 | 0.661 | 0.667 | 0.698 |
| fev-bench | SQL skill (mayor es mejor) | **50.83** | 50.39 | 49.42 | 48.99 | 47.28 |
| fev-bench | MASE skill (mayor es mejor) | **39.95** | 39.49 | 37.99 | 37.82 | 35.50 |

- **GIFT-Eval, comprobado** **[probado]**: recalculando la media geométrica del MASE y del CRPS (`mean_weighted_sum_quantile_loss`) relativos a Seasonal Naive con los resultados por dataset del leaderboard (`Salesforce/GIFT-Eval`, `results/<modelo>/all_results.csv`, 97 configuraciones), salen exactamente las cifras de la tabla para las cinco columnas: TimeH2O-v1 0.654 y 0.441, Timestral-209M 0.658 y 0.446, t0-gamma 0.661 y 0.452, TimesFM-3 0.667 y 0.456, Chronos-2 0.698 y 0.485. Son resultados enviados por cada proveedor (el `config.json` de TimeH2O-v1 declara `testdata_leakage: No` y enlaza un notebook de réplica en el repositorio de GIFT-Eval), no una evaluación independiente.
- **fev-bench, sin comprobar**: los datos del leaderboard (`autogluon/fev-bench`) no están entre los archivos del Space.
- TimeH2O-v1-tiny: la tarjeta da GIFT-Eval "univariate" (MASE 0.674, CRPS 0.489) y fev-bench "univariate" (SQL 37.87, MASE 29.05) **[documentado]**; con las 97 configuraciones completas del leaderboard salen MASE 0.687 y CRPS 0.468 **[probado]** (otro subconjunto, no comparable con la tarjeta).
- Ninguna evaluación independiente encontrada.

### Experimento en skforecast [probado]

Entorno aparte (`~/.venvs/timeh2o-exp`, fuera del repositorio): Python 3.12.3, torch 2.14.1+cpu (4 hilos), numpy 2.5.4, pandas 2.3.3, h2ofm 0.43.1, huggingface-hub 2.4.0 y skforecast 0.27.0 (editable, rama `0.27.x`); como referencia, chronos-forecasting 2.3.2 y timestral 1.0.0 (`a534bdb`). Hardware: 4 CPU Intel Xeon, 15 GB de RAM, sin GPU. Versiones exactas y cómo recrearlo: sección 11.

**Backtesting** (`exp2_backtest.py`, `results_exp2.csv`, `exp2.log`). Misma configuración que la tabla de comparación de `docs/user_guides/foundation-forecasting-models.ipynb`:

- `vic_electricity` horario (`resample(rule="h", closed="left", label="right")`), `Demand` con `Temperature` y `Holiday` como exog (las dos float, como en la guía).
- Datos de 2012-01-01 00:00 a 2014-12-30 23:00, `end_train = "2014-11-30 23:59:00"`: test del 01/12/2014 00:00 al 30/12/2014 23:00 (30 folds de 24 pasos, 720 predicciones), `refit=False`.
- `backtesting_foundation` con el prototipo registrado en tiempo de ejecución. Se mide el MAE de la mediana, la cobertura empírica de los intervalos `[0.1, 0.9]`, `[0.05, 0.95]` y `[0.01, 0.99]` (nominal 80 %, 90 % y 98 %), la anchura media del intervalo del 80 % y el tiempo del backtesting sin contar la carga de los pesos.
- Chronos-2 y Timestral-209M se han medido otra vez en el mismo entorno, con el mismo script: Chronos-2-small con 500 da 171.27 (el valor del notebook publicado), Chronos-2 con 4096 da 134.05 y Timestral-209M con 4096 da 96.89, exactamente las cifras de `dev/timestral_adapter/experiments/results_exp2.csv`. Los tiempos difieren algo (2,2 s, 23,0 s y 68,8 s frente a 1,4 s, 19,1 s y 73,0 s).

| Modelo | `context_length` | Exog | `patch_ensemble` | MAE | Cob. 80 % | Cob. 90 % | Cob. 98 % | Anchura 80 % | Tiempo (s) |
|:-------|-----:|:----:|:----:|----:|----:|----:|----:|----:|----:|
| Chronos-2-small | 500 | Sí | | 171.27 | 0.731 | 0.847 | 0.967 | 462.5 | 2.2 |
| Chronos-2 (120M) | 4096 | Sí | | 134.05 | 0.721 | 0.829 | 0.942 | 449.9 | 23.0 |
| Timestral-209M | 4096 | Sí | | 96.89 | 0.718 | 0.836 | 0.976 | 280.8 | 68.8 |
| Timestral-209M | 16384 | Sí | | **84.32** | 0.733 | 0.876 | 0.988 | 255.7 | 307.2 |
| TimeH2O-v1 | 500 | Sí | Sí | 103.06 | 0.818 | 0.908 | 0.968 | 398.8 | 59.2 |
| TimeH2O-v1 | 2048 | Sí | Sí | 100.30 | 0.787 | 0.882 | 0.957 | 307.6 | 218.9 |
| TimeH2O-v1 | 4096 | Sí | Sí | 98.30 | 0.810 | 0.893 | 0.969 | 312.9 | 194.6 |
| TimeH2O-v1 | 8192 | Sí | Sí | 103.84 | 0.794 | 0.900 | 0.964 | 330.9 | 166.5 |
| TimeH2O-v1 | 16384 | Sí | Sí | **84.24** | 0.826 | 0.919 | 0.983 | 317.5 | 114.0 |
| TimeH2O-v1 | 500 | Sí | No | 112.35 | 0.787 | 0.865 | 0.940 | 395.5 | 6.5 |
| TimeH2O-v1 | 4096 | Sí | No | 106.20 | 0.760 | 0.867 | 0.964 | 318.5 | 28.5 |
| TimeH2O-v1 | 8192 | Sí | No | 108.75 | 0.754 | 0.860 | 0.958 | 329.5 | 53.2 |
| TimeH2O-v1 | 500 | No | Sí | 155.57 | 0.717 | 0.826 | 0.924 | 436.4 | 27.9 |
| TimeH2O-v1 | 4096 | No | Sí | 150.27 | 0.761 | 0.849 | 0.928 | 443.2 | 75.4 |
| TimeH2O-v1 | 16384 | No | Sí | 142.24 | 0.811 | 0.886 | 0.958 | 486.8 | 51.5 |
| TimeH2O-v1 | 4096 | `Holiday` bool | Sí | 98.35 | 0.804 | 0.893 | 0.969 | 309.1 | 202.5 |
| TimeH2O-v1 | 16384 | `Holiday` bool | Sí | 87.59 | 0.804 | 0.907 | 0.979 | 311.2 | 136.9 |
| TimeH2O-v1-tiny | 500 | No | | 167.41 | 0.672 | 0.778 | 0.900 | 439.9 | 0.9 |
| TimeH2O-v1-tiny | 4096 | No | | 162.26 | 0.704 | 0.786 | 0.899 | 436.3 | 2.7 |
| TimeH2O-v1-tiny | 4096 | Sí (ignorada) | | 162.26 | 0.704 | 0.786 | 0.899 | 436.3 | 2.1 |
| TimeH2O-v1-tiny | 16384 | No | | 143.88 | 0.768 | 0.833 | 0.929 | 432.0 | 12.9 |

- **Con contexto largo, empate con Timestral-209M; con contexto corto, mucho mejor.** Con 16.384 de contexto, TimeH2O-v1 da un MAE de 84.24 y Timestral-209M, 84.32: la misma precisión, pero TimeH2O cubre mejor (0.826 frente a 0.733 en el intervalo del 80 %) y tarda 2,7 veces menos (114.0 s frente a 307.2 s). Con 500 de contexto, TimeH2O da 103.06, frente a 135.67 de Timestral-209M y 137.49 de Chronos-2 (120M) (`dev/timestral_adapter/experiments/results_exp2.csv`): un 24 % menos. Con 4096, 98.30 frente a 96.89 de Timestral-209M y 134.05 de Chronos-2. Como referencia, el mejor valor de la tabla de comparación del notebook publicado es el de TS-ICL con 500 (140.88).
- **El MAE no baja de forma monótona con el contexto:** 103.06 (500), 100.30 (2048), 98.30 (4096), 103.84 (8192) y 84.24 (16384). Con 16.384 horas, la ventana del primer fold empieza el 17/01/2013 e incluye todo diciembre de 2013; con 8.192, empieza el 24/12/2013. Los dos modelos mejoran mucho al pasar a 16.384 (TimeH2O un 14 % frente a 4096, Timestral un 13 %), y también el tiny (de 162.26 a 143.88) y TimeH2O sin exog (de 150.27 a 142.24): la mejora viene, probablemente, de ver el diciembre del año anterior (estacionalidad anual) **[inferido]**, y en otro dataset puede no repetirse.
- **El *ensemble* de parches mejora la precisión y cuesta mucho en CPU.** Sin él, el MAE sube un 9 % con 500 (112.35 frente a 103.06), un 8 % con 4096 (106.20 frente a 98.30) y un 5 % con 8192 (108.75 frente a 103.84), pero el backtesting es 9, 7 y 3 veces más rápido (6.5 s, 28.5 s y 53.2 s frente a 59.2 s, 194.6 s y 166.5 s). Con 16.384 solo cabe el parche de 16 pasos y es una sola pasada: con el *ensemble*, es el más barato de los contextos de 2.048 o más (114.0 s frente a 218.9, 194.6 y 166.5 s); 500 cuesta 59.2 s.
- **Las exog ayudan mucho:** sin ellas, el MAE sube a 155.57 (500), 150.27 (4096) y 142.24 (16384), entre un 51 % y un 69 % más. Con dos covariables, el backtesting tarda entre 2,1 y 2,6 veces más (59.2 s frente a 27.9 s con 500; 194.6 s frente a 75.4 s con 4096; 114.0 s frente a 51.5 s con 16384).
- **`Holiday` como booleana** (categórica para `h2ofm`) da lo mismo que como float con 4096 (98.35 frente a 98.30) y algo peor con 16384 (87.59 frente a 84.24, un 4 % más).
- **Cobertura cercana a la nominal.** Con exog y el *ensemble*, la cobertura del intervalo del 80 % va de 0.787 a 0.826, la del 90 % de 0.882 a 0.919 y la del 98 % de 0.957 a 0.983. Chronos-2 y Timestral-209M se quedan entre 0.72 y 0.73 para el 80 %, como todos los modelos del informe de Timestral (0.68 a 0.78). Los intervalos del 90 % y del 98 % de TimeH2O salen de la cola gaussiana extrapolada (fuera de [0.1, 0.9]) y cubren cerca de lo nominal. Sin el *ensemble*, la cobertura del 80 % baja (0.754 a 0.787).
- **El tiny** es mucho peor (167.41 con 500, 162.26 con 4096, 143.88 con 16384) y casi instantáneo (de 0,9 s a 12,9 s). Ignora las exog: con y sin ellas da exactamente lo mismo.
- Es **un solo dataset**, horario y con estacionalidad marcada. Indica la tendencia, pero no sustituye a un benchmark.

**Lote heterogéneo y cuantiles** (`exp3_multiseries.py`, `exp3.log`): ver sección 2, Covariables y Salida. Resumen: una sola llamada al backend para tres series de longitudes, fechas finales y exog distintas, con una diferencia máxima de 7,6e-6 frente a cada serie sola; niveles fuera de la rejilla en el orden pedido; 0 y 1 rechazados por el backend.

**Coste de las covariables** (`exp3_multiseries.py`, `exp3.log`; a través de `FoundationModel`, 512 pasos de contexto, 24 pasos; una covariable numérica, o una numérica y otra de etiquetas):

| Checkpoint | `patch_ensemble` | Series | Sin covariables | 1 covariable | 2 covariables |
|:-----------|:----:|-----:|-----:|-----:|-----:|
| v1 | `True` | 20 | 10,22 s | 25,20 s | 42,93 s |
| v1 | `False` | 20 | 0,66 s | 1,47 s | 1,93 s |
| v1 | `False` | 100 | 3,63 s | 7,09 s | 10,94 s |
| tiny | | 100 | 0,58 s | 0,57 s | 0,59 s |

- Cada covariable es una fila más por serie: con dos, el coste se multiplica por 3 sin *ensemble* (100 series) y por 4,2 con él (20 series). El tiny las ignora (con un aviso) y no cuesta más.

**Tiempo por llamada** (`exp1_smoke.py`, `exp1.log`): backend directo, sin covariables, 24 pasos, mediana de 3 llamadas tras una de calentamiento; carga desde la caché local.

| Checkpoint | Carga | 1 serie x 512 | 100 series x 512 | 1 serie x 4096 | 1 serie x 8192 | 1 serie x 16384 |
|:-----------|------:|--------------:|-----------------:|---------------:|---------------:|----------------:|
| v1, `patch_ensemble=True` (por defecto) | 0,7 s | 1,03 s | 93,1 s | 2,86 s | 2,25 s | 1,62 s |
| v1, `patch_ensemble=False` | | 0,15 s | 3,7 s | 0,53 s | 0,81 s | 1,65 s |
| tiny (un solo tamaño de parche) | 2,3 s | 0,02 s | 0,97 s | 0,10 s | 0,19 s | 0,50 s |

- Con el *ensemble*, el coste por llamada no es monótono en el contexto: 512 pasos es lo más barato medido (1,03 s, cuatro pasadas cortas), 4096 lo más caro (2,86 s, tres pasadas) y después baja a 2,25 s (8192, dos pasadas) y 1,62 s (16384, una pasada), porque a las historias largas les caben menos tamaños de parche. Sin el *ensemble*, el coste crece con el contexto (de 0,15 s a 1,65 s). Con 100 series de 512 pasos, el *ensemble* multiplica el tiempo por 25.
- Los tiempos de carga (2,3 s el tiny, el primero del script; 0,7 s v1) no se han medido en procesos separados: no sirven para comparar los dos checkpoints.
- Con el *ensemble*, cien series de 512 pasos en una llamada (93,1 s) cuestan casi lo mismo que cien llamadas de una serie (100 x 1,03 s): en CPU el lote apenas ahorra cómputo.

**Lotes grandes y relleno interno** (`exp6_large_batches.py`, `exp6.log`; v1, 24 pasos, sin covariables salvo donde se indica):

- **Escala lineal sin trocear en skforecast**: con `patch_ensemble=False`, 100, 400 y 1.000 series de 512 pasos tardan 3,7 s, 16,4 s y 39,8 s (de 37 a 41 ms por serie). El backend agrupa las series en pasadas de `batch_size` filas, así que no hace falta el `max_series_per_call` que el informe de Timestral proponía (su decisión D6).
- **Longitudes mezcladas**: 200 series de 500 a 4.096 pasos (media 2.414) tardan 116,4 s, casi como 200 series de 4.096 (121,6 s), porque las 200 caben en una pasada de 256 filas y el backend las rellena hasta la más larga. Ordenarlas por longitud no cambia nada (117,7 s, misma previsión). Con el *ensemble*, 40 series mezcladas tardan más que 40 de 4.096 (169,9 s frente a 138,2 s), porque las de 2.048 pasos o menos añaden la pasada con parches de 2; ordenadas, 162,8 s.
- **`batch_size` con covariables** (400 series de 512 pasos y dos covariables futuras, una numérica y otra de etiquetas): 40,7 s con 64, 44,4 s con 256 y 68,0 s con 1.024. En CPU, los lotes grandes no aceleran. El pico de memoria del proceso (4,25 GB) se alcanzó en el primer bloque (200 series de hasta 4.096 pasos, en un solo lote de 200 filas) y no cambió después, así que este experimento no mide la memoria por `batch_size`; en GPU, `batch_size` es la forma de acotarla **[inferido]**.

## 3. Tabla de capacidades frente a los adapters existentes

Valores de los atributos de clase en `skforecast/foundation/_adapters.py` (rama `0.27.x`, commit `fc7efd0`, que incluye `requires_equal_length_context` del PR #1450) y de `_MODEL_LICENSES` en `_utils.py` **[verificado en código]**. Timestral (propuesta de su informe, sin implementar) va como referencia. La fila de TimeH2O es la propuesta de la sección 4.

**Capacidades**

| Adapter | `allow_exog` | Solo pasadas | Categóricas | Exog heterogéneas en un lote | Mismo largo (PR #1450) | NaN en la serie | `SUPPORTED_QUANTILES` | `context_length` por defecto |
|:--------|:----:|:----:|:----:|:----:|:----:|:----:|:----|----:|
| Chronos-2 | Sí | Sí | Sí | No | No | Sí | `None` (cualquiera) | 8192 |
| TimesFM 2.5 | No | No | No | Sí | No | Sí | 0.1 ... 0.9 | 512 |
| TimesFM 3.0 | Sí | Sí | No | No | No | Sí | 0.1 ... 0.9 | 2048 |
| Moirai-2 | No | No | No | Sí | No | Sí | 0.1 ... 0.9 | 2048 |
| TabICL | Sí | No | No | No | No | Sí | `None` | 4096 |
| TabPFN-TS | Sí | No | No | Sí | No | Sí | `None` | 32768 |
| T0 | Sí | No | No | Sí | Sí | Sí | `None` | 8192 |
| Nori | Sí | No | No | Sí | No | Sí | `None` | 4096 |
| TS-ICL | Sí | Sí | No | No | No | Sí | 0.01 ... 0.99 (paso 0.01) | 4096 |
| Timestral (propuesta) | Sí | Sí | Sí | No | Sí | Sí | 21 fijos | 4096 |
| **TimeH2O (propuesta)** | **Sí** | **No** (v1 ignora las pasadas) | **Sí** (nativas en el backend) | **Sí** | **No** | **Sí** | **`None`** (fuera de [0.1, 0.9], extrapolados) | **16384** |

**Distribución, pesos y licencia**

| Adapter | `requires_hf_auth` | `requires_provider_auth` | `weights_repo_id` | `weights_in_hf_cache` | `backend_package` | `default_model_id` | Uso comercial restringido |
|:--------|:----:|:----:|:----|:----:|:----|:----|:----:|
| Chronos-2 | No | No | `None` | Sí | `chronos-forecasting` | `autogluon/chronos-2-small` | No |
| TimesFM 2.5 | No | No | `None` | Sí | `timesfm[torch]` | `google/timesfm-2.5-200m-pytorch` | No |
| TimesFM 3.0 | No | No | `None` | Sí | `timesfm[torch]` | `google/timesfm-3.0-pytorch` | Sí |
| Moirai-2 | No | No | `None` | Sí | `uni2ts` | `Salesforce/moirai-2.0-R-small` | Sí |
| TabICL | No | No | `jingang/TabICL` | Sí | `tabicl[forecast]` | `soda-inria/tabicl` | No |
| TabPFN-TS | No | Sí | `Prior-Labs/tabpfn_3_5` | No | `tabpfn-time-series>=1.3` | `priorlabs/tabpfn-ts` | Sí |
| T0 | No | No | `None` | Sí | `tfc-t0` | `theforecastingcompany/t0-alpha` | No |
| Nori | No | No | `None` | Sí | `synthefy-nori` | `Synthefy/Nori` | No |
| TS-ICL | No | No | `taharnbl/TS-ICL` | Sí | `tsicl` | `taharnbl/TS-ICL` | Sí |
| Timestral (propuesta) | No | No | `None` | Sí | `timestral @ git+https://github.com/mistralai/timestral.git` | `mistralai/Timestral-28M` | Sí |
| **TimeH2O (propuesta)** | **Sí** (el primero) | **No** | **`None`** | **Sí** | **`h2ofm>=0.43`** | **`h2oai/timeh2o-v1`** | **Sí** (no comercial y no productivo) |

Lo que más se parece:

- **Capacidades:** Chronos-2 (covariables categóricas nativas, cualquier cuantil) y T0 y Nori (`supports_heterogeneous_covariates=True`). TimeH2O es el único que combina categóricas nativas con lotes heterogéneos, y el primero con `requires_hf_auth=True`.
- **Constructor y carga:** TimesFM 3.0 (`device` como texto, `predict_kwargs` con claves reservadas, `model` inyectable, `_MODEL_ID_PREFIX`).
- **Construcción de las covariables:** T0 (histórico seguido del futuro, solo columnas futuras), pero sin relleno ni unión de columnas: cada serie lleva las suyas.
- **Cuantiles:** T0 (pide los niveles distintos y ordenados y reordena las columnas).

Con `SUPPORTED_QUANTILES = None`, TimeH2O da intervalos de cualquier nivel. Los del 80 % (`[0.1, 0.9]`, el valor por defecto de `predict_interval`) salen de cuantiles entrenados; los del 90 % y el 98 %, de la cola gaussiana extrapolada (sección 2, Experimento).

## 4. Propuesta de diseño

El esqueleto de esta sección es código completo, no un resumen. `dev/timeh2o_adapter/experiments/skeleton/` lo extrae de este documento y lo ejecuta contra el backend real y contra `0.27.x` con el PR #1450 (sección 12). El prototipo de los experimentos (`experiments/timeh2o_proto.py`) es una versión anterior y más corta, escrita antes del PR #1450 (ya declara `requires_equal_length_context`, que sin el PR no se usa); el diseño revisado es el de esta sección.

### Plantilla

`TimesFM3Adapter` para el constructor y la carga (`device` como texto resuelto con `_resolve_torch_device`, `predict_kwargs` con claves reservadas y `_validate_predict_kwargs`, `model` inyectable para los tests, `set_params` con un cierre `validate` y reinicio del modelo cuando cambia un argumento de `from_pretrained`), y `T0Adapter` para `predict` (una sola llamada con todas las series, niveles de cuantil distintos y ordenados, covariables formadas por el histórico seguido del futuro y solo de las columnas futuras). Con dos diferencias que hacen el adapter más delgado que T0: no hay relleno ni unión de columnas (cada serie lleva su contexto y sus columnas), y casi no hay conversión de tipos (las columnas llegan al backend con su dtype, como en Chronos-2; solo las booleanas pasan a 0/1, decisión D8).

### Registro

- Prefijo en `_ADAPTER_REGISTRY`: `"h2oai/timeh2o"`, al final del diccionario (el orden del registro es el de `list_adapters()`). La entrada que hoy cierra el diccionario (`"taharnbl/TS-ICL":    TSICLAdapter`) no lleva coma: añadírsela y alinear la clave nueva como las demás. Cubre los dos checkpoints y no captura otros modelos de H2O.ai (`h2oai/h2ogpt-*`, `h2oai/tabh2o-*`).
- `default_model_id`: `"h2oai/timeh2o-v1"` (decisión D5).
- Entrada en `_MODEL_LICENSES` (decisión D2), en `skforecast/foundation/_utils.py`, tras la de TS-ICL:

<!-- skeleton:license -->
```python
    "h2oai/timeh2o": (
        "h2o-model-non-commercial-licence-v1.0",
        "https://huggingface.co/h2oai/timeh2o-v1/blob/main/LICENSE",
        True,
        "H2O Model Non-Commercial Licence v1.0 (non-commercial and "
        "non-production use only)",
    ),
```

  El primer campo es el `license_name` de la tarjeta, que es lo que compara `tools/check_foundation_models_metadata.py`. El URL es fijo porque los dos repositorios publican el mismo `LICENSE` (mismo sha256) y la página es pública sin autenticarse **[probado]**. El cuarto campo es el texto del `LicenseWarning` ("The weights for '...' are released under {nombre}. Review the license terms before commercial or production use. See {URL}.").

### Atributos de capacidad

| Atributo | Valor | Justificación |
|:---------|:------|:--------------|
| `SUPPORTED_QUANTILES` | `None` | Cualquier nivel en (0, 1): interpolado entre los 9 entrenados o extrapolado con la cola gaussiana fuera de [0.1, 0.9] **[verificado en código]**. El backend rechaza 0, 1 y los niveles repetidos; el adapter quita los repetidos (como T0) y deja que el backend rechace 0 y 1, con un `ValueError` claro **[probado]**. |
| `allow_exog` | `True` | Covariables futuras. Se declara por adapter: el tiny las ignora con un aviso del backend (decisión D12). |
| `supports_past_only_covariates` | `False` | Los checkpoints v1 no usan covariables pasadas **[verificado en código]** **[documentado]**. `FoundationModel` avisa con `IgnoredArgumentWarning` y el adapter no las envía, para no repetir el aviso del backend. Si un checkpoint futuro las admite, habrá que revisarlo (es un atributo por adapter). |
| `supports_categorical_covariates` | `True` | El backend codifica las columnas de texto y `category` contra la historia de cada serie **[verificado en código]**; el adapter no las traduce. Las booleanas, que el backend también trataría como etiquetas, se pasan como 0/1 (decisión D8). |
| `supports_heterogeneous_covariates` | `True` | `forecast_many` recibe las covariables de cada serie por separado y la atención entre variables solo mezcla filas de la misma serie **[verificado en código]**; una serie da la misma previsión en un lote heterogéneo que sola (sección 2, Experimento) **[probado]**. `FoundationModel` llama al adapter una sola vez. |
| `requires_equal_length_context` | `False` | Atributo del PR #1450. `forecast_many` acepta longitudes distintas y rellena internamente sin cambiar la previsión, mientras que el relleno de `FoundationModel` sí la cambia (el *ensemble* de parches depende de la longitud de la historia; hasta 0,25 en `exp1.log`) **[probado]**. |
| `supports_nan_in_series` | `True` | NaN marca un valor ausente **[verificado en código]** **[probado]**. |
| `requires_hf_auth` | `True` | Repositorios gated (`gated: auto`) **[probado]**. Es el primer adapter con `True`: la documentación que dice "currently none of the supported models" cambia (sección 9). Ningún código de skforecast lo usa en tiempo de ejecución; lo exponen `get_model_info` y `list_adapters`, y lo compara `check_foundation_models_metadata.py` con el Hub. |
| `requires_provider_auth` | `False` | No hace falta cuenta de H2O ni nada fuera del Hub. |
| `weights_repo_id` | `None` | Los pesos se descargan de `model_id`. |
| `weights_in_hf_cache` | `True` | `snapshot_download` con la caché estándar del Hub **[verificado en código]**. |
| `backend_package` | `"h2ofm>=0.43"` | 0.42.0 no carga v1 **[probado]** (decisión D4). Es lo que el usuario pasa a `pip install` y lo que instala `tools/foundation_compat/run.py` (`distribution_name` devuelve `h2ofm`). |
| `default_model_id` | `"h2oai/timeh2o-v1"` | El tiny no admite covariables (decisión D5). |

### Parámetros del constructor

| Parámetro | Por defecto | Comentario |
|:----------|:------------|:-----------|
| `model_id` | | Comprobado con `_validate_model_id_prefix`, como TimesFM 3.0. |
| `model` | `None` | `h2ofm.TimeH2O` ya cargado. Para los tests. |
| `context_length` | `16384` (decisión D6) | El máximo del modelo. Con la configuración de la guía es el que mejor resultado da y, con el *ensemble* de parches, más rápido que 2048, 4096 y 8192 (una sola pasada con historias de más de 8.192 pasos; 500 es más rápido) (sección 2, Experimento). Sin el *ensemble* o con el tiny, un contexto más corto es más rápido (D6). Precedente: Chronos y TabPFN-TS usan el máximo de su modelo. |
| `device` | `"auto"` | CUDA > CPU; MPS pasa a CPU con aviso (decisión D10), con el helper del PR #1450. |
| `batch_size` | `256` (decisión D9) | El valor del backend. Argumento de `from_pretrained`: al cambiarlo se recarga el modelo. No cambia la previsión **[probado]**. |
| `predict_kwargs` | `None` | Se pasan a `forecast_many`: `patch_ensemble`, `clip_nonnegative`, `snap_integer`, `autoregressive` y `flip_equivariant` (este último sin efecto en los dos checkpoints). Quedan reservadas `contexts`, `horizon`, `past_covariates`, `future_covariates`, `quantile_levels` y `context_length` (el contexto ya llega recortado a `context_length`; pasar otro valor lo recortaría de nuevo sin que `FoundationModel` lo sepa). |

Las opciones de descarga de `from_pretrained` (`revision`, `token`, `cache_dir`, `local_files_only`) no se exponen, como en los demás adapters: el token se toma de `hf auth login` o de `HF_TOKEN`, y el modo sin conexión de `HF_HUB_OFFLINE` **[probado]**.

### Cambios de código compartido

Uno, además del PR #1450: el aviso de covariables futuras planas (decisión D15), como helper en `skforecast/foundation/_utils.py`, porque Timestral necesita el mismo aviso (su informe lo proponía como método del adapter). El adapter declara `requires_equal_length_context = False` (atributo obligatorio que añade el PR #1450) y usa `_resolve_torch_device(..., mps_fallback_message=..., stacklevel=6)`, también de ese PR. No hace falta relleno (el backend acepta longitudes distintas), ni troceado de lotes (el backend trocea por `batch_size` filas), ni `_match_quantile_indices` (el backend devuelve los niveles en el orden pedido), ni cambios en `check_model` de `tools/check_foundation_models_metadata.py` (la tarjeta tiene `license_name`; solo se añade el tiny a `EXTRA_MODEL_IDS`, sección 8). En lo que pide `.claude/rules/foundation.md`, el adapter lo cumple: solo traduce la entrada ya normalizada a la llamada del backend, sin bucles de relleno ni agrupado propios. El único bucle por serie del adapter es la construcción de las filas de covariables, que es traducción (como en TimesFM 3.0 y T0); el del aviso está en el helper.

**El aviso se aparta de la regla.** `.claude/rules/foundation.md` pide presentar como recomendación la opción centralizada en `FoundationModel` (aquí, D15 opción d: un atributo del contrato que active la comprobación en el bucle de grupos de `FoundationModel.predict`). La recomendación de D15 es otra: el adapter llama a un helper compartido. Motivos: el riesgo depende del backend y del checkpoint (el tiny de TimeH2O ignora las covariables y no se ha medido en los demás backends), el remedio que da el mensaje es propio de cada backend, y el helper ya evita duplicar el bucle, como `_warn_if_non_commercial`. Si los mantenedores prefieren seguir la regla al pie de la letra, D15 (d) es la alternativa, con su coste (un atributo obligatorio más en todos los adapters).

El helper va en `_utils.py`, tras `_warn_if_non_commercial`, y necesita importar `FeatureOutOfRangeWarning` (`from ..exceptions import FeatureOutOfRangeWarning, LicenseWarning, MissingValuesWarning`). El umbral (20 desviaciones típicas) sale de `exp8.log` y `exp9.log` (sección 2, Covariables; decisión D15).

<!-- skeleton:shared_utils -->
```python
def _is_number_like(col_data: pd.Series) -> bool:
    """
    Whether a covariate column holds numbers or booleans.

    Parameters
    ----------
    col_data : pandas Series
        A single covariate column.

    Returns
    -------
    is_number_like : bool
        `True` for numeric and boolean dtypes, and for `object` columns that
        only hold booleans and missing values.

    """

    return (
        pd.api.types.is_numeric_dtype(col_data)
        or pd.api.types.is_bool_dtype(col_data)
        or (
            col_data.dtype == object
            and pd.api.types.infer_dtype(col_data, skipna=True) == "boolean"
        )
    )


def _warn_if_flat_known_covariates(
    context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    model_name: str,
    remedy: str,
    stacklevel: int,
    max_deviation: float = 20.0,
) -> None:
    """
    Warn when a numeric known covariate is constant, or nearly constant, over
    the context of a series and takes other values over its forecast horizon.

    Parameters
    ----------
    context_exog : dict, None
        Per-series historical exogenous variables.
    exog : dict, None
        Per-series future exogenous variables.
    model_name : str
        Name of the model, used in the warning message.
    remedy : str
        Sentence appended to the warning message that tells the user how to
        avoid the problem with this model.
    stacklevel : int
        `stacklevel` of the warning, so that it points to the user's call.
    max_deviation : float, default 20.0
        A column is reported when the largest distance between its future
        values and the mean of its context is larger than `max_deviation`
        times the standard deviation of its context, or when the context is
        constant and the future values differ from it.

    Returns
    -------
    None

    Notes
    -----
    Some foundation models (Timestral, TimeH2O) normalize each covariate
    with statistics that grow from the start of the context. When a
    covariate barely varies over the context, any change over the horizon
    becomes a jump of many scales, far outside what the model saw in
    training, and the forecast, mainly its upper quantiles, can blow up.
    Only columns that are numeric or boolean (also `object` columns holding
    only booleans and missing values) both in `context_exog` and in `exog`
    are checked; NaN values are ignored.

    """

    if context_exog is None or exog is None:
        return

    affected = []
    for series_name, series_exog in exog.items():
        series_context_exog = context_exog.get(series_name)
        if series_exog is None or series_context_exog is None:
            continue
        fut_df = (
            series_exog.to_frame()
            if isinstance(series_exog, pd.Series)
            else series_exog
        )
        ctx_df = (
            series_context_exog.to_frame()
            if isinstance(series_context_exog, pd.Series)
            else series_context_exog
        )
        for col in fut_df.columns:
            if col not in ctx_df.columns or not (
                _is_number_like(ctx_df[col]) and _is_number_like(fut_df[col])
            ):
                continue
            history = ctx_df[col].to_numpy(dtype=np.float64, na_value=np.nan)
            future = fut_df[col].to_numpy(dtype=np.float64, na_value=np.nan)
            history = history[~np.isnan(history)]
            future = future[~np.isnan(future)]
            if history.size == 0 or future.size == 0:
                continue
            deviation = np.abs(future - history.mean()).max()
            if deviation > 0 and deviation > max_deviation * history.std():
                affected.append((series_name, col))

    if affected:
        warnings.warn(
            f"Known covariates that are constant, or nearly constant, over the "
            f"context and take other values over the forecast horizon, as "
            f"(series, column): {affected}. {model_name} can return very "
            f"large quantiles in this case. {remedy}",
            FeatureOutOfRangeWarning,
            stacklevel=stacklevel,
        )
```

### Esqueleto del adapter

Va en `skforecast/foundation/_adapters.py`, después del último adapter del módulo (`NoriAdapter` en `0.27.x`, o `TimestralAdapter` si ya está) y justo antes de `_ADAPTER_REGISTRY`. Importaciones: usa `_validate_positive_int`, `_validate_model_id_prefix`, `_apply_set_params`, `_warn_if_non_commercial` y `get_exog_signature`, que `_adapters.py` ya importa de `_utils.py`, `_resolve_torch_device`, definida en el propio módulo, y el helper nuevo, que hay que añadir a la importación `from ._utils import (...)` de `_adapters.py`.

<!-- skeleton:adapter -->
```python
class TimeH2OAdapter(_AdapterBase):
    """
    Adapter for H2O.ai TimeH2O foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"h2oai/timeh2o-v1"`. Must start with
        `"h2oai/timeh2o"`.
    model : object, default None
        Pre-loaded `h2ofm.TimeH2O` instance. If `None`, the model is loaded
        lazily on the first `predict` call.
    context_length : int, default 16384
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Must be a positive integer. 16384 is the maximum
        context of TimeH2O: with a larger value, the backend only uses the
        last 16384 observations.
    device : str, default 'auto'
        Device placement for the model. `"auto"` selects CUDA when it is
        available, otherwise CPU. Also accepts explicit values such as
        `"cuda"`, `"cuda:1"` or `"cpu"`, forwarded to
        `TimeH2O.from_pretrained`. MPS is not supported by h2ofm: `"mps"`
        (also with an index, or selected by `"auto"` on Apple Silicon
        without CUDA) falls back to CPU with a warning.
    batch_size : int, default 256
        Approximate number of rows (targets and covariates) that the backend
        runs at once, forwarded to `TimeH2O.from_pretrained`: whole series are
        added to a batch until it reaches `batch_size` rows, so a batch can
        exceed it by the rows of one series. It changes speed and memory use,
        not the forecasts. Must be a positive integer.
    predict_kwargs : dict, default None
        Additional keyword arguments forwarded verbatim to `forecast_many`
        (e.g. `patch_ensemble`, `clip_nonnegative`, `snap_integer`,
        `autoregressive`). Cannot include `contexts`, `horizon`,
        `past_covariates`, `future_covariates`, `quantile_levels` or
        `context_length`, which are managed internally.

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
    batch_size : int
        Approximate number of rows the backend runs at once.
    predict_kwargs : dict
        Additional keyword arguments forwarded to `forecast_many`.
    allow_exog : bool
        Whether this adapter accepts exogenous variables. Always `True`.
    supports_past_only_covariates : bool
        Whether historical exog columns without future values are used as
        past-only covariates. Always `False`: the TimeH2O checkpoints do not
        use past covariates, so `FoundationModel` warns and the adapter
        forwards only the columns that also have future values.
    supports_heterogeneous_covariates : bool
        Whether series with different covariate columns can be forecast in
        the same backend call. Always `True`: `forecast_many` takes the
        covariates of each series separately and only mixes the rows of the
        same series.
    requires_equal_length_context : bool
        Whether the backend needs every series of a call to have the same
        context length, so that `FoundationModel` left-pads them with NaN.
        Always `False`: `forecast_many` takes series of different lengths,
        and padding them would change the forecasts, because the patch sizes
        used for each series depend on the length of its context.
    supports_nan_in_series : bool
        Whether the backend accepts NaN values in the series used as
        context. Always `True`.
    supports_categorical_covariates : bool
        Whether the backend supports non-numeric covariates natively, so
        they do not have to be encoded as numbers. `True`: h2ofm encodes
        string and categorical columns against the history of each series.
    SUPPORTED_QUANTILES : list, None
        Quantile levels accepted by the backend. `None` means any level in
        `(0, 1)`.
    requires_hf_auth : bool
        Whether the checkpoints served by this adapter are gated on the
        Hugging Face Hub, so an authenticated account that has accepted the
        model license is needed. Declared per adapter, not per checkpoint.
        `True` for TimeH2O.
    requires_provider_auth : bool
        Whether the model provider requires its own account or license
        acceptance, outside the Hugging Face Hub, before the weights can be
        used.
    weights_repo_id : str, None
        Hugging Face repository the backend downloads the weights from when
        it is not `model_id`. `None` means the repository is `model_id`.
        `None` for TimeH2O: the weights are downloaded from `model_id`.
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
    TimeH2O predicts 9 quantiles (0.1 to 0.9). Any level in `(0, 1)` can be
    requested: levels between 0.1 and 0.9 are interpolated from the trained
    ones, and levels outside that range are extrapolated with a Gaussian tail
    anchored on the median and on the 0.1 or 0.9 quantile. The point
    forecast is the median (quantile 0.5).

    For each series, every column present in its future `exog` becomes a
    future covariate spanning `context + horizon`, built by concatenating the
    matching historical column from `context_exog` with the future values.
    Columns present only in `context_exog` are ignored, because TimeH2O does
    not use past covariates. Each series is forwarded with its own columns
    and series of different lengths are forecast in the same call: the
    backend only mixes the rows of the same series. Numeric and boolean
    columns are passed as numbers (booleans as 0 and 1, as in the other
    adapters). String and categorical columns are passed as labels, which
    h2ofm encodes against the history of each series; a missing label is a
    category of its own. NaN values in the series and in
    numeric covariates are treated as missing values; infinite values raise
    a `ValueError` in the backend. A numeric covariate that is NaN over the
    whole context is dropped by the backend. With future covariates, the forecast
    of the first steps also depends on the later future values, so it
    changes with `steps`.

    A known covariate that is constant, or nearly constant, over the
    context and changes over the forecast horizon (for example a price that
    changes only over the horizon, or a 0/1 indicator that is 0 over the
    whole context) can make TimeH2O return very large upper quantiles. The
    adapter raises a `FeatureOutOfRangeWarning` naming the affected series
    and columns (not with the tiny checkpoint, which ignores covariates). A
    context long enough to include earlier changes of the column avoids it.

    By default, h2ofm averages the forecasts of every patch size whose
    window holds the context of a series (`patch_ensemble=True`), one pass
    per size: series of up to 2048 observations use four passes, so on CPU
    they are several times slower than with
    `predict_kwargs={"patch_ensemble": False}`.

    The `"h2oai/timeh2o-v1-tiny"` checkpoint does not use covariates: h2ofm
    ignores them with a warning.

    The checkpoints are gated on the Hugging Face Hub: accept their
    conditions with a Hugging Face account and authenticate (`hf auth login`
    or the `HF_TOKEN` environment variable) before loading them. The
    pre-trained weights are released under a non-commercial license that
    also excludes production use, so loading them raises a `LicenseWarning`.
    h2ofm requires Python >= 3.12.

    References
    ----------
    .. [1] https://huggingface.co/h2oai/timeh2o-v1

    .. [2] https://pypi.org/project/h2ofm/

    """

    SUPPORTED_QUANTILES: list[float] | None = None
    allow_exog: bool = True
    supports_past_only_covariates: bool = False
    supports_categorical_covariates: bool = True
    supports_heterogeneous_covariates: bool = True
    requires_equal_length_context: bool = False
    supports_nan_in_series: bool = True
    requires_hf_auth: bool = True
    requires_provider_auth: bool = False
    weights_repo_id: str | None = None
    weights_in_hf_cache: bool = True
    backend_package: str = "h2ofm>=0.43"
    default_model_id: str = "h2oai/timeh2o-v1"

    _MODEL_ID_PREFIX: str = "h2oai/timeh2o"
    _RESERVED_PREDICT_KWARGS: frozenset[str] = frozenset({
        "contexts", "horizon", "past_covariates", "future_covariates",
        "quantile_levels", "context_length",
    })

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 16384,
        device: str = "auto",
        batch_size: int = 256,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. `"h2oai/timeh2o-v1"`. Must start with
            `"h2oai/timeh2o"`.
        model : object, default None
            Pre-loaded `h2ofm.TimeH2O` instance. If `None`, the model is
            loaded lazily on the first `predict` call.
        context_length : int, default 16384
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `context` is
            longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        device : str, default 'auto'
            Device placement for the model. `"auto"` selects CUDA when it
            is available, otherwise CPU. MPS falls back to CPU with a
            warning.
        batch_size : int, default 256
            Approximate number of rows the backend runs at once. Must be a
            positive integer.
        predict_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to
            `forecast_many`.

        """

        _validate_model_id_prefix(model_id, self._MODEL_ID_PREFIX, type(self).__name__)
        _validate_positive_int("context_length", context_length)
        _validate_positive_int("batch_size", batch_size)
        predict_kwargs = predict_kwargs or {}
        self._validate_predict_kwargs(predict_kwargs)

        self.model_id       = model_id
        self._model         = model
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.device         = device
        self.batch_size     = batch_size
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
        from `context`, `context_exog`, `exog`, `steps` and `quantiles`, so
        passing them explicitly would collide with the internal
        `forecast_many` call. `context_length` is reserved because the
        context is already trimmed to the adapter's `context_length`.

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
            Keys: `model_id`, `context_length`, `device`, `batch_size`,
            `predict_kwargs`.

        """

        return {
            "model_id":       self.model_id,
            "context_length": self.context_length,
            "device":         self.device,
            "batch_size":     self.batch_size,
            "predict_kwargs": self.predict_kwargs or None,
        }

    def set_params(self, **params) -> TimeH2OAdapter:
        """
        Set adapter parameters. Resets the model when parameters that affect
        loading change.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `device`, `batch_size`,
            `predict_kwargs`.

        Returns
        -------
        self : TimeH2OAdapter

        Notes
        -----
        `model_id`, `device` and `batch_size` are arguments of
        `TimeH2O.from_pretrained`, so they discard the cached model.
        `context_length` and `predict_kwargs` are applied at predict time and
        never trigger a reload.

        """

        def validate(candidate_params: dict) -> dict:
            if "model_id" in candidate_params:
                _validate_model_id_prefix(
                    candidate_params["model_id"],
                    self._MODEL_ID_PREFIX,
                    type(self).__name__,
                )
            for key in ("context_length", "batch_size"):
                if key in candidate_params:
                    _validate_positive_int(key, candidate_params[key])
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
                (
                    {"model_id", "device", "batch_size"},
                    lambda: setattr(self, "_model", None),
                ),
            ),
        )

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> TimeH2OAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since TimeH2O is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables.

        Returns
        -------
        self : TimeH2OAdapter

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
        Generate predictions using the TimeH2O model.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict
            Per-series context windows (already trimmed to
            `context_length`). Series may have different lengths.
        context_exog : dict, None
            Per-series historical exogenous variables (already trimmed).
            Only the columns also present in the `exog` of the same series
            are used.
        exog : dict, None
            Per-series future exogenous variables for the forecast
            horizon. Each column is forwarded as a future covariate of that
            series, concatenated with its historical values.
        quantiles : list of float or None
            Quantile levels to return, in the requested order. If `None`, a
            point forecast (median, quantile 0.5) is produced.

        Returns
        -------
        predictions : dict
            Keys are series names, in the same order as `context`. Each
            value is a 2-D array of shape `(steps, n_quantiles)` with columns
            ordered to match `quantiles`.

        Notes
        -----
        All series are forecast in one `forecast_many` call, each with its
        own context length and covariate columns. h2ofm only accepts
        distinct levels strictly between 0 and 1: the distinct levels are
        requested in ascending order and the columns are then reordered to
        match `quantiles`. A `ValueError` is raised by the backend for
        levels equal to 0 or 1.

        """

        # NOTE: the model is loaded lazily here so that the adapter can be
        # instantiated and fitted without requiring h2ofm to be installed.
        self._load_model()

        names = list(context.keys())
        requested = list(quantiles) if quantiles is not None else [0.5]
        # h2ofm rejects duplicated levels: query the distinct ones, then
        # reindex the columns back to the order the caller asked for.
        query_levels = sorted(set(requested))

        future_covariates = [
            self._build_future_covariates(
                context_exog = (
                    context_exog.get(series_name) if context_exog is not None else None
                ),
                exog         = exog.get(series_name) if exog is not None else None,
            )
            for series_name in names
        ]

        # Checkpoints without cross-variate attention (the tiny one) ignore
        # the covariates, so a flat covariate cannot affect their forecast.
        if self._model.config.variate_attention:
            _warn_if_flat_known_covariates(
                context_exog = context_exog,
                exog         = exog,
                model_name   = "TimeH2O",
                remedy       = (
                    "A `context_length` that includes earlier changes of these "
                    "columns avoids it; otherwise, consider removing them."
                ),
                stacklevel   = 5,
            )

        forecasts = self._model.forecast_many(
            [
                context[series_name].to_numpy(dtype=np.float64, na_value=np.nan)
                for series_name in names
            ],
            horizon           = steps,
            future_covariates = (
                future_covariates
                if any(rows is not None for rows in future_covariates)
                else None
            ),
            quantile_levels   = query_levels,
            **self.predict_kwargs,
        )

        quantile_column_indices = [query_levels.index(q) for q in requested]

        return {
            series_name: np.asarray(forecast.quantiles)[:, quantile_column_indices]
            for series_name, forecast in zip(names, forecasts)
        }

    @staticmethod
    def _bool_to_float(col_data: pd.Series) -> pd.Series:
        """
        Cast a boolean covariate column to float, so that it is passed as a
        0/1 number.

        Parameters
        ----------
        col_data : pandas Series
            A single covariate column.

        Returns
        -------
        col_data : pandas Series
            `col_data` as float (NaN for missing values) if it is boolean
            (`bool`, nullable `boolean`, or `object` holding only booleans
            and missing values, as after `FoundationModel` fills a missing
            timestamp), otherwise `col_data` unchanged.

        """

        if pd.api.types.is_bool_dtype(col_data) or (
            col_data.dtype == object
            and pd.api.types.infer_dtype(col_data, skipna=True) == "boolean"
        ):
            return pd.Series(
                col_data.to_numpy(dtype=np.float64, na_value=np.nan),
                index=col_data.index,
                name=col_data.name,
            )

        return col_data

    @classmethod
    def _build_future_covariates(
        cls,
        context_exog: pd.DataFrame | pd.Series | None,
        exog: pd.DataFrame | pd.Series | None,
    ) -> list[pd.Series] | None:
        """
        Build the future covariate rows of one series from its own exogenous
        variables.

        Parameters
        ----------
        context_exog : pandas DataFrame, pandas Series, None
            Historical exogenous variables aligned to the context of this
            series. Must contain every column of `exog`.
        exog : pandas DataFrame, pandas Series, None
            Future-known exogenous variables covering the forecast horizon
            of this series.

        Returns
        -------
        future_covariates : list, None
            One pandas Series per column of `exog`, in the order returned by
            `get_exog_signature`, with the historical values followed by the
            future values (`context_len + steps` values). `None` if the series
            has no future exog.

        Notes
        -----
        The rows keep the dtype of the columns, so h2ofm decides how to
        encode them: numeric columns are numbers and string or categorical
        columns hold labels. Boolean columns (also the nullable `boolean`
        dtype, and `object` columns that only hold booleans and missing
        values) are cast to float, so they are 0/1 numbers, as in the other
        adapters, instead of labels. Columns present only in `context_exog`
        are not forwarded, because TimeH2O does not use past covariates.

        """

        if exog is None:
            return None

        _, fut_cols = get_exog_signature(context_exog=context_exog, exog=exog)
        if not fut_cols:
            return None

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

        # h2ofm encodes boolean rows as labels: pass them as 0/1 numbers, as
        # the other adapters do. Each part is cast before the concatenation,
        # so that a missing value in one of them does not turn the row into
        # labels.
        return [
            pd.concat(
                [cls._bool_to_float(ctx_df[col]), cls._bool_to_float(fut_df[col])],
                ignore_index=True,
            )
            for col in fut_cols
        ]

    def _load_model(self) -> None:
        """
        Load the TimeH2O model into `self._model` if not already set.

        Returns
        -------
        None

        Notes
        -----
        This method is a no-op when `self._model` is already populated
        (either by a prior call or by the `model` constructor argument).
        The model is imported lazily from `h2ofm` and loaded via
        `TimeH2O.from_pretrained`, resolving `self.device` to a concrete
        device name first. h2ofm only runs on CPU and CUDA (it rejects MPS
        and uses float64 operations internally), so MPS falls back to CPU
        with a warning. `h2ofm` must be installed (it requires
        Python >= 3.12); an `ImportError` is raised otherwise. A
        `LicenseWarning` is issued immediately before the weights are
        loaded. The checkpoints are gated on the Hugging Face Hub: without
        an authenticated account that has accepted their conditions, the
        Hub error raised by h2ofm is propagated.

        """

        if self._model is not None:
            return
        try:
            from h2ofm import TimeH2O
        except ImportError as exc:
            raise ImportError(
                f"h2ofm is required for TimeH2OAdapter. Install it with "
                f'`pip install "{self.backend_package}"` (h2ofm requires '
                f"Python >= 3.12)."
            ) from exc

        _warn_if_non_commercial(self.model_id)

        self._model = TimeH2O.from_pretrained(
            self.model_id,
            device     = _resolve_torch_device(
                             self.device,
                             mps_fallback_message=(
                                 "MPS device is not supported by TimeH2O because "
                                 "the h2ofm library only runs on CPU and CUDA. "
                                 "Falling back to CPU."
                             ),
                             stacklevel=6,
                         ),
            batch_size = self.batch_size,
        )
```

Cada serie va como un objetivo independiente (series 1-D), el comportamiento por defecto del resto de adapters (Chronos-2 con `cross_learning=False`). El modo multivariante de `h2ofm` (`multivariate()`, objetivos de una serie que se ven entre sí) sería el equivalente de `cross_learning`; queda fuera de la primera versión, porque las exog de cada serie tendrían que pasar a ser filas compartidas y, con varios objetivos, el backend cambia la codificación de las categóricas (por rango en lugar de *target encoding*) **[verificado en código]**.

### Decisiones de diseño y su justificación

1. **Sin relleno ni agrupado:** `requires_equal_length_context = False` y `supports_heterogeneous_covariates = True`. `FoundationModel` llama al adapter una sola vez con todas las series, cada una con su longitud y sus columnas; el backend rellena y trocea por dentro sin cambiar la previsión, y el relleno externo sí la cambiaría (`exp1.log`).
2. **Sin troceado en skforecast.** El backend agrupa las series en pasadas de unas `batch_size` filas (series enteras hasta alcanzar el umbral) (sección 2, Hardware; sección 2, Experimento). Lo que en el informe de Timestral era la decisión D6 (`max_series_per_call`) aquí lo resuelve `batch_size`, que el adapter expone (decisión D9).
3. **Cuantiles:** `SUPPORTED_QUANTILES = None` y sin validación en el adapter; solo se quitan los niveles repetidos (como T0), porque el backend los rechaza y `ForecasterFoundation.predict_quantiles` puede pedirlos. 0 y 1 llegan al backend, que lanza un `ValueError` claro (`FoundationModel` admite el intervalo cerrado [0, 1]).
4. **Previsión puntual:** se pide el nivel 0.5, que es exactamente `forecast.point` (`exp1.log`, `run_checks.py`).
5. **Covariables con su dtype, salvo las booleanas:** cada fila es una `pandas.Series` con el dtype de la columna, así que el backend aplica su propia semántica (numéricas frente a etiquetas, `pd.NA` en los dtypes nullable, categorías). La única conversión es la de las booleanas a 0/1, para que una columna booleana signifique lo mismo que en los demás adapters (decisión D8). Los infinitos no se tocan: el backend los rechaza con un error claro (decisión D11).
6. **Covariables pasadas:** no se envían. `FoundationModel` ya avisa (`IgnoredArgumentWarning`, porque `supports_past_only_covariates = False`) y el backend avisaría otra vez con un `UserWarning` que apuntaría a `_adapters.py`.
7. **Acceso gated:** no hay código propio; el token lo gestiona `huggingface_hub` (`hf auth login` o `HF_TOKEN`) y el error del Hub se propaga (decisión D3).
8. **MPS:** helper compartido del PR #1450, con el mismo comportamiento que Moirai (decisión D10).
9. **Covariables planas:** aviso desde el helper compartido `_warn_if_flat_known_covariates`, llamado antes de `forecast_many` solo si el checkpoint usa covariables (`self._model.config.variate_attention`, `False` en el tiny), con `stacklevel=5`, que apunta a la llamada del usuario a `ForecasterFoundation.predict` o `predict_quantiles` (comprobado en `run_checks.py`) (decisión D15). Desde `predict_interval` apunta a `_forecaster_foundation.py` y desde `backtesting_foundation`, a `_validation.py`, igual que el aviso de MPS (aquí y en Moirai) **[probado]**. Con `FoundationModel.predict` llamado directamente apunta un marco por encima de la llamada del usuario (`sys:1` a nivel de módulo) **[probado]**, como el aviso de MPS de Moirai (`stacklevel=7`), que también apunta a `ForecasterFoundation.predict`.
10. **`_validate_predict_kwargs`:** se copia de `TimesFM3Adapter` (classmethod). Con Timestral serían tres copias: si se implementan los dos, conviene moverlo a `_utils.py` en el segundo PR que lo necesite.

`_AdapterBase` no cambia (el atributo nuevo lo añade el PR #1450).

## 5. Alineación con los adapters existentes

Comparación del esqueleto con `TimesFM3Adapter`, `ChronosAdapter` y `T0Adapter`, en el commit de referencia y en el PR #1450 **[verificado en código]**.

| Aspecto | TimesFM3 / Chronos / T0 | Esqueleto | Clasificación |
|:--------|:------------------------|:----------|:--------------|
| Docstring de clase | Parameters, Attributes, Notes, References. TimesFM3 documenta todos los atributos de capacidad; el PR #1450 añade `requires_equal_length_context` a los nueve | Estructura de TimesFM3, con `requires_equal_length_context`. Notes: cuantiles (interpolación y cola gaussiana), mediana, covariables futuras, tipos, lote heterogéneo, tiny, acceso gated y licencia | Igual (TimesFM3) |
| Orden y anotación de los atributos de clase | Orden fijo, todos anotados; `requires_equal_length_context` tras `supports_heterogeneous_covariates` (PR #1450) | El mismo orden y tipos | Igual |
| `_MODEL_ID_PREFIX`, `_RESERVED_PREDICT_KWARGS` | TimesFM 2.5 y 3.0 el primero; solo TimesFM 3.0 el segundo | Los dos | Igual (TimesFM3) |
| Firma del constructor | TimesFM3 `(model_id, *, model=None, context_length=2048, device="auto", predict_kwargs=None)`. Chronos `(model_id, *, pipeline, context_length, predict_kwargs, device_map, torch_dtype, cross_learning)`. T0 `(model_id, *, model, context_length, device_map, torch_dtype)` | `(model_id, *, model=None, context_length=16384, device="auto", batch_size=256, predict_kwargs=None)` | Igual que TimesFM3 más `batch_size` (decisión D9); sin `torch_dtype` porque `h2ofm` siempre usa float32 |
| `context_length` por defecto | 2048 / 8192 / 8192 | 16384 | Distinto y justificado (medido, sección 2) |
| Validación en `__init__` | TimesFM3: prefijo, entero positivo y claves reservadas | Igual, más `batch_size` entero positivo | Igual (TimesFM3) |
| Bloque de asignación | `=` alineados; orden `model_id, _model, context_, context_exog_, context_length, device, predict_kwargs, is_fitted` | Igual, con `batch_size` tras `device` | Igual |
| `get_params` | `model_id, context_length, device, predict_kwargs`, `predict_kwargs or None`; comillas simples | Más `batch_size`; comillas dobles | Comillas: distinto y justificado (el repo pide comillas dobles, `tools/ai/ai_context_header.md`) |
| `set_params` | Cierre `validate`, `resets` con las claves que recargan el modelo | Igual; recargan `model_id`, `device` y `batch_size` (este último es argumento de `from_pretrained`) | Igual (TimesFM3) |
| `fit` | `return super().fit(...)` con docstring | Igual | Igual |
| Orden de `predict` | Chronos y T0: carga, niveles, entradas por serie, llamada, columnas | Carga, niveles distintos y ordenados, covariables por serie, `forecast_many`, reordenado de columnas | Igual (T0) |
| Validación de cuantiles | TimesFM3: `_validate_supported_quantiles`; Chronos y T0: ninguna (`SUPPORTED_QUANTILES = None`) | Ninguna: el backend valida (0 < q < 1) | Igual (Chronos, T0) |
| Niveles repetidos | T0: `sorted(set(requested))` y reindexado | Igual | Igual (T0) |
| Previsión puntual | TimesFM3 usa `out.forecast`; Chronos y T0 piden `[0.5]` | Pide `[0.5]` (igual a `forecast.point`, comprobado) | Igual (Chronos, T0) |
| Comentario `# NOTE: the model is loaded lazily here ...` | Chronos, T0 y TS-ICL | Presente | Igual |
| Carga perezosa | `_load_model`; `_warn_if_non_commercial` justo antes de `from_pretrained` | Igual | Igual (TimesFM3) |
| Texto del `ImportError` | TimesFM: "timesfm is required for ... Install it with `pip install "{backend_package}"`" | "h2ofm is required for TimeH2OAdapter. Install it with `pip install "h2ofm>=0.43"` (h2ofm requires Python >= 3.12)." | Igual que TimesFM (comillas necesarias por `>=`), más la versión de Python |
| MPS | Moirai (PR #1450): `_resolve_torch_device(..., mps_fallback_message=..., stacklevel=7)` | Igual, con su mensaje y `stacklevel=6` | Igual (Moirai, Timestral) |
| Covariables | TimesFM3: `_build_covariates` por serie con `get_exog_signature`; T0: histórico más futuro, solo columnas futuras | `_build_future_covariates` por serie: una `pandas.Series` por columna futura, histórico seguido del futuro; solo las booleanas se convierten (a 0/1) | Igual en la forma (T0); el resto de la conversión de tipos la hace el backend, como en Chronos; las booleanas, como en Chronos, TimesFM 3.0 y T0 (float) |
| Relleno | T0 rellena (en `FoundationModel` tras el PR #1450) | Ninguno; `requires_equal_length_context = False` | Igual que los ocho restantes |
| Avisos propios del adapter | Ninguno avisa sobre los valores de las covariables. TabICL, TabPFN-TS y Nori avisan desde `predict` de un índice que no es `DatetimeIndex` (`stacklevel=3`, que apunta a `FoundationModel.predict`) y Moirai, del paso a CPU desde MPS (`stacklevel=7`, que apunta a `ForecasterFoundation.predict`). La propuesta de Timestral: un método del adapter (`_warn_if_constant_known_covariates`, `stacklevel=5`) | Helper compartido `_warn_if_flat_known_covariates` en `_utils.py`, `stacklevel=5` (apunta a `ForecasterFoundation.predict` y `predict_quantiles`; desde `predict_interval` y el backtesting, a código de skforecast, como Moirai) | Distinto y justificado: dos backends lo necesitan (decisión D15); mismo patrón que `_warn_if_non_commercial` |
| Llamadas con argumentos alineados | `context_exog   = ...` | Igual | Igual |

## 6. Decisiones abiertas

Cada decisión tiene opciones, una recomendación y lo que cambia según la respuesta. El esqueleto de la sección 4 implementa las recomendaciones. Ninguna está tomada.

### D1. Encaje de la licencia en skforecast

- **Contexto:** la licencia de los pesos solo permite uso no comercial y no productivo, define "Derivative" de forma amplia (cualquier obra que "utilises" el modelo) y prohíbe poner el modelo o un derivado a disposición de terceros o integrarlo en productos o servicios para otros (sección 2, Licencia). skforecast no distribuye los pesos: el adapter los descarga en la máquina del usuario, que antes ha aceptado la licencia con su cuenta en el Hub. Los scripts de `experiments/` (incluidos el prototipo `timeh2o_proto.py` y `skeleton/run_checks.py`) y el esqueleto de la sección 4 ya se publican en `dev/` con este documento, como metodología de los resultados (cláusula 4): no contienen pesos ni archivos del modelo, y solo los cargan, a través de `h2ofm`, con una cuenta de Hugging Face que haya aceptado la licencia. skforecast ya integra modelos con licencias no comerciales (TimesFM 3.0, Moirai-2, TabPFN-TS, TS-ICL), pero ninguna excluye también el uso en producción. Esto no es asesoramiento jurídico.
- **Opciones:**
  - (a) Integrarlo sin consultar, como los otros modelos no comerciales, con el `LicenseWarning` y la documentación.
  - (b) Integrarlo y, en paralelo, escribir a H2O (legal@h2o.ai): describir el adapter (sin redistribución de pesos, descarga tras aceptar la licencia), avisar de los resultados ya publicados en `dev/timeh2o_adapter/` y de la tabla de resultados de la guía, como pide la cláusula 4.2, mencionar los scripts y el esqueleto publicados con ellos, y preguntar por la licencia comercial que menciona la cláusula 12.2 (condiciones y contacto, para poder enlazarla en la documentación, como la de TabPFN). Publicar la release que lo incluya cuando respondan o al cumplirse el plazo que fije el campo `Decisión:` (con un responsable del correo; secciones 7 y 13).
  - (c) No integrarlo mientras la licencia excluya el uso en producción.
- **Recomendación:** (b). El coste es un correo; protege frente a una petición de retirada después de publicar, y la cláusula 4 exige de todos modos identificar la versión, la fecha y la metodología de los resultados que publique la guía.
- **Qué cambia:** con (a) o (b), nada en el código. Con (c), se archiva este documento hasta que cambie la licencia (sección 10).
- **Decisión:**

### D2. Entrada de licencia y `LicenseWarning`

- **Opciones:** (a) `commercial_use_restricted=True` con el URL fijo del `LICENSE` de v1 y el nombre "H2O Model Non-Commercial Licence v1.0 (non-commercial and non-production use only)" (sección 4). (b) Lo mismo con el URL a `None`, que `_get_license_url` convierte en la tarjeta de cada `model_id` (como Moirai).
- **Recomendación:** (a). El `LICENSE` es idéntico en los dos repositorios y su página es pública; apunta al texto que hay que leer. Precedentes: TimesFM 3.0 y TS-ICL.
- **Qué cambia:** con (b), el segundo campo, las filas de `test_get_model_info_license_url` y el URL que espera el test 30.
- **Decisión:**

### D3. Acceso gated: mensajes de error

- **Contexto:** sin token o sin aceptar las condiciones, el error lo lanza `huggingface_hub` dentro de `h2ofm`: `GatedRepoError` o, si el Hub no marca el 401 como gated, `FileNotFoundError("'h2oai/timeh2o-v1' is neither a local checkpoint nor an accessible Hugging Face repo")` (sección 2, Distribución). Cuál de los dos recibe un usuario sin token no se ha probado (los experimentos se ejecutaron con una cuenta autenticada).
- **Opciones:** (a) Propagar el error del backend y explicarlo en el docstring y en la documentación. (b) Capturar `GatedRepoError` y `FileNotFoundError` en `_load_model` y relanzar un `OSError` con la guía de skforecast (aceptar las condiciones en `https://huggingface.co/{model_id}` y autenticarse con `hf auth login` o `HF_TOKEN`), como hace T0 con su `TypeError`.
- **Recomendación:** (a), y comprobarlo al implementar (sección 7, Comprobación de D3; sección 13, punto 12) sin credenciales y con la caché del Hub vacía, llamando al backend como hará `_load_model`: `env -u HF_TOKEN -u HF_TOKEN_PATH -u HF_HUB_CACHE -u HUGGINGFACE_HUB_CACHE -u HF_HUB_OFFLINE HF_HOME=$(mktemp -d) python -c "from h2ofm import TimeH2O; TimeH2O.from_pretrained('h2oai/timeh2o-v1-tiny')"` (con `HF_HOME` vacío no hay token guardado ni pesos en caché, y la sesión de `hf` del usuario no se toca), anotando la excepción que llega. `FoundationModel(...)` y `fit` no sirven: los pesos se cargan en el primer `predict` y, al empezar el paso 2, el prefijo aún no está registrado. Tampoco sirve la caché habitual: con los pesos ya descargados, `huggingface_hub` devuelve la copia local aunque el Hub rechace la petición **[verificado en código]**. Si es el `FileNotFoundError`, que no menciona el gating, pasar a (b) solo para ese caso. Si no hay forma de probarlo sin credenciales (por ejemplo, detrás de un proxy que autentica las peticiones al Hub), quedarse con (a) y documentar las dos excepciones posibles.
- **Qué cambia:** con (b), un `try`/`except` en `_load_model`, su docstring y dos tests con un `from_pretrained` falso que lanza cada excepción.
- **Decisión:**

### D4. `backend_package`

- **Opciones:** (a) `"h2ofm>=0.43"`. (b) `"h2ofm"`. (c) Acotar también por arriba (`"h2ofm>=0.43,<0.44"`).
- **Recomendación:** (a). 0.42.0 no carga v1 **[probado]**, y con (b) un entorno con una versión antigua daría un error poco claro. (c) protegería frente a cambios de API (ya cambió un valor por defecto entre 0.42 y 0.43), pero ningún otro backend se acota por arriba, y el check semanal de `tools/foundation_compat` existe para detectar esas roturas.
- **Qué cambia:** la cadena en la clase, en `adapters.toml`, en el `ImportError`, en la documentación, en los tests 1 y 31 y en la fila de `test_get_model_info_output_for_each_adapter`.
- **Decisión:**

### D5. `default_model_id`

- **Opciones:** (a) `"h2oai/timeh2o-v1"`. (b) `"h2oai/timeh2o-v1-tiny"`.
- **Recomendación:** (a). El tiny no admite covariables, que son la razón de ser del adapter frente a TimesFM 2.5 o Moirai, y su precisión es mucho peor en la configuración de la guía (sección 2, Experimento). v1 pesa 435 MB, como Chronos-2.
- **Qué cambia:** con (b), la clase, `test_list_adapters_output`, los ejemplos y `EXTRA_MODEL_IDS`.
- **Decisión:**

### D6. `context_length` por defecto

- **Opciones:** (a) 16384, el máximo del modelo. (b) 4096, como TabICL, Nori, TS-ICL y la propuesta de Timestral. (c) 2048.
- **Recomendación:** (a). En la configuración de la guía es el más preciso (84.24 frente a 98.30 con 4096 y 100.30 con 2048) y, con el *ensemble*, más rápido que 2048 y 4096 (114.0 s frente a 218.9 s y 194.6 s), porque con historias de más de 8.192 pasos solo cabe el parche de 16 y es una sola pasada. El contexto solo limita lo que se guarda y lo que llega al backend: una serie más corta se usa entera, con su propio *ensemble*. Precedente: Chronos (8192) y TabPFN-TS (32768) usan el máximo de su modelo. La ganancia de 16.384 frente a 4096 puede depender de la estacionalidad anual del dataset (sección 2, Experimento). Con v1 y el *ensemble* (los valores por defecto), 4096 no es más rápido ni más preciso en ningún caso medido; en cambio, sin el *ensemble* o con el tiny, 16.384 es más lento que 4096 (0,53 s frente a 1,65 s por llamada sin *ensemble*, `exp1.log`; 2,7 s frente a 12,9 s en el backtesting del tiny, `exp2`), aunque sigue siendo más preciso. Interacción con D7: quien desactive el *ensemble* para ir más rápido debe saber que entonces un contexto más corto también ahorra tiempo (documentarlo junto a `patch_ensemble`).
- **Qué cambia:** la firma, los docstrings, la línea de `FoundationModel` ("Commonly used kwargs by model"), `test_get_model_info_output_for_each_adapter`, `test_TimeH2OAdapter_init_default_params` y las tablas de la documentación.
- **Decisión:**

### D7. `patch_ensemble`

- **Opciones:** (a) Mantener el valor del backend (`True`) y documentar `predict_kwargs={"patch_ensemble": False}` para ir más rápido. (b) Desactivarlo por defecto en el adapter.
- **Recomendación:** (a). Mantener los valores por defecto del backend, como en los demás adapters: con el *ensemble*, el MAE baja entre un 4,5 % y un 8,3 % (sección 2, Experimento). El coste en CPU es grande con historias cortas (de 7 a 25 veces) y nulo con historias de más de 8.192 pasos, así que la documentación debe explicar cuándo conviene `predict_kwargs={"patch_ensemble": False}`: muchas series cortas en CPU; en ese caso, un `context_length` más corto también ahorra tiempo (D6).
- **Qué cambia:** con (b), `__init__` pondría `{"patch_ensemble": False}` cuando `predict_kwargs` es `None`, `get_params` dejaría de devolver `None` y cambiarían los tests 1, 7 y 8.
- **Decisión:**

### D8. Covariables booleanas

- **Contexto:** `h2ofm` trata las columnas booleanas como categóricas (*target encoding* contra la historia de la serie), no como 0/1 (sección 2, Covariables). Los adapters de skforecast que reciben covariables numéricas (Chronos-2, TimesFM 3.0, T0, TS-ICL y Nori) las convierten a float **[verificado en código]**. Medido **[probado]**:
  - Con un indicador sintético que sube el nivel de la serie y aparece en la historia, booleana y float 0/1 dan la misma previsión salvo ruido de float32 (7,6e-6: la parte 2 de `exp4_covariates.py` con más decimales; `exp4.log` lo imprime con tres, 0.000).
  - En `vic_electricity`, `Holiday` como booleana da lo mismo con 4096 (98.35 frente a 98.30) y un 4 % más de error con 16384 (87.59 frente a 84.24) (`results_exp2.csv`). La demanda media en festivo es menor (4107.5 frente a 4691.5 en el entrenamiento), así que la codificación asigna a `True` el valor más bajo, lo que equivale a invertir el signo de la covariable; el modelo no es invariante a esa inversión **[inferido]**.
  - **En contra de convertirlas**: con un indicador que vale `False` en todo el contexto y `True` en el horizonte, la codificación categórica protege de la explosión de los cuantiles altos y el float no. En `exp4.log`, parte 5: como float, mediana máxima 6.05, cuantil 0.9 hasta 78.70 y 0.99 hasta 138.43; como booleana, 2.09, 4.03 y 5.67. En el fold del 24/12 de la guía con 500 de contexto (`exp7.log`): como float, MAE 228.94 pero anchura media del 80 % de 1365.3 y cuantil 0.9 hasta 6512.9; como booleana, MAE 271.90 y anchura 456.3. La protección consiste en ignorar el evento (la categoría no vista recibe la media de la serie), así que la mediana no lo aprovecha.
- **Opciones:** (a) Pasarlas tal cual, con la semántica del backend. (b) Convertirlas a float en el adapter (también `boolean`, con NaN en los NA), para que sean numéricas como en el resto de adapters.
- **Recomendación:** (b), con el aviso de D15 como contrapeso. Una columna booleana significa lo mismo con cualquier modelo de skforecast y, cuando el indicador aparece en el contexto, el resultado medido es igual o mejor (84.24 frente a 87.59 con 16384). El precio es perder la protección de la codificación categórica cuando el indicador no aparece en el contexto (exp4 parte 5 y el fold del 24/12, arriba): como float, la mediana usa el evento, pero los cuantiles altos se disparan. Ese caso es justo el que detecta el aviso de D15, que nombra la columna. Quien prefiera la codificación categórica puede pasar la columna como `category` o texto. La evidencia es escasa (un dataset y casos sintéticos).
- **Qué cambia:** con (a), desaparece `_bool_to_float` y `_build_future_covariates` deja de convertir las booleanas; cambian el párrafo de Notes y el docstring de `supports_categorical_covariates` (vuelven a decir que el backend codifica también las booleanas), y los tests 24 y 37. El helper de D15 puede seguir comprobando las booleanas o saltárselas (con TimeH2O ya no se disparan).
- **Decisión:**

### D9. `batch_size` como parámetro

- **Opciones:** (a) Exponerlo en el constructor (por defecto 256, el del backend), con recarga al cambiarlo. (b) No exponerlo.
- **Recomendación:** (a). Es el único control del backend sobre la memoria y el tamaño de las pasadas (en GPU, la forma de no quedarse sin memoria con muchas series o covariables **[inferido]**) y no cambia la previsión **[probado]**. En CPU, 64 y 256 dan tiempos parecidos y 1.024 es más lento (`exp6.log`). Es lo que en Timestral había que añadir al contrato (`max_series_per_call`); aquí basta un parámetro del adapter. Recargar el modelo al cambiarlo cuesta lo que una carga desde la caché (0,7 s en CPU).
- **Qué cambia:** con (b), se quita de la firma, de `get_params`, de los reinicios de `set_params`, de la llamada a `from_pretrained` y de los tests 1, 3, 8, 9, 12 y 30.
- **Decisión:**

### D10. MPS y `device`

- **Opciones:** (a) `_resolve_torch_device(self.device, mps_fallback_message=..., stacklevel=6)` del PR #1450: si sale MPS (con `"auto"` o explícito) avisa y usa CPU, como Moirai y la propuesta de Timestral. (b) Pasar `device` tal cual al backend: `"auto"` de `h2ofm` nunca elige MPS (CUDA o CPU, sin aviso) y `"mps"` lanza el `ValueError` del backend. (La opción c de versiones anteriores de este documento, resolver MPS en el adapter si el PR #1450 no se fusionaba, ya no aplica: está fusionado.)
- **Recomendación:** (a), por coherencia con Moirai y con el `device="auto"` documentado en skforecast (CUDA > MPS > CPU). En un Mac con `"auto"`, el aviso sale una vez por carga del modelo.
- **Qué cambia:** con (b), `_load_model` pasa `device=self.device` y el test 32 espera, con `"auto"` en un Mac simulado, CPU sin aviso y, con `"mps"`, el `ValueError` del backend (un `from_pretrained` falso que lo imite).
- **Decisión:**

### D11. Valores infinitos

- **Opciones:** (a) Dejar que el backend lance su `ValueError` ("may contain NaN, but not infinity"). (b) Convertirlos a NaN en el adapter, como propone el informe de Timestral para las covariables.
- **Recomendación:** (a). El error del backend es explícito; Timestral los convertía porque su backend los tomaba como valores sin avisar. Convertirlos sería preprocesado propio de un backend.
- **Qué cambia:** con (b), un `np.where(np.isinf(...))` en el contexto y en las filas numéricas, y su test.
- **Decisión:**

### D12. Checkpoint tiny y `allow_exog`

- **Opciones:** (a) Un solo adapter con `allow_exog=True`, y documentar que el tiny ignora las covariables (aviso del backend). (b) Una subclase registrada antes que el prefijo general (`"h2oai/timeh2o-v1-tiny"`, porque `_resolve_adapter` devuelve la primera coincidencia) con `allow_exog=False`, para que `FoundationModel` avise con `IgnoredArgumentWarning`.
- **Recomendación:** (a). `requires_hf_auth` ya se declara por adapter y no por checkpoint; una clase solo para el tiny añade otro adapter a todas las listas por un aviso que el backend ya da.
- **Qué cambia:** con (b), una clase más, una entrada más en el registro y en todas las listas de las secciones 8 y 9.
- **Decisión:**

### D13. Modelo del smoke test de `tools/foundation_compat`

- **Opciones:** (a) `"h2oai/timeh2o-v1"`, el `default_model_id`. (b) `"h2oai/timeh2o-v1-tiny"` (23 MB).
- **Recomendación:** (a). Sigue la convención (todas las entradas usan el `default_model_id`) y el smoke test con exog pasa por las covariables de verdad; con el tiny pasaría ignorándolas. Los pesos quedan en la caché de Actions.
- **Qué cambia:** solo `model_id` de `[timeh2o]`. En los dos casos, la cuenta de `HF_TOKEN` debe aceptar las condiciones del checkpoint elegido.
- **Decisión:**

### D14. Orden respecto al PR del adapter de Timestral

- **Contexto:** el PR #1450, del que dependían los dos adapters, está fusionado (`fc7efd0`). Queda el orden con el PR del adapter de Timestral (`feature/timestral-adapter`), que toca los mismos sitios (registro, `_MODEL_LICENSES`, tests que enumeran los adapters, `tools/foundation_compat/`, documentación y contexto de IA) y necesita el mismo aviso de covariables planas (D15).
- **Opciones:** (a) Independientes: el que esté listo primero se fusiona primero y añade el helper de D15; el segundo hace `git merge origin/0.27.x`, resuelve los conflictos (en listas y tablas se quedan las dos filas) y usa el helper (sustituyendo el método de Timestral, si ya estaba). (b) Esperar a que se fusione el de Timestral y construir este encima. (c) Un solo PR con los dos adapters y el helper.
- **Recomendación:** (a). Ninguno bloquea al otro, los conflictos son de filas añadidas, y (c) mezclaría dos revisiones de licencia distintas (D1) en un PR.
- **Qué cambia:** con (b), el helper y sus tests salen del paso 1 de la sección 7 si el PR de Timestral ya lo añadió (si conserva su método, este PR lo sustituye por el helper, como dice D15); el objeto falso y la entrada de licencia siguen en el paso 1, y el esqueleto va tras `TimestralAdapter`. Con (c), una sola release note con dos entradas y un PR el doble de grande.
- **Decisión:**

### D15. Covariables futuras planas en el contexto

- **Contexto:** sección 2, Covariables (`exp4.log`, `exp7.log`, `exp8.log`, `exp9.log`). Timestral tiene el mismo problema y su informe proponía un `FeatureOutOfRangeWarning` desde un método del adapter, solo para covariables exactamente constantes.
- **Opciones:**
  - (a) Solo documentarlo.
  - (b) Avisar con `FeatureOutOfRangeWarning` desde un helper compartido en `_utils.py` (`_warn_if_flat_known_covariates`, sección 4) que el adapter llama antes de `forecast_many` (solo con checkpoints que usan covariables), cuando algún valor futuro de una covariable numérica o booleana se aleja de la media de su contexto más de 20 desviaciones típicas del contexto (o el contexto es constante y el futuro cambia). Timestral usaría el mismo helper.
  - (c) Lo mismo que (b), con otro umbral: 5 avisa también con el precio de 6 desviaciones, pero da falsas alarmas en los tres folds de Navidad de la guía con contexto largo; 10 avisa en dos de ellos con 4096 de contexto.
  - (d) Centralizar la comprobación en `FoundationModel`, con un atributo nuevo del contrato que la active. Es la opción que `.claude/rules/foundation.md` pide presentar como recomendación.
  - (e) Que el adapter neutralice la covariable (por ejemplo, la historia a NaN): con TimeH2O equivale a quitarla, sin decírselo al usuario.
- **Recomendación:** (b), apartándose a sabiendas de la regla de `.claude/rules/foundation.md` (sección 4, Cambios de código compartido). El aviso es barato, avisa en los casos graves de 32 desviaciones o más (contexto plano, casi plano, indicador visto una vez) y no salta en la guía con contexto largo; no avisa en el precio de 6 desviaciones, que multiplica por 9 el cuantil 0.9 (sección 2, Covariables). El helper evita una segunda copia del bucle cuando llegue Timestral, como `_warn_if_non_commercial`. Frente a (d): el riesgo depende del backend y del checkpoint (el tiny ignora las covariables y en los demás backends no está medido), el remedio del mensaje es propio de cada backend, y (d) añadiría un atributo obligatorio a todos los adapters por un aviso de dos backends; si aparece un tercero, se puede mover a `FoundationModel`. (e) es un preprocesado propio del backend que cambia los datos del usuario.
- **Qué cambia:** con (a), desaparecen el helper, su llamada, el párrafo de Notes y los tests 35, 36 y del helper. Con (c), el valor por defecto de `max_deviation`. Con (d), un atributo más en `_REQUIRED_CLASS_ATTRIBUTES` (como hizo el PR #1450 con `requires_equal_length_context`: la tupla de `_AdapterBase`, todos los adapters y el mensaje esperado de `test_AdapterBase.py`) y la llamada pasa al bucle de grupos de `FoundationModel.predict`. Coordinación con Timestral: el PR que llegue segundo usa el helper del primero; si el de Timestral ya tiene su método, este PR lo sustituye por el helper.
- **Decisión:**

## 7. Plan de implementación

Pasos en orden. Cada paso indica los archivos, los símbolos, el cambio y los tests. Ramas desde `origin/0.27.x` (o la `X.Y.x` vigente) con nombre `<type>/<topic>`; los PR van contra la rama de release, nunca contra `main`.

### Requisito previo: PR #1450

Cumplido: el PR #1450 está fusionado en `0.27.x` (`fc7efd0`, 05/11/2026). El adapter declara el atributo obligatorio `requires_equal_length_context` (añadido a `_REQUIRED_CLASS_ATTRIBUTES` por ese PR) y usa `_resolve_torch_device(device, mps_fallback_message=..., stacklevel=...)`. El único cambio compartido nuevo es el helper `_warn_if_flat_known_covariates` (sección 4, decisión D15), que va en el PR del adapter (o en el de Timestral, si llega antes; decisión D14).

### PR del adapter (rama propuesta `feature/timeh2o-adapter`)

La rama se crea desde `origin/0.27.x`, que ya incluye el PR #1450. Para traer cambios posteriores de `0.27.x`, `git merge origin/0.27.x` y un push normal: nunca rebase, porque una rama publicada necesitaría un force push, que `push_guard.py` bloquea.

**Coordinación con Timestral.** El PR del adapter de Timestral (rama `feature/timestral-adapter`, el PR B del plan de su informe) toca los mismos sitios: el registro, `_MODEL_LICENSES`, los tests que enumeran los adapters, `tools/foundation_compat/`, la documentación y las fuentes del contexto de IA. El que se fusione segundo hace `git merge origin/0.27.x`, resuelve los conflictos (listas y tablas: se añaden las dos filas) y repite `verify`. El contador de `docs/overrides/home.html` (`+5`) sube uno por cada adapter.

**Acción previa de los mantenedores.** El secreto `HF_TOKEN` del repositorio (cuenta dedicada, de solo lectura) debe aceptar las condiciones de `h2oai/timeh2o-v1` (y del tiny, si se usa) en Hugging Face antes de ejecutar el workflow `foundation-backends.yml` con `timeh2o`; si no, el smoke test falla al descargar los pesos. Las pruebas automáticas son un "Non-Commercial Purpose" según la cláusula 1.4 de la licencia **[inferido]**. Si D1 es (b), el correo a H2O lo envía el responsable que fije la decisión antes de fusionar el PR, y la release espera a su respuesta o al plazo fijado.

**Comprobación de D3.** Al empezar el paso 2, antes de escribir `_load_model`, ver qué error recibe un usuario sin credenciales, llamando directamente a `TimeH2O.from_pretrained` con una caché vacía (procedimiento en D3): el adapter aún no está registrado, así que `FoundationModel` no sirve todavía.

| Paso | Archivo y símbolo | Cambio | Tests |
|:-----|:------------------|:-------|:------|
| 1 | `skforecast/foundation/tests/tests_foundation_models/fixtures_adapters.py`; `skforecast/foundation/_utils.py`: `_MODEL_LICENSES`, `_warn_if_flat_known_covariates` (nuevo) y la importación de `FeatureOutOfRangeWarning` | `FakeTimeH2O` y `_FakeTimeH2OForecast` (sección 8, borrador en `experiments/skeleton/fake_timeh2o.py`); la entrada de licencia y el helper de la sección 4 (si el PR de Timestral ya lo añadió, se reutiliza) | `test_utils.py`: los tests del helper (sección 8) |
| 2 | `skforecast/foundation/_adapters.py`: `TimeH2OAdapter` y `_ADAPTER_REGISTRY`; los tests que enumeran los adapters; `test_TimeH2OAdapter.py` (nuevo); `tools/foundation_compat/adapters.toml`, `tools/foundation_compat/tests/contracts.py`, `tools/foundation_compat/README.md`, `.github/workflows/foundation-backends.yml`; `tools/check_foundation_models_metadata.py` (`EXTRA_MODEL_IDS`) | En un solo paso (si se registra la clase sin licencia, sin `adapters.toml` o sin la matriz, los tests que recorren el registro se ponen en rojo): el esqueleto de la sección 4 tras `NoriAdapter` (o tras `TimestralAdapter`, si ya está), la entrada `"h2oai/timeh2o": TimeH2OAdapter` al final del registro y las entradas de la sección 8 | `test_TimeH2OAdapter.py`, los tests de la sección 8; el workflow `foundation-models-metadata.yml` (corre en los PR a `*.x` que tocan `_adapters.py` o `_utils.py`) comprueba licencia y gating contra el Hub |
| 3 | `skforecast/foundation/_foundation_model.py`: docstrings de `FoundationModel` y de sus propiedades | Sección 9 | No aplica |
| 4 | Documentación y fuentes del contexto de IA | Sección 9 | Checks de docs del `verify` |
| 5 | `docs/user_guides/foundation-forecasting-models.ipynb` y `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` | Sección 9. El primero ejecuta todos los modelos de la guía y necesita todos sus backends (uni2ts con `scipy>=1.12` como override, `TABPFN_TOKEN`, y ahora un token de Hugging Face con las condiciones de TimeH2O aceptadas y **Python >= 3.12**). Dos opciones, como en el plan de Timestral: (a) un entorno con todos los backends y `python tools/docs/execute_notebooks/execute_notebooks.py user_guides/foundation-forecasting-models.ipynb` (está en `EXCLUDE_NOTEBOOKS`, hay que pasarlo explícitamente); (b) ejecutar en un kernel las celdas de preparación, las nuevas de TimeH2O y las de código cuya salida cambia (13, `list_adapters()`, y 102; la 45 solo cambia el código; índices de la sección 9), conservando el resto de salidas y reconstruyendo `metrics_*` con los valores guardados, y decirlo en el PR. Incluir la versión evaluada (revisión del Hub y de `h2ofm`), la fecha y la metodología junto a los resultados (cláusula 4 de la licencia, decisión D1) | `execute_notebooks.py --check` en verde |
| 6 | Archivos generados | `python tools/ai/generate_ai_context_files.py` y después `--check` (skill `ai-context-sync`) | `ai-context-check` |
| 7 | `docs/releases/releases.md` | Entrada en la 0.27.0 (skill `release-note`, sección 9) | No aplica |

Antes de fusionar: `git merge origin/0.27.x` en la rama y repetir `verify` (un atributo obligatorio nuevo o una fila de test de otro adapter no dan conflicto en git, pero rompen los tests). Lanzar a mano `foundation-backends.yml` sobre la rama (`workflow_dispatch`) y poner el resultado en la sección de verificación del PR.

## 8. Plan de tests

Basado en cómo están escritos los tests de los adapters actuales **[verificado en código]**. Ningún test descarga pesos: todos usan objetos falsos de `fixtures_adapters.py`, inyectados con `model=` o asignando `adapter._model` (`ForecasterFoundation.__init__` clona el estimador y pierde el `model=` inyectado, así que en los tests a través del forecaster hay que asignar `forecaster.estimator.adapter._model = fake` después de crearlo, como hace `make_forecaster` en `fixtures_forecaster_foundation.py`). Los tests nunca importan `h2ofm`, así que corren en Python 3.10 y 3.11.

### Objeto falso: `FakeTimeH2O` (en `fixtures_adapters.py`)

Sigue a `FakeT0Forecaster` (salida cuyo último eje son los niveles) y a `FakeTimesFM3Forecaster` (registro de la última llamada y de todas en `calls`, comprobación de formas). Borrador probado con el esqueleto en `experiments/skeleton/fake_timeh2o.py` (chequeos `fake: ...` de `run_checks.py`); al pasarlo a `fixtures_adapters.py` se copian las dos clases y se añade `import types` a los imports de ese módulo, que no lo importa:

- `_FakeTimeH2OForecast(horizon, quantile_levels)`: `quantiles` de forma `(horizon, Q)` con la columna `i` igual a `quantile_levels[i]`, `quantile_levels` como `tuple`, `point` y `mean` a 0.5.
- `FakeTimeH2O(batch_size=256, device="cpu", variate_attention=True)`:
  - `config`: un `types.SimpleNamespace(variate_attention=...)`, porque el adapter lee `self._model.config.variate_attention` para saltarse el aviso de D15 con checkpoints que ignoran las covariables (el tiny).
  - `@classmethod from_pretrained(cls, pretrained_model_name_or_path, *, device="auto", revision=None, token=None, cache_dir=None, local_files_only=False, batch_size=256)`: la firma del backend; devuelve `cls(batch_size=batch_size, device=device)`. Los tests de device lo subclasifican para registrar `device`, como `_RecordingForecaster` en `test_TimesFM3Adapter.py`.
  - `forecast_many(self, contexts, horizon, *, past_covariates=None, future_covariates=None, quantile_levels=(0.1, ..., 0.9), flip_equivariant=True, clip_nonnegative=False, snap_integer=False, context_length=None, autoregressive=None, patch_ensemble=True)`: la firma del backend sin `**kwargs`, para que una clave desconocida de `predict_kwargs` dé `TypeError` como con el backend real.
  - Como el backend: `ValueError` si los niveles no son distintos o no están en (0, 1), si una serie no es 1-D numérica o tiene infinitos, si `future_covariates` no tiene una entrada por serie o si una fila no mide `historia + horizonte`.
  - Registra `last_contexts`, `last_horizon`, `last_future_covariates`, `last_past_covariates`, `last_quantile_levels`, `last_kwargs` y `calls`.

### `test_TimeH2OAdapter.py` (nuevo)

Misma estructura que `test_TimesFM3Adapter.py`. Imports: `re`, `sys`, `types`, `warnings`, `pytest`, `numpy` y `pandas`; `TimeH2OAdapter` de `skforecast.foundation._adapters`; `FoundationModel` (tests 27, 28, 34, 35 y 37) y `ForecasterFoundation` (tests 34 y 35) de `skforecast.foundation`; `LicenseWarning`, `IgnoredArgumentWarning` (test 28) y `FeatureOutOfRangeWarning` (tests 35 y 36) de `skforecast.exceptions`; `FakeTimeH2O` y los fixtures de `fixtures_adapters.py`; si el torch del test 32 se simula como en `test_device_handling.py`, también `MagicMock` y `patch` de `unittest.mock`. Helper `make_adapter(**kwargs)` con `model=FakeTimeH2O()`.

| # | Test | Qué comprueba |
|--:|:-----|:--------------|
| 1 | `test_TimeH2OAdapter_init_default_params` | `context_length == 16384`, `device == "auto"`, `batch_size == 256`, `predict_kwargs == {}`, `_model`, `context_` y `context_exog_` a `None`, `is_fitted` `False`; atributos de clase (exog, categóricas, heterogéneas, NaN y `requires_hf_auth` `True`; solo pasadas y `requires_equal_length_context` `False`; `SUPPORTED_QUANTILES is None`; `backend_package == "h2ofm>=0.43"`) |
| 2 | `test_TimeH2OAdapter_init_ValueError_when_invalid_context_length` | `[0, -1, None, "a"]` |
| 3 | `test_TimeH2OAdapter_init_ValueError_when_invalid_batch_size` | `[0, -1, None, 2.5]` |
| 4 | `test_TimeH2OAdapter_init_ValueError_when_model_id_not_timeh2o` | `"h2oai/h2ogpt-4096-llama2-7b"`, `"autogluon/chronos-2-small"`, `"mistralai/Timestral-28M"`, con el mensaje exacto de `_validate_model_id_prefix` |
| 5 | `test_TimeH2OAdapter_init_TypeError_when_loader_only_kwargs` | `revision`, `token`, `cache_dir`, `torch_dtype`: "unexpected keyword argument" |
| 6 | `test_TimeH2OAdapter_init_ValueError_when_predict_kwargs_has_reserved_key` | Parametrizado con las 6 claves reservadas |
| 7 | `test_TimeH2OAdapter_init_predict_kwargs_stored_by_reference` | `{"patch_ensemble": False}` es el mismo objeto |
| 8 | `test_TimeH2OAdapter_get_params_returns_expected_keys_and_values` | Claves `{model_id, context_length, device, batch_size, predict_kwargs}`; `predict_kwargs` `None` cuando está vacío |
| 9 | `test_TimeH2OAdapter_set_params_ValueError_when_invalid` | `context_length=-1`, `batch_size=0`, parámetros desconocidos (`revision`) |
| 10 | `test_TimeH2OAdapter_set_params_ValueError_when_model_id_not_timeh2o` | El adapter queda intacto |
| 11 | `test_TimeH2OAdapter_set_params_ValueError_when_predict_kwargs_has_reserved_key` | `predict_kwargs={"future_covariates": ...}` |
| 12 | `test_TimeH2OAdapter_set_params_updates_and_resets_model` | `model_id`, `device` y `batch_size`: el mismo valor no reinicia; uno nuevo pone `_model` a `None` y devuelve `self` |
| 13 | `test_TimeH2OAdapter_set_params_no_reset_for_non_reload_keys` | `context_length` y `predict_kwargs` conservan `_model` |
| 14, 15 | `test_TimeH2OAdapter_fit_output_single_series`, `..._multi_series` | Copiados de TimesFM3 |
| 16 | `test_TimeH2OAdapter_predict_point_forecast_single_series` | Forma `(12, 1)` con valor 0.5; `last_quantile_levels == [0.5]`; `last_contexts[0]` es float64 1-D de la longitud del contexto; `future_covariates is None` |
| 17 | `test_TimeH2OAdapter_predict_quantile_forecast_single_series` | `[0.9, 0.01, 0.5]` devuelve las columnas en ese orden; el backend recibe `[0.01, 0.5, 0.9]` |
| 18 | `test_TimeH2OAdapter_predict_duplicated_quantiles` | `[0.5, 0.1, 0.5]`: el backend recibe `[0.1, 0.5]`; columnas 1 y 3 iguales |
| 19 | `test_TimeH2OAdapter_predict_point_and_quantile_multi_series` | Un array por serie, en el orden de `context`; una sola llamada |
| 20 | `test_TimeH2OAdapter_predict_no_horizon_ceiling` | `steps=2000` da `(2000, 1)` |
| 21 | `test_TimeH2OAdapter_predict_forwards_predict_kwargs` | `last_kwargs["patch_ensemble"] is False` y `clip_nonnegative is True` |
| 22 | `test_TimeH2OAdapter_predict_future_covariates_history_then_future` | Una fila por columna futura, en el orden de `get_exog_signature`; cada fila es una `pandas.Series` de `ctx + steps` valores, histórico seguido del futuro |
| 23 | `test_TimeH2OAdapter_predict_past_only_columns_not_forwarded` | Una columna solo en `context_exog`: no aparece en las filas y `past_covariates is None` |
| 24 | `test_TimeH2OAdapter_predict_covariate_dtypes` | float, int, `Int64` con NA, bool, `boolean` con NA, `category` y texto: las booleanas (también `boolean`, con NaN en los NA) llegan como float 0/1; el resto conserva su dtype, que decide en `h2ofm` si la fila es numérica o de etiquetas |
| 25 | `test_TimeH2OAdapter_predict_series_without_exog_get_None` | `exog={"a": df, "b": None}`: la entrada de `b` es `None`; con todas `None`, `future_covariates is None` |
| 26 | `test_TimeH2OAdapter_predict_series_without_exog_key` | `context_exog={}` y `exog={}` (camino del backtesting): `future_covariates is None`, sin `KeyError` |
| 27 | `test_FoundationModel_TimeH2O_predict_heterogeneous_series_in_one_call` | A través de `FoundationModel(..., model=fake)` con series de distinta longitud y columnas distintas (como `_make_heterogeneous_inputs` de `test_predict.py`): una sola llamada, sin relleno (cada contexto con su longitud), filas por serie `[2, 1, None]` |
| 28 | `test_FoundationModel_TimeH2O_past_only_columns_IgnoredArgumentWarning` | El aviso de `FoundationModel` y ninguna fila pasada |
| 29 | `test_TimeH2OAdapter_predict_ValueError_when_quantile_out_of_range` | `[0.0]` y `[1.0]`: el `ValueError` del backend (fake) se propaga |
| 30 | `test_TimeH2OAdapter_load_model_LicenseWarning_and_forwards_device_and_batch_size` | Módulo `h2ofm` falso con `types.ModuleType` y `monkeypatch.setitem(sys.modules, ...)`; `pytest.warns(LicenseWarning, match="H2O Model Non-Commercial Licence")` con el URL del `LICENSE`; `from_pretrained` recibe `device="cpu"` y `batch_size=8`; una segunda llamada no avisa |
| 31 | `test_TimeH2OAdapter_load_model_ImportError_when_h2ofm_not_installed_no_LicenseWarning` | `monkeypatch.setitem(sys.modules, "h2ofm", None)`: `ImportError` con `re.escape('pip install "h2ofm>=0.43"')` y "Python >= 3.12", sin `LicenseWarning` |
| 32 | `test_TimeH2OAdapter_load_model_mps_falls_back_to_cpu_with_warning` | Torch simulado (cuda `False`, mps `True`): con `"auto"`, `"mps"` y `"mps:0"`, `from_pretrained` recibe `"cpu"` y sale el `UserWarning` "MPS device is not supported by TimeH2O" |
| 33 | `test_TimeH2OAdapter_load_model_noop_when_already_set` | `_model is fake` tras `_load_model()` |
| 34 | `test_ForecasterFoundation_TimeH2O_predict_interval_any_level` | Parametrizado con `[0.1, 0.9]`, `[0.05, 0.95]`, `[0.01, 0.99]`, `interval=0.9` y `interval=0.98` (desde la 0.23.0, `interval` va en cuantiles) |
| 35 | `test_ForecasterFoundation_TimeH2O_predict_FeatureOutOfRangeWarning_when_known_covariate_flat_in_context` | Precio constante en el contexto que cambia en el horizonte: un aviso que nombra `(serie, columna)` y cuyo `filename` es el del test (`stacklevel=5`); sin aviso si el futuro no cambia, si la columna varía o si es de texto |
| 36 | `test_TimeH2OAdapter_predict_no_FeatureOutOfRangeWarning_when_checkpoint_ignores_covariates` | Con `FakeTimeH2O(variate_attention=False)` y la misma covariable plana: ningún `FeatureOutOfRangeWarning` |
| 37 | `test_FoundationModel_TimeH2O_predict_bool_column_with_missing_context_timestamp` | Columna `bool` con una fecha del contexto que falta en `context_exog`: `FoundationModel` la rellena y llega como `object` (`False`, `True` y NaN); la fila llega al backend como float 0/1 con un NaN |

Tests del helper en `test_utils.py` (sus casos están probados en `run_checks.py`, bloque 5b, a través del adapter o llamando al helper directamente):

| Test | Qué comprueba |
|:-----|:--------------|
| `test_warn_if_flat_known_covariates_warns_when_constant_context_changes` | Constante en el contexto y distinto en el horizonte: `FeatureOutOfRangeWarning` con `(serie, columna)`, el nombre del modelo y el remedio |
| `test_warn_if_flat_known_covariates_warns_when_nearly_constant` | Ruido de 1e-4 y salto de 0.06: avisa |
| `test_warn_if_flat_known_covariates_bool_column` | Booleana a `False` en todo el contexto y `True` en el horizonte: avisa |
| `test_warn_if_flat_known_covariates_no_warning` | Parametrizado: covariable que varía, indicador con 50 unos en 500 pasos (3 desviaciones), futuro igual al contexto, historia entera a NaN, columna de texto o `category`, columna de texto en el contexto y numérica en el futuro, columna solo futura o solo pasada, `context_exog` o `exog` `None` (también el de una serie) |
| `test_warn_if_flat_known_covariates_object_bool_column` | Columna `object` con `False` y un valor ausente en el contexto y `True` en el horizonte: avisa (se comprueba como 0/1) |
| `test_warn_if_flat_known_covariates_max_deviation` | El umbral: con `max_deviation=5`, un salto de 6 desviaciones avisa; con 20, no |

### Tests existentes que enumeran los adapters

| Archivo | Símbolo y cambio |
|:--------|:-----------------|
| `test_resolve_adapter.py` | Importar `TimeH2OAdapter`; `"h2oai/timeh2o-v1"` y `"h2oai/timeh2o-v1-tiny"` en `test_resolve_adapter_returns_correct_class`; `"h2oai/timeh2o": TimeH2OAdapter` al `expected` de `test_ADAPTER_REGISTRY_contains_all_expected_entries` y, como para TimesFM, `assert _ADAPTER_REGISTRY[TimeH2OAdapter._MODEL_ID_PREFIX] is TimeH2OAdapter` (el test comprueba un subconjunto: no falla si falta una entrada). Opcional: `"h2oai/h2ogpt-4096-llama2-7b"` en `test_resolve_adapter_ValueError_when_unknown_prefix` |
| `test_list_adapters.py` | `("TimeH2OAdapter", "h2oai/timeh2o-v1")` al final del `expected` de `test_list_adapters_output`. Los otros dos tests recorren el registro |
| `test_get_model_info.py` | Filas en `test_get_model_info_output_for_each_adapter` (`("TimeH2OAdapter", 16384, "h2ofm>=0.43", True, True, True)`: es la primera fila con `requires_hf_auth` a `True`), `test_get_model_info_supported_quantiles` (`None`), `test_get_model_info_license_fields_match_LicenseWarning_registry` (grupo `True`), `test_get_model_info_license_url` (URL fijo del `LICENSE` de v1, también para el tiny) y `test_get_model_info_weights_location_for_each_adapter` (`(model_id, True, False)`); `"h2ofm"` en `backends` de `test_get_model_info_does_not_import_backend_libraries`. Recomendado: `test_get_model_info_output_TimeH2O` como `test_get_model_info_output_TimesFM3`, con el tiny, que cubre a la vez el URL fijo del `LICENSE` de v1 y `weights_repo_id == model_id` (`get_model_info` resuelve el `None` de la clase al `model_id`) |
| `test_check_exog_columns.py` | Importar; `"h2oai/timeh2o-v1"` en `_EXOG_ADAPTERS` y `_FUTURE_ONLY_ADAPTERS` (no en `_PAST_ONLY_ADAPTERS`); `(TimeH2OAdapter, False)` en `test_adapters_declare_supports_past_only_covariates` |
| `test_clone.py` | `"h2oai/timeh2o-v1"` en `test_clone_round_trip_every_adapter_default_params` y `("h2oai/timeh2o-v1", {"predict_kwargs": {"patch_ensemble": False}})` en `test_clone_round_trip_with_non_empty_config_dict`, con el id `"timeh2o"` en las dos listas `ids=[...]` |
| `test_utils.py` | `"h2oai/timeh2o-v1"` (id `"timeh2o"`) en `test_warn_if_non_commercial_warns_for_registered_prefixes` |
| `test_set_params.py` | Opcional: `("h2oai/timeh2o-v1", "is served by TimeH2OAdapter")` en `test_set_params_ValueError_when_model_id_served_by_other_adapter`, con su id |
| `test_predict.py` | Opcional: el test 27 ya cubre el lote heterogéneo |
| `tests_forecaster_foundation/test_init.py` | `("h2oai/timeh2o-v1", True)` en `test_init_tag_supports_categorical_features_follows_adapter` |
| `test_AdapterBase.py`, `test_backends_compat_config.py` | Sin cambios: recorren el registro, `adapters.toml` y la matriz, y fallan si falta alguna entrada |
| `test_device_handling.py`, `test_fit.py`, `test_get_params.py`, `test_init.py`, `test_repr.py`, `test_prepare_future_exog.py`, tests de `model_selection` | Sin cambios |

### `tools/foundation_compat` y CI

`tools/foundation_compat/adapters.toml` (decisión D13):

```toml
[timeh2o]
adapter = "TimeH2OAdapter"
backend_package = "h2ofm>=0.43"
model_id = "h2oai/timeh2o-v1"
# Gated repository: the account of HF_TOKEN must accept its conditions.
secrets = ["HF_TOKEN"]
```

`backend_package` tiene que ser idéntico al atributo de la clase (`test_adapters_toml_entries_match_the_adapter_classes`). Con `secrets`, `run.py` marca el smoke test como omitido cuando falta el token (por ejemplo, en PR desde forks), como hace con TabPFN; el contract test no lo necesita.

`tools/foundation_compat/tests/contracts.py`, en `BACKEND_CONTRACTS` (comprobado contra `h2ofm` 0.43.1 con `inspect.signature(...).bind_partial`, como `test_contract.py`, en el bloque 0 de `run_checks.py` **[probado]**):

```python
    "timeh2o": [
        {
            "target": "h2ofm:TimeH2O.from_pretrained",
            "n_positional": 1,
            "kwargs": ["device", "batch_size"],
        },
        {
            "target": "h2ofm:TimeH2O.forecast_many",
            "n_positional": 1,
            "kwargs": ["horizon", "future_covariates", "quantile_levels"],
        },
        {"target": "h2ofm.timeh2o:Forecast", "kwargs": ["quantiles", "quantile_levels"]},
    ],
```

`.github/workflows/foundation-backends.yml`: añadir `timeh2o` al final de la lista de una línea `adapter: [chronos, timesfm25, timesfm3, moirai, tabicl, tabpfn, t0, nori, tsicl]` (`test_workflow_matrix_matches_adapters_toml` la lee con una expresión regular). El workflow usa Python 3.12, compatible con `h2ofm` **[verificado en código]**.

`tools/foundation_compat/README.md`: añadir `h2ofm` a la lista de paquetes del primer párrafo. `HF_TOKEN` ya está en "Secrets of the repository": añadir que su cuenta debe aceptar las condiciones de TimeH2O y que, sin el token (por ejemplo, en PR desde forks), el smoke test de TimeH2O se marca como omitido, como el de TabPFN.

`tools/check_foundation_models_metadata.py`: `"h2oai/timeh2o-v1-tiny"` en `EXTRA_MODEL_IDS` (el v1 ya lo cubre `list_adapters()`). No hace falta cambiar `check_model`: la tarjeta tiene `license_name` y `gated: auto` coincide con `requires_hf_auth=True`. El workflow `foundation-models-metadata.yml` no pasa `HF_TOKEN`: la API del Hub y la página del `LICENSE` son públicas, así que funciona sin token **[probado]** (peticiones sin autenticar).

Fallo previo conocido del check `verify` (heredado del informe de Timestral): `.claude/skills/verify/affected.py` ejecuta `python -m pytest tools/foundation_compat/tests -q` cuando se toca `tools/foundation_compat/`, y ese comando falla con "the following arguments are required: --backend", porque `conftest.py` declara `--backend` como obligatorio **[probado]**. Es un fallo previo: decirlo en el PR. Sustitutos locales, los dos en un entorno con `h2ofm` (Python >= 3.12) y `pytest`: `python -m pytest tools/foundation_compat/tests/test_contract.py --backend timeh2o` (4 tests; sin `h2ofm` fallan 3) y `HF_HUB_OFFLINE=1 python -m pytest tools/foundation_compat/tests/test_smoke.py --backend timeh2o` (6 tests con los pesos reales, unos 10 s), que adelanta el smoke test del workflow **[probado]**. El entorno del backend real de la sección 11 no trae `pytest` (final de esa sección).

## 9. Puntos que hay que tocar

Lista obtenida con `git grep -il` sobre el árbol (sin `dev/`) buscando `TSICLAdapter`, `taharnbl`, `TS-ICL`, `NoriAdapter` y `synthefy` (los dos últimos adapters añadidos), más los archivos que enumeran T0 sin nombrarlos **[verificado en código]**. Es la misma lista que la sección 9 del informe de Timestral, más los puntos propios de TimeH2O (acceso gated, Python 3.12, covariables solo futuras con categóricas nativas). Los tests y `tools/` están en la sección 8.

### Paquete (a mano)

| Archivo | Dónde | Qué añadir |
|:--------|:------|:-----------|
| `skforecast/foundation/_adapters.py` | Clase nueva y `_ADAPTER_REGISTRY` | Paso 2 |
| `skforecast/foundation/_utils.py` | `_MODEL_LICENSES` | Paso 1. Opcional: el docstring de `_warn_if_non_commercial` habla de "non-commercial license", que aquí es exacto |
| `skforecast/foundation/_foundation_model.py` | Docstring de `FoundationModel`: frase inicial ("Currently supports ..."), lista de `model_id` por proveedor (`'h2oai/timeh2o-v1'`, `'h2oai/timeh2o-v1-tiny'`, "supports `exog`; the tiny checkpoint ignores it"), "Commonly used kwargs by model" (`- **H2O.ai TimeH2O** (`TimeH2OAdapter`): `context_length` (int, default 16384), `device` (str, default `'auto'`), `batch_size` (int, default 256), `predict_kwargs` (dict, default None).`), viñeta de Notes sobre el device ("`device` string ...: Moirai, TS-ICL, TimesFM 3.0") y References (model card y PyPI) | TimeH2O en todas |
| `skforecast/foundation/_foundation_model.py` | Docstrings de las propiedades `supports_past_only_covariates` (lista de `False`: "TabICL, TabPFN-TS, TFC-T0, Nori") y `supports_heterogeneous_covariates` (lista de `True`: "TFC-T0, TabPFN-TS, Nori, ...") | TimeH2O en las dos listas |
| `skforecast/model_selection/_search.py` | Docstring de `bayesian_search_foundation`, "Tunable parameters per model" y la nota de recarga | Opcional: `context_length` (sin recarga); `batch_size` no cambia la previsión, no tiene sentido buscarlo. Si se toca, `verify` ejecuta todo `skforecast/model_selection/tests/tests_search` (287 tests, unos 2 minutos) |

`_adapter_base.py`, `_model_info.py`, `__init__.py` y `pyproject.toml` no cambian: las clases de los adapters no se exportan y ningún backend de foundation es un extra de skforecast. `requires_hf_auth` ya existe y se expone en `get_model_info`.

### Documentación (a mano)

| Archivo | Dónde | Qué añadir |
|:--------|:------|:-----------|
| `docs/api/FoundationModel.md` | Bloques mkdocstrings, tras el último adapter | `::: skforecast.foundation._adapters.TimeH2OAdapter` con `options: show_bases: false` |
| `docs/user_guides/foundation-forecasting-models.ipynb` | Índices de `nb["cells"]` desde 0, antes de insertar la sección nueva (que desplaza las celdas 101 y 102). Celda 9 (diagrama HTML con una caja por modelo); celda 11 (tabla "### Supported Foundation Models": modelo, proveedor, `model_id`, covariables, contexto por defecto y máximo, horizonte, previsión puntual, cross learning, backend, instalación con botón de copia, enlaces); celda 13 (salida de `list_adapters()`: volver a ejecutarla, sale la fila de `TimeH2OAdapter`); celda 17 (modelos con exog); celda 45 (`model_ids_allow_exog`); sección nueva bajo "## Other foundation models", como "### EDF-Lab TS-ICL" (celdas 96 a 100), con `metrics_timeh2o`; celda 102 (código `# Comparison of backtesting metrics`, bajo el título "### Model comparison" de la 101, cuyo texto dice "the five foundation models") | Todo. La sección nueva sigue el patrón de TS-ICL (`context_length=500` y `backtesting_foundation(..., suppress_warnings = True)`), así que la fila de la comparación es la de 500 de contexto (MAE 103.06 en `results_exp2.csv`); el texto puede añadir el resultado con el contexto por defecto, 16384 (84.24). Con `suppress_warnings = True` no se ve el aviso de covariables planas: explicarlo en el texto y, para mostrarlo, una celda aparte con `suppress_warnings = False`, en la que salta en el fold del 24/12/2014 (con 16384 de contexto no salta). Además: cómo aceptar las condiciones y autenticarse (`hf auth login` o `HF_TOKEN`), Python >= 3.12, que la licencia excluye el uso en producción, el coste del ensemble de parches en CPU y, junto a la métrica, la versión evaluada (revisión del Hub, `h2ofm`), la fecha y la metodología (cláusula 4 de la licencia) |
| `docs/user_guides/foundation-forecasting-with-heterogeneous-series.ipynb` | Índices desde 0. Celda 2 (tabla "In short", fila sobre el agrupado); celda 11 (lista comentada de `model_id`); celda 13 (modelos con covariables); celda 18 ("## How it works", "Same covariate columns"); celda 21 ("## Reference: requirements and limitations per backend") | TimeH2O no agrupa y acepta categóricas. La frase "Only Chronos-2 accepts string and categorical columns natively" deja de ser cierta |
| `docs/quick-start/ai-assisted-forecasting.md` | Viñeta "Foundation models (zero-shot)" y fila `foundation-forecasting` de la tabla de skills | TimeH2O |
| `docs/overrides/home.html` | Fila `.models` ("Chronos-2 / TimesFM / Moirai-2 / TabPFN-TS / +5") | `+5` pasa a `+6` (o `+7` si Timestral ya está) |
| `README.md`, `docs/introduction-forecasting/introduction-forecasting.md`, `docs/user_guides/global-forecasting-overview.md`, `docs/animations/foundation-models.html`, `tools/docs/hooks/examples.yml`, `tools/docs/hooks/page_meta.yml` | Listas "such as ..." | Opcional |
| `mkdocs.yml` | | Nada |

### Release notes (a mano, skill `release-note`)

| Archivo | Dónde | Qué añadir |
|:--------|:------|:-----------|
| `docs/releases/releases.md` | `## 0.27.0 <small>In development</small>`: destacado en "The main changes in this release are:" y entrada en `**Added**` | Con el formato de TS-ICL y Nori (sección `## 0.24.0`): ``+ <span class="badge text-bg-feature">Feature</span> New <code>[TimeH2OAdapter]</code> in the <code>foundation</code> module wrapping H2O.ai's TimeH2O models, registered under the `'h2oai/timeh2o'` `model_id` prefix. Supports future-known numeric and categorical covariates, series with different exog columns in one call and any quantile level. The weights are gated on the Hugging Face Hub and released under a non-commercial, non-production license; `h2ofm` requires Python >= 3.12. [User guide](../user_guides/foundation-forecasting-models.ipynb)`` y el mismo texto sin la insignia en `**Added**`. El enlace al PR se añade cuando exista (skill `release-note`). La definición `[TimeH2OAdapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.TimeH2OAdapter` va junto a la de `[TSICLAdapter]`, en el grupo `<!-- foundation -->` |
| `docs/releases/releases.md` | Definiciones de enlaces al final | `[TimeH2OAdapter]: ../api/FoundationModel.md#skforecast.foundation._adapters.TimeH2OAdapter` |

### Contexto de IA, fuentes (a mano)

| Archivo | Dónde | Qué añadir |
|:--------|:------|:-----------|
| `tools/ai/llms-base.txt` | Árbol de `## Project Structure`; `### Module Relationships`; fila de ForecasterFoundation en `## Core Forecasters`; en `## Foundation Models (Zero-Shot)`: paquetes, fila de "Supported adapters", viñetas de `LicenseWarning` (añadir TimeH2O a la lista de licencias no comerciales), de covariables solo pasadas (TimeH2O no las usa) y de agrupado; comentario de `info.requires_hf_auth` | Todo |
| `llms.txt` (raíz; fuente de `docs/llms.txt`) | `## Core Forecasters`, `## Foundation Models`, `## API Reference` | TimeH2O |
| `skills/foundation-forecasting/SKILL.md` | `## Common Mistakes`: un punto nuevo sobre covariables futuras planas en el contexto (`FeatureOutOfRangeWarning`, TimeH2O y Timestral; remedio: contexto más largo o quitar la columna); `description`; `## Stop Conditions` (filas de exog, de licencia y una nueva: "weights gated on the Hugging Face Hub, accept the conditions and authenticate"); `## Installation` (`pip install "h2ofm>=0.43"`, Python >= 3.12); `## With Exogenous Variables`; `## Prediction Intervals and Quantiles` (cualquier nivel, extrapolado fuera de [0.1, 0.9]); tabla y licencias de `## Choosing a Model`; `## Common Mistakes`, puntos 3, 6 (lista de librerías: `h2ofm`, con Python >= 3.12), 7 (parámetros que recargan el modelo: `batch_size` en TimeH2O), 8 ("Only Chronos-2 accepts them natively") y 10 (TimeH2O ignora las columnas solo pasadas); el párrafo de exog ("Chronos-2, TimesFM 3.0, TabICL, ... accept exogenous variables" y las listas de solo pasadas y de agrupado); `## References` | Todo |
| `skills/foundation-forecasting/references/adapter-parameters.md` | `## Contents`; sección nueva `## TimeH2OAdapter: H2O.ai TimeH2O`; fila de `## Tunable Parameters and Model Reload Cost` (`batch_size` recarga); la viñeta "`context_length` is not uniformly cheap" (TimeH2O entre los gratuitos); `## Common Behavior`: la frase "`supports_categorical_covariates` is `True` only for Chronos-2" y "`requires_hf_auth` is `True` when the weights are gated on the Hugging Face Hub (currently none of the supported models)", que dejan de ser ciertas; listas de `### Exog column validation at predict time` | Todo |
| `skills/choosing-a-forecaster/SKILL.md` | Filas de ForecasterFoundation en `## Step 2a` y `## Step 2b`; `## Decision Flowchart` | Tres sitios |
| `skills/prediction-intervals/SKILL.md` | Frase de restricción de cuantiles en `## Foundation Model Intervals` | TimeH2O admite cualquier nivel |
| `skills/complete-api-reference/SKILL.md` | Viñeta de ForecasterFoundation en `### Forecaster Constructors` | TimeH2O |
| `skills/complete-api-reference/references/forecaster-constructors.md` | Bloque de `### ForecasterFoundation` | `batch_size, predict_kwargs (TimeH2O)`; la línea "predict_kwargs (TimesFM 3.0 only)" ya no es cierta (Chronos-2 también lo tiene): pasa a "predict_kwargs (Chronos-2, TimesFM 3.0, TimeH2O)" |
| `skills/hyperparameter-optimization/references/search-parameters.md` | Tabla "Worth searching" de `### Foundation Function` | Opcional |
| `tools/ai/README.md` | Fila `foundation-forecasting` de `## Skills` | TimeH2O |

### Contexto de IA, generados (no se editan)

Se regeneran con `python tools/ai/generate_ai_context_files.py` y se comprueban con `--check` (skill `ai-context-sync`): `AGENTS.md`, `.github/copilot-instructions.md`, `llms-full.txt`, `docs/llms-full.txt` y `docs/llms.txt`. El bloque `SKILLS-LIST` de `CLAUDE.md` no cambia.

## 10. Revisión de upstream antes de empezar

Antes de implementar, revisar esto. Estado a 05/11/2026 **[probado]** salvo donde se indica.

| Punto | Dónde mirarlo | Estado el 05/11/2026 | Si ha cambiado |
|:------|:--------------|:---------------------|:---------------|
| Versión de `h2ofm` en PyPI | https://pypi.org/pypi/h2ofm/json | 0.43.1 (27/10/2026); 0.42.0 no carga v1 | Comparar `src/h2ofm/timeh2o/_api.py` del sdist nuevo con 0.43.1 (firmas de `from_pretrained` y `forecast_many`, valores por defecto, `Forecast`); pasar `contracts.py` y `skeleton/run_checks.py`; valorar subir el mínimo de `backend_package` |
| `requires_python` y dependencias | Metadatos de PyPI | `>=3.12`, `torch>=2.10,<3` | Si baja a 3.10, quitar la nota de Python de la documentación y del `ImportError` |
| Repositorio del código | Model card ("GitHub (coming soon)"), PyPI (`project_urls`) | Ninguno público; `github.com/h2oai/h2ofm` es el repositorio de TabH2O, sin el código | Leer el código, los issues y las releases; añadirlo a References y a la documentación |
| Issues | https://github.com/h2oai/h2ofm/issues | 0 issues abiertos | Leer los nuevos que afecten a TimeH2O |
| Licencia de los pesos | https://huggingface.co/h2oai/timeh2o-v1/blob/main/LICENSE (y el del tiny) | H2O Model Non-Commercial Licence v1.0, idéntica en los dos (sha256 `3209e42b...`) | Revisar D1 y D2, la entrada de `_MODEL_LICENSES` y los textos de licencia de la documentación |
| `license_name` y gating | https://huggingface.co/api/models/h2oai/timeh2o-v1 (`cardData`, `gated`) | `h2o-model-non-commercial-licence-v1.0`, `gated: auto` | Ajustar el primer campo de `_MODEL_LICENSES` o `requires_hf_auth` (los compara `check_foundation_models_metadata.py`) |
| Revisión de los pesos | API del Hub, campo `sha` | v1 `1310be7...`, tiny `33e4aa2...` | Repetir `exp2_backtest.py` (v1; ver en la sección 12 la nota sobre el prototipo de Timestral) y actualizar las cifras de la documentación con la versión y la fecha (cláusula 4 de la licencia) |
| Checkpoints nuevos | https://huggingface.co/h2oai (buscar `timeh2o`) | Dos | Añadirlos a `EXTRA_MODEL_IDS` y a la lista de `model_id`; si alguno admite covariables pasadas, revisar `supports_past_only_covariates` (es por adapter) |
| Informe técnico y blog | Model card ("Paper (coming soon)"), https://h2o.ai/blog/2026/timeh2o-foundation-models | Sin informe; blog no leído | Leerlos: datos de entrenamiento, recomendaciones sobre el ensemble de parches y las covariables |
| Leaderboards | `Salesforce/GIFT-Eval` (`results/TimeH2O-v1`), `autogluon/fev-bench` | GIFT-Eval comprobado; fev-bench sin comprobar | Solo afecta al texto sobre los benchmarks |
| Integraciones de terceros | sktime, darts, AutoGluon | No buscadas | Revisar cómo tratan el acceso gated y la licencia |

## 11. Versiones fijadas y entorno

Qué entorno usar se pregunta al empezar (sección 0, paso 1). Esta sección da las versiones con las que se hicieron los experimentos y dos formas de crear un entorno con el backend real si hace falta uno: con uv (Linux, sin conda) o con conda.

### Versiones

| Componente | Versión |
|:-----------|:--------|
| h2ofm | 0.43.1 (PyPI) |
| Pesos en HF | Revisiones de la tabla de la sección 2 |
| Python | 3.12.3 |
| torch | 2.14.1+cpu (wheel de `https://download.pytorch.org/whl/cpu`) |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| huggingface-hub | 2.4.0 |
| safetensors | 0.8.0 |
| scikit-learn | 1.9.1 |
| chronos-forecasting | 2.3.2 (solo referencia en `exp2`) |
| timestral | 1.0.0, commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` (solo referencia en `exp2`) |
| skforecast | 0.27.0, editable: los experimentos `exp1` a `exp9`, con el commit `924ab7f` de `0.27.x`; el esqueleto y `run_checks.py`, con `fc7efd0` (PR #1450 fusionado) |
| Hardware | 4 CPU Intel Xeon (4 hilos de torch), 15 GB de RAM, sin GPU; Linux 6.18 x86_64 |

`dev/timeh2o_adapter/experiments/requirements.txt` fija lo necesario y `requirements-freeze.txt` es el `uv pip freeze` completo del entorno. Los comandos de las dos recetas se ejecutan desde la raíz del repositorio; `timeh2o-exp` es el nombre del entorno de los experimentos: sustituirlo por el indicado en el paso 1 de la sección 0.

### Recrear el entorno con uv (Linux, sin conda)

Para no descargar el wheel con CUDA, se baja el wheel de CPU desde `https://download.pytorch.org/whl/cpu/` y se instala desde el archivo (el wheel de PyPI con CUDA también funciona en CPU, con varios GB de dependencias). Hay que autenticarse en Hugging Face (`hf auth login` o la variable `HF_TOKEN`) con una cuenta que haya aceptado las condiciones de los dos repositorios.

```bash
uv venv ~/.venvs/timeh2o-exp --python 3.12
```

```bash
curl -sS -o /tmp/torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl "https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
```

```bash
uv pip install --python ~/.venvs/timeh2o-exp/bin/python /tmp/torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl -r dev/timeh2o_adapter/experiments/requirements.txt
```

```bash
uv pip install --python ~/.venvs/timeh2o-exp/bin/python -e ".[stats,plotting]"
```

Activarlo antes de `hf auth login` y de los comandos de la sección 12:

```bash
source ~/.venvs/timeh2o-exp/bin/activate
```

### Recrear el entorno en local (conda)

```bash
conda create -n timeh2o-exp python=3.12
```

```bash
conda activate timeh2o-exp
```

En Linux, para no descargar el wheel con CUDA, instalar antes torch desde el índice de CPU; en macOS el wheel de PyPI ya es de CPU (MPS no funciona con `h2ofm` 0.43):

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r dev/timeh2o_adapter/experiments/requirements.txt
```

```bash
pip install -e ".[stats,plotting]"
```

Autenticarse en Hugging Face con una cuenta que haya aceptado las condiciones de `h2oai/timeh2o-v1` y `h2oai/timeh2o-v1-tiny`:

```bash
hf auth login
```

Para el PR del adapter no hace falta este entorno salvo para volver a ejecutar el notebook, `run_checks.py` y los sustitutos locales del check de `tools/foundation_compat` (sección 8): los tests usan objetos falsos. Estos requisitos no instalan lo siguiente, que se añade según el uso con `uv pip install --python ~/.venvs/timeh2o-exp/bin/python <paquetes>` en la receta uv (su entorno no trae `pip`) o con `pip install <paquetes>` en la de conda: `pytest`, para esos sustitutos; `ipykernel`, para el notebook (las dos opciones del paso 5 de la sección 7) y, para la opción a, también `papermill` y los demás backends de la guía; y `ruff`, para el lint de la sección 12 si no se usa el del entorno de desarrollo (el repositorio no fija su versión; el lint pasa con 0.16.9 y con 0.17.4).

## 12. Reproducción de los experimentos

Todo se ejecuta desde `dev/timeh2o_adapter/experiments/`, con el entorno del backend real elegido al empezar (sección 0, paso 1; cómo crearlo, en la sección 11) activado, y una cuenta de Hugging Face que haya aceptado las condiciones de los dos checkpoints; ningún script depende de rutas absolutas. Fuera de esta carpeta solo se usan este documento (que `skeleton/extract_skeleton.py` recibe como argumento), el `pyproject.toml` del repositorio (lint) y, en `exp2_backtest.py`, el prototipo de Timestral (`../../timestral_adapter/experiments/timestral_proto.py`; ver la fila de `exp2`). `exp2`, `exp7` y `exp9` descargan además `vic_electricity` con `fetch_dataset`, así que necesitan red aunque los pesos estén en caché. Los pesos se descargan la primera vez a la caché del Hub (23 MB el tiny, 435 MB v1; Chronos-2 y Timestral-209M para `exp2`). Con los pesos ya en caché, `HF_HUB_OFFLINE=1` evita las consultas al Hub. Tiempos aproximados en CPU con 4 hilos. Al publicar los logs de `exp1` a `exp9` se ha acortado la descripción de la plataforma de su primera línea (a `Linux 6.18.44`, lo que imprime ahora `common.py`) y se han quitado las rutas absolutas de los avisos de `exp3.log`. En `exp2.log`, además, se quitaron las dos filas repetidas de Chronos-2 de la segunda ejecución (anexo A, fila 8) y se añadió a mano la línea `--- second run ... ---` que separa las dos ejecuciones; la primera se lanzó sin `HF_HUB_OFFLINE` y guardando también la salida de errores, por eso incluye barras de progreso. El resto es la salida de la ejecución.

| Archivo | Qué hace | Cómo se ejecuta | Resultado esperado |
|:--------|:---------|:----------------|:-------------------|
| `common.py` | Versiones del entorno, series sintéticas, cronómetro y pico de memoria | Lo importan los scripts | No aplica |
| `timeh2o_proto.py` | Prototipo del adapter registrado en tiempo de ejecución (`register()`), escrito antes del PR #1450 (corre con y sin él) | Lo importan `exp2` a `exp5` | No aplica |
| `exp1_smoke.py` | Backend directo: salida, lote de longitudes distintas frente a cada serie sola, relleno hecho por quien llama, tiempos por llamada de los dos checkpoints con y sin *ensemble* | `HF_HUB_OFFLINE=1 python exp1_smoke.py > exp1.log 2>&1` (unos 8 minutos) | `exp1.log` |
| `exp2_backtest.py` | Backtesting de la guía: TimeH2O (contextos, exog, *ensemble*, booleana, tiny) y referencias (Chronos-2, Timestral-209M) | `HF_HUB_OFFLINE=1 python -W ignore exp2_backtest.py > exp2.log 2>&1` (unos 35 minutos sin descargas; la primera vez, sin `HF_HUB_OFFLINE`) | Añade filas a `results_exp2.csv` y salta las que ya están: para repetirlo desde cero, borrar el CSV antes. El CSV y `exp2.log` salen de dos ejecuciones (la segunda, con las cuatro configuraciones añadidas al final de la lista, va al final del log). Con `0.27.x` desde `fc7efd0` el script falla al importar el prototipo de Timestral, escrito antes del PR #1450, que no declara `requires_equal_length_context` (`TypeError`, antes de medir nada). Para repetirlo, usar skforecast en `924ab7f` (por ejemplo, un `git worktree` instalado en modo editable) o declarar en local `requires_equal_length_context: bool = False` en `TimestralAdapter` de `timestral_proto.py`: el prototipo rellena por su cuenta, así que `False` conserva su comportamiento |
| `exp3_multiseries.py` | Prototipo por `FoundationModel`: lote heterogéneo frente a cada serie sola, llamadas al backend, cuantiles (fuera de la rejilla, repetidos, 0 y 1, `predict_interval`) y coste con 0, 1 y 2 covariables | `HF_HUB_OFFLINE=1 python exp3_multiseries.py > exp3.log 2>&1` | `exp3.log` |
| `exp4_covariates.py` | Covariables: pasadas, booleanas, texto, categóricas, etiqueta no vista, NaN, historia entera a NaN, infinitos, el caso del issue #7 de Timestral y el tiny | `HF_HUB_OFFLINE=1 python exp4_covariates.py > exp4.log 2>&1` | `exp4.log` |
| `exp5_backend_checks.py` | Backend directo: NaN, contexto entero a NaN (también por `FoundationModel`), dtype, determinismo, contexto máximo, horizontes largos, dependencia del horizonte, `batch_size`, dispositivos, `clip_nonnegative`, `snap_integer` y `mean` | `HF_HUB_OFFLINE=1 python exp5_backend_checks.py > exp5.log 2>&1` | `exp5.log` |
| `exp6_large_batches.py` | Relleno interno con longitudes mezcladas, lotes de 100 a 1.000 series y `batch_size` con covariables | `HF_HUB_OFFLINE=1 python exp6_large_batches.py > exp6.log 2>&1` | `exp6.log` |
| `exp7_flat_covariates.py` | Covariables planas: mecanismo (ruido, escalón diminuto), remedios (`patch_ensemble=False`, precio relativo, historia a NaN) y el fold de la guía con `Holiday` plano | `HF_HUB_OFFLINE=1 python -W ignore exp7_flat_covariates.py > exp7.log 2>&1` (unos 12 minutos) | `exp7.log`. El caso del escalón de 1e-7 en el contexto no prueba nada: en float32, 2,5000001 es 2,5 |
| `exp8_flat_threshold.py` | Umbral: precio con ruido de distinta desviación en el contexto e indicador con 1 a 64 unos | `HF_HUB_OFFLINE=1 python -W ignore exp8_flat_threshold.py > exp8.log 2>&1` (unos 20 minutos) | `exp8.log` |
| `exp9_guide_folds.py` | Folds de la guía en los que `Holiday` se separa del contexto más de 5 desviaciones, con 500, 4096 y 16384 de contexto | `HF_HUB_OFFLINE=1 python -W ignore exp9_guide_folds.py > exp9.log 2>&1` (unos 5 minutos) | `exp9.log` |
| `skeleton/extract_skeleton.py` | Extrae de este documento los bloques `<!-- skeleton:... -->` y escribe `shared_changes.py`, `timeh2o_skeleton.py` y `license_entry.py` | `cd skeleton && python extract_skeleton.py ../../research_timeh2o_adapter.md` | Los tres módulos generados (no se guardan) |
| `skeleton/run_checks.py` | 62 chequeos del esqueleto contra el backend real y `0.27.x` desde `fc7efd0` (sección 4 y docstring del script): contrato, firmas de `h2ofm`, carga, cuantiles, covariables, lotes heterogéneos, backtesting, aviso de covariables planas y booleanas `object` | `cd skeleton && HF_HUB_OFFLINE=1 python -W ignore run_checks.py > run_checks.log 2>&1`, después de `extract_skeleton.py` (menos de un minuto). Por defecto comprueba el código de este documento y lo registra en tiempo de ejecución, por encima del adapter del paquete si lo hay; con el adapter ya implementado, `--package` comprueba el del paquete, sin extraer nada (`> run_checks_package.log 2>&1`) | `62/62 checks passed` (`run_checks.log`; `--package`, 62/62 contra la implementación del ensayo del anexo A) |
| `skeleton/fake_timeh2o.py` | Borrador de `FakeTimeH2O` (sección 8), probado con el esqueleto en `run_checks.py` | Referencia para `fixtures_adapters.py` | No aplica |
| Lint | `ruff` sobre los módulos extraídos y los scripts | `cd skeleton && ruff check --no-cache --config ../../../../pyproject.toml shared_changes.py timeh2o_skeleton.py license_entry.py run_checks.py fake_timeh2o.py extract_skeleton.py ../*.py`, con el `ruff` del entorno de desarrollo (paso 1 de la sección 0) o instalado en el del backend como dice el final de la sección 11 | `All checks passed!` |

## 13. Definición de terminado de la implementación

1. Las decisiones de la sección 6 están resueltas y aplicadas (el PR #1450, del que depende el esqueleto, ya está fusionado).
2. Tests de los archivos tocados, de uno en uno y sin `-n`: `pytest skforecast/foundation/tests/tests_foundation_models/test_TimeH2OAdapter.py -x -q` mientras se itera, y al final `pytest skforecast/foundation/tests/ -q` y los tests de los llamantes de `FoundationModel.predict`: `pytest skforecast/model_selection/tests/tests_validation/test_backtesting_foundation.py skforecast/model_selection/tests/tests_search/test_bayesian_search_foundation.py -q`. No lanzar la suite completa.
3. `ruff check` sin errores en los archivos tocados.
4. Skill `verify` en verde antes del commit y antes de abrir el PR (lint, tests afectados, AI context y release notes). El check de `tools/foundation_compat` de `verify` falla por un problema previo (sección 8): decirlo.
5. Release note en `docs/releases/releases.md`, sección 0.27.0 (skill `release-note`).
6. AI context sincronizado: fuentes editadas a mano (sección 9), `python tools/ai/generate_ai_context_files.py` y `--check` en verde (skill `ai-context-sync`).
7. Los dos notebooks de la sección 9 ejecutados como dice el paso 5 de la sección 7, con `python tools/docs/execute_notebooks/execute_notebooks.py --check` en verde, y la versión, la fecha y la metodología junto a los resultados de TimeH2O.
8. `python tools/check_foundation_models_metadata.py` en verde para los dos checkpoints (necesita red).
9. Prueba con el backend real en CI: la cuenta de `HF_TOKEN` ha aceptado las condiciones; `foundation-backends.yml` lanzado a mano sobre la rama, con el resultado en la sección de verificación del PR.
10. `skeleton/run_checks.py --package` en verde (62/62) con el adapter implementado y los pesos reales: sin `--package` comprueba el código de este documento, no el del paquete. Para un backtesting con cobertura, `backtesting_foundation` no tiene `interval`: pedir `quantiles=[0.1, 0.5, 0.9]`.
11. Commits y PR a nombre del autor, contra la rama `X.Y.x`.
12. D3 comprobado sin credenciales (la excepción que recibe el usuario, anotada en el PR) o, si no se ha podido, las dos excepciones documentadas.
13. Si D1 es (b), el correo a H2O enviado antes de fusionar y la release a la espera de su respuesta o del plazo fijado.

## 14. Riesgos

- **Licencia.** Solo permite uso no comercial y no productivo: un usuario no puede usar las previsiones para decisiones reales de negocio, ni siquiera internas, aunque no haya ingresos de por medio. Es más restrictiva que todas las demás registradas en skforecast, revocable en cualquier momento y con una definición de "Derivative" amplia (decisión D1). El `LicenseWarning` lo hace visible al cargar, pero no lo impide. Hay que repetirlo en la documentación, porque es fácil leer "non-commercial" y pensar que el uso interno en una empresa está permitido.
- **Acceso gated.** Primer adapter con `requires_hf_auth=True`: el usuario necesita cuenta en Hugging Face, aceptar las condiciones y un token. El CI necesita que la cuenta de `HF_TOKEN` las acepte, y ejecutar el notebook de la guía, un token también. El mensaje de error sin token no está probado (decisión D3).
- **Python >= 3.12.** skforecast admite 3.10 y 3.11, donde `h2ofm` no se instala. Los tests unitarios no se ven afectados (usan el objeto falso), pero la documentación y el `ImportError` deben decirlo.
- **Backend sin repositorio público del código.** El código (Apache-2.0) solo está en PyPI; `github.com/h2oai/h2ofm` es el repositorio de TabH2O, donde también se reportan los fallos del modelo. Tres versiones en 15 días, con un valor por defecto que cambió entre versiones menores (`flip_equivariant`). El check semanal de `tools/foundation_compat` detectaría una rotura de la API, no un cambio de comportamiento.
- **Coste en CPU.** El *ensemble* de parches, activo por defecto, multiplica el coste con historias cortas: una serie de 512 pasos tarda 1,0 s con el *ensemble* y 0,15 s sin él, y 100 series, 93 s frente a 3,7 s (`exp1.log`). Las covariables multiplican además las filas. En backtesting con muchas series hará falta GPU, `patch_ensemble=False` o el checkpoint tiny (sin covariables).
- **Benchmarks.** Las cifras de GIFT-Eval se reproducen a partir de los resultados del leaderboard, pero son resultados enviados por H2O, no una evaluación independiente; las de fev-bench no las he podido comprobar. Nuestro experimento es un solo dataset, y la mejora con 16.384 de contexto se explica, probablemente, por la estacionalidad anual de `vic_electricity` (sección 2, Experimento) **[inferido]**.
- **Covariables.** Las pasadas se ignoran (avisa `FoundationModel`); las booleanas, que el backend trataría como categóricas, el adapter las pasa como 0/1 (D8), con lo que pierden la protección de la codificación categórica frente a un indicador que no aparece en el contexto (lo señala el aviso de D15); una covariable con toda la historia a NaN se descarta sin avisar; con covariables futuras, la previsión de los primeros pasos depende de `steps`, así que dos backtestings con distinto `steps` no son comparables paso a paso. Sobre todo, **una covariable futura constante o casi constante en el contexto que cambia en el horizonte dispara los cuantiles altos** (sección 2, Covariables): el aviso propuesto (D15) lo hace visible, pero no lo corrige, y su umbral (20 desviaciones típicas) se ha calibrado con un caso sintético y un dataset real; hay casos dañinos por debajo del umbral (un precio con un ruido de 0,01 que salta 0,06: 6 desviaciones, cuantil 0.9 nueve veces mayor) que no avisan.
- **Contexto entero a NaN.** El backend lo sustituye por ceros y devuelve una previsión en torno a 0 sin avisar. Es el mismo riesgo que describe el informe de Timestral para todos los adapters con `supports_nan_in_series=True` (la comprobación de `FoundationModel.fit` es anterior al recorte a `context_length`; el backtesting ya descarta esas ventanas). Propuesta para un issue aparte, no para este PR.
- **`MissingValuesWarning` engañoso.** Con índice de fechas, `FoundationModel` avisa de que ha rellenado con NaN fechas que faltaban en la exog futura siempre que esta tiene algún NaN, aunque no falte ninguna fecha (`_prepare_future_exog` comprueba `isnull()` después de reindexar) **[verificado en código]**. Es previo y ajeno a TimeH2O. Propuesta para un issue aparte, no para este PR.
- **MPS no funciona.** En Apple Silicon solo se puede usar la CPU, con el coste anterior.
- **Avisos del backend.** Los `UserWarning` de `h2ofm` (covariables ignoradas por el tiny, `context_length` por encima del máximo) apuntan a la línea de `_adapters.py` que llama a `forecast_many`, no al código del usuario (`skip_file_prefixes`) **[inferido]**.
- **No leídos:** el blog de H2O; no hay informe técnico ni repositorio del modelo.

## 15. Fuentes

Consultadas el 05/11/2026.

Oficiales:

- Model card de TimeH2O-v1: https://huggingface.co/h2oai/timeh2o-v1 (revisión `1310be7`). `config.json`, `README.md` y `LICENSE` descargados.
- Model card de TimeH2O-v1-tiny: https://huggingface.co/h2oai/timeh2o-v1-tiny (revisión `33e4aa2`).
- Licencia de los pesos: https://huggingface.co/h2oai/timeh2o-v1/blob/main/LICENSE
- Metadatos del Hub (gating, fechas, `license_name`, revisiones, archivos): https://huggingface.co/api/models/h2oai/timeh2o-v1 y https://huggingface.co/api/models/h2oai/timeh2o-v1-tiny
- PyPI: https://pypi.org/project/h2ofm/ (0.42.0, 0.43.0 y 0.43.1; sdists descargados y comparados). Código leído de 0.43.1: `h2ofm/_checkpoints.py`, `_validation.py`, `timeh2o/_api.py`, `_checkpoints.py`, `_warnings.py`, `_data/preprocess.py`, `_data/encoding.py`, `_data/dataset.py`, `_pipeline/pipeline.py`, `_pipeline/decoding.py`, `_pipeline/utils.py`, `_pipeline/tta.py`, `_pipeline/postprocess.py`, `_model/config.py`, `_model/model.py`, `_model/scaler.py`, `_model/_refinement.py`, `_model/flipeq.py` (parcial), README (`METADATA`), `LICENSE` y `THIRD_PARTY.md`.
- Repositorio de GitHub `h2oai/h2ofm` (repositorio de TabH2O, sin el código de `h2ofm`): https://github.com/h2oai/h2ofm (clonado, commit `da6e042`); issues: https://github.com/h2oai/h2ofm/issues (0 abiertos).
- Blog: https://h2o.ai/blog/2026/timeh2o-foundation-models (no leído).
- GIFT-Eval: https://huggingface.co/spaces/Salesforce/GIFT-Eval (`results/TimeH2O-v1`, `results/TimeH2O-v1-tiny`, `results/Timestral-209M`, `results/t0-gamma`, `results/TimesFM-3`, `results/Chronos-2`, `results/Seasonal_Naive`).
- fev-bench: https://huggingface.co/spaces/autogluon/fev-bench (sin los datos de la tabla).
- `huggingface_hub` 2.4.0: `utils/_http.py` y `errors.py` (errores de repositorios gated).

skforecast (commits `924ab7f` y `fc7efd0` de `0.27.x`; el segundo, con el PR #1450): `.claude/rules/foundation.md`, `skforecast/foundation/_adapter_base.py`, `_adapters.py`, `_foundation_model.py`, `_forecaster_foundation.py`, `_model_info.py`, `_utils.py`, `skforecast/foundation/tests/`, `tools/foundation_compat/`, `tools/check_foundation_models_metadata.py`, `.github/workflows/foundation-backends.yml` y `foundation-models-metadata.yml`, `pyproject.toml`, `skills/foundation-forecasting/`, `tools/ai/llms-base.txt`, los notebooks de foundation de `docs/user_guides/` y `dev/timestral_adapter/`.

## Anexo A. Revisión final del 05/11/2026

Afirmaciones revisadas contra el código de `h2ofm` 0.43.1, el código de skforecast (`0.27.x` y PR #1450), los logs y la ejecución del esqueleto, y cómo quedan. Las filas 1 a 17 son de mi propia revisión durante la redacción; de la 18 a la 50, de una segunda revisión por temas; de la 51 en adelante, de una tercera revisión y de un ensayo de implementación.

**Revisión propia durante la redacción**

| # | Afirmación revisada | Resultado | Dónde se corrigió |
|--:|:--------------------|:----------|:------------------|
| 1 | El repositorio `github.com/h2oai/h2ofm` es el del código de `h2ofm` | Falso: es el repositorio de acompañamiento de TabH2O (API alojada, *skill*, integraciones); el código solo está en PyPI | Sección 2, Distribución; sección 10 |
| 2 | Basta `backend_package = "h2ofm"` | Falso: 0.42.0 no carga `timeh2o-v1` (`ValueError` por `variable_patch_sizes`) **[probado]** | Sección 4 (`"h2ofm>=0.43"`), D4 |
| 3 | Las covariables booleanas son numéricas, como en los adapters de Chronos-2, TimesFM 3.0 y T0 (que las convierten a float) | Falso para `h2ofm`: las booleanas son etiquetas (*target encoding*) | Sección 2, Covariables; D8; test 24 |
| 4 | El relleno de `FoundationModel` del PR #1450 sería inocuo para TimeH2O | Falso: cambia la previsión de v1 hasta 0,25, porque el *ensemble* de parches depende de la longitud de la historia **[probado]** | `requires_equal_length_context = False` y su justificación (sección 4) |
| 5 | Sin token, el backend lanza `GatedRepoError` | Sin probar: según el código, también puede ser `FileNotFoundError` (401 sin la marca `GatedRepo`) | Sección 2, Distribución; D3 |
| 6 | Las cifras de GIFT-Eval de la model card | Comprobadas: se reproducen las diez de la tabla con los resultados por dataset del leaderboard | Sección 2, Resultados publicados |
| 7 | "La diferencia máxima es de 2,3e-5 en v1 y en tiny" | 1,5e-5 en v1 y 2,3e-5 en tiny (`exp1.log`) | Sección 2, Entrada |
| 8 | `exp2_backtest.py` salta las filas ya medidas | Fallo: comparaba `''` con `'None'` en las filas sin `patch_ensemble` y la segunda ejecución repitió Chronos-2-small (500) y Chronos-2 (4096). Dieron el mismo MAE (171.27 y 134.05), así que se confirma que son deterministas; las dos filas duplicadas se quitaron del CSV y del log | Script corregido; `results_exp2.csv` y `exp2.log` limpios |
| 9 | Contexto por defecto 4096, como en la propuesta de Timestral | Con la configuración de la guía, 16384 es más preciso (84.24 frente a 98.30) y más rápido con el *ensemble* (114 s frente a 195 s), y 8192 es peor que 4096 | D6; sección 4 |
| 10 | `run_checks.py`: una variable sin usar y una comparación con `0.9 - 0.8` | Corregidas antes de ejecutarlo | `skeleton/run_checks.py` |
| 11 | Las booleanas se pasan tal cual (primera versión del esqueleto) | Medido: como float dan igual o mejor resultado cuando el indicador aparece en el contexto (`exp4`, `exp2` con 16384) y significan lo mismo que en los demás adapters, pero pierden la protección de la codificación categórica cuando no aparece (fila 41) | Esqueleto (conversión a 0/1), D8, test 24 |
| 12 | TimeH2O no tiene el problema de las covariables planas del issue #7 de Timestral (no estaba medido) | Sí lo tiene, con menos intensidad (`exp4`); el mecanismo es el escalado causal, y lo confirman `exp7` a `exp9` | Sección 2, Covariables; D15; helper y aviso en el esqueleto; riesgos |
| 13 | El aviso se limita a covariables exactamente constantes, como en la propuesta de Timestral | Un ruido de 1e-4 o 1e-3 en el contexto dispara igual o más (`exp7`, `exp8`) | Criterio de 20 desviaciones típicas, calibrado con `exp8` y `exp9` |
| 14 | "TimeH2O es el más preciso medido" | Con 16384 de contexto empata con Timestral-209M (84.24 frente a 84.32, medido después); es mejor con contexto corto, mejor calibrado en todos los contextos medidos y, solo con 16384 de contexto, más rápido | Sección 1 y sección 2, Experimento |
| 15 | `exp3` comprobaba niveles repetidos con cuatro columnas | `FoundationModel` construye las columnas en un diccionario y devuelve una por nivel; el script fallaba con `IndexError` | `exp3` y `run_checks.py` corregidos y repetidos |
| 16 | `run_checks.py` exigía que no hubiera ningún aviso con todos los dtypes | `FoundationModel` lanza un `MissingValuesWarning` engañoso cuando la exog futura tiene NaN aunque no falte ninguna fecha; es previo y ajeno a TimeH2O | El chequeo solo cuenta los avisos del backend; queda para un issue aparte (sección 14) |
| 17 | Referencias cruzadas a decisiones (D3 en la licencia, D9 en `adapters.toml`, D6 en `batch_size`, D7 en MPS) | Apuntaban a decisiones que se renumeraron al redactar | Corregidas: D1, D13, D9 y D10 |

**Segunda revisión, por temas (05/11/2026)**

La revisión se hizo por temas (afirmaciones sobre el backend, sobre skforecast, cifras, esqueleto y coherencia del plan), contrastando el documento con las fuentes, y cada hallazgo se volvió a comprobar contra las fuentes antes de darlo por bueno. Resultado: 56 hallazgos confirmados, agrupados abajo por afirmación, y 8 descartados. Todos los confirmados están corregidos; `run_checks.py` se repitió al final (62/62).

| # | Afirmación revisada | Resultado | Dónde se corrigió |
|--:|:--------------------|:----------|:------------------|
| 18 | Mecanismo de la explosión con covariables planas: "normalización causal de cada fila" | Impreciso: son momentos causales acumulados por parche que incluyen el propio parche; con un contexto plano el primer parche del horizonte queda a unas `sqrt(n/p)` escalas, sea cual sea el cambio | Sección 2, Covariables; Notes del helper |
| 19 | "16.384 es el contexto más barato con el *ensemble*" | Falso: 500 es más barato (59.2 s); 16.384 es el más barato de los de 2.048 o más | Sección 2, Experimento; sección 4, constructor; D6 |
| 20 | "Con el *ensemble*, el coste no crece con el contexto sino al revés" | El coste por llamada no es monótono: sube de 512 a 4096 y baja después | Sección 2, Tiempo por llamada |
| 21 | La licencia no prevé una licencia empresarial | La cláusula 12.2 menciona una licencia comercial aparte, sin condiciones ni contacto | Sección 2, Licencia; D1 (b) pregunta por ella |
| 22 | `github.com/h2oai/h2ofm`: "0 issues y 9 pull requests" | El listado solo muestra los abiertos, y hay 8 pull requests (la ref 4 no lo es) | Sección 2, Distribución; secciones 10 y 15 |
| 23 | `batch_size` es el máximo de filas por pasada | Es un umbral aproximado: el backend añade series enteras hasta alcanzarlo, así que un lote puede pasarlo | Sección 2; docstrings del esqueleto; sección 4, decisiones |
| 24 | Un contexto entero a NaN da exactamente 0 en todos los cuantiles | La mediana es exactamente 0; el resto, prácticamente 0 | Sección 2, Entrada |
| 25 | Una covariable entera a NaN en la historia se descarta | Solo las numéricas; una de etiquetas se codifica como fila constante | Sección 2, Covariables; Notes del adapter |
| 26 | Con `autoregressive=True`, bloques de 1.024 pasos | Son `max_horizon // patch_size` parches (64 en v1, de 128 a 1.024 pasos según el parche) | Sección 2, Entrada |
| 27 | Booleana y float 0/1 dan exactamente la misma previsión | La misma salvo ruido de float32 (7,6e-6), y solo si las dos etiquetas aparecen en la historia | Sección 2, Covariables; D8 |
| 28 | El umbral de 20 desviaciones avisa en todos los casos graves medidos | Solo en los de 32 desviaciones o más; el precio de 6 desviaciones (cuantil 0.9 nueve veces mayor) no avisa | Sección 2, Covariables; D15; riesgos |
| 29 | El esqueleto avisa de covariables planas también con el tiny | El tiny ignora las covariables, así que el aviso sería falso | Llamada condicionada a `config.variate_attention`; `config` en el fake; test 36; dos chequeos en `run_checks.py` |
| 30 | Las booleanas llegan siempre como `bool` o `boolean` | Si falta una fecha del contexto, `FoundationModel` la rellena y la columna llega como `object`; convertir después de concatenar la dejaba como etiquetas | `_bool_to_float` por parte, antes de concatenar; `_is_number_like` en el helper; test 37; bloque 5c de `run_checks.py` |
| 31 | El helper solo comprueba columnas numéricas | Una columna de texto en el contexto y numérica en el futuro no se excluía | El helper exige que las dos partes sean numéricas o booleanas; chequeo y test |
| 32 | Celdas del notebook que cambian: 45 y 102 | También la 13 (`list_adapters()`); la 45 solo cambia el código | Sección 7, paso 5; sección 9 |
| 33 | "Es lo que pide `.claude/rules/foundation.md`" | La regla pide recomendar la opción centralizada, y D15 recomienda el helper llamado desde el adapter | Sección 4 (desvío explícito y motivos); D15 |
| 34 | La sección 9 está completa | Faltaban el docstring de `supports_past_only_covariates`, los puntos 6 y 7 y el párrafo de exog de la skill, la viñeta de `context_length` de `adapter-parameters.md` y la línea "predict_kwargs (TimesFM 3.0 only)" | Sección 9 |
| 35 | Fila de avisos de la sección 5: "ninguno de los nueve" | TabICL, TabPFN-TS y Nori avisan con `stacklevel=3` y Moirai, de MPS. Con `FoundationModel.predict` directo, el aviso apunta un marco por encima del usuario **[probado]** | Sección 5; sección 4, decisión 9 |
| 36 | Listas de tests en "Qué cambia" | Incompletas en D2, D4 y D9 | D2, D4 y D9 |
| 37 | `run_checks.py`: "MAE 24.000" en el backtesting | Leía la columna `n_evaluated`. El MAE real es 0.293 | Chequeo con `mean_absolute_error` y `n_evaluated == 24`; `run_checks.py` repetido |
| 38 | Cobertura del 80 % de "0.68 a 0.78 de todos los demás modelos" | El tiny da 0.672 | Sección 1 |
| 39 | D6: "4096 no es más rápido en ningún caso medido" | Solo con v1 y el *ensemble*; sin él o con el tiny, 16.384 es de 3 a 5 veces más lento | D6 (y su relación con D7) |
| 40 | D7: "el MAE baja entre un 5 % y un 9 %" | Entre un 4,5 % y un 8,3 % (las cifras de 5 % a 9 % son la subida sin el *ensemble*) | D7 |
| 41 | D8 y fila 11: como float, "igual o mejor" | Omitía el caso en contra: con un indicador que no aparece en el contexto, la codificación categórica evita la explosión de los cuantiles (exp4 parte 5, fold del 24/12) | D8 (compromiso explícito, con D15 de contrapeso); riesgos |
| 42 | Tiempos de la sección 12 (exp1 5 min, exp7 y exp8 40 min, exp2 45 min) | Unos 8, 12, 20 y 35 minutos | Sección 12 |
| 43 | `extract_skeleton.py` genera "dos archivos" | Genera tres módulos, que se borran antes del commit | Sección 12 |
| 44 | Pico de memoria "en el caso de 200 series de 4.096" | Se alcanzó en el primer bloque (200 series de hasta 4.096 pasos) | Sección 2, Lotes grandes |
| 45 | Fold de la guía "`Holiday` en Nochebuena" | Es el festivo de Navidad, que en el índice (UTC) empieza el 24/12 a las 14:00 | Secciones 1 y 2 |
| 46 | La carga del tiny es más lenta porque incluye la importación de torch | Sin medir en procesos separados | Sección 2: la explicación se quita y se dice que no son comparables |
| 47 | Los casos de los tests del helper y el contrato de `contracts.py` están probados en `run_checks.py` | Faltaban `category`, `None`, columnas mixtas, `max_deviation`, el mensaje, las booleanas `object` y el chequeo de firmas | Añadidos a `run_checks.py` (bloques 0 y 5b); sección 8; README de `foundation_compat` |
| 48 | Docstring de `run_checks.py`: cuantiles 0 y 1 | Solo se probaba el 0 | Chequeo del cuantil 1 añadido |
| 49 | El PR #1450 está "sin fusionar" | Se fusionó el 05/11/2026 (`fc7efd0`) durante la revisión | Secciones 0, 3, 7, 11, 12, 13 y 15; D10 sin la opción c; D14 reescrita (orden respecto a Timestral); `run_checks.py` sin `PYTHONPATH`, repetido contra `fc7efd0` |
| 50 | Marcador pendiente en este anexo | Faltaban las filas de la segunda revisión | Esta tabla |

**Tercera revisión y ensayo de implementación (05/11/2026)**

Una tercera revisión de lectura (coherencia, cifras, implementación en local y neutralidad, con una comprobación aparte de cada hallazgo) confirmó 30 hallazgos, que se reducen a 16 cambios, y descartó 11. Después se implementaron todos los pasos de la sección 7 salvo el 5 (los notebooks) en una copia desechable de `0.27.x` en `fc7efd0`, siguiendo este documento y con las recomendaciones de la sección 6 (D14 (a): no hay adapter de Timestral en `fc7efd0`). Python 3.12.3, torch 2.14.1+cpu, `h2ofm` 0.43.1, numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1; 4 CPU, sin GPU. Resultados **[probado]**:

- Los bloques de código de la sección 4 entraron en el paquete sin ningún cambio.
- `skforecast/foundation/tests/`: 995 tests en verde (68 de ellos en `test_TimeH2OAdapter.py`, 37 funciones); los llamantes de la sección 13, punto 2: 68 en verde; `ruff check` sin errores en los archivos tocados.
- `generate_ai_context_files.py --check`: al día; `execute_notebooks.py --check` sobre los dos notebooks, que el ensayo no editó ni ejecutó: limpios; `check_foundation_models_metadata.py`: los 17 modelos en verde.
- `tools/foundation_compat` con `--backend timeh2o`: contract test (4) y smoke test con los pesos reales (6) en verde.
- `run_checks.py --package` contra el adapter implementado: 62/62 (42 s).
- `backtesting_foundation` en la configuración de la guía, a través del adapter del paquete: tiny con 500 de contexto sin exog y con 4096 con exog, v1 con 500 y con 16384 con exog, con los mismos MAE y coberturas que `results_exp2.csv` (167.41, 162.26, 103.06 y 84.24); el aviso de covariables planas sale una vez con v1 y 500 de contexto, y ninguna en los demás casos.

El ensayo señaló 18 puntos más, agrupados en las filas 59 a 68; las filas 51 a 58 recogen los 16 cambios de la tercera revisión.

| # | Afirmación revisada | Resultado | Dónde se corrigió |
|:--|:--------------------|:----------|:------------------|
| 51 | `exp2` se puede repetir con `0.27.x` actual | Falla al importar el prototipo de Timestral, anterior al PR #1450 (`TypeError`) | Secciones 0 y 12; docstring de `exp2_backtest.py` |
| 52 | Ningún script usa archivos fuera de la carpeta; todos los experimentos sin conexión; ediciones de los logs | `exp2` usa el prototipo de Timestral y `exp2`, `exp7` y `exp9` descargan `vic_electricity`; la primera ejecución de `exp2` tuvo conexión; faltaban dos ediciones de `exp2.log` | Secciones 2 (Distribución) y 12 |
| 53 | Hardware "a 2,80 GHz" | Ningún log registra la CPU, y `run_checks.log` se generó en un Xeon a 2,10 GHz | Secciones 2 y 11, sin frecuencia |
| 54 | "7,6e-6, `exp4.log`" (D8) | El log lo imprime con tres decimales (0.000) | D8 |
| 55 | Recetas del entorno | La de uv no activaba el entorno; `ruff`, `ipykernel` y `papermill` no están en los requisitos; faltaban el directorio de trabajo y el nombre del entorno | Secciones 0 y 11; fila Lint de la sección 12 |
| 56 | "Los logs se añaden con `git add -f`" | Sin rutas, `git add -f` sobre la carpeta añadiría las cachés de `ruff` y `__pycache__`, con rutas absolutas | Sección 0; `--no-cache` en el lint |
| 57 | Referencias ("paso 5", "PR B", "paso A3"), consecuencia de D14 (b), imports de `test_TimeH2OAdapter.py` | Ambiguas, sin definir en este documento o incompletas | D14, D15 y secciones 7, 8 y 13 |
| 58 | Fila 37 de este anexo, sección 0 y fila 16 | Una celda de más, una frase repetida y una "tarea aparte" sin referencia | Este anexo; sección 0; sección 14 (riesgo nuevo) |
| 59 | `run_checks.py` "adaptado al adapter real" | Sin procedimiento: tal cual, sobrescribe el adapter del paquete con el de este documento | Modo `--package` de `run_checks.py`; secciones 12 y 13 |
| 60 | D3: comprobar el error sin token "al implementar" | No estaba en el plan, y no se puede probar detrás de un proxy que autentica las peticiones al Hub | D3; secciones 7 y 13 |
| 61 | Guía: el aviso de covariables planas "saltará" con 500 de contexto | La guía usa `suppress_warnings = True`, que lo oculta; no decía qué contexto usar en la comparación | Sección 9 |
| 62 | `stacklevel=5` apunta a `ForecasterFoundation.predict` | Solo con `predict` y `predict_quantiles` | Decisión 9 de la sección 4; sección 5 |
| 63 | Fallo previo de `verify` en `tools/foundation_compat` "sin comprobar" | Comprobado, con sustitutos locales | Sección 8 |
| 64 | "Ni cambios en `tools/check_foundation_models_metadata.py`" | Se añade el tiny a `EXTRA_MODEL_IDS` | Sección 4 |
| 65 | Números de celda de los notebooks | Son índices desde 0; la 102 es el código y la 101, el título | Secciones 7 y 9 |
| 66 | D1 (b): correo a H2O y "plazo razonable" | Sin responsable, sin plazo y fuera del plan | D1; secciones 7 y 13 |
| 67 | Texto de la release note | Formato del prefijo distinto del de TS-ICL y marcador de PR contrario a la skill `release-note` | Sección 9 |
| 68 | Detalles que el ensayo tuvo que deducir | Comprobar que esta carpeta está en `origin/0.27.x` antes de crear la rama, coma de la última entrada del registro, `import types` en `fixtures_adapters.py`, aserción de `test_resolve_adapter.py`, test de `get_model_info` con el tiny, `python -m pytest` con una instalación editable de otra copia, `quantiles` en el backtesting y coste de tocar `_search.py` | Secciones 0, 4, 8, 9 y 13; docstring de `fake_timeh2o.py` |
