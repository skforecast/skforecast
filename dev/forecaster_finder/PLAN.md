# Selector de forecaster ("Not sure where to start?"): plan de desarrollo

## 1. Estado

- Versión 2, del 2026-11-27. Sustituye a la versión 1 (commit `407746d`) después de la
  revisión independiente de `REVISION.md`. Base: skforecast 0.28.0.
- No hay nada implementado en `skforecast/`, `docs/` ni `skills/`.
- Las cifras de este documento salen de `benchmark/review.py` (protocolo justo, sección 4).
  Las de la versión 1 (ML congelado) no se citan en ninguna página: ver `REVISION.md` H1.
- Filas del roadmap: `DOC-choosing-forecaster-evidence` (fase 0), `FEAT-forecaster-finder`
  (fases 1 a 3), `FEAT-finder-probabilistic` (fase 4), y dos fallos de `Ets` que salieron de
  la revisión: `BUG-ets-freeze-params` y `QW-ets-m-over-24-silent`.
- Informe visual de la versión 1 (no actualizado):
  https://claude.ai/artifact/LdLnbpz9LXdJ4rqPaocysX.

Contenido de la carpeta:

| Ruta | Qué es |
|:--|:--|
| `PLAN.md` | Este documento |
| `REVISION.md` | Revisión independiente: hallazgos por gravedad, con evidencia y propuesta |
| `benchmark/benchmark.py` | Estudio original (ML congelado). Define los casos; se conserva porque reproduce las cifras de la versión 1 |
| `benchmark/review.py` | **Estudio de referencia**: mismos casos y bloques, ML reentrenado, `LightGBM+`, el LightGBM del fragmento, selección en validación, Chronos-2 small, base y synth; `--extra` y `--horizon` para los controles |
| `benchmark/recount.py` | Recuentos por ejecución, por dataset y por tipo de datos |
| `benchmark/finder_tree.py`, `build.py` | Prototipo de la versión 1 de la fuente única (5 preguntas, 100 fragmentos). No refleja todavía la sección 6 |
| `benchmark/check_snippets.py` | Avisos e historia mínima de los fragmentos |
| `benchmark/repro_ets_freeze_params.py` | Reproducción de `BUG-ets-freeze-params` |
| `benchmark/summarize.py`, `evidence.py`, `report_data.py` | Resúmenes de la versión 1 (leen `results/*.csv`) |
| `benchmark/results/` | CSV del estudio original |
| `benchmark/results/review/` | CSV del estudio de referencia, controles (`extra_*`), recuentos, fragmentos y `environment.json`; el error por serie y por bloque, en `detail/` |

Reproducir (desde `benchmark/`, con `uv pip install chronos-forecasting`):

```bash
python -I review.py              # unos 35 minutos en 4 núcleos sin GPU
python -I review.py --extra      # otros sitios de bdg2, otras series de M4, clasificación (15 min)
python -I review.py --horizon    # horizonte cuatro veces más corto (7 min)
python -I recount.py             # cifras de las secciones 4 y 6
python -I build.py               # 105 combinaciones, 100 fragmentos (versión 1)
python -I check_snippets.py      # avisos e historia mínima de los fragmentos
```

Entorno: Python 3.12.3, skforecast 0.28.0, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1,
scipy 1.18.1, numba 0.68.1, LightGBM 4.7.0, PyTorch 2.15.0 (CPU), chronos-forecasting 2.3.2.

## 2. Objetivo y alcance

Un bloque en la documentación que, con unas pocas respuestas en inglés cotidiano, da un punto
de partida concreto: qué forecasters probar, cuál primero, contra qué baseline, con el código
listo para copiar y un prompt para el asistente de IA. Es en parte marketing: debe atraer y
llevar a la siguiente acción (guías, skforecast-ai, Studio, consultoría) sin captar datos
personales.

**Alcance: un árbol de decisión sencillo.** Decide qué forecasters probar, en qué orden y
contra qué baseline. No decide la configuración: lags, diferenciación, hiperparámetros o qué
variables crear siguen en las guías y en las skills. El fragmento de código es un punto de
partida que funciona, con valores fijos según la frecuencia; no se presenta como
recomendación y enlaza a las guías. El resultado no da consejos de configuración.

Principio: **cada regla lleva a la vista su respaldo** (ejecuciones y datasets). Una pregunta
se queda solo si su respuesta cambia lo que se recomienda o el código.

## 3. Referentes

Sin cambios respecto a la versión 1 (el detalle, en el informe visual). Lo que se lleva al
diseño:

- **PyTorch "Start locally"**: todas las opciones a la vista y un resultado que es código y
  cambia con cada clic. Es el patrón que se adopta (sección 6). Contraejemplo de privacidad.
- **scikit-learn "Choosing the right estimator", guías de Azure y AWS**: humildad explícita
  ("es un punto de partida") y "si no funciona, prueba X".
- **ilustradas.news/#test y comparadores de voto**: el porqué de cada respuesta y la métrica
  explicada. No se lleva el formato de test largo con revelado final.
- **No encaja**: email antes del resultado, resultados de identidad, rastreo de terceros.
- **Hueco**: ninguna librería de forecasting en Python tiene una guía de elección con
  evidencia propia y un baseline obligatorio.

Medición disponible hoy: GA4 de Material (`mkdocs.yml:353-355`) con un aviso de
consentimiento sin `actions` (`mkdocs.yml:356-362`); no hay widget de feedback, eventos
propios ni UTM en los enlaces a Studio y skforecast-ai.

## 4. Estudio de referencia

### 4.1 Método

- Mismos bloques de test para todos los modelos de un caso (`TimeSeriesFold`).
- **Todos los modelos usan lo anterior a cada bloque**: ML se reentrena en cada bloque con
  ventana creciente (`refit=True`, `fixed_train_size=False`); AutoETS y AutoARIMA se
  reajustan en cada bloque (lo hace siempre `backtesting_stats`); Chronos-2 recibe todo lo
  anterior como contexto.
- Configuración fija, sin ajustar en test:
  - LightGBM (300 árboles, `learning_rate=0.05`, 15 hojas) con lags, media móvil,
    `calendar_features`, `StandardScaler` y `exog` cuando hay;
  - `LightGBM+` (más lags, dos medias móviles, 500 árboles, regularización suave), como
    control de que el LightGBM anterior no es demasiado flojo;
  - Chronos-2 small zero-shot (`autogluon/chronos-2-small`, Apache-2.0); base y synth, como
    controles de tamaño y de contaminación;
  - naive estacional (`ForecasterEquivalentDate(offset=periodo)`);
  - AutoETS y AutoARIMA en los casos de una serie.
- Métrica: MASE con escala fija (naive de un paso sobre la historia previa al test), media
  entre series, relativa al naive estacional: menos de 1 lo mejora.
- Recuento: por ejecución, por dataset (un voto por dataset) y por tipo de datos. Las cifras
  que se publican son por dataset.
- 12 datasets de regresión en 15 casos, 41 ejecuciones con las historias truncadas, y un
  dataset de clasificación.

### 4.2 Resultados con toda la historia (MASE relativo al naive estacional)

| Caso | Series | Historia | Chronos-2 | LightGBM congelado (v1) | LightGBM | `LightGBM+` | Mejor estadístico |
|:--|--:|--:|--:|--:|--:|--:|:--|
| *Una serie, nada conocido a futuro* | | | | | | | |
| `public_transport_madrid` | 1 | 547 | 0,40 | 1,39 | 1,25 | 1,06 | 0,97 (AutoETS) |
| `bicimad_weekly` | 1 | 382 | 0,59 | 0,75 | 0,73 | 0,75 | 0,93 (AutoARIMA) |
| `h2o_monthly` | 1 | 168 | 0,81 | 0,94 | 0,97 | 0,96 | 0,97 (AutoETS) |
| `fuel_gasoline_monthly` | 1 | 552 | 0,70 | 1,08 | 0,93 | 0,86 | 0,87 (AutoETS) |
| `expenditures_australia_monthly` | 1 | 393 | 0,69 | 2,10 | 0,93 | 0,81 | 0,62 (AutoARIMA) |
| `vic_electricity_daily_past_only` | 1 | 927 | 0,61 | 0,79 | 0,83 | n/a | n/a |
| `bike_sharing_hourly_past_only` | 1 | 8928 | 0,68 | 0,86 | 0,78 | n/a | n/a |
| `ett_hourly_past_only` | 1 | 8928 | 0,65 | 0,61 | 0,62 | n/a | n/a |
| *Una serie, variables conocidas a futuro* | | | | | | | |
| `vic_electricity_daily` | 1 | 927 | 0,49 | 0,38 | 0,39 | 0,43 | 1,18 (AutoETS) |
| `bike_sharing_daily` | 1 | 563 | 0,69 | 0,78 | 0,68 | 0,65 | 0,81 (AutoETS) |
| `bike_sharing_hourly` | 1 | 8928 | 0,63 | 0,71 | 0,69 | 0,65 | n/a |
| *Varias series* | | | | | | | |
| `store_sales_50_items` | 50 | 1602 | 0,66 | 0,71 | 0,70 | 0,66 | n/a |
| `bdg2_buildings_energy` | 40 | 563 | 0,80 | 0,92 | 0,90 | 0,89 | n/a |
| `m4_hourly_50_series` | 50 | 816 | 0,61 | 0,76 | 0,50 | 0,40 | n/a |
| `australia_tourism_quarterly` | 304 | 68 | 0,87 | 0,98 | 0,93 | 0,89 | n/a |

Clasificación (`vic_electricity_classification`): clasificador LightGBM reentrenado, accuracy
0,897 y balanced accuracy 0,739; "misma hora ayer", 0,836 y 0,752.

### 4.3 Cifras por tipo de datos (todas las historias; `recount.py`)

| Tipo de datos | Ejecuciones (datasets) | LightGBM menor que Chronos-2 | LightGBM peor que el naive | Chronos-2 peor que el naive | Mediana Chronos-2 / LightGBM |
|:--|--:|--:|--:|--:|:--|
| Una serie, nada conocido a futuro | 22 (8) | 3 | 11 | 1 | 0,69 / 1,00 |
| Una serie, variables conocidas a futuro | 11 (2) | 9 | 0 | 0 | 0,67 / 0,66 |
| Varias series | 8 (4) | 3 | 0 | 0 | 0,79 / 0,82 |
| Total | 41 (12) | 15 | 11 | 1 | 0,69 / 0,91 |

- Por dataset: Chronos-2 por delante de LightGBM en 8 de 12 (10 de 12 con toda la
  historia); mejora al naive en 12 de 12, y LightGBM, en 9.
- Estadísticos en mensual (12 ejecuciones, 3 datasets): menos error que Chronos-2 en 6 y que
  LightGBM en 12; AutoETS nunca quedó peor que el naive. En diario (13 ejecuciones), AutoETS
  no ganó nunca a Chronos-2. En semanal, `Ets(m=52)` no es estacional.
- Controles (detalle en `REVISION.md`): Chronos-2 synth mejora al naive en 15 de 15 y a
  LightGBM en 10 de 15; el modelo base no mejora al small de forma consistente (8 de 15);
  un horizonte 4 veces más corto no cambia el orden (9 de 12 en los dos); `LightGBM+` no
  cambia el orden (Chronos-2 gana en 27 de 38); en `bdg2`, Chronos-2 gana en los 9 sitios o
  muestras probados.
- Variables solo hasta hoy (3 casos, 3 tamaños de Chronos-2): de un 14 % menos de error a un
  4 % más. No hay regla.

### 4.4 Qué decide cada pregunta

| Pregunta | Resultado medido | Decisión |
|:--|:--|:--|
| Qué quieres predecir | Solo el clasificador predice categorías; no supera a "misma hora ayer" en balanced accuracy (1 dataset) | Se queda, por capacidad |
| Cuántas series | Con varias series no hay favorito claro | Se queda: cambia la clase, el baseline y **el mensaje** |
| Cada cuánto se registran | AutoETS empata con Chronos-2 en mensual y no sirve en diario ni semanal | Se queda: constantes del código y referencia estadística en mensual y trimestral |
| Qué más sabes, a futuro | Con variables conocidas a futuro, LightGBM alcanza o supera a Chronos-2 (2 datasets) | Se queda: añade `exog` y **cambia el mensaje** |
| Qué más sabes, solo hasta hoy | Sin ganancia consistente | Se funde con "No", con una nota |
| Qué te importa más | No ordena por precisión; el coste de producción no está medido; la explicabilidad es una capacidad | Se quita (decisión D3) |
| Cuánta historia | No cambia el orden dentro de cada tipo de datos | Sin pregunta |
| Qué horizonte | No cambia el orden | Sin pregunta |

### 4.5 Límites

- Pocos datasets por tipo: 8, 2 y 4. La regla con variables conocidas a futuro es la más
  débil, y sus variables son valores observados, no pronósticos.
- Una configuración por modelo, sin ajuste: no dice cuánto se acerca un LightGBM trabajado.
- Chronos-2 en CPU; el coste de producción no está medido.
- AutoETS y AutoARIMA medidos con `freeze_params=True` (`BUG-ets-freeze-params`).
- Diferencias de pocas centésimas no son significativas: en varios casos Chronos-2 gana la
  mitad de los bloques.

## 5. Correcciones de la base

Hallazgos sobre `skills/choosing-a-forecaster/SKILL.md` (S), `tools/ai/llms-base.txt` (L) y
relacionados, verificados contra el código el 2026-11-27 (`REVISION.md` H11).

| # | Dónde | Qué está mal | Corrección |
|--:|:--|:--|:--|
| 1 | S:50, 60, 74, 84, 92-93; L:724 | Fundacionales solo para "zero-shot / no training data / cold-start", fuera del árbol; "START HERE" en `ForecasterRecursive`; "Start simple: Begin with ForecasterRecursive" | Las reglas de 6.2: dos candidatos y un baseline, con el mensaje según el tipo de datos |
| 2 | S:170, 74 | "LightGBM is the best starting point" | "LightGBM is the ML candidate; with a single series and no variables known in advance it did worse than the seasonal naive in 11 of 22 runs: always compare it with the baseline" |
| 3 | S:49, 130; `statistical-models/SKILL.md:15` | "Short series (< 200 obs) → statistical" | "For a monthly series without extra variables, AutoETS is hard to beat as a reference"; quitar el umbral; no extenderlo a semanal ni diario |
| 4 | S:29, 50, 60, 93 | 11 modelos fundacionales sin distinguir licencias; 5 restringen el uso comercial (TimesFM 3.0, Moirai-2, TabPFN-TS, TS-ICL, Timestral) | Recomendar `autogluon/chronos-2-small` y avisar del resto |
| 5 | S:146 | No distingue variables conocidas a futuro de las conocidas solo hasta hoy | "ML needs the future values of `exog`. Chronos-2 also accepts variables known only up to today; in our tests they gave no consistent gain" |
| 6 | S:36-41, 58, 69 | El paso 1 mezcla número de series y tipo de objetivo; DirectMultiVariate dos veces; varias series categóricas sin hoja | Primera pregunta: tipo de objetivo; un clasificador por serie |
| 7 | S:77-79, 176; `baseline-forecasting` | Sin baseline para varias series ni para clasificación | Bucle por serie y regla estacional (`y.shift(periodo)`) |
| 8 | S:59-60, 89-93 | RNN antes que Foundation | Foundation dentro y antes; RNN como nota |
| 9 | S:151-157 | Matriz: Stats sin intervalos paramétricos ni importancias; Foundation sin `predict_interval`; **el clasificador sin importancias**; faltan las filas de valores ausentes y de `calendar_features` | Corregir a mano. Generarla desde `get_tags()` no es posible: Stats y Foundation no lo tienen |
| 10 | S:176 | `Arima(order=None)` sin `m`: no busca términos estacionales | `Arima(order=None, seasonal_order=None, m=periodo)` |
| 11 | `statistical-models/SKILL.md:24` | Cita `interval_method`, que no existe en Stats | "via `interval` or `alpha`" |
| 12 | S:123, 147 | "Per-series differentiation" en DirectMultiVariate: solo acepta `int` | "Same order for all series" |
| 13 | S:48, 73; L:67-81; `docs/introduction-forecasting/introduction-forecasting.md:168-184`; `home.html:189` | Jerga y cinco fuentes para la misma decisión | Una fuente que genere la skill y la página |
| 14 | S:171, 175, 50 | Calendario como exógena manual; intervalos con residuos in-sample; "no training data" | Corregir redacción |
| 15 | `docs/user_guides/comparing-forecasting-models.ipynb`, "How the models are compared" | "trained once [...] without refitting between folds": AutoARIMA y AutoETS sí se reajustan | Decirlo, o reentrenar también ML y volver a ejecutar la guía |

## 6. Diseño (fases 1 y 2)

### 6.1 Formato y preguntas

Un selector de una pantalla, no un test por pasos: cuatro filas de opciones con una
selección por defecto, y debajo el resultado, que cambia con cada clic. Título del bloque:
"Build your first comparison"; "Not sure where to start?" queda como gancho en la portada.

| # | Row | Options (hint) | Shown if |
|--:|:--|:--|:--|
| 1 | What do you want to forecast? | **A number** (sales, energy demand, website visits) · **A category** (low, medium or high demand; failure or no failure) | Always |
| 2 | How many time series? *A time series is one column of values recorded over time, such as the daily sales of one store.* | **One** · **Several** (each store, each product) | A number |
| 3 | How often are the values recorded? | Every hour · Every day · Every week · Every month · Every quarter · Something else | Always |
| 4 | Will you know anything else in advance? *Things that affect the values and that you know for future dates. skforecast calls them exogenous variables.* | **Yes** (holidays, planned promotions, prices, weather forecasts) · **No** (also if you only know them up to today, like sensor readings) | Always |

Selección por defecto: un número, una serie, cada día, no.

### 6.2 Reglas

| Respuestas | Titular | Candidatos | Respaldo que se muestra |
|:--|:--|:--|:--|
| Una categoría | "Start with a machine learning classifier and compare it with a simple rule" | `ForecasterRecursiveClassifier` con LightGBM; baseline: la categoría del mismo periodo una estación antes | "On one dataset the classifier was right more often (90% vs 84%) but did not beat the rule on balanced accuracy" |
| Un número, una serie, No | "Start with Chronos-2, then try to beat it with LightGBM" (decisión D2) | 1. Chronos-2, 2. LightGBM, baseline | "Lower error than LightGBM in 7 of 8 datasets; LightGBM did worse than the baseline in 11 of 22 runs" |
| Un número, una serie, Sí | "Race LightGBM against Chronos-2" | Los dos con `exog`, baseline | "No clear winner: LightGBM ahead in one dataset, level in the other" |
| Un número, varias series | "Race Chronos-2 against LightGBM" | `ForecasterRecursiveMultiSeries` y Chronos-2; baseline por serie | "Chronos-2 ahead in 3 of 4 datasets with the full history (2 of 4 over all histories), by small margins" |
| Además: una serie, mensual o trimestral | "...and keep AutoETS as a statistical reference" | `ForecasterStats(Ets(m=...))` | "On monthly data AutoETS matched Chronos-2 (3 datasets)" |
| Something else (frecuencia) | Los mismos candidatos | Código con `SEASON` y `STEPS` por editar | Enlace a la guía |

Notas fijas en el resultado (sustituyen a la pregunta de prioridad): "Need to explain the
forecast? LightGBM gives feature importances; foundation models do not" y "Forecasting many
series often? Chronos-2 needs PyTorch and, at scale, a GPU". Nota de licencias: Chronos-2 es
Apache-2.0; otros modelos fundacionales de skforecast restringen el uso comercial.

Las cifras del respaldo se generan de los CSV del estudio, no se escriben a mano.

### 6.3 Requisitos del fragmento

1. **Es lo que se midió**: el bloque de ML del fragmento y el del estudio salen de la misma
   función. Hoy no es así (`REVISION.md` H3).
2. `TimeSeriesFold(..., refit=True, fixed_train_size=False)`: todos los modelos usan lo
   anterior a cada bloque.
3. **Sin avisos**: CI ejecuta cada fragmento con los avisos como errores (hoy avisan 72 de
   100).
4. **Historia mínima razonable y escrita en la página**: debe ejecutarse con 2 semanas de
   datos horarios, 3 meses de diarios, 2 años de semanales, 3 años de mensuales y 4 años de
   trimestrales (hoy: 21 días, 150 días, 2 años, 5 años y 5 años).
5. Constantes por frecuencia (estación, horizonte, lags), con una línea de cabecera: "Example
   settings so the script runs; they are not tuned" y el enlace a la guía. Sin consejos de
   configuración en el resultado.
6. Sin AutoETS en semanal. AutoETS en mensual y trimestral solo cuando esté cerrado
   `BUG-ets-freeze-params`.
7. Línea de instalación (`pip install skforecast lightgbm chronos-forecasting`) y aviso de
   que la primera ejecución descarga los pesos.
8. Todo lo que se calcula se imprime (hoy `get_feature_importances()` es una expresión
   suelta).
9. Con 4 filas quedan 30 fragmentos distintos como mucho (20 de número y 10 de categoría),
   más los de "Something else".

### 6.4 Pantalla de resultado

1. Titular (6.2) y la alineación: candidatos, referencia estadística si aplica y baseline,
   cada uno con su clase.
2. Por qué, con el respaldo de la regla y el enlace a "How we chose these rules".
3. Ventana de código con el estilo de la portada (`docs/stylesheets/home.css`, resaltador de
   `docs/javascripts/home.js` como módulo compartido), botón Copy y la nota "This code is run
   on sample data every time the docs are built".
4. Acciones: "Copy prompt for your AI assistant" (respuestas, plan, enlace a `llms-full.txt`
   y "ask me for my data first"); "Compare models on your own data with skforecast-ai" (sin
   prometer la misma comparación mientras sus reglas difieran: `REVISION.md` H12); guías
   siguientes; consultoría discreta.
5. Enlace compartible con las respuestas en el `#`.
6. Sin JavaScript se ve la página estática completa (tabla de reglas y fragmentos).

### 6.5 Ubicación y enlaces

- Página `docs/user_guides/choose-a-forecaster.md`, primera de *Choosing a model*
  (`mkdocs.yml:44`), antes de *Comparing forecasting models*, con la sección "How we chose
  these rules".
- Portada: "Not sure which one to use?" (`docs/overrides/home.html:189`) cambia de destino.
- Quick start, introducción (bajo la tabla de forecasters) y *AI-assisted forecasting*.
- Entrada en `tools/docs/hooks/page_meta.yml`.

### 6.6 Fuente única y validación

- Fuente: un fichero de datos (YAML o JSON) con filas, reglas, constantes por frecuencia y
  la referencia a las cifras, en `skills/choosing-a-forecaster/references/`, más un módulo
  de plantillas en `tools/`. `llms-full.txt` lleva el mismo árbol a los asistentes, y
  skforecast-ai puede leer el fichero.
- Generación: un hook de MkDocs (como `tools/docs/hooks/examples_grid.py`) incrusta la tabla
  de reglas, el respaldo y los fragmentos en la página; `generate_ai_context_files.py`
  reescribe la sección de decisión de la skill entre marcadores, y `--check` falla si alguien
  la edita a mano.
- En cada PR (`--check`): todas las combinaciones tienen hoja; cada clase y cada argumento
  de los fragmentos existe (import y firma; `get_tags()` no sirve para Stats ni Foundation);
  `get_model_info()` confirma que el fundacional no restringe el uso comercial; los enlaces
  son páginas del `nav`.
- Ejecución de los fragmentos: workflow propio con PyTorch y `chronos-forecasting`, lanzado
  solo cuando cambian la fuente, las plantillas o `skforecast/foundation/` (filtro de
  rutas) y antes de cada publicación. Avisos como errores e historia mínima de 6.3.
- Estudio: se vuelve a ejecutar en cada versión menor y cuando cambie la versión de
  `chronos-forecasting` o de LightGBM. Si una cifra del respaldo cambia, la página cambia
  con ella; si una regla deja de cumplirse (el candidato que va primero deja de ganar en la
  mayoría de los datasets de su tipo), se rebaja a "sin favorito" en esa misma PR.

### 6.7 Qué medir

Eventos con consentimiento (`__md_get("__consent")?.analytics`), solo identificadores
cerrados: `finder_change {row, option}`, `finder_copy {kind}`, `finder_cta {target}`; widget
de feedback de Material en la página; UTM en los enlaces salientes.

Antes de publicar, anotar las visitas de 4 semanas de *Comparing forecasting models* y de la
tabla de la introducción: son la referencia. Sin ella no hay umbrales.

| Pregunta | Señal | Criterio a las 8 semanas |
|:--|:--|:--|
| ¿Llegan? | Visitas de la página | Al menos las de la tabla de la introducción a la que sustituye como destino |
| ¿Les sirve? | Copias de código o de prompt por visita | 1 de cada 5 visitas |
| ¿Convierte? | Clics a skforecast-ai, Studio y consultoría | Más que los mismos enlaces en la portada, por visita |
| ¿Acierta? | "Was this useful?" | Mayoría de respuestas positivas; revisar los comentarios |

### 6.8 Riesgos

- Reglas con pocos datasets: cada regla con su respaldo a la vista y "no clear winner" donde
  lo hay; ampliar el estudio (decisión D4).
- Evidencia que caduca: 6.6, con criterio de rebaja.
- Primera ejecución decepcionante de ML: requisitos 1 a 3 de 6.3.
- Dependencia de PyTorch para Chronos-2: el fragmento lo dice; el workflow de CI es aparte.
- Divergencia con skforecast-ai: fuente legible por máquina e incidencia en su repositorio.
- Coste de mantenimiento del JavaScript: sin dependencias, y la página funciona sin él.

## 7. Plan de desarrollo

### 7.1 Decisiones cerradas

1. Alineación: Chronos-2 small y LightGBM más un baseline estacional; clasificador más regla
   estacional para categorías. Solo se nombra Chronos-2 (único medido y Apache-2.0).
2. El mensaje depende del tipo de datos (6.2), no de la prioridad del usuario.
3. Cuatro filas. Sin preguntas de historia ni de horizonte. "Solo hasta hoy" se funde con
   "No".
4. AutoETS como referencia solo en mensual y trimestral de una serie.
5. Estudio de referencia: el de protocolo justo. Cifras por dataset, nunca "frente a la mejor
   variante de ML".
6. Requisitos del fragmento de 6.3.
7. Página estática primero; el bloque interactivo se monta encima y la página funciona sin
   JavaScript.
8. El estudio vive en `benchmarks/forecaster_finder/`, con requisitos fijados, desde la fase
   0; esta carpeta de `dev/` se borra al cerrar la última fila.
9. Analítica: GA4 con el consentimiento que ya existe. Sin migración.
10. Fase probabilística: una fila opcional, no una extensión con tres preguntas (sección 8).

### 7.2 Decisiones que necesitan respuesta

| ID | Decisión | Opciones | Recomendación |
|:--|:--|:--|:--|
| D1 | Formato del bloque interactivo | (a) selector de una pantalla; (b) test por pasos con resultado al final; (c) solo la página estática | (a). Es el formato honrado cuando 3 de 4 respuestas cambian sobre todo el código, da valor desde el primer clic y se puede reutilizar en la portada. (c) es el mínimo: se hace igualmente como fase 1 |
| D2 | Titular con una serie y nada conocido a futuro | (a) "Start with Chronos-2, then try to beat it with LightGBM"; (b) "Race Chronos-2 against LightGBM", como en el resto | (a). Es la única regla fuerte (7 de 8 datasets, y el control sintético descarta la contaminación). (b) es más cómodo para el posicionamiento "machine learning-first", pero esconde lo que más ayuda a quien empieza |
| D3 | Pregunta "What matters most to you?" | (a) quitarla y dejar dos notas fijas; (b) mantenerla como quinta fila que reordena y añade importancias | (a). No cambia ninguna recomendación medida |
| D4 | Ampliar el estudio antes de publicar cifras | (a) añadir datasets hasta 6 o más por tipo de datos, con variables a futuro como pronósticos donde se pueda (`h2o_exog`, `m5`, `ashrae_daily`, `bdg2_hourly`, `m4_daily`, `items_sales`, `website_visits`, `uk_daily_flights`, `air_quality_valencia`); (b) publicar con los 12 datasets y los avisos | (a), dentro de la fase 0: de 1 a 2 días más |
| D5 | Nivel de `BUG-ets-freeze-params` | P1 (resultado erróneo sin aviso con la configuración por defecto de `backtesting_stats` y AutoETS) u otro | P1. Bloquea el fragmento con AutoETS, no el resto |

### 7.3 Fases

**Fase 0. Estudio de referencia y base** (`DOC-choosing-forecaster-evidence`, P2)

- Entregables:
  1. `benchmarks/forecaster_finder/`: un solo script con el protocolo justo (sale de
     `review.py` y `recount.py`), requisitos fijados, los CSV y un README. Con D4 (a), los
     datasets nuevos.
  2. Correcciones 1 a 15 de la sección 5 en `skills/choosing-a-forecaster/SKILL.md`,
     `skills/statistical-models/SKILL.md`, `skills/baseline-forecasting/SKILL.md` (baseline
     por serie y de clasificación), el consejo 6 de `tools/ai/llms-base.txt` y la frase de
     *Comparing forecasting models*.
  3. Incidencia en skforecast-ai con H1, H4 y H12 de `REVISION.md`.
- Terminado cuando: las cifras de la skill coinciden con los CSV y son por dataset; `python
  tools/ai/generate_ai_context_files.py --check` y `verify` en verde; nota en
  `docs/releases/releases.md`.
- Esfuerzo: M (3 a 4 días; 5 con D4 (a)).
- No requiere los fallos de Ets: la skill puede citar AutoETS en mensual con el aviso.

**Fase 1. Fuente única y página estática** (`FEAT-forecaster-finder`, P3, parte 1)

- Entregables: fichero de reglas y módulo de plantillas (6.6); página
  `choose-a-forecaster` con la tabla de reglas, el respaldo generado, un fragmento por
  combinación representativa (una serie y varias, con y sin `exog`, categoría) con la tabla
  de constantes por frecuencia, y "How we chose these rules"; sección de la skill generada
  entre marcadores; enlaces de 6.5; workflow de ejecución de fragmentos; widget de feedback.
- Terminado cuando: página, skill y `llms-full.txt` salen de la misma fuente y `--check` lo
  comprueba; todos los fragmentos pasan con avisos como errores y con la historia mínima de
  6.3; la página se lee completa sin JavaScript; `verify` en verde y nota de versión.
- Esfuerzo: M (4 a 5 días).
- Requiere: fase 0. Para el fragmento con AutoETS, `BUG-ets-freeze-params`.

**Fase 2. Bloque interactivo** (`FEAT-forecaster-finder`, parte 2; según D1)

- Entregables: selector de 4 filas sobre la misma página (JavaScript propio, sin
  dependencias, datos generados por el hook), ventana de código, copiar código y prompt,
  enlace compartible, eventos con consentimiento, UTM, y el bloque enlazado desde la portada.
- Terminado cuando: cada combinación muestra exactamente el fragmento que ejecuta CI; se
  maneja con teclado y lector de pantalla (grupos de opciones con `role="radiogroup"`); sin
  consentimiento no se envía ningún evento; funciona en móvil (360 px) y en los dos temas.
- Esfuerzo: M (3 a 5 días).
- Requiere: fase 1.

**Fase 3. Medición y decisión** (`FEAT-forecaster-finder`, parte 3)

- Entregables: referencia de visitas previa (6.7) y una nota a las 8 semanas con las cuatro
  señales y la decisión: mantener, simplificar a la página estática o ampliar.
- Terminado cuando: la decisión está escrita en el roadmap.
- Esfuerzo: S.

**Fase 4. Fila de incertidumbre** (`FEAT-finder-probabilistic`, P4)

- Sección 8. Requiere la fase 3 con decisión de mantener.
- Esfuerzo: M (estudio, 2 a 3 días; fila y fragmentos, 2 días).

Orden y dependencias: fallos de Ets (aparte, P1 y P2) → fase 0 → fase 1 → fase 2 → fase 3 →
fase 4. La fase 0 tiene valor aunque no se haga nada más: hoy la skill dice a los asistentes
que empiecen por `ForecasterRecursive` con LightGBM y reserva los fundacionales para "sin
datos".

## 8. Fase 4: fila de incertidumbre (más adelante)

### 8.1 Qué es

Una quinta fila opcional del selector, no una extensión con preguntas propias:

| Row | Options (hint) |
|:--|:--|
| Do you also need the uncertainty of the forecast? | **No** · **A range** (a low and a high value for each date) · **Several percentiles** (10%, 50%, 90%, for stock or capacity planning) · **Totals or the chance of an event** (next week's total, the chance of running out) |

Con "No" (por defecto) nada cambia. Las otras opciones añaden al código el método que
corresponde a cada candidato y el backtest con cobertura y anchura.

### 8.2 Lo que hay en skforecast (verificado en el código el 2026-11-27)

| Forecaster | Métodos | Salidas |
|:--|:--|:--|
| Recursive, Direct, RecursiveMultiSeries, DirectMultiVariate | `'bootstrapping'` y `'conformal'`; residuos in-sample u out-of-sample (`set_out_sample_residuals`); `use_binned_residuals=True` por defecto | `predict_interval`, `predict_quantiles`, `predict_dist`, `predict_bootstrapping` (trayectorias) |
| Foundation | Cuantiles nativos (Chronos-2: cualquier nivel) | `predict_interval`, `predict_quantiles`; **sin** trayectorias. En `backtesting_foundation`, argumento `quantiles` |
| Stats | Paramétrico (`interval` o `alpha`) | `predict_interval` |
| EquivalentDate, Rnn | Solo `'conformal'` | `predict_interval` |
| RecursiveClassifier | Probabilidades de clase | `predict_proba` |

Lo que ya mide la documentación (`docs/user_guides/probabilistic-forecasting-overview.md`):
intervalo del 80 % con residuos in-sample, cobertura de alrededor del 61 %; bootstrapping en
modelos globales, del 96 % al 98 %; conformal, alrededor del 84 %; bootstrapping con 300
series, unas 35 veces más lento que conformal.

### 8.3 Reglas (hipótesis, por capacidad; el estudio las confirma o las cambia)

| Opción | Chronos-2 | LightGBM | AutoETS | Baseline |
|:--|:--|:--|:--|:--|
| A range | Cuantiles nativos | Conformal con residuos out-of-sample | Paramétrico | Conformal |
| Several percentiles | Cuantiles nativos | Bootstrapping con residuos out-of-sample (conformal da un solo intervalo) | No aplica | No aplica |
| Totals or the chance of an event | No puede (sin trayectorias): el resultado lo dice | `predict_bootstrapping` | No aplica | No aplica |
| Categoría | | `predict_proba`, sin fila | | |

Los residuos out-of-sample van en el código por defecto (un backtest de calibración antes
del test); no se pregunta. El nivel (80 %) es una constante editable. `use_binned_residuals`
se queda en su valor por defecto. Son configuración, fuera del alcance.

### 8.4 Estudio que la habilita

Mismos casos, bloques y protocolo justo que la sección 4, más:

- Métodos: conformal y bootstrapping con residuos out-of-sample en LightGBM, cuantiles
  nativos de Chronos-2, paramétricos de AutoETS y conformal del naive.
- Métricas: cobertura empírica del 80 %, anchura, WIS, cobertura por paso del horizonte y
  tiempo. Para totales, cobertura de la suma del horizonte.
- Criterio fijado antes de medir: un método vale si su cobertura queda a menos de 5 puntos
  de la nominal en la mayoría de los datasets; entre los que valen, el de menor WIS.
- Tamaño: en baja frecuencia hacen falta más bloques (36 predicciones mensuales dan un error
  típico de unos 7 puntos en una cobertura del 80 %); usar orígenes solapados.
- Qué debe contestar: ¿conformal o bootstrapping para "A range" en ML? ¿Los cuantiles nativos
  de Chronos-2 están calibrados sin ajuste? ¿El coste de bootstrapping con muchas series
  obliga a otra regla para "Several"?

### 8.5 Cuándo se activa

- Fase 3 cerrada con decisión de mantener el selector.
- Estudio de 8.4 hecho y reglas que lo reflejan.
- Fragmentos nuevos ejecutados en CI con los requisitos de 6.3.
- `skills/prediction-intervals/SKILL.md` (tabla de las líneas 16-20, sin los cuantiles
  nativos) y `references/interval-compatibility.md` (sin `ForecasterFoundation`) corregidas
  y regeneradas.
