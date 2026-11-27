# Selector de forecaster ("Not sure where to start?"): hallazgos y plan

## 1. Estado

- Fecha: 2026-11-27. Base: skforecast 0.28.0, `origin/0.28.x` en `ca0f2f4`.
- No hay nada implementado en `skforecast/`, `docs/` ni `skills/`: este documento, los
  scripts y los resultados de `benchmark/` son todo lo que existe.
- Informe visual con el gráfico del estudio y una maqueta del test que se puede usar
  (Artifact privado, se comparte desde su menú Share):
  https://claude.ai/artifact/LdLnbpz9LXdJ4rqPaocysX. Este documento es la referencia: el
  informe resume lo mismo.
- Fila del roadmap: `DOC-choosing-forecaster-evidence` (fase 0), `FEAT-forecaster-finder`
  (fase 1) y `FEAT-finder-probabilistic` (fase 2 del test), sección 8.

Contenido de la carpeta:

| Ruta | Qué es |
|:--|:--|
| `PLAN.md` | Este documento |
| `benchmark/benchmark.py` | El estudio: 16 casos sobre 13 datasets, mismo backtesting para todos |
| `benchmark/summarize.py`, `evidence.py`, `report_data.py` | Resúmenes: por familia, cifras de cada regla, datos del informe |
| `benchmark/finder_tree.py` | Prototipo de la fuente única: preguntas, reglas y plantillas de código |
| `benchmark/build.py` | Genera todas las combinaciones de respuestas, ejecuta cada fragmento distinto con datos sintéticos y escribe `finder.json` |
| `benchmark/results/` | CSV de cada caso y `environment.json` de la ejecución citada aquí |

Reproducir (desde `dev/forecaster_finder/benchmark/`, con `chronos-forecasting` instalado):

```bash
python -I benchmark.py            # todos los casos, unos 4 minutos de cómputo en 4 núcleos
python -I benchmark.py vic h2o    # solo los casos cuyo nombre contiene una de las palabras
python -I evidence.py             # cifras de cada regla
python -I build.py                # 105 combinaciones, 100 fragmentos, minuto y medio
```

Entorno de la ejecución citada: Python 3.12.3, skforecast 0.28.0, numpy 2.5.5, pandas 3.1.1,
scikit-learn 1.9.1, LightGBM 4.7.0, PyTorch 2.15.0 (CPU), Chronos-2 small.

## 2. Objetivo

Un test interactivo en la documentación: unas pocas preguntas en inglés, en vocabulario
cotidiano, y al final un punto de partida concreto (qué forecasters probar, en qué orden,
contra qué baseline, con el código listo para copiar y un prompt para el asistente de IA).
Es en parte marketing: debe atraer y llevar a la siguiente acción (guías, skforecast-ai,
Studio, consultoría) sin captar datos personales.

Principio de diseño, que sale del estudio: **cada regla del test está medida**. Una pregunta
solo se queda si su respuesta cambia lo que funciona mejor o el código que hay que escribir.

## 3. Referentes (resumen)

El detalle, con enlaces y la distinción entre lo verificado y lo que es opinión, está en el
informe (parte 1). Lo que se lleva al diseño:

- **ilustradas.news/#test** (recorrido completo con Playwright): el porqué después de cada
  respuesta, la métrica del resultado explicada, salidas de escape ("ninguna me convence"),
  progreso reanudable y resultado que lleva al contenido completo. No se lleva: la longitud
  (de 20 a 100 preguntas), la ficha de identidad para redes y el ranking agregado, que
  necesita backend.
- **PyTorch "Start locally"**: el resultado es un comando listo para copiar y se actualiza con
  cada clic. Contraejemplo de privacidad (HubSpot, Segment, Facebook, ZoomInfo...).
- **scikit-learn "Choosing the right estimator", guías de Azure y AWS**: humildad explícita
  ("es un punto de partida") y "si no funciona, prueba X".
- **Comparadores de voto** (Wahl-O-Mat, 26 millones de usos en 2025; StemWijzer; smartvote):
  pocas preguntas, respuestas visibles y explicadas. El metaanálisis de Munzert y Ramirez Ruiz
  muestra que el efecto observacional (OR 1,87) desaparece en experimentos (OR 1,04): medir
  el embudo, no atribuir causalidad.
- **No encaja**: email antes del resultado, resultados de identidad ("eres un
  ForecasterRecursive"), más de 5 o 6 preguntas, rastreo de terceros.
- **Hueco**: ninguna librería de forecasting en Python (Darts, sktime, Nixtla) tiene una guía
  de elección con evidencia propia y un baseline obligatorio.

Medición disponible hoy: GA4 de Material (`mkdocs.yml:354-356`) con un aviso de
consentimiento sin `actions` (`mkdocs.yml:357-363`); no hay widget de feedback, eventos
propios ni UTM en los enlaces a Studio y skforecast-ai.

## 4. Estudio medido

### 4.1 Método

- Mismos bloques de test para todos los modelos de un caso (`TimeSeriesFold`); ML entrenado
  una vez (`refit=False`); Chronos-2 con todo lo anterior a cada bloque como contexto
  (`backtesting_foundation`), como en *Comparing forecasting models*.
- Configuración fija, sin ajustar en test:
  - LightGBM (300 árboles, `learning_rate=0.05`, 15 hojas) y Ridge, con lags, media móvil,
    `calendar_features` y `exog` cuando hay, con y sin `differentiation=1`;
  - Chronos-2 small zero-shot (`autogluon/chronos-2-small`, Apache-2.0);
  - naive estacional (`ForecasterEquivalentDate(offset=periodo)`);
  - AutoETS y AutoARIMA en los casos de una serie.
- Métrica: MASE con escala fija (naive de un paso sobre toda la historia previa al test),
  media entre series, expresada relativa al naive estacional: menos de 1 lo mejora.
- Tiempo: backtest completo en CPU.
- Historia: en 10 casos se repite entrenando con historias truncadas (de 90 días a toda la
  serie; de 3 años a toda la serie en los mensuales).
- Variables solo hasta hoy en Chronos-2: con un bucle propio (`fit` con `exog` y `predict`
  sin `exog` futuro), porque `backtesting_foundation` aún no las admite (fila FE-02 del
  roadmap).

### 4.2 Casos

| Pregunta que pone a prueba | Casos | Test |
|:--|:--|:--|
| Una serie, variables conocidas a futuro | `vic_electricity` diario (temperatura, festivos); `bike_sharing` diario y horario (meteorología) | 6 × 28 días; 28 × 24 h |
| Una serie, sin variables | `public_transport_madrid` diario; `bicimad` semanal; `h2o`, `fuel_consumption`, `expenditures_australia` mensuales | 6 × 28 días; 6 × 8 semanas; 3 a 5 × 12 meses |
| Variables conocidas solo hasta hoy | `vic_electricity` diario (temperatura); `bike_sharing` horario (meteorología); `ett_m2` horario (6 series de carga) | 6 × 28 días; 28 × 24 h |
| Varias series | `store_sales` (50 productos, diario); `bdg2_daily` (40 edificios del sitio Fox, con temperatura); `m4_hourly` (50 series); `australia_tourism` (304 series trimestrales) | 8 × 28 días; 6 × 28 días; 4 × 48 h; 3 × 4 trimestres |
| Una categoría | `vic_electricity_classification` (bajo, medio, alto; horario) | 28 × 24 h |

### 4.3 Resultados con toda la historia (MASE relativo al naive estacional)

| Caso | Series | Variables | Historia | Chronos-2 | LightGBM | Mejor ML (variante) | Mejor estadístico | Tiempo Chronos-2 / LightGBM (s) |
|:--|--:|:--|--:|--:|--:|:--|:--|--:|
| `vic_electricity_daily` | 1 | a futuro | 927 | 0.49 | 0.38 | 0.38 (LightGBM) | 1.18 (AutoETS) | 1.1 / 0.2 |
| `bike_sharing_daily` | 1 | a futuro | 563 | 0.69 | 0.78 | 0.74 (Ridge diff) | 0.80 (AutoETS) | 0.6 / 0.2 |
| `bike_sharing_hourly` | 1 | a futuro | 8928 | 0.63 | 0.71 | 0.71 (LightGBM) | n/a | 12.1 / 0.4 |
| `public_transport_madrid` | 1 | ninguna | 547 | 0.40 | 1.39 | 1.07 (Ridge diff) | 0.98 (AutoETS) | 0.5 / 0.2 |
| `bicimad_weekly` | 1 | ninguna | 382 | 0.59 | 0.75 | 0.69 (LightGBM diff) | 1.05 (AutoETS) | 0.5 / 0.2 |
| `h2o_monthly` | 1 | ninguna | 168 | 0.81 | 0.94 | 0.94 (LightGBM) | 0.97 (AutoETS) | 0.4 / 0.1 |
| `fuel_gasoline_monthly` | 1 | ninguna | 552 | 0.70 | 1.08 | 0.85 (Ridge diff) | 0.87 (AutoETS) | 0.4 / 0.1 |
| `expenditures_australia_monthly` | 1 | ninguna | 393 | 0.69 | 2.10 | 0.78 (LightGBM diff) | 0.62 (AutoARIMA) | 0.4 / 0.1 |
| `vic_electricity_daily_past_only` | 1 | solo hasta hoy | 927 | 0.61 | 0.79 | 0.78 (DirectMultiVariate) | n/a | 0.5 / 0.1 |
| `bike_sharing_hourly_past_only` | 1 | solo hasta hoy | 8928 | 0.68 | 0.86 | 0.80 (DirectMultiVariate) | n/a | 3.8 / 0.4 |
| `ett_hourly_past_only` | 1 | solo hasta hoy | 8928 | 0.65 | 0.61 | 0.61 (LightGBM sin exog) | n/a | 3.2 / 0.6 |
| `store_sales_50_items` | 50 | ninguna | 1602 | 0.66 | 0.71 | 0.71 (LightGBM) | n/a | 9.6 / 0.9 |
| `bdg2_buildings_energy` | 40 | a futuro | 563 | 0.80 | 0.92 | 0.92 (LightGBM) | n/a | 4.4 / 0.5 |
| `m4_hourly_50_series` | 50 | ninguna | 816 | 0.61 | 0.76 | 0.46 (LightGBM diff) | n/a | 5.3 / 0.5 |
| `australia_tourism_quarterly` | 304 | ninguna | 68 | 0.87 | 0.98 | 0.84 (LightGBM diff) | n/a | 2.4 / 1.2 |

Clasificación (`vic_electricity_classification`): clasificador LightGBM, accuracy 0,874 y
balanced accuracy 0,736; "misma hora ayer", 0,836 y 0,752; "misma hora la semana anterior",
0,747 y 0,565; clase más frecuente, 0,649 y 0,333.

Las tablas con cada modelo y cada historia salen de `summarize.py`.

### 4.4 Cifras globales (todas las historias; `evidence.py`)

- **Una serie** (33 ejecuciones): Chronos-2 tuvo menos error que LightGBM en 26 y que la
  mejor variante de ML en 26; mejoró al naive en 32. LightGBM por defecto mejoró al naive en
  18. Mediana del MASE relativo: Chronos-2 0,68, LightGBM 0,94, mejor ML 0,86. Mediana del
  tiempo: 0,5 s y 0,1 s.
- **Varias series** (8 ejecuciones): Chronos-2 ganó a LightGBM por defecto en 7 y a la mejor
  variante de ML en 5; Chronos-2 mejoró al naive en las 8 y LightGBM por defecto, en 6. LightGBM fue
  de 2 a 11 veces más rápido (mediana 0,5 s frente a 3,5 s).
- **En total** (41 ejecuciones): Chronos-2 mejoró al naive en 40 (falla en `bicimad` con 2
  años de datos semanales); LightGBM por defecto, en 24.
- **Semanal, mensual y trimestral de una serie** (15 ejecuciones): AutoETS o AutoARIMA ganaron
  a Chronos-2 en 8 y a LightGBM por defecto en 12. En los 3 casos diarios con estadísticos
  quedaron por detrás de Chronos-2.
- **Variables solo hasta hoy**: Chronos-2 con ellas, 0,619, 0,700 y 0,647; sin ellas, 0,606,
  0,683 y 0,646. `ForecasterDirectMultiVariate` con esas series como predictores mejoró algo a
  LightGBM sin variables en 2 de 3 casos (0,778 frente a 0,793; 0,797 frente a 0,856), empeoró
  en el tercero (0,732 frente a 0,606) y nunca superó a Chronos-2 sin variables.
- **Historia**: solo en `vic_electricity` invierte el orden de forma consistente (LightGBM
  gana a partir de 6 meses y claramente desde un año; pierde con 3 meses). En `store_sales`
  alterna sin patrón; en el resto no cambia.

### 4.5 Qué decide cada pregunta

| Pregunta | Resultado medido | Decisión |
|:--|:--|:--|
| Qué te importa más | Chronos-2 más preciso en la mayoría; LightGBM de 2 a 11 veces más rápido con varias series y único con importancias | Se queda y decide el orden |
| Cuántas series | No cambia el ganador con la configuración por defecto | Se queda: cambia la clase, el baseline (uno por serie) y el aviso de coste |
| Cuánta historia | No invierte el orden salvo en un caso | **Se elimina**; el resultado avisa de que ML necesita más historia |
| Cada cuánto se registran | Estadísticos competitivos en semanal, mensual y trimestral; no en diario | **Pregunta nueva**: lags, calendario, horizonte, baseline y AutoETS en baja frecuencia |
| Qué más sabes, a futuro | LightGBM gana en `vic_electricity`; Chronos-2 en `bike_sharing` (2) y `bdg2` | Se queda: añade `exog` a los dos candidatos, no cambia el orden |
| Qué más sabes, solo hasta hoy | No mejoran ningún modelo | Se queda la opción, pero el resultado dice que empiece sin ellas |
| Qué quieres predecir | El clasificador gana en accuracy y empata en balanced accuracy con "misma hora ayer" | Se queda: clasificador + regla estacional como baseline |

### 4.6 Límites

- Una configuración por modelo, sin ajuste: un LightGBM con variables de dominio y ajuste
  puede ganar en un dataset concreto.
- Datasets públicos: pueden estar en el preentrenamiento de Chronos-2 y favorecerlo.
- Solo Chronos-2 small, en CPU. Los otros 10 modelos fundacionales no están medidos.
- ML tiene cuatro variantes por caso y Chronos-2 una: "frente a la mejor variante de ML"
  favorece a ML.
- Las fechas de `m4_hourly` son artificiales; se alinearon en un índice común y ese caso no
  usa variables de calendario.
- Diferencias de pocas centésimas no son significativas.

## 5. Revisión de la base

Hallazgos sobre `skills/choosing-a-forecaster/SKILL.md` (S), `tools/ai/llms-base.txt` (L) y
las skills relacionadas, verificados contra el código (C) y contra el estudio (E). Por
gravedad. Las líneas son las del 2026-11-27.

| # | Dónde | Qué está mal | Base | Corrección |
|--:|:--|:--|:--|:--|
| 1 | S:50, 60, 74, 84, 92-93; L:720 | Fundacionales solo para "zero-shot / no training data / cold-start", fuera del árbol; "START HERE" en `ForecasterRecursive` y "Start simple: Begin with ForecasterRecursive" | C, E | "Two candidates and a baseline": Chronos-2 y LightGBM siempre, ordenados por prioridad |
| 2 | S:170, 74 | "LightGBM is the best starting point": por defecto quedó peor que el naive en 17 de 38 ejecuciones (mensuales con tendencia, diarios sin variables) | E | Candidato fuerte con buenas variables; comprobar contra el baseline; `differentiation=1` si hay tendencia |
| 3 | S:49, 130; `statistical-models/SKILL.md:15` | "Short series (< 200 obs) → statistical": lo que decide es la frecuencia, no la longitud | E | En semanal, mensual y trimestral sin variables, AutoETS o AutoARIMA como referencia; quitar el umbral |
| 4 | S:29, 50, 60, 93 | 11 modelos fundacionales sin distinguir licencias; 5 restringen el uso comercial (`commercial_use_restricted=True`: TimesFM 3.0, Moirai-2, TabPFN-TS, TS-ICL, Timestral) | C | Recomendar `autogluon/chronos-2-small` y avisar del resto |
| 5 | S:146 | No distingue variables conocidas a futuro de las conocidas solo hasta hoy; medido, las segundas no ayudan | C, E | "ML needs the future values of `exog`; variables known only up to today rarely help, start without them" |
| 6 | S:36-41, 58, 69 | El paso 1 mezcla número de series y tipo de objetivo; DirectMultiVariate dos veces; varias series categóricas sin hoja (`ForecasterRecursiveClassifier` es `single-series`) | C | Primera pregunta: tipo de objetivo; un clasificador por serie |
| 7 | S:77-79, 176; `baseline-forecasting` | Sin baseline para varias series (EquivalentDate y Stats solo aceptan `pandas.Series`) ni para clasificación | C, E | Bucle por serie y regla estacional (`y.shift(periodo)`), como en los fragmentos |
| 8 | S:59-60, 89-93 | RNN antes que Foundation, contra la regla "Foundation before RNN" de `.claude/rules/docs.md` | C | Foundation dentro y antes; RNN como nota |
| 9 | S:151-157 | Matriz: `ForecasterStats` sin intervalos ni importancias (tiene `predict_interval` y `get_feature_importances`); Foundation sin `predict_interval`; falta la fila de valores ausentes | C | Añadir filas o generar la matriz desde `get_tags()` |
| 10 | S:176 | `Arima(order=None)` sin `m`: con `m=1`, `auto_arima` no busca términos estacionales | C | `Arima(order=None, seasonal_order=None, m=periodo)`; `estimator=[Arima(...), Ets(...)]` |
| 11 | `statistical-models/SKILL.md:24` | Cita un argumento `interval_method` que no existe en `ForecasterStats` ni en `backtesting_stats` | C | "via `interval` or `alpha`" |
| 12 | S:123, 147 | "Per-series differentiation" en DirectMultiVariate: solo acepta `int` | C | "Same order for all series" |
| 13 | S:48, 73; L:67-81; `introduction-forecasting.md:162-183`; `home.html:189` | Jerga ("Does the forecast relationship change across the horizon?") y cinco fuentes distintas para la misma decisión; skforecast-ai tiene su propio motor de reglas (no revisado) | C | Una fuente que genere la skill y el test |
| 14 | S:171, 175, 50 | Calendario como exógena manual (existe `calendar_features`); intervalos con residuos in-sample; "no training data" | C | Corregir redacción |

## 6. Diseño del test (fase 1)

### 6.1 Preguntas

En el orden de crear un forecaster: objetivo, forma de los datos, frecuencia, exógenas y,
al final, la elección del modelo.

| # | Question | Options (hint) | Shown if |
|--:|:--|:--|:--|
| 1 | What do you want to forecast? | **A number** (sales, energy demand, website visits, temperature) · **A category** (low, medium or high demand; failure or no failure) | Always |
| 2 | How many time series do you want to forecast? *A time series is one column of values recorded over time, such as the daily sales of one store.* | **One** (the total sales of a company) · **Several** (the sales of each store or each product) | A number |
| 3 | How often are the values recorded? | Every hour (or more often) · Every day · Every week · Every month · Every quarter (or every year) | Always |
| 4 | Do you know other things that affect the values? *In skforecast these are called exogenous variables.* | **Yes, and I know their future values** (holidays, planned promotions, prices, weather forecasts) · **Yes, but only up to today** (sensor readings, measured weather, sales of other products) · **No, only the dates** | Always |
| 5 | What matters most to you? | **The most accurate forecast** · **Speed and low cost** (many series, frequent forecasts, no GPU) · **Understanding what drives the forecast** (which variables matter and how much) | A number |

### 6.2 Reglas (`finder_tree.decide`)

- Categoría → `ForecasterRecursiveClassifier` con LightGBM; baseline "la categoría del mismo
  periodo una estación antes" (`y.shift(periodo)`), con el horizonte igual a la estación.
- Número:
  - precisión → 1. Chronos-2, 2. LightGBM;
  - coste o explicabilidad → 1. LightGBM, 2. Chronos-2 (con explicabilidad, el código añade
    `get_feature_importances()`);
  - siempre el naive estacional (uno por serie si son varias);
  - una serie semanal, mensual o trimestral → además AutoETS como referencia estadística.
- Las preguntas 2 a 4 cambian el código (clase, lags, calendario, horizonte, `exog`,
  baseline), no las familias recomendadas. Si un estudio futuro encuentra una regla de datos
  fiable (por ejemplo, "variables fuertes conocidas a futuro y un año de historia → ML
  primero"), se añade como pregunta.

Ajustes por frecuencia de los fragmentos (`finder_tree.FREQ`):

| Frecuencia | Horizonte × bloques | Lags | Media móvil | Calendario | Baseline |
|:--|:--|:--|:--|:--|:--|
| Hourly | 24 × 8 | `[1, 2, 3, 24, 48, 168]` | 24 | hour, day_of_week | misma hora del día anterior (24) |
| Daily | 14 × 8 | 14 | 7 | day_of_week, month | mismo día de la semana anterior (7) |
| Weekly | 8 × 6 | `[1, 2, 3, 4, 52]` | 4 | month | misma semana del año anterior (52) |
| Monthly | 12 × 3 | 12 | | month | mismo mes del año anterior (12) |
| Quarterly | 4 × 3 | 4 | | quarter | mismo trimestre del año anterior (4) |

### 6.3 Pantalla de resultado

1. Titular con el orden: "Start with Chronos-2, race it against LightGBM, and compare both
   with a baseline".
2. Respuestas como chips editables: cambiar una recalcula el resultado sin repetir el test.
3. Alineación: primer candidato, segundo, referencia estadística si aplica, baseline, cada uno
   con su clase de skforecast.
4. Por qué, con las cifras del estudio que corresponden a esas respuestas.
5. Ventana de código con el estilo de la portada: barra con tres puntos, `forecast.py`,
   Ubuntu Mono, la paleta de tokens de `docs/stylesheets/home.css` y el resaltador `hl()` de
   `docs/javascripts/home.js` (conviene sacarlo a un módulo compartido); botón Copy y la nota
   "This code runs as is: it is executed on sample data every time the docs are built".
   Nota de licencias: Chronos-2 es Apache-2.0; otros modelos restringen el uso comercial.
6. Acciones: "Copy prompt for your AI assistant" (respuestas, plan, enlace a
   `llms-full.txt` y "ask me for my data first"); "Run it on your own data with
   skforecast-ai"; guías siguientes según las respuestas; consultoría discreta.
7. Enlace compartible con las respuestas en el `#` (sin backend).

### 6.4 Ubicación y enlaces

- Página `docs/user_guides/choose-a-forecaster.md`, primera de *Choosing a model*
  (`mkdocs.yml:44`), antes de *Comparing forecasting models*, con una sección "How we chose
  these rules" que resume el estudio.
- Portada: "Not sure which one to use?" (`docs/overrides/home.html:189`) apunta hoy a la tabla
  de la introducción; cambiar el destino. Valorar un tercer botón en el hero
  (`home.html:68-70`).
- Quick start, introducción (bajo la tabla de forecasters) y *AI-assisted forecasting*.
- Entrada en `tools/docs/hooks/page_meta.yml`.

### 6.5 Fuente única y validación

- Fuente: `finder_tree.py` pasa a YAML (o se queda en Python) dentro de
  `skills/choosing-a-forecaster/references/`, para que `llms-full.txt` lleve el mismo árbol a
  los asistentes.
- Un hook de MkDocs (como `tools/docs/hooks/examples_grid.py`) incrusta preguntas, reglas y
  fragmentos en la página; `generate_ai_context_files.py` reescribe la sección de decisión de
  la skill entre marcadores, y `--check` falla si alguien la edita a mano.
- En cada PR (`--check`, que corre en PR a `*.x`): todas las combinaciones tienen hoja;
  clases y parámetros coherentes con `get_tags()`; `get_model_info()` confirma que el
  fundacional recomendado no restringe el uso comercial; enlaces a páginas del `nav`.
- Ejecución de los 100 fragmentos (`build.py`, minuto y medio en CPU con Chronos-2): un job
  propio con PyTorch, separado del de contexto de IA.
- `benchmark.py`: en cada versión o cuando cambie un forecaster, un adaptador o una regla. Si
  una regla deja de cumplirse, el test se actualiza con la evidencia nueva.

### 6.6 Qué medir

Eventos con consentimiento (`__md_get("__consent")?.analytics`), solo identificadores
cerrados: `finder_start`, `finder_answer {q, a}`, `finder_result {leaf}`,
`finder_copy {kind}`, `finder_cta {target}`; widget de feedback de Material; UTM en los
enlaces salientes.

| Pregunta | Señal | Umbral orientativo |
|:--|:--|:--|
| ¿Lo terminan? | inicio → resultado; abandono por pregunta | más del 60 % |
| ¿Les sirve? | copias de código y prompt, clics en guías | 1 de cada 3 resultados |
| ¿Convierte? | clics a skforecast-ai, Studio, consultoría | frente a los mismos enlaces de la portada |
| ¿Acierta? | "Was this useful?" y distribución de hojas | una hoja con más del 50 % indica una pregunta que no discrimina |

### 6.7 Riesgos

- Autoridad prestada: dos candidatos + baseline y las cifras a la vista.
- Evidencia que caduca: validación periódica con `benchmark.py`.
- Datasets públicos que favorecen a los fundacionales.
- Chronos-2 en producción: PyTorch y GPU recomendada a escala; el resultado lo dice.
- Divergencia con las reglas de skforecast-ai.

## 7. Plan por fases

### Fase 0: corregir la base (independiente del test)

- Corregir `skills/choosing-a-forecaster/SKILL.md` con los hallazgos 1 a 14, citando el
  estudio; `skills/statistical-models/SKILL.md:15` y `:24`; el consejo 6 de
  `tools/ai/llms-base.txt` ("Start with two candidates and a baseline").
- Añadir a `baseline-forecasting` el baseline por serie y el de clasificación.
- Llevar `benchmark/` a `benchmarks/forecaster_finder/` con sus requisitos fijados (las
  cifras de la skill deben ser reproducibles).
- Cierre: skill y `llms-base.txt` regenerados (`ai-context-sync`), `verify` en verde, nota en
  `docs/releases/releases.md`.

### Fase 1: test de elección de forecaster

1. **Fuente única y validación** (6.5): reglas, plantillas, checks en `--check` y job de
   ejecución de fragmentos.
2. **Página estática** `choose-a-forecaster`: diagrama y tabla de reglas generados de la
   fuente, sección "How we chose these rules", enlaces (6.4). Útil por sí sola, indexable.
3. **Capa interactiva** sobre la misma página: preguntas, resultado (6.3), copiar código y
   prompt, enlace compartible. JavaScript propio sin dependencias; el resaltador de
   `home.js` como módulo compartido.
4. **Medición** (6.6) y revisión a las 8 semanas: si no supera los umbrales, se queda la
   página estática.

### Fase 2 del test: probabilistic (más adelante)

Ver la sección 8. Empieza cuando la fase 1 esté publicada y su medición diga que se usa.

### Después

Otros tests con el mismo mecanismo, si la medición lo justifica: métrica de evaluación
(`metric-selection` ya tiene un árbol) y configuración del backtesting (depende mucho del
caso; dudoso).

## 8. Fase 2 del test: probabilistic (más adelante)

### 8.1 Qué es

Una extensión que se **activa** desde el resultado de la fase 1 para elegir el método
probabilístico (intervalos, cuantiles o escenarios) y añadirlo al código. No cambia nada de
la fase 1 si no se activa.

- **En la página**: en el resultado aparece un bloque "Also estimate the uncertainty?" con un
  botón; al pulsarlo se abren 2 o 3 preguntas en el mismo resultado, y la alineación, el
  código y el prompt se amplían con el método elegido. Las respuestas van también al `#`
  (`#finder-...-p-...`), así que el enlace compartido conserva la extensión.
- **En la fuente**: un módulo propio (`probabilistic` dentro de la fuente única) con sus
  preguntas, reglas y plantillas, y un interruptor (`PROBABILISTIC_ENABLED`) para publicarlo
  cuando esté medido. `build.py` genera y ejecuta también sus combinaciones.
- **Depende de la fase 1**: el método disponible depende del forecaster recomendado.

### 8.2 Lo que hay en skforecast (verificado en el código el 2026-11-27)

| Forecaster | Métodos | Salidas |
|:--|:--|:--|
| Recursive, Direct, RecursiveMultiSeries, DirectMultiVariate | `'bootstrapping'` y `'conformal'` (etiqueta `probabilistic_methods`); residuos in-sample (`store_in_sample_residuals` en `fit`) u out-of-sample (`set_out_sample_residuals(y_true, y_pred)`); `use_binned_residuals=True` por defecto | `predict_interval`, `predict_quantiles`, `predict_dist`, `predict_bootstrapping` (trayectorias) |
| Foundation | Cuantiles nativos del modelo (Chronos-2: cualquier nivel; otros, un conjunto fijo) | `predict_interval`, `predict_quantiles`; **sin** trayectorias (`predict_bootstrapping` no existe) |
| Stats | Paramétrico del modelo (`interval` o `alpha`) | `predict_interval` |
| EquivalentDate, Rnn | Solo `'conformal'` | `predict_interval` |
| RecursiveClassifier | Probabilidades de clase | `predict_proba` |
| Cualquiera | Regresión cuantil (un `ForecasterDirect` por cuantil con `objective='quantile'`, guía `probabilistic-forecasting-quantile-regression`) y recalibración con `ConformalIntervalCalibrator` | Cuantiles; intervalo recalibrado |

Backtesting: `interval` acepta una cobertura (`0.8`), una lista de cuantiles,
`'bootstrapping'` (trayectorias) o una distribución de `scipy.stats`; con un intervalo,
`metric` devuelve `coverage` e `interval_width`. Métricas: `calculate_coverage`,
`crps_from_predictions`, `crps_from_quantiles`, `create_mean_pinball_loss`, `winkler_score`,
`weighted_interval_score`.

Lo que ya mide la documentación (`docs/user_guides/probabilistic-forecasting-overview.md`):

- intervalo nominal del 80 % con residuos in-sample: cobertura empírica de alrededor del 61 %;
- bootstrapping en modelos globales: entre el 96 % y el 98 %; conformal, alrededor del 84 %;
- bootstrapping con 300 series, unas 35 veces más lento que conformal.

### 8.3 Preguntas (borrador, en inglés)

Se validan con el estudio de 8.5: la que no cambie el método ganador, se quita.

| # | Question | Options (hint) | Qué cambiaría (hipótesis) |
|--:|:--|:--|:--|
| P1 | What do you need besides the forecast? | **A range** (a low and a high value for each date) · **Several percentiles** (e.g. 10%, 50% and 90%, for stock or capacity planning) · **Possible scenarios** (e.g. the total of next week, or the chance of running out) | Intervalo → `predict_interval`; percentiles → `predict_quantiles` o regresión cuantil; escenarios → `predict_bootstrapping`, que solo existe en ML |
| P2 | Can you set aside a recent period to measure the model's real errors? *For example, the months just before your test period. skforecast calls them out-of-sample residuals.* | **Yes** · **No, my history is short** · **I'm not sure** | Sí → residuos out-of-sample (recomendado); no → cuantiles nativos de Chronos-2 o paramétricos de Stats, con aviso de que los residuos in-sample dan intervalos estrechos |
| P3 | How sure do you need to be? | **80%** · **90%** · **95%** | Solo cambia `interval` en el código; quizá baste con un valor por defecto editable en vez de una pregunta |
| (P4) | Does the uncertainty grow with the size of the values? *For example, bigger sales come with bigger errors.* | Yes · No · I'm not sure | `use_binned_residuals`. Candidata a desaparecer si el estudio muestra que el valor por defecto (`True`) nunca empeora |

Reglas iniciales (hipótesis a medir):

- Candidato Chronos-2 → cuantiles nativos para intervalo y percentiles; para escenarios,
  pasar a LightGBM con bootstrapping (Chronos-2 no da trayectorias).
- Candidato LightGBM:
  - intervalo → conformal con residuos out-of-sample (el más barato, cerca del nominal según
    la documentación);
  - percentiles → `predict_quantiles` con bootstrapping, frente a regresión cuantil con
    `ForecasterDirect`;
  - escenarios → bootstrapping (`predict_bootstrapping`);
  - sin periodo de validación → avisar y proponer el candidato fundacional o el estadístico.
- Referencia estadística → intervalos paramétricos.
- Baseline (EquivalentDate) → conformal, para comparar también la cobertura.
- Categoría → `predict_proba` (sin preguntas).

### 8.4 Pantalla de resultado (ampliación)

- Una tarjeta más en la alineación: "Uncertainty: conformal intervals with out-of-sample
  residuals", o "native quantiles", etc.
- Bloques nuevos en la ventana de código:
  - el backtest de calibración y `set_out_sample_residuals`, cuando aplique;
  - `predict_interval` o `predict_quantiles`;
  - el backtest con `interval=...`, que devuelve `coverage` e `interval_width`;
  - para escenarios, `predict_bootstrapping` y la suma de las trayectorias.
- "Por qué" con las cifras del estudio de 8.5 y la regla de oro: comprobar que la cobertura
  empírica se acerca a la nominal.
- Enlaces: la guía del método elegido, *Probabilistic forecasting overview*, *Prediction
  interval metrics*.

### 8.5 Estudio que la habilita

Ampliar `benchmark.py` con los mismos 13 datasets y bloques de test:

- **Métodos**: bootstrapping y conformal con residuos in-sample y out-of-sample, con y sin
  bins; regresión cuantil con `ForecasterDirect`; cuantiles nativos de Chronos-2;
  paramétricos de AutoETS y AutoARIMA; conformal del naive estacional.
- **Métricas**: cobertura empírica de 80 % y 95 %, anchura del intervalo, WIS o CRPS, pérdida
  pinball por cuantil y tiempo. Para escenarios, la cobertura del total del horizonte
  (suma de cada trayectoria frente a la suma real).
- **Preguntas que debe contestar**: ¿cambia el mejor método según lo que se pide (P1)?
  ¿Cuánto pierden los residuos in-sample frente a out-of-sample en cada familia (P2)? ¿Los
  bins empeoran alguna vez (P4)? ¿Los cuantiles nativos de Chronos-2 están calibrados sin
  ajuste, y mejoran con `ConformalIntervalCalibrator`?
- **Plantillas**: cada combinación nueva entra en `build.py` y se ejecuta con datos
  sintéticos.

### 8.6 Cuándo se activa

- Fase 1 publicada y con uso medido.
- Estudio de 8.5 hecho y reglas que lo reflejan; preguntas que no cambian el resultado,
  quitadas.
- Todas las combinaciones de la fase 1 por las de la fase 2 con fragmento ejecutado sin error.
- La skill `prediction-intervals` y su `references/interval-compatibility.md` corregidas con
  lo medido (falta `ForecasterFoundation` en la matriz de compatibilidad, y la tabla de
  `skills/prediction-intervals/SKILL.md:16-20` no incluye los cuantiles nativos), y
  regeneradas.

## 9. Decisiones abiertas

1. **ML primero o evidencia primero**: con la evidencia actual, la prioridad "precisión" pone
   primero a Chronos-2. Si se quiere reforzar ML, el camino es medir LightGBM con
   diferenciación, más lags y variables de dominio, no cambiar la regla sin evidencia.
2. **Qué fundacionales recomendar**: solo Chronos-2 está medido y es Apache-2.0. ¿Medir
   TiRex-2, TabICL o T0 (también sin restricciones comerciales) antes de mencionarlos?
3. **Analítica**: GA4 con consentimiento (ya está) o migrar todo el sitio a una opción sin
   cookies (Plausible, GoatCounter).
4. **Dónde vive el estudio**: `benchmarks/forecaster_finder/` (reproducible, con requisitos
   fijados) o solo citado por commit cuando se borre esta carpeta.
5. **Reglas de skforecast-ai**: revisarlas contra este árbol para que el test y la
   herramienta no recomienden cosas distintas con los mismos datos.
