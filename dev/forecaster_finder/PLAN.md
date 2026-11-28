# Selector de forecaster ("Not sure where to start?"): plan de desarrollo

## 1. Estado

- **Implementado el 2026-11-28 (fases 0, 1 y 2), en la rama `docs/forecaster-finder`: PR
  [#1550](https://github.com/skforecast/skforecast/pull/1550) contra `0.28.x`, con la
  rama base ya fusionada.** D9 y D10 aceptadas: dos candidatos con papeles distintos ("Compare two
  forecasters: LightGBM and Chronos-2") y página en la pestaña *Quick start*. Queda la fase
  3 (medición) y lo de 7.5.
- Este documento conserva el análisis y las decisiones. La descripción de lo construido está
  en `tools/docs/forecaster_finder/README.md` (selector) y en
  `benchmarks/forecaster_finder/README.md` (estudio); donde difieran de este plan, valen
  ellos.
- En la misma rama están arreglados los dos fallos de `Ets` que encontró la revisión (commit
  `5f020a0`, con tests y notas de versión).
- Filas del roadmap: `DOC-choosing-forecaster-evidence` (fase 0, hecha),
  `FEAT-forecaster-finder` (fases 1 y 2 hechas, fase 3 pendiente),
  `FEAT-finder-probabilistic` (fase 4), `BUG-ets-freeze-params` y `QW-ets-m-over-24-silent`
  (arregladas en esta rama) y `QW-shared-cv-refit-warnings`.

Dónde está cada cosa:

| Ruta | Qué es |
|:--|:--|
| `docs/quick-start/choose-a-forecaster.md` | La página "Which forecaster should I use?" |
| `docs/javascripts/finder.js`, `docs/stylesheets/finder.css`, `docs/overrides/finder.html` | El selector, sus estilos y la plantilla de la página |
| `docs/javascripts/finder-data.json` | Generado: los 105 estados posibles y los 36 scripts |
| `tools/docs/forecaster_finder/` | Fuente única (`finder.py`), generador (`build.py`), comprobaciones y README |
| `benchmarks/forecaster_finder/` | El estudio (`study.py`, `cases.py`, `cases_d4.py`, `rules.py`), sus resultados y README |
| `.github/workflows/forecaster-finder.yml` | CI: ficheros generados al día y todos los scripts ejecutados, en las PR que tocan el finder |
| `dev/forecaster_finder/PLAN.md`, `REVISION.md` | Este plan y la revisión independiente |
| `dev/forecaster_finder/benchmark/` | Restos de la versión 1 (ML congelado): resúmenes, sus CSV y la reproducción del fallo de `Ets`. No se citan en ninguna página |

Entorno de las cifras: Python 3.12.3, skforecast 0.28.0, numpy 2.5.5, pandas 3.1.1,
scikit-learn 1.9.1, scipy 1.18.1, numba 0.68.1, LightGBM 4.7.0, PyTorch 2.15.0 (CPU),
chronos-forecasting 2.3.2.

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
  LightGBM en 12; AutoETS nunca quedó peor que el naive. En trimestral (304 series de
  `australia_tourism`, un AutoETS por serie, medido con el arreglo de `Ets`): 0,859 frente a
  0,866 de Chronos-2 y 0,925 de LightGBM. En diario (13 ejecuciones), AutoETS no ganó nunca
  a Chronos-2. En semanal, `Ets(m=52)` no es estacional (ahora avisa).
- El LightGBM tal como lo escribe el fragmento del selector (`review.py --snippet`): peor
  que el naive en 14 de 38 ejecuciones, las 14 con una serie y nada conocido a futuro.
- Controles (detalle en `REVISION.md`): Chronos-2 synth mejora al naive en 15 de 15 y a
  LightGBM en 10 de 15; el modelo base no mejora al small de forma consistente (8 de 15);
  un horizonte 4 veces más corto no cambia el orden (9 de 12 en los dos); `LightGBM+` no
  cambia el orden (Chronos-2 gana en 27 de 38); en `bdg2`, Chronos-2 gana en los 9 sitios o
  muestras probados.
- Variables solo hasta hoy (3 casos, 3 tamaños de Chronos-2): de un 14 % menos de error a un
  4 % más. No hay regla.

### 4.4 Qué decide cada fila (con el estudio ampliado de 4.6)

| Fila | Resultado medido | Decisión |
|:--|:--|:--|
| Qué quieres predecir | Solo el clasificador predice categorías; no supera a "misma hora ayer" en balanced accuracy (1 dataset) | Se queda, por capacidad |
| Cuántas series | Chronos-2 por delante en 8 de 10 datasets con varias series, igual que con una | Se queda: cambia la clase, el baseline y la frase, no el orden |
| Un valor por... (frecuencia) | ARIMA supera al LightGBM sin ajustar en 6 de 8 datasets mensuales y trimestrales; en diario no aporta | Se queda: constantes del código y ARIMA como referencia en mensual y trimestral |
| Qué sabrás a futuro | Chronos-2 por delante en 5 de 7 datasets; es donde LightGBM queda más cerca | Se queda: añade `exog` y cambia la frase, no el orden |
| Qué sabes solo hasta hoy | Sin ganancia consistente | Se funde con "No" |
| Qué te importa más | No ordena por precisión | Quitada (D3) |
| Cuánta historia, qué horizonte | No cambian el orden | Sin fila |

### 4.5 Límites

- Pocos datasets por tipo: 8, 2 y 4. La regla con variables conocidas a futuro es la más
  débil, y sus variables son valores observados, no pronósticos.
- Una configuración por modelo, sin ajuste: no dice cuánto se acerca un LightGBM trabajado.
- Chronos-2 en CPU; el coste de producción no está medido.
- AutoETS y AutoARIMA medidos con `freeze_params=True` (valor por defecto): modelo y
  parámetros de suavizado del primer ajuste, estados reestimados en cada bloque.
- Diferencias de pocas centésimas no son significativas: en varios casos Chronos-2 gana la
  mitad de los bloques.

### 4.6 Estudio ampliado (D4): 26 datasets

Se añadieron 20 casos (`cases_d4.py`) con el mismo protocolo y solo los modelos que usan las
reglas: naive estacional, LightGBM del estudio, `LightGBM+`, el LightGBM del fragmento,
Chronos-2 small, y AutoETS y AutoARIMA en las series sueltas. Un voto por dataset en cada
tipo de datos (la media de sus ejecuciones); LightGBM es el que escribe el fragmento.

| Tipo de datos | Datasets | Chronos-2 con menos error | Con más de un 5 % de ventaja | LightGBM con más de un 5 % de ventaja | LightGBM peor que el naive | Chronos-2 peor que el naive |
|:--|--:|--:|--:|--:|--:|--:|
| Una serie, nada conocido a futuro | 16 | 14 | 13 | 0 | 7 | 0 |
| Una serie, variables conocidas a futuro | 7 | 5 | 4 | 1 | 1 | 0 |
| Varias series | 10 | 8 | 7 | 1 | 2 | 0 |

- **Con más datasets, Chronos-2 va por delante en los tres tipos de datos**, no solo en el
  primero. La conclusión anterior ("sin favorito" con variables conocidas a futuro o con
  varias series) salía de 2 y 4 datasets y no se sostiene con 7 y 10.
- Frente a `LightGBM+`: 12 de 13, 4 de 7 y 8 de 10. Tampoco cambia el cuadro.
- Variables conocidas a futuro, por tipo: de calendario o planificadas (`h2o_exog`, eventos
  de M5), Chronos-2 por delante en 1 de 2 datasets; meteorología observada, en 4 de 5.
  LightGBM gana en `vic_electricity` y `h2o_exog`, por un 4 % y un 6 %.
- Datasets nuevos: `website_visits`, `uk_daily_flights`, `wikipedia_visits`,
  `air_passengers`, `elnino`, `macrodata` (2 series), `co2`, `h2o_exog`, M5 (una tienda y 10
  tiendas, con eventos), `turbine_emission`, `air_quality_valencia`, `items_sales`,
  `ett_m1`, `m4_daily`; y casos nuevos de `bdg2`, `australia_tourism` y
  `public_transport_madrid`.

Referencia estadística, con una serie (AutoETS frente a AutoARIMA):

| Frecuencia | Datasets | AutoETS con menos error que AutoARIMA | Menos error que Chronos-2 (ETS / ARIMA) | Menos error que LightGBM (ETS / ARIMA) | Peor que el naive con toda la historia (ETS / ARIMA) |
|:--|--:|--:|:--|:--|:--|
| Mensual | 6 | 4 | 0 / 1 | 4 / 4 | 1 de 6 / 3 de 6 |
| Trimestral | 2 | 1 | 1 / 2 | 2 / 2 | 0 / 0 |
| Semanal | 2 | 0 (`Ets` no es estacional con `m=52`) | 0 / 1 | 1 / 1 | |
| Diario | 8 | 4 | 0 / 1 | 3 / 2 | |

- En las 304 series trimestrales de `australia_tourism`, un modelo por serie: AutoETS,
  mediana del error relativo 0,84 (mejor que Chronos-2 en el 52 % de las series, 39 s);
  AutoARIMA, 0,86 (50 %, 131 s); Chronos-2, 0,88 (2,4 s).
- No hay una diferencia clara de precisión entre los dos. AutoETS falla menos frente al
  naive en mensual; AutoARIMA es el modelo de la portada, el más reconocido y el único de
  los dos que sirve en semanal. **Se usa ARIMA** (decisión del 2026-11-27).
- ARIMA con las variables conocidas a futuro (5 casos): mejora en 2 (0,71 frente a 0,81 en
  `bike_sharing`; 0,78 frente a 1,01 en `h2o_exog`) y empeora en 2 (1,56 frente a 0,92 en
  `vic_electricity`; 0,98 frente a 0,77 en M5). El fragmento lo usa sin ellas.
- Ya no es cierto que el modelo estadístico "empate con Chronos-2" en mensual (salía de 3
  datasets): le gana en 0 o 1 de 6. Lo que sí respalda el estudio es que supera al LightGBM
  sin ajustar en 6 de 8 datasets mensuales y trimestrales.

### 4.7 LightGBM con un presupuesto de ajuste

Pregunta: el titular acordado dice "then try to beat it with LightGBM". ¿Cuántas veces lo
consigue quien hace un ajuste ligero y automático?

- Método (`review.py --tune`): `bayesian_search_forecaster` (o la versión multiserie) con 20
  pruebas, 10 sin diferenciar y 10 con `differentiation=1`; busca los lags (los del caso o
  los del LightGBM+), `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`,
  `colsample_bytree` y `reg_lambda`; métrica MASE en validación (hasta 3 bloques al final
  del entrenamiento, nunca el test); después, el mismo backtest de test que el resto, con
  reentrenamiento. Toda la historia. 33 pares de dataset y tipo de datos. Resultados en la
  fila "LightGBM tuned [refit]" de cada CSV y en `rules_tuned.csv`.

| Tipo de datos | Datasets | Sin ajustar, menos error que Chronos-2 | Ajustado, menos error que Chronos-2 | Ajustado, a un 5 % o mejor | El ajuste mejora a LightGBM | El ajuste lo empeora más de un 5 % | Mediana: Chronos-2 / sin ajustar / ajustado |
|:--|--:|--:|--:|--:|--:|--:|:--|
| Una serie, nada conocido a futuro | 16 | 2 | 4 | 5 | 9 | 5 | 0,69 / 0,90 / 0,79 |
| Una serie, variables conocidas a futuro | 7 | 2 | 1 | 1 | 1 | 5 | 0,66 / 0,76 / 0,93 |
| Varias series | 10 | 2 | 2 | 4 | 8 | 1 | 0,72 / 0,90 / 0,90 |
| **Total** | **33** | **6** | **7** | **10** | **18** | **11** | |

Lectura:

- Un ajuste ligero no cambia el orden: LightGBM ajustado tiene menos error que Chronos-2 en
  7 de 33, frente a 6 de 33 sin ajustar. La mediana del cociente entre el error del ajustado
  y el de Chronos-2 es 1,12.
- Donde gana, gana de verdad: `australia_tourism` (0,67 frente a 0,74; sin ajustar, 1,38),
  `co2` (0,26 frente a 0,38), `expenditures_australia` (0,59 frente a 0,69), `macrodata`
  (0,44 frente a 0,47), `m4_hourly` (0,36 frente a 0,61), `m4_daily` y `vic_electricity`.
  Cuatro de los siete son series mensuales, trimestrales o semanales con tendencia, donde
  diferenciar es lo que cambia el resultado.
- El ajuste automático no es gratis: empeora a LightGBM más de un 5 % en 11 de 33, y en 5 de
  los 7 datasets con variables conocidas a futuro. Con poca historia, 3 bloques de
  validación no bastan para elegir entre 20 configuraciones.
- Ni eligiendo a posteriori el mejor de los dos LightGBM (algo que el usuario no puede
  hacer) pasa de 10 de 33.
- Límites: 20 pruebas, un solo espacio de búsqueda, sin variables nuevas ni conocimiento del
  problema. No mide lo que consigue alguien que conoce sus datos; mide lo que consigue
  "ejecuta una búsqueda".

Consecuencia: "then try to beat it" promete una carrera que el usuario pierde unas dos de
cada tres veces aunque ajuste. No es un motivo para esconder a LightGBM, sino para cambiar
el argumento con el que se presenta (6.9).

## 5. Correcciones de la base

Hallazgos sobre `skills/choosing-a-forecaster/SKILL.md` (S), `tools/ai/llms-base.txt` (L) y
relacionados, verificados contra el código el 2026-11-27 (`REVISION.md` H11).

| # | Dónde | Qué está mal | Corrección |
|--:|:--|:--|:--|
| 1 | S:50, 60, 74, 84, 92-93; L:724 | Fundacionales solo para "zero-shot / no training data / cold-start", fuera del árbol; "START HERE" en `ForecasterRecursive`; "Start simple: Begin with ForecasterRecursive" | Las reglas de 6.2: Chronos-2 primero, LightGBM como el modelo que se trabaja, y un baseline |
| 2 | S:170, 74 | "LightGBM is the best starting point" | "LightGBM is the ML candidate; with a single series and no variables known in advance it did worse than the seasonal naive in 11 of 22 runs: always compare it with the baseline" |
| 3 | S:49, 130; `statistical-models/SKILL.md:15` | "Short series (< 200 obs) → statistical" | "For a monthly or quarterly series, a classic statistical model (ARIMA, ETS) is a strong reference: it often beats an untuned ML model"; quitar el umbral; no extenderlo a diario |
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

Un selector de una pantalla, no un test por pasos: cuatro filas de opciones, sin nada
elegido al llegar (D7), y el resultado, que se rellena con cada clic. Título del bloque:
"Build your first comparison"; el gancho en la portada es "Not sure which one to use?"
(6.5).

| # | Row | Options (hint) | Shown if |
|--:|:--|:--|:--|
| 1 | What do you want to forecast? | **A number** (sales, energy demand, website visits) · **A category** (low, medium or high demand; failure or no failure) | Always |
| 2 | How many time series? *A time series is one column of values recorded over time, such as the daily sales of one store.* | **One** · **Several** (each store, each product) | A number |
| 3 | Your data has one value per... *skforecast calls this the frequency of the series.* | Hour · Day · Week · Month · Quarter · Something else (15 minutes, a year...) | Always |
| 4 | Will you know anything else in advance? *Things that affect the values and that you know for future dates. skforecast calls them exogenous variables.* | **Yes** (holidays, planned promotions, prices, weather forecasts) · **No** (also if you only know them up to today, like sensor readings) | Always |

Sin selección por defecto: al llegar no hay nada elegido. Las filas van numeradas y se
pueden contestar en cualquier orden.

Bajo el título del bloque, una frase fija: "We tested each kind of model on 26 public
datasets, always with the same backtesting, to see where each one works best. Your answers
pick a starting point; the comparison on your own data decides." (el número lo pone
`build.py`).

### 6.2 Reglas

Un solo titular para los números. Las filas cambian la frase que lo acompaña y el código.

| Respuestas | Titular | Frase que lo acompaña (sin cifras) |
|:--|:--|:--|
| Una categoría | "Start with a machine learning classifier and compare it with a simple rule" | "A classifier is the model for categories. In our test it did not clearly beat the simple rule, so keep the rule as your baseline." |
| Un número, una serie, No | "Start with Chronos-2, then try to beat it with LightGBM" | "In our tests on single series like yours, Chronos-2 was usually the most accurate without any setup, and LightGBM often needed work on its features to beat the baseline." |
| Un número, una serie, Sí | El mismo | "In our tests Chronos-2 was ahead more often, also with variables known in advance, but this is where LightGBM came closest and sometimes won." |
| Un número, varias series | El mismo | "In our tests with several series Chronos-2 was ahead more often, and LightGBM won in some datasets." |
| Además: una serie, mensual o trimestral | "..., and keep ARIMA as a reference" | "ARIMA is the classic reference for monthly and quarterly data: in our tests it often did better than LightGBM out of the box." |
| Something else (frecuencia) | Los mismos | Código con `steps`, `season` y los lags marcados para editar |

- **Las cifras no van en el resultado.** Están en la sección "How we chose these rules" de
  la página (la tabla de 4.6, generada), con los límites del estudio.
- **Cada frase se comprueba contra el estudio** en cada compilación (`build.evidence()`): si
  los resultados dejan de respaldarla (por ejemplo, Chronos-2 deja de ir por delante en la
  mayoría de los datasets de un tipo), la compilación falla y la frase se reescribe.
- Notas fijas, plegadas bajo "Good to know": explicabilidad (solo LightGBM da importancias),
  coste (Chronos-2 necesita PyTorch y, a escala, GPU) y licencias (Chronos-2 es Apache-2.0).
- Los tres modelos del selector son los de la portada: LightGBM, Chronos-2 y ARIMA.

### 6.3 Requisitos del fragmento

El prototipo (`finder_tree.py`) ya los cumple; cada uno tiene su comprobación.

| # | Requisito | Cómo se cumple en el prototipo | Comprobación |
|--:|:--|:--|:--|
| 1 | Es lo que se midió | El bloque de ML es `LGBMRegressor(min_child_samples=5)` con los lags, la media móvil y el calendario de la frecuencia; `transformer_series=StandardScaler()` solo con varias series | `review.py --snippet` lo mide tal como se escribe: peor que el naive en 14 de 38 ejecuciones (antes 17 de 38), nunca con varias series |
| 2 | Todos los modelos usan lo anterior a cada bloque | `TimeSeriesFold(..., refit=True, fixed_train_size=False)` | Lectura del fragmento |
| 3 | Sin avisos | Las llamadas de Chronos-2 y de ARIMA llevan `suppress_warnings=True` (la primera, por H21 de `REVISION.md`; la segunda, por los avisos de convergencia de los órdenes que prueba la selección automática); `series` es un dict | `build.py` falla con cualquier aviso: 0 de 36 |
| 4 | Historia mínima razonable y escrita en la página | 2 semanas de datos horarios, 3 meses de diarios, 2 años de semanales, 3 años de mensuales y 6 años de trimestrales | `check_snippets.py`: los 18 tipos de fragmento pasan |
| 5 | Valores de ejemplo, no una recomendación | Cabecera "Example settings so that the script runs: they are not tuned" y `steps` y `season` como variables; sin consejos de configuración | Lectura |
| 6 | ARIMA solo en mensual y trimestral de una serie, sin exógenas | `Arima(order=None, seasonal_order=None, m=season)`, como en la portada | `build.py` |
| 7 | Instalación y pesos | Línea `pip install skforecast lightgbm chronos-forecasting` y aviso de que la primera ejecución descarga los pesos | Lectura |
| 8 | Todo lo que se calcula se imprime | Tabla final ordenada por error | `build.py` |
| 9 | Pocos fragmentos | 36 combinaciones, 36 fragmentos (24 de número y 12 de categoría, con "Something else") | `build.py`, 30 s |

Pendiente para la fase 1: el horizonte y el número de bloques del fragmento no son los del
estudio (medido: el horizonte no cambia el orden entre familias).

### 6.4 Pantalla de resultado

Dos piezas, en este orden. Solo la primera es llamativa.

**1. Recuadro de respuesta** (fondo oscuro, como el hero de la portada). Se construye con
cada respuesta, y el bloque que cambia se ilumina un momento:

- Sin respuestas: tres bloques vacíos y "Answer the first question to see which models to
  try", con el contador "0 of 4 answered".
- Tras la primera: el titular y los bloques. **Cada bloque lo encabeza su forecaster**
  (skforecast proporciona forecasters, no modelos), en la letra más grande del bloque;
  debajo, el modelo que envuelve ("with Chronos-2"), una línea de para qué sirve y el
  enlace a la guía de usuario del forecaster. Encima, su papel (First, Challenger,
  Reference).
- **El baseline no es un bloque**: es igual en todos los casos y como bloque era ruido. Va
  en una línea bajo los bloques, "Always compared with a seasonal baseline: the same day the
  week before. `ForecasterEquivalentDate` · User guide", que cambia con la frecuencia. En el
  script sigue como un modelo más.
- Mientras no se sabe cuántas series hay, el bloque de LightGBM enseña sus dos forecasters
  posibles (`ForecasterRecursive`, one series, o `ForecasterRecursiveMultiSeries`, several
  series), cada uno con su guía; "How many time series?" deja el que corresponde. La
  frecuencia dice qué repite el baseline y, en mensual y trimestral de una serie, añade el
  bloque de ARIMA; "Yes" a la cuarta pone la etiqueta "uses your known variables".
- Con todas: la frase de 6.2 y el enlace a "How we chose these rules".

Los estados parciales (105) los genera `finder_tree.blocks()`: la página no repite las
reglas.

**2. Script.** La ventana de código se rellena poco a poco. Mientras faltan respuestas
muestra, como comentarios, la línea "# Your script appears here when you answer the 2
questions left." y un esquema de lo ya decidido ("# 1. Chronos-2 (ForecasterFoundation)"...).
Al completarlas, las líneas del script entran una tras otra (menos de un segundo, y sin
animación si el usuario la tiene desactivada). Si después se cambia una respuesta, solo se
iluminan las líneas que cambian. No se enseña código a medias que se pueda copiar. Con el
script completo:

1. "What the script does", en tres líneas.
2. La ventana de código, con los imports plegados ("imports (16 lines)") y el botón Copy.
3. "Runs as it is: tested on sample data every time the docs are built. Needs at least N
   values".
4. Dos botones: "Copy script" y "Run it in Google Colab".
5. Plegado, en este orden: "Let skforecast-ai compare more models for you", "Using an AI
   assistant?" y "Good to know".

**Google Colab**: un único notebook con dos celdas, la de instalación y otra vacía con
"Paste your script here". El botón copia el script **con un bloque de datos de ejemplo
delante** (`finder_tree.sample_block`) y abre el notebook, así que lo pegado se ejecuta sin
tocar nada; el bloque dice "Replace this block with your own data". Esos datos son los
mismos con los que CI ejecuta cada script (en un espacio de nombres vacío, como en un
notebook nuevo). El notebook lo escribe `build.py` (`tools/docs/forecaster_finder/first_comparison.ipynb`);
el enlace solo funciona cuando esté en una rama pública.

**skforecast-ai** (plegado, solo en los casos de número): unas líneas que continúan el
script, con los mismos datos y el mismo `cv`:
`ForecastingAssistant().compare(data=y, cv=cv)` y `print(result.results)`. El texto dice
que es un paquete aparte, que elige la configuración de cada modelo a partir de los datos y
prueba más, y que su clasificación puede diferir. No hace falta que sus reglas coincidan
con las del selector. Comprobado con `python -I build.py --ai` en un entorno aparte
(skforecast-ai 0.5.0, skforecast 0.27.0, pandas 2.3.3): los 24 casos se ejecutan; con datos
horarios lleva `confirm_cost=True`. skforecast-ai no predice categorías: ahí no aparece.

**Asistentes de IA** (plegado), dos caminos según el asistente:

- Si admite MCP (Claude Code, Cursor, VS Code, Codex, Claude Desktop): conectar el servidor
  MCP de skforecast-ai (enlace a su guía) y enviarle un mensaje en lenguaje natural que el
  selector redacta con las respuestas, por ejemplo "Compare forecasting models for the next
  14 days of the column sales of data/my_data.csv, against a seasonal naive baseline, and
  show me the script of the best one." Es el camino preferente: el agente llama a
  herramientas deterministas y devuelve el script que se ejecutó.
- Si no lo admite (un chat sin herramientas): "Copy script as a prompt", que incluye el
  script probado y le dice al asistente que pida antes una muestra de los datos, que cambie
  solo la carga de datos y los valores de ejemplo, y que no toque los forecasters hasta que
  el script se ejecute.

Además: enlace compartible con las respuestas en el `#` (solo con las respuestas completas);
debajo del bloque, "How we chose these rules"; sin JavaScript se ve la página estática
completa (tabla de reglas y scripts de ejemplo).

### 6.5 Ubicación y enlaces (propuesta del 2026-11-28, decisión D10)

**Recomendación: página propia en la pestaña *Quick start*, segunda entrada, y la mayor
parte del esfuerzo en los enlaces que llevan a ella, no en el menú.** Sustituye a la
propuesta anterior (primera página de *User Guides → Choosing a model*).

Hechos de la documentación actual (leídos en `mkdocs.yml` y `docs/overrides/home.html`):

- La pestaña *Quick start* tiene 5 páginas y su primera página es el notebook *Quick start*,
  el destino del botón principal de la portada ("Get started", `home.html:68`). Enseña
  `ForecasterRecursive` con LightGBM y no dice por qué ese forecaster.
- *User Guides* tiene 52 páginas; la pestaña abre en *Table of contents*. La sección
  *Choosing a model* tiene una sola página, *Comparing forecasting models* (un notebook
  largo con el mismo tema, visto desde "cómo comparar bien").
- En la portada, justo debajo de las tres familias de modelos, hay una frase suelta: "Not
  sure which one to use? Compare all forecasters in the Introduction" (`home.html:189`). Es
  el punto exacto en el que surge la pregunta, y hoy lleva a una tabla.
- La página necesita ancho completo (preguntas a la izquierda, respuesta y código a la
  derecha a partir de 980 px): `hide: [navigation, toc]` o una plantilla propia, como la
  portada. Sirve en cualquier pestaña.
- `navigation.instant` está activo: el script del selector tiene que inicializarse en cada
  navegación (`document$.subscribe`), no solo al cargar.
- La documentación está versionada con `mike`: el enlace de Colab y las cifras del respaldo
  son por versión.

Opciones comparadas:

| Opción | A favor | En contra | Veredicto |
|:--|:--|:--|:--|
| **A. *Quick start*, segunda entrada** | Es una herramienta de primera visita, y esa pestaña es donde va quien empieza; menú corto (6 entradas), así que se ve; la pregunta "¿qué forecaster?" va justo antes o justo después del primer ejemplo; no cambia el destino de "Get started" | La pestaña mezcla tutorial y referencia (parámetros, atributos); el selector propone Chronos-2 y LightGBM y el notebook de al lado solo enseña LightGBM (se resuelve con una frase en cada uno) | **Recomendada** |
| B. *User Guides → Choosing a model*, primera | Agrupa por tema con *Comparing forecasting models*; la sección deja de tener una sola página | Una entrada entre 52, en la pestaña a la que se va cuando ya se sabe qué se busca; quien no sabe qué forecaster usar no abre las guías | Enlazarla desde ahí, no alojarla ahí |
| C. Pestaña propia | Máxima visibilidad | Ya hay 11 pestañas; es una página, no una sección; da a una herramienta de arranque el peso de un producto | No |
| D. Incrustado en la portada | Cero clics | La portada ya es larga; el selector necesita ancho y altura; sin URL propia no se puede enlazar ni medir aparte | Solo el gancho, no el selector |
| E. Sección de *Intro to Forecasting* o del notebook *Quick start* | Contexto natural | Páginas largas: el selector quedaría a media página y sin ancho completo | Enlace, no alojamiento |

Lo que pesa más que el menú: de dónde llega la gente. Una página nueva no recibe visitas
por estar en un menú; las recibe por los enlaces desde las páginas que ya tienen tráfico.
Por orden de valor esperado:

1. **Portada, bajo las familias de modelos** (`home.html:189`): la frase suelta pasa a ser
   un bloque visible con un botón, "Not sure which one to use? Answer four questions and
   get a script that runs." Es el mejor sitio de toda la documentación para el gancho.
2. **Notebook *Quick start*, arriba**: una nota de dos líneas ("This guide uses
   `ForecasterRecursive`. Not sure it is the right forecaster for your data? ..."). Es la
   página de entrada de quien pulsa "Get started".
3. **Portada, "Start here"** (`home.html:288`): una tarjeta más, después de *Quick start*.
4. **Introducción**, debajo de la tabla de forecasters, y **README** del repositorio.
5. ***User Guides***: enlace en la tarjeta *Choosing a model* de *Table of contents* (con
   `extra_links` de `tools/docs/hooks/toc_cards.yml`, que ya existe para esto) y una nota al
   principio de *Comparing forecasting models*. En el otro sentido, el resultado del
   selector enlaza a esa guía como paso siguiente ("compare more models properly").
6. *AI-assisted forecasting* y la skill `choosing-a-forecaster` (la URL, para que los
   asistentes la citen).

Detalles:

- Fichero `docs/quick-start/choose-a-forecaster.md`; título de la página "Which forecaster
  should I use?" (es la búsqueda literal) y entrada de menú "Choose a forecaster". La URL no
  lleva el nombre de la pestaña en el texto, pero sí en la ruta: si más adelante se mueve,
  hace falta una redirección (hoy no está el plugin `mkdocs-redirects`).
- Segunda entrada y no primera: la primera es la página en la que abre la pestaña, y
  cambiarla es cambiar el destino real de "Quick start". Subirla a primera, o cambiar "Get
  started", se decide con los datos de la fase 3.
- El gancho es "Not sure which one to use?", el que ya usa la portada. "Not sure where to
  start?" ya se usa en la página de consultoría (`docs/more/consulting.md:9`) con otro
  significado.
- El respaldo ("How we chose these rules") va plegado en la propia página, con la tabla
  corta, y enlaza al README de `benchmarks/forecaster_finder/` para el detalle.
- Entrada en `tools/docs/hooks/page_meta.yml`.

Lo que no he podido comprobar: de dónde entra hoy la gente. La recomendación supone que
quien empieza llega por la portada o por *Quick start*. Diez minutos en GA4 lo confirman o
lo cambian: el informe de páginas de entrada (qué parte de las sesiones empieza en la
portada, en *Quick start* y en una guía) y cuántos clics recibe hoy el enlace de
`home.html:189`. Si la mayoría entra por una guía desde un buscador y no pasa por *Quick
start*, la pestaña importa aún menos y los enlaces 5 y 6 suben de prioridad; la elección
entre A y B cambia poco el resultado en cualquiera de los dos casos.

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

Referencia para comparar después: las visitas de *Comparing forecasting models*, de la
introducción y de *Quick start* en las 4 semanas anteriores a publicar. **No hay que
esperar para tenerla** (corrige la versión anterior de este punto): GA4 ya guarda las
visitas de cada página y se pueden consultar el día de la publicación. Lo único que no se
recupera hacia atrás son los eventos que hoy no existen (copias de código, clics en los
botones del selector), y esos solo tienen sentido con la página publicada. Al publicar,
exportar ese informe y guardarlo con la nota de la fase 3.

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

### 6.9 Cómo se presentan LightGBM y Chronos-2 (propuesta del 2026-11-28, decisión D9)

El problema: skforecast se ha presentado siempre como una librería de machine learning, y
el estudio dice que, sin trabajar el modelo, Chronos-2 tiene menos error que LightGBM en 27
de 33 pares de dataset y tipo de datos, y que un ajuste automático ligero lo deja en 26 de
33 (4.7).

Lo que la evidencia permite y lo que no:

| Afirmación | ¿Se puede sostener? |
|:--|:--|
| "Empieza por LightGBM, es el mejor punto de partida" (la skill hoy) | No. Un asistente que lo siga da un primer resultado peor en unos 4 de cada 5 datasets, y peor que el naive en 7 de 16 con una serie y sin variables |
| "Chronos-2 es mejor que el machine learning" | No. El estudio mide un LightGBM sin trabajar o con 20 pruebas de búsqueda; no mide variables propias, conocimiento del problema ni más historia |
| "Sin ajustar nada, Chronos-2 suele tener menos error" | Sí (27 de 33; nunca peor que el naive por dataset) |
| "Intenta superarlo con LightGBM" (titular acordado) | A medias: con un ajuste ligero lo supera en 7 de 33 y queda a un 5 % o mejor en 10 de 33. Como reto, el usuario lo pierde dos de cada tres veces |
| "LightGBM es el modelo que entrenas con tus datos: admite tus variables, se explica, se ajusta, no depende de pesos externos ni de PyTorch y es barato de entrenar y de ejecutar en CPU" | Sí, por capacidad, no por el estudio |

**Recomendación: dos candidatos en primer lugar, con papeles distintos, y la identidad de
la librería en el flujo de trabajo, no en una familia de modelos.**

1. **Identidad.** skforecast es la librería de forecasting con machine learning: los
   forecasters, el backtesting, la búsqueda, los intervalos, la explicabilidad y el paso a
   producción. Los modelos fundacionales son una familia más que se usa *dentro* de ese
   flujo, con el mismo backtesting; no compiten con la librería. Es lo que ya dice la
   portada ("One API for every kind of model", con machine learning como primera tarjeta),
   así que la portada no cambia.
2. **Los dos van primero.** En cualquier problema nuevo se comparan los dos contra un
   baseline. No es incorrecto poner el machine learning en uno de los primeros lugares; lo
   incorrecto sería decir que es el más preciso de entrada.
3. **Cada uno con su argumento.**
   - Chronos-2 (`ForecasterFoundation`): "No training. Usually the most accurate before any
     tuning: the error to beat."
   - LightGBM (`ForecasterRecursive`, `ForecasterRecursiveMultiSeries`): "The model you
     train on your data. Add your own variables and knowledge of the problem, explain its
     forecasts, tune it. Out of the box it is usually behind Chronos-2."
   - La elección final es del usuario y depende de más cosas que el error: explicabilidad,
     coste a escala, dependencias, control. El selector no decide eso (alcance); lo nombra.
4. **Titular del selector.** Cambiar "Start with Chronos-2, then try to beat it with
   LightGBM" por uno que no sea una carrera: **"Compare two forecasters: LightGBM and
   Chronos-2"** (más "and keep ARIMA as a reference" en mensual y trimestral). La frase de
   debajo, comprobada contra el estudio, sigue diciendo cuál suele ir por delante.
5. **Orden del script: baseline, LightGBM, Chronos-2, y cada resultado impreso en cuanto
   se obtiene.** Motivo medido, no de imagen: sin acceso a Hugging Face el script falla en
   Chronos-2 a los 29 segundos, y como hoy es el primer bloque y la tabla se imprime al
   final, el usuario se queda sin nada (7.4, punto 2). Con LightGBM antes, siempre ve un
   resultado.
6. **Skill y `llms-base.txt` (fase 0).** Es donde más importa, porque los asistentes lo
   siguen al pie de la letra. Sustituir "machine learning-first" y los dos "START HERE" por
   la regla de los dos candidatos, por ejemplo: "For a new problem, compare two forecasters
   against a seasonal baseline: a gradient boosting forecaster (`ForecasterRecursive` or
   `ForecasterRecursiveMultiSeries` with LightGBM), which you train and shape with your own
   variables, and a foundation model (`ForecasterFoundation` with Chronos-2), which needs
   no training and is usually the more accurate of the two before any tuning." El consejo 6
   de `llms-base.txt` ("Start simple: Begin with ForecasterRecursive") pasa a "Start with a
   comparison: a baseline, LightGBM and Chronos-2".
7. **Notebook *Quick start*.** Sigue enseñando `ForecasterRecursive` con LightGBM: enseña
   la API, y ahí el machine learning primero es lo natural. Solo gana la nota que enlaza al
   selector (6.5).

Alternativas descartadas: mantener "ML first" como recomendación de modelo (contradice los
datos y deja en mal lugar a la documentación cuando el usuario prueba Chronos-2); pasar a
"Foundation first" sin más (convierte a skforecast en un envoltorio de un modelo ajeno y
esconde lo que la librería hace mejor).

La maqueta conserva el titular acordado (D2 y D6) hasta que se decida D9.

## 7. Plan de desarrollo

### 7.1 Decisiones cerradas

1. Alineación: Chronos-2 small y LightGBM más un baseline estacional; clasificador más regla
   estacional para categorías. Solo se nombra Chronos-2 (único medido y Apache-2.0).
2. **D1**: el bloque interactivo es un selector de una pantalla.
3. **D2 y D6**: titular "Start with Chronos-2, then try to beat it with LightGBM" en los
   tres tipos de datos.
4. **D3**: sin pregunta de prioridad; notas fijas plegadas.
5. **D4**: estudio ampliado a 26 datasets (4.6).
6. **D5**: el arreglo de `Ets` y sus notas de versión van en esta misma rama; el enlace a la
   PR se añade a las notas cuando se abra.
7. Cuatro filas. Sin filas de historia ni de horizonte. "Solo hasta hoy" se funde con "No".
   La fila de frecuencia se titula "Your data has one value per...".
8. Referencia estadística: ARIMA (el modelo de la portada), solo en mensual y trimestral de
   una serie y sin exógenas.
9. El resultado no enseña cifras: una frase por caso, comprobada contra el estudio, y las
   cifras en "How we chose these rules".
   Debajo del script, plegados: skforecast-ai (unas líneas que continúan el script) y los
   asistentes de IA (servidor MCP de skforecast-ai si el asistente lo admite; si no, el
   script como prompt).
16. **D7**: nada elegido al llegar; el recuadro de respuesta se rellena con cada clic, con
    el forecaster como titular de cada bloque, y el script aparece al completar.
10. Estudio de referencia: el de protocolo justo. Cifras por dataset, nunca "frente a la
    mejor variante de ML".
11. Requisitos del fragmento de 6.3.
12. Página estática primero; el bloque interactivo se monta encima y la página funciona sin
    JavaScript.
13. El estudio vive en `benchmarks/forecaster_finder/`, con requisitos fijados, desde la
    fase 0; esta carpeta de `dev/` se borra al cerrar la última fila.
14. Analítica: GA4 con el consentimiento que ya existe. Sin migración.
15. Fase probabilística: una fila opcional (sección 8).

### 7.2 Decisiones que necesitan respuesta

Ninguna. D1 a D8 cerradas el 2026-11-28; **D9 y D10 aceptadas el mismo día**:

- **D9**: titular "Compare two forecasters: LightGBM and Chronos-2", cada uno con su papel
  ("You train it", "No training"), y el script en el orden baseline, LightGBM, ARIMA y
  Chronos-2 al final, con cada resultado impreso al obtenerlo (6.9).
- **D10**: página propia en la pestaña *Quick start*, segunda entrada (6.5).

La barra de anuncios se descartó como gancho (7.5).

### 7.3 Fases

**Fase 0. Estudio de referencia y base** (`DOC-choosing-forecaster-evidence`, P2)

- Entregables:
  1. `benchmarks/forecaster_finder/`: un solo script con el protocolo justo (sale de
     `review.py`, `cases_d4.py` y `rules.py`), requisitos fijados, los CSV y un README. La
     ampliación de D4 ya está medida.
  2. Correcciones 1 a 15 de la sección 5 en `skills/choosing-a-forecaster/SKILL.md`,
     `skills/statistical-models/SKILL.md`, `skills/baseline-forecasting/SKILL.md` (baseline
     por serie y de clasificación), el consejo 6 de `tools/ai/llms-base.txt` y la frase de
     *Comparing forecasting models*.
  3. Incidencia en skforecast-ai con H1, H4 y H12 de `REVISION.md`.
- Terminado cuando: las cifras de la skill coinciden con los CSV y son por dataset; `python
  tools/ai/generate_ai_context_files.py --check` y `verify` en verde; nota en
  `docs/releases/releases.md`.
- Esfuerzo: M (3 a 4 días).
- Los fallos de `Ets` ya están arreglados en esta rama.

**Fase 1. Fuente única y página estática** (`FEAT-forecaster-finder`, P3, parte 1)

- Entregables: fichero de reglas y módulo de plantillas (6.6); página
  `choose-a-forecaster` (ubicación según D10, 6.5) con la tabla de reglas, el respaldo generado, un fragmento por
  combinación representativa (una serie y varias, con y sin `exog`, categoría) con la tabla
  de constantes por frecuencia, y "How we chose these rules"; sección de la skill generada
  entre marcadores; enlaces de 6.5; workflow de ejecución de fragmentos; widget de feedback.
- Terminado cuando: página, skill y `llms-full.txt` salen de la misma fuente y `--check` lo
  comprueba; todos los fragmentos pasan con avisos como errores y con la historia mínima de
  6.3; la página se lee completa sin JavaScript; `verify` en verde y nota de versión.
- Esfuerzo: M (4 a 5 días).
- Requiere: fase 0. Parte hecha en el prototipo: reglas, plantillas, cifras generadas y las
  dos comprobaciones de los fragmentos.

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

- Entregables: informe de visitas de las 4 semanas anteriores, exportado de GA4 el día de
  la publicación (6.7), y una nota a las 8 semanas con las cuatro
  señales y la decisión: mantener, simplificar a la página estática o ampliar.
- Terminado cuando: la decisión está escrita en el roadmap.
- Esfuerzo: S.

**Fase 4. Fila de incertidumbre** (`FEAT-finder-probabilistic`, P4)

- Sección 8. Requiere la fase 3 con decisión de mantener.
- Esfuerzo: M (estudio, 2 a 3 días; fila y fragmentos, 2 días).

Orden y dependencias: fase 0 → fase 1 → fase 2 → fase 3 → fase 4. El arreglo de `Ets` viaja
en la PR de esta rama (D5). La fase 0 tiene valor aunque no se haga nada más: hoy la skill dice a los asistentes
que empiecen por `ForecasterRecursive` con LightGBM y reserva los fundacionales para "sin
datos".

### 7.5 Estado de la implementación (2026-11-28)

Hecho:

| Fase | Qué | Dónde |
|:--|:--|:--|
| 0 | Estudio en `benchmarks/forecaster_finder/` con README y versiones fijadas | `benchmarks/forecaster_finder/` |
| 0 | Correcciones 1 a 15 de la sección 5 y posicionamiento de 6.9 | `skills/choosing-a-forecaster`, `statistical-models`, `baseline-forecasting`, `tools/ai/llms-base.txt` y la frase de *Comparing forecasting models* |
| 1 | Fuente única, generador con `--check`, página que funciona sin JavaScript, bloque de reglas de la skill generado | `tools/docs/forecaster_finder/`, la página y la skill |
| 1 | CI: ficheros generados al día, los 36 scripts ejecutados con avisos como errores y con su historia mínima; solo en las PR que tocan el finder (sin ejecución semanal, decisión del 2026-11-28) | `.github/workflows/forecaster-finder.yml` |
| 2 | Selector: 4 filas, recuadro de respuesta, script progresivo, copiar, Colab, skforecast-ai, asistentes, teclado, pantallas estrechas, tema oscuro, eventos con consentimiento | `docs/javascripts/finder.js`, `finder.css` |
| 2 | Las respuestas viajan en la URL (`#answers=number-one-daily-no`): se comparten y sobreviven a ir a una guía y volver | `finder.js` |

Ganchos puestos (de más a menos tráfico esperado):

1. Portada, bajo los botones del hero: "Not sure which forecaster to use? Find out in four
   questions".
2. Portada, bajo las familias de modelos: bloque con la primera pregunta. Cada respuesta
   abre el selector con esa pregunta ya contestada.
3. *Intro to Forecasting*: nota al principio de la página y frase bajo la tabla de
   forecasters.
4. Notebook *Quick start*: frase en la primera celda. Además, su enlace "Next" del pie
   lleva al selector.
5. README del repositorio (también es la página de PyPI): frase bajo la tabla de
   forecasters y fila en la tabla de documentación.
6. *User Guides*: tarjeta *Choosing a model* y frase al principio de *Comparing forecasting
   models*. FAQ: tarjeta *Forecasting Tips*.
7. *AI-assisted forecasting* y la skill `choosing-a-forecaster` (los asistentes citan la URL).
8. Páginas de ejemplos (inglés, español y chino): una línea bajo la introducción, en cada
   idioma, que avisa de que la página del selector está en inglés.
9. Notas de versión: destacado "Docs".

Decidido el 2026-11-28: **la barra de anuncios no se usa** para el selector (sigue anunciando
Studio y skforecast-ai).

Propuestos, sin hacer:

- **Fuera del repositorio**: los artículos de cienciadedatos.net que comparan modelos, la
  descripción del repositorio en GitHub y una publicación en LinkedIn al lanzar.
- Tarjeta en "Start here" de la portada: no la he puesto porque son 4 tarjetas en una
  rejilla de 4 y el bloque de las familias ya está en la misma página.

Pendiente antes de fusionar:

- Abrir la PR (no la abro sin preguntar) y añadir su enlace a las notas de versión.
- Probar a mano el botón "Run it in Google Colab": el enlace apunta a `main`, así que solo
  funciona cuando esta carpeta esté ahí.
- Que alguien más revise el arreglo de `Ets` como cambio de la librería.
- El sitio no se construye en esta sesión con la configuración tal cual (el plugin
  `privacy` descarga avatares de GitHub y la red lo bloquea): la página se ha probado con
  ese plugin desactivado. Construirlo en local antes de fusionar.

No hecho, a propósito: widget de feedback de Material en todo el sitio (el selector lleva
su propio "Was this useful?") y parámetros UTM en los enlaces salientes (ningún otro enlace
de la documentación los usa).

### 7.6 Revisión como usuario y lo que se cambió (2026-11-28)

Otra sesión usó la página como cinco personas distintas, con sus propios CSV, sin leer el
código (`user_review/INFORME.md`, con capturas). Veredicto: no enseñarla aún, porque el
botón principal copiaba un script sin datos. Decidido con el usuario: el script lleva sus
datos, el error pasa de MASE a MAE y se añade la predicción al final, plegada.

| Hallazgo | Qué se hizo |
|:--|:--|
| B1. "Copy script" copiaba un script sin datos (`NameError`) | El script incluye su bloque de datos: lo que se ve, lo que se copia y lo que se prueba es el mismo texto. Comprobado copiando desde el navegador y ejecutando lo copiado con skforecast 0.27.0 de PyPI |
| B2, B3. Errores engañosos al cargar un CSV | El bloque de datos explica `read_csv` (`index_col`, `parse_dates`, `dayfirst`), qué significa la frecuencia (`"MS"` frente a `"ME"`, `"W"` frente a `"W-MON"`) y el formato largo con `pivot`; una comprobación corta con un mensaje claro si la serie queda sin fechas, sin frecuencia o vacía. Probados los tres fallos del informe |
| C1. No predice y la MASE no se entiende | Error en MAE, en las unidades del usuario, y sección plegada "Forecast with the model you keep" con las líneas que entrenan cada modelo y predicen (ejecutadas en cada generación) |
| C2. "Something else" | Comentario con ejemplos (cada 15 minutos: `season = 96`) y frase de datos mínimos propia |
| C3. Varias series | Formato largo en el bloque de datos y nota de que `exog` puede ser un dict por serie |
| C4, C7. Clasificación | Notas propias (sin Chronos-2 ni PyTorch), sugerencia de predecir el número cuando las categorías salen de una lectura, y enlace a la guía de clasificación |
| C5. "Four questions" y contador | El contador empieza en "0 of 4" |
| C6. El `¶` borraba las respuestas | La URL solo se reescribe cuando el usuario contesta, y las respuestas se recuerdan mientras se sigue en la página; "Example script" ya no es un encabezado oculto |
| C8. 26 frente a 33 | Explicado en el texto, con el aviso de que Chronos-2 puede haber visto datos públicos |
| C10. 5,8 GB | El consejo de PyTorch para CPU va en la cabecera del script |
| C11. Código cortado en móvil | Las líneas largas se desplazan dentro de la ventana; comprobado a 360 y 400 px |
| C12, C13 | Aviso de que el fichero y las columnas del mensaje son de ejemplo; `to_string()` en la tabla de skforecast-ai |
| M1, M2, M3, M7 | "Start over" vuelve a la primera pregunta con el foco; "Copied" en el botón pulsado; "No" enlaza a abrir una incidencia; el script se ve entero, sin scroll interno, con la línea de `pip` antes de los imports |
| D4, D5, D7 | Textos corregidos; aviso sin JavaScript a ancho completo; el número de series se conserva al cambiar de tipo de objetivo |

Sin tocar: C9 (la búsqueda del sitio no encuentra la página; parece general del sitio, sin
comprobar), M4 (barras de progreso y aviso de Hugging Face en la salida), M5 (con menos datos
de los indicados el script no avisa), M6 (al contestar la cuarta pregunta en escritorio el
titular queda fuera de la vista) y los detalles D1 a D3 y D8 a D11.

### 7.7 Segunda revisión como usuario (2026-11-28)

Informe en `user_review/INFORME_2.md`: los 36 scripts copiados con el botón se ejecutan; lo
que falla son los datos reales sucios y detalles de estado y de texto. **Criterio fijado por
el usuario: la página es una prueba rápida para atraer y encaminar; conectar el script a los
datos propios es cosa del usuario.** No se añade más ayuda de carga de datos ni más líneas
de predicción.

Hecho:

- **Búsqueda del sitio**: con los textos de esta rama, cualquier búsqueda con "to", "how" o
  "should" no devolvía nada. Causa: un fallo de lunr.js (compara nodos de su grafo de
  términos con un texto sin separadores), que depende del vocabulario del sitio. Lo corrige
  el hook `tools/docs/hooks/search_fix.py` en el sitio construido. La página lleva
  `search: boost: 2`.
- **Recuento**: las cuatro preguntas se ven desde el principio (la del número de series
  desaparece con una categoría) y el contador y "N questions left" cuentan lo mismo.
- **URL**: lleva siempre las respuestas, también tras el aviso de cookies y al volver a la
  página en la misma pestaña; tolera añadidos (`&utm_source=x`, un punto final), y un enlace
  ilegible muestra el selector vacío, nunca las respuestas de otra visita.
- **Categoría**: accuracy en lugar de balanced accuracy (se entiende sola y elimina el aviso
  de scikit-learn). `build.py` ya no admite ningún aviso.
- **Coherencia**: "How we chose these rules" explica que la guía *Comparing forecasting
  models* mira tres datasets y que uno solo puede ir contra el recuento.
- **Textos**: "the last 3 forecasts of 14 days each (42 days)", cabeceras de la tabla,
  mensaje de la comprobación, "Something else", mensaje para el asistente.
- **Móvil**: al contestar la última pregunta, la respuesta se pone a la vista; en escritorio
  la página ya no salta.

Sin tocar, a propósito: huecos y series cortas con varias series, CSV con `;` y coma
decimal, predecir con el baseline o la regla, variables distintas por serie, ruido de la
salida, y lo que es del tema (aviso de cookies, botón "Back to top"). Nada de la librería
merece fila en el roadmap: lo que el informe daba por silencioso (Chronos-2 con variables
futuras desalineadas) sí lanza un `MissingValuesWarning`.

### 7.4 Comprobaciones antes de publicar (estado del 2026-11-28)

| # | Qué | Estado | Resultado |
|--:|:--|:--|:--|
| 1 | LightGBM con algo de ajuste | **Medido** (4.7) | Supera a Chronos-2 en 7 de 33 y queda a un 5 % o mejor en 10 de 33; el ajuste lo empeora en 11. Lleva a D9 |
| 2 | Primera ejecución real | **Medido** (`check_clean_install.sh`) | Funciona; dos hallazgos, abajo |
| 3 | Colab y MCP | **Comprobado, con un límite** | Abajo |
| 4 | Pantallas estrechas | **Hecho en la maqueta** | Por debajo de 980 px, las respuestas dadas se pliegan en etiquetas y solo queda abierta la pregunta siguiente, con la respuesta a la vista. Probado a 400 px |
| 5 | El resto de la documentación dice otra cosa | Pendiente (fase 0) | Texto propuesto en 6.9; depende de D9 |
| 6 | El arreglo de `Ets` | Pendiente de revisión | Corrige un fallo de la 0.27.0 publicada: que lo revise alguien más como cambio de la librería y decidir si merece versión de parche. Los notebooks de la documentación con AutoETS no se han vuelto a ejecutar |
| 7 | Clasificación | Sin cambios | Un solo dataset: la rama se sostiene por capacidad, no por evidencia |
| 8 | Referencia de visitas | **Retirado como bloqueo** | GA4 ya guarda las visitas por página; se exporta el día de publicar (6.7) |

**Punto 2, instalación limpia** (Linux, `pip`, Python 3.12, sin nada en caché):

- `pip install "skforecast[stats]" lightgbm chronos-forecasting`: 150 segundos y 5,8 GB,
  casi todo PyTorch con sus 15 paquetes de CUDA, que es lo que instala `pip` por defecto en
  Linux. Instala skforecast 0.27.0 y pandas 2.3.3 (el estudio usa 0.28.0 y pandas 3.1.1).
- Los 36 scripts se ejecutan en ese entorno. Primera ejecución del script mensual (tres
  modelos): 20 segundos, con la descarga de los pesos de Chronos-2 small (107 MB); la
  segunda, 7 segundos.
- **Hallazgo 1 (arreglado)**: el script con ARIMA fallaba con `ImportError` porque
  `statsmodels` es opcional. La línea de instalación de esos scripts y la del notebook
  ahora piden `"skforecast[stats]"`. Solo se ve en un entorno creado con la línea de
  instalación del propio script: **la CI de los fragmentos tiene que instalar así**, no con
  los extras de desarrollo.
- **Hallazgo 2 (nota añadida, orden por decidir en D9)**: sin acceso a Hugging Face (red de
  empresa), el script falla con `OSError` a los 29 segundos en el bloque de Chronos-2, que
  es el primero, y no imprime nada. La nota "First run" lo avisa; la solución de fondo es el
  orden baseline, LightGBM, Chronos-2 con cada resultado impreso al obtenerlo (6.9).
- Nota añadida también para quien no tiene GPU: instalar antes PyTorch solo para CPU
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). De los 5,8 GB,
  3,0 son los paquetes de CUDA y 1,2 PyTorch (0,8 en su versión de CPU); el tamaño de la
  instalación solo para CPU no está medido.

**Punto 3, Colab y MCP:**

- Notebook `first_comparison.ipynb` (celda de instalación y celda "Paste your script
  here"): ejecutado de principio a fin con `nbclient` en un kernel nuevo del entorno limpio,
  con dos scripts distintos pegados. La URL pública de Colab abre el notebook desde esta
  rama (HTTP 200, con las dos celdas). **No comprobado**: la ejecución dentro de Colab, que
  pide iniciar sesión en Google. Es una prueba manual de dos minutos: abrir el enlace de la
  maqueta, ejecutar la primera celda, pegar y ejecutar. Al publicar, el enlace tiene que
  apuntar a la rama de la versión, no a `docs/forecaster-finder`.
- Servidor MCP de skforecast-ai 0.5.0 (`check_mcp.py`): recorrido completo con un cliente
  MCP como el de un agente, sobre un CSV diario: `profile`, `plan`, `create_cv`, `compare`
  y `get_code`. Devuelve la comparación (gana `chronos-2-small`) y el script del mejor.
  **No comprobado**: un asistente real (Claude Code, Cursor) eligiendo esas herramientas a
  partir del mensaje en lenguaje natural que redacta el selector.
- Las líneas de skforecast-ai del desplegable: los 24 fragmentos que las llevan se ejecutan
  (`build.py --ai`, `skforecast_ai_check.json`).

Pequeño, sin decidir: en el script que se copia con datos de ejemplo, la línea `# pip
install ...` queda después del bloque de datos y de los imports; debería ser la primera
línea de lo que se copia.

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
