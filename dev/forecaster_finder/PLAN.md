# Selector de forecaster ("Not sure where to start?"): plan de desarrollo

## 1. Estado

- Versión 2, del 2026-11-27, con la segunda y la tercera pasadas del mismo día (arreglo de
  `Ets`, prototipo nuevo, estudio ampliado a 26 datasets y decisiones D1 a D5 cerradas). Sustituye a la versión 1
  (commit `407746d`) después de la revisión independiente de `REVISION.md`. Base:
  skforecast 0.28.0.
- Del selector no hay nada implementado en `docs/` ni `skills/`. En esta rama sí están
  arreglados los dos fallos de `Ets` que encontró la revisión (commit `5f020a0`, con tests,
  notas de versión y la skill `statistical-models`), y el prototipo de la fuente única
  (`benchmark/finder_tree.py`) ya sigue el diseño de la sección 6.
- Las cifras de este documento salen de `benchmark/review.py` (protocolo justo, sección 4).
  Las de la versión 1 (ML congelado) no se citan en ninguna página: ver `REVISION.md` H1.
- Filas del roadmap: `DOC-choosing-forecaster-evidence` (fase 0), `FEAT-forecaster-finder`
  (fases 1 a 3), `FEAT-finder-probabilistic` (fase 4); `BUG-ets-freeze-params` y
  `QW-ets-m-over-24-silent` (arregladas en esta rama, sin fusionar) y
  `QW-shared-cv-refit-warnings` (nueva).
- Maqueta navegable del selector con los datos del prototipo (36 combinaciones):
  `mock.html`, generada por `benchmark/mock.py` y publicada en
  https://claude.ai/artifact/VMvLQwo7Wf9xoK3VYAbgV6 (privada; se comparte desde su menú
  Share).
- Informe visual de la versión 1 (no actualizado, sus cifras son las del ML congelado):
  https://claude.ai/artifact/LdLnbpz9LXdJ4rqPaocysX.

Contenido de la carpeta:

| Ruta | Qué es |
|:--|:--|
| `PLAN.md` | Este documento |
| `REVISION.md` | Revisión independiente: hallazgos por gravedad, con evidencia y propuesta |
| `benchmark/benchmark.py` | Estudio original (ML congelado). Define los casos; se conserva porque reproduce las cifras de la versión 1 |
| `benchmark/review.py` | **Estudio de referencia**: mismos casos y bloques, ML reentrenado, `LightGBM+`, el LightGBM del fragmento, selección en validación, Chronos-2 small, base y synth; `--extra` y `--horizon` para los controles, `--stats` para volver a medir solo los estadísticos y `--snippet` para medir el LightGBM tal como lo escribe el fragmento |
| `benchmark/recount.py` | Recuentos de la revisión (15 casos originales): efecto del protocolo, por ejecución y por dataset |
| `benchmark/cases_d4.py` | Los 20 casos añadidos para llegar a 6 o más datasets por tipo de datos (`review.py --d4`) |
| `benchmark/rules.py` | **Respaldo de cada regla** con el estudio ampliado (26 datasets) y comparación de AutoETS con AutoARIMA; escribe `rules_*.csv`, que `build.py` usa para comprobar las frases de la página |
| `benchmark/finder_tree.py`, `build.py` | Prototipo de la fuente única con el diseño de la sección 6: 4 filas, reglas por tipo de datos, 36 fragmentos. `build.py` los ejecuta con los avisos como errores, calcula las cifras del respaldo y escribe `results/review/finder.json` (los datos de la página) |
| `benchmark/check_snippets.py` | Comprueba que cada tipo de fragmento se ejecuta con su historia mínima |
| `benchmark/mock.py`, `mock.html` | Maqueta de la página: las 4 filas y el resultado de cada combinación, con los datos de `finder.json` |
| `benchmark/repro_ets_freeze_params.py` | Reproducción de `BUG-ets-freeze-params` |
| `benchmark/summarize.py`, `evidence.py`, `report_data.py` | Resúmenes de la versión 1 (leen `results/*.csv`) |
| `benchmark/results/` | CSV del estudio original |
| `benchmark/results/review/` | CSV del estudio de referencia, controles (`extra_*`), recuentos (`recount_*`), fragmentos (`finder.json`, `snippet_*`), `environment.json` y, de la medición de los estadísticos tras el arreglo, `environment_stats.json` y `stats_previous.csv`; el error por serie y por bloque, en `detail/` |

Reproducir (desde `benchmark/`, con `uv pip install chronos-forecasting`):

```bash
python -I review.py              # unos 35 minutos en 4 núcleos sin GPU
python -I review.py --extra      # otros sitios de bdg2, otras series de M4, clasificación (15 min)
python -I review.py --horizon    # horizonte cuatro veces más corto (7 min)
python -I review.py --stats      # solo AutoETS y AutoARIMA, sustituidos en los CSV (3 min)
python -I review.py --snippet    # el LightGBM del fragmento, tal como se escribe (3 min)
python -I review.py --d4         # los 20 casos de cases_d4.py (25 min, con la carga de M5)
python -I review.py --arima      # AutoARIMA, también con exog, en diario, mensual y trimestral (2 min)
python -I recount.py             # cifras de la revisión (secciones 4.2 y 4.3)
python -I rules.py               # respaldo de las reglas con los 26 datasets (secciones 4.6 y 6.2)
python -I build.py               # 36 combinaciones, 36 fragmentos, medio minuto
python -I check_snippets.py      # historia mínima de cada tipo de fragmento
python -I mock.py                # maqueta navegable (../mock.html)
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

Un selector de una pantalla, no un test por pasos: cuatro filas de opciones con una
selección por defecto, y debajo el resultado, que cambia con cada clic. Título del bloque:
"Build your first comparison"; "Not sure where to start?" queda como gancho en la portada.

| # | Row | Options (hint) | Shown if |
|--:|:--|:--|:--|
| 1 | What do you want to forecast? | **A number** (sales, energy demand, website visits) · **A category** (low, medium or high demand; failure or no failure) | Always |
| 2 | How many time series? *A time series is one column of values recorded over time, such as the daily sales of one store.* | **One** · **Several** (each store, each product) | A number |
| 3 | Your data has one value per... *skforecast calls this the frequency of the series.* | Hour · Day · Week · Month · Quarter · Something else (15 minutes, a year...) | Always |
| 4 | Will you know anything else in advance? *Things that affect the values and that you know for future dates. skforecast calls them exogenous variables.* | **Yes** (holidays, planned promotions, prices, weather forecasts) · **No** (also if you only know them up to today, like sensor readings) | Always |

Selección por defecto: un número, una serie, un valor por día, no.

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

Más simple que en la versión anterior: sin tarjetas y sin cifras.

1. Titular y una o dos frases (6.2).
2. Una línea con los modelos que compara el script.
3. Ventana de código con el estilo de la portada (`docs/stylesheets/home.css`, resaltador de
   `docs/javascripts/home.js` como módulo compartido), botón Copy y la nota "Runs as it is:
   tested on sample data every time the docs are built. Needs at least N values".
4. Acciones: "Copy prompt for your AI assistant" (respuestas, plan, enlace a `llms-full.txt`
   y "ask me for my data first") y "Compare models on your own data with skforecast-ai" (sin
   prometer la misma comparación mientras sus reglas difieran: `REVISION.md` H12).
5. "Good to know", plegado.
6. Enlace compartible con las respuestas en el `#`.
7. Debajo del bloque, la sección "How we chose these rules": qué se probó, la tabla por tipo
   de datos y los límites.
8. Sin JavaScript se ve la página estática completa (tabla de reglas y fragmentos).

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
2. **D1**: el bloque interactivo es un selector de una pantalla.
3. **D2**: titular "Start with Chronos-2, then try to beat it with LightGBM". Con el estudio
   ampliado vale para los tres tipos de datos (pendiente de confirmar: D6).
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

| ID | Decisión | Opciones | Recomendación |
|:--|:--|:--|:--|
| D6 | El mismo titular en los tres tipos de datos | (a) "Start with Chronos-2, then try to beat it with LightGBM" siempre, con una frase distinta por caso; (b) mantener "Race both" con variables conocidas a futuro, donde la ventaja es menor (5 de 7 datasets, 3 de ellos por menos de un 10 %) | (a). Es lo que dice el estudio ampliado y es más simple. Implementado así en el prototipo |

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
  `choose-a-forecaster` con la tabla de reglas, el respaldo generado, un fragmento por
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

- Entregables: referencia de visitas previa (6.7) y una nota a las 8 semanas con las cuatro
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
