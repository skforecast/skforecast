# Segunda revisión como usuario: "Which forecaster should I use?"

Rama `docs/forecaster-finder` (commit `29a10c3`), sitio construido en local sin el plugin `privacy`, servido con `python -m http.server`. Navegador: Chromium con Playwright (1280×800 salvo que se indique otra cosa). Capturas en `2/`.

## 1. Resumen en cinco líneas

1. Todavía no se la enseñaría a usuarios reales: hay un fallo que no es de la página pero que la esconde. En esta rama, la búsqueda del sitio no encuentra nada con frases normales ("which forecaster should I use", "how do I start", "which model should I use"), y en la rama base y en skforecast.org esas mismas consultas sí encuentran resultados.
2. El selector en sí funciona bien: las 36 combinaciones dan un texto coherente y los 36 scripts, copiados con el botón y con sus líneas de predicción, se ejecutan sin error en entornos nuevos creados con su propia línea `pip install`.
3. Con datos reales, los casos sencillos (Elena, Iván, Nora, Tom) llegan a una predicción. Los que tienen datos "sucios" (Rosa: productos que empiezan más tarde y meses que faltan; Dani: CSV de Excel en español) se atascan o reciben cifras erróneas sin que la página les diga qué hacer.
4. Falta una salida para el caso más frecuente con datos reales: que gane el baseline o la regla. Ocurre con los datos de Rosa, y con los propios datos de ejemplo en 8 de las 12 combinaciones de categoría (la de Nora incluida). La página dice "keep what beats the baseline", pero no da líneas para predecir con el baseline.
5. Hay detalles de estado (la URL pierde las respuestas tras las cookies o tras pulsar un `¶`), de texto ("Answer four questions" cuando se ven tres) y de móvil (código cortado, el resultado queda bajo el pliegue) que no bloquean, pero restan confianza.

## 2. Primera impresión

**Página** (`01_primera_impresion_pagina.png`, a los diez segundos y con el aviso de cookies delante): un título claro, "Which forecaster should I use?", y debajo algo que parece un formulario con botones grandes y un panel oscuro con dos huecos que dicen "Forecaster". Entiendo que es un asistente: contestas y te dice qué modelo usar. Lo primero que haría es pulsar "A number". El aviso de cookies tapa la mitad inferior y difumina el resto, así que no se puede hacer nada hasta pulsar "Accept" o "Manage settings": no hay botón de "Reject" a la vista. Lo que choca nada más leer: "Answer four questions", pero cuento tres preguntas numeradas (1, 2, 3), y el panel del script dice "Your script appears here when you answer the 3 questions left" junto a "0 of 4 answered" (`04_cuatro_preguntas_se_ven_tres.png`).

**Portada** (`02_primera_impresion_portada.png`): es una portada de producto ("Time series forecasting, from prototype to production") con "Get started", "User guides" y `pip install skforecast`. Debajo, en letra pequeña, aparece "Not sure which forecaster to use? **Find out in four questions →**". Se ve, pero es un enlace de texto bajo una fila de botones grandes; alguien con prisa pulsa "Get started". Más abajo (a unos 2000 px) hay un bloque "Not sure which one to use?" con "A number / A category" que lleva al selector con la primera respuesta marcada; funciona, y es la entrada más clara de todas.

## 3. Hallazgos, de más a menos grave

### B1. Bloquea. La búsqueda del sitio no encuentra nada si la consulta lleva palabras como "I", "a", "how", "to" o "should"

- **A quién:** a todas las personas que buscan con una frase. No depende de esta página: afecta a cualquier búsqueda del sitio construido desde esta rama.
- **Qué hice:** en la caja "Search" de la cabecera (y con `index.html?q=...`) escribí lo que escribiría cada persona. Repetí las mismas consultas en https://skforecast.org/latest y en la rama `origin/0.28.x` construida igual que esta (mismo `mkdocs.yml` con `privacy` desactivado).
- **Esperaba:** que "which forecaster should I use" encontrara esta página.
- **Pasó:**

  | Consulta | Esta rama | 0.28.x (local) | skforecast.org |
  |---|---|---|---|
  | which forecaster should I use | **No matching documents** | (la página no existe) | (no existe) |
  | which model should I use | **No matching documents** | 101 resultados | 99 resultados |
  | how do I start | **No matching documents** | 101 | 99 |
  | how to choose a model | **No matching documents** | | |
  | choose a forecaster | **No matching documents** | | |
  | forecaster to use | **No matching documents** | | |
  | I / a | **No matching documents** | 101 / 101 | 99 / 99 |
  | which forecaster | 93 (esta página no sale entre las 4 primeras) | | 89 |
  | forecaster should use | 101 (esta página es la primera) | | |

  Captura: `03_busqueda_sin_resultados.png`.
- **Por qué importa:** Elena, Rosa o Dani escriben la pregunta tal cual; la caja dice "No matching documents" y concluyen que no hay nada. El título de la página no se encuentra a sí mismo. En el sitio publicado y en la rama base sí funciona, así que lo ha roto algo de esta rama; no he buscado la causa.

### B2. Bloquea. Rosa no puede ejecutar el script con sus 12 productos: un mes que falta o un producto que empezó tarde hacen fallar el baseline, y el mensaje recomienda cambiar a LightGBM

- **A quién:** Rosa (varias series, mensual, con precios conocidos: `#answers=number-several-monthly-yes`). Le pasaría a cualquiera con varias series de longitudes distintas o con huecos.
- **Qué hice:** copié el script con "Copy script", y en el bloque de datos usé exactamente las tres líneas del comentario, cambiando los nombres:
  ```python
  data = pd.read_csv("rosa_ventas_productos.csv", parse_dates=["mes"])
  wide = data.pivot(index="mes", columns="producto", values="ventas").asfreq("MS")
  series = {name: wide[name] for name in wide.columns}
  ```
  Su CSV (de `store_sales`) tiene 12 productos de 2013 a 2017; P09 a P12 empiezan en 2014-07, 2015-03, 2016-01 y 2016-06, y a P03 y a P07 les falta un mes.
- **Pasó:** falla en el bloque "1. Baseline" (línea 57) con:
  > `ValueError: 1 backtesting predictions are NaN at dates with observed values (first: 2017-11-01 00:00:00), so the metrics cannot be calculated. The most likely cause is a missing value in the series or in `exog`, used as a predictor of these dates, with an estimator that does not accept NaN (for example, a linear model). Fill the missing values, or use an estimator that accepts them, for example HistGradientBoostingRegressor or LGBMRegressor.`

  El baseline no tiene estimador, así que el consejo de "use LGBMRegressor" la manda en una dirección equivocada. Rellené los huecos interiores (`wide.interpolate(limit_area="inside")`, algo que la página no explica) y volvió a fallar, ahora con "5 backtesting predictions are NaN... (first: 2017-01-01)": es P12, que empezó en junio de 2016 y no tiene "el mismo mes del año anterior" en el periodo de test. Solo funcionó cuando quité P12 a mano. La página dice "Needs at least 36 values (3 years)" sin aclarar si es por serie; P10 (34 meses) y P11 (24) sí pasaron.
- **Por qué importa:** es el caso típico de un catálogo de productos. Rosa no sabe interpolar con pandas ni deducir que el problema es el producto más joven; con lo que pone la página, se queda aquí. Le habría ayudado una línea en el comentario de datos sobre qué hacer con meses sin dato y con series que empiezan tarde.

### B3. Bloquea (resultado erróneo). Dani: el CSV de Excel en español no está cubierto, y con la fecha mal leída el script imprime cifras sin sentido antes de fallar

- **A quién:** Dani (una serie trimestral, `#answers=number-one-quarterly-no`), con `Fecha;Ingresos`, `01/04/1998;251400,21`.
- **Qué hice, paso a paso, siguiendo solo los comentarios del script:**
  1. `pd.read_csv("dani_ingresos_trimestrales.csv", index_col="Fecha", parse_dates=True, dayfirst=True)` → `ValueError: 'Fecha' is not in list`. Los comentarios no mencionan `sep=";"`.
  2. Añadí `sep=";"` (conocimiento propio) → el script pasa la carga y falla en el baseline (línea 51) con `ValueError: could not convert string to float: '329792,09'`. Los comentarios no mencionan `decimal=","`, y la comprobación `dated = ...` del script no detecta que los valores son texto.
  3. Con `sep=";", decimal=","` pero **sin** `dayfirst=True` (el olvido más probable): pandas lee `01/04/1998` como 4 de enero, `asfreq("QS")` deja 57 de 77 valores en NaN, la comprobación del script no salta (solo mira si son **todos** NaN), y el script imprime:
     ```
     Baseline:  15293.355
     LightGBM:  53829.555
     ```
     con 6 recuadros `MissingValuesWarning`, y luego falla en ARIMA con `ValueError: `y` has missing values.`. Los dos números impresos son basura.
  4. Con `sep=";", decimal=",", dayfirst=True` funciona: Baseline 15866.5, LightGBM 14395.0, ARIMA 8456.9, Chronos-2 10917.6, y la sección de predicción da los 4 trimestres de 2018.
- **Por qué importa:** cualquier CSV exportado desde un Excel en español, francés o alemán usa `;` y coma decimal. Dani llega por "Intro to Forecasting", tiene datos limpios y aun así necesita dos parámetros que la página no menciona. Si su script no tuviera ARIMA (si eligiera "Month" con varias series, por ejemplo), el error del paso 3 no aparecería nunca y se iría con números inventados. En el semanal de Elena, con fechas `dd/mm/yyyy` y sin `dayfirst`, sí salta la comprobación del script (`` `y` needs dates with a frequency, and values: see above.``): el fallo silencioso solo ocurre cuando todas las fechas son día 1, como en datos mensuales y trimestrales.

### C1. Confunde. Cuando gana el baseline (o la regla) no hay forma de predecir con él, y eso pasa a menudo

- **A quién:** Rosa con sus datos (Baseline 85.894, LightGBM 96.997, Chronos-2 190.596); Nora con los datos de ejemplo (`Classifier: 0.903`, `Rule: 1.000`); y, **con los datos de ejemplo**, la regla gana al clasificador en 8 de las 12 combinaciones de categoría (por ejemplo `category-_-weekly-no`: 0.447 contra 0.795; `category-_-monthly-yes`: 0.472 contra 0.806). En las numéricas siempre hay algún modelo que bate al baseline, pero en tres scripts uno de los candidatos pierde contra él: LightGBM en `number-one-weekly-no` (1.364 contra 1.299) y en `number-one-hourly-no` (1.259 contra 1.241), y ARIMA en `number-one-monthly-yes` (1.892 contra 1.818).
- **Qué hice:** ejecuté cada script tal cual y abrí "Forecast with the model you keep".
- **Esperaba:** saber qué hacer si nada bate al baseline.
- **Pasó:** el texto del script dice "keep a model only if it beats the baseline" y la sección de predicción dice "Keep the lines of the model with the lowest error", pero solo trae líneas para LightGBM, ARIMA y Chronos-2 (o solo para el clasificador). En categoría, la respuesta dice además "In our test it did not clearly beat the simple rule, so keep the rule as your baseline", que se puede leer como "usa la regla para predecir" o como "sigue comparando con la regla". La regla se calcula con `y.shift(season)` y no hay ninguna línea para usarla en el futuro.
- **Por qué importa:** la primera vez que alguien ejecuta el ejemplo de categoría ve que el modelo pierde, y la página no le dice qué hacer. Rosa, con sus datos, se queda sin predicción "buena" que copiar. Le habría ayudado una línea de predicción para el baseline y una frase sobre qué significa que gane.

### C2. Confunde. Las variables conocidas que cambian por serie (los precios de Rosa) solo se mencionan; construirlas y pasar su futuro requiere bastante pandas, y si se olvida, LightGBM predice igual

- **A quién:** Rosa (`number-several-monthly-yes`).
- **Qué hice:** el comentario dice "exog: a DataFrame with the dates as index and the variables you know in advance, or a dict with one DataFrame per series if they differ between them". No hay ejemplo. Escribí (por mi cuenta) `price = data.pivot(index="mes", columns="producto", values="precio").asfreq("MS")` y `exog = {name: price[[name]].rename(columns={name: "precio"}) for name in series}`. Para predecir, la sección trae `exog_future = pd.DataFrame({"promotion": 0.0}, index=dates)` con el comentario "(sample values here)".
- **Pasó:** si Rosa deja esa línea de ejemplo, LightGBM **predice igualmente** los 66 valores con el precio en NaN, avisando solo con un recuadro `MissingExogWarning: {'precio'} not present in exog. All values will be NaN.`; Chronos-2 sí falla, con `ValueError: `exog` contains columns with no historical values in the context for series {'P01': ['promotion'], ...}`. Cuando construí el futuro como un dict por producto, funcionaron los dos. Además, la salida se trunca (`[66 rows x 2 columns]` con `...` en medio), así que no ve la predicción de cada producto sin saber pandas.
- **Por qué importa:** la persona que elige "Yes" porque conoce los precios es justo la que tiene esta estructura, y el camino más corto (no tocar la línea de ejemplo) le da una predicción de LightGBM que ignora los precios sin que ella lo note.

### C3. Confunde. Las líneas de skforecast-ai fallan con el formato de exog que el propio script recomienda para varias series

- **A quién:** Rosa, o cualquiera con varias series y un dict de exog.
- **Qué hice:** entorno nuevo con la línea del script y `pip install "skforecast-ai[foundation]"` (skforecast-ai 0.5.0); copié "Copy lines" de "Let skforecast-ai compare more models for you" y las pegué después del script ya adaptado.
- **Pasó:** `TypeError: cannot concatenate object of type '<class 'dict'>'; only Series and DataFrame objs are valid` en `data = pd.concat([pd.DataFrame(series), exog], axis=1)`. Con los datos de ejemplo (exog como DataFrame) funciona, pero entonces el ranking de varias series solo tiene **dos** modelos (ForecasterFoundation y ForecasterRecursiveMultiSeries) y no incluye el baseline, aunque la sección dice "tries more of them". Con una serie sí probó más modelos: Elena 4 (añade ForecasterDirect), Dani 5 (Ridge, ForecasterDirect...), Iván 4.
- **Por qué importa:** la sección promete "the same data and the same test periods"; para Rosa, ni lo uno ni lo otro.

### C4. Confunde. Una semana que falta llena la salida de avisos, y el comentario del script apunta a una causa que no es

- **A quién:** Elena (`number-one-weekly-no`), con una semana borrada del CSV (la del 26/12/2016, un cierre por vacaciones).
- **Pasó:** el script termina, pero imprime 9 recuadros `MissingValuesWarning` ("`last_window` has missing values. Most of machine learning models do not allow missing values..."). El comentario de datos dice "If the values turn into NaN after `asfreq`, the dates of the file do not match that frequency": Elena revisaría sus fechas, que están bien.
- **Por qué importa:** Elena sabe poco Python; nueve avisos que no entiende le quitan confianza en los números, y nada le dice que basta con rellenar esa semana.

### C5. Confunde. El recuento de preguntas no cuadra: "four questions", se ven tres, y el script dice "the last question" cuando faltan dos

- **A quién:** todas, a cualquier tamaño de pantalla.
- **Qué hice / pasó:**
  - Al llegar: "Answer four questions about your data", tres preguntas numeradas del 1 al 3, "0 of 4 answered" y "Your script appears here when you answer the 3 questions left" (`04_cuatro_preguntas_se_ven_tres.png`).
  - Al pulsar "A number" aparece "2 How many time series?" y las demás se renumeran (la de frecuencia pasa de 2 a 3).
  - Contestando en otro orden (Hour y Yes primero): "2 of 4 answered" y el script dice "Your script appears here when you answer **the last question**", con la pregunta 1 sin contestar y la de cuántas series todavía oculta (`05_ultima_pregunta_quedan_dos.png`). Además, al contestar "Yes" la página bajó hasta el script y dejó el resultado fuera de la vista.
  - Con "A category" son tres preguntas ("1 of 3 answered"), aunque el texto sigue diciendo "four".
- **Por qué importa:** es lo primero que se lee, y la primera promesa ya no se cumple a la vista. Quien contesta desordenado no sabe qué le falta.

### C6. Confunde. Contradicción con *Comparing forecasting models* sobre ARIMA y sobre quién gana con variables conocidas

- **A quién:** Dani (y cualquier usuario mensual o trimestral) que pasa de una página a otra; el selector enlaza a la guía como "Next step".
- **Qué leí:**
  - Selector, mensual o trimestral: "ARIMA is the classic reference for monthly and quarterly data: in our tests it often did better than LightGBM out of the box", y "In the 8 monthly and quarterly datasets with one series, ARIMA had a lower error than LightGBM in 6".
  - Guía, apartado "What if the series is short?" (mensual, h2o): ARIMA es el penúltimo (MASE 0.751), **peor que el seasonal naive** (0.704) y que LightGBM (0.642). En "Key takeaways": "Statistical (AutoARIMA, AutoETS) ... Behind LightGBM and Chronos-2 in the three datasets".
  - Selector, con "Yes": "Chronos-2 was ahead more often, also with variables known in advance". Guía: "Machine learning (LightGBM, Ridge): The most accurate model with exogenous variables", y en su tabla LightGBM 5.42 contra Chronos-2 5.86.
- **Por qué importa:** son conclusiones de estudios distintos y las dos pueden ser ciertas, pero ninguna página lo dice. El lector ve dos recomendaciones opuestas en el mismo sitio y no sabe a cuál hacer caso.

### C7. Confunde (no estoy seguro de que sea un fallo). Chronos-2 acepta sin quejarse una previsión de temperatura con fechas que no tocan

- **A quién:** Iván (`number-one-hourly-yes`).
- **Qué hice:** con sus datos (que terminan el 30/12 a las 12:00), le pasé a `predict` una previsión de temperatura desplazada 11 horas (de 00:00 a 23:00 del día siguiente, lo que él llamaría "mañana").
- **Pasó:** LightGBM falla con un mensaje claro: `ValueError: To make predictions `exog` must start one step ahead of `last_window`.`. Chronos-2 devuelve 24 predicciones fechadas de 13:00 a 12:00, sin aviso, y distintas de las que da con la previsión bien alineada (7835.9 contra 7828.7 en la primera hora), así que usa esos valores de alguna forma.
- **Por qué importa:** Chronos-2 es el modelo que le gana a Iván (214.9 contra 601.6 del baseline), así que es el que usará, y podría predecir con la temperatura de otras horas sin enterarse. No he mirado qué hace exactamente con las fechas.

### C8. Confunde. La URL deja de llevar las respuestas, y un enlace con algo añadido muestra las respuestas de otra visita

- **A quién:** cualquiera que comparte el enlace.
- **Qué hice / pasó:**
  - Llegué con un enlace compartido completo (`#answers=number-one-weekly-no`) y acepté (o rechacé, desde "Manage settings") las cookies: la página se recarga, las respuestas siguen en pantalla, pero la URL queda en `choose-a-forecaster.html#`. Si ahora copio la URL para pasársela a alguien, le llega el selector vacío. Igual con un enlace a medias (`#answers=number-_-_-_`).
  - Pulsé el `¶` de "How we chose these rules": la URL pasa a `#how-we-chose-these-rules` y las respuestas desaparecen de ella (en pantalla siguen).
  - Volví a la página desde la portada ("Find out in four questions") en la misma pestaña: me muestra las respuestas de antes, con la URL sin `#answers=`.
  - Abrí `#answers=number-one-weekly-no&utm_source=x` y `#answers=number-one-weekly-no.` (con el punto final que se cuela al copiar de una frase): en vez de "Week / No", la página mostró la combinación que yo había contestado antes en esa pestaña (`category-...-weekly-no`), sin ningún aviso, mientras la URL seguía diciendo `number-one-weekly-no`. En una pestaña nueva sin visitas previas, esos enlaces muestran el selector vacío.
  - Lo que sí funciona: recargar conserva las respuestas, ir a una guía y volver con "atrás" las conserva, y abrir el enlace completo en otro navegador muestra las cuatro respuestas.
- **Por qué importa:** Rosa comparte el enlace con su jefe para enseñarle la recomendación, y a él le llega otra cosa o nada.

### C9. Confunde. Avisos de scikit-learn sin explicación en los scripts de categoría

- **A quién:** Nora (`category-_-daily-yes`, con sus datos y con los de ejemplo).
- **Pasó:** cada ejecución imprime `UserWarning: y_pred contains classes not in y_true` (de `sklearn/metrics/_classification.py`) justo antes de `Rule:`. Nada en la página lo explica.
- **Por qué importa:** Nora no sabe si el resultado es válido.

### C10. Confunde. "Test: the last 3 periods" no dice cuánto es

- **A quién:** Elena, Dani, cualquiera que lea el script.
- **Pasó:** el comentario dice "Test: the last 3 periods" (o "2 periods" en mensual y trimestral), y el código usa `len(y) - 3 * steps`. Para Elena son 24 semanas, no 3. Para quien no programa, "period" es una semana.
- **Por qué importa:** Elena no sabe sobre cuántas semanas se calcula el error que lee.

### C11. Confunde. El mensaje para el asistente MCP lleva valores de ejemplo que no encajan con "Something else"

- **A quién:** Tom (`number-one-other-no`).
- **Pasó:** "Compare forecasting models for the next **14 values** of the column sales of data/my_data.csv...". Para lecturas cada 10 minutos, "14 values" no significa nada (el script de "Something else" sí explica que hay que editar `steps` y `season`). En varias series el mensaje dice "of each series of data/my_data.csv" y pierde "against a seasonal naive baseline", que sí está en el de una serie.
- **Por qué importa:** Tom prefiere el asistente; si pega el mensaje sin pensar, el asistente compara a 14 pasos (dos horas y veinte minutos).

### M1. Molesta. En el móvil el script se corta por la derecha, y la línea de instalación no se lee entera

- **A quién:** Nora en el móvil; 360 y 400 px, tema claro y oscuro.
- **Pasó:** el bloque de código mide 581 px de contenido en una caja de unos 300 px. La página no se desplaza en horizontal (bien), pero cada línea del script hay que deslizarla dentro de la caja. La primera línea, la que importa, se queda en `# pip install "skforecast[stats]" lightgbm chr` (`06_movil_400_codigo_cortado.png`). Copiar funciona (con el botón y tocando).
- **Por qué importa:** en el móvil Nora solo quiere entender qué tiene que instalar y qué cambiar; con el código cortado no puede.

### M2. Molesta. En el móvil, tras la última respuesta el resultado queda bajo el pliegue; en escritorio la página sí baja

- **A quién:** Nora (Pixel 7, 412 px).
- **Pasó:** al tocar la última respuesta la página no se mueve: del resultado solo asoma el encabezado "YOUR STARTING POINT 3 OF 3 ANSWERED / Start with a machine..." en el borde inferior (`08_pixel7_resultado_bajo_el_pliegue.png`). En 1280 px la página baja sola hasta el script (de `scrollY` 33 a 725). En 360 px el resultado sí se ve (`10_movil_360_completo_ok.png`). Los chips "A category ✎", "One value per day ✎" funcionan bien para cambiar una respuesta.

### M3. Molesta. El botón flotante "Back to top" tapa encabezados

- **A quién:** 400 px (y también en 1280 px, sobre el título, al volver hacia arriba).
- **Pasó:** "Back to top" queda encima de "Run the comparison on your data", del que solo se lee "Run the co… …r data" (`07_movil_400_back_to_top_tapa_titulo.png`). Es el comportamiento del tema, pero aquí cae justo sobre el encabezado del script.

### M4. Molesta. El aviso de cookies: sin "Reject", 50 pulsaciones de Tab hasta "Accept", y los enlaces a secciones pierden su destino

- **A quién:** cualquiera en su primera visita; especialmente con teclado.
- **Pasó:** el aviso bloquea toda la página (no se puede contestar nada hasta cerrarlo). Para rechazar hay que pulsar "Manage settings", desmarcar dos casillas y pulsar "Accept" (el botón sigue diciendo "Accept"). Con teclado, "Accept" es el elemento 50 en el orden de Tab: antes se recorren, invisibles detrás del velo, todos los enlaces de la cabecera y de la página. Al aceptar o rechazar, la página se recarga: las respuestas sobreviven (bien), pero un enlace a `#how-we-chose-these-rules` termina arriba del todo, sin el destino. Es del tema y afecta a todo el sitio.

### M5. Molesta. La salida tiene mucho ruido y dos formatos distintos

- **A quién:** Elena, Dani (perfiles poco técnicos).
- **Pasó:** las barras de progreso de tqdm se mezclan con los resultados (`0%| | 0/3 ...100%|██████████| 3/3`), la primera vez aparece `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN...`, y "Loading weights: 100% 92/92". Las predicciones de LightGBM salen como una serie (`2017-12-25 13979.42 ... Freq: W-MON, Name: pred`) y las de Chronos-2 como una tabla con una columna `level` que repite "ventas". Con varias series, `[66 rows x 2 columns]` con `...` en medio.
- **Por qué importa:** Elena tiene que buscar sus tres números entre el ruido, y no sabe qué es `level`.

### M6. Molesta. "Was this useful? No" lleva a abrir un issue en GitHub en blanco

- **Pasó:** "No" → "Thanks for telling us. If something failed or was not clear, **tell us what**." (`09_feedback_no.png`), que enlaza a `https://github.com/skforecast/skforecast/issues/new`, sin plantilla ni las respuestas que dio. "Yes" → "Thanks for telling us.". No se recuerda al recargar (vuelve "Was this useful? Yes · No").
- **Por qué importa:** Elena o Dani no tienen cuenta de GitHub; el que tiene un problema es justo quien no va a poder contarlo.

### M7. Molesta. "Start over" no se puede deshacer

- **Pasó:** tras "Start over" la URL queda sin respuestas y "atrás" no las recupera (sigue "0 of 4"). Un clic por error obliga a contestar de nuevo. No pide confirmación.

### M8. Molesta. La página tiene dos nombres

- **Pasó:** el título es "Which forecaster should I use?", pero la cabecera al desplazarse, el menú lateral del Quick start y el "Next" del Quick start dicen "Choose a forecaster". En la FAQ aparece dentro de "Forecasting Tips", entre "Forecasting with delayed historical data" y "Forecasting time series with trend: differentiation". Al buscar "choose a forecaster" no aparece (B1).

### D1. Detalle. Varios "Copied ✓" a la vez

Pulsando seguidos "Copy", "Copy lines" y "Copy message", los tres botones dicen "Copied ✓" durante unos segundos, pero en el portapapeles solo está el último (el mensaje MCP). Antes de terminar el selector no hay botón de copiar, así que no se puede copiar un script a medias (bien). El doble clic en una opción no deselecciona ni hace nada raro.

### D2. Detalle. Textos que se leen raro

- Categoría con "Something else": "the category of **the value** one season before".
- Justo después de "With an automatic search of 20 trials, LightGBM improved in 18 of the 33 cases", el párrafo siguiente empieza con "One configuration per model, without tuning": parece una contradicción hasta releer.
- En la tabla, "LightGBM had the lower error" no dice frente a quién (se entiende que frente a Chronos-2). "each dataset counts once in each row" no se entiende a la primera.
- El error de la comprobación del script dice "see above": se refiere a los comentarios del fichero, pero aparece en la terminal, donde "above" es el traceback.
- "Run it in Google Colab" copia el script y muestra "Script copied: paste it in the second cell of the notebook that just opened."; no he podido ver el notebook (es esperado), así que no sé si "second cell" encaja.
- Las dos opciones de la pregunta 1 tienen anchos distintos ("A number" más estrecha que "A category").

### D3. Detalle. La nota de la sección de predicción en categoría dice "the better score", y la del script "beats the rule"

En categoría se dice "Keep the lines of the model with the better score", pero solo hay líneas de un modelo (el clasificador). Ver C1.

## 4. El recorrido de cada persona

- **Elena (semanal, lunes, llega por la portada).** Encontró el enlace "Find out in four questions" y el bloque "A number / A category" de la portada. Contestó sin dudas (One, Week, No). El script funcionó a la primera con los datos de ejemplo. Con los suyos, primero falló `index_col="date"` (`'date' is not in list`), y luego `asfreq("W")` con fechas en lunes (`` `y` needs dates with a frequency, and values: see above.``); el comentario sí dice "W-MON if yours are dated on Mondays", así que lo arregló leyendo. Resultado: Baseline 778.8, LightGBM 905.5, Chronos-2 380.2; las líneas de predicción dieron las 8 semanas de 2017-12-25 a 2018-02-12 (unas 13 500 a 14 600 por semana, coherente con su historial). Con una semana que falta, nueve avisos (C4). En mi recorrido, unos 10 minutos hasta la predicción. Confiaría en Chronos-2; no entendería el ruido de la salida (M5).
- **Dani (trimestral, CSV de Excel en español, llega por "Intro to Forecasting").** El aviso "Not sure which forecaster to use?" está arriba en la introducción. Contestar, fácil. Tuvo que instalar otro entorno porque su script pide `skforecast[stats]` (lo dice la primera línea; el primer entorno de Elena falla con `'statsmodels' is an optional dependency...`). Con sus datos, se atascó dos veces por cosas que la página no dice (`sep`, `decimal`) y una tercera vez recibió cifras falsas (B3). Al final: ARIMA 8456.9 gana a Chronos-2 10917.6, y la predicción de 2018 funciona. Sin saber pandas no habría llegado.
- **Rosa (12 productos mensuales, formato largo, precios, llega por las guías).** En "User Guides" el selector está en la tarjeta "Choosing a model", bien visible. Contestó (Several, Month, Yes). Con sus datos no pasó del baseline (B2) hasta interpolar y quitar un producto, cosas que tuve que saber yo. Construir los precios por producto (C2) y su futuro también fue cosa mía. Resultado: el baseline gana a los dos modelos (C1), y skforecast-ai falla con su exog (C3). No confiaría ni sabría con qué predecir.
- **Iván (horaria con temperatura).** Llega directo. Script de ejemplo, bien. Con sus datos: Baseline 601.6, LightGBM 1060.7 (peor que el baseline), Chronos-2 214.9. Las líneas de predicción con su previsión de temperatura dieron las 24 horas siguientes (media 7700, frente a 7758 reales). Cambiar `exog_future` por su CSV fue fácil. Riesgo: Chronos-2 no comprueba las fechas de la previsión (C7). skforecast-ai (con `confirm_cost=True`) le dio el mismo ganador. Confiaría.
- **Nora (categoría diaria con festivos; móvil y luego portátil).** En el móvil, el aviso de cookies ocupa media pantalla; los chips para editar respuestas funcionan bien; el resultado queda bajo el pliegue (M2) y el código cortado (M1). En el portátil: entorno de solo `skforecast lightgbm` (9 s), script de ejemplo con `Rule: 1.000` > `Classifier: 0.903` (C1) y un `UserWarning` (C9). Con sus datos: Classifier 0.714, Rule 0.607, y las líneas de predicción con sus festivos futuros dieron 7 días de "bajo". Funcionó, pero no sabría si fiarse del aviso.
- **Tom (lecturas cada 10 minutos, Jupyter, asistente de IA).** "Something else" explica bien qué editar (`steps`, `season`, `lags`). Las fechas `01.01.2016 00:10:00` no se leen sin `dayfirst=True` (falla la comprobación del script, con un mensaje correcto). Con `"10min"`, `steps = season = 144`, `lags=144`, el script corre en 12 s (445 lecturas que faltan, solo avisos): Baseline 2.616, LightGBM 1.868, Chronos-2 1.052. Pegado en una celda de notebook funciona igual (8.5 s y 9.0 s). El mensaje MCP lleva "next 14 values" (C11); el prompt "Copy script as a prompt" es bueno: pide al asistente que pida una muestra de los datos, que no cambie los modelos y enlaza a `llms-full.txt` (que existe en skforecast.org). No menciona huecos ni fechas con puntos, y sus "settings marked as examples" no incluyen `window_sizes=7` ni `CalendarFeatures(["day_of_week", "month"])`, que para 10 minutos tampoco tienen sentido.

## 5. Tiempos medidos y entorno

Entorno: contenedor Linux x86_64 (kernel 6.18), Python 3.12.3, `python -m venv` + `pip` (nada de uv). Tres entornos nuevos, creados con la línea de cada script y el torch para CPU que sugiere el script:

| Entorno | Instalación | Tiempo |
|---|---|---|
| `pip install torch --index-url .../cpu` + `pip install skforecast lightgbm chronos-forecasting` | primero, sin caché | 21 s + 24 s |
| igual con `"skforecast[stats]"` | caché de pip ya caliente | 8 s + 21 s |
| `pip install skforecast lightgbm` (categoría) | | 9 s |
| `pip install "skforecast-ai[foundation]"` sobre el primero | | 4 s |

Versiones instaladas: **skforecast 0.27.0** (la de PyPI; esta página es de la 0.28 en desarrollo, y aun así los 36 scripts funcionan con la 0.27.0), lightgbm 4.7.0, chronos-forecasting 2.3.2, torch 2.15.0+cpu, transformers 5.24.1, pandas 2.3.3, numpy 2.5.5, scikit-learn 1.9.1, statsmodels 0.15.0, skforecast-ai 0.5.0.

| Caso | 1.ª ejecución | 2.ª ejecución |
|---|---|---|
| Elena, ejemplo (descarga Chronos-2) | 21.1 s | 7.1 s |
| Dani, ejemplo (con ARIMA; compila numba) | 36.8 s | 9.0 s |
| Rosa, ejemplo | 6.6 s | 6.9 s |
| Iván, ejemplo / con sus datos (26 269 horas) | 6.9 s / 7.5 s | 6.4 s |
| Nora, ejemplo | 5.2 s | 1.9 s |
| Tom, notebook (dos celdas, script + predicción) | 8.5 s | 9.0 s |
| Tom, con sus datos (52 704 lecturas) | 12.0 s | |
| Elena + skforecast-ai / Iván + skforecast-ai | 17.6 s / 30.7 s | |
| 36 combinaciones + líneas de predicción (pesos en caché) | 1.3 s a 9.2 s (categoría ~2 s, una serie 6 a 9 s) | |

## 6. Lo que funcionó bien y no tocaría

- Las 36 combinaciones: texto, tarjetas, baseline, frase de datos mínimos, script, líneas de predicción y mensaje MCP son coherentes entre sí (frecuencia, `steps`, `season`, "the same week the year before"...). Los 36 scripts copiados con el botón terminan con código 0 e imprimen sus predicciones.
- Copiar copia exactamente lo que se ve, imports plegados incluidos, y el botón confirma con "Copied ✓".
- Los comentarios del bloque de datos resuelven el caso típico (nombre de columna, `W-MON`, `QE`, `dayfirst`).
- "Something else" explica bien qué tocar.
- Los chips "✎" en el móvil para cambiar una respuesta, y que cambiar "A number" a "A category" y volver conserve el resto de respuestas.
- Recargar, ir a una guía y volver con "atrás", y abrir el enlace completo en otro navegador conservan las respuestas; las respuestas sobreviven al aviso de cookies.
- Teclado: el foco se ve siempre (borde naranja), las flechas mueven entre opciones, el orden de Tab tiene sentido y lo plegado no recibe el foco.
- Sin JavaScript: un aviso claro, las reglas y un script de ejemplo completo.
- Tema oscuro: legible, sin problemas de contraste en lo que capturé.
- Sin desplazamiento horizontal de la página en ningún ancho (360 a 1920).
- Todos los enlaces internos de la página y de las respuestas funcionan (200), y también ai.skforecast.org y la guía del servidor MCP.
- El prompt "Copy script as a prompt" está bien planteado para un asistente.

## 7. Lo que no pude probar

- El notebook de Colab (404 esperado): no sé qué instala su primera celda, ni si "paste it in the second cell" encaja con los cuatro tipos de script (con y sin `[stats]`, con y sin Chronos-2).
- El servidor MCP de skforecast-ai con un asistente real: solo leí el mensaje.
- Pegar "Copy script as a prompt" en un asistente real y ver qué hace con los datos de Rosa o de Dani: lo leí como lo leería un asistente.
- Windows y macOS (la nota de torch para CPU es solo para Linux), y una GPU.
- Un móvil real: usé la emulación de Pixel 7 de Playwright, con toques, no un teléfono.
- Lectores de pantalla: solo revisé el árbol de accesibilidad (las preguntas son `radiogroup` con `radio`, y el resultado tiene un `status`).
- Datos anuales para Tom: elegí 10 minutos.
- La causa de B1: no he mirado la configuración de la búsqueda, solo he comprobado que en la rama base construida igual sí funciona.
- Si el `0.28` que se publique cambia algo de lo que he ejecutado con skforecast 0.27.0 de PyPI.

## 8. Estado de los hallazgos del informe anterior (fase 2)

Leído después de escribir las secciones 1 a 7. Las personas de aquel informe eran otras (Ana, Luis, Marta, Pablo y Sara), así que comparo el fallo, no la persona.

| Anterior | Estado | Evidencia |
|---|---|---|
| **B1** "Copy script" copia un script sin datos (`NameError: name 'y'`) | **Arreglado** | Los 36 scripts copiados con "Copy script" traen un bloque de datos de ejemplo y terminan con código 0 en entornos nuevos (sección 5). |
| **B2** Cargar el CSV siguiendo los comentarios da errores engañosos (`Got -41`, índice sin frecuencia, formato largo) | **Arreglado en parte** | Los comentarios ya traen `read_csv(..., index_col="date", parse_dates=True)` y, en varias series, el `pivot`; un índice mal leído da `` `y` needs dates with a frequency, and values: see above.``. Persiste con datos reales más sucios: huecos y series que empiezan tarde (B2 nuevo) y CSV con `;` y coma decimal (B3 nuevo). |
| **B3** Fechas que no casan con la frecuencia: error engañoso | **Arreglado en parte** | Los comentarios explican `W-MON`, `QE`/`ME` y `dayfirst=True`, y Elena lo resolvió leyendo. Persiste cuando solo una parte de los valores queda en NaN: Dani, con `01/04/1998` leído como 4 de enero, ve números falsos antes del error (B3 nuevo). |
| **C1** El script compara pero no predice; MASE sin explicar | **Arreglado en parte** | La sección "Forecast with the model you keep" funciona en las 36 combinaciones, y el script ahora imprime el MAE "in the units of your data". Persiste que no hay forma de predecir con el baseline o la regla cuando ganan (C1 nuevo). |
| **C2** "Something else": script diario con `# <- edit` y sin ayuda | **Arreglado** | El comentario dice "With one value every 15 minutes and a daily pattern, season = 96 and steps = 96 forecast one day; with yearly data, season = 1", y la frase de datos mínimos dice "with these example settings: edit them for your data". Queda que `window_sizes=7` y `CalendarFeatures(["day_of_week", "month"])` no se señalan para editar (sección 4, Tom). |
| **C3** Varias series: formato largo y exog distinto por tienda no cubiertos | **Arreglado en parte** | El comentario trae el `pivot` del formato largo y menciona "a dict with one DataFrame per series". Sin ejemplo de cómo construirlo ni su futuro (C2 nuevo), y skforecast-ai falla con ese dict (C3 nuevo). |
| **C4** Clasificación: lecturas no usadas, test de 72 h con dos clases | **Persiste en parte** | "Good to know" ahora dice "If the categories come from a number (a level, a reading), you can also forecast the number...: choose A number". El test sigue siendo `3 * steps` (72 h en horario). No lo repetí con datos como los de Pablo. |
| **C5** "Four questions" frente a tres; el total cambia | **Persiste** | Al llegar se ven tres preguntas con "0 of 4 answered" y "the 3 questions left"; categoría sigue en "of 3" (C5 nuevo, `04_cuatro_preguntas_se_ven_tres.png`). |
| **C6** El `¶` de un encabezado borra las respuestas | **Arreglado en parte** | Las respuestas ya no se borran de la pantalla y la URL lleva el ancla (`#how-we-chose-these-rules`), pero la URL pierde las respuestas: compartirla no las comparte (C8 nuevo). No comprobé la entrada "Example script" del índice en el móvil. |
| **C7** Categoría: "Good to know" y "Next step" hablan de Chronos-2 y de regresión | **Arreglado en parte** | "Good to know" de categoría: "The script needs only skforecast and LightGBM: no PyTorch and no downloads.", y hay un enlace "more about forecasting categories" a la guía de clasificación. Persiste "In our test" en singular sin datos de clasificación en "How we chose these rules". |
| **C8** Cifras de "How we chose these rules" (26 frente a 33), datos públicos | **Arreglado** | "Some datasets appear in more than one row... so the rows add up to 33 cases" y "The datasets are public, so Chronos-2 may have seen some of them". Quedan detalles de redacción (D2 nuevo). |
| **C9** La búsqueda no encuentra la página | **Persiste (y ahora sé que es de esta rama)** | "which forecaster should I use" sigue en "No matching documents"; la rama 0.28.x construida igual y skforecast.org sí devuelven resultados con "which model should I use", "I" o "a" (B1 nuevo). |
| **C10** Instalación de 5,8 GB, consejo de torch CPU escondido | **Arreglado** | La línea para CPU va al principio del script ("Linux without a GPU? Install PyTorch for CPU first (about 4 GB less)"); la seguí y quedó `torch 2.15.0+cpu`. No medí el tamaño en disco. |
| **C11** En el móvil, tramos del código ocultos sin scroll | **Arreglado en parte** | Ya no se ocultan: las líneas no se parten y el `pre` se desplaza en horizontal (comprobado, `scrollLeft=200`). Sigue siendo incómodo y la línea de instalación no se lee de un vistazo (M1 nuevo). |
| **C12** Mensajes para asistentes con valores de ejemplo sin señalar | **Arreglado en parte** | Ahora dice "The file and the column names are examples: change them to yours." y el horizonte concreto ("next 8 weeks", "next 24 hours"). Persiste "next 14 values" en "Something else" (C11 nuevo). |
| **C13** La tabla de skforecast-ai sale cortada | **Arreglado** | Las líneas usan `print(result.results.to_string())` y se ven todas las columnas. |
| **M1** "Start over" deja la página a media altura y sin foco | **Arreglado en parte** | Tras "Start over" la página queda en `scrollY` 226 (antes 927), con las preguntas a la vista. No comprobé el foco. Nuevo: no se puede deshacer (M7 nuevo). |
| **M2** Avisos de copiado lejos del botón; Colab cambia el portapapeles | **Arreglado** | Cada botón cambia a "Copied ✓"; el aviso de Colab sale junto a su botón, y lo que copia Colab es el mismo script (3292 caracteres en ambos). |
| **M3** "Was this useful? No" no deja decir qué falló | **Arreglado en parte** | "No" ofrece "tell us what", pero lleva a un issue de GitHub en blanco que pide cuenta (M6 nuevo). |
| **M4** La salida asusta (tqdm, HF_TOKEN, `MissingValuesWarning`) | **Persiste** | Mismas barras y el mismo aviso de HF; una semana que falta da 9 recuadros (C4 y M5 nuevos). |
| **M5** Con menos datos de los necesarios no avisa | **No comprobado** | No repetí la prueba con una serie corta de una sola serie. En varias series, un producto con 19 meses hace fallar el baseline (B2 nuevo). |
| **M6** La recomendación se sale de la vista al contestar la última pregunta | **Persiste, cambiado** | Ahora la página baja sola al terminar (1280 px: de 33 a 725), hasta el script; el titular de la recomendación queda fuera por arriba. En el móvil no baja y el resultado queda bajo el pliegue (M2 nuevo). |
| **M7** Script en una caja de 520 px con scroll interno | **Arreglado** | El bloque mide el script entero (1560 px de alto, sin scroll vertical interno), y la línea `pip` va antes de "▸ imports". |
| **D1** Dos tarjetas vacías "Forecaster / Forecaster" | **Persiste** | Siguen en el estado vacío (`01_primera_impresion_pagina.png`). |
| **D2** "ForecasterRecursiveMulti / Series" partido a 360 px | **No comprobado** | No capturé esa tarjeta a 360 px con "Several". |
| **D3** Dos nombres para la página | **Persiste** | "Which forecaster should I use?" frente a "Choose a forecaster" (M8 nuevo). |
| **D4** Frases en inglés poco naturales | **Arreglado en parte** | "Choose this too if you only know them up to today", "the value one season before", "Runs as it is, with the sample data it includes", "`refit` has no effect on it". Persiste `y.shift(season)` en la tabla "The rules at a glance", y hay frases nuevas raras (D2 nuevo). |
| **D5** Sin JavaScript, el mensaje en una columna estrecha | **Arreglado** | El aviso ocupa el ancho de la caja. |
| **D6** Texto justificado con huecos en el móvil | **Persiste** | A 360 px, "Answer four questions about your data." sigue justificado con huecos (`10_movil_360_completo_ok.png`). |
| **D7** Volver de "A category" a "A number" pierde "How many time series?" | **Arreglado** | Several / Hour / Yes → A category → A number devuelve `number-several-hourly-yes`. |
| **D8** Una URL inválida en la misma pestaña mantiene el estado anterior | **Persiste** | `#answers=number-one-weekly-no&utm_source=x` o con un punto final muestran las respuestas de la visita anterior (C8 nuevo). |
| **D9** El README enlaza a una URL que da 404 | **No comprobado** | Es esperado hasta publicar; no lo miré. |
| **D10** "Back to top" tapa líneas del script | **Persiste** | Tapa el encabezado "Run the comparison on your data" a 400 px (M3 nuevo). |
| **D11** Las flechas cambian la respuesta al momento | **Persiste** | Flecha abajo en la pregunta 1 pasa de "A number" a "A category" (es el comportamiento estándar de un grupo de botones de opción). |
