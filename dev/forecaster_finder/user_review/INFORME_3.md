# Tercera revisión como usuario: "Which forecaster should I use?"

Rama `docs/forecaster-finder` (commit `27917bb`, después de fusionar `origin/0.28.x`). Construí el sitio en local sin el plugin `privacy` y lo serví con `python -m http.server`. Lo recorrí con Chromium y Playwright, a 1280×800 salvo que diga otra cosa, y miré cada captura antes de opinar. Copié los scripts con los botones de la página (leyendo el portapapeles del navegador) y los ejecuté en entornos nuevos creados con `python -m venv` y `pip`, con la línea de instalación de cada script. No abrí el código del selector ni el Markdown de la página. Las capturas están en `3/`, a escala 1.

## 1. Resumen en cinco líneas

1. **Sí se la enseñaría ya a usuarios reales.** No hay nada que bloquee: los 36 scripts, copiados con el botón y con sus líneas de predicción añadidas, se ejecutan tal cual en entornos nuevos (skforecast 0.27.0 de PyPI). La fusión no ha roto nada visible en la portada, la introducción, el Quick start, las tablas de contenidos ni las guías.
2. La página engancha y orienta: en menos de un minuto se llega a una recomendación con nombre, motivo, baseline y script, y desde la respuesta hay un camino para cada paso siguiente (datos propios, predecir, mejorar, entender, asistente de IA).
3. Lo que más resta es la respuesta para **categorías**: recomienda un clasificador y en la misma frase dice que "in our test it did not clearly beat the simple rule", enlaza a "How we chose these rules", donde no hay ni una cifra de categorías, y con los datos de ejemplo la regla gana en 7 de las 12 combinaciones.
4. En escritorio, al contestar la última pregunta, la página baja sola hasta el script y la recomendación (el titular y los nombres de los forecasters) queda fuera de la vista por arriba.
5. Detalles de llegada y seguimiento: la búsqueda no encuentra la página con "choose a forecaster" (su propio nombre en el menú) ni con "how do I start"; la indicación de instalar PyTorch para CPU "first" va después del comando (7,5 min y 5,8 GB en vez de 34 s y 1,4 GB); y la guía a la que lleva el "Next step" tiene dos enlaces rotos en sus propios "Next steps".

## 2. Primera impresión

**Página, diez segundos** (`3/01_primera_impresion_pagina.png`): el aviso de cookies tapa la mitad de abajo y difumina el resto, pero se lee el título "Which forecaster should I use?" y "Answer four questions about your data (three for a category). You get the forecasters to compare first, the baseline to compare them with and a script that runs as it is." Entiendo que es un cuestionario corto que me dice qué modelo probar y me da código. Lo que haría: aceptar las cookies y pulsar "A number". Al aceptar se ven las cuatro preguntas a la izquierda y, a la derecha, un panel oscuro "YOUR STARTING POINT · 0 OF 4 ANSWERED" con dos huecos que dicen "Forecaster". El recuento cuadra con lo que se ve.

**Portada, diez segundos** (`3/02_primera_impresion_portada.png`): "Time series forecasting, from prototype to production", una librería de Python. Debajo de los botones aparece "Not sure which forecaster to use? Find out in four questions →", visible sin bajar. Lo pulsaría si no sé por dónde empezar. Más abajo, la tarjeta "Not sure which one to use?" con "A number / A category" (`3/10_portada_selector.png`) entra al selector con la primera respuesta ya marcada; es la entrada más clara.

## 3. Hallazgos, de más a menos grave

### 1. Confunde. La respuesta para categorías se contradice y su enlace "How we chose these rules" no dice nada de categorías

- **A quién:** Pau (cualquier combinación de "A category").
- **Qué hice:** contesté A category / Day / Yes (y luego las 12 combinaciones de categoría). Leí la respuesta, pulsé "How we chose these rules", copié el script con "Copy script", lo ejecuté en un entorno nuevo con `pip install skforecast lightgbm` y abrí "Forecast with the model you keep".
- **Esperaba:** que la página me dijera con claridad qué usar y, al pulsar "How we chose these rules", ver en qué se basa esa frase.
- **Pasó:**
  - El titular dice "Start with a machine learning classifier and compare it with a simple rule", y justo debajo: "A classifier is the model for categories. In our test it did not clearly beat the simple rule, so keep the rule as your baseline." (`3/03_categoria_mensaje.png`). "Keep the rule as your baseline" se puede leer como "usa la regla" o como "sigue comparando con la regla".
  - El enlace "How we chose these rules" baja a una sección cuya tabla solo tiene filas para números ("One series, nothing known in advance", "One series, variables known in advance", "Several series"), y ninguna frase habla de clasificación (`3/04_como_elegimos_sin_categorias.png`). "Our test" no aparece en ningún sitio.
  - Con los datos de ejemplo del propio script, la regla gana en 7 de las 12 combinaciones. Por ejemplo, Day / Yes: `Classifier: 0.905` y `Rule: 1.000`; Month / No: `Classifier: 0.667` y `Rule: 0.917`. En Day / No gana el clasificador (`0.952` frente a `0.857`).
  - "Forecast with the model you keep" dice "Keep the lines of the classifier if it beats the rule" y solo trae líneas para el clasificador: si gana la regla, que es lo que ve Pau la mayoría de las veces, no hay nada que copiar.
- **Por qué importa:** Pau llega a una recomendación cuya única justificación remite a una sección que no la contiene, y la primera vez que ejecuta el ejemplo ve que el modelo recomendado pierde contra una regla trivial. Sale sin saber si usar el clasificador o la regla, que es justo lo que venía a resolver.

### 2. Molesta. En escritorio, al contestar la última pregunta la recomendación sale de la vista

- **A quién:** Lucía y Marcos, en un portátil o un monitor (1024, 1280 y 1920 px).
- **Qué hice:** contesté las cuatro preguntas con el ratón, la última la de "Will you know anything else in advance?".
- **Esperaba:** ver la recomendación, que es la respuesta a la pregunta del título.
- **Pasó:** la página baja sola (a 1280×900, de la pregunta 4 a `scrollY` 725) hasta "Run the comparison on your data". Lo que queda a la vista es el script; el titular "Compare two forecasters: LightGBM and Chronos-2" y los nombres de las tarjetas quedan por encima del borde (`3/05_escritorio_ultima_respuesta.png`). A 1920 px se ven las tarjetas cortadas por arriba, sin el nombre del forecaster. En el móvil (360 px y iPhone 13 emulado) es al revés y está bien: la página baja hasta el titular de la recomendación.
- **Por qué importa:** Lucía no sabe qué es un forecaster y viene a por una recomendación; lo primero que ve al terminar es una pared de código. Tiene que subir para leer lo que se le recomienda y por qué.

### 3. Molesta. La búsqueda no encuentra la página con "choose a forecaster", su propio nombre en el menú

- **A quién:** quien busca en el sitio en vez de navegar; Pau llega por la búsqueda.
- **Qué hice:** escribí frases en la caja "Search" de la portada y bajé por la lista de resultados hasta cargar todos los que da la página (de 16 a 34 resultados por consulta).
- **Pasó:**

  | Consulta | ¿Aparece la página? |
  |---|---|
  | which forecaster should I use | Sí, la primera |
  | what model should I use | Sí, la primera |
  | forecaster finder | Sí, la primera |
  | which forecaster | Sí, la segunda |
  | choose a forecaster | **No** (salen "ARIMA, SARIMAX, AutoARIMA", "Using AI assistants with skforecast", "ETS, AutoETS"; `3/06_busqueda_choose_a_forecaster.png`) |
  | choose a model, choose | **No** |
  | how do I start | **No** (salen "Independent multi-time series forecasting", "Installation Guide", "Introduction to forecasting", "Quick start") |
  | which model, best model | **No** |
  | predict a category, classification | **No** |

  La búsqueda funciona en general: "backtesting", "prediction intervals", "exogenous variables", "lightgbm" o "ForecasterRecursiveMultiSeries" devuelven primero la guía que corresponde.
- **Por qué importa:** la página se llama "Choose a forecaster" en la cabecera al desplazarse, en el menú del Quick start y en el "Next" del Quick start, pero buscar ese nombre no la encuentra. Pau, que busca "classification" o "predict a category", llega a la guía de clasificación, que tampoco enlaza al selector en su texto.

### 4. Molesta. "Install PyTorch for CPU first" va después del comando que hay que ejecutar primero

- **A quién:** Marcos y cualquiera en Linux sin GPU (lo normal en un portátil o un servidor de pruebas).
- **Qué hice:** el script empieza así (`3/07_lineas_de_instalacion.png`):
  ```
  # pip install skforecast lightgbm chronos-forecasting
  # Linux without a GPU? Install PyTorch for CPU first (about 4 GB less):
  #     pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
  Leyendo de arriba abajo, ejecuté la primera línea en un entorno nuevo. Después repetí con el orden que dice la segunda.
- **Pasó:** con la primera línea tal cual, **457 s y 5,8 GB** (PyTorch con CUDA y 15 paquetes `nvidia-*`). Con PyTorch para CPU antes, **34 s (20 + 14) y 1,4 GB**; el script da los mismos números. El consejo funciona y la cifra "about 4 GB less" es cierta (5,8 frente a 1,4), pero "first" llega una línea tarde: quien copia y pega la primera línea ya está descargando los 5,8 GB cuando lee la segunda.
- **Por qué importa:** son siete minutos y medio de espera en el primer contacto, justo cuando la página promete "un minuto". Es el momento en que alguien con prisa abandona.

### 5. Molesta. La guía a la que lleva el "Next step" tiene dos enlaces rotos en sus propios "Next steps"

- **A quién:** Marcos, Lucía e Irene, en todas las combinaciones numéricas.
- **Qué hice:** pulsé "Next step: compare more model families the same way" (debajo del script) y "Comparing forecasting models" (en "Next steps" al final de la página). Las dos llevan a *Comparing forecasting models*. Bajé a su sección "Next steps" y pulsé cada enlace.
- **Pasó:** "global models" (en "Recursive multi-step forecasting and global models...") y "probabilistic forecasting" (en "Hyperparameter tuning and probabilistic forecasting") apuntan a `global-forecasting-overview.md` y `probabilistic-forecasting-overview.md`, y dan 404 (`3/08_guia_comparing_next_steps.png`). Lo mismo pasa hoy en https://skforecast.org/latest/user_guides/comparing-forecasting-models.html, así que no lo ha roto la fusión. Los demás enlaces de la guía funcionan.
- **Por qué importa:** es el siguiente paso que la página recomienda en primer lugar, y desde ahí el usuario sigue descubriendo la librería. Dos de sus seis salidas son un 404.

### 6. Detalle. En el móvil y en tableta, el nombre del forecaster de varias series se parte por la mitad

- **A quién:** Irene (móvil) y cualquiera a 360, 400 o 1024 px con "Several".
- **Pasó:** la tarjeta dice "ForecasterRecursiveMulti" en una línea y "Series" en la siguiente (`3/09_movil_nombre_partido.png`). A 768 y 1280 px se ve entero.
- **Por qué importa:** es el nombre que Irene va a buscar o a pasar a su equipo, y partido parece otro nombre.

## 4. El recorrido de cada persona

- **Lucía (portada, dos minutos, no sabe qué es un forecaster).** Aceptó las cookies, vio "Find out in four questions →" sin bajar y llegó al selector. Las preguntas se entienden sin saber de series temporales: cada opción trae un ejemplo ("Sales, energy demand, website visits", "The total sales of a company") y las palabras técnicas se explican al lado ("skforecast calls this the frequency of the series", "skforecast calls them exogenous variables"). Tuvo una recomendación en menos de un minuto, pero en un portátil la página la llevó al script y tuvo que subir para leerla (hallazgo 2). Las etiquetas "YOU TRAIN IT" y "NO TRAINING" le explican la diferencia entre los dos modelos mejor que sus nombres. No ejecuta nada. Para seguir tiene "Learn the API step by step: Quick start" al final de la página. Se va con una idea clara de qué ofrece la librería.
- **Marcos (Intro to Forecasting, serie diaria).** El recuadro "Not sure which forecaster to use?" está arriba del todo en la introducción. Contestó A number / One / Day / No en unos segundos. Copió el script, lo instaló en un entorno nuevo con la primera línea (457 s, hallazgo 4) y lo ejecutó: 19,2 s la primera vez (descarga Chronos-2) y 6,9 s la segunda. La salida se entiende: `Baseline: 1.341`, `LightGBM: 1.334`, `Chronos-2: 0.791`, con el comentario "Lower is better: keep a model only if it beats the baseline". Para seguir: usar sus datos, el comentario del bloque de datos con `read_csv` y "Use your own data: Input data"; predecir, "Forecast with the model you keep" (las líneas funcionan y dan 14 días); mejorar, "Exogenous variables" y "Hyperparameter tuning and lags selection"; entender el porqué, "How we chose these rules". Supo adónde ir en cada caso. El "Next step" que tiene justo debajo del script es comparar más familias de modelos, cuya guía tiene dos enlaces rotos (hallazgo 5).
- **Irene (móvil, varias series, conoce sus promociones).** En el iPhone emulado, las respuestas se pliegan en etiquetas editables ("A number ✎", "Several ✎", "One value per week ✎") y al contestar la última la página baja hasta la recomendación: "Compare two forecasters: LightGBM and Chronos-2", las dos con la etiqueta "uses your known variables". Menos de un minuto. Para compartir no hay botón, pero la URL de la barra lleva las respuestas (`#answers=number-several-weekly-yes`) y la abrí en otro navegador sin historial: muestra las mismas respuestas, también con `)`, `.` o `&utm_source=slack` añadidos al final. Solo le molesta el nombre partido (hallazgo 6).
- **Pau (categoría).** Por la búsqueda no llegó ("classification" y "predict a category" no sacan la página, hallazgo 3); llegó por la tabla de contenidos de User Guides, tarjeta "Choosing a model". Contestó A category / Day (la pregunta de cuántas series desaparece y el recuento pasa a "of 3", como anuncia la introducción). Instalación de 8 s y 500 MB, ejecución de 5,2 s: `Classifier: 0.952`, `Rule: 0.857`. Con "Yes", la regla gana (`1.000` frente a `0.905`). La respuesta y la sección "How we chose these rules" no le aclaran qué usar (hallazgo 1). Para seguir tiene "Next step: more about forecasting categories", que lleva a la guía de clasificación.

## 5. Tiempos medidos y entorno

Entorno: contenedor Linux x86_64 (kernel 6.18, sin GPU), Python 3.12.3, entornos nuevos con `python -m venv` y `pip`. Construir el sitio: 60 s.

Versiones que instaló `pip`: skforecast **0.27.0** (PyPI), lightgbm 4.7.0, chronos-forecasting 2.3.2, torch 2.15.0 (con CUDA) o 2.15.0+cpu, pandas 2.3.3, numpy 2.5.5, statsmodels 0.15.0 (con `skforecast[stats]`), skforecast-ai 0.5.0.

| Instalación (entorno nuevo) | Tiempo | Tamaño |
|---|---|---|
| `pip install skforecast lightgbm chronos-forecasting` (primera línea, caché de pip vacía) | 457 s | 5,8 GB |
| `pip install torch --index-url .../cpu` y después la primera línea | 20 s + 14 s | 1,4 GB |
| `pip install "skforecast[stats]" lightgbm chronos-forecasting` (caché caliente) | 52 s | 5,8 GB |
| `pip install skforecast lightgbm` (categoría, caché caliente) | 8 s | 500 MB |
| primera línea + `"skforecast-ai[foundation]"` (caché caliente) | 69 s | |

| Ejecución | 1.ª | 2.ª |
|---|---|---|
| Marcos, una serie diaria (la 1.ª descarga Chronos-2) | 19,2 s | 6,9 s |
| Marcos, con PyTorch para CPU | 15,3 s | |
| 36 scripts + líneas de predicción, pesos ya descargados | categoría 1,4 a 5,2 s; números 6,5 a 9,9 s; mensual con ARIMA 31,0 s la primera vez | |
| Líneas de skforecast-ai (una serie diaria / varias series / mensual) | 38 s / 11 s / 10 s | |

Los 36 scripts terminaron con código 0 e imprimieron sus predicciones. Las tres pruebas de skforecast-ai también.

## 6. Lo que funcionó bien y no tocaría

- **Los scripts:** "Copy script", el "Copy" de la cabecera y "Run it in Google Colab" copian exactamente lo mismo, y los 36 se ejecutan tal cual. Lo que imprimen se lee de un vistazo (tres o cuatro líneas con el nombre del modelo y su error en las unidades de los datos).
- **Las 36 respuestas son coherentes:** el baseline ("the same hour the day before", "the same month the year before"...), `steps`, `season`, la frase de datos mínimos ("Needs at least 336 values (2 weeks)", "36 values (3 years)"...) y las líneas de predicción casan con la frecuencia elegida, y coinciden con "The scripts need at least 2 weeks of hourly data, 3 months of daily data..." al final de la página.
- **Las cifras:** la tabla suma 16 + 7 + 10 = 33 casos y lo explica; las frases de cada respuesta ("Chronos-2 was usually the most accurate", "LightGBM often needed work on its features to beat the baseline", "ARIMA ... often did better than LightGBM out of the box") cuadran con la tabla. La página avisa de que la guía *Comparing forecasting models* puede ir en contra en un conjunto de datos concreto, y lo comprobé: en su ejemplo mensual ARIMA queda detrás de LightGBM, y con variables exógenas LightGBM gana a Chronos-2.
- **El estado:** las respuestas sobreviven a recargar, a ir a una guía y volver (con "atrás" y con un enlace nuevo), a pulsar el `¶` de un encabezado (en pantalla), a aceptar las cookies o guardarlas desde "Manage settings" y a contestar en cualquier orden; el recuento ("1 OF 4", "3 OF 3") cuadra siempre con lo que se ve. "Start over" deja la página con las preguntas a la vista. El doble clic y varios botones de copiar seguidos no hacen nada raro.
- **Móvil, teclado, tema oscuro y sin JavaScript:** sin desplazamiento horizontal de la página en ningún ancho; el script se desliza dentro de su caja. Con teclado se contesta con Tab, flechas y espacio, y el foco se ve. En oscuro todo se lee. Sin JavaScript queda un aviso, las reglas y un script de ejemplo completo.
- **Las salidas:** cada respuesta enlaza a la guía de cada forecaster y del baseline, todas cargan; "Using an AI assistant?" trae un mensaje con el horizonte concreto y la advertencia "The file, the column names and the horizon are examples", y el prompt pide al asistente una muestra de los datos antes de tocar nada. "Was this useful? No" ofrece "tell us what" con un issue de GitHub con título.
- **El resto del sitio después de la fusión:** portada, introducción, Quick start, tablas de contenidos de guías y FAQ y varias guías cargan sin imágenes rotas ni errores visibles; los diagramas de la FAQ "How training matrices are built" se dibujan; los menús funcionan.

## 7. Lo que no pude probar

- El notebook de Colab (404 esperado): comprobé que el botón copia el mismo script y muestra "Script copied: paste it in the second cell of the notebook that just opened.", no el notebook.
- Un asistente de IA real y el servidor MCP de skforecast-ai: solo leí el mensaje y el prompt.
- Un móvil real, Safari, Windows y macOS: usé la emulación de Chromium (360, 400 px e iPhone 13). La diferencia de instalación del hallazgo 4 es solo de Linux.
- Lucía y Pau no los cronometré como personas: mis tiempos de contestar son de un navegador automático (segundos); una persona tardaría del orden de un minuto.
- La primera instalación (457 s) depende de la red del contenedor; las siguientes aprovecharon la caché de `pip`, así que sus tiempos son menores que los de un equipo vacío.
- pandas 3: `pip` instaló pandas 2.3.3.
- Hugging Face bloqueado: no lo probé.

## 8. Estado de los hallazgos anteriores (fase 2)

Leí `INFORME.md` e `INFORME_2.md` después de escribir las secciones 1 a 7, salvo una cosa: al leerlos volví a mirar mi captura de 1280 px tras la última respuesta, ya tomada en la fase 1, y de ahí sale el hallazgo 2. Solo incluyo los hallazgos anteriores que entran en el criterio de esta revisión (los de cargar y limpiar datos reales, no).

| Anterior | Estado | Evidencia |
|---|---|---|
| 1-B1 "Copy script" sin datos (`NameError`) | Arreglado | Los 36 scripts copiados terminan con código 0 en entornos nuevos. |
| 1-C1 El script compara pero no predice | Arreglado | "Forecast with the model you keep" funciona en las 36 combinaciones. |
| 2-C1 Si gana el baseline o la regla, no hay líneas para predecir con él | Persiste | Solo hay líneas para los modelos; en categoría, "Keep the lines of the classifier if it beats the rule" (hallazgo 1). |
| 1-C5 / 2-C5 "Four questions" frente a tres; recuento que no cuadra | Arreglado | Al llegar se ven cuatro preguntas y "0 OF 4"; la introducción dice "(three for a category)"; contestando la 4 y la 3 primero, "2 OF 4" y "the 2 questions left". |
| 1-C6 / 2-C8 El `¶` y las cookies quitan las respuestas de la URL; URL con algo añadido | Arreglado en parte | Aceptar cookies conserva `#answers=` y las URL con `)`, `.` o `&utm_source=` muestran sus respuestas en un navegador nuevo. Tras pulsar un `¶`, la URL pasa a `#how-we-chose-these-rules` y, compartida así, abre el selector vacío. No repetí la prueba de "otra visita en la misma pestaña". |
| 1-C7 / 2-C7 Categoría: "In our test" sin datos de clasificación | Persiste | "How we chose these rules" sigue sin filas ni frases de categorías (hallazgo 1). "Good to know" ya habla solo del clasificador. |
| 1-C9 / 2-B1 La búsqueda no encuentra la página | Arreglado en parte | "which forecaster should I use" la da la primera y las frases con "I", "a" o "should" ya devuelven resultados; "choose a forecaster" y "how do I start" no la encuentran (hallazgo 3). |
| 1-C10 / 2-C10 (instalación) Consejo de PyTorch para CPU | Arreglado en parte | Ya está arriba del script, pero después del comando: 457 s y 5,8 GB siguiendo el orden de lectura (hallazgo 4). |
| 2-C6 Contradicción con *Comparing forecasting models* | Arreglado | La página dice ahora "A single dataset can go against these counts: in its monthly example ARIMA is behind LightGBM, and with exogenous variables LightGBM is ahead of Chronos-2". |
| 2-C10 "Test: the last 3 periods" sin decir cuánto | Arreglado | "Test: the last 3 forecasts of 14 days each (42 days)." |
| 1-C12 / 2-C11 Mensajes para el asistente con valores de ejemplo | Arreglado en parte | "The file, the column names and the horizon are examples: change them to yours." Con "Something else" sigue "the next 14 values". |
| 1-C13 Tabla de skforecast-ai cortada | Arreglado | `to_string()`: se ven todas las columnas. |
| 2-C3 (parte) skforecast-ai con varias series prueba solo dos modelos | Persiste | Con los datos de ejemplo, el ranking de varias series tiene dos filas y ningún baseline, aunque la sección dice "tries more of them". |
| 1-M1 "Start over" deja la página a media altura | Arreglado | Tras "Start over", `scrollY` 226, con las preguntas a la vista. |
| 1-M2 Avisos de copiado lejos del botón; Colab cambia el portapapeles | Arreglado | Cada botón pasa a "Copied ✓"; Colab copia el mismo script y avisa junto a su botón. |
| 1-M3 / 2-M6 "Was this useful? No" | Arreglado en parte | "tell us what" abre un issue con el título "Feedback on "Which forecaster should I use?""; sigue pidiendo cuenta de GitHub. |
| 1-M6 / 2-M2 La recomendación sale de la vista al terminar | Arreglado en el móvil, persiste en escritorio | En el móvil la página baja hasta el titular; en escritorio baja hasta el script (hallazgo 2). |
| 1-M7 Script en una caja con scroll interno | Arreglado | El bloque muestra el script entero. |
| 1-C11 / 2-M1 Código cortado en el móvil | Arreglado en parte | No se oculta nada: el script se desliza en horizontal dentro de su caja. No lo incluyo como hallazgo. |
| 1-D1 Tarjetas vacías "Forecaster / Forecaster" | Persiste | Siguen en el estado vacío; no lo incluyo como hallazgo. |
| 1-D2 "ForecasterRecursiveMulti / Series" partido | Persiste | A 360, 400 y 1024 px (hallazgo 6). |
| 1-D3 / 2-M8 Dos nombres para la página | Persiste | "Which forecaster should I use?" en el título y "Choose a forecaster" en la cabecera y el menú; buscar el segundo no la encuentra (hallazgo 3). |
| 1-D4 / 2-D2 / 2-D3 Frases poco naturales | Arreglado en gran parte | La tabla dice "Lower error: Chronos-2 / LightGBM"; "Except in that search, each model has one configuration"; en categoría, "Keep the lines of the classifier if it beats the rule". Queda `y.shift(season)` en "The rules at a glance" y el doble comentario `# steps ahead of each forecast  # <- edit` en "Something else". |
| 1-D5 Sin JavaScript, columna estrecha | Arreglado | El aviso ocupa el ancho de la caja. |
| 1-D7 Volver de "A category" a "A number" pierde una respuesta | Arreglado | Several / Month / Yes → A category → A number devuelve `number-several-monthly-yes`. |
| 2-M7 "Start over" no se puede deshacer | Persiste | "Atrás" tras "Start over" sale de la página. No lo incluyo como hallazgo. |
