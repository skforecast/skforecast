# Revisión como usuario: "Which forecaster should I use?"

Rama `docs/forecaster-finder`, sitio construido en local (sin el plugin `privacy`) y servido en `http://localhost:8765`. Recorrido hecho con Chromium + Playwright mirando las capturas, y scripts ejecutados en entornos nuevos con `pip` (no `uv`), con CSV públicos guardados aparte. No he leído el código del selector ni el Markdown de la página.

Las capturas están en esta carpeta (PNG a escala 1); cada hallazgo visual dice cuál es.

## 1. Resumen

- **Todavía no se lo enseñaría a usuarios reales.** La parte visual y el selector están muy bien hechos, pero la promesa central de la página ("a script that runs as it is") no se cumple con el botón principal.
- "Copy script" copia un script **sin datos**: al pegarlo y ejecutarlo da `NameError: name 'y' is not defined`. Solo el botón de Colab copia una versión con datos de ejemplo, y en ningún sitio se explica cómo cargar un CSV.
- Cargar los datos es justo donde se atascan todas las personas: con el ejemplo del comentario (`data["sales"].asfreq("D")`) salen errores que no apuntan a la causa (`initial_train_size ... Got -41`, `no observed values in the test set`), y basta con fechas a fin de mes, semanas en lunes o fechas `dd/mm/yyyy` para fallar.
- Cuando funciona, el script imprime tres números (MASE) que nadie sabe interpretar (de 0,2 a 19 según los datos) y no hace ninguna predicción: quien llega queriendo predecir sus ventas se va sin predicción.
- Lo demás (casos de "Something else", varias series en formato largo, clasificación, móvil, "Start over", enlaces `¶`) son problemas menores o medianos y no hace falta rehacer nada para arreglarlos.

## 2. Primera impresión (diez segundos)

**Página** (`01_primera_impresion_pagina.png`): "Es un asistente que me hace preguntas sobre mis datos y me da un script". Se entiende enseguida. Lo primero que hago es pulsar "A number". Me choca que la frase de arriba diga "Answer **four** questions" y que el panel diga "0 OF **3** ANSWERED" con tres preguntas a la vista. Las dos tarjetas vacías "Forecaster / Forecaster" no dicen nada. El bloque "We tested each kind of model on 26 public datasets..." parece serio, pero no sé qué hacer con él antes de contestar.

**Portada** (`02_primera_impresion_portada.png`): "Librería de Python para predecir series temporales". El enlace "Not sure which forecaster to use? Find out in four questions →" se ve sin hacer scroll, justo debajo de los botones, y deja claro qué vas a obtener. Más abajo hay otra tarjeta, "Not sure which one to use?", con los botones "A number" / "A category", que llevan al selector con la primera respuesta ya marcada (bien resuelto).

## 3. Hallazgos (de más a menos grave)

### B1. "Copy script" copia un script que no se ejecuta: no tiene datos

- **Gravedad:** Bloquea.
- **A quién:** a todas las personas y en todas las combinaciones. Le pasa sobre todo a Ana, que no sabe Python.
- **Pasos:** 1) Contestar A number / One / Day / No. 2) Pulsar "Copy script". 3) Pegar el contenido en `first_comparison.py`. 4) Crear un entorno nuevo con `pip install skforecast lightgbm chronos-forecasting`, como dice la primera línea. 5) Ejecutar `python first_comparison.py`.
- **Esperaba:** que corriera, porque la página lo promete tres veces: "a script that runs as it is" (introducción de la página, portada, Intro, README), "Runs as it is: tested on sample data every time it changes" (debajo del script, `03_runs_as_it_is.png`) y "Example settings so that the script runs" (comentario del script).
- **Pasó:**
  ```
  File "first_comparison.py", line 19, in <module>
      steps=steps, initial_train_size=len(y) - 3 * steps,
  NameError: name 'y' is not defined
  ```
  El script solo trae un comentario: `# y: pandas Series with a DatetimeIndex, freq="D",  e.g. y = data["sales"].asfreq("D")`. No dice cómo obtener `data` a partir de un CSV.
- **Detalle:** el botón "Run it in Google Colab" sí copia otra versión, con un bloque `# Sample data so that the script runs here. Replace this block with your own data.` (una serie sintética con `pd.date_range` y `np.sin`). "Copy script" y el "Copy" de la cabecera del bloque copian la versión sin datos (comprobado: son idénticas, 2080 caracteres en el caso de Ana, frente a 2461 de la de Colab). Así que "tested on sample data" es cierto para un script que el usuario no recibe al pulsar el botón principal.
- **Por qué importa:** es lo primero que hace cualquiera, y falla en la línea 19 de un script que le han prometido que funciona. Ana no sabe qué es `y` ni por qué no está definida. A Ana le habría bastado con que el script empezara con una línea `data = pd.read_csv("my_data.csv", ...)` que ella pudiera cambiar.

### B2. Cargar el CSV siguiendo los comentarios lleva a errores que no explican qué pasa

- **Gravedad:** Bloquea.
- **A quién:** Ana (una serie, diario) y Marta (varias series, diario, formato largo).
- **Pasos para Ana:** con su CSV (`date,sales`, 1826 días de la tienda 1 de `store_sales`), hacer lo que sugiere el comentario: `data = pd.read_csv("ana_ventas_tienda.csv")` y `y = data["sales"].asfreq("D")`.
- **Pasó (Ana):**
  ```
  ValueError: `initial_train_size` must be an integer greater than 0, a date string, a pandas Timestamp, or None. Got -41.
  ```
  Ese `-41` no tiene nada que ver con lo que hizo. La causa es que el índice no es de fechas (faltan `index_col="date", parse_dates=True`), y eso la página no lo dice. Con esos dos argumentos, cosa que Ana no sabría poner, funciona.
- **Pasos para Marta:** su CSV es de formato largo (`date,store,sales,holiday,promo`: 30 series de M5, tienda × categoría, 1969 días; `holiday` = evento del calendario, `promo` = días SNAP de su estado). Siguiendo el comentario `series = {name: data[name] for name in data.columns}` con `index_col="date", parse_dates=True` y `exog = data[["holiday", "promo"]]`:
  ```
  ValueError: `y` has a pandas DatetimeIndex without a frequency. To avoid this error, set the frequency of the DatetimeIndex.
  ```
  El comentario da por hecho una tabla ancha, con una columna por tienda, y nada lo avisa. Para que funcione hace falta `data.pivot(index="date", columns="store", values="sales").asfreq("D")`, algo que Marta no sabe hacer.
- **Por qué importa:** la mayoría de los CSV reales de ventas por tienda están en formato largo, y la mayoría de usuarios no sabe qué es `asfreq`. Los dos errores mandan a buscar en el sitio equivocado.

### B3. Fechas que no coinciden con la frecuencia que elige el script: error engañoso

- **Gravedad:** Bloquea.
- **A quién:** Luis (mensual con fechas a fin de mes), cualquiera con datos semanales fechados en lunes y Ana si exporta su CSV desde un Excel en español (`dd/mm/yyyy`).
- **Pasos:** cargar bien el CSV (`index_col`, `parse_dates`) y usar la línea del comentario: `asfreq("MS")` (mensual), `asfreq("W")` (semanal) o `asfreq("D")` (diario).
  - Luis, con `month` = 1991-07-31, 1991-08-31... (los mismos datos de `h2o` con fecha de fin de mes).
  - Semanal, con las ventas de Ana agregadas por semanas que empiezan en lunes (`W-MON`).
  - Ana, con las fechas escritas como `25/12/2017` (pandas no las convierte y el índice queda como texto).
- **Pasó (en los tres casos):**
  ```
  ValueError: The series has no observed values in the test set of the folds, so the metrics cannot be calculated. Review `initial_train_size` of `cv`, or the span of the series.
  ```
  `asfreq("MS")` sobre fechas de fin de mes, o `asfreq("W")` (domingos) sobre lunes, deja la serie entera en `NaN` sin decir nada. El error manda a revisar `initial_train_size`, que el usuario no ha tocado. Los comentarios no explican qué significan `"MS"`, `"W"` ni `"QS"`.
- **Por qué importa:** Luis tiene experiencia y lo acabará encontrando, pero perderá un buen rato. Ana no lo encontrará.

### C1. El script compara, pero no predice, y los números no se entienden

- **Gravedad:** Confunde (casi Bloquea para Ana y Marta, que vienen a predecir).
- **A quién:** a todas. Lo que obtuvo cada una:
  - Ana: `Baseline: 0.869 / LightGBM: 0.547 / Chronos-2: 0.219`.
  - Luis: `Baseline: 0.876 / LightGBM: 0.874 / ARIMA: 0.821 / Chronos-2: 0.645`.
  - Marta: `Baseline: 1.003 / LightGBM: 0.771 / Chronos-2: 0.697`.
  - Sara: `Baseline: 19.166 / LightGBM: 16.341 / Chronos-2: 10.490`, después de ajustar `steps`, `season` y `lags` a 96.
- **Qué hay:** "Error of each model: lower is better. Keep a model only if it beats the baseline." y "Next step: compare more model families the same way" (enlace a *Comparing forecasting models*, que es otra comparación). En ningún sitio se dice qué es la MASE ni qué valor es bueno (la sigla solo aparece en "How we chose these rules": "The error is the MASE"). Tampoco se explica cómo, una vez elegido el ganador, se predicen las próximas dos semanas.
- **Por qué importa:** Ana quería saber cuánto venderá. Se va con tres números sin unidades y sin predicción. Sara ve que todos sus valores son mayores que 1 y no sabe si "todos son malos" o si es normal. Marta solo ve la media de sus 30 tiendas, no cada tienda. A todas les habría servido una línea final del tipo "el ganador es X; así se entrena y se predice".

### C2. "Something else" (15 minutos, un año...) da el script diario con `# <- edit` y sin más ayuda

- **Gravedad:** Confunde.
- **A quién:** Sara (datos cada 15 minutos, `ett_m2`, 69 680 valores) y cualquiera con datos anuales.
- **Pasos:** A number / One / Something else / No, "Copy script" y cargar el CSV con `asfreq("15min")`.
- **Pasó:** el script trae `steps = 14  # <- edit`, `season = 7  # <- edit`, `lags=14,  # <- edit for your data` y `CalendarFeatures(features=["day_of_week", "month"])`. Ejecutado sin tocar nada, **funciona y da números** (`9.101 / 5.179 / 3.597`), pero con una "estacionalidad" de 7 pasos (105 minutos) que no tiene sentido, y nada avisa de ello. No se dice qué poner (¿96 para un día?). Las variables de calendario no incluyen `hour`, que con 15 minutos sería lo primero. La frase de datos mínimos es "Needs at least 90 values": para datos anuales serían 90 años. La frase de base es "the step one season before", que no se entiende.
- **Por qué importa:** Sara puede salir con una comparación sin sentido y fiarse de ella. A Sara le habría bastado con una frase como "si tus datos son cada 15 minutos, un día son 96 pasos".

### C3. Varias series: formato largo y promociones distintas por tienda no están cubiertos

- **Gravedad:** Confunde.
- **A quién:** Marta (A number / Several / Day / Yes).
- **Qué pasó:** el comentario dice `exog: DataFrame with the same index and the values you know in advance`, un único DataFrame para todas las series. Sus promociones son distintas en cada tienda (dependen del estado). Probé a pasarlas como diccionario por tienda (`exog = {tienda: DataFrame(holiday, promo)}`) y funciona (`1.003 / 0.724 / 0.678`, 16 s), pero la página no menciona esa posibilidad; tuve que usar conocimiento de skforecast. Con solo los festivos comunes, tras pivotar, sale `1.003 / 0.771 / 0.697`.
- **Por qué importa:** Marta llega precisamente porque conoce sus promociones. Si las mete en un DataFrame común con 30 columnas `promo_tienda`, el script correrá y mezclará la información de todas las tiendas sin que ella lo sepa (esto no lo ejecuté: es lo que haría alguien que solo tiene el comentario).

### C4. Clasificación: las lecturas horarias de Pablo no se usan y el resultado es muy frágil

- **Gravedad:** Confunde (no estoy seguro de que sea un problema de la página y no de mis datos).
- **A quién:** Pablo (A category / Hour / No). Datos: `ett_m2` agregado por horas, con `oil_temperature` y `level` (low/medium/high por terciles), 17 420 horas.
- **Qué pasó:** la opción "No" ("Also if you only know them up to today, like sensor readings") le dice que conteste No, cosa correcta para no usar el futuro. Pero entonces el script solo usa los retardos de la categoría: sus lecturas numéricas no entran en ningún sitio, y la página no dice cómo aprovecharlas. Resultado: `Classifier: 0.500 / Rule: 0.889`. El periodo de prueba son las últimas 72 horas (3 × 24), en las que solo hay dos clases (63 `high` y 9 `medium`), y el clasificador predijo `high` en las 72 horas (comprobado con una tabla cruzada), de ahí el 0,500 de *balanced accuracy*. Con 2 años de datos, la conclusión depende de 3 días.
- **Por qué importa:** Pablo quería predecir el nivel "a partir de las lecturas" y la página ignora las lecturas sin decirlo. Además, con una prueba tan corta puede quedarse con la regla por casualidad.

### C5. "Four questions" frente a tres, y un contador que cambia de total

- **Gravedad:** Confunde.
- **A quién:** a todas, sobre todo a quien elige "A category" (Pablo).
- **Pasos:** abrir la página: dice "Answer four questions" y el panel, "0 OF 3 ANSWERED" (`01_primera_impresion_pagina.png`). Al pulsar "A number" aparece una pregunta nueva ("How many time series?") y el contador pasa a "1 OF 4 ANSWERED". Con "A category" el total se queda en 3 ("3 OF 3 ANSWERED"). La portada ("Find out in four questions", "Answer four questions about your data"), la Intro, las guías ("Four questions and a script that runs"), los ejemplos ("en cuatro preguntas", "通过四个问题") y el README dicen siempre cuatro.
- **Por qué importa:** es poco, pero es lo primero que se ve, y que el total cambie hace dudar de si se ha saltado algo.

### C6. Pulsar el `¶` de un encabezado borra todas las respuestas

- **Gravedad:** Confunde.
- **A quién:** cualquiera que quiera compartir un enlace a "How we chose these rules", "The rules at a glance" o "Next steps", o que use la tabla de contenidos.
- **Pasos:** contestar las cuatro preguntas (URL `...#answers=number-one-daily-no`), bajar a "The rules at a glance", pasar el ratón por el título y pulsar `¶`.
- **Pasó:** la URL se queda en `.../choose-a-forecaster.html`, **sin** `#the-rules-at-a-glance` y sin las respuestas, y el selector vuelve a "0 OF 3 ANSWERED" (`06_permalink_borra_respuestas.png`). Pasa igual con los tres encabezados. El botón atrás lo recupera. Al abrir desde fuera un enlace a una sección (`#how-we-chose-these-rules`), la página baja bien hasta ella, pero la URL pierde el ancla. En el menú del móvil aparece "Example script" en la tabla de contenidos de la página, pero con JavaScript esa sección está oculta: el enlace no lleva a ningún sitio visible.
- **Por qué importa:** se pierde el trabajo sin aviso, y el enlace "permanente" no sirve para compartir.

### C7. En "A category", "Good to know" y "Next step" hablan de otra cosa

- **Gravedad:** Confunde.
- **A quién:** Pablo (cualquier combinación de "A category").
- **Pasó:** el script solo usa un clasificador LightGBM. "Good to know" repite los cinco puntos de las series numéricas: descarga de Chronos-2 (110 MB) de Hugging Face, PyTorch con GPU, "Forecasting many series often? Chronos-2 needs PyTorch", licencias de modelos fundacionales y "foundation models do not [give feature importances]" (`08_categoria_good_to_know.png`). "Next step: compare more model families the same way" lleva a *Comparing forecasting models*, que trata de regresión. Además, "In our test it did not clearly beat the simple rule" se refiere a "our test" en singular, y ninguna tabla de "How we chose these rules" incluye clasificación.
- **Por qué importa:** Pablo se pregunta si tiene que instalar PyTorch o si se ha equivocado de script.

### C8. Las cifras de "How we chose these rules" no cuadran a la primera y piden confianza

- **Gravedad:** Confunde (sobre todo a Luis).
- **Qué se lee:** "26 public datasets", pero la tabla suma 16 + 7 + 10 = 33 filas, y luego "18 of the 33 **cases**". No se explica que un conjunto de datos cuenta dos veces. "LightGBM had the lower error: 2 of 16" y "LightGBM did worse than the baseline: 7 of 16", y aun así se recomienda siempre LightGBM. La razón está en un párrafo posterior ("A model trained with variables that capture what you know... which is why LightGBM is always one of the two candidates"), y la entiende quien ya lo sabía. "Chronos-2 is the small model, on CPU" es jerga. No se menciona que Chronos-2 puede haber visto conjuntos de datos públicos al entrenarse, mientras que la guía *Comparing forecasting models* sí avisa: "Results on public datasets may be optimistic". El enlace `benchmarks/forecaster_finder` da 404 en GitHub (apunta a `main`, como Colab; probablemente esperado).
- **Por qué importa:** Luis va a mirar esta tabla para decidir si se fía, y lo primero que encuentra es una cuenta que no cuadra.

### C9. La búsqueda del sitio no encuentra la página

- **Gravedad:** Confunde.
- **A quién:** Marta, que busca en las guías.
- **Pasos:** en la caja "Search", escribir `which forecaster should I use` o `choose a forecaster`: **"No matching documents"**, con avisos de lunr en la consola (`Invalid query: ... TypeError: Cannot read properties of undefined (reading '_index')`). `which forecaster` da 93 resultados, el primero "Quantile forecasting", y la página no está entre los seis primeros (`10_busqueda.png`). `forecaster finder` da lo mismo. `choose` tampoco la saca arriba. Solo aparece buscando `promotions` (segunda). No estoy seguro de que sea cosa de esta rama: parece que cualquier búsqueda con "a", "I" o "should" falla en todo el sitio.
- **Además:** en la barra lateral de User Guides, "Choosing a model" solo contiene "Comparing forecasting models"; el selector solo está en la tarjeta de la tabla de contenidos.

### C10. La instalación pesa 5,8 GB en Linux y el aviso está escondido debajo del script

- **Gravedad:** Confunde / Molesta.
- **A quién:** todas menos Pablo, en Linux.
- **Pasó:** `pip install skforecast lightgbm chronos-forecasting` descarga PyTorch con CUDA (`nvidia-cublas`, `nvidia-cudnn-cu13`...): el entorno ocupa **5,8 GB**. El aviso, con el comando para la versión solo CPU, está en "Good to know", que va plegado y debajo del script, así que se lee después de instalar. Probé el consejo: `pip install torch --index-url https://download.pytorch.org/whl/cpu` y luego la línea del script dejan `torch 2.15.0+cpu`, 1,4 GB, y el script de Ana da los mismos números. El consejo funciona, pero llega tarde.

### C11. En el móvil, las líneas largas del script se cortan

- **Gravedad:** Confunde (Pablo lee el script en el móvil).
- **Pasos:** a 360 o 400 px, A number / Several / Day / Yes, bajar al script.
- **Pasó:** las líneas se parten, pero los tramos sin espacios se salen por la derecha y quedan ocultos, sin barra de desplazamiento. Se lee `initial_train_size=len(next(iter(serie` y en la línea siguiente `- 3 * steps,`; falta `s.values())))` (`04_movil_codigo_cortado.png`). Midiendo carácter a carácter, a 360 px hay al menos 10 tramos cortados (`(offset=season), y=y, cv=cv, metric`, `min_child_samples=`, `FoundationModel`...) y a 400 px, 5. A partir de 768 px no pasa. Lo que se copia sí está completo.

### C12. Los mensajes para asistentes de IA llevan valores de ejemplo que no se señalan como tales

- **Gravedad:** Confunde (Sara).
- **Qué pasó:** "Copy message" (MCP) copia, por ejemplo, para Sara: "Compare forecasting models for the next periods of the column sales of data/my_data.csv, against a seasonal naive baseline...", y para Marta: "...The columns promotion and price are known in advance..." (`14_oscuro_mensaje_mcp.png`). Nada dice que haya que cambiar `data/my_data.csv`, `sales`, `promotion` ni `price`. "the next periods" no le dice al asistente cuánto predecir.
- **"Copy script as a prompt":** lo leí como lo leería un asistente. Está bien planteado: pide primero una muestra de los datos y el horizonte, solo deja cambiar la carga de datos y `steps`/`season`/`lags`, y enlaza `https://skforecast.org/latest/llms-full.txt` (responde 200). Me llevaría a hacer lo correcto en la carga. Pero para Sara le impide añadir `hour` a las variables de calendario, que se queda en `["day_of_week", "month"]`, y no le pide explicar la MASE ni predecir al final. No lo pegué en un asistente real (ver sección 7).

### C13. Las líneas de skforecast-ai funcionan, pero su tabla sale cortada

- **Gravedad:** Molesta.
- **Pasos:** `pip install "skforecast-ai[foundation]"` en el entorno de Ana (4 s, instala skforecast-ai 0.5.0) y añadir las cuatro líneas al final del script que ya funcionaba.
- **Pasó:** funciona (28 s en total). La salida es:
  ```
     rank                       name  ... mean_absolute_percentage_error n_evaluated
  0     1       ForecasterFoundation  ...                       0.026076          42
  1     2            ForecasterStats  ...                       0.061374          42
  ...
  [5 rows x 9 columns]
  ```
  pandas oculta las columnas del medio y no se ve la MASE, así que no se puede comparar con los números de arriba. El orden cambia (ARIMA queda segundo), y la página ya avisa de ello ("Its ranking can differ").

### M1. "Start over" deja al usuario a media página y sin foco

- **Gravedad:** Molesta.
- **Pasos:** a 1280 × 800, contestar todo, bajar hasta "Start over" (debajo de la pregunta 4) y pulsarlo.
- **Pasó:** el panel de la derecha se encoge y la página se queda en `scrollY` = 927, a la altura de "How we chose these rules". La pregunta 1 queda fuera de la vista (`05_start_over.png`) y el foco del teclado pasa al `body`.

### M2. Los avisos de "copiado" salen lejos del botón pulsado

- **Gravedad:** Molesta.
- **Pasos:** abrir "Using an AI assistant?" y pulsar "Copy script as a prompt".
- **Pasó:** el aviso "Prompt copied" aparece al lado de "Copy script" / "Run it in Google Colab", unos 400 px más arriba, no junto al botón pulsado (`07_prompt_copied_lejos.png`). En pantallas bajas puede quedar fuera de la vista. Lo mismo pasa con "Script copied with sample data: paste it in the second cell of the notebook that just opened." tras pulsar Colab, que además **sustituye en silencio** el portapapeles por un script distinto del que da "Copy script".

### M3. "Was this useful? No" no permite decir qué falló

- **Gravedad:** Molesta.
- **Pasó:** "Yes" y "No" responden igual: "Thanks for telling us." (`09_feedback_no.png`). No hay forma de explicar el problema ni enlace a una issue. No vi ninguna petición de red al pulsar (en local no hay analítica), así que no sé si se registra algo.

### M4. La salida del script asusta a quien empieza

- **Gravedad:** Molesta.
- **Pasó:** barras de tqdm (`0%| | 0/3 [00:00<?, ?it/s]`), `Loading weights: 100% 92/92` y, sobre todo, `Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.` Si al CSV de Ana le faltan días (le quité los 25 de diciembre, que es cuando cierran las tiendas), el script funciona, pero muestra varios recuadros grandes `MissingValuesWarning` ("Some rows of the training matrix have missing values. 4 of 1770 rows are dropped..." y "`last_window` has missing values...") antes de los resultados.

### M5. Con menos datos de los necesarios, el script no avisa

- **Gravedad:** Molesta.
- **Pasó:** "Needs at least 90 values (3 months)". Con los últimos 60 días de Ana, el script corre sin avisar y da `LightGBM: 1.350 / Chronos-2: 0.990`. Con 90 días exactos también corre.

### M6. En escritorio, la recomendación se sale de la vista al contestar la última pregunta

- **Gravedad:** Molesta.
- **Pasó:** a 1280 × 800 y a 1920 × 1080, la pregunta 4 queda debajo del borde inferior. Al bajar para contestarla, el titular ("Compare two forecasters: LightGBM and Chronos-2") se sale por arriba y lo que queda a la vista es el script (`11_recomendacion_fuera_de_vista.png`). La página no se mueve sola; es el scroll del propio usuario, y el panel no se queda fijo.

### M7. El script está en una caja de 520 px con scroll interno y difuminada al final

- **Gravedad:** Molesta.
- **Pasó:** el final del script (el bloque de Chronos-2, o el de la regla en clasificación) solo se ve haciendo scroll dentro de la caja. El borde inferior se difumina, sin ninguna indicación de que hay más. En pantalla, "▸ imports (9 lines)" aparece antes de `# pip install ...`, mientras que en el script copiado la línea de `pip` es la primera.

### Detalles

- **D1.** Estado vacío: dos tarjetas "Forecaster" / "Forecaster" con borde discontinuo que no aportan nada (`01_primera_impresion_pagina.png`).
- **D2.** A 360 px, el nombre se parte como "ForecasterRecursiveMulti / Series" (`12_movil_nombre_partido.png`).
- **D3.** Hay dos nombres para la misma página: "Which forecaster should I use?" (título H1) y "Choose a forecaster" (cabecera, menú, README, "Next" del Quick start).
- **D4.** Textos en inglés poco naturales o ambiguos: "Also if you only know them up to today" (¿"Even if"? ¿"Also choose this if"?); "the step one season before"; "Runs as it is: tested on sample data every time it changes" (¿qué es "it"?); "`refit` does not apply: it only sees the data before" (comentario del script, críptico); "The category one season before (`y.shift(season)`)" (código en una tabla para usuarios).
- **D5.** Sin JavaScript, el mensaje "The interactive finder needs JavaScript. The rules and an example script are below." aparece en una columna estrecha, de unos 300 px, dentro de una caja de ancho completo (`13_sin_javascript.png`).
- **D6.** En el móvil, el párrafo de introducción va justificado y quedan huecos grandes entre palabras ("Answer   four   questions"). Probablemente es el estilo de todo el sitio.
- **D7.** Si se cambia de "A number" a "A category" y se vuelve a "A number", la respuesta a "How many time series?" se pierde y hay que darla otra vez.
- **D8.** Una URL con respuestas inválidas (`#answers=foo-bar`), abierta en la misma pestaña, no avisa de nada: mantiene el estado anterior.
- **D9.** El README enlaza `https://skforecast.org/latest/quick-start/choose-a-forecaster.html`, que hoy da 404. Supongo que es esperado hasta publicar la versión, pero lo anoto.
- **D10.** La píldora "Back to top" de Material tapa líneas del script al subir (`08_categoria_good_to_know.png`, arriba).
- **D11.** Con las flechas del teclado, moverse por la pregunta 1 cambia la respuesta al momento (es el comportamiento normal de un grupo de botones de opción), así que solo con mirar la otra opción aparece o desaparece la pregunta 2.

## 4. El recorrido de cada persona

**Ana (portada, diario, una serie).** Encontró el enlace "Find out in four questions →" sin hacer scroll, en pocos segundos. Contestó sin dudas: "exogenous variables" está explicado con ejemplos. Pulsó "Copy script" y se encontró con `NameError` (B1). Siguiendo el comentario, `-41` (B2). Solo funcionó cuando escribí yo `pd.read_csv("...", index_col="date", parse_dates=True)`, cosa que ella no sabría hacer. Desde que llegó hasta ver un resultado: unos 2 minutos para contestar, 1 min 16 s de instalación, tres intentos y 17 s de ejecución. Una principiante sola probablemente se habría quedado en el `NameError`. Con el resultado delante (`0.869 / 0.547 / 0.219`) no sabe si es bueno ni cómo predecir sus ventas (C1). No confiaría en él, porque no lo entiende.

**Luis (Intro, mensual, una serie).** El aviso "Not sure which forecaster to use?" está arriba del todo en la Intro y lo vio a la primera. La recomendación añade ARIMA como referencia, que es lo que él esperaba. Se escribió la carga sin problemas y le funcionó a la primera (31 s la primera vez, 8 s la segunda): `0.876 / 0.874 / 0.821 / 0.645`. Con fechas a fin de mes habría chocado con B3. Antes de fiarse revisaría "How we chose these rules", y la cuenta 26/33 y la falta de un aviso sobre conjuntos de datos públicos y Chronos-2 le harían dudar (C8). Llega al resultado, pero no se fía del todo del estudio.

**Marta (guías, 30 series diarias, festivos y promociones).** La búsqueda no le ayudó (C9). La encontró en la tarjeta "Choosing a model" de la tabla de contenidos de User Guides. Contestó bien (Several / Day / Yes), y las etiquetas "uses your known variables" le confirman que sus festivos cuentan. El script supone una tabla ancha y un único `exog`. Su CSV es de formato largo y sus promociones son distintas en cada tienda: primer intento con error (B2); funcionó después de pivotar y de pasar `exog` como diccionario, dos cosas que la página no explica (C3). Tiempo de ejecución: 18 s la primera vez y 13 s la segunda. Ve solo la media de las 30 tiendas. Llegó al resultado con ayuda externa.

**Pablo (móvil y luego portátil, categoría horaria).** En el móvil, la tarjeta "A category" de la portada le deja ya en la pregunta 2. Las respuestas se pliegan en etiquetas ("A number ✎") y se ve muy bien (`15_movil_a_medias_ok.png`). Al leer el script en el móvil, faltan trozos de código (C11). Para pasar al portátil, la URL con `#answers=category-_-hourly-no` conserva las respuestas en otra pestaña o en otro navegador (comprobado). La instalación es ligera (7 s, 500 MB) y el script corre en 5 s: `Classifier: 0.500 / Rule: 0.889`. Sus lecturas no se usan y la prueba de 72 horas solo tiene dos clases (C4). "Good to know" le habla de Chronos-2, que no usa (C7). Llega al resultado rápido, pero con una conclusión débil.

**Sara (cada 15 minutos, prefiere un asistente).** "Something else" ("15 minutes, a year...") deja claro qué contestar. El script es el diario con `# <- edit` y sin indicaciones; ejecutado tal cual da números sin sentido (C2). Ajustando a 96 obtiene MASE de 10 a 19, que no sabe interpretar (C1). Por el lado del asistente: el mensaje MCP lleva valores de ejemplo sin marcar (C12). "Copy script as a prompt" está bien redactado y llevaría a un asistente a pedirle una muestra y a no tocar los modelos, pero el asistente no podría añadir la hora a las variables de calendario. Llega al resultado, pero no sabe si sirve.

## 5. Tiempos medidos y entorno

**Entorno:** Linux 6.18 (contenedor en la nube, 4 CPU, sin GPU), Python 3.12.3 del sistema, entornos creados con `python3 -m venv` e instalados con `pip` normal. Versiones que instaló `pip`: skforecast **0.27.0** (PyPI), lightgbm 4.7.0, chronos-forecasting 2.3.2, torch 2.15.0 (con CUDA; o `2.15.0+cpu` con el consejo de C10), transformers 5.24.1, pandas **2.3.3**, numpy 2.5.5, scikit-learn 1.9.1, statsmodels 0.15.0 (con `skforecast[stats]`) y skforecast-ai 0.5.0. Los scripts se ejecutaron contra el skforecast de PyPI, no contra el código de la rama (comprobado con `skforecast.__file__`). Chromium de Playwright, a escala 1. Construir el sitio: 61 s.

| Caso | Instalar | 1.ª ejecución | 2.ª ejecución | Resultado |
|---|---|---|---|---|
| Ana (`pip install skforecast lightgbm chronos-forecasting`, caché de pip vacía) | 1 min 16 s, 5,8 GB | 17,1 s (incluye la descarga de Chronos-2, 107 MB) | 6,8 s | 0.869 / 0.547 / 0.219 |
| Ana + skforecast-ai (`pip install "skforecast-ai[foundation]"`) | 4 s | 28,1 s (script + `compare`) | (no repetida) | ranking de 5 modelos |
| Ana con torch solo CPU (consejo de "Good to know") | (no medido con fiabilidad: caché de pip caliente) | 16,3 s | (no repetida) | mismos números, 1,4 GB |
| Luis (`"skforecast[stats]" lightgbm chronos-forecasting`, caché caliente) | 50 s, 5,9 GB | 30,7 s (pesos de Chronos ya en caché) | 7,8 s | 0.876 / 0.874 / 0.821 / 0.645 |
| Marta (como Ana, caché caliente) | 51 s | 18,3 s | 13,0 s | 1.003 / 0.771 / 0.697 |
| Pablo (`skforecast lightgbm`, caché caliente) | 7 s, 500 MB | 5,1 s | 2,0 s | 0.500 / 0.889 |
| Sara (entorno de Ana) | (el de Ana) | 7,8 s tal cual; 10,7 s con 96 | (no repetida) | 9.101 / 5.179 / 3.597; con 96: 19.166 / 16.341 / 10.490 |

Las instalaciones después de la primera aprovecharon la caché de `pip`, y las primeras ejecuciones después de la de Ana, los pesos de Chronos-2 ya descargados. En un equipo sin nada previo, cada una tardaría más.

**Datos usados** (todos de `skforecast.datasets.fetch_dataset`, guardados como CSV):
- Ana: `store_sales`, tienda 1, ventas totales diarias, 2013 a 2017.
- Luis: `h2o`, mensual, 1991 a 2008.
- Marta: `m5`, 30 series (tienda × categoría), diaria, con `holiday` y `promo` (SNAP).
- Pablo: `ett_m2` agregado por horas, con `level` por terciles.
- Sara: `ett_m2`, cada 15 minutos.

## 6. Lo que funcionó bien y no tocaría

- **Entradas:** portada (enlace encima del pliegue y tarjeta con "A number" / "A category" que entra con la respuesta marcada), aviso al principio de la Intro, enlace en Quick start, tarjeta en la tabla de contenidos de User Guides, FAQ, ejemplos en inglés, español y chino (estos dos dicen "(en inglés)" / "（英文页面）") y README. Todas llevan a la página correcta.
- **Preguntas:** las cuatro preguntas y sus ejemplos se entienden bien ("Things that affect the values and that you know for future dates. skforecast calls them exogenous variables.", "A time series is one column of values recorded over time"). La aclaración de "No" sobre las lecturas de sensores evita un error típico.
- **Respuesta progresiva:** la recomendación se va completando con cada respuesta y el script indica "So far it will compare..." mientras faltan preguntas.
- **URL:** las respuestas se guardan en la URL: sobreviven a recargar, a ir a una guía y volver con atrás, y a abrir la URL en otro navegador sin cookies. Se puede contestar en cualquier orden y una URL a medias también funciona. Contestar no añade entradas al historial: un solo "atrás" sale de la página.
- **Teclado:** son grupos de botones de opción de verdad, se puede contestar con Tab, flechas y espacio, el foco se ve (anillo naranja) y el orden tiene sentido. Copié el script sin ratón.
- **Móvil:** las respuestas se pliegan en etiquetas editables y no hay scroll horizontal en ningún ancho (360 a 1920 px).
- **Tema oscuro:** sin textos ilegibles ni colores que no cambien.
- **Sin JavaScript:** queda una página útil, con un aviso, las reglas y un script de ejemplo.
- **Scripts:** una vez cargados los datos, corren con skforecast 0.27.0 de PyPI en todas las combinaciones que probé, y en pocos segundos.
- **"Copy script as a prompt":** el texto está bien pensado: pide una muestra y limita los cambios.
- **"Good to know":** el consejo de PyTorch solo CPU funciona.

## 7. Lo que no pude probar

- **Un asistente de IA real:** no pegué "Copy script as a prompt" ni el mensaje MCP en un asistente, ni conecté el servidor MCP de skforecast-ai. Los leí como lo haría un asistente.
- **Google Colab:** el notebook no existe en `main` (404 esperado). Comprobé qué se copia y el mensaje que aparece, no el notebook.
- **Windows y macOS:** no pude medir la instalación en esos sistemas, donde PyTorch no trae CUDA.
- **Móvil real:** usé emulación de Chromium (360 y 400 px, y el perfil "Pixel 7"). Tampoco probé Safari ni un lector de pantalla.
- **Comentarios y analítica:** no sé si "Was this useful?" envía algo, porque en local no hay analítica.
- **pandas 3:** `pip` instaló pandas 2.3.3, así que no probé los scripts con pandas 3.
- **Hugging Face bloqueado:** no probé qué pasa con la red bloqueada (lo que avisa "Good to know").
- **Las 36 combinaciones:** las recorrí todas en el navegador y revisé texto, script y datos mínimos, pero solo ejecuté seis: diario una serie, mensual una serie, diario varias series con exógenas, categoría horaria, "Something else" una serie y semanal una serie (esta solo para B3).
- **Cookies:** la búsqueda y el aviso de cookies se comportan como en el resto del sitio; no comparé con el sitio publicado.
