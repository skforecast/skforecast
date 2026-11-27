# Revisión independiente del selector de forecaster

- Fecha: 2026-11-27. Revisa `PLAN.md` y `benchmark/` en el commit `407746d` (versión 1), más
  el informe visual. **Donde esta revisión y la versión 1 de `PLAN.md` difieran, vale la
  revisión.** `PLAN.md` ya está reescrito (versión 2) con lo que sale de aquí.
- Entorno de todas las cifras: Python 3.12.3, skforecast 0.28.0, numpy 2.5.5, pandas 3.1.1,
  scikit-learn 1.9.1, scipy 1.18.1, numba 0.68.1, LightGBM 4.7.0, PyTorch 2.15.0 (CPU),
  chronos-forecasting 2.3.2, transformers 5.24.1. Máquina de 4 núcleos, sin GPU.
- **Segunda pasada (mismo día): los dos fallos de `Ets` están arreglados en esta rama
  (`5f020a0`), el estudio se volvió a medir con el arreglo y el prototipo del selector está
  reescrito con el diseño propuesto. Lo que cambia respecto al texto de abajo está en la
  sección 6 y en las líneas "Estado" de H3, H4 y H7.**
- **Tercera pasada (mismo día), tras las respuestas a D1 a D5: estudio ampliado a 26
  datasets, ARIMA como referencia estadística y un resultado más simple. Está en la sección
  7 y corrige H2: con más datasets, Chronos-2 va por delante en los tres tipos de datos.**
- Etiquetas: **[Medido]** ejecutado en esta revisión, **[Código]** verificado leyendo el código
  de skforecast 0.28.0, **[Plausible]** razonado, sin prueba concluyente.

## 1. Veredicto

El trabajo anterior es reproducible y su idea central se sostiene: dos candidatos (Chronos-2 y
LightGBM) y un baseline estacional. Pero el estudio comparaba un ML congelado con dos
familias que usan todo lo anterior a cada bloque, y de ahí salen varias de sus cifras y dos de
sus reglas. Con un protocolo justo:

1. **Chronos-2 sigue por delante, con menos margen**: menos error que LightGBM en 26 de 41
   ejecuciones (antes 33) y en 8 de 12 datasets (antes 10). Mejora al naive en 40 de 41, y la
   versión entrenada solo con datos sintéticos, en 15 de 15: no es un efecto de haber visto
   los datasets.
2. **El orden no lo decide la prioridad del usuario, lo decide el tipo de datos**, justo lo que
   la versión 1 descartó. Una serie sin variables conocidas a futuro: Chronos-2 gana en 19 de
   22 ejecuciones y LightGBM queda peor que el naive en 11. Con variables conocidas a futuro
   o con varias series, empate. Las preguntas 2 y 4 sí cambian la recomendación.
3. **El fragmento de código no es lo que se midió** y es una mala primera experiencia: su
   LightGBM queda peor que el naive en 17 de 38 ejecuciones, 72 de los 100 fragmentos emiten
   avisos y el mensual necesita 5 años de datos para ejecutarse.
4. **La regla de AutoETS solo está respaldada en datos mensuales** (3 datasets). En semanal,
   `Ets(m=52)` ajusta sin avisar un modelo sin estacionalidad.
5. Apareció un fallo de la librería: `backtesting_stats` con AutoETS y el valor por defecto
   `freeze_params=True` devuelve predicciones del orden de -1e11 en algunas series
   (fila `BUG-ets-freeze-params` del roadmap).

Recomendación: seguir adelante, con el formato cambiado (un selector de una pantalla en vez
de un test por pasos), las reglas de la sección 3.2 y el estudio de referencia sustituido por
el de protocolo justo (`benchmark/review.py`).

## 2. Qué se hizo

| Paso | Script | Resultado |
|:--|:--|:--|
| Reproducir el estudio original | `benchmark.py` | Las 279 filas de `results/` salen idénticas (diferencia relativa máxima 0) |
| Controles de método, mismos casos y bloques | `review.py` | `results/review/*.csv`, y el error por serie y por bloque en `results/review/detail/` |
| Recuentos por ejecución, por dataset y por tipo de datos | `recount.py` | `results/review/recount_runs.csv`, `recount_summary.csv` |
| Otros sitios de `bdg2`, otras series de `m4_hourly`, clasificación | `review.py --extra` | `results/review/extra_*.csv` |
| Horizonte cuatro veces más corto | `review.py --horizon` | `results/review/extra_horizon.csv` |
| Los 100 fragmentos | `build.py`, `check_snippets.py` | Sin errores; avisos e historia mínima en `results/review/snippet_*.csv` |
| Fallo de AutoETS | `repro_ets_freeze_params.py` | Reproducción en 20 líneas |

Lo que añade `review.py` a cada caso:

- ML reentrenado en cada bloque con ventana creciente (`refit=True`,
  `fixed_train_size=False`), junto al ML congelado del original.
- `LightGBM+`: más lags (28 en diario más el 364 si hay dos años, 24 más 48 a 336 en horario,
  12 más el 24 en mensual), dos medias móviles, 500 árboles con `learning_rate=0.03`,
  submuestreo y regularización suaves. Sin ajustar en test.
- `LightGBM snippet`: lo que escribe hoy `finder_tree.py` (`LGBMRegressor` por defecto, sin
  escalar).
- La variante de ML elegida en un periodo de validación sacado del final del entrenamiento
  (hasta 3 bloques), entre las 4 originales y entre 6 (con `LightGBM+`).
- Chronos-2 small, base (`amazon/chronos-2`, 120M) y synth (`autogluon/chronos-2-synth`,
  120M, entrenado solo con datos sintéticos según su ficha).
- AutoARIMA en el caso semanal y AutoETS por serie en el trimestral multiserie.

## 3. Respuesta a las preguntas de la revisión

### 3.1 Las seis conclusiones de la sesión anterior

| # | Conclusión | Veredicto | Con el protocolo justo |
|--:|:--|:--|:--|
| 1 | Siempre Chronos-2 y LightGBM más un baseline; el orden lo decide solo la prioridad | **La alineación se sostiene; la regla del orden, no** | El tipo de datos separa los casos (H2). "26 de 33" y "7 de 8" pasan a 21 de 33 y 5 de 8; por dataset, 8 de 12 |
| 2 | LightGBM por defecto, peor que el naive en 17 de 38 | **En parte artefacto** | 11 de 41 con reentrenamiento, y las 11 son series sueltas sin variables conocidas a futuro. El LightGBM del fragmento sí queda en 17 de 38 |
| 3 | Frente a los estadísticos decide la frecuencia, no la longitud; ganaron en 8 de 15 | **Solo en mensual** | 6 de 12 en mensual (3 datasets, empate con Chronos-2). Semanal: 1 dataset y un AutoETS sin estacionalidad. Trimestral de una serie: ningún caso |
| 4 | La pregunta de la historia no cambia el orden | **Se sostiene** | Dentro de cada tipo de datos, el orden no cambia con la historia. Tampoco con el horizonte (H13) |
| 5 | Las variables solo hasta hoy no mejoraron ningún modelo | **Falso como afirmación general** | Con Chronos-2 base mejoran un 14 % en un caso; con synth, un 6 % en otro; con small, nada. Tres casos: no hay regla |
| 6 | Cinco preguntas | **Cuatro, y en una pantalla** | Ver 3.3 |

### 3.2 Reglas que sí respalda la evidencia

| Respuestas | Mensaje | Respaldo |
|:--|:--|:--|
| Una categoría | Clasificador más la regla estacional como baseline | Por capacidad (es lo que hay). Un dataset: el clasificador no supera a la regla en balanced accuracy (H15) |
| Un número, una serie, nada conocido a futuro | Chronos-2 primero; LightGBM como aspirante que hay que trabajar; baseline | Fuerte: 19 de 22 ejecuciones, 7 de 8 datasets; LightGBM peor que el naive en 11 de 22 |
| Un número, una serie, con variables conocidas a futuro | Los dos, sin favorito; baseline | Débil: LightGBM por delante en 9 de 11 ejecuciones, pero son 2 datasets y la meteorología es la observada, no un pronóstico |
| Un número, varias series | Los dos, sin favorito; baseline por serie | Medio: Chronos-2 por delante en 5 de 8 ejecuciones y en 2 de 4 datasets (3 con toda la historia), con márgenes pequeños; en `bdg2`, en los 9 sitios probados |
| Además, una serie mensual (y trimestral, por extensión) | AutoETS como referencia estadística | Medio en mensual (3 datasets); trimestral sin medir en una serie; semanal y diario, no |

Tabla de apoyo (LightGBM reentrenado frente a Chronos-2 small, MASE relativo al naive
estacional; `recount.py`):

| Tipo de datos | Ejecuciones (datasets) | LightGBM menor que Chronos-2 | `LightGBM+` menor que Chronos-2 | LightGBM peor que el naive | Chronos-2 peor que el naive | Mediana Chronos-2 / LightGBM / `LightGBM+` |
|:--|--:|--:|--:|--:|--:|:--|
| Una serie, nada conocido a futuro | 22 (8) | 3 | 1 de 19 | 11 | 1 | 0,69 / 1,00 / 1,06 |
| Una serie, variables conocidas a futuro | 11 (2) | 9 | 6 | 0 | 0 | 0,67 / 0,66 / 0,65 |
| Varias series | 8 (4) | 3 | 4 | 0 | 0 | 0,79 / 0,82 / 0,82 |

### 3.3 Diseño del test

- **¿Son claras las preguntas?** Sí, salvo dos opciones que meten en un mismo saco
  frecuencias con constantes distintas ("Every hour, or more often" y "Every quarter, or
  every year": H8), y la pregunta 5, que pide una preferencia sin decir qué se pierde.
- **¿Siguen el orden de crear un forecaster?** Sí: objetivo, forma de los datos, frecuencia,
  exógenas.
- **¿Sobra alguna?** La de prioridad (H9): con los dos candidatos siempre en el código y el
  resultado ordenado por error, solo cambia la numeración de los comentarios y añade
  `get_feature_importances()`. La de frecuencia se queda: es la más fácil de contestar, hace
  falta para que el código se ejecute sin editar y decide la referencia estadística. La
  alternativa más simple (tres preguntas y una constante `SEASON = 7` que el usuario edita)
  ahorra una fila y pierde "se ejecuta tal cual" y la regla de AutoETS; no compensa.
- **¿Test interactivo o página estática?** Las dos cosas, en ese orden, y el interactivo como
  selector, no como test (H10).
- **¿Los fragmentos son un punto de partida correcto?** Se ejecutan, pero no son lo medido,
  avisan en 72 de 100 casos y piden demasiada historia (H3, H7).

### 3.4 Posicionamiento

"Precisión → Chronos-2 primero" es coherente con la evidencia en un caso (una serie sin
variables conocidas a futuro) y no en los otros dos, donde hay empate. Decirlo así no choca
con el posicionamiento de la librería (ML y Foundation primero, estadísticos como
referencia): los dos candidatos son siempre ML y Foundation, y el mensaje honesto es que
Chronos-2 da una cifra fuerte sin configurar nada y que LightGBM es el modelo que se puede
mejorar, explicar y desplegar barato. Lo que el estudio no puede decir es cuánto se acerca
un LightGBM bien trabajado, y eso es lo que habría que medir para ir más lejos (H16).

### 3.5 Reglas de skforecast-ai

El acceso al repositorio `skforecast/skforecast-ai` fue denegado en esta sesión. Se revisó el
paquete público `skforecast-ai` 0.5.0 de PyPI (leído, no ejecutado). Ver H12.

## 4. Hallazgos, de más a menos graves

### Críticos

**H1. Solo ML estaba congelado.** [Código] [Medido]

- Evidencia en el código: `backtesting_stats` cambia `refit` a `True` para todo lo que no sea
  `Sarimax` (`_validation.py:2722-2736`, con un `IgnoredArgumentWarning` que el estudio
  silencia). AutoETS y AutoARIMA se reajustaron en cada bloque, y Chronos-2 recibe todo lo
  anterior. El único modelo que predice el último bloque con lo aprendido antes del primero
  es ML: hasta 5 años después en los casos mensuales (5 bloques de 12 meses).
- Efecto (41 ejecuciones): Chronos-2 con menos error que LightGBM, de 33 a 26. LightGBM peor
  que el naive, de 17 a 11. Mediana del MASE relativo de LightGBM, de 0,944 a 0,913; la
  distancia media a Chronos-2, de 0,38 a 0,16.
- Casos: `expenditures_australia` 2,10 → 0,93; `m4_hourly` 0,76 → 0,50; `bike_sharing`
  diario 0,78 → 0,68 (pasa a Chronos-2, 0,69); `fuel` 1,08 → 0,93.
- Lo mismo pasa en la guía publicada *Comparing forecasting models*: dice "The models are
  trained once on the training set, without refitting between folds", pero sus AutoARIMA y
  AutoETS sí se reajustan (usa `suppress_warnings=True`). Su test es corto (2 bloques), así
  que el efecto allí es pequeño; la frase es incorrecta.
- Propuesta: el estudio de referencia es `review.py` (ML reentrenado en cada bloque). Las
  cifras de la versión 1 no se citan en ninguna página. Corregir la frase de la guía.

**H2. El orden depende del tipo de datos, no de la prioridad.** [Medido]

- Evidencia: tabla de 3.2. Con una serie y nada conocido a futuro, LightGBM reentrenado gana
  en 3 de 22 ejecuciones y queda peor que el naive en 11; `LightGBM+` gana en 1 de 19. Con
  variables conocidas a futuro, LightGBM gana en 9 de 11. Con varias series, 3 de 8 (4 con
  `LightGBM+`).
- La versión 1 concluyó lo contrario ("las preguntas 2 a 4 cambian el código, no las
  familias recomendadas") porque el ML congelado perdía en casi todo.
- Límite: la celda con variables conocidas a futuro son 2 datasets (`vic_electricity`,
  `bike_sharing`); en `bike_sharing` horario y en `bdg2` (varias series, con temperatura)
  gana Chronos-2. La regla que sale es "sin favorito", no "ML primero".
- **Estado (tercera pasada)**: corregido por el estudio ampliado. Con 7 y 10 datasets en
  lugar de 2 y 4, Chronos-2 va por delante también con variables conocidas a futuro (5 de
  7) y con varias series (8 de 10). El límite que se señalaba aquí era el relevante: la
  conclusión "sin favorito" no se sostiene. Ver 7.1.
- Propuesta: las reglas de 3.2. Cada regla lleva en la página su respaldo (ejecuciones y
  datasets), y la débil se presenta como tal.

**H3. El fragmento no es la configuración medida y es una mala primera experiencia.**
[Medido] [Código]

- El estudio midió LightGBM con 300 árboles, `learning_rate=0.05`, 15 hojas,
  `StandardScaler` y horizonte de 28 días. El fragmento escribe `LGBMRegressor` por defecto,
  sin escalar, con horizonte de 14 días y `TimeSeriesFold` sin `refit` (congelado).
- Medido tal como lo escribe el fragmento: peor que el naive en 17 de 38 ejecuciones (15 con
  reentrenamiento). En `expenditures_australia`, de 3,0 a 4,1 veces el error del naive. En
  varias series sin escalar, 66 veces el error del naive en el sitio Bear de `bdg2`.
- Consecuencia de producto: en casi la mitad de los casos, la primera ejecución del modelo
  estrella de una librería "machine learning-first" pierde contra el naive, por el
  fragmento y no por el modelo.
- 72 de los 100 fragmentos emiten avisos que `build.py` oculta con
  `warnings.filterwarnings("ignore")`: 27, el `IgnoredArgumentWarning` de `backtesting_stats`
  (baja frecuencia, una serie); 45, el `InputTypeWarning` de pasar un DataFrame como
  `series` (varias series).
- Historia mínima para que se ejecute (`check_snippets.py`): 21 días en horario, 150 días en
  diario, 2 años en semanal (4 en clasificación), **5 años en mensual** y 5 en trimestral.
  Con menos, `ValueError` de `initial_train_size`.
- `forecaster.get_feature_importances()` va como expresión suelta: en un script no muestra
  nada.
- **Estado (segunda pasada)**: hecho en el prototipo. Los 36 fragmentos nuevos se ejecutan
  sin avisos, reentrenan cada modelo y funcionan con la historia mínima; su LightGBM, medido
  tal como se escribe, queda peor que el naive en 14 de 38 ejecuciones (todas con una serie
  y nada conocido a futuro) y nunca con varias series. Detalle en 6.3.
- Propuesta: (a) el bloque de ML del fragmento y el del estudio salen de la misma función,
  de modo que lo que se copia es lo que se midió; (b) `refit=True` y ventana creciente; (c)
  CI con los avisos como errores; (d) bloques de test que quepan en 3 años mensuales, 2 años
  semanales y 3 meses diarios, y la historia mínima escrita en la página; (e) una línea de
  cabecera: valores de ejemplo para que el script funcione, no ajustados, con el enlace a la
  guía.

**H4. AutoETS: referencia semanal sin estacionalidad y un fallo de la librería.**
[Código] [Medido]

- `Ets` no admite modelos estacionales con `m > 24` (`_ets_base.py:1228-1245`). Por
  `ForecasterStats`, `Ets(m=52)` no avisa: sobre una serie con estacionalidad anual clara
  selecciona `Ets(AAdN)`. En `bicimad` semanal quedó en 1,04, 1,04 y 1,05 veces el naive;
  AutoARIMA con `m=52`, en 0,93 (8 s), y Chronos-2, en 0,59. La versión 1 lo contaba dentro
  de "AutoETS o AutoARIMA ganaron en 8 de 15" y lo ponía en el fragmento semanal.
- Fallo nuevo: `backtesting_stats` con un `Ets` automático y `freeze_params=True` (valor por
  defecto) devuelve predicciones del orden de -1e11 desde el segundo bloque en 8 de las 304
  series de `australia_tourism`; con `freeze_params=False`, las mismas series dan
  predicciones normales. `best_params_` incluye `alpha`, `beta`, `gamma` y `phi`, y
  `freeze_params` los fija en los reajustes. Reproducción: `repro_ets_freeze_params.py`.
- Todas las cifras de AutoETS y AutoARIMA de la versión 1 y de esta revisión están medidas
  con `freeze_params=True`: modelo y parámetros elegidos en el primer ajuste.
- **Estado (segunda pasada)**: arreglados los dos (6.1). La causa no era `freeze_params`,
  sino que la verosimilitud aceptaba, y premiaba, modelos multiplicativos con nivel o
  predicciones no positivos.
- Propuesta: quitar AutoETS del caso semanal; filas `BUG-ets-freeze-params` (P1) y
  `QW-ets-m-over-24-silent` (P2) en el roadmap; el fragmento con AutoETS no se publica hasta
  cerrar la primera.

### Altos

**H5. Las ejecuciones no son independientes.** [Medido]

- 13 de las 33 ejecuciones de una serie son de dos datasets (`vic_electricity`, 6;
  `bike_sharing`, 7). Son 12 datasets de regresión, no 13 (el decimotercero es el de
  clasificación).
- Un voto por dataset (media de sus ejecuciones): Chronos-2 por delante de LightGBM
  congelado en 10 de 12; del reentrenado, en 8 de 12; de `LightGBM+`, en 8 de 11; de la
  mejor variante de ML elegida en test, en 6 de 12. Solo con toda la historia: 11, 10, 7 de
  11 y 9. Chronos-2 mejora al naive en 12 de 12; LightGBM reentrenado, en 9.
- Por bloques, con toda la historia: en `bike_sharing` horario Chronos-2 gana 16 de 28
  bloques, y en `ett`, 14 de 28. Diferencias de ese tamaño no distinguen dos modelos.
- Propuesta: toda cifra publicada se da por dataset, con el número de datasets al lado.

**H6. "La mejor variante de ML" no es alcanzable.** [Medido]

- Elegida en test entre 4 (lo que hacía `evidence.py`): mediana 0,849 en las 41 ejecuciones.
  Elegida en validación entre las mismas 4: 0,903. LightGBM reentrenado, sin elegir: 0,913.
- Elegir en una ventana de validación corta no mejora a no elegir, y frente a esa elección
  Chronos-2 gana en 31 de 41 (frente al LightGBM reentrenado, 26).
- Propuesta: quitar "frente a la mejor variante de ML" de todas las cifras. La comparación
  honrada es con una configuración fijada antes o elegida en validación.

**H7. La regla de la frecuencia descansa en tres datasets mensuales.** [Medido]

- Mensual (12 ejecuciones, 3 datasets): AutoETS o AutoARIMA con menos error que Chronos-2 en
  6; que LightGBM reentrenado, en 12; AutoETS no quedó nunca peor que el naive (AutoARIMA,
  en 5). Por dataset: 0,61 frente a 0,61, 0,75 frente a 0,71 y 0,90 frente a 0,88 (AutoETS
  frente a Chronos-2). Es un empate.
- Semanal: un dataset, y el AutoETS no es estacional (H4).
- Trimestral de una serie: ningún caso. En `australia_tourism` (304 series), AutoETS por
  serie: mediana del error relativo 0,85 frente a 0,88 de Chronos-2, mejor que Chronos-2 en
  el 51 % de las series, 45 s frente a 2,4 s, y el fallo de H4 en 8 series.
- Diario (13 ejecuciones): AutoETS no ganó nunca a Chronos-2 y quedó peor que el naive en 9.
- **Estado (segunda pasada)**: con el arreglo, AutoETS por serie en `australia_tourism`
  queda en 0,859 frente al naive (Chronos-2, 0,866; LightGBM, 0,925), mejor que el naive en
  el 74 % de las series y que Chronos-2 en el 52 %. El trimestral ya tiene respaldo medido
  (un dataset, 304 series): empate con Chronos-2, como en mensual.
- Propuesta: AutoETS como referencia en mensual y, por extensión declarada, en trimestral.
  No en semanal.

**H8. Opciones de frecuencia con constantes equivocadas.** [Código]

- "Every hour, or more often: every 15 or 30 minutes" lleva a `freq="h"`, lags 24 y baseline
  de 24: con datos de 15 minutos la estación es 96. "Every quarter, or every year" lleva a
  `freq="QS"` y baseline de 4.
- Propuesta: cinco opciones limpias y una sexta, "Something else", que da el código con las
  constantes señaladas para editar y enlaza a la guía.

**H9. La pregunta de la prioridad no cambia nada medible.** [Medido] [Plausible]

- La precisión no ordena los candidatos (H2). El coste se midió como segundos del backtest:
  con varias series, Chronos-2 tarda de 2,6 a 9,5 veces más que LightGBM congelado, pero
  solo de 1,2 a 2,0 veces más que LightGBM reentrenado (medianas 0,7 s, 1,9 s y 3,2 s). El
  tiempo de un backtest no es el coste de producción (dependencia de PyTorch, GPU a escala,
  inferencia frente a entrenamiento), que no está medido.
- La explicabilidad sí distingue, por capacidad: solo ML da importancias.
- Propuesta: quitar la pregunta y poner dos notas fijas en el resultado ("Need to explain
  the forecast?", "Forecasting many series often?"). Decisión del usuario (D2 de `PLAN.md`).

**H10. Un test por pasos no es el formato que corresponde.** [Plausible]

- Con 4 preguntas cerradas y un resultado que es sobre todo código, el patrón es el
  selector de PyTorch "Start locally" (que la propia versión 1 cita): todas las opciones a la
  vista, una selección por defecto y el código que cambia con cada clic. Un test con
  "siguiente" y revelado final promete un diagnóstico; aquí tres de cada cuatro respuestas
  cambian constantes del código, y quien repita el test lo verá.
- Producto: el valor diferencial frente a una tabla es el código que se ejecuta con sus
  datos y el prompt, no la recomendación. El selector da ese valor desde el primer clic, sin
  pasos que abandonar.
- Marketing: un bloque interactivo con código vivo es reutilizable en la portada y enlaza de
  forma natural con skforecast-ai, que ejecuta esa misma comparación sobre los datos del
  usuario. La página estática es lo que indexan los buscadores y lo que leen los asistentes
  (`llms-full.txt`).
- Propuesta: primero la página estática generada (reglas, respaldo y fragmentos), después el
  selector sobre la misma página. Nombre: "Not sure where to start?" se queda como gancho;
  el bloque se presenta como "Build your first comparison".

**H11. Las 14 correcciones de la sección 5: 13 se sostienen, con retoques.** [Código]

| # | Estado | Nota |
|--:|:--|:--|
| 1 | Correcta; la corrección propuesta cambia | Líneas de la skill, bien. `llms-base.txt`: la frase está en la línea **724**, no 720. La corrección pasa a ser las reglas de 3.2, no "ordenados por prioridad" |
| 2 | Correcta, con otra cifra | `SKILL.md:170`. El "17 de 38" es del ML congelado; con reentrenamiento, 11 de 41. No añadir el consejo de `differentiation=1`: está fuera del alcance y medido da resultados mixtos |
| 3 | Correcta en mensual | `SKILL.md:49, 130`, `statistical-models/SKILL.md:15`. Redactar con H7: mensual sí, semanal no |
| 4 | Correcta | `list_adapters()`: 11 adaptadores, 5 con `commercial_use_restricted=True` (TimesFM 3.0, Moirai-2, TabPFN-TS, TS-ICL, Timestral) |
| 5 | Correcta en el código; la frase medida, no | `SKILL.md:146`. Chronos-2 admite covariables solo pasadas (`supports_past_only_covariates=True`). "Rarely help" no está medido (H14): decir que no hay una ganancia consistente |
| 6 | Correcta | `SKILL.md:36-41, 58, 69`; etiqueta `forecasting_scope='single-series'` |
| 7 | Correcta | `ForecasterEquivalentDate` y `ForecasterStats` solo aceptan `pandas.Series` |
| 8 | Correcta | Regla en `.claude/rules/docs.md:35-39` |
| 9 | Correcta, incompleta | `predict_interval` de `ForecasterStats` está en `_forecaster_stats.py:1003` (no 1004); `get_feature_importances`, en `:1308`. **La alternativa "generar la matriz desde `get_tags()`" no es posible hoy**: `ForecasterStats` y `ForecasterFoundation` no tienen `get_tags()` |
| 10 | Correcta | `_auto_arima.py:889-890` |
| 11 | Correcta | `statistical-models/SKILL.md:24`; `interval_method` no existe en `ForecasterStats` ni en `backtesting_stats` |
| 12 | Correcta | `ForecasterDirectMultiVariate.differentiation: int \| None` |
| 13 | Correcta | La ruta real es `docs/introduction-forecasting/introduction-forecasting.md:168-184` |
| 14 | Correcta | `SKILL.md:171, 175, 50` |

Lo que falta en esa lista:

- La matriz dice que el clasificador no tiene importancias (`SKILL.md:157`):
  `ForecasterRecursiveClassifier.get_feature_importances` existe.
- No hay fila de `calendar_features` en la matriz (lo admiten Recursive, Direct,
  RecursiveMultiSeries y DirectMultiVariate; no el clasificador ni EquivalentDate).
- La frase de la guía *Comparing forecasting models* sobre el reentrenamiento (H1), que la
  skill cita en `SKILL.md:163`.
- `skills/baseline-forecasting/SKILL.md` no dice nada de varias series ni de clasificación
  (la corrección 7 lo pide; aquí queda verificado que no está).
- `mkdocs.yml`: la analítica está en las líneas 353-355 y el consentimiento en 356-362 (la
  versión 1 dice 354-356 y 357-363).
- La validación de la versión 1 "clases y parámetros coherentes con `get_tags()`" no puede
  cubrir Stats ni Foundation, por lo mismo que la corrección 9.

**H12. skforecast-ai recomienda otra cosa con los mismos datos.** [Código, paquete 0.5.0 de
PyPI, leído por un subagente y comprobado en tres puntos]

| Tema | skforecast-ai 0.5.0 | Árbol del selector |
|:--|:--|:--|
| Preferido | Siempre ML (`ForecasterRecursive` o `ForecasterRecursiveMultiSeries`); Foundation solo es candidato (`recommendation/forecaster_selection.py:76-98`) | Chronos-2 primero con una serie sin variables |
| Estimador | Ridge con menos de 250 observaciones, LightGBM en otro caso (`:214`) | LightGBM siempre |
| Estadístico | AutoARIMA, y solo si `m < 24`: diario, mensual, trimestral; nunca semanal | AutoETS en semanal, mensual y trimestral |
| Baseline | Estacional, pero no con varias series ni con huecos; `offset=1` si la estación supera el 33 % de la serie | Siempre, uno por serie |
| Backtesting | `refit=False`, `initial_train_size` del 70 % | `refit=False`, bloques fijos por frecuencia |
| Configuración | Lags por PACF, ventanas en varias escalas, sin horizonte por defecto | Listas fijas por frecuencia |
| Variables solo hasta hoy | No las distingue | Opción propia |
| Prioridad del usuario, clasificación | No existen | Pregunta 5 y rama de categoría |
| Coinciden | Estaciones del baseline (24, 7, 52, 12, 4), Chronos-2 small por defecto, LightGBM por defecto con datos suficientes, bloqueo de licencias no comerciales | |

- Una de sus reglas tiene respaldo en este estudio (no usar Ets ni Arima con `m >= 24`) y
  otra, un respaldo parcial (Ridge en series cortas: en mensual con 36 observaciones, Ridge
  reentrenado queda por delante de LightGBM en 2 de 3 datasets). Otra queda en entredicho: `refit=False` por defecto perjudica a ML en su `compare()` igual
  que en el estudio original (H1), porque Stats se reajusta siempre.
- Propuesta: la fuente única del selector (reglas y estaciones) se publica como datos (JSON
  o YAML) para que skforecast-ai la lea, y se abre allí una incidencia con H1 y H4. No es
  requisito para publicar el selector; sí lo es que el botón "Run it on your own data with
  skforecast-ai" no prometa "la misma comparación" mientras difieran.

### Medios

**H13. Lo que no cambia el orden entre familias.** [Medido]

- Un LightGBM más fuerte: frente a `LightGBM+`, Chronos-2 gana en 27 de 38 (frente al
  LightGBM de serie, 26 de 41). La configuración importa menos que el reentrenamiento.
  `min_child_samples` según el tamaño es razonable: con el valor por defecto de 20 y menos
  de 40 filas, LightGBM no puede dividir.
- El horizonte: con el mismo periodo de test y un horizonte 4 veces más corto, Chronos-2
  gana a LightGBM en 9 de 12 casos, igual que con el largo. No hace falta una pregunta
  sobre el horizonte.
- La forma de promediar entre series: media de MASE, media geométrica del error relativo,
  mediana y porcentaje de series dan el mismo orden en los 4 casos multiserie. En la escala
  del MASE no hay problema con una serie: se cancela al dividir por el naive.
- La historia: dentro de cada celda de 3.2 el orden no cambia con la historia.

**H14. Variables solo hasta hoy: el bucle es correcto, la conclusión no.** [Medido]

- El bucle propio sin covariables reproduce exactamente a `backtesting_foundation` en los
  tres casos y con los tres modelos (0,606, 0,683 y 0,646 con small). Es correcto.
- Con covariables frente a sin ellas:

| Caso | small | base | synth |
|:--|:--|:--|:--|
| `vic_electricity` diario | 0,62 / 0,61 | 0,62 / 0,61 | 0,63 / 0,64 |
| `bike_sharing` horario | 0,70 / 0,68 | **0,70 / 0,81** | 0,77 / 0,78 |
| `ett` horario | 0,65 / 0,65 | 0,64 / 0,61 | **0,66 / 0,71** |

- `ForecasterDirectMultiVariate` reentrenado: 0,78, 0,80 y 0,70; no supera a Chronos-2 sin
  variables y tarda de 150 a 250 s en los casos horarios.
- Propuesta: la opción "solo hasta hoy" se funde con "No" en el selector, con una nota. La
  página no afirma "did not improve any model".

**H15. Selección de series y clasificación.** [Medido]

- `bdg2`: Fox era el sitio donde Chronos-2 sacaba menos ventaja. En los 8 sitios con 40 edificios
  completos y en una muestra aleatoria de Fox, Chronos-2 gana en los 9 (0,70 a 0,91 frente a
  0,85 a 1,53 de LightGBM reentrenado, que queda peor que el naive en 3).
- `m4_hourly`: las 50 primeras series por identificador eran el caso favorable a ML (0,50
  frente a 0,61). Con 50 al azar de las de 1008 valores: 1,16 frente a 0,55; con 50 de las
  de 748: 0,92 frente a 0,73. El resultado de ese caso depende de la muestra; las fechas
  alineadas no afectan (el caso no usa calendario y `cross_learning` está desactivado).
- Clasificación: reentrenado, accuracy 0,897 y balanced accuracy 0,739, frente a 0,836 y
  0,752 de "misma hora ayer". Sin exógenas, 0,814 y 0,657. Un solo dataset, con el 6,7 % de
  la clase alta: la rama se justifica por capacidad, no por evidencia. El horizonte igual a
  la estación es correcto para que la regla use solo valores conocidos.

**H16. Qué habría que medir para decidir "precisión → Chronos-2 primero".** [Plausible]

- Más datasets por celda: hoy 8, 2 y 4. Con menos de 8 a 10 datasets independientes por
  celda no se puede publicar un porcentaje.
- Variables conocidas a futuro como se tienen en la realidad (pronósticos, no valores
  observados).
- ML con un presupuesto fijo de ajuste en validación (por ejemplo, 20 pruebas de
  `bayesian_search_forecaster`), que es la comparación que le importa a quien elige ML.
- Datos posteriores al entrenamiento de Chronos-2 o privados. El control con el modelo synth
  (H17) cubre la memorización, no la semejanza de dominio.
- Chronos-2 base en GPU y coste de producción, no segundos de backtest.

**H17. Contaminación y tamaño del modelo.** [Medido]

- Chronos-2 synth, que no ha visto ningún dato real: mejor que el naive en 15 de 15 casos
  con toda la historia, mejor que LightGBM reentrenado en 10 de 15 (small, 11). Medianas:
  small 0,659, base 0,646, synth 0,705. La ventaja de Chronos-2 no viene de haber visto los
  datasets.
- Base frente a small: mejor en 8 de 15, peor en 7 (en `bike_sharing` horario sin
  covariables, 0,81 frente a 0,68). No hay motivo para recomendar el modelo grande.

**H18. Fase 2 probabilística: una fila más, no una extensión con tres preguntas.**
[Código] [Plausible]

- La tabla de 8.2 es correcta contra el código: `predict_bootstrapping` no existe en
  `ForecasterFoundation`; EquivalentDate y Rnn, solo conformal; las seis métricas y
  `ConformalIntervalCalibrator` existen; las cifras de la guía (61 %, 96 a 98 %, 84 %, 35
  veces) están donde dice. Un matiz: `backtesting_foundation` recibe `quantiles`, no
  `interval`.
- P1 (qué necesitas) sí elige el método, por capacidad: un rango (cualquier método),
  varios percentiles (conformal no sirve), totales o probabilidades de un suceso (hacen
  falta trayectorias, solo ML con bootstrapping).
- P2 (¿puedes apartar un periodo reciente?) es configuración, y un principiante no puede
  contestarla. Los residuos out-of-sample deben ir en el código por defecto.
- P3 (nivel) es una constante y P4 (`use_binned_residuals`) es configuración. Fuera.
- Propuesta: una fila opcional del selector, "Do you also need the uncertainty?" (No, a
  range, several percentiles, totals or the chance of an event). El estudio que la habilita
  puede ser más corto que el de 8.5 (cobertura y anchura de conformal y bootstrapping con
  residuos out-of-sample, cuantiles nativos de Chronos-2 y paramétricos de AutoETS), pero
  necesita lo que 8.5 no dice: el mismo protocolo justo, un criterio de decisión fijado
  antes (cobertura a menos de 5 puntos de la nominal y, entre los que cumplen, menor WIS),
  cobertura por paso del horizonte, y más puntos de test en baja frecuencia (36
  predicciones mensuales dan un error típico de unos 7 puntos en una cobertura del 80 %).

**H19. Plan, prioridades y riesgos.** [Plausible]

- Falta en el plan de la versión 1: rehacer el estudio antes de corregir la skill (la fase 0
  iba a escribir en la skill las cifras del ML congelado); el coste de CI del job con
  PyTorch (minuto y medio de ejecución más la instalación y la descarga de pesos); qué pasa
  cuando Chronos-2 o LightGBM cambian de versión y una regla deja de cumplirse (quién lo
  decide y con qué umbral); la accesibilidad del bloque interactivo; y un criterio de
  retirada.
- Sobra: "Después: otros tests" (métrica, backtesting) antes de tener un dato de uso; el
  ranking de umbrales de 6.6 sin tráfico de referencia (no se sabe cuántas visitas tiene
  hoy *Choosing a model*); y la decisión abierta de la analítica (se usa lo que hay).
- Prioridades: `DOC-choosing-forecaster-evidence` sigue en P2 (documentación incorrecta que
  los usuarios y los asistentes siguen), pero requiere rehacer el estudio. El selector, P3.
  La fase probabilística, P4. Los dos fallos de Ets, por delante de todo (H4).
- Riesgo nuevo: publicar reglas con 2 a 8 datasets por celda. Mitigación: cada regla con su
  respaldo a la vista y la palabra "tie" donde lo hay.

### Bajos

**H20. Erratas y detalles de la versión 1.** [Código] [Medido]

- "17 de 38" y "26 de 33" usan denominadores distintos (el primero excluye los 3 casos de
  variables solo hasta hoy; el segundo los incluye).
- El docstring de `finder_tree.decide` dice "5 to 10 times faster"; el plan, "de 2 a 11".
- `TimeSeriesFold` del original usa `fixed_train_size=True` (valor por defecto): los
  estadísticos se reajustaron con ventana deslizante, no creciente.
- `environment.json` del original no guardaba la versión de `chronos-forecasting`.
- `build.py` escribe `finder.json` en la carpeta del código y no está en `.gitignore`.

**H21. No hay un `TimeSeriesFold` común a las tres familias sin avisos.** [Medido]

- Con `refit=False`, `backtesting_stats` avisa de que lo cambia a `True`; con `refit=True`,
  `backtesting_foundation` avisa de que lo ignora. La documentación recomienda comparar
  familias con el mismo `cv`. Encontrado en la segunda pasada, al ejecutar los fragmentos
  con los avisos como errores.
- Propuesta: fila `QW-shared-cv-refit-warnings` del roadmap. Entretanto, el fragmento usa
  `suppress_warnings=True` en la llamada de Chronos-2.

## 5. Lo que no se pudo comprobar

- El repositorio `skforecast-ai` (acceso denegado): se usó el paquete publicado, que puede
  ir por detrás de la rama de desarrollo.
- GPU y coste de producción de Chronos-2.
- Datos privados o posteriores a la publicación de Chronos-2.
- Los otros diez modelos fundacionales.
- El tráfico actual de la documentación, necesario para fijar umbrales de uso.

## 6. Segunda pasada: arreglo de Ets, estudio medido de nuevo y prototipo

Mismo entorno que el resto del documento. Commit del arreglo: `5f020a0`.

### 6.1 Los dos fallos de `Ets`, arreglados

**Causa del primero** [Código] [Medido]. La verosimilitud de un modelo con error
multiplicativo es `n log(sum(e^2) / n) + 2 sum(log|yhat|)`, con `e = (y - yhat) / yhat`.
Nada impedía que el nivel cruzara el cero: cerca de cero el segundo término se hace muy
negativo, de modo que un ajuste sin sentido puede tener una verosimilitud finita, incluso
buena. Al reajustar con los parámetros de suavizado fijos (lo que hace `freeze_params=True`),
la única salida del optimizador era la tendencia de las primeras observaciones; con alpha y
beta en 1e-4 esa tendencia se extrapola a toda la serie, el nivel cruza el cero y la
búsqueda se queda ahí. En "The Murray | Visiting": estados iniciales de nivel 102,7 y
tendencia -3,3 por trimestre para una serie de 72 trimestres entre 45 y 125.

**Arreglo** (`skforecast/stats/exponential_smoothing/_ets_base.py`):

1. Un modelo con error o estacionalidad multiplicativos no es válido si su nivel, su estado
   estacional o su predicción a un paso no son positivos (igual que ya pasaba con la
   tendencia multiplicativa). Se aplica al ajustar, no al simular trayectorias.
2. Con alpha fijo, los estados iniciales se buscan también desde la tendencia de toda la
   serie y desde "sin tendencia", y se elige el mejor.
3. `ets()` da `ValueError` si el optimizador no encuentra ningún punto válido (en la
   selección automática, ese candidato se descarta).

**Segundo fallo**: la selección automática con `m > 24` da un `IgnoredArgumentWarning` que
dice que solo se consideran modelos sin estacionalidad (no aparece con `seasonal=False`).
Lo recogen el docstring de `Ets` y `skills/statistical-models/SKILL.md`.

**Medido**:

| Comprobación | Antes | Después |
|:--|:--|:--|
| 912 reajustes con los parámetros congelados (304 series de `australia_tourism`, 3 tamaños): MAE medio a 4 pasos | 4,0e8 | 14,7 |
| Verosimilitud de esos 912 reajustes | | Mejor en 34, igual en 873, peor en ninguno (5 con diferencias menores de 2e-6) |
| Modelo elegido por la selección automática en las 304 series | | El mismo en las 304 |
| 60 series sintéticas (tendencia suave, estacionalidad multiplicativa, ruido del 15 %) por `backtesting_stats` | 3 con predicciones negativas o del orden de -1e10 | 0 |
| Tiempo de los 912 reajustes | 3,1 s | 4,3 s |
| Las 42 cifras de AutoETS y AutoARIMA del estudio | | 41 idénticas; cambia la de `australia_tourism` |

Tests: 8 nuevos (verosimilitud no válida, reajuste con parámetros fijos, estados iniciales,
`ValueError` sin punto válido, aviso con `m > 24` en la función, en la clase y sin él con
`seasonal=False`, y `backtesting_stats` con las tres semillas que fallaban). `verify`: lint
sin hallazgos nuevos y 457 tests de `stats`, `ForecasterStats`, `backtesting_stats` y la
búsqueda de stats, en verde. Notas de versión añadidas (corrección destacada, línea en
"Before upgrading", una entrada en Changed y otra en Fixed), sin enlace a PR porque no hay.

Existe desde la 0.27.0, cuando `freeze_params=True` empezó a congelar de verdad: es un fallo
de una versión publicada, no de la 0.28.x.

### 6.2 El estudio, medido de nuevo

Solo cambian los modelos estadísticos, así que solo ellos se volvieron a ejecutar
(`review.py --stats`; los valores anteriores quedan en `results/review/stats_previous.csv`).

- `australia_tourism` (304 series trimestrales), AutoETS por serie: 0,859 frente al naive
  (antes, 12,7 millones). Chronos-2, 0,866; LightGBM, 0,925. Mejor que el naive en el 74 %
  de las series, que Chronos-2 en el 52 % y que LightGBM en el 60 %. 39 s frente a 2,4 s de
  Chronos-2.
- Baja frecuencia con un modelo estadístico (16 ejecuciones): menos error que Chronos-2 en
  8 (AutoETS solo, en 7), que LightGBM en 14. Mensual, sin cambios: 6 de 12.
- El resto de cifras del documento no cambia.

### 6.3 El prototipo del selector, reescrito

`finder_tree.py` y `build.py` siguen ahora la sección 6 de `PLAN.md`: 4 filas, reglas por
tipo de datos, 36 combinaciones y 36 fragmentos.

| Requisito de `PLAN.md` 6.3 | Antes (versión 1) | Ahora |
|:--|:--|:--|
| Sin avisos | 72 de 100 fragmentos avisaban | 0 de 36 (`build.py` falla con cualquier aviso; solo se admite uno de scikit-learn que depende de los datos) |
| Todos los modelos usan lo anterior a cada bloque | ML congelado | `refit=True` y ventana creciente |
| Historia mínima | 21 días, 150 días, 2 años, 5 años, 5 años | 2 semanas, 3 meses, 2 años, 3 años, 6 años (horario, diario, semanal, mensual, trimestral); `check_snippets.py` lo comprueba |
| El LightGBM del fragmento, medido | Peor que el naive en 17 de 38; hasta 4,1 veces su error; 66 veces en un sitio de `bdg2` sin escalar | Peor que el naive en 14 de 38 (`review.py --snippet`); hasta 1,6 veces; con varias series, nunca (máximo 0,94) |
| Las cifras del respaldo se generan | Escritas a mano en la maqueta | `build.py` las calcula de `recount_runs.csv` |
| AutoETS | También en semanal | Solo mensual y trimestral de una serie |

El LightGBM del fragmento (valores por defecto más `min_child_samples=5`, series escaladas
solo en el multiserie) es algo más flojo que el del estudio (14 frente a 11 ejecuciones peor
que el naive, todas en "una serie, nada conocido a futuro"). El respaldo que enseña la
página se calcula con el del fragmento: "LightGBM did worse than the seasonal baseline in 14
of 22 runs". Si se quiere el del estudio (300 árboles, `learning_rate=0.05`, 15 hojas,
`StandardScaler`), el fragmento crece cuatro argumentos; no compensa.

Lo que el prototipo no resuelve: el horizonte y el número de bloques del fragmento no son
los del estudio (H13 mide que el horizonte no cambia el orden), y la clasificación solo
tiene un dataset.

## 7. Tercera pasada: estudio ampliado, ARIMA y un resultado más simple

Mismo entorno. Scripts: `cases_d4.py`, `review.py --d4`, `review.py --arima`, `rules.py`.
Resultados: `results/review/d4_*.csv`, `rules_runs.csv`, `rules_cells.csv`, `rules_stats.csv`.

### 7.1 D4: de 12 a 26 datasets [Medido]

20 casos nuevos, con toda la historia, el mismo protocolo y solo los modelos que usan las
reglas. Un voto por dataset en cada tipo de datos; LightGBM es el que escribe el fragmento.

| Tipo de datos | Datasets (antes) | Chronos-2 con menos error | LightGBM con más de un 5 % de ventaja | LightGBM peor que el naive | Chronos-2 peor que el naive |
|:--|--:|--:|--:|--:|--:|
| Una serie, nada conocido a futuro | 16 (8) | 14 | 0 | 7 | 0 |
| Una serie, variables conocidas a futuro | 7 (2) | 5 | 1 | 1 | 0 |
| Varias series | 10 (4) | 8 | 1 | 2 | 0 |

- **Corrige H2 y la tabla de 3.2.** "Sin favorito" con variables conocidas a futuro o con
  varias series salía de 2 y 4 datasets. Con 7 y 10, Chronos-2 va por delante en 5 y en 8.
  La regla queda en una sola: Chronos-2 primero y LightGBM como el modelo que se trabaja.
- Donde LightGBM queda más cerca es con variables conocidas a futuro: gana en
  `vic_electricity` (4 %) y `h2o_exog` (6 %), y en 3 de los 5 datasets que pierde lo hace por
  menos de un 10 %.
- Variables de verdad conocidas (calendario, eventos de M5, las dos de `h2o_exog`): Chronos-2
  por delante en 1 de 2 datasets. Con meteorología observada: 4 de 5. Sigue siendo la celda
  con menos datos y con el supuesto más favorable (valores observados como si fueran
  pronósticos).
- Frente a `LightGBM+`: 12 de 13, 4 de 7 y 8 de 10.
- Chronos-2 no queda peor que el naive en ningún dataset (en 1 de 61 ejecuciones).

### 7.2 AutoETS o AutoARIMA como referencia [Medido]

| Frecuencia (una serie) | Datasets | AutoETS con menos error que AutoARIMA | Menos error que Chronos-2 (ETS / ARIMA) | Menos error que LightGBM (ETS / ARIMA) | Peor que el naive con toda la historia (ETS / ARIMA) |
|:--|--:|--:|:--|:--|:--|
| Mensual | 6 | 4 | 0 / 1 | 4 / 4 | 1 de 6 / 3 de 6 |
| Trimestral | 2 | 1 | 1 / 2 | 2 / 2 | 0 / 0 |
| Semanal | 2 | 0 | 0 / 1 | 1 / 1 | |
| Diario | 8 | 4 | 0 / 1 | 3 / 2 | |

- 304 series trimestrales, un modelo por serie: AutoETS 0,84 de mediana (mejor que
  Chronos-2 en el 52 %, 39 s); AutoARIMA 0,86 (50 %, 131 s); Chronos-2 0,88 (2,4 s).
- **Corrige H7**: el modelo estadístico no "empata con Chronos-2" en mensual (le gana en 0 o
  1 de 6 datasets; salía de 3 datasets con varias historias cada uno). Lo que el estudio
  respalda es que supera al LightGBM sin ajustar en 6 de 8 datasets mensuales y
  trimestrales, y en trimestral sí queda a la altura de Chronos-2.
- Entre los dos no hay una diferencia clara. A favor de AutoETS: falla menos frente al naive
  en mensual (1 de 6 frente a 3 de 6) y no emite avisos. A favor de ARIMA: es el modelo de
  la portada y el más reconocido, y sirve con `m=52`. Decisión del usuario: ARIMA.
- Costes de ARIMA, medidos: la selección automática emite avisos de convergencia también con
  datos sintéticos limpios (el fragmento usa `suppress_warnings=True`, como la guía
  *Comparing forecasting models*), y con las variables conocidas a futuro es errático (mejora
  en 2 de 5 casos y empeora en 2, hasta 1,56 veces el naive en `vic_electricity`), así que
  el fragmento lo usa sin ellas.

### 7.3 Las tres dudas del usuario

1. **"How often are the values recorded?"** pasa a "Your data has one value per..." con las
   opciones Hour, Day, Week, Month, Quarter y Something else, y la pista "skforecast calls
   this the frequency of the series". Es una frase que se completa, sin la palabra
   "recorded" ni "frequency" en la pregunta. [Plausible]
2. **Complejidad**: de acuerdo en simplificar. El resultado ya no enseña fracciones como
   "lower error in 1 of 2 datasets": una frase fija bajo el título dice que cada tipo de
   modelo se probó en 26 datasets, cada resultado lleva una frase sin cifras, y las cifras
   están en "How we chose these rules". Con el estudio ampliado, además, el titular es el
   mismo en todos los casos de número. Se quitan las tarjetas; las notas van plegadas.
   Para que la frase no se quede vieja, `build.py` comprueba cada una contra los resultados
   del estudio y falla si dejan de respaldarla.
3. **ETS o ARIMA**: ARIMA (7.2).

### 7.4 Estado del prototipo

36 combinaciones, 36 fragmentos, 0 avisos, las 5 frases respaldadas por el estudio, los 18
tipos de fragmento se ejecutan con su historia mínima. Maqueta: `mock.html`.
