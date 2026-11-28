# Selector de forecaster ("Which forecaster should I use?"): lo que queda del plan

## 1. Estado

- **Hecho (fases 0, 1 y 2):** estudio de referencia, página y selector, en la PR
  [#1550](https://github.com/skforecast/skforecast/pull/1550) contra `0.28.x` (2026-11-28).
- **Pendiente:** lo de la sección 5 al publicar, la fase 3 (medición, sección 6) y la fase 4
  (fila de incertidumbre, sección 7), que depende de la 3.
- Lo construido se describe en `tools/docs/forecaster_finder/README.md` (selector) y en
  `benchmarks/forecaster_finder/README.md` (estudio). Donde difieran de este fichero, valen
  ellos.
- Este fichero se recortó el 2026-11-28 a lo que sigue vivo. El plan completo (estudio,
  diseño, estado de la implementación y las tres revisiones como usuario, 997 líneas) está en
  `1446a64:dev/forecaster_finder/PLAN.md`; las citas `FINDER §N` del roadmap se refieren a
  las secciones de esa versión. La revisión independiente del estudio está en
  `58e09f9:dev/forecaster_finder/REVISION.md`.
- Filas del roadmap: `FEAT-forecaster-finder` (fase 3 pendiente) y
  `FEAT-finder-probabilistic` (fase 4). Esta carpeta se borra al cerrar la última.

## 2. Alcance

La página es una prueba rápida que atrae y encamina: con cuatro respuestas en inglés
cotidiano da los forecasters que probar primero, el baseline con el que compararlos y un
script que funciona tal cual.

- Decide qué forecasters probar y contra qué baseline. No decide la configuración: lags,
  hiperparámetros y variables siguen en las guías y en las skills.
- No es una guía de usuario ni una librería como skforecast-ai. Conectar el script a los
  datos propios es cosa del lector, con las guías a las que enlaza. El lector sigue
  descubriendo la librería desde ahí: la página no tiene que dar todas las respuestas.
- Cada regla lleva su respaldo ("How we chose these rules"). Una pregunta se queda solo si
  su respuesta cambia lo que se recomienda o el código.

## 3. Decisiones cerradas

1. Dos candidatos con papeles distintos, no un ranking: LightGBM ("You train it") y
   Chronos-2 small ("No training"), siempre contra un baseline estacional. Para categorías,
   un clasificador y una regla sencilla.
2. ARIMA como referencia, solo con una serie mensual o trimestral.
3. Cuatro preguntas en una pantalla (tres para una categoría), nada elegido al llegar. Sin
   preguntas de historia, horizonte ni prioridad.
4. El resultado no enseña cifras: una frase por caso, comprobada contra el estudio al
   generar la página.
5. Script con datos de ejemplo, en el orden baseline, LightGBM, ARIMA y Chronos-2 al final,
   con cada resultado impreso al obtenerlo. Error en las unidades de los datos (MAE);
   acierto (accuracy) para categorías.
6. Página propia en la pestaña *Quick start*, que funciona sin JavaScript. Sin barra de
   anuncios.
7. Fuente única (`finder.py`) y ficheros generados; el CI se ejecuta solo en las PR que
   tocan el selector o su estudio, sin ejecución semanal.
8. Analítica: GA4 con el consentimiento que ya existe.
9. Los arreglos de `Ets` y `Arima` que salieron del estudio viajan en la misma PR.

## 4. Lo que enseñaron las tres revisiones como usuario

Leer antes de cambiar las preguntas, el bloque de datos o los textos.

- Lo que se ve, lo que copia el botón y lo que ejecuta el CI tienen que ser el mismo texto:
  la primera revisión encontró que el script copiado no traía datos y fallaba.
- El recuento de preguntas tiene que cuadrar con lo que se ve, y las respuestas tienen que
  sobrevivir a recargar, al aviso de cookies y a los enlaces de sección.
- Las líneas de instalación van en el orden en que se ejecutan: PyTorch para CPU primero
  (34 s y 1,4 GB frente a 457 s y 5,8 GB en Linux sin GPU, medido por el revisor).
- Para categorías hay una sola prueba, y la página lo dice con sus cifras.
- Descartado a propósito, por alcance: más ayuda para cargar datos reales, líneas para
  predecir con el baseline y notas sobre casos muy concretos de la librería.
- Conocido y sin tocar: la búsqueda no encuentra la página con "choose a forecaster" (sí con
  "choose forecaster" y con "which forecaster should I use"); en móvil el nombre
  `ForecasterRecursiveMultiSeries` se parte en dos líneas; "Start over" no se puede deshacer.

## 5. Al publicar

- Revisar los cambios de `Ets` y `Arima` como cambios de la librería. El de `Ets` corrige un
  fallo de la 0.27.0 publicada: decidir si merece una versión de parche.
- Fusionar la PR con squash: los commits antiguos de la rama llevan unos 6 MB de ficheros de
  trabajo borrados después.
- Construir el sitio con el plugin `privacy` (en la sesión de desarrollo no pudo descargar
  los recursos externos).
- Probar "Run it in Google Colab" cuando la rama llegue a `main`: abrir el enlace, ejecutar
  la primera celda, pegar el script y ejecutarlo. El enlace apunta a `main`.
- Exportar de GA4 las visitas de las 4 semanas anteriores de *Introduction to forecasting*,
  *Quick start* y *Comparing forecasting models*, como referencia para la fase 3.

## 6. Fase 3: medición y decisión (`FEAT-forecaster-finder`)

A las 8 semanas de publicar, una nota con las cuatro señales y la decisión: mantener,
simplificar a la página estática o ampliar. Termina cuando la decisión está en el roadmap.

Eventos, solo con consentimiento y con identificadores cerrados: `finder_change`,
`finder_copy`, `finder_open`, `finder_cta`, `finder_feedback` y `finder_reset`.

| Pregunta | Señal | Criterio a las 8 semanas |
|:--|:--|:--|
| ¿Llegan? | Visitas de la página | Al menos las de la tabla de la introducción a la que sustituye como destino |
| ¿Les sirve? | Copias de script o de prompt por visita | 1 de cada 5 visitas |
| ¿Convierte? | Clics a skforecast-ai, Studio y consultoría | Más que los mismos enlaces en la portada, por visita |
| ¿Acierta? | "Was this useful?" | Mayoría de respuestas positivas; revisar los comentarios |

## 7. Fase 4: fila de incertidumbre (`FEAT-finder-probabilistic`)

Solo si la fase 3 termina con la decisión de mantener. Una quinta pregunta opcional, "Do you
also need the uncertainty of the forecast?": **No** (por defecto, nada cambia), **A range**,
**Several percentiles** o **Totals or the chance of an event**. Las tres últimas añaden al
script el método de cada candidato y el backtest con cobertura y anchura.

Reglas, como hipótesis por capacidad; el estudio las confirma o las cambia:

| Opción | Chronos-2 | LightGBM | ARIMA | Baseline |
|:--|:--|:--|:--|:--|
| A range | Cuantiles nativos | Conformal con residuos out-of-sample | Paramétrico | Conformal |
| Several percentiles | Cuantiles nativos | Bootstrapping con residuos out-of-sample | No aplica | No aplica |
| Totals or the chance of an event | No puede (sin trayectorias): el resultado lo dice | `predict_bootstrapping` | No aplica | No aplica |

El nivel (80 %) es una constante editable; los residuos out-of-sample van en el código por
defecto, sin preguntar.

Estudio que la habilita, con los mismos casos y el mismo protocolo que el de referencia:

- Métricas: cobertura empírica del 80 %, anchura, WIS, cobertura por paso y tiempo. Para
  totales, cobertura de la suma del horizonte.
- Criterio fijado antes de medir: un método vale si su cobertura queda a menos de 5 puntos
  de la nominal en la mayoría de los datasets; entre los que valen, el de menor WIS.
- En baja frecuencia hacen falta más bloques (36 predicciones mensuales dan un error típico
  de unos 7 puntos en una cobertura del 80 %): usar orígenes solapados.
- Qué debe contestar: ¿conformal o bootstrapping para "A range" con LightGBM? ¿Los cuantiles
  nativos de Chronos-2 están calibrados sin ajuste? ¿El coste de bootstrapping con muchas
  series obliga a otra regla para varias series?

Se activa cuando: la fase 3 está cerrada con la decisión de mantener; el estudio está hecho
y las reglas lo reflejan; los scripts nuevos se ejecutan en el CI; y están corregidas
`skills/prediction-intervals/SKILL.md` (su tabla no incluye los cuantiles nativos) y
`references/interval-compatibility.md` (no incluye `ForecasterFoundation`). Lo que ofrece
cada forecaster, verificado en el código el 2026-11-27, está en la sección 8.2 del plan
completo.
