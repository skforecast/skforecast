# Prompt para la revisión como usuario

Para pegar en una sesión nueva. Todo lo que va debajo de la línea es el prompt.

---

Responde en español. Sigue CLAUDE.md.

## Qué te pido

En la rama `docs/forecaster-finder` del repositorio hay una página nueva de la documentación de skforecast, "Which forecaster should I use?" (`quick-start/choose-a-forecaster.html`), con un selector interactivo: cuatro preguntas sobre los datos, y como respuesta qué forecasters comparar, con qué baseline y un script de Python que promete ejecutarse tal cual. También hay enlaces nuevos a esa página desde la portada, la introducción, el Quick start, las guías, la FAQ y las páginas de ejemplos.

Quiero que la uses **como la usaría una persona cualquiera que llega a la documentación**, no como un desarrollador que revisa el código, y que escribas un informe con lo que falle, confunda, se vea mal o haga que alguien abandone. Otro agente arreglará después lo que encuentres: tu trabajo es solo encontrar y describir.

## Reglas

- **No leas cómo está hecha.** No abras `tools/docs/forecaster_finder/`, `benchmarks/forecaster_finder/`, `dev/forecaster_finder/` (salvo este fichero), `docs/javascripts/finder.js`, `docs/javascripts/finder-data.json`, `docs/stylesheets/finder.css` ni el Markdown de la página. Un usuario no los ve, y si los lees dejarás de tropezar donde tropezaría él. Si algo no se entiende desde la pantalla, eso es un hallazgo: no vayas al código a buscar la explicación.
- **Mira con los ojos.** Usa el navegador (Chromium y Playwright ya están instalados): haz clic, escribe, cambia de tamaño, y saca capturas y **míralas** antes de opinar. No sirve comprobar que un elemento existe en el HTML: sirve ver si se lee, si está cortado, si tapa algo, si se entiende.
- **Haz lo que dice la página, al pie de la letra.** Si dice "pip install ...", instala eso en un entorno nuevo, no en el que ya tengas. Si dice "replace this block with your own data", hazlo solo con lo que la página y los comentarios del script explican. Si tienes que adivinar algo o usar conocimiento que un principiante no tiene, anótalo.
- **No arregles nada ni cambies ficheros del proyecto.** Solo escribes el informe y sus capturas.
- **No des nada por bueno de antemano.** Lo ha probado quien lo hizo, con sus datos de ejemplo. No sé qué falla: por eso te lo pido.
- Sé concreto y honesto. "Está bien" no me sirve; "no he encontrado nada tras probar X, Y y Z" sí. Si no pudiste probar algo, dilo: no lo des por probado.

## Cómo ver la página

El sitio no está publicado; hay que construirlo. Esto no forma parte de la revisión, es solo para tenerlo delante:

```bash
git fetch origin docs/forecaster-finder && git checkout docs/forecaster-finder
uv venv ~/.venvs/docs --python 3.12 && uv pip install --python ~/.venvs/docs/bin/python -e ".[docs]"
# El plugin `privacy` descarga recursos externos y falla sin acceso a GitHub: constrúyelo sin él
sed 's/^  - privacy:$/  - privacy:\n      enabled: false/' mkdocs.yml > mkdocs.preview.yml
~/.venvs/docs/bin/mkdocs build -q -f mkdocs.preview.yml -d <tu carpeta temporal>/site
rm mkdocs.preview.yml
python -m http.server 8765 --directory <tu carpeta temporal>/site
```

Cosas que ya se saben y **no** son hallazgos: en esta copia local falta `versions.json` (el selector de versión); el botón "Run it in Google Colab" abre un notebook que todavía no existe en la rama `main` (prueba lo que pasa al pulsarlo y qué se copia, pero el 404 de Colab es esperado); y el aviso de cookies aparece en la primera visita, como en el sitio real (descártalo como haría un usuario, y anota si estorba más de lo normal).

## A quién interpretas

Haz el recorrido completo con cada una de estas personas. Para cada una, usa **datos de verdad, no los de ejemplo de la página**: coge un dataset público (por ejemplo con `skforecast.datasets.fetch_dataset`, o cualquier CSV público), guárdalo como CSV y a partir de ahí compórtate como alguien que solo tiene ese fichero.

1. **Ana, analista que empieza con Python.** Tiene un CSV con las ventas diarias de una tienda (una columna de fecha y una de ventas). No sabe qué es un forecaster, ni "exogenous", ni "backtesting". Llega por la portada.
2. **Luis, científico de datos con experiencia en scikit-learn.** Tiene la demanda mensual de varios años de un producto y quiere saber rápido por dónde empezar. Llega por "Intro to Forecasting".
3. **Marta, responsable de 20 o 30 tiendas.** Tiene las ventas diarias de cada tienda y sabe de antemano los festivos o las promociones. Llega buscando en las guías de usuario.
4. **Pablo, ingeniero de mantenimiento.** Quiere predecir una categoría (nivel bajo, medio o alto) a partir de lecturas horarias. Usa el móvil para mirar la documentación y luego el portátil.
5. **Sara, con datos cada 15 minutos** (o cualquier frecuencia que no esté entre las opciones). Además prefiere pedírselo a un asistente de IA.

## Qué recorrer con cada persona

1. **Llegar.** Empieza en la página por la que llega esa persona y encuentra el selector sin que yo te diga dónde está. ¿Se ve la entrada? ¿Se entiende qué te va a dar antes de hacer clic? Prueba todas las entradas nuevas que encuentres (portada, introducción, Quick start, tablas de contenidos de guías y de FAQ, páginas de ejemplos en los tres idiomas, README) y comprueba que llevan donde dicen.
2. **Contestar.** Lee cada pregunta y cada opción como esa persona. ¿Sabría qué contestar? ¿Hay alguna palabra que no entendería? ¿Qué pasa si se equivoca y cambia una respuesta, si contesta en otro orden, si pulsa "Start over", si recarga, si usa el botón atrás del navegador después de visitar una guía, si comparte la URL con otra persona?
3. **Entender la respuesta.** ¿Queda claro qué tiene que hacer ahora? ¿Los nombres que aparecen le dicen algo? ¿Los enlaces a las guías llevan a una página que le sirve? ¿Hay algo que se contradiga con lo que dice el resto de la documentación (por ejemplo, el Quick start)?
4. **Ejecutar el script con sus datos.** En un entorno nuevo, instala lo que el script dice e intenta que funcione con el CSV de esa persona siguiendo solo las instrucciones que da la página. Mide cuánto tarda cada paso (instalar, primera ejecución, segunda). Anota cada error, cada aviso y cada momento en que tuviste que saber algo que la página no dice. Cuando funcione: ¿se entiende lo que imprime? ¿Sabe esa persona qué hacer con esos números?
5. **Lo que hay debajo del script.** Abre cada sección plegada y prueba cada botón: copiar, Colab, las líneas de skforecast-ai (ejecútalas si puedes), el mensaje para un asistente con MCP, "Copy script as a prompt" (pégalo en un asistente o léelo como lo leería uno: ¿le llevaría a hacer lo correcto?), "Good to know", "Was this useful?".
6. **El resto de la página.** Lee lo que hay bajo el selector. ¿Aporta o repite? ¿Las cifras se entienden? ¿Los enlaces funcionan?

## Además, una vez

- **Tamaños de pantalla**: 360, 400, 768, 1024, 1280 y 1920 px de ancho. En cada uno, capturas del selector vacío, a medias y completo. Busca texto cortado, solapes, botones que no caben, desplazamiento horizontal, elementos que se quedan fuera de la vista al contestar.
- **Tema oscuro**: cambia el tema con el botón de la cabecera y repite las capturas principales. Busca texto ilegible o colores que no cambian.
- **Solo teclado**: completa el selector y copia el script sin tocar el ratón. ¿Se ve dónde está el foco? ¿El orden tiene sentido?
- **Sin JavaScript**: desactívalo y mira qué queda. ¿La página sigue sirviendo para algo?
- **Las 36 combinaciones**: recórrelas todas en el navegador (es rápido) y mira que el texto de la respuesta, el script y la frase sobre los datos mínimos tengan sentido juntos. No hace falta ejecutar las 36: elige las que te parezcan más raras.
- **Cosas que haría alguien con prisa**: doble clic, copiar antes de terminar, ir y volver entre páginas rápido, abrir el enlace compartido en otra pestaña, abrirlo con media respuesta.
- **Lectura**: repasa todos los textos en inglés de la página y de las entradas nuevas. Erratas, frases que no se entienden a la primera, jerga, promesas que la página no cumple, tono.
- **Primera impresión**: antes de nada de lo anterior, abre la página, mírala diez segundos y escribe qué crees que es y qué harías. Hazlo también con la portada.

## El informe

Escríbelo en `dev/forecaster_finder/user_review/INFORME.md`, con las capturas que lo respaldan en `dev/forecaster_finder/user_review/` (PNG a escala 1, las justas: una por hallazgo que sea visual). Haz commit **solo de esa carpeta** en la rama `docs/forecaster-finder` y súbelo; además, pega el informe completo en el chat.

Estructura:

1. **Resumen en cinco líneas**: si se lo enseñarías ya a usuarios reales o no, y por qué.
2. **Primera impresión** de la página y de la portada (lo que escribiste a los diez segundos).
3. **Hallazgos**, ordenados de más a menos grave, cada uno con:
   - Gravedad: **Bloquea** (el usuario no puede seguir o se va con un resultado erróneo), **Confunde** (sigue, pero con dudas o tras equivocarse), **Molesta** (lo nota y le resta confianza) o **Detalle**.
   - A quién le pasa (qué persona, qué respuestas, qué tamaño de pantalla, qué tema).
   - Qué hice, paso a paso, para que otro lo pueda repetir.
   - Qué esperaba y qué pasó, con el texto literal del error o del mensaje y la captura.
   - Por qué importa para ese usuario. No propongas cómo arreglarlo en el código; si tienes una idea de qué le habría ayudado como usuario, dila en una frase.
4. **El recorrido de cada persona** en unas líneas: llegó o no, cuánto tardó hasta ver un resultado con sus datos, dónde se atascó, si confiaría en lo que obtuvo.
5. **Tiempos medidos** (instalación, primera y segunda ejecución, por caso) y el entorno (sistema, Python, versiones instaladas).
6. **Lo que funcionó bien y no tocaría**, breve, para que no se rompa al arreglar lo demás.
7. **Lo que no pude probar** y por qué.

Si dudas entre incluir algo o no, inclúyelo y marca que no estás seguro. Prefiero un hallazgo de más que uno de menos.
