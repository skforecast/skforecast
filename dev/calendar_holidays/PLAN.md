# Plan: festivos en `CalendarFeatures` con el paquete `holidays` (FE-06)

Plan para que `CalendarFeatures` y `create_calendar_features` generen un indicador de festivo a partir del índice, con el paquete [`holidays`](https://github.com/vacanza/holidays) (organización vacanza) como dependencia opcional, o con una lista de fechas propia. Fila del roadmap: **FE-06**. Decisiones abiertas: **D-FE06** (sección 3 de este documento).

**Estado:** plan escrito el 2026-11-24 sobre `0.28.x` (`1f30b0580`), sin implementar. Nada del paquete se ha tocado. El prototipo de las dos funciones internas y las comprobaciones del paquete `holidays` están en `scripts/holidays_facts.py`, con su salida en `scripts/holidays_facts_output.txt`.

**Criterio (2026-11-24):** la funcionalidad se valora por ser una forma cómoda, automática y robusta de añadir festivos, no por la mejora de precisión medida en un conjunto de datos. Por eso este plan no amplía la medición de R27 §2 (mediana de -4,5 % de MAE con Ridge y -6,3 % con LightGBM en las 2 series diarias con festivos) y dedica el esfuerzo a los casos en los que el resultado sería erróneo sin aviso (secciones 4 y 5).

**Relación con otras filas:**

- **QW-11** añade features nuevas a la misma lista `features` de `CalendarFeatures` y a los mismos tests: seguidas, no en paralelo (cadena C12 del roadmap).
- **ISSUE-1507** es de `calculate_distance_from_holiday`, otra función del mismo fichero. No interfiere con la versión 1; sí con la versión 2 (sección 9).

## 1. Qué resuelve

Hoy un festivo es una variable exógena que el usuario construye a mano. Tiene que conseguir las fechas, crear la columna para el periodo de entrenamiento y volver a crearla para cada horizonte de predicción y para cada fold del backtesting. `calculate_distance_from_holiday` no lo evita: necesita la columna de festivos ya construida.

Los forecasters `ForecasterRecursive`, `ForecasterRecursiveMultiSeries`, `ForecasterDirect` y `ForecasterDirectMultiVariate` ya generan las features de calendario por su cuenta: llaman a `calendar_features.fit_transform(train_index)` al entrenar y a `calendar_features.transform(prediction_index)` en cada predicción. Si el festivo sale del índice, el usuario no construye nada, ni para entrenar ni para predecir:

```python
from skforecast.preprocessing import CalendarFeatures
from skforecast.recursive import ForecasterRecursive

calendar_features = CalendarFeatures(
    features        = ['day_of_week', 'month', 'holiday'],
    encoding        = 'onehot',
    holidays        = 'ES',               # o una lista de fechas propia
    holidays_kwargs = {'subdiv': 'MD'},   # se pasan a holidays.country_holidays
)
forecaster = ForecasterRecursive(
    estimator         = Ridge(),
    lags              = 14,
    calendar_features = calendar_features,
)
forecaster.fit(y=y)
forecaster.predict(steps=7)   # el festivo de los 7 días se genera solo
```

Lo ofrecen darts (`holidays_timeseries`), sktime (`HolidayFeatures`), Prophet (`add_country_holidays`) y pytimetk, todos con el mismo paquete.

## 2. Alcance

**Versión 1 (este plan):**

- Feature nueva `'holiday'`: entero 0 o 1, a partir de la fecha local de cada marca de tiempo.
- Calendario de un país, con las opciones del paquete (`subdiv`, `observed`, `categories`...), o una colección de fechas propia. Con fechas propias no hace falta el paquete.
- En la función `create_calendar_features` y en la clase `CalendarFeatures`. Los forecasters no cambian.
- Avisos para los tres casos en los que el resultado sería erróneo en silencio (años sin cobertura, versión distinta del paquete, frecuencias más gruesas que la diaria).

**Fuera de la versión 1** (sección 9): distancia al festivo, vísperas y puentes, número de festivos por periodo en datos semanales o mensuales, calendarios de mercados financieros, un calendario distinto por serie y el nombre del festivo como feature.

## 3. Decisiones de diseño (D-FE06)

Cada una con su propuesta. Cerrarlas antes de implementar.

### D1. Cómo se activa

- **(a) Propuesta:** `'holiday'` es un nombre más de `features`, y el parámetro `holidays` define el calendario.
  - Con `features=None` y `holidays` informado, la lista por defecto incluye `'holiday'` al final. Sin `holidays`, la lista por defecto no cambia (no hay cambio de comportamiento).
  - `'holiday'` en `features` sin `holidays`: `ValueError` en `__init__`.
  - `holidays` informado y una lista explícita sin `'holiday'`: `IgnoredArgumentWarning` (el mismo criterio que `max_values` con `encoding='onehot'`).
- **(b)** Solo el parámetro: `holidays` informado añade la columna, sin nombrarla en `features`.

Motivos de (a): `features` sigue siendo la lista completa de lo que se genera y fija el orden de las columnas; el forecaster muestra `calendar_features_names` leyendo `calendar_features.features`, así que con (a) los forecasters no se tocan y con (b) habría que cambiar los cuatro; y la versión 2 se amplía igual, con más nombres (`'time_to_holiday'`...) sobre el mismo calendario.

### D2. Forma de los parámetros

- **(a) Propuesta:** dos parámetros, en la función y en la clase:
  - `holidays : str, list, numpy ndarray, pandas DatetimeIndex, default None`. Un `str` es el código de país que entiende `holidays.country_holidays` (ISO 3166-1 alpha-2 en mayúsculas, como `'ES'`). Cualquier otra colección son fechas propias.
  - `holidays_kwargs : dict, default None`. Argumentos para `holidays.country_holidays`. Sigue el precedente de `spline_kwargs` y evita copiar la firma del paquete parámetro a parámetro.
- **(b)** Parámetros planos por opción: `holidays_country`, `holidays_subdiv`, `holidays_dates`... Más descubribles, pero cada opción nueva del paquete (`observed`, `categories`, `language`) sería un parámetro más.
- **(c)** Aceptar el objeto de `holidays` (`holidays.country_holidays('ES')`). Descartada: el objeto se rellena al consultarlo (es estado mutable dentro de un parámetro), su `repr` pasa de una línea a 651 caracteres tras la primera consulta, y recién creado está vacío, así que recorrerlo como colección de fechas daría cero festivos sin error.

Reglas de (a):

- Un `dict` en `holidays` (lo es el objeto del paquete) da `TypeError` con la indicación de pasar el código de país o `list(objeto.keys())`.
- `holidays_kwargs` no admite `country` ni `years`: los pone skforecast (`ValueError`).
- `holidays_kwargs` con fechas propias se ignora con `IgnoredArgumentWarning`.
- Los parámetros se guardan tal cual, sin convertir (`clone` de scikit-learn exige que `__init__` no los modifique).

### D3. Cómo se declara la dependencia

- **(a) Propuesta:** extra nuevo `holidays = ["holidays>=0.63"]` en `pyproject.toml`, y el mismo requisito en `all` y en `test`. Entrada en el diccionario `optional_dependencies` de `skforecast/utils/utils.py`, para usar `check_optional_dependency("holidays")` (mensaje estándar con `pip install`). Sin tope superior: el paquete publica una versión `0.x` cada pocas semanas y un tope caducaría enseguida.
- **(b)** Sin extra, como `skops` y `cloudpickle`: `ImportError` con un mensaje propio y el requisito solo en `test`.

El import va dentro de la función que resuelve las fechas. No se importa al importar skforecast ni cuando `holidays` es una lista de fechas.

### D4. Versión mínima

**Propuesta: `holidays>=0.63`.** Es la primera con los atributos `start_year` y `end_year`, que hacen falta para D5 (0.59 y 0.61 no los tienen; 0.63 sí). El prototipo da resultados idénticos a la consulta directa en 0.63, 0.65, 0.75, 0.85, 0.95 y 0.109. Como `test` lleva el requisito, `unit-tests-min-deps.yml` (resolución `lowest-direct`) prueba esa versión mínima en la CI.

### D5. Años sin cobertura

El paquete devuelve cero festivos, sin error, para los años que no cubre, y el rango depende del país: España empieza en 2008 y Alemania en 1991 (versión 0.109; sección 5). Una serie española que empiece en 2005 tendría tres años con la columna a cero.

**Propuesta:** aviso (categoría en D8) cuando algún año del índice quede fuera de `[start_year, end_year]` del calendario, con los años afectados y el rango cubierto. Con fechas propias, el mismo aviso cuando algún año del índice quede fuera del rango de años de la lista: es el caso de predecir 2026 con una lista que acaba en 2025. Alternativa: `ValueError`. Se propone aviso porque un tramo antiguo sin festivos puede ser aceptable y un error impediría entrenar.

### D6. Versión distinta del paquete entre el ajuste y la predicción

El calendario de un mismo país cambia entre versiones del paquete (sección 5: España en 1990 tiene 10 festivos con la 0.75 y ninguno con la 0.85). Un forecaster guardado y cargado en otro entorno generaría features distintas a las del entrenamiento.

**Propuesta:** `fit` guarda la versión en `holidays_version_` (`None` con fechas propias) y `transform` avisa si la versión instalada es otra. `transform` sin `fit` previo (la clase es sin estado) no comprueba nada. La documentación recomienda las fechas propias cuando se necesita reproducibilidad exacta.

Alternativa descartada: resolver las fechas en `fit` y guardarlas. Solo cubriría los años vistos al entrenar y la predicción de un año posterior saldría sin festivos.

### D7. Frecuencias más gruesas que la diaria

El indicador mira la fecha de cada marca de tiempo. En datos semanales o mensuales solo diría si el día de la etiqueta es festivo, que casi nunca es lo que se quiere.

**Propuesta:** se calcula igual, se documenta y se avisa cuando la frecuencia del índice se conoce y es más gruesa que la diaria (ni `Tick` ni `Day` ni día laborable). Si el índice no tiene frecuencia no se avisa. La feature útil para esos datos, el número de festivos del periodo, queda para la versión 2. Alternativa: `ValueError`.

### D8. Nombre de la columna y categoría de los avisos

- **Columna. Propuesta: `holiday`**, coherente con `weekend`. Si el usuario ya tiene una exógena con ese nombre, salta el error de nombres duplicados que ya existe (en la función, el de columnas que se solapan; en los forecasters, el de nombres repetidos en `X_train`). Alternativa: `is_holiday`.
- **Avisos de D5, D6 y D7. Propuesta:** una categoría nueva, `HolidayCalendarWarning`, en `skforecast/exceptions`, para poder silenciar los tres con un filtro. Ninguna de las existentes encaja (`FeatureOutOfRangeWarning` es de detección de drift; `SkforecastVersionWarning`, de la versión de skforecast). Alternativa: `UserWarning` sin categoría propia.

## 4. Comportamiento de la versión 1

Con las propuestas de la sección 3.

| Aspecto | Comportamiento |
|---|---|
| Valor | `int`, 1 si la fecha de la marca de tiempo está en el calendario y 0 si no. Nunca hay valores ausentes |
| Datos intradiarios | Todas las marcas del día festivo valen 1 |
| Zona horaria | Se usa la fecha local del índice, sin pasar a UTC (`2024-12-24 23:30` UTC visto desde Madrid es el 25 y vale 1). Un índice sin zona horaria se usa tal cual. Los años que se piden al paquete son los de las fechas locales |
| Codificación | Nunca se codifica, como `weekend`. Si está en `features_to_encode`, el `IgnoredArgumentWarning` actual ya lo cubre (no está en `max_values` ni en las categorías conocidas) |
| Orden de columnas | El actual: las no codificadas primero, en el orden de `features` |
| Fechas propias | Se aceptan cadenas, `date`, `Timestamp` (con hora o con zona horaria: se queda la fecha), `numpy.datetime64` y `DatetimeIndex`. Duplicados permitidos. Una fecha ausente o que no se puede convertir da `ValueError` |
| Opciones del país | `holidays_kwargs` se pasa sin interpretar. Los errores del paquete (subdivisión o categoría inexistente, argumento desconocido) se dejan pasar: su tipo cambia entre versiones (`TypeError` en 0.95 y `ValueError` en 0.109 para un argumento desconocido) |
| País inexistente | El paquete da `NotImplementedError: Country XX not available`, también para `'es'` en minúsculas. skforecast lo convierte en un `ValueError` que nombra el parámetro `holidays` y remite a `holidays.list_supported_countries()` |
| Cuándo se valida | En `__init__`: tipos, reglas de D1 y D2 y, si `holidays` es un `str`, que el paquete esté instalado y el país exista (se crea el calendario sin años, menos de 1 ms). Al cargar un forecaster guardado no se ejecuta `__init__`: si falta el paquete, el `ImportError` sale en `transform` |
| `fit` | Sigue calculando los nombres con las 2 primeras filas. Además guarda `holidays_version_`. No emite el aviso de D5 (lo emite `transform`, sobre el índice completo) |
| Multiserie | Las features de calendario se calculan una vez y se comparten entre todas las series: un solo calendario para todas. Para un calendario por serie, el festivo va como exógena por serie, y la función sirve para construirla: `create_calendar_features(index, features=['holiday'], holidays=pais, encoding=None)` |
| Avisos del propio paquete | Se dejan pasar (India avisa de que sus festivos solo están disponibles de 2001 a 2035) |
| Caché | Ninguna: resolver las fechas y marcar 24 pasos tarda menos de 0,5 ms |

## 5. Comprobaciones hechas

Script: `scripts/holidays_facts.py`; salida: `scripts/holidays_facts_output.txt`. Entorno: Python 3.13.18, `holidays` 0.109, pandas 2.3.3, numpy 2.5.5, Windows. Repetido con pandas 3.1.1 (mismos resultados) y con `holidays` 0.45, 0.57, 0.63, 0.65, 0.75, 0.85 y 0.95 instaladas en una carpeta aparte (`pip install --no-deps --target`). Los tiempos son de una máquina sin fijar a un núcleo: orientativos.

**Paquete.** Licencia MIT. Única dependencia: `python-dateutil` (ya la instala pandas). Requiere Python 3.11 o superior (0.109), el mínimo de skforecast. 250 países sin contar alias. El import, después de pandas, tarda de 51 a 60 ms.

**Método de cálculo.** Pedir al paquete los festivos de los años del índice y marcar con `numpy.isin` sobre las fechas (`datetime64[D]`) da el mismo resultado que consultar marca a marca en los 7 índices probados: diario, horario, horario con zona horaria, horario creado en UTC y convertido, 15 minutos con zona horaria, días laborables y sin frecuencia. Igual con pandas 2 (`ns`) y pandas 3 (`us`).

| Operación | Tiempo |
|---|---|
| Resolver los festivos de 1 año (ES, US, DE, IN, CN, SA) | de 0,17 a 0,85 ms |
| Resolver los festivos de 10 años | de 0,84 a 2,71 ms |
| Marcar 87 672 horas (10 años) | 4,7 ms |
| Consultar esas 87 672 horas marca a marca | 154 ms |
| `transform` de 24 pasos (resolver y marcar) | 0,46 ms |

**Años cubiertos** (0.109): cero festivos, sin error ni aviso, fuera de `[start_year, end_year]`.

| País | `start_year` | `end_year` | Festivos en 1990 | En 2024 | En 2150 |
|---|---|---|---|---|---|
| ES | 2008 | 2100 | 0 | 9 | 0 |
| DE | 1991 | 2100 | 0 | 9 | 0 |
| US | 1777 | 2100 | 11 | 11 | 0 |
| GB | 1872 | 2100 | 6 | 6 | 0 |
| JP | 1949 | 2099 | 19 | 21 | 0 |

**El calendario cambia entre versiones del paquete:**

| Versión | `start_year` de ES | Festivos de ES en 1990 | En 2040 | `start_year` de DE |
|---|---|---|---|---|
| 0.57 | no existe | 10 | 10 | no existe |
| 0.63 a 0.75 | 1901 | 10 | 10 | 1990 |
| 0.85 | 2008 | 0 | 9 | 1990 |
| 0.95 y 0.109 | 2008 | 0 | 9 | 1991 |

Las versiones anteriores a la 0.63 no tienen `start_year` ni `end_year` y devuelven festivos extrapolados para cualquier año (España: 10 en 1900 y 10 en 2150 con la 0.57). La 0.45 y la 0.57 fallan además con un `TypeError` interno con China en alguno de los años probados (de 1900 a 2150).

**Opciones que cambian el resultado.** `observed` (por defecto `True`): en Estados Unidos, el lunes 2021-07-05 es festivo con `observed=True` y no lo es con `False`. `subdiv`: España tiene 9 festivos nacionales en 2024, 12 con Madrid y 13 con Cataluña. `categories`: por defecto solo `public`; Estados Unidos admite además `government`, `half_day` y `unofficial`, y Alemania `catholic` y `school`.

**Guardar con `skops`.** Un parámetro `str`, `dict`, lista de cadenas o array de `numpy.datetime64` se guarda sin tipos que confirmar. Una lista de `datetime.date` o de `Timestamp` se guarda, pero al cargar hay que confiar en esos tipos. Un `DatetimeIndex` no se puede guardar (`TypeError`). Consecuencia: la documentación recomienda las fechas propias como lista de cadenas, y hay que decidir al implementar si `save_forecaster(backend='skops')` da un error claro con un `DatetimeIndex` en `holidays`.

## 6. Implementación

Sin cambios en los forecasters (con D1 (a)).

1. **`skforecast/preprocessing/_calendar.py`**
   - Dos funciones privadas, con el prototipo de `scripts/holidays_facts.py` como especificación: `_resolve_holiday_dates(holidays, holidays_kwargs, years)` devuelve las fechas ordenadas y sin duplicados como `datetime64[D]` (import de `holidays` dentro), e `_is_holiday(index, holiday_dates)` devuelve el array de 0 y 1. Los años son el mínimo y el máximo de las fechas locales del índice.
   - Una función privada de validación, común a la función pública y a la clase, con las reglas de D1 y D2.
   - `create_calendar_features`: parámetros `holidays` y `holidays_kwargs`, `'holiday'` en `datetime_attrs`, y los avisos de D5 y D7.
   - `CalendarFeatures`: los mismos parámetros, `'holiday'` en `allowed_features`, validación en `__init__`, `holidays_version_` en `fit` y el aviso de D6 en `transform`.
   - Docstrings de los dos, incluida la sección `Notes` (fecha local, intradiario, frecuencias gruesas, multiserie, reproducibilidad).
2. **`skforecast/exceptions/`**: `HolidayCalendarWarning` (D8), exportada y en la referencia de la API.
3. **`pyproject.toml`** y **`skforecast/utils/utils.py`**: extra `holidays`, `all`, `test` y `optional_dependencies` (D3). Añadir `holidays` a la lista de `show_versions`.
4. **`tools/ai/ai_context_header.md`**: la línea "Optional" de dependencias.

## 7. Tests

En `skforecast/preprocessing/tests/tests_calendar/`. Los valores esperados con un país se fijan con fechas que no cambian entre versiones del paquete (1 de enero y 25 de diciembre de un año reciente), no con el número de festivos de un año.

- **Valores:** país, país con `subdiv`, `observed=False`, y fechas propias en cada tipo admitido, con el mismo resultado en todos.
- **Índices:** diario, horario, con zona horaria, creado en UTC y convertido (incluido el cambio de año: `2024-12-31 23:30` UTC en Madrid), días laborables, sin frecuencia y desordenado. Con pandas 2 y pandas 3.
- **Integración con la clase:** `features=None` con y sin `holidays`; orden de columnas con cada `encoding`; `holiday` nunca se codifica; `get_feature_names_out`; `clone`; `transform` sin `fit`.
- **Errores y avisos:** cada regla de D1 y D2; país inexistente; fecha ausente; `dict` en `holidays`; años fuera de cobertura (país y fechas propias); frecuencia semanal y mensual; versión distinta (cambiando `holidays_version_` a mano); paquete no instalado (simulado).
- **Forecasters:** un test por cada uno de los cuatro que comprueba que `holiday` está en `X_train` y que `predict` da lo mismo que pasar esa columna como exógena.
- **Guardar y cargar:** `save_forecaster` y `load_forecaster` con `joblib` y con `skops`, con país y con lista de cadenas.
- **`test_check_optional_dependency.py`:** ya compara `optional_dependencies` con `pyproject.toml`; debe seguir pasando.

## 8. Documentación y contexto de IA

- `docs/user_guides/calendar-features.ipynb`: sección nueva de festivos, antes de "Distance to holidays". Incluye la tabla de qué calendario usar (país, subdivisión, fechas propias), los avisos y la receta multiserie.
- `docs/user_guides/migrate-from-prophet.ipynb`: la fila de la tabla de equivalencias de `add_country_holidays`.
- `docs/quick-start/how-to-install.md`: el extra nuevo.
- `docs/releases/releases.md`: nota en la versión en desarrollo (skill `release-note`).
- Contexto de IA (skill `ai-context-sync`): `tools/ai/llms-base.txt`, `skills/feature-engineering/SKILL.md` y `references/calendar-features-reference.md`, `skills/complete-api-reference/references/preprocessing-signatures.md`; después, regenerar.
- `verify` antes de dar el cambio por terminado.

## 9. Versión 2 (ideas, sin planificar)

- **Distancia al festivo** (`'time_to_holiday'`, `'time_since_holiday'`). No sirve aplicar `calculate_distance_from_holiday` a la columna del indicador: solo ve los festivos que caen dentro del índice, así que en una predicción de 7 días sin festivos devuelve `fill_na` (0 por defecto, el mismo valor que en un festivo). Hay que calcular la distancia desde las fechas del calendario, pidiendo también el año anterior y el siguiente, y compartir ese núcleo (`searchsorted`) con la función actual. La unidad (días, o la de la frecuencia como hace hoy la función) está por decidir; ISSUE-1507 toca ese mismo punto.
- **Vísperas, días posteriores y puentes**, como features con nombre.
- **Número de festivos del periodo** para datos semanales y mensuales (D7).
- **Mercados financieros** (`holidays.financial_holidays`).
- **Un calendario por serie** en multiserie.
- **Nombre del festivo:** no cabe hoy, porque las features de calendario entran como matriz numérica.

## 10. Riesgos

| Riesgo | Mitigación |
|---|---|
| Columna a cero sin aviso en años sin cobertura | D5 |
| Features distintas tras cargar el forecaster en otro entorno | D6; fechas propias para reproducibilidad exacta |
| Indicador sin sentido en datos semanales o mensuales | D7 |
| Calendario único en multiserie con series de varios países | Documentado; receta con exógena por serie (sección 4) |
| Dependencia nueva | Opcional, import diferido, sin coste si no se usa; versión mínima probada en la CI (D4) |
| Tests que dependen de datos del paquete | Fechas estables como valores esperados (sección 7) |
