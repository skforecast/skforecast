# Verificación adversarial: TabICLAdapter

Fecha: 2026-11-06. Rama docs/foundation-adapters-review (desde origin/0.27.x, skforecast 0.27.0).

Entorno: ~/.venvs/fm-review-tabicl (Python 3.12.3, tabicl 2.2.0, torch 2.15.0+cpu, numpy 2.5.4,
pandas 2.3.3, huggingface-hub 2.4.0, scikit-learn 1.9.1). Máquina x86_64 Linux, 4 CPU, sin GPU.
Código upstream: ~/fm-review-src/tabicl (git fetch hecho hoy; HEAD de main d83091b, tags v2.0.2 a v2.2.0).

No se ha ejecutado ninguna medida de velocidad ni de precisión entre versiones o variantes.

## Qué se ha re-ejecutado

| Fichero | Contenido |
|---|---|
| pypi_tabicl.log | Historial de PyPI: última 2.2.0 (2026-09-02), requires_python >=3.10, torch>=2.2, extra forecast sin gluonts |
| hub.log | jingang/TabICL sha 4dcd344, bsd-3-clause, no gated, 4 ckpt; historial de commits; búsqueda "tabicl" en el Hub; soda-inria/tabicl da 404 |
| contract_smoke_2.2.0.log | contrato + smoke con tabicl 2.2.0: 9 passed |
| old_backend_2.0.2.log | entorno recreado con tabicl 2.0.2: contrato 1 failed, 2 passed (TypeError point_estimate) |
| old_backend_2.0.3.log, old_backend_2.0.3_smoke.log | tabicl 2.0.3: contrato 3 passed, smoke 6 passed. Entornos borrados después |
| rerun_compare_direct.log | compare_direct.py del investigador re-ejecutado |
| verify_tabicl.py / verify_tabicl.log | comprobaciones propias en subprocesos limpios (warnings tal como los ve el usuario) |
| upstream_issues.log | búsqueda de issues y PR upstream |
| unit_tests.log | test_TabICLAdapter.py: 50 passed |

Además: log del job de CI 127875169082 (run 42212742430) leído con la API de GitHub: tabicl 2.2.0,
torch 2.15.0+cpu, huggingface_hub 2.4.0, contract 3 passed, smoke 6 passed, held_back false.

## Correcciones importantes

1. **Fila resumen**: no hay "20 commits sin release" en main. `git rev-list --count v2.2.0..origin/main`
   da 2 (c91f00d y d83091b). Los 20 commits son v2.1.1..v2.2.0. [probado]
2. **TAB-F8**: en el caso B de compare_direct.py el tramo NaN de s3 (posiciones 50 a 59 de 200)
   queda fuera de la ventana de 128 filas, así que el caso no ejercita NaN. Además, al re-ejecutarlo
   el caso B da max_abs_diff de hasta 3.8e-6 (allclose True), no 0 exacto. Con una prueba propia con
   10 NaN dentro de la ventana de 64 filas la diferencia es 0.0. La conclusión (el adaptador traduce
   bien) se mantiene. [probado]
3. **TAB-P1**: proto_fixes.py no demuestra la reemisión: lo comprueba con un catch_warnings(record=True)
   externo, que captura el warning también sin el parche (lo muestra la probe 10 de
   capabilities_check.log). Lo he comprobado en un subproceso sin catch_warnings: sin parche no sale
   nada en stderr; con el parche aparece el UserWarning y la barra tqdm sigue oculta. [probado]
   Matices de riesgo: warn_explicit sin registry no deduplica por ubicación (en un backtesting el
   aviso saldría en cada fold, respetando los filtros del usuario); los warnings de los workers de
   loky (varias series en CPU) no pasan por la redirección del proceso padre. [inferido]
4. **TAB-P3**: subir el mínimo en backend_package no da un error más claro a quien ya tiene 2.0.2:
   tabicl.forecast se importa bien, el ImportError del adaptador no salta y el TypeError sigue igual.
   Solo cambia el comando de instalación y la documentación. Para un mensaje claro haría falta una
   comprobación de versión en _load_model (precedente: TimesFM3Adapter). La 2.0.2 solo fue la última
   versión entre el 2026-02-23 y el 2026-03-02, así que el impacto práctico es pequeño. [verificado en código]
5. **TAB-F7 / F18**: llms-base.txt línea 62 es una frase genérica sobre todos los adaptadores; la
   línea 513 ya documenta que los pesos de 'soda-inria/tabicl' vienen de 'jingang/TabICL'. El texto
   impreciso es sobre todo el docstring del adaptador. [verificado en código]

## Confirmado

- 2.0.2 falla en el constructor (output_selection), 2.0.3 pasa; el renombrado está en CHANGES 2.0.3. [probado]
- Firmas iguales en v2.0.3, v2.1.0, v2.1.1 y v2.2.0; `tabicl.forecast` exporta TabICLForecaster en todas.
  CHANGES 2.1.0 (PR #84) recomienda importar desde `tabicl` (nivel superior), pero `tabicl.forecast`
  no lleva prefijo y sigue siendo válido; no hace falta cambiar el import. [verificado en código]
- c91f00d: el diff solo añade tokenizers>=0.20 en dependencias; no hay pin de huggingface_hub. [verificado en código]
- Hub: jingang/TabICL sin cambios de ficheros desde 2026-02-12; los demás repos "tabicl" del Hub son
  de terceros. regressor.py v2.2.0 fija repo_id='jingang/TabICL' y checkpoint por defecto
  tabicl-regressor-v2-20260212.ckpt. [probado]
- Licencia: tarjeta bsd-3-clause; LICENSE de v2.2.0 dice que src/tabicl/forecast es derivado de
  TabPFN-TS bajo Apache-2.0. La issue upstream #83 (licencia de los pesos) está cerrada, pero sus
  comentarios no se pudieron leer (API de GitHub no disponible para ese repo). [probado / documentado]
- Cuantiles int y numpy.float32 fallan con 'Quantiles must be a list of floats'; numpy.float64 pasa
  (subclase de float). Con float() los resultados son idénticos. Los nombres raros de columna con
  float32 ('q_0.10000000149011612') salen del f-string de FoundationModel, no del parche. [probado]
- Covariable de texto: ValueError del backend (ForecastEngine pasa numpy, no DataFrame). [probado]
- Timestamps enteros: se interpretan como nanosegundos desde 1970; el workaround sigue siendo necesario. [probado]
- temporal_features=["index", "datetime"] funciona vía el adaptador; ["fourier"] da TypeError
  (issue #172, PR #174 abierto). [probado]
- set_params(context_length) solo recrea TabICLForecaster; el checkpoint se carga con torch.load
  dentro de TabICLRegressor.fit, una vez por serie y llamada. [verificado en código]
- TQDM_DISABLE: envwrap lee os.environ al decorar tqdm.__init__ (al importar). [verificado en código]

## Omisiones encontradas

- **TAB-V1** (baja): FoundationModel recorta a las últimas context_length filas, NaN incluidos, y
  después tabicl descarta los NaN; el modelo ve menos de context_length observaciones. Una llamada
  directa con la serie completa conserva las últimas max_context_length válidas (diferencia 0.35 en
  la prueba). Es una decisión centralizada de FoundationModel, igual para todos los adaptadores:
  basta con documentarla, si se quiere. [probado]
- **TAB-V2** (baja): el docstring de SUPPORTED_QUANTILES y llms-base.txt línea 492 dicen "(0, 1)",
  pero tabicl acepta 0.0 y 1.0 (y las Notes del propio docstring dicen [0, 1]). Añadir a TAB-P5. [probado]
