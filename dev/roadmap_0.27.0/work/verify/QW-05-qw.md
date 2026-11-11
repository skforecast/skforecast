# QW-05: save_forecaster no debe guardar los pesos del modelo foundation cargados tras un predict

**Veredicto: keep_modified** (esfuerzo S: unas 30 líneas de código, un test de round-trip y nota de versión). Es más que un quick win de tamaño: con TimesFM 2.5 el guardado tras un predict **falla** con los backends por defecto.

## Qué existe hoy

- Los 9 adaptadores guardan el backend como atributo (`_adapters.py:233,743,1213,1866,2316,2965,3567,4082,4601`) y lo cargan de forma perezosa en `_load_pipeline`/`_load_model`/`_load_module` (`_adapters.py:401,931,1645,2022,2535,3194,3739,4255,4842`) en el primer `predict`.
- Hay además cachés puras que tampoco se excluyen: `MoiraiAdapter._forecast_obj` (envuelve el módulo, `_adapters.py:1870`) y `TSICLAdapter._resolved_device` (`_adapters.py:4083`, se fija a `"cuda"` o `"cpu"` y viajaría en el pickle a otra máquina).
- Ningún `__getstate__`, `__setstate__` ni `__reduce__` en `skforecast/foundation/` (grep vacío). `save_forecaster` hace `joblib.dump` directo (`utils.py:3476`); skops no se admite para foundation (`utils.py:3490-3500`).
- `get_params` de los adaptadores ya excluye el backend a propósito, así que `clone` y `deepcopy_forecaster` (`utils.py:4999`) son ligeros. Solo el pickle arrastra los pesos.
- Nada en docs, `tools/ai/llms-base.txt` ni `skills/foundation-forecasting` explica qué se guarda. No está en la lista de identificados. Relacionado con BUG-04 (objeto precargado perdido en `clone`) y PLAN-CKPT.

## Reproducción del "before" (skforecast 0.26.0, Python 3.12, torch 2.14.1+cpu, chronos-forecasting 2.3.2, joblib 1.6.0)

Scripts en `/tmp/claude-0/-home-user-skforecast/8505cf39-716d-5c4c-adf6-669c7ec65b16/scratchpad/qw05/` (`before.py`, `timesfm_fake.py`, `proto_fix.py`).

| Caso | Medido |
|---|---|
| Chronos-2 small, guardar tras `fit` | 0,01 MB |
| Chronos-2 small, guardar tras `predict` | **111,81 MB** (confirmado) |
| Cargar el fichero de 112 MB | 4,25 s |
| TimesFM 2.5 (paquete `timesfm` simulado), guardar tras `predict`, `joblib` | **`PicklingError`**: `_TimesFMCompat` es una clase local de `_load_model` (`_adapters.py:970-975`) |
| Ídem con `backend="pickle"` | `AttributeError: Can't pickle local object` |
| Ídem con `backend="cloudpickle"` | OK |
| Fichero que deja el fallo | `.joblib` de 2 bytes y `.pkl` de 0 bytes (corruptos) |

La fricción es real y el flujo es el normal: `backtesting_foundation` usa el forecaster original sin copia (`_validation.py:2373`), así que tras validar con backtesting el forecaster ya lleva los pesos y el guardado posterior pesa cientos de MB (Chronos-2 base, TimesFM o TabPFN son mayores) o, con TimesFM 2.5, falla. Además, un pickle de tensores en GPU no se puede cargar en una máquina sin CUDA (comportamiento conocido de torch, no medido aquí): entrenar en GPU y servir en CPU se rompe.

## Prototipo (monkeypatch, sin tocar el paquete)

`__getstate__` en `_AdapterBase` que, si el backend se cargó de forma perezosa (flag puesto en el `_load_*`), guarda `_pipeline`/`_model`/`_module` como `None`; las cachés `_forecast_obj` y `_resolved_device` se vacían siempre.

| Medida | Resultado |
|---|---|
| Tamaño tras `predict` | 0,011 MB (save 0,002 s) |
| Original en memoria | conserva el pipeline (no se recarga) |
| `load_forecaster` + primer `predict` (recarga desde la caché de HF) | 0,001 s + 0,33 s, frente a 4,25 s del fichero pesado |
| Predicciones tras cargar | idénticas (`np.allclose` True) |
| `FoundationModel(pipeline=objeto_del_usuario)` | se conserva (111,81 MB), sin pérdida silenciosa |

## Ajustes respecto a la propuesta

1. **Conservar el objeto pasado por el usuario.** Vaciarlo cambiaría en silencio un modelo ajustado por el base al cargar (mismo fallo que BUG-04). Usar un flag de carga perezosa en los 9 `_load_*` (una línea cada uno) en lugar de un flag "del usuario" en `__init__`: así `set_params(model_id=...)` seguido de recarga queda bien marcado sin tocar los resets. Compartir con BUG-04 la declaración de qué atributo es el backend.
2. Vaciar también `_forecast_obj` (Moirai) y `_resolved_device` (TS-ICL, para que `device="auto"` se resuelva en la máquina de destino).
3. Presentarlo también como **Fixed**: "save_forecaster fallaba con TimesFM 2.5 después de predecir". El `__getstate__` lo resuelve sin mover `_TimesFMCompat`.
4. Implementación centralizada en `_AdapterBase` (es un cambio del contrato, encaja en la regla de `foundation.md`), sin lógica por adaptador.

## Riesgos de API

- **Cambio de comportamiento**: el fichero deja de ser autocontenido. Quien hoy despliega sin internet ni caché de HF y, por casualidad, guardaba tras predecir, pasará a descargar los pesos en el primer `predict`. Es poco probable que alguien dependa de ello (solo ocurre si se predijo antes de guardar y no está documentado), pero hay que decirlo en la nota de versión y en la guía `save-load-forecaster.ipynb`: precargar la caché de HF (`HF_HOME`, `HF_HUB_OFFLINE=1`) o pasar el objeto precargado.
- Ficheros guardados con versiones anteriores (con pesos) se siguen cargando: no hace falta `__setstate__` (`getattr` con valor por defecto para el flag).
- `copy.deepcopy` de un adaptador también vaciaría el backend cargado y la copia lo recargaría. No hay usos internos sobre foundation (`deepcopy_forecaster` usa `clone`); aceptable.
- Sin parámetros nuevos ni columnas de salida nuevas. No cambia ninguna firma pública; basta una línea en `skills/foundation-forecasting` sobre el guardado (opcional).

## Esfuerzo

`_adapter_base.py` (+15 líneas), `_adapters.py` (9 líneas en los cargadores), test de round-trip con el `FakePipeline` del fixture (tamaño, atributo `None` tras cargar, predicción igual, objeto del usuario conservado) y nota de versión. Unas 2 a 4 horas.
