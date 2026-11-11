# BUG-04: el pipeline o modelo precargado se descarta al crear ForecasterFoundation

**Veredicto: keep (reproducido; resultado erróneo y silencioso; esfuerzo S).**

Script: `dev/roadmap_0.27.0/repro/BUG-04.py` (parte A sin red con pipelines falsos; parte B con Chronos-2 small real ajustado 30 pasos con `BaseChronosPipeline.fit`). skforecast 0.26.0, Python 3.12.3, chronos-forecasting 2.3.2, torch 2.14.1+cpu, scikit-learn 1.9.1.

## Resultado

| Comprobación | Observado | Esperado |
|---|---|---|
| `FoundationModel(pipeline=ft).predict` | 999 (usa `ft`) | 999 |
| `ForecasterFoundation(fm).estimator.adapter._pipeline` | `None` | `ft` |
| `ForecasterFoundation.predict` | 0 (recarga el checkpoint base, 1 recarga, 0 avisos) | 999 |
| `backtesting_foundation` | todas las predicciones = 0 | 999 |
| `deepcopy_forecaster` (lo que llama `bayesian_search_foundation`) tras el parche manual del fixture | `None` | `ft` |
| Chronos-2 real: FoundationModel ajustado | `[10.231 9.79 9.795 9.487 9.152 10.464 10.704]` | |
| Chronos-2 real: ForecasterFoundation | `[12.083 12.797 11.061 8.647 6.958 7.668 9.977]` = modelo base exacto | = ajustado |

Diferencia máxima ajustado frente a base: 3.0 unidades en una serie de nivel ~10.

## Causa raíz

- `skforecast/foundation/_forecaster_foundation.py:156`: `self.estimator = clone(estimator)`.
- `skforecast/foundation/_foundation_model.py:1295-1321`: `get_params` delega en `adapter.get_params()`, que excluye a propósito `pipeline`/`model`/`module` para no copiar pesos; `clone` reconstruye el adaptador con `_pipeline=None` y `_load_pipeline` (`_adapters.py:421-436`) recarga `model_id`.
- `skforecast/utils/utils.py:4999`: `deepcopy_forecaster` vuelve a clonar (afecta a `bayesian_search_foundation`, `_search.py:2247`).
- El fixture de tests lo sabe (`fixtures_forecaster_foundation.py:196-211`) y asigna el atributo privado después de construir; la documentación pública no lo advierte: docstrings de los 9 adaptadores (`_adapters.py:185,703,1175,1835,2267,2908,3532,4043,4542`) y `skills/foundation-forecasting/references/adapter-parameters.md:38` presentan el parámetro como utilizable.

## ¿Lo encuentra un usuario normal?

Nicho, pero no rebuscado: falla en el 100% de los usos de un parámetro documentado. El caso grave es el fine-tuning (Chronos-2 trae `.fit()` de serie): el usuario evalúa con `backtesting_foundation` y obtiene las métricas del modelo base creyendo que son del ajustado. No hay ninguna vía pública alternativa (`_resolve_adapter` exige el prefijo de Hugging Face). Caso menor: quien pasa un pipeline ya cargado con dtype o dispositivo propios pierde esa configuración y paga una recarga. Precedente: en la 0.26.0 se trató como bug el `get_params` incompleto de Arima/Ets que el clone reseteaba en silencio (`releases.md:221`).

## Arreglo propuesto (prototipo verificado en scratchpad)

Centralizado en `FoundationModel` (sin tocar la lógica de cada adaptador):

1. Cada adaptador declara el nombre del atributo del objeto precargado (`_pipeline` en Chronos, `_module` en Moirai, `_model` en el resto) y guarda si lo pasó el usuario.
2. `FoundationModel.__sklearn_clone__` (scikit-learn >= 1.6 lo soporta): clon estándar por `get_params` y, si el objeto lo pasó el usuario, copiar la **referencia** al clon (sin copiar pesos). Así lo heredan `ForecasterFoundation.__init__`, `deepcopy_forecaster` y cualquier `clone`, sin cambios en esas funciones.
3. `set_params(model_id=...)` en el clon sigue reseteando solo el clon (comprobado: el original queda intacto).
4. Test con `FakePipeline`: predict, backtesting y `deepcopy_forecaster` conservan el objeto; simplificar `make_forecaster` del fixture; nota de versión en Fixed.

Alternativa mínima si no se quiere compartir la referencia: `UserWarning` en `ForecasterFoundation.__init__` cuando el objeto precargado se va a perder. Coordinar con QW-05 (`__getstate__`/guardado) para que también conserve o avise del objeto del usuario. Completa PLAN-CKPT: aunque exista `finetuned_checkpoint`, el objeto precargado debe sobrevivir al clone.
