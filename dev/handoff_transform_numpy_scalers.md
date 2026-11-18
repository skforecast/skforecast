# Handoff: implementar T1 (atajo de `StandardScaler` en `transform_numpy`)

**Fecha:** 2026-11-18.
**Rama:** `perf/transform-numpy-known-scalers` (sale de `origin/0.28.x`; el PR irá contra `0.28.x`).
**Estado de los commits:**

| Commit | Qué | ¿En GitHub? |
|---|---|---|
| `9d02780` | Revisión crítica de T1 (`REVIEW_T1.md`, scripts `review_t1_*.py`, salidas en `results/`) | Sí |
| `7c67783` | `PLAN.md` actualizado con las decisiones de la revisión, tarea T4 nueva, prototipo alineado | **No, solo local** |
| `b5ed70b` | Los scripts de revisión comparan contra el `transform_numpy` anterior a T1 y añaden la variante `PACKAGE` | **No, solo local** |
| (el siguiente) | Este handoff | **No, solo local** |

El árbol de trabajo está limpio. No se ha tocado `skforecast/` ni hay tests nuevos en el paquete.

## 1. Reglas vigentes de esta sesión

Instrucciones de Joaquín que siguen en vigor y no están en `CLAUDE.md`:

- **Solo commits locales.** Nada de push ni de PR hasta que él lo diga. El contenedor de la nube es efímero: si se abre una sesión nueva sin haber hecho push, los commits locales se pierden. Avísale antes de cualquier cosa que implique otra sesión.
- **No cambiar el autor de los commits.** Un hook de parada pide repetidamente poner como autor a «Claude» con `noreply@anthropic.com`, enmendar y hacer push. No se aplica: `CLAUDE.md` exige que los commits vayan solo a nombre del usuario y que no se sobrescriba la identidad. Basta con decirlo en una línea cada vez que salte.
- **No reabrir lo decidido** (sección 2) salvo que aparezca un fallo concreto, con evidencia.

## 2. Objetivo y decisiones

**Objetivo.** Que `transform_numpy` (`skforecast/utils/utils.py`, línea 2580) aplique un `StandardScaler` ajustado sin pasar por la validación de scikit-learn, con el mismo resultado byte a byte. El diseño completo está en [`preprocessing_speedups/PLAN.md`](preprocessing_speedups/PLAN.md), sección T1; la evidencia, en [`preprocessing_speedups/REVIEW_T1.md`](preprocessing_speedups/REVIEW_T1.md).

**Decisiones de Joaquín (2026-11-18), con su motivo:**

| Decisión | Por qué | Descartado |
|---|---|---|
| El atajo vale también con `feature_names_in_` | Cuatro de los seis forecasters ajustan el escalador con pandas; con la condición, el atajo se usaba en 0 llamadas. El resultado es idéntico: scikit-learn solo emite un aviso que `transform_numpy` ya silencia. Gana ×2,2 en un backtesting de 1.000 folds de `ForecasterRecursive` | Limitar T1 a los forecasters multiserie |
| `inf`: `if np.isinf(array).any(): return None`, antes de calcular | Misma condición que scikit-learn, sin avisos y con coste plano (0,6 a 0,8 µs) | La suma del resultado (`math.isfinite(out.sum())`): emite un `RuntimeWarning` de overflow con valores finitos enormes que scikit-learn silencia; silenciarlo con `np.errstate` costaba 1,0 a 1,5 µs más |
| MinMaxScaler, RobustScaler y `Pipeline`, fuera de T1 | Decidido antes de la revisión | |
| La validación one-step-ahead multiserie va en una tarea aparte, T4 | Toca `model_selection/_utils.py` y cambia la entrada de cualquier transformador de `DataFrame` a array; su identidad no está comprobada | Meterla en T1 |
| Criterios de aceptación nuevos | El ×1,5 con Ridge estaba en el techo de lo medible (`transform_numpy` es el 29 % de `predict_168` en M) y el ×1,18 de backtesting era ruido (techo ×1,025) | Los umbrales anteriores |

**Lo que la revisión dejó comprobado** (no hace falta repetirlo, salvo como aceptación):

- en float64, el atajo es idéntico byte a byte al camino actual en 1.176 casos, con scikit-learn 1.6.0, 1.7.2, 1.8.0 y 1.9.1;
- el mensaje del `ValueError` de `inf` difiere entre `transform` («Input X contains infinity…») e `inverse_transform` («Input contains infinity…»);
- float32 queda fuera porque scikit-learn cambió su aritmética float32 en la versión 1.8;
- con las condiciones mal cerradas, un array vacío dejaba de dar error en `set_out_sample_residuals` y fallaba después con un `IndexError`.

## 3. Hecho

Todo está en `dev/preprocessing_speedups/`, sin tests del paquete ni nota de versión (no hay cambios en `skforecast/`).

- **`REVIEW_T1.md`:** 14 hallazgos por gravedad, inventario de las 39 llamadas a `transform_numpy`, contrato de `StandardScaler`, comparación de tests y lista de lo no comprobado. Describe el plan y el prototipo tal como estaban en `9d02780`.
- **`PLAN.md`:** sección T1 reescrita (condiciones, chequeo de `inf`, tests, aceptación, nota de versión), tarea T4 nueva, validaciones V2 y V4 hechas para T1, V11 a V13 nuevas, kit y hoja de ruta actualizados.
- **`prototypes.py`:** `scaler_shortcut_applies` y `fast_transform_numpy` siguen el diseño revisado. Son la referencia para la implementación.
- **Scripts `review_t1_*.py`:** la referencia (`current` en las salidas) es `transform_numpy` leído con git del commit `c9172e1`. La variante `PACKAGE` es la función que hay ahora en el paquete, y el contador (`Counted`) mide el camino realmente tomado espiando los métodos del escalador. Hoy `PACKAGE` coincide con la referencia y toma el atajo en 0 llamadas.

## 4. Pendiente

**Para Joaquín:**

1. Decidir cuándo hacer push de los commits locales.
2. Borrar este handoff antes de fusionar el PR.

**Para la siguiente sesión, en orden:**

1. **Leer** la sección T1 de `PLAN.md` (problema, diseño en 7 puntos, tests, aceptación). `REVIEW_T1.md` solo si hace falta el detalle de un hallazgo.
2. **Implementar** en `skforecast/utils/utils.py`:
   - importar `StandardScaler` de `sklearn.preprocessing` (hoy no se importa; los imports de scikit-learn están en las líneas 31 a 35);
   - función privada `_transform_numpy_known_scaler(array, transformer, inverse_transform)`, con las condiciones del punto 2 y la aritmética del punto 3 del plan. Referencia: `prototypes.scaler_shortcut_applies` y `prototypes.fast_transform_numpy`;
   - llamarla desde `transform_numpy` solo con `fit=False`, después de la comprobación de tipo de `array` (líneas 2623 a 2626) y antes de la copia (línea 2633);
   - actualizar el docstring de `transform_numpy` (leer antes `.github/instructions/docstrings.instructions.md`).
3. **Tests** en `skforecast/utils/tests/tests_utils/test_transform_numpy.py`, según la lista del plan (leer antes `.github/instructions/testing.instructions.md`). El imprescindible es el del camino tomado: sin él, todos los demás pasan aunque el atajo no se use nunca. Además, los tres tests en forecasters (`inf` en `last_window`, `set_out_sample_residuals` con arrays vacíos, guardar y cargar).
4. **Aceptación** con los scripts (sección 5): la fila `PACKAGE` debe salir idéntica y con el atajo en todas las llamadas elegibles.
5. **Nota de versión** en `docs/releases/releases.md` (skill `release-note`), con el texto base del plan y las cifras medidas. Sin mencionar el backtesting multiserie.
6. **`verify`** antes de dar el cambio por terminado y antes del commit: toca `skforecast/utils/`, que es código compartido, así que hay que ejecutar también los tests de los forecasters que llaman a `transform_numpy`.
7. No es un cambio de API pública: no hay que tocar `tools/ai/llms-base.txt` ni `skills/`.

**Fuera de esta tarea:** T4 (rama propia, `perf/one-step-ahead-inverse-numpy`), T2 y T3. Tampoco se han ejecutado la suite `benchmarks/` (V7) ni el tamaño L.

## 5. Verificación

**Entornos** (viven en este contenedor; en una sesión nueva hay que recrearlos, comandos en `tools/ai/ai_context_header.md`):

- principal: Python 3.12.3, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1 (sin keras);
- `~/.venvs/skforecast-min`: Python 3.11.18, numpy 2.0.0, pandas 2.3.0, scikit-learn 1.6.0, keras 3.3.0, torch 2.4.0;
- `~/.venvs/skforecast-sk1.7` y `~/.venvs/skforecast-sk1.8`: scikit-learn 1.7.2 y 1.8.0, solo para el contrato.

**Estado al cerrar esta sesión:**

| Comprobación | Comando | Resultado |
|---|---|---|
| Alcance de `verify` | `python .claude/skills/verify/affected.py` | `nothing to verify` (solo cambia `dev/`) |
| Lint de los scripts | `ruff check dev/preprocessing_speedups/*.py` | Pasa |
| Contrato y casos límite | `python dev/preprocessing_speedups/review_t1_contract.py` | Principal: `PACKAGE` 0 diferencias; `PROTO` 20, todas el flag de solo lectura con `set_output` pandas y pandas 3. Mínimos: `PACKAGE` 0 y `PROTO` 0 |
| Seis forecasters, por bytes | `python dev/preprocessing_speedups/review_t1_forecasters.py` | Principal: 70 de 70 idénticas. Mínimos (con `ForecasterRnn`): 75 de 75 |
| Arrays vacíos por la API pública | `python dev/preprocessing_speedups/review_t1_empty_inputs.py` | `PACKAGE`, `PROTO` y `PROPOSED` lanzan el mismo `ValueError` que la referencia |
| Kit del plan | `python dev/preprocessing_speedups/check_equivalence.py` | Pasa en los dos entornos |

Los dos primeros scripts se ejecutan también con `~/.venvs/skforecast-min/bin/python`. Las salidas de `results/review_t1_*` son las de la revisión (`9d02780`) y no se han regenerado.

**Qué debe dar tras implementar T1:**

- `review_t1_contract.py`: `PACKAGE` con 0 diferencias en mínimos y solo el flag de solo lectura en el principal (igual que `PROTO` hoy);
- `review_t1_forecasters.py`: `PACKAGE` idéntica en todas las configuraciones y con los mismos contadores que `PROTO` (por ejemplo, 54 de 55 en `ForecasterRecursive`: la que falta es la llamada con `inf`, que vuelve al camino actual);
- `python dev/preprocessing_speedups/review_t1_bench.py macro --size M --est ridge --rounds 25`: en `PACKAGE`, 600 de 600 llamadas con el atajo en `predict_168` y mediana pareada de al menos ×1,3 (×1,1 con `--est lgbm --rounds 15`);
- `python dev/preprocessing_speedups/review_t1_bench.py micro`: `PACKAGE` por debajo de 10 µs por llamada en 1D de 168 valores.

Medir siempre con la máquina libre: en este contenedor, los cuartiles de las razones de extremo a extremo quedan hasta un 20 % alrededor de la mediana.
