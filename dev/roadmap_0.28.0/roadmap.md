# Roadmap de skforecast 0.28.0

Fecha: 2026-11-11. Base: rama `0.27.x` (0.27.0 en desarrollo). Las rutas son relativas a la
raíz del repositorio.

Recoge el mantenimiento de dependencias que se aplazó en la revisión de `pyproject.toml` del
2026-11-11 (sección 1) y todo lo que la 0.27.0 no hizo (sección 3, decisión del 2026-11-11).
La evidencia de los elementos de la sección 3 sigue en
[`dev/roadmap_0.27.0/roadmap.md`](../roadmap_0.27.0/roadmap.md).

Entorno de medida: Python 3.14.3 en macOS arm64, numpy 2.5.4, pandas 2.3.3, scikit-learn
1.9.1, scipy 1.18.1, numba 0.68.1, optuna 5.0.0, statsmodels 0.15.0, torch 2.15.0, keras
3.15.1, lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, skops 0.16.0, pyarrow 26.0.1, pytest
9.2.0. Entornos aislados creados con `uv`, ya borrados. Versiones de PyPI consultadas el
2026-11-11.

Esfuerzo: S (horas a 1 día, con tests y docs), M (2 a 5 días), L (más de una semana).

---

## 1. Mantenimiento de dependencias

| ID | Qué | Esfuerzo | Estado |
|---|---|---|---|
| DEP-python310 | Retirar Python 3.10 | S | Hecho: #1496, fusionada en `0.28.x` el 2026-11-12 |
| DEP-numpy2-pandas23 | Subir los mínimos a numpy 2.0 y pandas 2.3 | S | Implementado el 2026-11-13 en `chore/numpy2-pandas23-minimums` (sin commit; faltan los workflows a mano); antes de la PR 6 de la migración a pandas 3 |
| MAINT-docs-deps | Actualizar los pins del extra `docs`, salvo `mkdocs-jupyter` | S | Opcional, medido: sin efecto visible |
| MAINT-test-deps | Limpiar el extra `test` | S | Opcional |

Son pequeños y no dependen entre sí, salvo `tomli` (MAINT-test-deps), que se quita con
DEP-python310, y `pandas[parquet]` (MAINT-test-deps), cuyo mínimo sube con
DEP-numpy2-pandas23.

No forma parte de este roadmap: `torch>=2.4, <2.16` se subió en la 0.27.0 (commit
`7f7c4b804`, junto con la retirada de `pytest-xdist` del extra `test`). Probado con torch
2.15.0: `pytest skforecast/deep_learning` da 157 passed, 45 skipped (los de TensorFlow, que
no se ejecutan en Python 3.14).

### DEP-python310: retirar Python 3.10

Ya estaba en el roadmap de la 0.27.0 ("anunciar en 0.27 y retirar en 0.28"). Anunciado en
la nota de la 0.27.0 el 2026-11-11 (Changed, grupo *Dependencies*).

Va en su propia PR, no en la PR 6 de la migración a pandas 3, que lo preveía (decisión D1
(b) de `dev/pandas3_migration/INFORME.md`; cambio acordado el 2026-11-12). No depende de
pandas 3 y puede ir a `0.28.x` en cualquier momento, pero antes de la PR 6, cuya matriz de
CI da por hecho el mínimo 3.11.

Motivo:

- Python 3.10 llegó a fin de vida el 2026-10-01. Precedente: 3.9 (fin de vida en octubre de
  2025) se retiró en la 0.19.0, el 2025-11-28.
- Uso: 3,4 % de las descargas de skforecast en PyPI de los últimos 30 días y 10,1 % entre
  el 2026-04-08 y el 2026-11-10 (pypistats, todas las versiones de skforecast juntas, sin
  contar las descargas de versión desconocida).
- En 3.10 las dependencias ya están congeladas. Lo máximo que instala el resolvedor:

  | Paquete | Python 3.10 | Python 3.12 |
  |---|---|---|
  | numpy | 2.2.6 | 2.5.4 |
  | scipy | 1.15.3 | 1.18.1 |
  | scikit-learn | 1.7.2 | 1.9.1 |
  | matplotlib | 3.10.9 | 3.11.2 |
  | keras | 3.12.4 | 3.15.1 |
  | torch | 2.14.1 | 2.15.0 |
  | xgboost | 3.2.0 | 3.4.1 |

  pandas 3 exige Python 3.11.
- Quien siga en 3.10 se queda en la 0.27.0: pip respeta `requires-python`.

El beneficio en el código es modesto: desaparecen `tomli` en los tests y el caso especial de
T0 (en 3.10 pip instala `tfc-t0` 0.6.0, que se comporta distinto que la 0.6.1).

Cambios (hechos en #1496, merge `4016ede06`, salvo lo indicado):

- [x] `pyproject.toml:65`: `requires-python = ">=3.11"`; quitar el classifier de la línea 26.
- [x] `pyproject.toml:69,74,84,112`: `statsmodels>=0.13.3`. La 0.13.2 no tiene wheel para
  Python 3.11 y el resolvedor de mínimos ya elige la 0.13.3. El resto de mínimos se resuelven
  igual en 3.11 que en 3.10. La PR 6 de la migración a pandas 3 lo sube después a
  `>=0.14.6` (la 0.14.5 no se importa con pandas 3.1; `dev/pandas3_migration/INFORME.md`,
  sección 7).
- [x] `.github/workflows/unit-tests.yml:22`: quitar `'3.10'` de la matriz.
- [x] `.github/workflows/unit-tests-min-deps.yml:27-30`: Python 3.11.
- [x] `.github/workflows/publish-pypi.yml:142`: `['3.11', '3.14']`.
- [x] `skforecast/utils/tests/tests_utils/test_check_optional_dependency.py:5`: `tomli` por
  `tomllib`.
- [x] `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py:9-12`:
  quitar el `try/except` de `tomli`.
- [x] `.claude/skills/verify/test_affected.py:365`: quitar el `skipif` de Python menor que 3.11.
- [x] `pyproject.toml:123`: quitar `tomli>=2.0` del extra `test`.
- [x] `skforecast/foundation/_adapters.py:3561`: quitar la nota de Python 3.10 del docstring
  de `T0Adapter`. Decidir si el mínimo pasa a `tfc-t0[torch]>=0.6.1`
  (`tools/foundation_compat/adapters.toml:53`). Decidido: se queda en
  `tfc-t0[torch]>=0.6`. La 0.6.0 ya no se instala en Python 3.11 o superior, porque pip
  elige la 0.6.1, y subir el mínimo no aporta nada.
- [x] Docs: `docs/quick-start/how-to-install.md:5,10`, `README.md:11` (badge) y
  `docs/skforecast-ai/skforecast-ai.md:3,30` (comprobar antes si skforecast-ai sigue la
  misma versión mínima). `skforecast-ai.md` no se tocó: describe skforecast-ai, que
  todavía admite 3.10. Pendiente: abrir un issue en `skforecast/skforecast-ai` para que suba
  su mínimo y, después, actualizar la página.
- [x] Contexto IA: `tools/ai/llms-base.txt:13` y `tools/ai/ai_context_header.md:68`
  (`uv venv ... --python 3.10`), y regenerar (skill `ai-context-sync`).
- [x] Nota en `docs/releases/releases.md`, sección 0.28.0.

Sin comprobar: la suite con las versiones mínimas en Python 3.11 (es la suite completa, la
ejecuta `unit-tests-min-deps.yml`). Solo se resolvieron las versiones, con
`uv pip compile --resolution lowest-direct --python-version 3.11 --only-binary :all:` en
macOS arm64. La #1496 no ejecutó ese workflow, porque las PR a `0.28.x` no lanzan los tests.
Lanzarlo a mano sobre `0.28.x`, o esperar a la PR de la release a `main`. Al publicar la
0.28.0, subir también en el feedstock de conda-forge el mínimo de Python a 3.11 y el de
`statsmodels` a 0.14.6 (era 0.13.3 hasta DEP-numpy2-pandas23).

### DEP-numpy2-pandas23: numpy 2.0 y pandas 2.3 como mínimos

Decidido por el usuario el 2026-11-13, durante la revisión de la PR 6 de la migración a
pandas 3 (`dev/pandas3_migration/INFORME.md`, sección 10, "Revisión de la PR 6, el
2026-11-13"). Va en su propia PR, como DEP-python310, y antes de la PR 6: cambia lo que
prueba la matriz de CI de la PR 6 (la celda de pandas 2 y el job de mínimos). En la
migración sustituye a la decisión D2 (a) (mínimo pandas 2.2.0) por la (c).

Motivo:

- numpy 1.26 (la última 1.x, 2023-10) ya no recibe correcciones y no tiene wheels para
  Python 3.13 ni 3.14. pandas 2.3 es la única serie 2.x que aún recibe correcciones.
- Con numpy 1.x como mínimo, el pin `pandas<4.0` de la PR 6 permite una combinación que no
  prueba ningún job: si otra librería fija `numpy<2`, el resolvedor elige pandas 3.0.6 con
  numpy 1.26.4 (pandas 3.0.x acepta numpy 1.26; 3.1 exige 2.0.2). En ese entorno, la suite
  completa (Windows, Python 3.12, pyarrow 25.0.1) dio 255 fallos: unos 220 son columnas
  enteras `int32` en lugar de `int64` (efecto de numpy 1.x en Windows, también con pandas
  2.2.0) y unos 30 sin clasificar.
- Quita trabajo: la restricción `pyarrow<26` de `unit-tests-min-deps.yml` (#1498; pyarrow 26
  exige numpy 2 sin declararlo) deja de hacer falta, y las columnas enteras dejan de salir
  `int32` en Windows (numpy 2 usa int64 por defecto).
- SPEC 0 (soporte de lo publicado en los dos últimos años) pediría hoy numpy 2.2, pandas
  2.3, scipy 1.15 y numba 0.61. Esta PR solo sube lo que exige numpy 2.0 más pandas 2.3;
  alinearse con SPEC 0 es otra decisión.
- Quien no pueda salir de numpy 1.x o de pandas 2.2 se queda en la 0.27, pero tiene que
  fijar `skforecast<0.28`: `pip install -U skforecast` no se queda en la 0.27, sube numpy y
  pandas (comprobado con `--dry-run` sobre numpy 1.26.4 y pandas 2.2.3: instalaría numpy
  2.5.4, pandas 2.3.3, scipy 1.18.1 y numba 0.68.1). pip solo elige la 0.27 cuando `numpy<2`
  entra en la misma resolución.

Mínimos nuevos. Calculados con `uv pip compile --resolution lowest-direct` (uv 0.13.8, Python
3.11, Linux) y comprobados importando cada paquete con numpy 2.0.0 en Windows (importar no
basta: ver lightgbm en la revisión final, más abajo). Las versiones
intermedias que se citan declaran `numpy<2` (no se instalan) o se instalan y fallan al
importar (`numpy.dtype size changed` o `numpy.core.multiarray failed to import`):

| Paquete | Ahora | Nuevo | Por qué | Fecha |
|---|---|---|---|---|
| numpy | 1.26.1 | 2.0 | Decisión | 2024-06-16 |
| pandas | 2.2 | 2.3 | Decisión (2.2.0 y 2.2.1 no funcionan con numpy 2) | 2025-06-05 |
| scipy | 1.12 | 1.13 | 1.12 declara `numpy<2` | 2024-04-02 |
| numba | 0.59 | 0.60 | Primera con numpy 2 | 2024-06-13 |
| statsmodels (extras) | 0.13.3 | 0.14.6 (ver abajo) | 0.13.3 a 0.14.1 fallan; 0.14.2 funciona | 2025-12-05 |
| matplotlib (extras) | 3.7 | 3.8.4 | 3.7.0 a 3.8.3 fallan o no se instalan | 2024-04-04 |
| catboost (extra `test`) | 1.2 | 1.2.8 | 1.2 a 1.2.5 fallan; 1.2.6 y 1.2.7 no se instalan | 2025-04-13 |
| lightgbm (extra `test`) | 4.0 | 4.4 | 4.0.0 a 4.3.0 se importan pero fallan al usarlos (revisión final) | 2024-06-15 |
| `pandas[parquet]` (extra `test`) | 2.2 | 2.3 | Coherencia con el core | |

Sin cambio, comprobados con numpy 2.0.0: scikit-learn 1.6.0, xgboost
2.1.0, optuna 4.0.0, skops 0.14.0, pyarrow 26.0.1 (ya no hace falta limitarlo), keras 3.3.0
y torch 2.4.0 (compatible con numpy 2 desde la 2.3; en Windows no carga `fbgemm.dll`, como
siempre, el job de mínimos corre en Linux).

statsmodels: la primera versión compatible con numpy 2 es la 0.14.2, pero la PR 6 necesita
la 0.14.6 (la 0.14.5 no se importa con pandas 3.1). **Decidido el 2026-11-13: `>=0.14.6`**
directamente, un solo cambio para los usuarios; la PR 6 deja de tocar statsmodels (con
`pandas<4.0` y un mínimo 0.14.2, el resolvedor podría elegir 0.14.2 a 0.14.5 con pandas 3,
y `skforecast.stats` no se importaría).

Moirai: `uni2ts` 2.0.0 (2025-11-04, el backend del `MoiraiAdapter`, la última versión)
exige `numpy~=1.26.0`, `scipy~=1.11.3` y `torch<2.5`, y su dependencia `gluonts` 0.14.4,
`pandas<2.2` (la 0.14.3 admite `pandas<3`). **Comprobado el 2026-11-13: funciona con numpy
2.** Copia de `origin/0.28.x` con los mínimos nuevos, Windows, Python 3.12, `uv` 0.13.8,
`overrides = ["numpy>=2", "scipy>=1.13"]`; resuelve uni2ts 2.0.0, gluonts 0.14.3, torch
2.4.1+cpu (carga en Windows; el fallo de `fbgemm.dll` es de la 2.4.0), pandas 2.3.3 y scipy
1.18.1:

| numpy | Contrato | Smoke (pesos reales) | `test_MoiraiAdapter.py` | `skforecast/foundation/tests` |
|---|---|---|---|---|
| 2.5.4 | 5 pasan | 5 pasan, 2 omitidos (sin exog) | 36 pasan | 1130 pasan |
| 2.0.2 | 5 pasan | 5 pasan, 2 omitidos | 36 pasan | 1130 pasan |

Los cuantiles de `predict_quantiles` (dos series, 24 pasos) son idénticos bit a bit con
numpy 1.26.4 (entorno de hoy, `origin/0.28.x` sin cambios), 2.0.2 y 2.5.4. Sin el override
de numpy, el entorno de moirai no se resuelve (el job programado fallaría). Los otros diez
backends resuelven exactamente lo mismo antes y después (Linux, Python 3.12, `uv pip
compile`); solo moirai pasa de numpy 1.26.4 a 2.5.4.

Para el usuario el problema ya existía: con la 0.27 publicada, `pip install skforecast
uni2ts` resuelve a skforecast 0.20.1 (sin `MoiraiAdapter`) por el conflicto de scipy, y
`pip install uni2ts` sobre un skforecast instalado baja numpy a 1.26.4, scipy a 1.11.4 y
pandas a 2.1.4. Receta comprobada con pip en un entorno limpio (skforecast de la copia,
después `pip install uni2ts "pandas>=2.3"` y `pip install "numpy>=2" "scipy>=1.13"`): queda
numpy 2.5.4, pandas 2.3.3, gluonts 0.14.3, y pasan el contrato y el smoke. Con uv, un solo
comando con `--override`. Decisión del usuario: overrides más documentación; si algún
adaptador dejara de ser compatible, se podría deprecar o desactivar (no hace falta con
Moirai).

Comprobado también: un forecaster guardado con numpy 1.26.4 se carga con numpy 2.5.4 y al
revés, con joblib y con skops, con las mismas predicciones (Ridge, `transformer_y`, exog).

Cambios (rama `chore/numpy2-pandas23-minimums`, sin commit):

- [x] `pyproject.toml:54-61`: `numpy>=2.0`, `pandas>=2.3, <3.0` (la PR 6 lo abre a `<4.0`),
  `scipy>=1.13`, `numba>=0.60`.
- [x] `pyproject.toml:72, 78, 82, 112`: `matplotlib>=3.8.4, <3.12`.
- [x] `pyproject.toml:68, 73, 83, 111`: `statsmodels>=0.14.6, <0.16`.
- [x] `pyproject.toml:117`: `catboost>=1.2.8, <2.0`; `:122`: `pandas[parquet]>=2.3`;
  `:115`: `lightgbm>=4.4, <5.0` (revisión final).
- [x] `skforecast/utils/utils.py:95-106` (`optional_dependencies`): matplotlib y
  statsmodels. `test_check_optional_dependency.py` exige que coincida con
  `pyproject.toml`.
- [x] `.github/workflows/unit-tests-min-deps.yml:44-49`: quitar la restricción `pyarrow<26`
  y su comentario.
- [x] `tools/foundation_compat/adapters.toml` (moirai): `overrides = ["numpy>=2",
  "scipy>=1.13"]` (un override sustituye también el requisito de skforecast, así que repite
  su mínimo de scipy), con el comentario.
- [x] Docs: `docs/quick-start/how-to-install.md:32, 33, 36, 39, 59, 68, 69, 79`.
- [x] Moirai: tarjeta de la tabla de backends de
  `docs/user_guides/foundation-forecasting-models.ipynb` (`pip install uni2ts
  "pandas>=2.3"`) y celda markdown nueva tras la tabla con la receta completa (pip y uv);
  Notes del docstring de `MoiraiAdapter`; `skills/foundation-forecasting/SKILL.md`
  (instalación), `references/adapter-parameters.md` (línea **Backend**) y
  `tools/ai/llms-base.txt` (sección de foundation).
- [x] Contexto IA: `tools/ai/ai_context_header.md:39-40`, y regenerar (skill
  `ai-context-sync`).
- [x] Nota en `docs/releases/releases.md`, sección 0.28.0, con su "Before upgrading" (dos
  entradas sin enlace hasta abrir la PR), y quitada la frase de statsmodels 0.13.3 de la
  entrada de la #1496.
- [x] El docstring de `test_covariate_arrays.py:45` ("With pandas < 2.3...").
- [x] Restos de numpy 1.x (revisión de todo el repositorio, salvo `dev/` y las notas de
  versiones): la ruta a `np.trapz` de `skforecast/metrics/metrics.py:587-590` y la
  tolerancia `int64 or int32` de `test_cast_catboost_categorical_columns_dataframe.py:78`
  (`astype(int)` da int64 en Windows con numpy 2). Se quedan: los `int(...)` comentados
  "Required since numpy 2.0" (`model_selection/_utils.py:128`, `_split.py:1231`), la rama
  `int32` de Windows de `test_reshape_exog_long_to_dict.py:372` (`np.random.randint` sigue
  dando int32 en Windows con numpy 2.5.4) y `WINDOWS_LEGACY_BLAS` (Python 3.11, no numpy 1).
- [x] Fuera de la entrada: `tools/foundation_compat/run.py:290-291` pasaba
  `venv\Scripts\python` sin `.exe` a uv y no funcionaba en Windows.

Verificación hecha: `verify` en verde (ruff; 2188 tests de `foundation`, `metrics` y
`utils` con el cwd en `C:` mayúscula; contexto IA, `page_meta`, `--check` del notebook;
`run.py --adapter moirai --level contract`: 5 pasan, `held_back: false`). Resolución
`lowest-direct` del job de mínimos (Linux, Python 3.11) sin la restricción de pyarrow:
numpy 2.0.0, pandas 2.3.0, scipy 1.13.0, numba 0.60.0, statsmodels 0.14.6, matplotlib 3.8.4,
catboost 1.2.8, pyarrow 26.0.1, torch 2.4.0. Sin suite completa local ni arnés (decisión del
usuario: el código del paquete no cambia y el entorno de desarrollo ya está en el rango
nuevo).

Revisión final, el 2026-11-13 (misma rama, sin commit):

- **lightgbm 4.0.0 no funciona con numpy 2** y era el mínimo del extra `test`: el job de
  mínimos habría fallado. Con un DataFrame, la 4.0.0 llama a `np.find_common_type`
  (eliminada en numpy 2.0), y las 4.1.0 a 4.3.0 fallan en `set_label` con
  `np.array(..., copy=False)` (101 tests de 2121). La 4.4.0 es la primera que pasa.
  `pyproject.toml:115` pasa a `lightgbm>=4.4, <5.0` (solo el extra `test`, sin nota de
  versión). La comprobación anterior solo importaba cada paquete.
- Comprobado en un entorno con los mínimos (Windows, Python 3.11, `lowest-direct`, sin
  torch ni keras: numpy 2.0.0, pandas 2.3.0, scikit-learn 1.6.0, scipy 1.13.0, numba
  0.60.0, lightgbm 4.4.0, xgboost 2.1.0, catboost 1.2.8): los 78 ficheros de test que
  importan lightgbm, xgboost o catboost, 2121 pasan. El resto de la suite con los mínimos
  (statsmodels, matplotlib, optuna, skops, keras, torch) solo lo prueba el job de CI.
- Nota de versión: quitada la frase "With older versions, pip keeps installing skforecast
  0.27.0" (falsa para dependencias, ver "Motivo") y la de "the first versions that work
  with numpy 2" (no es cierta para statsmodels 0.14.6); el enlace a la guía de migración
  de numpy apunta a `doc/stable` en lugar de `devdocs`.
- `uv pip install skforecast uni2ts --override overrides.txt` comprobado con `uv pip
  compile`: mantiene el skforecast más reciente y resuelve gluonts 0.14.3, numpy 2.5.4,
  pandas 2.3.3 y scipy 1.18.1.
- El `ImportError` de `MoiraiAdapter` decía `pip install uni2ts`, que por sí solo baja
  numpy, scipy y pandas: ahora da los dos pasos de la receta (decisión del usuario), con su
  test en `test_MoiraiAdapter.py`. El "Before upgrading" queda resumido a dos líneas; el
  detalle de los mínimos está en Changed.

Pendiente: lanzar a mano sobre la rama `unit-tests-min-deps.yml` (la suite completa con los
mínimos nuevos, en Linux) y `foundation-backends.yml` (moirai con los overrides nuevos),
una vez subida.

Al publicar la 0.28.0, subir los mismos mínimos en el feedstock de conda-forge.

### MAINT-docs-deps: pins del extra `docs`

Nada es obligatorio: los pins actuales construyen el sitio sin avisos.

| Paquete | Pin actual | Última | Acción |
|---|---|---|---|
| mkdocs-material | 9.7.7 | 9.8.1 | Subir |
| mkdocstrings | 1.0.7 | 1.0.8 | Subir |
| mkdocstrings-python | 2.0.9 | 2.0.10 | Subir |
| pymdown-extensions | 12.2 | 13.0 | Subir |
| black | 26.5.1 | 26.10.1 | Subir |
| mkdocs-jupyter | 0.26.3 | 0.27.0 | **No subir** |
| mike | 2.2.0 | 2.2.0 | Nada |
| mkdocs | 1.6.1 | 1.6.1 | Nada |

Medido con tres builds (`mkdocs build`, Python 3.14, 108 páginas HTML, 0 avisos en los tres):

- **Todo a la última menos `mkdocs-jupyter`**: el HTML es idéntico al de los pins actuales
  salvo la etiqueta `<meta name="generator">`. Subirlos no cambia nada visible; solo sirve
  para no quedarse atrás.
- **Con `mkdocs-jupyter` 0.27.0**: cambia el texto de 28 de las 61 páginas de notebooks. Las
  fórmulas pasan de `$...$` a `<span class="arithmatex">\(...\)</span>` y 100 de las 108
  páginas llevan un script que carga MathJax 4.1.3 desde `cdn.jsdelivr.net`. Choca con el
  KaTeX local (`docs/vendor/katex`, `docs/javascripts/katex.js`) y con el plugin `privacy`
  (ninguna conexión a terceros antes de aceptar cookies). Además exige Python 3.12. No se
  abrió el sitio en un navegador. Para subirlo hay que encontrar antes cómo desactivar ese
  script.

Cambios:

- [ ] Subir los cinco pins marcados en `pyproject.toml:92-103`. Las versiones nuevas de
  `mkdocs-material` y `mkdocstrings` exigen Python 3.11 y `pymdown-extensions` 13.0 exige
  3.11.5 (`mkdocstrings` 1.0.7 ya exigía 3.11).
- [ ] Añadir `pymdown-extensions` a la lista `ignore` de `.github/dependabot.yml`, como el
  resto de pins de docs; hoy dependabot propondrá la 13.0.
- [ ] Planificar: `mkdocs-material` 9.8.1 avisa en el build de que llega a fin de vida el
  2027-05-05 (sucesor: Zensical, guía de migración en
  https://zensical.org/docs/compatibility/mkdocs/migration/). Es una tarea aparte (M o L),
  sin analizar.

### MAINT-test-deps: extra `test`

Ningún límite del extra necesita cambiar. Con todo a la última permitida y pandas 2.3.3 (el
entorno de medida de la cabecera), los 91 tests que fallan en el workflow semanal
`unit-tests-latest-deps.yml` dan 90 passed, 1 skipped. Los fallos de ese workflow se deben
solo a pandas 3 (ver la sección 2).

Limpiezas:

- [x] Quitar `pytest-xdist>=3.8`: hecho en la 0.27.0 (commit `7f7c4b804`). Ningún workflow
  usa `-n`, las reglas de tests lo prohíben y `.claude/settings.json` lo deniega.
- [ ] `pandas[parquet]>=2.2, <3.0` (`pyproject.toml:124`), por coherencia con el core. No
  cambia la resolución. Si se hace, la PR 6 de la migración a pandas 3 tiene que pasarlo a
  `<4.0` junto con el core. El mínimo pasa a 2.3 con DEP-numpy2-pandas23 (hoy está en la
  línea 122, sin techo).
- [x] `tomli>=2.0`: se quita con DEP-python310. Hecho en #1496.

---

## 2. Otros hallazgos de la misma revisión (sin tarea asignada)

- **`pandas<3.0` se mantiene.** El workflow `unit-tests-latest-deps.yml` fuerza pandas 3.1.1
  saltándose el límite y da 91 failed y 6732 passed en Python 3.14, sobre `main` (run del
  2026-11-09; 91 failed también en 3.12 y 3.13). Son fallos de índices, dtypes, zonas
  horarias y frecuencias. El workflow lleva en rojo al menos desde el 2026-10-05. El trabajo
  de compatibilidad está en `dev/pandas3_migration/`: el límite pasa a `<4.0` en su PR 6,
  y hasta entonces se mantiene `<3.0`.
- **tensorflow**: el límite `<2.22` es correcto hoy (última estable: 2.21.0). La 2.22.0rc0 ya
  trae wheels para Python 3.14: cuando salga la final, subir a `<2.23` y revisar el marcador
  `python_version < "3.14"` (`pyproject.toml:86`) y los `skipif` de los tres tests de
  TensorFlow en `skforecast/deep_learning/tests/`.
- **Límites inferiores**: correctos. `unit-tests-min-deps.yml` pasó el 2026-11-03 con las
  versiones mínimas declaradas en Python 3.10 (7363 passed, 27 skipped).

---

## 3. Lo que pasa de la 0.27.0 (decisión del 2026-11-11)

El 2026-11-11 se cerró el alcance de la 0.27.0: todo lo que no estaba hecho pasa aquí. La
evidencia de cada elemento (mediciones, antes y después, esbozos del arreglo y scripts en
`dev/roadmap_0.27.0/repro/`) sigue en [el roadmap de la 0.27.0](../roadmap_0.27.0/roadmap.md), que queda como
archivo: aquí va lo necesario para priorizar y empezar cada uno. La columna "Detalle" dice
dónde está su fila allí.

Esfuerzo como en la cabecera. Las líneas de código citadas en el roadmap de la 0.27.0 son de
su fecha: localizar el código por nombre.

### 3.1 BUG-04 y modelos afinados

**Decisión.** El arreglo de BUG-04 no entra en la 0.27.0: al revisarlo, el tema (usar y
cargar modelos foundation afinados) resultó más grande que el bug. Se planifica entero en la
0.28.0. `dev/PLAN_UnifiedFinetunedCheckpoint.md` (PLAN-CKPT, parámetro común
`finetuned_checkpoint`) se eliminó el 2026-11-11 por desactualizado: **queda pendiente
crear un plan nuevo** a partir de lo de abajo (la revisión se hizo sin leerlo, a propósito,
y propone otro diseño: un argumento `adapter`).

**Lo que se hizo (descartado).** Hubo una rama local `fix/foundation-clone-preloaded-model`
con el arreglo (4 commits, sin push ni PR), borrada el 2026-11-11 al aplazar el tema. Su
diseño queda aquí como punto de partida:

- Contrato: dos atributos obligatorios nuevos en `_REQUIRED_CLASS_ATTRIBUTES`,
  `backend_attribute` (`'_pipeline'`, `'_module'` o `'_model'`) y `backend_param`
  (`'pipeline'`, `'module'`, `'model'` o `None`). Los 11 adaptadores aceptan un objeto
  precargado. `_AdapterBase` no es pública (no aparece en docs, skills ni `llms-base.txt`).
- `FoundationModel.__sklearn_clone__` clona `get_params()` y pasa al clon la misma
  referencia del objeto del usuario (anotado en `adapter._preloaded_backend`), sin copiar
  pesos; un backend cargado de forma perezosa no pasa al clon. Meterlo en `get_params` se
  descartó: `clone` haría un `deepcopy` (Chronos-2 small: 112 MB y unos 0,1 s por copia; 5
  `deepcopy_forecaster` suman +20 MB de RSS con la referencia frente a +562 MB copiando).
- `set_params` que cambia un parámetro fijado en el backend (`model_id`, dispositivo,
  dtype...) descarta el objeto con `IgnoredArgumentWarning`. El repr muestra solo la clase.
- `save_forecaster` guarda el objeto con sus pesos. Para QW-05 basta comparar por
  identidad con `_preloaded_backend` en `__getstate__` (prototipo: 0,01 MB con backend
  perezoso, 111,81 MB con el del usuario).
- `BUG-04.py`: las dos partes dan lo esperado con la rama (A7 cambiado para comprobar que
  `clone` comparte el objeto; parte B con `ft.model.eval()`).

**Pendiente de la revisión del 2026-11-11 sobre esa rama:**

1. Llamar a `eval()` sobre el objeto del usuario antes de predecir:
   `BaseChronosPipeline.fit` (chronos-forecasting 2.3.2) devuelve el modelo en modo
   entrenamiento y, con dropout activo, dos `predict` seguidos difieren hasta 0,35. Los
   adaptadores ya lo hacen cuando cargan ellos el modelo (Moirai-2, T0).
2. Con un objeto precargado, buscar `context_length` lo descarta (con aviso) en TimesFM 2.5,
   Moirai-2, TabICL y TabPFN-TS, porque ese parámetro reinicia el backend: documentarlo; en
   Moirai-2 se evita con unas 6 líneas (PR aparte).
3. Al rehacerlo: el código de la rama se fusionaba sin conflictos con #1479 (los dos tocan
   `_adapter_base.py`, `_adapters.py` y `_foundation_model.py`).

**Revisión de los backends (2026-11-11).** Fuentes: código de los paquetes instalados y de
las últimas versiones (chronos-forecasting 2.3.2, timesfm 3.0.1, uni2ts 2.0.0, tabicl
2.2.0, tabpfn-time-series 1.4.0 con tabpfn 9.2.0, tfc-t0 0.6.1, synthefy-nori 0.23.0,
tsicl 0.2.1, timestral git a534bdb, tirex-2 0.5.0) y un fine-tuning real de Chronos-2
small.

| Backend | Afinar | Ruta local o repositorio propio con la carga actual del adaptador | Checkpoint que dice su familia |
|---|---|---|---|
| Chronos-2 | Oficial: `pipeline.fit(...)`, completo o LoRA | Sí, el completo (`save_pretrained`, probado). LoRA probablemente no: `BaseChronosPipeline.from_pretrained` no tiene la rama de LoRA de `Chronos2Pipeline.from_pretrained` (sin probar, falta `peft`) | Sí (`architectures`, `chronos_pipeline_class`) |
| TimesFM 2.5 | Ejemplo oficial con LoRA sobre el port de transformers | Sí, pero el resultado del ejemplo tiene otro formato (4 de 232 tensores coinciden): no se carga | No |
| TimesFM 3 | No (solo inferencia) | Sí (directorio o `.safetensors`/`.pth`) | No |
| Moirai-2 | No (uni2ts solo afina Moirai 1.x) | Sí | No |
| TabICL | Oficial, pero sobre tablas (`FinetunedTabICLRegressor`) | No por `model_id` (repositorio fijo); sí con `tabicl_config={"model_path": ...}` | En parte |
| TabPFN-TS | Solo el de `tabpfn`, sobre tablas | No por `model_id`; sí con `model_path` en su configuración, solo en modo local | En parte |
| T0 | Lo escribe el usuario (`forward`) | Sí (`PyTorchModelHubMixin`) | No |
| Nori | No aplica (aprendizaje en contexto) | No por `model_id`; sí con `nori_config={"model_path": ...}` | En parte |
| TS-ICL | No aplica (aprendizaje en contexto) | No: el backend tiene `model_path`, skforecast no lo expone | No |
| Timestral | No | Sí (directorio) | No |
| TiRex-2 | Solo en la versión de pago | Sí (directorio) | No |

**En skforecast:**

- `_resolve_adapter` elige el adaptador por el prefijo de `model_id`: una ruta local o un
  repositorio propio (`./chronos2-ft`, `miorg/chronos2-ft`) da `ValueError: No adapter
  found`, aunque el backend sepa cargarlo. El único camino hoy es el objeto precargado, con
  `model_id` como simple etiqueta.
- `_warn_if_non_commercial` también va por el prefijo de `model_id`: con una ruta local se
  perdería el aviso de licencia (copias de TimesFM 3, Moirai-2 o TS-ICL, cuya licencia
  cubre también las versiones afinadas).
- TabICL y TS-ICL descargan en silencio los pesos preentrenados en `model_path` si no
  existe: una ruta mal escrita da el modelo base sin aviso.
- skforecast no afina: el `fit` de los adaptadores solo guarda el contexto.

**Qué necesitan los usuarios:** (1) usar en memoria un modelo recién afinado (lo resuelve
BUG-04 con el punto 1 de arriba); (2) cargar un checkpoint desde disco o desde su propio
repositorio (producción, reproducibilidad, entornos sin internet, también copias locales de
los pesos base), que hoy solo bloquea el prefijo de `model_id`.

**Propuesta:**

- BUG-04 con el diseño de arriba, más los puntos 1 y 2.
- Argumento `adapter` en `FoundationModel` (nombre o clase) que sustituye a la resolución
  por prefijo: `FoundationModel("./chronos2-ft", adapter="chronos")`. `model_id` describe
  los pesos y llega tal cual a la carga del backend (sirve sin más en Chronos-2, TimesFM
  2.5 y 3, Moirai-2, T0, Timestral y TiRex-2), y el forecaster guardado lleva la ruta, no
  los pesos (encaja con QW-05). S a M: `_resolve_adapter`, `get_params`, repr,
  `get_model_info`, el aviso de licencia por adaptador y origen de los pesos, y convivencia
  con la tabla de ids no soportados (plan 02 de `dev/foundation_adapters_review/`).
  Descartados: un `finetuned_checkpoint` aparte (duplica `model_id`) y detectar la familia
  por los ficheros (solo fiable con Chronos-2).
- Pequeños: cargar Chronos-2 con `Chronos2Pipeline.from_pretrained` para aceptar LoRA
  (probar con `peft`); comprobar que `model_path` existe antes de pasárselo a TabICL y
  TS-ICL; exponer `model_path` en TS-ICL solo si se pide.
- Fuera de alcance: TimesFM 2.5 afinado con transformers (otro formato) y ofrecer el
  fine-tuning desde skforecast.

### 3.2 Foundation (resto)

| ID | Qué | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|
| Planes 02 a 09 de la revisión de adaptadores | Alinear los adaptadores con sus backends: 02 cambios comunes (error con contexto todo NaN, tabla central de ids no soportados, cuantiles a `float`, helper de versión, `stacklevel`), 03 TS-ICL (alta, "desfasado"), 04 TimesFM, 05 Chronos-2, 06 Moirai-2, 07 TabICL, 08 TabPFN-TS, 09 Nori | S a M cada uno | Planes escritos el 2026-11-06, con decisiones pendientes (sección 4 del README) y orden fijado (02 antes que 03 a 09; 03 a 09 en serie; 07 antes que 08). El 01 (T0) se hizo en la 0.27.0 (#1478) | [`dev/foundation_adapters_review/README.md`](../foundation_adapters_review/README.md). Antes de empezar, volver a comprobar versiones y checkpoints (sección 1.5 de `10_docs_ci.md`) |
| Plan 10, parte de CI | Caché de los pesos de TabPFN en `foundation-backends.yml` (3.5), segunda entrada para Nori-1-5 (3.6) y README de `tools/foundation_compat` (3.7) | S | La re-ejecución de las guías del plan 10 se hizo en la 0.27.0 | `dev/foundation_adapters_review/10_docs_ci.md` |
| BUG-06 | `predict()` da la media y `predict_interval()` la mediana en TabICL, Nori y TabPFN con `point_estimate` distinto de `'median'`; `backtesting_foundation` con cuantiles no devuelve `pred` | M | Requisito de FE-05 y de GAP-foundation-backtesting-interval | Sección 4 |
| QW-05 | `save_forecaster` sin los pesos foundation cargados tras `predict` (0,01 MB frente a 111,81 MB) | S | Con BUG-04 basta comparar por identidad con el objeto del usuario en `__getstate__` (3.1) | Sección 2 |
| QW-02 | Aviso cuando se ajusta con exog y `predict` no la recibe (Chronos-2, TimesFM 3, TS-ICL pierden el efecto en silencio) | S | Decisión pendiente: invierte un silencio documentado | Sección 2 y decisión pendiente 3 |
| FE-02 | `past_only_exog` en `backtesting_foundation` y `bayesian_search_foundation` | S | BUG-10 (requisito) ya está hecho (#1479) | Sección 3 |
| GAP-foundation-backtesting-interval | `interval` en `backtesting_foundation` (hoy solo `quantiles`) | S | Después de BUG-06 | Sección 6 |
| PERF-01 | `backtesting_foundation` llama al modelo una vez por fold aunque es zero-shot | M | Gana 3,4 veces con una serie y nada con 50 | Sección 4 (rendimiento) |
| Moirai-2 y `context_length` con objeto precargado | Buscar `context_length` descarta el objeto del usuario; en Moirai-2 se evita con unas 6 líneas | S | Seguimiento de BUG-04 (3.1) | 3.1 |
| PLAN-TTM | Adaptador TinyTimeMixer | M | Plan sin código, no verificado | [`dev/PLAN_TinyTimeMixerAdapter.md`](../PLAN_TinyTimeMixerAdapter.md) |

### 3.3 Backtesting, búsquedas y métricas

| ID | Qué | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|
| FE-05 | Comparar varios forecasters de cualquier familia en un solo backtesting, sobre los mismos puntos y con la misma escala | M | Diseño analizado y prototipado (`repro/FE-05-prototype.py`); decisiones pendientes: nombre, puntos comunes por defecto, alcance de la v1. Requisito: BUG-06. Usa `n_evaluated` y las columnas `step` e `y_true` de la 0.27.0 | Sección 3 ("FE-05: análisis de diseño") y decisión pendiente 7 |
| BUG-02 | Con `fold_stride < steps` las métricas solo miden los primeros horizontes y las búsquedas eligen por el error a 1 paso | S | Decisión pendiente 1 (parámetro nuevo en `TimeSeriesFold`, defecto actual y aviso). Con `n_evaluated` y `step` el efecto ya es visible | Sección 4 y decisión pendiente 1 |
| PERF-03 | Métricas pedidas por nombre en numpy (búsqueda `OneStepAheadFold` con 1000 series: 2,6 veces más rápida con Ridge) | S | Toca `_get_metric`; incluir la versión numpy de RMSE | Sección 4 (rendimiento) |
| BUG-01 | `backtesting` con `initial_train_size=None` sobre un forecaster ajustado evalúa in-sample sin aviso (métricas 2,5 veces optimistas) | S | Solo aviso silenciable: es un uso documentado | Sección 4 |
| BUG-11 | `backtesting_forecaster` con `ForecasterEquivalentDate` e `interval` da `AttributeError` con el método por defecto | S | | Sección 4 |
| FE-04 | `ForecasterEquivalentDate` en `backtesting_forecaster_multiseries`, con los mismos folds que el modelo global | M | | Sección 3 |
| BUG-refit-directmultivariate-short | Búsquedas de `ForecasterDirectMultiVariate` con series cortas y refit | S a M | Hallado en el PR 2b de BUG-search-short-series | Sección 4 |
| BUG-09 (b) | Resto de BUG-09 en `backtesting_stats` (la parte a se hizo, #1419) | S | Nicho | Sección 4 |
| Agrupar los avisos repetidos de un backtesting | Hoy un backtesting con `refit=True` emite el mismo aviso una vez por fold, con cualquier `n_jobs`: 7 avisos iguales con huecos en `y` (QW-13), con `ConvergenceWarning` de un MLP o con QW-13 en multiserie (medido el 2026-11-08 con #1467). Propuesta: uno solo con "(in 6 of 7 folds)", como ya hacen las búsquedas | M | Fuera de #1467 (PR 2c) a propósito: cambia también lo que se ve con `n_jobs=1` | Alternativa E de la sección 15.5 del documento de BUG-search-short-series (borrado el 2026-11-11, en el historial de git) |

### 3.4 Forecasters y multiserie

| ID | Qué | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|
| QW-01 | Aviso cuando `predict()` con `levels=None` deja fuera series que no llegan a la última fecha | S | | Sección 2 |
| QW-06 | Alinear `exog` por índice en `predict` de Recursive y Direct | S | Junto con QW-14 (mismo camino) | Sección 2 |
| QW-14 | Un solo aviso de `exog` con NaN en `predict` | S | | Sección 2 |
| QW-15 | Avisos del `fit` en paralelo de Direct y DMV | S | Sin medir | Sección 2 |
| BUG-07 | El formato largo con MultiIndex solo funciona si sale de `reshape_series_wide_to_long` | S | | Sección 4 |
| BUG-classifier-nan-last-window | Clasificador sin soporte de NaN y NaN en la última ventana | S | Prioridad baja | Sección 4 |
| `RollingFeatures.transform` con NaN | En predicción ignora los NaN y `min_periods`; en entrenamiento (`transform_batch`) los respeta. Hacerlo coherente mejora un 2% en el experimento 5 | S | Elemento nuevo (2026-11-11): estaba solo en la sección 7 del documento de BUG-search-short-series, borrado | Experimento 5 de ese documento, en el historial de git |
| HANDOFF-T2, T3 y T4 | Categóricas nativas dentro de `CalibratedClassifierCV` o `TransformedTargetRegressor` (T2); Pipeline con LightGBM o CatBoost y exog categórica (T3); T4 | S | Pendientes de decisión de Joaquín; no reverificados | `dev/handoff_utils_pendiente.md`; sección 6 |

### 3.5 Preprocessing

| ID | Qué | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|
| QW-11 | `hour_of_week` y días de fin de semana del mes en `CalendarFeatures`, aviso con features anuales y menos de un año de datos, guía de codificación | S | Medido en 18 series (`hour_of_week`: -24% de MAE con Ridge en horarias). Decisión pendiente 8 | Sección 2 ("QW-11: análisis") |
| FE-06 | Festivos en `CalendarFeatures` con `holidays` como dependencia opcional | M | Decisión pendiente 9 (solo 2 series medidas) | Sección 3 |
| FE-03 | Ventanas estacionales en `RollingFeatures` (`season_length`) | M | Mejora el MAE un 2% y solo con árboles | Sección 3 |

### 3.6 Stats

| ID | Qué | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|
| PERF-02 | Ajuste de Ets y Sarimax de 3 a 70 veces más lento con carga en la máquina | S | Volver a medir: BUG-08 (#1425) cambió el escenario | Sección 4 (rendimiento) |
| DEP-arima-dictstyle | `FutureWarning` sin versión desde la 0.22 | S | Decidir: retirarlo o fijar la versión | Sección 6 |

### 3.7 Docs, deep learning y mantenimiento

| ID | Qué | Esfuerzo | Estado | Detalle |
|---|---|---|---|---|
| QW-07 | Mensajes de error accionables en cuatro puntos | S | Prototipo hecho | Sección 2 |
| DOC-skills-percentiles-stale | Skills con percentiles (dan `ValueError` desde la 0.25) y `set_params` de EquivalentDate | S | Junto a QW-07 (b); regenerar el contexto IA | Sección 6 |
| DOC-rnn-exog-example | El ejemplo con exógenas de `docs/user_guides/forecasting-with-deep-learning-rnn-lstm.ipynb` no es reproducible entre versiones de keras y torch: con keras 3.15.1 y torch 2.15.0, `EarlyStopping(patience=3)` corta en la época 12 (antes 37) y el MAE de `users` sube de 47,75 a 76,46. Las cuatro primeras épocas son idénticas; la pérdida de validación difiere en el tercer decimal en la quinta. `patience=5` con `restore_best_weights=True` no lo recupera (la mejor época sigue siendo la novena). La guía de la 0.27.0 conserva la salida del 2026-11-09 (keras 3.14.0, torch 2.14.0). Propuesta: hacer el ejemplo robusto (semilla de keras, más `epochs`, otro learning rate o `min_delta`) y reejecutar | S | Prioridad baja (deep learning). Hallado el 2026-11-11 en la pasada de notebooks previa a la 0.27.0 (#1481) | Esta fila |
| BUG-rnn-set-params | `ForecasterRnn.set_params` falla siempre | XS | Prioridad baja | Sección 4 |
| ISSUE-1131 | Exógenas as-of | L | FE-02 cubre la parte foundation; decidir el alcance del resto | Sección 6 |
| DOC-warning-panel-paths | Los paneles de los avisos de skforecast (`Location : {filename}:{lineno}`, `rich_warning_handler` en `skforecast/exceptions/exceptions.py`) publican en la documentación la ruta absoluta de la máquina que ejecutó los notebooks: 20 guías muestran `/Users/<usuario>/code/github/skforecast/...`, y desde un worktree la ruta es aún más larga y se parte en dos líneas. Propuesta: que el ejecutor de notebooks (`tools/docs/execute_notebooks/`) muestre la ruta relativa a la raíz del repositorio (`skforecast/utils/utils.py:4204`), por ejemplo con una variable de entorno que lea el manejador, y no cambiar lo que ve el usuario en su propio entorno. Después, reejecutar las guías afectadas | S | Hallado el 2026-11-11 en la pasada de notebooks previa a la 0.27.0 | Esta fila |
| DOC-crps-guide-arviz | `faq/probabilistic-forecasting-crps-score` falla en la celda de imports: `pymc_marketing` importa `arviz-plots` 1.0.0, que usa `matplotlib.style.core`, ausente en matplotlib 3.11 (`AttributeError`). No es código de skforecast ni de la guía; sus salidas guardadas siguen limpias. Opciones: esperar a una versión compatible de `arviz-plots`, o calcular el CRPS de la guía sin `pymc_marketing` (con `properscoring` y `CRPS`, que importan bien) | S | Hallado el 2026-11-11 (Python 3.14.3, matplotlib 3.11.2) | Esta fila |
| Rama `ci/minimum-versions-job` y #1404 | Borrar la rama; fusionar #1404 (dependabot, contra `main`) o reabrirlo contra la rama de release | XS | | Sección 6 |

Para versiones posteriores (sin cambios): la sección 5 del roadmap de la 0.27.0 (BUG-05,
FE-01 completo, WAPE, Kalman de Arima estacional más rápido, ISSUE-995, ISSUE-974...). Y,
aparcada, la opción c2 de BUG-search-short-series (predecir series con menos historia que la
ventana, cold start), solo como argumento explícito del usuario: su investigación se borró
el 2026-11-11 y está en el historial de git (`dev/roadmap_0.27.0/BUG-search-short-series.md`).

En la 0.29.0: quitar las deprecaciones de la 0.27.0, DEP-cv-differentiation (#1423) y
DEP-datasets-version (#1440).

### 3.8 Orden propuesto (por confirmar)

Con las prioridades de la 0.27.0 (machine learning y foundation primero):

1. BUG-04 y el argumento `adapter` (3.1): resultado erróneo silencioso en foundation.
2. Planes 02 y 03 de foundation, y BUG-06 (requisito de FE-05).
3. FE-05, después BUG-02 y PERF-03.
4. QW-01 y BUG-01 (avisos).
5. QW-11 y FE-06.
6. Mantenimiento de dependencias (sección 1): independiente, en cualquier momento.
7. El resto.
