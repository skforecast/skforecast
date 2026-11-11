# Roadmap de skforecast 0.28.0

Fecha: 2026-11-11. Base: rama `0.27.x` (0.27.0 en desarrollo). Las rutas son relativas a la
raíz del repositorio.

Documento inicial: de momento solo recoge el mantenimiento de dependencias que se aplazó en
la revisión de `pyproject.toml` del 2026-11-11. Lo que el roadmap de la 0.27.0 deja para
después no se copia aquí: sigue en su apartado "Para después de 0.27.0" (sección 0) y en su
sección 5, en [`dev/roadmap_0.27.0/roadmap.md`](../roadmap_0.27.0/roadmap.md).

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
| DEP-python310 | Retirar Python 3.10 | S | Decidido el 2026-11-11: se retira en 0.28.0 y se anuncia en 0.27.0 |
| MAINT-docs-deps | Actualizar los pins del extra `docs`, salvo `mkdocs-jupyter` | S | Opcional, medido: sin efecto visible |
| MAINT-test-deps | Limpiar el extra `test` | S | Opcional |

Los tres son pequeños y no dependen entre sí, salvo `tomli` (MAINT-test-deps), que se quita
con DEP-python310.

No forma parte de este roadmap: `torch>=2.4, <2.16` se subió en la 0.27.0 (commit
`7f7c4b804`, junto con la retirada de `pytest-xdist` del extra `test`). Probado con torch
2.15.0: `pytest skforecast/deep_learning` da 157 passed, 45 skipped (los de TensorFlow, que
no se ejecutan en Python 3.14).

### DEP-python310: retirar Python 3.10

Ya estaba en el roadmap de la 0.27.0 ("anunciar en 0.27 y retirar en 0.28"). El anuncio en
la nota de la 0.27.0 sigue pendiente.

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

Cambios:

- [ ] `pyproject.toml:65`: `requires-python = ">=3.11"`; quitar el classifier de la línea 26.
- [ ] `pyproject.toml:69,74,84,112`: `statsmodels>=0.13.3`. La 0.13.2 no tiene wheel para
  Python 3.11 y el resolvedor de mínimos ya elige la 0.13.3. El resto de mínimos se resuelven
  igual en 3.11 que en 3.10.
- [ ] `.github/workflows/unit-tests.yml:22`: quitar `'3.10'` de la matriz.
- [ ] `.github/workflows/unit-tests-min-deps.yml:27-30`: Python 3.11.
- [ ] `.github/workflows/publish-pypi.yml:142`: `['3.11', '3.14']`.
- [ ] `skforecast/utils/tests/tests_utils/test_check_optional_dependency.py:5`: `tomli` por
  `tomllib`.
- [ ] `skforecast/foundation/tests/tests_foundation_models/test_backends_compat_config.py:9-12`:
  quitar el `try/except` de `tomli`.
- [ ] `.claude/skills/verify/test_affected.py:365`: quitar el `skipif` de Python menor que 3.11.
- [ ] `pyproject.toml:123`: quitar `tomli>=2.0` del extra `test`.
- [ ] `skforecast/foundation/_adapters.py:3561`: quitar la nota de Python 3.10 del docstring
  de `T0Adapter`. Decidir si el mínimo pasa a `tfc-t0[torch]>=0.6.1`
  (`tools/foundation_compat/adapters.toml:53`).
- [ ] Docs: `docs/quick-start/how-to-install.md:5,10`, `README.md:11` (badge) y
  `docs/skforecast-ai/skforecast-ai.md:3,30` (comprobar antes si skforecast-ai sigue la
  misma versión mínima).
- [ ] Contexto IA: `tools/ai/llms-base.txt:13` y `tools/ai/ai_context_header.md:68`
  (`uv venv ... --python 3.10`), y regenerar (skill `ai-context-sync`).
- [ ] Nota en `docs/releases/releases.md`, sección 0.28.0.

Sin comprobar: la suite con las versiones mínimas en Python 3.11 (es la suite completa, la
ejecuta `unit-tests-min-deps.yml`). Solo se resolvieron las versiones, con
`uv pip compile --resolution lowest-direct --python-version 3.11 --only-binary :all:` en
macOS arm64; falta confirmarlo en Linux.

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
  cambia la resolución.
- [ ] `tomli>=2.0`: se quita con DEP-python310.

---

## 2. Otros hallazgos de la misma revisión (sin tarea asignada)

- **`pandas<3.0` se mantiene.** El workflow `unit-tests-latest-deps.yml` fuerza pandas 3.1.1
  saltándose el límite y da 91 failed y 6732 passed en Python 3.14, sobre `main` (run del
  2026-11-09; 91 failed también en 3.12 y 3.13). Son fallos de índices, dtypes, zonas
  horarias y frecuencias. El workflow lleva en rojo al menos desde el 2026-10-05. El trabajo
  de compatibilidad está en `dev/pandas3_migration/`.
- **tensorflow**: el límite `<2.22` es correcto hoy (última estable: 2.21.0). La 2.22.0rc0 ya
  trae wheels para Python 3.14: cuando salga la final, subir a `<2.23` y revisar el marcador
  `python_version < "3.14"` (`pyproject.toml:86`) y los `skipif` de los tres tests de
  TensorFlow en `skforecast/deep_learning/tests/`.
- **Límites inferiores**: correctos. `unit-tests-min-deps.yml` pasó el 2026-11-03 con las
  versiones mínimas declaradas en Python 3.10 (7363 passed, 27 skipped).
