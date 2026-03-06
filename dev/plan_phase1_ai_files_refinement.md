# Fase 1 — Refinamiento de archivos AI

> Revisión crítica de cada archivo, propósito real, y decisiones sobre qué mantener, eliminar o mejorar.

**Fecha**: 2026-03-06  
**Estado**: Draft  
**Prerequisito**: Todos los archivos de la estrategia AI ya existen en el repo (branch `feature__ai_enhancements`)

---

## Estado actual

Ya están creados **todos** los archivos que contemplaba el plan original. El sistema funciona: un script genera 8 archivos derivados a partir de 4 fuentes manuales. Antes de seguir avanzando, hay que decidir qué archivos realmente merecen la pena y refinar los que se quedan.

---

## Inventario completo

### Archivos fuente (mantenidos a mano)

| # | Archivo | Líneas | Propósito |
|---|---------|--------|-----------|
| 1 | `tools/ai/llms-base.txt` | ~730 | **La fuente única de verdad.** Referencia completa de la API: estructura del proyecto, todos los forecasters, imports, ejemplos, datasets, tips. Es el contenido que todos los demás archivos consumen. |
| 2 | `tools/ai/ai_context_header.md` | ~65 | **Header solo para contributors.** Comandos de pytest, estilo de código (NumPy docstrings, PEP 8, ruff), dependencias. Se antepone al contenido de `llms-base.txt` en los archivos de IDE. Un usuario que hace `pip install skforecast` nunca lo ve — solo es útil para quien clona el repo y desarrolla. |
| 3 | `llms.txt` (raíz) | ~113 | **Índice público per [llmstxt.org](https://llmstxt.org).** Un directorio de enlaces a documentación, quick start, guías, API. Los LLMs con web search (ChatGPT, Perplexity, Grok) lo buscan automáticamente en `dominio.org/llms.txt`. No contiene contenido real — es un mapa de navegación. |
| 4 | `skills/` (12 directorios) | ~80-130 c/u | **Módulos de conocimiento especializado.** Cada skill es un workflow completo (ej: "cómo hacer forecasting de una sola serie", "cómo elegir el forecaster correcto"). Siguen el estándar [Agent Skills spec](https://agentskills.io). Los agentes AI avanzados (Copilot, Claude Code) los cargan bajo demanda. |

### Archivos generados (por `generate_ai_context_files.py`)

| # | Archivo | Líneas | Generado desde | Propósito |
|---|---------|--------|----------------|-----------|
| 5 | `llms-full.txt` (raíz) | ~2600 | `llms-base.txt` + todos los `skills/` | **Contexto completo para cualquier LLM.** Un usuario puede pegar la URL `skforecast.org/llms-full.txt` en ChatGPT, Claude, Gemini, etc. y el LLM tiene toda la información. También sirve como knowledge file para Custom GPTs. |
| 6 | `.github/copilot-instructions.md` | ~800 | `ai_context_header.md` + `llms-base.txt` | **Contexto para GitHub Copilot en VS Code.** Se inyecta automáticamente en cada prompt cuando el repo está abierto. El IDE más usado del mundo. |
| 7 | `AGENTS.md` | ~800 | `ai_context_header.md` + `llms-base.txt` | **Contexto para Claude Code y OpenAI Codex.** Ambos lo leen automáticamente. Es el archivo estándar *de facto* que la mayoría de agentes AI buscan en la raíz de un repo. |
| 8 | `.claude/CLAUDE.md` | ~800 | `ai_context_header.md` + `llms-base.txt` | **Contexto para Claude Code.** Claude Code busca primero `.claude/CLAUDE.md` y luego `AGENTS.md`. Como ambos tienen el mismo contenido, este archivo es **redundante**. |
| 9 | `.windsurfrules` | ~800 | `ai_context_header.md` + `llms-base.txt` | **Contexto para Windsurf (Codeium IDE).** Windsurf lo lee automáticamente. Sin embargo, las versiones recientes de Windsurf también leen `AGENTS.md`. Cuota de mercado muy baja. |
| 10 | `.cursor/rules/skforecast.mdc` | ~805 | YAML frontmatter + `ai_context_header.md` + `llms-base.txt` | **Contexto para Cursor IDE.** Cursor tiene su propio formato `.mdc` con frontmatter YAML (`description`, `globs`). Le indica a Cursor que active estas reglas para archivos `**/*.py`. |
| 11 | `docs/llms.txt` | ~113 | Copia de `llms.txt` raíz | **Copia para la web.** MkDocs sirve este archivo en `skforecast.org/llms.txt`. Sin esta copia, el índice no sería accesible en la URL pública. |
| 12 | `docs/llms-full.txt` | ~2600 | Copia de `llms-full.txt` raíz | **Copia para la web.** MkDocs lo sirve en `skforecast.org/llms-full.txt`. Es la URL que los usuarios pegan en sus LLMs. |

### Archivos de configuración/soporte

| # | Archivo | Propósito |
|---|---------|-----------|
| 13 | `tools/ai/generate_ai_context_files.py` | **El script generador.** Lee las 4 fuentes, valida (versión, imports, frontmatter de skills), y genera los 8 archivos derivados. Modo `--check` para CI. Sin este script, mantener 8 archivos a mano sería insostenible. |
| 14 | `.gitattributes` | **Colapsa diffs en GitHub.** Marca los archivos generados como `linguist-generated=true` para que los PRs no muestren 800 líneas de diff por cada archivo. 9 líneas, imprescindible. |
| 15 | `.github/agents/skforecast-dev.agent.md` | **Agente personalizado para VS Code.** Define un "Skforecast Development Agent" especializado en vibe coding. Contiene estructura del proyecto, forecasters, estilo de código, módulos clave. Solo funciona en VS Code con GitHub Copilot. |
| 16 | `pyproject.toml` (línea en `[project.urls]`) | **URL pública.** `"LLM Context" = "https://skforecast.org/llms.txt"` — aparece en PyPI y en metadata del paquete. Coste: 1 línea. |

### Archivos que NO existen (mencionados en el plan original)

| Archivo | Estado | Decisión propuesta |
|---------|--------|-------------------|
| `.github/instructions/tests.md` | No existe | **Posponer.** Requiere análisis manual de patterns de tests. Copilot ya tiene `copilot-instructions.md` con estilo de código general. Solo vale la pena si se detecta que Copilot genera tests malos. |
| CI workflow `ai-context-drift-check.yml` | No existe | **Pendiente para Fase 2.** Útil pero no urgente. De momento, correr `--check` manualmente antes del release. |

---

## Análisis: qué mantener y qué no

### ✅ MANTENER — Archivos imprescindibles

#### 1. `tools/ai/llms-base.txt` — La fuente única

**Por qué es necesario:** Sin este archivo no hay contenido para ningún otro. Es la API reference completa que alimenta todo el sistema.

**Qué revisar:**
- [ ] ¿La versión (`v0.21.0+`) coincide con `skforecast/__init__.py`? El script lo valida, pero hay que correrlo.
- [ ] ¿Todos los imports públicos de cada `__init__.py` están representados? El script también lo valida.
- [ ] ¿Los ejemplos de código son correctos y ejecutables con la versión actual?
- [ ] ¿Faltan forecasters, clases o funciones nuevas?

**Acción:** Ejecutar `python tools/ai/generate_ai_context_files.py --check` y corregir lo que reporte.

---

#### 2. `tools/ai/ai_context_header.md` — Header de desarrollo

**Por qué es necesario:** Sin este header, los archivos de IDE no tendrían información de testing ni estilo de código. Un LLM que edite código en el repo no sabría que se usa ruff, docstrings NumPy, ni cómo correr los tests.

**Qué revisar:**
- [ ] ¿Las versiones mínimas de dependencias son correctas (`numpy>=1.26`, `pandas>=2.1`, etc.)?
- [ ] ¿Los comandos de pytest son los correctos?
- [ ] ¿Falta alguna convención de código importante (ej: single quotes string, que está en `ruff.toml`)?

**Acción:** Comparar con `ruff.toml`, `pyproject.toml` y `pytest.ini` para asegurar consistencia.

---

#### 3. `llms.txt` (raíz) — Índice público

**Por qué es necesario:** Es el estándar de la industria. ChatGPT, Perplexity, y otros LLMs con web search buscan `/llms.txt` automáticamente. No tenerlo es perder visibilidad gratis.

**Qué revisar:**
- [ ] ¿Los enlaces apuntan a páginas que existen en la documentación actual?
- [ ] ¿Falta alguna sección importante de la documentación?
- [ ] ¿El enlace a `llms-full.txt` es correcto?

**Acción:** Verificar cada URL manualmente o con un script.

---

#### 4. `skills/` — Módulos de conocimiento

**Por qué son necesarios:** Sin skills, `llms-full.txt` sería idéntico a `llms-base.txt` y no habría workflows detallados. Los 12 skills cubren todos los casos de uso principales de skforecast.

**Qué revisar para cada skill:**
- [ ] ¿El código de ejemplo es correcto y ejecutable?
- [ ] ¿Usa la API actual (no hay métodos deprecados)?
- [ ] ¿El frontmatter tiene `name` y `description` válidos?
- [ ] ¿Está dentro del límite de 500 líneas?
- [ ] ¿El `name` del frontmatter coincide con el nombre del directorio?

**Skills existentes:**

| Skill | Qué cubre |
|-------|-----------|
| `forecasting-single-series` | Workflow completo con ForecasterRecursive: datos, fit, predict, backtest, intervalos |
| `forecasting-multiple-series` | ForecasterRecursiveMultiSeries: series como columnas o dict, encoding, predicciones |
| `statistical-models` | ForecasterStats con Arima, Sarimax, Ets, Arar: auto ARIMA, seasonal, backtesting |
| `hyperparameter-optimization` | Grid, random, bayesian search con TimeSeriesFold y OneStepAheadFold |
| `prediction-intervals` | Bootstrapping, conformal, cuantiles: predict_interval, backtesting con intervalos |
| `feature-engineering` | RollingFeatures, DateTimeFeatureTransformer, custom transformers, exog variables |
| `feature-selection` | RFECV, SelectFromModel: selección de lags, window features y exog |
| `drift-detection` | RangeDriftDetector, PopulationDriftDetector: monitoreo en producción |
| `deep-learning-forecasting` | ForecasterRnn con Keras: create_and_compile_model, LSTM, series temporales |
| `choosing-a-forecaster` | Guía de decisión: "tengo X situación → usa Y forecaster" |
| `troubleshooting-common-errors` | Errores frecuentes que cometen los LLMs + soluciones correctas |
| `complete-api-reference` | Índice + `references/method-signatures.md` con firmas detalladas de todos los métodos públicos |

**Acción:** Revisar cada skill uno por uno, verificar ejemplos contra la API actual.

---

#### 5. `generate_ai_context_files.py` — Script generador

**Por qué es necesario:** Es lo que hace viable mantener 8+ archivos desde 4 fuentes. Sin él, cada cambio de API requeriría editar 8 archivos manualmente.

**Qué revisar:**
- [ ] ¿Genera correctamente sin errores?
- [ ] ¿El modo `--check` detecta archivos desactualizados?
- [ ] ¿La validación de imports es completa?
- [ ] ¿Hay subpackages nuevos que no estén en la lista de validación?

**Acción:** Ejecutar el script y verificar que pasa sin warnings.

---

#### 6. `.github/copilot-instructions.md` — GitHub Copilot

**Por qué es necesario:** GitHub Copilot es el asistente AI más usado del mundo en IDEs. Este archivo se inyecta automáticamente sin que el usuario haga nada. Coste: cero (auto-generado).

**Acción:** Solo verificar que se genera correctamente.

---

#### 7. `AGENTS.md` — Agentes genéricos

**Por qué es necesario:** Es el archivo estándar *de facto* para agentes AI. Lo leen:
- **Claude Code** (como fallback si no hay `.claude/CLAUDE.md`)
- **OpenAI Codex**
- **Aider**
- **Continue.dev**
- **LangChain agents**
- Y cualquier herramienta que siga la convención emergente

Es el archivo con **mayor cobertura de herramientas** de todos. Coste: cero (auto-generado).

**Acción:** Solo verificar que se genera correctamente.

---

#### 8. `llms-full.txt` — Contexto completo

**Por qué es necesario:** Es la distribución universal. Cualquier ser humano puede pegar `skforecast.org/llms-full.txt` en cualquier LLM del mundo y obtener respuestas correctas sobre skforecast. No depende de ningún IDE, plataforma, ni herramienta específica.

**Acción:** Verificar que la concatenación de skills es correcta y legible.

---

#### 9. `docs/llms.txt` + `docs/llms-full.txt` — Copias para web

**Por qué son necesarios:** Sin estas copias, MkDocs no puede servir los archivos en `skforecast.org/llms.txt` y `skforecast.org/llms-full.txt`. Las URLs públicas dejarían de funcionar.

**Acción:** Verificar que son copias idénticas de los archivos raíz.

---

#### 10. `.gitattributes` — Colapsar diffs

**Por qué es necesario:** Sin él, cada PR que toque `llms-base.txt` mostraría ~6400 líneas de diff (800 × 8 archivos generados). Con `.gitattributes`, GitHub los colapsa automáticamente.

**Acción:** Verificar que lista todos los archivos generados.

---

#### 11. `pyproject.toml` — URL en metadata

**Por qué es necesario:** 1 línea que hace que `skforecast.org/llms.txt` aparezca en la página de PyPI. Cualquier LLM que busque info del paquete en PyPI lo encuentra.

**Acción:** Verificar que la URL es correcta.

---

#### 12. `.github/agents/skforecast-dev.agent.md` — Agente de desarrollo

**Por qué es necesario:** Define un agente especializado en VS Code para desarrollo del repo. A diferencia de `copilot-instructions.md` (que da contexto pasivo), el agent define un *rol* activo: "eres un experto en time series, tu trabajo es extender skforecast". Útil para vibe coding dentro del repo.

**Qué revisar:**
- [ ] ¿La estructura del proyecto está actualizada?
- [ ] ¿Los forecasters listados coinciden con los actuales?
- [ ] ¿Las convenciones de código son consistentes con `ai_context_header.md`?
- [ ] ¿Las referencias a utils, preprocessing, exceptions son correctas?

**Acción:** Comparar con `llms-base.txt` y asegurar consistencia.

---

### ❌ ELIMINAR — Archivos que no merecen la pena

#### `.claude/CLAUDE.md` — Redundante con `AGENTS.md`

**Propósito original:** Claude Code busca `.claude/CLAUDE.md` con prioridad sobre `AGENTS.md`.

**Por qué eliminarlo:**
- Claude Code lee `AGENTS.md` perfectamente si `.claude/CLAUDE.md` no existe.
- Ambos tienen **contenido idéntico**.
- Añade un directorio (`.claude/`) a la raíz del repo sin aportar nada extra.
- Si mañana Claude Code cambia su precedencia, `AGENTS.md` sigue funcionando.

**Impacto de eliminarlo:** Cero. Claude Code usa `AGENTS.md` como fallback.

**Acción:** Eliminar `.claude/CLAUDE.md` y el directorio `.claude/`. Quitar de `generate_ai_context_files.py` y `.gitattributes`.

---

#### `.windsurfrules` — Bajo retorno, contamina la raíz

**Propósito original:** Windsurf IDE lo lee automáticamente.

**Por qué eliminarlo:**
- Windsurf tiene cuota de mercado muy baja (<5% entre devs que usan AI coding).
- Las versiones recientes de Windsurf **también leen `AGENTS.md`**.
- El archivo está en la **raíz del proyecto** sin extensión, contaminando el directorio.
- 800 líneas de un archivo que casi nadie usa.

**Impacto de eliminarlo:** Mínimo. Los pocos usuarios de Windsurf que clonen el repo tendrán `AGENTS.md`.

**Acción:** Eliminar `.windsurfrules`. Quitar de `generate_ai_context_files.py` y `.gitattributes`.

---

### ⚠️ VALORAR — Decisión del equipo

#### `.cursor/rules/skforecast.mdc` — Depende del uso interno

**Propósito:** Cursor IDE tiene su propio formato con frontmatter YAML que permite configurar description y globs.

**A favor de mantenerlo:**
- Cursor tiene una base de usuarios significativa (más que Windsurf).
- El frontmatter `globs: "**/*.py"` permite que Cursor active las reglas solo para archivos Python — algo que `AGENTS.md` no puede hacer.
- Es auto-generado, coste de mantenimiento ~cero.

**En contra:**
- Cursor también lee `AGENTS.md` como fallback.
- Añade un directorio `.cursor/rules/` al repo.
- Si nadie del equipo usa Cursor, es peso muerto.

**Recomendación:** Mantenerlo si alguien del equipo usa Cursor. Eliminarlo si no.

---

## Plan de acción

### Paso 1 — Ejecutar validación

```bash
python tools/ai/generate_ai_context_files.py --check
```

Verificar que no hay errores de versión, imports, ni skills mal formados.

### Paso 2 — Eliminar archivos redundantes

1. Eliminar `.claude/CLAUDE.md` y directorio `.claude/`
2. Eliminar `.windsurfrules`
3. Actualizar `generate_ai_context_files.py`: quitar ambos de `IDE_TARGETS`
4. Actualizar `.gitattributes`: quitar ambas líneas
5. Regenerar: `python tools/ai/generate_ai_context_files.py`

### Paso 3 — Revisar contenido de archivos fuente

Para cada archivo fuente, verificar que:

- [ ] `tools/ai/llms-base.txt` — Versión correcta, imports completos, ejemplos actuales
- [ ] `tools/ai/ai_context_header.md` — Dependencias y convenciones actualizadas
- [ ] `llms.txt` — URLs válidas, secciones completas
- [ ] `skills/` — Código ejecutable, API actual, frontmatter válido (12 skills)

### Paso 4 — Revisar agente de desarrollo

- [ ] `.github/agents/skforecast-dev.agent.md` — Consistente con `llms-base.txt`

### Paso 5 — Regenerar y verificar

```bash
python tools/ai/generate_ai_context_files.py
```

Verificar que los archivos generados son correctos.

### Paso 6 — Decidir sobre Cursor

- [ ] ¿Alguien del equipo usa Cursor? → Mantener `.cursor/rules/skforecast.mdc`
- [ ] ¿Nadie lo usa? → Eliminar y quitar de `generate_ai_context_files.py`

---

## Resumen visual

Tras la limpieza, el sistema quedaría así:

```
FUENTES (mantener a mano)          GENERADOS (por el script)
─────────────────────────          ────────────────────────
tools/ai/llms-base.txt      ───→  .github/copilot-instructions.md  (Copilot)
tools/ai/ai_context_header.md     AGENTS.md                        (Claude Code, Codex, Aider, ...)
llms.txt                     ───→  docs/llms.txt                   (web)
skills/ (12 skills)          ───→  llms-full.txt                   (cualquier LLM)
                                   docs/llms-full.txt              (web)
                                   .cursor/rules/skforecast.mdc    (Cursor, opcional)

SOPORTE
───────
tools/ai/generate_ai_context_files.py   (generador + validador)
.gitattributes                           (colapsar diffs en PRs)
.github/agents/skforecast-dev.agent.md   (agente VS Code para vibe coding)
pyproject.toml [project.urls]            (URL en PyPI)
```

**Archivos fuente:** 4 (+ 12 skills)  
**Archivos generados:** 5-6 (según decisión de Cursor)  
**Archivos de soporte:** 4  
**Total:** ~25 archivos → **~22 archivos** tras limpieza  
**Eliminados:** `.claude/CLAUDE.md`, `.windsurfrules` (+ sus entradas en script y gitattributes)
