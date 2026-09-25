# AI Context System

This directory contains the source files and generation script that power all AI-assisted development in skforecast. Every AI context file in the repository — IDE instructions, LLM references, and documentation copies — is generated from the files here.

## File types at a glance

VS Code + GitHub Copilot recognises several file conventions. Each serves a different role:

| File type | Location | What it is | Best-practice objective |
|-----------|----------|------------|------------------------|
| **copilot-instructions.md** | `.github/copilot-instructions.md` | Global project context injected into **every** Copilot Chat conversation. Contains the API reference, code style, and testing conventions that Copilot should always know about. | Provide the AI with a persistent baseline of project-specific knowledge so it never generates code that contradicts the architecture, style, or API conventions of the repo. |
| **instructions** | `.github/instructions/*.instructions.md` | Targeted coding conventions that activate **only when the open file matches** an `applyTo` glob in the YAML frontmatter (e.g. docstring rules for `*.py`, testing rules for `tests/`). | Keep the global context lean by extracting domain-specific rules into separate files that load only when relevant, avoiding prompt bloat while ensuring precision in specialized tasks. |
| **prompts** | `.github/prompts/*.prompt.md` | Reusable prompt templates the developer runs **on demand** from the Copilot Chat prompt picker. Used for review checklists and guided workflows. | Standardize repetitive developer workflows (reviews, audits, migrations) into shareable, version-controlled templates that any team member can invoke identically. |
| **skills** | `skills/*/SKILL.md` | Self-contained workflow guides that the AI agent **discovers automatically** when the user's question matches the skill description. Each skill covers one topic end-to-end. | Encode deep domain knowledge (decision trees, pitfalls, end-to-end examples) in modular documents the agent retrieves on demand, enabling expert-level guidance without overloading the base context. |
| **agents** | `.github/agents/*.agent.md` | Optional custom agent modes that appear in the Copilot Chat **agent picker** when such files exist. This repository does not currently define custom agents. | Future extension point for purpose-built agent configurations (e.g. a "reviewer" agent that only reads files and runs tests, or a "docs" agent restricted to documentation folders) so developers get a tailored experience without manual prompt engineering. |
| **AGENTS.md** | `AGENTS.md` (repo root) | Equivalent to `copilot-instructions.md` but for IDEs that follow the AGENTS.md convention (Claude Code, Codex CLI, Aider). Same content, different standard. | Ensure consistent AI behaviour across different IDEs and tools by providing the same project context through each tool's native convention. |

### Analogy: cooking a dish

Imagine you hire a chef (the AI) to work in your kitchen (the repo).

- **copilot-instructions.md** is the **house rules** posted on the kitchen wall: "we only use olive oil, knives go in this drawer, clean as you go." The chef reads them once and follows them in every dish, every day.
- **instructions** are **station cards** taped next to each workstation: "at the grill, sear 2 min per side" or "at the pastry station, always sift flour." The chef only reads the card for the station she's working at right now.
- **prompts** are **recipe cards** in a drawer. The chef never opens the drawer on her own — you hand her a specific card when you say "follow this exact recipe for the soufflé." She executes it step by step, once.
- **skills** are **cooking reference books** on the shelf (one for sauces, one for bread, one for plating). The chef pulls one down when you ask something like "how do I make a béchamel?" — she knows which book to grab by the topic.
- **agents** are **specialist hats**. When they exist, selecting one changes the chef's focus, tools, and constraints for the whole session.

| Kitchen analogy | AI file type | Loaded when |
|-----------------|-------------|-------------|
| House rules on the wall | `copilot-instructions.md` | Always |
| Station card at the grill | `.instructions.md` | Editing a matching file |
| Recipe card you hand over | `.prompt.md` | You explicitly invoke it |
| Reference book on the shelf | `skills/*/SKILL.md` | AI decides it's relevant |
| Specialist hat | `.agent.md` | You select the agent mode |

## How VS Code / GitHub Copilot uses these files

When you open the skforecast repo in VS Code with GitHub Copilot, different files are loaded into the AI's context at different times. Understanding when each file type activates is key to understanding the system.

### copilot-instructions.md (always active)

`.github/copilot-instructions.md` is **injected into every Copilot Chat conversation** automatically. The user does nothing — VS Code reads this file and appends it to the system prompt. This is the baseline context that Copilot always has about skforecast: project structure, API overview, code style, and testing conventions.

- **When**: every conversation, every message
- **Scope**: the entire repository
- **Content**: `ai_context_header.md` (dev conventions) + `llms-base.txt` (API reference)

### .instructions.md files (auto-activated by file pattern)

Files in `.github/instructions/` have an `applyTo` glob in their YAML frontmatter. VS Code automatically adds the matching instruction file to the context **only when the user is editing a file that matches the pattern**. This keeps the context focused — test conventions only appear when writing tests.

| File | Pattern | Active when editing |
|------|---------|---------------------|
| `testing.instructions.md` | `**/tests/**` | Any file inside a `tests/` directory |
| `docstrings.instructions.md` | `skforecast/**/*.py` | Any Python file in the package |

- **When**: only when the active file matches `applyTo`
- **Scope**: additive — loaded on top of `copilot-instructions.md`
- **Content**: detailed conventions that would be too verbose for the global context

### .prompt.md files (user-invoked)

Files in `.github/prompts/` are **reusable prompt templates** that the user explicitly runs via the Copilot Chat prompt picker (type `/` or use the attachment button). They are never loaded automatically.

| File | Purpose |
|------|---------|
| `review-llms-base.prompt.md` | Checklist to review `llms-base.txt` against actual source code |
| `review-skill.prompt.md` | Checklist to review a skill folder for API correctness |

- **When**: only when the user explicitly selects the prompt
- **Scope**: single conversation
- **Content**: structured review checklists with references to source files

### Skills (discovered on demand)

Files in `skills/*/SKILL.md` are **specialized workflow guides** that AI agents can discover and load when the user asks about a specific topic. In VS Code, Copilot reads skills from the `skills/` directory when their description matches the user's question.

- **When**: on demand, when the agent determines a skill is relevant
- **Scope**: single conversation
- **Content**: end-to-end workflows with decision trees, code examples, and pitfalls

### .agent.md files (custom agent modes)

Files in `.github/agents/` define **custom agent modes** that appear in the Copilot Chat agent picker (the model/mode dropdown). Each `.agent.md` file creates a specialized AI persona with its own system prompt and, optionally, restricted tools or scoped instructions. skforecast does not currently ship custom agents; this is documented as a future extension point.

- **When**: only when the user selects the agent mode from the picker
- **Scope**: entire conversation while that mode is active
- **Content**: a YAML frontmatter (`name`, `description`, `tools`) plus a system-level prompt body that shapes the agent's behaviour

### AGENTS.md (other IDEs)

`AGENTS.md` at the repo root serves the same purpose as `copilot-instructions.md` but for IDEs that follow the AGENTS.md convention (Claude Code, Codex CLI, Aider, and others). It contains identical content.

- **When**: automatically on project open, depending on the IDE
- **Scope**: entire repository

### Summary: context loading order

```
Always loaded:
  └─ .github/copilot-instructions.md        (global API + code style)

Loaded when file pattern matches:
  └─ .github/instructions/docstrings.instructions.md   (when editing *.py)
  └─ .github/instructions/testing.instructions.md      (when editing tests/)

Loaded on demand by AI agent:
  └─ skills/*/SKILL.md                       (topic-specific workflows)

Optional future extension if agent files are added:
  └─ .github/agents/*.agent.md               (custom agent personas)

Loaded when user explicitly invokes:
  └─ .github/prompts/review-*.prompt.md      (review checklists)
```

## Strategy

The AI context system follows three principles:

1. **Single source of truth** — All generated files derive from `llms-base.txt` and `ai_context_header.md`. Edit the source, regenerate, done.
2. **Layered context** — Global context is always present, specialized context loads only when relevant, avoiding prompt bloat.
3. **CI-enforced consistency** — A GitHub Actions workflow runs `--check` mode on every PR to prevent stale generated files from reaching main.

## Source files (human-maintained)

These are the only files you edit directly:

| File | Lines | Purpose |
|------|-------|---------|
| `tools/ai/llms-base.txt` | ~670 | Core API reference: all forecasters, imports, examples, workflows |
| `tools/ai/ai_context_header.md` | ~40 | Dev-only context: testing commands, code style, dependencies |
| `llms.txt` (root) | ~120 | Public index per [llmstxt.org](https://llmstxt.org) spec with links to docs |
| `skills/*/SKILL.md` | 17 skills | Modular workflow guides, one per topic |
| `skills/*/references/*.md` | 12 files | Supplementary reference tables for some skills |
| `.github/instructions/*.md` | 2 files | Pattern-matched coding conventions |
| `.github/prompts/*.md` | 2 files | Reusable review checklists |
| `context7.json` (root) | ~50 | Context7 indexing config: excluded folders and files, agent `rules` |
| `.claude-plugin/marketplace.json` | ~25 | Claude Code plugin marketplace that publishes `skills/` as the `skforecast` plugin |

## Generated files (do not edit)

All marked with `<!-- AUTO-GENERATED -->` header and tracked in `.gitattributes` as `linguist-generated=true` (collapsed in GitHub PR diffs).

| File | Source | Content |
|------|--------|---------|
| `.github/copilot-instructions.md` | header + llms-base | IDE context for GitHub Copilot |
| `AGENTS.md` | header + llms-base | IDE context for Claude Code, Codex, Aider |
| `llms-full.txt` | llms-base + 17 skills | Complete LLM reference (~5000 lines) |
| `docs/llms.txt` | copy of root `llms.txt` | Served at skforecast.org/latest/llms.txt |
| `docs/llms-full.txt` | copy of `llms-full.txt` | Served at skforecast.org/latest/llms-full.txt |

## Skills

17 self-contained workflow guides in `skills/`. Each has a `SKILL.md` with YAML frontmatter (`name`, `description`) and optional `references/` subfolder.

| Skill | References | Topic |
|-------|-----------|-------|
| `forecasting-single-series` | — | ForecasterRecursive / ForecasterDirect |
| `forecasting-multiple-series` | — | ForecasterRecursiveMultiSeries global model |
| `statistical-models` | `model-parameters.md` | ARIMA, SARIMAX, ETS, ARAR |
| `metric-selection` | `metric-compatibility.md` | Choosing evaluation metrics (MASE, RMSSE, CRPS) |
| `backtesting-configuration` | — | TimeSeriesFold / OneStepAheadFold configuration |
| `hyperparameter-optimization` | `search-parameters.md` | Grid, random, Bayesian search |
| `prediction-intervals` | `interval-compatibility.md` | Bootstrapping, conformal, quantile |
| `autocorrelation-and-lag-selection` | — | ACF / PACF analysis and lag selection |
| `feature-engineering` | `rolling-stats-reference.md`, `calendar-features-reference.md` | RollingFeatures, calendar features |
| `feature-selection` | — | RFECV, SelectFromModel |
| `drift-detection` | — | RangeDriftDetector, PopulationDriftDetector |
| `deep-learning-forecasting` | `architecture-options.md` | ForecasterRnn, LSTM/GRU |
| `foundation-forecasting` | `adapter-parameters.md` | ForecasterFoundation, Chronos-2, TimesFM 2.5/3.0, Moirai-2, TabICL, TabPFN-TS, TFC-T0, Synthefy Nori, TS-ICL |
| `choosing-a-forecaster` | — | Decision guide for forecaster selection |
| `baseline-forecasting` | — | ForecasterEquivalentDate and naive baselines |
| `troubleshooting-common-errors` | — | Common mistakes and fixes |
| `complete-api-reference` | `forecaster-constructors.md`, `forecaster-methods.md`, `model-selection-signatures.md`, `preprocessing-signatures.md` | All constructor and method signatures |

## Distribution to user agents

The files above only reach someone working inside this repository. Three channels
deliver the same content to a user working in their own project. None of them
duplicates `skills/`: all read the folder at the repository root.

| Channel | Config in this repo | What the user runs |
|---------|---------------------|--------------------|
| **Claude Code plugin** | `.claude-plugin/marketplace.json` | `/plugin marketplace add skforecast/skforecast`, then `/plugin install skforecast@skforecast` |
| **Any Agent Skills client** (Cursor, Copilot, Codex, Gemini CLI, ...) | None: [`npx skills`](https://github.com/vercel-labs/skills) discovers `skills/*/SKILL.md` | `npx skills add skforecast/skforecast` |
| **Context7** (MCP docs server) | `context7.json` | Nothing: agents with the Context7 MCP server query `/skforecast/skforecast` |

Notes:

- The plugin entry uses `"source": "./skills"` with `"strict": false` and `"skills": "."`,
  so only the `skills/` folder (about 300 KB) is copied to the user plugin cache, not
  the whole repository. Validate changes with `claude plugin validate . --strict`.
- All three channels read the default branch (`main`), so users get the skills of the
  latest release, and changes made in a release branch take effect at release time.
- `context7.json` `rules` are injected into the agent together with the retrieved
  snippets. Keep them short (max 255 characters each, max 50 rules) and limited to
  two kinds: the default path (which forecaster to choose and how to evaluate it, from
  `skills/choosing-a-forecaster`) and mistakes LLMs actually make (removed names and
  arguments, from `skills/troubleshooting-common-errors`). Rules are prepended to every
  query, so leave niche topics to the indexed docs.
- `AGENTS.md` stays indexed on purpose: it is the only indexed copy of
  `tools/ai/llms-base.txt` (the `tools/` folder and `llms-full.txt` are excluded).
- `"branch": "main"` is explicit because the Context7 index was created when the
  default branch was `master`, which no longer exists.
- `.github/workflows/context7-refresh.yml` asks Context7 to re-index the library when
  the indexed content changes in `main` (or on demand with "Run workflow"). Without it,
  Context7 refreshes unpopular libraries every 45 days at most. It needs the repository
  secret `CONTEXT7_API_KEY`, created at [context7.com/dashboard](https://context7.com/dashboard).
  If the indexed paths in `context7.json` change, update the `paths` filter of the workflow.
- Claiming the library at [context7.com](https://context7.com) (a manual step for a
  maintainer) unlocks an admin panel, version management and faster refreshes.
- There is no pip-based skills installer on purpose: it would require shipping a copy
  of `skills/` inside the package and maintaining a per-agent directory mapping that
  `npx skills` already maintains.

## Generation script

```bash
# Generate all files
python tools/ai/generate_ai_context_files.py

# CI mode: fail if any generated file is stale or missing
python tools/ai/generate_ai_context_files.py --check

# Validate URLs in llms.txt are reachable
python tools/ai/generate_ai_context_files.py --check-urls

# Skip specific URL patterns during validation
python tools/ai/generate_ai_context_files.py --check-urls --ignore-urls llms-full.txt
```

URL validation requests each link with `HEAD`, falls back to `GET` when the server
rejects or stalls on `HEAD`, and retries transient failures (timeouts, connection
errors, HTTP 408/425/429/5xx) up to 3 times with an increasing delay.

A `https://skforecast.org/latest/...` page that returns 404 is reported as
"pending publication" instead of an error when its source exists in `docs/`
(`<path>.ipynb`, `<path>.md` or `<path>/index.md`). The documentation site is
deployed at release time, so a user guide added during a release cycle is not
reachable until then. A 404 with no local source (a typo, or a page that was
removed) still fails the check.

### What `--check` validates

| Check | What it verifies |
|-------|-----------------|
| **Skill structure** | Every `skills/*/SKILL.md` has valid YAML frontmatter, `name` matches directory, body ≤ 500 lines |
| **Version consistency** | `Version:` in `llms-base.txt` and the plugin `version` in `.claude-plugin/marketplace.json` match `__version__` in `skforecast/__init__.py`. `CITATION.cff` has no version on purpose; if one is added, it must match too |
| **Distribution manifests** | `context7.json` and `.claude-plugin/marketplace.json` are valid JSON, every non-glob `excludeFolders` entry and the plugin `source` exist, and each Context7 rule is at most 255 characters |
| **Imports consistency** | Every public export in subpackage `__init__.py` files appears as an import in `llms-base.txt` |
| **File freshness** | Each generated file matches what the script would produce right now |

### CI enforcement

`.github/workflows/ai-context-check.yml` runs `--check` on every pull request targeting `main` or a release branch (`*.x`). If any generated file is stale, the PR check fails with a message indicating which files need regeneration.

## File map

```
skforecast/
├── llms.txt                              # Public index (human-maintained)
├── llms-full.txt                         # Complete reference (generated)
├── AGENTS.md                             # IDE context (generated)
├── context7.json                         # Context7 indexing config (human-maintained)
├── .claude-plugin/
│   └── marketplace.json                  # Claude Code plugin marketplace (human-maintained)
├── .gitattributes                        # Marks generated files
├── .github/
│   ├── copilot-instructions.md           # IDE context (generated)
│   ├── instructions/
│   │   ├── docstrings.instructions.md    # → skforecast/**/*.py
│   │   └── testing.instructions.md       # → **/tests/**
│   ├── prompts/
│   │   ├── review-llms-base.prompt.md    # Review checklist for llms-base.txt
│   │   └── review-skill.prompt.md        # Review checklist for skills
│   └── workflows/
│       ├── ai-context-check.yml          # CI: validates generated files
│       └── context7-refresh.yml          # Re-index Context7 when main changes
├── skills/
│   ├── forecasting-single-series/
│   │   └── SKILL.md
│   ├── complete-api-reference/
│   │   ├── SKILL.md
│   │   └── references/
│   │       └── forecaster-constructors.md (and 3 more)
│   └── ... (15 more skills)
├── tools/ai/
│   ├── README.md                         # This file
│   ├── llms-base.txt                     # Core API reference (source)
│   ├── ai_context_header.md              # Dev conventions (source)
│   └── generate_ai_context_files.py      # Generation + validation script
└── docs/
    ├── llms.txt                          # Website copy (generated)
    └── llms-full.txt                     # Website copy (generated)
```

## Common tasks

**Add a new public class or function to the API:**

1. Edit `tools/ai/llms-base.txt` — add the import line and any relevant docs
2. Run `python tools/ai/generate_ai_context_files.py`
3. Commit the source change and all regenerated files

**Add a new skill:**

1. Create `skills/<skill-name>/SKILL.md` with YAML frontmatter (`name`, `description`)
2. Add the skill name to `SKILL_ORDER` in `generate_ai_context_files.py`
3. Optionally add `skills/<skill-name>/references/*.md` for supplementary content
4. Run `python tools/ai/generate_ai_context_files.py`

**Add a new instruction file:**

1. Create `.github/instructions/<name>.instructions.md` with YAML frontmatter (`description`, `applyTo`)
2. No regeneration needed — VS Code picks it up directly

**Review a skill for correctness:**

1. Open Copilot Chat and invoke the `review-skill` prompt
2. Specify which skill to review

**Bump the version:**

1. Update `__version__` in `skforecast/__init__.py`
2. Update `Version:` in `tools/ai/llms-base.txt`
3. Update the plugin `version` in `.claude-plugin/marketplace.json`
4. Regenerate — the `--check` validation will catch mismatches

## Claude Code harness

Claude Code (VS Code extension, CLI and cloud sessions on claude.ai/code) reads `CLAUDE.md` and the tracked `.claude/` directory. Everything a teammate or a cloud session needs is committed; personal preferences stay in each person's `~/.claude/`.

| What | Where | Shared |
|:-----|:------|:-------|
| Always-loaded instructions | `CLAUDE.md`, which imports `tools/ai/ai_context_header.md` (not the full `AGENTS.md`, to keep the fixed context small) | Yes |
| Path-scoped rules (tests, docstrings, foundation, docs, AI context files) | `.claude/rules/*.md`, loaded when Claude reads a file matching their `paths` | Yes |
| Permissions, env, hooks and attribution (off: commits and PRs are authored by the user alone) | `.claude/settings.json` | Yes |
| Hooks (standard library Python) | `.claude/hooks/`: `protect_generated.py` blocks edits to generated files, `ruff_check.py` reports new ruff findings after an edit, `session_start_remote.py` installs the environment in cloud sessions, `attribution_guard.py` blocks AI attribution in commits and PRs. Tests: `python -m pytest .claude/hooks -q -p no:cacheprovider` | Yes |
| Contributor workflows | `.claude/skills/`: `verify` (definition of done: lint, affected tests, conditional checks), `ai-context-sync`, `/open-pr`, `/release-bump`, `/handoff` (writes `dev/handoff_<slug>.md` to continue in another session) | Yes |
| Machine-specific permissions | `.claude/settings.local.json` (git-ignored) | No |
| Personal instructions for this repo | `CLAUDE.local.md` (git-ignored) | No |
| Model, effort, attribution, permission mode | `~/.claude/settings.json` | No |
| Auto memory | `~/.claude/projects/<repo>/memory/` (machine-local, never reaches cloud sessions) | No |

Conventions that should apply to everyone go into `CLAUDE.md` or `.claude/rules/`, not into auto memory.

### Local setup (VS Code)

1. Use Claude Code 2.1.283 or later (`claude --version`), needed for teleport and the current permission modes.
2. Open the repository with the conda environment active. If the extension does not inherit it, enable `claudeCode.usePythonEnvironment` in the VS Code user settings.
3. Optional VS Code user settings: `claudeCode.initialPermissionMode` (the starting mode cannot be set from project settings).
4. Hooks run with `python3` (or `python` if `python3` is missing). On Windows, if hooks fail, disable the `python3` App Execution Alias of the Microsoft Store so the real interpreter is used.
5. Run `/hooks` and `/memory` once to confirm the project hooks and rules are loaded.

### Cloud setup (claude.ai/code)

1. Connect GitHub (Claude GitHub App, or `/web-setup` from the CLI).
2. Create an environment for the repository with network access **Trusted**. If the first session cannot install CPU torch or the `fetch_dataset` tests fail, switch to **Custom**, keep the default domains and add `download.pytorch.org` and `raw.githubusercontent.com`.
3. No setup script and no secrets are needed: the `SessionStart` hook creates `~/.venvs/skforecast` with uv, installs `-e ".[test]"` with CPU torch, and puts it on `PATH`. Resumed sessions skip the install unless `pyproject.toml` changed.
4. Cloud sessions push to their own branch; open the PR against the release branch (`X.Y.x`) with `/open-pr`.

Moving work between places: `claude --cloud "task"` sends a task to a cloud session (push the branch first), and `claude --teleport` (or the Web tab of the session history in VS Code) brings a cloud session back to the local checkout.
Teleport only goes from the cloud to local and needs a clean working tree; in any other direction (local to cloud, or to the other maintainer), run `/handoff <slug>` and continue from `dev/handoff_<slug>.md`.
