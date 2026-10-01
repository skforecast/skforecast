---
name: ai-context-sync
description: Keeps skforecast's AI context files (AGENTS.md, llms-full.txt, Copilot instructions, user-facing skills) in sync after a code or docs change. Use when a public signature, default, return value, export or forecaster capability changes, when editing tools/ai/llms-base.txt or skills/, or when the ai-context-check CI job fails.
---

# Sync the AI context files

The generated files (`AGENTS.md`, `.github/copilot-instructions.md`, `llms-full.txt`, `docs/llms-full.txt`, `docs/llms.txt`) are built by `tools/ai/generate_ai_context_files.py`. Never edit them; edit the sources and regenerate.

## 1. Find what the change affects

For every public function, class, parameter or attribute that changed, grep it in the three source locations:

```bash
grep -rn "<name>" tools/ai/llms-base.txt skills/ llms.txt
```

Check signatures, defaults, return values (number and order of unpacked values), import paths and capability tables (for example the foundation adapters table).

## 2. Edit the right source

| What changed | Source to edit |
|:-------------|:---------------|
| Contributor rules (testing, style, environment) | `tools/ai/ai_context_header.md` |
| Public API, imports, quick examples | `tools/ai/llms-base.txt` |
| A forecasting workflow | `skills/<name>/SKILL.md` and its `references/` |
| Short index for LLMs | `llms.txt` (root) |
| New public export | Import line in `tools/ai/llms-base.txt` (the `--check` run compares it with each subpackage `__init__.py`) |
| New user-facing skill | `skills/<name>/SKILL.md` and `SKILL_ORDER` in the generator (body at or below 500 lines) |

Keep the user-facing skills about using skforecast. Contributor workflows belong in `.claude/skills/`, which the generator ignores.

## 3. Regenerate and check

```bash
python tools/ai/generate_ai_context_files.py
python tools/ai/generate_ai_context_files.py --check
```

`--check` validates frontmatter, versions, exports, Python snippets and freshness, but not whether examples in the sources still match the code. Review the diff of `llms-full.txt` to confirm the examples you touched read correctly.

## 4. Report

List the sources edited, the generated files that changed, and the result of `--check`.
