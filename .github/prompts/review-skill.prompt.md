---
description: "Review a skforecast skill folder for correctness, completeness, and consistency with the current API"
agent: "agent"
---

# Review Skill Folder

You are reviewing a **skforecast skill** inside the `skills/` directory. A skill is a self-contained workflow document (`SKILL.md`) that teaches an AI agent how to perform a specific skforecast task. Some skills also have a `references/` subfolder with supplementary files.

## Skill to Review

Review the skill folder at: `skills/{{skill-name}}/`

Read every file in the skill folder before starting the review.

## Reference Sources

Use these files as the source of truth for the current API:

- [tools/ai/llms-base.txt](../../tools/ai/llms-base.txt) — Complete API reference (imports, classes, methods, examples)
- [skforecast/__init__.py](../../skforecast/__init__.py) — Package version

Also inspect the relevant source modules under `skforecast/` when you need to verify method signatures, parameter names, or default values.

## Review Checklist

Evaluate each item and report **PASS**, **WARN**, or **FAIL** with a brief explanation.

### 1. Frontmatter

- [ ] Has valid YAML frontmatter between `---` markers
- [ ] `name` field matches the skill directory name exactly
- [ ] `description` field exists and is a clear, concise summary of when to use this skill
- [ ] Description contains trigger phrases that help AI agents discover it

### 2. Content Quality

- [ ] Has a clear "When to Use" section explaining the use case
- [ ] Provides a complete, end-to-end workflow (not just snippets)
- [ ] Covers the main happy path and common variations
- [ ] Does NOT duplicate large blocks of content from `llms-base.txt` — it should complement, not repeat

### 3. Code Correctness

- [ ] All imports use the **current** module paths (check against `llms-base.txt` and actual `__init__.py` files)
- [ ] No deprecated class names (e.g., `ForecasterAutoreg` → `ForecasterRecursive`, `ForecasterAutoregMultiSeries` → `ForecasterRecursiveMultiSeries`)
- [ ] Method calls use correct parameter names and signatures (verify against source code if uncertain)
- [ ] Examples are syntactically valid Python
- [ ] Default values in examples match the actual defaults in the source code

### 4. API Consistency

- [ ] All classes and functions mentioned exist in the current version
- [ ] Import paths match the actual package structure under `skforecast/`
- [ ] No references to removed or renamed parameters
- [ ] Version-specific notes (if any) reference the correct version

### 5. Size and Structure

- [ ] Total skill content is within ~500 lines (skills should be focused, not exhaustive)
- [ ] Well-organized with clear headings and logical flow
- [ ] If it has a `references/` subfolder, those files are also correct and referenced properly

## Output Format

Provide your review as:

```
## Skill Review: <skill-name>

### Summary
<One paragraph overall assessment>

### Checklist Results
| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Frontmatter valid | PASS/WARN/FAIL | ... |
| ... | ... | ... | ... |

### Issues Found
1. **[FAIL/WARN]** Description of the issue
   - File: `SKILL.md` line X
   - Current: `<what it says>`
   - Expected: `<what it should say>`
   - Fix: <suggested correction>

### Suggested Improvements
- <Optional improvements that aren't errors but would make the skill better>
```

If you find issues, propose specific fixes with the exact text to change. If everything passes, confirm the skill is in good shape.
