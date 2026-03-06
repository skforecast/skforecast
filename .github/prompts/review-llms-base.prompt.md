---
description: "Review llms-base.txt for correctness against actual skforecast source code, __init__.py exports, and version"
agent: "agent"
---

# Review llms-base.txt

You are reviewing [tools/ai/llms-base.txt](../../tools/ai/llms-base.txt) — the **single source of truth** for all AI context files in skforecast. Every generated file (`.github/copilot-instructions.md`, `AGENTS.md`, `llms-full.txt`, etc.) derives from this file, so any error here propagates everywhere.

## Reference Sources

Read these files to perform the review:

- [tools/ai/llms-base.txt](../../tools/ai/llms-base.txt) — The file under review
- [skforecast/__init__.py](../../skforecast/__init__.py) — Package version

Also read the `__init__.py` of every public subpackage to verify exports:

- `skforecast/recursive/__init__.py`
- `skforecast/direct/__init__.py`
- `skforecast/deep_learning/__init__.py`
- `skforecast/preprocessing/__init__.py`
- `skforecast/model_selection/__init__.py`
- `skforecast/feature_selection/__init__.py`
- `skforecast/metrics/__init__.py`
- `skforecast/datasets/__init__.py`
- `skforecast/stats/__init__.py`
- `skforecast/drift_detection/__init__.py`
- `skforecast/plot/__init__.py`
- `skforecast/exceptions/__init__.py`
- `skforecast/experimental/__init__.py` (if it exists)

## Review Checklist

Evaluate each item and report **PASS**, **WARN**, or **FAIL** with a brief explanation.

### 1. Version

- [ ] The version string in `llms-base.txt` (e.g., `v0.21.0+`) matches `skforecast/__init__.py` `__version__`
- [ ] Python version list matches `pyproject.toml` `[project] requires-python` and classifiers

### 2. Project Structure

- [ ] Every subpackage directory listed under "Project Structure" actually exists in `skforecast/`
- [ ] No existing subpackage directories are missing from the structure listing
- [ ] The brief descriptions next to each subpackage accurately reflect its contents

### 3. Core Forecasters Table

- [ ] Every forecaster class listed in the table exists in the codebase
- [ ] No forecaster classes are missing from the table
- [ ] The "Use Case" descriptions are accurate

### 4. Key Classes and Imports

This is the most critical section. For each import line in the "Key Classes and Imports" section:

- [ ] The import path is valid (the module and class/function actually exist)
- [ ] Every **public** symbol exported by each subpackage's `__init__.py` is listed
- [ ] No symbols are listed that don't exist or have been removed
- [ ] No deprecated names are used (e.g., `ForecasterAutoreg` instead of `ForecasterRecursive`)

### 5. Code Examples

For each code example in the file:

- [ ] Imports use correct, current module paths
- [ ] Class instantiation uses valid parameter names
- [ ] Method calls use correct parameter names and plausible default values
- [ ] The example is syntactically valid Python
- [ ] No mixing of old and new API conventions

### 6. Datasets Section

- [ ] The `fetch_dataset` names listed actually exist (check `skforecast/datasets/`)
- [ ] The described frequencies and characteristics match the actual datasets

### 7. Documentation Links

- [ ] URLs follow the pattern `https://skforecast.org/latest/...`
- [ ] Section paths (quick-start, user_guides, api, examples, releases) are plausible

### 8. Completeness

- [ ] No major feature or workflow is completely absent from the file
- [ ] Recently added classes or functions (check git log if needed) are documented
- [ ] The "Tips for Best Results" section reflects current best practices

## Output Format

```
## llms-base.txt Review

### Summary
<One paragraph overall assessment — is this file ready for release?>

### Version & Metadata
| Check | Status | Notes |
|-------|--------|-------|
| Version match | PASS/FAIL | ... |
| Python versions | PASS/FAIL | ... |

### Project Structure
| Check | Status | Notes |
|-------|--------|-------|
| All dirs listed | PASS/FAIL | ... |
| No dirs missing | PASS/FAIL | ... |

### Imports Audit
| Subpackage | Exports in __init__.py | Listed in llms-base.txt | Status |
|------------|----------------------|------------------------|--------|
| recursive | ForecasterRecursive, ... | ForecasterRecursive, ... | PASS/FAIL |
| direct | ... | ... | ... |
| ... | ... | ... | ... |

### Code Examples
| Example | Location | Status | Notes |
|---------|----------|--------|-------|
| Basic usage | line ~X | PASS/FAIL | ... |
| Multi-series | line ~X | PASS/FAIL | ... |
| ... | ... | ... | ... |

### Issues Found
1. **[FAIL/WARN]** Description
   - Line: ~X in llms-base.txt
   - Current: `<what it says>`
   - Expected: `<what it should say>`
   - Fix: <exact text replacement>

### Missing Symbols
<List any public symbols from __init__.py files that are NOT in llms-base.txt>

### Suggested Improvements
<Optional improvements that aren't errors>
```

Provide exact text fixes for every issue found. If the file is fully correct, confirm it is ready for release.
