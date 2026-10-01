---
name: review-user-guide
description: Reviews an existing skforecast user guide (notebook or Markdown) against the current code, reports the findings and waits for approval before editing it. Use when a guide has to be checked, updated or improved for a documentation release.
disable-model-invocation: true
argument-hint: "<path to the user guide, e.g. docs/user_guides/feature-selection.ipynb>"
---

# Review a user guide

User guide to review: $ARGUMENTS

The code in this repository is the only source of truth. Prior knowledge of skforecast may be outdated: when it conflicts with the repo, the repo wins. Never assume an API exists; verify it in the source.

Keep the guide simple: follow "Keep user guides simple" in `.claude/rules/docs.md`. A good review proposes cuts and links as often as additions.

## Step 1. Gather context (read only, no edits)

1. Read the whole guide: markdown cells, code cells and stored outputs.
2. List every skforecast class, function, argument and attribute it uses, and read their signatures, defaults and docstrings in the source.
3. Read the release notes in `docs/releases/releases.md` (version in development and the last few releases) and note the changes that affect the guide: renamed, deprecated or removed APIs, changed defaults, new options.
4. Check that the guide will run against the repo code, not an installed copy. The notebook kernel runs from the guide's folder, where the repo is not on the path, so a regular `pip install skforecast` in the environment shadows the repo. Run the check from that folder:
   ```bash
   cd docs/user_guides && python -c "import skforecast; print(skforecast.__version__, skforecast.__file__)"
   ```
   The path must point to the repo's `skforecast/` folder. If it points to `site-packages`, tell the user (fix: `pip install -e .`) and, until then, run every execution in this skill with `PYTHONPATH=<repo root>`.
5. Read 2 or 3 other guides in the same folder to learn the house conventions: structure, admonitions, heading levels, plotting style, dataset loading, tone.
6. Check `mkdocs.yml` for where the guide sits in the navigation and which related guides it could link to.

## Step 2. Execute

For a notebook, it must have no uncommitted changes (otherwise ask first). Execute it in place with the repo's runner, which collects warnings and converts the progress bars:

```bash
python tools/docs/execute_notebooks/execute_notebooks.py <path relative to docs/, e.g. user_guides/feature-selection.ipynb>
```

The original stays in git: compare the new outputs with `git diff`, and restore it with `git restore <guide>` if the user does not approve any change. For a Markdown guide, copy its code blocks into a temporary script and run it from the repo root.

Record every error, every warning (the runner's log in `tools/docs/execute_notebooks/logs/`, especially `DeprecationWarning` and `FutureWarning`) and every output that differs meaningfully from the stored one. If execution is too slow or needs unavailable resources (GPU, gated models), tell the user and continue with a static review.

## Step 3. Review

Four passes, in this order of priority:

1. **Technical correctness**: API usage against the current source, errors and warnings, consistency between code, outputs and text (every sentence that quotes a result must match the output), forecasting methodology (data leakage, splits, backtesting setup, exogenous variables and their future values, metrics, interpretation of results).
2. **Currency**: options added in recent versions that the guide should show, and explanations that are missing (what problem a feature solves, when to use it, key parameters, pitfalls). Prefer a link to the API reference or another guide over repeating it.
3. **Structure**: logical flow (motivation, concept, minimal example, advanced usage, takeaways), redundant sections, code cells without explanation, reproducibility (fixed seeds, datasets from `skforecast.datasets`), execution time.
4. **Language and style**: grammar, clarity, concision, consistent terminology, the house conventions from Step 1. Keep the author's voice.

## Step 4. Report and stop

Before editing anything, present:

1. **Summary**: at most 5 sentences.
2. **Findings table**: `#` | `Location` (cell number or section) | `Pass` | `Severity` (Critical, Major, Minor, Suggestion) | `Issue` | `Proposed fix`, sorted by severity. For API findings, cite the source file and line that prove the issue.
3. **Proposed enhancements**: new sections, examples or explanations, each with a short justification, and the cuts that would make the guide shorter.
4. **Open questions**: anything uncertain or that needs a maintainer decision, including library bugs or API changes found during the review.

Then wait for approval. The user may accept all, accept some by number, or ask for changes.

## Step 5. Apply the approved changes

- Edit the guide in place. For notebooks, edit cell by cell with the notebook editing tool and keep the cell structure, metadata and cell types.
- Prefer minimal, targeted edits over rewrites. Never remove content without a stated reason.
- All code must be complete and runnable.
- Re-execute the guide with the runner (Step 2) and confirm it finishes with no errors and no deprecation warnings. Never hand-write outputs.
- After the re-execution, check again every sentence that quotes an output (number of selected features, metric values, which features are kept): new outputs often make them stale.
- Update `mkdocs.yml` or other files only if the user approved it.
- Finish with a short list of everything modified.

## Constraints

- Do not modify library code, tests or other guides. Report library bugs in Open questions instead of fixing them.
- Do not commit or push.
- Delete any temporary files you created.
