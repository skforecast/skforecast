---
name: release-note
description: Adds or updates the entry of a finished user-facing change in docs/releases/releases.md, following the skforecast release notes conventions (badges, highlights, Added/Changed/Fixed, API reference links). Use when a feature, fix, API change, enhancement or documentation change is done, or when verify or open-pr finds a user-facing change without a release note.
argument-hint: "[short description of the change, PR or issue number]"
---

# Write the release note of a finished change

Change to describe: $ARGUMENTS (if empty, infer it from the branch).

The release notes are read by skforecast users on the documentation site. Describe what changes for them, not how the code was written. The code is the source of truth: verify every name, default, argument and error message in the source before writing it.

## 1. Scope the change

- Release branch: `X.Y.x` from `__version__` in `skforecast/__init__.py` (e.g. `0.26.0` → `0.26.x`).
- Changed files: `git diff --name-only <release-branch>...HEAD` plus `git status --short`. Read the diff of the package code and docs, and the commit messages for intent.
- Issue or PR numbers: from `$ARGUMENTS`, the commits or `gh pr view` when a PR exists.
- Not user-facing, no entry: tests only, internal refactors with no change in behavior or performance, CI, `.claude/`, `dev/`. Say so and stop.

## 2. Read the target section

Read the section of the version in development (`## X.Y.Z <small>In development</small> { id="X.Y.Z" }`, created by `/release-bump`) and the previous release, to match the tone and find related entries:

- **Same cycle**: if the change modifies something introduced in this same section (for example, a bug in a feature not released yet), update that entry instead of adding a new one. Users never saw the bug, so it is not a Fixed entry.
- **Same object**: if an entry already describes a change of the same object, extend it rather than adding a near-duplicate.

Never edit the sections of released versions.

## 3. Classify

| Subsection | What goes there |
|:-----------|:----------------|
| **Added** | New public classes, functions, parameters, attributes, adapters; compatibility with a new version of a dependency (`Added \`torch 2.12\` compatibility.`) |
| **Changed** | Behavior, defaults, renames, removals, deprecations, performance, documentation, site and repository changes |
| **Fixed** | Bugs that existed in a released version |

Then decide whether the change also deserves a **highlight** in "The main changes in this release are:". Highlights are for new features, API changes, notable enhancements (with measured gains) and major docs work (a new user guide, a redesigned page). Small fixes and minor changes go only in their subsection. An API change is detailed under Changed and highlighted with the API Change badge.

## 4. Write the detailed entry

Format: a bullet starting with `+ `, one blank line between bullets. Place it next to the entries on the same topic in its subsection (for example, all `Arima` fixes together); otherwise at the end of the subsection.

- **Added**: what is new and where it lives, what it does for the user, the key options. Name the module with its link: `New function <code>[x]</code> in the <code>[model_selection]</code> module to ...`.
- **Changed**: the old behavior, the new behavior and the consequence for users ("Behavior is unchanged", "The selected features for a given `random_state` may differ from previous versions", "forecasters pickled by earlier versions cannot be loaded").
- **Fixed**: usually `Fixed an issue in <code>[X]</code> where ...`; a short fix can also state the wrong and the correct behavior directly (`<code>[X]</code> raised ... when ... Now ...`). Give the symptom (the exact exception text when there is one, e.g. `TypeError: 'NoneType' object is not subscriptable`), the cause in one sentence, who is affected (which forecasters, functions and settings) and the new behavior.
- **Deprecations and removals**: name the version. "Deprecated since 0.23.0, `interval` must now be ... Passing ... raises a `ValueError` instead."
- **Performance**: give the measured numbers and the scenario ("removes around 0.3 seconds from the import of every forecaster module").
- **Serialized models**: if the change breaks forecasters saved with previous versions, add after the highlights the admonition used in 0.22.0 and 0.23.0 (`!!! warning "Serialized models incompatibility"`), or tell the user if one already exists.

Style:

- Present tense for the current behavior, past tense for the bug.
- Public names users see, not internal helpers (unless the helper name is what appears in the error or in the API).
- No en dashes or em dashes (contributor rule; some old entries break it). Use commas, colons, semicolons or parentheses.
- Trailers at the end of the entry, in this order: `[User guide](../user_guides/<notebook>.ipynb#<anchor>)` (paths are relative to `docs/releases/`), then `([#1234](https://github.com/skforecast/skforecast/issues/1234))` or `.../pull/1234`, several separated by commas.
- External contributions: "Thanks to the [Team](url) team for contributing this adapter."

## 5. Write the highlight (if warranted)

The highlight is a shorter version of the detailed entry (usually its first one or two sentences), with the same links and trailers, preceded by a badge:

```html
<span class="badge text-bg-feature">Feature</span>
<span class="badge text-bg-enhancement">Enhancement</span>
<span class="badge text-bg-api-change">API Change</span>
<span class="badge text-bg-fix">Fix</span>
<span class="badge text-bg-docs">Docs</span>
```

Badges appear only in the highlights, never in Added, Changed or Fixed. Order the highlights by badge (Feature, Enhancement, API Change, Fix, Docs) and by importance within each badge.

## 6. Link references

- Public classes, functions and modules: `<code>[Name]</code>`. Parameters, attributes, methods, values and private names: plain backticks (`fit`, `interval=[0.05, 0.95]`).
- Every `<code>[Name]</code>` needs a definition at the bottom of the file, under `<!-- Links to API Reference -->`, in its `<!-- module -->` group. Reuse the existing one if present.
- New definition: find the object in `docs/api/*.md` (`grep -rn "::: skforecast.*\.Name$" docs/api/`). The anchor is the mkdocstrings path: `[get_model_info]: ../api/FoundationModel.md#skforecast.foundation._model_info.get_model_info`. Forecaster pages without an anchor link to the page (`[ForecasterRecursive]: ../api/ForecasterRecursive.md`).
- If the object is not in any API page, use plain backticks and tell the user that the API page is missing.

## 7. Check

```bash
python - <<'EOF'
import re, pathlib
t = pathlib.Path("docs/releases/releases.md").read_text()
dev = re.search(r"^## .*In development.*?(?=^## )", t, flags=re.M | re.S).group(0)
used = set(re.findall(r"<code>\[([^\]]+)\]</code>", dev))
defined = set(re.findall(r"^\[([^\]]+)\]:", t, flags=re.M))
print("Undefined references:", sorted(used - defined) or "none")
EOF
```

The check is limited to the version in development: older releases have known broken references that are out of scope. Also confirm that the layout of the section is intact: two blank lines before `**Added**`, `**Changed**`, `**Fixed**` and the next `##` heading.

## 8. Report

Show the inserted or updated text, the subsection and whether a highlight was added (and why not, if it was not). List anything left for the user: a missing API page, an admonition, wording that needs their judgment.

## Examples from previous releases

Added:

```markdown
+ New argument `include_drift` in <code>[Arima]</code> to include a linear drift term when the order is specified manually (`d + D <= 1`), equivalent to `include.drift` in R's `forecast::Arima`. The `best_params_` attribute found by the automatic model selection now also includes `fit_intercept` and `include_drift`, so passing them to `set_params` fits exactly the selected model.
```

Changed:

```markdown
+ `TimesFMAdapter` renamed to <code>[TimesFM25Adapter]</code> (`'google/timesfm-2.5-*'` ids). The adapter is reached through <code>[FoundationModel]</code>, so user code is unaffected, but forecasters pickled by earlier versions with a `TimesFMAdapter` cannot be loaded.
```

Fixed:

```markdown
+ Fixed an issue in <code>[backtesting_forecaster_multiseries]</code> where <code>[ForecasterDirectMultiVariate]</code> raised `TypeError: 'NoneType' object is not subscriptable` with `use_in_sample_residuals=False`, because only one of `out_sample_residuals_` and `out_sample_residuals_by_bin_` was restored after each `fit()` depending on `use_binned_residuals`. Both attributes are now restored.
```

Highlight with its trailers:

```markdown
+ <span class="badge text-bg-feature">Feature</span> New function <code>[grid_search_equivalent_date]</code> in the <code>[model_selection]</code> module to search the best baseline configuration (`offset`, `n_offsets`, `agg_func`) of a <code>[ForecasterEquivalentDate]</code> using time series backtesting. [User guide](../user_guides/forecasting-baseline.ipynb#searching-for-the-best-configuration)
```
