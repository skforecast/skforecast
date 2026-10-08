---
name: release-note
description: Adds or updates the entry of a finished user-facing change in docs/releases/releases.md, following the skforecast release notes conventions (short entries, highlights with badges, Added/Changed/Fixed grouped by area, pull request links), or consolidates the section of the version in development before a release. Use when a feature, fix, API change, enhancement or documentation change is done, when verify or open-pr finds a user-facing change without a release note, or with the argument `consolidate` before publishing a release.
argument-hint: "[short description of the change, PR or issue number | consolidate]"
---

# Write the release note of a finished change

Change to describe: $ARGUMENTS (if empty, infer it from the branch). If it is `consolidate`, go to [Consolidate before a release](#consolidate-before-a-release).

The release notes are read by skforecast users on the documentation site, most of them to decide whether to upgrade and what will change for them. An entry says what the user sees, in one or two sentences. The technical detail (cause, implementation, benchmarks, every edge case) belongs in the pull request, which the entry links. The code is the source of truth: verify every name, default, argument and error message in the source before writing it.

## 1. Scope the change

- Release branch: `X.Y.x` from `__version__` in `skforecast/__init__.py` (e.g. `0.26.0` → `0.26.x`).
- Changed files: `git diff --name-only <release-branch>...HEAD` plus `git status --short`. Read the diff of the package code and docs, and the commit messages for intent.
- Issue or PR numbers: from `$ARGUMENTS`, the commits or `gh pr view` when a PR exists.

No entry (say so and stop) when the change is not visible to a user of the package:

- Tests, CI, `.claude/`, `dev/`, `tools/`, internal refactors with no change in behavior or performance.
- Private classes, functions and attributes (names starting with `_`), even if they change the internal design.
- Repository files (`SECURITY.md`, `CITATION.cff`, templates, `context7.json`, plugin manifests) and the infrastructure of the documentation site (plugins, templates, how notebooks are rendered).
- A bug in something added in this same cycle: update that entry instead.

Documentation gets an entry only for what a reader would notice: a new user guide, a new section, a redesigned page.

## 2. Read the target section

Read the section of the version in development (`## X.Y.Z <small>In development</small> { id="X.Y.Z" }`, created by `/release-bump`) to find related entries:

- **Same cycle**: if the change modifies something introduced in this same section, update that entry instead of adding a new one. Users never saw the bug, so it is not a Fixed entry.
- **Same object or same symptom**: if an entry already covers the same object or the same kind of problem (for example, "issues of `save_forecaster` with `backend='skops'`"), extend it and add the pull request to its trailer, rather than adding a near-duplicate. Several small fixes of one function are one entry.

Never edit the sections of released versions.

## 3. Classify

| Subsection | What goes there |
|:-----------|:----------------|
| **Added** | New public classes, functions, parameters, attributes, adapters; compatibility with a new version of a dependency (`Added \`torch 2.12\` compatibility.`) |
| **Changed** | Behavior, defaults, renames, removals, deprecations, performance, minimum versions of the dependencies, documentation |
| **Fixed** | Bugs that existed in a released version |

When a subsection has more than about 10 entries, its entries are grouped by area under a label in italics on its own line (`*Performance*`, `*Forecasters*`, `*Multiple series*`, `*Statistical models*` or one per model, `*Foundation models*`, `*Backtesting, hyperparameter search and feature selection*`, `*Save and load*`, `*Dependencies*`, `*Documentation*`...). Use the labels already in the section, add one only when no label fits, and do not use headings (`###`), which would enter the table of contents.

Then decide whether the change also deserves:

- A **highlight** in "The main changes in this release are:". Highlights are for new features, API changes, notable enhancements (with a measured gain) and fixes of wrong results in a common use. At most 8 to 10 per release: if the list is full, either the new one replaces a less important one or it stays out.
- A line in the **"Before upgrading"** admonition (see step 6), when the results of existing code change without any change in that code, or when existing code stops working.

## 4. Write the entry

Format: a bullet starting with `+ `, one blank line between bullets, placed in its group next to the entries on the same topic.

**Length: one or two sentences, about 40 words. Three sentences and 80 words is the ceiling**, for a change that needs a workaround or a migration. If it does not fit, the entry is describing the implementation or several changes at once.

- **Added**: what is new, where it lives and what it does for the user. Name the main options, not every field or attribute.
- **Changed**: the new behavior, the old one when it helps to recognise the change, and the consequence for users ("Results are unchanged", "The selected features for a given `random_state` may differ from previous versions"). If the user has to do something, say what.
- **Fixed**: the symptom the user saw and under which conditions (which forecasters, arguments, estimators). `Fixed an issue in <code>[X]</code> where ...`, or the wrong behavior stated directly (`<code>[X]</code> raised ... when ...`). Add the new behavior only when it is not simply "it works now".
- **Deprecations and removals**: name the version and the replacement. "Deprecated since 0.23.0, `interval` must now be ... Passing ... raises a `ValueError` instead."
- **Performance**: one or two measured numbers with their scenario ("2.3 times faster with 500 series"). Not the full benchmark table and not what was optimized internally.
- **Serialized models**: if the change breaks forecasters saved with previous versions, add after the highlights the admonition used in 0.22.0 and 0.23.0 (`!!! warning "Serialized models incompatibility"`), or tell the user if one already exists.

Leave out:

- The cause of a bug and how it was fixed (algorithms, compilation flags, data structures, "the rows were looked up by label"). One short clause is fine when it tells the user whether they were affected.
- Lists of every parameter, attribute or field involved: name two or three and end with `...`.
- Secondary error messages and edge cases. Quote an exception only when it is short and a user would search for it (`TypeError: 'NoneType' object is not subscriptable`); otherwise name its type (a `CatBoostError` about `cat_features`).
- Internal file paths, private names and names of internal helpers.
- What did not change, except the one sentence that reassures about results.

Style:

- Present tense for the current behavior, past tense for the bug.
- No en dashes or em dashes (contributor rule; some old entries break it). Use commas, colons, semicolons or parentheses.
- Trailers at the end of the entry, in this order: `[User guide](../user_guides/<notebook>.ipynb#<anchor>)` (paths are relative to `docs/releases/`), then the pull request, `([#1234](https://github.com/skforecast/skforecast/pull/1234))`, preceded by the issue it closes when there is one (`.../issues/1234`), several separated by commas.
- **Every entry links its pull request.** It is where the detail left out of the entry lives. If the pull request does not exist yet, write the entry without it and add the link when it is opened (`/open-pr` checks it). A change committed directly to the release branch has no link.
- External contributions: "Thanks to the [Team](url) team for contributing this adapter."

## 5. Write the highlight (if warranted)

The highlight is one sentence, two at most, with the same links and trailers as the entry, preceded by a badge:

```html
<span class="badge text-bg-feature">Feature</span>
<span class="badge text-bg-enhancement">Enhancement</span>
<span class="badge text-bg-api-change">API Change</span>
<span class="badge text-bg-fix">Fix</span>
<span class="badge text-bg-docs">Docs</span>
```

Badges appear only in the highlights, never in Added, Changed or Fixed. Order the highlights by badge (Feature, Enhancement, API Change, Fix, Docs) and by importance within each badge. Merge related highlights (all the gains of one forecaster, the new home page and the new examples page) into one.

## 6. "Before upgrading" admonition

After the highlights, only when the release has something to list:

```markdown
!!! warning "Before upgrading"

    Some results change with this version, without any change in your code:

    + <code>[Ets]</code>: estimates, predictions and prediction intervals change, because ...
    + ...

    And some code needs to be updated: ... See **Changed**.
```

One line per object, saying what changes and, when there is one, how to keep the previous behavior. It summarises entries that are detailed in their subsection; it never replaces them.

## 7. Link references

- Public classes, functions and modules: `<code>[Name]</code>`. Parameters, attributes, methods, values and private names: plain backticks (`fit`, `interval=[0.05, 0.95]`).
- Every `<code>[Name]</code>` needs a definition at the bottom of the file, under `<!-- Links to API Reference -->`, in its `<!-- module -->` group. Reuse the existing one if present.
- New definition: find the object in `docs/api/*.md` (`grep -rn "::: skforecast.*\.Name$" docs/api/`). The anchor is the mkdocstrings path: `[get_model_info]: ../api/FoundationModel.md#skforecast.foundation._model_info.get_model_info`. Forecaster pages without an anchor link to the page (`[ForecasterRecursive]: ../api/ForecasterRecursive.md`).
- If the object is not in any API page, use plain backticks and tell the user that the API page is missing.

## 8. Check

```bash
python - <<'EOF'
import re, pathlib
t = pathlib.Path("docs/releases/releases.md").read_text()
dev = re.search(r"^## .*In development.*?(?=^## )", t, flags=re.M | re.S).group(0)
used = set(re.findall(r"<code>\[([^\]]+)\]</code>", dev))
defined = set(re.findall(r"^\[([^\]]+)\]:", t, flags=re.M))
print("Undefined references:", sorted(used - defined) or "none")
entries = re.findall(r"^\+ .*", dev, flags=re.M)
highlights = [e for e in entries if 'class="badge' in e]
print(f"Words: {len(dev.split())} | entries: {len(entries) - len(highlights)} | highlights: {len(highlights)}")
for e in entries:
    if len(e.split()) > 80:
        print(f"Too long ({len(e.split())} words): {e[:90]}")
no_link = [e for e in entries if "/pull/" not in e and "/issues/" not in e]
print(f"Entries without a pull request or issue link: {len(no_link)}")
EOF
```

The check is limited to the version in development: older releases have known broken references and long entries that are out of scope. Also confirm that the layout of the section is intact: two blank lines before `**Added**`, `**Changed**`, `**Fixed**` and the next `##` heading.

## 9. Report

Show the inserted or updated text, the subsection and group, and whether a highlight or a "Before upgrading" line was added (and why not, if it was not). List anything left for the user: a missing API page, a missing pull request link, detail that was left out and is not in the pull request description either.

## Consolidate before a release

Entries are added one pull request at a time, so before a release the section of the version in development is read as a whole, as a user would. Run it with `/release-note consolidate`, before the date replaces `In development`.

1. Run the check of step 8 and read the whole section.
2. Remove the entries that do not pass the filter of step 1, and those about bugs introduced in this same cycle.
3. Merge the entries about the same object or the same symptom, keeping all their pull request links.
4. Shorten every entry over the ceiling, following step 4.
5. Group the subsections with more than about 10 entries by area, and order the groups from the most to the least used part of the library.
6. Review the highlights (at most 8 to 10, merged and ordered by badge) and write or update the "Before upgrading" admonition from the entries whose results or code change.
7. Run the check again and report: words and entries before and after, the entries removed (so the user can restore one), the entries without link and the wording that needs their judgment.

As a reference, a release with many changes (0.26.0) has about 80 entries and 3,500 words; most releases have 10 to 30 entries.

## Examples

Added:

```markdown
+ New argument `include_drift` in <code>[Arima]</code> to include a linear drift term when the order is specified manually (`d + D <= 1`), equivalent to `include.drift` in R's `forecast::Arima`. `best_params_` now includes `fit_intercept` and `include_drift`, so `set_params` fits exactly the selected model.
```

Changed:

```markdown
+ <code>[select_features]</code> and <code>[select_features_multiseries]</code> sample the records without replacement and keep them in their original order, so selectors with internal cross-validation no longer see the same record in train and validation. The selected features for a given `random_state` may differ from previous versions. ([#1327](https://github.com/skforecast/skforecast/pull/1327))
```

Performance (under Changed):

```markdown
+ Forecasters with an `ExtraTreesRegressor` or an `ExtraTreeRegressor` predict about 15 times faster (33 ms instead of 508 ms for 100 steps with 100 trees), with the same predictions. ([#1363](https://github.com/skforecast/skforecast/pull/1363))
```

Fixed, one issue:

```markdown
+ Fixed an issue in <code>[ForecasterDirect]</code> and <code>[ForecasterDirectMultiVariate]</code> with `differentiation` where the predictions were wrong when `steps` was not consecutive from 1 (for example, `steps=[3, 4, 5]`). It also affected backtesting and hyperparameter search with `gap > 0`. ([#1345](https://github.com/skforecast/skforecast/pull/1345))
```

Fixed, several issues of one function merged:

```markdown
+ Fixed three issues in <code>[backtesting_stats]</code>: `IndexingError: Too many indexers` with `gap > 0`, a single estimator and no interval (also in <code>[grid_search_stats]</code> and <code>[random_search_stats]</code>); an `estimator_params` column not aligned with `estimator_id` with several estimators; and a confusing `NotImplementedError` with an intermittent `refit`. ([#1331](https://github.com/skforecast/skforecast/pull/1331))
```

Too detailed (the cause, the internals and every case, 150 words), and its release note:

```markdown
+ Fixed an issue in <code>[Arima]</code> where the Kalman filter did not propagate the state covariance over missing values. On a missing observation, it kept the previous filtered covariance instead of the predicted one, so the uncertainty did not grow over the gap, and when the series started with missing values the stationary and diffuse priors of the initial state were lost. This affected the likelihood, coefficients, ...

+ Fixed an issue in models estimated by maximum likelihood on series with missing values, where the uncertainty did not grow over the gaps. It affected coefficients, fitted values and prediction intervals. ([#1335](https://github.com/skforecast/skforecast/pull/1335))
```

Highlight with its trailers:

```markdown
+ <span class="badge text-bg-feature">Feature</span> New function <code>[grid_search_equivalent_date]</code> in the <code>[model_selection]</code> module to search the best baseline configuration (`offset`, `n_offsets`, `agg_func`) of a <code>[ForecasterEquivalentDate]</code> using time series backtesting. [User guide](../user_guides/forecasting-baseline.ipynb#searching-for-the-best-configuration)
```
