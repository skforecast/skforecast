# Code Review: `skforecast/foundation/` — Bugs, Best Practices, Optimization


### Tier 1: quick correctness and robustness wins — DONE

### Tier 2: internal consistency and small perf (low risk, no API change) — DONE

### Tier 3: shared adapter base class and delegated refactors (biggest de-dup, enabler)
Highest effort. Split into three PRs (C1 mechanical helpers, C2 base class + migration, C3 Moirai behavior change) so the one intentional behavior change stays isolated from the pure refactor. Do NOT bundle all of this as a single "pure refactor" diff.

The bullets below are ordered to follow the recommended implementation order (PR C1 helpers first, then PR C2 base class, then PR C3 Moirai behavior change).

**PR C1 (mechanical helpers and extractions, zero behavior change):**
> NOTE: all `:line` references in this section have drifted relative to the current source. Re-locate every site by content (and re-verify the site counts) before implementing.

- **Mechanical helpers**: `_validate_positive_int(name, value)` (16 `context_length` sites verified byte-identical, message `` `{name}` must be a positive integer. Got {value!r}.``; Moirai's split f-string concatenates to the same string, so it collapses cleanly; also covers TimesFM `max_horizon`, 2 more sites). A `_tensor_to_numpy` detach helper scoped to the **3 torch-output sites (Chronos / T0 / TSICL)**, returning the tensor's native dtype (no float cast). **Nori's `_to_numpy` is deliberately excluded and left unchanged**: it is a semantically different operation (defensively coerce an unknown `NoriRegressor.predict()` output to `float64`), NOT a detach of a known `float32` tensor, so the two contracts are NOT unified.
- **Descoped from C1** (reviewed, not worth the abstraction):
  - Covariate-conversion helper: only 2 real owners (`_to_covariate_array` in Chronos and TSICL, not 4), sharing only a 2-line numeric→`float32` cast; the substance is the diverging else-branch (Chronos passes non-numeric through, TSICL raises with two owner-specific pinned messages). A shared helper would hide that difference behind a flag/callback and risk breaking the pinned TSICL messages. Optionally a tiny `_numeric_to_float32(col)` micro-helper, marginal for 2 owners.
  - Quantile-to-index helper: a bare `.index(q)` does not warrant a helper. Keep only a shared 9-level grid *constant* for the fixed-grid adapters (TimesFM / Moirai).
- **ForecasterFoundation-side extractions (in `_forecaster_foundation.py`)**: all **descoped from C1**.
  - `_predict_impl`: `predict` / `predict_quantiles` differ only by `quantiles=None` vs `quantiles=list(quantiles)`; too thin to warrant a shared method, kept as independent bodies.
  - `_check_is_fitted_or_context(context, method)`: the triplicated `NotFittedError` guard differs only in the method name in the message; a 3-4 line raise-guard at 3 sites is thinner than `_predict_impl` and not worth extracting. Leave the guards inline.
- **FoundationModel-side extraction (in `_foundation_model.py`)**: `_reset_fit_state()` **descoped from C1**. The three reset blocks are NOT identical (`__init__` interleaves `creation_date` + version lines, `set_params` adds 3 adapter resets), so a shared helper would cover only a ~10-line core while callers keep heterogeneous extras around it, splitting reset state across helper + caller for marginal savings at 3 sites. Left as-is.

**PR C2 (`_FoundationAdapterBase` base class + migration):**
- **3.1 `_FoundationAdapterBase` base class**: share the `get_params` / `set_params` skeleton (invalid-key check, apply loop, compare-and-reset, `f"Invalid parameter(s) for {type(self).__name__}: ..."` message, ordered `get_params` assembly) across all 8 adapters. This is NOT purely declarative, the code forces overridable hooks:
  - Reset targets are heterogeneous (Chronos nulls `_pipeline`, five null `_model`, Moirai nulls `_module` + `_forecast_obj`, TSICL splits `_model` vs `_resolved_device`), so the base needs an overridable `_reset_cache(changed_keys)` hook rather than a single declared reset-attr name.
  - `get_params` normalizes output (`predict_kwargs or None` `:200`, `tabicl_config or None` `:1455`, `nori_config or None`), so the param spec must carry a per-key getter transform, not just a name (T0 `test_T0Adapter.py:58-63` and Nori `test_NoriAdapter.py:103-110` pin the exact returned dict).
  - Per-adapter validators (point_estimate, mode, show_progress, n_fourier_terms, add_calendar_features, max_horizon) stay per-subclass; the base only shares the skeleton. Keep it a plain base with hooks, not a metaclass or validator-DSL. Realistic saving is the ~40-line get/set_params body per adapter, not ~350 lines.
  - Migrate the 3 already-compatible compare-and-reset adapters (TabICL / TabPFN / Nori at `_adapters.py:1515 / 2147 / 3691`) first, then the 5 key-presence adapters (Chronos / TimesFM / Moirai / T0 / TSICL).
- **1.7 TabICL / TabPFN duplication (scope honestly)**: the base class removes only the ~80 shared get/set_params lines from this pair. The bulk of the ~594 / ~639-line near-duplication is in `fit` / `predict` / `_build_context_df` / `_build_future_df` / timestamp helpers and is a SEPARATE, larger, behaviorally riskier refactor (TabPFN adds `mode`, a third `point_estimate='mode'`, different config). Do not scope it under the get/set_params base.

**PR C3 (Moirai behavior change, isolate it):**
- **1.6 Moirai `set_params` (intentional behavior change)**: switch from key-presence reset (`_adapters.py:1051-1053`) to compare-and-reset, and special-case `device`-only changes to `.to(...)` the cached `_forecast_obj` (reusing the MPS-fallback logic in `_ensure_forecast_obj`, `:1244-1253`) instead of nulling it. Test-safe: no existing test pins "unchanged value still resets" (both `test_MoiraiAdapter.py:114-133` and `test_device_handling.py:234-245` only pass changed values), but ADD a test for the new no-reset-on-unchanged and device-`.to()` paths. Preserve TSICL's deliberate device-is-not-a-model-reset-key behavior (`test_TSICLAdapter.py:455`).

### Tier 4: docs-only and careful optimizations (defer, schedule separately)
- **3.4 TimesFM**: docstring note on `max_horizon` / `context_length` advising backtesting users to set `max_horizon` up front to avoid recompilation. No code change.
- **3.5 device-handling**: document the 4-strategy difference in `FoundationModel`'s docstring, defer signature unification (adding `device` to TimesFM) to a version bump.
- **3.3 full `.copy()` before trimming to `context_length`**: real perf win for direct `fit` / `predict` on long histories, but needs care (trim before preprocess must preserve `context_range_`, rework exog alignment order, long-format MultiIndex path stays as-is). Needs new large-series plus exog tests. Schedule as its own PR.
- Low-priority defensive / cosmetic (only if already touching the code): scalar `interval` float artifact (`:861`, cosmetic, codebase-wide pattern), degenerate / duplicate quantile-column defensive check (`:878`).

### Recommended sequencing
1. **PR A (Tier 1)**: DONE, all six batched with regression tests.
2. **PR B (Tier 2)**: consistency plus TSICL device cache plus error-rewrite robustness.
3. **PR C (Tier 3)**: split into three so the behavior change stays isolated. **C1** mechanical helpers (`_validate_positive_int`, detach helper, covariate helpers, ForecasterFoundation `_check_is_fitted_or_context` + `_predict_impl`, FoundationModel `_reset_fit_state`), all zero-behavior-change. **C2** `_FoundationAdapterBase` skeleton plus migration (compare-and-reset TabICL / TabPFN / Nori first, then the 5 key-presence adapters). **C3** the Moirai compare-and-reset + device-`.to()` behavior change, with its own added test and a changelog note. Do after A and B merge.
4. **PR D (Tier 4)**: docstrings now, the 3.3 copy-trim optimization as a separate follow-up PR.

Baseline before starting and after each PR: `pytest skforecast/foundation/tests/ -vv` (env `skforecast_24_py13`), plus `ruff`.


## Verification plan (once any of the above is implemented)

- Run the existing suite first to confirm no regressions: `pytest skforecast/foundation/tests/ -vv`.
- For each bug fix, add a targeted regression test in the matching `tests_foundation_models/test_*.py` or `tests_forecaster_foundation/test_*.py` file (e.g. a `levels=[]` case in `test_predict.py`, a 2-point irregular context in `test_TabICLAdapter.py`/`test_TabPFNAdapter.py`/`test_NoriAdapter.py`, a pre-fitted-forecaster case for `bayesian_search_foundation`).
- For the adapter `get_params`/`set_params` consolidation, re-run all 8 `test_*Adapter.py` files since they pin exact error-message prefixes and reset-key semantics — confirm messages still match `re.escape(...)` assertions.
- No UI/browser verification applicable (library code, not a frontend feature).
