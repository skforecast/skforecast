# Code Review: `skforecast/foundation/` — Bugs, Best Practices, Optimization


### Tier 1: quick correctness and robustness wins — DONE

### Tier 2: internal consistency and small perf (low risk, no API change) — DONE

### Tier 3: shared adapter base class and delegated refactors (biggest de-dup, enabler)
Highest effort. Split into three PRs (C1 mechanical helpers, C2 base class + migration, C3 Moirai behavior change) so the one intentional behavior change stays isolated from the pure refactor. Do NOT bundle all of this as a single "pure refactor" diff.

The bullets below are ordered to follow the recommended implementation order (PR C1 helpers first, then PR C2 base class, then PR C3 Moirai behavior change).

**PR C1 (mechanical helpers and extractions, zero behavior change):**
- **Mechanical helpers**: `_validate_positive_int(name, value)` (context_length, 16 sites confirmed, must reproduce the exact pinned message and also cover TimesFM `max_horizon`); tensor-to-numpy detach helper (3 sites `:344-348 / 2801-2805 / 3289-3293`, confirm the float-cast contract vs the already-extracted `NoriAdapter._to_numpy` `:3875-3897` which forces `dtype=float`); covariate conversion helpers (permissive vs strict, 4 sites, preserve TSICL's raw-array path and each owner-specific error message pinned at T0 `:273` / Nori `:339` / TSICL `:648`,`:669`); one shared 9-level quantile-grid constant plus a quantile-to-index helper.
- **ForecasterFoundation-side extractions (in `_forecaster_foundation.py`)**: `_check_is_fitted_or_context(context, method)` for the triplicated `NotFittedError` guard (`:750-754 / 856-860 / 960-964`, differ only in the method name in the message) and `_predict_impl` shared by `predict` / `predict_quantiles` (`:750-766` vs `:960-976`, differ only in `quantiles=None` vs `quantiles=list(quantiles)`). Note: these live in the forecaster, NOT `FoundationModel` (the model file has a single `ValueError` guard and no `predict_quantiles`).
- **FoundationModel-side extraction (in `_foundation_model.py`)**: `_reset_fit_state()` for the shared 10-line metadata core of the reset blocks at `:240-252 / 526-535 / 1066-1078`. The three blocks are NOT identical: `__init__` interleaves `creation_date` + version lines, `set_params` adds 3 adapter resets, so callers keep their extras around the shared call.

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
