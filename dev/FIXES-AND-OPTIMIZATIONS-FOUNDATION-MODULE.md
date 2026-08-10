# Code Review: `skforecast/foundation/` — Bugs, Best Practices, Optimization


### Tier 1: quick correctness and robustness wins — DONE
Implemented and merged (PR A): `bool` `steps` guard, `get_params(deep)` typing, `T0Adapter` dedup,
and `index_type_.__name__` repr cleanup. Each shipped with a regression test. The `None`-safe
`quantiles` / `interval` defaults were reverted afterward to keep `predict_interval` /
`predict_quantiles` aligned with the mutable-default-literal convention used by every other
forecaster (`ForecasterRecursive`, `ForecasterDirect`, etc.), which do not accept `None` for
these arguments.

### Tier 2: internal consistency and small perf (low risk, no API change)
7. **1.3** inconsistent `is_fitted` gating: gate all delegated fit-derived properties on `self.is_fitted` (`_forecaster_foundation.py:288-396`). Now defensive, since cloning fixes the crash.
8. **3.2** TSICL re-imports torch and re-resolves device on every `predict` (`_adapters.py:3255, 3270`): cache `self._resolved_device` at model load, invalidate with `self._model` on `set_params`.
9. **1.8** brittle `str.replace(AdapterClassName, "FoundationModel")` error rewriting (`_foundation_model.py:1053-1058`): make the message rewrite robust (anchor on the class-name token or restructure adapter errors).

### Tier 3: shared adapter base class and delegated refactors (biggest de-dup, enabler)
Highest effort but unlocks most of section 2. Build the base first, then migrate.
- **3.1 / section 2** `_FoundationAdapterBase` mixin: declarative `_param_names` / `_reset_keys` / `_validators` per subclass, replacing ~350 lines of hand-written `get_params` / `set_params` across all 8 adapters.
- Riding on the base class:
  - **1.6 Moirai `set_params`**: switch from key-presence reset to compare-and-reset (matching TabICL / TabPFN / Nori at `_adapters.py:1516 / 2148 / 3687`), and special-case `device`-only changes to `.to(...)` the cached object instead of nulling it. Intentional behavior change, requires updating `test_MoiraiAdapter.py`'s reset-semantics test.
  - **1.7 TabICL / TabPFN dedup** (~590 / 640 lines, near-verbatim): collapse onto the shared base.
  - `_validate_positive_int` helper (context_length, ~16 sites), tensor to numpy detach helper (3 sites), covariate conversion helpers (permissive vs strict, 4 sites), one shared 9-level quantile-grid constant plus a quantile to index helper.
- FoundationModel-side extractions (independent, can go in parallel): `_reset_fit_state()` (triplicated reset block `:1060+`), `_check_is_fitted_or_context(context, method)` (triplicated `NotFittedError` guard), `_predict_impl` shared by `predict` / `predict_quantiles`.

### Tier 4: docs-only and careful optimizations (defer, schedule separately)
- **3.4 TimesFM**: docstring note on `max_horizon` / `context_length` advising backtesting users to set `max_horizon` up front to avoid recompilation. No code change.
- **3.5 device-handling**: document the 4-strategy difference in `FoundationModel`'s docstring, defer signature unification (adding `device` to TimesFM) to a version bump.
- **3.3 full `.copy()` before trimming to `context_length`**: real perf win for direct `fit` / `predict` on long histories, but needs care (trim before preprocess must preserve `context_range_`, rework exog alignment order, long-format MultiIndex path stays as-is). Needs new large-series plus exog tests. Schedule as its own PR.
- Low-priority defensive / cosmetic (only if already touching the code): scalar `interval` float artifact (`:861`, cosmetic, codebase-wide pattern), degenerate / duplicate quantile-column defensive check (`:878`).

### Recommended sequencing
1. **PR A (Tier 1)**: DONE, all six batched with regression tests.
2. **PR B (Tier 2)**: consistency plus TSICL device cache plus error-rewrite robustness.
3. **PR C (Tier 3)**: shared base class plus Moirai / TabICL / TabPFN migration plus FoundationModel extractions. Do after A and B merge so the diff is pure refactor.
4. **PR D (Tier 4)**: docstrings now, the 3.3 copy-trim optimization as a separate follow-up PR.

Baseline before starting and after each PR: `pytest skforecast/foundation/tests/ -vv` (env `skforecast_24_py13`), plus `ruff`.

---

## 1. Bugs / edge-case failures (verified)


### 1.3 Inconsistent `is_fitted` gating on delegated properties → self-contradictory public state
- `context_`/`context_exog_` gate on `self.is_fitted` (`:236`, `:263`); `index_type_`, `index_freq_`, `context_range_`, `series_names_in_`, `is_multiple_series_`, `exog_in_`, `exog_names_in_`, `exog_names_in_per_series_`, `exog_type_in_`, `fit_date` (`:287-395`) do not.
- Combined with 1.2 (shared/unclonable estimator), `forecaster.is_fitted` can be `False` while `forecaster.series_names_in_`/`context_range_` still return stale data from a shared estimator — and `__repr__` (`:484-488`) does `self.context_range_.items()` unconditionally when `is_fitted`, which can then `AttributeError` on `None` if the invariant is violated (i.e., simply printing the object can crash in this state).
- **Fix:** gate all delegated fit-derived properties on `self.is_fitted` consistently, once cloning (1.2) is fixed this mostly stops being reachable, but the gating should still be consistent defensively.


### 1.6 `MoiraiAdapter.set_params` reloads the entire pretrained module on *any* reset-key presence, including no-op device changes
- `_adapters.py:1043-1052`: `valid = {'model_id', 'context_length', 'device'}`; any of these being **present** in the call (not necessarily changed) nulls `self._module`/`self._forecast_obj`, forcing a full HuggingFace re-download/reload on the next `predict`. Unlike `TabICLAdapter`/`TabPFNAdapter`/`NoriAdapter`, which compare old vs. new value first and only reset on real changes — this is a genuine cross-adapter behavioral inconsistency (5 of 8 adapters reset unconditionally on key-presence, 3 compare-and-reset).
- `device`-only changes don't need a reload at all — `.to(device)` on the already-loaded module/forecast object would suffice.
- **Fix:** unify all adapters on "reset only if the value actually changed" (see §3.1's shared base class), and for Moirai specifically, special-case `device`-only changes to call `.to(...)` on the cached `_forecast_obj` instead of nulling it (reusing the existing MPS-fallback warning logic in `_ensure_forecast_obj`, `:1244-1251`, factored into a small shared helper). **Note:** this will likely require updating `test_MoiraiAdapter.py`'s existing `set_params` reset-semantics test — flag it as an intentional behavior change, not a silent one.

### 1.7 `TabICLAdapter`/`TabPFNAdapter` line-for-line duplicated (~190 lines)
- The two adapters are near-verbatim copies. Their shared timestamp helper (`_get_future_timestamps`) has since been consolidated onto `expand_index`, but the remaining ~190 lines (`get_params`/`set_params`, covariate handling, `predict` body) are still duplicated and should be unified (see §2's shared-base-class item).

### 1.8 Minor / low-severity, worth a one-line fix each
- **Duplicate dict key** in `_ADAPTER_REGISTRY` (`_adapters.py:4084` and `:4087`, `"Synthefy/Nori"` twice) — harmless today (same value) but dead code a linter would flag (ruff `F601`); delete the second occurrence.
- **`FoundationModel.set_params`** (`_foundation_model.py:1044-1049`) rewrites only `ValueError` messages via `str.replace(AdapterClassName, "FoundationModel")` — brittle if the adapter ever raises a different exception type, or if a user-supplied value happens to contain the adapter's class name as a substring.
- **`get_params(deep: Any = None)`** (`_foundation_model.py:998`) — should be typed `deep: bool = True` per sklearn convention even though it's documented as unused; current `Any` typing is inconsistent with the rest of the file's careful typing.
- **`steps=True`** silently accepted as `steps=1` in `FoundationModel.predict` (`:880`, `isinstance(steps, (int, np.integer))` doesn't exclude `bool`) — trivial, low priority.
- **`predict_quantiles(quantiles=None)`** (`_forecaster_foundation.py:965`) does an unconditional `list(quantiles)`, so passing `None` (a very plausible mistake given every other param in the same signature accepts `None` as "use default") raises a low-quality `TypeError: 'NoneType' object is not iterable` instead of a clear validation error.
- **Scalar `interval` float artifact** (`_forecaster_foundation.py:861`, `0.5 - interval/2` not exactly `0.1` for `interval=0.8`) — confirmed as real float non-exactness, but verified to be **cosmetic only** here (never reaches the user; `predict_interval` renames columns before returning) and is an existing codebase-wide pattern (`ForecasterRecursive`, `ForecasterDirectMultiVariate`, etc. do the same thing) — not Foundation-specific, low priority, mention only if doing a global cleanup.
- **Duplicate/degenerate quantile columns**: if `lower_q`/`upper_q` ever coincide with `0.5` in float arithmetic, `predict_interval`'s column selection (`:878`) silently produces duplicate labels rather than raising — edge case, low likelihood, worth a defensive check if touching this code anyway.

---

## 2. Best-practice / style findings worth fixing alongside the above

- **Mutable default arguments**: `predict_interval(interval=[0.1, 0.9])` and `predict_quantiles(quantiles=[0.1, 0.5, 0.9])` (`_forecaster_foundation.py:779, 900`) use mutable list literals as defaults. Not currently mutated in place, but fragile — switch to `None` sentinel + in-body construction.
- **Triplicated `NotFittedError` guard + near-identical docstrings** across `predict`/`predict_interval`/`predict_quantiles` (`_forecaster_foundation.py:744-748/850-854/954-958` and their "Notes" sections) — extract to `self._check_is_fitted_or_context(context, method_name)`.
- **`predict`/`predict_quantiles` bodies ~90% identical** — worth a shared private `_predict_impl` helper.
- **Fragile type-name parsing**: `str(self.index_type_).split('.')[-1][:-2]` (used twice, `__repr__`/`_repr_html_`) should just be `self.index_type_.__name__`.
- **Triplicated `FoundationModel` attribute-reset block** (`__init__`/`fit`/`set_params`, `_foundation_model.py:241-249/524-533/1051-1060`) — extract to `self._reset_fit_state()`.
- **~350 lines of near-identical `get_params`/`set_params` boilerplate** duplicated across all 8 adapter classes in `_adapters.py` — candidate for a small `_FoundationAdapterBase` mixin (declarative `_param_names`/`_reset_keys`/`_validators` per subclass rather than hand-written logic per class).
- **`context_length` positive-int validation** repeated ~16 times verbatim — extract to one `_validate_positive_int(name, value)` helper in `_utils.py`.
- **Tensor→numpy "detach" pattern** copy-pasted 3x (Chronos/T0/TSICL `predict`) — only `NoriAdapter` extracted it; promote to a shared `_utils.py` helper.
- **Numeric-covariate conversion helpers** (`_to_covariate_array`/`_to_float_array`) reimplemented 4x with two genuinely distinct behaviors (permissive vs. strict) — consolidate into two shared functions parametrized by an `owner` name (for adapter-specific error text), not four copies.
- **Default 9-level quantile grid** (`[0.1..0.9]`) declared independently 4 times; two different implementations of "map quantile → grid index" (arithmetic in TimesFM vs. tolerance-search in Moirai) — one shared constant + one shared helper.
- **O(n²) manual dedup loop** in `T0Adapter._build_future_covariates` (`_adapters.py:2911-2915`, "append if not in list") — replace with `list(dict.fromkeys(...))`.
- **Non-idiomatic `next(generator)` inside a list comprehension** for a fixed-list lookup in `MoiraiAdapter.predict` (`:1151-1157`) — although verified unreachable as a bug (same list/tolerance guarantees a match), still worth simplifying to a plain loop or precomputed dict for clarity.
- **Inconsistent naming for "cached backend object"**: `_pipeline` (Chronos) vs. `_model` (5 adapters) vs. `_module`+`_forecast_obj` (Moirai, two objects) — harmless but makes generic reasoning about adapters harder.

---

## 3. Optimization opportunities

### 3.1 `TabICLAdapter._to_covariate_array`/TSICL's tolerance-search vs. TimesFM's arithmetic — see §2, mostly a clarity win, not a speed win.

### 3.2 `TSICLAdapter.predict` re-resolves the torch device (and re-imports `torch`) on **every** `predict()` call
- `_adapters.py:3241, 3256` — `import torch` and `_resolve_torch_device(self.device)` execute every call rather than being cached once at model-load time (as Moirai/T0 already effectively do via their guarded `_load_*` methods).
- **Fix:** cache `self._resolved_device` in `_load_model()`, invalidate it alongside `self._model` on relevant `set_params` changes, reuse it in `predict`. Low-risk, purely internal.

### 3.3 Full `.copy()` of the *entire* input series/exog before trimming to `context_length`
- `check_preprocess_series` (`skforecast/utils/utils.py:3411-3414`) copies all input series in full; only afterward does `FoundationModel._check_preprocess_context` (`_foundation_model.py:469-472`) trim to `.iloc[-context_length:]`. For `fit()` on long histories (the common global-forecasting case — many series, years of history, but `context_length` is typically 512-8192), cost scales with total input size, not the actual retained window.
- The hot backtesting loop (`skforecast/model_selection/_validation.py:2423-2458`) already works around this via `check_inputs=False` and pre-trimming — but a user's direct `.fit()`/`.predict(context=large_series)` call (default `check_inputs=True`) pays the full cost.
- **Fix (needs care, not a quick win):** trim before calling the shared preprocessing for the common cheap-to-slice input shapes (wide `DataFrame`, `dict[str, Series]`, plain `Series`), capturing the untrimmed first/last index beforehand so `context_range_` still reflects the true full input range (required by existing tests). `exog` alignment currently happens against the untrimmed series — trimming order needs to be reworked carefully (align first or trim both in lockstep) and the long-format MultiIndex input path is harder to pre-slice cheaply (leave it on the current path). Needs new tests with large series + exog before merging.

### 3.4 `TimesFMAdapter` recompiles whenever `steps > max_horizon`
- Confirmed against actual `timesfm` 2.5 source: the compiled decode graph always runs a fixed number of autoregressive steps derived from `max_horizon`, independent of the runtime `horizon`. skforecast's current "compile lazily, grow `max_horizon` on demand" strategy avoids wasted decode iterations for constant-horizon workloads (the common backtesting case) but pays repeated recompilation if horizon grows across calls within a session (e.g. expanding-window backtesting).
- Upstream's own documented pattern is "compile once with a generous `max_horizon`, then call `forecast(horizon=<actual, smaller value>)`" with no recompilation needed.
- **Fix:** this is a genuine trade-off, not a bug — the actionable fix is a **docstring note** on `TimesFMAdapter`'s `max_horizon`/`context_length` params advising users doing backtesting with a known maximum horizon to set `max_horizon` explicitly up front to avoid repeated recompilation. No code change required.

### 3.5 Device-handling inconsistency across adapters (4 different strategies in 8 classes)
- Chronos/T0 pass `device_map` straight to `from_pretrained`; Moirai/TSICL resolve via the shared `_resolve_torch_device` before `.to(device)`; TimesFM exposes no device control at all; TabICL/TabPFN/Nori bury it in opaque config dicts.
- **Fix:** this is a public-API-shape difference (adding a `device` param to `TimesFMAdapter` would be a signature change) — recommend documenting the difference clearly in `FoundationModel`'s docstring rather than forcing uniformity now; defer any signature unification to a version bump.

---

## Verification plan (once any of the above is implemented)

- Run the existing suite first to confirm no regressions: `pytest skforecast/foundation/tests/ -vv`.
- For each bug fix, add a targeted regression test in the matching `tests_foundation_models/test_*.py` or `tests_forecaster_foundation/test_*.py` file (e.g. a `levels=[]` case in `test_predict.py`, a 2-point irregular context in `test_TabICLAdapter.py`/`test_TabPFNAdapter.py`/`test_NoriAdapter.py`, a pre-fitted-forecaster case for `bayesian_search_foundation`).
- For the adapter `get_params`/`set_params` consolidation, re-run all 8 `test_*Adapter.py` files since they pin exact error-message prefixes and reset-key semantics — confirm messages still match `re.escape(...)` assertions.
- No UI/browser verification applicable (library code, not a frontend feature).
