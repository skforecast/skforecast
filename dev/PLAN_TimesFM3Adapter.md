# Upgrade `TimesFMAdapter` for TimesFM 3.0 (dual-support 2.5 + 3.0, with exog)

## Context

Google released TimesFM 3.0, and `timesfm==3.0.1` is installed in the
`skforecast_24_py13` conda env. Our current `TimesFMAdapter`
(`skforecast/foundation/_adapters.py`, class `TimesFMAdapter`, ~lines 483-881)
only speaks the **v2.5** API (`TimesFM_2p5_200M_torch.compile(ForecastConfig(...))`
+ `model.forecast(horizon, inputs) -> (point, quantiles)`).

**Verified against the installed package** (source of truth, since the HF/GitHub
docs are thin): `timesfm 3.0.1` ships **both** APIs side by side:

- `timesfm.TimesFM_2p5_200M_torch` — still present, still has `.compile()` and
  `.forecast(horizon, inputs) -> tuple`. **The current adapter keeps working for
  `google/timesfm-2.5-*` ids.**
- `timesfm.TimesFM3Forecaster` (`timesfm3.timesfm3_forecaster`) — the new v3.0 API,
  **completely different**:
  - Load: `TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device=None, **kwargs)` (kwargs -> `ModelConfig`, e.g. `per_core_batch_size`).
  - **No `compile()` step.** Context length and horizon are handled internally
    (context rounded up to input-patch boundary, capped at `global_context` ~15360;
    horizon rounded up to `output_patch_length`=64 then sliced).
  - Inference: `predict_batch(contexts: list[np.ndarray], horizon: int, past_only_covariates=None, past_future_covariates=None, ts_ids=None, return_quantiles=False, use_znorm=False, make_positive=False, use_symmetric_averaging=False, sort_quantiles=True, padding_mode="none") -> Iterator[ForecastOutput]`. `predict(...)` is the single-series wrapper.
  - `ForecastOutput` (frozen dataclass): `.forecast` = point forecast (**median**
    quantile), 1-D shape `(horizon,)`; `.quantiles` = `(horizon, 9)` only if
    `return_quantiles=True`.
  - **Quantile layout changed**: v3.0 returns **9** columns for levels
    `[0.1, 0.2, ..., 0.9]` with **median (q0.5) at index 4**. v2.5 returned **10**
    columns (index 0 = mean, indices 1-9 = q0.1-q0.9).
  - **Native covariate support**: `past_only_covariates` and
    `past_future_covariates` (numeric float32 arrays), plus multivariate targets.

**Decisions locked with the user:**
1. **Dual-support** — keep the v2.5 path, add a v3.0 path, dispatched on `model_id`.
2. **Add exog now** — flip `allow_exog` to `True` (for v3 ids) and wire covariates
   through `predict_batch`.

The `google/timesfm` registry prefix (`_adapters.py` `_ADAPTER_REGISTRY`, ~line 3952)
already routes both `-2.5-` and `-3.0-` ids to `TimesFMAdapter`, so **no registry
change is needed**.

## Adapter contract that must be preserved

(from `skforecast/foundation/_foundation_model.py`) — the v3 path must honor exactly
the same duck-typed contract as the other adapters:

- `__init__(model_id, *, ...)`, plus settable instance attrs `context_`,
  `context_exog_`, `is_fitted` (FoundationModel writes these directly in `set_params`).
- `fit(context, context_exog) -> self`; `predict(steps, context, context_exog, exog, quantiles) -> dict[str, np.ndarray]` where each value is shape **`(steps, n_quantiles)`** (or `(steps, 1)` when `quantiles is None`) — consumed at `_foundation_model.py` ~lines 1009-1012.
- `get_params()` keys stable (for `clone`/`repr`), must include `model_id`; must
  exclude the heavy loaded model.
- `allow_exog` read as an instance attr at `_foundation_model.py` ~line 954 and
  `_forecaster_foundation.py` ~line 661.
- Exog arrives pre-aligned: `context_exog[name]` = per-series **historical** exog
  aligned to context; `exog[name]` = per-series **future** exog with exactly `steps`
  rows (via `_prepare_future_exog`). This maps naturally to TimesFM's past/future
  covariate split.

## Refactoring plan

Most work is in `skforecast/foundation/_adapters.py` (`TimesFMAdapter`) + tests + docs.
The shared license-notice mechanism (step 3b) also adds a `LicenseWarning` to
`skforecast/exceptions/__init__.py` and a one-line loader call to the `MoiraiAdapter`,
`TabPFNAdapter`, and `TSICLAdapter` loaders.

### 1. Version dispatch

- Add a private helper to detect the backend version from `model_id`:
  `"timesfm-3" / "3.0" -> v3`, `"timesfm-2.5" / "2p5" -> v25`; raise a clear
  `ValueError` for unrecognized `google/timesfm-*` ids (guides the user to a
  supported id). Store `self._backend = "v3" | "v25"` in `__init__`.
- **`allow_exog` becomes an instance attribute** set in `__init__`:
  `self.allow_exog = (self._backend == "v3")`. Keep a class-level `allow_exog = False`
  as the default fallback. Instance access in FoundationModel/ForecasterFoundation is
  unaffected (identity checks like `allow_exog is False` still hold: `True`/`False`
  are singletons). TimesFM is the only adapter that toggles this per version; this is
  inherent to dual-support and will be documented. v2.5 keeps exog disabled; v3
  enables it.
- `SUPPORTED_QUANTILES` stays `[0.1, ..., 0.9]` (class attr): both versions support
  exactly this set; only the internal column index differs.

### 2. Constructor / params (union, documented per-version)

Keep the surface minimal and aligned with the other adapters (Chronos has 6 keys):

- Shared: `model_id`, `context_length`. Keep the test-injection handle `model=`.
- `context_length` gains a **`None` sentinel default** resolved per version inside
  `__init__`: `None -> 512` for v2.5 (unchanged default), `None -> 2048` (or higher)
  for v3, which supports up to ~15360. An explicit int is always honored.
  `get_params()` returns the *resolved* int so `clone`/`repr` stay stable. (Rationale:
  a hardcoded 512 default would badly under-use v3's long context.)
- v2.5-only: `max_horizon`, `forecast_config_kwargs` (used/validated only on the v2.5
  path).
- v3-only:
  - `device: str = "auto"` — resolved via the existing `_resolve_torch_device`
    (`_adapters.py` ~line 27; CUDA > MPS > CPU), matching `MoiraiAdapter`/`TabICLAdapter`.
    Passed to `TimesFM3Forecaster.from_pretrained(device=...)`.
  - `predict_kwargs: dict | None = None` — forwarded to `predict_batch`
    (`use_znorm`, `make_positive`, `use_symmetric_averaging`, `sort_quantiles`),
    mirroring Chronos's `predict_kwargs`. **Guard**: reject/strip keys the adapter
    manages itself (`contexts`, `horizon`, `return_quantiles`,
    `past_only_covariates`, `past_future_covariates`, `padding_mode`, `ts_ids`) so a
    user cannot break the call.
- `per_core_batch_size` is **not** exposed as a named param (keep the surface small;
  default 4 is fine): listed as a follow-up.
- `get_params()` keys: `model_id`, `context_length`, `max_horizon`,
  `forecast_config_kwargs`, `device`, `predict_kwargs`. `set_params` resets
  `self._model = None` on change of any **model-affecting** key
  (`model_id`, `context_length`, `max_horizon`, `forecast_config_kwargs`, `device`);
  `predict_kwargs` changes do **not** reset the loaded model. Reuse `_apply_set_params`,
  same pattern as today (`_adapters.py` ~lines 659-668).

### 3. Loading (`_load_model`) — branch on `self._backend`

- **v2.5**: unchanged — keep the `_TimesFMCompat` subclass workaround +
  `_ensure_compiled`.
- **v3**: `self._model = timesfm.TimesFM3Forecaster.from_pretrained(self.model_id, device=_resolve_torch_device(self.device))`.
  **No `compile` / `_ensure_compiled`** on the v3 path (the compat subclass and
  `ForecastConfig` are not needed: `TimesFM3Forecaster.from_pretrained` forwards
  only whitelisted kwargs, so the huggingface_hub `proxies/resume_download` leakage
  bug does not apply). Wrap `ImportError` with the existing install hint, same as the
  other adapters' loaders. Additionally, if `timesfm` imports but lacks
  `TimesFM3Forecaster` (older `timesfm<3.0`), raise a clear error telling the user to
  upgrade.

### 3b. Non-commercial license notice (shared mechanism, all adapters)

**Motivation.** Several foundation-model *weights* carry non-commercial licenses (the
skforecast code and the backend Python packages stay permissive; the restriction is on
the checkpoint a user pulls via `from_pretrained`). A user can silently download
restricted weights without realizing they cannot deploy them. Today no adapter warns
about this; the only license handling is T0's *access-gating* help
([_adapters.py:2758-2768](skforecast/foundation/_adapters.py#L2758-L2768)), which is a
different concern (how to download, not whether you may use it commercially).

Build this as a **shared, registry-driven mechanism** rather than TimesFM-only code, so
every current and future adapter benefits from one consistent code path.

**License status of the current adapters** (verified against HF model cards, Sep 2026):

| Adapter | `model_id` prefix | Weights license | Commercial |
|---------|-------------------|-----------------|:---------:|
| Chronos | `autogluon/chronos`, `amazon/chronos` | Apache 2.0 | yes |
| TimesFM 2.5 | `google/timesfm-2.5` | Apache 2.0 | yes |
| **TimesFM 3.0** | `google/timesfm-3.0` | TimesFM Non-Commercial License v1.0 | **no** |
| **Moirai** | `Salesforce/moirai` | CC-BY-NC-4.0 | **no** |
| TabICL | `soda-inria/tabicl` | BSD-3-Clause | yes |
| **TabPFN-TS** | `priorlabs/tabpfn` | TabPFN License v1.0 (non-commercial w/o enterprise) | **no** |
| T0 | `theforecastingcompany/t0` | Apache 2.0 (gated repo) | yes |
| Nori | `Synthefy/Nori` | Apache 2.0 | yes |
| **TS-ICL** | `taharnbl/TS-ICL` | tsicl-v1-license-v1.0 (non-commercial) | **no** |

So four targets need the notice: **TimesFM 3.0, Moirai, TabPFN-TS, TS-ICL**.

**Design.**

1. **`LicenseWarning` category** — add to `skforecast/exceptions/__init__.py` and
   register it in `warn_skforecast_categories` / `set_warnings_style` alongside the
   existing categories, so users can filter it by name and the styled-warning system
   picks it up.
2. **Registry** in `_adapters.py` (near `_ADAPTER_REGISTRY`), keyed by `model_id`
   prefix, longest-prefix match wins (so `google/timesfm-3.0` and `google/timesfm-2.5`
   resolve independently). Only non-commercial entries are listed; anything not matched
   is treated as unrestricted (no warning):
   ```python
   _NON_COMMERCIAL_LICENSES = {
       "google/timesfm-3.0": ("TimesFM Non-Commercial License v1.0",
                               "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE"),
       "Salesforce/moirai":  ("CC-BY-NC-4.0",
                               "https://huggingface.co/Salesforce/moirai-2.0-R-small"),
       "priorlabs/tabpfn":   ("TabPFN License v1.0 (non-commercial without an enterprise license)",
                               "https://github.com/PriorLabs/TabPFN/blob/main/LICENSE"),
       "taharnbl/TS-ICL":    ("tsicl-v1-license-v1.0 (non-commercial)",
                               "https://huggingface.co/taharnbl/TS-ICL"),
   }
   ```
   (Confirm each URL/name against the model card at implementation time.)
3. **Helper** `_warn_if_non_commercial(model_id)` in `_adapters.py` (or `_utils.py`):
   longest-prefix lookup; if matched, `warnings.warn(msg, category=LicenseWarning, stacklevel=...)`.
   Message (no en/em dashes, per repo style): the weights for `<model_id>` are released
   under `<license_name>`, which restricts use to non-commercial / non-production
   purposes; review the license before deploying; see `<url>`.
4. **Wiring** — call `_warn_if_non_commercial(self.model_id)` from each adapter's
   `_load_model` at the point weights are loaded (a no-op-when-already-loaded loader
   makes it fire once per load). Wire all four affected adapters:
   - `TimesFMAdapter._load_model` (v3 branch only; v2.5 `model_id` is not in the
     registry so it stays silent automatically — no special-casing needed).
   - `MoiraiAdapter`, `TabPFNAdapter`, `TSICLAdapter` loaders.
   The registry-driven call is a single line per adapter, and Chronos/TabICL/T0/Nori
   add nothing (their prefixes are absent from the registry).
5. **Suppressible** — because it routes through the skforecast warnings system, the
   existing `suppress_warnings` argument / module warning filter silences it in
   backtesting/refit loops.

**Tests** (`tests_foundation_models/`):
- Unit-test `_warn_if_non_commercial` directly: matched prefixes warn with
  `LicenseWarning`; unmatched (e.g. `autogluon/chronos-2-small`, `google/timesfm-2.5-*`)
  do not.
- Per affected adapter (via the injected-model path so no download): assert a
  `LicenseWarning` is raised on load, and that it is silenced under `suppress_warnings`.
- TimesFM specifically: warning on a v3 `model_id`, no warning on a v2.5 `model_id`.

**Docs** — surface the non-commercial restriction wherever these ids appear
(user-guide notebook, `AGENTS.md` adapter table, API reference, skill docs). Note in
the adapter-comparison docs which models are non-commercial so users can choose before
committing.

**Scope note.** This plan builds the shared mechanism and wires **all four**
non-commercial adapters (TimesFM 3.0, Moirai, TabPFN-TS, TS-ICL) in this branch, per the
user's decision. Touching the other three adapters is limited to the one-line loader
call plus their docstrings/tests; no behavioral change beyond the new warning.

### 4. `predict` — dispatch to `_predict_v25` (existing body, unchanged) vs `_predict_v3`

The `steps > max_horizon` `ValueError` currently in `predict` must be **gated to the
v2.5 path only**: v3 has no compile ceiling, so applying the v2.5-oriented default
(512) would wrongly reject legitimate long-horizon v3 calls.

`_predict_v3`:
- Validate requested `quantiles` against `SUPPORTED_QUANTILES` first (keep the existing
  loop; v3 cannot interpolate off-grid levels either).
- `self._load_model()`; `names = list(context.keys())`;
  `contexts = [context[n].to_numpy() for n in names]`.
- Build covariates per series (see step 5) into `past_only` / `past_future` lists;
  set `has_cov = any covariate is not None`.
- `outs = list(self._model.predict_batch(contexts=contexts, horizon=steps, past_only_covariates=po, past_future_covariates=pf, return_quantiles=(quantiles is not None), padding_mode=("edge" if has_cov else "none"), **self.predict_kwargs))`.
  **`padding_mode="edge"` is required whenever covariates are present** (see the
  finding below): `predict_batch` rounds the internal horizon up to a multiple of 64
  and expects `past_future_covariates` to span `context + global_horizon`; edge mode
  pads the future side from `steps` to that length. Without it, an arbitrary `steps`
  misaligns/errors.
- Build the per-series quantile index from the model's **actual** quantile vector
  (`self._model.config.quantiles`, synced from the checkpoint), not a hardcoded
  formula: `q_idx = [nearest_index(self._model.config.quantiles, q) for q in quantiles]`
  with a tolerance check. This stays correct if a checkpoint ships a different quantile
  grid, and mirrors the tolerance-matching the v2.5 path already does.
- Assemble output dict (`predict_batch` already slices to `[:steps]`):
  - point (`quantiles is None`): `predictions[name] = np.asarray(out.forecast).reshape(-1, 1)` -> `(steps, 1)`. Note the v3 point forecast is the **median** quantile (v2.5's index 0 was the mean); both are valid point forecasts, documented in the docstring.
  - quantiles: `predictions[name] = np.asarray(out.quantiles)[:, q_idx]` -> `(steps, n_q)`.

### 5. Covariate mapping (v3 only)

Given skforecast's exog semantics (known-future covariates, with history in
`context_exog` and future in `exog`), for each series `name`:
- Columns present in `exog[name]` -> **known-future** -> one
  `past_future_covariates` row per column of length `context_len + steps` =
  `concat(context_exog[name][col].to_numpy(), exog[name][col].to_numpy())`.
- Columns present only in `context_exog[name]` (not in `exog`) -> **past-only** ->
  `past_only_covariates` row of length `context_len`.
- Stack rows into `(n_cov, L)` float32 arrays; pass `None` when a series has no
  covariates. Covariate/context alignment is guaranteed: `_check_preprocess_context`
  trims `context` and `context_exog` with the identical `iloc[-context_length:]`
  (`_foundation_model.py` ~lines 484-496).
- **Numeric-only**: `predict_batch` casts covariates via `np.array(..., dtype=np.float32)`,
  so string/categorical columns would raise deep inside the backend. Do **not** reuse
  `ChronosAdapter._to_covariate_array` (it deliberately passes object dtype through for
  Chronos's native categorical support). Instead add a small numeric coercion that
  raises a clear `ValueError` naming the offending column, directing the user to encode
  categoricals upstream via `transformer_exog`. Handle pandas nullable dtypes the same
  way Chronos does (cast via `.astype(np.float32)`).

### 6. Docstrings & reference docs

- `TimesFMAdapter` class + method docstrings: document dual-version behavior, the v3
  covariate support, the 9-quantile layout, `device`/`predict_kwargs`, and update the
  HF link to `google/timesfm-3.0-pytorch`. Follow
  `.github/instructions/docstrings.instructions.md` (NumPy style, no en/em dashes).
- `FoundationModel` docstring (`_foundation_model.py` around lines 48-51, 89-91, 331):
  add the v3.0 id, note exog now supported for v3.
- `AGENTS.md` (embedded reference): update the adapter table row (~line 515:
  `Exog: Yes` for v3, version label, default `context_length`) and prose mentions
  (~lines 86, 107, 121, 484). Note: `AGENTS.md` is the source;
  `.github/copilot-instructions.md` is auto-generated — do not edit it.
- Skill docs referencing TimesFM: `skills/foundation-forecasting/SKILL.md` +
  `references/adapter-parameters.md`, `skills/complete-api-reference/*`,
  `skills/choosing-a-forecaster/SKILL.md`,
  `skills/hyperparameter-optimization/references/search-parameters.md`,
  `skills/prediction-intervals/SKILL.md`.
- `docs/llms-base.txt` / `docs/llms-full.txt` if they carry the adapter table.

### 7. Tests

File: `skforecast/foundation/tests/tests_foundation_models/test_TimesFMAdapter.py`
(+ fixtures in `.../fixtures_adapters.py`). Tests inject a fake via `model=`, never
importing the real backend — keep that approach. Read
`.github/instructions/testing.instructions.md` before editing anything under
`**/tests/**`.

- Keep all existing v2.5 tests (they still pass; the v2.5 path is unchanged).
  Ensure `make_adapter` uses a `-2.5-` `model_id` for those.
- Add `FakeTimesFM3Forecaster` fixture: implements `from_pretrained` + `predict_batch`
  yielding objects with `.forecast` `(steps,)` and `.quantiles` `(steps, 9)`, plus a
  `.config.quantiles` list so the index-mapping code has a real grid to match against.
- New v3 tests: version dispatch (`_backend`), `allow_exog is True` for a v3 id /
  `False` for a v2.5 id, point & quantile predict (single + multi-series), quantile
  index mapping against the model's quantile vector, covariate wiring
  (`past_future` + `past_only` arrays, correct lengths), `padding_mode="edge"` used
  when covariates are present (assert a non-multiple-of-64 `steps` with exog returns
  `steps`-length output), non-numeric covariate `ValueError`, off-grid quantile
  `ValueError`, `get_params`/`set_params` for the new keys.
- Check `test_resolve_adapter.py` — add a `google/timesfm-3.0-pytorch -> TimesFMAdapter`
  assertion.

## Verification

Before running any Python, per `AGENTS.md`, confirm the env (`skforecast_24_py13`).
Call the env python directly with `PYTHONIOENCODING=utf-8`; do not use `conda run`
(it crashes on rich/unicode on Windows, per prior findings). Env python:
`C:/Users/Joaquin/miniconda3/envs/skforecast_24_py13/python.exe`.

1. Unit tests (no network, fakes only):
   `python.exe -m pytest skforecast/foundation/tests/tests_foundation_models/test_TimesFMAdapter.py skforecast/foundation/tests/tests_foundation_models/test_resolve_adapter.py -vv`
2. Full foundation suite for regressions:
   `python.exe -m pytest skforecast/foundation/tests -q`
3. **Live smoke test** (downloads weights from HF on first run; confirm the env has
   network + the user is OK downloading ~0.3B params) — a short script:
   - v3 point + quantile: `FoundationModel(model_id="google/timesfm-3.0-pytorch")`,
     `ForecasterFoundation`, `fit` on a demo series (`fetch_dataset("h2o")`),
     `predict(steps=12)` and `predict_interval(steps=12, interval=[0.1, 0.9])`;
     assert output shapes/columns.
   - v3 with exog: `fetch_dataset("h2o_exog")`, fit with `exog`, predict with future
     `exog`; assert it runs and differs from the no-exog run.
   - v2.5 regression: repeat point/quantile with
     `google/timesfm-2.5-200m-pytorch`; assert unchanged behavior.
4. `ruff check` on the edited files (PEP8, line length 88, double quotes).

## Risks & review notes (validation pass)

- **Covariate future-padding (highest risk)**: `padding_mode="edge"` is mandatory when
  covariates are present, otherwise arbitrary `steps` misaligns
  `past_future_covariates`. Covered by a dedicated test.
- **`steps > max_horizon` gate**: must be v2.5-only (see step 4); a leftover global
  check would break long-horizon v3 forecasts.
- **Quantile indexing**: derive from `self._model.config.quantiles` rather than a
  `*10-1` formula, so a checkpoint with a different grid does not silently return the
  wrong columns.
- **`context_length` default change** (int -> `None` sentinel): a minor public-API
  tweak; `get_params` still returns a concrete int, and existing v2.5 behavior
  (default 512) is preserved, so `clone`/serialization stay intact. Confirm no test
  asserts the literal default via `inspect.signature`.
- **Point-forecast semantics differ across versions** (v2.5 mean at col 0 vs v3
  median): documented; not a bug, but note in the docstring so users understand the
  point column.
- **Non-numeric exog** raises a clear adapter-level `ValueError` instead of a cryptic
  backend cast error.
- **No dead code introduced**: `_TimesFMCompat` and `_ensure_compiled` remain used by
  the retained v2.5 path; `SUPPORTED_QUANTILES` is used by both. The v3 path adds only
  necessary members.
- **`allow_exog` per-version toggle** is the one deviation from the other adapters'
  static class attribute; justified by dual-support and documented.
- **Backend availability**: v3 requires `timesfm>=3.0`; the retained v2.5 path also
  works under `timesfm 3.0.1` (the `TimesFM_2p5_200M_torch` class is still shipped).
  If a user has an older `timesfm<3.0`, a v3 `model_id` must surface a clear
  upgrade error (check for `TimesFM3Forecaster`).

## Out of scope (flag as follow-ups)

- Native **multivariate** targets (v3 supports multiple target variates per series);
  skforecast's wide/long dict contract stays univariate-per-series.
- Exposing `per_core_batch_size` and every v3 predict-time knob beyond the
  `predict_kwargs` passthrough.

## Documentation to update (do not skip)

The code change is not complete until the docstrings and all user-facing
documentation reflect the upgraded adapter (dual-version support, TimesFM 3.0 id,
native exog/covariates, the 9-quantile layout, new `device`/`predict_kwargs` params).
This expands on step 6 and must be treated as part of the deliverable:

- **Docstrings** — `TimesFMAdapter` class and every method docstring in
  `skforecast/foundation/_adapters.py`, plus the `FoundationModel` docstring in
  `skforecast/foundation/_foundation_model.py` (version label, ids, exog support).
  Follow `.github/instructions/docstrings.instructions.md` (NumPy style, no en/em
  dashes).
- **User guide notebook** — `docs/user_guides/foundation-forecasting-models.ipynb`:
  update the TimesFM section(s) with the v3.0 `model_id`, note dual-version support,
  add an exog/covariate usage example, and update the quantile description. Re-run the
  affected cells so outputs match.
- **API reference** — `docs/api/FoundationModel.md` (parameter table / adapter list).
- **LLM references** — `docs/llms.txt` and `docs/llms-full.txt` (adapter table rows:
  TimesFM `Exog: Yes` for v3, updated default `context_length`, version label). These
  mirror `AGENTS.md`; keep them consistent.
- **Quick start** — `docs/quick-start/ai-assisted-forecasting.md` if it lists the
  TimesFM adapter/capabilities.
- **Release notes** — add an entry to `docs/releases/releases.md` describing TimesFM
  3.0 support and the new exog capability.
- **Embedded reference + skills** — `AGENTS.md` (source of truth; the auto-generated
  `.github/copilot-instructions.md` is regenerated from it, do not hand-edit) and the
  skill docs listed in step 6.

After editing, verify docs consistency: search the repo for `timesfm-2.5` /
`TimesFM 2.5` / the old `context_length` default and confirm every occurrence is
either intentionally version-specific or updated.
