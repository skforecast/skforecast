# Plan: Add a `TinyTimeMixerAdapter` (IBM TTM) foundation-model adapter

## Context

skforecast's `foundation` subsystem wraps zero-shot time-series foundation models
behind a uniform adapter contract (`ChronosAdapter`, `TimesFMAdapter`,
`MoiraiAdapter`, `TabICLAdapter`, `TabPFNAdapter`, `T0Adapter`, `NoriAdapter`,
`TSICLAdapter`). Goal: add IBM Granite's **TinyTimeMixer (TTM)** so users can run
`ForecasterFoundation` / `backtesting_foundation` / `bayesian_search_foundation`
against `ibm-granite/granite-timeseries-ttm-r3` (the default; and r2, r1) with zero
training. The stock **r3** checkpoint returns native quantile (probabilistic)
forecasts zero-shot; exogenous variables (skforecast `exog`) still require a
fine-tuned channel-mixing checkpoint on every revision.

Phasing (each phase ships independently): **Phase 1** = point forecasts (all
revisions) + native zero-shot quantiles (r3); **Phase 2** = exog for fine-tuned
checkpoints loaded via `finetuned_checkpoint` (+ `exog_channels`); **Phase 3, optional**
= a standalone `finetune_tinytimemixer` helper that produces a fine-tuned checkpoint to
load (File 6). Fine-tuning is NEVER done inside the adapter's `fit` (see File 6 for why).

All adapters live in one module: `skforecast/foundation/_adapters.py`. The closest
template to copy is **`TimesFMAdapter`** (`_adapters.py:483-880`): fixed horizon,
lazy load. Device handling mirrors `T0Adapter` (`_adapters.py:2438+`):
`_resolve_torch_device`, `.to(device)`, optional `torch_dtype`. The exog + quantile
output contract mirrors **`TabPFNAdapter`** (`_adapters.py:2109`): `context_exog`
(past covariates), `exog` (future covariates), and `(steps, n_quantiles)` output.

### Verified TTM behavior (source: `granite-tsfm` main + released r3/r2 `config.json`)

- **Native zero-shot quantiles in stock r3.** The stock
  `ibm-granite/granite-timeseries-ttm-r3` checkpoint (released 2026-03-31,
  Apache-2.0) ships `multi_quantile_head=True`,
  `quantile_levels=[0.1, 0.2, ..., 0.9]`, `forecast_loss_type="joint"`,
  `distribution_output="student_t"`. So r3 returns probabilistic forecasts with NO
  fine-tuning, and the trained levels are readable from the loaded config via
  `getattr(config, "quantile_levels", None)` (sorted). The model still emits a
  point head too (joint loss). `ForecasterFoundation` has NO bootstrapping/conformal
  fallback (`_forecaster_foundation.py:792`: intervals come "directly from the
  model's native quantile output"), so this native head is exactly what
  `predict_interval` / `predict_quantiles` consume.
- **Point-forecast only in the stock r1/r2 checkpoints.** `MultiQuantileHead` /
  `MultiPinballLoss` exist but are gated behind `multi_quantile_head=False` there and
  only active after *fine-tuning*. Released r2 ships `loss="mse"`, no quantile keys.
  Conformal intervals exist only as a separate post-hoc `granite-tsfm` wrapper
  (`PostHocProbabilisticProcessor`), which skforecast does not use. => on r1/r2 the
  adapter path is point-only and must raise on any `quantiles` request unless the
  loaded config exposes a quantile head or the user declares one.
- **No exogenous variables in ANY stock checkpoint (incl. r3).** FCM / channel-mixing
  (`enable_forecast_channel_mixing=False`, `exogenous_channel_indices=null`,
  `num_input_channels=1` in the released r3, r2, r1 checkpoints) requires
  fine-tuning. The r3 "exogenous integration" headline is a fine-tune capability,
  not stock. => the default adapter path has `allow_exog=False` on all revisions.
- **Fine-tuning adds covariates (all revisions) and a quantile head (r1/r2).** TTM
  fine-tunes cheaply to add channel mixing (exog) and, on r1/r2, a quantile head.
  This adapter supports fine-tuned checkpoints through opt-in parameters
  (`exog_channels`, `quantile_levels`); on stock r3 the quantile head is already
  present and its levels are auto-derived from the config (see below).
- **Fixed `context_length` AND fixed `prediction_length` per checkpoint.** Input must
  be exactly `context_length` (left-pad if shorter); horizon is the checkpoint's fixed
  `prediction_length` (can only be shortened). The r3 root checkpoint pairs 512/30;
  `get_model` resolves other pairs (e.g. 512/96) from the r3 grid.
- **Internal instance normalization** (`scaling="std"`) => feed raw values, read raw back.
- Standard torch `PreTrainedModel`: `.to(device)`, `.eval()`, `forward(past_values=...)`.

### Design decisions (confirmed with user)

- **Default `model_id = "ibm-granite/granite-timeseries-ttm-r3"`** (user decision).
  r3 is Apache-2.0, uses the same `get_model` loader, and returns quantiles
  zero-shot. r2/r1 remain fully supported (point-only stock; exog/quantiles after
  fine-tuning). Note: the separate `ibm-research/ttm-r3` research variant is
  CC-BY-NC-SA-4.0, but the `ibm-granite/...` checkpoint this adapter targets is
  Apache-2.0 (verified in the HF model-card front matter).
- **Quantiles: zero-shot on r3 (auto-derived), else declared/fine-tuned.** On stock
  r3 the quantile head is present; the adapter auto-populates `quantile_levels` from
  the loaded `config.quantile_levels` (sorted) when the user did not pass them, so
  `predict_interval` / `predict_quantiles` work with no extra params. On stock r1/r2
  (no head) requesting quantiles raises unless the user loads a fine-tuned head and
  declares `quantile_levels`. Exog is `allow_exog=False` by default on every
  revision and is enabled only by declaring `exog_channels` (fine-tuned checkpoints).
  Guard messages describe *this adapter's configured path*, NOT a fixed claim about
  the checkpoint, so they stay accurate across revisions.
- **`quantile_levels` = auto-derive, allow override** (user decision). If the user
  passes `quantile_levels`, it is honored and cross-checked against the config head
  (mismatch raises); requested quantiles must be a subset of it. If the user does
  NOT pass it but the config exposes a quantile head, the adapter adopts
  `sorted(config.quantile_levels)`. This keeps r3 zero-shot while preserving control
  for fine-tuned heads with custom levels.
- **Exog capability source = explicit params, config-validated** (user decision).
  `exog_channels` is cross-checked against the loaded checkpoint `config` at load
  time; a mismatch raises. Rationale: `allow_exog` is read by `FoundationModel`
  *before* the lazy load (`_foundation_model.py:333, 954`), so it must be known at
  construction; and TTM `config` stores exog as channel *indices*, not the column
  names skforecast needs, so the name->channel mapping can only come from the user.
- **NO change to `FoundationModel`'s public API** (user decision). Capability params
  (`finetuned_checkpoint`, `exog_channels`, `quantile_levels`, ...) already flow
  through `FoundationModel.__init__`'s `**kwargs` to the adapter ctor
  (`_foundation_model.py:252`). Custom fine-tuned checkpoints are loaded from a
  **clone-safe path parameter** `finetuned_checkpoint` (a local dir OR a hub repo id).
  Because `_resolve_adapter` prefix-matches `model_id` (`_adapters.py:3978`), the user
  still passes a `model_id` that starts with the registered prefix
  `ibm-granite/granite-timeseries-ttm` as the routing key + label; when
  `finetuned_checkpoint` is set, `_load_model` loads the weights with
  `TinyTimeMixerForPrediction.from_pretrained(finetuned_checkpoint)` instead of
  `get_model`, so the label is never used to download anything.
  - **Why a path, not `model=` injection.** `ForecasterFoundation.__init__` runs
    `clone(estimator)` (`_forecaster_foundation.py:139`), and
    `FoundationModel.get_params` (which returns `adapter.get_params()`) INTENTIONALLY
    excludes the in-memory `model` (`_foundation_model.py:1041`), so an injected
    `model=` is dropped the moment the estimator is wrapped in a forecaster (and again
    on every backtest/search deep-copy) -- the adapter would then lazy-load the STOCK
    label checkpoint and raise on the exog mismatch. A string `finetuned_checkpoint`
    is part of `get_params`, so it survives `clone`, and the lazy `from_pretrained`
    reload matches how every other adapter loads its weights. `model=` remains ONLY a
    test-injection hook (not clone-safe; never a user path).
- **Quantile levels = subset-required** (user decision). Requested quantiles must be a
  subset of the effective `quantile_levels` (declared or auto-derived); otherwise
  raise, listing the available levels (mirrors `TimesFMAdapter`'s fixed-set behavior).
  Interpolation is deferred.
- **Undeclared config capability: quantiles auto-adopt, exog warns.** If the config
  shows a quantile head but the user did NOT pass `quantile_levels`, the adapter
  ADOPTS the config levels (r3 zero-shot; no warning). If the config shows channel
  mixing but the user did NOT pass `exog_channels`, emit an `IgnoredArgumentWarning`
  at load (does NOT raise, does NOT auto-enable): exog cannot be auto-enabled because
  the name->channel mapping and the `allow_exog` pre-load timing both require the user
  param. This still catches the silent-wrong-result trap for a fine-tuned exog
  checkpoint used without `exog_channels`.
- **Fixed horizon**: constructor param `prediction_length` selects the checkpoint
  horizon; `predict` requires `steps <= prediction_length`, slices the fixed output down
  to `steps`, and raises if `steps > prediction_length` (mirrors
  `TimesFMAdapter.max_horizon`).
- **Defaults**: `context_length=512`, `prediction_length=96`, `exog_channels=None`,
  `quantile_levels=None`, `finetuned_checkpoint=None`; documented default
  `model_id="ibm-granite/granite-timeseries-ttm-r3"` (`get_model` resolves the
  512/96 r3 grid entry).

### Quantile and exog support (stock r3 quantiles + fine-tuned exog)

Enabling exog/quantiles is real work beyond a capability flag; the adapter implements
it as follows.

- **Capability declaration** (constructor):
  - `exog_channels: list[str] | None` -- ordered exog column names. The order MUST match
    the exogenous channel order the checkpoint was fine-tuned with. Presence sets the
    instance attribute `allow_exog = True` (so `FoundationModel` forwards `exog` /
    `context_exog` instead of dropping them).
  - `quantile_levels: list[float] | None` -- OPTIONAL override of the head's levels
    (each in `(0, 1)`). Leave `None` for stock r3: the adapter reads the levels from
    the config. Pass it to pin a fine-tuned head's custom levels.
- **`allow_exog` is an instance attribute** (default class attr `False`, overridden in
  `__init__` from `exog_channels is not None`). This keeps the capability known at
  construction, satisfying the pre-load read in `FoundationModel`.
- **Config reconciliation at load** (`_reconcile_capabilities`, runs once after the
  model is available, for stock, `finetuned_checkpoint`, and test-injected models):
  - Quantiles: if `quantile_levels` is `None` and the config exposes a quantile head,
    set `self.quantile_levels = sorted(config.quantile_levels)` (r3 auto-derive). If
    `quantile_levels` was declared but the config shows no head => `ValueError`.
  - Exog: `exog_channels` set but the config shows no channel mixing (or a mismatched
    channel count) => `ValueError`; channel mixing present but `exog_channels`
    undeclared => `IgnoredArgumentWarning`.
- **`predict` ordering**: `_load_model()` + `_reconcile_capabilities()` run BEFORE the
  `quantiles` subset check, so the auto-derived `quantile_levels` is populated before
  validation. (`allow_exog` is unaffected -- still fixed at construction.)
- **Input assembly** (`predict`): build multi-channel tensors of width
  `n_channels = 1 + len(exog_channels)`: channel 0 = target, channels `1..k` = exog in
  `exog_channels` order, filled from `context_exog` (past) and `exog` (future-known),
  with `past_observed_mask` / future mask marking observed vs padded positions.
- **Output extraction** (`predict`): point head reads `.prediction_outputs`
  `(n, pred_len, C)`; the quantile head output has layout
  `(n, C, n_quantiles, pred_len)` (confirmed from `MultiQuantileHead`), with the
  quantile axis following `sorted(config.quantile_levels)`. Select the target channel,
  then the requested quantile columns by their index in the sorted levels. The exact
  output ATTRIBUTE NAME (point is `.prediction_outputs`) and the future-covariate
  forward kwargs remain backend details flagged in "Open implementation checks".

---

## TTM backend API reference (embedded so no re-research is needed)

- **Install**: `pip install granite-tsfm`. Not in mainline `transformers`; goes through
  `tsfm_public`. Requires (current main) python 3.11-3.13, `torch>=2.10`,
  `transformers[torch]>=4.57`; older releases allowed py3.10 / looser torch. Depend loosely.
- **Recommended loader for stock checkpoints** (resolves the checkpoint branch):
  ```python
  from tsfm_public.toolkit.get_model import get_model
  model = get_model(
      "ibm-granite/granite-timeseries-ttm-r3",
      context_length=512, prediction_length=96,
      # optional: freq_prefix_tuning, freq, prefer_l1_loss, prefer_longer_context,
      #           force_return ("zeropad"/"rolling"/...), model_revision
  )
  ```
  `get_model` reads `tsfm_public/resources/model_paths_config/ttm.yaml`, picks the entry
  whose context/prediction fits, and calls `from_pretrained(..., revision=<branch>,
  prediction_filter_length=...)` internally.
- **Loading a fine-tuned / custom checkpoint** (the adapter calls this itself inside
  `_load_model` when `finetuned_checkpoint` is set; the value is a local dir or a hub
  repo id):
  ```python
  from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
  ttm = TinyTimeMixerForPrediction.from_pretrained("./my-ttm-finetuned")  # or hub repo
  ```
- **Forward pass** (point path used by the adapter):
  ```python
  out = model(past_values=x, past_observed_mask=mask)   # x: (batch, context_length, n_channels)
  preds = out.prediction_outputs                        # (batch, prediction_length, n_channels)
  ```
  `TinyTimeMixerForPrediction` is a `PreTrainedModel`. `past_observed_mask` (same shape as
  `past_values`, 1=observed, 0=missing/pad) lets the internal std-scaler ignore padding.
  Future-known covariates are an FCM/fine-tuning feature (see "Open implementation
  checks"). The quantile output is native on stock r3: `MultiQuantileHead` emits a
  tensor of shape `(n, C, n_quantiles, pred_len)` with the quantile axis in
  `sorted(config.quantile_levels)` order (the exact output attribute name is the one
  remaining open check).
- **Checkpoint grid** (context / prediction => HF revision):
  - r1 (`ibm-granite/granite-timeseries-ttm-r1`): 512/96 (`main`), 1024/96 (`1024_96_v1`).
  - r2 (`ibm-granite/granite-timeseries-ttm-r2`): 512/96 (`main`), 512/192, 512/336,
    512/720, 1024/{96,192,336,720}, 1536/{96,192,336,720}. Richer grid; larger pretraining.
  - r3 (`ibm-granite/granite-timeseries-ttm-r3`, Apache-2.0): root `main` = 512/30;
    grid (from `ttm.yaml`) spans context 52-3072 and prediction 16-720
    (e.g. 512/{48,96,336}, 1024/{96,720}, 1536/{96,720}, 2048/{96,720}, 2560/..,
    3072/..), each in standard and `lite` sizes. Stock r3 has the quantile head on
    (`multi_quantile_head=True`, `quantile_levels=[0.1..0.9]`,
    `forecast_loss_type="joint"`), single channel, no exog.
  - `prediction_filter_length` can shorten a checkpoint's horizon (not lengthen).
- **Device**: standard torch; move model and `past_values` to the same device.
- **Normalization**: internal (`scaling="std"`); raw input/output.

---

## User-facing usage examples

### Stock r3 (zero-shot, native quantiles)

```python
from skforecast.foundation import FoundationModel, ForecasterFoundation

model = FoundationModel(model_id="ibm-granite/granite-timeseries-ttm-r3")  # default
forecaster = ForecasterFoundation(estimator=model)
forecaster.fit(series=data["target"])
preds     = forecaster.predict(steps=24)                          # point forecast
preds_int = forecaster.predict_interval(steps=24, interval=[0.1, 0.9])  # native, no boot
preds_q   = forecaster.predict_quantiles(steps=24, quantiles=[0.1, 0.5, 0.9])
# quantile_levels auto-derived from the checkpoint config ([0.1..0.9]); a request
# outside those levels raises. No exog (stock r3 is single-channel).
```

### Stock r1/r2 (zero-shot, point-only)

```python
model = FoundationModel(model_id="ibm-granite/granite-timeseries-ttm-r2")
forecaster = ForecasterFoundation(estimator=model)
forecaster.fit(series=data["target"])
preds = forecaster.predict(steps=24)                 # point forecast
# forecaster.predict_interval(...) / predict_quantiles(...) -> ValueError (point-only)
```

### Fine-tuned checkpoint (exog + quantiles), no `FoundationModel` API change

```python
from skforecast.foundation import FoundationModel, ForecasterFoundation

model = FoundationModel(
    model_id="ibm-granite/granite-timeseries-ttm-r2",   # label + routing key only
    finetuned_checkpoint="./checkpoints/my-ttm-r2-finetuned",  # local dir OR
                                                         # "myorg/my-ttm" hub repo;
                                                         # loaded lazily, clone-safe
    context_length=512,
    prediction_length=96,
    exog_channels=["temperature", "is_holiday"],         # enables allow_exog=True
    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],         # optional: pin custom head
                                                         # levels; omit to auto-derive
    device="auto",
)
forecaster = ForecasterFoundation(estimator=model)
forecaster.fit(series=data["target"], exog=data[["temperature", "is_holiday"]])

preds     = forecaster.predict(steps=24, exog=future_exog[["temperature", "is_holiday"]])
preds_int = forecaster.predict_interval(steps=24, interval=[0.1, 0.9],
                                        exog=future_exog[["temperature", "is_holiday"]])
preds_q   = forecaster.predict_quantiles(steps=24, quantiles=[0.25, 0.5, 0.75],
                                         exog=future_exog[["temperature", "is_holiday"]])
```

---

## Adapter contract (all inputs pre-normalized by `FoundationModel`)

`FoundationModel` does all normalization/validation/trimming, then calls the adapter with
canonical dicts. Registry resolution: `_resolve_adapter` prefix-matches `model_id` against
`_ADAPTER_REGISTRY` (`_adapters.py:3949-3985`); `FoundationModel.__init__`
(`_foundation_model.py:251-252`) does `adapter_cls(model_id=model_id, **kwargs)` -- so all
capability kwargs reach the adapter with NO `FoundationModel` change.

Required by `FoundationModel`:
- `allow_exog: bool` -- class attr default `False`, set as an INSTANCE attr in `__init__`
  from `exog_channels is not None`. Used at `_foundation_model.py:954-973`: when `False`,
  any exog is dropped to `None` before calling the adapter (with an
  `IgnoredArgumentWarning`); when `True`, `_prepare_future_exog` runs and `exog` /
  `context_exog` reach `predict`.
- `__init__(self, model_id, *, ...keyword-only...)`.
- `fit(self, context, context_exog) -> self` (both passed by keyword).
- `predict(self, steps, context, context_exog, exog, quantiles) -> dict[str, np.ndarray]`
  (all passed by keyword). Output: `{name: (steps, n_quantiles) array}`; point =
  `(steps, 1)`. Context is already trimmed to the last `context_length` observations
  upstream (it is NOT padded upstream, so the adapter must left-pad short series);
  `context_exog` is trimmed in parallel.
- `get_params()` returns `model_id` + every ctor kwarg EXCEPT the in-memory `model`
  test-injection (empty dict => `None`); excluding `model` keeps clones weight-free,
  which is why fine-tuned weights are passed as the clone-safe `finetuned_checkpoint`
  path rather than an in-memory object.
- `set_params(**params)` via shared `_apply_set_params`.
- Writable attrs read via `FoundationModel` properties: `model_id`, `context_`,
  `context_exog_`, `context_length`, `is_fitted`.

Reuse from `skforecast/foundation/_utils.py`: `_validate_positive_int(name, value)`,
`_tensor_to_numpy(values)`, `_apply_set_params(instance, params, *, validate, resets)`;
and module-level `_resolve_torch_device(device)` (`_adapters.py:27`).

---

## File 1: `skforecast/foundation/_adapters.py` -- new class

Insert after `TSICLAdapter` (end of the class list, before the `_ADAPTER_REGISTRY` dict).
Full implementation to paste (house style: double quotes, <=88 cols, NumPy docstrings, no
en/em dashes). Where a backend detail is unverified it is marked with an "open check"
comment; resolve those against the pinned `granite-tsfm` during coding.

```python
class TinyTimeMixerAdapter:
    """
    Adapter for IBM Granite TinyTimeMixer (TTM) foundation models.

    TinyTimeMixer is a compact, pre-trained time series model. Each released
    checkpoint is tailored to a fixed context length and a fixed forecast
    length. The stock (zero-shot) checkpoints produce point forecasts with no
    quantile head and no exogenous variables. The stock r3 checkpoint
    (Apache-2.0) additionally ships a native quantile head, so it returns
    probabilistic forecasts zero-shot and its levels are auto-derived from the
    config. A checkpoint fine-tuned with channel mixing can also use exogenous
    variables (declare `exog_channels`); a fine-tuned quantile head with custom
    levels can be pinned with `quantile_levels`. Inputs are standardized
    internally (instance normalization), so raw values can be passed directly.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "ibm-granite/granite-timeseries-ttm-r3"
        (default, native quantiles) or "...-ttm-r2" / "...-ttm-r1" (point-only).
        For a custom fine-tuned checkpoint, set `finetuned_checkpoint` (path or
        hub repo) and pass a `model_id` starting with
        "ibm-granite/granite-timeseries-ttm" (used only as a routing key and
        label; `finetuned_checkpoint` provides the weights).
    model : object, default None
        Pre-loaded `TinyTimeMixerForPrediction` instance. TEST-INJECTION ONLY:
        it is not returned by `get_params`, so it does NOT survive `clone`
        (`ForecasterFoundation.__init__` clones the estimator) and must not be
        used as a way to pass a fine-tuned checkpoint. Use
        `finetuned_checkpoint` for that. If `None`, the model is loaded lazily
        on the first call to `predict`.
    finetuned_checkpoint : str, default None
        Local directory or HuggingFace hub repo id of a fine-tuned TinyTimeMixer
        checkpoint. When set, `_load_model` loads it with
        `TinyTimeMixerForPrediction.from_pretrained(finetuned_checkpoint)`
        instead of resolving a stock checkpoint via `get_model`; `model_id`
        is then only a routing key + label. Being a plain string, it is part of
        `get_params` and survives cloning, so it is the supported way to use a
        fine-tuned checkpoint (exog and/or custom quantile head) through
        `ForecasterFoundation`, backtesting, and search.
    context_length : int, default 512
        Number of historical observations used as context. At fit time only the
        last `context_length` observations are stored. At predict time the
        context is left-padded (with a masked pad) when shorter than
        `context_length`. Must match a value supported by the checkpoint and be
        a positive integer.
    prediction_length : int, default 96
        Fixed forecast length of the checkpoint. `predict` accepts
        `steps <= prediction_length` and slices the output to `steps`; a larger
        `steps` raises a `ValueError`. Must be a positive integer.
    device : str, default 'auto'
        Device placement. `"auto"` selects the best available accelerator
        (CUDA > MPS > CPU). Also accepts `"cuda"`, `"mps"`, or `"cpu"`.
    torch_dtype : object, default None
        Torch dtype the model and inputs are cast to (e.g. `torch.bfloat16`).
        When `None` the default `float32` weights are kept.
    exog_channels : list, default None
        Ordered exogenous column names, present ONLY for a checkpoint fine-tuned
        with channel mixing. The order must match the exogenous channel order
        the checkpoint was fine-tuned with. When provided, `allow_exog` is set
        to `True` and skforecast `exog` / `context_exog` are used; when `None`
        (default), the adapter is point/no-exog. Validated against the loaded
        checkpoint config (a mismatch raises).
    quantile_levels : list, default None
        Quantile levels the head returns, each in `(0, 1)`. When `None` (default)
        the levels are auto-derived from the loaded checkpoint config if it has a
        quantile head (stock r3); if the config has no head (stock r1/r2),
        requesting quantiles raises. Pass an explicit list to pin a fine-tuned
        head's custom levels; it is validated against the config (a mismatch
        raises). A requested quantile must be a subset of the effective levels.
    get_model_kwargs : dict, default None
        Additional keyword arguments forwarded verbatim to
        `tsfm_public.toolkit.get_model.get_model` (e.g. `freq_prefix_tuning`,
        `prefer_l1_loss`, `force_return`, `model_revision`). Ignored when
        `finetuned_checkpoint` is set (loaded via `from_pretrained`) or a `model`
        is test-injected. Do not include `context_length` or `prediction_length`
        here; those are controlled by the corresponding adapter parameters.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting (when exog is used).
    context_length : int
        Number of historical observations used as context.
    prediction_length : int
        Fixed forecast length of the checkpoint.
    device : str
        Device placement for the model.
    torch_dtype : object
        Torch dtype for the model and inputs.
    exog_channels : list or None
        Ordered exogenous channel names, or `None` for the point/no-exog path.
    quantile_levels : list or None
        Trained quantile-head levels, or `None` for the point-only path.
    get_model_kwargs : dict
        Additional keyword arguments forwarded to `get_model`.
    finetuned_checkpoint : str or None
        Local dir or hub repo id of a fine-tuned checkpoint, or `None` to use a
        stock checkpoint resolved via `get_model`.
    allow_exog : bool
        Whether the adapter forwards exogenous variables (True iff
        `exog_channels` is set).
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    Stock r3 returns native quantiles zero-shot (levels auto-derived from the
    config). Stock r1/r2 are point forecast only: requesting `quantiles` raises
    unless a fine-tuned quantile head is loaded. Exogenous variables are never
    used on any stock checkpoint (`allow_exog = False`); a checkpoint fine-tuned
    with channel mixing unlocks exog once you declare `exog_channels`. Declared
    capabilities are validated against the checkpoint config at load time and a
    mismatch raises. If the config shows channel mixing you did not declare via
    `exog_channels`, an `IgnoredArgumentWarning` is emitted so you are not
    silently given plain forecasts (a config quantile head, by contrast, is
    adopted automatically).

    Custom / fine-tuned checkpoints are used by passing `finetuned_checkpoint`
    (a local dir or hub repo id loaded lazily with `from_pretrained`), with a
    `model_id` starting with "ibm-granite/granite-timeseries-ttm" so this adapter
    is selected. This keeps the `FoundationModel` public API unchanged and, being
    a plain string, survives the `clone` in `ForecasterFoundation.__init__` (an
    in-memory `model=` would not).

    Both `context_length` and `prediction_length` are fixed per checkpoint;
    `get_model` resolves the matching HuggingFace revision for stock checkpoints.

    References
    ----------
    .. [1] https://github.com/ibm-granite/granite-tsfm

    .. [2] https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2

    .. [3] https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1

    """

    allow_exog: bool = False

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        finetuned_checkpoint: str | None = None,
        context_length: int = 512,
        prediction_length: int = 96,
        device: str = "auto",
        torch_dtype: Any | None = None,
        exog_channels: list[str] | None = None,
        quantile_levels: list[float] | None = None,
        get_model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. "ibm-granite/granite-timeseries-ttm-r2".
        model : object, default None
            Pre-loaded `TinyTimeMixerForPrediction` instance. Test-injection
            only (excluded from `get_params`, so not clone-safe). If `None`, the
            model is loaded lazily on the first call to `predict`.
        finetuned_checkpoint : str, default None
            Local dir or hub repo id of a fine-tuned checkpoint, loaded lazily
            via `from_pretrained`. The clone-safe way to use a fine-tuned
            checkpoint (exog / custom quantile head).
        context_length : int, default 512
            Number of historical observations used as context. Must be a
            positive integer supported by the checkpoint.
        prediction_length : int, default 96
            Fixed forecast length of the checkpoint. Must be a positive integer.
        device : str, default 'auto'
            Device placement. `"auto"` selects CUDA > MPS > CPU.
        torch_dtype : object, default None
            Torch dtype the model and inputs are cast to.
        exog_channels : list, default None
            Ordered exogenous column names for a channel-mixing checkpoint.
        quantile_levels : list, default None
            Trained quantile-head levels (each in `(0, 1)`).
        get_model_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to `get_model`.

        """

        _validate_positive_int("context_length", context_length)
        _validate_positive_int("prediction_length", prediction_length)
        if finetuned_checkpoint is not None and not isinstance(
            finetuned_checkpoint, str
        ):
            raise ValueError(
                "`finetuned_checkpoint` must be a local directory path or a "
                "HuggingFace hub repo id (str), or None."
            )
        exog_channels, quantile_levels = self._validate_capability_params(
            exog_channels, quantile_levels
        )

        self.model_id             = model_id
        self._model               = model
        self.finetuned_checkpoint = finetuned_checkpoint
        self.context_             = None
        self.context_exog_        = None
        self.context_length       = context_length
        self.prediction_length    = prediction_length
        self.device               = device
        self.torch_dtype          = torch_dtype
        self.exog_channels        = exog_channels
        self.quantile_levels      = quantile_levels
        self.get_model_kwargs     = get_model_kwargs or {}
        self.allow_exog           = exog_channels is not None
        self.is_fitted            = False
        self._capabilities_checked = False
        # Effective quantile levels used at predict time: the declared
        # `quantile_levels` if given, else the config head's levels adopted at
        # load (stock r3). `quantile_levels` itself is never mutated, so
        # `get_params` stays clone-faithful.
        self._effective_quantile_levels = quantile_levels

    @staticmethod
    def _validate_capability_params(
        exog_channels: Any,
        quantile_levels: Any,
    ) -> tuple[list[str] | None, list[float] | None]:
        """
        Validate and normalise `exog_channels` and `quantile_levels`.

        Parameters
        ----------
        exog_channels : Any
            Candidate exogenous channel names, or `None`.
        quantile_levels : Any
            Candidate quantile levels, or `None`.

        Returns
        -------
        exog_channels : list or None
            Normalised list of column-name strings, or `None`.
        quantile_levels : list or None
            Normalised list of floats in `(0, 1)`, or `None`.

        """

        if exog_channels is not None:
            if (
                not isinstance(exog_channels, (list, tuple))
                or len(exog_channels) == 0
                or not all(isinstance(c, str) for c in exog_channels)
            ):
                raise ValueError(
                    "`exog_channels` must be a non-empty list of column-name "
                    "strings (the exogenous channels the checkpoint was "
                    "fine-tuned with), or None."
                )
            exog_channels = list(exog_channels)

        if quantile_levels is not None:
            if (
                not isinstance(quantile_levels, (list, tuple))
                or len(quantile_levels) == 0
                or not all(
                    isinstance(q, (int, float)) and 0 < q < 1
                    for q in quantile_levels
                )
            ):
                raise ValueError(
                    "`quantile_levels` must be a non-empty list of floats in "
                    "(0, 1) matching the checkpoint's trained quantile head, or "
                    "None."
                )
            # Stored sorted to match the head's quantile axis, which the model
            # orders by `sorted(config.quantile_levels)`.
            quantile_levels = sorted(float(q) for q in quantile_levels)

        return exog_channels, quantile_levels

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `finetuned_checkpoint`, `context_length`,
            `prediction_length`, `device`, `torch_dtype`, `exog_channels`,
            `quantile_levels`, `get_model_kwargs`. `get_model_kwargs` is returned
            as `None` when no additional config was set. The in-memory `model`
            test-injection is intentionally excluded (kept out so clones stay
            weight-free).

        """
        return {
            "model_id":             self.model_id,
            "finetuned_checkpoint": self.finetuned_checkpoint,
            "context_length":       self.context_length,
            "prediction_length":    self.prediction_length,
            "device":               self.device,
            "torch_dtype":          self.torch_dtype,
            "exog_channels":        self.exog_channels,
            "quantile_levels":      self.quantile_levels,
            "get_model_kwargs":     self.get_model_kwargs or None,
        }

    def set_params(self, **params) -> TinyTimeMixerAdapter:
        """
        Set adapter parameters. Resets the loaded model when a parameter baked
        into it changes (`model_id`, `finetuned_checkpoint`, `context_length`,
        `prediction_length`, `device`, `torch_dtype`, `get_model_kwargs`), then
        re-derives `allow_exog` / the effective quantile levels and forces
        capability re-validation on the next `predict`.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `finetuned_checkpoint`, `context_length`,
            `prediction_length`, `device`, `torch_dtype`, `exog_channels`,
            `quantile_levels`, `get_model_kwargs`.

        Returns
        -------
        self : TinyTimeMixerAdapter

        """

        def validate(p: dict) -> dict:
            if "context_length" in p:
                _validate_positive_int("context_length", p["context_length"])
            if "prediction_length" in p:
                _validate_positive_int("prediction_length", p["prediction_length"])
            if (
                "finetuned_checkpoint" in p
                and p["finetuned_checkpoint"] is not None
                and not isinstance(p["finetuned_checkpoint"], str)
            ):
                raise ValueError(
                    "`finetuned_checkpoint` must be a local directory path or a "
                    "HuggingFace hub repo id (str), or None."
                )
            if "get_model_kwargs" in p:
                p["get_model_kwargs"] = p["get_model_kwargs"] or {}
            if "exog_channels" in p:
                p["exog_channels"], _ = self._validate_capability_params(
                    p["exog_channels"], None
                )
            if "quantile_levels" in p:
                _, p["quantile_levels"] = self._validate_capability_params(
                    None, p["quantile_levels"]
                )
            return p

        _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                (
                    {"model_id", "finetuned_checkpoint", "context_length",
                     "prediction_length", "device", "torch_dtype",
                     "get_model_kwargs"},
                    lambda: setattr(self, "_model", None),
                ),
            ),
        )

        # `_apply_set_params` runs its reset callbacks BEFORE assigning the new
        # values, so state derived from the just-assigned params is recomputed
        # here, once they are in place (reading them inside a reset lambda would
        # see the stale, pre-assignment values). `_effective_quantile_levels` is
        # reset to the declared value; `_reconcile_capabilities` re-derives it
        # from the (possibly reloaded) checkpoint config on the next `predict`,
        # which always runs before the value is read.
        self.allow_exog = self.exog_channels is not None
        self._effective_quantile_levels = self.quantile_levels
        self._capabilities_checked = False

        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> TinyTimeMixerAdapter:
        """
        Store the training series (and past covariates when exog is enabled).
        No model training occurs since TinyTimeMixer is used at inference time.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series past covariates when `allow_exog` is True, else None.

        Returns
        -------
        self : TinyTimeMixerAdapter

        """

        self.context_ = context
        self.context_exog_ = context_exog
        self.is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using TinyTimeMixer.

        All input normalization, validation, and context trimming is performed
        upstream by `FoundationModel`; this method receives pre-processed dicts
        only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast. Must be `<= prediction_length`.
        context : dict pandas Series
            Per-series context windows (already trimmed to `context_length`).
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series past covariates (already trimmed), used only when
            `allow_exog` is True.
        exog : dict pandas DataFrame, pandas Series, or None
            Per-series future covariates for the forecast horizon, used only
            when `allow_exog` is True.
        quantiles : list of float or None
            Quantile levels to return. `None` produces a point forecast
            (shape `(steps, 1)`). Requires the loaded checkpoint to have a
            quantile head (native on r3, or fine-tuned); the requested levels
            must be a subset of the effective levels (declared `quantile_levels`
            or, if not declared, the config head's levels).

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)` (point = `(steps, 1)`).

        Raises
        ------
        ValueError
            If `steps > prediction_length`; if `quantiles` is requested but the
            checkpoint has no quantile head; or if a requested quantile is not a
            subset of the effective quantile levels.

        """

        if steps > self.prediction_length:
            raise ValueError(
                f"`steps` ({steps}) exceeds `prediction_length` "
                f"({self.prediction_length}). TinyTimeMixer uses a fixed "
                f"forecast length per checkpoint; request "
                f"`steps <= prediction_length` or load a checkpoint with a "
                f"longer `prediction_length`."
            )

        # Load + reconcile first so `quantile_levels` auto-derived from a native
        # quantile head (e.g. stock r3) is populated before the subset check.
        self._load_model()
        self._reconcile_capabilities()

        if quantiles is not None:
            if self._effective_quantile_levels is None:
                raise ValueError(
                    "The loaded TinyTimeMixer checkpoint has no quantile head, "
                    "so prediction intervals and quantile forecasts are not "
                    "available. Use an r3 checkpoint (native zero-shot "
                    "quantiles), or a checkpoint fine-tuned with a quantile head "
                    "and pass `quantile_levels` listing its levels."
                )
            missing = [
                q for q in quantiles if q not in self._effective_quantile_levels
            ]
            if missing:
                raise ValueError(
                    f"Requested quantiles {missing} are not among the "
                    f"checkpoint's quantile levels "
                    f"{list(self._effective_quantile_levels)}. Request a subset "
                    f"of the available levels."
                )

        import torch

        series_names_in = list(context.keys())
        forward_kwargs = self._build_inputs(series_names_in, context,
                                            context_exog, exog, torch)

        with torch.no_grad():
            output = self._model(**forward_kwargs)

        return self._extract_predictions(output, series_names_in, steps, quantiles)

    def _build_inputs(
        self,
        series_names_in: list[str],
        context: dict[str, pd.Series],
        context_exog: dict[str, Any] | None,
        exog: dict[str, Any] | None,
        torch: Any,
    ) -> dict[str, Any]:
        """
        Build the forward-pass tensors (left-padded, masked, multi-channel).

        Channel 0 is the target; channels 1..k are the exogenous channels in
        `exog_channels` order (only when `allow_exog` is True). Future-known
        covariates are supplied over the horizon with the target channel masked.

        Parameters
        ----------
        series_names_in : list
            Series names, in order.
        context : dict pandas Series
            Per-series context windows.
        context_exog : dict or None
            Per-series past covariates.
        exog : dict or None
            Per-series future covariates.
        torch : module
            The imported torch module.

        Returns
        -------
        forward_kwargs : dict
            Keyword arguments for the model forward call.

        """

        n_series = len(series_names_in)
        ctx_len = self.context_length
        n_exog = len(self.exog_channels) if self.allow_exog else 0
        n_channels = 1 + n_exog

        past_values = np.zeros((n_series, ctx_len, n_channels), dtype=np.float32)
        past_observed_mask = np.zeros(
            (n_series, ctx_len, n_channels), dtype=np.float32
        )
        for i, name in enumerate(series_names_in):
            values = context[name].to_numpy(dtype=np.float32)
            length = min(values.shape[0], ctx_len)
            past_values[i, ctx_len - length:, 0] = values[-length:]
            past_observed_mask[i, ctx_len - length:, 0] = 1.0
            if n_exog:
                ce = context_exog[name]
                ce = ce[self.exog_channels] if hasattr(ce, "columns") else ce
                ce = np.asarray(ce, dtype=np.float32).reshape(-1, n_exog)
                past_values[i, ctx_len - length:, 1:] = ce[-length:, :]
                past_observed_mask[i, ctx_len - length:, 1:] = 1.0

        device = _resolve_torch_device(self.device)
        x = torch.as_tensor(past_values, device=device)
        mask = torch.as_tensor(past_observed_mask, device=device)
        if self.torch_dtype is not None:
            x = x.to(self.torch_dtype)
            mask = mask.to(self.torch_dtype)
        forward_kwargs = {"past_values": x, "past_observed_mask": mask}

        if n_exog:
            pred_len = self.prediction_length
            future_values = np.zeros(
                (n_series, pred_len, n_channels), dtype=np.float32
            )
            future_observed_mask = np.zeros_like(future_values)
            for i, name in enumerate(series_names_in):
                fe = exog[name]
                fe = fe[self.exog_channels] if hasattr(fe, "columns") else fe
                fe = np.asarray(fe, dtype=np.float32).reshape(-1, n_exog)
                h = min(fe.shape[0], pred_len)
                future_values[i, :h, 1:] = fe[:h, :]
                future_observed_mask[i, :h, 1:] = 1.0
            fv = torch.as_tensor(future_values, device=device)
            fmask = torch.as_tensor(future_observed_mask, device=device)
            if self.torch_dtype is not None:
                fv = fv.to(self.torch_dtype)
                fmask = fmask.to(self.torch_dtype)
            # OPEN CHECK: confirm the exact kwarg names for future covariates and
            # their mask on the pinned granite-tsfm (e.g. `future_values`,
            # `future_observed_mask`); adjust here if they differ.
            forward_kwargs["future_values"] = fv
            forward_kwargs["future_observed_mask"] = fmask

        return forward_kwargs

    def _extract_predictions(
        self,
        output: Any,
        series_names_in: list[str],
        steps: int,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Extract per-series arrays from the model output.

        Point path reads `.prediction_outputs` `(n, pred_len, C)` and returns
        `(steps, 1)` per series. Quantile path reads the quantile output tensor
        and returns `(steps, len(quantiles))` per series, selecting the target
        channel and the requested quantile columns.

        Parameters
        ----------
        output : object
            Model forward output.
        series_names_in : list
            Series names, in order.
        steps : int
            Forecast horizon to slice to.
        quantiles : list of float or None
            Requested quantile levels, or None for a point forecast.

        Returns
        -------
        predictions : dict
            Per-series arrays of shape `(steps, n_quantiles)`.

        """

        # OPEN CHECK: target channel index (config.prediction_channel_indices;
        # assumed 0 here).
        target_ch = 0
        predictions: dict[str, np.ndarray] = {}

        if quantiles is None:
            raw = _tensor_to_numpy(output.prediction_outputs)   # (n, pred_len, C)
            for i, name in enumerate(series_names_in):
                predictions[name] = raw[i, :steps, target_ch].reshape(-1, 1)
            return predictions

        # Quantile head output layout is (n, C, n_quantiles, pred_len), quantile
        # axis in sorted order (confirmed from MultiQuantileHead;
        # `_effective_quantile_levels` is stored sorted). OPEN CHECK: the exact
        # output ATTRIBUTE NAME on the pinned granite-tsfm output dataclass.
        q_raw = _tensor_to_numpy(getattr(output, "quantile_outputs"))
        levels = self._effective_quantile_levels
        q_idx = [levels.index(q) for q in quantiles]
        for i, name in enumerate(series_names_in):
            arr = q_raw[i, target_ch][q_idx, :steps]            # (n_q, steps)
            predictions[name] = arr.T                           # (steps, n_q)
        return predictions

    def _load_model(self) -> None:
        """
        Load the TinyTimeMixer model into `self._model` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `granite-tsfm` is not installed.

        Notes
        -----
        When `finetuned_checkpoint` is set, the weights are loaded from that
        local dir or hub repo id via
        `TinyTimeMixerForPrediction.from_pretrained`. Otherwise the model is
        loaded via `tsfm_public.toolkit.get_model.get_model`, which resolves the
        HuggingFace revision matching the requested `context_length` /
        `prediction_length`. Either way the model is moved to the resolved device
        (and optional `torch_dtype`) and switched to eval mode. This method is a
        no-op when `self._model` is already populated (e.g. a test-injected
        `model`).

        """

        if self._model is not None:
            return
        try:
            from tsfm_public.models.tinytimemixer import (
                TinyTimeMixerForPrediction,
            )
            from tsfm_public.toolkit.get_model import get_model
        except ImportError as exc:
            raise ImportError(
                "granite-tsfm is required for TinyTimeMixerAdapter. "
                "Install it with `pip install granite-tsfm`."
            ) from exc

        if self.finetuned_checkpoint is not None:
            model = TinyTimeMixerForPrediction.from_pretrained(
                self.finetuned_checkpoint
            )
        else:
            model = get_model(
                self.model_id,
                context_length    = self.context_length,
                prediction_length = self.prediction_length,
                **self.get_model_kwargs,
            )
        device = _resolve_torch_device(self.device)
        model = model.to(device)
        if self.torch_dtype is not None:
            model = model.to(self.torch_dtype)
        self._model = model.eval()

    def _reconcile_capabilities(self) -> None:
        """
        Reconcile declared capabilities with the loaded checkpoint config, once.

        Raises when a declared capability (`exog_channels` / `quantile_levels`)
        is not supported by the checkpoint config. When the config exposes a
        quantile head and `quantile_levels` was not declared, adopts the config
        levels (sorted) so a native head (e.g. stock r3) works zero-shot. When
        the config exposes channel mixing and `exog_channels` was not declared,
        warns (`IgnoredArgumentWarning`), since exog cannot be auto-enabled.
        Runs for stock, `finetuned_checkpoint`, and test-injected models; a no-op
        after the first successful check (reset when `set_params` changes a
        relevant parameter).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a declared capability is not supported by the checkpoint config.

        """

        if self._capabilities_checked:
            return
        self._capabilities_checked = True

        config = getattr(self._model, "config", None)
        if config is None:
            # Cannot introspect (e.g. a hand-written stub); trust declarations.
            return

        # OPEN CHECK: confirm the exact config attribute names on the pinned
        # granite-tsfm release.
        config_quantile_levels = getattr(config, "quantile_levels", None)
        ckpt_has_quantiles = bool(
            getattr(config, "multi_quantile_head", False)
            or config_quantile_levels
        )
        num_input_channels = getattr(config, "num_input_channels", 1)
        ckpt_has_exog = bool(
            getattr(config, "enable_forecast_channel_mixing", False)
            or getattr(config, "exogenous_channel_indices", None)
            or num_input_channels > 1
        )

        # 1. Quantiles: declared-but-unsupported => raise; supported-but-undeclared
        #    => adopt the config levels (native head, e.g. stock r3). The declared
        #    `quantile_levels` is never mutated; the effective set is stored
        #    separately for predict.
        if self.quantile_levels is not None and not ckpt_has_quantiles:
            raise ValueError(
                "`quantile_levels` was set, but the loaded TinyTimeMixer "
                "checkpoint config shows no quantile head. Use an r3 checkpoint "
                "or one fine-tuned with a quantile head, or drop "
                "`quantile_levels`."
            )
        if (
            self.quantile_levels is None
            and ckpt_has_quantiles
            and config_quantile_levels
        ):
            self._effective_quantile_levels = sorted(
                float(q) for q in config_quantile_levels
            )

        # 2. Exog: declared-but-unsupported => raise.
        if self.exog_channels is not None:
            if not ckpt_has_exog:
                raise ValueError(
                    "`exog_channels` was set, but the loaded TinyTimeMixer "
                    "checkpoint config shows no channel mixing (no exogenous "
                    "channels). Load a checkpoint fine-tuned with channel "
                    "mixing, or drop `exog_channels`."
                )
            expected_exog = num_input_channels - 1
            if expected_exog >= 0 and len(self.exog_channels) != expected_exog:
                raise ValueError(
                    f"`exog_channels` has {len(self.exog_channels)} names but "
                    f"the checkpoint expects {expected_exog} exogenous "
                    f"channel(s) (num_input_channels={num_input_channels}). "
                    f"Provide exactly the exogenous channels, in the order the "
                    f"checkpoint was fine-tuned with."
                )

        # 3. Channel mixing supported but exog undeclared => warn (cannot be
        #    auto-enabled: needs the name->channel mapping and the pre-load
        #    `allow_exog` timing). Quantiles, by contrast, were auto-adopted above.
        if ckpt_has_exog and self.exog_channels is None:
            from ..exceptions import IgnoredArgumentWarning
            warnings.warn(
                "The loaded TinyTimeMixer checkpoint appears to support "
                "exogenous variables (channel mixing), but `exog_channels` was "
                "not declared; the adapter ignores exog. Pass `exog_channels` "
                "to use them.",
                IgnoredArgumentWarning,
            )
```

## File 1 (cont.): register in `_ADAPTER_REGISTRY` (`_adapters.py:3949-3959`)

Add one entry (prefix match covers r3 `...-ttm-r3`, r2 `...-ttm-r2`, r1
`...-ttm-r1`, and the fine-tuned "label" ids that start with the same prefix):

```python
    "ibm-granite/granite-timeseries-ttm": TinyTimeMixerAdapter,
```

---

## File 2: `skforecast/foundation/_foundation_model.py` -- docstring only (NO API change)

The public `FoundationModel` signature is unchanged; capability params flow through
`**kwargs`. Only docstrings are updated:
- "Available model IDs" list (~lines 40-74): add
  `ibm-granite/granite-timeseries-ttm-r3` (default, native quantiles),
  `ibm-granite/granite-timeseries-ttm-r2`, `ibm-granite/granite-timeseries-ttm-r1`.
- "Commonly used kwargs by model" (~83-115): add TTM line noting `context_length`,
  `prediction_length`, `device`, `torch_dtype`, `exog_channels`, `quantile_levels`
  (optional; auto-derived on r3), `finetuned_checkpoint`, `get_model_kwargs`, and the
  `finetuned_checkpoint=<path>` pattern for fine-tuned checkpoints.
- "References" (~193-240): add granite-tsfm GitHub + HF r1/r2 entries.

No change to `__init__.py` (adapters are not exported there).

---

## File 3: Tests -- `skforecast/foundation/tests/tests_foundation_models/`

Follow `.github/instructions/testing.instructions.md` and the `test_TabPFNAdapter.py`
(exog + quantiles) / `test_TimesFMAdapter.py` (fixed horizon) templates. NO
`importorskip` / `monkeypatch`; inject a hand-written fake via the `model` ctor arg so the
lazy loader is a no-op. `torch` IS available in the `test` extra (`pyproject.toml:117`), so
building input tensors in `predict` works.

### 3a. Add to `fixtures_adapters.py`

```python
class FakeTinyTimeMixerOutput:
    """Stand-in for TinyTimeMixerForPredictionOutput (point and/or quantile)."""
    def __init__(self, prediction_outputs=None, quantile_outputs=None):
        self.prediction_outputs = prediction_outputs
        if quantile_outputs is not None:
            self.quantile_outputs = quantile_outputs


class FakeTinyTimeMixerConfig:
    """Minimal stand-in for the checkpoint config. Defaults describe a plain
    point-only checkpoint (no quantile head, single channel), matching stock
    r1/r2. For a stock-r3-style config pass `multi_quantile_head=True` and
    `quantile_levels=[0.1, 0.2, ..., 0.9]`."""
    def __init__(
        self,
        multi_quantile_head=False,
        quantile_levels=None,
        enable_forecast_channel_mixing=False,
        exogenous_channel_indices=None,
        num_input_channels=1,
    ):
        self.multi_quantile_head = multi_quantile_head
        self.quantile_levels = quantile_levels
        self.enable_forecast_channel_mixing = enable_forecast_channel_mixing
        self.exogenous_channel_indices = exogenous_channel_indices
        self.num_input_channels = num_input_channels


class FakeTinyTimeMixer:
    """
    Minimal stand-in for `TinyTimeMixerForPrediction` used in tests. Records the
    tensors it is called with. Point mode returns the last observed target value
    repeated across the (fixed) prediction length; quantile mode additionally
    returns a `quantile_outputs` tensor. Pass a `config` to simulate a capable
    (fine-tuned) checkpoint.
    """
    def __init__(self, prediction_length=96, config=None, n_quantiles=None):
        self.prediction_length = prediction_length
        self.config = config if config is not None else FakeTinyTimeMixerConfig()
        self.n_quantiles = n_quantiles
        self.last_forward_kwargs = None

    def eval(self):
        return self

    def to(self, arg):  # accepts a device string or a torch dtype
        return self

    def __call__(self, past_values, past_observed_mask=None, **kwargs):
        self.last_forward_kwargs = {
            "past_values": past_values,
            "past_observed_mask": past_observed_mask,
            **kwargs,
        }
        arr = np.asarray(past_values)                          # (n, ctx, C)
        n, _, n_channels = arr.shape
        last_target = arr[:, -1, 0]                            # (n,) target ch 0
        point = np.repeat(
            last_target[:, None], self.prediction_length, axis=1
        ).astype(np.float32)[:, :, None]                       # (n, pred_len, 1)
        # pad point to n_channels so channel-0 slicing works like the real model
        point = np.repeat(point, n_channels, axis=2)
        if self.n_quantiles is None:
            return FakeTinyTimeMixerOutput(prediction_outputs=point)
        # quantile_outputs layout (n, C, n_quantiles, pred_len); each quantile
        # offset by its index so tests can distinguish columns. point is
        # (n, pred_len, C) -> (n, C, pred_len) then add the quantile axis.
        base = np.transpose(point, (0, 2, 1))                  # (n, C, pred_len)
        q = np.stack(
            [base + k for k in range(self.n_quantiles)], axis=2
        ).astype(np.float32)                                   # (n, C, n_q, pred_len)
        return FakeTinyTimeMixerOutput(prediction_outputs=point, quantile_outputs=q)
```

Note: tensors reaching the fake are torch tensors; `np.asarray(tensor_on_cpu)` works.
If a device dtype makes `np.asarray` fail in CI, cast via `.cpu().numpy()` inside the fake.
Reuse existing shared fixtures/builders in this file (`y`, `y_wide`, `y_dict`,
`exog`-style builders, `prepare_fit_args`, `prepare_predict_args`). Align the fake's
`quantile_outputs` layout with whatever the "Open implementation checks" confirm; if the
real layout differs, update both `_extract_predictions` and this fake together.

### 3b. New file `test_TinyTimeMixerAdapter.py`

Header `# Unit test TinyTimeMixerAdapter`, imports in house order, then a local injector:

```python
def make_adapter(fake_config=None, n_quantiles=None, **kwargs) -> TinyTimeMixerAdapter:
    prediction_length = kwargs.get("prediction_length", 96)
    adapter = TinyTimeMixerAdapter(
        model_id=kwargs.pop("model_id", "ibm-granite/granite-timeseries-ttm-r3"),
        **kwargs,
    )
    adapter._model = FakeTinyTimeMixer(
        prediction_length=prediction_length,
        config=fake_config,
        n_quantiles=n_quantiles,
    )
    return adapter
```

Tests (grouped with `# ====` banners, multi-line docstrings, parametrized, aligned):

1. **init / capability params**:
   - `test_init_default_params`: `context_length==512`, `prediction_length==96`,
     `device=="auto"`, `torch_dtype is None`, `exog_channels is None`,
     `quantile_levels is None`, `finetuned_checkpoint is None`,
     `get_model_kwargs=={}`, `_model is None`, `context_ is None`,
     `context_exog_ is None`, `is_fitted is False`, `allow_exog is False`,
     `_capabilities_checked is False`, `_effective_quantile_levels is None`.
   - `test_init_allow_exog_true_when_exog_channels_set`:
     `exog_channels=["a","b"]` => `allow_exog is True`.
   - `test_init_ValueError_when_context_length_invalid` /
     `..._prediction_length_invalid` parametrized `[0, -1, None]`.
   - `test_init_ValueError_when_exog_channels_invalid` parametrized
     `[[], [1, 2], "a", 3]`.
   - `test_init_ValueError_when_quantile_levels_invalid` parametrized
     `[[], [0], [1.5], [0.5, "x"], 0.5]`.
   - `test_init_ValueError_when_finetuned_checkpoint_invalid` parametrized
     `[3, ["p"], {}]` (non-str, non-None).
2. **get_params/set_params**: exact key set (incl. `finetuned_checkpoint`,
   `exog_channels`, `quantile_levels`); `model` is NOT a key; `set_params` updates
   + resets `_model` to `None` for model-baked keys (incl. `finetuned_checkpoint`)
   and `_capabilities_checked` to `False`; unknown param -> "Invalid parameter";
   no `_model` reset when a model-baked value is unchanged; invalid values raise.
   - `test_set_params_exog_channels_rederives_allow_exog`: on an adapter built with
     `exog_channels=None` (so `allow_exog is False`), `set_params(exog_channels=
     ["a","b"])` => `allow_exog is True` afterwards (guards the reset-ordering fix:
     `allow_exog` is re-derived AFTER `_apply_set_params` assigns the new value);
     and `set_params(exog_channels=None)` flips it back to `False`.
3. **fit**: `test_fit_output_single_series` (asserts `is_fitted`, stored context,
   stored `context_exog_`, does-not-modify-input); `test_fit_output_multi_series`
   parametrized `[y_wide, y_dict]`.
4. **predict (point, default path)**:
   - `test_predict_ValueError_when_quantiles_requested_without_quantile_levels`
     (match the point-only message).
   - `test_predict_ValueError_when_steps_exceed_prediction_length`
     (`prediction_length=96`, `steps=200`).
   - `test_predict_output_single_series` / `..._multi_series`: dict keys, each value
     shape `(steps, 1)`, values via `np.testing.assert_array_almost_equal` against the
     fake's deterministic output (last target repeated, sliced to `steps`).
   - `test_predict_left_pads_short_context`: context shorter than `context_length`
     => inspect `adapter._model.last_forward_kwargs["past_values"]` shape
     `(n, context_length, 1)` and `past_observed_mask` (zeros in the left-pad region,
     ones elsewhere).
5. **predict (exog path)**:
   - `test_predict_exog_builds_multichannel_input`: `exog_channels=["e1","e2"]` +
     matching config (`num_input_channels=3`, channel mixing on); inspect
     `last_forward_kwargs`: `past_values` width == 3, exog channels populated from
     `context_exog`, and `future_values` / `future_observed_mask` present with the
     horizon exog filled and target channel masked (zeros).
   - `test_predict_exog_output_shape`: point output still `(steps, 1)`.
6. **predict (quantile path)**:
   - `test_predict_quantiles_output_shape`: explicit `quantile_levels=[0.1,0.5,0.9]`,
     `n_quantiles=3`, matching quantile-head config; request `quantiles=[0.1,0.9]`
     => each value shape `(steps, 2)`; assert columns map to the requested levels
     (fake's per-index offset, sorted order), reading the corrected
     `(n, C, n_quantiles, pred_len)` fake output.
   - `test_predict_autoderives_quantile_levels_from_config`: r3-style config
     (`multi_quantile_head=True`, `quantile_levels=[0.1..0.9]`, `n_quantiles=9`),
     NO `quantile_levels` passed; request `quantiles=[0.1,0.9]` => shape
     `(steps, 2)`; `adapter.quantile_levels is None` (declared unchanged) and
     `adapter._effective_quantile_levels == [0.1,...,0.9]` after predict.
   - `test_predict_explicit_quantile_levels_override_config`: user passes a subset
     of custom levels; it is honored (not overwritten) and the request is
     subset-validated against it.
   - `test_predict_quantile_column_order_uses_sorted_levels`: pass
     `quantile_levels` unsorted; assert columns map by the SORTED index.
   - `test_predict_ValueError_when_quantile_not_subset`: request `[0.05]` not in
     the effective levels => raises listing available levels.
7. **capability reconciliation** (`_reconcile_capabilities`):
   - `test_predict_raises_when_quantile_levels_declared_but_config_lacks_head`:
     `quantile_levels` set + plain config => `ValueError`.
   - `test_predict_raises_when_exog_channels_declared_but_config_lacks_mixing`:
     `exog_channels` set + plain config => `ValueError`.
   - `test_predict_raises_when_exog_channels_count_mismatches_config`:
     `exog_channels=["a"]` but `num_input_channels=3` => `ValueError`.
   - `test_predict_no_warning_for_plain_checkpoint`: default (r1/r2-style) config +
     no declared caps => NO `IgnoredArgumentWarning`.
   - `test_predict_adopts_undeclared_quantiles_without_warning`: r3-style config
     (`multi_quantile_head=True`, `quantile_levels=[0.1..0.9]`) and no
     `quantile_levels` => NO warning; `adapter._effective_quantile_levels`
     populated from the config (sorted), `adapter.quantile_levels is None`.
   - `test_predict_warns_when_config_has_undeclared_exog`: parametrize
     `enable_forecast_channel_mixing=True`, `exogenous_channel_indices=[1]`,
     `num_input_channels=2` (no `exog_channels`) => one `IgnoredArgumentWarning`
     mentioning "exogenous variables".
   - `test_predict_checks_only_once`: two `predict` calls => reconciliation runs
     once (`_capabilities_checked is True`; exactly one warning across both; and
     an auto-derived `quantile_levels` is not re-derived).
   - `test_set_params_reset_reenables_capability_check`: after a check, a
     `set_params` change resets `_capabilities_checked` (and `_model` for
     model-baked keys / `allow_exog` for `exog_channels`).
   - `test_reconcile_no_config_is_silent`: model without a `config` attribute does
     not raise and does not warn (declarations trusted).

### 3c. Update shared adapter-enumeration tests
- `test_resolve_adapter.py` (3 spots): import `TinyTimeMixerAdapter`; add
  `("ibm-granite/granite-timeseries-ttm-r3", TinyTimeMixerAdapter)` (plus r2/r1 cases) to
  the `model_id -> cls` parametrization; add the registry entry to the re-declared
  expected dict.
- `test_clone.py`: add a TTM `model_id` row (r3, + `ids`) and a config row that exercises
  `finetuned_checkpoint` / `exog_channels` / `quantile_levels` / `get_model_kwargs`,
  asserting these survive `clone` (the clone-safe fine-tuned path). If the shared
  test harness would instantiate the real backend, keep this row to config that does
  not trigger a load (clone only round-trips `get_params`, so no backend import).

---

## File 4: Docs -- add a TTM row/entry everywhere the adapter list is enumerated

Hand-edited sources (identical adapter table in three files):
- `AGENTS.md`: table (~512-521), architecture class-name line (~107), backend pip list (~484).
- `.github/copilot-instructions.md`: mirror (~467-476, ~62, ~439).
- `tools/ai/llms-base.txt`: mirror (~467-476, ~62, ~439).

New table row (AGENTS.md column format; columns: Adapter | model_id prefix | Exog |
Default context_length | Quantiles):
```
| TinyTimeMixerAdapter (IBM) | `ibm-granite/granite-timeseries-ttm` | With fine-tuning (`exog_channels`) | 512 | Zero-shot (r3); fine-tuned head (r1/r2) |
```
Architecture/class-name line: append `TinyTimeMixerAdapter`. Backend pip list: append
`granite-tsfm`. Where the foundation forecaster IDs are enumerated, list r3 (default,
native quantiles), r2, r1; note all three are Apache-2.0.

Skill docs:
- `skills/foundation-forecasting/SKILL.md`: front-matter model list, Installation
  (`pip install granite-tsfm`), "Choosing a Model" table (stock r3 = native zero-shot
  quantiles, no exog, fixed context/horizon; stock r1/r2 = point-only; fine-tuned = exog
  via `exog_channels`, custom quantile head via `quantile_levels`), quantile-support
  paragraph (r3 quantiles zero-shot with auto-derived levels, subset-required; r1/r2
  point-only; explicit `quantile_levels` overrides; undeclared channel mixing triggers an
  `IgnoredArgumentWarning`), references. Include the stock-r3 quantile snippet and the
  fine-tuned `finetuned_checkpoint=<path>` snippet.
- `skills/foundation-forecasting/references/adapter-parameters.md`: contents list; a new
  `## TinyTimeMixerAdapter` section (prefix `ibm-granite/granite-timeseries-ttm`;
  `allow_exog` derived from `exog_channels`; param table: `finetuned_checkpoint`,
  `context_length`, `prediction_length`, `device`, `torch_dtype`, `exog_channels`,
  `quantile_levels`, `get_model_kwargs` (mention `model` only as a test-only,
  non-clone-safe injection); a "Quantiles and fine-tuned checkpoints"
  note (r3 native quantiles auto-derived from config; explicit `quantile_levels`
  overrides; fine-tuned exog via `finetuned_checkpoint=<path>` with a `model_id`
  starting with the registered prefix -> no `FoundationModel` API change and clone-safe
  (an in-memory `model=` is dropped by `clone`); declared caps validated against config,
  subset-required
  quantiles, warn on undeclared channel mixing); add a row to the "Tunable parameters
  and model reload cost" table.

Generated (do NOT hand-edit; regenerate with the `tools/ai/` build script after editing the
sources above): `docs/llms-full.txt`, `site/llms*.txt`, root `llms*.txt`.

`CLAUDE.md` (skills list only) and `README.md` (no adapter table) need no per-adapter edit.

---

## File 5: `pyproject.toml` -- NO change

Foundation backends are intentionally not declared as extras (lazy-imported). Document the
manual `pip install granite-tsfm` in the adapter `ImportError` and SKILL.md, matching the
existing convention. Tests inject a fake, so `granite-tsfm` is not needed in the `test` extra.

---

## File 6 (OPTIONAL, Phase 3): `finetune_tinytimemixer` helper

Rationale: enabling exog (and custom quantile heads) requires a fine-tuned TTM
checkpoint. The adapter consumes such a checkpoint via `finetuned_checkpoint` (a path or
hub repo loaded lazily) but does NOT train (see the design decision below). To lower the
fine-tuning barrier WITHOUT corrupting the zero-shot `fit`/backtesting/search contract,
add a standalone, explicit helper that the user calls deliberately (once, offline) and
whose saved checkpoint is then loaded via `finetuned_checkpoint`. This is a
separate concern from `ForecasterFoundation` and touches neither `FoundationModel` nor
the adapter's `fit`.

### Why NOT put fine-tuning inside the adapter `fit` (design decision, confirmed)

- `backtesting_foundation` deep-copies `cv` and forces `refit=True`,
  `fixed_train_size=False` because foundation `fit` is assumed free
  (`_forecaster_foundation.py:792`, "no weights are ever trained"). Training in `fit`
  would re-fine-tune on EVERY backtest fold and EVERY `bayesian_search_foundation`
  trial. Non-starter.
- `allow_exog` is read by `FoundationModel` BEFORE `fit`, so exog cannot be "detected"
  late; the user must declare it up front regardless. Auto-train-on-detect removes no
  declaration, only adds a training loop.
- Every other adapter treats `fit` as "store context"; training in TTM's `fit` would be
  surprising and inconsistent.
- Correct fine-tuning is a large, fragile surface (channel-mixing config, backbone
  freezing, optimizer/loss/epochs/lr/val-split/seeds/batching) that `granite-tsfm`
  already implements; the helper WRAPS that tooling rather than reinventing it.

### Placement and export

New module `skforecast/foundation/_finetune.py`; export
`finetune_tinytimemixer` from `skforecast/foundation/__init__.py` (user-facing utility,
adapters are not exported). All heavy imports (`tsfm_public`, `transformers` `Trainer`)
are lazy, inside the function, matching the backend convention (no new extra).

### Signature (house style: double quotes, <=88 cols, NumPy docstring)

```python
def finetune_tinytimemixer(
    y: pd.Series,
    exog: pd.DataFrame,
    *,
    model_id: str = "ibm-granite/granite-timeseries-ttm-r3",
    context_length: int = 512,
    prediction_length: int = 96,
    exog_kind: str = "future_known",   # "future_known" -> control_columns
                                       # "past_only"     -> conditional_columns
    freeze_backbone: bool = True,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    validation_size: float = 0.1,
    device: str = "auto",
    output_dir: str | None = None,
    random_state: int | None = None,
    trainer_kwargs: dict | None = None,
) -> tuple[Any, list[str]]:
    """
    Fine-tune a TinyTimeMixer checkpoint for exogenous channel mixing and return
    the trained model plus the ordered exog channel names to inject.

    Returns
    -------
    model : TinyTimeMixerForPrediction
        Fine-tuned model. Persist it with `model.save_pretrained(<dir>)` (or set
        `output_dir`, which saves it for you) and pass that `<dir>` as
        `FoundationModel(finetuned_checkpoint=<dir>)`; this is the clone-safe way
        to consume it (an in-memory `model=` would be dropped by the `clone` in
        `ForecasterFoundation.__init__`).
    exog_channels : list of str
        Exogenous column names in the channel order the model was fine-tuned
        with; pass verbatim as `FoundationModel(..., exog_channels=...)`.
    """
```

### Workflow it wraps (grounded in `granite-tsfm`; unverified specifics = OPEN CHECK)

1. Assemble the wide training frame: reset `y`'s DatetimeIndex to a timestamp column,
   join `exog` (aligned on index), so columns = `[timestamp, target, *exog.columns]`.
   Preserve `exog.columns` order as the returned `exog_channels`.
2. Build a `TimeSeriesPreprocessor` with
   `target_columns=[y.name]`, and `control_columns=list(exog.columns)` (future-known)
   or `conditional_columns=...` (past-only), `context_length`, `prediction_length`,
   `scaling=True`, `scaler_type="standard"`. Derive train/valid split via
   `get_datasets(tsp, frame, split_params)` (`validation_size`, `random_state`).
3. Load the trainable model via
   `get_model(model_id, context_length=..., prediction_length=...,
   num_input_channels=tsp.num_input_channels,
   prediction_channel_indices=tsp.prediction_channel_indices,
   decoder_mode="mix_channel", enable_forecast_channel_mixing=True)`.
   OPEN CHECK: exact FCM kwargs and whether `exogenous_channel_indices` /
   `fcm_context_length` must be passed explicitly on the pinned release; on r3 the
   native quantile head should be preserved through exog fine-tuning (confirm).
4. If `freeze_backbone`: `for p in model.backbone.parameters(): p.requires_grad =
   False` (head/decoder-only fine-tune; fast, small data).
5. Train with HF `Trainer` + `TrainingArguments(output_dir=output_dir or a temp dir,
   num_train_epochs=num_epochs, learning_rate=learning_rate,
   per_device_train_batch_size=batch_size, seed=random_state, ...trainer_kwargs)` on the
   resolved device.
6. Return `(model.eval(), exog_channels)`. When `output_dir` is given, the helper calls
   `model.save_pretrained(output_dir)` so that dir can be passed straight to
   `FoundationModel(finetuned_checkpoint=output_dir)`; when it is `None` the helper does
   not persist (keep side effects explicit and minimal), and the caller saves the
   returned model themselves before using `finetuned_checkpoint`.

### Usage

```python
from skforecast.foundation import (
    FoundationModel, ForecasterFoundation, finetune_tinytimemixer,
)

ttm, exog_channels = finetune_tinytimemixer(
    y=data["target"],
    exog=data[["temperature", "is_holiday"]],
    prediction_length=96,
    num_epochs=5,
    output_dir="./checkpoints/my-ttm-r3-finetuned",   # saved for reuse
)

model = FoundationModel(
    model_id="ibm-granite/granite-timeseries-ttm-r3",
    finetuned_checkpoint="./checkpoints/my-ttm-r3-finetuned",  # clone-safe path
    exog_channels=exog_channels,          # order returned by the helper
)
forecaster = ForecasterFoundation(estimator=model)
forecaster.fit(series=data["target"], exog=data[["temperature", "is_holiday"]])
preds = forecaster.predict(steps=24, exog=future_exog[["temperature", "is_holiday"]])
```

### Tests

Fine-tuning cannot be exercised without the real backend, so keep unit tests light and
do NOT train in CI:
- `test_finetune_import_error_without_backend`: monkeypatch-free; call with the backend
  absent (simulate via the lazy `ImportError` path) => clear `ImportError` naming
  `granite-tsfm`.
- `test_finetune_returns_exog_channels_order`: with a tiny fake `get_model` / `Trainer`
  injected through module-level seams (or a `_trainer_factory` param), assert the
  returned `exog_channels` equals `list(exog.columns)` and that the frame assembly and
  column-specifier wiring are correct. Real training is a manual/optional smoke test.
- Mark any real-backend fine-tune test `@pytest.mark.slow` and skip by default.

### Docs

Add a short "Fine-tuning for exogenous variables" subsection to
`skills/foundation-forecasting/SKILL.md` and the `adapter-parameters.md` fine-tuned
note, pointing at `finetune_tinytimemixer` + the `finetuned_checkpoint=<path>` snippet,
and stating the train-once/save/load workflow keeps backtesting and search zero-shot.

### Scope note

Phase 3 is independent and optional. Phases 1-2 (point + zero-shot r3 quantiles;
injection-based exog) ship without it. The helper only reduces the friction of PRODUCING
a fine-tuned checkpoint; the adapter's consumption path is unchanged.

---

## Verification

1. Confirm conda env (memory: `skforecast_24_py13`) before running Python; verify with
   `where.exe python`.
2. `pytest skforecast/foundation/tests/tests_foundation_models/test_TinyTimeMixerAdapter.py -vv`
   plus the updated `test_resolve_adapter.py` and `test_clone.py` -- all green, no real
   backend imported.
3. Registry sanity:
   `FoundationModel(model_id="ibm-granite/granite-timeseries-ttm-r3").adapter` is a
   `TinyTimeMixerAdapter` (and r2/r1); a fine-tuned "label" id (same prefix) +
   `finetuned_checkpoint=<path>` also resolves to `TinyTimeMixerAdapter` with
   `allow_exog` reflecting `exog_channels`. Clone-safety check: after
   `ForecasterFoundation(estimator=model)` (which runs `clone`),
   `forecaster.estimator.adapter.get_params()["finetuned_checkpoint"]` still equals the
   path and `allow_exog` is still `True` (confirming the fine-tuned config survives the
   clone, unlike an in-memory `model=`).
4. r3 quantile path (fake r3 config): `.predict_interval(...)` / `.predict_quantiles(...)`
   work with NO `quantile_levels` passed (auto-derived from config); an explicit
   `quantile_levels` overrides; requesting a level outside the effective set raises.
5. Point-only guards (fake r1/r2 config): `.predict_interval(...)` /
   `.predict_quantiles(...)` raise the reworded `ValueError`; `.predict(steps=...)`
   returns a point forecast; `steps > prediction_length` raises.
6. Fine-tuned exog guards: declaring `exog_channels` the (fake) config does not support
   raises at first `predict`; undeclared channel mixing warns.
7. Optional real smoke test (throwaway env with `granite-tsfm`; note torch>=2.10 may clash
   with the pinned env): (a) stock r3 512/96 zero-shot quantile forecast, confirming the
   quantile output attribute name and the `(n, C, n_q, pred_len)` layout, plus `steps<=96`
   slicing and short-context left-padding; (b) if a fine-tuned checkpoint is available, an
   exog forecast validating the future-covariate forward kwargs.
8. `ruff check` clean (line length 88, double quotes, no en/em dashes in comments/docstrings).
9. (Phase 3, if implemented) `finetune_tinytimemixer` light tests green (import-error path,
   returned `exog_channels` order + frame/column-specifier wiring via injected fake
   trainer); no training in CI. Optional manual smoke test: fine-tune on a small exog
   dataset, `save_pretrained`, load via `finetuned_checkpoint=<path>`, and confirm
   `predict` / `predict_interval` with exog.

## Open implementation checks (resolve during coding)

Confirmed against `granite-tsfm` main during R3 validation:
- Stock r3 config exposes `multi_quantile_head=True`,
  `quantile_levels=[0.1..0.9]`, `forecast_loss_type="joint"`, single channel, no
  channel mixing (config attribute names used by `_reconcile_capabilities` are
  correct: `multi_quantile_head`, `quantile_levels`,
  `enable_forecast_channel_mixing`, `exogenous_channel_indices`,
  `num_input_channels`).
- `get_model` resolves the r3 grid (context 52-3072, pred 16-720) via `ttm.yaml`;
  512/96 exists as `512-96-dec-512-r3`.
- Quantile head (`MultiQuantileHead`) emits `(n, C, n_quantiles, pred_len)` with the
  quantile axis in `sorted(config.quantile_levels)` order.

Still to confirm during coding:
- The exact quantile output ATTRIBUTE NAME on `TinyTimeMixerForPredictionOutput`
  (point field confirmed `.prediction_outputs`; the plan assumes
  `output.quantile_outputs`). Align `_extract_predictions` and the test fake together
  if it differs. Also re-confirm `forward` accepts `past_observed_mask`.
- The **future-known covariate** forward kwargs (names for `future_values` /
  `future_observed_mask`, and whether the target channel must be masked vs
  zero-filled) and whether past covariates belong in `past_values` alongside the
  target or in a separate argument. Adjust `_build_inputs` accordingly. (Fine-tuned
  exog path only; not exercised by stock r3.)
- The target channel index (`prediction_channel_indices`, assumed 0).
- Whether `context_exog` is trimmed to `context_length` upstream (parallel to
  `context`); if not, trim in `_build_inputs`.
