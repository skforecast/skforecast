# Plan: Add a `TinyTimeMixerAdapter` (IBM TTM) foundation-model adapter

## Context

skforecast's `foundation` subsystem wraps zero-shot time-series foundation models
behind a uniform adapter contract (`ChronosAdapter`, `TimesFMAdapter`,
`MoiraiAdapter`, `TabICLAdapter`, `TabPFNAdapter`, `T0Adapter`, `NoriAdapter`,
`TSICLAdapter`). Goal: add IBM Granite's **TinyTimeMixer (TTM)** so users can run
`ForecasterFoundation` / `backtesting_foundation` / `bayesian_search_foundation`
against `ibm-granite/granite-timeseries-ttm-r2` (and r1) with zero training.

All adapters live in one module: `skforecast/foundation/_adapters.py`. The closest
template to copy is **`TimesFMAdapter`** (`_adapters.py:483-880`): fixed horizon,
`allow_exog=False`, lazy load. Device handling mirrors `T0Adapter` (`_adapters.py:2438+`):
`_resolve_torch_device`, `.to(device)`, optional `torch_dtype`.

### Verified TTM behavior (source: `granite-tsfm` main + released r2 `config.json`)

- **Point-forecast only (zero-shot).** `MultiQuantileHead` / `MultiPinballLoss` exist
  but are gated behind `multi_quantile_head=False` and only active after *fine-tuning*.
  Released r2 ships `loss="mse"`, no quantile keys. Conformal intervals exist only as a
  separate post-hoc `granite-tsfm` wrapper (`PostHocProbabilisticProcessor`), which
  skforecast does not use. `ForecasterFoundation` has NO bootstrapping/conformal fallback
  (`_forecaster_foundation.py:792`: intervals come "directly from the model's native
  quantile output"). => a point-only adapter must raise on any `quantiles` request.
- **No exogenous variables (zero-shot).** FCM / channel-mixing
  (`enable_forecast_channel_mixing=False`, `exogenous_channel_indices=null`,
  `num_input_channels=1` in the released checkpoint) requires fine-tuning. => `allow_exog=False`.
- **Fixed `context_length` AND fixed `prediction_length` per checkpoint.** Input must be
  exactly `context_length` (left-pad if shorter); horizon is the checkpoint's fixed
  `prediction_length` (can only be shortened).
- **Internal instance normalization** (`scaling="std"`) => feed raw values, read raw back.
- Standard torch `PreTrainedModel`: `.to(device)`, `.eval()`, `forward(past_values=...)`.

### Design decisions (confirmed with user)

- **Point-only**: `allow_exog = False`; requesting `quantiles` (i.e. `predict_interval` /
  `predict_quantiles`) raises a clear `ValueError`.
- **Fixed horizon**: constructor param `prediction_length` selects the checkpoint horizon;
  `predict` requires `steps <= prediction_length`, slices the fixed output down to `steps`,
  and raises if `steps > prediction_length` (mirrors `TimesFMAdapter.max_horizon`).
- **Defaults**: `context_length=512`, `prediction_length=96`; documented default
  `model_id="ibm-granite/granite-timeseries-ttm-r2"` (r2 `main` branch).

---

## TTM backend API reference (embedded so no re-research is needed)

- **Install**: `pip install granite-tsfm`. Not in mainline `transformers`; goes through
  `tsfm_public`. Requires (current main) python 3.11-3.13, `torch>=2.10`,
  `transformers[torch]>=4.57`; older releases allowed py3.10 / looser torch. Depend loosely.
- **Recommended loader** (resolves the checkpoint branch automatically):
  ```python
  from tsfm_public.toolkit.get_model import get_model
  model = get_model(
      "ibm-granite/granite-timeseries-ttm-r2",
      context_length=512, prediction_length=96,
      # optional: freq_prefix_tuning, freq, prefer_l1_loss, prefer_longer_context,
      #           force_return ("zeropad"/"rolling"/...), model_revision
  )
  ```
  `get_model` reads `tsfm_public/resources/model_paths_config/ttm.yaml`, picks the entry
  whose context/prediction fits, and calls `from_pretrained(..., revision=<branch>,
  prediction_filter_length=...)` internally.
- **Forward pass** (this is the path the adapter uses):
  ```python
  out = model(past_values=x, past_observed_mask=mask)   # x: (batch, context_length, n_channels)
  preds = out.prediction_outputs                        # (batch, prediction_length, n_channels)
  ```
  `TinyTimeMixerForPrediction` is a `PreTrainedModel`. `past_observed_mask` (same shape as
  `past_values`, 1=observed, 0=missing/pad) lets the internal std-scaler ignore padding.
- **Checkpoint grid** (context / prediction => HF revision):
  - r1 (`ibm-granite/granite-timeseries-ttm-r1`): 512/96 (`main`), 1024/96 (`1024_96_v1`).
  - r2 (`ibm-granite/granite-timeseries-ttm-r2`): 512/96 (`main`), 512/192, 512/336,
    512/720, 1024/{96,192,336,720}, 1536/{96,192,336,720}. Richer grid; larger pretraining.
  - `prediction_filter_length` can shorten a checkpoint's horizon (not lengthen).
- **Device**: standard torch; move model and `past_values` to the same device.
- **Normalization**: internal (`scaling="std"`); raw input/output.

---

## Adapter contract (all inputs pre-normalized by `FoundationModel`)

`FoundationModel` does all normalization/validation/trimming, then calls the adapter with
canonical dicts. Registry resolution: `_resolve_adapter` prefix-matches `model_id` against
`_ADAPTER_REGISTRY` (`_adapters.py:3949-3985`); `FoundationModel.__init__`
(`_foundation_model.py:251-252`) does `adapter_cls(model_id=model_id, **kwargs)`.

Required by `FoundationModel`:
- Class attr `allow_exog: bool = False` (used at `_foundation_model.py:954-973` to drop any
  exog to `None` before calling the adapter, with an `IgnoredArgumentWarning`).
- `__init__(self, model_id, *, ...keyword-only...)`.
- `fit(self, context, context_exog) -> self` (both passed by keyword).
- `predict(self, steps, context, context_exog, exog, quantiles) -> dict[str, np.ndarray]`
  (all passed by keyword). Output: `{name: (steps, n_quantiles) array}`; point = `(steps, 1)`.
  Context is already trimmed to the last `context_length` observations upstream (it is NOT
  padded upstream, so the adapter must left-pad short series).
- `get_params()` returns `model_id` + every ctor kwarg (empty dict => `None`).
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
en/em dashes):

```python
class TinyTimeMixerAdapter:
    """
    Adapter for IBM Granite TinyTimeMixer (TTM) foundation models.

    TinyTimeMixer is a compact, pre-trained time series model. Each released
    checkpoint is tailored to a fixed context length and a fixed forecast
    length, and produces point forecasts (zero-shot checkpoints have no active
    quantile head and do not use exogenous variables; those are fine-tuning
    features). Inputs are standardized internally (instance normalization), so
    raw values can be passed directly.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "ibm-granite/granite-timeseries-ttm-r2".
    model : object, default None
        Pre-loaded `TinyTimeMixerForPrediction` instance. If `None`, the model
        is loaded lazily on the first call to `predict`. Intended for testing.
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
    get_model_kwargs : dict, default None
        Additional keyword arguments forwarded verbatim to
        `tsfm_public.toolkit.get_model.get_model` (e.g. `freq_prefix_tuning`,
        `prefer_l1_loss`, `force_return`, `model_revision`). Do not include
        `context_length` or `prediction_length` here; those are controlled by
        the corresponding adapter parameters.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Not used, present here for API consistency by convention.
    context_length : int
        Number of historical observations used as context.
    prediction_length : int
        Fixed forecast length of the checkpoint.
    device : str
        Device placement for the model.
    torch_dtype : object
        Torch dtype for the model and inputs.
    get_model_kwargs : dict
        Additional keyword arguments forwarded to `get_model`.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    Zero-shot TinyTimeMixer checkpoints (r1, r2) are point forecast only: there
    is no active quantile head, so requesting `quantiles` (via
    `predict_interval` or `predict_quantiles`) raises a `ValueError`. Exogenous
    variables are not supported (`allow_exog = False`); any `exog` or
    `context_exog` is dropped upstream by `FoundationModel`.

    Both `context_length` and `prediction_length` are fixed per checkpoint;
    `get_model` resolves the matching HuggingFace revision for the requested
    combination.

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
        context_length: int = 512,
        prediction_length: int = 96,
        device: str = "auto",
        torch_dtype: Any | None = None,
        get_model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. "ibm-granite/granite-timeseries-ttm-r2".
        model : object, default None
            Pre-loaded `TinyTimeMixerForPrediction` instance. If `None`, the
            model is loaded lazily on the first call to `predict`.
        context_length : int, default 512
            Number of historical observations used as context. Must be a
            positive integer supported by the checkpoint.
        prediction_length : int, default 96
            Fixed forecast length of the checkpoint. Must be a positive integer.
        device : str, default 'auto'
            Device placement. `"auto"` selects CUDA > MPS > CPU.
        torch_dtype : object, default None
            Torch dtype the model and inputs are cast to.
        get_model_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to `get_model`.

        """

        _validate_positive_int("context_length", context_length)
        _validate_positive_int("prediction_length", prediction_length)

        self.model_id          = model_id
        self._model            = model
        self.context_          = None
        self.context_exog_     = None
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.device            = device
        self.torch_dtype       = torch_dtype
        self.get_model_kwargs  = get_model_kwargs or {}
        self.is_fitted         = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `prediction_length`, `device`,
            `torch_dtype`, `get_model_kwargs`. `get_model_kwargs` is returned as
            `None` when no additional config was set.

        """
        return {
            "model_id":          self.model_id,
            "context_length":    self.context_length,
            "prediction_length": self.prediction_length,
            "device":            self.device,
            "torch_dtype":       self.torch_dtype,
            "get_model_kwargs":  self.get_model_kwargs or None,
        }

    def set_params(self, **params) -> TinyTimeMixerAdapter:
        """
        Set adapter parameters. Resets the model when a parameter baked into the
        loaded model changes (`model_id`, `context_length`, `prediction_length`,
        `device`, `torch_dtype`, `get_model_kwargs`).

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `prediction_length`,
            `device`, `torch_dtype`, `get_model_kwargs`.

        Returns
        -------
        self : TinyTimeMixerAdapter

        """

        def validate(p: dict) -> dict:
            if "context_length" in p:
                _validate_positive_int("context_length", p["context_length"])
            if "prediction_length" in p:
                _validate_positive_int("prediction_length", p["prediction_length"])
            if "get_model_kwargs" in p:
                p["get_model_kwargs"] = p["get_model_kwargs"] or {}
            return p

        return _apply_set_params(
            self, params,
            validate=validate,
            resets=(
                (
                    {"model_id", "context_length", "prediction_length",
                     "device", "torch_dtype", "get_model_kwargs"},
                    lambda: setattr(self, "_model", None),
                ),
            ),
        )

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: Any,
    ) -> TinyTimeMixerAdapter:
        """
        Store the training series.
        No model training occurs since TinyTimeMixer is a zero-shot inference
        model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : Any
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : TinyTimeMixerAdapter

        """

        self.context_ = context
        self.is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: Any,
        exog: Any,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate point predictions using TinyTimeMixer.

        All input normalization, validation, and context trimming is performed
        upstream by `FoundationModel`; this method receives pre-processed dicts
        only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast. Must be `<= prediction_length`.
        context : dict pandas Series
            Per-series context windows (already trimmed to `context_length`).
        context_exog : Any
            Not used, present here for API consistency by convention.
        exog : Any
            Not used, present here for API consistency by convention.
        quantiles : list of float or None
            Must be `None`. Zero-shot TinyTimeMixer is point forecast only.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, 1)`.

        Raises
        ------
        ValueError
            If `quantiles` is not `None`, or if `steps > prediction_length`.

        """

        if quantiles is not None:
            raise ValueError(
                "TinyTimeMixerAdapter produces point forecasts only. Zero-shot "
                "TinyTimeMixer checkpoints (r1, r2) have no active quantile "
                "head, so prediction intervals and quantile forecasts are not "
                "supported. Call `predict` without `quantiles` (do not use "
                "`predict_interval` or `predict_quantiles`)."
            )

        if steps > self.prediction_length:
            raise ValueError(
                f"`steps` ({steps}) exceeds `prediction_length` "
                f"({self.prediction_length}). TinyTimeMixer uses a fixed "
                f"forecast length per checkpoint; request "
                f"`steps <= prediction_length` or load a checkpoint with a "
                f"longer `prediction_length`."
            )

        self._load_model()

        import torch

        series_names_in = list(context.keys())
        n_series = len(series_names_in)
        ctx_len = self.context_length

        past_values = np.zeros((n_series, ctx_len, 1), dtype=np.float32)
        past_observed_mask = np.zeros((n_series, ctx_len, 1), dtype=np.float32)
        for i, name in enumerate(series_names_in):
            values = context[name].to_numpy(dtype=np.float32)
            length = min(values.shape[0], ctx_len)
            past_values[i, ctx_len - length:, 0] = values[-length:]
            past_observed_mask[i, ctx_len - length:, 0] = 1.0

        device = _resolve_torch_device(self.device)
        x = torch.as_tensor(past_values, device=device)
        mask = torch.as_tensor(past_observed_mask, device=device)
        if self.torch_dtype is not None:
            x = x.to(self.torch_dtype)
            mask = mask.to(self.torch_dtype)

        with torch.no_grad():
            output = self._model(past_values=x, past_observed_mask=mask)

        # prediction_outputs: (n_series, prediction_length, n_channels)
        preds = _tensor_to_numpy(output.prediction_outputs)

        predictions: dict[str, np.ndarray] = {}
        for i, name in enumerate(series_names_in):
            predictions[name] = preds[i, :steps, 0].reshape(-1, 1)

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
        The model is loaded lazily via `tsfm_public.toolkit.get_model.get_model`,
        which resolves the HuggingFace revision matching the requested
        `context_length` / `prediction_length`, then moved to the resolved
        device (and optional `torch_dtype`) and switched to eval mode. This
        method is a no-op when `self._model` is already populated.

        """

        if self._model is not None:
            return
        try:
            from tsfm_public.toolkit.get_model import get_model
        except ImportError as exc:
            raise ImportError(
                "granite-tsfm is required for TinyTimeMixerAdapter. "
                "Install it with `pip install granite-tsfm`."
            ) from exc

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
```

## File 1 (cont.): register in `_ADAPTER_REGISTRY` (`_adapters.py:3949-3959`)

Add one entry (prefix match covers r1 `...-ttm-r1` and r2 `...-ttm-r2`):

```python
    "ibm-granite/granite-timeseries-ttm": TinyTimeMixerAdapter,
```

---

## File 2: `skforecast/foundation/_foundation_model.py` -- docstring only

Add TTM to the three enumerations in the `FoundationModel` docstring:
- "Available model IDs" list (~lines 40-74): add
  `ibm-granite/granite-timeseries-ttm-r2`, `ibm-granite/granite-timeseries-ttm-r1`.
- "Commonly used kwargs by model" (~83-115): add TTM line noting `context_length`,
  `prediction_length`, `device`, `torch_dtype`, `get_model_kwargs`.
- "References" (~193-240): add granite-tsfm GitHub + HF r1/r2 entries.

No change to `__init__.py` (adapters are not exported there).

---

## File 3: Tests -- `skforecast/foundation/tests/tests_foundation_models/`

Follow `.github/instructions/testing.instructions.md` and the `test_MoiraiAdapter.py` /
`test_TimesFMAdapter.py` template. NO `importorskip` / `monkeypatch`; inject a hand-written
fake via the `model` ctor arg so the lazy loader is a no-op. `torch` IS available in the
`test` extra (`pyproject.toml:117`), so building the input tensor in `predict` works.

### 3a. Add to `fixtures_adapters.py`

```python
class FakeTinyTimeMixerOutput:
    """Stand-in for TinyTimeMixerForPredictionOutput."""
    def __init__(self, prediction_outputs):
        self.prediction_outputs = prediction_outputs


class FakeTinyTimeMixer:
    """
    Minimal stand-in for `TinyTimeMixerForPrediction` used in tests. Records the
    tensors it is called with and returns a deterministic point forecast: the
    last observed context value repeated across the (fixed) prediction length.
    """
    def __init__(self, prediction_length=96):
        self.prediction_length = prediction_length
        self.last_past_values = None
        self.last_past_observed_mask = None

    def eval(self):
        return self

    def to(self, arg):  # accepts a device string or a torch dtype
        return self

    def __call__(self, past_values, past_observed_mask=None):
        self.last_past_values = past_values
        self.last_past_observed_mask = past_observed_mask
        arr = np.asarray(past_values)                      # (n, ctx, 1)
        last_value = arr[:, -1, :]                         # (n, 1)
        preds = np.repeat(
            last_value[:, None, :], self.prediction_length, axis=1
        ).astype(np.float32)                               # (n, pred_len, 1)
        return FakeTinyTimeMixerOutput(preds)
```

Note: `past_values` reaching the fake is a torch tensor; `np.asarray(torch_tensor_on_cpu)`
works. If a device dtype makes `np.asarray` fail in CI, cast via `.cpu().numpy()` inside the
fake. Reuse existing shared fixtures/builders in this file (`y`, `y_wide`, `y_dict`,
`prepare_fit_args`, `prepare_predict_args`).

### 3b. New file `test_TinyTimeMixerAdapter.py`

Header `# Unit test TinyTimeMixerAdapter`, imports in house order, then a local injector:

```python
def make_adapter(**kwargs) -> TinyTimeMixerAdapter:
    prediction_length = kwargs.get("prediction_length", 96)
    adapter = TinyTimeMixerAdapter(
        model_id=kwargs.pop("model_id", "ibm-granite/granite-timeseries-ttm-r2"),
        **kwargs,
    )
    adapter._model = FakeTinyTimeMixer(prediction_length=prediction_length)
    return adapter
```

Tests (grouped with `# ====` banners, multi-line docstrings, parametrized, aligned):
1. **init**: `test_init_default_params` asserts `context_length==512`,
   `prediction_length==96`, `device=="auto"`, `torch_dtype is None`,
   `get_model_kwargs=={}`, `_model is None`, `context_ is None`, `is_fitted is False`,
   `allow_exog is False`. `test_init_ValueError_when_context_length_invalid` and
   `..._prediction_length_invalid` parametrized `[0, -1, None]`.
2. **get_params/set_params**: exact key set; `set_params` updates + resets `_model` to
   `None`; unknown param -> "Invalid parameter"; no reset when value unchanged;
   invalid values raise.
3. **fit**: `test_fit_output_single_series` (asserts `is_fitted`, stored context,
   does-not-modify-input via `assert_series_equal(y, y_copy)`);
   `test_fit_output_multi_series` parametrized `[y_wide, y_dict]`.
4. **predict**:
   - `test_predict_ValueError_when_quantiles_requested` (match the point-only message).
   - `test_predict_ValueError_when_steps_exceed_prediction_length`
     (e.g. `prediction_length=96`, `steps=200`).
   - `test_predict_output_single_series` and `..._multi_series`: assert dict keys, each
     value shape `(steps, 1)`, values via `np.testing.assert_array_almost_equal` against
     the fake's deterministic output (last context value repeated, sliced to `steps`).
   - `test_predict_left_pads_short_context`: context shorter than `context_length` =>
     inspect `adapter._model.last_past_values` shape `(n, context_length, 1)` and
     `last_past_observed_mask` (zeros in the left pad region, ones elsewhere).
   Build predict args with `prepare_predict_args` / `prepare_fit_args`.

### 3c. Update shared adapter-enumeration tests
- `test_resolve_adapter.py` (3 spots): import `TinyTimeMixerAdapter`; add
  `("ibm-granite/granite-timeseries-ttm-r2", TinyTimeMixerAdapter)` (and an r1 case) to the
  `model_id -> cls` parametrization; add the registry entry to the re-declared expected dict.
- `test_clone.py`: add a TTM `model_id` row (+ `ids`) and a `get_model_kwargs`-style config row.

---

## File 4: Docs -- add a TTM row/entry everywhere the adapter list is enumerated

Hand-edited sources (identical adapter table in three files):
- `AGENTS.md`: table (~512-521), architecture class-name line (~107), backend pip list (~484).
- `.github/copilot-instructions.md`: mirror (~467-476, ~62, ~439).
- `tools/ai/llms-base.txt`: mirror (~467-476, ~62, ~439).

New table row (AGENTS.md column format):
```
| TinyTimeMixerAdapter (IBM) | `ibm-granite/granite-timeseries-ttm` | No | 512 | Point forecast only |
```
Architecture/class-name line: append `TinyTimeMixerAdapter`. Backend pip list: append
`granite-tsfm`.

Skill docs:
- `skills/foundation-forecasting/SKILL.md`: front-matter model list, Installation
  (`pip install granite-tsfm`), "Choosing a Model" table (note point-only, no exog, fixed
  context/horizon), quantile-support paragraph (TTM = point only), references.
- `skills/foundation-forecasting/references/adapter-parameters.md`: contents list; a new
  `## TinyTimeMixerAdapter` section (prefix `ibm-granite/granite-timeseries-ttm`,
  `allow_exog=False`, point-only, param table: `context_length`, `prediction_length`,
  `device`, `torch_dtype`, `get_model_kwargs`); add a row to the
  "Tunable parameters and model reload cost" table.

Generated (do NOT hand-edit; regenerate with the `tools/ai/` build script after editing the
sources above): `docs/llms-full.txt`, `site/llms*.txt`, root `llms*.txt`.

`CLAUDE.md` (skills list only) and `README.md` (no adapter table) need no per-adapter edit.

---

## File 5: `pyproject.toml` -- NO change

Foundation backends are intentionally not declared as extras (lazy-imported). Document the
manual `pip install granite-tsfm` in the adapter `ImportError` and SKILL.md, matching the
existing convention. Tests inject a fake, so `granite-tsfm` is not needed in the `test` extra.

---

## Verification

1. Confirm conda env (memory: `skforecast_24_py13`) before running Python; verify with
   `where.exe python`.
2. `pytest skforecast/foundation/tests/tests_foundation_models/test_TinyTimeMixerAdapter.py -vv`
   plus the updated `test_resolve_adapter.py` and `test_clone.py` -- all green, no real
   backend imported.
3. Registry sanity:
   `FoundationModel(model_id="ibm-granite/granite-timeseries-ttm-r2").adapter` is a
   `TinyTimeMixerAdapter`.
4. Point-only guards: `ForecasterFoundation` `.predict_interval(...)` /
   `.predict_quantiles(...)` raise the clear `ValueError`; `.predict(steps=...)` returns a
   point forecast; `steps > prediction_length` raises.
5. Optional real smoke test (throwaway env with `granite-tsfm`; note torch>=2.10 may clash
   with the pinned env): load r2 512/96, forecast a demo series, verify `steps<=96` slicing
   and short-context left-padding.
6. `ruff check` clean (line length 88, double quotes, no en/em dashes in comments/docstrings).

## Open implementation checks (resolve during coding)
- Confirm `TinyTimeMixerForPrediction.forward` accepts `past_observed_mask` on the pinned
  `granite-tsfm` version and that the output field is `.prediction_outputs`
  (search `class TinyTimeMixerForPredictionOutput` in `modeling_tinytimemixer.py`).
- Confirm `get_model` accepts `context_length`/`prediction_length` as shown and resolves
  the r2 `main` (512/96) checkpoint without extra kwargs.
