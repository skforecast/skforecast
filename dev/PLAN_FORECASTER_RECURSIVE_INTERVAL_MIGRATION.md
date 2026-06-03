# Plan: Adapt `ForecasterRecursive` to the Unified Interval/Quantile API

> Scope: **`ForecasterRecursive` only** (`skforecast/recursive/_forecaster_recursive.py`).
> This is the pilot implementation. Once validated here, the same pattern is rolled
> out to the other forecasters in a separate effort.
>
> Goal: move `predict_interval` from the **0–100 percentile** scale to the unified
> **0–1 quantile** scale, with a backward-compatible deprecation path, without
> changing default behavior silently.

---

## 1. Current state (grounded in code)

### `predict_interval` (line ~2245)

```python
def predict_interval(
    self,
    steps,
    last_window=None,
    exog=None,
    method='bootstrapping',
    interval=[5, 95],          # ← 0–100 percentiles, default = 90% coverage
    n_boot=250,
    ...
):
```

Internal logic:

```python
# bootstrapping branch
if isinstance(interval, (list, tuple)):
    check_interval(interval=interval, ensure_symmetric_intervals=False)
    interval = np.array(interval) / 100          # ← /100 conversion
else:
    check_interval(alpha=interval, alpha_literal='interval')
    interval = np.array([0.5 - interval / 2, 0.5 + interval / 2])
...
predictions_interval = boot_predictions.quantile(q=interval, axis=1).transpose()
predictions_interval.columns = ['lower_bound', 'upper_bound']

# conformal branch
if isinstance(interval, (list, tuple)):
    check_interval(interval=interval, ensure_symmetric_intervals=True)
    nominal_coverage = (interval[1] - interval[0]) / 100    # ← /100 conversion
else:
    check_interval(alpha=interval, alpha_literal='interval')
    nominal_coverage = interval
```

### `predict_quantiles` (line ~2393)

Already on the 0–1 scale. Output columns `q_{q}`. Only the internal validation
call (`check_interval(quantiles=...)`) is shared.

```python
check_interval(quantiles=quantiles)
...
predictions.columns = [f'q_{q}' for q in quantiles]
```

### `check_interval` (`utils/utils.py`, line ~1023)

The `interval=` branch validates the **0–100** range:
- bounds must be `>= 0 and < 100` (lower), `> 0 and <= 100` (upper)
- optional symmetry check sums to `100`.

### Key facts confirmed

1. The actual default is `interval=[5, 95]` → **90% coverage** (the docstring example
   `0.95 → [2.5, 97.5]` refers to the float-coverage path, not the default).
2. `float` coverage path already uses 0–1 — **no change needed** there.
3. Only **two** internal places do the `/100` conversion (bootstrapping + conformal).
4. `predict_quantiles` is already correct; only its shared `check_interval` call is affected.

---

## 2. Target behavior

| Input | Interpretation | Output |
|-------|----------------|--------|
| `interval=[0.05, 0.95]` | quantiles 0–1 (new) | `pred, lower_bound, upper_bound` |
| `interval=0.90` (float) | coverage 0–1 (unchanged) | `pred, lower_bound, upper_bound` |
| `interval=[5, 95]` | legacy percentiles → `/100` + `FutureWarning` | `pred, lower_bound, upper_bound` |
| `interval=[1, 50]` (mixed ≤1 and >1) | ambiguous | `ValueError` |
| `interval=1.0` (float) | 100% coverage | `ValueError` |
| `interval=[0, 1]` | valid quantiles | works (edge: full range) |

**Default stays at 90% coverage**: change default param from `[5, 95]` to `[0.05, 0.95]`.

---

## 3. Implementation tasks

### 3.1 New shared helper `_normalize_interval_scale` in `utils/utils.py`

Single source of truth for scale detection. Returns quantiles in 0–1 and emits the
legacy `FutureWarning` / raises `ValueError` on mixed input.

```python
def _normalize_interval_scale(interval):
    """
    Normalize a 2-value interval to the 0–1 quantile scale.

    - all values in [0, 1]      -> returned unchanged (new API)
    - all values in (1, 100]    -> divided by 100 + FutureWarning (legacy)
    - mixed (<=1 and >1)        -> ValueError (ambiguous)

    Returns a list[float] of two quantiles in 0–1.
    """
```

Detection rule for the value exactly `1`: treated as a valid quantile (upper bound),
**unless** it appears together with another value `> 1` (then it is "mixed" → error).

**Estimate: 2.0 h** (logic + messages + unit tests for the helper).

### 3.2 Update `check_interval` (`utils/utils.py`)

Re-point the `interval=` branch to validate the **0–1** range, delegating scale
detection to `_normalize_interval_scale`. Keep `quantiles=` and `alpha=` branches
unchanged. Preserve `ensure_symmetric_intervals` (now expressed on 0–1: bounds sum to 1).

**Estimate: 1.5 h** (refactor branch + adjust symmetry math + update docstring).

### 3.3 Update `ForecasterRecursive.predict_interval`

1. Change default: `interval=[5, 95]` → `interval=[0.05, 0.95]`.
2. Bootstrapping branch: replace `np.array(interval) / 100` with
   `interval = _normalize_interval_scale(interval)`.
3. Conformal branch: replace `(interval[1] - interval[0]) / 100` with
   normalized quantiles, then `nominal_coverage = q[1] - q[0]`.
4. Float path: add explicit `ValueError` for `interval == 1.0`.
5. Update docstring (params + `.. versionchanged::` tag per repo convention).

**Estimate: 2.5 h** (code + docstring rewrite + version tag).

### 3.4 `predict_quantiles`

No public API change. Verify the shared `check_interval(quantiles=...)` path is
unaffected after 3.2. Add/adjust a docstring note clarifying why it returns
`q_*` columns while `predict_interval` returns `lower_bound/upper_bound`.

**Estimate: 0.5 h.**

---

## 4. Tests

Location: `skforecast/recursive/tests/` (per `testing.instructions.md`).

| Test area | What to cover | Est. |
|-----------|---------------|------|
| `test_check_interval.py` | new 0–1 validation, legacy `>1` warning, mixed → `ValueError`, symmetry on 0–1 | 1.5 h |
| `_normalize_interval_scale` unit tests | all rows of the target table incl. edge `1` and `[0,1]` | 1.0 h |
| `test_predict_interval.py` | new scale, conformal + bootstrapping, `ValueError(1.0)`, mixed | 2.0 h |
| Regression guards | `predict_interval([5,95])` (with warning) == old fixed values; `[5,95]` == `[0.05,0.95]`; default `predict_interval()` == prior 90% output | 2.0 h |
| `pytest.warns(FutureWarning)` | one per branch (bootstrapping + conformal) | 0.5 h |
| `predict_quantiles` unchanged | snapshot still passes | 0.5 h |

**Tests subtotal: 7.5 h.**

> The "regression guards" are the most important: they prove the `/100` move and the
> default change preserve numbers exactly. Compare against hard-coded expected arrays.

---

## 5. Documentation

- Docstrings: `predict_interval`, `check_interval` (+ `.. versionchanged::` tags).
- `predict_quantiles` clarifying note.
- `changelog.md` entry: scale change, default stays 90% coverage, deprecation notice.
- User-guide / `llms-*.txt` updates are **out of scope** for this `ForecasterRecursive`-only
  pass (done globally once all forecasters are migrated).

**Estimate: 1.5 h.**

---

## 6. Out of scope (explicitly)

- Other forecasters (`Direct`, `MultiSeries`, `Classifier`, `RNN`, `Foundation`).
- `backtesting_forecaster` routing and `p_` → `q_` column renaming.
- `ConformalIntervalCalibrator`, plotting, metrics consumers.
- Removal of percentile support (that is the N+2 release).

Note: `_normalize_interval_scale` and the `check_interval` refactor are written so the
remaining forecasters can reuse them with no further changes.

---

## 7. Effort summary (senior developer)

| Block | Task | Hours |
|-------|------|------:|
| Impl | `_normalize_interval_scale` helper | 2.0 |
| Impl | `check_interval` refactor | 1.5 |
| Impl | `predict_interval` adaptation | 2.5 |
| Impl | `predict_quantiles` verification + note | 0.5 |
| Tests | unit + interval + regression + warnings | 7.5 |
| Docs | docstrings + changelog | 1.5 |
| | **Subtotal** | **15.5** |
| | Contingency / review / CI iteration (~20%) | 3.0 |
| | **Total** | **~18.5 h** |

**Estimated effort: ~18–19 hours** (roughly 2.5 working days) for a senior developer,
including tests and review iteration but excluding cross-forecaster rollout.

---

## 8. Risk notes

1. **Conformal symmetry check** currently sums to `100`; on the 0–1 scale it must sum
   to `1`. Easy to miss — covered by a dedicated test.
2. **Edge value `1`**: `[0, 1]` valid, `[1, 50]` mixed/error. Detection rule must be
   exact; the helper unit tests pin it down.
3. **Default change**: only the *representation* changes (`[5,95]`→`[0.05,0.95]`), not
   the coverage (90%). Regression guard asserts identical numeric output.
4. **Shared `check_interval`**: used by many call sites beyond `ForecasterRecursive`.
   Refactor must stay backward compatible for callers still passing percentiles until
   their own migration — the `FutureWarning` path guarantees that.
