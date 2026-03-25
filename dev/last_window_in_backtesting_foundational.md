# Role of `last_window` in `backtesting_foundational`

## Overview

`last_window` is the **context slice of the series** passed to `ForecasterFoundational.predict()` on each backtesting fold. It replaces the internal history stored by `fit()`, giving the model the correct historical context anchored to each fold's test period.

For a zero-shot model like Chronos-2, `fit()` only stores history — it does not train. **`last_window` is the sole input the model uses to generate predictions**, making its size and content critical to understand.

---

## How `last_window` is constructed per fold

### Step 1 — `window_size` is set to `context_length`

In `_backtesting_foundational`, the cross-validator is configured:

```python
cv.set_params({'window_size': forecaster.window_size, ...})
```

And `forecaster.window_size` is:

```python
self.window_size = (
    estimator.adapter.context_length   # e.g. 2048
    if estimator.adapter.context_length is not None
    else 1
)
```

So `cv.window_size = context_length`.

### Step 2 — `TimeSeriesFold.split()` computes the window indices

For each fold `i`:

```
test_iloc_start       = initial_train_size + i * fold_stride  (when refit=False)
last_window_iloc_start = test_iloc_start - window_size        (= test_start - context_length)
last_window_iloc_end   = test_iloc_start
```

The window is always the `context_length` observations **immediately before** the test period.

### Step 3 — The window is sliced from the full series

```python
window_start, window_end = fold[2]  # iloc positions
last_window = series.iloc[window_start : window_end]
```

### Step 4 — The adapter trims again as a safety net

Inside `Chronos2Adapter.predict()`:

```python
if last_window is not None and self.context_length is not None:
    history = history.iloc[-self.context_length:]  # no-op if len <= context_length
```

This is redundant once the window is correctly sized by the splitter, but guards against edge cases.

---

## How `last_window` size evolves across folds

The key formula is:

```
last_window_size(i) = min(test_iloc_start(i), context_length)
                    = min(initial_train_size + i * fold_stride, context_length)
```

Negative `iloc` indices in pandas are clipped to `0`, so when
`test_iloc_start - context_length < 0`, the window starts at index 0 and is
**shorter than `context_length`**.

There are two regimes:

| Condition | Behaviour |
|---|---|
| `initial_train_size + i * fold_stride < context_length` | **Expanding** — window grows each fold |
| `initial_train_size + i * fold_stride >= context_length` | **Fixed** — window is exactly `context_length`, slides forward |

The transition fold index is:

```
i_transition = ceil((context_length - initial_train_size) / fold_stride)
```

---

## Scenarios

### Scenario 1 — Series shorter than `context_length` (most common with Chronos-2 defaults)

**Setup**: `context_length=2048`, `n_obs=1000`, `initial_train_size=100`, `steps=50`, `fold_stride=50`

Since the full series (1000 obs) is shorter than `context_length` (2048), the window is
**always expanding** and never reaches the cap.

```
transition fold = ceil((2048 - 100) / 50) = ceil(38.96) = 39
```

Fold 39 would need `test_iloc_start = 100 + 39*50 = 2050 > 1000`, so it doesn't exist.
The cap is **never reached** — all 18 folds behave as expanding.

| Fold | `test_iloc_start` | `window_start` (clipped) | `last_window` size |
|------|-------------------|--------------------------|--------------------|
| 0    | 100               | 0                        | **100**            |
| 1    | 150               | 0                        | **150**            |
| 2    | 200               | 0                        | **200**            |
| 5    | 350               | 0                        | **350**            |
| 10   | 600               | 0                        | **600**            |
| 17   | 950               | 0                        | **950**            |

**Pattern**: grows by `fold_stride` (50) each fold, always anchored at the
beginning of the series.

---

### Scenario 2 — Series longer than `context_length`

**Setup**: `context_length=512`, `n_obs=5000`, `initial_train_size=100`, `steps=50`, `fold_stride=50`

```
transition fold = ceil((512 - 100) / 50) = ceil(8.24) = 9
```

From fold 9 onward, the window is fixed at 512 and slides forward.

| Fold | `test_iloc_start` | `window_start` (raw) | Clipped? | `last_window` size |
|------|-------------------|----------------------|----------|--------------------|
| 0    | 100               | −412                 | → 0      | **100** (expanding)|
| 1    | 150               | −362                 | → 0      | **150** (expanding)|
| 5    | 350               | −162                 | → 0      | **350** (expanding)|
| 8    | 500               | −12                  | → 0      | **500** (expanding)|
| 9    | 550               | 38                   | no       | **512** (capped)   |
| 10   | 600               | 88                   | no       | **512** (sliding)  |
| 50   | 2600              | 2088                 | no       | **512** (sliding)  |
| 97   | 4950              | 4438                 | no       | **512** (sliding)  |

**Pattern**: expands for the first 9 folds, then **fixed at 512 observations**,
sliding forward by 50 positions per fold.

---

### Scenario 3 — Large `initial_train_size` (window capped from fold 0)

**Setup**: `context_length=512`, `n_obs=5000`, `initial_train_size=1000`, `steps=50`, `fold_stride=50`

```
transition fold = ceil((512 - 1000) / 50) = ceil(-9.76) = -9  → already past from fold 0
```

Since `initial_train_size > context_length`, the window is **immediately capped** at
`context_length` in every fold.

| Fold | `test_iloc_start` | `window_start` | `last_window` size |
|------|-------------------|----------------|--------------------|
| 0    | 1000              | 488            | **512**            |
| 1    | 1050              | 538            | **512**            |
| 10   | 1500              | 988            | **512**            |

**Pattern**: always exactly 512 observations, sliding from the very first fold.

---

### Scenario 4 — Effect of `refit`

`refit` controls whether `forecaster.fit()` is called again each fold. It has **no effect
on `last_window`** because the window is always derived from `test_iloc_start`, which is
the same regardless of `refit`:

```
test_iloc_start = initial_train_size + i * fold_stride   ← same for all refit values
```

What changes with `refit` is only the **training slice** passed to `fit()`:

| `refit` | `fixed_train_size` | Training slice at fold `i` | Effect on `last_window` |
|---|---|---|---|
| `False` | any | `[0, initial_train_size)` — fits once | **None** |
| `True` | `True` (default) | `[i*fold_stride, initial_train_size + i*fold_stride)` — slides | **None** |
| `True` | `False` | `[0, initial_train_size + i*fold_stride)` — expands | **None** |

> **Key insight for zero-shot models**: since `last_window` is always explicitly passed
> to `predict()`, it overrides the `_history` stored by `fit()`. For Chronos-2, refitting
> only updates `_history`, which is then immediately discarded. `refit=True` adds
> computational overhead without affecting predictions.

---

### Scenario 5 — `fold_stride < steps` (overlapping folds)

**Setup**: `context_length=512`, `n_obs=5000`, `initial_train_size=500`,
`steps=50`, `fold_stride=10`

The test sets overlap by 40 observations per fold. `last_window` still ends exactly
at `test_iloc_start`, so consecutive windows overlap by `steps - fold_stride = 40` obs:

| Fold | `test_iloc_start` | `last_window` covers          | `last_window` size |
|------|-------------------|-------------------------------|--------------------|
| 0    | 500               | `[0, 500)`   (capped from 0)  | **500** (expanding)|
| 1    | 510               | `[0, 510)`   (capped from 0)  | **510** (expanding)|
| 2    | 520               | `[0, 520)`   (capped from 0)  | **520** (expanding)|
| 2    | 520               | `[8, 520)`                    | **512** (capped)   |
| 10   | 600               | `[88, 600)`                   | **512** (sliding)  |

Windows slide by only 10 observations per fold. This means 40 out of every 50 context
observations are shared between consecutive folds — the model sees nearly identical
context, which is likely wasteful.

---

### Scenario 6 — `gap > 0` (forecast gap between training and test)

**Setup**: `context_length=512`, `n_obs=5000`, `initial_train_size=500`,
`steps=50`, `gap=10`

With a gap, the test period starts `gap` steps after the training set ends. The fold
structure becomes:

```
[train_start, train_end] ... gap ... [test_start, test_end]
                                      ← steps=50 →
```

`last_window` still ends at **`test_iloc_start`** (the start of the gap+test block),
not at the actual test start. This is correct — the history must include everything
up to and including the gap, so the model can project `steps + gap` steps ahead and
then the gap rows are discarded.

| Fold | `test_iloc_start` | `last_window` covers | Actual test period |
|------|-------------------|----------------------|--------------------|
| 0    | 500               | `[0, 500)`           | `[510, 560)` (after gap=10) |
| 1    | 550               | `[38, 550)`          | `[560, 610)` |

---

## Summary table

| Parameter | Effect on `last_window` |
|---|---|
| `context_length` | Sets the **maximum** window size |
| `initial_train_size` | Determines the window size in fold 0 |
| `fold_stride` | Controls how much the window slides (or grows) per fold |
| `n_obs` vs `context_length` | If series < cap → always expanding; if series > cap → transitions to sliding |
| `refit` | **No effect** on `last_window` |
| `fixed_train_size` | **No effect** on `last_window` |
| `gap` | `last_window` ends at `test_start - gap`, not at test period start |

### Window behaviour at a glance

```
Series length < context_length:
  fold 0:  [========]                         (initial_train_size obs, expanding)
  fold 1:  [===========]                      (+fold_stride)
  fold 2:  [==============]                   (+fold_stride)
  ...      always anchored at index 0 → expanding window

Series length > context_length:
  fold 0:  [========]                         (initial_train_size obs, expanding)
  ...
  fold k:  [=========================]        (context_length obs, cap reached)
  fold k+1:  [=========================]      (slides forward by fold_stride)
  fold k+2:    [=========================]    (slides forward by fold_stride)
  ...      fixed size, sliding window
```
