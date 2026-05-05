# Refactoring: `skforecast/preprocessing/_calendar.py`

**Branch:** `feature_calendar_features`
**Date:** 2026-05-04
**Scope:** Bug fixes, validation, and clarity improvements to the calendar/datetime feature module.
**Tests:** 208/208 preprocessing tests pass.
**Diff:** 4 files changed, +214 / −75 lines.

---

## Files changed

| File | Purpose |
|---|---|
| `skforecast/preprocessing/_calendar.py` | Core module — bug fixes, validation, docstrings |
| `skforecast/preprocessing/tests/tests_preprocessing/test_create_datetime_features.py` | New tests for week 53, day-of-year 366, spline continuity, distinctness |
| `skforecast/preprocessing/tests/tests_preprocessing/test_DateTimeFeatureTransformer.py` | Regenerated spline snapshot |
| `skforecast/preprocessing/tests/tests_preprocessing/fixtures_preprocessing.py` | Bumped `day_of_year_cols_onehot` to range(1, 367) |

---

## Summary of changes

The original module had three categories of issues:

1. **Silent failure modes** — features that should be encoded weren't, leap-year/ISO-week-53 dates produced NaN onehot rows, unknown `spline_kwargs` keys were silently dropped, unnamed Series produced columns named `0`.
2. **A real correctness bug in periodic spline encoding** — December collapsed onto January, week 53 onto week 1, day 366 onto day 1, because the knot placement made the period equal to `max_val − min_val` instead of `max_val`.
3. **Polish issues** — duplicated validation, unhelpful warning text, fragile `fill_na` handling.

All three were addressed.

---

## Phase 1 — Validations & warnings

### 1.0 Lifted `_FEATURE_KNOWN_CATEGORIES` to module scope

**Why:** the categories table was defined inside the `onehot` branch but is also needed for the new `features_to_encode` warning logic. Single source of truth.

**Where:** [_calendar.py:20-30](skforecast/preprocessing/_calendar.py#L20-L30)

```python
_FEATURE_KNOWN_CATEGORIES = {
    "month": list(range(1, 13)),
    "week": list(range(1, 54)),
    "day_of_week": list(range(0, 7)),
    "day_of_month": list(range(1, 32)),
    "day_of_year": list(range(1, 367)),
    "hour": list(range(0, 24)),
    "minute": list(range(0, 60)),
    "second": list(range(0, 60)),
    "quarter": list(range(1, 5)),
}
```

### 1.1 Warn when explicitly-listed `features_to_encode` can't be encoded

**Why:** users requesting (e.g.) `features_to_encode=['year']` with `encoding='cyclical'` previously got their year column kept as a raw integer with no feedback.

**Where:** [_calendar.py:166-186](skforecast/preprocessing/_calendar.py#L166-L186)

**Behaviour:**
- Triggers only when `features_to_encode` is **explicitly** passed (not `None`). Default behaviour is unchanged for users who just list features without specifying which to encode.
- Uses `IgnoredArgumentWarning` (existing skforecast warning class).
- Lists the encodable features for the chosen encoding so the user can fix their call.

### 1.2 Validate `spline_kwargs` keys

**Why:** unknown keys (typos like `degrees` or unsupported keys like `knots`/`sparse_output`) were silently dropped or would break downstream concat.

**Where:** [_calendar.py:209-217](skforecast/preprocessing/_calendar.py#L209-L217)

**Decision:** allow **any** sklearn `SplineTransformer` kwarg except:
- `knots` — computed internally from `max_values`; passing it would conflict.
- `sparse_output` — incompatible with the DataFrame return path.

This is **option (B)** from the plan (chosen explicitly by the user during execution). Other kwargs (`degree`, `include_bias`, `extrapolation`, `n_knots`, `order`) are forwarded as-is. Defaults: `degree=3`, `include_bias=True`, `extrapolation='periodic'`.

### 1.3 Removed duplicated encoding validation in `__init__`

**Why:** validation was happening both in `DateTimeFeatureTransformer.__init__` and in `create_datetime_features`. sklearn convention is to validate at fit-time, not construction-time.

**Where:** [_calendar.py:353-368](skforecast/preprocessing/_calendar.py#L353-L368)

**User-visible change:** an invalid `encoding` value now raises at `.fit(...)` instead of at `DateTimeFeatureTransformer(encoding='bad')`. Same error message; just delayed.

### 1.4 Raise on unnamed Series with `keep_original_columns=True`

**Why:** previously, `create_datetime_features(unnamed_series)` would produce an output DataFrame with a column literally named `0`, which silently propagates downstream and is hard to debug.

**Where:** [_calendar.py:269-274](skforecast/preprocessing/_calendar.py#L269-L274)

**Behaviour:** raises `ValueError` with a clear message suggesting either setting `X.name` or using `keep_original_columns=False`.

---

## Phase 2 — Leap years & ISO week 53 (onehot fix)

**Why:** ISO calendar weeks can be 53; leap years have day-of-year 366. The previous defaults (`week=52`, `day_of_year=365`) caused:
- **Onehot**: `pd.Categorical(value, categories=[1..52])` silently turned week-53 values into NaN, producing all-zero rows in the dummy columns.
- **Spline** (with old period = max-min): values out of range mapped to wrap-around equivalents.

**Changes:**

- `default_max_values["week"]`: 52 → 53 ([_calendar.py:125](skforecast/preprocessing/_calendar.py#L125))
- `default_max_values["day_of_year"]`: 365 → 366 ([_calendar.py:128](skforecast/preprocessing/_calendar.py#L128))
- `_FEATURE_KNOWN_CATEGORIES["week"]`: `range(1, 53)` → `range(1, 54)` ([_calendar.py:22](skforecast/preprocessing/_calendar.py#L22))
- `_FEATURE_KNOWN_CATEGORIES["day_of_year"]`: `range(1, 366)` → `range(1, 367)` ([_calendar.py:25](skforecast/preprocessing/_calendar.py#L25))
- Test fixture `day_of_year_cols_onehot`: `range(1, 366)` → `range(1, 367)` (the `week_cols_onehot` fixture was already permissive at 1-53).

**New test:** `test_create_datetime_features_onehot_week_53_and_day_of_year_366` — uses `2020-12-31` (week 53, day-of-year 366) and asserts the corresponding onehot columns light up correctly.

---

## Phase 3 — Polish

### 3.1 Validate `fill_na` type in `calculate_distance_from_holiday`

**Why:** the output columns are `Int64`, so `fillna(0.5)` would raise a confusing `TypeError` deep in pandas. Better to fail fast at the entry point.

**Where:** [_calendar.py:574-580](skforecast/preprocessing/_calendar.py#L574-L580)

**Behaviour:** raises `TypeError` if `fill_na` is not an `int`, `np.integer`, or `numpy.nan`. Booleans are explicitly excluded.

**Default changed:** `fill_na=0.` → `fill_na=0`. Functionally identical for the `Int64` output, but now the default passes its own validator.

### 3.2 Guard `fit()` against empty input

**Why:** `SplineTransformer.fit_transform` on empty data raises a confusing sklearn error.

**Where:** [_calendar.py:387-388](skforecast/preprocessing/_calendar.py#L387-L388)

**Behaviour:** raises `ValueError("Cannot fit on empty input.")` upfront. Added a one-line comment ([_calendar.py:390](skforecast/preprocessing/_calendar.py#L390)) explaining why the `iloc[:2]` slice is safe (encoding is stateless).

**Note:** `transform()` does not have this guard. Calling `transform` on empty data is unusual; if a user does it, they get the underlying error. This is a deliberate non-symmetric choice.

### 3.3 Improved `infer_freq` warning text

**Why:** the previous warning suggested `X.asfreq(...)`, which can reindex and silently drop rows.

**Where:** [_calendar.py:611-618](skforecast/preprocessing/_calendar.py#L611-L618)

**New text:** suggests `X.index.freq = pd.infer_freq(X.index)` or passing an index with a known frequency.

---

## Phase 4 — Spline knot placement (follow-up)

This was a deeper correctness bug discovered while writing tests for Phase 2. Originally scoped out, then added when the user requested the follow-up.

### The bug

```python
# Before
knots = np.linspace(min_val, max_val, n_knots).reshape(-1, 1)
```

For `month` (`min=1, max=12, n_knots=13`):
- `linspace(1, 12, 13)` produces `[1.0, 1.917, 2.833, ..., 12.0]` — **non-integer knots**.
- `knots[-1] - knots[0] = 11`, so the periodic period was **11**.
- Month 12 sits at the rightmost knot; periodic extrapolation makes it identical to month 1.

This collapsed all 1-indexed cyclical features at their boundary:
- December = January (period 11 instead of 12)
- ISO week 53 = week 1 (period 52 instead of 53)
- Day 366 = day 1 (period 365 instead of 366)
- Quarter 4 = Quarter 1 (period 3 instead of 4)

### The fix

**Where:** [_calendar.py:239-246](skforecast/preprocessing/_calendar.py#L239-L246)

```python
# After
n_knots = n_knots_global if n_knots_global is not None else max_val + 1
min_val = default_min_values.get(feature, 0)
# Period = max_val; knots span [min_val, min_val + max_val] so
# that 1-indexed features (e.g. month=12) sit inside the period
# rather than at its boundary, preserving cyclical neighborhood.
knots = np.linspace(min_val, min_val + max_val, n_knots).reshape(-1, 1)
```

For `month` with default `n_knots=13`: knots become `[1, 2, ..., 13]`, period = 12. Month 12 sits at knot 12 (one knot from each of knot 11 and knot 13≡knot 1). Cyclical neighborhood preserved.

For 0-indexed features (`hour`, `minute`, `second`, `day_of_week`), the new formula gives the same result as before — they were already correct. The fix unifies the formula.

### Snapshot regeneration

The existing snapshot test `test_create_datetime_features_spline_encoding_expected_values` was hand-computed against the buggy knot placement. Regenerated using:

```python
knots = np.linspace(1, 13, 13).reshape(-1, 1)
SplineTransformer(degree=3, knots=knots, extrapolation='periodic',
                  include_bias=True).fit_transform([[1],[4],[7],[10]])
```

The new values are much cleaner: months 1, 4, 7, 10 (quarterly spacing) each activate exactly 3 splines as `(1/6, 4/6, 1/6)` — perfectly symmetric. Updated test uses named constants `one_sixth = 1/6` and `four_sixths = 4/6` for clarity.

### New tests

- `test_create_datetime_features_spline_week_53_continuity` — week 53 is equidistant from week 52 and week 1 (one knot step from each), with strictly positive distance to both.
- `test_create_datetime_features_spline_month_12_distinct_from_month_1` — locks in that December and January no longer collapse.

---

## Backward-incompatibility callouts

To include in changelog / release notes:

| Change | Impact | User action |
|---|---|---|
| Spline encoding values change for all features | **HIGH** — numeric output differs | Retrain models; re-fit pipelines |
| Cyclical sin/cos values for `week` shift slightly (max 52→53) | **MEDIUM** — small numeric drift | Re-fit if exact reproduction needed |
| Cyclical sin/cos values for `day_of_year` shift slightly (max 365→366) | **MEDIUM** | Re-fit if exact reproduction needed |
| Onehot encoding adds `week_53` column | **LOW** — schema gains one column | Update downstream column lists |
| Onehot encoding adds `day_of_year_366` column | **LOW** — schema gains one column | Update downstream column lists |
| Unnamed Series + `keep_original_columns=True` raises `ValueError` | **LOW** — fail-fast | Set `X.name` or pass `keep_original_columns=False` |
| `features_to_encode` with non-encodable names emits `IgnoredArgumentWarning` | **LOW** — log noise only | Remove non-encodable names from `features_to_encode` |
| `spline_kwargs={'knots': …}` or `{'sparse_output': True}` raises `ValueError` | **LOW** — was effectively broken before | Don't pass these keys |
| Non-integer `fill_na` raises `TypeError` | **LOW** | Pass `int` or `np.nan` |
| Default `fill_na` changed `0.` → `0` | **NONE** — functionally identical for `Int64` output | — |
| Invalid `encoding` raises at `fit()` instead of `__init__` | **NEGLIGIBLE** | — |

---

## Test coverage

### New tests (5)

| Test | Verifies |
|---|---|
| `test_create_datetime_features_onehot_week_53_and_day_of_year_366` | Week 53 / day 366 produce dedicated onehot columns |
| `test_create_datetime_features_spline_week_53_continuity` | Week 53 is a true neighbor of weeks 52 and 1 in spline space |
| `test_create_datetime_features_spline_month_12_distinct_from_month_1` | Spline values for Dec and Jan differ |

(Plus the existing `test_create_datetime_features_spline_encoding_expected_values` was regenerated, not added.)

### Edge cases verified manually (sanity script)

- 2-year hourly index with all default features (cyclical/spline/onehot)
- Leap-year and ISO-week-53 detection across multiple years
- `spline_kwargs={'extrapolation': 'continue'}` forwards correctly
- Blocked `spline_kwargs` (`knots`, `sparse_output`) raise
- `fill_na=0.5` raises with helpful message
- `fit()` on empty DataFrame raises with helpful message
- Unnamed Series + `keep_original_columns=True` raises with helpful message

---

## Non-blocking issues flagged for future consideration

1. **`fit()` empty-input guard is asymmetric with `transform()`** — fit() raises a clear error on empty input; transform() doesn't. Calling `transform` on empty data is unusual but the inconsistency is worth noting.

2. **Class-level `Attributes` docstring** ([_calendar.py:343-345](skforecast/preprocessing/_calendar.py#L343-L345)) doesn't mention `extrapolation` for `spline_kwargs`. The `Parameters` section is fully updated; the `Attributes` section just describes stored params, so brevity is acceptable.

3. **Knot-placement comment specificity** ([_calendar.py:243-245](skforecast/preprocessing/_calendar.py#L243-L245)) describes the periodic-boundary reasoning. If a user passes `extrapolation='continue'`, the comment's premise no longer applies — but the formula is still correct (knots are valid for any extrapolation mode).

4. **`numpy.float32(np.nan)` is rejected by `fill_na` validation** because `isinstance(np.float32(...), float)` is False. Acceptable edge case — users typically pass Python `int`, `0`, or `np.nan`.

---

## Additional issues / optimizations in `_calendar.py` not covered by the plan

1. **`transform()` mutates state** at [_calendar.py:381](skforecast/preprocessing/_calendar.py#L381) (`self.feature_names_out_ = ...`). This violates sklearn convention (transform should be pure). It also masks the case where `transform` is called without `fit`. Recommendation: drop the assignment in `transform` and rely on the one in `fit` — the values are deterministic from the constructor params.

2. **`__init__` validates `encoding` only**, not the other params. The plan moves encoding validation to fit(); while we're there, decide whether to also validate `features` / `features_to_encode` / `spline_kwargs` keys early. Right now it's mixed.

3. **Custom `max_values` keys not in `features` are silently ignored** in both cyclical ([_calendar.py:152-156](skforecast/preprocessing/_calendar.py#L152-L156)) and spline ([_calendar.py:200-216](skforecast/preprocessing/_calendar.py#L200-L216)) branches. E.g. `max_values={'month': 12, 'mnth': 12}` (typo) silently produces no encoded month columns. Worth a warning.

4. **`calculate_distance_from_holiday` doesn't validate column existence**. Both `holiday_column` ([_calendar.py:546](skforecast/preprocessing/_calendar.py#L546)) and `date_column` ([_calendar.py:562](skforecast/preprocessing/_calendar.py#L562)) are accessed without an existence check. A pandas `KeyError` is fine but a typed skforecast error would be more consistent with the rest of the module.

5. **`fit()` uses `X.iloc[:2]`** at [_calendar.py:341](skforecast/preprocessing/_calendar.py#L341) to compute output names without scanning. This works because the encoding is value-independent (onehot uses `feature_known_categories`, not observed values; spline uses fixed knots; cyclical produces fixed `_sin`/`_cos`). The Phase 3.2 empty-input guard already handles `len(X) == 0`; just confirm `iloc[:2]` on a 1-row input still produces all expected columns (it should, since column count doesn't depend on row count).

6. **Onehot for `day_of_year` produces 366 columns**. For multi-year hourly data this is a meaningful memory hit. Not a bug, but a doc note ("for high-cardinality features prefer cyclical or spline") would be useful.

7. **`default_features` and `default_max_values`** are each used exactly once and could be inlined into the `if ... is None: ... = [...]` blocks at [_calendar.py:82-110](skforecast/preprocessing/_calendar.py#L82-L110). Pure cosmetics.

8. **`re.split(r'[-_]', freq_str)[0]`** at [_calendar.py:428](skforecast/preprocessing/_calendar.py#L428) — splitting on `_` is harmless but unnecessary; pandas frequency aliases use `-` as the only separator. Not worth changing unless a counterexample shows up.

9. **`isocalendar().week.astype(int)`** at [_calendar.py:117](skforecast/preprocessing/_calendar.py#L117) — `.week` is already int-typed in pandas ≥ 2.0, so `.astype(int)` is a no-op. Tiny cleanup.

---

## How to verify

```bash
# From repo root, with skforecast_22_py13 env active:
pytest skforecast/preprocessing/tests/ -v
```

Expected: **208 passed, 3 warnings**. The warnings are the expected `IgnoredArgumentWarning` from the two `onehot_year_and_weekend_never_encoded` tests, which exercise the new `features_to_encode` warning path.
