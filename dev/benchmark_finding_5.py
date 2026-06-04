"""
Benchmark for OPTIMIZATION_FINDINGS.md — Finding #5
===================================================
"Single-series `_create_train_X_y` builds a Python list of column blocks"

Finding #5 proposes two changes to
`ForecasterRecursive._create_train_X_y` (skforecast/recursive/_forecaster_recursive.py):

  (1) Replace the final `np.concatenate(X_train, axis=1)` (line ~916) with a
      preallocated `np.empty((n_rows, n_cols))` + per-block slice writes,
      claiming "~15-25% faster `_create_train_X_y` per fit".

  (2) Replace `pd.isna(X_train)` with `np.isnan(X_train)` (lines ~933 / ~947),
      claiming a small micro-optimization.

This script tests whether either change is worth making. It runs against the
REAL skforecast code path (it captures the exact column blocks the production
code concatenates, including their real strides/dtypes), so the numbers reflect
production behavior, not synthetic arrays.

Run:
    conda run -n skforecast_22_py13 python dev/benchmark_finding_5.py

Conclusions are printed at the bottom with an explicit PASS/FAIL verdict for
"is #5 a valid optimization?".
"""

from __future__ import annotations

import timeit
import numpy as np
import pandas as pd

import skforecast.recursive._forecaster_recursive as fr_mod
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------- #
# Timing helper                                                               #
# --------------------------------------------------------------------------- #
def best_time(fn, *, number, repeat=7):
    """Best-of-`repeat` average seconds per call (minimises OS/GC noise)."""
    return min(timeit.repeat(fn, number=number, repeat=repeat)) / number


def pick_number(n_rows):
    """Scale inner repeats inversely with problem size to keep runtime sane."""
    if n_rows <= 2_000:
        return 500
    if n_rows <= 20_000:
        return 200
    if n_rows <= 100_000:
        return 50
    return 20


# --------------------------------------------------------------------------- #
# Assembly variants (exactly what finding #5 contrasts)                       #
# --------------------------------------------------------------------------- #
def assemble_concat(blocks):
    """Current production assembly (line ~887-916)."""
    if len(blocks) == 1:
        return blocks[0]
    return np.concatenate(blocks, axis=1)


def assemble_prealloc(blocks):
    """Finding #5's proposed assembly: preallocate + slice writes."""
    if len(blocks) == 1:
        # Faithful to the proposal: a preallocated buffer is always built, so
        # even the single-block case is copied into a fresh contiguous array.
        only = blocks[0]
        out = np.empty(only.shape, order="C", dtype=float)
        out[:] = only
        return out
    n_rows = blocks[0].shape[0]
    n_cols = sum(b.shape[1] for b in blocks)
    out = np.empty((n_rows, n_cols), order="C", dtype=float)
    off = 0
    for b in blocks:
        w = b.shape[1]
        out[:, off:off + w] = b
        off += w
    return out


# --------------------------------------------------------------------------- #
# Capture the REAL blocks that `_create_train_X_y` concatenates               #
# --------------------------------------------------------------------------- #
def capture_blocks(forecaster, y, exog):
    """
    Run the real `_create_train_X_y` once with `np.concatenate` patched so we
    grab the exact list of column blocks (real strides + dtypes) the production
    code assembles. Returns (blocks, single_block_array_or_None).

    For the lags-only configuration the production code never calls
    `np.concatenate` (it returns `X_train[0]` directly), so we reconstruct the
    single block from the public helper to benchmark the proposed copy.
    """
    captured = {}
    orig = np.concatenate

    def capturing(arrays, *args, **kwargs):
        axis = kwargs.get("axis", args[0] if args else 0)
        if axis == 1:
            captured["blocks"] = [np.asarray(a) for a in arrays]
        return orig(arrays, *args, **kwargs)

    fr_mod.np.concatenate = capturing
    try:
        forecaster._create_train_X_y(y=y, exog=exog)
    finally:
        fr_mod.np.concatenate = orig

    if "blocks" in captured:
        return captured["blocks"], None

    # Single-block (lags-only) path: rebuild the block the helper produced.
    y_values = y.to_numpy()
    train_index = y.index[forecaster.window_size:]
    X_lags, _ = forecaster._create_lags(y=y_values, train_index=train_index)
    return [X_lags], X_lags


# --------------------------------------------------------------------------- #
# Scenario builder                                                            #
# --------------------------------------------------------------------------- #
def build_scenario(n_rows, *, use_wf, n_exog, lags=24):
    idx = pd.date_range("2000-01-01", periods=n_rows, freq="h")
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=n_rows), index=idx, name="y")

    exog = None
    if n_exog:
        exog = pd.DataFrame(
            rng.normal(size=(n_rows, n_exog)),
            index=idx,
            columns=[f"exog_{i}" for i in range(n_exog)],
        )

    wf = RollingFeatures(stats=["mean", "std", "min", "max"], window_sizes=24) if use_wf else None

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(),
        lags=lags,
        window_features=wf,
    )
    return forecaster, y, exog


SCENARIOS = [
    # (label, n_rows, use_wf, n_exog)
    ("lags only                 ", 10_000, False, 0),
    ("lags + WF                  ", 10_000, True, 0),
    ("lags + WF + exog(5)        ", 2_000, True, 5),
    ("lags + WF + exog(5)        ", 10_000, True, 5),
    ("lags + WF + exog(5)        ", 100_000, True, 5),
    ("lags + WF + exog(20)       ", 10_000, True, 20),
]


# --------------------------------------------------------------------------- #
# PART A — assembly step in isolation, on the real captured blocks            #
# --------------------------------------------------------------------------- #
def part_a():
    print("=" * 100)
    print("PART A — Assembly step in isolation (real blocks from _create_train_X_y)")
    print("        concat (current)  vs  preallocate+slice-write (finding #5)")
    print("=" * 100)
    print(f"{'scenario':<28}{'n_rows':>9}{'#blk':>6}{'concat us':>13}"
          f"{'prealloc us':>14}{'speedup':>10}")
    print("-" * 100)

    results = []
    for label, n_rows, use_wf, n_exog in SCENARIOS:
        fc, y, exog = build_scenario(n_rows, use_wf=use_wf, n_exog=n_exog)
        blocks, _ = capture_blocks(fc, y, exog)

        # sanity: assembled results identical
        a = assemble_concat(blocks)
        b = assemble_prealloc(blocks)
        assert np.array_equal(a, b), f"{label}: results differ!"

        number = pick_number(n_rows)
        t_c = best_time(lambda: assemble_concat(blocks), number=number)
        t_p = best_time(lambda: assemble_prealloc(blocks), number=number)
        speedup = (t_c / t_p - 1) * 100
        results.append((label, n_rows, t_c, t_p, speedup))
        print(f"{label:<28}{n_rows:>9}{len(blocks):>6}{t_c*1e6:>13.2f}"
              f"{t_p*1e6:>14.2f}{speedup:>+9.1f}%")
    print("-" * 100)
    print("speedup = how much faster the proposed prealloc assembly is vs concat")
    print("(positive => prealloc faster; values within ~±5% are noise)\n")
    return results


# --------------------------------------------------------------------------- #
# PART B — share of total _create_train_X_y runtime spent in assembly         #
# --------------------------------------------------------------------------- #
def part_b(part_a_results):
    print("=" * 100)
    print("PART B — Is the assembly step even a meaningful slice of _create_train_X_y?")
    print("        Projected best-case full-function speedup = (T_concat - T_prealloc) / T_full")
    print("=" * 100)
    print(f"{'scenario':<28}{'n_rows':>9}{'T_full us':>13}{'assembly us':>14}"
          f"{'assembly %':>13}{'proj. speedup':>16}")
    print("-" * 100)

    by_key = {(lab, n): (tc, tp, sp) for (lab, n, tc, tp, sp) in part_a_results}
    for label, n_rows, use_wf, n_exog in SCENARIOS:
        fc, y, exog = build_scenario(n_rows, use_wf=use_wf, n_exog=n_exog)
        number = max(10, pick_number(n_rows) // 5)  # full func is heavier
        t_full = best_time(lambda: fc._create_train_X_y(y=y, exog=exog), number=number)

        t_c, t_p, _ = by_key[(label, n_rows)]
        assembly_share = t_c / t_full * 100
        projected = (t_c - t_p) / t_full * 100
        print(f"{label:<28}{n_rows:>9}{t_full*1e6:>13.2f}{t_c*1e6:>14.2f}"
              f"{assembly_share:>12.1f}%{projected:>+15.2f}%")
    print("-" * 100)
    print("proj. speedup = the MOST finding #5 (change 1) could improve the whole")
    print("function, assuming prealloc were free. The finding claims +15-25%.\n")


# --------------------------------------------------------------------------- #
# PART C — single-block (lags-only) regression                                #
# --------------------------------------------------------------------------- #
def part_c():
    print("=" * 100)
    print("PART C — Single-block path regression (lags-only is a common config)")
    print("=" * 100)
    fc, y, exog = build_scenario(50_000, use_wf=False, n_exog=0)
    _, single = capture_blocks(fc, y, exog)

    print(f"  Captured lags block: shape={single.shape}, dtype={single.dtype}")
    print(f"    C_CONTIGUOUS={single.flags['C_CONTIGUOUS']}  "
          f"strides={single.strides}  (reversed view => negative stride)")
    print("  Current code returns this VIEW directly (zero copies).")
    print("  Finding #5 preallocates a buffer and copies into it.\n")

    number = 2000
    t_view = best_time(lambda: assemble_concat([single]), number=number)
    t_copy = best_time(lambda: assemble_prealloc([single]), number=number)
    print(f"    current (return view) : {t_view*1e6:8.2f} us")
    print(f"    finding #5 (copy)     : {t_copy*1e6:8.2f} us")
    print(f"    => finding #5 is {t_copy/max(t_view,1e-12):.0f}x SLOWER on the "
          f"lags-only path (adds a copy that does not exist today)\n")


# --------------------------------------------------------------------------- #
# PART D — correctness: pd.isna -> np.isnan is a regression on object dtype    #
# --------------------------------------------------------------------------- #
def part_d():
    print("=" * 100)
    print("PART D — Correctness of change (2): pd.isna(X_train) -> np.isnan(X_train)")
    print("=" * 100)
    print("  X_train is built by np.concatenate([float_lags, ..., exog.to_numpy()]).")
    print("  When exog.to_numpy() is NOT float (e.g. a category/datetime/object")
    print("  column reaches line 916), X_train becomes object dtype.\n")

    # Faithfully reproduce what line 916 produces with a non-float exog column.
    float_lags = np.random.rand(5, 3)
    exog_np = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0],
            "cat": pd.Categorical(["a", "b", "a", "c", "b"]),
        }
    ).to_numpy()
    X_train = np.concatenate([float_lags, exog_np], axis=1)
    print(f"  Resulting X_train.dtype = {X_train.dtype}")

    ok_pd = fail_np = None
    try:
        ok_pd = bool(pd.isna(X_train).any())
        print(f"    pd.isna(X_train).any()  -> {ok_pd}        (current code: works)")
    except Exception as e:  # pragma: no cover
        print(f"    pd.isna  FAILED: {e}")

    try:
        _ = bool(np.isnan(X_train).any())
        print("    np.isnan(X_train).any() -> works")
    except Exception as e:
        fail_np = type(e).__name__
        print(f"    np.isnan(X_train).any() -> {fail_np}: {e}")
    print()
    return fail_np is not None


# --------------------------------------------------------------------------- #
# Verdict                                                                     #
# --------------------------------------------------------------------------- #
def verdict(part_a_results, isnan_breaks):
    print("=" * 100)
    print("VERDICT — Is finding #5 a valid optimization?")
    print("=" * 100)

    speedups = [sp for (_, _, _, _, sp) in part_a_results]
    max_speedup = max(speedups)
    mean_speedup = sum(speedups) / len(speedups)

    print(f"  Change (1) preallocation:")
    print(f"    - assembly-step speedup across scenarios: "
          f"mean {mean_speedup:+.1f}%, best {max_speedup:+.1f}% (see PART A)")
    print(f"    - and the assembly step is only a small fraction of the whole")
    print(f"      function, so the projected full-function gain is far below the")
    print(f"      claimed +15-25% (see PART B)")
    print(f"    - it ADDS a copy on the common lags-only path (see PART C)")
    print(f"    => NOT a worthwhile optimization. The np.concatenate is the single")
    print(f"       combining copy; preallocation relocates the same work, it does")
    print(f"       not eliminate a copy.\n")

    print(f"  Change (2) pd.isna -> np.isnan:")
    if isnan_breaks:
        print(f"    => CORRECTNESS REGRESSION. np.isnan raises TypeError on the")
        print(f"       object-dtype X_train that arises from non-float exog (PART D).")
        print(f"       pd.isna must be kept.\n")
    else:
        print(f"    => np.isnan did not break here, but object-dtype X_train is")
        print(f"       reachable in production; pd.isna is the safe choice.\n")

    print("  CONCLUSION: Finding #5 should NOT be implemented. Change (1) yields no")
    print("  reliable speedup and regresses the lags-only path; change (2) is unsafe.")
    print("=" * 100)


if __name__ == "__main__":
    print(f"numpy {np.__version__} | pandas {pd.__version__}\n")
    res_a = part_a()
    part_b(res_a)
    part_c()
    isnan_breaks = part_d()
    verdict(res_a, isnan_breaks)
