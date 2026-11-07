"""
QW-13: a single warning, with counts, for the training rows with NaN
(roadmap 0.27.0, section 2).

Baseline on `0.27.x` (commit cb6a280d3) and result of the change: the
`MissingValuesWarning` raised when the training
matrices have NaN, and the rows kept, in the 5 forecasters that raise them
(`ForecasterRecursive`, `ForecasterRecursiveClassifier`,
`ForecasterRecursiveMultiSeries`, `ForecasterDirect` and
`ForecasterDirectMultiVariate`).

Data: 100 days, `lags=5` (window_size 5, 95 training rows per series).

- `y`: NaN at positions 30 and 60 of the target series.
- `exog`: NaN at position 45 of the exogenous variable.
- `both`: the two.

Each case runs with `dropna_from_series=True` and `False`.

1. Rows: `create_train_X_y` of each forecaster, the warnings seen with the
   default filters (each case in its own `catch_warnings`, so the registry of
   "default" does not hide a warning already seen in a previous case) and the
   rows with NaN target, with NaN predictors (and non-NaN target) and kept.
2. Direct and DMV: the rows dropped in each step (`_filter_nan_X_y_step`), and
   the rows dropped in at least one step (same row of `X_train_autoreg`).
   2b: the same with `steps=24`, `lags=24` and 500 hours, with isolated NaN,
   a block of NaN and a NaN near the start and near the end.
3. Searches: `grid_search_forecaster` (Recursive, `lags_grid=[5, 10]`) and
   `grid_search_forecaster_multiseries` with `TimeSeriesFold(refit=True)`, to
   see how `_SearchWarnings` groups the two warnings, with `n_jobs='auto'` and
   `n_jobs=1` (the warnings printed by the joblib workers are captured from
   the file descriptors 1 and 2 and counted).
4. The warning emitted in each case of 1, 2 and 2b against the approved text,
   built here from the counts measured (per step in Direct and DMV), and the
   check that its first sentence does not change. On the baseline (before the
   change, commit cb6a280d3) this section printed the proposal instead.
"""
import os
import re
import sys
import tempfile
import warnings
from collections import Counter
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
from skforecast.model_selection import (
    TimeSeriesFold,
    grid_search_forecaster,
    grid_search_forecaster_multiseries,
)
from skforecast.exceptions import MissingValuesWarning
from skforecast.model_selection._utils import _split_first_sentence
from skforecast.recursive import (
    ForecasterRecursive,
    ForecasterRecursiveClassifier,
    ForecasterRecursiveMultiSeries,
)

N = 100
LAGS = 5
NAN_Y = [30, 60]
NAN_EXOG = [45]
INDEX = pd.date_range("2022-01-01", periods=N, freq="D")
rng = np.random.default_rng(123)


def make_data(case: str):
    y = pd.Series(10 + rng.normal(size=N).cumsum(), index=INDEX, name="y")
    y_2 = pd.Series(5 + rng.normal(size=N).cumsum(), index=INDEX, name="y_2")
    exog = pd.Series(rng.normal(size=N), index=INDEX, name="exog")
    if case in ("y", "both"):
        y.iloc[NAN_Y] = np.nan
    if case in ("exog", "both"):
        exog.iloc[NAN_EXOG] = np.nan
    return y, y_2, exog


def to_classes(y: pd.Series) -> pd.Series:
    labels = pd.cut(y, bins=3, labels=["low", "mid", "high"]).astype(object)
    labels[y.isna()] = np.nan
    return labels


def warning_lines(records) -> list[str]:
    return [
        f"{r.category.__name__}: {getattr(r.message, 'message', str(r.message))}"
        for r in records
    ]


def print_warnings(records, indent: str = "    ") -> None:
    lines = warning_lines(records)
    if not lines:
        print(f"{indent}(no warnings)")
    for line in lines:
        print(f"{indent}- {line}")


def count_rows(forecaster, kind: str, case: str, dropna: bool) -> dict:
    """Rows of the training matrix of a Recursive / Classifier / MultiSeries."""

    y, y_2, exog = make_data(case)
    with warnings.catch_warnings(record=True) as records:
        if kind == "multiseries":
            series = pd.DataFrame({"y": y, "y_2": y_2})
            X_train, y_train = forecaster.create_train_X_y(series=series, exog=exog)
        elif kind == "classifier":
            X_train, y_train = forecaster.create_train_X_y(y=to_classes(y), exog=exog)
        else:
            X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)

    n_series = 2 if kind == "multiseries" else 1
    n_total = (N - LAGS) * n_series
    # NaN targets only in `y` (positions after the window).
    n_nan_target = sum(1 for p in NAN_Y if p >= LAGS) if case in ("y", "both") else 0
    n_kept = len(y_train)
    if dropna:
        n_nan_pred = n_total - n_nan_target - n_kept
    else:
        n_nan_pred = int(X_train.isna().any(axis=1).sum())

    return {
        "records": records,
        "n_total": n_total,
        "n_nan_target": n_nan_target,
        "n_nan_pred": n_nan_pred,
        "n_kept": n_kept,
    }


def count_rows_direct(forecaster, kind: str, case: str, data=None) -> dict:
    """Rows dropped in each step of Direct / DMV."""

    y, y_2, exog = make_data(case) if data is None else data
    with warnings.catch_warnings(record=True) as records:
        if kind == "dmv":
            series = pd.DataFrame({"y": y, "y_2": y_2})
            out = forecaster._create_train_X_y(series=series, exog=exog)
        else:
            out = forecaster._create_train_X_y(y=y, exog=exog)
    X_train_autoreg, X_train_exog, X_train_calendar, y_train = out[:4]
    n_rows = X_train_autoreg.shape[0]

    per_step = {}
    dropped_any = np.zeros(n_rows, dtype=bool)
    nan_target_any = np.zeros(n_rows, dtype=bool)
    nan_pred_any = np.zeros(n_rows, dtype=bool)
    for step in forecaster.steps:
        X_step, y_step = forecaster._create_train_X_y_step(
            X_train_autoreg  = X_train_autoreg,
            X_train_exog     = X_train_exog,
            X_train_calendar = X_train_calendar,
            y_train          = y_train,
            step             = step,
        )
        nan_target = np.isnan(y_step)
        nan_pred = pd.isna(X_step).any(axis=1) & ~nan_target
        X_kept, _, _ = forecaster._filter_nan_X_y_step(X_step, y_step)
        dropped = nan_target | (nan_pred if forecaster.dropna_from_series else False)
        per_step[step] = (
            int(nan_target.sum()), int(nan_pred.sum()), len(X_kept), len(y_step)
        )
        dropped_any |= dropped
        nan_target_any |= nan_target
        nan_pred_any |= nan_pred

    return {
        "records": records,
        "n_rows": n_rows,
        "per_step": per_step,
        "n_dropped_any": int(dropped_any.sum()),
        "n_nan_target_any": int(nan_target_any.sum()),
        # Rows with NaN predictors (and non-NaN target) in at least one step and
        # NaN target in none of them, so the two counts add up to the union.
        "n_nan_pred_any": int((nan_pred_any & ~nan_target_any).sum()),
        "dropped_range": (
            min(v[3] - v[2] for v in per_step.values()),
            max(v[3] - v[2] for v in per_step.values()),
        ),
        "nan_target_range": (
            min(v[0] for v in per_step.values()), max(v[0] for v in per_step.values())
        ),
        "nan_pred_range": (
            min(v[1] for v in per_step.values()), max(v[1] for v in per_step.values())
        ),
    }


def proposed_message(
    n_nan_target: int | tuple[int, int],
    n_nan_pred: int | tuple[int, int],
    n_total: int,
    dropna: bool,
    data_name: str = "y",
    n_dropped: tuple[int, int] | None = None,
) -> str | None:
    """
    Text approved for the new warning (QW-13.md, section 4), written here
    independently of `_warn_nan_training_rows` to check it. In Direct and DMV
    the counts are per step: a tuple (min, max) over the steps, and
    `n_dropped` is the (min, max) of the rows dropped per step.
    """

    per_step = isinstance(n_nan_target, tuple)
    if not per_step:
        n_nan_target, n_nan_pred = (n_nan_target,) * 2, (n_nan_pred,) * 2
    has_t, has_p = max(n_nan_target) > 0, max(n_nan_pred) > 0
    if not has_t and not has_p:
        return None

    def fmt(n: tuple[int, int]) -> str:
        return f"{n[0]}" if n[0] == n[1] else f"{n[0]} to {n[1]}"

    def verb(n: tuple[int, int]) -> str:
        return "is" if tuple(n) == (1, 1) else "are"

    where = " per step" if per_step else ""
    pred = "with NaN predictors (lags, window features or exog)."
    if n_dropped is None:
        n_dropped = (n_nan_target[0] + n_nan_pred[0],) * 2
    if dropna and has_t and has_p:
        counts = (
            f"{fmt(n_dropped)} of {n_total} rows {verb(n_dropped)} dropped{where}: "
            f"{fmt(n_nan_target)} with a NaN target and {fmt(n_nan_pred)} {pred}"
        )
    elif dropna and has_p:
        counts = f"{fmt(n_nan_pred)} of {n_total} rows {verb(n_nan_pred)} dropped{where}, {pred}"
    elif not has_p:
        counts = (
            f"{fmt(n_nan_target)} of {n_total} rows {verb(n_nan_target)} "
            f"dropped{where}, with a NaN target."
        )
    elif has_t:
        counts = (
            f"{fmt(n_nan_target)} of {n_total} rows {verb(n_nan_target)} "
            f"dropped{where} (NaN target) and {fmt(n_nan_pred)} "
            f"{verb(n_nan_pred)} kept {pred}"
        )
    else:
        counts = f"{fmt(n_nan_pred)} of {n_total} rows {verb(n_nan_pred)} kept{where} {pred}"
    action = ""
    if has_p:
        action = (
            "To keep the rows with NaN predictors, set "
            "`forecaster.dropna_from_series = False` (the estimator must "
            "support NaN)."
            if dropna else
            "The estimator must support NaN. To drop those rows, set "
            "`forecaster.dropna_from_series = True`."
        )
    first = "Some rows of the training matrix have missing values."
    cause = f"They come from interspersed NaNs in `{data_name}` or `exog`."
    return " ".join(p for p in (first, counts, cause, action) if p)


FORECASTERS = {
    "ForecasterRecursive": (
        "recursive",
        lambda d: ForecasterRecursive(
            HistGradientBoostingRegressor(max_iter=10), lags=LAGS, dropna_from_series=d
        ),
    ),
    "ForecasterRecursiveClassifier": (
        "classifier",
        lambda d: ForecasterRecursiveClassifier(
            HistGradientBoostingClassifier(max_iter=10), lags=LAGS, dropna_from_series=d
        ),
    ),
    "ForecasterRecursiveMultiSeries": (
        "multiseries",
        lambda d: ForecasterRecursiveMultiSeries(
            HistGradientBoostingRegressor(max_iter=10), lags=LAGS, dropna_from_series=d
        ),
    ),
    "ForecasterDirect": (
        "direct",
        lambda d: ForecasterDirect(
            HistGradientBoostingRegressor(max_iter=10), steps=3, lags=LAGS,
            dropna_from_series=d
        ),
    ),
    "ForecasterDirectMultiVariate": (
        "dmv",
        lambda d: ForecasterDirectMultiVariate(
            HistGradientBoostingRegressor(max_iter=10), level="y", steps=3,
            lags=LAGS, dropna_from_series=d
        ),
    ),
}

CASES = ["y", "exog", "both"]
proposals = []

print("=" * 88)
print("1-2. Warnings and rows of the training matrix")
print("=" * 88)
for name, (kind, make) in FORECASTERS.items():
    for dropna in (True, False):
        for case in CASES:
            forecaster = make(dropna)
            print(f"\n{name}, dropna_from_series={dropna}, NaN in {case}")
            data_name = "series" if kind in ("multiseries", "dmv") else "y"
            if kind in ("direct", "dmv"):
                res = count_rows_direct(forecaster, kind, case)
                print_warnings(res["records"])
                for step, (n_t, n_p, n_k, n_r) in res["per_step"].items():
                    print(
                        f"    step {step}: {n_t} NaN target, {n_p} NaN predictors, "
                        f"{n_k} of {n_r} rows kept"
                    )
                print(
                    f"    at least one step: {res['n_dropped_any']} of "
                    f"{res['n_rows']} rows dropped ({res['n_nan_target_any']} NaN "
                    f"target, {res['n_nan_pred_any']} NaN predictors)"
                )
                msg = proposed_message(
                    res["nan_target_range"], res["nan_pred_range"], res["n_rows"],
                    dropna, data_name, n_dropped=res["dropped_range"]
                )
            else:
                res = count_rows(forecaster, kind, case, dropna)
                print_warnings(res["records"])
                print(
                    f"    {res['n_nan_target']} NaN target, {res['n_nan_pred']} NaN "
                    f"predictors, {res['n_kept']} of {res['n_total']} rows kept"
                )
                msg = proposed_message(
                    res["n_nan_target"], res["n_nan_pred"], res["n_total"],
                    dropna, data_name
                )
            proposals.append((name, dropna, case, msg, res["records"]))

print()
print("=" * 88)
print("2b. Direct and DMV with steps=24, lags=24, 500 hours: rows dropped per step")
print("=" * 88)
N_H = 500
index_h = pd.date_range("2022-01-01", periods=N_H, freq="h")
gap_cases = {
    "2 isolated NaN (200, 300)": [200, 300],
    "block of 10 NaN (200-209)": list(range(200, 210)),
    "1 NaN near the start (30)": [30],
    "1 NaN near the end (490)": [490],
}
for gap_name, positions in gap_cases.items():
    y_h = pd.Series(10 + rng.normal(size=N_H).cumsum(), index=index_h, name="y")
    y_h.iloc[positions] = np.nan
    y_2_h = pd.Series(rng.normal(size=N_H).cumsum(), index=index_h, name="y_2")
    exog_h = pd.Series(rng.normal(size=N_H), index=index_h, name="exog")
    for kind, cls_kwargs in (
        ("direct", {}),
        ("dmv", {"level": "y"}),
    ):
        cls = ForecasterDirect if kind == "direct" else ForecasterDirectMultiVariate
        for dropna in (True, False):
            forecaster = cls(
                HistGradientBoostingRegressor(max_iter=10), steps=24, lags=24,
                dropna_from_series=dropna, **cls_kwargs
            )
            res = count_rows_direct(
                forecaster, kind, "y", data=(y_h, y_2_h, exog_h)
            )
            dropped = [
                n_r - n_k for (_, _, n_k, n_r) in res["per_step"].values()
            ]
            nan_target = [n_t for (n_t, _, _, _) in res["per_step"].values()]
            print(
                f"{cls.__name__:<29} dropna={dropna!s:<5} {gap_name:<26}: "
                f"per step {min(dropped)}-{max(dropped)} of {res['n_rows']} dropped "
                f"(NaN target {min(nan_target)}-{max(nan_target)}); "
                f"at least one step {res['n_dropped_any']} "
                f"({res['n_nan_target_any']} NaN target, "
                f"{res['n_nan_pred_any']} NaN predictors)"
            )
            proposals.append((
                f"{cls.__name__}, steps=24, lags=24", dropna, gap_name,
                proposed_message(
                    res["nan_target_range"], res["nan_pred_range"], res["n_rows"],
                    dropna, "series" if kind == "dmv" else "y",
                    n_dropped=res["dropped_range"]
                ),
                res["records"],
            ))


@contextmanager
def capture_fds():
    """
    Redirect the file descriptors 1 and 2 to a temporary file, to capture the
    warnings printed by the joblib workers of the backtesting (`n_jobs > 1`),
    which reach neither `_SearchWarnings` nor `catch_warnings`.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    out = {"text": ""}
    with tempfile.TemporaryFile(mode="w+") as tmp:
        for fd in (1, 2):
            os.dup2(tmp.fileno(), fd)
        try:
            yield out
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, saved_fd in saved.items():
                os.dup2(saved_fd, fd)
                os.close(saved_fd)
            tmp.seek(0)
            out["text"] = tmp.read()


def print_worker_warnings(text: str) -> None:
    # Each warning printed by a worker is a box whose first line of text
    # follows the title line.
    counts = Counter(
        m.group(1).strip()[:60]
        for m in re.finditer(r"MissingValuesWarning ─+╮\n│ (.+?)│", text)
    )
    if not counts:
        print("    workers: (no warnings)")
    for start, n in counts.items():
        print(f"    workers: {n} x {start}...")


print()
print("=" * 88)
print(
    "3. Searches (grid, TimeSeriesFold(steps=10, initial_train_size=60, refit=True),\n"
    "   lags_grid=[5, 10], 2 values of max_depth: 4 candidates, 4 folds)"
)
print("=" * 88)
for dropna in (True, False):
    for n_jobs in ("auto", 1):
        cv = TimeSeriesFold(
            steps=10, initial_train_size=60, refit=True, verbose=False
        )
        y, y_2, exog = make_data("both")
        searches = {
            "grid_search_forecaster, ForecasterRecursive": lambda: grid_search_forecaster(
                forecaster    = FORECASTERS["ForecasterRecursive"][1](dropna),
                y             = y,
                exog          = exog,
                cv            = cv,
                param_grid    = {"max_depth": [3, None]},
                lags_grid     = [5, 10],
                metric        = "mean_absolute_error",
                return_best   = False,
                n_jobs        = n_jobs,
                show_progress = False,
            ),
            "grid_search_forecaster_multiseries, ForecasterRecursiveMultiSeries": (
                lambda: grid_search_forecaster_multiseries(
                    forecaster    = FORECASTERS["ForecasterRecursiveMultiSeries"][1](dropna),
                    series        = {"y": y, "y_2": y_2},
                    exog          = exog,
                    cv            = cv,
                    param_grid    = {"max_depth": [3, None]},
                    lags_grid     = [5, 10],
                    metric        = "mean_absolute_error",
                    return_best   = False,
                    n_jobs        = n_jobs,
                    show_progress = False,
                )
            ),
        }
        for search_name, run in searches.items():
            print(
                f"\n{search_name}, dropna_from_series={dropna}, n_jobs={n_jobs!r}",
                flush=True
            )
            with capture_fds() as worker_out:
                with warnings.catch_warnings(record=True) as records:
                    run()
            print_warnings(records)
            print_worker_warnings(worker_out["text"])

print()
print("=" * 88)
print("4. Warnings emitted vs the approved text (sections 1, 2 and 2b)")
print("=" * 88)
first_sentences = set()
n_ok = 0
for name, dropna, case, expected, records in proposals:
    texts = [
        getattr(r.message, "message", str(r.message)) for r in records
        if issubclass(r.category, MissingValuesWarning)
    ]
    new = [t for t in texts if t.startswith("Some rows of the training matrix")]
    old = [t for t in texts if t.startswith("NaNs detected") or "`exog` has missing" in t]
    ok = new == ([expected] if expected else []) and not old
    n_ok += ok
    if not ok:
        print(f"\nMISMATCH {name}, dropna_from_series={dropna}, NaN in {case}")
        print(f"    expected: {expected}")
        print(f"    emitted : {texts}")
    first_sentences.update(_split_first_sentence(t)[0] for t in new)
print(f"\n{n_ok} of {len(proposals)} cases emit exactly the approved text, once")
print(f"Distinct first sentences: {len(first_sentences)}")
for sentence in first_sentences:
    print(f"    {sentence!r}")
assert n_ok == len(proposals)
assert len(first_sentences) == 1
# The counts must not be in the first sentence (`_SearchWarnings` groups by it).
assert not re.search(r"\d", next(iter(first_sentences)))
