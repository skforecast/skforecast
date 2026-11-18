"""
Line profile of the skforecast code (and only of it) with `sys.monitoring`:
for every line, how many times it runs and the time spent in it, callees
outside skforecast (pandas, numpy, scikit-learn) included and skforecast
callees excluded. Only the code objects of skforecast are instrumented, so
pandas runs at full speed.

Usage: python lineprof.py <out.json> <operation> <n_series> <exog_kind> [fmt] [steps]
  operation: predict | predict_stored | backtesting | fit | create_train_X_y
"""
import common  # noqa: F401  (first: blocks pyarrow, pins the process)

import json
import os
import sys
import time
import types

import skforecast
from common import (
    LAGS, TimeSeriesFold, backtesting_forecaster_multiseries, env_info,
    exog_for_predict, make_forecaster, make_wide, to_format,
)

mon = sys.monitoring
TOOL = mon.PROFILER_ID
E = mon.events
ROOT = skforecast.__path__[0]

hits = {}    # (file, line) -> times executed
own = {}     # (file, line) -> seconds, skforecast callees excluded
incl = {}    # (file, line) -> seconds, everything included
# One entry per running skforecast frame: [code, line, t_line, t_children]
stack = []
perf = time.perf_counter


def _close_line(frame, now):
    code, line, t_line, t_children = frame
    if line is not None:
        key = (code.co_filename, line)
        elapsed = now - t_line
        incl[key] = incl.get(key, 0.0) + elapsed
        own[key] = own.get(key, 0.0) + elapsed - t_children


def on_start(code, offset):
    stack.append([code, None, 0.0, 0.0, perf()])


def on_line(code, line):
    now = perf()
    while stack and stack[-1][0] is not code:  # frame left by an exception
        _pop(now)
    if not stack:
        stack.append([code, None, 0.0, 0.0, now])
    frame = stack[-1]
    _close_line(frame[:4], now)
    key = (code.co_filename, line)
    hits[key] = hits.get(key, 0) + 1
    frame[1] = line
    frame[3] = 0.0
    frame[2] = perf()


def _pop(now):
    frame = stack.pop()
    _close_line(frame[:4], now)
    if stack:
        # Time of the whole frame, charged to the line of the caller as time
        # spent in a skforecast callee.
        stack[-1][3] += now - frame[4]


def on_return(code, offset, retval):
    now = perf()
    while stack and stack[-1][0] is not code:
        _pop(now)
    if stack:
        _pop(now)
        if stack:
            # The callback itself must not be charged to the caller's line.
            extra = perf() - now
            stack[-1][2] += extra


def _codes_of(code, seen):
    if code in seen:
        return
    seen.add(code)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _codes_of(const, seen)


def instrument():
    import gc

    # Every function object of skforecast alive in the process (wrappers and
    # wrapped functions included) and the code objects nested in them.
    seen = set()
    for obj in gc.get_objects():
        if isinstance(obj, types.FunctionType):
            code = obj.__code__
            if code.co_filename.startswith(ROOT) and "tests" not in code.co_filename:
                _codes_of(code, seen)
    mon.use_tool_id(TOOL, "lineprof")
    mon.register_callback(TOOL, E.PY_START, on_start)
    mon.register_callback(TOOL, E.PY_RESUME, on_start)
    mon.register_callback(TOOL, E.LINE, on_line)
    mon.register_callback(TOOL, E.PY_RETURN, on_return)
    mon.register_callback(TOOL, E.PY_YIELD, on_return)
    events = E.PY_START | E.PY_RESUME | E.LINE | E.PY_RETURN | E.PY_YIELD
    for code in seen:
        mon.set_local_events(TOOL, code, events)
    return seen


def stop(seen):
    for code in seen:
        mon.set_local_events(TOOL, code, 0)
    mon.free_tool_id(TOOL)


def build(operation, n_series, exog_kind, fmt, steps):
    """Return (callable to profile, number of units: calls or folds)."""
    T = 400
    if operation in ("predict", "predict_stored"):
        n_calls = 100 if n_series <= 100 else 10
        series, exog = make_wide(n_series, T + n_calls + steps)
        names = list(series.columns)
        f = make_forecaster()
        f.fit(series=series.iloc[:T], exog=None if exog_kind == "none" else exog.iloc[:T])
        if operation == "predict":
            inputs = [
                (series.iloc[T + i - LAGS: T + i],
                 exog_for_predict(exog.iloc[T + i: T + i + steps], names, exog_kind))
                for i in range(n_calls)
            ]
        else:
            ex = exog_for_predict(exog.iloc[T: T + steps], names, exog_kind)
            inputs = [(None, ex)] * n_calls

        def fn():
            for last_window, ex in inputs:
                f.predict(steps=steps, last_window=last_window, exog=ex)

        return fn, n_calls
    if operation in ("predict_mv", "predict_mv_stored"):
        from sklearn.linear_model import LinearRegression
        from skforecast.direct import ForecasterDirectMultiVariate

        n_calls = 100
        series, exog = make_wide(n_series, T + n_calls + steps)
        f = ForecasterDirectMultiVariate(
            LinearRegression(), level=series.columns[0], lags=LAGS, steps=steps
        )
        f.fit(series=series.iloc[:T], exog=None if exog_kind == "none" else exog.iloc[:T])
        if operation == "predict_mv":
            inputs = [
                (series.iloc[T + i - LAGS: T + i],
                 None if exog_kind == "none" else exog.iloc[T + i: T + i + steps])
                for i in range(n_calls)
            ]
        else:
            inputs = [(None, None if exog_kind == "none" else exog.iloc[T: T + steps])] * n_calls

        def fn():
            for last_window, ex in inputs:
                f.predict(last_window=last_window, exog=ex)

        return fn, n_calls
    if operation == "backtesting":
        n_folds = 50 if n_series <= 100 else 10
        series, exog = make_wide(n_series, T + n_folds * steps)
        series, exog = to_format(series, None if exog_kind == "none" else exog, fmt)
        cv = TimeSeriesFold(steps=steps, initial_train_size=T, refit=False)
        kw = dict(
            forecaster=make_forecaster(), series=series, exog=exog, cv=cv,
            metric="mean_absolute_error", n_jobs=1, verbose=False,
            show_progress=False, suppress_warnings=True,
        )
        return (lambda: backtesting_forecaster_multiseries(**kw)), n_folds
    if operation in ("fit", "create_train_X_y"):
        series, exog = make_wide(n_series, 500)
        series, exog = to_format(series, None if exog_kind == "none" else exog, fmt)
        f = make_forecaster()
        if operation == "fit":
            return (lambda: f.fit(series=series, exog=exog, suppress_warnings=True)), 1
        return (
            lambda: f.create_train_X_y(series=series, exog=exog, suppress_warnings=True)
        ), 1
    raise ValueError(operation)


if __name__ == "__main__":
    out, operation, n_series, exog_kind = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    fmt = sys.argv[5] if len(sys.argv) > 5 else "wide"
    steps = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    fn, n_units = build(operation, n_series, exog_kind, fmt, steps)
    fn()  # warm-up, not profiled
    t0 = perf()
    fn()
    plain = perf() - t0
    # Several profiled passes: the time of every line is its minimum over them.
    seen = instrument()
    n_passes = int(os.environ.get("PROF_PASSES", "3"))
    own_min, incl_min, profiled = {}, {}, float("inf")
    for _ in range(n_passes):
        hits.clear(), own.clear(), incl.clear(), stack.clear()
        t0 = perf()
        fn()
        profiled = min(profiled, perf() - t0)
        for key, value in own.items():
            own_min[key] = min(own_min.get(key, float("inf")), value)
        for key, value in incl.items():
            incl_min[key] = min(incl_min.get(key, float("inf")), value)
    stop(seen)
    own, incl = own_min, incl_min
    src = {}
    rows = []
    for key in hits:
        file, line = key
        if file not in src:
            with open(file, encoding="utf-8") as fh:
                src[file] = fh.read().splitlines()
        rows.append({
            "file": file[len(ROOT) + 1:].replace("\\", "/"), "line": line,
            "hits": hits[key] / n_units,
            "own_us": own.get(key, 0.0) / n_units * 1e6,
            "incl_us": incl.get(key, 0.0) / n_units * 1e6,
            "code": src[file][line - 1].strip()[:90],
        })
    with open(out, "w") as fh:
        json.dump({
            "rows": rows, "n_units": n_units, "plain_us": plain / n_units * 1e6,
            "profiled_us": profiled / n_units * 1e6,
            "args": sys.argv[2:], **env_info(),
        }, fh)
    info = env_info()
    print(info["pandas"], "pyarrow" if info["pyarrow_loaded"] else "no-pyarrow", sys.argv[2:],
          f"plain {plain / n_units * 1e6:.0f} us/unit, profiled {profiled / n_units * 1e6:.0f}, "
          f"sum own {sum(r['own_us'] for r in rows):.0f}, lines {len(rows)}")
