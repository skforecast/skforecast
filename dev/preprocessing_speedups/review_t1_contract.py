"""
Review of T1, points 2 to 4: contract of `StandardScaler.transform` and
`inverse_transform`, and behavior of the shortcut in the edge cases.

Three candidates (`review_t1_common.py`) are compared with the current
`transform_numpy`, by bytes (`tobytes`, stricter than `assert_array_equal`,
which does not tell -0.0 from 0.0), dtype, shape, exception (type and message)
and warnings:

- PLAN: T1 exactly as written in PLAN.md.
- PROTO: `prototypes.fast_transform_numpy`.
- PROPOSED: the conditions proposed in REVIEW_T1.md, with the decided inf check.
- PROPOSED-B: the same conditions, looking for inf in the input before computing.

Run it in the main environment and in the one with the minimum versions:

python dev/preprocessing_speedups/review_t1_contract.py
~/.venvs/skforecast-min/bin/python dev/preprocessing_speedups/review_t1_contract.py
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pandas as pd
import sklearn
from prototypes import fast_transform_numpy
from review_t1_common import (
    ORIG_TRANSFORM_NUMPY,
    env_line,
    plan_literal,
    proposed,
    proposed_isinf_first,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CANDIDATES = {"PLAN": plan_literal, "PROTO": fast_transform_numpy, "PROPOSED": proposed,
              "PROPOSED-B": proposed_isinf_first}


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def outcome(func, array, transformer, **kw):
    """Result, or exception, and warnings of a call. The input must not change."""
    before = array.copy() if hasattr(array, "copy") else array
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            res = func(array=array, transformer=transformer, **kw)
            if isinstance(res, np.ndarray):
                out = ("ok", type(res).__name__, str(res.dtype), res.shape,
                       np.ascontiguousarray(res).tobytes(),
                       (res.flags.c_contiguous, res.flags.f_contiguous, res.flags.writeable))
                shares = np.shares_memory(res, array) if isinstance(array, np.ndarray) else False
            else:
                out = ("ok", type(res).__name__, "", getattr(res, "shape", None), repr(res), ())
                shares = False
        except Exception as exc:
            out = ("raise", type(exc).__name__, str(exc))
            shares = False
    warns = tuple(sorted({(w.category.__name__, str(w.message)) for w in rec}))
    unchanged = True
    if isinstance(array, np.ndarray) and array.dtype != object:
        unchanged = np.ascontiguousarray(array).tobytes() == np.ascontiguousarray(before).tobytes()
    return out, warns, unchanged, shares


def describe(out):
    if out[0] == "ok":
        body = out[4]
        digest = hashlib.sha256(body if isinstance(body, bytes) else body.encode()).hexdigest()[:8]
        flags = "" if len(out) < 6 else f" flags(C,F,W)={tuple(int(v) for v in out[5])}"
        return f"{out[1]} {out[2]} {out[3]} sha={digest}{flags}"
    msg = out[2].replace("\n", " ")
    return f"{out[1]}: {msg[:110]}"


N_CHECKED = 0
DIFFS = []


def compare(label, array, transformer, verbose=False, **kw):
    """Compare each candidate with the current `transform_numpy`."""
    global N_CHECKED
    ref, ref_w, ref_unchanged, ref_shares = outcome(ORIG_TRANSFORM_NUMPY, array, transformer, **kw)
    row = {}
    for name, func in CANDIDATES.items():
        out, w, unchanged, shares = outcome(func, array, transformer, **kw)
        problems = []
        if out != ref:
            problems.append("result")
        if w != ref_w:
            problems.append("warnings")
        if not unchanged:
            problems.append("INPUT MODIFIED")
        if shares != ref_shares:
            problems.append("SHARES MEMORY")
        row[name] = (out, w, problems)
        N_CHECKED += 1
        if problems:
            DIFFS.append((label, name, problems))
    if verbose or any(p for _, _, p in row.values()) or not ref_unchanged:
        print(f"\n[{label}]")
        print(f"   current : {describe(ref)}" + (f" | warnings: {ref_w}" if ref_w else "")
              + ("" if ref_unchanged else " | INPUT MODIFIED")
              + (" | shares memory with the input" if ref_shares else ""))
        for name, (out, w, problems) in row.items():
            tag = "same" if not problems else "DIFFERENT (" + ", ".join(problems) + ")"
            line = f"   {name:10s}: {tag}"
            if problems:
                line += f" -> {describe(out)}" + (f" | warnings: {w}" if w != ref_w else "")
            print(line)
    return ref


def fitted(data, names=False, **params):
    data = np.asarray(data)
    sc = StandardScaler(**params)
    if names:
        sc.fit(pd.DataFrame({"y": data}))
    else:
        sc.fit(data.reshape(-1, 1))
    return sc


def hand_set(mean=3.141592653589793, scale=1.7320508075688772, **params):
    """Fitted scaler with fixed attributes: the same state in every version."""
    sc = StandardScaler(**params).fit(np.array([[0.0], [1.0]]))
    if sc.mean_ is not None:
        sc.mean_ = np.array([mean])
    if sc.scale_ is not None:
        sc.scale_ = np.array([scale])
    return sc


def main():
    print(env_line())
    rng = np.random.default_rng(123)
    train = 50 + rng.normal(0, 3, 500).cumsum()

    # ------------------------------------------------------------------ 1
    print("\n===== 1. Fitted state of StandardScaler (attributes, dtype, zero variance)")
    fits = {
        "default": fitted(train),
        "with_mean=False": fitted(train, with_mean=False),
        "with_std=False": fitted(train, with_std=False),
        "with_mean=False, with_std=False": fitted(train, with_mean=False, with_std=False),
        "fitted with a DataFrame": fitted(train, names=True),
        "copy=False": fitted(train.copy(), copy=False),
        "constant series": fitted(np.full(50, 7.5)),
        "almost constant series": fitted(1e6 + np.array([0.0, 1e-10] * 25)),
        "one observation": fitted(np.array([4.2])),
        "series with NaN": fitted(np.where(np.arange(500) % 7 == 0, np.nan, train)),
        "all NaN": fitted(np.full(20, np.nan)),
        "float32 series": fitted(train.astype(np.float32)),
        "int64 series": fitted(np.arange(100)),
        "values of 1e200 (variance overflows)": fitted(train * 1e200),
        "values of 1e-200 (variance underflows)": fitted(train * 1e-200),
        "set_output(transform='pandas')": fitted(train).set_output(transform="pandas"),
    }
    two_calls = StandardScaler()
    two_calls.partial_fit(train[:250].reshape(-1, 1))
    two_calls.partial_fit(train[250:].reshape(-1, 1))
    fits["partial_fit twice"] = two_calls
    for name, sc in fits.items():
        attrs = sorted(k for k in vars(sc) if k.endswith("_") and not k.startswith("_"))
        mean, scale = sc.mean_, sc.scale_
        with np.errstate(all="ignore"):
            print(
                f"{name:40s} mean_={None if mean is None else (mean.dtype.name, mean.shape, float(mean[0]))} "
                f"scale_={None if scale is None else (scale.dtype.name, scale.shape, float(scale[0]))} "
                f"attrs={attrs}"
            )
    print("state digest (default fit):", sha(fits["default"].mean_, fits["default"].scale_))

    # ------------------------------------------------------------------ 2
    print("\n===== 2. Bitwise identity on float64 (every candidate against the current path)")
    inputs_1d = {}
    for n in (1, 2, 3, 7, 8, 15, 16, 17, 48, 168, 1000):
        inputs_1d[f"normal n={n}"] = 50 + rng.normal(0, 5, n)
    special = np.array([0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1e300, -1e300,
                        fits["default"].mean_[0], np.nextafter(fits["default"].mean_[0], np.inf),
                        3.141592653589793, 1.7320508075688772, 1.0, -1.0, 1e-17, 0.1, 1 / 3])
    inputs_1d["special values"] = special
    with_nan = 50 + rng.normal(0, 5, 168)
    with_nan[::5] = np.nan
    inputs_1d["with NaN"] = with_nan
    inputs_1d["all NaN"] = np.full(48, np.nan)
    base = 50 + rng.normal(0, 5, (168, 6))
    inputs_1d["column of a C matrix (not contiguous)"] = base[:, 2]
    inputs_1d["reversed (negative stride)"] = base[::-1, 1]
    ro = 50 + rng.normal(0, 5, 100)
    ro.flags.writeable = False
    inputs_1d["read-only"] = ro
    inputs_1d["read-only view of pandas"] = pd.Series(50 + rng.normal(0, 5, 100)).to_numpy()

    cube = rng.normal(0, 2, (24, 5, 50))
    inputs_2d = {
        "C (48, 250)": rng.normal(0, 2, (48, 250)),
        "F (48, 250)": np.asfortranarray(rng.normal(0, 2, (48, 250))),
        "slice of a cube [:, i, :] (not contiguous)": cube[:, 2, :],
        "(24, 3) conformal": rng.normal(0, 2, (24, 3)),
        "(10, 1)": rng.normal(0, 2, (10, 1)),
        "(1, 7)": rng.normal(0, 2, (1, 7)),
        "(1, 1)": rng.normal(0, 2, (1, 1)),
        "2D with NaN": np.where(rng.random((24, 30)) < 0.2, np.nan, rng.normal(0, 2, (24, 30))),
        "transposed view": rng.normal(0, 2, (30, 24)).T,
    }
    ro2 = rng.normal(0, 2, (24, 10))
    ro2.flags.writeable = False
    inputs_2d["2D read-only"] = ro2

    scalers = dict(fits)
    for wm in (True, False):
        for ws in (True, False):
            scalers[f"hand-set with_mean={wm}, with_std={ws}"] = hand_set(with_mean=wm, with_std=ws)

    digest_fit, digest_fixed = hashlib.sha256(), hashlib.sha256()
    per_scaler = {name: hashlib.sha256() for name in scalers}
    before = len(DIFFS)
    n_cases = 0
    with np.errstate(all="ignore"):
        for sname, sc in scalers.items():
            for iname, arr in inputs_1d.items():
                for inverse in (False, True):
                    ref = compare(f"{sname} | {iname} | inverse={inverse}", arr, sc,
                                  fit=False, inverse_transform=inverse)
                    n_cases += 1
                    if ref[0] == "ok":
                        (digest_fixed if sname.startswith("hand-set") else digest_fit).update(ref[4])
                        per_scaler[sname].update(ref[4])
                        # The current path must equal the direct call to scikit-learn
                        method = sc.inverse_transform if inverse else sc.transform
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            direct = np.asarray(method(arr.copy().reshape(-1, 1))).ravel()
                        assert direct.tobytes() == ref[4], (sname, iname, inverse)
            for iname, arr in inputs_2d.items():
                ref = compare(f"{sname} | 2D {iname} | inverse=True", arr, sc,
                              fit=False, inverse_transform=True)
                n_cases += 1
                if ref[0] == "ok":
                    (digest_fixed if sname.startswith("hand-set") else digest_fit).update(ref[4])
                    per_scaler[sname].update(ref[4])
                # Transform of a 2D array: outside the shortcut (current path)
                compare(f"{sname} | 2D {iname} | inverse=False", arr, sc,
                        fit=False, inverse_transform=False)
                n_cases += 1
    print(f"\n{n_cases} cases x {len(CANDIDATES)} candidates, "
          f"{len(DIFFS) - before} differences with the current path.")
    print("digest of the outputs of the current path, by scaler (to compare between environments):")
    for sname, h in per_scaler.items():
        state = b"".join(v.tobytes() for v in (scalers[sname].mean_, scalers[sname].scale_)
                         if v is not None)
        print(f"   {sname:45s} state={hashlib.sha256(state).hexdigest()[:10]} "
              f"outputs={h.hexdigest()[:10]}")
    print("digest of the outputs, scalers fitted in this environment :", digest_fit.hexdigest()[:16])
    print("digest of the outputs, scalers with hand-set attributes   :", digest_fixed.hexdigest()[:16])

    # ------------------------------------------------------------------ 3
    print("\n===== 3. float32 (outside the shortcut): what scikit-learn returns")
    x32 = (50 + rng.normal(0, 5, 1000)).astype(np.float32)
    sc = hand_set()
    out_t = sc.transform(x32.reshape(-1, 1).copy()).ravel()
    out_i = sc.inverse_transform(x32.reshape(-1, 1).copy()).ravel()
    print("dtype of transform / inverse:", out_t.dtype, out_i.dtype)
    print("digest float32 transform:", sha(out_t), "| inverse:", sha(out_i))
    in64 = x32.astype(np.float64)
    via64 = ((in64 - sc.mean_[0]) / sc.scale_[0]).astype(np.float32)
    in32 = x32.copy()
    in32 -= sc.mean_.astype(np.float32)
    in32 /= sc.scale_.astype(np.float32)
    print(f"transform equals 'compute in float64 and round' : {out_t.tobytes() == via64.tobytes()} "
          f"({int((out_t != via64).sum())} of {x32.size} values differ)")
    print(f"transform equals 'cast mean_/scale_ to float32'  : {out_t.tobytes() == in32.tobytes()} "
          f"({int((out_t != in32).sum())} of {x32.size} values differ)")
    for name, arr in {"float32 1D": x32[:48], "float32 2D": x32[:96].reshape(24, 4)}.items():
        for inverse in (False, True):
            compare(f"{name} | inverse={inverse}", arr, fits["default"], verbose=True,
                    fit=False, inverse_transform=inverse)

    # ------------------------------------------------------------------ 4
    print("\n===== 4. Infinite values: same ValueError as today")
    sc = fits["default"]
    a = 50 + rng.normal(0, 5, 48)
    cases = {
        "+inf": np.where(np.arange(48) == 5, np.inf, a),
        "-inf": np.where(np.arange(48) == 5, -np.inf, a),
        "+inf and -inf": np.where(np.arange(48) == 5, np.inf, np.where(np.arange(48) == 9, -np.inf, a)),
        "inf and NaN": np.where(np.arange(48) == 5, np.inf, np.where(np.arange(48) % 4 == 0, np.nan, a)),
        "inf in the last position": np.r_[a[:-1], np.inf],
    }
    for name, arr in cases.items():
        for inverse in (False, True):
            compare(f"{name} | 1D | inverse={inverse}", arr, sc, verbose=True,
                    fit=False, inverse_transform=inverse)
    m = rng.normal(0, 2, (24, 10))
    m[3, 4] = np.inf
    compare("+inf | 2D | inverse=True", m, sc, verbose=True, fit=False, inverse_transform=True)
    for name, s2 in {"with_mean=False, with_std=False": fits["with_mean=False, with_std=False"],
                     "scale_=nan (values of 1e200)": fits["values of 1e200 (variance overflows)"],
                     "fitted with a DataFrame": fits["fitted with a DataFrame"]}.items():
        for inverse in (False, True):
            with np.errstate(all="ignore"):
                compare(f"+inf | scaler {name} | inverse={inverse}", cases["+inf"], s2,
                        verbose=True, fit=False, inverse_transform=inverse)
    with sklearn.config_context(assume_finite=True):
        for inverse in (False, True):
            compare(f"+inf with sklearn assume_finite=True | inverse={inverse}", cases["+inf"], sc,
                    verbose=True, fit=False, inverse_transform=inverse)
    # Message of the direct call (the current path only forwards it)
    for inverse in (False, True):
        method = sc.inverse_transform if inverse else sc.transform
        try:
            method(cases["+inf"].reshape(-1, 1))
        except ValueError as exc:
            print(f"direct scikit-learn, inverse={inverse}: ValueError: {str(exc)[:90]}")
    print("\n-- Finite values whose sum overflows (scikit-learn ignores the overflow of its sum)")
    huge = np.full(48, 1.5e308)
    for inverse in (False, True):
        compare(f"finite 1.5e308 x 48 | inverse={inverse}", huge, hand_set(mean=0.0, scale=1.0),
                verbose=True, fit=False, inverse_transform=inverse)
    compare("finite, the result overflows to inf | inverse=True", np.full(8, 1e308),
            hand_set(mean=0.0, scale=100.0), verbose=True, fit=False, inverse_transform=True)

    # ------------------------------------------------------------------ 5
    print("\n===== 5. Cases that must follow the current path (or fail as today)")
    sc = fits["default"]
    named = fits["fitted with a DataFrame"]
    x = 50 + rng.normal(0, 5, 24)

    class MyScaler(StandardScaler):
        def transform(self, X, copy=None):
            return super().transform(X, copy=copy) + 1000.0

        def inverse_transform(self, X, copy=None):
            return super().inverse_transform(X - 1000.0, copy=copy)

    class Tagged(np.ndarray):
        pass

    two_features = StandardScaler().fit(rng.normal(0, 1, (50, 2)))
    refit_params = fitted(train, with_std=False).set_params(with_std=True)
    pipe = Pipeline([("s", StandardScaler())]).fit(train.reshape(-1, 1))
    edge = [
        ("scaler fitted with feature names (ForecasterRecursive, Direct, Stats, Rnn)", x, named),
        ("unfitted StandardScaler", x, StandardScaler()),
        ("unfitted StandardScaler(with_mean=False, with_std=False)", x,
         StandardScaler(with_mean=False, with_std=False)),
        ("subclass of StandardScaler", x, MyScaler().fit(train.reshape(-1, 1))),
        ("Pipeline([StandardScaler])", x, pipe),
        ("set_output(transform='pandas')", x, fits["set_output(transform='pandas')"]),
        ("scaler with 2 features, 1D array", x, two_features),
        ("with_std changed to True after fit (scale_ is None)", x, refit_params),
        ("empty 1D array", np.empty(0), sc),
        ("pandas Series float64 as `array`", pd.Series(x), sc),
        ("list as `array`", list(x), sc),
        ("masked array", np.ma.masked_invalid(np.where(np.arange(24) == 3, np.nan, x)), sc),
        ("ndarray subclass", x.view(Tagged), sc),
        ("0-d array", np.array(3.5), sc),
        ("int64 array", np.arange(24), sc),
        ("float16 array", x.astype(np.float16), sc),
        ("bool array", np.arange(24) % 2 == 0, sc),
        ("object array of floats", x.astype(object), sc),
        ("big-endian float64", x.astype(">f8"), sc),
        ("complex array", x.astype(complex), sc),
    ]
    for label, arr, tr in edge:
        for inverse in (False, True):
            with np.errstate(all="ignore"):
                compare(f"{label} | inverse={inverse}", arr, tr, verbose=True,
                        fit=False, inverse_transform=inverse)
    edge_inverse = [
        ("empty 2D (0, 3)", np.empty((0, 3)), sc),
        ("empty 2D (3, 0)", np.empty((3, 0)), sc),
        ("3D (4, 3, 2)", rng.normal(0, 1, (4, 3, 2)), sc),
        ("3D (4, 1, 2)", rng.normal(0, 1, (4, 1, 2)), sc),
        ("np.matrix (6, 4)", np.asmatrix(rng.normal(0, 1, (6, 4))), sc),
        ("scaler with 2 features, (10, 2)", rng.normal(0, 1, (10, 2)), two_features),
        ("scaler with 2 features, (10, 3)", rng.normal(0, 1, (10, 3)), two_features),
    ]
    for label, arr, tr in edge_inverse:
        compare(f"{label} | inverse=True", arr, tr, verbose=True, fit=False, inverse_transform=True)
    with sklearn.config_context(transform_output="pandas"):
        for inverse in (False, True):
            compare(f"sklearn transform_output='pandas' | inverse={inverse}", x, sc, verbose=True,
                    fit=False, inverse_transform=inverse)
    # fit=True never uses the shortcut and leaves the scaler fitted as today
    for label, tr in [("fit=True, new scaler", StandardScaler()), ("fit=True, fitted scaler", fitted(train))]:
        compare(label, x, tr, verbose=True, fit=True, inverse_transform=False)
    compare("transformer=None returns the same object", x, None, verbose=True)
    assert plan_literal(x, None) is x and proposed(x, None) is x and fast_transform_numpy(x, None) is x

    # ------------------------------------------------------------------ 6
    print(f"\n===== Summary: {N_CHECKED} comparisons")
    for name in CANDIDATES:
        labels = [f"{lab} ({', '.join(p)})" for lab, cand, p in DIFFS if cand == name]
        print(f"\n{name}: {len(labels)} differences with the current `transform_numpy`")
        for lab in labels:
            print(f"   - {lab}")


if __name__ == "__main__":
    main()
