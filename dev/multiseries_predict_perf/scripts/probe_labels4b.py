"""
Change 4b label probe: the exog of a level is aligned with the predictions
in `_create_predict_inputs`. Before 4b, an exog whose index or columns are not
`identical` to the targets goes to `reindex`; with 4b, if its index `equals`
the index of the predictions (the check `Index.reindex` makes) and its columns
are `identical`, it is a shallow copy with the index of the predictions and
the training columns.

Compares, errors included, the frame of each level returned by:
- `base`: the code of change 4 (`1587f1069`),
- `new`: the code of 4b (and `new_len`, with a `len` guard first, measured
  and not implemented),
- `reindex`: a plain `reindex(index=pi, columns=cols)`,
on values, dtypes, index and columns (`identical`, type, `freq`), `attrs` and
`flags`, and checks that the user's object is not modified. Column names and
column index types as in `probe_labels7.py` (the training columns are built as
the package builds them). The columns are also replaced on the 4b path:
`identical` columns can have other labels (`None` and NaN in pandas 2, 176
differences without it); the user's index is the index of the predictions,
named, without `freq`, with another equal `freq`, as `object`, with another
unit, time zone or type, longer, shorter or shifted. It does not import
skforecast.

Usage: [NO_PYARROW=1] python probe_labels4b.py (about 10 to 15 minutes per
environment; the three can run at the same time, they are not timings).
Results in `MEDIDAS.md`, section 13.
"""
import os
import sys

if os.environ.get("NO_PYARROW") == "1":
    class _BlockPyarrow:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ModuleNotFoundError("No module named 'pyarrow'", name="pyarrow")
            return None
    sys.meta_path.insert(0, _BlockPyarrow)

import collections
import time
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
try:
    import pyarrow  # noqa: F401
    HAS_PA = True
except ModuleNotFoundError:
    HAS_PA = False

ts = pd.Timestamp
LABELS = {
    "str": ["a", "b", "c"],
    "int": [1, 2, 3],
    "int_range0": [0, 1, 2],
    "int_desc": [3, 2, 1],
    "int_one": [7],
    "np_int": [np.int64(1), np.int64(2), np.int64(3)],
    "float": [1.0, 2.0, 3.0],
    "float_nan": [1.0, np.nan, 3.0],
    "bool": [True, False],
    "bool_int": [True, 2, 3],
    "obj_int_str": [1, "b", "c"],
    "obj_none": ["a", None, "c"],
    "obj_pdna": ["a", pd.NA, "c"],
    "np_str": [np.str_("a"), np.str_("b"), np.str_("c")],
    "tz_utc": [ts("2020-01-01", tz="UTC"), ts("2020-01-02", tz="UTC")],
    "naive_ts": [ts("2020-01-01"), ts("2020-01-02")],
    "tuples": [("x", 1), ("x", 2), ("y", 1)],
}
KINDS = {
    "plain": lambda labels: pd.Index(labels),
    "object": lambda labels: pd.Index(labels, dtype=object),
    "category": lambda labels: pd.CategoricalIndex(labels),
    "string_python": lambda labels: pd.Index(labels, dtype="string[python]"),
    "range": lambda labels: pd.RangeIndex(len(labels)),
}
if HAS_PA:
    KINDS["string_pyarrow"] = lambda labels: pd.Index(labels, dtype="string[pyarrow]")
    KINDS["str_pyarrow"] = lambda labels: pd.Index(
        labels, dtype=pd.StringDtype("pyarrow", na_value=np.nan))

PRED_INDEXES = {
    "date_D": pd.date_range("2021-02-10", periods=3, freq="D"),
    "date_D_us": pd.date_range("2021-02-10", periods=3, freq="D", unit="us"),
    "date_h_tz": pd.date_range("2021-02-10", periods=3, freq="h", tz="Europe/Madrid"),
    "date_MS": pd.date_range("2021-02-01", periods=3, freq="MS"),
    "date_one": pd.date_range("2021-02-10", periods=1, freq="h"),
    "range": pd.RangeIndex(40, 43),
}


def user_indexes(pi):
    """Indexes of the user's exog, keyed by name. Some fail to build."""
    out = {
        "same": lambda: pi,
        "copy": lambda: pi.copy(),
        "named": lambda: pi.rename("date"),
        "named_none_str": lambda: pi.rename("None"),
        "object": lambda: pd.Index(list(pi), dtype=object),
        "object_named": lambda: pd.Index(list(pi), dtype=object, name="date"),
        "longer": lambda: pi.append(pi[-1:] + (pi[-1] - pi[-2] if len(pi) > 1 else 1)),
        "shorter": lambda: pi[:-1] if len(pi) > 1 else pi[:0],
        "earlier": lambda: (pi - (pi[1] - pi[0] if len(pi) > 1 else 1)),
        "reversed": lambda: pi[::-1],
        "categorical": lambda: pd.CategoricalIndex(pi),
        "multi": lambda: pd.MultiIndex.from_arrays([pi]),
    }
    if isinstance(pi, pd.DatetimeIndex):
        out.update({
            "nofreq": lambda: pd.DatetimeIndex(pi, freq=None),
            "nofreq_named": lambda: pd.DatetimeIndex(pi, freq=None, name="date"),
            "other_freq": lambda: pd.date_range(pi[0], periods=len(pi), freq=(
                "24h" if pi.freqstr == "D" else "60min" if pi.freqstr == "h" else None)),
            "unit_ns": lambda: pi.as_unit("ns"),
            "unit_us": lambda: pi.as_unit("us"),
            "unit_s": lambda: pi.as_unit("s"),
            "tz_utc": lambda: (pi.tz_convert("UTC") if pi.tz is not None
                               else pi.tz_localize("UTC")),
            "tz_naive": lambda: (pi.tz_localize(None) if pi.tz is not None else pi),
            "period": lambda: pi.to_period(),
            "str": lambda: pd.Index(pi.astype(str)),
        })
    else:
        out.update({
            "int64": lambda: pd.Index(np.asarray(pi), dtype="int64"),
            "int32": lambda: pd.Index(np.asarray(pi), dtype="int32"),
            "uint64": lambda: pd.Index(np.asarray(pi), dtype="uint64"),
            "float": lambda: pd.Index(np.asarray(pi), dtype="float64"),
            "Int64": lambda: pd.Index(np.asarray(pi), dtype="Int64"),
            "range_named": lambda: pi.rename("date"),
            "range_other": lambda: pd.RangeIndex(pi.start, pi.stop + 2, 1)[:len(pi)],
        })
    return out


def user_data(n_rows, n_cols, flavour, rng):
    """Columns of several dtypes (`mixed`) or all float."""
    if flavour == "float":
        return {i: rng.normal(size=n_rows) for i in range(n_cols)}
    makers = [
        lambda: rng.normal(size=n_rows),
        lambda: pd.array(rng.integers(0, 5, n_rows), dtype="Int64"),
        lambda: pd.Categorical(rng.integers(0, 3, n_rows)),
        lambda: rng.integers(0, 5, n_rows),
        lambda: pd.array([True, None] * n_rows, dtype="boolean")[:n_rows],
        lambda: np.array(["x", "y"] * n_rows, dtype=object)[:n_rows],
    ]
    return {i: makers[i % len(makers)]() for i in range(n_cols)}


def training_columns(cols):
    """Columns as the package builds them (change 7), from the training dtypes."""
    train = pd.DataFrame(np.zeros((2, len(cols))))
    train.columns = cols
    dtypes = train.dtypes.to_dict()
    return pd.DataFrame(data={col: [] for col, _ in dtypes.items()}, copy=False).columns


def base(ev, pi, cols):
    if not (
        ev.index.name is pi.name
        and ev.index.identical(pi)
        and ev.columns.identical(cols)
    ):
        ev = ev.reindex(index=pi, columns=cols)
    return ev, "fast" if ev is not None else None


def new(ev, pi, cols, len_guard=False):
    path = "fast"
    if not (
        ev.index.name is pi.name
        and ev.index.identical(pi)
        and ev.columns.identical(cols)
    ):
        if (
            (not len_guard or len(ev) == len(pi))
            and ev.index.equals(pi)
            and ev.columns.identical(cols)
        ):
            ev = ev.copy(deep=False)
            ev.index = pi
            ev.columns = cols
            path = "4b"
        else:
            ev = ev.reindex(index=pi, columns=cols)
            path = "reindex"
    return ev, path


def plain_reindex(ev, pi, cols):
    return ev.reindex(index=pi, columns=cols), "reindex"


def compare(a, b):
    if type(a) is not type(b):
        return "type"
    if not (a.index.identical(b.index) and type(a.index) is type(b.index)):
        return "index"
    if getattr(a.index, "freq", None) != getattr(b.index, "freq", None):
        return "freq"
    if not a.columns.identical(b.columns) or type(a.columns) is not type(b.columns):
        return "columns"
    if [str(x) for x in a.dtypes] != [str(x) for x in b.dtypes]:
        return "dtypes"
    if a.attrs != b.attrs or a.flags != b.flags:
        return "attrs/flags"
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True, check_freq=True,
                                      check_index_type=True, check_column_type=True,
                                      check_flags=True)
    except AssertionError as e:
        return f"values {e}"[:200]
    return None


def run(fn, uv, pi, cols):
    snapshot = (uv.index, uv.columns, uv.copy(deep=True))
    try:
        out, path = fn(uv, pi, cols)
        err = None
    except Exception as e:  # noqa: BLE001
        out, path, err = None, "error", f"{type(e).__name__}: {e}"
    if not (uv.index is snapshot[0] and uv.columns is snapshot[1]
            and uv.equals(snapshot[2])):
        err = f"user object modified ({err})"
    return out, path, err


CORE_USER_INDEXES = {"same", "named", "nofreq"}
CANDIDATES = {"new": new, "new_len": lambda e, p, c: new(e, p, c, len_guard=True)}
REFERENCES = {"base": base, "reindex": plain_reindex}
T0 = time.perf_counter()
diffs = collections.defaultdict(list)
paths = collections.Counter()
n_cases = 0
for lname, labels in LABELS.items():
    for kname, kind in KINDS.items():
        try:
            tcols = kind(labels)
        except Exception:
            continue
        cols = training_columns(tcols)
        for pname, pi in PRED_INDEXES.items():
            for uname, make_uix in user_indexes(pi).items():
                try:
                    uix = make_uix()
                except Exception:
                    continue
                for ukname, ukind in KINDS.items():
                    # Column kinds are crossed with the key user indexes;
                    # the other user indexes only with the training kind.
                    if uname not in CORE_USER_INDEXES and ukname != kname:
                        continue
                    try:
                        ucols = ukind(labels)
                    except Exception:
                        continue
                    for flavour in ("float", "mixed"):
                        rng = np.random.default_rng(0)
                        user = pd.DataFrame(
                            user_data(len(uix), len(ucols), flavour, rng), index=uix
                        )
                        user.columns = ucols
                        variants = {
                            "all": user, "reversed": user.iloc[:, ::-1],
                            "first": user.iloc[:, :1],
                        }
                        if flavour == "float":
                            variants["attrs"] = user.copy()
                            variants["attrs"].attrs = {"source": "file"}
                            try:
                                variants["no_dups"] = user.set_flags(
                                    allows_duplicate_labels=False)
                            except Exception:
                                pass
                        for vname, uv in variants.items():
                            n_cases += 1
                            key = (lname, kname, pname, uname, ukname, flavour, vname)
                            ref = {r: run(f, uv, pi, cols) for r, f in REFERENCES.items()}
                            for cname, fn in CANDIDATES.items():
                                got, path, err = run(fn, uv, pi, cols)
                                paths[(cname, path)] += 1
                                if cname == "new" and path == "4b":
                                    paths[("4b_by_user_index", uname)] += 1
                                for rname, (exp, _, exp_err) in ref.items():
                                    if exp_err or err:
                                        if exp_err != err:
                                            diffs[(cname, rname)].append(
                                                key + ("err", exp_err, err))
                                        continue
                                    r = compare(exp, got)
                                    if r:
                                        diffs[(cname, rname)].append(key + (r,))

print(f"pandas {pd.__version__} pyarrow={HAS_PA} infer_string="
      f"{pd.get_option('future.infer_string')}")
print(f"cases {n_cases} in {time.perf_counter() - T0:.0f} s")
for k in sorted(k for k in paths if k[0] in CANDIDATES):
    print("  path", k, paths[k])
print("  4b path by user index:",
      dict(sorted((k[1], v) for k, v in paths.items() if k[0] == "4b_by_user_index")))
for cname in CANDIDATES:
    for rname in REFERENCES:
        d = diffs[(cname, rname)]
        print(f"{cname} vs {rname}: {len(d)} differences")
        kinds = collections.Counter((x[3], x[7]) for x in d)
        for (uname, what), count in kinds.most_common(8):
            ex = next(x for x in d if x[3] == uname and x[7] == what)
            print(f"    {count:5d} user index {uname}, {what}: {ex}"[:300])
