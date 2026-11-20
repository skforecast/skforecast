"""P4: pd.Index(categories_[i]).get_indexer(values) vs OrdinalEncoder.transform,
with the encoder configured as in the forecasters
(dtype=float, handle_unknown='use_encoded_value', unknown_value=nan,
encoded_missing_value=nan, set_output('pandas'))."""
import _env
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

warnings.simplefilter("ignore")


def make_encoder():
    return OrdinalEncoder(
        dtype=float, handle_unknown="use_encoded_value",
        unknown_value=np.nan, encoded_missing_value=np.nan,
    ).set_output(transform="pandas")


def proto_values(enc, s):
    """Proposal as written in the study (micro_direct.py)."""
    codes = pd.Index(enc.categories_[0]).get_indexer(s.to_numpy()).astype(float)
    codes[codes < 0] = np.nan
    return codes


def proto_cat_lookup(enc, s):
    """'By codes' variant: map the categories once, then take by code."""
    lk = pd.Index(enc.categories_[0]).get_indexer(s.cat.categories).astype(float)
    lk[lk < 0] = np.nan
    lk = np.append(lk, np.nan)  # code -1 -> NaN
    return lk[s.cat.codes.to_numpy()]


def proto_raw_codes(enc, s):
    """Wrong shortcut: use `.cat.codes` directly."""
    c = s.cat.codes.to_numpy().astype(float)
    c[c < 0] = np.nan
    return c


def run(fn):
    try:
        return np.asarray(fn(), dtype=float)
    except Exception as e:
        return f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"


def same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str)
    return np.array_equal(a, b, equal_nan=True)


S = pd.Series
C = pd.Categorical
cases = {
    "int64 values": (S([3, 1, 2, 1]), S([1, 2, 3, 4])),
    "int64 fit, float64 predict": (S([3, 1, 2, 1]), S([1.0, 2.0, 2.5, np.nan])),
    "float64 with NaN in fit": (S([0.5, np.nan, 1.5]), S([1.5, np.nan, 0.5, 9.0])),
    "float64 no NaN in fit, NaN at predict": (S([0.5, 1.5]), S([1.5, np.nan])),
    "bool values": (S([True, False, True]), S([False, True])),
    "bool fit, int predict (1 == True)": (S([True, False]), S([1, 0, 2])),
    "int fit, bool predict": (S([0, 1, 2]), S([True, False])),
    "int fit, str-digit predict": (S([1, 2]), S(["1", "2"], dtype=object)),
    "object str": (S(["b", "a", "c"], dtype=object), S(["a", "z", "c"], dtype=object)),
    "object str + None at fit": (S(["b", None, "a"], dtype=object), S(["a", None, "z"], dtype=object)),
    "object str + np.nan at fit": (S(["b", np.nan, "a"], dtype=object), S(["a", np.nan, None, "z"], dtype=object)),
    "object str, None/nan/pd.NA only at predict": (S(["b", "a"], dtype=object), S(["a", None, np.nan, pd.NA], dtype=object)),
    "inferred text dtype (object in p2, str in p3) + None at fit": (S(["b", None, "a"]), S(["a", None, "z"])),
    "inferred text dtype, None only at predict": (S(["b", "a"]), S(["a", None, "z"])),
    "object mixed int/str": (S([1, "a", 2], dtype=object), S([1, "a"], dtype=object)),
    "object with 1, 1.0, True (equal under ==)": (S([1, 2, 3], dtype=object), S([1.0, True, 2], dtype=object)),
    "numpy str_ values (U dtype array -> Series)": (S(np.array(["b", "a"])), S(np.array(["a", "b", "q"]))),
    "pandas 'str'/string dtype": (S(["b", "a", None], dtype="string"), S(["a", None, "q"], dtype="string")),
    "object fit, string dtype predict": (S(["b", "a"], dtype=object), S(["a", pd.NA, "q"], dtype="string")),
    "string dtype fit, object predict": (S(["b", "a"], dtype="string"), S(["a", None, "q"], dtype=object)),
    "category str, sorted": (S(C(["b", "a", "c"])), S(C(["a", "c", "b"], categories=["a", "b", "c"]))),
    "category, categories NOT sorted": (S(C(["b", "a", "c"], categories=["c", "b", "a"])),
                                         S(C(["a", "c", "b"], categories=["c", "b", "a"]))),
    "category, other order at predict": (S(C(["b", "a", "c"], categories=["a", "b", "c"])),
                                          S(C(["a", "c", "b"], categories=["c", "a", "b"]))),
    "category, extra category at predict": (S(C(["b", "a"], categories=["a", "b"])),
                                             S(C(["a", "z", "b"], categories=["a", "b", "z"]))),
    "category, unused category at fit": (S(C(["b", "a"], categories=["a", "b", "x"])),
                                          S(C(["a", "b", "x"], categories=["a", "b", "x"]))),
    "category with NaN (code -1)": (S(C(["b", None, "a"])), S(C(["a", None, "b"]))),
    "category int": (S(C([10, 30, 20])), S(C([30, 10, 99]))),
    "category int fit, plain int predict": (S(C([10, 30, 20])), S([30, 10, 99])),
    "category int fit, float predict": (S(C([10, 30, 20])), S([30.0, 10.0, np.nan])),
    "category float incl NaN": (S(C([0.5, np.nan, 1.5])), S(C([1.5, np.nan, 0.5]))),
    "datetime values": (S(pd.to_datetime(["2020-01-02", "2020-01-01"])), S(pd.to_datetime(["2020-01-01", "2021-01-01"]))),
    "Int64 nullable with NA": (S([1, 2, None], dtype="Int64"), S([2, None, 5], dtype="Int64")),
    "inf values": (S([np.inf, 1.0, -np.inf]), S([np.inf, -np.inf, 1.0])),
    "-0.0 vs 0.0": (S([0.0, 1.0]), S([-0.0, 1.0])),
    "large ints > 2**53": (S([2**53, 2**53 + 1]), S([2**53 + 1, 2**53])),
    "int fit, float near-int predict": (S([2**53, 2**53 + 1]), S([float(2**53)])),
    "tuple-like/unsortable object": (S([(1, 2), "a"], dtype=object), S(["a"], dtype=object)),
}

n_diff = 0
for name, (fit_s, pred_s) in cases.items():
    enc = make_encoder()
    try:
        enc.fit(fit_s.to_frame("c"))
    except Exception as e:
        print(f"[fit error ] {name}: {type(e).__name__}: {str(e)[:70]}")
        continue
    ref = run(lambda: enc.transform(pred_s.to_frame("c"))["c"].to_numpy())
    out = [("get_indexer(values)", run(lambda: proto_values(enc, pred_s)))]
    if isinstance(pred_s.dtype, pd.CategoricalDtype):
        out.append(("lookup by categories", run(lambda: proto_cat_lookup(enc, pred_s))))
        out.append(("raw .cat.codes", run(lambda: proto_raw_codes(enc, pred_s))))
    flags = [(lab, same(ref, o)) for lab, o in out]
    tag = "OK  " if all(ok for _, ok in flags) else "DIFF"
    print(f"[{tag}] {name}  | categories_={list(enc.categories_[0])}")
    if tag == "DIFF":
        n_diff += 1
        print(f"         OrdinalEncoder      : {ref}")
        for (lab, o), (_, ok) in zip(out, flags):
            if not ok:
                print(f"         {lab:20s}: {o}")
print("cases with some difference:", n_diff)
print(_env.versions())
