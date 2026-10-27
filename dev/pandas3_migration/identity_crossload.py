"""
Cross-version persistence: load the forecasters saved by identity_cases.py
in the OTHER pandas version (results/persisted/pandas<version>/) with joblib
and skops, predict, and compare the fingerprints with the predictions of the
same forecaster saved and loaded in the current version.

    /root/.venvs/skforecast/bin/python dev/pandas3_migration/identity_crossload.py --other 3.1.0
    PYTHONPATH=. /root/.venvs/skforecast-p3/bin/python dev/pandas3_migration/identity_crossload.py --other 2.3.3
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import describe, env_info, fingerprint, result_path, save_json  # noqa: E402
from identity_cases import STEPS, make_data, split  # noqa: E402


def _lw_index_dtype(fc):
    lw = getattr(fc, "last_window_", None)
    if lw is None:
        return None
    if isinstance(lw, dict):
        return {k: str(v.index.dtype) for k, v in lw.items()}
    return str(lw.index.dtype)


def main(other: str, index_unit: str | None):
    from skforecast.utils import load_forecaster
    d = make_data(index_unit)
    s = split(d)
    own_dir = os.path.join(HERE, "results", "persisted", f"pandas{pd.__version__}")
    other_dir = os.path.join(HERE, "results", "persisted", f"pandas{other}")
    out = {"env": env_info(), "other_pandas": other, "index_unit": index_unit, "cases": {}}
    for backend, ext in (("joblib", "joblib"), ("skops", "skops")):
        for name, kw in (
            ("recursive", {"steps": STEPS, "exog": s["exog_text_test"]}),
            ("multiseries", {"steps": STEPS, "exog": s["exog_dict_test"]}),
        ):
            rec = {}
            for label, folder in (("own", own_dir), ("other", other_dir)):
                path = os.path.join(folder, f"{name}.{ext}")
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    try:
                        fc = load_forecaster(path, backend=backend, trusted=True, verbose=False)
                        pred = fc.predict(**kw)
                        rec[label] = {
                            "strict": fingerprint(pred), "normalized": fingerprint(pred, strict=False),
                            "describe": describe(pred),
                            "exog_dtypes_in_": {k: str(v) for k, v in fc.exog_dtypes_in_.items()},
                            "index_freq_": str(getattr(fc, "index_freq_", None)),
                            "last_window_index_dtype": _lw_index_dtype(fc),
                            "skforecast_version_in_file": getattr(fc, "skforecast_version", None),
                            "error": None,
                        }
                    except Exception:  # noqa: BLE001
                        rec[label] = {"error": traceback.format_exc()[-1500:]}
                rec[label]["warnings"] = sorted({f"{x.category.__name__}: {str(x.message)[:140]}" for x in w})
            same_strict = rec["own"].get("strict") == rec["other"].get("strict")
            same_norm = rec["own"].get("normalized") == rec["other"].get("normalized")
            rec["summary"] = ("ERROR" if rec["other"].get("error") else
                              "identical" if same_strict else
                              "values identical, dtype/unit differ" if same_norm else "DIFFERENT VALUES")
            out["cases"][f"{backend}_{name}"] = rec
            print(f"{backend:7s} {name:12s} file from pandas {other}: {rec['summary']}")
            if rec["other"].get("error"):
                print("   ", "\n    ".join(rec["other"]["error"].splitlines()[-12:]))
            for wmsg in rec["other"]["warnings"]:
                print("    warning:", wmsg[:160])
            if not rec["other"].get("error"):
                print("    exog_dtypes_in_ own:", rec["own"]["exog_dtypes_in_"], "| other:", rec["other"]["exog_dtypes_in_"])
                print("    last_window index own:", rec["own"]["last_window_index_dtype"], "| other:", rec["other"]["last_window_index_dtype"])
    path = result_path("identity_crossload", f"from{other}" + (f"_{index_unit}" if index_unit else ""))
    save_json(path, out)
    print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", required=True)
    ap.add_argument("--index-unit", default=None)
    a = ap.parse_args()
    main(a.other, a.index_unit)
