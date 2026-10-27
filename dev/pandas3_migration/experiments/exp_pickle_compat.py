"""
Pickle compatibility of pandas objects across versions (what joblib relies on).

Step 1 (save, in each version):
    bash dev/pandas3_migration/run_in_both.sh experiments/exp_pickle_compat.py save
Step 2 (load the files of the other version, in each version):
    bash dev/pandas3_migration/run_in_both.sh experiments/exp_pickle_compat.py load

Files go to results/pickles/pandas<version>/. skforecast's own forecasters are
covered by identity_crossload.py (joblib and skops backends).
"""

import os
import pickle
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "results", "pickles")


def objects():
    return {
        "date_range_D": pd.date_range("2020-01-01", periods=5, freq="D"),
        "date_range_D_ns": pd.date_range("2020-01-01", periods=5, freq="D").as_unit("ns"),
        "date_range_nofreq": pd.DatetimeIndex(pd.date_range("2020-01-01", periods=5, freq="D").to_numpy()),
        "date_range_h": pd.date_range("2020-01-01", periods=5, freq="h"),
        "date_range_MS": pd.date_range("2020-01-01", periods=5, freq="MS"),
        "range_index_series": pd.Series([1.0, 2.0], index=pd.RangeIndex(2)),
        "series_str": pd.Series(["a", "b"]),
        "df_cat": pd.DataFrame({"c": pd.Categorical(["a", "b"])}),
        "df_float_dt_index": pd.DataFrame({"v": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2, freq="D")),
        "offset_Day": pd.offsets.Day(1),
        "offset_Hour": pd.offsets.Hour(1),
    }


def save():
    d = os.path.join(ROOT, f"pandas{pd.__version__}")
    os.makedirs(d, exist_ok=True)
    for k, v in objects().items():
        with open(os.path.join(d, k + ".pkl"), "wb") as fh:
            pickle.dump(v, fh, protocol=5)
    print("saved with pandas", pd.__version__, "->", d)


def load():
    mine = f"pandas{pd.__version__}"
    for folder in sorted(os.listdir(ROOT)):
        if folder == mine:
            continue
        print(f"--- loading pickles written by {folder} in pandas {pd.__version__}")
        d = os.path.join(ROOT, folder)
        for f in sorted(os.listdir(d)):
            try:
                with open(os.path.join(d, f), "rb") as fh:
                    v = pickle.load(fh)
                print(f"  {f:26s} OK   {type(v).__name__} {getattr(v, 'dtype', '')} freq={getattr(v, 'freq', '')}")
            except Exception as e:  # noqa: BLE001
                print(f"  {f:26s} FAIL {type(e).__name__}: {str(e)[:90]}")


if __name__ == "__main__":
    {"save": save, "load": load}[sys.argv[1]]()
