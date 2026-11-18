"""
Review of T1, point 3: forecasters saved with joblib, pickle and skops, by this
version and by previous releases of skforecast.

1. `make` fits and saves the forecasters. Run it with the current code and, in
   a separate environment, with a previous release (it only needs the public
   API, not the review helpers):

   python dev/preprocessing_speedups/review_t1_saved.py make <dir_current>
   <python of the old environment> dev/preprocessing_speedups/review_t1_saved.py make <dir_old>

2. `check` loads them with the current code, reports the state of their
   scalers and compares the predictions of the current `transform_numpy` with
   those of the candidates, by bytes:

   python dev/preprocessing_speedups/review_t1_saved.py check <dir_current> <dir_old>
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N = 300
# Backend of `save_forecaster` and the extension it writes
BACKENDS = {"joblib": "joblib", "pickle": "pkl", "skops": "skops"}


def data():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=N, freq="h")
    y = pd.Series(50 + rng.normal(0, 3, N).cumsum(), index=idx, name="y")
    series = pd.DataFrame(
        {f"s{i}": 20 * (i + 1) + rng.normal(0, 2, N).cumsum() for i in range(3)}, index=idx
    )
    return y, series


def build():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    from skforecast.direct import ForecasterDirect, ForecasterDirectMultiVariate
    from skforecast.recursive import ForecasterRecursive, ForecasterRecursiveMultiSeries

    y, series = data()
    out = {}
    f = ForecasterRecursive(Ridge(), lags=5, transformer_y=StandardScaler())
    f.fit(y, store_in_sample_residuals=True)
    out["recursive"] = f
    f = ForecasterDirect(Ridge(), lags=5, steps=5, transformer_y=StandardScaler())
    f.fit(y, store_in_sample_residuals=True)
    out["direct"] = f
    f = ForecasterRecursiveMultiSeries(Ridge(), lags=5, transformer_series=StandardScaler())
    f.fit(series, store_in_sample_residuals=True)
    out["multiseries"] = f
    f = ForecasterRecursiveMultiSeries(
        Ridge(), lags=5, encoding="onehot",
        transformer_series={"s0": StandardScaler(), "s1": StandardScaler(with_std=False),
                            "s2": None, "_unknown_level": StandardScaler()},
    )
    f.fit(series, store_in_sample_residuals=True)
    out["multiseries_dict"] = f
    f = ForecasterDirectMultiVariate(Ridge(), level="s0", lags=5, steps=5,
                                     transformer_series=StandardScaler())
    f.fit(series, store_in_sample_residuals=True)
    out["multivariate"] = f
    return out


def predictions(f):
    """Point predictions and intervals, as arrays."""
    out = {}
    kw = {"levels": None} if type(f).__name__ == "ForecasterRecursiveMultiSeries" else {}
    for name, call in {
        "predict": lambda: f.predict(steps=5, **kw),
        "conformal": lambda: f.predict_interval(steps=5, method="conformal", **kw),
        "bootstrapping": lambda: f.predict_interval(steps=5, method="bootstrapping",
                                                    n_boot=25, **kw),
    }.items():
        res = call()
        res = res.select_dtypes("number") if isinstance(res, pd.DataFrame) else res
        out[name] = np.ascontiguousarray(res.to_numpy(dtype=float))
    return out


def make(folder):
    import sklearn

    import skforecast
    from skforecast.utils import save_forecaster

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"skforecast": skforecast.__version__, "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "python": sys.version.split()[0], "files": {}}
    for name, f in build().items():
        preds = {k: v.tolist() for k, v in predictions(f).items()}
        for backend, extension in BACKENDS.items():
            file = folder / f"{name}.{extension}"
            try:
                try:
                    save_forecaster(f, file_name=str(file), backend=backend, verbose=False)
                except TypeError:
                    if backend != "joblib":
                        raise
                    save_forecaster(f, file_name=str(file), verbose=False)
                meta["files"][file.name] = preds
            except Exception as exc:
                print(f"[not saved] {file.name}: {type(exc).__name__}: {str(exc)[:120]}")
    (folder / "meta.json").write_text(json.dumps(meta))
    print(f"saved {len(meta['files'])} files in {folder} with skforecast {meta['skforecast']}, "
          f"scikit-learn {meta['scikit-learn']}, numpy {meta['numpy']}, pandas {meta['pandas']}")


def scalers(f):
    if getattr(f, "transformer_series_", None) is not None:
        return dict(f.transformer_series_)
    return {"transformer_y": f.transformer_y}


def check(folders):
    from prototypes import fast_transform_numpy
    from review_t1_common import (
        ORIG_TRANSFORM_NUMPY,
        Counted,
        env_line,
        patched,
        plan_applies,
        plan_literal,
        proposed,
        proposed_applies,
        prototype_conditions,
    )
    from sklearn.preprocessing import StandardScaler

    from skforecast.utils import load_forecaster

    print("loading with:", env_line())
    candidates = {
        "PLAN": (plan_literal, plan_applies),
        "PROTO": (fast_transform_numpy, prototype_conditions),
        "PROPOSED": (proposed, proposed_applies),
    }
    for folder in map(Path, folders):
        meta = json.loads((folder / "meta.json").read_text())
        print(f"\n##### {folder.name}: saved with skforecast {meta['skforecast']}, "
              f"scikit-learn {meta['scikit-learn']}, numpy {meta['numpy']}, "
              f"pandas {meta['pandas']}, Python {meta['python']}")
        for file_name, saved_preds in meta["files"].items():
            try:
                kw = {"trusted": True} if file_name.endswith(".skops") else {}
                f = load_forecaster(str(folder / file_name), verbose=False, **kw)
            except Exception as exc:
                print(f"{file_name:26s} [not loaded] {type(exc).__name__}: {str(exc)[:150]}")
                continue
            state = {}
            for key, t in scalers(f).items():
                if t is None:
                    continue
                state.setdefault(
                    f"exact type={type(t) is StandardScaler}, "
                    f"names={hasattr(t, 'feature_names_in_')}, "
                    f"n_features_in_={getattr(t, 'n_features_in_', None)}, "
                    f"mean_={None if t.mean_ is None else (type(t.mean_).__name__, t.mean_.dtype.name, t.mean_.shape)}, "
                    f"scale_={None if t.scale_ is None else (type(t.scale_).__name__, t.scale_.dtype.name, t.scale_.shape)}",
                    [],
                ).append(key)
            with patched(ORIG_TRANSFORM_NUMPY):
                ref = predictions(f)
            same_as_saved = all(
                np.array(saved_preds[k]).tobytes() == ref[k].tobytes() for k in ref
            )
            cells = []
            for label, (func, applies) in candidates.items():
                counted = Counted(func, applies)
                with patched(counted):
                    got = predictions(f)
                equal = all(got[k].tobytes() == ref[k].tobytes() for k in ref)
                cells.append(f"{label}: {'identical' if equal else 'DIFFERENT'}, "
                             f"shortcut in {counted.hits}/{counted.calls} calls")
            print(f"{file_name:26s} " + " | ".join(cells)
                  + f" | equal to the predictions of the saving version: {same_as_saved}")
            for line, keys in state.items():
                print(f"{'':26s}   {', '.join(keys)}: {line}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    if len(sys.argv) < 3 or sys.argv[1] not in ("make", "check"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "make":
        make(sys.argv[2])
    else:
        check(sys.argv[2:])
