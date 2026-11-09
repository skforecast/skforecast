import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from skforecast.datasets import load_demo_dataset
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import OneStepAheadFold, TimeSeriesFold, grid_search_forecaster_multiseries
from skforecast.metrics import mean_absolute_scaled_error
import importlib.util
from pathlib import Path


def _load(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_paths = _load("BUG-mase-window-size_03_paths.py")
find_slice, ENDS, wide, as_dict, n_train = (
    _paths.find_slice, _paths.ENDS, _paths.wide, _paths.as_dict, _paths.n_train
)
M = ["mean_absolute_error", mean_absolute_scaled_error]
print("| Path | input | level | MASE | implied scale | y_train slice (iloc of the span) |")
print("|---|---|---|---|---|---|")
for cvname, cv in [("grid multiseries OSA", lambda: OneStepAheadFold(initial_train_size=n_train)),
                   ("grid multiseries TimeSeriesFold", lambda: TimeSeriesFold(steps=12, initial_train_size=n_train, refit=False))]:
    for dname, data in [("wide", wide), ("dict", as_dict)]:
        for lvl in ["s1", "s3"]:
            r = grid_search_forecaster_multiseries(
                forecaster=ForecasterRecursiveMultiSeries(Ridge(), lags=24), series=data, levels=[lvl],
                cv=cv(), param_grid={"alpha": [1.0]}, metric=M, return_best=False, n_jobs=1,
                verbose=False, show_progress=False)
            mae, mase = r.filter(like="mean_absolute_error").iat[0, 0], r.filter(like="mean_absolute_scaled").iat[0, 0]
            print(f"| {cvname} | {dname} | {lvl} | {mase:.4f} | {mae/mase:.5f} | {find_slice(wide[lvl], mae/mase, ENDS)[:40]} |")
# pooled check
d = [np.diff(wide["s1"].to_numpy()[24:168]), np.diff(wide["s2"].to_numpy()[24:168]), np.diff(wide["s3"].to_numpy()[30:168])]
print("pooled scale, s1,s2 without 24 + s3 full:", round(np.mean(np.abs(np.concatenate(d))), 5))
d = [np.diff(wide["s1"].to_numpy()[24:168]), np.diff(wide["s2"].to_numpy()[24:168]), np.diff(wide["s3"].to_numpy()[54:168])]
print("pooled scale, all without first 24 obs:", round(np.mean(np.abs(np.concatenate(d))), 5))
