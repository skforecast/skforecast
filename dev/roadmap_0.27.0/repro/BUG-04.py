"""
BUG-04: a pre-loaded pipeline/model passed to FoundationModel (e.g. a
fine-tuned Chronos-2) is silently discarded by ForecasterFoundation.

Part A runs offline with fake pipelines (no download needed): a "fine-tuned"
pipeline predicts 999 and the "base" pipeline that the adapter reloads from
`model_id` predicts 0 (BaseChronosPipeline.from_pretrained is patched).

Part B (optional) uses the real Chronos-2 small checkpoint: it fine-tunes it
for a few steps with `BaseChronosPipeline.fit` (chronos-forecasting >= 2.1)
and compares predictions. It is skipped if chronos or the checkpoint is not
available. Set BUG04_SKIP_REAL=1 to skip it.

Run: python dev/roadmap_0.27.0/repro/BUG-04.py
"""

import os
import sys
import tempfile
import warnings
from unittest import mock

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import skforecast
from skforecast.foundation import FoundationModel, ForecasterFoundation
from skforecast.model_selection import TimeSeriesFold, backtesting_foundation
from skforecast.utils import deepcopy_forecaster

print(f"skforecast {skforecast.__version__}, python {sys.version.split()[0]}")

rng = np.random.default_rng(0)
n = 400
idx = pd.date_range("2020-01-01", periods=n, freq="D")
y = pd.Series(
    10 + 3 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 1, n),
    index=idx,
    name="y",
)


class FakePipeline:
    """Minimal stand-in for a Chronos-2 pipeline returning a constant."""

    def __init__(self, value):
        self.value = value

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        n_q = len(quantile_levels)
        arr = np.full((1, prediction_length, n_q), float(self.value))
        mean = np.full((1, prediction_length), float(self.value))
        return [arr] * len(inputs), [mean] * len(inputs)


def first(pred):
    return pred["pred"].iloc[0]


print("\n" + "=" * 72)
print("PART A: offline, fake pipelines (fine-tuned = 999, base reloaded = 0)")
print("=" * 72)

finetuned = FakePipeline(999.0)
base_loads = []


def fake_from_pretrained(model_id, **kwargs):
    base_loads.append(model_id)
    return FakePipeline(0.0)


# Make the base reload work offline even if chronos is not installed
try:
    import chronos  # noqa: F401
    patcher = mock.patch(
        "chronos.BaseChronosPipeline.from_pretrained", side_effect=fake_from_pretrained
    )
except ImportError:
    fake_mod = mock.MagicMock()
    fake_mod.BaseChronosPipeline.from_pretrained.side_effect = fake_from_pretrained
    patcher = mock.patch.dict(sys.modules, {"chronos": fake_mod})

results = {}
with patcher:
    fm = FoundationModel("autogluon/chronos-2-small", pipeline=finetuned, device_map="cpu")
    results["A1"] = (fm.adapter._pipeline is finetuned, True)

    # A2: FoundationModel used directly honours the pre-loaded pipeline
    p_fm = first(fm.fit(series=y).predict(steps=7))
    results["A2"] = (p_fm, 999.0)

    # A3: ForecasterFoundation.__init__ clones the estimator -> pipeline lost
    fm2 = FoundationModel("autogluon/chronos-2-small", pipeline=finetuned, device_map="cpu")
    forecaster = ForecasterFoundation(fm2)
    results["A3"] = (forecaster.estimator.adapter._pipeline, "<FakePipeline 999>")

    # A4: predict through the forecaster -> base model reloaded, no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        forecaster.fit(series=y)
        p_f = first(forecaster.predict(steps=7))
        n_warn = len(w)
    results["A4"] = (p_f, 999.0)
    results["A4b"] = (f"{len(base_loads)} reload(s) of {base_loads}, {n_warn} warnings", "0 reloads")

    # A5: backtesting_foundation (no copy, but forecaster already lost it)
    cv = TimeSeriesFold(steps=7, initial_train_size=300, refit=False)
    _, bt_pred = backtesting_foundation(
        forecaster=forecaster, series=y, cv=cv, metric="mean_absolute_error",
        show_progress=False,
    )
    results["A5"] = (sorted(bt_pred["pred"].unique().tolist()), [999.0])

    # A6: workaround used by the test fixture (assign private attribute after
    # construction) is lost again by deepcopy_forecaster, which is what
    # bayesian_search_foundation calls on the forecaster
    forecaster.estimator.adapter._pipeline = finetuned
    copy_ = deepcopy_forecaster(forecaster)
    results["A6"] = (copy_.estimator.adapter._pipeline, "<FakePipeline 999>")

    # A7: get_params exposes no `pipeline` key, so clone cannot keep it
    results["A7"] = ("pipeline" in fm.get_params(), True)

labels = {
    "A1": "FoundationModel(pipeline=ft).adapter._pipeline is ft",
    "A2": "FoundationModel.predict first value",
    "A3": "ForecasterFoundation(fm).estimator.adapter._pipeline",
    "A4": "ForecasterFoundation.predict first value",
    "A4b": "Base checkpoint reloads during forecaster predict",
    "A5": "backtesting_foundation unique predictions",
    "A6": "deepcopy_forecaster(...).estimator.adapter._pipeline",
    "A7": "'pipeline' in FoundationModel.get_params()",
}
for k, (obs, exp) in results.items():
    print(f"[{k}] {labels[k]}\n      observed: {obs!r}\n      expected: {exp!r}")

bug_a = (
    results["A2"][0] == 999.0
    and results["A3"][0] is None
    and results["A4"][0] == 0.0
    and results["A6"][0] is None
)
print(f"\nPART A reproduces the bug: {bug_a}")

print("\n" + "=" * 72)
print("PART B: real Chronos-2 small fine-tuned with BaseChronosPipeline.fit")
print("=" * 72)

if os.environ.get("BUG04_SKIP_REAL") == "1":
    print("Skipped (BUG04_SKIP_REAL=1)")
    sys.exit(0)

try:
    import torch
    from chronos import BaseChronosPipeline
    torch.manual_seed(0)
    base = BaseChronosPipeline.from_pretrained("autogluon/chronos-2-small", device_map="cpu")
    if not hasattr(base, "fit"):
        raise RuntimeError("chronos-forecasting without fine-tuning support")
except Exception as exc:  # pragma: no cover
    print(f"Skipped: {type(exc).__name__}: {exc}")
    sys.exit(0)

with tempfile.TemporaryDirectory() as tmp:
    ft = base.fit(
        inputs=[y.to_numpy()[:-30]], prediction_length=7, num_steps=30,
        learning_rate=1e-3, batch_size=8, output_dir=tmp,
        remove_printer_callback=True,
    )

fm_real = FoundationModel("autogluon/chronos-2-small", pipeline=ft, device_map="cpu")
p_fm = fm_real.fit(series=y).predict(steps=7)["pred"].to_numpy()

f_real = ForecasterFoundation(
    FoundationModel("autogluon/chronos-2-small", pipeline=ft, device_map="cpu")
)
print("Pipeline survives ForecasterFoundation init:",
      f_real.estimator.adapter._pipeline is ft)
f_real.fit(series=y)
p_f = f_real.predict(steps=7)["pred"].to_numpy()

fm_base = FoundationModel("autogluon/chronos-2-small", pipeline=base, device_map="cpu")
p_base = fm_base.fit(series=y).predict(steps=7)["pred"].to_numpy()

print("FoundationModel(pipeline=ft)          :", np.round(p_fm, 3))
print("ForecasterFoundation(FM(pipeline=ft)) :", np.round(p_f, 3), "(observed)")
print("Base checkpoint                       :", np.round(p_base, 3))
print("Expected forecaster == fine-tuned     :", np.round(p_fm, 3))
print("Forecaster == base?      ", np.allclose(p_f, p_base, atol=1e-4))
print("Forecaster == fine-tuned?", np.allclose(p_f, p_fm, atol=1e-4))
print("Max |fine-tuned - base|  :", round(float(np.max(np.abs(p_fm - p_base))), 3))
