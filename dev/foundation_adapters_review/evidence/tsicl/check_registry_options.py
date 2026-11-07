# Registry prefixes, capability attributes and two backend options that the
# adapter does not pass (`allow_covar_forecast`, `batch_size`), checked for
# behaviour only (no accuracy or speed comparison).
#
# Usage (from the repository root):
#     ~/.venvs/fm-review-tsicl/bin/python dev/foundation_adapters_review/evidence/tsicl/check_registry_options.py
import inspect
import platform
import warnings
from importlib.metadata import version

import numpy as np
import torch

from skforecast.foundation import get_model_info
from skforecast.foundation._adapters import _resolve_adapter

warnings.simplefilter("ignore")
print(
    f"python={platform.python_version()} tsicl={version('tsicl')} "
    f"torch={torch.__version__} numpy={np.__version__} skforecast={version('skforecast')}"
)

print("\n== registry")
for model_id in ["taharnbl/TS-ICL", "taharnbl/ts-icl", "EDF-Lab/TS-ICL", "taharnbl/TS-ICL-v2"]:
    try:
        print(f"{model_id!r} -> {_resolve_adapter(model_id).__name__}")
    except ValueError as exc:
        print(f"{model_id!r} -> ValueError: {str(exc)[:80]}")

print("\n== get_model_info('taharnbl/TS-ICL')")
print(get_model_info("taharnbl/TS-ICL"))

from tsicl import TSICL  # noqa: E402

print("\n== TSICL.forecast signature")
print(inspect.signature(TSICL.forecast))
print("\n== TSICL.__init__ signature")
print(inspect.signature(TSICL.__init__))

model = TSICL()
print(f"\nmax_context_length={model.max_context_length} max_target_length={model.max_target_length}")
tf = model.forecaster.tf_icl
print(f"quantile grid: start={tf.start_quantile} end={tf.end_quantile} n={tf.nb_quantiles}")
print(f"_get_device('mps') without CUDA -> {model._get_device(torch.device('mps'))}")

rng = np.random.default_rng(0)
t = np.arange(600 + 24)
drv = (np.sin(2 * np.pi * t / 37.0) + rng.normal(0, 0.05, len(t))).astype(np.float32)
y = (30 + 6 * drv + rng.normal(0, 0.3, len(t))).astype(np.float32)
item_past_only = {"target": y[:600], "past_covariates": {"drv": drv[:600]}}
item_none = {"target": y[:600]}


def q(inputs, **kwargs):
    _, out = model.forecast(
        inputs=inputs, prediction_length=24, quantile_levels=[0.5],
        context_length=512, device=torch.device("cpu"), denormalize=True,
        squeeze_output=False, **kwargs,
    )
    return out[0][0].numpy()


base = q([item_none])
p_false = q([item_past_only])
p_true = q([item_past_only], allow_covar_forecast=True)
print("\n== allow_covar_forecast with past-only covariates")
print(f"allow_covar_forecast=False vs no covariates: max_abs_diff={np.max(np.abs(p_false - base)):.3e}")
print(f"allow_covar_forecast=True  vs no covariates: max_abs_diff={np.max(np.abs(p_true - base)):.3e}")

print("\n== output type of forecast(squeeze_output=False) for list inputs")
_, same = model.forecast(inputs=[item_none, item_none], prediction_length=24, quantile_levels=[0.1, 0.5],
                         context_length=512, device=torch.device("cpu"), squeeze_output=False)
_, diff = model.forecast(inputs=[item_none, {"target": y[:300]}], prediction_length=24, quantile_levels=[0.1, 0.5],
                         context_length=512, device=torch.device("cpu"), squeeze_output=False)
print(f"same lengths: {type(same).__name__} shape={tuple(same.shape) if hasattr(same, 'shape') else [tuple(x.shape) for x in same]}")
print(f"different lengths: {type(diff).__name__} shapes={[tuple(x.shape) for x in diff]}")
