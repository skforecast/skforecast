# Unit test set_cpu_gpu_device
# ==============================================================================
import pytest
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from skforecast.utils import set_cpu_gpu_device


@pytest.mark.parametrize("estimator, initial_device, new_device",
    [(XGBRegressor(), "cpu", "gpu"),
     (XGBRegressor(), "cuda", "cpu"),
     (XGBRegressor(), "cuda:0", "cpu"),
     (LGBMRegressor(), "gpu", "cpu"),
     (LGBMRegressor(), "cpu", "cuda")],
    ids=["XGB-cpu-to-gpu", "XGB-cuda-to-cpu", "XGB-cuda:0-to-cpu",
         "LGBM-gpu-to-cpu", "LGBM-cpu-to-cuda"]
)
def test_set_cpu_gpu_device_changes_device(estimator, initial_device, new_device):
    """
    Test that the device is set as passed, without translating it to another
    name, and that the previous device is returned.
    """
    estimator.set_params(**{'device': initial_device})
    original = set_cpu_gpu_device(estimator, new_device)

    assert original == initial_device
    assert estimator.get_params()['device'] == new_device


@pytest.mark.parametrize("estimator, initial_device",
    [(XGBRegressor(), "cuda:0"),
     (XGBRegressor(), "gpu"),
     (LGBMRegressor(), "cuda")],
    ids=["XGB-cuda:0", "XGB-gpu", "LGBM-cuda"]
)
def test_set_cpu_gpu_device_restores_original_device(estimator, initial_device):
    """
    Test that passing the value returned by a previous call restores the
    original device verbatim.
    """
    estimator.set_params(**{'device': initial_device})
    original = set_cpu_gpu_device(estimator, "cpu")

    assert estimator.get_params()['device'] == "cpu"

    set_cpu_gpu_device(estimator, original)

    assert estimator.get_params()['device'] == initial_device


@pytest.mark.parametrize("estimator, new_device, expected_new_device",
    [(XGBRegressor(), "gpu", "gpu"),
     (XGBRegressor(), "cpu", None),
     (LGBMRegressor(), "gpu", "gpu"),
     (LGBMRegressor(), "cpu", None)],
    ids=["XGB-gpu", "XGB-cpu", "LGBM-gpu", "LGBM-cpu"]
)
def test_set_cpu_gpu_device_when_device_not_set(estimator, new_device, expected_new_device):
    """
    Test that `None` is returned when the device of the estimator is not set.
    Setting 'cpu' leaves it unset, since both libraries already use the CPU.
    """
    original_device = set_cpu_gpu_device(estimator, new_device)

    assert original_device is None
    assert estimator.get_params().get('device') == expected_new_device


def test_set_cpu_gpu_device_no_change_if_same():
    """
    Test that the device is not changed if the same device is passed.
    """
    estimator = XGBRegressor(device="cuda")
    _ = set_cpu_gpu_device(estimator, "cuda")

    assert estimator.get_params()['device'] == "cuda"


@pytest.mark.parametrize("estimator",
    [LinearRegression(),
     CatBoostRegressor(task_type="GPU", verbose=0, allow_writing_files=False)],
    ids=lambda est: type(est).__name__
)
def test_set_cpu_gpu_device_unsupported_model_returns_none(estimator):
    """
    Test that the function returns None and does not modify the estimator
    when the model is not supported.
    """
    params = estimator.get_params()
    result = set_cpu_gpu_device(estimator, "cpu")

    assert result is None
    assert estimator.get_params() == params


def test_set_cpu_gpu_device_none_device_returns_current():
    """
    Test that the function returns the current device when None is passed.
    """
    estimator = XGBRegressor(device="cuda")
    original = set_cpu_gpu_device(estimator, None)

    assert original == "cuda"
    assert estimator.get_params()['device'] == "cuda"
