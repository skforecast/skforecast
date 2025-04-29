# Unit test set_cpu_gpu_device
# ==============================================================================
import re
import pytest
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from skforecast.utils import set_cpu_gpu_device


@pytest.mark.parametrize("regressor, initial_device, new_device, expected_new_device", 
    [(XGBRegressor(), "cpu", "gpu", "cuda"),
     (XGBRegressor(), "cuda", "cpu", "cpu"),
     (LGBMRegressor(), "gpu", "cpu", "cpu"),
     (LGBMRegressor(), "cpu", "gpu", "gpu")],
)
def test_set_cpu_gpu_device_changes_device(regressor, initial_device, new_device, expected_new_device):
    """
    Test that the device is changed correctly when a new device is passed.
    """
    regressor.set_params(**{'device': initial_device})
    original = set_cpu_gpu_device(regressor, new_device)

    assert original == initial_device
    assert regressor.get_params()['device'] == expected_new_device


def test_set_cpu_gpu_device_no_change_if_same():
    """
    Test that the device is not changed if the same device is passed.
    """
    regressor = XGBRegressor(device="cuda")
    _ = set_cpu_gpu_device(regressor, "cuda")

    assert regressor.get_params()['device'] == "cuda"


def test_set_cpu_gpu_device_invalid_device():
    """
    Test that an error is raised when an invalid device is passed.
    """
    regressor = LGBMRegressor()

    msg = re.escape("`device` must be 'gpu', 'cpu', 'cuda', or None.")
    with pytest.raises(ValueError, match=msg):
        set_cpu_gpu_device(regressor, "tpu")


@pytest.mark.parametrize("regressor, new_device, expected_original_device, expected_new_device", 
    [(XGBRegressor(), "gpu", None, "cuda"),
     (XGBRegressor(), "cpu", None, "cpu"),
     (LGBMRegressor(), "gpu", None, "gpu"),
     (LGBMRegressor(), "cpu", None, "cpu")],
)
def test_set_cpu_gpu_device_when_device_not_in_init(regressor, new_device, expected_original_device, expected_new_device):
    """
    Test what happens when the device is not in the init of the regressor.
    """

    original_device = set_cpu_gpu_device(regressor, new_device)

    assert original_device == expected_original_device
    assert regressor.get_params()['device'] == expected_new_device


def test_set_cpu_gpu_device_unsupported_model_returns_none():
    """
    Test that the function returns None when the model is not supported.
    """
    regressor = LinearRegression()
    result = set_cpu_gpu_device(regressor, "cpu")

    assert result is None


def test_set_cpu_gpu_device_none_device_returns_current():
    """
    Test that the function returns the current device when None is passed.
    """
    regressor = XGBRegressor(device="cuda")
    original = set_cpu_gpu_device(regressor, None)

    assert original == "cuda"