# Unit test _lazy_njit
# ==============================================================================
import os
import subprocess
import sys
from pathlib import Path
import pytest
import numpy as np
from skforecast.preprocessing._preprocessing import (
    _lazy_njit,
    _np_mean_jit,
    _np_std_jit,
    _np_min_jit,
    _np_max_jit,
    _np_sum_jit,
    _np_median_jit,
    _np_min_max_ratio_jit,
    _np_cv_jit,
    _ewm_jit,
    _n_unique_jit,
    _n_changes_jit,
)

# Fixtures
x = np.array([1., 2., 3., 4., 5.])
x_classes = np.array([1, 1, 2, 2, 3])


@pytest.mark.parametrize(
    "func, X, kwargs, expected",
    [
        (_np_mean_jit, x, {}, 3.0),
        (_np_std_jit, x, {}, 1.5811388300841898),
        (_np_min_jit, x, {}, 1.0),
        (_np_max_jit, x, {}, 5.0),
        (_np_sum_jit, x, {}, 15.0),
        (_np_median_jit, x, {}, 3.0),
        (_np_min_max_ratio_jit, x, {}, 0.2),
        (_np_cv_jit, x, {}, 0.5270462766947299),
        (_ewm_jit, x, {}, 3.67678771050449),
        (_ewm_jit, x, {"alpha": 0.5}, 4.161290322580645),
        (_n_unique_jit, x_classes, {}, 3),
        (_n_changes_jit, x_classes, {}, 2),
    ],
    ids=[
        "mean", "std", "min", "max", "sum", "median", "min_max_ratio", "cv",
        "ewm", "ewm_alpha_0.5", "n_unique", "n_changes",
    ],
)
def test_lazy_njit_output_of_jit_functions(func, X, kwargs, expected):
    """
    Test that the lazily compiled functions return the expected values and
    accept keyword arguments.
    """
    result = func(X, **kwargs)

    assert np.isclose(result, expected)


def test_lazy_njit_compiles_once_on_first_call(monkeypatch):
    """
    Test that _lazy_njit does not compile at decoration time, compiles with
    numba.njit on the first call and reuses the compiled function afterwards.
    """
    import numba

    calls = []
    original_njit = numba.njit

    def njit_spy(func):
        calls.append(func.__name__)
        return original_njit(func)

    monkeypatch.setattr(numba, "njit", njit_spy)

    @_lazy_njit
    def add_one(v):
        return v + 1

    assert add_one.__name__ == "add_one"
    assert calls == []
    assert add_one(1) == 2
    assert add_one(2) == 3
    assert calls == ["add_one"]


def test_importing_forecasters_does_not_import_numba():
    """
    Test that importing the forecaster modules does not import numba. It is
    deferred to the first use of RollingFeatures. A subprocess is needed
    because the test session already has numba loaded.
    """
    repo_root = Path(__file__).resolve().parents[4]
    code = (
        "import sys; "
        "import skforecast.recursive, skforecast.direct, "
        "skforecast.preprocessing, skforecast.model_selection; "
        "print('numba' in sys.modules)"
    )
    env = {**os.environ, "PYTHONPATH": str(repo_root)}

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False"], result.stdout
