# Unit test _resolve_n_jobs
# ==============================================================================
import re
from unittest.mock import patch
import pytest
import numpy as np
from skforecast.utils import _resolve_n_jobs


@pytest.mark.parametrize(
    "n_jobs",
    [1.0, '2', None, np.int64(2), np.int64(-2), np.int64(0)],
    ids=lambda n_jobs: f'n_jobs: {n_jobs!r}'
)
def test_resolve_n_jobs_TypeError_when_n_jobs_not_int(n_jobs):
    """
    Test TypeError is raised when `n_jobs` is not a Python int, including NumPy
    integers, which the input checks of backtesting do not accept either.
    """
    err_msg = re.escape(
        f"`n_jobs` must be an integer or `'auto'`. Got {type(n_jobs)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        _resolve_n_jobs(n_jobs=n_jobs)


@pytest.mark.parametrize(
    "n_jobs",
    [0, False],
    ids=lambda n_jobs: f'n_jobs: {n_jobs!r}'
)
def test_resolve_n_jobs_ValueError_when_n_jobs_is_0(n_jobs):
    """
    Test ValueError is raised when `n_jobs` is 0, which joblib does not allow
    (`False` is 0).
    """
    err_msg = re.escape(
        f"`n_jobs` must be a non-zero integer or 'auto'. Got {n_jobs}."
    )
    with pytest.raises(ValueError, match=err_msg):
        _resolve_n_jobs(n_jobs=n_jobs)


@pytest.mark.parametrize(
    "n_jobs, expected",
    [(1, 1),
     (3, 3),
     (10, 10),
     (True, 1),
     (-1, 4),
     (-2, 3),
     (-4, 1),
     (-10, 1)],
    ids=lambda value: f'{value!r}'
)
def test_resolve_n_jobs_output(n_jobs, expected):
    """
    Test the output of `_resolve_n_jobs` with 4 CPUs (`joblib.cpu_count`
    patched): positive values are returned as they are, and negative values
    follow the rule of joblib, `cpu_count() + 1 + n_jobs` with a minimum of 1
    (`-1` all the CPUs, `-2` all but one). The output is a Python int.
    """
    with patch("joblib.cpu_count", return_value=4):
        n_jobs_resolved = _resolve_n_jobs(n_jobs=n_jobs)

    assert n_jobs_resolved == expected
    assert type(n_jobs_resolved) is int
