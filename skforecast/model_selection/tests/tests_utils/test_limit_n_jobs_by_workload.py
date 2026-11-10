# Unit test limit_n_jobs_by_workload
# ==============================================================================
import sys
import pytest
from skforecast.model_selection._utils import limit_n_jobs_by_workload


@pytest.mark.parametrize(
    "n_jobs, n_tasks, n_fits, fit_seconds",
    [(1, 10, 9, 100.0),
     (8, 1, 0, 100.0),
     (8, 10, 0, 100.0),
     (8, 10, 0, None)],
    ids=["n_jobs 1", "one task", "no refit", "no refit, fit not timed"]
)
def test_limit_n_jobs_by_workload_output_1_when_nothing_to_share(
    n_jobs, n_tasks, n_fits, fit_seconds
):
    """
    Test that limit_n_jobs_by_workload returns 1 when `n_jobs` is 1, when there is
    at most one task or when no task fits the forecaster (no refit), whatever
    the duration of the initial fit.
    """
    n_jobs = limit_n_jobs_by_workload(
                 n_jobs=n_jobs, n_tasks=n_tasks, n_fits=n_fits, fit_seconds=fit_seconds
             )

    assert n_jobs == 1


@pytest.mark.parametrize(
    "platform, fit_seconds, expected",
    [("win32", 0.4, 1),
     ("win32", 0.5, 8),
     ("linux", 0.2, 1),
     ("linux", 0.3, 8),
     ("darwin", 0.2, 1),
     ("darwin", 0.3, 8)],
    ids=lambda value: f"{value}"
)
def test_limit_n_jobs_by_workload_output_with_the_minimum_work_of_each_platform(
    monkeypatch, platform, fit_seconds, expected
):
    """
    Test that limit_n_jobs_by_workload returns 1 when the estimated work
    (`n_fits * fit_seconds`, here 9 fits) is below the minimum of the platform
    (4 seconds on Windows, 2 seconds on Linux and macOS, 4 seconds on any
    other platform), and `n_jobs` otherwise.
    """
    monkeypatch.setattr(sys, "platform", platform)
    n_jobs = limit_n_jobs_by_workload(
                 n_jobs=8, n_tasks=10, n_fits=9, fit_seconds=fit_seconds
             )

    assert n_jobs == expected


@pytest.mark.parametrize(
    "n_jobs, n_tasks, fit_seconds, expected",
    [(21, 10, 1.0, 10),
     (21, 10, None, 10),
     (4, 10, 1.0, 4),
     (4, 10, None, 4)],
    ids=lambda value: f"{value}"
)
def test_limit_n_jobs_by_workload_output_capped_by_number_of_tasks(
    n_jobs, n_tasks, fit_seconds, expected
):
    """
    Test that limit_n_jobs_by_workload does not return more jobs than tasks when
    the estimated work is above the minimum of every platform (9 fits of 1
    second), or when the initial fit was not timed (`fit_seconds=None`), in
    which case the work is not estimated.
    """
    n_jobs = limit_n_jobs_by_workload(
                 n_jobs=n_jobs, n_tasks=n_tasks, n_fits=9, fit_seconds=fit_seconds
             )

    assert n_jobs == expected
