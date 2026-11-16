# Unit test limit_n_jobs_by_prediction_workload
# ==============================================================================
import sys
import pytest
from skforecast.model_selection._utils import limit_n_jobs_by_prediction_workload


@pytest.mark.parametrize(
    "n_jobs, n_tasks",
    [(1, 99), (8, 1), (8, 0)],
    ids=["n_jobs 1", "one task", "no task"]
)
def test_limit_n_jobs_by_prediction_workload_output_1_when_nothing_to_share(
    n_jobs, n_tasks
):
    """
    Test that limit_n_jobs_by_prediction_workload returns 1 when `n_jobs` is 1
    or there is at most one task, whatever the time of the predictions.
    """
    n_jobs = limit_n_jobs_by_prediction_workload(
                 n_jobs=n_jobs, n_tasks=n_tasks, predict_seconds=100.0
             )

    assert n_jobs == 1


@pytest.mark.parametrize(
    "platform, predict_seconds, expected",
    [("win32", 0.12, 1),
     ("win32", 0.13, 8),
     ("linux", 0.06, 1),
     ("linux", 0.07, 8),
     ("darwin", 0.06, 1),
     ("darwin", 0.07, 8),
     ("other", 0.12, 1),
     ("other", 0.13, 8)],
    ids=lambda value: f"{value}"
)
def test_limit_n_jobs_by_prediction_workload_output_with_the_minimum_of_each_platform(
    monkeypatch, platform, predict_seconds, expected
):
    """
    Test that limit_n_jobs_by_prediction_workload returns 1 when the estimated
    work (`n_tasks * predict_seconds`, here 99 tasks) is below the minimum of
    the platform (12 seconds on Windows, 6 seconds on Linux and macOS, 12
    seconds on any other platform), and `n_jobs` otherwise.
    """
    monkeypatch.setattr(sys, "platform", platform)
    n_jobs = limit_n_jobs_by_prediction_workload(
                 n_jobs=8, n_tasks=99, predict_seconds=predict_seconds
             )

    assert n_jobs == expected


@pytest.mark.parametrize(
    "n_jobs, n_tasks, expected",
    [(21, 10, 10), (4, 10, 4)],
    ids=lambda value: f"{value}"
)
def test_limit_n_jobs_by_prediction_workload_output_capped_by_number_of_tasks(
    n_jobs, n_tasks, expected
):
    """
    Test that limit_n_jobs_by_prediction_workload does not return more jobs
    than tasks when the work is enough to parallelize.
    """
    n_jobs = limit_n_jobs_by_prediction_workload(
                 n_jobs=n_jobs, n_tasks=n_tasks, predict_seconds=100.0
             )

    assert n_jobs == expected
