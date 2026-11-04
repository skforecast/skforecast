# Unit test _sample_param_grid
# ==============================================================================
import warnings
import pytest
from scipy.stats import uniform
from sklearn.model_selection import ParameterSampler
from skforecast.model_selection._utils import _sample_param_grid


@pytest.mark.parametrize(
    "lags_grid, expected_for_each_lags",
    [(True, " for each value of `lags_grid`"), (False, "")],
    ids=["lags_grid", "no lags_grid"],
)
def test_sample_param_grid_UserWarning_when_space_smaller_than_n_iter(
    lags_grid, expected_for_each_lags
):
    """
    Test that, when the total space of parameters is smaller than `n_iter`,
    the warning refers to the grid search of skforecast and the warning of
    scikit-learn is not shown.
    """
    param_distributions = {'alpha': [0.1, 1.0], 'fit_intercept': [True, False]}

    with pytest.warns(UserWarning) as record:
        param_grid = _sample_param_grid(
            param_distributions = param_distributions,
            n_iter              = 6,
            random_state        = 123,
            grid_search_name    = 'grid_search_forecaster',
            lags_grid           = lags_grid
        )

    assert len(param_grid) == 4
    assert [str(w.message) for w in record] == [
        f"The total space of parameters (4) is smaller than `n_iter` (6), so the "
        f"search evaluates the 4 combinations{expected_for_each_lags}. For an "
        f"exhaustive search, use `grid_search_forecaster`."
    ]


def test_sample_param_grid_same_output_as_ParameterSampler_without_warning():
    """
    Test that the combinations are those of scikit-learn's ParameterSampler,
    and no warning is raised when the space is larger than `n_iter`.
    """
    param_distributions = {'alpha': uniform(0, 1), 'fit_intercept': [True, False]}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        param_grid = _sample_param_grid(
            param_distributions = param_distributions,
            n_iter              = 5,
            random_state        = 123,
            grid_search_name    = 'grid_search_forecaster'
        )

    expected = list(
        ParameterSampler(param_distributions, n_iter=5, random_state=123)
    )
    assert param_grid == expected
