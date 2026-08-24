# Unit test grid_search_equivalent_date
# ==============================================================================
import re
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from skforecast.recursive import ForecasterEquivalentDate, ForecasterRecursive
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold
from skforecast.model_selection._search import grid_search_equivalent_date

# Fixtures
from ....recursive.tests.tests_forecaster_equivalent_date.fixtures_forecaster_equivalent_date import y

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar


def test_grid_search_equivalent_date_TypeError_when_forecaster_not_equivalent_date():
    """
    Test TypeError is raised when forecaster is not a ForecasterEquivalentDate.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    err_msg = re.escape(
        "`forecaster` must be of type `ForecasterEquivalentDate`, for all "
        "other types of forecasters use the functions available in the "
        "`model_selection` module."
    )
    with pytest.raises(TypeError, match=err_msg):
        grid_search_equivalent_date(
            forecaster = forecaster,
            y          = y,
            cv         = cv,
            param_grid = {'offset': [1, 7]},
            metric     = 'mean_absolute_error'
        )


def test_grid_search_equivalent_date_TypeError_when_cv_not_time_series_fold():
    """
    Test TypeError is raised when cv is not a TimeSeriesFold object.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = OneStepAheadFold(initial_train_size=len(y) - 12)

    err_msg = re.escape(
        "`cv` must be an instance of `TimeSeriesFold`. Got OneStepAheadFold."
    )
    with pytest.raises(TypeError, match=err_msg):
        grid_search_equivalent_date(
            forecaster = forecaster,
            y          = y,
            cv         = cv,
            param_grid = {'offset': [1, 7]},
            metric     = 'mean_absolute_error'
        )


@pytest.mark.parametrize(
    'param_grid',
    [1, 'not_valid', ('offset', 1)],
    ids=lambda pg: f'param_grid: {pg}'
)
def test_grid_search_equivalent_date_TypeError_when_param_grid_invalid_type(param_grid):
    """
    Test TypeError is raised when param_grid is neither a dict nor a list.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    err_msg = re.escape(
        f"`param_grid` must be a dict or a list of dicts. "
        f"Got {type(param_grid).__name__}."
    )
    with pytest.raises(TypeError, match=err_msg):
        grid_search_equivalent_date(
            forecaster = forecaster,
            y          = y,
            cv         = cv,
            param_grid = param_grid,
            metric     = 'mean_absolute_error'
        )


def test_grid_search_equivalent_date_TypeError_when_param_grid_list_element_not_dict():
    """
    Test TypeError is raised when an element of the param_grid list is not a dict.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    err_msg = re.escape(
        "When `param_grid` is a list, each element must be a dict "
        "of parameters. Got list."
    )
    with pytest.raises(TypeError, match=err_msg):
        grid_search_equivalent_date(
            forecaster = forecaster,
            y          = y,
            cv         = cv,
            param_grid = [{'offset': 1, 'n_offsets': 7}, [1, 2]],
            metric     = 'mean_absolute_error'
        )


def test_grid_search_equivalent_date_ValueError_when_duplicated_metric():
    """
    Test ValueError is raised when metric list contains duplicated names.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    err_msg = re.escape("When `metric` is a `list`, each metric name must be unique.")
    with pytest.raises(ValueError, match=err_msg):
        grid_search_equivalent_date(
            forecaster = forecaster,
            y          = y,
            cv         = cv,
            param_grid = [{'offset': 1, 'n_offsets': 7}],
            metric     = ['mean_absolute_error', 'mean_absolute_error']
        )


def test_output_grid_search_equivalent_date_dict_param_grid():
    """
    Test output when param_grid is a dict (Cartesian product).
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    results = grid_search_equivalent_date(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  param_grid  = {'offset': [1, 7], 'n_offsets': [1, 2]},
                  metric      = 'mean_absolute_error',
                  return_best = False,
                  verbose     = False
              )

    expected_results = pd.DataFrame(
        data  = {
            'params'             : [{'n_offsets': 2, 'offset': 7},
                                    {'n_offsets': 1, 'offset': 7},
                                    {'n_offsets': 2, 'offset': 1},
                                    {'n_offsets': 1, 'offset': 1}],
            'mean_absolute_error': np.array([0.2357615, 0.2365943, 0.25581102, 0.2573989]),
            'n_offsets'          : [2, 1, 2, 1],
            'offset'             : [7, 7, 1, 1]
        },
        index = pd.Index(np.array([0, 1, 2, 3]), dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_output_grid_search_equivalent_date_list_param_grid():
    """
    Test output when param_grid is a list of explicit configurations. Only the
    requested (offset, n_offsets) pairs must be evaluated, no Cartesian
    product across dictionaries. A pandas DateOffset configuration placed after
    an integer offset configuration must not inherit its window size.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    results = grid_search_equivalent_date(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  param_grid  = [{'offset': 1, 'n_offsets': 7},
                                 {'offset': 7, 'n_offsets': 1},
                                 {'offset': pd.DateOffset(days=7), 'n_offsets': 2}],
                  metric      = 'mean_absolute_error',
                  return_best = False,
                  verbose     = False
              )

    expected_results = pd.DataFrame(
        data  = {
            'params'             : [{'n_offsets': 2, 'offset': pd.DateOffset(days=7)},
                                    {'n_offsets': 1, 'offset': 7},
                                    {'n_offsets': 7, 'offset': 1}],
            'mean_absolute_error': np.array([0.2357615, 0.2365943, 0.25050685]),
            'offset'             : [pd.DateOffset(days=7), 7, 1],
            'n_offsets'          : [2, 1, 7]
        },
        index = pd.Index(np.array([0, 1, 2]), dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_grid_search_equivalent_date_ValueError_when_list_config_has_list_value():
    """
    Test ValueError is raised when a list configuration contains a list-valued
    parameter (each list element must define a single scalar configuration).
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    err_msg = re.escape(
        "When `param_grid` is a list, each dictionary must define a "
        "single configuration with scalar values. Parameters "
        "['agg_func'] have list values. To evaluate multiple "
        "values, either use the dict form of `param_grid` (Cartesian "
        "product) or add one dictionary per configuration."
    )
    with pytest.raises(ValueError, match=err_msg):
        grid_search_equivalent_date(
            forecaster = forecaster,
            y          = y,
            cv         = cv,
            param_grid = [{'offset': 7, 'n_offsets': 2,
                           'agg_func': [np.mean, np.median]}],
            metric     = 'mean_absolute_error'
        )


def test_grid_search_equivalent_date_ValueError_when_alias_in_dict_param_grid():
    """
    Test ValueError is raised when `alias` is used with the dict form of
    param_grid.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    err_msg = re.escape(
        "`alias` is only supported when `param_grid` is a list of "
        "configurations, not when `param_grid` is a dict."
    )
    with pytest.raises(ValueError, match=err_msg):
        grid_search_equivalent_date(
            forecaster = forecaster,
            y          = y,
            cv         = cv,
            param_grid = {'offset': [1, 7], 'alias': ['a', 'b']},
            metric     = 'mean_absolute_error'
        )


def test_output_grid_search_equivalent_date_agg_func_multiple_configs():
    """
    Test output when the same offset and n_offsets are evaluated with different
    aggregation functions using one dictionary per configuration.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    results = grid_search_equivalent_date(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  param_grid  = [{'offset': 7, 'n_offsets': 2, 'agg_func': np.mean},
                                 {'offset': 7, 'n_offsets': 2, 'agg_func': np.median}],
                  metric      = 'mean_absolute_error',
                  return_best = False,
                  verbose     = False
              )

    expected_results = pd.DataFrame(
        data  = {
            'params'             : [{'agg_func': np.mean, 'n_offsets': 2, 'offset': 7},
                                    {'agg_func': np.median, 'n_offsets': 2, 'offset': 7}],
            'mean_absolute_error': np.array([0.2357615, 0.2357615]),
            'offset'             : [7, 7],
            'n_offsets'          : [2, 2],
            'agg_func'           : ['mean', 'median']
        },
        index = pd.Index(np.array([0, 1]), dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_output_grid_search_equivalent_date_with_alias():
    """
    Test that an optional `alias` key labels each configuration in the results
    and is not passed to the forecaster.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    results = grid_search_equivalent_date(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  param_grid  = [{'alias': '7-day moving average',
                                  'offset': 1, 'n_offsets': 7},
                                 {'alias': 'mean of lag-7 and lag-14',
                                  'offset': 7, 'n_offsets': 2}],
                  metric      = 'mean_absolute_error',
                  return_best = False,
                  verbose     = False
              )

    expected_results = pd.DataFrame(
        data  = {
            'alias'              : ['mean of lag-7 and lag-14', '7-day moving average'],
            'params'             : [{'n_offsets': 2, 'offset': 7},
                                    {'n_offsets': 7, 'offset': 1}],
            'mean_absolute_error': np.array([0.2357615, 0.25050685]),
            'offset'             : [7, 1],
            'n_offsets'          : [2, 7]
        },
        index = pd.Index(np.array([0, 1]), dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_output_grid_search_equivalent_date_config_exceeding_training_size_skipped():
    """
    Test that a configuration whose `offset * n_offsets` exceeds the training
    size is skipped with a RuntimeWarning and excluded from the results.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    warn_msg = re.escape("Parameters skipped: {'offset': 20, 'n_offsets': 2}.")
    with pytest.warns(RuntimeWarning, match=warn_msg):
        results = grid_search_equivalent_date(
                      forecaster  = forecaster,
                      y           = y,
                      cv          = cv,
                      param_grid  = [{'offset': 7, 'n_offsets': 2},
                                     {'offset': 20, 'n_offsets': 2}],
                      metric      = 'mean_absolute_error',
                      return_best = False,
                      verbose     = False
                  )

    expected_results = pd.DataFrame(
        data  = {
            'params'             : [{'n_offsets': 2, 'offset': 7}],
            'mean_absolute_error': np.array([0.2357615]),
            'offset'             : [7],
            'n_offsets'          : [2]
        },
        index = pd.Index(np.array([0]), dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_grid_search_equivalent_date_RuntimeWarning_when_all_configs_skipped():
    """
    Test that a RuntimeWarning is raised and an empty results DataFrame is
    returned when all configurations are skipped (all raise exceptions).
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    warn_msg = re.escape(
        "No valid parameter combinations found. All combinations raised exceptions."
    )
    with pytest.warns(RuntimeWarning, match=warn_msg):
        results = grid_search_equivalent_date(
                      forecaster  = forecaster,
                      y           = y,
                      cv          = cv,
                      param_grid  = [{'offset': 20, 'n_offsets': 2}],
                      metric      = 'mean_absolute_error',
                      return_best = False,
                      verbose     = False
                  )

    assert results.empty


def test_grid_search_equivalent_date_multiple_metrics():
    """
    Test output when metric is a list of metrics.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    results = grid_search_equivalent_date(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  param_grid  = [{'offset': 1, 'n_offsets': 7},
                                 {'offset': 7, 'n_offsets': 2}],
                  metric      = ['mean_absolute_error', 'mean_squared_error'],
                  return_best = False,
                  verbose     = False
              )

    expected_results = pd.DataFrame(
        data  = {
            'params'             : [{'n_offsets': 2, 'offset': 7},
                                    {'n_offsets': 7, 'offset': 1}],
            'mean_absolute_error': np.array([0.2357615, 0.25050685]),
            'mean_squared_error' : np.array([0.07591218, 0.09048708]),
            'offset'             : [7, 1],
            'n_offsets'          : [2, 7]
        },
        index = pd.Index(np.array([0, 1]), dtype="int64")
    )

    pd.testing.assert_frame_equal(results, expected_results, atol=0.0001)


def test_grid_search_equivalent_date_return_best():
    """
    Test that return_best refits the forecaster with the best configuration and
    leaves it fitted.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)

    grid_search_equivalent_date(
        forecaster  = forecaster,
        y           = y,
        cv          = cv,
        param_grid  = [{'offset': 1, 'n_offsets': 7},
                       {'offset': 7, 'n_offsets': 2}],
        metric      = 'mean_absolute_error',
        return_best = True,
        verbose     = False
    )

    assert forecaster.is_fitted
    assert forecaster.offset == 7
    assert forecaster.n_offsets == 2


def test_grid_search_equivalent_date_output_file():
    """
    Test that the results are saved to a tab-separated file when output_file is
    provided.
    """
    forecaster = ForecasterEquivalentDate(offset=1, n_offsets=1)
    cv = TimeSeriesFold(steps=3, initial_train_size=len(y) - 12, refit=False)
    output_file = 'test_grid_search_equivalent_date_output_file.txt'

    results = grid_search_equivalent_date(
                  forecaster  = forecaster,
                  y           = y,
                  cv          = cv,
                  param_grid  = [{'offset': 1, 'n_offsets': 7},
                                 {'offset': 7, 'n_offsets': 2}],
                  metric      = 'mean_absolute_error',
                  return_best = False,
                  verbose     = False,
                  output_file = output_file
              )

    assert os.path.isfile(output_file)
    output_file_content = pd.read_csv(output_file, sep='\t', low_memory=False)
    assert len(output_file_content) == len(results)
    os.remove(output_file)
