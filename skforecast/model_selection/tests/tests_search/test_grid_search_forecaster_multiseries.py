# Unit test grid_search_forecaster_multiseries
# ==============================================================================
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import grid_search_forecaster_multiseries
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.metrics import mean_absolute_scaled_error

# Fixtures
from ..fixtures_model_selection_multiseries import series

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # hide progress bar

THIS_DIR = Path(__file__).parent.parent
series_item_sales = pd.read_parquet(THIS_DIR/'fixture_multi_series_items_sales.parquet')
series_item_sales = series_item_sales.asfreq('D')
exog_item_sales = pd.DataFrame({'day_of_week': series_item_sales.index.dayofweek}, index = series_item_sales.index)
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiSeries_with_mocked():
    """
    Test output of grid_search_forecaster_multiseries in ForecasterRecursiveMultiSeries 
    with mocked (mocked done in Skforecast v0.5.0)
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series) - 12,
            steps              = 3,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = {'alpha': [0.01, 0.1, 1]}

    results = grid_search_forecaster_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  cv                  = cv,
                  metric              = 'mean_absolute_error',
                  aggregate_metric    = 'weighted_average',
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  return_best         = False,
                  verbose             = True
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2'], ['l1', 'l2']],
        'lags'  : [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
        'lags_label': [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2], [1, 2], [1, 2]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error__weighted_average':  np.array(
            [0.20968100463227382, 0.20969259779858337, 0.20977945312386406, 
             0.21077344827205086, 0.21078653113227208, 0.21078779824759553]
        ),                                                               
        'alpha' : np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiSeries_multiple_aggregated_metrics_with_mocked():
    """
    Test output of grid_search_forecaster_multiseries in ForecasterRecursiveMultiSeries 
    with mocked (mocked done in Skforecast v0.5.0)
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = Ridge(random_state=123),
                     lags               = 2, 
                     encoding           = 'onehot',
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series) - 12,
            steps              = 3,
            refit              = False,
            fixed_train_size   = False,
         )
    lags_grid = [2, 4]
    param_grid = {'alpha': [0.01, 0.1, 1]}

    results = grid_search_forecaster_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  cv                  = cv,
                  metric              = ['mean_absolute_error', 'mean_absolute_scaled_error'],
                  aggregate_metric    = ['weighted_average', 'average', 'pooling'],
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  return_best         = False,
                  verbose             = False,
                  show_progress       = False,
              )
    
    expected_results = pd.DataFrame({
        'levels': {0: ['l1', 'l2'],
        1: ['l1', 'l2'],
        2: ['l1', 'l2'],
        3: ['l1', 'l2'],
        4: ['l1', 'l2'],
        5: ['l1', 'l2']},
        'lags': {0: np.array([1, 2, 3, 4]),
        1: np.array([1, 2, 3, 4]),
        2: np.array([1, 2, 3, 4]),
        3: np.array([1, 2]),
        4: np.array([1, 2]),
        5: np.array([1, 2])},
        'lags_label': {0: np.array([1, 2, 3, 4]),
        1: np.array([1, 2, 3, 4]),
        2: np.array([1, 2, 3, 4]),
        3: np.array([1, 2]),
        4: np.array([1, 2]),
        5: np.array([1, 2])},
        'params': {0: {'alpha': 0.01},
        1: {'alpha': 0.1},
        2: {'alpha': 1},
        3: {'alpha': 1},
        4: {'alpha': 0.1},
        5: {'alpha': 0.01}},
        'mean_absolute_error__weighted_average': {0: 0.20968100547390048,
        1: 0.20969259864077977,
        2: 0.20977945397058564,
        3: 0.21077344921320568,
        4: 0.21078653208835063,
        5: 0.21078779920557153},
        'mean_absolute_error__average': {0: 0.20968100547390048,
        1: 0.20969259864077974,
        2: 0.20977945397058564,
        3: 0.21077344921320565,
        4: 0.21078653208835063,
        5: 0.21078779920557153},
        'mean_absolute_error__pooling': {0: 0.20968100547390045,
        1: 0.2096925986407798,
        2: 0.20977945397058564,
        3: 0.21077344921320565,
        4: 0.21078653208835063,
        5: 0.21078779920557153},
        'mean_absolute_scaled_error__weighted_average': {0: 0.798335248515498,
        1: 0.7983823516895249,
        2: 0.7987389489370742,
        3: 0.8120054679421529,
        4: 0.8119798820564519,
        5: 0.8119753734760063},
        'mean_absolute_scaled_error__average': {0: 0.7983352485154981,
        1: 0.7983823516895249,
        2: 0.7987389489370743,
        3: 0.812005467942153,
        4: 0.8119798820564519,
        5: 0.8119753734760063},
        'mean_absolute_scaled_error__pooling': {0: 0.7811335117239494,
        1: 0.7811767002385028,
        2: 0.7815002660695128,
        3: 0.7988421945907456,
        4: 0.7988917793592797,
        5: 0.7988965817986066},
        'alpha': {0: 0.01, 1: 0.1, 2: 1.0, 3: 1.0, 4: 0.1, 5: 0.01}
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiVariate_with_mocked():
    """
    Test output of grid_search_forecaster_multiseries in ForecasterDirectMultiVariate 
    with mocked (mocked done in Skforecast v0.6.0)
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor          = Ridge(random_state=123),
                     level              = 'l1',
                     lags               = 2,
                     steps              = 3, 
                     transformer_series = None
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series) - 12,
            steps              = 3,
            refit              = False,
            fixed_train_size   = False
         )
         
    lags_grid = [2, 4]
    param_grid = {'alpha': [0.01, 0.1, 1]}

    results = grid_search_forecaster_multiseries(
                  forecaster          = forecaster,
                  series              = series,
                  param_grid          = param_grid,
                  cv                  = cv,
                  metric              = 'mean_absolute_error',
                  aggregate_metric    = 'weighted_average',
                  levels              = None,
                  exog                = None,
                  lags_grid           = lags_grid,
                  return_best         = False,
                  verbose             = True
              )
    
    expected_results = pd.DataFrame({
        'levels': [['l1'], ['l1'], ['l1'], ['l1'], ['l1'], ['l1']],
        'lags': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'lags_label': [[1, 2], [1, 2], [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        'params': [{'alpha': 0.01}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 1}, {'alpha': 0.1}, {'alpha': 0.01}],
        'mean_absolute_error':  np.array(
            [0.20115194, 0.20183032, 0.20566862, 0.22224269, 0.22625017, 0.22644284]
        ),                                                               
        'alpha': np.array([0.01, 0.1, 1., 1., 0.1, 0.01])
        },
        index = pd.Index([0, 1, 2, 3, 4, 5], dtype='int64')
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiSeries_one_step_ahead():
    """
    Test output of grid_search_forecaster_multiseries when series and exog are DataFrames,
    forecaster is ForecasterRecursiveMultiSeries and method is one_step_ahead.
    """
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    forecaster = ForecasterRecursiveMultiSeries(
            regressor          = Ridge(random_state=123),
            lags               = 10,
            encoding           = 'ordinal',
            transformer_series = StandardScaler(),
            transformer_exog   = StandardScaler(),
            weight_func        = None,
            series_weights     = None,
            differentiation    = None,
            dropna_from_series = False,
        )
    cv = OneStepAheadFold(
            initial_train_size = 1000,
         )
    levels = ["item_1", "item_2", "item_3"]
    param_grid = {
        "alpha": np.logspace(-1, 1, 3),
    }
    lags_grid = [3, 7]

    results = grid_search_forecaster_multiseries(
        forecaster         = forecaster,
        series             = series_item_sales,
        exog               = exog_item_sales,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        cv                 = cv,
        metric             = metrics,
        aggregate_metric   = ["average", "weighted_average", "pooling"],
        levels             = levels,
        return_best        = False,
        n_jobs             = 'auto',
        verbose            = False,
        show_progress      = False
    )
    
    expected_results = pd.DataFrame({
        "levels": [
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_2", "item_3"],
            ["item_1", "item_2", "item_3"],
        ],
        "lags": [
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ],
        "lags_label": [
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ],
        "params": [
            {"alpha": 10.0},
            {"alpha": 1.0},
            {"alpha": 0.1},
            {"alpha": 10.0},
            {"alpha": 1.0},
            {"alpha": 0.1},
        ],
        "mean_absolute_error__average": [
            2.3946101679147835,
            2.396947110515903,
            2.3971835092279474,
            2.601624329202654,
            2.605630817902648,
            2.6060367442128034,
        ],
        "mean_absolute_error__weighted_average": [
            2.3946101679147835,
            2.396947110515903,
            2.3971835092279474,
            2.601624329202654,
            2.605630817902648,
            2.606036744212804,
        ],
        "mean_absolute_error__pooling": [
            2.3946101679147835,
            2.396947110515903,
            2.3971835092279483,
            2.601624329202654,
            2.6056308179026484,
            2.606036744212804,
        ],
        "mean_absolute_percentage_error__average": [
            0.13256162082722348,
            0.13271450150371536,
            0.13272997272758183,
            0.1426513799658221,
            0.14290701426454108,
            0.1429329239957371,
        ],
        "mean_absolute_percentage_error__weighted_average": [
            0.13256162082722348,
            0.13271450150371536,
            0.13272997272758183,
            0.1426513799658221,
            0.14290701426454108,
            0.1429329239957371,
        ],
        "mean_absolute_percentage_error__pooling": [
            0.13256162082722345,
            0.1327145015037154,
            0.13272997272758183,
            0.1426513799658221,
            0.14290701426454108,
            0.14293292399573712,
        ],
        "mean_absolute_scaled_error__average": [
            0.9160114137335943,
            0.9165140769691532,
            0.916564930542703,
            0.9988413247804616,
            0.9998432724761427,
            0.9999449761167062,
        ],
        "mean_absolute_scaled_error__weighted_average": [
            0.9160114137335942,
            0.9165140769691533,
            0.916564930542703,
            0.9988413247804617,
            0.9998432724761427,
            0.9999449761167061,
        ],
        "mean_absolute_scaled_error__pooling": [
            0.9579938854777973,
            0.9589288087285971,
            0.9590233830036539,
            1.0399318034229648,
            1.041533293296968,
            1.041695551804101,
        ],
        "alpha": [10.0, 1.0, 0.1, 10.0, 1.0, 0.1],
    },
    index = pd.RangeIndex(start=0, stop=6, step=1)
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiVariate_one_step_ahead():
    """
    Test output of grid_search_forecaster_multiseries when series and exog are DataFrames,
    forecaster is ForecasterDirectMultiVariate and method is one_step_ahead.
    """
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    forecaster = ForecasterDirectMultiVariate(
                    regressor          = Ridge(random_state=123),
                    lags               = 10,
                    steps              = 10,
                    level              = 'item_1',
                    transformer_series = StandardScaler(),
                    transformer_exog   = StandardScaler(),
                )

    cv = OneStepAheadFold(
            initial_train_size = 1000,
        )
    levels = ["item_1", "item_2", "item_3"]
    param_grid = {
        "alpha": np.logspace(-1, 1, 3),
    }
    lags_grid = [3, 7]

    results = grid_search_forecaster_multiseries(
        forecaster         = forecaster,
        series             = series_item_sales,
        exog               = exog_item_sales,
        cv                 = cv,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        metric             = metrics,
        aggregate_metric   = ["average", "weighted_average", "pooling"],
        levels             = levels,
        return_best        = False,
        verbose            = False,
        show_progress      = False
    )
    
    expected_results = pd.DataFrame(
        {
            "levels": [
                ["item_1"],
                ["item_1"],
                ["item_1"],
                ["item_1"],
                ["item_1"],
                ["item_1"],
            ],
            "lags": [
                np.array([1, 2, 3, 4, 5, 6, 7]),
                np.array([1, 2, 3, 4, 5, 6, 7]),
                np.array([1, 2, 3, 4, 5, 6, 7]),
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
            ],
            "lags_label": [
                np.array([1, 2, 3, 4, 5, 6, 7]),
                np.array([1, 2, 3, 4, 5, 6, 7]),
                np.array([1, 2, 3, 4, 5, 6, 7]),
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
            ],
            "params": [
                {"alpha": 10.0},
                {"alpha": 1.0},
                {"alpha": 0.1},
                {"alpha": 10.0},
                {"alpha": 1.0},
                {"alpha": 0.1},
            ],
            "mean_absolute_error": [
                0.7629704997414917,
                0.7667482865205157,
                0.7671081143069952,
                0.8525261078812101,
                0.8541841441709307,
                0.8543521794925618,
            ],
            "mean_absolute_percentage_error": [
                0.03604811720204804,
                0.036197365349618,
                0.03621109426633075,
                0.04045012068206693,
                0.040460698841325624,
                0.04046154712278539,
            ],
            "mean_absolute_scaled_error": [
                0.5008239495431114,
                0.5033037388611042,
                0.5035399345898264,
                0.5579807349172816,
                0.5590659243313126,
                0.5591759039218367,
            ],
            "alpha": [10.0, 1.0, 0.1, 10.0, 1.0, 0.1],
        },
        index=pd.RangeIndex(start=0, stop=6, step=1),
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_grid_search_forecaster_multiseries_ForecasterAutoregMultiSeries_one_step_ahead_input_is_dict():
    """
    Test output of grid_search_forecaster_multiseries when series and exog are dictionaries,
    forecaster is ForecasterRecursiveMultiSeries and method is one_step_ahead.
    """
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    forecaster = ForecasterRecursiveMultiSeries(
            regressor          = LGBMRegressor(random_state=678, verbose=-1),
            lags               = 3,
            encoding           = 'ordinal',
            transformer_series = StandardScaler(),
            transformer_exog   = StandardScaler(),
            weight_func        = None,
            series_weights     = None,
            differentiation    = None,
            dropna_from_series = False,
        )
    cv = OneStepAheadFold(
            initial_train_size = 213
         )
    levels = ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"]
    param_grid = {
            "n_estimators": [5, 10],
            "max_depth": [2, 3]
        }
    lags_grid = [3, 5]

    results = grid_search_forecaster_multiseries(
        forecaster         = forecaster,
        series             = series_dict,
        exog               = exog_dict,
        cv                 = cv,
        param_grid         = param_grid,
        lags_grid          = lags_grid,
        metric             = metrics,
        aggregate_metric   = ["average", "weighted_average", "pooling"],
        levels             = levels,
        return_best        = False,
        verbose            = False,
        show_progress      = False
    )
    
    expected_results = pd.DataFrame({
        "levels": [
            ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
            ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
            ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
            ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
            ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
            ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
            ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
            ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
        ],
        "lags": [
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4, 5]),
        ],
        "lags_label": [
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4, 5]),
        ],
        "params": [
            {"max_depth": 3, "n_estimators": 10},
            {"max_depth": 3, "n_estimators": 10},
            {"max_depth": 2, "n_estimators": 10},
            {"max_depth": 2, "n_estimators": 10},
            {"max_depth": 3, "n_estimators": 5},
            {"max_depth": 3, "n_estimators": 5},
            {"max_depth": 2, "n_estimators": 5},
            {"max_depth": 2, "n_estimators": 5},
        ],
        "mean_absolute_error__average": [
            595.4248731239852,
            601.61658409234,
            641.5971327961081,
            643.5830924971917,
            677.9145593017354,
            687.1976066932838,
            709.8383108516048,
            712.5538939535741,
        ],
        "mean_absolute_error__weighted_average": [
            601.0754794775827,
            602.8512611773792,
            637.9906397322952,
            635.3319731010223,
            669.4535161312964,
            674.4638410667131,
            694.4718207943295,
            695.1809840497659,
        ],
        "mean_absolute_error__pooling": [
            601.0754794775827,
            602.8512611773791,
            637.9906397322952,
            635.3319731010222,
            669.4535161312965,
            674.4638410667131,
            694.4718207943295,
            695.1809840497659,
        ],
        "mean_absolute_percentage_error__average": [
            1.13122611897686e16,
            1.1653837498770236e16,
            1.1977069509022024e16,
            1.2092182501085004e16,
            1.2255524771350234e16,
            1.2487020718197592e16,
            1.3255138678699688e16,
            1.320416830274058e16,
        ],
        "mean_absolute_percentage_error__weighted_average": [
            1.6602167501530896e16,
            1.7103473739202362e16,
            1.757785740892441e16,
            1.7746800217419718e16,
            1.7986525563708256e16,
            1.832627501087992e16,
            1.945358482341537e16,
            1.9378779379561716e16,
        ],
        "mean_absolute_percentage_error__pooling": [
            1.6602167501530896e16,
            1.7103473739202362e16,
            1.7577857408924412e16,
            1.7746800217419718e16,
            1.7986525563708258e16,
            1.832627501087992e16,
            1.9453584823415372e16,
            1.9378779379561716e16,
        ],
        "mean_absolute_scaled_error__average": [
            1.338257547877582,
            1.3458574988611296,
            1.418344242255843,
            1.4196630001824362,
            1.502472367026163,
            1.516064192858667,
            1.5547477191379595,
            1.5594122630466858,
        ],
        "mean_absolute_scaled_error__weighted_average": [
            1.6314647598931409,
            1.627988835703755,
            1.720233806106665,
            1.7212080197022548,
            1.8204941416870104,
            1.8231712390456873,
            1.8766737381944514,
            1.8860125959513818,
        ],
        "mean_absolute_scaled_error__pooling": [
            1.4086529286833727,
            1.4224992984312164,
            1.505414844206229,
            1.4889348761536338,
            1.5689005596025196,
            1.5914777035729348,
            1.6386889129680593,
            1.6291942735674578,
        ],
        "max_depth": [3, 3, 2, 2, 3, 3, 2, 2],
        "n_estimators": [10, 10, 10, 10, 5, 5, 5, 5],
        },
        index = pd.RangeIndex(start=0, stop=8, step=1)
    )

    pd.testing.assert_frame_equal(results, expected_results)
