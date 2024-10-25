# Unit test bayesian_search_forecaster_multiseries
# ==============================================================================
import re
import pytest
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.direct import ForecasterDirectMultiVariate
from skforecast.model_selection import bayesian_search_forecaster_multiseries
from skforecast.model_selection._split import TimeSeriesFold, OneStepAheadFold
from sklearn.metrics import mean_absolute_percentage_error
from skforecast.metrics import mean_absolute_scaled_error

# Fixtures
from ..fixtures_model_selection_multiseries import series
THIS_DIR = Path(__file__).parent.parent
series_item_sales = pd.read_parquet(THIS_DIR/'fixture_multi_series_items_sales.parquet')
series_item_sales = series_item_sales.asfreq('D')
exog_item_sales = pd.DataFrame({'day_of_week': series_item_sales.index.dayofweek}, index = series_item_sales.index)
series_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series.joblib')
exog_dict = joblib.load(THIS_DIR/'fixture_sample_multi_series_exog.joblib')
end_train = "2016-07-31 23:59:00"
series_dict_train = {k: v.loc[:end_train,] for k, v in series_dict.items()}
exog_dict_train = {k: v.loc[:end_train,] for k, v in exog_dict.items()}
series_dict_test = {k: v.loc[end_train:,] for k, v in series_dict.items()}
exog_dict_test = {k: v.loc[end_train:,] for k, v in exog_dict.items()}


def test_ValueError_bayesian_search_forecaster_multiseries_when_return_best_and_len_series_exog_different():
    """
    Test ValueError is raised in bayesian_search_forecaster_multiseries when 
    return_best and length of `series` and `exog` do not match.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot'
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = True,
            fixed_train_size   = True
    )
    exog = series[:30].copy()

    def search_space(trial):  # pragma: no cover
        search_space  = {'alpha': trial.suggest_float('alpha', 1e-2, 1.0)}

        return search_space

    err_msg = re.escape(
        (f"`exog` must have same number of samples as `series`. "
         f"length `exog`: ({len(exog)}), length `series`: ({len(series)})")
    )
    with pytest.raises(ValueError, match = err_msg):
        bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series,
            exog               = exog,
            cv                 = cv,
            search_space       = search_space,
            metric             = 'mean_absolute_error',
            aggregate_metric   = 'weighted_average',
            n_trials           = 10,
            random_state       = 123,
            return_best        = True,
            verbose            = False,
        )


def test_results_output_bayesian_search_forecaster_multiseries_ForecasterAutoregMultiSeries():
    """
    Test output of bayesian_search_forecaster_multiseries in 
    ForecasterRecursiveMultiSeries with mocked (mocked done in Skforecast v0.12.0).
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     regressor = Ridge(random_state=123),
                     lags      = 2,
                     encoding  = 'onehot',
                     transformer_series = StandardScaler()
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series[:-12]),
            steps              = 3,
            refit              = False,
            fixed_train_size   = True
    )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, 4])
        }

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.23598059857016607}, 0.20880273566554797,
            0.23598059857016607],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.398196343012209}, 0.2088108334929226,
            0.398196343012209],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.4441865222328282}, 0.2088131177203144,
            0.4441865222328282],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.53623586010342}, 0.2088176743151537,
            0.53623586010342],
        [list(['l1', 'l2']), np.array([1, 2, 3, 4]),
            {'alpha': 0.7252189487445193}, 0.2088269659234972,
            0.7252189487445193],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.5558016213920624},
            0.20913532870732662, 0.5558016213920624],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.6995044937418831},
            0.20913908163271674, 0.6995044937418831],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.7406154516747153},
            0.20914015254956903, 0.7406154516747153],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.8509374761370117},
            0.20914302038975385, 0.8509374761370117],
        [list(['l1', 'l2']), np.array([1, 2]), {'alpha': 0.9809565564007693},
            0.20914638910039665, 0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error__weighted_average', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error__weighted_average': float, 
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_bayesian_search_forecaster_multiseries_series_and_exog_dict_with_mocked():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries
    when series and exog are dictionaries (mocked done in Skforecast v0.12.0).
    """

    forecaster = ForecasterRecursiveMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding="ordinal",
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_train["id_1000"]),
            steps              = 24,
            refit              = False,
            fixed_train_size   = True
    )
    lags_grid = [[5], [1, 7, 14]]

    def search_space(trial):
        search_space = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 5),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "lags": trial.suggest_categorical("lags", lags_grid),
        }

        return search_space

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

        results_search, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict,
            exog               = exog_dict,
            cv                 = cv,
            search_space       = search_space,
            metric             = "mean_absolute_error",
            aggregate_metric   = "weighted_average",
            n_trials           = 3,
            return_best        = False,
            show_progress      = False,
            verbose            = False,
            suppress_warnings  = True,
        )

    expected = pd.DataFrame(
        np.array(
            [
                [
                    list(["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"]),
                    np.array([1, 7, 14]),
                    {"n_estimators": 4, "max_depth": 3},
                    709.8836514262415,
                    4,
                    3,
                ],
                [
                    list(["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"]),
                    np.array([1, 7, 14]),
                    {"n_estimators": 3, "max_depth": 3},
                    721.1848222120482,
                    3,
                    3,
                ],
                [
                    list(["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"]),
                    np.array([5]),
                    {"n_estimators": 4, "max_depth": 3},
                    754.3537196425694,
                    4,
                    3,
                ],
            ],
            dtype=object,
        ),
        columns=[
            "levels",
            "lags",
            "params",
            "mean_absolute_error__weighted_average",
            "n_estimators",
            "max_depth",
        ],
        index=pd.Index([0, 1, 2], dtype="int64"),
    ).astype(
        {
            "mean_absolute_error__weighted_average": float,
            "n_estimators": int,
            "max_depth": int,
        }
    )

    results_search = results_search.astype(
        {
            "mean_absolute_error__weighted_average": float,
            "n_estimators": int,
            "max_depth": int,
        }
    )

    pd.testing.assert_frame_equal(expected, results_search)


def test_output_bayesian_search_forecaster_multiseries_series_and_exog_dict_multiple_metrics_aggregated_with_mocked():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries
    when series and exog are dictionaries and multiple aggregated metrics
    (mocked done in Skforecast v0.12.0).
    """

    forecaster = ForecasterRecursiveMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding="ordinal",
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_train["id_1000"]),
            steps              = 24,
            refit              = False,
    )
    lags_grid = [[5], [1, 7, 14]]

    def search_space(trial):
        search_space = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 5),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "lags": trial.suggest_categorical("lags", lags_grid),
        }

        return search_space

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="optuna")

        results_search, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict,
            cv                 = cv,
            exog               = exog_dict,
            search_space       = search_space,
            metric             = ['mean_absolute_error', 'mean_absolute_scaled_error'],
            aggregate_metric   = ['weighted_average', 'average', 'pooling'],
            n_trials           = 3,
            return_best        = False,
            show_progress      = False,
            verbose            = False,
            suppress_warnings  = True,
        )

    expected = pd.DataFrame(
        {
            "levels": [
                ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
                ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
                ["id_1000", "id_1001", "id_1002", "id_1003", "id_1004"],
            ],
            "lags": [np.array([1, 7, 14]), np.array([1, 7, 14]), np.array([5])],
            "params": [
                {"n_estimators": 4, "max_depth": 3},
                {"n_estimators": 3, "max_depth": 3},
                {"n_estimators": 4, "max_depth": 3},
            ],
            "mean_absolute_error__weighted_average": [
                749.8761502029433,
                760.659082077477,
                777.6874712018467,
            ],
            "mean_absolute_error__average": [
                709.8836514262415,
                721.1848222120482,
                754.3537196425694,
            ],
            "mean_absolute_error__pooling": [
                709.8836514262414,
                721.1848222120483,
                754.3537196425694,
            ],
            "mean_absolute_scaled_error__weighted_average": [
                1.720281630023928,
                1.7468333254211739,
                1.7217882951816943,
            ],
            "mean_absolute_scaled_error__average": [
                2.0713511786893473,
                2.104173580194738,
                2.06373433155371,
            ],
            "mean_absolute_scaled_error__pooling": [
                1.7240004031507636,
                1.7514460598462835,
                1.7678687830707494,
            ],
            "n_estimators": [4, 3, 4],
            "max_depth": [3, 3, 3],
        }
    )

    pd.testing.assert_frame_equal(expected, results_search)


def test_output_bayesian_search_forecaster_multiseries_series_and_exog_dict_with_mocked_skip_folds():
    """
    Test output of bayesian_search_forecaster_multiseries in ForecasterRecursiveMultiSeries 
    when series and exog are dictionaries (mocked done in Skforecast v0.12.0) and skip_folds.
    """

    forecaster = ForecasterRecursiveMultiSeries(
        regressor=LGBMRegressor(
            n_estimators=2, random_state=123, verbose=-1, max_depth=2
        ),
        lags=14,
        encoding='ordinal',
        dropna_from_series=False,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    cv = TimeSeriesFold(
            initial_train_size = len(series_dict_train["id_1000"]),
            steps              = 24,
            refit              = False,
            skip_folds         = 2,
    )
    lags_grid = [[5], [1, 7, 14]]

    def search_space(trial):
        search_space = {
            "n_estimators": trial.suggest_int("n_estimators", 2, 5),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "lags": trial.suggest_categorical("lags", lags_grid),
        }

        return search_space

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
        
        results_search, _ = bayesian_search_forecaster_multiseries(
            forecaster         = forecaster,
            series             = series_dict,
            exog               = exog_dict,
            cv                 = cv,
            search_space       = search_space,
            metric             = 'mean_absolute_error',          
            n_trials           = 3,
            return_best        = False,
            show_progress      = False,
            verbose            = False,
            suppress_warnings  = True
        )
    
    expected = pd.DataFrame(
        np.array([[list(['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']),
        np.array([1,  7, 14]), {'n_estimators': 4, 'max_depth': 3}, 
            718.447039241195, 694.843611295513, 694.843611295513, 4, 3],
        [list(['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']),
        np.array([1,  7, 14]), {'n_estimators': 3, 'max_depth': 3},
            726.7646039380159, 704.8898538858386, 704.8898538858386, 3, 3],
        [list(['id_1000', 'id_1001', 'id_1002', 'id_1003', 'id_1004']),
         np.array([5]), {'n_estimators': 4, 'max_depth': 3},
            735.3584948212444, 730.1888301097707, 730.1888301097707, 4, 3]], dtype=object),
        columns=['levels', 'lags', 'params', 
                 'mean_absolute_error__weighted_average', 
                 'mean_absolute_error__average', 
                 'mean_absolute_error__pooling',
                 'n_estimators', 'max_depth'],
        index=pd.Index([0, 1, 2], dtype='int64')
    ).astype({
        'mean_absolute_error__weighted_average': float,
        'mean_absolute_error__average': float,
        'mean_absolute_error__pooling': float,
        'n_estimators': int,
        'max_depth': int
    })

    results_search = results_search.astype({
        'mean_absolute_error__weighted_average': float,
        'mean_absolute_error__average': float,
        'mean_absolute_error__pooling': float,
        'n_estimators': int,
        'max_depth': int
    })

    pd.testing.assert_frame_equal(expected, results_search)


def test_results_output_bayesian_search_forecaster_multivariate_ForecasterAutoregMultiVariate():
    """
    Test output of bayesian_search_forecaster_multivariate in 
    ForecasterDirectMultiVariate with mocked (mocked done in Skforecast v0.12.0).
    """
    forecaster = ForecasterDirectMultiVariate(
                     regressor = Ridge(random_state=123),
                     level     = 'l1',
                     lags      = 2,
                     steps     = 3
                 )
    cv = TimeSeriesFold(
            initial_train_size = len(series) - 12,
            steps              = 3,
            refit              = False,
    )

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-2, 1.0),
            'lags': trial.suggest_categorical('lags', [2, {'l1': 4, 'l2': [2, 3]}])
        }

        return search_space

    results = bayesian_search_forecaster_multiseries(
                  forecaster         = forecaster,
                  series             = series,
                  cv                 = cv,
                  search_space       = search_space,
                  metric             = 'mean_absolute_error',
                  aggregate_metric   = 'weighted_average',
                  n_trials           = 10,
                  random_state       = 123,
                  return_best        = False,
                  verbose            = False,
              )[0]
    
    expected_results = pd.DataFrame(
        np.array([[list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.23598059857016607}, 0.19308110319514993,
            0.23598059857016607],
        [list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.398196343012209}, 0.1931744420708601,
            0.398196343012209],
        [list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.4441865222328282}, 0.1932004954044704,
            0.4441865222328282],
        [list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.53623586010342}, 0.19325210858832276,
            0.53623586010342],
        [list(['l1']), {'l1': np.array([1, 2, 3, 4]), 'l2': np.array([2, 3])},
            {'alpha': 0.7252189487445193}, 0.19335589494249983,
            0.7252189487445193],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.5558016213920624},
            0.20131081099888368, 0.5558016213920624],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.6995044937418831},
            0.2013710017368262, 0.6995044937418831],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.7406154516747153},
            0.2013880862681147, 0.7406154516747153],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.8509374761370117},
            0.20143363961627603, 0.8509374761370117],
        [list(['l1']), np.array([1, 2]), {'alpha': 0.9809565564007693},
            0.20148678375852938, 0.9809565564007693]], dtype=object),
        columns=['levels', 'lags', 'params', 'mean_absolute_error', 'alpha'],
        index=pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64')
    ).astype({
        'mean_absolute_error': float,
        'alpha': float
    })

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_bayesian_search_forecaster_multiseries_ForecasterAutoregMultiVariate_one_step_ahead():
    """
    Test output of bayesian_search_forecaster_multiseries when forecaster is ForecasterRecursiveMultiSeries
    and method is one_step_ahead.
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

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
            'lags': trial.suggest_categorical('lags', [3, 5]),
        }

        return search_space

    results, _ = bayesian_search_forecaster_multiseries(
        forecaster         = forecaster,
        series             = series_item_sales,
        exog               = exog_item_sales,
        cv                 = cv,
        search_space       = search_space,
        n_trials           = 5,
        metric             = metrics,
        aggregate_metric   = ["average", "weighted_average", "pooling"],
        return_best        = False,
        n_jobs             = 'auto',
        verbose            = False,
        show_progress      = False
    )

    expected_results = pd.DataFrame({
        "levels": [["item_1"], ["item_1"], ["item_1"], ["item_1"], ["item_1"]],
        "lags": [
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ],
        "params": [
            {"alpha": 0.2252709077935534},
            {"alpha": 0.4279898484823994},
            {"alpha": 15.094374246471325},
            {"alpha": 2.031835829826598},
            {"alpha": 766.6289057556013},
        ],
        "mean_absolute_error": [
            0.8093780037271759,
            0.8094817020309184,
            0.8518429054223443,
            0.8539920097550104,
            1.0811971619067005,
        ],
        "mean_absolute_percentage_error": [
            0.03830937965446373,
            0.03831611554799997,
            0.0404554529541771,
            0.04045967644664634,
            0.05359275342361736,
        ],
        "mean_absolute_scaled_error": [
            0.5296430977017792,
            0.5297109560949745,
            0.5575335769867841,
            0.5589401718158097,
            0.7076465828014749,
        ],
        "alpha": [
            0.2252709077935534,
            0.4279898484823994,
            15.094374246471325,
            2.031835829826598,
            766.6289057556013,
        ],
    },
        index=pd.RangeIndex(start=0, stop=5, step=1)
    )

    pd.testing.assert_frame_equal(results, expected_results)


def test_output_bayesian_search_forecaster_multiseries_ForecasterAutoregMultiSeries_one_step_ahead():
    """
    Test output of bayesian_search_forecaster_multiseries when forecaster is ForecasterRecursiveMultiSeries
    and method is one_step_ahead.
    """
    metrics = ['mean_absolute_error', mean_absolute_percentage_error, mean_absolute_scaled_error]
    forecaster = ForecasterRecursiveMultiSeries(
            regressor          = Ridge(random_state=123),
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
            initial_train_size = 1000,
    )
    levels = ["item_1", "item_2", "item_3"]

    def search_space(trial):
        search_space  = {
            'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
            'lags': trial.suggest_categorical('lags', [3, 5]),
        }

        return search_space

    results, _ = bayesian_search_forecaster_multiseries(
        forecaster         = forecaster,
        series             = series_item_sales,
        exog               = exog_item_sales,
        search_space       = search_space,
        cv                 = cv,
        n_trials           = 5,
        metric             = metrics,
        levels             = levels,
        aggregate_metric   = ["average", "weighted_average", "pooling"],
        return_best        = False,
        n_jobs             = 'auto',
        verbose            = False,
        show_progress      = False
    )

    expected_results = pd.DataFrame(
        {
            "levels": [
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
                ["item_1", "item_2", "item_3"],
            ],
            "lags": [
                np.array([1, 2, 3]),
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
            ],
            "params": [
                {"alpha": 766.6289057556013},
                {"alpha": 0.4279898484823994},
                {"alpha": 0.2252709077935534},
                {"alpha": 15.094374246471325},
                {"alpha": 2.031835829826598},
            ],
            "mean_absolute_error__average": [
                2.487682902584588,
                2.5778104346448623,
                2.5779063732830942,
                2.599406545255343,
                2.605166414875337,
            ],
            "mean_absolute_error__weighted_average": [
                2.487682902584588,
                2.5778104346448623,
                2.5779063732830947,
                2.599406545255343,
                2.6051664148753373,
            ],
            "mean_absolute_error__pooling": [
                2.4876829025845875,
                2.5778104346448623,
                2.5779063732830947,
                2.599406545255343,
                2.6051664148753373,
            ],
            "mean_absolute_percentage_error__average": [
                0.13537962851907534,
                0.14171211559040905,
                0.1417181361315976,
                0.1425099620058197,
                0.14287737387127633,
            ],
            "mean_absolute_percentage_error__weighted_average": [
                0.13537962851907534,
                0.14171211559040905,
                0.14171813613159756,
                0.14250996200581972,
                0.14287737387127633,
            ],
            "mean_absolute_percentage_error__pooling": [
                0.13537962851907534,
                0.14171211559040905,
                0.14171813613159756,
                0.14250996200581975,
                0.14287737387127633,
            ],
            "mean_absolute_scaled_error__average": [
                0.9807885628114613,
                0.9887040054273549,
                0.9887272793513265,
                0.9982897158009892,
                0.999726922014544,
            ],
            "mean_absolute_scaled_error__weighted_average": [
                0.9807885628114613,
                0.9887040054273548,
                0.9887272793513265,
                0.9982897158009892,
                0.9997269220145439,
            ],
            "mean_absolute_scaled_error__pooling": [
                0.994386675351447,
                1.0309527259247835,
                1.0309910949992809,
                1.039045301850066,
                1.0413476602398484,
            ],
            "alpha": [
                766.6289057556013,
                0.4279898484823994,
                0.2252709077935534,
                15.094374246471325,
                2.031835829826598,
            ],
        },
        index=pd.RangeIndex(start=0, stop=5, step=1),
    )

    pd.testing.assert_frame_equal(results, expected_results)
