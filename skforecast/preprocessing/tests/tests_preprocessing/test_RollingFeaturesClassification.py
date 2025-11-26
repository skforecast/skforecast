# Unit test RollingFeaturesClassification
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.preprocessing import RollingFeaturesClassification

# Fixtures
from .fixtures_preprocessing import X_classification


def test_RollingFeaturesClassification_validate_params():
    """
    Test RollingFeaturesClassification _validate_params method.
    """

    params = {
        0: {'stats': 5, 'window_sizes': 5},
        1: {'stats': 'not_valid_stat', 'window_sizes': 5},
        2: {'stats': 'proportion', 'window_sizes': 'not_int_list'},
        3: {'stats': 'proportion', 'window_sizes': [5, 6]},
        4: {'stats': ['proportion', 'mode'], 'window_sizes': [6]},
        5: {'stats': ['proportion', 'mode', 'proportion'], 'window_sizes': [6, 5, 6]},
        6: {'stats': ['proportion'], 'window_sizes': [5], 'min_periods': 'not_int_list'},
        7: {'stats': ['proportion'], 'window_sizes': 5, 'min_periods': [5, 4]},
        8: {'stats': ['proportion', 'mode'], 'window_sizes': 6, 'min_periods': [5]},
        9: {'stats': ['proportion', 'mode'], 'window_sizes': [5, 3], 'min_periods': [5, 4]},
        10: {'stats': ['proportion', 'mode'], 'window_sizes': [5, 6], 
             'min_periods': None, 'features_names': 'not_list'},
        11: {'stats': ['proportion'], 'window_sizes': 5, 
             'min_periods': 5, 'features_names': ['proportion_5', 'mode_6']},
        12: {'stats': ['proportion', 'mode'], 'window_sizes': [5, 6], 
             'min_periods': 4, 'features_names': ['proportion_5']},
        13: {'stats': ['proportion', 'mode'], 'window_sizes': [5, 6], 
             'min_periods': 5, 'features_names': ['proportion_5', 'mode_6'], 'fillna': {}},
        14: {'stats': ['proportion', 'mode'], 'window_sizes': [5, 6], 
             'min_periods': [5, 5], 'features_names': None, 'fillna': 'not_valid_fillna'}
    }
    
    # stats
    err_msg = re.escape(
        f"`stats` must be a string or a list of strings. Got {type(params[0]['stats'])}."
    ) 
    with pytest.raises(TypeError, match = err_msg):
        RollingFeaturesClassification(**params[0])
    err_msg = re.escape(
        "Statistic 'not_valid_stat' is not allowed. Allowed stats are: ['proportion', "
        "'mode', 'entropy', 'n_changes', 'n_unique']."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[1])

    # window_sizes
    err_msg = re.escape(
        f"`window_sizes` must be an int or a list of ints. Got {type(params[2]['window_sizes'])}."
    ) 
    with pytest.raises(TypeError, match = err_msg):
        RollingFeaturesClassification(**params[2])
    err_msg = re.escape(
        "Length of `window_sizes` list (2) must match length of `stats` list (1)."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[3])
    err_msg = re.escape(
        "Length of `window_sizes` list (1) must match length of `stats` list (2)."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[4])
    
    # Check duplicates (stats, window_sizes)
    err_msg = re.escape(
        "Duplicate (stat, window_size) pairs are not allowed.\n"
        "    `stats`        : ['proportion', 'mode', 'proportion']\n"
        "    `window_sizes` : [6, 5, 6]"
    )
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[5])

    # min_periods
    err_msg = re.escape(
        f"`min_periods` must be an int, list of ints, or None. Got {type(params[6]['min_periods'])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        RollingFeaturesClassification(**params[6])
    err_msg = re.escape(
        "Length of `min_periods` list (2) must match length of `stats` list (1)."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[7])
    err_msg = re.escape(
        "Length of `min_periods` list (1) must match length of `stats` list (2)."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[8])
    err_msg = re.escape(
        "Each `min_period` must be less than or equal to its corresponding `window_size`."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[9])

    # features_names
    err_msg = re.escape(
        f"`features_names` must be a list of strings or None. Got {type(params[10]['features_names'])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        RollingFeaturesClassification(**params[10])
    err_msg = re.escape(
        "Length of `features_names` list (2) must match length of `stats` list (1)."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[11])
    err_msg = re.escape(
        "Length of `features_names` list (1) must match length of `stats` list (2)."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[12])

    # fillna
    err_msg = re.escape(
        f"`fillna` must be a float, string, or None. Got {type(params[13]['fillna'])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        RollingFeaturesClassification(**params[13])
    err_msg = re.escape(
        "'not_valid_fillna' is not allowed. Allowed `fillna` "
        "values are: ['mean', 'median', 'ffill', 'bfill'] or a float value."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeaturesClassification(**params[14])


@pytest.mark.parametrize(
    "params", 
    [{'stats': ['proportion', 'entropy'], 'window_sizes': 5, 'min_periods': None, 
      'features_names': None, 'fillna': 'ffill'},
     {'stats': ['proportion', 'entropy'], 'window_sizes': [5, 5], 'min_periods': 5, 
      'features_names': ['roll_proportion_5', 'roll_entropy_5'], 
      'fillna': 'ffill'}], 
    ids = lambda params: f'params: {params}')
def test_RollingFeaturesClassification_init_store_parameters(params):
    """
    Test RollingFeaturesClassification initialization and stored parameters.
    """

    rolling = RollingFeaturesClassification(**params)

    assert rolling.stats == ['proportion', 'entropy']
    assert rolling.n_stats == 2
    assert rolling.window_sizes == [5, 5]
    assert rolling.max_window_size == 5
    assert rolling.min_periods == [5, 5]
    assert rolling.features_names == ['roll_proportion_5', 'roll_entropy_5']
    assert rolling.fillna == 'ffill'

    unique_rolling_windows = {
        '5_5': {
            'params': {'window': 5, 'min_periods': 5, 'center': False, 'closed': 'left'}, 
            'stats_idx': [0, 1],
            'stats_names': ['roll_proportion_5', 'roll_entropy_5'],
            'rolling_obj': None
        }
    }

    assert rolling.unique_rolling_windows == unique_rolling_windows


def test_RollingFeaturesClassification_init_store_parameters_multiple_stats():
    """
    Test RollingFeaturesClassification initialization and stored parameters 
    when multiple stats are passed.
    """

    rolling = RollingFeaturesClassification(stats=['proportion', 'mode', 'entropy'], window_sizes=[5, 5, 6])

    assert rolling.stats == ['proportion', 'mode', 'entropy']
    assert rolling.n_stats == 3
    assert rolling.window_sizes == [5, 5, 6]
    assert rolling.max_window_size == 6
    assert rolling.min_periods == [5, 5, 6]
    assert rolling.features_names == ['roll_proportion_5', 'roll_mode_5', 'roll_entropy_6']
    assert rolling.fillna is None

    unique_rolling_windows = {
        '5_5': {
            'params': {'window': 5, 'min_periods': 5, 'center': False, 'closed': 'left'}, 
            'stats_idx': [0, 1],
            'stats_names': ['roll_proportion_5', 'roll_mode_5'],
            'rolling_obj': None
        },
        '6_6': {
            'params': {'window': 6, 'min_periods': 6, 'center': False, 'closed': 'left'}, 
            'stats_idx': [2],
            'stats_names': ['roll_entropy_6'],
            'rolling_obj': None
        }
    }

    assert rolling.unique_rolling_windows == unique_rolling_windows


def test_RollingFeaturesClassification_ValueError_apply_stat_when_stat_not_implemented():
    """
    Test RollingFeaturesClassification ValueError _apply_stat_pandas and _apply_stat_numpy 
    when applying a statistic not implemented.
    """
    
    rolling = RollingFeaturesClassification(stats='proportion', window_sizes=10)
    X_window = X_classification.iloc[-10:]
    rolling_obj = X_window.rolling(**rolling.unique_rolling_windows['10_10']['params'])

    err_msg = re.escape("Statistic 'not_valid' is not implemented.") 
    with pytest.raises(ValueError, match = err_msg):
        rolling._apply_stat_pandas(X=X_window, rolling_obj=rolling_obj, stat='not_valid')
    with pytest.raises(ValueError, match = err_msg):
        rolling._apply_stat_numpy_jit(X_window.to_numpy(), 'not_valid')


def test_RollingFeaturesClassification_apply_stat_pandas_numpy():
    """
    Test RollingFeaturesClassification _apply_stat_pandas and _apply_stat_numpy methods.
    """

    stats = [
        'proportion', 'mode', 'entropy', 'n_changes', 'n_unique'
    ]
    
    rolling = RollingFeaturesClassification(stats=stats, window_sizes=10)
    X_window_pandas = X_classification.iloc[-11:]
    X_window_numpy = X_classification.to_numpy()[-11:-1]
    rolling.classes = list(np.sort(X_window_pandas.unique()))

    for stat in stats:

        rolling_obj = X_window_pandas.rolling(**rolling.unique_rolling_windows['10_10']['params'])
        stat_pandas = rolling._apply_stat_pandas(
            X=X_window_pandas, rolling_obj=rolling_obj, stat=stat
        )
        if isinstance(stat_pandas, pd.DataFrame):
            stat_pandas = stat_pandas.iloc[-1].to_numpy()
        else:
            stat_pandas = stat_pandas.iat[-1]
        stat_numpy = rolling._apply_stat_numpy_jit(X_window_numpy, stat)

        np.testing.assert_almost_equal(stat_pandas, stat_numpy, decimal=7)


def test_RollingFeaturesClassification_transform_batch():
    """
    Test RollingFeaturesClassification transform_batch method.
    """
    X_datetime = X_classification.copy()
    X_datetime.index = pd.date_range(
        start='1990-01-01', periods=len(X_classification), freq='D'
    )

    stats = [
        'proportion', 'mode', 'entropy', 'n_changes', 'n_unique'
    ]
    rolling = RollingFeaturesClassification(stats=stats, window_sizes=4)
    rolling_features = rolling.transform_batch(X_datetime).head(10)

    expected = pd.DataFrame(
        data={
            "roll_proportion_4_class_1": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                0.5,
                0.25,
            ],
            "roll_proportion_4_class_2": [
                0.5,
                0.5,
                0.5,
                0.75,
                0.75,
                0.5,
                0.5,
                0.5,
                0.25,
                0.5,
            ],
            "roll_proportion_4_class_3": [
                0.25,
                0.25,
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.25,
                0.25,
            ],
            "roll_mode_4": [2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            "roll_entropy_4": [
                1.5,
                1.5,
                1.5,
                0.8112781244591328,
                0.8112781244591328,
                1.0,
                1.0,
                1.0,
                1.5,
                1.5,
            ],
            "roll_n_changes_4": [3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0],
            "roll_n_unique_4": [3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0],
        },
        index=pd.date_range(start="1990-01-05", periods=10, freq="D"),
    )

    pd.testing.assert_frame_equal(rolling_features, expected)


def test_RollingFeaturesClassification_transform_batch_different_rolling():
    """
    Test RollingFeaturesClassification transform_batch method with different rolling windows.
    """
    X_datetime = X_classification.copy()
    X_datetime.index = pd.date_range(
        start='1990-01-01', periods=len(X_classification), freq='D'
    )

    stats = ['proportion', 'mode', 'entropy', 'entropy']
    window_sizes = [4, 5, 6, 4]
    min_periods = [3, 5, 6, 3]
    features_names = ['my_proportion', 'my_mode', 'my_entropy', 'my_entropy_2']

    rolling = RollingFeaturesClassification(
        stats=stats, window_sizes=window_sizes, min_periods=min_periods,
        features_names=features_names, fillna='bfill'
    )
    rolling_features = rolling.transform_batch(X_datetime).head(15)

    expected = pd.DataFrame(
        data=np.array(
            [
                [0.25, 0.5, 0.25, 1.0, 1.45914792, 1.5],
                [0.25, 0.75, 0.0, 2.0, 1.45914792, 0.81127812],
                [0.25, 0.75, 0.0, 2.0, 1.25162917, 0.81127812],
                [0.5, 0.5, 0.0, 2.0, 0.91829583, 1.0],
                [0.5, 0.5, 0.0, 1.0, 1.0, 1.0],
                [0.5, 0.5, 0.0, 2.0, 1.0, 1.0],
                [0.5, 0.25, 0.25, 1.0, 1.45914792, 1.5],
                [0.25, 0.5, 0.25, 1.0, 1.45914792, 1.5],
                [0.0, 0.75, 0.25, 2.0, 1.45914792, 0.81127812],
                [0.0, 0.75, 0.25, 2.0, 1.25162917, 0.81127812],
                [0.0, 0.75, 0.25, 2.0, 0.91829583, 0.81127812],
                [0.25, 0.5, 0.25, 2.0, 1.45914792, 1.5],
                [0.25, 0.25, 0.5, 2.0, 1.45914792, 1.5],
                [0.25, 0.0, 0.75, 3.0, 1.45914792, 0.81127812],
                [0.25, 0.25, 0.5, 3.0, 1.45914792, 1.5],
            ]
        ),
        columns=[
            "my_proportion_class_1",
            "my_proportion_class_2",
            "my_proportion_class_3",
            "my_mode",
            "my_entropy",
            "my_entropy_2",
        ],
        index=pd.date_range(start="1990-01-07", periods=15, freq="D"),
    )

    pd.testing.assert_frame_equal(rolling_features, expected)


@pytest.mark.parametrize(
    "fillna", 
    ['mean', 'median', 'ffill', 'bfill', None, 5., 0], 
    ids = lambda fillna: f'fillna: {fillna}')
def test_RollingFeaturesClassification_transform_batch_fillna_all_methods(fillna):
    """
    Test RollingFeaturesClassification transform_batch method with all fillna methods.
    """
    X_datetime = X_classification.head(10).copy()
    X_datetime.index = pd.date_range(start='1990-01-01', periods=len(X_datetime), freq='D')
    X_datetime.iloc[5] = np.nan

    expected_dict = {
        'mean': np.array([1., 1., 2., 1.5, 1.5, 1.5, 2.]),
        'median': np.array([1., 1., 2., 1.5, 1.5, 1.5, 2.]),
        'ffill': np.array([1., 1., 2., 2., 2., 2., 2.]),
        'bfill': np.array([1., 1., 2., 2., 2., 2., 2.]),
        'None': np.array([1.,  1.,  2., np.nan, np.nan, np.nan,  2.]),
        '5.0': np.array([1., 1., 2., 5., 5., 5., 2.]),
        '0': np.array([1., 1., 2., 0., 0., 0., 2.]),
    } 

    rolling = RollingFeaturesClassification(stats=['mode'], window_sizes=3, fillna=fillna)
    rolling_features = rolling.transform_batch(X_datetime)

    expected = pd.DataFrame(
        data=expected_dict[f'{fillna}'], columns=['roll_mode_3'], 
        index=pd.date_range(start='1990-01-04', periods=7, freq='D')
    )

    pd.testing.assert_frame_equal(rolling_features, expected)


def test_RollingFeaturesClassification_ValueError_transform_without_classes():
    """
    Test RollingFeaturesClassification ValueError is raised when calling transform before
    transform_batch to infer classes.
    """
    rolling = RollingFeaturesClassification(stats='proportion', window_sizes=4)

    err_msg = re.escape(
        "Classes must be specified before calling transform. "
        "Call `transform_batch` first to infer classes from data."
    ) 
    with pytest.raises(ValueError, match = err_msg):
        rolling.transform(X_classification.to_numpy(copy=True))


def test_RollingFeaturesClassification_transform():
    """
    Test RollingFeaturesClassification transform method.
    """
    stats = [
        'proportion', 'mode', 'entropy', 'n_changes', 'n_unique'
    ]
    rolling = RollingFeaturesClassification(stats=stats, window_sizes=4)
    rolling.classes = list(np.sort(X_classification.unique()))
    rolling_features = rolling.transform(X_classification.to_numpy(copy=True))

    expected = np.array([0.25, 0.5, 0.25, 2., 1.5, 3., 3.])

    np.testing.assert_array_almost_equal(rolling_features, expected)


def test_RollingFeaturesClassification_transform_2d():
    """
    Test RollingFeaturesClassification transform method with 2 dimensions.
    """

    X_2d = X_classification.to_numpy(copy=True)
    X_2d = np.tile(X_2d, (2, 1)).T

    stats = [
        'proportion', 'mode', 'entropy', 'n_changes', 'n_unique'
    ]
    rolling = RollingFeaturesClassification(stats=stats, window_sizes=4)
    rolling.classes = list(np.sort(X_classification.unique()))
    rolling_features = rolling.transform(X_2d)

    expected = np.array([0.25, 0.5, 0.25, 2., 1.5, 3., 3.])
    expected = np.array([expected, expected])

    np.testing.assert_array_almost_equal(rolling_features, expected)


def test_RollingFeaturesClassification_transform_with_nans():
    """
    Test RollingFeaturesClassification transform method with nans.
    """

    X_nans = X_classification.to_numpy(copy=True).astype(float)
    X_nans[-7] = np.nan

    stats = [
        'proportion', 'mode', 'entropy', 'n_changes', 'n_unique'
    ]
    window_sizes = [10, 10, 15, 4, 15]
    rolling = RollingFeaturesClassification(stats=stats, window_sizes=window_sizes)
    rolling.classes = list(np.sort(X_classification.unique()))
    rolling_features = rolling.transform(X_nans)

    expected = np.array([0.33333333, 0.55555556, 0.11111111, 2.0, 1.49261407, 3.0, 3.0])

    np.testing.assert_array_almost_equal(rolling_features, expected)


def test_RollingFeaturesClassification_transform_with_nans_2d():
    """
    Test RollingFeaturesClassification transform method with nans and 2 dimensions.
    """

    X_2d_nans = X_classification.to_numpy(copy=True).astype(float)
    X_2d_nans = np.tile(X_2d_nans, (2, 1)).T
    X_2d_nans[-7, 0] = np.nan
    X_2d_nans[-5, 1] = np.nan

    stats = ['proportion', 'mode', 'entropy', 'n_changes', 'n_unique']
    window_sizes = [10, 10, 15, 4, 15]
    rolling = RollingFeaturesClassification(stats=stats, window_sizes=window_sizes)
    rolling.classes = list(np.sort(X_classification.unique()))
    rolling_features = rolling.transform(X_2d_nans)

    expected_0 = np.array(
        [0.33333333, 0.55555556, 0.11111111, 2.0, 1.49261407, 3.0, 3.0]
    )
    expected_1 = np.array(
        [0.33333333, 0.55555556, 0.11111111, 2.0, 1.49261407, 3.0, 3.0]
    )
    expected = np.array([expected_0, expected_1])

    np.testing.assert_array_almost_equal(rolling_features, expected)
