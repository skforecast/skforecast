# Unit test RollingFeatures
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.preprocessing import RollingFeatures

# Fixtures
from .fixtures_preprocessing import X


def test_RollingFeatures_validate_params():
    """
    Test RollingFeatures _validate_params method.
    """

    params = {
        0: {'stats': 5, 'window_sizes': 5},
        1: {'stats': 'not_valid_stat', 'window_sizes': 5},
        2: {'stats': 'mean', 'window_sizes': 'not_int_list'},
        3: {'stats': 'mean', 'window_sizes': [5, 6]},
        4: {'stats': ['mean', 'median'], 'window_sizes': [6]},
        5: {'stats': ['mean', 'median', 'mean'], 'window_sizes': [6, 5, 6]},
        6: {'stats': ['mean'], 'window_sizes': [5], 'min_periods': 'not_int_list'},
        7: {'stats': ['mean'], 'window_sizes': 5, 'min_periods': [5, 4]},
        8: {'stats': ['mean', 'median'], 'window_sizes': 6, 'min_periods': [5]},
        9: {'stats': ['mean', 'median'], 'window_sizes': [5, 3], 'min_periods': [5, 4]},
        10: {'stats': ['mean', 'median'], 'window_sizes': [5, 6], 
             'min_periods': None, 'features_names': 'not_list'},
        11: {'stats': ['mean'], 'window_sizes': 5, 
             'min_periods': 5, 'features_names': ['mean_5', 'median_6']},
        12: {'stats': ['mean', 'median'], 'window_sizes': [5, 6], 
             'min_periods': 4, 'features_names': ['mean_5']},
        13: {'stats': ['mean', 'median'], 'window_sizes': [5, 6], 
             'min_periods': 5, 'features_names': ['mean_5', 'median_6'], 'fillna': {}},
        14: {'stats': ['mean', 'median'], 'window_sizes': [5, 6], 
             'min_periods': [5, 5], 'features_names': None, 'fillna': 'not_valid_fillna'},
    }
    
    # stats
    err_msg = re.escape(
        f"`stats` must be a string or a list of strings. Got {type(params[0]['stats'])}."
    ) 
    with pytest.raises(TypeError, match = err_msg):
        RollingFeatures(**params[0])
    err_msg = re.escape(
       ("Statistic 'not_valid_stat' is not allowed. Allowed stats are: ['mean', "
        "'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation'].")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[1])

    # window_sizes
    err_msg = re.escape(
        f"`window_sizes` must be an int or a list of ints. Got {type(params[2]['window_sizes'])}."
    ) 
    with pytest.raises(TypeError, match = err_msg):
        RollingFeatures(**params[2])
    err_msg = re.escape(
        ("Length of `window_sizes` list (2) "
         "must match length of `stats` list (1).")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[3])
    err_msg = re.escape(
        ("Length of `window_sizes` list (1) "
         "must match length of `stats` list (2).")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[4])
    
    # Check duplicates (stats, window_sizes)
    err_msg = re.escape("Duplicate (stat, window_size) pairs are not allowed.")
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[5])

    # min_periods
    err_msg = re.escape(
        f"`min_periods` must be an int, list of ints, or None. Got {type(params[6]['min_periods'])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        RollingFeatures(**params[6])
    err_msg = re.escape(
        ("Length of `min_periods` list (2) "
         "must match length of `stats` list (1).")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[7])
    err_msg = re.escape(
        ("Length of `min_periods` list (1) "
         "must match length of `stats` list (2).")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[8])
    err_msg = re.escape(
        ("Each min_period must be less than or equal to its "
         "corresponding window_size.")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[9])

    # features_names
    err_msg = re.escape(
        f"`features_names` must be a list of strings or None. Got {type(params[10]['features_names'])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        RollingFeatures(**params[10])
    err_msg = re.escape(
        ("Length of `features_names` list (2) "
         "must match length of `stats` list (1).")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[11])
    err_msg = re.escape(
        ("Length of `features_names` list (1) "
         "must match length of `stats` list (2).")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[12])

    # fillna
    err_msg = re.escape(
        f"`fillna` must be a float, string, or None. Got {type(params[13]['fillna'])}."
    )
    with pytest.raises(TypeError, match = err_msg):
        RollingFeatures(**params[13])
    err_msg = re.escape(
        ("'not_valid_fillna' is not allowed. Allowed `fillna` "
         "values are: ['mean', 'median', 'ffill', 'bfill'] or a float value.")
    ) 
    with pytest.raises(ValueError, match = err_msg):
        RollingFeatures(**params[14])


@pytest.mark.parametrize(
        "params", 
        [{'stats': 'mean', 'window_sizes': 5, 
          'min_periods': None, 'features_names': None, 'fillna': 'ffill'},
         {'stats': ['mean'], 'window_sizes': [5], 
          'min_periods': 5, 'features_names': ['roll_mean_5'], 'fillna': 'ffill'}], 
        ids = lambda params: f'params: {params}')
def test_RollingFeatures_init_store_parameters(params):
    """
    Test RollingFeatures initialization and stored parameters.
    """

    rolling = RollingFeatures(**params)

    assert rolling.stats == ['mean']
    assert rolling.n_stats == 1
    assert rolling.window_sizes == [5]
    assert rolling.max_window_size == 5
    assert rolling.min_periods == [5]
    assert rolling.features_names == ['roll_mean_5']
    assert rolling.fillna == 'ffill'

    unique_rolling_windows = {
        '5_5': {'params': {'window': 5, 'min_periods': 5, 'center': False, 'closed': 'left'}, 
                'stats_idx': [0],
                'stats_names': ['roll_mean_5'],
                'rolling_obj': None}
    }

    assert rolling.unique_rolling_windows == unique_rolling_windows


def test_RollingFeatures_init_store_parameters_multiple_stats():
    """
    Test RollingFeatures initialization and stored parameters 
    when multiple stats are passed.
    """

    rolling = RollingFeatures(stats=['mean', 'median', 'sum'], window_sizes=[5, 5, 6])

    assert rolling.stats == ['mean', 'median', 'sum']
    assert rolling.n_stats == 3
    assert rolling.window_sizes == [5, 5, 6]
    assert rolling.max_window_size == 6
    assert rolling.min_periods == [5, 5, 6]
    assert rolling.features_names == ['roll_mean_5', 'roll_median_5', 'roll_sum_6']
    assert rolling.fillna is None

    unique_rolling_windows = {
        '5_5': {'params': {'window': 5, 'min_periods': 5, 'center': False, 'closed': 'left'}, 
                'stats_idx': [0, 1],
                'stats_names': ['roll_mean_5', 'roll_median_5'],
                'rolling_obj': None},
        '6_6': {'params': {'window': 6, 'min_periods': 6, 'center': False, 'closed': 'left'}, 
                'stats_idx': [2],
                'stats_names': ['roll_sum_6'],
                'rolling_obj': None}
    }

    assert rolling.unique_rolling_windows == unique_rolling_windows


def test_RollingFeatures_ValueError_apply_stat_when_stat_not_implemented():
    """
    Test RollingFeatures ValueError _apply_stat_pandas and _apply_stat_numpy 
    when applying a statistic not implemented.
    """
    
    rolling = RollingFeatures(stats='mean', window_sizes=10)
    X_window = X.iloc[-10:]
    rolling_obj = X_window.rolling(**rolling.unique_rolling_windows['10_10']['params'])

    err_msg = re.escape("Statistic 'not_valid' is not implemented.") 
    with pytest.raises(ValueError, match = err_msg):
        rolling._apply_stat_pandas(rolling_obj, 'not_valid')
    with pytest.raises(ValueError, match = err_msg):
        rolling._apply_stat_numpy_jit(X_window.to_numpy(), 'not_valid')


def test_RollingFeatures_apply_stat_pandas_numpy():
    """
    Test RollingFeatures _apply_stat_pandas and _apply_stat_numpy methods.
    """

    stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 
             'ratio_min_max', 'coef_variation']
    
    rolling = RollingFeatures(stats=stats, window_sizes=10)
    X_window_pandas = X.iloc[-11:]
    X_window_numpy = X.to_numpy()[-11:-1]

    for stat in stats:

        rolling_obj = X_window_pandas.rolling(**rolling.unique_rolling_windows['10_10']['params'])
        stat_pandas = rolling._apply_stat_pandas(rolling_obj, stat).iat[-1]
        stat_numpy = rolling._apply_stat_numpy_jit(X_window_numpy, stat)

        np.testing.assert_almost_equal(stat_pandas, stat_numpy, decimal=7)


def test_RollingFeatures_transform_batch():
    """
    Test RollingFeatures transform_batch method.
    """
    X_datetime = X.copy()
    X_datetime.index = pd.date_range(start='1990-01-01', periods=len(X), freq='D')

    stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 
             'ratio_min_max', 'coef_variation']
    rolling = RollingFeatures(stats=stats, window_sizes=4)
    rolling_features = rolling.transform_batch(X_datetime).head(10)
    
    expected = pd.DataFrame(
        data = np.array([[0.44019369, 0.22156465, 0.22685145, 0.69646919, 1.76077474,
                          0.41872705, 0.32571642, 0.50333446],
                         [0.44594363, 0.23054861, 0.22685145, 0.71946897, 1.78377452,
                          0.41872705, 0.31530401, 0.51699048],
                         [0.48018541, 0.20796803, 0.22685145, 0.71946897, 1.92074165,
                          0.48721061, 0.31530401, 0.43309944],
                         [0.6686636 , 0.24087135, 0.42310646, 0.9807642 , 2.6746544 ,
                          0.63539187, 0.43140488, 0.360228  ],
                         [0.70204234, 0.22810163, 0.42310646, 0.9807642 , 2.80816937,
                          0.70214935, 0.43140488, 0.3249115 ],
                         [0.64240807, 0.25196046, 0.42310646, 0.9807642 , 2.5696323 ,
                          0.58288082, 0.43140488, 0.39221247],
                         [0.63466084, 0.26125613, 0.39211752, 0.9807642 , 2.53864336,
                          0.58288082, 0.39980815, 0.41164685],
                         [0.4752643 , 0.15089728, 0.34317802, 0.68482974, 1.90105718,
                          0.43652471, 0.50111436, 0.31750182],
                         [0.48631929, 0.17157163, 0.34317802, 0.72904971, 1.94527715,
                          0.43652471, 0.47071964, 0.35279628],
                         [0.47572937, 0.17331344, 0.34317802, 0.72904971, 1.90291749,
                          0.41534488, 0.47071964, 0.364311  ]]),
        columns = ['roll_mean_4', 'roll_std_4', 'roll_min_4', 'roll_max_4',
                   'roll_sum_4', 'roll_median_4', 'roll_ratio_min_max_4',
                   'roll_coef_variation_4'],
        index = pd.date_range(start='1990-01-05', periods=10, freq='D')
    )

    pd.testing.assert_frame_equal(rolling_features, expected)


def test_RollingFeatures_transform_batch_different_rolling_and_fillna():
    """
    Test RollingFeatures transform_batch method with different rolling windows 
    and fillna.
    """
    X_datetime = X.copy()
    X_datetime.index = pd.date_range(start='1990-01-01', periods=len(X), freq='D')
    X_datetime.iloc[9] = np.nan

    stats = ['mean', 'std', 'mean', 'max']
    window_sizes = [4, 5, 6, 4]
    min_periods = [3, 5, 6, 3]
    features_names = ['my_mean', 'my_std', 'my_mean_2', 'my_max']
    
    rolling = RollingFeatures(
        stats=stats, window_sizes=window_sizes, min_periods=min_periods,
        features_names=features_names, fillna='bfill'
    )
    rolling_features = rolling.transform_batch(X_datetime).head(15)
    
    expected = pd.DataFrame(
        data = np.array([
                   [0.48018541, 0.19992199, 0.4838917 , 0.71946897],
                   [0.6686636 , 0.28732186, 0.5312742 , 0.9807642 ],
                   [0.70204234, 0.20872596, 0.5977226 , 0.9807642 ],
                   [0.64240807, 0.22090887, 0.64006934, 0.9807642 ],
                   [0.71550861, 0.23906853, 0.45108626, 0.9807642 ],
                   [0.50297989, 0.23906853, 0.45108626, 0.68482974],
                   [0.51771988, 0.23906853, 0.45108626, 0.72904971],
                   [0.50359999, 0.23906853, 0.45108626, 0.72904971],
                   [0.39261947, 0.23906853, 0.45108626, 0.72904971],
                   [0.40633603, 0.23906853, 0.45108626, 0.72904971],
                   [0.40857245, 0.27992064, 0.45108626, 0.73799541],
                   [0.34455232, 0.2608389 , 0.42430521, 0.73799541],
                   [0.37349579, 0.26830576, 0.33203888, 0.73799541],
                   [0.40687257, 0.23934903, 0.3475354 , 0.73799541],
                   [0.35533061, 0.24575419, 0.42622702, 0.53182759]]),
        columns = ['my_mean', 'my_std', 'my_mean_2', 'my_max'],
        index = pd.date_range(start='1990-01-07', periods=15, freq='D')
    )

    pd.testing.assert_frame_equal(rolling_features, expected)


@pytest.mark.parametrize(
        "fillna", 
        ['mean', 'median', 'ffill', 'bfill', None, 5., 0], 
        ids = lambda fillna: f'fillna: {fillna}')
def test_RollingFeatures_transform_batch_fillna_all_methods(fillna):
    """
    Test RollingFeatures transform_batch method with all fillna methods.
    """
    X_datetime = X.head(10).copy()
    X_datetime.index = pd.date_range(start='1990-01-01', periods=len(X_datetime), freq='D')
    X_datetime.iloc[5] = np.nan

    base_array = np.array([0.40315332, 0.35476852, 0.49921173, 
                           np.nan, np.nan, np.nan, 0.715509])
    expected_dict = {
        'mean': np.array([0.49316055, 0.49316055, 0.49316055]),
        'median': np.array([0.45118253, 0.45118253, 0.45118253]),
        'ffill': np.array([0.49921173, 0.49921173, 0.49921173]),
        'bfill': np.array([0.71550861, 0.71550861, 0.71550861]),
        5.: np.array([5., 5., 5.]),
        0: np.array([0, 0, 0]),
    } 

    rolling = RollingFeatures(stats=['mean'], window_sizes=3, fillna=fillna)
    rolling_features = rolling.transform_batch(X_datetime)

    expected_array = base_array
    if fillna is not None:
        expected_array[-4:-1] = expected_dict[fillna]
    expected = pd.DataFrame(
        data=expected_array, columns=['roll_mean_3'], 
        index=pd.date_range(start='1990-01-04', periods=7, freq='D')
    )

    pd.testing.assert_frame_equal(rolling_features, expected)


def test_RollingFeatures_transform():
    """
    Test RollingFeatures transform method.
    """

    stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 
             'ratio_min_max', 'coef_variation']
    rolling = RollingFeatures(stats=stats, window_sizes=4)
    rolling_features = rolling.transform(X.to_numpy(copy=True))
    
    expected = np.array([0.65024343, 0.23013666, 0.48303426, 0.98555979, 
                         2.6009737 , 0.56618983, 0.49011157, 0.35392385])

    np.testing.assert_array_almost_equal(rolling_features, expected)


def test_RollingFeatures_transform_2d():
    """
    Test RollingFeatures transform method with 2 dimensions.
    """

    X_2d = X.to_numpy(copy=True)
    X_2d = np.tile(X_2d, (2, 1)).T

    stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 
             'ratio_min_max', 'coef_variation']
    rolling = RollingFeatures(stats=stats, window_sizes=4)
    rolling_features = rolling.transform(X_2d)
    
    expected = np.array([0.65024343, 0.23013666, 0.48303426, 0.98555979, 
                         2.6009737 , 0.56618983, 0.49011157, 0.35392385])
    expected = np.array([expected, expected])

    np.testing.assert_array_almost_equal(rolling_features, expected)


def test_RollingFeatures_transform_with_nans():
    """
    Test RollingFeatures transform method with nans.
    """
    
    X_nans = X.to_numpy(copy=True)
    X_nans[-7] = np.nan

    stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 
             'ratio_min_max', 'coef_variation']
    window_sizes = [10, 10, 15, 4, 15, 4, 4, 4]
    rolling = RollingFeatures(stats=stats, window_sizes=window_sizes)
    rolling_features = rolling.transform(X_nans)
    
    expected = np.array([0.53051056, 0.28145959, 0.11561840, 0.98555979, 
                         7.85259345, 0.56618983, 0.49011157, 0.35392385])

    np.testing.assert_array_almost_equal(rolling_features, expected)


def test_RollingFeatures_transform_with_nans_2d():
    """
    Test RollingFeatures transform method with nans and 2 dimensions.
    """
    
    X_2d_nans = X.to_numpy(copy=True)
    X_2d_nans = np.tile(X_2d_nans, (2, 1)).T
    X_2d_nans[-7, 0] = np.nan
    X_2d_nans[-5, 1] = np.nan

    stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 
             'ratio_min_max', 'coef_variation']
    window_sizes = [10, 10, 15, 4, 15, 4, 4, 4]
    rolling = RollingFeatures(stats=stats, window_sizes=window_sizes)
    rolling_features = rolling.transform(X_2d_nans)
    
    expected_0 = np.array([0.53051056, 0.28145959, 0.11561840, 0.98555979, 
                           7.85259345, 0.56618983, 0.49011157, 0.35392385])
    expected_1 = np.array([0.548774, 0.26592 , 0.115618, 0.98556, 
                           8.016964, 0.56618983, 0.49011157, 0.35392385])
    expected = np.array([expected_0, expected_1])

    np.testing.assert_array_almost_equal(rolling_features, expected)
