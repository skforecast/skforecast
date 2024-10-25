# Unit test fit ForecasterRecursiveMultiSeries
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from skforecast.preprocessing import RollingFeatures
from ....recursive import ForecasterRecursiveMultiSeries


def test_fit_correct_dict_create_series_weights_weight_func_transformer_series():
    """
    Test fit method creates correctly all the auxiliary dicts, series_weights_,
    weight_func_, transformer_series_.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10)), 
                           'l2': pd.Series(np.arange(10)), 
                           'l3': pd.Series(np.arange(10))})
                    
    series.index = pd.DatetimeIndex(
                       ['2022-01-04', '2022-01-05', '2022-01-06', 
                        '2022-01-07', '2022-01-08', '2022-01-09', 
                        '2022-01-10', '2022-01-11', '2022-01-12', 
                        '2022-01-13'], dtype='datetime64[ns]', freq='D' 
                   )

    def custom_weights(index):  # pragma: no cover
        """
        Return 0 if index is between '2022-01-08' and '2022-01-10', 1 otherwise.
        """
        weights = np.where(
                      (index >= '2022-01-08') & (index <= '2022-01-10'),
                      0,
                      1
                  )
        
        return weights

    transformer_series = {'l1': StandardScaler(), '_unknown_level': StandardScaler()}
    weight_func = {'l2': custom_weights}    
    series_weights = {'l1': 3., 'l3': 0.5, 'l4': 2.}

    forecaster = ForecasterRecursiveMultiSeries(
                     regressor          = LinearRegression(), 
                     lags               = 3,
                     transformer_series = transformer_series,
                     weight_func        = weight_func,
                     series_weights     = series_weights
                 )
    
    forecaster.fit(series=series, store_in_sample_residuals=False)

    expected_transformer_series_ = {
        'l1': forecaster.transformer_series_['l1'], 
        'l2': None, 
        'l3': None,
        '_unknown_level': forecaster.transformer_series_['_unknown_level']
    }
    expected_weight_func_ = {'l1': lambda index: np.ones_like(index, dtype=float), 'l2': custom_weights, 'l3': lambda index: np.ones_like(index, dtype=float)}
    expected_series_weights_ = {'l1': 3., 'l2': 1., 'l3': 0.5}

    assert forecaster.transformer_series_.keys() == expected_transformer_series_.keys()
    for k in expected_transformer_series_.keys():
        if expected_transformer_series_[k] is None:
            assert isinstance(forecaster.transformer_series_[k], type(None))
        else:
            assert isinstance(forecaster.transformer_series_[k], StandardScaler)
    assert forecaster.weight_func_.keys() == expected_weight_func_.keys()
    for key in forecaster.weight_func_.keys():
        assert forecaster.weight_func_[key].__code__.co_code == expected_weight_func_[key].__code__.co_code
    assert forecaster.series_weights_ == expected_series_weights_

    forecaster.fit(series=series[['l1', 'l2']], store_in_sample_residuals=False)

    expected_transformer_series_ = {
        'l1': forecaster.transformer_series_['l1'], 
        'l2': None,
        '_unknown_level': forecaster.transformer_series_['_unknown_level']
    }
    expected_weight_func_ = {'l1': lambda index: np.ones_like(index, dtype=float), 'l2': custom_weights}
    expected_series_weights_ = {'l1': 3., 'l2': 1.}

    assert forecaster.transformer_series_.keys() == expected_transformer_series_.keys()
    for k in expected_transformer_series_.keys():
        if expected_transformer_series_[k] is None:
            assert isinstance(forecaster.transformer_series_[k], type(None))
        else:
            assert isinstance(forecaster.transformer_series_[k], StandardScaler)
    assert forecaster.weight_func_.keys() == expected_weight_func_.keys()
    for key in forecaster.weight_func_.keys():
        assert forecaster.weight_func_[key].__code__.co_code == expected_weight_func_[key].__code__.co_code
    assert forecaster.series_weights_ == expected_series_weights_


def test_forecaster_DatetimeIndex_index_freq_stored():
    """
    Test serie_with_DatetimeIndex.index.freqstr is stored in forecaster.index_freq_.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})

    series.index = pd.date_range(start='2022-01-01', periods=5, freq='1D')

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    results = forecaster.index_freq_

    expected = series.index.freqstr

    assert results == expected


def test_forecaster_index_step_stored():
    """
    Test serie without DatetimeIndex, step is stored in forecaster.index_freq_.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)
    results = forecaster.index_freq_

    expected = series.index.step

    assert results == expected


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'ordinal_category', 'onehot'], 
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_in_sample_residuals_stored(encoding):
    """
    Test that values of in_sample_residuals_ are stored after fitting
    when `store_in_sample_residuals=True`.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})

    rolling = RollingFeatures(
        stats=['ratio_min_max', 'median'], window_sizes=4
    )
    forecaster = ForecasterRecursiveMultiSeries(
        LinearRegression(), lags=3, encoding=encoding, window_features=rolling
    )
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster.in_sample_residuals_

    expected = {'1': np.array([-4.4408921e-16, 0.0000000e+00]),
                '2': np.array([0., 0.]),
                '_unknown_level': np.array([-4.4408921e-16, 0.0000000e+00, 0., 0.])}
    
    X_train_window_features_names_out_ = ['roll_ratio_min_max_4', 'roll_median_4']
    X_train_features_names_out_ = (
        ['lag_1', 'lag_2', 'lag_3', 'roll_ratio_min_max_4', 'roll_median_4', '_level_skforecast'] 
        if encoding != 'onehot' 
        else ['lag_1', 'lag_2', 'lag_3', 'roll_ratio_min_max_4', 'roll_median_4', '1', '2']
    )

    assert forecaster.series_names_in_ == ['1', '2']
    assert forecaster.X_train_series_names_in_ == ['1', '2']
    assert forecaster.X_train_window_features_names_out_ == X_train_window_features_names_out_
    assert forecaster.X_train_features_names_out_ == X_train_features_names_out_
    assert isinstance(results, dict)
    assert np.all(isinstance(x, np.ndarray) for x in results.values())
    assert results.keys() == expected.keys()
    assert np.all(np.all(np.isclose(results[k], expected[k])) for k in expected.keys())


def test_fit_in_sample_residuals_stored_encoding_None():
    """
    Test that values of in_sample_residuals_ are stored after fitting
    when `store_in_sample_residuals=True` and encoding=None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=None)
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results = forecaster.in_sample_residuals_

    expected = {'_unknown_level': np.array([-4.4408921e-16, 0.0000000e+00, 0., 0.])}

    assert forecaster.series_names_in_ == ['1', '2']
    assert forecaster.X_train_series_names_in_ == ['1', '2']
    assert forecaster.X_train_features_names_out_ == ['lag_1', 'lag_2', 'lag_3']
    assert isinstance(results, dict)
    assert np.all(isinstance(x, np.ndarray) for x in results.values())
    assert results.keys() == expected.keys()
    assert np.all(np.all(np.isclose(results[k], expected[k])) for k in expected.keys())


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'ordinal_category', 'onehot'], 
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_same_residuals_when_residuals_greater_than_1000(encoding):
    """
    Test fit return same residuals when residuals len is greater than 1000.
    Testing with two different forecaster. Residuals shouldn't be more than 
    1000 values.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(1010)), 
                           '2': pd.Series(np.arange(1010))})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=encoding)
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_1 = forecaster.in_sample_residuals_
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=encoding)
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_2 = forecaster.in_sample_residuals_

    assert isinstance(results_1, dict)
    assert np.all(isinstance(x, np.ndarray) for x in results_1.values())
    assert isinstance(results_2, dict)
    assert np.all(isinstance(x, np.ndarray) for x in results_2.values())
    assert results_1.keys() == results_2.keys()
    assert np.all(len(results_1[k] == 1000) for k in results_1.keys())
    assert np.all(len(results_2[k] == 1000) for k in results_2.keys())
    assert np.all(np.all(results_1[k] == results_2[k]) for k in results_2.keys())


def test_fit_same_residuals_when_residuals_greater_than_1000_encoding_None():
    """
    Test fit return same residuals when residuals len is greater than 1000.
    Testing with two different forecaster. Residuals shouldn't be more than 
    1000 values and encoding=None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(1010)), 
                           '2': pd.Series(np.arange(1010))})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=None)
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_1 = forecaster.in_sample_residuals_
    
    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=None)
    forecaster.fit(series=series, store_in_sample_residuals=True)
    results_2 = forecaster.in_sample_residuals_

    assert isinstance(results_1, dict)
    assert np.all(isinstance(x, np.ndarray) for x in results_1.values())
    assert isinstance(results_2, dict)
    assert np.all(isinstance(x, np.ndarray) for x in results_2.values())
    assert results_1.keys() == results_2.keys()
    assert np.all(len(results_1[k] == 1000) for k in results_1.keys())
    assert np.all(len(results_2[k] == 1000) for k in results_2.keys())
    assert np.all(np.all(results_1[k] == results_2[k]) for k in results_2.keys())


@pytest.mark.parametrize("encoding", 
                         ['ordinal', 'ordinal_category', 'onehot'], 
                         ids = lambda encoding: f'encoding: {encoding}')
def test_fit_in_sample_residuals_not_stored(encoding):
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False`.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=encoding)
    forecaster.fit(series=series, store_in_sample_residuals=False)
    results = forecaster.in_sample_residuals_

    expected = {'1': None, '2': None, '_unknown_level': None}

    assert isinstance(results, dict)
    assert results.keys() == expected.keys()
    assert np.all(results[k] == expected[k] for k in expected.keys())


def test_fit_in_sample_residuals_not_stored_encoding_None():
    """
    Test that values of in_sample_residuals_ are not stored after fitting
    when `store_in_sample_residuals=False` and encoding=None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5)), 
                           '2': pd.Series(np.arange(5))})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3,
                                              encoding=None)
    forecaster.fit(series=series, store_in_sample_residuals=False)
    results = forecaster.in_sample_residuals_

    expected = {'_unknown_level': None}

    assert isinstance(results, dict)
    assert results.keys() == expected.keys()
    assert np.all(results[k] == expected[k] for k in expected.keys())


def test_fit_last_window_stored():
    """
    Test that values of last window are stored after fitting.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(5, dtype=float)), 
                           '2': pd.Series(np.arange(5, dtype=float))})

    forecaster = ForecasterRecursiveMultiSeries(LinearRegression(), lags=3)
    forecaster.fit(series=series)

    expected = {
        '1': pd.Series(
                 data  = np.array([2., 3., 4.]),
                 index = pd.RangeIndex(start=2, stop=5, step=1),
                 name  = '1',
                 dtype = float
             ),
        '2': pd.Series(
                 data  = np.array([2., 3., 4.]),
                 index = pd.RangeIndex(start=2, stop=5, step=1),
                 name  = '2',
                 dtype = float
             )
    }

    for k in forecaster.last_window_.keys():
        pd.testing.assert_series_equal(forecaster.last_window_[k], expected[k])


@pytest.mark.parametrize("encoding, encoding_mapping_", 
                         [('ordinal'         , {'1': 0, '2': 1}), 
                          ('ordinal_category', {'1': 0, '2': 1}),
                          ('onehot'          , {'1': 0, '2': 1}),
                          (None              , {'1': 0, '2': 1})], 
                         ids = lambda dt: f'encoding, mapping: {dt}')
def test_fit_encoding_mapping(encoding, encoding_mapping_):
    """
    Test the encoding mapping of _create_train_X_y.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    
    forecaster = ForecasterRecursiveMultiSeries(
                     LinearRegression(),
                     lags     = 3,
                     encoding = encoding,
                 )
    forecaster.fit(series=series, suppress_warnings=True)
    
    assert forecaster.encoding_mapping_ == encoding_mapping_