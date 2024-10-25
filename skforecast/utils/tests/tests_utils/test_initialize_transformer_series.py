# Unit test initialize_transformer_series
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skforecast.utils import initialize_transformer_series
from skforecast.exceptions import IgnoredArgumentWarning


@pytest.mark.parametrize(
    "forecaster_name, encoding, expected", 
    [('ForecasterRecursiveMultiSeries', 'onehot', {'col1': None, 'col2': None, '_unknown_level': None}),
     ('ForecasterRecursiveMultiSeries', None, {'_unknown_level': None}),
     ('ForecasterDirectMultiVariate', None, {'col1': None, 'col2': None}),
     ('ForecasterRNN', None, {'col1': None, 'col2': None})], 
    ids=lambda params: f'params: {params}')
def test_initialize_transformer_series_when_transformer_series_is_None(forecaster_name, encoding, expected):
    """
    Test initialize_transformer_series when `transformer_series` is None.
    """
    series_names_in_ = ['col1', 'col2']
    transformer_series = None
    
    transformer_series_ = initialize_transformer_series(
                              forecaster_name    = forecaster_name,
                              series_names_in_   = series_names_in_,
                              encoding           = encoding,
                              transformer_series = transformer_series
                          )

    assert transformer_series_ == expected


@pytest.mark.parametrize(
    "forecaster_name, encoding, expected", 
    [('ForecasterRecursiveMultiSeries', 'onehot', {'col1': StandardScaler(), 'col2': StandardScaler(), '_unknown_level': StandardScaler()}),
     ('ForecasterRecursiveMultiSeries', None, {'_unknown_level': StandardScaler()}),
     ('ForecasterDirectMultiVariate', None, {'col1': StandardScaler(), 'col2': StandardScaler()}),
     ('ForecasterRNN', None, {'col1': StandardScaler(), 'col2': StandardScaler()})], 
    ids=lambda params: f'params: {params}')
def test_initialize_transformer_series_when_transformer_series_is_StandardScaler(forecaster_name, encoding, expected):
    """
    Test initialize_transformer_series when `transformer_series` is a StandardScaler.
    """
    series_names_in_ = ['col1', 'col2']
    transformer_series = StandardScaler()
    
    transformer_series_ = initialize_transformer_series(
                              forecaster_name    = forecaster_name,
                              series_names_in_   = series_names_in_,
                              encoding           = encoding,
                              transformer_series = transformer_series
                          )

    assert transformer_series_.keys() == expected.keys()
    for k in expected.keys():
        assert isinstance(transformer_series_[k], StandardScaler)


@pytest.mark.parametrize(
    "forecaster_name, encoding, expected", 
    [('ForecasterRecursiveMultiSeries', 'onehot', {'col1': StandardScaler(), 'col2': StandardScaler(), '_unknown_level': StandardScaler()}),
     ('ForecasterDirectMultiVariate', None, {'col1': StandardScaler(), 'col2': StandardScaler()}),
     ('ForecasterRNN', None, {'col1': StandardScaler(), 'col2': StandardScaler()})], 
    ids=lambda params: f'params: {params}')
def test_initialize_transformer_series_when_transformer_series_is_dict(forecaster_name, encoding, expected):
    """
    Test initialize_transformer_series when `transformer_series` is a dict.
    Encoding can not be None when transformer_series is a dict.
    """
    series_names_in_ = ['col1', 'col2']
    if forecaster_name == 'ForecasterRecursiveMultiSeries':
        transformer_series = {'col1': StandardScaler(), 
                              'col2': StandardScaler(), 
                              '_unknown_level': StandardScaler()}
    else:
        transformer_series = {'col1': StandardScaler(), 
                              'col2': StandardScaler()}
    
    transformer_series_ = initialize_transformer_series(
                              forecaster_name    = forecaster_name,
                              series_names_in_   = series_names_in_,
                              encoding           = encoding,
                              transformer_series = transformer_series
                          )

    assert transformer_series_.keys() == expected.keys()
    for k in expected.keys():
        assert isinstance(transformer_series_[k], StandardScaler)


@pytest.mark.parametrize(
    "forecaster_name, encoding, expected", 
    [('ForecasterRecursiveMultiSeries', 'onehot', {'col1': StandardScaler(), 'col2': None, '_unknown_level': StandardScaler()}),
     ('ForecasterDirectMultiVariate', None, {'col1': StandardScaler(), 'col2': None}),
     ('ForecasterRNN', None, {'col1': StandardScaler(), 'col2': None})], 
    ids=lambda params: f'params: {params}')
def test_initialize_transformer_series_IgnoredArgumentWarning_when_levels_of_transformer_series_not_equal_to_series_names_in_(forecaster_name, encoding, expected):
    """
    Test IgnoredArgumentWarning is raised when `transformer_series` is a dict and its keys 
    are not the same as series_names_in_.
    """
    series_names_in_ = ['col1', 'col2']
    if forecaster_name == 'ForecasterRecursiveMultiSeries':
        transformer_series = {'col1': StandardScaler(), 
                              'col3': StandardScaler(), 
                              '_unknown_level': StandardScaler()}
    else:
        transformer_series = {'col1': StandardScaler(), 
                              'col3': StandardScaler()}
    
    series_not_in_transformer_series = set(['col2'])
    warn_msg = re.escape(
        (f"{series_not_in_transformer_series} not present in `transformer_series`."
         f" No transformation is applied to these series.")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        transformer_series_ = initialize_transformer_series(
                                  forecaster_name    = forecaster_name,
                                  series_names_in_   = series_names_in_,
                                  encoding           = encoding,
                                  transformer_series = transformer_series
                              )

    assert transformer_series_.keys() == expected.keys()
    for k in expected.keys():
        if expected[k] is None:
            assert isinstance(transformer_series_[k], type(None))
        else:
            assert isinstance(transformer_series_[k], StandardScaler)