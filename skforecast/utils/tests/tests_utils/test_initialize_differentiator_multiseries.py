# Unit test initialize_differentiator_multiseries
# ==============================================================================
import re
import pytest
from skforecast.preprocessing import TimeSeriesDifferentiator
from skforecast.utils import initialize_differentiator_multiseries
from skforecast.exceptions import IgnoredArgumentWarning


def test_initialize_differentiator_when_differentiator_is_None():
    """
    Test initialize_differentiator_multiseries when `differentiator` is None.
    """
    
    differentiator_ = initialize_differentiator_multiseries(
                          series_names_in_ = ['col1', 'col2'],
                          differentiator   = None
                      )

    assert differentiator_ == {'col1': None, 'col2': None}


def test_initialize_differentiator_when_differentiator_is_TimeSeriesDifferentiator():
    """
    Test initialize_differentiator_multiseries when `differentiator` is a TimeSeriesDifferentiator.
    """
    series_names_in_ = ['col1', 'col2']
    differentiator = TimeSeriesDifferentiator(order=1, window_size=5)
    
    differentiator_ = initialize_differentiator_multiseries(
                          series_names_in_ = series_names_in_,
                          differentiator   = differentiator
                      )
    
    expected = {'col1': differentiator, 'col2': differentiator}

    assert differentiator_.keys() == expected.keys()
    for k in expected.keys():
        assert isinstance(differentiator_[k], TimeSeriesDifferentiator)
        assert differentiator_[k].order == expected[k].order
        assert differentiator_[k].window_size == expected[k].window_size


def test_initialize_differentiator_when_differentiator_is_dict():
    """
    Test initialize_differentiator_multiseries when `differentiator` is a dict.
    """
    series_names_in_ = ['col1', 'col2', 'col3']
    differentiator = {
        'col1': TimeSeriesDifferentiator(order=1, window_size=5),
        'col2': TimeSeriesDifferentiator(order=2, window_size=5)
    }
    
    differentiator_ = initialize_differentiator_multiseries(
                          series_names_in_ = series_names_in_,
                          differentiator   = differentiator
                      )
    
    expected = {
        'col1': TimeSeriesDifferentiator(order=1, window_size=5),
        'col2': TimeSeriesDifferentiator(order=2, window_size=5),
        'col3': None
    }

    assert differentiator_.keys() == expected.keys()
    for k in expected.keys():
        if expected[k] is None:
            assert isinstance(differentiator_[k], type(None))
        else:
            assert isinstance(differentiator_[k], TimeSeriesDifferentiator)
            assert differentiator_[k].order == expected[k].order
            assert differentiator_[k].window_size == expected[k].window_size


def test_initialize_differentiator_multiseries_IgnoredArgumentWarning_when_keys_of_differentiator_not_equal_to_series_names_in_():
    """
    Test IgnoredArgumentWarning is raised when `differentiator` is a dict and its keys 
    are not the same as `series_names_in_`.
    """
    series_names_in_ = ['col1', 'col2']
    differentiator = {
        'col1': TimeSeriesDifferentiator(order=1, window_size=5),
        'col3': TimeSeriesDifferentiator(order=2, window_size=5)
    }
    
    series_not_in_differentiator = set(['col2'])
    warn_msg = re.escape(
        f"{series_not_in_differentiator} not present in `differentiation`."
        f" No differentiation is applied to these series."
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        differentiator_ = initialize_differentiator_multiseries(
                              series_names_in_ = series_names_in_,
                              differentiator   = differentiator
                          )
    
    expected = {
        'col1': TimeSeriesDifferentiator(order=1, window_size=5),
        'col2': None
    }

    assert differentiator_.keys() == expected.keys()
    for k in expected.keys():
        if expected[k] is None:
            assert isinstance(differentiator_[k], type(None))
        else:
            assert isinstance(differentiator_[k], TimeSeriesDifferentiator)
            assert differentiator_[k].order == expected[k].order
            assert differentiator_[k].window_size == expected[k].window_size
