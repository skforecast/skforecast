# Unit test input_to_frame
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import input_to_frame


@pytest.mark.parametrize("input_name, expected_name", 
                         [('y', 'y'),
                          ('last_window', 'y'),
                          ('exog', 'exog'),
                          ('exog_val', 'exog')], 
                         ids = lambda x: f'{x}')
def test_input_to_frame_when_Series_without_name(input_name, expected_name):
    """
    Test input_to_frame names the column of a Series without name according to
    `input_name`. Before, 'exog_val' (used by ForecasterRnn) raised a KeyError.
    """
    data = pd.Series(np.arange(3))
    results = input_to_frame(data=data, input_name=input_name)
    expected = pd.DataFrame({expected_name: np.arange(3)})

    pd.testing.assert_frame_equal(results, expected)


@pytest.mark.parametrize("input_name", 
                         ['y', 'last_window', 'exog', 'exog_val'], 
                         ids = lambda x: f'{x}')
def test_input_to_frame_when_Series_with_name_or_DataFrame(input_name):
    """
    Test input_to_frame keeps the name of a Series and returns a DataFrame as is.
    """
    data = pd.Series(np.arange(3), name='my_series')
    results = input_to_frame(data=data, input_name=input_name)
    expected = pd.DataFrame({'my_series': np.arange(3)})
    pd.testing.assert_frame_equal(results, expected)

    data = pd.DataFrame({'col_1': np.arange(3), 'col_2': np.arange(3)})
    results = input_to_frame(data=data, input_name=input_name)
    pd.testing.assert_frame_equal(results, data)
