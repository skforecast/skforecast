# Unit test get_exog_dtypes
# ==============================================================================
import pandas as pd
from skforecast.utils import get_exog_dtypes


def test_get_exog_dtypes_when_pandas_Series():
    """
    Check get_exog_dtypes when exog is pandas Series.
    """
    exog = pd.Series([1, 2, 3], name='y')
    exog_dtypes = get_exog_dtypes(exog)

    expected = {'y': exog.dtypes}

    assert exog_dtypes['y'] == expected['y']


def test_get_exog_dtypes_when_pandas_DataFrame():
    """
    Check get_exog_dtypes when exog is pandas DataFrame.
    """
    exog = pd.DataFrame({
        'A': [1, 2, 3], 
        'B': [1.1, 2.2, 3.3],
        'C': pd.Categorical(range(3)),
        'D': [True]*3,
        'E': ['string']*3
    })
    exog_dtypes = get_exog_dtypes(exog)

    expected = {
        'A': exog['A'].dtypes,
        'B': exog['B'].dtypes,
        'C': exog['C'].dtypes,
        'D': exog['D'].dtypes,
        'E': exog['E'].dtypes
    }

    for k in exog_dtypes.keys():
        assert exog_dtypes[k] == expected[k]