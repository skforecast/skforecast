# Unit test check_exog
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_exog
from skforecast.exceptions import MissingValuesWarning


@pytest.mark.parametrize("exog", 
                         [10, [1, 2, 3], np.arange(10).reshape(-1, 1)], 
                         ids = lambda exog : f'exog: {exog}')
def test_check_exog_TypeError_when_exog_not_series_or_dataframe(exog):
    """
    Check TypeError is raised when exog is not a pandas Series or a
    pandas DataFrame.
    """
    err_msg = re.escape(
        f"`exog` must be a pandas Series or DataFrame. Got {type(exog)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_exog(exog=exog)


def test_check_exog_ValueError_when_exog_pandas_Series_with_no_name():
    """
    Check ValueError is raised when exog is a pandas Series with no name.
    """
    exog = pd.Series([1, 2, 3])
    err_msg = re.escape("When `exog` is a pandas Series, it must have a name.")
    with pytest.raises(ValueError, match = err_msg):
        check_exog(exog=exog)


@pytest.mark.parametrize("exog", 
                         [pd.Series([0, 1, None], name='exog'),
                          pd.Series([0, 1, np.nan], name='exog'),
                          pd.DataFrame([[1, 2], [3, None], [5, 6]])], 
                         ids = lambda exog : f'exog type: {type(exog)}')
def test_check_exog_MissingValuesWarning_when_exog_has_missing_values_and_allow_nan_False(exog):
    """
    Check MissingValuesWarning is issued when y has missing values.
    """
    warn_msg = re.escape(
        ("`exog` has missing values. Most machine learning models do "
         "not allow missing values. Fitting the forecaster may fail.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_exog(exog, allow_nan=False)