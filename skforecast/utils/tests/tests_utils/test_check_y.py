# Unit test check_y
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.utils import check_y


@pytest.mark.parametrize("y", 
                         [10, [1, 2, 3], np.arange(10).reshape(-1, 1)], 
                         ids = lambda y : f'y: {y}'
                        )
def test_check_y_exception_when_y_not_pandas_series(y):
    """
    Check exception is raised when y is not a pandas Series.
    """
    err_msg = re.escape('`y` must be a pandas Series.')
    with pytest.raises(TypeError, match = err_msg):
        check_y(y=y)


def test_check_y_exception_when_y_has_missing_values():
    """
    Check exception is raised when y has missing values.
    """
    err_msg = re.escape('`y` has missing values.')
    with pytest.raises(ValueError, match = err_msg):
        check_y(pd.Series([0, 1, None]))