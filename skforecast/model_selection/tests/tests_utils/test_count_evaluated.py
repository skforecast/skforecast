# Unit test _count_evaluated
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
from skforecast.model_selection._utils import _count_evaluated


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        (np.array([1., 2., 3.]), np.array([1., 2., 3.]), 3),
        (np.array([1., np.nan, 3.]), np.array([1., 2., 3.]), 2),
        (np.array([1., 2., 3.]), np.array([np.nan, 2., 3.]), 2),
        (np.array([np.nan, 2., 3.]), np.array([np.nan, np.nan, 3.]), 1),
        (pd.Series([np.nan, np.nan]), pd.Series([1., 2.]), 0),
        (pd.Series(["a", None, "b"]), pd.Series(["a", "b", "b"]), 2),
        (np.array([]), np.array([]), 0),
    ],
    ids=["all", "missing y_true", "missing y_pred", "both", "none", "classes", "empty"]
)
def test_count_evaluated_output(y_true, y_pred, expected):
    """
    Test that only the points with a true value and a prediction, both not
    missing, are counted, also with classes (strings) and pandas Series.
    """
    n_evaluated = _count_evaluated(y_true=y_true, y_pred=y_pred)

    assert n_evaluated == expected
    assert isinstance(n_evaluated, int)
