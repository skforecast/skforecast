# Unit test TimeSeriesDifferentiator fit
# ==============================================================================
import re
import pytest
import numpy as np
from skforecast.preprocessing import TimeSeriesDifferentiator


def test_TimeSeriesDifferentiator_fit():
    """
    Test that TimeSeriesDifferentiator fit method returns self.
    """
    X = np.arange(10)
    tsd = TimeSeriesDifferentiator(order=1)

    assert tsd.fit(X) == tsd


def test_TimeSeriesDifferentiator_fit_initial_values():
    """
    Test that TimeSeriesDifferentiator fit method sets initial_values attribute.
    """
    X = np.arange(10)
    tsd = TimeSeriesDifferentiator(order=1)
    tsd.fit(X)

    assert tsd.initial_values == [0]
    assert tsd.last_values == [9]

