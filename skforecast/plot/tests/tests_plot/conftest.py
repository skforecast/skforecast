# conftest.py - Shared configuration for plot tests
# ==============================================================================
import pytest
import matplotlib
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _use_agg_backend():
    """
    Use the non-interactive 'Agg' backend for all plot tests and close all
    figures after each test to prevent resource leaks.
    """
    matplotlib.use("Agg")
    yield
    plt.close("all")
