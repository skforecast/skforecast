# Unit test plot_multivariate_time_series_corr
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skforecast.plot import plot_multivariate_time_series_corr


def _make_corr(n: int = 4, seed: int = 123) -> pd.DataFrame:
    """Return a square correlation DataFrame of size n x n."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((100, n))
    columns = [f"series_{i}" for i in range(n)]
    return pd.DataFrame(data, columns=columns).corr()


def test_plot_multivariate_time_series_corr_output():
    """
    Test that plot_multivariate_time_series_corr returns a Figure with heatmap
    (QuadMesh), n*n text annotations and the expected x-axis label when no
    external ax is provided.
    """
    n = 4
    corr = _make_corr(n=n)

    fig = plot_multivariate_time_series_corr(corr=corr)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2  # 1 main axes + 1 colorbar axes
    ax = fig.axes[0]
    assert len(ax.collections) >= 1  # QuadMesh from heatmap
    texts = [
        child for child in ax.get_children()
        if isinstance(child, matplotlib.text.Text) and child.get_text()
    ]
    assert len(texts) >= n * n
    assert ax.get_xlabel() == "Time series"


def test_plot_multivariate_time_series_corr_output_with_custom_ax():
    """
    Test that plot_multivariate_time_series_corr draws on the provided ax and
    returns the Figure that owns it.
    """
    corr = _make_corr(n=4)
    fig_ext, ax_ext = plt.subplots(1, 1)

    fig = plot_multivariate_time_series_corr(corr=corr, ax=ax_ext)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig is fig_ext
    assert len(ax_ext.collections) >= 1  # heatmap drawn on provided ax


def test_plot_multivariate_time_series_corr_output_with_fig_kw():
    """
    Test that fig_kw keyword arguments (e.g. figsize) are forwarded to
    plt.subplots and reflected in the resulting Figure.
    """
    corr = _make_corr(n=3)

    fig = plot_multivariate_time_series_corr(corr=corr, figsize=(12, 10))

    width, height = fig.get_size_inches()
    assert width == pytest.approx(12)
    assert height == pytest.approx(10)
