# Unit test plot_multivariate_time_series_corr
# ==============================================================================
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
    Test that plot_multivariate_time_series_corr returns a Figure with a single
    Axes object and a heatmap (QuadMesh) when no external ax is provided.
    """
    matplotlib.use("Agg")
    corr = _make_corr(n=4)

    fig = plot_multivariate_time_series_corr(corr=corr)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2  # 1 main axes + 1 colorbar axes
    ax = fig.axes[0]
    assert len(ax.collections) >= 1  # QuadMesh from heatmap
    plt.close(fig)


def test_plot_multivariate_time_series_corr_output_with_custom_ax():
    """
    Test that plot_multivariate_time_series_corr draws on the provided ax and
    returns the Figure that owns it.
    """
    matplotlib.use("Agg")
    corr = _make_corr(n=4)
    fig_ext, ax_ext = plt.subplots(1, 1)

    fig = plot_multivariate_time_series_corr(corr=corr, ax=ax_ext)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig is fig_ext
    assert len(ax_ext.collections) >= 1  # heatmap drawn on provided ax
    plt.close(fig)


def test_plot_multivariate_time_series_corr_output_annotations():
    """
    Test that the heatmap contains text annotations equal to the number of
    cells in the correlation matrix (n x n).
    """
    matplotlib.use("Agg")
    n = 4
    corr = _make_corr(n=n)

    fig = plot_multivariate_time_series_corr(corr=corr)

    ax = fig.axes[0]
    texts = [child for child in ax.get_children() if isinstance(child, matplotlib.text.Text)
             and child.get_text()]
    # n * n annotation texts for each cell
    assert len(texts) >= n * n
    plt.close(fig)


def test_plot_multivariate_time_series_corr_output_x_label():
    """
    Test that the x-axis label is set to 'Time series'.
    """
    matplotlib.use("Agg")
    corr = _make_corr(n=3)

    fig = plot_multivariate_time_series_corr(corr=corr)

    assert fig.axes[0].get_xlabel() == "Time series"
    plt.close(fig)
