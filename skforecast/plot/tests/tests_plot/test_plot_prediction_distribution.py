# Unit test plot_prediction_distribution
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.collections import PolyCollection, LineCollection
from skforecast.plot import plot_prediction_distribution


def _make_bootstrapping_predictions(steps: int = 12, n_boot: int = 200, seed: int = 123) -> pd.DataFrame:
    """
    Return a DataFrame shaped (steps, n_boot) mirroring the output of
    ``forecaster.predict_bootstrapping()``.
    The index is a DatetimeIndex of monthly periods.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=steps, freq="ME")
    return pd.DataFrame(rng.standard_normal((steps, n_boot)), index=index)


def test_plot_prediction_distribution_output():
    """
    Test that plot_prediction_distribution returns a Figure with one Axes per
    forecast step, each containing a filled KDE area (PolyCollection), a
    vertical mean line (LineCollection) and the KDE curve (Line2D).
    """
    steps = 6
    bootstrapping_predictions = _make_bootstrapping_predictions(steps=steps)

    fig = plot_prediction_distribution(bootstrapping_predictions=bootstrapping_predictions)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == steps
    for ax in fig.axes:
        poly_collections = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(poly_collections) >= 1, "Each axis should have a filled KDE area"
        line_collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(line_collections) >= 1, "Each axis should have a dashed mean line"
        assert len(ax.lines) >= 1, "Each axis should contain the KDE line"


def test_plot_prediction_distribution_output_with_bw_method():
    """
    Test that plot_prediction_distribution runs without error when bw_method
    is specified and returns a Figure with the correct number of axes.
    """
    steps = 4
    bootstrapping_predictions = _make_bootstrapping_predictions(steps=steps)

    fig = plot_prediction_distribution(bootstrapping_predictions=bootstrapping_predictions, bw_method=0.5)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == steps


def test_plot_prediction_distribution_output_single_step():
    """
    Test that plot_prediction_distribution handles a single forecast step.
    """
    bootstrapping_predictions = _make_bootstrapping_predictions(steps=1)

    fig = plot_prediction_distribution(bootstrapping_predictions=bootstrapping_predictions)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
