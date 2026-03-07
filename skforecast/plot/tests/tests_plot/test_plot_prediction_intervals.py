# Unit test plot_prediction_intervals
# ==============================================================================
import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skforecast.plot import plot_prediction_intervals


def _make_fixtures(steps: int = 24, seed: int = 123):
    """
    Return a tuple of (predictions, y_true, target_variable) suitable for
    ``plot_prediction_intervals``.

    ``y_true`` is built so its index ends at the same timestamp as
    ``predictions``, guaranteeing overlap for the ``.loc`` slice inside the
    function.
    """
    rng = np.random.default_rng(seed)
    # predictions: steps hourly timestamps starting 2022-01-01
    pred_index = pd.date_range("2022-01-01", periods=steps, freq="h")
    pred_values = rng.standard_normal(steps)
    predictions = pd.DataFrame(
        {
            "pred": pred_values,
            "lower_bound": pred_values - 1.0,
            "upper_bound": pred_values + 1.0,
        },
        index=pred_index,
    )
    # y_true: 24 hours of history before predictions + the prediction window
    history = 24
    full_index = pd.date_range(
        end=pred_index[-1], periods=steps + history, freq="h"
    )
    y_true = pd.Series(rng.standard_normal(len(full_index)), index=full_index, name="value")
    target_variable = "value"
    return predictions, y_true, target_variable


def test_plot_prediction_intervals_output_with_ax():
    """
    Test that plot_prediction_intervals returns None, draws real values and
    predictions (lines), the fill-between region (collection), sets titles
    and creates a legend with the expected labels on the provided Axes.
    """
    predictions, y_true, target_variable = _make_fixtures()
    fig, ax = plt.subplots()

    result = plot_prediction_intervals(
        predictions=predictions,
        y_true=y_true,
        target_variable=target_variable,
        title="Test Title",
        xaxis_title="X Label",
        yaxis_title="Y Label",
        ax=ax,
    )

    assert result is None
    assert len(ax.lines) >= 2           # real value + prediction lines
    assert len(ax.collections) >= 1     # fill_between region
    assert ax.get_title() == "Test Title"
    assert ax.get_xlabel() == "X Label"
    assert ax.get_ylabel() == "Y Label"
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "real value" in legend_texts
    assert "prediction" in legend_texts
    assert "prediction interval" in legend_texts


def test_plot_prediction_intervals_output_without_ax():
    """
    Test that plot_prediction_intervals creates its own figure when no ax is
    provided and still draws lines and collections on the current axes.
    """
    predictions, y_true, target_variable = _make_fixtures()

    result = plot_prediction_intervals(
        predictions=predictions,
        y_true=y_true,
        target_variable=target_variable,
    )

    assert result is None
    fig = plt.gcf()
    ax = fig.axes[0]
    assert len(ax.lines) >= 2
    assert len(ax.collections) >= 1


def test_plot_prediction_intervals_output_initial_x_zoom():
    """
    Test that plot_prediction_intervals applies initial_x_zoom and the x-axis
    limits reflect the requested zoom range.
    """
    predictions, y_true, target_variable = _make_fixtures()
    fig, ax = plt.subplots()
    zoom_end = predictions.index[len(predictions) // 2]
    zoom = [predictions.index[0], zoom_end]

    plot_prediction_intervals(
        predictions=predictions,
        y_true=y_true,
        target_variable=target_variable,
        initial_x_zoom=zoom,
        ax=ax,
    )

    xlim = ax.get_xlim()
    # After applying zoom the right limit should be less than the full range
    fig_no_zoom, ax_no_zoom = plt.subplots()
    plot_prediction_intervals(
        predictions=predictions,
        y_true=y_true,
        target_variable=target_variable,
        ax=ax_no_zoom,
    )
    xlim_full = ax_no_zoom.get_xlim()
    assert xlim[1] < xlim_full[1]


def test_plot_prediction_intervals_output_y_true_as_dataframe():
    """
    Test that plot_prediction_intervals works when y_true is a DataFrame.
    """
    predictions, y_true_series, target_variable = _make_fixtures()
    y_true_df = y_true_series.to_frame()
    fig, ax = plt.subplots()

    plot_prediction_intervals(
        predictions=predictions,
        y_true=y_true_df,
        target_variable=target_variable,
        ax=ax,
    )

    assert len(ax.lines) >= 2
    assert len(ax.collections) >= 1
