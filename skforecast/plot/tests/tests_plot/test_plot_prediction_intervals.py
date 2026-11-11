# Unit test plot_prediction_intervals
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skforecast.plot import plot_prediction_intervals


def _make_fixtures(steps: int = 24, seed: int = 123):
    """
    Return a tuple of (predictions, y_true, target_variable) suitable for
    `plot_prediction_intervals`.

    `y_true` is built so its index ends at the same timestamp as
    `predictions`, guaranteeing overlap for the `.loc` slice inside the
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


@pytest.mark.parametrize(
    'use_y_true_column',
    [True, False],
    ids=lambda use: f'y_true column in predictions: {use}'
)
def test_plot_prediction_intervals_output_when_y_true_or_target_variable_omitted(use_y_true_column):
    """
    Test that the real values come from the column 'y_true' of `predictions`
    when `y_true` is not given, and from the only column of `y_true` when
    `target_variable` is not given.
    """
    predictions, y_true, _ = _make_fixtures()
    expected = y_true.loc[predictions.index].to_numpy()
    fig, ax = plt.subplots()

    if use_y_true_column:
        predictions = predictions.assign(y_true=y_true.loc[predictions.index])
        plot_prediction_intervals(predictions=predictions, ax=ax)
    else:
        plot_prediction_intervals(predictions=predictions, y_true=y_true, ax=ax)

    real_value_line = ax.lines[0]
    assert real_value_line.get_label() == "real value"
    np.testing.assert_array_almost_equal(real_value_line.get_ydata(), expected)


def _predictions_multiple_levels():
    """
    Predictions of two levels in long format, as returned by the multi-series
    backtesting.
    """
    predictions, _, _ = _make_fixtures()
    return pd.concat([
        predictions.assign(level="l1", y_true=0.0),
        predictions.assign(level="l2", y_true=0.0),
    ])


@pytest.mark.parametrize(
    'kwargs, err_msg',
    [
        (
            {'predictions': _make_fixtures()[0]},
            "`y_true` must be provided when `predictions` does not have the "
            "column 'y_true' (the predictions of the backtesting functions "
            "include it)."
        ),
        (
            {'predictions': _predictions_multiple_levels()},
            "`predictions` contains more than one level. Filter the rows of one "
            "level before plotting, for example: "
            "`predictions[predictions['level'] == 'level_name']`."
        ),
        (
            {
                'predictions': _make_fixtures()[0],
                'y_true': _make_fixtures()[1].to_frame().assign(other=1.0),
            },
            "`target_variable` must be provided when `y_true` has more "
            "than one column."
        ),
    ],
    ids=['no y_true', 'multiple levels', 'y_true several columns']
)
def test_plot_prediction_intervals_ValueError_when_invalid_inputs(kwargs, err_msg):
    """
    Test ValueError when the real values cannot be determined: no `y_true` and
    no column 'y_true' in `predictions`, predictions of several levels, or a
    `y_true` with several columns and no `target_variable`.
    """
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        plot_prediction_intervals(**kwargs)
