################################################################################
#                                   Plot                                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
from typing import Union, Any, Optional
import numpy as np
import pandas as pd
from ..utils import check_optional_dependency

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.graphics.tsaplots import plot_acf
except Exception as e:
    package_name = str(e).split(" ")[-1].replace("'", "")
    check_optional_dependency(package_name=package_name)


def plot_residuals(
    residuals: Union[np.ndarray, pd.Series] = None,
    y_true: Union[np.ndarray, pd.Series] = None,
    y_pred: Union[np.ndarray, pd.Series] = None,
    fig: matplotlib.figure.Figure = None,
    **fig_kw
) -> matplotlib.figure.Figure:
    """
    Parameters
    ----------
    residuals : pandas Series, numpy ndarray, default None.
        Values of residuals. If `None`, residuals are calculated internally using
        `y_true` and `y_true`.
    y_true : pandas Series, numpy ndarray, default None.
        Ground truth (correct) values. Ignored if residuals is not `None`.
    y_pred : pandas Series, numpy ndarray, default None. 
        Values of predictions. Ignored if residuals is not `None`.
    fig : matplotlib.figure.Figure, default None. 
        Pre-existing fig for the plot. Otherwise, call matplotlib.pyplot.figure()
        internally.
    fig_kw : dict
        Other keyword arguments are passed to matplotlib.pyplot.figure()

    Returns
    -------
    fig: matplotlib.figure.Figure
        Matplotlib Figure.
    
    """
    
    if residuals is None and (y_true is None or y_pred is None):
        raise ValueError(
            "If `residuals` argument is None then, `y_true` and `y_pred` must be provided."
        )
        
    if residuals is None:
        residuals = y_true - y_pred
            
    if fig is None:
        fig = plt.figure(constrained_layout=True, **fig_kw)
        
    gs  = matplotlib.gridspec.GridSpec(2, 2, figure=fig)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    
    ax1.plot(residuals)
    sns.histplot(residuals, kde=True, bins=30, ax=ax2)
    plot_acf(residuals, ax=ax3, lags=60)
    
    ax1.set_title("Residuals")
    ax2.set_title("Distribution")
    ax3.set_title("Autocorrelation")

    return fig


def plot_multivariate_time_series_corr(
    corr: pd.DataFrame,
    ax: matplotlib.axes.Axes = None,
    **fig_kw
) -> matplotlib.figure.Figure:
    """
    Heatmap plot of a correlation matrix.

    Parameters
    ----------
    corr : pandas DataFrame
        correlation matrix
    ax : matplotlib.axes.Axes, default None
        Pre-existing ax for the plot. Otherwise, call matplotlib.pyplot.subplots() 
        internally.
    fig_kw : dict
        Other keyword arguments are passed to matplotlib.pyplot.subplots()
    
    Returns
    -------
    fig: matplotlib.figure.Figure
        Matplotlib Figure.

    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kw)
    
    sns.heatmap(
        corr,
        annot=True,
        linewidths=.5,
        ax=ax,
        cmap=sns.color_palette("viridis", as_cmap=True)
    )

    ax.set_xlabel('Time series')
    
    return fig


def plot_prediction_distribution(
    bootstrapping_predictions: pd.DataFrame,
    bw_method: Optional[Any] = None,
    **fig_kw
) -> matplotlib.figure.Figure:
    """
    Ridge plot of bootstrapping predictions. This plot is very useful to understand 
    the uncertainty of forecasting predictions.

    Parameters
    ----------
    bootstrapping_predictions : pandas DataFrame
        Bootstrapping predictions created with `Forecaster.predict_bootstrapping`.
    bw_method : str, scalar, Callable, default None
        The method used to calculate the estimator bandwidth. This can be 'scott', 
        'silverman', a scalar constant or a Callable. If None (default), 'scott' 
        is used. See scipy.stats.gaussian_kde for more information.
    fig_kw : dict
        All additional keyword arguments are passed to the `pyplot.figure` call.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib Figure.
    
    """

    index = bootstrapping_predictions.index.astype(str).to_list()[::-1]
    palette = sns.cubehelix_palette(len(index), rot=-.25, light=.7, reverse=False)
    fig, axs = plt.subplots(len(index), 1, sharex=True, **fig_kw)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i, step in enumerate(index):
        plot = (
            bootstrapping_predictions.loc[step, :]
            .plot.kde(ax=axs[i], bw_method=bw_method, lw=0.5)
        )

        # Fill density area
        x = plot.get_children()[0]._x
        y = plot.get_children()[0]._y
        axs[i].fill_between(x, y, color=palette[i])
        prediction_mean = bootstrapping_predictions.loc[step, :].mean()
        
        # Closest point on x to the prediction mean
        idx = np.abs(x - prediction_mean).argmin()
        axs[i].vlines(x[idx], ymin=0, ymax=y[idx], linestyle="dashed", color='w')

        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].set_yticklabels([])
        axs[i].set_yticks([])
        axs[i].set_ylabel(step, rotation='horizontal')
        axs[i].set_xlabel('prediction')

    fig.subplots_adjust(hspace=-0)
    fig.suptitle('Forecasting distribution per step')

    return fig


def set_dark_theme(
    custom_style: Optional[dict] = None
) -> None:
    """
    Set aspects of the visual theme for all matplotlib plots.
    This function changes the global defaults for all plots using the matplotlib
    rcParams system. The theme includes specific colors for figure and axes
    backgrounds, gridlines, text, labels, and ticks. It also sets the font size
    and line width.

    Parameters
    ----------
    custom_style : dict, default None
        Optional dictionary containing custom styles to be added or override the
        default dark theme. It is applied after the default theme is set by
        using the `plt.rcParams.update()` method.

    Returns
    -------
    None
    
    """

    plt.style.use('fivethirtyeight')
    dark_style = {
        'figure.facecolor': '#212946',
        'axes.facecolor': '#212946',
        'savefig.facecolor': '#212946',
        'axes.grid': True,
        'axes.grid.which': 'both',
        'axes.spines.left': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.bottom': False,
        'grid.color': '#2A3459',
        'grid.linewidth': '1',
        'text.color': '0.9',
        'axes.labelcolor': '0.9',
        'xtick.color': '0.9',
        'ytick.color': '0.9',
        'font.size': 10,
        'lines.linewidth': 1.5
    }

    if custom_style is not None:
        dark_style.update(custom_style)
        
    plt.rcParams.update(dark_style)


def plot_prediction_intervals(
    predictions: pd.DataFrame,
    y_true: pd.DataFrame,
    target_variable: str,
    initial_x_zoom: list = None,
    title: str = None,
    xaxis_title: str = None,
    yaxis_title: str = None,
    ax: plt.Axes = None,
    kwargs_subplots: dict[str, object] = {'figsize': (7, 3)},
    kwargs_fill_between: dict[str, object] = {'color': '#444444', 'alpha': 0.3}
):
    """
    Plot predicted intervals vs real values using matplotlib.

    Parameters
    ----------
    predictions : pandas DataFrame
        Predicted values and intervals. Expected columns are 'pred', 'lower_bound'
        and 'upper_bound'.
    y_true : pandas DataFrame
        Real values of target variable.
    target_variable : str
        Name of target variable.
    initial_x_zoom : list, default None
        Initial zoom of x-axis, by default None.
    title : str, default None
        Title of the plot, by default None.
    xaxis_title : str, default None
        Title of x-axis, by default None.
    yaxis_title : str, default None
        Title of y-axis, by default None.
    ax : matplotlib axes, default None
        Axes where to plot, by default None.
    kwargs_subplots : dict, default {'figsize': (7, 3)}
        Additional keyword arguments (key, value mappings) to pass to `plt.subplots`.
    kwargs_fill_between : dict, default {'color': '#444444', 'alpha': 0.3}
        Additional keyword arguments (key, value mappings) to pass to `ax.fill_between`.

    Returns
    -------
    None
    
    """
    
    if ax is None:
        fig, ax = plt.subplots(**kwargs_subplots)

    y_true.loc[predictions.index, target_variable].plot(ax=ax, label='Real value')
    predictions['pred'].plot(ax=ax, label='prediction')
    ax.fill_between(
        predictions.index,
        predictions['lower_bound'],
        predictions['upper_bound'],
        **kwargs_fill_between
    )
    ax.set_ylabel(yaxis_title)
    ax.set_xlabel(xaxis_title)
    ax.set_title(title)
    ax.legend()

    if initial_x_zoom is not None:
        ax.set_xlim(initial_x_zoom)
