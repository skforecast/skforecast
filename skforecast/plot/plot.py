################################################################################
#                                   Plot                                       #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

from __future__ import annotations
import os
from typing import Any
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from ..utils import check_optional_dependency, input_to_frame
from ..exceptions import IgnoredArgumentWarning

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import seaborn as sns
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.tsa.stattools import acf, pacf
except Exception as e:
    package_name = str(e).split(" ")[-1].replace("'", "")
    check_optional_dependency(package_name=package_name)


def plot_residuals(
    residuals: np.ndarray | pd.Series | None = None,
    y_true: np.ndarray | pd.Series | None = None,
    y_pred: np.ndarray | pd.Series | None = None,
    fig: matplotlib.figure.Figure | None = None,
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
    ax: matplotlib.axes.Axes | None = None,
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
    bw_method: Any | None = None,
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
    custom_style: dict | None = None
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
        'figure.facecolor': '#001633',
        'axes.facecolor': '#001633',
        'savefig.facecolor': '#001633',
        'axes.grid': True,
        'axes.grid.which': 'both',
        'axes.spines.left': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.bottom': False,
        'grid.color': '#212946',
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
    y_true: pd.Series | pd.DataFrame,
    target_variable: str,
    initial_x_zoom: list[str] | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    ax: plt.Axes | None = None,
    kwargs_subplots: dict[str, object] = {'figsize': (7, 3)},
    kwargs_fill_between: dict[str, object] = {'color': '#444444', 'alpha': 0.3, 'zorder': 1}
):
    """
    Plot predicted intervals vs real values using matplotlib.

    Parameters
    ----------
    predictions : pandas DataFrame
        Predicted values and intervals. Expected columns are 'pred', 'lower_bound'
        and 'upper_bound'.
    y_true : pandas Series, pandas DataFrame
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

    if isinstance(y_true, pd.Series):
        y_true = y_true.to_frame()

    y_true.loc[predictions.index, target_variable].plot(ax=ax, label='real value')
    predictions['pred'].plot(ax=ax, label='prediction')
    ax.fill_between(
        predictions.index,
        predictions['lower_bound'],
        predictions['upper_bound'],
        label='prediction interval',
        **kwargs_fill_between
    )
    ax.set_ylabel(yaxis_title)
    ax.set_xlabel(xaxis_title)
    ax.set_title(title)
    ax.legend()

    if initial_x_zoom is not None:
        ax.set_xlim(initial_x_zoom)


def calculate_lag_autocorrelation(
    data: pd.Series | pd.DataFrame,
    n_lags: int = 50,
    last_n_samples: int | None = None,
    sort_by: str = "partial_autocorrelation_abs",
    acf_kwargs: dict[str, object] = {},
    pacf_kwargs: dict[str, object] = {},
) -> pd.DataFrame:
    """
    Calculate autocorrelation and partial autocorrelation for a time series.
    This is a wrapper around statsmodels.acf [1]_ and statsmodels.pacf [2]_.

    Parameters
    ----------
    data : pandas Series, pandas DataFrame
        Time series to calculate autocorrelation. If a DataFrame is provided,
        it must have exactly one column.
    n_lags : int
        Number of lags to calculate autocorrelation.
    last_n_samples : int or None, default None
        Number of most recent samples to use. If None, use the entire series. 
        Note that partial correlations can only be computed for lags up to 
        50% of the sample size. For example, if the series has 10 samples, 
        `n_lags` must be less than or equal to 5. This parameter is useful
        to speed up calculations when the series is very long.
    sort_by : str, default 'partial_autocorrelation_abs'
        Sort results by 'lag', 'partial_autocorrelation_abs', 
        'partial_autocorrelation', 'autocorrelation_abs' or 'autocorrelation'.
    acf_kwargs : dict, default {}
        Optional arguments to pass to statsmodels.tsa.stattools.acf.
    pacf_kwargs : dict, default {}
        Optional arguments to pass to statsmodels.tsa.stattools.pacf.

    Returns
    -------
    results : pandas DataFrame
        Autocorrelation and partial autocorrelation values.

    References
    ----------
    .. [1] Statsmodels acf API Reference.
           https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
    
    .. [2] Statsmodels pacf API Reference.
           https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html

    Examples
    --------
    ```python
    import pandas as pd
    from skforecast.plot import calculate_lag_autocorrelation

    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    calculate_lag_autocorrelation(data = data, n_lags = 4)
    
    #    lag  partial_autocorrelation_abs  partial_autocorrelation  autocorrelation_abs  autocorrelation
    # 0    1                     0.777778                 0.777778             0.700000         0.700000
    # 1    4                     0.360707                -0.360707             0.078788        -0.078788
    # 2    3                     0.274510                -0.274510             0.148485         0.148485
    # 3    2                     0.227273                -0.227273             0.412121         0.412121
    ```

    """

    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"`data` must be a pandas Series or a DataFrame with a single column. "
            f"Got {type(data)}."
        )
    if isinstance(data, pd.DataFrame) and data.shape[1] != 1:
        raise ValueError(
            f"If `data` is a DataFrame, it must have exactly one column. "
            f"Got {data.shape[1]} columns."
        )
    if not isinstance(n_lags, int) or n_lags <= 0:
        raise TypeError(f"`n_lags` must be a positive integer. Got {n_lags}.")
    
    if last_n_samples is not None:
        if not isinstance(last_n_samples, int) or last_n_samples <= 0:
            raise TypeError(f"`last_n_samples` must be a positive integer. Got {last_n_samples}.")
        data = data.iloc[-last_n_samples:]

    if sort_by not in [
        "lag", "partial_autocorrelation_abs", "partial_autocorrelation",
        "autocorrelation_abs", "autocorrelation",
    ]:
        raise ValueError(
            "`sort_by` must be 'lag', 'partial_autocorrelation_abs', 'partial_autocorrelation', "
            "'autocorrelation_abs' or 'autocorrelation'."
        )

    pacf_values = pacf(data, nlags=n_lags, **pacf_kwargs)
    acf_values = acf(data, nlags=n_lags, **acf_kwargs)

    results = pd.DataFrame(
        {
            "lag": range(n_lags + 1),
            "partial_autocorrelation_abs": np.abs(pacf_values),
            "partial_autocorrelation": pacf_values,
            "autocorrelation_abs": np.abs(acf_values),
            "autocorrelation": acf_values,
        }
    ).iloc[1:]

    if sort_by == "lag":
        results = results.sort_values(by=sort_by, ascending=True).reset_index(drop=True)
    else:
        results = results.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    return results


def backtesting_gif_creator(
    data: pd.Series | pd.DataFrame,
    cv: object,
    series_to_plot: list[str] | None = None,
    plot_last_window: bool = False,
    filename: str = "backtesting.gif",
    plt_style: str = "ggplot",
    figsize: tuple[int, int] = (9, 4),
    colors: dict[str, str] | None = None,
    title_template: str = "Backtesting — Fold {fold_num} (refit: {refit})",
    fps: int = 2,
    dpi: int = 100
) -> str:
    """
    Create a GIF of the backtesting folds using Matplotlib FuncAnimation.

    Parameters
    ----------
    data : pandas Series, pandas DataFrame
        Time series data to be used for backtesting. Can be a single series or 
        multiple series.
    cv : TimeSeriesFold
        TimeSeriesFold object with the information needed to split the data into folds.
    series_to_plot : list, default None
        List of series names to plot. If None, all series will be plotted.
    plot_last_window : bool, default False
        Whether to plot the last window of the time series. If True, window_size 
        must be specified in the cv object. 
    filename : str, default "backtesting.gif"
        Name of the output GIF file.
    figsize : tuple, default (9, 4)
        Size of the figure.
    plt_style : str, default "ggplot"
        Style to use for the plots. See Matplotlib styles for available options 
        here: https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html.
    colors : dict, default None
        Dictionary with colors for the different elements of the plot. Must 
        contain the following keys: ['train', 'last_window', 'gap', 'test', 'v_lines'].
        Defaults colors are {"train": "#329239", "last_window": "#f7931a",
        "gap": "#4d4d4d", "test": "#0d579b", "v_lines": "#252525"}
    title_template : str, default "Backtesting — Fold {fold_num} (refit: {refit})"
        Template for the title of each plot.
    fps : int, default 2
        Frames per second for the GIF.
    dpi : int, default 100
        Dots per inch for the GIF.

    Returns
    -------
    filename_path : str
        Path to the output GIF file.

    """

    if isinstance(data, pd.Series):
        data = input_to_frame(
                   data       = data, 
                   input_name = data.name if data.name is not None else 'y'
               )

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"`data` must be a pandas Series or DataFrame. Got {type(data)}."
        )

    if not isinstance(data.index, (pd.RangeIndex, pd.DatetimeIndex)):
        raise TypeError(
            f"`data` must have a pandas RangeIndex or DatetimeIndex. Got {type(data.index)}."
        )

    cv_name = type(cv).__name__
    if cv_name != "TimeSeriesFold":
        raise TypeError(
            f"`cv` must be a 'TimeSeriesFold' object. Got '{cv_name}'."
        )

    if not isinstance(fps, int) or fps <= 0:
        raise TypeError(f"`fps` must be a positive integer. Got {fps}.")
    if not isinstance(dpi, int) or dpi <= 0:
        raise TypeError(f"`dpi` must be a positive integer. Got {dpi}.")

    if series_to_plot is None:
        series_to_plot = data.columns.to_list()
    else:
        if not isinstance(series_to_plot, list):
            raise TypeError(
                f"`series_to_plot` must be a list of column names. Got {type(series_to_plot)}."
            )
        missing_cols = [col for col in series_to_plot if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in `data`: {missing_cols}")

    folds = cv.split(X=data)
    if cv.return_all_indexes:
        # NOTE: +1 to prevent iloc pandas from deleting the last observation
        folds = [
            [
                fold[0],
                [fold[1][0], fold[1][-1] + 1],
                (
                    [fold[2][0], fold[2][-1] + 1]
                    if cv.window_size is not None
                    else []
                ),
                [fold[3][0], fold[3][-1] + 1],
                [fold[4][0], fold[4][-1] + 1],
                fold[5],
            ]
            for fold in folds
        ]

    # --------- Paleta y límites ----------
    if colors is None:
        colors = {
            "train": "#329239",
            "last_window": "#f7931a",
            "gap": "#4d4d4d",
            "test": "#0d579b",
            "v_lines": "#252525",
        }
    else:
        if not set(colors.keys()) >= {"train", "last_window", "gap", "test", "v_lines"}:
            raise ValueError(
                f"`colors` must contain the following keys: "
                f"{['train', 'last_window', 'gap', 'test', 'v_lines']}"
            )

    # Límites Y comunes para evitar 'saltos' entre frames
    y_values = data[series_to_plot].to_numpy().flatten()
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)

    # Margen visual
    y_pad = 0.05 * (y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0)
    y_min_plot = (y_min - y_pad) if np.isfinite(y_min) else -1
    y_max_plot = (y_max + y_pad) if np.isfinite(y_max) else 1

    # --------- Figura base ----------
    with plt.style.context(plt_style):
        fig, ax = plt.subplots(figsize=figsize)

        # Para ejes X (Datetimes vs Rangos)
        x_index = data.index

        def _draw_fold(ax, fold, title: str):
            ax.clear()

            # Series completas en fondo
            for col in series_to_plot:
                ax.plot(
                    x_index,
                    data[col].to_numpy(),
                    linewidth=1.5,
                    alpha=0.9,
                    label=col,
                )

            y_min_relative = (y_min - y_min_plot) / (y_max_plot - y_min_plot)
            y_max_relative = (y_max - y_min_plot) / (y_max_plot - y_min_plot)

            # Rellenos: usamos axvspan (más robusto que fill_between para rangos)
            # Train
            train_start, train_end = fold[1]
            ax.axvspan(
                x_index[train_start],
                x_index[train_end],
                y_min_relative,
                y_max_relative,
                facecolor=colors["train"],
                alpha=0.2 if plot_last_window else 0.4,
                zorder=0,
                label="Train",
            )

            # last_window
            if plot_last_window:
                if cv.window_size is None:
                    warnings.warn(
                        "Last window cannot be calculated because the `window_size` "
                        "of the `cv` object is None.",
                        IgnoredArgumentWarning
                    )
                else:
                    last_window_start, last_window_end = fold[2]
                    ax.axvspan(
                        x_index[last_window_start],
                        x_index[last_window_end],
                        y_min_relative,
                        y_max_relative,
                        facecolor=colors["last_window"],
                        alpha=0.4,
                        zorder=0,
                        label="Last window",
                    )

            # Gap (si existe)
            gap_start = fold[3][0]
            gap_end = fold[4][0]
            if gap_start != gap_end:
                ax.axvspan(
                    x_index[gap_start],
                    x_index[gap_end],
                    y_min_relative,
                    y_max_relative,
                    facecolor="#bababa",
                    alpha=0.9,
                    zorder=0,
                    label="Gap",
                    hatch="////",
                    edgecolor=colors["gap"], 
                )

            # Test
            test_start, test_end = fold[4]
            ax.axvspan(
                x_index[test_start],
                x_index[test_end - 1],
                y_min_relative,
                y_max_relative,
                facecolor=colors["test"],
                alpha=0.4,
                zorder=0,
                label="Test",
            )

            def vline(ax, i):
                ax.axvline(
                    x_index[i],
                    ymin=y_min_relative,
                    ymax=y_max_relative,
                    lw=1,
                    ls="--",
                    alpha=0.5,
                    color=colors["v_lines"],
                )

            # marcas
            vline(ax, train_end)
            vline(ax, test_start)

            # Límites, leyenda y título
            ax.set_ylim(y_min_plot, y_max_plot)
            ax.set_title(title)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.8)

            # Formato X: dejamos que Matplotlib elija; si quieres años, se puede añadir DateFormatter.
            # Si el índice es RangeIndex, etiquetado por defecto funciona bien.

        # --------- Funciones de animación ----------
        def init():
            # Dibujo del primer fold como estado inicial
            _draw_fold(ax, folds[0], title=title_template.format(fold_num=1, refit=0))
            return (ax,)

        def update(i):
            fold = folds[i]
            fold_num = min(i + 1, len(folds) - n_extra)
            refit = fold[5] if len(fold) > 4 else i
            title = title_template.format(fold_num=fold_num, refit=refit)
            _draw_fold(ax, fold, title=title)
            return (ax,)

        # --------- Construcción y guardado ----------
        # Add extra folds for pause at the end
        n_extra = 2
        for i in range(n_extra):
            folds.append(folds[-1])

        ani = FuncAnimation(
            fig,
            update,
            frames=len(folds),
            init_func=init,
            blit=False,       # blit=False es más estable con spans/leyendas
            repeat=False
        )

        # Guardar GIF
        writer = PillowWriter(fps=max(1, int(fps)))
        ani.save(Path(filename).with_suffix('.gif'), writer=writer, dpi=dpi)
        plt.close(fig)

    filename_path = os.path.join(os.getcwd(), filename)

    return Path(filename_path)
