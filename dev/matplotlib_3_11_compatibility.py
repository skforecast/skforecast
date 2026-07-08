import pytest
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from skforecast.plot.plot import (
    plot_residuals,
    plot_multivariate_time_series_corr,
    plot_prediction_distribution,
    plot_prediction_intervals,
    set_dark_theme,
    backtesting_gif_creator
)
from skforecast.model_selection import TimeSeriesFold

def test_plot_residuals_compatibility():
    """
    Test that `plot_residuals` executes without errors and returns a Figure.
    """
    y_true = np.random.normal(size=100)
    y_pred = y_true + np.random.normal(scale=0.1, size=100)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore irrelevant warnings during execution
        fig = plot_residuals(y_true=y_true, y_pred=y_pred)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

def test_plot_multivariate_time_series_corr_compatibility():
    """
    Test that `plot_multivariate_time_series_corr` executes without errors and returns a Figure.
    """
    corr = pd.DataFrame(np.random.rand(4, 4), columns=list('ABCD'), index=list('ABCD'))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot_multivariate_time_series_corr(corr=corr)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

def test_plot_prediction_distribution_compatibility():
    """
    Test that `plot_prediction_distribution` executes without errors and returns a Figure.
    """
    preds = pd.DataFrame(
        np.random.rand(5, 100), 
        index=pd.date_range("2020-01-01", periods=5)
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot_prediction_distribution(bootstrapping_predictions=preds)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

def test_plot_prediction_intervals_compatibility():
    """
    Test that `plot_prediction_intervals` executes without errors and updates the given Axes.
    """
    preds = pd.DataFrame({
        'pred': np.random.rand(10),
        'lower_bound': np.random.rand(10) - 0.1,
        'upper_bound': np.random.rand(10) + 0.1
    }, index=pd.date_range("2020-01-01", periods=10))
    y_true = pd.Series(np.random.rand(10), index=preds.index, name='y')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots()
        plot_prediction_intervals(
            predictions=preds,
            y_true=y_true,
            target_variable='y',
            ax=ax
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

def test_set_dark_theme_compatibility():
    """
    Test that `set_dark_theme` executes without errors.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        set_dark_theme()
        # Reset to default style to avoid side effects in other tests
        plt.style.use('default')

def test_backtesting_gif_creator_compatibility(tmp_path):
    """
    Test that `backtesting_gif_creator` executes without errors and generates a GIF file.
    """
    data = pd.Series(np.random.rand(50), index=pd.date_range("2020-01-01", periods=50), name='y')
    cv = TimeSeriesFold(
        steps=5,
        initial_train_size=10,
        refit=False,
    )
    # Mock the window_size as it's typically set when used within a forecaster
    cv.window_size = 5 
    
    file_path = tmp_path / "test_backtesting.gif"
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_path = backtesting_gif_creator(
            data=data,
            cv=cv,
            filename=str(file_path),
            plot_last_window=True,
            fps=1,
            dpi=50
        )
        assert result_path.exists()
