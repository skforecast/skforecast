# Unit test backtesting_gif_creator
# ==============================================================================
import os
import re
import pytest
import numpy as np
import pandas as pd
import matplotlib
from skforecast.model_selection import TimeSeriesFold
from ... import backtesting_gif_creator


def test_backtesting_gif_creator_raise_error_invalid_arguments():
    """
    Test that backtesting_gif_creator raises an error when invalid 
    arguments are passed.
    """

    # Don't use interactive backends during testing
    matplotlib.use("Agg")

    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv = TimeSeriesFold(
             initial_train_size=5,
             steps=2,
         )

    wrong_data = np.arange(10)
    err_msg = re.escape(
        f"`data` must be a pandas Series or DataFrame. Got {type(wrong_data)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        backtesting_gif_creator(data=wrong_data, cv=cv)

    wrong_data = pd.DataFrame(np.arange(10))
    wrong_data.index = pd.Index(np.arange(10))
    err_msg = re.escape(
        f"`data` must have a pandas RangeIndex or DatetimeIndex. Got {type(wrong_data.index)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        backtesting_gif_creator(data=wrong_data, cv=cv)

    wrong_cv = object()
    err_msg = re.escape(
        f"`cv` must be a 'TimeSeriesFold' object. Got '{type(wrong_cv).__name__}'."
    )
    with pytest.raises(TypeError, match=err_msg):
        backtesting_gif_creator(data=data, cv=wrong_cv)

    fps = -1
    err_msg = re.escape(f"`fps` must be a positive integer. Got {fps}.")
    with pytest.raises(TypeError, match=err_msg):
        backtesting_gif_creator(data=data, cv=cv, fps=fps)

    dpi = -1
    err_msg = re.escape(f"`dpi` must be a positive integer. Got {dpi}.")
    with pytest.raises(TypeError, match=err_msg):
        backtesting_gif_creator(data=data, cv=cv, dpi=dpi)

    series_to_plot = "x"
    err_msg = re.escape(
        f"`series_to_plot` must be a list of column names. Got {type(series_to_plot)}."
    )
    with pytest.raises(TypeError, match=err_msg):
        backtesting_gif_creator(data=data, cv=cv, series_to_plot=series_to_plot)

    series_to_plot = ["x"]
    err_msg = re.escape("Columns not found in `data`: ['x']")
    with pytest.raises(ValueError, match=err_msg):
        backtesting_gif_creator(data=data, cv=cv, series_to_plot=series_to_plot)

    wrong_colors = {
        "train": "#329239",
        "test": "#0d579b",
    }
    err_msg = re.escape(
        f"`colors` must contain the following keys: "
        f"{['train', 'last_window', 'gap', 'test', 'v_lines']}"
    )
    with pytest.raises(ValueError, match=err_msg):
        backtesting_gif_creator(data=data, cv=cv, colors=wrong_colors)


@pytest.mark.parametrize(
    "cv_kwargs", 
    [{'steps': 10, 'initial_train_size': 70},
     {'steps': 10, 'initial_train_size': 70, 'refit': True},
     {'steps': 10, 'initial_train_size': 70, 'refit': True, 'fixed_train_size': False},
     {'steps': 10, 'initial_train_size': 70, 'gap': 5},
     {'steps': 8, 'initial_train_size': 70, 'gap': 5, 'allow_incomplete_fold': False, 'refit': 2},
     {'steps': 8, 'initial_train_size': 70, 'skip_folds': 2, 'allow_incomplete_fold': False},
     {'steps': 8, 'initial_train_size': 70, 'fold_stride': 5}
     ], 
    ids=lambda cv_kwargs: f'cv_kwargs: {cv_kwargs}'
)
def test_backtesting_gif_creator_output(cv_kwargs):
    """
    Check that backtesting_gif_creator returns the expected output.
    """
    data = pd.DataFrame({
        'y': np.arange(0, 100),
        'exog': np.arange(100, 200)
    })
    cv = TimeSeriesFold(**cv_kwargs, verbose=False)
    cv_window_size = TimeSeriesFold(**cv_kwargs, window_size=5, verbose=False)

    backtesting_gif_creator(data=data['y'], cv=cv, filename="backtesting.gif", dpi=50)
    backtesting_gif_creator(data=data, cv=cv_window_size, filename="backtesting.gif", dpi=50)
    os.remove("backtesting.gif")
