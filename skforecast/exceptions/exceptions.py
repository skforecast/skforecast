################################################################################
#                            skforecast.exceptions                             #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

"""
The skforecast.exceptions module contains all the custom warnings and error 
classes used across skforecast.
"""
import warnings
import textwrap
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

class DataTypeWarning(UserWarning):
    """
    Warning used to notify there are dtypes in the exogenous data that are not
    'int', 'float', 'bool' or 'category'. Most machine learning models do not
    accept other data types, therefore the forecaster `fit` and `predict` may fail.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=DataTypeWarning)"
        )
        return self.message + "\n" + extra_message


class DataTransformationWarning(UserWarning):
    """
    Warning used to notify that the output data is in the transformed space.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=DataTransformationWarning)"
        )
        return self.message + "\n" + extra_message


class IgnoredArgumentWarning(UserWarning):
    """
    Warning used to notify that an argument is ignored when using a method 
    or a function.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=IgnoredArgumentWarning)"
        )
        return self.message + "\n" + extra_message


class IndexWarning(UserWarning):
    """
    Warning used to notify that the index of the input data is not a
    expected type. 
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=IndexWarning)"
        )
        return self.message + "\n" + extra_message


class LongTrainingWarning(UserWarning):
    """
    Warning used to notify that a large number of models will be trained and the
    the process may take a while to run.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=LongTrainingWarning)"
        )
        return self.message + "\n" + extra_message


class MissingExogWarning(UserWarning):
    """
    Warning used to indicate that there are missing exogenous variables in the
    data. Most machine learning models do not accept missing values, so the
    Forecaster's `fit' and `predict' methods may fail.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=MissingExogWarning)"
        )
        return self.message + "\n" + extra_message


class MissingValuesWarning(UserWarning):
    """
    Warning used to indicate that there are missing values in the data. This 
    warning occurs when the input data contains missing values, or the training
    matrix generates missing values. Most machine learning models do not accept
    missing values, so the Forecaster's `fit' and `predict' methods may fail.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=MissingValuesWarning)"
        )
        return self.message + "\n" + extra_message


class OneStepAheadValidationWarning(UserWarning):
    """
    Warning used to notify that the one-step-ahead validation is being used.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=OneStepAheadValidationWarning)"
        )
        return self.message + "\n" + extra_message


class ResidualsUsageWarning(UserWarning):
    """
    Warning used to notify that a residual are not correctly used in the
    probabilitic forecasting process.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=ResidualsUsageWarning)"
        )
        return self.message + "\n" + extra_message


class UnknownLevelWarning(UserWarning):
    """
    Warning used to notify that a level being predicted was not part of the
    training data.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=UnknownLevelWarning)"
        )
        return self.message + "\n" + extra_message


class SaveLoadSkforecastWarning(UserWarning):
    """
    Warning used to notify any issues that may arise when saving or loading
    a forecaster.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=SaveLoadSkforecastWarning)"
        )
        return self.message + "\n" + extra_message


class SkforecastVersionWarning(UserWarning):
    """
    Warning used to notify that the skforecast version installed in the 
    environment differs from the version used to initialize the forecaster.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=SkforecastVersionWarning)"
        )
        return self.message + "\n" + extra_message


warn_skforecast_categories = [
    DataTypeWarning,
    DataTransformationWarning,
    IgnoredArgumentWarning,
    IndexWarning,
    LongTrainingWarning,
    MissingExogWarning,
    MissingValuesWarning,
    OneStepAheadValidationWarning,
    ResidualsUsageWarning,
    UnknownLevelWarning,
    SaveLoadSkforecastWarning,
    SkforecastVersionWarning,    
]


def format_warning_handler(message, category, filename, lineno, file=None, line=None):
    """
    Custom warning handler to format warnings in a box for skforecast custom
    warnings.

    Parameters
    ----------
    message : str
        Warning message.
    category : str
        Warning category.
    filename : str
        Filename where the warning was raised.
    lineno : int
        Line number where the warning was raised.
    file : file
        File where the warning was raised.
    line : str
        Line where the warning was raised.

    Returns
    -------
    None

    """

    if isinstance(message, tuple(warn_skforecast_categories)):
        width = 88
        title = type(message).__name__
        output_text = ["\n"]

        wrapped_message = textwrap.fill(str(message), width=width - 2, expand_tabs=True, replace_whitespace=True)
        title_top_border = f"╭{'─' * ((width - len(title) - 2) // 2)} {title} {'─' * ((width - len(title) - 2) // 2)}╮"
        if len(title) % 2 != 0:
            title_top_border = title_top_border[:-1] + '─' + "╮"
        bottom_border = f"╰{'─' * width}╯"
        output_text.append(title_top_border)

        for line in wrapped_message.split('\n'):
            output_text.append(f"│ {line.ljust(width - 2)} │")

        output_text.append(bottom_border)
        output_text = "\n".join(output_text)
        color = '\033[38;5;208m'
        reset = '\033[0m'
        output_text = f"{color}{output_text}{reset}"
        print(output_text)
    else:
        # Fallback to default Python warning formatting
        warnings._original_showwarning(message, category, filename, lineno, file, line)


def rich_warning_handler(message, category, filename, lineno, file=None, line=None):
    """
    Custom handler for warnings that uses rich to display formatted panels.
    """
    if isinstance(message, tuple(warn_skforecast_categories)):
        console = Console()
        len_category_name = len(category.__name__)
        panel = Panel(
            Text(
                f"{message.message}\n\n"
                f"Category: {category.__name__}\n"
                f"Location: {filename}:{lineno}\n"
                f"Supress: {str(message)[-(79 + len_category_name) :]}",
                justify="left",
            ),
            title=category.__name__,
            title_align="center",
            border_style="color(214)",
            width=88,
        )
        console.print(panel)
    else:
        # Fallback to default Python warning formatting
        warnings._original_showwarning(message, category, filename, lineno, file, line)


if not hasattr(warnings, "_original_showwarning"):
    warnings._original_showwarning = warnings.showwarning

warnings.showwarning = rich_warning_handler