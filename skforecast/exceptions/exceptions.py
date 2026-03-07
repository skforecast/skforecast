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
import inspect
from functools import wraps
import textwrap
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def runtime_deprecated(
    replacement: str = None, 
    version: str = None, 
    removal: str = None, 
    category=FutureWarning
) -> object:
    """
    Decorator to mark functions or classes as deprecated.
    Works for both function and class targets, and ensures warnings are visible
    even inside Jupyter notebooks.
    """
    def decorator(obj):
        is_function = inspect.isfunction(obj) or inspect.ismethod(obj)
        is_class = inspect.isclass(obj)

        if not (is_function or is_class):
            raise TypeError("@runtime_deprecated can only be used on functions or classes")

        # ----- Build warning message -----
        name = obj.__name__
        message = f"{name}() is deprecated" if is_function else f"{name} class is deprecated"
        if version:
            message += f" since version {version}"
        if replacement:
            message += f"; use {replacement} instead"
        if removal:
            message += f". It will be removed in version {removal}."
        else:
            message += "."

        def issue_warning():
            """Emit warning in a way that always shows in notebooks."""
            with warnings.catch_warnings():
                warnings.simplefilter("always", category)
                warnings.warn(message, category, stacklevel=3)

        # ----- Case 1: decorating a function -----
        if is_function:
            @wraps(obj)
            def wrapper(*args, **kwargs):
                issue_warning()
                return obj(*args, **kwargs)

            # Add metadata
            wrapper.__deprecated__ = True
            wrapper.__replacement__ = replacement
            wrapper.__version__ = version
            wrapper.__removal__ = removal
            return wrapper

        # ----- Case 2: decorating a class -----
        elif is_class:
            orig_init = getattr(obj, "__init__", None)
            orig_new = getattr(obj, "__new__", None)

            # Only wrap whichever exists (some classes use __new__, others __init__)
            if orig_new and (orig_new is not object.__new__):
                @wraps(orig_new)
                def wrapped_new(cls, *args, **kwargs):
                    issue_warning()
                    return orig_new(cls, *args, **kwargs)
                obj.__new__ = staticmethod(wrapped_new)

            elif orig_init:
                @wraps(orig_init)
                def wrapped_init(self, *args, **kwargs):
                    issue_warning()
                    return orig_init(self, *args, **kwargs)
                obj.__init__ = wrapped_init

            # Add metadata
            obj.__deprecated__ = True
            obj.__replacement__ = replacement
            obj.__version__ = version
            obj.__removal__ = removal

            return obj

    return decorator


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


class ExogenousInterpretationWarning(UserWarning):
    """
    Warning used to notify about important implications when using exogenous 
    variables with models that use a two-step approach (e.g., regression + ARAR).
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=ExogenousInterpretationWarning)"
        )
        return self.message + "\n" + extra_message


class FeatureOutOfRangeWarning(UserWarning):
    """
    Warning used to notify that a feature is out of the range seen during training.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=FeatureOutOfRangeWarning)"
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


class InputTypeWarning(UserWarning):
    """
    Warning used to notify that input format is not the most efficient or
    recommended for the forecaster.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=InputTypeWarning)"
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
    probabilistic forecasting process.
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
    ExogenousInterpretationWarning,
    FeatureOutOfRangeWarning,
    IgnoredArgumentWarning,
    InputTypeWarning,
    LongTrainingWarning,
    MissingExogWarning,
    MissingValuesWarning,
    OneStepAheadValidationWarning,
    ResidualsUsageWarning,
    UnknownLevelWarning,
    SaveLoadSkforecastWarning,
    SkforecastVersionWarning
]


def format_warning_handler(
    message: str, 
    category: str, 
    filename: str, 
    lineno: str, 
    file: object = None, 
    line: str = None
) -> None:
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
    file : file, default None
        File where the warning was raised.
    line : str, default None
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


def rich_warning_handler(
    message: str, 
    category: str, 
    filename: str, 
    lineno: str, 
    file: object = None, 
    line: str = None
) -> None:
    """
    Custom handler for warnings that uses rich to display formatted panels.

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
    file : file, default None
        File where the warning was raised.
    line : str, default None
        Line where the warning was raised.

    Returns
    -------
    None

    """
    
    if isinstance(message, tuple(warn_skforecast_categories)):
        console = Console()

        category_name = category.__name__
        text = (
            f"{message.message}\n\n"
            f"Category : skforecast.exceptions.{category_name}\n"
            f"Location : {filename}:{lineno}\n"
            f"Suppress : warnings.simplefilter('ignore', category={category_name})"
        )

        panel = Panel(
            Text(text, justify="left"),
            title        = category_name,
            title_align  = "center",
            border_style = "color(214)",
            width        = 88,
        )
        
        console.print(panel)
    else:
        # Fallback to default Python warning formatting
        warnings._original_showwarning(message, category, filename, lineno, file, line)


def set_warnings_style(style: str = 'skforecast') -> None:
    """
    Set the warning handler based on the provided style.

    Parameters
    ----------
    style : str, default='skforecast'
        The style of the warning handler. Either 'skforecast' or 'default'.
    
    Returns
    -------
    None

    """
    if style == "skforecast":
        if not hasattr(warnings, "_original_showwarning"):
            warnings._original_showwarning = warnings.showwarning
        warnings.showwarning = rich_warning_handler
    else:
        warnings.showwarning = warnings._original_showwarning


set_warnings_style(style='skforecast')
