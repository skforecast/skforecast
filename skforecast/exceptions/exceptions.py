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
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=DataTypeWarning)"
        )
        return self.message + " " + extra_message


class DataTransformationWarning(UserWarning):
    """
    Warning used to notify that the output data is in the transformed space.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=DataTransformationWarning)"
        )
        return self.message + " " + extra_message


class IgnoredArgumentWarning(UserWarning):
    """
    Warning used to notify that an argument is ignored when using a method 
    or a function.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=IgnoredArgumentWarning)"
        )
        return self.message + " " + extra_message


class IndexWarning(UserWarning):
    """
    Warning used to notify that the index of the input data is not a
    expected type. 
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=IndexWarning)"
        )
        return self.message + " " + extra_message


class LongTrainingWarning(UserWarning):
    """
    Warning used to notify that a large number of models will be trained and the
    the process may take a while to run.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=LongTrainingWarning)"
        )
        return self.message + " " + extra_message


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
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=MissingExogWarning)"
        )
        return self.message + " " + extra_message


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
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=MissingValuesWarning)"
        )
        return self.message + " " + extra_message


class OneStepAheadValidationWarning(UserWarning):
    """
    Warning used to notify that the one-step-ahead validation is being used.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=OneStepAheadValidationWarning)"
        )
        return self.message + " " + extra_message


class ResidualsUsageWarning(UserWarning):
    """
    Warning used to notify that a residual are not correctly used in the
    probabilitic forecasting process.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=ResidualsUsageWarning)"
        )
        return self.message + " " + extra_message


class UnknownLevelWarning(UserWarning):
    """
    Warning used to notify that a level being predicted was not part of the
    training data.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=UnknownLevelWarning)"
        )
        return self.message + " " + extra_message


class SaveLoadSkforecastWarning(UserWarning):
    """
    Warning used to notify any issues that may arise when saving or loading
    a forecaster.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=SaveLoadSkforecastWarning)"
        )
        return self.message + " " + extra_message


class SkforecastVersionWarning(UserWarning):
    """
    Warning used to notify that the skforecast version installed in the 
    environment differs from the version used to initialize the forecaster.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self):
        extra_message = (
            "\n You can suppress this warning using: "
            "warnings.simplefilter('ignore', category=SkforecastVersionWarning)"
        )
        return self.message + " " + extra_message


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