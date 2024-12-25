# Unit test check_predict_input
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.utils import check_predict_input
from skforecast.utils import check_exog
from skforecast.utils import preprocess_exog
from skforecast.utils import preprocess_last_window
from skforecast.exceptions import MissingValuesWarning
from skforecast.exceptions import MissingExogWarning
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.exceptions import UnknownLevelWarning

if pd.__version__ < '2.2.0':
    freq = "M"
else:
    freq = "ME"


def test_check_predict_input_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    err_msg = re.escape(
        ("This Forecaster instance is not fitted yet. Call `fit` with "
         "appropriate arguments before using predict.")
    )
    with pytest.raises(NotFittedError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 5,
            is_fitted        = False,
            exog_in_         = False,
            index_type_      = None,
            index_freq_      = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )
        
        
def test_check_predict_input_ValueError_when_steps_int_lower_than_1():
    """
    Test ValueError is raised when steps is a value lower than 1.
    """
    steps = -5

    err_msg = re.escape(f"`steps` must be an integer greater than or equal to 1. Got {steps}.")
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = steps,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = None,
            index_freq_      = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )
        
        
def test_check_predict_input_ValueError_when_steps_list_lower_than_1():
    """
    Test ValueError is raised when steps is a list with a value lower than 1. 
    (`ForecasterDirect` and `ForecasterDirectMultiVariate`).
    """
    steps = [0, 1, 2]

    err_msg = re.escape(
        (f"The minimum value of `steps` must be equal to or greater than 1. "
         f"Got {min(steps)}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterDirect',
            steps            = steps,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = None,
            index_freq_      = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_last_step_greater_than_max_steps():
    """
    Test ValueError is raised when max(steps) > max_steps. (`ForecasterDirect` 
    and `ForecasterDirectMultiVariate`).
    """
    steps = list(np.arange(20) + 1)
    max_steps = 10

    err_msg = re.escape(
        (f"The maximum value of `steps` must be less than or equal to "
         f"the value of steps defined when initializing the forecaster. "
         f"Got {max(steps)}, but the maximum is {max_steps}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterDirectMultiVariate',
            steps            = steps,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = None,
            index_freq_      = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = max_steps,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_ForecasterRecursiveMultiSeries_and_level_not_str_list_or_None():
    """
    Test TypeError is raised when `levels` is not a str, a list or None.
    """
    levels = 5

    err_msg = re.escape("`levels` must be a `list` of column names, a `str` of a column name or `None`.")   
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = None,
            index_freq_      = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = levels,
            series_names_in_ = ['1', '2']
        )


@pytest.mark.parametrize("levels     , series_names_in_", 
                         [('1'       , ['2', '3']), 
                          (['1']     , ['2', '3']), 
                          (['1', '2'], ['2', '3'])])
def test_check_predict_input_UnknownLevelWarning_when_ForecasterRecursiveMultiSeries_and_level_not_in_series_names_in__onehot(levels, series_names_in_):
    """
    Test UnknownLevelWarning is raised when `levels` is not in `self.series_names_in_` in a 
    ForecasterRecursiveMultiSeries.
    """
    last_window = pd.DataFrame(
        {'1': np.arange(10), '2': np.arange(10), '3': np.arange(10)}, 
        index = pd.date_range(start='01/01/2018', periods=10, freq=freq)
    )

    warn_msg = re.escape(
        ("`levels` {'1'} were not included in training. The resulting "
         "one-hot encoded columns for this feature will be all zeros.")
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        check_predict_input(
            forecaster_name   = 'ForecasterRecursiveMultiSeries',
            steps             = 5,
            is_fitted         = True,
            exog_in_          = False,
            index_type_       = pd.DatetimeIndex,
            index_freq_       = freq,
            window_size       = 5,
            last_window       = last_window,
            last_window_exog  = None,
            exog              = None,
            exog_type_in_     = None,
            exog_names_in_    = None,
            interval          = [2.5, 97.5],
            alpha             = None,
            max_steps         = None,
            levels            = levels,
            series_names_in_  = series_names_in_,
            levels_forecaster = None,
            encoding          = 'onehot'
        )


@pytest.mark.parametrize("encoding", ['ordinal', 'ordinal_category'])
@pytest.mark.parametrize("levels     , series_names_in_", 
                         [('1'       , ['2', '3']), 
                          (['1']     , ['2', '3']), 
                          (['1', '2'], ['2', '3'])])
def test_check_predict_input_UnknownLevelWarning_when_ForecasterRecursiveMultiSeries_and_level_not_in_series_names_in_(encoding, levels, series_names_in_):
    """
    Test UnknownLevelWarning is raised when `levels` is not in `self.series_names_in_` in a 
    ForecasterRecursiveMultiSeries.
    """
    last_window = pd.DataFrame(
        {'1': np.arange(10), '2': np.arange(10), '3': np.arange(10)}, 
        index = pd.date_range(start='01/01/2018', periods=10, freq=freq)
    )

    warn_msg = re.escape(
        ("`levels` {'1'} were not included in training. "
         "Unknown levels are encoded as NaN, which may cause the "
         "prediction to fail if the regressor does not accept NaN values.")
    )
    with pytest.warns(UnknownLevelWarning, match = warn_msg):
        check_predict_input(
            forecaster_name   = 'ForecasterRecursiveMultiSeries',
            steps             = 5,
            is_fitted         = True,
            exog_in_          = False,
            index_type_       = pd.DatetimeIndex,
            index_freq_       = freq,
            window_size       = 5,
            last_window       = last_window,
            last_window_exog  = None,
            exog              = None,
            exog_type_in_     = None,
            exog_names_in_    = None,
            interval          = None,
            alpha             = None,
            max_steps         = None,
            levels            = levels,
            series_names_in_  = series_names_in_,
            levels_forecaster = None,
            encoding          = encoding
        )


@pytest.mark.parametrize("levels     , levels_forecaster", 
                         [('1'       , '2'), 
                          (['1']     , ['2', '3']), 
                          (['1', '2'], ['2', '3'])])
def test_check_predict_input_ValueError_when_ForecasterRnn_and_level_not_in_levels_forecaster(levels, levels_forecaster):
    """
    Test ValueError is raised when `levels` is not in `self.levels` 
    (levels_forecaster) in a ForecasterRnn.
    """
    err_msg = re.escape(
        (f"`levels` names must be included in the series used during fit "
         f"({levels_forecaster}). Got {levels}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name   = 'ForecasterRnn',
            steps             = 5,
            is_fitted         = True,
            exog_in_          = False,
            index_type_       = None,
            index_freq_       = None,
            window_size       = None,
            last_window       = None,
            last_window_exog  = None,
            exog              = None,
            exog_type_in_     = None,
            exog_names_in_    = None,
            interval          = None,
            alpha             = None,
            max_steps         = None,
            levels            = levels,
            series_names_in_  = None,
            levels_forecaster = levels_forecaster,
        )


def test_check_predict_input_ValueError_when_exog_is_none_and_exog_in_is_true():
    """
    """
    err_msg = re.escape(
        ("Forecaster trained with exogenous variable/s. "
         "Same variable/s must be provided when predicting.")
    )   
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = None,
            index_freq_      = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_exog_is_not_none_and_exog_in_is_false():
    """
    """
    err_msg = re.escape(
        ("Forecaster trained without exogenous variable/s. "
         "`exog` must be `None` when predicting.")
    )   
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = None,
            index_freq_      = None,
            window_size      = None,
            last_window      = None,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_last_window_not_stored_during_training_single_series():
    """
    """
    last_window = None
    err_msg = re.escape(
        ("`last_window` was not stored during training. If you don't want "
         "to retrain the Forecaster, provide `last_window` as argument.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type_in_    = pd.Series,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


@pytest.mark.parametrize("forecaster_name", 
                         ['ForecasterRecursiveMultiSeries',
                          'ForecasterDirectMultiVariate'], 
                         ids=lambda ft: f'forecaster_name: {ft}')
def test_check_predict_input_TypeError_when_last_window_is_not_pandas_DataFrame(forecaster_name):
    """
    `ForecasterRecursiveMultiSeries` and `ForecasterDirectMultiVariate`.
    """
    last_window = np.arange(5)

    err_msg = re.escape(f"`last_window` must be a pandas DataFrame. Got {type(last_window)}.")
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = forecaster_name,
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type_in_    = pd.Series,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = '1',
            series_names_in_ = ['1', '2']
        )


@pytest.mark.parametrize("levels     , last_window", 
                         [('1'       , pd.DataFrame({'3': [1, 2, 3], '4': [1, 2, 3]})), 
                          (['1']     , pd.DataFrame({'3': [1, 2, 3], '4': [1, 2, 3]})), 
                          (['1', '2'], pd.DataFrame({'3': [1, 2, 3], '4': [1, 2, 3]}))], 
                         ids = lambda values: f'levels: {values}')
def test_check_predict_input_ValueError_when_levels_not_in_last_window_ForecasterRecursiveMultiSeries(levels, last_window):
    """
    Check ValueError is raised when levels are no the same as last_window column names.
    """
    last_window_cols = last_window.columns.to_list()
    missing_levels = set(levels) - set(last_window_cols)
    err_msg = re.escape(
        f"`last_window` must contain a column(s) named as the level(s) to be predicted. "
        f"The following `levels` are missing in `last_window`: {missing_levels}\n"
        f"Ensure that `last_window` contains all the necessary columns "
        f"corresponding to the `levels` being predicted.\n"
        f"    Argument `levels`     : {levels}\n"
        f"    `last_window` columns : {last_window_cols}\n"
        f"Example: If `levels = ['series_1', 'series_2']`, make sure "
        f"`last_window` includes columns named 'series_1' and 'series_2'."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = 2,
            last_window      = last_window,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type_in_    = pd.Series,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = levels,
            series_names_in_ = ['1', '2']
        )


def test_check_predict_input_ValueError_when_series_names_in__not_last_window_ForecasterDirectMultiVariate():
    """
    Check ValueError is raised when column names of series using during fit do not
    match with last_window column names.
    """
    last_window = pd.DataFrame({'l1': [1, 2, 3], '4': [1, 2, 3]})
    series_names_in_ = ['l1', 'l2']

    err_msg = re.escape(
        ("`last_window` columns must be the same as the `series` "
         "column names used to create the X_train matrix.\n"
         "    `last_window` columns    : ['l1', '4']\n"
         "    `series` columns X train : ['l1', 'l2']")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterDirectMultiVariate',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = 2,
            last_window      = last_window,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type_in_    = pd.Series,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = series_names_in_
        )


def test_check_predict_input_TypeError_when_last_window_is_not_pandas_series():
    """
    """
    last_window = np.arange(5)
    err_msg = re.escape(
        f"`last_window` must be a pandas Series or DataFrame. Got {type(last_window)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type_in_    = pd.Series,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_length_last_window_is_lower_than_window_size():
    """
    """
    window_size = 10

    err_msg = re.escape(
        (f"`last_window` must have as many values as needed to "
         f"generate the predictors. For this forecaster it is {window_size}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = window_size,
            last_window      = pd.Series(np.arange(5)),
            last_window_exog = None,
            exog             = pd.Series(np.arange(10)),
            exog_type_in_    = pd.Series,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_MissingValuesWarning_when_last_window_has_missing_values():
    """
    """
    warn_msg = re.escape(
        ("`last_window` has missing values. Most of machine learning models do "
         "not allow missing values. Prediction method may fail.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = 5,
            last_window      = pd.Series([1, 2, 3, 4, 5, np.nan]),
            last_window_exog = None,
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_last_window_index_is_not_of_index_type():
    """
    """
    last_window = pd.Series(np.arange(10))
    index_type_ = pd.DatetimeIndex
    _, last_window_index = preprocess_last_window(
                                last_window = last_window.iloc[:0]
                           ) 

    err_msg = re.escape(
        (f"Expected index of type {index_type_} for `last_window`. "
         f"Got {type(last_window_index)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = index_type_,
            index_freq_      = None,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_last_window_index_frequency_is_not_index_freq():
    """
    """
    last_window = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='D'))
    index_freq_ = 'Y'
    _, last_window_index = preprocess_last_window(
                                last_window = last_window.iloc[:0]
                           )

    err_msg = re.escape(
        (f"Expected frequency of type {index_freq_} for `last_window`. "
         f"Got {last_window_index.freqstr}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = index_freq_,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_exog_is_not_pandas_series_or_dataframe_multiseries():
    """
    """
    err_msg = re.escape(
        f"`exog` must be a pandas Series, DataFrame or dict. Got {type(np.arange(10))}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.DataFrame(np.arange(10), columns=['l1'], index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = np.arange(10),
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_TypeError_when_exog_is_not_of_exog_type():
    """
    """
    last_window = pd.Series(
        np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq),
        name = '1'
    ).to_frame()
    exog = pd.Series(np.arange(10))
    exog_type_in_ = dict

    err_msg = re.escape(f"Expected type for `exog`: {exog_type_in_}. Got {type(exog)}.")
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = exog,
            exog_type_in_    = exog_type_in_,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = ['1'],
            series_names_in_ = ['1', '2']
        )


def test_check_predict_input_TypeError_when_exog_is_not_pandas_series_or_dataframe():
    """
    """
    err_msg = re.escape(
        f"`exog` must be a pandas Series or DataFrame. Got {type(np.arange(10))}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.DataFrame(np.arange(10), columns=['l1'], index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = np.arange(10),
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_TypeError_when_exog_dict_and_not_pandas_series_or_DataFrame():
    """
    """
    err_msg = re.escape(
        f"`exog` for series 'l1' must be a pandas Series or DataFrame. Got {type(np.arange(10))}"
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.DataFrame(np.arange(10), columns=['l1'], index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = {'l1': np.arange(10)},
            exog_type_in_    = dict,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_MissingExogWarning_when_exog_dict_and_no_key_for_some_levels():
    """
    """
    warn_msg = re.escape(
        ("`exog` does not contain keys for levels {'l2'}. "
         "Missing levels are filled with NaN. Most of machine learning "
         "models do not allow missing values. Prediction method may fail.")
    )
    with pytest.warns(MissingExogWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.DataFrame({'l1': np.arange(10), 'l2': np.arange(10)}, index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = {'l1': pd.DataFrame(np.arange(10), columns=['exog_1'], index=pd.date_range(start='11/1/2018', periods=10, freq=freq))},
            exog_type_in_    = dict,
            exog_names_in_   = ['exog_1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = ['l1', 'l2'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_MissingValuesWarning_when_exog_has_missing_values():
    """
    """
    warn_msg = re.escape(
        ("`exog` has missing values. Most of machine learning models do "
         "not allow missing values. Prediction method may fail.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 3,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 2,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = pd.Series([1, 2, 3, np.nan], index=pd.date_range(start='11/1/2018', periods=4, freq=freq), name='exog1'),
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


@pytest.mark.parametrize("steps", [10, [1, 2, 3, 4, 5, 6], [2, 6]], 
                         ids=lambda steps: f'steps: {steps}')
def test_check_predict_input_MissingValuesWarning_when_len_exog_is_less_than_steps_MultiSeries(steps):
    """
    """
    last_step = max(steps) if isinstance(steps, list) else steps
    warn_msg = re.escape(
        (f"`exog` doesn't have as many values as steps "
         f"predicted, {last_step}. Missing values are filled "
         f"with NaN. Most of machine learning models do not "
         f"allow missing values. Prediction method may fail."),
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = steps,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), name='l1', index=pd.date_range(start='1/1/2018', periods=10, freq=freq)).to_frame(),
            last_window_exog = None,
            exog             = pd.Series(np.arange(5), name='exog1', index=pd.date_range(start='11/1/2018', periods=5, freq=freq)),
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )
            

@pytest.mark.parametrize("steps", [10, [1, 2, 3, 4, 5, 6], [2, 6]], 
                         ids=lambda steps: f'steps: {steps}')
def test_check_predict_input_ValueError_when_len_exog_is_less_than_steps(steps):
    """
    """
    last_step = max(steps) if isinstance(steps, list) else steps
    err_msg = re.escape(
        (f"`exog` must have at least as many values as "
         f"steps predicted, {last_step}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = steps,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = pd.Series(np.arange(5)),
            exog_type_in_    = pd.Series,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_MissingExogWarning_when_exog_is_DataFrame_without_columns_in_exog_names_in__MultiSeries():
    """
    Raise MissingExogWarning when there are missing columns in `exog` when 
    Forecaster multi series.
    """
    exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col2'])
    exog.index = pd.date_range(start='1/1/2018', periods=5, freq=freq)
    exog_names_in_ = ['col1', 'col3']

    warn_msg = re.escape(
        ("{'col3'} not present in `exog`. All "
         "values will be NaN.")
    )
    with pytest.warns(MissingExogWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.DataFrame(np.arange(10), columns=['l1'], index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = exog,
            exog_type_in_    = pd.DataFrame,
            exog_names_in_   = exog_names_in_,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_ValueError_when_exog_is_DataFrame_without_columns_in_exog_names_in_():
    """
    Raise ValueError when there are missing columns in `exog`.
    """
    exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col2'])
    exog_names_in_ = ['col1', 'col3']

    err_msg = re.escape(
        (f"Missing columns in `exog`. Expected {exog_names_in_}. "
         f"Got {exog.columns.to_list()}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = exog,
            exog_type_in_    = pd.DataFrame,
            exog_names_in_   = exog_names_in_,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_exog_is_Series_with_no_name():
    """
    Raise ValueError when `exog` is a pandas Series with no name.
    """
    exog = pd.Series(np.arange(10))
    exog_names_in_ = ['exog1']

    err_msg = re.escape(
         ("When `exog` is a pandas Series, it must have a name. Got None.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = exog,
            exog_type_in_    = pd.Series,
            exog_names_in_   = exog_names_in_,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_IgnoredArgumentWarning_when_exog_is_Series_with_name_not_in_exog_names_in__multiseries():
    """
    Raise IgnoredArgumentWarning when `exog` is a pandas Series and its name is not in 
    `exog_names_in_` when Forecaster multi series.
    """
    exog = pd.Series(np.arange(10), name='exog2')
    exog.index = pd.date_range(start='11/01/2018', periods=10, freq=freq)
    exog_names_in_ = ['exog1']

    last_window = pd.Series(
        np.arange(10), 
        index = pd.date_range(start='01/01/2018', periods=10, freq=freq), 
        name  = 'l1'
    ).to_frame()

    warn_msg = re.escape(
        ("'exog2' was not observed during training. `exog` is ignored. "
         "Exogenous variables must be one of: ['exog1'].")
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = exog,
            exog_type_in_    = pd.Series,
            exog_names_in_   = exog_names_in_,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_ValueError_when_exog_is_Series_with_name_not_in_exog_names_in__multiseries():
    """
    Raise ValueError when `exog` is a pandas Series and its name is not in 
    `exog_names_in_` when Forecaster multi series.
    """
    exog = pd.Series(np.arange(10), name='exog2')
    exog_names_in_ = ['exog1']

    err_msg = re.escape(
        ("'exog2' was not observed during training. "
         "Exogenous variables must be: ['exog1'].")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = exog,
            exog_type_in_    = pd.Series,
            exog_names_in_   = exog_names_in_,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_exog_index_is_not_of_index_type():
    """
    """
    exog = pd.Series(np.arange(10), name='exog1')
    index_type_ = pd.DatetimeIndex
    check_exog(exog = exog)
    _, exog_index = preprocess_exog(exog=exog.iloc[:0, ])

    err_msg = re.escape(
        (f"Expected index of type {index_type_} for `exog`. "
         f"Got {type(exog_index)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = index_type_,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = exog,
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_exog_index_frequency_is_not_index_freq():
    """
    """
    exog = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10), name='exog1')
    index_freq_ = freq
    check_exog(exog = exog)
    _, exog_index = preprocess_exog(exog=exog.iloc[:0, ])

    err_msg = re.escape(
        (f"Expected frequency of type {index_freq_} for `exog`. "
         f"Got {exog_index.freqstr}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = None,
            exog             = exog,
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_exog_index_does_not_follow_last_window_index_DatetimeIndex():
    """
    Raise exception if `exog` index does not start at the end of `last_window` index when DatetimeIndex.
    """
    exog_datetime = pd.Series(data=np.random.rand(10), name='exog1')
    exog_datetime.index = pd.date_range(start='2022-03-01', periods=10, freq=freq)
    lw_datetime = pd.Series(data=np.random.rand(10))
    lw_datetime.index = pd.date_range(start='2022-01-01', periods=10, freq=freq)

    expected_index = '2022-11-30 00:00:00'

    err_msg = re.escape(
        (f"To make predictions `exog` must start one step "
         f"ahead of `last_window`.\n"
         f"    `last_window` ends at : {lw_datetime.index[-1]}.\n"
         f"    `exog` starts at : {exog_datetime.index[0]}.\n"
         f"     Expected index : {expected_index}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = lw_datetime,
            last_window_exog = None,
            exog             = exog_datetime,
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_exog_index_does_not_follow_last_window_index_RangeIndex():
    """
    Raise ValueError if `exog` index does not start at the end of `last_window` index when RangeIndex.
    """
    exog_datetime = pd.Series(data=np.random.rand(10), name='exog1')
    exog_datetime.index = pd.RangeIndex(start=11, stop=21)
    lw_datetime = pd.Series(data=np.random.rand(10))
    lw_datetime.index = pd.RangeIndex(start=0, stop=10)

    expected_index = 10

    err_msg = re.escape(
        (f"To make predictions `exog` must start one step "
         f"ahead of `last_window`.\n"
         f"    `last_window` ends at : {lw_datetime.index[-1]}.\n"
         f"    `exog` starts at : {exog_datetime.index[0]}.\n"
         f"     Expected index : {expected_index}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = lw_datetime,
            last_window_exog = None,
            exog             = exog_datetime,
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_ForecasterSarimax_last_window_exog_is_not_None_and_exog_in_is_false():
    """
    Check ValueError is raised when last_window_exog is not None, but exog_in_     
    is False.
    """
    err_msg = re.escape(
        ("Forecaster trained without exogenous variable/s. "
         "`last_window_exog` must be `None` when predicting.")
    )   
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 3,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = 3,
            last_window      = pd.Series(np.arange(5)),
            last_window_exog = pd.Series(np.arange(5)),
            exog             = None,
            exog_type_in_    = None,
            exog_names_in_   = None,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_ForecasterSarimax_last_window_exog_is_not_pandas_Series_or_DataFrame():
    """
    """
    last_window_exog = np.arange(5)
    err_msg = re.escape(
        (f"`last_window_exog` must be a pandas Series or a "
         f"pandas DataFrame. Got {type(last_window_exog)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = 3,
            last_window      = pd.Series(np.arange(5)),
            last_window_exog = last_window_exog,
            exog             = pd.Series(np.arange(10), index=pd.RangeIndex(start=5, stop=15), name='exog1'),
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_ForecasterSarimax_length_last_window_exog_is_lower_than_window_size():
    """
    """
    window_size = 10

    err_msg = re.escape(
        (f"`last_window_exog` must have as many values as needed to "
         f"generate the predictors. For this forecaster it is {window_size}.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = window_size,
            last_window      = pd.Series(np.arange(10)),
            last_window_exog = pd.Series(np.arange(5)),
            exog             = pd.Series(np.arange(10), index=pd.RangeIndex(start=10, stop=20), name='exog1'),
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_MissingValuesWarning_when_ForecasterSarimax_last_window_exog_has_missing_values():
    """
    """
    warn_msg = re.escape(
        ("`last_window_exog` has missing values. Most of machine learning models "
         "do not allow missing values. Prediction method may fail.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = None,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10)),
            last_window_exog = pd.Series([1, 2, 3, 4, 5, np.nan], name='exog1'),
            exog             = pd.Series(np.arange(10), index=pd.RangeIndex(start=10, stop=20), name='exog1'),
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_ForecasterSarimax_last_window_exog_index_is_not_of_index_type():
    """
    """
    last_window_exog = pd.Series(np.arange(10), name='exog1')
    index_type_ = pd.DatetimeIndex
    _, last_window_exog_index = preprocess_last_window(
        last_window = last_window_exog.iloc[:0]
    ) 

    err_msg = re.escape(
        (f"Expected index of type {index_type_} for `last_window_exog`. "
         f"Got {type(last_window_exog_index)}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = index_type_,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = pd.Series(np.arange(10), index=pd.date_range(start='30/11/2018', periods=10, freq=freq), name='exog1'),
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_ForecasterSarimax_last_window_exog_index_frequency_is_not_index_freq():
    """
    """
    
    last_window_exog = pd.Series(np.arange(10), index=pd.date_range(start='2018', periods=10, freq='YE'))
    index_freq_ = freq
    _, last_window_exog_index = preprocess_last_window(
                                last_window = last_window_exog.iloc[:0]
                           )

    err_msg = re.escape(
        (f"Expected frequency of type {index_freq_} for `last_window_exog`. "
         f"Got {last_window_exog_index.freqstr}.")
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = index_freq_,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = pd.Series(np.arange(10), index=pd.date_range(start='30/11/2018', periods=10, freq=freq), name='exog1'),
            exog_type_in_    = pd.Series,
            exog_names_in_   = ['exog1'],
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_last_window_exog_is_DataFrame_without_columns_in_exog_names_in_():
    """
    Raise ValueError when there are missing columns in `last_window_exog`, ForecasterSarimax.
    """
    
    exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col3'])
    exog.index = pd.date_range(start='6/1/2018', periods=5, freq=freq)
    exog_names_in_ = ['col1', 'col3']

    last_window_exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col2'])
    last_window_exog.index = pd.date_range(start='1/1/2018', periods=5, freq=freq)

    err_msg = re.escape(
        (f"Missing columns in `last_window_exog`. Expected {exog_names_in_}. "
         f"Got {last_window_exog.columns.to_list()}.") 
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(5), index=pd.date_range(start='1/1/2018', periods=5, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = exog,
            exog_type_in_    = pd.DataFrame,
            exog_names_in_   = exog_names_in_,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


# @pytest.mark.skipif(pd.__version__ > '2.2.0', reason="requires pandas < 2.2.0")
def test_check_predict_input_ValueError_when_last_window_exog_is_Series_with_no_name():
    """
    Raise ValueError when `last_window_exog` has no name, ForecasterSarimax.
    """
    
    exog = pd.Series(np.arange(5), name='exog1')
    exog.index = pd.date_range(start='6/1/2018', periods=5, freq=freq)
    exog_names_in_ = ['exog1']

    last_window_exog = pd.Series(np.arange(5))
    last_window_exog.index = pd.date_range(start='1/1/2018', periods=5, freq=freq)
    
    err_msg = re.escape(
        ("When `last_window_exog` is a pandas Series, it must have a name. Got None.")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(5), index=pd.date_range(start='1/1/2018', periods=5, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = exog,
            exog_type_in_    = pd.Series,
            exog_names_in_   = exog_names_in_,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_last_window_exog_is_Series_without_name_in_exog_names_in_():
    """
    Raise ValueError when `last_window_exog` name is not in `exog_names_in_`, ForecasterSarimax.
    """
    exog = pd.Series(np.arange(5), name='exog1')
    exog.index = pd.date_range(start='6/1/2018', periods=5, freq=freq)
    exog_names_in_ = ['exog1']

    last_window_exog = pd.Series(np.arange(5), name='exog2')
    last_window_exog.index = pd.date_range(start='1/1/2018', periods=5, freq=freq)
    
    err_msg = re.escape(
        ("'exog2' was not observed during training. "
         "Exogenous variables must be: ['exog1'].")
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterSarimax',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(5), index=pd.date_range(start='1/1/2018', periods=5, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = exog,
            exog_type_in_    = pd.Series,
            exog_names_in_   = exog_names_in_,
            interval         = None,
            alpha            = None,
            max_steps        = None,
            levels           = None,
            series_names_in_ = None
        )