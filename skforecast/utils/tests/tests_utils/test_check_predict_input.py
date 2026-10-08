# Unit test check_predict_input
# ==============================================================================
import re
import pytest
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from skforecast.utils import check_predict_input
from skforecast.utils import check_exog
from skforecast.utils import check_extract_values_and_index
from skforecast.utils import expand_index
from skforecast.exceptions import MissingValuesWarning
from skforecast.exceptions import MissingExogWarning
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.exceptions import UnknownLevelWarning

freq = "ME"


def test_check_predict_input_NotFittedError_when_fitted_is_False():
    """
    Test NotFittedError is raised when fitted is False.
    """
    err_msg = re.escape(
        "This Forecaster instance is not fitted yet. Call `fit` with "
        "appropriate arguments before using predict."
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )
        
        
def test_check_predict_input_ValueError_when_steps_int_lower_than_1():
    """
    Test ValueError is raised when steps is a value lower than 1.
    """
    steps = -5

    err_msg = re.escape(
        f"`steps` must be an integer greater than or equal to 1. Got {steps}."
    )
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
            exog_names_in_   = None,
            max_step         = None,
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
        f"The minimum value of `steps` must be equal to or greater than 1. "
        f"Got {min(steps)}."
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_last_step_greater_than_max_step():
    """
    Test ValueError is raised when max(steps) > max_step. (`ForecasterDirect` 
    and `ForecasterDirectMultiVariate`).
    """
    steps = list(np.arange(20) + 1)
    max_step = 10

    err_msg = re.escape(
        f"The maximum value of `steps` must be less than or equal to "
        f"the value of steps defined when initializing the forecaster. "
        f"Got {max(steps)}, but the maximum is {max_step}."
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
            exog_names_in_   = None,
            max_step         = max_step,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_ForecasterRecursiveMultiSeries_and_level_not_str_list_or_None():
    """
    Test TypeError is raised when `levels` is not a str, a list or None.
    """
    levels = 5

    err_msg = re.escape(
        "`levels` must be a `list` of column names, a `str` of a column name or `None`."
    )
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
            exog_names_in_   = None,
            max_step         = None,
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
        "`levels` {'1'} were not included in training. The resulting "
        "one-hot encoded columns for this feature will be all zeros."
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
            exog_names_in_    = None,
            max_step          = None,
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
        "`levels` {'1'} were not included in training. "
        "Unknown levels are encoded as NaN, which may cause the "
        "prediction to fail if the estimator does not accept NaN values."
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
            exog_names_in_    = None,
            max_step          = None,
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
        f"`levels` names must be included in the series used during fit "
        f"({levels_forecaster}). Got {levels}."
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
            exog_names_in_    = None,
            max_step          = None,
            levels            = levels,
            series_names_in_  = None,
            levels_forecaster = levels_forecaster,
        )


def test_check_predict_input_ValueError_when_exog_is_none_and_exog_in_is_true():
    """
    """
    err_msg = re.escape(
        "Forecaster trained with exogenous variable/s. "
        "Same variable/s must be provided when predicting."
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_exog_is_not_none_and_exog_in_is_false():
    """
    """
    err_msg = re.escape(
        "Forecaster trained without exogenous variable/s. "
        "`exog` must be `None` when predicting."
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_last_window_not_stored_during_training_single_series():
    """
    """
    last_window = None
    err_msg = re.escape(
        "`last_window` was not stored during training. If you don't want "
        "to retrain the Forecaster, provide `last_window` as argument."
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
            exog_names_in_   = None,
            max_step         = None,
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

    err_msg = re.escape(
        f"`last_window` must be a pandas DataFrame. Got {type(last_window)}."
    )
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
            exog_names_in_   = None,
            max_step         = None,
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
            exog_names_in_   = None,
            max_step         = None,
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
        "`last_window` columns must be the same as the `series` "
        "column names used to create the X_train matrix.\n"
        "    `last_window` columns    : ['l1', '4']\n"
        "    `series` columns X train : ['l1', 'l2']"
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = series_names_in_
        )


def test_check_predict_input_ValueError_when_series_names_in__not_last_window_ForecasterRnn():
    """
    Check ValueError is raised when `last_window` does not contain all the 
    series used as input during fit in ForecasterRnn, also the ones that are 
    not predicted (not in `levels`). Before, the missing series were silently
    replaced by another column of `last_window`.
    """
    last_window = pd.DataFrame(
        {'l1': [1, 2, 3]}, index=pd.date_range(start='1/1/2018', periods=3, freq=freq)
    )

    err_msg = re.escape(
        "`last_window` columns must be the same as the `series` "
        "column names used to create the X_train matrix.\n"
        "    `last_window` columns    : ['l1']\n"
        "    `series` columns X train : ['l1', 'l2']"
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name   = 'ForecasterRnn',
            steps             = 2,
            is_fitted         = True,
            exog_in_          = False,
            index_type_       = pd.DatetimeIndex,
            index_freq_       = freq,
            window_size       = 2,
            last_window       = last_window,
            levels            = ['l1'],
            levels_forecaster = ['l1'],
            series_names_in_  = ['l1', 'l2']
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


@pytest.mark.parametrize(
    'forecaster_name',
    ['ForecasterRecursive', 'ForecasterDirect', 'ForecasterRecursiveClassifier',
     'ForecasterStats', 'ForecasterEquivalentDate'],
    ids=lambda name: f'forecaster: {name}'
)
def test_check_predict_input_ValueError_when_last_window_has_several_columns(
    forecaster_name
):
    """
    Test ValueError is raised in single series forecasters when `last_window`
    is a DataFrame with more than one column, since its values would be
    interleaved when converted to a 1D array.
    """
    last_window = pd.DataFrame(
        data  = {'y': np.arange(10, dtype=float), 'other': np.arange(10, dtype=float)},
        index = pd.RangeIndex(start=0, stop=10)
    )

    err_msg = re.escape(
        "`last_window` must be a pandas Series or a DataFrame with a single "
        "column. Got 2 columns."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name = forecaster_name,
            steps           = 5,
            is_fitted       = True,
            exog_in_        = False,
            index_type_     = pd.RangeIndex,
            index_freq_     = 1,
            window_size     = 5,
            last_window     = last_window
        )


def test_check_predict_input_ValueError_when_length_last_window_is_lower_than_window_size():
    """
    """
    window_size = 10

    err_msg = re.escape(
        f"`last_window` must have as many values as needed to "
        f"generate the predictors. For this forecaster it is {window_size}."
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_MissingValuesWarning_when_last_window_has_missing_values():
    """
    """
    warn_msg = re.escape(
        "`last_window` has missing values. Most of machine learning models do "
        "not allow missing values. Prediction method may either raise an "
        "error or return NaN predictions."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = pd.RangeIndex,
            index_freq_      = 1,
            window_size      = 5,
            last_window      = pd.Series([1, 2, 3, 4, 5, np.nan]),
            last_window_exog = None,
            exog             = None,
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


@pytest.mark.parametrize("forecaster_name, last_window, levels, series_names_in_", 
    [('ForecasterRecursive', 
      pd.Series([np.nan, 2, 3, 4, 5, 6]), None, None),
     ('ForecasterRecursiveMultiSeries', 
      pd.DataFrame({'l1': [1, 2, 3, 4, 5, 6], 'l2': [1, 2, 3, 4, 5, np.nan]}), ['l1'], ['l1', 'l2']),
     ('ForecasterDirectMultiVariate', 
      pd.DataFrame({'l1': [1, 2, 3, 4, 5, 6], 'l2': [1, 2, 3, 4, 5, np.nan]}), None, ['l1']),
     ('ForecasterRnn', 
      pd.DataFrame({'l1': [1, 2, 3, 4, 5, 6], 'l2': [np.nan, 2, 3, 4, 5, 6]}), ['l1'], ['l1', 'l2']),
     ('ForecasterRecursive', 
      pd.Series([pd.NA, 2, 3, 4, 5, 6], dtype='Float64'), None, None),
     ('ForecasterRecursiveMultiSeries', 
      pd.DataFrame({'l1': [pd.NA, 2, 3, 4, 5, 6], 'l2': [1, 2, 3, 4, 5, pd.NA]}, dtype='Float64'), 
      ['l1'], ['l1', 'l2']),
     ('ForecasterDirectMultiVariate', 
      pd.DataFrame({'l1': [None, 2, 3, 4, 5, 6], 'l2': [1, 2, 3, 4, 5, None]}, dtype='double[pyarrow]'), 
      None, ['l1'])], 
    ids = ['ForecasterRecursive', 'ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 
           'ForecasterRnn', 'ForecasterRecursive-Float64', 'ForecasterRecursiveMultiSeries-Float64', 
           'ForecasterDirectMultiVariate-pyarrow'])
def test_check_predict_input_no_MissingValuesWarning_when_missing_values_not_used_to_predict(
    forecaster_name, last_window, levels, series_names_in_
):
    """
    Test no MissingValuesWarning is issued when the missing values of 
    `last_window` are outside the last `window_size` rows or in series that 
    are not used to predict (levels not predicted in ForecasterRecursiveMultiSeries,
    series without lags in ForecasterDirectMultiVariate).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=MissingValuesWarning)
        check_predict_input(
            forecaster_name   = forecaster_name,
            steps             = 2,
            is_fitted         = True,
            exog_in_          = False,
            index_type_       = pd.RangeIndex,
            index_freq_       = 1,
            window_size       = 5,
            last_window       = last_window,
            levels            = levels,
            levels_forecaster = levels,
            series_names_in_  = series_names_in_
        )


@pytest.mark.parametrize("forecaster_name, last_window, levels, series_names_in_", 
    [('ForecasterStats', 
      pd.Series([np.nan, 2, 3, 4, 5, 6]), None, None),
     ('ForecasterRecursiveMultiSeries', 
      pd.DataFrame({'l1': [1, 2, 3, 4, 5, 6], 'l2': [1, 2, 3, 4, 5, np.nan]}), ['l2'], ['l1', 'l2']),
     ('ForecasterDirectMultiVariate', 
      pd.DataFrame({'l1': [1, 2, 3, 4, 5, 6], 'l2': [1, 2, 3, 4, 5, np.nan]}), None, ['l1', 'l2']),
     ('ForecasterRnn', 
      pd.DataFrame({'l1': [1, 2, 3, 4, 5, 6], 'l2': [1, 2, 3, 4, 5, np.nan]}), ['l1'], ['l1', 'l2']),
     ('ForecasterStats', 
      pd.Series([pd.NA, 2, 3, 4, 5, 6], dtype='Float64'), None, None),
     ('ForecasterRecursiveMultiSeries', 
      pd.DataFrame({'l1': [1, 2, 3, 4, 5, 6], 'l2': [1, pd.NA, 3, 4, 5, 6]}, dtype='Float64'), 
      ['l2'], ['l1', 'l2']),
     ('ForecasterDirectMultiVariate', 
      pd.DataFrame({'l1': [1, 2, 3, 4, 5, 6], 'l2': [1, None, 3, 4, 5, 6]}, dtype='double[pyarrow]'), 
      None, ['l1', 'l2'])], 
    ids = ['ForecasterStats', 'ForecasterRecursiveMultiSeries', 'ForecasterDirectMultiVariate', 
           'ForecasterRnn', 'ForecasterStats-Float64', 'ForecasterRecursiveMultiSeries-Float64', 
           'ForecasterDirectMultiVariate-pyarrow'])
def test_check_predict_input_MissingValuesWarning_when_missing_values_used_to_predict(
    forecaster_name, last_window, levels, series_names_in_
):
    """
    Test MissingValuesWarning is issued when the missing values of `last_window`
    are used to predict. ForecasterStats uses the whole `last_window`.
    """
    warn_msg = re.escape(
        "`last_window` has missing values. Most of machine learning models do "
        "not allow missing values. Prediction method may either raise an "
        "error or return NaN predictions."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name   = forecaster_name,
            steps             = 2,
            is_fitted         = True,
            exog_in_          = False,
            index_type_       = pd.RangeIndex,
            index_freq_       = 1,
            window_size       = 5,
            last_window       = last_window,
            levels            = levels,
            levels_forecaster = levels,
            series_names_in_  = series_names_in_
        )


@pytest.mark.parametrize("values", 
    [pd.Series([1., np.nan, 3., 4., 5.]),
     pd.Series([1., pd.NA, 3., 4., 5.], dtype='Float64'),
     pd.Series([1., None, 3., 4., 5.], dtype='double[pyarrow]'),
     pd.Series([1, np.nan, 3, 1, 2]).astype('category'),
     pd.Series(pd.to_datetime(['2020-01-01', None, '2020-01-03', '2020-01-04', '2020-01-05']))], 
    ids = ['float64', 'Float64', 'double[pyarrow]', 'category', 'datetime'])
def test_check_predict_input_MissingValuesWarning_when_exog_has_missing_values_of_any_dtype(values):
    """
    Test MissingValuesWarning is issued when `exog` has missing values of any 
    dtype, also in a DataFrame with columns of different dtypes.
    """
    exog = pd.DataFrame(
        {'exog_1': np.arange(5, dtype=float), 'exog_2': values.to_numpy()},
        index = pd.RangeIndex(start=10, stop=15)
    ).astype({'exog_2': values.dtype})

    warn_msg = re.escape(
        "`exog` has missing values. Most of machine learning models "
        "do not allow missing values. Prediction method may fail."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = 1,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10, dtype=float)),
            exog             = exog,
            exog_names_in_   = ['exog_1', 'exog_2']
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=MissingValuesWarning)
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = 1,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10, dtype=float)),
            exog             = exog.dropna().reindex(exog.index).ffill().bfill(),
            exog_names_in_   = ['exog_1', 'exog_2']
        )


def test_check_predict_input_TypeError_when_last_window_index_is_not_of_index_type():
    """
    """
    last_window = pd.Series(np.arange(10))
    index_type_ = pd.DatetimeIndex
    _, last_window_index = check_extract_values_and_index(
        data=last_window, data_label='`last_window`', ignore_freq=False, return_values=False
    )

    err_msg = re.escape(
        f"Expected index of type {index_type_} for `last_window`. "
        f"Got {type(last_window_index)}."
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_last_window_index_frequency_is_not_index_freq():
    """
    """
    last_window = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq='D'))
    index_freq_ = 'YE'
    _, last_window_index = check_extract_values_and_index(
        data=last_window, data_label='`last_window`', ignore_freq=False, return_values=False
    )

    err_msg = re.escape(
        f"Expected frequency of type {index_freq_} for `last_window`. "
        f"Got {last_window_index.freq}."
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_last_window_RangeIndex_step_is_not_index_freq():
    """
    Test TypeError is raised when `last_window` has a RangeIndex with a step
    that does not match `index_freq_`.
    """
    index_freq_ = 1
    last_window = pd.Series(np.arange(10), index=pd.RangeIndex(start=0, stop=20, step=2))
    _, last_window_index = check_extract_values_and_index(
        data=last_window, data_label='`last_window`', ignore_freq=False, return_values=False
    )

    err_msg = re.escape(
        f"Expected step of type {index_freq_} for `last_window`. "
        f"Got {last_window_index.step}."
    )
    with pytest.raises(TypeError, match=err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = pd.RangeIndex,
            index_freq_      = index_freq_,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = None,
            exog_names_in_   = None,
            max_step         = None,
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
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
            exog_names_in_   = None,
            max_step         = None,
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_MissingExogWarning_when_exog_dict_and_no_key_for_some_levels():
    """
    """
    warn_msg = re.escape(
        "`exog` does not contain keys for levels {'l2'}. "
        "Missing levels are filled with NaN. Most of machine learning "
        "models do not allow missing values. Prediction method may fail."
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
            exog_names_in_   = ['exog_1'],
            max_step         = None,
            levels           = ['l1', 'l2'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_MissingValuesWarning_when_exog_has_missing_values():
    """
    """
    warn_msg = re.escape(
        "`exog` has missing values. Most of machine learning models do "
        "not allow missing values. Prediction method may fail."
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
            exog_names_in_   = ['exog1'],
            max_step         = None,
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
        f"`exog` doesn't have as many values as steps "
        f"predicted, {last_step}. Missing values are filled "
        f"with NaN. Most of machine learning models do not "
        f"allow missing values. Prediction method may fail."
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
            exog_names_in_   = ['exog1'],
            max_step         = None,
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
        f"`exog` must have at least as many values as "
        f"steps predicted, {last_step}."
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
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_MissingExogWarning_when_exog_is_DataFrame_without_columns_in_exog_names_in__MultiSeries():
    """
    Raise MissingExogWarning when there are missing columns in `exog` when 
    Forecaster multi series.
    """
    exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col2'])
    exog.index = pd.date_range(start='11/30/2018', periods=5, freq=freq)
    exog_names_in_ = ['col1', 'col3']

    warn_msg = re.escape(
        "{'col3'} not present in `exog`. All values will be NaN."
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
            exog_names_in_   = exog_names_in_,
            max_step         = None,
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
        f"Missing columns in `exog`. Expected {exog_names_in_}. "
        f"Got {exog.columns.to_list()}."
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
            exog_names_in_   = exog_names_in_,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_exog_is_Series_and_exog_names_in_has_more_columns():
    """
    Raise ValueError when `exog` is a pandas Series whose name is in 
    `exog_names_in_`, but the forecaster was trained with more exogenous 
    variables. Before, the forecaster raised a KeyError without context.
    """
    exog = pd.Series(
        np.arange(5), name='col1', 
        index=pd.date_range(start='1/11/2018', periods=5, freq=freq)
    )
    exog_names_in_ = ['col1', 'col3']

    err_msg = re.escape(
        f"Missing columns in `exog`. Expected {exog_names_in_}. Got ['col1']."
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
            exog_names_in_   = exog_names_in_,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


@pytest.mark.parametrize("exog_type", 
                         ['wide', 'dict'], 
                         ids = lambda exog_type: f'exog_type: {exog_type}')
def test_check_predict_input_MissingExogWarning_when_exog_is_Series_and_exog_names_in_has_more_columns_MultiSeries(exog_type):
    """
    Raise MissingExogWarning when `exog` is a pandas Series whose name is in 
    `exog_names_in_`, but the forecaster was trained with more exogenous 
    variables, when Forecaster multi series. Before, no warning was issued 
    with a dict and the predictions of that series were NaN.
    """
    exog = pd.Series(
        np.arange(5), name='col1', 
        index=pd.date_range(start='1/11/2018', periods=5, freq=freq)
    )
    exog_name = '`exog`'
    if exog_type == 'dict':
        exog = {'l1': exog}
        exog_name = "`exog` for series 'l1'"
    exog_names_in_ = ['col1', 'col3']

    warn_msg = re.escape(
        f"{{'col3'}} not present in {exog_name}. All values will be NaN."
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
            exog_names_in_   = exog_names_in_,
            max_step         = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_ValueError_when_exog_is_Series_with_no_name():
    """
    Raise ValueError when `exog` is a pandas Series with no name.
    """
    exog = pd.Series(np.arange(10))
    exog_names_in_ = ['exog1']

    err_msg = re.escape(
        "When `exog` is a pandas Series, it must have a name. Got None."
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
            exog_names_in_   = exog_names_in_,
            max_step         = None,
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
        "'exog2' was not observed during training. `exog` is ignored. "
        "Exogenous variables must be one of: ['exog1']."
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
            exog_names_in_   = exog_names_in_,
            max_step         = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_ValueError_when_exog_is_Series_with_name_not_in_exog_names_in_():
    """
    Raise ValueError when `exog` is a pandas Series and its name is not in 
    `exog_names_in_`.
    """
    exog = pd.Series(np.arange(10), name='exog2')
    exog_names_in_ = ['exog1']

    err_msg = re.escape(
        "'exog2' was not observed during training. "
        "Exogenous variables must be: ['exog1']."
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
            exog_names_in_   = exog_names_in_,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_exog_index_is_not_of_index_type_multiseries():
    """
    """
    exog = pd.Series(np.arange(10), name='exog1')
    exog.index = pd.RangeIndex(start=0, stop=10, step=1)
    index_type_ = pd.DatetimeIndex

    last_window = pd.Series(
        np.arange(10), 
        index = pd.date_range(start='01/01/2018', periods=10, freq=freq), 
        name  = 'l1'
    ).to_frame()

    err_msg = re.escape(
        f"Expected index of type {index_type_} for `exog`. "
        f"Got {type(exog.index)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = index_type_,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = last_window,
            last_window_exog = None,
            exog             = exog,
            exog_names_in_   = ['exog1'],
            max_step         = None,
            levels           = ['l1'],
            series_names_in_ = ['l1', 'l2']
        )


def test_check_predict_input_TypeError_when_exog_index_is_not_of_index_type():
    """
    """
    exog = pd.Series(np.arange(10), name='exog1')
    index_type_ = pd.DatetimeIndex
    check_exog(exog = exog)
    _, exog_index = check_extract_values_and_index(
        data=exog, data_label='`exog`', ignore_freq=True, return_values=False
    )

    err_msg = re.escape(
        f"Expected index of type {index_type_} for `exog`. "
        f"Got {type(exog_index)}."
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
            exog_names_in_   = ['exog1'],
            max_step         = None,
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
        f"To make predictions `exog` must start one step "
        f"ahead of `last_window`.\n"
        f"    `last_window` ends at : {lw_datetime.index[-1]}.\n"
        f"    `exog` starts at : {exog_datetime.index[0]}.\n"
        f"    Expected index : {expected_index}."
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
            exog_names_in_   = ['exog1'],
            max_step         = None,
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
        f"To make predictions `exog` must start one step "
        f"ahead of `last_window`.\n"
        f"    `last_window` ends at : {lw_datetime.index[-1]}.\n"
        f"    `exog` starts at : {exog_datetime.index[0]}.\n"
        f"    Expected index : {expected_index}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursive',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = 1,
            window_size      = 5,
            last_window      = lw_datetime,
            last_window_exog = None,
            exog             = exog_datetime,
            exog_names_in_   = ['exog1'],
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


@pytest.mark.parametrize(
    'forecaster_name, steps, exog_index, position, expected_date, exog_date',
    [
        ('ForecasterRecursive', 5,
         pd.date_range(start='2020-02-20', periods=5, freq='h'),
         1, '2020-02-21 00:00:00', '2020-02-20 01:00:00'),
        ('ForecasterRecursive', 5,
         pd.DatetimeIndex(['2020-02-20', '2020-02-22', '2020-02-24',
                           '2020-02-26', '2020-02-28']),
         1, '2020-02-21 00:00:00', '2020-02-22 00:00:00'),
        ('ForecasterRecursive', 5,
         pd.DatetimeIndex(['2020-02-20', '2020-02-21', '2020-02-21',
                           '2020-02-22', '2020-02-23']),
         2, '2020-02-22 00:00:00', '2020-02-21 00:00:00'),
        ('ForecasterDirect', [3, 4, 5],
         pd.DatetimeIndex(['2020-02-20', '2020-02-22', '2020-02-24',
                           '2020-02-26', '2020-02-28']),
         1, '2020-02-21 00:00:00', '2020-02-22 00:00:00'),
    ],
    ids=['hourly', 'gaps_without_freq', 'duplicated_date', 'Direct_steps_3_4_5']
)
def test_check_predict_input_ValueError_when_exog_index_does_not_follow_freq(
    forecaster_name, steps, exog_index, position, expected_date, exog_date
):
    """
    Test ValueError is raised when `exog` starts one step ahead of
    `last_window`, but its index does not follow the frequency of `last_window`
    for the steps predicted (other frequency, gaps or duplicated dates).
    Forecasters use `exog` by position, so they would use values of other dates.
    """
    last_window = pd.Series(
        data  = np.arange(10, dtype=float),
        index = pd.date_range(start='2020-02-10', periods=10, freq='D')
    )
    exog = pd.Series(data=np.arange(5, dtype=float), index=exog_index, name='exog1')

    err_msg = re.escape(
        f"`exog` must have consecutive values following the frequency of "
        f"`last_window` for the 5 steps predicted.\n"
        f"    Expected index at position {position} : {expected_date}.\n"
        f"    `exog` index at position {position} : {exog_date}.\n"
        f"If there is no data for some steps, add them to `exog` explicitly "
        f"as NaN, for example:\n"
        f"    exog = exog.reindex(expand_index(last_window.index, steps=5))\n"
        f"where `expand_index` is in `skforecast.utils`, and `last_window` is "
        f"the window used to predict (by default, the last window stored in "
        f"the forecaster)."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name = forecaster_name,
            steps           = steps,
            is_fitted       = True,
            exog_in_        = True,
            index_type_     = pd.DatetimeIndex,
            index_freq_     = 'D',
            window_size     = 5,
            last_window     = last_window,
            exog            = exog,
            exog_names_in_  = ['exog1']
        )


def test_check_predict_input_ValueError_when_exog_RangeIndex_has_other_step():
    """
    Test ValueError is raised when `exog` starts one step ahead of
    `last_window`, but its RangeIndex has a different step.
    """
    last_window = pd.Series(
        data  = np.arange(10, dtype=float),
        index = pd.RangeIndex(start=0, stop=10)
    )
    exog = pd.Series(
        data  = np.arange(5, dtype=float),
        index = pd.RangeIndex(start=10, stop=20, step=2),
        name  = 'exog1'
    )

    err_msg = re.escape(
        "`exog` must have consecutive values following the frequency of "
        "`last_window` for the 5 steps predicted.\n"
        "    Expected index at position 1 : 11.\n"
        "    `exog` index at position 1 : 12.\n"
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name = 'ForecasterRecursive',
            steps           = 5,
            is_fitted       = True,
            exog_in_        = True,
            index_type_     = pd.RangeIndex,
            index_freq_     = 1,
            window_size     = 5,
            last_window     = last_window,
            exog            = exog,
            exog_names_in_  = ['exog1']
        )


@pytest.mark.parametrize(
    'last_window_index, exog_index',
    [
        (pd.date_range(start='2020-02-10', periods=10, freq='D'),
         pd.date_range(start='2020-02-20', periods=15, freq='D')),
        (pd.date_range(start='2020-02-10', periods=10, freq='D'),
         pd.DatetimeIndex(['2020-02-20', '2020-02-21', '2020-02-22',
                           '2020-02-23', '2020-02-24'])),
        (pd.date_range(start='2020-02-10', periods=10, freq='D'),
         pd.DatetimeIndex(['2020-02-20', '2020-02-21', '2020-02-22',
                           '2020-02-23', '2020-02-24', '2020-03-01'])),
        (pd.date_range(start='2024-03-20', periods=10, freq='D', tz='Europe/Madrid'),
         pd.DatetimeIndex(['2024-03-30', '2024-03-31', '2024-04-01',
                           '2024-04-02', '2024-04-03'], tz='Europe/Madrid')),
        (pd.date_range(end='2024-03-31', periods=10, freq='D', tz='Europe/Madrid'),
         pd.DatetimeIndex(['2024-04-01', '2024-04-02', '2024-04-03',
                           '2024-04-04', '2024-04-05'], tz='Europe/Madrid')),
    ],
    ids=['extra_rows', 'without_freq', 'gap_after_last_step',
         'tz_aware_crosses_dst', 'tz_aware_last_window_ends_on_dst_day']
)
def test_check_predict_input_no_error_when_exog_index_follows_last_window_freq(
    last_window_index, exog_index
):
    """
    Test no error or warning is raised when the index of `exog` follows the
    frequency of `last_window` for the steps predicted: with extra rows, without
    `freq` (e.g. read from a CSV file), with a gap after the last step predicted
    and timezone-aware across a daylight saving change (Europe/Madrid).
    """
    last_window = pd.Series(data=np.arange(10, dtype=float), index=last_window_index)
    exog = pd.Series(
        data  = np.arange(len(exog_index), dtype=float),
        index = exog_index,
        name  = 'exog1'
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        check_predict_input(
            forecaster_name = 'ForecasterRecursive',
            steps           = 5,
            is_fitted       = True,
            exog_in_        = True,
            index_type_     = pd.DatetimeIndex,
            index_freq_     = 'D',
            window_size     = 5,
            last_window     = last_window,
            exog            = exog,
            exog_names_in_  = ['exog1']
        )


def test_check_predict_input_MissingValuesWarning_when_exog_gaps_are_reindexed():
    """
    Test that `exog` with gaps is accepted once the missing dates are added as
    NaN with `expand_index`, as suggested in the error message. Only the
    warning of `exog` with missing values is issued.
    """
    last_window = pd.Series(
        data  = np.arange(10, dtype=float),
        index = pd.date_range(start='2020-02-10', periods=10, freq='D')
    )
    exog = pd.Series(
        data  = np.arange(5, dtype=float),
        index = pd.DatetimeIndex(['2020-02-20', '2020-02-22', '2020-02-24',
                                  '2020-02-26', '2020-02-28']),
        name  = 'exog1'
    )
    exog = exog.reindex(expand_index(last_window.index, steps=5))

    warn_msg = re.escape(
        "`exog` has missing values. Most of machine learning models do "
        "not allow missing values. Prediction method may fail."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name = 'ForecasterRecursive',
            steps           = 5,
            is_fitted       = True,
            exog_in_        = True,
            index_type_     = pd.DatetimeIndex,
            index_freq_     = 'D',
            window_size     = 5,
            last_window     = last_window,
            exog            = exog,
            exog_names_in_  = ['exog1']
        )


@pytest.mark.parametrize(
    'exog_format', ['wide', 'dict'], ids=lambda fmt: f'exog: {fmt}'
)
@pytest.mark.parametrize(
    'exog_index, position, missing_date',
    [
        (pd.DatetimeIndex(['2020-02-20', '2020-02-22', '2020-02-24',
                           '2020-02-26', '2020-02-28']),
         1, '2020-02-21 00:00:00'),
        (pd.date_range(start='2020-02-22', periods=5, freq='D'),
         0, '2020-02-20 00:00:00'),
    ],
    ids=['gaps', 'starts_2_steps_late']
)
def test_check_predict_input_MissingValuesWarning_when_MultiSeries_exog_misses_dates(
    exog_index, position, missing_date, exog_format
):
    """
    Test MissingValuesWarning is raised in ForecasterRecursiveMultiSeries when
    `exog` (wide or dict) does not have the dates of some of the steps
    predicted. `exog` is aligned with the predictions by date, so those values
    are NaN.
    """
    last_window = pd.DataFrame(
        data  = {'l1': np.arange(10, dtype=float)},
        index = pd.date_range(start='2020-02-10', periods=10, freq='D')
    )
    exog = pd.Series(data=np.arange(5, dtype=float), index=exog_index, name='exog1')
    if exog_format == 'dict':
        exog = {'l1': exog}
    exog_name = "`exog`" if exog_format == 'wide' else "`exog` for series 'l1'"

    warn_msg = re.escape(
        f"{exog_name} has no value for some of the 5 steps predicted. The "
        f"first one is {missing_date} (position {position}). Missing values "
        f"are filled with NaN. Most of machine learning models do not allow "
        f"missing values. Prediction method may fail."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = 'D',
            window_size      = 5,
            last_window      = last_window,
            exog             = exog,
            exog_names_in_   = ['exog1'],
            levels           = ['l1'],
            series_names_in_ = ['l1']
        )


@pytest.mark.parametrize(
    'exog_format', ['wide', 'dict'], ids=lambda fmt: f'exog: {fmt}'
)
def test_check_predict_input_MissingValuesWarning_when_MultiSeries_exog_is_empty(
    exog_format
):
    """
    Test that, in ForecasterRecursiveMultiSeries, an empty `exog` (wide or
    dict) only issues the MissingValuesWarning of `exog` shorter than steps.
    `exog` is aligned with the predictions by date, so all its values are NaN.
    """
    last_window = pd.DataFrame(
        data  = {'l1': np.arange(10, dtype=float)},
        index = pd.date_range(start='2020-02-10', periods=10, freq='D')
    )
    exog = pd.Series(
        data=[], index=pd.DatetimeIndex([]), name='exog1', dtype=float
    )
    if exog_format == 'dict':
        exog = {'l1': exog}
    exog_name = "`exog`" if exog_format == 'wide' else "`exog` for series 'l1'"

    warn_msg = re.escape(
        f"{exog_name} doesn't have as many values as steps predicted, 5. "
        f"Missing values are filled with NaN. Most of machine learning models "
        f"do not allow missing values. Prediction method may fail."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg) as record:
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = 'D',
            window_size      = 5,
            last_window      = last_window,
            exog             = exog,
            exog_names_in_   = ['exog1'],
            levels           = ['l1'],
            series_names_in_ = ['l1']
        )

    assert len(record) == 1


@pytest.mark.parametrize(
    'exog_format', ['wide', 'dict'], ids=lambda fmt: f'exog: {fmt}'
)
@pytest.mark.parametrize(
    'exog_index',
    [
        pd.date_range(start='2020-02-01', periods=30, freq='D'),
        pd.date_range(start='2020-02-20', periods=120, freq='h'),
    ],
    ids=['starts_before_first_step', 'hourly']
)
def test_check_predict_input_no_warning_when_MultiSeries_exog_has_all_dates(
    exog_index, exog_format
):
    """
    Test no warning is raised in ForecasterRecursiveMultiSeries when `exog`
    (wide or dict) does not start at the first step predicted or has another
    frequency, but it has the dates of all the steps predicted. `exog` is
    aligned with the predictions by date, so no value is missing.
    """
    last_window = pd.DataFrame(
        data  = {'l1': np.arange(10, dtype=float)},
        index = pd.date_range(start='2020-02-10', periods=10, freq='D')
    )
    exog = pd.Series(
        data  = np.arange(len(exog_index), dtype=float),
        index = exog_index,
        name  = 'exog1'
    )
    if exog_format == 'dict':
        exog = {'l1': exog}

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = 'D',
            window_size      = 5,
            last_window      = last_window,
            exog             = exog,
            exog_names_in_   = ['exog1'],
            levels           = ['l1'],
            series_names_in_ = ['l1']
        )


@pytest.mark.parametrize(
    'exog_format', ['wide', 'dict'], ids=lambda fmt: f'exog: {fmt}'
)
def test_check_predict_input_ValueError_when_MultiSeries_exog_has_duplicated_dates(
    exog_format
):
    """
    Test ValueError is raised in ForecasterRecursiveMultiSeries when the index
    of `exog` (wide or dict) has duplicated dates, since it cannot be aligned
    with the predictions by date.
    """
    last_window = pd.DataFrame(
        data  = {'l1': np.arange(10, dtype=float)},
        index = pd.date_range(start='2020-02-10', periods=10, freq='D')
    )
    exog = pd.Series(
        data  = np.arange(5, dtype=float),
        index = pd.DatetimeIndex(['2020-02-20', '2020-02-21', '2020-02-21',
                                  '2020-02-22', '2020-02-23']),
        name  = 'exog1'
    )
    if exog_format == 'dict':
        exog = {'l1': exog}
    exog_name = "`exog`" if exog_format == 'wide' else "`exog` for series 'l1'"

    err_msg = re.escape(
        f"The index of {exog_name} has duplicated values, for example "
        f"2020-02-21 00:00:00. Each date must appear only once."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterRecursiveMultiSeries',
            steps            = 5,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = 'D',
            window_size      = 5,
            last_window      = last_window,
            exog             = exog,
            exog_names_in_   = ['exog1'],
            levels           = ['l1'],
            series_names_in_ = ['l1']
        )


def test_check_predict_input_ValueError_when_ForecasterStats_last_window_exog_is_not_None_and_exog_in_is_false():
    """
    Check ValueError is raised when last_window_exog is not None, but exog_in_     
    is False.
    """
    err_msg = re.escape(
        "Forecaster trained without exogenous variable/s. "
        "`last_window_exog` must be `None` when predicting."
    )   
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterStats',
            steps            = 3,
            is_fitted        = True,
            exog_in_         = False,
            index_type_      = pd.RangeIndex,
            index_freq_      = 1,
            window_size      = 3,
            last_window      = pd.Series(np.arange(5)),
            last_window_exog = pd.Series(np.arange(5)),
            exog             = None,
            exog_names_in_   = None,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_ForecasterStats_last_window_exog_is_not_pandas_Series_or_DataFrame():
    """
    """
    last_window_exog = np.arange(5)
    err_msg = re.escape(
        f"`last_window_exog` must be a pandas Series or a "
        f"pandas DataFrame. Got {type(last_window_exog)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterStats',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = 1,
            window_size      = 3,
            last_window      = pd.Series(np.arange(5)),
            last_window_exog = last_window_exog,
            exog             = pd.Series(np.arange(10), index=pd.RangeIndex(start=5, stop=15), name='exog1'),
            exog_names_in_   = ['exog1'],
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_ForecasterStats_length_last_window_exog_is_lower_than_window_size():
    """
    """
    window_size = 10

    err_msg = re.escape(
        f"`last_window_exog` must have as many values as needed to "
        f"generate the predictors. For this forecaster it is {window_size}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterStats',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = 1,
            window_size      = window_size,
            last_window      = pd.Series(np.arange(10)),
            last_window_exog = pd.Series(np.arange(5)),
            exog             = pd.Series(np.arange(10), index=pd.RangeIndex(start=10, stop=20), name='exog1'),
            exog_names_in_   = ['exog1'],
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_MissingValuesWarning_when_ForecasterStats_last_window_exog_has_missing_values():
    """
    """
    warn_msg = re.escape(
        "`last_window_exog` has missing values. Most of machine learning models "
        "do not allow missing values. Prediction method may fail."
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterStats',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.RangeIndex,
            index_freq_      = 1,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10)),
            last_window_exog = pd.Series([1, 2, 3, 4, 5, np.nan], name='exog1'),
            exog             = pd.Series(np.arange(10), index=pd.RangeIndex(start=10, stop=20), name='exog1'),
            exog_names_in_   = ['exog1'],
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_ForecasterStats_last_window_exog_index_is_not_of_index_type():
    """
    """
    last_window_exog = pd.Series(np.arange(10), name='exog1')
    index_type_ = pd.DatetimeIndex
    _, last_window_exog_index = check_extract_values_and_index(
        data=last_window_exog, data_label='`last_window_exog`', return_values=False
    )

    err_msg = re.escape(
        f"Expected index of type {index_type_} for `last_window_exog`. "
        f"Got {type(last_window_exog_index)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterStats',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = index_type_,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = pd.Series(np.arange(10), index=pd.date_range(start='30/11/2018', periods=10, freq=freq), name='exog1'),
            exog_names_in_   = ['exog1'],
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_TypeError_when_ForecasterStats_last_window_exog_index_frequency_is_not_index_freq():
    """
    """
    
    last_window_exog = pd.Series(np.arange(10), index=pd.date_range(start='2018', periods=10, freq='YE'))
    index_freq_ = freq
    _, last_window_exog_index = check_extract_values_and_index(
        data=last_window_exog, data_label='`last_window_exog`', return_values=False
    )

    err_msg = re.escape(
        f"Expected frequency of type {index_freq_} for `last_window_exog`. "
        f"Got {last_window_exog_index.freq}."
    )
    with pytest.raises(TypeError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterStats',
            steps            = 10,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = index_freq_,
            window_size      = 5,
            last_window      = pd.Series(np.arange(10), index=pd.date_range(start='1/1/2018', periods=10, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = pd.Series(np.arange(10), index=pd.date_range(start='30/11/2018', periods=10, freq=freq), name='exog1'),
            exog_names_in_   = ['exog1'],
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_last_window_exog_is_DataFrame_without_columns_in_exog_names_in_():
    """
    Raise ValueError when there are missing columns in `last_window_exog`, ForecasterStats.
    """
    
    exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col3'])
    exog.index = pd.date_range(start='6/1/2018', periods=5, freq=freq)
    exog_names_in_ = ['col1', 'col3']

    last_window_exog = pd.DataFrame(np.arange(10).reshape(5, 2), columns=['col1', 'col2'])
    last_window_exog.index = pd.date_range(start='1/1/2018', periods=5, freq=freq)

    err_msg = re.escape(
        f"Missing columns in `last_window_exog`. Expected {exog_names_in_}. "
        f"Got {last_window_exog.columns.to_list()}."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterStats',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(5), index=pd.date_range(start='1/1/2018', periods=5, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = exog,
            exog_names_in_   = exog_names_in_,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_last_window_exog_is_Series_with_no_name():
    """
    Raise ValueError when `last_window_exog` has no name, ForecasterStats.
    """
    
    exog = pd.Series(np.arange(5), name='exog1')
    exog.index = pd.date_range(start='6/1/2018', periods=5, freq=freq)
    exog_names_in_ = ['exog1']

    last_window_exog = pd.Series(np.arange(5))
    last_window_exog.index = pd.date_range(start='1/1/2018', periods=5, freq=freq)
    
    err_msg = re.escape(
        "When `last_window_exog` is a pandas Series, it must have a name. Got None."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterStats',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(5), index=pd.date_range(start='1/1/2018', periods=5, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = exog,
            exog_names_in_   = exog_names_in_,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )


def test_check_predict_input_ValueError_when_last_window_exog_is_Series_without_name_in_exog_names_in_():
    """
    Raise ValueError when `last_window_exog` name is not in `exog_names_in_`, ForecasterStats.
    """
    exog = pd.Series(np.arange(5), name='exog1')
    exog.index = pd.date_range(start='6/1/2018', periods=5, freq=freq)
    exog_names_in_ = ['exog1']

    last_window_exog = pd.Series(np.arange(5), name='exog2')
    last_window_exog.index = pd.date_range(start='1/1/2018', periods=5, freq=freq)
    
    err_msg = re.escape(
        "'exog2' was not observed during training. "
        "Exogenous variables must be: ['exog1']."
    )
    with pytest.raises(ValueError, match = err_msg):
        check_predict_input(
            forecaster_name  = 'ForecasterStats',
            steps            = 2,
            is_fitted        = True,
            exog_in_         = True,
            index_type_      = pd.DatetimeIndex,
            index_freq_      = freq,
            window_size      = 5,
            last_window      = pd.Series(np.arange(5), index=pd.date_range(start='1/1/2018', periods=5, freq=freq)),
            last_window_exog = last_window_exog,
            exog             = exog,
            exog_names_in_   = exog_names_in_,
            max_step         = None,
            levels           = None,
            series_names_in_ = None
        )
