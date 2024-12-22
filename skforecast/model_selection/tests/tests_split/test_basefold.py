# Unit test BaseFold
# ==============================================================================
import pytest
import re
import numpy as np
import pandas as pd
from ....exceptions import IgnoredArgumentWarning
from skforecast.model_selection._split import BaseFold


valid_params = {
    "steps": 5,
    "initial_train_size": 100,
    "window_size": 10,
    "differentiation": None,
    "refit": True,
    "fixed_train_size": True,
    "gap": 0,
    "skip_folds": None,
    "allow_incomplete_fold": True,
    "return_all_indexes": False,
    "verbose": True,
}


def test_basefold_validate_params_no_exception():
    """
    Test that no exceptions are raised for valid parameters.
    """
    cv = BaseFold()
    params = dict(valid_params)
    cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_steps():
    """
    Test that ValueError is raised when steps is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["steps"] = 0
    msg = f"`steps` must be an integer greater than 0. Got {params['steps']}."
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["steps"] = "invalid"
    msg = f"`steps` must be an integer greater than 0. Got {params['steps']}."
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["steps"] = 1.0
    msg = f"`steps` must be an integer greater than 0. Got {params['steps']}."
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_initial_train_size_TimeSeriesFold():
    """
    Test that ValueError is raised when initial_train_size is invalid 
    for TimeSeriesFold.
    """
    cv = BaseFold()
    params = dict(valid_params)

    # Test invalid string
    params["initial_train_size"] = "invalid"
    msg = (
        "`initial_train_size` must be a valid date string accepted "
        "by pandas to_datetime, an integer, a pandas "
        f"Timestamp, or None. Got {params["initial_train_size"]}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)
    
    # Test invalid integer
    params["initial_train_size"] = -1
    msg = (
        "`initial_train_size` must be an integer greater than 0, "
        f"a date string, a pandas Timestamp, or None. Got {params["initial_train_size"]}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)
    
    # Test invalid type
    params["initial_train_size"] = 1.0
    msg = (
        "`initial_train_size` must be an integer, a date string, "
        f"a pandas Timestamp, or None. Got {params["initial_train_size"]}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)
    
    # Test valid None
    params["initial_train_size"] = None
    cv._validate_params(cv_name="TimeSeriesFold", **params)

    # Test valid integer
    params["initial_train_size"] = 10
    cv._validate_params(cv_name="TimeSeriesFold", **params)
    
    # Test valid date string
    params["initial_train_size"] = "2022-01-01"
    cv._validate_params(cv_name="TimeSeriesFold", **params)
    
    # Test valid pandas Timestamp
    params["initial_train_size"] = pd.Timestamp("2022-01-01")
    cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_refit():
    """
    Test that TypeError is raised when refit is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["refit"] = "invalid"
    msg = "`refit` must be a boolean or an integer equal or greater than 0. Got invalid."
    with pytest.raises(TypeError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["refit"] = -1
    msg = "`refit` must be a boolean or an integer equal or greater than 0. Got -1."
    with pytest.raises(TypeError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_fixed_train_size():
    """
    Test that TypeError is raised when fixed_train_size is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["fixed_train_size"] = "invalid"
    msg = "`fixed_train_size` must be a boolean: `True`, `False`. Got invalid."
    with pytest.raises(TypeError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_gap():
    """
    Test that ValueError is raised when gap is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["gap"] = -1
    msg = "`gap` must be an integer greater than or equal to 0. Got -1."
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["gap"] = "invalid"
    msg = "`gap` must be an integer greater than or equal to 0. Got invalid."
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["gap"] = 1.0
    msg = "`gap` must be an integer greater than or equal to 0. Got 1.0."
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_skip_folds():
    """
    Test that ValueError is raised when skip_folds is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["skip_folds"] = 'invalid'
    msg = (
        "`skip_folds` must be an integer greater than 0, a list of "
        "integers or `None`. Got invalid."
    )
    with pytest.raises(TypeError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["skip_folds"] = -1
    msg = (
        f"`skip_folds` must be an integer greater than 0, a list of "
        f"integers or `None`. Got {params['skip_folds']}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["skip_folds"] = [1, 2, -1]
    msg = (
        f"`skip_folds` list must contain integers greater than or "
        f"equal to 1. The first fold is always needed to train the "
        f"forecaster. Got {params['skip_folds']}."
    )
    msg = re.escape(msg)
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_allow_incomplete_fold():
    """
    Test that TypeError is raised when allow_incomplete_fold is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["allow_incomplete_fold"] = "invalid"
    msg = (
        f"`allow_incomplete_fold` must be a boolean: `True`, `False`. "
        f"Got {params['allow_incomplete_fold']}."
    )
    with pytest.raises(TypeError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_initial_train_size_OneStepAheadFold():
    """
    Test that ValueError is raised when initial_train_size is invalid for 
    OneStepAheadFold.
    """
    cv = BaseFold()
    params = dict(valid_params)
    
    # Test invalid string
    params["initial_train_size"] = "invalid"
    msg = (
        "`initial_train_size` must be a valid date string accepted "
        "by pandas to_datetime, an integer, or a pandas Timestamp. "
        f"Got {params['initial_train_size']}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="OneStepAheadFold", **params)
    
    # Test invalid integer
    params["initial_train_size"] = 0
    msg = (
        "`initial_train_size` must be an integer greater than 0, "
        f"a date string, or a pandas Timestamp. Got {params['initial_train_size']}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="OneStepAheadFold", **params)
    
    # Test invalid type
    params["initial_train_size"] = 1.5
    msg = (
        "`initial_train_size` must be an integer, a date string, or "
        f"a pandas Timestamp. Got {params['initial_train_size']}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="OneStepAheadFold", **params)
    
    # Test valid integer
    params["initial_train_size"] = 10
    cv._validate_params(cv_name="OneStepAheadFold", **params)
    
    # Test valid date string
    params["initial_train_size"] = "2022-01-01"
    cv._validate_params(cv_name="OneStepAheadFold", **params)
    
    # Test valid pandas Timestamp
    params["initial_train_size"] = pd.Timestamp("2022-01-01")
    cv._validate_params(cv_name="OneStepAheadFold", **params)


def test_basefold_validate_params_raise_invalid_return_all_indexes():
    """
    Test that TypeError is raised when return_all_indexes is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["return_all_indexes"] = "invalid"
    msg = (
        f"`return_all_indexes` must be a boolean: `True`, `False`. "
        f"Got {params['return_all_indexes']}."
    )
    with pytest.raises(TypeError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_differentiation():
    """
    Test that ValueError is raised when differentiation is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["differentiation"] = "invalid"
    msg = (
        f"`differentiation` must be None or an integer greater than or equal to 0. "
        f"Got {params['differentiation']}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["differentiation"] = -1
    msg = (
        f"`differentiation` must be None or an integer greater than or equal to 0. "
        f"Got {params['differentiation']}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["differentiation"] = 1.0
    msg = (
        f"`differentiation` must be None or an integer greater than or equal to 0. "
        f"Got {params['differentiation']}."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_validate_params_raise_invalid_verbose():
    """
    Test that TypeError is raised when verbose is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["verbose"] = "invalid"
    msg = f"`verbose` must be a boolean: `True`, `False`. Got {params['verbose']}."
    with pytest.raises(TypeError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


def test_basefold_extract_index_when_X_is_series():
    """
    Test that the index is correctly extracted when X is a pd.Series.
    """
    cv = BaseFold()
    X = pd.Series(np.arange(100))
    X.index = pd.date_range(start="2022-01-01", periods=100, freq="D")
    index = cv._extract_index(X)
    pd.testing.assert_index_equal(index, X.index)


def test_basefold_extract_index_when_X_is_pandas():
    """
    Test that the index is correctly extracted when X is a pd.DataFrame.
    """
    cv = BaseFold()
    X = pd.DataFrame(np.arange(100))
    X.index = pd.date_range(start="2022-01-01", periods=100, freq="D")
    index = cv._extract_index(X)
    pd.testing.assert_index_equal(index, X.index)


def test_basefold_extract_index_when_X_is_index():
    """
    Test that the index is correctly extracted when X is a pd.Index.
    """
    cv = BaseFold()
    X = pd.Index(pd.date_range(start="2022-01-01", periods=100, freq="D"))
    index = cv._extract_index(X)
    pd.testing.assert_index_equal(index, X)


def test_basefold_extract_index_when_X_is_dict():
    """
    Test that the index is correctly extracted when X is a dict.
    """
    cv = BaseFold()
    X = {
        "a": pd.Series(np.arange(10), index=pd.date_range(start="2022-01-01", periods=10, freq="D")),
        "b": pd.Series(np.arange(10), index=pd.date_range(start="2022-01-10", periods=10, freq="D")),
    }
    index = cv._extract_index(X)
    expected_index = pd.date_range(start="2022-01-01", end="2022-01-19", freq="D")
    pd.testing.assert_index_equal(index, expected_index)


def test_basefold_extract_index_raise_error_when_X_is_dict_with_series_with_different_freq():
    """
    Test that ValueError is raised when X is a dict with series with different frequencies.
    """
    cv = BaseFold()
    X = {
        "a": pd.Series(np.arange(10), index=pd.date_range(start="2022-01-01", periods=10, freq="D")),
        "b": pd.Series(np.arange(10), index=pd.date_range(start="2022-01-10", periods=10, freq="H")),
    }
    msg = 'All series with frequency must have the same frequency.'
    with pytest.raises(ValueError, match=msg):
        cv._extract_index(X)


def test_basefold_extract_index_raise_error_when_X_is_dict_with_series_non_with_freq():
    """
    Test that ValueError is raised when X is a dict with series with different frequencies.
    """
    cv = BaseFold()
    X = {
        "a": pd.Series(np.arange(10), index=pd.date_range(start="2022-01-01", periods=10)),
        "b": pd.Series(np.arange(10), index=pd.date_range(start="2022-01-10", periods=10)),
    }
    X["a"].index.freq = None
    X["b"].index.freq = None
    msg = 'At least one series must have a frequency.'
    with pytest.raises(ValueError, match=msg):
        cv._extract_index(X)


def test_basefold_set_params_TypeError_non_dict():
    """
    Test TypeError is raised set_params method with non-dictionary input.
    """
    fold = BaseFold()
    new_params = ['not', 'a', 'dict']

    msg = (
        f"`params` must be a dictionary. Got {type(new_params)}."
    )
    with pytest.raises(TypeError, match=msg):
        fold.set_params(new_params)


def test_basefold_set_params_valid_params():
    """
    Test set_params method with valid parameters.
    """
    cv = BaseFold(steps=3, initial_train_size=10)
    new_params = {
        'steps': 5,
        'initial_train_size': 15,
        'verbose': False
    }
    cv.set_params(new_params)
    assert cv.steps == 5
    assert cv.initial_train_size == 15
    assert cv.verbose is False


def test_set_params_partial_update_and_unknown_param():
    """
    Test set_params method with partial update of parameters.
    """
    cv = BaseFold(steps=3, initial_train_size=10, verbose=True)
    new_params = {
        'steps': 5,
        'unknown_param': 5
    }
    warn_msg = re.escape(
        "Unknown parameters: {'unknown_param'}. They have been ignored."
    )
    with pytest.warns(IgnoredArgumentWarning, match=warn_msg):
        cv.set_params(new_params)
    
    assert cv.steps == 5
    assert cv.initial_train_size == 10
    assert cv.verbose is True
