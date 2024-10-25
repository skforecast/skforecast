# Unit test BaseFold
# ==============================================================================
import pytest
import re
import numpy as np
import pandas as pd
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


def test_basefold_validate_params_raise_invalid_initial_train_size():
    """
    Test that ValueError is raised when initial_train_size is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["initial_train_size"] = "invalid"
    msg = (
        "`initial_train_size` must be an integer greater than 0 or None. "
        "Got invalid."
    )
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)
    
    params["initial_train_size"] = -1
    msg = (
        "`initial_train_size` must be an integer greater than 0 or None. "
        "Got -1."
    )
    with pytest.raises(ValueError, match=msg):
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


def test_basefold_validate_params_raise_invalid_window_size():
    """
    Test that ValueError is raised when window_size is invalid.
    """
    cv = BaseFold()
    params = dict(valid_params)
    params["window_size"] = 0
    msg = f"`window_size` must be an integer greater than 0. Got {params['window_size']}."
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["window_size"] = "invalid"
    msg = f"`window_size` must be an integer greater than 0. Got {params['window_size']}."
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)

    params["window_size"] = 1.0
    msg = f"`window_size` must be an integer greater than 0. Got {params['window_size']}."
    with pytest.raises(ValueError, match=msg):
        cv._validate_params(cv_name="TimeSeriesFold", **params)


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
