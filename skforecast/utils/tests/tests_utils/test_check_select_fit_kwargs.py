# Unit test check_select_fit_kwargs
# ==============================================================================
import re
import pytest
import inspect
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skforecast.exceptions import IgnoredArgumentWarning
from skforecast.utils import check_select_fit_kwargs
from lightgbm import LGBMRegressor


def test_check_select_fit_kwargs_raise_TypeError_when_fit_kwargs_is_not_dict():
    """
    Test check_select_fit_kwargs raises TypeError when fit_kwargs is not a dictionary.
    """
    fit_kwargs = 'not_dict'
    estimator = LGBMRegressor()

    err_msg = re.escape(
            f"Argument `fit_kwargs` must be a dict. Got {type(fit_kwargs)}."
        )
    with pytest.raises(TypeError, match = err_msg):
        check_select_fit_kwargs(estimator=estimator, fit_kwargs=fit_kwargs)



def test_check_select_fit_kwargs_IgnoredArgumentWarning_when_fit_kwargs_has_arguments_not_in_fit():
    """
    Test check_select_fit_kwargs issues IgnoredArgumentWarning when fit_kwargs 
    has arguments not in estimator fit method.
    """
    fit_kwargs = {'no_valid_argument': 10}
    estimator = LGBMRegressor()
    non_used_keys = [k for k in fit_kwargs.keys()
                     if k not in inspect.signature(estimator.fit).parameters]
    
    warn_msg = re.escape(
            (f"Argument/s {non_used_keys} ignored since they are not used by the "
             f"estimator's `fit` method.")
        )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):       
        check_select_fit_kwargs(estimator=estimator, fit_kwargs=fit_kwargs)


def test_check_select_fit_kwargs_IgnoredArgumentWarning_when_estimator_fit_has_var_kwargs():
    """
    Test check_select_fit_kwargs issues IgnoredArgumentWarning saying that
    arguments passed through `**kwargs` are not supported when the `fit` method
    of the estimator accepts them (Pipeline).
    """
    fit_kwargs = {'lgbmregressor__feature_name': 'auto'}
    estimator = make_pipeline(StandardScaler(), LGBMRegressor())
    
    warn_msg = re.escape(
        "Argument/s ['lgbmregressor__feature_name'] ignored since they are not "
        "explicit arguments of the estimator's `fit` method. Arguments "
        "passed through `**kwargs`, for example to the steps of a "
        "scikit-learn Pipeline (`step__argument`), are not supported."
    )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):       
        results = check_select_fit_kwargs(estimator=estimator, fit_kwargs=fit_kwargs)

    assert results == {}


def test_check_select_fit_kwargs_IgnoredArgumentWarning_when_fit_kwargs_has_sample_weight():
    """
    Test check_select_fit_kwargs issues IgnoredArgumentWarning when fit_kwargs 
    has sample_weight argument.
    """
    fit_kwargs = {'sample_weight': [1,2,3]}
    estimator = LGBMRegressor()
    
    warn_msg = re.escape(
            ("The `sample_weight` argument is ignored. Use `weight_func` to pass "
             "a function that defines the individual weights for each sample "
             "based on its index.")
        )
    with pytest.warns(IgnoredArgumentWarning, match = warn_msg):       
        check_select_fit_kwargs(estimator=estimator, fit_kwargs=fit_kwargs)


def test_check_select_fit_kwargs_ignores_sample_weight():
    """
    Test check_select_fit_kwargs method ignores sample_weight argument.
    """
    results = check_select_fit_kwargs(
                  estimator  = LGBMRegressor(),
                  fit_kwargs = {'sample_weight': [1,2,3]}
              )
    expected = {}

    assert results == expected


def test_check_select_fit_kwargs_ignores_arguments_not_in_estimator_fit():
    """
    Test check_select_fit_kwargs method ignores arguments not in estimator fit method.
    """
    results = check_select_fit_kwargs(
                  estimator  = LGBMRegressor(),
                  fit_kwargs = {'feature_name':'auto', 'no_valid_argument': 10}
              )
    expected = {'feature_name':'auto'}

    assert results == expected


def test_check_select_fit_kwargs_does_not_modify_user_fit_kwargs():
    """
    Test check_select_fit_kwargs does not modify the dictionary passed by the
    user when `sample_weight` and arguments not in estimator fit are removed.
    """
    fit_kwargs = {'sample_weight': [1, 2, 3], 'feature_name': 'auto', 'no_valid_argument': 10}
    fit_kwargs_copy = fit_kwargs.copy()

    results = check_select_fit_kwargs(
                  estimator  = LGBMRegressor(),
                  fit_kwargs = fit_kwargs
              )

    assert results == {'feature_name': 'auto'}
    assert fit_kwargs == fit_kwargs_copy
