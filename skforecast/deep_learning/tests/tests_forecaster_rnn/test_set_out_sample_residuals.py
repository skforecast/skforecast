# Unit test set_out_sample_residuals ForecasterRnn
# ==============================================================================
import os
import re
import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
os.environ["KERAS_BACKEND"] = "torch"
import keras
from skforecast.deep_learning import create_and_compile_model
from skforecast.deep_learning import ForecasterRnn

# Fixtures
np.random.seed(123)
series = pd.DataFrame(
    {
        "l1": np.random.rand(50),
        "l2": np.random.rand(50)
    },
    index=pd.date_range(start='2000-01-01', periods=50, freq='MS')
)
exog = pd.DataFrame(
    {
        "exog_1": np.random.rand(50),
        "exog_2": np.random.rand(50)
    },
    index=pd.date_range(start='2000-01-01', periods=50, freq='MS')
)

model = create_and_compile_model(
    series=series,
    exog=exog,
    lags=3,
    steps=5,
    levels=["l1", "l2"],
    recurrent_units=64,
    dense_units=32,
)

model_no_exog = create_and_compile_model(
    series=series,
    lags=3,
    steps=5,
    levels=["l1", "l2"],
    recurrent_units=10,
    dense_units=5,
)


def test_set_out_sample_residuals_NotFittedError_when_forecaster_not_fitted():
    """
    Test NotFittedError is raised when forecaster is not fitted.
    """
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([1, 2, 3, 4, 5])}
    y_pred = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        "This forecaster is not fitted yet. Call `fit` with appropriate "
        "arguments before using `set_out_sample_residuals()`."
    )
    with pytest.raises(NotFittedError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_true_is_not_dict():
    """
    Test TypeError is raised when y_true is not a dict.
    """
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    forecaster.is_fitted = True
    y_true = 'not_dict'
    y_pred = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        f"`y_true` must be a dictionary of numpy ndarrays or pandas Series. "
        f"Got {type(y_true)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_pred_is_not_dict():
    """
    Test TypeError is raised when y_pred is not a dict.
    """
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([1, 2, 3, 4, 5])}
    y_pred = 'not_dict'

    err_msg = re.escape(
        f"`y_pred` must be a dictionary of numpy ndarrays or pandas Series. "
        f"Got {type(y_pred)}."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_pred_and_y_true_keys_do_not_match():
    """
    Test TypeError is raised when y_pred and y_true keys do not match.
    """
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([1, 2, 3, 4, 5])}
    y_pred = {'l3': np.array([1, 2, 3, 4, 5]), 'l4': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        f"`y_true` and `y_pred` must have the same keys. "
        f"Got {set(y_true.keys())} and {set(y_pred.keys())}."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_true_contains_no_numpy_ndarrays_or_pandas_series():
    """
    Test TypeError is raised when y_true contains no numpy ndarrays or pandas series.
    """
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': 'not_ndarray'}
    y_pred = {'l1': np.array([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        f"Values of `y_true` must be numpy ndarrays or pandas Series. "
        f"Got {type(y_true['l1'])} for series l1."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_TypeError_when_y_pred_contains_no_numpy_ndarrays_or_pandas_series():
    """
    Test TypeError is raised when y_pred contains no numpy ndarrays or pandas series.
    """
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': np.array([1, 2, 3, 4, 5])}
    y_pred = {'l1': 'not_ndarray'}

    err_msg = re.escape(
        f"Values of `y_pred` must be numpy ndarrays or pandas Series. "
        f"Got {type(y_pred['l1'])} for series l1."
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_elements_with_different_lengths():
    """
    Test ValueError is raised when y_true and y_pred have elements with different lengths.
    """
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([1, 2, 3, 4, 5])}
    y_pred = {'l1': np.array([1, 2, 3, 4, 5]), 'l2': np.array([1, 2])}

    err_msg = re.escape(
        '`y_true` and `y_pred` must have the same length. Got 5 and 2 for series l2.'
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_y_true_and_y_pred_have_series_with_different_indexes():
    """
    Test ValueError is raised when y_true and y_pred have series with different indexes.
    """
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    forecaster.is_fitted = True
    y_true = {'l1': pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])}
    y_pred = {'l1': pd.Series([1, 2, 3, 4, 5])}

    err_msg = re.escape(
        "When containing pandas Series, elements in `y_true` and "
        "`y_pred` must have the same index."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_ValueError_when_inputs_does_not_match_the_target_levels():
    """
    Test ValueError is raised when inputs does not contain keys that match any level.
    """
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    forecaster.fit(series=series)
    y_true = {'l3': np.array([1, 2, 3])}
    y_pred = {'l3': np.array([1, 2, 3])}

    err_msg = re.escape(
        f"Provided keys in `y_pred` and `y_true` do not match any of the "
        f"target time series in the forecaster, {forecaster.levels}. Residuals "
        f"cannot be updated."
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_no_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and 
    append is False.
    """
    rng = np.random.default_rng(123)
    y_true = {
        'l1': pd.Series(rng.normal(loc=10, scale=10, size=1000)), 
        'l2': pd.Series(rng.normal(loc=10, scale=10, size=1000))
    }
    y_pred = {
        'l1': pd.Series(rng.normal(loc=10, scale=10, size=1000)), 
        'l2': pd.Series(rng.normal(loc=10, scale=10, size=1000))
    }

    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3, transformer_series=None
    )
    forecaster.fit(series=series)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=False)
    results = {
        'l1': np.sort(forecaster.out_sample_residuals_['l1']),
        'l2': np.sort(forecaster.out_sample_residuals_['l2'])
    }

    expected = {
        'l1': np.sort(y_true['l1'] - y_pred['l1']),
        'l2': np.sort(y_true['l2'] - y_pred['l2']),
    }

    assert forecaster.out_sample_residuals_.keys() == expected.keys()
    for key in results.keys():
        np.testing.assert_array_almost_equal(expected[key], results[key])


def test_set_out_sample_residuals_when_residuals_length_is_less_than_10000_and_append():
    """
    Test residuals stored when new residuals length is less than 10_000 and 
    append is True.
    """
    rng = np.random.default_rng(123)
    y_true = {
        'l1': pd.Series(rng.normal(loc=10, scale=10, size=1000)),
        'l2': pd.Series(rng.normal(loc=10, scale=10, size=1000))
    }
    y_pred = {
        'l1': pd.Series(rng.normal(loc=10, scale=10, size=1000)),
        'l2': pd.Series(rng.normal(loc=10, scale=10, size=1000))
    }
    
    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3, transformer_series=None
    )
    forecaster.fit(series=series)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred, append=True)
    results = {
        'l1': np.sort(forecaster.out_sample_residuals_['l1']),
        'l2': np.sort(forecaster.out_sample_residuals_['l2'])
    }

    residuals_1 = (y_true['l1'] - y_pred['l1'])
    residuals_2 = (y_true['l2'] - y_pred['l2'])
    expected = {
        'l1': np.sort(np.concatenate((residuals_1, residuals_1))),
        'l2': np.sort(np.concatenate((residuals_2, residuals_2)))
    }

    assert forecaster.out_sample_residuals_.keys() == expected.keys()
    for key in results.keys():
        np.testing.assert_array_almost_equal(expected[key], results[key])


def test_set_out_sample_residuals_when_residuals_length_is_greater_than_10000():
    """
    Test length residuals stored when its length is greater than 10_000.
    """
    rng = np.random.default_rng(123)
    y_true = {
        'l1': pd.Series(rng.normal(loc=10, scale=10, size=50_000)),
        'l2': pd.Series(rng.normal(loc=10, scale=10, size=50_000))
    }
    y_pred = {
        'l1': pd.Series(rng.normal(loc=10, scale=10, size=50_000)),
        'l2': pd.Series(rng.normal(loc=10, scale=10, size=50_000))
    }

    forecaster = ForecasterRnn(
        estimator=model_no_exog, levels=["l1", "l2"], lags=3
    )
    forecaster.fit(series=series)
    forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)

    assert list(forecaster.out_sample_residuals_.keys()) == ['l1', 'l2']
    for key in forecaster.out_sample_residuals_.keys():
        assert len(forecaster.out_sample_residuals_[key]) == 10_000


def test_out_sample_residuals_and_in_sample_residuals_equivalence():
    """
    Test out sample residuals are equivalent to in-sample residuals 
    when training data and training predictions are passed.
    """
    forecaster = ForecasterRnn(
        estimator=model, levels=["l1", "l2"], lags=3
    )
    forecaster.fit(series=series, exog=exog, store_in_sample_residuals=True)
    X_train, exog_train, y_train, _ = forecaster.create_train_X_y(series=series, exog=exog)
    y_pred_train = forecaster.estimator.predict(
        x=X_train if exog_train is None else [X_train, exog_train], verbose=0
    )

    y_true = []
    y_pred = []
    for i, step in enumerate(forecaster.steps):
        y_true.append(y_train[:, i, :])
        y_pred.append(y_pred_train[:, i, :])
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_true_dict = {}
    y_pred_dict = {}
    for i, level in enumerate(forecaster.levels):
        y_true_dict[level] = forecaster.transformer_series_[level].inverse_transform(
            y_true[:, i].reshape(-1, 1)
        ).ravel()
        y_pred_dict[level] = forecaster.transformer_series_[level].inverse_transform(
            y_pred[:, i].reshape(-1, 1)
        ).ravel()

    forecaster.set_out_sample_residuals(y_true=y_true_dict, y_pred=y_pred_dict)

    assert forecaster.in_sample_residuals_.keys() == forecaster.out_sample_residuals_.keys()
    for k in forecaster.in_sample_residuals_.keys():
        np.testing.assert_array_almost_equal(
            forecaster.in_sample_residuals_[k], forecaster.out_sample_residuals_[k]
        )
