# Unit test save_forecaster and load_forecaster
# ==============================================================================
import os
import re
import joblib
import pickle
import pytest
import inspect
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .... import __version__
from ....recursive import ForecasterRecursive
from ....recursive import ForecasterRecursiveMultiSeries
from ....recursive import ForecasterStats
from ....stats import Arima
from ...utils import save_forecaster
from ...utils import load_forecaster
from ....exceptions import SkforecastVersionWarning, SaveLoadSkforecastWarning


def custom_weights(y):  # pragma: no cover
    """
    """
    return np.ones(len(y))


def custom_weights2(y):  # pragma: no cover
    """
    """
    return np.arange(1, len(y) + 1)


class UserWindowFeature:  # pragma: no cover
    def __init__(self, window_sizes, features_names):
        self.window_sizes = window_sizes
        self.features_names = features_names

    def transform_batch(self):
        pass

    def transform(self):
        pass


def test_save_and_load_forecaster_persistence():
    """ 
    Test if a loaded forecaster is exactly the same as the original one.
    """
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, transformer_y=StandardScaler()
    )
    rng = np.random.default_rng(12345)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)
    save_forecaster(forecaster=forecaster, file_name='forecaster.joblib', verbose=True)
    forecaster_loaded = load_forecaster(file_name='forecaster.joblib', verbose=True)
    os.remove('forecaster.joblib')

    for key in vars(forecaster).keys():
    
        attribute_forecaster = forecaster.__getattribute__(key)
        attribute_forecaster_loaded = forecaster_loaded.__getattribute__(key)

        if key in ['estimator', 'binner', 'transformer_y', 'transformer_exog', 'categorical_encoder']:
            assert joblib.hash(attribute_forecaster) == joblib.hash(attribute_forecaster_loaded)
        elif isinstance(attribute_forecaster, np.ndarray):
            np.testing.assert_array_almost_equal(attribute_forecaster, attribute_forecaster_loaded)
        elif isinstance(attribute_forecaster, pd.Series):
            pd.testing.assert_series_equal(attribute_forecaster, attribute_forecaster_loaded)
        elif isinstance(attribute_forecaster, pd.DataFrame):
            pd.testing.assert_frame_equal(attribute_forecaster, attribute_forecaster_loaded)
        elif isinstance(attribute_forecaster, pd.Index):
            pd.testing.assert_index_equal(attribute_forecaster, attribute_forecaster_loaded)
        elif isinstance(attribute_forecaster, dict):
            assert attribute_forecaster.keys() == attribute_forecaster_loaded.keys()
            for k in attribute_forecaster.keys():
                if isinstance(attribute_forecaster[k], np.ndarray):
                    np.testing.assert_array_almost_equal(attribute_forecaster[k], attribute_forecaster_loaded[k])
                elif isinstance(attribute_forecaster[k], pd.Series):
                    pd.testing.assert_series_equal(attribute_forecaster[k], attribute_forecaster_loaded[k])
                elif isinstance(attribute_forecaster[k], pd.DataFrame):
                    pd.testing.assert_frame_equal(attribute_forecaster[k], attribute_forecaster_loaded[k])
                elif isinstance(attribute_forecaster[k], pd.Index):
                    pd.testing.assert_index_equal(attribute_forecaster[k], attribute_forecaster_loaded[k])
                else:
                    assert attribute_forecaster[k] == attribute_forecaster_loaded[k]
        else:
            assert attribute_forecaster == attribute_forecaster_loaded


def test_save_and_load_forecaster_SkforecastVersionWarning():
    """ 
    Test warning used to notify that the skforecast version installed in the 
    environment differs from the version used to create the forecaster.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)
    forecaster.skforecast_version = '0.0.0'
    save_forecaster(forecaster=forecaster, file_name='forecaster.joblib', verbose=False)

    warn_msg = re.escape(
        f"The skforecast version installed in the environment differs "
        f"from the version used to create the forecaster.\n"
        f"    Installed Version  : {__version__}\n"
        f"    Forecaster Version : 0.0.0\n"
        f"This may create incompatibilities when using the library."
    )
    with pytest.warns(SkforecastVersionWarning, match = warn_msg):
        load_forecaster(file_name='forecaster.joblib', verbose=False)
        os.remove('forecaster.joblib')


def _simulate_main_namespace(monkeypatch, funcs):
    """
    Make `funcs` look as if they were defined in the '__main__' namespace
    (e.g. a notebook), so save_forecaster treats them as needing export, while
    keeping them findable by joblib/pickle within the test process.
    """
    import __main__
    for fun in set(funcs):
        monkeypatch.setattr(__main__, fun.__name__, fun, raising=False)
        monkeypatch.setattr(fun, '__module__', '__main__')


@pytest.mark.parametrize("weight_func",
                         [custom_weights,
                          {'serie_1': custom_weights,
                           'serie_2': custom_weights2},
                          {'serie_1': custom_weights}],
                         ids = lambda func: f'type: {type(func)}')
def test_save_forecaster_save_custom_functions(weight_func, monkeypatch):
    """
    Test custom functions defined in '__main__' are saved as .py files.
    """
    series = pd.DataFrame(
        {'serie_1': np.random.normal(size=20),
         'serie_2': np.random.normal(size=20)}
    ).to_dict(orient='series')

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     weight_func        = weight_func,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)

    weight_functions = (
        list(weight_func.values()) if isinstance(weight_func, dict) else [weight_func]
    )
    _simulate_main_namespace(monkeypatch, weight_functions)

    warn_msg = re.escape(
        "Custom function(s) used to create weights are defined in the "
        "'__main__' namespace and have been saved as:"
    )
    with pytest.warns(SaveLoadSkforecastWarning, match = warn_msg):
        save_forecaster(
            forecaster=forecaster, file_name='forecaster.joblib', save_custom_functions=True
        )
    load_forecaster(file_name='forecaster.joblib', verbose=True)
    os.remove('forecaster.joblib')

    for fun in set(weight_functions):
        weight_func_file = fun.__name__ + '.py'
        assert os.path.exists(weight_func_file)
        with open(weight_func_file, 'r') as file:
            assert inspect.getsource(fun) == file.read()
        os.remove(weight_func_file)


@pytest.mark.parametrize("weight_func",
                         [custom_weights,
                          {'serie_1': custom_weights,
                           'serie_2': custom_weights2}],
                         ids = lambda func: f'func: {func}')
def test_save_forecaster_warning_dont_save_custom_functions(weight_func, monkeypatch):
    """
    Test SaveLoadSkforecastWarning when '__main__' custom functions are not saved.
    """
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator   = LinearRegression(),
                     lags        = 5,
                     weight_func = weight_func
                 )

    weight_functions = (
        list(weight_func.values()) if isinstance(weight_func, dict) else [weight_func]
    )
    _simulate_main_namespace(monkeypatch, weight_functions)

    warn_msg = re.escape(
        "Custom function(s) used to create weights are defined in the "
        "'__main__' namespace and have not been saved. To save them "
        "automatically, set `save_custom_functions=True`."
    )
    with pytest.warns(SaveLoadSkforecastWarning, match = warn_msg):
        save_forecaster(
            forecaster=forecaster, file_name='forecaster.joblib', save_custom_functions=False
        )
        os.remove('forecaster.joblib')


@pytest.mark.parametrize("save_custom_functions",
                         [True, False],
                         ids = lambda v: f'save_custom_functions: {v}')
def test_save_forecaster_module_weight_func_no_py_no_warning(save_custom_functions):
    """
    Test that a weight_func imported from a module is not exported as a .py file
    and raises no SaveLoadSkforecastWarning (joblib/pickle restore it by reference).
    """
    series = pd.DataFrame(
        {'serie_1': np.random.normal(size=20),
         'serie_2': np.random.normal(size=20)}
    ).to_dict(orient='series')

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator   = LinearRegression(),
                     lags        = 5,
                     weight_func = custom_weights
                 )
    forecaster.fit(series=series)

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter('error', SaveLoadSkforecastWarning)
        save_forecaster(
            forecaster=forecaster,
            file_name='forecaster.joblib',
            save_custom_functions=save_custom_functions,
        )

    assert not os.path.exists('custom_weights.py')
    os.remove('forecaster.joblib')


def test_save_load_forecaster_module_weight_func_round_trip():
    """
    Test that a weight_func imported from a module round-trips without a manual
    import, returning the same module-level function object.
    """
    series = pd.DataFrame(
        {'serie_1': np.random.normal(size=20),
         'serie_2': np.random.normal(size=20)}
    ).to_dict(orient='series')

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator   = LinearRegression(),
                     lags        = 5,
                     weight_func = custom_weights
                 )
    forecaster.fit(series=series)

    save_forecaster(forecaster=forecaster, file_name='forecaster.joblib')
    forecaster_loaded = load_forecaster(file_name='forecaster.joblib', verbose=False)
    os.remove('forecaster.joblib')

    assert forecaster_loaded.weight_func is custom_weights


def test_save_forecaster_warning_when_user_defined_window_features():
    """
    Test SaveLoadSkforecastWarning when user-defined window features.
    """

    window_features = UserWindowFeature(
        window_sizes=[1, 2], features_names=['feature_1', 'feature_2']
    )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator       = LinearRegression(),
                     lags            = 5,
                     window_features = window_features
                 )

    warn_msg = re.escape(
        "The Forecaster includes custom user-defined classes in the "
        "`window_features` argument. These classes are not saved automatically "
        "when saving the Forecaster. Please ensure you save these classes "
        "manually and import them before loading the Forecaster.\n"
        "    Custom classes: " + ', '.join({'UserWindowFeature'}),
    )
    with pytest.warns(SaveLoadSkforecastWarning, match = warn_msg):
        save_forecaster(
            forecaster=forecaster, file_name='forecaster.joblib', save_custom_functions=False
        )
        os.remove('forecaster.joblib')


def test_save_forecaster_ValueError_when_invalid_backend():
    """
    Test ValueError when an invalid backend is passed to save_forecaster.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    err_msg = re.escape(
        "Invalid `backend` argument: 'invalid_backend'. "
        "Valid options are: 'cloudpickle', 'joblib', 'pickle'."
    )
    with pytest.raises(ValueError, match=err_msg):
        save_forecaster(
            forecaster=forecaster,
            file_name='forecaster',
            backend='invalid_backend',
            verbose=False,
        )


def test_load_forecaster_ValueError_when_invalid_backend():
    """
    Test ValueError when an invalid backend is passed to load_forecaster.
    """
    err_msg = re.escape(
        "Invalid `backend` argument: 'invalid_backend'. "
        "Valid options are: 'cloudpickle', 'joblib', 'pickle'."
    )
    with pytest.raises(ValueError, match=err_msg):
        load_forecaster(file_name='forecaster.joblib', backend='invalid_backend')


def test_load_forecaster_ValueError_when_unrecognized_extension():
    """
    Test ValueError when load_forecaster cannot infer the backend from an
    unrecognized file extension.
    """
    err_msg = re.escape(
        "Cannot infer backend from file extension '.xyz'. "
        "Recognized extensions: '.cloudpickle', '.joblib', '.pickle', '.pkl'. "
        "Provide the `backend` argument explicitly."
    )
    with pytest.raises(ValueError, match=err_msg):
        load_forecaster(file_name='forecaster.xyz')


def test_load_forecaster_backend_auto_detection():
    """
    Test that load_forecaster correctly infers the backend from the file
    extension when `backend=None`.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    rng = np.random.default_rng(123)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)

    backends = [('joblib', '.joblib'), ('pickle', '.pkl'), ('cloudpickle', '.cloudpickle')]
    for backend, extension in backends:
        save_forecaster(
            forecaster=forecaster,
            file_name='forecaster_autodetect',
            backend=backend,
            verbose=False,
        )
        file_path = 'forecaster_autodetect' + extension
        forecaster_loaded = load_forecaster(file_name=file_path, backend=None, verbose=False)
        os.remove(file_path)
        assert forecaster_loaded.skforecast_version == forecaster.skforecast_version


def _assert_attribute_equal(a, b):
    """
    Recursively assert that two forecaster attributes are equal, dispatching on
    type (numpy ndarray, pandas Series, DataFrame, Index, dict, or scalar).
    """
    if isinstance(a, np.ndarray):
        np.testing.assert_array_almost_equal(a, b)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b)
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, pd.Index):
        pd.testing.assert_index_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a.keys():
            _assert_attribute_equal(a[k], b[k])
    else:
        assert a == b


def _build_fitted_forecaster_recursive():
    """
    Build and fit a ForecasterRecursive (DataFrame `last_window_`, Index
    `training_range_`).
    """
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, transformer_y=StandardScaler()
    )
    rng = np.random.default_rng(12345)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)

    return forecaster


def _build_fitted_forecaster_multiseries():
    """
    Build and fit a ForecasterRecursiveMultiSeries (dict `last_window_`, dict
    `training_range_`).
    """
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=LinearRegression(), lags=3, transformer_series=StandardScaler()
    )
    rng = np.random.default_rng(12345)
    series = pd.DataFrame(
        {'serie_1': rng.normal(size=100), 'serie_2': rng.normal(size=100)}
    )
    forecaster.fit(series=series)

    return forecaster


def _build_fitted_forecaster_stats():
    """
    Build and fit a ForecasterStats (Series `last_window_`, DatetimeIndex
    `training_range_`).
    """
    forecaster = ForecasterStats(estimator=Arima(order=(1, 0, 0)))
    rng = np.random.default_rng(12345)
    idx = pd.date_range('2020-01-01', periods=60, freq='MS')
    y = pd.Series(rng.normal(size=60), index=idx)
    forecaster.fit(y=y)

    return forecaster


@pytest.mark.parametrize(
    "build_forecaster",
    [
        _build_fitted_forecaster_recursive,
        _build_fitted_forecaster_multiseries,
        _build_fitted_forecaster_stats,
    ],
    ids=['ForecasterRecursive', 'ForecasterRecursiveMultiSeries', 'ForecasterStats']
)
@pytest.mark.parametrize(
    "backend, extension",
    [('pickle', '.pkl'), ('cloudpickle', '.cloudpickle')],
    ids=['pickle', 'cloudpickle']
)
def test_save_and_load_forecaster_round_trip(backend, extension, build_forecaster):
    """
    Test that forecasters of different types round-trip through the pickle and
    cloudpickle backends. Covers the `last_window_` shapes (DataFrame for single
    series, dict for multi-series, Series for ForecasterStats) and the
    `training_range_` shapes (Index and dict of Index) that the future skops
    backend must decompose and reconstruct, plus functional equivalence of the
    predictions. Deep attribute equality for the default joblib backend is
    covered by `test_save_and_load_forecaster_persistence`.
    """
    forecaster = build_forecaster()
    predictions = forecaster.predict(steps=5)

    file_base = f'forecaster_round_trip_{backend}_{type(forecaster).__name__}'
    save_forecaster(
        forecaster=forecaster, file_name=file_base, backend=backend, verbose=False
    )
    expected_file = file_base + extension
    assert os.path.exists(expected_file)

    forecaster_loaded = load_forecaster(
        file_name=expected_file, backend=backend, verbose=False
    )
    os.remove(expected_file)

    # Functional equivalence: the loaded forecaster predicts identically.
    _assert_attribute_equal(predictions, forecaster_loaded.predict(steps=5))
    # Shape-bearing attributes that the future skops backend must reconstruct.
    _assert_attribute_equal(forecaster.last_window_, forecaster_loaded.last_window_)
    _assert_attribute_equal(forecaster.training_range_, forecaster_loaded.training_range_)


@pytest.mark.parametrize("weight_func",
                         [custom_weights,
                          {'serie_1': custom_weights, 'serie_2': custom_weights2}],
                         ids=lambda func: f'weight_func: {type(func)}')
def test_save_forecaster_cloudpickle_no_py_file_for_weight_func(weight_func):
    """
    Test that cloudpickle embeds custom weight functions in the file and does
    not create separate .py files nor raise a SaveLoadSkforecastWarning.
    """
    series = pd.DataFrame(
        {'serie_1': np.random.normal(size=20),
         'serie_2': np.random.normal(size=20)}
    ).to_dict(orient='series')

    forecaster = ForecasterRecursiveMultiSeries(
                     estimator          = LinearRegression(),
                     lags               = 5,
                     weight_func        = weight_func,
                     transformer_series = StandardScaler()
                 )
    forecaster.fit(series=series)

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter('error', SaveLoadSkforecastWarning)
        save_forecaster(
            forecaster=forecaster,
            file_name='forecaster_cloudpickle_wf',
            backend='cloudpickle',
            save_custom_functions=False,
            verbose=False,
        )

    expected_file = 'forecaster_cloudpickle_wf.cloudpickle'
    assert os.path.exists(expected_file)
    os.remove(expected_file)

    weight_functions = weight_func.values() if isinstance(weight_func, dict) else [weight_func]
    for func in weight_functions:
        py_file = func.__name__ + '.py'
        assert not os.path.exists(py_file)


def test_save_forecaster_cloudpickle_no_warning_for_window_features():
    """
    Test that cloudpickle does not raise SaveLoadSkforecastWarning for custom
    user-defined window features classes.
    """
    window_features = UserWindowFeature(
        window_sizes=[1, 2], features_names=['feature_1', 'feature_2']
    )
    forecaster = ForecasterRecursiveMultiSeries(
                     estimator       = LinearRegression(),
                     lags            = 5,
                     window_features = window_features
                 )

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter('error', SaveLoadSkforecastWarning)
        save_forecaster(
            forecaster=forecaster,
            file_name='forecaster_cloudpickle_wf2',
            backend='cloudpickle',
            verbose=False,
        )

    expected_file = 'forecaster_cloudpickle_wf2.cloudpickle'
    assert os.path.exists(expected_file)
    os.remove(expected_file)


def test_save_and_load_forecaster_cloudpickle_embeds_local_weight_func():
    """
    Test that cloudpickle embeds by value a custom weight function defined in a
    local scope (not importable by reference), so the loaded forecaster stays
    functional. The pickle backend cannot serialize such a function.
    """
    def local_weights(index):
        return np.ones(len(index))

    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, weight_func=local_weights
    )
    rng = np.random.default_rng(12345)
    y = pd.Series(rng.normal(size=50))
    forecaster.fit(y=y)
    predictions_before = forecaster.predict(steps=5)

    # The pickle backend cannot serialize a locally-defined function
    with pytest.raises((pickle.PicklingError, AttributeError)):
        save_forecaster(
            forecaster=forecaster,
            file_name='forecaster_local_wf',
            backend='pickle',
            verbose=False,
        )
    if os.path.exists('forecaster_local_wf.pkl'):
        os.remove('forecaster_local_wf.pkl')

    # cloudpickle embeds the function by value
    save_forecaster(
        forecaster=forecaster,
        file_name='forecaster_local_wf',
        backend='cloudpickle',
        verbose=False,
    )
    forecaster_loaded = load_forecaster(
        file_name='forecaster_local_wf.cloudpickle', verbose=False
    )
    os.remove('forecaster_local_wf.cloudpickle')

    assert callable(forecaster_loaded.weight_func)
    np.testing.assert_array_equal(
        forecaster_loaded.weight_func(y.index), np.ones(len(y))
    )
    pd.testing.assert_series_equal(
        predictions_before, forecaster_loaded.predict(steps=5)
    )
