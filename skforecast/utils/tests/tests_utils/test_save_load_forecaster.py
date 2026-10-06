# Unit test save_forecaster and load_forecaster
# ==============================================================================
import os
import re
import zoneinfo
import joblib
import pickle
import pytest
import inspect
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import skops.io
from skops.io.exceptions import UntrustedTypesFoundException

from .... import __version__
from ....recursive import ForecasterRecursive
from ....recursive import ForecasterRecursiveMultiSeries
from ....recursive import ForecasterRecursiveClassifier
from ....recursive import ForecasterStats
from ....recursive import ForecasterEquivalentDate
from ....direct import ForecasterDirect
from ....direct import ForecasterDirectMultiVariate
from ....stats import Arima
from ....preprocessing import RollingFeatures
from ....preprocessing import RollingFeaturesClassification
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
                         ids = ['func: function', 'func: dict of functions'])
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

    with warnings.catch_warnings():
        warnings.simplefilter('error', SaveLoadSkforecastWarning)
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


@pytest.mark.parametrize(
    "forecaster",
    [
        ForecasterRecursive(
            estimator=LinearRegression(),
            lags=3,
            window_features=RollingFeatures(stats=['mean'], window_sizes=3)
        ),
        ForecasterRecursiveClassifier(
            estimator=LogisticRegression(),
            lags=3,
            window_features=RollingFeaturesClassification(
                stats=['proportion'], window_sizes=3
            )
        ),
    ],
    ids=['RollingFeatures', 'RollingFeaturesClassification']
)
def test_save_forecaster_no_warning_when_skforecast_window_features(
    forecaster, tmp_path
):
    """
    Test that no SaveLoadSkforecastWarning is raised when the window features
    are skforecast classes, since they do not need to be saved by the user.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', SaveLoadSkforecastWarning)
        save_forecaster(
            forecaster=forecaster, file_name=str(tmp_path / 'forecaster.joblib')
        )


def test_save_forecaster_ValueError_when_invalid_backend():
    """
    Test ValueError when an invalid backend is passed to save_forecaster.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    err_msg = re.escape(
        "Invalid `backend` argument: 'invalid_backend'. "
        "Valid options are: 'cloudpickle', 'joblib', 'pickle', 'skops'."
    )
    with pytest.raises(ValueError, match=err_msg):
        save_forecaster(
            forecaster=forecaster,
            file_name='forecaster',
            backend='invalid_backend',
            verbose=False,
        )


@pytest.mark.parametrize(
    "file_name, backend, expected_file",
    [
        ('model', 'joblib', 'model.joblib'),
        ('model.joblib', 'joblib', 'model.joblib'),
        ('model.pkl', 'joblib', 'model.joblib'),
        ('model.PKL', 'pickle', 'model.pkl'),
        ('model_v1.2', 'joblib', 'model_v1.2.joblib'),
        ('forecaster_2026.10.04', 'pickle', 'forecaster_2026.10.04.pkl'),
        ('model.bin', 'joblib', 'model.bin.joblib'),
    ],
    ids=[
        'no extension',
        'same extension',
        'other backend extension',
        'uppercase extension',
        'dotted name',
        'dotted date',
        'unknown extension',
    ]
)
def test_save_forecaster_file_name_extension(
    file_name, backend, expected_file, tmp_path
):
    """
    Test that save_forecaster adds the backend extension to the file name and
    only replaces the extension when it is a backend extension, so the dots in
    the name are kept. The saved file loads with the inferred backend.
    """
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=pd.Series(np.arange(20, dtype=float)))
    save_forecaster(
        forecaster=forecaster, file_name=str(tmp_path / file_name), backend=backend
    )
    forecaster_loaded = load_forecaster(
        file_name=str(tmp_path / expected_file), verbose=False
    )

    assert os.listdir(tmp_path) == [expected_file]
    np.testing.assert_array_equal(forecaster_loaded.lags, forecaster.lags)


def test_load_forecaster_ValueError_when_invalid_backend():
    """
    Test ValueError when an invalid backend is passed to load_forecaster.
    """
    err_msg = re.escape(
        "Invalid `backend` argument: 'invalid_backend'. "
        "Valid options are: 'cloudpickle', 'joblib', 'pickle', 'skops'."
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
        "Recognized extensions: '.cloudpickle', '.joblib', '.pickle', '.pkl', "
        "'.skops'. Provide the `backend` argument explicitly."
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

    backends = [
        ('joblib', '.joblib'),
        ('pickle', '.pkl'),
        ('cloudpickle', '.cloudpickle'),
        ('skops', '.skops'),
    ]
    for backend, extension in backends:
        save_forecaster(
            forecaster=forecaster,
            file_name='forecaster_autodetect',
            backend=backend,
            verbose=False,
        )
        file_path = 'forecaster_autodetect' + extension
        forecaster_loaded = load_forecaster(
            file_name=file_path, backend=None, trusted=True, verbose=False
        )
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


def _build_fitted_forecaster_recursive_datetime():
    """
    Build and fit a ForecasterRecursive on a DatetimeIndex series (DataFrame
    `last_window_` with a DatetimeIndex, DatetimeIndex `training_range_`).
    """
    forecaster = ForecasterRecursive(
        estimator=LinearRegression(), lags=3, transformer_y=StandardScaler()
    )
    rng = np.random.default_rng(12345)
    idx = pd.date_range('2020-01-01', periods=100, freq='D')
    y = pd.Series(rng.normal(size=100), index=idx)
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


def _build_fitted_forecaster_direct():
    """
    Build and fit a ForecasterDirect (DataFrame `last_window_`, Index
    `training_range_`).
    """
    forecaster = ForecasterDirect(
        estimator=LinearRegression(), steps=5, lags=3, transformer_y=StandardScaler()
    )
    rng = np.random.default_rng(12345)
    y = pd.Series(rng.normal(size=100))
    forecaster.fit(y=y)

    return forecaster


def _build_fitted_forecaster_direct_multivariate():
    """
    Build and fit a ForecasterDirectMultiVariate (DataFrame `last_window_`,
    Index `training_range_`).
    """
    forecaster = ForecasterDirectMultiVariate(
        estimator=LinearRegression(),
        level='serie_1',
        steps=5,
        lags=3,
        transformer_series=StandardScaler(),
    )
    rng = np.random.default_rng(12345)
    series = pd.DataFrame(
        {'serie_1': rng.normal(size=100), 'serie_2': rng.normal(size=100)}
    )
    forecaster.fit(series=series)

    return forecaster


def _build_fitted_forecaster_classifier():
    """
    Build and fit a ForecasterRecursiveClassifier (DataFrame `last_window_`,
    Index `training_range_`).
    """
    forecaster = ForecasterRecursiveClassifier(estimator=LogisticRegression(), lags=3)
    rng = np.random.default_rng(12345)
    y = pd.Series(rng.choice(['a', 'b', 'c'], size=100))
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
    `training_range_` shapes (Index and dict of Index), plus functional
    equivalence of the predictions. Deep attribute equality for the default
    joblib backend is covered by `test_save_and_load_forecaster_persistence`.
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
    # Shape-bearing attributes (`last_window_`, `training_range_`).
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

    with warnings.catch_warnings():
        warnings.simplefilter('error', SaveLoadSkforecastWarning)
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

    with warnings.catch_warnings():
        warnings.simplefilter('error', SaveLoadSkforecastWarning)
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


@pytest.mark.parametrize(
    "build_forecaster",
    [
        _build_fitted_forecaster_recursive,
        _build_fitted_forecaster_recursive_datetime,
        _build_fitted_forecaster_multiseries,
        _build_fitted_forecaster_direct,
        _build_fitted_forecaster_direct_multivariate,
        _build_fitted_forecaster_classifier,
    ],
    ids=[
        'ForecasterRecursive',
        'ForecasterRecursive_datetime',
        'ForecasterRecursiveMultiSeries',
        'ForecasterDirect',
        'ForecasterDirectMultiVariate',
        'ForecasterRecursiveClassifier',
    ]
)
def test_save_and_load_forecaster_round_trip_skops(build_forecaster):
    """
    Test that forecasters with a scikit-learn estimator round-trip through the
    skops backend. Covers the `last_window_`/`training_range_` shapes (DataFrame
    and Index for single series, dict of each for multi-series) that skops must
    decompose and reconstruct, functional equivalence of the predictions, and
    that `save_forecaster` does not mutate the in-memory forecaster.
    """
    forecaster = build_forecaster()
    predictions = forecaster.predict(steps=5)
    last_window_before = forecaster.last_window_
    training_range_before = forecaster.training_range_

    file_base = f'forecaster_round_trip_skops_{type(forecaster).__name__}'
    save_forecaster(
        forecaster=forecaster, file_name=file_base, backend='skops', verbose=False
    )
    expected_file = file_base + '.skops'
    assert os.path.exists(expected_file)

    # save_forecaster must not mutate the in-memory forecaster: skops
    # serializes a decomposed copy.
    assert forecaster.last_window_ is last_window_before
    assert forecaster.training_range_ is training_range_before

    forecaster_loaded = load_forecaster(
        file_name=expected_file, backend='skops', trusted=True, verbose=False
    )
    os.remove(expected_file)

    # Functional equivalence: the loaded forecaster predicts identically.
    _assert_attribute_equal(predictions, forecaster_loaded.predict(steps=5))
    # Shape-bearing attributes that the skops backend reconstructs.
    _assert_attribute_equal(forecaster.last_window_, forecaster_loaded.last_window_)
    _assert_attribute_equal(forecaster.training_range_, forecaster_loaded.training_range_)


@pytest.mark.parametrize(
    "forecaster, index",
    [
        (
            ForecasterRecursive(
                estimator=LinearRegression(),
                lags=3,
                window_features=RollingFeatures(stats=['mean', 'std'], window_sizes=4),
            ),
            pd.date_range('2020-01-01', periods=100, freq='D'),
        ),
        (
            ForecasterRecursive(
                estimator=LinearRegression(),
                lags=3,
                window_features=RollingFeatures(stats=['mean', 'std'], window_sizes=4),
            ),
            pd.RangeIndex(100),
        ),
        (
            ForecasterDirect(
                estimator=LinearRegression(),
                steps=5,
                lags=3,
                window_features=RollingFeatures(stats=['mean', 'std'], window_sizes=4),
            ),
            pd.date_range('2020-01-01', periods=100, freq='D'),
        ),
        (
            ForecasterRecursiveMultiSeries(
                estimator=LinearRegression(),
                lags=3,
                window_features=RollingFeatures(stats=['mean', 'std'], window_sizes=4),
            ),
            pd.date_range('2020-01-01', periods=100, freq='D'),
        ),
        (
            ForecasterDirectMultiVariate(
                estimator=LinearRegression(),
                level='serie_1',
                steps=5,
                lags=3,
                window_features=RollingFeatures(stats=['mean', 'std'], window_sizes=4),
            ),
            pd.date_range('2020-01-01', periods=100, freq='D'),
        ),
        (
            ForecasterRecursiveClassifier(
                estimator=LogisticRegression(),
                lags=3,
                window_features=RollingFeaturesClassification(
                    stats=['proportion', 'mode'], window_sizes=4
                ),
            ),
            pd.date_range('2020-01-01', periods=100, freq='D'),
        ),
    ],
    ids=[
        'ForecasterRecursive',
        'ForecasterRecursive_RangeIndex',
        'ForecasterDirect',
        'ForecasterRecursiveMultiSeries',
        'ForecasterDirectMultiVariate',
        'ForecasterRecursiveClassifier',
    ]
)
def test_save_and_load_forecaster_round_trip_skops_window_features(
    forecaster, index, tmp_path
):
    """
    Test that forecasters with window features round-trip through the skops
    backend. The window features must not keep the pandas Rolling objects of
    the training series, which skops cannot serialize.
    """
    rng = np.random.default_rng(12345)
    if isinstance(forecaster, ForecasterRecursiveClassifier):
        y = pd.Series(rng.choice(['a', 'b', 'c'], size=100), index=index)
        forecaster.fit(y=y)
    elif isinstance(
        forecaster, (ForecasterRecursiveMultiSeries, ForecasterDirectMultiVariate)
    ):
        series = pd.DataFrame(
            {'serie_1': rng.normal(size=100), 'serie_2': rng.normal(size=100)},
            index=index,
        )
        forecaster.fit(series=series)
    else:
        y = pd.Series(rng.normal(size=100), index=index)
        forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)

    file_name = str(tmp_path / 'forecaster.skops')
    save_forecaster(
        forecaster=forecaster, file_name=file_name, backend='skops', verbose=False
    )
    forecaster_loaded = load_forecaster(
        file_name=file_name, backend='skops', trusted=True, verbose=False
    )

    _assert_attribute_equal(predictions, forecaster_loaded.predict(steps=5))


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range('2024-03-25', periods=200, freq='h', tz='Europe/Madrid'),
        pd.date_range('2024-03-25', periods=140, freq='h', tz='Europe/Madrid'),
        pd.date_range(
            '2024-02-01', periods=30, freq='D', tz=zoneinfo.ZoneInfo('America/New_York')
        ),
        pd.date_range('2024-01-01', periods=100, freq='500ms'),
        pd.date_range(
            '2024-11-01',
            periods=36,
            freq=pd.offsets.CustomBusinessDay(holidays=['2024-12-25']),
        ),
    ],
    ids=[
        'dst_change_in_training',
        'dst_change_in_predictions',
        'zoneinfo_dst_change_in_predictions',
        'freq_500ms',
        'custom_business_day_with_holidays',
    ]
)
def test_save_and_load_forecaster_round_trip_skops_datetime_index(index, tmp_path):
    """
    Test that a forecaster trained on a DatetimeIndex round-trips through the
    skops backend and predicts the same values with the same index (time zone,
    daylight saving time changes and frequency).
    """
    rng = np.random.default_rng(12345)
    y = pd.Series(rng.normal(size=len(index)), index=index)
    forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=15)

    file_name = str(tmp_path / 'forecaster.skops')
    save_forecaster(
        forecaster=forecaster, file_name=file_name, backend='skops', verbose=False
    )
    forecaster_loaded = load_forecaster(
        file_name=file_name, backend='skops', trusted=True, verbose=False
    )

    pd.testing.assert_series_equal(predictions, forecaster_loaded.predict(steps=15))


@pytest.mark.parametrize(
    "exog_type",
    ['categorical', 'pyarrow'],
    ids=lambda exog_type: f'exog: {exog_type}'
)
@pytest.mark.parametrize(
    "forecaster_class",
    [ForecasterRecursive, ForecasterRecursiveMultiSeries],
    ids=lambda forecaster_class: forecaster_class.__name__
)
def test_save_and_load_forecaster_round_trip_skops_exog_dtypes(
    forecaster_class, exog_type, tmp_path
):
    """
    Test that a forecaster trained with categorical exog (categories of int32
    and str, all of them seen in training) or pyarrow exog round-trips through
    the skops backend, keeps the same exog dtypes and predicts the same values.
    """
    rng = np.random.default_rng(12345)
    index = pd.date_range('2020-01-01', periods=65, freq='D')
    if exog_type == 'categorical':
        exog = pd.DataFrame(
            {
                'day_of_week': pd.Categorical(index.day_of_week),
                'day_name': pd.Categorical(index.day_name()),
            },
            index=index,
        )
    else:
        exog = pd.DataFrame(
            {'exog_1': rng.normal(size=65)}, index=index, dtype='double[pyarrow]'
        )
    exog_train = exog.iloc[:60]
    exog_predict = exog.iloc[60:]
    y = pd.Series(rng.normal(size=60), index=index[:60])

    forecaster = forecaster_class(estimator=LinearRegression(), lags=3)
    if isinstance(forecaster, ForecasterRecursiveMultiSeries):
        series = {'serie_1': y, 'serie_2': y * 2}
        forecaster.fit(
            series=series, exog={'serie_1': exog_train, 'serie_2': exog_train}
        )
        exog_predict = {'serie_1': exog_predict, 'serie_2': exog_predict}
    else:
        forecaster.fit(y=y, exog=exog_train)
    predictions = forecaster.predict(steps=5, exog=exog_predict)

    file_name = str(tmp_path / 'forecaster.skops')
    save_forecaster(
        forecaster=forecaster, file_name=file_name, backend='skops', verbose=False
    )
    forecaster_loaded = load_forecaster(
        file_name=file_name, backend='skops', trusted=True, verbose=False
    )

    assert not predictions.isna().to_numpy().any()
    assert forecaster_loaded.exog_dtypes_in_ == forecaster.exog_dtypes_in_
    assert forecaster_loaded.exog_dtypes_out_ == forecaster.exog_dtypes_out_
    _assert_attribute_equal(
        predictions, forecaster_loaded.predict(steps=5, exog=exog_predict)
    )


@pytest.mark.parametrize(
    "forecaster, index",
    [
        (
            ForecasterEquivalentDate(offset=pd.DateOffset(days=7), n_offsets=2),
            pd.date_range('2020-01-01', periods=60, freq='D'),
        ),
        (
            ForecasterRecursive(estimator=LinearRegression(), lags=3),
            pd.date_range('2020-01-31', periods=48, freq=pd.DateOffset(months=1)),
        ),
    ],
    ids=['ForecasterEquivalentDate_offset', 'ForecasterRecursive_freq']
)
def test_save_and_load_forecaster_round_trip_skops_DateOffset(
    forecaster, index, tmp_path
):
    """
    Test that a forecaster with a generic pandas DateOffset (the `offset` of
    ForecasterEquivalentDate, and its `window_size` before fitting, or the
    frequency of the series) round-trips through the skops backend, before
    and after fitting, and predicts the same values.
    """
    file_name = str(tmp_path / 'forecaster.skops')
    save_forecaster(
        forecaster=forecaster, file_name=file_name, backend='skops', verbose=False
    )
    forecaster_loaded = load_forecaster(
        file_name=file_name, backend='skops', trusted=True, verbose=False
    )

    assert forecaster_loaded.window_size == forecaster.window_size

    rng = np.random.default_rng(12345)
    y = pd.Series(rng.normal(size=len(index)), index=index)
    forecaster.fit(y=y)
    predictions = forecaster.predict(steps=5)
    save_forecaster(
        forecaster=forecaster, file_name=file_name, backend='skops', verbose=False
    )
    forecaster_loaded = load_forecaster(
        file_name=file_name, backend='skops', trusted=True, verbose=False
    )

    pd.testing.assert_series_equal(predictions, forecaster_loaded.predict(steps=5))


def test_load_forecaster_skops_raises_when_untrusted_by_default():
    """
    Test that load_forecaster with backend='skops' and the default
    `trusted=False` raises UntrustedTypesFoundException, listing the untrusted
    types, instead of silently trusting the whole file.
    """

    forecaster = _build_fitted_forecaster_recursive()
    file_base = 'forecaster_skops_untrusted'
    save_forecaster(
        forecaster=forecaster, file_name=file_base, backend='skops', verbose=False
    )
    expected_file = file_base + '.skops'

    err_msg = re.escape(
        "skops does not load these types unless you explicitly trust them."
    )
    with pytest.raises(UntrustedTypesFoundException, match=err_msg) as excinfo:
        load_forecaster(file_name=expected_file, backend='skops', verbose=False)

    # The skops-generated part of the message lists the actual untrusted types.
    assert 'ForecasterRecursive' in str(excinfo.value)

    os.remove(expected_file)


def test_load_forecaster_skops_trusted_list():
    """
    Test that load_forecaster with backend='skops' loads the forecaster when the
    untrusted types are passed explicitly as a list to `trusted`, and that the
    loaded forecaster predicts identically.
    """

    forecaster = _build_fitted_forecaster_recursive()
    predictions = forecaster.predict(steps=5)
    file_base = 'forecaster_skops_trusted_list'
    save_forecaster(
        forecaster=forecaster, file_name=file_base, backend='skops', verbose=False
    )
    expected_file = file_base + '.skops'

    trusted = skops.io.get_untrusted_types(file=expected_file)
    forecaster_loaded = load_forecaster(
        file_name=expected_file, backend='skops', trusted=trusted, verbose=False
    )
    os.remove(expected_file)

    _assert_attribute_equal(predictions, forecaster_loaded.predict(steps=5))


def test_save_forecaster_NotImplementedError_when_skops_unsupported():
    """
    Test that backend='skops' raises NotImplementedError for forecaster types
    whose underlying estimator skops cannot serialize (e.g. ForecasterStats,
    which wraps a statsmodels model).
    """
    forecaster = _build_fitted_forecaster_stats()
    err_msg = re.escape(
        "backend='skops' is not supported for ForecasterStats."
    )
    with pytest.raises(NotImplementedError, match=err_msg):
        save_forecaster(
            forecaster=forecaster,
            file_name='forecaster',
            backend='skops',
            verbose=False,
        )
